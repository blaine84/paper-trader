"""
CEO Agent

Acts as the executive layer for the paper-trader project: not just reporting
what happened, but diagnosing constraints and recommending the highest-leverage
moves to make the system more valuable.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from db.schema import AgentMemory, DailyLog, Trade, TradeEvent, get_session
from utils.llm import call_llm, parse_json_response
from utils.setup_aware_evaluator import SETUP_EXIT_EVENT_TYPES

logger = logging.getLogger("ceo")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATHS = [
    PROJECT_ROOT / "logs" / "orchestrator.log",
    PROJECT_ROOT / "logs" / "service.log",
]

CEO_SYSTEM_PROMPT = """You are the CEO of a multi-agent paper trading company.
Your job is to increase the system's value every week.

You are not a Slack reporter. You are an executive operator.
Be direct, specific, and opinionated. Identify the biggest constraint, propose
concrete next actions, and connect technical issues to business/product value.

Think in terms of:
- profitability and trading quality
- reliability and autonomy
- model cost and local-model viability
- productization and monetization
- engineering leverage
- decision tracking: did changes actually improve outcomes?

Return ONLY valid JSON. No markdown, no preamble.
"""

CEO_OUTPUT_SCHEMA = """
Return JSON with exactly these top-level fields:
{
  "period": "daily|weekly",
  "company_health": "short executive assessment",
  "biggest_constraint": "the single constraint most limiting company value right now",
  "top_3_priorities": [
    {"priority": "...", "why": "...", "owner": "...", "timeframe": "today|this_week|later"}
  ],
  "build_next": [
    {"task": "specific build", "impact": "why it matters", "effort": "small|medium|large"}
  ],
  "stop_doing": ["thing to stop or reduce"],
  "monetization_angle": "how this moves toward a valuable product/business",
  "risks": [
    {"risk": "...", "severity": "low|medium|high", "mitigation": "..."}
  ],
  "delegations": [
    {"agent": "researcher|analyst|pm|reviewer|engineer|product|ceo", "task": "...", "success_metric": "..."}
  ],
  "questions_for_blaine": ["only ask questions that unblock decisions"],
  "executive_summary": "3-5 sentences suitable for Slack"
}
"""


def run(engine, period: str = "daily", send_slack: bool | None = None) -> dict[str, Any]:
    """Generate and persist a CEO operating memo."""
    period = (period or "daily").lower().strip()
    if period not in {"daily", "weekly"}:
        raise ValueError(f"Unsupported CEO period: {period}")

    context = build_context(engine, period)
    prompt = build_prompt(context)

    raw = call_llm(CEO_SYSTEM_PROMPT, prompt, json_mode=True, tier=os.getenv("CEO_LLM_TIER", "high"), purpose=f"ceo:{period}")
    memo = parse_json_response(raw)
    memo = normalize_memo(memo, period)
    memo["generated_at"] = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
    memo["context_window"] = context.get("window")

    store_memo(engine, memo, period)

    if send_slack is None:
        send_slack = os.getenv("CEO_SEND_SLACK", "1").strip().lower() not in {"0", "false", "no"}
    if send_slack:
        try_send_slack(memo)

    return memo


def build_context(engine, period: str) -> dict[str, Any]:
    days = 7 if period == "weekly" else 2
    since = datetime.utcnow() - timedelta(days=days)
    today = date.today().isoformat()

    return {
        "period": period,
        "date": today,
        "window": {"days": days, "since_utc": since.isoformat() + "Z"},
        "performance": gather_performance(engine, since),
        "daily_reviews": gather_recent_memory(engine, "daily_review", "daily_review", limit=5 if period == "weekly" else 2),
        "recent_ceo_memos": gather_recent_memory(engine, "ceo", None, limit=3),
        "agent_outputs": gather_agent_outputs(engine),
        "recent_errors": gather_recent_errors(limit=80 if period == "weekly" else 40),
        "git_changes": gather_git_changes(days=days),
        "open_positions": gather_open_positions(engine),
        "setup_aware_governance": gather_setup_aware_governance_inputs(engine),
    }


def gather_performance(engine, since: datetime) -> dict[str, Any]:
    db = get_session(engine)
    try:
        trades = db.query(Trade).filter(Trade.entry_time >= since).all()
        closed = [t for t in trades if t.status == "closed"]
        wins = [t for t in closed if (t.pnl or 0) > 0]
        losses = [t for t in closed if (t.pnl or 0) <= 0]
        pnl = sum(t.pnl or 0 for t in closed)
        daily_logs = db.query(DailyLog).filter(DailyLog.date >= since.date().isoformat()).order_by(DailyLog.date.desc()).all()
        return {
            "trades_opened": len(trades),
            "trades_closed": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0,
            "closed_pnl": round(pnl, 2),
            "daily_logs": [
                {
                    "date": d.date,
                    "daily_pnl": d.daily_pnl,
                    "daily_pnl_pct": d.daily_pnl_pct,
                    "trades_taken": d.trades_taken,
                    "wins": d.winning_trades,
                    "losses": d.losing_trades,
                    "notes": d.notes,
                }
                for d in daily_logs[:7]
            ],
        }
    finally:
        db.close()


def gather_recent_memory(engine, agent: str, key: str | None, limit: int = 5) -> list[dict[str, Any]]:
    db = get_session(engine)
    try:
        q = db.query(AgentMemory).filter_by(agent=agent)
        if key is not None:
            q = q.filter_by(key=key)
        rows = q.order_by(AgentMemory.timestamp.desc()).limit(limit).all()
        out = []
        for row in rows:
            value: Any = row.value
            try:
                value = json.loads(row.value)
            except Exception:
                if isinstance(value, str) and len(value) > 2000:
                    value = value[:2000] + "..."
            out.append({
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                "agent": row.agent,
                "symbol": row.symbol,
                "key": row.key,
                "value": value,
            })
        return out
    finally:
        db.close()


def gather_agent_outputs(engine) -> dict[str, Any]:
    return {
        "quant_researcher": gather_recent_memory(engine, "quant_researcher", "strategy_recommendations", 1),
        "reviewer_selection": gather_recent_memory(engine, "reviewer", "selection_feedback", 1),
        "reviewer_execution": gather_recent_memory(engine, "reviewer", "execution_feedback", 1),
        "position_health": gather_recent_memory(engine, "position_health", "health_check", 1),
        "news_monitor": gather_recent_memory(engine, "news_monitor", None, 2),
        "price_monitor_alerts": gather_recent_memory(engine, "price_monitor", "live_alerts", 1),
        "lifecycle_exceptions": gather_recent_memory(engine, "position_timer", "eod_exposure_exception", 2),
    }


def gather_open_positions(engine) -> list[dict[str, Any]]:
    from db.schema import Position

    db = get_session(engine)
    try:
        return [
            {
                "profile": p.profile,
                "symbol": p.symbol,
                "side": p.side,
                "quantity": p.quantity,
                "avg_cost": p.avg_cost,
                "opened_at": p.opened_at.isoformat() if p.opened_at else None,
            }
            for p in db.query(Position).all()
        ]
    finally:
        db.close()


def gather_setup_aware_governance_inputs(engine) -> dict[str, Any]:
    """Gather setup-aware exit governance data for CEO memo context.

    Queries trade_events for setup-aware event types from the last 24 hours
    and collects counts and up to 3 representative examples per category.
    """
    db = get_session(engine)
    try:
        since = datetime.utcnow() - timedelta(hours=24)
        events = (
            db.query(TradeEvent)
            .filter(
                TradeEvent.event_type.in_(SETUP_EXIT_EVENT_TYPES),
                TradeEvent.timestamp >= since,
            )
            .order_by(TradeEvent.timestamp.desc())
            .all()
        )

        # Parse all events into dicts with payload
        parsed_events = []
        for ev in events:
            payload = {}
            if ev.payload_json:
                try:
                    payload = json.loads(ev.payload_json)
                except Exception:
                    pass
            parsed_events.append({
                "trade_id": ev.trade_id,
                "event_type": ev.event_type,
                "symbol": ev.symbol,
                "timestamp": ev.timestamp.isoformat() if ev.timestamp else None,
                "payload": payload,
            })

        # Category 1: Forced exits by setup type
        forced_exits = [e for e in parsed_events if e["event_type"] == "setup_exit_force_close"]
        forced_by_setup: dict[str, int] = {}
        for e in forced_exits:
            st = e["payload"].get("setup_type", "unknown")
            forced_by_setup[st] = forced_by_setup.get(st, 0) + 1
        forced_examples = [
            {
                "trade_id": e["trade_id"],
                "symbol": e["symbol"],
                "setup_type": e["payload"].get("setup_type", "unknown"),
                "minutes_held": e["payload"].get("minutes_held"),
            }
            for e in forced_exits[:3]
        ]

        # Category 2: Revalidated holds by setup type
        revalidated_holds = [e for e in parsed_events if e["event_type"] == "setup_exit_revalidated_hold"]
        revalidated_by_setup: dict[str, int] = {}
        for e in revalidated_holds:
            st = e["payload"].get("setup_type", "unknown")
            revalidated_by_setup[st] = revalidated_by_setup.get(st, 0) + 1
        revalidated_examples = [
            {
                "trade_id": e["trade_id"],
                "symbol": e["symbol"],
                "setup_type": e["payload"].get("setup_type", "unknown"),
                "minutes_held": e["payload"].get("minutes_held"),
            }
            for e in revalidated_holds[:3]
        ]

        # Category 3: News-breakout trades closed by timer vs invalidation
        news_breakout_force = [
            e for e in forced_exits
            if e["payload"].get("setup_type") in ("news_breakout", "news_catalyst")
        ]
        news_breakout_invalidated = [
            e for e in parsed_events
            if e["event_type"] == "setup_exit_thesis_invalidated"
            and e["payload"].get("setup_type") in ("news_breakout", "news_catalyst")
        ]
        news_examples = [
            {
                "trade_id": e["trade_id"],
                "symbol": e["symbol"],
                "setup_type": e["payload"].get("setup_type", "unknown"),
                "close_type": "timer" if e["event_type"] == "setup_exit_force_close" else "thesis_invalidated",
                "minutes_held": e["payload"].get("minutes_held"),
            }
            for e in (news_breakout_force + news_breakout_invalidated)[:3]
        ]

        # Category 4: Missing exit metadata prevented extension
        missing_metadata = [
            e for e in parsed_events
            if e["event_type"] == "setup_exit_revalidation_failed"
            and "missing criteria" in (e["payload"].get("reason", "") or "").lower()
        ]
        missing_metadata_examples = [
            {
                "trade_id": e["trade_id"],
                "symbol": e["symbol"],
                "setup_type": e["payload"].get("setup_type", "unknown"),
                "reason": e["payload"].get("reason", ""),
            }
            for e in missing_metadata[:3]
        ]

        return {
            "forced_exits": {
                "count": len(forced_exits),
                "by_setup_type": forced_by_setup,
                "examples": forced_examples,
            },
            "revalidated_holds": {
                "count": len(revalidated_holds),
                "by_setup_type": revalidated_by_setup,
                "examples": revalidated_examples,
            },
            "news_breakout_exits": {
                "timer_closes": len(news_breakout_force),
                "thesis_invalidation_closes": len(news_breakout_invalidated),
                "examples": news_examples,
            },
            "missing_metadata_denials": {
                "count": len(missing_metadata),
                "examples": missing_metadata_examples,
            },
        }
    finally:
        db.close()


def gather_recent_errors(limit: int = 60) -> list[str]:
    patterns = ("error", "exception", "traceback", "failed", "critical", "timeout", "warning")
    lines: list[str] = []
    for path in LOG_PATHS:
        if not path.exists():
            continue
        try:
            recent = path.read_text(errors="replace").splitlines()[-3000:]
        except Exception as exc:
            lines.append(f"{path.name}: failed to read log: {exc}")
            continue
        for line in recent:
            low = line.lower()
            if any(p in low for p in patterns):
                lines.append(f"{path.name}: {line}")
    return lines[-limit:]


def gather_git_changes(days: int) -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            ["git", "log", f"--since={days} days ago", "--pretty=format:%h|%aI|%s", "--max-count=20"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return []
        changes = []
        for line in result.stdout.splitlines():
            parts = line.split("|", 2)
            if len(parts) == 3:
                changes.append({"hash": parts[0], "timestamp": parts[1], "message": parts[2]})
        return changes
    except Exception as exc:
        return [{"hash": "", "timestamp": "", "message": f"git log failed: {exc}"}]


def build_prompt(context: dict[str, Any]) -> str:
    return f"""
{CEO_OUTPUT_SCHEMA}

OPERATING CONTEXT:
{json.dumps(context, indent=2, default=str)}

Give Blaine the CEO memo. Be concrete. Prefer a small number of high-leverage actions over a broad wish list.
"""


def normalize_memo(memo: dict[str, Any], period: str) -> dict[str, Any]:
    defaults = {
        "period": period,
        "company_health": "unknown",
        "biggest_constraint": "unknown",
        "top_3_priorities": [],
        "build_next": [],
        "stop_doing": [],
        "monetization_angle": "",
        "risks": [],
        "delegations": [],
        "questions_for_blaine": [],
        "executive_summary": "",
    }
    if not isinstance(memo, dict):
        memo = {}
    normalized = {**defaults, **memo}
    normalized["period"] = period
    for field in ["top_3_priorities", "build_next", "stop_doing", "risks", "delegations", "questions_for_blaine"]:
        if not isinstance(normalized.get(field), list):
            normalized[field] = []
    for field in ["company_health", "biggest_constraint", "monetization_angle", "executive_summary"]:
        if not isinstance(normalized.get(field), str):
            normalized[field] = str(normalized.get(field, ""))
    return normalized


def store_memo(engine, memo: dict[str, Any], period: str) -> None:
    db = get_session(engine)
    try:
        db.add(AgentMemory(
            agent="ceo",
            symbol=date.today().isoformat(),
            key=f"{period}_memo",
            value=json.dumps(memo, default=str),
        ))
        db.commit()
        logger.info("CEO %s memo stored", period)
    finally:
        db.close()


def try_send_slack(memo: dict[str, Any]) -> None:
    try:
        from utils.slack_notifier import SlackNotifier

        notifier = SlackNotifier()
        if not notifier.is_enabled():
            return

        title = "CEO Weekly Strategy Memo" if memo.get("period") == "weekly" else "CEO Daily Operating Memo"
        blocks = format_slack_blocks(title, memo)
        result = notifier.send_blocks(blocks, f"{title}: {memo.get('biggest_constraint', '')}")
        if not result.get("ok"):
            logger.warning("CEO Slack delivery failed: %s", result.get("error"))
    except Exception as exc:
        logger.warning("CEO Slack delivery error: %s", exc)


def format_slack_blocks(title: str, memo: dict[str, Any]) -> list[dict[str, Any]]:
    def section(text: str) -> dict[str, Any]:
        return {"type": "section", "text": {"type": "mrkdwn", "text": text[:2900]}}

    priorities = memo.get("top_3_priorities") or []
    priority_lines = []
    for i, item in enumerate(priorities[:3], 1):
        if isinstance(item, dict):
            priority_lines.append(f"{i}. *{item.get('priority', 'Priority')}* — {item.get('why', '')}")
        else:
            priority_lines.append(f"{i}. {item}")

    builds = memo.get("build_next") or []
    build_lines = []
    for item in builds[:3]:
        if isinstance(item, dict):
            build_lines.append(f"• {item.get('task', '')} ({item.get('effort', 'unknown')} effort)")
        else:
            build_lines.append(f"• {item}")

    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": title[:150]}},
        section(f"*Health:* {memo.get('company_health', '')}\n*Biggest constraint:* {memo.get('biggest_constraint', '')}"),
    ]
    if memo.get("executive_summary"):
        blocks.append(section(f"*Summary:* {memo['executive_summary']}"))
    if priority_lines:
        blocks.append(section("*Top priorities:*\n" + "\n".join(priority_lines)))
    if build_lines:
        blocks.append(section("*Build next:*\n" + "\n".join(build_lines)))
    if memo.get("monetization_angle"):
        blocks.append(section(f"*Monetization angle:* {memo['monetization_angle']}"))
    return blocks[:10]


def print_memo(memo: dict[str, Any]) -> None:
    print(json.dumps(memo, indent=2, default=str))


if __name__ == "__main__":
    from db.schema import init_db

    import sys

    period = sys.argv[1] if len(sys.argv) > 1 else "daily"
    engine = init_db(str(PROJECT_ROOT / "db" / "paper_trader.db"))
    print_memo(run(engine, period=period))
