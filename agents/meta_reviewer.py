"""
Meta Reviewer Agent
Weekly performance analysis across all agents.
Identifies degradation, suggests improvements, and writes
recommendations that agents read as context.

Runs Sunday after weekly prep.
"""

import json
import logging
from datetime import datetime, timedelta
from utils.llm import call_llm, parse_json_response
from utils.case_library import query_cases, get_win_rate_by_setup
from db.schema import AgentMemory, Trade, DailyLog, get_session
from models.pm_profiles import PM_PROFILES, ACTIVE_PROFILES

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a meta-reviewer for a multi-agent paper trading system.
Your job is to analyze the performance of each agent over the past week,
identify trends, flag degradation, and write actionable recommendations.

The system has these agents:
- Scout: finds additional symbols to watch each morning
- Researcher: summarizes news and sentiment
- Analyst: generates technical signals (LONG/SHORT/HOLD) with setup types
- Portfolio Manager (x3 profiles): makes trade decisions
- Reviewer: scores closed trades and extracts lessons
- Quant Researcher: evaluates strategy effectiveness

You review ALL of them. You are the quality control layer.

Respond in JSON:
{
  "week": "YYYY-MM-DD",
  "overall_assessment": "one paragraph summary of system health",

  "agent_reviews": {
    "scout": {
      "grade": "A|B|C|D|F",
      "trend": "improving|stable|degrading",
      "findings": "what you observed",
      "recommendations": ["specific actionable items"]
    },
    "analyst": { ... },
    "pm_conservative": { ... },
    "pm_moderate": { ... },
    "pm_aggressive": { ... },
    "reviewer": { ... },
    "quant_researcher": { ... }
  },

  "system_recommendations": [
    {
      "priority": "high|medium|low",
      "category": "signal_quality|execution|risk_management|strategy|infrastructure",
      "title": "short title",
      "description": "detailed recommendation",
      "affected_agents": ["analyst", "pm_moderate"]
    }
  ],

  "code_suggestions": [
    {
      "priority": "high|medium|low",
      "type": "refactor|feature|bugfix|optimization",
      "title": "short title",
      "description": "what to change and why",
      "files_affected": ["agents/analyst.py"]
    }
  ],

  "key_metrics": {
    "total_trades": 0,
    "win_rate": 0,
    "avg_pnl_pct": 0,
    "best_setup": "setup_type",
    "worst_setup": "setup_type",
    "best_profile": "profile_id",
    "worst_profile": "profile_id"
  }
}

Be specific and data-driven. Reference actual numbers. Don't be vague.
"""


def _gather_weekly_data(engine) -> dict:
    """Collect all performance data from the past 7 days."""
    db = get_session(engine)
    cutoff = datetime.utcnow() - timedelta(days=7)

    # Closed trades this week
    trades = (
        db.query(Trade)
        .filter_by(status="closed")
        .filter(Trade.exit_time >= cutoff)
        .all()
    )

    trade_data = []
    for t in trades:
        trade_data.append({
            "symbol": t.symbol,
            "profile": t.profile,
            "direction": t.direction,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "review_score": t.review_score,
            "stop_price": t.stop_price,
        })

    # Cases this week
    from models.case import Case
    cases = (
        db.query(Case)
        .filter(Case.created_at >= cutoff)
        .all()
    )
    case_data = []
    for c in cases:
        case_data.append({
            "symbol": c.symbol,
            "setup_type": c.setup_type,
            "outcome": c.outcome,
            "pnl_pct": c.pnl_pct,
            "selection_score": c.selection_score,
            "execution_score": c.execution_score,
            "profile": c.profile,
        })

    # Win rates by setup
    win_rates = get_win_rate_by_setup(engine)

    # Per-profile stats
    profile_stats = {}
    for pid in ACTIVE_PROFILES:
        pt = [t for t in trade_data if t["profile"] == pid]
        if pt:
            wins = sum(1 for t in pt if (t["pnl"] or 0) > 0)
            profile_stats[pid] = {
                "trades": len(pt),
                "wins": wins,
                "win_rate": round(wins / len(pt) * 100, 1),
                "avg_pnl": round(sum(t["pnl"] or 0 for t in pt) / len(pt), 2),
                "total_pnl": round(sum(t["pnl"] or 0 for t in pt), 2),
                "avg_review_score": round(
                    sum(t["review_score"] or 0 for t in pt if t["review_score"]) /
                    max(1, sum(1 for t in pt if t["review_score"])), 1
                ),
            }
        else:
            profile_stats[pid] = {"trades": 0, "wins": 0, "win_rate": 0, "avg_pnl": 0, "total_pnl": 0}

    # Daily logs
    daily_logs = (
        db.query(DailyLog)
        .filter(DailyLog.date >= cutoff.strftime("%Y-%m-%d"))
        .order_by(DailyLog.date)
        .all()
    )
    daily_data = [
        {"date": d.date, "pnl": d.daily_pnl, "trades": d.trades_taken,
         "wins": d.winning_trades, "losses": d.losing_trades}
        for d in daily_logs
    ]

    # Recent agent feedback
    feedback = {}
    for key in ["selection_feedback", "execution_feedback"]:
        mem = (
            db.query(AgentMemory)
            .filter_by(agent="reviewer", key=key)
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if mem:
            feedback[key] = mem.value

    # Previous meta review (for trend comparison)
    prev_review = (
        db.query(AgentMemory)
        .filter_by(agent="meta_reviewer", key="weekly_review")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    prev_review_data = None
    if prev_review:
        try:
            full = json.loads(prev_review.value)
            prev_review_data = {
                "week": full.get("week"),
                "key_metrics": full.get("key_metrics"),
                "agent_grades": {k: {"grade": v.get("grade"), "trend": v.get("trend")}
                                 for k, v in full.get("agent_reviews", {}).items()},
            }
        except Exception:
            pass

    db.close()

    return {
        "trades": trade_data,
        "cases": case_data,
        "win_rates_by_setup": win_rates,
        "profile_stats": profile_stats,
        "daily_logs": daily_data,
        "feedback": feedback,
        "previous_review": prev_review_data,
    }


def run(engine) -> dict:
    """Run the weekly meta review."""
    log.info("Meta Reviewer: gathering weekly data...")
    data = _gather_weekly_data(engine)

    # Dynamic strategies
    from utils.strategy_store import get_all_strategies
    all_strats = get_all_strategies(engine)
    dynamic = {k: v for k, v in all_strats.items() if v.get("source") == "dynamic"}

    user_prompt = f"""
Week ending: {datetime.utcnow().strftime('%Y-%m-%d')}

TRADES THIS WEEK ({len(data['trades'])} total):
{json.dumps(data['trades'], indent=2)}

CASES THIS WEEK ({len(data['cases'])} total):
{json.dumps(data['cases'], indent=2)}

WIN RATES BY SETUP TYPE (all time):
{json.dumps(data['win_rates_by_setup'], indent=2)}

PROFILE PERFORMANCE THIS WEEK:
{json.dumps(data['profile_stats'], indent=2)}

DAILY P&L LOG:
{json.dumps(data['daily_logs'], indent=2)}

REVIEWER FEEDBACK:
{json.dumps(data['feedback'], indent=2)}

DYNAMIC STRATEGIES (agent-proposed):
{json.dumps(dynamic, indent=2, default=str) if dynamic else 'None yet'}

PREVIOUS META REVIEW:
{json.dumps(data['previous_review'], indent=2) if data['previous_review'] else 'First review — no prior data'}

Analyze the system's performance this week. Grade each agent.
Compare against last week if previous review exists.
Identify what's working, what's degrading, and what needs to change.
Suggest specific code refactors and feature additions that would improve the system.
"""

    raw = call_llm(SYSTEM_PROMPT, user_prompt, json_mode=True, purpose="meta_reviewer")
    result = parse_json_response(raw)
    result["generated_at"] = datetime.utcnow().isoformat()

    # Store the review
    db = get_session(engine)
    db.add(AgentMemory(
        agent="meta_reviewer",
        symbol=None,
        key="weekly_review",
        value=json.dumps(result),
    ))

    # Store per-agent recommendations so they read them
    for agent_name, review in result.get("agent_reviews", {}).items():
        recs = review.get("recommendations", [])
        if recs:
            db.add(AgentMemory(
                agent="meta_reviewer",
                symbol=agent_name,
                key="agent_recommendation",
                value=json.dumps({
                    "grade": review.get("grade"),
                    "trend": review.get("trend"),
                    "recommendations": recs,
                    "week": result.get("week"),
                }),
            ))

    # Store code suggestions separately for easy retrieval
    code_suggestions = result.get("code_suggestions", [])
    if code_suggestions:
        db.add(AgentMemory(
            agent="meta_reviewer",
            symbol=None,
            key="code_suggestions",
            value=json.dumps(code_suggestions),
        ))

    db.commit()
    db.close()

    log.info(f"Meta Reviewer complete: {result.get('overall_assessment', '')[:100]}")
    return result
