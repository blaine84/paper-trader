"""
Reviewer Agent
Reviews closed trades, extracts structured cases for the case library.
No prose summaries — only typed, queryable records.
"""

import json
import logging
from datetime import datetime
from utils.finnhub_client import FinnhubClient
from utils.llm import call_llm, parse_json_response
from utils.case_library import store_case, get_win_rate_by_setup, format_cases_for_prompt
from db.schema import Trade, AgentMemory, get_session
from feedback_loop.analyst_feedback import queue_reviewer_flags
from utils.trade_events import log_trade_event
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


SYSTEM_PROMPT = """You are a trading case analyst. Your job is to extract structured,
reusable lessons from closed trades — NOT prose summaries.

CRITICAL: Score selection and execution SEPARATELY. They measure different things
and feed different feedback loops.

SELECTION SCORE (1-10) — feeds Scout + Analyst:
  "Was the setup read correctly?"
  - Did the Analyst correctly identify direction, setup type, key levels?
  - Did the invalidation condition hold or break as expected?
  - Was the setup quality (strength/confidence) accurate in hindsight?
  - Did the catalyst/regime read match what actually happened?
  Score independently of how PM traded it — a great read on a poorly executed trade still scores high here.

EXECUTION SCORE (1-10) — feeds PM (per profile):
  "Given the signal, did PM make good execution decisions?"
  - Did PM enter at a sensible level (key level, VWAP, breakout) or did it chase?
  - Was the stop placed at the invalidation level or arbitrary?
  - Was position size appropriate for the profile's rules?
  - Was the exit disciplined — at target, at stop, or did PM override?
  - Did PM correctly pass on weak/low-confidence signals?
  Score independently of whether the setup was a good one — clean execution on a losing trade can still score well.

REVIEW SCORE = average of the two (for overall tracking).

For each trade, return a case object:

{
  "trade_id": 42,
  "symbol": "TSLA",
  "date": "2026-03-22",
  "profile": "aggressive",

  "setup_type": one of: news_breakout | gap_and_go | technical_breakout | momentum_fade |
                gap_fill | range_breakout | vwap_reclaim | earnings_reaction |
                sector_rotation | reversal,

  "catalyst_type": one of: analyst_upgrade | analyst_downgrade | earnings_beat |
                   earnings_miss | macro_event | product_launch | sector_move |
                   short_squeeze | technical_only | news_headline | regulatory,

  "float_profile": one of: micro_cap | small_cap | mid_cap | large_cap | mega_cap,
  "sector": e.g. tech | energy | financials | healthcare | consumer | industrials | etf,

  "premarket_gap_pct": float (positive = gap up, negative = gap down),
  "premarket_volume_rank": one of: low | medium | high | extreme,
  "market_regime": one of: risk_on | risk_off | mixed,

  "entry_timing": one of: open | first_15min | first_30min | mid_day | power_hour | close,
  "bias": one of: LONG | SHORT,
  "signal_strength": one of: weak | moderate | strong,
  "signal_confidence": one of: low | medium | high,
  "invalidation": "the condition that would invalidate the setup",

  "rsi_at_entry": float or null,
  "above_vwap": "true" or "false",
  "above_daily_resistance": "true" or "false",
  "ema_trend": one of: bullish | bearish | neutral,
  "bb_position": one of: upper | middle | lower | outside_upper | outside_lower,
  "entry_vs_level": how PM entered relative to key levels e.g. at_support | above_vwap | at_breakout | chased | early,

  "outcome": one of: success | failure | partial,
  "pnl_pct": float,
  "holding_minutes": integer,

  "lesson": "one actionable sentence — the single most important thing this trade teaches",

  "conditions_for_success": ["field=value", ...],
  "conditions_to_avoid": ["field=value", ...],

  "confidence": one of: low | medium | high,

  "selection_score": float 1-10,
  "execution_score": float 1-10,
  "review_score": float 1-10  // average of the two
}

Return JSON:
{
  "cases": [ ... ],
  "selection_feedback": "feedback for Scout + Analyst — what setups/catalysts/regimes are being read correctly or not",
  "execution_feedback": {
    "conservative": "feedback specific to conservative PM execution",
    "moderate": "feedback specific to moderate PM execution",
    "aggressive": "feedback specific to aggressive PM execution"
  }
}

Be precise. Infer fields from context. Never leave outcome, selection_score, or execution_score null.
"""


def run(engine, min_unreviewed: int = 1) -> dict:
    db = get_session(engine)
    fh = FinnhubClient()

    # Pull from review queue first, fall back to scanning unreviewed trades
    from db.schema import ReviewQueue
    queued = (
        db.query(ReviewQueue)
        .filter_by(status="pending")
        .order_by(ReviewQueue.queued_at)
        .limit(3)
        .all()
    )

    if queued:
        trade_ids = [q.trade_id for q in queued]
        unreviewed = db.query(Trade).filter(Trade.id.in_(trade_ids)).all()
    else:
        # Fallback: scan for unreviewed closed trades not in queue
        unreviewed = (
            db.query(Trade)
            .filter_by(status="closed")
            .filter(Trade.review_score == None)
            .order_by(Trade.exit_time.desc())
            .limit(3)
            .all()
        )

    if len(unreviewed) < min_unreviewed:
        db.close()
        return {"message": f"Only {len(unreviewed)} unreviewed trades, skipping."}

    # Pull a few recent cases as examples (show the model what good output looks like)
    from utils.case_library import query_cases
    recent_cases = query_cases(engine, limit=3)
    example_text = format_cases_for_prompt(recent_cases) if recent_cases else "No prior cases yet — establish the library."

    # Get market regime context
    regime_mem = (
        db.query(AgentMemory)
        .filter_by(agent="researcher", key="market_context")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    regime_context = regime_mem.value if regime_mem else "unknown"

    # Format trades for review
    trade_data = []
    for t in unreviewed:
        # Compute holding time
        holding_minutes = None
        if t.entry_time and t.exit_time:
            holding_minutes = int((t.exit_time - t.entry_time).total_seconds() / 60)

        trade_data.append({
            "trade_id": t.id,
            "symbol": t.symbol,
            "profile": t.profile,
            "direction": t.direction,
            "quantity": t.quantity,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "entry_time": t.entry_time.isoformat() if t.entry_time else None,
            "exit_time": t.exit_time.isoformat() if t.exit_time else None,
            "holding_minutes": holding_minutes,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "reason_entry": t.reason_entry,
            "reason_exit": t.reason_exit,
        })

    user_prompt = f"""
Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
Market context today: {regime_context}

TRADES TO REVIEW:
{json.dumps(trade_data, indent=2)}

RECENT CASE LIBRARY EXAMPLES (for reference):
{example_text}

Extract structured cases for each trade. Return the JSON.
"""

    raw = call_llm(SYSTEM_PROMPT, user_prompt, json_mode=True)
    result = parse_json_response(raw)

    # Store cases in the case library
    reviewed_trade_ids = [t.id for t in unreviewed]  # capture before session closes
    for case_data in result.get("cases", []):
        trade = db.query(Trade).filter_by(id=case_data.get("trade_id")).first()
        if trade:
            trade.review_score = case_data.get("review_score")
            trade.review_notes = case_data.get("lesson")
            log_trade_event(
                db, "review_completed", trade_id=trade.id, agent="reviewer",
                symbol=trade.symbol, profile=trade.profile, price=trade.exit_price,
                message=case_data.get("lesson"),
                payload={
                    "review_score": case_data.get("review_score"),
                    "selection_score": case_data.get("selection_score"),
                    "execution_score": case_data.get("execution_score"),
                    "setup_type": case_data.get("setup_type"),
                    "outcome": case_data.get("outcome"),
                },
            )
    db.commit()
    db.close()

    for case_data in result.get("cases", []):
        store_case(engine, case_data)

    try:
        queue_reviewer_flags(engine, result.get("cases", []))
    except Exception as exc:
        log = logging.getLogger(__name__)
        log.warning("Analyst feedback queueing failed: %s", exc)

    # Route feedback — reopen session
    db = get_session(engine)

    # Mark queue entries as reviewed
    from db.schema import ReviewQueue
    for q in db.query(ReviewQueue).filter(ReviewQueue.trade_id.in_(reviewed_trade_ids)).all():
        q.status = "reviewed"
        q.reviewed_at = datetime.utcnow()

    selection_fb = result.get("selection_feedback", "")
    if selection_fb:
        db.add(AgentMemory(
            agent="reviewer",
            symbol=None,
            key="selection_feedback",
            value=selection_fb,
        ))

    # Route execution feedback → each PM profile separately
    execution_fb = result.get("execution_feedback", {})
    if isinstance(execution_fb, dict):
        for profile_id, fb_text in execution_fb.items():
            if fb_text:
                db.add(AgentMemory(
                    agent="reviewer",
                    symbol=None,
                    key=f"execution_feedback_{profile_id}",
                    value=fb_text,
                ))
    elif isinstance(execution_fb, str) and execution_fb:
        # fallback if LLM returned a string instead of dict
        db.add(AgentMemory(
            agent="reviewer",
            symbol=None,
            key="execution_feedback",
            value=execution_fb,
        ))

    db.commit()
    db.close()

    # Extract behavioral parameters from execution feedback
    from utils.behavioral_params import extract_params_from_feedback
    if isinstance(execution_fb, dict):
        for profile_id, fb_text in execution_fb.items():
            if fb_text:
                try:
                    extract_params_from_feedback(engine, profile_id, fb_text)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Behavioral param extraction failed for {profile_id}: {e}")

    # Print EOD scorecard
    _print_scorecard(result)

    # Score Scout picks separately
    _review_scout_picks(engine)

    return result


def _print_scorecard(result: dict):
    """Print a rich EOD scorecard to the terminal."""
    cases = result.get("cases", [])
    if not cases:
        return

    console.print()
    console.print(Panel(
        f"[bold white]EOD Review — {datetime.utcnow().strftime('%Y-%m-%d')}[/bold white]",
        style="bold magenta",
        box=box.DOUBLE,
    ))

    # Per-trade scorecard
    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold cyan",
        title="Trade Scores",
    )
    table.add_column("Symbol", style="bold")
    table.add_column("Profile", style="dim")
    table.add_column("Setup")
    table.add_column("Outcome")
    table.add_column("P&L %")
    table.add_column("Selection", justify="right")
    table.add_column("Execution", justify="right")
    table.add_column("Overall", justify="right")

    for c in cases:
        outcome = c.get("outcome", "?")
        outcome_color = {"success": "green", "failure": "red", "partial": "yellow"}.get(outcome, "white")
        pnl = c.get("pnl_pct")
        pnl_str = f"{pnl:+.2f}%" if pnl is not None else "?"
        pnl_color = "green" if (pnl or 0) >= 0 else "red"

        sel = c.get("selection_score")
        exe = c.get("execution_score")
        rev = c.get("review_score")

        def score_str(s):
            if s is None:
                return "[dim]—[/dim]"
            color = "green" if s >= 7 else "yellow" if s >= 5 else "red"
            return f"[{color}]{s:.1f}[/{color}]"

        table.add_row(
            c.get("symbol", "?"),
            c.get("profile", "?"),
            c.get("setup_type", "?") or "?",
            f"[{outcome_color}]{outcome}[/{outcome_color}]",
            f"[{pnl_color}]{pnl_str}[/{pnl_color}]",
            score_str(sel),
            score_str(exe),
            score_str(rev),
        )

    console.print(table)

    # Per-trade lesson panels
    for c in cases:
        sym = c.get("symbol", "?")
        lesson = c.get("lesson", "")
        sel = c.get("selection_score")
        exe = c.get("execution_score")
        works = c.get("conditions_for_success", [])
        avoid = c.get("conditions_to_avoid", [])
        invalidation = c.get("invalidation", "")

        lines = []
        if lesson:
            lines.append(f"[bold]Lesson:[/bold] {lesson}")
        if invalidation:
            lines.append(f"[bold]Invalidation:[/bold] {invalidation}")
        if works:
            lines.append(f"[bold]Works when:[/bold] {', '.join(works)}")
        if avoid:
            lines.append(f"[bold]Avoid when:[/bold] {', '.join(avoid)}")

        sel_color = "green" if (sel or 0) >= 7 else "yellow" if (sel or 0) >= 5 else "red"
        exe_color = "green" if (exe or 0) >= 7 else "yellow" if (exe or 0) >= 5 else "red"
        title = (
            f"[bold]{sym}[/bold]  "
            f"selection [{sel_color}]{sel or '?'}[/{sel_color}]  "
            f"execution [{exe_color}]{exe or '?'}[/{exe_color}]"
        )

        if lines:
            console.print(Panel("\n".join(lines), title=title, box=box.ROUNDED, style="dim"))

    # Feedback summaries
    sel_fb = result.get("selection_feedback", "")
    if sel_fb:
        console.print(Panel(
            sel_fb,
            title="[bold yellow]📊 Selection Feedback → Scout + Analyst[/bold yellow]",
            box=box.ROUNDED,
        ))

    exe_fb = result.get("execution_feedback", {})
    if isinstance(exe_fb, dict):
        for profile_id, fb in exe_fb.items():
            if fb:
                emoji = {"conservative": "🛡️", "moderate": "⚖️", "aggressive": "🔥"}.get(profile_id, "")
                console.print(Panel(
                    fb,
                    title=f"[bold green]⚙️ Execution Feedback → {emoji} {profile_id.capitalize()} PM[/bold green]",
                    box=box.ROUNDED,
                ))
    elif isinstance(exe_fb, str) and exe_fb:
        console.print(Panel(exe_fb, title="[bold green]⚙️ Execution Feedback[/bold green]", box=box.ROUNDED))

    console.print()


def _review_scout_picks(engine):
    """
    Score today's Scout picks as cases.
    Stores a case per pick even if no trade was taken — builds a screening record.
    """
    db = get_session(engine)
    fh = FinnhubClient()
    today = datetime.utcnow().strftime("%Y-%m-%d")

    picks_mem = (
        db.query(AgentMemory)
        .filter_by(agent="scout", key="daily_picks")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    if not picks_mem:
        db.close()
        return

    data = json.loads(picks_mem.value)
    if data.get("date") != today or not data.get("picks"):
        db.close()
        return

    pick_results = []
    for pick in data["picks"]:
        sym = pick["symbol"]
        try:
            quote = fh.get_quote(sym)
            pick_results.append({
                "symbol": sym,
                "catalyst": pick.get("catalyst"),
                "direction_bias": pick.get("direction_bias"),
                "conviction": pick.get("conviction"),
                "eod_change_pct": quote.get("change_pct"),
            })
        except Exception:
            continue

    if not pick_results:
        db.close()
        return

    system = """You are scoring a stock scout's daily picks.
For each pick, extract a structured case (same schema as trade cases).
These are SCREENING records — no trade was necessarily taken.
outcome = success if the pick moved in the predicted direction by >1%, failure otherwise.

Return JSON:
{
  "cases": [ ... same case schema ... ],
  "scout_feedback": "one paragraph referencing specific case fields"
}"""

    user = f"""
Today: {today}
Market context: {data.get('market_tone', 'unknown')}

Scout picks and EOD results:
{json.dumps(pick_results, indent=2)}

Extract structured cases.
"""

    try:
        raw = call_llm(system, user, json_mode=True)
        result = parse_json_response(raw)

        for case_data in result.get("cases", []):
            # Tag as scout screening record
            case_data["profile"] = "scout"
            store_case(engine, case_data)

        feedback = result.get("scout_feedback", "")
        if feedback:
            mem = AgentMemory(
                agent="reviewer",
                symbol=None,
                key="scout_feedback",
                value=feedback,
            )
            db.add(mem)
            db.commit()

    except Exception:
        pass

    db.close()
