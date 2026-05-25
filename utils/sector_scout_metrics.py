"""Success metrics tracking and reporting for the Sector Scout pipeline.

Computes daily metrics from AgentMemory persistence data:
- Expanded candidates surfaced per day
- % reaching Analyst LONG/SHORT
- % reaching PM eligible
- Executed trade outcomes
- Follow-through for rejected candidates (price movement after rejection)
- Top Reason_Codes for rejected and penalized candidates

Key patterns read:
- run_summary:{YYYY-MM-DD}:{run_type}     → RunSummary JSON
- expanded_watchlist:{YYYY-MM-DD}          → Today's expanded symbols + metadata
- candidate_row:{YYYY-MM-DD}:{run_type}:{symbol} → Per-symbol CandidateRow

See: design.md §7 (Observability), requirements.md §10.6, §11.1–§11.6
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import date, datetime, timezone

from db.schema import AgentMemory, Trade, get_session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_daily_metrics(engine, target_date: str | None = None) -> dict:
    """Compute sector scout success metrics for a given date.

    Args:
        engine: SQLAlchemy engine.
        target_date: Date string (YYYY-MM-DD). Defaults to today.

    Returns:
        {
            "date": str,
            "expanded_candidates_surfaced": int,
            "pct_reaching_analyst_long_short": float,
            "pct_reaching_pm_eligible": float,
            "executed_trade_outcomes": list[dict],
            "top_rejection_reason_codes": list[dict],
            "top_penalty_reason_codes": list[dict],
            "follow_through_rejected": list[dict],
        }
    """
    if target_date is None:
        target_date = date.today().isoformat()

    db = get_session(engine)
    try:
        # 1. Get expanded watchlist symbols for the date
        expanded_symbols = _get_expanded_symbols(db, target_date)
        expanded_count = len(expanded_symbols)

        # 2. Check which expanded candidates reached Analyst LONG/SHORT
        analyst_long_short = _get_analyst_long_short(db, expanded_symbols)
        pct_analyst = (
            (len(analyst_long_short) / expanded_count * 100.0)
            if expanded_count > 0
            else 0.0
        )

        # 3. Check which expanded candidates reached PM eligible
        pm_eligible = _get_pm_eligible(db, expanded_symbols)
        pct_pm = (
            (len(pm_eligible) / expanded_count * 100.0)
            if expanded_count > 0
            else 0.0
        )

        # 4. Get executed trade outcomes for expanded candidates
        executed_outcomes = _get_executed_trade_outcomes(db, expanded_symbols, target_date)

        # 5. Get top rejection and penalty reason codes from run summaries
        top_rejection_codes = _get_top_reason_codes(db, target_date, code_type="rejection")
        top_penalty_codes = _get_top_reason_codes(db, target_date, code_type="penalty")

        # 6. Follow-through for rejected candidates
        follow_through = _get_follow_through_rejected(db, target_date)

        return {
            "date": target_date,
            "expanded_candidates_surfaced": expanded_count,
            "pct_reaching_analyst_long_short": round(pct_analyst, 1),
            "pct_reaching_pm_eligible": round(pct_pm, 1),
            "executed_trade_outcomes": executed_outcomes,
            "top_rejection_reason_codes": top_rejection_codes,
            "top_penalty_reason_codes": top_penalty_codes,
            "follow_through_rejected": follow_through,
        }

    except Exception:
        logger.error("Failed to compute daily metrics for %s", target_date, exc_info=True)
        return {
            "date": target_date,
            "expanded_candidates_surfaced": 0,
            "pct_reaching_analyst_long_short": 0.0,
            "pct_reaching_pm_eligible": 0.0,
            "executed_trade_outcomes": [],
            "top_rejection_reason_codes": [],
            "top_penalty_reason_codes": [],
            "follow_through_rejected": [],
        }
    finally:
        db.close()


def format_metrics_for_review(metrics: dict) -> str:
    """Format metrics dict as a human-readable summary for daily review.

    Returns a formatted string suitable for inclusion in the daily review
    or CEO memo.
    """
    lines = []
    lines.append("=== Sector Scout Metrics ===")
    lines.append(f"Date: {metrics.get('date', 'N/A')}")
    lines.append("")

    # Funnel summary
    surfaced = metrics.get("expanded_candidates_surfaced", 0)
    lines.append(f"Expanded Candidates Surfaced: {surfaced}")
    lines.append(
        f"  → Reached Analyst LONG/SHORT: {metrics.get('pct_reaching_analyst_long_short', 0.0):.1f}%"
    )
    lines.append(
        f"  → Reached PM Eligible: {metrics.get('pct_reaching_pm_eligible', 0.0):.1f}%"
    )
    lines.append("")

    # Executed trade outcomes
    outcomes = metrics.get("executed_trade_outcomes", [])
    if outcomes:
        lines.append(f"Executed Trades from Expanded Candidates: {len(outcomes)}")
        for outcome in outcomes:
            symbol = outcome.get("symbol", "?")
            pnl_pct = outcome.get("pnl_pct")
            status = outcome.get("status", "?")
            pnl_str = f"{pnl_pct:+.2f}%" if pnl_pct is not None else "open"
            lines.append(f"  {symbol}: {status} ({pnl_str})")
    else:
        lines.append("Executed Trades from Expanded Candidates: 0")
    lines.append("")

    # Top rejection reason codes
    rejection_codes = metrics.get("top_rejection_reason_codes", [])
    if rejection_codes:
        lines.append("Top Rejection Reason Codes:")
        for entry in rejection_codes[:5]:
            lines.append(f"  {entry.get('code', '?')}: {entry.get('count', 0)}")
    lines.append("")

    # Top penalty reason codes
    penalty_codes = metrics.get("top_penalty_reason_codes", [])
    if penalty_codes:
        lines.append("Top Penalty Reason Codes:")
        for entry in penalty_codes[:5]:
            lines.append(f"  {entry.get('code', '?')}: {entry.get('count', 0)}")
    lines.append("")

    # Follow-through for rejected candidates
    follow_through = metrics.get("follow_through_rejected", [])
    if follow_through:
        lines.append("Follow-Through (Rejected Candidates):")
        for entry in follow_through[:5]:
            symbol = entry.get("symbol", "?")
            move = entry.get("subsequent_move_pct")
            move_str = f"{move:+.2f}%" if move is not None else "N/A"
            lines.append(f"  {symbol}: {move_str} after rejection")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _get_expanded_symbols(db, target_date: str) -> list[str]:
    """Get expanded watchlist symbols for a given date from AgentMemory."""
    memory_key = f"expanded_watchlist:{target_date}"
    record = (
        db.query(AgentMemory)
        .filter_by(agent="sector_scout", key=memory_key)
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    if not record:
        return []

    try:
        data = json.loads(record.value)
        return data.get("symbols", [])
    except (json.JSONDecodeError, TypeError):
        return []


def _get_analyst_long_short(db, expanded_symbols: list[str]) -> list[str]:
    """Find which expanded symbols have an Analyst LONG or SHORT signal."""
    if not expanded_symbols:
        return []

    reached = []
    for symbol in expanded_symbols:
        signal_mem = (
            db.query(AgentMemory)
            .filter_by(agent="analyst", symbol=symbol, key="signal")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if signal_mem:
            try:
                signal_data = json.loads(signal_mem.value)
                bias = signal_data.get("signal", "").upper()
                if bias in ("LONG", "SHORT"):
                    reached.append(symbol)
            except (json.JSONDecodeError, TypeError):
                pass

    return reached


def _get_pm_eligible(db, expanded_symbols: list[str]) -> list[str]:
    """Find which expanded symbols reached PM eligible status.

    A symbol is considered PM eligible if it has a trade (open or closed)
    or if the PM agent recorded an eligibility decision in AgentMemory.
    """
    if not expanded_symbols:
        return []

    eligible = []
    for symbol in expanded_symbols:
        # Check if there's a trade for this symbol (indicates PM acted on it)
        trade = (
            db.query(Trade)
            .filter(Trade.symbol == symbol)
            .order_by(Trade.entry_time.desc())
            .first()
        )
        if trade:
            eligible.append(symbol)
            continue

        # Check PM eligibility memory
        pm_mem = (
            db.query(AgentMemory)
            .filter_by(agent="pm", symbol=symbol, key="eligibility")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if pm_mem:
            try:
                data = json.loads(pm_mem.value)
                if data.get("eligible", False):
                    eligible.append(symbol)
            except (json.JSONDecodeError, TypeError):
                pass

    return eligible


def _get_executed_trade_outcomes(
    db, expanded_symbols: list[str], target_date: str
) -> list[dict]:
    """Get trade outcomes for expanded candidates."""
    if not expanded_symbols:
        return []

    outcomes = []
    for symbol in expanded_symbols:
        trades = (
            db.query(Trade)
            .filter(Trade.symbol == symbol)
            .order_by(Trade.entry_time.desc())
            .all()
        )
        for trade in trades:
            # Only include trades entered on or after the target date
            if trade.entry_time and trade.entry_time.strftime("%Y-%m-%d") >= target_date:
                outcomes.append({
                    "symbol": trade.symbol,
                    "direction": trade.direction,
                    "status": trade.status,
                    "pnl": trade.pnl,
                    "pnl_pct": trade.pnl_pct,
                    "entry_time": trade.entry_time.isoformat() if trade.entry_time else None,
                    "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
                })

    return outcomes


def _get_top_reason_codes(
    db, target_date: str, code_type: str = "rejection"
) -> list[dict]:
    """Get top reason codes from run summaries for the given date.

    Args:
        db: Database session.
        target_date: Date string (YYYY-MM-DD).
        code_type: "rejection" for hard gate codes, "penalty" for penalty codes.

    Returns:
        List of {"code": str, "count": int} sorted by count descending.
    """
    # Query all run summaries for the date
    run_types = ["premarket", "confirmation", "midday"]
    counter: Counter = Counter()

    for run_type in run_types:
        memory_key = f"run_summary:{target_date}:{run_type}"
        record = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout", key=memory_key)
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if not record:
            continue

        try:
            summary = json.loads(record.value)
        except (json.JSONDecodeError, TypeError):
            continue

        if code_type == "rejection":
            # reason_counts from run summary contains hard gate rejection codes
            reason_counts = summary.get("reason_counts", {})
            for code, count in reason_counts.items():
                counter[code] += count
        elif code_type == "penalty":
            # Extract penalty codes from candidate rows
            _accumulate_penalty_codes_from_candidates(db, target_date, run_type, counter)

    # Sort by count descending, return top entries
    return [
        {"code": code, "count": count}
        for code, count in counter.most_common(10)
    ]


def _accumulate_penalty_codes_from_candidates(
    db, target_date: str, run_type: str, counter: Counter
) -> None:
    """Accumulate penalty reason codes from persisted candidate rows."""
    # Query all candidate_row records for this date and run_type
    key_prefix = f"candidate_row:{target_date}:{run_type}:"
    records = (
        db.query(AgentMemory)
        .filter_by(agent="sector_scout")
        .filter(AgentMemory.key.like(f"{key_prefix}%"))
        .all()
    )

    for record in records:
        try:
            candidate = json.loads(record.value)
        except (json.JSONDecodeError, TypeError):
            continue

        penalties = candidate.get("penalties_applied", [])
        for penalty in penalties:
            penalty_type = penalty.get("type", "unknown_penalty")
            counter[penalty_type] += 1


def _get_follow_through_rejected(db, target_date: str) -> list[dict]:
    """Get follow-through data for rejected candidates.

    Looks at candidate rows that were rejected (hard_gate_passed=False)
    and checks if there's any subsequent price movement data available.

    Since we don't have real-time price tracking for rejected candidates,
    this returns the rejection data with the candidate's move_pct at
    rejection time as a baseline for future comparison.
    """
    run_types = ["premarket", "confirmation", "midday"]
    follow_through = []

    for run_type in run_types:
        # Look for run summary to get rejections
        memory_key = f"run_summary:{target_date}:{run_type}"
        record = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout", key=memory_key)
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if not record:
            continue

        try:
            summary = json.loads(record.value)
        except (json.JSONDecodeError, TypeError):
            continue

        # Get the expanded watchlist symbols to identify which were NOT picked
        picks = summary.get("expanded_watchlist_symbols", [])

        # Check candidate rows for finalists that weren't picked
        key_prefix = f"candidate_row:{target_date}:{run_type}:"
        candidate_records = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout")
            .filter(AgentMemory.key.like(f"{key_prefix}%"))
            .all()
        )

        for crec in candidate_records:
            try:
                candidate = json.loads(crec.value)
            except (json.JSONDecodeError, TypeError):
                continue

            symbol = candidate.get("symbol", "")
            if not symbol or symbol in picks:
                continue

            # This candidate was a finalist but not picked — track follow-through
            move_pct = candidate.get("move_pct")
            follow_through.append({
                "symbol": symbol,
                "sector": candidate.get("sector", ""),
                "scout_score": candidate.get("scout_score", 0.0),
                "move_pct_at_rejection": move_pct,
                "subsequent_move_pct": None,  # Would need EOD price data to compute
                "reason_codes": candidate.get("reason_codes", []),
            })

    return follow_through
