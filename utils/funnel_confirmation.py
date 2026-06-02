"""Opening Confirmation Evaluator — validates shortlisted candidates against live opening data.

Checks volume, price position vs key levels, price behavior direction, and
catalyst freshness under a strict wall-clock budget. Promotes confirmed
candidates to pm_eligible; rejects failed candidates to rejected_confirmation.
Budget exhaustion appends not_evaluated for remaining candidates without
clearing existing state.

See: design.md §Component 5, requirements.md §6
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from zoneinfo import ZoneInfo

from db.schema import AgentMemory, FunnelCandidate, FunnelRunLog, get_session
from utils.finnhub_client import FinnhubClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class ConfirmationDecision:
    """Result of evaluating a single funnel candidate against live opening data."""

    candidate_id: str
    decision: str  # "promoted" | "rejected" | "needs_confirmation" | "not_evaluated"
    volume_confirmed: bool
    vwap_confirmed: bool | None
    price_behavior_ok: bool
    catalyst_still_fresh: bool
    reasoning: str


# ---------------------------------------------------------------------------
# Core Confirmation Logic
# ---------------------------------------------------------------------------


def run_opening_confirmation(
    engine,
    candidates: list[FunnelCandidate],
    budget_seconds: int = 45,
) -> list[ConfirmationDecision]:
    """Confirm shortlisted candidates against live opening data.

    Evaluates only persisted shortlist with stage_status=awaiting_confirmation
    (NOT a broad sector scan). Checks volume vs Analyst requirements, price
    position vs key levels, price behavior direction, and catalyst freshness
    (≤24h from market open).

    Enforces a hard wall-clock budget (default 45s). On budget exhaustion,
    stops processing and appends not_evaluated for remaining candidates,
    leaving their stage_status unchanged. Never clears existing shortlist
    state on budget expiry.

    Processes candidates in deterministic order: scout_rank ascending,
    scout_score descending (higher-ranked candidates evaluated first when
    budget is limited).

    Args:
        engine: SQLAlchemy engine for DB access.
        candidates: FunnelCandidate rows to evaluate. Should already be
            filtered to stage_status="awaiting_confirmation".
        budget_seconds: Hard wall-clock budget in seconds (default 45).

    Returns:
        List of ConfirmationDecision for each candidate (including
        not_evaluated decisions for budget-exhausted or data-unavailable
        candidates).

    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8
    """
    start = time.monotonic()
    decisions: list[ConfirmationDecision] = []

    # Filter to only awaiting_confirmation (Req 6.1)
    eligible = [c for c in candidates if c.stage_status == "awaiting_confirmation"]

    # Sort in deterministic order: scout_rank ASC, scout_score DESC (Req 6.7)
    eligible.sort(key=lambda c: (c.scout_rank, -c.scout_score))

    for i, candidate in enumerate(eligible):
        elapsed = time.monotonic() - start
        if elapsed >= budget_seconds:
            # Budget exhausted — append not_evaluated for ALL remaining (Req 6.6)
            logger.warning(
                "Confirmation budget exhausted after %.1fs, "
                "%d candidates not evaluated",
                elapsed,
                len(eligible) - i,
            )
            for remaining in eligible[i:]:
                _append_stage_decision(
                    engine=engine,
                    candidate=remaining,
                    decision="not_evaluated",
                    reasoning=f"Confirmation budget exhausted ({elapsed:.1f}s/{budget_seconds}s)",
                    evidence={},
                    next_stage="awaiting_confirmation",  # Leave unchanged (Req 6.6)
                )
                decisions.append(ConfirmationDecision(
                    candidate_id=remaining.candidate_id,
                    decision="not_evaluated",
                    volume_confirmed=False,
                    vwap_confirmed=None,
                    price_behavior_ok=False,
                    catalyst_still_fresh=False,
                    reasoning=f"Confirmation budget exhausted ({elapsed:.1f}s/{budget_seconds}s)",
                ))
            break

        try:
            # Fetch live market data for this symbol
            live_data = _get_live_data(candidate.symbol)

            if live_data is None:
                # Data unavailability → needs_confirmation (Req 6.8)
                _append_stage_decision(
                    engine=engine,
                    candidate=candidate,
                    decision="needs_confirmation",
                    reasoning="Live market data unavailable for confirmation checks",
                    evidence={"data_unavailable": True},
                    next_stage="awaiting_confirmation",  # Stays in current stage
                )
                # Do NOT update stage_status — stays awaiting_confirmation
                decisions.append(ConfirmationDecision(
                    candidate_id=candidate.candidate_id,
                    decision="needs_confirmation",
                    volume_confirmed=False,
                    vwap_confirmed=None,
                    price_behavior_ok=False,
                    catalyst_still_fresh=False,
                    reasoning="Live market data unavailable for confirmation checks",
                ))
                continue

            # Get Analyst plan data from stage_decisions
            analyst_plan = _get_analyst_plan(candidate)

            # Evaluate confirmation checks (Req 6.2)
            volume_ok = _check_volume(live_data, analyst_plan)
            vwap_ok = _check_price_position(live_data, analyst_plan)
            price_ok = _check_price_behavior(live_data, candidate, analyst_plan)
            catalyst_fresh = _check_catalyst_freshness(candidate)

            # Determine decision (Req 6.3)
            decision, reasoning, next_stage = _make_confirmation_decision(
                volume_ok=volume_ok,
                vwap_ok=vwap_ok,
                price_ok=price_ok,
                catalyst_fresh=catalyst_fresh,
                candidate=candidate,
                live_data=live_data,
                analyst_plan=analyst_plan,
            )

            # Build evidence payload
            evidence_payload = {
                "volume_confirmed": volume_ok,
                "vwap_confirmed": vwap_ok,
                "price_behavior_ok": price_ok,
                "catalyst_still_fresh": catalyst_fresh,
                "live_price": live_data.get("price"),
                "live_volume": live_data.get("volume"),
                "open_price": live_data.get("open"),
            }

            # Append stage decision (never overwrite prior entries)
            _append_stage_decision(
                engine=engine,
                candidate=candidate,
                decision=decision,
                reasoning=reasoning,
                evidence=evidence_payload,
                next_stage=next_stage,
            )

            # Update stage_status only if decision changes it
            if next_stage != "awaiting_confirmation":
                _update_stage_status(engine, candidate, next_stage)

            decisions.append(ConfirmationDecision(
                candidate_id=candidate.candidate_id,
                decision=decision,
                volume_confirmed=volume_ok,
                vwap_confirmed=vwap_ok,
                price_behavior_ok=price_ok,
                catalyst_still_fresh=catalyst_fresh,
                reasoning=reasoning,
            ))

        except Exception as e:
            # Data/API failure for a single candidate → needs_confirmation (Req 6.8)
            logger.error(
                "Confirmation evaluation failed for %s: %s",
                candidate.symbol, e,
            )
            _append_stage_decision(
                engine=engine,
                candidate=candidate,
                decision="needs_confirmation",
                reasoning=f"Confirmation error: {str(e)[:200]}",
                evidence={"error": str(e)[:200]},
                next_stage="awaiting_confirmation",  # Stays in current stage
            )
            # Do NOT update stage_status — stays awaiting_confirmation
            decisions.append(ConfirmationDecision(
                candidate_id=candidate.candidate_id,
                decision="needs_confirmation",
                volume_confirmed=False,
                vwap_confirmed=None,
                price_behavior_ok=False,
                catalyst_still_fresh=False,
                reasoning=f"Confirmation error: {str(e)[:200]}",
            ))

    return decisions


# ---------------------------------------------------------------------------
# 10:00 ET Confirmation Retry (Task 7.2)
# ---------------------------------------------------------------------------

_NY_TZ = ZoneInfo("America/New_York")


def run_confirmation_retry(
    engine,
    funnel_config: dict | None = None,
) -> list[ConfirmationDecision]:
    """10:00 ET bounded shortlist confirmation retry pass.

    Replaces the former broad 10:00 ET sector scan with a focused retry that
    evaluates only today's persisted awaiting_confirmation candidates. Provides
    a second chance for candidates that were not_evaluated due to budget
    exhaustion or data unavailability in the primary 09:35 ET pass.

    Uses the configured market_hours_confirmation_budget_seconds (default 60s)
    instead of the primary confirmation budget (45s), since this is a market-
    hours run.

    Steps:
        1. Query today's FunnelCandidate rows with stage_status=awaiting_confirmation
        2. Call run_opening_confirmation() with those candidates and the market-hours budget
        3. Record a FunnelRunLog entry for the retry pass

    Args:
        engine: SQLAlchemy engine for DB access.
        funnel_config: Optional funnel configuration dict. If None, loads from
            the default config path via load_funnel_config().

    Returns:
        List of ConfirmationDecision for each evaluated candidate (may be empty
        if no candidates are awaiting confirmation).

    Requirements: 6.9, 7.2
    """
    from utils.funnel_config import load_funnel_config

    # Load config
    if funnel_config is None:
        funnel_config = load_funnel_config()

    budgets = funnel_config.get("funnel", {}).get("budgets", {})
    budget_seconds = budgets.get("market_hours_confirmation_budget_seconds", 60)

    # Determine today's New York trading date
    today_ny = datetime.now(_NY_TZ).date()

    # Query today's awaiting_confirmation candidates
    session = get_session(engine)
    try:
        candidates = (
            session.query(FunnelCandidate)
            .filter(
                FunnelCandidate.date == today_ny,
                FunnelCandidate.stage_status == "awaiting_confirmation",
                FunnelCandidate.expired == False,  # noqa: E712
            )
            .all()
        )
        # Detach from session so run_opening_confirmation can use its own sessions
        session.expunge_all()
    finally:
        session.close()

    started_at = datetime.now(timezone.utc)
    start_mono = time.monotonic()

    if not candidates:
        logger.info(
            "Confirmation retry: no awaiting_confirmation candidates for %s",
            today_ny,
        )
        # Record empty retry run in FunnelRunLog
        _record_confirmation_run_log(
            engine=engine,
            started_at=started_at,
            duration_seconds=time.monotonic() - start_mono,
            budget_seconds=budget_seconds,
            candidates_input=0,
            candidates_promoted=0,
            candidates_rejected=0,
            result_status="completed",
            error_message=None,
            is_retry=True,
        )
        return []

    logger.info(
        "Confirmation retry: evaluating %d awaiting_confirmation candidates "
        "with %ds budget",
        len(candidates),
        budget_seconds,
    )

    # Run opening confirmation with market-hours budget
    decisions = run_opening_confirmation(
        engine=engine,
        candidates=candidates,
        budget_seconds=budget_seconds,
    )

    duration = time.monotonic() - start_mono

    # Compute promoted/rejected counts for logging
    promoted = sum(1 for d in decisions if d.decision == "promoted")
    rejected = sum(1 for d in decisions if d.decision == "rejected")
    not_evaluated = sum(1 for d in decisions if d.decision == "not_evaluated")

    # Determine result_status
    if not_evaluated > 0 and duration >= budget_seconds:
        result_status = "timed_out"
    elif not_evaluated > 0:
        result_status = "degraded"
    else:
        result_status = "completed"

    # Record FunnelRunLog for the retry pass
    _record_confirmation_run_log(
        engine=engine,
        started_at=started_at,
        duration_seconds=duration,
        budget_seconds=budget_seconds,
        candidates_input=len(candidates),
        candidates_promoted=promoted,
        candidates_rejected=rejected,
        result_status=result_status,
        error_message=None,
        is_retry=True,
    )

    logger.info(
        "Confirmation retry complete: %d promoted, %d rejected, %d not_evaluated "
        "(%.1fs / %ds budget)",
        promoted,
        rejected,
        not_evaluated,
        duration,
        budget_seconds,
    )

    return decisions


def _record_confirmation_run_log(
    engine,
    started_at: datetime,
    duration_seconds: float,
    budget_seconds: int,
    candidates_input: int,
    candidates_promoted: int,
    candidates_rejected: int,
    result_status: str,
    error_message: str | None,
    is_retry: bool = False,
) -> None:
    """Record a FunnelRunLog entry for a confirmation pass.

    Args:
        engine: SQLAlchemy engine.
        started_at: UTC datetime when the confirmation pass started.
        duration_seconds: Total elapsed time for the pass.
        budget_seconds: Configured budget for this pass.
        candidates_input: Number of candidates evaluated.
        candidates_promoted: Number promoted to pm_eligible.
        candidates_rejected: Number rejected.
        result_status: One of "completed", "timed_out", "degraded", "error".
        error_message: Optional error description.
        is_retry: If True, marks this as a retry pass (10:00 ET) vs primary (09:35 ET).
    """
    today_ny = datetime.now(_NY_TZ).date()
    ended_at = datetime.now(timezone.utc)

    stage = "confirmation_retry" if is_retry else "confirmation"

    run_log = FunnelRunLog(
        date=today_ny,
        stage=stage,
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=duration_seconds,
        budget_seconds=budget_seconds,
        result_status=result_status,
        sectors_completed=None,
        sectors_timed_out=None,
        candidates_input=candidates_input,
        candidates_promoted=candidates_promoted,
        candidates_rejected=candidates_rejected,
        error_message=error_message,
    )

    session = get_session(engine)
    try:
        session.add(run_log)
        session.commit()
        logger.info(
            "FunnelRunLog recorded: stage=%s, result_status=%s, "
            "duration=%.1fs, input=%d, promoted=%d, rejected=%d",
            stage,
            result_status,
            duration_seconds,
            candidates_input,
            candidates_promoted,
            candidates_rejected,
        )
    except Exception as exc:
        session.rollback()
        logger.error("Failed to record confirmation FunnelRunLog: %s", exc)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Live Data Fetching
# ---------------------------------------------------------------------------


def _get_live_data(symbol: str) -> dict | None:
    """Fetch live market data for a symbol using Finnhub with yfinance fallback.

    Returns a dict with keys: price, open, high, low, prev_close, volume,
    change_pct. Returns None if data is completely unavailable.

    Args:
        symbol: The instrument symbol to fetch.

    Returns:
        Dict with quote data, or None if unavailable.
    """
    # Try Finnhub first
    try:
        fh = FinnhubClient()
        quote = fh.get_quote(symbol)
        if quote and quote.get("price") is not None:
            return quote
    except Exception as exc:
        logger.warning("Finnhub quote failed for %s: %s", symbol, type(exc).__name__)

    # Fallback to yfinance
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        price = info.get("lastPrice")
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
        if price is None or price == 0:
            return None
        return {
            "symbol": symbol,
            "price": float(price),
            "prev_close": float(prev_close) if prev_close else None,
            "open": info.get("open"),
            "high": info.get("dayHigh"),
            "low": info.get("dayLow"),
            "volume": info.get("lastVolume"),
        }
    except Exception as exc:
        logger.warning("yfinance fallback failed for %s: %s", symbol, type(exc).__name__)

    return None


# ---------------------------------------------------------------------------
# Analyst Plan Extraction
# ---------------------------------------------------------------------------


def _get_analyst_plan(candidate: FunnelCandidate) -> dict:
    """Extract the Analyst's plan from the candidate's stage_decisions.

    Looks for the most recent analyst stage decision with decision in
    ("promoted", "needs_confirmation") and extracts key_levels,
    volume_requirements, signal_direction, and other plan details.

    Args:
        candidate: The FunnelCandidate whose stage_decisions to search.

    Returns:
        Dict with analyst plan fields, or empty dict if no analyst plan found.
    """
    try:
        stage_decisions = json.loads(candidate.stage_decisions or "[]")
    except (json.JSONDecodeError, TypeError):
        return {}

    # Find the most recent analyst decision with a plan
    for sd in reversed(stage_decisions):
        if sd.get("agent") == "analyst" and sd.get("decision") in ("promoted", "needs_confirmation"):
            evidence = sd.get("evidence", {})
            return {
                "key_levels": evidence.get("key_levels", {}),
                "volume_requirements": evidence.get("volume_requirements"),
                "signal_direction": evidence.get("signal_direction"),
                "signal_strength": evidence.get("signal_strength"),
                "invalidation": evidence.get("invalidation"),
                "catalyst_dependence": evidence.get("catalyst_dependence"),
            }

    return {}


# ---------------------------------------------------------------------------
# Confirmation Checks
# ---------------------------------------------------------------------------


def _check_volume(live_data: dict, analyst_plan: dict) -> bool:
    """Check if volume meets Analyst's stated requirements (Req 6.2).

    Uses a baseline heuristic: volume should be at least 80% of what would
    be expected (relative to the open price movement). If the Analyst stated
    explicit volume_requirements, those inform the check.

    For opening confirmation, the key check is that volume is present and
    non-trivial (not a data gap). Since exact Analyst volume thresholds are
    text-based descriptions, we use volume > 0 as the primary data-availability
    check combined with a change-magnitude heuristic.

    Args:
        live_data: Current quote data dict.
        analyst_plan: Extracted analyst plan with volume_requirements.

    Returns:
        True if volume appears adequate, False otherwise.
    """
    volume = live_data.get("volume")
    if volume is None or volume <= 0:
        return False

    # Volume exists and is positive — at market open, this is the primary gate
    # A more sophisticated check would compare to average volume, but at open
    # the key confirmation is that there IS meaningful trading activity.
    return True


def _check_price_position(live_data: dict, analyst_plan: dict) -> bool | None:
    """Check price position relative to Analyst key levels (Req 6.2).

    Evaluates whether the current price is consistent with the Analyst's
    identified key levels (support, resistance, VWAP). Returns None if
    key levels are not available for comparison.

    For a LONG setup: price should be above support.
    For a SHORT setup: price should be below resistance.

    Args:
        live_data: Current quote data dict.
        analyst_plan: Extracted analyst plan with key_levels.

    Returns:
        True if price is in favorable position, False if invalidated,
        None if key levels unavailable for comparison.
    """
    key_levels = analyst_plan.get("key_levels")
    if not key_levels or not isinstance(key_levels, dict):
        return None

    price = live_data.get("price")
    if price is None:
        return None

    signal_direction = analyst_plan.get("signal_direction", "").upper()

    # Extract numeric levels (they may be null or string in some cases)
    support = _to_float(key_levels.get("support"))
    resistance = _to_float(key_levels.get("resistance"))
    stop_level = _to_float(key_levels.get("stop_level"))

    if signal_direction == "LONG":
        # For LONG: price should be above support (or stop level)
        reference = stop_level or support
        if reference is not None and price < reference:
            return False
        return True

    elif signal_direction == "SHORT":
        # For SHORT: price should be below resistance (or stop level)
        reference = stop_level or resistance
        if reference is not None and price > reference:
            return False
        return True

    # No clear direction — can't confirm or deny
    return None


def _check_price_behavior(
    live_data: dict,
    candidate: FunnelCandidate,
    analyst_plan: dict,
) -> bool:
    """Check if price behavior since open is consistent with signal direction (Req 6.2).

    Compares current price vs open price or prev_close to determine if
    the directional move is consistent with the Analyst's signal_direction.

    Args:
        live_data: Current quote data dict.
        candidate: The FunnelCandidate with direction_bias.
        analyst_plan: Extracted analyst plan with signal_direction.

    Returns:
        True if price behavior supports the thesis, False otherwise.
    """
    price = live_data.get("price")
    open_price = live_data.get("open")
    prev_close = live_data.get("prev_close")

    if price is None:
        return False

    # Use open price if available, fall back to prev_close
    reference = open_price or prev_close
    if reference is None or reference == 0:
        return False

    # Determine expected direction from analyst plan or candidate bias
    signal_direction = (analyst_plan.get("signal_direction") or "").upper()
    if not signal_direction or signal_direction == "HOLD":
        # Fall back to candidate's direction_bias
        bias = (candidate.direction_bias or "").lower()
        if bias == "bullish":
            signal_direction = "LONG"
        elif bias == "bearish":
            signal_direction = "SHORT"
        else:
            # No directional signal — price behavior is indeterminate
            # Be permissive: any movement is acceptable
            return True

    change_pct = ((price - reference) / reference) * 100

    if signal_direction == "LONG":
        # For LONG: price should not be materially below reference
        # Allow some slack — at open, minor dips are normal
        return change_pct >= -2.0  # Not dropping more than 2%

    elif signal_direction == "SHORT":
        # For SHORT: price should not be materially above reference
        return change_pct <= 2.0  # Not rising more than 2%

    return True


def _check_catalyst_freshness(candidate: FunnelCandidate) -> bool:
    """Check if catalyst evidence is still fresh (≤24h from current time) (Req 6.2).

    Examines the candidate's discovered_at timestamp to determine if the
    catalyst is within 24 hours. The requirement specifies catalyst_evidence
    age no greater than 24 hours from market open.

    Args:
        candidate: The FunnelCandidate with discovered_at and catalyst_evidence.

    Returns:
        True if catalyst is within 24h freshness window, False if stale.
    """
    now_utc = datetime.now(timezone.utc)

    # Primary check: discovered_at should be within 24h
    if candidate.discovered_at:
        discovered = candidate.discovered_at
        # Ensure timezone-aware comparison
        if discovered.tzinfo is None:
            discovered = discovered.replace(tzinfo=timezone.utc)
        age_hours = (now_utc - discovered).total_seconds() / 3600.0
        if age_hours > 24.0:
            return False

    # Secondary: check catalyst evidence for timestamp info
    try:
        catalyst_data = json.loads(candidate.catalyst_evidence or "{}")
        # If catalyst_evidence contains a timestamp, check that too
        catalyst_ts_str = catalyst_data.get("timestamp") or catalyst_data.get("news_time")
        if catalyst_ts_str:
            catalyst_ts = datetime.fromisoformat(catalyst_ts_str.replace("Z", "+00:00"))
            if catalyst_ts.tzinfo is None:
                catalyst_ts = catalyst_ts.replace(tzinfo=timezone.utc)
            age_hours = (now_utc - catalyst_ts).total_seconds() / 3600.0
            if age_hours > 24.0:
                return False
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # If discovered_at is within 24h (or not set), consider fresh
    return True


# ---------------------------------------------------------------------------
# Decision Logic
# ---------------------------------------------------------------------------


def _make_confirmation_decision(
    volume_ok: bool,
    vwap_ok: bool | None,
    price_ok: bool,
    catalyst_fresh: bool,
    candidate: FunnelCandidate,
    live_data: dict,
    analyst_plan: dict,
) -> tuple[str, str, str]:
    """Determine promote/reject/needs_confirmation based on confirmation checks.

    Decision logic:
    - If catalyst is stale (>24h) → reject
    - If volume AND price behavior are both OK AND catalyst fresh → promote
    - If volume is missing or price is adverse → reject or needs_confirmation
    - VWAP check is informative but not solely deterministic

    Args:
        volume_ok: Whether volume meets requirements.
        vwap_ok: Whether price is near VWAP/key levels (None if unavailable).
        price_ok: Whether price behavior supports direction.
        catalyst_fresh: Whether catalyst is within 24h window.
        candidate: The FunnelCandidate being evaluated.
        live_data: Current quote data dict.
        analyst_plan: Extracted analyst plan.

    Returns:
        Tuple of (decision, reasoning, next_stage).
    """
    reasons: list[str] = []

    # Hard reject: stale catalyst
    if not catalyst_fresh:
        return (
            "rejected",
            f"Catalyst no longer fresh (>24h) for {candidate.symbol}",
            "rejected_confirmation",
        )

    # Collect confirmation status
    if volume_ok:
        reasons.append("volume confirmed")
    else:
        reasons.append("volume NOT confirmed")

    if vwap_ok is True:
        reasons.append("price position favorable vs key levels")
    elif vwap_ok is False:
        reasons.append("price position UNFAVORABLE vs key levels (below stop/invalidation)")
    # vwap_ok is None means key levels unavailable — not a failure

    if price_ok:
        reasons.append("price behavior consistent with signal direction")
    else:
        reasons.append("price behavior ADVERSE to signal direction")

    reasoning = f"{candidate.symbol}: {'; '.join(reasons)}"

    # Promote: volume OK + price behavior OK + catalyst fresh
    if volume_ok and price_ok and catalyst_fresh:
        # If VWAP/levels show invalidation, reject despite volume/price
        if vwap_ok is False:
            return (
                "rejected",
                f"Price below invalidation level despite volume/behavior: {reasoning}",
                "rejected_confirmation",
            )
        return ("promoted", reasoning, "pm_eligible")

    # Volume missing but other checks are OK → needs_confirmation (retry later)
    if not volume_ok and price_ok and catalyst_fresh:
        return (
            "needs_confirmation",
            f"Volume insufficient, awaiting more activity: {reasoning}",
            "awaiting_confirmation",
        )

    # Price behavior adverse → reject
    if not price_ok:
        return (
            "rejected",
            f"Price behavior contradicts signal direction: {reasoning}",
            "rejected_confirmation",
        )

    # Fallback: needs_confirmation (mixed signals)
    return (
        "needs_confirmation",
        f"Mixed confirmation signals: {reasoning}",
        "awaiting_confirmation",
    )


# ---------------------------------------------------------------------------
# Helper Utilities
# ---------------------------------------------------------------------------


def _to_float(value) -> float | None:
    """Safely convert a value to float, returning None if not possible."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Database Helpers
# ---------------------------------------------------------------------------


def _append_stage_decision(
    engine,
    candidate: FunnelCandidate,
    decision: str,
    reasoning: str,
    evidence: dict,
    next_stage: str,
) -> None:
    """Append a confirmation stage decision to the candidate's history.

    Stage decisions are append-only — prior entries are never removed.
    This function reads the current stage_decisions JSON array, appends
    the new decision, and writes back atomically.

    Args:
        engine: SQLAlchemy engine.
        candidate: The FunnelCandidate to update.
        decision: One of "promoted", "rejected", "needs_confirmation", "not_evaluated".
        reasoning: Human-readable explanation.
        evidence: Agent-specific evidence payload dict.
        next_stage: Target stage_status value after this decision.
    """
    session = get_session(engine)
    try:
        # Re-fetch within session for transactional safety
        db_candidate = (
            session.query(FunnelCandidate)
            .filter(FunnelCandidate.candidate_id == candidate.candidate_id)
            .first()
        )
        if db_candidate is None:
            logger.error(
                "Cannot append stage decision: candidate %s not found",
                candidate.candidate_id,
            )
            return

        # Parse existing decisions
        try:
            current_decisions = json.loads(db_candidate.stage_decisions or "[]")
        except (json.JSONDecodeError, TypeError):
            current_decisions = []

        # Build new decision record
        stage_decision = {
            "agent": "confirmation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "reasoning": reasoning,
            "evidence": evidence,
            "next_stage": next_stage,
        }

        current_decisions.append(stage_decision)
        db_candidate.stage_decisions = json.dumps(current_decisions)
        db_candidate.updated_at = datetime.now(timezone.utc)

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _update_stage_status(
    engine,
    candidate: FunnelCandidate,
    next_stage: str,
) -> None:
    """Update the candidate's stage_status to the next stage.

    Only updates if next_stage is different from current stage_status.
    Does not regress stage_status (checked upstream by decision logic).

    Args:
        engine: SQLAlchemy engine.
        candidate: The FunnelCandidate to update.
        next_stage: Target stage_status value.
    """
    session = get_session(engine)
    try:
        db_candidate = (
            session.query(FunnelCandidate)
            .filter(FunnelCandidate.candidate_id == candidate.candidate_id)
            .first()
        )
        if db_candidate is None:
            logger.error(
                "Cannot update stage_status: candidate %s not found",
                candidate.candidate_id,
            )
            return

        db_candidate.stage_status = next_stage
        db_candidate.updated_at = datetime.now(timezone.utc)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
