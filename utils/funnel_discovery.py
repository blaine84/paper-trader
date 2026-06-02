"""Funnel Discovery Service — bounded premarket sector scanning with curation.

Orchestrates per-sector budget enforcement, Chief Scout LLM curation with
deterministic fallback, and FunnelCandidate persistence.

See: design.md §Component 2, requirements.md §1, §2
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from db.schema import FunnelCandidate, FunnelRunLog, get_session
from utils.finnhub_client import FinnhubClient
from utils.sector_scout import (
    load_sector_scout_config,
    run_sector_screeners,
)
from utils.sector_scout_chief import run_chief_scout, chief_scout_fallback
from utils.sector_scout_models import CandidateRow, ChiefScoutPick

logger = logging.getLogger(__name__)

NY_TZ = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Result Dataclass
# ---------------------------------------------------------------------------


@dataclass
class DiscoveryResult:
    """Result of a bounded premarket discovery execution."""

    candidates: list = field(default_factory=list)
    sectors_completed: list[str] = field(default_factory=list)
    sectors_timed_out: list[str] = field(default_factory=list)
    sectors_skipped: list[str] = field(default_factory=list)
    selection_mode: str = "deterministic_fallback"
    partial_screening: bool = False
    total_duration_seconds: float = 0.0
    pipeline_budget_exhausted: bool = False


# ---------------------------------------------------------------------------
# FunnelRunLog Recording (Task 3.4)
# ---------------------------------------------------------------------------


def record_discovery_run_log(
    engine,
    result: DiscoveryResult,
    started_at: datetime,
    budget_seconds: float,
    error_message: str | None = None,
) -> FunnelRunLog:
    """Create a FunnelRunLog entry for a discovery execution.

    Records timing, sector completion status, candidate counts, and result
    status for every discovery run — including empty or degraded results.

    Result status determination:
    - "completed": Full discovery completed normally (including empty discovery
      with zero candidates — a valid result per Requirement 1.8).
    - "degraded": Usable partial/fallback output was persisted. This applies
      when deterministic fallback was used because Chief Scout LLM curation
      failed or timed out (Requirement 2.3).
    - "timed_out": No usable output produced (pipeline budget exhausted with
      no candidates from any sector).
    - "error": Failed unexpectedly with no usable output.

    Args:
        engine: SQLAlchemy engine for DB access.
        result: The DiscoveryResult from the discovery execution.
        started_at: UTC datetime when discovery started.
        budget_seconds: Total pipeline budget configured for this run.
        error_message: Optional error/timeout description (set when Chief Scout
            curation failed/timed out or unexpected error occurred).

    Returns:
        The persisted FunnelRunLog instance.

    Requirements: 1.8, 1.12, 2.3, 2.5, 7.6
    """
    ended_at = datetime.now(timezone.utc)

    # Determine result_status based on outcome
    result_status = _determine_discovery_result_status(result, error_message)

    # Compute the New York trading date
    ny_date = datetime.now(NY_TZ).date()

    # sectors_completed and sectors_timed_out stored as JSON arrays in Text columns
    sectors_completed_json = json.dumps(result.sectors_completed) if result.sectors_completed else None
    sectors_timed_out_json = json.dumps(result.sectors_timed_out) if result.sectors_timed_out else None

    run_log = FunnelRunLog(
        date=ny_date,
        stage="discovery",
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=result.total_duration_seconds,
        budget_seconds=budget_seconds,
        result_status=result_status,
        sectors_completed=sectors_completed_json,
        sectors_timed_out=sectors_timed_out_json,
        candidates_input=len(result.candidates),
        candidates_promoted=len(result.candidates),
        candidates_rejected=0,
        error_message=error_message,
    )

    session = get_session(engine)
    try:
        session.expire_on_commit = False
        session.add(run_log)
        session.commit()
        logger.info(
            "FunnelRunLog recorded: stage=discovery, result_status=%s, "
            "duration=%.1fs, candidates=%d, sectors_completed=%s, "
            "sectors_timed_out=%s",
            result_status,
            result.total_duration_seconds,
            len(result.candidates),
            result.sectors_completed,
            result.sectors_timed_out,
        )
    except Exception as exc:
        session.rollback()
        logger.error("Failed to record FunnelRunLog: %s", exc)
        raise
    finally:
        session.close()

    return run_log


def _determine_discovery_result_status(
    result: DiscoveryResult,
    error_message: str | None,
) -> str:
    """Determine the result_status for a discovery FunnelRunLog entry.

    Status semantics (from design doc / Requirement 2.5):
    - "completed": Full intended stage completed normally. This includes empty
      discovery where no candidates qualified — the pipeline ran to completion.
    - "degraded": Usable partial/fallback output was persisted despite curation
      failure. Applied when selection_mode is "deterministic_fallback" AND there
      is an error_message (indicating Chief Scout attempted and failed).
    - "timed_out": No usable stage output was produced (all sectors skipped/timed
      out, zero candidates).
    - "error": Failed unexpectedly with no usable output.
    """
    # Error case: explicit error with no usable candidates
    if error_message and not result.candidates:
        # If we have an error but sectors did complete (just no finalists),
        # it's still "completed" if the pipeline ran to completion.
        # However if error_message is set and no candidates exist due to
        # an unexpected failure, that's "error".
        if result.pipeline_budget_exhausted and not result.sectors_completed:
            return "timed_out"
        # If there are completed sectors but no candidates and an error,
        # distinguish between "empty discovery completed" vs "error"
        if not result.sectors_completed and not result.sectors_timed_out:
            return "error"

    # Deterministic fallback with an error_message means Chief Scout curation
    # failed/timed out but we have usable fallback candidates → "degraded"
    if (
        result.selection_mode == "deterministic_fallback"
        and error_message
        and result.candidates
    ):
        return "degraded"

    # Pipeline budget exhausted with no candidates from any completed sector
    if result.pipeline_budget_exhausted and not result.candidates:
        if not result.sectors_completed:
            return "timed_out"

    # Successful completion — includes empty discovery when pipeline ran normally
    # (Requirement 1.8: empty discovery = "completed", not "missing/timed_out")
    if result.selection_mode == "chief_scout":
        return "completed"

    # Deterministic fallback without error_message means no curation was
    # attempted (e.g. no finalists, or insufficient budget < 10s but not
    # a failure). This is a normal completion.
    if result.selection_mode == "deterministic_fallback" and not error_message:
        return "completed"

    return "completed"


# ---------------------------------------------------------------------------
# Chief Scout Curation with Deterministic Fallback (Task 3.2)
# ---------------------------------------------------------------------------


def _deterministic_fallback_sort_key(candidate) -> tuple:
    """Sort key for deterministic fallback ranking.

    Ranking order per requirements 1.10, 2.1:
      1. scout_score descending
      2. symbol ascending (tie-breaker)

    This is the funnel-specific fallback ranking which differs from the
    existing chief_scout_fallback() in sector_scout_chief.py — that one
    uses a more complex 4-field sort key suited for expanded watchlist.
    The funnel fallback uses the simpler 2-field ranking specified in
    the requirements: scout_score DESC, symbol ASC for ties.
    """
    if hasattr(candidate, "scout_score"):
        score = candidate.scout_score
        symbol = candidate.symbol
    elif isinstance(candidate, dict):
        score = candidate.get("scout_score", 0.0)
        symbol = candidate.get("symbol", "")
    else:
        return (0.0, "")

    return (-score, symbol)


# ---------------------------------------------------------------------------
# Adapter Functions (Task 3.3)
# ---------------------------------------------------------------------------


def adapt_candidate_row_to_funnel(row: CandidateRow) -> dict:
    """Adapt a CandidateRow (from sector screeners) to FunnelCandidate fields.

    CandidateRow has: symbol, sector, sector_name, scout_score, move_pct,
    relative_volume, dollar_volume, news_freshness_minutes, sector_confirmation,
    spread_pct, etc.

    This adapter normalizes CandidateRow into the FunnelCandidate field set
    for the deterministic fallback path.

    Args:
        row: A scored CandidateRow from sector screening.

    Returns:
        Dict of FunnelCandidate field values (excluding rank, which is
        assigned by persist_discovery_candidates based on position).
    """
    # Build catalyst evidence from news and move data
    catalyst_evidence = json.dumps({
        "news_headlines": row.news_headlines or [],
        "news_freshness_minutes": row.news_freshness_minutes,
        "move_pct": row.move_pct,
        "relative_volume": row.relative_volume,
        "dollar_volume": row.dollar_volume,
    })

    # Build sector context JSON
    sector_context = json.dumps({
        "sector": row.sector,
        "sector_name": row.sector_name,
        "sector_etf": row.sector_etf,
        "sector_etf_move_pct": row.sector_etf_move_pct,
        "sector_confirmed": row.sector_confirmed,
    })

    # Derive selection reason from component scores and move data
    reason_parts = []
    if row.move_pct is not None:
        reason_parts.append(f"Move: {row.move_pct:+.1f}%")
    if row.relative_volume is not None:
        reason_parts.append(f"RelVol: {row.relative_volume:.1f}x")
    if row.news_freshness_minutes is not None:
        reason_parts.append(f"News: {row.news_freshness_minutes:.0f}min ago")
    if row.sector_confirmed:
        reason_parts.append("Sector confirmed")
    selection_reason = "; ".join(reason_parts) if reason_parts else "Deterministic screening score"

    # Derive primary risk
    risk_parts = []
    if row.spread_pct is not None and row.spread_pct > 0.5:
        risk_parts.append(f"Wide spread ({row.spread_pct:.2f}%)")
    if row.news_freshness_minutes is not None and row.news_freshness_minutes > 120:
        risk_parts.append("Aging news catalyst")
    if not row.sector_confirmed:
        risk_parts.append("No sector confirmation")
    if row.missing_data_flags:
        risk_parts.append(f"Missing data: {', '.join(row.missing_data_flags)}")
    primary_risk = "; ".join(risk_parts) if risk_parts else "Standard screening risk"

    # Derive direction_bias from move direction
    direction_bias: str | None = None
    if row.move_pct is not None:
        if row.move_pct > 0.5:
            direction_bias = "bullish"
        elif row.move_pct < -0.5:
            direction_bias = "bearish"
        else:
            direction_bias = "neutral"

    return {
        "symbol": row.symbol,
        "scout_score": row.scout_score,
        "direction_bias": direction_bias,
        "catalyst_evidence": catalyst_evidence,
        "selection_reason": selection_reason,
        "primary_risk": primary_risk,
        "sector_context": sector_context,
        "preliminary_setup_type": None,
    }


def adapt_chief_scout_pick_to_funnel(pick: ChiefScoutPick) -> dict:
    """Adapt a ChiefScoutPick (from Chief Scout LLM curation) to FunnelCandidate fields.

    Maps ChiefScoutPick fields to FunnelCandidate columns:
    - pick.source_candidate_score → scout_score
    - pick.direction_bias → direction_bias
    - pick.catalyst_summary → catalyst_evidence (JSON-wrapped)
    - pick.reason → selection_reason
    - pick.risk → primary_risk
    - pick.sector → sector_context (JSON-wrapped with sector info)

    Args:
        pick: A ChiefScoutPick dict from Chief Scout LLM curation.

    Returns:
        Dict of FunnelCandidate field values (excluding rank, which is
        assigned by persist_discovery_candidates based on position).
    """
    # ChiefScoutPick is a TypedDict, access via dict notation
    symbol = pick["symbol"] if isinstance(pick, dict) else pick.symbol
    sector = pick["sector"] if isinstance(pick, dict) else pick.sector
    direction_bias = pick["direction_bias"] if isinstance(pick, dict) else pick.direction_bias
    conviction = pick["conviction"] if isinstance(pick, dict) else pick.conviction
    catalyst_summary = pick["catalyst_summary"] if isinstance(pick, dict) else pick.catalyst_summary
    reason = pick["reason"] if isinstance(pick, dict) else pick.reason
    risk = pick["risk"] if isinstance(pick, dict) else pick.risk
    source_score = pick["source_candidate_score"] if isinstance(pick, dict) else pick.source_candidate_score

    # Build catalyst evidence from LLM curation output
    catalyst_evidence = json.dumps({
        "catalyst_summary": catalyst_summary,
        "conviction": conviction,
        "curated_by": "chief_scout",
    })

    # Build sector context JSON
    sector_context = json.dumps({
        "sector": sector,
    })

    # Validate direction_bias
    valid_bias = {"bullish", "bearish", "neutral"}
    if direction_bias not in valid_bias:
        direction_bias = None

    return {
        "symbol": symbol,
        "scout_score": float(source_score) if source_score is not None else 0.0,
        "direction_bias": direction_bias,
        "catalyst_evidence": catalyst_evidence,
        "selection_reason": reason or "Chief Scout curated pick",
        "primary_risk": risk or "No specific risk identified",
        "sector_context": sector_context,
        "preliminary_setup_type": None,
    }


# ---------------------------------------------------------------------------
# Persistence (Task 3.3)
# ---------------------------------------------------------------------------

# Stage status values that represent advancement beyond initial discovery
_ADVANCED_STAGES = {
    "awaiting_analysis",
    "awaiting_confirmation",
    "pm_eligible",
    "executed",
}


def persist_discovery_candidates(
    engine,
    finalists: list,
    selection_mode: str,
    source_run: str,
    max_shortlist: int = 5,
) -> list[FunnelCandidate]:
    """Create FunnelCandidate rows from ranked finalists.

    Each candidate is persisted with:
    - stage_status = "awaiting_research" (immediately ready for Researcher)
    - An initial Scout stage decision appended to stage_decisions
    - No intermediate 'discovered' state is externally visible

    Handles same-day duplicates (date, symbol):
    - Updates mutable fields (catalyst_evidence, selection_reason, etc.)
    - Updates scout_score only if new score is higher
    - Appends a new Scout stage decision (never removes prior decisions)
    - Never regresses stage_status if candidate has advanced beyond awaiting_research

    Args:
        engine: SQLAlchemy engine.
        finalists: Ranked list of CandidateRow or ChiefScoutPick objects.
        selection_mode: "chief_scout" or "deterministic_fallback".
        source_run: "premarket", "confirmation", or "manual_intraday".
        max_shortlist: Maximum candidates to persist (default 5).

    Returns:
        List of persisted FunnelCandidate objects.
    """
    import uuid
    from db.schema import get_session

    # Enforce max_shortlist ceiling
    bounded = finalists[:max_shortlist]
    if not bounded:
        return []

    # Compute today's New York trading date
    ny_tz = ZoneInfo("America/New_York")
    now_utc = datetime.now(timezone.utc)
    today_ny = now_utc.astimezone(ny_tz).date()

    persisted: list[FunnelCandidate] = []
    session = get_session(engine)

    try:
        for rank_idx, finalist in enumerate(bounded, start=1):
            # Determine if this is a CandidateRow or ChiefScoutPick and adapt
            if isinstance(finalist, CandidateRow):
                fields = adapt_candidate_row_to_funnel(finalist)
            elif isinstance(finalist, dict):
                # ChiefScoutPick is a TypedDict — appears as dict at runtime
                fields = adapt_chief_scout_pick_to_funnel(finalist)
            else:
                # Fallback: try CandidateRow adapter
                fields = adapt_candidate_row_to_funnel(finalist)

            symbol = fields["symbol"]

            # Build the Scout stage decision record
            scout_decision = {
                "agent": "scout",
                "timestamp": now_utc.isoformat(),
                "decision": "promoted",
                "reasoning": fields["selection_reason"],
                "evidence": {
                    "scout_score": fields["scout_score"],
                    "direction_bias": fields["direction_bias"],
                    "catalyst_evidence": fields["catalyst_evidence"],
                    "selection_mode": selection_mode,
                    "source_run": source_run,
                    "scout_rank": rank_idx,
                },
                "next_stage": "awaiting_research",
            }

            # Check for existing same-day duplicate
            existing = (
                session.query(FunnelCandidate)
                .filter(
                    FunnelCandidate.date == today_ny,
                    FunnelCandidate.symbol == symbol,
                )
                .first()
            )

            if existing:
                # Same-day duplicate handling:
                # - Update mutable fields
                # - Update scout_score only if new score is higher
                # - Append Scout decision (never remove prior)
                # - Never regress stage_status
                _update_existing_candidate(
                    existing, fields, rank_idx, selection_mode, source_run,
                    scout_decision
                )
                persisted.append(existing)
            else:
                # Create new FunnelCandidate
                candidate = FunnelCandidate(
                    candidate_id=str(uuid.uuid4()),
                    date=today_ny,
                    symbol=symbol,
                    discovered_at=now_utc,
                    source_run=source_run,
                    selection_mode=selection_mode,
                    scout_rank=rank_idx,
                    scout_score=fields["scout_score"],
                    direction_bias=fields["direction_bias"],
                    catalyst_evidence=fields["catalyst_evidence"],
                    selection_reason=fields["selection_reason"],
                    primary_risk=fields["primary_risk"],
                    sector_context=fields["sector_context"],
                    preliminary_setup_type=fields.get("preliminary_setup_type"),
                    stage_status="awaiting_research",
                    stage_decisions=json.dumps([scout_decision]),
                    expired=False,
                )
                session.add(candidate)
                persisted.append(candidate)

        session.commit()
        # Refresh and expunge all objects so they retain loaded state after session close
        for candidate in persisted:
            session.refresh(candidate)
            session.expunge(candidate)
    except Exception:
        session.rollback()
        logger.exception("Failed to persist discovery candidates")
        raise
    finally:
        session.close()

    logger.info(
        "Persisted %d funnel candidates (selection_mode=%s, source_run=%s)",
        len(persisted),
        selection_mode,
        source_run,
    )
    return persisted


def _update_existing_candidate(
    existing: FunnelCandidate,
    fields: dict,
    rank: int,
    selection_mode: str,
    source_run: str,
    scout_decision: dict,
) -> None:
    """Update an existing same-day FunnelCandidate with new discovery data.

    Rules per design spec:
    - candidate_id: immutable
    - discovered_at: immutable
    - scout_score/scout_rank: updated if new score is higher
    - selection_mode, source_run: updated to latest context
    - catalyst_evidence, selection_reason, primary_risk, sector_context: updated
    - direction_bias, preliminary_setup_type: updated if changed
    - stage_decisions: append-only (new Scout decision appended)
    - stage_status: never regressed if already advanced past awaiting_research
    - authoritative_setup_type: immutable once set
    - trade_event_id, blocked_candidate_id: immutable once set
    """
    new_score = fields["scout_score"]

    # Update scout_score and rank only if new score is higher
    if new_score > existing.scout_score:
        existing.scout_score = new_score
        existing.scout_rank = rank

    # Always update mutable evidence fields (latest evidence replaces stale)
    existing.catalyst_evidence = fields["catalyst_evidence"]
    existing.selection_reason = fields["selection_reason"]
    existing.primary_risk = fields["primary_risk"]
    existing.sector_context = fields["sector_context"]

    # Update context fields
    existing.selection_mode = selection_mode
    existing.source_run = source_run

    # Update direction_bias and preliminary_setup_type if changed
    if fields["direction_bias"] is not None:
        existing.direction_bias = fields["direction_bias"]
    if fields.get("preliminary_setup_type") is not None:
        existing.preliminary_setup_type = fields["preliminary_setup_type"]

    # Append Scout decision (stage_decisions is append-only)
    try:
        current_decisions = json.loads(existing.stage_decisions or "[]")
    except (json.JSONDecodeError, TypeError):
        current_decisions = []
    current_decisions.append(scout_decision)
    existing.stage_decisions = json.dumps(current_decisions)

    # Never regress stage_status if candidate has advanced
    if existing.stage_status not in _ADVANCED_STAGES:
        existing.stage_status = "awaiting_research"

    existing.updated_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Chief Scout Curation with Deterministic Fallback (Task 3.2)
# ---------------------------------------------------------------------------


def run_chief_scout_curation(
    finalists: list[CandidateRow],
    remaining_budget: float,
    config: dict,
    engine,
    core_watchlist: list[str] | None = None,
    max_shortlist: int = 5,
) -> tuple[list, str, str | None]:
    """Attempt Chief Scout LLM curation with deterministic fallback.

    Invokes Chief Scout when remaining budget > 10s and finalists exist.
    Falls back to deterministic ranking (scout_score DESC, symbol ASC)
    when curation fails, times out, or cannot start.

    Args:
        finalists: Ranked finalist CandidateRows from sector screening.
        remaining_budget: Seconds remaining in total pipeline budget.
        config: Sector scout configuration dict (passed to Chief Scout).
        engine: SQLAlchemy engine for case library context.
        core_watchlist: Current core watchlist symbols for Chief Scout context.
        max_shortlist: Maximum candidates to return (default 5).

    Returns:
        Tuple of (curated_list, selection_mode, curation_error) where:
        - curated_list: List of candidates (CandidateRow or ChiefScoutPick)
        - selection_mode: "chief_scout" or "deterministic_fallback"
        - curation_error: None if curation succeeded or was not attempted,
          otherwise a string describing the timeout/error that triggered fallback.
    """
    if core_watchlist is None:
        core_watchlist = []

    # If no finalists, return empty with deterministic fallback mode
    if not finalists:
        logger.info("Chief Scout curation: no finalists to curate")
        return [], "deterministic_fallback", None

    # Check if we have enough budget to attempt LLM curation (>10s required)
    if remaining_budget <= 10:
        logger.info(
            "Chief Scout curation: insufficient budget (%.1fs remaining, need >10s), "
            "using deterministic fallback",
            remaining_budget,
        )
        ranked = _apply_deterministic_fallback(finalists, max_shortlist)
        curation_error = (
            f"Chief Scout curation skipped: insufficient budget "
            f"({remaining_budget:.1f}s remaining, need >10s)"
        )
        return ranked, "deterministic_fallback", curation_error

    # Attempt Chief Scout LLM curation
    try:
        curated = _invoke_chief_scout_with_timeout(
            finalists=finalists,
            config=config,
            engine=engine,
            core_watchlist=core_watchlist,
            timeout=remaining_budget,
        )

        if curated is not None and len(curated) > 0:
            # Chief Scout succeeded — truncate to max_shortlist
            logger.info(
                "Chief Scout curation: LLM returned %d picks (max_shortlist=%d)",
                len(curated),
                max_shortlist,
            )
            return curated[:max_shortlist], "chief_scout", None
        else:
            # Chief Scout returned empty or None — fall back
            logger.warning(
                "Chief Scout curation: LLM returned no picks, "
                "using deterministic fallback"
            )
            ranked = _apply_deterministic_fallback(finalists, max_shortlist)
            return ranked, "deterministic_fallback", "Chief Scout LLM returned no picks"

    except TimeoutError:
        logger.warning(
            "Chief Scout curation: timed out (budget=%.1fs), "
            "using deterministic fallback",
            remaining_budget,
        )
        ranked = _apply_deterministic_fallback(finalists, max_shortlist)
        curation_error = (
            f"Chief Scout curation timed out after {remaining_budget:.1f}s budget"
        )
        return ranked, "deterministic_fallback", curation_error

    except Exception as exc:
        logger.warning(
            "Chief Scout curation: failed (%s: %s), using deterministic fallback",
            type(exc).__name__,
            exc,
        )
        ranked = _apply_deterministic_fallback(finalists, max_shortlist)
        curation_error = f"Chief Scout curation failed: {type(exc).__name__}: {exc}"
        return ranked, "deterministic_fallback", curation_error


def _apply_deterministic_fallback(
    finalists: list[CandidateRow],
    max_shortlist: int,
) -> list[CandidateRow]:
    """Rank finalists deterministically: scout_score DESC, symbol ASC for ties.

    Returns at most max_shortlist candidates.

    This implements Requirements 1.10, 2.1:
    - Ranked by scout_score descending
    - Ties broken by symbol name ascending
    - Bounded by max_discovery_shortlist ceiling
    """
    sorted_finalists = sorted(finalists, key=_deterministic_fallback_sort_key)
    return sorted_finalists[:max_shortlist]


def _invoke_chief_scout_with_timeout(
    finalists: list[CandidateRow],
    config: dict,
    engine,
    core_watchlist: list[str],
    timeout: float,
) -> list[ChiefScoutPick] | None:
    """Invoke Chief Scout LLM with a timeout using ThreadPoolExecutor.

    Args:
        finalists: All finalists from completed sectors.
        config: Sector scout config dict.
        engine: SQLAlchemy engine for case library context.
        core_watchlist: Core watchlist symbols.
        timeout: Maximum seconds to wait for LLM response.

    Returns:
        List of ChiefScoutPick dicts if successful, None if no picks.

    Raises:
        TimeoutError: If LLM call exceeds the timeout.
        Exception: If LLM call fails for any reason.
    """
    # Group finalists by sector for the Chief Scout prompt
    finalists_by_sector: dict[str, list[CandidateRow]] = {}
    for candidate in finalists:
        sector = candidate.sector if hasattr(candidate, "sector") else "unknown"
        if sector not in finalists_by_sector:
            finalists_by_sector[sector] = []
        finalists_by_sector[sector].append(candidate)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            run_chief_scout,
            finalists_by_sector,
            core_watchlist,
            config,
            engine,
        )

        try:
            result = future.result(timeout=timeout)
        except FuturesTimeout:
            future.cancel()
            raise TimeoutError(
                f"Chief Scout LLM call exceeded timeout of {timeout:.1f}s"
            )

    # Check for LLM-level error in the result
    llm_error = result.get("llm_error")
    if llm_error:
        raise RuntimeError(f"Chief Scout LLM error: {llm_error}")

    picks = result.get("picks", [])
    return picks if picks else None


# ---------------------------------------------------------------------------
# Core Discovery Pipeline (Task 3.1 scaffold — to be fully implemented)
# ---------------------------------------------------------------------------


def get_enabled_sectors(config: dict) -> list[str]:
    """Return list of enabled sector keys from sector scout config.

    Respects the max_sectors_per_run ceiling from budget_ceilings.

    Args:
        config: Parsed sector_scout_config.yaml dict.

    Returns:
        List of enabled sector key strings, limited by max_sectors_per_run.
    """
    sector_buckets = config.get("sector_buckets", {})
    budget_ceilings = config.get("budget_ceilings", {})
    max_sectors_per_run: int = int(budget_ceilings.get("max_sectors_per_run", 7))

    enabled: list[str] = []
    for sector_key, bucket in sector_buckets.items():
        if bucket.get("enabled", False):
            enabled.append(sector_key)
        if len(enabled) >= max_sectors_per_run:
            break

    return enabled


def run_sector_with_timeout(
    sector_key: str,
    config: dict,
    timeout: float,
    fh: FinnhubClient | None = None,
    core_watchlist: list[str] | None = None,
) -> list[CandidateRow]:
    """Run a single sector's screening with wall-clock timeout enforcement.

    Uses concurrent.futures.ThreadPoolExecutor with Future.result(timeout=...)
    for per-sector cancellation. Does NOT use SIGALRM.

    Note: The thread may continue running after timeout since Python threads
    cannot be forcibly killed. However, the caller treats the sector as
    timed_out and moves on.

    Args:
        sector_key: The sector bucket key to screen.
        config: Sector scout config.
        timeout: Maximum wall-clock seconds for this sector.
        fh: Optional FinnhubClient instance. Created internally if not provided.
        core_watchlist: Optional core watchlist for exclusion. Defaults to empty.

    Returns:
        List of CandidateRow finalists from this sector.

    Raises:
        TimeoutError: If screening exceeds the per-sector budget.
    """
    from utils.sector_scout import (
        collect_candidate_data,
        apply_hard_gates,
        compute_scout_score,
        apply_score_penalties,
    )

    if core_watchlist is None:
        core_watchlist = []

    sector_buckets = config.get("sector_buckets", {})
    bucket = sector_buckets.get(sector_key, {})
    budget_ceilings = config.get("budget_ceilings", {})
    max_candidates_per_sector = int(budget_ceilings.get("max_candidates_per_sector", 20))

    symbols: list[str] = bucket.get("symbols", [])
    core_set = set(core_watchlist)

    # Exclude Core_Watchlist symbols unless core_re_ranking is enabled
    if not bucket.get("core_re_ranking", False):
        symbols = [s for s in symbols if s not in core_set]

    # Enforce max_candidates_per_sector ceiling
    if len(symbols) > max_candidates_per_sector:
        symbols = symbols[:max_candidates_per_sector]

    def _screen_sector():
        # Create FinnhubClient inside the thread if not provided
        client = fh if fh is not None else FinnhubClient()
        scored: list[CandidateRow] = []
        for symbol in symbols:
            row = collect_candidate_data(symbol, sector_key, config, client)
            passed, reason_code = apply_hard_gates(row, config)
            if not passed:
                continue
            row = compute_scout_score(row, config)
            row = apply_score_penalties(row, config)
            scored.append(row)
        return scored

    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix=f"sector_{sector_key}"
    ) as executor:
        future = executor.submit(_screen_sector)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout:
            future.cancel()
            raise TimeoutError(
                f"Sector '{sector_key}' screening exceeded budget of {timeout:.1f}s"
            )


def run_funnel_discovery(
    engine,
    config: dict,
    funnel_config: dict,
) -> DiscoveryResult:
    """Execute bounded premarket discovery and persist candidates.

    Implements the full discovery pipeline:
    1. Per-sector deterministic screening with budget enforcement
    2. Chief Scout curation OR deterministic fallback
    3. Rank and persist top candidates

    Enforces per_sector_budget (default 15s) per sector using
    concurrent.futures timeout, and total_pipeline_budget (default 90s)
    across all sectors. When total budget is exhausted before all sectors
    complete, remaining sectors go to sectors_skipped.

    Args:
        engine: SQLAlchemy engine.
        config: Sector scout configuration dict.
        funnel_config: Funnel-specific config (budgets, ceilings).

    Returns:
        DiscoveryResult with candidates and operational metadata.

    Postconditions:
        - sectors_completed + sectors_timed_out + sectors_skipped covers all
          enabled sectors exactly once.
        - total_duration_seconds reflects actual wall-clock time.
        - pipeline_budget_exhausted is True iff any sectors were skipped.
        - partial_screening is True iff any sectors timed out or were skipped.
    """
    funnel = funnel_config.get("funnel", funnel_config)
    budgets = funnel.get("budgets", {})
    ceilings = funnel.get("ceilings", {})
    per_sector_budget: float = float(budgets.get("per_sector_seconds", 15))
    total_budget: float = float(budgets.get("total_pipeline_seconds", 90))
    max_shortlist: int = int(ceilings.get("max_discovery_shortlist", 5))

    pipeline_start = time.monotonic()
    all_finalists: list[CandidateRow] = []
    sectors_completed: list[str] = []
    sectors_timed_out: list[str] = []
    sectors_skipped: list[str] = []

    enabled_sectors = get_enabled_sectors(config)
    core_watchlist: list[str] = config.get("core_watchlist", [])

    # Create FinnhubClient once, shared across sector screenings
    try:
        fh = FinnhubClient()
    except (ValueError, Exception) as exc:
        logger.error("Cannot create FinnhubClient: %s — proceeding with None", exc)
        fh = None

    logger.info(
        "Funnel discovery starting: %d enabled sectors, "
        "per_sector_budget=%.1fs, total_budget=%.1fs, max_shortlist=%d",
        len(enabled_sectors),
        per_sector_budget,
        total_budget,
        max_shortlist,
    )

    # Phase 1: Per-sector deterministic screening with budget enforcement
    for sector_key in enabled_sectors:
        elapsed = time.monotonic() - pipeline_start
        remaining = total_budget - elapsed

        # Skip remaining sectors when total budget exhausted
        if remaining <= 0:
            sectors_skipped.append(sector_key)
            logger.warning(
                "Funnel discovery: total budget exhausted (%.1fs elapsed), "
                "skipping sector '%s'",
                elapsed,
                sector_key,
            )
            continue

        # Per-sector budget is the minimum of configured budget and remaining total
        sector_budget = min(per_sector_budget, remaining)

        logger.info(
            "Funnel discovery: screening sector '%s' (budget=%.1fs, remaining=%.1fs)",
            sector_key,
            sector_budget,
            remaining,
        )

        try:
            sector_finalists = run_sector_with_timeout(
                sector_key,
                config,
                timeout=sector_budget,
                fh=fh,
                core_watchlist=core_watchlist,
            )
            all_finalists.extend(sector_finalists)
            sectors_completed.append(sector_key)
            logger.info(
                "Funnel discovery: sector '%s' completed — %d finalists",
                sector_key,
                len(sector_finalists),
            )
        except (TimeoutError, FuturesTimeout):
            sectors_timed_out.append(sector_key)
            logger.warning(
                "Funnel discovery: sector '%s' timed out after %.1fs budget",
                sector_key,
                sector_budget,
            )
            continue
        except Exception as exc:
            # Unexpected error in sector — treat as timed out to preserve
            # partial results from other sectors
            sectors_timed_out.append(sector_key)
            logger.error(
                "Funnel discovery: sector '%s' failed with unexpected error: %s",
                sector_key,
                exc,
            )
            continue

    # Phase 2: Chief Scout curation OR deterministic fallback
    elapsed = time.monotonic() - pipeline_start
    remaining = total_budget - elapsed

    curated_list, selection_mode, curation_error = run_chief_scout_curation(
        finalists=all_finalists,
        remaining_budget=remaining,
        config=config,
        engine=engine,
        core_watchlist=core_watchlist,
        max_shortlist=max_shortlist,
    )

    # Phase 3: Persist candidates
    candidates = persist_discovery_candidates(
        engine, curated_list, selection_mode, source_run="premarket",
        max_shortlist=max_shortlist,
    )

    total_duration = time.monotonic() - pipeline_start

    result = DiscoveryResult(
        candidates=candidates,
        sectors_completed=sectors_completed,
        sectors_timed_out=sectors_timed_out,
        sectors_skipped=sectors_skipped,
        selection_mode=selection_mode,
        partial_screening=bool(sectors_timed_out or sectors_skipped),
        total_duration_seconds=total_duration,
        pipeline_budget_exhausted=bool(sectors_skipped),
    )

    # Phase 4: Record FunnelRunLog for this discovery execution
    # Always create a log entry regardless of outcome (Requirement 1.8, 1.12, 7.6)
    started_at_utc = datetime.fromtimestamp(
        time.time() - total_duration, tz=timezone.utc
    )
    try:
        record_discovery_run_log(
            engine=engine,
            result=result,
            started_at=started_at_utc,
            budget_seconds=total_budget,
            error_message=curation_error,
        )
    except Exception as exc:
        # Log failure but don't fail the discovery — candidates are already persisted
        logger.error("Failed to record discovery FunnelRunLog: %s", exc)

    return result
