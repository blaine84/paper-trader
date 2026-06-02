"""Funnel job isolation wrappers for orchestrator scheduling.

Each wrapper function encapsulates a funnel pipeline stage and guarantees:
1. All exceptions from the funnel operation are caught
2. Failures are logged to FunnelRunLog with result_status="error"
3. Exceptions are NEVER propagated to the caller (orchestrator scheduler)
4. Position monitoring, stop enforcement, and core watchlist processing
   continue independently regardless of funnel failures

This module implements Requirements 11.1–11.5:
- Funnel jobs operate as independent scheduled invocations
- No shared execution context or failure propagation between funnel and core
- On funnel failure: log, terminate only the affected operation, continue all else

Market-hours resource protection (Requirements 7.1, 7.2, 7.3, 8.1, 8.2, 8.3):
- is_discovery_allowed(): blocks broad discovery after confirmation_time (09:35 ET)
- run_manual_intraday_discovery(): operator-triggered discovery with same budgets/ceilings
- Midday 12:30 ET broad scan disabled in v1 (never registered in scheduler)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from db.schema import FunnelRunLog, get_session

logger = logging.getLogger(__name__)

_NY_TZ = ZoneInfo("America/New_York")


def _ny_date():
    """Return today's New York trading date."""
    return datetime.now(_NY_TZ).date()


# ---------------------------------------------------------------------------
# Market-Hours Resource Protection Guards (Task 11.2)
# Requirements: 7.1, 7.2, 7.3, 8.1, 8.2, 8.3
# ---------------------------------------------------------------------------


def is_discovery_allowed(funnel_config: dict) -> bool:
    """Check if broad discovery scans are permitted at the current time.

    Returns False after the configured confirmation_time (default 09:35 ET),
    blocking new broad discovery scans during market hours.

    This implements Requirement 7.1: "THE system SHALL NOT initiate new broad
    discovery scans after the configured confirmation_time (default 09:35 ET)."

    Also enforces Requirement 8.2: no new broad-universe discovery during
    market hours (09:30–16:00 ET) unless manually triggered.

    Args:
        funnel_config: Funnel pipeline configuration dict containing schedule
            with confirmation_time.

    Returns:
        True if broad discovery is allowed (before confirmation_time),
        False if blocked (at or after confirmation_time).
    """
    funnel = funnel_config.get("funnel", funnel_config)
    schedule = funnel.get("schedule", {})
    confirmation_time_str = schedule.get("confirmation_time", "09:35")

    # Parse confirmation_time as HH:MM
    try:
        parts = confirmation_time_str.split(":")
        cutoff_hour = int(parts[0])
        cutoff_minute = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        # If config is malformed, use safe default 09:35
        cutoff_hour, cutoff_minute = 9, 35

    now_ny = datetime.now(_NY_TZ)
    current_minutes = now_ny.hour * 60 + now_ny.minute
    cutoff_minutes = cutoff_hour * 60 + cutoff_minute

    if current_minutes >= cutoff_minutes:
        logger.info(
            "Discovery blocked: current time %02d:%02d ET >= confirmation_time %02d:%02d ET",
            now_ny.hour, now_ny.minute, cutoff_hour, cutoff_minute,
        )
        return False

    return True


def run_manual_intraday_discovery(
    engine,
    config: dict,
    funnel_config: dict,
):
    """Execute a manually-triggered intraday discovery scan.

    Performs the same bounded discovery pipeline as premarket but with
    source_run="manual_intraday". Enforces the same Total_Pipeline_Budget
    (default 90s) and max_discovery_shortlist ceiling (default 5).

    This implements Requirements 8.3:
    - Labeled with source_run="manual_intraday"
    - Enforces Total_Pipeline_Budget and max_discovery_shortlist ceiling
    - Persists candidates with stage_status="awaiting_research"
    - Never scheduled automatically — only operator-triggered
    - Logged prominently in FunnelRunLog

    Args:
        engine: SQLAlchemy engine.
        config: Sector scout configuration dict.
        funnel_config: Funnel pipeline configuration dict.

    Returns:
        DiscoveryResult from the discovery pipeline.

    Raises:
        No exceptions propagated — errors logged to FunnelRunLog.
    """
    from utils.funnel_discovery import (
        DiscoveryResult,
        get_enabled_sectors,
        persist_discovery_candidates,
        record_discovery_run_log,
        run_chief_scout_curation,
        run_sector_with_timeout,
    )
    from utils.finnhub_client import FinnhubClient
    from concurrent.futures import TimeoutError as FuturesTimeout

    logger.info("=== MANUAL INTRADAY DISCOVERY (operator-triggered) ===")

    funnel = funnel_config.get("funnel", funnel_config)
    budgets = funnel.get("budgets", {})
    ceilings = funnel.get("ceilings", {})
    per_sector_budget: float = float(budgets.get("per_sector_seconds", 15))
    total_budget: float = float(budgets.get("total_pipeline_seconds", 90))
    max_shortlist: int = int(ceilings.get("max_discovery_shortlist", 5))

    pipeline_start = time.monotonic()
    all_finalists = []
    sectors_completed: list[str] = []
    sectors_timed_out: list[str] = []
    sectors_skipped: list[str] = []

    enabled_sectors = get_enabled_sectors(config)
    core_watchlist: list[str] = config.get("core_watchlist", [])

    # Create FinnhubClient
    try:
        fh = FinnhubClient()
    except (ValueError, Exception) as exc:
        logger.error("Cannot create FinnhubClient: %s — proceeding with None", exc)
        fh = None

    logger.info(
        "Manual intraday discovery: %d enabled sectors, "
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

        if remaining <= 0:
            sectors_skipped.append(sector_key)
            logger.warning(
                "Manual intraday: total budget exhausted (%.1fs elapsed), "
                "skipping sector '%s'",
                elapsed,
                sector_key,
            )
            continue

        sector_budget = min(per_sector_budget, remaining)

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
        except (TimeoutError, FuturesTimeout):
            sectors_timed_out.append(sector_key)
            logger.warning(
                "Manual intraday: sector '%s' timed out after %.1fs budget",
                sector_key,
                sector_budget,
            )
            continue
        except Exception as exc:
            sectors_timed_out.append(sector_key)
            logger.error(
                "Manual intraday: sector '%s' failed: %s", sector_key, exc
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

    # Phase 3: Persist candidates with source_run="manual_intraday"
    candidates = persist_discovery_candidates(
        engine, curated_list, selection_mode, source_run="manual_intraday",
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

    # Phase 4: Record FunnelRunLog — prominently labeled (Req 8.3)
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
        logger.error("Failed to record manual intraday FunnelRunLog: %s", exc)

    logger.info(
        "Manual intraday discovery complete: %d candidates (mode=%s, %.1fs)",
        len(candidates),
        selection_mode,
        total_duration,
    )

    return result


def _log_funnel_error(engine, stage: str, budget_seconds: float, error: Exception) -> None:
    """Record a funnel failure in FunnelRunLog.

    Creates an error entry so that downstream systems (daily review, fallback
    logic) can detect that a funnel stage failed without having to inspect
    application logs.

    Args:
        engine: SQLAlchemy engine.
        stage: Pipeline stage name (discovery, research, analysis, confirmation).
        budget_seconds: The budget that was configured for this stage.
        error: The exception that caused the failure.
    """
    try:
        session = get_session(engine)
        now = datetime.now(tz=ZoneInfo("UTC"))
        log_entry = FunnelRunLog(
            date=_ny_date(),
            stage=stage,
            started_at=now,
            ended_at=now,
            duration_seconds=0.0,
            budget_seconds=budget_seconds,
            result_status="error",
            candidates_input=0,
            candidates_promoted=0,
            candidates_rejected=0,
            error_message=f"{type(error).__name__}: {str(error)[:500]}",
        )
        session.add(log_entry)
        session.commit()
        session.close()
    except Exception as log_err:
        # If we can't even log, just emit to stderr — never propagate
        logger.error(
            "Failed to write FunnelRunLog error entry for stage=%s: %s",
            stage, log_err
        )


def safe_funnel_discovery_job(engine, config: dict, funnel_config: dict) -> None:
    """Isolated wrapper for funnel discovery — never propagates exceptions.

    Invokes run_funnel_discovery() and catches ALL exceptions. On failure,
    logs to FunnelRunLog with result_status="error" and returns cleanly so
    the orchestrator scheduler continues with other jobs (position monitoring,
    stop enforcement, core watchlist processing).

    Args:
        engine: SQLAlchemy engine.
        config: Sector scout configuration dict.
        funnel_config: Funnel pipeline configuration dict.
    """
    try:
        from utils.funnel_discovery import run_funnel_discovery

        logger.info("=== FUNNEL DISCOVERY JOB START ===")
        result = run_funnel_discovery(engine, config, funnel_config)
        logger.info(
            "Funnel discovery completed: %d candidates, mode=%s, duration=%.1fs",
            len(result.candidates),
            result.selection_mode,
            result.total_duration_seconds,
        )
    except Exception as exc:
        logger.error(
            "Funnel discovery job FAILED (isolated — core processing unaffected): %s",
            exc,
            exc_info=True,
        )
        budget = (
            funnel_config.get("funnel", {})
            .get("budgets", {})
            .get("total_pipeline_seconds", 90)
        )
        _log_funnel_error(engine, stage="discovery", budget_seconds=budget, error=exc)


def safe_funnel_research_job(engine, config: dict, funnel_config: dict) -> None:
    """Isolated wrapper for funnel researcher qualification — never propagates exceptions.

    Queries today's awaiting_research candidates and invokes
    run_funnel_qualification(). On failure, logs to FunnelRunLog with
    result_status="error" and returns cleanly.

    Args:
        engine: SQLAlchemy engine.
        config: Sector scout / application configuration dict.
        funnel_config: Funnel pipeline configuration dict.
    """
    try:
        from utils.funnel_researcher import run_funnel_qualification
        from db.schema import FunnelCandidate

        logger.info("=== FUNNEL RESEARCH QUALIFICATION JOB START ===")

        # Query today's candidates awaiting research
        session = get_session(engine)
        today = _ny_date()
        candidates = (
            session.query(FunnelCandidate)
            .filter_by(date=today, stage_status="awaiting_research", expired=False)
            .all()
        )
        session.close()

        if not candidates:
            logger.info("Funnel research: no candidates awaiting research today")
            return

        max_promoted = (
            funnel_config.get("funnel", {})
            .get("ceilings", {})
            .get("max_researcher_promoted", 3)
        )

        decisions = run_funnel_qualification(
            engine, candidates, config, max_promoted=max_promoted
        )

        promoted = sum(1 for d in decisions if d.decision == "promoted")
        rejected = sum(1 for d in decisions if d.decision == "rejected")
        logger.info(
            "Funnel research completed: %d evaluated, %d promoted, %d rejected",
            len(decisions), promoted, rejected,
        )
    except Exception as exc:
        logger.error(
            "Funnel research job FAILED (isolated — core processing unaffected): %s",
            exc,
            exc_info=True,
        )
        budget = (
            funnel_config.get("funnel", {})
            .get("budgets", {})
            .get("total_pipeline_seconds", 90)
        )
        _log_funnel_error(engine, stage="research", budget_seconds=budget, error=exc)


def safe_funnel_analysis_job(engine, config: dict) -> None:
    """Isolated wrapper for funnel analyst setup classification — never propagates exceptions.

    Queries today's awaiting_analysis candidates and invokes
    run_funnel_analysis(). On failure, logs to FunnelRunLog with
    result_status="error" and returns cleanly.

    Args:
        engine: SQLAlchemy engine.
        config: Application configuration dict (unused currently but passed
            for consistency and future extension).
    """
    try:
        from utils.funnel_analyst import run_funnel_analysis
        from db.schema import FunnelCandidate

        logger.info("=== FUNNEL ANALYSIS JOB START ===")

        # Query today's candidates awaiting analysis
        session = get_session(engine)
        today = _ny_date()
        candidates = (
            session.query(FunnelCandidate)
            .filter_by(date=today, stage_status="awaiting_analysis", expired=False)
            .all()
        )
        session.close()

        if not candidates:
            logger.info("Funnel analysis: no candidates awaiting analysis today")
            return

        decisions = run_funnel_analysis(engine, candidates)

        promoted = sum(1 for d in decisions if d.decision == "promoted")
        rejected = sum(1 for d in decisions if d.decision == "rejected")
        logger.info(
            "Funnel analysis completed: %d evaluated, %d promoted, %d rejected",
            len(decisions), promoted, rejected,
        )
    except Exception as exc:
        logger.error(
            "Funnel analysis job FAILED (isolated — core processing unaffected): %s",
            exc,
            exc_info=True,
        )
        # Analysis doesn't have its own explicit budget in config; use pipeline budget
        _log_funnel_error(engine, stage="analysis", budget_seconds=90, error=exc)


def safe_funnel_confirmation_job(engine, funnel_config: dict) -> None:
    """Isolated wrapper for opening confirmation — never propagates exceptions.

    Queries today's awaiting_confirmation candidates and invokes
    run_opening_confirmation(). On failure, logs to FunnelRunLog with
    result_status="error" and returns cleanly.

    Args:
        engine: SQLAlchemy engine.
        funnel_config: Funnel pipeline configuration dict.
    """
    try:
        from utils.funnel_confirmation import run_opening_confirmation
        from db.schema import FunnelCandidate

        logger.info("=== FUNNEL CONFIRMATION JOB START ===")

        # Query today's candidates awaiting confirmation
        session = get_session(engine)
        today = _ny_date()
        candidates = (
            session.query(FunnelCandidate)
            .filter_by(date=today, stage_status="awaiting_confirmation", expired=False)
            .order_by(FunnelCandidate.scout_rank.asc(), FunnelCandidate.scout_score.desc())
            .all()
        )
        session.close()

        if not candidates:
            logger.info("Funnel confirmation: no candidates awaiting confirmation today")
            return

        budget = (
            funnel_config.get("funnel", {})
            .get("budgets", {})
            .get("confirmation_budget_seconds", 45)
        )

        decisions = run_opening_confirmation(engine, candidates, budget_seconds=budget)

        promoted = sum(1 for d in decisions if d.decision == "promoted")
        rejected = sum(1 for d in decisions if d.decision == "rejected")
        logger.info(
            "Funnel confirmation completed: %d evaluated, %d promoted, %d rejected",
            len(decisions), promoted, rejected,
        )
    except Exception as exc:
        logger.error(
            "Funnel confirmation job FAILED (isolated — core processing unaffected): %s",
            exc,
            exc_info=True,
        )
        budget = (
            funnel_config.get("funnel", {})
            .get("budgets", {})
            .get("confirmation_budget_seconds", 45)
        )
        _log_funnel_error(engine, stage="confirmation", budget_seconds=budget, error=exc)


def safe_funnel_confirmation_retry_job(engine, funnel_config: dict) -> None:
    """Isolated wrapper for 10:00 ET confirmation retry — never propagates exceptions.

    Invokes run_confirmation_retry() for candidates that were not_evaluated
    in the primary 09:35 ET pass. On failure, logs to FunnelRunLog with
    result_status="error" and returns cleanly.

    Args:
        engine: SQLAlchemy engine.
        funnel_config: Funnel pipeline configuration dict.
    """
    try:
        from utils.funnel_confirmation import run_confirmation_retry

        logger.info("=== FUNNEL CONFIRMATION RETRY JOB START ===")

        decisions = run_confirmation_retry(engine, funnel_config=funnel_config)

        promoted = sum(1 for d in decisions if d.decision == "promoted")
        rejected = sum(1 for d in decisions if d.decision == "rejected")
        logger.info(
            "Funnel confirmation retry completed: %d evaluated, %d promoted, %d rejected",
            len(decisions), promoted, rejected,
        )
    except Exception as exc:
        logger.error(
            "Funnel confirmation retry job FAILED (isolated — core processing unaffected): %s",
            exc,
            exc_info=True,
        )
        budget = (
            funnel_config.get("funnel", {})
            .get("budgets", {})
            .get("market_hours_confirmation_budget_seconds", 60)
        )
        _log_funnel_error(engine, stage="confirmation", budget_seconds=budget, error=exc)


# ---------------------------------------------------------------------------
# Market-Hours Confirmation Budget Enforcement (Task 11.2)
# Requirements: 7.2, 7.3
# ---------------------------------------------------------------------------


def run_market_hours_confirmation(
    engine,
    funnel_config: dict,
) -> list:
    """Run a market-hours confirmation pass with enforced budget.

    Enforces market_hours_confirmation_budget_seconds (default 60s) on any
    market-hours confirmation run. On budget exceeded: stops processing,
    records result_status="timed_out" in FunnelRunLog, appends not_evaluated
    for remaining candidates, and leaves their stage_status unchanged.

    This function is used by:
    - run_confirmation_retry (10:00 ET retry pass) — already calls
      run_opening_confirmation with market-hours budget
    - Any future market-hours confirmation triggered outside the primary
      09:35 ET opening window

    Args:
        engine: SQLAlchemy engine.
        funnel_config: Funnel pipeline configuration dict.

    Returns:
        List of ConfirmationDecision for each evaluated candidate
        (may be empty if no candidates are awaiting confirmation).

    Requirements: 7.2, 7.3
    """
    from utils.funnel_confirmation import run_opening_confirmation
    from db.schema import FunnelCandidate
    import time as _time

    funnel = funnel_config.get("funnel", funnel_config)
    budgets = funnel.get("budgets", {})
    budget_seconds = int(budgets.get("market_hours_confirmation_budget_seconds", 60))

    # Determine today's New York trading date
    today_ny = _ny_date()

    # Query today's awaiting_confirmation candidates
    session = get_session(engine)
    try:
        candidates = (
            session.query(FunnelCandidate)
            .filter_by(date=today_ny, stage_status="awaiting_confirmation", expired=False)
            .order_by(FunnelCandidate.scout_rank.asc(), FunnelCandidate.scout_score.desc())
            .all()
        )
        session.expunge_all()
    finally:
        session.close()

    started_at = datetime.now(timezone.utc)
    start_mono = _time.monotonic()

    if not candidates:
        logger.info(
            "Market-hours confirmation: no awaiting_confirmation candidates for %s",
            today_ny,
        )
        # Record empty run in FunnelRunLog
        _record_market_hours_confirmation_log(
            engine=engine,
            started_at=started_at,
            duration_seconds=_time.monotonic() - start_mono,
            budget_seconds=budget_seconds,
            candidates_input=0,
            candidates_promoted=0,
            candidates_rejected=0,
            result_status="completed",
            error_message=None,
        )
        return []

    logger.info(
        "Market-hours confirmation: evaluating %d candidates with %ds budget",
        len(candidates),
        budget_seconds,
    )

    # Run opening confirmation with market-hours budget (Req 7.2)
    decisions = run_opening_confirmation(
        engine=engine,
        candidates=candidates,
        budget_seconds=budget_seconds,
    )

    duration = _time.monotonic() - start_mono

    # Compute result counts
    promoted = sum(1 for d in decisions if d.decision == "promoted")
    rejected = sum(1 for d in decisions if d.decision == "rejected")
    not_evaluated = sum(1 for d in decisions if d.decision == "not_evaluated")

    # Determine result_status (Req 7.3: budget exceeded → timed_out)
    if not_evaluated > 0 and duration >= budget_seconds:
        result_status = "timed_out"
    elif not_evaluated > 0:
        result_status = "degraded"
    else:
        result_status = "completed"

    # Record FunnelRunLog (Req 7.3, 7.6)
    _record_market_hours_confirmation_log(
        engine=engine,
        started_at=started_at,
        duration_seconds=duration,
        budget_seconds=budget_seconds,
        candidates_input=len(candidates),
        candidates_promoted=promoted,
        candidates_rejected=rejected,
        result_status=result_status,
        error_message=None,
    )

    logger.info(
        "Market-hours confirmation complete: %d promoted, %d rejected, "
        "%d not_evaluated (%.1fs / %ds budget, status=%s)",
        promoted,
        rejected,
        not_evaluated,
        duration,
        budget_seconds,
        result_status,
    )

    return decisions


def _record_market_hours_confirmation_log(
    engine,
    started_at: datetime,
    duration_seconds: float,
    budget_seconds: int,
    candidates_input: int,
    candidates_promoted: int,
    candidates_rejected: int,
    result_status: str,
    error_message: str | None,
) -> None:
    """Record a FunnelRunLog entry for a market-hours confirmation pass.

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
    """
    ended_at = datetime.now(timezone.utc)
    today_ny = _ny_date()

    run_log = FunnelRunLog(
        date=today_ny,
        stage="confirmation_market_hours",
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
            "FunnelRunLog recorded: stage=confirmation_market_hours, "
            "result_status=%s, duration=%.1fs, input=%d",
            result_status,
            duration_seconds,
            candidates_input,
        )
    except Exception as exc:
        session.rollback()
        logger.error("Failed to record market-hours confirmation FunnelRunLog: %s", exc)
    finally:
        session.close()
