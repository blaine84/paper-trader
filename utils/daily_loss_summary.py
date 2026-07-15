"""
Daily Loss Summary — Aggregated candidate attrition for a single trading day.

Computes loss summary from pm_candidate_events telemetry for all cycles on a
trading day (not a single cycle_id). Queries persisted events only.

Fail-open: Returns partial summary with error_indication on query failure.
persist_daily_loss_summary() is also fail-open (try/except around INSERT).

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixed set for dominant_blocker_stage
# ---------------------------------------------------------------------------

_VALID_BLOCKER_STAGES = frozenset({
    "no_signals",
    "pm_rejection",
    "gate_rejection",
    "sizing_rejection",
    "execution_failure",
})


# ---------------------------------------------------------------------------
# DailyLossSummary dataclass
# ---------------------------------------------------------------------------

@dataclass
class DailyLossSummary:
    """Aggregated candidate attrition for a single trading day across all cycles."""

    trade_date: str
    profile_id: str
    signals_seen: int
    candidates_built: int
    preflight_failed: int
    offered_to_pm: int
    pm_rejected: int
    pm_rejected_by_reason: dict[str, int]  # reason_code → count
    pm_accepted: int
    gate_sizing_rejected: int
    execution_failed: int
    executed: int
    lifecycle_incomplete: int
    top_blocking_reasons: list[str]  # Top 3 by count, alpha tie-break
    dominant_blocker_stage: str | None  # Set when executed == 0
    error_indication: str | None = None  # Set on partial failure


# ---------------------------------------------------------------------------
# SQL queries
# ---------------------------------------------------------------------------

_COUNT_EVENTS_BY_TYPE = text("""
    SELECT event_type, COUNT(*) as cnt
    FROM pm_candidate_events
    WHERE DATE(created_at) = :trade_date
      AND profile_id = :profile_id
    GROUP BY event_type
""")

_COUNT_DISTINCT_CANDIDATES = text("""
    SELECT COUNT(DISTINCT candidate_id) as cnt
    FROM pm_candidate_events
    WHERE DATE(created_at) = :trade_date
      AND profile_id = :profile_id
""")

_COUNT_EXECUTED_CANDIDATES = text("""
    SELECT COUNT(DISTINCT candidate_id) as cnt
    FROM pm_candidate_events
    WHERE DATE(created_at) = :trade_date
      AND profile_id = :profile_id
      AND event_type IN ('lifecycle_incomplete', 'execution_success')
""")

_GET_PM_REJECT_EVENT_DATA = text("""
    SELECT event_data
    FROM pm_candidate_events
    WHERE DATE(created_at) = :trade_date
      AND profile_id = :profile_id
      AND event_type = 'pm_reject'
""")

_UPSERT_DAILY_LOSS_SUMMARY = text("""
    INSERT INTO daily_loss_summaries (
        trade_date, profile_id, signals_seen, candidates_built,
        preflight_failed, offered_to_pm, pm_rejected,
        pm_rejected_by_reason_json, pm_accepted, gate_sizing_rejected,
        execution_failed, executed, lifecycle_incomplete,
        top_blocking_reasons_json, dominant_blocker_stage,
        error_indication, created_at
    ) VALUES (
        :trade_date, :profile_id, :signals_seen, :candidates_built,
        :preflight_failed, :offered_to_pm, :pm_rejected,
        :pm_rejected_by_reason_json, :pm_accepted, :gate_sizing_rejected,
        :execution_failed, :executed, :lifecycle_incomplete,
        :top_blocking_reasons_json, :dominant_blocker_stage,
        :error_indication, :created_at
    )
    ON CONFLICT(trade_date, profile_id) DO UPDATE SET
        signals_seen = excluded.signals_seen,
        candidates_built = excluded.candidates_built,
        preflight_failed = excluded.preflight_failed,
        offered_to_pm = excluded.offered_to_pm,
        pm_rejected = excluded.pm_rejected,
        pm_rejected_by_reason_json = excluded.pm_rejected_by_reason_json,
        pm_accepted = excluded.pm_accepted,
        gate_sizing_rejected = excluded.gate_sizing_rejected,
        execution_failed = excluded.execution_failed,
        executed = excluded.executed,
        lifecycle_incomplete = excluded.lifecycle_incomplete,
        top_blocking_reasons_json = excluded.top_blocking_reasons_json,
        dominant_blocker_stage = excluded.dominant_blocker_stage,
        error_indication = excluded.error_indication,
        created_at = excluded.created_at
""")


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------

def _compute_top_blocking_reasons(
    pm_rejected_by_reason: dict[str, int],
) -> list[str]:
    """Compute top 3 blocking reasons by descending count, alphabetical tie-break.

    No padding if fewer than 3 distinct reason codes exist.

    Requirements: 8.2
    """
    if not pm_rejected_by_reason:
        return []

    # Sort by (-count, reason_code) for descending count then alphabetical
    sorted_reasons = sorted(
        pm_rejected_by_reason.items(),
        key=lambda item: (-item[1], item[0]),
    )

    return [reason for reason, _count in sorted_reasons[:3]]


def _compute_dominant_blocker_stage(
    candidates_built: int,
    pm_rejected: int,
    gate_sizing_rejected: int,
    execution_failed: int,
) -> str | None:
    """Determine dominant blocker stage when zero trades executed.

    Selects from fixed set: no_signals, pm_rejection, gate_rejection,
    sizing_rejection, execution_failure.

    When executed == 0, returns the stage that rejected the most candidates.
    If candidates_built == 0, returns "no_signals" (nothing entered the pipeline).

    Requirements: 8.3
    """
    if candidates_built == 0:
        return "no_signals"

    # Build stage → count mapping from available event counts
    # gate_sizing_rejected combines gate_fail + sizing_fail events
    stage_counts = {
        "pm_rejection": pm_rejected,
        "gate_rejection": gate_sizing_rejected,
        "sizing_rejection": 0,  # sizing is included in gate_sizing_rejected for now
        "execution_failure": execution_failed,
    }

    # Find the max count
    max_count = max(stage_counts.values())
    if max_count == 0:
        # No rejections recorded — likely all candidates stuck in pipeline
        return "pm_rejection"

    # Pick the stage with highest count; alphabetical tie-break
    candidates_stages = sorted(
        [(stage, count) for stage, count in stage_counts.items() if count == max_count],
        key=lambda x: x[0],
    )

    return candidates_stages[0][0]


def _parse_rejection_reasons(event_data_rows: list) -> dict[str, int]:
    """Parse pm_reject event_data JSON to extract rejection_reason_code counts.

    Requirements: 8.1
    """
    reason_counts: dict[str, int] = {}

    for row in event_data_rows:
        event_data_str = row[0] if row else None
        if not event_data_str:
            reason_counts["other"] = reason_counts.get("other", 0) + 1
            continue

        try:
            event_data = json.loads(event_data_str)
            reason_code = event_data.get("rejection_reason_code", "other")
            if not reason_code:
                reason_code = "other"
            reason_counts[reason_code] = reason_counts.get(reason_code, 0) + 1
        except (json.JSONDecodeError, TypeError, AttributeError):
            reason_counts["other"] = reason_counts.get("other", 0) + 1

    return reason_counts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_daily_loss_summary(
    engine,
    trade_date: str,
    profile_id: str,
) -> DailyLossSummary:
    """Compute loss summary from pm_candidate_events telemetry for all cycles on a trading day.

    Queries by trade_date (not a single cycle_id) to aggregate across all PM cycles
    that ran during the day. Queries persisted events only (not reconstructed from logs).

    Fail-open: returns partial summary with error indication on query failure.

    Args:
        engine: SQLAlchemy engine for database access.
        trade_date: ISO date string (YYYY-MM-DD) for the trading day.
        profile_id: The active trading profile identifier.

    Returns:
        DailyLossSummary with computed counts and derived fields.

    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
    """
    # Initialize default counts
    signals_seen = 0
    candidates_built = 0
    preflight_failed = 0
    offered_to_pm = 0
    pm_rejected = 0
    pm_accepted = 0
    gate_sizing_rejected = 0
    execution_failed = 0
    executed = 0
    lifecycle_incomplete = 0
    pm_rejected_by_reason: dict[str, int] = {}
    error_indication: str | None = None

    try:
        with engine.connect() as conn:
            params = {"trade_date": trade_date, "profile_id": profile_id}

            # Count distinct candidates for candidates_built
            row = conn.execute(_COUNT_DISTINCT_CANDIDATES, params).fetchone()
            candidates_built = row[0] if row else 0

            # Count events by type
            event_counts: dict[str, int] = {}
            rows = conn.execute(_COUNT_EVENTS_BY_TYPE, params).fetchall()
            for r in rows:
                event_counts[r[0]] = r[1]

            # Map event_type counts to summary fields
            signals_seen = event_counts.get("signal_generated", 0) + event_counts.get("candidate_registered", 0)
            # If no explicit signal events, derive from candidates_built
            if signals_seen == 0:
                signals_seen = candidates_built

            preflight_failed = event_counts.get("preflight_failed", 0)
            offered_to_pm = event_counts.get("preflight_passed", 0)
            pm_rejected = event_counts.get("pm_reject", 0)
            pm_accepted = event_counts.get("pm_accept", 0)
            gate_sizing_rejected = event_counts.get("gate_fail", 0) + event_counts.get("sizing_fail", 0)
            execution_failed = event_counts.get("execution_failed", 0)
            lifecycle_incomplete = event_counts.get("lifecycle_incomplete", 0)

            # Count executed candidates (those that reached execution success or have lifecycle events)
            exec_row = conn.execute(_COUNT_EXECUTED_CANDIDATES, params).fetchone()
            executed = exec_row[0] if exec_row else 0

            # If no lifecycle_incomplete or execution_success events, check for pm_accept minus failures
            if executed == 0 and pm_accepted > 0:
                # Executed = accepted minus those that failed at gates/sizing/execution
                executed = max(0, pm_accepted - gate_sizing_rejected - execution_failed)

            # Parse rejection reasons from pm_reject event_data
            reject_rows = conn.execute(_GET_PM_REJECT_EVENT_DATA, params).fetchall()
            pm_rejected_by_reason = _parse_rejection_reasons(reject_rows)

    except Exception as exc:
        error_indication = f"Query failure: {str(exc)[:512]}"
        logger.error(
            "Daily loss summary query failed for trade_date=%s profile_id=%s",
            trade_date,
            profile_id,
            exc_info=True,
        )

    # Compute derived fields
    top_blocking_reasons = _compute_top_blocking_reasons(pm_rejected_by_reason)

    dominant_blocker_stage: str | None = None
    if executed == 0:
        dominant_blocker_stage = _compute_dominant_blocker_stage(
            candidates_built=candidates_built,
            pm_rejected=pm_rejected,
            gate_sizing_rejected=gate_sizing_rejected,
            execution_failed=execution_failed,
        )

    return DailyLossSummary(
        trade_date=trade_date,
        profile_id=profile_id,
        signals_seen=signals_seen,
        candidates_built=candidates_built,
        preflight_failed=preflight_failed,
        offered_to_pm=offered_to_pm,
        pm_rejected=pm_rejected,
        pm_rejected_by_reason=pm_rejected_by_reason,
        pm_accepted=pm_accepted,
        gate_sizing_rejected=gate_sizing_rejected,
        execution_failed=execution_failed,
        executed=executed,
        lifecycle_incomplete=lifecycle_incomplete,
        top_blocking_reasons=top_blocking_reasons,
        dominant_blocker_stage=dominant_blocker_stage,
        error_indication=error_indication,
    )


# ---------------------------------------------------------------------------
# Accepted-but-blocked query
# ---------------------------------------------------------------------------

# Map event_type to blocking stage name
_EVENT_TYPE_TO_BLOCKING_STAGE: dict[str, str] = {
    "gate_fail": "gate_rejection",
    "sizing_fail": "sizing_rejection",
    "execution_failed": "execution_failure",
}

_ACCEPTED_BUT_BLOCKED_QUERY = text("""
    SELECT
        fail_events.candidate_id,
        fail_events.event_type,
        fail_events.created_at
    FROM pm_candidate_events fail_events
    INNER JOIN (
        SELECT DISTINCT candidate_id
        FROM pm_candidate_events
        WHERE DATE(created_at) = :trade_date
          AND profile_id = :profile_id
          AND event_type = 'pm_accept'
    ) accepted ON fail_events.candidate_id = accepted.candidate_id
    WHERE DATE(fail_events.created_at) = :trade_date
      AND fail_events.profile_id = :profile_id
      AND fail_events.event_type IN ('gate_fail', 'sizing_fail', 'execution_failed')
    ORDER BY fail_events.candidate_id, fail_events.created_at ASC
""")


def query_accepted_but_blocked(
    engine,
    trade_date: str,
    profile_id: str,
) -> dict[str, int]:
    """Query count of accepted-but-blocked candidates grouped by first_blocking_stage.

    An "accepted-but-blocked" candidate is one that received a pm_accept event
    but then failed at a subsequent pipeline stage (gate_fail, sizing_fail, or
    execution_failed) without reaching EXECUTED state.

    Args:
        engine: SQLAlchemy engine.
        trade_date: ISO date string (YYYY-MM-DD).
        profile_id: Active trading profile.

    Returns:
        Dict mapping first_blocking_stage to count. Empty dict on error.

    Requirements: 5.6
    """
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                _ACCEPTED_BUT_BLOCKED_QUERY,
                {"trade_date": trade_date, "profile_id": profile_id},
            ).fetchall()

        # For each candidate, pick the FIRST failure event (by created_at)
        # Rows are already ordered by candidate_id, created_at ASC
        first_blocking: dict[str, str] = {}  # candidate_id → blocking_stage
        for row in rows:
            candidate_id = row[0]
            event_type = row[1]
            if candidate_id not in first_blocking:
                stage = _EVENT_TYPE_TO_BLOCKING_STAGE.get(event_type)
                if stage:
                    first_blocking[candidate_id] = stage

        # Group by blocking stage and count
        stage_counts: dict[str, int] = {}
        for stage in first_blocking.values():
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

        return stage_counts

    except Exception:
        logger.error(
            "query_accepted_but_blocked failed for trade_date=%s profile_id=%s",
            trade_date,
            profile_id,
            exc_info=True,
        )
        return {}


def persist_daily_loss_summary(engine, summary: DailyLossSummary) -> bool:
    """Write or upsert DailyLossSummary to daily_loss_summaries table.

    Fail-open: on INSERT failure, logs at ERROR and returns False.
    Never raises.

    Args:
        engine: SQLAlchemy engine for database access.
        summary: The DailyLossSummary to persist.

    Returns:
        True if persisted successfully, False on failure.

    Requirements: 8.4, 8.5
    """
    try:
        with engine.connect() as conn:
            conn.execute(_UPSERT_DAILY_LOSS_SUMMARY, {
                "trade_date": summary.trade_date,
                "profile_id": summary.profile_id,
                "signals_seen": summary.signals_seen,
                "candidates_built": summary.candidates_built,
                "preflight_failed": summary.preflight_failed,
                "offered_to_pm": summary.offered_to_pm,
                "pm_rejected": summary.pm_rejected,
                "pm_rejected_by_reason_json": json.dumps(summary.pm_rejected_by_reason),
                "pm_accepted": summary.pm_accepted,
                "gate_sizing_rejected": summary.gate_sizing_rejected,
                "execution_failed": summary.execution_failed,
                "executed": summary.executed,
                "lifecycle_incomplete": summary.lifecycle_incomplete,
                "top_blocking_reasons_json": json.dumps(summary.top_blocking_reasons),
                "dominant_blocker_stage": summary.dominant_blocker_stage,
                "error_indication": summary.error_indication,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            conn.commit()

        logger.info(
            "Daily loss summary persisted for trade_date=%s profile_id=%s "
            "candidates_built=%d executed=%d dominant_blocker=%s",
            summary.trade_date,
            summary.profile_id,
            summary.candidates_built,
            summary.executed,
            summary.dominant_blocker_stage,
        )
        return True

    except Exception:
        logger.error(
            "Daily loss summary persist failed for trade_date=%s profile_id=%s",
            summary.trade_date,
            summary.profile_id,
            exc_info=True,
        )
        return False
