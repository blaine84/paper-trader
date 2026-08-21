"""Fast-path observe mode validation — shadow comparison and rollout criteria.

Provides shadow comparison metrics that compare fast-path outcomes against PM
decisions for the same session.  Used during the mandatory observe-mode rollout
phase (FAST_PATH_MODE="observe") to validate that the fast path produces
outcomes consistent with PM decisions before enabling execution delegation.

Fail mode: fail-open — comparison and metric errors are logged and partial
results returned.  Observation never blocks the pipeline.

See: .kiro/specs/fast-path-deterministic-execution/design.md
Requirements: 1.5, 1.11, 1.12, 10.8
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 11.1 — Shadow comparison metric
# (Requirements: 1.5, 1.11, 10.8)
# ---------------------------------------------------------------------------


def compute_shadow_comparison(
    engine: Any, profile_id: str, session_start: str, session_end: str
) -> dict:
    """Compare fast-path outcomes against PM decisions for the same session.

    Queries fast_path_events and pm_candidates within the given time window,
    matches them by symbol+direction+profile, and computes agreement metrics.

    Agreement logic:
      - fast-path ``missed_move`` agrees with PM ``execution_failed`` or
        PM rejection with reason containing "target" or "stale"
      - fast-path ``stand_down`` agrees with PM rejection (any reason)
      - fast-path ``trade_executed`` agrees with PM ``executed`` state
      - fast-path ``pending_order_created`` agrees with PM ``executed``
        or PM acceptance (candidate reached execution path)
      - fast-path ``watch_created``/``watch_promoted`` are informational —
        counted but not penalized for disagreement

    Timing advantage:
      - For each matched pair, computes seconds between fast-path evaluation
        and PM decision timestamp.  Positive = fast path acted earlier.

    Args:
        engine: SQLAlchemy engine for database access.
        profile_id: Profile to compare (e.g. "moderate", "conservative").
        session_start: ISO UTC timestamp string — start of the session window.
        session_end: ISO UTC timestamp string — end of the session window.

    Returns:
        Dict with:
          - total_triggers: number of fast-path events in the window
          - outcomes_by_type: dict of outcome_type -> count
          - agreement_count: number of outcomes that agree with PM decision
          - disagreement_count: number of outcomes that disagree with PM decision
          - unmatched_count: outcomes with no corresponding PM decision
          - agreement_rate: float (0.0-1.0) — agreement / (agreement + disagreement)
          - timing_advantage_avg_seconds: avg seconds fast path acted before PM
          - timing_deltas: list of timing deltas (seconds) for matched pairs
          - disagreements: list of dicts describing each disagreement
        Returns a zeroed-out dict on error — never raises.
    """
    empty_result = {
        "total_triggers": 0,
        "outcomes_by_type": {},
        "agreement_count": 0,
        "disagreement_count": 0,
        "unmatched_count": 0,
        "agreement_rate": 0.0,
        "timing_advantage_avg_seconds": 0.0,
        "timing_deltas": [],
        "disagreements": [],
    }

    try:
        with engine.connect() as conn:
            # Fetch fast-path events in the session window
            fp_rows = conn.execute(
                text(
                    """
                    SELECT event_id, symbol, direction, setup_type,
                           outcome_type, outcome_reason_code, evaluated_at,
                           source_signal_id, trigger_id
                    FROM fast_path_events
                    WHERE profile_id = :profile_id
                      AND evaluated_at >= :session_start
                      AND evaluated_at <= :session_end
                    ORDER BY evaluated_at ASC
                    """
                ),
                {
                    "profile_id": profile_id,
                    "session_start": session_start,
                    "session_end": session_end,
                },
            ).fetchall()

            # Fetch PM candidate decisions in the same window
            pm_rows = conn.execute(
                text(
                    """
                    SELECT candidate_id, symbol, direction, setup_type,
                           state, rejection_reason, created_at,
                           source_signal_id
                    FROM pm_candidates
                    WHERE profile_id = :profile_id
                      AND created_at >= :session_start
                      AND created_at <= :session_end
                    ORDER BY created_at ASC
                    """
                ),
                {
                    "profile_id": profile_id,
                    "session_start": session_start,
                    "session_end": session_end,
                },
            ).fetchall()

        # Build PM decision lookup: (symbol, direction) -> list of decisions
        pm_lookup: dict[tuple[str, str], list[dict]] = {}
        for row in pm_rows:
            key = (row[1], row[2])  # symbol, direction
            entry = {
                "candidate_id": row[0],
                "symbol": row[1],
                "direction": row[2],
                "setup_type": row[3],
                "state": row[4],
                "rejection_reason": row[5],
                "created_at": row[6],
                "source_signal_id": row[7],
            }
            pm_lookup.setdefault(key, []).append(entry)

        # Compare each fast-path event against PM decisions
        total_triggers = len(fp_rows)
        outcomes_by_type: dict[str, int] = {}
        agreement_count = 0
        disagreement_count = 0
        unmatched_count = 0
        timing_deltas: list[float] = []
        disagreements: list[dict] = []

        for fp_row in fp_rows:
            fp_event = {
                "event_id": fp_row[0],
                "symbol": fp_row[1],
                "direction": fp_row[2],
                "setup_type": fp_row[3],
                "outcome_type": fp_row[4],
                "outcome_reason_code": fp_row[5],
                "evaluated_at": fp_row[6],
                "source_signal_id": fp_row[7],
                "trigger_id": fp_row[8],
            }

            outcome_type = fp_event["outcome_type"]
            outcomes_by_type[outcome_type] = outcomes_by_type.get(outcome_type, 0) + 1

            # Find matching PM decision by symbol + direction
            key = (fp_event["symbol"], fp_event["direction"])
            pm_matches = pm_lookup.get(key, [])

            if not pm_matches:
                unmatched_count += 1
                continue

            # Use the closest PM decision by time (or same source_signal_id)
            pm_decision = _find_best_pm_match(fp_event, pm_matches)
            if pm_decision is None:
                unmatched_count += 1
                continue

            # Compute timing delta
            timing_delta = _compute_timing_delta(
                fp_event["evaluated_at"], pm_decision["created_at"]
            )
            if timing_delta is not None:
                timing_deltas.append(timing_delta)

            # Check agreement
            agrees = _check_agreement(outcome_type, pm_decision)
            if agrees:
                agreement_count += 1
            else:
                disagreement_count += 1
                disagreements.append({
                    "event_id": fp_event["event_id"],
                    "symbol": fp_event["symbol"],
                    "fp_outcome": outcome_type,
                    "fp_reason": fp_event["outcome_reason_code"],
                    "pm_state": pm_decision["state"],
                    "pm_rejection_reason": pm_decision["rejection_reason"],
                })

        # Compute agreement rate
        comparable = agreement_count + disagreement_count
        agreement_rate = (agreement_count / comparable) if comparable > 0 else 0.0

        # Compute average timing advantage
        timing_avg = (
            sum(timing_deltas) / len(timing_deltas) if timing_deltas else 0.0
        )

        result = {
            "total_triggers": total_triggers,
            "outcomes_by_type": outcomes_by_type,
            "agreement_count": agreement_count,
            "disagreement_count": disagreement_count,
            "unmatched_count": unmatched_count,
            "agreement_rate": round(agreement_rate, 4),
            "timing_advantage_avg_seconds": round(timing_avg, 2),
            "timing_deltas": timing_deltas,
            "disagreements": disagreements,
        }

        # Log summary
        logger.info(
            "FAST_PATH_SHADOW_COMPARISON: profile=%s total=%d "
            "agree=%d disagree=%d unmatched=%d rate=%.2f%% "
            "avg_timing_advantage=%.1fs",
            profile_id,
            total_triggers,
            agreement_count,
            disagreement_count,
            unmatched_count,
            agreement_rate * 100,
            timing_avg,
        )

        return result

    except Exception as exc:
        logger.error(
            "fast_path_observe: compute_shadow_comparison failed: %s", exc
        )
        return empty_result


# ---------------------------------------------------------------------------
# 11.3 — Rollout gate criteria (operational documentation)
# (Requirements: 1.11, 1.12)
# ---------------------------------------------------------------------------

# These criteria are operational requirements that must be satisfied before
# transitioning FAST_PATH_MODE from "observe" to "enabled".  They are checked
# by operators reviewing shadow comparison output — not enforced in code
# (the code-level gate is FAST_PATH_ENABLED_SETUP_TYPES starting narrow).
#
# ROLLOUT GATE CRITERIA:
#
#   1. MINIMUM OBSERVATION PERIOD
#      At least 1 full trading session (market open to close) with
#      FAST_PATH_MODE="observe" active and the fast-path monitor completing
#      ticks throughout.
#
#   2. SHADOW COMPARISON AGREEMENT RATE
#      agreement_rate >= 0.80 (80%) for `missed_move` and `stand_down`
#      outcomes specifically.  These are the conservative no-action outcomes
#      that must align with PM rejections.
#
#   3. NO FALSE EXECUTION OUTCOMES
#      Zero `trade_executed` outcomes from the fast path that correspond to
#      PM rejections for the same symbol+direction in the same session.
#      A false positive trade_executed would mean the fast path would have
#      traded something the PM explicitly rejected.
#
#   4. TICK BUDGET COMPLIANCE
#      The fast-path monitor must complete within the configured interval
#      budget (FAST_PATH_MONITOR_INTERVAL_SECONDS) on >= 95% of ticks
#      during the observation period.  Measured via evaluation_duration_ms
#      fields on fast_path_events and tick-skip warnings in logs.
#
#   5. CODE-LEVEL GATE (ALREADY ENFORCED)
#      FAST_PATH_ENABLED_SETUP_TYPES starts with a single type
#      ("momentum_fade") regardless of observe-mode results.  Rollout
#      expands one type at a time after per-type validation.
#
# PROCEDURE:
#   1. Set FAST_PATH_MODE="observe" in production environment
#   2. Run for at least one full market session
#   3. Call compute_shadow_comparison() for the session
#   4. Verify criteria 1-4 above are met
#   5. If criteria pass, set FAST_PATH_MODE="enabled"
#   6. Monitor first enabled session closely (FAST_PATH_ENABLED_SETUP_TYPES
#      limits blast radius to one setup type)
#   7. Expand FAST_PATH_ENABLED_SETUP_TYPES incrementally after per-type
#      validation

ROLLOUT_GATE_CRITERIA = {
    "min_observe_sessions": 1,
    "min_agreement_rate_conservative_outcomes": 0.80,
    "max_false_trade_executed": 0,
    "min_tick_budget_compliance_rate": 0.95,
}
"""Operational criteria thresholds — used for documentation and optional
programmatic validation.  These are the minimum values that must be met
before FAST_PATH_MODE can be transitioned from 'observe' to 'enabled'."""


def check_rollout_criteria(
    shadow_result: dict, tick_budget_compliance_rate: float
) -> dict:
    """Check whether observe-mode results meet rollout gate criteria.

    This is a convenience function for operators to validate shadow comparison
    results against documented criteria.  It does NOT enforce anything — the
    transition to enabled mode is a manual operator decision.

    Args:
        shadow_result: Output from compute_shadow_comparison().
        tick_budget_compliance_rate: Float (0.0-1.0) indicating the fraction
            of ticks that completed within the interval budget.

    Returns:
        Dict with:
          - ready: bool — True if all criteria pass
          - criteria_results: dict of criterion_name -> {passed: bool, value, threshold}
          - blocking_criteria: list of criterion names that failed
    """
    criteria_results: dict[str, dict] = {}
    blocking: list[str] = []

    # Criterion 1: minimum observation (caller must verify session count externally)
    total_triggers = shadow_result.get("total_triggers", 0)
    has_data = total_triggers > 0
    criteria_results["has_observation_data"] = {
        "passed": has_data,
        "value": total_triggers,
        "threshold": ">0 events",
    }
    if not has_data:
        blocking.append("has_observation_data")

    # Criterion 2: agreement rate for conservative outcomes
    agreement_rate = shadow_result.get("agreement_rate", 0.0)
    min_rate = ROLLOUT_GATE_CRITERIA["min_agreement_rate_conservative_outcomes"]
    rate_pass = agreement_rate >= min_rate
    criteria_results["agreement_rate"] = {
        "passed": rate_pass,
        "value": agreement_rate,
        "threshold": min_rate,
    }
    if not rate_pass:
        blocking.append("agreement_rate")

    # Criterion 3: no false trade_executed
    disagreements = shadow_result.get("disagreements", [])
    false_executions = [
        d for d in disagreements if d.get("fp_outcome") == "trade_executed"
    ]
    no_false_exec = len(false_executions) == 0
    criteria_results["no_false_trade_executed"] = {
        "passed": no_false_exec,
        "value": len(false_executions),
        "threshold": ROLLOUT_GATE_CRITERIA["max_false_trade_executed"],
    }
    if not no_false_exec:
        blocking.append("no_false_trade_executed")

    # Criterion 4: tick budget compliance
    min_compliance = ROLLOUT_GATE_CRITERIA["min_tick_budget_compliance_rate"]
    budget_pass = tick_budget_compliance_rate >= min_compliance
    criteria_results["tick_budget_compliance"] = {
        "passed": budget_pass,
        "value": tick_budget_compliance_rate,
        "threshold": min_compliance,
    }
    if not budget_pass:
        blocking.append("tick_budget_compliance")

    return {
        "ready": len(blocking) == 0,
        "criteria_results": criteria_results,
        "blocking_criteria": blocking,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_best_pm_match(fp_event: dict, pm_matches: list[dict]) -> dict | None:
    """Find the best matching PM decision for a fast-path event.

    Prefers matching by source_signal_id when available, then falls back to
    the closest PM decision by timestamp within the same session.
    """
    fp_signal_id = fp_event.get("source_signal_id")

    # Try exact match on source_signal_id first
    if fp_signal_id:
        for pm in pm_matches:
            if pm.get("source_signal_id") == fp_signal_id:
                return pm

    # Fall back to closest by time
    fp_time = _parse_iso(fp_event.get("evaluated_at", ""))
    if fp_time is None:
        return pm_matches[0] if pm_matches else None

    best = None
    best_delta = float("inf")
    for pm in pm_matches:
        pm_time = _parse_iso(pm.get("created_at", ""))
        if pm_time is None:
            continue
        delta = abs((fp_time - pm_time).total_seconds())
        if delta < best_delta:
            best_delta = delta
            best = pm

    return best


def _compute_timing_delta(fp_timestamp: str, pm_timestamp: str) -> float | None:
    """Compute timing delta in seconds between fast-path and PM decision.

    Returns positive values when fast path acted before PM (timing advantage).
    Returns None if either timestamp cannot be parsed.
    """
    fp_time = _parse_iso(fp_timestamp)
    pm_time = _parse_iso(pm_timestamp)

    if fp_time is None or pm_time is None:
        return None

    # Positive = PM was later (fast path has timing advantage)
    return (pm_time - fp_time).total_seconds()


def _check_agreement(fp_outcome_type: str, pm_decision: dict) -> bool:
    """Determine if a fast-path outcome agrees with a PM decision.

    Agreement rules:
      - missed_move agrees with PM rejection or execution_failed state
      - stand_down agrees with PM rejection (any terminal non-executed state)
      - trade_executed agrees with PM executed state
      - pending_order_created agrees with PM executed or reserved state
      - watch_created / watch_promoted are informational — always agree
    """
    pm_state = (pm_decision.get("state") or "").lower()
    pm_reason = (pm_decision.get("rejection_reason") or "").lower()

    # Terminal PM states that indicate rejection/failure
    pm_rejected = pm_state in (
        "rejected",
        "gate_rejected",
        "sizing_rejected",
        "execution_failed",
        "expired",
        "not_selected",
    )
    pm_executed = pm_state == "executed"

    if fp_outcome_type == "missed_move":
        # Agrees with PM rejection or execution failure
        return pm_rejected or "target" in pm_reason or "stale" in pm_reason

    elif fp_outcome_type == "stand_down":
        # Agrees with PM rejection
        return pm_rejected

    elif fp_outcome_type == "trade_executed":
        # Agrees only with PM execution
        return pm_executed

    elif fp_outcome_type == "pending_order_created":
        # Agrees with PM execution or reserved (candidate was accepted)
        return pm_executed or pm_state in ("executed", "reserved")

    elif fp_outcome_type in ("watch_created", "watch_promoted"):
        # Informational — always considered agreement
        return True

    # Unknown outcome type — conservative: disagree
    return False


def _parse_iso(timestamp_str: str) -> datetime | None:
    """Parse an ISO UTC timestamp string into a datetime.

    Handles common formats: with/without microseconds, with Z or +00:00 suffix.
    Returns None on parse failure.
    """
    if not timestamp_str:
        return None

    try:
        # Handle various ISO formats
        clean = timestamp_str.replace("Z", "+00:00")
        if "." in clean:
            # With fractional seconds
            return datetime.fromisoformat(clean)
        else:
            # Without fractional seconds — add microseconds
            if "+" in clean:
                parts = clean.split("+")
                return datetime.fromisoformat(parts[0] + ".000000+" + parts[1])
            return datetime.fromisoformat(clean + "+00:00")
    except (ValueError, TypeError):
        return None
