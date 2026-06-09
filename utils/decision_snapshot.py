"""Decision Snapshot — immutable decision-time records for full audit trail.

Stores the complete decision-time state as immutable records in
pm_candidate_events. Later observations (outcomes, shadow results) are
separate append-only records that link back to the original decision
but never mutate it.

Requirements: 20.1–20.7
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Event types that represent pre-trade facts (immutable at creation)
PRE_TRADE_EVENT_TYPES = frozenset({
    "decision_snapshot",      # Complete frozen state at decision time
    "offered",                # Candidate was offered to PM
    "pm_accept",              # PM accepted the candidate
    "pm_reject",              # PM rejected the candidate
    "pm_not_selected",        # PM did not mention the candidate
    "alignment_observation",  # Alignment policy evaluation result
})

# Event types that represent post-trade observations (append-only, linked)
POST_TRADE_EVENT_TYPES = frozenset({
    "pipeline_executed",         # Trade was successfully executed
    "pipeline_gate_rejected",    # Gate pipeline rejected
    "pipeline_sizing_rejected",  # Position sizer rejected
    "shadow_outcome",            # Shadow outcome scoring result
    "realized_outcome",          # Actual trade P&L outcome
    "recovery_released",         # Crash recovery action
})


def record_decision_snapshot(
    engine,
    candidate_id: str,
    cycle_id: str,
    profile_id: str,
    *,
    context_snapshot_json: str | None = None,
    benchmark_mapping_json: str | None = None,
    pm_decision: str | None = None,
    pm_rationale: str | None = None,
    pm_risk_multiplier: float | None = None,
    alignment_outcome: str | None = None,
    alignment_rule: str | None = None,
    alignment_measurements: dict | None = None,
) -> None:
    """Record an immutable decision-time snapshot.

    This creates a single comprehensive record of the decision-time state
    that NEVER gets mutated. Later observations link back via candidate_id.

    Requirements:
    - 20.1: Store candidate + benchmark context snapshot as immutable record
    - 20.2: Store PM decision + rationale linked to snapshot
    - 20.3: Store alignment result linked to snapshot
    - 20.4: Later observations are append-only linked records
    - 20.5: Never mutate original decision snapshot
    - 20.6: Distinguish pre-trade facts from post-trade observations
    - 20.7: No later inputs overwrite original fields

    Args:
        engine: SQLAlchemy engine.
        candidate_id: The candidate this snapshot is for.
        cycle_id: The PM cycle ID.
        profile_id: The profile ID.
        context_snapshot_json: Frozen context at decision time.
        benchmark_mapping_json: Benchmark mapping at decision time.
        pm_decision: "accept" | "reject" | "not_selected"
        pm_rationale: PM's stated rationale.
        pm_risk_multiplier: PM's requested risk multiplier (if any).
        alignment_outcome: Alignment policy outcome (if evaluated).
        alignment_rule: Rule that fired (if any).
        alignment_measurements: Measurements used (if any).
    """
    from sqlalchemy import text

    snapshot_data = {
        "record_type": "pre_trade_fact",
        "immutable": True,
        "context_snapshot": json.loads(context_snapshot_json) if context_snapshot_json else None,
        "benchmark_mapping": json.loads(benchmark_mapping_json) if benchmark_mapping_json else None,
        "pm_decision": pm_decision,
        "pm_rationale": pm_rationale,
        "pm_risk_multiplier": pm_risk_multiplier,
        "alignment_outcome": alignment_outcome,
        "alignment_rule": alignment_rule,
        "alignment_measurements": alignment_measurements,
        "snapshot_version": "1.0.0",
    }

    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO pm_candidate_events
                    (candidate_id, cycle_id, profile_id, event_type, event_data, created_at)
                    VALUES (:cid, :cycle_id, :profile_id, :event_type, :event_data, :created_at)
                """),
                {
                    "cid": candidate_id,
                    "cycle_id": cycle_id,
                    "profile_id": profile_id,
                    "event_type": "decision_snapshot",
                    "event_data": json.dumps(snapshot_data, default=str),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            conn.commit()
    except Exception as exc:
        logger.warning("Failed to record decision snapshot for %s: %s", candidate_id, exc)


def record_post_trade_observation(
    engine,
    candidate_id: str,
    cycle_id: str,
    profile_id: str,
    event_type: str,
    observation_data: dict,
) -> None:
    """Record a post-trade observation linked to the original decision.

    Post-trade observations are append-only — they never mutate the original
    decision_snapshot record. Each observation links to the original candidate
    via candidate_id (Requirement 20.4).

    Retrospective inferences (e.g., "sector weakness caused the loss") are
    stored with record_type "retrospective_inference" to distinguish them from
    facts present in the original snapshot (Requirement 20.6).

    Args:
        engine: SQLAlchemy engine.
        candidate_id: Links to the original decision snapshot.
        cycle_id: The PM cycle.
        profile_id: The profile.
        event_type: Must be a POST_TRADE_EVENT_TYPES value.
        observation_data: Dict with observation details.
    """
    from sqlalchemy import text

    if event_type not in POST_TRADE_EVENT_TYPES:
        logger.warning(
            "Attempted to record unknown post-trade event type '%s' for %s",
            event_type,
            candidate_id,
        )
        return

    data = {
        "record_type": "post_trade_observation",
        **observation_data,
    }

    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO pm_candidate_events
                    (candidate_id, cycle_id, profile_id, event_type, event_data, created_at)
                    VALUES (:cid, :cycle_id, :profile_id, :event_type, :event_data, :created_at)
                """),
                {
                    "cid": candidate_id,
                    "cycle_id": cycle_id,
                    "profile_id": profile_id,
                    "event_type": event_type,
                    "event_data": json.dumps(data, default=str),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            conn.commit()
    except Exception as exc:
        logger.warning("Failed to record post-trade observation for %s: %s", candidate_id, exc)


def is_pre_trade_fact(event_type: str) -> bool:
    """Distinguish pre-trade facts from post-trade observations (Requirement 20.5)."""
    return event_type in PRE_TRADE_EVENT_TYPES


def is_post_trade_observation(event_type: str) -> bool:
    """Distinguish post-trade observations from pre-trade facts (Requirement 20.5)."""
    return event_type in POST_TRADE_EVENT_TYPES
