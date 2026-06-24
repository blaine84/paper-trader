"""Checkpoint Logger — structured funnel event emission.

Emits structured events at each pipeline stage to expose where entry
opportunities advance or die. All emissions are fail-open: errors are
logged but never block the pipeline.

Requirements: 9.1, 9.2, 9.3, 9.4, 11.1, 11.2, 11.4
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import text

from utils.db_retry import with_lock_retry

logger = logging.getLogger(__name__)

VALID_STAGES = frozenset({
    "analyst_signal_seen",
    "candidate_registered",
    "candidate_offered_to_pm",
    "pm_candidate_accepted",
    "pm_candidate_rejected",
    "pm_contract_violation",
    "order_materialized",
    "gate_evaluated",
    "order_fired",
    "order_rejected",
})

OUTCOME_CATEGORIES = frozenset({
    "pm_rejected_candidate",
    "pm_accepted_and_executed",
    "pm_accepted_and_gate_rejected",
    "pm_contract_violation",
    "invalid_stale_candidate_id",
    "legacy_freeform_ignored",
    "deterministic_gate_rejected",
})


@dataclass(frozen=True)
class CheckpointEvent:
    """A single checkpoint funnel event."""

    stage: str
    cycle_id: str
    profile: str
    candidate_id: str | None = None
    lineage_id: str | None = None
    symbol: str | None = None
    setup_type: str | None = None
    decision: str | None = None
    reason_code: str | None = None
    timestamp: str | None = None  # ISO 8601 UTC; auto-set if None
    metadata: dict | None = None


class CheckpointLogger:
    """Fail-open checkpoint event emitter.

    All emissions are wrapped in try/except. On any error,
    logs at ERROR level and continues without raising.
    """

    def __init__(self, engine):
        self._engine = engine

    def emit(self, event: CheckpointEvent) -> None:
        """Emit a checkpoint event to the checkpoint_events table.

        Validates stage. Fail-open on any error.
        """
        try:
            if event.stage not in VALID_STAGES:
                logger.error(
                    "Invalid checkpoint stage: %s (valid: %s)",
                    event.stage,
                    sorted(VALID_STAGES),
                )
                return

            ts = event.timestamp or datetime.now(timezone.utc).isoformat()
            metadata_json = json.dumps(event.metadata) if event.metadata else None

            self._execute_emit_write(event, ts, metadata_json)
        except Exception:
            logger.error(
                "Failed to emit checkpoint event: stage=%s candidate_id=%s",
                event.stage,
                event.candidate_id,
                exc_info=True,
            )

    @with_lock_retry
    def _execute_emit_write(
        self, event: CheckpointEvent, ts: str, metadata_json: str | None
    ) -> None:
        """Execute the DB INSERT for a checkpoint event. Retried on lock contention."""
        with self._engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO checkpoint_events
                    (stage, cycle_id, candidate_id, lineage_id, profile, symbol,
                     setup_type, decision, reason_code, metadata_json, created_at)
                    VALUES (:stage, :cycle_id, :candidate_id, :lineage_id, :profile,
                            :symbol, :setup_type, :decision, :reason_code,
                            :metadata_json, :created_at)
                """),
                {
                    "stage": event.stage,
                    "cycle_id": event.cycle_id,
                    "candidate_id": event.candidate_id,
                    "lineage_id": event.lineage_id,
                    "profile": event.profile,
                    "symbol": event.symbol,
                    "setup_type": event.setup_type,
                    "decision": event.decision,
                    "reason_code": event.reason_code,
                    "metadata_json": metadata_json,
                    "created_at": ts,
                },
            )

    def emit_outcome(self, category: str, event: CheckpointEvent) -> None:
        """Emit a checkpoint event with outcome categorization.

        Validates both stage and outcome_category. Fail-open on any error.
        """
        try:
            if event.stage not in VALID_STAGES:
                logger.error(
                    "Invalid checkpoint stage: %s",
                    event.stage,
                )
                return
            if category not in OUTCOME_CATEGORIES:
                logger.error(
                    "Invalid outcome category: %s (valid: %s)",
                    category,
                    sorted(OUTCOME_CATEGORIES),
                )
                return

            ts = event.timestamp or datetime.now(timezone.utc).isoformat()
            metadata_json = json.dumps(event.metadata) if event.metadata else None

            with self._engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO checkpoint_events
                        (stage, outcome_category, cycle_id, candidate_id, lineage_id,
                         profile, symbol, setup_type, decision, reason_code,
                         metadata_json, created_at)
                        VALUES (:stage, :outcome_category, :cycle_id, :candidate_id,
                                :lineage_id, :profile, :symbol, :setup_type, :decision,
                                :reason_code, :metadata_json, :created_at)
                    """),
                    {
                        "stage": event.stage,
                        "outcome_category": category,
                        "cycle_id": event.cycle_id,
                        "candidate_id": event.candidate_id,
                        "lineage_id": event.lineage_id,
                        "profile": event.profile,
                        "symbol": event.symbol,
                        "setup_type": event.setup_type,
                        "decision": event.decision,
                        "reason_code": event.reason_code,
                        "metadata_json": metadata_json,
                        "created_at": ts,
                    },
                )
        except Exception:
            logger.error(
                "Failed to emit outcome checkpoint: category=%s stage=%s candidate_id=%s",
                category,
                event.stage,
                event.candidate_id,
                exc_info=True,
            )
