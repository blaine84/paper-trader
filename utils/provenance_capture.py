"""Provenance Event Capture.

Responsible for creating and persisting provenance events. Designed for
minimal latency via synchronous batched writes.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.6, 3.7, 3.8, 3.9, 3.10, 14.2, 14.3, 14.5, 14.6
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from utils.gate_config import PM_PROVENANCE_LATENCY_BUDGET_MS
from utils.geometry_calculator import GeometryResult, ValidationStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provenance Coverage Metrics (thread-safe)
# Tracks provenance capture success/failure rates per Requirement 14.3.
# On provenance failure, decrement coverage metric (i.e. increment failure count).
# ---------------------------------------------------------------------------

_coverage_lock = threading.Lock()
_provenance_attempts: int = 0
_provenance_failures: int = 0


def increment_provenance_attempt() -> None:
    """Record a provenance capture attempt."""
    global _provenance_attempts
    with _coverage_lock:
        _provenance_attempts += 1


def increment_provenance_failure() -> None:
    """Record a provenance capture failure (decrements effective coverage)."""
    global _provenance_failures
    with _coverage_lock:
        _provenance_failures += 1


def get_provenance_coverage_metrics() -> dict:
    """Get current provenance coverage metrics.

    Returns dict with attempts, failures, successes, success_rate_pct.
    Thread-safe snapshot.
    """
    with _coverage_lock:
        attempts = _provenance_attempts
        failures = _provenance_failures

    success_rate = ((attempts - failures) / attempts * 100) if attempts > 0 else 0.0
    return {
        "attempts": attempts,
        "failures": failures,
        "successes": attempts - failures,
        "success_rate_pct": round(success_rate, 2),
    }


def reset_provenance_coverage_metrics() -> None:
    """Reset coverage metrics (for testing)."""
    global _provenance_attempts, _provenance_failures
    with _coverage_lock:
        _provenance_attempts = 0
        _provenance_failures = 0


# Maximum payload size per event in bytes (1 MB)
_MAX_EVENT_PAYLOAD_BYTES = 1_048_576

# Credential keys to redact from input/output contracts
_CREDENTIAL_KEYS = frozenset({
    "api_key",
    "bearer_token",
    "authorization",
    "session_cookie",
    "access_token",
    "refresh_token",
    "private_key",
    "secret_key",
})

# Keys to always preserve (never redact)
_PRESERVED_KEYS = frozenset({
    "rationale",
    "reasoning",
    "setup_reasoning",
})

# Stage-to-attribution mapping
STAGE_TO_ATTRIBUTION: dict[str, str] = {
    "trusted_input": "trusted_input_invalid",
    "raw_pm_output": "raw_pm_output_invalid",
    "parsed_pm_decision": "parse_or_normalization_invalid",
    "candidate_resolution": "candidate_resolution_invalid",
    "price_repair": "price_repair_invalid",
    "behavioral_adjustment": "behavioral_adjustment_invalid",
    "pre_gate_snapshot": "pre_gate_contract_invalid",
    "gate_reconstruction": "gate_reconstruction_invalid",
}

# Terminal stage names
_TERMINAL_STAGES = frozenset({
    "parse_failure",
    "pm_rejection",
    "pm_not_selected",
    "candidate_mismatch",
})


@dataclass(frozen=True)
class ProvenanceEvent:
    """Immutable provenance event for one stage transformation."""

    lineage_id: str
    stage_name: str
    stage_version: str
    sequence_number: int              # monotonically increasing per lineage
    mutation_ordinal: int             # ordinal within same stage (for multiple mutations)
    timestamp: datetime               # UTC, millisecond precision
    input_contract: dict              # geometry fields before this stage
    output_contract: dict             # geometry fields after this stage
    fields_changed: list[str]         # Material_Mutation field names
    mutation_reason_code: str         # recognized code or "unattributed_mutation"
    rule_id: str | None               # configuration/rule that authorized mutation
    geometry_before: dict             # GeometryResult serialized
    geometry_after: dict              # GeometryResult serialized
    validation_before: str            # ValidationStatus value
    validation_after: str             # ValidationStatus value
    attempt_ordinal: int = 1          # for retry tracking
    payload_truncated: bool = False
    is_terminal: bool = False         # True for pm_rejection, parse_failure, etc.


@dataclass
class ProvenanceChain:
    """Accumulator for a single candidate's provenance chain within a PM cycle."""

    lineage_id: str
    pm_mode: str                      # "candidate_id" or "legacy_free_form"
    events: list[ProvenanceEvent] = field(default_factory=list)
    _next_seq: int = 1

    @property
    def expected_stages(self) -> list[str]:
        """Return mode-specific expected stage set."""
        if self.pm_mode == "candidate_id":
            return [
                "trusted_input", "raw_pm_output", "parsed_pm_decision",
                "candidate_resolution", "behavioral_adjustment", "pre_gate_snapshot",
            ]
        return [
            "trusted_input", "raw_pm_output", "parsed_pm_decision",
            "price_repair", "behavioral_adjustment", "pre_gate_snapshot",
        ]

    def record_event(
        self,
        stage_name: str,
        stage_version: str,
        input_contract: dict,
        output_contract: dict,
        fields_changed: list[str],
        mutation_reason_code: str,
        rule_id: str | None,
        geometry_before: GeometryResult,
        geometry_after: GeometryResult,
        mutation_ordinal: int = 1,
        attempt_ordinal: int = 1,
        is_terminal: bool = False,
    ) -> ProvenanceEvent:
        """Record a new event, auto-assigning sequence number and timestamp."""
        seq = self._next_seq
        self._next_seq += 1

        validation_before = geometry_before.validation_status.value
        validation_after = geometry_after.validation_status.value

        event = ProvenanceEvent(
            lineage_id=self.lineage_id,
            stage_name=stage_name,
            stage_version=stage_version,
            sequence_number=seq,
            mutation_ordinal=mutation_ordinal,
            timestamp=datetime.now(timezone.utc),
            input_contract=input_contract,
            output_contract=output_contract,
            fields_changed=fields_changed,
            mutation_reason_code=mutation_reason_code,
            rule_id=rule_id,
            geometry_before=_serialize_geometry(geometry_before),
            geometry_after=_serialize_geometry(geometry_after),
            validation_before=validation_before,
            validation_after=validation_after,
            attempt_ordinal=attempt_ordinal,
            is_terminal=is_terminal,
        )
        self.events.append(event)
        return event

    def record_passthrough(
        self, stage_name: str, stage_version: str, validation_status: str,
    ) -> ProvenanceEvent:
        """Record a compact pass-through event (no material mutation)."""
        seq = self._next_seq
        self._next_seq += 1

        event = ProvenanceEvent(
            lineage_id=self.lineage_id,
            stage_name=stage_name,
            stage_version=stage_version,
            sequence_number=seq,
            mutation_ordinal=1,
            timestamp=datetime.now(timezone.utc),
            input_contract={},
            output_contract={},
            fields_changed=[],
            mutation_reason_code="passthrough",
            rule_id=None,
            geometry_before={},
            geometry_after={},
            validation_before=validation_status,
            validation_after=validation_status,
            attempt_ordinal=1,
            is_terminal=False,
        )
        self.events.append(event)
        return event

    def record_terminal(
        self, stage_name: str, stage_version: str, reason: str,
    ) -> ProvenanceEvent:
        """Record a terminal event (parse_failure, pm_rejection, pm_not_selected, candidate_mismatch).

        Terminal events set is_terminal=True and validation_after=NOT_APPLICABLE.
        Subsequent stages are not expected and won't trigger incomplete_provenance.
        """
        seq = self._next_seq
        self._next_seq += 1

        event = ProvenanceEvent(
            lineage_id=self.lineage_id,
            stage_name=stage_name,
            stage_version=stage_version,
            sequence_number=seq,
            mutation_ordinal=1,
            timestamp=datetime.now(timezone.utc),
            input_contract={},
            output_contract={},
            fields_changed=[],
            mutation_reason_code=reason,
            rule_id=None,
            geometry_before={},
            geometry_after={},
            validation_before="not_applicable",
            validation_after="not_applicable",
            attempt_ordinal=1,
            is_terminal=True,
        )
        self.events.append(event)
        return event

    def get_first_invalid_stage(self) -> str | None:
        """Traverse chain and return the first stage with validation_after == 'invalid'."""
        sorted_events = sorted(self.events, key=lambda e: e.sequence_number)
        for event in sorted_events:
            if event.validation_after == ValidationStatus.INVALID.value:
                return event.stage_name
        return None

    def get_missing_stages(self) -> list[str]:
        """Return expected stages not present in chain (excluding terminal/not_applicable).

        If a terminal event exists, all expected stages AFTER the terminal's
        position in the expected stage list are not considered missing.
        Stages that have events with validation_after == "not_applicable" are
        also excluded from the missing list.
        """
        recorded_stages: set[str] = set()
        terminal_stage_name: str | None = None
        not_applicable_stages: set[str] = set()

        for event in self.events:
            recorded_stages.add(event.stage_name)
            if event.is_terminal:
                terminal_stage_name = event.stage_name
            if event.validation_after == ValidationStatus.NOT_APPLICABLE.value:
                not_applicable_stages.add(event.stage_name)

        expected = self.expected_stages

        # If a terminal event exists, determine the cut-off point
        # Terminal stage names may not be in the expected list (e.g. "parse_failure"),
        # but they indicate no further stages should be expected.
        # We find the position in the chain where the terminal occurred and
        # exclude all expected stages that come after the last recorded non-terminal stage.
        if terminal_stage_name is not None:
            # Find the latest recorded expected stage before/at the terminal
            # The terminal event's sequence_number tells us the cut-off
            terminal_seq = None
            for event in self.events:
                if event.is_terminal:
                    terminal_seq = event.sequence_number
                    break

            # Find which expected stages were recorded before the terminal
            pre_terminal_recorded: set[str] = set()
            for event in self.events:
                if event.sequence_number < terminal_seq:
                    pre_terminal_recorded.add(event.stage_name)

            # Determine the last expected stage index that was recorded pre-terminal
            last_expected_idx = -1
            for idx, stage in enumerate(expected):
                if stage in pre_terminal_recorded:
                    last_expected_idx = idx

            # Only stages up to and including last_expected_idx+1 are expected
            # (the next one after what was recorded could be the terminal point)
            # Actually: stages after the terminal are NOT expected
            # We consider only stages up to the terminal's logical position
            if terminal_stage_name in expected:
                # Terminal stage IS in expected list - include stages up to it
                terminal_idx = expected.index(terminal_stage_name)
                expected = expected[:terminal_idx + 1]
            else:
                # Terminal stage is NOT in expected list (e.g. "parse_failure")
                # Only stages before the terminal event are expected
                expected = expected[:last_expected_idx + 1]

        missing = []
        for stage in expected:
            if stage not in recorded_stages and stage not in not_applicable_stages:
                missing.append(stage)

        return missing

    def get_attribution_category(self) -> str:
        """Classify the first-invalid-stage into an attribution category.

        Uses STAGE_TO_ATTRIBUTION mapping. If no invalid stage found and
        missing stages exist: "incomplete_provenance". If all valid:
        "policy_rejection_of_valid_contract". Fallback: "unknown".
        """
        first_invalid = self.get_first_invalid_stage()

        if first_invalid is not None:
            return STAGE_TO_ATTRIBUTION.get(first_invalid, "unknown")

        missing = self.get_missing_stages()
        if missing:
            return "incomplete_provenance"

        # All stages present and valid
        return "policy_rejection_of_valid_contract"


def _serialize_geometry(geometry: GeometryResult) -> dict:
    """Serialize a GeometryResult to a JSON-compatible dict."""
    return {
        "direction": geometry.direction,
        "entry_price": str(geometry.entry_price),
        "stop_price": str(geometry.stop_price),
        "target_price": str(geometry.target_price),
        "quantity": str(geometry.quantity),
        "risk_distance": str(geometry.risk_distance),
        "reward_distance": str(geometry.reward_distance),
        "reward_to_risk": str(geometry.reward_to_risk),
        "per_unit_risk": str(geometry.per_unit_risk),
        "total_dollar_risk": str(geometry.total_dollar_risk),
        "stop_direction_valid": geometry.stop_direction_valid,
        "target_direction_valid": geometry.target_direction_valid,
        "is_valid": geometry.is_valid,
        "validation_status": geometry.validation_status.value,
    }


def _redact_credentials(data: dict) -> dict:
    """Recursively redact credential keys from a dict, preserving rationale/reasoning."""
    if not isinstance(data, dict):
        return data

    redacted = {}
    for key, value in data.items():
        key_lower = key.lower()
        # Preserve rationale/reasoning fields always
        if key_lower in _PRESERVED_KEYS:
            redacted[key] = value
        elif key_lower in _CREDENTIAL_KEYS:
            redacted[key] = "[REDACTED]"
        elif isinstance(value, dict):
            redacted[key] = _redact_credentials(value)
        elif isinstance(value, list):
            redacted[key] = [
                _redact_credentials(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            redacted[key] = value

    return redacted


class _DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""

    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def persist_provenance_chain(engine, chain: ProvenanceChain) -> None:
    """Persist the complete provenance chain in a single batched INSERT.

    Synchronous — executes within the same transaction boundary as the PM cycle.
    Measures and logs elapsed time; warns if exceeds PM_PROVENANCE_LATENCY_BUDGET_MS.

    Enforces:
    - Max 1 MB per event payload (truncate non-geometry fields first)
    - Redact credentials (API keys, bearer tokens) but PRESERVE user-visible
      rationale, reasoning, and setup_reasoning fields
    - UNIQUE(lineage_id, sequence_number) prevents duplicate persistence

    Failure isolation (Requirements 14.2, 14.3):
    - On ANY exception: log error with lineage_id and stage names,
      increment provenance failure metric, do NOT re-raise.
    - Gate pipeline continues unchanged — provenance is observational,
      never blocks trade decisions.
    - Structural geometry validation (compute_geometry is_valid=False) is
      gate logic, not provenance logic, and always blocks independently.
    """
    if not chain.events:
        return

    increment_provenance_attempt()
    start_time = time.perf_counter()

    try:
        insert_sql = text("""
            INSERT INTO provenance_events (
                lineage_id, stage_name, stage_version, sequence_number,
                mutation_ordinal, timestamp, input_contract_json,
                output_contract_json, fields_changed_json,
                mutation_reason_code, rule_id, geometry_before_json,
                geometry_after_json, validation_before, validation_after,
                attempt_ordinal, is_terminal, payload_truncated
            ) VALUES (
                :lineage_id, :stage_name, :stage_version, :sequence_number,
                :mutation_ordinal, :timestamp, :input_contract_json,
                :output_contract_json, :fields_changed_json,
                :mutation_reason_code, :rule_id, :geometry_before_json,
                :geometry_after_json, :validation_before, :validation_after,
                :attempt_ordinal, :is_terminal, :payload_truncated
            )
        """)

        with engine.begin() as conn:
            for event in chain.events:
                # Redact credentials from contracts
                redacted_input = _redact_credentials(event.input_contract)
                redacted_output = _redact_credentials(event.output_contract)

                # Serialize to JSON
                input_json = json.dumps(redacted_input, cls=_DecimalEncoder)
                output_json = json.dumps(redacted_output, cls=_DecimalEncoder)
                fields_changed_json = json.dumps(event.fields_changed)
                geometry_before_json = json.dumps(event.geometry_before, cls=_DecimalEncoder)
                geometry_after_json = json.dumps(event.geometry_after, cls=_DecimalEncoder)

                # Check total payload size and truncate if needed
                payload_truncated = event.payload_truncated
                total_size = (
                    len(input_json.encode("utf-8"))
                    + len(output_json.encode("utf-8"))
                    + len(fields_changed_json.encode("utf-8"))
                    + len(geometry_before_json.encode("utf-8"))
                    + len(geometry_after_json.encode("utf-8"))
                )

                if total_size > _MAX_EVENT_PAYLOAD_BYTES:
                    # Truncate non-geometry fields first (input/output contracts)
                    input_json = "{}"
                    output_json = "{}"
                    payload_truncated = True

                    # Recheck after truncation
                    total_size = (
                        len(input_json.encode("utf-8"))
                        + len(output_json.encode("utf-8"))
                        + len(fields_changed_json.encode("utf-8"))
                        + len(geometry_before_json.encode("utf-8"))
                        + len(geometry_after_json.encode("utf-8"))
                    )

                    if total_size > _MAX_EVENT_PAYLOAD_BYTES:
                        # Further truncate fields_changed
                        fields_changed_json = "[]"

                params = {
                    "lineage_id": event.lineage_id,
                    "stage_name": event.stage_name,
                    "stage_version": event.stage_version,
                    "sequence_number": event.sequence_number,
                    "mutation_ordinal": event.mutation_ordinal,
                    "timestamp": event.timestamp.isoformat(),
                    "input_contract_json": input_json,
                    "output_contract_json": output_json,
                    "fields_changed_json": fields_changed_json,
                    "mutation_reason_code": event.mutation_reason_code,
                    "rule_id": event.rule_id,
                    "geometry_before_json": geometry_before_json,
                    "geometry_after_json": geometry_after_json,
                    "validation_before": event.validation_before,
                    "validation_after": event.validation_after,
                    "attempt_ordinal": event.attempt_ordinal,
                    "is_terminal": 1 if event.is_terminal else 0,
                    "payload_truncated": 1 if payload_truncated else 0,
                }

                try:
                    conn.execute(insert_sql, params)
                except IntegrityError:
                    # UNIQUE(lineage_id, sequence_number) violation — skip duplicate
                    logger.warning(
                        "Duplicate provenance event skipped: lineage_id=%s, sequence_number=%d",
                        event.lineage_id,
                        event.sequence_number,
                    )

    except Exception as exc:
        # Fail-open: log error with identifying fields, decrement coverage, do NOT re-raise.
        # Gate pipeline continues unchanged (Requirement 14.2).
        increment_provenance_failure()
        logger.error(
            "Failed to persist provenance chain: lineage_id=%s, stages=%s, error=%s",
            chain.lineage_id,
            [e.stage_name for e in chain.events],
            str(exc),
        )
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms > PM_PROVENANCE_LATENCY_BUDGET_MS:
            logger.warning(
                "Provenance persistence exceeded latency budget: "
                "lineage_id=%s, elapsed_ms=%.1f, budget_ms=%d",
                chain.lineage_id,
                elapsed_ms,
                PM_PROVENANCE_LATENCY_BUDGET_MS,
            )
