"""Telemetry Classifier — categorize PM output events for routing.

Classifies PM cycle events into distinct telemetry categories so that
dashboards accurately distinguish malformed output from legitimate trade
rejections. Malformed output and unknown candidate selections are NEVER
recorded as blocked_trade_candidates.

See: design.md §utils/telemetry_classifier.py
Requirements: 9.1, 9.2, 9.3, 9.4
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.candidate_pipeline import PipelineResult
    from utils.decision_contract import CandidateDecision


class TelemetryCategory(Enum):
    """Distinct telemetry categories for PM output events.

    Categories (Requirement 9.1):
    - MALFORMED_PM_OUTPUT: Invalid schema, missing required fields, unparseable response
    - UNKNOWN_CANDIDATE: PM selected a candidate_id not in the offered set
    - PM_REJECTION: Model explicitly chose to reject a valid candidate
    - GATE_REJECTION: Deterministic gate pipeline rejected the candidate
    - SIZING_REJECTION: Position sizer rejected the candidate
    - EXECUTED: Candidate successfully passed all checks and was executed
    - CONTRACT_VIOLATION: Other contract violations (extra fields, invalid values)
    """

    MALFORMED_PM_OUTPUT = "malformed_pm_output"
    UNKNOWN_CANDIDATE = "unknown_candidate_selection"
    PM_REJECTION = "pm_candidate_rejection"
    GATE_REJECTION = "gate_rejection"
    SIZING_REJECTION = "sizing_rejection"
    EXECUTED = "executed_candidate"
    CONTRACT_VIOLATION = "contract_violation"


def classify_pm_event(
    decision: CandidateDecision | None,
    parse_violation: dict | None,
    pipeline_result: PipelineResult | None,
) -> TelemetryCategory:
    """Classify a PM output event for telemetry routing.

    Priority order:
    1. If parse_violation is set and type is MALFORMED_RESPONSE → MALFORMED_PM_OUTPUT
    2. If parse_violation is set and type is UNKNOWN_CANDIDATE → UNKNOWN_CANDIDATE
    3. If parse_violation is set (any other type) → CONTRACT_VIOLATION
    4. If pipeline_result is set:
       - outcome=="executed" → EXECUTED
       - outcome=="gate_rejected" → GATE_REJECTION
       - outcome=="sizing_rejected" → SIZING_REJECTION
       - outcome=="reservation_failed" → UNKNOWN_CANDIDATE
       - outcome=="execution_failed" → GATE_REJECTION (treated same)
    5. If decision is set and decision.decision == "reject" → PM_REJECTION
    6. Default → CONTRACT_VIOLATION

    Args:
        decision: The parsed PM decision for one candidate (may be None).
        parse_violation: A violation dict from the decision contract parser (may be None).
            Expected to have a "type" key (e.g., "MALFORMED_RESPONSE", "UNKNOWN_CANDIDATE").
        pipeline_result: The result of the candidate execution pipeline (may be None).

    Returns:
        The appropriate TelemetryCategory for this event.
    """
    # Priority 1-3: Parse violations take highest priority
    if parse_violation is not None:
        violation_type = parse_violation.get("type", "")

        if violation_type == "MALFORMED_RESPONSE":
            return TelemetryCategory.MALFORMED_PM_OUTPUT

        if violation_type == "UNKNOWN_CANDIDATE":
            return TelemetryCategory.UNKNOWN_CANDIDATE

        # Any other violation type (EXTRA_FIELDS, INVALID_DECISION, etc.)
        return TelemetryCategory.CONTRACT_VIOLATION

    # Priority 4: Pipeline result outcomes
    if pipeline_result is not None:
        outcome = pipeline_result.outcome

        if outcome == "executed":
            return TelemetryCategory.EXECUTED

        if outcome == "gate_rejected":
            return TelemetryCategory.GATE_REJECTION

        if outcome == "sizing_rejected":
            return TelemetryCategory.SIZING_REJECTION

        if outcome == "reservation_failed":
            return TelemetryCategory.UNKNOWN_CANDIDATE

        if outcome == "execution_failed":
            return TelemetryCategory.GATE_REJECTION

    # Priority 5: PM explicit rejection
    if decision is not None and decision.decision == "reject":
        return TelemetryCategory.PM_REJECTION

    # Priority 6: Default fallback
    return TelemetryCategory.CONTRACT_VIOLATION


def should_record_as_blocked_candidate(category: TelemetryCategory) -> bool:
    """Return True only if this category represents a legitimate trade rejection.

    MALFORMED_PM_OUTPUT and UNKNOWN_CANDIDATE are NEVER recorded as
    blocked_trade_candidates because they were never valid trade candidates
    (Requirement 9.2).

    CONTRACT_VIOLATION is also not a legitimate trade rejection — it indicates
    a protocol issue rather than a real trading decision.

    Only PM_REJECTION, GATE_REJECTION, and SIZING_REJECTION represent cases
    where a legitimate candidate was offered and rejected through normal
    decision or validation processes (Requirement 9.3).

    Args:
        category: The classified telemetry category for the event.

    Returns:
        True if the event should be recorded as a blocked trade candidate.
    """
    return category in (
        TelemetryCategory.PM_REJECTION,
        TelemetryCategory.GATE_REJECTION,
        TelemetryCategory.SIZING_REJECTION,
    )


def get_correlation_candidate_id(
    category: TelemetryCategory,
    decision: CandidateDecision | None,
    parse_violation: dict | None,
    pipeline_result: PipelineResult | None,
) -> str | None:
    """Extract candidate_id as a correlation field where valid (Requirement 9.4).

    Returns the candidate_id when one can be reliably identified from the event.
    Returns None for malformed output where no valid candidate_id exists.

    Args:
        category: The classified telemetry category.
        decision: The parsed PM decision (may be None).
        parse_violation: A violation dict (may be None).
        pipeline_result: The pipeline execution result (may be None).

    Returns:
        The candidate_id string if valid, or None if not determinable.
    """
    # For malformed output, there's typically no valid candidate_id
    if category == TelemetryCategory.MALFORMED_PM_OUTPUT:
        return None

    # Pipeline results always have a candidate_id
    if pipeline_result is not None:
        return pipeline_result.candidate_id

    # Parse violations may include a candidate_id field
    if parse_violation is not None:
        candidate_id = parse_violation.get("candidate_id")
        if candidate_id is not None:
            return candidate_id

    # Decisions always have a candidate_id
    if decision is not None:
        return decision.candidate_id

    return None
