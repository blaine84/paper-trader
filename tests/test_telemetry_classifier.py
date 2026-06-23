"""Unit tests for utils/telemetry_classifier.py.

Validates the classification logic and should_record_as_blocked_candidate helper.
Requirements: 9.1, 9.2, 9.3, 9.4
"""

import pytest

from utils.telemetry_classifier import (
    TelemetryCategory,
    classify_pm_event,
    get_correlation_candidate_id,
    should_record_as_blocked_candidate,
)
from utils.decision_contract import CandidateDecision
from utils.candidate_pipeline import PipelineResult


# ---------------------------------------------------------------------------
# classify_pm_event — parse violation priority
# ---------------------------------------------------------------------------


class TestClassifyParseViolations:
    """Priority 1-3: parse violations take highest precedence."""

    def test_malformed_response_returns_malformed_pm_output(self):
        violation = {"type": "MALFORMED_RESPONSE", "reason": "missing 'decisions' key"}
        result = classify_pm_event(None, violation, None)
        assert result == TelemetryCategory.MALFORMED_PM_OUTPUT

    def test_unknown_candidate_violation_returns_unknown_candidate(self):
        violation = {"type": "UNKNOWN_CANDIDATE", "candidate_id": "abc-123"}
        result = classify_pm_event(None, violation, None)
        assert result == TelemetryCategory.UNKNOWN_CANDIDATE

    def test_other_violation_returns_contract_violation(self):
        violation = {"type": "EXTRA_FIELDS", "candidate_id": "abc-123", "fields": ["foo"]}
        result = classify_pm_event(None, violation, None)
        assert result == TelemetryCategory.CONTRACT_VIOLATION

    def test_invalid_decision_violation_returns_contract_violation(self):
        violation = {"type": "INVALID_DECISION", "candidate_id": "x", "decision": "maybe"}
        result = classify_pm_event(None, violation, None)
        assert result == TelemetryCategory.CONTRACT_VIOLATION

    def test_duplicate_candidate_violation_returns_contract_violation(self):
        violation = {"type": "DUPLICATE_CANDIDATE", "candidate_id": "dup-1"}
        result = classify_pm_event(None, violation, None)
        assert result == TelemetryCategory.CONTRACT_VIOLATION

    def test_violation_takes_priority_over_decision(self):
        """Parse violation should override decision if both present."""
        decision = CandidateDecision(candidate_id="abc", decision="reject")
        violation = {"type": "MALFORMED_RESPONSE", "reason": "bad json"}
        result = classify_pm_event(decision, violation, None)
        assert result == TelemetryCategory.MALFORMED_PM_OUTPUT

    def test_violation_takes_priority_over_pipeline_result(self):
        """Parse violation should override pipeline result if both present."""
        pipeline = PipelineResult(candidate_id="abc", outcome="executed")
        violation = {"type": "UNKNOWN_CANDIDATE", "candidate_id": "abc"}
        result = classify_pm_event(None, violation, pipeline)
        assert result == TelemetryCategory.UNKNOWN_CANDIDATE


# ---------------------------------------------------------------------------
# classify_pm_event — pipeline result priority
# ---------------------------------------------------------------------------


class TestClassifyPipelineResults:
    """Priority 4: pipeline result outcomes (when no violation)."""

    def test_executed_outcome(self):
        pipeline = PipelineResult(candidate_id="c1", outcome="executed")
        result = classify_pm_event(None, None, pipeline)
        assert result == TelemetryCategory.EXECUTED

    def test_gate_rejected_outcome(self):
        pipeline = PipelineResult(candidate_id="c1", outcome="gate_rejected")
        result = classify_pm_event(None, None, pipeline)
        assert result == TelemetryCategory.GATE_REJECTION

    def test_sizing_rejected_outcome(self):
        pipeline = PipelineResult(candidate_id="c1", outcome="sizing_rejected")
        result = classify_pm_event(None, None, pipeline)
        assert result == TelemetryCategory.SIZING_REJECTION

    def test_reservation_failed_outcome(self):
        pipeline = PipelineResult(candidate_id="c1", outcome="reservation_failed")
        result = classify_pm_event(None, None, pipeline)
        assert result == TelemetryCategory.UNKNOWN_CANDIDATE

    def test_execution_failed_treated_as_gate_rejection(self):
        pipeline = PipelineResult(candidate_id="c1", outcome="execution_failed")
        result = classify_pm_event(None, None, pipeline)
        assert result == TelemetryCategory.GATE_REJECTION


# ---------------------------------------------------------------------------
# classify_pm_event — decision priority
# ---------------------------------------------------------------------------


class TestClassifyDecisions:
    """Priority 5: PM decision when no violation and no pipeline result."""

    def test_reject_decision(self):
        decision = CandidateDecision(candidate_id="c1", decision="reject")
        result = classify_pm_event(decision, None, None)
        assert result == TelemetryCategory.PM_REJECTION

    def test_accept_decision_without_pipeline_falls_to_default(self):
        """An accept decision with no pipeline result hits the default."""
        decision = CandidateDecision(candidate_id="c1", decision="accept")
        result = classify_pm_event(decision, None, None)
        assert result == TelemetryCategory.CONTRACT_VIOLATION


# ---------------------------------------------------------------------------
# classify_pm_event — default
# ---------------------------------------------------------------------------


class TestClassifyDefault:
    """Priority 6: default fallback."""

    def test_all_none_returns_contract_violation(self):
        result = classify_pm_event(None, None, None)
        assert result == TelemetryCategory.CONTRACT_VIOLATION


# ---------------------------------------------------------------------------
# should_record_as_blocked_candidate
# ---------------------------------------------------------------------------


class TestShouldRecordAsBlockedCandidate:
    """Requirement 9.2: malformed output NEVER recorded as blocked trade."""

    def test_malformed_pm_output_not_blocked(self):
        assert should_record_as_blocked_candidate(TelemetryCategory.MALFORMED_PM_OUTPUT) is False

    def test_unknown_candidate_not_blocked(self):
        assert should_record_as_blocked_candidate(TelemetryCategory.UNKNOWN_CANDIDATE) is False

    def test_contract_violation_not_blocked(self):
        assert should_record_as_blocked_candidate(TelemetryCategory.CONTRACT_VIOLATION) is False

    def test_executed_not_blocked(self):
        assert should_record_as_blocked_candidate(TelemetryCategory.EXECUTED) is False

    def test_pm_rejection_is_blocked(self):
        """Requirement 9.3: legitimate rejection IS recorded as blocked."""
        assert should_record_as_blocked_candidate(TelemetryCategory.PM_REJECTION) is True

    def test_gate_rejection_is_blocked(self):
        assert should_record_as_blocked_candidate(TelemetryCategory.GATE_REJECTION) is True

    def test_sizing_rejection_is_blocked(self):
        assert should_record_as_blocked_candidate(TelemetryCategory.SIZING_REJECTION) is True


# ---------------------------------------------------------------------------
# get_correlation_candidate_id
# ---------------------------------------------------------------------------


class TestGetCorrelationCandidateId:
    """Requirement 9.4: include candidate_id as correlation field where valid."""

    def test_malformed_output_returns_none(self):
        result = get_correlation_candidate_id(
            TelemetryCategory.MALFORMED_PM_OUTPUT, None, None, None
        )
        assert result is None

    def test_pipeline_result_provides_candidate_id(self):
        pipeline = PipelineResult(candidate_id="pipe-123", outcome="executed")
        result = get_correlation_candidate_id(
            TelemetryCategory.EXECUTED, None, None, pipeline
        )
        assert result == "pipe-123"

    def test_violation_with_candidate_id_field(self):
        violation = {"type": "UNKNOWN_CANDIDATE", "candidate_id": "viol-456"}
        result = get_correlation_candidate_id(
            TelemetryCategory.UNKNOWN_CANDIDATE, None, violation, None
        )
        assert result == "viol-456"

    def test_decision_provides_candidate_id(self):
        decision = CandidateDecision(candidate_id="dec-789", decision="reject")
        result = get_correlation_candidate_id(
            TelemetryCategory.PM_REJECTION, decision, None, None
        )
        assert result == "dec-789"

    def test_no_source_returns_none(self):
        result = get_correlation_candidate_id(
            TelemetryCategory.CONTRACT_VIOLATION, None, None, None
        )
        assert result is None
