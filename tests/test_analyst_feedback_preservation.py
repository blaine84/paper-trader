"""
Preservation property tests for analyst feedback non-buggy paths.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

These tests verify that functions NOT affected by the bug condition
(queue_reviewer_flags, apply_signal_mitigation, _derive_flags,
evaluate_auto_mitigation, maybe_reset_weekly_mitigations) produce
correct results. They are run on UNFIXED code to establish baseline
behavior, then re-run after the fix to confirm no regressions.
"""

import json
from datetime import datetime, timedelta

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from sqlalchemy import create_engine

from db.schema import (
    Base,
    AnalystFeedbackQueue,
    AnalystMitigation,
    get_session,
)
from feedback_loop.analyst_feedback import (
    apply_signal_mitigation,
    evaluate_auto_mitigation,
    maybe_reset_weekly_mitigations,
    queue_reviewer_flags,
    _derive_flags,
    SIGNAL_SCORES,
    CONFIDENCE_SCORES,
    ENTRY_WINDOW_LIMITS,
    NO_DATA_REJECT_THRESHOLD,
    RESET_ACCEPTANCE_THRESHOLD,
    MIN_RESPONSES_FOR_MITIGATION,
)


@pytest.fixture
def engine():
    eng = create_engine("sqlite://")
    Base.metadata.create_all(eng)
    return eng


# ---------------------------------------------------------------------------
# Strategies for generating test data
# ---------------------------------------------------------------------------

setup_types = st.sampled_from(["gap_and_go", "orb", "momentum_fade", "short_squeeze", "vwap_reclaim", "breakout"])
signal_strengths = st.sampled_from(["weak", "moderate", "strong", ""])
signal_confidences = st.sampled_from(["low", "medium", "high", ""])
outcomes = st.sampled_from(["success", "failure", "breakeven", ""])
severities = st.sampled_from(["low", "medium", "high", "critical"])
symbols = st.sampled_from(["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "GOOG", "META", "AMZN"])
dates = st.sampled_from(["2026-01-15", "2026-02-20", "2026-03-10", "2026-04-29"])

reviewer_case_strategy = st.fixed_dictionaries({
    "trade_id": st.integers(min_value=1, max_value=1000),
    "symbol": symbols,
    "date": dates,
    "setup_type": setup_types,
    "selection_score": st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    "signal_strength": signal_strengths,
    "signal_confidence": signal_confidences,
    "holding_minutes": st.integers(min_value=0, max_value=300),
    "outcome": outcomes,
})

signal_strategy = st.fixed_dictionaries({
    "signal": st.sampled_from(["LONG", "SHORT", "HOLD"]),
    "strength": st.sampled_from(["weak", "moderate", "strong"]),
    "confidence": st.sampled_from(["low", "medium", "high"]),
    "setup_type": setup_types,
    "reasoning": st.text(min_size=0, max_size=50),
})

mitigation_config_strategy = st.fixed_dictionaries({
    "level": st.integers(min_value=1, max_value=5),
    "deployment_multiplier": st.floats(min_value=0.25, max_value=1.0, allow_nan=False, allow_infinity=False),
    "signal_threshold_bump": st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
})


# ---------------------------------------------------------------------------
# Property 2.1: queue_reviewer_flags preservation
# Validates: Requirements 3.1
# ---------------------------------------------------------------------------

class TestQueueReviewerFlagsPreservation:
    """Verify queue_reviewer_flags persists flags to DB and returns them correctly."""

    @given(cases=st.lists(reviewer_case_strategy, min_size=1, max_size=5))
    @settings(max_examples=50, deadline=None)
    def test_flags_persisted_and_returned(self, cases):
        """
        **Validates: Requirements 3.1**

        For any set of reviewer cases, queue_reviewer_flags should:
        1. Return a list of derived flags
        2. Persist those flags to the DB
        3. Each returned flag has required keys
        """
        eng = create_engine("sqlite://")
        Base.metadata.create_all(eng)

        result = queue_reviewer_flags(eng, cases)

        # Result is always a list
        assert isinstance(result, list)

        # Each flag in result has required keys
        for flag in result:
            assert "symbol" in flag
            assert "date" in flag
            assert "flag_type" in flag
            assert "severity" in flag
            assert "recommendation" in flag
            assert "reviewer_context" in flag

        # Verify DB persistence matches returned count
        db = get_session(eng)
        rows = db.query(AnalystFeedbackQueue).all()
        db.close()
        assert len(rows) == len(result)

    @given(cases=st.lists(reviewer_case_strategy, min_size=1, max_size=3))
    @settings(max_examples=30, deadline=None)
    def test_deduplication(self, cases):
        """
        **Validates: Requirements 3.1**

        Calling queue_reviewer_flags twice with the same cases should
        deduplicate - second call returns empty list.
        """
        eng = create_engine("sqlite://")
        Base.metadata.create_all(eng)

        first = queue_reviewer_flags(eng, cases)
        second = queue_reviewer_flags(eng, cases)

        # Second call should return empty (all already exist)
        assert second == []

        # DB should only have the flags from first call
        db = get_session(eng)
        rows = db.query(AnalystFeedbackQueue).all()
        db.close()
        assert len(rows) == len(first)

    def test_empty_cases(self):
        """
        **Validates: Requirements 3.1**

        Empty or None cases should return empty list.
        """
        eng = create_engine("sqlite://")
        Base.metadata.create_all(eng)

        assert queue_reviewer_flags(eng, []) == []
        assert queue_reviewer_flags(eng, None) == []


# ---------------------------------------------------------------------------
# Property 2.2: apply_signal_mitigation preservation
# Validates: Requirements 3.2
# ---------------------------------------------------------------------------

class TestApplySignalMitigationPreservation:
    """Verify apply_signal_mitigation deterministic throttle logic."""

    @given(signal=signal_strategy, mitigation=mitigation_config_strategy)
    @settings(max_examples=100, deadline=None)
    def test_hold_signals_unchanged(self, signal, mitigation):
        """
        **Validates: Requirements 3.2**

        Signals already at HOLD should not be modified (except adding mitigation metadata).
        """
        signal = dict(signal)
        signal["signal"] = "HOLD"
        setup = signal["setup_type"]
        mitigations = {setup: mitigation}

        result = apply_signal_mitigation(signal, mitigations)

        # HOLD signals stay HOLD
        assert result["signal"] == "HOLD"
        # No original_signal key added (wasn't converted)
        assert "original_signal" not in result

    @given(signal=signal_strategy, mitigation=mitigation_config_strategy)
    @settings(max_examples=100, deadline=None)
    def test_no_matching_mitigation_unchanged(self, signal, mitigation):
        """
        **Validates: Requirements 3.2**

        Signals with no matching mitigation for their setup_type should be unchanged.
        """
        signal = dict(signal)
        # Use a mitigation keyed to a different setup type
        mitigations = {"nonexistent_setup_xyz": mitigation}

        result = apply_signal_mitigation(signal, mitigations)

        # Signal should be completely unchanged
        assert result["signal"] == signal["signal"]
        assert "mitigation" not in result
        assert "original_signal" not in result

    @given(signal=signal_strategy, mitigation=mitigation_config_strategy)
    @settings(max_examples=100, deadline=None)
    def test_score_calculation_and_conversion(self, signal, mitigation):
        """
        **Validates: Requirements 3.2**

        For non-HOLD signals with matching mitigation:
        - effective_score = ((strength_score + confidence_score) / 2) * deployment_multiplier
        - required_score = 2.0 + signal_threshold_bump
        - If effective_score < required_score: convert to HOLD
        - If effective_score >= required_score: keep original signal
        """
        signal = dict(signal)
        assume(signal["signal"] != "HOLD")
        setup = signal["setup_type"]
        original_signal_value = signal["signal"]
        mitigations = {setup: mitigation}

        strength_score = SIGNAL_SCORES.get(signal["strength"].lower(), 1.0)
        confidence_score = CONFIDENCE_SCORES.get(signal["confidence"].lower(), 1.0)
        effective_score = ((strength_score + confidence_score) / 2.0) * mitigation["deployment_multiplier"]
        required_score = 2.0 + mitigation["signal_threshold_bump"]

        result = apply_signal_mitigation(signal, mitigations)

        # Mitigation metadata always added
        assert "mitigation" in result

        if effective_score >= required_score:
            # Signal preserved
            assert result["signal"] == original_signal_value
            assert "original_signal" not in result
        else:
            # Converted to HOLD
            assert result["signal"] == "HOLD"
            assert result["original_signal"] == original_signal_value
            assert result["strength"] == "weak"
            assert result["confidence"] == "low"

    def test_non_dict_input_returned_as_is(self):
        """
        **Validates: Requirements 3.2**

        Non-dict inputs should be returned unchanged.
        """
        assert apply_signal_mitigation(None, {}) is None
        assert apply_signal_mitigation("string", {}) == "string"
        assert apply_signal_mitigation(42, {}) == 42

    def test_empty_mitigations_unchanged(self):
        """
        **Validates: Requirements 3.2**

        Empty mitigations dict means no changes.
        """
        signal = {"signal": "LONG", "strength": "strong", "confidence": "high", "setup_type": "orb", "reasoning": "test"}
        result = apply_signal_mitigation(signal, {})
        assert result["signal"] == "LONG"
        assert "mitigation" not in result


# ---------------------------------------------------------------------------
# Property 2.3: _derive_flags preservation
# Validates: Requirements 3.6
# ---------------------------------------------------------------------------

class TestDeriveFlagsPreservation:
    """Verify _derive_flags produces correct flags based on thresholds."""

    @given(case=reviewer_case_strategy)
    @settings(max_examples=100, deadline=None)
    def test_flag_derivation_correctness(self, case):
        """
        **Validates: Requirements 3.6**

        For any case dict, _derive_flags should produce flags based on:
        - selection_score <= 5 → selection_score_below_threshold
        - signal_strength == "weak" and outcome != "success" → signal_strength_below_threshold
        - signal_confidence == "low" and outcome != "success" → signal_confidence_below_threshold
        - holding_minutes > ENTRY_WINDOW_LIMITS[setup_type] → hold_time_violated
        """
        flags = _derive_flags(case)

        assert isinstance(flags, list)

        flag_types = [f["flag_type"] for f in flags]

        selection_score = float(case.get("selection_score") or 0)
        signal_strength = str(case.get("signal_strength") or "").lower()
        signal_confidence = str(case.get("signal_confidence") or "").lower()
        holding_minutes = case.get("holding_minutes") or 0
        outcome = str(case.get("outcome") or "")
        setup_type = case.get("setup_type")

        # Check selection_score flag
        if selection_score and selection_score <= 5:
            assert "selection_score_below_threshold" in flag_types
        else:
            assert "selection_score_below_threshold" not in flag_types

        # Check signal_strength flag
        if signal_strength == "weak" and outcome != "success":
            assert "signal_strength_below_threshold" in flag_types
        else:
            assert "signal_strength_below_threshold" not in flag_types

        # Check signal_confidence flag
        if signal_confidence == "low" and outcome != "success":
            assert "signal_confidence_below_threshold" in flag_types
        else:
            assert "signal_confidence_below_threshold" not in flag_types

        # Check hold_time_violated flag
        limit = ENTRY_WINDOW_LIMITS.get(setup_type or "")
        if limit and holding_minutes and holding_minutes > limit:
            assert f"{setup_type}_hold_time_violated" in flag_types
        else:
            if setup_type:
                assert f"{setup_type}_hold_time_violated" not in flag_types

    @given(case=reviewer_case_strategy)
    @settings(max_examples=50, deadline=None)
    def test_flag_structure(self, case):
        """
        **Validates: Requirements 3.6**

        Every derived flag must have the required structure.
        """
        flags = _derive_flags(case)

        for flag in flags:
            assert "trade_id" in flag
            assert "symbol" in flag
            assert "setup_type" in flag
            assert "date" in flag
            assert "flag_type" in flag
            assert "severity" in flag
            assert "recommendation" in flag
            assert "reviewer_context" in flag
            assert flag["severity"] in {"low", "medium", "high", "critical"}

    @given(case=st.fixed_dictionaries({
        "trade_id": st.integers(min_value=1, max_value=100),
        "symbol": symbols,
        "date": dates,
        "setup_type": setup_types,
        "selection_score": st.floats(min_value=0.0, max_value=3.0, allow_nan=False, allow_infinity=False),
        "signal_strength": st.just("weak"),
        "signal_confidence": st.just("low"),
        "holding_minutes": st.integers(min_value=61, max_value=300),
        "outcome": st.just("failure"),
    }))
    @settings(max_examples=30, deadline=None)
    def test_all_flags_triggered(self, case):
        """
        **Validates: Requirements 3.6**

        When all conditions are met, all 4 flag types should be generated.
        """
        assume(case["selection_score"] > 0)  # selection_score must be truthy
        assume(case["setup_type"] in ENTRY_WINDOW_LIMITS)

        flags = _derive_flags(case)
        flag_types = [f["flag_type"] for f in flags]

        assert "selection_score_below_threshold" in flag_types
        assert "signal_strength_below_threshold" in flag_types
        assert "signal_confidence_below_threshold" in flag_types
        assert f"{case['setup_type']}_hold_time_violated" in flag_types


# ---------------------------------------------------------------------------
# Property 2.4: evaluate_auto_mitigation preservation
# Validates: Requirements 3.3
# ---------------------------------------------------------------------------

class TestEvaluateAutoMitigationPreservation:
    """Verify mitigation escalation logic for high unsupported-reject rates."""

    @given(
        num_rejects=st.integers(min_value=2, max_value=10),
        num_accepts=st.integers(min_value=0, max_value=3),
    )
    @settings(max_examples=50, deadline=None)
    def test_escalation_when_reject_rate_exceeds_threshold(self, num_rejects, num_accepts):
        """
        **Validates: Requirements 3.3**

        When unsupported reject rate > NO_DATA_REJECT_THRESHOLD (0.50)
        and total responses >= MIN_RESPONSES_FOR_MITIGATION (2),
        mitigation should be escalated.
        """
        total = num_rejects + num_accepts
        reject_rate = num_rejects / total
        assume(reject_rate > NO_DATA_REJECT_THRESHOLD)
        assume(total >= MIN_RESPONSES_FOR_MITIGATION)

        eng = create_engine("sqlite://")
        Base.metadata.create_all(eng)

        now = datetime.utcnow()
        db = get_session(eng)

        # Add no-data rejects
        for i in range(num_rejects):
            db.add(AnalystFeedbackQueue(
                symbol="TEST",
                setup_type="gap_and_go",
                date="2026-04-29",
                flag_type=f"flag_{i}",
                severity="high",
                recommendation="Test",
                reviewer_context=json.dumps({}),
                due_at=now,
                responded_at=now - timedelta(hours=1),
                status="responded",
                analyst_response="reject",
                analyst_supporting_data=json.dumps([]),
                no_data_reject=True,
            ))

        # Add accepts
        for i in range(num_accepts):
            db.add(AnalystFeedbackQueue(
                symbol="TEST",
                setup_type="gap_and_go",
                date="2026-04-29",
                flag_type=f"accept_flag_{i}",
                severity="medium",
                recommendation="Test",
                reviewer_context=json.dumps({}),
                due_at=now,
                responded_at=now - timedelta(hours=1),
                status="responded",
                analyst_response="accept",
                analyst_supporting_data=json.dumps([]),
                no_data_reject=False,
            ))

        db.commit()
        db.close()

        evaluate_auto_mitigation(eng)

        db = get_session(eng)
        mitigation = db.query(AnalystMitigation).filter_by(setup_type="gap_and_go").first()
        db.close()

        assert mitigation is not None
        assert mitigation.active is True
        assert mitigation.level == 1
        assert mitigation.deployment_multiplier == 0.75
        assert mitigation.signal_threshold_bump == 0.5

    @given(
        num_rejects=st.integers(min_value=0, max_value=2),
        num_accepts=st.integers(min_value=3, max_value=10),
    )
    @settings(max_examples=50, deadline=None)
    def test_no_escalation_when_reject_rate_below_threshold(self, num_rejects, num_accepts):
        """
        **Validates: Requirements 3.3**

        When unsupported reject rate <= NO_DATA_REJECT_THRESHOLD (0.50),
        no mitigation should be created.
        """
        total = num_rejects + num_accepts
        reject_rate = num_rejects / total
        assume(reject_rate <= NO_DATA_REJECT_THRESHOLD)
        assume(total >= MIN_RESPONSES_FOR_MITIGATION)

        eng = create_engine("sqlite://")
        Base.metadata.create_all(eng)

        now = datetime.utcnow()
        db = get_session(eng)

        for i in range(num_rejects):
            db.add(AnalystFeedbackQueue(
                symbol="TEST",
                setup_type="orb",
                date="2026-04-29",
                flag_type=f"flag_{i}",
                severity="high",
                recommendation="Test",
                reviewer_context=json.dumps({}),
                due_at=now,
                responded_at=now - timedelta(hours=1),
                status="responded",
                analyst_response="reject",
                analyst_supporting_data=json.dumps([]),
                no_data_reject=True,
            ))

        for i in range(num_accepts):
            db.add(AnalystFeedbackQueue(
                symbol="TEST",
                setup_type="orb",
                date="2026-04-29",
                flag_type=f"accept_flag_{i}",
                severity="medium",
                recommendation="Test",
                reviewer_context=json.dumps({}),
                due_at=now,
                responded_at=now - timedelta(hours=1),
                status="responded",
                analyst_response="accept",
                analyst_supporting_data=json.dumps([]),
                no_data_reject=False,
            ))

        db.commit()
        db.close()

        evaluate_auto_mitigation(eng)

        db = get_session(eng)
        mitigation = db.query(AnalystMitigation).filter_by(setup_type="orb").first()
        db.close()

        assert mitigation is None

    def test_insufficient_responses_no_mitigation(self):
        """
        **Validates: Requirements 3.3**

        When total responses < MIN_RESPONSES_FOR_MITIGATION (2),
        no mitigation should be created even with 100% reject rate.
        """
        eng = create_engine("sqlite://")
        Base.metadata.create_all(eng)

        now = datetime.utcnow()
        db = get_session(eng)
        db.add(AnalystFeedbackQueue(
            symbol="TEST",
            setup_type="momentum_fade",
            date="2026-04-29",
            flag_type="flag_0",
            severity="high",
            recommendation="Test",
            reviewer_context=json.dumps({}),
            due_at=now,
            responded_at=now - timedelta(hours=1),
            status="responded",
            analyst_response="reject",
            analyst_supporting_data=json.dumps([]),
            no_data_reject=True,
        ))
        db.commit()
        db.close()

        evaluate_auto_mitigation(eng)

        db = get_session(eng)
        mitigation = db.query(AnalystMitigation).filter_by(setup_type="momentum_fade").first()
        db.close()

        assert mitigation is None


# ---------------------------------------------------------------------------
# Property 2.5: maybe_reset_weekly_mitigations preservation
# Validates: Requirements 3.4
# ---------------------------------------------------------------------------

class TestMaybeResetWeeklyMitigationsPreservation:
    """Verify reset logic when acceptance rate exceeds threshold."""

    @given(
        num_accepts=st.integers(min_value=5, max_value=10),
        num_rejects=st.integers(min_value=0, max_value=1),
    )
    @settings(max_examples=50, deadline=None)
    def test_reset_when_acceptance_exceeds_threshold(self, num_accepts, num_rejects):
        """
        **Validates: Requirements 3.4**

        When acceptance rate > RESET_ACCEPTANCE_THRESHOLD (0.80),
        active mitigations should be reset.
        """
        total = num_accepts + num_rejects
        acceptance_rate = num_accepts / total
        assume(acceptance_rate > RESET_ACCEPTANCE_THRESHOLD)

        eng = create_engine("sqlite://")
        Base.metadata.create_all(eng)

        now = datetime.utcnow()
        db = get_session(eng)

        # Add active mitigation
        db.add(AnalystMitigation(
            setup_type="short_squeeze",
            level=2,
            deployment_multiplier=0.5,
            signal_threshold_bump=1.0,
            active=True,
            applied_at=now - timedelta(days=3),
        ))

        # Add accepted responses
        for i in range(num_accepts):
            db.add(AnalystFeedbackQueue(
                symbol="TEST",
                setup_type="short_squeeze",
                date="2026-04-29",
                flag_type=f"flag_{i}",
                severity="medium",
                recommendation="Test",
                reviewer_context=json.dumps({}),
                due_at=now,
                responded_at=now - timedelta(hours=i + 1),
                status="responded",
                analyst_response="accept",
                analyst_supporting_data=json.dumps([]),
                no_data_reject=False,
            ))

        # Add valid rejects (with supporting data - these count as valid)
        for i in range(num_rejects):
            db.add(AnalystFeedbackQueue(
                symbol="TEST",
                setup_type="short_squeeze",
                date="2026-04-29",
                flag_type=f"reject_flag_{i}",
                severity="medium",
                recommendation="Test",
                reviewer_context=json.dumps({}),
                due_at=now,
                responded_at=now - timedelta(hours=i + 1),
                status="responded",
                analyst_response="reject",
                analyst_supporting_data=json.dumps(["evidence"]),
                no_data_reject=False,
            ))

        db.commit()
        db.close()

        maybe_reset_weekly_mitigations(eng)

        db = get_session(eng)
        mitigation = db.query(AnalystMitigation).filter_by(setup_type="short_squeeze").first()
        db.close()

        assert mitigation.active is False
        assert mitigation.level == 0
        assert mitigation.deployment_multiplier == 1.0
        assert mitigation.signal_threshold_bump == 0.0

    @given(
        num_accepts=st.integers(min_value=1, max_value=4),
        num_rejects=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=50, deadline=None)
    def test_no_reset_when_acceptance_below_threshold(self, num_accepts, num_rejects):
        """
        **Validates: Requirements 3.4**

        When acceptance rate <= RESET_ACCEPTANCE_THRESHOLD (0.80),
        mitigations should remain active.
        """
        total = num_accepts + num_rejects
        acceptance_rate = num_accepts / total
        assume(acceptance_rate <= RESET_ACCEPTANCE_THRESHOLD)

        eng = create_engine("sqlite://")
        Base.metadata.create_all(eng)

        now = datetime.utcnow()
        db = get_session(eng)

        db.add(AnalystMitigation(
            setup_type="gap_and_go",
            level=1,
            deployment_multiplier=0.75,
            signal_threshold_bump=0.5,
            active=True,
            applied_at=now - timedelta(days=2),
        ))

        for i in range(num_accepts):
            db.add(AnalystFeedbackQueue(
                symbol="TEST",
                setup_type="gap_and_go",
                date="2026-04-29",
                flag_type=f"flag_{i}",
                severity="medium",
                recommendation="Test",
                reviewer_context=json.dumps({}),
                due_at=now,
                responded_at=now - timedelta(hours=i + 1),
                status="responded",
                analyst_response="accept",
                analyst_supporting_data=json.dumps([]),
                no_data_reject=False,
            ))

        # Add valid rejects (with supporting data)
        for i in range(num_rejects):
            db.add(AnalystFeedbackQueue(
                symbol="TEST",
                setup_type="gap_and_go",
                date="2026-04-29",
                flag_type=f"reject_flag_{i}",
                severity="medium",
                recommendation="Test",
                reviewer_context=json.dumps({}),
                due_at=now,
                responded_at=now - timedelta(hours=i + 1),
                status="responded",
                analyst_response="reject",
                analyst_supporting_data=json.dumps(["evidence"]),
                no_data_reject=False,
            ))

        db.commit()
        db.close()

        maybe_reset_weekly_mitigations(eng)

        db = get_session(eng)
        mitigation = db.query(AnalystMitigation).filter_by(setup_type="gap_and_go").first()
        db.close()

        assert mitigation.active is True
        assert mitigation.level == 1

    def test_no_active_mitigations_noop(self):
        """
        **Validates: Requirements 3.4**

        When no active mitigations exist, function should be a no-op.
        """
        eng = create_engine("sqlite://")
        Base.metadata.create_all(eng)

        # Should not raise
        maybe_reset_weekly_mitigations(eng)

        db = get_session(eng)
        mitigations = db.query(AnalystMitigation).all()
        db.close()
        assert mitigations == []
