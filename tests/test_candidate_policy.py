"""
Integration tests for candidate_policy support in the Decision Replay Agent.

Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6

Tests:
- Valid candidate_policy produces a replay with the candidate thresholds
- Invalid candidate_policy raises ValueError with missing field list
- Candidate_policy does not mutate os.environ or module constants after replay
- Delta attribution identifies which threshold/flag differs between candidate and current policy
"""

import copy
import os
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import patch, MagicMock

import pytest

from agents.decision_replay import run, _build_policy, BatchRunSummary
from core.replay.gate_adapter import (
    GatePolicyConfig,
    ReplayGateContext,
    build_replay_clock,
    build_deterministic_id_provider,
    build_gate_policy_config_from_snapshot,
)
from core.replay.gate_replayer import replay_gates, GateTrace
from core.replay.delta_classifier import classify_delta, DecisionDelta
from core.replay.policy_version import (
    PolicyVersion,
    validate_candidate_policy,
    build_current_policy_version,
)
from utils.gate_config import (
    STOP_DISTANCE_RULES,
    DEFAULT_STOP_DISTANCE_RULE,
    REDUCED_RR_THRESHOLDS_BY_PROFILE,
    HIGH_BETA_CLUSTER,
    QUALIFYING_MIN_SIGNAL_STRENGTH,
    QUALIFYING_SETUP_TYPES,
    MIN_WIN_RATE_BY_SETUP,
    DEFAULT_MIN_WIN_RATE,
    OVERRIDE_MIN_CONFIDENCE_SCORE,
)


# ---------------------------------------------------------------------------
# Helper: Build a complete valid candidate policy
# ---------------------------------------------------------------------------


def _make_valid_candidate_policy(**overrides) -> dict:
    """Construct a complete, valid candidate policy dict.

    Overrides allow customizing specific fields for test scenarios.
    """
    policy = {
        "name": "test_candidate_policy",
        "gate_revision": "test-rev-abc123",
        "feature_flags": {
            "SETUP_SPECIFIC_RR_THRESHOLDS": True,
            "MODERATE_NEAR_MISS_PILOT": False,
        },
        "gate_ordering_version": "v1.0",
        "adapter_version": "1.0.0",
        "thresholds": {
            "min_win_rate_by_setup": {"news_breakout": 0.40, "momentum_fade": 0.35},
            "default_min_win_rate": 0.40,
            "stop_distance_rules": STOP_DISTANCE_RULES,
            "default_stop_distance_rule": DEFAULT_STOP_DISTANCE_RULE,
            "override_min_confidence_score": 8.0,
            "reduced_rr_thresholds_by_profile": {"aggressive": 0.5, "moderate": 0.75},
            "qualifying_min_signal_strength": 7.5,
            "qualifying_setup_types": ["news_breakout", "technical_breakout"],
        },
    }
    # Apply overrides
    for key, value in overrides.items():
        if key == "thresholds" and isinstance(value, dict):
            policy["thresholds"].update(value)
        else:
            policy[key] = value
    return policy


def _make_gate_policy_config_from_candidate(candidate_policy: dict) -> GatePolicyConfig:
    """Build GatePolicyConfig from a candidate_policy dict (mirrors _build_policy logic)."""
    return build_gate_policy_config_from_snapshot({
        "gate_config": candidate_policy.get("thresholds", candidate_policy),
        "feature_flags": candidate_policy.get("feature_flags", {}),
    })


def _make_replay_context(
    signal_strength: float | None = 9.0,
    confidence_value=None,
    setup_type: str = "news_breakout",
    profile: str = "aggressive",
) -> ReplayGateContext:
    """Build a ReplayGateContext suitable for end-to-end gate replay."""
    return ReplayGateContext(
        account_equity=Decimal("100000"),
        available_cash=Decimal("50000"),
        open_positions=(),
        case_library_stats={
            "news_breakout": {
                "aggressive": {
                    "total_cases": 20,
                    "wins": 12,
                    "losses": 8,
                    "win_rate": 0.60,
                    "rolling_win_rate": 0.60,
                    "rolling_cases": 5,
                    "consecutive_losses": 0,
                    "rolling_avg_pnl": 50.0,
                }
            }
        },
        similarity_stats=None,
        analyst_signal_payload={
            "signal_strength": signal_strength,
            "confidence": confidence_value or "high",
            "setup_type": setup_type,
        },
        signal_strength=signal_strength,
        confidence_value=confidence_value or "high",
        selection_score=8.5,
        execution_score=8.0,
        override_confidence_score=9.0,
        override_reason="Strong catalyst",
        atr_value=3.5,
        atr_timestamp=datetime(2024, 11, 15, 14, 0, 0, tzinfo=timezone.utc),
        current_price=250.00,
        entry_price=Decimal("250.00"),
        stop_price=Decimal("247.50"),
        target_price=Decimal("252.78"),
        quantity=Decimal("40"),
        max_dollar_risk=200.0,
        symbol="TSLA",
        profile=profile,
        direction="LONG",
        setup_type=setup_type,
        catalyst_type="news_breakout",
        trade_metadata="Strong earnings beat",
        trade_rationale="TSLA news breakout on earnings",
        atr_source="5min_bars",
        rationale="Strong earnings beat",
        thesis="Momentum continuation",
        indicators=("earnings_beat", "volume_surge"),
        quote_timestamp=datetime(2024, 11, 15, 14, 0, 0, tzinfo=timezone.utc),
        strength="high",
        conviction="high",
    )


# ---------------------------------------------------------------------------
# Test 1: Valid candidate_policy produces replay with candidate thresholds
# ---------------------------------------------------------------------------


class TestValidCandidatePolicyReplay:
    """A valid candidate_policy produces a replay that uses the candidate's thresholds."""

    def test_build_policy_from_candidate_produces_correct_config(self):
        """_build_policy with candidate_policy returns GatePolicyConfig matching the candidate."""
        candidate_policy = _make_valid_candidate_policy(
            thresholds={"qualifying_min_signal_strength": 6.0}
        )

        pv, gate_config = _build_policy("current", candidate_policy)

        # Policy version should reflect the candidate
        assert pv.name == "test_candidate_policy"
        assert pv.gate_revision == "test-rev-abc123"
        assert pv.gate_ordering_version == "v1.0"
        assert pv.adapter_version == "1.0.0"

        # GatePolicyConfig should use candidate's threshold
        assert gate_config.qualifying_min_signal_strength == 6.0

    def test_candidate_policy_thresholds_used_in_gate_evaluation(self):
        """When candidate_policy uses a stricter R:R threshold, the gate rejects."""
        # Make a candidate policy with stricter default R:R (2.0 instead of 1.25)
        candidate_policy = _make_valid_candidate_policy(
            thresholds={
                "reduced_rr_thresholds_by_profile": {"aggressive": 2.0, "moderate": 2.0},
            },
        )

        gate_config = _make_gate_policy_config_from_candidate(candidate_policy)
        context = _make_replay_context(signal_strength=9.0)
        cutoff = datetime(2024, 11, 15, 14, 32, 0, tzinfo=timezone.utc)

        pv = PolicyVersion(
            name="test_candidate",
            gate_revision="test",
            config_digest="test",
            feature_flags={"SETUP_SPECIFIC_RR_THRESHOLDS": True},
            benchmark_version=None,
            config_source_timestamp=cutoff,
            gate_ordering_version="v1.0",
            adapter_version="1.0.0",
        )

        # R:R is ~1.11 — below the candidate's stricter 2.0 threshold
        trace = replay_gates(
            context=context,
            policy_config=gate_config,
            policy_version=pv,
            replay_id="test_candidate_strict",
            candidate_id="test_001",
            cutoff=cutoff,
            diagnostic_mode=False,
        )

        # Should reject because R:R 1.11 < candidate's 2.0
        assert trace.final_decision == "reject"
        assert trace.final_gate == "risk_geometry_gate"

    def test_candidate_policy_lenient_threshold_allows(self):
        """When candidate_policy uses a more lenient threshold, the gate allows."""
        # Make a candidate policy with very lenient R:R (0.3)
        candidate_policy = _make_valid_candidate_policy(
            thresholds={
                "reduced_rr_thresholds_by_profile": {"aggressive": 0.3, "moderate": 0.5},
            },
        )

        gate_config = _make_gate_policy_config_from_candidate(candidate_policy)
        context = _make_replay_context(signal_strength=9.0)
        cutoff = datetime(2024, 11, 15, 14, 32, 0, tzinfo=timezone.utc)

        pv = PolicyVersion(
            name="lenient_candidate",
            gate_revision="test",
            config_digest="test",
            feature_flags={"SETUP_SPECIFIC_RR_THRESHOLDS": True},
            benchmark_version=None,
            config_source_timestamp=cutoff,
            gate_ordering_version="v1.0",
            adapter_version="1.0.0",
        )

        trace = replay_gates(
            context=context,
            policy_config=gate_config,
            policy_version=pv,
            replay_id="test_candidate_lenient",
            candidate_id="test_002",
            cutoff=cutoff,
            diagnostic_mode=False,
        )

        # R:R 1.11 > 0.3 → should allow
        assert trace.final_decision == "allow"

    def test_candidate_policy_feature_flag_change_affects_gate(self):
        """Disabling SETUP_SPECIFIC_RR_THRESHOLDS in candidate policy uses default threshold."""
        # With flag disabled, gate falls back to default 1.25 threshold
        candidate_policy = _make_valid_candidate_policy(
            feature_flags={
                "SETUP_SPECIFIC_RR_THRESHOLDS": False,
                "MODERATE_NEAR_MISS_PILOT": False,
            }
        )

        gate_config = _make_gate_policy_config_from_candidate(candidate_policy)
        context = _make_replay_context(signal_strength=9.0)
        cutoff = datetime(2024, 11, 15, 14, 32, 0, tzinfo=timezone.utc)

        pv = PolicyVersion(
            name="flag_disabled",
            gate_revision="test",
            config_digest="test",
            feature_flags={"SETUP_SPECIFIC_RR_THRESHOLDS": False},
            benchmark_version=None,
            config_source_timestamp=cutoff,
            gate_ordering_version="v1.0",
            adapter_version="1.0.0",
        )

        trace = replay_gates(
            context=context,
            policy_config=gate_config,
            policy_version=pv,
            replay_id="test_flag_disabled",
            candidate_id="test_003",
            cutoff=cutoff,
            diagnostic_mode=False,
        )

        # With SETUP_SPECIFIC_RR_THRESHOLDS=False, uses default 1.25 threshold
        # R:R 1.11 < 1.25 → should reject
        assert trace.final_decision == "reject"


# ---------------------------------------------------------------------------
# Test 2: Invalid candidate_policy raises ValueError with missing field list
# ---------------------------------------------------------------------------


class TestInvalidCandidatePolicyRejection:
    """Invalid candidate_policy raises ValueError listing missing fields."""

    def test_missing_name_raises_value_error(self):
        """candidate_policy without name is rejected by validate_candidate_policy."""
        policy = _make_valid_candidate_policy()
        del policy["name"]

        is_valid, missing = validate_candidate_policy(policy)
        assert is_valid is False
        assert "name" in missing

    def test_missing_gate_revision_raises_value_error(self):
        """candidate_policy without gate_revision is rejected."""
        policy = _make_valid_candidate_policy()
        del policy["gate_revision"]

        is_valid, missing = validate_candidate_policy(policy)
        assert is_valid is False
        assert "gate_revision" in missing

    def test_missing_multiple_fields_lists_all(self):
        """candidate_policy missing multiple fields lists all missing fields."""
        policy = _make_valid_candidate_policy()
        del policy["name"]
        del policy["gate_revision"]
        del policy["feature_flags"]

        is_valid, missing = validate_candidate_policy(policy)
        assert is_valid is False
        assert "name" in missing
        assert "gate_revision" in missing
        assert "feature_flags" in missing

    def test_missing_threshold_fields_fails_validation(self):
        """candidate_policy missing required threshold fields is rejected."""
        policy = _make_valid_candidate_policy()
        del policy["thresholds"]["min_win_rate_by_setup"]
        del policy["thresholds"]["stop_distance_rules"]

        is_valid, missing = validate_candidate_policy(policy)
        assert is_valid is False
        assert any("min_win_rate_by_setup" in f for f in missing)
        assert any("stop_distance_rules" in f for f in missing)

    def test_non_boolean_feature_flags_fails_validation(self):
        """candidate_policy with non-boolean feature flags is rejected."""
        policy = _make_valid_candidate_policy(
            feature_flags={"SETUP_SPECIFIC_RR_THRESHOLDS": "yes"}
        )

        is_valid, missing = validate_candidate_policy(policy)
        assert is_valid is False
        assert any("feature_flags" in f.lower() or "boolean" in f.lower() for f in missing)

    def test_empty_name_fails_validation(self):
        """candidate_policy with empty name string is rejected."""
        policy = _make_valid_candidate_policy(name="   ")

        is_valid, missing = validate_candidate_policy(policy)
        assert is_valid is False

    def test_run_with_invalid_candidate_policy_raises(self):
        """Calling run() with an invalid candidate_policy raises ValueError immediately."""
        engine = MagicMock()
        bad_policy = {"name": "incomplete"}  # Missing most required fields

        with pytest.raises(ValueError) as exc_info:
            run(engine, candidate_policy=bad_policy)

        assert "Missing" in str(exc_info.value) or "missing" in str(exc_info.value).lower()

    def test_run_raises_with_specific_missing_fields_in_message(self):
        """run() ValueError message lists the specific missing fields."""
        engine = MagicMock()
        # Policy with name but missing gate_revision and feature_flags
        bad_policy = {
            "name": "partial",
            "gate_ordering_version": "v1.0",
            "adapter_version": "1.0.0",
        }

        with pytest.raises(ValueError) as exc_info:
            run(engine, candidate_policy=bad_policy)

        error_msg = str(exc_info.value)
        assert "gate_revision" in error_msg
        assert "feature_flags" in error_msg


# ---------------------------------------------------------------------------
# Test 3: Candidate_policy does not mutate os.environ or module constants
# ---------------------------------------------------------------------------


class TestCandidatePolicyNoMutation:
    """Candidate_policy replay is report-only: no mutation of production config."""

    def test_os_environ_unchanged_after_build_policy(self, monkeypatch):
        """Building a candidate policy does not mutate os.environ."""
        # Capture env state
        monkeypatch.setenv("SETUP_SPECIFIC_RR_THRESHOLDS", "false")
        monkeypatch.setenv("MODERATE_NEAR_MISS_PILOT", "false")
        env_before = dict(os.environ)

        candidate_policy = _make_valid_candidate_policy(
            feature_flags={
                "SETUP_SPECIFIC_RR_THRESHOLDS": True,
                "MODERATE_NEAR_MISS_PILOT": True,
            }
        )

        # Build the policy
        pv, gate_config = _build_policy("current", candidate_policy)

        # os.environ must not have been mutated
        assert os.environ.get("SETUP_SPECIFIC_RR_THRESHOLDS") == "false"
        assert os.environ.get("MODERATE_NEAR_MISS_PILOT") == "false"

    def test_module_constants_unchanged_after_gate_replay(self):
        """Module-level gate_config constants are unchanged after candidate replay."""
        from utils import gate_config as gc

        # Capture original values
        original_qualifying_min = gc.QUALIFYING_MIN_SIGNAL_STRENGTH
        original_reduced_rr = copy.deepcopy(gc.REDUCED_RR_THRESHOLDS_BY_PROFILE)
        original_stop_rules = gc.STOP_DISTANCE_RULES
        original_default_win_rate = gc.DEFAULT_MIN_WIN_RATE

        # Build a candidate policy with DIFFERENT thresholds
        candidate_policy = _make_valid_candidate_policy(
            thresholds={
                "qualifying_min_signal_strength": 99.0,
                "default_min_win_rate": 0.99,
                "reduced_rr_thresholds_by_profile": {"aggressive": 5.0},
            }
        )

        gate_config = _make_gate_policy_config_from_candidate(candidate_policy)
        context = _make_replay_context(signal_strength=9.0)
        cutoff = datetime(2024, 11, 15, 14, 32, 0, tzinfo=timezone.utc)

        pv = PolicyVersion(
            name="mutation_test",
            gate_revision="test",
            config_digest="test",
            feature_flags={"SETUP_SPECIFIC_RR_THRESHOLDS": True},
            benchmark_version=None,
            config_source_timestamp=cutoff,
            gate_ordering_version="v1.0",
            adapter_version="1.0.0",
        )

        # Run the gate replay
        replay_gates(
            context=context,
            policy_config=gate_config,
            policy_version=pv,
            replay_id="mutation_test",
            candidate_id="test_mutation",
            cutoff=cutoff,
            diagnostic_mode=True,
        )

        # Module constants must be unchanged
        assert gc.QUALIFYING_MIN_SIGNAL_STRENGTH == original_qualifying_min
        assert gc.REDUCED_RR_THRESHOLDS_BY_PROFILE == original_reduced_rr
        assert gc.STOP_DISTANCE_RULES is original_stop_rules
        assert gc.DEFAULT_MIN_WIN_RATE == original_default_win_rate

    def test_os_environ_unchanged_after_gate_replay(self, monkeypatch):
        """os.environ is unchanged after replaying with a candidate_policy."""
        monkeypatch.setenv("SETUP_SPECIFIC_RR_THRESHOLDS", "false")
        monkeypatch.setenv("MODERATE_NEAR_MISS_PILOT", "false")
        monkeypatch.setenv("PM_CANDIDATE_MODE", "disabled")

        candidate_policy = _make_valid_candidate_policy(
            feature_flags={
                "SETUP_SPECIFIC_RR_THRESHOLDS": True,
                "MODERATE_NEAR_MISS_PILOT": True,
            }
        )

        gate_config = _make_gate_policy_config_from_candidate(candidate_policy)
        context = _make_replay_context()
        cutoff = datetime(2024, 11, 15, 14, 32, 0, tzinfo=timezone.utc)

        pv = PolicyVersion(
            name="env_test",
            gate_revision="test",
            config_digest="test",
            feature_flags={
                "SETUP_SPECIFIC_RR_THRESHOLDS": True,
                "MODERATE_NEAR_MISS_PILOT": True,
            },
            benchmark_version=None,
            config_source_timestamp=cutoff,
            gate_ordering_version="v1.0",
            adapter_version="1.0.0",
        )

        replay_gates(
            context=context,
            policy_config=gate_config,
            policy_version=pv,
            replay_id="env_test",
            candidate_id="test_env",
            cutoff=cutoff,
            diagnostic_mode=True,
        )

        # Verify env vars were NOT mutated
        assert os.environ.get("SETUP_SPECIFIC_RR_THRESHOLDS") == "false"
        assert os.environ.get("MODERATE_NEAR_MISS_PILOT") == "false"
        assert os.environ.get("PM_CANDIDATE_MODE") == "disabled"

    def test_candidate_policy_gate_config_is_frozen(self):
        """The GatePolicyConfig built from candidate_policy is frozen (immutable)."""
        candidate_policy = _make_valid_candidate_policy()
        gate_config = _make_gate_policy_config_from_candidate(candidate_policy)

        # GatePolicyConfig is a frozen dataclass — assignment should raise
        with pytest.raises(AttributeError):
            gate_config.qualifying_min_signal_strength = 99.0


# ---------------------------------------------------------------------------
# Test 4: Delta attribution identifies threshold/flag difference
# ---------------------------------------------------------------------------


class TestDeltaAttributionForCandidatePolicy:
    """Delta attribution identifies the specific threshold/flag that differs."""

    def test_delta_attributed_to_diverging_gate_threshold(self):
        """When candidate policy uses stricter threshold, delta identifies the gate."""
        cutoff = datetime(2024, 11, 15, 14, 32, 0, tzinfo=timezone.utc)
        context = _make_replay_context(signal_strength=9.0)

        # Current policy: lenient (0.50 R:R threshold for aggressive)
        current_policy = _make_valid_candidate_policy(
            thresholds={"reduced_rr_thresholds_by_profile": {"aggressive": 0.5}},
        )
        current_config = _make_gate_policy_config_from_candidate(current_policy)
        current_pv = PolicyVersion(
            name="current",
            gate_revision="test",
            config_digest="test",
            feature_flags={"SETUP_SPECIFIC_RR_THRESHOLDS": True},
            benchmark_version=None,
            config_source_timestamp=cutoff,
            gate_ordering_version="v1.0",
            adapter_version="1.0.0",
        )

        # Replay with current policy (lenient): R:R 1.11 > 0.5 → allow
        current_trace = replay_gates(
            context=context,
            policy_config=current_config,
            policy_version=current_pv,
            replay_id="current_replay",
            candidate_id="test_attrib",
            cutoff=cutoff,
        )
        assert current_trace.final_decision == "allow"

        # Candidate policy: strict (2.0 R:R threshold for aggressive)
        candidate_policy = _make_valid_candidate_policy(
            thresholds={"reduced_rr_thresholds_by_profile": {"aggressive": 2.0}},
        )
        candidate_config = _make_gate_policy_config_from_candidate(candidate_policy)
        candidate_pv = PolicyVersion(
            name="strict_candidate",
            gate_revision="test",
            config_digest="test",
            feature_flags={"SETUP_SPECIFIC_RR_THRESHOLDS": True},
            benchmark_version=None,
            config_source_timestamp=cutoff,
            gate_ordering_version="v1.0",
            adapter_version="1.0.0",
        )

        # Replay with candidate policy (strict): R:R 1.11 < 2.0 → reject
        candidate_trace = replay_gates(
            context=context,
            policy_config=candidate_config,
            policy_version=candidate_pv,
            replay_id="candidate_replay",
            candidate_id="test_attrib",
            cutoff=cutoff,
        )
        assert candidate_trace.final_decision == "reject"

        # Classify the delta: current allows, candidate rejects
        delta = classify_delta(
            original_decision="allow",
            original_gate=None,
            original_reason_code=None,
            original_geometry={
                "entry_price": Decimal("250.00"),
                "stop_price": Decimal("247.50"),
                "target_price": Decimal("252.78"),
            },
            original_size=Decimal("40"),
            replay_trace=candidate_trace,
            replay_classification="exact",
        )

        # The delta should identify replay_rejects_original_allow
        assert delta.classification == "replay_rejects_original_allow"
        # First diverging gate should be risk_geometry_gate
        assert delta.first_diverging_gate == "risk_geometry_gate"

    def test_delta_attributed_to_feature_flag_difference(self):
        """When candidate disables a feature flag, delta identifies the divergence."""
        cutoff = datetime(2024, 11, 15, 14, 32, 0, tzinfo=timezone.utc)
        context = _make_replay_context(signal_strength=9.0)

        # With flag ENABLED: uses reduced threshold 0.5, allows R:R 1.11
        enabled_policy = _make_valid_candidate_policy(
            feature_flags={"SETUP_SPECIFIC_RR_THRESHOLDS": True, "MODERATE_NEAR_MISS_PILOT": False},
        )
        enabled_config = _make_gate_policy_config_from_candidate(enabled_policy)
        enabled_pv = PolicyVersion(
            name="enabled",
            gate_revision="test",
            config_digest="test",
            feature_flags={"SETUP_SPECIFIC_RR_THRESHOLDS": True},
            benchmark_version=None,
            config_source_timestamp=cutoff,
            gate_ordering_version="v1.0",
            adapter_version="1.0.0",
        )

        enabled_trace = replay_gates(
            context=context,
            policy_config=enabled_config,
            policy_version=enabled_pv,
            replay_id="enabled_trace",
            candidate_id="test_flag",
            cutoff=cutoff,
        )
        assert enabled_trace.final_decision == "allow"

        # With flag DISABLED: uses default threshold 1.25, rejects R:R 1.11
        disabled_policy = _make_valid_candidate_policy(
            feature_flags={"SETUP_SPECIFIC_RR_THRESHOLDS": False, "MODERATE_NEAR_MISS_PILOT": False},
        )
        disabled_config = _make_gate_policy_config_from_candidate(disabled_policy)
        disabled_pv = PolicyVersion(
            name="disabled",
            gate_revision="test",
            config_digest="test",
            feature_flags={"SETUP_SPECIFIC_RR_THRESHOLDS": False},
            benchmark_version=None,
            config_source_timestamp=cutoff,
            gate_ordering_version="v1.0",
            adapter_version="1.0.0",
        )

        disabled_trace = replay_gates(
            context=context,
            policy_config=disabled_config,
            policy_version=disabled_pv,
            replay_id="disabled_trace",
            candidate_id="test_flag",
            cutoff=cutoff,
        )
        assert disabled_trace.final_decision == "reject"
        assert disabled_trace.final_gate == "risk_geometry_gate"

        # Classify delta: enabled allows, disabled rejects
        delta = classify_delta(
            original_decision="allow",
            original_gate=None,
            original_reason_code=None,
            original_geometry={
                "entry_price": Decimal("250.00"),
                "stop_price": Decimal("247.50"),
                "target_price": Decimal("252.78"),
            },
            original_size=Decimal("40"),
            replay_trace=disabled_trace,
            replay_classification="exact",
        )

        assert delta.classification == "replay_rejects_original_allow"
        assert delta.first_diverging_gate == "risk_geometry_gate"

    def test_same_policy_produces_no_decision_divergence(self):
        """Same candidate_policy as current produces no decision direction change."""
        cutoff = datetime(2024, 11, 15, 14, 32, 0, tzinfo=timezone.utc)
        context = _make_replay_context(signal_strength=9.0)

        # Use identical policy for both
        policy = _make_valid_candidate_policy(
            thresholds={"reduced_rr_thresholds_by_profile": {"aggressive": 0.5}},
        )
        config = _make_gate_policy_config_from_candidate(policy)
        pv = PolicyVersion(
            name="same",
            gate_revision="test",
            config_digest="test",
            feature_flags={"SETUP_SPECIFIC_RR_THRESHOLDS": True},
            benchmark_version=None,
            config_source_timestamp=cutoff,
            gate_ordering_version="v1.0",
            adapter_version="1.0.0",
        )

        # Run the same policy twice — both should produce "allow"
        trace = replay_gates(
            context=context,
            policy_config=config,
            policy_version=pv,
            replay_id="same_test",
            candidate_id="test_same",
            cutoff=cutoff,
        )

        assert trace.final_decision == "allow"

        # Classify delta comparing against original allow decision
        # Use the geometry from the trace itself to ensure no spurious geometry diff
        from core.replay.delta_classifier import _extract_replay_geometry, _extract_replay_size

        replay_geometry = _extract_replay_geometry(trace)
        replay_size = _extract_replay_size(trace)

        delta = classify_delta(
            original_decision="allow",
            original_gate=None,
            original_reason_code=None,
            original_geometry={
                "entry_price": replay_geometry.get("entry_price", Decimal("250.00")),
                "stop_price": replay_geometry.get("stop_price", Decimal("247.50")),
                "target_price": replay_geometry.get("target_price", Decimal("252.78")),
            },
            original_size=replay_size if replay_size is not None else Decimal("40"),
            replay_trace=trace,
            replay_classification="exact",
        )

        # No direction change — both allow
        assert delta.original_decision == "allow"
        assert delta.replay_decision == "allow"
        # Should be one of the "same_allow" variants (no direction flip)
        assert delta.classification in (
            "same_allow",
            "same_final_allow_different_trace",
            "same_direction_different_size",
            "same_direction_different_geometry",
            "same_direction_different_size_and_geometry",
        )
        # Not a direction change
        assert delta.classification != "replay_rejects_original_allow"
        assert delta.classification != "replay_allows_original_reject"
