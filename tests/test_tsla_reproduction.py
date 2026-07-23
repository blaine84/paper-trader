"""
TSLA Aggressive news_breakout R:R Defect Reproduction Test.

**Validates: Requirements 6.3, 6.4, Success Criteria 4**

This integration test reproduces the historical TSLA defect where:
- An aggressive news_breakout candidate with signal_strength=9.0 and confidence="high"
- The original risk_geometry_gate used a DEFAULT 1.25 R:R threshold because
  signal metadata (signal_strength, confidence_level) was dropped/null when
  passed to the gate (metadata_wiring_defect)
- With the fix, the gate correctly receives metadata, recognizes it qualifies
  for setup-specific R:R thresholds, and uses the reduced 0.50 threshold
- The candidate had R:R of 1.11 — above 0.50 but below 1.25
- Therefore: original rejected (R:R 1.11 < 1.25), replay allows (R:R 1.11 > 0.50)
- Delta: replay_allows_original_reject
- Divergence cause: metadata_wiring_defect
- First diverging gate: risk_geometry_gate

End-to-end pipeline: snapshot → reconstruct → replay → classify → report
"""

import os
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from core.replay.gate_adapter import (
    GatePolicyConfig,
    ReplayGateContext,
    build_replay_clock,
    build_deterministic_id_provider,
)
from core.replay.gate_replayer import (
    GateTrace,
    GateTraceEntry,
    replay_gates,
    evaluate_single_gate,
    HARD_REJECTION_DECISIONS,
)
from core.replay.delta_classifier import (
    DecisionDelta,
    classify_delta,
    identify_first_diverging_gate,
    classify_divergence_cause,
)
from core.replay.policy_version import PolicyVersion
from utils.gate_config import (
    REDUCED_RR_THRESHOLDS_BY_PROFILE,
    QUALIFYING_MIN_SIGNAL_STRENGTH,
    QUALIFYING_SETUP_TYPES,
    STOP_DISTANCE_RULES,
    DEFAULT_STOP_DISTANCE_RULE,
    HIGH_BETA_CLUSTER,
)


# ---------------------------------------------------------------------------
# Test Fixtures — Historical TSLA Aggressive news_breakout Candidate
# ---------------------------------------------------------------------------


@pytest.fixture
def tsla_entry_timestamp():
    """Historical TSLA entry timestamp."""
    return datetime(2024, 11, 15, 14, 32, 0, tzinfo=timezone.utc)


@pytest.fixture
def tsla_geometry():
    """TSLA trade geometry producing R:R of ~1.11.

    Entry: 250.00, Stop: 247.50 (distance=2.50), Target: 252.78 (distance=2.78)
    R:R = 2.78 / 2.50 = 1.112
    This is above 0.50 (aggressive reduced threshold) but below 1.25 (default).
    """
    return {
        "entry_price": Decimal("250.00"),
        "stop_price": Decimal("247.50"),
        "target_price": Decimal("252.78"),
        "quantity": Decimal("40"),
    }


@pytest.fixture
def tsla_policy_version():
    """Policy version representing the current (fixed) policy."""
    return PolicyVersion(
        name="current",
        gate_revision="v2.1.0-fixed",
        config_digest="test_digest",
        feature_flags={"SETUP_SPECIFIC_RR_THRESHOLDS": True},
        benchmark_version=None,
        config_source_timestamp=datetime(2024, 11, 15, 10, 0, 0),
        gate_ordering_version="v1.0",
        adapter_version="1.0.0",
    )


@pytest.fixture
def tsla_gate_policy_config():
    """GatePolicyConfig with SETUP_SPECIFIC_RR_THRESHOLDS enabled (current policy).

    This represents the FIXED state where the gate receives signal metadata
    and can use reduced R:R thresholds for qualifying setups.
    """
    return GatePolicyConfig(
        # Setup quality gate (pass-through values, not the focus here)
        min_win_rate_by_setup={"news_breakout": 0.40},
        default_min_win_rate=0.40,
        min_win_rate_by_setup_profile={"news_breakout": {"aggressive": 0.25}},
        default_min_win_rate_by_profile={"aggressive": 0.25},
        rolling_window=5,
        min_cases_for_block=5,
        min_rolling_cases=3,
        consecutive_loss_pause_threshold=3,
        recovery_min_rolling_cases=5,
        recovery_win_rate_margin=0.15,
        require_positive_rolling_avg_pnl_for_recovery=True,
        rolling_recovery_probe_size_multiplier=0.25,
        near_miss_margin_pct=0.05,
        # Pre-trade quality gate
        override_min_confidence_score=8.0,
        # Risk geometry gate — the key thresholds
        stop_distance_rules=STOP_DISTANCE_RULES,
        default_stop_distance_rule=DEFAULT_STOP_DISTANCE_RULE,
        reduced_rr_thresholds_by_profile=REDUCED_RR_THRESHOLDS_BY_PROFILE,
        high_beta_cluster=frozenset(HIGH_BETA_CLUSTER),
        qualifying_min_signal_strength=QUALIFYING_MIN_SIGNAL_STRENGTH,
        qualifying_setup_types=frozenset(QUALIFYING_SETUP_TYPES),
        # Feature flags — SETUP_SPECIFIC_RR_THRESHOLDS is ENABLED (the fix)
        feature_flags={
            "SETUP_SPECIFIC_RR_THRESHOLDS": True,
            "MODERATE_NEAR_MISS_PILOT": False,
        },
        # Extended decision boundary (Phase 2)
        edge_score_min_threshold=0.4,
        hard_rejection_min_winrate=0.0,
        hard_rejection_min_sample_size=0,
        adaptive_throttle_loss_threshold=3,
        portfolio_risk_limits={},
        correlation_limits={},
        # Catalyst specificity gate
        catalyst_specificity_profile_thresholds={
            "aggressive": {"allow": 6, "warn": 4},
        },
        catalyst_specificity_sector_sympathy_size_multiplier={
            "aggressive": 0.5,
        },
        # Policy identity
        gate_ordering_version="v1.0",
        adapter_version="1.0.0",
    )


@pytest.fixture
def tsla_replay_context_with_metadata(tsla_geometry, tsla_entry_timestamp):
    """ReplayGateContext with signal metadata PRESENT (current policy — the fix).

    signal_strength=9.0 and confidence_value="high" are properly wired,
    so _is_qualifying_setup() returns True and reduced threshold (0.50) applies.
    """
    return ReplayGateContext(
        # Account state
        account_equity=Decimal("100000"),
        available_cash=Decimal("50000"),
        open_positions=(),
        # Case library (enough for setup quality gate to pass)
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
        # Signal state — THE KEY: metadata IS present in replay (the fix)
        analyst_signal_payload={
            "signal_strength": 9.0,
            "confidence": "high",
            "setup_type": "news_breakout",
        },
        signal_strength=9.0,  # Present in replay (was null in original)
        confidence_value="high",  # Present in replay (was null in original)
        selection_score=8.5,
        execution_score=8.0,
        override_confidence_score=9.0,
        override_reason="Strong news catalyst",
        # Market data
        atr_value=3.5,
        atr_timestamp=tsla_entry_timestamp,
        current_price=250.00,
        # Geometry
        entry_price=tsla_geometry["entry_price"],
        stop_price=tsla_geometry["stop_price"],
        target_price=tsla_geometry["target_price"],
        quantity=tsla_geometry["quantity"],
        max_dollar_risk=200.0,
        # Metadata
        symbol="TSLA",
        profile="aggressive",
        direction="LONG",
        setup_type="news_breakout",
        catalyst_type="news_breakout",
        trade_metadata="Strong earnings beat with guidance raise",
        trade_rationale="TSLA news breakout on earnings",
        atr_source="5min_bars",
        # Catalyst gate fields
        rationale="Strong earnings beat",
        thesis="Momentum continuation on institutional buying",
        indicators=("earnings_beat", "guidance_raise", "volume_surge"),
        quote_timestamp=tsla_entry_timestamp,
        strength="high",
        conviction="high",
    )


@pytest.fixture
def tsla_replay_context_without_metadata(tsla_geometry, tsla_entry_timestamp):
    """ReplayGateContext simulating the ORIGINAL defect.

    signal_strength=None and confidence_value=None — the wiring defect
    means _is_qualifying_setup() returns False, and the default 1.25
    threshold applies, rejecting the candidate with R:R of 1.11.
    """
    return ReplayGateContext(
        # Account state
        account_equity=Decimal("100000"),
        available_cash=Decimal("50000"),
        open_positions=(),
        # Case library
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
        # Signal state — THE DEFECT: metadata is NULL (not wired)
        analyst_signal_payload=None,
        signal_strength=None,  # <-- DROPPED due to wiring defect
        confidence_value=None,  # <-- DROPPED due to wiring defect
        selection_score=8.5,
        execution_score=8.0,
        override_confidence_score=9.0,
        override_reason="Strong news catalyst",
        # Market data
        atr_value=3.5,
        atr_timestamp=tsla_entry_timestamp,
        current_price=250.00,
        # Geometry (same as replay)
        entry_price=tsla_geometry["entry_price"],
        stop_price=tsla_geometry["stop_price"],
        target_price=tsla_geometry["target_price"],
        quantity=tsla_geometry["quantity"],
        max_dollar_risk=200.0,
        # Metadata
        symbol="TSLA",
        profile="aggressive",
        direction="LONG",
        setup_type="news_breakout",
        catalyst_type="news_breakout",
        trade_metadata="Strong earnings beat with guidance raise",
        trade_rationale="TSLA news breakout on earnings",
        atr_source="5min_bars",
        # Catalyst gate fields
        rationale="Strong earnings beat",
        thesis="Momentum continuation on institutional buying",
        indicators=("earnings_beat", "guidance_raise", "volume_surge"),
        quote_timestamp=tsla_entry_timestamp,
        strength="high",
        conviction="high",
    )


# ---------------------------------------------------------------------------
# Test: Original gate used default 1.25 threshold (metadata dropped)
# ---------------------------------------------------------------------------


class TestOriginalGateWithDefaultThreshold:
    """Assert original gate used default 1.25 threshold because signal metadata
    was dropped (metadata_wiring_defect)."""

    def test_original_soft_warns_with_default_threshold(
        self,
        tsla_replay_context_without_metadata,
        tsla_gate_policy_config,
        tsla_policy_version,
        tsla_entry_timestamp,
    ):
        """When signal_strength is None, the gate uses the default 1.25 threshold
        for aggressive profile against R:R of 1.11.

        NOTE: Expected value changed from "reject" to "allow". Commit
        "soften risk geometry rr gates" (1d0300a) made an R:R below the applicable
        threshold a SOFT warning (decision="adjusted_allowed" / canonical "warn",
        risk_geometry_soft_gate=True) rather than a hard rejection. The replay
        adapter normalizes "adjusted_allowed" to "allow". The default 1.25 threshold
        is still the one applied (metadata dropped) — visible in the reason string —
        which is what this test guards.
        """
        replay_clock = build_replay_clock(tsla_entry_timestamp)
        id_provider = build_deterministic_id_provider("test", "risk_geometry_gate", "tsla_001")

        # Evaluate the risk geometry gate directly with missing metadata
        entry = evaluate_single_gate(
            gate_name="risk_geometry_gate",
            context=tsla_replay_context_without_metadata,
            policy_config=tsla_gate_policy_config,
            replay_clock=replay_clock,
            id_provider=id_provider,
        )

        # Softened behavior: R:R below the default 1.25 threshold now soft-warns
        # (adjusted_allowed), which the adapter normalizes to "allow" rather than
        # the old hard "reject".
        assert entry.decision == "allow", (
            f"Expected soft-warn normalized to allow with default threshold, got: {entry.decision}"
        )
        # The reason must still reflect the default 1.25 threshold being applied
        # (i.e. metadata was dropped and the reduced 0.50 threshold was NOT used).
        assert "1.25" in entry.reason_code or "RISK_REWARD" in entry.reason_code, (
            f"Expected reason mentioning 1.25 threshold, got: {entry.reason_code}"
        )

    def test_default_threshold_is_1_25_for_aggressive_tsla(self):
        """Verify the default R:R threshold for aggressive TSLA (high_beta) is 1.25."""
        rule = STOP_DISTANCE_RULES["high_beta_mega_cap_intraday"]
        by_profile = rule.get("min_reward_to_risk_by_profile", {})
        assert by_profile.get("aggressive") == 1.25


# ---------------------------------------------------------------------------
# Test: Current policy correctly receives metadata and uses 0.50 threshold
# ---------------------------------------------------------------------------


class TestCurrentPolicyWithQualifiedThreshold:
    """Assert current policy correctly receives metadata and uses the qualified
    0.50 threshold for aggressive news_breakout with high signal."""

    def test_replay_allows_with_reduced_threshold(
        self,
        tsla_replay_context_with_metadata,
        tsla_gate_policy_config,
        tsla_policy_version,
        tsla_entry_timestamp,
    ):
        """When signal_strength=9.0 and confidence="high", the gate qualifies
        for reduced R:R threshold (0.50) and allows R:R of 1.11."""
        replay_clock = build_replay_clock(tsla_entry_timestamp)
        id_provider = build_deterministic_id_provider("test", "risk_geometry_gate", "tsla_001")

        entry = evaluate_single_gate(
            gate_name="risk_geometry_gate",
            context=tsla_replay_context_with_metadata,
            policy_config=tsla_gate_policy_config,
            replay_clock=replay_clock,
            id_provider=id_provider,
        )

        # Gate should ALLOW because R:R 1.11 > reduced 0.50
        assert entry.decision == "allow", (
            f"Expected allow with reduced threshold, got: {entry.decision} "
            f"(reason: {entry.reason_code})"
        )

    def test_reduced_threshold_is_0_50_for_aggressive(self):
        """Verify the reduced R:R threshold for aggressive profile is 0.50."""
        assert REDUCED_RR_THRESHOLDS_BY_PROFILE["aggressive"] == 0.5

    def test_qualifying_criteria_met(self):
        """Verify the TSLA fixture meets all qualifying criteria."""
        # setup_type in QUALIFYING_SETUP_TYPES
        assert "news_breakout" in QUALIFYING_SETUP_TYPES

        # signal_strength >= QUALIFYING_MIN_SIGNAL_STRENGTH
        assert 9.0 >= QUALIFYING_MIN_SIGNAL_STRENGTH

        # TSLA is in HIGH_BETA_CLUSTER
        assert "TSLA" in HIGH_BETA_CLUSTER


# ---------------------------------------------------------------------------
# Test: Full replay pipeline — delta classification
# ---------------------------------------------------------------------------


class TestDeltaClassification:
    """Assert delta classification is replay_allows_original_reject."""

    def test_full_replay_produces_replay_allows_original_reject(
        self,
        tsla_replay_context_with_metadata,
        tsla_gate_policy_config,
        tsla_policy_version,
        tsla_entry_timestamp,
        tsla_geometry,
    ):
        """Run full replay pipeline and verify delta classification."""
        # Run replay with current policy (metadata present → allows)
        gate_trace = replay_gates(
            context=tsla_replay_context_with_metadata,
            policy_config=tsla_gate_policy_config,
            policy_version=tsla_policy_version,
            replay_id="tsla_reproduction_test",
            candidate_id="tsla_001",
            cutoff=tsla_entry_timestamp,
            diagnostic_mode=False,
        )

        # Replay should allow
        assert gate_trace.final_decision == "allow", (
            f"Replay should allow, got: {gate_trace.final_decision} "
            f"(gate: {gate_trace.final_gate}, reason: {gate_trace.final_reason_code})"
        )

        # Classify delta: original rejected at risk_geometry_gate, replay allows
        delta = classify_delta(
            original_decision="reject",
            original_gate="risk_geometry_gate",
            original_reason_code="RISK_REWARD_BELOW_THRESHOLD",
            original_geometry={
                "entry_price": tsla_geometry["entry_price"],
                "stop_price": tsla_geometry["stop_price"],
                "target_price": tsla_geometry["target_price"],
            },
            original_size=tsla_geometry["quantity"],
            replay_trace=gate_trace,
            replay_classification="exact",
        )

        # Assert delta classification
        assert delta.classification == "replay_allows_original_reject", (
            f"Expected replay_allows_original_reject, got: {delta.classification}"
        )

    def test_delta_original_and_replay_decisions(
        self,
        tsla_replay_context_with_metadata,
        tsla_gate_policy_config,
        tsla_policy_version,
        tsla_entry_timestamp,
        tsla_geometry,
    ):
        """Verify original=reject, replay=allow in the delta."""
        gate_trace = replay_gates(
            context=tsla_replay_context_with_metadata,
            policy_config=tsla_gate_policy_config,
            policy_version=tsla_policy_version,
            replay_id="tsla_reproduction_test",
            candidate_id="tsla_001",
            cutoff=tsla_entry_timestamp,
        )

        delta = classify_delta(
            original_decision="reject",
            original_gate="risk_geometry_gate",
            original_reason_code="RISK_REWARD_BELOW_THRESHOLD",
            original_geometry={
                "entry_price": tsla_geometry["entry_price"],
                "stop_price": tsla_geometry["stop_price"],
                "target_price": tsla_geometry["target_price"],
            },
            original_size=tsla_geometry["quantity"],
            replay_trace=gate_trace,
            replay_classification="exact",
        )

        assert delta.original_decision == "reject"
        assert delta.replay_decision == "allow"


# ---------------------------------------------------------------------------
# Test: Divergence cause is metadata_wiring_defect
# ---------------------------------------------------------------------------


class TestDivergenceCause:
    """Assert divergence cause is metadata_wiring_defect."""

    def test_divergence_cause_is_metadata_wiring_defect(
        self,
        tsla_replay_context_with_metadata,
        tsla_gate_policy_config,
        tsla_policy_version,
        tsla_entry_timestamp,
        tsla_geometry,
    ):
        """The divergence cause should be metadata_wiring_defect because
        signal_strength was null in the original (dropped by wiring)."""
        gate_trace = replay_gates(
            context=tsla_replay_context_with_metadata,
            policy_config=tsla_gate_policy_config,
            policy_version=tsla_policy_version,
            replay_id="tsla_reproduction_test",
            candidate_id="tsla_001",
            cutoff=tsla_entry_timestamp,
        )

        delta = classify_delta(
            original_decision="reject",
            original_gate="risk_geometry_gate",
            original_reason_code="RISK_REWARD_BELOW_THRESHOLD",
            original_geometry={
                "entry_price": tsla_geometry["entry_price"],
                "stop_price": tsla_geometry["stop_price"],
                "target_price": tsla_geometry["target_price"],
            },
            original_size=tsla_geometry["quantity"],
            replay_trace=gate_trace,
            replay_classification="exact",
        )

        assert delta.divergence_cause == "metadata_wiring_defect", (
            f"Expected metadata_wiring_defect, got: {delta.divergence_cause}"
        )


# ---------------------------------------------------------------------------
# Test: First diverging gate identifies risk_geometry_gate
# ---------------------------------------------------------------------------


class TestFirstDivergingGate:
    """Assert first diverging gate identifies risk_geometry_gate."""

    def test_first_diverging_gate_is_risk_geometry_gate(
        self,
        tsla_replay_context_with_metadata,
        tsla_gate_policy_config,
        tsla_policy_version,
        tsla_entry_timestamp,
        tsla_geometry,
    ):
        """The first (and only) diverging gate is risk_geometry_gate because
        that's where the missing metadata caused a different decision."""
        gate_trace = replay_gates(
            context=tsla_replay_context_with_metadata,
            policy_config=tsla_gate_policy_config,
            policy_version=tsla_policy_version,
            replay_id="tsla_reproduction_test",
            candidate_id="tsla_001",
            cutoff=tsla_entry_timestamp,
        )

        delta = classify_delta(
            original_decision="reject",
            original_gate="risk_geometry_gate",
            original_reason_code="RISK_REWARD_BELOW_THRESHOLD",
            original_geometry={
                "entry_price": tsla_geometry["entry_price"],
                "stop_price": tsla_geometry["stop_price"],
                "target_price": tsla_geometry["target_price"],
            },
            original_size=tsla_geometry["quantity"],
            replay_trace=gate_trace,
            replay_classification="exact",
        )

        assert delta.first_diverging_gate == "risk_geometry_gate", (
            f"Expected risk_geometry_gate, got: {delta.first_diverging_gate}"
        )


# ---------------------------------------------------------------------------
# Test: End-to-end pipeline — snapshot → reconstruct → replay → classify → report
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    """Verify the complete end-to-end pipeline integration."""

    def test_full_pipeline_snapshot_to_report(
        self,
        tsla_replay_context_with_metadata,
        tsla_replay_context_without_metadata,
        tsla_gate_policy_config,
        tsla_policy_version,
        tsla_entry_timestamp,
        tsla_geometry,
    ):
        """End-to-end: simulate original (no metadata) → replay (with metadata)
        → classify delta → verify all expected fields in the report."""

        # --- Step 1: Original evaluation (simulates the snapshot state) ---
        # The original had signal_strength=None, so it gets rejected
        original_trace = replay_gates(
            context=tsla_replay_context_without_metadata,
            policy_config=tsla_gate_policy_config,
            policy_version=tsla_policy_version,
            replay_id="original_evaluation",
            candidate_id="tsla_001",
            cutoff=tsla_entry_timestamp,
            diagnostic_mode=False,
        )
        # Expected value changed from "reject" to "allow": commit
        # "soften risk geometry rr gates" (1d0300a) turned an R:R-below-threshold
        # outcome into a soft warning (adjusted_allowed / canonical "warn") that the
        # adapter normalizes to "allow", so the live pipeline no longer HARD-rejects
        # the missing-metadata path. The historical hard rejection is still exercised
        # via the explicit original_decision="reject" passed to classify_delta below.
        assert original_trace.final_decision == "allow"
        rg_entry = next(
            e for e in original_trace.entries if e.gate_name == "risk_geometry_gate"
        )
        # The default 1.25 threshold is still applied (soft R:R warning) since
        # metadata was dropped.
        assert "1.25" in rg_entry.reason_code or "RISK_REWARD" in rg_entry.reason_code

        # --- Step 2: Replay evaluation (current policy with metadata) ---
        replay_trace = replay_gates(
            context=tsla_replay_context_with_metadata,
            policy_config=tsla_gate_policy_config,
            policy_version=tsla_policy_version,
            replay_id="replay_evaluation",
            candidate_id="tsla_001",
            cutoff=tsla_entry_timestamp,
            diagnostic_mode=False,
        )
        assert replay_trace.final_decision == "allow"

        # --- Step 3: Classify delta ---
        delta = classify_delta(
            original_decision="reject",
            original_gate="risk_geometry_gate",
            original_reason_code="RISK_REWARD_BELOW_THRESHOLD",
            original_geometry={
                "entry_price": tsla_geometry["entry_price"],
                "stop_price": tsla_geometry["stop_price"],
                "target_price": tsla_geometry["target_price"],
            },
            original_size=tsla_geometry["quantity"],
            replay_trace=replay_trace,
            replay_classification="exact",
        )

        # --- Step 4: Verify all expected outcomes ---
        # Classification
        assert delta.classification == "replay_allows_original_reject"
        # Decisions
        assert delta.original_decision == "reject"
        assert delta.replay_decision == "allow"
        # First diverging gate
        assert delta.first_diverging_gate == "risk_geometry_gate"
        # Divergence cause
        assert delta.divergence_cause == "metadata_wiring_defect"
        # Divergence evidence should be present
        assert delta.divergence_evidence is not None

    def test_rr_ratio_is_between_thresholds(self, tsla_geometry):
        """Verify the fixture R:R is between 0.50 and 1.25 (the key invariant)."""
        entry = float(tsla_geometry["entry_price"])
        stop = float(tsla_geometry["stop_price"])
        target = float(tsla_geometry["target_price"])

        stop_distance = abs(entry - stop)
        target_distance = target - entry  # LONG direction

        rr = target_distance / stop_distance

        # R:R must be above reduced threshold (0.50) but below default (1.25)
        assert rr > 0.50, f"R:R {rr:.4f} should be > 0.50"
        assert rr < 1.25, f"R:R {rr:.4f} should be < 1.25"
        # Approximately 1.11
        assert abs(rr - 1.112) < 0.01, f"Expected R:R ~1.11, got {rr:.4f}"

    def test_gate_trace_entries_complete(
        self,
        tsla_replay_context_with_metadata,
        tsla_gate_policy_config,
        tsla_policy_version,
        tsla_entry_timestamp,
    ):
        """Verify the gate trace contains all expected gate entries when allowed."""
        trace = replay_gates(
            context=tsla_replay_context_with_metadata,
            policy_config=tsla_gate_policy_config,
            policy_version=tsla_policy_version,
            replay_id="trace_test",
            candidate_id="tsla_001",
            cutoff=tsla_entry_timestamp,
        )

        # All 4 core gates should have been evaluated (all passed)
        gate_names = [e.gate_name for e in trace.entries]
        assert "setup_quality_gate" in gate_names
        assert "pre_trade_quality_gate" in gate_names
        assert "catalyst_specificity_gate" in gate_names
        assert "risk_geometry_gate" in gate_names

        # All gates should allow
        for entry in trace.entries:
            assert entry.decision == "allow", (
                f"Gate {entry.gate_name} should allow, got: {entry.decision} "
                f"(reason: {entry.reason_code})"
            )

    def test_diagnostic_mode_evaluates_all_gates(
        self,
        tsla_replay_context_without_metadata,
        tsla_gate_policy_config,
        tsla_policy_version,
        tsla_entry_timestamp,
    ):
        """In diagnostic mode, all four core gates are evaluated and recorded."""
        trace = replay_gates(
            context=tsla_replay_context_without_metadata,
            policy_config=tsla_gate_policy_config,
            policy_version=tsla_policy_version,
            replay_id="diagnostic_test",
            candidate_id="tsla_001",
            cutoff=tsla_entry_timestamp,
            diagnostic_mode=True,
        )

        # Expected value changed from "reject" to "allow": commit
        # "soften risk geometry rr gates" (1d0300a) made the R:R-below-threshold
        # outcome a soft warning (adjusted_allowed / canonical "warn") that the
        # adapter normalizes to "allow", so the missing-metadata path no longer
        # hard-rejects at risk_geometry_gate.
        assert trace.final_decision == "allow"
        assert trace.diagnostic_mode is True

        # All 4 gates should be in the trace (diagnostic evaluates all)
        gate_names = [e.gate_name for e in trace.entries]
        assert len(gate_names) == 4
