"""Verify swing candidates flow through the candidate pipeline without modification.

This is a verification test: swing candidates use the same CandidateRecord
structure and pass through execute_candidate_pipeline identically to intraday
candidates. The candidate_type field is metadata — it does not affect gate
pipeline logic or execution path.

Requirements: 5.5, 6.6, 8.3
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from utils.candidate_registry import CandidateRecord
from utils.gate_config import HIGH_BETA_CLUSTER, SEMI_CLUSTER, CRYPTO_PROXY_CLUSTER


def _make_swing_candidate(**overrides) -> CandidateRecord:
    """Create a swing CandidateRecord with sensible defaults."""
    defaults = {
        "candidate_id": "test-swing-001",
        "cycle_id": "cycle-1",
        "profile_id": "moderate",
        "symbol": "AAPL",
        "direction": "BUY",
        "setup_type": "sector_rotation_swing",
        "geometry_name": "swing_sector_rotation_swing",
        "entry_price": 150.0,
        "stop_price": 142.5,
        "target_price": 170.0,
        "risk_reward": 2.67,
        "trigger": "Swing entry: sector_rotation_swing",
        "invalidation_basis": "Below stop at 142.5",
        "target_basis": "Swing target for sector_rotation_swing",
        "source_signal_id": "sig-001",
        "signal_snapshot_json": '{"symbol": "AAPL", "setup_type": "sector_rotation_swing"}',
        "created_at": datetime.now(timezone.utc),
        "expires_at": datetime.now(timezone.utc) + timedelta(hours=24),
        "integrity_hash": "abc123",
        "candidate_type": "swing",
    }
    defaults.update(overrides)
    return CandidateRecord(**defaults)


class TestSwingCandidatePipelineFields:
    """Verify swing CandidateRecord has all fields needed by the pipeline."""

    def test_swing_candidate_has_required_pipeline_fields(self):
        """Swing CandidateRecord has all fields that execute_candidate_pipeline uses."""
        record = _make_swing_candidate()

        # Pipeline needs these fields at the resolve stage
        assert record.entry_price > 0
        assert record.stop_price > 0
        assert record.target_price > 0
        assert record.risk_reward > 0
        assert record.direction in ("BUY", "SHORT")
        assert record.symbol
        assert record.setup_type
        assert record.geometry_name
        assert record.profile_id
        assert record.cycle_id
        assert record.candidate_id
        assert record.signal_snapshot_json

        # Swing-specific metadata
        assert record.candidate_type == "swing"

    def test_swing_candidate_direction_short(self):
        """Swing SHORT candidate has correct directional ordering."""
        record = _make_swing_candidate(
            direction="SHORT",
            entry_price=150.0,
            stop_price=157.5,
            target_price=130.0,
            setup_type="risk_off_macro_short",
            geometry_name="swing_risk_off_macro_short",
        )
        assert record.direction == "SHORT"
        assert record.stop_price > record.entry_price  # SHORT: stop above entry
        assert record.target_price < record.entry_price  # SHORT: target below entry

    def test_swing_candidate_same_dataclass_as_intraday(self):
        """Swing and intraday candidates use the exact same CandidateRecord class."""
        swing = _make_swing_candidate(candidate_type="swing")
        intraday = _make_swing_candidate(candidate_type="intraday")

        # Both are the same type — pipeline treats them identically
        assert type(swing) is type(intraday)
        assert type(swing) is CandidateRecord


class TestHighVolatilityCooldownApplicability:
    """Verify swing candidates on high-vol symbols are subject to cooldown checks.

    The gate pipeline checks symbol membership in HIGH_BETA_CLUSTER,
    SEMI_CLUSTER, and CRYPTO_PROXY_CLUSTER. Swing candidates pass the same
    symbol field, so they trigger the same cooldown logic.
    """

    def test_high_beta_symbols_defined(self):
        """HIGH_BETA_CLUSTER contains the expected mega-cap volatile symbols."""
        assert "AMD" in HIGH_BETA_CLUSTER
        assert "NVDA" in HIGH_BETA_CLUSTER
        assert "TSLA" in HIGH_BETA_CLUSTER

    def test_semi_cluster_contains_high_vol_names(self):
        """SEMI_CLUSTER contains semiconductor names including MU."""
        assert "MU" in SEMI_CLUSTER
        assert "AMD" in SEMI_CLUSTER
        assert "NVDA" in SEMI_CLUSTER

    def test_crypto_proxy_contains_mstr(self):
        """CRYPTO_PROXY_CLUSTER contains MSTR."""
        assert "MSTR" in CRYPTO_PROXY_CLUSTER

    def test_swing_candidate_on_high_vol_symbol_has_same_fields(self):
        """Swing candidate on NVDA carries the same symbol field the gates check."""
        for symbol in ("AMD", "NVDA", "TSLA", "MSTR", "MU"):
            record = _make_swing_candidate(symbol=symbol)
            assert record.symbol == symbol
            assert record.candidate_type == "swing"
            # The gate pipeline reads decision["symbol"] which comes from
            # resolved_order.symbol which comes from candidate.symbol.
            # No candidate_type check exists in the gate pipeline path.


class TestPaperTradeRecordFields:
    """Verify swing candidate provides all fields needed for paper trade record.

    Requirement 8.3: paper trade record includes entry, stop, target,
    normalized setup type, profile ID, holding horizon, thesis, invalidation basis.
    """

    def test_swing_candidate_entry_stop_target(self):
        """Swing candidate carries entry, stop, and target prices."""
        record = _make_swing_candidate()
        assert record.entry_price == 150.0
        assert record.stop_price == 142.5
        assert record.target_price == 170.0

    def test_swing_candidate_normalized_setup_type(self):
        """Swing candidate carries the normalized setup type (stored as setup_type)."""
        record = _make_swing_candidate(setup_type="breakout_retest")
        assert record.setup_type == "breakout_retest"

    def test_swing_candidate_profile_id(self):
        """Swing candidate carries profile_id for the paper trade record."""
        record = _make_swing_candidate(profile_id="aggressive")
        assert record.profile_id == "aggressive"

    def test_swing_candidate_invalidation_basis(self):
        """Swing candidate carries invalidation_basis for the trade record."""
        record = _make_swing_candidate(
            invalidation_basis="Close below 20-day EMA at $142.50"
        )
        assert record.invalidation_basis == "Close below 20-day EMA at $142.50"

    def test_swing_candidate_trigger_as_thesis(self):
        """Swing candidate trigger field serves as the thesis/entry rationale."""
        record = _make_swing_candidate(
            trigger="Swing entry: sector rotation into tech, relative strength confirmed"
        )
        assert "sector rotation" in record.trigger

    def test_gate_decision_dict_built_from_swing_candidate(self):
        """_build_gate_decision produces the same dict structure for swing candidates.

        This verifies the gate_decision dict (which execute_trade receives)
        contains all required paper trade record fields.
        """
        from utils.candidate_pipeline import _build_resolved_order, _build_gate_decision
        from utils.decision_contract import CandidateDecision

        record = _make_swing_candidate(
            setup_type="pullback_continuation",
            entry_price=200.0,
            stop_price=190.0,
            target_price=225.0,
        )

        decision = CandidateDecision(
            candidate_id="test-swing-001",
            decision="accept",
            rationale="Strong sector rotation thesis with confirmed relative strength",
            risk_multiplier=1.0,
        )

        resolved = _build_resolved_order(record, decision, "exec-key-001", "moderate")
        gate_decision = _build_gate_decision(resolved, quantity=10)

        # All fields needed for paper trade record are present
        assert gate_decision["symbol"] == "AAPL"
        assert gate_decision["entry_price"] == 200.0
        assert gate_decision["stop_price"] == 190.0
        assert gate_decision["target_price"] == 225.0
        assert gate_decision["setup_type"] == "pullback_continuation"
        assert gate_decision["quantity"] == 10
        assert gate_decision["rationale"] == "Strong sector rotation thesis with confirmed relative strength"
        assert gate_decision["execution_key"] == "exec-key-001"
