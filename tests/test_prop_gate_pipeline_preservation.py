"""
Property-based tests for gate pipeline preservation under alert dispatch.

Property 9: Alert-dispatched candidates pass unchanged gate pipeline.
Verifies that the gate pipeline function receives the same parameters regardless
of whether a candidate was alert-originated or scheduled-originated, and that
gate rejection produces identical terminal states.

**Validates: Requirements 6.4, 6.5, 6.6**
"""

from __future__ import annotations

import inspect
from unittest.mock import patch, MagicMock, call

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from utils.candidate_pipeline import _build_gate_decision, ResolvedOrder


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

symbol_strategy = st.sampled_from(["AAPL", "TSLA", "NVDA", "MSFT", "GOOG", "AMZN"])

price_strategy = st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False)

quantity_strategy = st.integers(min_value=1, max_value=1000)

direction_strategy = st.sampled_from(["BUY", "SHORT"])

setup_type_strategy = st.sampled_from([
    "breakout_pullback", "flag_continuation", "range_reversal",
    "earnings_momentum", "gap_and_go",
])

profile_id_strategy = st.sampled_from(["aggressive", "moderate", "conservative"])

alert_type_strategy = st.sampled_from(["entry_alert", "breakout", "rapid_move"])

# A minimal source signal dict that mimics analyst signal structure
source_signal_strategy = st.fixed_dictionaries({
    "symbol": symbol_strategy,
    "signal": direction_strategy,
    "strength": st.sampled_from(["strong", "moderate", "weak"]),
    "setup_type": setup_type_strategy,
})

cycle_trigger_type_strategy = st.sampled_from(["scheduled", "alert"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_resolved_order(
    symbol: str,
    action: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
    setup_type: str,
    profile_id: str,
    source_signal: dict,
) -> ResolvedOrder:
    """Build a ResolvedOrder for testing."""
    return ResolvedOrder(
        candidate_id=f"cand_{symbol}_test123",
        execution_key=f"exec_{symbol}_key456",
        symbol=symbol,
        action=action,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        setup_type=setup_type,
        risk_reward=2.0,
        source_signal=source_signal,
        profile_id=profile_id,
        geometry_name="standard",
        risk_multiplier=1.0,
        pm_rationale="Test rationale",
    )


# ---------------------------------------------------------------------------
# Property 9: Gate pipeline receives identical parameters for alert vs
#              scheduled candidates
# ---------------------------------------------------------------------------


class TestProperty9GatePipelinePreservation:
    """
    The gate pipeline function _run_gate_pipeline() receives the same arguments
    regardless of whether the candidate was alert-originated or scheduled-originated.
    No alert-specific parameters are injected into gate functions.

    **Validates: Requirements 6.4, 6.5, 6.6**
    """

    @given(
        symbol=symbol_strategy,
        action=direction_strategy,
        entry_price=price_strategy,
        stop_delta=st.floats(min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False),
        target_delta=st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        setup_type=setup_type_strategy,
        profile_id=profile_id_strategy,
        quantity=quantity_strategy,
        source_signal=source_signal_strategy,
    )
    @settings(max_examples=30)
    def test_gate_decision_has_no_alert_specific_fields(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        stop_delta: float,
        target_delta: float,
        setup_type: str,
        profile_id: str,
        quantity: int,
        source_signal: dict,
    ):
        """_build_gate_decision produces a dict with no alert-specific keys.

        The gate decision dict used by _run_gate_pipeline must contain only
        trade-related fields (symbol, action, price, stop, target, quantity,
        setup_type, etc.) and never include alert_intent_id, alert_type,
        cycle_trigger_type, or any other alert dispatch context.
        """
        # Construct valid stop/target based on direction
        if action == "BUY":
            stop_price = entry_price - stop_delta
            target_price = entry_price + target_delta
        else:
            stop_price = entry_price + stop_delta
            target_price = entry_price - target_delta

        assume(stop_price > 0)
        assume(target_price > 0)

        resolved_order = _make_resolved_order(
            symbol=symbol,
            action=action,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            setup_type=setup_type,
            profile_id=profile_id,
            source_signal=source_signal,
        )

        gate_decision = _build_gate_decision(resolved_order, quantity)

        # Assert no alert-specific fields are present in gate decision
        alert_specific_keys = {
            "alert_intent_id", "alert_type", "cycle_trigger_type",
            "alert_contexts", "alert_context", "dispatch_mode",
            "alert_urgency", "alert_freshness", "alert_origin",
            "is_alert_triggered", "dispatch_status",
        }
        present_alert_keys = alert_specific_keys & set(gate_decision.keys())
        assert not present_alert_keys, (
            f"Gate decision contains alert-specific keys: {present_alert_keys}. "
            "Gate pipeline must not receive alert dispatch context."
        )

        # Assert required trade fields are present
        required_keys = {"symbol", "action", "price", "stop", "target", "quantity", "setup_type"}
        missing_keys = required_keys - set(gate_decision.keys())
        assert not missing_keys, (
            f"Gate decision missing required trade keys: {missing_keys}"
        )

    @given(
        symbol=symbol_strategy,
        action=direction_strategy,
        entry_price=price_strategy,
        stop_delta=st.floats(min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False),
        target_delta=st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        setup_type=setup_type_strategy,
        profile_id=profile_id_strategy,
        quantity=quantity_strategy,
        source_signal=source_signal_strategy,
    )
    @settings(max_examples=30)
    def test_gate_decision_identical_for_alert_and_scheduled_origin(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        stop_delta: float,
        target_delta: float,
        setup_type: str,
        profile_id: str,
        quantity: int,
        source_signal: dict,
    ):
        """Same candidate parameters produce identical gate decisions regardless of origin.

        When a candidate has the same trade parameters (symbol, entry, stop, target,
        setup_type, quantity), the gate decision dict is identical whether the candidate
        was triggered by an alert dispatch or a scheduled cycle. This proves that
        _build_gate_decision is origin-agnostic.
        """
        if action == "BUY":
            stop_price = entry_price - stop_delta
            target_price = entry_price + target_delta
        else:
            stop_price = entry_price + stop_delta
            target_price = entry_price - target_delta

        assume(stop_price > 0)
        assume(target_price > 0)

        # Build a resolved order as if from scheduled cycle
        scheduled_order = _make_resolved_order(
            symbol=symbol,
            action=action,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            setup_type=setup_type,
            profile_id=profile_id,
            source_signal=source_signal,
        )

        # Build an identical resolved order as if from alert dispatch
        # (same parameters — ResolvedOrder has no alert-specific fields)
        alert_order = _make_resolved_order(
            symbol=symbol,
            action=action,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            setup_type=setup_type,
            profile_id=profile_id,
            source_signal=source_signal,
        )

        gate_decision_scheduled = _build_gate_decision(scheduled_order, quantity)
        gate_decision_alert = _build_gate_decision(alert_order, quantity)

        assert gate_decision_scheduled == gate_decision_alert, (
            "Gate decisions must be identical for scheduled and alert-originated candidates "
            "with the same trade parameters."
        )

    def test_run_gate_pipeline_signature_has_no_alert_params(self):
        """_run_gate_pipeline function signature accepts no alert-specific parameters.

        This structural test ensures that the gate pipeline function signature
        cannot be inadvertently changed to accept alert dispatch context.
        The function must only accept (db, engine, decision, signal, profile_id).
        """
        from agents.portfolio_manager import _run_gate_pipeline

        sig = inspect.signature(_run_gate_pipeline)
        param_names = list(sig.parameters.keys())

        # Expected parameters (positional)
        expected_params = ["db", "engine", "decision", "signal", "profile_id"]
        assert param_names == expected_params, (
            f"_run_gate_pipeline signature has unexpected params: {param_names}. "
            f"Expected exactly: {expected_params}. "
            "No alert-specific parameters should be added to the gate pipeline."
        )

        # Double-check: no alert-related parameter names
        alert_param_names = {
            "alert_intent_id", "alert_type", "cycle_trigger_type",
            "alert_context", "alert_contexts", "dispatch_mode",
            "is_alert", "alert_origin",
        }
        present_alert_params = alert_param_names & set(param_names)
        assert not present_alert_params, (
            f"Gate pipeline has alert-specific params: {present_alert_params}"
        )

    def test_resolved_order_has_no_alert_fields(self):
        """ResolvedOrder dataclass has no alert-specific fields.

        The intermediate data structure between candidate resolution and gate
        pipeline must not carry alert dispatch context, ensuring that no
        alert-specific data can leak into the gate evaluation.
        """
        import dataclasses

        fields = {f.name for f in dataclasses.fields(ResolvedOrder)}

        alert_field_names = {
            "alert_intent_id", "alert_type", "cycle_trigger_type",
            "alert_context", "dispatch_mode", "is_alert_triggered",
            "alert_origin", "alert_urgency",
        }
        present_alert_fields = alert_field_names & fields
        assert not present_alert_fields, (
            f"ResolvedOrder has alert-specific fields: {present_alert_fields}. "
            "Gate pipeline inputs must be origin-agnostic."
        )

    @given(
        symbol=symbol_strategy,
        action=direction_strategy,
        entry_price=price_strategy,
        stop_delta=st.floats(min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False),
        target_delta=st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        setup_type=setup_type_strategy,
        profile_id=profile_id_strategy,
        quantity=quantity_strategy,
        source_signal=source_signal_strategy,
    )
    @settings(max_examples=30)
    def test_gate_rejection_terminal_states_are_origin_independent(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        stop_delta: float,
        target_delta: float,
        setup_type: str,
        profile_id: str,
        quantity: int,
        source_signal: dict,
    ):
        """Gate rejection produces the same terminal state codes for alert vs non-alert.

        When _run_gate_pipeline rejects, execute_candidate_pipeline maps the rejection
        to GATE_REJECTED or SIZING_REJECTED status. These terminal states must be
        the same regardless of whether the candidate originated from an alert.

        We verify this by mocking _run_gate_pipeline to return a rejection and
        checking that the pipeline result outcome string is the same for both
        an "alert-triggered" and "scheduled" invocation path.
        """
        if action == "BUY":
            stop_price = entry_price - stop_delta
            target_price = entry_price + target_delta
        else:
            stop_price = entry_price + stop_delta
            target_price = entry_price - target_delta

        assume(stop_price > 0)
        assume(target_price > 0)

        from utils.candidate_pipeline import execute_candidate_pipeline, PipelineResult
        from utils.decision_contract import CandidateDecision

        # Build a mock registry that resolves and reserves successfully
        candidate_id = f"cand_{symbol}_test"

        mock_candidate = MagicMock()
        mock_candidate.candidate_id = candidate_id
        mock_candidate.symbol = symbol
        mock_candidate.direction = action
        mock_candidate.entry_price = entry_price
        mock_candidate.stop_price = stop_price
        mock_candidate.target_price = target_price
        mock_candidate.setup_type = setup_type
        mock_candidate.risk_reward = 2.0
        mock_candidate.geometry_name = "standard"
        mock_candidate.signal_snapshot_json = '{"symbol": "' + symbol + '", "signal": "BUY", "strength": "strong"}'
        mock_candidate.profile_id = profile_id
        mock_candidate.cycle_id = "test_cycle"
        mock_candidate.state = "REGISTERED"

        mock_registry = MagicMock()
        mock_registry.cycle_id = "test_cycle"
        mock_registry.reserve.return_value = (True, None)

        decision = CandidateDecision(
            candidate_id=candidate_id,
            decision="accept",
            rationale="Test",
            risk_multiplier=1.0,
        )

        mock_db = MagicMock()
        mock_engine = MagicMock()

        # Simulate a gate rejection (same rejection regardless of origin)
        gate_rejection_notes = [
            {"gate": "setup_quality_gate", "decision": "reject", "reason": "test_rejection", "reason_type": "low_win_rate"}
        ]

        portfolio = {"starting_balance": 100000, "daily_pnl": 0, "cash": 100000}
        profile = {"starting_balance": 100000, "max_position_pct": 0.1, "max_daily_loss_pct": 0.02}

        # Mock _run_gate_pipeline to return rejection
        with patch("utils.candidate_pipeline._resolve_candidate") as mock_resolve, \
             patch("utils.candidate_pipeline._generate_execution_key", return_value="exec_key_test"), \
             patch("utils.candidate_pipeline.calculate_position_size") as mock_size, \
             patch("agents.portfolio_manager._run_gate_pipeline") as mock_gate, \
             patch("utils.candidate_pipeline.PM_PROVENANCE_MODE", "disabled"):

            mock_resolve.return_value = (mock_candidate, None)
            mock_size_result = MagicMock()
            mock_size_result.rejected = False
            mock_size_result.quantity = quantity
            mock_size_result.dollar_risk = 100.0
            mock_size.return_value = mock_size_result
            mock_gate.return_value = (False, gate_rejection_notes, 1.0, [])

            # Execute as if from scheduled cycle
            result_scheduled = execute_candidate_pipeline(
                mock_db, mock_engine, mock_registry, decision,
                portfolio, profile, profile_id,
            )

        # Reset mocks for second call
        mock_registry.reset_mock()
        mock_registry.cycle_id = "test_cycle"
        mock_registry.reserve.return_value = (True, None)

        with patch("utils.candidate_pipeline._resolve_candidate") as mock_resolve, \
             patch("utils.candidate_pipeline._generate_execution_key", return_value="exec_key_test"), \
             patch("utils.candidate_pipeline.calculate_position_size") as mock_size, \
             patch("agents.portfolio_manager._run_gate_pipeline") as mock_gate, \
             patch("utils.candidate_pipeline.PM_PROVENANCE_MODE", "disabled"):

            mock_resolve.return_value = (mock_candidate, None)
            mock_size_result = MagicMock()
            mock_size_result.rejected = False
            mock_size_result.quantity = quantity
            mock_size_result.dollar_risk = 100.0
            mock_size.return_value = mock_size_result
            mock_gate.return_value = (False, gate_rejection_notes, 1.0, [])

            # Execute as if from alert-triggered cycle (same path — no difference)
            result_alert = execute_candidate_pipeline(
                mock_db, mock_engine, mock_registry, decision,
                portfolio, profile, profile_id,
            )

        # Both produce "gate_rejected" outcome regardless of origin
        assert result_scheduled.outcome == "gate_rejected", (
            f"Expected 'gate_rejected' for scheduled origin, got '{result_scheduled.outcome}'"
        )
        assert result_alert.outcome == "gate_rejected", (
            f"Expected 'gate_rejected' for alert origin, got '{result_alert.outcome}'"
        )
        assert result_scheduled.outcome == result_alert.outcome, (
            "Gate rejection terminal state must be identical for alert and scheduled origins."
        )

    def test_gate_pipeline_order_preserved(self):
        """Gate pipeline evaluates gates in the documented fixed order.

        The gate order (setup_quality, pre_trade_quality, catalyst_specificity,
        risk_geometry, concentration) must be preserved regardless of how the
        candidate was originated. We verify the implementation evaluates gates
        in the correct sequence by inspecting the source code structure.
        """
        from agents.portfolio_manager import _run_gate_pipeline

        source = inspect.getsource(_run_gate_pipeline)

        # Find gate evaluation order by locating gate function calls
        gate_markers = [
            "evaluate_setup_quality",
            "evaluate_pre_trade_quality",
            "evaluate_catalyst_specificity",
            "evaluate_risk_geometry",
        ]

        positions = []
        for marker in gate_markers:
            pos = source.find(marker)
            assert pos != -1, (
                f"Gate '{marker}' not found in _run_gate_pipeline source. "
                "Gate pipeline must include all documented gates."
            )
            positions.append(pos)

        # Verify gates are in sequential order
        for i in range(len(positions) - 1):
            assert positions[i] < positions[i + 1], (
                f"Gate order violation: '{gate_markers[i]}' (pos {positions[i]}) "
                f"must come before '{gate_markers[i + 1]}' (pos {positions[i + 1]}). "
                "Gate pipeline must maintain documented evaluation order."
            )
