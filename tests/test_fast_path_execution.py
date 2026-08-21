"""Tests for fast-path execution delegation.

Validates:
- trade_executed outcome delegates to gate pipeline and execute_trade
- trade_executed uses frozen geometry from trigger, not fresh signal data
- Trigger without source_signal_id or source_watch_id -> rejected (no_strategy_provenance)
- pending_order_created delegates to pending_order_creation when mode enabled
- pending_order_created records intent only when mode not enabled
- watch_created creates a watch candidate row
- Execution failure does not crash monitor — logged and event updated

Requirements: 10.2-10.5, 10.9-10.11, 4.4, cross-cutting acceptance test 2
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

from utils.fast_path_evaluator import FastPathOutcome
from utils.fast_path_execution import (
    execute_fast_path_pending_order,
    execute_fast_path_trade,
    execute_fast_path_watch,
)
from utils.fast_path_registry import TriggerRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trigger(**overrides) -> TriggerRecord:
    """Build a minimally valid SHORT momentum_fade TriggerRecord with provenance."""
    defaults = {
        "trigger_id": str(uuid.uuid4()),
        "symbol": "TSLA",
        "profile_id": "moderate",
        "direction": "SHORT",
        "setup_type": "momentum_fade",
        "trigger_type": "entry_zone",
        "trigger_level": 351.61,
        "trigger_zone_upper": 352.00,
        "trigger_zone_lower": 350.50,
        "entry_price": 351.61,
        "stop_price": 355.00,
        "target_price": 348.00,
        "geometry_name": "momentum_fade_short",
        "source_signal_id": "sig-001",
        "source_watch_id": None,
        "invalidation_basis": "close above 355",
        "target_basis": "prior support",
        "state": "active",
        "registered_at": "2026-08-20T14:30:00+00:00",
        "expires_at": "2026-08-20T14:35:00+00:00",
        "signal_snapshot_json": '{"setup_type":"momentum_fade","entry":351.61}',
        "context_json": None,
    }
    defaults.update(overrides)
    return TriggerRecord(**defaults)


def _make_outcome(trigger: TriggerRecord, **overrides) -> FastPathOutcome:
    """Build a FastPathOutcome matching the given trigger."""
    defaults = {
        "outcome_type": "trade_executed",
        "outcome_reason_code": "all_gates_passed",
        "trigger_id": trigger.trigger_id,
        "symbol": trigger.symbol,
        "profile_id": trigger.profile_id,
        "direction": trigger.direction,
        "setup_type": trigger.setup_type,
        "current_price": 350.50,
        "entry_price": trigger.entry_price,
        "stop_price": trigger.stop_price,
        "target_price": trigger.target_price,
        "reward_to_risk": 1.06,
    }
    defaults.update(overrides)
    return FastPathOutcome(**defaults)


# ---------------------------------------------------------------------------
# Tests: execute_fast_path_trade — delegation to gate pipeline/execute_trade
# ---------------------------------------------------------------------------


class TestExecuteFastPathTradeDelegation:
    """trade_executed outcome delegates to gate pipeline and execute_trade."""

    @patch("utils.fast_path_execution.PM_CANDIDATE_MODE", "enabled")
    @patch("utils.fast_path_execution.FAST_PATH_MODE", "enabled")
    def test_delegates_to_candidate_pipeline_when_candidate_mode_enabled(self):
        """When PM_CANDIDATE_MODE is enabled, uses execute_candidate_pipeline."""
        trigger = _make_trigger()
        outcome = _make_outcome(trigger)

        mock_pipeline_result = MagicMock()
        mock_pipeline_result.outcome = "executed"
        mock_pipeline_result.error = None

        with patch(
            "utils.candidate_pipeline.execute_candidate_pipeline",
            return_value=mock_pipeline_result,
        ) as mock_pipeline, patch(
            "utils.candidate_registry.CandidateRegistry"
        ) as mock_registry_cls:
            # Set up registry mock
            mock_registry = MagicMock()
            mock_registry_cls.return_value = mock_registry

            result = execute_fast_path_trade(outcome, trigger, MagicMock(), MagicMock())

        assert result is True
        mock_pipeline.assert_called_once()

    @patch("utils.fast_path_execution.PM_CANDIDATE_MODE", "disabled")
    @patch("utils.fast_path_execution.FAST_PATH_MODE", "enabled")
    def test_delegates_to_execute_trade_when_candidate_mode_disabled(self):
        """When PM_CANDIDATE_MODE is disabled, calls execute_trade directly."""
        trigger = _make_trigger()
        outcome = _make_outcome(trigger)

        with patch(
            "agents.portfolio_manager.execute_trade",
            return_value=(True, "Trade executed"),
        ) as mock_execute:
            result = execute_fast_path_trade(outcome, trigger, MagicMock(), MagicMock())

        assert result is True
        mock_execute.assert_called_once()
        # Verify normalized=True is passed
        call_kwargs = mock_execute.call_args[1]
        assert call_kwargs["normalized"] is True

    @patch("utils.fast_path_execution.PM_CANDIDATE_MODE", "disabled")
    @patch("utils.fast_path_execution.FAST_PATH_MODE", "enabled")
    def test_execute_trade_failure_returns_false(self):
        """When execute_trade returns failure, execute_fast_path_trade returns False."""
        trigger = _make_trigger()
        outcome = _make_outcome(trigger)

        with patch(
            "agents.portfolio_manager.execute_trade",
            return_value=(False, "Position sizing rejected"),
        ):
            result = execute_fast_path_trade(outcome, trigger, MagicMock(), MagicMock())

        assert result is False


# ---------------------------------------------------------------------------
# Tests: execute_fast_path_trade — frozen geometry usage
# ---------------------------------------------------------------------------


class TestExecuteFastPathTradeFrozenGeometry:
    """trade_executed uses frozen geometry from trigger, not fresh signal data."""

    @patch("utils.fast_path_execution.PM_CANDIDATE_MODE", "disabled")
    @patch("utils.fast_path_execution.FAST_PATH_MODE", "enabled")
    def test_decision_uses_trigger_entry_price(self):
        """Decision dict uses trigger.entry_price (frozen geometry)."""
        trigger = _make_trigger(
            entry_price=351.61,
            stop_price=355.00,
            target_price=348.00,
        )
        outcome = _make_outcome(trigger, current_price=350.50)

        with patch(
            "agents.portfolio_manager.execute_trade",
            return_value=(True, "ok"),
        ) as mock_execute:
            execute_fast_path_trade(outcome, trigger, MagicMock(), MagicMock())

        # Extract the decision dict passed to execute_trade
        call_kwargs = mock_execute.call_args[1]
        decision = call_kwargs["decision"]

        # Frozen geometry from trigger
        assert decision["entry_price"] == 351.61
        assert decision["stop_price"] == 355.00
        assert decision["target_price"] == 348.00
        # Current price from outcome (used as execution price field)
        assert decision["price"] == 350.50
        assert decision["symbol"] == "TSLA"
        assert decision["action"] == "SHORT"

    @patch("utils.fast_path_execution.PM_CANDIDATE_MODE", "disabled")
    @patch("utils.fast_path_execution.FAST_PATH_MODE", "enabled")
    def test_decision_does_not_use_fresh_signal_data(self):
        """Even if outcome has different prices, decision uses trigger's frozen geometry."""
        trigger = _make_trigger(
            entry_price=351.61,
            stop_price=355.00,
            target_price=348.00,
        )
        # Outcome reports different current_price but trigger geometry is authoritative
        outcome = _make_outcome(
            trigger,
            current_price=349.00,
            entry_price=349.00,  # This should NOT override trigger geometry
            stop_price=353.00,  # This should NOT override trigger geometry
            target_price=346.00,  # This should NOT override trigger geometry
        )

        with patch(
            "agents.portfolio_manager.execute_trade",
            return_value=(True, "ok"),
        ) as mock_execute:
            execute_fast_path_trade(outcome, trigger, MagicMock(), MagicMock())

        call_kwargs = mock_execute.call_args[1]
        decision = call_kwargs["decision"]

        # Must use trigger's frozen geometry, not outcome's potentially stale fields
        assert decision["entry_price"] == 351.61
        assert decision["stop_price"] == 355.00
        assert decision["target_price"] == 348.00
        # current_price from outcome is used as the price field
        assert decision["price"] == 349.00


# ---------------------------------------------------------------------------
# Tests: Provenance rejection (no_strategy_provenance)
# ---------------------------------------------------------------------------


class TestProvenanceRejection:
    """Trigger without source_signal_id or source_watch_id is rejected."""

    @patch("utils.fast_path_execution.FAST_PATH_MODE", "enabled")
    def test_no_provenance_returns_false(self):
        """Trigger with both source_signal_id=None and source_watch_id=None is rejected."""
        trigger = _make_trigger(
            source_signal_id=None,
            source_watch_id=None,
        )
        outcome = _make_outcome(trigger)

        result = execute_fast_path_trade(outcome, trigger, MagicMock(), MagicMock())

        assert result is False

    @patch("utils.fast_path_execution.FAST_PATH_MODE", "enabled")
    def test_provenance_with_signal_id_allowed(self):
        """Trigger with source_signal_id is allowed past provenance check."""
        trigger = _make_trigger(
            source_signal_id="sig-abc",
            source_watch_id=None,
        )
        outcome = _make_outcome(trigger)

        with patch(
            "utils.fast_path_execution.PM_CANDIDATE_MODE", "disabled"
        ), patch(
            "agents.portfolio_manager.execute_trade",
            return_value=(True, "ok"),
        ):
            result = execute_fast_path_trade(outcome, trigger, MagicMock(), MagicMock())

        assert result is True

    @patch("utils.fast_path_execution.FAST_PATH_MODE", "enabled")
    def test_provenance_with_watch_id_allowed(self):
        """Trigger with source_watch_id (but no signal_id) is allowed."""
        trigger = _make_trigger(
            source_signal_id=None,
            source_watch_id="watch-xyz",
        )
        outcome = _make_outcome(trigger)

        with patch(
            "utils.fast_path_execution.PM_CANDIDATE_MODE", "disabled"
        ), patch(
            "agents.portfolio_manager.execute_trade",
            return_value=(True, "ok"),
        ):
            result = execute_fast_path_trade(outcome, trigger, MagicMock(), MagicMock())

        assert result is True

    @patch("utils.fast_path_execution.FAST_PATH_MODE", "enabled")
    def test_pending_order_no_provenance_returns_false(self):
        """Pending order trigger without provenance is also rejected."""
        trigger = _make_trigger(
            source_signal_id=None,
            source_watch_id=None,
        )
        outcome = _make_outcome(trigger, outcome_type="pending_order_created")

        result = execute_fast_path_pending_order(
            outcome, trigger, MagicMock(), MagicMock()
        )

        assert result is False


# ---------------------------------------------------------------------------
# Tests: execute_fast_path_pending_order — delegation
# ---------------------------------------------------------------------------


class TestExecuteFastPathPendingOrderDelegation:
    """pending_order_created delegates to pending_order_creation when mode enabled."""

    @patch("utils.fast_path_execution.PENDING_ORDER_MODE", "enabled")
    @patch("utils.fast_path_execution.FAST_PATH_MODE", "enabled")
    def test_delegates_to_maybe_create_pending_order(self):
        """When PENDING_ORDER_MODE is enabled, calls maybe_create_pending_order."""
        trigger = _make_trigger()
        outcome = _make_outcome(trigger, outcome_type="pending_order_created")

        mock_creation_result = MagicMock()
        mock_creation_result.created = True
        mock_creation_result.order_id = "order-123"

        with patch(
            "utils.pending_order_creation.maybe_create_pending_order",
            return_value=mock_creation_result,
        ) as mock_create:
            result = execute_fast_path_pending_order(
                outcome, trigger, MagicMock(), MagicMock()
            )

        assert result is True
        mock_create.assert_called_once()
        # Verify it passes the trigger's frozen geometry
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["intended_entry"] == trigger.entry_price
        assert call_kwargs["stop"] == trigger.stop_price
        assert call_kwargs["target"] == trigger.target_price
        assert call_kwargs["symbol"] == trigger.symbol

    @patch("utils.fast_path_execution.PENDING_ORDER_MODE", "enabled")
    @patch("utils.fast_path_execution.FAST_PATH_MODE", "enabled")
    def test_pending_order_declined_returns_false(self):
        """When maybe_create_pending_order declines, returns False."""
        trigger = _make_trigger()
        outcome = _make_outcome(trigger, outcome_type="pending_order_created")

        mock_creation_result = MagicMock()
        mock_creation_result.created = False
        mock_creation_result.decline_reason = "target_already_exceeded"

        with patch(
            "utils.pending_order_creation.maybe_create_pending_order",
            return_value=mock_creation_result,
        ):
            result = execute_fast_path_pending_order(
                outcome, trigger, MagicMock(), MagicMock()
            )

        assert result is False


# ---------------------------------------------------------------------------
# Tests: execute_fast_path_pending_order — intent recording (mode not enabled)
# ---------------------------------------------------------------------------


class TestExecuteFastPathPendingOrderIntentOnly:
    """pending_order_created records intent only when mode not enabled."""

    @patch("utils.fast_path_execution.PENDING_ORDER_MODE", "disabled")
    @patch("utils.fast_path_execution.FAST_PATH_MODE", "enabled")
    def test_records_intent_when_pending_order_mode_disabled(self):
        """When PENDING_ORDER_MODE is disabled, records intent and returns True."""
        trigger = _make_trigger()
        outcome = _make_outcome(trigger, outcome_type="pending_order_created")

        # Mock engine connection for intent recording (fail-open)
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        result = execute_fast_path_pending_order(
            outcome, trigger, MagicMock(), mock_engine
        )

        assert result is True

    @patch("utils.fast_path_execution.PENDING_ORDER_MODE", "observe")
    @patch("utils.fast_path_execution.FAST_PATH_MODE", "enabled")
    def test_records_intent_when_pending_order_mode_observe(self):
        """When PENDING_ORDER_MODE is observe, records intent and returns True."""
        trigger = _make_trigger()
        outcome = _make_outcome(trigger, outcome_type="pending_order_created")

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        result = execute_fast_path_pending_order(
            outcome, trigger, MagicMock(), mock_engine
        )

        assert result is True

    @patch("utils.fast_path_execution.PENDING_ORDER_MODE", "disabled")
    @patch("utils.fast_path_execution.FAST_PATH_MODE", "enabled")
    def test_does_not_call_maybe_create_pending_order_when_disabled(self):
        """Intent-only mode does NOT call maybe_create_pending_order."""
        trigger = _make_trigger()
        outcome = _make_outcome(trigger, outcome_type="pending_order_created")

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        with patch(
            "utils.pending_order_creation.maybe_create_pending_order",
        ) as mock_create:
            execute_fast_path_pending_order(
                outcome, trigger, MagicMock(), mock_engine
            )

        mock_create.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: execute_fast_path_watch — watch creation
# ---------------------------------------------------------------------------


class TestExecuteFastPathWatch:
    """watch_created creates a watch candidate row."""

    @patch("utils.fast_path_execution.FAST_PATH_MODE", "enabled")
    def test_watch_created_delegates_to_create_watch_candidate(self):
        """watch_created outcome calls create_watch_candidate."""
        trigger = _make_trigger()
        outcome = _make_outcome(trigger, outcome_type="watch_created",
                                outcome_reason_code="awaiting_confirmation")

        with patch(
            "utils.watch_candidates.create_watch_candidate",
            return_value="watch-id-123",
        ) as mock_create:
            result = execute_fast_path_watch(
                outcome, trigger, MagicMock(), MagicMock()
            )

        assert result is True
        mock_create.assert_called_once()
        # Verify frozen geometry is passed
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["symbol"] == "TSLA"
        assert call_kwargs["profile_id"] == "moderate"
        assert call_kwargs["direction"] == "SHORT"
        assert call_kwargs["entry_price"] == 351.61
        assert call_kwargs["stop_price"] == 355.00
        assert call_kwargs["target_price"] == 348.00
        assert call_kwargs["setup_type"] == "momentum_fade"

    @patch("utils.fast_path_execution.FAST_PATH_MODE", "enabled")
    def test_watch_creation_failure_returns_false(self):
        """When create_watch_candidate returns None, returns False."""
        trigger = _make_trigger()
        outcome = _make_outcome(trigger, outcome_type="watch_created",
                                outcome_reason_code="awaiting_confirmation")

        with patch(
            "utils.watch_candidates.create_watch_candidate",
            return_value=None,
        ):
            result = execute_fast_path_watch(
                outcome, trigger, MagicMock(), MagicMock()
            )

        assert result is False

    @patch("utils.fast_path_execution.FAST_PATH_MODE", "enabled")
    def test_watch_promoted_returns_true(self):
        """watch_promoted outcome logs and returns True (already promoted)."""
        trigger = _make_trigger(source_watch_id="watch-existing-001")
        outcome = _make_outcome(trigger, outcome_type="watch_promoted",
                                outcome_reason_code="watch_matured")

        result = execute_fast_path_watch(
            outcome, trigger, MagicMock(), MagicMock()
        )

        assert result is True

    @patch("utils.fast_path_execution.FAST_PATH_MODE", "observe")
    def test_watch_not_created_in_observe_mode(self):
        """Watches are not created when FAST_PATH_MODE is observe."""
        trigger = _make_trigger()
        outcome = _make_outcome(trigger, outcome_type="watch_created",
                                outcome_reason_code="awaiting_confirmation")

        result = execute_fast_path_watch(
            outcome, trigger, MagicMock(), MagicMock()
        )

        assert result is False


# ---------------------------------------------------------------------------
# Tests: Execution failure resilience
# ---------------------------------------------------------------------------


class TestExecutionFailureResilience:
    """Execution failure does not crash monitor — logged and event updated."""

    @patch("utils.fast_path_execution.PM_CANDIDATE_MODE", "disabled")
    @patch("utils.fast_path_execution.FAST_PATH_MODE", "enabled")
    def test_execute_trade_exception_does_not_crash(self):
        """Exception in execute_trade is caught, returns False, does not raise."""
        trigger = _make_trigger()
        outcome = _make_outcome(trigger)

        with patch(
            "agents.portfolio_manager.execute_trade",
            side_effect=RuntimeError("Database locked"),
        ):
            # Should NOT raise — must catch and return False
            result = execute_fast_path_trade(
                outcome, trigger, MagicMock(), MagicMock()
            )

        assert result is False

    @patch("utils.fast_path_execution.PENDING_ORDER_MODE", "enabled")
    @patch("utils.fast_path_execution.FAST_PATH_MODE", "enabled")
    def test_pending_order_exception_does_not_crash(self):
        """Exception in maybe_create_pending_order is caught, returns False."""
        trigger = _make_trigger()
        outcome = _make_outcome(trigger, outcome_type="pending_order_created")

        with patch(
            "utils.pending_order_creation.maybe_create_pending_order",
            side_effect=RuntimeError("Connection refused"),
        ):
            result = execute_fast_path_pending_order(
                outcome, trigger, MagicMock(), MagicMock()
            )

        assert result is False

    @patch("utils.fast_path_execution.FAST_PATH_MODE", "enabled")
    def test_watch_creation_exception_does_not_crash(self):
        """Exception in create_watch_candidate is caught, returns False."""
        trigger = _make_trigger()
        outcome = _make_outcome(trigger, outcome_type="watch_created",
                                outcome_reason_code="awaiting_confirmation")

        with patch(
            "utils.watch_candidates.create_watch_candidate",
            side_effect=RuntimeError("SQLite busy"),
        ):
            result = execute_fast_path_watch(
                outcome, trigger, MagicMock(), MagicMock()
            )

        assert result is False

    @patch("utils.fast_path_execution.PM_CANDIDATE_MODE", "disabled")
    @patch("utils.fast_path_execution.FAST_PATH_MODE", "enabled")
    def test_event_metadata_update_failure_does_not_crash(self):
        """Even if _update_event_failure_metadata fails, execution still returns False cleanly."""
        trigger = _make_trigger()
        outcome = _make_outcome(trigger)

        # execute_trade fails, then _update_event_failure_metadata also fails
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = RuntimeError("Engine dead")

        with patch(
            "agents.portfolio_manager.execute_trade",
            return_value=(False, "sizing_rejected"),
        ):
            # Should not raise despite engine issues for metadata update
            result = execute_fast_path_trade(
                outcome, trigger, MagicMock(), mock_engine
            )

        assert result is False


# ---------------------------------------------------------------------------
# Tests: FAST_PATH_MODE guard
# ---------------------------------------------------------------------------


class TestFastPathModeGuard:
    """Functions guard against being called when mode is not enabled."""

    @patch("utils.fast_path_execution.FAST_PATH_MODE", "disabled")
    def test_trade_returns_false_when_disabled(self):
        trigger = _make_trigger()
        outcome = _make_outcome(trigger)
        result = execute_fast_path_trade(outcome, trigger, MagicMock(), MagicMock())
        assert result is False

    @patch("utils.fast_path_execution.FAST_PATH_MODE", "observe")
    def test_trade_returns_false_when_observe(self):
        trigger = _make_trigger()
        outcome = _make_outcome(trigger)
        result = execute_fast_path_trade(outcome, trigger, MagicMock(), MagicMock())
        assert result is False

    @patch("utils.fast_path_execution.FAST_PATH_MODE", "disabled")
    def test_pending_order_returns_false_when_disabled(self):
        trigger = _make_trigger()
        outcome = _make_outcome(trigger, outcome_type="pending_order_created")
        result = execute_fast_path_pending_order(
            outcome, trigger, MagicMock(), MagicMock()
        )
        assert result is False
