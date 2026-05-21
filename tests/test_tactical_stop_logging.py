"""
Integration tests for tactical stop event logging.

Validates that the risk_geometry_gate emits correct trade events with
appropriate payload fields depending on whether the tactical exception
path or the global fallback path produced the final decision.

**Validates: Requirements 7.1, 7.2, 7.3**
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pytest

from utils.risk_geometry_gate import evaluate_risk_geometry


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

# Base trade geometry from regression test: aggressive NVDA tactical support bounce
_TRADE_TIMESTAMP = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
_ATR_TIMESTAMP = _TRADE_TIMESTAMP - timedelta(minutes=2)


def _tactical_pass_kwargs():
    """Return kwargs for a trade that qualifies for the tactical exception.

    Uses same geometry as regression tests: entry 220.61, stop 220.16, target 221.27.
    stop_distance = 0.45 > tactical_min_stop = max(220.61*0.002, 0.3555*1.0) = 0.44122
    """
    return dict(
        entry_price=220.61,
        stop_price=220.16,
        target_price=221.27,
        quantity=100,
        direction="BUY",
        symbol="NVDA",
        setup_type="support_bounce",
        atr_5min=0.3555,
        atr_timestamp=_ATR_TIMESTAMP,
        trade_timestamp=_TRADE_TIMESTAMP,
        max_dollar_risk=5000.0,
        profile="aggressive",
    )


def _fallback_kwargs():
    """Return kwargs for a trade that does NOT qualify for tactical exception (moderate profile).

    Same geometry as tactical pass but with moderate profile, which is not in
    tactical_stop_by_profile config.
    """
    return dict(
        entry_price=220.61,
        stop_price=220.16,
        target_price=221.27,
        quantity=100,
        direction="BUY",
        symbol="NVDA",
        setup_type="support_bounce",
        atr_5min=0.3555,
        atr_timestamp=_ATR_TIMESTAMP,
        trade_timestamp=_TRADE_TIMESTAMP,
        max_dollar_risk=5000.0,
        profile="moderate",
    )


# ---------------------------------------------------------------------------
# Requirement 7.1: Tactical pass emits event with tactical fields in payload
# ---------------------------------------------------------------------------


class TestTacticalPassEventPayload:
    """Tactical pass emits risk_geometry_gate_evaluated with tactical fields."""

    @patch("utils.risk_geometry_gate.log_trade_event")
    def test_tactical_pass_emits_event_with_tactical_stop_applied(self, mock_log):
        """Tactical pass event payload contains tactical_stop_applied: True."""
        db = MagicMock()
        result = evaluate_risk_geometry(db=db, **_tactical_pass_kwargs())

        # Confirm the trade actually took the tactical path
        assert result.get("tactical_stop_applied") is True
        assert result.get("reason_code") == "PASSED_TACTICAL"

        # Find the risk_geometry_gate_evaluated event call
        evaluated_calls = [
            c for c in mock_log.call_args_list
            if c[0][1] == "risk_geometry_gate_evaluated"
        ]
        assert len(evaluated_calls) == 1

        call_kwargs = evaluated_calls[0][1]
        payload = call_kwargs["payload"]

        assert payload["tactical_stop_applied"] is True

    @patch("utils.risk_geometry_gate.log_trade_event")
    def test_tactical_pass_emits_event_with_tactical_min_stop_distance(self, mock_log):
        """Tactical pass event payload contains tactical_min_stop_distance."""
        db = MagicMock()
        result = evaluate_risk_geometry(db=db, **_tactical_pass_kwargs())

        assert result.get("tactical_stop_applied") is True

        evaluated_calls = [
            c for c in mock_log.call_args_list
            if c[0][1] == "risk_geometry_gate_evaluated"
        ]
        assert len(evaluated_calls) == 1

        call_kwargs = evaluated_calls[0][1]
        payload = call_kwargs["payload"]

        assert "tactical_min_stop_distance" in payload
        assert isinstance(payload["tactical_min_stop_distance"], float)
        assert payload["tactical_min_stop_distance"] > 0

    @patch("utils.risk_geometry_gate.log_trade_event")
    def test_tactical_pass_event_payload_has_gate_name(self, mock_log):
        """Tactical pass event payload includes gate_name field."""
        db = MagicMock()
        evaluate_risk_geometry(db=db, **_tactical_pass_kwargs())

        evaluated_calls = [
            c for c in mock_log.call_args_list
            if c[0][1] == "risk_geometry_gate_evaluated"
        ]
        assert len(evaluated_calls) == 1

        call_kwargs = evaluated_calls[0][1]
        payload = call_kwargs["payload"]

        assert payload["gate_name"] == "risk_geometry_gate"


# ---------------------------------------------------------------------------
# Requirement 7.2: Fallback path emits event WITHOUT tactical fields
# ---------------------------------------------------------------------------


class TestFallbackPathEventPayload:
    """Fallback path emits event without tactical_stop_applied or tactical_min_stop_distance."""

    @patch("utils.risk_geometry_gate.log_trade_event")
    def test_fallback_path_event_does_not_contain_tactical_stop_applied(self, mock_log):
        """Fallback (non-tactical) path event payload does NOT contain tactical_stop_applied."""
        db = MagicMock()
        result = evaluate_risk_geometry(db=db, **_fallback_kwargs())

        # Confirm the trade did NOT take the tactical path
        assert "tactical_stop_applied" not in result

        # Find the risk_geometry_gate_evaluated event call
        evaluated_calls = [
            c for c in mock_log.call_args_list
            if c[0][1] == "risk_geometry_gate_evaluated"
        ]
        assert len(evaluated_calls) == 1

        call_kwargs = evaluated_calls[0][1]
        payload = call_kwargs["payload"]

        assert "tactical_stop_applied" not in payload

    @patch("utils.risk_geometry_gate.log_trade_event")
    def test_fallback_path_event_does_not_contain_tactical_min_stop_distance(self, mock_log):
        """Fallback (non-tactical) path event payload does NOT contain tactical_min_stop_distance."""
        db = MagicMock()
        result = evaluate_risk_geometry(db=db, **_fallback_kwargs())

        assert "tactical_stop_applied" not in result

        evaluated_calls = [
            c for c in mock_log.call_args_list
            if c[0][1] == "risk_geometry_gate_evaluated"
        ]
        assert len(evaluated_calls) == 1

        call_kwargs = evaluated_calls[0][1]
        payload = call_kwargs["payload"]

        assert "tactical_min_stop_distance" not in payload

    @patch("utils.risk_geometry_gate.log_trade_event")
    def test_fallback_with_non_qualifying_setup_no_tactical_fields(self, mock_log):
        """Non-qualifying setup on aggressive profile still has no tactical fields in event."""
        db = MagicMock()
        kwargs = _tactical_pass_kwargs()
        kwargs["setup_type"] = "breakout_continuation"  # non-qualifying setup

        result = evaluate_risk_geometry(db=db, **kwargs)

        assert "tactical_stop_applied" not in result

        evaluated_calls = [
            c for c in mock_log.call_args_list
            if c[0][1] == "risk_geometry_gate_evaluated"
        ]
        assert len(evaluated_calls) == 1

        call_kwargs = evaluated_calls[0][1]
        payload = call_kwargs["payload"]

        assert "tactical_stop_applied" not in payload
        assert "tactical_min_stop_distance" not in payload


# ---------------------------------------------------------------------------
# Requirement 7.3: Exactly one event emitted per gate evaluation
# ---------------------------------------------------------------------------


class TestExactlyOneEventPerEvaluation:
    """Exactly one risk_geometry_gate_evaluated event per gate evaluation."""

    @patch("utils.risk_geometry_gate.log_trade_event")
    def test_tactical_pass_emits_exactly_one_event(self, mock_log):
        """Tactical pass path emits exactly one risk_geometry_gate_evaluated event."""
        db = MagicMock()
        result = evaluate_risk_geometry(db=db, **_tactical_pass_kwargs())

        assert result.get("tactical_stop_applied") is True

        evaluated_calls = [
            c for c in mock_log.call_args_list
            if c[0][1] == "risk_geometry_gate_evaluated"
        ]
        assert len(evaluated_calls) == 1

    @patch("utils.risk_geometry_gate.log_trade_event")
    def test_fallback_path_emits_exactly_one_event(self, mock_log):
        """Fallback (global) path emits exactly one risk_geometry_gate_evaluated event."""
        db = MagicMock()
        result = evaluate_risk_geometry(db=db, **_fallback_kwargs())

        assert "tactical_stop_applied" not in result

        evaluated_calls = [
            c for c in mock_log.call_args_list
            if c[0][1] == "risk_geometry_gate_evaluated"
        ]
        assert len(evaluated_calls) == 1

    @patch("utils.risk_geometry_gate.log_trade_event")
    def test_tactical_fail_fallback_emits_exactly_one_event(self, mock_log):
        """Trade that qualifies for tactical but fails validation emits exactly one event."""
        db = MagicMock()
        # Use a stop that's too tight for tactical (below tactical_min_stop)
        kwargs = _tactical_pass_kwargs()
        kwargs["stop_price"] = 220.50  # stop_distance = 0.11, below tactical min

        result = evaluate_risk_geometry(db=db, **kwargs)

        # Should have fallen through to global path
        assert "tactical_stop_applied" not in result

        evaluated_calls = [
            c for c in mock_log.call_args_list
            if c[0][1] == "risk_geometry_gate_evaluated"
        ]
        assert len(evaluated_calls) == 1

    @patch("utils.risk_geometry_gate.log_trade_event")
    def test_no_event_when_db_is_none(self, mock_log):
        """No event logged when db is None."""
        result = evaluate_risk_geometry(db=None, **_tactical_pass_kwargs())

        # Gate still returns a result
        assert result.get("tactical_stop_applied") is True

        # But no event was logged
        mock_log.assert_not_called()
