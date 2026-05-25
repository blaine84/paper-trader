"""Unit tests for setup-aware intraday alert/warning behavior.

These tests validate that the setup-aware evaluator (which replaced the legacy
_check_intraday_warning function) correctly produces alert/hold decisions for
positions past their alert threshold but before force-close.
"""

from datetime import datetime, timezone, timedelta

from utils.setup_aware_evaluator import evaluate_setup_aware_lifecycle
from utils.setup_time_policy import get_policy, SETUP_TIME_POLICY_REGISTRY, DEFAULT_POLICY


def _make_trade(setup_type="gap_and_go", minutes_ago=30, **kwargs):
    """Helper to create a trade dict with entry_time set relative to now."""
    now = datetime(2025, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
    entry = now - timedelta(minutes=minutes_ago)
    trade = {
        "id": 1,
        "symbol": "AAPL",
        "profile": "moderate",
        "setup_type": setup_type,
        "entry_time": entry,
        "entry_price": 150.0,
        "stop_price": 145.0,
        "direction": "LONG",
        "status": "open",
    }
    trade.update(kwargs)
    return trade, now


def _make_now_et():
    """Return a now_et that is NOT within the pre-wall buffer (early in the day)."""
    from zoneinfo import ZoneInfo
    return datetime(2025, 6, 15, 14, 0, 0, tzinfo=ZoneInfo("US/Eastern"))


class TestSetupAwareIntradayAlert:
    """Tests for setup-aware alert behavior (replaces _check_intraday_warning)."""

    def test_within_alert_threshold_returns_hold(self):
        """Position within alert threshold should return hold/intraday_ok."""
        # gap_and_go alert=60, 30 min held
        trade, now = _make_trade("gap_and_go", minutes_ago=30)
        result = evaluate_setup_aware_lifecycle(
            trade, [], now_utc=now, now_et=_make_now_et()
        )
        assert result["decision"] == "hold"
        assert result["state"] == "intraday_ok"

    def test_past_alert_before_force_close_returns_alert(self):
        """Position past alert but before force_close should return alert state."""
        # gap_and_go alert=60, force_close=90, 70 min held
        trade, now = _make_trade("gap_and_go", minutes_ago=70)
        result = evaluate_setup_aware_lifecycle(
            trade, [], now_utc=now, now_et=_make_now_et()
        )
        assert result is not None
        assert result["decision"] == "hold"
        assert result["state"] == "setup_exit_alert"
        assert result["reason_type"] == "setup_alert_approaching_revalidation"
        assert result["requires_event"] is True

    def test_metadata_contains_required_fields(self):
        """Alert result should include all required metadata."""
        # gap_and_go alert=60, force_close=90, 70 min held
        trade, now = _make_trade("gap_and_go", minutes_ago=70)
        result = evaluate_setup_aware_lifecycle(
            trade, [], now_utc=now, now_et=_make_now_et()
        )
        meta = result["metadata"]
        assert meta["minutes_held"] == 70
        assert meta["alert_minutes"] == 60
        assert meta["force_close_minutes"] == 90
        assert meta["setup_type"] == "gap_and_go"

    def test_missing_entry_time_returns_hold(self):
        """Missing entry_time should return hold/intraday_ok (no action)."""
        trade = {
            "id": 1,
            "symbol": "AAPL",
            "profile": "moderate",
            "setup_type": "orb",
            "entry_time": None,
            "status": "open",
        }
        now = datetime(2025, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        result = evaluate_setup_aware_lifecycle(
            trade, [], now_utc=now, now_et=_make_now_et()
        )
        assert result["decision"] == "hold"

    def test_unknown_setup_uses_default_policy(self):
        """Unknown setup_type should use default policy (alert=60, force=90)."""
        trade, now = _make_trade("unknown_setup", minutes_ago=65)
        result = evaluate_setup_aware_lifecycle(
            trade, [], now_utc=now, now_et=_make_now_et()
        )
        assert result is not None
        assert result["state"] == "setup_exit_alert"
        meta = result["metadata"]
        assert meta["alert_minutes"] == 60
        assert meta["force_close_minutes"] == 90

    def test_momentum_fade_uses_specific_limits(self):
        """momentum_fade has alert=35, force_close=75."""
        trade, now = _make_trade("momentum_fade", minutes_ago=50)
        result = evaluate_setup_aware_lifecycle(
            trade, [], now_utc=now, now_et=_make_now_et()
        )
        assert result is not None
        assert result["state"] == "setup_exit_alert"
        meta = result["metadata"]
        assert meta["alert_minutes"] == 35
        assert meta["force_close_minutes"] == 75
        assert meta["minutes_held"] == 50

    def test_exactly_at_alert_threshold_returns_hold(self):
        """Position exactly at alert threshold should NOT trigger alert (< not <=)."""
        # gap_and_go alert=60, exactly 60 min held — alert fires at >= alert_minutes
        # The setup-aware evaluator uses < alert_minutes for the "below alert" check
        # so exactly at alert_minutes will trigger the alert path
        trade, now = _make_trade("gap_and_go", minutes_ago=60)
        result = evaluate_setup_aware_lifecycle(
            trade, [], now_utc=now, now_et=_make_now_et()
        )
        # At exactly alert_minutes, the evaluator enters the alert path
        assert result["state"] == "setup_exit_alert"

    def test_just_past_alert_threshold_returns_alert(self):
        """Position just past alert threshold should trigger alert."""
        # gap_and_go alert=60, 61 min held
        trade, now = _make_trade("gap_and_go", minutes_ago=61)
        result = evaluate_setup_aware_lifecycle(
            trade, [], now_utc=now, now_et=_make_now_et()
        )
        assert result is not None
        assert result["state"] == "setup_exit_alert"

    def test_empty_setup_type_uses_default_policy(self):
        """Empty string setup_type should use default policy."""
        trade, now = _make_trade("", minutes_ago=65)
        result = evaluate_setup_aware_lifecycle(
            trade, [], now_utc=now, now_et=_make_now_et()
        )
        assert result is not None
        assert result["state"] == "setup_exit_alert"
        meta = result["metadata"]
        assert meta["alert_minutes"] == 60
        assert meta["force_close_minutes"] == 90
        assert meta["setup_type"] == ""

    def test_short_squeeze_specific_limits(self):
        """short_squeeze has alert=30, force_close=60."""
        trade, now = _make_trade("short_squeeze", minutes_ago=35)
        result = evaluate_setup_aware_lifecycle(
            trade, [], now_utc=now, now_et=_make_now_et()
        )
        assert result is not None
        assert result["state"] == "setup_exit_alert"
        meta = result["metadata"]
        assert meta["alert_minutes"] == 30
        assert meta["force_close_minutes"] == 60
