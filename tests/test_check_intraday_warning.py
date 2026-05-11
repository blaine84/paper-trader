"""Unit tests for _check_intraday_warning function."""

from datetime import datetime, timezone, timedelta

from utils.position_lifecycle_governance import (
    _check_intraday_warning,
    DEFAULT_LIMITS,
    SETUP_TIME_LIMITS,
)


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
        "status": "open",
    }
    trade.update(kwargs)
    return trade, now


class TestCheckIntradayWarning:
    """Tests for _check_intraday_warning (Priority 7)."""

    def test_within_alert_threshold_returns_none(self):
        """Position within alert threshold should not trigger warning."""
        # gap_and_go alert=60, 30 min held
        trade, now = _make_trade("gap_and_go", minutes_ago=30)
        result = _check_intraday_warning(trade, now)
        assert result is None

    def test_past_alert_before_force_close_returns_warning(self):
        """Position past alert but before force_close should return warning."""
        # gap_and_go alert=60, force_close=90, 70 min held
        trade, now = _make_trade("gap_and_go", minutes_ago=70)
        result = _check_intraday_warning(trade, now)
        assert result is not None
        assert result["decision"] == "warn"
        assert result["state"] == "intraday_warning"
        assert result["reason_type"] == "intraday_time_warning"
        assert result["requires_event"] is True

    def test_metadata_contains_required_fields(self):
        """Warning result should include all required metadata."""
        # gap_and_go alert=60, force_close=90, 70 min held
        trade, now = _make_trade("gap_and_go", minutes_ago=70)
        result = _check_intraday_warning(trade, now)
        meta = result["metadata"]
        assert meta["minutes_held"] == 70
        assert meta["alert_limit"] == 60
        assert meta["force_limit"] == 90
        assert meta["setup_type"] == "gap_and_go"

    def test_missing_entry_time_returns_none(self):
        """Missing entry_time should return None."""
        trade = {
            "id": 1,
            "symbol": "AAPL",
            "profile": "moderate",
            "setup_type": "orb",
            "entry_time": None,
            "status": "open",
        }
        now = datetime(2025, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
        result = _check_intraday_warning(trade, now)
        assert result is None

    def test_unknown_setup_uses_default_limits(self):
        """Unknown setup_type should use DEFAULT_LIMITS (alert=60, force=90)."""
        trade, now = _make_trade("unknown_setup", minutes_ago=65)
        result = _check_intraday_warning(trade, now)
        assert result is not None
        assert result["metadata"]["alert_limit"] == 60
        assert result["metadata"]["force_limit"] == 90

    def test_momentum_fade_uses_specific_limits(self):
        """momentum_fade has alert=45, force_close=75."""
        trade, now = _make_trade("momentum_fade", minutes_ago=50)
        result = _check_intraday_warning(trade, now)
        assert result is not None
        assert result["metadata"]["alert_limit"] == 45
        assert result["metadata"]["force_limit"] == 75
        assert result["metadata"]["minutes_held"] == 50

    def test_exactly_at_alert_threshold_returns_none(self):
        """Position exactly at alert threshold should NOT trigger (> not >=)."""
        # gap_and_go alert=60, exactly 60 min held
        trade, now = _make_trade("gap_and_go", minutes_ago=60)
        result = _check_intraday_warning(trade, now)
        assert result is None

    def test_just_past_alert_threshold_returns_warning(self):
        """Position just past alert threshold should trigger warning."""
        # gap_and_go alert=60, 61 min held
        trade, now = _make_trade("gap_and_go", minutes_ago=61)
        result = _check_intraday_warning(trade, now)
        assert result is not None
        assert result["decision"] == "warn"

    def test_empty_setup_type_uses_default_limits(self):
        """Empty string setup_type should use DEFAULT_LIMITS."""
        trade, now = _make_trade("", minutes_ago=65)
        result = _check_intraday_warning(trade, now)
        assert result is not None
        assert result["metadata"]["alert_limit"] == 60
        assert result["metadata"]["force_limit"] == 90
        assert result["metadata"]["setup_type"] == ""

    def test_short_squeeze_specific_limits(self):
        """short_squeeze has alert=30, force_close=60."""
        trade, now = _make_trade("short_squeeze", minutes_ago=35)
        result = _check_intraday_warning(trade, now)
        assert result is not None
        assert result["metadata"]["alert_limit"] == 30
        assert result["metadata"]["force_limit"] == 60
