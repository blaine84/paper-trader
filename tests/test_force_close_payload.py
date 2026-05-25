"""Unit tests for force-close payload construction and news_breakout legacy override prevention.

Validates that:
- Force-close decisions include all required payload fields (setup_type, policy limit,
  actual minutes held, revalidation_attempted flag)
- Force-close decisions map to `setup_exit_force_close` event type
- news_breakout/news_catalyst trades are NOT force-closed at the legacy 90-minute
  threshold when within their valid extension window
- Non-extension-eligible setups ARE force-closed at their policy limit

Validates: Requirements 4.1, 4.2, 4.3, 4.7
"""

from datetime import datetime, time, timezone

import pytest

from utils.setup_aware_evaluator import (
    LIFECYCLE_STATE_TO_EVENT_TYPE,
    evaluate_setup_aware_lifecycle,
)
from utils.setup_time_policy import get_policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trade(
    setup_type="news_breakout",
    direction="LONG",
    entry_price=150.0,
    stop_price=145.0,
    target_price=160.0,
    entry_time=None,
    invalidators=None,
):
    """Create a trade dict for testing."""
    if entry_time is None:
        entry_time = datetime(2026, 5, 26, 9, 30, 0, tzinfo=timezone.utc)
    return {
        "id": 1,
        "symbol": "AMD",
        "profile": "moderate",
        "direction": direction,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "setup_type": setup_type,
        "status": "open",
        "quantity": 100,
        "entry_time": entry_time,
        "thesis": "Strong catalyst with volume confirmation",
        "invalidators": invalidators,
    }


def _now_utc_at_minutes(entry_time, minutes):
    """Compute a now_utc that is `minutes` after entry_time."""
    from datetime import timedelta

    return entry_time + timedelta(minutes=minutes)


def _now_et_morning():
    """Return a now_et time in the morning (well before EOD hard wall)."""
    return datetime(2026, 5, 26, 10, 0, 0)


# ---------------------------------------------------------------------------
# Force-close payload completeness
# ---------------------------------------------------------------------------


class TestForceClosePayloadCompleteness:
    """Force-close decisions include all required fields in metadata."""

    def test_non_extension_force_close_includes_setup_type(self):
        """Non-extension-eligible force close includes setup_type in metadata."""
        trade = _make_trade(setup_type="momentum_fade", entry_price=50.0, stop_price=48.0)
        entry_time = trade["entry_time"]
        now_utc = _now_utc_at_minutes(entry_time, 76)  # Past 75-min force_close

        result = evaluate_setup_aware_lifecycle(
            trade, [], now_utc=now_utc, now_et=_now_et_morning()
        )

        assert result["state"] == "setup_time_limit_exceeded"
        assert result["metadata"]["setup_type"] == "momentum_fade"

    def test_non_extension_force_close_includes_force_close_minutes(self):
        """Non-extension-eligible force close includes the policy limit (force_close_minutes)."""
        trade = _make_trade(setup_type="momentum_fade", entry_price=50.0, stop_price=48.0)
        entry_time = trade["entry_time"]
        now_utc = _now_utc_at_minutes(entry_time, 76)

        result = evaluate_setup_aware_lifecycle(
            trade, [], now_utc=now_utc, now_et=_now_et_morning()
        )

        assert result["metadata"]["force_close_minutes"] == 75

    def test_non_extension_force_close_includes_minutes_held(self):
        """Non-extension-eligible force close includes actual minutes held."""
        trade = _make_trade(setup_type="momentum_fade", entry_price=50.0, stop_price=48.0)
        entry_time = trade["entry_time"]
        now_utc = _now_utc_at_minutes(entry_time, 76)

        result = evaluate_setup_aware_lifecycle(
            trade, [], now_utc=now_utc, now_et=_now_et_morning()
        )

        assert result["metadata"]["minutes_held"] == 76

    def test_non_extension_force_close_includes_revalidation_attempted(self):
        """Non-extension-eligible force close includes revalidation_attempted=False."""
        trade = _make_trade(setup_type="momentum_fade", entry_price=50.0, stop_price=48.0)
        entry_time = trade["entry_time"]
        now_utc = _now_utc_at_minutes(entry_time, 76)

        result = evaluate_setup_aware_lifecycle(
            trade, [], now_utc=now_utc, now_et=_now_et_morning()
        )

        assert result["metadata"]["revalidation_attempted"] is False

    def test_extension_eligible_force_close_includes_revalidation_attempted_true(self):
        """Extension-eligible force close (time expired) includes revalidation_attempted=True."""
        # news_breakout at 91 min with valid criteria but price below entry and no positive indicators
        trade = _make_trade(
            setup_type="news_breakout",
            entry_price=150.0,
            stop_price=145.0,
            target_price=160.0,
        )
        entry_time = trade["entry_time"]
        now_utc = _now_utc_at_minutes(entry_time, 91)
        # Price below entry, above stop, no positive indicators → close_time_expired
        # (146 is above stop 145 but below entry 150, target_progress negative)
        market_ts = now_utc

        result = evaluate_setup_aware_lifecycle(
            trade,
            [],
            now_utc=now_utc,
            now_et=_now_et_morning(),
            current_price=146.0,
            market_data_timestamp=market_ts,
        )

        assert result["state"] == "setup_time_limit_exceeded"
        assert result["metadata"]["revalidation_attempted"] is True

    def test_force_close_state_maps_to_setup_exit_force_close_event(self):
        """The state setup_time_limit_exceeded maps to setup_exit_force_close event type."""
        assert LIFECYCLE_STATE_TO_EVENT_TYPE["setup_time_limit_exceeded"] == "setup_exit_force_close"

    def test_force_close_decision_is_close(self):
        """Force-close decisions have decision='close'."""
        trade = _make_trade(setup_type="momentum_fade", entry_price=50.0, stop_price=48.0)
        entry_time = trade["entry_time"]
        now_utc = _now_utc_at_minutes(entry_time, 76)

        result = evaluate_setup_aware_lifecycle(
            trade, [], now_utc=now_utc, now_et=_now_et_morning()
        )

        assert result["decision"] == "close"

    def test_force_close_requires_event(self):
        """Force-close decisions have requires_event=True for event emission."""
        trade = _make_trade(setup_type="momentum_fade", entry_price=50.0, stop_price=48.0)
        entry_time = trade["entry_time"]
        now_utc = _now_utc_at_minutes(entry_time, 76)

        result = evaluate_setup_aware_lifecycle(
            trade, [], now_utc=now_utc, now_et=_now_et_morning()
        )

        assert result["requires_event"] is True

    def test_force_close_has_close_reason_string(self):
        """Force-close decisions include a human-readable close_reason."""
        trade = _make_trade(setup_type="momentum_fade", entry_price=50.0, stop_price=48.0)
        entry_time = trade["entry_time"]
        now_utc = _now_utc_at_minutes(entry_time, 76)

        result = evaluate_setup_aware_lifecycle(
            trade, [], now_utc=now_utc, now_et=_now_et_morning()
        )

        assert result["close_reason"] is not None
        assert "momentum_fade" in result["close_reason"]
        assert "76" in result["close_reason"]


# ---------------------------------------------------------------------------
# News_breakout NOT closed at legacy 90-minute threshold
# ---------------------------------------------------------------------------


class TestNewsBreakoutLegacyOverridePrevention:
    """news_breakout/news_catalyst trades are NOT force-closed at legacy 90 minutes."""

    def test_news_breakout_at_91_minutes_not_force_closed(self):
        """news_breakout at 91 minutes is NOT force-closed — it enters revalidation.

        The legacy system would force-close at 90 minutes. The setup-aware system
        recognizes news_breakout has force_close_minutes=120 and revalidate_minutes=90,
        so at 91 minutes it performs revalidation rather than force-closing.
        With valid criteria and price above entry → hold_valid_until_next_window.
        """
        trade = _make_trade(
            setup_type="news_breakout",
            entry_price=150.0,
            stop_price=145.0,
            target_price=160.0,
        )
        entry_time = trade["entry_time"]
        now_utc = _now_utc_at_minutes(entry_time, 91)
        market_ts = now_utc

        result = evaluate_setup_aware_lifecycle(
            trade,
            [],
            now_utc=now_utc,
            now_et=_now_et_morning(),
            current_price=152.0,  # Above entry → positive indicator
            market_data_timestamp=market_ts,
        )

        # Should NOT be force-closed — should be held via revalidation
        assert result["decision"] != "close" or result["state"] != "setup_time_limit_exceeded"
        assert result["state"] == "setup_revalidation_hold"
        assert result["decision"] == "hold"

    def test_news_catalyst_at_91_minutes_not_force_closed(self):
        """news_catalyst at 91 minutes is NOT force-closed — same policy as news_breakout."""
        trade = _make_trade(
            setup_type="news_catalyst",
            entry_price=150.0,
            stop_price=145.0,
            target_price=160.0,
        )
        entry_time = trade["entry_time"]
        now_utc = _now_utc_at_minutes(entry_time, 91)
        market_ts = now_utc

        result = evaluate_setup_aware_lifecycle(
            trade,
            [],
            now_utc=now_utc,
            now_et=_now_et_morning(),
            current_price=152.0,
            market_data_timestamp=market_ts,
        )

        assert result["state"] == "setup_revalidation_hold"
        assert result["decision"] == "hold"

    def test_news_breakout_at_91_minutes_with_valid_criteria_gets_revalidation(self):
        """news_breakout at 91 min with valid stop gets revalidation, not immediate close."""
        trade = _make_trade(
            setup_type="news_breakout",
            entry_price=150.0,
            stop_price=145.0,
            target_price=160.0,
        )
        entry_time = trade["entry_time"]
        now_utc = _now_utc_at_minutes(entry_time, 91)
        market_ts = now_utc

        # Price above entry → hold
        result = evaluate_setup_aware_lifecycle(
            trade,
            [],
            now_utc=now_utc,
            now_et=_now_et_morning(),
            current_price=155.0,
            market_data_timestamp=market_ts,
        )

        assert result["state"] == "setup_revalidation_hold"
        assert result["metadata"]["revalidation_attempted"] is True

    def test_news_breakout_at_121_minutes_with_valid_criteria_gets_revalidation(self):
        """news_breakout at 121 min with valid criteria gets revalidation (not immediate close).

        At 121 minutes, the trade is past force_close_minutes=120 but still within
        max_extension_minutes=180. With valid criteria and positive indicators,
        it should get a hold decision via revalidation.
        """
        trade = _make_trade(
            setup_type="news_breakout",
            entry_price=150.0,
            stop_price=145.0,
            target_price=160.0,
        )
        entry_time = trade["entry_time"]
        now_utc = _now_utc_at_minutes(entry_time, 121)
        market_ts = now_utc

        result = evaluate_setup_aware_lifecycle(
            trade,
            [],
            now_utc=now_utc,
            now_et=_now_et_morning(),
            current_price=155.0,  # Above entry → positive indicator
            market_data_timestamp=market_ts,
        )

        # Should be held via revalidation, not force-closed
        assert result["state"] == "setup_revalidation_hold"
        assert result["decision"] == "hold"
        assert result["metadata"]["revalidation_attempted"] is True

    def test_news_breakout_policy_has_120_not_90_force_close(self):
        """Verify news_breakout policy uses 120-minute force_close, not legacy 90."""
        policy = get_policy("news_breakout")
        assert policy.force_close_minutes == 120
        assert policy.force_close_minutes != 90

    def test_news_catalyst_policy_has_120_not_90_force_close(self):
        """Verify news_catalyst policy uses 120-minute force_close, not legacy 90."""
        policy = get_policy("news_catalyst")
        assert policy.force_close_minutes == 120
        assert policy.force_close_minutes != 90


# ---------------------------------------------------------------------------
# Non-extension-eligible setup IS force-closed at policy limit
# ---------------------------------------------------------------------------


class TestNonExtensionForceClose:
    """Non-extension-eligible setups ARE force-closed at their policy limit."""

    def test_momentum_fade_at_76_minutes_is_force_closed(self):
        """momentum_fade at 76 minutes IS force-closed (force_close_minutes=75)."""
        trade = _make_trade(
            setup_type="momentum_fade",
            entry_price=50.0,
            stop_price=48.0,
            target_price=55.0,
        )
        entry_time = trade["entry_time"]
        now_utc = _now_utc_at_minutes(entry_time, 76)

        result = evaluate_setup_aware_lifecycle(
            trade, [], now_utc=now_utc, now_et=_now_et_morning()
        )

        assert result["decision"] == "close"
        assert result["state"] == "setup_time_limit_exceeded"

    def test_orb_at_76_minutes_is_force_closed(self):
        """orb at 76 minutes IS force-closed (force_close_minutes=75)."""
        trade = _make_trade(
            setup_type="orb",
            entry_price=50.0,
            stop_price=48.0,
            target_price=55.0,
        )
        entry_time = trade["entry_time"]
        now_utc = _now_utc_at_minutes(entry_time, 76)

        result = evaluate_setup_aware_lifecycle(
            trade, [], now_utc=now_utc, now_et=_now_et_morning()
        )

        assert result["decision"] == "close"
        assert result["state"] == "setup_time_limit_exceeded"

    def test_short_squeeze_at_61_minutes_is_force_closed(self):
        """short_squeeze at 61 minutes IS force-closed (force_close_minutes=60)."""
        trade = _make_trade(
            setup_type="short_squeeze",
            entry_price=50.0,
            stop_price=48.0,
            target_price=55.0,
        )
        entry_time = trade["entry_time"]
        now_utc = _now_utc_at_minutes(entry_time, 61)

        result = evaluate_setup_aware_lifecycle(
            trade, [], now_utc=now_utc, now_et=_now_et_morning()
        )

        assert result["decision"] == "close"
        assert result["state"] == "setup_time_limit_exceeded"

    def test_momentum_fade_force_close_metadata_complete(self):
        """momentum_fade force close has all required metadata fields."""
        trade = _make_trade(
            setup_type="momentum_fade",
            entry_price=50.0,
            stop_price=48.0,
            target_price=55.0,
        )
        entry_time = trade["entry_time"]
        now_utc = _now_utc_at_minutes(entry_time, 76)

        result = evaluate_setup_aware_lifecycle(
            trade, [], now_utc=now_utc, now_et=_now_et_morning()
        )

        meta = result["metadata"]
        assert meta["setup_type"] == "momentum_fade"
        assert meta["force_close_minutes"] == 75
        assert meta["minutes_held"] == 76
        assert meta["revalidation_attempted"] is False
        assert meta["extension_eligible"] is False

    def test_gap_and_go_at_91_minutes_is_force_closed(self):
        """gap_and_go at 91 minutes IS force-closed (force_close_minutes=90).

        This is the same 90-minute threshold as legacy, but gap_and_go is
        non-extension-eligible so it correctly gets force-closed.
        """
        trade = _make_trade(
            setup_type="gap_and_go",
            entry_price=50.0,
            stop_price=48.0,
            target_price=55.0,
        )
        entry_time = trade["entry_time"]
        now_utc = _now_utc_at_minutes(entry_time, 91)

        result = evaluate_setup_aware_lifecycle(
            trade, [], now_utc=now_utc, now_et=_now_et_morning()
        )

        assert result["decision"] == "close"
        assert result["state"] == "setup_time_limit_exceeded"


# ---------------------------------------------------------------------------
# Force-close state consistency
# ---------------------------------------------------------------------------


class TestForceCloseStateConsistency:
    """All force-close decisions use the correct state value."""

    def test_non_extension_force_close_state(self):
        """Non-extension force close uses state=setup_time_limit_exceeded."""
        trade = _make_trade(setup_type="momentum_fade", entry_price=50.0, stop_price=48.0)
        entry_time = trade["entry_time"]
        now_utc = _now_utc_at_minutes(entry_time, 76)

        result = evaluate_setup_aware_lifecycle(
            trade, [], now_utc=now_utc, now_et=_now_et_morning()
        )

        assert result["state"] == "setup_time_limit_exceeded"

    def test_max_extension_force_close_state(self):
        """Max extension reached uses state=setup_time_limit_exceeded."""
        trade = _make_trade(
            setup_type="news_breakout",
            entry_price=150.0,
            stop_price=145.0,
            target_price=160.0,
        )
        entry_time = trade["entry_time"]
        # At 180 min (max_extension_minutes for news_breakout)
        now_utc = _now_utc_at_minutes(entry_time, 181)

        result = evaluate_setup_aware_lifecycle(
            trade, [], now_utc=now_utc, now_et=_now_et_morning()
        )

        assert result["state"] == "setup_time_limit_exceeded"
        assert result["decision"] == "close"

    def test_revalidation_time_expired_force_close_state(self):
        """Revalidation producing close_time_expired uses state=setup_time_limit_exceeded."""
        trade = _make_trade(
            setup_type="news_breakout",
            entry_price=150.0,
            stop_price=145.0,
            target_price=160.0,
        )
        entry_time = trade["entry_time"]
        now_utc = _now_utc_at_minutes(entry_time, 91)
        market_ts = now_utc

        # Price below entry, above stop, no positive indicators → close_time_expired
        result = evaluate_setup_aware_lifecycle(
            trade,
            [],
            now_utc=now_utc,
            now_et=_now_et_morning(),
            current_price=146.0,
            market_data_timestamp=market_ts,
        )

        assert result["state"] == "setup_time_limit_exceeded"
        assert result["decision"] == "close"
