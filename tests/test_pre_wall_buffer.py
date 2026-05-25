"""Unit tests for the EOD pre-wall buffer check in evaluate_setup_aware_lifecycle.

Validates Requirement 7.6: IF a setup-aware extension is active and the trade
reaches within 15 minutes of the EOD_Hard_Wall, THEN THE Lifecycle_Evaluator
SHALL revoke the extension and close the position before the hard wall,
regardless of revalidation status.

The pre-wall buffer fires when now_et time >= 3:30 PM ET (for the default
3:45 PM hard wall), producing a close decision with reason_type
`setup_pre_wall_buffer_close` and metadata `extension_revoked: True`.
"""

from datetime import datetime, time, timedelta, timezone

import pytest

from utils.setup_aware_evaluator import evaluate_setup_aware_lifecycle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# US/Eastern offset for EDT (UTC-4)
ET_OFFSET = timezone(timedelta(hours=-4))


def _make_trade(
    setup_type: str = "news_breakout",
    entry_time: datetime | None = None,
    direction: str = "LONG",
    entry_price: float = 150.0,
    stop_price: float = 148.0,
    target_price: float = 155.0,
) -> dict:
    """Create a minimal trade dict for testing."""
    if entry_time is None:
        # Default: entered at 10:00 AM ET (14:00 UTC during EDT)
        entry_time = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
    return {
        "id": 1,
        "symbol": "AMD",
        "profile": "moderate",
        "direction": direction,
        "entry_price": entry_price,
        "entry_time": entry_time,
        "stop_price": stop_price,
        "target_price": target_price,
        "setup_type": setup_type,
        "status": "open",
        "quantity": 100,
        "thesis": "Catalyst-driven breakout above VWAP",
        "invalidators": {"support": 148.5},
    }


def _make_now(hour: int, minute: int, second: int = 0) -> tuple[datetime, datetime]:
    """Create (now_utc, now_et) pair for a given ET time on 2026-06-15.

    Returns UTC and ET datetimes for the same instant.
    """
    now_et = datetime(2026, 6, 15, hour, minute, second, tzinfo=ET_OFFSET)
    now_utc = now_et.astimezone(timezone.utc)
    return now_utc, now_et


# ---------------------------------------------------------------------------
# Tests: Pre-wall buffer triggers close at 3:30 PM ET
# ---------------------------------------------------------------------------


class TestPreWallBufferClose:
    """Tests that the pre-wall buffer closes trades at or after 3:30 PM ET."""

    def test_trade_at_330pm_et_triggers_pre_wall_close(self):
        """Trade evaluated at exactly 3:30 PM ET → close with pre_wall_buffer_close."""
        trade = _make_trade(setup_type="news_breakout")
        now_utc, now_et = _make_now(15, 30)

        result = evaluate_setup_aware_lifecycle(
            trade,
            events=[],
            now_utc=now_utc,
            now_et=now_et,
            current_price=152.0,
            market_data_timestamp=now_utc - timedelta(seconds=5),
        )

        assert result["decision"] == "close"
        assert result["state"] == "setup_time_limit_exceeded"
        assert result["reason_type"] == "setup_pre_wall_buffer_close"
        assert result["metadata"]["extension_revoked"] is True

    def test_trade_at_331pm_et_triggers_pre_wall_close(self):
        """Trade evaluated at 3:31 PM ET → close with pre_wall_buffer_close."""
        trade = _make_trade(setup_type="news_breakout")
        now_utc, now_et = _make_now(15, 31)

        result = evaluate_setup_aware_lifecycle(
            trade,
            events=[],
            now_utc=now_utc,
            now_et=now_et,
            current_price=152.0,
            market_data_timestamp=now_utc - timedelta(seconds=5),
        )

        assert result["decision"] == "close"
        assert result["state"] == "setup_time_limit_exceeded"
        assert result["reason_type"] == "setup_pre_wall_buffer_close"
        assert result["metadata"]["extension_revoked"] is True

    def test_trade_at_329pm_et_not_closed_by_pre_wall_buffer(self):
        """Trade evaluated at 3:29 PM ET → NOT closed by pre-wall buffer."""
        trade = _make_trade(setup_type="news_breakout")
        now_utc, now_et = _make_now(15, 29)

        result = evaluate_setup_aware_lifecycle(
            trade,
            events=[],
            now_utc=now_utc,
            now_et=now_et,
            current_price=152.0,
            market_data_timestamp=now_utc - timedelta(seconds=5),
        )

        # Should NOT be closed by pre-wall buffer
        assert result["reason_type"] != "setup_pre_wall_buffer_close"

    def test_trade_at_344pm_et_triggers_pre_wall_close(self):
        """Trade evaluated at 3:44 PM ET → close with pre_wall_buffer_close."""
        trade = _make_trade(setup_type="news_breakout")
        now_utc, now_et = _make_now(15, 44)

        result = evaluate_setup_aware_lifecycle(
            trade,
            events=[],
            now_utc=now_utc,
            now_et=now_et,
            current_price=152.0,
            market_data_timestamp=now_utc - timedelta(seconds=5),
        )

        assert result["decision"] == "close"
        assert result["state"] == "setup_time_limit_exceeded"
        assert result["reason_type"] == "setup_pre_wall_buffer_close"
        assert result["metadata"]["extension_revoked"] is True


class TestPreWallBufferWithExtensionEligible:
    """Tests that pre-wall buffer revokes extensions for extension-eligible trades."""

    def test_news_breakout_with_active_extension_at_330pm_revoked(self):
        """Extension-eligible trade (news_breakout) with active extension at 3:30 PM → revoked, close."""
        # Entry at 12:00 PM ET → held ~210 min by 3:30 PM (past revalidation boundaries)
        entry_time = datetime(2026, 6, 15, 16, 0, 0, tzinfo=timezone.utc)  # 12:00 PM ET
        trade = _make_trade(
            setup_type="news_breakout",
            entry_time=entry_time,
            entry_price=150.0,
            stop_price=148.0,
            target_price=155.0,
        )
        now_utc, now_et = _make_now(15, 30)

        # Simulate an active extension via a prior revalidation hold event
        events = [
            {
                "event_type": "setup_exit_revalidated_hold",
                "minutes_held": 120,
                "next_limit_if_extended": 150,
                "decision_outcome": "hold_valid_until_next_window",
            }
        ]

        result = evaluate_setup_aware_lifecycle(
            trade,
            events=events,
            now_utc=now_utc,
            now_et=now_et,
            current_price=152.0,
            market_data_timestamp=now_utc - timedelta(seconds=5),
        )

        assert result["decision"] == "close"
        assert result["state"] == "setup_time_limit_exceeded"
        assert result["reason_type"] == "setup_pre_wall_buffer_close"
        assert result["metadata"]["extension_revoked"] is True


class TestPreWallBufferNonExtensionEligible:
    """Tests that pre-wall buffer also closes non-extension-eligible trades."""

    def test_momentum_fade_at_330pm_closed_by_pre_wall_buffer(self):
        """Non-extension-eligible trade (momentum_fade) at 3:30 PM → also closed by pre-wall buffer."""
        # Entry at 2:30 PM ET → held 60 min by 3:30 PM
        entry_time = datetime(2026, 6, 15, 18, 30, 0, tzinfo=timezone.utc)  # 2:30 PM ET
        trade = _make_trade(
            setup_type="momentum_fade",
            entry_time=entry_time,
            entry_price=100.0,
            stop_price=99.0,
            target_price=102.0,
        )
        now_utc, now_et = _make_now(15, 30)

        result = evaluate_setup_aware_lifecycle(
            trade,
            events=[],
            now_utc=now_utc,
            now_et=now_et,
            current_price=100.5,
            market_data_timestamp=now_utc - timedelta(seconds=5),
        )

        assert result["decision"] == "close"
        assert result["state"] == "setup_time_limit_exceeded"
        assert result["reason_type"] == "setup_pre_wall_buffer_close"
        assert result["metadata"]["extension_revoked"] is True

    def test_unknown_setup_at_330pm_closed_by_pre_wall_buffer(self):
        """Unknown setup type at 3:30 PM → also closed by pre-wall buffer."""
        entry_time = datetime(2026, 6, 15, 18, 0, 0, tzinfo=timezone.utc)  # 2:00 PM ET
        trade = _make_trade(
            setup_type="some_unknown_type",
            entry_time=entry_time,
            entry_price=100.0,
            stop_price=99.0,
            target_price=102.0,
        )
        now_utc, now_et = _make_now(15, 30)

        result = evaluate_setup_aware_lifecycle(
            trade,
            events=[],
            now_utc=now_utc,
            now_et=now_et,
            current_price=100.5,
            market_data_timestamp=now_utc - timedelta(seconds=5),
        )

        assert result["decision"] == "close"
        assert result["state"] == "setup_time_limit_exceeded"
        assert result["reason_type"] == "setup_pre_wall_buffer_close"
        assert result["metadata"]["extension_revoked"] is True
