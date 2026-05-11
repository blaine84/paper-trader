"""Unit tests for _check_exit_now and _parse_dt helpers."""

from datetime import datetime, timezone, timedelta

from utils.position_lifecycle_governance import _check_exit_now, _parse_dt


# ---------------------------------------------------------------------------
# _parse_dt tests
# ---------------------------------------------------------------------------


def test_parse_dt_none_returns_none():
    assert _parse_dt(None) is None


def test_parse_dt_invalid_string_returns_none():
    assert _parse_dt("not-a-date") is None


def test_parse_dt_empty_string_returns_none():
    assert _parse_dt("") is None


def test_parse_dt_integer_returns_none():
    assert _parse_dt(12345) is None


def test_parse_dt_aware_datetime_returned_as_is():
    dt = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    result = _parse_dt(dt)
    assert result is dt


def test_parse_dt_naive_datetime_gets_utc():
    dt = datetime(2025, 1, 15, 10, 0, 0)
    result = _parse_dt(dt)
    assert result.tzinfo == timezone.utc
    assert result.year == 2025


def test_parse_dt_iso_string_with_tz():
    result = _parse_dt("2025-01-15T10:00:00+00:00")
    assert result == datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)


def test_parse_dt_iso_string_naive_gets_utc():
    result = _parse_dt("2025-01-15T10:00:00")
    assert result == datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# _check_exit_now tests
# ---------------------------------------------------------------------------


def _make_trade(**overrides):
    """Create a minimal trade dict for testing."""
    trade = {
        "id": 1,
        "symbol": "AAPL",
        "profile": "moderate",
        "setup_type": "news_catalyst",
        "entry_time": datetime(2025, 1, 15, 9, 30, 0, tzinfo=timezone.utc),
        "status": "open",
    }
    trade.update(overrides)
    return trade


def _make_exit_now_event(decided_at="2025-01-15T14:00:00+00:00", decided_by="pm_john", **extra):
    """Create a valid EXIT_NOW event."""
    payload = {
        "decision": "EXIT_NOW",
        "decided_at": decided_at,
        "decided_by": decided_by,
    }
    payload.update(extra)
    return {
        "event_type": "news_reconfirmation_submitted",
        "timestamp": "2025-01-15T14:00:00+00:00",
        "payload": payload,
    }


NOW_UTC = datetime(2025, 1, 15, 15, 0, 0, tzinfo=timezone.utc)


def test_no_events_returns_none():
    trade = _make_trade()
    result = _check_exit_now([], trade, NOW_UTC)
    assert result is None


def test_no_exit_now_events_returns_none():
    trade = _make_trade()
    events = [
        {
            "event_type": "news_reconfirmation_submitted",
            "timestamp": "2025-01-15T14:00:00+00:00",
            "payload": {
                "decision": "RECONFIRM_AND_HOLD",
                "decided_at": "2025-01-15T14:00:00+00:00",
                "decided_by": "pm_john",
            },
        }
    ]
    result = _check_exit_now(events, trade, NOW_UTC)
    assert result is None


def test_valid_exit_now_returns_close_required():
    trade = _make_trade()
    events = [_make_exit_now_event()]
    result = _check_exit_now(events, trade, NOW_UTC)

    assert result is not None
    assert result["decision"] == "close"
    assert result["state"] == "close_required"
    assert result["reason_type"] == "pm_operator_exit_requested"
    assert result["requires_event"] is True
    assert result["close_reason"] == "PM/operator requested exit (EXIT_NOW)"
    assert result["metadata"]["decided_by"] == "pm_john"


def test_exit_now_missing_decided_at_is_invalid():
    trade = _make_trade()
    events = [
        {
            "event_type": "news_reconfirmation_submitted",
            "timestamp": "2025-01-15T14:00:00+00:00",
            "payload": {
                "decision": "EXIT_NOW",
                "decided_by": "pm_john",
                # decided_at missing
            },
        }
    ]
    result = _check_exit_now(events, trade, NOW_UTC)
    assert result is None


def test_exit_now_missing_decided_by_is_invalid():
    trade = _make_trade()
    events = [
        {
            "event_type": "news_reconfirmation_submitted",
            "timestamp": "2025-01-15T14:00:00+00:00",
            "payload": {
                "decision": "EXIT_NOW",
                "decided_at": "2025-01-15T14:00:00+00:00",
                # decided_by missing
            },
        }
    ]
    result = _check_exit_now(events, trade, NOW_UTC)
    assert result is None


def test_exit_now_empty_decided_by_is_invalid():
    trade = _make_trade()
    events = [
        {
            "event_type": "news_reconfirmation_submitted",
            "timestamp": "2025-01-15T14:00:00+00:00",
            "payload": {
                "decision": "EXIT_NOW",
                "decided_at": "2025-01-15T14:00:00+00:00",
                "decided_by": "   ",
            },
        }
    ]
    result = _check_exit_now(events, trade, NOW_UTC)
    assert result is None


def test_exit_now_unparseable_decided_at_is_invalid():
    trade = _make_trade()
    events = [
        {
            "event_type": "news_reconfirmation_submitted",
            "timestamp": "2025-01-15T14:00:00+00:00",
            "payload": {
                "decision": "EXIT_NOW",
                "decided_at": "not-a-date",
                "decided_by": "pm_john",
            },
        }
    ]
    result = _check_exit_now(events, trade, NOW_UTC)
    assert result is None


def test_latest_exit_now_wins():
    """When multiple valid EXIT_NOW events exist, the latest by decided_at wins."""
    trade = _make_trade()
    events = [
        _make_exit_now_event(decided_at="2025-01-15T12:00:00+00:00", decided_by="pm_early"),
        _make_exit_now_event(decided_at="2025-01-15T14:00:00+00:00", decided_by="pm_late"),
    ]
    result = _check_exit_now(events, trade, NOW_UTC)

    assert result is not None
    assert result["metadata"]["decided_by"] == "pm_late"


def test_wrong_event_type_ignored():
    """Events with event_type != 'news_reconfirmation_submitted' are ignored."""
    trade = _make_trade()
    events = [
        {
            "event_type": "overnight_authorized",
            "timestamp": "2025-01-15T14:00:00+00:00",
            "payload": {
                "decision": "EXIT_NOW",
                "decided_at": "2025-01-15T14:00:00+00:00",
                "decided_by": "pm_john",
            },
        }
    ]
    result = _check_exit_now(events, trade, NOW_UTC)
    assert result is None


def test_non_dict_payload_ignored():
    """Events with non-dict payload are skipped gracefully."""
    trade = _make_trade()
    events = [
        {
            "event_type": "news_reconfirmation_submitted",
            "timestamp": "2025-01-15T14:00:00+00:00",
            "payload": "not a dict",
        }
    ]
    result = _check_exit_now(events, trade, NOW_UTC)
    assert result is None


def test_exit_now_with_datetime_object_decided_at():
    """decided_at can be a datetime object, not just a string."""
    trade = _make_trade()
    dt = datetime(2025, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
    events = [_make_exit_now_event(decided_at=dt)]
    result = _check_exit_now(events, trade, NOW_UTC)

    assert result is not None
    assert result["decision"] == "close"
    assert result["state"] == "close_required"
