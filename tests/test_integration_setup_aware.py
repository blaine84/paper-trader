"""Integration tests for setup-aware exit governance in position_lifecycle_governance.

Tests the priority chain, parameter passing, shadow mode, and backward
compatibility of the setup-aware evaluator integrated into
evaluate_position_lifecycle.

Requirements validated: 7.1, 7.4, 7.7
"""

from datetime import datetime, timedelta, timezone

from utils.position_lifecycle_governance import evaluate_position_lifecycle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW_UTC = datetime(2025, 5, 26, 14, 0, 0, tzinfo=timezone.utc)
# 10:00 AM ET — well before EOD hard wall (3:45 PM ET)
NOW_ET = datetime(2025, 5, 26, 10, 0, 0)


def _make_trade(**overrides) -> dict:
    """Create a minimal intraday trade dict."""
    trade = {
        "id": 100,
        "symbol": "AMD",
        "profile": "moderate",
        "direction": "LONG",
        "entry_price": 150.0,
        "entry_time": NOW_UTC - timedelta(minutes=30),
        "stop_price": 147.0,
        "target_price": 160.0,
        "setup_type": "momentum_fade",
        "status": "open",
        "quantity": 10,
        "thesis": "Momentum fade setup",
        "invalidators": None,
    }
    trade.update(overrides)
    return trade


def _make_exit_now_event(decided_at=None, decided_by="pm_operator"):
    """Create a valid EXIT_NOW governance event."""
    if decided_at is None:
        decided_at = (NOW_UTC - timedelta(minutes=5)).isoformat()
    return {
        "event_type": "news_reconfirmation_submitted",
        "timestamp": decided_at,
        "payload": {
            "decision": "EXIT_NOW",
            "decided_at": decided_at,
            "decided_by": decided_by,
        },
    }


def _make_overnight_auth_event(expires_at=None):
    """Create a valid overnight_authorized event."""
    if expires_at is None:
        expires_at = (NOW_UTC + timedelta(hours=12)).isoformat()
    return {
        "event_type": "overnight_authorized",
        "timestamp": NOW_UTC.isoformat(),
        "payload": {
            "expires_at": expires_at,
        },
    }


# ---------------------------------------------------------------------------
# Test 1: Priority chain — closed/cancelled trade returns "skip" (Priority 1)
# ---------------------------------------------------------------------------


def test_priority_1_closed_trade_returns_skip():
    """When a trade is closed/cancelled, evaluate_position_lifecycle returns
    'skip' — priority 1 takes precedence over setup-aware logic.
    """
    trade = _make_trade(status="closed")
    result = evaluate_position_lifecycle(
        trade,
        [],
        now_utc=NOW_UTC,
        now_et=NOW_ET,
        current_price=155.0,
    )
    assert result["decision"] == "skip"
    assert result["state"] == "skipped"
    assert result["reason_type"] == "position_already_closed"


# ---------------------------------------------------------------------------
# Test 2: Priority chain — EXIT_NOW returns close_required (Priority 2)
# ---------------------------------------------------------------------------


def test_priority_2_exit_now_returns_close_required():
    """When a trade has an EXIT_NOW event, evaluate_position_lifecycle returns
    close_required — priority 2 takes precedence over setup-aware logic.
    """
    trade = _make_trade()
    events = [_make_exit_now_event()]
    result = evaluate_position_lifecycle(
        trade,
        events,
        now_utc=NOW_UTC,
        now_et=NOW_ET,
        current_price=155.0,
    )
    assert result["decision"] == "close"
    assert result["state"] == "close_required"
    assert result["reason_type"] == "pm_operator_exit_requested"


# ---------------------------------------------------------------------------
# Test 3: Setup-aware evaluator — momentum_fade past force_close returns close
# ---------------------------------------------------------------------------


def test_setup_aware_force_close_momentum_fade():
    """For an intraday trade past its force_close_minutes (momentum_fade at
    76 min, limit=75), evaluate_position_lifecycle returns a close decision
    from the setup-aware evaluator with state=setup_time_limit_exceeded.
    """
    # momentum_fade: force_close_minutes=75, so 76 min should trigger close
    entry_time = NOW_UTC - timedelta(minutes=76)
    trade = _make_trade(
        setup_type="momentum_fade",
        entry_time=entry_time,
    )
    result = evaluate_position_lifecycle(
        trade,
        [],
        now_utc=NOW_UTC,
        now_et=NOW_ET,
        current_price=149.0,
    )
    assert result["decision"] == "close"
    assert result["state"] == "setup_time_limit_exceeded"
    assert "setup_time_limit" in result["reason_type"]


# ---------------------------------------------------------------------------
# Test 4: Setup-aware evaluator — intraday trade below alert returns hold
# ---------------------------------------------------------------------------


def test_setup_aware_below_alert_returns_intraday_ok():
    """For an intraday trade below alert_minutes, evaluate_position_lifecycle
    returns intraday_ok (hold) from the setup-aware evaluator since no
    governance action is required.
    """
    # momentum_fade: alert_minutes=35, so 20 min should be below alert
    entry_time = NOW_UTC - timedelta(minutes=20)
    trade = _make_trade(
        setup_type="momentum_fade",
        entry_time=entry_time,
    )
    result = evaluate_position_lifecycle(
        trade,
        [],
        now_utc=NOW_UTC,
        now_et=NOW_ET,
        current_price=151.0,
    )
    # The setup-aware evaluator returns hold/intraday_ok which doesn't
    # trigger the "close" or "warn" check, so we fall through to priority 7
    # default which also returns intraday_ok
    assert result["decision"] == "allow"
    assert result["state"] == "intraday_ok"


# ---------------------------------------------------------------------------
# Test 5: New parameters accepted — market_data_timestamp and shadow_mode
# ---------------------------------------------------------------------------


def test_new_parameters_accepted_without_error():
    """Verify evaluate_position_lifecycle accepts market_data_timestamp and
    shadow_mode parameters without raising any errors.
    """
    trade = _make_trade()
    market_ts = NOW_UTC - timedelta(seconds=5)

    # Should not raise
    result = evaluate_position_lifecycle(
        trade,
        [],
        now_utc=NOW_UTC,
        now_et=NOW_ET,
        current_price=151.0,
        market_data_timestamp=market_ts,
        shadow_mode=False,
    )
    assert result is not None
    assert "decision" in result

    # Also test with shadow_mode=True
    result_shadow = evaluate_position_lifecycle(
        trade,
        [],
        now_utc=NOW_UTC,
        now_et=NOW_ET,
        current_price=151.0,
        market_data_timestamp=market_ts,
        shadow_mode=True,
    )
    assert result_shadow is not None
    assert "decision" in result_shadow


# ---------------------------------------------------------------------------
# Test 6: Shadow mode — metadata includes shadow_mode and legacy_decision
# ---------------------------------------------------------------------------


def test_shadow_mode_includes_metadata():
    """When shadow_mode=True, the returned decision metadata includes
    'shadow_mode': True and 'legacy_decision' dict.
    """
    # Use a trade past force_close to trigger a decisive setup-aware result
    # momentum_fade: force_close_minutes=75
    entry_time = NOW_UTC - timedelta(minutes=76)
    trade = _make_trade(
        setup_type="momentum_fade",
        entry_time=entry_time,
    )
    result = evaluate_position_lifecycle(
        trade,
        [],
        now_utc=NOW_UTC,
        now_et=NOW_ET,
        current_price=149.0,
        market_data_timestamp=NOW_UTC - timedelta(seconds=5),
        shadow_mode=True,
    )
    # The setup-aware evaluator should return a close decision with shadow metadata
    assert result["decision"] == "close"
    assert result["metadata"].get("shadow_mode") is True
    assert "legacy_decision" in result["metadata"]
    assert isinstance(result["metadata"]["legacy_decision"], dict)


# ---------------------------------------------------------------------------
# Test 7: Backward compatibility — overnight_authorized still works
# ---------------------------------------------------------------------------


def test_backward_compatibility_overnight_authorized():
    """A trade with valid overnight authorization still returns
    overnight_authorized (priority 7 default unchanged).
    """
    # Trade below alert threshold so setup-aware evaluator returns hold/intraday_ok
    entry_time = NOW_UTC - timedelta(minutes=20)
    trade = _make_trade(
        setup_type="momentum_fade",
        entry_time=entry_time,
    )
    events = [_make_overnight_auth_event()]

    result = evaluate_position_lifecycle(
        trade,
        events,
        now_utc=NOW_UTC,
        now_et=NOW_ET,
        current_price=151.0,
    )
    assert result["decision"] == "allow"
    assert result["state"] == "overnight_authorized"
    assert result["reason_type"] == "overnight_authorization_valid"
