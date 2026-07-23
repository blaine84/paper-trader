"""
Bug Condition Exploration Test — Property 1

Validates: Requirements 1.1, 1.2, 1.3, 1.4

This test encodes the EXPECTED (correct) behavior: when a trade has a valid
target_price and is not a momentum_fade setup, position_timer.run() should
NOT call _close_position, deferring the exit to price_monitor.

On UNFIXED code this test is EXPECTED TO FAIL — failure confirms the bug
exists (the missing target_price guard causes _close_position to fire).
"""

import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, call

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, Trade, AgentMemory

# NOTE: These tests were written against the pre-refactor, monolithic
# position_timer implementation. As of commit 09d7b10 ("feat: implement open
# position lifecycle governance evaluator"), agents/position_timer.py was
# rewritten into a thin executor and all policy logic moved into
# utils/position_lifecycle_governance.py. The symbols these tests depend on
# (_alert_status, _revalidate_momentum_fade, the module-level SETUP_TIME_LIMITS
# table) and the run() result shape they assert on (force_closes / alerts) no
# longer exist in agents.position_timer and have no equivalent. Skipped rather
# than deleted to preserve the original test intent for reference.
pytestmark = pytest.mark.skip(
    reason="Targets pre-refactor position_timer API (_alert_status, "
    "_revalidate_momentum_fade, SETUP_TIME_LIMITS, force_closes result key) "
    "removed in commit 09d7b10 when policy moved to "
    "utils.position_lifecycle_governance; target_price guard is now handled by "
    "the governance evaluator + swing reclassification."
)

# ── Constants from position_timer ──
SETUP_TIME_LIMITS = {
    "momentum_fade": {"stale": 35, "alert": 45, "revalidate": 60, "force_close": 75},
    "gap_and_go":    {"alert": 60, "force_close": 90},
    "vwap_reclaim":  {"alert": 60, "force_close": 90},
    "orb":           {"alert": 45, "force_close": 75},
    "trend_pullback": {"alert": 90, "force_close": 120},
    "news_catalyst": {"alert": 60, "force_close": 90},
    "short_squeeze": {"alert": 30, "force_close": 60},
}

# Setup types that trigger the bug (all non-momentum_fade intraday setups)
BUG_SETUP_TYPES = [
    "gap_and_go", "vwap_reclaim", "orb",
    "news_catalyst", "trend_pullback", "short_squeeze",
]

HARD_WALL_HOUR = 15
HARD_WALL_MINUTE = 45


# ── Helpers ──

def _make_engine():
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    return engine


def _make_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()


def _seed_trade(db, symbol, setup_type, target_price, entry_time, profile="moderate"):
    """Insert an open trade and a matching analyst signal."""
    trade = Trade(
        symbol=symbol,
        direction="LONG",
        quantity=100,
        entry_price=450.0,
        entry_time=entry_time,
        status="open",
        stop_price=445.0,
        target_price=target_price,
        profile=profile,
    )
    db.add(trade)
    # Seed analyst signal so _get_setup_type_for_trade returns the setup_type
    db.add(AgentMemory(
        agent="analyst",
        symbol=symbol,
        key="signal",
        value=json.dumps({"setup_type": setup_type}),
        timestamp=entry_time - timedelta(minutes=1),
    ))
    db.commit()
    return trade


# ── Hypothesis strategies ──

setup_type_strategy = st.sampled_from(BUG_SETUP_TYPES)

target_price_strategy = st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False)


# ── Property-based test: time-exceeded path ──

@given(
    setup_type=setup_type_strategy,
    target_price=target_price_strategy,
    extra_minutes=st.integers(min_value=1, max_value=120),
)
@settings(max_examples=50, deadline=None)
def test_property_no_force_close_when_target_set_time_exceeded(
    setup_type, target_price, extra_minutes
):
    """
    **Validates: Requirements 1.1, 1.2, 1.3**

    Property: For any trade where setup_type is not momentum_fade,
    target_price is set (> 0), and minutes_held > force_close limit,
    _close_position should NOT be called.

    On unfixed code this FAILS because the guard is missing.
    """
    from agents.position_timer import run, _alert_status

    force_close_limit = SETUP_TIME_LIMITS[setup_type]["force_close"]
    minutes_held = force_close_limit + extra_minutes

    engine = _make_engine()
    db = _make_session(engine)

    now_utc = datetime.utcnow()
    entry_time = now_utc - timedelta(minutes=minutes_held)

    _seed_trade(db, "TEST", setup_type, target_price, entry_time)
    db.close()

    # Clear alert status from prior runs
    _alert_status.clear()

    # Mock datetime.now(et_tz) to return a time BEFORE the hard wall
    # so we isolate the time-exceeded path only
    mock_now_et = MagicMock()
    mock_now_et.hour = 10
    mock_now_et.minute = 30

    with (
        patch("agents.position_timer._get_current_price", return_value=452.0),
        patch("agents.position_timer._close_position") as mock_close,
        patch("agents.position_timer.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = mock_now_et
        mock_dt.utcnow.return_value = now_utc
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        result = run(engine)

    # EXPECTED BEHAVIOR: _close_position should NOT be called
    # because the trade has a valid target_price.
    # On UNFIXED code, this assertion FAILS — proving the bug.
    assert mock_close.call_count == 0, (
        f"Bug confirmed: _close_position was called {mock_close.call_count} time(s) "
        f"for {setup_type} with target_price={target_price}, "
        f"minutes_held={minutes_held} (limit: {force_close_limit}). "
        f"The trade has an active price target — force close should be skipped."
    )


# ── Property-based test: hard wall path ──

@given(
    setup_type=setup_type_strategy,
    target_price=target_price_strategy,
    wall_minute=st.integers(min_value=45, max_value=59),
)
@settings(max_examples=50, deadline=None)
def test_property_no_force_close_when_target_set_hard_wall(
    setup_type, target_price, wall_minute
):
    """
    **Validates: Requirements 1.4**

    Property: For any intraday trade where target_price is set (> 0)
    and the hard wall (3:45 PM ET) has been reached, _close_position
    should NOT be called.

    On unfixed code this FAILS because the guard is missing.
    """
    from agents.position_timer import run, _alert_status

    engine = _make_engine()
    db = _make_session(engine)

    now_utc = datetime.utcnow()
    # Entry 30 min ago — well within limits, so only the hard wall triggers
    entry_time = now_utc - timedelta(minutes=30)

    _seed_trade(db, "WALL", setup_type, target_price, entry_time)
    db.close()

    _alert_status.clear()

    # Mock datetime.now(et_tz) to return past the hard wall
    mock_now_et = MagicMock()
    mock_now_et.hour = HARD_WALL_HOUR
    mock_now_et.minute = wall_minute

    with (
        patch("agents.position_timer._get_current_price", return_value=452.0),
        patch("agents.position_timer._close_position") as mock_close,
        patch("agents.position_timer.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = mock_now_et
        mock_dt.utcnow.return_value = now_utc
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        result = run(engine)

    # EXPECTED BEHAVIOR: _close_position should NOT be called
    # because the trade has a valid target_price.
    # On UNFIXED code, this assertion FAILS — proving the bug.
    assert mock_close.call_count == 0, (
        f"Bug confirmed: _close_position was called {mock_close.call_count} time(s) "
        f"for {setup_type} with target_price={target_price} at hard wall "
        f"(15:{wall_minute:02d} ET). "
        f"The trade has an active price target — hard wall close should be skipped."
    )


# ── Concrete example tests ──

def test_gap_and_go_with_target_95min_force_closed():
    """
    **Validates: Requirements 1.1, 1.2**

    Concrete example: gap_and_go with target_price=455.0, held 95 min
    (limit: 90). On unfixed code, _close_position IS called — bug.
    """
    from agents.position_timer import run, _alert_status

    engine = _make_engine()
    db = _make_session(engine)

    now_utc = datetime.utcnow()
    entry_time = now_utc - timedelta(minutes=95)

    _seed_trade(db, "AAPL", "gap_and_go", 455.0, entry_time)
    db.close()

    _alert_status.clear()

    mock_now_et = MagicMock()
    mock_now_et.hour = 10
    mock_now_et.minute = 30

    with (
        patch("agents.position_timer._get_current_price", return_value=452.30),
        patch("agents.position_timer._close_position") as mock_close,
        patch("agents.position_timer.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = mock_now_et
        mock_dt.utcnow.return_value = now_utc
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        result = run(engine)

    assert mock_close.call_count == 0, (
        "Bug confirmed: gap_and_go with target_price=455.0 held 95 min was force-closed. "
        "Expected: skip force close, defer to price_monitor."
    )


def test_hard_wall_346pm_with_target_force_closed():
    """
    **Validates: Requirements 1.4**

    Concrete example: gap_and_go with target_price=455.0 at 3:46 PM ET.
    On unfixed code, _close_position IS called — bug.
    """
    from agents.position_timer import run, _alert_status

    engine = _make_engine()
    db = _make_session(engine)

    now_utc = datetime.utcnow()
    entry_time = now_utc - timedelta(minutes=30)

    _seed_trade(db, "MSFT", "gap_and_go", 455.0, entry_time)
    db.close()

    _alert_status.clear()

    mock_now_et = MagicMock()
    mock_now_et.hour = 15
    mock_now_et.minute = 46

    with (
        patch("agents.position_timer._get_current_price", return_value=452.30),
        patch("agents.position_timer._close_position") as mock_close,
        patch("agents.position_timer.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = mock_now_et
        mock_dt.utcnow.return_value = now_utc
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        result = run(engine)

    assert mock_close.call_count == 0, (
        "Bug confirmed: gap_and_go with target_price=455.0 at 3:46 PM ET was force-closed "
        "via hard wall. Expected: skip hard wall close, defer to price_monitor."
    )


def test_orb_with_target_80min_force_closed():
    """
    **Validates: Requirements 1.3**

    Concrete example: orb with target_price=450.0, held 80 min
    (limit: 75). On unfixed code, _close_position IS called — bug.
    """
    from agents.position_timer import run, _alert_status

    engine = _make_engine()
    db = _make_session(engine)

    now_utc = datetime.utcnow()
    entry_time = now_utc - timedelta(minutes=80)

    _seed_trade(db, "TSLA", "orb", 450.0, entry_time)
    db.close()

    _alert_status.clear()

    mock_now_et = MagicMock()
    mock_now_et.hour = 10
    mock_now_et.minute = 30

    with (
        patch("agents.position_timer._get_current_price", return_value=445.0),
        patch("agents.position_timer._close_position") as mock_close,
        patch("agents.position_timer.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = mock_now_et
        mock_dt.utcnow.return_value = now_utc
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        result = run(engine)

    assert mock_close.call_count == 0, (
        "Bug confirmed: orb with target_price=450.0 held 80 min was force-closed. "
        "Expected: skip force close, defer to price_monitor."
    )
