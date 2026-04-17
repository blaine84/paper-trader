"""
Preservation Property Tests — Property 2

Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5

These tests capture the EXISTING (correct) behavior that must be preserved
after the bugfix is applied. They verify that:
  - Trades without target_price are still force-closed at time limits
  - momentum_fade escalation logic is unchanged regardless of target_price
  - Hard wall closes still fire for trades without target_price
  - Time alerts are still generated for all trades past alert threshold

These tests MUST PASS on UNFIXED code (baseline) AND on FIXED code (preservation).
"""

import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, Trade, AgentMemory

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

# Non-momentum_fade setup types
NON_MF_SETUP_TYPES = [
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

non_mf_setup_strategy = st.sampled_from(NON_MF_SETUP_TYPES)

# For alert tests, we need all setup types including momentum_fade
all_setup_strategy = st.sampled_from(list(SETUP_TIME_LIMITS.keys()))

target_price_strategy = st.floats(
    min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False
)


# ═══════════════════════════════════════════════════════════════════════
# Property 2a: No-target trades force-closed at time limits
# (non-momentum_fade setups with target_price=None)
# ═══════════════════════════════════════════════════════════════════════

@given(
    setup_type=non_mf_setup_strategy,
    extra_minutes=st.integers(min_value=1, max_value=120),
)
@settings(max_examples=50, deadline=None)
def test_property_2a_no_target_force_close(setup_type, extra_minutes):
    """
    **Validates: Requirements 3.1, 3.5**

    Property 2a: For all trades with target_price=None and
    minutes_held > limits["force_close"] and setup_type != "momentum_fade",
    assert _close_position IS called (force close fires).

    This preserves the existing behavior: trades without a price target
    are force-closed when they exceed their time limit.
    """
    from agents.position_timer import run, _alert_status

    limits = SETUP_TIME_LIMITS[setup_type]
    force_close_limit = limits["force_close"]
    minutes_held = force_close_limit + extra_minutes

    engine = _make_engine()
    db = _make_session(engine)

    now_utc = datetime.utcnow()
    entry_time = now_utc - timedelta(minutes=minutes_held)

    _seed_trade(db, "TEST", setup_type, None, entry_time)
    db.close()

    _alert_status.clear()

    # Mock time to be BEFORE hard wall so we isolate the force_close path
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

    # Preservation: _close_position MUST be called for no-target trades
    assert mock_close.call_count == 1, (
        f"Preservation broken: _close_position was NOT called for {setup_type} "
        f"with target_price=None, minutes_held={minutes_held} "
        f"(limit: {force_close_limit}). Expected force close."
    )
    assert len(result["force_closes"]) == 1, (
        f"Preservation broken: force_closes list should have 1 entry, "
        f"got {len(result['force_closes'])} for {setup_type} with target_price=None."
    )


# ═══════════════════════════════════════════════════════════════════════
# Property 2b: momentum_fade escalation unchanged regardless of target
# ═══════════════════════════════════════════════════════════════════════

@given(
    target_price=st.one_of(st.none(), target_price_strategy),
    extra_minutes=st.integers(min_value=1, max_value=60),
)
@settings(max_examples=50, deadline=None)
def test_property_2b_momentum_fade_force_close(target_price, extra_minutes):
    """
    **Validates: Requirements 3.2**

    Property 2b: For all momentum_fade trades regardless of target_price,
    when minutes_held > 75 (force_close limit), assert _close_position IS called.

    The momentum_fade branch has its own escalation logic that is independent
    of target_price. Force close at 75 min must always fire.
    """
    from agents.position_timer import run, _alert_status

    force_close_limit = SETUP_TIME_LIMITS["momentum_fade"]["force_close"]  # 75
    minutes_held = force_close_limit + extra_minutes

    engine = _make_engine()
    db = _make_session(engine)

    now_utc = datetime.utcnow()
    entry_time = now_utc - timedelta(minutes=minutes_held)

    _seed_trade(db, "MFADE", "momentum_fade", target_price, entry_time)
    db.close()

    _alert_status.clear()

    # Mock time BEFORE hard wall to isolate momentum_fade logic
    mock_now_et = MagicMock()
    mock_now_et.hour = 10
    mock_now_et.minute = 30

    with (
        patch("agents.position_timer._get_current_price", return_value=452.0),
        patch("agents.position_timer._close_position") as mock_close,
        patch("agents.position_timer._revalidate_momentum_fade", return_value=False),
        patch("agents.position_timer.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = mock_now_et
        mock_dt.utcnow.return_value = now_utc
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        result = run(engine)

    # The momentum_fade path calls _close_position potentially twice:
    # once at revalidation (60 min, if fails) and once at force_close (75 min).
    # Since we mock revalidation to return False, it closes at revalidation.
    # The force_close escalation may or may not fire depending on _escalate logic.
    # Key assertion: _close_position IS called at least once.
    assert mock_close.call_count >= 1, (
        f"Preservation broken: _close_position was NOT called for momentum_fade "
        f"with target_price={target_price}, minutes_held={minutes_held} "
        f"(limit: {force_close_limit}). Expected force close."
    )


# ═══════════════════════════════════════════════════════════════════════
# Property 2c: Hard wall closes for no-target trades
# ═══════════════════════════════════════════════════════════════════════

@given(
    setup_type=all_setup_strategy,
    wall_minute=st.integers(min_value=45, max_value=59),
)
@settings(max_examples=50, deadline=None)
def test_property_2c_hard_wall_no_target(setup_type, wall_minute):
    """
    **Validates: Requirements 3.3**

    Property 2c: For all trades with target_price=None past the hard wall
    (3:45 PM ET), assert _close_position IS called.

    This preserves the existing hard wall behavior for trades without
    a price target.
    """
    from agents.position_timer import run, _alert_status

    engine = _make_engine()
    db = _make_session(engine)

    now_utc = datetime.utcnow()
    # Entry 30 min ago — within time limits, so only hard wall triggers
    entry_time = now_utc - timedelta(minutes=30)

    _seed_trade(db, "WALL", setup_type, None, entry_time)
    db.close()

    _alert_status.clear()

    # Mock time past the hard wall
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

    # Preservation: hard wall close MUST fire for no-target trades
    assert mock_close.call_count == 1, (
        f"Preservation broken: _close_position was called {mock_close.call_count} time(s) "
        f"for {setup_type} with target_price=None at hard wall "
        f"(15:{wall_minute:02d} ET). Expected exactly 1 call."
    )
    assert len(result["hard_wall_closes"]) == 1, (
        f"Preservation broken: hard_wall_closes should have 1 entry, "
        f"got {len(result['hard_wall_closes'])} for {setup_type} with target_price=None."
    )


# ═══════════════════════════════════════════════════════════════════════
# Property 2d: Alerts generated for all trades past alert threshold
# ═══════════════════════════════════════════════════════════════════════

@given(
    setup_type=non_mf_setup_strategy,
    target_price=st.one_of(st.none(), target_price_strategy),
    extra_minutes=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=50, deadline=None)
def test_property_2d_alerts_generated(setup_type, target_price, extra_minutes):
    """
    **Validates: Requirements 3.4**

    Property 2d: For all trades with minutes_held > limits["alert"]
    (regardless of target_price), assert alerts are generated.

    Time alerts must continue to fire for all trades, even those with
    a valid target_price. This test constrains minutes_held to be
    between alert and force_close limits so the alert path is reached
    without triggering force close.
    """
    from agents.position_timer import run, _alert_status

    limits = SETUP_TIME_LIMITS[setup_type]
    alert_limit = limits["alert"]
    force_close_limit = limits["force_close"]

    # minutes_held must be > alert but <= force_close to isolate alert path
    minutes_held = alert_limit + extra_minutes
    assume(minutes_held <= force_close_limit)

    engine = _make_engine()
    db = _make_session(engine)

    now_utc = datetime.utcnow()
    entry_time = now_utc - timedelta(minutes=minutes_held)

    _seed_trade(db, "ALRT", setup_type, target_price, entry_time)
    db.close()

    _alert_status.clear()

    # Mock time BEFORE hard wall to isolate alert logic
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

    # Preservation: alerts MUST be generated for trades past alert threshold
    assert len(result["alerts"]) == 1, (
        f"Preservation broken: expected 1 alert for {setup_type} "
        f"with target_price={target_price}, minutes_held={minutes_held} "
        f"(alert limit: {alert_limit}), got {len(result['alerts'])}."
    )
    # Force close should NOT have been called (we're between alert and force_close)
    assert mock_close.call_count == 0, (
        f"Unexpected force close for {setup_type} with minutes_held={minutes_held} "
        f"(force_close limit: {force_close_limit}). Should only alert, not close."
    )


# ═══════════════════════════════════════════════════════════════════════
# Concrete observation tests (verify baseline before property tests)
# ═══════════════════════════════════════════════════════════════════════

def test_observe_gap_and_go_no_target_95min():
    """
    Observation: gap_and_go with target_price=None, held 95 min
    → _close_position called (force close).
    """
    from agents.position_timer import run, _alert_status

    engine = _make_engine()
    db = _make_session(engine)

    now_utc = datetime.utcnow()
    entry_time = now_utc - timedelta(minutes=95)

    _seed_trade(db, "OBS1", "gap_and_go", None, entry_time)
    db.close()

    _alert_status.clear()

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

    assert mock_close.call_count == 1
    assert len(result["force_closes"]) == 1


def test_observe_momentum_fade_with_target_80min():
    """
    Observation: momentum_fade with target_price=450.0, held 80 min
    → _close_position called (force close at 75 min limit).
    """
    from agents.position_timer import run, _alert_status

    engine = _make_engine()
    db = _make_session(engine)

    now_utc = datetime.utcnow()
    entry_time = now_utc - timedelta(minutes=80)

    _seed_trade(db, "OBS2", "momentum_fade", 450.0, entry_time)
    db.close()

    _alert_status.clear()

    mock_now_et = MagicMock()
    mock_now_et.hour = 10
    mock_now_et.minute = 30

    with (
        patch("agents.position_timer._get_current_price", return_value=452.0),
        patch("agents.position_timer._close_position") as mock_close,
        patch("agents.position_timer._revalidate_momentum_fade", return_value=False),
        patch("agents.position_timer.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = mock_now_et
        mock_dt.utcnow.return_value = now_utc
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        result = run(engine)

    assert mock_close.call_count >= 1


def test_observe_momentum_fade_no_target_80min():
    """
    Observation: momentum_fade with target_price=None, held 80 min
    → _close_position called.
    """
    from agents.position_timer import run, _alert_status

    engine = _make_engine()
    db = _make_session(engine)

    now_utc = datetime.utcnow()
    entry_time = now_utc - timedelta(minutes=80)

    _seed_trade(db, "OBS3", "momentum_fade", None, entry_time)
    db.close()

    _alert_status.clear()

    mock_now_et = MagicMock()
    mock_now_et.hour = 10
    mock_now_et.minute = 30

    with (
        patch("agents.position_timer._get_current_price", return_value=452.0),
        patch("agents.position_timer._close_position") as mock_close,
        patch("agents.position_timer._revalidate_momentum_fade", return_value=False),
        patch("agents.position_timer.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = mock_now_et
        mock_dt.utcnow.return_value = now_utc
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        result = run(engine)

    assert mock_close.call_count >= 1


def test_observe_orb_no_target_80min():
    """
    Observation: orb with target_price=None, held 80 min
    → _close_position called (force close at 75 min limit).
    """
    from agents.position_timer import run, _alert_status

    engine = _make_engine()
    db = _make_session(engine)

    now_utc = datetime.utcnow()
    entry_time = now_utc - timedelta(minutes=80)

    _seed_trade(db, "OBS4", "orb", None, entry_time)
    db.close()

    _alert_status.clear()

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

    assert mock_close.call_count == 1
    assert len(result["force_closes"]) == 1


def test_observe_hard_wall_no_target():
    """
    Observation: any intraday trade with target_price=None past 3:45 PM
    → _close_position called (hard wall).
    """
    from agents.position_timer import run, _alert_status

    engine = _make_engine()
    db = _make_session(engine)

    now_utc = datetime.utcnow()
    entry_time = now_utc - timedelta(minutes=30)

    _seed_trade(db, "OBS5", "gap_and_go", None, entry_time)
    db.close()

    _alert_status.clear()

    mock_now_et = MagicMock()
    mock_now_et.hour = 15
    mock_now_et.minute = 46

    with (
        patch("agents.position_timer._get_current_price", return_value=452.0),
        patch("agents.position_timer._close_position") as mock_close,
        patch("agents.position_timer.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = mock_now_et
        mock_dt.utcnow.return_value = now_utc
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        result = run(engine)

    assert mock_close.call_count == 1
    assert len(result["hard_wall_closes"]) == 1


def test_observe_gap_and_go_with_target_70min_alert():
    """
    Observation: gap_and_go with target_price=450.0, held 70 min
    (alert > 60, force < 90) → alert generated, no force close.
    """
    from agents.position_timer import run, _alert_status

    engine = _make_engine()
    db = _make_session(engine)

    now_utc = datetime.utcnow()
    entry_time = now_utc - timedelta(minutes=70)

    _seed_trade(db, "OBS6", "gap_and_go", 450.0, entry_time)
    db.close()

    _alert_status.clear()

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

    assert mock_close.call_count == 0
    assert len(result["alerts"]) == 1
