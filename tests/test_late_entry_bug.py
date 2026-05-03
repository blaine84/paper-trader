"""
Bug Condition Exploration Test — Late Entry & Missing Swing Reclassification

Validates: Requirements 1.1, 1.3, 1.4, 1.5

This test encodes the EXPECTED (correct) behavior:
  1a — gap_and_go entries outside the 60-min window should be rejected (action="PASS")
  1b — intraday trades with target_price surviving past force_close should be
       reclassified to "swing" in the DB
  1c — intraday trades with target_price at the 3:45 PM hard wall should be
       reclassified to "swing" in the DB

On UNFIXED code these tests are EXPECTED TO FAIL — failure confirms the bugs exist:
  - No entry timing gate in PM Phase 2
  - No swing reclassification logic in position_timer
"""

import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, Trade, AgentMemory, Position, Balance

# ── Constants matching the design doc ──

ENTRY_WINDOW_LIMITS = {
    "gap_and_go": 60,
    "orb": 60,
    "momentum_fade": 60,
    "short_squeeze": 60,
}

SETUP_TIME_LIMITS = {
    "momentum_fade": {"stale": 35, "alert": 45, "revalidate": 60, "force_close": 75},
    "gap_and_go":    {"alert": 60, "force_close": 90},
    "vwap_reclaim":  {"alert": 60, "force_close": 90},
    "orb":           {"alert": 45, "force_close": 75},
    "trend_pullback": {"alert": 90, "force_close": 120},
    "news_catalyst": {"alert": 60, "force_close": 90},
    "short_squeeze": {"alert": 30, "force_close": 60},
}

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
    return sessionmaker(bind=engine)()


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


def _seed_pm_environment(db, symbol, profile="moderate"):
    """Seed the minimal DB state needed for run_profile Phase 2 to execute."""
    db.add(Balance(cash=100_000, profile=profile))
    db.add(AgentMemory(
        agent="analyst",
        symbol=symbol,
        key="signal",
        value=json.dumps({
            "setup_type": "gap_and_go",
            "bias": "LONG",
            "strength": "strong",
            "confidence": "high",
            "entry_price": 450.0,
            "stop_price": 445.0,
            "target_price": 460.0,
        }),
        timestamp=datetime.utcnow(),
    ))
    db.commit()


# ═══════════════════════════════════════════════════════════════════════════
# 1a — Late entry: gap_and_go entries outside 60-min window should be rejected
# ═══════════════════════════════════════════════════════════════════════════

@given(
    minutes_since_open=st.integers(min_value=61, max_value=360),
)
@settings(max_examples=30, deadline=None)
def test_property_1a_late_gap_and_go_entry_rejected(minutes_since_open):
    """
    **Validates: Requirements 1.1, 1.5**

    Property 1a: For any gap_and_go entry where minutes_since_open > 60,
    the PM Phase 2 entry loop SHALL reject the entry (action="PASS").

    On unfixed code this FAILS because no entry timing gate exists —
    the LLM's BUY decision proceeds to execute_trade unchecked.
    """
    from agents.portfolio_manager import run_profile

    engine = _make_engine()
    db = _make_session(engine)

    symbol = "TEST"
    _seed_pm_environment(db, symbol, "moderate")
    db.close()

    # Mock the LLM to return a BUY decision for gap_and_go
    mock_llm_response = json.dumps({
        "decisions": [{
            "symbol": symbol,
            "action": "BUY",
            "quantity": 50,
            "price": 450.0,
            "setup_type": "gap_and_go",
            "rationale": "Gap and go setup, strong momentum",
            "stop": 445.0,
            "target": 460.0,
        }],
        "portfolio_notes": "",
    })

    with (
        patch("agents.portfolio_manager.call_llm", return_value=mock_llm_response),
        patch("agents.portfolio_manager.parse_json_response",
              return_value=json.loads(mock_llm_response)),
        patch("agents.portfolio_manager.FinnhubClient") as mock_fh_cls,
        patch("agents.portfolio_manager.get_relevant_cases", return_value=[]),
        patch("agents.portfolio_manager.format_cases_digest_for_pm", return_value=""),
        patch("agents.portfolio_manager.build_pm_strategy_context", return_value=""),
        patch("agents.portfolio_manager.get_win_rate_by_setup", return_value=[]),
        patch("agents.portfolio_manager.execute_trade") as mock_exec_trade,
        patch("agents.portfolio_manager.check_track_record",
              return_value={"verdict": "OK", "sample_size": 0}),
        patch("agents.portfolio_manager.build_entry_contract", return_value={}),
        patch("utils.behavioral_params.get_behavioral_params",
              return_value={"notes": ""}),
        patch("utils.behavioral_params.apply_params_to_decision",
              side_effect=lambda d, *a: d),
        patch("agents.portfolio_manager.datetime") as mock_dt,
    ):
        # Mock FinnhubClient methods
        mock_fh = MagicMock()
        mock_fh.get_quote.return_value = {"price": 450.0}
        mock_fh_cls.return_value = mock_fh

        # Mock execute_trade to succeed
        mock_exec_trade.return_value = (True, "OK")

        # Mock datetime.now to return a time outside the entry window
        from pytz import timezone as _tz
        et_tz = _tz("America/New_York")
        fake_now_et = datetime.now(et_tz).replace(
            hour=9, minute=30, second=0, microsecond=0
        ) + timedelta(minutes=minutes_since_open)
        mock_dt.now.return_value = fake_now_et
        mock_dt.utcnow.return_value = datetime.utcnow()
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        result = run_profile(engine, [symbol], "moderate")

    # EXPECTED BEHAVIOR: The entry should be BLOCKED by the timing gate.
    # The decision should have action="PASS" and execute_trade should NOT be called.
    #
    # On UNFIXED code: execute_trade IS called because there is no timing gate.
    # This assertion FAILS — proving the bug.
    decisions = result.get("decisions", [])
    buy_executed = any(
        d.get("symbol") == symbol
        and d.get("action") == "BUY"
        and d.get("executed") is True
        for d in decisions
    )
    assert not buy_executed, (
        f"Bug confirmed: gap_and_go BUY entry was executed at {minutes_since_open} min "
        f"after market open (limit: 60 min). No entry timing gate exists — "
        f"the entry should have been rejected with action='PASS'."
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1b — Missing reclassification at force_close
# ═══════════════════════════════════════════════════════════════════════════

setup_type_strategy = st.sampled_from(BUG_SETUP_TYPES)
target_price_strategy = st.floats(
    min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False,
)


@given(
    setup_type=setup_type_strategy,
    target_price=target_price_strategy,
    extra_minutes=st.integers(min_value=1, max_value=120),
)
@settings(max_examples=50, deadline=None)
def test_property_1b_missing_reclassification_at_force_close(
    setup_type, target_price, extra_minutes
):
    """
    **Validates: Requirements 1.3**

    Property 1b: For any intraday trade (not momentum_fade) where target_price
    is set and minutes_held > force_close limit, position_timer.run() SHALL
    update trade.setup_type to "swing" in the DB.

    On unfixed code this FAILS because no reclassification logic exists —
    the trade's setup_type remains unchanged.
    """
    from agents.position_timer import run, _alert_status

    force_close_limit = SETUP_TIME_LIMITS[setup_type]["force_close"]
    minutes_held = force_close_limit + extra_minutes

    engine = _make_engine()
    db = _make_session(engine)

    now_utc = datetime.utcnow()
    entry_time = now_utc - timedelta(minutes=minutes_held)

    trade = _seed_trade(db, "TEST", setup_type, target_price, entry_time)
    trade_id = trade.id
    db.close()

    _alert_status.clear()

    # Mock time to be BEFORE the hard wall so we isolate the force_close path
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

        run(engine)

    # Check the DB: setup_type should now be "swing"
    db2 = _make_session(engine)
    updated_trade = db2.query(Trade).filter_by(id=trade_id).first()
    db2.close()

    # EXPECTED BEHAVIOR: setup_type should be reclassified to "swing"
    # On UNFIXED code: setup_type remains the original value — proving the bug.
    assert updated_trade.setup_type == "swing", (
        f"Bug confirmed: {setup_type} trade with target_price={target_price} held "
        f"{minutes_held} min (limit: {force_close_limit}) was NOT reclassified to 'swing'. "
        f"setup_type is still '{updated_trade.setup_type}'. "
        f"No reclassification logic exists in position_timer."
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1c — Missing reclassification at hard wall
# ═══════════════════════════════════════════════════════════════════════════

@given(
    setup_type=setup_type_strategy,
    target_price=target_price_strategy,
    wall_minute=st.integers(min_value=45, max_value=59),
)
@settings(max_examples=50, deadline=None)
def test_property_1c_missing_reclassification_at_hard_wall(
    setup_type, target_price, wall_minute
):
    """
    **Validates: Requirements 1.4**

    Property 1c: For any intraday trade (not momentum_fade) where target_price
    is set and the 3:45 PM hard wall is reached, position_timer.run() SHALL
    update trade.setup_type to "swing" in the DB.

    On unfixed code this FAILS because the hard wall skip just defers to
    price_monitor without reclassifying — setup_type remains unchanged.
    """
    from agents.position_timer import run, _alert_status

    engine = _make_engine()
    db = _make_session(engine)

    now_utc = datetime.utcnow()
    # Entry 30 min ago — well within force_close limits, so only hard wall triggers
    entry_time = now_utc - timedelta(minutes=30)

    trade = _seed_trade(db, "WALL", setup_type, target_price, entry_time)
    trade_id = trade.id
    db.close()

    _alert_status.clear()

    # Mock time to be PAST the hard wall
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

        run(engine)

    # Check the DB: setup_type should now be "swing"
    db2 = _make_session(engine)
    updated_trade = db2.query(Trade).filter_by(id=trade_id).first()
    db2.close()

    # EXPECTED BEHAVIOR: setup_type should be reclassified to "swing"
    # On UNFIXED code: setup_type remains the original value — proving the bug.
    assert updated_trade.setup_type == "swing", (
        f"Bug confirmed: {setup_type} trade with target_price={target_price} at hard wall "
        f"(15:{wall_minute:02d} ET) was NOT reclassified to 'swing'. "
        f"setup_type is still '{updated_trade.setup_type}'. "
        f"The hard wall skip defers to price_monitor without reclassifying."
    )
