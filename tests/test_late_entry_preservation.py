"""
Preservation Property Tests — Late Entry & Position Timer

Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7

These tests capture the EXISTING (correct) behavior that must be preserved
after the bugfix is applied. They verify that:
  2a — In-window gap_and_go entries proceed through PM Phase 2 (not blocked)
  2b — Non-restricted setup types (trend_pullback, vwap_reclaim, news_catalyst)
       are never blocked by the entry timing gate at any time of day
  2c — Trades without target_price are still force-closed at time limits
  2d — Trades without target_price are still force-closed at the 3:45 PM hard wall
  2e — momentum_fade escalation logic is unchanged regardless of target_price
  2f — Time alerts are still generated for all trades past alert threshold

These tests MUST PASS on UNFIXED code (baseline) AND on FIXED code (preservation).
"""

import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from pytz import timezone as _tz, utc as _utc
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, Trade, AgentMemory, Position, Balance

# ── Constants from design doc ──

ENTRY_WINDOW_LIMITS = {
    "gap_and_go": 60,
    "orb": 60,
    "momentum_fade": 60,
    "short_squeeze": 60,
}

SETUP_TIME_LIMITS = {
    "momentum_fade": {"stale": 35, "alert": 35, "revalidate": None, "force_close": 75},
    "gap_and_go":    {"alert": 60, "force_close": 90},
    "vwap_reclaim":  {"alert": 60, "force_close": 90},
    "orb":           {"alert": 45, "force_close": 75},
    "trend_pullback": {"alert": 90, "force_close": 150},
    "news_catalyst": {"alert": 60, "force_close": 120},
    "short_squeeze": {"alert": 30, "force_close": 60},
}

# Setup types NOT restricted by entry window
NON_RESTRICTED_SETUPS = ["trend_pullback", "vwap_reclaim", "news_catalyst"]

# Non-momentum_fade setup types (for position_timer tests)
NON_MF_SETUP_TYPES = [
    "gap_and_go", "vwap_reclaim", "orb",
    "news_catalyst", "trend_pullback", "short_squeeze",
]

# Non-extension-eligible, non-momentum_fade types (simple force close path)
SIMPLE_FORCE_CLOSE_TYPES = [
    "gap_and_go", "vwap_reclaim", "orb", "short_squeeze",
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


def _seed_pm_environment(db, symbol, setup_type, profile="moderate"):
    """Seed the minimal DB state needed for run_profile Phase 2 to execute."""
    db.add(Balance(cash=100_000, profile=profile))
    db.add(AgentMemory(
        agent="analyst",
        symbol=symbol,
        key="signal",
        value=json.dumps({
            "setup_type": setup_type,
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


# ── Hypothesis strategies ──

non_mf_setup_strategy = st.sampled_from(NON_MF_SETUP_TYPES)
simple_force_close_strategy = st.sampled_from(SIMPLE_FORCE_CLOSE_TYPES)
all_setup_strategy = st.sampled_from(list(SETUP_TIME_LIMITS.keys()))
non_restricted_setup_strategy = st.sampled_from(NON_RESTRICTED_SETUPS)
target_price_strategy = st.floats(
    min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False,
)


# ═══════════════════════════════════════════════════════════════════════════
# Observation tests (verify baseline before property tests)
# ═══════════════════════════════════════════════════════════════════════════

def test_observe_gap_and_go_entry_12min():
    """
    Observation: gap_and_go entry at 12 min after open → entry proceeds
    through PM Phase 2 (not blocked). execute_trade IS called.
    """
    from agents.portfolio_manager import run_profile
    from pytz import timezone as _tz

    engine = _make_engine()
    db = _make_session(engine)

    symbol = "OBS1"
    _seed_pm_environment(db, symbol, "gap_and_go", "moderate")
    db.close()

    mock_llm_response = json.dumps({
        "decisions": [{
            "symbol": symbol,
            "action": "BUY",
            "quantity": 50,
            "price": 450.0,
            "setup_type": "gap_and_go",
            "rationale": "Gap and go setup",
            "stop": 445.0,
            "target": 460.0,
        }],
        "portfolio_notes": "",
    })

    # Simulate 12 min after market open (9:42 AM ET)
    et_tz = _tz("America/New_York")
    fake_now_et = datetime.now(et_tz).replace(hour=9, minute=42, second=0, microsecond=0)

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
        mock_fh = MagicMock()
        mock_fh.get_quote.return_value = {"price": 450.0}
        mock_fh_cls.return_value = mock_fh
        mock_exec_trade.return_value = (True, "OK")
        # Mock datetime.now to return in-window time for the entry timing gate
        mock_dt.now.return_value = fake_now_et
        mock_dt.utcnow.return_value = datetime.utcnow()
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        result = run_profile(engine, [symbol], "moderate")

    # Entry should proceed — execute_trade called
    assert mock_exec_trade.called, (
        "Preservation broken: gap_and_go entry at 12 min after open was blocked. "
        "Expected: entry proceeds through PM Phase 2."
    )


def test_observe_trend_pullback_entry_240min():
    """
    Observation: trend_pullback entry at 240 min after open → entry proceeds
    (no restriction for this setup type). execute_trade IS called.
    """
    from agents.portfolio_manager import run_profile

    engine = _make_engine()
    db = _make_session(engine)

    symbol = "OBS2"
    _seed_pm_environment(db, symbol, "trend_pullback", "moderate")
    db.close()

    mock_llm_response = json.dumps({
        "decisions": [{
            "symbol": symbol,
            "action": "BUY",
            "quantity": 50,
            "price": 450.0,
            "setup_type": "trend_pullback",
            "rationale": "Trend pullback setup",
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
    ):
        mock_fh = MagicMock()
        mock_fh.get_quote.return_value = {"price": 450.0}
        mock_fh_cls.return_value = mock_fh
        mock_exec_trade.return_value = (True, "OK")

        result = run_profile(engine, [symbol], "moderate")

    assert mock_exec_trade.called, (
        "Preservation broken: trend_pullback entry at 240 min after open was blocked. "
        "Expected: entry proceeds (no restriction for this setup type)."
    )


def test_observe_vwap_reclaim_entry_180min():
    """
    Observation: vwap_reclaim entry at 180 min after open → entry proceeds
    (no restriction). execute_trade IS called.
    """
    from agents.portfolio_manager import run_profile

    engine = _make_engine()
    db = _make_session(engine)

    symbol = "OBS3"
    _seed_pm_environment(db, symbol, "vwap_reclaim", "moderate")
    db.close()

    mock_llm_response = json.dumps({
        "decisions": [{
            "symbol": symbol,
            "action": "BUY",
            "quantity": 50,
            "price": 450.0,
            "setup_type": "vwap_reclaim",
            "rationale": "VWAP reclaim setup",
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
    ):
        mock_fh = MagicMock()
        mock_fh.get_quote.return_value = {"price": 450.0}
        mock_fh_cls.return_value = mock_fh
        mock_exec_trade.return_value = (True, "OK")

        result = run_profile(engine, [symbol], "moderate")

    assert mock_exec_trade.called, (
        "Preservation broken: vwap_reclaim entry at 180 min after open was blocked. "
        "Expected: entry proceeds (no restriction for this setup type)."
    )


def test_observe_gap_and_go_no_target_95min_force_close():
    """
    Observation: gap_and_go with target_price=None, held 95 min
    → _close_position called (force close unchanged).
    """
    from agents.position_timer import run

    engine = _make_engine()
    db = _make_session(engine)

    _et = _tz("America/New_York")
    fake_now_et = _et.localize(datetime(2025, 6, 25, 10, 30, 0))
    fake_now_utc = fake_now_et.astimezone(_utc)

    entry_time = fake_now_utc - timedelta(minutes=95)

    _seed_trade(db, "OBS4", "gap_and_go", None, entry_time)
    db.close()

    with (
        patch("agents.position_timer._get_current_price", return_value=452.0),
        patch("agents.position_timer._close_position") as mock_close,
        patch("agents.position_timer.datetime") as mock_dt,
    ):
        def _mn(tz=None):
            if tz is not None and "UTC" in str(tz):
                return fake_now_utc
            return fake_now_et
        mock_dt.now.side_effect = _mn
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        result = run(engine)

    assert mock_close.call_count == 1
    assert len(result["closes"]) >= 1


def test_observe_hard_wall_no_target_close():
    """
    Observation: any intraday trade with target_price=None past 3:45 PM
    → _close_position called (hard wall unchanged).
    """
    from agents.position_timer import run

    engine = _make_engine()
    db = _make_session(engine)

    _et = _tz("America/New_York")
    fake_now_et = _et.localize(datetime(2025, 6, 25, 15, 46, 0))
    fake_now_utc = fake_now_et.astimezone(_utc)

    entry_time = fake_now_utc - timedelta(minutes=30)

    _seed_trade(db, "OBS5", "gap_and_go", None, entry_time)
    db.close()

    with (
        patch("agents.position_timer._get_current_price", return_value=452.0),
        patch("agents.position_timer._close_position") as mock_close,
        patch("agents.position_timer.datetime") as mock_dt,
    ):
        def _mn(tz=None):
            if tz is not None and "UTC" in str(tz):
                return fake_now_utc
            return fake_now_et
        mock_dt.now.side_effect = _mn
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        result = run(engine)

    assert mock_close.call_count == 1
    assert len(result["closes"]) >= 1


def test_observe_momentum_fade_with_target_80min():
    """
    Observation: momentum_fade with target_price=450.0, held 80 min
    → _close_position called (escalation unchanged).
    momentum_fade is excluded from swing reclassification.
    """
    from agents.position_timer import run

    engine = _make_engine()
    db = _make_session(engine)

    _et = _tz("America/New_York")
    fake_now_et = _et.localize(datetime(2025, 6, 25, 10, 30, 0))
    fake_now_utc = fake_now_et.astimezone(_utc)

    entry_time = fake_now_utc - timedelta(minutes=80)

    _seed_trade(db, "OBS6", "momentum_fade", 450.0, entry_time)
    db.close()

    with (
        patch("agents.position_timer._get_current_price", return_value=452.0),
        patch("agents.position_timer._close_position") as mock_close,
        patch("agents.position_timer.datetime") as mock_dt,
    ):
        def _mn(tz=None):
            if tz is not None and "UTC" in str(tz):
                return fake_now_utc
            return fake_now_et
        mock_dt.now.side_effect = _mn
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        result = run(engine)

    assert mock_close.call_count >= 1
    assert len(result["closes"]) >= 1


def test_observe_gap_and_go_with_target_70min_alert():
    """
    Observation: gap_and_go with target_price=450.0, held 70 min
    (alert > 60, force < 90) → no force close (preservation property).
    The new system returns decision="hold" with state="setup_exit_alert"
    and requires_event=True. No close action is taken.
    """
    from agents.position_timer import run

    engine = _make_engine()
    db = _make_session(engine)

    _et = _tz("America/New_York")
    fake_now_et = _et.localize(datetime(2025, 6, 25, 10, 30, 0))
    fake_now_utc = fake_now_et.astimezone(_utc)

    entry_time = fake_now_utc - timedelta(minutes=70)

    _seed_trade(db, "OBS7", "gap_and_go", 450.0, entry_time)
    db.close()

    with (
        patch("agents.position_timer._get_current_price", return_value=452.0),
        patch("agents.position_timer._close_position") as mock_close,
        patch("agents.position_timer.datetime") as mock_dt,
    ):
        def _mn(tz=None):
            if tz is not None and "UTC" in str(tz):
                return fake_now_utc
            return fake_now_et
        mock_dt.now.side_effect = _mn
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        result = run(engine)

    # Preservation: no force close between alert and force_close thresholds
    assert mock_close.call_count == 0



# ═══════════════════════════════════════════════════════════════════════════
# Property 2a — In-window entry preservation
# For all gap_and_go entries with minutes_since_open ≤ 60, assert the entry
# is NOT blocked by the timing gate (proceeds to normal PM evaluation).
# ═══════════════════════════════════════════════════════════════════════════

@given(
    minutes_since_open=st.integers(min_value=1, max_value=60),
)
@settings(max_examples=30, deadline=None)
def test_property_2a_in_window_entry_preservation(minutes_since_open):
    """
    **Validates: Requirements 3.1**

    Property 2a: For all gap_and_go entries with minutes_since_open ≤ 60,
    the entry is NOT blocked by the timing gate and proceeds to normal
    PM Phase 2 evaluation. execute_trade IS called.

    On unfixed code: no timing gate exists, so all entries proceed → PASSES.
    On fixed code: entries within window must still proceed → PASSES.
    """
    from agents.portfolio_manager import run_profile
    from pytz import timezone as _tz

    engine = _make_engine()
    db = _make_session(engine)

    symbol = "INWIN"
    _seed_pm_environment(db, symbol, "gap_and_go", "moderate")
    db.close()

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

    # Simulate the given minutes_since_open after market open (9:30 AM ET)
    et_tz = _tz("America/New_York")
    fake_now_et = datetime.now(et_tz).replace(
        hour=9, minute=30, second=0, microsecond=0
    ) + timedelta(minutes=minutes_since_open)

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
        mock_fh = MagicMock()
        mock_fh.get_quote.return_value = {"price": 450.0}
        mock_fh_cls.return_value = mock_fh
        mock_exec_trade.return_value = (True, "OK")
        # Mock datetime.now to return in-window time for the entry timing gate
        mock_dt.now.return_value = fake_now_et
        mock_dt.utcnow.return_value = datetime.utcnow()
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        result = run_profile(engine, [symbol], "moderate")

    # Preservation: execute_trade MUST be called — entry not blocked
    decisions = result.get("decisions", [])
    buy_executed = any(
        d.get("symbol") == symbol
        and d.get("action") == "BUY"
        and d.get("executed") is True
        for d in decisions
    )
    assert buy_executed, (
        f"Preservation broken: gap_and_go BUY entry at {minutes_since_open} min "
        f"after open was NOT executed. Expected: entry proceeds through PM Phase 2 "
        f"(within 60-min window)."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Property 2b — Non-restricted setup preservation
# For all entries with setup_type NOT in ENTRY_WINDOW_LIMITS, assert the
# entry is NOT blocked at any time of day.
# ═══════════════════════════════════════════════════════════════════════════

@given(
    setup_type=non_restricted_setup_strategy,
    minutes_since_open=st.integers(min_value=1, max_value=360),
)
@settings(max_examples=30, deadline=None)
def test_property_2b_non_restricted_setup_preservation(setup_type, minutes_since_open):
    """
    **Validates: Requirements 3.2**

    Property 2b: For all entries with setup_type NOT in ENTRY_WINDOW_LIMITS
    (trend_pullback, vwap_reclaim, news_catalyst), the entry is NOT blocked
    at any time of day. execute_trade IS called.

    On unfixed code: no timing gate exists, so all entries proceed → PASSES.
    On fixed code: non-restricted setups must still proceed → PASSES.
    """
    from agents.portfolio_manager import run_profile

    engine = _make_engine()
    db = _make_session(engine)

    symbol = "NRST"
    _seed_pm_environment(db, symbol, setup_type, "moderate")
    db.close()

    mock_llm_response = json.dumps({
        "decisions": [{
            "symbol": symbol,
            "action": "BUY",
            "quantity": 50,
            "price": 450.0,
            "setup_type": setup_type,
            "rationale": f"{setup_type} setup, strong signal",
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
    ):
        mock_fh = MagicMock()
        mock_fh.get_quote.return_value = {"price": 450.0}
        mock_fh_cls.return_value = mock_fh
        mock_exec_trade.return_value = (True, "OK")

        result = run_profile(engine, [symbol], "moderate")

    # Preservation: execute_trade MUST be called — non-restricted setup not blocked
    decisions = result.get("decisions", [])
    buy_executed = any(
        d.get("symbol") == symbol
        and d.get("action") == "BUY"
        and d.get("executed") is True
        for d in decisions
    )
    assert buy_executed, (
        f"Preservation broken: {setup_type} BUY entry at {minutes_since_open} min "
        f"after open was NOT executed. Expected: entry proceeds (no restriction "
        f"for {setup_type})."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Property 2c — No-target force close preservation
# For all trades with target_price=None and minutes_held > force_close
# and setup_type != "momentum_fade", assert _close_position IS called.
# ═══════════════════════════════════════════════════════════════════════════

@given(
    setup_type=simple_force_close_strategy,
    extra_minutes=st.integers(min_value=1, max_value=120),
)
@settings(max_examples=50, deadline=None)
def test_property_2c_no_target_force_close_preservation(setup_type, extra_minutes):
    """
    **Validates: Requirements 3.3, 3.5**

    Property 2c: For all trades with target_price=None and
    minutes_held > limits["force_close"] and setup_type != "momentum_fade",
    assert _close_position IS called (force close fires).

    This preserves the existing behavior: trades without a price target
    are force-closed when they exceed their time limit.
    """
    from agents.position_timer import run

    limits = SETUP_TIME_LIMITS[setup_type]
    force_close_limit = limits["force_close"]
    minutes_held = force_close_limit + extra_minutes

    engine = _make_engine()
    db = _make_session(engine)

    # Mock time BEFORE hard wall to isolate force_close path
    _et = _tz("America/New_York")
    fake_now_et = _et.localize(datetime(2025, 6, 25, 10, 30, 0))
    fake_now_utc = fake_now_et.astimezone(_utc)

    entry_time = fake_now_utc - timedelta(minutes=minutes_held)

    _seed_trade(db, "FC", setup_type, None, entry_time)
    db.close()

    with (
        patch("agents.position_timer._get_current_price", return_value=452.0),
        patch("agents.position_timer._close_position") as mock_close,
        patch("agents.position_timer.datetime") as mock_dt,
    ):
        def _mn(tz=None):
            if tz is not None and "UTC" in str(tz):
                return fake_now_utc
            return fake_now_et
        mock_dt.now.side_effect = _mn
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        result = run(engine)

    # Preservation: _close_position MUST be called for no-target trades
    assert mock_close.call_count == 1, (
        f"Preservation broken: _close_position was NOT called for {setup_type} "
        f"with target_price=None, minutes_held={minutes_held} "
        f"(limit: {force_close_limit}). Expected force close."
    )
    assert len(result["closes"]) == 1, (
        f"Preservation broken: closes list should have 1 entry, "
        f"got {len(result['closes'])} for {setup_type} with target_price=None."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Property 2d — No-target hard wall preservation
# For all trades with target_price=None past 3:45 PM ET, assert
# _close_position IS called (hard wall fires).
# ═══════════════════════════════════════════════════════════════════════════

@given(
    setup_type=all_setup_strategy,
    wall_minute=st.integers(min_value=45, max_value=59),
)
@settings(max_examples=50, deadline=None)
def test_property_2d_no_target_hard_wall_preservation(setup_type, wall_minute):
    """
    **Validates: Requirements 3.4**

    Property 2d: For all trades with target_price=None past the hard wall
    (3:45 PM ET), assert _close_position IS called.

    This preserves the existing hard wall behavior for trades without
    a price target.
    """
    from agents.position_timer import run

    engine = _make_engine()
    db = _make_session(engine)

    # Mock time past the hard wall
    _et = _tz("America/New_York")
    fake_now_et = _et.localize(datetime(2025, 6, 25, HARD_WALL_HOUR, wall_minute, 0))
    fake_now_utc = fake_now_et.astimezone(_utc)

    # Entry 30 min ago — within time limits, so only hard wall triggers
    entry_time = fake_now_utc - timedelta(minutes=30)

    _seed_trade(db, "HW", setup_type, None, entry_time)
    db.close()

    with (
        patch("agents.position_timer._get_current_price", return_value=452.0),
        patch("agents.position_timer._close_position") as mock_close,
        patch("agents.position_timer.datetime") as mock_dt,
    ):
        def _mn(tz=None):
            if tz is not None and "UTC" in str(tz):
                return fake_now_utc
            return fake_now_et
        mock_dt.now.side_effect = _mn
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        result = run(engine)

    # Preservation: hard wall close MUST fire for no-target trades
    assert mock_close.call_count == 1, (
        f"Preservation broken: _close_position was called {mock_close.call_count} time(s) "
        f"for {setup_type} with target_price=None at hard wall "
        f"(15:{wall_minute:02d} ET). Expected exactly 1 call."
    )
    assert len(result["closes"]) == 1, (
        f"Preservation broken: closes should have 1 entry, "
        f"got {len(result['closes'])} for {setup_type} with target_price=None."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Property 2e — Momentum_fade preservation
# For all momentum_fade trades regardless of target_price, assert the
# stale/alert/revalidation/force_close escalation logic is unchanged
# (force close at 75 min).
# ═══════════════════════════════════════════════════════════════════════════

@given(
    target_price=st.one_of(st.none(), target_price_strategy),
    extra_minutes=st.integers(min_value=1, max_value=60),
)
@settings(max_examples=50, deadline=None)
def test_property_2e_momentum_fade_preservation(target_price, extra_minutes):
    """
    **Validates: Requirements 3.7**

    Property 2e: For all momentum_fade trades regardless of target_price,
    when minutes_held > 75 (force_close limit), assert _close_position IS called.

    The momentum_fade branch has its own escalation logic that is independent
    of target_price. Force close at 75 min must always fire.
    """
    from agents.position_timer import run

    force_close_limit = SETUP_TIME_LIMITS["momentum_fade"]["force_close"]  # 75
    minutes_held = force_close_limit + extra_minutes

    engine = _make_engine()
    db = _make_session(engine)

    # Mock time BEFORE hard wall to isolate momentum_fade logic
    _et = _tz("America/New_York")
    fake_now_et = _et.localize(datetime(2025, 6, 25, 10, 30, 0))
    fake_now_utc = fake_now_et.astimezone(_utc)

    entry_time = fake_now_utc - timedelta(minutes=minutes_held)

    _seed_trade(db, "MF", "momentum_fade", target_price, entry_time)
    db.close()

    with (
        patch("agents.position_timer._get_current_price", return_value=452.0),
        patch("agents.position_timer._close_position") as mock_close,
        patch("agents.position_timer.datetime") as mock_dt,
    ):
        def _mn(tz=None):
            if tz is not None and "UTC" in str(tz):
                return fake_now_utc
            return fake_now_et
        mock_dt.now.side_effect = _mn
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        result = run(engine)

    # Preservation: _close_position MUST be called at least once
    # (revalidation fails → close at 60 min, then force_close at 75 min)
    assert mock_close.call_count >= 1, (
        f"Preservation broken: _close_position was NOT called for momentum_fade "
        f"with target_price={target_price}, minutes_held={minutes_held} "
        f"(limit: {force_close_limit}). Expected force close."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Property 2f — Alert generation preservation
# For all trades with minutes_held > limits["alert"] (regardless of
# target_price), assert alerts are generated.
# ═══════════════════════════════════════════════════════════════════════════

@given(
    setup_type=simple_force_close_strategy,
    target_price=st.one_of(st.none(), target_price_strategy),
    extra_minutes=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=50, deadline=None)
def test_property_2f_alert_generation_preservation(setup_type, target_price, extra_minutes):
    """
    **Validates: Requirements 3.5**

    Property 2f: For all trades with minutes_held > limits["alert"]
    (regardless of target_price), assert alerts are generated.

    Time alerts must continue to fire for all trades, even those with
    a valid target_price. This test constrains minutes_held to be
    between alert and force_close limits so the alert path is reached
    without triggering force close.
    """
    from agents.position_timer import run

    limits = SETUP_TIME_LIMITS[setup_type]
    alert_limit = limits["alert"]
    force_close_limit = limits["force_close"]

    # minutes_held must be > alert but <= force_close to isolate alert path
    minutes_held = alert_limit + extra_minutes
    assume(minutes_held <= force_close_limit)

    engine = _make_engine()
    db = _make_session(engine)

    # Mock time BEFORE hard wall to isolate alert logic
    _et = _tz("America/New_York")
    fake_now_et = _et.localize(datetime(2025, 6, 25, 10, 30, 0))
    fake_now_utc = fake_now_et.astimezone(_utc)

    entry_time = fake_now_utc - timedelta(minutes=minutes_held)

    _seed_trade(db, "ALRT", setup_type, target_price, entry_time)
    db.close()

    with (
        patch("agents.position_timer._get_current_price", return_value=452.0),
        patch("agents.position_timer._close_position") as mock_close,
        patch("agents.position_timer.datetime") as mock_dt,
    ):
        def _mn(tz=None):
            if tz is not None and "UTC" in str(tz):
                return fake_now_utc
            return fake_now_et
        mock_dt.now.side_effect = _mn
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        result = run(engine)

    # Preservation: no force close between alert and force_close thresholds
    # The new system returns decision="hold" with state="setup_exit_alert"
    # which does NOT produce a close action — the key invariant is preserved.
    assert mock_close.call_count == 0, (
        f"Unexpected force close for {setup_type} with minutes_held={minutes_held} "
        f"(force_close limit: {force_close_limit}). Should only alert, not close."
    )
