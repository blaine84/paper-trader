"""
Preservation Property Tests — PM Signal Filter & Price Repair

Validates: Requirements 3.1, 3.2, 3.3, 3.5

These tests capture the EXISTING (correct) behavior that must be preserved
after the bugfix is applied. They verify that:

  Preservation A — Actionable signals (LONG/SHORT with strength meeting
      profile threshold) remain in entry_signals (clause 3.1, 3.5)
  Preservation B — Trades with entry price within 5% of live quote use
      LLM-provided stop/target as-is (clause 3.2)
  Preservation C — CLOSE actions bypass price validation entirely (clause 3.3)

These tests MUST PASS on UNFIXED code (baseline) AND on FIXED code (preservation).
"""

import json
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, Balance, Trade, Position, AgentMemory
from agents.portfolio_manager import (
    execute_trade,
    STRENGTH_ORDER,
    _meets_threshold,
)


# ── Helpers ──

def _make_engine():
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    return engine


def _make_session(engine):
    return sessionmaker(bind=engine)()


def _seed_balance(db, profile_id: str, cash: float = 100_000.0):
    db.add(Balance(profile=profile_id, cash=cash))
    db.commit()


def _seed_analyst_signal(db, symbol: str, signal: dict):
    db.add(AgentMemory(
        agent="analyst",
        symbol=symbol,
        key="signal",
        value=json.dumps(signal),
    ))
    db.commit()


def _build_entry_signals(signals: dict, held_symbols: set, profile: dict) -> dict:
    """
    Replicate the entry_signals comprehension from run_profile() line ~2122.

    Fixed code filters out:
        - HOLD direction signals
        - Signals below profile's min_signal_strength threshold
    """
    entry_signals = {
        sym: sig for sym, sig in signals.items()
        if sym not in held_symbols
        and sig.get("signal", "").upper() != "HOLD"
        and _meets_threshold(sig.get("strength", "weak"), profile["min_signal_strength"])
    }
    return entry_signals


# ═══════════════════════════════════════════════════════════════════════════
# Observation tests (verify baseline before property tests)
# ═══════════════════════════════════════════════════════════════════════════


def test_observe_long_strong_signal_included():
    """
    Observation: LONG signal with strength="strong" for moderate profile
    (min_signal_strength="moderate") → signal IS included in entry_signals.
    """
    signals = {
        "AAPL": {
            "signal": "LONG",
            "strength": "strong",
            "confidence": "high",
            "setup_type": "gap_and_go",
        }
    }
    held_symbols = set()
    profile = {"min_signal_strength": "moderate"}

    entry_signals = _build_entry_signals(signals, held_symbols, profile)

    assert "AAPL" in entry_signals, (
        "Observation failed: LONG/strong signal should be in entry_signals "
        "for moderate profile."
    )


def test_observe_short_moderate_signal_included():
    """
    Observation: SHORT signal with strength="moderate" for moderate profile
    (min_signal_strength="moderate") → signal IS included in entry_signals.
    """
    signals = {
        "SPY": {
            "signal": "SHORT",
            "strength": "moderate",
            "confidence": "medium",
            "setup_type": "trend_pullback",
        }
    }
    held_symbols = set()
    profile = {"min_signal_strength": "moderate"}

    entry_signals = _build_entry_signals(signals, held_symbols, profile)

    assert "SPY" in entry_signals, (
        "Observation failed: SHORT/moderate signal should be in entry_signals "
        "for moderate profile."
    )


def test_observe_weak_signal_included_for_aggressive():
    """
    Observation: LONG signal with strength="weak" for aggressive profile
    (min_signal_strength="weak") → signal IS included in entry_signals.
    """
    signals = {
        "TSLA": {
            "signal": "LONG",
            "strength": "weak",
            "confidence": "low",
            "setup_type": "momentum_fade",
        }
    }
    held_symbols = set()
    profile = {"min_signal_strength": "weak"}

    entry_signals = _build_entry_signals(signals, held_symbols, profile)

    assert "TSLA" in entry_signals, (
        "Observation failed: LONG/weak signal should be in entry_signals "
        "for aggressive profile (min_signal_strength='weak')."
    )


def test_observe_small_deviation_passthrough():
    """
    Observation: BUY at $450, live price=$460 (2.2% deviation ≤5%)
    → stop and target remain at original LLM-provided values.
    """
    engine = _make_engine()
    db = _make_session(engine)
    profile_id = "moderate"
    _seed_balance(db, profile_id)

    _seed_analyst_signal(db, "SPY", {
        "signal": "LONG",
        "strength": "strong",
        "confidence": "high",
        "setup_type": "gap_and_go",
        "market_regime": "risk_on",
        "bias": "LONG",
        "indicators": {
            "above_vwap": True,
            "ema_trend": "bullish",
            "rsi": 55.0,
            "macd_bias": "bullish",
            "bb_position": "upper",
        },
    })

    original_stop = 445.0
    original_target = 460.0

    decision = {
        "symbol": "SPY",
        "action": "BUY",
        "quantity": 50,
        "price": 450.0,
        "stop": original_stop,
        "target": original_target,
        "rationale": "preservation observation test",
        "setup_type": "gap_and_go",
        "market_regime": "risk_on",
    }

    # Live price within 5% of entry: 460/450 = 2.2% deviation
    mock_quote = {"price": 460.0, "symbol": "SPY"}
    with patch("agents.portfolio_manager.FinnhubClient") as MockFH:
        mock_instance = MagicMock()
        mock_instance.get_quote.return_value = mock_quote
        mock_instance.get_candles.return_value = []
        MockFH.return_value = mock_instance

        ok, msg = execute_trade(db, decision, profile_id)

    final_stop = decision.get("stop")
    final_target = decision.get("target")

    assert final_stop == original_stop, (
        f"Observation failed: stop should remain at {original_stop}, "
        f"got {final_stop} (small deviation passthrough)."
    )
    assert final_target == original_target, (
        f"Observation failed: target should remain at {original_target}, "
        f"got {final_target} (small deviation passthrough)."
    )

    db.close()


def test_observe_close_action_bypass():
    """
    Observation: CLOSE action with arbitrary prices → executes without
    price validation rejection.
    """
    engine = _make_engine()
    db = _make_session(engine)
    profile_id = "moderate"
    _seed_balance(db, profile_id)

    # Create an open position to close
    pos = Position(
        symbol="AAPL", quantity=50, avg_cost=150.0,
        profile=profile_id, side="long",
    )
    db.add(pos)
    trade = Trade(
        symbol="AAPL", direction="LONG", quantity=50,
        entry_price=150.0, status="open", profile=profile_id,
        stop_price=145.0, target_price=160.0,
    )
    db.add(trade)
    db.commit()

    decision = {
        "symbol": "AAPL",
        "action": "CLOSE",
        "quantity": 50,
        "price": 155.0,
        "rationale": "preservation observation test — close bypass",
    }

    # Mock FinnhubClient — live price is wildly different from decision price
    mock_quote = {"price": 300.0, "symbol": "AAPL"}
    with patch("agents.portfolio_manager.FinnhubClient") as MockFH:
        mock_instance = MagicMock()
        mock_instance.get_quote.return_value = mock_quote
        MockFH.return_value = mock_instance

        ok, msg = execute_trade(db, decision, profile_id)

    assert ok is True, (
        f"Observation failed: CLOSE action should succeed regardless of price "
        f"deviation. Got ok={ok}, msg='{msg}'."
    )

    db.close()


# ═══════════════════════════════════════════════════════════════════════════
# Preservation A — Actionable Signals Still Included
# For all (direction ∈ {LONG, SHORT}, strength, profile_threshold) where
# STRENGTH_ORDER[strength] >= STRENGTH_ORDER[threshold], signal remains
# in entry_signals.
# ═══════════════════════════════════════════════════════════════════════════


@given(
    direction=st.sampled_from(["LONG", "SHORT"]),
    strength=st.sampled_from(["weak", "moderate", "strong"]),
    profile_threshold=st.sampled_from(["weak", "moderate", "strong"]),
)
@settings(max_examples=50, deadline=None)
def test_preservation_a_actionable_signals_included(direction, strength, profile_threshold):
    """
    **Validates: Requirements 3.1, 3.5**

    Preservation A: For all LONG/SHORT signals with strength meeting or
    exceeding the profile's min_signal_strength threshold, the signal
    remains in entry_signals.

    On unfixed code: the comprehension includes ALL signals for symbols
    without open positions, so actionable signals are always included → PASSES.
    On fixed code: actionable signals must still be included → PASSES.
    """
    # Only test cases where strength meets the threshold
    assume(STRENGTH_ORDER[strength] >= STRENGTH_ORDER[profile_threshold])

    symbol = "TEST"
    signals = {
        symbol: {
            "signal": direction,
            "strength": strength,
            "confidence": "medium",
            "setup_type": "gap_and_go",
        }
    }
    held_symbols = set()
    profile = {"min_signal_strength": profile_threshold}

    entry_signals = _build_entry_signals(signals, held_symbols, profile)

    assert symbol in entry_signals, (
        f"Preservation broken: {direction} signal with strength={strength} "
        f"should be in entry_signals for profile with "
        f"min_signal_strength={profile_threshold}. "
        f"STRENGTH_ORDER: {strength}={STRENGTH_ORDER[strength]}, "
        f"{profile_threshold}={STRENGTH_ORDER[profile_threshold]}."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Preservation B — Small Deviation Passthrough
# For all (entry_price, live_price) where abs(entry - live) / live <= 0.05,
# stop and target are untouched after execute_trade().
# ═══════════════════════════════════════════════════════════════════════════


@given(
    base_price=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    deviation_pct=st.floats(min_value=-0.049, max_value=0.049, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=None)
def test_preservation_b_small_deviation_passthrough(base_price, deviation_pct):
    """
    **Validates: Requirements 3.2**

    Preservation B: For all trades with entry price within 5% of live quote,
    stop and target are untouched after execute_trade().

    On unfixed code: the price sanity block only triggers at >5% deviation,
    so stop/target are never modified for small deviations → PASSES.
    On fixed code: small deviations must still pass through → PASSES.
    """
    # Compute entry and live prices such that deviation ≤ 5%
    live_price = round(base_price, 2)
    entry_price = round(live_price * (1 + deviation_pct), 2)

    # Verify the deviation is actually ≤ 5%
    if live_price <= 0 or entry_price <= 0:
        assume(False)
    actual_deviation = abs(entry_price - live_price) / live_price
    assume(actual_deviation <= 0.05)

    engine = _make_engine()
    db = _make_session(engine)
    profile_id = "moderate"
    _seed_balance(db, profile_id)

    _seed_analyst_signal(db, "TDEV", {
        "signal": "LONG",
        "strength": "strong",
        "confidence": "high",
        "setup_type": "gap_and_go",
        "market_regime": "risk_on",
        "bias": "LONG",
        "indicators": {
            "above_vwap": True,
            "ema_trend": "bullish",
            "rsi": 55.0,
            "macd_bias": "bullish",
            "bb_position": "upper",
        },
    })

    # Set stop below entry and target above entry for a valid LONG geometry
    original_stop = round(entry_price * 0.98, 2)
    original_target = round(entry_price * 1.04, 2)

    decision = {
        "symbol": "TDEV",
        "action": "BUY",
        "quantity": 10,
        "price": entry_price,
        "stop": original_stop,
        "target": original_target,
        "rationale": "preservation test — small deviation passthrough",
        "setup_type": "gap_and_go",
        "market_regime": "risk_on",
    }

    mock_quote = {"price": live_price, "symbol": "TDEV"}
    with patch("agents.portfolio_manager.FinnhubClient") as MockFH:
        mock_instance = MagicMock()
        mock_instance.get_quote.return_value = mock_quote
        mock_instance.get_candles.return_value = []
        MockFH.return_value = mock_instance

        ok, msg = execute_trade(db, decision, profile_id)

    # Stop and target should remain at their original values
    final_stop = decision.get("stop")
    final_target = decision.get("target")

    assert final_stop == original_stop, (
        f"Preservation broken: stop was modified from {original_stop} to "
        f"{final_stop} for small deviation ({actual_deviation:.3%}). "
        f"entry_price={entry_price}, live_price={live_price}."
    )
    assert final_target == original_target, (
        f"Preservation broken: target was modified from {original_target} to "
        f"{final_target} for small deviation ({actual_deviation:.3%}). "
        f"entry_price={entry_price}, live_price={live_price}."
    )

    db.close()


# ═══════════════════════════════════════════════════════════════════════════
# Preservation C — CLOSE Action Bypass
# For all CLOSE decisions with arbitrary prices, execute_trade() does not
# modify or reject based on price.
# ═══════════════════════════════════════════════════════════════════════════


@given(
    decision_price=st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    live_price=st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=None)
def test_preservation_c_close_action_bypass(decision_price, live_price):
    """
    **Validates: Requirements 3.3**

    Preservation C: For all CLOSE decisions with arbitrary prices,
    execute_trade() does not modify or reject based on price deviation.
    CLOSE actions bypass price validation entirely.

    On unfixed code: CLOSE actions go through the price sanity block
    (entry price may be replaced with live price) but then proceed to
    the CLOSE branch which does not use stop/target → PASSES.
    On fixed code: CLOSE actions must still bypass → PASSES.
    """
    engine = _make_engine()
    db = _make_session(engine)
    profile_id = "moderate"
    _seed_balance(db, profile_id)

    # Create an open position to close
    entry_cost = round(decision_price * 0.9, 2) or 1.0
    pos = Position(
        symbol="CLSE", quantity=50, avg_cost=entry_cost,
        profile=profile_id, side="long",
    )
    db.add(pos)
    trade = Trade(
        symbol="CLSE", direction="LONG", quantity=50,
        entry_price=entry_cost, status="open", profile=profile_id,
        stop_price=round(entry_cost * 0.95, 2),
        target_price=round(entry_cost * 1.10, 2),
    )
    db.add(trade)
    db.commit()

    decision = {
        "symbol": "CLSE",
        "action": "CLOSE",
        "quantity": 50,
        "price": round(decision_price, 2),
        "rationale": "preservation test — close bypass",
    }

    mock_quote = {"price": round(live_price, 2), "symbol": "CLSE"}
    with patch("agents.portfolio_manager.FinnhubClient") as MockFH:
        mock_instance = MagicMock()
        mock_instance.get_quote.return_value = mock_quote
        MockFH.return_value = mock_instance

        ok, msg = execute_trade(db, decision, profile_id)

    # CLOSE should succeed regardless of price deviation
    assert ok is True, (
        f"Preservation broken: CLOSE action was rejected with msg='{msg}'. "
        f"decision_price={decision_price:.2f}, live_price={live_price:.2f}. "
        f"CLOSE actions should bypass price validation entirely."
    )

    db.close()
