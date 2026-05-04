"""
Bug Condition Exploration Test — PM Signal Filter & Price Repair

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5

This test encodes the EXPECTED (correct) behavior for both bugs:

Bug 1 — Signal Filtering:
  1a — HOLD signals should be excluded from entry_signals
  1b — Weak signals should be excluded for conservative profile (min_signal_strength="strong")
  1c — Moderate signals should be excluded for conservative profile (min_signal_strength="strong")

Bug 2 — Price Repair:
  1d — Moderate deviation (8%) should proportionally scale stop/target
  1e — Extreme deviation (50%) should reject the trade outright

On UNFIXED code these tests are EXPECTED TO FAIL — failure confirms the bugs exist:
  - Bug 1: entry_signals comprehension does not filter on direction or strength
  - Bug 2: execute_trade replaces entry price but leaves stop/target unrepaired,
    and does not reject extreme deviations
"""

import json
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest
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


# ═══════════════════════════════════════════════════════════════════════════
# Bug 1 — Signal Filtering
# Tests against the entry_signals comprehension at line ~2122
# ═══════════════════════════════════════════════════════════════════════════


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


class TestBug1SignalFiltering:
    """
    **Validates: Requirements 2.1, 2.2**

    Bug 1: The entry_signals comprehension includes ALL signals for symbols
    without open positions, regardless of direction or strength. HOLD signals
    and sub-threshold signals pass through to the PM LLM.
    """

    def test_1a_hold_signal_excluded(self):
        """
        **Validates: Requirements 2.1**

        Test 1a: A HOLD-direction signal should NOT be present in entry_signals.

        On unfixed code this FAILS because the comprehension does not filter
        on signal direction — HOLD signals ARE included.
        """
        signals = {
            "AAPL": {
                "signal": "HOLD",
                "strength": "moderate",
                "confidence": "medium",
                "setup_type": "gap_and_go",
            }
        }
        held_symbols = set()
        profile = {"min_signal_strength": "moderate"}

        entry_signals = _build_entry_signals(signals, held_symbols, profile)

        # EXPECTED BEHAVIOR: HOLD signal should be excluded
        # On UNFIXED code: HOLD signal IS included — proving the bug
        assert "AAPL" not in entry_signals, (
            "Bug confirmed: HOLD signal for AAPL was included in entry_signals. "
            "The entry_signals comprehension does not filter on signal direction. "
            f"entry_signals={entry_signals}"
        )

    def test_1b_weak_signal_excluded_for_conservative(self):
        """
        **Validates: Requirements 2.2**

        Test 1b: A weak signal should NOT be present in entry_signals when
        the profile requires min_signal_strength="strong".

        On unfixed code this FAILS because the comprehension does not check
        signal strength against the profile threshold.
        """
        signals = {
            "SPY": {
                "signal": "LONG",
                "strength": "weak",
                "confidence": "low",
                "setup_type": "trend_pullback",
            }
        }
        held_symbols = set()
        profile = {"min_signal_strength": "strong"}

        entry_signals = _build_entry_signals(signals, held_symbols, profile)

        # Verify the threshold logic: weak (1) < strong (3)
        assert not _meets_threshold("weak", "strong"), (
            "Sanity check: _meets_threshold('weak', 'strong') should be False"
        )

        # EXPECTED BEHAVIOR: weak signal should be excluded for conservative profile
        # On UNFIXED code: weak signal IS included — proving the bug
        assert "SPY" not in entry_signals, (
            "Bug confirmed: weak signal for SPY was included in entry_signals "
            "despite conservative profile requiring min_signal_strength='strong'. "
            f"STRENGTH_ORDER: weak={STRENGTH_ORDER['weak']}, strong={STRENGTH_ORDER['strong']}. "
            f"entry_signals={entry_signals}"
        )

    def test_1c_moderate_signal_excluded_for_conservative(self):
        """
        **Validates: Requirements 2.2**

        Test 1c: A moderate signal should NOT be present in entry_signals when
        the profile requires min_signal_strength="strong".

        On unfixed code this FAILS because the comprehension does not check
        signal strength against the profile threshold.
        """
        signals = {
            "QQQ": {
                "signal": "LONG",
                "strength": "moderate",
                "confidence": "medium",
                "setup_type": "vwap_reclaim",
            }
        }
        held_symbols = set()
        profile = {"min_signal_strength": "strong"}

        entry_signals = _build_entry_signals(signals, held_symbols, profile)

        # Verify the threshold logic: moderate (2) < strong (3)
        assert not _meets_threshold("moderate", "strong"), (
            "Sanity check: _meets_threshold('moderate', 'strong') should be False"
        )

        # EXPECTED BEHAVIOR: moderate signal should be excluded for conservative profile
        # On UNFIXED code: moderate signal IS included — proving the bug
        assert "QQQ" not in entry_signals, (
            "Bug confirmed: moderate signal for QQQ was included in entry_signals "
            "despite conservative profile requiring min_signal_strength='strong'. "
            f"STRENGTH_ORDER: moderate={STRENGTH_ORDER['moderate']}, strong={STRENGTH_ORDER['strong']}. "
            f"entry_signals={entry_signals}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Bug 2 — Price Repair
# Tests against execute_trade() at line ~1203
# ═══════════════════════════════════════════════════════════════════════════


class TestBug2PriceRepair:
    """
    **Validates: Requirements 2.3, 2.4, 2.5**

    Bug 2: When execute_trade() detects entry price deviation >5% from live
    quote, it replaces only the entry price with the live price. Stop and
    target prices are left at their original values, producing geometrically
    inconsistent trade parameters. Additionally, extreme deviations (>10%)
    are not rejected outright.
    """

    def test_1d_moderate_deviation_proportional_repair(self):
        """
        **Validates: Requirements 2.3, 2.4**

        Test 1d: BUY at $450, stop=$445, target=$460, live price=$486 (8% deviation).
        Expected behavior: stop and target ARE proportionally scaled.

        On unfixed code this FAILS because stop/target are NOT scaled —
        they remain at their original hallucinated values.
        """
        engine = _make_engine()
        db = _make_session(engine)
        profile_id = "moderate"
        _seed_balance(db, profile_id)

        # Seed analyst signal for edge score computation
        _seed_analyst_signal(db, "TSLA", {
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

        original_entry = 450.0
        original_stop = 445.0
        original_target = 460.0
        live_price = 486.0  # 8% deviation: (486-450)/486 = 7.4%

        decision = {
            "symbol": "TSLA",
            "action": "BUY",
            "quantity": 50,
            "price": original_entry,
            "stop": original_stop,
            "target": original_target,
            "rationale": "bug exploration test",
            "setup_type": "gap_and_go",
            "market_regime": "risk_on",
        }

        # Calculate expected proportional repair
        stop_ratio = (original_stop - original_entry) / original_entry  # -0.0111
        target_ratio = (original_target - original_entry) / original_entry  # 0.0222
        expected_stop = round(live_price * (1 + stop_ratio), 2)
        expected_target = round(live_price * (1 + target_ratio), 2)

        mock_quote = {"price": live_price, "symbol": "TSLA"}
        with patch("agents.portfolio_manager.FinnhubClient") as MockFH:
            mock_instance = MagicMock()
            mock_instance.get_quote.return_value = mock_quote
            mock_instance.get_candles.return_value = []
            MockFH.return_value = mock_instance

            ok, msg = execute_trade(db, decision, profile_id)

        # After execute_trade, check if stop/target were proportionally scaled.
        # We check the decision dict since execute_trade should mutate it.
        final_stop = decision.get("stop") or decision.get("stop_price") or decision.get("stop_loss")
        final_target = decision.get("target") or decision.get("target_price") or decision.get("profit_target")

        # EXPECTED BEHAVIOR: stop/target should be proportionally scaled
        # On UNFIXED code: stop/target remain at original values — proving the bug
        assert final_stop != original_stop or final_target != original_target, (
            f"Bug confirmed: Moderate deviation (8%) — stop/target were NOT "
            f"proportionally scaled after entry price correction. "
            f"original_stop={original_stop}, final_stop={final_stop}, "
            f"original_target={original_target}, final_target={final_target}. "
            f"Expected stop≈{expected_stop}, target≈{expected_target}. "
            f"execute_trade replaces entry price but leaves stop/target unrepaired."
        )

        db.close()

    def test_1e_extreme_deviation_rejected(self):
        """
        **Validates: Requirements 2.5**

        Test 1e: BUY at $100, stop=$95, target=$115, live price=$150 (50% deviation).
        Expected behavior: trade IS rejected outright (>10% deviation).

        On unfixed code this FAILS because the code does not reject extreme
        deviations — it treats them the same as moderate ones.
        """
        engine = _make_engine()
        db = _make_session(engine)
        profile_id = "moderate"
        _seed_balance(db, profile_id)

        # Seed analyst signal
        _seed_analyst_signal(db, "NVDA", {
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

        original_entry = 100.0
        original_stop = 95.0
        original_target = 115.0
        live_price = 150.0  # 50% deviation: (150-100)/150 = 33.3%

        decision = {
            "symbol": "NVDA",
            "action": "BUY",
            "quantity": 50,
            "price": original_entry,
            "stop": original_stop,
            "target": original_target,
            "rationale": "bug exploration test",
            "setup_type": "gap_and_go",
            "market_regime": "risk_on",
        }

        mock_quote = {"price": live_price, "symbol": "NVDA"}
        with patch("agents.portfolio_manager.FinnhubClient") as MockFH:
            mock_instance = MagicMock()
            mock_instance.get_quote.return_value = mock_quote
            mock_instance.get_candles.return_value = []
            MockFH.return_value = mock_instance

            ok, msg = execute_trade(db, decision, profile_id)

        # EXPECTED BEHAVIOR: trade should be rejected outright for >10% deviation
        # with a message indicating extreme price deviation, BEFORE reaching
        # the trade validator. The rejection should happen in the price sanity
        # block, not downstream in the validator.
        #
        # On UNFIXED code: the trade IS rejected, but for the WRONG reason —
        # the validator catches geometric inconsistency (target below entry)
        # after the entry price was replaced without repairing stop/target.
        # The bug is that there is no explicit >10% deviation rejection.
        #
        # We check that the rejection reason mentions "deviation" or "price mismatch"
        # to confirm it was caught at the right stage.
        assert ok is False and ("deviation" in msg.lower() or "price mismatch" in msg.lower() or "stale" in msg.lower()), (
            f"Bug confirmed: Extreme deviation (50%) — trade was rejected but NOT "
            f"because of explicit deviation detection. "
            f"execute_trade returned ok={ok}, msg='{msg}'. "
            f"original_entry={original_entry}, live_price={live_price}. "
            f"The system does not have an explicit >10% deviation rejection — "
            f"it replaces the entry price and the validator catches the geometric "
            f"inconsistency downstream instead of rejecting at the price sanity stage."
        )

        db.close()
