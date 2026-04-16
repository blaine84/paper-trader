"""
Unit tests for evaluate_invalidators() in agents/price_monitor.py.
Tests the Thesis Invalidation Engine — Task 5.1.
"""

import json
import pytest
from unittest.mock import MagicMock

from agents.price_monitor import evaluate_invalidators


def _make_trade(invalidators_list):
    """Create a mock trade with the given invalidators JSON."""
    trade = MagicMock()
    trade.id = 1
    if invalidators_list is None:
        trade.invalidators = None
    elif isinstance(invalidators_list, str):
        trade.invalidators = invalidators_list
    else:
        trade.invalidators = json.dumps(invalidators_list)
    return trade


# ─── EMPTY / MALFORMED INPUT ─────────────────────────────────────────────────

class TestEmptyAndMalformedInput:
    def test_none_invalidators_returns_empty(self):
        trade = _make_trade(None)
        assert evaluate_invalidators(trade, 100.0) == []

    def test_empty_string_returns_empty(self):
        trade = _make_trade("")
        assert evaluate_invalidators(trade, 100.0) == []

    def test_whitespace_string_returns_empty(self):
        trade = _make_trade("   ")
        assert evaluate_invalidators(trade, 100.0) == []

    def test_malformed_json_returns_empty(self):
        trade = _make_trade("{not valid json")
        assert evaluate_invalidators(trade, 100.0) == []

    def test_json_not_a_list_returns_empty(self):
        trade = MagicMock()
        trade.id = 2
        trade.invalidators = json.dumps({"type": "price_below_level"})
        assert evaluate_invalidators(trade, 100.0) == []

    def test_non_dict_items_skipped(self):
        trade = _make_trade(["not_a_dict", 42])
        assert evaluate_invalidators(trade, 100.0) == []


# ─── TICK CONFIRMATION ────────────────────────────────────────────────────────

class TestTickConfirmation:
    def test_price_below_level_breached(self):
        inv = [{"type": "price_below_level", "reference": "162.50", "confirmation": "tick", "lookback_bars": 0}]
        trade = _make_trade(inv)
        result = evaluate_invalidators(trade, 160.0)
        assert len(result) == 1
        assert result[0]["type"] == "price_below_level"

    def test_price_below_level_not_breached(self):
        inv = [{"type": "price_below_level", "reference": "162.50", "confirmation": "tick", "lookback_bars": 0}]
        trade = _make_trade(inv)
        result = evaluate_invalidators(trade, 165.0)
        assert result == []

    def test_price_below_level_exact_price_not_breached(self):
        """At exactly the reference level, price_below_level is NOT breached (need strictly below)."""
        inv = [{"type": "price_below_level", "reference": "162.50", "confirmation": "tick", "lookback_bars": 0}]
        trade = _make_trade(inv)
        result = evaluate_invalidators(trade, 162.50)
        assert result == []

    def test_price_above_level_breached(self):
        inv = [{"type": "price_above_level", "reference": "170.00", "confirmation": "tick", "lookback_bars": 0}]
        trade = _make_trade(inv)
        result = evaluate_invalidators(trade, 172.0)
        assert len(result) == 1
        assert result[0]["type"] == "price_above_level"

    def test_price_above_level_not_breached(self):
        inv = [{"type": "price_above_level", "reference": "170.00", "confirmation": "tick", "lookback_bars": 0}]
        trade = _make_trade(inv)
        result = evaluate_invalidators(trade, 168.0)
        assert result == []

    def test_price_above_level_exact_price_not_breached(self):
        inv = [{"type": "price_above_level", "reference": "170.00", "confirmation": "tick", "lookback_bars": 0}]
        trade = _make_trade(inv)
        result = evaluate_invalidators(trade, 170.0)
        assert result == []


# ─── STRUCTURE BREAK SKIPPED ──────────────────────────────────────────────────

class TestStructureBreakSkipped:
    def test_structure_break_always_skipped(self):
        inv = [{"type": "structure_break", "reference": "higher_low", "confirmation": "5m_close", "lookback_bars": 1}]
        trade = _make_trade(inv)
        result = evaluate_invalidators(trade, 100.0)
        assert result == []


# ─── 5M CLOSE CONFIRMATION ───────────────────────────────────────────────────

class TestFiveMinCloseConfirmation:
    def test_5m_close_breached_below(self):
        inv = [{"type": "price_below_level", "reference": "162.50", "confirmation": "5m_close", "lookback_bars": 1}]
        trade = _make_trade(inv)
        candle_data = {"closes": [161.0]}
        result = evaluate_invalidators(trade, 161.0, candle_data)
        assert len(result) == 1

    def test_5m_close_not_breached_below(self):
        inv = [{"type": "price_below_level", "reference": "162.50", "confirmation": "5m_close", "lookback_bars": 1}]
        trade = _make_trade(inv)
        candle_data = {"closes": [163.0]}
        result = evaluate_invalidators(trade, 161.0, candle_data)
        assert result == []

    def test_5m_close_breached_above(self):
        inv = [{"type": "price_above_level", "reference": "170.00", "confirmation": "5m_close", "lookback_bars": 1}]
        trade = _make_trade(inv)
        candle_data = {"closes": [171.0]}
        result = evaluate_invalidators(trade, 171.0, candle_data)
        assert len(result) == 1

    def test_5m_close_skipped_when_no_candle_data(self):
        inv = [{"type": "price_below_level", "reference": "162.50", "confirmation": "5m_close", "lookback_bars": 1}]
        trade = _make_trade(inv)
        result = evaluate_invalidators(trade, 160.0, candle_data=None)
        assert result == []

    def test_5m_close_multiple_lookback_bars(self):
        inv = [{"type": "price_below_level", "reference": "162.50", "confirmation": "5m_close", "lookback_bars": 3}]
        trade = _make_trade(inv)
        # All 3 recent closes below reference
        candle_data = {"closes": [165.0, 161.0, 160.5, 161.2]}
        result = evaluate_invalidators(trade, 160.0, candle_data)
        assert len(result) == 1

    def test_5m_close_not_enough_bars(self):
        inv = [{"type": "price_below_level", "reference": "162.50", "confirmation": "5m_close", "lookback_bars": 3}]
        trade = _make_trade(inv)
        candle_data = {"closes": [161.0, 160.5]}  # Only 2 bars, need 3
        result = evaluate_invalidators(trade, 160.0, candle_data)
        assert result == []

    def test_5m_close_partial_breach_not_confirmed(self):
        """If one of the lookback bars is above reference, breach is not confirmed."""
        inv = [{"type": "price_below_level", "reference": "162.50", "confirmation": "5m_close", "lookback_bars": 2}]
        trade = _make_trade(inv)
        candle_data = {"closes": [161.0, 163.0]}  # Second bar above reference
        result = evaluate_invalidators(trade, 160.0, candle_data)
        assert result == []


# ─── INDICATOR REFERENCE RESOLUTION ──────────────────────────────────────────

class TestIndicatorResolution:
    def test_vwap_resolved_from_candle_data(self):
        inv = [{"type": "price_below_level", "reference": "VWAP", "confirmation": "tick", "lookback_bars": 0}]
        trade = _make_trade(inv)
        candle_data = {"vwap": 163.20}
        result = evaluate_invalidators(trade, 162.0, candle_data)
        assert len(result) == 1

    def test_vwap_case_insensitive(self):
        inv = [{"type": "price_below_level", "reference": "vwap", "confirmation": "tick", "lookback_bars": 0}]
        trade = _make_trade(inv)
        candle_data = {"VWAP": 163.20}
        result = evaluate_invalidators(trade, 162.0, candle_data)
        assert len(result) == 1

    def test_unresolvable_indicator_skipped(self):
        inv = [{"type": "price_below_level", "reference": "VWAP", "confirmation": "tick", "lookback_bars": 0}]
        trade = _make_trade(inv)
        # No candle_data provided — VWAP can't be resolved
        result = evaluate_invalidators(trade, 160.0)
        assert result == []


# ─── MULTIPLE INVALIDATORS (OR LOGIC) ────────────────────────────────────────

class TestMultipleInvalidators:
    def test_one_of_two_breached(self):
        inv = [
            {"type": "price_below_level", "reference": "162.50", "confirmation": "tick", "lookback_bars": 0},
            {"type": "price_below_level", "reference": "155.00", "confirmation": "tick", "lookback_bars": 0},
        ]
        trade = _make_trade(inv)
        result = evaluate_invalidators(trade, 160.0)
        # Only the first is breached (160 < 162.50), second is not (160 > 155)
        assert len(result) == 1
        assert result[0]["reference"] == "162.50"

    def test_both_breached(self):
        inv = [
            {"type": "price_below_level", "reference": "162.50", "confirmation": "tick", "lookback_bars": 0},
            {"type": "price_below_level", "reference": "165.00", "confirmation": "tick", "lookback_bars": 0},
        ]
        trade = _make_trade(inv)
        result = evaluate_invalidators(trade, 160.0)
        assert len(result) == 2

    def test_none_breached(self):
        inv = [
            {"type": "price_below_level", "reference": "150.00", "confirmation": "tick", "lookback_bars": 0},
            {"type": "price_above_level", "reference": "170.00", "confirmation": "tick", "lookback_bars": 0},
        ]
        trade = _make_trade(inv)
        result = evaluate_invalidators(trade, 160.0)
        assert result == []

    def test_structure_break_mixed_with_price_checks(self):
        inv = [
            {"type": "structure_break", "reference": "higher_low", "confirmation": "5m_close", "lookback_bars": 1},
            {"type": "price_below_level", "reference": "162.50", "confirmation": "tick", "lookback_bars": 0},
        ]
        trade = _make_trade(inv)
        result = evaluate_invalidators(trade, 160.0)
        # structure_break skipped, price_below_level breached
        assert len(result) == 1
        assert result[0]["type"] == "price_below_level"


# ─── MISSING / INVALID FIELDS ────────────────────────────────────────────────

class TestMissingFields:
    def test_missing_reference_skipped(self):
        inv = [{"type": "price_below_level", "confirmation": "tick", "lookback_bars": 0}]
        trade = _make_trade(inv)
        result = evaluate_invalidators(trade, 160.0)
        assert result == []

    def test_unknown_type_skipped(self):
        inv = [{"type": "unknown_type", "reference": "162.50", "confirmation": "tick", "lookback_bars": 0}]
        trade = _make_trade(inv)
        result = evaluate_invalidators(trade, 160.0)
        assert result == []

    def test_unknown_confirmation_skipped(self):
        inv = [{"type": "price_below_level", "reference": "162.50", "confirmation": "1h_close", "lookback_bars": 0}]
        trade = _make_trade(inv)
        result = evaluate_invalidators(trade, 160.0)
        assert result == []
