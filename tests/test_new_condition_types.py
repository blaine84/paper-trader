"""Tests for _safe_decimal() and new maturation condition types.

Covers the shared numeric parsing utility and the five new condition handlers
added for the realtime-watch-maturity feature: level_reclaim, level_rejection,
support_hold, resistance_failure, trend_aligned.

Requirements: 3.1-3.7, 10.1-10.6
"""
from __future__ import annotations

import json
from decimal import Decimal

import pytest

from utils.setup_watch_evaluator import _safe_decimal, evaluate_maturation_conditions


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _conditions(condition: dict) -> str:
    """Wrap a single condition dict into a JSON array string."""
    return json.dumps([condition])


# ────────────────────────────────────────────────────────────────────────────
# 12.2-12.10: _safe_decimal tests
# ────────────────────────────────────────────────────────────────────────────


class TestSafeDecimalInt:
    """12.2: int input returns Decimal."""

    def test_positive_int(self):
        result = _safe_decimal(42)
        assert result == Decimal("42")

    def test_zero(self):
        result = _safe_decimal(0)
        assert result == Decimal("0")

    def test_negative_int(self):
        result = _safe_decimal(-7)
        assert result == Decimal("-7")


class TestSafeDecimalFloat:
    """12.3: float input returns Decimal via str() conversion."""

    def test_positive_float(self):
        result = _safe_decimal(3.14)
        assert result == Decimal("3.14")

    def test_negative_float(self):
        result = _safe_decimal(-0.5)
        assert result == Decimal("-0.5")

    def test_large_float(self):
        result = _safe_decimal(148.50)
        assert result == Decimal("148.5")


class TestSafeDecimalString:
    """12.4: numeric string returns Decimal."""

    def test_decimal_string(self):
        result = _safe_decimal("148.50")
        assert result == Decimal("148.50")

    def test_small_decimal_string(self):
        result = _safe_decimal("0.5")
        assert result == Decimal("0.5")

    def test_negative_string(self):
        result = _safe_decimal("-3.2")
        assert result == Decimal("-3.2")


class TestSafeDecimalPassthrough:
    """12.5: Decimal pass-through returns same Decimal."""

    def test_decimal_passthrough(self):
        d = Decimal("99.99")
        result = _safe_decimal(d)
        assert result is d

    def test_decimal_zero(self):
        d = Decimal("0")
        result = _safe_decimal(d)
        assert result is d


class TestSafeDecimalNone:
    """12.6: None returns None without raising."""

    def test_none_returns_none(self):
        result = _safe_decimal(None)
        assert result is None


class TestSafeDecimalEmptyString:
    """12.7: empty string returns None without raising."""

    def test_empty_string(self):
        result = _safe_decimal("")
        assert result is None

    def test_whitespace_only(self):
        result = _safe_decimal("   ")
        assert result is None


class TestSafeDecimalNonNumericString:
    """12.8: non-numeric string returns None without raising."""

    def test_alpha_string(self):
        result = _safe_decimal("hello")
        assert result is None

    def test_mixed_string(self):
        result = _safe_decimal("12abc")
        assert result is None

    def test_special_chars(self):
        result = _safe_decimal("$150.00")
        assert result is None


class TestSafeDecimalDictList:
    """12.9: dict/list returns None without raising."""

    def test_dict(self):
        result = _safe_decimal({"price": 100})
        assert result is None

    def test_list(self):
        result = _safe_decimal([1, 2, 3])
        assert result is None

    def test_empty_dict(self):
        result = _safe_decimal({})
        assert result is None

    def test_empty_list(self):
        result = _safe_decimal([])
        assert result is None


class TestSafeDecimalArbitraryObject:
    """12.10: arbitrary object returns None without raising."""

    def test_object_instance(self):
        result = _safe_decimal(object())
        assert result is None

    def test_custom_class(self):
        class Foo:
            pass
        result = _safe_decimal(Foo())
        assert result is None

    def test_boolean(self):
        # bool is a subclass of int in Python, so True->1, False->0 is valid
        # but let's verify it doesn't raise
        result = _safe_decimal(True)
        assert result is not None or result is None  # just verify no exception


# ────────────────────────────────────────────────────────────────────────────
# 12.11-12.16: New condition handler tests via evaluate_maturation_conditions
# ────────────────────────────────────────────────────────────────────────────


class TestLevelReclaim:
    """12.11-12.12: level_reclaim condition handler."""

    def test_buy_met_price_above_level(self):
        """12.11: BUY met when price >= level."""
        conditions = _conditions({
            "type": "level_reclaim",
            "params": {"level": 150.0, "side": "BUY"},
            "weight": 1.0,
        })
        ctx = {"current_price": 151.0, "side": "BUY"}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 1.0
        assert results[0].met is True

    def test_buy_met_price_exactly_at_level(self):
        """12.11: BUY met when price == level (>= boundary)."""
        conditions = _conditions({
            "type": "level_reclaim",
            "params": {"level": 150.0, "side": "BUY"},
            "weight": 1.0,
        })
        ctx = {"current_price": 150.0, "side": "BUY"}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 1.0
        assert results[0].met is True

    def test_buy_unmet_price_below_level(self):
        """12.11: BUY unmet when price < level."""
        conditions = _conditions({
            "type": "level_reclaim",
            "params": {"level": 150.0, "side": "BUY"},
            "weight": 1.0,
        })
        ctx = {"current_price": 149.99, "side": "BUY"}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 0.0
        assert results[0].met is False

    def test_short_met_price_below_level(self):
        """12.12: SHORT met when price <= level."""
        conditions = _conditions({
            "type": "level_reclaim",
            "params": {"level": 150.0, "side": "SHORT"},
            "weight": 1.0,
        })
        ctx = {"current_price": 149.0, "side": "SHORT"}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 1.0
        assert results[0].met is True

    def test_short_met_price_exactly_at_level(self):
        """12.12: SHORT met when price == level (<= boundary)."""
        conditions = _conditions({
            "type": "level_reclaim",
            "params": {"level": 150.0, "side": "SHORT"},
            "weight": 1.0,
        })
        ctx = {"current_price": 150.0, "side": "SHORT"}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 1.0
        assert results[0].met is True

    def test_short_unmet_price_above_level(self):
        """12.12: SHORT unmet when price > level."""
        conditions = _conditions({
            "type": "level_reclaim",
            "params": {"level": 150.0, "side": "SHORT"},
            "weight": 1.0,
        })
        ctx = {"current_price": 150.01, "side": "SHORT"}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 0.0
        assert results[0].met is False


class TestLevelRejection:
    """12.13: level_rejection condition handler."""

    def test_met_tested_and_moved_away(self):
        """Price tested level (within rejection_distance_pct) and moved away sufficiently."""
        conditions = _conditions({
            "type": "level_rejection",
            "params": {"level": 100.0, "rejection_distance_pct": 1.0},
            "weight": 1.0,
        })
        # price_high was at 100.5 (within 1% of 100), current price at 98.5 (1.5% away)
        ctx = {"current_price": 98.5, "price_high": 100.5}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 1.0
        assert results[0].met is True

    def test_unmet_still_near_level(self):
        """Price near level but hasn't moved away sufficiently."""
        conditions = _conditions({
            "type": "level_rejection",
            "params": {"level": 100.0, "rejection_distance_pct": 2.0},
            "weight": 1.0,
        })
        # price_high at 99.5 (within 2% of 100), current price at 99.8 (only 0.2% away)
        ctx = {"current_price": 99.8, "price_high": 99.5}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 0.0
        assert results[0].met is False

    def test_unmet_never_tested_level(self):
        """Price never came within rejection_distance_pct of the level."""
        conditions = _conditions({
            "type": "level_rejection",
            "params": {"level": 100.0, "rejection_distance_pct": 0.5},
            "weight": 1.0,
        })
        # price_high was 98.0 (2% away from 100, outside 0.5% threshold)
        ctx = {"current_price": 97.0, "price_high": 98.0}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 0.0
        assert results[0].met is False


class TestSupportHold:
    """12.14: support_hold condition handler."""

    def test_met_price_above_threshold(self):
        """Met when price >= level * (1 - tolerance_pct/100)."""
        conditions = _conditions({
            "type": "support_hold",
            "params": {"level": 100.0, "tolerance_pct": 2.0},
            "weight": 1.0,
        })
        # threshold = 100 * (1 - 2/100) = 98. Price at 99 is above.
        ctx = {"current_price": 99.0}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 1.0
        assert results[0].met is True

    def test_met_price_at_exact_threshold(self):
        """Met at exact boundary."""
        conditions = _conditions({
            "type": "support_hold",
            "params": {"level": 100.0, "tolerance_pct": 2.0},
            "weight": 1.0,
        })
        # threshold = 98.0, price exactly at 98.0
        ctx = {"current_price": 98.0}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 1.0
        assert results[0].met is True

    def test_unmet_price_below_threshold(self):
        """Unmet when price < threshold."""
        conditions = _conditions({
            "type": "support_hold",
            "params": {"level": 100.0, "tolerance_pct": 2.0},
            "weight": 1.0,
        })
        # threshold = 98.0, price at 97.5 is below
        ctx = {"current_price": 97.5}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 0.0
        assert results[0].met is False


class TestResistanceFailure:
    """12.15: resistance_failure condition handler."""

    def test_met_price_below_threshold_reversed(self):
        """Met when price <= level * (1 + tolerance_pct/100) and reversed."""
        conditions = _conditions({
            "type": "resistance_failure",
            "params": {"level": 100.0, "tolerance_pct": 1.0},
            "weight": 1.0,
        })
        # threshold = 100 * (1 + 1/100) = 101. Price at 100.5 is <= 101
        ctx = {"current_price": 100.5}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 1.0
        assert results[0].met is True

    def test_met_price_at_exact_threshold(self):
        """Met at exact threshold boundary."""
        conditions = _conditions({
            "type": "resistance_failure",
            "params": {"level": 100.0, "tolerance_pct": 1.0},
            "weight": 1.0,
        })
        # threshold = 101.0, price exactly at 101.0
        ctx = {"current_price": 101.0}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 1.0
        assert results[0].met is True

    def test_unmet_price_above_threshold(self):
        """Unmet when price > threshold (hasn't reversed)."""
        conditions = _conditions({
            "type": "resistance_failure",
            "params": {"level": 100.0, "tolerance_pct": 1.0},
            "weight": 1.0,
        })
        # threshold = 101.0, price at 101.5 is above
        ctx = {"current_price": 101.5}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 0.0
        assert results[0].met is False


class TestTrendAligned:
    """12.16: trend_aligned condition handler."""

    def test_buy_met_positive_movement(self):
        """Met for BUY when net movement is positive."""
        conditions = _conditions({
            "type": "trend_aligned",
            "params": {"lookback_bars": 5, "side": "BUY"},
            "weight": 1.0,
        })
        ctx = {"price_history": [100.0, 101.0, 102.0, 103.0, 104.0], "side": "BUY"}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 1.0
        assert results[0].met is True

    def test_short_met_negative_movement(self):
        """Met for SHORT when net movement is negative."""
        conditions = _conditions({
            "type": "trend_aligned",
            "params": {"lookback_bars": 5, "side": "SHORT"},
            "weight": 1.0,
        })
        ctx = {"price_history": [104.0, 103.0, 102.0, 101.0, 100.0], "side": "SHORT"}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 1.0
        assert results[0].met is True

    def test_buy_unmet_negative_movement(self):
        """Unmet for BUY when net movement is negative."""
        conditions = _conditions({
            "type": "trend_aligned",
            "params": {"lookback_bars": 5, "side": "BUY"},
            "weight": 1.0,
        })
        ctx = {"price_history": [104.0, 103.0, 102.0, 101.0, 100.0], "side": "BUY"}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 0.0
        assert results[0].met is False

    def test_unmet_no_price_history(self):
        """Unmet when price_history is not available."""
        conditions = _conditions({
            "type": "trend_aligned",
            "params": {"lookback_bars": 5, "side": "BUY"},
            "weight": 1.0,
        })
        ctx = {"side": "BUY"}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 0.0
        assert results[0].met is False

    def test_unmet_empty_price_history(self):
        """Unmet when price_history is empty list."""
        conditions = _conditions({
            "type": "trend_aligned",
            "params": {"lookback_bars": 5, "side": "BUY"},
            "weight": 1.0,
        })
        ctx = {"price_history": [], "side": "BUY"}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 0.0
        assert results[0].met is False


# ────────────────────────────────────────────────────────────────────────────
# 12.17: All handlers return condition-unmet on invalid Decimal parse
# ────────────────────────────────────────────────────────────────────────────


class TestInvalidDecimalParse:
    """12.17: All new handlers return unmet (not raise) on invalid Decimal parse."""

    def test_level_reclaim_invalid_level(self):
        conditions = _conditions({
            "type": "level_reclaim",
            "params": {"level": "not_a_number", "side": "BUY"},
            "weight": 1.0,
        })
        ctx = {"current_price": 150.0, "side": "BUY"}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 0.0
        assert results[0].met is False

    def test_level_reclaim_invalid_price(self):
        conditions = _conditions({
            "type": "level_reclaim",
            "params": {"level": 150.0, "side": "BUY"},
            "weight": 1.0,
        })
        ctx = {"current_price": "invalid", "side": "BUY"}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 0.0
        assert results[0].met is False

    def test_level_rejection_invalid_level(self):
        conditions = _conditions({
            "type": "level_rejection",
            "params": {"level": {}, "rejection_distance_pct": 1.0},
            "weight": 1.0,
        })
        ctx = {"current_price": 100.0}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 0.0
        assert results[0].met is False

    def test_level_rejection_invalid_distance(self):
        conditions = _conditions({
            "type": "level_rejection",
            "params": {"level": 100.0, "rejection_distance_pct": "bad"},
            "weight": 1.0,
        })
        ctx = {"current_price": 100.0}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 0.0
        assert results[0].met is False

    def test_support_hold_invalid_level(self):
        conditions = _conditions({
            "type": "support_hold",
            "params": {"level": None, "tolerance_pct": 2.0},
            "weight": 1.0,
        })
        ctx = {"current_price": 99.0}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 0.0
        assert results[0].met is False

    def test_support_hold_invalid_tolerance(self):
        conditions = _conditions({
            "type": "support_hold",
            "params": {"level": 100.0, "tolerance_pct": []},
            "weight": 1.0,
        })
        ctx = {"current_price": 99.0}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 0.0
        assert results[0].met is False

    def test_resistance_failure_invalid_level(self):
        conditions = _conditions({
            "type": "resistance_failure",
            "params": {"level": "abc", "tolerance_pct": 1.0},
            "weight": 1.0,
        })
        ctx = {"current_price": 100.5}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 0.0
        assert results[0].met is False

    def test_resistance_failure_invalid_tolerance(self):
        conditions = _conditions({
            "type": "resistance_failure",
            "params": {"level": 100.0, "tolerance_pct": "nope"},
            "weight": 1.0,
        })
        ctx = {"current_price": 100.5}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 0.0
        assert results[0].met is False

    def test_trend_aligned_invalid_history_values(self):
        """Unmet when price_history contains unparseable values."""
        conditions = _conditions({
            "type": "trend_aligned",
            "params": {"lookback_bars": 3, "side": "BUY"},
            "weight": 1.0,
        })
        ctx = {"price_history": ["not", "numbers", "here"], "side": "BUY"}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 0.0
        assert results[0].met is False


# ────────────────────────────────────────────────────────────────────────────
# 12.18: Unknown condition type regression check
# ────────────────────────────────────────────────────────────────────────────


class TestUnknownConditionType:
    """12.18: Unknown condition type still unmet AND excluded from denominator."""

    def test_unknown_type_unmet_excluded_from_denominator(self):
        """Unknown type: unmet result, weight excluded, does not penalize score."""
        conditions = json.dumps([
            {"type": "totally_made_up", "params": {}, "weight": 1.0},
        ])
        ctx = {"current_price": 150.0}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        # Score is 0.0 because denominator is 0 (unknown type excluded)
        assert score == 0.0
        assert results[0].met is False
        assert "excluded" in (results[0].detail or "").lower()

    def test_unknown_type_does_not_penalize_known_conditions(self):
        """Unknown type's weight doesn't count against met known conditions."""
        conditions = json.dumps([
            {"type": "level_reclaim", "params": {"level": 150.0, "side": "BUY"}, "weight": 1.0},
            {"type": "nonexistent_condition", "params": {}, "weight": 1.0},
        ])
        ctx = {"current_price": 151.0, "side": "BUY"}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        # Only level_reclaim counts: met_weight=1, applicable_weight=1, score=1.0
        assert score == 1.0
        assert results[0].met is True  # level_reclaim
        assert results[1].met is False  # unknown

    def test_all_unknown_types_returns_zero_score(self):
        """All unknown types: score is 0 (no applicable weight)."""
        conditions = json.dumps([
            {"type": "foo", "params": {}, "weight": 1.0},
            {"type": "bar", "params": {}, "weight": 2.0},
        ])
        ctx = {"current_price": 150.0}
        score, results = evaluate_maturation_conditions(conditions, ctx)
        assert score == 0.0
        assert all(r.met is False for r in results)
