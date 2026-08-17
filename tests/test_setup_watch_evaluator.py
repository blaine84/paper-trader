"""Tests for utils/setup_watch_evaluator.py — condition evaluation and maturity scoring.

Requirements: 4.2, 4.2.1, 4.3, 4.12, 5.1-5.3, 5.3.1, 5.7, 12.8
"""
from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from utils.setup_watch_evaluator import (
    ConditionResult,
    EvaluationResult,
    evaluate_invalidation_conditions,
    evaluate_maturation_conditions,
    evaluate_watch,
    validate_draft_geometry,
)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _market_context(**overrides):
    ctx = {
        "current_price": 150.0,
        "market_regime": "bullish",
        "catalyst_timestamp": (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat(),
        "held_symbols": set(),
        "key_levels": {"support": [145.0, 140.0], "resistance": [155.0, 160.0]},
        "symbol": "AAPL",
        "current_hour_et": 11,
    }
    ctx.update(overrides)
    return ctx


def _mat_conds(*conditions) -> str:
    """Build a JSON list of maturation conditions."""
    return json.dumps(list(conditions))


def _inv_conds(*conditions) -> str:
    """Build a JSON list of invalidation conditions."""
    return json.dumps(list(conditions))


# ────────────────────────────────────────────────────────────────────────────
# Test 1: price_zone — met inside range, unmet outside, Decimal-exact at boundaries
# ────────────────────────────────────────────────────────────────────────────


class TestPriceZone:
    """price_zone maturation handler tests."""

    def test_met_inside_range(self):
        conds = _mat_conds({"type": "price_zone", "params": {"low": 140, "high": 160}, "weight": 1.0})
        ctx = _market_context(current_price=150.0)
        score, results = evaluate_maturation_conditions(conds, ctx)
        assert score == 1.0
        assert results[0].met is True

    def test_unmet_outside_range(self):
        conds = _mat_conds({"type": "price_zone", "params": {"low": 160, "high": 170}, "weight": 1.0})
        ctx = _market_context(current_price=150.0)
        score, results = evaluate_maturation_conditions(conds, ctx)
        assert score == 0.0
        assert results[0].met is False

    def test_met_at_lower_boundary(self):
        """Price exactly at lower boundary — inclusive."""
        conds = _mat_conds({"type": "price_zone", "params": {"low": 150, "high": 160}, "weight": 1.0})
        ctx = _market_context(current_price=150.0)
        score, results = evaluate_maturation_conditions(conds, ctx)
        assert score == 1.0
        assert results[0].met is True

    def test_met_at_upper_boundary(self):
        """Price exactly at upper boundary — inclusive."""
        conds = _mat_conds({"type": "price_zone", "params": {"low": 140, "high": 150}, "weight": 1.0})
        ctx = _market_context(current_price=150.0)
        score, results = evaluate_maturation_conditions(conds, ctx)
        assert score == 1.0
        assert results[0].met is True

    def test_unmet_just_below_lower_boundary(self):
        """Price at 139.99 with low=140 — unmet (Decimal exact)."""
        conds = _mat_conds({"type": "price_zone", "params": {"low": "140.00", "high": "160.00"}, "weight": 1.0})
        ctx = _market_context(current_price="139.99")
        score, results = evaluate_maturation_conditions(conds, ctx)
        assert score == 0.0
        assert results[0].met is False

    def test_unmet_just_above_upper_boundary(self):
        """Price at 160.01 with high=160 — unmet (Decimal exact)."""
        conds = _mat_conds({"type": "price_zone", "params": {"low": "140.00", "high": "160.00"}, "weight": 1.0})
        ctx = _market_context(current_price="160.01")
        score, results = evaluate_maturation_conditions(conds, ctx)
        assert score == 0.0
        assert results[0].met is False


# ────────────────────────────────────────────────────────────────────────────
# Test 2: regime_aligned, catalyst_fresh, time_window, key_level_proximity
# ────────────────────────────────────────────────────────────────────────────


class TestRegimeAligned:
    def test_met_matching_regime(self):
        conds = _mat_conds({"type": "regime_aligned", "params": {"required_regime": "bullish"}, "weight": 1.0})
        ctx = _market_context(market_regime="bullish")
        score, results = evaluate_maturation_conditions(conds, ctx)
        assert score == 1.0
        assert results[0].met is True

    def test_unmet_different_regime(self):
        conds = _mat_conds({"type": "regime_aligned", "params": {"required_regime": "bearish"}, "weight": 1.0})
        ctx = _market_context(market_regime="bullish")
        score, results = evaluate_maturation_conditions(conds, ctx)
        assert score == 0.0
        assert results[0].met is False

    def test_case_insensitive(self):
        conds = _mat_conds({"type": "regime_aligned", "params": {"required_regime": "Bullish"}, "weight": 1.0})
        ctx = _market_context(market_regime="BULLISH")
        score, results = evaluate_maturation_conditions(conds, ctx)
        assert score == 1.0
        assert results[0].met is True


class TestCatalystFresh:
    def test_met_within_max_age(self):
        fresh_ts = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        conds = _mat_conds({"type": "catalyst_fresh", "params": {"max_age_minutes": 60}, "weight": 1.0})
        ctx = _market_context(catalyst_timestamp=fresh_ts)
        score, results = evaluate_maturation_conditions(conds, ctx)
        assert score == 1.0
        assert results[0].met is True

    def test_unmet_beyond_max_age(self):
        old_ts = (datetime.now(timezone.utc) - timedelta(minutes=120)).isoformat()
        conds = _mat_conds({"type": "catalyst_fresh", "params": {"max_age_minutes": 60}, "weight": 1.0})
        ctx = _market_context(catalyst_timestamp=old_ts)
        score, results = evaluate_maturation_conditions(conds, ctx)
        assert score == 0.0
        assert results[0].met is False


class TestTimeWindow:
    def test_met_within_window(self):
        conds = _mat_conds({"type": "time_window", "params": {"start_hour": 9, "end_hour": 16}, "weight": 1.0})
        ctx = _market_context(current_hour_et=11)
        score, results = evaluate_maturation_conditions(conds, ctx)
        assert score == 1.0
        assert results[0].met is True

    def test_unmet_outside_window(self):
        conds = _mat_conds({"type": "time_window", "params": {"start_hour": 9, "end_hour": 11}, "weight": 1.0})
        ctx = _market_context(current_hour_et=14)
        score, results = evaluate_maturation_conditions(conds, ctx)
        assert score == 0.0
        assert results[0].met is False

    def test_met_at_boundary_start(self):
        conds = _mat_conds({"type": "time_window", "params": {"start_hour": 9, "end_hour": 16}, "weight": 1.0})
        ctx = _market_context(current_hour_et=9)
        score, results = evaluate_maturation_conditions(conds, ctx)
        assert score == 1.0
        assert results[0].met is True

    def test_met_at_boundary_end(self):
        conds = _mat_conds({"type": "time_window", "params": {"start_hour": 9, "end_hour": 16}, "weight": 1.0})
        ctx = _market_context(current_hour_et=16)
        score, results = evaluate_maturation_conditions(conds, ctx)
        assert score == 1.0
        assert results[0].met is True


class TestKeyLevelProximity:
    def test_met_near_support(self):
        """Price at 145 is 0% away from support level 145 — should be met."""
        conds = _mat_conds({
            "type": "key_level_proximity",
            "params": {"level_type": "support", "within_pct": 2.0},
            "weight": 1.0,
        })
        ctx = _market_context(current_price=145.0, key_levels={"support": [145.0], "resistance": [160.0]})
        score, results = evaluate_maturation_conditions(conds, ctx)
        assert score == 1.0
        assert results[0].met is True

    def test_unmet_far_from_support(self):
        """Price at 150 is ~3.4% from support 145 — outside 2% threshold."""
        conds = _mat_conds({
            "type": "key_level_proximity",
            "params": {"level_type": "support", "within_pct": 2.0},
            "weight": 1.0,
        })
        ctx = _market_context(current_price=150.0, key_levels={"support": [145.0], "resistance": [160.0]})
        score, results = evaluate_maturation_conditions(conds, ctx)
        assert score == 0.0
        assert results[0].met is False

    def test_met_near_resistance(self):
        """Price at 155 is 0% from resistance 155 — met."""
        conds = _mat_conds({
            "type": "key_level_proximity",
            "params": {"level_type": "resistance", "within_pct": 1.0},
            "weight": 1.0,
        })
        ctx = _market_context(current_price=155.0, key_levels={"support": [140.0], "resistance": [155.0]})
        score, results = evaluate_maturation_conditions(conds, ctx)
        assert score == 1.0
        assert results[0].met is True


# ────────────────────────────────────────────────────────────────────────────
# Test 3: price_breach triggers on correct side only (Decimal)
# ────────────────────────────────────────────────────────────────────────────


class TestPriceBreach:
    def test_triggers_above(self):
        """Price above level triggers when direction=above."""
        conds = _inv_conds({"type": "price_breach", "params": {"level": "155.00", "direction": "above"}})
        ctx = _market_context(current_price="155.01")
        triggered, reason = evaluate_invalidation_conditions(conds, ctx)
        assert triggered is True
        assert "price_breach" in reason

    def test_no_trigger_below_when_direction_above(self):
        """Price below level does NOT trigger when direction=above."""
        conds = _inv_conds({"type": "price_breach", "params": {"level": "155.00", "direction": "above"}})
        ctx = _market_context(current_price="154.99")
        triggered, reason = evaluate_invalidation_conditions(conds, ctx)
        assert triggered is False
        assert reason is None

    def test_triggers_below(self):
        """Price below level triggers when direction=below."""
        conds = _inv_conds({"type": "price_breach", "params": {"level": "140.00", "direction": "below"}})
        ctx = _market_context(current_price="139.99")
        triggered, reason = evaluate_invalidation_conditions(conds, ctx)
        assert triggered is True
        assert "price_breach" in reason

    def test_no_trigger_above_when_direction_below(self):
        """Price above level does NOT trigger when direction=below."""
        conds = _inv_conds({"type": "price_breach", "params": {"level": "140.00", "direction": "below"}})
        ctx = _market_context(current_price="140.01")
        triggered, reason = evaluate_invalidation_conditions(conds, ctx)
        assert triggered is False
        assert reason is None

    def test_at_level_does_not_trigger_above(self):
        """Price exactly at level does NOT trigger direction=above (strictly >)."""
        conds = _inv_conds({"type": "price_breach", "params": {"level": "150.00", "direction": "above"}})
        ctx = _market_context(current_price="150.00")
        triggered, reason = evaluate_invalidation_conditions(conds, ctx)
        assert triggered is False

    def test_at_level_does_not_trigger_below(self):
        """Price exactly at level does NOT trigger direction=below (strictly <)."""
        conds = _inv_conds({"type": "price_breach", "params": {"level": "150.00", "direction": "below"}})
        ctx = _market_context(current_price="150.00")
        triggered, reason = evaluate_invalidation_conditions(conds, ctx)
        assert triggered is False


# ────────────────────────────────────────────────────────────────────────────
# Test 4: regime_flip, catalyst_expired, exposure_conflict
# ────────────────────────────────────────────────────────────────────────────


class TestRegimeFlip:
    def test_triggers_when_in_blocked_set(self):
        conds = _inv_conds({"type": "regime_flip", "params": {"blocked_regimes": ["bearish", "choppy"]}})
        ctx = _market_context(market_regime="bearish")
        triggered, reason = evaluate_invalidation_conditions(conds, ctx)
        assert triggered is True
        assert "regime_flip" in reason

    def test_no_trigger_when_not_blocked(self):
        conds = _inv_conds({"type": "regime_flip", "params": {"blocked_regimes": ["bearish", "choppy"]}})
        ctx = _market_context(market_regime="bullish")
        triggered, reason = evaluate_invalidation_conditions(conds, ctx)
        assert triggered is False


class TestCatalystExpired:
    def test_triggers_when_age_exceeds_max(self):
        old_ts = (datetime.now(timezone.utc) - timedelta(minutes=120)).isoformat()
        conds = _inv_conds({"type": "catalyst_expired", "params": {"max_age_minutes": 60}})
        ctx = _market_context(catalyst_timestamp=old_ts)
        triggered, reason = evaluate_invalidation_conditions(conds, ctx)
        assert triggered is True
        assert "catalyst_expired" in reason

    def test_no_trigger_when_fresh(self):
        fresh_ts = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        conds = _inv_conds({"type": "catalyst_expired", "params": {"max_age_minutes": 60}})
        ctx = _market_context(catalyst_timestamp=fresh_ts)
        triggered, reason = evaluate_invalidation_conditions(conds, ctx)
        assert triggered is False


class TestExposureConflict:
    def test_triggers_when_symbol_held(self):
        conds = _inv_conds({"type": "exposure_conflict", "params": {}})
        ctx = _market_context(symbol="AAPL", held_symbols={"AAPL", "MSFT"})
        triggered, reason = evaluate_invalidation_conditions(conds, ctx)
        assert triggered is True
        assert "exposure_conflict" in reason

    def test_no_trigger_when_symbol_not_held(self):
        conds = _inv_conds({"type": "exposure_conflict", "params": {}})
        ctx = _market_context(symbol="AAPL", held_symbols={"MSFT", "GOOG"})
        triggered, reason = evaluate_invalidation_conditions(conds, ctx)
        assert triggered is False

    def test_no_trigger_with_empty_holdings(self):
        conds = _inv_conds({"type": "exposure_conflict", "params": {}})
        ctx = _market_context(symbol="AAPL", held_symbols=set())
        triggered, reason = evaluate_invalidation_conditions(conds, ctx)
        assert triggered is False


# ────────────────────────────────────────────────────────────────────────────
# Test 5: Invalidation is evaluated before maturation
# ────────────────────────────────────────────────────────────────────────────


class TestInvalidationPriority:
    def test_invalidation_short_circuits_maturation(self):
        """When invalidated, maturation is never evaluated — score is 0, results empty."""
        mat_conds = _mat_conds(
            {"type": "price_zone", "params": {"low": 140, "high": 160}, "weight": 1.0}
        )
        inv_conds = _inv_conds(
            {"type": "exposure_conflict", "params": {}}
        )
        ctx = _market_context(symbol="AAPL", held_symbols={"AAPL"})

        result = evaluate_watch(mat_conds, inv_conds, ctx)

        assert result.invalidated is True
        assert result.invalidation_reason is not None
        assert result.maturity_score == 0.0
        assert result.condition_results == []

    def test_no_invalidation_allows_maturation(self):
        """When not invalidated, maturation proceeds normally."""
        mat_conds = _mat_conds(
            {"type": "price_zone", "params": {"low": 140, "high": 160}, "weight": 1.0}
        )
        inv_conds = _inv_conds(
            {"type": "exposure_conflict", "params": {}}
        )
        ctx = _market_context(symbol="AAPL", held_symbols=set())

        result = evaluate_watch(mat_conds, inv_conds, ctx)

        assert result.invalidated is False
        assert result.invalidation_reason is None
        assert result.maturity_score == 1.0
        assert len(result.condition_results) == 1


# ────────────────────────────────────────────────────────────────────────────
# Test 6: Weighted score formula verified against hand-computed values
# ────────────────────────────────────────────────────────────────────────────


class TestWeightedScoreFormula:
    def test_single_met_weight_1(self):
        """1 condition met with weight 1 → score = 1/1 = 1.0"""
        conds = _mat_conds(
            {"type": "price_zone", "params": {"low": 140, "high": 160}, "weight": 1.0}
        )
        ctx = _market_context(current_price=150.0)
        score, _ = evaluate_maturation_conditions(conds, ctx)
        assert score == 1.0

    def test_one_met_one_unmet_equal_weight(self):
        """2 conditions, weight 1 each, 1 met → score = 1/2 = 0.5"""
        conds = _mat_conds(
            {"type": "price_zone", "params": {"low": 140, "high": 160}, "weight": 1.0},
            {"type": "regime_aligned", "params": {"required_regime": "bearish"}, "weight": 1.0},
        )
        ctx = _market_context(current_price=150.0, market_regime="bullish")
        score, _ = evaluate_maturation_conditions(conds, ctx)
        assert abs(score - 0.5) < 1e-9

    def test_weighted_asymmetric(self):
        """Met weight=3, unmet weight=1 → score = 3/4 = 0.75"""
        conds = _mat_conds(
            {"type": "price_zone", "params": {"low": 140, "high": 160}, "weight": 3.0},
            {"type": "regime_aligned", "params": {"required_regime": "bearish"}, "weight": 1.0},
        )
        ctx = _market_context(current_price=150.0, market_regime="bullish")
        score, _ = evaluate_maturation_conditions(conds, ctx)
        assert abs(score - 0.75) < 1e-9

    def test_complex_weighting(self):
        """Met weights: 2+1=3, Total: 2+1+2=5, score = 3/5 = 0.6"""
        conds = _mat_conds(
            {"type": "price_zone", "params": {"low": 140, "high": 160}, "weight": 2.0},  # met
            {"type": "regime_aligned", "params": {"required_regime": "bullish"}, "weight": 1.0},  # met
            {"type": "regime_aligned", "params": {"required_regime": "bearish"}, "weight": 2.0},  # unmet
        )
        ctx = _market_context(current_price=150.0, market_regime="bullish")
        score, _ = evaluate_maturation_conditions(conds, ctx)
        assert abs(score - 0.6) < 1e-9


# ────────────────────────────────────────────────────────────────────────────
# Test 7: Score is 0.0 with nothing met, 1.0 with everything met
# ────────────────────────────────────────────────────────────────────────────


class TestScoreExtremes:
    def test_all_met(self):
        conds = _mat_conds(
            {"type": "price_zone", "params": {"low": 140, "high": 160}, "weight": 1.0},
            {"type": "regime_aligned", "params": {"required_regime": "bullish"}, "weight": 1.0},
        )
        ctx = _market_context(current_price=150.0, market_regime="bullish")
        score, _ = evaluate_maturation_conditions(conds, ctx)
        assert score == 1.0

    def test_none_met(self):
        conds = _mat_conds(
            {"type": "price_zone", "params": {"low": 200, "high": 210}, "weight": 1.0},
            {"type": "regime_aligned", "params": {"required_regime": "bearish"}, "weight": 1.0},
        )
        ctx = _market_context(current_price=150.0, market_regime="bullish")
        score, _ = evaluate_maturation_conditions(conds, ctx)
        assert score == 0.0


# ────────────────────────────────────────────────────────────────────────────
# Test 8: Zero total weight returns 0.0 rather than dividing by zero
# ────────────────────────────────────────────────────────────────────────────


class TestZeroWeight:
    def test_empty_conditions_returns_zero(self):
        """No conditions → total_applicable_weight=0, score=0.0."""
        conds = _mat_conds()  # empty list
        ctx = _market_context()
        score, results = evaluate_maturation_conditions(conds, ctx)
        assert score == 0.0
        assert results == []

    def test_all_unknown_types_returns_zero(self):
        """All conditions are unknown type → excluded from denominator → score=0.0."""
        conds = _mat_conds(
            {"type": "future_condition_v2", "params": {}, "weight": 1.0},
            {"type": "another_unknown", "params": {}, "weight": 2.0},
        )
        ctx = _market_context()
        score, results = evaluate_maturation_conditions(conds, ctx)
        assert score == 0.0
        assert len(results) == 2
        assert all(r.met is False for r in results)


# ────────────────────────────────────────────────────────────────────────────
# Test 9: All conditions unknown/excluded — returns 0.0 with DEBUG log, no exception
# ────────────────────────────────────────────────────────────────────────────


class TestAllUnknownWithDebugLog:
    def test_debug_logged_on_zero_applicable_weight(self, caplog):
        """When all conditions are unknown, a DEBUG message is logged."""
        conds = _mat_conds(
            {"type": "future_condition_v2", "params": {}, "weight": 1.0},
        )
        ctx = _market_context()

        with caplog.at_level(logging.DEBUG, logger="utils.setup_watch_evaluator"):
            score, results = evaluate_maturation_conditions(conds, ctx)

        assert score == 0.0
        assert any("applicable weight is zero" in msg for msg in caplog.messages)

    def test_no_exception_raised(self):
        """Zero applicable weight does not raise — returns cleanly."""
        conds = _mat_conds(
            {"type": "future_condition_v2", "params": {}, "weight": 5.0},
            {"type": "mystery_type", "params": {}, "weight": 3.0},
        )
        ctx = _market_context()
        # Should not raise
        score, results = evaluate_maturation_conditions(conds, ctx)
        assert score == 0.0


# ────────────────────────────────────────────────────────────────────────────
# Test 10: Unknown maturation type is unmet AND excluded from the denominator
# ────────────────────────────────────────────────────────────────────────────


class TestUnknownMaturationType:
    def test_excluded_from_denominator(self):
        """Unknown type excluded: met weight=1, total applicable=1 (not 2), score=1.0."""
        conds = _mat_conds(
            {"type": "price_zone", "params": {"low": 140, "high": 160}, "weight": 1.0},  # met
            {"type": "future_condition_v2", "params": {}, "weight": 1.0},  # unknown → excluded
        )
        ctx = _market_context(current_price=150.0)
        score, results = evaluate_maturation_conditions(conds, ctx)
        # If unknown were in denominator, score would be 1/2=0.5
        # Since excluded, score = 1/1 = 1.0
        assert score == 1.0

    def test_marked_as_unmet(self):
        """Unknown type is reported as unmet in results."""
        conds = _mat_conds(
            {"type": "future_condition_v2", "params": {}, "weight": 1.0},
        )
        ctx = _market_context()
        _, results = evaluate_maturation_conditions(conds, ctx)
        assert len(results) == 1
        assert results[0].met is False
        assert results[0].condition_type == "future_condition_v2"

    def test_debug_logged_for_unknown_type(self, caplog):
        """A DEBUG log is emitted for each unknown maturation type."""
        conds = _mat_conds(
            {"type": "future_condition_v2", "params": {}, "weight": 1.0},
        )
        ctx = _market_context()

        with caplog.at_level(logging.DEBUG, logger="utils.setup_watch_evaluator"):
            evaluate_maturation_conditions(conds, ctx)

        assert any("Unknown maturation condition type" in msg for msg in caplog.messages)


# ────────────────────────────────────────────────────────────────────────────
# Test 11: Unknown invalidation type does NOT trigger rejection
# ────────────────────────────────────────────────────────────────────────────


class TestUnknownInvalidationType:
    def test_unknown_type_does_not_trigger(self):
        """An unknown invalidation type does not trigger invalidation."""
        conds = _inv_conds(
            {"type": "future_invalidation_v2", "params": {}},
        )
        ctx = _market_context()
        triggered, reason = evaluate_invalidation_conditions(conds, ctx)
        assert triggered is False
        assert reason is None

    def test_unknown_type_does_not_block_other_conditions(self):
        """Unknown type is skipped; a subsequent known type still fires."""
        conds = _inv_conds(
            {"type": "future_invalidation_v2", "params": {}},
            {"type": "exposure_conflict", "params": {}},
        )
        ctx = _market_context(symbol="AAPL", held_symbols={"AAPL"})
        triggered, reason = evaluate_invalidation_conditions(conds, ctx)
        assert triggered is True
        assert "exposure_conflict" in reason

    def test_debug_logged_for_unknown_invalidation(self, caplog):
        """A DEBUG log is emitted for unknown invalidation types."""
        conds = _inv_conds(
            {"type": "future_invalidation_v2", "params": {}},
        )
        ctx = _market_context()

        with caplog.at_level(logging.DEBUG, logger="utils.setup_watch_evaluator"):
            evaluate_invalidation_conditions(conds, ctx)

        assert any("Unknown invalidation condition type" in msg for msg in caplog.messages)


# ────────────────────────────────────────────────────────────────────────────
# Test 12: validate_draft_geometry
# ────────────────────────────────────────────────────────────────────────────


class TestValidateDraftGeometry:
    def test_valid_buy_geometry(self):
        """BUY valid: stop < entry < target."""
        geom = json.dumps({"entry": "150.00", "stop": "145.00", "target": "160.00"})
        assert validate_draft_geometry(geom, "BUY") is True

    def test_invalid_buy_inverted_stop(self):
        """BUY invalid: stop > entry (inverted)."""
        geom = json.dumps({"entry": "150.00", "stop": "155.00", "target": "160.00"})
        assert validate_draft_geometry(geom, "BUY") is False

    def test_invalid_buy_inverted_target(self):
        """BUY invalid: target < entry (inverted)."""
        geom = json.dumps({"entry": "150.00", "stop": "145.00", "target": "148.00"})
        assert validate_draft_geometry(geom, "BUY") is False

    def test_valid_short_geometry(self):
        """SHORT valid: stop > entry > target."""
        geom = json.dumps({"entry": "150.00", "stop": "155.00", "target": "140.00"})
        assert validate_draft_geometry(geom, "SHORT") is True

    def test_invalid_short_inverted_stop(self):
        """SHORT invalid: stop < entry (inverted)."""
        geom = json.dumps({"entry": "150.00", "stop": "145.00", "target": "140.00"})
        assert validate_draft_geometry(geom, "SHORT") is False

    def test_invalid_short_inverted_target(self):
        """SHORT invalid: target > entry (inverted)."""
        geom = json.dumps({"entry": "150.00", "stop": "155.00", "target": "152.00"})
        assert validate_draft_geometry(geom, "SHORT") is False

    def test_none_geometry_is_valid(self):
        """No geometry provided → treated as valid."""
        assert validate_draft_geometry(None, "BUY") is True

    def test_empty_string_geometry_is_valid(self):
        """Empty string → treated as valid."""
        assert validate_draft_geometry("", "BUY") is True

    def test_incomplete_geometry_is_valid(self):
        """Missing keys → treated as valid (not invalid)."""
        geom = json.dumps({"entry": "150.00", "stop": "145.00"})  # no target
        assert validate_draft_geometry(geom, "BUY") is True

    def test_case_insensitive_side(self):
        """Side is case-insensitive."""
        geom = json.dumps({"entry": "150.00", "stop": "145.00", "target": "160.00"})
        assert validate_draft_geometry(geom, "buy") is True
        assert validate_draft_geometry(geom, "Buy") is True


# ────────────────────────────────────────────────────────────────────────────
# Test 13: Evaluator does not mutate input condition structures
# ────────────────────────────────────────────────────────────────────────────


class TestNoMutation:
    def test_maturation_json_unchanged(self):
        """evaluate_watch does not mutate the maturation JSON string."""
        mat_json = json.dumps([
            {"type": "price_zone", "params": {"low": 140, "high": 160}, "weight": 1.0},
            {"type": "regime_aligned", "params": {"required_regime": "bullish"}, "weight": 1.0},
        ])
        inv_json = json.dumps([
            {"type": "price_breach", "params": {"level": "130", "direction": "below"}},
        ])
        ctx = _market_context()

        # Deep copy to compare after
        mat_json_before = copy.deepcopy(mat_json)
        inv_json_before = copy.deepcopy(inv_json)

        evaluate_watch(mat_json, inv_json, ctx)

        assert mat_json == mat_json_before
        assert inv_json == inv_json_before

    def test_market_context_unchanged(self):
        """evaluate_watch does not mutate the caller's market_context dict."""
        mat_json = json.dumps([
            {"type": "price_zone", "params": {"low": 140, "high": 160}, "weight": 1.0},
        ])
        inv_json = json.dumps([])
        ctx = _market_context()
        ctx_before = copy.deepcopy(ctx)

        evaluate_watch(mat_json, inv_json, ctx)

        assert ctx == ctx_before

    def test_parsed_condition_structures_unchanged(self):
        """Parsed condition dicts inside the JSON are not mutated."""
        conditions = [
            {"type": "price_zone", "params": {"low": 140, "high": 160}, "weight": 1.0},
            {"type": "key_level_proximity", "params": {"level_type": "support", "within_pct": 3.0}, "weight": 2.0},
        ]
        mat_json = json.dumps(conditions)
        conditions_before = copy.deepcopy(conditions)

        evaluate_maturation_conditions(mat_json, _market_context())

        # Verify original list was not changed (JSON string is immutable anyway,
        # but verify that parsing doesn't somehow affect things)
        assert json.loads(mat_json) == conditions_before


# ────────────────────────────────────────────────────────────────────────────
# Test 14: volume_threshold / spread_acceptable treated as unknown types in v1
# ────────────────────────────────────────────────────────────────────────────


class TestDeferredConditionTypes:
    def test_volume_threshold_treated_as_unknown_maturation(self):
        """volume_threshold is not supported — treated as unmet, excluded from denominator."""
        conds = _mat_conds(
            {"type": "volume_threshold", "params": {"min_relative_volume": 1.5}, "weight": 1.0},
            {"type": "price_zone", "params": {"low": 140, "high": 160}, "weight": 1.0},  # met
        )
        ctx = _market_context(current_price=150.0)
        score, results = evaluate_maturation_conditions(conds, ctx)
        # volume_threshold excluded from denominator → score = 1/1 = 1.0
        assert score == 1.0
        # volume_threshold marked unmet
        vol_result = next(r for r in results if r.condition_type == "volume_threshold")
        assert vol_result.met is False

    def test_spread_acceptable_treated_as_unknown_maturation(self):
        """spread_acceptable is not supported — treated as unmet, excluded from denominator."""
        conds = _mat_conds(
            {"type": "spread_acceptable", "params": {"max_spread_pct": 0.1}, "weight": 2.0},
            {"type": "regime_aligned", "params": {"required_regime": "bullish"}, "weight": 1.0},  # met
        )
        ctx = _market_context(market_regime="bullish")
        score, results = evaluate_maturation_conditions(conds, ctx)
        # spread_acceptable excluded → score = 1/1 = 1.0
        assert score == 1.0
        spread_result = next(r for r in results if r.condition_type == "spread_acceptable")
        assert spread_result.met is False

    def test_volume_threshold_as_invalidation_does_not_trigger(self):
        """volume_threshold as invalidation type does not trigger."""
        conds = _inv_conds(
            {"type": "volume_threshold", "params": {"min_relative_volume": 1.5}},
        )
        ctx = _market_context()
        triggered, reason = evaluate_invalidation_conditions(conds, ctx)
        assert triggered is False

    def test_spread_acceptable_as_invalidation_does_not_trigger(self):
        """spread_acceptable as invalidation type does not trigger."""
        conds = _inv_conds(
            {"type": "spread_acceptable", "params": {"max_spread_pct": 0.1}},
        )
        ctx = _market_context()
        triggered, reason = evaluate_invalidation_conditions(conds, ctx)
        assert triggered is False
