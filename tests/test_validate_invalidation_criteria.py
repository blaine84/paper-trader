"""Unit tests for _validate_invalidation_criteria() in setup_aware_evaluator.

Tests cover:
- Valid stop price on loss side (LONG and SHORT)
- Invalid stop prices (zero, wrong side, missing)
- Structural invalidation levels (VWAP, support, resistance)
- Rejection of confidence scores, target price, or sentiment alone
- news_breakout catalyst-specific invalidation
- Edge cases: missing entry_price, non-numeric values
"""

import pytest

from utils.setup_aware_evaluator import _validate_invalidation_criteria
from utils.setup_time_policy import get_policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trade(
    direction="LONG",
    entry_price=150.0,
    stop_price=None,
    invalidators=None,
    target_price=None,
    confidence=None,
    sentiment=None,
    thesis=None,
    **kwargs,
):
    """Build a minimal trade dict for testing."""
    trade = {
        "direction": direction,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "invalidators": invalidators,
        "target_price": target_price,
        "thesis": thesis,
    }
    if confidence is not None:
        trade["confidence"] = confidence
    if sentiment is not None:
        trade["sentiment"] = sentiment
    trade.update(kwargs)
    return trade


# ---------------------------------------------------------------------------
# Tests: Valid stop price
# ---------------------------------------------------------------------------


class TestValidStopPrice:
    """Stop price validation: non-zero, on loss side of entry."""

    def test_long_valid_stop_below_entry(self):
        trade = _make_trade(direction="LONG", entry_price=150.0, stop_price=145.0)
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is True
        assert "stop_price" in criteria
        assert criteria["stop_price"] == 145.0

    def test_short_valid_stop_above_entry(self):
        trade = _make_trade(direction="SHORT", entry_price=150.0, stop_price=155.0)
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is True
        assert "stop_price" in criteria
        assert criteria["stop_price"] == 155.0

    def test_stop_price_as_string(self):
        trade = _make_trade(direction="LONG", entry_price=150.0, stop_price="145.50")
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is True
        assert criteria["stop_price"] == 145.50

    def test_stop_loss_field_used_as_fallback(self):
        trade = _make_trade(direction="LONG", entry_price=150.0, stop_price=None)
        trade["stop_loss"] = 146.0
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is True
        assert criteria["stop_price"] == 146.0


# ---------------------------------------------------------------------------
# Tests: Invalid stop price
# ---------------------------------------------------------------------------


class TestInvalidStopPrice:
    """Stop price rejection: zero, wrong side, missing."""

    def test_zero_stop_price_rejected(self):
        trade = _make_trade(direction="LONG", entry_price=150.0, stop_price=0.0)
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        # Zero stop is invalid, but might pass via structural levels
        assert "stop_price" not in criteria or criteria.get("stop_price") != 0.0

    def test_long_stop_above_entry_rejected(self):
        """Stop on profit side (above entry for long) is not valid."""
        trade = _make_trade(direction="LONG", entry_price=150.0, stop_price=155.0)
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is False
        assert "stop_price" not in criteria

    def test_short_stop_below_entry_rejected(self):
        """Stop on profit side (below entry for short) is not valid."""
        trade = _make_trade(direction="SHORT", entry_price=150.0, stop_price=145.0)
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is False
        assert "stop_price" not in criteria

    def test_none_stop_price(self):
        trade = _make_trade(direction="LONG", entry_price=150.0, stop_price=None)
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is False

    def test_non_numeric_stop_price(self):
        trade = _make_trade(direction="LONG", entry_price=150.0, stop_price="invalid")
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is False


# ---------------------------------------------------------------------------
# Tests: Structural invalidation levels
# ---------------------------------------------------------------------------


class TestStructuralLevels:
    """Structural levels (VWAP, support, resistance) as numeric prices."""

    def test_vwap_in_invalidators_dict(self):
        trade = _make_trade(
            direction="LONG",
            entry_price=150.0,
            stop_price=None,
            invalidators={"vwap": 148.5},
        )
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is True
        assert "vwap_level" in criteria
        assert criteria["vwap_level"] == 148.5

    def test_support_in_invalidators_dict(self):
        trade = _make_trade(
            direction="LONG",
            entry_price=150.0,
            stop_price=None,
            invalidators={"support": 147.0},
        )
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is True
        assert "support_level" in criteria

    def test_resistance_in_invalidators_dict(self):
        trade = _make_trade(
            direction="SHORT",
            entry_price=150.0,
            stop_price=None,
            invalidators={"resistance": 155.0},
        )
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is True
        assert "resistance_level" in criteria

    def test_structural_level_as_string_numeric(self):
        trade = _make_trade(
            direction="LONG",
            entry_price=150.0,
            stop_price=None,
            invalidators={"vwap": "148.25"},
        )
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is True
        assert criteria["vwap_level"] == 148.25

    def test_structural_level_in_list_of_dicts(self):
        trade = _make_trade(
            direction="LONG",
            entry_price=150.0,
            stop_price=None,
            invalidators=[{"vwap": 148.0}, {"support": 146.0}],
        )
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is True

    def test_zero_structural_level_rejected(self):
        trade = _make_trade(
            direction="LONG",
            entry_price=150.0,
            stop_price=None,
            invalidators={"vwap": 0},
        )
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is False

    def test_non_numeric_structural_level_rejected(self):
        trade = _make_trade(
            direction="LONG",
            entry_price=150.0,
            stop_price=None,
            invalidators={"vwap": "strong support zone"},
        )
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is False


# ---------------------------------------------------------------------------
# Tests: Rejection of insufficient criteria
# ---------------------------------------------------------------------------


class TestInsufficientCriteria:
    """Confidence scores, target price, or sentiment alone are NOT sufficient."""

    def test_confidence_score_alone_rejected(self):
        trade = _make_trade(
            direction="LONG",
            entry_price=150.0,
            stop_price=None,
            confidence=0.85,
        )
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is False
        assert "confidence" in reason.lower() or "insufficient" in reason.lower()

    def test_target_price_alone_rejected(self):
        trade = _make_trade(
            direction="LONG",
            entry_price=150.0,
            stop_price=None,
            target_price=160.0,
        )
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is False

    def test_sentiment_alone_rejected(self):
        trade = _make_trade(
            direction="LONG",
            entry_price=150.0,
            stop_price=None,
            sentiment="bullish",
        )
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is False

    def test_all_qualitative_together_rejected(self):
        trade = _make_trade(
            direction="LONG",
            entry_price=150.0,
            stop_price=None,
            target_price=160.0,
            confidence=0.9,
            sentiment="very bullish",
        )
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is False


# ---------------------------------------------------------------------------
# Tests: news_breakout catalyst-specific invalidation
# ---------------------------------------------------------------------------


class TestCatalystInvalidation:
    """news_breakout accepts catalyst-specific invalidation with deterministic trigger."""

    def test_catalyst_with_numeric_trigger(self):
        trade = _make_trade(
            direction="LONG",
            entry_price=150.0,
            stop_price=None,
            invalidators={
                "catalyst_invalidation": {"trigger": 145.0}
            },
        )
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is True
        assert criteria.get("catalyst_invalidation") is True

    def test_catalyst_with_deterministic_text_trigger(self):
        trade = _make_trade(
            direction="LONG",
            entry_price=150.0,
            stop_price=None,
            invalidators={
                "catalyst_invalidation": {"trigger": "price breaks below VWAP at 148.5"}
            },
        )
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is True

    def test_catalyst_not_accepted_for_non_news_breakout(self):
        """Catalyst invalidation is only accepted for news_breakout setup type."""
        trade = _make_trade(
            direction="LONG",
            entry_price=150.0,
            stop_price=None,
            invalidators={
                "catalyst_invalidation": {"trigger": 145.0}
            },
        )
        policy = get_policy("trend_pullback")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is False

    def test_catalyst_in_list_format(self):
        trade = _make_trade(
            direction="LONG",
            entry_price=150.0,
            stop_price=None,
            invalidators=[
                {"type": "catalyst", "trigger": 145.0}
            ],
        )
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is True


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: missing entry_price, empty trade, etc."""

    def test_missing_entry_price(self):
        trade = _make_trade(direction="LONG", entry_price=None, stop_price=145.0)
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is False
        assert "entry_price" in reason

    def test_zero_entry_price(self):
        trade = _make_trade(direction="LONG", entry_price=0.0, stop_price=145.0)
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is False

    def test_empty_invalidators(self):
        trade = _make_trade(direction="LONG", entry_price=150.0, stop_price=None, invalidators={})
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is False

    def test_empty_direction(self):
        trade = _make_trade(direction="", entry_price=150.0, stop_price=145.0)
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        # Empty direction means stop can't be validated as on loss side
        assert is_valid is False

    def test_stop_valid_takes_priority_over_structural(self):
        """When both stop and structural levels are present, stop is used."""
        trade = _make_trade(
            direction="LONG",
            entry_price=150.0,
            stop_price=145.0,
            invalidators={"vwap": 148.0},
        )
        policy = get_policy("news_breakout")
        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is True
        assert "stop_price" in criteria
        assert criteria["stop_price"] == 145.0

    def test_returns_tuple_of_three(self):
        """Always returns a 3-tuple."""
        trade = _make_trade(direction="LONG", entry_price=150.0, stop_price=145.0)
        policy = get_policy("news_breakout")
        result = _validate_invalidation_criteria(trade, policy)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)
        assert isinstance(result[2], dict)
