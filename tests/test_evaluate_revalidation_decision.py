"""Unit tests for _evaluate_revalidation_decision() internal function.

Tests the deterministic revalidation decision engine for setup-aware exit governance.
Validates Requirements 3.1, 3.2, 3.3, 3.5, 3.6, 3.7.
"""

import pytest

from utils.setup_aware_evaluator import _evaluate_revalidation_decision
from utils.setup_time_policy import get_policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trade(
    direction="LONG",
    entry_price=150.0,
    stop_price=145.0,
    target_price=160.0,
    setup_type="news_breakout",
):
    return {
        "id": 1,
        "symbol": "AMD",
        "profile": "moderate",
        "direction": direction,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "setup_type": setup_type,
        "status": "open",
        "quantity": 100,
    }


# ---------------------------------------------------------------------------
# Step 1: Price breach → close_thesis_invalidated
# ---------------------------------------------------------------------------


class TestPriceBreach:
    """Test that price breaching stop/invalidation level triggers thesis invalidation."""

    def test_long_price_at_stop(self):
        """LONG: current_price == stop_price → breached."""
        trade = _make_trade(direction="LONG", entry_price=150.0)
        policy = get_policy("news_breakout")
        criteria = {"stop_price": 145.0}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=145.0, minutes_held=95.0, criteria=criteria
        )

        assert outcome == "close_thesis_invalidated"
        assert "145.0" in reason
        assert meta["breach_type"] == "stop_price"

    def test_long_price_below_stop(self):
        """LONG: current_price < stop_price → breached."""
        trade = _make_trade(direction="LONG", entry_price=150.0)
        policy = get_policy("news_breakout")
        criteria = {"stop_price": 145.0}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=143.0, minutes_held=95.0, criteria=criteria
        )

        assert outcome == "close_thesis_invalidated"

    def test_short_price_at_stop(self):
        """SHORT: current_price == stop_price → breached."""
        trade = _make_trade(direction="SHORT", entry_price=150.0, stop_price=155.0, target_price=140.0)
        policy = get_policy("news_breakout")
        criteria = {"stop_price": 155.0}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=155.0, minutes_held=95.0, criteria=criteria
        )

        assert outcome == "close_thesis_invalidated"
        assert meta["breach_type"] == "stop_price"

    def test_short_price_above_stop(self):
        """SHORT: current_price > stop_price → breached."""
        trade = _make_trade(direction="SHORT", entry_price=150.0, stop_price=155.0, target_price=140.0)
        policy = get_policy("news_breakout")
        criteria = {"stop_price": 155.0}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=157.0, minutes_held=95.0, criteria=criteria
        )

        assert outcome == "close_thesis_invalidated"

    def test_long_breach_support_level(self):
        """LONG: price breaches support_level when no stop_price in criteria."""
        trade = _make_trade(direction="LONG", entry_price=150.0)
        policy = get_policy("news_breakout")
        criteria = {"support_level": 147.0}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=146.0, minutes_held=95.0, criteria=criteria
        )

        assert outcome == "close_thesis_invalidated"
        assert meta["breach_type"] == "support_level"

    def test_short_breach_resistance_level(self):
        """SHORT: price breaches resistance_level."""
        trade = _make_trade(direction="SHORT", entry_price=150.0, target_price=140.0)
        policy = get_policy("news_breakout")
        criteria = {"resistance_level": 153.0}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=154.0, minutes_held=95.0, criteria=criteria
        )

        assert outcome == "close_thesis_invalidated"
        assert meta["breach_type"] == "resistance_level"


# ---------------------------------------------------------------------------
# Step 2: Max extension reached → close_time_expired
# ---------------------------------------------------------------------------


class TestMaxExtension:
    """Test that exceeding max_extension_minutes triggers time expiry."""

    def test_at_max_extension(self):
        """minutes_held == max_extension_minutes → close_time_expired."""
        trade = _make_trade(direction="LONG", entry_price=150.0)
        policy = get_policy("news_breakout")  # max_extension_minutes=180
        criteria = {"stop_price": 145.0}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=155.0, minutes_held=180.0, criteria=criteria
        )

        assert outcome == "close_time_expired"
        assert "max extension" in reason
        assert meta["max_extension_minutes"] == 180

    def test_past_max_extension(self):
        """minutes_held > max_extension_minutes → close_time_expired."""
        trade = _make_trade(direction="LONG", entry_price=150.0)
        policy = get_policy("news_breakout")
        criteria = {"stop_price": 145.0}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=160.0, minutes_held=200.0, criteria=criteria
        )

        assert outcome == "close_time_expired"

    def test_breach_takes_priority_over_max_extension(self):
        """Price breach (step 1) takes priority over max extension (step 2)."""
        trade = _make_trade(direction="LONG", entry_price=150.0)
        policy = get_policy("news_breakout")
        criteria = {"stop_price": 145.0}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=144.0, minutes_held=200.0, criteria=criteria
        )

        assert outcome == "close_thesis_invalidated"


# ---------------------------------------------------------------------------
# Step 3: Positive indicators → hold_valid_until_next_window
# ---------------------------------------------------------------------------


class TestPositiveIndicators:
    """Test that positive indicators produce hold decision."""

    def test_long_price_above_entry(self):
        """LONG: price >= entry → hold."""
        trade = _make_trade(direction="LONG", entry_price=150.0, target_price=160.0)
        policy = get_policy("news_breakout")
        criteria = {"stop_price": 145.0}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=152.0, minutes_held=95.0, criteria=criteria
        )

        assert outcome == "hold_valid_until_next_window"
        assert "price_at_or_above_entry" in meta["positive_indicators"]

    def test_long_price_at_entry(self):
        """LONG: price == entry → hold."""
        trade = _make_trade(direction="LONG", entry_price=150.0, target_price=160.0)
        policy = get_policy("news_breakout")
        criteria = {"stop_price": 145.0}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=150.0, minutes_held=95.0, criteria=criteria
        )

        assert outcome == "hold_valid_until_next_window"

    def test_long_price_above_vwap(self):
        """LONG: price >= vwap_level → hold."""
        trade = _make_trade(direction="LONG", entry_price=150.0, target_price=160.0)
        policy = get_policy("news_breakout")
        criteria = {"stop_price": 145.0, "vwap_level": 148.5}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=149.0, minutes_held=95.0, criteria=criteria
        )

        assert outcome == "hold_valid_until_next_window"
        assert "price_at_or_above_vwap" in meta["positive_indicators"]

    def test_long_price_above_support(self):
        """LONG: price >= support_level → hold."""
        trade = _make_trade(direction="LONG", entry_price=150.0, target_price=160.0)
        policy = get_policy("news_breakout")
        criteria = {"stop_price": 145.0, "support_level": 147.0}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=148.0, minutes_held=95.0, criteria=criteria
        )

        assert outcome == "hold_valid_until_next_window"
        assert "price_at_or_above_support" in meta["positive_indicators"]

    def test_long_target_progress_above_25pct(self):
        """LONG: target_progress >= 25% → hold."""
        # entry=150, target=160, so 25% progress = 152.5
        trade = _make_trade(direction="LONG", entry_price=150.0, target_price=160.0)
        policy = get_policy("news_breakout")
        criteria = {"stop_price": 145.0}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=152.5, minutes_held=95.0, criteria=criteria
        )

        assert outcome == "hold_valid_until_next_window"
        assert "target_progress_above_25pct" in meta["positive_indicators"]
        assert meta["target_progress"] >= 25.0

    def test_short_price_below_entry(self):
        """SHORT: price <= entry → hold."""
        trade = _make_trade(direction="SHORT", entry_price=150.0, stop_price=155.0, target_price=140.0)
        policy = get_policy("news_breakout")
        criteria = {"stop_price": 155.0}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=148.0, minutes_held=95.0, criteria=criteria
        )

        assert outcome == "hold_valid_until_next_window"
        assert "price_at_or_below_entry" in meta["positive_indicators"]

    def test_short_target_progress_above_25pct(self):
        """SHORT: target_progress >= 25% → hold."""
        # entry=150, target=140, so 25% progress means price at 147.5
        trade = _make_trade(direction="SHORT", entry_price=150.0, stop_price=155.0, target_price=140.0)
        policy = get_policy("news_breakout")
        criteria = {"stop_price": 155.0}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=147.5, minutes_held=95.0, criteria=criteria
        )

        assert outcome == "hold_valid_until_next_window"
        assert "target_progress_above_25pct" in meta["positive_indicators"]


# ---------------------------------------------------------------------------
# Step 4: No positive indicator → close_time_expired
# ---------------------------------------------------------------------------


class TestNoPositiveIndicator:
    """Test that absence of positive indicators produces close_time_expired."""

    def test_long_price_below_entry_no_support(self):
        """LONG: price below entry, no VWAP/support, target_progress < 25% → close."""
        # entry=150, target=160, price=146 → progress = (146-150)/(160-150)*100 = -40%
        trade = _make_trade(direction="LONG", entry_price=150.0, target_price=160.0)
        policy = get_policy("news_breakout")
        criteria = {"stop_price": 145.0}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=146.0, minutes_held=95.0, criteria=criteria
        )

        assert outcome == "close_time_expired"
        assert "no positive indicator" in reason

    def test_short_price_above_entry_no_resistance(self):
        """SHORT: price above entry, no resistance, target_progress < 25% → close."""
        trade = _make_trade(direction="SHORT", entry_price=150.0, stop_price=155.0, target_price=140.0)
        policy = get_policy("news_breakout")
        criteria = {"stop_price": 155.0}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=152.0, minutes_held=95.0, criteria=criteria
        )

        assert outcome == "close_time_expired"
        assert "no positive indicator" in reason


# ---------------------------------------------------------------------------
# Division-by-zero protection
# ---------------------------------------------------------------------------


class TestDivisionByZero:
    """Test target_progress calculation handles division by zero."""

    def test_target_equals_entry_long(self):
        """LONG: target == entry → target_progress = 0, no crash."""
        trade = _make_trade(direction="LONG", entry_price=150.0, target_price=150.0)
        policy = get_policy("news_breakout")
        criteria = {"stop_price": 145.0}

        # Price above entry → hold (price_at_or_above_entry indicator)
        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=151.0, minutes_held=95.0, criteria=criteria
        )

        assert outcome == "hold_valid_until_next_window"
        assert meta["target_progress"] == 0.0

    def test_target_equals_entry_short(self):
        """SHORT: target == entry → target_progress = 0, no crash."""
        trade = _make_trade(direction="SHORT", entry_price=150.0, stop_price=155.0, target_price=150.0)
        policy = get_policy("news_breakout")
        criteria = {"stop_price": 155.0}

        # Price below entry → hold (price_at_or_below_entry indicator)
        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=149.0, minutes_held=95.0, criteria=criteria
        )

        assert outcome == "hold_valid_until_next_window"
        assert meta["target_progress"] == 0.0

    def test_no_target_price(self):
        """No target_price → target_progress = 0, no crash."""
        trade = _make_trade(direction="LONG", entry_price=150.0, target_price=None)
        policy = get_policy("news_breakout")
        criteria = {"stop_price": 145.0}

        # Price above entry → hold
        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=151.0, minutes_held=95.0, criteria=criteria
        )

        assert outcome == "hold_valid_until_next_window"
        assert meta["target_progress"] == 0.0


# ---------------------------------------------------------------------------
# Direction-aware comparisons
# ---------------------------------------------------------------------------


class TestDirectionAwareness:
    """Test that direction correctly flips comparison logic."""

    def test_long_not_breached_above_stop(self):
        """LONG: price above stop → not breached."""
        trade = _make_trade(direction="LONG", entry_price=150.0, target_price=160.0)
        policy = get_policy("news_breakout")
        criteria = {"stop_price": 145.0}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=152.0, minutes_held=95.0, criteria=criteria
        )

        assert outcome == "hold_valid_until_next_window"

    def test_short_not_breached_below_stop(self):
        """SHORT: price below stop → not breached."""
        trade = _make_trade(direction="SHORT", entry_price=150.0, stop_price=155.0, target_price=140.0)
        policy = get_policy("news_breakout")
        criteria = {"stop_price": 155.0}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=148.0, minutes_held=95.0, criteria=criteria
        )

        assert outcome == "hold_valid_until_next_window"

    def test_missing_direction_defaults_to_long(self):
        """Missing direction defaults to LONG behavior."""
        trade = _make_trade(direction="LONG", entry_price=150.0, target_price=160.0)
        trade.pop("direction", None)  # Remove direction
        policy = get_policy("news_breakout")
        criteria = {"stop_price": 145.0}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=152.0, minutes_held=95.0, criteria=criteria
        )

        assert outcome == "hold_valid_until_next_window"
        assert meta["direction"] == "LONG"
