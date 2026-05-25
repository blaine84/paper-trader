"""Unit tests for _compute_next_revalidation_boundary() and _is_at_revalidation_boundary().

Validates Requirements 1.12, 1.13:
- 1.12: Extension-eligible trades revalidate at force_close_minutes, then at each
  subsequent revalidation_interval_minutes until max_extension_minutes.
- 1.13: force_close_minutes is NOT a hard close for extension-eligible trades;
  it marks the first mandatory revalidation-or-close boundary after revalidate_minutes.

The boundary sequence for extension-eligible trades is:
  [revalidate_minutes, force_close_minutes, force_close + interval, ..., max_extension]
"""

from datetime import datetime, timezone

from utils.setup_aware_evaluator import (
    _compute_next_revalidation_boundary,
    _is_at_revalidation_boundary,
)
from utils.setup_time_policy import SetupTimePolicy, get_policy


# ---------------------------------------------------------------------------
# Fixtures / Helpers
# ---------------------------------------------------------------------------

ENTRY_TIME = datetime(2026, 5, 26, 10, 0, 0, tzinfo=timezone.utc)

# news_breakout: revalidate=90, force_close=120, interval=30, max=180
NEWS_BREAKOUT_POLICY = get_policy("news_breakout")

# trend_pullback: revalidate=120, force_close=150, interval=30, max=180
TREND_PULLBACK_POLICY = get_policy("trend_pullback")

# momentum_fade: not extension-eligible
MOMENTUM_FADE_POLICY = get_policy("momentum_fade")

# Default/unknown: not extension-eligible
DEFAULT_POLICY = get_policy("unknown_setup")


# ---------------------------------------------------------------------------
# Tests: _compute_next_revalidation_boundary
# ---------------------------------------------------------------------------


class TestComputeNextRevalidationBoundary:
    """Tests for _compute_next_revalidation_boundary."""

    # --- news_breakout boundary sequence: [90, 120, 150, 180] ---

    def test_news_breakout_before_first_boundary(self):
        """minutes_held=85 → next boundary = 90."""
        result = _compute_next_revalidation_boundary(ENTRY_TIME, NEWS_BREAKOUT_POLICY, 85.0)
        assert result == 90

    def test_news_breakout_at_first_boundary(self):
        """minutes_held=90 → next boundary = 120 (force_close)."""
        result = _compute_next_revalidation_boundary(ENTRY_TIME, NEWS_BREAKOUT_POLICY, 90.0)
        assert result == 120

    def test_news_breakout_between_first_and_second(self):
        """minutes_held=91 → next boundary = 120."""
        result = _compute_next_revalidation_boundary(ENTRY_TIME, NEWS_BREAKOUT_POLICY, 91.0)
        assert result == 120

    def test_news_breakout_at_second_boundary(self):
        """minutes_held=120 → next boundary = 150."""
        result = _compute_next_revalidation_boundary(ENTRY_TIME, NEWS_BREAKOUT_POLICY, 120.0)
        assert result == 150

    def test_news_breakout_between_second_and_third(self):
        """minutes_held=121 → next boundary = 150."""
        result = _compute_next_revalidation_boundary(ENTRY_TIME, NEWS_BREAKOUT_POLICY, 121.0)
        assert result == 150

    def test_news_breakout_at_third_boundary(self):
        """minutes_held=150 → next boundary = 180 (max extension)."""
        result = _compute_next_revalidation_boundary(ENTRY_TIME, NEWS_BREAKOUT_POLICY, 150.0)
        assert result == 180

    def test_news_breakout_between_third_and_max(self):
        """minutes_held=151 → next boundary = 180."""
        result = _compute_next_revalidation_boundary(ENTRY_TIME, NEWS_BREAKOUT_POLICY, 151.0)
        assert result == 180

    def test_news_breakout_at_max_extension(self):
        """minutes_held=180 → None (at max, no further boundaries)."""
        result = _compute_next_revalidation_boundary(ENTRY_TIME, NEWS_BREAKOUT_POLICY, 180.0)
        assert result is None

    def test_news_breakout_past_max_extension(self):
        """minutes_held=181 → None (past max)."""
        result = _compute_next_revalidation_boundary(ENTRY_TIME, NEWS_BREAKOUT_POLICY, 181.0)
        assert result is None

    # --- trend_pullback boundary sequence: [120, 150, 180] ---

    def test_trend_pullback_before_first_boundary(self):
        """minutes_held=100 → next boundary = 120."""
        result = _compute_next_revalidation_boundary(ENTRY_TIME, TREND_PULLBACK_POLICY, 100.0)
        assert result == 120

    def test_trend_pullback_at_first_boundary(self):
        """minutes_held=120 → next boundary = 150."""
        result = _compute_next_revalidation_boundary(ENTRY_TIME, TREND_PULLBACK_POLICY, 120.0)
        assert result == 150

    def test_trend_pullback_at_second_boundary(self):
        """minutes_held=150 → next boundary = 180."""
        result = _compute_next_revalidation_boundary(ENTRY_TIME, TREND_PULLBACK_POLICY, 150.0)
        assert result == 180

    def test_trend_pullback_past_max(self):
        """minutes_held=180 → None."""
        result = _compute_next_revalidation_boundary(ENTRY_TIME, TREND_PULLBACK_POLICY, 180.0)
        assert result is None

    # --- Non-extension-eligible setups ---

    def test_non_extension_eligible_returns_none(self):
        """momentum_fade (not extension-eligible) always returns None."""
        result = _compute_next_revalidation_boundary(ENTRY_TIME, MOMENTUM_FADE_POLICY, 50.0)
        assert result is None

    def test_default_policy_returns_none(self):
        """Unknown/default policy (not extension-eligible) returns None."""
        result = _compute_next_revalidation_boundary(ENTRY_TIME, DEFAULT_POLICY, 50.0)
        assert result is None

    # --- Edge cases ---

    def test_minutes_held_zero(self):
        """minutes_held=0 → first boundary (90 for news_breakout)."""
        result = _compute_next_revalidation_boundary(ENTRY_TIME, NEWS_BREAKOUT_POLICY, 0.0)
        assert result == 90

    def test_minutes_held_exactly_at_boundary_returns_next(self):
        """When exactly at a boundary, returns the NEXT one (strictly greater)."""
        # At 90, next is 120
        assert _compute_next_revalidation_boundary(ENTRY_TIME, NEWS_BREAKOUT_POLICY, 90.0) == 120
        # At 120, next is 150
        assert _compute_next_revalidation_boundary(ENTRY_TIME, NEWS_BREAKOUT_POLICY, 120.0) == 150
        # At 150, next is 180
        assert _compute_next_revalidation_boundary(ENTRY_TIME, NEWS_BREAKOUT_POLICY, 150.0) == 180

    def test_fractional_minutes_held(self):
        """Fractional minutes_held works correctly."""
        # 89.9 → next is 90
        result = _compute_next_revalidation_boundary(ENTRY_TIME, NEWS_BREAKOUT_POLICY, 89.9)
        assert result == 90

        # 90.1 → next is 120
        result = _compute_next_revalidation_boundary(ENTRY_TIME, NEWS_BREAKOUT_POLICY, 90.1)
        assert result == 120


# ---------------------------------------------------------------------------
# Tests: _is_at_revalidation_boundary
# ---------------------------------------------------------------------------


class TestIsAtRevalidationBoundary:
    """Tests for _is_at_revalidation_boundary."""

    def test_below_first_boundary_returns_false(self):
        """minutes_held < revalidate_minutes → not at boundary."""
        result = _is_at_revalidation_boundary(85.0, ENTRY_TIME, NEWS_BREAKOUT_POLICY)
        assert result is False

    def test_at_first_boundary_returns_true(self):
        """minutes_held == revalidate_minutes → at boundary."""
        result = _is_at_revalidation_boundary(90.0, ENTRY_TIME, NEWS_BREAKOUT_POLICY)
        assert result is True

    def test_past_first_boundary_returns_true(self):
        """minutes_held > revalidate_minutes → at boundary."""
        result = _is_at_revalidation_boundary(91.0, ENTRY_TIME, NEWS_BREAKOUT_POLICY)
        assert result is True

    def test_at_force_close_boundary_returns_true(self):
        """minutes_held at force_close_minutes → at boundary."""
        result = _is_at_revalidation_boundary(120.0, ENTRY_TIME, NEWS_BREAKOUT_POLICY)
        assert result is True

    def test_past_force_close_returns_true(self):
        """minutes_held past force_close → still at boundary (extension-eligible)."""
        result = _is_at_revalidation_boundary(150.0, ENTRY_TIME, NEWS_BREAKOUT_POLICY)
        assert result is True

    def test_at_max_extension_returns_true(self):
        """minutes_held at max_extension → at boundary."""
        result = _is_at_revalidation_boundary(180.0, ENTRY_TIME, NEWS_BREAKOUT_POLICY)
        assert result is True

    def test_non_extension_eligible_returns_false(self):
        """Non-extension-eligible setup always returns False."""
        result = _is_at_revalidation_boundary(100.0, ENTRY_TIME, MOMENTUM_FADE_POLICY)
        assert result is False

    def test_default_policy_returns_false(self):
        """Default/unknown policy always returns False."""
        result = _is_at_revalidation_boundary(100.0, ENTRY_TIME, DEFAULT_POLICY)
        assert result is False

    def test_trend_pullback_below_revalidate_returns_false(self):
        """trend_pullback with minutes_held < 120 → not at boundary."""
        result = _is_at_revalidation_boundary(100.0, ENTRY_TIME, TREND_PULLBACK_POLICY)
        assert result is False

    def test_trend_pullback_at_revalidate_returns_true(self):
        """trend_pullback with minutes_held == 120 → at boundary."""
        result = _is_at_revalidation_boundary(120.0, ENTRY_TIME, TREND_PULLBACK_POLICY)
        assert result is True

    def test_zero_minutes_held_returns_false(self):
        """minutes_held=0 → not at boundary."""
        result = _is_at_revalidation_boundary(0.0, ENTRY_TIME, NEWS_BREAKOUT_POLICY)
        assert result is False

    def test_fractional_minutes_just_below_boundary(self):
        """89.9 minutes → not at boundary (below 90)."""
        result = _is_at_revalidation_boundary(89.9, ENTRY_TIME, NEWS_BREAKOUT_POLICY)
        assert result is False

    def test_fractional_minutes_just_above_boundary(self):
        """90.1 minutes → at boundary."""
        result = _is_at_revalidation_boundary(90.1, ENTRY_TIME, NEWS_BREAKOUT_POLICY)
        assert result is True
