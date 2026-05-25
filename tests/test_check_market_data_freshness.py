"""Unit tests for _check_market_data_freshness() in setup_aware_evaluator.

Validates Requirement 7.5: If market data required for revalidation is stale
by more than 30 seconds, null, or returns an API error at the revalidation
window, the evaluator SHALL fail closed to the base force-close limit and
SHALL log the specific data unavailability reason.
"""

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from utils.setup_aware_evaluator import _check_market_data_freshness


class TestCheckMarketDataFreshness:
    """Tests for _check_market_data_freshness."""

    def test_none_current_price_returns_not_fresh(self):
        """When current_price is None, data is not fresh."""
        now = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        ts = datetime(2026, 5, 26, 14, 29, 50, tzinfo=timezone.utc)

        is_fresh, reason = _check_market_data_freshness(None, ts, now)

        assert is_fresh is False
        assert reason == "current_price is None"

    def test_none_market_data_timestamp_returns_not_fresh(self):
        """When market_data_timestamp is None, data is not fresh."""
        now = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)

        is_fresh, reason = _check_market_data_freshness(150.0, None, now)

        assert is_fresh is False
        assert reason == "market_data_timestamp is None"

    def test_fresh_data_within_threshold(self):
        """Data within 30s threshold is fresh."""
        now = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        ts = datetime(2026, 5, 26, 14, 29, 40, tzinfo=timezone.utc)  # 20s ago

        is_fresh, reason = _check_market_data_freshness(150.0, ts, now)

        assert is_fresh is True
        assert reason == "market data fresh"

    def test_stale_data_beyond_threshold(self):
        """Data older than 30s is stale."""
        now = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        ts = datetime(2026, 5, 26, 14, 29, 25, tzinfo=timezone.utc)  # 35s ago

        is_fresh, reason = _check_market_data_freshness(150.0, ts, now)

        assert is_fresh is False
        assert "stale by 35s" in reason
        assert "max 30s" in reason

    def test_exactly_at_threshold_is_fresh(self):
        """Data exactly at 30s boundary is still fresh (not stale)."""
        now = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        ts = datetime(2026, 5, 26, 14, 29, 30, tzinfo=timezone.utc)  # exactly 30s

        is_fresh, reason = _check_market_data_freshness(150.0, ts, now)

        assert is_fresh is True
        assert reason == "market data fresh"

    def test_one_second_past_threshold_is_stale(self):
        """Data 31s old exceeds the 30s threshold."""
        now = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        ts = datetime(2026, 5, 26, 14, 29, 29, tzinfo=timezone.utc)  # 31s ago

        is_fresh, reason = _check_market_data_freshness(150.0, ts, now)

        assert is_fresh is False
        assert "stale by 31s" in reason

    def test_custom_max_staleness_override(self):
        """max_staleness_seconds parameter overrides env var and default."""
        now = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        ts = datetime(2026, 5, 26, 14, 29, 10, tzinfo=timezone.utc)  # 50s ago

        # With default 30s, this would be stale
        is_fresh_default, _ = _check_market_data_freshness(150.0, ts, now)
        assert is_fresh_default is False

        # With custom 60s threshold, this is fresh
        is_fresh_custom, reason = _check_market_data_freshness(150.0, ts, now, max_staleness_seconds=60)
        assert is_fresh_custom is True
        assert reason == "market data fresh"

    def test_env_var_overrides_default(self):
        """SETUP_AWARE_MAX_MARKET_DATA_STALENESS_SECONDS env var overrides default 30."""
        now = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        ts = datetime(2026, 5, 26, 14, 29, 20, tzinfo=timezone.utc)  # 40s ago

        # With default 30s, this is stale
        is_fresh, _ = _check_market_data_freshness(150.0, ts, now)
        assert is_fresh is False

        # With env var set to 60, this is fresh
        with patch.dict(os.environ, {"SETUP_AWARE_MAX_MARKET_DATA_STALENESS_SECONDS": "60"}):
            is_fresh, reason = _check_market_data_freshness(150.0, ts, now)
            assert is_fresh is True
            assert reason == "market data fresh"

    def test_naive_now_utc_treated_as_utc(self):
        """Naive now_utc datetime is treated as UTC."""
        now = datetime(2026, 5, 26, 14, 30, 0)  # naive
        ts = datetime(2026, 5, 26, 14, 29, 50, tzinfo=timezone.utc)  # 10s ago

        is_fresh, reason = _check_market_data_freshness(150.0, ts, now)

        assert is_fresh is True
        assert reason == "market data fresh"

    def test_naive_market_data_timestamp_treated_as_utc(self):
        """Naive market_data_timestamp is treated as UTC."""
        now = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        ts = datetime(2026, 5, 26, 14, 29, 50)  # naive, 10s ago

        is_fresh, reason = _check_market_data_freshness(150.0, ts, now)

        assert is_fresh is True
        assert reason == "market data fresh"

    def test_both_naive_datetimes_treated_as_utc(self):
        """Both naive datetimes are treated as UTC and compared correctly."""
        now = datetime(2026, 5, 26, 14, 30, 0)  # naive
        ts = datetime(2026, 5, 26, 14, 29, 50)  # naive, 10s ago

        is_fresh, reason = _check_market_data_freshness(150.0, ts, now)

        assert is_fresh is True
        assert reason == "market data fresh"

    def test_very_stale_data_reports_correct_staleness(self):
        """Very stale data (5 minutes) reports correct staleness value."""
        now = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        ts = datetime(2026, 5, 26, 14, 25, 0, tzinfo=timezone.utc)  # 300s ago

        is_fresh, reason = _check_market_data_freshness(150.0, ts, now)

        assert is_fresh is False
        assert "stale by 300s" in reason
        assert "max 30s" in reason
