"""Unit tests for the FallbackRouter component.

Tests provider ordering, backoff filtering, consumer category resolution,
and fallback eligibility determination.
"""
from __future__ import annotations

import pytest

from utils.market_data_reliability.backoff import BackoffTracker
from utils.market_data_reliability.fallback import FallbackRouter, _consumer_category


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backoff_tracker():
    """A fresh BackoffTracker with default durations."""
    return BackoffTracker({"rate_limit": 60.0, "network_error": 30.0, "empty_response": 15.0})


@pytest.fixture
def fallback_matrix():
    """Default fallback matrix matching config.py defaults."""
    return {
        ("quote", "execution"): ["finnhub", "yfinance", "alpaca"],
        ("quote", "display"): ["finnhub", "yfinance", "alpaca"],
        ("quote", "monitoring"): ["finnhub", "yfinance", "alpaca"],
        ("candle", "execution"): ["finnhub", "yfinance"],
        ("candle", "display"): ["finnhub", "yfinance"],
        ("atr", "execution"): ["finnhub", "yfinance"],
        ("atr", "display"): ["finnhub", "yfinance"],
        ("volume", "execution"): ["finnhub", "yfinance"],
        ("volume", "display"): ["finnhub", "yfinance"],
        ("previous_close", "all"): ["finnhub", "yfinance", "alpaca"],
    }


@pytest.fixture
def router(fallback_matrix, backoff_tracker):
    """A FallbackRouter wired with defaults."""
    return FallbackRouter(fallback_matrix, backoff_tracker)


# ---------------------------------------------------------------------------
# Consumer category resolution
# ---------------------------------------------------------------------------


class TestConsumerCategory:
    """Tests for consumer → category resolution in fallback module."""

    def test_pm_maps_to_execution(self):
        assert _consumer_category("PM") == "execution"

    def test_risk_geometry_gate_maps_to_execution(self):
        assert _consumer_category("Risk_Geometry_Gate") == "execution"

    def test_dashboard_api_maps_to_display(self):
        assert _consumer_category("Dashboard_API") == "display"

    def test_analyst_maps_to_display(self):
        assert _consumer_category("Analyst") == "display"

    def test_reviewer_maps_to_display(self):
        assert _consumer_category("Reviewer") == "display"

    def test_ceo_output_maps_to_display(self):
        assert _consumer_category("CEO_Output") == "display"

    def test_price_monitor_maps_to_monitoring(self):
        assert _consumer_category("Price_Monitor") == "monitoring"

    def test_alert_dispatcher_maps_to_monitoring(self):
        assert _consumer_category("Alert_Dispatcher") == "monitoring"

    def test_unknown_consumer_defaults_to_execution(self):
        assert _consumer_category("Unknown_Thing") == "execution"


# ---------------------------------------------------------------------------
# resolve_providers
# ---------------------------------------------------------------------------


class TestResolveProviders:
    """Tests for FallbackRouter.resolve_providers()."""

    def test_pm_quote_returns_all_three_providers(self, router):
        result = router.resolve_providers("quote", "PM")
        assert result == ["finnhub", "yfinance", "alpaca"]

    def test_pm_candle_returns_two_providers(self, router):
        result = router.resolve_providers("candle", "PM")
        assert result == ["finnhub", "yfinance"]

    def test_dashboard_quote_returns_all_three(self, router):
        result = router.resolve_providers("quote", "Dashboard_API")
        assert result == ["finnhub", "yfinance", "alpaca"]

    def test_price_monitor_quote_returns_monitoring_providers(self, router):
        result = router.resolve_providers("quote", "Price_Monitor")
        assert result == ["finnhub", "yfinance", "alpaca"]

    def test_previous_close_uses_all_catch_all(self, router):
        """previous_close is keyed as ("previous_close", "all") in the matrix."""
        result = router.resolve_providers("previous_close", "PM")
        assert result == ["finnhub", "yfinance", "alpaca"]

    def test_unknown_data_type_returns_empty(self, router):
        result = router.resolve_providers("unknown_type", "PM")
        assert result == []

    def test_unknown_consumer_resolves_to_execution(self, router):
        """Unknown consumers default to execution category."""
        result = router.resolve_providers("quote", "SomeNewConsumer")
        assert result == ["finnhub", "yfinance", "alpaca"]

    def test_backoff_filters_provider(self, router, backoff_tracker):
        """A provider in backoff is removed from the result list."""
        backoff_tracker.record_failure("finnhub", "quote", "rate_limit")
        result = router.resolve_providers("quote", "PM")
        assert result == ["yfinance", "alpaca"]

    def test_backoff_scoped_by_data_type(self, router, backoff_tracker):
        """Backoff for quote does not affect candle resolution."""
        backoff_tracker.record_failure("finnhub", "quote", "rate_limit")
        result = router.resolve_providers("candle", "PM")
        assert result == ["finnhub", "yfinance"]

    def test_all_providers_in_backoff_returns_empty(self, router, backoff_tracker):
        """When all providers are in backoff, result is empty (fail closed)."""
        backoff_tracker.record_failure("finnhub", "quote", "rate_limit")
        backoff_tracker.record_failure("yfinance", "quote", "network_error")
        backoff_tracker.record_failure("alpaca", "quote", "empty_response")
        result = router.resolve_providers("quote", "PM")
        assert result == []

    def test_order_preserved_after_filtering(self, router, backoff_tracker):
        """Primary provider skipped, but remaining order preserved."""
        backoff_tracker.record_failure("finnhub", "candle", "network_error")
        result = router.resolve_providers("candle", "PM")
        assert result == ["yfinance"]

    def test_success_clears_backoff(self, router, backoff_tracker):
        """After record_success, provider is available again."""
        backoff_tracker.record_failure("finnhub", "quote", "rate_limit")
        backoff_tracker.record_success("finnhub", "quote")
        result = router.resolve_providers("quote", "PM")
        assert result == ["finnhub", "yfinance", "alpaca"]


# ---------------------------------------------------------------------------
# is_fallback_eligible
# ---------------------------------------------------------------------------


class TestIsFallbackEligible:
    """Tests for FallbackRouter.is_fallback_eligible()."""

    def test_quote_execution_eligible(self, router):
        """Quote for PM has 3 providers → fallback eligible."""
        assert router.is_fallback_eligible("quote", "PM") is True

    def test_candle_execution_eligible(self, router):
        """Candle for PM has 2 providers → fallback eligible."""
        assert router.is_fallback_eligible("candle", "PM") is True

    def test_previous_close_eligible_via_all(self, router):
        """previous_close uses 'all' catch-all with 3 providers."""
        assert router.is_fallback_eligible("previous_close", "Dashboard_API") is True

    def test_unknown_data_type_not_eligible(self, router):
        """No matrix entry → not eligible."""
        assert router.is_fallback_eligible("unknown_type", "PM") is False

    def test_single_provider_not_eligible(self):
        """A matrix entry with only one provider is not fallback eligible."""
        matrix = {("quote", "execution"): ["finnhub"]}
        tracker = BackoffTracker({"rate_limit": 60.0})
        router = FallbackRouter(matrix, tracker)
        assert router.is_fallback_eligible("quote", "PM") is False

    def test_eligibility_ignores_backoff_state(self, router, backoff_tracker):
        """is_fallback_eligible reflects static config, not runtime backoff."""
        backoff_tracker.record_failure("finnhub", "quote", "rate_limit")
        backoff_tracker.record_failure("yfinance", "quote", "rate_limit")
        # Still eligible per config even though 2 of 3 are in backoff
        assert router.is_fallback_eligible("quote", "PM") is True

    def test_display_consumer_fallback_eligible(self, router):
        """Dashboard has fallback for quote and candle."""
        assert router.is_fallback_eligible("quote", "Dashboard_API") is True
        assert router.is_fallback_eligible("candle", "Dashboard_API") is True

    def test_monitoring_consumer_quote_eligible(self, router):
        """Price_Monitor has quote fallback."""
        assert router.is_fallback_eligible("quote", "Price_Monitor") is True
