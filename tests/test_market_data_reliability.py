"""Unit tests for Market Data Reliability Layer components.

Tests specific examples and edge cases for the reliability layer modules.
"""
from __future__ import annotations

import pytest

from utils.market_data_reliability.config import ReliabilityConfig, _DEFAULT_FRESHNESS_THRESHOLDS
from utils.market_data_reliability.freshness import FreshnessClassifier, resolve_consumer_category
from utils.market_data_reliability.snapshot import FreshnessThreshold


# ---------------------------------------------------------------------------
# FreshnessClassifier unit tests
# ---------------------------------------------------------------------------


class TestResolveConsumerCategory:
    """Tests for consumer → category resolution."""

    def test_pm_maps_to_execution(self):
        assert resolve_consumer_category("PM") == "execution"

    def test_risk_geometry_gate_maps_to_execution(self):
        assert resolve_consumer_category("Risk_Geometry_Gate") == "execution"

    def test_dashboard_api_maps_to_display(self):
        assert resolve_consumer_category("Dashboard_API") == "display"

    def test_analyst_maps_to_display(self):
        assert resolve_consumer_category("Analyst") == "display"

    def test_reviewer_maps_to_display(self):
        assert resolve_consumer_category("Reviewer") == "display"

    def test_ceo_output_maps_to_display(self):
        assert resolve_consumer_category("CEO_Output") == "display"

    def test_price_monitor_maps_to_execution(self):
        """Monitoring consumers use execution thresholds (strict)."""
        assert resolve_consumer_category("Price_Monitor") == "execution"

    def test_alert_dispatcher_maps_to_execution(self):
        """Monitoring consumers use execution thresholds (strict)."""
        assert resolve_consumer_category("Alert_Dispatcher") == "execution"

    def test_unknown_consumer_defaults_to_execution(self):
        """Unknown consumers fail-closed with execution thresholds."""
        assert resolve_consumer_category("Unknown_Consumer") == "execution"


class TestFreshnessClassifier:
    """Tests for FreshnessClassifier.classify()."""

    @pytest.fixture
    def classifier(self):
        """Classifier with default thresholds."""
        return FreshnessClassifier(_DEFAULT_FRESHNESS_THRESHOLDS)

    # --- Defensive: negative age ---

    def test_negative_age_returns_unavailable(self, classifier):
        result = classifier.classify(-1.0, "quote", "PM", "open")
        assert result == "unavailable"

    def test_negative_age_ignores_market_session(self, classifier):
        """Negative age takes priority over market_closed."""
        result = classifier.classify(-5.0, "quote", "PM", "closed")
        assert result == "unavailable"

    # --- Market closed ---

    def test_market_closed_returns_market_closed(self, classifier):
        result = classifier.classify(10.0, "quote", "PM", "closed")
        assert result == "market_closed"

    def test_market_closed_any_data_type(self, classifier):
        result = classifier.classify(500.0, "candle", "Dashboard_API", "closed")
        assert result == "market_closed"

    # --- Quote execution thresholds (fresh < 30, aging < 120, stale >= 120) ---

    def test_quote_execution_fresh(self, classifier):
        result = classifier.classify(15.0, "quote", "PM", "open")
        assert result == "fresh"

    def test_quote_execution_fresh_boundary(self, classifier):
        """At exactly fresh_threshold, should be aging (not fresh)."""
        result = classifier.classify(30.0, "quote", "PM", "open")
        assert result == "aging"

    def test_quote_execution_aging(self, classifier):
        result = classifier.classify(60.0, "quote", "PM", "open")
        assert result == "aging"

    def test_quote_execution_aging_boundary(self, classifier):
        """At exactly aging_threshold, should be stale."""
        result = classifier.classify(120.0, "quote", "PM", "open")
        assert result == "stale"

    def test_quote_execution_stale(self, classifier):
        result = classifier.classify(200.0, "quote", "PM", "open")
        assert result == "stale"

    # --- Quote display thresholds (fresh < 60, aging < 300, stale >= 300) ---

    def test_quote_display_fresh(self, classifier):
        result = classifier.classify(45.0, "quote", "Dashboard_API", "open")
        assert result == "fresh"

    def test_quote_display_aging(self, classifier):
        result = classifier.classify(150.0, "quote", "Dashboard_API", "open")
        assert result == "aging"

    def test_quote_display_stale(self, classifier):
        result = classifier.classify(400.0, "quote", "Dashboard_API", "open")
        assert result == "stale"

    # --- ATR execution thresholds (fresh < 300, aging < 900, stale >= 900) ---

    def test_atr_execution_fresh(self, classifier):
        result = classifier.classify(100.0, "atr", "PM", "open")
        assert result == "fresh"

    def test_atr_execution_aging(self, classifier):
        result = classifier.classify(500.0, "atr", "Risk_Geometry_Gate", "open")
        assert result == "aging"

    def test_atr_execution_stale(self, classifier):
        result = classifier.classify(1000.0, "atr", "PM", "open")
        assert result == "stale"

    # --- previous_close uses "all" category ---

    def test_previous_close_fresh(self, classifier):
        """previous_close uses wildcard 'all' category for any consumer."""
        result = classifier.classify(1800.0, "previous_close", "PM", "open")
        assert result == "fresh"

    def test_previous_close_aging(self, classifier):
        result = classifier.classify(5000.0, "previous_close", "Dashboard_API", "open")
        assert result == "aging"

    def test_previous_close_stale(self, classifier):
        result = classifier.classify(8000.0, "previous_close", "PM", "open")
        assert result == "stale"

    # --- Monitoring consumers use execution thresholds ---

    def test_price_monitor_uses_execution_thresholds(self, classifier):
        """Price_Monitor gets strict execution thresholds."""
        # 35s for quote should be aging with execution thresholds (30s fresh boundary)
        result = classifier.classify(35.0, "quote", "Price_Monitor", "open")
        assert result == "aging"

    def test_alert_dispatcher_uses_execution_thresholds(self, classifier):
        """Alert_Dispatcher gets strict execution thresholds."""
        result = classifier.classify(25.0, "quote", "Alert_Dispatcher", "open")
        assert result == "fresh"

    # --- Market session variants (non-closed) ---

    def test_pre_market_classifies_normally(self, classifier):
        result = classifier.classify(10.0, "quote", "PM", "pre_market")
        assert result == "fresh"

    def test_after_hours_classifies_normally(self, classifier):
        result = classifier.classify(200.0, "quote", "PM", "after_hours")
        assert result == "stale"

    # --- Zero age ---

    def test_zero_age_is_fresh(self, classifier):
        result = classifier.classify(0.0, "quote", "PM", "open")
        assert result == "fresh"

    # --- Unknown data_type with no threshold ---

    def test_unknown_data_type_returns_stale(self, classifier):
        """Unknown data types fail-closed as stale."""
        result = classifier.classify(5.0, "unknown_type", "PM", "open")
        assert result == "stale"

    # --- Custom thresholds ---

    def test_custom_thresholds(self):
        """Classifier works with custom threshold values."""
        custom = {
            ("quote", "execution"): FreshnessThreshold(fresh_threshold=10.0, aging_threshold=50.0),
        }
        classifier = FreshnessClassifier(custom)
        assert classifier.classify(5.0, "quote", "PM", "open") == "fresh"
        assert classifier.classify(10.0, "quote", "PM", "open") == "aging"
        assert classifier.classify(30.0, "quote", "PM", "open") == "aging"
        assert classifier.classify(50.0, "quote", "PM", "open") == "stale"
