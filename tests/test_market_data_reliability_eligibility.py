"""Unit tests for EligibilityResolver.

Tests the eligibility resolution logic for execution, display, and monitoring
consumers against trusted, degraded, and untrusted snapshots.

Requirements: 3.5, 3.6, 6.3, 6.4, 7.1
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from utils.market_data_reliability.eligibility import EligibilityResolver
from utils.market_data_reliability.snapshot import Snapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot(trust_state: str = "trusted", symbol: str = "AAPL") -> Snapshot:
    """Build a minimal Snapshot with the given trust_state."""
    now = datetime(2025, 7, 15, 10, 0, 0)
    return Snapshot(
        symbol=symbol,
        data_type="quote",
        requested_at=now,
        provider="finnhub",
        provider_status="success",
        market_session="open",
        last_price=Decimal("150.00"),
        bid=Decimal("149.99"),
        ask=Decimal("150.01"),
        previous_close=Decimal("148.50"),
        open=Decimal("149.00"),
        high=Decimal("151.00"),
        low=Decimal("148.00"),
        volume=1000000,
        fetched_at=now,
        source_timestamp=now,
        age_seconds=2.0,
        freshness_state="fresh",
        trust_state=trust_state,
        degradation_reasons=(),
        raw_provider_latency_ms=45.0,
        fallback_primary_provider=None,
    )


# ---------------------------------------------------------------------------
# EligibilityResolver tests
# ---------------------------------------------------------------------------


class TestEligibilityResolverTrusted:
    """All consumers are eligible when trust_state == 'trusted'."""

    @pytest.fixture
    def resolver(self):
        return EligibilityResolver()

    @pytest.fixture
    def trusted_snapshot(self):
        return _make_snapshot("trusted")

    def test_execution_consumer_eligible(self, resolver, trusted_snapshot):
        result = resolver.is_eligible(trusted_snapshot, "PM", "quote")
        assert result.eligible is True
        assert result.reason_code is None
        assert result.snapshot is trusted_snapshot

    def test_risk_geometry_gate_eligible(self, resolver, trusted_snapshot):
        result = resolver.is_eligible(trusted_snapshot, "Risk_Geometry_Gate", "quote")
        assert result.eligible is True
        assert result.reason_code is None

    def test_display_consumer_eligible(self, resolver, trusted_snapshot):
        result = resolver.is_eligible(trusted_snapshot, "Dashboard_API", "quote")
        assert result.eligible is True
        assert result.reason_code is None

    def test_monitoring_consumer_eligible(self, resolver, trusted_snapshot):
        result = resolver.is_eligible(trusted_snapshot, "Price_Monitor", "quote")
        assert result.eligible is True
        assert result.reason_code is None

    def test_unknown_consumer_eligible_on_trusted(self, resolver, trusted_snapshot):
        result = resolver.is_eligible(trusted_snapshot, "UnknownConsumer", "quote")
        assert result.eligible is True
        assert result.reason_code is None


class TestEligibilityResolverUntrusted:
    """No consumer is eligible when trust_state == 'untrusted'."""

    @pytest.fixture
    def resolver(self):
        return EligibilityResolver()

    @pytest.fixture
    def untrusted_snapshot(self):
        return _make_snapshot("untrusted")

    def test_execution_consumer_ineligible(self, resolver, untrusted_snapshot):
        result = resolver.is_eligible(untrusted_snapshot, "PM", "quote")
        assert result.eligible is False
        assert result.reason_code == "data_untrusted"
        assert result.snapshot is untrusted_snapshot

    def test_display_consumer_ineligible_even_with_allow_stale(self, resolver, untrusted_snapshot):
        result = resolver.is_eligible(
            untrusted_snapshot, "Dashboard_API", "quote", allow_stale_for_display=True
        )
        assert result.eligible is False
        assert result.reason_code == "data_untrusted"

    def test_monitoring_consumer_ineligible(self, resolver, untrusted_snapshot):
        result = resolver.is_eligible(untrusted_snapshot, "Price_Monitor", "quote")
        assert result.eligible is False
        assert result.reason_code == "data_untrusted"

    def test_unknown_consumer_ineligible(self, resolver, untrusted_snapshot):
        result = resolver.is_eligible(untrusted_snapshot, "SomeRandomThing", "quote")
        assert result.eligible is False
        assert result.reason_code == "data_untrusted"


class TestEligibilityResolverDegradedExecution:
    """Execution consumers are ineligible on degraded data."""

    @pytest.fixture
    def resolver(self):
        return EligibilityResolver()

    @pytest.fixture
    def degraded_snapshot(self):
        return _make_snapshot("degraded")

    def test_pm_ineligible(self, resolver, degraded_snapshot):
        result = resolver.is_eligible(degraded_snapshot, "PM", "quote")
        assert result.eligible is False
        assert result.reason_code == "data_degraded_execution"

    def test_risk_geometry_gate_ineligible(self, resolver, degraded_snapshot):
        result = resolver.is_eligible(degraded_snapshot, "Risk_Geometry_Gate", "candle")
        assert result.eligible is False
        assert result.reason_code == "data_degraded_execution"

    def test_unknown_consumer_treated_as_execution(self, resolver, degraded_snapshot):
        """Unknown consumers fail-closed as execution."""
        result = resolver.is_eligible(degraded_snapshot, "NewConsumer", "quote")
        assert result.eligible is False
        assert result.reason_code == "data_degraded_execution"


class TestEligibilityResolverDegradedMonitoring:
    """Monitoring consumers are ineligible on degraded data."""

    @pytest.fixture
    def resolver(self):
        return EligibilityResolver()

    @pytest.fixture
    def degraded_snapshot(self):
        return _make_snapshot("degraded")

    def test_price_monitor_ineligible(self, resolver, degraded_snapshot):
        result = resolver.is_eligible(degraded_snapshot, "Price_Monitor", "quote")
        assert result.eligible is False
        assert result.reason_code == "data_degraded_monitoring"

    def test_alert_dispatcher_ineligible(self, resolver, degraded_snapshot):
        result = resolver.is_eligible(degraded_snapshot, "Alert_Dispatcher", "quote")
        assert result.eligible is False
        assert result.reason_code == "data_degraded_monitoring"


class TestEligibilityResolverDegradedDisplay:
    """Display consumers: eligible with degraded data only if allow_stale_for_display."""

    @pytest.fixture
    def resolver(self):
        return EligibilityResolver()

    @pytest.fixture
    def degraded_snapshot(self):
        return _make_snapshot("degraded")

    def test_dashboard_eligible_with_allow_stale(self, resolver, degraded_snapshot):
        result = resolver.is_eligible(
            degraded_snapshot, "Dashboard_API", "quote", allow_stale_for_display=True
        )
        assert result.eligible is True
        assert result.reason_code is None

    def test_dashboard_ineligible_without_allow_stale(self, resolver, degraded_snapshot):
        result = resolver.is_eligible(
            degraded_snapshot, "Dashboard_API", "quote", allow_stale_for_display=False
        )
        assert result.eligible is False
        assert result.reason_code == "data_stale_display"

    def test_analyst_eligible_with_allow_stale(self, resolver, degraded_snapshot):
        result = resolver.is_eligible(
            degraded_snapshot, "Analyst", "quote", allow_stale_for_display=True
        )
        assert result.eligible is True
        assert result.reason_code is None

    def test_reviewer_ineligible_without_allow_stale(self, resolver, degraded_snapshot):
        result = resolver.is_eligible(
            degraded_snapshot, "Reviewer", "candle", allow_stale_for_display=False
        )
        assert result.eligible is False
        assert result.reason_code == "data_stale_display"

    def test_ceo_output_eligible_with_allow_stale(self, resolver, degraded_snapshot):
        result = resolver.is_eligible(
            degraded_snapshot, "CEO_Output", "quote", allow_stale_for_display=True
        )
        assert result.eligible is True
        assert result.reason_code is None


class TestEligibilityResolverSnapshotPassthrough:
    """EligibilityResult always carries the evaluated snapshot."""

    @pytest.fixture
    def resolver(self):
        return EligibilityResolver()

    def test_snapshot_present_on_eligible(self, resolver):
        snap = _make_snapshot("trusted")
        result = resolver.is_eligible(snap, "PM", "quote")
        assert result.snapshot is snap

    def test_snapshot_present_on_ineligible(self, resolver):
        snap = _make_snapshot("untrusted")
        result = resolver.is_eligible(snap, "PM", "quote")
        assert result.snapshot is snap
