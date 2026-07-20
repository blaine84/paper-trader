"""Unit tests for Dashboard API integration with the Market Data Reliability Layer.

Tests that freshness_state, trust_state, freshness_label, and is_actionable fields
are correctly added to API response payloads, and that production safety rules
(feature flag guard, fail-open) are respected.

Requirements: 9.1, 9.2, 9.3, 9.4
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import patch

import pytest

from utils.market_data_reliability.dashboard_integration import (
    enrich_market_data_response,
    get_freshness_label,
)
from utils.market_data_reliability.snapshot import Snapshot
from web.app import (
    _add_market_data_reliability_fields,
    _build_dashboard_market_data_snapshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot(
    freshness_state: str = "fresh",
    trust_state: str = "trusted",
    symbol: str = "AAPL",
) -> Snapshot:
    """Create a test Snapshot with the given freshness/trust states."""
    now = datetime.now(timezone.utc)
    return Snapshot(
        symbol=symbol,
        data_type="quote",
        requested_at=now,
        provider="finnhub",
        provider_status="success",
        market_session="open",
        last_price=Decimal("187.43"),
        bid=Decimal("187.42"),
        ask=Decimal("187.44"),
        previous_close=Decimal("186.00"),
        open=Decimal("186.50"),
        high=Decimal("188.00"),
        low=Decimal("185.50"),
        volume=1000000,
        fetched_at=now,
        source_timestamp=now,
        age_seconds=5.0,
        freshness_state=freshness_state,
        trust_state=trust_state,
        degradation_reasons=() if trust_state == "trusted" else ("stale_source_timestamp",),
        raw_provider_latency_ms=45.0,
        fallback_primary_provider=None,
    )


# ---------------------------------------------------------------------------
# get_freshness_label tests
# ---------------------------------------------------------------------------


class TestGetFreshnessLabel:
    """Tests for get_freshness_label function."""

    def test_fresh_trusted_returns_none(self):
        assert get_freshness_label("fresh", "trusted") is None

    def test_aging_trusted_returns_delayed_label(self):
        result = get_freshness_label("aging", "trusted")
        assert result == "Data is slightly delayed"

    def test_aging_degraded_returns_delayed_label(self):
        result = get_freshness_label("aging", "degraded")
        assert result == "Data is slightly delayed"

    def test_stale_degraded_returns_stale_warning(self):
        result = get_freshness_label("stale", "degraded")
        assert result == "Data may be stale - do not use for trading decisions"

    def test_stale_untrusted_returns_unreliable_warning(self):
        result = get_freshness_label("stale", "untrusted")
        assert result == "Data is unreliable - do not use for trading decisions"

    def test_unavailable_untrusted_returns_unavailable_label(self):
        result = get_freshness_label("unavailable", "untrusted")
        assert result == "Data unavailable from all providers"

    def test_market_closed_trusted_returns_closed_label(self):
        result = get_freshness_label("market_closed", "trusted")
        assert result == "Market closed - showing last session data"

    def test_market_closed_degraded_returns_closed_label(self):
        result = get_freshness_label("market_closed", "degraded")
        assert result == "Market closed - showing last session data"

    def test_unknown_combination_returns_none(self):
        """Unknown freshness/trust combinations return None (fail-open)."""
        result = get_freshness_label("unknown_state", "unknown_trust")
        assert result is None


# ---------------------------------------------------------------------------
# enrich_market_data_response tests — feature flag guard
# ---------------------------------------------------------------------------


class TestEnrichFeatureFlag:
    """Tests that enrichment respects MARKET_DATA_RELIABILITY_MODE."""

    @patch("utils.market_data_reliability.dashboard_integration.MARKET_DATA_RELIABILITY_MODE", "disabled")
    def test_disabled_mode_returns_data_unchanged(self):
        """When mode is disabled, no enrichment fields are added."""
        data = {"symbol": "AAPL", "price": 187.43}
        snapshot = _make_snapshot()

        result = enrich_market_data_response(data, snapshot)

        assert result is data
        assert "freshness_state" not in result
        assert "trust_state" not in result
        assert "freshness_label" not in result
        assert "is_actionable" not in result

    @patch("utils.market_data_reliability.dashboard_integration.MARKET_DATA_RELIABILITY_MODE", "observe")
    def test_observe_mode_enriches_data(self):
        """Observe mode adds enrichment fields."""
        data = {"symbol": "AAPL", "price": 187.43}
        snapshot = _make_snapshot()

        result = enrich_market_data_response(data, snapshot)

        assert "freshness_state" in result
        assert "trust_state" in result

    @patch("utils.market_data_reliability.dashboard_integration.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_enforcing_mode_enriches_data(self):
        """Enforcing mode adds enrichment fields."""
        data = {"symbol": "AAPL", "price": 187.43}
        snapshot = _make_snapshot()

        result = enrich_market_data_response(data, snapshot)

        assert "freshness_state" in result
        assert "trust_state" in result


# ---------------------------------------------------------------------------
# enrich_market_data_response tests — enrichment behavior
# ---------------------------------------------------------------------------


class TestEnrichBehavior:
    """Tests for enrich_market_data_response enrichment logic."""

    @patch("utils.market_data_reliability.dashboard_integration.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_fresh_trusted_is_actionable(self):
        """Fresh + trusted data is marked as actionable."""
        data = {"symbol": "AAPL", "price": 187.43}
        snapshot = _make_snapshot(freshness_state="fresh", trust_state="trusted")

        result = enrich_market_data_response(data, snapshot)

        assert result["freshness_state"] == "fresh"
        assert result["trust_state"] == "trusted"
        assert result["freshness_label"] is None
        assert result["is_actionable"] is True

    @patch("utils.market_data_reliability.dashboard_integration.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_stale_degraded_not_actionable(self):
        """Stale + degraded data is marked as not actionable with a label."""
        data = {"symbol": "AAPL", "price": 187.43}
        snapshot = _make_snapshot(freshness_state="stale", trust_state="degraded")

        result = enrich_market_data_response(data, snapshot)

        assert result["freshness_state"] == "stale"
        assert result["trust_state"] == "degraded"
        assert result["freshness_label"] == "Data may be stale - do not use for trading decisions"
        assert result["is_actionable"] is False

    @patch("utils.market_data_reliability.dashboard_integration.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_stale_untrusted_not_actionable(self):
        """Stale + untrusted data is marked as not actionable with unreliable label."""
        data = {"symbol": "SPY", "price": 0}
        snapshot = _make_snapshot(freshness_state="stale", trust_state="untrusted")

        result = enrich_market_data_response(data, snapshot)

        assert result["freshness_state"] == "stale"
        assert result["trust_state"] == "untrusted"
        assert result["freshness_label"] == "Data is unreliable - do not use for trading decisions"
        assert result["is_actionable"] is False

    @patch("utils.market_data_reliability.dashboard_integration.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_unavailable_untrusted_not_actionable(self):
        """Unavailable + untrusted data is marked as not actionable."""
        data = {"symbol": "SPY", "price": 0}
        snapshot = _make_snapshot(freshness_state="unavailable", trust_state="untrusted")

        result = enrich_market_data_response(data, snapshot)

        assert result["freshness_state"] == "unavailable"
        assert result["trust_state"] == "untrusted"
        assert result["freshness_label"] == "Data unavailable from all providers"
        assert result["is_actionable"] is False

    @patch("utils.market_data_reliability.dashboard_integration.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_market_closed_not_actionable(self):
        """Market closed data is not actionable."""
        data = {"symbol": "AAPL", "price": 187.43}
        snapshot = _make_snapshot(freshness_state="market_closed", trust_state="trusted")

        result = enrich_market_data_response(data, snapshot)

        assert result["freshness_state"] == "market_closed"
        assert result["trust_state"] == "trusted"
        assert result["freshness_label"] == "Market closed - showing last session data"
        assert result["is_actionable"] is False

    @patch("utils.market_data_reliability.dashboard_integration.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_aging_trusted_is_actionable(self):
        """Aging + trusted data is still actionable (trusted data with slight delay)."""
        data = {"symbol": "AAPL", "price": 187.43}
        snapshot = _make_snapshot(freshness_state="aging", trust_state="trusted")

        result = enrich_market_data_response(data, snapshot)

        assert result["freshness_state"] == "aging"
        assert result["trust_state"] == "trusted"
        assert result["freshness_label"] == "Data is slightly delayed"
        assert result["is_actionable"] is True

    @patch("utils.market_data_reliability.dashboard_integration.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_existing_fields_preserved(self):
        """Enrichment does not remove or modify existing fields."""
        data = {
            "symbol": "AAPL",
            "price": 187.43,
            "change_pct": 0.76,
            "signal": "bullish",
        }
        snapshot = _make_snapshot(freshness_state="fresh", trust_state="trusted")

        result = enrich_market_data_response(data, snapshot)

        assert result["symbol"] == "AAPL"
        assert result["price"] == 187.43
        assert result["change_pct"] == 0.76
        assert result["signal"] == "bullish"

    @patch("utils.market_data_reliability.dashboard_integration.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_machine_readable_state_in_payload(self):
        """Degradation state is machine-readable (freshness_state and trust_state are strings)."""
        data = {"symbol": "AAPL"}
        snapshot = _make_snapshot(freshness_state="stale", trust_state="untrusted")

        result = enrich_market_data_response(data, snapshot)

        # Machine-readable: string values that can be parsed by API clients
        assert isinstance(result["freshness_state"], str)
        assert isinstance(result["trust_state"], str)
        assert isinstance(result["is_actionable"], bool)


# ---------------------------------------------------------------------------
# enrich_market_data_response tests — fail-open behavior
# ---------------------------------------------------------------------------


class TestEnrichFailOpen:
    """Tests that enrichment fails open (returns original data on error)."""

    @patch("utils.market_data_reliability.dashboard_integration.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_none_snapshot_returns_unchanged(self):
        """When snapshot is None, data is returned unchanged."""
        data = {"symbol": "AAPL", "price": 187.43}

        result = enrich_market_data_response(data, None)

        assert result is data
        assert "freshness_state" not in result

    @patch("utils.market_data_reliability.dashboard_integration.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_broken_snapshot_returns_unchanged(self):
        """When snapshot attribute access raises, data is returned unchanged."""
        # Create an object that raises on attribute access
        class BrokenSnapshot:
            @property
            def freshness_state(self):
                raise RuntimeError("broken")

            @property
            def symbol(self):
                return "BROKEN"

        data = {"symbol": "AAPL", "price": 187.43}

        # Should not raise — fail-open
        result = enrich_market_data_response(data, BrokenSnapshot())

        assert result is data
        assert "freshness_state" not in result

    @patch("utils.market_data_reliability.dashboard_integration.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_returns_same_reference(self):
        """Enrichment mutates in place and returns the same dict reference."""
        data = {"symbol": "AAPL"}
        snapshot = _make_snapshot()

        result = enrich_market_data_response(data, snapshot)

        assert result is data


# ---------------------------------------------------------------------------
# /api/data row wiring helpers
# ---------------------------------------------------------------------------


class TestDashboardApiRowWiring:
    """Tests for dashboard row reliability wiring in web.app."""

    def test_observe_mode_adds_market_data_fields_without_overwriting_catalyst_label(
        self, monkeypatch
    ):
        """Dashboard rows expose MDR state while preserving catalyst freshness label."""
        monkeypatch.setenv("MARKET_DATA_RELIABILITY_MODE", "observe")
        now = datetime.now(timezone.utc)
        quote = {
            "price": 187.43,
            "change_pct": 0.76,
            "_provider": "finnhub",
            "_timestamp": now.isoformat(),
            "_open": 186.50,
            "_high": 188.00,
            "_low": 185.50,
            "_prev_close": 186.00,
        }
        row = {
            "symbol": "AAPL",
            "price": 187.43,
            "freshness_label": "Fresh catalyst",
        }

        snapshot = _build_dashboard_market_data_snapshot(
            "AAPL", quote, market_open=True, now=now
        )
        result = _add_market_data_reliability_fields(row, snapshot)

        assert result["freshness_label"] == "Fresh catalyst"
        assert result["freshness_state"] == "fresh"
        assert result["trust_state"] == "trusted"
        assert result["market_data_freshness_label"] is None
        assert result["is_actionable"] is True

    def test_disabled_mode_leaves_dashboard_row_unenriched(self, monkeypatch):
        """Disabled MDR mode does not add dashboard reliability fields."""
        monkeypatch.setenv("MARKET_DATA_RELIABILITY_MODE", "disabled")
        now = datetime.now(timezone.utc)
        quote = {
            "price": 187.43,
            "change_pct": 0.76,
            "_provider": "finnhub",
            "_timestamp": now.isoformat(),
        }
        row = {"symbol": "AAPL", "freshness_label": "Fresh catalyst"}

        snapshot = _build_dashboard_market_data_snapshot(
            "AAPL", quote, market_open=True, now=now
        )
        result = _add_market_data_reliability_fields(row, snapshot)

        assert result == {"symbol": "AAPL", "freshness_label": "Fresh catalyst"}
