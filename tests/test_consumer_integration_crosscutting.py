"""Cross-cutting unit tests for all three consumer integration points.

Verifies end-to-end flow across PM pipeline, Price Monitor, and Dashboard API
integrations using a shared ReliabilityLayer instance. Confirms that:
- check_market_data_readiness → ReliabilityLayer → check_candidate_readiness works
- TelemetryCollector records are populated after integration calls
- All three integrations share the same layer and observe consistent state
- Feature flag modes apply uniformly across all consumers

Requirements: 7.1-7.5, 8.1-8.5, 9.1-9.4
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from utils.market_data_reliability.config import ReliabilityConfig
from utils.market_data_reliability.dashboard_integration import (
    enrich_market_data_response,
)
from utils.market_data_reliability.layer import ReliabilityLayer
from utils.market_data_reliability.pipeline_integration import (
    MarketDataReadinessResult,
    check_market_data_readiness,
    reset_reliability_layer_singleton,
)
from utils.market_data_reliability.price_monitor_integration import (
    PriceMonitorIntegration,
)
from utils.market_data_reliability.snapshot import (
    CandidateReadiness,
    EligibilityResult,
    Snapshot,
    SnapshotResult,
)
from utils.market_data_reliability.telemetry import TelemetryCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(mode: str = "enforcing") -> ReliabilityConfig:
    """Create a minimal ReliabilityConfig for testing."""
    from utils.market_data_reliability.snapshot import FreshnessThreshold

    return ReliabilityConfig(
        mode=mode,
        freshness_thresholds={
            ("quote", "execution"): FreshnessThreshold(30.0, 120.0),
            ("quote", "display"): FreshnessThreshold(60.0, 300.0),
            ("quote", "monitoring"): FreshnessThreshold(30.0, 120.0),
            ("atr", "execution"): FreshnessThreshold(300.0, 900.0),
            ("atr", "display"): FreshnessThreshold(600.0, 1800.0),
            ("volume", "execution"): FreshnessThreshold(60.0, 300.0),
        },
        cache_ttls={"quote": 15.0, "candle": 60.0, "atr": 120.0, "volume": 30.0},
        provider_timeouts={"finnhub": 10.0, "yfinance": 15.0},
        provider_retry_limits={"finnhub": 2, "yfinance": 2},
        backoff_durations={"rate_limit": 60.0, "network_error": 30.0, "empty_response": 15.0},
        fallback_matrix={
            ("quote", "execution"): ["finnhub", "yfinance"],
            ("quote", "display"): ["finnhub", "yfinance"],
            ("quote", "monitoring"): ["finnhub", "yfinance"],
            ("atr", "execution"): ["finnhub", "yfinance"],
            ("volume", "execution"): ["finnhub", "yfinance"],
        },
    )


def _make_snapshot(
    symbol: str = "AAPL",
    data_type: str = "quote",
    trust_state: str = "trusted",
    freshness_state: str = "fresh",
    degradation_reasons: tuple[str, ...] = (),
    provider: str = "finnhub",
    provider_status: str = "success",
    age_seconds: float = 5.0,
) -> Snapshot:
    """Create a test Snapshot with sensible defaults."""
    now = datetime.now(timezone.utc)
    return Snapshot(
        symbol=symbol,
        data_type=data_type,
        requested_at=now,
        provider=provider,
        provider_status=provider_status,
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
        age_seconds=age_seconds,
        freshness_state=freshness_state,
        trust_state=trust_state,
        degradation_reasons=degradation_reasons,
        raw_provider_latency_ms=45.0,
        fallback_primary_provider=None,
    )


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the pipeline integration singleton before/after each test."""
    reset_reliability_layer_singleton()
    yield
    reset_reliability_layer_singleton()


# ---------------------------------------------------------------------------
# End-to-end: all three integrations using one ReliabilityLayer
# ---------------------------------------------------------------------------


class TestEndToEndAllConsumers:
    """Verify all three integration points use the same layer consistently."""

    def test_all_consumers_see_same_degraded_state(self):
        """When the layer reports untrusted data, all consumers react correctly.

        PM pipeline blocks, Price Monitor suppresses, Dashboard labels degraded.
        """
        mock_layer = MagicMock()
        untrusted_snapshot = _make_snapshot(
            trust_state="untrusted",
            freshness_state="unavailable",
            degradation_reasons=("all_providers_failed",),
        )
        eligible_false = EligibilityResult(
            eligible=False, reason_code="untrusted_data", snapshot=untrusted_snapshot
        )
        mock_layer.get_snapshot.return_value = SnapshotResult(
            snapshot=untrusted_snapshot, eligibility=eligible_false
        )
        mock_layer.check_candidate_readiness.return_value = CandidateReadiness(
            ready=False,
            missing_data_types=("quote",),
            reason_codes=("market_data_unavailable",),
            snapshots={"quote": untrusted_snapshot},
        )

        # 1. PM pipeline: should block
        with patch(
            "utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "enforcing"
        ), patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=mock_layer,
        ):
            pm_result = check_market_data_readiness("AAPL", ["quote"])
        assert pm_result.proceed is False
        assert "market_data_unavailable" in pm_result.reason_codes

        # 2. Price Monitor: should suppress triggers
        pm_integration = PriceMonitorIntegration()
        trigger_result = pm_integration.evaluate_trigger_eligibility(
            "AAPL", mock_layer, "enforcing"
        )
        assert trigger_result.proceed is False
        assert trigger_result.snapshot is untrusted_snapshot

        # 3. Dashboard: should label as not actionable
        with patch(
            "utils.market_data_reliability.dashboard_integration.MARKET_DATA_RELIABILITY_MODE",
            "enforcing",
        ):
            data = {"symbol": "AAPL", "price": 150.00}
            enriched = enrich_market_data_response(data, untrusted_snapshot)
        assert enriched["trust_state"] == "untrusted"
        assert enriched["freshness_state"] == "unavailable"
        assert enriched["is_actionable"] is False

    def test_all_consumers_see_same_trusted_state(self):
        """When the layer reports trusted data, all consumers proceed normally."""
        mock_layer = MagicMock()
        trusted_snapshot = _make_snapshot(
            trust_state="trusted",
            freshness_state="fresh",
        )
        eligible_true = EligibilityResult(
            eligible=True, reason_code=None, snapshot=trusted_snapshot
        )
        mock_layer.get_snapshot.return_value = SnapshotResult(
            snapshot=trusted_snapshot, eligibility=eligible_true
        )
        mock_layer.check_candidate_readiness.return_value = CandidateReadiness(
            ready=True,
            missing_data_types=(),
            reason_codes=(),
            snapshots={"quote": trusted_snapshot},
        )

        # 1. PM pipeline: should proceed
        with patch(
            "utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "enforcing"
        ), patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=mock_layer,
        ):
            pm_result = check_market_data_readiness("AAPL", ["quote"])
        assert pm_result.proceed is True
        assert pm_result.reason_codes == ()

        # 2. Price Monitor: should proceed
        pm_integration = PriceMonitorIntegration()
        trigger_result = pm_integration.evaluate_trigger_eligibility(
            "AAPL", mock_layer, "enforcing"
        )
        assert trigger_result.proceed is True
        assert trigger_result.snapshot is trusted_snapshot

        # 3. Dashboard: should mark as actionable
        with patch(
            "utils.market_data_reliability.dashboard_integration.MARKET_DATA_RELIABILITY_MODE",
            "enforcing",
        ):
            data = {"symbol": "AAPL", "price": 150.00}
            enriched = enrich_market_data_response(data, trusted_snapshot)
        assert enriched["trust_state"] == "trusted"
        assert enriched["freshness_state"] == "fresh"
        assert enriched["is_actionable"] is True


# ---------------------------------------------------------------------------
# Pipeline flow: check_market_data_readiness → ReliabilityLayer
# ---------------------------------------------------------------------------


class TestPipelineToLayerFlow:
    """Verify the check_market_data_readiness → ReliabilityLayer path."""

    def test_readiness_calls_check_candidate_readiness_with_correct_args(self):
        """check_market_data_readiness delegates to layer.check_candidate_readiness."""
        mock_layer = MagicMock()
        mock_layer.check_candidate_readiness.return_value = CandidateReadiness(
            ready=True,
            missing_data_types=(),
            reason_codes=(),
            snapshots={},
        )

        with patch(
            "utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "enforcing"
        ), patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=mock_layer,
        ):
            check_market_data_readiness("TSLA", ["quote", "atr", "volume"])

        mock_layer.check_candidate_readiness.assert_called_once_with(
            symbol="TSLA",
            required_data_types=["quote", "atr", "volume"],
            consumer="PM",
        )

    def test_readiness_propagates_multiple_reason_codes(self):
        """Multiple missing data types produce multiple reason codes in result."""
        mock_layer = MagicMock()
        mock_layer.check_candidate_readiness.return_value = CandidateReadiness(
            ready=False,
            missing_data_types=("quote", "atr", "volume"),
            reason_codes=("quote_stale", "atr_stale", "volume_unavailable"),
            snapshots={},
        )

        with patch(
            "utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "enforcing"
        ), patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=mock_layer,
        ):
            result = check_market_data_readiness("AMD", ["quote", "atr", "volume"])

        assert result.proceed is False
        assert set(result.reason_codes) == {"quote_stale", "atr_stale", "volume_unavailable"}
        assert set(result.missing_data_types) == {"quote", "atr", "volume"}

    def test_observe_mode_uniform_across_integrations(self):
        """Observe mode never blocks any consumer — PM, Price Monitor all proceed."""
        mock_layer = MagicMock()
        untrusted_snap = _make_snapshot(
            trust_state="untrusted",
            freshness_state="unavailable",
            degradation_reasons=("all_providers_failed",),
        )
        mock_layer.check_candidate_readiness.return_value = CandidateReadiness(
            ready=False,
            missing_data_types=("quote",),
            reason_codes=("market_data_unavailable",),
            snapshots={"quote": untrusted_snap},
        )
        eligible_false = EligibilityResult(
            eligible=False, reason_code="untrusted_data", snapshot=untrusted_snap
        )
        mock_layer.get_snapshot.return_value = SnapshotResult(
            snapshot=untrusted_snap, eligibility=eligible_false
        )

        # PM pipeline in observe
        with patch(
            "utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "observe"
        ), patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=mock_layer,
        ):
            pm_result = check_market_data_readiness("AAPL", ["quote"])
        assert pm_result.proceed is True  # observe never blocks

        # Price Monitor in observe
        pm_integration = PriceMonitorIntegration()
        trigger_result = pm_integration.evaluate_trigger_eligibility(
            "AAPL", mock_layer, "observe"
        )
        assert trigger_result.proceed is True  # observe never blocks


# ---------------------------------------------------------------------------
# Telemetry population after integration calls
# ---------------------------------------------------------------------------


class TestTelemetryPopulation:
    """Verify TelemetryCollector records after integration calls."""

    def test_enforcing_block_records_fail_closed_telemetry(self):
        """When PM pipeline blocks a candidate, telemetry.record_fail_closed is called."""
        mock_telemetry = MagicMock()
        mock_layer = MagicMock()
        mock_layer._telemetry = mock_telemetry
        mock_layer.check_candidate_readiness.return_value = CandidateReadiness(
            ready=False,
            missing_data_types=("quote",),
            reason_codes=("quote_stale",),
            snapshots={},
        )

        with patch(
            "utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "enforcing"
        ), patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=mock_layer,
        ):
            result = check_market_data_readiness("NVDA", ["quote"])

        assert result.proceed is False
        mock_telemetry.record_fail_closed.assert_called_once_with(
            candidate_id="NVDA:PM",
            reason="quote_stale",
        )

    def test_telemetry_records_multiple_reasons_as_comma_separated(self):
        """Multiple reason codes are joined with comma in telemetry reason."""
        mock_telemetry = MagicMock()
        mock_layer = MagicMock()
        mock_layer._telemetry = mock_telemetry
        mock_layer.check_candidate_readiness.return_value = CandidateReadiness(
            ready=False,
            missing_data_types=("quote", "volume"),
            reason_codes=("quote_stale", "volume_unavailable"),
            snapshots={},
        )

        with patch(
            "utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "enforcing"
        ), patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=mock_layer,
        ):
            check_market_data_readiness("GOOG", ["quote", "volume"])

        mock_telemetry.record_fail_closed.assert_called_once_with(
            candidate_id="GOOG:PM",
            reason="quote_stale,volume_unavailable",
        )

    def test_observe_mode_does_not_record_fail_closed_telemetry(self):
        """Observe mode does not write fail-closed telemetry (no actual block)."""
        mock_telemetry = MagicMock()
        mock_layer = MagicMock()
        mock_layer._telemetry = mock_telemetry
        mock_layer.check_candidate_readiness.return_value = CandidateReadiness(
            ready=False,
            missing_data_types=("quote",),
            reason_codes=("quote_stale",),
            snapshots={},
        )

        with patch(
            "utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "observe"
        ), patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=mock_layer,
        ):
            result = check_market_data_readiness("AAPL", ["quote"])

        assert result.proceed is True
        mock_telemetry.record_fail_closed.assert_not_called()

    def test_telemetry_collector_accumulates_across_consumers(self):
        """TelemetryCollector accumulates metrics from multiple consumer calls."""
        telemetry = TelemetryCollector()

        # Simulate pipeline blocking a candidate
        telemetry.record_fail_closed("AAPL:PM", "quote_stale")

        # Simulate provider calls from price monitor evaluation
        telemetry.record_provider_call("finnhub", "quote", success=True)
        telemetry.record_provider_call("finnhub", "quote", success=True)
        telemetry.record_provider_call("yfinance", "atr", success=False)

        # Simulate cache hit from dashboard
        telemetry.record_cache_hit("quote")

        summary = telemetry.get_cycle_summary()
        assert summary["provider_calls_success"] == 2
        assert summary["provider_calls_failure"] == 1
        assert summary["cache_hits"] == 1
        assert summary["fail_closed_decisions"] == 1
        assert summary["fail_closed_details"] == [
            {"candidate_id": "AAPL:PM", "reason": "quote_stale"}
        ]

    def test_telemetry_reset_clears_all_counters(self):
        """Telemetry reset clears all counters for a new cycle."""
        telemetry = TelemetryCollector()
        telemetry.record_fail_closed("AAPL:PM", "quote_stale")
        telemetry.record_provider_call("finnhub", "quote", success=True)
        telemetry.record_cache_hit("quote")

        telemetry.reset()

        summary = telemetry.get_cycle_summary()
        assert summary["provider_calls_success"] == 0
        assert summary["provider_calls_failure"] == 0
        assert summary["cache_hits"] == 0
        assert summary["fail_closed_decisions"] == 0
        assert summary["fail_closed_details"] == []


# ---------------------------------------------------------------------------
# Feature flag disabled mode: no consumer is affected
# ---------------------------------------------------------------------------


class TestDisabledModeUniform:
    """Disabled mode ensures no consumer is interfered with."""

    def test_disabled_mode_all_consumers_proceed(self):
        """In disabled mode, all three consumers proceed without layer calls."""
        mock_layer = MagicMock()

        # PM pipeline: disabled mode
        with patch(
            "utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "disabled"
        ):
            pm_result = check_market_data_readiness("AAPL", ["quote"])
        assert pm_result.proceed is True
        # Pipeline doesn't call layer in disabled mode
        mock_layer.check_candidate_readiness.assert_not_called()

        # Price Monitor: disabled mode
        pm_integration = PriceMonitorIntegration()
        trigger_result = pm_integration.evaluate_trigger_eligibility(
            "AAPL", mock_layer, "disabled"
        )
        assert trigger_result.proceed is True
        mock_layer.get_snapshot.assert_not_called()

        # Dashboard: disabled mode
        with patch(
            "utils.market_data_reliability.dashboard_integration.MARKET_DATA_RELIABILITY_MODE",
            "disabled",
        ):
            data = {"symbol": "AAPL", "price": 150.00}
            snapshot = _make_snapshot()
            enriched = enrich_market_data_response(data, snapshot)
        assert "freshness_state" not in enriched
        assert "trust_state" not in enriched


# ---------------------------------------------------------------------------
# Fail-open uniformity across all consumers
# ---------------------------------------------------------------------------


class TestFailOpenUniform:
    """All three integrations fail open (never crash the pipeline)."""

    def test_all_consumers_fail_open_on_layer_error(self):
        """If the reliability layer raises, all consumers proceed."""
        mock_layer = MagicMock()
        mock_layer.get_snapshot.side_effect = RuntimeError("unexpected failure")
        mock_layer.check_candidate_readiness.side_effect = RuntimeError("unexpected failure")

        # PM pipeline: fail-open
        with patch(
            "utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "enforcing"
        ), patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=mock_layer,
        ):
            pm_result = check_market_data_readiness("AAPL", ["quote"])
        assert pm_result.proceed is True

        # Price Monitor: fail-open
        pm_integration = PriceMonitorIntegration()
        trigger_result = pm_integration.evaluate_trigger_eligibility(
            "AAPL", mock_layer, "enforcing"
        )
        assert trigger_result.proceed is True

        # Dashboard: fail-open (broken snapshot)
        with patch(
            "utils.market_data_reliability.dashboard_integration.MARKET_DATA_RELIABILITY_MODE",
            "enforcing",
        ):
            data = {"symbol": "AAPL", "price": 150.00}
            enriched = enrich_market_data_response(data, None)
        assert enriched is data
        assert "freshness_state" not in enriched

    def test_layer_none_does_not_crash_pipeline(self):
        """When layer initialization returns None, pipeline proceeds gracefully."""
        with patch(
            "utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "enforcing"
        ), patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=None,
        ):
            result = check_market_data_readiness("AAPL", ["quote", "atr"])

        assert result.proceed is True
        assert result.reason_codes == ()
        assert result.missing_data_types == ()


# ---------------------------------------------------------------------------
# Real ReliabilityLayer integration (with mock provider)
# ---------------------------------------------------------------------------


class TestRealLayerIntegration:
    """Integration tests using a real ReliabilityLayer with mocked provider."""

    def test_layer_check_candidate_readiness_enforcing_all_fail(self):
        """Real layer: all providers fail → check_candidate_readiness returns not ready."""
        config = _make_config(mode="enforcing")

        def failing_provider(provider: str, symbol: str, data_type: str) -> dict:
            raise RuntimeError(f"Provider {provider} is down")

        layer = ReliabilityLayer(config, fetch_from_provider=failing_provider)
        readiness = layer.check_candidate_readiness("AAPL", ["quote"], "PM")

        assert readiness.ready is False
        assert "quote" in readiness.missing_data_types
        assert len(readiness.reason_codes) > 0

    def test_layer_check_candidate_readiness_enforcing_success(self):
        """Real layer: provider succeeds → check_candidate_readiness returns ready."""
        config = _make_config(mode="enforcing")

        def successful_provider(provider: str, symbol: str, data_type: str) -> dict:
            return {
                "symbol": symbol,
                "c": 150.00,
                "h": 151.00,
                "l": 148.00,
                "o": 149.00,
                "pc": 148.50,
                "t": int(datetime.now(timezone.utc).timestamp()),
            }

        layer = ReliabilityLayer(config, fetch_from_provider=successful_provider)
        readiness = layer.check_candidate_readiness("AAPL", ["quote"], "PM")

        assert readiness.ready is True
        assert readiness.missing_data_types == ()
        assert readiness.reason_codes == ()
        assert "quote" in readiness.snapshots

    def test_layer_telemetry_populated_after_provider_calls(self):
        """Real layer: telemetry is populated after get_snapshot calls."""
        config = _make_config(mode="enforcing")

        call_count = {"n": 0}

        def alternating_provider(provider: str, symbol: str, data_type: str) -> dict:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("first provider fails")
            return {
                "symbol": symbol,
                "c": 150.00,
                "h": 151.00,
                "l": 148.00,
                "o": 149.00,
                "pc": 148.50,
                "t": int(datetime.now(timezone.utc).timestamp()),
            }

        layer = ReliabilityLayer(config, fetch_from_provider=alternating_provider)
        result = layer.get_snapshot("AAPL", "quote", "PM")

        summary = layer.get_cycle_telemetry()
        # First provider failed, second (fallback) succeeded
        assert summary["provider_calls_failure"] >= 1
        assert summary["provider_calls_success"] >= 1
        assert summary["fallback_usage"] >= 1

    def test_layer_disabled_mode_produces_passthrough_result(self):
        """Real layer in disabled mode: returns passthrough without provider calls."""
        config = _make_config(mode="disabled")

        def should_not_call(provider: str, symbol: str, data_type: str) -> dict:
            raise AssertionError("Provider should not be called in disabled mode")

        layer = ReliabilityLayer(config, fetch_from_provider=should_not_call)
        readiness = layer.check_candidate_readiness("AAPL", ["quote"], "PM")

        assert readiness.ready is True
        assert readiness.missing_data_types == ()

    def test_full_cycle_pm_block_then_dashboard_labels(self):
        """End-to-end: provider fails → PM blocked → dashboard labels degraded."""
        config = _make_config(mode="enforcing")

        def failing_provider(provider: str, symbol: str, data_type: str) -> dict:
            raise RuntimeError("All providers down")

        layer = ReliabilityLayer(config, fetch_from_provider=failing_provider)

        # PM check — should block
        readiness = layer.check_candidate_readiness("AAPL", ["quote"], "PM")
        assert readiness.ready is False

        # Get the snapshot used by the layer for dashboard enrichment
        snapshot_result = layer.get_snapshot("AAPL", "quote", "Dashboard_API")
        snapshot = snapshot_result.snapshot

        # Dashboard enrichment — should label as not actionable
        with patch(
            "utils.market_data_reliability.dashboard_integration.MARKET_DATA_RELIABILITY_MODE",
            "enforcing",
        ):
            data = {"symbol": "AAPL", "price": 0}
            enriched = enrich_market_data_response(data, snapshot)

        assert enriched["trust_state"] == "untrusted"
        assert enriched["is_actionable"] is False
        assert enriched["freshness_state"] == "unavailable"
