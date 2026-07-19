"""Unit tests for Price Monitor integration with Market Data Reliability Layer.

Tests trigger eligibility evaluation, rate-limited logging, degradation alerts,
suppressed trigger telemetry, and fail-open behavior.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from utils.market_data_reliability.price_monitor_integration import (
    PriceMonitorIntegration,
    TriggerEligibility,
    _LOG_RATE_LIMIT_SECONDS,
    evaluate_trigger_eligibility,
)
from utils.market_data_reliability.snapshot import (
    EligibilityResult,
    Snapshot,
    SnapshotResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_snapshot(
    symbol: str = "AAPL",
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
        data_type="quote",
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


def _make_snapshot_result(
    snapshot: Snapshot,
    eligible: bool = True,
    reason_code: str | None = None,
) -> SnapshotResult:
    """Create a SnapshotResult from a snapshot and eligibility verdict."""
    eligibility = EligibilityResult(
        eligible=eligible,
        reason_code=reason_code,
        snapshot=snapshot,
    )
    return SnapshotResult(snapshot=snapshot, eligibility=eligibility)


@pytest.fixture
def integration():
    """Fresh PriceMonitorIntegration instance with no rate-limit state."""
    return PriceMonitorIntegration()


@pytest.fixture
def mock_layer():
    """Mock ReliabilityLayer."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Test: disabled mode always proceeds (Requirement 8.1 guard)
# ---------------------------------------------------------------------------


class TestDisabledMode:
    """When MARKET_DATA_RELIABILITY_MODE == 'disabled', always proceed."""

    def test_disabled_mode_returns_proceed_true(self, integration, mock_layer):
        result = integration.evaluate_trigger_eligibility("AAPL", mock_layer, "disabled")
        assert result.proceed is True
        assert result.reason is None
        assert result.snapshot is None

    def test_disabled_mode_does_not_call_layer(self, integration, mock_layer):
        integration.evaluate_trigger_eligibility("AAPL", mock_layer, "disabled")
        mock_layer.get_snapshot.assert_not_called()


# ---------------------------------------------------------------------------
# Test: trusted data proceeds (Requirement 8.1)
# ---------------------------------------------------------------------------


class TestTrustedData:
    """When snapshot is trusted, triggers should proceed normally."""

    def test_trusted_snapshot_returns_proceed_true(self, integration, mock_layer):
        snapshot = _make_snapshot(trust_state="trusted")
        mock_layer.get_snapshot.return_value = _make_snapshot_result(
            snapshot, eligible=True
        )
        result = integration.evaluate_trigger_eligibility("AAPL", mock_layer, "enforcing")
        assert result.proceed is True
        assert result.reason is None
        assert result.snapshot is snapshot

    def test_trusted_snapshot_in_observe_mode(self, integration, mock_layer):
        snapshot = _make_snapshot(trust_state="trusted")
        mock_layer.get_snapshot.return_value = _make_snapshot_result(
            snapshot, eligible=True
        )
        result = integration.evaluate_trigger_eligibility("AAPL", mock_layer, "observe")
        assert result.proceed is True
        assert result.snapshot is snapshot


# ---------------------------------------------------------------------------
# Test: degraded/untrusted data suppresses triggers (Requirement 8.1, 8.2)
# ---------------------------------------------------------------------------


class TestDegradedData:
    """When snapshot is degraded or untrusted, triggers should be suppressed in enforcing mode."""

    def test_degraded_snapshot_suppresses_in_enforcing(self, integration, mock_layer):
        snapshot = _make_snapshot(
            trust_state="degraded",
            freshness_state="stale",
            degradation_reasons=("stale_source_timestamp",),
        )
        mock_layer.get_snapshot.return_value = _make_snapshot_result(
            snapshot, eligible=False, reason_code="stale_data"
        )
        result = integration.evaluate_trigger_eligibility("AAPL", mock_layer, "enforcing")
        assert result.proceed is False
        assert result.reason is not None
        assert "degraded" in result.reason
        assert result.snapshot is snapshot

    def test_untrusted_snapshot_suppresses_in_enforcing(self, integration, mock_layer):
        snapshot = _make_snapshot(
            trust_state="untrusted",
            freshness_state="unavailable",
            degradation_reasons=("all_providers_failed",),
        )
        mock_layer.get_snapshot.return_value = _make_snapshot_result(
            snapshot, eligible=False, reason_code="untrusted_data"
        )
        result = integration.evaluate_trigger_eligibility("AAPL", mock_layer, "enforcing")
        assert result.proceed is False
        assert result.reason is not None
        assert "untrusted" in result.reason

    def test_degraded_snapshot_proceeds_in_observe_mode(self, integration, mock_layer):
        """Observe mode: logs suppression but still proceeds (Requirement 14.3)."""
        snapshot = _make_snapshot(
            trust_state="degraded",
            freshness_state="stale",
            degradation_reasons=("stale_source_timestamp",),
        )
        mock_layer.get_snapshot.return_value = _make_snapshot_result(
            snapshot, eligible=False, reason_code="stale_data"
        )
        result = integration.evaluate_trigger_eligibility("AAPL", mock_layer, "observe")
        assert result.proceed is True
        assert result.snapshot is snapshot


# ---------------------------------------------------------------------------
# Test: data-degradation safety alert emission (Requirement 8.2)
# ---------------------------------------------------------------------------


class TestDegradationAlert:
    """A separate data-degradation safety alert should be emitted on degradation."""

    def test_degradation_alert_logged(self, integration, mock_layer, caplog):
        snapshot = _make_snapshot(
            trust_state="untrusted",
            freshness_state="unavailable",
            degradation_reasons=("all_providers_failed",),
            provider="finnhub",
            provider_status="error",
            age_seconds=120.0,
        )
        mock_layer.get_snapshot.return_value = _make_snapshot_result(
            snapshot, eligible=False, reason_code="untrusted_data"
        )
        with caplog.at_level(logging.WARNING):
            integration.evaluate_trigger_eligibility("AAPL", mock_layer, "enforcing")

        # Should contain the data_degradation_safety_alert log message
        alert_messages = [
            r for r in caplog.records if "data_degradation_safety_alert" in r.message
        ]
        assert len(alert_messages) >= 1
        alert_msg = alert_messages[0].message
        assert "AAPL" in alert_msg
        assert "finnhub" in alert_msg
        assert "untrusted" in alert_msg


# ---------------------------------------------------------------------------
# Test: structured telemetry on suppressed triggers (Requirement 8.3)
# ---------------------------------------------------------------------------


class TestSuppressedTriggerTelemetry:
    """Suppressed triggers should write structured telemetry with reason and state."""

    def test_suppressed_trigger_telemetry_logged(self, integration, mock_layer, caplog):
        snapshot = _make_snapshot(
            trust_state="degraded",
            freshness_state="stale",
            degradation_reasons=("stale_source_timestamp",),
            provider="yfinance",
            provider_status="success",
            age_seconds=200.0,
        )
        mock_layer.get_snapshot.return_value = _make_snapshot_result(
            snapshot, eligible=False, reason_code="stale_data"
        )
        with caplog.at_level(logging.INFO):
            integration.evaluate_trigger_eligibility("AAPL", mock_layer, "enforcing")

        telemetry_messages = [
            r for r in caplog.records if "suppressed_trigger_telemetry" in r.message
        ]
        assert len(telemetry_messages) >= 1
        msg = telemetry_messages[0].message
        assert "AAPL" in msg
        assert "degraded" in msg
        assert "stale" in msg
        assert "quote" in msg
        assert "Price_Monitor" in msg


# ---------------------------------------------------------------------------
# Test: rate-limited logging (Requirement 8.4)
# ---------------------------------------------------------------------------


class TestRateLimitedLogging:
    """Degraded-check log entries should be rate-limited per symbol."""

    def test_first_call_logs_warning(self, integration, mock_layer, caplog):
        snapshot = _make_snapshot(
            trust_state="untrusted",
            degradation_reasons=("all_providers_failed",),
        )
        mock_layer.get_snapshot.return_value = _make_snapshot_result(
            snapshot, eligible=False, reason_code="untrusted_data"
        )
        with caplog.at_level(logging.WARNING):
            integration.evaluate_trigger_eligibility("AAPL", mock_layer, "enforcing")

        suppression_warnings = [
            r for r in caplog.records
            if "trigger suppressed" in r.message and r.levelno == logging.WARNING
        ]
        assert len(suppression_warnings) >= 1

    def test_repeated_calls_within_window_rate_limited(self, integration, mock_layer, caplog):
        """Second call within 60s window should not re-log the suppression warning."""
        snapshot = _make_snapshot(
            trust_state="untrusted",
            degradation_reasons=("all_providers_failed",),
        )
        mock_layer.get_snapshot.return_value = _make_snapshot_result(
            snapshot, eligible=False, reason_code="untrusted_data"
        )
        with caplog.at_level(logging.WARNING):
            integration.evaluate_trigger_eligibility("AAPL", mock_layer, "enforcing")
            caplog.clear()
            integration.evaluate_trigger_eligibility("AAPL", mock_layer, "enforcing")

        suppression_warnings = [
            r for r in caplog.records
            if "trigger suppressed" in r.message and r.levelno == logging.WARNING
        ]
        # Second call should be rate-limited — no additional suppression warning
        assert len(suppression_warnings) == 0

    def test_different_symbols_not_rate_limited(self, integration, mock_layer, caplog):
        """Different symbols have independent rate-limit windows."""
        snapshot_aapl = _make_snapshot(
            symbol="AAPL",
            trust_state="untrusted",
            degradation_reasons=("all_providers_failed",),
        )
        snapshot_msft = _make_snapshot(
            symbol="MSFT",
            trust_state="untrusted",
            degradation_reasons=("all_providers_failed",),
        )
        mock_layer.get_snapshot.side_effect = [
            _make_snapshot_result(snapshot_aapl, eligible=False, reason_code="untrusted"),
            _make_snapshot_result(snapshot_msft, eligible=False, reason_code="untrusted"),
        ]
        with caplog.at_level(logging.WARNING):
            integration.evaluate_trigger_eligibility("AAPL", mock_layer, "enforcing")
            integration.evaluate_trigger_eligibility("MSFT", mock_layer, "enforcing")

        suppression_warnings = [
            r for r in caplog.records
            if "trigger suppressed" in r.message and r.levelno == logging.WARNING
        ]
        # Both symbols should log (separate rate-limit windows)
        assert len(suppression_warnings) == 2

    def test_rate_limit_resets_after_interval(self, integration, mock_layer, caplog):
        """After the rate-limit window expires, logging should resume."""
        snapshot = _make_snapshot(
            trust_state="untrusted",
            degradation_reasons=("all_providers_failed",),
        )
        mock_layer.get_snapshot.return_value = _make_snapshot_result(
            snapshot, eligible=False, reason_code="untrusted_data"
        )

        # First call — logs
        with caplog.at_level(logging.WARNING):
            integration.evaluate_trigger_eligibility("AAPL", mock_layer, "enforcing")

        # Simulate time passing beyond the rate-limit window
        integration._last_log_time["AAPL"] -= (_LOG_RATE_LIMIT_SECONDS + 1)

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            integration.evaluate_trigger_eligibility("AAPL", mock_layer, "enforcing")

        suppression_warnings = [
            r for r in caplog.records
            if "trigger suppressed" in r.message and r.levelno == logging.WARNING
        ]
        assert len(suppression_warnings) == 1


# ---------------------------------------------------------------------------
# Test: automatic recovery (Requirement 8.5)
# ---------------------------------------------------------------------------


class TestAutomaticRecovery:
    """When trusted data returns, triggers should resume automatically."""

    def test_recovery_after_degradation(self, integration, mock_layer):
        """After degradation, trusted data should allow triggers to proceed."""
        # First: degraded → suppressed
        degraded_snap = _make_snapshot(
            trust_state="degraded",
            degradation_reasons=("stale_source_timestamp",),
        )
        mock_layer.get_snapshot.return_value = _make_snapshot_result(
            degraded_snap, eligible=False, reason_code="stale_data"
        )
        result1 = integration.evaluate_trigger_eligibility("AAPL", mock_layer, "enforcing")
        assert result1.proceed is False

        # Second: trusted data returns → proceed
        trusted_snap = _make_snapshot(trust_state="trusted")
        mock_layer.get_snapshot.return_value = _make_snapshot_result(
            trusted_snap, eligible=True
        )
        result2 = integration.evaluate_trigger_eligibility("AAPL", mock_layer, "enforcing")
        assert result2.proceed is True
        assert result2.snapshot is trusted_snap


# ---------------------------------------------------------------------------
# Test: fail-open behavior (production safety)
# ---------------------------------------------------------------------------


class TestFailOpen:
    """If the reliability check raises, proceed with triggers (fail-open)."""

    def test_layer_raises_returns_proceed(self, integration, mock_layer):
        mock_layer.get_snapshot.side_effect = RuntimeError("internal error")
        result = integration.evaluate_trigger_eligibility("AAPL", mock_layer, "enforcing")
        assert result.proceed is True
        assert result.reason is None
        assert result.snapshot is None

    def test_layer_raises_logs_error(self, integration, mock_layer, caplog):
        mock_layer.get_snapshot.side_effect = RuntimeError("kaboom")
        with caplog.at_level(logging.ERROR):
            integration.evaluate_trigger_eligibility("AAPL", mock_layer, "enforcing")

        error_messages = [
            r for r in caplog.records if r.levelno == logging.ERROR
        ]
        assert len(error_messages) >= 1
        assert "failing open" in error_messages[0].message


# ---------------------------------------------------------------------------
# Test: module-level convenience function
# ---------------------------------------------------------------------------


class TestModuleLevelFunction:
    """The module-level evaluate_trigger_eligibility function delegates correctly."""

    def test_module_function_disabled_mode(self):
        mock_layer = MagicMock()
        result = evaluate_trigger_eligibility("AAPL", mock_layer, "disabled")
        assert result.proceed is True
        assert result.snapshot is None

    def test_module_function_trusted_data(self):
        mock_layer = MagicMock()
        snapshot = _make_snapshot(trust_state="trusted")
        mock_layer.get_snapshot.return_value = _make_snapshot_result(
            snapshot, eligible=True
        )
        result = evaluate_trigger_eligibility("AAPL", mock_layer, "enforcing")
        assert result.proceed is True
        assert result.snapshot is snapshot

    def test_module_function_untrusted_data(self):
        mock_layer = MagicMock()
        snapshot = _make_snapshot(
            trust_state="untrusted",
            degradation_reasons=("cross_symbol_response",),
        )
        mock_layer.get_snapshot.return_value = _make_snapshot_result(
            snapshot, eligible=False, reason_code="untrusted_data"
        )
        result = evaluate_trigger_eligibility("AAPL", mock_layer, "enforcing")
        assert result.proceed is False
        assert result.reason is not None
