"""Unit tests for market data reliability pipeline integration.

Tests the check_market_data_readiness() function and MarketDataReadinessResult
contract, verifying feature-flag behavior, fail-open semantics, telemetry
recording, and mode-dependent candidate blocking.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from utils.market_data_reliability.pipeline_integration import (
    MarketDataReadinessResult,
    check_market_data_readiness,
    reset_reliability_layer_singleton,
)
from utils.market_data_reliability.snapshot import CandidateReadiness, Snapshot


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the module-level ReliabilityLayer singleton before and after each test."""
    reset_reliability_layer_singleton()
    yield
    reset_reliability_layer_singleton()


# ---------------------------------------------------------------------------
# MarketDataReadinessResult contract tests
# ---------------------------------------------------------------------------


class TestMarketDataReadinessResult:
    """Verify the MarketDataReadinessResult dataclass contract."""

    def test_proceed_true_has_empty_reasons(self):
        result = MarketDataReadinessResult(
            proceed=True, reason_codes=(), missing_data_types=()
        )
        assert result.proceed is True
        assert result.reason_codes == ()
        assert result.missing_data_types == ()

    def test_proceed_false_has_reason_codes(self):
        result = MarketDataReadinessResult(
            proceed=False,
            reason_codes=("quote_stale",),
            missing_data_types=("quote",),
        )
        assert result.proceed is False
        assert "quote_stale" in result.reason_codes
        assert "quote" in result.missing_data_types

    def test_is_frozen(self):
        result = MarketDataReadinessResult(
            proceed=True, reason_codes=(), missing_data_types=()
        )
        with pytest.raises(AttributeError):
            result.proceed = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Disabled mode — no interference
# ---------------------------------------------------------------------------


class TestDisabledMode:
    """When MARKET_DATA_RELIABILITY_MODE == 'disabled', always proceed."""

    @patch("utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "disabled")
    def test_disabled_returns_proceed_true(self):
        """Disabled mode returns proceed=True without touching the layer."""
        result = check_market_data_readiness("AAPL")
        assert result.proceed is True
        assert result.reason_codes == ()
        assert result.missing_data_types == ()

    @patch("utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "disabled")
    def test_disabled_does_not_initialize_layer(self):
        """Disabled mode skips layer initialization entirely."""
        with patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer"
        ) as mock_get:
            result = check_market_data_readiness("TSLA", ["quote", "atr"])
        mock_get.assert_not_called()
        assert result.proceed is True


# ---------------------------------------------------------------------------
# Observe mode — log but don't block
# ---------------------------------------------------------------------------


class TestObserveMode:
    """When mode == 'observe', run checks but never block."""

    @patch("utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "observe")
    def test_observe_returns_proceed_true_even_when_data_missing(self):
        """Observe mode always proceeds, even when data would block in enforcing."""
        mock_layer = MagicMock()
        # Simulate the layer saying data is NOT ready
        mock_layer.check_candidate_readiness.return_value = CandidateReadiness(
            ready=False,
            missing_data_types=("quote",),
            reason_codes=("quote_stale",),
            snapshots={},
        )

        with patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=mock_layer,
        ):
            result = check_market_data_readiness("AAPL", ["quote"])

        assert result.proceed is True
        # Reason codes are still populated for observability
        assert result.reason_codes == ("quote_stale",)
        assert result.missing_data_types == ("quote",)

    @patch("utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "observe")
    def test_observe_returns_proceed_true_when_data_ready(self):
        """Observe mode returns proceed=True when data is fine."""
        mock_layer = MagicMock()
        mock_layer.check_candidate_readiness.return_value = CandidateReadiness(
            ready=True,
            missing_data_types=(),
            reason_codes=(),
            snapshots={},
        )

        with patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=mock_layer,
        ):
            result = check_market_data_readiness("AAPL", ["quote"])

        assert result.proceed is True
        assert result.reason_codes == ()
        assert result.missing_data_types == ()

    @patch("utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "observe")
    def test_observe_logs_warning_when_would_block(self, caplog):
        """Observe mode logs a warning for candidates that would be blocked."""
        import logging

        mock_layer = MagicMock()
        mock_layer.check_candidate_readiness.return_value = CandidateReadiness(
            ready=False,
            missing_data_types=("atr",),
            reason_codes=("atr_stale",),
            snapshots={},
        )

        with patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=mock_layer,
        ):
            with caplog.at_level(logging.WARNING):
                result = check_market_data_readiness("MSFT", ["quote", "atr"])

        assert result.proceed is True
        assert "OBSERVE" in caplog.text
        assert "MSFT" in caplog.text


# ---------------------------------------------------------------------------
# Enforcing mode — block on untrusted/unavailable data
# ---------------------------------------------------------------------------


class TestEnforcingMode:
    """When mode == 'enforcing', block candidates on degraded data."""

    @patch("utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_enforcing_blocks_when_data_unavailable(self):
        """Enforcing mode returns proceed=False when data is not ready."""
        mock_layer = MagicMock()
        mock_layer.check_candidate_readiness.return_value = CandidateReadiness(
            ready=False,
            missing_data_types=("quote",),
            reason_codes=("market_data_unavailable",),
            snapshots={},
        )

        with patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=mock_layer,
        ):
            result = check_market_data_readiness("AAPL", ["quote"])

        assert result.proceed is False
        assert "market_data_unavailable" in result.reason_codes
        assert "quote" in result.missing_data_types

    @patch("utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_enforcing_allows_when_data_ready(self):
        """Enforcing mode returns proceed=True when data is trusted."""
        mock_layer = MagicMock()
        mock_layer.check_candidate_readiness.return_value = CandidateReadiness(
            ready=True,
            missing_data_types=(),
            reason_codes=(),
            snapshots={},
        )

        with patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=mock_layer,
        ):
            result = check_market_data_readiness("AAPL", ["quote"])

        assert result.proceed is True
        assert result.reason_codes == ()
        assert result.missing_data_types == ()

    @patch("utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_enforcing_records_blocker_telemetry(self):
        """Enforcing mode records telemetry when blocking a candidate."""
        mock_telemetry = MagicMock()
        mock_layer = MagicMock()
        mock_layer._telemetry = mock_telemetry
        mock_layer.check_candidate_readiness.return_value = CandidateReadiness(
            ready=False,
            missing_data_types=("quote", "atr"),
            reason_codes=("quote_stale", "atr_stale"),
            snapshots={},
        )

        with patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=mock_layer,
        ):
            result = check_market_data_readiness("NVDA", ["quote", "atr"])

        assert result.proceed is False
        # Telemetry should be recorded
        mock_telemetry.record_fail_closed.assert_called_once_with(
            candidate_id="NVDA:PM",
            reason="quote_stale,atr_stale",
        )

    @patch("utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_enforcing_multiple_missing_data_types(self):
        """Enforcing mode reports all missing data types and reason codes."""
        mock_layer = MagicMock()
        mock_layer.check_candidate_readiness.return_value = CandidateReadiness(
            ready=False,
            missing_data_types=("quote", "volume"),
            reason_codes=("quote_stale", "volume_unavailable"),
            snapshots={},
        )

        with patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=mock_layer,
        ):
            result = check_market_data_readiness("AMD", ["quote", "volume"])

        assert result.proceed is False
        assert set(result.reason_codes) == {"quote_stale", "volume_unavailable"}
        assert set(result.missing_data_types) == {"quote", "volume"}

    @patch("utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_enforcing_distinguishes_market_data_degraded_from_no_trade(self):
        """Blocked result reason_codes distinguish data degradation from 'no trade'."""
        mock_layer = MagicMock()
        mock_layer.check_candidate_readiness.return_value = CandidateReadiness(
            ready=False,
            missing_data_types=("quote",),
            reason_codes=("provider_rate_limited",),
            snapshots={},
        )

        with patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=mock_layer,
        ):
            result = check_market_data_readiness("TSLA", ["quote"])

        # This is market-data degradation, not "no valid trade"
        assert result.proceed is False
        assert "provider_rate_limited" in result.reason_codes


# ---------------------------------------------------------------------------
# Default required_data_types behavior
# ---------------------------------------------------------------------------


class TestDefaultRequiredDataTypes:
    """Verify default required_data_types is ["quote"] when None."""

    @patch("utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_none_defaults_to_quote(self):
        """When required_data_types is None, defaults to ['quote']."""
        mock_layer = MagicMock()
        mock_layer.check_candidate_readiness.return_value = CandidateReadiness(
            ready=True,
            missing_data_types=(),
            reason_codes=(),
            snapshots={},
        )

        with patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=mock_layer,
        ):
            check_market_data_readiness("AAPL")

        # Should have been called with ["quote"] as default
        mock_layer.check_candidate_readiness.assert_called_once_with(
            symbol="AAPL",
            required_data_types=["quote"],
            consumer="PM",
        )

    @patch("utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_explicit_data_types_passed_through(self):
        """Explicit required_data_types list is passed through unchanged."""
        mock_layer = MagicMock()
        mock_layer.check_candidate_readiness.return_value = CandidateReadiness(
            ready=True,
            missing_data_types=(),
            reason_codes=(),
            snapshots={},
        )

        with patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=mock_layer,
        ):
            check_market_data_readiness("AAPL", ["quote", "atr", "volume"])

        mock_layer.check_candidate_readiness.assert_called_once_with(
            symbol="AAPL",
            required_data_types=["quote", "atr", "volume"],
            consumer="PM",
        )


# ---------------------------------------------------------------------------
# Fail-open behavior
# ---------------------------------------------------------------------------


class TestFailOpen:
    """Verify fail-open: internal errors never block the pipeline."""

    @patch("utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_layer_init_failure_returns_proceed(self):
        """If ReliabilityLayer fails to initialize, proceed (fail-open)."""
        with patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=None,
        ):
            result = check_market_data_readiness("AAPL", ["quote"])

        assert result.proceed is True
        assert result.reason_codes == ()

    @patch("utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_layer_raises_exception_returns_proceed(self):
        """If layer.check_candidate_readiness raises, proceed (fail-open)."""
        mock_layer = MagicMock()
        mock_layer.check_candidate_readiness.side_effect = RuntimeError("boom")

        with patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=mock_layer,
        ):
            result = check_market_data_readiness("AAPL", ["quote"])

        assert result.proceed is True
        assert result.reason_codes == ()

    @patch("utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_telemetry_failure_does_not_prevent_blocking(self):
        """If telemetry recording fails, the block result is still returned."""
        mock_layer = MagicMock()
        mock_layer.check_candidate_readiness.return_value = CandidateReadiness(
            ready=False,
            missing_data_types=("quote",),
            reason_codes=("quote_stale",),
            snapshots={},
        )
        # Make telemetry recording fail
        mock_layer._telemetry.record_fail_closed.side_effect = RuntimeError("telemetry down")

        with patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=mock_layer,
        ):
            result = check_market_data_readiness("AAPL", ["quote"])

        # Should still block — telemetry failure is non-fatal
        assert result.proceed is False
        assert "quote_stale" in result.reason_codes


# ---------------------------------------------------------------------------
# Consumer parameter
# ---------------------------------------------------------------------------


class TestConsumerParameter:
    """Verify consumer parameter is passed through correctly."""

    @patch("utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_default_consumer_is_pm(self):
        """Default consumer is 'PM'."""
        mock_layer = MagicMock()
        mock_layer.check_candidate_readiness.return_value = CandidateReadiness(
            ready=True, missing_data_types=(), reason_codes=(), snapshots={}
        )

        with patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=mock_layer,
        ):
            check_market_data_readiness("AAPL", ["quote"])

        mock_layer.check_candidate_readiness.assert_called_once_with(
            symbol="AAPL",
            required_data_types=["quote"],
            consumer="PM",
        )

    @patch("utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_custom_consumer_passed_through(self):
        """Custom consumer is passed through to the layer."""
        mock_layer = MagicMock()
        mock_layer.check_candidate_readiness.return_value = CandidateReadiness(
            ready=True, missing_data_types=(), reason_codes=(), snapshots={}
        )

        with patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=mock_layer,
        ):
            check_market_data_readiness("AAPL", ["quote"], consumer="Risk_Geometry_Gate")

        mock_layer.check_candidate_readiness.assert_called_once_with(
            symbol="AAPL",
            required_data_types=["quote"],
            consumer="Risk_Geometry_Gate",
        )


# ---------------------------------------------------------------------------
# PM is never asked to invent missing prices (Requirement 7.5)
# ---------------------------------------------------------------------------


class TestPMNeverEstimatesMissingPrices:
    """Verify that the integration prevents PM from estimating missing data.

    When market data is unavailable/untrusted, the pipeline blocks the candidate
    rather than passing degraded data to PM for it to invent or estimate.
    """

    @patch("utils.gate_config.MARKET_DATA_RELIABILITY_MODE", "enforcing")
    def test_unavailable_data_blocks_not_estimates(self):
        """Unavailable data causes block, not an estimate request to PM."""
        mock_layer = MagicMock()
        mock_layer.check_candidate_readiness.return_value = CandidateReadiness(
            ready=False,
            missing_data_types=("quote", "atr"),
            reason_codes=("market_data_unavailable", "atr_stale"),
            snapshots={},
        )

        with patch(
            "utils.market_data_reliability.pipeline_integration._get_reliability_layer",
            return_value=mock_layer,
        ):
            result = check_market_data_readiness("AAPL", ["quote", "atr"])

        # Candidate is blocked — PM never sees it
        assert result.proceed is False
        # Reason codes are structured, not "estimate" or "interpolate"
        assert all(
            code in ("market_data_unavailable", "atr_stale", "quote_stale",
                     "volume_unavailable", "provider_rate_limited")
            for code in result.reason_codes
        )
