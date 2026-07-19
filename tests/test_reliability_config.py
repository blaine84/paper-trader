"""Unit tests for ReliabilityConfig environment variable loading."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from utils.market_data_reliability.config import ReliabilityConfig
from utils.market_data_reliability.snapshot import FreshnessThreshold


class TestReliabilityConfigDefaults:
    """Verify safe defaults when no env vars are set."""

    def test_default_mode_is_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = ReliabilityConfig.from_environment()
            assert cfg.mode == "disabled"

    def test_default_freshness_thresholds_count(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = ReliabilityConfig.from_environment()
            assert len(cfg.freshness_thresholds) == 8

    def test_default_quote_execution_thresholds(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = ReliabilityConfig.from_environment()
            threshold = cfg.freshness_thresholds[("quote", "execution")]
            assert threshold.fresh_threshold == 30.0
            assert threshold.aging_threshold == 120.0

    def test_default_cache_ttls(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = ReliabilityConfig.from_environment()
            assert cfg.cache_ttls["quote"] == 15.0
            assert cfg.cache_ttls["candle"] == 60.0
            assert cfg.cache_ttls["atr"] == 120.0
            assert cfg.cache_ttls["volume"] == 30.0
            assert cfg.cache_ttls["previous_close"] == 3600.0

    def test_default_backoff_durations(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = ReliabilityConfig.from_environment()
            assert cfg.backoff_durations["rate_limit"] == 60.0
            assert cfg.backoff_durations["network_error"] == 30.0
            assert cfg.backoff_durations["empty_response"] == 15.0

    def test_default_provider_timeouts(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = ReliabilityConfig.from_environment()
            assert cfg.provider_timeouts["finnhub"] == 10.0
            assert cfg.provider_timeouts["yfinance"] == 15.0
            assert cfg.provider_timeouts["alpaca"] == 10.0

    def test_default_provider_retry_limits(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = ReliabilityConfig.from_environment()
            assert cfg.provider_retry_limits["finnhub"] == 2
            assert cfg.provider_retry_limits["yfinance"] == 2
            assert cfg.provider_retry_limits["alpaca"] == 2

    def test_default_fallback_matrix_has_entries(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = ReliabilityConfig.from_environment()
            assert len(cfg.fallback_matrix) > 0
            assert ("quote", "execution") in cfg.fallback_matrix

    def test_config_is_frozen(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = ReliabilityConfig.from_environment()
            with pytest.raises(Exception):
                cfg.mode = "enforcing"  # type: ignore[misc]


class TestReliabilityConfigEnvOverrides:
    """Verify environment variable overrides work correctly."""

    def test_mode_observe(self):
        with patch.dict(os.environ, {"MARKET_DATA_RELIABILITY_MODE": "observe"}):
            cfg = ReliabilityConfig.from_environment()
            assert cfg.mode == "observe"

    def test_mode_enforcing(self):
        with patch.dict(os.environ, {"MARKET_DATA_RELIABILITY_MODE": "enforcing"}):
            cfg = ReliabilityConfig.from_environment()
            assert cfg.mode == "enforcing"

    def test_freshness_threshold_override(self):
        env = {
            "MDR_FRESHNESS_QUOTE_EXECUTION_FRESH": "20",
            "MDR_FRESHNESS_QUOTE_EXECUTION_AGING": "90",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = ReliabilityConfig.from_environment()
            threshold = cfg.freshness_thresholds[("quote", "execution")]
            assert threshold.fresh_threshold == 20.0
            assert threshold.aging_threshold == 90.0

    def test_cache_ttl_override(self):
        with patch.dict(os.environ, {"MDR_CACHE_TTL_QUOTE": "30"}, clear=True):
            cfg = ReliabilityConfig.from_environment()
            assert cfg.cache_ttls["quote"] == 30.0

    def test_provider_timeout_override(self):
        with patch.dict(os.environ, {"MDR_PROVIDER_TIMEOUT_FINNHUB": "5"}, clear=True):
            cfg = ReliabilityConfig.from_environment()
            assert cfg.provider_timeouts["finnhub"] == 5.0

    def test_provider_retries_override(self):
        with patch.dict(os.environ, {"MDR_PROVIDER_RETRIES_FINNHUB": "5"}, clear=True):
            cfg = ReliabilityConfig.from_environment()
            assert cfg.provider_retry_limits["finnhub"] == 5

    def test_backoff_duration_override(self):
        with patch.dict(os.environ, {"MDR_BACKOFF_RATE_LIMIT": "120"}, clear=True):
            cfg = ReliabilityConfig.from_environment()
            assert cfg.backoff_durations["rate_limit"] == 120.0

    def test_fallback_matrix_override(self):
        env = {"MDR_FALLBACK_QUOTE_EXECUTION": "alpaca,finnhub"}
        with patch.dict(os.environ, env, clear=True):
            cfg = ReliabilityConfig.from_environment()
            assert cfg.fallback_matrix[("quote", "execution")] == ["alpaca", "finnhub"]


class TestReliabilityConfigInvalidValues:
    """Verify invalid env vars produce safe defaults (fail-closed)."""

    def test_invalid_mode_defaults_to_disabled(self):
        with patch.dict(os.environ, {"MARKET_DATA_RELIABILITY_MODE": "bogus"}):
            cfg = ReliabilityConfig.from_environment()
            assert cfg.mode == "disabled"

    def test_non_numeric_freshness_uses_default(self):
        env = {"MDR_FRESHNESS_QUOTE_EXECUTION_FRESH": "not_a_number"}
        with patch.dict(os.environ, env, clear=True):
            cfg = ReliabilityConfig.from_environment()
            threshold = cfg.freshness_thresholds[("quote", "execution")]
            assert threshold.fresh_threshold == 30.0

    def test_negative_cache_ttl_uses_default(self):
        with patch.dict(os.environ, {"MDR_CACHE_TTL_QUOTE": "-5"}, clear=True):
            cfg = ReliabilityConfig.from_environment()
            assert cfg.cache_ttls["quote"] == 15.0

    def test_zero_timeout_uses_default(self):
        with patch.dict(os.environ, {"MDR_PROVIDER_TIMEOUT_FINNHUB": "0"}, clear=True):
            cfg = ReliabilityConfig.from_environment()
            assert cfg.provider_timeouts["finnhub"] == 10.0

    def test_non_numeric_backoff_uses_default(self):
        with patch.dict(os.environ, {"MDR_BACKOFF_RATE_LIMIT": "abc"}, clear=True):
            cfg = ReliabilityConfig.from_environment()
            assert cfg.backoff_durations["rate_limit"] == 60.0

    def test_empty_fallback_uses_default(self):
        env = {"MDR_FALLBACK_QUOTE_EXECUTION": "  ,  , "}
        with patch.dict(os.environ, env, clear=True):
            cfg = ReliabilityConfig.from_environment()
            assert cfg.fallback_matrix[("quote", "execution")] == ["finnhub", "yfinance", "alpaca"]

    def test_aging_less_than_fresh_uses_defaults(self):
        """When aging < fresh (invalid invariant), revert both to safe defaults."""
        env = {
            "MDR_FRESHNESS_QUOTE_EXECUTION_FRESH": "200",
            "MDR_FRESHNESS_QUOTE_EXECUTION_AGING": "50",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = ReliabilityConfig.from_environment()
            threshold = cfg.freshness_thresholds[("quote", "execution")]
            # Should revert to safe defaults
            assert threshold.fresh_threshold == 30.0
            assert threshold.aging_threshold == 120.0

    def test_execution_thresholds_are_strict(self):
        """Execution consumers must have stricter thresholds than display."""
        with patch.dict(os.environ, {}, clear=True):
            cfg = ReliabilityConfig.from_environment()
            exec_thresh = cfg.freshness_thresholds[("quote", "execution")]
            disp_thresh = cfg.freshness_thresholds[("quote", "display")]
            # Execution fresh threshold must be <= display (stricter)
            assert exec_thresh.fresh_threshold <= disp_thresh.fresh_threshold
            assert exec_thresh.aging_threshold <= disp_thresh.aging_threshold
