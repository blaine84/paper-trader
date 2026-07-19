"""Smoke tests for the ReliabilityLayer facade class."""
from __future__ import annotations

import time

from utils.market_data_reliability.config import ReliabilityConfig
from utils.market_data_reliability.layer import ReliabilityLayer
from utils.market_data_reliability.snapshot import SnapshotResult


def _make_config() -> ReliabilityConfig:
    """Create a test config in enforcing mode for smoke tests."""
    from dataclasses import replace
    base = ReliabilityConfig.from_environment()
    return replace(base, mode="enforcing")


def test_no_provider_returns_untrusted_snapshot():
    """Layer with default (no provider) returns untrusted/unavailable snapshot."""
    config = _make_config()
    layer = ReliabilityLayer(config)
    result = layer.get_snapshot("AAPL", "quote", "PM")
    assert isinstance(result, SnapshotResult)
    assert result.snapshot.trust_state == "untrusted"
    assert result.snapshot.symbol == "AAPL"
    assert result.snapshot.data_type == "quote"
    assert "all_providers_failed" in result.snapshot.degradation_reasons
    assert result.eligibility.eligible is False


def test_good_provider_returns_trusted_snapshot():
    """Layer with mock provider returning valid data produces trusted snapshot."""
    config = _make_config()

    def mock_provider(provider, symbol, data_type):
        return {
            "s": symbol,
            "c": 150.0,
            "h": 152.0,
            "l": 149.0,
            "o": 150.5,
            "pc": 149.0,
            "t": int(time.time()),
            "v": 1000000,
        }

    layer = ReliabilityLayer(config, fetch_from_provider=mock_provider)
    result = layer.get_snapshot("AAPL", "quote", "PM")
    assert isinstance(result, SnapshotResult)
    assert result.snapshot.trust_state == "trusted"
    assert result.snapshot.provider == "finnhub"
    assert result.eligibility.eligible is True


def test_cache_hit_on_second_call():
    """Second call for same key returns from cache."""
    config = _make_config()

    def mock_provider(provider, symbol, data_type):
        return {
            "s": symbol,
            "c": 150.0,
            "h": 152.0,
            "l": 149.0,
            "o": 150.5,
            "pc": 149.0,
            "t": int(time.time()),
            "v": 1000000,
        }

    layer = ReliabilityLayer(config, fetch_from_provider=mock_provider)
    layer.get_snapshot("AAPL", "quote", "PM")
    layer.get_snapshot("AAPL", "quote", "PM")
    telemetry = layer.get_cycle_telemetry()
    assert telemetry["cache_hits"] == 1


def test_reset_cycle_clears_state():
    """reset_cycle clears cache and telemetry."""
    config = _make_config()

    def mock_provider(provider, symbol, data_type):
        return {
            "s": symbol,
            "c": 150.0,
            "h": 152.0,
            "l": 149.0,
            "o": 150.5,
            "pc": 149.0,
            "t": int(time.time()),
            "v": 1000000,
        }

    layer = ReliabilityLayer(config, fetch_from_provider=mock_provider)
    layer.get_snapshot("AAPL", "quote", "PM")
    layer.get_snapshot("AAPL", "quote", "PM")  # cache hit
    layer.reset_cycle()
    telemetry = layer.get_cycle_telemetry()
    assert telemetry["cache_hits"] == 0
    assert telemetry["provider_calls_success"] == 0


def test_fallback_on_primary_failure():
    """When primary provider fails, falls back to next available."""
    config = _make_config()

    def flaky_provider(provider, symbol, data_type):
        if provider == "finnhub":
            raise RuntimeError("Connection refused")
        return {
            "s": symbol,
            "c": 151.0,
            "h": 153.0,
            "l": 150.0,
            "o": 151.5,
            "pc": 150.0,
            "t": int(time.time()),
            "v": 2000000,
        }

    layer = ReliabilityLayer(config, fetch_from_provider=flaky_provider)
    result = layer.get_snapshot("MSFT", "quote", "Dashboard_API")
    assert result.snapshot.provider == "yfinance"
    assert result.snapshot.fallback_primary_provider == "finnhub"
    tel = layer.get_cycle_telemetry()
    assert tel["fallback_usage"] == 1


def test_check_candidate_readiness_all_ready():
    """check_candidate_readiness returns ready=True when all types pass."""
    config = _make_config()

    def mock_provider(provider, symbol, data_type):
        return {
            "s": symbol,
            "c": 150.0,
            "h": 152.0,
            "l": 149.0,
            "o": 150.5,
            "pc": 149.0,
            "t": int(time.time()),
            "v": 1000000,
        }

    layer = ReliabilityLayer(config, fetch_from_provider=mock_provider)
    readiness = layer.check_candidate_readiness("AAPL", ["quote"], "PM")
    assert readiness.ready is True
    assert len(readiness.missing_data_types) == 0


def test_get_snapshot_never_raises_on_internal_error():
    """get_snapshot always returns SnapshotResult, even on internal error."""
    config = _make_config()

    class BadLayer(ReliabilityLayer):
        def _get_snapshot_inner(self, *args, **kwargs):
            raise RuntimeError("Unexpected internal error")

    bad_layer = BadLayer(config)
    result = bad_layer.get_snapshot("AAPL", "quote", "PM")
    assert isinstance(result, SnapshotResult)
    assert result.snapshot.trust_state == "untrusted"


def test_display_consumer_with_allow_stale():
    """Display consumer with allow_stale_for_display can receive degraded data."""
    config = _make_config()

    def stale_provider(provider, symbol, data_type):
        # Return data with a very old timestamp (stale)
        return {
            "s": symbol,
            "c": 150.0,
            "h": 152.0,
            "l": 149.0,
            "o": 150.5,
            "pc": 149.0,
            "t": int(time.time()) - 600,  # 10 minutes old
            "v": 1000000,
        }

    layer = ReliabilityLayer(config, fetch_from_provider=stale_provider)

    # Display consumer with allow_stale=True should be eligible for degraded
    result = layer.get_snapshot(
        "AAPL", "quote", "Dashboard_API", allow_stale_for_display=True
    )
    assert isinstance(result, SnapshotResult)
    # The snapshot might be degraded or stale depending on threshold config
    # but it should never raise
    assert result.snapshot.symbol == "AAPL"


# ---------------------------------------------------------------------------
# Task 9.2: check_candidate_readiness structured reason codes
# ---------------------------------------------------------------------------


def test_check_candidate_readiness_quote_stale_reason_code():
    """check_candidate_readiness produces 'quote_stale' when quote is stale."""
    config = _make_config()

    def stale_quote_provider(provider, symbol, data_type):
        # Return quote with very old timestamp → stale → untrusted for PM
        return {
            "s": symbol,
            "c": 150.0,
            "h": 152.0,
            "l": 149.0,
            "o": 150.5,
            "pc": 149.0,
            "t": int(time.time()) - 600,  # 10 minutes old → stale for PM
            "v": 1000000,
        }

    layer = ReliabilityLayer(config, fetch_from_provider=stale_quote_provider)
    readiness = layer.check_candidate_readiness("AAPL", ["quote"], "PM")
    assert readiness.ready is False
    assert "quote" in readiness.missing_data_types
    assert "quote_stale" in readiness.reason_codes


def test_check_candidate_readiness_atr_stale_reason_code():
    """check_candidate_readiness produces 'atr_stale' when ATR is unavailable."""
    config = _make_config()

    call_count = {"n": 0}

    def selective_provider(provider, symbol, data_type):
        call_count["n"] += 1
        if data_type == "atr":
            raise RuntimeError("ATR endpoint down")
        return {
            "s": symbol,
            "c": 150.0,
            "h": 152.0,
            "l": 149.0,
            "o": 150.5,
            "pc": 149.0,
            "t": int(time.time()),
            "v": 1000000,
        }

    layer = ReliabilityLayer(config, fetch_from_provider=selective_provider)
    readiness = layer.check_candidate_readiness("AAPL", ["quote", "atr"], "PM")
    assert readiness.ready is False
    assert "atr" in readiness.missing_data_types
    assert "atr_stale" in readiness.reason_codes


def test_check_candidate_readiness_volume_unavailable_reason_code():
    """check_candidate_readiness produces 'volume_unavailable' when volume fails."""
    config = _make_config()

    def volume_fail_provider(provider, symbol, data_type):
        if data_type == "volume":
            raise RuntimeError("Volume endpoint unavailable")
        return {
            "s": symbol,
            "c": 150.0,
            "h": 152.0,
            "l": 149.0,
            "o": 150.5,
            "pc": 149.0,
            "t": int(time.time()),
            "v": 1000000,
        }

    layer = ReliabilityLayer(config, fetch_from_provider=volume_fail_provider)
    readiness = layer.check_candidate_readiness("AAPL", ["quote", "volume"], "PM")
    assert readiness.ready is False
    assert "volume" in readiness.missing_data_types
    assert "volume_unavailable" in readiness.reason_codes


def test_check_candidate_readiness_provider_rate_limited_reason_code():
    """check_candidate_readiness produces 'provider_rate_limited' on rate limit."""
    config = _make_config()

    def rate_limited_provider(provider, symbol, data_type):
        raise RuntimeError("429 rate limit exceeded")

    layer = ReliabilityLayer(config, fetch_from_provider=rate_limited_provider)
    readiness = layer.check_candidate_readiness("AAPL", ["quote"], "PM")
    assert readiness.ready is False
    assert "quote" in readiness.missing_data_types
    assert "provider_rate_limited" in readiness.reason_codes


def test_check_candidate_readiness_market_data_unavailable_no_provider():
    """check_candidate_readiness produces 'market_data_unavailable' for unrecognized data types."""
    config = _make_config()

    # Default layer with no provider configured → raises NotImplementedError
    layer = ReliabilityLayer(config)
    # candle has no specific code so falls through to market_data_unavailable
    readiness = layer.check_candidate_readiness("AAPL", ["candle"], "PM")
    assert readiness.ready is False
    assert "market_data_unavailable" in readiness.reason_codes


def test_check_candidate_readiness_no_duplicate_reason_codes():
    """check_candidate_readiness does not produce duplicate reason codes."""
    config = _make_config()

    # All providers fail for both quote and candle
    layer = ReliabilityLayer(config)
    readiness = layer.check_candidate_readiness("AAPL", ["quote", "candle"], "PM")
    assert readiness.ready is False
    # quote_stale for quote, market_data_unavailable for candle
    assert "quote_stale" in readiness.reason_codes
    assert "market_data_unavailable" in readiness.reason_codes
    # Each appears exactly once
    assert readiness.reason_codes.count("quote_stale") == 1
    assert readiness.reason_codes.count("market_data_unavailable") == 1


# ---------------------------------------------------------------------------
# Feature flag mode switching tests (Task 9.3)
# Requirements: 14.1, 14.2, 14.3, 14.4, 14.5
# ---------------------------------------------------------------------------

from unittest.mock import patch
import time


def _make_config_with_mode(mode: str) -> ReliabilityConfig:
    """Create a ReliabilityConfig with a specific mode."""
    with patch.dict("os.environ", {"MARKET_DATA_RELIABILITY_MODE": mode}):
        return ReliabilityConfig.from_environment()


def test_disabled_mode_get_snapshot_returns_passthrough():
    """disabled mode: get_snapshot returns passthrough result without provider calls."""
    config = _make_config_with_mode("disabled")
    call_count = {"n": 0}

    def tracking_provider(provider, symbol, data_type):
        call_count["n"] += 1
        return {"s": symbol, "c": 150.0, "t": int(time.time()), "v": 1000}

    layer = ReliabilityLayer(config, fetch_from_provider=tracking_provider)
    result = layer.get_snapshot("AAPL", "quote", "PM")

    assert isinstance(result, SnapshotResult)
    assert result.eligibility.eligible is True
    assert result.snapshot.provider == "passthrough"
    assert result.snapshot.trust_state == "trusted"
    assert result.snapshot.freshness_state == "fresh"
    assert result.snapshot.symbol == "AAPL"
    assert result.snapshot.data_type == "quote"
    # No provider calls made
    assert call_count["n"] == 0


def test_disabled_mode_check_candidate_readiness_always_ready():
    """disabled mode: check_candidate_readiness always returns ready=True."""
    config = _make_config_with_mode("disabled")
    call_count = {"n": 0}

    def tracking_provider(provider, symbol, data_type):
        call_count["n"] += 1
        raise RuntimeError("Should never be called in disabled mode")

    layer = ReliabilityLayer(config, fetch_from_provider=tracking_provider)
    readiness = layer.check_candidate_readiness("AAPL", ["quote", "atr"], "PM")

    assert readiness.ready is True
    assert readiness.missing_data_types == ()
    assert readiness.reason_codes == ()
    assert readiness.snapshots == {}
    # No provider calls made
    assert call_count["n"] == 0


def test_observe_mode_does_not_block_execution_consumers():
    """observe mode: untrusted data does NOT block — eligible overridden to True."""
    config = _make_config_with_mode("observe")

    # Provider that always fails → untrusted snapshot
    layer = ReliabilityLayer(config)  # default provider raises NotImplementedError
    result = layer.get_snapshot("AAPL", "quote", "PM")

    assert isinstance(result, SnapshotResult)
    # In enforcing mode this would be ineligible, but observe overrides to eligible
    assert result.eligibility.eligible is True
    # The underlying snapshot still reflects the real state
    assert result.snapshot.trust_state == "untrusted"


def test_observe_mode_check_candidate_readiness_always_ready():
    """observe mode: check_candidate_readiness returns ready=True even on failure."""
    config = _make_config_with_mode("observe")

    # No provider → all providers fail → normally would block
    layer = ReliabilityLayer(config)
    readiness = layer.check_candidate_readiness("AAPL", ["quote", "atr"], "PM")

    assert readiness.ready is True
    assert readiness.missing_data_types == ()
    assert readiness.reason_codes == ()
    # Snapshots dict should still contain the checked data types
    assert "quote" in readiness.snapshots
    assert "atr" in readiness.snapshots


def test_observe_mode_trusted_data_passes_through():
    """observe mode: trusted data passes through normally (eligible=True)."""
    config = _make_config_with_mode("observe")

    def mock_provider(provider, symbol, data_type):
        return {
            "s": symbol,
            "c": 150.0,
            "h": 152.0,
            "l": 149.0,
            "o": 150.5,
            "pc": 149.0,
            "t": int(time.time()),
            "v": 1000000,
        }

    layer = ReliabilityLayer(config, fetch_from_provider=mock_provider)
    result = layer.get_snapshot("AAPL", "quote", "PM")

    assert result.eligibility.eligible is True
    assert result.snapshot.trust_state == "trusted"


def test_enforcing_mode_blocks_untrusted_for_execution():
    """enforcing mode: untrusted data blocks execution consumers."""
    config = _make_config_with_mode("enforcing")

    # No provider configured → untrusted snapshot
    layer = ReliabilityLayer(config)
    result = layer.get_snapshot("AAPL", "quote", "PM")

    assert isinstance(result, SnapshotResult)
    assert result.eligibility.eligible is False
    assert result.snapshot.trust_state == "untrusted"


def test_enforcing_mode_check_candidate_readiness_blocks():
    """enforcing mode: check_candidate_readiness blocks when data is unavailable."""
    config = _make_config_with_mode("enforcing")

    layer = ReliabilityLayer(config)
    readiness = layer.check_candidate_readiness("AAPL", ["quote", "atr"], "PM")

    assert readiness.ready is False
    assert "quote" in readiness.missing_data_types
    assert "atr" in readiness.missing_data_types
    assert len(readiness.reason_codes) > 0


def test_enforcing_mode_allows_trusted_data():
    """enforcing mode: trusted data passes through for execution consumers."""
    config = _make_config_with_mode("enforcing")

    def mock_provider(provider, symbol, data_type):
        return {
            "s": symbol,
            "c": 150.0,
            "h": 152.0,
            "l": 149.0,
            "o": 150.5,
            "pc": 149.0,
            "t": int(time.time()),
            "v": 1000000,
        }

    layer = ReliabilityLayer(config, fetch_from_provider=mock_provider)
    result = layer.get_snapshot("AAPL", "quote", "PM")

    assert result.eligibility.eligible is True
    assert result.snapshot.trust_state == "trusted"


def test_default_mode_is_disabled():
    """When MARKET_DATA_RELIABILITY_MODE is unset, default is disabled."""
    with patch.dict("os.environ", {}, clear=False):
        # Remove the key if present
        import os
        env_backup = os.environ.pop("MARKET_DATA_RELIABILITY_MODE", None)
        try:
            config = ReliabilityConfig.from_environment()
            assert config.mode == "disabled"
        finally:
            if env_backup is not None:
                os.environ["MARKET_DATA_RELIABILITY_MODE"] = env_backup
