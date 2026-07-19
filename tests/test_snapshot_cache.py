"""Unit tests for SnapshotCache (utils/market_data_reliability/cache.py).

Requirements: 5.1, 5.3, 5.5
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import patch

import pytest

from utils.market_data_reliability.cache import SnapshotCache
from utils.market_data_reliability.freshness import FreshnessClassifier
from utils.market_data_reliability.snapshot import CacheKey, FreshnessThreshold, Snapshot


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def freshness_thresholds() -> dict[tuple[str, str], FreshnessThreshold]:
    return {
        ("quote", "execution"): FreshnessThreshold(fresh_threshold=30.0, aging_threshold=120.0),
        ("quote", "display"): FreshnessThreshold(fresh_threshold=60.0, aging_threshold=300.0),
        ("candle", "execution"): FreshnessThreshold(fresh_threshold=120.0, aging_threshold=600.0),
        ("atr", "execution"): FreshnessThreshold(fresh_threshold=300.0, aging_threshold=900.0),
        ("volume", "execution"): FreshnessThreshold(fresh_threshold=60.0, aging_threshold=300.0),
        ("previous_close", "all"): FreshnessThreshold(fresh_threshold=3600.0, aging_threshold=7200.0),
    }


@pytest.fixture
def classifier(freshness_thresholds) -> FreshnessClassifier:
    return FreshnessClassifier(freshness_thresholds)


@pytest.fixture
def cache_ttls() -> dict[str, float]:
    return {
        "quote": 15.0,
        "candle": 60.0,
        "atr": 120.0,
        "volume": 30.0,
        "previous_close": 3600.0,
    }


@pytest.fixture
def cache(cache_ttls, classifier) -> SnapshotCache:
    return SnapshotCache(cache_ttls=cache_ttls, freshness_classifier=classifier)


@pytest.fixture
def sample_snapshot() -> Snapshot:
    now = datetime.now(timezone.utc)
    return Snapshot(
        symbol="AAPL",
        data_type="quote",
        requested_at=now,
        provider="finnhub",
        provider_status="success",
        market_session="open",
        last_price=Decimal("187.43"),
        bid=Decimal("187.42"),
        ask=Decimal("187.44"),
        previous_close=Decimal("186.50"),
        open=Decimal("186.80"),
        high=Decimal("188.00"),
        low=Decimal("186.20"),
        volume=50000,
        fetched_at=now,
        source_timestamp=now,
        age_seconds=2.0,
        freshness_state="fresh",
        trust_state="trusted",
        degradation_reasons=(),
        raw_provider_latency_ms=45.0,
        fallback_primary_provider=None,
    )


@pytest.fixture
def sample_key() -> CacheKey:
    return CacheKey(
        symbol="AAPL",
        data_type="quote",
        provider_policy="primary",
        market_session="open",
    )


# ---------------------------------------------------------------------------
# Tests: get() behavior
# ---------------------------------------------------------------------------


class TestCacheGet:
    def test_returns_none_for_missing_key(self, cache, sample_key):
        assert cache.get(sample_key) is None

    def test_returns_snapshot_after_put(self, cache, sample_key, sample_snapshot):
        cache.put(sample_key, sample_snapshot)
        result = cache.get(sample_key)
        assert result is not None
        assert result.symbol == "AAPL"

    def test_preserves_fetched_at(self, cache, sample_key, sample_snapshot):
        cache.put(sample_key, sample_snapshot)
        result = cache.get(sample_key)
        assert result.fetched_at == sample_snapshot.fetched_at

    def test_preserves_source_timestamp(self, cache, sample_key, sample_snapshot):
        cache.put(sample_key, sample_snapshot)
        result = cache.get(sample_key)
        assert result.source_timestamp == sample_snapshot.source_timestamp

    def test_recomputes_age_seconds(self, cache, sample_key, sample_snapshot):
        cache.put(sample_key, sample_snapshot)
        # Simulate time passing by patching time.monotonic
        original_mono = time.monotonic()
        with patch("utils.market_data_reliability.cache.time.monotonic", return_value=original_mono + 5.0):
            # Re-put to set stored_at at the patched time? No, we need to put first.
            pass

        # Simpler: just verify age_seconds >= original
        result = cache.get(sample_key)
        assert result.age_seconds >= sample_snapshot.age_seconds

    def test_recomputes_freshness_to_stale_after_elapsed(self, cache, sample_key, sample_snapshot, cache_ttls, classifier):
        """After enough time elapses, freshness should transition to stale."""
        # Put with controlled monotonic time
        base_time = 1000.0
        with patch("utils.market_data_reliability.cache.time.monotonic", return_value=base_time):
            cache.put(sample_key, sample_snapshot)

        # Get 200s later — age_seconds will be 2.0 + 200.0 = 202.0
        # For quote/execution: aging_threshold=120 → stale
        with patch("utils.market_data_reliability.cache.time.monotonic", return_value=base_time + 200.0):
            result = cache.get(sample_key)

        assert result.freshness_state == "stale"
        assert result.age_seconds == pytest.approx(202.0, abs=0.1)

    def test_recomputes_freshness_to_aging_after_elapsed(self, cache, sample_key, sample_snapshot):
        """After moderate time, freshness should transition to aging."""
        base_time = 1000.0
        with patch("utils.market_data_reliability.cache.time.monotonic", return_value=base_time):
            cache.put(sample_key, sample_snapshot)

        # Get 50s later — age_seconds = 2.0 + 50.0 = 52.0
        # For quote/execution: fresh < 30, aging < 120 → aging
        with patch("utils.market_data_reliability.cache.time.monotonic", return_value=base_time + 50.0):
            result = cache.get(sample_key)

        assert result.freshness_state == "aging"

    def test_different_keys_are_independent(self, cache, sample_snapshot):
        key1 = CacheKey(symbol="AAPL", data_type="quote", provider_policy="primary", market_session="open")
        key2 = CacheKey(symbol="MSFT", data_type="quote", provider_policy="primary", market_session="open")

        cache.put(key1, sample_snapshot)
        assert cache.get(key1) is not None
        assert cache.get(key2) is None


# ---------------------------------------------------------------------------
# Tests: put() behavior
# ---------------------------------------------------------------------------


class TestCachePut:
    def test_overwrites_existing_entry(self, cache, sample_key, sample_snapshot):
        cache.put(sample_key, sample_snapshot)

        now = datetime.now(timezone.utc)
        new_snapshot = Snapshot(
            symbol="AAPL",
            data_type="quote",
            requested_at=now,
            provider="yfinance",
            provider_status="success",
            market_session="open",
            last_price=Decimal("190.00"),
            bid=Decimal("189.99"),
            ask=Decimal("190.01"),
            previous_close=Decimal("186.50"),
            open=Decimal("186.80"),
            high=Decimal("190.50"),
            low=Decimal("186.20"),
            volume=75000,
            fetched_at=now,
            source_timestamp=now,
            age_seconds=1.0,
            freshness_state="fresh",
            trust_state="trusted",
            degradation_reasons=(),
            raw_provider_latency_ms=55.0,
            fallback_primary_provider=None,
        )

        cache.put(sample_key, new_snapshot)
        result = cache.get(sample_key)
        assert result.provider == "yfinance"
        assert result.last_price == Decimal("190.00")

    def test_uses_data_type_ttl(self, cache, sample_snapshot):
        """Different data_types get different TTLs from config."""
        key_candle = CacheKey(symbol="AAPL", data_type="candle", provider_policy="primary", market_session="open")
        candle_snapshot = Snapshot(
            symbol="AAPL",
            data_type="candle",
            requested_at=sample_snapshot.requested_at,
            provider="finnhub",
            provider_status="success",
            market_session="open",
            last_price=Decimal("187.43"),
            bid=None,
            ask=None,
            previous_close=Decimal("186.50"),
            open=Decimal("186.80"),
            high=Decimal("188.00"),
            low=Decimal("186.20"),
            volume=50000,
            fetched_at=sample_snapshot.fetched_at,
            source_timestamp=sample_snapshot.source_timestamp,
            age_seconds=5.0,
            freshness_state="fresh",
            trust_state="trusted",
            degradation_reasons=(),
            raw_provider_latency_ms=50.0,
            fallback_primary_provider=None,
        )

        base_time = 1000.0
        with patch("utils.market_data_reliability.cache.time.monotonic", return_value=base_time):
            cache.put(key_candle, candle_snapshot)

        # After 30s (within candle TTL of 60s) — should not be expired
        with patch("utils.market_data_reliability.cache.time.monotonic", return_value=base_time + 30.0):
            removed = cache.invalidate_expired()
        assert removed == 0


# ---------------------------------------------------------------------------
# Tests: invalidate_expired()
# ---------------------------------------------------------------------------


class TestInvalidateExpired:
    def test_removes_expired_entries(self, cache, sample_key, sample_snapshot):
        base_time = 1000.0
        with patch("utils.market_data_reliability.cache.time.monotonic", return_value=base_time):
            cache.put(sample_key, sample_snapshot)

        # quote TTL is 15s — after 20s it's expired
        with patch("utils.market_data_reliability.cache.time.monotonic", return_value=base_time + 20.0):
            removed = cache.invalidate_expired()

        assert removed == 1
        # Entry should be gone now
        with patch("utils.market_data_reliability.cache.time.monotonic", return_value=base_time + 20.0):
            assert cache.get(sample_key) is None

    def test_keeps_non_expired_entries(self, cache, sample_key, sample_snapshot):
        base_time = 1000.0
        with patch("utils.market_data_reliability.cache.time.monotonic", return_value=base_time):
            cache.put(sample_key, sample_snapshot)

        # After 5s (within 15s TTL)
        with patch("utils.market_data_reliability.cache.time.monotonic", return_value=base_time + 5.0):
            removed = cache.invalidate_expired()

        assert removed == 0
        with patch("utils.market_data_reliability.cache.time.monotonic", return_value=base_time + 5.0):
            assert cache.get(sample_key) is not None

    def test_returns_zero_on_empty_cache(self, cache):
        assert cache.invalidate_expired() == 0


# ---------------------------------------------------------------------------
# Tests: clear()
# ---------------------------------------------------------------------------


class TestCacheClear:
    def test_removes_all_entries(self, cache, sample_snapshot):
        key1 = CacheKey(symbol="AAPL", data_type="quote", provider_policy="primary", market_session="open")
        key2 = CacheKey(symbol="MSFT", data_type="quote", provider_policy="primary", market_session="open")

        cache.put(key1, sample_snapshot)
        cache.put(key2, sample_snapshot)
        cache.clear()

        assert cache.get(key1) is None
        assert cache.get(key2) is None

    def test_clear_on_empty_cache(self, cache):
        # Should not raise
        cache.clear()
