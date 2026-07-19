"""Unit tests for SnapshotNormalizer.

Tests normalize_quote and normalize_candles with specific provider formats,
edge cases, and integration with ResponseValidator/FreshnessClassifier/TrustClassifier.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from decimal import Decimal

import pytest

from utils.market_data_reliability.config import _DEFAULT_FRESHNESS_THRESHOLDS
from utils.market_data_reliability.freshness import FreshnessClassifier
from utils.market_data_reliability.normalizer import (
    SnapshotNormalizer,
    _to_decimal,
    _to_int,
    _parse_unix_timestamp,
)
from utils.market_data_reliability.snapshot import Snapshot
from utils.market_data_reliability.trust import TrustClassifier
from utils.market_data_reliability.validator import ResponseValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def normalizer():
    """SnapshotNormalizer with default sub-components."""
    validator = ResponseValidator(staleness_threshold_seconds=300.0)
    freshness_classifier = FreshnessClassifier(_DEFAULT_FRESHNESS_THRESHOLDS)
    trust_classifier = TrustClassifier()
    return SnapshotNormalizer(
        validator=validator,
        freshness_classifier=freshness_classifier,
        trust_classifier=trust_classifier,
        default_market_session="open",
    )


@pytest.fixture
def now():
    """A 'now' timestamp for testing — uses actual current time so that
    the ResponseValidator's stale_source_timestamp check (which compares
    against the real clock) does not trip on test data."""
    return datetime.now(tz=timezone.utc)


@pytest.fixture
def finnhub_quote(now):
    """Standard Finnhub quote response."""
    # source_timestamp is 10 seconds before 'now'
    ts = int((now - timedelta(seconds=10)).timestamp())
    return {
        "s": "AAPL",
        "c": 187.43,
        "h": 188.50,
        "l": 186.20,
        "o": 187.00,
        "pc": 186.90,
        "t": ts,
        "v": 55000000,
    }


@pytest.fixture
def finnhub_candles(now):
    """Standard Finnhub candle response with multiple candles."""
    ts1 = int((now - timedelta(seconds=3600)).timestamp())
    ts2 = int((now - timedelta(seconds=20)).timestamp())
    return {
        "s": "AAPL",
        "c": [187.43, 188.10],
        "h": [188.50, 189.00],
        "l": [186.20, 187.00],
        "o": [187.00, 187.50],
        "t": [ts1, ts2],
        "v": [5000000, 6000000],
    }


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestToDecimal:
    """Tests for _to_decimal helper."""

    def test_float_converts(self):
        assert _to_decimal(187.43) == Decimal("187.43")

    def test_int_converts(self):
        assert _to_decimal(100) == Decimal("100")

    def test_string_converts(self):
        assert _to_decimal("42.5") == Decimal("42.5")

    def test_none_returns_none(self):
        assert _to_decimal(None) is None

    def test_invalid_string_returns_none(self):
        assert _to_decimal("not_a_number") is None

    def test_empty_string_returns_none(self):
        assert _to_decimal("") is None


class TestToInt:
    """Tests for _to_int helper."""

    def test_int_returns_int(self):
        assert _to_int(55000000) == 55000000

    def test_float_truncates(self):
        assert _to_int(5000.9) == 5000

    def test_string_converts(self):
        assert _to_int("123") == 123

    def test_none_returns_none(self):
        assert _to_int(None) is None

    def test_invalid_string_returns_none(self):
        assert _to_int("abc") is None


class TestParseUnixTimestamp:
    """Tests for _parse_unix_timestamp helper."""

    def test_valid_epoch(self):
        result = _parse_unix_timestamp(1689424200)
        assert result is not None
        assert result.tzinfo == timezone.utc

    def test_none_returns_none(self):
        assert _parse_unix_timestamp(None) is None

    def test_zero_returns_none(self):
        assert _parse_unix_timestamp(0) is None

    def test_negative_returns_none(self):
        assert _parse_unix_timestamp(-100) is None

    def test_string_epoch(self):
        result = _parse_unix_timestamp("1689424200")
        assert result is not None


# ---------------------------------------------------------------------------
# normalize_quote tests
# ---------------------------------------------------------------------------


class TestNormalizeQuote:
    """Tests for SnapshotNormalizer.normalize_quote()."""

    def test_basic_finnhub_quote(self, normalizer, finnhub_quote, now):
        requested_at = now - timedelta(milliseconds=150)
        snapshot = normalizer.normalize_quote(
            finnhub_quote, "AAPL", "finnhub", requested_at, now
        )

        assert isinstance(snapshot, Snapshot)
        assert snapshot.symbol == "AAPL"
        assert snapshot.data_type == "quote"
        assert snapshot.provider == "finnhub"
        assert snapshot.provider_status == "success"
        assert snapshot.market_session == "open"

    def test_price_fields_are_decimal(self, normalizer, finnhub_quote, now):
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_quote(
            finnhub_quote, "AAPL", "finnhub", requested_at, now
        )

        assert snapshot.last_price == Decimal("187.43")
        assert snapshot.high == Decimal("188.5")
        assert snapshot.low == Decimal("186.2")
        assert snapshot.open == Decimal("187.0")
        assert snapshot.previous_close == Decimal("186.9")
        assert isinstance(snapshot.last_price, Decimal)
        assert isinstance(snapshot.high, Decimal)

    def test_unavailable_fields_set_to_none(self, normalizer, finnhub_quote, now):
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_quote(
            finnhub_quote, "AAPL", "finnhub", requested_at, now
        )

        # Finnhub quote does not provide bid/ask
        assert snapshot.bid is None
        assert snapshot.ask is None
        # Fallback provider not used
        assert snapshot.fallback_primary_provider is None

    def test_volume_is_int(self, normalizer, finnhub_quote, now):
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_quote(
            finnhub_quote, "AAPL", "finnhub", requested_at, now
        )

        assert snapshot.volume == 55000000
        assert isinstance(snapshot.volume, int)

    def test_source_timestamp_parsed(self, normalizer, finnhub_quote, now):
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_quote(
            finnhub_quote, "AAPL", "finnhub", requested_at, now
        )

        assert snapshot.source_timestamp is not None
        assert snapshot.source_timestamp.tzinfo == timezone.utc

    def test_age_seconds_computed(self, normalizer, finnhub_quote, now):
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_quote(
            finnhub_quote, "AAPL", "finnhub", requested_at, now
        )

        # Source was 10 seconds before fetched_at
        assert abs(snapshot.age_seconds - 10.0) < 1.0

    def test_freshness_classified(self, normalizer, finnhub_quote, now):
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_quote(
            finnhub_quote, "AAPL", "finnhub", requested_at, now
        )

        # 10s age with execution quote thresholds (fresh < 30s)
        assert snapshot.freshness_state == "fresh"

    def test_trust_classified(self, normalizer, finnhub_quote, now):
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_quote(
            finnhub_quote, "AAPL", "finnhub", requested_at, now
        )

        assert snapshot.trust_state == "trusted"
        assert snapshot.degradation_reasons == ()

    def test_raw_provider_latency(self, normalizer, finnhub_quote, now):
        requested_at = now - timedelta(milliseconds=150)
        snapshot = normalizer.normalize_quote(
            finnhub_quote, "AAPL", "finnhub", requested_at, now
        )

        assert snapshot.raw_provider_latency_ms is not None
        assert abs(snapshot.raw_provider_latency_ms - 150.0) < 1.0

    def test_fetched_at_and_requested_at(self, normalizer, finnhub_quote, now):
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_quote(
            finnhub_quote, "AAPL", "finnhub", requested_at, now
        )

        assert snapshot.fetched_at == now
        assert snapshot.requested_at == requested_at

    def test_empty_response_produces_degraded_snapshot(self, normalizer, now):
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_quote(
            {}, "AAPL", "finnhub", requested_at, now
        )

        assert snapshot.provider_status == "empty"
        assert snapshot.last_price is None
        assert snapshot.high is None
        # empty_response is non-critical in trust classification → degraded
        assert snapshot.trust_state == "degraded"
        assert "empty_response" in snapshot.degradation_reasons

    def test_error_response_sets_price_fields_to_none(self, normalizer, now):
        raw = {"error": "Internal server error", "s": "AAPL"}
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_quote(
            raw, "AAPL", "finnhub", requested_at, now
        )

        assert snapshot.provider_status == "error"
        assert snapshot.last_price is None
        assert snapshot.open is None
        assert snapshot.high is None
        assert snapshot.low is None
        assert snapshot.previous_close is None
        assert snapshot.volume is None

    def test_rate_limited_response(self, normalizer, now):
        raw = {"error": "rate limit exceeded", "status": "429"}
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_quote(
            raw, "AAPL", "finnhub", requested_at, now
        )

        assert snapshot.provider_status == "rate_limited"
        assert snapshot.last_price is None

    def test_cross_symbol_produces_degraded_snapshot(self, normalizer, now):
        raw = {
            "s": "MSFT",
            "c": 350.00,
            "h": 355.00,
            "l": 348.00,
            "o": 349.00,
            "pc": 349.50,
            "t": int((now - timedelta(seconds=5)).timestamp()),
            "v": 30000000,
        }
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_quote(
            raw, "AAPL", "finnhub", requested_at, now
        )

        # cross_symbol_response is safety-critical → untrusted
        assert snapshot.trust_state == "untrusted"
        assert "cross_symbol_response" in snapshot.degradation_reasons

    def test_missing_timestamp_produces_degradation(self, normalizer, now):
        raw = {
            "s": "AAPL",
            "c": 187.43,
            "h": 188.50,
            "l": 186.20,
            "o": 187.00,
            "pc": 186.90,
            "v": 55000000,
            # no "t" field
        }
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_quote(
            raw, "AAPL", "finnhub", requested_at, now
        )

        assert snapshot.source_timestamp is None
        # missing_source_timestamp is non-critical → degraded
        assert "missing_source_timestamp" in snapshot.degradation_reasons

    def test_frozen_snapshot(self, normalizer, finnhub_quote, now):
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_quote(
            finnhub_quote, "AAPL", "finnhub", requested_at, now
        )

        with pytest.raises(Exception):
            snapshot.symbol = "MSFT"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# normalize_candles tests
# ---------------------------------------------------------------------------


class TestNormalizeCandles:
    """Tests for SnapshotNormalizer.normalize_candles()."""

    def test_basic_finnhub_candles(self, normalizer, finnhub_candles, now):
        requested_at = now - timedelta(milliseconds=200)
        snapshot = normalizer.normalize_candles(
            finnhub_candles, "AAPL", "finnhub", requested_at, now
        )

        assert isinstance(snapshot, Snapshot)
        assert snapshot.symbol == "AAPL"
        assert snapshot.data_type == "candle"
        assert snapshot.provider == "finnhub"
        assert snapshot.provider_status == "success"

    def test_uses_latest_candle_prices(self, normalizer, finnhub_candles, now):
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_candles(
            finnhub_candles, "AAPL", "finnhub", requested_at, now
        )

        # Should use last elements (index -1)
        assert snapshot.last_price == Decimal("188.1")
        assert snapshot.open == Decimal("187.5")
        assert snapshot.high == Decimal("189.0")
        assert snapshot.low == Decimal("187.0")

    def test_uses_latest_candle_volume(self, normalizer, finnhub_candles, now):
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_candles(
            finnhub_candles, "AAPL", "finnhub", requested_at, now
        )

        assert snapshot.volume == 6000000

    def test_uses_latest_candle_timestamp(self, normalizer, finnhub_candles, now):
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_candles(
            finnhub_candles, "AAPL", "finnhub", requested_at, now
        )

        assert snapshot.source_timestamp is not None
        # Latest candle is 20 seconds before now
        assert abs(snapshot.age_seconds - 20.0) < 1.0

    def test_candle_bid_ask_none(self, normalizer, finnhub_candles, now):
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_candles(
            finnhub_candles, "AAPL", "finnhub", requested_at, now
        )

        assert snapshot.bid is None
        assert snapshot.ask is None

    def test_candle_previous_close_none(self, normalizer, finnhub_candles, now):
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_candles(
            finnhub_candles, "AAPL", "finnhub", requested_at, now
        )

        assert snapshot.previous_close is None

    def test_empty_candle_response(self, normalizer, now):
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_candles(
            {}, "AAPL", "finnhub", requested_at, now
        )

        assert snapshot.provider_status == "empty"
        assert snapshot.last_price is None
        # empty_response is non-critical in trust classification → degraded
        assert snapshot.trust_state == "degraded"
        assert "empty_response" in snapshot.degradation_reasons

    def test_single_candle(self, normalizer, now):
        ts = int((now - timedelta(seconds=15)).timestamp())
        raw = {
            "s": "AAPL",
            "c": [190.00],
            "h": [191.00],
            "l": [189.00],
            "o": [189.50],
            "t": [ts],
            "v": [1000000],
        }
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_candles(
            raw, "AAPL", "finnhub", requested_at, now
        )

        assert snapshot.last_price == Decimal("190.0")
        assert snapshot.volume == 1000000

    def test_candle_freshness_classification(self, normalizer, finnhub_candles, now):
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_candles(
            finnhub_candles, "AAPL", "finnhub", requested_at, now
        )

        # 20s age with candle execution thresholds (fresh < 120s)
        assert snapshot.freshness_state == "fresh"

    def test_candle_with_error_response(self, normalizer, now):
        raw = {"error": "Internal server error"}
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_candles(
            raw, "AAPL", "finnhub", requested_at, now
        )

        assert snapshot.provider_status == "error"
        assert snapshot.last_price is None


# ---------------------------------------------------------------------------
# Market session and edge case tests
# ---------------------------------------------------------------------------


class TestNormalizerMarketSession:
    """Tests for market session handling."""

    def test_closed_market_session(self, now):
        validator = ResponseValidator(staleness_threshold_seconds=300.0)
        freshness_classifier = FreshnessClassifier(_DEFAULT_FRESHNESS_THRESHOLDS)
        trust_classifier = TrustClassifier()
        normalizer = SnapshotNormalizer(
            validator=validator,
            freshness_classifier=freshness_classifier,
            trust_classifier=trust_classifier,
            default_market_session="closed",
        )

        ts = int((now - timedelta(seconds=10)).timestamp())
        raw = {
            "s": "AAPL", "c": 187.43, "h": 188.50, "l": 186.20,
            "o": 187.00, "pc": 186.90, "t": ts, "v": 55000000,
        }
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_quote(
            raw, "AAPL", "finnhub", requested_at, now
        )

        assert snapshot.market_session == "closed"
        assert snapshot.freshness_state == "market_closed"

    def test_no_source_timestamp_age_is_zero(self, normalizer, now):
        raw = {
            "s": "AAPL",
            "c": 187.43,
            "h": 188.50,
            "l": 186.20,
            "o": 187.00,
            "pc": 186.90,
            "v": 55000000,
            # No "t" field
        }
        requested_at = now - timedelta(milliseconds=100)
        snapshot = normalizer.normalize_quote(
            raw, "AAPL", "finnhub", requested_at, now
        )

        assert snapshot.source_timestamp is None
        assert snapshot.age_seconds == 0.0
