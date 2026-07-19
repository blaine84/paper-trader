"""Unit tests for Snapshot serialization and deserialization.

Tests specific examples and edge cases for the serialization round-trip.
Requirements: 13.1, 13.2, 13.3, 13.4, 13.5
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from decimal import Decimal

import pytest

from utils.market_data_reliability.serialization import deserialize, serialize
from utils.market_data_reliability.snapshot import Snapshot


def _make_full_snapshot() -> Snapshot:
    """Helper: create a fully populated Snapshot for testing."""
    return Snapshot(
        symbol="AAPL",
        data_type="quote",
        requested_at=datetime(2026, 7, 15, 14, 30, 0, tzinfo=timezone.utc),
        provider="finnhub",
        provider_status="success",
        market_session="open",
        last_price=Decimal("187.43"),
        bid=Decimal("187.42"),
        ask=Decimal("187.44"),
        previous_close=Decimal("186.01"),
        open=Decimal("186.50"),
        high=Decimal("188.00"),
        low=Decimal("185.75"),
        volume=12345678,
        fetched_at=datetime(2026, 7, 15, 14, 30, 1, tzinfo=timezone.utc),
        source_timestamp=datetime(2026, 7, 15, 14, 30, 0, tzinfo=timezone.utc),
        age_seconds=1.0,
        freshness_state="fresh",
        trust_state="trusted",
        degradation_reasons=(),
        raw_provider_latency_ms=45.2,
        fallback_primary_provider=None,
    )


def _make_minimal_snapshot() -> Snapshot:
    """Helper: create a Snapshot with many None fields."""
    return Snapshot(
        symbol="TSLA",
        data_type="candle",
        requested_at=datetime(2026, 7, 15, 9, 30, 0, tzinfo=timezone.utc),
        provider="yfinance",
        provider_status="error",
        market_session="closed",
        last_price=None,
        bid=None,
        ask=None,
        previous_close=None,
        open=None,
        high=None,
        low=None,
        volume=None,
        fetched_at=datetime(2026, 7, 15, 9, 30, 2, tzinfo=timezone.utc),
        source_timestamp=None,
        age_seconds=0.0,
        freshness_state="unavailable",
        trust_state="untrusted",
        degradation_reasons=("all_providers_failed", "empty_response"),
        raw_provider_latency_ms=None,
        fallback_primary_provider=None,
    )


class TestSerialize:
    """Tests for serialize() function."""

    def test_decimal_fields_encoded_as_strings(self):
        snapshot = _make_full_snapshot()
        data = serialize(snapshot)

        assert data["last_price"] == "187.43"
        assert data["bid"] == "187.42"
        assert data["ask"] == "187.44"
        assert data["previous_close"] == "186.01"
        assert data["open"] == "186.50"
        assert data["high"] == "188.00"
        assert data["low"] == "185.75"

    def test_none_decimal_fields_encoded_as_none(self):
        snapshot = _make_minimal_snapshot()
        data = serialize(snapshot)

        assert data["last_price"] is None
        assert data["bid"] is None
        assert data["ask"] is None

    def test_datetime_fields_encoded_as_iso_8601(self):
        snapshot = _make_full_snapshot()
        data = serialize(snapshot)

        assert data["requested_at"] == "2026-07-15T14:30:00+00:00"
        assert data["fetched_at"] == "2026-07-15T14:30:01+00:00"
        assert data["source_timestamp"] == "2026-07-15T14:30:00+00:00"

    def test_none_datetime_fields_encoded_as_none(self):
        snapshot = _make_minimal_snapshot()
        data = serialize(snapshot)

        assert data["source_timestamp"] is None

    def test_tuple_fields_encoded_as_list(self):
        snapshot = _make_full_snapshot()
        data = serialize(snapshot)
        assert data["degradation_reasons"] == []

        snapshot_degraded = _make_minimal_snapshot()
        data_degraded = serialize(snapshot_degraded)
        assert data_degraded["degradation_reasons"] == ["all_providers_failed", "empty_response"]
        assert isinstance(data_degraded["degradation_reasons"], list)

    def test_passthrough_fields_unchanged(self):
        snapshot = _make_full_snapshot()
        data = serialize(snapshot)

        assert data["symbol"] == "AAPL"
        assert data["data_type"] == "quote"
        assert data["provider"] == "finnhub"
        assert data["provider_status"] == "success"
        assert data["market_session"] == "open"
        assert data["volume"] == 12345678
        assert data["age_seconds"] == 1.0
        assert data["freshness_state"] == "fresh"
        assert data["trust_state"] == "trusted"
        assert data["raw_provider_latency_ms"] == 45.2
        assert data["fallback_primary_provider"] is None

    def test_all_snapshot_fields_present_in_output(self):
        snapshot = _make_full_snapshot()
        data = serialize(snapshot)

        expected_fields = set(Snapshot.__dataclass_fields__.keys())
        assert set(data.keys()) == expected_fields

    def test_high_precision_decimal_preserved(self):
        """Decimal precision is not lost during serialization."""
        snapshot = Snapshot(
            symbol="BRK.A",
            data_type="quote",
            requested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            provider="finnhub",
            provider_status="success",
            market_session="open",
            last_price=Decimal("623456.7890123456789012345678"),
            bid=Decimal("0.000000000000000000001"),
            ask=Decimal("99999999999999.99"),
            previous_close=None,
            open=None,
            high=None,
            low=None,
            volume=None,
            fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            source_timestamp=None,
            age_seconds=0.0,
            freshness_state="fresh",
            trust_state="trusted",
            degradation_reasons=(),
            raw_provider_latency_ms=None,
            fallback_primary_provider=None,
        )
        data = serialize(snapshot)

        assert data["last_price"] == "623456.7890123456789012345678"
        # str(Decimal) may use scientific notation but Decimal round-trip is exact
        assert Decimal(data["bid"]) == Decimal("0.000000000000000000001")
        assert data["ask"] == "99999999999999.99"


class TestDeserialize:
    """Tests for deserialize() function."""

    def test_decimal_fields_restored_from_strings(self):
        data = serialize(_make_full_snapshot())
        restored = deserialize(data)

        assert restored.last_price == Decimal("187.43")
        assert isinstance(restored.last_price, Decimal)
        assert restored.bid == Decimal("187.42")
        assert isinstance(restored.bid, Decimal)

    def test_none_decimal_fields_restored_as_none(self):
        data = serialize(_make_minimal_snapshot())
        restored = deserialize(data)

        assert restored.last_price is None
        assert restored.bid is None

    def test_datetime_fields_restored_from_iso_8601(self):
        data = serialize(_make_full_snapshot())
        restored = deserialize(data)

        assert restored.requested_at == datetime(2026, 7, 15, 14, 30, 0, tzinfo=timezone.utc)
        assert restored.fetched_at == datetime(2026, 7, 15, 14, 30, 1, tzinfo=timezone.utc)
        assert restored.source_timestamp == datetime(2026, 7, 15, 14, 30, 0, tzinfo=timezone.utc)

    def test_none_datetime_fields_restored_as_none(self):
        data = serialize(_make_minimal_snapshot())
        restored = deserialize(data)

        assert restored.source_timestamp is None

    def test_tuple_fields_restored_from_list(self):
        data = serialize(_make_minimal_snapshot())
        restored = deserialize(data)

        assert restored.degradation_reasons == ("all_providers_failed", "empty_response")
        assert isinstance(restored.degradation_reasons, tuple)

    def test_empty_tuple_restored(self):
        data = serialize(_make_full_snapshot())
        restored = deserialize(data)

        assert restored.degradation_reasons == ()
        assert isinstance(restored.degradation_reasons, tuple)


class TestSerializationRoundTrip:
    """Tests for serialize → deserialize round-trip equality."""

    def test_full_snapshot_round_trip(self):
        original = _make_full_snapshot()
        restored = deserialize(serialize(original))
        assert restored == original

    def test_minimal_snapshot_round_trip(self):
        original = _make_minimal_snapshot()
        restored = deserialize(serialize(original))
        assert restored == original

    def test_snapshot_with_fallback_provider_round_trip(self):
        original = Snapshot(
            symbol="NVDA",
            data_type="atr",
            requested_at=datetime(2026, 3, 10, 15, 0, 0, tzinfo=timezone.utc),
            provider="alpaca",
            provider_status="success",
            market_session="open",
            last_price=Decimal("950.25"),
            bid=None,
            ask=None,
            previous_close=Decimal("945.00"),
            open=Decimal("947.50"),
            high=Decimal("952.00"),
            low=Decimal("944.80"),
            volume=5000000,
            fetched_at=datetime(2026, 3, 10, 15, 0, 0, 500000, tzinfo=timezone.utc),
            source_timestamp=datetime(2026, 3, 10, 14, 59, 58, tzinfo=timezone.utc),
            age_seconds=2.5,
            freshness_state="fresh",
            trust_state="trusted",
            degradation_reasons=(),
            raw_provider_latency_ms=123.456,
            fallback_primary_provider="finnhub",
        )
        restored = deserialize(serialize(original))
        assert restored == original

    def test_high_precision_decimal_round_trip(self):
        """Full Decimal precision preserved across round-trip."""
        original = Snapshot(
            symbol="X",
            data_type="quote",
            requested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            provider="finnhub",
            provider_status="success",
            market_session="open",
            last_price=Decimal("123456789.012345678901234567"),
            bid=Decimal("0.000000000000000000000000001"),
            ask=Decimal("9999999999999999999999999.99"),
            previous_close=Decimal("1E+10"),
            open=Decimal("0.1"),
            high=Decimal("1000"),
            low=Decimal("0.01"),
            volume=0,
            fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            source_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            age_seconds=0.0,
            freshness_state="fresh",
            trust_state="trusted",
            degradation_reasons=(),
            raw_provider_latency_ms=0.0,
            fallback_primary_provider=None,
        )
        restored = deserialize(serialize(original))
        assert restored == original
        # Verify Decimal precision explicitly
        assert restored.last_price == Decimal("123456789.012345678901234567")
        assert restored.bid == Decimal("0.000000000000000000000000001")

    def test_timezone_aware_datetime_round_trip(self):
        """Timezone-aware datetimes preserve timezone across round-trip."""
        est = timezone(timedelta(hours=-5))
        original = Snapshot(
            symbol="SPY",
            data_type="quote",
            requested_at=datetime(2026, 7, 15, 9, 30, 0, tzinfo=est),
            provider="yfinance",
            provider_status="success",
            market_session="open",
            last_price=Decimal("450.00"),
            bid=None,
            ask=None,
            previous_close=None,
            open=None,
            high=None,
            low=None,
            volume=None,
            fetched_at=datetime(2026, 7, 15, 9, 30, 1, tzinfo=est),
            source_timestamp=datetime(2026, 7, 15, 9, 30, 0, tzinfo=est),
            age_seconds=1.0,
            freshness_state="fresh",
            trust_state="trusted",
            degradation_reasons=(),
            raw_provider_latency_ms=None,
            fallback_primary_provider=None,
        )
        restored = deserialize(serialize(original))
        assert restored == original
