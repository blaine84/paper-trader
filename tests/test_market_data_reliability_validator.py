"""Unit tests for ResponseValidator.

Tests validation of raw provider responses including cross-symbol detection,
invalid price detection, timestamp handling, provider errors, rate limiting,
empty responses, and malformed JSON.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta

import pytest

from utils.market_data_reliability.validator import ResponseValidator


@pytest.fixture
def validator():
    """Default validator with 300s staleness threshold."""
    return ResponseValidator(staleness_threshold_seconds=300.0)


class TestCrossSymbolResponse:
    """Tests for cross_symbol_response detection (Requirement 4.1)."""

    def test_detects_cross_symbol_in_s_key(self, validator):
        raw = {"s": "MSFT", "c": 150.0, "t": time.time()}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert result.is_valid is False
        assert "cross_symbol_response" in result.degradation_reasons

    def test_detects_cross_symbol_in_symbol_key(self, validator):
        raw = {"symbol": "GOOG", "current_price": 100.0, "timestamp": time.time()}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert result.is_valid is False
        assert "cross_symbol_response" in result.degradation_reasons

    def test_matching_symbol_passes(self, validator):
        raw = {"s": "AAPL", "c": 150.0, "t": time.time()}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert "cross_symbol_response" not in result.degradation_reasons

    def test_symbol_comparison_is_case_insensitive(self, validator):
        raw = {"s": "aapl", "c": 150.0, "t": time.time()}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert "cross_symbol_response" not in result.degradation_reasons

    def test_no_symbol_key_does_not_flag(self, validator):
        raw = {"c": 150.0, "t": time.time()}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert "cross_symbol_response" not in result.degradation_reasons


class TestInvalidPrice:
    """Tests for invalid_price detection (Requirement 4.2)."""

    def test_zero_price_is_invalid(self, validator):
        raw = {"c": 0, "t": time.time()}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert result.is_valid is False
        assert "invalid_price" in result.degradation_reasons

    def test_negative_price_is_invalid(self, validator):
        raw = {"c": -5.0, "t": time.time()}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert result.is_valid is False
        assert "invalid_price" in result.degradation_reasons

    def test_positive_price_is_valid(self, validator):
        raw = {"c": 150.50, "t": time.time()}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert "invalid_price" not in result.degradation_reasons

    def test_invalid_price_only_for_price_data_types(self, validator):
        # ATR data type should not check prices
        raw = {"c": -5.0, "t": time.time()}
        result = validator.validate(raw, symbol="AAPL", data_type="atr")
        assert "invalid_price" not in result.degradation_reasons

    def test_invalid_price_applies_to_candle(self, validator):
        raw = {"o": 0.0, "t": time.time()}
        result = validator.validate(raw, symbol="AAPL", data_type="candle")
        assert result.is_valid is False
        assert "invalid_price" in result.degradation_reasons

    def test_invalid_price_applies_to_previous_close(self, validator):
        raw = {"pc": -1.0, "t": time.time()}
        result = validator.validate(raw, symbol="AAPL", data_type="previous_close")
        assert result.is_valid is False
        assert "invalid_price" in result.degradation_reasons


class TestMissingSourceTimestamp:
    """Tests for missing_source_timestamp detection (Requirement 4.3)."""

    def test_no_timestamp_key_flags_missing(self, validator):
        raw = {"s": "AAPL", "c": 150.0}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert "missing_source_timestamp" in result.degradation_reasons
        # Missing timestamp is non-critical
        assert result.is_valid is True

    def test_unparsable_timestamp_flags_missing(self, validator):
        raw = {"s": "AAPL", "c": 150.0, "t": "not-a-timestamp"}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert "missing_source_timestamp" in result.degradation_reasons

    def test_valid_epoch_timestamp_does_not_flag(self, validator):
        raw = {"s": "AAPL", "c": 150.0, "t": time.time()}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert "missing_source_timestamp" not in result.degradation_reasons

    def test_valid_iso_timestamp_does_not_flag(self, validator):
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        raw = {"s": "AAPL", "c": 150.0, "timestamp": now_iso}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert "missing_source_timestamp" not in result.degradation_reasons

    def test_zero_epoch_flags_missing(self, validator):
        raw = {"s": "AAPL", "c": 150.0, "t": 0}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert "missing_source_timestamp" in result.degradation_reasons


class TestStaleSourceTimestamp:
    """Tests for stale_source_timestamp detection (Requirement 4.4)."""

    def test_old_timestamp_flags_stale(self, validator):
        old_time = time.time() - 600  # 10 minutes old (> 300s threshold)
        raw = {"s": "AAPL", "c": 150.0, "t": old_time}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert "stale_source_timestamp" in result.degradation_reasons
        # Stale timestamp is non-critical
        assert result.is_valid is True

    def test_recent_timestamp_does_not_flag_stale(self, validator):
        recent_time = time.time() - 10  # 10 seconds old (< 300s threshold)
        raw = {"s": "AAPL", "c": 150.0, "t": recent_time}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert "stale_source_timestamp" not in result.degradation_reasons

    def test_custom_threshold(self):
        validator = ResponseValidator(staleness_threshold_seconds=60.0)
        old_time = time.time() - 90  # 90 seconds old (> 60s threshold)
        raw = {"s": "AAPL", "c": 150.0, "t": old_time}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert "stale_source_timestamp" in result.degradation_reasons


class TestProviderError:
    """Tests for provider_error detection (Requirement 4.5)."""

    def test_error_key_flags_provider_error(self, validator):
        raw = {"error": "API key invalid", "s": "AAPL"}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert result.is_valid is False
        assert "provider_error" in result.degradation_reasons

    def test_empty_error_does_not_flag(self, validator):
        raw = {"error": "", "s": "AAPL", "c": 150.0, "t": time.time()}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert "provider_error" not in result.degradation_reasons

    def test_none_error_does_not_flag(self, validator):
        raw = {"error": None, "s": "AAPL", "c": 150.0, "t": time.time()}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert "provider_error" not in result.degradation_reasons


class TestRateLimited:
    """Tests for rate_limited detection (Requirement 4.5)."""

    def test_status_429_flags_rate_limited(self, validator):
        raw = {"status": 429, "s": "AAPL"}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert result.is_valid is False
        assert "rate_limited" in result.degradation_reasons

    def test_rate_limit_text_in_error_flags(self, validator):
        raw = {"error": "API rate limit exceeded", "s": "AAPL"}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert "rate_limited" in result.degradation_reasons

    def test_too_many_requests_text_flags(self, validator):
        raw = {"error": "Too many requests", "s": "AAPL"}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert "rate_limited" in result.degradation_reasons

    def test_string_429_status(self, validator):
        raw = {"status_code": "429", "s": "AAPL"}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert "rate_limited" in result.degradation_reasons


class TestEmptyResponse:
    """Tests for empty_response detection (Requirement 4.5)."""

    def test_empty_dict_flags_empty(self, validator):
        result = validator.validate({}, symbol="AAPL", data_type="quote")
        assert result.is_valid is False
        assert "empty_response" in result.degradation_reasons

    def test_all_none_values_flags_empty(self, validator):
        raw = {"c": None, "h": None, "l": None}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert result.is_valid is False
        assert "empty_response" in result.degradation_reasons

    def test_explicit_empty_marker_flags_empty(self, validator):
        raw = {"_empty": True, "symbol": "AAPL"}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert result.is_valid is False
        assert "empty_response" in result.degradation_reasons


class TestMalformedJson:
    """Tests for malformed_json detection (Requirement 4.5)."""

    def test_parse_error_marker_flags_malformed(self, validator):
        raw = {"_parse_error": True}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert result.is_valid is False
        assert "malformed_json" in result.degradation_reasons


class TestValidResponse:
    """Tests for fully valid responses."""

    def test_complete_finnhub_quote_is_valid(self, validator):
        raw = {
            "s": "AAPL",
            "c": 187.43,
            "h": 188.50,
            "l": 186.20,
            "o": 187.00,
            "pc": 186.90,
            "t": time.time(),
        }
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert result.is_valid is True
        assert result.degradation_reasons == ()

    def test_valid_response_with_no_issues(self, validator):
        raw = {"s": "MSFT", "c": 420.0, "t": time.time()}
        result = validator.validate(raw, symbol="MSFT", data_type="quote")
        assert result.is_valid is True
        assert result.degradation_reasons == ()


class TestMultipleIssues:
    """Tests for responses with multiple validation issues."""

    def test_cross_symbol_and_invalid_price(self, validator):
        raw = {"s": "MSFT", "c": -5.0, "t": time.time()}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert result.is_valid is False
        assert "cross_symbol_response" in result.degradation_reasons
        assert "invalid_price" in result.degradation_reasons

    def test_stale_and_missing_ts_not_both_present(self, validator):
        # If timestamp is missing, stale check shouldn't run
        raw = {"s": "AAPL", "c": 150.0}
        result = validator.validate(raw, symbol="AAPL", data_type="quote")
        assert "missing_source_timestamp" in result.degradation_reasons
        assert "stale_source_timestamp" not in result.degradation_reasons
