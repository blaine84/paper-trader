"""
Unit tests for utils/atr_helper.py — compute_intraday_atr().

Validates ATR computation, edge case handling, and return structure.
"""

from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

from utils.atr_helper import compute_intraday_atr

# Patch target: the FinnhubClient class where it's imported from
FINNHUB_CLIENT_PATH = "utils.finnhub_client.FinnhubClient"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candles(highs, lows, closes, timestamps=None, source="yfinance"):
    """Build a candle dict matching FinnhubClient.get_candles() output."""
    n = len(highs)
    if timestamps is None:
        timestamps = list(range(1000000, 1000000 + n * 300, 300))
    return {
        "symbol": "TEST",
        "resolution": "5",
        "timestamps": timestamps,
        "open": [h - 0.5 for h in highs],  # not used by ATR
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [1000] * n,
        "source": source,
    }


# ---------------------------------------------------------------------------
# Tests: Valid ATR computation
# ---------------------------------------------------------------------------


class TestComputeIntradayATR:
    """Tests for compute_intraday_atr with valid data."""

    @patch(FINNHUB_CLIENT_PATH)
    def test_basic_atr_computation(self, mock_client_cls):
        """ATR should match manual computation for known data."""
        # 16 candles → 15 TR values → ATR-14 uses last 14
        highs = [10.0, 10.5, 11.0, 10.8, 11.2, 10.9, 11.5, 11.3,
                 10.7, 11.1, 10.6, 11.4, 10.5, 11.0, 10.8, 11.2]
        lows = [9.5, 9.8, 10.2, 10.0, 10.5, 10.1, 10.8, 10.5,
                10.0, 10.3, 9.9, 10.6, 9.8, 10.2, 10.0, 10.5]
        closes = [9.8, 10.2, 10.5, 10.3, 10.8, 10.4, 11.0, 10.8,
                  10.3, 10.7, 10.2, 11.0, 10.1, 10.6, 10.3, 10.9]
        timestamps = list(range(1700000000, 1700000000 + 16 * 300, 300))

        candles = _make_candles(highs, lows, closes, timestamps)
        mock_client_cls.return_value.get_candles.return_value = candles

        result = compute_intraday_atr("TEST")

        assert result["atr"] is not None
        assert result["atr"] > 0
        assert result["candle_count"] == 16
        assert result["source"] == "yfinance"
        assert result["timestamp"] == datetime.fromtimestamp(timestamps[-1], tz=timezone.utc)

        # Manually compute expected ATR
        tr_values = []
        for i in range(1, 16):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            tr_values.append(tr)
        expected_atr = sum(tr_values[-14:]) / 14
        assert abs(result["atr"] - expected_atr) < 1e-10

    @patch(FINNHUB_CLIENT_PATH)
    def test_minimum_candles_period_plus_one(self, mock_client_cls):
        """Exactly period+1 candles should produce a valid ATR."""
        # 15 candles → 14 TR values → ATR-14 uses all 14
        n = 15
        highs = [100.0 + i * 0.5 for i in range(n)]
        lows = [99.0 + i * 0.5 for i in range(n)]
        closes = [99.5 + i * 0.5 for i in range(n)]
        timestamps = list(range(1700000000, 1700000000 + n * 300, 300))

        candles = _make_candles(highs, lows, closes, timestamps)
        mock_client_cls.return_value.get_candles.return_value = candles

        result = compute_intraday_atr("TEST")

        assert result["atr"] is not None
        assert result["candle_count"] == 15

    @patch(FINNHUB_CLIENT_PATH)
    def test_source_field_propagated(self, mock_client_cls):
        """Source field from candles should be propagated to result."""
        n = 16
        highs = [100.0 + i for i in range(n)]
        lows = [99.0 + i for i in range(n)]
        closes = [99.5 + i for i in range(n)]
        timestamps = list(range(1700000000, 1700000000 + n * 300, 300))

        candles = _make_candles(highs, lows, closes, timestamps, source="finnhub")
        mock_client_cls.return_value.get_candles.return_value = candles

        result = compute_intraday_atr("TEST")

        assert result["source"] == "finnhub"


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge case handling."""

    @patch(FINNHUB_CLIENT_PATH)
    def test_empty_candles(self, mock_client_cls):
        """Empty candle dict should return empty result."""
        mock_client_cls.return_value.get_candles.return_value = {}

        result = compute_intraday_atr("TEST")

        assert result == {"atr": None, "timestamp": None, "candle_count": 0, "source": None}

    @patch(FINNHUB_CLIENT_PATH)
    def test_none_candles(self, mock_client_cls):
        """None candle response should return empty result."""
        mock_client_cls.return_value.get_candles.return_value = None

        result = compute_intraday_atr("TEST")

        assert result == {"atr": None, "timestamp": None, "candle_count": 0, "source": None}

    @patch(FINNHUB_CLIENT_PATH)
    def test_insufficient_candles(self, mock_client_cls):
        """Fewer than period+1 candles should return atr=None."""
        # Only 10 candles, need 15 for ATR-14
        n = 10
        highs = [100.0 + i for i in range(n)]
        lows = [99.0 + i for i in range(n)]
        closes = [99.5 + i for i in range(n)]
        timestamps = list(range(1700000000, 1700000000 + n * 300, 300))

        candles = _make_candles(highs, lows, closes, timestamps)
        mock_client_cls.return_value.get_candles.return_value = candles

        result = compute_intraday_atr("TEST")

        assert result["atr"] is None
        assert result["timestamp"] is None
        assert result["candle_count"] == 10
        assert result["source"] == "yfinance"

    @patch(FINNHUB_CLIENT_PATH)
    def test_all_zero_data(self, mock_client_cls):
        """All-zero OHLCV data should return atr=None."""
        n = 16
        highs = [100.0] * n  # same high/low/close → TR = 0
        lows = [100.0] * n
        closes = [100.0] * n
        timestamps = list(range(1700000000, 1700000000 + n * 300, 300))

        candles = _make_candles(highs, lows, closes, timestamps)
        mock_client_cls.return_value.get_candles.return_value = candles

        result = compute_intraday_atr("TEST")

        assert result["atr"] is None
        assert result["timestamp"] is not None  # timestamp still set
        assert result["candle_count"] == 16

    @patch(FINNHUB_CLIENT_PATH)
    def test_finnhub_client_init_failure(self, mock_client_cls):
        """FinnhubClient constructor failure should return empty result."""
        mock_client_cls.side_effect = ValueError("FINNHUB_API_KEY not set")

        result = compute_intraday_atr("TEST")

        assert result == {"atr": None, "timestamp": None, "candle_count": 0, "source": None}

    @patch(FINNHUB_CLIENT_PATH)
    def test_get_candles_exception(self, mock_client_cls):
        """Exception during get_candles should return empty result."""
        mock_client_cls.return_value.get_candles.side_effect = RuntimeError("network error")

        result = compute_intraday_atr("TEST")

        assert result == {"atr": None, "timestamp": None, "candle_count": 0, "source": None}

    @patch(FINNHUB_CLIENT_PATH)
    def test_missing_high_key(self, mock_client_cls):
        """Candles missing 'high' key should return empty result."""
        mock_client_cls.return_value.get_candles.return_value = {
            "symbol": "TEST",
            "low": [1.0],
            "close": [1.0],
            "timestamps": [1700000000],
            "source": "yfinance",
        }

        result = compute_intraday_atr("TEST")

        assert result == {"atr": None, "timestamp": None, "candle_count": 0, "source": None}

    @patch(FINNHUB_CLIENT_PATH)
    def test_custom_period_and_resolution(self, mock_client_cls):
        """Custom period and resolution parameters should be respected."""
        n = 22  # period=20 needs 21 candles minimum
        highs = [100.0 + i * 0.3 for i in range(n)]
        lows = [99.0 + i * 0.3 for i in range(n)]
        closes = [99.5 + i * 0.3 for i in range(n)]
        timestamps = list(range(1700000000, 1700000000 + n * 300, 300))

        candles = _make_candles(highs, lows, closes, timestamps)
        mock_client_cls.return_value.get_candles.return_value = candles

        result = compute_intraday_atr("TEST", resolution="15", period=20, days=5)

        # Verify the client was called with correct params
        mock_client_cls.return_value.get_candles.assert_called_once_with("TEST", "15", 5)
        assert result["atr"] is not None
        assert result["candle_count"] == 22
