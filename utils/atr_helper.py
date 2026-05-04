"""
Lightweight ATR helper for intraday volatility measurement.

Fetches 5-min candles via FinnhubClient and computes ATR-14
with freshness metadata for use by the RiskGeometryGate.
"""

import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)


def compute_intraday_atr(
    symbol: str,
    resolution: str = "5",
    period: int = 14,
    days: int = 2,
) -> dict:
    """Fetch 5-min candles and compute ATR with freshness metadata.

    Args:
        symbol: Ticker symbol (e.g., "AAPL").
        resolution: Candle resolution (default "5" for 5-minute).
        period: ATR period (default 14).
        days: Number of days of candle history to fetch.

    Returns:
        {
            "atr": float | None,
            "timestamp": datetime | None,  # timestamp of latest candle used
            "candle_count": int,
            "source": str | None,          # "yfinance" | "finnhub" | None
        }
    """
    empty_result = {"atr": None, "timestamp": None, "candle_count": 0, "source": None}

    # Fetch candles — handle FinnhubClient init failure (e.g., missing API key)
    try:
        from utils.finnhub_client import FinnhubClient

        client = FinnhubClient()
        candles = client.get_candles(symbol, resolution, days)
    except Exception as e:
        log.warning("ATR helper: failed to fetch candles for %s: %s", symbol, e)
        return empty_result

    # Validate candle data
    if not candles or not isinstance(candles, dict):
        log.debug("ATR helper: no candle data returned for %s", symbol)
        return empty_result

    high = candles.get("high", [])
    low = candles.get("low", [])
    close = candles.get("close", [])
    timestamps = candles.get("timestamps", [])

    # All required lists must be present and non-empty
    if not high or not low or not close or not timestamps:
        log.debug("ATR helper: missing OHLCV fields for %s", symbol)
        return empty_result

    # Need at least period + 1 candles (prev_close needed for first TR)
    candle_count = len(timestamps)
    if candle_count < period + 1:
        log.debug(
            "ATR helper: insufficient candles for %s (%d < %d)",
            symbol,
            candle_count,
            period + 1,
        )
        return {
            "atr": None,
            "timestamp": None,
            "candle_count": candle_count,
            "source": candles.get("source"),
        }

    # Compute True Range values starting from index 1
    tr_values = []
    for i in range(1, candle_count):
        h = high[i]
        l = low[i]
        prev_c = close[i - 1]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        tr_values.append(tr)

    # Need at least `period` TR values for ATR
    if len(tr_values) < period:
        log.debug("ATR helper: not enough TR values for %s", symbol)
        return {
            "atr": None,
            "timestamp": None,
            "candle_count": candle_count,
            "source": candles.get("source"),
        }

    # ATR = SMA of last `period` TR values
    atr_window = tr_values[-period:]

    # Edge case: all-zero data
    if all(v == 0 for v in atr_window):
        log.debug("ATR helper: all-zero TR values for %s", symbol)
        return {
            "atr": None,
            "timestamp": datetime.fromtimestamp(timestamps[-1], tz=timezone.utc),
            "candle_count": candle_count,
            "source": candles.get("source"),
        }

    atr = sum(atr_window) / period
    timestamp = datetime.fromtimestamp(timestamps[-1], tz=timezone.utc)
    source = candles.get("source")

    return {
        "atr": atr,
        "timestamp": timestamp,
        "candle_count": candle_count,
        "source": source,
    }
