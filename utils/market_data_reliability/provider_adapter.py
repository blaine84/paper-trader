"""Live provider adapter for the Market Data Reliability Layer.

Bridges the reliability layer's provider/data_type contract to the market-data
helpers already used by the runtime. The adapter returns raw dicts in the
normalizer's canonical provider shape:

- quotes: ``s/c/h/l/o/pc/t/v``
- candles: ``s/c/h/l/o/t/v`` with list values
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any


def _iso_to_epoch(value: Any) -> int:
    if not value:
        return int(time.time())
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
        except ValueError:
            return int(time.time())
    return int(time.time())


def _quote_from_finnhub_client(symbol: str) -> dict:
    from utils.finnhub_client import FinnhubClient

    q = FinnhubClient().get_quote(symbol, retries=0)
    if not q or not q.get("price"):
        raise RuntimeError(f"empty quote response for {symbol}")
    return {
        "s": symbol,
        "c": q.get("price"),
        "h": q.get("high"),
        "l": q.get("low"),
        "o": q.get("open"),
        "pc": q.get("prev_close"),
        "t": _iso_to_epoch(q.get("timestamp")),
        "v": q.get("volume"),
    }


def _quote_from_yfinance(symbol: str) -> dict:
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    fast = ticker.fast_info
    price = fast.get("lastPrice")
    if not price:
        raise RuntimeError(f"empty yfinance quote response for {symbol}")
    return {
        "s": symbol,
        "c": price,
        "h": fast.get("dayHigh"),
        "l": fast.get("dayLow"),
        "o": fast.get("open"),
        "pc": fast.get("previousClose"),
        "t": int(time.time()),
        "v": fast.get("lastVolume"),
    }


def _candles_from_client(symbol: str, provider: str, resolution: str = "5") -> dict:
    from utils.finnhub_client import FinnhubClient

    client = FinnhubClient()
    if provider == "finnhub":
        candles = client._get_candles_finnhub(symbol, resolution, days=2)
    elif provider == "yfinance":
        candles = client._get_candles_yfinance(symbol, resolution, days=2)
    elif provider == "alpaca":
        candles = client._get_candles_alpaca(symbol, resolution, days=2)
    else:
        raise ValueError(f"Unsupported market-data provider: {provider}")

    if not candles:
        raise RuntimeError(
            f"empty candle response for {symbol} provider={provider}"
        )

    return {
        "s": symbol,
        "t": candles.get("timestamps", []),
        "o": candles.get("open", []),
        "h": candles.get("high", []),
        "l": candles.get("low", []),
        "c": candles.get("close", []),
        "v": candles.get("volume", []),
    }


def fetch_market_data(provider: str, symbol: str, data_type: str) -> dict:
    """Fetch live market data for ReliabilityLayer.

    ``atr`` and ``volume`` readiness both require fresh intraday bars. The ATR
    value itself is computed later by risk geometry helpers; here we only prove
    that trusted fresh candle/volume context is available for execution.
    """

    provider = provider.lower()
    data_type = data_type.lower()

    if data_type == "quote":
        if provider == "finnhub":
            return _quote_from_finnhub_client(symbol)
        if provider == "yfinance":
            return _quote_from_yfinance(symbol)
        raise ValueError(f"Unsupported quote provider: {provider}")

    if data_type in {"candle", "atr", "volume"}:
        return _candles_from_client(symbol, provider, resolution="5")

    raise ValueError(f"Unsupported market-data type: {data_type}")
