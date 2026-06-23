"""Tests for candle provider ordering and normalization."""

from types import SimpleNamespace

import pytest

from utils import finnhub_client
from utils.finnhub_client import FinnhubClient


@pytest.fixture(autouse=True)
def finnhub_env(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "test-finnhub-key")
    monkeypatch.setattr(finnhub_client.finnhub, "Client", lambda api_key: SimpleNamespace())


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"{self.status_code} error")

    def json(self):
        return self._payload


def test_intraday_candles_use_alpaca_primary(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "alpaca-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "alpaca-secret")
    calls = []

    def fake_get(url, params, headers, timeout):
        calls.append((url, params, headers, timeout))
        return FakeResponse({
            "bars": {
                "SPY": [
                    {"t": "2026-06-23T14:30:00Z", "o": 630.0, "h": 631.0, "l": 629.5, "c": 630.5, "v": 1000},
                    {"t": "2026-06-23T14:35:00Z", "o": 630.5, "h": 632.0, "l": 630.0, "c": 631.5, "v": 1200},
                ]
            }
        })

    monkeypatch.setattr(finnhub_client.requests, "get", fake_get)
    monkeypatch.setattr(FinnhubClient, "_get_candles_yfinance", lambda *args, **kwargs: pytest.fail("yfinance should not be called"))
    monkeypatch.setattr(FinnhubClient, "_get_candles_finnhub", lambda *args, **kwargs: pytest.fail("finnhub should not be called"))

    candles = FinnhubClient().get_candles("SPY", resolution="5", days=2)

    assert candles == {
        "symbol": "SPY",
        "resolution": "5",
        "timestamps": [1782225000, 1782225300],
        "open": [630.0, 630.5],
        "high": [631.0, 632.0],
        "low": [629.5, 630.0],
        "close": [630.5, 631.5],
        "volume": [1000, 1200],
        "source": "alpaca",
    }
    assert calls[0][0] == "https://data.alpaca.markets/v2/stocks/bars"
    assert calls[0][1]["symbols"] == "SPY"
    assert calls[0][1]["timeframe"] == "5Min"
    assert calls[0][1]["feed"] == "iex"
    assert calls[0][2]["APCA-API-KEY-ID"] == "alpaca-key"


def test_intraday_candles_skip_alpaca_without_credentials(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    def fake_yfinance(self, symbol, resolution, days):
        return {
            "symbol": symbol,
            "resolution": resolution,
            "timestamps": [1],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [100],
        }

    monkeypatch.setattr(finnhub_client.requests, "get", lambda *args, **kwargs: pytest.fail("alpaca should not be called"))
    monkeypatch.setattr(FinnhubClient, "_get_candles_yfinance", fake_yfinance)
    monkeypatch.setattr(FinnhubClient, "_get_candles_finnhub", lambda *args, **kwargs: pytest.fail("finnhub should not be called"))

    candles = FinnhubClient().get_candles("SPY", resolution="5", days=2)

    assert candles["source"] == "yfinance"
    assert candles["close"] == [10.5]


def test_intraday_candles_fall_back_when_alpaca_empty(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "alpaca-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "alpaca-secret")
    monkeypatch.setattr(finnhub_client.requests, "get", lambda *args, **kwargs: FakeResponse({"bars": {}}))
    monkeypatch.setattr(FinnhubClient, "_get_candles_yfinance", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        FinnhubClient,
        "_get_candles_finnhub",
        lambda self, symbol, resolution, days: {
            "symbol": symbol,
            "resolution": resolution,
            "timestamps": [1],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [100],
        },
    )

    candles = FinnhubClient().get_candles("SPY", resolution="5", days=2)

    assert candles["source"] == "finnhub"
    assert candles["close"] == [10.5]


def test_daily_candles_keep_finnhub_primary(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "alpaca-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "alpaca-secret")
    monkeypatch.setattr(finnhub_client.requests, "get", lambda *args, **kwargs: pytest.fail("alpaca should not be called"))
    monkeypatch.setattr(
        FinnhubClient,
        "_get_candles_finnhub",
        lambda self, symbol, resolution, days: {
            "symbol": symbol,
            "resolution": resolution,
            "timestamps": [1],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [100],
        },
    )

    candles = FinnhubClient().get_candles("SPY", resolution="D", days=30)

    assert candles["source"] == "finnhub"
