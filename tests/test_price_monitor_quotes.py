"""Tests for price monitor quote provider priority."""

import sys
from types import SimpleNamespace

import agents.price_monitor as price_monitor


def setup_function():
    price_monitor._quote_cache.clear()


def test_get_batch_quotes_uses_yfinance_primary_and_cache(monkeypatch):
    finnhub_calls = []

    class FakeFinnhub:
        def get_quote(self, symbol, retries=2):
            finnhub_calls.append(symbol)
            return {"price": {"SPY": 738.57, "AMD": 523.5}[symbol]}

    class FakeTickers:
        def __init__(self, symbols):
            assert symbols == "SPY AMD"
            self.tickers = {
                "SPY": SimpleNamespace(fast_info={"lastPrice": "738.57"}),
                "AMD": SimpleNamespace(fast_info={"lastPrice": "523.5"}),
            }

    monkeypatch.setattr(price_monitor, "FinnhubClient", FakeFinnhub)
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Tickers=FakeTickers))

    assert price_monitor.get_batch_quotes(["SPY", "AMD", "SPY"]) == {
        "SPY": 738.57,
        "AMD": 523.5,
    }
    assert finnhub_calls == []

    assert price_monitor.get_batch_quotes(["SPY", "AMD"]) == {
        "SPY": 738.57,
        "AMD": 523.5,
    }
    assert finnhub_calls == []


def test_get_batch_quotes_prefer_finnhub_uses_finnhub_primary_and_cache(monkeypatch):
    calls = []

    class FakeFinnhub:
        def get_quote(self, symbol, retries=2):
            calls.append((symbol, retries))
            return {"price": {"SPY": 738.57, "AMD": 523.5}[symbol]}

    monkeypatch.setattr(price_monitor, "FinnhubClient", FakeFinnhub)

    assert price_monitor.get_batch_quotes(["SPY", "AMD", "SPY"], prefer_finnhub=True) == {
        "SPY": 738.57,
        "AMD": 523.5,
    }
    assert calls == [("SPY", 2), ("AMD", 2)]

    assert price_monitor.get_batch_quotes(["SPY", "AMD"], prefer_finnhub=True) == {
        "SPY": 738.57,
        "AMD": 523.5,
    }
    assert calls == [("SPY", 2), ("AMD", 2)]


def test_get_batch_quotes_falls_back_to_finnhub_without_retry_sleep(monkeypatch):
    class EmptyTickers:
        def __init__(self, symbols):
            assert symbols == "SPY"
            self.tickers = {
                "SPY": SimpleNamespace(fast_info={}),
            }

    class FakeFinnhub:
        def get_quote(self, symbol, retries=2):
            assert retries == 0
            return {"price": 738.57}

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Tickers=EmptyTickers))
    monkeypatch.setattr(price_monitor, "FinnhubClient", FakeFinnhub)

    assert price_monitor.get_batch_quotes(["SPY"]) == {"SPY": 738.57}
