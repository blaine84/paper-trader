"""Tests for price monitor quote provider priority."""

import sys
from types import SimpleNamespace

import agents.price_monitor as price_monitor


def setup_function():
    price_monitor._quote_cache.clear()


def test_get_batch_quotes_uses_finnhub_primary_and_cache(monkeypatch):
    calls = []

    class FakeFinnhub:
        def get_quote(self, symbol):
            calls.append(symbol)
            return {"price": {"SPY": 738.57, "AMD": 523.5}[symbol]}

    monkeypatch.setattr(price_monitor, "FinnhubClient", FakeFinnhub)

    assert price_monitor.get_batch_quotes(["SPY", "AMD", "SPY"]) == {
        "SPY": 738.57,
        "AMD": 523.5,
    }
    assert calls == ["SPY", "AMD"]

    assert price_monitor.get_batch_quotes(["SPY", "AMD"]) == {
        "SPY": 738.57,
        "AMD": 523.5,
    }
    assert calls == ["SPY", "AMD"]


def test_get_batch_quotes_falls_back_to_yfinance(monkeypatch):
    class BrokenFinnhub:
        def get_quote(self, symbol):
            raise RuntimeError("rate limited")

    class FakeTickers:
        def __init__(self, symbols):
            assert symbols == "SPY"
            self.tickers = {
                "SPY": SimpleNamespace(fast_info={"lastPrice": "738.57"}),
            }

    monkeypatch.setattr(price_monitor, "FinnhubClient", BrokenFinnhub)
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Tickers=FakeTickers))

    assert price_monitor.get_batch_quotes(["SPY"]) == {"SPY": 738.57}
