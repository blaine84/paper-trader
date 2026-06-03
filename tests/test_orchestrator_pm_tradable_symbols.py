"""Tests for PM tradable symbol watchlist handling."""

import orchestrator


def test_parse_symbol_list_normalizes_and_dedupes():
    assert orchestrator._parse_symbol_list(" spy, META,meta,, MU ") == [
        "SPY",
        "META",
        "MU",
    ]


def test_pm_base_watchlist_includes_non_core_tradable_symbols(monkeypatch):
    monkeypatch.setattr(orchestrator, "WATCHLIST", ["SPY", "QQQ", "AMD"])
    monkeypatch.setattr(orchestrator, "PM_TRADABLE_SYMBOLS", ["META", "MU"])

    assert orchestrator._pm_base_watchlist() == ["SPY", "QQQ", "AMD", "META", "MU"]


def test_pm_base_watchlist_dedupes_core_overlap(monkeypatch):
    monkeypatch.setattr(orchestrator, "WATCHLIST", ["SPY", "META", "AMD"])
    monkeypatch.setattr(orchestrator, "PM_TRADABLE_SYMBOLS", ["META", "MU"])

    assert orchestrator._pm_base_watchlist() == ["SPY", "META", "AMD", "MU"]
