from utils.focus_list import select_focus_symbols


class FakeFinnhub:
    def __init__(self, quotes):
        self.quotes = quotes

    def get_quote(self, symbol):
        return dict(self.quotes[symbol])


def _quote(symbol, change_pct, price=100, high=101, low=99, prev_close=100):
    return {
        "symbol": symbol,
        "price": price,
        "high": high,
        "low": low,
        "prev_close": prev_close,
        "change_pct": change_pct,
    }


def test_select_focus_symbols_caps_optional_symbols_by_score():
    fh = FakeFinnhub({
        "SPY": _quote("SPY", 0.2),
        "AMD": _quote("AMD", 4.0, price=104, high=104, low=99),
        "MSFT": _quote("MSFT", -1.0, price=99, high=101, low=99),
        "MSTR": _quote("MSTR", 5.0, price=105, high=105, low=98),
    })

    result = select_focus_symbols(
        None,
        ["SPY", "AMD", "MSFT", "MSTR"],
        max_symbols=2,
        source_bonuses={"MSTR": 3.0, "AMD": 2.0},
        fh=fh,
    )

    assert result == ["MSTR", "AMD"]


def test_select_focus_symbols_keeps_required_symbols_even_past_cap():
    fh = FakeFinnhub({
        "SPY": _quote("SPY", 0.2),
        "AMD": _quote("AMD", 4.0, price=104, high=104, low=99),
        "MSFT": _quote("MSFT", -1.0, price=99, high=101, low=99),
    })

    result = select_focus_symbols(
        None,
        ["SPY", "AMD", "MSFT"],
        max_symbols=1,
        required_symbols=["MSFT"],
        source_bonuses={"AMD": 2.0},
        fh=fh,
    )

    assert result == ["MSFT", "AMD"]


def test_select_focus_symbols_deduplicates_inputs():
    fh = FakeFinnhub({
        "AMD": _quote("AMD", 4.0, price=104, high=104, low=99),
        "MSFT": _quote("MSFT", -1.0, price=99, high=101, low=99),
    })

    result = select_focus_symbols(
        None,
        ["amd", "AMD", "MSFT", "msft"],
        max_symbols=3,
        fh=fh,
    )

    assert sorted(result) == ["AMD", "MSFT"]
