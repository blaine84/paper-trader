from utils.regime_evidence import (
    build_regime_evidence,
    format_regime_evidence_for_prompt,
)


def _quote(price, prev_close):
    return {
        "price": price,
        "prev_close": prev_close,
        "change_pct": round(((price - prev_close) / prev_close) * 100, 3),
    }


def test_build_regime_evidence_complete_from_quote_fetcher():
    quotes = {
        "^VIX": _quote(18.0, 20.0),
        "^TNX": _quote(43.25, 42.0),
        "TLT": _quote(90.0, 91.0),
        "XLK": _quote(250.0, 245.0),
        "VLUE": _quote(120.0, 121.0),
        "IWF": _quote(430.0, 425.0),
        "IWD": _quote(190.0, 191.0),
        "QQQ": _quote(575.0, 570.0),
        "SPY": _quote(650.0, 648.0),
        "RSP": _quote(185.0, 184.0),
        "IWM": _quote(230.0, 229.0),
    }

    result = build_regime_evidence(lambda symbol: quotes[symbol])

    assert result["evidence_quality"] == "complete"
    assert result["proxies"]["vix"]["value"] == 18.0
    assert result["proxies"]["ten_year_yield"]["value_pct"] == 4.325
    assert result["proxies"]["tech_value_ratio"]["symbols"] == ["XLK", "VLUE"]
    assert result["proxies"]["breadth_proxy"]["symbols"] == ["RSP", "SPY"]
    assert result["errors"] == {}


def test_build_regime_evidence_partial_when_some_proxies_missing():
    quotes = {
        "^VIX": _quote(18.0, 20.0),
        "^TNX": _quote(43.25, 42.0),
    }

    result = build_regime_evidence(lambda symbol: quotes.get(symbol))

    assert result["evidence_quality"] == "partial"
    assert result["proxies"]["tech_value_ratio"] is None
    assert result["errors"]["XLK"] == "empty_quote"


def test_format_regime_evidence_for_prompt_is_compact_and_explicit():
    evidence = build_regime_evidence(lambda symbol: {
        "^VIX": _quote(18.0, 20.0),
        "^TNX": _quote(43.25, 42.0),
    }.get(symbol))

    text = format_regime_evidence_for_prompt(evidence)

    assert "Evidence quality: partial" in text
    assert "VIX (^VIX): 18.0" in text
    assert "10Y yield (^TNX): 4.325%" in text
    assert "Missing/degraded proxies:" in text
