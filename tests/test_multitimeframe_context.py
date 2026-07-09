from datetime import datetime, timedelta, timezone

from utils.multitimeframe_context import (
    build_directional_alignment,
    build_multitimeframe_context,
    format_multitimeframe_context_for_prompt,
    get_sector_mapping,
    get_sector_etf,
)


class FakeMarketDataClient:
    def __init__(self, candles_by_key):
        self.candles_by_key = candles_by_key
        self.calls = []

    def get_candles(self, symbol, resolution="5", days=2):
        self.calls.append((symbol, resolution, days))
        return self.candles_by_key.get((symbol, resolution), {})


def _candles(symbol, resolution, count=80, start=100.0, step=1.0, volume=1000):
    base = datetime(2026, 7, 8, 13, 30, tzinfo=timezone.utc)
    if resolution == "D":
        delta = timedelta(days=1)
    elif resolution == "60":
        delta = timedelta(hours=1)
    else:
        delta = timedelta(minutes=5)

    timestamps = [int((base + i * delta).timestamp()) for i in range(count)]
    closes = [start + i * step for i in range(count)]
    return {
        "symbol": symbol,
        "resolution": resolution,
        "timestamps": timestamps,
        "open": [c - 0.25 for c in closes],
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
        "volume": [volume + i for i in range(count)],
        "source": "fake",
    }


def _multi_session_5m_candles(symbol, sessions=6, bars_per_session=12):
    base = datetime(2026, 7, 1, 13, 30, tzinfo=timezone.utc)
    timestamps = []
    closes = []
    volumes = []
    for day in range(sessions):
        session_start = base + day * timedelta(days=1)
        for bar in range(bars_per_session):
            timestamps.append(int((session_start + bar * timedelta(minutes=5)).timestamp()))
            closes.append(100 + day + (bar * 0.1))
            volumes.append(1000 + (day * 50) + bar)
    return {
        "symbol": symbol,
        "resolution": "5",
        "timestamps": timestamps,
        "open": [c - 0.25 for c in closes],
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
        "volume": volumes,
        "source": "fake",
    }


def test_build_multitimeframe_context_uses_existing_data_and_benchmarks():
    candles_5m = _candles("AMD", "5", count=120, start=100, step=0.1)
    indicators_5m = {
        "price": 112,
        "trend": "bullish",
        "macd_cross": "bullish",
        "rsi": 61.0,
    }
    client = FakeMarketDataClient({
        ("AMD", "5"): _multi_session_5m_candles("AMD"),
        ("AMD", "60"): _candles("AMD", "60", count=80, start=95, step=0.4),
        ("AMD", "D"): _candles("AMD", "D", count=230, start=80, step=0.25),
        ("SPY", "D"): _candles("SPY", "D", count=80, start=500, step=0.1),
        ("QQQ", "D"): _candles("QQQ", "D", count=80, start=450, step=0.1),
        ("SMH", "D"): _candles("SMH", "D", count=80, start=250, step=0.05),
    })

    context = build_multitimeframe_context(
        "AMD",
        client,
        candles_5m=candles_5m,
        indicators_5m=indicators_5m,
    )

    assert context["symbol"] == "AMD"
    assert context["sector_etf"] == "SMH"
    assert context["timeframes"]["5m"]["trend"] == "bullish"
    assert context["timeframes"]["60m"]["trend"] == "bullish"
    assert context["timeframes"]["daily"]["trend"] == "bullish"
    assert context["relative_strength"]["vs_spy_5d"] is not None
    assert context["relative_strength"]["vs_sector_5d"] is not None
    assert context["volume_context"]["same_time_of_day"]["ratio"] is not None
    assert context["sector_context"]["sector_confirmed"] is not None
    assert context["breadth_proxy"]["bias"] in {"supportive", "mixed", "hostile"}
    assert context["directional_alignment"]["bias"] == "bullish"
    assert ("AMD", "5", 2) not in client.calls


def test_format_multitimeframe_context_for_prompt_is_compact():
    context = {
        "sector_etf": "SMH",
        "timeframes": {
            "5m": {"trend": "bullish", "macd_bias": "bullish", "rsi": 61, "price": 112, "support": 110, "resistance": 115},
            "60m": {"trend": "bullish", "macd_bias": "bullish", "rsi": 58, "price": 112, "support": 108, "resistance": 116},
            "daily": {"trend": "neutral", "macd_bias": "bullish", "rsi": 55, "price": 112, "support": 100, "resistance": 120},
        },
        "relative_strength": {"vs_spy_5d": 2.1, "vs_qqq_5d": 1.4, "vs_sector_5d": 0.8},
        "volume_context": {
            "intraday_vs_prior_session": 1.2,
            "daily_vs_20d_avg": 1.5,
            "same_time_of_day": {"ratio": 1.7, "sessions_used": 5},
        },
        "sector_context": {
            "sector_key": "ai_semi",
            "sector_name": "AI / Semiconductors",
            "sector_etf": "SMH",
            "sector_etf_return_5d": 1.2,
            "sector_confirmed": True,
        },
        "breadth_proxy": {
            "bias": "supportive",
            "watchlist_advancers_pct": 66.7,
            "benchmark_bias": "bullish",
            "sector_bias": "bullish",
        },
        "directional_alignment": {"bias": "bullish", "score": 5, "agreement": "aligned", "reasons": ["5m_bullish"]},
        "errors": [],
    }

    prompt = format_multitimeframe_context_for_prompt(context)

    assert "MULTI-TIMEFRAME CONTEXT" in prompt
    assert "5m: trend=bullish" in prompt
    assert "vs_spy_5d=2.1" in prompt
    assert "same_time_ratio=1.7" in prompt
    assert "breadth_proxy" in prompt
    assert len(prompt) < 1600


def test_directional_alignment_detects_conflict():
    timeframes = {
        "5m": {"trend": "bullish"},
        "60m": {"trend": "bearish"},
        "daily": {"trend": "bearish"},
    }
    rs = {"vs_spy_5d": -1.5, "vs_qqq_5d": None, "vs_sector_5d": -2.0}

    alignment = build_directional_alignment(timeframes, rs)

    assert alignment["bias"] == "bearish"
    assert alignment["agreement"] == "conflicted"


def test_get_sector_etf_uses_core_mapping():
    assert get_sector_etf("AMD") == "SMH"
    assert get_sector_etf("TSLA") == "DRIV"
    assert get_sector_mapping("AMD")["sector_name"] == "AI / Semiconductors"
