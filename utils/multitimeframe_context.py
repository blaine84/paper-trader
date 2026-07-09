"""Compact multi-timeframe market context for Analyst and downstream PMs.

Builds a JSON-safe context shard from existing candle providers. The shard is
small enough for prompts and structured enough for candidate/PM policy use.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

from utils.technicals import compute_indicators

log = logging.getLogger(__name__)

BENCHMARK_SYMBOLS = ("SPY", "QQQ")
_CANDLE_CACHE: dict[tuple[int, str, str, int], dict] = {}
_CACHE_LOCK = threading.Lock()


def build_multitimeframe_context(
    symbol: str,
    market_data_client: Any,
    *,
    candles_5m: dict | None = None,
    indicators_5m: dict | None = None,
) -> dict:
    """Build a compact multi-timeframe context shard.

    Uses existing FinnhubClient-compatible `get_candles()` access. Failures are
    captured in the returned payload instead of raising; Analyst should keep
    working even when one timeframe or benchmark is missing.
    """
    symbol = str(symbol or "").upper()
    errors: list[str] = []

    if candles_5m is None:
        candles_5m = _get_candles_cached(market_data_client, symbol, "5", 2, errors)
    if indicators_5m is None:
        indicators_5m = _safe_compute_indicators(candles_5m, "5m", errors)

    candles_60m = _get_candles_cached(market_data_client, symbol, "60", 10, errors)
    indicators_60m = _safe_compute_indicators(candles_60m, "60m", errors)

    daily_candles = _get_candles_cached(market_data_client, symbol, "D", 220, errors)
    daily_indicators = _safe_compute_indicators(daily_candles, "daily", errors)

    sector_etf = get_sector_etf(symbol)
    relative_strength = build_relative_strength(
        symbol, daily_candles, market_data_client, sector_etf, errors
    )
    timeframes = {
        "5m": summarize_timeframe("5m", candles_5m, indicators_5m),
        "60m": summarize_timeframe("60m", candles_60m, indicators_60m),
        "daily": summarize_timeframe("daily", daily_candles, daily_indicators),
    }

    context = {
        "symbol": symbol,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timeframes": timeframes,
        "relative_strength": relative_strength,
        "volume_context": build_volume_context(candles_5m, daily_candles),
        "directional_alignment": build_directional_alignment(timeframes, relative_strength),
        "sector_etf": sector_etf,
        "errors": errors,
    }
    return context


def format_multitimeframe_context_for_prompt(context: dict) -> str:
    """Return a concise prompt block from a context shard."""
    if not context:
        return "MULTI-TIMEFRAME CONTEXT: unavailable"

    tf = context.get("timeframes", {})
    align = context.get("directional_alignment", {})
    rs = context.get("relative_strength", {})
    vol = context.get("volume_context", {})

    def _tf_line(label: str) -> str:
        row = tf.get(label, {}) or {}
        return (
            f"  {label}: trend={row.get('trend')} macd={row.get('macd_bias')} "
            f"rsi={row.get('rsi')} price={row.get('price')} "
            f"support={row.get('support')} resistance={row.get('resistance')}"
        )

    lines = [
        "MULTI-TIMEFRAME CONTEXT:",
        _tf_line("5m"),
        _tf_line("60m"),
        _tf_line("daily"),
        (
            "  relative_strength: "
            f"vs_spy_5d={_fmt(rs.get('vs_spy_5d'))} "
            f"vs_qqq_5d={_fmt(rs.get('vs_qqq_5d'))} "
            f"vs_sector_5d={_fmt(rs.get('vs_sector_5d'))} "
            f"sector_etf={context.get('sector_etf')}"
        ),
        (
            "  volume: "
            f"intraday_vs_prior_session={_fmt(vol.get('intraday_vs_prior_session'))} "
            f"daily_vs_20d_avg={_fmt(vol.get('daily_vs_20d_avg'))}"
        ),
        (
            "  alignment: "
            f"bias={align.get('bias')} score={align.get('score')} "
            f"agreement={align.get('agreement')} reasons={align.get('reasons')}"
        ),
    ]
    if context.get("errors"):
        lines.append(f"  data_warnings: {context.get('errors')[:3]}")
    return "\n".join(lines)


def _get_candles_cached(
    client: Any, symbol: str, resolution: str, days: int, errors: list[str]
) -> dict:
    key = (id(client), symbol, resolution, days)
    with _CACHE_LOCK:
        if key in _CANDLE_CACHE:
            return _CANDLE_CACHE[key]
    try:
        candles = client.get_candles(symbol, resolution=resolution, days=days) or {}
    except Exception as exc:
        errors.append(f"{symbol}:{resolution}:fetch_failed:{exc}")
        candles = {}
    with _CACHE_LOCK:
        _CANDLE_CACHE[key] = candles
    return candles


def _safe_compute_indicators(candles: dict | None, label: str, errors: list[str]) -> dict:
    if not candles:
        errors.append(f"{label}:missing_candles")
        return {}
    try:
        result = compute_indicators(candles)
        if result.get("error"):
            errors.append(f"{label}:{result['error']}")
        return result
    except Exception as exc:
        errors.append(f"{label}:indicator_failed:{exc}")
        return {}


def summarize_timeframe(label: str, candles: dict | None, indicators: dict | None) -> dict:
    indicators = indicators or {}
    candles = candles or {}
    closes = _series(candles, "close")
    highs = _series(candles, "high")
    lows = _series(candles, "low")
    volumes = _series(candles, "volume")

    return {
        "label": label,
        "source": candles.get("source"),
        "trend": _trend_from_indicators(indicators),
        "macd_bias": _macd_bias(indicators),
        "rsi": indicators.get("rsi"),
        "price": _last(closes) or indicators.get("price"),
        "support": _recent_min(lows, 20),
        "resistance": _recent_max(highs, 20),
        "return_1": _pct_return(closes, 1),
        "return_5": _pct_return(closes, 5),
        "return_20": _pct_return(closes, 20),
        "volume_last": _last(volumes),
        "volume_avg_20": _avg(volumes[-20:]) if volumes else None,
    }


def build_relative_strength(
    symbol: str,
    daily_candles: dict,
    client: Any,
    sector_etf: str | None,
    errors: list[str],
) -> dict:
    symbol_closes = _series(daily_candles, "close")
    result = {}
    for benchmark in BENCHMARK_SYMBOLS:
        bench_candles = _get_candles_cached(client, benchmark, "D", 60, errors)
        result[f"vs_{benchmark.lower()}_5d"] = _relative_return(symbol_closes, _series(bench_candles, "close"), 5)
        result[f"vs_{benchmark.lower()}_20d"] = _relative_return(symbol_closes, _series(bench_candles, "close"), 20)

    if sector_etf:
        sector_candles = _get_candles_cached(client, sector_etf, "D", 60, errors)
        result["sector_etf"] = sector_etf
        result["vs_sector_5d"] = _relative_return(symbol_closes, _series(sector_candles, "close"), 5)
        result["vs_sector_20d"] = _relative_return(symbol_closes, _series(sector_candles, "close"), 20)
    else:
        result["sector_etf"] = None
        result["vs_sector_5d"] = None
        result["vs_sector_20d"] = None
    return result


def build_volume_context(candles_5m: dict | None, daily_candles: dict | None) -> dict:
    return {
        "intraday_vs_prior_session": _intraday_volume_vs_prior_session(candles_5m or {}),
        "daily_vs_20d_avg": _last_volume_vs_average(daily_candles or {}, 20),
    }


def build_directional_alignment(timeframes: dict, relative_strength: dict) -> dict:
    score = 0
    reasons: list[str] = []

    weights = {"5m": 1, "60m": 2, "daily": 3}
    for label, weight in weights.items():
        trend = (timeframes.get(label) or {}).get("trend")
        if trend == "bullish":
            score += weight
            reasons.append(f"{label}_bullish")
        elif trend == "bearish":
            score -= weight
            reasons.append(f"{label}_bearish")

    for key in ("vs_spy_5d", "vs_qqq_5d", "vs_sector_5d"):
        value = relative_strength.get(key)
        if value is None:
            continue
        if value > 1.0:
            score += 1
            reasons.append(f"{key}_positive")
        elif value < -1.0:
            score -= 1
            reasons.append(f"{key}_negative")

    if score >= 4:
        bias = "bullish"
    elif score <= -4:
        bias = "bearish"
    else:
        bias = "mixed"

    trends = [(timeframes.get(k) or {}).get("trend") for k in ("5m", "60m", "daily")]
    non_null = [t for t in trends if t in ("bullish", "bearish")]
    agreement = (
        "aligned" if non_null and len(set(non_null)) == 1
        else "conflicted" if len(set(non_null)) > 1
        else "insufficient"
    )

    return {"bias": bias, "score": score, "agreement": agreement, "reasons": reasons}


def get_sector_etf(symbol: str) -> str | None:
    """Return configured sector ETF for a symbol, with core fallback mappings."""
    symbol = str(symbol or "").upper()
    core = {
        "AMD": "SMH", "NVDA": "SMH", "MU": "SMH", "AVGO": "SMH", "SMCI": "SMH",
        "ARM": "SMH", "INTC": "SMH", "MSFT": "QQQ", "META": "QQQ", "TSLA": "DRIV",
        "SPY": "SPY", "QQQ": "QQQ", "IWM": "IWM", "DIA": "DIA", "XLK": "XLK",
        "XLF": "XLF", "XLE": "XLE", "TLT": "TLT", "GLD": "GLD",
    }
    if symbol in core:
        return core[symbol]

    try:
        from utils.sector_scout import load_sector_scout_config
        config = load_sector_scout_config()
        for bucket in (config.get("sector_buckets") or {}).values():
            symbols = {str(s).upper() for s in bucket.get("symbols", [])}
            if symbol in symbols:
                return bucket.get("sector_etf")
    except Exception:
        return None
    return None


def _trend_from_indicators(indicators: dict) -> str | None:
    trend = indicators.get("trend") or indicators.get("ema_trend")
    if trend in ("bullish", "bearish", "neutral"):
        return trend
    return None


def _macd_bias(indicators: dict) -> str | None:
    bias = indicators.get("macd_cross") or indicators.get("macd_bias")
    if bias in ("bullish", "bearish", "neutral"):
        return bias
    return None


def _series(candles: dict, key: str) -> list[float]:
    values = candles.get(key) or []
    result = []
    for value in values:
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            continue
    return result


def _last(values: list[float]) -> float | None:
    return round(values[-1], 4) if values else None


def _recent_min(values: list[float], window: int) -> float | None:
    return round(min(values[-window:]), 4) if values else None


def _recent_max(values: list[float], window: int) -> float | None:
    return round(max(values[-window:]), 4) if values else None


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _pct_return(values: list[float], periods: int) -> float | None:
    if len(values) <= periods:
        return None
    start = values[-periods - 1]
    end = values[-1]
    if start == 0:
        return None
    return round(((end - start) / start) * 100, 2)


def _relative_return(symbol_closes: list[float], benchmark_closes: list[float], periods: int) -> float | None:
    symbol_return = _pct_return(symbol_closes, periods)
    benchmark_return = _pct_return(benchmark_closes, periods)
    if symbol_return is None or benchmark_return is None:
        return None
    return round(symbol_return - benchmark_return, 2)


def _last_volume_vs_average(candles: dict, window: int) -> float | None:
    volumes = _series(candles, "volume")
    if len(volumes) <= window:
        return None
    avg = _avg(volumes[-window - 1:-1])
    if not avg:
        return None
    return round(volumes[-1] / avg, 2)


def _intraday_volume_vs_prior_session(candles: dict) -> float | None:
    volumes = _series(candles, "volume")
    timestamps = candles.get("timestamps") or candles.get("timestamp") or candles.get("t") or []
    if not volumes or len(volumes) != len(timestamps):
        return None

    dates = []
    for ts in timestamps:
        try:
            dates.append(datetime.fromtimestamp(float(ts), timezone.utc).date().isoformat())
        except Exception:
            dates.append(None)

    valid_dates = [d for d in dates if d]
    if len(set(valid_dates)) < 2:
        return None
    current_date = valid_dates[-1]
    prior_dates = [d for d in sorted(set(valid_dates)) if d != current_date]
    prior_date = prior_dates[-1] if prior_dates else None
    if not prior_date:
        return None

    current_indices = [i for i, d in enumerate(dates) if d == current_date]
    prior_indices = [i for i, d in enumerate(dates) if d == prior_date]
    if not current_indices or not prior_indices:
        return None
    n = min(len(current_indices), len(prior_indices))
    current_sum = sum(volumes[i] for i in current_indices[:n])
    prior_sum = sum(volumes[i] for i in prior_indices[:n])
    if prior_sum <= 0:
        return None
    return round(current_sum / prior_sum, 2)


def _fmt(value: Any) -> str:
    return "n/a" if value is None else str(value)
