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
    breadth_symbols: list[str] | tuple[str, ...] | None = None,
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
    candles_5m_baseline = candles_5m
    if _session_count(candles_5m_baseline) < 5:
        candles_5m_baseline = _get_candles_cached(
            market_data_client, symbol, "5", 20, errors
        )
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
    sector_context = build_sector_context(
        symbol, daily_candles, market_data_client, sector_etf, errors
    )
    breadth_proxy = build_breadth_proxy(
        market_data_client,
        breadth_symbols=breadth_symbols,
        sector_etf=sector_etf,
        errors=errors,
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
        "volume_context": build_volume_context(
            candles_5m,
            daily_candles,
            candles_5m_baseline=candles_5m_baseline,
        ),
        "sector_context": sector_context,
        "breadth_proxy": breadth_proxy,
        "directional_alignment": build_directional_alignment(
            timeframes, relative_strength, sector_context, breadth_proxy
        ),
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
    tod = vol.get("same_time_of_day") or {}
    sector = context.get("sector_context", {}) or {}
    breadth = context.get("breadth_proxy", {}) or {}

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
            f"daily_vs_20d_avg={_fmt(vol.get('daily_vs_20d_avg'))} "
            f"same_time_ratio={_fmt(tod.get('ratio'))} "
            f"same_time_sessions={_fmt(tod.get('sessions_used'))}"
        ),
        (
            "  sector_context: "
            f"sector={sector.get('sector_key') or 'n/a'} "
            f"name={sector.get('sector_name') or 'n/a'} "
            f"etf={sector.get('sector_etf') or context.get('sector_etf')} "
            f"etf_5d={_fmt(sector.get('sector_etf_return_5d'))} "
            f"confirmed={_fmt(sector.get('sector_confirmed'))}"
        ),
        (
            "  breadth_proxy: "
            f"bias={breadth.get('bias')} "
            f"watchlist_advancers={_fmt(breadth.get('watchlist_advancers_pct'))} "
            f"benchmarks={breadth.get('benchmark_bias')} "
            f"sector={breadth.get('sector_bias')}"
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


def build_volume_context(
    candles_5m: dict | None,
    daily_candles: dict | None,
    *,
    candles_5m_baseline: dict | None = None,
) -> dict:
    return {
        "intraday_vs_prior_session": _intraday_volume_vs_prior_session(candles_5m or {}),
        "same_time_of_day": _same_time_of_day_volume_baseline(
            candles_5m_baseline or candles_5m or {}
        ),
        "daily_vs_20d_avg": _last_volume_vs_average(daily_candles or {}, 20),
    }


def build_sector_context(
    symbol: str,
    daily_candles: dict,
    client: Any,
    sector_etf: str | None,
    errors: list[str],
) -> dict:
    mapping = get_sector_mapping(symbol)
    sector_candles = (
        _get_candles_cached(client, sector_etf, "D", 60, errors)
        if sector_etf else {}
    )
    symbol_return_5d = _pct_return(_series(daily_candles, "close"), 5)
    sector_return_5d = _pct_return(_series(sector_candles, "close"), 5)
    sector_return_1d = _pct_return(_series(sector_candles, "close"), 1)
    sector_confirmed = _same_direction(symbol_return_5d, sector_return_5d)

    return {
        "sector_key": mapping.get("sector_key"),
        "sector_name": mapping.get("sector_name"),
        "sector_etf": sector_etf,
        "source": mapping.get("source"),
        "peer_count": len(mapping.get("symbols") or []),
        "sector_etf_return_1d": sector_return_1d,
        "sector_etf_return_5d": sector_return_5d,
        "symbol_return_5d": symbol_return_5d,
        "sector_confirmed": sector_confirmed,
    }


def build_breadth_proxy(
    client: Any,
    *,
    breadth_symbols: list[str] | tuple[str, ...] | None,
    sector_etf: str | None,
    errors: list[str],
) -> dict:
    """Build a lightweight participation proxy from known symbols/ETFs.

    This is not exchange-wide breadth. It intentionally uses the current
    watchlist plus benchmark/sector ETFs to estimate whether the tape supports
    a candidate's direction.
    """
    symbols = _dedupe_symbols([*(breadth_symbols or ()), *BENCHMARK_SYMBOLS])
    if sector_etf:
        symbols = _dedupe_symbols([*symbols, sector_etf])

    returns_1d: dict[str, float] = {}
    for sym in symbols[:30]:
        candles = _get_candles_cached(client, sym, "D", 60, errors)
        ret = _pct_return(_series(candles, "close"), 1)
        if ret is not None:
            returns_1d[sym] = ret

    watchlist_symbols = [
        sym for sym in _dedupe_symbols(breadth_symbols or ())
        if sym in returns_1d
    ]
    watchlist_returns = [returns_1d[sym] for sym in watchlist_symbols]
    benchmark_returns = {
        sym: returns_1d.get(sym)
        for sym in BENCHMARK_SYMBOLS
        if returns_1d.get(sym) is not None
    }
    sector_return = returns_1d.get(sector_etf) if sector_etf else None

    advancers = sum(1 for ret in watchlist_returns if ret > 0)
    decliners = sum(1 for ret in watchlist_returns if ret < 0)
    total = len(watchlist_returns)
    advancers_pct = round((advancers / total) * 100, 1) if total else None

    benchmark_avg = _avg(list(benchmark_returns.values()))
    if advancers_pct is None and benchmark_avg is None:
        bias = "unknown"
    elif (advancers_pct or 0) >= 60 and (benchmark_avg or 0) > 0:
        bias = "supportive"
    elif (advancers_pct or 100) <= 40 and (benchmark_avg or 0) < 0:
        bias = "hostile"
    else:
        bias = "mixed"

    return {
        "type": "watchlist_etf_proxy",
        "bias": bias,
        "symbols_sampled": total,
        "watchlist_advancers": advancers,
        "watchlist_decliners": decliners,
        "watchlist_advancers_pct": advancers_pct,
        "benchmark_returns_1d": benchmark_returns,
        "benchmark_bias": _return_bias(benchmark_avg),
        "sector_etf": sector_etf,
        "sector_return_1d": sector_return,
        "sector_bias": _return_bias(sector_return),
    }


def build_directional_alignment(
    timeframes: dict,
    relative_strength: dict,
    sector_context: dict | None = None,
    breadth_proxy: dict | None = None,
) -> dict:
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

    if (sector_context or {}).get("sector_confirmed") is True:
        score += 1
        reasons.append("sector_confirmed")
    elif (sector_context or {}).get("sector_confirmed") is False:
        score -= 1
        reasons.append("sector_not_confirmed")

    breadth_bias = (breadth_proxy or {}).get("bias")
    if breadth_bias == "supportive":
        score += 1
        reasons.append("breadth_supportive")
    elif breadth_bias == "hostile":
        score -= 1
        reasons.append("breadth_hostile")

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
    return get_sector_mapping(symbol).get("sector_etf")


def get_sector_mapping(symbol: str) -> dict:
    """Return sector metadata from Sector Scout config with core fallbacks."""
    symbol = str(symbol or "").upper()
    core = {
        "AMD": ("ai_semi_core", "AI / Semiconductors", "SMH"),
        "NVDA": ("ai_semi_core", "AI / Semiconductors", "SMH"),
        "MU": ("ai_semi", "AI / Semiconductors", "SMH"),
        "AVGO": ("ai_semi", "AI / Semiconductors", "SMH"),
        "SMCI": ("ai_semi", "AI / Semiconductors", "SMH"),
        "ARM": ("ai_semi", "AI / Semiconductors", "SMH"),
        "INTC": ("ai_semi", "AI / Semiconductors", "SMH"),
        "MSFT": ("mega_cap_tech", "Mega-Cap Tech", "QQQ"),
        "META": ("mega_cap_tech", "Mega-Cap Tech", "QQQ"),
        "AAPL": ("mega_cap_tech", "Mega-Cap Tech", "QQQ"),
        "GOOGL": ("mega_cap_tech", "Mega-Cap Tech", "QQQ"),
        "AMZN": ("mega_cap_tech", "Mega-Cap Tech", "QQQ"),
        "TSLA": ("ev_high_beta_core", "EV / High-Beta", "DRIV"),
        "SPY": ("benchmark", "S&P 500", "SPY"),
        "QQQ": ("benchmark", "Nasdaq 100", "QQQ"),
        "IWM": ("benchmark", "Russell 2000", "IWM"),
        "DIA": ("benchmark", "Dow Industrials", "DIA"),
        "XLK": ("sector_etf", "Technology", "XLK"),
        "XLF": ("sector_etf", "Financials", "XLF"),
        "XLE": ("sector_etf", "Energy", "XLE"),
        "TLT": ("macro", "Long Bonds", "TLT"),
        "GLD": ("macro", "Gold", "GLD"),
    }

    try:
        from utils.sector_scout import load_sector_scout_config
        config = load_sector_scout_config()
        for key, bucket in (config.get("sector_buckets") or {}).items():
            symbols = {str(s).upper() for s in bucket.get("symbols", [])}
            if symbol in symbols:
                return {
                    "sector_key": key,
                    "sector_name": bucket.get("name"),
                    "sector_etf": bucket.get("sector_etf"),
                    "symbols": sorted(symbols),
                    "source": "sector_scout_config",
                }
    except Exception:
        pass

    if symbol in core:
        sector_key, sector_name, sector_etf = core[symbol]
        return {
            "sector_key": sector_key,
            "sector_name": sector_name,
            "sector_etf": sector_etf,
            "symbols": [symbol],
            "source": "core_fallback",
        }

    return {
        "sector_key": None,
        "sector_name": None,
        "sector_etf": None,
        "symbols": [],
        "source": "unknown",
    }


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


def _same_time_of_day_volume_baseline(candles: dict) -> dict:
    volumes = _series(candles, "volume")
    timestamps = candles.get("timestamps") or candles.get("timestamp") or candles.get("t") or []
    if not volumes or len(volumes) != len(timestamps):
        return {
            "ratio": None,
            "sessions_used": 0,
            "reason": "missing_intraday_volume_or_timestamps",
        }

    rows = []
    for idx, ts in enumerate(timestamps):
        try:
            dt = datetime.fromtimestamp(float(ts), timezone.utc)
            rows.append((dt.date().isoformat(), dt.time().isoformat(), volumes[idx]))
        except Exception:
            continue
    if not rows:
        return {"ratio": None, "sessions_used": 0, "reason": "no_valid_rows"}

    current_date = rows[-1][0]
    current_times = [time_key for date_key, time_key, _ in rows if date_key == current_date]
    if not current_times:
        return {"ratio": None, "sessions_used": 0, "reason": "no_current_session"}
    cutoff_time = current_times[-1]

    by_date: dict[str, float] = {}
    for date_key, time_key, volume in rows:
        if time_key <= cutoff_time:
            by_date[date_key] = by_date.get(date_key, 0.0) + volume

    current_volume = by_date.get(current_date)
    prior_volumes = [
        volume for date_key, volume in sorted(by_date.items())
        if date_key != current_date and volume > 0
    ]
    if current_volume is None or not prior_volumes:
        return {
            "ratio": None,
            "sessions_used": len(prior_volumes),
            "reason": "insufficient_prior_sessions",
        }

    avg_prior = sum(prior_volumes) / len(prior_volumes)
    if avg_prior <= 0:
        return {
            "ratio": None,
            "sessions_used": len(prior_volumes),
            "reason": "zero_prior_volume",
        }

    return {
        "ratio": round(current_volume / avg_prior, 2),
        "current_cumulative_volume": round(current_volume, 2),
        "average_prior_cumulative_volume": round(avg_prior, 2),
        "sessions_used": len(prior_volumes),
        "cutoff_time_utc": cutoff_time,
    }


def _session_count(candles: dict | None) -> int:
    if not candles:
        return 0
    timestamps = candles.get("timestamps") or candles.get("timestamp") or candles.get("t") or []
    dates = set()
    for ts in timestamps:
        try:
            dates.add(datetime.fromtimestamp(float(ts), timezone.utc).date().isoformat())
        except Exception:
            continue
    return len(dates)


def _dedupe_symbols(symbols: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in symbols:
        symbol = str(raw or "").upper().strip()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        result.append(symbol)
    return result


def _same_direction(left: float | None, right: float | None) -> bool | None:
    if left is None or right is None:
        return None
    if abs(left) < 0.1 or abs(right) < 0.1:
        return None
    return (left > 0 and right > 0) or (left < 0 and right < 0)


def _return_bias(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value > 0.2:
        return "bullish"
    if value < -0.2:
        return "bearish"
    return "neutral"


def _fmt(value: Any) -> str:
    return "n/a" if value is None else str(value)
