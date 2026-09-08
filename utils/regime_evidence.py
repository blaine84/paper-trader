"""Macro-regime evidence helpers.

Builds a small, defensive evidence object from market proxies already available
through yfinance. This is advisory context for Researcher/Quant/Analyst/PM, not
an execution data dependency.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Mapping

log = logging.getLogger(__name__)


Quote = dict[str, float | int | str | None]
QuoteFetcher = Callable[[str], Quote | None]


DEFAULT_SYMBOLS = (
    "^VIX",
    "^TNX",
    "TLT",
    "XLK",
    "VLUE",
    "IWF",
    "IWD",
    "QQQ",
    "SPY",
    "RSP",
    "IWM",
)

REQUIRED_EVIDENCE_KEYS = (
    "vix",
    "ten_year_yield",
    "tech_value_ratio",
    "breadth_proxy",
)


def _to_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _change_pct(price: float | None, prev_close: float | None) -> float | None:
    if price is None or not prev_close:
        return None
    return round(((price - prev_close) / prev_close) * 100, 3)


def _ratio(
    quotes: Mapping[str, Quote],
    numerator: str,
    denominator: str,
) -> dict | None:
    n = quotes.get(numerator) or {}
    d = quotes.get(denominator) or {}
    n_price = _to_float(n.get("price"))
    d_price = _to_float(d.get("price"))
    if n_price is None or not d_price:
        return None

    value = n_price / d_price
    n_prev = _to_float(n.get("prev_close"))
    d_prev = _to_float(d.get("prev_close"))
    prev_value = n_prev / d_prev if n_prev is not None and d_prev else None
    return {
        "ratio": round(value, 4),
        "change_pct": _change_pct(value, prev_value),
        "symbols": [numerator, denominator],
    }


def fetch_yfinance_quote(symbol: str) -> Quote | None:
    """Fetch a compact quote with yfinance fast_info."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    fast = ticker.fast_info
    price = _to_float(fast.get("lastPrice"))
    prev_close = _to_float(fast.get("previousClose"))
    if price is None:
        return None
    return {
        "symbol": symbol,
        "price": price,
        "prev_close": prev_close,
        "change_pct": _change_pct(price, prev_close),
        "timestamp": int(time.time()),
        "provider": "yfinance",
    }


def _ten_year_yield(raw_value: float | None) -> float | None:
    """Normalize Yahoo ^TNX from tenths-of-percent to percent when needed."""
    if raw_value is None:
        return None
    if raw_value > 20:
        return round(raw_value / 10, 3)
    return round(raw_value, 3)


def _round(value, digits: int = 3) -> float | None:
    numeric = _to_float(value)
    return round(numeric, digits) if numeric is not None else None


def build_regime_evidence(fetch_quote: QuoteFetcher | None = None) -> dict:
    """Collect macro proxy evidence without creating a hard trading dependency."""
    fetch = fetch_quote or fetch_yfinance_quote
    quotes: dict[str, Quote] = {}
    errors: dict[str, str] = {}

    for symbol in DEFAULT_SYMBOLS:
        try:
            quote = fetch(symbol)
        except Exception as exc:
            errors[symbol] = type(exc).__name__
            log.warning("Regime evidence quote failed for %s: %s", symbol, exc)
            continue
        if quote:
            quotes[symbol] = quote
        else:
            errors[symbol] = "empty_quote"

    vix_quote = quotes.get("^VIX") or {}
    tnx_quote = quotes.get("^TNX") or {}
    tnx_raw = _to_float(tnx_quote.get("price"))

    evidence = {
        "source": "yfinance",
        "generated_at": int(time.time()),
        "proxies": {
            "vix": {
                "symbol": "^VIX",
                "value": _round(vix_quote.get("price")),
                "change_pct": _round(vix_quote.get("change_pct")),
            } if vix_quote else None,
            "ten_year_yield": {
                "symbol": "^TNX",
                "value_pct": _ten_year_yield(tnx_raw),
                "raw": _round(tnx_raw),
                "change_pct": _round(tnx_quote.get("change_pct")),
            } if tnx_quote else None,
            "rates_pressure": {
                "symbol": "TLT",
                "change_pct": _round((quotes.get("TLT") or {}).get("change_pct")),
            } if quotes.get("TLT") else None,
            "tech_value_ratio": _ratio(quotes, "XLK", "VLUE"),
            "growth_value_ratio": _ratio(quotes, "IWF", "IWD"),
            "risk_appetite": _ratio(quotes, "QQQ", "SPY"),
            "breadth_proxy": _ratio(quotes, "RSP", "SPY"),
            "small_cap_risk": _ratio(quotes, "IWM", "SPY"),
        },
        "errors": errors,
    }

    available_required = [
        key for key in REQUIRED_EVIDENCE_KEYS
        if evidence["proxies"].get(key) is not None
    ]
    available_count = len([v for v in evidence["proxies"].values() if v is not None])
    if len(available_required) == len(REQUIRED_EVIDENCE_KEYS):
        quality = "complete"
    elif available_count:
        quality = "partial"
    else:
        quality = "missing"

    evidence["evidence_quality"] = quality
    evidence["available_required"] = available_required
    evidence["available_count"] = available_count
    return evidence


def format_regime_evidence_for_prompt(evidence: dict | None) -> str:
    """Render regime evidence in a compact, prompt-friendly shape."""
    if not evidence:
        return "Regime evidence unavailable."

    proxies = evidence.get("proxies") or {}
    lines = [
        f"Evidence quality: {evidence.get('evidence_quality', 'missing')}",
        "Macro proxies:",
    ]

    vix = proxies.get("vix")
    if vix:
        lines.append(
            f"- VIX (^VIX): {vix.get('value')} ({vix.get('change_pct')}% vs prev close)"
        )
    tnx = proxies.get("ten_year_yield")
    if tnx:
        lines.append(
            f"- 10Y yield (^TNX): {tnx.get('value_pct')}% ({tnx.get('change_pct')}% vs prev close)"
        )
    rates = proxies.get("rates_pressure")
    if rates:
        lines.append(f"- TLT rates proxy: {rates.get('change_pct')}% vs prev close")

    for label, key in (
        ("Tech/value", "tech_value_ratio"),
        ("Growth/value", "growth_value_ratio"),
        ("QQQ/SPY risk appetite", "risk_appetite"),
        ("RSP/SPY breadth proxy", "breadth_proxy"),
        ("IWM/SPY small-cap risk", "small_cap_risk"),
    ):
        value = proxies.get(key)
        if value:
            lines.append(
                f"- {label}: ratio {value.get('ratio')} ({value.get('change_pct')}% vs prev close)"
            )

    errors = evidence.get("errors") or {}
    if errors:
        missing = ", ".join(sorted(errors.keys()))
        lines.append(f"Missing/degraded proxies: {missing}")

    return "\n".join(lines)
