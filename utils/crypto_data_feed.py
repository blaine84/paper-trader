"""Crypto market-data feed — PROTOTYPE (Phase 0 of the crypto expansion spec).

De-risks the one piece of the Bitcoin expansion that is pure plumbing: fetching
crypto quotes and candles and returning them in the SAME canonical provider
shape the Market Data Reliability Layer already consumes, so the existing
normalizer and ``Snapshot`` contract need no changes to accept BTC.

Canonical shapes (identical to ``utils/market_data_reliability/provider_adapter.py``):
- quotes:  ``{"s","c","h","l","o","pc","t","v"}``  (scalars)
- candles: ``{"s","t","o","h","l","c","v"}``        (``t/o/h/l/c/v`` are lists)

This module is INTENTIONALLY standalone and NOT wired into the runtime. It is a
reference implementation to prove the data layer before Phase 1 touches
classification, the session model, scheduler gating, or governance. Nothing here
registers a job, enters a watchlist, or opens a trade.

Providers, in order of preference:
1. Alpaca crypto endpoint (``ALPACA_API_KEY``/``ALPACA_SECRET_KEY``) — consistent
   with the equity path already in ``FinnhubClient``.
2. Coinbase public API — no key required, good for a prototype smoke test.

Design mirrors the spec: crypto has no "market closed" state, quotes are
continuous, and symbols normalize to one canonical internal form.

Related: docs/spec-crypto-asset-class-expansion-2026-09-30.md (R5).
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

log = logging.getLogger(__name__)

# ── Symbol normalization ────────────────────────────────────────────────────
# Canonical internal symbol is "BTC-USD". Accept the common variants the
# equity-oriented watchlist / LLM output might produce.

_CANONICAL = "BTC-USD"
_BTC_VARIANTS = frozenset({"BTC", "BTC-USD", "BTCUSD", "BTC/USD", "XBT", "XBTUSD"})


def normalize_crypto_symbol(sym: str) -> str:
    """Normalize any recognized BTC variant to the canonical internal symbol.

    Raises ``ValueError`` for a symbol this prototype does not recognize as BTC —
    the first pass is Bitcoin only (spec Non-Goal 2).
    """
    upper = (sym or "").upper().strip()
    if upper in _BTC_VARIANTS:
        return _CANONICAL
    raise ValueError(f"Unsupported crypto symbol for prototype (BTC only): {sym!r}")


def _alpaca_symbol(canonical: str) -> str:
    """Alpaca crypto uses 'BTC/USD'."""
    return canonical.replace("-", "/")


def _coinbase_product(canonical: str) -> str:
    """Coinbase uses the product id 'BTC-USD' as-is."""
    return canonical


# ── Alpaca crypto provider ───────────────────────────────────────────────────

_ALPACA_CRYPTO_BASE = os.getenv(
    "ALPACA_CRYPTO_DATA_BASE_URL", "https://data.alpaca.markets"
).rstrip("/")

_ALPACA_TF = {"1": "1Min", "5": "5Min", "15": "15Min", "60": "1Hour", "D": "1Day"}


def _alpaca_headers() -> dict | None:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        return None
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _alpaca_quote(canonical: str) -> dict:
    """Latest trade + daily bar from Alpaca crypto, in canonical quote shape."""
    headers = _alpaca_headers()
    if headers is None:
        raise RuntimeError("alpaca_credentials_missing")

    pair = _alpaca_symbol(canonical)
    # Latest trade for the current price.
    trade_resp = requests.get(
        f"{_ALPACA_CRYPTO_BASE}/v1beta3/crypto/us/latest/trades",
        params={"symbols": pair},
        headers=headers,
        timeout=float(os.getenv("ALPACA_TIMEOUT_SECONDS", "10")),
    )
    trade_resp.raise_for_status()
    trades = (trade_resp.json().get("trades") or {})
    trade = trades.get(pair)
    if not trade or not trade.get("p"):
        raise RuntimeError(f"empty alpaca crypto trade for {pair}")
    price = trade["p"]
    ts = _iso_to_epoch(trade.get("t"))

    # Today's daily bar for o/h/l + previous close (best-effort; non-fatal).
    o = h = l = pc = None
    v = None
    try:
        bar_resp = requests.get(
            f"{_ALPACA_CRYPTO_BASE}/v1beta3/crypto/us/bars",
            params={
                "symbols": pair,
                "timeframe": "1Day",
                "limit": 2,
                "sort": "desc",
            },
            headers=headers,
            timeout=float(os.getenv("ALPACA_TIMEOUT_SECONDS", "10")),
        )
        bar_resp.raise_for_status()
        bars = (bar_resp.json().get("bars") or {}).get(pair) or []
        if bars:
            today = bars[0]
            o, h, l = today.get("o"), today.get("h"), today.get("l")
            v = today.get("v")
            if len(bars) > 1:
                pc = bars[1].get("c")
    except Exception as e:  # noqa: BLE001 — enrichment only
        log.debug("alpaca crypto daily-bar enrichment failed for %s: %s", pair, e)

    return _quote_dict(canonical, c=price, o=o, h=h, l=l, pc=pc, t=ts, v=v)


def _alpaca_candles(canonical: str, resolution: str, days: int) -> dict:
    headers = _alpaca_headers()
    if headers is None:
        raise RuntimeError("alpaca_credentials_missing")

    timeframe = _ALPACA_TF.get(resolution)
    if not timeframe:
        raise ValueError(f"Unsupported crypto resolution: {resolution}")

    pair = _alpaca_symbol(canonical)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    resp = requests.get(
        f"{_ALPACA_CRYPTO_BASE}/v1beta3/crypto/us/bars",
        params={
            "symbols": pair,
            "timeframe": timeframe,
            "start": start.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "end": end.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "limit": 10000,
            "sort": "asc",
        },
        headers=headers,
        timeout=float(os.getenv("ALPACA_TIMEOUT_SECONDS", "10")),
    )
    resp.raise_for_status()
    bars = (resp.json().get("bars") or {}).get(pair) or []
    if not bars:
        raise RuntimeError(f"empty alpaca crypto bars for {pair}")
    return _candles_dict(canonical, bars, ts_key="t", keyset=("o", "h", "l", "c", "v"))


# ── Coinbase public provider (no key — prototype fallback) ───────────────────

_COINBASE_BASE = os.getenv("COINBASE_BASE_URL", "https://api.exchange.coinbase.com").rstrip("/")
# Coinbase granularity is in seconds.
_COINBASE_GRAN = {"1": 60, "5": 300, "15": 900, "60": 3600, "D": 86400}


def _coinbase_quote(canonical: str) -> dict:
    product = _coinbase_product(canonical)
    # /ticker gives price + volume; /stats gives o/h/l + 24h.
    tick = requests.get(f"{_COINBASE_BASE}/products/{product}/ticker", timeout=10)
    tick.raise_for_status()
    t = tick.json()
    price = _to_float(t.get("price"))
    if price is None:
        raise RuntimeError(f"empty coinbase ticker for {product}")

    o = h = l = pc = None
    try:
        st = requests.get(f"{_COINBASE_BASE}/products/{product}/stats", timeout=10)
        st.raise_for_status()
        s = st.json()
        o, h, l = _to_float(s.get("open")), _to_float(s.get("high")), _to_float(s.get("low"))
        pc = o  # Coinbase "open" is the 24h-ago price; closest proxy for prev_close.
    except Exception as e:  # noqa: BLE001
        log.debug("coinbase stats enrichment failed for %s: %s", product, e)

    return _quote_dict(
        canonical, c=price, o=o, h=h, l=l, pc=pc,
        t=_iso_to_epoch(t.get("time")), v=_to_float(t.get("volume")),
    )


def _coinbase_candles(canonical: str, resolution: str, days: int) -> dict:
    product = _coinbase_product(canonical)
    gran = _COINBASE_GRAN.get(resolution)
    if not gran:
        raise ValueError(f"Unsupported crypto resolution: {resolution}")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    resp = requests.get(
        f"{_COINBASE_BASE}/products/{product}/candles",
        params={
            "granularity": gran,
            "start": start.replace(microsecond=0).isoformat(),
            "end": end.replace(microsecond=0).isoformat(),
        },
        timeout=10,
    )
    resp.raise_for_status()
    # Coinbase candle row: [time, low, high, open, close, volume], newest first.
    rows = sorted(resp.json(), key=lambda r: r[0])
    if not rows:
        raise RuntimeError(f"empty coinbase candles for {product}")
    return {
        "s": canonical,
        "t": [int(r[0]) for r in rows],
        "o": [_to_float(r[3]) for r in rows],
        "h": [_to_float(r[2]) for r in rows],
        "l": [_to_float(r[1]) for r in rows],
        "c": [_to_float(r[4]) for r in rows],
        "v": [_to_float(r[5]) for r in rows],
    }


# ── Public entry point (mirrors provider_adapter.fetch_market_data) ──────────

def fetch_crypto_market_data(symbol: str, data_type: str, *, resolution: str = "5",
                             days: int = 2, provider: str | None = None) -> dict:
    """Fetch crypto market data in the reliability layer's canonical shape.

    ``data_type``: "quote" | "candle" | "atr" | "volume" (atr/volume reuse candles).
    ``provider``: force "alpaca" or "coinbase"; default tries Alpaca then Coinbase.
    """
    canonical = normalize_crypto_symbol(symbol)
    data_type = data_type.lower()
    providers = [provider] if provider else _default_provider_order()

    last_err: Exception | None = None
    for prov in providers:
        try:
            if data_type == "quote":
                return _alpaca_quote(canonical) if prov == "alpaca" else _coinbase_quote(canonical)
            if data_type in {"candle", "atr", "volume"}:
                if prov == "alpaca":
                    return _alpaca_candles(canonical, resolution, days)
                return _coinbase_candles(canonical, resolution, days)
            raise ValueError(f"Unsupported crypto data_type: {data_type}")
        except Exception as e:  # noqa: BLE001 — try next provider
            last_err = e
            log.warning("crypto feed provider=%s %s failed for %s: %s",
                        prov, data_type, canonical, e)
            continue
    raise RuntimeError(f"all crypto providers failed for {canonical} {data_type}: {last_err}")


def _default_provider_order() -> list[str]:
    return ["alpaca", "coinbase"] if _alpaca_headers() else ["coinbase"]


# ── Shared helpers ───────────────────────────────────────────────────────────

def _quote_dict(canonical, *, c, o, h, l, pc, t, v) -> dict:
    return {"s": canonical, "c": c, "h": h, "l": l, "o": o, "pc": pc, "t": t, "v": v}


def _candles_dict(canonical, bars, *, ts_key, keyset) -> dict:
    o_k, h_k, l_k, c_k, v_k = keyset
    out: dict[str, Any] = {"s": canonical, "t": [], "o": [], "h": [], "l": [], "c": [], "v": []}
    for b in bars:
        ts = b.get(ts_key)
        if ts is None:
            continue
        out["t"].append(_iso_to_epoch(ts))
        out["o"].append(b.get(o_k))
        out["h"].append(b.get(h_k))
        out["l"].append(b.get(l_k))
        out["c"].append(b.get(c_k))
        out["v"].append(b.get(v_k, 0))
    return out


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso_to_epoch(value: Any) -> int:
    if not value:
        return int(time.time())
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
        except ValueError:
            return int(time.time())
    return int(time.time())


# ── Standalone smoke test ────────────────────────────────────────────────────
# Run directly to prove the plumbing: `python -m utils.crypto_data_feed`
# Uses Coinbase (no key) by default so it works without any credentials.

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(f"Canonical symbol: {normalize_crypto_symbol('btc')}")
    print(f"Provider order:   {_default_provider_order()}")

    print("\n== QUOTE ==")
    q = fetch_crypto_market_data("BTC", "quote")
    for k, v in q.items():
        print(f"  {k}: {v}")

    print("\n== 5m CANDLES (last ~2d) ==")
    c = fetch_crypto_market_data("BTC-USD", "candle", resolution="5", days=1)
    n = len(c["t"])
    print(f"  bars: {n}")
    if n:
        print(f"  first t={c['t'][0]}  last close={c['c'][-1]}")
