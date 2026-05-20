"""
Technical indicator calculations using the `ta` library.
Input: candle data dict from FinnhubClient.get_candles()
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import ta


def _candle_session_dates(candles: dict) -> list[str | None]:
    """Return UTC calendar dates for each candle timestamp, when available."""
    timestamps = candles.get("timestamps") or candles.get("timestamp") or candles.get("t") or []
    dates: list[str | None] = []
    for ts in timestamps:
        try:
            if isinstance(ts, (int, float)):
                dates.append(datetime.fromtimestamp(ts, timezone.utc).date().isoformat())
            else:
                dates.append(str(ts)[:10])
        except Exception:
            dates.append(None)
    return dates


def compute_indicators(candles: dict) -> dict:
    """
    Compute common technical indicators from candle data.
    Returns a dict of current (latest) indicator values.
    """
    if not candles or candles.get("s") == "no_data":
        return {}

    df = pd.DataFrame({
        "open":   candles["open"],
        "high":   candles["high"],
        "low":    candles["low"],
        "close":  candles["close"],
        "volume": candles["volume"],
    })

    session_dates = _candle_session_dates(candles)
    if len(session_dates) == len(df):
        df["session_date"] = session_dates

    if len(df) < 20:
        return {"error": "Not enough candles for indicators"}

    # Trend
    df["ema9"]  = ta.trend.ema_indicator(df["close"], window=9)
    df["ema21"] = ta.trend.ema_indicator(df["close"], window=21)
    df["ema50"] = ta.trend.ema_indicator(df["close"], window=50)
    macd = ta.trend.MACD(df["close"])
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"]   = macd.macd_diff()

    # Momentum
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)

    # Volatility
    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

    # Volume. VWAP must reset each session; get_candles(days=2) intentionally
    # includes yesterday for indicator warmup, so a plain 2-day cumulative VWAP
    # contaminates today's intraday read.
    if "session_date" in df.columns and df["session_date"].notna().any():
        pv = df["close"] * df["volume"]
        grouped_dates = df["session_date"]
        df["vwap"] = pv.groupby(grouped_dates, dropna=False).cumsum() / df["volume"].groupby(grouped_dates, dropna=False).cumsum()
    else:
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    last = df.iloc[-1]
    latest_date = last.get("session_date") if "session_date" in df.columns else None
    today_df = df[df["session_date"] == latest_date] if latest_date else df
    prior_df = df[df["session_date"] != latest_date] if latest_date else df.iloc[0:0]

    result = {
        "price":       round(last["close"], 2),
        "ema9":        round(last["ema9"], 2),
        "ema21":       round(last["ema21"], 2),
        "ema50":       round(last["ema50"], 2),
        "macd":        round(last["macd"], 4),
        "macd_signal": round(last["macd_signal"], 4),
        "macd_diff":   round(last["macd_diff"], 4),
        "rsi":         round(last["rsi"], 2),
        "bb_upper":    round(last["bb_upper"], 2),
        "bb_lower":    round(last["bb_lower"], 2),
        "bb_mid":      round(last["bb_mid"], 2),
        "atr":         round(last["atr"], 2),
        "vwap":        round(last["vwap"], 2),
        "trend":       "bullish" if last["ema9"] > last["ema21"] else "bearish",
        "macd_cross":  "bullish" if last["macd_diff"] > 0 else "bearish",
    }

    if latest_date and not today_df.empty:
        result.update({
            "session_date": latest_date,
            "session_open": round(float(today_df.iloc[0]["open"]), 2),
            "session_high": round(float(today_df["high"].max()), 2),
            "session_low": round(float(today_df["low"].min()), 2),
        })
    if not prior_df.empty:
        result.update({
            "prior_session_high": round(float(prior_df["high"].max()), 2),
            "prior_session_low": round(float(prior_df["low"].min()), 2),
            "prior_session_close": round(float(prior_df.iloc[-1]["close"]), 2),
        })

    return result
