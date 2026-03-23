"""
Technical indicator calculations using the `ta` library.
Input: candle data dict from FinnhubClient.get_candles()
"""

import pandas as pd
import ta


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

    # Volume
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    last = df.iloc[-1]

    return {
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
