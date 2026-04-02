"""
Backtester
Rule-based backtest using the same technical indicators as the Analyst.
Pulls historical daily OHLCV via yfinance and simulates entries/exits
based on signal rules for each strategy type.

Usage:
  python backtest.py                          # all symbols, all strategies, 1 year
  python backtest.py --symbols SPY QQQ TSLA  # specific symbols
  python backtest.py --days 180              # last 180 days
  python backtest.py --strategy gap_and_go   # single strategy
  python backtest.py --export results.csv    # export to CSV
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

DEFAULT_SYMBOLS = [s.strip() for s in os.getenv("WATCHLIST", "SPY,QQQ,IWM,TSLA,NVDA,AMD").split(",")]
DEFAULT_DAYS = 365
RISK_REWARD = 2.0       # target = entry + (entry - stop) * RR
STOP_ATR_MULT = 1.5     # stop = entry - ATR * multiplier
COMMISSION = 0.0        # paper trading, no commission


def fetch_data(symbol: str, days: int) -> pd.DataFrame:
    import yfinance as yf
    end = datetime.utcnow()
    start = end - timedelta(days=days + 60)  # extra buffer for indicator warmup
    df = yf.download(symbol, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), interval="1d",
                     progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    return df.dropna()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema9"]  = ta.trend.ema_indicator(df["close"], window=9)
    df["ema21"] = ta.trend.ema_indicator(df["close"], window=21)
    df["ema50"] = ta.trend.ema_indicator(df["close"], window=50)
    macd = ta.trend.MACD(df["close"])
    df["macd_diff"] = macd.macd_diff()
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    df["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1) * 100
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    return df.dropna()


# ─── SIGNAL RULES ─────────────────────────────────────────────────────────────

def signal_gap_and_go(row) -> str:
    """Gap >2% with volume, EMA bullish, RSI not overextended."""
    if (row["gap_pct"] > 2.0 and
        row["vol_ratio"] > 1.5 and
        row["ema9"] > row["ema21"] and
        row["rsi"] < 75 and
        row["close"] > row["vwap"]):
        return "LONG"
    if (row["gap_pct"] < -2.0 and
        row["vol_ratio"] > 1.5 and
        row["ema9"] < row["ema21"] and
        row["rsi"] > 25):
        return "SHORT"
    return "HOLD"


def signal_vwap_reclaim(row, prev_row) -> str:
    """Price was below VWAP, now reclaims it with bullish EMA trend."""
    if (prev_row["close"] < prev_row["vwap"] and
        row["close"] > row["vwap"] and
        row["ema9"] > row["ema21"] and
        row["rsi"] > 40 and row["rsi"] < 70):
        return "LONG"
    return "HOLD"


def signal_orb(row, prev_row) -> str:
    """Price breaks above prior day high with volume."""
    if (row["close"] > prev_row["high"] and
        row["vol_ratio"] > 1.2 and
        row["ema9"] > row["ema21"]):
        return "LONG"
    if (row["close"] < prev_row["low"] and
        row["vol_ratio"] > 1.2 and
        row["ema9"] < row["ema21"]):
        return "SHORT"
    return "HOLD"


def signal_trend_pullback(row, prev_row) -> str:
    """Pullback to EMA9/21 in an uptrend."""
    if (row["ema9"] > row["ema21"] > row["ema50"] and
        row["close"] > row["vwap"] and
        prev_row["close"] < prev_row["ema21"] and
        row["close"] > row["ema21"] and
        row["rsi"] > 40 and row["rsi"] < 65):
        return "LONG"
    return "HOLD"


def signal_momentum_fade(row) -> str:
    """Fade overextended RSI with BB breach."""
    if row["rsi"] > 80 and row["close"] > row["bb_upper"] and row["vol_ratio"] > 2.0:
        return "SHORT"
    if row["rsi"] < 20 and row["close"] < row["bb_lower"] and row["vol_ratio"] > 2.0:
        return "LONG"
    return "HOLD"


SIGNAL_FNS = {
    "gap_and_go":     lambda r, p: signal_gap_and_go(r),
    "vwap_reclaim":   signal_vwap_reclaim,
    "orb":            signal_orb,
    "trend_pullback": signal_trend_pullback,
    "momentum_fade":  lambda r, p: signal_momentum_fade(r),
}


# ─── BACKTEST ENGINE ──────────────────────────────────────────────────────────

def backtest_strategy(df: pd.DataFrame, symbol: str, strategy: str, days: int) -> list[dict]:
    cutoff = datetime.utcnow() - timedelta(days=days)
    df = df[df.index >= pd.Timestamp(cutoff)].copy()
    if len(df) < 5:
        return []

    signal_fn = SIGNAL_FNS[strategy]
    trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        signal = signal_fn(row, prev)
        if signal == "HOLD":
            continue

        entry = row["close"]
        atr = row["atr"]
        if signal == "LONG":
            stop = entry - atr * STOP_ATR_MULT
            target = entry + (entry - stop) * RISK_REWARD
        else:
            stop = entry + atr * STOP_ATR_MULT
            target = entry - (entry - stop) * RISK_REWARD

        # Simulate exit over next 5 bars
        exit_price = None
        exit_reason = "timeout"
        for j in range(i + 1, min(i + 6, len(df))):
            future = df.iloc[j]
            if signal == "LONG":
                if future["low"] <= stop:
                    exit_price = stop
                    exit_reason = "stop"
                    break
                if future["high"] >= target:
                    exit_price = target
                    exit_reason = "target"
                    break
            else:
                if future["high"] >= stop:
                    exit_price = stop
                    exit_reason = "stop"
                    break
                if future["low"] <= target:
                    exit_price = target
                    exit_reason = "target"
                    break

        if exit_price is None:
            exit_price = df.iloc[min(i + 5, len(df) - 1)]["close"]

        if signal == "LONG":
            pnl_pct = (exit_price - entry) / entry * 100
        else:
            pnl_pct = (entry - exit_price) / entry * 100

        trades.append({
            "date": df.index[i].strftime("%Y-%m-%d"),
            "symbol": symbol,
            "strategy": strategy,
            "signal": signal,
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "target": round(target, 2),
            "exit": round(exit_price, 2),
            "exit_reason": exit_reason,
            "pnl_pct": round(pnl_pct, 2),
            "win": pnl_pct > 0,
        })

    return trades


# ─── REPORTING ────────────────────────────────────────────────────────────────

def print_report(all_trades: list[dict]):
    if not all_trades:
        print("No trades generated.")
        return

    df = pd.DataFrame(all_trades)
    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS — {len(df)} trades")
    print(f"{'='*60}")

    # Overall
    wins = df["win"].sum()
    total = len(df)
    avg_pnl = df["pnl_pct"].mean()
    total_pnl = df["pnl_pct"].sum()
    win_rate = wins / total * 100
    print(f"\n  Overall: {total} trades | {win_rate:.1f}% win rate | avg {avg_pnl:+.2f}% | total {total_pnl:+.2f}%")

    # By strategy
    print(f"\n  {'Strategy':<20} {'Trades':>7} {'Win%':>7} {'Avg P&L':>9} {'Total P&L':>10}")
    print(f"  {'-'*55}")
    for strat, g in df.groupby("strategy"):
        wr = g["win"].mean() * 100
        ap = g["pnl_pct"].mean()
        tp = g["pnl_pct"].sum()
        print(f"  {strat:<20} {len(g):>7} {wr:>6.1f}% {ap:>+8.2f}% {tp:>+9.2f}%")

    # By symbol
    print(f"\n  {'Symbol':<10} {'Trades':>7} {'Win%':>7} {'Avg P&L':>9} {'Total P&L':>10}")
    print(f"  {'-'*45}")
    for sym, g in df.groupby("symbol"):
        wr = g["win"].mean() * 100
        ap = g["pnl_pct"].mean()
        tp = g["pnl_pct"].sum()
        print(f"  {sym:<10} {len(g):>7} {wr:>6.1f}% {ap:>+8.2f}% {tp:>+9.2f}%")

    # Exit reasons
    print(f"\n  Exit reasons: {df['exit_reason'].value_counts().to_dict()}")
    print(f"{'='*60}\n")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Paper Trader Backtester")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--strategy", type=str, default=None, help="Single strategy to test")
    parser.add_argument("--export", type=str, default=None, help="Export results to CSV path")
    args = parser.parse_args()

    strategies = [args.strategy] if args.strategy else list(SIGNAL_FNS.keys())
    all_trades = []

    for sym in args.symbols:
        print(f"Fetching {sym}...", end=" ", flush=True)
        df = fetch_data(sym, args.days)
        if df.empty:
            print("no data")
            continue
        df = add_indicators(df)
        print(f"{len(df)} bars")

        for strat in strategies:
            trades = backtest_strategy(df, sym, strat, args.days)
            all_trades.extend(trades)

    print_report(all_trades)

    if args.export and all_trades:
        pd.DataFrame(all_trades).to_csv(args.export, index=False)
        print(f"Exported to {args.export}")


if __name__ == "__main__":
    main()


def run_backtest(symbols: list[str] = None, days: int = 90, strategies: list[str] = None) -> dict:
    """
    Programmatic entry point for agents.
    Returns a summary dict with win rates and edge scores per strategy.
    """
    symbols = symbols or DEFAULT_SYMBOLS
    strategies = strategies or list(SIGNAL_FNS.keys())
    all_trades = []

    for sym in symbols:
        try:
            df = fetch_data(sym, days)
            if df.empty:
                continue
            df = add_indicators(df)
            for strat in strategies:
                trades = backtest_strategy(df, sym, strat, days)
                all_trades.extend(trades)
        except Exception:
            continue

    if not all_trades:
        return {"strategies": {}, "symbols": {}, "total_trades": 0, "days": days}

    df = pd.DataFrame(all_trades)

    strategy_summary = {}
    for strat, g in df.groupby("strategy"):
        strategy_summary[strat] = {
            "trades": len(g),
            "win_rate": round(g["win"].mean() * 100, 1),
            "avg_pnl_pct": round(g["pnl_pct"].mean(), 2),
            "total_pnl_pct": round(g["pnl_pct"].sum(), 2),
            "has_edge": g["win"].mean() >= 0.50 and g["pnl_pct"].mean() > 0,
        }

    symbol_summary = {}
    for sym, g in df.groupby("symbol"):
        symbol_summary[sym] = {
            "trades": len(g),
            "win_rate": round(g["win"].mean() * 100, 1),
            "avg_pnl_pct": round(g["pnl_pct"].mean(), 2),
        }

    return {
        "strategies": strategy_summary,
        "symbols": symbol_summary,
        "total_trades": len(df),
        "days": days,
        "generated_at": datetime.utcnow().isoformat(),
    }
