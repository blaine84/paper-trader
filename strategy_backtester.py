"""
Strategy Backtester
Evaluates dynamic strategies against historical OHLCV data using the
existing backtest engine's data fetching and indicator pipeline.

Wraps backtest.fetch_data() and backtest.add_indicators() with dynamic
condition evaluation based on a strategy's ideal_conditions JSON.
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from backtest import fetch_data, add_indicators
from db.schema import get_session, AgentMemory

log = logging.getLogger(__name__)


class StrategyBacktester:
    """Evaluates dynamic strategies against historical OHLCV data."""

    def __init__(
        self,
        engine,
        days: int = 365,
        risk_reward: float = 2.0,
        stop_atr_mult: float = 1.5,
        max_holding_bars: int = 5,
    ):
        self.engine = engine
        self.days = days
        self.risk_reward = risk_reward
        self.stop_atr_mult = stop_atr_mult
        self.max_holding_bars = max_holding_bars

    def evaluate_conditions(
        self, row: pd.Series, prev_row: pd.Series, conditions: dict
    ) -> str:
        """
        Evaluate ideal_conditions against a candle row with indicators.
        Returns "LONG", "SHORT", or "HOLD".

        Maps condition keys to indicator columns:
          - rsi_at_entry   → rsi (comparison)
          - above_vwap     → close > vwap (boolean)
          - ema_trend      → ema9 vs ema21 (equality)
          - bb_position    → close vs bb_upper/bb_lower (position check)
          - premarket_gap_pct → gap_pct (comparison)
          - vol_ratio / premarket_volume_rank → vol_ratio (threshold)
          - market_regime  → skipped (not available from indicators)
        """
        if not conditions:
            return "HOLD"

        # Keys that cannot be evaluated from indicator data alone
        SKIP_KEYS = {
            "market_regime",
            "entry_timing",
            "float_profile",
            "catalyst_required",
            "catalyst_type",
        }

        evaluable = {k: v for k, v in conditions.items() if k not in SKIP_KEYS}

        if not evaluable:
            return "HOLD"

        long_signals = 0
        short_signals = 0
        total_checks = 0
        passed_checks = 0

        for key, value in evaluable.items():
            result = self._evaluate_single_condition(key, value, row, prev_row)
            if result is None:
                # Condition key not recognized or data missing — skip
                continue
            total_checks += 1
            if result == "PASS_LONG":
                passed_checks += 1
                long_signals += 1
            elif result == "PASS_SHORT":
                passed_checks += 1
                short_signals += 1
            elif result == "PASS":
                passed_checks += 1
            # else: FAIL — condition not met

        if total_checks == 0:
            return "HOLD"

        # All evaluable conditions must pass
        if passed_checks < total_checks:
            return "HOLD"

        # Determine direction from signals
        if long_signals > short_signals:
            return "LONG"
        elif short_signals > long_signals:
            return "SHORT"

        # If no directional bias from conditions, use EMA trend as tiebreaker
        if row.get("ema9") is not None and row.get("ema21") is not None:
            if row["ema9"] > row["ema21"]:
                return "LONG"
            elif row["ema9"] < row["ema21"]:
                return "SHORT"

        return "HOLD"

    def _evaluate_single_condition(
        self, key: str, value, row: pd.Series, prev_row: pd.Series
    ) -> str | None:
        """
        Evaluate a single condition. Returns:
          "PASS_LONG"  — condition met, suggests long
          "PASS_SHORT" — condition met, suggests short
          "PASS"       — condition met, no directional bias
          "FAIL"       — condition not met
          None         — condition not evaluable (skip)
        """
        if key == "rsi_at_entry":
            rsi = row.get("rsi")
            if rsi is None or pd.isna(rsi):
                return None
            return self._eval_comparison(rsi, value)

        elif key == "above_vwap":
            close = row.get("close")
            vwap = row.get("vwap")
            if close is None or vwap is None or pd.isna(close) or pd.isna(vwap):
                return None
            is_above = close > vwap
            if isinstance(value, bool):
                if value and is_above:
                    return "PASS_LONG"
                elif not value and not is_above:
                    return "PASS_SHORT"
                return "FAIL"
            return "PASS" if is_above else "FAIL"

        elif key == "ema_trend":
            ema9 = row.get("ema9")
            ema21 = row.get("ema21")
            if ema9 is None or ema21 is None or pd.isna(ema9) or pd.isna(ema21):
                return None
            current_trend = "bullish" if ema9 > ema21 else "bearish"
            if isinstance(value, list):
                if current_trend in value:
                    return "PASS_LONG" if current_trend == "bullish" else "PASS_SHORT"
                return "FAIL"
            elif isinstance(value, str):
                if current_trend == value.lower():
                    return "PASS_LONG" if current_trend == "bullish" else "PASS_SHORT"
                return "FAIL"
            return None

        elif key == "bb_position":
            close = row.get("close")
            bb_upper = row.get("bb_upper")
            bb_lower = row.get("bb_lower")
            if any(
                v is None or (isinstance(v, float) and pd.isna(v))
                for v in [close, bb_upper, bb_lower]
            ):
                return None
            if isinstance(value, list):
                for pos in value:
                    if pos == "outside_upper" and close > bb_upper:
                        return "PASS_SHORT"  # overextended high → fade short
                    if pos == "outside_lower" and close < bb_lower:
                        return "PASS_LONG"  # overextended low → fade long
                    if pos == "inside" and bb_lower <= close <= bb_upper:
                        return "PASS"
                return "FAIL"
            elif isinstance(value, str):
                if value == "outside_upper" and close > bb_upper:
                    return "PASS_SHORT"
                if value == "outside_lower" and close < bb_lower:
                    return "PASS_LONG"
                if value == "inside" and bb_lower <= close <= bb_upper:
                    return "PASS"
                return "FAIL"
            return None

        elif key == "premarket_gap_pct":
            gap_pct = row.get("gap_pct")
            if gap_pct is None or pd.isna(gap_pct):
                return None
            return self._eval_comparison(gap_pct, value)

        elif key in ("vol_ratio", "premarket_volume_rank"):
            vol_ratio = row.get("vol_ratio")
            if vol_ratio is None or pd.isna(vol_ratio):
                return None
            if isinstance(value, list):
                # Map rank labels to thresholds
                thresholds = {
                    "low": 0.5,
                    "medium": 1.0,
                    "high": 1.5,
                    "extreme": 2.5,
                }
                for rank in value:
                    threshold = thresholds.get(rank, 1.0)
                    if vol_ratio >= threshold:
                        return "PASS"
                return "FAIL"
            elif isinstance(value, (int, float)):
                return "PASS" if vol_ratio >= value else "FAIL"
            elif isinstance(value, str):
                return self._eval_comparison(vol_ratio, value)
            return None

        # Unknown condition key
        return None

    def _eval_comparison(self, actual: float, spec) -> str | None:
        """
        Evaluate a comparison spec like "> 2.0", "< 20", or a range.
        Returns "PASS", "PASS_LONG", "PASS_SHORT", or "FAIL".
        """
        if isinstance(spec, (int, float)):
            return "PASS" if actual >= spec else "FAIL"

        if isinstance(spec, str):
            spec_clean = spec.strip()
            # Handle compound specs like "> 80 (short) or < 20 (long)"
            if " or " in spec_clean.lower():
                parts = spec_clean.lower().split(" or ")
                for part in parts:
                    part = part.strip()
                    result = self._eval_single_comparison(actual, part)
                    if result and result != "FAIL":
                        return result
                return "FAIL"
            return self._eval_single_comparison(actual, spec_clean)

        if isinstance(spec, list):
            # Range: [low, high]
            if len(spec) == 2 and all(isinstance(x, (int, float)) for x in spec):
                return "PASS" if spec[0] <= actual <= spec[1] else "FAIL"

        return None

    def _eval_single_comparison(self, actual: float, spec: str) -> str | None:
        """Evaluate a single comparison string like '> 2.0' or '> 80 (short)'."""
        spec = spec.strip()
        direction = None

        # Extract directional hint
        if "(short)" in spec.lower():
            direction = "SHORT"
            spec = spec.lower().replace("(short)", "").strip()
        elif "(long)" in spec.lower():
            direction = "LONG"
            spec = spec.lower().replace("(long)", "").strip()

        # Remove any trailing text after the number
        try:
            if spec.startswith(">="):
                threshold = float(spec[2:].strip().split()[0])
                passed = actual >= threshold
            elif spec.startswith(">"):
                threshold = float(spec[1:].strip().split()[0])
                passed = actual > threshold
            elif spec.startswith("<="):
                threshold = float(spec[2:].strip().split()[0])
                passed = actual <= threshold
            elif spec.startswith("<"):
                threshold = float(spec[1:].strip().split()[0])
                passed = actual < threshold
            else:
                # Try parsing as a plain number
                threshold = float(spec.split()[0])
                passed = actual >= threshold
        except (ValueError, IndexError):
            return None

        if not passed:
            return "FAIL"

        if direction == "SHORT":
            return "PASS_SHORT"
        elif direction == "LONG":
            return "PASS_LONG"
        return "PASS"

    def _simulate_trade(self, df: pd.DataFrame, i: int, signal: str) -> dict | None:
        """
        Simulate a trade starting at candle index i with the given signal.
        Computes ATR-based stop/target and scans subsequent candles for exit.

        Returns a trade dict or None if ATR is invalid.
        """
        row = df.iloc[i]
        entry = row["close"]
        atr = row.get("atr")

        if atr is None or pd.isna(atr) or atr <= 0:
            return None

        if signal == "LONG":
            stop = entry - atr * self.stop_atr_mult
            target = entry + (entry - stop) * self.risk_reward
        else:  # SHORT
            stop = entry + atr * self.stop_atr_mult
            target = entry - (stop - entry) * self.risk_reward

        # Scan subsequent candles for exit
        exit_price = None
        exit_reason = "timeout"

        for j in range(i + 1, min(i + 1 + self.max_holding_bars, len(df))):
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
            else:  # SHORT
                if future["high"] >= stop:
                    exit_price = stop
                    exit_reason = "stop"
                    break
                if future["low"] <= target:
                    exit_price = target
                    exit_reason = "target"
                    break

        if exit_price is None:
            # Timeout: exit at last available candle's close
            timeout_idx = min(i + self.max_holding_bars, len(df) - 1)
            exit_price = df.iloc[timeout_idx]["close"]

        if signal == "LONG":
            pnl_pct = (exit_price - entry) / entry * 100
        else:
            pnl_pct = (entry - exit_price) / entry * 100

        return {
            "date": df.index[i].strftime("%Y-%m-%d")
            if hasattr(df.index[i], "strftime")
            else str(df.index[i]),
            "symbol": "",  # filled by caller
            "signal": signal,
            "entry_price": round(entry, 4),
            "stop_price": round(stop, 4),
            "target_price": round(target, 4),
            "exit_price": round(exit_price, 4),
            "exit_reason": exit_reason,
            "pnl_pct": round(pnl_pct, 4),
        }

    def compute_summary(self, trades: list[dict]) -> dict:
        """
        Compute summary statistics from a trade log.
        Returns dict with total_trades, win_rate, avg_pnl_pct,
        max_drawdown, and sharpe_ratio.
        """
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_pnl_pct": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
            }

        total = len(trades)
        pnl_values = [t["pnl_pct"] for t in trades]
        wins = sum(1 for p in pnl_values if p > 0)
        win_rate = wins / total
        avg_pnl = sum(pnl_values) / total

        # Max drawdown: largest peak-to-trough decline in cumulative P&L
        cumulative = np.cumsum(pnl_values)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Sharpe ratio: mean / std of returns (annualized not needed for
        # relative comparison; use raw ratio)
        std_pnl = float(np.std(pnl_values, ddof=1)) if total > 1 else 0.0
        sharpe_ratio = avg_pnl / std_pnl if std_pnl > 0 else 0.0

        return {
            "total_trades": total,
            "win_rate": round(win_rate, 4),
            "avg_pnl_pct": round(avg_pnl, 4),
            "max_drawdown": round(max_drawdown, 4),
            "sharpe_ratio": round(sharpe_ratio, 4),
        }

    def run(
        self, strategy, symbols: list[str] | None = None
    ) -> dict:
        """
        Run backtest for a dynamic strategy across symbols.
        Returns a BacktestReport dict.

        Args:
            strategy: A DynamicStrategy ORM object or dict-like with
                      .key, .ideal_conditions, .bias attributes.
            symbols: List of ticker symbols. Defaults to watchlist.
        """
        import os

        if symbols is None:
            symbols = [
                s.strip()
                for s in os.getenv(
                    "WATCHLIST", "SPY,QQQ,IWM,TSLA,NVDA,AMD"
                ).split(",")
            ]

        # Parse ideal_conditions
        conditions = {}
        if hasattr(strategy, "ideal_conditions"):
            raw = strategy.ideal_conditions
            if isinstance(raw, str):
                try:
                    conditions = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    conditions = {}
            elif isinstance(raw, dict):
                conditions = raw

        strategy_key = strategy.key if hasattr(strategy, "key") else str(strategy)
        bias = getattr(strategy, "bias", None) or ""

        all_trades = []
        symbols_tested = []
        start_date = None
        end_date = None

        for sym in symbols:
            try:
                df = fetch_data(sym, self.days)
                if df.empty:
                    log.warning(f"No data for {sym}, skipping")
                    continue

                df = add_indicators(df)
                if len(df) < 5:
                    continue

                symbols_tested.append(sym)

                # Track date range
                if start_date is None or df.index[0] < start_date:
                    start_date = df.index[0]
                if end_date is None or df.index[-1] > end_date:
                    end_date = df.index[-1]

                for i in range(1, len(df)):
                    row = df.iloc[i]
                    prev_row = df.iloc[i - 1]

                    signal = self.evaluate_conditions(row, prev_row, conditions)

                    # Respect strategy bias if specified
                    if signal != "HOLD" and bias:
                        bias_lower = bias.lower()
                        if "long" in bias_lower and "short" not in bias_lower:
                            if signal == "SHORT":
                                signal = "HOLD"
                        elif "short" in bias_lower and "long" not in bias_lower:
                            if signal == "LONG":
                                signal = "HOLD"

                    if signal == "HOLD":
                        continue

                    trade = self._simulate_trade(df, i, signal)
                    if trade:
                        trade["symbol"] = sym
                        all_trades.append(trade)

            except Exception as e:
                log.warning(f"Error backtesting {sym}: {e}")
                continue

        # Build report
        summary = self.compute_summary(all_trades)

        def _fmt_date(dt):
            if dt is None:
                return ""
            if hasattr(dt, "strftime"):
                return dt.strftime("%Y-%m-%d")
            return str(dt)

        report = {
            "metadata": {
                "strategy_key": strategy_key,
                "backtest_start_date": _fmt_date(start_date),
                "backtest_end_date": _fmt_date(end_date),
                "symbols_tested": symbols_tested,
                "generated_at": datetime.utcnow().isoformat(),
            },
            "trade_log": all_trades,
            "summary": summary,
        }

        # Persist as AgentMemory record
        self._persist_report(strategy_key, report)

        return report

    def _persist_report(self, strategy_key: str, report: dict):
        """Save the backtest report as an AgentMemory record."""
        try:
            db = get_session(self.engine)
            memory_key = f"backtest_report_{strategy_key}"

            # Upsert: remove old report if exists
            existing = (
                db.query(AgentMemory)
                .filter_by(agent="strategy_backtester", key=memory_key)
                .first()
            )
            if existing:
                existing.value = json.dumps(report)
                existing.timestamp = datetime.utcnow()
            else:
                record = AgentMemory(
                    agent="strategy_backtester",
                    key=memory_key,
                    value=json.dumps(report),
                )
                db.add(record)

            db.commit()
            db.close()
            log.info(f"Persisted backtest report for {strategy_key}")
        except Exception as e:
            log.error(f"Failed to persist backtest report for {strategy_key}: {e}")
