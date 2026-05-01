"""
Unit tests for StrategyBacktester.
Tests core logic: condition evaluation, trade simulation, summary computation,
and report structure.
"""

import json
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from db.schema import Base, DynamicStrategy, AgentMemory, get_session
from sqlalchemy import create_engine
from strategy_backtester import StrategyBacktester


@pytest.fixture
def in_memory_engine():
    """Create an in-memory SQLite engine with all tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def backtester(in_memory_engine):
    """Create a StrategyBacktester with default params."""
    return StrategyBacktester(
        engine=in_memory_engine,
        days=365,
        risk_reward=2.0,
        stop_atr_mult=1.5,
        max_holding_bars=5,
    )


def _make_row(**overrides):
    """Create a pd.Series mimicking a candle row with indicators."""
    defaults = {
        "open": 100.0,
        "high": 102.0,
        "low": 98.0,
        "close": 101.0,
        "volume": 1_000_000,
        "ema9": 100.5,
        "ema21": 99.5,
        "ema50": 98.0,
        "rsi": 55.0,
        "bb_upper": 105.0,
        "bb_lower": 95.0,
        "atr": 2.0,
        "vwap": 100.0,
        "gap_pct": 1.0,
        "vol_ratio": 1.2,
        "macd_diff": 0.5,
    }
    defaults.update(overrides)
    return pd.Series(defaults)


def _make_df(n=20, base_price=100.0, atr=2.0):
    """Create a simple DataFrame with OHLCV + indicators for simulation."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    data = {
        "open": [base_price + i * 0.5 for i in range(n)],
        "high": [base_price + i * 0.5 + 2 for i in range(n)],
        "low": [base_price + i * 0.5 - 1 for i in range(n)],
        "close": [base_price + i * 0.5 + 0.5 for i in range(n)],
        "volume": [1_000_000] * n,
        "atr": [atr] * n,
        "ema9": [base_price + i * 0.5 for i in range(n)],
        "ema21": [base_price + i * 0.4 for i in range(n)],
        "vwap": [base_price + i * 0.3 for i in range(n)],
        "rsi": [55.0] * n,
        "bb_upper": [base_price + 10] * n,
        "bb_lower": [base_price - 10] * n,
        "gap_pct": [1.0] * n,
        "vol_ratio": [1.2] * n,
    }
    return pd.DataFrame(data, index=dates)


# ─── evaluate_conditions tests ────────────────────────────────────────────────


class TestEvaluateConditions:
    def test_empty_conditions_returns_hold(self, backtester):
        row = _make_row()
        prev = _make_row()
        assert backtester.evaluate_conditions(row, prev, {}) == "HOLD"

    def test_only_skip_keys_returns_hold(self, backtester):
        row = _make_row()
        prev = _make_row()
        conditions = {"market_regime": ["risk_on"], "entry_timing": ["first_15min"]}
        assert backtester.evaluate_conditions(row, prev, conditions) == "HOLD"

    def test_bullish_ema_trend_returns_long(self, backtester):
        row = _make_row(ema9=105.0, ema21=100.0)
        prev = _make_row()
        conditions = {"ema_trend": ["bullish"]}
        assert backtester.evaluate_conditions(row, prev, conditions) == "LONG"

    def test_bearish_ema_trend_returns_short(self, backtester):
        row = _make_row(ema9=95.0, ema21=100.0)
        prev = _make_row()
        conditions = {"ema_trend": ["bearish"]}
        assert backtester.evaluate_conditions(row, prev, conditions) == "SHORT"

    def test_above_vwap_true_returns_long(self, backtester):
        row = _make_row(close=105.0, vwap=100.0, ema9=105.0, ema21=100.0)
        prev = _make_row()
        conditions = {"above_vwap": True}
        assert backtester.evaluate_conditions(row, prev, conditions) == "LONG"

    def test_above_vwap_false_returns_short(self, backtester):
        row = _make_row(close=95.0, vwap=100.0, ema9=95.0, ema21=100.0)
        prev = _make_row()
        conditions = {"above_vwap": False}
        assert backtester.evaluate_conditions(row, prev, conditions) == "SHORT"

    def test_rsi_comparison(self, backtester):
        row = _make_row(rsi=85.0, ema9=95.0, ema21=100.0)
        prev = _make_row()
        conditions = {"rsi_at_entry": "> 80"}
        assert backtester.evaluate_conditions(row, prev, conditions) != "HOLD"

    def test_bb_outside_upper_returns_short(self, backtester):
        row = _make_row(close=110.0, bb_upper=105.0, bb_lower=95.0, ema9=95.0, ema21=100.0)
        prev = _make_row()
        conditions = {"bb_position": ["outside_upper"]}
        assert backtester.evaluate_conditions(row, prev, conditions) == "SHORT"

    def test_bb_outside_lower_returns_long(self, backtester):
        row = _make_row(close=90.0, bb_upper=105.0, bb_lower=95.0, ema9=105.0, ema21=100.0)
        prev = _make_row()
        conditions = {"bb_position": ["outside_lower"]}
        assert backtester.evaluate_conditions(row, prev, conditions) == "LONG"

    def test_vol_ratio_with_rank_labels(self, backtester):
        row = _make_row(vol_ratio=2.0, ema9=105.0, ema21=100.0)
        prev = _make_row()
        conditions = {"premarket_volume_rank": ["high", "extreme"]}
        result = backtester.evaluate_conditions(row, prev, conditions)
        assert result in ("LONG", "SHORT")

    def test_multiple_conditions_all_must_pass(self, backtester):
        # Bullish EMA + above VWAP + high volume → LONG
        row = _make_row(ema9=105.0, ema21=100.0, close=105.0, vwap=100.0, vol_ratio=2.0)
        prev = _make_row()
        conditions = {
            "ema_trend": ["bullish"],
            "above_vwap": True,
            "premarket_volume_rank": ["high"],
        }
        assert backtester.evaluate_conditions(row, prev, conditions) == "LONG"

    def test_mixed_conditions_one_fails_returns_hold(self, backtester):
        # Bullish EMA but below VWAP → HOLD (not all conditions met)
        row = _make_row(ema9=105.0, ema21=100.0, close=95.0, vwap=100.0)
        prev = _make_row()
        conditions = {"ema_trend": ["bullish"], "above_vwap": True}
        assert backtester.evaluate_conditions(row, prev, conditions) == "HOLD"

    def test_signal_always_valid(self, backtester):
        """Any combination of conditions should return LONG, SHORT, or HOLD."""
        row = _make_row()
        prev = _make_row()
        for conditions in [
            {"rsi_at_entry": "> 50"},
            {"above_vwap": True},
            {"ema_trend": "bullish"},
            {"bb_position": "inside"},
            {"premarket_gap_pct": "> 2.0"},
            {"vol_ratio": 1.5},
        ]:
            result = backtester.evaluate_conditions(row, prev, conditions)
            assert result in ("LONG", "SHORT", "HOLD")


# ─── _simulate_trade tests ───────────────────────────────────────────────────


class TestSimulateTrade:
    def test_long_trade_hits_target(self, backtester):
        df = _make_df(n=10, base_price=100.0, atr=2.0)
        # Make future candles go high enough to hit target
        # entry = close at i=1 = 101.0, stop = 101.0 - 3.0 = 98.0, target = 101.0 + 6.0 = 107.0
        df.loc[df.index[3], "high"] = 110.0
        trade = backtester._simulate_trade(df, 1, "LONG")
        assert trade is not None
        assert trade["exit_reason"] == "target"
        assert trade["signal"] == "LONG"

    def test_long_trade_hits_stop(self, backtester):
        df = _make_df(n=10, base_price=100.0, atr=2.0)
        # Make future candles go low enough to hit stop
        df.loc[df.index[2], "low"] = 90.0
        trade = backtester._simulate_trade(df, 1, "LONG")
        assert trade is not None
        assert trade["exit_reason"] == "stop"

    def test_trade_timeout(self, backtester):
        df = _make_df(n=10, base_price=100.0, atr=2.0)
        # Keep prices in a narrow range so neither stop nor target is hit
        for j in range(2, 8):
            df.loc[df.index[j], "high"] = 102.0
            df.loc[df.index[j], "low"] = 100.0
            df.loc[df.index[j], "close"] = 101.0
        trade = backtester._simulate_trade(df, 1, "LONG")
        assert trade is not None
        assert trade["exit_reason"] == "timeout"

    def test_short_trade_structure(self, backtester):
        df = _make_df(n=10, base_price=100.0, atr=2.0)
        df.loc[df.index[2], "low"] = 85.0  # hit target for short
        trade = backtester._simulate_trade(df, 1, "SHORT")
        assert trade is not None
        assert trade["signal"] == "SHORT"
        assert trade["stop_price"] > trade["entry_price"]
        assert trade["target_price"] < trade["entry_price"]

    def test_long_stop_target_ordering(self, backtester):
        df = _make_df(n=10, base_price=100.0, atr=2.0)
        trade = backtester._simulate_trade(df, 1, "LONG")
        assert trade is not None
        assert trade["stop_price"] < trade["entry_price"] < trade["target_price"]

    def test_short_stop_target_ordering(self, backtester):
        df = _make_df(n=10, base_price=100.0, atr=2.0)
        trade = backtester._simulate_trade(df, 1, "SHORT")
        assert trade is not None
        assert trade["target_price"] < trade["entry_price"] < trade["stop_price"]

    def test_zero_atr_returns_none(self, backtester):
        df = _make_df(n=10, base_price=100.0, atr=0.0)
        trade = backtester._simulate_trade(df, 1, "LONG")
        assert trade is None

    def test_trade_has_required_keys(self, backtester):
        df = _make_df(n=10, base_price=100.0, atr=2.0)
        trade = backtester._simulate_trade(df, 1, "LONG")
        assert trade is not None
        required_keys = {
            "date", "symbol", "signal", "entry_price", "stop_price",
            "target_price", "exit_price", "exit_reason", "pnl_pct",
        }
        assert required_keys.issubset(trade.keys())


# ─── compute_summary tests ───────────────────────────────────────────────────


class TestComputeSummary:
    def test_empty_trades(self, backtester):
        summary = backtester.compute_summary([])
        assert summary["total_trades"] == 0
        assert summary["win_rate"] == 0.0

    def test_all_winners(self, backtester):
        trades = [{"pnl_pct": 2.0}, {"pnl_pct": 3.0}, {"pnl_pct": 1.5}]
        summary = backtester.compute_summary(trades)
        assert summary["total_trades"] == 3
        assert summary["win_rate"] == 1.0
        assert abs(summary["avg_pnl_pct"] - (2.0 + 3.0 + 1.5) / 3) < 0.01

    def test_mixed_trades(self, backtester):
        trades = [
            {"pnl_pct": 5.0},
            {"pnl_pct": -2.0},
            {"pnl_pct": 3.0},
            {"pnl_pct": -1.0},
        ]
        summary = backtester.compute_summary(trades)
        assert summary["total_trades"] == 4
        assert summary["win_rate"] == 0.5
        assert abs(summary["avg_pnl_pct"] - 1.25) < 0.01

    def test_max_drawdown_negative(self, backtester):
        trades = [
            {"pnl_pct": 5.0},
            {"pnl_pct": -3.0},
            {"pnl_pct": -4.0},
            {"pnl_pct": 2.0},
        ]
        summary = backtester.compute_summary(trades)
        assert summary["max_drawdown"] <= 0.0

    def test_summary_has_required_keys(self, backtester):
        trades = [{"pnl_pct": 1.0}]
        summary = backtester.compute_summary(trades)
        required = {"total_trades", "win_rate", "avg_pnl_pct", "max_drawdown", "sharpe_ratio"}
        assert required.issubset(summary.keys())


# ─── run() integration test with mocked yfinance ─────────────────────────────


class TestRun:
    def test_run_produces_valid_report(self, backtester, in_memory_engine):
        """Test run() with mocked fetch_data to avoid network calls."""
        df = _make_df(n=50, base_price=100.0, atr=2.0)
        # Set conditions that will trigger signals
        for i in range(len(df)):
            df.iloc[i, df.columns.get_loc("ema9")] = 105.0
            df.iloc[i, df.columns.get_loc("ema21")] = 100.0
            df.iloc[i, df.columns.get_loc("close")] = 106.0
            df.iloc[i, df.columns.get_loc("vwap")] = 100.0

        strategy = MagicMock()
        strategy.key = "test_strategy"
        strategy.ideal_conditions = json.dumps({
            "ema_trend": ["bullish"],
            "above_vwap": True,
        })
        strategy.bias = "LONG"

        with patch("strategy_backtester.fetch_data", return_value=df), \
             patch("strategy_backtester.add_indicators", return_value=df):
            report = backtester.run(strategy, symbols=["SPY"])

        # Verify report structure
        assert "metadata" in report
        assert "trade_log" in report
        assert "summary" in report

        meta = report["metadata"]
        assert meta["strategy_key"] == "test_strategy"
        assert "backtest_start_date" in meta
        assert "backtest_end_date" in meta
        assert "symbols_tested" in meta
        assert "generated_at" in meta

        summary = report["summary"]
        assert "total_trades" in summary
        assert "win_rate" in summary
        assert "avg_pnl_pct" in summary
        assert "max_drawdown" in summary
        assert "sharpe_ratio" in summary

        # Verify trades have correct structure
        for trade in report["trade_log"]:
            assert "date" in trade
            assert "symbol" in trade
            assert trade["signal"] in ("LONG", "SHORT")
            assert "entry_price" in trade
            assert "stop_price" in trade
            assert "target_price" in trade
            assert "exit_price" in trade
            assert trade["exit_reason"] in ("stop", "target", "timeout")
            assert "pnl_pct" in trade

    def test_run_persists_to_agent_memory(self, backtester, in_memory_engine):
        """Verify the report is saved as an AgentMemory record."""
        df = _make_df(n=20, base_price=100.0, atr=2.0)

        strategy = MagicMock()
        strategy.key = "persist_test"
        strategy.ideal_conditions = json.dumps({"ema_trend": ["bullish"]})
        strategy.bias = ""

        with patch("strategy_backtester.fetch_data", return_value=df), \
             patch("strategy_backtester.add_indicators", return_value=df):
            backtester.run(strategy, symbols=["SPY"])

        db = get_session(in_memory_engine)
        record = db.query(AgentMemory).filter_by(
            agent="strategy_backtester",
            key="backtest_report_persist_test",
        ).first()
        assert record is not None
        report_data = json.loads(record.value)
        assert "metadata" in report_data
        assert "trade_log" in report_data
        assert "summary" in report_data
        db.close()

    def test_run_handles_empty_data(self, backtester, in_memory_engine):
        """Verify run() handles symbols with no data gracefully."""
        strategy = MagicMock()
        strategy.key = "empty_test"
        strategy.ideal_conditions = json.dumps({})
        strategy.bias = ""

        with patch("strategy_backtester.fetch_data", return_value=pd.DataFrame()):
            report = backtester.run(strategy, symbols=["FAKE"])

        assert report["summary"]["total_trades"] == 0
        assert report["trade_log"] == []
