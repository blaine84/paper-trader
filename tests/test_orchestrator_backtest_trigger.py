"""
Tests for the backtest trigger logic in orchestrator.run_pipeline_evaluation().

Verifies that strategies with status="backtest" and no backtest_report_id
get backtested, their report ID is set, and the gate result is applied.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from sqlalchemy import create_engine
from db.schema import Base, DynamicStrategy, AgentMemory, get_session


@pytest.fixture
def engine():
    """In-memory SQLite engine with all tables created."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _make_strategy(engine, **overrides):
    """Helper to create a DynamicStrategy record."""
    defaults = {
        "key": "test_strat",
        "name": "Test Strategy",
        "description": "A test strategy",
        "status": "backtest",
        "pipeline_stage": "backtest",
        "ideal_conditions": "{}",
        "failure_conditions": "[]",
        "execution_notes": "[]",
    }
    defaults.update(overrides)
    db = get_session(engine)
    strat = DynamicStrategy(**defaults)
    db.add(strat)
    db.commit()
    db.refresh(strat)
    db.expunge(strat)
    db.close()
    return strat


def _passing_report(strategy_key="test_strat"):
    """A backtest report that passes the gate (>=50 trades, >0.55 win rate)."""
    return {
        "metadata": {
            "strategy_key": strategy_key,
            "backtest_start_date": "2024-01-01",
            "backtest_end_date": "2025-01-01",
            "symbols_tested": ["SPY", "QQQ"],
            "generated_at": datetime.utcnow().isoformat(),
        },
        "trade_log": [],
        "summary": {
            "total_trades": 60,
            "win_rate": 0.62,
            "avg_pnl_pct": 1.5,
            "max_drawdown": -3.0,
            "sharpe_ratio": 1.2,
        },
    }


def _failing_report(strategy_key="test_strat"):
    """A backtest report that fails the gate (<50 trades)."""
    return {
        "metadata": {
            "strategy_key": strategy_key,
            "backtest_start_date": "2024-01-01",
            "backtest_end_date": "2025-01-01",
            "symbols_tested": ["SPY"],
            "generated_at": datetime.utcnow().isoformat(),
        },
        "trade_log": [],
        "summary": {
            "total_trades": 10,
            "win_rate": 0.40,
            "avg_pnl_pct": -0.5,
            "max_drawdown": -8.0,
            "sharpe_ratio": -0.3,
        },
    }


class TestBacktestTrigger:
    """Tests for the backtest trigger in orchestrator.run_pipeline_evaluation()."""

    @patch("orchestrator.get_engine")
    @patch("strategy_backtester.StrategyBacktester")
    def test_triggers_backtest_for_pending_strategy(self, MockBacktester, mock_get_engine, engine):
        """A strategy with status=backtest and no report_id should be backtested."""
        mock_get_engine.return_value = engine
        strat = _make_strategy(engine, key="pending_bt")

        report = _passing_report("pending_bt")
        mock_instance = MagicMock()
        mock_instance.run.return_value = report
        MockBacktester.return_value = mock_instance

        from orchestrator import run_pipeline_evaluation
        run_pipeline_evaluation()

        # Verify backtester was instantiated and run was called
        MockBacktester.assert_called_once_with(engine)
        mock_instance.run.assert_called_once()

        # Verify backtest_report_id was set
        db = get_session(engine)
        updated = db.query(DynamicStrategy).filter_by(key="pending_bt").first()
        assert updated.backtest_report_id == "backtest_report_pending_bt"
        db.close()

    @patch("orchestrator.get_engine")
    @patch("strategy_backtester.StrategyBacktester")
    def test_advances_on_passing_backtest(self, MockBacktester, mock_get_engine, engine):
        """A passing backtest should advance the strategy to paper_trade."""
        mock_get_engine.return_value = engine
        strat = _make_strategy(engine, key="pass_bt")

        report = _passing_report("pass_bt")
        mock_instance = MagicMock()
        mock_instance.run.return_value = report
        MockBacktester.return_value = mock_instance

        from orchestrator import run_pipeline_evaluation
        run_pipeline_evaluation()

        db = get_session(engine)
        updated = db.query(DynamicStrategy).filter_by(key="pass_bt").first()
        assert updated.status == "paper_trade"
        assert updated.pipeline_stage == "paper_trade"
        assert updated.paper_trade_start_date is not None
        db.close()

    @patch("orchestrator.get_engine")
    @patch("strategy_backtester.StrategyBacktester")
    def test_fails_on_failing_backtest(self, MockBacktester, mock_get_engine, engine):
        """A failing backtest should set the strategy to backtest_failed."""
        mock_get_engine.return_value = engine
        strat = _make_strategy(engine, key="fail_bt")

        report = _failing_report("fail_bt")
        mock_instance = MagicMock()
        mock_instance.run.return_value = report
        MockBacktester.return_value = mock_instance

        from orchestrator import run_pipeline_evaluation
        run_pipeline_evaluation()

        db = get_session(engine)
        updated = db.query(DynamicStrategy).filter_by(key="fail_bt").first()
        assert updated.status == "backtest_failed"
        assert updated.failure_reason is not None
        db.close()

    @patch("orchestrator.get_engine")
    @patch("strategy_backtester.StrategyBacktester")
    def test_skips_strategy_with_existing_report(self, MockBacktester, mock_get_engine, engine):
        """A strategy that already has a backtest_report_id should NOT be re-backtested."""
        mock_get_engine.return_value = engine
        _make_strategy(
            engine, key="already_bt",
            backtest_report_id="backtest_report_already_bt",
        )

        from orchestrator import run_pipeline_evaluation
        run_pipeline_evaluation()

        # Backtester should NOT have been called
        MockBacktester.assert_not_called()

    @patch("orchestrator.get_engine")
    @patch("strategy_backtester.StrategyBacktester")
    def test_one_failure_does_not_block_others(self, MockBacktester, mock_get_engine, engine):
        """If one backtest fails with an exception, others should still run."""
        mock_get_engine.return_value = engine
        _make_strategy(engine, key="strat_a")
        _make_strategy(engine, key="strat_b")

        report_b = _passing_report("strat_b")

        mock_instance = MagicMock()
        # First call raises, second call succeeds
        mock_instance.run.side_effect = [
            RuntimeError("yfinance timeout"),
            report_b,
        ]
        MockBacktester.return_value = mock_instance

        from orchestrator import run_pipeline_evaluation
        run_pipeline_evaluation()

        # run() should have been called twice (once per strategy)
        assert mock_instance.run.call_count == 2

        # strat_b should have its report_id set
        db = get_session(engine)
        b = db.query(DynamicStrategy).filter_by(key="strat_b").first()
        assert b.backtest_report_id == "backtest_report_strat_b"
        db.close()
