"""
Unit tests for deployment_pipeline.py
Tests gate evaluation, win rate computation, apply/escalate, and run_pipeline_evaluation.
"""

import json
import pytest
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from db.schema import Base, DynamicStrategy, AgentMemory, get_session
from models.case import Case
from deployment_pipeline import (
    PIPELINE_STAGES,
    BACKTEST_MIN_TRADES,
    WIN_RATE_THRESHOLD,
    TIME_GATE_DAYS,
    TIME_GATE_MIN_TRADES,
    GateResult,
    evaluate_backtest_gate,
    evaluate_time_gated_stage,
    compute_stage_win_rate,
    apply_gate_result,
    escalate_failure,
    run_pipeline_evaluation,
)


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


# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

class TestConstants:
    def test_pipeline_stages(self):
        assert PIPELINE_STAGES == ["backtest", "paper_trade", "live_50", "live_100"]

    def test_thresholds(self):
        assert BACKTEST_MIN_TRADES == 50
        assert WIN_RATE_THRESHOLD == 0.55
        assert TIME_GATE_DAYS == 7


# -----------------------------------------------------------------------
# evaluate_backtest_gate
# -----------------------------------------------------------------------

class TestEvaluateBacktestGate:
    def test_advance_when_passing(self):
        report = {"summary": {"total_trades": 60, "win_rate": 0.60}}
        result = evaluate_backtest_gate(report)
        assert result.decision == "advance"
        assert result.next_stage == "paper_trade"

    def test_fail_insufficient_trades(self):
        report = {"summary": {"total_trades": 30, "win_rate": 0.70}}
        result = evaluate_backtest_gate(report)
        assert result.decision == "fail"
        assert "insufficient trades" in result.reason

    def test_fail_low_win_rate(self):
        report = {"summary": {"total_trades": 100, "win_rate": 0.50}}
        result = evaluate_backtest_gate(report)
        assert result.decision == "fail"
        assert "win rate below threshold" in result.reason

    def test_fail_exact_threshold_win_rate(self):
        """Win rate exactly at 0.55 should fail (must be > 0.55)."""
        report = {"summary": {"total_trades": 60, "win_rate": 0.55}}
        result = evaluate_backtest_gate(report)
        assert result.decision == "fail"

    def test_advance_just_above_threshold(self):
        report = {"summary": {"total_trades": 50, "win_rate": 0.5501}}
        result = evaluate_backtest_gate(report)
        assert result.decision == "advance"

    def test_metrics_populated(self):
        report = {"summary": {"total_trades": 75, "win_rate": 0.62}}
        result = evaluate_backtest_gate(report)
        assert result.metrics["total_trades"] == 75
        assert result.metrics["win_rate"] == 0.62

    def test_empty_summary(self):
        report = {"summary": {}}
        result = evaluate_backtest_gate(report)
        assert result.decision == "fail"


# -----------------------------------------------------------------------
# evaluate_time_gated_stage
# -----------------------------------------------------------------------

class TestEvaluateTimeGatedStage:
    def test_wait_when_time_not_elapsed(self, engine):
        strat = _make_strategy(
            engine,
            status="paper_trade",
            pipeline_stage="paper_trade",
            paper_trade_start_date=datetime.utcnow() - timedelta(days=3),
        )
        result = evaluate_time_gated_stage(
            strat, "paper_trade", datetime.utcnow(), win_rate=0.70, total_trades=10
        )
        assert result.decision == "wait"

    def test_advance_paper_trade_to_live_50(self, engine):
        strat = _make_strategy(
            engine,
            status="paper_trade",
            pipeline_stage="paper_trade",
            paper_trade_start_date=datetime.utcnow() - timedelta(days=8),
        )
        result = evaluate_time_gated_stage(
            strat, "paper_trade", datetime.utcnow(), win_rate=0.60, total_trades=10
        )
        assert result.decision == "advance"
        assert result.next_stage == "live_50"

    def test_advance_live_50_to_live_100(self, engine):
        strat = _make_strategy(
            engine,
            key="strat_l50",
            status="live_50",
            pipeline_stage="live_50",
            live_50_start_date=datetime.utcnow() - timedelta(days=10),
        )
        result = evaluate_time_gated_stage(
            strat, "live_50", datetime.utcnow(), win_rate=0.60, total_trades=10
        )
        assert result.decision == "advance"
        assert result.next_stage == "live_100"

    def test_fail_when_time_elapsed_and_low_win_rate(self, engine):
        strat = _make_strategy(
            engine,
            status="paper_trade",
            pipeline_stage="paper_trade",
            paper_trade_start_date=datetime.utcnow() - timedelta(days=8),
        )
        result = evaluate_time_gated_stage(
            strat, "paper_trade", datetime.utcnow(), win_rate=0.40, total_trades=10
        )
        assert result.decision == "fail"

    def test_wait_when_not_enough_trades(self, engine):
        """Even if 7 days passed, wait if fewer than 5 trades."""
        strat = _make_strategy(
            engine,
            status="paper_trade",
            pipeline_stage="paper_trade",
            paper_trade_start_date=datetime.utcnow() - timedelta(days=10),
        )
        result = evaluate_time_gated_stage(
            strat, "paper_trade", datetime.utcnow(), win_rate=0.70, total_trades=3
        )
        assert result.decision == "wait"
        assert "not enough trades" in result.reason

    def test_wait_when_no_start_date(self, engine):
        strat = _make_strategy(
            engine,
            status="paper_trade",
            pipeline_stage="paper_trade",
        )
        result = evaluate_time_gated_stage(
            strat, "paper_trade", datetime.utcnow(), win_rate=0.70, total_trades=10
        )
        assert result.decision == "wait"


# -----------------------------------------------------------------------
# compute_stage_win_rate
# -----------------------------------------------------------------------

class TestComputeStageWinRate:
    def test_no_cases(self, engine):
        win_rate, total = compute_stage_win_rate(
            engine, "test_strat", datetime.utcnow() - timedelta(days=30)
        )
        assert win_rate == 0.0
        assert total == 0

    def test_with_cases(self, engine):
        db = get_session(engine)
        # 3 wins, 2 losses = 0.6 win rate
        for i, outcome in enumerate(["success", "success", "success", "failure", "failure"]):
            db.add(Case(
                symbol="SPY",
                date="2025-01-10",
                setup_type="test_strat",
                outcome=outcome,
                pnl_pct=1.0 if outcome == "success" else -1.0,
                lesson="test",
            ))
        db.commit()
        db.close()

        win_rate, total = compute_stage_win_rate(
            engine, "test_strat", datetime(2025, 1, 1)
        )
        assert total == 5
        assert abs(win_rate - 0.6) < 0.001

    def test_filters_by_date(self, engine):
        db = get_session(engine)
        # Old case (before since_date)
        db.add(Case(
            symbol="SPY", date="2024-12-01", setup_type="test_strat",
            outcome="success", pnl_pct=1.0, lesson="old",
        ))
        # New case (after since_date)
        db.add(Case(
            symbol="SPY", date="2025-01-15", setup_type="test_strat",
            outcome="failure", pnl_pct=-1.0, lesson="new",
        ))
        db.commit()
        db.close()

        win_rate, total = compute_stage_win_rate(
            engine, "test_strat", datetime(2025, 1, 1)
        )
        assert total == 1
        assert win_rate == 0.0  # only the failure case is in range

    def test_filters_by_strategy_key(self, engine):
        db = get_session(engine)
        db.add(Case(
            symbol="SPY", date="2025-01-10", setup_type="other_strat",
            outcome="success", pnl_pct=1.0, lesson="other",
        ))
        db.add(Case(
            symbol="SPY", date="2025-01-10", setup_type="test_strat",
            outcome="failure", pnl_pct=-1.0, lesson="mine",
        ))
        db.commit()
        db.close()

        win_rate, total = compute_stage_win_rate(
            engine, "test_strat", datetime(2025, 1, 1)
        )
        assert total == 1
        assert win_rate == 0.0


# -----------------------------------------------------------------------
# apply_gate_result
# -----------------------------------------------------------------------

class TestApplyGateResult:
    def test_advance_sets_status_and_stage(self, engine):
        strat = _make_strategy(engine, status="backtest", pipeline_stage="backtest")
        result = GateResult(
            decision="advance", next_stage="paper_trade",
            reason="passed", metrics={"total_trades": 60, "win_rate": 0.60},
        )
        apply_gate_result(engine, strat, result)

        db = get_session(engine)
        updated = db.query(DynamicStrategy).filter_by(id=strat.id).first()
        assert updated.status == "paper_trade"
        assert updated.pipeline_stage == "paper_trade"
        assert updated.paper_trade_start_date is not None
        db.close()

    def test_fail_sets_backtest_failed(self, engine):
        strat = _make_strategy(
            engine, status="paper_trade", pipeline_stage="paper_trade",
            paper_trade_start_date=datetime.utcnow() - timedelta(days=10),
        )
        result = GateResult(
            decision="fail", next_stage=None,
            reason="win rate too low", metrics={"win_rate": 0.40},
        )
        apply_gate_result(engine, strat, result)

        db = get_session(engine)
        updated = db.query(DynamicStrategy).filter_by(id=strat.id).first()
        assert updated.status == "backtest_failed"
        assert updated.failure_stage == "paper_trade"
        assert updated.failure_reason == "win rate too low"
        db.close()

    def test_fail_creates_escalation(self, engine):
        strat = _make_strategy(
            engine, key="esc_strat", status="live_50", pipeline_stage="live_50",
            live_50_start_date=datetime.utcnow() - timedelta(days=10),
        )
        result = GateResult(
            decision="fail", next_stage=None,
            reason="live 50% failed", metrics={"win_rate": 0.30},
        )
        apply_gate_result(engine, strat, result)

        db = get_session(engine)
        mem = db.query(AgentMemory).filter_by(
            agent="quant_researcher", key="pipeline_failure_esc_strat"
        ).first()
        assert mem is not None
        data = json.loads(mem.value)
        assert data["strategy_key"] == "esc_strat"
        assert data["failed_stage"] == "live_50"
        db.close()

    def test_wait_does_nothing(self, engine):
        strat = _make_strategy(engine, status="paper_trade", pipeline_stage="paper_trade")
        result = GateResult(
            decision="wait", next_stage=None,
            reason="waiting", metrics={},
        )
        apply_gate_result(engine, strat, result)

        db = get_session(engine)
        updated = db.query(DynamicStrategy).filter_by(id=strat.id).first()
        assert updated.status == "paper_trade"  # unchanged
        db.close()

    def test_advance_to_live_50_sets_start_date(self, engine):
        strat = _make_strategy(
            engine, key="adv_l50", status="paper_trade", pipeline_stage="paper_trade",
            paper_trade_start_date=datetime.utcnow() - timedelta(days=10),
        )
        result = GateResult(
            decision="advance", next_stage="live_50",
            reason="passed", metrics={},
        )
        apply_gate_result(engine, strat, result)

        db = get_session(engine)
        updated = db.query(DynamicStrategy).filter_by(id=strat.id).first()
        assert updated.status == "live_50"
        assert updated.live_50_start_date is not None
        db.close()

    def test_advance_to_live_100_sets_start_date(self, engine):
        strat = _make_strategy(
            engine, key="adv_l100", status="live_50", pipeline_stage="live_50",
            live_50_start_date=datetime.utcnow() - timedelta(days=10),
        )
        result = GateResult(
            decision="advance", next_stage="live_100",
            reason="passed", metrics={},
        )
        apply_gate_result(engine, strat, result)

        db = get_session(engine)
        updated = db.query(DynamicStrategy).filter_by(id=strat.id).first()
        assert updated.status == "live_100"
        assert updated.live_100_start_date is not None
        db.close()


# -----------------------------------------------------------------------
# escalate_failure
# -----------------------------------------------------------------------

class TestEscalateFailure:
    def test_creates_agent_memory(self, engine):
        strat = _make_strategy(engine, key="fail_strat")
        escalate_failure(
            engine, strat, "backtest", "too few trades",
            {"total_trades": 10, "win_rate": 0.30},
        )

        db = get_session(engine)
        mem = db.query(AgentMemory).filter_by(
            agent="quant_researcher", key="pipeline_failure_fail_strat"
        ).first()
        assert mem is not None
        data = json.loads(mem.value)
        assert data["strategy_key"] == "fail_strat"
        assert data["failed_stage"] == "backtest"
        assert data["failure_reason"] == "too few trades"
        assert data["performance_snapshot"]["total_trades"] == 10
        db.close()

    def test_upserts_existing_record(self, engine):
        strat = _make_strategy(engine, key="upsert_strat")
        escalate_failure(engine, strat, "backtest", "first failure", {})
        escalate_failure(engine, strat, "paper_trade", "second failure", {})

        db = get_session(engine)
        records = db.query(AgentMemory).filter_by(
            agent="quant_researcher", key="pipeline_failure_upsert_strat"
        ).all()
        assert len(records) == 1
        data = json.loads(records[0].value)
        assert data["failed_stage"] == "paper_trade"
        db.close()


# -----------------------------------------------------------------------
# run_pipeline_evaluation
# -----------------------------------------------------------------------

class TestRunPipelineEvaluation:
    def test_evaluates_backtest_with_report(self, engine):
        """Strategy in backtest with a passing report should advance."""
        strat = _make_strategy(
            engine, key="bt_pass",
            status="backtest", pipeline_stage="backtest",
            backtest_report_id="backtest_report_bt_pass",
        )
        # Store a passing backtest report
        db = get_session(engine)
        db.add(AgentMemory(
            agent="strategy_backtester",
            key="backtest_report_bt_pass",
            value=json.dumps({
                "summary": {"total_trades": 60, "win_rate": 0.62},
            }),
        ))
        db.commit()
        db.close()

        results = run_pipeline_evaluation(engine)
        assert len(results) == 1
        assert results[0]["decision"] == "advance"

        # Verify strategy was advanced
        db = get_session(engine)
        updated = db.query(DynamicStrategy).filter_by(key="bt_pass").first()
        assert updated.status == "paper_trade"
        db.close()

    def test_skips_backtest_without_report(self, engine):
        """Strategy in backtest without a report should be skipped."""
        _make_strategy(
            engine, key="bt_no_report",
            status="backtest", pipeline_stage="backtest",
        )
        results = run_pipeline_evaluation(engine)
        assert len(results) == 0

    def test_skips_live_100(self, engine):
        """Strategies in live_100 should not be evaluated."""
        _make_strategy(
            engine, key="l100",
            status="live_100", pipeline_stage="live_100",
            live_100_start_date=datetime.utcnow() - timedelta(days=30),
        )
        results = run_pipeline_evaluation(engine)
        assert len(results) == 0

    def test_evaluates_paper_trade(self, engine):
        """Paper trade strategy with enough time and good win rate should advance."""
        start = datetime.utcnow() - timedelta(days=10)
        strat = _make_strategy(
            engine, key="pt_pass",
            status="paper_trade", pipeline_stage="paper_trade",
            paper_trade_start_date=start,
        )
        # Add cases with good win rate
        db = get_session(engine)
        for i in range(8):
            db.add(Case(
                symbol="SPY",
                date=start.strftime("%Y-%m-%d"),
                setup_type="pt_pass",
                outcome="success" if i < 6 else "failure",
                pnl_pct=1.0 if i < 6 else -1.0,
                lesson="test",
            ))
        db.commit()
        db.close()

        results = run_pipeline_evaluation(engine)
        assert len(results) == 1
        assert results[0]["decision"] == "advance"

    def test_does_not_evaluate_non_pipeline_strategies(self, engine):
        """Strategies with status 'active' or 'retired' should not be evaluated."""
        _make_strategy(engine, key="active_strat", status="active", pipeline_stage=None)
        _make_strategy(engine, key="retired_strat", status="retired", pipeline_stage=None)
        results = run_pipeline_evaluation(engine)
        assert len(results) == 0
