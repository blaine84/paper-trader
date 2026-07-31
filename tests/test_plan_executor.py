"""Tests for plan executor — fresh-quote execution for triggered trade plans.

Validates: Requirements 4.1, 4.2, 4.3, 4.5, 4.9, 4.13
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from utils.plan_executor import (
    PlanExecutionResult,
    execute_triggered_plan,
    recalculate_geometry,
)
from utils.trade_plan_registry import (
    PlanState,
    TradePlan,
    TradePlanRegistry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_tables(engine):
    """Create trade_plans and trade_plan_events tables in memory."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS trade_plans (
                plan_id TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL,
                cycle_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                setup_type TEXT NOT NULL,
                geometry_name TEXT,
                entry_reference REAL NOT NULL,
                entry_zone_upper REAL NOT NULL,
                entry_zone_lower REAL NOT NULL,
                stop_price REAL NOT NULL,
                target_price REAL NOT NULL,
                risk_reward REAL NOT NULL,
                trigger_type TEXT NOT NULL,
                trigger_condition_json TEXT NOT NULL,
                trigger_confirmation_required INTEGER NOT NULL DEFAULT 0,
                invalidation_logic_json TEXT,
                analyst_reasoning TEXT,
                pm_rationale TEXT,
                source_signal_id TEXT,
                signal_snapshot_json TEXT,
                state TEXT NOT NULL DEFAULT 'planned',
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                triggered_at TEXT,
                executed_at TEXT,
                missed_at TEXT,
                miss_reason TEXT,
                rejection_reason TEXT,
                integrity_hash TEXT NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS trade_plan_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id TEXT NOT NULL,
                cycle_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_data TEXT,
                fresh_price REAL,
                from_state TEXT,
                to_state TEXT,
                created_at TEXT NOT NULL
            )
        """))


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    _create_tables(eng)
    return eng


@pytest.fixture
def registry(engine):
    return TradePlanRegistry(engine)


def _make_plan(
    plan_id: str = "plan-001",
    candidate_id: str = "cand-001",
    symbol: str = "TSLA",
    direction: str = "BUY",
    setup_type: str = "momentum_fade",
    state: PlanState = PlanState.TRIGGERED,
    entry_reference: float = 250.0,
    entry_zone_upper: float = 252.0,
    entry_zone_lower: float = 248.0,
    stop_price: float = 245.0,
    target_price: float = 260.0,
    risk_reward: float = 2.0,
    expires_at: datetime | None = None,
) -> TradePlan:
    """Build a TradePlan with reasonable defaults for testing."""
    now = datetime.now(timezone.utc)
    if expires_at is None:
        expires_at = now + timedelta(minutes=60)
    return TradePlan(
        plan_id=plan_id,
        candidate_id=candidate_id,
        cycle_id="cycle-001",
        profile_id="aggressive",
        symbol=symbol,
        direction=direction,
        setup_type=setup_type,
        geometry_name="standard",
        entry_reference=entry_reference,
        entry_zone_upper=entry_zone_upper,
        entry_zone_lower=entry_zone_lower,
        stop_price=stop_price,
        target_price=target_price,
        risk_reward=risk_reward,
        trigger_type="price_in_zone",
        trigger_condition_json=json.dumps({"type": "price_in_zone"}),
        trigger_confirmation_required=False,
        invalidation_logic_json=json.dumps({"invalidation_basis": "support_break"}),
        analyst_reasoning="Strong momentum setup",
        pm_rationale="Approved for plan",
        source_signal_id="sig-001",
        signal_snapshot_json=json.dumps({"symbol": symbol}),
        state=state,
        created_at=now,
        expires_at=expires_at,
        triggered_at=now if state == PlanState.TRIGGERED else None,
        executed_at=None,
        missed_at=None,
        integrity_hash="testhash123",
    )


def _insert_plan_in_db(engine, plan: TradePlan) -> None:
    """Insert a plan directly into the database for testing executor state transitions."""
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO trade_plans (
                    plan_id, candidate_id, cycle_id, profile_id,
                    symbol, direction, setup_type, geometry_name,
                    entry_reference, entry_zone_upper, entry_zone_lower,
                    stop_price, target_price, risk_reward,
                    trigger_type, trigger_condition_json,
                    trigger_confirmation_required,
                    invalidation_logic_json,
                    analyst_reasoning, pm_rationale,
                    source_signal_id, signal_snapshot_json,
                    state, created_at, expires_at,
                    triggered_at, executed_at, missed_at,
                    miss_reason, rejection_reason,
                    integrity_hash
                ) VALUES (
                    :plan_id, :candidate_id, :cycle_id, :profile_id,
                    :symbol, :direction, :setup_type, :geometry_name,
                    :entry_reference, :entry_zone_upper, :entry_zone_lower,
                    :stop_price, :target_price, :risk_reward,
                    :trigger_type, :trigger_condition_json,
                    :trigger_confirmation_required,
                    :invalidation_logic_json,
                    :analyst_reasoning, :pm_rationale,
                    :source_signal_id, :signal_snapshot_json,
                    :state, :created_at, :expires_at,
                    :triggered_at, :executed_at, :missed_at,
                    :miss_reason, :rejection_reason,
                    :integrity_hash
                )
            """),
            {
                "plan_id": plan.plan_id,
                "candidate_id": plan.candidate_id,
                "cycle_id": plan.cycle_id,
                "profile_id": plan.profile_id,
                "symbol": plan.symbol,
                "direction": plan.direction,
                "setup_type": plan.setup_type,
                "geometry_name": plan.geometry_name,
                "entry_reference": plan.entry_reference,
                "entry_zone_upper": plan.entry_zone_upper,
                "entry_zone_lower": plan.entry_zone_lower,
                "stop_price": plan.stop_price,
                "target_price": plan.target_price,
                "risk_reward": plan.risk_reward,
                "trigger_type": plan.trigger_type,
                "trigger_condition_json": plan.trigger_condition_json,
                "trigger_confirmation_required": 1 if plan.trigger_confirmation_required else 0,
                "invalidation_logic_json": plan.invalidation_logic_json,
                "analyst_reasoning": plan.analyst_reasoning,
                "pm_rationale": plan.pm_rationale,
                "source_signal_id": plan.source_signal_id,
                "signal_snapshot_json": plan.signal_snapshot_json,
                "state": plan.state.value,
                "created_at": plan.created_at.isoformat(),
                "expires_at": plan.expires_at.isoformat(),
                "triggered_at": plan.triggered_at.isoformat() if plan.triggered_at else None,
                "executed_at": None,
                "missed_at": None,
                "miss_reason": None,
                "rejection_reason": None,
                "integrity_hash": plan.integrity_hash,
            },
        )


# ---------------------------------------------------------------------------
# Test: disabled mode returns immediately
# ---------------------------------------------------------------------------


@patch("utils.plan_executor.TRIGGERED_PLAN_MODE", "disabled")
def test_disabled_mode_returns_immediately():
    """When TRIGGERED_PLAN_MODE is disabled, execute_triggered_plan returns immediately
    with success=False and reason='feature_disabled'."""
    plan = _make_plan()
    # engine doesn't matter — should return before touching DB
    result = execute_triggered_plan(None, plan, 250.0)

    assert isinstance(result, PlanExecutionResult)
    assert result.success is False
    assert result.reason == "feature_disabled"
    assert result.fill_price is None
    assert result.geometry_recalculated is False


# ---------------------------------------------------------------------------
# Test: stale quote → PlanExecutionResult(success=False, reason="quote_too_stale")
# ---------------------------------------------------------------------------


@patch("utils.plan_executor.TRIGGERED_PLAN_MODE", "enabled")
@patch("utils.plan_executor._fetch_fresh_execution_quote")
def test_stale_quote_marks_missed(mock_fetch, engine):
    """When the fresh quote is older than PLAN_EXECUTION_MAX_QUOTE_AGE_SECONDS,
    the plan is marked missed with reason='quote_too_stale'."""
    plan = _make_plan(state=PlanState.TRIGGERED)
    _insert_plan_in_db(engine, plan)

    # Return a price but with age exceeding the max (default 5s)
    stale_timestamp = datetime.now(timezone.utc) - timedelta(seconds=10)
    mock_fetch.return_value = (251.0, stale_timestamp, 10.0)

    result = execute_triggered_plan(engine, plan, 251.0)

    assert result.success is False
    assert result.reason == "quote_too_stale"
    assert result.fill_price == 251.0
    assert result.geometry_recalculated is False

    # Verify plan state transitioned to MISSED in database
    reg = TradePlanRegistry(engine)
    fetched = reg.get_plan("plan-001")
    assert fetched.state == PlanState.MISSED


# ---------------------------------------------------------------------------
# Test: geometry recalculation preserves R:R ratio
# ---------------------------------------------------------------------------


def test_recalculate_geometry_preserves_rr_ratio_long():
    """For a LONG plan, recalculate_geometry preserves stop/target ratios
    and R:R is approximately the same as original."""
    plan = _make_plan(
        direction="BUY",
        entry_reference=250.0,
        stop_price=245.0,   # 5 below entry
        target_price=260.0, # 10 above entry → R:R = 2.0
    )

    # Fresh price slightly different from entry_reference
    geometry = recalculate_geometry(plan, 251.0)

    # stop_ratio = (245 - 250) / 250 = -0.02
    # target_ratio = (260 - 250) / 250 = 0.04
    # new_stop = 251 + 251 * (-0.02) = 251 - 5.02 = 245.98
    # new_target = 251 + 251 * 0.04 = 251 + 10.04 = 261.04
    assert abs(geometry["entry_price"] - 251.0) < 0.01
    assert abs(geometry["stop_price"] - 245.98) < 0.01
    assert abs(geometry["target_price"] - 261.04) < 0.01

    # R:R should stay at 2.0
    assert abs(geometry["risk_reward"] - 2.0) < 0.01


def test_recalculate_geometry_preserves_rr_ratio_short():
    """For a SHORT plan, recalculate_geometry preserves stop/target ratios
    and R:R is approximately the same as original."""
    plan = _make_plan(
        direction="SHORT",
        entry_reference=100.0,
        stop_price=105.0,   # 5 above entry (stop for short)
        target_price=90.0,  # 10 below entry → R:R = 2.0
    )

    # Fresh price slightly different
    geometry = recalculate_geometry(plan, 101.0)

    # stop_ratio = (105 - 100) / 100 = 0.05
    # target_ratio = (90 - 100) / 100 = -0.10
    # new_stop = 101 + 101 * 0.05 = 101 + 5.05 = 106.05
    # new_target = 101 + 101 * (-0.10) = 101 - 10.10 = 90.90
    assert abs(geometry["entry_price"] - 101.0) < 0.01
    assert abs(geometry["stop_price"] - 106.05) < 0.01
    assert abs(geometry["target_price"] - 90.90) < 0.01

    # R:R: risk = new_stop - fresh = 106.05 - 101 = 5.05
    #       reward = fresh - new_target = 101 - 90.9 = 10.10
    #       R:R = 10.10 / 5.05 = 2.0
    assert abs(geometry["risk_reward"] - 2.0) < 0.01


# ---------------------------------------------------------------------------
# Test: gate pipeline rejection → plan marked rejected/missed
# ---------------------------------------------------------------------------


@patch("utils.plan_executor.TRIGGERED_PLAN_MODE", "enabled")
@patch("utils.plan_executor._fetch_fresh_execution_quote")
@patch("agents.portfolio_manager._run_gate_pipeline")
@patch("agents.portfolio_manager.execute_trade")
def test_gate_pipeline_rejection_marks_plan_rejected(
    mock_execute_trade, mock_gate_pipeline, mock_fetch, engine
):
    """When gate pipeline returns proceed=False, the plan is marked rejected."""
    plan = _make_plan(state=PlanState.TRIGGERED)
    _insert_plan_in_db(engine, plan)

    # Fresh quote: price in zone, age OK
    mock_fetch.return_value = (250.5, datetime.now(timezone.utc), 1.0)
    # Gate pipeline rejects
    mock_gate_pipeline.return_value = (
        False,
        [{"gate": "risk_geometry", "decision": "reject", "reason": "R:R too low"}],
        1.0,
        {},
    )

    result = execute_triggered_plan(engine, plan, 250.5)

    assert result.success is False
    assert "risk_geometry" in result.reason or "gate_pipeline" in result.reason
    assert result.geometry_recalculated is True

    # Verify plan state transitioned to REJECTED in database
    reg = TradePlanRegistry(engine)
    fetched = reg.get_plan("plan-001")
    assert fetched.state == PlanState.REJECTED

    # execute_trade should NOT have been called
    mock_execute_trade.assert_not_called()


# ---------------------------------------------------------------------------
# Test: successful execution → mark_entered called
# ---------------------------------------------------------------------------


@patch("utils.plan_executor.TRIGGERED_PLAN_MODE", "enabled")
@patch("utils.plan_executor._fetch_fresh_execution_quote")
@patch("agents.portfolio_manager._run_gate_pipeline")
@patch("agents.portfolio_manager.execute_trade")
def test_successful_execution_marks_entered(
    mock_execute_trade, mock_gate_pipeline, mock_fetch, engine
):
    """When gates pass and execute_trade succeeds, plan transitions to ENTERED."""
    plan = _make_plan(state=PlanState.TRIGGERED)
    _insert_plan_in_db(engine, plan)

    # Fresh quote: price in zone, age OK
    mock_fetch.return_value = (250.5, datetime.now(timezone.utc), 1.0)
    # Gate pipeline passes
    mock_gate_pipeline.return_value = (True, [], 1.0, {})
    # execute_trade succeeds
    mock_execute_trade.return_value = (True, "Trade executed successfully")

    result = execute_triggered_plan(engine, plan, 250.5)

    assert result.success is True
    assert result.fill_price == 250.5
    assert result.reason == "entered"
    assert result.geometry_recalculated is True

    # Verify plan state transitioned to ENTERED in database
    reg = TradePlanRegistry(engine)
    fetched = reg.get_plan("plan-001")
    assert fetched.state == PlanState.ENTERED

    # execute_trade should have been called with fresh geometry
    mock_execute_trade.assert_called_once()
    call_args = mock_execute_trade.call_args
    decision = call_args[0][1]  # second positional arg is decision dict
    assert decision["entry_price"] == 250.5
    assert decision["symbol"] == "TSLA"


# ---------------------------------------------------------------------------
# Test: price moved beyond zone between trigger and execution
# ---------------------------------------------------------------------------


@patch("utils.plan_executor.TRIGGERED_PLAN_MODE", "enabled")
@patch("utils.plan_executor._fetch_fresh_execution_quote")
def test_price_beyond_zone_marks_missed(mock_fetch, engine):
    """When the fresh price has moved beyond the entry zone (even with tolerance),
    the plan is marked missed with reason='price_beyond_zone'."""
    plan = _make_plan(
        state=PlanState.TRIGGERED,
        entry_reference=250.0,
        entry_zone_upper=252.0,
        entry_zone_lower=248.0,
    )
    _insert_plan_in_db(engine, plan)

    # Fresh quote: price is way above the upper zone bound
    # Zone upper = 252, tolerance 0.5% of 250 = 1.25, effective upper = 253.25
    # Price 260 is well beyond
    mock_fetch.return_value = (260.0, datetime.now(timezone.utc), 1.0)

    result = execute_triggered_plan(engine, plan, 260.0)

    assert result.success is False
    assert result.reason == "price_beyond_zone"
    assert result.fill_price == 260.0

    # Verify plan state transitioned to MISSED in database
    reg = TradePlanRegistry(engine)
    fetched = reg.get_plan("plan-001")
    assert fetched.state == PlanState.MISSED


@patch("utils.plan_executor.TRIGGERED_PLAN_MODE", "enabled")
@patch("utils.plan_executor._fetch_fresh_execution_quote")
def test_price_past_target_marks_missed(mock_fetch, engine):
    """When the fresh price has already crossed the target, the plan is marked missed."""
    plan = _make_plan(
        state=PlanState.TRIGGERED,
        direction="BUY",
        entry_reference=250.0,
        entry_zone_upper=252.0,
        entry_zone_lower=248.0,
        target_price=260.0,
    )
    _insert_plan_in_db(engine, plan)

    # Fresh price is at/past the target for a BUY (price >= target)
    mock_fetch.return_value = (261.0, datetime.now(timezone.utc), 1.0)

    result = execute_triggered_plan(engine, plan, 261.0)

    assert result.success is False
    # Could be "price_beyond_zone" or "price_past_target" depending on order of checks
    # In the implementation, zone check happens first, so 261 > effective upper 253.25
    # means it's "price_beyond_zone"
    assert result.reason in ("price_beyond_zone", "price_past_target")

    reg = TradePlanRegistry(engine)
    fetched = reg.get_plan("plan-001")
    assert fetched.state == PlanState.MISSED


@patch("utils.plan_executor.TRIGGERED_PLAN_MODE", "enabled")
@patch("utils.plan_executor._fetch_fresh_execution_quote")
def test_price_past_target_short_marks_missed(mock_fetch, engine):
    """For a SHORT plan, when fresh price has dropped below target, mark missed."""
    plan = _make_plan(
        state=PlanState.TRIGGERED,
        direction="SHORT",
        entry_reference=100.0,
        entry_zone_upper=101.0,  # Short: upper slightly above ref
        entry_zone_lower=99.0,
        stop_price=105.0,
        target_price=90.0,
    )
    _insert_plan_in_db(engine, plan)

    # Fresh price is below the lower bound with tolerance AND past target
    # Zone lower = 99, tolerance 0.5% of 100 = 0.5, effective lower = 98.5
    # Price 88 is way below effective lower → "price_beyond_zone"
    mock_fetch.return_value = (88.0, datetime.now(timezone.utc), 1.0)

    result = execute_triggered_plan(engine, plan, 88.0)

    assert result.success is False
    assert result.reason in ("price_beyond_zone", "price_past_target")

    reg = TradePlanRegistry(engine)
    fetched = reg.get_plan("plan-001")
    assert fetched.state == PlanState.MISSED


# ---------------------------------------------------------------------------
# Test: no fresh quote available → plan stays TRIGGERED for retry
# ---------------------------------------------------------------------------


@patch("utils.plan_executor.TRIGGERED_PLAN_MODE", "enabled")
@patch("utils.plan_executor._fetch_fresh_execution_quote")
def test_no_fresh_quote_stays_triggered(mock_fetch, engine):
    """When _fetch_fresh_execution_quote returns None (provider exhausted),
    the plan stays TRIGGERED for retry next tick — NOT marked MISSED."""
    plan = _make_plan(state=PlanState.TRIGGERED)
    _insert_plan_in_db(engine, plan)

    # Provider returns nothing (exhausted/circuit-broken)
    mock_fetch.return_value = (None, None, float("inf"))

    result = execute_triggered_plan(engine, plan, 250.0)

    assert result.success is False
    assert "retry" in result.reason or "no_fresh_quote" in result.reason
    assert result.fill_price is None

    # Plan should still be TRIGGERED — not moved to MISSED
    reg = TradePlanRegistry(engine)
    fetched = reg.get_plan("plan-001")
    assert fetched.state == PlanState.TRIGGERED


# ---------------------------------------------------------------------------
# Test: execute_trade failure → plan MISSED
# ---------------------------------------------------------------------------


@patch("utils.plan_executor.TRIGGERED_PLAN_MODE", "enabled")
@patch("utils.plan_executor._fetch_fresh_execution_quote")
@patch("agents.portfolio_manager._run_gate_pipeline")
@patch("agents.portfolio_manager.execute_trade")
def test_execute_trade_failure_marks_missed(
    mock_execute_trade, mock_gate_pipeline, mock_fetch, engine
):
    """When gates pass but execute_trade returns failure, plan is marked missed."""
    plan = _make_plan(state=PlanState.TRIGGERED)
    _insert_plan_in_db(engine, plan)

    # Fresh quote in zone, age OK
    mock_fetch.return_value = (250.5, datetime.now(timezone.utc), 1.0)
    # Gate pipeline passes
    mock_gate_pipeline.return_value = (True, [], 1.0, {})
    # execute_trade fails
    mock_execute_trade.return_value = (False, "Insufficient buying power")

    result = execute_triggered_plan(engine, plan, 250.5)

    assert result.success is False
    assert result.reason == "execution_failed"
    assert result.fill_price == 250.5

    # Plan transitions to MISSED
    reg = TradePlanRegistry(engine)
    fetched = reg.get_plan("plan-001")
    assert fetched.state == PlanState.MISSED
