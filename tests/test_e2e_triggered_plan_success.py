"""End-to-end test: successful triggered execution.

Simulates the full happy-path lifecycle of a triggered trade plan:
1. TSLA long candidate with entry_reference 250, entry zone [248, 252]
2. Plan created and transitions to WATCHING
3. Monitor tick 1: fresh price = 245 (below zone) -> stays WATCHING
4. Monitor tick 2: fresh price = 249 (in zone) -> TRIGGERED
5. Executor: fresh quote = 249.50, recalculate geometry, gates pass
6. Plan transitions to ENTERED
7. Trade created with entry_price=249.50 (fresh, not 250)
8. Stop/target proportionally adjusted from fresh price

Requirements: 5.1, 5.2, 5.3, 5.9
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from db.schema import Base
from utils.trade_plan_registry import (
    TradePlan,
    TradePlanRegistry,
    PlanState,
)
from utils.plan_monitor import PlanMonitor, _quote_rate_state, _confirmation_ticks
from utils.plan_executor import execute_triggered_plan, recalculate_geometry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_trade_plan_tables(engine):
    """Create trade_plans and trade_plan_events tables (raw DDL for in-memory tests)."""
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
    """In-memory SQLite engine with ORM tables + trade plan DDL."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    _create_trade_plan_tables(eng)
    return eng


@pytest.fixture
def registry(engine):
    return TradePlanRegistry(engine)


@pytest.fixture(autouse=True)
def reset_module_state():
    """Reset module-level state between tests."""
    _quote_rate_state.symbol_last_fetched.clear()
    _quote_rate_state.calls_this_window.clear()
    _confirmation_ticks.clear()
    yield
    _quote_rate_state.symbol_last_fetched.clear()
    _quote_rate_state.calls_this_window.clear()
    _confirmation_ticks.clear()


def _make_tsla_long_plan(
    plan_id: str = "plan-tsla-001",
    state: PlanState = PlanState.PLANNED,
) -> TradePlan:
    """Build a TSLA long trade plan with entry zone [248, 252], entry_reference=250."""
    now = datetime.now(timezone.utc)
    return TradePlan(
        plan_id=plan_id,
        candidate_id="cand-tsla-001",
        cycle_id="cycle-001",
        profile_id="aggressive",
        symbol="TSLA",
        direction="BUY",
        setup_type="momentum_fade",
        geometry_name="standard",
        entry_reference=250.0,
        entry_zone_upper=252.0,
        entry_zone_lower=248.0,
        stop_price=245.0,
        target_price=260.0,
        risk_reward=2.0,
        trigger_type="price_in_zone",
        trigger_condition_json=json.dumps({"type": "price_in_zone"}),
        trigger_confirmation_required=False,
        invalidation_logic_json=None,
        analyst_reasoning="Strong momentum setup near support",
        pm_rationale="Approved for triggered plan",
        source_signal_id="sig-tsla-001",
        signal_snapshot_json=json.dumps({"symbol": "TSLA", "setup_type": "momentum_fade"}),
        state=state,
        created_at=now,
        expires_at=now + timedelta(minutes=60),
        triggered_at=None,
        executed_at=None,
        missed_at=None,
        integrity_hash="",
    )


# ---------------------------------------------------------------------------
# E2E Test: Full successful triggered execution lifecycle
# ---------------------------------------------------------------------------


class TestE2ETriggeredPlanSuccess:
    """End-to-end: TSLA long plan from creation through triggered execution."""

    @patch("utils.plan_monitor.TRIGGERED_PLAN_MODE", "enabled")
    @patch("utils.plan_executor.TRIGGERED_PLAN_MODE", "enabled")
    def test_full_lifecycle_successful_execution(self, engine, registry):
        """Validates: Requirements 5.1, 5.2, 5.3, 5.9

        Full lifecycle:
        1. Create plan -> PLANNED
        2. Monitor tick activates -> WATCHING
        3. Tick 1: price 245 (below zone) -> stays WATCHING
        4. Tick 2: price 249 (in zone) -> TRIGGERED
        5. Executor: fresh quote 249.50 -> geometry recalculated -> gates pass -> ENTERED
        6. Trade created with fresh price, proportional stop/target
        """
        # Step 1: Create plan
        plan = _make_tsla_long_plan()
        registry.create_plan(plan)

        # Verify initial state
        fetched = registry.get_plan("plan-tsla-001")
        assert fetched.state == PlanState.PLANNED

        # Step 2 & 3: Monitor tick 1 — price at 245 (below zone [248, 252])
        # This should activate PLANNED -> WATCHING, then evaluate trigger (price outside zone)
        with patch("utils.plan_monitor._record_missed_setup_event"):
            with patch.object(
                PlanMonitor,
                "_get_rate_limited_quotes",
                return_value={"TSLA": 245.0},
            ):
                monitor = PlanMonitor(engine)
                result1 = monitor.run()

        # Plan should now be WATCHING (activated from PLANNED)
        fetched = registry.get_plan("plan-tsla-001")
        assert fetched.state == PlanState.WATCHING
        assert result1.plans_triggered == 0
        assert result1.plans_checked >= 1

        # Step 4: Monitor tick 2 — price at 249 (inside zone [248, 252])
        # Should trigger the plan
        with patch("utils.plan_monitor._record_missed_setup_event"):
            with patch.object(
                PlanMonitor,
                "_get_rate_limited_quotes",
                return_value={"TSLA": 249.0},
            ):
                monitor2 = PlanMonitor(engine)
                result2 = monitor2.run()

        # Plan should now be TRIGGERED
        fetched = registry.get_plan("plan-tsla-001")
        assert fetched.state == PlanState.TRIGGERED
        assert result2.plans_triggered == 1

        # Step 5: Execute the triggered plan with fresh quote = 249.50
        # Mock _fetch_fresh_execution_quote to return 249.50
        # Mock _run_gate_pipeline to pass
        # Mock execute_trade to succeed
        fresh_quote_ts = datetime.now(timezone.utc)

        with patch(
            "utils.plan_executor._fetch_fresh_execution_quote",
            return_value=(249.50, fresh_quote_ts, 1.0),  # price, timestamp, age_seconds
        ):
            with patch(
                "agents.portfolio_manager._run_gate_pipeline",
                return_value=(True, [], 1.0, {}),  # proceed, notes, multiplier, extras
            ):
                with patch(
                    "agents.portfolio_manager.execute_trade",
                    return_value=(True, "Trade executed successfully"),
                ) as mock_execute_trade:
                    # Re-fetch the plan in TRIGGERED state for executor
                    triggered_plan = registry.get_plan("plan-tsla-001")
                    exec_result = execute_triggered_plan(
                        engine, triggered_plan, 249.0
                    )

        # Step 6: Verify plan transitions to ENTERED
        assert exec_result.success is True
        assert exec_result.fill_price == 249.50
        assert exec_result.reason == "entered"
        assert exec_result.geometry_recalculated is True

        final_plan = registry.get_plan("plan-tsla-001")
        assert final_plan.state == PlanState.ENTERED

        # Step 7: Verify trade created with fresh price (249.50, not 250)
        mock_execute_trade.assert_called_once()
        call_args = mock_execute_trade.call_args
        decision_dict = call_args[0][1]  # second positional arg is the decision dict
        assert decision_dict["entry_price"] == 249.50
        assert decision_dict["price"] == 249.50
        assert decision_dict["symbol"] == "TSLA"
        assert decision_dict["action"] == "BUY"

        # Step 8: Verify stop/target proportionally adjusted from fresh price
        # Original: entry=250, stop=245, target=260
        # stop_ratio = (245 - 250) / 250 = -0.02
        # target_ratio = (260 - 250) / 250 = 0.04
        # new_stop = 249.50 * (1 - 0.02) = 249.50 + 249.50*(-0.02) = 244.51
        # new_target = 249.50 * (1 + 0.04) = 249.50 + 249.50*(0.04) = 259.48
        expected_stop = 249.50 * (1 + (245.0 - 250.0) / 250.0)
        expected_target = 249.50 * (1 + (260.0 - 250.0) / 250.0)

        assert abs(decision_dict["stop_price"] - expected_stop) < 0.01
        assert abs(decision_dict["target_price"] - expected_target) < 0.01

    @patch("utils.plan_monitor.TRIGGERED_PLAN_MODE", "enabled")
    def test_price_below_zone_stays_watching(self, engine, registry):
        """Monitor tick with price below zone does not trigger plan.

        Validates: Requirements 5.1, 5.3
        """
        plan = _make_tsla_long_plan()
        registry.create_plan(plan)

        # Activate to WATCHING first
        registry.activate("plan-tsla-001")

        with patch("utils.plan_monitor._record_missed_setup_event"):
            with patch.object(
                PlanMonitor,
                "_get_rate_limited_quotes",
                return_value={"TSLA": 245.0},
            ):
                monitor = PlanMonitor(engine)
                result = monitor.run()

        fetched = registry.get_plan("plan-tsla-001")
        assert fetched.state == PlanState.WATCHING
        assert result.plans_triggered == 0

    @patch("utils.plan_monitor.TRIGGERED_PLAN_MODE", "enabled")
    def test_price_in_zone_triggers_plan(self, engine, registry):
        """Monitor tick with price in zone triggers the plan.

        Validates: Requirements 5.1, 5.2
        """
        plan = _make_tsla_long_plan()
        registry.create_plan(plan)
        registry.activate("plan-tsla-001")

        with patch("utils.plan_monitor._record_missed_setup_event"):
            with patch.object(
                PlanMonitor,
                "_get_rate_limited_quotes",
                return_value={"TSLA": 249.0},
            ):
                monitor = PlanMonitor(engine)
                result = monitor.run()

        fetched = registry.get_plan("plan-tsla-001")
        assert fetched.state == PlanState.TRIGGERED
        assert result.plans_triggered == 1

    @patch("utils.plan_executor.TRIGGERED_PLAN_MODE", "enabled")
    def test_executor_uses_fresh_price_not_reference(self, engine, registry):
        """Executor uses fresh market quote (249.50) not entry_reference (250).

        Validates: Requirements 5.2, 5.3
        """
        plan = _make_tsla_long_plan(state=PlanState.PLANNED)
        registry.create_plan(plan)
        registry.activate("plan-tsla-001")
        registry.trigger("plan-tsla-001")

        triggered_plan = registry.get_plan("plan-tsla-001")
        fresh_quote_ts = datetime.now(timezone.utc)

        with patch(
            "utils.plan_executor._fetch_fresh_execution_quote",
            return_value=(249.50, fresh_quote_ts, 1.0),
        ):
            with patch(
                "agents.portfolio_manager._run_gate_pipeline",
                return_value=(True, [], 1.0, {}),
            ):
                with patch(
                    "agents.portfolio_manager.execute_trade",
                    return_value=(True, "OK"),
                ) as mock_trade:
                    result = execute_triggered_plan(engine, triggered_plan, 249.0)

        assert result.success is True
        assert result.fill_price == 249.50

        # The decision passed to execute_trade must use 249.50
        decision = mock_trade.call_args[0][1]
        assert decision["entry_price"] == 249.50
        assert decision["entry_price"] != 250.0  # NOT the reference

    @patch("utils.plan_executor.TRIGGERED_PLAN_MODE", "enabled")
    def test_stop_target_proportionally_adjusted(self, engine, registry):
        """Stop and target are proportionally adjusted from fresh price.

        Original: entry=250, stop=245, target=260
        Fresh: 249.50
        Expected stop = 249.50 + 249.50 * ((245-250)/250) = 249.50 - 4.99 = 244.51
        Expected target = 249.50 + 249.50 * ((260-250)/250) = 249.50 + 9.98 = 259.48

        Validates: Requirements 5.9
        """
        plan = _make_tsla_long_plan(state=PlanState.PLANNED)
        registry.create_plan(plan)
        registry.activate("plan-tsla-001")
        registry.trigger("plan-tsla-001")

        triggered_plan = registry.get_plan("plan-tsla-001")
        fresh_quote_ts = datetime.now(timezone.utc)

        with patch(
            "utils.plan_executor._fetch_fresh_execution_quote",
            return_value=(249.50, fresh_quote_ts, 1.0),
        ):
            with patch(
                "agents.portfolio_manager._run_gate_pipeline",
                return_value=(True, [], 1.0, {}),
            ):
                with patch(
                    "agents.portfolio_manager.execute_trade",
                    return_value=(True, "OK"),
                ) as mock_trade:
                    execute_triggered_plan(engine, triggered_plan, 249.0)

        decision = mock_trade.call_args[0][1]

        # Proportional calculation
        stop_ratio = (245.0 - 250.0) / 250.0   # -0.02
        target_ratio = (260.0 - 250.0) / 250.0  # 0.04
        expected_stop = 249.50 + 249.50 * stop_ratio    # 244.51
        expected_target = 249.50 + 249.50 * target_ratio  # 259.48

        assert abs(decision["stop_price"] - expected_stop) < 0.01
        assert abs(decision["target_price"] - expected_target) < 0.01
        assert decision["risk_reward"] > 0

    def test_recalculate_geometry_preserves_ratios(self):
        """Geometry recalculation preserves stop/target ratios from original plan.

        Validates: Requirements 5.9
        """
        plan = _make_tsla_long_plan()
        geometry = recalculate_geometry(plan, 249.50)

        # Original: entry=250, stop=245 (ratio=-0.02), target=260 (ratio=+0.04)
        assert geometry["entry_price"] == 249.50
        assert abs(geometry["stop_price"] - 244.51) < 0.01
        assert abs(geometry["target_price"] - 259.48) < 0.01
        assert geometry["risk_reward"] > 0

    @patch("utils.plan_monitor.TRIGGERED_PLAN_MODE", "enabled")
    def test_plan_event_trail_records_full_lifecycle(self, engine, registry):
        """The trade_plan_events table records the full lifecycle transitions.

        Validates: Requirements 5.1, 5.2, 5.3
        """
        plan = _make_tsla_long_plan()
        registry.create_plan(plan)
        registry.activate("plan-tsla-001")
        registry.trigger("plan-tsla-001")
        registry.mark_entered("plan-tsla-001")

        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT event_type, from_state, to_state FROM trade_plan_events "
                "WHERE plan_id = :pid ORDER BY id ASC"
            ), {"pid": "plan-tsla-001"})
            events = result.mappings().all()

        # Expect: plan_created, state_watching, state_triggered, state_entered
        event_types = [e["event_type"] for e in events]
        assert "plan_created" in event_types
        assert "state_watching" in event_types
        assert "state_triggered" in event_types
        assert "state_entered" in event_types

        # Verify state transitions are sequential
        watching_event = next(e for e in events if e["event_type"] == "state_watching")
        assert watching_event["from_state"] == "planned"
        assert watching_event["to_state"] == "watching"

        triggered_event = next(e for e in events if e["event_type"] == "state_triggered")
        assert triggered_event["from_state"] == "watching"
        assert triggered_event["to_state"] == "triggered"

        entered_event = next(e for e in events if e["event_type"] == "state_entered")
        assert entered_event["from_state"] == "triggered"
        assert entered_event["to_state"] == "entered"
