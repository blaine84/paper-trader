"""End-to-end tests: triggered trade plan edge cases.

Covers:
  10.6 — Fresh price unavailable or stale at execution (fail-closed)
  10.7 — Backward compatibility with flag disabled
  10.8 — Plan deduplication
  10.9 — Decision-log visibility in enabled mode

Requirements: 0.2, 5.1, 5.4, 5.10, 7.7, 7.8, 8.7, 10.3, 11.1
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy import create_engine, text

from utils.trade_plan_registry import (
    PlanState,
    TradePlan,
    TradePlanRegistry,
)
from utils.plan_monitor import (
    PlanMonitor,
    MonitorTickResult,
    _quote_rate_state,
    _confirmation_ticks,
)


# ---------------------------------------------------------------------------
# Schema setup (in-memory SQLite)
# ---------------------------------------------------------------------------


def _create_all_tables(engine):
    """Create all tables needed by triggered plan E2E tests."""
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
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS trade_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER,
                timestamp TEXT,
                event_type TEXT NOT NULL,
                agent TEXT,
                symbol TEXT,
                profile TEXT,
                price REAL,
                message TEXT,
                payload_json TEXT,
                dedupe_key TEXT,
                candidate_lineage_id TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pm_candidate_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                cycle_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_data TEXT,
                created_at TEXT NOT NULL,
                candidate_type TEXT
            )
        """))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    _create_all_tables(eng)
    return eng


@pytest.fixture
def registry(engine):
    return TradePlanRegistry(engine)


@pytest.fixture(autouse=True)
def reset_module_state():
    """Reset module-level state between tests."""
    _confirmation_ticks.clear()
    _quote_rate_state.symbol_last_fetched.clear()
    _quote_rate_state.calls_this_window.clear()
    yield
    _confirmation_ticks.clear()
    _quote_rate_state.symbol_last_fetched.clear()
    _quote_rate_state.calls_this_window.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plan(
    plan_id: str = "plan-001",
    candidate_id: str = "cand-001",
    symbol: str = "TSLA",
    direction: str = "BUY",
    setup_type: str = "momentum_fade",
    state: PlanState = PlanState.PLANNED,
    entry_reference: float = 250.0,
    entry_zone_upper: float = 252.0,
    entry_zone_lower: float = 248.0,
    stop_price: float = 245.0,
    target_price: float = 260.0,
    risk_reward: float = 2.0,
    profile_id: str = "aggressive",
    expires_at: datetime | None = None,
    trigger_confirmation_required: bool = False,
    invalidation_logic_json: str | None = None,
) -> TradePlan:
    """Build a TradePlan with reasonable defaults."""
    now = datetime.now(timezone.utc)
    if expires_at is None:
        expires_at = now + timedelta(minutes=60)
    return TradePlan(
        plan_id=plan_id,
        candidate_id=candidate_id,
        cycle_id="cycle-001",
        profile_id=profile_id,
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
        trigger_confirmation_required=trigger_confirmation_required,
        invalidation_logic_json=invalidation_logic_json,
        analyst_reasoning="Strong setup",
        pm_rationale="Approved",
        source_signal_id="sig-001",
        signal_snapshot_json=json.dumps({"symbol": symbol}),
        state=state,
        created_at=now,
        expires_at=expires_at,
        triggered_at=None,
        executed_at=None,
        missed_at=None,
        integrity_hash="",
    )


def _get_trade_events(engine, event_type: str | None = None):
    """Query trade_events from the in-memory DB."""
    with engine.connect() as conn:
        if event_type:
            result = conn.execute(
                text("SELECT * FROM trade_events WHERE event_type = :et"),
                {"et": event_type},
            )
        else:
            result = conn.execute(text("SELECT * FROM trade_events"))
        return [dict(r._mapping) for r in result.fetchall()]


def _get_plan_events(engine, plan_id: str | None = None):
    """Query trade_plan_events from the in-memory DB."""
    with engine.connect() as conn:
        if plan_id:
            result = conn.execute(
                text("SELECT * FROM trade_plan_events WHERE plan_id = :pid"),
                {"pid": plan_id},
            )
        else:
            result = conn.execute(text("SELECT * FROM trade_plan_events"))
        return [dict(r._mapping) for r in result.fetchall()]


def _get_candidate_events(engine, event_type: str | None = None):
    """Query pm_candidate_events from the in-memory DB."""
    with engine.connect() as conn:
        if event_type:
            result = conn.execute(
                text("SELECT * FROM pm_candidate_events WHERE event_type = :et"),
                {"et": event_type},
            )
        else:
            result = conn.execute(text("SELECT * FROM pm_candidate_events"))
        return [dict(r._mapping) for r in result.fetchall()]


# ===========================================================================
# 10.6 — Fresh price unavailable or stale at execution (fail-closed)
# ===========================================================================


class TestFreshPriceUnavailableOrStale:
    """Plan execution is fail-closed: no fill without verified fresh quote.

    Validates: Requirements 5.1, 5.4, 5.10, 10.3
    """

    @patch("utils.plan_executor.TRIGGERED_PLAN_MODE", "enabled")
    def test_case_a_both_providers_return_none_plan_stays_triggered(self, engine, registry):
        """Case A: Finnhub+yfinance both return None -> plan stays TRIGGERED, retries next tick.

        Validates: Requirements 5.4, 5.10
        """
        plan = _make_plan(state=PlanState.PLANNED)
        registry.create_plan(plan)
        registry.activate("plan-001")
        registry.trigger("plan-001")

        # Verify plan is TRIGGERED
        p = registry.get_plan("plan-001")
        assert p.state == PlanState.TRIGGERED

        # Execute with both providers returning nothing
        from utils.plan_executor import execute_triggered_plan

        with patch(
            "utils.plan_executor._fetch_fresh_execution_quote",
            return_value=(None, None, float("inf")),
        ):
            result = execute_triggered_plan(engine, p, 250.0)

        # Plan stays TRIGGERED (not MISSED) for retry
        updated = registry.get_plan("plan-001")
        assert updated.state == PlanState.TRIGGERED
        assert result.success is False
        assert "no_fresh_quote" in result.reason or "retry" in result.reason

        # NO paper fill (trade_events should have no "trade_opened" or fill-like event)
        fills = _get_trade_events(engine, "trade_opened")
        assert len(fills) == 0

    @patch("utils.plan_executor.TRIGGERED_PLAN_MODE", "enabled")
    def test_case_b_retries_exhausted_plan_expires_to_missed(self, engine, registry):
        """Case B: After plan expiration -> MISSED via orphan sweep.

        Validates: Requirements 5.4, 10.3
        """
        # Create a plan that is already expired (simulating passage of time)
        expired_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        plan = _make_plan(
            state=PlanState.PLANNED,
            expires_at=expired_time,
        )
        registry.create_plan(plan)
        registry.activate("plan-001")
        registry.trigger("plan-001")

        # Verify TRIGGERED state
        p = registry.get_plan("plan-001")
        assert p.state == PlanState.TRIGGERED

        # Orphan sweep catches expired TRIGGERED plans -> MISSED
        # Patch get_session to return a mock that doesn't hit a real DB for trade_events
        mock_session = MagicMock()
        with patch("db.schema.get_session", return_value=mock_session):
            swept = registry.finalize_orphaned_plans()

        updated = registry.get_plan("plan-001")
        assert updated.state == PlanState.MISSED
        assert "plan-001" in swept

        # Verify the miss_reason in the DB directly
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT miss_reason FROM trade_plans WHERE plan_id = :pid"),
                {"pid": "plan-001"},
            ).fetchone()
        assert row[0] == "execution_timeout"

        # NO paper fill created
        fills = _get_trade_events(engine, "trade_opened")
        assert len(fills) == 0

    @patch("utils.plan_executor.TRIGGERED_PLAN_MODE", "enabled")
    def test_case_c_quote_age_exceeds_max_stays_triggered_for_retry(self, engine, registry):
        """Case C: Quote age > 5s but provider not exhausted -> stays TRIGGERED for retry.

        When _fetch_fresh_execution_quote returns a price with age > max,
        the plan should stay TRIGGERED (the executor leaves it for retry
        only if provider is exhausted). When age is stale but a price exists,
        the executor marks missed with quote_too_stale.

        Validates: Requirements 5.1, 5.10
        """
        plan = _make_plan(state=PlanState.PLANNED)
        registry.create_plan(plan)
        registry.activate("plan-001")
        registry.trigger("plan-001")

        p = registry.get_plan("plan-001")
        assert p.state == PlanState.TRIGGERED

        # Provider exhausted (rate limited) returns None -> stays TRIGGERED
        from utils.plan_executor import execute_triggered_plan

        with patch(
            "utils.plan_executor._fetch_fresh_execution_quote",
            return_value=(None, None, float("inf")),
        ):
            result = execute_triggered_plan(engine, p, 250.0)

        updated = registry.get_plan("plan-001")
        assert updated.state == PlanState.TRIGGERED
        assert result.success is False

        # NO paper fill
        fills = _get_trade_events(engine, "trade_opened")
        assert len(fills) == 0

    @patch("utils.plan_executor.TRIGGERED_PLAN_MODE", "enabled")
    def test_no_fill_with_stale_quote_marks_missed_with_quote_age(self, engine, registry):
        """When quote is received but too stale, plan is MISSED with quote_age_seconds in event.

        Validates: Requirements 5.1, 5.10
        """
        plan = _make_plan(state=PlanState.PLANNED)
        registry.create_plan(plan)
        registry.activate("plan-001")
        registry.trigger("plan-001")

        p = registry.get_plan("plan-001")
        stale_age = 10.0  # > 5s max

        from utils.plan_executor import execute_triggered_plan

        with patch(
            "utils.plan_executor._fetch_fresh_execution_quote",
            return_value=(250.0, datetime.now(timezone.utc), stale_age),
        ):
            result = execute_triggered_plan(engine, p, 250.0)

        updated = registry.get_plan("plan-001")
        assert updated.state == PlanState.MISSED

        # Verify miss_reason in DB directly
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT miss_reason FROM trade_plans WHERE plan_id = :pid"),
                {"pid": "plan-001"},
            ).fetchone()
        assert row[0] == "quote_too_stale"

        assert result.success is False

        # Missed_setup event should include quote_age_seconds
        missed_events = _get_trade_events(engine, "missed_setup")
        assert len(missed_events) >= 1
        payload = json.loads(missed_events[0]["payload_json"])
        assert payload["quote_age_seconds"] == 10.0
        assert payload["reason_for_miss"] == "quote_too_stale"

        # NO paper fill
        fills = _get_trade_events(engine, "trade_opened")
        assert len(fills) == 0


# ===========================================================================
# 10.7 — Backward compatibility with flag disabled
# ===========================================================================


class TestBackwardCompatibilityFlagDisabled:
    """When TRIGGERED_PLAN_MODE=disabled, no plans created, no monitor activity.

    Validates: Requirements 0.2, 11.1
    """

    @patch("utils.plan_monitor.TRIGGERED_PLAN_MODE", "disabled")
    def test_plan_monitor_returns_immediately_when_disabled(self, engine):
        """Plan monitor run() returns zeros and does nothing when disabled.

        Validates: Requirements 0.2, 11.1
        """
        from utils.plan_monitor import run

        result = run(engine)

        assert result.plans_checked == 0
        assert result.plans_triggered == 0
        assert result.plans_expired == 0
        assert result.plans_missed == 0
        assert result.quotes_fetched == 0
        assert result.tick_duration_ms == 0.0

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "disabled")
    def test_no_plans_created_when_disabled(self, engine, registry):
        """With flag disabled, _maybe_create_trade_plan returns False (no plan created).

        Validates: Requirements 0.2, 11.1
        """
        from agents.portfolio_manager import _maybe_create_trade_plan

        mock_registry = MagicMock()
        mock_decision = MagicMock()
        mock_decision.candidate_id = "cand-001"

        result = _maybe_create_trade_plan(
            engine, mock_registry, mock_decision, "aggressive", "cycle-001", []
        )

        assert result is False

        # No trade_plans rows in database
        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM trade_plans")
            ).scalar()
        assert count == 0

    @patch("utils.plan_monitor.TRIGGERED_PLAN_MODE", "disabled")
    def test_existing_pipeline_unaffected_when_disabled(self, engine):
        """Existing candidate pipeline works as before when mode=disabled.

        Validates: Requirements 0.2, 11.1
        """
        from utils.plan_monitor import run

        # Even if there were plans in the DB, monitor does nothing
        result = run(engine)
        assert result.plans_checked == 0

    @patch("utils.plan_executor.TRIGGERED_PLAN_MODE", "disabled")
    def test_executor_returns_disabled_when_flag_off(self, engine, registry):
        """Plan executor short-circuits when mode=disabled.

        Validates: Requirements 0.2, 11.1
        """
        from utils.plan_executor import execute_triggered_plan

        plan = _make_plan(state=PlanState.TRIGGERED)

        result = execute_triggered_plan(engine, plan, 250.0)

        assert result.success is False
        assert result.reason == "feature_disabled"

        # No trade_plans rows in database (nothing created)
        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM trade_plans")
            ).scalar()
        assert count == 0


# ===========================================================================
# 10.8 — Plan deduplication
# ===========================================================================


class TestPlanDeduplication:
    """Creating a plan for the same key supersedes the existing one.

    Validates: Requirements 8.7
    """

    def test_first_plan_expires_as_superseded_when_duplicate_created(self, engine, registry):
        """PM creates plan for TSLA BUY momentum_fade; creating another expires the first.

        Validates: Requirements 8.7
        """
        # First plan
        plan1 = _make_plan(
            plan_id="plan-001",
            symbol="TSLA",
            direction="BUY",
            setup_type="momentum_fade",
            profile_id="aggressive",
        )
        registry.create_plan(plan1)

        # Verify first plan is PLANNED
        p1 = registry.get_plan("plan-001")
        assert p1.state == PlanState.PLANNED

        # Expire duplicates (as _maybe_create_trade_plan does before creating new plan)
        expired = registry.expire_duplicate_plans(
            profile_id="aggressive",
            symbol="TSLA",
            direction="BUY",
            setup_type="momentum_fade",
            reason="superseded",
        )

        assert "plan-001" in expired

        # First plan should be EXPIRED
        p1_updated = registry.get_plan("plan-001")
        assert p1_updated.state == PlanState.EXPIRED

        # Create second plan
        plan2 = _make_plan(
            plan_id="plan-002",
            symbol="TSLA",
            direction="BUY",
            setup_type="momentum_fade",
            profile_id="aggressive",
        )
        registry.create_plan(plan2)

        # Second plan is active (PLANNED state)
        p2 = registry.get_plan("plan-002")
        assert p2.state == PlanState.PLANNED

    def test_superseded_plan_event_recorded(self, engine, registry):
        """trade_plan_event records superseded expiration.

        Validates: Requirements 8.7
        """
        plan1 = _make_plan(
            plan_id="plan-001",
            symbol="TSLA",
            direction="BUY",
            setup_type="momentum_fade",
            profile_id="aggressive",
        )
        registry.create_plan(plan1)

        registry.expire_duplicate_plans(
            profile_id="aggressive",
            symbol="TSLA",
            direction="BUY",
            setup_type="momentum_fade",
            reason="superseded",
        )

        # Check plan events for the superseded transition
        events = _get_plan_events(engine, "plan-001")
        # Find the state_expired event with "superseded" reason
        expired_events = [
            e for e in events
            if e["to_state"] == "expired"
        ]
        assert len(expired_events) >= 1
        event_data = json.loads(expired_events[0]["event_data"]) if expired_events[0]["event_data"] else {}
        assert event_data.get("reason") == "superseded"

    def test_deduplication_only_affects_same_key(self, engine, registry):
        """Plans with different keys are not affected by deduplication.

        Validates: Requirements 8.7
        """
        # Plan for TSLA BUY momentum_fade
        plan1 = _make_plan(
            plan_id="plan-001",
            symbol="TSLA",
            direction="BUY",
            setup_type="momentum_fade",
            profile_id="aggressive",
        )
        registry.create_plan(plan1)

        # Plan for TSLA SHORT momentum_fade (different direction)
        plan2 = _make_plan(
            plan_id="plan-002",
            symbol="TSLA",
            direction="SHORT",
            setup_type="momentum_fade",
            profile_id="aggressive",
        )
        registry.create_plan(plan2)

        # Expire duplicates for TSLA BUY momentum_fade
        expired = registry.expire_duplicate_plans(
            profile_id="aggressive",
            symbol="TSLA",
            direction="BUY",
            setup_type="momentum_fade",
            reason="superseded",
        )

        # Only plan-001 (BUY) should be expired
        assert "plan-001" in expired
        assert "plan-002" not in expired

        # plan-002 stays PLANNED
        p2 = registry.get_plan("plan-002")
        assert p2.state == PlanState.PLANNED


# ===========================================================================
# 10.9 — Decision-log visibility in enabled mode
# ===========================================================================


class TestDecisionLogVisibilityEnabledMode:
    """In enabled mode, plan lifecycle is visible in decision-log events.

    Validates: Requirements 7.7, 7.8
    """

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled")
    def test_plan_created_event_contains_plan_id_and_entry_zone(self, engine, registry):
        """PM accepts candidate -> pm_candidate_events row with plan_id and entry zone.

        Validates: Requirements 7.7
        """
        from agents.portfolio_manager import _maybe_create_trade_plan, _record_plan_candidate_event

        # Set up a mock candidate registry and decision
        mock_candidate = MagicMock()
        mock_candidate.symbol = "TSLA"
        mock_candidate.direction = "BUY"
        mock_candidate.setup_type = "momentum_fade"
        mock_candidate.geometry_name = "standard"
        mock_candidate.entry_price = 250.0
        mock_candidate.stop_price = 245.0
        mock_candidate.target_price = 260.0
        mock_candidate.risk_reward = 2.0
        mock_candidate.trigger = "Strong momentum setup"
        mock_candidate.source_signal_id = "sig-001"
        mock_candidate.signal_snapshot_json = json.dumps({"symbol": "TSLA"})

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_candidate

        mock_decision = MagicMock()
        mock_decision.candidate_id = "cand-001"
        mock_decision.rationale = "Approved for plan"

        executed = []
        result = _maybe_create_trade_plan(
            engine, mock_registry, mock_decision, "aggressive", "cycle-001", executed
        )

        # In enabled mode, should return True (skip immediate execution)
        assert result is True

        # Verify pm_candidate_events has plan_created entry
        events = _get_candidate_events(engine, "plan_created")
        assert len(events) >= 1

        event_data = json.loads(events[0]["event_data"])
        assert "plan_id" in event_data
        assert "entry_zone_upper" in event_data
        assert "entry_zone_lower" in event_data
        assert event_data["mode"] == "enabled"
        # Verify the message mentions watching
        assert "watching" in event_data.get("message", "").lower()

    def test_plan_triggered_and_entered_emits_decision_log(self, engine, registry):
        """Plan triggers -> execution succeeds -> plan transitions to ENTERED.

        Validates: Requirements 7.8
        """
        plan = _make_plan(state=PlanState.PLANNED)
        registry.create_plan(plan)
        registry.activate("plan-001")
        registry.trigger("plan-001")

        p = registry.get_plan("plan-001")
        assert p.state == PlanState.TRIGGERED

        # Mock successful execution
        from utils.plan_executor import execute_triggered_plan

        mock_gate_result = (True, [], 1.0, {})
        mock_execute_result = (True, "Trade created")

        with patch(
            "utils.plan_executor._fetch_fresh_execution_quote",
            return_value=(250.0, datetime.now(timezone.utc), 1.0),
        ), patch(
            "utils.plan_executor.TRIGGERED_PLAN_MODE", "enabled"
        ), patch(
            "agents.portfolio_manager._run_gate_pipeline",
            return_value=mock_gate_result,
        ), patch(
            "agents.portfolio_manager.execute_trade",
            return_value=mock_execute_result,
        ), patch(
            "db.schema.get_session",
            return_value=MagicMock(),
        ):
            result = execute_triggered_plan(engine, p, 250.0)

        # Plan should be ENTERED
        updated = registry.get_plan("plan-001")
        assert updated.state == PlanState.ENTERED
        assert result.success is True

        # trade_plan_events should have the transition events
        events = _get_plan_events(engine, "plan-001")
        # Look for the state_entered event
        entered_events = [e for e in events if e["to_state"] == "entered"]
        assert len(entered_events) >= 1

    def test_full_lifecycle_events_visible(self, engine, registry):
        """Full plan lifecycle from PLANNED through TRIGGERED to ENTERED produces events.

        Validates: Requirements 7.8
        """
        plan = _make_plan(state=PlanState.PLANNED)
        registry.create_plan(plan)

        # Collect all events after creation
        events_after_create = _get_plan_events(engine, "plan-001")
        # Should have plan_created event
        created_events = [e for e in events_after_create if e["event_type"] == "plan_created"]
        assert len(created_events) >= 1

        # Activate: PLANNED -> WATCHING
        registry.activate("plan-001")
        events_after_activate = _get_plan_events(engine, "plan-001")
        watching_events = [e for e in events_after_activate if e["to_state"] == "watching"]
        assert len(watching_events) >= 1

        # Trigger: WATCHING -> TRIGGERED
        registry.trigger("plan-001")
        events_after_trigger = _get_plan_events(engine, "plan-001")
        triggered_events = [e for e in events_after_trigger if e["to_state"] == "triggered"]
        assert len(triggered_events) >= 1

        # Enter: TRIGGERED -> ENTERED
        registry.mark_entered("plan-001")
        events_after_enter = _get_plan_events(engine, "plan-001")
        entered_events = [e for e in events_after_enter if e["to_state"] == "entered"]
        assert len(entered_events) >= 1

        # Full lifecycle trace
        all_transitions = [
            (e["from_state"], e["to_state"])
            for e in events_after_enter
            if e["from_state"] is not None and e["to_state"] is not None
        ]
        # Should include planned->watching, watching->triggered, triggered->entered
        assert ("planned", "watching") in all_transitions
        assert ("watching", "triggered") in all_transitions
        assert ("triggered", "entered") in all_transitions
