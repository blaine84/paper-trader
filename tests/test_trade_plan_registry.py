"""Tests for TradePlanRegistry — CAS state machine for triggered trade plans.

Validates: plan creation, state transitions, CAS semantics, event emission,
deduplication, and orphan sweep.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text

from utils.trade_plan_registry import (
    PlanState,
    TradePlan,
    TradePlanRegistry,
    TradePlanRegistryError,
    TERMINAL_STATES,
    _compute_plan_integrity_hash,
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
    state: PlanState = PlanState.PLANNED,
    expires_at: datetime | None = None,
) -> TradePlan:
    """Build a TradePlan with reasonable defaults for testing."""
    now = datetime.now(timezone.utc)
    if expires_at is None:
        expires_at = now + timedelta(minutes=60)
    plan = TradePlan(
        plan_id=plan_id,
        candidate_id=candidate_id,
        cycle_id="cycle-001",
        profile_id="aggressive",
        symbol=symbol,
        direction=direction,
        setup_type=setup_type,
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
        invalidation_logic_json=json.dumps({"invalidation_basis": "support_break"}),
        analyst_reasoning="Strong momentum setup",
        pm_rationale="Approved for plan",
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
    return plan


# ---------------------------------------------------------------------------
# Plan Creation
# ---------------------------------------------------------------------------


def test_create_plan_inserts_with_planned_state(registry):
    plan = _make_plan()
    plan_id = registry.create_plan(plan)
    assert plan_id == "plan-001"

    fetched = registry.get_plan("plan-001")
    assert fetched is not None
    assert fetched.state == PlanState.PLANNED
    assert fetched.symbol == "TSLA"
    assert fetched.direction == "BUY"


def test_create_plan_computes_integrity_hash(registry):
    plan = _make_plan()
    registry.create_plan(plan)
    fetched = registry.get_plan("plan-001")
    assert fetched.integrity_hash != ""
    assert len(fetched.integrity_hash) == 64  # SHA-256 hex


def test_create_plan_emits_event(engine, registry):
    plan = _make_plan()
    registry.create_plan(plan)

    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT * FROM trade_plan_events WHERE plan_id = :pid"
        ), {"pid": "plan-001"})
        events = result.mappings().all()

    assert len(events) == 1
    assert events[0]["event_type"] == "plan_created"
    assert events[0]["to_state"] == "planned"


# ---------------------------------------------------------------------------
# State Transitions — Happy Path
# ---------------------------------------------------------------------------


def test_activate_transitions_planned_to_watching(registry):
    plan = _make_plan()
    registry.create_plan(plan)
    registry.activate("plan-001")
    fetched = registry.get_plan("plan-001")
    assert fetched.state == PlanState.WATCHING


def test_trigger_transitions_watching_to_triggered(registry):
    plan = _make_plan()
    registry.create_plan(plan)
    registry.activate("plan-001")
    registry.trigger("plan-001")
    fetched = registry.get_plan("plan-001")
    assert fetched.state == PlanState.TRIGGERED
    assert fetched.triggered_at is not None


def test_mark_entered_transitions_triggered_to_entered(registry):
    plan = _make_plan()
    registry.create_plan(plan)
    registry.activate("plan-001")
    registry.trigger("plan-001")
    registry.mark_entered("plan-001")
    fetched = registry.get_plan("plan-001")
    assert fetched.state == PlanState.ENTERED
    assert fetched.executed_at is not None


def test_mark_executed_alias_works(registry):
    plan = _make_plan()
    registry.create_plan(plan)
    registry.activate("plan-001")
    registry.trigger("plan-001")
    registry.mark_executed("plan-001")
    fetched = registry.get_plan("plan-001")
    assert fetched.state == PlanState.ENTERED


def test_mark_missed_from_watching(registry):
    plan = _make_plan()
    registry.create_plan(plan)
    registry.activate("plan-001")
    registry.mark_missed("plan-001", reason="price_past_target", fresh_price=265.0)
    fetched = registry.get_plan("plan-001")
    assert fetched.state == PlanState.MISSED
    assert fetched.missed_at is not None


def test_mark_missed_from_triggered(registry):
    plan = _make_plan()
    registry.create_plan(plan)
    registry.activate("plan-001")
    registry.trigger("plan-001")
    registry.mark_missed("plan-001", reason="price_beyond_zone", fresh_price=240.0)
    fetched = registry.get_plan("plan-001")
    assert fetched.state == PlanState.MISSED


def test_mark_expired_from_planned(registry):
    plan = _make_plan()
    registry.create_plan(plan)
    registry.mark_expired("plan-001")
    fetched = registry.get_plan("plan-001")
    assert fetched.state == PlanState.EXPIRED


def test_mark_expired_from_watching(registry):
    plan = _make_plan()
    registry.create_plan(plan)
    registry.activate("plan-001")
    registry.mark_expired("plan-001")
    fetched = registry.get_plan("plan-001")
    assert fetched.state == PlanState.EXPIRED


def test_mark_rejected_from_watching(registry):
    plan = _make_plan()
    registry.create_plan(plan)
    registry.activate("plan-001")
    registry.mark_rejected("plan-001", reason="invalidation_triggered")
    fetched = registry.get_plan("plan-001")
    assert fetched.state == PlanState.REJECTED


# ---------------------------------------------------------------------------
# CAS Semantics — Terminal states cannot transition
# ---------------------------------------------------------------------------


def test_terminal_state_cannot_transition(registry):
    plan = _make_plan()
    registry.create_plan(plan)
    registry.activate("plan-001")
    registry.mark_expired("plan-001")

    # Trying to activate an expired plan should fail
    with pytest.raises(TradePlanRegistryError):
        registry.activate("plan-001")


def test_activate_on_watching_fails(registry):
    plan = _make_plan()
    registry.create_plan(plan)
    registry.activate("plan-001")

    # Already WATCHING — cannot activate again
    with pytest.raises(TradePlanRegistryError):
        registry.activate("plan-001")


def test_trigger_on_planned_fails(registry):
    """Cannot skip WATCHING and go directly to TRIGGERED."""
    plan = _make_plan()
    registry.create_plan(plan)

    with pytest.raises(TradePlanRegistryError):
        registry.trigger("plan-001")


# ---------------------------------------------------------------------------
# Event Emission — Every transition emits an event
# ---------------------------------------------------------------------------


def test_every_transition_emits_event(engine, registry):
    plan = _make_plan()
    registry.create_plan(plan)
    registry.activate("plan-001")
    registry.trigger("plan-001")
    registry.mark_entered("plan-001")

    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT event_type, from_state, to_state FROM trade_plan_events "
            "WHERE plan_id = :pid ORDER BY id"
        ), {"pid": "plan-001"})
        events = result.mappings().all()

    # plan_created + state_watching + state_triggered + state_entered = 4 events
    assert len(events) == 4
    assert events[0]["event_type"] == "plan_created"
    assert events[1]["event_type"] == "state_watching"
    assert events[1]["from_state"] == "planned"
    assert events[1]["to_state"] == "watching"
    assert events[2]["event_type"] == "state_triggered"
    assert events[3]["event_type"] == "state_entered"


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def test_get_active_plans_returns_planned_and_watching(registry):
    plan1 = _make_plan(plan_id="plan-001")
    plan2 = _make_plan(plan_id="plan-002", candidate_id="cand-002")
    registry.create_plan(plan1)
    registry.create_plan(plan2)
    registry.activate("plan-002")  # WATCHING

    active = registry.get_active_plans()
    assert len(active) == 2
    states = {p.plan_id: p.state for p in active}
    assert states["plan-001"] == PlanState.PLANNED
    assert states["plan-002"] == PlanState.WATCHING


def test_get_active_plans_filters_by_symbol(registry):
    plan1 = _make_plan(plan_id="plan-001", symbol="TSLA")
    plan2 = _make_plan(plan_id="plan-002", symbol="AAPL", candidate_id="cand-002")
    registry.create_plan(plan1)
    registry.create_plan(plan2)

    active = registry.get_active_plans(symbol="TSLA")
    assert len(active) == 1
    assert active[0].symbol == "TSLA"


def test_get_triggered_plans(registry):
    plan = _make_plan()
    registry.create_plan(plan)
    registry.activate("plan-001")
    registry.trigger("plan-001")

    triggered = registry.get_triggered_plans()
    assert len(triggered) == 1
    assert triggered[0].state == PlanState.TRIGGERED


def test_get_plan_returns_none_for_unknown(registry):
    assert registry.get_plan("nonexistent") is None


def test_has_active_plan_for_candidate(registry):
    plan = _make_plan()
    registry.create_plan(plan)
    assert registry.has_active_plan_for_candidate("cand-001") is True
    assert registry.has_active_plan_for_candidate("cand-999") is False

    # After expiration, no longer active
    registry.mark_expired("plan-001")
    assert registry.has_active_plan_for_candidate("cand-001") is False


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_expire_duplicate_plans(registry):
    plan1 = _make_plan(plan_id="plan-001")
    registry.create_plan(plan1)
    registry.activate("plan-001")

    # Expire duplicates for same key
    expired = registry.expire_duplicate_plans(
        profile_id="aggressive",
        symbol="TSLA",
        direction="BUY",
        setup_type="momentum_fade",
        reason="superseded",
    )
    assert expired == ["plan-001"]
    fetched = registry.get_plan("plan-001")
    assert fetched.state == PlanState.EXPIRED


def test_expire_duplicate_plans_no_duplicates(registry):
    plan1 = _make_plan(plan_id="plan-001", symbol="TSLA")
    registry.create_plan(plan1)

    expired = registry.expire_duplicate_plans(
        profile_id="aggressive",
        symbol="AAPL",  # different symbol
        direction="BUY",
        setup_type="momentum_fade",
    )
    assert expired == []


# ---------------------------------------------------------------------------
# Orphan Sweep
# ---------------------------------------------------------------------------


def test_finalize_orphaned_plans_expires_past_ttl(registry):
    expired_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    plan = _make_plan(plan_id="plan-001", expires_at=expired_time)
    registry.create_plan(plan)

    swept = registry.finalize_orphaned_plans()
    assert "plan-001" in swept
    assert swept["plan-001"] == PlanState.EXPIRED

    fetched = registry.get_plan("plan-001")
    assert fetched.state == PlanState.EXPIRED


def test_finalize_orphaned_plans_misses_triggered_past_ttl(registry):
    expired_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    plan = _make_plan(plan_id="plan-001", expires_at=expired_time)
    registry.create_plan(plan)
    registry.activate("plan-001")
    registry.trigger("plan-001")

    swept = registry.finalize_orphaned_plans()
    assert "plan-001" in swept
    assert swept["plan-001"] == PlanState.MISSED

    fetched = registry.get_plan("plan-001")
    assert fetched.state == PlanState.MISSED


def test_finalize_orphaned_plans_ignores_non_expired(registry):
    future_time = datetime.now(timezone.utc) + timedelta(minutes=60)
    plan = _make_plan(plan_id="plan-001", expires_at=future_time)
    registry.create_plan(plan)

    swept = registry.finalize_orphaned_plans()
    assert swept == {}
    fetched = registry.get_plan("plan-001")
    assert fetched.state == PlanState.PLANNED


# ---------------------------------------------------------------------------
# Integrity Hash
# ---------------------------------------------------------------------------


def test_mark_rejected_from_triggered(registry):
    """mark_rejected on TRIGGERED state transitions to REJECTED."""
    plan = _make_plan()
    registry.create_plan(plan)
    registry.activate("plan-001")
    registry.trigger("plan-001")
    registry.mark_rejected("plan-001", reason="gate_failure")
    fetched = registry.get_plan("plan-001")
    assert fetched.state == PlanState.REJECTED


def test_mark_missed_event_has_fresh_price(engine, registry):
    """mark_missed event records fresh_price in the trade_plan_events row."""
    plan = _make_plan()
    registry.create_plan(plan)
    registry.activate("plan-001")
    registry.mark_missed("plan-001", reason="price_past_target", fresh_price=265.0)

    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT fresh_price FROM trade_plan_events "
            "WHERE plan_id = :pid AND event_type = 'state_missed'"
        ), {"pid": "plan-001"})
        row = result.mappings().first()

    assert row is not None
    assert row["fresh_price"] == 265.0


def test_expire_duplicate_plans_superseded_event(engine, registry):
    """Expired duplicate gets reason='superseded' in plan event data."""
    plan1 = _make_plan(plan_id="plan-001")
    registry.create_plan(plan1)
    registry.activate("plan-001")

    registry.expire_duplicate_plans(
        profile_id="aggressive",
        symbol="TSLA",
        direction="BUY",
        setup_type="momentum_fade",
        reason="superseded",
    )

    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT event_data FROM trade_plan_events "
            "WHERE plan_id = :pid AND event_type = 'state_expired'"
        ), {"pid": "plan-001"})
        row = result.mappings().first()

    assert row is not None
    event_data = json.loads(row["event_data"])
    assert event_data["reason"] == "superseded"


def test_concurrent_cas_only_one_succeeds(engine, registry):
    """Simulates concurrent CAS: after first transition succeeds,
    a second attempt on the same plan fails (rowcount=0).
    """
    plan = _make_plan()
    registry.create_plan(plan)
    registry.activate("plan-001")

    # First mark_expired succeeds
    registry.mark_expired("plan-001")
    assert registry.get_plan("plan-001").state == PlanState.EXPIRED

    # Second attempt to transition (e.g. mark_missed) fails — CAS rejects
    with pytest.raises(TradePlanRegistryError):
        registry.mark_missed("plan-001", reason="price_past_target", fresh_price=270.0)


# ---------------------------------------------------------------------------
# Integrity Hash
# ---------------------------------------------------------------------------


def test_integrity_hash_is_deterministic():
    plan = _make_plan()
    h1 = _compute_plan_integrity_hash(plan)
    h2 = _compute_plan_integrity_hash(plan)
    assert h1 == h2
    assert len(h1) == 64


def test_integrity_hash_changes_with_fields():
    plan1 = _make_plan(plan_id="plan-001")
    plan2 = _make_plan(plan_id="plan-002")
    h1 = _compute_plan_integrity_hash(plan1)
    h2 = _compute_plan_integrity_hash(plan2)
    assert h1 != h2


def test_integrity_hash_changes_with_geometry():
    """Hash differs when geometry fields (stop/target) change."""
    plan1 = _make_plan(plan_id="plan-001")
    plan2 = TradePlan(
        plan_id="plan-001",
        candidate_id="cand-001",
        cycle_id="cycle-001",
        profile_id="aggressive",
        symbol="TSLA",
        direction="BUY",
        setup_type="momentum_fade",
        geometry_name="standard",
        entry_reference=250.0,
        entry_zone_upper=252.0,
        entry_zone_lower=248.0,
        stop_price=240.0,  # Different stop
        target_price=260.0,
        risk_reward=2.0,
        trigger_type="price_in_zone",
        trigger_condition_json=json.dumps({"type": "price_in_zone"}),
        trigger_confirmation_required=False,
        invalidation_logic_json=json.dumps({"invalidation_basis": "support_break"}),
        analyst_reasoning="Strong momentum setup",
        pm_rationale="Approved for plan",
        source_signal_id="sig-001",
        signal_snapshot_json=json.dumps({"symbol": "TSLA"}),
        state=PlanState.PLANNED,
        created_at=plan1.created_at,
        expires_at=plan1.expires_at,
        triggered_at=None,
        executed_at=None,
        missed_at=None,
        integrity_hash="",
    )
    h1 = _compute_plan_integrity_hash(plan1)
    h2 = _compute_plan_integrity_hash(plan2)
    assert h1 != h2
