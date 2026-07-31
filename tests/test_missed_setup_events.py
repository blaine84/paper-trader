"""Tests for missed setup event recording (task 9.2).

Validates: Requirements 6.1–6.7

Tests that _record_missed_setup_event() in plan_monitor.py and plan_executor.py
correctly records structured "missed_setup" trade events with full plan metadata
for each reason_for_miss category.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, call

import pytest
from sqlalchemy import create_engine, text

from db.schema import Base, TradeEvent, get_session
from utils.trade_plan_registry import PlanState, TradePlan


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _create_tables(engine):
    """Create trade_plans, trade_plan_events, and trade_events tables in memory."""
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
    """In-memory SQLite engine with full schema for trade_events ORM table."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    _create_tables(eng)
    return eng


def _make_plan(
    plan_id: str = "plan-001",
    candidate_id: str = "cand-001",
    symbol: str = "MU",
    direction: str = "SHORT",
    setup_type: str = "momentum_fade",
    profile_id: str = "aggressive",
    cycle_id: str = "cycle-001",
    entry_reference: float = 782.67,
    entry_zone_upper: float = 785.0,
    entry_zone_lower: float = 780.0,
    stop_price: float = 790.0,
    target_price: float = 760.0,
    risk_reward: float = 3.1,
    geometry_name: str = "standard",
    state: PlanState = PlanState.WATCHING,
    expires_at: datetime | None = None,
) -> TradePlan:
    """Build a TradePlan with reasonable defaults for missed-setup testing."""
    now = datetime.now(timezone.utc)
    if expires_at is None:
        expires_at = now + timedelta(minutes=60)
    return TradePlan(
        plan_id=plan_id,
        candidate_id=candidate_id,
        cycle_id=cycle_id,
        profile_id=profile_id,
        symbol=symbol,
        direction=direction,
        setup_type=setup_type,
        geometry_name=geometry_name,
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
        analyst_reasoning="Strong fade setup near VWAP",
        pm_rationale="Approved for plan",
        source_signal_id="sig-001",
        signal_snapshot_json=json.dumps({"symbol": symbol}),
        state=state,
        created_at=now,
        expires_at=expires_at,
        triggered_at=None,
        executed_at=None,
        missed_at=None,
        integrity_hash="testhash_missed",
    )


# Required payload fields per Requirements 6.2
REQUIRED_PAYLOAD_FIELDS = frozenset({
    "plan_id",
    "candidate_id",
    "cycle_id",
    "symbol",
    "direction",
    "setup_type",
    "geometry_name",
    "profile_id",
    "entry_reference",
    "entry_zone_upper",
    "entry_zone_lower",
    "fresh_price_at_miss",
    "quote_timestamp",
    "quote_age_seconds",
    "intended_target",
    "original_stop",
    "original_risk_reward",
    "reason_for_miss",
})


# ---------------------------------------------------------------------------
# Test: missed event recorded for price_past_target (Requirement 6.3)
# ---------------------------------------------------------------------------


class TestMissedEventPricePastTarget:
    """Missed event recorded when plan misses due to price_past_target."""

    def test_plan_executor_records_price_past_target(self, engine):
        """plan_executor._record_missed_setup_event records event with reason price_past_target."""
        from utils.plan_executor import _record_missed_setup_event

        plan = _make_plan()
        fresh_price = 755.0  # Past target of 760 for SHORT
        quote_ts = datetime.now(timezone.utc)
        quote_age = 2.5

        _record_missed_setup_event(engine, plan, fresh_price, quote_ts, quote_age, "price_past_target")

        # Verify event stored in trade_events
        db = get_session(engine)
        try:
            events = db.query(TradeEvent).filter(
                TradeEvent.event_type == "missed_setup"
            ).all()
            assert len(events) == 1
            event = events[0]
            assert event.symbol == "MU"
            assert event.profile == "aggressive"

            payload = json.loads(event.payload_json)
            assert payload["reason_for_miss"] == "price_past_target"
            assert payload["fresh_price_at_miss"] == fresh_price
            assert payload["plan_id"] == "plan-001"
        finally:
            db.close()

    def test_plan_monitor_records_price_past_target(self, engine):
        """plan_monitor._record_missed_setup_event records event with reason price_past_target."""
        from utils.plan_monitor import _record_missed_setup_event

        plan = _make_plan()
        fresh_price = 755.0
        eval_ts = datetime.now(timezone.utc)
        quote_age = 0.0  # Monitor doesn't track quote age per evaluation

        _record_missed_setup_event(engine, plan, fresh_price, eval_ts, quote_age, "price_past_target")

        db = get_session(engine)
        try:
            events = db.query(TradeEvent).filter(
                TradeEvent.event_type == "missed_setup"
            ).all()
            assert len(events) == 1
            payload = json.loads(events[0].payload_json)
            assert payload["reason_for_miss"] == "price_past_target"
            assert payload["intended_target"] == 760.0
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Test: missed event recorded for price_beyond_zone (Requirement 6.3)
# ---------------------------------------------------------------------------


class TestMissedEventPriceBeyondZone:
    """Missed event recorded when plan misses due to price_beyond_zone."""

    def test_plan_executor_records_price_beyond_zone(self, engine):
        """plan_executor records missed_setup with reason price_beyond_zone."""
        from utils.plan_executor import _record_missed_setup_event

        plan = _make_plan()
        fresh_price = 770.0  # Well below the entry zone lower of 780.0
        quote_ts = datetime.now(timezone.utc)
        quote_age = 3.0

        _record_missed_setup_event(engine, plan, fresh_price, quote_ts, quote_age, "price_beyond_zone")

        db = get_session(engine)
        try:
            events = db.query(TradeEvent).filter(
                TradeEvent.event_type == "missed_setup"
            ).all()
            assert len(events) == 1
            payload = json.loads(events[0].payload_json)
            assert payload["reason_for_miss"] == "price_beyond_zone"
            assert payload["fresh_price_at_miss"] == 770.0
            assert payload["entry_zone_upper"] == 785.0
            assert payload["entry_zone_lower"] == 780.0
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Test: missed event recorded for plan_expired (Requirement 6.3)
# ---------------------------------------------------------------------------


class TestMissedEventPlanExpired:
    """Missed event recorded when plan expires (plan_expired)."""

    def test_plan_monitor_records_plan_expired(self, engine):
        """plan_monitor records missed_setup with reason plan_expired when TTL exceeded."""
        from utils.plan_monitor import _record_missed_setup_event

        plan = _make_plan(
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        )
        # For expired plans, fresh_price is 0.0 (no quote evaluated)
        eval_ts = datetime.now(timezone.utc)

        _record_missed_setup_event(engine, plan, 0.0, eval_ts, 0.0, "plan_expired")

        db = get_session(engine)
        try:
            events = db.query(TradeEvent).filter(
                TradeEvent.event_type == "missed_setup"
            ).all()
            assert len(events) == 1
            payload = json.loads(events[0].payload_json)
            assert payload["reason_for_miss"] == "plan_expired"
            assert payload["plan_id"] == "plan-001"
            assert payload["symbol"] == "MU"
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Test: missed event recorded for invalidation_triggered (Requirement 6.3)
# ---------------------------------------------------------------------------


class TestMissedEventInvalidationTriggered:
    """Missed event recorded when plan is invalidated (invalidation_triggered)."""

    def test_plan_monitor_records_invalidation_triggered(self, engine):
        """plan_monitor records missed_setup with reason invalidation_triggered."""
        from utils.plan_monitor import _record_missed_setup_event

        plan = _make_plan(
            plan_id="plan-invalidated",
            symbol="TSLA",
            direction="BUY",
            entry_reference=250.0,
            entry_zone_upper=252.0,
            entry_zone_lower=248.0,
            stop_price=245.0,
            target_price=270.0,
        )
        eval_ts = datetime.now(timezone.utc)
        # Price that breached invalidation level
        fresh_price = 242.0

        _record_missed_setup_event(engine, plan, fresh_price, eval_ts, 0.0, "invalidation_triggered")

        db = get_session(engine)
        try:
            events = db.query(TradeEvent).filter(
                TradeEvent.event_type == "missed_setup"
            ).all()
            assert len(events) == 1
            payload = json.loads(events[0].payload_json)
            assert payload["reason_for_miss"] == "invalidation_triggered"
            assert payload["plan_id"] == "plan-invalidated"
            assert payload["fresh_price_at_miss"] == 242.0
            assert payload["symbol"] == "TSLA"
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Test: missed event recorded for no_fresh_price_available (Requirement 6.3)
# ---------------------------------------------------------------------------


class TestMissedEventNoFreshPriceAvailable:
    """Missed event recorded when orphan sweep catches plan with no fresh price."""

    def test_orphan_sweep_records_no_fresh_price_available(self, engine):
        """finalize_orphaned_plans records missed_setup with reason no_fresh_price_available
        for TRIGGERED plans that never obtained a fresh execution quote."""
        from utils.trade_plan_registry import TradePlanRegistry, PlanState

        registry = TradePlanRegistry(engine)

        # Create a plan and advance it to TRIGGERED, then expire it
        plan = _make_plan(
            plan_id="plan-no-price",
            state=PlanState.PLANNED,
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        )
        # Insert directly in TRIGGERED state with expired TTL
        _insert_triggered_plan(engine, plan)

        # Run orphan sweep — should mark MISSED and record event
        swept = registry.finalize_orphaned_plans()
        assert "plan-no-price" in swept
        assert swept["plan-no-price"] == PlanState.MISSED

        # Verify missed_setup trade event was recorded
        db = get_session(engine)
        try:
            events = db.query(TradeEvent).filter(
                TradeEvent.event_type == "missed_setup"
            ).all()
            assert len(events) == 1
            payload = json.loads(events[0].payload_json)
            assert payload["reason_for_miss"] == "no_fresh_price_available"
            assert payload["plan_id"] == "plan-no-price"
            assert payload["fresh_price_at_miss"] is None
        finally:
            db.close()


def _insert_triggered_plan(engine, plan: TradePlan):
    """Insert a plan directly in TRIGGERED state for orphan sweep testing."""
    from sqlalchemy import text as sql_text
    with engine.begin() as conn:
        conn.execute(sql_text("""
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
                miss_reason, rejection_reason, integrity_hash
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
                :miss_reason, :rejection_reason, :integrity_hash
            )
        """), {
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
            "state": "triggered",
            "created_at": plan.created_at.isoformat(),
            "expires_at": plan.expires_at.isoformat(),
            "triggered_at": plan.created_at.isoformat(),
            "executed_at": None,
            "missed_at": None,
            "miss_reason": None,
            "rejection_reason": None,
            "integrity_hash": plan.integrity_hash,
        })


# ---------------------------------------------------------------------------
# Test: missed event recorded for quote_too_stale (Requirement 6.3)
# ---------------------------------------------------------------------------


class TestMissedEventQuoteTooStale:
    """Missed event recorded when quote too stale (quote_too_stale)."""

    def test_plan_executor_records_quote_too_stale(self, engine):
        """plan_executor records missed_setup with reason quote_too_stale."""
        from utils.plan_executor import _record_missed_setup_event

        plan = _make_plan(state=PlanState.TRIGGERED)
        fresh_price = 783.0
        quote_ts = datetime.now(timezone.utc) - timedelta(seconds=10)
        quote_age = 10.0  # Exceeds PLAN_EXECUTION_MAX_QUOTE_AGE_SECONDS (5s)

        _record_missed_setup_event(engine, plan, fresh_price, quote_ts, quote_age, "quote_too_stale")

        db = get_session(engine)
        try:
            events = db.query(TradeEvent).filter(
                TradeEvent.event_type == "missed_setup"
            ).all()
            assert len(events) == 1
            payload = json.loads(events[0].payload_json)
            assert payload["reason_for_miss"] == "quote_too_stale"
            assert payload["quote_age_seconds"] == 10.0
            assert payload["fresh_price_at_miss"] == 783.0
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Test: payload contains ALL required fields (Requirements 6.2, 6.7)
# ---------------------------------------------------------------------------


class TestMissedEventPayloadCompleteness:
    """Missed event payload contains all required fields including quote_age_seconds and quote_timestamp."""

    def test_executor_payload_has_all_required_fields(self, engine):
        """plan_executor missed_setup payload contains every field from Requirement 6.2."""
        from utils.plan_executor import _record_missed_setup_event

        plan = _make_plan(
            plan_id="plan-complete",
            candidate_id="cand-complete",
            cycle_id="cycle-complete",
            symbol="NVDA",
            direction="BUY",
            setup_type="technical_breakout",
            profile_id="moderate",
            entry_reference=500.0,
            entry_zone_upper=502.0,
            entry_zone_lower=498.0,
            stop_price=495.0,
            target_price=520.0,
            risk_reward=4.0,
            geometry_name="breakout_geo",
        )
        quote_ts = datetime(2025, 7, 29, 14, 30, 0, tzinfo=timezone.utc)
        quote_age = 4.2

        _record_missed_setup_event(
            engine, plan, 503.5, quote_ts, quote_age, "price_beyond_zone"
        )

        db = get_session(engine)
        try:
            events = db.query(TradeEvent).filter(
                TradeEvent.event_type == "missed_setup"
            ).all()
            assert len(events) == 1
            payload = json.loads(events[0].payload_json)

            # Verify ALL required fields are present
            missing = REQUIRED_PAYLOAD_FIELDS - set(payload.keys())
            assert not missing, f"Missing required payload fields: {missing}"

            # Verify field values are correct
            assert payload["plan_id"] == "plan-complete"
            assert payload["candidate_id"] == "cand-complete"
            assert payload["cycle_id"] == "cycle-complete"
            assert payload["symbol"] == "NVDA"
            assert payload["direction"] == "BUY"
            assert payload["setup_type"] == "technical_breakout"
            assert payload["geometry_name"] == "breakout_geo"
            assert payload["profile_id"] == "moderate"
            assert payload["entry_reference"] == 500.0
            assert payload["entry_zone_upper"] == 502.0
            assert payload["entry_zone_lower"] == 498.0
            assert payload["fresh_price_at_miss"] == 503.5
            assert payload["quote_timestamp"] == "2025-07-29T14:30:00+00:00"
            assert payload["quote_age_seconds"] == 4.2
            assert payload["intended_target"] == 520.0
            assert payload["original_stop"] == 495.0
            assert payload["original_risk_reward"] == 4.0
            assert payload["reason_for_miss"] == "price_beyond_zone"
        finally:
            db.close()

    def test_monitor_payload_has_all_required_fields(self, engine):
        """plan_monitor missed_setup payload contains every field from Requirement 6.2."""
        from utils.plan_monitor import _record_missed_setup_event

        plan = _make_plan(
            plan_id="plan-mon-complete",
            candidate_id="cand-mon-complete",
            symbol="AAPL",
            direction="BUY",
            setup_type="gap_and_go",
            profile_id="conservative",
            entry_reference=195.0,
            entry_zone_upper=196.0,
            entry_zone_lower=194.0,
            stop_price=192.0,
            target_price=202.0,
            risk_reward=2.3,
            geometry_name="gap_geo",
        )
        eval_ts = datetime(2025, 7, 29, 15, 0, 0, tzinfo=timezone.utc)

        _record_missed_setup_event(
            engine, plan, 203.0, eval_ts, 0.0, "price_past_target"
        )

        db = get_session(engine)
        try:
            events = db.query(TradeEvent).filter(
                TradeEvent.event_type == "missed_setup"
            ).all()
            assert len(events) == 1
            payload = json.loads(events[0].payload_json)

            missing = REQUIRED_PAYLOAD_FIELDS - set(payload.keys())
            assert not missing, f"Missing required payload fields: {missing}"

            # Verify quote_timestamp and quote_age_seconds present
            assert payload["quote_timestamp"] is not None
            assert "quote_age_seconds" in payload
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Test: missed events are queryable by symbol, profile, reason_for_miss
#       (Requirement 6.4)
# ---------------------------------------------------------------------------


class TestMissedEventsQueryable:
    """Missed events are queryable by symbol, profile, reason_for_miss."""

    def test_query_by_symbol(self, engine):
        """Missed events can be filtered by symbol."""
        from utils.plan_executor import _record_missed_setup_event

        now = datetime.now(timezone.utc)
        plan_mu = _make_plan(plan_id="plan-mu", symbol="MU")
        plan_tsla = _make_plan(plan_id="plan-tsla", symbol="TSLA", direction="BUY",
                               target_price=300.0)

        _record_missed_setup_event(engine, plan_mu, 755.0, now, 2.0, "price_past_target")
        _record_missed_setup_event(engine, plan_tsla, 305.0, now, 1.5, "price_past_target")

        db = get_session(engine)
        try:
            mu_events = db.query(TradeEvent).filter(
                TradeEvent.event_type == "missed_setup",
                TradeEvent.symbol == "MU",
            ).all()
            assert len(mu_events) == 1
            assert json.loads(mu_events[0].payload_json)["plan_id"] == "plan-mu"

            tsla_events = db.query(TradeEvent).filter(
                TradeEvent.event_type == "missed_setup",
                TradeEvent.symbol == "TSLA",
            ).all()
            assert len(tsla_events) == 1
            assert json.loads(tsla_events[0].payload_json)["plan_id"] == "plan-tsla"
        finally:
            db.close()

    def test_query_by_profile(self, engine):
        """Missed events can be filtered by profile."""
        from utils.plan_executor import _record_missed_setup_event

        now = datetime.now(timezone.utc)
        plan_agg = _make_plan(plan_id="plan-agg", profile_id="aggressive")
        plan_mod = _make_plan(plan_id="plan-mod", profile_id="moderate")

        _record_missed_setup_event(engine, plan_agg, 755.0, now, 2.0, "price_past_target")
        _record_missed_setup_event(engine, plan_mod, 755.0, now, 2.0, "price_beyond_zone")

        db = get_session(engine)
        try:
            agg_events = db.query(TradeEvent).filter(
                TradeEvent.event_type == "missed_setup",
                TradeEvent.profile == "aggressive",
            ).all()
            assert len(agg_events) == 1
            assert json.loads(agg_events[0].payload_json)["profile_id"] == "aggressive"

            mod_events = db.query(TradeEvent).filter(
                TradeEvent.event_type == "missed_setup",
                TradeEvent.profile == "moderate",
            ).all()
            assert len(mod_events) == 1
            assert json.loads(mod_events[0].payload_json)["profile_id"] == "moderate"
        finally:
            db.close()

    def test_query_by_reason_for_miss_in_payload(self, engine):
        """Missed events can be filtered by reason_for_miss via payload_json."""
        from utils.plan_executor import _record_missed_setup_event

        now = datetime.now(timezone.utc)
        plan1 = _make_plan(plan_id="plan-r1")
        plan2 = _make_plan(plan_id="plan-r2")
        plan3 = _make_plan(plan_id="plan-r3")

        _record_missed_setup_event(engine, plan1, 755.0, now, 2.0, "price_past_target")
        _record_missed_setup_event(engine, plan2, 770.0, now, 3.0, "price_beyond_zone")
        _record_missed_setup_event(engine, plan3, 783.0, now, 10.0, "quote_too_stale")

        db = get_session(engine)
        try:
            all_events = db.query(TradeEvent).filter(
                TradeEvent.event_type == "missed_setup",
            ).all()
            assert len(all_events) == 3

            # Filter by reason via payload
            reasons = {
                json.loads(e.payload_json)["reason_for_miss"]
                for e in all_events
            }
            assert reasons == {"price_past_target", "price_beyond_zone", "quote_too_stale"}

            # Can identify specific event by reason
            past_target = [
                e for e in all_events
                if json.loads(e.payload_json)["reason_for_miss"] == "price_past_target"
            ]
            assert len(past_target) == 1
            assert json.loads(past_target[0].payload_json)["plan_id"] == "plan-r1"
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Test: missed event preserves full plan metadata for counterfactual analysis
#       (Requirement 6.7)
# ---------------------------------------------------------------------------


class TestMissedEventPreservesMetadata:
    """Missed event preserves full plan metadata for counterfactual analysis."""

    def test_full_plan_metadata_preserved_for_counterfactual(self, engine):
        """All plan geometry, entry zone, and identification fields are preserved."""
        from utils.plan_executor import _record_missed_setup_event

        plan = _make_plan(
            plan_id="plan-counterfactual",
            candidate_id="cand-cf-001",
            cycle_id="cycle-afternoon",
            symbol="AMD",
            direction="BUY",
            setup_type="technical_breakout",
            profile_id="aggressive",
            entry_reference=160.0,
            entry_zone_upper=161.5,
            entry_zone_lower=158.5,
            stop_price=156.0,
            target_price=172.0,
            risk_reward=3.0,
            geometry_name="breakout_standard",
        )
        quote_ts = datetime(2025, 7, 29, 10, 15, 30, tzinfo=timezone.utc)

        _record_missed_setup_event(
            engine, plan, 173.0, quote_ts, 1.8, "price_past_target"
        )

        db = get_session(engine)
        try:
            events = db.query(TradeEvent).filter(
                TradeEvent.event_type == "missed_setup"
            ).all()
            assert len(events) == 1
            payload = json.loads(events[0].payload_json)

            # Identity fields for lineage
            assert payload["plan_id"] == "plan-counterfactual"
            assert payload["candidate_id"] == "cand-cf-001"
            assert payload["cycle_id"] == "cycle-afternoon"

            # Classification fields
            assert payload["symbol"] == "AMD"
            assert payload["direction"] == "BUY"
            assert payload["setup_type"] == "technical_breakout"
            assert payload["geometry_name"] == "breakout_standard"
            assert payload["profile_id"] == "aggressive"

            # Entry zone for "what was the plan"
            assert payload["entry_reference"] == 160.0
            assert payload["entry_zone_upper"] == 161.5
            assert payload["entry_zone_lower"] == 158.5

            # Geometry for counterfactual P&L analysis
            assert payload["original_stop"] == 156.0
            assert payload["intended_target"] == 172.0
            assert payload["original_risk_reward"] == 3.0

            # Market state at miss for "what actually happened"
            assert payload["fresh_price_at_miss"] == 173.0
            assert payload["quote_timestamp"] == "2025-07-29T10:15:30+00:00"
            assert payload["quote_age_seconds"] == 1.8
            assert payload["reason_for_miss"] == "price_past_target"
        finally:
            db.close()

    def test_monitor_preserves_metadata_on_expired(self, engine):
        """plan_monitor preserves full metadata even for expired plans."""
        from utils.plan_monitor import _record_missed_setup_event

        plan = _make_plan(
            plan_id="plan-expired-meta",
            candidate_id="cand-exp-001",
            symbol="GOOGL",
            direction="BUY",
            setup_type="vwap_reclaim",
            profile_id="moderate",
            entry_reference=180.0,
            entry_zone_upper=181.0,
            entry_zone_lower=179.0,
            stop_price=177.0,
            target_price=190.0,
            risk_reward=3.3,
            geometry_name="vwap_geo",
        )
        eval_ts = datetime(2025, 7, 29, 16, 0, 0, tzinfo=timezone.utc)

        _record_missed_setup_event(engine, plan, 0.0, eval_ts, 0.0, "plan_expired")

        db = get_session(engine)
        try:
            events = db.query(TradeEvent).filter(
                TradeEvent.event_type == "missed_setup"
            ).all()
            assert len(events) == 1
            payload = json.loads(events[0].payload_json)

            # All metadata preserved even though no price evaluation occurred
            assert payload["plan_id"] == "plan-expired-meta"
            assert payload["candidate_id"] == "cand-exp-001"
            assert payload["symbol"] == "GOOGL"
            assert payload["setup_type"] == "vwap_reclaim"
            assert payload["original_stop"] == 177.0
            assert payload["intended_target"] == 190.0
            assert payload["original_risk_reward"] == 3.3
            assert payload["reason_for_miss"] == "plan_expired"
        finally:
            db.close()
