"""Integration tests for PM plan creation in the candidate acceptance loop.

Exercises the real production helpers in ``agents/portfolio_manager.py``
(``_maybe_create_trade_plan`` / ``_build_trade_plan_from_candidate``) against a
real ``CandidateRegistry`` and ``TradePlanRegistry`` backed by in-memory SQLite,
verifying behavior across all TRIGGERED_PLAN_MODE values.

``_maybe_create_trade_plan`` returns the "skip immediate execution" signal that
the acceptance loop consumes with ``continue``, so its return value is the
observable contract for whether ``execute_candidate_pipeline`` runs.

Requirements: 7.1, 7.2, 7.3, 7.5, 7.6, 7.7, 8.7, 11.1, 11.5
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from agents.portfolio_manager import (
    _build_trade_plan_from_candidate,
    _maybe_create_trade_plan,
)
from utils.candidate_registry import CandidateRegistry, CandidateState
from utils.trade_plan_registry import (
    PlanState,
    TradePlanRegistry,
    TradePlanRegistryError,
)

CYCLE_ID = "cycle-001"
PROFILE_ID = "aggressive"


# ---------------------------------------------------------------------------
# Schema setup (in-memory SQLite)
# ---------------------------------------------------------------------------


def _create_all_tables(engine):
    """Create the candidate + trade plan tables needed by these tests."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pm_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                cycle_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                setup_type TEXT NOT NULL,
                geometry_name TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_price REAL NOT NULL,
                target_price REAL NOT NULL,
                risk_reward REAL NOT NULL,
                trigger TEXT,
                invalidation_basis TEXT,
                target_basis TEXT,
                source_signal_id TEXT NOT NULL,
                signal_snapshot_json TEXT NOT NULL,
                state TEXT NOT NULL,
                integrity_hash TEXT NOT NULL,
                execution_key TEXT,
                reserved_at TEXT,
                created_at TEXT,
                expires_at TEXT NOT NULL,
                context_snapshot_json TEXT,
                benchmark_mapping_json TEXT,
                rejection_reason TEXT,
                candidate_lineage_id TEXT,
                candidate_type TEXT DEFAULT 'intraday'
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
    _create_all_tables(eng)
    return eng


@pytest.fixture
def registry(engine):
    """A real CandidateRegistry bound to the in-memory engine."""
    return CandidateRegistry(engine, CYCLE_ID, PROFILE_ID)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _insert_candidate(
    engine,
    *,
    candidate_id: str = "cand-001",
    symbol: str = "TSLA",
    direction: str = "BUY",
    setup_type: str = "momentum_fade",
    entry_price: float = 250.0,
    stop_price: float = 245.0,
    target_price: float = 260.0,
    risk_reward: float = 2.0,
    state: str = "registered",
):
    """Insert a REGISTERED candidate row so registry.get() returns a real record."""
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO pm_candidates (
                    candidate_id, cycle_id, profile_id, symbol, direction,
                    setup_type, geometry_name, entry_price, stop_price,
                    target_price, risk_reward, trigger, source_signal_id,
                    signal_snapshot_json, state, integrity_hash,
                    created_at, expires_at, candidate_type
                ) VALUES (
                    :candidate_id, :cycle_id, :profile_id, :symbol, :direction,
                    :setup_type, 'standard', :entry_price, :stop_price,
                    :target_price, :risk_reward, 'Momentum fade near VWAP',
                    'sig-001', :snapshot, :state, 'hash-001',
                    :created_at, :expires_at, 'intraday'
                )
            """),
            {
                "candidate_id": candidate_id,
                "cycle_id": CYCLE_ID,
                "profile_id": PROFILE_ID,
                "symbol": symbol,
                "direction": direction,
                "setup_type": setup_type,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "risk_reward": risk_reward,
                "snapshot": json.dumps({"symbol": symbol}),
                "state": state,
                "created_at": now.isoformat(),
                "expires_at": (now + timedelta(minutes=30)).isoformat(),
            },
        )
    return candidate_id


def _make_decision(candidate_id: str = "cand-001", rationale: str = "Approved for execution"):
    """A CandidateDecision-like object (only candidate_id/rationale are read)."""
    decision = MagicMock()
    decision.candidate_id = candidate_id
    decision.rationale = rationale
    decision.risk_multiplier = 1.0
    return decision


def _candidate_state(engine, candidate_id: str) -> str:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT state FROM pm_candidates WHERE candidate_id = :cid"),
            {"cid": candidate_id},
        ).scalar()


def _plan_ids(engine) -> list[str]:
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(
            text("SELECT plan_id FROM trade_plans ORDER BY created_at")
        ).fetchall()]


def _candidate_events(engine, candidate_id: str, event_type: str) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT * FROM pm_candidate_events
                WHERE candidate_id = :cid AND event_type = :et
            """),
            {"cid": candidate_id, "et": event_type},
        ).mappings().all()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Tests: mode="enabled" — plan created, immediate execution skipped
# ---------------------------------------------------------------------------


class TestModeEnabled:
    """TRIGGERED_PLAN_MODE=enabled: PM acceptance creates a plan and skips
    immediate execution (Req 7.1, 7.2, 7.5, 7.6, 7.7)."""

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled")
    def test_plan_created_on_pm_acceptance(self, engine, registry):
        _insert_candidate(engine)
        executed: list = []

        skip = _maybe_create_trade_plan(
            engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, executed,
        )

        assert skip is True
        plan_ids = _plan_ids(engine)
        assert len(plan_ids) == 1

        plan = TradePlanRegistry(engine).get_plan(plan_ids[0])
        assert plan.state == PlanState.PLANNED
        assert plan.symbol == "TSLA"
        assert plan.direction == "BUY"
        assert plan.candidate_id == "cand-001"
        assert plan.cycle_id == CYCLE_ID
        assert plan.profile_id == PROFILE_ID

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled")
    def test_immediate_execution_skipped(self, engine, registry):
        """The helper returns True, which the acceptance loop consumes with
        ``continue`` — so execute_candidate_pipeline never runs."""
        _insert_candidate(engine)
        executed: list = []

        skip = _maybe_create_trade_plan(
            engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, executed,
        )

        assert skip is True
        # Candidate was never reserved for execution (Req 11.2: registry untouched)
        assert _candidate_state(engine, "cand-001") == CandidateState.REGISTERED.value

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled")
    def test_candidate_gets_plan_created_outcome(self, engine, registry):
        """The accepted candidate is reported with outcome='plan_created' and
        executed=False (Req 7.3)."""
        _insert_candidate(engine)
        executed: list = []

        _maybe_create_trade_plan(
            engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, executed,
        )

        assert len(executed) == 1
        entry = executed[0]
        assert entry["outcome"] == "plan_created"
        assert entry["executed"] is False
        assert entry["symbol"] == "TSLA"
        assert entry["action"] == "BUY"
        assert entry["candidate_id"] == "cand-001"
        assert entry["plan_id"] == _plan_ids(engine)[0]

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled")
    def test_decision_log_event_emitted(self, engine, registry):
        """A plan_created candidate event carries the dashboard message and zone
        bounds (Req 7.7)."""
        _insert_candidate(engine)
        executed: list = []

        _maybe_create_trade_plan(
            engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, executed,
        )

        events = _candidate_events(engine, "cand-001", "plan_created")
        assert len(events) == 1
        payload = json.loads(events[0]["event_data"])
        assert payload["plan_id"] == _plan_ids(engine)[0]
        assert payload["mode"] == "enabled"
        assert "plan created, watching for trigger" in payload["message"]
        assert payload["entry_zone_lower"] == 249.0
        assert payload["entry_zone_upper"] == 250.0

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled")
    def test_plan_created_event_in_trade_plan_events(self, engine, registry):
        _insert_candidate(engine)
        _maybe_create_trade_plan(
            engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, [],
        )

        with engine.connect() as conn:
            events = conn.execute(text(
                "SELECT * FROM trade_plan_events WHERE plan_id = :pid"
            ), {"pid": _plan_ids(engine)[0]}).mappings().all()

        assert len(events) == 1
        assert events[0]["event_type"] == "plan_created"
        assert events[0]["to_state"] == "planned"

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled")
    def test_candidate_finalizes_to_not_selected(self, engine, registry):
        """A plan-created candidate stays REGISTERED and is swept to a terminal
        NOT_SELECTED state by the existing finalize_cycle() (Req 11.4, 11.5)."""
        _insert_candidate(engine)

        skip = _maybe_create_trade_plan(
            engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, [],
        )
        assert skip is True

        assignments = registry.finalize_cycle()
        assert assignments["cand-001"] == CandidateState.NOT_SELECTED
        assert _candidate_state(engine, "cand-001") == CandidateState.NOT_SELECTED.value

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled")
    def test_missing_candidate_record_is_noop(self, engine, registry):
        """No candidate row → no plan, and execution is not skipped (fail-open)."""
        executed: list = []

        skip = _maybe_create_trade_plan(
            engine, registry, _make_decision("nonexistent"), PROFILE_ID, CYCLE_ID, executed,
        )

        assert skip is False
        assert _plan_ids(engine) == []
        assert executed == []


# ---------------------------------------------------------------------------
# Tests: mode="observe" — plan created AND execution proceeds
# ---------------------------------------------------------------------------


class TestModeObserve:
    """TRIGGERED_PLAN_MODE=observe: plans are created for telemetry only and
    nothing about the execution flow changes (Req 0.3, 10.6)."""

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "observe")
    def test_plan_created_and_execution_proceeds(self, engine, registry):
        _insert_candidate(engine)
        executed: list = []

        skip = _maybe_create_trade_plan(
            engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, executed,
        )

        # Plan persisted...
        plan_ids = _plan_ids(engine)
        assert len(plan_ids) == 1
        assert TradePlanRegistry(engine).get_plan(plan_ids[0]).state == PlanState.PLANNED
        # ...but the pipeline is NOT skipped.
        assert skip is False

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "observe")
    def test_no_candidate_state_or_outcome_change(self, engine, registry):
        """Observe mode must not touch candidate state or append an outcome —
        the candidate goes through the normal pipeline."""
        _insert_candidate(engine)
        executed: list = []

        _maybe_create_trade_plan(
            engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, executed,
        )

        assert executed == []
        assert _candidate_state(engine, "cand-001") == CandidateState.REGISTERED.value
        # No plan_created decision-log event in observe mode
        assert _candidate_events(engine, "cand-001", "plan_created") == []

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "observe")
    def test_short_candidate_plan_persisted(self, engine, registry):
        _insert_candidate(
            engine, symbol="AAPL", direction="SHORT",
            entry_price=200.0, stop_price=205.0, target_price=190.0,
        )

        _maybe_create_trade_plan(
            engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, [],
        )

        plan = TradePlanRegistry(engine).get_plan(_plan_ids(engine)[0])
        assert plan.symbol == "AAPL"
        assert plan.direction == "SHORT"
        # SHORT: lower = reference, upper = reference + (stop - reference) * 0.20
        assert plan.entry_zone_lower == 200.0
        assert plan.entry_zone_upper == 201.0

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "observe")
    def test_plan_creation_failure_does_not_block_execution(self, engine, registry):
        """Fail-open: a create_plan error leaves the pipeline untouched."""
        _insert_candidate(engine)
        executed: list = []

        with patch.object(
            TradePlanRegistry, "create_plan",
            side_effect=TradePlanRegistryError("DB full"),
        ):
            skip = _maybe_create_trade_plan(
                engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, executed,
            )

        assert skip is False
        assert _plan_ids(engine) == []
        assert executed == []
        assert _candidate_state(engine, "cand-001") == CandidateState.REGISTERED.value


# ---------------------------------------------------------------------------
# Tests: mode="disabled" — no plan creation at all
# ---------------------------------------------------------------------------


class TestModeDisabled:
    """TRIGGERED_PLAN_MODE=disabled: existing pipeline runs unchanged (Req 0.2, 11.1)."""

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "disabled")
    def test_no_plan_created(self, engine, registry):
        _insert_candidate(engine)
        executed: list = []

        skip = _maybe_create_trade_plan(
            engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, executed,
        )

        assert skip is False
        assert _plan_ids(engine) == []
        assert executed == []
        assert _candidate_state(engine, "cand-001") == CandidateState.REGISTERED.value

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "disabled")
    def test_no_plan_events_written(self, engine, registry):
        _insert_candidate(engine)
        _maybe_create_trade_plan(
            engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, [],
        )

        with engine.connect() as conn:
            assert conn.execute(text("SELECT COUNT(*) FROM trade_plan_events")).scalar() == 0
        assert _candidate_events(engine, "cand-001", "plan_created") == []


# ---------------------------------------------------------------------------
# Tests: fail-open on plan creation failure
# ---------------------------------------------------------------------------


class TestFailOpen:
    """Plan creation failures must never block the candidate pipeline."""

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled")
    def test_create_plan_error_does_not_block(self, engine, registry):
        _insert_candidate(engine)
        executed: list = []

        with patch.object(TradePlanRegistry, "create_plan", side_effect=Exception("disk full")):
            skip = _maybe_create_trade_plan(
                engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, executed,
            )

        # skip=False → the loop falls through to execute_candidate_pipeline
        assert skip is False
        assert executed == []

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled")
    def test_entry_zone_derivation_error_does_not_block(self, engine, registry):
        _insert_candidate(engine)
        executed: list = []

        with patch(
            "utils.entry_zone.derive_entry_zone",
            side_effect=ValueError("bad geometry"),
        ):
            skip = _maybe_create_trade_plan(
                engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, executed,
            )

        assert skip is False
        assert _plan_ids(engine) == []
        assert executed == []

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled")
    def test_duplicate_expiry_error_does_not_block(self, engine, registry):
        _insert_candidate(engine)
        executed: list = []

        with patch.object(
            TradePlanRegistry, "expire_duplicate_plans",
            side_effect=TradePlanRegistryError("locked"),
        ):
            skip = _maybe_create_trade_plan(
                engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, executed,
            )

        assert skip is False
        assert _plan_ids(engine) == []


# ---------------------------------------------------------------------------
# Tests: deduplication (Req 8.7)
# ---------------------------------------------------------------------------


class TestDeduplication:
    """A new plan for the same (profile, symbol, direction, setup_type) expires
    the existing active plan with reason='superseded'."""

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled")
    def test_existing_active_plan_expired_on_new_creation(self, engine, registry):
        _insert_candidate(engine, candidate_id="cand-001")
        _maybe_create_trade_plan(
            engine, registry, _make_decision("cand-001"), PROFILE_ID, CYCLE_ID, [],
        )
        plan_id_1 = _plan_ids(engine)[0]

        _insert_candidate(engine, candidate_id="cand-002")
        _maybe_create_trade_plan(
            engine, registry, _make_decision("cand-002"), PROFILE_ID, CYCLE_ID, [],
        )
        plan_id_2 = [p for p in _plan_ids(engine) if p != plan_id_1][0]

        plan_registry = TradePlanRegistry(engine)
        assert plan_registry.get_plan(plan_id_1).state == PlanState.EXPIRED
        assert plan_registry.get_plan(plan_id_2).state == PlanState.PLANNED

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled")
    def test_superseded_reason_recorded_in_plan_event(self, engine, registry):
        _insert_candidate(engine, candidate_id="cand-001")
        _maybe_create_trade_plan(
            engine, registry, _make_decision("cand-001"), PROFILE_ID, CYCLE_ID, [],
        )
        plan_id_1 = _plan_ids(engine)[0]

        _insert_candidate(engine, candidate_id="cand-002")
        _maybe_create_trade_plan(
            engine, registry, _make_decision("cand-002"), PROFILE_ID, CYCLE_ID, [],
        )

        with engine.connect() as conn:
            events = conn.execute(text("""
                SELECT * FROM trade_plan_events
                WHERE plan_id = :pid AND to_state = 'expired'
            """), {"pid": plan_id_1}).mappings().all()

        assert len(events) == 1
        payload = json.loads(events[0]["event_data"]) if events[0]["event_data"] else {}
        assert payload.get("reason") == "superseded"

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled")
    def test_different_symbol_not_expired(self, engine, registry):
        _insert_candidate(engine, candidate_id="cand-001", symbol="TSLA")
        _maybe_create_trade_plan(
            engine, registry, _make_decision("cand-001"), PROFILE_ID, CYCLE_ID, [],
        )
        plan_id_1 = _plan_ids(engine)[0]

        _insert_candidate(engine, candidate_id="cand-002", symbol="AAPL")
        _maybe_create_trade_plan(
            engine, registry, _make_decision("cand-002"), PROFILE_ID, CYCLE_ID, [],
        )
        plan_id_2 = [p for p in _plan_ids(engine) if p != plan_id_1][0]

        plan_registry = TradePlanRegistry(engine)
        assert plan_registry.get_plan(plan_id_1).state == PlanState.PLANNED
        assert plan_registry.get_plan(plan_id_2).state == PlanState.PLANNED

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled")
    def test_different_direction_not_expired(self, engine, registry):
        _insert_candidate(engine, candidate_id="cand-001", direction="BUY")
        _maybe_create_trade_plan(
            engine, registry, _make_decision("cand-001"), PROFILE_ID, CYCLE_ID, [],
        )
        plan_id_1 = _plan_ids(engine)[0]

        _insert_candidate(
            engine, candidate_id="cand-002", direction="SHORT",
            entry_price=250.0, stop_price=255.0, target_price=240.0,
        )
        _maybe_create_trade_plan(
            engine, registry, _make_decision("cand-002"), PROFILE_ID, CYCLE_ID, [],
        )

        plan_registry = TradePlanRegistry(engine)
        assert plan_registry.get_plan(plan_id_1).state == PlanState.PLANNED


# ---------------------------------------------------------------------------
# Tests: plan field derivation (Req 1.2, 1.3, 1.7, 2.9)
# ---------------------------------------------------------------------------


class TestPlanFieldDerivation:
    """_build_trade_plan_from_candidate populates the plan deterministically."""

    def test_long_entry_zone_has_no_tolerance_baked_in(self, engine, registry):
        _insert_candidate(engine, entry_price=100.0, stop_price=95.0, direction="BUY")
        record = registry.get("cand-001")

        plan, ez = _build_trade_plan_from_candidate(
            record, _make_decision(), PROFILE_ID, CYCLE_ID,
        )

        # LONG: upper = reference, lower = reference - (reference-stop)*0.20
        assert plan.entry_zone_upper == 100.0
        assert plan.entry_zone_lower == 99.0
        assert plan.entry_reference == 100.0
        # Tolerance lives on the zone for evaluation time, not in the bounds
        assert float(ez.upper) == plan.entry_zone_upper
        assert float(ez.lower) == plan.entry_zone_lower

    def test_short_entry_zone(self, engine, registry):
        _insert_candidate(
            engine, direction="SHORT", entry_price=780.0,
            stop_price=790.0, target_price=760.0,
        )
        record = registry.get("cand-001")

        plan, _ = _build_trade_plan_from_candidate(
            record, _make_decision(), PROFILE_ID, CYCLE_ID,
        )

        assert plan.entry_zone_lower == 780.0
        assert plan.entry_zone_upper == 782.0

    def test_plan_references_correct_candidate_id(self, engine, registry):
        _insert_candidate(engine, candidate_id="my-candidate-uuid")
        record = registry.get("my-candidate-uuid")

        plan, _ = _build_trade_plan_from_candidate(
            record, _make_decision("my-candidate-uuid"), PROFILE_ID, CYCLE_ID,
        )

        assert plan.candidate_id == "my-candidate-uuid"
        assert plan.source_signal_id == "sig-001"

    def test_created_at_and_expires_at(self, engine, registry):
        _insert_candidate(engine)
        record = registry.get("cand-001")

        before = datetime.now(timezone.utc)
        plan, _ = _build_trade_plan_from_candidate(
            record, _make_decision(), PROFILE_ID, CYCLE_ID,
        )
        after = datetime.now(timezone.utc)

        from utils.gate_config import PLAN_DEFAULT_EXPIRATION_MINUTES

        assert before <= plan.created_at <= after
        expected = plan.created_at + timedelta(minutes=PLAN_DEFAULT_EXPIRATION_MINUTES)
        assert abs((plan.expires_at - expected).total_seconds()) < 1

    def test_persisted_timestamps_round_trip(self, engine, registry):
        """created_at/expires_at survive the DB round-trip (Req 1.6)."""
        with patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled"):
            _insert_candidate(engine)
            _maybe_create_trade_plan(
                engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, [],
            )

        from utils.gate_config import PLAN_DEFAULT_EXPIRATION_MINUTES

        plan = TradePlanRegistry(engine).get_plan(_plan_ids(engine)[0])
        delta = (plan.expires_at - plan.created_at).total_seconds()
        assert abs(delta - PLAN_DEFAULT_EXPIRATION_MINUTES * 60) < 1

    def test_trigger_and_invalidation_payloads(self, engine, registry):
        _insert_candidate(engine, entry_price=100.0, stop_price=95.0)
        record = registry.get("cand-001")

        plan, _ = _build_trade_plan_from_candidate(
            record, _make_decision(), PROFILE_ID, CYCLE_ID,
        )

        assert plan.trigger_type == "price_in_zone"
        assert plan.trigger_confirmation_required is True
        trigger = json.loads(plan.trigger_condition_json)
        assert trigger["type"] == "price_in_zone"
        assert trigger["entry_zone_upper"] == 100.0
        assert trigger["entry_zone_lower"] == 99.0
        invalidation = json.loads(plan.invalidation_logic_json)
        assert invalidation["type"] == "stop_breach"
        assert invalidation["stop_price"] == 95.0

    def test_provenance_fields_carried_from_candidate(self, engine, registry):
        _insert_candidate(engine)
        record = registry.get("cand-001")

        plan, _ = _build_trade_plan_from_candidate(
            record, _make_decision(rationale="High conviction fade"), PROFILE_ID, CYCLE_ID,
        )

        assert plan.analyst_reasoning == "Momentum fade near VWAP"
        assert plan.pm_rationale == "High conviction fade"
        assert plan.state == PlanState.PLANNED
        assert plan.triggered_at is None
        assert plan.executed_at is None
        assert plan.missed_at is None

    def test_geometry_carried_from_candidate(self, engine, registry):
        _insert_candidate(
            engine, entry_price=250.0, stop_price=245.0,
            target_price=260.0, risk_reward=2.0,
        )
        record = registry.get("cand-001")

        plan, _ = _build_trade_plan_from_candidate(
            record, _make_decision(), PROFILE_ID, CYCLE_ID,
        )

        assert plan.stop_price == 245.0
        assert plan.target_price == 260.0
        assert plan.risk_reward == 2.0
        assert plan.setup_type == "momentum_fade"
        assert plan.geometry_name == "standard"

# ---------------------------------------------------------------------------
# Tests: pre-creation staleness guard (price already past target)
# ---------------------------------------------------------------------------


def _cache(symbol: str, price: float, age_seconds: float = 5.0) -> dict:
    """Build a shared-quote-cache payload: {symbol: (timestamp, price)}."""
    import time as _time

    return {symbol: (_time.time() - age_seconds, price)}


def _patch_cache(cache: dict):
    """Patch the shared quote cache AND the provider entry point.

    ``get_batch_quotes`` is patched with a MagicMock so tests can assert the
    staleness guard never consumes provider quota.
    """
    return (
        patch("agents.price_monitor._quote_cache", cache),
        patch("agents.price_monitor.get_batch_quotes", MagicMock()),
    )


class TestPreCreationStalenessGuard:
    """A candidate whose cached price has already blown past its target must not
    produce a plan — that only churns PLANNED → WATCHING → MISSED.

    The guard is cache-only (no provider calls) and fail-open.
    """

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled")
    def test_long_price_past_target_rejects_plan(self, engine, registry):
        _insert_candidate(
            engine, symbol="TSLA", direction="BUY",
            entry_price=250.0, stop_price=245.0, target_price=260.0,
        )
        executed: list = []

        cache_patch, provider_patch = _patch_cache(_cache("TSLA", 262.5))
        with cache_patch, provider_patch as provider:
            skip = _maybe_create_trade_plan(
                engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, executed,
            )

        assert skip is False
        assert _plan_ids(engine) == []
        assert executed == []
        provider.assert_not_called()

        events = _candidate_events(engine, "cand-001", "plan_rejected_at_creation")
        assert len(events) == 1
        payload = json.loads(events[0]["event_data"])
        assert payload["reason"] == "price_already_past_target"
        assert payload["symbol"] == "TSLA"
        assert payload["direction"] == "BUY"
        assert payload["observed_price"] == 262.5
        assert payload["target_price"] == 260.0
        assert payload["entry_zone_upper"] == 250.0
        assert payload["entry_zone_lower"] == 249.0
        assert payload["quote_age_seconds"] >= 0

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled")
    def test_short_price_past_target_rejects_plan(self, engine, registry):
        _insert_candidate(
            engine, symbol="AAPL", direction="SHORT",
            entry_price=200.0, stop_price=205.0, target_price=190.0,
        )
        executed: list = []

        cache_patch, provider_patch = _patch_cache(_cache("AAPL", 188.0))
        with cache_patch, provider_patch as provider:
            skip = _maybe_create_trade_plan(
                engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, executed,
            )

        assert skip is False
        assert _plan_ids(engine) == []
        assert executed == []
        provider.assert_not_called()

        payload = json.loads(
            _candidate_events(engine, "cand-001", "plan_rejected_at_creation")[0]["event_data"]
        )
        assert payload["reason"] == "price_already_past_target"
        assert payload["direction"] == "SHORT"
        assert payload["observed_price"] == 188.0
        assert payload["target_price"] == 190.0

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled")
    def test_price_not_past_target_creates_plan(self, engine, registry):
        """Regression guard: a live setup still becomes a plan."""
        _insert_candidate(
            engine, symbol="TSLA", direction="BUY",
            entry_price=250.0, stop_price=245.0, target_price=260.0,
        )
        executed: list = []

        cache_patch, provider_patch = _patch_cache(_cache("TSLA", 250.4))
        with cache_patch, provider_patch as provider:
            skip = _maybe_create_trade_plan(
                engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, executed,
            )

        assert skip is True
        assert len(_plan_ids(engine)) == 1
        assert _candidate_events(engine, "cand-001", "plan_rejected_at_creation") == []
        provider.assert_not_called()

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled")
    def test_no_cached_quote_creates_plan(self, engine, registry):
        """Fail-open: absence of a cached quote must not block plan creation."""
        _insert_candidate(engine, symbol="TSLA", direction="BUY", target_price=260.0)
        executed: list = []

        cache_patch, provider_patch = _patch_cache({})
        with cache_patch, provider_patch as provider:
            skip = _maybe_create_trade_plan(
                engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, executed,
            )

        assert skip is True
        assert len(_plan_ids(engine)) == 1
        assert _candidate_events(engine, "cand-001", "plan_rejected_at_creation") == []
        provider.assert_not_called()

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled")
    def test_staleness_check_exception_creates_plan(self, engine, registry):
        """Fail-open: a raising staleness check falls through to plan creation."""
        _insert_candidate(engine, symbol="TSLA", direction="BUY", target_price=260.0)
        executed: list = []

        with patch(
            "agents.portfolio_manager._get_cached_quote_for_staleness",
            side_effect=RuntimeError("cache exploded"),
        ):
            skip = _maybe_create_trade_plan(
                engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, executed,
            )

        assert skip is True
        assert len(_plan_ids(engine)) == 1
        assert _candidate_events(engine, "cand-001", "plan_rejected_at_creation") == []

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "observe")
    def test_observe_mode_also_rejects_stale_plan(self, engine, registry):
        """Observe mode still skips the churn, and its return value (False) keeps
        execution behavior identical."""
        _insert_candidate(
            engine, symbol="TSLA", direction="BUY",
            entry_price=250.0, stop_price=245.0, target_price=260.0,
        )
        executed: list = []

        cache_patch, provider_patch = _patch_cache(_cache("TSLA", 265.0))
        with cache_patch, provider_patch:
            skip = _maybe_create_trade_plan(
                engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, executed,
            )

        assert skip is False
        assert _plan_ids(engine) == []
        assert executed == []
        assert _candidate_state(engine, "cand-001") == CandidateState.REGISTERED.value

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "disabled")
    def test_disabled_mode_never_reads_cache(self, engine, registry):
        """The existing early return means the guard is not reached at all."""
        _insert_candidate(engine, symbol="TSLA", target_price=260.0)

        cache_patch, provider_patch = _patch_cache(_cache("TSLA", 999.0))
        with cache_patch, provider_patch as provider:
            skip = _maybe_create_trade_plan(
                engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, [],
            )

        assert skip is False
        assert _plan_ids(engine) == []
        assert _candidate_events(engine, "cand-001", "plan_rejected_at_creation") == []
        provider.assert_not_called()

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled")
    def test_existing_active_plan_survives_stale_rejection(self, engine, registry):
        """A rejected-at-creation candidate must not expire the active plan it
        would have superseded."""
        _insert_candidate(engine, candidate_id="cand-001", symbol="TSLA", target_price=260.0)
        cache_patch, provider_patch = _patch_cache(_cache("TSLA", 250.4))
        with cache_patch, provider_patch:
            _maybe_create_trade_plan(
                engine, registry, _make_decision("cand-001"), PROFILE_ID, CYCLE_ID, [],
            )
        plan_id_1 = _plan_ids(engine)[0]

        _insert_candidate(engine, candidate_id="cand-002", symbol="TSLA", target_price=260.0)
        cache_patch, provider_patch = _patch_cache(_cache("TSLA", 261.0))
        with cache_patch, provider_patch:
            skip = _maybe_create_trade_plan(
                engine, registry, _make_decision("cand-002"), PROFILE_ID, CYCLE_ID, [],
            )

        assert skip is False
        assert _plan_ids(engine) == [plan_id_1]
        assert TradePlanRegistry(engine).get_plan(plan_id_1).state == PlanState.PLANNED


class TestProductionStalenessRegressions:
    """Real cases observed in production on the day the guard was added: plans
    were created and marked MISSED (reason_for_miss='price_past_target') on the
    very next monitor tick."""

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled")
    def test_msft_buy_price_past_target(self, engine, registry):
        """MSFT BUY: entry zone 450.56–450.74, target 452.09, price 457.86."""
        _insert_candidate(
            engine, symbol="MSFT", direction="BUY",
            entry_price=450.74, stop_price=449.84, target_price=452.09,
        )
        executed: list = []

        cache_patch, provider_patch = _patch_cache(_cache("MSFT", 457.86))
        with cache_patch, provider_patch as provider:
            skip = _maybe_create_trade_plan(
                engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, executed,
            )

        assert skip is False
        assert _plan_ids(engine) == []
        provider.assert_not_called()

        payload = json.loads(
            _candidate_events(engine, "cand-001", "plan_rejected_at_creation")[0]["event_data"]
        )
        assert payload["reason"] == "price_already_past_target"
        assert payload["observed_price"] == 457.86
        assert payload["target_price"] == 452.09

    @patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled")
    def test_mu_short_price_past_target(self, engine, registry):
        """MU SHORT: entry zone 863.28–863.63, target 860.69, price 840.69."""
        _insert_candidate(
            engine, symbol="MU", direction="SHORT",
            entry_price=863.28, stop_price=865.0, target_price=860.69,
        )
        executed: list = []

        cache_patch, provider_patch = _patch_cache(_cache("MU", 840.69))
        with cache_patch, provider_patch as provider:
            skip = _maybe_create_trade_plan(
                engine, registry, _make_decision(), PROFILE_ID, CYCLE_ID, executed,
            )

        assert skip is False
        assert _plan_ids(engine) == []
        provider.assert_not_called()

        payload = json.loads(
            _candidate_events(engine, "cand-001", "plan_rejected_at_creation")[0]["event_data"]
        )
        assert payload["reason"] == "price_already_past_target"
        assert payload["observed_price"] == 840.69
        assert payload["target_price"] == 860.69
        assert payload["direction"] == "SHORT"
