"""Integration tests for fast-path observe mode — zero side effects contract.

Verifies the observe-mode contract (FAST_PATH_MODE="observe"):
  - Running a full monitor tick produces fast_path_events rows
  - NO rows are created in trades, pending_orders, or watch_candidates by
    the fast path (no execution delegation happens)
  - Shadow comparison metrics can be computed against PM decisions

Also validates the shadow comparison logic (task 11.1) and rollout gate
criteria checks (task 11.3).

Requirements: 1.5, 10.8, cross-cutting acceptance test 7
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text

from db.schema import (
    init_fast_path_events_schema,
    init_fast_path_triggers_schema,
)
from utils.fast_path_monitor import FastPathMonitor
from utils.fast_path_observe import (
    ROLLOUT_GATE_CRITERIA,
    check_rollout_criteria,
    compute_shadow_comparison,
)
from utils.fast_path_registry import FastPathRegistry, TriggerRecord

# Use current time so trigger expiry (NOW + 300s) stays in the future relative
# to the registry's expires_at > now filter, which compares ISO strings.
NOW = datetime.now(timezone.utc).replace(microsecond=0)
PROFILE = "moderate"


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """In-memory SQLite with fast-path tables + the delegation target tables.

    The trades, pending_orders, and watch_candidates tables are created so we
    can assert the fast path never writes to them in observe mode.
    """
    eng = create_engine("sqlite:///:memory:")
    init_fast_path_triggers_schema(eng)
    init_fast_path_events_schema(eng)

    with eng.begin() as conn:
        # Minimal delegation-target tables — observe mode must never touch these.
        conn.execute(text(
            """
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR(10) NOT NULL,
                profile_id VARCHAR(64),
                created_at DATETIME
            )
            """
        ))
        conn.execute(text(
            """
            CREATE TABLE pending_orders (
                order_id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                state TEXT
            )
            """
        ))
        conn.execute(text(
            """
            CREATE TABLE watch_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                watch_id TEXT NOT NULL UNIQUE,
                symbol TEXT NOT NULL,
                profile_id TEXT,
                state TEXT
            )
            """
        ))
        # pm_candidates — used for shadow comparison against PM decisions.
        conn.execute(text(
            """
            CREATE TABLE pm_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL UNIQUE,
                cycle_id TEXT,
                profile_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                setup_type TEXT,
                state TEXT NOT NULL DEFAULT 'registered',
                rejection_reason TEXT,
                source_signal_id TEXT,
                created_at DATETIME
            )
            """
        ))
    return eng


def _trigger(**overrides) -> TriggerRecord:
    """Build a trigger whose target is already crossed → missed_move outcome.

    SHORT with current price below target means the target has already been
    reached, producing a deterministic missed_move (a no-execution outcome).
    """
    defaults = dict(
        trigger_id=str(uuid.uuid4()),
        symbol="TSLA",
        profile_id=PROFILE,
        direction="SHORT",
        setup_type="momentum_fade",
        trigger_type="entry_zone",
        trigger_level=351.61,
        trigger_zone_upper=352.00,
        trigger_zone_lower=350.50,
        entry_price=351.61,
        stop_price=355.00,
        target_price=348.97,
        geometry_name="short_momentum_fade",
        source_signal_id=str(uuid.uuid4()),
        source_watch_id=None,
        invalidation_basis="price above 355.00",
        target_basis="prior support",
        state="active",
        registered_at=_iso(NOW),
        expires_at=_iso(NOW + timedelta(seconds=300)),
        signal_snapshot_json='{"setup_type": "momentum_fade"}',
        context_json=None,
    )
    defaults.update(overrides)
    return TriggerRecord(**defaults)


def _count(engine, table: str) -> int:
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()


# ---------------------------------------------------------------------------
# 11.2 — Zero side effects in observe mode
# ---------------------------------------------------------------------------


def test_observe_mode_creates_fast_path_events(engine):
    """A monitor tick in observe mode records a fast_path_events row."""
    registry = FastPathRegistry(db=engine, profile_id=PROFILE)
    # Target already crossed for a SHORT (price 342 < target 348.97) → missed_move
    registry.register_trigger(_trigger())

    monitor = FastPathMonitor(engine, [PROFILE])

    with patch("utils.fast_path_monitor.FAST_PATH_MODE", "observe"), \
         patch(
             "utils.fast_path_monitor._fetch_quotes",
             return_value={"TSLA": {"price": 342.08, "age_ms": 0, "reliable": True}},
         ):
        summary = monitor.run_tick()

    assert summary["fired"] >= 1
    assert _count(engine, "fast_path_events") >= 1


def test_observe_mode_no_trades_created(engine):
    """Observe mode must NOT create any trades rows via the fast path."""
    registry = FastPathRegistry(db=engine, profile_id=PROFILE)
    registry.register_trigger(_trigger())

    monitor = FastPathMonitor(engine, [PROFILE])

    with patch("utils.fast_path_monitor.FAST_PATH_MODE", "observe"), \
         patch(
             "utils.fast_path_monitor._fetch_quotes",
             return_value={"TSLA": {"price": 342.08, "age_ms": 0, "reliable": True}},
         ):
        monitor.run_tick()

    assert _count(engine, "trades") == 0


def test_observe_mode_no_pending_orders_created(engine):
    """Observe mode must NOT create any pending_orders rows via the fast path."""
    registry = FastPathRegistry(db=engine, profile_id=PROFILE)
    registry.register_trigger(_trigger())

    monitor = FastPathMonitor(engine, [PROFILE])

    with patch("utils.fast_path_monitor.FAST_PATH_MODE", "observe"), \
         patch(
             "utils.fast_path_monitor._fetch_quotes",
             return_value={"TSLA": {"price": 342.08, "age_ms": 0, "reliable": True}},
         ):
        monitor.run_tick()

    assert _count(engine, "pending_orders") == 0


def test_observe_mode_no_watch_candidates_created(engine):
    """Observe mode must NOT create any watch_candidates rows via the fast path."""
    registry = FastPathRegistry(db=engine, profile_id=PROFILE)
    registry.register_trigger(_trigger())

    monitor = FastPathMonitor(engine, [PROFILE])

    with patch("utils.fast_path_monitor.FAST_PATH_MODE", "observe"), \
         patch(
             "utils.fast_path_monitor._fetch_quotes",
             return_value={"TSLA": {"price": 342.08, "age_ms": 0, "reliable": True}},
         ):
        monitor.run_tick()

    assert _count(engine, "watch_candidates") == 0


def test_observe_mode_no_delegation_call(engine):
    """Observe mode must NOT invoke the execution delegation path at all."""
    registry = FastPathRegistry(db=engine, profile_id=PROFILE)
    registry.register_trigger(_trigger())

    monitor = FastPathMonitor(engine, [PROFILE])

    with patch("utils.fast_path_monitor.FAST_PATH_MODE", "observe"), \
         patch(
             "utils.fast_path_monitor._fetch_quotes",
             return_value={"TSLA": {"price": 342.08, "age_ms": 0, "reliable": True}},
         ), \
         patch("utils.fast_path_monitor._delegate_execution") as mock_delegate:
        monitor.run_tick()

    # No execution delegation should ever be called in observe mode.
    mock_delegate.assert_not_called()


def test_observe_mode_full_contract(engine):
    """Full observe-mode contract: events created, zero side effects.

    Cross-cutting acceptance test 7: in observe mode, no portfolio state is
    modified while all outcomes are recorded.
    """
    registry = FastPathRegistry(db=engine, profile_id=PROFILE)
    registry.register_trigger(_trigger(symbol="TSLA"))
    registry.register_trigger(_trigger(symbol="AAPL", direction="BUY",
                                       entry_price=180.0, stop_price=175.0,
                                       target_price=190.0,
                                       trigger_zone_lower=179.0,
                                       trigger_zone_upper=181.0))

    monitor = FastPathMonitor(engine, [PROFILE])

    quotes = {
        # TSLA SHORT: price below target → missed_move
        "TSLA": {"price": 342.08, "age_ms": 0, "reliable": True},
        # AAPL BUY: price above target → missed_move
        "AAPL": {"price": 195.0, "age_ms": 0, "reliable": True},
    }

    with patch("utils.fast_path_monitor.FAST_PATH_MODE", "observe"), \
         patch("utils.fast_path_monitor._fetch_quotes", return_value=quotes):
        summary = monitor.run_tick()

    # Events ARE created
    assert _count(engine, "fast_path_events") >= 1
    assert summary["fired"] >= 1

    # Zero side effects across all delegation targets
    assert _count(engine, "trades") == 0
    assert _count(engine, "pending_orders") == 0
    assert _count(engine, "watch_candidates") == 0


# ---------------------------------------------------------------------------
# 11.1 — Shadow comparison metric
# ---------------------------------------------------------------------------


def _insert_event(engine, **overrides):
    row = {
        "event_id": str(uuid.uuid4()),
        "trigger_id": str(uuid.uuid4()),
        "source_signal_id": "sig-1",
        "symbol": "TSLA",
        "profile_id": PROFILE,
        "setup_type": "momentum_fade",
        "direction": "SHORT",
        "entry_price": 351.61,
        "stop_price": 355.0,
        "target_price": 348.97,
        "current_price": 342.08,
        "reward_to_risk": 2.0,
        "outcome_type": "missed_move",
        "outcome_reason_code": "target_already_crossed",
        "annotation_status": "annotation_pending",
        "narration": "TSLA target already crossed.",
        "narration_source": "template",
        "evaluated_at": _iso(NOW),
        "created_at": _iso(NOW),
    }
    row.update(overrides)
    cols = ", ".join(row.keys())
    binds = ", ".join(f":{k}" for k in row)
    with engine.begin() as conn:
        conn.execute(
            text(f"INSERT INTO fast_path_events ({cols}) VALUES ({binds})"), row
        )


def _insert_pm(engine, **overrides):
    row = {
        "candidate_id": str(uuid.uuid4()),
        "cycle_id": "cycle-1",
        "profile_id": PROFILE,
        "symbol": "TSLA",
        "direction": "SHORT",
        "setup_type": "momentum_fade",
        "state": "rejected",
        "rejection_reason": "target_already_exceeded",
        "source_signal_id": "sig-1",
        "created_at": _iso(NOW + timedelta(seconds=150)),
    }
    row.update(overrides)
    cols = ", ".join(row.keys())
    binds = ", ".join(f":{k}" for k in row)
    with engine.begin() as conn:
        conn.execute(
            text(f"INSERT INTO pm_candidates ({cols}) VALUES ({binds})"), row
        )


def test_shadow_comparison_agreement(engine):
    """missed_move fast-path outcome agrees with PM target-crossed rejection."""
    _insert_event(engine, outcome_type="missed_move")
    _insert_pm(engine, state="rejected", rejection_reason="target_already_exceeded")

    result = compute_shadow_comparison(
        engine, PROFILE,
        session_start=_iso(NOW - timedelta(minutes=1)),
        session_end=_iso(NOW + timedelta(minutes=10)),
    )

    assert result["total_triggers"] == 1
    assert result["outcomes_by_type"]["missed_move"] == 1
    assert result["agreement_count"] == 1
    assert result["disagreement_count"] == 0
    assert result["agreement_rate"] == 1.0


def test_shadow_comparison_timing_advantage(engine):
    """Timing advantage reflects fast path acting before the PM decision."""
    _insert_event(engine, evaluated_at=_iso(NOW))
    _insert_pm(engine, created_at=_iso(NOW + timedelta(seconds=150)))

    result = compute_shadow_comparison(
        engine, PROFILE,
        session_start=_iso(NOW - timedelta(minutes=1)),
        session_end=_iso(NOW + timedelta(minutes=10)),
    )

    # Fast path acted 150s before PM
    assert result["timing_advantage_avg_seconds"] == pytest.approx(150.0, abs=1.0)


def test_shadow_comparison_disagreement(engine):
    """trade_executed disagrees when PM rejected the same signal."""
    _insert_event(engine, outcome_type="trade_executed",
                  outcome_reason_code="all_gates_passed")
    _insert_pm(engine, state="rejected", rejection_reason="gate_rejected")

    result = compute_shadow_comparison(
        engine, PROFILE,
        session_start=_iso(NOW - timedelta(minutes=1)),
        session_end=_iso(NOW + timedelta(minutes=10)),
    )

    assert result["agreement_count"] == 0
    assert result["disagreement_count"] == 1
    assert result["agreement_rate"] == 0.0
    assert len(result["disagreements"]) == 1
    assert result["disagreements"][0]["fp_outcome"] == "trade_executed"


def test_shadow_comparison_unmatched(engine):
    """A fast-path event with no PM decision is counted as unmatched."""
    _insert_event(engine, symbol="NVDA")
    # No PM decision for NVDA

    result = compute_shadow_comparison(
        engine, PROFILE,
        session_start=_iso(NOW - timedelta(minutes=1)),
        session_end=_iso(NOW + timedelta(minutes=10)),
    )

    assert result["unmatched_count"] == 1
    assert result["agreement_count"] == 0
    assert result["disagreement_count"] == 0


def test_shadow_comparison_empty_window(engine):
    """No events in the window yields a zeroed-out result without error."""
    result = compute_shadow_comparison(
        engine, PROFILE,
        session_start=_iso(NOW),
        session_end=_iso(NOW + timedelta(minutes=1)),
    )

    assert result["total_triggers"] == 0
    assert result["agreement_rate"] == 0.0


# ---------------------------------------------------------------------------
# 11.3 — Rollout gate criteria
# ---------------------------------------------------------------------------


def test_rollout_criteria_pass():
    """All criteria pass → ready is True."""
    shadow_result = {
        "total_triggers": 50,
        "agreement_rate": 0.90,
        "disagreements": [],
    }
    result = check_rollout_criteria(shadow_result, tick_budget_compliance_rate=0.98)

    assert result["ready"] is True
    assert result["blocking_criteria"] == []


def test_rollout_criteria_low_agreement_blocks():
    """Agreement rate below 80% blocks rollout."""
    shadow_result = {
        "total_triggers": 50,
        "agreement_rate": 0.60,
        "disagreements": [],
    }
    result = check_rollout_criteria(shadow_result, tick_budget_compliance_rate=0.98)

    assert result["ready"] is False
    assert "agreement_rate" in result["blocking_criteria"]


def test_rollout_criteria_false_execution_blocks():
    """A false trade_executed disagreement blocks rollout."""
    shadow_result = {
        "total_triggers": 50,
        "agreement_rate": 0.90,
        "disagreements": [{"fp_outcome": "trade_executed", "symbol": "TSLA"}],
    }
    result = check_rollout_criteria(shadow_result, tick_budget_compliance_rate=0.98)

    assert result["ready"] is False
    assert "no_false_trade_executed" in result["blocking_criteria"]


def test_rollout_criteria_tick_budget_blocks():
    """Tick budget compliance below 95% blocks rollout."""
    shadow_result = {
        "total_triggers": 50,
        "agreement_rate": 0.90,
        "disagreements": [],
    }
    result = check_rollout_criteria(shadow_result, tick_budget_compliance_rate=0.80)

    assert result["ready"] is False
    assert "tick_budget_compliance" in result["blocking_criteria"]


def test_rollout_criteria_thresholds_documented():
    """The documented threshold constants match the spec."""
    assert ROLLOUT_GATE_CRITERIA["min_observe_sessions"] == 1
    assert ROLLOUT_GATE_CRITERIA["min_agreement_rate_conservative_outcomes"] == 0.80
    assert ROLLOUT_GATE_CRITERIA["max_false_trade_executed"] == 0
    assert ROLLOUT_GATE_CRITERIA["min_tick_budget_compliance_rate"] == 0.95
