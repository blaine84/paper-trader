"""Integration tests for the Realtime Watch Maturity feature.

Exercises the full flow using an in-memory SQLite database with the real
setup watch schema: bridge evaluation, promotion convergence, missed-move
detection, pending-order guard, draft geometry, market-data reliability,
feature flags, evidence packages, source provenance, and entry zone tightening.

Requirements: 1.1-1.9, 4.1-4.9, 5.1-5.8, 7.1-7.8, 9.2-9.4, 12.1-12.6
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from db.schema import init_pending_order_schema, init_setup_watch_schema
from utils.pending_order_time import now_utc, to_iso
from utils.setup_watch_registry import (
    ACTIVE_STATES,
    PERMITTED_TRANSITIONS,
    TERMINAL_STATES,
    SetupWatch,
    SetupWatchRegistry,
    SetupWatchRegistryError,
    WatchState,
)

NOW = datetime(2026, 9, 10, 14, 30, 0, tzinfo=timezone.utc)
FUTURE = NOW + timedelta(hours=24)
PROFILE = "moderate"
CYCLE = "cycle_001"
CYCLE_2 = "cycle_002"


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


def _create_pm_candidates_table(engine):
    """Minimal pm_candidates table for candidate builder integration."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
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
                    rejection_reason_code TEXT,
                    candidate_lineage_id TEXT,
                    candidate_type TEXT DEFAULT 'intraday',
                    holding_horizon INTEGER,
                    normalized_setup_type TEXT
                )
                """
            )
        )


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    init_setup_watch_schema(eng)
    init_pending_order_schema(eng)
    _create_pm_candidates_table(eng)
    return eng


@pytest.fixture
def registry(engine):
    return SetupWatchRegistry(engine)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _maturation_price_zone(low: float, high: float, weight: float = 1.0) -> dict:
    return {"type": "price_zone", "params": {"low": low, "high": high}, "weight": weight}


def _maturation_support_hold(level: float, tolerance_pct: float = 1.0, weight: float = 1.0) -> dict:
    return {"type": "support_hold", "params": {"level": level, "tolerance_pct": tolerance_pct}, "weight": weight}


def _invalidation_price_breach(level: float, direction: str = "below") -> dict:
    return {"type": "price_breach", "params": {"level": level, "direction": direction}}


def _insert_watch(
    engine,
    *,
    watch_id=None,
    profile_id=PROFILE,
    symbol="AAPL",
    side="BUY",
    setup_type="support_bounce_swing",
    state="watching",
    created_at=None,
    expires_at=None,
    observed_cycles=0,
    maturity_score=0.0,
    promoted_cycle_id=None,
    maturation_conditions=None,
    invalidation_conditions=None,
    ready_at=None,
    ready_reference_price=None,
    source_cycle_id=None,
    source_type="analyst",
    source_id="signal_123",
    draft_geometry_json=None,
    entry_zone_json=None,
    last_evaluation_json=None,
    terminal_reason=None,
):
    """Directly insert a watch row for test setup."""
    if watch_id is None:
        watch_id = str(uuid.uuid4())
    if created_at is None:
        created_at = NOW
    if expires_at is None:
        expires_at = FUTURE
    if maturation_conditions is None:
        maturation_conditions = [
            _maturation_price_zone(148.0, 152.0),
            _maturation_support_hold(147.0, tolerance_pct=1.0),
        ]
    if invalidation_conditions is None:
        invalidation_conditions = [_invalidation_price_breach(140.0)]
    if source_cycle_id is None:
        source_cycle_id = CYCLE

    now_iso = to_iso(created_at)
    exp_iso = to_iso(expires_at)
    ready_iso = to_iso(ready_at) if ready_at else None

    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO setup_watches "
                "(watch_id, profile_id, symbol, side, setup_type, state, "
                " thesis, source_type, source_id, source_cycle_id, "
                " maturation_conditions_json, invalidation_conditions_json, "
                " last_evaluation_json, entry_zone_json, draft_geometry_json, "
                " maturity_score, created_at, updated_at, expires_at, "
                " observed_cycles, promoted_cycle_id, ready_at, "
                " ready_reference_price, integrity_hash, state_changed_at, "
                " terminal_reason) "
                "VALUES "
                "(:wid, :pid, :sym, :side, :stype, :state, "
                " :thesis, :src_type, :src_id, :cycle_id, "
                " :mat_json, :inv_json, "
                " :last_eval, :entry_zone, :draft_geom, "
                " :score, :now, :now, :exp, "
                " :cycles, :promoted_cycle, :ready_at, "
                " :ready_price, :hash, :state_changed_at, "
                " :terminal_reason)"
            ),
            {
                "wid": watch_id,
                "pid": profile_id,
                "sym": symbol,
                "side": side,
                "stype": setup_type,
                "state": state,
                "thesis": "A solid technical thesis for integration testing purposes",
                "src_type": source_type,
                "src_id": source_id,
                "cycle_id": source_cycle_id,
                "mat_json": json.dumps(maturation_conditions),
                "inv_json": json.dumps(invalidation_conditions),
                "last_eval": last_evaluation_json,
                "entry_zone": entry_zone_json,
                "draft_geom": draft_geometry_json,
                "score": maturity_score,
                "now": now_iso,
                "exp": exp_iso,
                "cycles": observed_cycles,
                "promoted_cycle": promoted_cycle_id,
                "ready_at": ready_iso,
                "ready_price": ready_reference_price,
                "hash": "test_hash_" + watch_id[:8],
                "state_changed_at": now_iso if state != "watching" else None,
                "terminal_reason": terminal_reason,
            },
        )
        conn.commit()
    return watch_id


def _state_of(engine, watch_id) -> str:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT state FROM setup_watches WHERE watch_id = :wid"),
            {"wid": watch_id},
        ).scalar()


def _terminal_reason_of(engine, watch_id) -> str | None:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT terminal_reason FROM setup_watches WHERE watch_id = :wid"),
            {"wid": watch_id},
        ).scalar()


def _event_types_for(engine, watch_id) -> list[str]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT event_type FROM setup_watch_events "
                "WHERE watch_id = :wid ORDER BY created_at"
            ),
            {"wid": watch_id},
        ).fetchall()
    return [r[0] for r in rows]


def _events_for(engine, watch_id) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT event_type, event_data, from_state, to_state, maturity_score "
                "FROM setup_watch_events "
                "WHERE watch_id = :wid ORDER BY created_at"
            ),
            {"wid": watch_id},
        ).fetchall()
    return [
        {
            "event_type": r[0],
            "event_data": r[1],
            "from_state": r[2],
            "to_state": r[3],
            "maturity_score": r[4],
        }
        for r in rows
    ]


def _get_watch_field(engine, watch_id, field) -> str | None:
    with engine.connect() as conn:
        return conn.execute(
            text(f"SELECT {field} FROM setup_watches WHERE watch_id = :wid"),
            {"wid": watch_id},
        ).scalar()


# ────────────────────────────────────────────────────────────────────────────
# 19.2: Full bridge flow with real DB
# ────────────────────────────────────────────────────────────────────────────


@patch("utils.setup_watch_bridge.SETUP_WATCH_REALTIME_MODE", "enabled")
def test_full_bridge_flow_condition_met_cas_transition(engine, registry):
    """19.2: approaching_level alert → evaluation → condition met → CAS
    watching→maturing → maturity_updated_by_monitor event emitted."""
    # Create a WATCHING watch with a price_zone condition [148, 152]
    watch_id = _insert_watch(
        engine,
        state="watching",
        maturity_score=0.0,
        maturation_conditions=[
            _maturation_price_zone(148.0, 152.0, weight=1.0),
        ],
    )

    # Alert with price=150 inside the zone [148, 152]
    # BUY watch, support approach from above (price > level)
    alerts = [
        {
            "type": "approaching_level",
            "symbol": "AAPL",
            "price": 150.0,
            "level_name": "support",
            "level_value": 148.0,
            "distance_pct": 1.3,
        }
    ]

    from utils.setup_watch_bridge import evaluate_alerts

    result = evaluate_alerts(engine, alerts)

    # The bridge should have evaluated the watch
    assert result.alerts_processed == 1
    assert result.watches_evaluated == 1

    # The price_zone condition (148, 152) should be met at price=150
    # → score becomes 1.0 → watching→maturing transition
    new_state = _state_of(engine, watch_id)
    assert new_state in ("maturing", "ready"), f"Expected maturing or ready, got {new_state}"

    # maturity_updated_by_monitor event should have been emitted
    events = _event_types_for(engine, watch_id)
    assert "maturity_updated_by_monitor" in events


# ────────────────────────────────────────────────────────────────────────────
# 19.3: Bridge → promotion convergence
# ────────────────────────────────────────────────────────────────────────────


@patch("utils.setup_watch_bridge.SETUP_WATCH_REALTIME_MODE", "enabled")
@patch("utils.setup_watch_manager.SETUP_WATCH_MODE", "enabled")
@patch("utils.setup_watch_manager.MARKET_DATA_RELIABILITY_MODE", "disabled")
@patch("utils.setup_watch_manager.SETUP_WATCH_MISSED_MOVE_ENABLED", False)
def test_bridge_promotes_ready_watch(engine, registry):
    """19.3: Bridge advances watch to READY, invokes shared promotion,
    candidate registered (promotion transition occurs)."""
    # Create a MATURING watch close to threshold — one evaluation tick should push to READY
    watch_id = _insert_watch(
        engine,
        state="maturing",
        maturity_score=0.5,
        observed_cycles=3,  # >= SETUP_WATCH_PROMOTION_MIN_CYCLES default (2)
        maturation_conditions=[
            _maturation_price_zone(148.0, 152.0, weight=1.0),
        ],
    )

    # Alert that triggers condition met → score >= threshold → READY
    alerts = [
        {
            "type": "approaching_level",
            "symbol": "AAPL",
            "price": 150.0,
            "level_name": "support",
            "level_value": 148.0,
            "distance_pct": 1.3,
        }
    ]

    from utils.setup_watch_bridge import evaluate_alerts

    result = evaluate_alerts(engine, alerts)

    # Watch should advance to READY, and _try_promote_ready_watch invoked
    new_state = _state_of(engine, watch_id)
    # After bridge promotion, state should be promoted or ready
    # (promotion depends on _promote_ready_watch succeeding)
    assert new_state in ("ready", "promoted"), f"Expected ready or promoted, got {new_state}"
    assert result.state_transitions >= 1


# ────────────────────────────────────────────────────────────────────────────
# 19.4: Scheduled + bridge promotion race (CAS)
# ────────────────────────────────────────────────────────────────────────────


@patch("utils.setup_watch_manager.MARKET_DATA_RELIABILITY_MODE", "disabled")
@patch("utils.setup_watch_manager.SETUP_WATCH_MISSED_MOVE_ENABLED", False)
def test_scheduled_and_bridge_promotion_race(engine, registry):
    """19.4: Both scheduled evaluator and bridge attempt promotion for same
    watch/cycle. Exactly one succeeds via CAS; loser logs warning."""
    # Create a READY watch meeting all promotion criteria
    watch_id = _insert_watch(
        engine,
        state="ready",
        maturity_score=0.9,
        observed_cycles=5,
        ready_at=NOW - timedelta(minutes=10),
        ready_reference_price=150.0,
    )

    from utils.setup_watch_manager import _promote_ready_watch

    # First promotion attempt succeeds
    watch = registry.get_watch(watch_id)
    success_1 = _promote_ready_watch(registry, watch, cycle_id=CYCLE)
    assert success_1 is True
    assert _state_of(engine, watch_id) == "promoted"

    # Second promotion attempt for same watch should fail (CAS lost)
    # Re-fetch watch (now in promoted state)
    watch_v2 = registry.get_watch(watch_id)
    # Attempting CAS from READY→PROMOTED on a watch that's already promoted fails
    success_2 = _promote_ready_watch(registry, watch_v2, cycle_id=CYCLE)
    assert success_2 is False


# ────────────────────────────────────────────────────────────────────────────
# 19.5: Missed-move end-to-end
# ────────────────────────────────────────────────────────────────────────────


def test_missed_move_end_to_end(engine, registry):
    """19.5: Watch reaches READY, price crosses target, watch transitions
    to MISSED, no candidate created."""
    # Create a READY watch with draft geometry showing target at 155.0
    draft_geom = json.dumps({"entry": "150.0", "stop": "147.0", "target": "155.0", "risk_reward": "1.67"})
    watch_id = _insert_watch(
        engine,
        state="ready",
        maturity_score=0.9,
        observed_cycles=3,
        ready_at=NOW - timedelta(minutes=5),
        ready_reference_price=150.0,
        draft_geometry_json=draft_geom,
    )

    from utils.missed_move_detector import apply_missed_move_transition, check_missed_move

    watch = registry.get_watch(watch_id)

    # Current price (156.0) exceeds target (155.0) for BUY
    result = check_missed_move(watch, 156.0)
    assert result.missed is True
    assert result.reason == "target_already_crossed"
    assert result.target_price == Decimal("155.0")

    # Apply transition
    transitioned = apply_missed_move_transition(registry, watch, result)
    assert transitioned is True
    assert _state_of(engine, watch_id) == "missed"
    assert _terminal_reason_of(engine, watch_id) == "target_already_crossed"


# ────────────────────────────────────────────────────────────────────────────
# 19.6: Pending-order target-crossed guard
# ────────────────────────────────────────────────────────────────────────────


def test_pending_order_target_crossed_guard(engine, registry):
    """19.6: Watch → promote → candidate → PM accepts → guard blocks
    pending order → watch transitions to MISSED."""
    # Create a PROMOTED watch with draft geometry target at 155.0
    draft_geom = json.dumps({"entry": "150.0", "stop": "147.0", "target": "155.0", "risk_reward": "1.67"})
    watch_id = _insert_watch(
        engine,
        state="promoted",
        maturity_score=0.9,
        observed_cycles=3,
        ready_at=NOW - timedelta(minutes=10),
        ready_reference_price=150.0,
        draft_geometry_json=draft_geom,
        promoted_cycle_id=CYCLE,
    )

    from utils.missed_move_detector import (
        apply_missed_move_transition,
        check_target_crossed_for_pending_order,
    )

    watch = registry.get_watch(watch_id)

    # Fresh price (157.0) has crossed the target (155.0)
    result = check_target_crossed_for_pending_order(watch, 157.0)
    assert result.missed is True
    assert result.reason == "target_already_crossed"

    # Apply the MISSED transition
    transitioned = apply_missed_move_transition(registry, watch, result)
    assert transitioned is True
    assert _state_of(engine, watch_id) == "missed"
    assert _terminal_reason_of(engine, watch_id) == "target_already_crossed"


# ────────────────────────────────────────────────────────────────────────────
# 19.7: Draft geometry at creation
# ────────────────────────────────────────────────────────────────────────────


@patch("utils.setup_watch_manager.SETUP_WATCH_MODE", "enabled")
def test_draft_geometry_at_creation_with_levels(engine):
    """19.7a: Signal with support + resistance → entry_zone and
    draft_geometry stored at creation time."""
    from utils.setup_watch_manager import create_setup_watch

    watch_id = create_setup_watch(
        engine,
        symbol="AAPL",
        profile_id=PROFILE,
        side="BUY",
        setup_type="support_bounce_swing",
        thesis="A solid bounce thesis for AAPL off support level with resistance target",
        source_type="analyst",
        source_id="signal_100",
        source_cycle_id=CYCLE,
        maturation_conditions=[
            _maturation_price_zone(148.0, 152.0),
            _maturation_support_hold(147.0),
        ],
        invalidation_conditions=[_invalidation_price_breach(140.0)],
        key_levels={"support": 148.0, "resistance": 155.0},
    )

    assert watch_id is not None

    # Check entry_zone was computed
    ez_json = _get_watch_field(engine, watch_id, "entry_zone_json")
    assert ez_json is not None
    entry_zone = json.loads(ez_json)
    assert "low" in entry_zone
    assert "high" in entry_zone

    # Check draft geometry was computed
    dg_json = _get_watch_field(engine, watch_id, "draft_geometry_json")
    assert dg_json is not None
    draft_geom = json.loads(dg_json)
    assert "entry" in draft_geom
    assert "stop" in draft_geom
    assert "target" in draft_geom
    assert "risk_reward" in draft_geom


@patch("utils.setup_watch_manager.SETUP_WATCH_MODE", "enabled")
def test_draft_geometry_at_creation_without_levels(engine):
    """19.7b: Signal without sufficient levels → both entry_zone and
    draft_geometry NULL."""
    from utils.setup_watch_manager import create_setup_watch

    watch_id = create_setup_watch(
        engine,
        symbol="MSFT",
        profile_id=PROFILE,
        side="BUY",
        setup_type="support_bounce_swing",
        thesis="A bounce thesis for MSFT without sufficient levels for geometry",
        source_type="analyst",
        source_id="signal_101",
        source_cycle_id=CYCLE,
        maturation_conditions=[
            _maturation_price_zone(290.0, 295.0),
            _maturation_support_hold(289.0),
        ],
        invalidation_conditions=[_invalidation_price_breach(280.0)],
        key_levels={},  # empty — no levels
    )

    assert watch_id is not None

    ez_json = _get_watch_field(engine, watch_id, "entry_zone_json")
    dg_json = _get_watch_field(engine, watch_id, "draft_geometry_json")
    assert ez_json is None
    assert dg_json is None


# ────────────────────────────────────────────────────────────────────────────
# 19.8: Market-data reliability gate
# ────────────────────────────────────────────────────────────────────────────


def test_market_data_reliability_gate_defers_then_succeeds(engine, registry):
    """19.8: Promotion deferred when check_market_data_readiness returns
    proceed=False, succeeds on next cycle when proceed=True."""
    watch_id = _insert_watch(
        engine,
        state="ready",
        maturity_score=0.9,
        observed_cycles=5,
        ready_at=NOW - timedelta(minutes=10),
        ready_reference_price=150.0,
    )

    from utils.setup_watch_manager import _promote_ready_watch

    # Mock readiness result with proceed=False
    mock_result_fail = MagicMock()
    mock_result_fail.proceed = False
    mock_result_fail.reason_codes = ("stale_quote",)
    mock_result_fail.missing_data_types = ("realtime_quote",)

    with patch("utils.setup_watch_manager.MARKET_DATA_RELIABILITY_MODE", "enabled"), \
         patch("utils.setup_watch_manager.SETUP_WATCH_REALTIME_MODE", "enabled"), \
         patch("utils.setup_watch_manager.SETUP_WATCH_MISSED_MOVE_ENABLED", False), \
         patch(
             "utils.market_data_reliability.pipeline_integration.check_market_data_readiness",
             return_value=mock_result_fail,
         ):
        watch = registry.get_watch(watch_id)
        success = _promote_ready_watch(registry, watch, cycle_id=CYCLE)
        # proceed=False → promotion deferred
        assert success is False
        assert _state_of(engine, watch_id) == "ready"

    # Now mock proceed=True and promote again
    mock_result_pass = MagicMock()
    mock_result_pass.proceed = True

    with patch("utils.setup_watch_manager.MARKET_DATA_RELIABILITY_MODE", "enabled"), \
         patch("utils.setup_watch_manager.SETUP_WATCH_REALTIME_MODE", "enabled"), \
         patch("utils.setup_watch_manager.SETUP_WATCH_MISSED_MOVE_ENABLED", False), \
         patch(
             "utils.market_data_reliability.pipeline_integration.check_market_data_readiness",
             return_value=mock_result_pass,
         ):
        watch = registry.get_watch(watch_id)
        success = _promote_ready_watch(registry, watch, cycle_id=CYCLE)
        assert success is True
        assert _state_of(engine, watch_id) == "promoted"


# ────────────────────────────────────────────────────────────────────────────
# 19.9: Feature flag — disabled mode
# ────────────────────────────────────────────────────────────────────────────


@patch("utils.setup_watch_bridge.SETUP_WATCH_REALTIME_MODE", "disabled")
def test_disabled_mode_produces_zero_db_ops(engine, registry):
    """19.9: SETUP_WATCH_REALTIME_MODE=disabled produces zero DB reads/writes
    from the bridge path."""
    watch_id = _insert_watch(engine, state="watching", maturity_score=0.0)

    alerts = [
        {
            "type": "approaching_level",
            "symbol": "AAPL",
            "price": 150.0,
            "level_name": "support",
            "level_value": 148.0,
            "distance_pct": 1.3,
        }
    ]

    from utils.setup_watch_bridge import evaluate_alerts

    result = evaluate_alerts(engine, alerts)

    # Zero-result: no processing occurred
    assert result.alerts_processed == 0
    assert result.watches_evaluated == 0
    assert result.state_transitions == 0
    assert result.missed_moves_detected == 0
    assert result.errors == 0

    # Watch state unchanged
    assert _state_of(engine, watch_id) == "watching"

    # No events emitted
    events = _event_types_for(engine, watch_id)
    assert len(events) == 0


# ────────────────────────────────────────────────────────────────────────────
# 19.10: Feature flag — observe mode
# ────────────────────────────────────────────────────────────────────────────


@patch("utils.setup_watch_bridge.SETUP_WATCH_REALTIME_MODE", "observe")
def test_observe_mode_emits_events_no_transitions(engine, registry):
    """19.10: SETUP_WATCH_REALTIME_MODE=observe emits events but zero
    state transitions from the bridge path."""
    watch_id = _insert_watch(
        engine,
        state="watching",
        maturity_score=0.0,
        maturation_conditions=[
            _maturation_price_zone(148.0, 152.0, weight=1.0),
        ],
    )

    alerts = [
        {
            "type": "approaching_level",
            "symbol": "AAPL",
            "price": 150.0,
            "level_name": "support",
            "level_value": 148.0,
            "distance_pct": 1.3,
        }
    ]

    from utils.setup_watch_bridge import evaluate_alerts

    result = evaluate_alerts(engine, alerts)

    # Evaluation happened
    assert result.watches_evaluated == 1

    # Zero state transitions in observe mode
    assert result.state_transitions == 0

    # Watch state should still be WATCHING
    assert _state_of(engine, watch_id) == "watching"

    # But events should be emitted (maturity_updated_by_monitor)
    events = _event_types_for(engine, watch_id)
    assert "maturity_updated_by_monitor" in events


# ────────────────────────────────────────────────────────────────────────────
# 19.11: MISSED only from READY/PROMOTED
# ────────────────────────────────────────────────────────────────────────────


def test_missed_only_from_ready_or_promoted(engine, registry):
    """19.11: MISSED state only reachable from READY and PROMOTED.
    WATCHING and MATURING cannot transition to MISSED."""
    # Verify PERMITTED_TRANSITIONS
    assert (WatchState.READY, WatchState.MISSED) in PERMITTED_TRANSITIONS
    assert (WatchState.PROMOTED, WatchState.MISSED) in PERMITTED_TRANSITIONS
    assert (WatchState.WATCHING, WatchState.MISSED) not in PERMITTED_TRANSITIONS
    assert (WatchState.MATURING, WatchState.MISSED) not in PERMITTED_TRANSITIONS

    # Verify with actual DB: trying to transition WATCHING→MISSED fails
    watch_id = _insert_watch(engine, state="watching", symbol="TSLA")
    with pytest.raises(SetupWatchRegistryError):
        registry.transition_state(watch_id, WatchState.WATCHING, WatchState.MISSED)

    # MATURING→MISSED also fails
    watch_id_2 = _insert_watch(engine, state="maturing", maturity_score=0.3, symbol="AMD")
    with pytest.raises(SetupWatchRegistryError):
        registry.transition_state(watch_id_2, WatchState.MATURING, WatchState.MISSED)

    # READY→MISSED succeeds
    watch_id_3 = _insert_watch(engine, state="ready", maturity_score=0.9, symbol="NVDA")
    registry.transition_state(
        watch_id_3, WatchState.READY, WatchState.MISSED,
        terminal_reason="target_already_crossed",
    )
    assert _state_of(engine, watch_id_3) == "missed"


# ────────────────────────────────────────────────────────────────────────────
# 19.12: Promotion evidence package
# ────────────────────────────────────────────────────────────────────────────


@patch("utils.setup_watch_manager.MARKET_DATA_RELIABILITY_MODE", "disabled")
@patch("utils.setup_watch_manager.SETUP_WATCH_MISSED_MOVE_ENABLED", False)
def test_promotion_evidence_package_fields(engine, registry):
    """19.12: Promoted candidate's evidence package contains all required
    fields with correct values."""
    draft_geom = json.dumps({"entry": "150.0", "stop": "147.0", "target": "155.0", "risk_reward": "1.67"})
    entry_zone = json.dumps({"low": "148.0", "high": "152.0"})
    last_eval = json.dumps({
        "maturity_score": 0.9,
        "condition_results": [
            {"condition_type": "price_zone", "met": True, "detail": "in zone"},
        ],
        "evaluated_at": "2026-09-10T14:20:00Z",
    })

    watch_id = _insert_watch(
        engine,
        state="ready",
        maturity_score=0.9,
        observed_cycles=5,
        ready_at=NOW - timedelta(minutes=10),
        ready_reference_price=150.0,
        draft_geometry_json=draft_geom,
        entry_zone_json=entry_zone,
        last_evaluation_json=last_eval,
        source_type="analyst",
        source_id="signal_200",
        source_cycle_id=CYCLE,
    )

    from utils.setup_watch_manager import _build_evidence_package, _promote_ready_watch

    watch = registry.get_watch(watch_id)
    signals = {"AAPL": {"current_price": 151.0, "key_levels": {"support": 148.0}}}

    # Build evidence package
    evidence = _build_evidence_package(watch, CYCLE, signals)

    # Required fields from Req 6.1-6.7
    assert evidence["watch_id"] == watch_id
    assert evidence["setup_type"] == "support_bounce_swing"
    assert evidence["thesis"] is not None
    assert evidence["maturity_score"] == 0.9
    assert evidence["observed_cycles"] == 5
    assert evidence["last_evaluation_json"] is not None
    assert evidence["ready_reference_price"] == 150.0
    assert evidence["ready_at"] is not None
    assert evidence["source_trigger"] == CYCLE
    assert evidence["entry_zone"] is not None
    assert evidence["draft_geometry"] is not None
    assert evidence["current_price"] == 151.0
    assert evidence["key_levels"] == {"support": 148.0}
    assert evidence["source_provenance"]["source_type"] == "analyst"
    assert evidence["source_provenance"]["source_id"] == "signal_200"
    assert evidence["source_provenance"]["source_cycle_id"] == CYCLE

    # Now promote and verify the transition succeeds
    success = _promote_ready_watch(registry, watch, cycle_id=CYCLE, signals=signals)
    assert success is True
    assert _state_of(engine, watch_id) == "promoted"


# ────────────────────────────────────────────────────────────────────────────
# 19.13: Source provenance storage
# ────────────────────────────────────────────────────────────────────────────


@patch("utils.setup_watch_manager.SETUP_WATCH_MODE", "enabled")
def test_source_provenance_stored_and_validated(engine):
    """19.13: Watch created with source_type='price_monitor' and valid
    source_id → stored. Null source_id → rejected."""
    from utils.setup_watch_manager import create_setup_watch

    # Valid creation: price_monitor + source_id
    watch_id = create_setup_watch(
        engine,
        symbol="AMD",
        profile_id=PROFILE,
        side="BUY",
        setup_type="support_bounce_swing",
        thesis="A technical bounce thesis for AMD from price monitor observation",
        source_type="price_monitor",
        source_id="pm_alert_123",
        source_cycle_id=CYCLE,
        maturation_conditions=[
            _maturation_price_zone(100.0, 105.0),
            _maturation_support_hold(99.0),
        ],
        invalidation_conditions=[_invalidation_price_breach(95.0)],
    )
    assert watch_id is not None

    # Verify provenance stored
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT source_type, source_id, source_cycle_id "
                "FROM setup_watches WHERE watch_id = :wid"
            ),
            {"wid": watch_id},
        ).fetchone()
    assert row[0] == "price_monitor"
    assert row[1] == "pm_alert_123"
    assert row[2] == CYCLE

    # Rejected creation: price_monitor + null source_id
    watch_id_2 = create_setup_watch(
        engine,
        symbol="AMD",
        profile_id=PROFILE,
        side="SHORT",
        setup_type="support_bounce_swing",
        thesis="A short thesis for AMD from price monitor without source_id field",
        source_type="price_monitor",
        source_id=None,  # Null — should be rejected for new types
        source_cycle_id=CYCLE,
        maturation_conditions=[
            _maturation_price_zone(100.0, 105.0),
            _maturation_support_hold(99.0),
        ],
        invalidation_conditions=[_invalidation_price_breach(95.0)],
    )
    assert watch_id_2 is None


# ────────────────────────────────────────────────────────────────────────────
# 19.14: Entry zone tightening
# ────────────────────────────────────────────────────────────────────────────


def test_entry_zone_tightening(engine, registry):
    """19.14: Fresh signal with narrower zone replaces stored zone."""
    # Store an initial wide entry zone
    wide_zone = json.dumps({"low": "145.0", "high": "155.0"})
    watch_id = _insert_watch(
        engine,
        state="maturing",
        maturity_score=0.5,
        entry_zone_json=wide_zone,
    )

    from utils.setup_watch_manager import _maybe_tighten_entry_zone

    watch = registry.get_watch(watch_id)

    # Signal with narrower key levels → should tighten
    signals = {
        "AAPL": {
            "current_price": 150.0,
            "key_levels": {"support": 149.0, "resistance": 151.0},
        }
    }

    _maybe_tighten_entry_zone(engine, watch, signals)

    # Entry zone should now be tighter (narrower than 145-155)
    new_ez_json = _get_watch_field(engine, watch_id, "entry_zone_json")
    assert new_ez_json is not None
    new_ez = json.loads(new_ez_json)

    # The new zone derived from 149 + 151 should be narrower than 145-155
    new_width = Decimal(str(new_ez["high"])) - Decimal(str(new_ez["low"]))
    old_width = Decimal("155.0") - Decimal("145.0")
    assert new_width < old_width
