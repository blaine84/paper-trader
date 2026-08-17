"""Tests for utils/setup_watch_registry.py — CAS state machine.

Requirements: 1.9, 1.12, 3.1-3.10, 4.12, 11.1-11.3, 11.8, 12.5-12.6, 12.10
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from db.schema import init_setup_watch_schema
from utils.pending_order_time import now_utc, to_iso
from utils.setup_watch_registry import (
    ACTIVE_STATES,
    PERMITTED_TRANSITIONS,
    TERMINAL_STATES,
    SetupWatch,
    SetupWatchRegistry,
    SetupWatchRegistryError,
    WatchState,
    compute_watch_integrity_hash,
)

NOW = datetime(2026, 8, 14, 14, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    init_setup_watch_schema(eng)
    return eng


@pytest.fixture
def registry(engine):
    return SetupWatchRegistry(engine)


def _conditions(n: int) -> str:
    """Build a JSON list of n maturation conditions."""
    return json.dumps([{"type": "price_zone", "weight": 1.0}] * n)


def _invalidation(n: int = 1) -> str:
    """Build a JSON list of n invalidation conditions."""
    return json.dumps([{"type": "price_breach", "level": "100"}] * n)


def make_watch(**overrides) -> SetupWatch:
    defaults = dict(
        watch_id=str(uuid.uuid4()),
        profile_id="moderate",
        symbol="AAPL",
        side="BUY",
        setup_type="technical_breakout",
        state=WatchState.WATCHING,
        thesis="Stock approaching key support with strong volume",
        source_type="analyst",
        source_id="signal_123",
        source_cycle_id="cycle_001",
        maturation_conditions_json=_conditions(2),
        invalidation_conditions_json=_invalidation(1),
        last_evaluation_json=None,
        entry_zone_json=None,
        draft_geometry_json=None,
        maturity_score=0.0,
        created_at=NOW,
        updated_at=NOW,
        expires_at=NOW + timedelta(hours=6),
        state_changed_at=None,
        observed_cycles=0,
        ready_at=None,
        ready_reference_price=None,
        terminal_reason=None,
        promoted_cycle_id=None,
        execution_ref_type=None,
        execution_ref_id=None,
        integrity_hash="",
    )
    defaults.update(overrides)
    return SetupWatch(**defaults)


def create(registry, **overrides) -> str:
    return registry.create_watch(make_watch(**overrides))


def state_of(engine, watch_id) -> str:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT state FROM setup_watches WHERE watch_id = :wid"),
            {"wid": watch_id},
        ).scalar()


# ---------------------------------------------------------------------------
# 1. create_watch inserts with state='watching' and emits event
# ---------------------------------------------------------------------------


def test_create_watch_inserts_with_watching_state(registry, engine):
    wid = create(registry)
    assert state_of(engine, wid) == "watching"


def test_create_watch_emits_event(registry, engine):
    wid = create(registry)
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT event_type, watch_id FROM setup_watch_events "
                "WHERE watch_id = :wid"
            ),
            {"wid": wid},
        ).fetchone()
    assert row is not None
    assert row[0] == "watch_created"


# ---------------------------------------------------------------------------
# 2. create_watch supersedes an existing active watch for the same key
# ---------------------------------------------------------------------------


def test_create_watch_supersedes_existing_active(registry, engine):
    wid1 = create(registry, symbol="AAPL", side="BUY", setup_type="breakout")
    wid2 = create(registry, symbol="AAPL", side="BUY", setup_type="breakout")

    assert state_of(engine, wid1) == "expired"
    assert state_of(engine, wid2) == "watching"

    # Verify terminal_reason
    with engine.connect() as conn:
        reason = conn.execute(
            text(
                "SELECT terminal_reason FROM setup_watches WHERE watch_id = :wid"
            ),
            {"wid": wid1},
        ).scalar()
    assert reason == "superseded"


# ---------------------------------------------------------------------------
# 3. create_watch rejects below SETUP_WATCH_MIN_CONDITION_COUNT maturation conditions
# ---------------------------------------------------------------------------


def test_create_watch_rejects_insufficient_maturation_conditions(registry):
    # Default min is 2, so providing 1 should raise
    with pytest.raises(SetupWatchRegistryError, match="maturation_conditions_json"):
        create(registry, maturation_conditions_json=_conditions(1))


def test_create_watch_rejects_zero_maturation_conditions(registry):
    with pytest.raises(SetupWatchRegistryError, match="maturation_conditions_json"):
        create(registry, maturation_conditions_json="[]")


# ---------------------------------------------------------------------------
# 4. create_watch rejects zero invalidation conditions
# ---------------------------------------------------------------------------


def test_create_watch_rejects_zero_invalidation_conditions(registry):
    with pytest.raises(SetupWatchRegistryError, match="invalidation_conditions_json"):
        create(registry, invalidation_conditions_json="[]")


# ---------------------------------------------------------------------------
# 5. transition_state succeeds for every pair in PERMITTED_TRANSITIONS
# ---------------------------------------------------------------------------


def test_transition_state_succeeds_for_all_permitted(registry, engine):
    for from_state, to_state in PERMITTED_TRANSITIONS:
        wid = str(uuid.uuid4())
        # Directly insert with from_state to test each transition
        now_iso = to_iso(NOW)
        with engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO setup_watches "
                    "(watch_id, profile_id, symbol, side, setup_type, state, "
                    " thesis, source_type, source_cycle_id, "
                    " maturation_conditions_json, invalidation_conditions_json, "
                    " maturity_score, created_at, updated_at, expires_at, "
                    " observed_cycles, integrity_hash) "
                    "VALUES "
                    "(:wid, :pid, :sym, :side, :stype, :state, "
                    " :thesis, :src_type, :cycle_id, "
                    " :mat_json, :inv_json, "
                    " 0.0, :now, :now, :exp, "
                    " 0, :hash)"
                ),
                {
                    "wid": wid,
                    "pid": f"profile_{from_state.value}_{to_state.value}",
                    "sym": "AAPL",
                    "side": "BUY",
                    "stype": "breakout",
                    "state": from_state.value,
                    "thesis": "Test thesis text here",
                    "src_type": "analyst",
                    "cycle_id": "c1",
                    "mat_json": _conditions(2),
                    "inv_json": _invalidation(1),
                    "now": now_iso,
                    "exp": to_iso(NOW + timedelta(hours=8)),
                    "hash": "abc123",
                },
            )
            conn.commit()

        kwargs = {}
        if to_state == WatchState.READY:
            kwargs["ready_reference_price"] = 150.0
        if to_state in TERMINAL_STATES:
            kwargs["terminal_reason"] = "test"
        if to_state == WatchState.PROMOTED:
            kwargs["promoted_cycle_id"] = "cycle_x"
        if to_state == WatchState.ORDERED:
            kwargs["execution_ref_type"] = "trade"
            kwargs["execution_ref_id"] = "t_123"

        registry.transition_state(
            wid, from_state, to_state, **kwargs
        )
        assert state_of(engine, wid) == to_state.value


# ---------------------------------------------------------------------------
# 6. transition_state raises for pairs not in PERMITTED_TRANSITIONS
# ---------------------------------------------------------------------------


def test_transition_state_raises_for_illegal_pair(registry):
    # WATCHING -> ORDERED is not permitted
    wid = create(registry)
    with pytest.raises(SetupWatchRegistryError, match="Illegal transition"):
        registry.transition_state(wid, WatchState.WATCHING, WatchState.ORDERED)


def test_transition_state_raises_for_reverse_maturation(registry, engine):
    # WATCHING -> READY is not permitted (must pass through MATURING)
    wid = create(registry)
    with pytest.raises(SetupWatchRegistryError, match="Illegal transition"):
        registry.transition_state(wid, WatchState.WATCHING, WatchState.READY)


# ---------------------------------------------------------------------------
# 7. transition_state raises when CAS rowcount == 0
# ---------------------------------------------------------------------------


def test_transition_state_raises_on_cas_mismatch(registry, engine):
    wid = create(registry)
    # Watch is in WATCHING, but we claim it's in MATURING
    with pytest.raises(SetupWatchRegistryError, match="CAS transition failed"):
        registry.transition_state(wid, WatchState.MATURING, WatchState.READY, ready_reference_price=100.0)


# ---------------------------------------------------------------------------
# 8. Terminal states cannot transition onward
# ---------------------------------------------------------------------------


def test_terminal_states_cannot_transition(registry, engine):
    for terminal_state in TERMINAL_STATES:
        wid = str(uuid.uuid4())
        now_iso = to_iso(NOW)
        with engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO setup_watches "
                    "(watch_id, profile_id, symbol, side, setup_type, state, "
                    " thesis, source_type, source_cycle_id, "
                    " maturation_conditions_json, invalidation_conditions_json, "
                    " maturity_score, created_at, updated_at, expires_at, "
                    " observed_cycles, integrity_hash) "
                    "VALUES "
                    "(:wid, :pid, :sym, :side, :stype, :state, "
                    " :thesis, :src_type, :cycle_id, "
                    " :mat_json, :inv_json, "
                    " 0.0, :now, :now, :exp, "
                    " 0, :hash)"
                ),
                {
                    "wid": wid,
                    "pid": f"p_{terminal_state.value}",
                    "sym": "MSFT",
                    "side": "SHORT",
                    "stype": "pullback",
                    "state": terminal_state.value,
                    "thesis": "Terminal test",
                    "src_type": "analyst",
                    "cycle_id": "c1",
                    "mat_json": _conditions(2),
                    "inv_json": _invalidation(1),
                    "now": now_iso,
                    "exp": to_iso(NOW + timedelta(hours=8)),
                    "hash": "term",
                },
            )
            conn.commit()

        # No outgoing transition exists from a terminal state
        for _, to_state in PERMITTED_TRANSITIONS:
            with pytest.raises(SetupWatchRegistryError):
                registry.transition_state(
                    wid, terminal_state, to_state
                )


# ---------------------------------------------------------------------------
# 9. → READY sets ready_at and ready_reference_price
# ---------------------------------------------------------------------------


def test_transition_to_ready_sets_ready_at_and_reference_price(registry, engine):
    wid = create(registry)
    # WATCHING → MATURING
    registry.transition_state(wid, WatchState.WATCHING, WatchState.MATURING)
    # MATURING → READY
    registry.transition_state(
        wid, WatchState.MATURING, WatchState.READY, ready_reference_price=155.50
    )

    watch = registry.get_watch(wid)
    assert watch.ready_at is not None
    assert watch.ready_reference_price == 155.50


# ---------------------------------------------------------------------------
# 10. Re-entry to ready after regression does NOT overwrite either field
# ---------------------------------------------------------------------------


def test_reentry_to_ready_does_not_overwrite_fields(registry, engine):
    wid = create(registry)
    # WATCHING → MATURING → READY (first entry)
    registry.transition_state(wid, WatchState.WATCHING, WatchState.MATURING)
    registry.transition_state(
        wid, WatchState.MATURING, WatchState.READY, ready_reference_price=150.0
    )
    watch_after_first = registry.get_watch(wid)
    first_ready_at = watch_after_first.ready_at
    first_price = watch_after_first.ready_reference_price

    # READY → MATURING (regression)
    registry.transition_state(wid, WatchState.READY, WatchState.MATURING)
    # MATURING → READY (re-entry with different price)
    registry.transition_state(
        wid, WatchState.MATURING, WatchState.READY, ready_reference_price=200.0
    )
    watch_after_reentry = registry.get_watch(wid)

    # Should retain the original values thanks to COALESCE
    assert watch_after_reentry.ready_at == first_ready_at
    assert watch_after_reentry.ready_reference_price == first_price


# ---------------------------------------------------------------------------
# 11. → ORDERED sets execution_ref_type and execution_ref_id atomically
# ---------------------------------------------------------------------------


def test_transition_to_ordered_sets_execution_refs(registry, engine):
    wid = create(registry)
    registry.transition_state(wid, WatchState.WATCHING, WatchState.MATURING)
    registry.transition_state(
        wid, WatchState.MATURING, WatchState.READY, ready_reference_price=100.0
    )
    registry.transition_state(
        wid, WatchState.READY, WatchState.PROMOTED, promoted_cycle_id="c_99"
    )
    registry.transition_state(
        wid,
        WatchState.PROMOTED,
        WatchState.ORDERED,
        execution_ref_type="trade",
        execution_ref_id="trade_abc",
    )

    watch = registry.get_watch(wid)
    assert watch.state == WatchState.ORDERED
    assert watch.execution_ref_type == "trade"
    assert watch.execution_ref_id == "trade_abc"


# ---------------------------------------------------------------------------
# 12. update_evaluation writes score and last_evaluation_json but leaves
#     condition columns byte-identical
# ---------------------------------------------------------------------------


def test_update_evaluation_preserves_condition_columns(registry, engine):
    wid = create(registry)

    # Read condition columns before
    with engine.connect() as conn:
        row_before = conn.execute(
            text(
                "SELECT maturation_conditions_json, invalidation_conditions_json "
                "FROM setup_watches WHERE watch_id = :wid"
            ),
            {"wid": wid},
        ).fetchone()

    eval_json = json.dumps({"score": 0.75, "conditions": [{"met": True}]})
    registry.update_evaluation(wid, 0.75, eval_json)

    # Read condition columns after
    with engine.connect() as conn:
        row_after = conn.execute(
            text(
                "SELECT maturation_conditions_json, invalidation_conditions_json, "
                "maturity_score, last_evaluation_json "
                "FROM setup_watches WHERE watch_id = :wid"
            ),
            {"wid": wid},
        ).fetchone()

    # Condition columns byte-identical
    assert row_after[0] == row_before[0]
    assert row_after[1] == row_before[1]
    # Score and evaluation updated
    assert row_after[2] == 0.75
    assert row_after[3] == eval_json


# ---------------------------------------------------------------------------
# 13. update_evaluation is fail-open on error
# ---------------------------------------------------------------------------


def test_update_evaluation_fail_open(engine):
    """update_evaluation with a bad watch_id or broken engine does not raise."""
    registry = SetupWatchRegistry(engine)
    # Non-existent watch — should not raise, just log
    registry.update_evaluation("nonexistent_id", 0.5, '{"test": true}')

    # Engine that raises on connect
    broken_engine = MagicMock()
    broken_engine.connect.side_effect = RuntimeError("DB gone")
    broken_registry = SetupWatchRegistry(broken_engine)
    # Must not raise
    broken_registry.update_evaluation("any_id", 0.5, '{"x": 1}')


# ---------------------------------------------------------------------------
# 14. increment_observed_cycles increments only the given ids
# ---------------------------------------------------------------------------


def test_increment_observed_cycles_targets_specific_ids(registry, engine):
    wid1 = create(registry, symbol="A")
    wid2 = create(registry, symbol="B")
    wid3 = create(registry, symbol="C")

    registry.increment_observed_cycles([wid1, wid3])

    w1 = registry.get_watch(wid1)
    w2 = registry.get_watch(wid2)
    w3 = registry.get_watch(wid3)

    assert w1.observed_cycles == 1
    assert w2.observed_cycles == 0
    assert w3.observed_cycles == 1


def test_increment_observed_cycles_empty_list(registry):
    """Empty list is a no-op, no error."""
    registry.increment_observed_cycles([])


# ---------------------------------------------------------------------------
# 15. expire_elapsed skips watches with future expires_at
# ---------------------------------------------------------------------------


def test_expire_elapsed_skips_future_watches(registry, engine):
    # One watch expiring in the past
    past_wid = create(
        registry,
        symbol="PAST",
        expires_at=NOW - timedelta(hours=1),
    )
    # One watch expiring in the future
    future_wid = create(
        registry,
        symbol="FUTURE",
        expires_at=NOW + timedelta(hours=10),
    )

    with patch("utils.setup_watch_registry.now_utc", return_value=NOW):
        count = registry.expire_elapsed("moderate")

    assert count == 1
    assert state_of(engine, past_wid) == "expired"
    assert state_of(engine, future_wid) == "watching"


# ---------------------------------------------------------------------------
# 16. expire_stale_promoted spares the current cycle
# ---------------------------------------------------------------------------


def test_expire_stale_promoted_spares_current_cycle(registry, engine):
    # Create two watches, promote both with different cycle IDs
    wid_old = create(registry, symbol="OLD")
    wid_current = create(registry, symbol="CUR")

    # Manually set both to promoted state with different cycle IDs
    now_iso = to_iso(NOW)
    with engine.connect() as conn:
        conn.execute(
            text(
                "UPDATE setup_watches SET state='promoted', promoted_cycle_id='old_cycle' "
                "WHERE watch_id = :wid"
            ),
            {"wid": wid_old},
        )
        conn.execute(
            text(
                "UPDATE setup_watches SET state='promoted', promoted_cycle_id='current_cycle' "
                "WHERE watch_id = :wid"
            ),
            {"wid": wid_current},
        )
        conn.commit()

    count = registry.expire_stale_promoted("moderate", "current_cycle")

    assert count == 1
    assert state_of(engine, wid_old) == "expired"
    assert state_of(engine, wid_current) == "promoted"


# ---------------------------------------------------------------------------
# 17. count_active excludes terminal states; count_active_for_symbol spans profiles
# ---------------------------------------------------------------------------


def test_count_active_excludes_terminal(registry, engine):
    create(registry, symbol="X")  # watching = active
    wid2 = create(registry, symbol="Y")
    # Expire one
    registry.transition_state(wid2, WatchState.WATCHING, WatchState.EXPIRED)

    assert registry.count_active("moderate") == 1


def test_count_active_for_symbol_spans_profiles(registry, engine):
    create(registry, profile_id="prof_a", symbol="TSLA")
    create(registry, profile_id="prof_b", symbol="TSLA")
    create(registry, profile_id="prof_c", symbol="OTHER")

    assert registry.count_active_for_symbol("TSLA") == 2


# ---------------------------------------------------------------------------
# 18. Partial unique index blocks a second active watch for the same key
# ---------------------------------------------------------------------------


def test_partial_unique_index_blocks_duplicate_active_key(engine):
    """Direct INSERT of two active watches for the same key should fail."""
    now_iso = to_iso(NOW)
    base = {
        "pid": "moderate",
        "sym": "GOOG",
        "side": "BUY",
        "stype": "breakout",
        "state": "watching",
        "thesis": "Test thesis data",
        "src_type": "analyst",
        "cycle_id": "c1",
        "mat_json": _conditions(2),
        "inv_json": _invalidation(1),
        "now": now_iso,
        "exp": to_iso(NOW + timedelta(hours=8)),
        "hash": "h1",
    }

    insert_sql = text(
        "INSERT INTO setup_watches "
        "(watch_id, profile_id, symbol, side, setup_type, state, "
        " thesis, source_type, source_cycle_id, "
        " maturation_conditions_json, invalidation_conditions_json, "
        " maturity_score, created_at, updated_at, expires_at, "
        " observed_cycles, integrity_hash) "
        "VALUES "
        "(:wid, :pid, :sym, :side, :stype, :state, "
        " :thesis, :src_type, :cycle_id, "
        " :mat_json, :inv_json, "
        " 0.0, :now, :now, :exp, "
        " 0, :hash)"
    )

    with engine.connect() as conn:
        conn.execute(insert_sql, {**base, "wid": "watch_aaa"})
        conn.commit()

    with engine.connect() as conn:
        with pytest.raises(Exception):  # IntegrityError
            conn.execute(insert_sql, {**base, "wid": "watch_bbb"})
            conn.commit()


def test_partial_unique_index_allows_terminal_duplicates(engine):
    """Terminal-state watches don't conflict with the unique index."""
    now_iso = to_iso(NOW)
    base = {
        "pid": "moderate",
        "sym": "GOOG",
        "side": "BUY",
        "stype": "breakout",
        "thesis": "Test thesis data",
        "src_type": "analyst",
        "cycle_id": "c1",
        "mat_json": _conditions(2),
        "inv_json": _invalidation(1),
        "now": now_iso,
        "exp": to_iso(NOW + timedelta(hours=8)),
        "hash": "h1",
    }

    insert_sql = text(
        "INSERT INTO setup_watches "
        "(watch_id, profile_id, symbol, side, setup_type, state, "
        " thesis, source_type, source_cycle_id, "
        " maturation_conditions_json, invalidation_conditions_json, "
        " maturity_score, created_at, updated_at, expires_at, "
        " observed_cycles, integrity_hash) "
        "VALUES "
        "(:wid, :pid, :sym, :side, :stype, :state, "
        " :thesis, :src_type, :cycle_id, "
        " :mat_json, :inv_json, "
        " 0.0, :now, :now, :exp, "
        " 0, :hash)"
    )

    with engine.connect() as conn:
        # Insert one expired and one active — no conflict
        conn.execute(insert_sql, {**base, "wid": "w_expired", "state": "expired"})
        conn.execute(insert_sql, {**base, "wid": "w_active", "state": "watching"})
        conn.commit()


# ---------------------------------------------------------------------------
# 19. record_outcome returns False on duplicate (watch_id, window_label)
# ---------------------------------------------------------------------------


def test_record_outcome_returns_false_on_duplicate(registry, engine):
    wid = create(registry)
    outcome = {
        "watch_id": wid,
        "profile_id": "moderate",
        "symbol": "AAPL",
        "side": "BUY",
        "window_label": "w15",
        "window_minutes": 15,
        "reference_price": 150.0,
        "evaluated_at": to_iso(NOW),
        "mfe_pct": 0.02,
        "mae_pct": -0.01,
        "entry_zone_touched": 1,
        "would_have_hit_target": 1,
        "would_have_hit_stop": 0,
        "scorable": 1,
        "unscorable_reason": None,
        "created_at": to_iso(NOW),
    }
    assert registry.record_outcome(outcome) is True
    # Duplicate should return False (benign)
    assert registry.record_outcome(outcome) is False


# ---------------------------------------------------------------------------
# 20. Immutability triggers block UPDATE and DELETE on events and outcomes
# ---------------------------------------------------------------------------


def test_immutability_trigger_blocks_update_on_events(registry, engine):
    wid = create(registry)
    # Event was emitted by create_watch; try to UPDATE it
    with engine.connect() as conn:
        with pytest.raises(Exception, match="immutable.*UPDATE"):
            conn.execute(
                text(
                    "UPDATE setup_watch_events SET event_type = 'hacked' "
                    "WHERE watch_id = :wid"
                ),
                {"wid": wid},
            )


def test_immutability_trigger_blocks_delete_on_events(registry, engine):
    wid = create(registry)
    with engine.connect() as conn:
        with pytest.raises(Exception, match="immutable.*DELETE"):
            conn.execute(
                text("DELETE FROM setup_watch_events WHERE watch_id = :wid"),
                {"wid": wid},
            )


def test_immutability_trigger_blocks_update_on_outcomes(registry, engine):
    wid = create(registry)
    outcome = {
        "watch_id": wid,
        "profile_id": "moderate",
        "symbol": "AAPL",
        "side": "BUY",
        "window_label": "w30",
        "window_minutes": 30,
        "reference_price": 150.0,
        "evaluated_at": to_iso(NOW),
        "mfe_pct": 0.05,
        "mae_pct": -0.02,
        "entry_zone_touched": 0,
        "would_have_hit_target": 0,
        "would_have_hit_stop": 0,
        "scorable": 1,
        "unscorable_reason": None,
        "created_at": to_iso(NOW),
    }
    registry.record_outcome(outcome)

    with engine.connect() as conn:
        with pytest.raises(Exception, match="immutable.*UPDATE"):
            conn.execute(
                text(
                    "UPDATE setup_watch_outcomes SET mfe_pct = 99.0 "
                    "WHERE watch_id = :wid"
                ),
                {"wid": wid},
            )


def test_immutability_trigger_blocks_delete_on_outcomes(registry, engine):
    wid = create(registry)
    outcome = {
        "watch_id": wid,
        "profile_id": "moderate",
        "symbol": "AAPL",
        "side": "BUY",
        "window_label": "w60",
        "window_minutes": 60,
        "reference_price": 150.0,
        "evaluated_at": to_iso(NOW),
        "mfe_pct": 0.01,
        "mae_pct": -0.005,
        "entry_zone_touched": 1,
        "would_have_hit_target": 1,
        "would_have_hit_stop": 0,
        "scorable": 1,
        "unscorable_reason": None,
        "created_at": to_iso(NOW),
    }
    registry.record_outcome(outcome)

    with engine.connect() as conn:
        with pytest.raises(Exception, match="immutable.*DELETE"):
            conn.execute(
                text("DELETE FROM setup_watch_outcomes WHERE watch_id = :wid"),
                {"wid": wid},
            )


# ---------------------------------------------------------------------------
# 21. Event emission failure does not propagate
# ---------------------------------------------------------------------------


def test_event_emission_failure_does_not_propagate(engine):
    """If _execute_event_write raises, the state transition still completes.

    The _emit_event method wraps _execute_event_write in try/except (fail-open),
    so we mock at the lower level to confirm the exception is swallowed.
    """
    registry = SetupWatchRegistry(engine)
    wid = create(registry)

    # Patch _execute_event_write so event INSERT fails, but _emit_event swallows it
    with patch.object(
        registry, "_execute_event_write", side_effect=RuntimeError("event DB down")
    ):
        # State transition still completes — event failure is swallowed
        registry.transition_state(wid, WatchState.WATCHING, WatchState.MATURING)

    assert state_of(engine, wid) == "maturing"


# ---------------------------------------------------------------------------
# 22. side is normalized to uppercase on creation
# ---------------------------------------------------------------------------


def test_side_normalized_to_uppercase_buy(registry, engine):
    wid = create(registry, side="buy")
    with engine.connect() as conn:
        stored_side = conn.execute(
            text("SELECT side FROM setup_watches WHERE watch_id = :wid"),
            {"wid": wid},
        ).scalar()
    assert stored_side == "BUY"


def test_side_normalized_to_uppercase_short(registry, engine):
    wid = create(registry, side="short")
    with engine.connect() as conn:
        stored_side = conn.execute(
            text("SELECT side FROM setup_watches WHERE watch_id = :wid"),
            {"wid": wid},
        ).scalar()
    assert stored_side == "SHORT"


def test_side_normalized_mixed_case(registry, engine):
    wid = create(registry, side="Short")
    with engine.connect() as conn:
        stored_side = conn.execute(
            text("SELECT side FROM setup_watches WHERE watch_id = :wid"),
            {"wid": wid},
        ).scalar()
    assert stored_side == "SHORT"


# ---------------------------------------------------------------------------
# Additional validation tests
# ---------------------------------------------------------------------------


def test_create_watch_rejects_invalid_json_maturation(registry):
    with pytest.raises(SetupWatchRegistryError, match="not valid JSON"):
        create(registry, maturation_conditions_json="not json at all")


def test_create_watch_rejects_invalid_json_invalidation(registry):
    with pytest.raises(SetupWatchRegistryError, match="not valid JSON"):
        create(registry, invalidation_conditions_json="{broken")


def test_create_watch_rejects_invalid_side(registry):
    with pytest.raises(SetupWatchRegistryError, match="side must be"):
        create(registry, side="SELL")


def test_integrity_hash_is_deterministic():
    w = make_watch(watch_id="fixed_id", profile_id="p1", symbol="X", side="BUY",
                   setup_type="t", thesis="thesis")
    h1 = compute_watch_integrity_hash(w)
    h2 = compute_watch_integrity_hash(w)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex digest


def test_get_active_watches_returns_only_nonterminal(registry, engine):
    wid_active = create(registry, symbol="ACT")
    wid_expired = create(registry, symbol="EXP")
    registry.transition_state(wid_expired, WatchState.WATCHING, WatchState.EXPIRED)

    active = registry.get_active_watches("moderate")
    active_ids = [w.watch_id for w in active]
    assert wid_active in active_ids
    assert wid_expired not in active_ids


def test_get_promoted_watches(registry, engine):
    wid = create(registry)
    registry.transition_state(wid, WatchState.WATCHING, WatchState.MATURING)
    registry.transition_state(wid, WatchState.MATURING, WatchState.READY, ready_reference_price=100.0)
    registry.transition_state(wid, WatchState.READY, WatchState.PROMOTED, promoted_cycle_id="c_55")

    promoted = registry.get_promoted_watches("moderate", "c_55")
    assert len(promoted) == 1
    assert promoted[0].watch_id == wid

    # Different cycle yields empty
    assert registry.get_promoted_watches("moderate", "c_other") == []
