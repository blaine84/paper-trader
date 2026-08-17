"""Tests for the setup_watches / setup_watch_events / setup_watch_outcomes schema.

Requirements: 1.1–1.12

Mirrors tests/test_pending_order_schema.py in structure and coverage.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, inspect, text

from db.schema import init_setup_watch_schema

NOW = datetime(2026, 8, 17, 14, 30, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    init_setup_watch_schema(eng)
    return eng


def _watch_row(**overrides) -> dict:
    """A minimally valid setup_watches row."""
    row = {
        "watch_id": str(uuid.uuid4()),
        "profile_id": "moderate",
        "symbol": "AAPL",
        "side": "BUY",
        "setup_type": "technical_breakout",
        "state": "watching",
        "thesis": "Testing a breakout above key resistance",
        "source_type": "analyst",
        "source_id": None,
        "source_cycle_id": "cycle_001",
        "maturation_conditions_json": '[{"type": "price_zone", "params": {"low": 148.5, "high": 150.0}, "weight": 0.5}]',
        "invalidation_conditions_json": '[{"type": "price_breach", "params": {"level": 145.0, "direction": "below"}}]',
        "last_evaluation_json": None,
        "entry_zone_json": None,
        "draft_geometry_json": None,
        "maturity_score": 0.0,
        "created_at": _iso(NOW),
        "updated_at": _iso(NOW),
        "expires_at": _iso(NOW + timedelta(hours=8)),
        "state_changed_at": None,
        "observed_cycles": 0,
        "ready_at": None,
        "ready_reference_price": None,
        "terminal_reason": None,
        "promoted_cycle_id": None,
        "execution_ref_type": None,
        "execution_ref_id": None,
        "integrity_hash": "abc123deadbeef",
    }
    row.update(overrides)
    return row


def _insert_watch(engine, **overrides) -> str:
    row = _watch_row(**overrides)
    columns = ", ".join(row.keys())
    placeholders = ", ".join(f":{k}" for k in row)
    with engine.connect() as conn:
        conn.execute(
            text(f"INSERT INTO setup_watches ({columns}) VALUES ({placeholders})"),
            row,
        )
        conn.commit()
    return row["watch_id"]


def _insert_event(engine, watch_id: str, event_type: str = "watch_created") -> None:
    with engine.connect() as conn:
        conn.execute(
            text(
                """
                INSERT INTO setup_watch_events
                    (watch_id, profile_id, symbol, event_type, event_data,
                     from_state, to_state, maturity_score, created_at)
                VALUES
                    (:watch_id, :profile_id, :symbol, :event_type, :event_data,
                     :from_state, :to_state, :maturity_score, :created_at)
                """
            ),
            {
                "watch_id": watch_id,
                "profile_id": "moderate",
                "symbol": "AAPL",
                "event_type": event_type,
                "event_data": '{"reason": "test"}',
                "from_state": None,
                "to_state": "watching",
                "maturity_score": 0.0,
                "created_at": _iso(NOW),
            },
        )
        conn.commit()


def _insert_outcome(engine, watch_id: str, window_label: str = "w15") -> None:
    with engine.connect() as conn:
        conn.execute(
            text(
                """
                INSERT INTO setup_watch_outcomes
                    (watch_id, profile_id, symbol, side, window_label,
                     window_minutes, reference_price, evaluated_at,
                     mfe_pct, mae_pct, entry_zone_touched,
                     would_have_hit_target, would_have_hit_stop,
                     scorable, unscorable_reason, created_at)
                VALUES
                    (:watch_id, :profile_id, :symbol, :side, :window_label,
                     :window_minutes, :reference_price, :evaluated_at,
                     :mfe_pct, :mae_pct, :entry_zone_touched,
                     :would_have_hit_target, :would_have_hit_stop,
                     :scorable, :unscorable_reason, :created_at)
                """
            ),
            {
                "watch_id": watch_id,
                "profile_id": "moderate",
                "symbol": "AAPL",
                "side": "BUY",
                "window_label": window_label,
                "window_minutes": 15,
                "reference_price": 149.50,
                "evaluated_at": _iso(NOW + timedelta(minutes=15)),
                "mfe_pct": 1.2,
                "mae_pct": -0.5,
                "entry_zone_touched": 1,
                "would_have_hit_target": 1,
                "would_have_hit_stop": 0,
                "scorable": 1,
                "unscorable_reason": None,
                "created_at": _iso(NOW + timedelta(minutes=15)),
            },
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Table and column creation
# ---------------------------------------------------------------------------


def test_init_creates_all_tables(engine):
    inspector = inspect(engine)
    assert inspector.has_table("setup_watches")
    assert inspector.has_table("setup_watch_events")
    assert inspector.has_table("setup_watch_outcomes")


def test_setup_watches_has_every_declared_column(engine):
    actual = {c["name"] for c in inspect(engine).get_columns("setup_watches")}
    expected = set(_watch_row().keys())
    assert expected == actual, f"missing: {expected - actual}, extra: {actual - expected}"


def test_setup_watch_events_has_every_declared_column(engine):
    actual = {c["name"] for c in inspect(engine).get_columns("setup_watch_events")}
    expected = {
        "id", "watch_id", "profile_id", "symbol", "event_type", "event_data",
        "from_state", "to_state", "maturity_score", "created_at",
    }
    assert expected == actual


def test_setup_watch_outcomes_has_every_declared_column(engine):
    actual = {c["name"] for c in inspect(engine).get_columns("setup_watch_outcomes")}
    expected = {
        "id", "watch_id", "profile_id", "symbol", "side", "window_label",
        "window_minutes", "reference_price", "evaluated_at",
        "mfe_pct", "mae_pct", "entry_zone_touched",
        "would_have_hit_target", "would_have_hit_stop",
        "scorable", "unscorable_reason", "created_at",
    }
    assert expected == actual


def test_state_defaults_to_watching(engine):
    """A row inserted without explicit state must land in 'watching'."""
    watch_id = str(uuid.uuid4())
    with engine.connect() as conn:
        conn.execute(
            text(
                """
                INSERT INTO setup_watches
                    (watch_id, profile_id, symbol, side, setup_type,
                     thesis, source_type, source_cycle_id,
                     maturation_conditions_json, invalidation_conditions_json,
                     created_at, updated_at, expires_at, integrity_hash)
                VALUES
                    (:wid, 'moderate', 'AAPL', 'BUY', 'technical_breakout',
                     'Test thesis here.', 'analyst', 'cycle_001',
                     '[]', '[]',
                     :created, :updated, :expires, 'hash')
                """
            ),
            {
                "wid": watch_id,
                "created": _iso(NOW),
                "updated": _iso(NOW),
                "expires": _iso(NOW + timedelta(hours=8)),
            },
        )
        conn.commit()
        state = conn.execute(
            text("SELECT state FROM setup_watches WHERE watch_id = :wid"),
            {"wid": watch_id},
        ).scalar()
    assert state == "watching"


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


def test_insert_watch_event_outcome(engine):
    """Full lifecycle: insert a watch, event, and outcome successfully."""
    watch_id = _insert_watch(engine)
    _insert_event(engine, watch_id)
    _insert_outcome(engine, watch_id)

    with engine.connect() as conn:
        watches = conn.execute(text("SELECT COUNT(*) FROM setup_watches")).scalar()
        events = conn.execute(text("SELECT COUNT(*) FROM setup_watch_events")).scalar()
        outcomes = conn.execute(text("SELECT COUNT(*) FROM setup_watch_outcomes")).scalar()

    assert watches == 1
    assert events == 1
    assert outcomes == 1


# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------


def test_every_declared_index_exists(engine):
    inspector = inspect(engine)
    watch_indexes = {ix["name"] for ix in inspector.get_indexes("setup_watches")}
    event_indexes = {ix["name"] for ix in inspector.get_indexes("setup_watch_events")}
    outcome_indexes = {ix["name"] for ix in inspector.get_indexes("setup_watch_outcomes")}

    assert {
        "idx_setup_watches_profile_state",
        "idx_setup_watches_symbol_state",
        "idx_setup_watches_state_expires",
        "idx_setup_watches_active_key",
    } <= watch_indexes

    assert {
        "idx_setup_watch_events_watch",
        "idx_setup_watch_events_type",
    } <= event_indexes

    assert {
        "idx_setup_watch_outcomes_watch_window",
        "idx_setup_watch_outcomes_profile_window",
    } <= outcome_indexes


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_init_is_idempotent(engine):
    """Safe to run on every orchestrator startup."""
    init_setup_watch_schema(engine)
    init_setup_watch_schema(engine)

    inspector = inspect(engine)
    assert inspector.has_table("setup_watches")
    assert inspector.has_table("setup_watch_events")
    assert inspector.has_table("setup_watch_outcomes")


def test_init_preserves_existing_rows(engine):
    """Re-running the migration must not disturb live data."""
    watch_id = _insert_watch(engine)
    _insert_event(engine, watch_id)
    _insert_outcome(engine, watch_id)

    init_setup_watch_schema(engine)
    init_setup_watch_schema(engine)

    with engine.connect() as conn:
        watches = conn.execute(text("SELECT COUNT(*) FROM setup_watches")).scalar()
        events = conn.execute(text("SELECT COUNT(*) FROM setup_watch_events")).scalar()
        outcomes = conn.execute(text("SELECT COUNT(*) FROM setup_watch_outcomes")).scalar()
        thesis = conn.execute(
            text("SELECT thesis FROM setup_watches WHERE watch_id = :wid"),
            {"wid": watch_id},
        ).scalar()

    assert watches == 1
    assert events == 1
    assert outcomes == 1
    assert thesis == "Testing a breakout above key resistance"


def test_repeated_init_does_not_duplicate_columns(engine):
    before = [c["name"] for c in inspect(engine).get_columns("setup_watches")]
    init_setup_watch_schema(engine)
    after = [c["name"] for c in inspect(engine).get_columns("setup_watches")]

    assert before == after
    assert len(after) == len(set(after)), "duplicate column names after re-init"


# ---------------------------------------------------------------------------
# Immutability of the events audit trail
# ---------------------------------------------------------------------------


def test_event_update_is_blocked(engine):
    watch_id = _insert_watch(engine)
    _insert_event(engine, watch_id)

    with pytest.raises(Exception, match="immutable"):
        with engine.connect() as conn:
            conn.execute(
                text("UPDATE setup_watch_events SET event_type = 'tampered'")
            )
            conn.commit()


def test_event_delete_is_blocked(engine):
    watch_id = _insert_watch(engine)
    _insert_event(engine, watch_id)

    with pytest.raises(Exception, match="immutable"):
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM setup_watch_events"))
            conn.commit()


def test_events_survive_a_blocked_mutation_attempt(engine):
    """A rejected UPDATE must leave the row intact."""
    watch_id = _insert_watch(engine)
    _insert_event(engine, watch_id, event_type="watch_created")

    with pytest.raises(Exception, match="immutable"):
        with engine.connect() as conn:
            conn.execute(
                text("UPDATE setup_watch_events SET event_type = 'tampered'")
            )
            conn.commit()

    with engine.connect() as conn:
        event_type = conn.execute(
            text("SELECT event_type FROM setup_watch_events")
        ).scalar()
    assert event_type == "watch_created"


# ---------------------------------------------------------------------------
# Immutability of the outcomes table
# ---------------------------------------------------------------------------


def test_outcome_update_is_blocked(engine):
    watch_id = _insert_watch(engine)
    _insert_outcome(engine, watch_id)

    with pytest.raises(Exception, match="immutable"):
        with engine.connect() as conn:
            conn.execute(
                text("UPDATE setup_watch_outcomes SET mfe_pct = 99.9")
            )
            conn.commit()


def test_outcome_delete_is_blocked(engine):
    watch_id = _insert_watch(engine)
    _insert_outcome(engine, watch_id)

    with pytest.raises(Exception, match="immutable"):
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM setup_watch_outcomes"))
            conn.commit()


def test_outcomes_survive_a_blocked_mutation_attempt(engine):
    """A rejected UPDATE must leave the row intact."""
    watch_id = _insert_watch(engine)
    _insert_outcome(engine, watch_id)

    with pytest.raises(Exception, match="immutable"):
        with engine.connect() as conn:
            conn.execute(
                text("UPDATE setup_watch_outcomes SET mfe_pct = 99.9")
            )
            conn.commit()

    with engine.connect() as conn:
        mfe = conn.execute(
            text("SELECT mfe_pct FROM setup_watch_outcomes")
        ).scalar()
    assert abs(mfe - 1.2) < 0.001


# ---------------------------------------------------------------------------
# setup_watches itself is mutable (CAS state transitions must work)
# ---------------------------------------------------------------------------


def test_setup_watches_is_mutable(engine):
    """setup_watches is NOT immutable — CAS state transitions must work."""
    watch_id = _insert_watch(engine)

    with engine.connect() as conn:
        result = conn.execute(
            text(
                "UPDATE setup_watches SET state = 'maturing' "
                "WHERE watch_id = :wid AND state = 'watching'"
            ),
            {"wid": watch_id},
        )
        conn.commit()
        assert result.rowcount == 1

        state = conn.execute(
            text("SELECT state FROM setup_watches WHERE watch_id = :wid"),
            {"wid": watch_id},
        ).scalar()
    assert state == "maturing"


# ---------------------------------------------------------------------------
# Partial unique index: active-key constraint
# ---------------------------------------------------------------------------


def test_two_active_watches_for_same_key_are_rejected(engine):
    """At most one active watch per (profile, symbol, side, setup_type)."""
    _insert_watch(engine, state="watching")

    with pytest.raises(Exception):
        _insert_watch(engine, state="watching")


def test_watching_and_maturing_collide_on_same_key(engine):
    """Both are active states, so they participate in the constraint."""
    _insert_watch(engine, state="watching")

    with pytest.raises(Exception):
        _insert_watch(engine, state="maturing")


def test_active_states_all_collide(engine):
    """ready and promoted are also active states."""
    _insert_watch(engine, state="ready")

    with pytest.raises(Exception):
        _insert_watch(engine, state="promoted")


@pytest.mark.parametrize(
    "terminal_state", ["expired", "rejected", "ordered"]
)
def test_terminal_watches_do_not_block_a_new_active_watch(engine, terminal_state):
    """History accumulates freely — only active rows are constrained."""
    _insert_watch(engine, state=terminal_state)
    _insert_watch(engine, state=terminal_state)

    # A fresh active watch for the same key is still allowed.
    watch_id = _insert_watch(engine, state="watching")

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM setup_watches")).scalar()
        active = conn.execute(
            text("SELECT COUNT(*) FROM setup_watches WHERE state = 'watching'")
        ).scalar()

    assert total == 3
    assert active == 1
    assert watch_id is not None


@pytest.mark.parametrize(
    "differing_field,value",
    [
        ("symbol", "AMD"),
        ("side", "SHORT"),
        ("setup_type", "gap_and_go"),
        ("profile_id", "aggressive"),
    ],
)
def test_watches_differing_in_any_key_component_coexist(engine, differing_field, value):
    """The constraint is on the full 4-tuple, not any single column."""
    _insert_watch(engine, state="watching")
    _insert_watch(engine, state="watching", **{differing_field: value})

    with engine.connect() as conn:
        active = conn.execute(
            text("SELECT COUNT(*) FROM setup_watches WHERE state = 'watching'")
        ).scalar()
    assert active == 2


# ---------------------------------------------------------------------------
# Outcomes unique index: (watch_id, window_label)
# ---------------------------------------------------------------------------


def test_duplicate_outcome_for_same_window_is_rejected(engine):
    """Each watch scored at most once per window."""
    watch_id = _insert_watch(engine)
    _insert_outcome(engine, watch_id, window_label="w15")

    with pytest.raises(Exception):
        _insert_outcome(engine, watch_id, window_label="w15")


def test_different_windows_for_same_watch_coexist(engine):
    """A watch can have outcomes for w15, w30, w60."""
    watch_id = _insert_watch(engine)
    _insert_outcome(engine, watch_id, window_label="w15")
    _insert_outcome(engine, watch_id, window_label="w30")
    _insert_outcome(engine, watch_id, window_label="w60")

    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM setup_watch_outcomes WHERE watch_id = :wid"),
            {"wid": watch_id},
        ).scalar()
    assert count == 3


def test_same_window_label_for_different_watches_coexist(engine):
    """Different watches can each have a w15 outcome."""
    w1 = _insert_watch(engine, symbol="AAPL")
    w2 = _insert_watch(engine, symbol="MSFT")
    _insert_outcome(engine, w1, window_label="w15")
    _insert_outcome(engine, w2, window_label="w15")

    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM setup_watch_outcomes")
        ).scalar()
    assert count == 2
