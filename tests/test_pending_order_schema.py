"""Tests for the pending_orders / pending_order_events schema.

Requirements: 2.8, 2.9, 2.10, 2.11, 2.12, 7.4

Note: the check_schema() wiring is deliberately NOT covered here. orchestrator.py
imports utils.resource_telemetry, which does a bare `import resource` (Unix-only
stdlib), so orchestrator cannot be imported on Windows at all. That wiring is
verified in the task 9 integration tests on a platform where it imports.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, inspect, text

from db.schema import init_pending_order_schema
from utils.pending_order_time import to_iso

NOW = datetime(2026, 8, 14, 14, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    init_pending_order_schema(eng)
    return eng


def _order_row(**overrides) -> dict:
    """A minimally valid pending_orders row."""
    row = {
        "order_id": str(uuid.uuid4()),
        "profile_id": "moderate",
        "symbol": "META",
        "side": "BUY",
        "setup_type": "technical_breakout",
        "geometry_name": None,
        "candidate_id": None,
        "cycle_id": None,
        "source_signal_id": None,
        "plan_id": None,
        "limit_price": 593.87,
        "stop_price": 588.00,
        "target_price": 605.00,
        "risk_reward": 1.9,
        "intended_quantity": 10,
        "fresh_price_at_creation": 601.24,
        "runaway_pct_at_creation": 0.0124,
        "pm_rationale": None,
        "signal_snapshot_json": None,
        "state": "pending",
        "created_at": to_iso(NOW),
        "expires_at": to_iso(NOW + timedelta(hours=2)),
        "last_evaluated_bar_ts": None,
        "filled_at": None,
        "terminal_at": None,
        "fill_price": None,
        "fill_policy": None,
        "fill_bar_ts": None,
        "terminal_reason": None,
        "trade_id": None,
        "integrity_hash": "deadbeef",
    }
    row.update(overrides)
    return row


def _insert_order(engine, **overrides) -> str:
    row = _order_row(**overrides)
    columns = ", ".join(row.keys())
    placeholders = ", ".join(f":{k}" for k in row)
    with engine.connect() as conn:
        conn.execute(
            text(f"INSERT INTO pending_orders ({columns}) VALUES ({placeholders})"),
            row,
        )
        conn.commit()
    return row["order_id"]


def _insert_event(engine, order_id: str, event_type: str = "state_pending") -> None:
    with engine.connect() as conn:
        conn.execute(
            text(
                """
                INSERT INTO pending_order_events
                    (order_id, profile_id, symbol, event_type, event_data,
                     from_state, to_state, reference_price, created_at)
                VALUES
                    (:order_id, :profile_id, :symbol, :event_type, :event_data,
                     :from_state, :to_state, :reference_price, :created_at)
                """
            ),
            {
                "order_id": order_id,
                "profile_id": "moderate",
                "symbol": "META",
                "event_type": event_type,
                "event_data": '{"reason": "test"}',
                "from_state": None,
                "to_state": "pending",
                "reference_price": 601.24,
                "created_at": to_iso(NOW),
            },
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Table and column creation
# ---------------------------------------------------------------------------


def test_init_creates_both_tables(engine):
    inspector = inspect(engine)
    assert inspector.has_table("pending_orders")
    assert inspector.has_table("pending_order_events")


def test_pending_orders_has_every_declared_column(engine):
    actual = {c["name"] for c in inspect(engine).get_columns("pending_orders")}
    expected = set(_order_row().keys())
    assert expected == actual, f"missing: {expected - actual}, extra: {actual - expected}"


def test_pending_order_events_has_every_declared_column(engine):
    actual = {c["name"] for c in inspect(engine).get_columns("pending_order_events")}
    expected = {
        "id", "order_id", "profile_id", "symbol", "event_type", "event_data",
        "from_state", "to_state", "reference_price", "created_at",
    }
    assert expected == actual


def test_state_defaults_to_pending(engine):
    """A row inserted without an explicit state must land in 'pending'."""
    order_id = str(uuid.uuid4())
    with engine.connect() as conn:
        conn.execute(
            text(
                """
                INSERT INTO pending_orders
                    (order_id, profile_id, symbol, side, setup_type,
                     limit_price, stop_price, target_price, risk_reward,
                     fresh_price_at_creation, runaway_pct_at_creation,
                     created_at, expires_at, integrity_hash)
                VALUES
                    (:oid, 'moderate', 'META', 'BUY', 'technical_breakout',
                     593.87, 588.0, 605.0, 1.9,
                     601.24, 0.0124,
                     :created, :expires, 'hash')
                """
            ),
            {
                "oid": order_id,
                "created": to_iso(NOW),
                "expires": to_iso(NOW + timedelta(hours=2)),
            },
        )
        conn.commit()
        state = conn.execute(
            text("SELECT state FROM pending_orders WHERE order_id = :oid"),
            {"oid": order_id},
        ).scalar()
    assert state == "pending"


def test_all_linkage_columns_accept_null(engine):
    """The live legacy PM path produces no candidate, cycle, signal or plan id."""
    order_id = _insert_order(
        engine, candidate_id=None, cycle_id=None,
        source_signal_id=None, plan_id=None,
    )
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT candidate_id, cycle_id, source_signal_id, plan_id "
                "FROM pending_orders WHERE order_id = :oid"
            ),
            {"oid": order_id},
        ).fetchone()
    assert row == (None, None, None, None)


# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------


def test_every_declared_index_exists(engine):
    inspector = inspect(engine)
    order_indexes = {ix["name"] for ix in inspector.get_indexes("pending_orders")}
    event_indexes = {
        ix["name"] for ix in inspector.get_indexes("pending_order_events")
    }

    assert {
        "idx_pending_orders_state",
        "idx_pending_orders_symbol_state",
        "idx_pending_orders_profile_state",
        "idx_pending_orders_candidate",
        "idx_pending_orders_active_key",
    } <= order_indexes

    assert {
        "idx_pending_order_events_order",
        "idx_pending_order_events_type",
    } <= event_indexes


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_init_is_idempotent(engine):
    """Safe to run on every orchestrator startup."""
    init_pending_order_schema(engine)
    init_pending_order_schema(engine)

    inspector = inspect(engine)
    assert inspector.has_table("pending_orders")
    assert inspector.has_table("pending_order_events")


def test_init_preserves_existing_rows(engine):
    """Re-running the migration must not disturb live data."""
    order_id = _insert_order(engine)
    _insert_event(engine, order_id)

    init_pending_order_schema(engine)
    init_pending_order_schema(engine)

    with engine.connect() as conn:
        orders = conn.execute(text("SELECT COUNT(*) FROM pending_orders")).scalar()
        events = conn.execute(
            text("SELECT COUNT(*) FROM pending_order_events")
        ).scalar()
        limit_price = conn.execute(
            text("SELECT limit_price FROM pending_orders WHERE order_id = :oid"),
            {"oid": order_id},
        ).scalar()

    assert orders == 1
    assert events == 1
    assert limit_price == pytest.approx(593.87)


def test_repeated_init_does_not_duplicate_columns(engine):
    before = [c["name"] for c in inspect(engine).get_columns("pending_orders")]
    init_pending_order_schema(engine)
    after = [c["name"] for c in inspect(engine).get_columns("pending_orders")]

    assert before == after
    assert len(after) == len(set(after)), "duplicate column names after re-init"


# ---------------------------------------------------------------------------
# Immutability of the audit trail
# ---------------------------------------------------------------------------


def test_event_update_is_blocked(engine):
    order_id = _insert_order(engine)
    _insert_event(engine, order_id)

    with pytest.raises(Exception, match="immutable"):
        with engine.connect() as conn:
            conn.execute(
                text("UPDATE pending_order_events SET event_type = 'tampered'")
            )
            conn.commit()


def test_event_delete_is_blocked(engine):
    order_id = _insert_order(engine)
    _insert_event(engine, order_id)

    with pytest.raises(Exception, match="immutable"):
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM pending_order_events"))
            conn.commit()


def test_events_survive_a_blocked_mutation_attempt(engine):
    """A rejected UPDATE must leave the row intact, not partially applied."""
    order_id = _insert_order(engine)
    _insert_event(engine, order_id, event_type="state_pending")

    with pytest.raises(Exception, match="immutable"):
        with engine.connect() as conn:
            conn.execute(
                text("UPDATE pending_order_events SET event_type = 'tampered'")
            )
            conn.commit()

    with engine.connect() as conn:
        event_type = conn.execute(
            text("SELECT event_type FROM pending_order_events")
        ).scalar()
    assert event_type == "state_pending"


def test_pending_orders_itself_is_mutable(engine):
    """pending_orders is NOT immutable — CAS state transitions must work.

    Only the event trail is append-only. Guards against copy-pasting the
    immutability triggers onto the wrong table.
    """
    order_id = _insert_order(engine)

    with engine.connect() as conn:
        result = conn.execute(
            text(
                "UPDATE pending_orders SET state = 'filling' "
                "WHERE order_id = :oid AND state = 'pending'"
            ),
            {"oid": order_id},
        )
        conn.commit()
        assert result.rowcount == 1

        state = conn.execute(
            text("SELECT state FROM pending_orders WHERE order_id = :oid"),
            {"oid": order_id},
        ).scalar()
    assert state == "filling"


# ---------------------------------------------------------------------------
# The active-key uniqueness constraint (Requirement 7.4)
# ---------------------------------------------------------------------------


def test_two_active_orders_for_the_same_key_are_rejected(engine):
    """At most one ACTIVE order per (profile, symbol, side, setup_type)."""
    _insert_order(engine, state="pending")

    with pytest.raises(Exception):
        _insert_order(engine, state="pending")


def test_pending_and_filling_collide_on_the_same_key(engine):
    """'filling' is an active state, so it participates in the constraint."""
    _insert_order(engine, state="pending")

    with pytest.raises(Exception):
        _insert_order(engine, state="filling")


@pytest.mark.parametrize(
    "terminal_state", ["filled", "expired", "canceled", "rejected"]
)
def test_terminal_orders_do_not_block_a_new_active_order(engine, terminal_state):
    """History must accumulate freely — only active rows are constrained."""
    _insert_order(engine, state=terminal_state)
    _insert_order(engine, state=terminal_state)

    # A fresh active order for the same key is still allowed.
    order_id = _insert_order(engine, state="pending")

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM pending_orders")).scalar()
        active = conn.execute(
            text("SELECT COUNT(*) FROM pending_orders WHERE state = 'pending'")
        ).scalar()

    assert total == 3
    assert active == 1
    assert order_id is not None


@pytest.mark.parametrize(
    "differing_field,value",
    [
        ("symbol", "AMD"),
        ("side", "SHORT"),
        ("setup_type", "gap_and_go"),
        ("profile_id", "aggressive"),
    ],
)
def test_orders_differing_in_any_key_component_coexist(engine, differing_field, value):
    """The constraint is on the full 4-tuple, not any single column."""
    _insert_order(engine, state="pending")
    _insert_order(engine, state="pending", **{differing_field: value})

    with engine.connect() as conn:
        active = conn.execute(
            text("SELECT COUNT(*) FROM pending_orders WHERE state = 'pending'")
        ).scalar()
    assert active == 2
