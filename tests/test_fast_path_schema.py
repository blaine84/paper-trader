"""Tests for the fast_path_triggers and fast_path_events schema.

Validates:
- Both tables create without error (idempotent)
- INSERT succeeds on both tables
- UPDATE on fast_path_events core columns raises (immutability trigger)
- UPDATE on fast_path_events annotation columns succeeds (allowed)
- DELETE on fast_path_events raises (immutability trigger)
- fast_path_triggers is mutable (CAS state transitions work)

Requirements: 9.4
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, inspect, text

from db.schema import init_fast_path_events_schema, init_fast_path_triggers_schema

NOW = datetime(2026, 8, 20, 14, 30, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    init_fast_path_triggers_schema(eng)
    init_fast_path_events_schema(eng)
    return eng


def _trigger_row(**overrides) -> dict:
    """A minimally valid fast_path_triggers row."""
    row = {
        "trigger_id": str(uuid.uuid4()),
        "symbol": "TSLA",
        "profile_id": "moderate",
        "direction": "SHORT",
        "setup_type": "momentum_fade",
        "trigger_type": "entry_zone",
        "trigger_level": 351.61,
        "trigger_zone_upper": 352.00,
        "trigger_zone_lower": 350.50,
        "entry_price": 351.61,
        "stop_price": 355.00,
        "target_price": 348.97,
        "geometry_name": "short_momentum_fade",
        "source_signal_id": str(uuid.uuid4()),
        "source_watch_id": None,
        "invalidation_basis": "price above 355.00",
        "target_basis": "prior support",
        "state": "active",
        "registered_at": _iso(NOW),
        "expires_at": _iso(NOW + timedelta(seconds=300)),
        "fired_at": None,
        "resolved_at": None,
        "resolution_event_id": None,
        "signal_snapshot_json": '{"setup_type": "momentum_fade"}',
        "context_json": None,
    }
    row.update(overrides)
    return row


def _event_row(**overrides) -> dict:
    """A minimally valid fast_path_events row."""
    row = {
        "event_id": str(uuid.uuid4()),
        "cycle_id": "cycle_001",
        "candidate_id": None,
        "source_signal_id": str(uuid.uuid4()),
        "trigger_id": str(uuid.uuid4()),
        "symbol": "TSLA",
        "profile_id": "moderate",
        "setup_type": "momentum_fade",
        "direction": "SHORT",
        "entry_price": 351.61,
        "stop_price": 355.00,
        "target_price": 348.97,
        "current_price": 342.08,
        "reward_to_risk": 0.78,
        "outcome_type": "missed_move",
        "outcome_reason_code": "target_already_crossed",
        "outcome_metadata_json": None,
        "blocking_rule_name": None,
        "blocking_rule_threshold": None,
        "annotation_status": "annotation_pending",
        "annotation_json": None,
        "annotation_timestamp": None,
        "narration": "TSLA broke 351.61, but the move had already crossed target; no order created.",
        "narration_source": "template",
        "evaluated_at": _iso(NOW),
        "market_data_age_ms": 1200,
        "evaluation_duration_ms": 45,
    }
    row.update(overrides)
    return row


def _insert_trigger(engine, **overrides) -> str:
    row = _trigger_row(**overrides)
    columns = ", ".join(row.keys())
    placeholders = ", ".join(f":{k}" for k in row)
    with engine.connect() as conn:
        conn.execute(
            text(f"INSERT INTO fast_path_triggers ({columns}) VALUES ({placeholders})"),
            row,
        )
        conn.commit()
    return row["trigger_id"]


def _insert_event(engine, **overrides) -> str:
    row = _event_row(**overrides)
    columns = ", ".join(row.keys())
    placeholders = ", ".join(f":{k}" for k in row)
    with engine.connect() as conn:
        conn.execute(
            text(f"INSERT INTO fast_path_events ({columns}) VALUES ({placeholders})"),
            row,
        )
        conn.commit()
    return row["event_id"]


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------


def test_init_creates_both_tables(engine):
    inspector = inspect(engine)
    assert inspector.has_table("fast_path_triggers")
    assert inspector.has_table("fast_path_events")


def test_init_is_idempotent(engine):
    """Safe to run on every orchestrator startup."""
    init_fast_path_triggers_schema(engine)
    init_fast_path_events_schema(engine)
    init_fast_path_triggers_schema(engine)
    init_fast_path_events_schema(engine)

    inspector = inspect(engine)
    assert inspector.has_table("fast_path_triggers")
    assert inspector.has_table("fast_path_events")


def test_init_preserves_existing_rows(engine):
    """Re-running the migration must not disturb live data."""
    trigger_id = _insert_trigger(engine)
    _insert_event(engine, trigger_id=trigger_id)

    init_fast_path_triggers_schema(engine)
    init_fast_path_events_schema(engine)

    with engine.connect() as conn:
        triggers = conn.execute(text("SELECT COUNT(*) FROM fast_path_triggers")).scalar()
        events = conn.execute(text("SELECT COUNT(*) FROM fast_path_events")).scalar()

    assert triggers == 1
    assert events == 1


# ---------------------------------------------------------------------------
# INSERT succeeds
# ---------------------------------------------------------------------------


def test_insert_trigger_succeeds(engine):
    trigger_id = _insert_trigger(engine)

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM fast_path_triggers")).scalar()
        state = conn.execute(
            text("SELECT state FROM fast_path_triggers WHERE trigger_id = :tid"),
            {"tid": trigger_id},
        ).scalar()

    assert count == 1
    assert state == "active"


def test_insert_event_succeeds(engine):
    event_id = _insert_event(engine)

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM fast_path_events")).scalar()
        outcome = conn.execute(
            text("SELECT outcome_type FROM fast_path_events WHERE event_id = :eid"),
            {"eid": event_id},
        ).scalar()

    assert count == 1
    assert outcome == "missed_move"


def test_insert_multiple_events_succeeds(engine):
    """Multiple fast-path events can coexist."""
    _insert_event(engine, outcome_type="missed_move")
    _insert_event(engine, outcome_type="stand_down", outcome_reason_code="stale_market_data")
    _insert_event(engine, outcome_type="trade_executed", outcome_reason_code="gates_passed")

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM fast_path_events")).scalar()

    assert count == 3


# ---------------------------------------------------------------------------
# Immutability of fast_path_events (UPDATE blocked)
# ---------------------------------------------------------------------------


def test_event_update_is_blocked(engine):
    _insert_event(engine)

    with pytest.raises(Exception, match="immutable"):
        with engine.connect() as conn:
            conn.execute(
                text("UPDATE fast_path_events SET outcome_type = 'tampered'")
            )
            conn.commit()


def test_event_annotation_update_allowed(engine):
    """Annotation columns (annotation_status, annotation_json, annotation_timestamp,
    narration, narration_source, outcome_metadata_json) may be updated."""
    event_id = _insert_event(engine)

    with engine.connect() as conn:
        conn.execute(
            text(
                "UPDATE fast_path_events SET "
                "annotation_status = 'annotated', "
                "annotation_json = :payload, "
                "annotation_timestamp = :ts, "
                "narration = 'LLM enriched narration', "
                "narration_source = 'llm_enriched', "
                "outcome_metadata_json = '{\"detail\": \"extra\"}' "
                "WHERE event_id = :eid"
            ),
            {
                "eid": event_id,
                "payload": '{"thesis": "momentum exhaustion"}',
                "ts": _iso(NOW),
            },
        )
        conn.commit()

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT annotation_status, annotation_json, annotation_timestamp, "
                "narration, narration_source, outcome_metadata_json "
                "FROM fast_path_events WHERE event_id = :eid"
            ),
            {"eid": event_id},
        ).mappings().one()

    assert row["annotation_status"] == "annotated"
    assert row["annotation_json"] == '{"thesis": "momentum exhaustion"}'
    assert row["annotation_timestamp"] == _iso(NOW)
    assert row["narration"] == "LLM enriched narration"
    assert row["narration_source"] == "llm_enriched"
    assert row["outcome_metadata_json"] == '{"detail": "extra"}'


def test_events_survive_blocked_update(engine):
    """A rejected UPDATE must leave the row intact."""
    event_id = _insert_event(engine, outcome_type="missed_move")

    with pytest.raises(Exception, match="immutable"):
        with engine.connect() as conn:
            conn.execute(
                text("UPDATE fast_path_events SET outcome_type = 'tampered'")
            )
            conn.commit()

    with engine.connect() as conn:
        outcome = conn.execute(
            text("SELECT outcome_type FROM fast_path_events WHERE event_id = :eid"),
            {"eid": event_id},
        ).scalar()
    assert outcome == "missed_move"


# ---------------------------------------------------------------------------
# Immutability of fast_path_events (DELETE blocked)
# ---------------------------------------------------------------------------


def test_event_delete_is_blocked(engine):
    _insert_event(engine)

    with pytest.raises(Exception, match="immutable"):
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM fast_path_events"))
            conn.commit()


def test_events_survive_blocked_delete(engine):
    """A rejected DELETE must leave the row intact."""
    _insert_event(engine)

    with pytest.raises(Exception, match="immutable"):
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM fast_path_events"))
            conn.commit()

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM fast_path_events")).scalar()
    assert count == 1


# ---------------------------------------------------------------------------
# fast_path_triggers is mutable (CAS state transitions must work)
# ---------------------------------------------------------------------------


def test_trigger_update_succeeds(engine):
    """fast_path_triggers is NOT immutable — CAS state transitions must work."""
    trigger_id = _insert_trigger(engine)

    with engine.connect() as conn:
        result = conn.execute(
            text(
                "UPDATE fast_path_triggers SET state = 'fired', "
                "fired_at = :now, resolved_at = :now "
                "WHERE trigger_id = :tid AND state = 'active'"
            ),
            {"tid": trigger_id, "now": _iso(NOW)},
        )
        conn.commit()
        assert result.rowcount == 1

        state = conn.execute(
            text("SELECT state FROM fast_path_triggers WHERE trigger_id = :tid"),
            {"tid": trigger_id},
        ).scalar()
    assert state == "fired"


def test_trigger_delete_succeeds(engine):
    """fast_path_triggers allows DELETE (though not used in production, no trigger blocks it)."""
    trigger_id = _insert_trigger(engine)

    with engine.connect() as conn:
        conn.execute(
            text("DELETE FROM fast_path_triggers WHERE trigger_id = :tid"),
            {"tid": trigger_id},
        )
        conn.commit()
        count = conn.execute(text("SELECT COUNT(*) FROM fast_path_triggers")).scalar()
    assert count == 0


# ---------------------------------------------------------------------------
# Indexes exist
# ---------------------------------------------------------------------------


def test_trigger_indexes_exist(engine):
    inspector = inspect(engine)
    indexes = {ix["name"] for ix in inspector.get_indexes("fast_path_triggers")}
    assert {"idx_fpt_state", "idx_fpt_symbol_state", "idx_fpt_expires"} <= indexes


def test_event_indexes_exist(engine):
    inspector = inspect(engine)
    indexes = {ix["name"] for ix in inspector.get_indexes("fast_path_events")}
    assert {
        "idx_fpe_symbol_outcome",
        "idx_fpe_profile_outcome",
        "idx_fpe_trigger",
        "idx_fpe_evaluated",
    } <= indexes
