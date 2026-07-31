"""Tests for the triggered-trade-plan schema DDL in db/schema.py."""

import pytest
from sqlalchemy import create_engine, inspect, text

from db.schema import init_trade_plan_schema


EXPECTED_TRADE_PLAN_COLUMNS = {
    "plan_id",
    "candidate_id",
    "cycle_id",
    "profile_id",
    "symbol",
    "direction",
    "setup_type",
    "geometry_name",
    "entry_reference",
    "entry_zone_upper",
    "entry_zone_lower",
    "stop_price",
    "target_price",
    "risk_reward",
    "trigger_type",
    "trigger_condition_json",
    "trigger_confirmation_required",
    "invalidation_logic_json",
    "analyst_reasoning",
    "pm_rationale",
    "source_signal_id",
    "signal_snapshot_json",
    "state",
    "created_at",
    "expires_at",
    "triggered_at",
    "executed_at",
    "missed_at",
    "miss_reason",
    "rejection_reason",
    "integrity_hash",
}


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    init_trade_plan_schema(eng)
    return eng


def test_trade_plans_table_created(engine):
    assert inspect(engine).has_table("trade_plans")


def test_trade_plans_has_all_expected_columns(engine):
    columns = {col["name"] for col in inspect(engine).get_columns("trade_plans")}
    assert EXPECTED_TRADE_PLAN_COLUMNS == columns


def test_trade_plans_indexes_created(engine):
    index_names = {idx["name"] for idx in inspect(engine).get_indexes("trade_plans")}
    assert {
        "idx_trade_plans_state",
        "idx_trade_plans_symbol_state",
        "idx_trade_plans_candidate",
        "idx_trade_plans_cycle",
    } <= index_names


def test_init_is_idempotent(engine):
    # Second (and third) run must not raise — safe on every startup.
    init_trade_plan_schema(engine)
    init_trade_plan_schema(engine)
    assert inspect(engine).has_table("trade_plans")


def test_init_preserves_existing_rows(engine):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO trade_plans (
                    plan_id, candidate_id, cycle_id, profile_id, symbol,
                    direction, setup_type, entry_reference, entry_zone_upper,
                    entry_zone_lower, stop_price, target_price, risk_reward,
                    trigger_type, trigger_condition_json, created_at,
                    expires_at, integrity_hash
                ) VALUES (
                    'plan-1', 'cand-1', 'cycle-1', 'aggressive', 'TSLA',
                    'LONG', 'momentum_fade', 250.0, 252.0,
                    248.0, 245.0, 260.0, 2.0,
                    'price_in_zone', '{}', '2026-05-25T14:00:00',
                    '2026-05-25T15:00:00', 'hash-1'
                )
                """
            )
        )

    init_trade_plan_schema(engine)

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT state, symbol FROM trade_plans WHERE plan_id = 'plan-1'")
        ).one()
    assert row.state == "planned"
    assert row.symbol == "TSLA"


def test_state_defaults_to_planned(engine):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO trade_plans (
                    plan_id, candidate_id, cycle_id, profile_id, symbol,
                    direction, setup_type, entry_reference, entry_zone_upper,
                    entry_zone_lower, stop_price, target_price, risk_reward,
                    trigger_type, trigger_condition_json, created_at,
                    expires_at, integrity_hash
                ) VALUES (
                    'plan-2', 'cand-2', 'cycle-1', 'moderate', 'MU',
                    'SHORT', 'breakdown_retest', 782.0, 784.0,
                    780.0, 790.0, 760.0, 2.2,
                    'price_in_zone', '{}', '2026-05-25T14:00:00',
                    '2026-05-25T15:00:00', 'hash-2'
                )
                """
            )
        )

    with engine.connect() as conn:
        state = conn.execute(
            text("SELECT state FROM trade_plans WHERE plan_id = 'plan-2'")
        ).scalar()
        confirmation = conn.execute(
            text(
                "SELECT trigger_confirmation_required FROM trade_plans "
                "WHERE plan_id = 'plan-2'"
            )
        ).scalar()
    assert state == "planned"
    assert confirmation == 0

EXPECTED_TRADE_PLAN_EVENT_COLUMNS = {
    "id",
    "plan_id",
    "cycle_id",
    "profile_id",
    "event_type",
    "event_data",
    "fresh_price",
    "from_state",
    "to_state",
    "created_at",
}


def _insert_event(engine, event_id_note="evt"):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO trade_plan_events (
                    plan_id, cycle_id, profile_id, event_type, event_data,
                    fresh_price, from_state, to_state, created_at
                ) VALUES (
                    'plan-1', 'cycle-1', 'aggressive', :event_type,
                    '{"reason": "triggered"}', 249.5, 'watching', 'triggered',
                    '2026-05-25T14:05:00'
                )
                """
            ),
            {"event_type": event_id_note},
        )


def test_trade_plan_events_table_created(engine):
    assert inspect(engine).has_table("trade_plan_events")


def test_trade_plan_events_has_all_expected_columns(engine):
    columns = {
        col["name"] for col in inspect(engine).get_columns("trade_plan_events")
    }
    assert EXPECTED_TRADE_PLAN_EVENT_COLUMNS == columns


def test_trade_plan_events_indexes_created(engine):
    index_names = {
        idx["name"] for idx in inspect(engine).get_indexes("trade_plan_events")
    }
    assert {
        "idx_trade_plan_events_plan",
        "idx_trade_plan_events_type",
    } <= index_names


def test_trade_plan_events_insert_succeeds(engine):
    _insert_event(engine, "triggered")

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT plan_id, event_type, fresh_price, from_state, to_state "
                "FROM trade_plan_events WHERE plan_id = 'plan-1'"
            )
        ).one()
    assert row.event_type == "triggered"
    assert row.fresh_price == 249.5
    assert row.from_state == "watching"
    assert row.to_state == "triggered"


def test_trade_plan_events_update_blocked(engine):
    _insert_event(engine, "triggered")

    with pytest.raises(Exception, match="immutable"):
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE trade_plan_events SET event_type = 'tampered'")
            )


def test_trade_plan_events_delete_blocked(engine):
    _insert_event(engine, "triggered")

    with pytest.raises(Exception, match="immutable"):
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM trade_plan_events"))


def test_trade_plan_events_init_is_idempotent_and_preserves_rows(engine):
    _insert_event(engine, "triggered")

    init_trade_plan_schema(engine)
    init_trade_plan_schema(engine)

    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM trade_plan_events")
        ).scalar()
    assert count == 1


# ---------------------------------------------------------------------------
# Startup migration path: check_schema() must create the trade plan tables so
# a live database is upgraded automatically on restart.
#
# Requirements: 1.8, 9.4
# ---------------------------------------------------------------------------


def test_check_schema_creates_trade_plan_tables(tmp_path):
    from db.schema import init_db
    from orchestrator import check_schema

    db_path = tmp_path / "startup.db"
    eng = init_db(str(db_path))

    inspector = inspect(eng)
    assert not inspector.has_table("trade_plans")
    assert not inspector.has_table("trade_plan_events")

    check_schema(eng)

    inspector = inspect(eng)
    assert inspector.has_table("trade_plans")
    assert inspector.has_table("trade_plan_events")

    columns = {col["name"] for col in inspector.get_columns("trade_plans")}
    assert EXPECTED_TRADE_PLAN_COLUMNS == columns
    event_columns = {
        col["name"] for col in inspector.get_columns("trade_plan_events")
    }
    assert EXPECTED_TRADE_PLAN_EVENT_COLUMNS == event_columns


def test_check_schema_is_idempotent_and_preserves_plan_rows(tmp_path):
    from db.schema import init_db
    from orchestrator import check_schema

    db_path = tmp_path / "startup_idempotent.db"
    eng = init_db(str(db_path))

    check_schema(eng)

    with eng.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO trade_plans (
                    plan_id, candidate_id, cycle_id, profile_id, symbol,
                    direction, setup_type, entry_reference, entry_zone_upper,
                    entry_zone_lower, stop_price, target_price, risk_reward,
                    trigger_type, trigger_condition_json, created_at,
                    expires_at, integrity_hash
                ) VALUES (
                    'plan-startup', 'cand-startup', 'cycle-1', 'aggressive',
                    'TSLA', 'LONG', 'momentum_fade', 250.0, 252.0,
                    248.0, 245.0, 260.0, 2.0,
                    'price_in_zone', '{}', '2026-05-25T14:00:00',
                    '2026-05-25T15:00:00', 'hash-startup'
                )
                """
            )
        )

    # Second startup must not raise and must not touch existing rows.
    check_schema(eng)

    with eng.connect() as conn:
        row = conn.execute(
            text(
                "SELECT state, symbol FROM trade_plans "
                "WHERE plan_id = 'plan-startup'"
            )
        ).one()
    assert row.state == "planned"
    assert row.symbol == "TSLA"
