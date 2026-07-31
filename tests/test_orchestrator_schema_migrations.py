from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, inspect, text

import orchestrator


def test_check_schema_adds_exit_category_to_existing_cases_table():
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE cases (id INTEGER PRIMARY KEY, symbol VARCHAR(10))"))

    orchestrator.check_schema(engine)

    columns = {column["name"] for column in inspect(engine).get_columns("cases")}
    assert "exit_category" in columns


def test_check_schema_initializes_replay_lineage_columns():
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE trades (id INTEGER PRIMARY KEY)"))
        conn.execute(
            text(
                "CREATE TABLE trade_events ("
                "id INTEGER PRIMARY KEY, "
                "event_type VARCHAR(64), "
                "trade_id INTEGER"
                ")"
            )
        )
        conn.execute(text("CREATE TABLE funnel_candidates (id INTEGER PRIMARY KEY)"))
        conn.execute(text("CREATE TABLE blocked_trade_candidates (id INTEGER PRIMARY KEY)"))
        conn.execute(text("CREATE TABLE pm_candidates (id INTEGER PRIMARY KEY)"))

    orchestrator.check_schema(engine)

    inspector = inspect(engine)
    assert "candidate_lineage_id" in {
        column["name"] for column in inspector.get_columns("trades")
    }
    assert "candidate_lineage_id" in {
        column["name"] for column in inspector.get_columns("trade_events")
    }


def test_check_schema_creates_candidate_events_with_generated_id():
    engine = create_engine("sqlite://")

    orchestrator.check_schema(engine)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO pm_candidate_events
                (candidate_id, cycle_id, profile_id, event_type, event_data)
                VALUES ('', 'cycle-1', 'moderate', 'swing_no_candidates', '{}')
                """
            )
        )
        row = conn.execute(
            text(
                """
                SELECT id, candidate_type
                FROM pm_candidate_events
                WHERE cycle_id = 'cycle-1'
                """
            )
        ).one()

    assert row.id is not None
    assert row.candidate_type == "intraday"


def test_check_schema_repairs_decision_snapshot_identity_default(monkeypatch):
    engine = create_engine("sqlite://")
    repaired_tables = []

    def record_identity_repair(_engine, _inspector, table_name):
        repaired_tables.append(table_name)

    monkeypatch.setattr(
        orchestrator,
        "_ensure_postgres_identity_default",
        record_identity_repair,
    )

    orchestrator.check_schema(engine)

    assert "decision_snapshots" in repaired_tables


# ---------------------------------------------------------------------------
# Postgres identity/sequence-default repair coverage
#
# Every table below is declared with a bare `id INTEGER PRIMARY KEY`, which
# auto-assigns on SQLite (rowid alias) but leaves Postgres with no default,
# so INSERTs that omit `id` fail with a NOT NULL violation.
# ---------------------------------------------------------------------------

IDENTITY_REPAIR_TABLES = [
    # already wired before this change
    "pm_candidates",
    "pm_candidate_events",
    "decision_snapshots",
    # provenance namespace
    "pm_raw_responses",
    "response_lineage_links",
    "provenance_events",
    "provenance_findings",
    # replay namespace
    "replay_audit_records",
    "replay_batch_runs",
    "replay_batch_items",
    "replay_annotations",
    "replay_counterfactual_outcomes",
    # triggered trade plans
    "trade_plan_events",
]


@pytest.mark.parametrize("table_name", IDENTITY_REPAIR_TABLES)
def test_check_schema_wires_identity_repair_for_table(monkeypatch, table_name):
    """check_schema() must run the identity repair for every bare-id table."""
    engine = create_engine("sqlite://")
    repaired_tables = []

    def record_identity_repair(_engine, _inspector, name):
        repaired_tables.append(name)

    monkeypatch.setattr(
        orchestrator,
        "_ensure_postgres_identity_default",
        record_identity_repair,
    )

    orchestrator.check_schema(engine)

    assert table_name in repaired_tables


@pytest.mark.parametrize("table_name", IDENTITY_REPAIR_TABLES)
def test_identity_repair_runs_after_table_exists(monkeypatch, table_name):
    """The repair must be called with an inspector that already sees the table.

    Ordering bug guard: calling the repair before the corresponding init
    function would silently no-op on Postgres (has_table() returns False).
    """
    engine = create_engine("sqlite://")
    seen = {}

    def record_identity_repair(_engine, inspector, name):
        seen[name] = inspector.has_table(name)

    monkeypatch.setattr(
        orchestrator,
        "_ensure_postgres_identity_default",
        record_identity_repair,
    )

    orchestrator.check_schema(engine)

    assert seen.get(table_name) is True, (
        f"identity repair for {table_name} ran before the table existed"
    )


def _make_mock_postgres_engine():
    """Mock engine reporting the postgresql dialect; returns (engine, conn)."""
    engine = MagicMock()
    engine.dialect.name = "postgresql"
    conn = MagicMock()
    engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)
    return engine, conn


def _executed_sql(conn):
    return [str(c[0][0]) for c in conn.execute.call_args_list if c[0]]


@pytest.mark.parametrize("table_name", IDENTITY_REPAIR_TABLES)
def test_identity_repair_sets_sequence_default_on_postgres(table_name):
    """On Postgres, a bare integer id gets a sequence + DEFAULT nextval."""
    engine, conn = _make_mock_postgres_engine()
    inspector = MagicMock()
    inspector.has_table.return_value = True
    inspector.get_columns.return_value = [
        {"name": "id", "default": None},
        {"name": "created_at", "default": "CURRENT_TIMESTAMP"},
    ]

    orchestrator._ensure_postgres_identity_default(engine, inspector, table_name)

    sql = " ".join(_executed_sql(conn))
    sequence = f"{table_name}_id_seq"
    assert f"CREATE SEQUENCE IF NOT EXISTS {sequence}" in sql
    assert f"OWNED BY {table_name}.id" in sql
    assert f"setval('{sequence}'" in sql
    assert f"SELECT MAX(id) FROM {table_name}" in sql
    assert (
        f"ALTER TABLE {table_name} ALTER COLUMN id SET DEFAULT nextval('{sequence}')"
        in sql
    )
    # Non-destructive: never drops or rewrites data.
    upper = sql.upper()
    assert "DROP TABLE" not in upper
    assert "DELETE FROM" not in upper
    assert "UPDATE " not in upper


def test_identity_repair_is_noop_when_default_already_present():
    """Idempotent: a column that already has a default is left alone."""
    engine, conn = _make_mock_postgres_engine()
    inspector = MagicMock()
    inspector.has_table.return_value = True
    inspector.get_columns.return_value = [
        {"name": "id", "default": "nextval('response_lineage_links_id_seq'::regclass)"},
    ]

    orchestrator._ensure_postgres_identity_default(
        engine, inspector, "response_lineage_links"
    )

    assert conn.execute.call_args_list == []


def test_identity_repair_is_noop_on_sqlite():
    engine = create_engine("sqlite://")
    inspector = MagicMock()
    inspector.has_table.return_value = True

    orchestrator._ensure_postgres_identity_default(
        engine, inspector, "response_lineage_links"
    )

    # SQLite short-circuits before touching the inspector's columns.
    assert inspector.get_columns.call_args_list == []


def test_identity_repair_is_noop_when_table_absent():
    engine, conn = _make_mock_postgres_engine()
    inspector = MagicMock()
    inspector.has_table.return_value = False

    orchestrator._ensure_postgres_identity_default(
        engine, inspector, "trade_plan_events"
    )

    assert conn.execute.call_args_list == []


# ---------------------------------------------------------------------------
# SQLite regression guard: every affected table accepts an INSERT that
# omits `id`. Rows are inserted in FK-dependency order.
# ---------------------------------------------------------------------------

def test_affected_tables_accept_insert_without_explicit_id():
    engine = create_engine("sqlite://")
    orchestrator.check_schema(engine)

    now = "2024-01-15 10:00:00"
    response_id = "resp-1"
    replay_id = "replay-1"
    batch_run_id = "batch-1"

    inserts = [
        (
            "pm_raw_responses",
            """
            INSERT INTO pm_raw_responses (
                response_id, pm_cycle_id, profile, model_id, timestamp,
                prompt_version_id, candidate_ids_supplied_json,
                original_payload_hash, parse_status, payload_size_bytes
            ) VALUES (
                :response_id, 'cycle-1', 'moderate', 'model-x', :now,
                'v1', '[]', 'hash-1', 'ok', 10
            )
            """,
        ),
        (
            "response_lineage_links",
            """
            INSERT INTO response_lineage_links (response_id, lineage_id, candidate_id)
            VALUES (:response_id, 'lineage-1', 'cand-1')
            """,
        ),
        (
            "provenance_events",
            """
            INSERT INTO provenance_events (
                lineage_id, stage_name, stage_version, sequence_number,
                timestamp, mutation_reason_code, geometry_before_json,
                geometry_after_json, validation_before, validation_after
            ) VALUES (
                'lineage-1', 'gates', 'v1', 1, :now, 'none', '{}', '{}',
                'valid', 'valid'
            )
            """,
        ),
        (
            "provenance_findings",
            """
            INSERT INTO provenance_findings (
                finding_id, lineage_id, finding_type, stage_name,
                severity, details_json
            ) VALUES ('finding-1', 'lineage-1', 'mismatch', 'gates', 'warn', '{}')
            """,
        ),
        (
            "decision_snapshots",
            """
            INSERT INTO decision_snapshots (
                snapshot_id, schema_version, candidate_lineage_id, timestamp,
                symbol, profile, direction, decision_payload_json, entry_price,
                stop_price, target_price, quantity, account_equity,
                available_cash, gate_config_json, feature_flags_json,
                policy_version_id
            ) VALUES (
                'snap-1', '1', 'lineage-1', :now, 'AAPL', 'moderate', 'long',
                '{}', '100', '95', '110', '10', '10000', '5000', '{}', '{}', 'p1'
            )
            """,
        ),
        (
            "replay_audit_records",
            """
            INSERT INTO replay_audit_records (
                replay_id, candidate_id, source_candidate_ids_json,
                replay_cutoff, input_sources_json, policy_version_json,
                replay_status, era
            ) VALUES (
                :replay_id, 'cand-1', '[]', :now, '{}', '{}', 'exact', 'modern'
            )
            """,
        ),
        (
            "replay_annotations",
            """
            INSERT INTO replay_annotations (
                replay_id, author, annotation_timestamp, content
            ) VALUES (:replay_id, 'op', :now, 'note')
            """,
        ),
        (
            "replay_counterfactual_outcomes",
            """
            INSERT INTO replay_counterfactual_outcomes (
                replay_id, candidate_id, direction, proposed_entry_price,
                simulated_fill_price, fill_rule, stop_price, target_price, status
            ) VALUES (
                :replay_id, 'cand-1', 'long', '100', '100.1', 'next_open',
                '95', '110', 'scored'
            )
            """,
        ),
        (
            "replay_batch_runs",
            """
            INSERT INTO replay_batch_runs (
                batch_run_id, started_at, mode, policy_version_json, status
            ) VALUES (:batch_run_id, :now, 'batch', '{}', 'running')
            """,
        ),
        (
            "replay_batch_items",
            """
            INSERT INTO replay_batch_items (
                batch_run_id, candidate_id, processing_order
            ) VALUES (:batch_run_id, 'cand-1', 1)
            """,
        ),
        (
            "trade_plan_events",
            """
            INSERT INTO trade_plan_events (
                plan_id, cycle_id, profile_id, event_type, created_at
            ) VALUES ('plan-1', 'cycle-1', 'moderate', 'plan_created', :now)
            """,
        ),
        (
            "pm_candidate_events",
            """
            INSERT INTO pm_candidate_events (
                candidate_id, cycle_id, profile_id, event_type, event_data
            ) VALUES ('cand-1', 'cycle-1', 'moderate', 'registered', '{}')
            """,
        ),
    ]

    params = {
        "now": now,
        "response_id": response_id,
        "replay_id": replay_id,
        "batch_run_id": batch_run_id,
    }

    with engine.begin() as conn:
        for table, sql in inserts:
            conn.execute(text(sql), params)
            generated_id = conn.execute(
                text(f"SELECT id FROM {table} ORDER BY id DESC LIMIT 1")
            ).scalar()
            assert generated_id is not None, f"{table}.id was not auto-assigned"
            assert generated_id >= 1
