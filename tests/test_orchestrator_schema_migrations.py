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
