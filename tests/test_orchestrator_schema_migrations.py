from sqlalchemy import create_engine, inspect, text

import orchestrator


def test_check_schema_adds_exit_category_to_existing_cases_table():
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE cases (id INTEGER PRIMARY KEY, symbol VARCHAR(10))"))

    orchestrator.check_schema(engine)

    columns = {column["name"] for column in inspect(engine).get_columns("cases")}
    assert "exit_category" in columns
