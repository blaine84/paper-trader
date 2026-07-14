"""Unit tests for dialect-aware SQL helpers in utils/dialect_sql.py.

Validates Requirements 3.2, 3.3 — dialect-specific SQL patterns produce
correct output for both SQLite and Postgres engines.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from utils.dialect_sql import (
    _date_cutoff_filter,
    _default_timestamp,
    _json_field,
    _pk_column,
    _upsert_outcome_sql,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_sqlite_engine():
    engine = MagicMock()
    engine.dialect.name = "sqlite"
    return engine


def _make_pg_engine():
    engine = MagicMock()
    engine.dialect.name = "postgresql"
    return engine


@pytest.fixture
def sqlite_engine():
    return _make_sqlite_engine()


@pytest.fixture
def pg_engine():
    return _make_pg_engine()


# ── _date_cutoff_filter tests ───────────────────────────────────────────────


class TestDateCutoffFilter:
    def test_sqlite_default_param(self, sqlite_engine):
        result = _date_cutoff_filter(sqlite_engine, "b.created_at")
        assert result == "datetime(b.created_at) >= datetime('now', :cutoff)"

    def test_postgres_default_param(self, pg_engine):
        result = _date_cutoff_filter(pg_engine, "b.created_at")
        assert result == "b.created_at >= NOW() + CAST(:cutoff AS interval)"

    def test_custom_param_name_sqlite(self, sqlite_engine):
        result = _date_cutoff_filter(sqlite_engine, "t.updated_at", param_name="window")
        assert result == "datetime(t.updated_at) >= datetime('now', :window)"

    def test_custom_param_name_postgres(self, pg_engine):
        result = _date_cutoff_filter(pg_engine, "t.updated_at", param_name="window")
        assert result == "t.updated_at >= NOW() + CAST(:window AS interval)"


# ── _json_field tests ───────────────────────────────────────────────────────


class TestJsonField:
    def test_sqlite(self, sqlite_engine):
        result = _json_field(sqlite_engine, "o.notes_json", "key")
        assert result == "json_extract(o.notes_json, '$.key')"

    def test_postgres(self, pg_engine):
        result = _json_field(pg_engine, "o.notes_json", "key")
        assert result == "o.notes_json::jsonb->>'key'"

    def test_nested_key_sqlite(self, sqlite_engine):
        result = _json_field(sqlite_engine, "data", "nested.field")
        assert result == "json_extract(data, '$.nested.field')"

    def test_nested_key_postgres(self, pg_engine):
        result = _json_field(pg_engine, "data", "nested.field")
        assert result == "data::jsonb->>'nested.field'"


# ── _upsert_outcome_sql tests ──────────────────────────────────────────────


class TestUpsertOutcomeSql:
    def test_contains_on_conflict(self):
        sql = _upsert_outcome_sql()
        assert "ON CONFLICT" in sql

    def test_contains_do_nothing(self):
        sql = _upsert_outcome_sql()
        assert "DO NOTHING" in sql

    def test_contains_insert_into(self):
        sql = _upsert_outcome_sql()
        assert "INSERT INTO blocked_trade_candidate_outcomes" in sql

    def test_contains_conflict_columns(self):
        sql = _upsert_outcome_sql()
        assert "(blocked_candidate_id, eval_window)" in sql


# ── _pk_column tests ────────────────────────────────────────────────────────


class TestPkColumn:
    def test_sqlite(self, sqlite_engine):
        result = _pk_column(sqlite_engine)
        assert result == "id INTEGER PRIMARY KEY AUTOINCREMENT"

    def test_postgres(self, pg_engine):
        result = _pk_column(pg_engine)
        assert result == "id SERIAL PRIMARY KEY"


# ── _default_timestamp tests ────────────────────────────────────────────────


class TestDefaultTimestamp:
    def test_sqlite(self, sqlite_engine):
        result = _default_timestamp(sqlite_engine)
        assert "strftime" in result
        assert "DEFAULT" in result

    def test_postgres(self, pg_engine):
        result = _default_timestamp(pg_engine)
        assert result == "DEFAULT CURRENT_TIMESTAMP"
