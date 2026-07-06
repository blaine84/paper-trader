"""Dialect-aware SQL helpers for Postgres/SQLite compatibility.

Provides helper functions that emit the correct SQL fragments depending
on the engine dialect (SQLite vs Postgres). These are internal helpers
used by runtime modules that contain raw text() queries needing
dialect-conditional syntax.
"""

from __future__ import annotations

from sqlalchemy.engine import Engine

from db.schema import is_sqlite


def _date_cutoff_filter(engine: Engine, column: str, param_name: str = "cutoff") -> str:
    """Return a WHERE clause fragment for date cutoff filtering.

    SQLite: datetime({column}) >= datetime('now', :{param_name})
    Postgres: {column} >= NOW() + :{param_name}::interval
    """
    if is_sqlite(engine):
        return f"datetime({column}) >= datetime('now', :{param_name})"
    else:
        return f"{column} >= NOW() + :{param_name}::interval"


def _json_field(engine: Engine, column: str, key: str) -> str:
    """Return SQL expression to extract a text value from a JSON column.

    SQLite: json_extract({column}, '$.{key}')
    Postgres: {column}::jsonb->>'{key}'
    """
    if is_sqlite(engine):
        return f"json_extract({column}, '$.{key}')"
    else:
        return f"{column}::jsonb->>'{key}'"


def _upsert_outcome_sql() -> str:
    """Return INSERT ... ON CONFLICT DO NOTHING SQL (works on both dialects).

    Uses ON CONFLICT (blocked_candidate_id, eval_window) DO NOTHING which
    is supported by both SQLite >=3.24 and Postgres.
    """
    return """
        INSERT INTO blocked_trade_candidate_outcomes (
            blocked_candidate_id, eval_window, evaluated_at, eval_price,
            pnl_pct, mfe_pct, mae_pct, stop_hit, target_hit, first_hit,
            first_hit_at, outcome_label, gate_verdict, notes_json
        ) VALUES (
            :blocked_candidate_id, :eval_window, :evaluated_at, :eval_price,
            :pnl_pct, :mfe_pct, :mae_pct, :stop_hit, :target_hit, :first_hit,
            :first_hit_at, :outcome_label, :gate_verdict, :notes_json
        ) ON CONFLICT (blocked_candidate_id, eval_window) DO NOTHING
    """


def _pk_column(engine: Engine) -> str:
    """Return primary key column definition appropriate for the dialect.

    SQLite: id INTEGER PRIMARY KEY AUTOINCREMENT
    Postgres: id SERIAL PRIMARY KEY
    """
    if is_sqlite(engine):
        return "id INTEGER PRIMARY KEY AUTOINCREMENT"
    else:
        return "id SERIAL PRIMARY KEY"


def _default_timestamp(engine: Engine) -> str:
    """Return DEFAULT expression for timestamp columns.

    SQLite: DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    Postgres: DEFAULT CURRENT_TIMESTAMP
    """
    if is_sqlite(engine):
        return "DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
    else:
        return "DEFAULT CURRENT_TIMESTAMP"
