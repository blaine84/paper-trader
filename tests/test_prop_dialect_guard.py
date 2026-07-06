"""
Property-based test for dialect guard (Property 3).

Property 3: SQLite-Only DDL Guarded by Dialect Check

Verifies that running check_schema(), init_provenance_schema(),
init_alert_dispatch_schema(), and init_replay_db() against a Postgres engine
executes no PRAGMA statements or SQLite-specific DDL.

**Validates: Requirements 3.4**
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mock_postgres_engine():
    """Create a mock engine that reports as postgresql dialect.

    Returns (engine, conn) where conn captures all executed SQL via
    conn.execute.call_args_list.
    """
    engine = MagicMock()
    engine.dialect.name = "postgresql"

    # Set up connect() context manager
    conn = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    # Set up begin() context manager
    engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    return engine, conn


def _extract_sql_strings(conn_mock) -> list[str]:
    """Extract all SQL strings from a mock connection's execute calls."""
    sql_strings = []
    for call_args in conn_mock.execute.call_args_list:
        if call_args[0]:
            sql_str = str(call_args[0][0])
            sql_strings.append(sql_str)
    return sql_strings


def _assert_no_pragma(sql_strings: list[str], context: str):
    """Assert that no SQL string contains a PRAGMA statement."""
    for sql in sql_strings:
        assert "PRAGMA" not in sql.upper(), (
            f"Found PRAGMA in SQL executed against Postgres engine "
            f"({context}): {sql}"
        )


def _assert_no_sqlite_trigger_syntax(sql_strings: list[str], context: str):
    """Assert no SQL uses SQLite-specific trigger syntax (SELECT RAISE(ABORT, ...))."""
    for sql in sql_strings:
        assert "RAISE(ABORT" not in sql.upper(), (
            f"Found SQLite-specific RAISE(ABORT, ...) trigger syntax "
            f"in SQL executed against Postgres engine ({context}): {sql}"
        )


def _assert_no_autoincrement(sql_strings: list[str], context: str):
    """Assert no SQL uses AUTOINCREMENT (SQLite-only keyword)."""
    for sql in sql_strings:
        assert "AUTOINCREMENT" not in sql.upper(), (
            f"Found AUTOINCREMENT keyword in SQL executed against Postgres engine "
            f"({context}): {sql}"
        )


def _assert_no_strftime(sql_strings: list[str], context: str):
    """Assert no SQL uses strftime() (SQLite-only function)."""
    for sql in sql_strings:
        assert "STRFTIME" not in sql.upper(), (
            f"Found strftime() function in SQL executed against Postgres engine "
            f"({context}): {sql}"
        )


# ---------------------------------------------------------------------------
# Strategies for table existence scenarios
# ---------------------------------------------------------------------------

# Whether inspector reports various tables as existing (affects migration paths)
table_exists_strategy = st.booleans()

# Column presence scenarios for migration checks
column_present_strategy = st.booleans()


# ---------------------------------------------------------------------------
# Property 3: SQLite-Only DDL Guarded by Dialect Check
#
# When engine dialect is "postgresql", no PRAGMA, AUTOINCREMENT, strftime(),
# or RAISE(ABORT, ...) syntax is emitted by any schema init function.
#
# **Validates: Requirements 3.4**
# ---------------------------------------------------------------------------


class TestProperty3SqliteOnlyDdlGuardedByDialectCheck:
    """
    Property 3: SQLite-Only DDL Guarded by Dialect Check.

    Verifies that running check_schema(), init_provenance_schema(),
    init_alert_dispatch_schema(), and init_replay_db() against a Postgres
    engine executes no PRAGMA statements or SQLite-specific DDL.

    **Validates: Requirements 3.4**
    """

    @given(
        has_trades_table=table_exists_strategy,
        has_trade_events_table=table_exists_strategy,
        has_cases_table=table_exists_strategy,
        thesis_col_exists=column_present_strategy,
        dedupe_key_exists=column_present_strategy,
    )
    @settings(max_examples=200, deadline=None)
    def test_check_schema_no_pragma_on_postgres(
        self,
        has_trades_table: bool,
        has_trade_events_table: bool,
        has_cases_table: bool,
        thesis_col_exists: bool,
        dedupe_key_exists: bool,
    ):
        """check_schema() emits no PRAGMA or SQLite DDL against a Postgres engine.

        Varies table existence and column presence to exercise different
        migration code paths.
        """
        engine, conn = make_mock_postgres_engine()

        # Build inspector mock
        mock_inspector = MagicMock()

        def has_table(name):
            if name == "trades":
                return has_trades_table
            if name == "trade_events":
                return has_trade_events_table
            if name == "cases":
                return has_cases_table
            # Default: tables from other schema init modules exist
            return True

        mock_inspector.has_table.side_effect = has_table

        def get_columns(table_name):
            """Return mock columns depending on scenario."""
            base_cols = [{"name": "id"}, {"name": "created_at"}]
            if table_name == "trades":
                cols = list(base_cols)
                if thesis_col_exists:
                    cols.extend([
                        {"name": "thesis"}, {"name": "setup_type"},
                        {"name": "invalidators"}, {"name": "stop_role"},
                        {"name": "stop_updated_by"}, {"name": "stop_updated_at"},
                    ])
                return cols
            if table_name == "trade_events":
                cols = list(base_cols)
                if dedupe_key_exists:
                    cols.append({"name": "dedupe_key"})
                return cols
            if table_name == "cases":
                return [{"name": "id"}, {"name": "exit_category"}]
            # For other tables, return generic columns
            return base_cols

        mock_inspector.get_columns.side_effect = get_columns

        # sa_inspect is imported locally inside check_schema as:
        #   from sqlalchemy import inspect as sa_inspect
        # init_replay_db is also imported locally from db.replay_schema.
        # We patch sqlalchemy.inspect and the sub-init functions at their source.
        # Also patch backfill_stop_roles which uses ORM sessions.
        with patch("sqlalchemy.inspect", return_value=mock_inspector), \
             patch("orchestrator.init_provenance_schema"), \
             patch("db.replay_schema.init_replay_db"), \
             patch("orchestrator.backfill_stop_roles"):
            from orchestrator import check_schema
            check_schema(engine)

        sql_strings = _extract_sql_strings(conn)
        _assert_no_pragma(sql_strings, "check_schema")
        _assert_no_sqlite_trigger_syntax(sql_strings, "check_schema")
        _assert_no_autoincrement(sql_strings, "check_schema")

    @given(data=st.data())
    @settings(max_examples=200, deadline=None)
    def test_init_provenance_schema_no_pragma_on_postgres(self, data):
        """init_provenance_schema() emits no PRAGMA or SQLite DDL on Postgres."""
        engine, conn = make_mock_postgres_engine()

        from db.provenance_schema import init_provenance_schema
        init_provenance_schema(engine)

        sql_strings = _extract_sql_strings(conn)
        _assert_no_pragma(sql_strings, "init_provenance_schema")
        _assert_no_sqlite_trigger_syntax(sql_strings, "init_provenance_schema")
        _assert_no_autoincrement(sql_strings, "init_provenance_schema")
        _assert_no_strftime(sql_strings, "init_provenance_schema")

    @given(
        alert_intents_exists=table_exists_strategy,
        occurrence_col_exists=column_present_strategy,
        dispatch_log_exists=table_exists_strategy,
    )
    @settings(max_examples=200, deadline=None)
    def test_init_alert_dispatch_schema_no_pragma_on_postgres(
        self,
        alert_intents_exists: bool,
        occurrence_col_exists: bool,
        dispatch_log_exists: bool,
    ):
        """init_alert_dispatch_schema() emits no PRAGMA or SQLite DDL on Postgres.

        Varies table/column existence to exercise migration paths.
        """
        engine, conn = make_mock_postgres_engine()

        # Mock inspector for column migration checks
        mock_inspector = MagicMock()

        def get_columns(table_name):
            if table_name == "alert_intents":
                cols = [{"name": "id"}, {"name": "alert_intent_id"}]
                if occurrence_col_exists:
                    cols.append({"name": "occurrence_count_at_deferral"})
                return cols
            if table_name == "alert_dispatch_log":
                return [
                    {"name": "id"}, {"name": "alert_intent_id"},
                    {"name": "dedupe_key"}, {"name": "configured_mode"},
                    {"name": "freshness_age_seconds"}, {"name": "first_seen_age_seconds"},
                    {"name": "dispatch_batch_symbols"}, {"name": "trigger_price"},
                    {"name": "occurrence_count"},
                ]
            return [{"name": "id"}]

        mock_inspector.get_columns.side_effect = get_columns

        with patch("utils.alert_dispatch_schema.sa_inspect", return_value=mock_inspector):
            from utils.alert_dispatch_schema import init_alert_dispatch_schema
            init_alert_dispatch_schema(engine)

        sql_strings = _extract_sql_strings(conn)
        _assert_no_pragma(sql_strings, "init_alert_dispatch_schema")
        _assert_no_sqlite_trigger_syntax(sql_strings, "init_alert_dispatch_schema")
        _assert_no_autoincrement(sql_strings, "init_alert_dispatch_schema")
        _assert_no_strftime(sql_strings, "init_alert_dispatch_schema")

    @given(
        has_blocked_table=table_exists_strategy,
        has_funnel_table=table_exists_strategy,
        lineage_col_exists=column_present_strategy,
    )
    @settings(max_examples=200, deadline=None)
    def test_init_replay_db_no_pragma_on_postgres(
        self,
        has_blocked_table: bool,
        has_funnel_table: bool,
        lineage_col_exists: bool,
    ):
        """init_replay_db() emits no PRAGMA or SQLite DDL on Postgres.

        Varies table/column existence to exercise lineage migration paths.
        """
        engine, conn = make_mock_postgres_engine()

        # Mock inspector for _migrate_lineage_columns
        mock_inspector = MagicMock()

        def has_table(name):
            if name == "blocked_trade_candidates":
                return has_blocked_table
            if name == "funnel_candidates":
                return has_funnel_table
            # Other tables in lineage migration list
            return True

        mock_inspector.has_table.side_effect = has_table

        def get_columns(table_name):
            cols = [{"name": "id"}, {"name": "created_at"}]
            if lineage_col_exists:
                cols.append({"name": "candidate_lineage_id"})
            return cols

        mock_inspector.get_columns.side_effect = get_columns

        with patch("db.replay_schema.sa_inspect", return_value=mock_inspector):
            from db.replay_schema import init_replay_db
            init_replay_db(engine)

        sql_strings = _extract_sql_strings(conn)
        _assert_no_pragma(sql_strings, "init_replay_db")
        _assert_no_sqlite_trigger_syntax(sql_strings, "init_replay_db")
        _assert_no_autoincrement(sql_strings, "init_replay_db")
        _assert_no_strftime(sql_strings, "init_replay_db")

    @given(data=st.data())
    @settings(max_examples=200, deadline=None)
    def test_no_event_listener_registered_on_postgres(self, data):
        """No SQLAlchemy 'connect' event listener is registered on Postgres engines.

        This verifies that PRAGMA-setting event listeners (WAL, busy_timeout,
        foreign_keys) are never attached when the dialect is postgresql.
        """
        engine, conn = make_mock_postgres_engine()

        # Track event.listens_for calls
        with patch("db.provenance_schema.event") as mock_event_prov, \
             patch("db.replay_schema.event") as mock_event_replay:

            from db.provenance_schema import init_provenance_schema
            init_provenance_schema(engine)

            # Verify no listens_for was called (PRAGMA listeners)
            assert not mock_event_prov.listens_for.called, (
                "init_provenance_schema() registered an event listener on Postgres engine"
            )

        with patch("utils.alert_dispatch_schema.event") as mock_event_alert, \
             patch("utils.alert_dispatch_schema.sa_inspect") as mock_inspect:
            mock_inspect.return_value.get_columns.return_value = [
                {"name": "id"}, {"name": "occurrence_count_at_deferral"},
                {"name": "dedupe_key"}, {"name": "configured_mode"},
                {"name": "freshness_age_seconds"}, {"name": "first_seen_age_seconds"},
                {"name": "dispatch_batch_symbols"}, {"name": "trigger_price"},
                {"name": "occurrence_count"},
            ]

            from utils.alert_dispatch_schema import init_alert_dispatch_schema
            init_alert_dispatch_schema(engine)

            assert not mock_event_alert.listens_for.called, (
                "init_alert_dispatch_schema() registered an event listener on Postgres engine"
            )
