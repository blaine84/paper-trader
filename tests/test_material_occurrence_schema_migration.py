"""Tests for material_occurrence_count schema migration — Task 2.3.

Verifies:
- Fresh DB gets material_occurrence_count on alert_intents with DEFAULT 1
- Fresh DB gets material_occurrence_count on alert_dispatch_log
- Re-running migration on DB that already has column is idempotent
- Backfill sets material_occurrence_count = occurrence_count_at_deferral for rows where snapshot > 1
- Backfill leaves material_occurrence_count = 1 for rows where snapshot is NULL or 1

Requirements: 0.1, 5.3, 6.6
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect as sa_inspect, text

from utils.alert_dispatch_schema import init_alert_dispatch_schema


@pytest.fixture
def engine():
    """In-memory SQLite engine with alert dispatch schema initialized."""
    eng = create_engine("sqlite://", echo=False)
    init_alert_dispatch_schema(eng)
    return eng


class TestMaterialOccurrenceAlertIntents:
    """material_occurrence_count column on alert_intents table."""

    def test_fresh_db_has_material_occurrence_count_column(self, engine):
        """Fresh DB has material_occurrence_count on alert_intents."""
        inspector = sa_inspect(engine)
        columns = {col["name"] for col in inspector.get_columns("alert_intents")}
        assert "material_occurrence_count" in columns

    def test_material_occurrence_count_default_is_1(self, engine):
        """material_occurrence_count defaults to 1 on new inserts."""
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO alert_intents (
                    alert_intent_id, symbol, alert_type, trigger_price,
                    dedupe_key, first_seen_at, last_seen_at,
                    expiration_at, occurrence_count
                ) VALUES (
                    'test-uuid-001', 'NVDA', 'entry_alert', '145.50',
                    'NVDA:entry_alert:abc123', '2025-01-15T10:30:00Z',
                    '2025-01-15T10:30:00Z', '2025-01-15T16:00:00Z', 1
                )
            """))

        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT material_occurrence_count FROM alert_intents WHERE alert_intent_id = 'test-uuid-001'"
            )).fetchone()
        assert row[0] == 1


class TestMaterialOccurrenceDispatchLog:
    """material_occurrence_count column on alert_dispatch_log table."""

    def test_fresh_db_has_material_occurrence_count_column(self, engine):
        """Fresh DB has material_occurrence_count on alert_dispatch_log."""
        inspector = sa_inspect(engine)
        columns = {col["name"] for col in inspector.get_columns("alert_dispatch_log")}
        assert "material_occurrence_count" in columns

    def test_material_occurrence_count_is_nullable(self, engine):
        """material_occurrence_count on alert_dispatch_log accepts NULL (existing rows unaffected)."""
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO alert_dispatch_log (
                    alert_intent_id, symbol, alert_type, urgency,
                    dispatch_status, dispatched_at
                ) VALUES (
                    'intent-log-001', 'NVDA', 'entry_alert', 'medium',
                    'would_dispatch', '2025-01-15T10:30:00Z'
                )
            """))

        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT material_occurrence_count FROM alert_dispatch_log WHERE alert_intent_id = 'intent-log-001'"
            )).fetchone()
        assert row[0] is None


class TestMaterialOccurrenceMigrationIdempotency:
    """Re-running migration on DB that already has column is idempotent."""

    def test_rerun_migration_no_error(self, engine):
        """Calling init_alert_dispatch_schema twice does not raise."""
        # First call already happened in the fixture. Call again:
        init_alert_dispatch_schema(engine)

        # Verify column still present and intact
        inspector = sa_inspect(engine)
        columns = {col["name"] for col in inspector.get_columns("alert_intents")}
        assert "material_occurrence_count" in columns

    def test_rerun_migration_preserves_data(self, engine):
        """Re-running migration doesn't alter existing data."""
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO alert_intents (
                    alert_intent_id, symbol, alert_type, trigger_price,
                    dedupe_key, first_seen_at, last_seen_at,
                    expiration_at, occurrence_count, material_occurrence_count
                ) VALUES (
                    'test-uuid-idem', 'AMD', 'entry_alert', '155.00',
                    'AMD:entry_alert:xyz789', '2025-01-15T10:00:00Z',
                    '2025-01-15T10:00:00Z', '2025-01-15T16:00:00Z', 5, 3
                )
            """))

        # Re-run migration
        init_alert_dispatch_schema(engine)

        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT occurrence_count, material_occurrence_count FROM alert_intents "
                "WHERE alert_intent_id = 'test-uuid-idem'"
            )).fetchone()
        assert row[0] == 5
        assert row[1] == 3


class TestMaterialOccurrenceBackfill:
    """Backfill logic for material_occurrence_count from occurrence_count_at_deferral."""

    def test_backfill_sets_material_count_from_snapshot_greater_than_1(self):
        """Backfill sets material_occurrence_count = occurrence_count_at_deferral
        for rows where occurrence_count_at_deferral > 1."""
        # Create engine WITHOUT calling init_alert_dispatch_schema so we can
        # simulate a pre-migration state and then run the migration.
        eng = create_engine("sqlite://", echo=False)

        # Create table WITHOUT material_occurrence_count
        with eng.begin() as conn:
            conn.execute(text("""
                CREATE TABLE alert_intents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_intent_id TEXT NOT NULL UNIQUE,
                    symbol TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    direction TEXT,
                    trigger_price TEXT NOT NULL,
                    source_level TEXT,
                    urgency TEXT NOT NULL DEFAULT 'medium',
                    reason TEXT,
                    dedupe_key TEXT NOT NULL,
                    filter_status TEXT NOT NULL DEFAULT 'unclassified',
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    occurrence_count INTEGER NOT NULL DEFAULT 1,
                    expiration_at TEXT NOT NULL,
                    dispatch_status TEXT NOT NULL DEFAULT 'pending',
                    dispatch_reason TEXT,
                    dispatched_at TEXT,
                    deferred_until TEXT,
                    dispatch_attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_dispatch_error TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    occurrence_count_at_deferral INTEGER DEFAULT 0
                )
            """))

            # Insert rows with varying occurrence_count_at_deferral
            conn.execute(text("""
                INSERT INTO alert_intents (
                    alert_intent_id, symbol, alert_type, trigger_price,
                    dedupe_key, first_seen_at, last_seen_at,
                    expiration_at, occurrence_count, occurrence_count_at_deferral
                ) VALUES
                ('uuid-snap-3', 'NVDA', 'entry_alert', '145.00',
                 'NVDA:entry:1', '2025-01-15T10:00:00Z', '2025-01-15T10:00:00Z',
                 '2025-01-15T16:00:00Z', 5, 3),
                ('uuid-snap-7', 'AMD', 'entry_alert', '155.00',
                 'AMD:entry:1', '2025-01-15T10:00:00Z', '2025-01-15T10:00:00Z',
                 '2025-01-15T16:00:00Z', 10, 7)
            """))

        # Now run full schema init (includes migration + backfill)
        init_alert_dispatch_schema(eng)

        with eng.connect() as conn:
            rows = conn.execute(text(
                "SELECT alert_intent_id, material_occurrence_count "
                "FROM alert_intents ORDER BY alert_intent_id"
            )).fetchall()

        results = {row[0]: row[1] for row in rows}
        assert results["uuid-snap-3"] == 3
        assert results["uuid-snap-7"] == 7

    def test_backfill_leaves_material_count_1_when_snapshot_is_null(self):
        """Backfill leaves material_occurrence_count = 1 for rows where
        occurrence_count_at_deferral is NULL."""
        eng = create_engine("sqlite://", echo=False)

        with eng.begin() as conn:
            conn.execute(text("""
                CREATE TABLE alert_intents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_intent_id TEXT NOT NULL UNIQUE,
                    symbol TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    direction TEXT,
                    trigger_price TEXT NOT NULL,
                    source_level TEXT,
                    urgency TEXT NOT NULL DEFAULT 'medium',
                    reason TEXT,
                    dedupe_key TEXT NOT NULL,
                    filter_status TEXT NOT NULL DEFAULT 'unclassified',
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    occurrence_count INTEGER NOT NULL DEFAULT 1,
                    expiration_at TEXT NOT NULL,
                    dispatch_status TEXT NOT NULL DEFAULT 'pending',
                    dispatch_reason TEXT,
                    dispatched_at TEXT,
                    deferred_until TEXT,
                    dispatch_attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_dispatch_error TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    occurrence_count_at_deferral INTEGER DEFAULT 0
                )
            """))

            conn.execute(text("""
                INSERT INTO alert_intents (
                    alert_intent_id, symbol, alert_type, trigger_price,
                    dedupe_key, first_seen_at, last_seen_at,
                    expiration_at, occurrence_count, occurrence_count_at_deferral
                ) VALUES
                ('uuid-null-snap', 'TSLA', 'entry_alert', '300.00',
                 'TSLA:entry:1', '2025-01-15T10:00:00Z', '2025-01-15T10:00:00Z',
                 '2025-01-15T16:00:00Z', 4, NULL)
            """))

        init_alert_dispatch_schema(eng)

        with eng.connect() as conn:
            row = conn.execute(text(
                "SELECT material_occurrence_count FROM alert_intents "
                "WHERE alert_intent_id = 'uuid-null-snap'"
            )).fetchone()
        assert row[0] == 1

    def test_backfill_leaves_material_count_1_when_snapshot_is_1(self):
        """Backfill leaves material_occurrence_count = 1 for rows where
        occurrence_count_at_deferral = 1 (no material progress beyond initial)."""
        eng = create_engine("sqlite://", echo=False)

        with eng.begin() as conn:
            conn.execute(text("""
                CREATE TABLE alert_intents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_intent_id TEXT NOT NULL UNIQUE,
                    symbol TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    direction TEXT,
                    trigger_price TEXT NOT NULL,
                    source_level TEXT,
                    urgency TEXT NOT NULL DEFAULT 'medium',
                    reason TEXT,
                    dedupe_key TEXT NOT NULL,
                    filter_status TEXT NOT NULL DEFAULT 'unclassified',
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    occurrence_count INTEGER NOT NULL DEFAULT 1,
                    expiration_at TEXT NOT NULL,
                    dispatch_status TEXT NOT NULL DEFAULT 'pending',
                    dispatch_reason TEXT,
                    dispatched_at TEXT,
                    deferred_until TEXT,
                    dispatch_attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_dispatch_error TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    occurrence_count_at_deferral INTEGER DEFAULT 0
                )
            """))

            conn.execute(text("""
                INSERT INTO alert_intents (
                    alert_intent_id, symbol, alert_type, trigger_price,
                    dedupe_key, first_seen_at, last_seen_at,
                    expiration_at, occurrence_count, occurrence_count_at_deferral
                ) VALUES
                ('uuid-snap-1', 'AAPL', 'entry_alert', '190.00',
                 'AAPL:entry:1', '2025-01-15T10:00:00Z', '2025-01-15T10:00:00Z',
                 '2025-01-15T16:00:00Z', 3, 1)
            """))

        init_alert_dispatch_schema(eng)

        with eng.connect() as conn:
            row = conn.execute(text(
                "SELECT material_occurrence_count FROM alert_intents "
                "WHERE alert_intent_id = 'uuid-snap-1'"
            )).fetchone()
        assert row[0] == 1
