"""Tests for pm_alert_events immutable audit table — Task 3.4.

Verifies:
- Table creation with correct columns
- Indexes on (alert_intent_id, event_type) and event_at
- Immutability triggers blocking UPDATE and DELETE
- INSERT works correctly for append-only usage

Requirements: 5.3, 7.4
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from utils.alert_dispatch_schema import init_alert_dispatch_schema


@pytest.fixture
def engine():
    """In-memory SQLite engine with alert dispatch schema initialized."""
    eng = create_engine("sqlite://", echo=False)
    init_alert_dispatch_schema(eng)
    return eng


class TestPmAlertEventsTable:
    """pm_alert_events table creation and column verification."""

    def test_table_exists(self, engine):
        """pm_alert_events table is created by init_alert_dispatch_schema."""
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='pm_alert_events'"
            )).fetchone()
        assert result is not None
        assert result[0] == "pm_alert_events"

    def test_columns_match_spec(self, engine):
        """pm_alert_events has all required columns from the design spec."""
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(pm_alert_events)"))
            columns = {row[1]: row[2] for row in result.fetchall()}

        expected_columns = {
            "id", "alert_intent_id", "event_type", "symbol",
            "alert_type", "profile_id", "event_at", "event_data",
        }
        assert expected_columns.issubset(set(columns.keys()))

    def test_insert_succeeds(self, engine):
        """INSERT into pm_alert_events works (append-only)."""
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO pm_alert_events
                    (alert_intent_id, event_type, symbol, alert_type, profile_id, event_at, event_data)
                VALUES ('intent-001', 'claimed', 'NVDA', 'entry_alert', 'aggressive', '2025-01-15T10:30:00Z', '{"detail": "test"}')
            """))

        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT alert_intent_id, event_type, symbol FROM pm_alert_events WHERE alert_intent_id = 'intent-001'"
            )).fetchone()
        assert row[0] == "intent-001"
        assert row[1] == "claimed"
        assert row[2] == "NVDA"


class TestPmAlertEventsIndexes:
    """Verify required indexes exist on pm_alert_events."""

    def test_intent_type_index_exists(self, engine):
        """Index on (alert_intent_id, event_type) exists."""
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_pm_alert_events_intent_type'"
            )).fetchone()
        assert result is not None

    def test_event_at_index_exists(self, engine):
        """Index on event_at exists."""
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_pm_alert_events_event_at'"
            )).fetchone()
        assert result is not None


class TestPmAlertEventsImmutability:
    """UPDATE and DELETE on pm_alert_events should be blocked by triggers."""

    def test_update_raises(self, engine):
        """UPDATE on pm_alert_events triggers ABORT."""
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO pm_alert_events
                    (alert_intent_id, event_type, symbol, alert_type, profile_id, event_at)
                VALUES ('intent-upd', 'claimed', 'AAPL', 'breakout', 'moderate', '2025-01-15T11:00:00Z')
            """))

        with pytest.raises(Exception, match="pm_alert_events is immutable.*UPDATE blocked"):
            with engine.begin() as conn:
                conn.execute(text("""
                    UPDATE pm_alert_events SET event_type = 'completed' WHERE alert_intent_id = 'intent-upd'
                """))

    def test_delete_raises(self, engine):
        """DELETE on pm_alert_events triggers ABORT."""
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO pm_alert_events
                    (alert_intent_id, event_type, symbol, alert_type, profile_id, event_at)
                VALUES ('intent-del', 'duplicate_noop', 'TSLA', 'rapid_move', 'aggressive', '2025-01-15T12:00:00Z')
            """))

        with pytest.raises(Exception, match="pm_alert_events is immutable.*DELETE blocked"):
            with engine.begin() as conn:
                conn.execute(text("""
                    DELETE FROM pm_alert_events WHERE alert_intent_id = 'intent-del'
                """))

    def test_multiple_inserts_allowed(self, engine):
        """Multiple INSERTs are allowed (append-only semantics)."""
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO pm_alert_events
                    (alert_intent_id, event_type, symbol, alert_type, profile_id, event_at)
                VALUES ('intent-multi', 'claimed', 'MSFT', 'entry_alert', 'moderate', '2025-01-15T09:00:00Z')
            """))
            conn.execute(text("""
                INSERT INTO pm_alert_events
                    (alert_intent_id, event_type, symbol, alert_type, profile_id, event_at)
                VALUES ('intent-multi', 'completed', 'MSFT', 'entry_alert', 'moderate', '2025-01-15T09:01:00Z')
            """))

        with engine.connect() as conn:
            count = conn.execute(text(
                "SELECT COUNT(*) FROM pm_alert_events WHERE alert_intent_id = 'intent-multi'"
            )).scalar()
        assert count == 2
