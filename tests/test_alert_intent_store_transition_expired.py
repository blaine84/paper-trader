"""Tests for AlertIntentStore.transition_to_expired() — Task 4.1 validation.

Verifies:
- Active intents (pending, dispatched, claimed_by_scheduled) transition to expired
- Terminal-state intents (consumed, expired, suppressed, dispatch_failed) are guarded
- Returns True/False correctly
- dispatch_reason is set on successful transition

Requirements: 3.3, 4.1, 9.6
"""

from __future__ import annotations

import logging
from datetime import datetime

import pytest
from sqlalchemy import create_engine, text

from utils.alert_dispatch_schema import init_alert_dispatch_schema
from utils.alert_intent_store import AlertIntentStore


@pytest.fixture
def store():
    """Create an in-memory SQLite engine with schema and return a store."""
    engine = create_engine("sqlite://", echo=False)
    init_alert_dispatch_schema(engine)
    return AlertIntentStore(engine)


def _insert_intent(store, dispatch_status: str = "pending", *, intent_id: int = None) -> int:
    """Insert an intent row directly and return its id."""
    data = {
        "symbol": "NVDA",
        "alert_type": "entry_alert",
        "direction": "long",
        "trigger_price": "145.50",
        "source_level": "breakout above 145",
        "urgency": "medium",
        "reason": "Price crossed key resistance",
        "dedupe_key": "NVDA:entry_alert:abc123def456",
        "filter_status": "unclassified",
        "first_seen_at": "2025-01-15T10:30:00.000Z",
        "last_seen_at": "2025-01-15T10:30:00.000Z",
        "expiration_at": "2025-01-15T16:00:00.000Z",
    }
    intent = store.record_or_update_intent(data)
    row_id = intent.id

    # Override dispatch_status if needed
    if dispatch_status != "pending":
        with store._engine.begin() as conn:
            conn.execute(
                text("UPDATE alert_intents SET dispatch_status = :status WHERE id = :id"),
                {"status": dispatch_status, "id": row_id},
            )
    return row_id


class TestTransitionToExpired:
    """Tests for transition_to_expired method."""

    def test_pending_transitions_to_expired(self, store):
        """A pending intent transitions to expired successfully."""
        row_id = _insert_intent(store, "pending")

        result = store.transition_to_expired(row_id, "freshness_expired")

        assert result is True
        with store._engine.begin() as conn:
            row = conn.execute(
                text("SELECT dispatch_status, dispatch_reason FROM alert_intents WHERE id = :id"),
                {"id": row_id},
            ).fetchone()
        assert row[0] == "expired"
        assert row[1] == "freshness_expired"

    def test_dispatched_transitions_to_expired(self, store):
        """A dispatched intent transitions to expired successfully."""
        row_id = _insert_intent(store, "dispatched")

        result = store.transition_to_expired(row_id, "age_limit_reached")

        assert result is True
        with store._engine.begin() as conn:
            row = conn.execute(
                text("SELECT dispatch_status, dispatch_reason FROM alert_intents WHERE id = :id"),
                {"id": row_id},
            ).fetchone()
        assert row[0] == "expired"
        assert row[1] == "age_limit_reached"

    def test_claimed_by_scheduled_transitions_to_expired(self, store):
        """A claimed_by_scheduled intent transitions to expired successfully."""
        row_id = _insert_intent(store, "claimed_by_scheduled")

        result = store.transition_to_expired(row_id, "market_session_ended")

        assert result is True
        with store._engine.begin() as conn:
            row = conn.execute(
                text("SELECT dispatch_status, dispatch_reason FROM alert_intents WHERE id = :id"),
                {"id": row_id},
            ).fetchone()
        assert row[0] == "expired"
        assert row[1] == "market_session_ended"

    @pytest.mark.parametrize("terminal_status", [
        "consumed",
        "expired",
        "suppressed",
        "dispatch_failed",
    ])
    def test_terminal_state_returns_false(self, store, terminal_status):
        """Terminal-state intents cannot be transitioned — returns False."""
        row_id = _insert_intent(store, terminal_status)

        result = store.transition_to_expired(row_id, "should_not_apply")

        assert result is False
        # Status unchanged
        with store._engine.begin() as conn:
            row = conn.execute(
                text("SELECT dispatch_status FROM alert_intents WHERE id = :id"),
                {"id": row_id},
            ).fetchone()
        assert row[0] == terminal_status

    def test_terminal_state_logs_warning(self, store, caplog):
        """Terminal-state transition attempt logs a WARNING."""
        row_id = _insert_intent(store, "consumed")

        with caplog.at_level(logging.WARNING, logger="utils.alert_intent_store"):
            store.transition_to_expired(row_id, "test_reason")

        assert any("terminal state" in rec.message for rec in caplog.records)

    def test_nonexistent_intent_returns_false(self, store):
        """Non-existent intent_id returns False (rowcount 0)."""
        result = store.transition_to_expired(99999, "no_such_intent")
        assert result is False

    def test_reason_is_stored(self, store):
        """The reason string is persisted as dispatch_reason."""
        row_id = _insert_intent(store, "pending")
        store.transition_to_expired(row_id, "undetermined_freshness")

        with store._engine.begin() as conn:
            row = conn.execute(
                text("SELECT dispatch_reason FROM alert_intents WHERE id = :id"),
                {"id": row_id},
            ).fetchone()
        assert row[0] == "undetermined_freshness"
