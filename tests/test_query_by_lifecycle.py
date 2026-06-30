"""Tests for AlertIntentStore.query_by_lifecycle() — Task 4.4 validation.

Verifies filtering by dispatch_status list, alert_type, symbol, time range,
ordering by first_seen_at DESC, LIMIT 1000, and empty result behavior.

Requirements: 9.4, 9.5
"""

from __future__ import annotations

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


def _insert_intent(store, *, symbol="NVDA", alert_type="entry_alert",
                   dispatch_status="pending", first_seen_at="2025-01-15T10:30:00.000Z",
                   dedupe_key=None, **extra):
    """Insert a test intent directly for query tests."""
    import uuid
    if dedupe_key is None:
        dedupe_key = str(uuid.uuid4())
    data = {
        "symbol": symbol,
        "alert_type": alert_type,
        "direction": "long",
        "trigger_price": "100.00",
        "source_level": None,
        "urgency": "medium",
        "reason": None,
        "dedupe_key": dedupe_key,
        "filter_status": "passed",
        "first_seen_at": first_seen_at,
        "last_seen_at": first_seen_at,
        "expiration_at": "2025-01-15T23:00:00.000Z",
    }
    data.update(extra)
    intent = store.record_or_update_intent(data)
    # If we need a non-pending status, update it directly
    if dispatch_status != "pending":
        with store._engine.begin() as conn:
            conn.execute(text(
                "UPDATE alert_intents SET dispatch_status = :status WHERE id = :id"
            ), {"status": dispatch_status, "id": intent.id})
    return intent


class TestQueryByLifecycle:
    """Tests for query_by_lifecycle method."""

    def test_returns_empty_list_when_no_intents(self, store):
        """Empty DB returns empty list, not error."""
        result = store.query_by_lifecycle()
        assert result == []

    def test_returns_empty_list_when_no_matches(self, store):
        """No matching filters returns empty list without raising."""
        _insert_intent(store, symbol="NVDA", dispatch_status="pending")
        result = store.query_by_lifecycle(status_list=["expired"])
        assert result == []

    def test_filters_by_single_status(self, store):
        """Filtering by a single status returns only matching intents."""
        _insert_intent(store, dedupe_key="key1", dispatch_status="pending")
        _insert_intent(store, dedupe_key="key2", dispatch_status="expired")
        _insert_intent(store, dedupe_key="key3", dispatch_status="consumed")

        result = store.query_by_lifecycle(status_list=["pending"])
        assert len(result) == 1
        assert result[0].dispatch_status == "pending"

    def test_filters_by_multiple_statuses(self, store):
        """Filtering by multiple statuses returns all matching."""
        _insert_intent(store, dedupe_key="key1", dispatch_status="pending")
        _insert_intent(store, dedupe_key="key2", dispatch_status="expired")
        _insert_intent(store, dedupe_key="key3", dispatch_status="consumed")

        result = store.query_by_lifecycle(status_list=["pending", "consumed"])
        assert len(result) == 2
        statuses = {r.dispatch_status for r in result}
        assert statuses == {"pending", "consumed"}

    def test_filters_by_alert_type(self, store):
        """Filtering by alert_type returns only matching type."""
        _insert_intent(store, dedupe_key="key1", alert_type="entry_alert")
        _insert_intent(store, dedupe_key="key2", alert_type="rapid_move")

        result = store.query_by_lifecycle(alert_type="entry_alert")
        assert len(result) == 1
        assert result[0].alert_type == "entry_alert"

    def test_filters_by_symbol(self, store):
        """Filtering by symbol returns only matching symbol."""
        _insert_intent(store, dedupe_key="key1", symbol="NVDA")
        _insert_intent(store, dedupe_key="key2", symbol="AAPL")

        result = store.query_by_lifecycle(symbol="AAPL")
        assert len(result) == 1
        assert result[0].symbol == "AAPL"

    def test_filters_by_start_time(self, store):
        """start_time filters intents with first_seen_at >= start_time."""
        _insert_intent(store, dedupe_key="key1", first_seen_at="2025-01-15T09:00:00.000Z")
        _insert_intent(store, dedupe_key="key2", first_seen_at="2025-01-15T11:00:00.000Z")

        result = store.query_by_lifecycle(start_time="2025-01-15T10:00:00.000Z")
        assert len(result) == 1
        assert result[0].first_seen_at == datetime(2025, 1, 15, 11, 0, 0)

    def test_filters_by_end_time(self, store):
        """end_time filters intents with first_seen_at <= end_time."""
        _insert_intent(store, dedupe_key="key1", first_seen_at="2025-01-15T09:00:00.000Z")
        _insert_intent(store, dedupe_key="key2", first_seen_at="2025-01-15T11:00:00.000Z")

        result = store.query_by_lifecycle(end_time="2025-01-15T10:00:00.000Z")
        assert len(result) == 1
        assert result[0].first_seen_at == datetime(2025, 1, 15, 9, 0, 0)

    def test_filters_by_time_range(self, store):
        """start_time and end_time together define a window."""
        _insert_intent(store, dedupe_key="key1", first_seen_at="2025-01-15T08:00:00.000Z")
        _insert_intent(store, dedupe_key="key2", first_seen_at="2025-01-15T10:00:00.000Z")
        _insert_intent(store, dedupe_key="key3", first_seen_at="2025-01-15T12:00:00.000Z")

        result = store.query_by_lifecycle(
            start_time="2025-01-15T09:00:00.000Z",
            end_time="2025-01-15T11:00:00.000Z",
        )
        assert len(result) == 1
        assert result[0].first_seen_at == datetime(2025, 1, 15, 10, 0, 0)

    def test_combined_filters(self, store):
        """Multiple filters combine with AND logic."""
        _insert_intent(store, dedupe_key="key1", symbol="NVDA", alert_type="entry_alert",
                       dispatch_status="pending")
        _insert_intent(store, dedupe_key="key2", symbol="NVDA", alert_type="rapid_move",
                       dispatch_status="pending")
        _insert_intent(store, dedupe_key="key3", symbol="AAPL", alert_type="entry_alert",
                       dispatch_status="expired")

        result = store.query_by_lifecycle(
            status_list=["pending"],
            alert_type="entry_alert",
            symbol="NVDA",
        )
        assert len(result) == 1
        assert result[0].symbol == "NVDA"
        assert result[0].alert_type == "entry_alert"
        assert result[0].dispatch_status == "pending"

    def test_ordered_by_first_seen_at_desc(self, store):
        """Results are ordered by first_seen_at DESC (newest first)."""
        _insert_intent(store, dedupe_key="key1", first_seen_at="2025-01-15T08:00:00.000Z")
        _insert_intent(store, dedupe_key="key2", first_seen_at="2025-01-15T12:00:00.000Z")
        _insert_intent(store, dedupe_key="key3", first_seen_at="2025-01-15T10:00:00.000Z")

        result = store.query_by_lifecycle()
        assert len(result) == 3
        assert result[0].first_seen_at == datetime(2025, 1, 15, 12, 0, 0)
        assert result[1].first_seen_at == datetime(2025, 1, 15, 10, 0, 0)
        assert result[2].first_seen_at == datetime(2025, 1, 15, 8, 0, 0)

    def test_no_filters_returns_all(self, store):
        """Calling with no filters returns all intents (up to 1000)."""
        _insert_intent(store, dedupe_key="key1")
        _insert_intent(store, dedupe_key="key2")
        _insert_intent(store, dedupe_key="key3")

        result = store.query_by_lifecycle()
        assert len(result) == 3

    def test_returns_alert_intent_objects(self, store):
        """Returned items are AlertIntent dataclass instances."""
        from utils.alert_intent_store import AlertIntent
        _insert_intent(store, dedupe_key="key1")

        result = store.query_by_lifecycle()
        assert len(result) == 1
        assert isinstance(result[0], AlertIntent)
