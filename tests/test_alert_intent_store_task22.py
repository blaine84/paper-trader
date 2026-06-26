"""Tests for AlertIntentStore.record_or_update_intent() — Task 2.2 validation.

Verifies the UPSERT pattern, UUID4 generation, defaults, _row_to_intent helper,
and behavior when existing intents are in terminal states.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, text

from utils.alert_dispatch_schema import init_alert_dispatch_schema
from utils.alert_intent_store import AlertIntent, AlertIntentStore


@pytest.fixture
def store():
    """Create an in-memory SQLite engine with schema and return a store."""
    engine = create_engine("sqlite://", echo=False)
    init_alert_dispatch_schema(engine)
    return AlertIntentStore(engine)


def _sample_intent_data(**overrides) -> dict:
    """Factory for intent data dicts."""
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
    data.update(overrides)
    return data


class TestRecordOrUpdateIntent:
    """Tests for record_or_update_intent UPSERT method."""

    def test_insert_new_intent_returns_alert_intent(self, store):
        """New insert returns an AlertIntent dataclass with correct fields."""
        result = store.record_or_update_intent(_sample_intent_data())

        assert isinstance(result, AlertIntent)
        assert result.symbol == "NVDA"
        assert result.alert_type == "entry_alert"
        assert result.direction == "long"
        assert result.trigger_price == Decimal("145.50")
        assert result.source_level == "breakout above 145"
        assert result.urgency == "medium"
        assert result.reason == "Price crossed key resistance"
        assert result.dedupe_key == "NVDA:entry_alert:abc123def456"

    def test_insert_sets_correct_defaults(self, store):
        """New insert sets filter_status=unclassified, occurrence_count=1,
        dispatch_attempt_count=0, dispatch_status=pending."""
        result = store.record_or_update_intent(_sample_intent_data())

        assert result.filter_status == "unclassified"
        assert result.occurrence_count == 1
        assert result.dispatch_attempt_count == 0
        assert result.dispatch_status == "pending"

    def test_insert_generates_uuid4(self, store):
        """New insert generates a valid UUID4 string."""
        result = store.record_or_update_intent(_sample_intent_data())

        import uuid
        parsed = uuid.UUID(result.alert_intent_id, version=4)
        assert str(parsed) == result.alert_intent_id

    def test_upsert_increments_occurrence_count(self, store):
        """Second insert with same dedupe_key increments occurrence_count."""
        data = _sample_intent_data()
        store.record_or_update_intent(data)

        data["last_seen_at"] = "2025-01-15T10:31:00.000Z"
        result = store.record_or_update_intent(data)

        assert result.occurrence_count == 2

    def test_upsert_updates_last_seen_at(self, store):
        """Upsert updates last_seen_at to the new value."""
        data = _sample_intent_data()
        store.record_or_update_intent(data)

        data["last_seen_at"] = "2025-01-15T10:31:00.000Z"
        result = store.record_or_update_intent(data)

        assert result.last_seen_at == datetime(2025, 1, 15, 10, 31, 0)

    def test_upsert_extends_expiration_via_max(self, store):
        """Upsert uses MAX to extend expiration_at, never shrinks it."""
        data = _sample_intent_data(expiration_at="2025-01-15T16:00:00.000Z")
        store.record_or_update_intent(data)

        # Later expiration — should extend
        data["last_seen_at"] = "2025-01-15T10:31:00.000Z"
        data["expiration_at"] = "2025-01-15T17:00:00.000Z"
        result = store.record_or_update_intent(data)
        assert result.expiration_at == datetime(2025, 1, 15, 17, 0, 0)

        # Earlier expiration — should NOT shrink
        data["last_seen_at"] = "2025-01-15T10:32:00.000Z"
        data["expiration_at"] = "2025-01-15T15:00:00.000Z"
        result = store.record_or_update_intent(data)
        assert result.expiration_at == datetime(2025, 1, 15, 17, 0, 0)

    def test_upsert_preserves_original_uuid(self, store):
        """Upsert preserves the original alert_intent_id (UUID)."""
        data = _sample_intent_data()
        first = store.record_or_update_intent(data)

        data["last_seen_at"] = "2025-01-15T10:31:00.000Z"
        second = store.record_or_update_intent(data)

        assert second.alert_intent_id == first.alert_intent_id

    def test_new_insert_after_terminal_state(self, store):
        """When existing intent is consumed (terminal), a new insert is created."""
        data = _sample_intent_data()
        first = store.record_or_update_intent(data)

        # Manually mark as consumed (terminal)
        with store._engine.begin() as conn:
            conn.execute(
                text("UPDATE alert_intents SET dispatch_status='consumed' WHERE id=:id"),
                {"id": first.id},
            )

        # Same dedupe_key — should create new row
        data["last_seen_at"] = "2025-01-15T11:00:00.000Z"
        second = store.record_or_update_intent(data)

        assert second.id != first.id
        assert second.occurrence_count == 1
        assert second.dispatch_status == "pending"

    def test_defaults_applied_when_missing_from_data(self, store):
        """filter_status and urgency default when not in intent_data."""
        data = _sample_intent_data()
        del data["filter_status"]
        del data["urgency"]

        result = store.record_or_update_intent(data)
        assert result.filter_status == "unclassified"
        assert result.urgency == "medium"

    def test_nullable_fields_handled(self, store):
        """Direction, source_level, reason can be None."""
        data = _sample_intent_data(direction=None, source_level=None, reason=None)
        result = store.record_or_update_intent(data)

        assert result.direction is None
        assert result.source_level is None
        assert result.reason is None

    def test_row_to_intent_parses_decimal(self, store):
        """trigger_price is parsed as Decimal."""
        result = store.record_or_update_intent(
            _sample_intent_data(trigger_price="99.123456")
        )
        assert result.trigger_price == Decimal("99.123456")

    def test_multiple_dedupe_keys_independent(self, store):
        """Different dedupe_keys produce independent intents."""
        r1 = store.record_or_update_intent(
            _sample_intent_data(dedupe_key="KEY_A")
        )
        r2 = store.record_or_update_intent(
            _sample_intent_data(dedupe_key="KEY_B")
        )

        assert r1.id != r2.id
        assert r1.occurrence_count == 1
        assert r2.occurrence_count == 1
