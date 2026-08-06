"""Tests for record_or_update_intent — material occurrence tracking (Task 4.4).

Validates that occurrence_count and material_occurrence_count update correctly
under various upsert scenarios: new inserts, repeated stable observations,
source_level changes, direction changes, and price moves above/below threshold.

Requirements: 0.1–0.5, 1.1–1.5
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest
from sqlalchemy import create_engine

from utils.alert_dispatch_schema import init_alert_dispatch_schema
from utils.alert_intent_store import AlertIntentStore


@pytest.fixture
def engine():
    """In-memory SQLite engine with alert dispatch schema."""
    eng = create_engine("sqlite://", echo=False)
    init_alert_dispatch_schema(eng)
    return eng


@pytest.fixture
def store(engine):
    """AlertIntentStore backed by the in-memory engine."""
    return AlertIntentStore(engine)


def _intent_data(**overrides) -> dict:
    """Factory for intent data dicts with sensible defaults."""
    data = {
        "symbol": "AMD",
        "alert_type": "entry_alert",
        "direction": "long",
        "trigger_price": "155.20",
        "source_level": "resistance_155",
        "urgency": "medium",
        "reason": "Price near resistance",
        "dedupe_key": "AMD:entry_alert:resistance_155",
        "filter_status": "unclassified",
        "first_seen_at": "2025-01-20T10:00:00.000000Z",
        "last_seen_at": "2025-01-20T10:00:00.000000Z",
        "expiration_at": "2025-01-20T16:00:00.000000Z",
    }
    data.update(overrides)
    return data


class TestNewInsert:
    """New INSERT sets both counters to 1."""

    def test_new_insert_sets_occurrence_count_1(self, store):
        result = store.record_or_update_intent(_intent_data())
        assert result.occurrence_count == 1

    def test_new_insert_sets_material_occurrence_count_1(self, store):
        result = store.record_or_update_intent(_intent_data())
        assert result.material_occurrence_count == 1


class TestRepeatedUpsertNoMaterialChange:
    """Repeated upsert (no material change) increments occurrence_count only."""

    def test_occurrence_count_increments(self, store):
        store.record_or_update_intent(_intent_data())
        # Second call: same source_level, direction, price within threshold
        result = store.record_or_update_intent(_intent_data(
            last_seen_at="2025-01-20T10:01:00.000000Z",
            trigger_price="155.22",  # 0.013% change, well below 0.5%
        ))
        assert result.occurrence_count == 2

    def test_material_occurrence_count_stays_same(self, store):
        store.record_or_update_intent(_intent_data())
        result = store.record_or_update_intent(_intent_data(
            last_seen_at="2025-01-20T10:01:00.000000Z",
            trigger_price="155.22",  # 0.013% change
        ))
        assert result.material_occurrence_count == 1

    def test_three_upserts_no_material_change(self, store):
        store.record_or_update_intent(_intent_data())
        store.record_or_update_intent(_intent_data(
            last_seen_at="2025-01-20T10:01:00.000000Z",
            trigger_price="155.22",
        ))
        result = store.record_or_update_intent(_intent_data(
            last_seen_at="2025-01-20T10:02:00.000000Z",
            trigger_price="155.18",  # drift within threshold
        ))
        assert result.occurrence_count == 3
        assert result.material_occurrence_count == 1


class TestSourceLevelChange:
    """Upsert with source_level change increments BOTH counters."""

    def test_source_level_change_increments_both(self, store):
        store.record_or_update_intent(_intent_data())
        result = store.record_or_update_intent(_intent_data(
            last_seen_at="2025-01-20T10:05:00.000000Z",
            source_level="resistance_157",
        ))
        assert result.occurrence_count == 2
        assert result.material_occurrence_count == 2


class TestDirectionChange:
    """Upsert with direction change increments BOTH counters."""

    def test_direction_change_increments_both(self, store):
        store.record_or_update_intent(_intent_data())
        result = store.record_or_update_intent(_intent_data(
            last_seen_at="2025-01-20T10:05:00.000000Z",
            direction="short",
        ))
        assert result.occurrence_count == 2
        assert result.material_occurrence_count == 2


class TestPriceAboveThreshold:
    """Upsert with price > 0.5% threshold increments BOTH counters."""

    def test_price_above_threshold_increments_both(self, store):
        # Start at 155.20, move to 156.20 → ~0.64% > 0.5%
        store.record_or_update_intent(_intent_data())
        result = store.record_or_update_intent(_intent_data(
            last_seen_at="2025-01-20T10:05:00.000000Z",
            trigger_price="156.20",
        ))
        assert result.occurrence_count == 2
        assert result.material_occurrence_count == 2


class TestPriceBelowOrEqualThreshold:
    """Upsert with price <= 0.5% threshold increments only occurrence_count."""

    def test_price_at_threshold_increments_only_occurrence(self, store):
        # Start at 155.20, move to 155.976 → exactly 0.5% → NOT material (strictly greater required)
        store.record_or_update_intent(_intent_data())
        result = store.record_or_update_intent(_intent_data(
            last_seen_at="2025-01-20T10:05:00.000000Z",
            trigger_price="155.976",  # 155.20 * 1.005 = 155.976 → exactly 0.5%
        ))
        assert result.occurrence_count == 2
        assert result.material_occurrence_count == 1

    def test_price_below_threshold_increments_only_occurrence(self, store):
        # Start at 155.20, move to 155.50 → ~0.19% < 0.5%
        store.record_or_update_intent(_intent_data())
        result = store.record_or_update_intent(_intent_data(
            last_seen_at="2025-01-20T10:05:00.000000Z",
            trigger_price="155.50",
        ))
        assert result.occurrence_count == 2
        assert result.material_occurrence_count == 1


class TestFreshnessFieldsAlwaysUpdate:
    """last_seen_at, trigger_price, expiration_at always update regardless of materiality."""

    def test_last_seen_at_updates_on_non_material(self, store):
        store.record_or_update_intent(_intent_data())
        result = store.record_or_update_intent(_intent_data(
            last_seen_at="2025-01-20T10:05:00.000000Z",
            trigger_price="155.22",  # non-material
        ))
        assert result.last_seen_at == datetime(2025, 1, 20, 10, 5, 0)

    def test_trigger_price_updates_on_non_material(self, store):
        store.record_or_update_intent(_intent_data())
        result = store.record_or_update_intent(_intent_data(
            last_seen_at="2025-01-20T10:05:00.000000Z",
            trigger_price="155.22",  # non-material price drift
        ))
        assert result.trigger_price == Decimal("155.22")

    def test_expiration_at_extends_on_non_material(self, store):
        store.record_or_update_intent(_intent_data())
        result = store.record_or_update_intent(_intent_data(
            last_seen_at="2025-01-20T10:05:00.000000Z",
            trigger_price="155.22",
            expiration_at="2025-01-20T17:00:00.000000Z",  # later than original 16:00
        ))
        assert result.expiration_at == datetime(2025, 1, 20, 17, 0, 0)

    def test_last_seen_at_updates_on_material(self, store):
        store.record_or_update_intent(_intent_data())
        result = store.record_or_update_intent(_intent_data(
            last_seen_at="2025-01-20T10:10:00.000000Z",
            source_level="resistance_157",  # material change
        ))
        assert result.last_seen_at == datetime(2025, 1, 20, 10, 10, 0)

    def test_trigger_price_updates_on_material(self, store):
        store.record_or_update_intent(_intent_data())
        result = store.record_or_update_intent(_intent_data(
            last_seen_at="2025-01-20T10:10:00.000000Z",
            trigger_price="157.50",  # material price move
        ))
        assert result.trigger_price == Decimal("157.50")


class TestDisabledModePopulatesBothCounters:
    """Both counters populate in disabled mode (flag gates reads, not writes).

    Since writes always happen regardless of ALERT_MATERIAL_OCCURRENCE_MODE,
    we verify both counters behave correctly. The default mode IS disabled,
    so no patching needed — this validates the invariant.
    """

    def test_insert_populates_both_in_disabled_mode(self, store):
        result = store.record_or_update_intent(_intent_data())
        assert result.occurrence_count == 1
        assert result.material_occurrence_count == 1

    def test_material_change_populates_both_in_disabled_mode(self, store):
        store.record_or_update_intent(_intent_data())
        result = store.record_or_update_intent(_intent_data(
            last_seen_at="2025-01-20T10:05:00.000000Z",
            source_level="resistance_157",  # material
        ))
        assert result.occurrence_count == 2
        assert result.material_occurrence_count == 2

    def test_non_material_upsert_still_populates_material_counter_in_disabled_mode(self, store):
        """material_occurrence_count is present and stable even in disabled mode."""
        store.record_or_update_intent(_intent_data())
        result = store.record_or_update_intent(_intent_data(
            last_seen_at="2025-01-20T10:05:00.000000Z",
            trigger_price="155.22",  # non-material
        ))
        assert result.occurrence_count == 2
        assert result.material_occurrence_count == 1
