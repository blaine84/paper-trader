"""Unit tests for AlertDispatcher observe logic in material occurrence enabled mode — Task 5.8.

Integration-style tests using a real in-memory SQLite engine with the full alert
dispatch schema. Tests verify:
1. Same alert observed 5 times (material_occ stable at 1) → exactly 1 would_dispatch
   (via the deferral + dedup mechanism working together)
2. has_would_dispatch matches on material_occurrence_count (not occurrence_count)
3. _set_deferred snapshots material_occurrence_count when enabled
4. Secondary _is_material_price_change check bypassed in enabled mode
5. Disabled mode produces existing behavior (occurrence_count-based dedup)

Requirements: 2.1–2.5
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text

from utils.alert_dispatch_schema import init_alert_dispatch_schema
from utils.alert_dispatcher import AlertDispatcher
from utils.alert_intent_store import AlertIntentStore, AlertIntent


# ─── Fixtures ───────────────────────────────────────────────────────────────

_BASE_TIME = datetime(2025, 1, 15, 10, 0, 0)
_ISO_FMT = "%Y-%m-%dT%H:%M:%S.000Z"


@pytest.fixture
def engine():
    """In-memory SQLite engine with full alert dispatch schema."""
    eng = create_engine("sqlite://", echo=False)
    init_alert_dispatch_schema(eng)
    return eng


@pytest.fixture
def store(engine):
    """AlertIntentStore backed by the in-memory engine."""
    return AlertIntentStore(engine)


@pytest.fixture
def dispatcher(engine, store):
    """AlertDispatcher with no-op PM cycle callbacks."""
    return AlertDispatcher(
        engine=engine,
        intent_store=store,
        begin_pm_cycle=lambda s: True,
        end_pm_cycle=lambda s: None,
    )


# ─── Helpers ────────────────────────────────────────────────────────────────


def _insert_intent(store, *, symbol="AMD", alert_type="entry_alert",
                   direction="long", trigger_price="155.20",
                   source_level="resistance_155", dedupe_key=None,
                   filter_status="passed") -> AlertIntent:
    """Insert a classified pending intent via the real store."""
    if dedupe_key is None:
        dedupe_key = f"{symbol}:{alert_type}:{uuid.uuid4().hex[:16]}"
    data = {
        "symbol": symbol,
        "alert_type": alert_type,
        "direction": direction,
        "trigger_price": trigger_price,
        "source_level": source_level,
        "urgency": "medium",
        "reason": "Test alert",
        "dedupe_key": dedupe_key,
        "filter_status": filter_status,
        "first_seen_at": _BASE_TIME.strftime(_ISO_FMT),
        "last_seen_at": _BASE_TIME.strftime(_ISO_FMT),
        "expiration_at": (_BASE_TIME + timedelta(hours=6)).strftime(_ISO_FMT),
    }
    return store.record_or_update_intent(data)


def _upsert_same_intent(store, dedupe_key, *, trigger_price="155.22",
                        source_level="resistance_155", direction="long") -> AlertIntent:
    """Re-observe the same dedupe_key with slightly different price (within 0.5%)."""
    now = _BASE_TIME + timedelta(minutes=2)
    data = {
        "symbol": "AMD",
        "alert_type": "entry_alert",
        "direction": direction,
        "trigger_price": trigger_price,
        "source_level": source_level,
        "urgency": "medium",
        "reason": "Test alert",
        "dedupe_key": dedupe_key,
        "filter_status": "passed",
        "first_seen_at": _BASE_TIME.strftime(_ISO_FMT),
        "last_seen_at": now.strftime(_ISO_FMT),
        "expiration_at": (_BASE_TIME + timedelta(hours=6)).strftime(_ISO_FMT),
    }
    return store.record_or_update_intent(data)


def _count_would_dispatch(engine, alert_intent_id: str) -> int:
    """Count would_dispatch rows in audit log for an intent."""
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT COUNT(*) FROM alert_dispatch_log
            WHERE alert_intent_id = :aid AND dispatch_status = 'would_dispatch'
        """), {"aid": alert_intent_id}).fetchone()
    return row[0]


def _get_deferred_snapshot(engine, intent_id: int) -> int:
    """Read occurrence_count_at_deferral from the DB."""
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT occurrence_count_at_deferral FROM alert_intents WHERE id = :id
        """), {"id": intent_id}).fetchone()
    return row[0]


# ─── Tests: enabled mode ────────────────────────────────────────────────────


class TestObserveEnabledMode:
    """Tests for _handle_observe when ALERT_MATERIAL_OCCURRENCE_MODE == 'enabled'."""

    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "enabled")
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_same_alert_5_times_one_would_dispatch(self, engine, store, dispatcher):
        """Same alert observed 5 times (material_occ stable at 1) → exactly 1 would_dispatch.

        Full integration test: upsert + deferral + observe dedup together.
        After first _handle_observe, deferral is set with snapshot=material_occ=1.
        Subsequent raw ticks keep material_occ=1, so _is_deferred returns True
        and _handle_observe is never called again.

        Requirements: 2.1, 2.2, 2.5
        """
        # Initial insert
        dedupe_key = f"AMD:entry_alert:{uuid.uuid4().hex[:16]}"
        intent = _insert_intent(store, dedupe_key=dedupe_key, trigger_price="155.20")
        alert_intent_id = intent.alert_intent_id

        # Tick 1: first observation → should produce would_dispatch + set deferral
        now = _BASE_TIME + timedelta(minutes=1)
        dispatcher._handle_observe(intent, now)
        assert _count_would_dispatch(engine, alert_intent_id) == 1

        # Ticks 2-5: same alert, slightly drifting price (within 0.5%)
        # After deferral is set, _is_deferred returns True → observe is skipped
        prices = ["155.22", "155.18", "155.25", "155.21"]
        for price in prices:
            # Re-observe via store: occurrence_count bumps, material stays at 1
            intent = _upsert_same_intent(store, dedupe_key, trigger_price=price)
            assert intent.material_occurrence_count == 1

            # Re-read fresh intent
            fresh_intent = store.get_intent_by_id(intent.id)

            # Simulate the dispatcher flow: check deferral first
            is_deferred = dispatcher._is_deferred(fresh_intent, now)
            assert is_deferred is True, (
                f"Expected deferred (material_occ={fresh_intent.material_occurrence_count} == "
                f"snapshot={fresh_intent.occurrence_count_at_deferral})"
            )
            # Since deferred, _handle_observe is NOT called

        # After 5 ticks: exactly 1 would_dispatch
        assert _count_would_dispatch(engine, alert_intent_id) == 1

    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "enabled")
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_has_would_dispatch_matches_on_material_occurrence_count(self, engine, store, dispatcher):
        """Dedup uses material_occurrence_count, not occurrence_count.

        When material_occ is stable at 1 and occurrence_count increases,
        the has_would_dispatch check (keyed on material_occurrence_count as the
        occurrence_count param) still finds the existing row and skips.

        Requirements: 2.3, 2.4
        """
        dedupe_key = f"AMD:entry_alert:{uuid.uuid4().hex[:16]}"
        intent = _insert_intent(store, dedupe_key=dedupe_key, trigger_price="155.20")

        now = _BASE_TIME + timedelta(minutes=1)

        # First observation — writes would_dispatch with trigger_price=155.20
        dispatcher._handle_observe(intent, now)
        assert _count_would_dispatch(engine, intent.alert_intent_id) == 1

        # Bump raw occurrence_count to 5, leave material at 1, keep trigger_price same
        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE alert_intents SET occurrence_count = 5 WHERE id = :id"
            ), {"id": intent.id})

        # Re-read intent (occurrence_count=5, material_occurrence_count=1, same price)
        fresh_intent = store.get_intent_by_id(intent.id)
        assert fresh_intent.occurrence_count == 5
        assert fresh_intent.material_occurrence_count == 1

        # In enabled mode: passes material_occurrence_count=1 as the occurrence_count
        # to has_would_dispatch_for_occurrence. Since the audit row was written with
        # occurrence_count=1 (the material count at time of write), this matches.
        dispatcher._handle_observe(fresh_intent, now)
        assert _count_would_dispatch(engine, intent.alert_intent_id) == 1

    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "enabled")
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_set_deferred_snapshots_material_occurrence_count(self, engine, store, dispatcher):
        """_set_deferred snapshots material_occurrence_count when enabled.

        After _handle_observe in enabled mode, occurrence_count_at_deferral
        should equal the material_occurrence_count (not occurrence_count).

        Requirements: 2.1, 2.5
        """
        dedupe_key = f"AMD:entry_alert:{uuid.uuid4().hex[:16]}"
        intent = _insert_intent(store, dedupe_key=dedupe_key, trigger_price="155.20")

        # Bump occurrence_count to 5, leave material at 1
        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE alert_intents SET occurrence_count = 5 WHERE id = :id"
            ), {"id": intent.id})

        fresh_intent = store.get_intent_by_id(intent.id)
        assert fresh_intent.occurrence_count == 5
        assert fresh_intent.material_occurrence_count == 1

        # First observation: sets deferral
        now = _BASE_TIME + timedelta(minutes=1)
        dispatcher._handle_observe(fresh_intent, now)

        # Verify snapshot is material_occurrence_count (1), not occurrence_count (5)
        snapshot = _get_deferred_snapshot(engine, fresh_intent.id)
        assert snapshot == 1, (
            f"Expected occurrence_count_at_deferral to be material_occurrence_count=1, "
            f"got {snapshot} (occurrence_count was 5)"
        )

    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "enabled")
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_secondary_price_change_check_bypassed(self, engine, store, dispatcher):
        """In enabled mode, _is_material_price_change check is bypassed.

        In enabled mode, the dispatcher does NOT perform a secondary price-change
        threshold check. Dedup relies solely on material_occurrence_count.
        Even if trigger_price on the intent differs from the audit log row,
        the code path for enabled mode never calls _is_material_price_change.

        We verify by confirming that with material_occ=2, a new would_dispatch is
        written regardless of price change magnitude (no secondary filter).

        Requirements: 2.2, 2.5
        """
        dedupe_key = f"AMD:entry_alert:{uuid.uuid4().hex[:16]}"
        intent = _insert_intent(store, dedupe_key=dedupe_key, trigger_price="155.20")

        now = _BASE_TIME + timedelta(minutes=1)

        # First observation → would_dispatch with material_occ=1
        dispatcher._handle_observe(intent, now)
        assert _count_would_dispatch(engine, intent.alert_intent_id) == 1

        # Simulate a material change at upsert time: bump material_occurrence_count to 2
        # but use a price that is only 0.2% different (would be filtered by
        # _is_material_price_change in disabled mode's secondary check)
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE alert_intents
                SET material_occurrence_count = 2,
                    occurrence_count = 3,
                    trigger_price = '155.50'
                WHERE id = :id
            """), {"id": intent.id})

        fresh_intent = store.get_intent_by_id(intent.id)
        assert fresh_intent.material_occurrence_count == 2

        # In enabled mode: passes material_occurrence_count=2 to has_would_dispatch.
        # No row exists for occ=2 → writes new would_dispatch.
        # The secondary _is_material_price_change check is NOT applied in enabled mode.
        dispatcher._handle_observe(fresh_intent, now)
        assert _count_would_dispatch(engine, intent.alert_intent_id) == 2

    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "enabled")
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_material_occurrence_count_in_audit_log(self, engine, store, dispatcher):
        """Enabled mode includes material_occurrence_count in audit log record.

        Requirements: 2.1
        """
        dedupe_key = f"AMD:entry_alert:{uuid.uuid4().hex[:16]}"
        intent = _insert_intent(store, dedupe_key=dedupe_key, trigger_price="155.20")

        now = _BASE_TIME + timedelta(minutes=1)
        dispatcher._handle_observe(intent, now)

        # Verify the audit log row includes material_occurrence_count
        with engine.begin() as conn:
            row = conn.execute(text("""
                SELECT material_occurrence_count FROM alert_dispatch_log
                WHERE alert_intent_id = :aid AND dispatch_status = 'would_dispatch'
            """), {"aid": intent.alert_intent_id}).fetchone()
        assert row is not None
        assert row[0] == 1


# ─── Tests: disabled mode ───────────────────────────────────────────────────


class TestObserveDisabledMode:
    """Tests for _handle_observe when ALERT_MATERIAL_OCCURRENCE_MODE == 'disabled'.

    Verifies existing behavior: occurrence_count-based dedup, secondary price
    change check applied.
    """

    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "disabled")
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_disabled_mode_dedup_uses_occurrence_count(self, engine, store, dispatcher):
        """Disabled mode: occurrence_count drives dedup.

        Same occurrence_count → dedup. Incremented occurrence_count → new would_dispatch.

        Requirements: 2.1, 2.2
        """
        dedupe_key = f"AMD:entry_alert:{uuid.uuid4().hex[:16]}"
        intent = _insert_intent(store, dedupe_key=dedupe_key, trigger_price="155.20")

        now = _BASE_TIME + timedelta(minutes=1)

        # First observation (occurrence_count=1) → write would_dispatch
        dispatcher._handle_observe(intent, now)
        assert _count_would_dispatch(engine, intent.alert_intent_id) == 1

        # Same state repeated → deduped (no new row)
        dispatcher._handle_observe(intent, now)
        assert _count_would_dispatch(engine, intent.alert_intent_id) == 1

        # Increment occurrence_count → new would_dispatch
        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE alert_intents SET occurrence_count = 2 WHERE id = :id"
            ), {"id": intent.id})
        fresh_intent = store.get_intent_by_id(intent.id)
        dispatcher._handle_observe(fresh_intent, now)
        assert _count_would_dispatch(engine, intent.alert_intent_id) == 2

    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "disabled")
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_disabled_mode_set_deferred_snapshots_occurrence_count(self, engine, store, dispatcher):
        """Disabled mode: _set_deferred snapshots occurrence_count (not material).

        After _handle_observe, occurrence_count_at_deferral should equal
        occurrence_count.

        Requirements: 2.1
        """
        dedupe_key = f"AMD:entry_alert:{uuid.uuid4().hex[:16]}"
        intent = _insert_intent(store, dedupe_key=dedupe_key, trigger_price="155.20")

        # Bump occurrence_count to 3
        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE alert_intents SET occurrence_count = 3 WHERE id = :id"
            ), {"id": intent.id})

        fresh_intent = store.get_intent_by_id(intent.id)
        assert fresh_intent.occurrence_count == 3

        now = _BASE_TIME + timedelta(minutes=1)
        dispatcher._handle_observe(fresh_intent, now)

        # Snapshot should be occurrence_count (3), not material_occurrence_count (1)
        snapshot = _get_deferred_snapshot(engine, fresh_intent.id)
        assert snapshot == 3, (
            f"Expected occurrence_count_at_deferral=3 in disabled mode, got {snapshot}"
        )
