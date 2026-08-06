"""Unit tests for AlertDispatcher re-arm on deferral expiry — Task 5.9.

Integration-style tests using a real in-memory SQLite engine with the full alert
dispatch schema. Tests verify:
1. Deferral expired + no material change → increment_material_occurrence called → new would_dispatch allowed
2. Deferral expired + material change already happened → no extra increment
3. Re-arm respects cooldown (at most once per interval)
4. Disabled mode does not call _check_rearm_on_deferral_expiry

Requirements: 3.3, 3.4, 7.3
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


def _insert_intent(store, engine, *, symbol="AMD", alert_type="entry_alert",
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


def _set_intent_deferred(engine, intent_id: int, *, deferred_until: datetime,
                         occurrence_count_at_deferral: int,
                         material_occurrence_count: int) -> None:
    """Directly set deferral state on an intent in the DB."""
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE alert_intents
            SET deferred_until = :deferred_until,
                occurrence_count_at_deferral = :occ_snap,
                material_occurrence_count = :mat_occ
            WHERE id = :id
        """), {
            "deferred_until": deferred_until.strftime(_ISO_FMT),
            "occ_snap": occurrence_count_at_deferral,
            "mat_occ": material_occurrence_count,
            "id": intent_id,
        })


def _count_would_dispatch(engine, alert_intent_id: str) -> int:
    """Count would_dispatch rows in audit log for an intent."""
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT COUNT(*) FROM alert_dispatch_log
            WHERE alert_intent_id = :aid AND dispatch_status = 'would_dispatch'
        """), {"aid": alert_intent_id}).fetchone()
    return row[0]


def _get_material_occurrence_count(engine, intent_id: int) -> int:
    """Read material_occurrence_count from the DB."""
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT material_occurrence_count FROM alert_intents WHERE id = :id
        """), {"id": intent_id}).fetchone()
    return row[0]


def _get_deferred_snapshot(engine, intent_id: int) -> int:
    """Read occurrence_count_at_deferral from the DB."""
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT occurrence_count_at_deferral FROM alert_intents WHERE id = :id
        """), {"id": intent_id}).fetchone()
    return row[0]


# ─── Tests: Deferral expired, no material change ────────────────────────────


class TestRearmDeferralExpiredNoMaterialChange:
    """Deferral expired + no material change → increment called → new would_dispatch allowed.

    Requirements: 3.3, 3.4
    """

    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "enabled")
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_rearm_increments_material_occurrence_count(self, engine, store, dispatcher):
        """When deferral expires with no independent material change, re-arm bumps counter.

        Setup: deferred_until in the past, material_occurrence_count == occurrence_count_at_deferral.
        Expected: _check_rearm_on_deferral_expiry increments material_occurrence_count by 1.

        Requirements: 3.3, 3.4
        """
        dedupe_key = f"AMD:entry_alert:{uuid.uuid4().hex[:16]}"
        intent = _insert_intent(store, engine, dedupe_key=dedupe_key)

        # Set deferral in the past with material_occ == snapshot
        past_deferral = _BASE_TIME - timedelta(minutes=5)
        _set_intent_deferred(
            engine, intent.id,
            deferred_until=past_deferral,
            occurrence_count_at_deferral=1,
            material_occurrence_count=1,
        )

        # Re-read intent
        fresh_intent = store.get_intent_by_id(intent.id)
        assert fresh_intent.material_occurrence_count == 1
        assert fresh_intent.occurrence_count_at_deferral == 1

        # Call re-arm check
        now = _BASE_TIME
        result_intent = dispatcher._check_rearm_on_deferral_expiry(fresh_intent, now)

        # material_occurrence_count should have been bumped to 2
        assert result_intent.material_occurrence_count == 2
        assert _get_material_occurrence_count(engine, intent.id) == 2

    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "enabled")
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_rearm_allows_new_would_dispatch(self, engine, store, dispatcher):
        """After re-arm bumps material counter, _handle_observe writes a new would_dispatch.

        Full flow: first observe → deferral set → time passes → deferral expires →
        re-arm bumps counter → second observe produces new would_dispatch.

        Requirements: 3.3, 3.4, 7.3
        """
        dedupe_key = f"AMD:entry_alert:{uuid.uuid4().hex[:16]}"
        intent = _insert_intent(store, engine, dedupe_key=dedupe_key)

        # First observation: writes would_dispatch with material_occ=1
        now = _BASE_TIME + timedelta(minutes=1)
        dispatcher._handle_observe(intent, now)
        assert _count_would_dispatch(engine, intent.alert_intent_id) == 1

        # Intent now has deferral set (from _handle_observe → _set_deferred)
        fresh_intent = store.get_intent_by_id(intent.id)
        assert fresh_intent.deferred_until is not None
        assert fresh_intent.occurrence_count_at_deferral == 1

        # Advance time past the deferral window
        now_after_deferral = now + timedelta(minutes=16)

        # Re-arm: deferral expired, no independent change → bump material counter
        rearmed_intent = dispatcher._check_rearm_on_deferral_expiry(fresh_intent, now_after_deferral)
        assert rearmed_intent.material_occurrence_count == 2

        # Now _handle_observe should write a second would_dispatch (material_occ=2)
        dispatcher._handle_observe(rearmed_intent, now_after_deferral)
        assert _count_would_dispatch(engine, intent.alert_intent_id) == 2


# ─── Tests: Deferral expired, material change already happened ──────────────


class TestRearmMaterialChangeAlreadyHappened:
    """Deferral expired + material change already happened → no extra increment.

    Requirements: 3.3, 3.4
    """

    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "enabled")
    def test_no_increment_when_material_change_already_occurred(self, engine, store, dispatcher):
        """If material_occurrence_count > occurrence_count_at_deferral, no re-arm bump.

        This means an independent material change already happened (e.g. source_level
        changed at upsert time), so the re-arm logic should NOT add an extra bump.

        Requirements: 3.3, 3.4
        """
        dedupe_key = f"AMD:entry_alert:{uuid.uuid4().hex[:16]}"
        intent = _insert_intent(store, engine, dedupe_key=dedupe_key)

        # Set deferral in the past: material_occ=2 > snapshot=1
        past_deferral = _BASE_TIME - timedelta(minutes=5)
        _set_intent_deferred(
            engine, intent.id,
            deferred_until=past_deferral,
            occurrence_count_at_deferral=1,
            material_occurrence_count=2,
        )

        fresh_intent = store.get_intent_by_id(intent.id)
        assert fresh_intent.material_occurrence_count == 2
        assert fresh_intent.occurrence_count_at_deferral == 1

        # Call re-arm check
        now = _BASE_TIME
        result_intent = dispatcher._check_rearm_on_deferral_expiry(fresh_intent, now)

        # Should return intent unchanged — no extra bump
        assert result_intent.material_occurrence_count == 2
        assert _get_material_occurrence_count(engine, intent.id) == 2

    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "enabled")
    def test_no_increment_exact_boundary(self, engine, store, dispatcher):
        """material_occurrence_count=3 vs snapshot=2 → no bump (already changed).

        Covers case where multiple independent material changes occurred since deferral.
        """
        dedupe_key = f"AMD:entry_alert:{uuid.uuid4().hex[:16]}"
        intent = _insert_intent(store, engine, dedupe_key=dedupe_key)

        past_deferral = _BASE_TIME - timedelta(minutes=5)
        _set_intent_deferred(
            engine, intent.id,
            deferred_until=past_deferral,
            occurrence_count_at_deferral=2,
            material_occurrence_count=3,
        )

        fresh_intent = store.get_intent_by_id(intent.id)
        now = _BASE_TIME
        result_intent = dispatcher._check_rearm_on_deferral_expiry(fresh_intent, now)

        assert result_intent.material_occurrence_count == 3
        assert _get_material_occurrence_count(engine, intent.id) == 3


# ─── Tests: Re-arm respects cooldown ────────────────────────────────────────


class TestRearmRespectsCooldown:
    """Re-arm respects cooldown: at most once per interval.

    After re-arm, _handle_observe writes a new would_dispatch and sets a new deferral.
    Subsequent calls within the new deferral window are blocked by _is_deferred.

    Requirements: 3.3, 3.4, 7.3
    """

    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "enabled")
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_rearm_then_subsequent_calls_stay_deferred(self, engine, store, dispatcher):
        """After re-arm + new observe, the intent enters a new deferral period.

        Flow:
        1. First observe → would_dispatch, deferral set (snapshot=1)
        2. Deferral expires → re-arm bumps to material_occ=2
        3. Second observe → would_dispatch, new deferral set (snapshot=2)
        4. Within new deferral: _is_deferred returns True → no more dispatches

        Requirements: 3.3, 7.3
        """
        dedupe_key = f"AMD:entry_alert:{uuid.uuid4().hex[:16]}"
        intent = _insert_intent(store, engine, dedupe_key=dedupe_key)

        # Step 1: First observation
        t1 = _BASE_TIME + timedelta(minutes=1)
        dispatcher._handle_observe(intent, t1)
        assert _count_would_dispatch(engine, intent.alert_intent_id) == 1

        # Verify deferral set with snapshot=1
        intent_after_first = store.get_intent_by_id(intent.id)
        assert intent_after_first.occurrence_count_at_deferral == 1

        # Step 2: Advance past deferral → re-arm
        t2 = t1 + timedelta(minutes=16)  # past the 15-min cooldown
        rearmed = dispatcher._check_rearm_on_deferral_expiry(intent_after_first, t2)
        assert rearmed.material_occurrence_count == 2

        # Step 3: Second observe writes new would_dispatch, sets new deferral
        dispatcher._handle_observe(rearmed, t2)
        assert _count_would_dispatch(engine, intent.alert_intent_id) == 2

        # Verify new deferral snapshot = 2
        intent_after_second = store.get_intent_by_id(intent.id)
        assert intent_after_second.occurrence_count_at_deferral == 2
        assert intent_after_second.deferred_until is not None

        # Step 4: Within new deferral window, _is_deferred returns True
        t3 = t2 + timedelta(minutes=5)  # 5 min into the new 15-min window
        assert dispatcher._is_deferred(intent_after_second, t3) is True

    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "enabled")
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_second_rearm_call_within_deferral_no_effect(self, engine, store, dispatcher):
        """Calling _check_rearm_on_deferral_expiry on an intent still within deferral is a no-op.

        The guard `deferred_until > now` prevents re-arm when deferral hasn't expired.

        Requirements: 7.3
        """
        dedupe_key = f"AMD:entry_alert:{uuid.uuid4().hex[:16]}"
        intent = _insert_intent(store, engine, dedupe_key=dedupe_key)

        # First observe → set deferral
        t1 = _BASE_TIME + timedelta(minutes=1)
        dispatcher._handle_observe(intent, t1)

        intent_after = store.get_intent_by_id(intent.id)

        # Call re-arm while still within deferral window (deferred_until > now)
        t_within = t1 + timedelta(minutes=5)
        result = dispatcher._check_rearm_on_deferral_expiry(intent_after, t_within)

        # Should return unchanged (deferred_until > now guard prevents action)
        assert result.material_occurrence_count == intent_after.material_occurrence_count
        assert _get_material_occurrence_count(engine, intent.id) == 1


# ─── Tests: Disabled mode ───────────────────────────────────────────────────


class TestRearmDisabledMode:
    """Disabled mode does not call re-arm logic.

    When ALERT_MATERIAL_OCCURRENCE_MODE == "disabled", _check_rearm_on_deferral_expiry
    returns the intent unchanged (early return).

    Requirements: 3.3, 3.4
    """

    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "disabled")
    def test_disabled_mode_returns_intent_unchanged(self, engine, store, dispatcher):
        """In disabled mode, _check_rearm_on_deferral_expiry is a no-op.

        Even if deferral has expired and material_occ == snapshot, no increment
        occurs because the mode gate is disabled.
        """
        dedupe_key = f"AMD:entry_alert:{uuid.uuid4().hex[:16]}"
        intent = _insert_intent(store, engine, dedupe_key=dedupe_key)

        # Set deferral in the past: conditions that WOULD trigger re-arm in enabled mode
        past_deferral = _BASE_TIME - timedelta(minutes=5)
        _set_intent_deferred(
            engine, intent.id,
            deferred_until=past_deferral,
            occurrence_count_at_deferral=1,
            material_occurrence_count=1,
        )

        fresh_intent = store.get_intent_by_id(intent.id)
        now = _BASE_TIME

        result_intent = dispatcher._check_rearm_on_deferral_expiry(fresh_intent, now)

        # Should return unchanged — disabled mode short-circuits
        assert result_intent.material_occurrence_count == 1
        assert _get_material_occurrence_count(engine, intent.id) == 1

    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "disabled")
    def test_disabled_mode_no_db_write(self, engine, store, dispatcher):
        """In disabled mode, no DB write occurs (increment_material_occurrence not called).

        Verify by checking that the material_occurrence_count in DB is unchanged.
        """
        dedupe_key = f"AMD:entry_alert:{uuid.uuid4().hex[:16]}"
        intent = _insert_intent(store, engine, dedupe_key=dedupe_key)

        past_deferral = _BASE_TIME - timedelta(minutes=5)
        _set_intent_deferred(
            engine, intent.id,
            deferred_until=past_deferral,
            occurrence_count_at_deferral=1,
            material_occurrence_count=1,
        )

        fresh_intent = store.get_intent_by_id(intent.id)
        now = _BASE_TIME

        # Call re-arm in disabled mode
        dispatcher._check_rearm_on_deferral_expiry(fresh_intent, now)

        # DB should be unchanged
        db_mat_occ = _get_material_occurrence_count(engine, intent.id)
        assert db_mat_occ == 1
