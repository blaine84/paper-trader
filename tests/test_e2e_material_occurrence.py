"""End-to-end tests for material occurrence deduplication — Tasks 8.1–8.6.

Full integration tests exercising the complete flow from intent recording
through dispatch evaluation using real SQLite engine, real AlertIntentStore,
and real AlertDispatcher.

These tests validate the motivating bug case and key scenarios for the
alert-dispatch-dedupe spec.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from utils.alert_dispatch_schema import init_alert_dispatch_schema
from utils.alert_intent_store import AlertIntentStore, build_dedupe_key
from utils.alert_dispatcher import AlertDispatcher


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    eng = create_engine("sqlite://", echo=False)
    init_alert_dispatch_schema(eng)
    return eng


@pytest.fixture
def store(engine):
    return AlertIntentStore(engine)


@pytest.fixture
def dispatcher(engine, store):
    begin_pm = MagicMock(return_value=True)
    end_pm = MagicMock()
    return AlertDispatcher(
        engine=engine,
        intent_store=store,
        begin_pm_cycle=begin_pm,
        end_pm_cycle=end_pm,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _make_intent_data(
    symbol="AMD",
    alert_type="entry_alert",
    trigger_price="155.20",
    source_level="resistance_155",
    direction="long",
    **overrides,
):
    """Factory for creating intent data dicts with sensible defaults."""
    now = datetime.utcnow()
    data = {
        "symbol": symbol,
        "alert_type": alert_type,
        "direction": direction,
        "trigger_price": trigger_price,
        "source_level": source_level,
        "urgency": "medium",
        "reason": "Price near resistance",
        "dedupe_key": build_dedupe_key(symbol, alert_type, source_level),
        "filter_status": "unclassified",
        "first_seen_at": now.strftime(_ISO_FMT),
        "last_seen_at": now.strftime(_ISO_FMT),
        "expiration_at": (now + timedelta(hours=4)).strftime(_ISO_FMT),
    }
    data.update(overrides)
    return data


def _classify_intent(store, intent, status="passed", urgency="high"):
    """Classify an intent so it becomes eligible for dispatch evaluation."""
    store.update_classification(intent.id, status, urgency)


# ---------------------------------------------------------------------------
# Task 8.1: E2E test — AMD resistance scenario (motivating bug case)
# ---------------------------------------------------------------------------


class TestE2EAmdResistanceScenario:
    """E2E: AMD entry_alert at resistance_155, price drifting over 5 ticks.

    ALERT_MATERIAL_OCCURRENCE_MODE=enabled
    PM_ALERT_DISPATCH_MODE=observe

    Simulates:
      AMD entry_alert at resistance_155
      Price: 155.20 → 155.22 → 155.18 → 155.25 → 155.21 over 5 ticks

    Asserts:
      - Exactly 1 would_dispatch row in alert_dispatch_log
      - last_seen_at is the latest timestamp
      - occurrence_count = 5
      - material_occurrence_count = 1

    Validates: All requirements, Acceptance Criteria 1-4
    """

    @patch("utils.alert_dispatcher.AlertDispatcher._is_market_hours", return_value=True)
    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "enabled")
    @patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "observe")
    @patch("utils.gate_config.PM_ALERT_MODE_ENTRY_ALERT", "")
    @patch("utils.gate_config.PM_ALERT_MODE_BREAKOUT", "")
    @patch("utils.gate_config.PM_ALERT_MODE_RAPID_MOVE", "")
    @patch("utils.gate_config.PM_ALERT_MODE_TARGET_HIT", "")
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_ENTRY_ALERT_MINUTES", 60)
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_amd_resistance_5_ticks_one_would_dispatch(
        self, mock_market, engine, store, dispatcher
    ):
        """Motivating bug case: AMD at resistance_155 observed 5 times with
        ordinary price drift produces exactly 1 would_dispatch row."""

        # Tick prices — all within 0.5% of each other (0.5% of 155.20 = 0.776)
        # Max drift from any tick to the next: 155.25 - 155.18 = 0.07 / 155.18 = 0.045%
        tick_prices = ["155.20", "155.22", "155.18", "155.25", "155.21"]

        # Use real current time so freshness checks pass (dispatcher uses utcnow)
        now = datetime.utcnow()

        # --- Tick 1: Create the initial alert intent ---
        intent = store.record_or_update_intent(
            _make_intent_data(
                trigger_price=tick_prices[0],
                first_seen_at=now.strftime(_ISO_FMT),
                last_seen_at=now.strftime(_ISO_FMT),
            )
        )

        # Classify as "passed" with urgency "high" (simulating LLM filter completion)
        _classify_intent(store, intent)

        # Run dispatcher after first observation
        dispatcher.evaluate_and_dispatch()

        # --- Ticks 2-5: Re-record same alert with slight price drift ---
        for i in range(1, 5):
            tick_time = datetime.utcnow()
            store.record_or_update_intent(
                _make_intent_data(
                    trigger_price=tick_prices[i],
                    last_seen_at=tick_time.strftime(_ISO_FMT),
                )
            )
            # Run dispatcher after each re-record
            dispatcher.evaluate_and_dispatch()

        # --- Assertions ---

        # 1. Exactly 1 would_dispatch row in alert_dispatch_log
        with engine.begin() as conn:
            wd_rows = conn.execute(text(
                "SELECT dispatch_status, occurrence_count, material_occurrence_count "
                "FROM alert_dispatch_log "
                "WHERE alert_intent_id = :aid AND dispatch_status = 'would_dispatch'"
            ), {"aid": intent.alert_intent_id}).fetchall()

        assert len(wd_rows) == 1, (
            f"Expected exactly 1 would_dispatch row, got {len(wd_rows)}. "
            f"Rows: {wd_rows}"
        )

        # 2. Verify the intent's final state from DB
        with engine.begin() as conn:
            final_row = conn.execute(text(
                "SELECT occurrence_count, material_occurrence_count, "
                "last_seen_at, trigger_price "
                "FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()

        # occurrence_count = 5 (incremented on every tick)
        assert final_row[0] == 5, (
            f"Expected occurrence_count=5, got {final_row[0]}"
        )

        # material_occurrence_count = 1 (no material change detected)
        assert final_row[1] == 1, (
            f"Expected material_occurrence_count=1, got {final_row[1]}"
        )

        # last_seen_at is the latest timestamp (most recent tick)
        # Verify it's more recent than the initial timestamp
        from utils.alert_intent_store import _parse_iso_dt
        final_last_seen = _parse_iso_dt(final_row[2])
        assert final_last_seen >= now, (
            f"Expected last_seen_at >= initial time, got {final_row[2]}"
        )

        # trigger_price reflects the last tick's price
        assert final_row[3] == "155.21", (
            f"Expected trigger_price=155.21, got {final_row[3]}"
        )


# ---------------------------------------------------------------------------
# Task 8.2: E2E test — material change triggers new dispatch
# ---------------------------------------------------------------------------


class TestE2EMaterialChangeTrigger:
    """E2E: material change (source_level change) bypasses deferral and produces new dispatch.

    ALERT_MATERIAL_OCCURRENCE_MODE=enabled
    PM_ALERT_DISPATCH_MODE=observe

    Flow:
      1. AMD entry_alert at resistance_155, price=155.20 → first would_dispatch + deferral set
      2. Next tick: re-record with source_level changed to "resistance_157" (same dedupe_key)
      3. Run dispatcher → material change detected, deferral bypassed, second would_dispatch

    Asserts:
      - 2 would_dispatch rows
      - material_occurrence_count=2
      - occurrence_count=2

    Validates: Requirements 3.1, 3.2, 3.5
    """

    @patch("utils.alert_dispatcher.AlertDispatcher._is_market_hours", return_value=True)
    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "enabled")
    @patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "observe")
    @patch("utils.gate_config.PM_ALERT_MODE_ENTRY_ALERT", "")
    @patch("utils.gate_config.PM_ALERT_MODE_BREAKOUT", "")
    @patch("utils.gate_config.PM_ALERT_MODE_RAPID_MOVE", "")
    @patch("utils.gate_config.PM_ALERT_MODE_TARGET_HIT", "")
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_ENTRY_ALERT_MINUTES", 60)
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_material_change_source_level_triggers_new_dispatch(
        self, mock_market, engine, store, dispatcher
    ):
        """source_level change from resistance_155 to resistance_157 is detected
        as material at upsert time, increments material_occurrence_count, and
        bypasses deferral to produce a second would_dispatch."""

        now = datetime.utcnow()

        # Use same dedupe_key for both — we pass it explicitly so that the
        # source_level change doesn't change the key (test the material detection,
        # not the key change path)
        dedupe_key = build_dedupe_key("AMD", "entry_alert", "resistance_155")

        # --- Tick 1: Initial observation at resistance_155 ---
        intent = store.record_or_update_intent(
            _make_intent_data(
                trigger_price="155.20",
                source_level="resistance_155",
                dedupe_key=dedupe_key,
                first_seen_at=now.strftime(_ISO_FMT),
                last_seen_at=now.strftime(_ISO_FMT),
            )
        )

        # Classify so dispatcher will evaluate
        _classify_intent(store, intent)

        # Run dispatcher → first would_dispatch + deferral set
        dispatcher.evaluate_and_dispatch()

        # Verify first would_dispatch written
        with engine.begin() as conn:
            wd_rows = conn.execute(text(
                "SELECT dispatch_status FROM alert_dispatch_log "
                "WHERE alert_intent_id = :aid AND dispatch_status = 'would_dispatch'"
            ), {"aid": intent.alert_intent_id}).fetchall()
        assert len(wd_rows) == 1, f"Expected 1 would_dispatch after tick 1, got {len(wd_rows)}"

        # Verify deferral was set
        with engine.begin() as conn:
            deferred_row = conn.execute(text(
                "SELECT deferred_until, occurrence_count_at_deferral "
                "FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()
        assert deferred_row[0] is not None, "Expected deferred_until to be set after first dispatch"
        assert deferred_row[1] == 1, (
            f"Expected occurrence_count_at_deferral=1, got {deferred_row[1]}"
        )

        # --- Tick 2: Same dedupe_key but source_level changes to resistance_157 ---
        tick2_time = datetime.utcnow()
        store.record_or_update_intent(
            _make_intent_data(
                trigger_price="157.50",
                source_level="resistance_157",
                dedupe_key=dedupe_key,
                last_seen_at=tick2_time.strftime(_ISO_FMT),
            )
        )

        # Run dispatcher → material change detected, deferral bypassed, second would_dispatch
        dispatcher.evaluate_and_dispatch()

        # --- Assertions ---

        # 1. Exactly 2 would_dispatch rows
        with engine.begin() as conn:
            wd_rows = conn.execute(text(
                "SELECT dispatch_status, occurrence_count, material_occurrence_count "
                "FROM alert_dispatch_log "
                "WHERE alert_intent_id = :aid AND dispatch_status = 'would_dispatch'"
            ), {"aid": intent.alert_intent_id}).fetchall()

        assert len(wd_rows) == 2, (
            f"Expected 2 would_dispatch rows (one per material occurrence), "
            f"got {len(wd_rows)}. Rows: {wd_rows}"
        )

        # 2. Verify intent's final state
        with engine.begin() as conn:
            final_row = conn.execute(text(
                "SELECT occurrence_count, material_occurrence_count "
                "FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()

        # occurrence_count = 2 (2 observations total)
        assert final_row[0] == 2, (
            f"Expected occurrence_count=2, got {final_row[0]}"
        )

        # material_occurrence_count = 2 (source_level changed → material)
        assert final_row[1] == 2, (
            f"Expected material_occurrence_count=2, got {final_row[1]}"
        )


# ---------------------------------------------------------------------------
# Task 8.3: E2E test — disabled mode preserves existing behavior
# ---------------------------------------------------------------------------


class TestE2EDisabledModePreservesExistingBehavior:
    """E2E: disabled mode preserves existing (buggy) behavior.

    ALERT_MATERIAL_OCCURRENCE_MODE=disabled
    PM_ALERT_DISPATCH_MODE=observe

    Simulates:
      AMD entry_alert at resistance_155
      Price: 155.20 → 155.22 → 155.18 → 155.25 → 155.21 over 5 ticks

    In disabled mode, the existing occurrence_count-based dedup applies:
      - First observe writes would_dispatch with occurrence_count=1, sets deferral (snapshot=1)
      - When occurrence_count increments to 2,3,4,5 the legacy _is_deferred() comparison
        sees occ > snapshot → bypasses deferral → allows new would_dispatch
      - This is the existing (buggy) behavior the flag is supposed to fix.

    Asserts:
      - More than 1 would_dispatch row (demonstrates the bug exists in disabled mode)
      - material_occurrence_count column IS populated (always populates regardless of mode)
        and equals 1 (no material change occurred)
      - Proves: (a) disabled mode = existing behavior, (b) material_occurrence_count
        populates but doesn't affect dispatch

    Validates: Requirements 8.2
    """

    @patch("utils.alert_dispatcher.AlertDispatcher._is_market_hours", return_value=True)
    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "disabled")
    @patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "observe")
    @patch("utils.gate_config.PM_ALERT_MODE_ENTRY_ALERT", "")
    @patch("utils.gate_config.PM_ALERT_MODE_BREAKOUT", "")
    @patch("utils.gate_config.PM_ALERT_MODE_RAPID_MOVE", "")
    @patch("utils.gate_config.PM_ALERT_MODE_TARGET_HIT", "")
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_ENTRY_ALERT_MINUTES", 60)
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_disabled_mode_amd_resistance_multiple_would_dispatch(
        self, mock_market, engine, store, dispatcher
    ):
        """Disabled mode: AMD at resistance_155 observed 5 times with ordinary
        price drift produces MORE than 1 would_dispatch row — demonstrating the
        bug exists in disabled mode. material_occurrence_count still populates
        (=1) but doesn't affect dispatch decisions."""

        # Tick prices — all within 0.5% of each other
        tick_prices = ["155.20", "155.22", "155.18", "155.25", "155.21"]

        now = datetime.utcnow()

        # --- Tick 1: Create the initial alert intent ---
        intent = store.record_or_update_intent(
            _make_intent_data(
                trigger_price=tick_prices[0],
                first_seen_at=now.strftime(_ISO_FMT),
                last_seen_at=now.strftime(_ISO_FMT),
            )
        )

        # Classify as "passed" with urgency "high"
        _classify_intent(store, intent)

        # Run dispatcher after first observation
        dispatcher.evaluate_and_dispatch()

        # --- Ticks 2-5: Re-record with slight price drift ---
        for i in range(1, 5):
            tick_time = datetime.utcnow()
            store.record_or_update_intent(
                _make_intent_data(
                    trigger_price=tick_prices[i],
                    last_seen_at=tick_time.strftime(_ISO_FMT),
                )
            )
            # Run dispatcher after each re-record
            dispatcher.evaluate_and_dispatch()

        # --- Assertions ---

        # 1. More than 1 would_dispatch row (the bug in disabled mode)
        with engine.begin() as conn:
            wd_rows = conn.execute(text(
                "SELECT dispatch_status, occurrence_count, material_occurrence_count "
                "FROM alert_dispatch_log "
                "WHERE alert_intent_id = :aid AND dispatch_status = 'would_dispatch'"
            ), {"aid": intent.alert_intent_id}).fetchall()

        assert len(wd_rows) > 1, (
            f"Expected MORE than 1 would_dispatch row in disabled mode "
            f"(demonstrating the existing bug), got {len(wd_rows)}. "
            f"Rows: {wd_rows}"
        )

        # 2. material_occurrence_count IS populated and equals 1
        #    (always populates regardless of mode, but no material change occurred)
        with engine.begin() as conn:
            final_row = conn.execute(text(
                "SELECT occurrence_count, material_occurrence_count "
                "FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()

        # occurrence_count should be 5 (increments every tick)
        assert final_row[0] == 5, (
            f"Expected occurrence_count=5, got {final_row[0]}"
        )

        # material_occurrence_count = 1 (populates but no material change)
        assert final_row[1] == 1, (
            f"Expected material_occurrence_count=1 (populates but doesn't affect "
            f"dispatch in disabled mode), got {final_row[1]}"
        )

        # 3. Verify material_occurrence_count is populated on audit log rows too
        with engine.begin() as conn:
            audit_mat_counts = conn.execute(text(
                "SELECT material_occurrence_count FROM alert_dispatch_log "
                "WHERE alert_intent_id = :aid AND dispatch_status = 'would_dispatch'"
            ), {"aid": intent.alert_intent_id}).fetchall()

        # All audit rows should have material_occurrence_count = 1
        for row in audit_mat_counts:
            assert row[0] == 1, (
                f"Expected material_occurrence_count=1 on audit log row, got {row[0]}"
            )


# ---------------------------------------------------------------------------
# Task 8.4: E2E test — cooldown expiry re-arm
# ---------------------------------------------------------------------------


class TestE2ECooldownExpiryRearm:
    """E2E: cooldown expiry re-arms a stable alert for a second would_dispatch.

    ALERT_MATERIAL_OCCURRENCE_MODE=enabled
    PM_ALERT_DISPATCH_MODE=observe

    Flow:
      1. AMD entry_alert at resistance_155, price=155.20 → first would_dispatch + deferral (15 min)
      2. Advance time: set deferred_until to past in DB (simulate 15+ min passing)
      3. Update last_seen_at to recent (so freshness check passes)
      4. Run dispatcher → re-arm triggers (bumps material_occurrence_count 1→2) → second would_dispatch + new deferral

    Asserts:
      - 2 would_dispatch rows total
      - material_occurrence_count = 2
      - occurrence_count_at_deferral = 2 (new deferral snapshot)
      - deferred_until is set to a future time (new deferral window)

    Validates: Requirements 3.3, 3.4, 7.3
    """

    @patch("utils.alert_dispatcher.AlertDispatcher._is_market_hours", return_value=True)
    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "enabled")
    @patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "observe")
    @patch("utils.gate_config.PM_ALERT_MODE_ENTRY_ALERT", "")
    @patch("utils.gate_config.PM_ALERT_MODE_BREAKOUT", "")
    @patch("utils.gate_config.PM_ALERT_MODE_RAPID_MOVE", "")
    @patch("utils.gate_config.PM_ALERT_MODE_TARGET_HIT", "")
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_ENTRY_ALERT_MINUTES", 60)
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_cooldown_expiry_rearm_bumps_material_counter(
        self, mock_market, engine, store, dispatcher
    ):
        """Deferral expires with condition still active → re-arm bumps
        material_occurrence_count, second would_dispatch allowed, new deferral set."""

        now = datetime.utcnow()

        # --- Step 1: Create alert intent and dispatch first time ---
        intent = store.record_or_update_intent(
            _make_intent_data(
                trigger_price="155.20",
                source_level="resistance_155",
                first_seen_at=now.strftime(_ISO_FMT),
                last_seen_at=now.strftime(_ISO_FMT),
            )
        )

        # Classify so dispatcher will evaluate
        _classify_intent(store, intent)

        # Run dispatcher → first would_dispatch + deferral set (15 min cooldown)
        dispatcher.evaluate_and_dispatch()

        # Verify first would_dispatch written
        with engine.begin() as conn:
            wd_rows = conn.execute(text(
                "SELECT dispatch_status FROM alert_dispatch_log "
                "WHERE alert_intent_id = :aid AND dispatch_status = 'would_dispatch'"
            ), {"aid": intent.alert_intent_id}).fetchall()
        assert len(wd_rows) == 1, f"Expected 1 would_dispatch after first dispatch, got {len(wd_rows)}"

        # Verify deferral was set with snapshot = 1
        with engine.begin() as conn:
            deferred_row = conn.execute(text(
                "SELECT deferred_until, occurrence_count_at_deferral "
                "FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()
        assert deferred_row[0] is not None, "Expected deferred_until to be set"
        assert deferred_row[1] == 1, (
            f"Expected occurrence_count_at_deferral=1, got {deferred_row[1]}"
        )

        # --- Step 2: Advance time past deferral ---
        # Directly set deferred_until to a past timestamp in the DB (simulate 15+ min passing)
        past_deferral = (now - timedelta(minutes=1)).strftime(_ISO_FMT)
        # Also update last_seen_at to recent (so freshness check passes)
        recent_time = datetime.utcnow().strftime(_ISO_FMT)

        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE alert_intents
                SET deferred_until = :past_deferral,
                    last_seen_at = :recent_time
                WHERE id = :id
            """), {
                "past_deferral": past_deferral,
                "recent_time": recent_time,
                "id": intent.id,
            })

        # --- Step 3: Run dispatcher again → re-arm triggers ---
        dispatcher.evaluate_and_dispatch()

        # --- Step 4: Assertions ---

        # 1. Exactly 2 would_dispatch rows total
        with engine.begin() as conn:
            wd_rows = conn.execute(text(
                "SELECT dispatch_status, occurrence_count, material_occurrence_count "
                "FROM alert_dispatch_log "
                "WHERE alert_intent_id = :aid AND dispatch_status = 'would_dispatch'"
            ), {"aid": intent.alert_intent_id}).fetchall()

        assert len(wd_rows) == 2, (
            f"Expected 2 would_dispatch rows (one initial + one re-arm), "
            f"got {len(wd_rows)}. Rows: {wd_rows}"
        )

        # 2. material_occurrence_count = 2 (re-arm bumped it)
        with engine.begin() as conn:
            final_row = conn.execute(text(
                "SELECT occurrence_count, material_occurrence_count, "
                "occurrence_count_at_deferral, deferred_until "
                "FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()

        assert final_row[1] == 2, (
            f"Expected material_occurrence_count=2 after re-arm, got {final_row[1]}"
        )

        # 3. occurrence_count_at_deferral = 2 (new deferral snapshot)
        assert final_row[2] == 2, (
            f"Expected occurrence_count_at_deferral=2 (new deferral snapshot), "
            f"got {final_row[2]}"
        )

        # 4. deferred_until is set to a future time (new deferral window)
        from utils.alert_intent_store import _parse_iso_dt
        new_deferred_until = _parse_iso_dt(final_row[3])
        assert new_deferred_until > datetime.utcnow(), (
            f"Expected deferred_until to be in the future (new deferral), "
            f"got {final_row[3]}"
        )


# ---------------------------------------------------------------------------
# Task 8.5: E2E test — genuinely re-armed alert after condition clears
# ---------------------------------------------------------------------------


class TestE2ERearmAfterConditionClears:
    """E2E: condition disappears (intent expires), new observation with same dedupe_key.

    ALERT_MATERIAL_OCCURRENCE_MODE=enabled
    PM_ALERT_DISPATCH_MODE=observe

    Flow:
      1. AMD entry_alert at resistance_155, price=155.20 → first would_dispatch + deferral set
      2. Simulate condition disappearing: set intent dispatch_status = 'expired' in DB
      3. New observation with SAME dedupe_key arrives (condition re-appears after clearing)
      4. Since record_or_update_intent only matches active statuses
         ('pending', 'dispatched', 'claimed_by_scheduled'), the expired row is NOT found
         → a new INSERT occurs with occurrence_count=1, material_occurrence_count=1
      5. Classify the new intent and run dispatcher → new would_dispatch

    Asserts:
      - 2 would_dispatch rows total (one from first intent, one from new intent)
      - New intent has material_occurrence_count=1 (fresh start)
      - New intent has occurrence_count=1 (fresh start)
      - The fix does NOT suppress this genuinely re-armed alert

    Validates: Requirements 3.3, 7.3, 7.4
    """

    @patch("utils.alert_dispatcher.AlertDispatcher._is_market_hours", return_value=True)
    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "enabled")
    @patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "observe")
    @patch("utils.gate_config.PM_ALERT_MODE_ENTRY_ALERT", "")
    @patch("utils.gate_config.PM_ALERT_MODE_BREAKOUT", "")
    @patch("utils.gate_config.PM_ALERT_MODE_RAPID_MOVE", "")
    @patch("utils.gate_config.PM_ALERT_MODE_TARGET_HIT", "")
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_ENTRY_ALERT_MINUTES", 60)
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_condition_clears_then_reappears_produces_new_dispatch(
        self, mock_market, engine, store, dispatcher
    ):
        """Condition disappears (intent expires), same dedupe_key re-appears →
        new INSERT (not UPDATE), material_occurrence_count=1, new would_dispatch.
        Validates the fix does NOT suppress genuinely re-armed alerts."""

        now = datetime.utcnow()
        dedupe_key = build_dedupe_key("AMD", "entry_alert", "resistance_155")

        # --- Step 1: Create initial alert intent and dispatch ---
        intent_1 = store.record_or_update_intent(
            _make_intent_data(
                trigger_price="155.20",
                source_level="resistance_155",
                dedupe_key=dedupe_key,
                first_seen_at=now.strftime(_ISO_FMT),
                last_seen_at=now.strftime(_ISO_FMT),
            )
        )

        # Classify so dispatcher will evaluate
        _classify_intent(store, intent_1)

        # Run dispatcher → first would_dispatch + deferral set
        dispatcher.evaluate_and_dispatch()

        # Verify first would_dispatch written
        with engine.begin() as conn:
            wd_rows = conn.execute(text(
                "SELECT dispatch_status FROM alert_dispatch_log "
                "WHERE alert_intent_id = :aid AND dispatch_status = 'would_dispatch'"
            ), {"aid": intent_1.alert_intent_id}).fetchall()
        assert len(wd_rows) == 1, (
            f"Expected 1 would_dispatch after first dispatch, got {len(wd_rows)}"
        )

        # --- Step 2: Simulate condition disappearing ---
        # Directly set the intent's dispatch_status to 'expired' in DB
        # (as if the condition cleared and the intent was expired)
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE alert_intents
                SET dispatch_status = 'expired'
                WHERE id = :id
            """), {"id": intent_1.id})

        # Verify the intent is now expired
        with engine.begin() as conn:
            status_row = conn.execute(text(
                "SELECT dispatch_status FROM alert_intents WHERE id = :id"
            ), {"id": intent_1.id}).fetchone()
        assert status_row[0] == "expired", (
            f"Expected intent status='expired', got {status_row[0]}"
        )

        # --- Step 3: New observation with SAME dedupe_key arrives ---
        # Since record_or_update_intent only matches active statuses
        # ('pending', 'dispatched', 'claimed_by_scheduled'), the expired
        # row won't be found → a new INSERT occurs
        new_time = datetime.utcnow()
        intent_2 = store.record_or_update_intent(
            _make_intent_data(
                trigger_price="155.20",
                source_level="resistance_155",
                dedupe_key=dedupe_key,
                first_seen_at=new_time.strftime(_ISO_FMT),
                last_seen_at=new_time.strftime(_ISO_FMT),
            )
        )

        # Verify this is a NEW row (different id from intent_1)
        assert intent_2.id != intent_1.id, (
            f"Expected new INSERT (different id), but got same id={intent_2.id}. "
            f"The expired intent should not have been updated."
        )

        # Verify new intent has fresh counters
        assert intent_2.occurrence_count == 1, (
            f"Expected occurrence_count=1 on new intent, got {intent_2.occurrence_count}"
        )
        assert intent_2.material_occurrence_count == 1, (
            f"Expected material_occurrence_count=1 on new intent, "
            f"got {intent_2.material_occurrence_count}"
        )

        # --- Step 4: Classify and run dispatcher on new intent ---
        _classify_intent(store, intent_2)

        # Run dispatcher → new would_dispatch for the fresh intent
        dispatcher.evaluate_and_dispatch()

        # --- Step 5: Assertions ---

        # 1. Total 2 would_dispatch rows (one from first intent, one from new intent)
        with engine.begin() as conn:
            all_wd_rows = conn.execute(text(
                "SELECT alert_intent_id, dispatch_status, occurrence_count, "
                "material_occurrence_count "
                "FROM alert_dispatch_log "
                "WHERE dispatch_status = 'would_dispatch'"
            )).fetchall()

        assert len(all_wd_rows) == 2, (
            f"Expected 2 would_dispatch rows total (one from each intent), "
            f"got {len(all_wd_rows)}. Rows: {all_wd_rows}"
        )

        # 2. Verify they belong to different alert_intent_ids
        intent_ids = {row[0] for row in all_wd_rows}
        assert len(intent_ids) == 2, (
            f"Expected would_dispatch rows from 2 different intents, "
            f"got intent_ids: {intent_ids}"
        )
        assert intent_1.alert_intent_id in intent_ids
        assert intent_2.alert_intent_id in intent_ids

        # 3. New intent in DB has material_occurrence_count=1 (fresh start)
        with engine.begin() as conn:
            new_intent_row = conn.execute(text(
                "SELECT occurrence_count, material_occurrence_count "
                "FROM alert_intents WHERE id = :id"
            ), {"id": intent_2.id}).fetchone()

        assert new_intent_row[0] == 1, (
            f"Expected new intent occurrence_count=1, got {new_intent_row[0]}"
        )
        assert new_intent_row[1] == 1, (
            f"Expected new intent material_occurrence_count=1, got {new_intent_row[1]}"
        )

        # 4. The fix does NOT suppress this genuinely re-armed alert
        # (verified by the presence of the second would_dispatch above)
        # Also verify the new intent's dispatch log row has the correct counters
        with engine.begin() as conn:
            new_wd_row = conn.execute(text(
                "SELECT occurrence_count, material_occurrence_count "
                "FROM alert_dispatch_log "
                "WHERE alert_intent_id = :aid AND dispatch_status = 'would_dispatch'"
            ), {"aid": intent_2.alert_intent_id}).fetchone()

        assert new_wd_row is not None, "Expected would_dispatch row for new intent"
        assert new_wd_row[1] == 1, (
            f"Expected material_occurrence_count=1 on new intent's would_dispatch, "
            f"got {new_wd_row[1]}"
        )
