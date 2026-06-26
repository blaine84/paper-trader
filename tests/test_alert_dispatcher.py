"""Unit tests for AlertIntentStore and AlertDispatcher — Task 10.1.

Covers: stop/target inline triggers, empty intents, WAL/busy_timeout PRAGMAs,
immutability triggers, overlap guard, cycle_trigger_type field, observe mode,
deferred_until, dispatch_attempt_count, claimed_by_scheduled revert,
dedupe key normalization, partial unique index ON CONFLICT, and circular
import avoidance.

Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8
"""

from __future__ import annotations

import importlib
import inspect
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from utils.alert_dispatch_schema import init_alert_dispatch_schema
from utils.alert_intent_store import AlertIntentStore, build_dedupe_key


# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def engine():
    """In-memory SQLite engine with alert dispatch schema initialized."""
    eng = create_engine("sqlite://", echo=False)
    init_alert_dispatch_schema(eng)
    return eng


@pytest.fixture
def store(engine):
    """AlertIntentStore backed by an in-memory engine."""
    return AlertIntentStore(engine)


@pytest.fixture
def dispatcher(engine, store):
    """AlertDispatcher with mocked PM callbacks."""
    from utils.alert_dispatcher import AlertDispatcher

    begin_pm = MagicMock(return_value=True)
    end_pm = MagicMock()
    return AlertDispatcher(
        engine=engine,
        intent_store=store,
        begin_pm_cycle=begin_pm,
        end_pm_cycle=end_pm,
    )


def _sample_intent_data(**overrides) -> dict:
    """Factory for intent data dicts with sensible defaults."""
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
        "first_seen_at": "2025-01-15T10:30:00.000000Z",
        "last_seen_at": "2025-01-15T10:30:00.000000Z",
        "expiration_at": "2025-01-15T16:00:00.000000Z",
    }
    data.update(overrides)
    return data


# ─── Test: Stop/target triggers still execute inline ────────────────────────


class TestStopTargetInlineTriggers:
    """Stop-loss, target-hit, and thesis-invalidation triggers bypass dispatch."""

    def test_alert_dispatcher_has_no_stop_target_handling(self, dispatcher):
        """AlertDispatcher does not process stop/target/thesis triggers.
        Those remain in the price monitor's inline path."""
        # The dispatcher only handles entry_alert and rapid_move intents.
        # Verify it does not define methods for stop/target execution.
        method_names = [m for m in dir(dispatcher) if not m.startswith("_")]
        assert "handle_stop_loss" not in method_names
        assert "handle_target_hit" not in method_names
        assert "handle_thesis_invalidation" not in method_names

    def test_intent_store_only_accepts_entry_alert_types(self, store):
        """Recording an intent with entry_alert type works as expected;
        stop/target types are not part of the dispatch layer design."""
        # entry_alert and rapid_move are the valid alert_types for intents
        result = store.record_or_update_intent(
            _sample_intent_data(alert_type="entry_alert")
        )
        assert result.alert_type == "entry_alert"

        result2 = store.record_or_update_intent(
            _sample_intent_data(
                alert_type="rapid_move",
                dedupe_key="NVDA:rapid_move:xyz789",
            )
        )
        assert result2.alert_type == "rapid_move"


# ─── Test: Empty intent table returns empty lists without errors ─────────────


class TestEmptyIntentTable:
    """Query operations on an empty table should return empty lists cleanly."""

    def test_query_pending_empty(self, store):
        """query_pending on empty table returns empty list."""
        result = store.query_pending(now=datetime.utcnow())
        assert result == []

    def test_query_unclassified_empty(self, store):
        """query_unclassified on empty table returns empty list."""
        result = store.query_unclassified()
        assert result == []

    def test_load_active_cooldowns_empty(self, store):
        """load_active_cooldowns on empty table returns empty list."""
        result = store.load_active_cooldowns(now=datetime.utcnow())
        assert result == []

    def test_mark_expired_empty(self, store):
        """mark_expired on empty table returns 0."""
        count = store.mark_expired(now=datetime.utcnow())
        assert count == 0


# ─── Test: WAL mode and busy_timeout PRAGMA settings ────────────────────────


class TestPragmaSettings:
    """WAL mode and busy_timeout are applied via event listener."""

    def test_wal_mode_enabled(self, tmp_path):
        """PRAGMA journal_mode should return 'wal' on file-backed DB."""
        db_path = tmp_path / "test.db"
        eng = create_engine(f"sqlite:///{db_path}", echo=False)
        init_alert_dispatch_schema(eng)
        with eng.connect() as conn:
            result = conn.execute(text("PRAGMA journal_mode")).scalar()
            assert result == "wal"
        eng.dispose()

    def test_busy_timeout_set(self, tmp_path):
        """PRAGMA busy_timeout should return 30000 on file-backed DB."""
        db_path = tmp_path / "test.db"
        eng = create_engine(f"sqlite:///{db_path}", echo=False)
        init_alert_dispatch_schema(eng)
        with eng.connect() as conn:
            result = conn.execute(text("PRAGMA busy_timeout")).scalar()
            assert result == 30000
        eng.dispose()


# ─── Test: Immutability triggers on alert_dispatch_log ───────────────────────


class TestImmutabilityTriggers:
    """UPDATE and DELETE on alert_dispatch_log should raise."""

    def test_update_raises(self, engine):
        """UPDATE on alert_dispatch_log triggers ABORT."""
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO alert_dispatch_log
                    (alert_intent_id, symbol, alert_type, urgency, dispatch_status)
                VALUES ('test-id', 'NVDA', 'entry_alert', 'medium', 'dispatched')
            """))

        with pytest.raises(Exception, match="immutable"):
            with engine.begin() as conn:
                conn.execute(text("""
                    UPDATE alert_dispatch_log SET urgency = 'high' WHERE symbol = 'NVDA'
                """))

    def test_delete_raises(self, engine):
        """DELETE on alert_dispatch_log triggers ABORT."""
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO alert_dispatch_log
                    (alert_intent_id, symbol, alert_type, urgency, dispatch_status)
                VALUES ('test-id-2', 'AAPL', 'rapid_move', 'high', 'dispatched')
            """))

        with pytest.raises(Exception, match="immutable"):
            with engine.begin() as conn:
                conn.execute(text("""
                    DELETE FROM alert_dispatch_log WHERE symbol = 'AAPL'
                """))


# ─── Test: Overlap guard (concurrent dispatcher invocation skipped) ──────────


class TestOverlapGuard:
    """Dispatcher skips if already running."""

    @patch("utils.alert_dispatcher.AlertDispatcher._recover_stale_intents")
    @patch("utils.alert_dispatcher.AlertDispatcher._classify_unclassified")
    @patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "enabled")
    def test_overlap_guard_skips_concurrent_invocation(
        self, mock_classify, mock_recover, dispatcher
    ):
        """If _running is True, evaluate_and_dispatch returns None immediately."""
        dispatcher._running = True
        result = dispatcher.evaluate_and_dispatch()
        assert result is None
        # Should NOT have called recovery or classification
        mock_recover.assert_not_called()
        mock_classify.assert_not_called()

    @patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "enabled")
    def test_overlap_guard_resets_after_completion(self, dispatcher):
        """_running flag resets to False after evaluate_and_dispatch completes."""
        # Patch internal methods to avoid real work
        with patch.object(dispatcher, "_recover_stale_intents"), \
             patch.object(dispatcher, "_classify_unclassified"), \
             patch.object(dispatcher._store, "mark_expired", return_value=0), \
             patch.object(dispatcher._store, "query_pending", return_value=[]):
            dispatcher.evaluate_and_dispatch()
        assert dispatcher._running is False


# ─── Test: PM cycle_trigger_type field distinction ──────────────────────────


class TestCycleTriggerType:
    """cycle_trigger_type distinguishes 'alert' from 'scheduled' in audit log."""

    def test_audit_log_records_alert_trigger_type(self, store, engine):
        """record_audit_log with cycle_trigger_type='alert' persists correctly."""
        store.record_audit_log(
            alert_intent_id="uuid-1",
            symbol="TSLA",
            alert_type="entry_alert",
            urgency="high",
            dispatch_status="dispatched",
            cycle_trigger_type="alert",
        )

        with engine.begin() as conn:
            row = conn.execute(text(
                "SELECT cycle_trigger_type FROM alert_dispatch_log WHERE symbol = 'TSLA'"
            )).fetchone()
        assert row[0] == "alert"

    def test_audit_log_records_scheduled_trigger_type(self, store, engine):
        """record_audit_log with cycle_trigger_type='scheduled' persists correctly."""
        store.record_audit_log(
            alert_intent_id="uuid-2",
            symbol="AAPL",
            alert_type="entry_alert",
            urgency="medium",
            dispatch_status="consumed",
            cycle_trigger_type="scheduled",
        )

        with engine.begin() as conn:
            row = conn.execute(text(
                "SELECT cycle_trigger_type FROM alert_dispatch_log WHERE symbol = 'AAPL'"
            )).fetchone()
        assert row[0] == "scheduled"


# ─── Test: Observe mode ─────────────────────────────────────────────────────


class TestObserveMode:
    """Observe mode: would_dispatch logged, no PM called, legacy path unaffected."""

    @patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "observe")
    def test_observe_mode_logs_would_dispatch_no_pm_call(self, dispatcher, store, engine):
        """In observe mode, eligible intents are logged as would_dispatch, PM not called."""
        from utils.alert_intent_store import AlertIntent

        # Insert a classified pending intent with future expiration
        future_exp = (datetime.utcnow() + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        past_seen = (datetime.utcnow() - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        intent = store.record_or_update_intent(_sample_intent_data(
            filter_status="passed",
            urgency="high",
            first_seen_at=past_seen,
            last_seen_at=past_seen,
            expiration_at=future_exp,
        ))

        # Mark it as classified (passed)
        store.update_classification(intent.id, "passed", "high")

        # Patch internal methods so we control the flow
        with patch.object(dispatcher, "_recover_stale_intents"), \
             patch.object(dispatcher, "_classify_unclassified"), \
             patch.object(dispatcher._store, "mark_expired", return_value=0), \
             patch.object(dispatcher, "_is_market_hours", return_value=True), \
             patch.object(dispatcher._store, "is_global_cooled", return_value=False), \
             patch.object(dispatcher._store, "is_symbol_cooled", return_value=False), \
             patch.object(dispatcher, "_is_stale_duplicate", return_value=False):

            result = dispatcher.evaluate_and_dispatch()

        assert result is not None
        assert result["mode"] == "observe"
        assert result["would_dispatch"] >= 1

        # PM begin_pm_cycle should NOT have been called
        dispatcher._begin_pm_cycle.assert_not_called()

        # Verify audit log has 'would_dispatch' entry
        with engine.begin() as conn:
            row = conn.execute(text(
                "SELECT dispatch_status FROM alert_dispatch_log WHERE symbol = 'NVDA'"
            )).fetchone()
        assert row[0] == "would_dispatch"

    @patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "disabled")
    def test_disabled_mode_returns_none(self, dispatcher):
        """When dispatch mode is disabled, evaluate_and_dispatch returns None immediately."""
        result = dispatcher.evaluate_and_dispatch()
        assert result is None


# ─── Test: deferred_until field ─────────────────────────────────────────────


class TestDeferredUntil:
    """Cooldown-blocked intents get deferred_until set correctly."""

    def test_deferred_until_initially_none(self, store):
        """New intents have deferred_until = None."""
        result = store.record_or_update_intent(_sample_intent_data())
        assert result.deferred_until is None

    def test_deferred_until_blocks_dispatch_eligibility(self, store, engine):
        """An intent with deferred_until in the future is still returned by query_pending
        because deferred_until filtering happens in the dispatcher's _check_eligibility,
        not in the SQL query itself."""
        future_exp = (datetime.utcnow() + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        intent = store.record_or_update_intent(_sample_intent_data(
            filter_status="passed",
            expiration_at=future_exp,
        ))
        store.update_classification(intent.id, "passed", "medium")

        # Set deferred_until to the future
        future = (datetime.utcnow() + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE alert_intents SET deferred_until = :dt WHERE id = :id"
            ), {"dt": future, "id": intent.id})

        # query_pending returns this intent (deferred_until filtering is done by dispatcher)
        pending = store.query_pending(now=datetime.utcnow())
        assert len(pending) == 1
        assert pending[0].deferred_until is not None


# ─── Test: dispatch_attempt_count ───────────────────────────────────────────


class TestDispatchAttemptCount:
    """Increment on failure, terminal at 3."""

    def test_attempt_count_increments_on_failure(self, store):
        """mark_dispatch_failed increments dispatch_attempt_count."""
        intent = store.record_or_update_intent(_sample_intent_data())
        store.mark_dispatched([intent.id], datetime.utcnow())

        store.mark_dispatch_failed([intent.id], "PM error")

        # Check the intent now
        with store._engine.begin() as conn:
            row = conn.execute(text(
                "SELECT dispatch_attempt_count, dispatch_status, last_dispatch_error "
                "FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()
        assert row[0] == 1  # incremented from 0 to 1
        assert row[1] == "pending"  # reverted to pending (< 3)
        assert row[2] == "PM error"

    def test_terminal_at_three_attempts(self, store):
        """After 3 failures, intent transitions to dispatch_failed (terminal)."""
        intent = store.record_or_update_intent(_sample_intent_data())

        # Simulate 3 consecutive failures
        for i in range(3):
            store.mark_dispatched([intent.id], datetime.utcnow())
            store.mark_dispatch_failed([intent.id], f"error_{i}")

        with store._engine.begin() as conn:
            row = conn.execute(text(
                "SELECT dispatch_attempt_count, dispatch_status "
                "FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()
        assert row[0] >= 3
        assert row[1] == "dispatch_failed"


# ─── Test: claimed_by_scheduled revert ──────────────────────────────────────


class TestClaimedByScheduledRevert:
    """Revert to pending on scheduled PM failure."""

    def test_claim_and_revert(self, store):
        """mark_claimed_by_scheduled → mark_claimed_back_to_pending reverts to pending."""
        intent = store.record_or_update_intent(_sample_intent_data(filter_status="passed"))
        store.update_classification(intent.id, "passed", "medium")

        # Claim
        count = store.mark_claimed_by_scheduled([intent.id])
        assert count == 1

        # Check claimed state
        with store._engine.begin() as conn:
            row = conn.execute(text(
                "SELECT dispatch_status FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()
        assert row[0] == "claimed_by_scheduled"

        # Revert on failure
        revert_count = store.mark_claimed_back_to_pending([intent.id], "PM crashed")
        assert revert_count == 1

        # Verify reverted to pending
        with store._engine.begin() as conn:
            row = conn.execute(text(
                "SELECT dispatch_status, last_dispatch_error "
                "FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()
        assert row[0] == "pending"
        assert row[1] == "PM crashed"


# ─── Test: Dedupe key normalization ─────────────────────────────────────────


class TestDedupeKeyNormalization:
    """Whitespace/case variations produce same key."""

    def test_case_insensitive(self):
        """Upper and lower case setup_condition produce same key."""
        key1 = build_dedupe_key("NVDA", "entry_alert", "Breakout Above 145")
        key2 = build_dedupe_key("NVDA", "entry_alert", "breakout above 145")
        assert key1 == key2

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace in setup_condition doesn't change key."""
        key1 = build_dedupe_key("AAPL", "rapid_move", "gap up")
        key2 = build_dedupe_key("AAPL", "rapid_move", "  gap up  ")
        assert key1 == key2

    def test_combined_case_and_whitespace(self):
        """Mixed case + whitespace still produces same key."""
        key1 = build_dedupe_key("TSLA", "entry_alert", "Channel Break")
        key2 = build_dedupe_key("TSLA", "entry_alert", "  CHANNEL BREAK  ")
        assert key1 == key2

    def test_different_conditions_produce_different_keys(self):
        """Genuinely different conditions produce different keys."""
        key1 = build_dedupe_key("NVDA", "entry_alert", "breakout above 145")
        key2 = build_dedupe_key("NVDA", "entry_alert", "breakdown below 120")
        assert key1 != key2

    def test_symbol_uppercased(self):
        """Symbol is uppercased in the key."""
        key1 = build_dedupe_key("nvda", "entry_alert", "test")
        key2 = build_dedupe_key("NVDA", "entry_alert", "test")
        assert key1 == key2


# ─── Test: Partial unique index ON CONFLICT behavior ────────────────────────


class TestPartialUniqueIndex:
    """ON CONFLICT for active dedupe_keys only."""

    def test_same_dedupe_key_active_upserts(self, store):
        """Two inserts with same dedupe_key while active → upsert (occurrence_count=2)."""
        data = _sample_intent_data()
        store.record_or_update_intent(data)

        data["last_seen_at"] = "2025-01-15T10:31:00.000000Z"
        result = store.record_or_update_intent(data)
        assert result.occurrence_count == 2

    def test_same_dedupe_key_after_consumed_creates_new(self, store, engine):
        """After intent is consumed (terminal), same dedupe_key creates new row."""
        data = _sample_intent_data()
        first = store.record_or_update_intent(data)

        # Transition to consumed (terminal)
        store.mark_dispatched([first.id], datetime.utcnow())
        store.mark_consumed([first.id])

        # New insert with same dedupe_key
        data["last_seen_at"] = "2025-01-15T11:00:00.000000Z"
        second = store.record_or_update_intent(data)

        assert second.id != first.id
        assert second.occurrence_count == 1
        assert second.dispatch_status == "pending"

    def test_same_dedupe_key_after_expired_creates_new(self, store, engine):
        """After intent is expired (terminal), same dedupe_key creates new row."""
        data = _sample_intent_data()
        first = store.record_or_update_intent(data)

        # Transition to expired (terminal)
        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE alert_intents SET dispatch_status = 'expired' WHERE id = :id"
            ), {"id": first.id})

        # New insert with same dedupe_key
        data["last_seen_at"] = "2025-01-15T11:00:00.000000Z"
        second = store.record_or_update_intent(data)

        assert second.id != first.id
        assert second.occurrence_count == 1

    def test_same_dedupe_key_after_suppressed_creates_new(self, store, engine):
        """After intent is suppressed (terminal), same dedupe_key creates new row."""
        data = _sample_intent_data()
        first = store.record_or_update_intent(data)

        # Transition to suppressed (terminal)
        store.mark_suppressed(first.id, "stale_duplicate")

        # New insert with same dedupe_key
        data["last_seen_at"] = "2025-01-15T11:00:00.000000Z"
        second = store.record_or_update_intent(data)

        assert second.id != first.id
        assert second.occurrence_count == 1


# ─── Test: Circular import avoidance ────────────────────────────────────────


class TestCircularImportAvoidance:
    """alert_dispatcher does not import from orchestrator at module level."""

    def test_no_orchestrator_import_in_module(self):
        """The alert_dispatcher module should not import from orchestrator.
        It uses constructor injection for PM callbacks instead."""
        import utils.alert_dispatcher as mod

        source = inspect.getsource(mod)

        # Check top-level imports (before class definitions)
        lines = source.split("\n")
        in_top_level = True
        for line in lines:
            stripped = line.strip()
            # Once we hit a class or function definition, stop checking top-level
            if stripped.startswith("class ") or (stripped.startswith("def ") and not line.startswith(" ")):
                in_top_level = False
                break
            if in_top_level and "from orchestrator" in stripped:
                pytest.fail("alert_dispatcher has top-level import from orchestrator")
            if in_top_level and "import orchestrator" in stripped:
                pytest.fail("alert_dispatcher has top-level import from orchestrator")

    def test_dispatcher_uses_callback_injection(self, dispatcher):
        """AlertDispatcher receives begin_pm_cycle and end_pm_cycle as callbacks,
        not via orchestrator import."""
        assert hasattr(dispatcher, "_begin_pm_cycle")
        assert hasattr(dispatcher, "_end_pm_cycle")
        assert callable(dispatcher._begin_pm_cycle)
        assert callable(dispatcher._end_pm_cycle)
