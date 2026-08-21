"""Tests for FastPathMonitor overrun and concurrency protection.

Validates:
- Full tick with one active trigger that fires produces an event row
- Full tick with no active triggers completes without error
- Stale triggers are expired before evaluation
- Quote batching fetches each symbol only once per tick
- Tick lock: concurrent tick attempt is skipped (not queued)
- Per-trigger watchdog: slow trigger produces stand_down("evaluation_timeout")
- Max outcomes cap: only trade_executed and pending_order_created count against cap
- Deferred queue: deferred triggers processed first on next tick
- Deferred items re-queued when still over cap
- Overrun protection logs warning and returns partial results

Requirements: 1.8, 2.8, 9.1, cross-cutting acceptance test 8
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from db.schema import init_fast_path_events_schema, init_fast_path_triggers_schema
from utils.fast_path_evaluator import FastPathOutcome
from utils.fast_path_monitor import (
    FastPathMonitor,
    _EXECUTION_PATH_OUTCOMES,
    _TRIGGER_EVALUATION_TIMEOUT_SECONDS,
)


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    init_fast_path_triggers_schema(eng)
    init_fast_path_events_schema(eng)
    return eng


@dataclass
class FakeTrigger:
    """Minimal trigger for testing monitor mechanisms."""

    trigger_id: str = "trig-001"
    symbol: str = "TSLA"
    profile_id: str = "moderate"
    direction: str = "SHORT"
    setup_type: str = "momentum_fade"
    trigger_type: str = "entry_zone"
    trigger_level: float = 351.0
    entry_price: float = 351.61
    stop_price: float = 355.00
    target_price: float = 348.97
    source_signal_id: str | None = "signal-abc"
    source_watch_id: str | None = None


def _make_outcome(trigger_id: str = "trig-001", outcome_type: str = "trade_executed", **overrides) -> FastPathOutcome:
    defaults = {
        "outcome_type": outcome_type,
        "outcome_reason_code": "gates_passed",
        "trigger_id": trigger_id,
        "symbol": "TSLA",
        "profile_id": "moderate",
        "direction": "SHORT",
        "setup_type": "momentum_fade",
        "current_price": 350.50,
        "entry_price": 351.61,
        "stop_price": 355.00,
        "target_price": 348.97,
        "metadata": None,
    }
    defaults.update(overrides)
    return FastPathOutcome(**defaults)


# ---------------------------------------------------------------------------
# Tick Lock Tests
# ---------------------------------------------------------------------------


class TestTickLock:
    """Tick lock: non-blocking acquire, skip on contention."""

    def test_concurrent_tick_skipped_with_skipped_count(self, engine):
        """If tick lock is held, run_tick returns immediately with skipped=1."""
        monitor = FastPathMonitor(engine, ["moderate"])

        # Manually acquire the lock to simulate an in-progress tick
        monitor._tick_lock.acquire()
        try:
            result = monitor.run_tick()
            assert result["skipped"] == 1
            assert result["evaluated"] == 0
            assert result["fired"] == 0
        finally:
            monitor._tick_lock.release()

    def test_tick_lock_logs_warning_on_skip(self, engine, caplog):
        """Skipped tick logs the expected warning message."""
        monitor = FastPathMonitor(engine, ["moderate"])

        monitor._tick_lock.acquire()
        try:
            import logging
            with caplog.at_level(logging.WARNING, logger="utils.fast_path_monitor"):
                monitor.run_tick()
            assert "tick still running, skipping" in caplog.text
        finally:
            monitor._tick_lock.release()

    def test_tick_lock_released_after_normal_tick(self, engine):
        """Lock is released after a successful tick completes."""
        monitor = FastPathMonitor(engine, ["moderate"])

        # Mock registry to return no triggers (simple tick)
        monitor._registries["moderate"] = MagicMock()
        monitor._registries["moderate"].expire_stale_triggers.return_value = 0
        monitor._registries["moderate"].get_active_triggers.return_value = []

        monitor.run_tick()

        # Lock should be released — acquiring should succeed
        acquired = monitor._tick_lock.acquire(blocking=False)
        assert acquired is True
        monitor._tick_lock.release()

    def test_tick_lock_released_after_exception(self, engine):
        """Lock is released even when _execute_tick raises."""
        monitor = FastPathMonitor(engine, ["moderate"])

        # Force an exception inside _execute_tick
        with patch.object(monitor, "_execute_tick", side_effect=RuntimeError("boom")):
            result = monitor.run_tick()

        # Should have caught the error
        assert "error" in result

        # Lock should still be released
        acquired = monitor._tick_lock.acquire(blocking=False)
        assert acquired is True
        monitor._tick_lock.release()

    def test_second_tick_runs_after_first_completes(self, engine):
        """After first tick finishes, second tick runs normally (not skipped)."""
        monitor = FastPathMonitor(engine, ["moderate"])

        # Mock registry for simple tick
        monitor._registries["moderate"] = MagicMock()
        monitor._registries["moderate"].expire_stale_triggers.return_value = 0
        monitor._registries["moderate"].get_active_triggers.return_value = []

        result1 = monitor.run_tick()
        result2 = monitor.run_tick()

        assert result1["skipped"] == 0
        assert result2["skipped"] == 0


# ---------------------------------------------------------------------------
# Per-Trigger Watchdog Tests
# ---------------------------------------------------------------------------


class TestPerTriggerWatchdog:
    """Watchdog: evaluation exceeding 3s produces stand_down(evaluation_timeout)."""

    def test_slow_evaluation_produces_stand_down_timeout(self, engine):
        """Trigger evaluation taking > 3s returns stand_down('evaluation_timeout')."""
        monitor = FastPathMonitor(engine, ["moderate"])
        trigger = FakeTrigger()
        quote = {"price": 350.0, "age_ms": 100, "reliable": True}

        # Mock evaluate_trigger to simulate slow execution (sleeps 3.1s)
        def slow_evaluate(trig, q, profile_state):
            time.sleep(3.1)
            return _make_outcome()

        with patch("utils.fast_path_monitor.evaluate_trigger", side_effect=slow_evaluate):
            outcome = monitor._evaluate_with_timeout(trigger, quote, "moderate")

        assert outcome is not None
        assert outcome.outcome_type == "stand_down"
        assert outcome.outcome_reason_code == "evaluation_timeout"
        assert outcome.trigger_id == "trig-001"
        assert outcome.symbol == "TSLA"

    def test_fast_evaluation_returns_original_outcome(self, engine):
        """Trigger evaluation under 3s returns the actual evaluated outcome."""
        monitor = FastPathMonitor(engine, ["moderate"])
        trigger = FakeTrigger()
        quote = {"price": 350.0, "age_ms": 100, "reliable": True}

        expected_outcome = _make_outcome(outcome_type="missed_move", outcome_reason_code="target_already_crossed")

        with patch("utils.fast_path_monitor.evaluate_trigger", return_value=expected_outcome):
            outcome = monitor._evaluate_with_timeout(trigger, quote, "moderate")

        assert outcome is expected_outcome

    def test_fast_evaluation_returning_none_passes_through(self, engine):
        """When trigger condition not met (None), watchdog returns None."""
        monitor = FastPathMonitor(engine, ["moderate"])
        trigger = FakeTrigger()
        quote = {"price": 350.0, "age_ms": 100, "reliable": True}

        with patch("utils.fast_path_monitor.evaluate_trigger", return_value=None):
            outcome = monitor._evaluate_with_timeout(trigger, quote, "moderate")

        assert outcome is None

    def test_watchdog_logs_warning_on_timeout(self, engine, caplog):
        """Timeout produces a warning log message."""
        monitor = FastPathMonitor(engine, ["moderate"])
        trigger = FakeTrigger()
        quote = {"price": 350.0, "age_ms": 100, "reliable": True}

        def slow_evaluate(trig, q, profile_state):
            time.sleep(3.1)
            return _make_outcome()

        import logging
        with caplog.at_level(logging.WARNING, logger="utils.fast_path_monitor"):
            with patch("utils.fast_path_monitor.evaluate_trigger", side_effect=slow_evaluate):
                monitor._evaluate_with_timeout(trigger, quote, "moderate")

        assert "exceeds" in caplog.text
        assert "watchdog" in caplog.text

    def test_watchdog_timeout_includes_duration_metadata(self, engine):
        """stand_down(evaluation_timeout) includes evaluation_duration_s in metadata."""
        monitor = FastPathMonitor(engine, ["moderate"])
        trigger = FakeTrigger()
        quote = {"price": 350.0, "age_ms": 100, "reliable": True}

        def slow_evaluate(trig, q, profile_state):
            time.sleep(3.1)
            return _make_outcome()

        with patch("utils.fast_path_monitor.evaluate_trigger", side_effect=slow_evaluate):
            outcome = monitor._evaluate_with_timeout(trigger, quote, "moderate")

        assert outcome.metadata is not None
        assert "evaluation_duration_s" in outcome.metadata
        assert outcome.metadata["evaluation_duration_s"] >= 3.0


# ---------------------------------------------------------------------------
# Max Outcomes Cap Tests
# ---------------------------------------------------------------------------


class TestMaxOutcomesCap:
    """Max outcomes cap: only trade_executed and pending_order_created count."""

    def test_execution_path_outcomes_frozenset_correct(self):
        """_EXECUTION_PATH_OUTCOMES contains exactly trade_executed and pending_order_created."""
        assert _EXECUTION_PATH_OUTCOMES == frozenset({"trade_executed", "pending_order_created"})

    def test_cap_limits_delegated_executions(self, engine):
        """When cap is reached, additional execution-path outcomes are deferred."""
        monitor = FastPathMonitor(engine, ["moderate"])

        # Create 6 triggers that all fire as trade_executed
        triggers = [
            FakeTrigger(trigger_id=f"trig-{i:03d}", symbol=f"SYM{i}")
            for i in range(6)
        ]

        mock_registry = MagicMock()
        mock_registry.expire_stale_triggers.return_value = 0
        mock_registry.get_active_triggers.return_value = triggers
        monitor._registries["moderate"] = mock_registry

        # All triggers produce trade_executed outcomes
        def make_trade_outcome(trig, q, profile_state):
            return _make_outcome(
                trigger_id=trig.trigger_id,
                outcome_type="trade_executed",
            )

        delegate_calls = []

        def track_delegate(outcome, trigger, eng):
            delegate_calls.append(outcome.trigger_id)

        with patch("utils.fast_path_monitor.evaluate_trigger", side_effect=make_trade_outcome), \
             patch("utils.fast_path_monitor._fetch_quotes", return_value={
                 f"SYM{i}": {"price": 350.0, "age_ms": 0, "reliable": True}
                 for i in range(6)
             }), \
             patch("utils.fast_path_monitor._persist_event", return_value="evt-001"), \
             patch("utils.fast_path_monitor._delegate_execution", side_effect=track_delegate), \
             patch("utils.fast_path_monitor.FAST_PATH_MODE", "enabled"), \
             patch("utils.fast_path_monitor.FAST_PATH_MAX_OUTCOMES_PER_TICK", 5):
            result = monitor.run_tick()

        # 5 delegated, 1 deferred
        assert len(delegate_calls) == 5
        assert result["deferred"] == 1
        assert result["fired"] == 6

    def test_non_execution_outcomes_dont_count_against_cap(self, engine):
        """stand_down, missed_move, watch_created do not count against the cap."""
        monitor = FastPathMonitor(engine, ["moderate"])

        # Create 10 triggers: 8 produce stand_down, 2 produce trade_executed
        triggers = [
            FakeTrigger(trigger_id=f"trig-{i:03d}", symbol=f"SYM{i}")
            for i in range(10)
        ]

        mock_registry = MagicMock()
        mock_registry.expire_stale_triggers.return_value = 0
        mock_registry.get_active_triggers.return_value = triggers
        monitor._registries["moderate"] = mock_registry

        call_count = [0]

        def mixed_outcomes(trig, q, profile_state):
            idx = call_count[0]
            call_count[0] += 1
            if idx < 8:
                return _make_outcome(
                    trigger_id=trig.trigger_id,
                    outcome_type="stand_down",
                    outcome_reason_code="invalid_geometry",
                )
            return _make_outcome(
                trigger_id=trig.trigger_id,
                outcome_type="trade_executed",
            )

        delegate_calls = []

        def track_delegate(outcome, trigger, eng):
            delegate_calls.append(outcome.trigger_id)

        with patch("utils.fast_path_monitor.evaluate_trigger", side_effect=mixed_outcomes), \
             patch("utils.fast_path_monitor._fetch_quotes", return_value={
                 f"SYM{i}": {"price": 350.0, "age_ms": 0, "reliable": True}
                 for i in range(10)
             }), \
             patch("utils.fast_path_monitor._persist_event", return_value="evt-001"), \
             patch("utils.fast_path_monitor._delegate_execution", side_effect=track_delegate), \
             patch("utils.fast_path_monitor.FAST_PATH_MODE", "enabled"), \
             patch("utils.fast_path_monitor.FAST_PATH_MAX_OUTCOMES_PER_TICK", 5):
            result = monitor.run_tick()

        # Both trade_executed got delegated (only 2, well under cap of 5)
        assert len(delegate_calls) == 2
        assert result["deferred"] == 0
        assert result["fired"] == 10

    def test_pending_order_created_counts_against_cap(self, engine):
        """pending_order_created outcomes count against execution cap."""
        monitor = FastPathMonitor(engine, ["moderate"])

        # 6 triggers all producing pending_order_created
        triggers = [
            FakeTrigger(trigger_id=f"trig-{i:03d}", symbol=f"SYM{i}")
            for i in range(6)
        ]

        mock_registry = MagicMock()
        mock_registry.expire_stale_triggers.return_value = 0
        mock_registry.get_active_triggers.return_value = triggers
        monitor._registries["moderate"] = mock_registry

        def make_pending(trig, q, profile_state):
            return _make_outcome(
                trigger_id=trig.trigger_id,
                outcome_type="pending_order_created",
                outcome_reason_code="price_away_limit_valid",
            )

        delegate_calls = []

        def track_delegate(outcome, trigger, eng):
            delegate_calls.append(outcome.trigger_id)

        with patch("utils.fast_path_monitor.evaluate_trigger", side_effect=make_pending), \
             patch("utils.fast_path_monitor._fetch_quotes", return_value={
                 f"SYM{i}": {"price": 350.0, "age_ms": 0, "reliable": True}
                 for i in range(6)
             }), \
             patch("utils.fast_path_monitor._persist_event", return_value="evt-001"), \
             patch("utils.fast_path_monitor._delegate_execution", side_effect=track_delegate), \
             patch("utils.fast_path_monitor.FAST_PATH_MODE", "enabled"), \
             patch("utils.fast_path_monitor.FAST_PATH_MAX_OUTCOMES_PER_TICK", 5):
            result = monitor.run_tick()

        assert len(delegate_calls) == 5
        assert result["deferred"] == 1


# ---------------------------------------------------------------------------
# Deferred Queue Tests
# ---------------------------------------------------------------------------


class TestDeferredQueue:
    """Deferred queue: items processed first on next tick, re-queue if still over cap."""

    def test_deferred_items_processed_first_on_next_tick(self, engine):
        """Deferred outcomes from a prior tick are delegated before new triggers evaluate."""
        monitor = FastPathMonitor(engine, ["moderate"])

        # Pre-populate deferred queue with a prior outcome
        deferred_outcome = _make_outcome(trigger_id="deferred-001", outcome_type="trade_executed")
        deferred_trigger = FakeTrigger(trigger_id="deferred-001", symbol="AAPL")
        monitor._deferred_queue.append((deferred_outcome, deferred_trigger))

        # Registry returns one new trigger that also fires
        new_trigger = FakeTrigger(trigger_id="new-001", symbol="MSFT")
        mock_registry = MagicMock()
        mock_registry.expire_stale_triggers.return_value = 0
        mock_registry.get_active_triggers.return_value = [new_trigger]
        monitor._registries["moderate"] = mock_registry

        def make_trade(trig, q, profile_state):
            return _make_outcome(trigger_id=trig.trigger_id, outcome_type="trade_executed")

        delegate_order = []

        def track_delegate(outcome, trigger, eng):
            delegate_order.append(outcome.trigger_id)

        with patch("utils.fast_path_monitor.evaluate_trigger", side_effect=make_trade), \
             patch("utils.fast_path_monitor._fetch_quotes", return_value={
                 "MSFT": {"price": 400.0, "age_ms": 0, "reliable": True},
             }), \
             patch("utils.fast_path_monitor._persist_event", return_value="evt-001"), \
             patch("utils.fast_path_monitor._delegate_execution", side_effect=track_delegate), \
             patch("utils.fast_path_monitor.FAST_PATH_MODE", "enabled"), \
             patch("utils.fast_path_monitor.FAST_PATH_MAX_OUTCOMES_PER_TICK", 10):
            result = monitor.run_tick()

        # Deferred item processed first, then new trigger
        assert delegate_order[0] == "deferred-001"
        assert delegate_order[1] == "new-001"
        assert result["deferred"] == 0

    def test_deferred_items_requeued_when_cap_reached_from_deferred(self, engine):
        """If deferred items alone exceed the cap, excess are re-queued."""
        monitor = FastPathMonitor(engine, ["moderate"])

        # Pre-populate deferred queue with 3 items, but cap is 2
        for i in range(3):
            outcome = _make_outcome(trigger_id=f"deferred-{i:03d}", outcome_type="trade_executed")
            trigger = FakeTrigger(trigger_id=f"deferred-{i:03d}")
            monitor._deferred_queue.append((outcome, trigger))

        # No new triggers
        mock_registry = MagicMock()
        mock_registry.expire_stale_triggers.return_value = 0
        mock_registry.get_active_triggers.return_value = []
        monitor._registries["moderate"] = mock_registry

        delegate_calls = []

        def track_delegate(outcome, trigger, eng):
            delegate_calls.append(outcome.trigger_id)

        with patch("utils.fast_path_monitor._delegate_execution", side_effect=track_delegate), \
             patch("utils.fast_path_monitor.FAST_PATH_MODE", "enabled"), \
             patch("utils.fast_path_monitor.FAST_PATH_MAX_OUTCOMES_PER_TICK", 2):
            result = monitor.run_tick()

        # Only 2 delegated, 1 re-queued
        assert len(delegate_calls) == 2
        assert result["deferred"] == 1
        assert len(monitor._deferred_queue) == 1
        assert monitor._deferred_queue[0][0].trigger_id == "deferred-002"

    def test_deferred_queue_cleared_before_processing(self, engine):
        """Deferred queue is cleared atomically — items moved to local list."""
        monitor = FastPathMonitor(engine, ["moderate"])

        outcome = _make_outcome(trigger_id="d-001", outcome_type="trade_executed")
        trigger = FakeTrigger(trigger_id="d-001")
        monitor._deferred_queue.append((outcome, trigger))

        mock_registry = MagicMock()
        mock_registry.expire_stale_triggers.return_value = 0
        mock_registry.get_active_triggers.return_value = []
        monitor._registries["moderate"] = mock_registry

        with patch("utils.fast_path_monitor._delegate_execution"), \
             patch("utils.fast_path_monitor.FAST_PATH_MODE", "enabled"), \
             patch("utils.fast_path_monitor.FAST_PATH_MAX_OUTCOMES_PER_TICK", 10):
            monitor.run_tick()

        # Queue should be empty after processing (not re-queued)
        assert len(monitor._deferred_queue) == 0

    def test_ticks_are_independent_no_cross_tick_queueing(self, engine):
        """Each tick is independent — deferred delegation is within-tick state."""
        monitor = FastPathMonitor(engine, ["moderate"])

        mock_registry = MagicMock()
        mock_registry.expire_stale_triggers.return_value = 0
        mock_registry.get_active_triggers.return_value = []
        monitor._registries["moderate"] = mock_registry

        with patch("utils.fast_path_monitor.FAST_PATH_MODE", "enabled"), \
             patch("utils.fast_path_monitor.FAST_PATH_MAX_OUTCOMES_PER_TICK", 5):
            # Two ticks with no triggers — both should complete independently
            r1 = monitor.run_tick()
            r2 = monitor.run_tick()

        assert r1["skipped"] == 0
        assert r2["skipped"] == 0
        assert r1["deferred"] == 0
        assert r2["deferred"] == 0

    def test_deferred_execution_failure_does_not_crash(self, engine):
        """If deferred execution raises, log and continue to next."""
        monitor = FastPathMonitor(engine, ["moderate"])

        # Two deferred items — first one will fail
        for i in range(2):
            outcome = _make_outcome(trigger_id=f"d-{i}", outcome_type="trade_executed")
            trigger = FakeTrigger(trigger_id=f"d-{i}")
            monitor._deferred_queue.append((outcome, trigger))

        mock_registry = MagicMock()
        mock_registry.expire_stale_triggers.return_value = 0
        mock_registry.get_active_triggers.return_value = []
        monitor._registries["moderate"] = mock_registry

        call_count = [0]

        def failing_delegate(outcome, trigger, eng):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("delegation failed")

        with patch("utils.fast_path_monitor._delegate_execution", side_effect=failing_delegate), \
             patch("utils.fast_path_monitor.FAST_PATH_MODE", "enabled"), \
             patch("utils.fast_path_monitor.FAST_PATH_MAX_OUTCOMES_PER_TICK", 10):
            # Should not raise
            result = monitor.run_tick()

        # Both attempted (first failed, second succeeded)
        assert call_count[0] == 2

    def test_observe_mode_does_not_delegate(self, engine):
        """In observe mode, execution-path outcomes are NOT delegated (no deferred either)."""
        monitor = FastPathMonitor(engine, ["moderate"])

        triggers = [FakeTrigger(trigger_id=f"trig-{i}", symbol=f"SYM{i}") for i in range(3)]

        mock_registry = MagicMock()
        mock_registry.expire_stale_triggers.return_value = 0
        mock_registry.get_active_triggers.return_value = triggers
        monitor._registries["moderate"] = mock_registry

        def make_trade(trig, q, profile_state):
            return _make_outcome(trigger_id=trig.trigger_id, outcome_type="trade_executed")

        delegate_calls = []

        def track_delegate(outcome, trigger, eng):
            delegate_calls.append(outcome.trigger_id)

        with patch("utils.fast_path_monitor.evaluate_trigger", side_effect=make_trade), \
             patch("utils.fast_path_monitor._fetch_quotes", return_value={
                 f"SYM{i}": {"price": 350.0, "age_ms": 0, "reliable": True}
                 for i in range(3)
             }), \
             patch("utils.fast_path_monitor._persist_event", return_value="evt-001"), \
             patch("utils.fast_path_monitor._delegate_execution", side_effect=track_delegate), \
             patch("utils.fast_path_monitor.FAST_PATH_MODE", "observe"), \
             patch("utils.fast_path_monitor.FAST_PATH_MAX_OUTCOMES_PER_TICK", 5):
            result = monitor.run_tick()

        # No delegation in observe mode
        assert len(delegate_calls) == 0
        assert result["deferred"] == 0
        assert result["fired"] == 3


# ---------------------------------------------------------------------------
# Full Tick Integration Tests
# ---------------------------------------------------------------------------


class TestFullTickWithFiredTrigger:
    """A full tick with one active trigger that fires produces an event row."""

    def test_fired_trigger_produces_event_row_in_db(self, engine):
        """Full tick: trigger fires → event persisted to fast_path_events table."""
        monitor = FastPathMonitor(engine, ["moderate"])

        trigger = FakeTrigger(trigger_id="trig-fire-001", symbol="TSLA")

        mock_registry = MagicMock()
        mock_registry.expire_stale_triggers.return_value = 0
        mock_registry.get_active_triggers.return_value = [trigger]
        monitor._registries["moderate"] = mock_registry

        fired_outcome = _make_outcome(
            trigger_id="trig-fire-001",
            outcome_type="trade_executed",
            outcome_reason_code="gates_passed",
        )

        with patch("utils.fast_path_monitor.evaluate_trigger", return_value=fired_outcome), \
             patch("utils.fast_path_monitor._fetch_quotes", return_value={
                 "TSLA": {"price": 350.0, "age_ms": 50, "reliable": True},
             }), \
             patch("utils.fast_path_monitor._delegate_execution"), \
             patch("utils.fast_path_monitor.FAST_PATH_MODE", "enabled"), \
             patch("utils.fast_path_monitor.FAST_PATH_MAX_OUTCOMES_PER_TICK", 5):
            result = monitor.run_tick()

        assert result["fired"] == 1
        assert result["evaluated"] == 1

        # Verify event row was persisted
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT * FROM fast_path_events WHERE trigger_id = :tid"),
                {"tid": "trig-fire-001"},
            ).fetchall()

        assert len(rows) == 1
        row = rows[0]._mapping
        assert row["symbol"] == "TSLA"
        assert row["outcome_type"] == "trade_executed"
        assert row["outcome_reason_code"] == "gates_passed"
        assert row["profile_id"] == "moderate"
        assert row["annotation_status"] == "annotation_pending"
        assert row["narration_source"] == "template"
        assert row["event_id"] is not None

    def test_fired_trigger_mark_fired_called_with_event_id(self, engine):
        """Full tick: after persistence, registry.mark_fired is called with event_id."""
        monitor = FastPathMonitor(engine, ["moderate"])

        trigger = FakeTrigger(trigger_id="trig-fire-002", symbol="AAPL")

        mock_registry = MagicMock()
        mock_registry.expire_stale_triggers.return_value = 0
        mock_registry.get_active_triggers.return_value = [trigger]
        monitor._registries["moderate"] = mock_registry

        fired_outcome = _make_outcome(
            trigger_id="trig-fire-002",
            outcome_type="missed_move",
            outcome_reason_code="target_already_crossed",
        )

        with patch("utils.fast_path_monitor.evaluate_trigger", return_value=fired_outcome), \
             patch("utils.fast_path_monitor._fetch_quotes", return_value={
                 "AAPL": {"price": 200.0, "age_ms": 0, "reliable": True},
             }), \
             patch("utils.fast_path_monitor.FAST_PATH_MODE", "observe"), \
             patch("utils.fast_path_monitor.FAST_PATH_MAX_OUTCOMES_PER_TICK", 5):
            monitor.run_tick()

        # mark_fired should have been called with the trigger_id and a UUID event_id
        mock_registry.mark_fired.assert_called_once()
        call_args = mock_registry.mark_fired.call_args
        assert call_args[0][0] == "trig-fire-002"
        # event_id should be a valid UUID string
        event_id = call_args[0][1]
        assert len(event_id) == 36  # UUID4 format: 8-4-4-4-12


class TestFullTickNoActiveTriggers:
    """A full tick with no active triggers completes without error."""

    def test_no_active_triggers_completes_cleanly(self, engine):
        """Tick with no active triggers returns a clean summary with zeros."""
        monitor = FastPathMonitor(engine, ["moderate"])

        mock_registry = MagicMock()
        mock_registry.expire_stale_triggers.return_value = 0
        mock_registry.get_active_triggers.return_value = []
        monitor._registries["moderate"] = mock_registry

        result = monitor.run_tick()

        assert result["evaluated"] == 0
        assert result["fired"] == 0
        assert result["expired"] == 0
        assert result["skipped"] == 0
        assert result["deferred"] == 0

    def test_no_active_triggers_does_not_fetch_quotes(self, engine):
        """When no triggers exist, quote fetch is not called."""
        monitor = FastPathMonitor(engine, ["moderate"])

        mock_registry = MagicMock()
        mock_registry.expire_stale_triggers.return_value = 0
        mock_registry.get_active_triggers.return_value = []
        monitor._registries["moderate"] = mock_registry

        with patch("utils.fast_path_monitor._fetch_quotes") as mock_fetch:
            monitor.run_tick()

        mock_fetch.assert_not_called()

    def test_multiple_profiles_no_triggers_all_complete(self, engine):
        """Multiple profiles with no triggers: all registries polled, clean result."""
        monitor = FastPathMonitor(engine, ["conservative", "moderate", "aggressive"])

        for pid in ["conservative", "moderate", "aggressive"]:
            mock_reg = MagicMock()
            mock_reg.expire_stale_triggers.return_value = 0
            mock_reg.get_active_triggers.return_value = []
            monitor._registries[pid] = mock_reg

        result = monitor.run_tick()

        assert result["evaluated"] == 0
        assert result["fired"] == 0
        for pid in ["conservative", "moderate", "aggressive"]:
            monitor._registries[pid].expire_stale_triggers.assert_called_once()
            monitor._registries[pid].get_active_triggers.assert_called_once()


# ---------------------------------------------------------------------------
# Stale Trigger Expiry Tests
# ---------------------------------------------------------------------------


class TestStaleTriggerExpiry:
    """Stale triggers are expired before evaluation."""

    def test_expire_stale_triggers_called_before_evaluation(self, engine):
        """expire_stale_triggers is called before get_active_triggers per profile."""
        monitor = FastPathMonitor(engine, ["moderate"])

        call_order = []

        mock_registry = MagicMock()
        mock_registry.expire_stale_triggers.side_effect = lambda: (
            call_order.append("expire") or 2
        )
        mock_registry.get_active_triggers.side_effect = lambda: (
            call_order.append("get_active") or []
        )
        monitor._registries["moderate"] = mock_registry

        monitor.run_tick()

        assert call_order == ["expire", "get_active"]

    def test_expired_count_reflected_in_summary(self, engine):
        """Expired triggers count appears in tick summary."""
        monitor = FastPathMonitor(engine, ["moderate"])

        mock_registry = MagicMock()
        mock_registry.expire_stale_triggers.return_value = 3
        mock_registry.get_active_triggers.return_value = []
        monitor._registries["moderate"] = mock_registry

        result = monitor.run_tick()

        assert result["expired"] == 3

    def test_expired_triggers_across_multiple_profiles(self, engine):
        """Expired count aggregated across profiles."""
        monitor = FastPathMonitor(engine, ["conservative", "moderate"])

        for pid, expired_count in [("conservative", 2), ("moderate", 4)]:
            mock_reg = MagicMock()
            mock_reg.expire_stale_triggers.return_value = expired_count
            mock_reg.get_active_triggers.return_value = []
            monitor._registries[pid] = mock_reg

        result = monitor.run_tick()

        assert result["expired"] == 6

    def test_expire_error_does_not_crash_tick(self, engine):
        """If expire_stale_triggers raises, tick continues with evaluation."""
        monitor = FastPathMonitor(engine, ["moderate"])

        mock_registry = MagicMock()
        mock_registry.expire_stale_triggers.side_effect = RuntimeError("db locked")
        mock_registry.get_active_triggers.return_value = []
        monitor._registries["moderate"] = mock_registry

        # Should not raise
        result = monitor.run_tick()

        # Evaluation still attempted
        mock_registry.get_active_triggers.assert_called_once()
        assert result["expired"] == 0


# ---------------------------------------------------------------------------
# Quote Batching Tests
# ---------------------------------------------------------------------------


class TestQuoteBatching:
    """Quote batching fetches each symbol only once per tick."""

    def test_duplicate_symbols_fetched_once(self, engine):
        """Multiple triggers for same symbol → single quote fetch."""
        monitor = FastPathMonitor(engine, ["moderate"])

        # Three triggers for TSLA, one for AAPL
        triggers = [
            FakeTrigger(trigger_id="t-1", symbol="TSLA"),
            FakeTrigger(trigger_id="t-2", symbol="TSLA"),
            FakeTrigger(trigger_id="t-3", symbol="TSLA"),
            FakeTrigger(trigger_id="t-4", symbol="AAPL"),
        ]

        mock_registry = MagicMock()
        mock_registry.expire_stale_triggers.return_value = 0
        mock_registry.get_active_triggers.return_value = triggers
        monitor._registries["moderate"] = mock_registry

        fetch_called_with = []

        def mock_fetch(symbols):
            fetch_called_with.append(symbols)
            return {s: {"price": 350.0, "age_ms": 0, "reliable": True} for s in symbols}

        with patch("utils.fast_path_monitor._fetch_quotes", side_effect=mock_fetch), \
             patch("utils.fast_path_monitor.evaluate_trigger", return_value=None), \
             patch("utils.fast_path_monitor.FAST_PATH_MODE", "observe"), \
             patch("utils.fast_path_monitor.FAST_PATH_MAX_OUTCOMES_PER_TICK", 5):
            monitor.run_tick()

        # _fetch_quotes called once with deduplicated set
        assert len(fetch_called_with) == 1
        assert fetch_called_with[0] == {"TSLA", "AAPL"}

    def test_symbols_across_profiles_deduplicated(self, engine):
        """Same symbol in different profiles → still fetched once."""
        monitor = FastPathMonitor(engine, ["conservative", "moderate"])

        mock_reg_1 = MagicMock()
        mock_reg_1.expire_stale_triggers.return_value = 0
        mock_reg_1.get_active_triggers.return_value = [
            FakeTrigger(trigger_id="c-1", symbol="TSLA", profile_id="conservative"),
        ]

        mock_reg_2 = MagicMock()
        mock_reg_2.expire_stale_triggers.return_value = 0
        mock_reg_2.get_active_triggers.return_value = [
            FakeTrigger(trigger_id="m-1", symbol="TSLA", profile_id="moderate"),
        ]

        monitor._registries["conservative"] = mock_reg_1
        monitor._registries["moderate"] = mock_reg_2

        fetch_called_with = []

        def mock_fetch(symbols):
            fetch_called_with.append(symbols)
            return {s: {"price": 350.0, "age_ms": 0, "reliable": True} for s in symbols}

        with patch("utils.fast_path_monitor._fetch_quotes", side_effect=mock_fetch), \
             patch("utils.fast_path_monitor.evaluate_trigger", return_value=None), \
             patch("utils.fast_path_monitor.FAST_PATH_MODE", "observe"), \
             patch("utils.fast_path_monitor.FAST_PATH_MAX_OUTCOMES_PER_TICK", 5):
            monitor.run_tick()

        # Only one symbol in the set
        assert len(fetch_called_with) == 1
        assert fetch_called_with[0] == {"TSLA"}

    def test_missing_quote_skips_trigger_evaluation(self, engine):
        """If a symbol has no quote (fetch failure), its triggers are skipped."""
        monitor = FastPathMonitor(engine, ["moderate"])

        triggers = [
            FakeTrigger(trigger_id="t-1", symbol="TSLA"),
            FakeTrigger(trigger_id="t-2", symbol="BROKEN"),
        ]

        mock_registry = MagicMock()
        mock_registry.expire_stale_triggers.return_value = 0
        mock_registry.get_active_triggers.return_value = triggers
        monitor._registries["moderate"] = mock_registry

        evaluate_calls = []

        def track_evaluate(trig, q, ps):
            evaluate_calls.append(trig.trigger_id)
            return None

        # Only return quote for TSLA, not BROKEN
        with patch("utils.fast_path_monitor._fetch_quotes", return_value={
                 "TSLA": {"price": 350.0, "age_ms": 0, "reliable": True},
             }), \
             patch("utils.fast_path_monitor.evaluate_trigger", side_effect=track_evaluate), \
             patch("utils.fast_path_monitor.FAST_PATH_MODE", "observe"), \
             patch("utils.fast_path_monitor.FAST_PATH_MAX_OUTCOMES_PER_TICK", 5):
            result = monitor.run_tick()

        # Only TSLA trigger was evaluated
        assert evaluate_calls == ["t-1"]
        # Both counted as evaluated in the summary (iterating all triggers)
        assert result["evaluated"] == 2


# ---------------------------------------------------------------------------
# Overrun Protection Tests
# ---------------------------------------------------------------------------


class TestOverrunProtection:
    """Overrun protection logs warning and returns partial results."""

    def test_tick_overrun_logs_warning_when_cap_reached(self, engine, caplog):
        """Overrun: cap reached logs info about deferring."""
        monitor = FastPathMonitor(engine, ["moderate"])

        # 3 triggers, cap of 2
        triggers = [
            FakeTrigger(trigger_id=f"trig-{i}", symbol=f"SYM{i}")
            for i in range(3)
        ]

        mock_registry = MagicMock()
        mock_registry.expire_stale_triggers.return_value = 0
        mock_registry.get_active_triggers.return_value = triggers
        monitor._registries["moderate"] = mock_registry

        def make_trade(trig, q, ps):
            return _make_outcome(trigger_id=trig.trigger_id, outcome_type="trade_executed")

        import logging
        with caplog.at_level(logging.INFO, logger="utils.fast_path_monitor"):
            with patch("utils.fast_path_monitor.evaluate_trigger", side_effect=make_trade), \
                 patch("utils.fast_path_monitor._fetch_quotes", return_value={
                     f"SYM{i}": {"price": 350.0, "age_ms": 0, "reliable": True}
                     for i in range(3)
                 }), \
                 patch("utils.fast_path_monitor._persist_event", return_value="evt-001"), \
                 patch("utils.fast_path_monitor._delegate_execution"), \
                 patch("utils.fast_path_monitor.FAST_PATH_MODE", "enabled"), \
                 patch("utils.fast_path_monitor.FAST_PATH_MAX_OUTCOMES_PER_TICK", 2):
                result = monitor.run_tick()

        assert "execution cap reached" in caplog.text
        assert "deferring" in caplog.text

    def test_partial_results_returned_on_overrun(self, engine):
        """Overrun: partial results show both delegated and deferred counts."""
        monitor = FastPathMonitor(engine, ["moderate"])

        # 4 triggers, cap of 2 → 2 delegated, 2 deferred
        triggers = [
            FakeTrigger(trigger_id=f"trig-{i}", symbol=f"SYM{i}")
            for i in range(4)
        ]

        mock_registry = MagicMock()
        mock_registry.expire_stale_triggers.return_value = 0
        mock_registry.get_active_triggers.return_value = triggers
        monitor._registries["moderate"] = mock_registry

        def make_trade(trig, q, ps):
            return _make_outcome(trigger_id=trig.trigger_id, outcome_type="trade_executed")

        delegate_calls = []

        def track_delegate(outcome, trigger, eng):
            delegate_calls.append(outcome.trigger_id)

        with patch("utils.fast_path_monitor.evaluate_trigger", side_effect=make_trade), \
             patch("utils.fast_path_monitor._fetch_quotes", return_value={
                 f"SYM{i}": {"price": 350.0, "age_ms": 0, "reliable": True}
                 for i in range(4)
             }), \
             patch("utils.fast_path_monitor._persist_event", return_value="evt-001"), \
             patch("utils.fast_path_monitor._delegate_execution", side_effect=track_delegate), \
             patch("utils.fast_path_monitor.FAST_PATH_MODE", "enabled"), \
             patch("utils.fast_path_monitor.FAST_PATH_MAX_OUTCOMES_PER_TICK", 2):
            result = monitor.run_tick()

        # Partial results: 2 delegated, 2 deferred
        assert len(delegate_calls) == 2
        assert result["deferred"] == 2
        assert result["fired"] == 4
        # All events still persisted regardless of cap
        assert result["outcomes"]["trade_executed"] == 4

    def test_all_events_persisted_despite_overrun(self, engine):
        """All outcomes written to fast_path_events even when cap defers execution."""
        monitor = FastPathMonitor(engine, ["moderate"])

        # 3 triggers, cap of 1
        triggers = [
            FakeTrigger(trigger_id=f"trig-{i}", symbol=f"SYM{i}")
            for i in range(3)
        ]

        mock_registry = MagicMock()
        mock_registry.expire_stale_triggers.return_value = 0
        mock_registry.get_active_triggers.return_value = triggers
        monitor._registries["moderate"] = mock_registry

        def make_trade(trig, q, ps):
            return _make_outcome(trigger_id=trig.trigger_id, outcome_type="trade_executed")

        persist_calls = []
        original_persist = None

        def counting_persist(outcome, trigger, eng):
            persist_calls.append(outcome.trigger_id)
            return f"evt-{outcome.trigger_id}"

        with patch("utils.fast_path_monitor.evaluate_trigger", side_effect=make_trade), \
             patch("utils.fast_path_monitor._fetch_quotes", return_value={
                 f"SYM{i}": {"price": 350.0, "age_ms": 0, "reliable": True}
                 for i in range(3)
             }), \
             patch("utils.fast_path_monitor._persist_event", side_effect=counting_persist), \
             patch("utils.fast_path_monitor._delegate_execution"), \
             patch("utils.fast_path_monitor.FAST_PATH_MODE", "enabled"), \
             patch("utils.fast_path_monitor.FAST_PATH_MAX_OUTCOMES_PER_TICK", 1):
            monitor.run_tick()

        # All 3 events persisted even though cap is 1
        assert len(persist_calls) == 3

    def test_overrun_tick_lock_released_properly(self, engine):
        """After an overrun tick, the lock is released for the next tick."""
        monitor = FastPathMonitor(engine, ["moderate"])

        triggers = [
            FakeTrigger(trigger_id=f"trig-{i}", symbol=f"SYM{i}")
            for i in range(3)
        ]

        mock_registry = MagicMock()
        mock_registry.expire_stale_triggers.return_value = 0
        mock_registry.get_active_triggers.return_value = triggers
        monitor._registries["moderate"] = mock_registry

        def make_trade(trig, q, ps):
            return _make_outcome(trigger_id=trig.trigger_id, outcome_type="trade_executed")

        with patch("utils.fast_path_monitor.evaluate_trigger", side_effect=make_trade), \
             patch("utils.fast_path_monitor._fetch_quotes", return_value={
                 f"SYM{i}": {"price": 350.0, "age_ms": 0, "reliable": True}
                 for i in range(3)
             }), \
             patch("utils.fast_path_monitor._persist_event", return_value="evt-001"), \
             patch("utils.fast_path_monitor._delegate_execution"), \
             patch("utils.fast_path_monitor.FAST_PATH_MODE", "enabled"), \
             patch("utils.fast_path_monitor.FAST_PATH_MAX_OUTCOMES_PER_TICK", 1):
            monitor.run_tick()

        # Lock should be released — next tick should be possible
        acquired = monitor._tick_lock.acquire(blocking=False)
        assert acquired is True
        monitor._tick_lock.release()
