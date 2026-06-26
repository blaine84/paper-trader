"""Integration tests for end-to-end alert dispatch flows.

Exercises the full lifecycle from intent recording through dispatch,
using in-memory SQLite and mocked PM/LLM calls.

Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from utils.alert_dispatch_schema import init_alert_dispatch_schema
from utils.alert_intent_store import AlertIntentStore, build_dedupe_key
from utils.alert_dispatcher import AlertDispatcher


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


def _make_intent_data(symbol="NVDA", **overrides):
    """Factory for creating intent data dicts."""
    now = datetime.utcnow()
    data = {
        "symbol": symbol,
        "alert_type": "entry_alert",
        "direction": "long",
        "trigger_price": "145.50",
        "source_level": "breakout above 145",
        "urgency": "medium",
        "reason": "Price crossed resistance",
        "dedupe_key": build_dedupe_key(symbol, "entry_alert", "breakout above 145"),
        "filter_status": "unclassified",
        "first_seen_at": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "last_seen_at": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "expiration_at": (now + timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    }
    data.update(overrides)
    return data


def _classify_intent(store, intent, status="passed", urgency="high"):
    """Helper to classify an intent so it becomes eligible for dispatch."""
    store.update_classification(intent.id, status, urgency)


# ---------------------------------------------------------------------------
# Test: Full flow — record intent → classify → evaluate → dispatch PM → consume
# ---------------------------------------------------------------------------


class TestFullDispatchFlow:
    """End-to-end: record → classify → evaluate → dispatch → consume."""

    @patch("utils.alert_dispatcher.AlertDispatcher._is_market_hours", return_value=True)
    @patch("agents.portfolio_manager.run_profile")
    @patch("models.pm_profiles.ACTIVE_PROFILES", ["test_profile"])
    @patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "enabled")
    def test_full_flow_record_to_consume(self, mock_run_profile, mock_market, engine, store, dispatcher):
        # Step 1: Record intent
        intent = store.record_or_update_intent(_make_intent_data())
        assert intent.dispatch_status == "pending"
        assert intent.filter_status == "unclassified"

        # Step 2: Classify (simulating what _classify_unclassified would do)
        _classify_intent(store, intent, "passed", "high")

        # Step 3: Evaluate and dispatch (enabled mode)
        result = dispatcher.evaluate_and_dispatch()

        # Verify PM was called
        assert mock_run_profile.called
        assert result is not None
        assert result["dispatched"] == 1
        assert "NVDA" in result["symbols"]

        # Step 4: Verify intent is consumed
        now = datetime.utcnow()
        pending = store.query_pending(now=now)
        assert len(pending) == 0


# ---------------------------------------------------------------------------
# Test: PM failure recovery — dispatch → PM fails → revert → redispatch → succeed
# ---------------------------------------------------------------------------


class TestPMFailureRecovery:
    """PM failure reverts intent to pending, redispatch succeeds."""

    @patch("utils.alert_dispatcher.AlertDispatcher._is_market_hours", return_value=True)
    @patch("models.pm_profiles.ACTIVE_PROFILES", ["test_profile"])
    @patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "enabled")
    def test_pm_failure_reverts_then_redispatch_succeeds(self, mock_market, engine, store, dispatcher):
        # Record and classify intent
        intent = store.record_or_update_intent(_make_intent_data())
        _classify_intent(store, intent, "passed", "high")

        # First dispatch: PM fails
        with patch("agents.portfolio_manager.run_profile", side_effect=RuntimeError("PM crashed")):
            result = dispatcher.evaluate_and_dispatch()

        # Intent should be back to pending with attempt_count incremented
        assert result is None
        with engine.begin() as conn:
            row = conn.execute(text(
                "SELECT dispatch_status, dispatch_attempt_count, last_dispatch_error "
                "FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()
            assert row[0] == "pending"
            assert row[1] == 1
            assert "PM crashed" in row[2]

        # Clear cooldowns so the redispatch can proceed
        # (In real usage, the cooldown would expire after configured minutes)
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM alert_cooldowns"))

        # Second dispatch: PM succeeds
        with patch("agents.portfolio_manager.run_profile"):
            result = dispatcher.evaluate_and_dispatch()

        assert result is not None
        assert result["dispatched"] == 1

        # Verify consumed
        with engine.begin() as conn:
            row = conn.execute(text(
                "SELECT dispatch_status FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()
            assert row[0] == "consumed"


# ---------------------------------------------------------------------------
# Test: PM terminal failure — 3 consecutive PM failures → dispatch_failed
# ---------------------------------------------------------------------------


class TestPMTerminalFailure:
    """3 consecutive PM failures transitions intent to dispatch_failed."""

    @patch("utils.alert_dispatcher.AlertDispatcher._is_market_hours", return_value=True)
    @patch("models.pm_profiles.ACTIVE_PROFILES", ["test_profile"])
    @patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "enabled")
    def test_three_failures_reaches_terminal_state(self, mock_market, engine, store, dispatcher):
        intent = store.record_or_update_intent(_make_intent_data())
        _classify_intent(store, intent, "passed", "high")

        # Fail 3 times — clear cooldowns between each attempt so dispatch proceeds
        for i in range(3):
            with patch("agents.portfolio_manager.run_profile", side_effect=RuntimeError(f"PM fail #{i+1}")):
                dispatcher.evaluate_and_dispatch()
            # Clear cooldowns to allow redispatch attempt
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM alert_cooldowns"))

        # Verify terminal dispatch_failed state
        with engine.begin() as conn:
            row = conn.execute(text(
                "SELECT dispatch_status, dispatch_attempt_count FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()
            assert row[0] == "dispatch_failed"
            assert row[1] >= 3


# ---------------------------------------------------------------------------
# Test: Restart survival — persist cooldowns → reinitialize → verify enforcement
# ---------------------------------------------------------------------------


class TestRestartSurvival:
    """Cooldowns persist across simulated restart and still suppress dispatch."""

    @patch("utils.alert_dispatcher.AlertDispatcher._is_market_hours", return_value=True)
    @patch("models.pm_profiles.ACTIVE_PROFILES", ["test_profile"])
    @patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "enabled")
    def test_cooldowns_survive_restart(self, mock_market, engine, store, dispatcher):
        # Record cooldown for NVDA (expires in 15 minutes)
        now = datetime.utcnow()
        store.record_cooldown("NVDA", now + timedelta(minutes=15))

        # Simulate restart: create a new store and dispatcher from same engine
        store2 = AlertIntentStore(engine)
        dispatcher2 = AlertDispatcher(
            engine=engine,
            intent_store=store2,
            begin_pm_cycle=MagicMock(return_value=True),
            end_pm_cycle=MagicMock(),
        )

        # Record and classify a new intent for NVDA
        intent = store2.record_or_update_intent(_make_intent_data(symbol="NVDA"))
        _classify_intent(store2, intent, "passed", "high")

        # Dispatch should be suppressed by persisted cooldown
        with patch("agents.portfolio_manager.run_profile") as mock_pm:
            result = dispatcher2.evaluate_and_dispatch()

        # PM should NOT have been called (cooldown still active)
        assert result is None
        mock_pm.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Scheduled + alert coordination — scheduled PM claims → succeeds → consumed
# ---------------------------------------------------------------------------


class TestScheduledAlertCoordination:
    """Scheduled PM claims pending intents and consumes them on success."""

    @patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "enabled")
    def test_scheduled_pm_claims_and_consumes(self, engine, store, dispatcher):
        # Record and classify intent
        intent = store.record_or_update_intent(_make_intent_data())
        _classify_intent(store, intent, "passed", "high")

        # Scheduled PM claims the intent
        symbols, intent_ids = dispatcher.consume_for_scheduled_cycle()

        assert "NVDA" in symbols
        assert intent.id in intent_ids

        # Verify claimed_by_scheduled status
        with engine.begin() as conn:
            row = conn.execute(text(
                "SELECT dispatch_status FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()
            assert row[0] == "claimed_by_scheduled"

        # Confirm consumption (PM succeeded)
        dispatcher.confirm_scheduled_consumption(intent_ids)

        # Verify consumed
        with engine.begin() as conn:
            row = conn.execute(text(
                "SELECT dispatch_status FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()
            assert row[0] == "consumed"


# ---------------------------------------------------------------------------
# Test: Scheduled PM failure — claims intents → PM fails → reverts to pending
# ---------------------------------------------------------------------------


class TestScheduledPMFailure:
    """Scheduled PM failure reverts claimed intents back to pending."""

    @patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "enabled")
    def test_scheduled_pm_failure_reverts_claims(self, engine, store, dispatcher):
        # Record and classify intent
        intent = store.record_or_update_intent(_make_intent_data())
        _classify_intent(store, intent, "passed", "high")

        # Scheduled PM claims the intent
        symbols, intent_ids = dispatcher.consume_for_scheduled_cycle()

        assert len(intent_ids) > 0

        # PM fails — revert claim
        dispatcher.revert_scheduled_claim(intent_ids, "Scheduled PM error")

        # Verify intent back to pending with error recorded
        with engine.begin() as conn:
            row = conn.execute(text(
                "SELECT dispatch_status, last_dispatch_error FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()
            assert row[0] == "pending"
            assert "Scheduled PM error" in row[1]


# ---------------------------------------------------------------------------
# Test: Feature flag modes — "disabled" preserves behavior, "observe" logs only, "enabled" dispatches
# ---------------------------------------------------------------------------


class TestFeatureFlagModes:
    """Feature flag modes control dispatch behavior."""

    @patch("utils.alert_dispatcher.AlertDispatcher._is_market_hours", return_value=True)
    @patch("models.pm_profiles.ACTIVE_PROFILES", ["test_profile"])
    @patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "disabled")
    def test_disabled_mode_returns_none(self, mock_market, engine, store, dispatcher):
        intent = store.record_or_update_intent(_make_intent_data())
        _classify_intent(store, intent, "passed", "high")

        with patch("agents.portfolio_manager.run_profile") as mock_pm:
            result = dispatcher.evaluate_and_dispatch()

        assert result is None
        mock_pm.assert_not_called()

    @patch("utils.alert_dispatcher.AlertDispatcher._is_market_hours", return_value=True)
    @patch("models.pm_profiles.ACTIVE_PROFILES", ["test_profile"])
    @patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "observe")
    def test_observe_mode_logs_without_dispatch(self, mock_market, engine, store, dispatcher):
        intent = store.record_or_update_intent(_make_intent_data())
        _classify_intent(store, intent, "passed", "high")

        with patch("agents.portfolio_manager.run_profile") as mock_pm:
            result = dispatcher.evaluate_and_dispatch()

        # Observe mode: returns result with mode="observe" but does NOT call PM
        assert result is not None
        assert result["mode"] == "observe"
        assert result["would_dispatch"] >= 1
        mock_pm.assert_not_called()

        # Intent should still be pending (not consumed)
        with engine.begin() as conn:
            row = conn.execute(text(
                "SELECT dispatch_status FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()
            assert row[0] == "pending"

    @patch("utils.alert_dispatcher.AlertDispatcher._is_market_hours", return_value=True)
    @patch("models.pm_profiles.ACTIVE_PROFILES", ["test_profile"])
    @patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "enabled")
    def test_enabled_mode_dispatches(self, mock_market, engine, store, dispatcher):
        intent = store.record_or_update_intent(_make_intent_data())
        _classify_intent(store, intent, "passed", "high")

        with patch("agents.portfolio_manager.run_profile") as mock_pm:
            result = dispatcher.evaluate_and_dispatch()

        assert result is not None
        assert result["dispatched"] == 1
        mock_pm.assert_called()


# ---------------------------------------------------------------------------
# Test: Observe mode no double-trigger — doesn't interfere with legacy path
# ---------------------------------------------------------------------------


class TestObserveModeNoDoubleTrigger:
    """Observe mode does not alter intent state, so legacy path can proceed."""

    @patch("utils.alert_dispatcher.AlertDispatcher._is_market_hours", return_value=True)
    @patch("models.pm_profiles.ACTIVE_PROFILES", ["test_profile"])
    @patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "observe")
    def test_observe_mode_leaves_intents_pending(self, mock_market, engine, store, dispatcher):
        intent = store.record_or_update_intent(_make_intent_data())
        _classify_intent(store, intent, "passed", "high")

        # Run in observe mode
        result = dispatcher.evaluate_and_dispatch()

        assert result is not None
        assert result["mode"] == "observe"

        # Intent remains pending — legacy path can still pick it up
        now = datetime.utcnow()
        pending = store.query_pending(now=now)
        assert len(pending) == 1
        assert pending[0].id == intent.id


# ---------------------------------------------------------------------------
# Test: Crash recovery — simulate crash during dispatch → restart → stale intents recovered
# ---------------------------------------------------------------------------


class TestCrashRecovery:
    """Stale dispatched intents are recovered on restart."""

    def test_stale_dispatched_intents_recovered_to_pending(self, engine, store, dispatcher):
        # Record intent and manually set it to dispatched with stale timestamp
        intent = store.record_or_update_intent(_make_intent_data())
        _classify_intent(store, intent, "passed", "high")

        # Simulate crash: mark as dispatched with a stale timestamp (20 min ago)
        stale_time = datetime.utcnow() - timedelta(minutes=20)
        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE alert_intents SET dispatch_status = 'dispatched', "
                "dispatched_at = :ts WHERE id = :id"
            ), {"ts": stale_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), "id": intent.id})

        # Simulate restart: call recover
        now = datetime.utcnow()
        recovered = store.recover_stale_active_intents(
            dispatch_stale_minutes=10,
            scheduled_max_runtime_minutes=15,
            now=now,
        )

        assert recovered >= 1

        # Verify intent reverted to pending
        with engine.begin() as conn:
            row = conn.execute(text(
                "SELECT dispatch_status FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()
            assert row[0] == "pending"

    def test_stale_claimed_by_scheduled_recovered(self, engine, store, dispatcher):
        """claimed_by_scheduled intents stuck beyond timeout revert to pending."""
        intent = store.record_or_update_intent(_make_intent_data())
        _classify_intent(store, intent, "passed", "high")

        # Simulate crash: mark as claimed_by_scheduled with stale timestamp
        stale_time = datetime.utcnow() - timedelta(minutes=20)
        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE alert_intents SET dispatch_status = 'claimed_by_scheduled', "
                "dispatched_at = :ts WHERE id = :id"
            ), {"ts": stale_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), "id": intent.id})

        now = datetime.utcnow()
        recovered = store.recover_stale_active_intents(
            dispatch_stale_minutes=10,
            scheduled_max_runtime_minutes=15,
            now=now,
        )

        assert recovered >= 1

        with engine.begin() as conn:
            row = conn.execute(text(
                "SELECT dispatch_status FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()
            assert row[0] == "pending"

    def test_stale_dispatched_with_high_attempt_count_becomes_failed(self, engine, store, dispatcher):
        """Stale dispatched intents with attempt_count >= 3 go to dispatch_failed."""
        intent = store.record_or_update_intent(_make_intent_data())
        _classify_intent(store, intent, "passed", "high")

        # Simulate: dispatched, stale, attempt_count already at 3
        stale_time = datetime.utcnow() - timedelta(minutes=20)
        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE alert_intents SET dispatch_status = 'dispatched', "
                "dispatched_at = :ts, dispatch_attempt_count = 3 WHERE id = :id"
            ), {"ts": stale_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), "id": intent.id})

        now = datetime.utcnow()
        recovered = store.recover_stale_active_intents(
            dispatch_stale_minutes=10,
            scheduled_max_runtime_minutes=15,
            now=now,
        )

        assert recovered >= 1

        with engine.begin() as conn:
            row = conn.execute(text(
                "SELECT dispatch_status FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()
            assert row[0] == "dispatch_failed"


# ---------------------------------------------------------------------------
# Test: Classification batch limit — only classifies up to PM_ALERT_CLASSIFY_MAX_PER_PASS
# ---------------------------------------------------------------------------


class TestClassificationBatchLimit:
    """Dispatcher classifies at most PM_ALERT_CLASSIFY_MAX_PER_PASS per pass."""

    @patch("utils.alert_dispatcher.AlertDispatcher._is_market_hours", return_value=True)
    @patch("models.pm_profiles.ACTIVE_PROFILES", ["test_profile"])
    @patch("utils.gate_config.PM_ALERT_CLASSIFY_MAX_PER_PASS", 5)
    @patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "enabled")
    def test_batch_limit_enforced(self, mock_market, engine, store, dispatcher):
        # Create 8 unclassified intents (batch limit is 5)
        for i in range(8):
            store.record_or_update_intent(_make_intent_data(
                symbol=f"SYM{i}",
                dedupe_key=build_dedupe_key(f"SYM{i}", "entry_alert", f"setup_{i}"),
            ))

        # Mock filter_alert_with_llm to return actionable
        mock_filter_result = {"actionable": True, "urgency": "high"}

        with patch("agents.price_monitor.filter_alert_with_llm", return_value=mock_filter_result):
            # Run classify step only
            dispatcher._classify_unclassified()

        # Check how many were classified
        with engine.begin() as conn:
            classified_count = conn.execute(text(
                "SELECT COUNT(*) FROM alert_intents WHERE filter_status != 'unclassified'"
            )).scalar()
            unclassified_count = conn.execute(text(
                "SELECT COUNT(*) FROM alert_intents WHERE filter_status = 'unclassified'"
            )).scalar()

        # At most 5 should be classified
        assert classified_count <= 5
        assert unclassified_count >= 3
