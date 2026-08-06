"""Regression tests for material occurrence deduplication — Task 6.

Integration tests validating that the material occurrence fix prevents
repeated would_dispatch audit rows when the same alert condition is
observed multiple times without material change.

Uses real SQLite engine, real AlertIntentStore, real AlertDispatcher.

Acceptance Criteria 1 (Task 6.1):
  Same alert observed 5 times with small price drift < 0.5% produces
  exactly 1 would_dispatch audit row.

Acceptance Criteria 2, 3 (Task 6.2):
  Repeated observations update last_seen_at and trigger_price, increment
  occurrence_count, but material_occurrence_count remains at 1.
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


def _make_intent_data(symbol="AMD", alert_type="entry_alert", trigger_price="155.20", **overrides):
    """Factory for creating intent data dicts with sensible defaults."""
    now = datetime.utcnow()
    data = {
        "symbol": symbol,
        "alert_type": alert_type,
        "direction": "long",
        "trigger_price": trigger_price,
        "source_level": "resistance_155",
        "urgency": "medium",
        "reason": "Price near resistance",
        "dedupe_key": build_dedupe_key(symbol, alert_type, "resistance_155"),
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


class TestObserveSameAlertMultipleTicksOneWouldDispatch:
    """Create one active alert intent. Re-record same alert 5 times
    (same symbol/type/direction/source_level, small price drift < 0.5%).
    Run dispatcher after each. Assert exactly 1 would_dispatch audit row.

    Validates: Acceptance Criteria 1
    """

    @patch("utils.alert_dispatcher.AlertDispatcher._is_market_hours", return_value=True)
    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "enabled")
    @patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "observe")
    @patch("utils.gate_config.PM_ALERT_MODE_ENTRY_ALERT", "")
    @patch("utils.gate_config.PM_ALERT_MODE_BREAKOUT", "")
    @patch("utils.gate_config.PM_ALERT_MODE_RAPID_MOVE", "")
    @patch("utils.gate_config.PM_ALERT_MODE_TARGET_HIT", "")
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_ENTRY_ALERT_MINUTES", 15)
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_observe_same_alert_multiple_ticks_one_would_dispatch(
        self, mock_market, engine, store, dispatcher
    ):
        """Same alert observed 5 times with price drift < 0.5% yields exactly 1 would_dispatch."""
        # Tick prices: all within 0.5% of 155.20 (0.5% of 155.20 = 0.776)
        # Max drift: 155.20 → 155.25 = 0.032% — well within threshold
        tick_prices = ["155.20", "155.22", "155.18", "155.25", "155.21"]

        # Create the initial intent (tick 1)
        intent = store.record_or_update_intent(
            _make_intent_data(trigger_price=tick_prices[0])
        )
        _classify_intent(store, intent)

        # Run dispatcher after first observation
        dispatcher.evaluate_and_dispatch()

        # Ticks 2-5: re-record same alert with slight price drift, then dispatch
        for price in tick_prices[1:]:
            now = datetime.utcnow()
            store.record_or_update_intent(
                _make_intent_data(
                    trigger_price=price,
                    last_seen_at=now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                )
            )
            dispatcher.evaluate_and_dispatch()

        # Assert: exactly 1 would_dispatch audit row
        with engine.begin() as conn:
            rows = conn.execute(text(
                "SELECT dispatch_status, occurrence_count, material_occurrence_count "
                "FROM alert_dispatch_log "
                "WHERE alert_intent_id = :aid AND dispatch_status = 'would_dispatch'"
            ), {"aid": intent.alert_intent_id}).fetchall()

        assert len(rows) == 1, (
            f"Expected exactly 1 would_dispatch row, got {len(rows)}. "
            f"Rows: {rows}"
        )

        # Verify the intent's counters: occurrence_count=5, material_occurrence_count=1
        with engine.begin() as conn:
            intent_row = conn.execute(text(
                "SELECT occurrence_count, material_occurrence_count "
                "FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()

        assert intent_row[0] == 5, f"Expected occurrence_count=5, got {intent_row[0]}"
        assert intent_row[1] == 1, f"Expected material_occurrence_count=1, got {intent_row[1]}"


class TestDeferredUnchangedAlertDoesNotBypassOnRawCount:
    """Set deferred_until in the future. Re-observe same alert (occurrence_count
    increments, material stays). Run dispatcher. Assert no new would_dispatch.

    The first dispatcher pass writes one would_dispatch and sets deferral for 15 min.
    Subsequent raw observations increment occurrence_count but NOT material_occurrence_count.
    Because material_occurrence_count == occurrence_count_at_deferral, the deferral
    holds and no additional would_dispatch rows are written.

    Validates: Acceptance Criteria 4
    """

    @patch("utils.alert_dispatcher.AlertDispatcher._is_market_hours", return_value=True)
    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "enabled")
    @patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "observe")
    @patch("utils.gate_config.PM_ALERT_MODE_ENTRY_ALERT", "")
    @patch("utils.gate_config.PM_ALERT_MODE_BREAKOUT", "")
    @patch("utils.gate_config.PM_ALERT_MODE_RAPID_MOVE", "")
    @patch("utils.gate_config.PM_ALERT_MODE_TARGET_HIT", "")
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_ENTRY_ALERT_MINUTES", 15)
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_deferred_unchanged_alert_does_not_bypass_on_raw_count(
        self, mock_market, engine, store, dispatcher
    ):
        """Deferral holds when only raw occurrence_count increases (material stays unchanged)."""
        # Tick prices: all within 0.5% of 155.20 (0.5% of 155.20 = 0.776)
        tick_prices = ["155.20", "155.22", "155.18", "155.25", "155.21"]

        # Record initial intent and classify as "passed" with urgency "high"
        intent = store.record_or_update_intent(
            _make_intent_data(trigger_price=tick_prices[0])
        )
        _classify_intent(store, intent, status="passed", urgency="high")

        # First dispatcher pass: writes would_dispatch + sets deferral for 15 min
        dispatcher.evaluate_and_dispatch()

        # Verify: exactly 1 would_dispatch after first pass
        with engine.begin() as conn:
            rows = conn.execute(text(
                "SELECT id FROM alert_dispatch_log "
                "WHERE alert_intent_id = :aid AND dispatch_status = 'would_dispatch'"
            ), {"aid": intent.alert_intent_id}).fetchall()
        assert len(rows) == 1, f"Expected 1 would_dispatch after first pass, got {len(rows)}"

        # Verify deferral was set
        with engine.begin() as conn:
            intent_row = conn.execute(text(
                "SELECT deferred_until, occurrence_count_at_deferral "
                "FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()
        assert intent_row[0] is not None, "deferred_until should be set after first dispatch"
        assert intent_row[1] is not None, "occurrence_count_at_deferral should be set"

        # Ticks 2-5: Re-record same alert with slight price drift, then run dispatcher
        for i, price in enumerate(tick_prices[1:], start=2):
            now = datetime.utcnow()
            store.record_or_update_intent(
                _make_intent_data(
                    trigger_price=price,
                    last_seen_at=now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                )
            )
            # Run dispatcher — deferral should hold, no new would_dispatch
            dispatcher.evaluate_and_dispatch()

        # Assert: STILL exactly 1 would_dispatch (deferral prevented additional ones)
        with engine.begin() as conn:
            rows = conn.execute(text(
                "SELECT dispatch_status, occurrence_count, material_occurrence_count "
                "FROM alert_dispatch_log "
                "WHERE alert_intent_id = :aid AND dispatch_status = 'would_dispatch'"
            ), {"aid": intent.alert_intent_id}).fetchall()

        assert len(rows) == 1, (
            f"Expected exactly 1 would_dispatch row (deferral should hold), got {len(rows)}. "
            f"Rows: {rows}"
        )

        # Verify occurrence_count DID increment (proving raw count went up)
        with engine.begin() as conn:
            final_row = conn.execute(text(
                "SELECT occurrence_count, material_occurrence_count "
                "FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()

        assert final_row[0] == 5, (
            f"Expected occurrence_count=5 (raw count incremented each tick), got {final_row[0]}"
        )
        # material_occurrence_count stays at 1 (no material change detected)
        assert final_row[1] == 1, (
            f"Expected material_occurrence_count=1 (unchanged), got {final_row[1]}"
        )


class TestRepeatedObservationUpdatesLastSeenWithoutMaterialOccurrence:
    """Record same alert multiple times. Assert last_seen_at advances,
    trigger_price updates, occurrence_count increments, and
    material_occurrence_count remains at 1.

    Validates: Acceptance Criteria 2, 3
    """

    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "enabled")
    def test_repeated_observation_updates_last_seen_without_material_occurrence(
        self, engine, store
    ):
        """Repeated observations update freshness metadata but don't increment material count."""
        base_time = datetime(2024, 6, 15, 14, 0, 0)

        # Timestamps T0 through T3, each 1 minute apart
        timestamps = [base_time + timedelta(minutes=i) for i in range(4)]

        # Prices: all within 0.5% of 155.20 (0.5% = 0.776)
        # Max drift: 155.25 - 155.18 = 0.07 / 155.20 = 0.045% — well within threshold
        prices = ["155.20", "155.22", "155.18", "155.25"]

        # T0: Initial record
        intent = store.record_or_update_intent(
            _make_intent_data(
                trigger_price=prices[0],
                first_seen_at=timestamps[0].strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                last_seen_at=timestamps[0].strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            )
        )

        # Verify initial state
        assert intent.occurrence_count == 1
        assert intent.material_occurrence_count == 1

        # T1, T2, T3: Re-record with new prices and timestamps
        for i in range(1, 4):
            updated_intent = store.record_or_update_intent(
                _make_intent_data(
                    trigger_price=prices[i],
                    last_seen_at=timestamps[i].strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                )
            )

            # last_seen_at advances to the latest timestamp
            expected_ts = timestamps[i].strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            actual_ts = updated_intent.last_seen_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            assert actual_ts == expected_ts, (
                f"Tick {i}: expected last_seen_at={expected_ts}, got {actual_ts}"
            )

            # trigger_price reflects the latest price
            assert str(updated_intent.trigger_price) == prices[i], (
                f"Tick {i}: expected trigger_price={prices[i]}, "
                f"got {updated_intent.trigger_price}"
            )

            # occurrence_count incremented
            assert updated_intent.occurrence_count == i + 1, (
                f"Tick {i}: expected occurrence_count={i + 1}, "
                f"got {updated_intent.occurrence_count}"
            )

            # material_occurrence_count remains at 1
            assert updated_intent.material_occurrence_count == 1, (
                f"Tick {i}: expected material_occurrence_count=1, "
                f"got {updated_intent.material_occurrence_count}"
            )

        # Final verification directly from DB for completeness
        with engine.begin() as conn:
            row = conn.execute(text(
                "SELECT occurrence_count, material_occurrence_count, "
                "trigger_price, last_seen_at "
                "FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()

        assert row[0] == 4, f"Final occurrence_count should be 4, got {row[0]}"
        assert row[1] == 1, f"Final material_occurrence_count should be 1, got {row[1]}"
        assert row[2] == "155.25", f"Final trigger_price should be 155.25, got {row[2]}"


class TestDispatchModeSameAlertDoesNotRepeatPmCycle:
    """With dispatch enabled, repeated same-key observations produce one PM
    cycle, not one per tick.

    Flow:
    1. Set up dispatcher in "dispatch" mode (not "observe")
    2. Record initial intent, classify it
    3. Run dispatcher multiple times (5 ticks) with same alert (small price drift < 0.5%)
    4. Track how many times begin_pm_cycle is called (use MagicMock)
    5. Assert: begin_pm_cycle was called at most once (not once per tick)

    After the first successful dispatch, the intent is consumed. Subsequent
    observations create new unclassified rows. Even when classified, per-symbol
    cooldown prevents re-dispatch within the cooldown window.

    Validates: Acceptance Criteria 6
    """

    @patch("utils.alert_dispatcher.AlertDispatcher._is_market_hours", return_value=True)
    @patch("utils.gate_config.ALERT_MATERIAL_OCCURRENCE_MODE", "enabled")
    @patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "dispatch")
    @patch("utils.gate_config.PM_ALERT_MODE_ENTRY_ALERT", "")
    @patch("utils.gate_config.PM_ALERT_MODE_BREAKOUT", "")
    @patch("utils.gate_config.PM_ALERT_MODE_RAPID_MOVE", "")
    @patch("utils.gate_config.PM_ALERT_MODE_TARGET_HIT", "")
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_ENTRY_ALERT_MINUTES", 60)
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_dispatch_mode_same_alert_does_not_repeat_pm_cycle(
        self, mock_market, engine, store
    ):
        """Repeated same-key observations produce one PM cycle, not one per tick."""
        begin_pm = MagicMock(return_value=True)
        end_pm = MagicMock()
        dispatcher = AlertDispatcher(
            engine=engine,
            intent_store=store,
            begin_pm_cycle=begin_pm,
            end_pm_cycle=end_pm,
        )

        # Tick prices: all within 0.5% of 155.20 (0.5% of 155.20 = 0.776)
        tick_prices = ["155.20", "155.22", "155.18", "155.25", "155.21"]

        # Tick 1: Record initial intent and classify it
        intent = store.record_or_update_intent(
            _make_intent_data(trigger_price=tick_prices[0])
        )
        _classify_intent(store, intent)

        # Run dispatcher — should dispatch and call begin_pm_cycle once
        with patch("agents.portfolio_manager.run_profile"):
            with patch("models.pm_profiles.ACTIVE_PROFILES", ["default"]):
                dispatcher.evaluate_and_dispatch()

        # Ticks 2-5: Re-record same alert with slight price drift, classify, dispatch
        for price in tick_prices[1:]:
            now = datetime.utcnow()
            new_intent = store.record_or_update_intent(
                _make_intent_data(
                    trigger_price=price,
                    last_seen_at=now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                )
            )
            # Classify the new row (simulating LLM classification completing)
            _classify_intent(store, new_intent)

            # Run dispatcher — cooldown should block re-dispatch
            with patch("agents.portfolio_manager.run_profile"):
                with patch("models.pm_profiles.ACTIVE_PROFILES", ["default"]):
                    dispatcher.evaluate_and_dispatch()

        # Assert: begin_pm_cycle was called at most once (not once per tick)
        assert begin_pm.call_count <= 1, (
            f"Expected begin_pm_cycle to be called at most once, "
            f"but it was called {begin_pm.call_count} times"
        )


class TestRearmedAlertAllowsNewWouldDispatch:
    """Simulate deferral window expiring. Condition still active. Run dispatcher.
    Assert second would_dispatch after re-arm.

    Flow:
    1. Record initial intent and classify it
    2. Run dispatcher once → first would_dispatch + deferral set (cooldown 15 min)
    3. Simulate time passing: directly update deferred_until in the DB to a past
       timestamp (simulating deferral expiry) AND update last_seen_at to a recent
       timestamp (so freshness check passes)
    4. Run dispatcher again → should trigger re-arm (bumps material_occurrence_count)
       + write second would_dispatch
    5. Assert: exactly 2 would_dispatch rows total

    Validates: Acceptance Criteria 5
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
    def test_rearmed_alert_allows_new_would_dispatch(
        self, mock_market, engine, store, dispatcher
    ):
        """Deferral expires with stable condition → re-arm → second would_dispatch."""
        now = datetime.utcnow()

        # Step 1: Record initial intent and classify
        intent = store.record_or_update_intent(
            _make_intent_data(
                trigger_price="155.20",
                first_seen_at=now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                last_seen_at=now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            )
        )
        _classify_intent(store, intent)

        # Step 2: Run dispatcher → first would_dispatch + deferral set
        dispatcher.evaluate_and_dispatch()

        # Verify: 1 would_dispatch written
        with engine.begin() as conn:
            rows = conn.execute(text(
                "SELECT id FROM alert_dispatch_log "
                "WHERE alert_intent_id = :aid AND dispatch_status = 'would_dispatch'"
            ), {"aid": intent.alert_intent_id}).fetchall()
        assert len(rows) == 1, f"Expected 1 would_dispatch after first pass, got {len(rows)}"

        # Verify deferral was set with correct snapshot
        with engine.begin() as conn:
            intent_row = conn.execute(text(
                "SELECT deferred_until, occurrence_count_at_deferral, material_occurrence_count "
                "FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()
        assert intent_row[0] is not None, "deferred_until should be set"
        assert intent_row[1] == 1, f"occurrence_count_at_deferral should be 1, got {intent_row[1]}"
        assert intent_row[2] == 1, f"material_occurrence_count should be 1, got {intent_row[2]}"

        # Step 3: Simulate time passing — set deferred_until to the past
        # and update last_seen_at to a recent timestamp so freshness check passes
        past_deferral = (now - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        recent_last_seen = now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE alert_intents SET deferred_until = :past, last_seen_at = :recent "
                "WHERE id = :id"
            ), {"past": past_deferral, "recent": recent_last_seen, "id": intent.id})

        # Step 4: Run dispatcher again → re-arm should trigger + second would_dispatch
        dispatcher.evaluate_and_dispatch()

        # Step 5: Assert exactly 2 would_dispatch rows total
        with engine.begin() as conn:
            rows = conn.execute(text(
                "SELECT dispatch_status, occurrence_count, material_occurrence_count "
                "FROM alert_dispatch_log "
                "WHERE alert_intent_id = :aid AND dispatch_status = 'would_dispatch' "
                "ORDER BY id ASC"
            ), {"aid": intent.alert_intent_id}).fetchall()

        assert len(rows) == 2, (
            f"Expected exactly 2 would_dispatch rows (first dispatch + re-arm), "
            f"got {len(rows)}. Rows: {rows}"
        )

        # First row should have material_occurrence_count=1
        assert rows[0][2] == 1, (
            f"First would_dispatch should have material_occurrence_count=1, got {rows[0][2]}"
        )
        # Second row should have material_occurrence_count=2 (re-armed)
        assert rows[1][2] == 2, (
            f"Second would_dispatch should have material_occurrence_count=2, got {rows[1][2]}"
        )

        # Verify the intent's material_occurrence_count was bumped to 2
        with engine.begin() as conn:
            final_row = conn.execute(text(
                "SELECT occurrence_count, material_occurrence_count, "
                "occurrence_count_at_deferral, deferred_until "
                "FROM alert_intents WHERE id = :id"
            ), {"id": intent.id}).fetchone()

        assert final_row[1] == 2, (
            f"Expected material_occurrence_count=2 after re-arm, got {final_row[1]}"
        )
        # After re-arm + second would_dispatch, a new deferral should be set
        # with occurrence_count_at_deferral = 2 (the new material count)
        assert final_row[2] == 2, (
            f"Expected occurrence_count_at_deferral=2 after re-arm, got {final_row[2]}"
        )
        assert final_row[3] is not None, "deferred_until should be set after second dispatch"
