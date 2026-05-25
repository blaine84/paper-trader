"""
Integration tests for review visibility and events.

Tests the interaction between setup-aware event types, dedupe key generation,
event payload construction, daily review summaries, CEO memo inputs, and
case-memory classification.

Validates Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
"""

import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine

from db.schema import Base, TradeEvent, get_session
from utils.setup_aware_evaluator import (
    LIFECYCLE_STATE_TO_EVENT_TYPE,
    SETUP_AWARE_STATES,
    SETUP_EXIT_EVENT_TYPES,
    build_setup_exit_event_payload,
    generate_dedupe_key,
)
from utils.case_memory_classifier import (
    CASE_MEMORY_EXIT_CATEGORIES,
    classify_trade_exit,
)
from agents.daily_review import gather_setup_aware_exit_summary
from agents.ceo import gather_setup_aware_governance_inputs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine():
    """Create an in-memory SQLite engine with all tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


# ---------------------------------------------------------------------------
# Test 1: SETUP_EXIT_EVENT_TYPES contains exactly 6 event types
# Validates: Requirement 6.2
# ---------------------------------------------------------------------------


class TestSetupExitEventTypes:
    """Verify SETUP_EXIT_EVENT_TYPES contains exactly the 6 required event types."""

    def test_exactly_six_event_types(self):
        assert len(SETUP_EXIT_EVENT_TYPES) == 6

    def test_contains_all_required_types(self):
        expected = {
            "setup_exit_alert",
            "setup_exit_revalidation_due",
            "setup_exit_revalidated_hold",
            "setup_exit_revalidation_failed",
            "setup_exit_force_close",
            "setup_exit_thesis_invalidated",
        }
        assert SETUP_EXIT_EVENT_TYPES == expected


# ---------------------------------------------------------------------------
# Test 2: Dedupe key generation — same minute produces identical keys,
# different minutes produce different keys
# Validates: Requirement 6.3
# ---------------------------------------------------------------------------


class TestDedupeKeyIntegration:
    """Verify dedupe key behavior for suppressing duplicate events."""

    def test_same_event_type_trade_id_same_minute_produces_identical_keys(self):
        """Events within the same minute for same trade+type get the same key."""
        ts1 = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        ts2 = datetime(2026, 5, 26, 14, 30, 45, tzinfo=timezone.utc)
        ts3 = datetime(2026, 5, 26, 14, 30, 59, 999999, tzinfo=timezone.utc)

        key1 = generate_dedupe_key("setup_exit_revalidated_hold", 42, ts1)
        key2 = generate_dedupe_key("setup_exit_revalidated_hold", 42, ts2)
        key3 = generate_dedupe_key("setup_exit_revalidated_hold", 42, ts3)

        assert key1 == key2 == key3

    def test_different_minutes_produce_different_keys(self):
        """Events in different minutes produce different dedupe keys."""
        ts1 = datetime(2026, 5, 26, 14, 30, 59, tzinfo=timezone.utc)
        ts2 = datetime(2026, 5, 26, 14, 31, 0, tzinfo=timezone.utc)

        key1 = generate_dedupe_key("setup_exit_force_close", 10, ts1)
        key2 = generate_dedupe_key("setup_exit_force_close", 10, ts2)

        assert key1 != key2

    def test_dedupe_suppresses_duplicate_within_same_minute(self):
        """Simulates the executor checking dedupe_key before writing events."""
        ts = datetime(2026, 5, 26, 14, 30, 25, tzinfo=timezone.utc)

        # Build two payloads for the same event within the same minute
        payload1 = build_setup_exit_event_payload(
            trade_id=42,
            setup_type="news_breakout",
            event_type="setup_exit_revalidated_hold",
            minutes_held=95.0,
            decision_outcome="hold_valid_until_next_window",
            invalidation_criteria_used={"stop_price": 145.0},
            timestamp_utc=ts,
            base_force_close_limit=120,
            next_limit_if_extended=150,
            revalidation_attempted=True,
            extension_active=True,
            close_reason=None,
        )

        ts2 = ts + timedelta(seconds=30)
        payload2 = build_setup_exit_event_payload(
            trade_id=42,
            setup_type="news_breakout",
            event_type="setup_exit_revalidated_hold",
            minutes_held=95.5,
            decision_outcome="hold_valid_until_next_window",
            invalidation_criteria_used={"stop_price": 145.0},
            timestamp_utc=ts2,
            base_force_close_limit=120,
            next_limit_if_extended=150,
            revalidation_attempted=True,
            extension_active=True,
            close_reason=None,
        )

        # Same dedupe_key → executor would suppress the second write
        assert payload1["dedupe_key"] == payload2["dedupe_key"]


# ---------------------------------------------------------------------------
# Test 3: build_setup_exit_event_payload produces dict with all 13 required fields
# Validates: Requirement 6.1
# ---------------------------------------------------------------------------


class TestEventPayloadFields:
    """Verify event payload contains all 13 required fields."""

    REQUIRED_FIELDS = {
        "trade_id",
        "setup_type",
        "event_type",
        "minutes_held",
        "decision_outcome",
        "invalidation_criteria_used",
        "timestamp_utc",
        "base_force_close_limit",
        "next_limit_if_extended",
        "revalidation_attempted",
        "extension_active",
        "close_reason",
        "dedupe_key",
    }

    def test_payload_has_all_13_required_fields(self):
        ts = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        payload = build_setup_exit_event_payload(
            trade_id=1,
            setup_type="news_breakout",
            event_type="setup_exit_alert",
            minutes_held=65.0,
            decision_outcome="warn_revalidation_due",
            invalidation_criteria_used=None,
            timestamp_utc=ts,
            base_force_close_limit=120,
            next_limit_if_extended=None,
            revalidation_attempted=False,
            extension_active=False,
            close_reason=None,
        )

        assert set(payload.keys()) == self.REQUIRED_FIELDS

    def test_payload_has_exactly_13_fields(self):
        ts = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        payload = build_setup_exit_event_payload(
            trade_id=99,
            setup_type="trend_pullback",
            event_type="setup_exit_revalidated_hold",
            minutes_held=125.0,
            decision_outcome="hold_valid_until_next_window",
            invalidation_criteria_used={"stop_price": 390.0, "support": 388.0},
            timestamp_utc=ts,
            base_force_close_limit=150,
            next_limit_if_extended=180,
            revalidation_attempted=True,
            extension_active=True,
            close_reason=None,
        )

        assert len(payload) == 13


# ---------------------------------------------------------------------------
# Test 4: classify_trade_exit produces exactly one of 4 categories
# Validates: Requirement 6.6
# ---------------------------------------------------------------------------


class TestCaseMemoryClassificationIntegration:
    """Verify case-memory classification produces exactly one category per trade."""

    def test_thesis_invalidation_produces_one_category(self):
        trade = {
            "id": 1, "symbol": "AMD", "setup_type": "news_breakout",
            "direction": "LONG", "entry_price": 150.0, "exit_price": 145.0,
        }
        events = [{"event_type": "setup_exit_thesis_invalidated"}]
        result = classify_trade_exit(trade, events)
        assert result in CASE_MEMORY_EXIT_CATEGORIES
        assert result == "valid_exit_thesis_invalidated"

    def test_missing_metadata_produces_one_category(self):
        trade = {
            "id": 2, "symbol": "NVDA", "setup_type": "news_catalyst",
            "direction": "LONG", "entry_price": 100.0, "exit_price": 102.0,
        }
        events = [
            {"event_type": "setup_exit_revalidation_failed",
             "reason": "Extension denied: missing criteria"},
        ]
        result = classify_trade_exit(trade, events)
        assert result in CASE_MEMORY_EXIT_CATEGORIES
        assert result == "forced_exit_missing_metadata"

    def test_profitable_thesis_dev_force_close_produces_one_category(self):
        trade = {
            "id": 3, "symbol": "MSFT", "setup_type": "trend_pullback",
            "direction": "LONG", "entry_price": 400.0, "exit_price": 410.0,
        }
        events = [{"event_type": "setup_exit_force_close"}]
        result = classify_trade_exit(trade, events)
        assert result in CASE_MEMORY_EXIT_CATEGORIES
        assert result == "valid_entry_bad_exit_policy"

    def test_losing_trade_no_special_events_produces_one_category(self):
        trade = {
            "id": 4, "symbol": "SPY", "setup_type": "orb",
            "direction": "LONG", "entry_price": 450.0, "exit_price": 445.0,
        }
        events = [{"event_type": "stop_triggered"}]
        result = classify_trade_exit(trade, events)
        assert result in CASE_MEMORY_EXIT_CATEGORIES
        assert result == "bad_entry"

    def test_all_event_combinations_produce_exactly_one_category(self):
        """Various trade/event combos always produce exactly one valid category."""
        test_cases = [
            # Thesis invalidation with multiple events
            (
                {"id": 10, "setup_type": "news_breakout", "direction": "LONG",
                 "entry_price": 100.0, "exit_price": 95.0},
                [{"event_type": "setup_exit_alert"},
                 {"event_type": "setup_exit_thesis_invalidated"}],
            ),
            # Force close on non-thesis-dev setup
            (
                {"id": 11, "setup_type": "momentum_fade", "direction": "SHORT",
                 "entry_price": 50.0, "exit_price": 48.0},
                [{"event_type": "setup_exit_force_close"}],
            ),
            # Stale data failure
            (
                {"id": 12, "setup_type": "news_breakout", "direction": "LONG",
                 "entry_price": 100.0, "exit_price": 101.0},
                [{"event_type": "setup_exit_revalidation_failed",
                  "message": "Revalidation failed: stale data"}],
            ),
            # Empty events
            (
                {"id": 13, "setup_type": "gap_and_go", "direction": "LONG",
                 "entry_price": 30.0, "exit_price": 28.0},
                [],
            ),
        ]
        for trade, events in test_cases:
            result = classify_trade_exit(trade, events)
            assert result in CASE_MEMORY_EXIT_CATEGORIES, (
                f"Trade {trade['id']} got invalid category: {result}"
            )


# ---------------------------------------------------------------------------
# Test 5: LIFECYCLE_STATE_TO_EVENT_TYPE maps all 5 SETUP_AWARE_STATES
# to valid event types in SETUP_EXIT_EVENT_TYPES
# Validates: Requirement 6.1, 6.2
# ---------------------------------------------------------------------------


class TestLifecycleStateToEventTypeMapping:
    """Verify state-to-event-type mapping covers all states with valid types."""

    def test_all_five_setup_aware_states_are_mapped(self):
        assert len(SETUP_AWARE_STATES) == 5
        for state in SETUP_AWARE_STATES:
            assert state in LIFECYCLE_STATE_TO_EVENT_TYPE, (
                f"State '{state}' not found in LIFECYCLE_STATE_TO_EVENT_TYPE"
            )

    def test_all_mapped_values_are_valid_event_types(self):
        for state, event_type in LIFECYCLE_STATE_TO_EVENT_TYPE.items():
            assert event_type in SETUP_EXIT_EVENT_TYPES, (
                f"State '{state}' maps to '{event_type}' which is not in SETUP_EXIT_EVENT_TYPES"
            )

    def test_mapping_has_exactly_five_entries(self):
        assert len(LIFECYCLE_STATE_TO_EVENT_TYPE) == 5


# ---------------------------------------------------------------------------
# Test 6: gather_setup_aware_exit_summary returns expected schema
# Validates: Requirement 6.4
# ---------------------------------------------------------------------------


class TestDailyReviewSummarySchema:
    """Verify gather_setup_aware_exit_summary returns the expected schema."""

    EXPECTED_KEYS = {
        "total_setup_aware_events",
        "by_setup_type",
        "timer_based_closes",
        "thesis_invalidation_closes",
        "revalidated_holds",
    }

    def test_empty_db_returns_expected_schema(self):
        engine = _make_engine()
        today = datetime.now().strftime("%Y-%m-%d")
        result = gather_setup_aware_exit_summary(engine, today)

        assert set(result.keys()) == self.EXPECTED_KEYS
        assert result["total_setup_aware_events"] == 0
        assert isinstance(result["by_setup_type"], dict)
        assert result["timer_based_closes"] == 0
        assert result["thesis_invalidation_closes"] == 0
        assert result["revalidated_holds"] == 0

    def test_populated_db_returns_expected_schema(self):
        engine = _make_engine()
        today = datetime.now().strftime("%Y-%m-%d")
        db = get_session(engine)

        # Add various setup-aware events
        db.add(TradeEvent(
            trade_id=1,
            event_type="setup_exit_force_close",
            timestamp=datetime.now(),
            payload_json=json.dumps({"setup_type": "news_breakout"}),
        ))
        db.add(TradeEvent(
            trade_id=2,
            event_type="setup_exit_thesis_invalidated",
            timestamp=datetime.now(),
            payload_json=json.dumps({"setup_type": "news_catalyst"}),
        ))
        db.add(TradeEvent(
            trade_id=3,
            event_type="setup_exit_revalidated_hold",
            timestamp=datetime.now(),
            payload_json=json.dumps({"setup_type": "trend_pullback"}),
        ))
        db.commit()
        db.close()

        result = gather_setup_aware_exit_summary(engine, today)

        assert set(result.keys()) == self.EXPECTED_KEYS
        assert result["total_setup_aware_events"] == 3
        assert result["timer_based_closes"] == 1
        assert result["thesis_invalidation_closes"] == 1
        assert result["revalidated_holds"] == 1
        assert "news_breakout" in result["by_setup_type"]
        assert "news_catalyst" in result["by_setup_type"]
        assert "trend_pullback" in result["by_setup_type"]

    def test_summarizes_setup_aware_exits_separately(self):
        """Setup-aware events are counted separately; non-setup events excluded."""
        engine = _make_engine()
        today = datetime.now().strftime("%Y-%m-%d")
        db = get_session(engine)

        # Setup-aware event
        db.add(TradeEvent(
            trade_id=1,
            event_type="setup_exit_force_close",
            timestamp=datetime.now(),
            payload_json=json.dumps({"setup_type": "orb"}),
        ))
        # Non-setup-aware event (should be excluded)
        db.add(TradeEvent(
            trade_id=2,
            event_type="news_expiry_force_close",
            timestamp=datetime.now(),
            payload_json=json.dumps({"setup_type": "news_breakout"}),
        ))
        db.commit()
        db.close()

        result = gather_setup_aware_exit_summary(engine, today)

        assert result["total_setup_aware_events"] == 1
        assert "orb" in result["by_setup_type"]
        # news_breakout should NOT appear because its event_type is not setup-aware
        assert "news_breakout" not in result["by_setup_type"]


# ---------------------------------------------------------------------------
# Test 7: CEO memo includes counts and examples
# Validates: Requirement 6.5
# ---------------------------------------------------------------------------


class TestCeoMemoInputs:
    """Verify CEO memo gather function includes counts and examples."""

    def test_empty_db_returns_structured_result(self):
        engine = _make_engine()
        result = gather_setup_aware_governance_inputs(engine)

        assert "forced_exits" in result
        assert "revalidated_holds" in result
        assert "news_breakout_exits" in result
        assert "missing_metadata_denials" in result

        assert result["forced_exits"]["count"] == 0
        assert result["forced_exits"]["examples"] == []
        assert result["revalidated_holds"]["count"] == 0
        assert result["revalidated_holds"]["examples"] == []

    def test_forced_exits_counted_with_examples(self):
        engine = _make_engine()
        db = get_session(engine)

        db.add(TradeEvent(
            trade_id=1,
            event_type="setup_exit_force_close",
            symbol="AMD",
            timestamp=datetime.utcnow(),
            payload_json=json.dumps({
                "setup_type": "news_breakout",
                "minutes_held": 125.0,
            }),
        ))
        db.add(TradeEvent(
            trade_id=2,
            event_type="setup_exit_force_close",
            symbol="NVDA",
            timestamp=datetime.utcnow(),
            payload_json=json.dumps({
                "setup_type": "momentum_fade",
                "minutes_held": 80.0,
            }),
        ))
        db.commit()
        db.close()

        result = gather_setup_aware_governance_inputs(engine)

        assert result["forced_exits"]["count"] == 2
        assert "news_breakout" in result["forced_exits"]["by_setup_type"]
        assert "momentum_fade" in result["forced_exits"]["by_setup_type"]
        assert len(result["forced_exits"]["examples"]) == 2

    def test_revalidated_holds_counted_with_examples(self):
        engine = _make_engine()
        db = get_session(engine)

        db.add(TradeEvent(
            trade_id=3,
            event_type="setup_exit_revalidated_hold",
            symbol="TSLA",
            timestamp=datetime.utcnow(),
            payload_json=json.dumps({
                "setup_type": "news_catalyst",
                "minutes_held": 95.0,
            }),
        ))
        db.commit()
        db.close()

        result = gather_setup_aware_governance_inputs(engine)

        assert result["revalidated_holds"]["count"] == 1
        assert result["revalidated_holds"]["by_setup_type"]["news_catalyst"] == 1
        assert len(result["revalidated_holds"]["examples"]) == 1
        assert result["revalidated_holds"]["examples"][0]["symbol"] == "TSLA"

    def test_examples_capped_at_three(self):
        engine = _make_engine()
        db = get_session(engine)

        for i in range(5):
            db.add(TradeEvent(
                trade_id=i + 1,
                event_type="setup_exit_force_close",
                symbol=f"SYM{i}",
                timestamp=datetime.utcnow(),
                payload_json=json.dumps({
                    "setup_type": "news_breakout",
                    "minutes_held": 120.0 + i,
                }),
            ))
        db.commit()
        db.close()

        result = gather_setup_aware_governance_inputs(engine)

        assert result["forced_exits"]["count"] == 5
        assert len(result["forced_exits"]["examples"]) <= 3

    def test_news_breakout_timer_vs_invalidation(self):
        engine = _make_engine()
        db = get_session(engine)

        db.add(TradeEvent(
            trade_id=1,
            event_type="setup_exit_force_close",
            symbol="AMD",
            timestamp=datetime.utcnow(),
            payload_json=json.dumps({"setup_type": "news_breakout", "minutes_held": 120.0}),
        ))
        db.add(TradeEvent(
            trade_id=2,
            event_type="setup_exit_thesis_invalidated",
            symbol="NVDA",
            timestamp=datetime.utcnow(),
            payload_json=json.dumps({"setup_type": "news_breakout", "minutes_held": 85.0}),
        ))
        db.commit()
        db.close()

        result = gather_setup_aware_governance_inputs(engine)

        assert result["news_breakout_exits"]["timer_closes"] == 1
        assert result["news_breakout_exits"]["thesis_invalidation_closes"] == 1
        assert len(result["news_breakout_exits"]["examples"]) == 2
