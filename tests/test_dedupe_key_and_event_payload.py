"""Unit tests for generate_dedupe_key() and build_setup_exit_event_payload().

Tests the dedupe key generation and event payload construction for setup-aware
exit governance events.
Validates Requirements 3.9, 3.10, 6.1, 6.3.
"""

from datetime import datetime, timezone

import pytest

from utils.setup_aware_evaluator import (
    LIFECYCLE_STATE_TO_EVENT_TYPE,
    SETUP_EXIT_EVENT_TYPES,
    build_setup_exit_event_payload,
    generate_dedupe_key,
)


# ---------------------------------------------------------------------------
# generate_dedupe_key tests
# ---------------------------------------------------------------------------


class TestGenerateDedupeKey:
    """Test dedupe key generation format and truncation behavior."""

    def test_basic_format(self):
        """Key follows format: {event_type}:{trade_id}:{timestamp_truncated_to_minute}."""
        ts = datetime(2026, 5, 26, 14, 30, 45, 123456, tzinfo=timezone.utc)
        key = generate_dedupe_key("setup_exit_revalidated_hold", 42, ts)
        assert key == "setup_exit_revalidated_hold:42:2026-05-26T14:30"

    def test_truncates_seconds_and_microseconds(self):
        """Timestamp is truncated to minute — seconds and microseconds are removed."""
        ts1 = datetime(2026, 5, 26, 14, 30, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2026, 5, 26, 14, 30, 59, 999999, tzinfo=timezone.utc)
        key1 = generate_dedupe_key("setup_exit_alert", 10, ts1)
        key2 = generate_dedupe_key("setup_exit_alert", 10, ts2)
        assert key1 == key2

    def test_different_minutes_produce_different_keys(self):
        """Events in different minutes produce different dedupe keys."""
        ts1 = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        ts2 = datetime(2026, 5, 26, 14, 31, 0, tzinfo=timezone.utc)
        key1 = generate_dedupe_key("setup_exit_force_close", 5, ts1)
        key2 = generate_dedupe_key("setup_exit_force_close", 5, ts2)
        assert key1 != key2

    def test_different_trade_ids_produce_different_keys(self):
        """Same event_type and timestamp but different trade_id → different keys."""
        ts = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        key1 = generate_dedupe_key("setup_exit_alert", 1, ts)
        key2 = generate_dedupe_key("setup_exit_alert", 2, ts)
        assert key1 != key2

    def test_different_event_types_produce_different_keys(self):
        """Same trade_id and timestamp but different event_type → different keys."""
        ts = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        key1 = generate_dedupe_key("setup_exit_alert", 1, ts)
        key2 = generate_dedupe_key("setup_exit_force_close", 1, ts)
        assert key1 != key2

    def test_naive_datetime_handled(self):
        """Naive datetime (no tzinfo) still produces valid key format."""
        ts = datetime(2026, 5, 26, 14, 30, 45)
        key = generate_dedupe_key("setup_exit_thesis_invalidated", 99, ts)
        assert key == "setup_exit_thesis_invalidated:99:2026-05-26T14:30"

    def test_all_event_types(self):
        """All SETUP_EXIT_EVENT_TYPES produce valid keys."""
        ts = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        for event_type in SETUP_EXIT_EVENT_TYPES:
            key = generate_dedupe_key(event_type, 1, ts)
            assert key.startswith(f"{event_type}:1:")
            assert key.endswith("2026-01-01T09:30")


# ---------------------------------------------------------------------------
# build_setup_exit_event_payload tests
# ---------------------------------------------------------------------------


class TestBuildSetupExitEventPayload:
    """Test event payload construction with all required fields."""

    def test_all_required_fields_present(self):
        """Payload contains all required SetupExitEventPayload fields."""
        ts = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        payload = build_setup_exit_event_payload(
            trade_id=42,
            setup_type="news_breakout",
            event_type="setup_exit_revalidated_hold",
            minutes_held=95.5,
            decision_outcome="hold_valid_until_next_window",
            invalidation_criteria_used={"stop_price": 145.0},
            timestamp_utc=ts,
            base_force_close_limit=120,
            next_limit_if_extended=150,
            revalidation_attempted=True,
            extension_active=True,
            close_reason=None,
        )

        assert payload["trade_id"] == 42
        assert payload["setup_type"] == "news_breakout"
        assert payload["event_type"] == "setup_exit_revalidated_hold"
        assert payload["minutes_held"] == 95.5
        assert payload["decision_outcome"] == "hold_valid_until_next_window"
        assert payload["invalidation_criteria_used"] == {"stop_price": 145.0}
        assert payload["timestamp_utc"] == "2026-05-26T14:30:00+00:00"
        assert payload["base_force_close_limit"] == 120
        assert payload["next_limit_if_extended"] == 150
        assert payload["revalidation_attempted"] is True
        assert payload["extension_active"] is True
        assert payload["close_reason"] is None
        assert "dedupe_key" in payload

    def test_dedupe_key_included_and_correct(self):
        """Payload includes a dedupe_key generated from event_type, trade_id, timestamp."""
        ts = datetime(2026, 5, 26, 14, 30, 45, tzinfo=timezone.utc)
        payload = build_setup_exit_event_payload(
            trade_id=42,
            setup_type="news_breakout",
            event_type="setup_exit_revalidated_hold",
            minutes_held=95.5,
            decision_outcome="hold_valid_until_next_window",
            invalidation_criteria_used=None,
            timestamp_utc=ts,
            base_force_close_limit=120,
            next_limit_if_extended=150,
            revalidation_attempted=True,
            extension_active=True,
            close_reason=None,
        )

        expected_key = "setup_exit_revalidated_hold:42:2026-05-26T14:30"
        assert payload["dedupe_key"] == expected_key

    def test_timestamp_utc_is_iso_format(self):
        """timestamp_utc field is ISO-8601 formatted string."""
        ts = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        payload = build_setup_exit_event_payload(
            trade_id=1,
            setup_type="momentum_fade",
            event_type="setup_exit_force_close",
            minutes_held=80.0,
            decision_outcome="close_time_expired",
            invalidation_criteria_used=None,
            timestamp_utc=ts,
            base_force_close_limit=75,
            next_limit_if_extended=None,
            revalidation_attempted=False,
            extension_active=False,
            close_reason="Setup time limit exceeded",
        )

        # Should be parseable as ISO-8601
        parsed = datetime.fromisoformat(payload["timestamp_utc"])
        assert parsed == ts

    def test_close_reason_included_when_closing(self):
        """close_reason is populated for close decisions."""
        ts = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        payload = build_setup_exit_event_payload(
            trade_id=5,
            setup_type="orb",
            event_type="setup_exit_force_close",
            minutes_held=80.0,
            decision_outcome="close_time_expired",
            invalidation_criteria_used=None,
            timestamp_utc=ts,
            base_force_close_limit=75,
            next_limit_if_extended=None,
            revalidation_attempted=False,
            extension_active=False,
            close_reason="Setup time limit exceeded: orb held 80 min (limit: 75 min)",
        )

        assert payload["close_reason"] == "Setup time limit exceeded: orb held 80 min (limit: 75 min)"

    def test_invalidation_criteria_none_when_not_applicable(self):
        """invalidation_criteria_used is None for non-revalidation events."""
        ts = datetime(2026, 5, 26, 10, 0, 0, tzinfo=timezone.utc)
        payload = build_setup_exit_event_payload(
            trade_id=3,
            setup_type="short_squeeze",
            event_type="setup_exit_alert",
            minutes_held=35.0,
            decision_outcome="warn_revalidation_due",
            invalidation_criteria_used=None,
            timestamp_utc=ts,
            base_force_close_limit=60,
            next_limit_if_extended=None,
            revalidation_attempted=False,
            extension_active=False,
            close_reason=None,
        )

        assert payload["invalidation_criteria_used"] is None

    def test_next_limit_if_extended_none_for_non_extension(self):
        """next_limit_if_extended is None for non-extension-eligible setups."""
        ts = datetime(2026, 5, 26, 10, 0, 0, tzinfo=timezone.utc)
        payload = build_setup_exit_event_payload(
            trade_id=7,
            setup_type="gap_and_go",
            event_type="setup_exit_force_close",
            minutes_held=92.0,
            decision_outcome="close_time_expired",
            invalidation_criteria_used=None,
            timestamp_utc=ts,
            base_force_close_limit=90,
            next_limit_if_extended=None,
            revalidation_attempted=False,
            extension_active=False,
            close_reason="Time limit exceeded",
        )

        assert payload["next_limit_if_extended"] is None


# ---------------------------------------------------------------------------
# LIFECYCLE_STATE_TO_EVENT_TYPE mapping tests
# ---------------------------------------------------------------------------


class TestLifecycleStateToEventTypeMapping:
    """Test the mapping from lifecycle states to event types."""

    def test_all_setup_aware_states_mapped(self):
        """All SETUP_AWARE_STATES have a corresponding event type mapping."""
        from utils.setup_aware_evaluator import SETUP_AWARE_STATES

        for state in SETUP_AWARE_STATES:
            assert state in LIFECYCLE_STATE_TO_EVENT_TYPE

    def test_mapped_event_types_are_valid(self):
        """All mapped event types are in SETUP_EXIT_EVENT_TYPES."""
        for event_type in LIFECYCLE_STATE_TO_EVENT_TYPE.values():
            assert event_type in SETUP_EXIT_EVENT_TYPES

    def test_specific_mappings(self):
        """Verify specific state → event_type mappings from design doc."""
        assert LIFECYCLE_STATE_TO_EVENT_TYPE["setup_exit_alert"] == "setup_exit_alert"
        assert LIFECYCLE_STATE_TO_EVENT_TYPE["setup_revalidation_hold"] == "setup_exit_revalidated_hold"
        assert LIFECYCLE_STATE_TO_EVENT_TYPE["setup_revalidation_failed"] == "setup_exit_revalidation_failed"
        assert LIFECYCLE_STATE_TO_EVENT_TYPE["setup_time_limit_exceeded"] == "setup_exit_force_close"
        assert LIFECYCLE_STATE_TO_EVENT_TYPE["setup_thesis_invalidated"] == "setup_exit_thesis_invalidated"
