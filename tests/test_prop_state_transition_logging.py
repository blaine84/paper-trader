"""
Property-based tests for state transition structured logging using Hypothesis.

Property 14: State transition structured logging.

Validates that every state transition emits an INFO-level log containing:
symbol, alert_type, old_status, new_status, reason.

The reason field is truncated to max 200 chars. The log format matches:
"ALERT_STATE_TRANSITION: symbol=%s alert_type=%s old_status=%s new_status=%s reason=%s"

Uses a real in-memory SQLite database with AlertIntentStore. Inserts intents,
performs transitions, and checks caplog for the expected structured log entries.

**Validates: Requirements 9.3**
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from sqlalchemy import create_engine, text

from utils.alert_dispatch_schema import init_alert_dispatch_schema
from utils.alert_intent_store import AlertIntentStore


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

SYMBOLS = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOG"]
ALERT_TYPES = ["entry_alert", "breakout", "rapid_move", "target_hit"]

st_symbol = st.sampled_from(SYMBOLS)
st_alert_type = st.sampled_from(ALERT_TYPES)

# Reason strings of varying lengths (including > 200 chars to test truncation)
st_reason = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"), max_codepoint=127),
    min_size=1,
    max_size=400,
)

# Transitions to test:
# pending → dispatched (mark_dispatched)
# dispatched → consumed (mark_consumed)
# pending → expired (transition_to_expired)
# pending → suppressed (mark_suppressed)
TRANSITIONS = [
    "pending_to_dispatched",
    "dispatched_to_consumed",
    "pending_to_expired",
    "pending_to_suppressed",
]

st_transition = st.sampled_from(TRANSITIONS)

_ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"
_BASE_TIME = datetime(2025, 6, 25, 14, 0, 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_store():
    """Create an in-memory SQLite engine with schema and return (engine, store)."""
    engine = create_engine("sqlite://", echo=False)
    init_alert_dispatch_schema(engine)
    return engine, AlertIntentStore(engine)


def _insert_intent(store, engine, symbol: str, alert_type: str) -> int:
    """Insert a pending intent and return its id."""
    data = {
        "symbol": symbol,
        "alert_type": alert_type,
        "direction": "long",
        "trigger_price": "150.00",
        "source_level": None,
        "urgency": "high",
        "reason": None,
        "dedupe_key": f"{symbol}:{alert_type}:{uuid.uuid4().hex[:16]}",
        "filter_status": "passed",
        "first_seen_at": _BASE_TIME.strftime(_ISO_FMT),
        "last_seen_at": _BASE_TIME.strftime(_ISO_FMT),
        "expiration_at": (_BASE_TIME + timedelta(hours=2)).strftime(_ISO_FMT),
    }
    intent = store.record_or_update_intent(data)
    return intent.id


def _set_status(engine, intent_id: int, status: str) -> None:
    """Directly set an intent's dispatch_status for test setup."""
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE alert_intents SET dispatch_status = :status WHERE id = :id"),
            {"status": status, "id": intent_id},
        )


# ---------------------------------------------------------------------------
# Property 14: State transition structured logging
# **Validates: Requirements 9.3**
# ---------------------------------------------------------------------------


class TestProperty14StateTransitionStructuredLogging:
    """
    Property 14: State transition structured logging.

    For each successful state transition:
    - Exactly one INFO log record containing "ALERT_STATE_TRANSITION"
    - Log contains correct symbol, alert_type, old_status, new_status
    - Reason in log is max 200 chars (truncated if longer)

    For terminal-state attempts:
    - No ALERT_STATE_TRANSITION log (only WARNING about skipped)

    **Validates: Requirements 9.3**
    """

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_mark_dispatched_emits_transition_log(
        self, symbol: str, alert_type: str, caplog
    ):
        """mark_dispatched emits INFO log with correct transition details."""
        engine, store = _create_store()
        intent_id = _insert_intent(store, engine, symbol, alert_type)

        with caplog.at_level(logging.INFO, logger="utils.alert_intent_store"):
            caplog.clear()
            store.mark_dispatched([intent_id], dispatch_ts=_BASE_TIME)

        # Find ALERT_STATE_TRANSITION records
        transition_records = [
            r for r in caplog.records
            if "ALERT_STATE_TRANSITION" in r.getMessage()
        ]

        assert len(transition_records) == 1, (
            f"Expected exactly 1 ALERT_STATE_TRANSITION log, got {len(transition_records)}"
        )

        msg = transition_records[0].getMessage()
        assert transition_records[0].levelno == logging.INFO
        assert f"symbol={symbol}" in msg
        assert f"alert_type={alert_type}" in msg
        assert "old_status=pending" in msg
        assert "new_status=dispatched" in msg

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_mark_consumed_emits_transition_log(
        self, symbol: str, alert_type: str, caplog
    ):
        """mark_consumed emits INFO log with correct transition details."""
        engine, store = _create_store()
        intent_id = _insert_intent(store, engine, symbol, alert_type)
        # Set to dispatched first (mark_consumed only transitions dispatched/claimed_by_scheduled)
        _set_status(engine, intent_id, "dispatched")

        with caplog.at_level(logging.INFO, logger="utils.alert_intent_store"):
            caplog.clear()
            store.mark_consumed([intent_id])

        transition_records = [
            r for r in caplog.records
            if "ALERT_STATE_TRANSITION" in r.getMessage()
        ]

        assert len(transition_records) == 1, (
            f"Expected exactly 1 ALERT_STATE_TRANSITION log, got {len(transition_records)}"
        )

        msg = transition_records[0].getMessage()
        assert transition_records[0].levelno == logging.INFO
        assert f"symbol={symbol}" in msg
        assert f"alert_type={alert_type}" in msg
        assert "old_status=dispatched" in msg
        assert "new_status=consumed" in msg

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
        reason=st_reason,
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_transition_to_expired_emits_transition_log(
        self, symbol: str, alert_type: str, reason: str, caplog
    ):
        """transition_to_expired emits INFO log with correct transition details."""
        engine, store = _create_store()
        intent_id = _insert_intent(store, engine, symbol, alert_type)

        with caplog.at_level(logging.INFO, logger="utils.alert_intent_store"):
            caplog.clear()
            result = store.transition_to_expired(intent_id, reason=reason)

        assert result is True

        transition_records = [
            r for r in caplog.records
            if "ALERT_STATE_TRANSITION" in r.getMessage()
        ]

        assert len(transition_records) == 1, (
            f"Expected exactly 1 ALERT_STATE_TRANSITION log, got {len(transition_records)}"
        )

        msg = transition_records[0].getMessage()
        assert transition_records[0].levelno == logging.INFO
        assert f"symbol={symbol}" in msg
        assert f"alert_type={alert_type}" in msg
        assert "old_status=pending" in msg
        assert "new_status=expired" in msg
        # Reason in log should be truncated to max 200 chars
        truncated_reason = reason[:200]
        assert f"reason={truncated_reason}" in msg

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
        reason=st_reason,
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_mark_suppressed_emits_transition_log(
        self, symbol: str, alert_type: str, reason: str, caplog
    ):
        """mark_suppressed emits INFO log with correct transition details."""
        engine, store = _create_store()
        intent_id = _insert_intent(store, engine, symbol, alert_type)

        with caplog.at_level(logging.INFO, logger="utils.alert_intent_store"):
            caplog.clear()
            store.mark_suppressed(intent_id, reason=reason)

        transition_records = [
            r for r in caplog.records
            if "ALERT_STATE_TRANSITION" in r.getMessage()
        ]

        assert len(transition_records) == 1, (
            f"Expected exactly 1 ALERT_STATE_TRANSITION log, got {len(transition_records)}"
        )

        msg = transition_records[0].getMessage()
        assert transition_records[0].levelno == logging.INFO
        assert f"symbol={symbol}" in msg
        assert f"alert_type={alert_type}" in msg
        assert "old_status=pending" in msg
        assert "new_status=suppressed" in msg
        # Reason in log should be truncated to max 200 chars
        truncated_reason = reason[:200]
        assert f"reason={truncated_reason}" in msg

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
        reason=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), max_codepoint=127),
            min_size=201,
            max_size=400,
        ),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_reason_truncated_to_200_chars(
        self, symbol: str, alert_type: str, reason: str, caplog
    ):
        """Reason field in transition log is truncated to max 200 characters."""
        assume(len(reason) > 200)

        engine, store = _create_store()
        intent_id = _insert_intent(store, engine, symbol, alert_type)

        with caplog.at_level(logging.INFO, logger="utils.alert_intent_store"):
            caplog.clear()
            store.transition_to_expired(intent_id, reason=reason)

        transition_records = [
            r for r in caplog.records
            if "ALERT_STATE_TRANSITION" in r.getMessage()
        ]

        assert len(transition_records) == 1

        msg = transition_records[0].getMessage()
        # The reason in the log should NOT contain the full >200 char string
        assert reason not in msg, (
            f"Full reason (len={len(reason)}) should not appear in log"
        )
        # But the first 200 chars should appear
        truncated = reason[:200]
        assert f"reason={truncated}" in msg, (
            f"Truncated reason (first 200 chars) should appear in log"
        )

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
        reason=st_reason,
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_log_format_matches_expected_pattern(
        self, symbol: str, alert_type: str, reason: str, caplog
    ):
        """Log format matches: ALERT_STATE_TRANSITION: symbol=%s alert_type=%s old_status=%s new_status=%s reason=%s."""
        engine, store = _create_store()
        intent_id = _insert_intent(store, engine, symbol, alert_type)

        with caplog.at_level(logging.INFO, logger="utils.alert_intent_store"):
            caplog.clear()
            store.mark_suppressed(intent_id, reason=reason)

        transition_records = [
            r for r in caplog.records
            if "ALERT_STATE_TRANSITION" in r.getMessage()
        ]

        assert len(transition_records) == 1

        msg = transition_records[0].getMessage()
        truncated_reason = reason[:200]
        expected = (
            f"ALERT_STATE_TRANSITION: symbol={symbol} alert_type={alert_type} "
            f"old_status=pending new_status=suppressed reason={truncated_reason}"
        )
        assert msg == expected, (
            f"Log message does not match expected format.\n"
            f"Expected: {expected!r}\n"
            f"Actual:   {msg!r}"
        )

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
        terminal_state=st.sampled_from(["consumed", "expired", "suppressed", "dispatch_failed"]),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_terminal_state_attempt_no_transition_log(
        self, symbol: str, alert_type: str, terminal_state: str, caplog
    ):
        """Attempting transition from terminal state emits no ALERT_STATE_TRANSITION log (only WARNING)."""
        engine, store = _create_store()
        intent_id = _insert_intent(store, engine, symbol, alert_type)
        # Put into terminal state
        _set_status(engine, intent_id, terminal_state)

        with caplog.at_level(logging.DEBUG, logger="utils.alert_intent_store"):
            caplog.clear()
            # Try to transition to expired from a terminal state
            result = store.transition_to_expired(intent_id, reason="test_reason")

        assert result is False, (
            f"Expected False (no transition) from terminal state {terminal_state}"
        )

        # No ALERT_STATE_TRANSITION log should be emitted
        transition_records = [
            r for r in caplog.records
            if "ALERT_STATE_TRANSITION" in r.getMessage()
        ]
        assert len(transition_records) == 0, (
            f"Expected no ALERT_STATE_TRANSITION log for terminal state attempt, "
            f"got {len(transition_records)}"
        )

        # Should have a WARNING about skipped transition
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
        ]
        assert len(warning_records) >= 1, (
            f"Expected at least one WARNING log for terminal state skip, got {len(warning_records)}"
        )

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
        terminal_state=st.sampled_from(["consumed", "expired", "suppressed", "dispatch_failed"]),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_mark_suppressed_terminal_state_no_transition_log(
        self, symbol: str, alert_type: str, terminal_state: str, caplog
    ):
        """mark_suppressed on a terminal-state intent emits no ALERT_STATE_TRANSITION log."""
        engine, store = _create_store()
        intent_id = _insert_intent(store, engine, symbol, alert_type)
        _set_status(engine, intent_id, terminal_state)

        with caplog.at_level(logging.DEBUG, logger="utils.alert_intent_store"):
            caplog.clear()
            store.mark_suppressed(intent_id, reason="should_be_skipped")

        transition_records = [
            r for r in caplog.records
            if "ALERT_STATE_TRANSITION" in r.getMessage()
        ]
        assert len(transition_records) == 0, (
            f"Expected no ALERT_STATE_TRANSITION log when mark_suppressed on "
            f"terminal state {terminal_state}, got {len(transition_records)}"
        )
