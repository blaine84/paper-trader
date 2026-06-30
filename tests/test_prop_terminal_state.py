"""
Property-based tests for terminal state immutability in AlertIntentStore.

Property 13: Terminal state immutability.

Validates that once an alert intent reaches a terminal state (consumed, expired,
suppressed, dispatch_failed), no transition method can change its dispatch_status.
Terminal state transitions are logged as warnings.

**Validates: Requirements 9.6, 8.5**
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from hypothesis import given, settings, assume
from hypothesis import strategies as st
from sqlalchemy import create_engine, text

from utils.alert_dispatch_schema import init_alert_dispatch_schema
from utils.alert_intent_store import AlertIntentStore, TERMINAL_STATES


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

SYMBOLS = ["NVDA", "AAPL", "TSLA", "MSFT", "GOOG"]
ALERT_TYPES = ["entry_alert", "breakout", "rapid_move", "target_hit"]
TERMINAL_STATE_LIST = list(TERMINAL_STATES)

st_symbol = st.sampled_from(SYMBOLS)
st_alert_type = st.sampled_from(ALERT_TYPES)
st_terminal_state = st.sampled_from(TERMINAL_STATE_LIST)

_BASE_TIME = datetime(2025, 1, 15, 12, 0, 0)
_ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_store():
    """Create an in-memory SQLite engine with schema and return a store."""
    engine = create_engine("sqlite://", echo=False)
    init_alert_dispatch_schema(engine)
    return AlertIntentStore(engine)


def _insert_terminal_intent(store: AlertIntentStore, symbol: str, alert_type: str, terminal_state: str) -> int:
    """Insert an intent and force it into a terminal state. Returns the intent id."""
    data = {
        "symbol": symbol,
        "alert_type": alert_type,
        "direction": "long",
        "trigger_price": "150.00",
        "source_level": None,
        "urgency": "medium",
        "reason": "test intent",
        "dedupe_key": str(uuid.uuid4()),
        "filter_status": "passed",
        "first_seen_at": _BASE_TIME.strftime(_ISO_FMT),
        "last_seen_at": _BASE_TIME.strftime(_ISO_FMT),
        "expiration_at": (_BASE_TIME + timedelta(hours=4)).strftime(_ISO_FMT),
    }
    intent = store.record_or_update_intent(data)

    # Force the intent into the target terminal state directly
    with store._engine.begin() as conn:
        conn.execute(
            text("UPDATE alert_intents SET dispatch_status = :status WHERE id = :id"),
            {"status": terminal_state, "id": intent.id},
        )

    return intent.id


def _get_dispatch_status(store: AlertIntentStore, intent_id: int) -> str:
    """Query the current dispatch_status for an intent by id."""
    with store._engine.begin() as conn:
        row = conn.execute(
            text("SELECT dispatch_status FROM alert_intents WHERE id = :id"),
            {"id": intent_id},
        ).fetchone()
    return row[0] if row else "NOT_FOUND"


# ---------------------------------------------------------------------------
# Property 13: Terminal state immutability
# ---------------------------------------------------------------------------


class TestProperty13TerminalStateImmutability:
    """
    Property 13: Terminal state immutability.

    Once an intent reaches a terminal state (consumed, expired, suppressed,
    dispatch_failed), no transition method can change its dispatch_status.

    **Validates: Requirements 9.6, 8.5**
    """

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
        terminal_state=st_terminal_state,
    )
    @settings(max_examples=50)
    def test_transition_to_expired_cannot_change_terminal_state(
        self, symbol: str, alert_type: str, terminal_state: str
    ):
        """transition_to_expired returns False and does not change terminal intents."""
        store = _create_store()
        intent_id = _insert_terminal_intent(store, symbol, alert_type, terminal_state)

        # Attempt to transition to expired
        result = store.transition_to_expired(intent_id, reason="freshness_expired")

        # transition_to_expired should return False for terminal intents
        assert result is False, (
            f"transition_to_expired returned True for intent in terminal state "
            f"'{terminal_state}' (symbol={symbol}, alert_type={alert_type})"
        )

        # dispatch_status must remain unchanged
        actual_status = _get_dispatch_status(store, intent_id)
        assert actual_status == terminal_state, (
            f"dispatch_status changed from '{terminal_state}' to '{actual_status}' "
            f"after transition_to_expired attempt"
        )

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
        terminal_state=st_terminal_state,
    )
    @settings(max_examples=50)
    def test_mark_dispatched_cannot_change_terminal_state(
        self, symbol: str, alert_type: str, terminal_state: str
    ):
        """mark_dispatched skips terminal intents (rowcount reflects skip)."""
        store = _create_store()
        intent_id = _insert_terminal_intent(store, symbol, alert_type, terminal_state)

        dispatch_ts = _BASE_TIME + timedelta(hours=1)
        updated = store.mark_dispatched([intent_id], dispatch_ts)

        # Should update 0 rows (terminal intent skipped)
        assert updated == 0, (
            f"mark_dispatched updated {updated} rows for intent in terminal state "
            f"'{terminal_state}' — expected 0"
        )

        # dispatch_status must remain unchanged
        actual_status = _get_dispatch_status(store, intent_id)
        assert actual_status == terminal_state, (
            f"dispatch_status changed from '{terminal_state}' to '{actual_status}' "
            f"after mark_dispatched attempt"
        )

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
        terminal_state=st_terminal_state,
    )
    @settings(max_examples=50)
    def test_mark_consumed_cannot_change_terminal_state(
        self, symbol: str, alert_type: str, terminal_state: str
    ):
        """mark_consumed skips terminal intents (rowcount reflects skip)."""
        store = _create_store()
        intent_id = _insert_terminal_intent(store, symbol, alert_type, terminal_state)

        updated = store.mark_consumed([intent_id])

        # Should update 0 rows (terminal intent skipped)
        assert updated == 0, (
            f"mark_consumed updated {updated} rows for intent in terminal state "
            f"'{terminal_state}' — expected 0"
        )

        # dispatch_status must remain unchanged
        actual_status = _get_dispatch_status(store, intent_id)
        assert actual_status == terminal_state, (
            f"dispatch_status changed from '{terminal_state}' to '{actual_status}' "
            f"after mark_consumed attempt"
        )

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
        terminal_state=st_terminal_state,
    )
    @settings(max_examples=50)
    def test_mark_suppressed_cannot_change_terminal_state(
        self, symbol: str, alert_type: str, terminal_state: str
    ):
        """mark_suppressed skips terminal intents (no-op for terminal states)."""
        store = _create_store()
        intent_id = _insert_terminal_intent(store, symbol, alert_type, terminal_state)

        # mark_suppressed doesn't return a count, but it should not change the status
        store.mark_suppressed(intent_id, reason="test_suppression")

        # dispatch_status must remain unchanged
        actual_status = _get_dispatch_status(store, intent_id)
        assert actual_status == terminal_state, (
            f"dispatch_status changed from '{terminal_state}' to '{actual_status}' "
            f"after mark_suppressed attempt"
        )

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
        terminal_state=st_terminal_state,
    )
    @settings(max_examples=50)
    def test_mark_dispatch_failed_cannot_change_terminal_state(
        self, symbol: str, alert_type: str, terminal_state: str
    ):
        """mark_dispatch_failed skips terminal intents."""
        store = _create_store()
        intent_id = _insert_terminal_intent(store, symbol, alert_type, terminal_state)

        # mark_dispatch_failed should not touch terminal-state intents
        store.mark_dispatch_failed([intent_id], error="simulated failure")

        # dispatch_status must remain unchanged
        actual_status = _get_dispatch_status(store, intent_id)
        assert actual_status == terminal_state, (
            f"dispatch_status changed from '{terminal_state}' to '{actual_status}' "
            f"after mark_dispatch_failed attempt"
        )

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
        terminal_state=st_terminal_state,
    )
    @settings(max_examples=50)
    def test_all_transitions_attempted_on_same_terminal_intent(
        self, symbol: str, alert_type: str, terminal_state: str
    ):
        """Applying ALL transition methods sequentially does not alter a terminal intent."""
        store = _create_store()
        intent_id = _insert_terminal_intent(store, symbol, alert_type, terminal_state)

        dispatch_ts = _BASE_TIME + timedelta(hours=1)

        # Attempt every transition method
        store.transition_to_expired(intent_id, reason="freshness_expired")
        store.mark_dispatched([intent_id], dispatch_ts)
        store.mark_consumed([intent_id])
        store.mark_suppressed(intent_id, reason="test_suppression")
        store.mark_dispatch_failed([intent_id], error="simulated error")

        # After all attempts, dispatch_status must still be the original terminal state
        actual_status = _get_dispatch_status(store, intent_id)
        assert actual_status == terminal_state, (
            f"dispatch_status changed from '{terminal_state}' to '{actual_status}' "
            f"after applying all transition methods"
        )
