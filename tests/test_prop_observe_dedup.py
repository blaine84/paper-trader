"""
Property-based tests for observe deduplication using Hypothesis.

Property 5: Observe deduplication — no repeated rows for unchanged occurrences.

Validates that _handle_observe():
1. Calling _handle_observe() twice with the same intent (unchanged dedupe_key,
   trigger_price, occurrence_count) produces exactly 1 would_dispatch audit row
2. Calling _handle_observe() with a changed occurrence (different trigger_price
   or incremented occurrence_count) produces a new audit row
3. The has_would_dispatch_for_occurrence() check prevents duplicate writes

Uses a real in-memory SQLite database with AlertIntentStore and AlertDispatcher.

**Validates: Requirements 2.3, 2.4, 8.2**
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st
from sqlalchemy import create_engine, text

from utils.alert_dispatch_schema import init_alert_dispatch_schema
from utils.alert_intent_store import AlertIntentStore, AlertIntent
from utils.alert_dispatcher import AlertDispatcher


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

SYMBOLS = ["NVDA", "AAPL", "TSLA"]
ALERT_TYPES = ["entry_alert", "breakout", "rapid_move"]

st_symbol = st.sampled_from(SYMBOLS)
st_alert_type = st.sampled_from(ALERT_TYPES)
st_trigger_price = st.decimals(min_value=Decimal("1.00"), max_value=Decimal("9999.99"), places=2)
st_occurrence_count = st.integers(min_value=1, max_value=100)

_BASE_TIME = datetime(2025, 1, 15, 10, 0, 0)
_ISO_FMT = "%Y-%m-%dT%H:%M:%S.000Z"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_store_and_dispatcher():
    """Create in-memory SQLite with schema, store, and dispatcher."""
    engine = create_engine("sqlite://", echo=False)
    init_alert_dispatch_schema(engine)
    store = AlertIntentStore(engine)
    dispatcher = AlertDispatcher(
        engine=engine,
        intent_store=store,
        begin_pm_cycle=lambda s: True,
        end_pm_cycle=lambda s: None,
    )
    return engine, store, dispatcher


def _insert_intent(store, *, symbol: str, alert_type: str, trigger_price: Decimal,
                   occurrence_count: int, dedupe_key: str) -> AlertIntent:
    """Insert a test intent with passed filter and pending status."""
    now = _BASE_TIME
    data = {
        "symbol": symbol,
        "alert_type": alert_type,
        "direction": "long",
        "trigger_price": str(trigger_price),
        "source_level": None,
        "urgency": "medium",
        "reason": None,
        "dedupe_key": dedupe_key,
        "filter_status": "passed",
        "first_seen_at": now.strftime(_ISO_FMT),
        "last_seen_at": now.strftime(_ISO_FMT),
        "expiration_at": "2025-01-30T23:00:00.000Z",
    }
    intent = store.record_or_update_intent(data)

    # Set occurrence_count directly if > 1
    if occurrence_count > 1:
        with store._engine.begin() as conn:
            conn.execute(
                text("UPDATE alert_intents SET occurrence_count = :oc WHERE id = :id"),
                {"oc": occurrence_count, "id": intent.id},
            )
        # Re-read to get updated intent
        with store._engine.begin() as conn:
            row = conn.execute(
                text("SELECT * FROM alert_intents WHERE id = :id"),
                {"id": intent.id},
            ).fetchone()
        intent = store._row_to_intent(row)

    return intent


def _count_would_dispatch_rows(engine, alert_intent_id: str) -> int:
    """Count would_dispatch rows in alert_dispatch_log for a given intent."""
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT COUNT(*) FROM alert_dispatch_log
            WHERE alert_intent_id = :aid AND dispatch_status = 'would_dispatch'
        """), {"aid": alert_intent_id}).fetchone()
    return row[0]


# ---------------------------------------------------------------------------
# Property 5: Observe deduplication — no repeated rows for unchanged
#              occurrences
# ---------------------------------------------------------------------------


class TestProperty5ObserveDeduplication:
    """
    Property 5: Observe deduplication — no repeated rows for unchanged occurrences.

    Validates:
    - After N calls to _handle_observe with SAME state → exactly 1 would_dispatch row
    - After a material change (trigger_price or occurrence_count) → a new row appears
    - has_would_dispatch_for_occurrence returns True after first observe, blocking duplicates

    **Validates: Requirements 2.3, 2.4, 8.2**
    """

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
        trigger_price=st_trigger_price,
        occurrence_count=st_occurrence_count,
        num_calls=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=50)
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_repeated_observe_same_state_produces_one_row(
        self,
        symbol: str,
        alert_type: str,
        trigger_price: Decimal,
        occurrence_count: int,
        num_calls: int,
    ):
        """After N calls to _handle_observe with SAME state → exactly 1 would_dispatch row."""
        engine, store, dispatcher = _create_store_and_dispatcher()

        dedupe_key = f"{symbol}:{alert_type}:{uuid.uuid4().hex[:16]}"
        intent = _insert_intent(
            store,
            symbol=symbol,
            alert_type=alert_type,
            trigger_price=trigger_price,
            occurrence_count=occurrence_count,
            dedupe_key=dedupe_key,
        )

        now = _BASE_TIME + timedelta(minutes=1)

        # Call _handle_observe N times with the same intent state
        for _ in range(num_calls):
            dispatcher._handle_observe(intent, now)

        # Exactly 1 would_dispatch row should exist
        count = _count_would_dispatch_rows(engine, intent.alert_intent_id)
        assert count == 1, (
            f"Expected exactly 1 would_dispatch row after {num_calls} calls "
            f"with same state, got {count}"
        )

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
        trigger_price=st_trigger_price,
        new_trigger_price=st_trigger_price,
        occurrence_count=st_occurrence_count,
    )
    @settings(max_examples=50)
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_changed_trigger_price_produces_new_row(
        self,
        symbol: str,
        alert_type: str,
        trigger_price: Decimal,
        new_trigger_price: Decimal,
        occurrence_count: int,
    ):
        """After trigger_price change >0.5% → a new audit row appears."""
        assume(trigger_price != new_trigger_price)
        # Requirement 8.3: price change must exceed 0.5% to be material
        assume(trigger_price > 0)
        pct_change = abs(new_trigger_price - trigger_price) / trigger_price
        assume(pct_change > Decimal("0.005"))

        engine, store, dispatcher = _create_store_and_dispatcher()

        dedupe_key = f"{symbol}:{alert_type}:{uuid.uuid4().hex[:16]}"
        intent = _insert_intent(
            store,
            symbol=symbol,
            alert_type=alert_type,
            trigger_price=trigger_price,
            occurrence_count=occurrence_count,
            dedupe_key=dedupe_key,
        )

        now = _BASE_TIME + timedelta(minutes=1)

        # First observation
        dispatcher._handle_observe(intent, now)
        assert _count_would_dispatch_rows(engine, intent.alert_intent_id) == 1

        # Simulate trigger_price change by updating DB and re-reading intent
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE alert_intents SET trigger_price = :tp WHERE id = :id"),
                {"tp": str(new_trigger_price), "id": intent.id},
            )
            row = conn.execute(
                text("SELECT * FROM alert_intents WHERE id = :id"),
                {"id": intent.id},
            ).fetchone()
        changed_intent = store._row_to_intent(row)

        # Second observation with changed trigger_price
        dispatcher._handle_observe(changed_intent, now)
        count = _count_would_dispatch_rows(engine, changed_intent.alert_intent_id)
        assert count == 2, (
            f"Expected 2 would_dispatch rows after trigger_price change "
            f"({trigger_price} → {new_trigger_price}), got {count}"
        )

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
        trigger_price=st_trigger_price,
        occurrence_count=st_occurrence_count,
    )
    @settings(max_examples=50)
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_incremented_occurrence_count_produces_new_row(
        self,
        symbol: str,
        alert_type: str,
        trigger_price: Decimal,
        occurrence_count: int,
    ):
        """After occurrence_count increment → a new audit row appears."""
        engine, store, dispatcher = _create_store_and_dispatcher()

        dedupe_key = f"{symbol}:{alert_type}:{uuid.uuid4().hex[:16]}"
        intent = _insert_intent(
            store,
            symbol=symbol,
            alert_type=alert_type,
            trigger_price=trigger_price,
            occurrence_count=occurrence_count,
            dedupe_key=dedupe_key,
        )

        now = _BASE_TIME + timedelta(minutes=1)

        # First observation
        dispatcher._handle_observe(intent, now)
        assert _count_would_dispatch_rows(engine, intent.alert_intent_id) == 1

        # Simulate occurrence_count increment by updating DB and re-reading intent
        new_occurrence_count = occurrence_count + 1
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE alert_intents SET occurrence_count = :oc WHERE id = :id"),
                {"oc": new_occurrence_count, "id": intent.id},
            )
            row = conn.execute(
                text("SELECT * FROM alert_intents WHERE id = :id"),
                {"id": intent.id},
            ).fetchone()
        changed_intent = store._row_to_intent(row)

        # Second observation with incremented occurrence_count
        dispatcher._handle_observe(changed_intent, now)
        count = _count_would_dispatch_rows(engine, changed_intent.alert_intent_id)
        assert count == 2, (
            f"Expected 2 would_dispatch rows after occurrence_count increment "
            f"({occurrence_count} → {new_occurrence_count}), got {count}"
        )

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
        trigger_price=st_trigger_price,
        occurrence_count=st_occurrence_count,
    )
    @settings(max_examples=50)
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_has_would_dispatch_blocks_duplicates(
        self,
        symbol: str,
        alert_type: str,
        trigger_price: Decimal,
        occurrence_count: int,
    ):
        """has_would_dispatch_for_occurrence returns True after first observe, blocking duplicates."""
        engine, store, dispatcher = _create_store_and_dispatcher()

        dedupe_key = f"{symbol}:{alert_type}:{uuid.uuid4().hex[:16]}"
        intent = _insert_intent(
            store,
            symbol=symbol,
            alert_type=alert_type,
            trigger_price=trigger_price,
            occurrence_count=occurrence_count,
            dedupe_key=dedupe_key,
        )

        now = _BASE_TIME + timedelta(minutes=1)

        # Before any observation — should return False
        assert not store.has_would_dispatch_for_occurrence(
            alert_intent_id=intent.alert_intent_id,
            dedupe_key=intent.dedupe_key,
            trigger_price=intent.trigger_price,
            occurrence_count=intent.occurrence_count,
        ), "has_would_dispatch should be False before any observation"

        # First observation
        dispatcher._handle_observe(intent, now)

        # After observation — should return True (blocking duplicates)
        assert store.has_would_dispatch_for_occurrence(
            alert_intent_id=intent.alert_intent_id,
            dedupe_key=intent.dedupe_key,
            trigger_price=intent.trigger_price,
            occurrence_count=intent.occurrence_count,
        ), "has_would_dispatch should be True after first observation"

        # But with different occurrence_count — should return False (new occurrence)
        assert not store.has_would_dispatch_for_occurrence(
            alert_intent_id=intent.alert_intent_id,
            dedupe_key=intent.dedupe_key,
            trigger_price=intent.trigger_price,
            occurrence_count=intent.occurrence_count + 1,
        ), "has_would_dispatch should be False for a new occurrence_count"
