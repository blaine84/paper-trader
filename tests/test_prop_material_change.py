"""
Property-based tests for material change detection using Hypothesis.

Property 12: Observation permits new record on material change.

Validates:
1. A trigger_price change >0.5% is detected as material (permits new observation)
2. A trigger_price change <=0.5% is NOT material (suppresses new observation)
3. A dedupe_key change is always material (permits new observation)
4. An occurrence_count increment is always material (permits new observation)

Tests the `_is_material_price_change()` static method directly, and also tests
the full `_handle_observe()` flow with a real DB to verify end-to-end behavior.

**Validates: Requirements 8.3, 8.4**
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

SYMBOLS = ["NVDA", "AAPL", "TSLA", "MSFT"]
ALERT_TYPES = ["entry_alert", "breakout", "rapid_move"]

st_symbol = st.sampled_from(SYMBOLS)
st_alert_type = st.sampled_from(ALERT_TYPES)

# Base prices for material change tests
st_base_price = st.decimals(min_value=Decimal("100.00"), max_value=Decimal("500.00"), places=2)

# Multiplier that produces >0.5% change (either up or down)
st_material_multiplier = st.one_of(
    st.decimals(min_value=Decimal("1.006"), max_value=Decimal("1.100"), places=3),
    st.decimals(min_value=Decimal("0.900"), max_value=Decimal("0.994"), places=3),
)

# Multiplier that produces <=0.5% change (strictly between 0.995 and 1.005, exclusive)
st_non_material_multiplier = st.decimals(
    min_value=Decimal("0.996"), max_value=Decimal("1.004"), places=3
)

st_occurrence_count = st.integers(min_value=1, max_value=50)

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


def _update_intent_field(store, intent_id: int, **kwargs) -> AlertIntent:
    """Update specific fields on an intent and return the refreshed AlertIntent."""
    set_parts = ", ".join(f"{k} = :{k}" for k in kwargs)
    params = dict(kwargs, id=intent_id)
    with store._engine.begin() as conn:
        conn.execute(
            text(f"UPDATE alert_intents SET {set_parts} WHERE id = :id"),
            params,
        )
        row = conn.execute(
            text("SELECT * FROM alert_intents WHERE id = :id"),
            {"id": intent_id},
        ).fetchone()
    return store._row_to_intent(row)


# ---------------------------------------------------------------------------
# Property 12: Observation permits new record on material change
# ---------------------------------------------------------------------------


class TestProperty12MaterialPriceChangeDetection:
    """
    Property 12 (pure function): _is_material_price_change() correctly identifies
    whether a trigger_price change exceeds the 0.5% threshold.

    **Validates: Requirements 8.3, 8.4**
    """

    @given(
        base_price=st_base_price,
        multiplier=st_material_multiplier,
    )
    @settings(max_examples=200)
    def test_price_change_above_threshold_is_material(
        self,
        base_price: Decimal,
        multiplier: Decimal,
    ):
        """A trigger_price change >0.5% is detected as material."""
        new_price = (base_price * multiplier).quantize(Decimal("0.01"))
        # Verify the change is actually >0.5%
        pct_change = abs((new_price - base_price) / base_price) * 100
        assume(pct_change > Decimal("0.5"))

        result = AlertDispatcher._is_material_price_change(new_price, base_price)
        assert result is True, (
            f"Expected material change for {base_price} → {new_price} "
            f"({pct_change:.4f}% change), but got False"
        )

    @given(
        base_price=st_base_price,
        multiplier=st_non_material_multiplier,
    )
    @settings(max_examples=200)
    def test_price_change_within_threshold_is_not_material(
        self,
        base_price: Decimal,
        multiplier: Decimal,
    ):
        """A trigger_price change <=0.5% is NOT material (suppresses new observation)."""
        new_price = (base_price * multiplier).quantize(Decimal("0.01"))
        # Verify the change is actually <=0.5%
        pct_change = abs((new_price - base_price) / base_price) * 100
        assume(pct_change <= Decimal("0.5"))
        # Also ensure we actually have a valid comparison (not same price)
        assume(new_price != base_price)

        result = AlertDispatcher._is_material_price_change(new_price, base_price)
        assert result is False, (
            f"Expected non-material change for {base_price} → {new_price} "
            f"({pct_change:.4f}% change), but got True"
        )

    @given(base_price=st_base_price)
    @settings(max_examples=200)
    def test_none_previous_price_is_material(self, base_price: Decimal):
        """A None previous_price is treated as material (fail-open)."""
        result = AlertDispatcher._is_material_price_change(base_price, None)
        assert result is True, "None previous_price should be material (fail-open)"

    @given(base_price=st_base_price)
    @settings(max_examples=200)
    def test_none_current_price_is_material(self, base_price: Decimal):
        """A None current_price is treated as material (fail-open)."""
        result = AlertDispatcher._is_material_price_change(None, base_price)
        assert result is True, "None current_price should be material (fail-open)"

    @given(new_price=st_base_price)
    @settings(max_examples=200)
    def test_zero_previous_price_is_material(self, new_price: Decimal):
        """A zero previous_price is treated as material (can't compute from zero)."""
        result = AlertDispatcher._is_material_price_change(new_price, Decimal("0"))
        assert result is True, "Zero previous_price should be material (can't compute %)"


class TestProperty12MaterialChangeIntegration:
    """
    Property 12 (integration): The full _handle_observe() flow correctly permits
    or suppresses new observation records based on material change detection.

    Tests with a real in-memory SQLite database to verify end-to-end behavior.

    **Validates: Requirements 8.3, 8.4**
    """

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
        base_price=st_base_price,
        multiplier=st_non_material_multiplier,
        occurrence_count=st_occurrence_count,
    )
    @settings(max_examples=50)
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_non_material_price_change_suppresses_observation(
        self,
        symbol: str,
        alert_type: str,
        base_price: Decimal,
        multiplier: Decimal,
        occurrence_count: int,
    ):
        """After first observation, a price change <=0.5% does NOT produce a new row."""
        new_price = (base_price * multiplier).quantize(Decimal("0.01"))
        pct_change = abs((new_price - base_price) / base_price) * 100
        assume(pct_change <= Decimal("0.5"))
        assume(new_price != base_price)

        engine, store, dispatcher = _create_store_and_dispatcher()

        dedupe_key = f"{symbol}:{alert_type}:{uuid.uuid4().hex[:16]}"
        intent = _insert_intent(
            store,
            symbol=symbol,
            alert_type=alert_type,
            trigger_price=base_price,
            occurrence_count=occurrence_count,
            dedupe_key=dedupe_key,
        )

        now = _BASE_TIME + timedelta(minutes=1)

        # First observation — should produce 1 row
        dispatcher._handle_observe(intent, now)
        assert _count_would_dispatch_rows(engine, intent.alert_intent_id) == 1

        # Update trigger_price to a non-material change
        changed_intent = _update_intent_field(
            store, intent.id, trigger_price=str(new_price)
        )

        # Second observation with non-material price change — still 1 row (suppressed)
        dispatcher._handle_observe(changed_intent, now)
        count = _count_would_dispatch_rows(engine, changed_intent.alert_intent_id)
        assert count == 1, (
            f"Expected 1 row (suppressed non-material change "
            f"{base_price} → {new_price}, {pct_change:.4f}%), got {count}"
        )

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
        base_price=st_base_price,
        multiplier=st_material_multiplier,
        occurrence_count=st_occurrence_count,
    )
    @settings(max_examples=50)
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_material_price_change_permits_new_observation(
        self,
        symbol: str,
        alert_type: str,
        base_price: Decimal,
        multiplier: Decimal,
        occurrence_count: int,
    ):
        """After first observation, a price change >0.5% produces a new row."""
        new_price = (base_price * multiplier).quantize(Decimal("0.01"))
        pct_change = abs((new_price - base_price) / base_price) * 100
        assume(pct_change > Decimal("0.5"))

        engine, store, dispatcher = _create_store_and_dispatcher()

        dedupe_key = f"{symbol}:{alert_type}:{uuid.uuid4().hex[:16]}"
        intent = _insert_intent(
            store,
            symbol=symbol,
            alert_type=alert_type,
            trigger_price=base_price,
            occurrence_count=occurrence_count,
            dedupe_key=dedupe_key,
        )

        now = _BASE_TIME + timedelta(minutes=1)

        # First observation — should produce 1 row
        dispatcher._handle_observe(intent, now)
        assert _count_would_dispatch_rows(engine, intent.alert_intent_id) == 1

        # Update trigger_price to a material change
        changed_intent = _update_intent_field(
            store, intent.id, trigger_price=str(new_price)
        )

        # Second observation with material price change — 2 rows (new observation)
        dispatcher._handle_observe(changed_intent, now)
        count = _count_would_dispatch_rows(engine, changed_intent.alert_intent_id)
        assert count == 2, (
            f"Expected 2 rows (material change "
            f"{base_price} → {new_price}, {pct_change:.4f}%), got {count}"
        )

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
        base_price=st_base_price,
        occurrence_count=st_occurrence_count,
    )
    @settings(max_examples=50)
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_dedupe_key_change_always_permits_new_observation(
        self,
        symbol: str,
        alert_type: str,
        base_price: Decimal,
        occurrence_count: int,
    ):
        """A dedupe_key change is always material (permits new observation)."""
        engine, store, dispatcher = _create_store_and_dispatcher()

        dedupe_key_1 = f"{symbol}:{alert_type}:{uuid.uuid4().hex[:16]}"
        intent = _insert_intent(
            store,
            symbol=symbol,
            alert_type=alert_type,
            trigger_price=base_price,
            occurrence_count=occurrence_count,
            dedupe_key=dedupe_key_1,
        )

        now = _BASE_TIME + timedelta(minutes=1)

        # First observation
        dispatcher._handle_observe(intent, now)
        assert _count_would_dispatch_rows(engine, intent.alert_intent_id) == 1

        # Change dedupe_key (new setup_condition) — same price and occurrence_count
        dedupe_key_2 = f"{symbol}:{alert_type}:{uuid.uuid4().hex[:16]}"
        changed_intent = _update_intent_field(
            store, intent.id, dedupe_key=dedupe_key_2
        )

        # Second observation with different dedupe_key — new row permitted
        dispatcher._handle_observe(changed_intent, now)
        count = _count_would_dispatch_rows(engine, changed_intent.alert_intent_id)
        assert count == 2, (
            f"Expected 2 rows (dedupe_key change "
            f"'{dedupe_key_1}' → '{dedupe_key_2}'), got {count}"
        )

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
        base_price=st_base_price,
        occurrence_count=st_occurrence_count,
    )
    @settings(max_examples=50)
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    def test_occurrence_count_increment_always_permits_new_observation(
        self,
        symbol: str,
        alert_type: str,
        base_price: Decimal,
        occurrence_count: int,
    ):
        """An occurrence_count increment is always material (permits new observation)."""
        engine, store, dispatcher = _create_store_and_dispatcher()

        dedupe_key = f"{symbol}:{alert_type}:{uuid.uuid4().hex[:16]}"
        intent = _insert_intent(
            store,
            symbol=symbol,
            alert_type=alert_type,
            trigger_price=base_price,
            occurrence_count=occurrence_count,
            dedupe_key=dedupe_key,
        )

        now = _BASE_TIME + timedelta(minutes=1)

        # First observation
        dispatcher._handle_observe(intent, now)
        assert _count_would_dispatch_rows(engine, intent.alert_intent_id) == 1

        # Increment occurrence_count — same price and dedupe_key
        new_oc = occurrence_count + 1
        changed_intent = _update_intent_field(
            store, intent.id, occurrence_count=new_oc
        )

        # Second observation with incremented occurrence_count — new row permitted
        dispatcher._handle_observe(changed_intent, now)
        count = _count_would_dispatch_rows(engine, changed_intent.alert_intent_id)
        assert count == 2, (
            f"Expected 2 rows (occurrence_count increment "
            f"{occurrence_count} → {new_oc}), got {count}"
        )
