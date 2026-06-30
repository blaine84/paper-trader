"""
Property-based tests for complete audit record on non-disabled evaluation.

Property 10: Complete audit record on every non-disabled evaluation.

Validates that every intent that enters evaluation with effective mode != "disabled"
produces at least one audit record in alert_dispatch_log, and that each audit row
contains the required fields (alert_intent_id, symbol, alert_type, dispatch_status,
reason). Also validates that disabled intents produce zero audit rows.

Uses a real in-memory SQLite database.

**Validates: Requirements 7.1, 7.2, 7.3, 8.1**
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

SYMBOLS = ["NVDA", "AAPL", "TSLA", "MSFT", "AMZN"]
# Only dispatchable types (target_hit excluded from dispatch)
ALERT_TYPES = ["entry_alert", "breakout", "rapid_move"]
URGENCIES = ["high", "medium", "low"]
# Modes that should produce audit records (non-disabled)
NON_DISABLED_MODES = ["observe", "dispatch"]

st_symbol = st.sampled_from(SYMBOLS)
st_alert_type = st.sampled_from(ALERT_TYPES)
st_urgency = st.sampled_from(URGENCIES)
st_non_disabled_mode = st.sampled_from(NON_DISABLED_MODES)
st_trigger_price = st.decimals(min_value=Decimal("1.00"), max_value=Decimal("9999.99"), places=2)

_BASE_TIME = datetime(2025, 6, 25, 10, 30, 0)  # Wed 10:30 AM (within market hours)
_ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


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
                   urgency: str = "medium", fresh: bool = True) -> AlertIntent:
    """Insert a test intent with passed filter and pending status.

    If fresh=True, sets last_seen_at to 1 minute ago (within freshness limits).
    If fresh=False, sets last_seen_at to 60 minutes ago (beyond any freshness limit).
    """
    now = _BASE_TIME
    last_seen_offset = timedelta(minutes=1) if fresh else timedelta(minutes=60)
    dedupe_key = f"{symbol}:{alert_type}:{uuid.uuid4().hex[:16]}"

    data = {
        "symbol": symbol,
        "alert_type": alert_type,
        "direction": "long",
        "trigger_price": str(trigger_price),
        "source_level": None,
        "urgency": urgency,
        "reason": None,
        "dedupe_key": dedupe_key,
        "filter_status": "passed",
        "first_seen_at": (now - timedelta(minutes=5)).strftime(_ISO_FMT),
        "last_seen_at": (now - last_seen_offset).strftime(_ISO_FMT),
        "expiration_at": (now + timedelta(hours=2)).strftime(_ISO_FMT),
    }
    return store.record_or_update_intent(data)


def _count_audit_rows(engine, alert_intent_id: str) -> int:
    """Count audit rows in alert_dispatch_log for a given intent."""
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT COUNT(*) FROM alert_dispatch_log
            WHERE alert_intent_id = :aid
        """), {"aid": alert_intent_id}).fetchone()
    return row[0]


def _get_audit_rows(engine, alert_intent_id: str) -> list[dict]:
    """Get all audit rows for an intent with required fields."""
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT alert_intent_id, symbol, alert_type, dispatch_status, reason
            FROM alert_dispatch_log
            WHERE alert_intent_id = :aid
        """), {"aid": alert_intent_id}).fetchall()
    return [
        {
            "alert_intent_id": r[0],
            "symbol": r[1],
            "alert_type": r[2],
            "dispatch_status": r[3],
            "reason": r[4],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Property 10: Complete audit record on every non-disabled evaluation
# ---------------------------------------------------------------------------


class TestProperty10AuditCompleteness:
    """
    Property 10: Complete audit record on every non-disabled evaluation.

    Validates:
    - Every non-disabled intent that enters evaluation produces >= 1 audit record
    - Each audit record has non-null required fields
    - Disabled intents produce zero audit rows

    **Validates: Requirements 7.1, 7.2, 7.3, 8.1**
    """

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
        trigger_price=st_trigger_price,
        urgency=st_urgency,
        mode=st_non_disabled_mode,
    )
    @settings(max_examples=30, deadline=None)
    @patch("utils.alert_dispatcher.datetime")
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    @patch("utils.gate_config.PM_ALERT_DISPATCH_STALE_MINUTES", 60)
    @patch("utils.gate_config.PM_ALERT_SCHEDULED_MAX_RUNTIME_MINUTES", 30)
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_ENTRY_ALERT_MINUTES", 15)
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_BREAKOUT_MINUTES", 10)
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_RAPID_MOVE_MINUTES", 5)
    @patch("utils.gate_config.PM_ALERT_GLOBAL_COOLDOWN_MINUTES", 5)
    def test_non_disabled_fresh_intent_produces_audit_record(
        self,
        mock_datetime,
        symbol: str,
        alert_type: str,
        trigger_price: Decimal,
        urgency: str,
        mode: str,
    ):
        """Every fresh non-disabled intent evaluated produces at least one audit record."""
        engine, store, dispatcher = _create_store_and_dispatcher()

        # Insert a fresh intent
        intent = _insert_intent(
            store,
            symbol=symbol,
            alert_type=alert_type,
            trigger_price=trigger_price,
            urgency=urgency,
            fresh=True,
        )

        # Mock datetime.utcnow() to return a time within market hours
        mock_datetime.utcnow.return_value = _BASE_TIME

        # Build per-alert mode config: set the tested alert_type to the given mode
        mode_env = {at: mode for at in ALERT_TYPES}

        with patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", mode), \
             patch("utils.gate_config.PM_ALERT_MODE_ENTRY_ALERT", mode_env.get("entry_alert", "")), \
             patch("utils.gate_config.PM_ALERT_MODE_BREAKOUT", mode_env.get("breakout", "")), \
             patch("utils.gate_config.PM_ALERT_MODE_RAPID_MOVE", mode_env.get("rapid_move", "")), \
             patch("utils.gate_config.PM_ALERT_MODE_TARGET_HIT", "disabled"):
            dispatcher.evaluate_and_dispatch()

        # Verify: at least one audit record exists
        count = _count_audit_rows(engine, intent.alert_intent_id)
        assert count >= 1, (
            f"Expected >= 1 audit record for non-disabled intent "
            f"(symbol={symbol}, alert_type={alert_type}, mode={mode}), got {count}"
        )

        # Verify: each audit row has required non-null fields
        rows = _get_audit_rows(engine, intent.alert_intent_id)
        for row in rows:
            assert row["alert_intent_id"] is not None, "audit row missing alert_intent_id"
            assert row["symbol"] is not None, "audit row missing symbol"
            assert row["alert_type"] is not None, "audit row missing alert_type"
            assert row["dispatch_status"] is not None, "audit row missing dispatch_status"

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
        trigger_price=st_trigger_price,
        urgency=st_urgency,
    )
    @settings(max_examples=30, deadline=None)
    @patch("utils.alert_dispatcher.datetime")
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    @patch("utils.gate_config.PM_ALERT_DISPATCH_STALE_MINUTES", 60)
    @patch("utils.gate_config.PM_ALERT_SCHEDULED_MAX_RUNTIME_MINUTES", 30)
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_ENTRY_ALERT_MINUTES", 15)
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_BREAKOUT_MINUTES", 10)
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_RAPID_MOVE_MINUTES", 5)
    @patch("utils.gate_config.PM_ALERT_GLOBAL_COOLDOWN_MINUTES", 5)
    def test_stale_non_disabled_intent_produces_expired_audit_record(
        self,
        mock_datetime,
        symbol: str,
        alert_type: str,
        trigger_price: Decimal,
        urgency: str,
    ):
        """A stale non-disabled intent produces an expired audit record (freshness enforcement)."""
        engine, store, dispatcher = _create_store_and_dispatcher()

        # Insert a STALE intent (last_seen_at = 60 min ago, beyond all freshness limits)
        intent = _insert_intent(
            store,
            symbol=symbol,
            alert_type=alert_type,
            trigger_price=trigger_price,
            urgency=urgency,
            fresh=False,
        )

        # Mock datetime.utcnow() to return a time within market hours
        mock_datetime.utcnow.return_value = _BASE_TIME

        # Use observe mode (non-disabled) so the intent enters evaluation
        with patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "observe"), \
             patch("utils.gate_config.PM_ALERT_MODE_ENTRY_ALERT", "observe"), \
             patch("utils.gate_config.PM_ALERT_MODE_BREAKOUT", "observe"), \
             patch("utils.gate_config.PM_ALERT_MODE_RAPID_MOVE", "observe"), \
             patch("utils.gate_config.PM_ALERT_MODE_TARGET_HIT", "disabled"):
            dispatcher.evaluate_and_dispatch()

        # Verify: at least one audit record exists (from freshness expiry)
        count = _count_audit_rows(engine, intent.alert_intent_id)
        assert count >= 1, (
            f"Expected >= 1 audit record for stale non-disabled intent "
            f"(symbol={symbol}, alert_type={alert_type}), got {count}"
        )

        # Verify: the audit record has dispatch_status="expired"
        rows = _get_audit_rows(engine, intent.alert_intent_id)
        expired_rows = [r for r in rows if r["dispatch_status"] == "expired"]
        assert len(expired_rows) >= 1, (
            f"Expected at least one 'expired' audit record for stale intent, "
            f"got statuses: {[r['dispatch_status'] for r in rows]}"
        )

        # Verify required fields on expired row
        for row in expired_rows:
            assert row["alert_intent_id"] is not None
            assert row["symbol"] is not None
            assert row["alert_type"] is not None
            assert row["dispatch_status"] == "expired"
            assert row["reason"] in ("freshness_expired", "undetermined_freshness")

    @given(
        symbol=st_symbol,
        alert_type=st_alert_type,
        trigger_price=st_trigger_price,
        urgency=st_urgency,
    )
    @settings(max_examples=30, deadline=None)
    @patch("utils.alert_dispatcher.datetime")
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    @patch("utils.gate_config.PM_ALERT_DISPATCH_STALE_MINUTES", 60)
    @patch("utils.gate_config.PM_ALERT_SCHEDULED_MAX_RUNTIME_MINUTES", 30)
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_ENTRY_ALERT_MINUTES", 15)
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_BREAKOUT_MINUTES", 10)
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_RAPID_MOVE_MINUTES", 5)
    @patch("utils.gate_config.PM_ALERT_GLOBAL_COOLDOWN_MINUTES", 5)
    def test_disabled_intent_produces_zero_audit_rows(
        self,
        mock_datetime,
        symbol: str,
        alert_type: str,
        trigger_price: Decimal,
        urgency: str,
    ):
        """Disabled intents produce zero audit rows (only DEBUG log)."""
        engine, store, dispatcher = _create_store_and_dispatcher()

        # Insert a fresh intent
        intent = _insert_intent(
            store,
            symbol=symbol,
            alert_type=alert_type,
            trigger_price=trigger_price,
            urgency=urgency,
            fresh=True,
        )

        # Mock datetime.utcnow() to return a time within market hours
        mock_datetime.utcnow.return_value = _BASE_TIME

        # Set ALL modes to disabled (global disabled → short-circuits, so use
        # per-alert disabled with global=observe to exercise per-intent routing)
        with patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "observe"), \
             patch("utils.gate_config.PM_ALERT_MODE_ENTRY_ALERT", "disabled"), \
             patch("utils.gate_config.PM_ALERT_MODE_BREAKOUT", "disabled"), \
             patch("utils.gate_config.PM_ALERT_MODE_RAPID_MOVE", "disabled"), \
             patch("utils.gate_config.PM_ALERT_MODE_TARGET_HIT", "disabled"):
            dispatcher.evaluate_and_dispatch()

        # Verify: zero audit rows for this disabled intent
        count = _count_audit_rows(engine, intent.alert_intent_id)
        assert count == 0, (
            f"Expected 0 audit rows for disabled intent "
            f"(symbol={symbol}, alert_type={alert_type}), got {count}"
        )

    @given(
        data=st.data(),
        num_intents=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=30, deadline=None)
    @patch("utils.alert_dispatcher.datetime")
    @patch("utils.gate_config.PM_ALERT_SYMBOL_COOLDOWN_MINUTES", 15)
    @patch("utils.gate_config.PM_ALERT_DISPATCH_STALE_MINUTES", 60)
    @patch("utils.gate_config.PM_ALERT_SCHEDULED_MAX_RUNTIME_MINUTES", 30)
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_ENTRY_ALERT_MINUTES", 15)
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_BREAKOUT_MINUTES", 10)
    @patch("utils.gate_config.PM_ALERT_FRESHNESS_RAPID_MOVE_MINUTES", 5)
    @patch("utils.gate_config.PM_ALERT_GLOBAL_COOLDOWN_MINUTES", 5)
    def test_multiple_non_disabled_intents_all_produce_audit_records(
        self,
        mock_datetime,
        data,
        num_intents: int,
    ):
        """Multiple non-disabled intents in a single evaluation all produce audit records."""
        engine, store, dispatcher = _create_store_and_dispatcher()

        # Generate multiple intents with different alert_types and symbols
        intents = []
        for _ in range(num_intents):
            symbol = data.draw(st_symbol)
            alert_type = data.draw(st_alert_type)
            trigger_price = data.draw(st_trigger_price)
            urgency = data.draw(st_urgency)
            intent = _insert_intent(
                store,
                symbol=symbol,
                alert_type=alert_type,
                trigger_price=trigger_price,
                urgency=urgency,
                fresh=True,
            )
            intents.append(intent)

        # Mock datetime.utcnow() to return a time within market hours
        mock_datetime.utcnow.return_value = _BASE_TIME

        # Use observe mode for all types (non-disabled, will produce would_dispatch records)
        with patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "observe"), \
             patch("utils.gate_config.PM_ALERT_MODE_ENTRY_ALERT", "observe"), \
             patch("utils.gate_config.PM_ALERT_MODE_BREAKOUT", "observe"), \
             patch("utils.gate_config.PM_ALERT_MODE_RAPID_MOVE", "observe"), \
             patch("utils.gate_config.PM_ALERT_MODE_TARGET_HIT", "disabled"):
            dispatcher.evaluate_and_dispatch()

        # Verify: each non-disabled intent has at least one audit record
        for intent in intents:
            count = _count_audit_rows(engine, intent.alert_intent_id)
            assert count >= 1, (
                f"Expected >= 1 audit record for intent "
                f"(alert_intent_id={intent.alert_intent_id}, "
                f"symbol={intent.symbol}, alert_type={intent.alert_type}), got {count}"
            )

            # Verify required fields are non-null
            rows = _get_audit_rows(engine, intent.alert_intent_id)
            for row in rows:
                assert row["alert_intent_id"] is not None, "audit row missing alert_intent_id"
                assert row["symbol"] is not None, "audit row missing symbol"
                assert row["alert_type"] is not None, "audit row missing alert_type"
                assert row["dispatch_status"] is not None, "audit row missing dispatch_status"
