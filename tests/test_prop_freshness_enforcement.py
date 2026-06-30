"""
Property-based tests for freshness enforcement using Hypothesis.

Property 6: Freshness enforcement expires stale intents.

Validates that _check_freshness() correctly determines whether an alert intent
is fresh enough for dispatch consideration based on its alert_type, last_seen_at,
and the configured freshness limits.

**Validates: Requirements 3.1, 3.3, 3.5**
"""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from utils.alert_dispatcher import AlertDispatcher
from utils.alert_intent_store import AlertIntent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Default freshness limits (minutes) per alert type
_DEFAULT_FRESHNESS_LIMITS = {
    "entry_alert": 15,
    "breakout": 10,
    "rapid_move": 5,
}

# All dispatchable alert types (target_hit excluded)
_DISPATCHABLE_TYPES = ["entry_alert", "breakout", "rapid_move"]

# A fixed reference "now" for deterministic testing
_FIXED_NOW = datetime(2025, 6, 25, 14, 0, 0)

# Patch targets for freshness config
_FRESHNESS_PATCHES = {
    "entry_alert": "utils.gate_config.PM_ALERT_FRESHNESS_ENTRY_ALERT_MINUTES",
    "breakout": "utils.gate_config.PM_ALERT_FRESHNESS_BREAKOUT_MINUTES",
    "rapid_move": "utils.gate_config.PM_ALERT_FRESHNESS_RAPID_MOVE_MINUTES",
}


def _make_dispatcher() -> AlertDispatcher:
    """Create a minimal AlertDispatcher with mock dependencies."""
    engine = MagicMock()
    store = MagicMock()
    begin_pm = MagicMock(return_value=True)
    end_pm = MagicMock()
    return AlertDispatcher(engine, store, begin_pm, end_pm)


def _make_intent(
    alert_type: str,
    last_seen_at=None,
    symbol: str = "AAPL",
) -> AlertIntent:
    """Create an AlertIntent with controlled fields for freshness testing."""
    now = _FIXED_NOW
    return AlertIntent(
        id=1,
        alert_intent_id="test-uuid-1234",
        symbol=symbol,
        alert_type=alert_type,
        direction="long",
        trigger_price=Decimal("150.00"),
        source_level=None,
        urgency="high",
        reason=None,
        dedupe_key=f"{symbol}:{alert_type}:abc123",
        filter_status="passed",
        first_seen_at=now - timedelta(minutes=30),
        last_seen_at=last_seen_at,
        occurrence_count=1,
        expiration_at=now + timedelta(hours=1),
        dispatch_status="pending",
        dispatch_reason=None,
        dispatched_at=None,
        deferred_until=None,
        occurrence_count_at_deferral=0,
        dispatch_attempt_count=0,
        last_dispatch_error=None,
    )


def _patch_freshness_defaults():
    """Context manager that patches all freshness limits to their defaults."""
    return _patch_freshness(15, 10, 5)


def _patch_freshness(entry_alert: int, breakout: int, rapid_move: int):
    """Context manager that patches freshness limits to specified values."""
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(patch(_FRESHNESS_PATCHES["entry_alert"], entry_alert))
    stack.enter_context(patch(_FRESHNESS_PATCHES["breakout"], breakout))
    stack.enter_context(patch(_FRESHNESS_PATCHES["rapid_move"], rapid_move))
    return stack


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Alert types that have freshness limits
dispatchable_type_strategy = st.sampled_from(_DISPATCHABLE_TYPES)

# Age in minutes as positive floats (up to 120 min)
age_minutes_strategy = st.floats(
    min_value=0.001, max_value=120.0, allow_nan=False, allow_infinity=False
)


# ---------------------------------------------------------------------------
# Property 6: Freshness enforcement expires stale intents
# ---------------------------------------------------------------------------


class TestProperty6FreshnessEnforcement:
    """
    Freshness enforcement correctly classifies intents as fresh or stale
    based on their alert_type and age relative to the configured freshness limit.

    **Validates: Requirements 3.1, 3.3, 3.5**
    """

    @given(
        alert_type=dispatchable_type_strategy,
        age_minutes=age_minutes_strategy,
    )
    @settings(max_examples=200)
    def test_stale_intents_return_false(self, alert_type: str, age_minutes: float):
        """Intents older than their configured freshness limit return False (stale)."""
        limit = _DEFAULT_FRESHNESS_LIMITS[alert_type]
        assume(age_minutes >= limit)

        dispatcher = _make_dispatcher()
        now = _FIXED_NOW
        last_seen_at = now - timedelta(minutes=age_minutes)
        intent = _make_intent(alert_type=alert_type, last_seen_at=last_seen_at)

        with _patch_freshness_defaults():
            result = dispatcher._check_freshness(intent, now)

        assert result is False, (
            f"Intent with alert_type='{alert_type}', age={age_minutes:.2f}min "
            f"(limit={limit}min) should be stale (False), got True"
        )

    @given(
        alert_type=dispatchable_type_strategy,
        age_minutes=age_minutes_strategy,
    )
    @settings(max_examples=200)
    def test_fresh_intents_return_true(self, alert_type: str, age_minutes: float):
        """Intents within their configured freshness limit return True (fresh)."""
        limit = _DEFAULT_FRESHNESS_LIMITS[alert_type]
        # Strict less-than: the boundary (age == limit) is stale
        assume(age_minutes < limit)

        dispatcher = _make_dispatcher()
        now = _FIXED_NOW
        last_seen_at = now - timedelta(minutes=age_minutes)
        intent = _make_intent(alert_type=alert_type, last_seen_at=last_seen_at)

        with _patch_freshness_defaults():
            result = dispatcher._check_freshness(intent, now)

        assert result is True, (
            f"Intent with alert_type='{alert_type}', age={age_minutes:.2f}min "
            f"(limit={limit}min) should be fresh (True), got False"
        )

    @given(
        alert_type=dispatchable_type_strategy,
    )
    @settings(max_examples=200)
    def test_none_last_seen_at_returns_false(self, alert_type: str):
        """Intents with None last_seen_at always return False (fail-closed)."""
        dispatcher = _make_dispatcher()
        now = _FIXED_NOW
        intent = _make_intent(alert_type=alert_type, last_seen_at=None)

        with _patch_freshness_defaults():
            result = dispatcher._check_freshness(intent, now)

        assert result is False, (
            f"Intent with alert_type='{alert_type}' and last_seen_at=None "
            f"should be stale (False, fail-closed), got True"
        )

    @given(
        age_minutes=age_minutes_strategy,
    )
    @settings(max_examples=200)
    def test_target_hit_always_returns_false(self, age_minutes: float):
        """target_hit intents always return False (excluded from dispatch)."""
        dispatcher = _make_dispatcher()
        now = _FIXED_NOW
        last_seen_at = now - timedelta(minutes=age_minutes)
        intent = _make_intent(alert_type="target_hit", last_seen_at=last_seen_at)

        with _patch_freshness_defaults():
            result = dispatcher._check_freshness(intent, now)

        assert result is False, (
            f"target_hit intent with age={age_minutes:.2f}min "
            f"should always be stale (False, excluded), got True"
        )

    @given(
        alert_type=dispatchable_type_strategy,
        freshness_limit=st.integers(min_value=1, max_value=120),
        age_minutes=age_minutes_strategy,
    )
    @settings(max_examples=200)
    def test_freshness_limit_varies_by_alert_type(
        self, alert_type: str, freshness_limit: int, age_minutes: float
    ):
        """Freshness check respects the configured limit for each alert type.

        For any configurable freshness limit, age < limit → fresh, age >= limit → stale.
        This validates that the limit is independently configurable per type.
        """
        dispatcher = _make_dispatcher()
        now = _FIXED_NOW
        last_seen_at = now - timedelta(minutes=age_minutes)
        intent = _make_intent(alert_type=alert_type, last_seen_at=last_seen_at)

        with patch(_FRESHNESS_PATCHES[alert_type], freshness_limit):
            result = dispatcher._check_freshness(intent, now)

        if age_minutes < freshness_limit:
            assert result is True, (
                f"alert_type='{alert_type}', age={age_minutes:.2f}min, "
                f"limit={freshness_limit}min → should be fresh (True), got False"
            )
        else:
            assert result is False, (
                f"alert_type='{alert_type}', age={age_minutes:.2f}min, "
                f"limit={freshness_limit}min → should be stale (False), got True"
            )
