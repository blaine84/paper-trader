"""
Property-based tests for dispatch-once deferral logic using Hypothesis.

Property 4: Dispatch-once deferral halts re-evaluation.
Validates that _is_deferred() correctly determines whether an AlertIntent
should be skipped (deferred) or re-evaluated based on deferred_until timing
and occurrence_count changes.

**Validates: Requirements 2.1, 2.5**
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from utils.alert_dispatcher import AlertDispatcher
from utils.alert_intent_store import AlertIntent


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Generate a fixed "now" datetime in a reasonable range
now_strategy = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31),
)

# Generate positive timedeltas for future deferral (1 second to 24 hours)
future_delta_strategy = st.timedeltas(
    min_value=timedelta(seconds=1),
    max_value=timedelta(hours=24),
)

# Generate positive timedeltas for past/expired deferral
past_delta_strategy = st.timedeltas(
    min_value=timedelta(seconds=0),
    max_value=timedelta(hours=24),
)

# Occurrence counts (non-negative integers)
occurrence_count_strategy = st.integers(min_value=0, max_value=1000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dispatcher() -> AlertDispatcher:
    """Create a minimal AlertDispatcher with mock dependencies."""
    engine = MagicMock()
    store = MagicMock()
    begin_pm_cycle = MagicMock(return_value=True)
    end_pm_cycle = MagicMock()
    return AlertDispatcher(engine, store, begin_pm_cycle, end_pm_cycle)


def _make_intent(
    deferred_until: datetime | None,
    occurrence_count: int,
    occurrence_count_at_deferral: int,
) -> AlertIntent:
    """Create an AlertIntent with controlled deferral fields."""
    return AlertIntent(
        id=1,
        alert_intent_id="test-intent-001",
        symbol="AAPL",
        alert_type="entry_alert",
        direction="long",
        trigger_price=Decimal("150.00"),
        source_level="support_bounce",
        urgency="high",
        reason="Test intent",
        dedupe_key="AAPL:entry_alert:abc123",
        filter_status="passed",
        first_seen_at=datetime(2024, 6, 1, 10, 0, 0),
        last_seen_at=datetime(2024, 6, 1, 10, 5, 0),
        occurrence_count=occurrence_count,
        expiration_at=datetime(2024, 6, 1, 12, 0, 0),
        dispatch_status="pending",
        dispatch_reason=None,
        dispatched_at=None,
        deferred_until=deferred_until,
        occurrence_count_at_deferral=occurrence_count_at_deferral,
        dispatch_attempt_count=0,
        last_dispatch_error=None,
    )


# ---------------------------------------------------------------------------
# Property 4: Dispatch-once deferral halts re-evaluation
# **Validates: Requirements 2.1, 2.5**
# ---------------------------------------------------------------------------


class TestProperty4DispatchOnceDeferralHaltsReEvaluation:
    """
    Property 4: Dispatch-once deferral halts re-evaluation.

    The _is_deferred() method returns True (halting re-evaluation) only when:
    - deferred_until is set AND in the future (> now)
    - AND occurrence_count has not changed since deferral was set

    It returns False (allowing re-evaluation) when:
    - deferred_until is None (never deferred)
    - deferred_until <= now (deferral expired)
    - occurrence_count > occurrence_count_at_deferral (material change)

    **Validates: Requirements 2.1, 2.5**
    """

    @given(
        now=now_strategy,
        future_delta=future_delta_strategy,
        occurrence_count=occurrence_count_strategy,
    )
    @settings(max_examples=200)
    def test_deferred_until_future_and_unchanged_occurrence_returns_true(
        self, now: datetime, future_delta: timedelta, occurrence_count: int
    ):
        """_is_deferred() returns True when deferred_until > now AND occurrence_count unchanged."""
        deferred_until = now + future_delta
        dispatcher = _make_dispatcher()
        intent = _make_intent(
            deferred_until=deferred_until,
            occurrence_count=occurrence_count,
            occurrence_count_at_deferral=occurrence_count,  # unchanged
        )

        result = dispatcher._is_deferred(intent, now)

        assert result is True, (
            f"Expected True (deferred): deferred_until={deferred_until} > now={now}, "
            f"occurrence_count={occurrence_count} == occurrence_count_at_deferral={occurrence_count}"
        )

    @given(
        now=now_strategy,
        past_delta=past_delta_strategy,
        occurrence_count=occurrence_count_strategy,
    )
    @settings(max_examples=200)
    def test_deferred_until_expired_returns_false(
        self, now: datetime, past_delta: timedelta, occurrence_count: int
    ):
        """_is_deferred() returns False when deferred_until <= now (expired deferral)."""
        deferred_until = now - past_delta  # In the past or exactly now
        dispatcher = _make_dispatcher()
        intent = _make_intent(
            deferred_until=deferred_until,
            occurrence_count=occurrence_count,
            occurrence_count_at_deferral=occurrence_count,
        )

        result = dispatcher._is_deferred(intent, now)

        assert result is False, (
            f"Expected False (deferral expired): deferred_until={deferred_until} <= now={now}"
        )

    @given(
        now=now_strategy,
        future_delta=future_delta_strategy,
        occurrence_count_at_deferral=occurrence_count_strategy,
        additional_occurrences=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=200)
    def test_material_change_overrides_active_deferral(
        self,
        now: datetime,
        future_delta: timedelta,
        occurrence_count_at_deferral: int,
        additional_occurrences: int,
    ):
        """_is_deferred() returns False when occurrence_count > occurrence_count_at_deferral (material change)."""
        deferred_until = now + future_delta  # Still in the future
        current_occurrence = occurrence_count_at_deferral + additional_occurrences
        dispatcher = _make_dispatcher()
        intent = _make_intent(
            deferred_until=deferred_until,
            occurrence_count=current_occurrence,
            occurrence_count_at_deferral=occurrence_count_at_deferral,
        )

        result = dispatcher._is_deferred(intent, now)

        assert result is False, (
            f"Expected False (material change): occurrence_count={current_occurrence} > "
            f"occurrence_count_at_deferral={occurrence_count_at_deferral}, "
            f"even though deferred_until={deferred_until} > now={now}"
        )

    @given(
        now=now_strategy,
        occurrence_count=occurrence_count_strategy,
        occurrence_count_at_deferral=occurrence_count_strategy,
    )
    @settings(max_examples=200)
    def test_deferred_until_none_returns_false(
        self, now: datetime, occurrence_count: int, occurrence_count_at_deferral: int
    ):
        """_is_deferred() returns False when deferred_until is None (not deferred)."""
        dispatcher = _make_dispatcher()
        intent = _make_intent(
            deferred_until=None,
            occurrence_count=occurrence_count,
            occurrence_count_at_deferral=occurrence_count_at_deferral,
        )

        result = dispatcher._is_deferred(intent, now)

        assert result is False, (
            f"Expected False (not deferred): deferred_until=None, "
            f"regardless of occurrence_count={occurrence_count} vs "
            f"occurrence_count_at_deferral={occurrence_count_at_deferral}"
        )
