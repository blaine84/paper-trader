"""
Property-based tests for Setup-Aware Lifecycle Evaluator core logic.

Tests Properties 8, 9, 10, 14 from the design document using Hypothesis.
Feature: setup-aware-exit-governance

**Validates: Requirements 3.1, 3.3, 3.4, 7.5**
"""

from datetime import datetime, time, timedelta, timezone

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from utils.setup_aware_evaluator import (
    REVALIDATION_DECISIONS,
    _evaluate_revalidation_decision,
    evaluate_setup_aware_lifecycle,
)
from utils.setup_time_policy import get_policy, SETUP_TIME_POLICY_REGISTRY


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

FIXED_NOW_UTC = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
# now_et well before EOD hard wall (15:45 ET) to avoid pre-wall buffer interference
FIXED_NOW_ET = datetime(2026, 5, 26, 12, 0, 0)

st_price = st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
st_direction = st.sampled_from(["LONG", "SHORT"])
st_extension_eligible_setup = st.sampled_from(["news_breakout", "news_catalyst", "trend_pullback"])

st_known_setup = st.sampled_from(list(SETUP_TIME_POLICY_REGISTRY.keys()))
st_any_setup = st.one_of(
    st_known_setup,
    st.text(min_size=0, max_size=20).filter(lambda s: s not in SETUP_TIME_POLICY_REGISTRY),
)

# Minutes held strategies for extension-eligible setups at revalidation boundaries
# news_breakout/news_catalyst: revalidate_minutes=90, max=180
# trend_pullback: revalidate_minutes=120, max=180
st_minutes_at_revalidation = st.floats(
    min_value=90.0, max_value=179.9, allow_nan=False, allow_infinity=False
)


def _make_trade_at_boundary(
    setup_type: str,
    direction: str,
    entry_price: float,
    stop_price: float | None,
    target_price: float | None,
    minutes_held: float,
) -> tuple[dict, datetime]:
    """Create a trade dict and entry_time such that minutes_held matches."""
    entry_time = FIXED_NOW_UTC - timedelta(minutes=minutes_held)
    trade = {
        "id": 1,
        "symbol": "AMD",
        "profile": "moderate",
        "direction": direction,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "setup_type": setup_type,
        "status": "open",
        "quantity": 100,
        "entry_time": entry_time.isoformat(),
    }
    return trade, entry_time


# ---------------------------------------------------------------------------
# Property 8: Revalidation produces exactly one of five decisions
# Feature: setup-aware-exit-governance, Property 8: Revalidation produces exactly one of five decisions
# ---------------------------------------------------------------------------


class TestProperty8RevalidationProducesExactlyOneOfFiveDecisions:
    """
    For any extension-eligible trade at a revalidation boundary with valid
    criteria and fresh market data, _evaluate_revalidation_decision returns
    exactly one of the 5 REVALIDATION_DECISIONS outcomes.

    **Validates: Requirements 3.1**
    """

    @given(
        setup_type=st_extension_eligible_setup,
        direction=st_direction,
        entry_price=st_price,
        stop_offset_pct=st.floats(min_value=0.01, max_value=0.10, allow_nan=False, allow_infinity=False),
        target_offset_pct=st.floats(min_value=0.01, max_value=0.30, allow_nan=False, allow_infinity=False),
        current_price=st_price,
        minutes_held=st.floats(min_value=90.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_revalidation_decision_is_one_of_five(
        self, setup_type, direction, entry_price, stop_offset_pct, target_offset_pct,
        current_price, minutes_held,
    ):
        """_evaluate_revalidation_decision always returns one of the 5 valid outcomes."""
        policy = get_policy(setup_type)

        # Build valid criteria with a stop on the correct side
        if direction == "LONG":
            stop_price = round(entry_price * (1 - stop_offset_pct), 2)
            target_price = round(entry_price * (1 + target_offset_pct), 2)
        else:
            stop_price = round(entry_price * (1 + stop_offset_pct), 2)
            target_price = round(entry_price * (1 - target_offset_pct), 2)

        assume(stop_price > 0)
        assume(target_price > 0)

        trade = {
            "id": 1,
            "symbol": "AMD",
            "direction": direction,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "setup_type": setup_type,
        }
        criteria = {"stop_price": stop_price}

        outcome, reason, metadata = _evaluate_revalidation_decision(
            trade, policy, current_price, minutes_held, criteria
        )

        assert outcome in REVALIDATION_DECISIONS, (
            f"Unexpected outcome '{outcome}' not in REVALIDATION_DECISIONS. "
            f"setup={setup_type}, direction={direction}, entry={entry_price}, "
            f"current={current_price}, stop={stop_price}, minutes={minutes_held}"
        )
        assert isinstance(reason, str) and len(reason) > 0
        assert isinstance(metadata, dict)


# ---------------------------------------------------------------------------
# Property 9: Revalidation determinism
# Feature: setup-aware-exit-governance, Property 9: Revalidation determinism
# ---------------------------------------------------------------------------


class TestProperty9RevalidationDeterminism:
    """
    Calling evaluate_setup_aware_lifecycle twice with identical inputs produces
    identical outputs.

    **Validates: Requirements 3.3**
    """

    @given(
        setup_type=st_any_setup,
        direction=st_direction,
        entry_price=st_price,
        stop_price=st.one_of(st_price, st.none()),
        target_price=st.one_of(st_price, st.none()),
        current_price=st.one_of(st_price, st.none()),
        minutes_held=st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
        has_market_ts=st.booleans(),
    )
    @settings(max_examples=200)
    def test_identical_inputs_produce_identical_outputs(
        self, setup_type, direction, entry_price, stop_price, target_price,
        current_price, minutes_held, has_market_ts,
    ):
        """Two calls with same inputs produce the same result dict."""
        entry_time = FIXED_NOW_UTC - timedelta(minutes=minutes_held)
        market_data_timestamp = (
            FIXED_NOW_UTC - timedelta(seconds=10) if has_market_ts else None
        )

        trade = {
            "id": 1,
            "symbol": "AMD",
            "profile": "moderate",
            "direction": direction,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "setup_type": setup_type,
            "status": "open",
            "quantity": 100,
            "entry_time": entry_time.isoformat(),
        }
        events: list[dict] = []

        result1 = evaluate_setup_aware_lifecycle(
            trade, events,
            now_utc=FIXED_NOW_UTC,
            now_et=FIXED_NOW_ET,
            current_price=current_price,
            market_data_timestamp=market_data_timestamp,
            shadow_mode=False,
        )

        result2 = evaluate_setup_aware_lifecycle(
            trade, events,
            now_utc=FIXED_NOW_UTC,
            now_et=FIXED_NOW_ET,
            current_price=current_price,
            market_data_timestamp=market_data_timestamp,
            shadow_mode=False,
        )

        assert result1 == result2, (
            f"Non-deterministic output for setup={setup_type}, direction={direction}, "
            f"entry={entry_price}, current={current_price}, minutes={minutes_held}. "
            f"Result1: {result1}\nResult2: {result2}"
        )


# ---------------------------------------------------------------------------
# Property 10: Missing data triggers fail-closed behavior
# Feature: setup-aware-exit-governance, Property 10: Missing data triggers fail-closed behavior
# ---------------------------------------------------------------------------


class TestProperty10MissingDataTriggersFailClosed:
    """
    When current_price is None OR market_data_timestamp is None, and the trade
    is at a revalidation boundary, the evaluator produces a close decision
    (fail-closed).

    **Validates: Requirements 3.4, 7.5**
    """

    @given(
        setup_type=st_extension_eligible_setup,
        direction=st_direction,
        entry_price=st_price,
        stop_offset_pct=st.floats(min_value=0.01, max_value=0.10, allow_nan=False, allow_infinity=False),
        minutes_held=st.floats(min_value=90.0, max_value=179.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_none_current_price_produces_close(
        self, setup_type, direction, entry_price, stop_offset_pct, minutes_held,
    ):
        """When current_price is None at revalidation boundary, evaluator closes."""
        policy = get_policy(setup_type)
        # Ensure we're at or past the revalidation boundary
        assume(minutes_held >= policy.revalidate_minutes)
        assume(policy.max_extension_minutes is None or minutes_held < policy.max_extension_minutes)

        if direction == "LONG":
            stop_price = round(entry_price * (1 - stop_offset_pct), 2)
        else:
            stop_price = round(entry_price * (1 + stop_offset_pct), 2)
        assume(stop_price > 0)

        entry_time = FIXED_NOW_UTC - timedelta(minutes=minutes_held)
        trade = {
            "id": 1,
            "symbol": "AMD",
            "profile": "moderate",
            "direction": direction,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": entry_price * 1.1 if direction == "LONG" else entry_price * 0.9,
            "setup_type": setup_type,
            "status": "open",
            "quantity": 100,
            "entry_time": entry_time.isoformat(),
        }

        result = evaluate_setup_aware_lifecycle(
            trade, [],
            now_utc=FIXED_NOW_UTC,
            now_et=FIXED_NOW_ET,
            current_price=None,  # Missing price
            market_data_timestamp=FIXED_NOW_UTC - timedelta(seconds=5),
            shadow_mode=False,
        )

        assert result["decision"] == "close", (
            f"Expected close decision when current_price=None, got '{result['decision']}'. "
            f"setup={setup_type}, minutes={minutes_held}, state={result.get('state')}"
        )

    @given(
        setup_type=st_extension_eligible_setup,
        direction=st_direction,
        entry_price=st_price,
        stop_offset_pct=st.floats(min_value=0.01, max_value=0.10, allow_nan=False, allow_infinity=False),
        minutes_held=st.floats(min_value=90.0, max_value=179.0, allow_nan=False, allow_infinity=False),
        current_price=st_price,
    )
    @settings(max_examples=200)
    def test_none_market_data_timestamp_produces_close(
        self, setup_type, direction, entry_price, stop_offset_pct, minutes_held,
        current_price,
    ):
        """When market_data_timestamp is None at revalidation boundary, evaluator closes."""
        policy = get_policy(setup_type)
        assume(minutes_held >= policy.revalidate_minutes)
        assume(policy.max_extension_minutes is None or minutes_held < policy.max_extension_minutes)

        if direction == "LONG":
            stop_price = round(entry_price * (1 - stop_offset_pct), 2)
        else:
            stop_price = round(entry_price * (1 + stop_offset_pct), 2)
        assume(stop_price > 0)

        entry_time = FIXED_NOW_UTC - timedelta(minutes=minutes_held)
        trade = {
            "id": 1,
            "symbol": "AMD",
            "profile": "moderate",
            "direction": direction,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": entry_price * 1.1 if direction == "LONG" else entry_price * 0.9,
            "setup_type": setup_type,
            "status": "open",
            "quantity": 100,
            "entry_time": entry_time.isoformat(),
        }

        result = evaluate_setup_aware_lifecycle(
            trade, [],
            now_utc=FIXED_NOW_UTC,
            now_et=FIXED_NOW_ET,
            current_price=current_price,
            market_data_timestamp=None,  # Missing timestamp
            shadow_mode=False,
        )

        assert result["decision"] == "close", (
            f"Expected close decision when market_data_timestamp=None, "
            f"got '{result['decision']}'. "
            f"setup={setup_type}, minutes={minutes_held}, state={result.get('state')}"
        )


# ---------------------------------------------------------------------------
# Property 14: Alert vs revalidation boundary decision exclusivity
# Feature: setup-aware-exit-governance, Property 14: Alert vs revalidation boundary decision exclusivity
# ---------------------------------------------------------------------------


class TestProperty14AlertVsRevalidationBoundaryExclusivity:
    """
    For any trade at alert_minutes (but before revalidate_minutes), the evaluator
    produces setup_exit_alert as an alert-only state. For any trade at an actual
    revalidation boundary (at or past revalidate_minutes), the evaluator produces
    a decisive outcome and does NOT produce setup_exit_alert.

    **Validates: Requirement 3 (alert-only vs decisive boundary semantics)**
    """

    @given(
        setup_type=st_extension_eligible_setup,
        direction=st_direction,
        entry_price=st_price,
        stop_offset_pct=st.floats(min_value=0.01, max_value=0.10, allow_nan=False, allow_infinity=False),
        alert_offset=st.floats(min_value=0.0, max_value=29.0, allow_nan=False, allow_infinity=False),
        current_price=st_price,
    )
    @settings(max_examples=200)
    def test_below_revalidation_boundary_produces_alert(
        self, setup_type, direction, entry_price, stop_offset_pct, alert_offset,
        current_price,
    ):
        """Trade past alert but before revalidation boundary → setup_exit_alert."""
        policy = get_policy(setup_type)
        # minutes_held is between alert_minutes and revalidate_minutes
        minutes_held = policy.alert_minutes + alert_offset
        assume(minutes_held < policy.revalidate_minutes)

        if direction == "LONG":
            stop_price = round(entry_price * (1 - stop_offset_pct), 2)
        else:
            stop_price = round(entry_price * (1 + stop_offset_pct), 2)
        assume(stop_price > 0)

        entry_time = FIXED_NOW_UTC - timedelta(minutes=minutes_held)
        trade = {
            "id": 1,
            "symbol": "AMD",
            "profile": "moderate",
            "direction": direction,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": entry_price * 1.1 if direction == "LONG" else entry_price * 0.9,
            "setup_type": setup_type,
            "status": "open",
            "quantity": 100,
            "entry_time": entry_time.isoformat(),
        }

        market_ts = FIXED_NOW_UTC - timedelta(seconds=5)
        result = evaluate_setup_aware_lifecycle(
            trade, [],
            now_utc=FIXED_NOW_UTC,
            now_et=FIXED_NOW_ET,
            current_price=current_price,
            market_data_timestamp=market_ts,
            shadow_mode=False,
        )

        assert result["state"] == "setup_exit_alert", (
            f"Expected setup_exit_alert for trade between alert and revalidation. "
            f"setup={setup_type}, minutes={minutes_held}, "
            f"alert={policy.alert_minutes}, revalidate={policy.revalidate_minutes}, "
            f"got state='{result['state']}'"
        )

    @given(
        setup_type=st_extension_eligible_setup,
        direction=st_direction,
        entry_price=st_price,
        stop_offset_pct=st.floats(min_value=0.01, max_value=0.10, allow_nan=False, allow_infinity=False),
        boundary_offset=st.floats(min_value=0.0, max_value=89.0, allow_nan=False, allow_infinity=False),
        current_price=st_price,
    )
    @settings(max_examples=200)
    def test_at_revalidation_boundary_produces_decisive_outcome(
        self, setup_type, direction, entry_price, stop_offset_pct, boundary_offset,
        current_price,
    ):
        """Trade at or past revalidation boundary → decisive outcome, NOT alert."""
        policy = get_policy(setup_type)
        # minutes_held is at or past revalidate_minutes (but below max_extension)
        minutes_held = policy.revalidate_minutes + boundary_offset
        assume(policy.max_extension_minutes is None or minutes_held < policy.max_extension_minutes)

        if direction == "LONG":
            stop_price = round(entry_price * (1 - stop_offset_pct), 2)
        else:
            stop_price = round(entry_price * (1 + stop_offset_pct), 2)
        assume(stop_price > 0)

        entry_time = FIXED_NOW_UTC - timedelta(minutes=minutes_held)
        trade = {
            "id": 1,
            "symbol": "AMD",
            "profile": "moderate",
            "direction": direction,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": entry_price * 1.1 if direction == "LONG" else entry_price * 0.9,
            "setup_type": setup_type,
            "status": "open",
            "quantity": 100,
            "entry_time": entry_time.isoformat(),
        }

        market_ts = FIXED_NOW_UTC - timedelta(seconds=5)
        result = evaluate_setup_aware_lifecycle(
            trade, [],
            now_utc=FIXED_NOW_UTC,
            now_et=FIXED_NOW_ET,
            current_price=current_price,
            market_data_timestamp=market_ts,
            shadow_mode=False,
        )

        # At a revalidation boundary, the state should NOT be setup_exit_alert
        # It should be a decisive outcome (hold, close, or invalidated)
        assert result["state"] != "setup_exit_alert", (
            f"Expected decisive outcome at revalidation boundary, "
            f"got setup_exit_alert. "
            f"setup={setup_type}, minutes={minutes_held}, "
            f"revalidate={policy.revalidate_minutes}, "
            f"decision='{result['decision']}', state='{result['state']}'"
        )
        # The decision should be either 'hold' or 'close' (decisive)
        assert result["decision"] in ("hold", "close"), (
            f"Expected hold or close at revalidation boundary, "
            f"got '{result['decision']}'. "
            f"setup={setup_type}, minutes={minutes_held}"
        )
