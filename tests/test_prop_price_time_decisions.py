"""
Property-based tests for price/time revalidation decisions using Hypothesis.

Tests Properties 11, 12, 13 from the setup-aware-exit-governance design:
- Property 11: Price breach triggers thesis invalidation
- Property 12: Valid conditions produce hold decision
- Property 13: Max extension reached triggers time expiry

These properties validate the deterministic decision logic in
`_evaluate_revalidation_decision` across a wide input space.
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from utils.setup_aware_evaluator import _evaluate_revalidation_decision
from utils.setup_time_policy import get_policy, SETUP_TIME_POLICY_REGISTRY


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Extension-eligible setup types (thesis-development setups)
st_extension_eligible_setup = st.sampled_from(["news_breakout", "news_catalyst", "trend_pullback"])

# Prices: realistic stock prices
st_price = st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False)

# Trade direction
st_direction = st.sampled_from(["LONG", "SHORT"])

# Minutes held (0 to 300 covers all policy windows)
st_minutes_held = st.floats(min_value=0.0, max_value=300.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Property 11: Price breach triggers thesis invalidation
# Feature: setup-aware-exit-governance, Property 11: Price breach triggers thesis invalidation
# ---------------------------------------------------------------------------


class TestProperty11PriceBreachTriggersThesisInvalidation:
    """
    For any trade where current_price has breached the documented stop or
    structural invalidation level (below stop for longs, above stop for shorts),
    the evaluator SHALL produce `close_thesis_invalidated` regardless of time
    held or extension status.

    **Validates: Requirements 3.5**
    """

    @given(
        setup_type=st_extension_eligible_setup,
        entry_price=st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        stop_distance_pct=st.floats(min_value=0.01, max_value=0.15, allow_nan=False, allow_infinity=False),
        breach_amount_pct=st.floats(min_value=0.0, max_value=0.10, allow_nan=False, allow_infinity=False),
        minutes_held=st_minutes_held,
    )
    @settings(max_examples=200)
    def test_long_price_at_or_below_stop_triggers_thesis_invalidation(
        self,
        setup_type: str,
        entry_price: float,
        stop_distance_pct: float,
        breach_amount_pct: float,
        minutes_held: float,
    ):
        """LONG: current_price <= stop_price → close_thesis_invalidated."""
        policy = get_policy(setup_type)
        stop_price = round(entry_price * (1 - stop_distance_pct), 2)
        # Price at or below stop (breach)
        current_price = round(stop_price * (1 - breach_amount_pct), 2)

        # Ensure the breach condition holds
        assume(current_price <= stop_price)
        assume(stop_price < entry_price)
        assume(current_price > 0)

        trade = {
            "id": 1,
            "symbol": "TEST",
            "direction": "LONG",
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": entry_price * 1.1,
            "setup_type": setup_type,
            "status": "open",
            "quantity": 100,
        }
        criteria = {"stop_price": stop_price}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=current_price, minutes_held=minutes_held, criteria=criteria
        )

        assert outcome == "close_thesis_invalidated", (
            f"LONG breach: price={current_price} <= stop={stop_price} should trigger "
            f"close_thesis_invalidated, got {outcome}"
        )

    @given(
        setup_type=st_extension_eligible_setup,
        entry_price=st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        stop_distance_pct=st.floats(min_value=0.01, max_value=0.15, allow_nan=False, allow_infinity=False),
        breach_amount_pct=st.floats(min_value=0.0, max_value=0.10, allow_nan=False, allow_infinity=False),
        minutes_held=st_minutes_held,
    )
    @settings(max_examples=200)
    def test_short_price_at_or_above_stop_triggers_thesis_invalidation(
        self,
        setup_type: str,
        entry_price: float,
        stop_distance_pct: float,
        breach_amount_pct: float,
        minutes_held: float,
    ):
        """SHORT: current_price >= stop_price → close_thesis_invalidated."""
        policy = get_policy(setup_type)
        stop_price = round(entry_price * (1 + stop_distance_pct), 2)
        # Price at or above stop (breach)
        current_price = round(stop_price * (1 + breach_amount_pct), 2)

        # Ensure the breach condition holds
        assume(current_price >= stop_price)
        assume(stop_price > entry_price)

        trade = {
            "id": 1,
            "symbol": "TEST",
            "direction": "SHORT",
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": entry_price * 0.9,
            "setup_type": setup_type,
            "status": "open",
            "quantity": 100,
        }
        criteria = {"stop_price": stop_price}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=current_price, minutes_held=minutes_held, criteria=criteria
        )

        assert outcome == "close_thesis_invalidated", (
            f"SHORT breach: price={current_price} >= stop={stop_price} should trigger "
            f"close_thesis_invalidated, got {outcome}"
        )


# ---------------------------------------------------------------------------
# Property 12: Valid conditions produce hold decision
# Feature: setup-aware-exit-governance, Property 12: Valid conditions produce hold decision
# ---------------------------------------------------------------------------


class TestProperty12ValidConditionsProduceHoldDecision:
    """
    For any extension-eligible trade at a revalidation boundary where:
    current price has NOT breached the stop/invalidation level, minutes_held
    has NOT exceeded max_extension_minutes, AND at least one of (price >= entry,
    price >= VWAP/support, target_progress >= 25%) holds, the evaluator SHALL
    produce `hold_valid_until_next_window`.

    **Validates: Requirements 3.6**
    """

    @given(
        setup_type=st_extension_eligible_setup,
        entry_price=st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        stop_distance_pct=st.floats(min_value=0.01, max_value=0.15, allow_nan=False, allow_infinity=False),
        price_above_entry_pct=st.floats(min_value=0.0, max_value=0.20, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_long_price_at_or_above_entry_produces_hold(
        self,
        setup_type: str,
        entry_price: float,
        stop_distance_pct: float,
        price_above_entry_pct: float,
    ):
        """LONG: price >= entry AND price > stop AND minutes < max → hold."""
        policy = get_policy(setup_type)
        stop_price = round(entry_price * (1 - stop_distance_pct), 2)
        current_price = round(entry_price * (1 + price_above_entry_pct), 2)

        # Ensure valid conditions: price above stop, price >= entry, within max extension
        assume(current_price > stop_price)
        assume(current_price >= entry_price)
        assume(stop_price < entry_price)

        # Use minutes_held below max_extension_minutes
        max_ext = policy.max_extension_minutes
        assume(max_ext is not None)
        minutes_held = max_ext - 10.0  # Well within limit

        trade = {
            "id": 1,
            "symbol": "TEST",
            "direction": "LONG",
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": entry_price * 1.1,
            "setup_type": setup_type,
            "status": "open",
            "quantity": 100,
        }
        criteria = {"stop_price": stop_price}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=current_price, minutes_held=minutes_held, criteria=criteria
        )

        assert outcome == "hold_valid_until_next_window", (
            f"LONG valid: price={current_price} >= entry={entry_price}, "
            f"price > stop={stop_price}, minutes={minutes_held} < max={max_ext} "
            f"should produce hold, got {outcome}"
        )

    @given(
        setup_type=st_extension_eligible_setup,
        entry_price=st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        stop_distance_pct=st.floats(min_value=0.01, max_value=0.15, allow_nan=False, allow_infinity=False),
        price_below_entry_pct=st.floats(min_value=0.0, max_value=0.20, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_short_price_at_or_below_entry_produces_hold(
        self,
        setup_type: str,
        entry_price: float,
        stop_distance_pct: float,
        price_below_entry_pct: float,
    ):
        """SHORT: price <= entry AND price < stop AND minutes < max → hold."""
        policy = get_policy(setup_type)
        stop_price = round(entry_price * (1 + stop_distance_pct), 2)
        current_price = round(entry_price * (1 - price_below_entry_pct), 2)

        # Ensure valid conditions: price below stop, price <= entry, within max extension
        assume(current_price < stop_price)
        assume(current_price <= entry_price)
        assume(stop_price > entry_price)
        assume(current_price > 0)

        max_ext = policy.max_extension_minutes
        assume(max_ext is not None)
        minutes_held = max_ext - 10.0

        trade = {
            "id": 1,
            "symbol": "TEST",
            "direction": "SHORT",
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": entry_price * 0.9,
            "setup_type": setup_type,
            "status": "open",
            "quantity": 100,
        }
        criteria = {"stop_price": stop_price}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=current_price, minutes_held=minutes_held, criteria=criteria
        )

        assert outcome == "hold_valid_until_next_window", (
            f"SHORT valid: price={current_price} <= entry={entry_price}, "
            f"price < stop={stop_price}, minutes={minutes_held} < max={max_ext} "
            f"should produce hold, got {outcome}"
        )

    @given(
        setup_type=st_extension_eligible_setup,
        entry_price=st.floats(min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        stop_distance_pct=st.floats(min_value=0.01, max_value=0.10, allow_nan=False, allow_infinity=False),
        target_distance_pct=st.floats(min_value=0.05, max_value=0.30, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_long_target_progress_above_25pct_produces_hold(
        self,
        setup_type: str,
        entry_price: float,
        stop_distance_pct: float,
        target_distance_pct: float,
    ):
        """LONG: target_progress >= 25% AND price > stop AND minutes < max → hold."""
        policy = get_policy(setup_type)
        stop_price = round(entry_price * (1 - stop_distance_pct), 2)
        target_price = round(entry_price * (1 + target_distance_pct), 2)

        # Price at 25% or more of the way to target
        # target_progress = (current - entry) / (target - entry) * 100
        # For >= 25%: current >= entry + 0.25 * (target - entry)
        min_price_for_25pct = entry_price + 0.25 * (target_price - entry_price)
        current_price = round(min_price_for_25pct + 0.01, 2)

        # Ensure conditions hold
        assume(current_price > stop_price)
        assume(target_price > entry_price)
        assume(stop_price < entry_price)

        max_ext = policy.max_extension_minutes
        assume(max_ext is not None)
        minutes_held = max_ext - 10.0

        trade = {
            "id": 1,
            "symbol": "TEST",
            "direction": "LONG",
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "setup_type": setup_type,
            "status": "open",
            "quantity": 100,
        }
        criteria = {"stop_price": stop_price}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=current_price, minutes_held=minutes_held, criteria=criteria
        )

        assert outcome == "hold_valid_until_next_window", (
            f"LONG target progress: price={current_price}, entry={entry_price}, "
            f"target={target_price}, progress should be >= 25%, got {outcome}"
        )


# ---------------------------------------------------------------------------
# Property 13: Max extension reached triggers time expiry
# Feature: setup-aware-exit-governance, Property 13: Max extension reached triggers time expiry
# ---------------------------------------------------------------------------


class TestProperty13MaxExtensionReachedTriggersTimeExpiry:
    """
    For any extension-eligible trade where minutes_held >= max_extension_minutes,
    the evaluator SHALL produce `close_time_expired` regardless of price state
    or invalidation criteria validity.

    **Validates: Requirements 3.7**
    """

    @given(
        setup_type=st_extension_eligible_setup,
        entry_price=st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        stop_distance_pct=st.floats(min_value=0.01, max_value=0.15, allow_nan=False, allow_infinity=False),
        extra_minutes=st.floats(min_value=0.0, max_value=120.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_minutes_at_or_above_max_extension_triggers_time_expiry(
        self,
        setup_type: str,
        entry_price: float,
        stop_distance_pct: float,
        extra_minutes: float,
    ):
        """minutes_held >= max_extension_minutes → close_time_expired (price NOT breached)."""
        policy = get_policy(setup_type)
        max_ext = policy.max_extension_minutes
        assume(max_ext is not None)

        stop_price = round(entry_price * (1 - stop_distance_pct), 2)
        # Price well above stop (NOT breached) — ensures breach doesn't take priority
        current_price = round(entry_price * 1.05, 2)

        assume(current_price > stop_price)
        assume(stop_price < entry_price)

        # minutes_held at or above max extension
        minutes_held = max_ext + extra_minutes

        trade = {
            "id": 1,
            "symbol": "TEST",
            "direction": "LONG",
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": entry_price * 1.1,
            "setup_type": setup_type,
            "status": "open",
            "quantity": 100,
        }
        criteria = {"stop_price": stop_price}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=current_price, minutes_held=minutes_held, criteria=criteria
        )

        assert outcome == "close_time_expired", (
            f"minutes_held={minutes_held} >= max_extension={max_ext} should trigger "
            f"close_time_expired, got {outcome}"
        )

    @given(
        setup_type=st_extension_eligible_setup,
        entry_price=st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        stop_distance_pct=st.floats(min_value=0.01, max_value=0.15, allow_nan=False, allow_infinity=False),
        extra_minutes=st.floats(min_value=0.0, max_value=120.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_short_max_extension_triggers_time_expiry(
        self,
        setup_type: str,
        entry_price: float,
        stop_distance_pct: float,
        extra_minutes: float,
    ):
        """SHORT: minutes_held >= max_extension_minutes → close_time_expired (price NOT breached)."""
        policy = get_policy(setup_type)
        max_ext = policy.max_extension_minutes
        assume(max_ext is not None)

        stop_price = round(entry_price * (1 + stop_distance_pct), 2)
        # Price well below stop (NOT breached for SHORT)
        current_price = round(entry_price * 0.95, 2)

        assume(current_price < stop_price)
        assume(stop_price > entry_price)
        assume(current_price > 0)

        # minutes_held at or above max extension
        minutes_held = max_ext + extra_minutes

        trade = {
            "id": 1,
            "symbol": "TEST",
            "direction": "SHORT",
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": entry_price * 0.9,
            "setup_type": setup_type,
            "status": "open",
            "quantity": 100,
        }
        criteria = {"stop_price": stop_price}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=current_price, minutes_held=minutes_held, criteria=criteria
        )

        assert outcome == "close_time_expired", (
            f"SHORT: minutes_held={minutes_held} >= max_extension={max_ext} should trigger "
            f"close_time_expired, got {outcome}"
        )

    @given(
        setup_type=st_extension_eligible_setup,
        entry_price=st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        stop_distance_pct=st.floats(min_value=0.01, max_value=0.15, allow_nan=False, allow_infinity=False),
        breach_amount_pct=st.floats(min_value=0.001, max_value=0.10, allow_nan=False, allow_infinity=False),
        extra_minutes=st.floats(min_value=0.0, max_value=120.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_price_breach_takes_priority_over_max_extension(
        self,
        setup_type: str,
        entry_price: float,
        stop_distance_pct: float,
        breach_amount_pct: float,
        extra_minutes: float,
    ):
        """When BOTH price breach AND max extension hold, breach (step 1) wins over time (step 2)."""
        policy = get_policy(setup_type)
        max_ext = policy.max_extension_minutes
        assume(max_ext is not None)

        stop_price = round(entry_price * (1 - stop_distance_pct), 2)
        # Price below stop (breached for LONG)
        current_price = round(stop_price * (1 - breach_amount_pct), 2)

        assume(current_price <= stop_price)
        assume(stop_price < entry_price)
        assume(current_price > 0)

        # Also past max extension
        minutes_held = max_ext + extra_minutes

        trade = {
            "id": 1,
            "symbol": "TEST",
            "direction": "LONG",
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": entry_price * 1.1,
            "setup_type": setup_type,
            "status": "open",
            "quantity": 100,
        }
        criteria = {"stop_price": stop_price}

        outcome, reason, meta = _evaluate_revalidation_decision(
            trade, policy, current_price=current_price, minutes_held=minutes_held, criteria=criteria
        )

        # Price breach (step 1) takes priority over max extension (step 2)
        assert outcome == "close_thesis_invalidated", (
            f"Price breach should take priority over max extension: "
            f"price={current_price} <= stop={stop_price}, minutes={minutes_held} >= max={max_ext}, "
            f"got {outcome}"
        )
