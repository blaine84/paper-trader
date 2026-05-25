"""
Property-based tests for invalidation and extension logic.

Tests Properties 4, 5, 6, 7 from the Setup-Aware Exit Governance design document.

**Validates: Requirements 1.12, 1.13, 2.1, 2.2, 2.4, 2.5, 2.6, 2.7**
"""

from datetime import datetime, timedelta, timezone

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from utils.setup_aware_evaluator import (
    _compute_next_revalidation_boundary,
    _validate_invalidation_criteria,
    evaluate_setup_aware_lifecycle,
)
from utils.setup_time_policy import (
    SetupTimePolicy,
    get_policy,
    SETUP_TIME_POLICY_REGISTRY,
)


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

# Extension-eligible setup types
st_extension_eligible_setup = st.sampled_from(["news_breakout", "news_catalyst", "trend_pullback"])

# Direction
st_direction = st.sampled_from(["LONG", "SHORT"])

# Prices: positive floats in a realistic range
st_price = st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False)

# Small positive offset for stop distance
st_stop_offset_pct = st.floats(min_value=0.005, max_value=0.15, allow_nan=False, allow_infinity=False)

# Revalidation interval index (which boundary we're at)
st_boundary_index = st.integers(min_value=0, max_value=3)


# ---------------------------------------------------------------------------
# Helper: Build a trade dict
# ---------------------------------------------------------------------------


def _make_trade(
    direction="LONG",
    entry_price=150.0,
    stop_price=None,
    target_price=None,
    setup_type="news_breakout",
    invalidators=None,
    confidence=None,
    sentiment=None,
    confidence_score=None,
    entry_time=None,
    **kwargs,
):
    """Build a minimal trade dict for testing."""
    trade = {
        "id": 1,
        "symbol": "TEST",
        "profile": "moderate",
        "direction": direction,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "setup_type": setup_type,
        "status": "open",
        "quantity": 100,
        "thesis": "Test thesis",
        "invalidators": invalidators,
        "entry_time": entry_time or datetime(2026, 5, 26, 10, 0, 0, tzinfo=timezone.utc),
    }
    if confidence is not None:
        trade["confidence"] = confidence
    if sentiment is not None:
        trade["sentiment"] = sentiment
    if confidence_score is not None:
        trade["confidence_score"] = confidence_score
    trade.update(kwargs)
    return trade


# ---------------------------------------------------------------------------
# Property 4: Extension-eligible revalidation loop
# ---------------------------------------------------------------------------


class TestProperty4ExtensionEligibleRevalidationLoop:
    """
    For any extension-eligible trade with valid invalidation criteria and fresh
    market data where the revalidation decision is `hold_valid_until_next_window`,
    the next revalidation boundary is exactly `revalidation_interval_minutes` after
    the current boundary. Verify the boundary sequence is correct.

    **Validates: Requirements 1.12, 1.13**
    """

    @given(
        setup_type=st_extension_eligible_setup,
        entry_price=st.floats(min_value=50.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        stop_offset_pct=st_stop_offset_pct,
        boundary_index=st_boundary_index,
    )
    @settings(max_examples=200)
    def test_hold_grants_interval_extension(
        self, setup_type, entry_price, stop_offset_pct, boundary_index
    ):
        """When revalidation produces hold_valid_until_next_window, the next boundary
        is exactly revalidation_interval_minutes after the current boundary."""
        policy = get_policy(setup_type)
        assert policy.extension_eligible
        assert policy.revalidation_interval_minutes is not None
        assert policy.revalidate_minutes is not None
        assert policy.max_extension_minutes is not None

        # Build the boundary sequence for this policy
        boundaries = [policy.revalidate_minutes]
        if policy.force_close_minutes > policy.revalidate_minutes:
            boundaries.append(policy.force_close_minutes)
        next_b = policy.force_close_minutes + policy.revalidation_interval_minutes
        while next_b <= policy.max_extension_minutes:
            if next_b not in boundaries:
                boundaries.append(next_b)
            next_b += policy.revalidation_interval_minutes
        if policy.max_extension_minutes not in boundaries:
            boundaries.append(policy.max_extension_minutes)

        # Pick a boundary that is NOT the last one (so there's a next boundary)
        assume(boundary_index < len(boundaries) - 1)
        current_boundary = boundaries[boundary_index]
        expected_next_boundary = boundaries[boundary_index + 1]

        # Compute next boundary using the implementation
        # minutes_held is at the current boundary
        next_boundary = _compute_next_revalidation_boundary(
            datetime(2026, 5, 26, 10, 0, 0, tzinfo=timezone.utc),
            policy,
            float(current_boundary),
        )

        assert next_boundary is not None, (
            f"Expected next boundary after {current_boundary} min for {setup_type}, got None"
        )
        assert next_boundary == expected_next_boundary, (
            f"setup_type={setup_type}, current_boundary={current_boundary}: "
            f"expected next={expected_next_boundary}, got next={next_boundary}"
        )

        # Verify the interval between consecutive boundaries is exactly
        # revalidation_interval_minutes (for boundaries after force_close)
        if current_boundary >= policy.force_close_minutes:
            expected_interval = policy.revalidation_interval_minutes
            actual_interval = expected_next_boundary - current_boundary
            assert actual_interval == expected_interval, (
                f"setup_type={setup_type}: interval from {current_boundary} to "
                f"{expected_next_boundary} is {actual_interval}, "
                f"expected {expected_interval}"
            )

    @given(
        setup_type=st_extension_eligible_setup,
        entry_price=st.floats(min_value=50.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        stop_offset_pct=st_stop_offset_pct,
    )
    @settings(max_examples=200)
    def test_hold_decision_includes_next_boundary_in_metadata(
        self, setup_type, entry_price, stop_offset_pct
    ):
        """When evaluator produces hold, metadata includes next_revalidation_boundary."""
        policy = get_policy(setup_type)
        assert policy.revalidate_minutes is not None

        # Set up a trade at the first revalidation boundary with valid criteria
        # and price above entry (positive indicator)
        stop_price = entry_price * (1 - stop_offset_pct)
        current_price = entry_price * 1.02  # Price above entry (positive indicator)

        # Ensure stop is valid (below entry for LONG)
        assume(stop_price > 0)
        assume(stop_price < entry_price)
        assume(current_price > stop_price)

        entry_time = datetime(2026, 5, 26, 10, 0, 0, tzinfo=timezone.utc)
        # Place now_utc at exactly revalidate_minutes after entry
        now_utc = entry_time + timedelta(minutes=policy.revalidate_minutes)
        # now_et well before EOD hard wall
        now_et = datetime(2026, 5, 26, 12, 0, 0)

        trade = _make_trade(
            direction="LONG",
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=entry_price * 1.10,
            setup_type=setup_type,
            entry_time=entry_time,
        )

        result = evaluate_setup_aware_lifecycle(
            trade,
            [],
            now_utc=now_utc,
            now_et=now_et,
            current_price=current_price,
            market_data_timestamp=now_utc - timedelta(seconds=5),
        )

        # Should produce a hold decision
        if result.get("decision") == "hold" and result.get("state") == "setup_revalidation_hold":
            metadata = result.get("metadata", {})
            next_boundary = metadata.get("next_revalidation_boundary")
            assert next_boundary is not None, (
                f"Hold decision for {setup_type} missing next_revalidation_boundary in metadata"
            )
            # Next boundary should be greater than revalidate_minutes
            assert next_boundary > policy.revalidate_minutes, (
                f"next_revalidation_boundary ({next_boundary}) should be > "
                f"revalidate_minutes ({policy.revalidate_minutes})"
            )

    @given(
        setup_type=st_extension_eligible_setup,
    )
    @settings(max_examples=50)
    def test_past_max_extension_returns_none(self, setup_type):
        """When minutes_held >= max_extension_minutes, next boundary is None."""
        policy = get_policy(setup_type)
        assert policy.max_extension_minutes is not None

        entry_time = datetime(2026, 5, 26, 10, 0, 0, tzinfo=timezone.utc)
        result = _compute_next_revalidation_boundary(
            entry_time, policy, float(policy.max_extension_minutes)
        )
        assert result is None, (
            f"Expected None for {setup_type} at max_extension={policy.max_extension_minutes}, "
            f"got {result}"
        )


# ---------------------------------------------------------------------------
# Property 5: Missing or malformed invalidation criteria deny extension
# ---------------------------------------------------------------------------


class TestProperty5MissingCriteriaDenyExtension:
    """
    For any extension-eligible trade at a revalidation boundary where stop_price
    is None AND invalidators is empty/None, the evaluator denies extension
    (produces close decision with reason containing "missing criteria").

    **Validates: Requirements 2.1, 2.2, 2.4**
    """

    @given(
        setup_type=st_extension_eligible_setup,
        entry_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        direction=st_direction,
        invalidators=st.sampled_from([None, {}, []]),
    )
    @settings(max_examples=200)
    def test_no_stop_no_invalidators_denies_extension(
        self, setup_type, entry_price, direction, invalidators
    ):
        """Trade with no stop_price and empty/None invalidators is denied extension."""
        policy = get_policy(setup_type)
        assert policy.extension_eligible
        assert policy.revalidate_minutes is not None

        entry_time = datetime(2026, 5, 26, 10, 0, 0, tzinfo=timezone.utc)
        # Place now_utc at the revalidation boundary
        now_utc = entry_time + timedelta(minutes=policy.revalidate_minutes)
        now_et = datetime(2026, 5, 26, 12, 0, 0)

        trade = _make_trade(
            direction=direction,
            entry_price=entry_price,
            stop_price=None,
            target_price=None,
            setup_type=setup_type,
            invalidators=invalidators,
            entry_time=entry_time,
        )

        result = evaluate_setup_aware_lifecycle(
            trade,
            [],
            now_utc=now_utc,
            now_et=now_et,
            current_price=entry_price * 1.01,
            market_data_timestamp=now_utc - timedelta(seconds=5),
        )

        # Should produce a close decision
        assert result["decision"] == "close", (
            f"Expected close for {setup_type} with no criteria, "
            f"got decision={result['decision']}, state={result.get('state')}"
        )
        # Reason type should indicate missing criteria
        assert result["reason_type"] == "setup_revalidation_denied_missing_criteria", (
            f"Expected reason_type='setup_revalidation_denied_missing_criteria', "
            f"got '{result['reason_type']}'"
        )

    @given(
        setup_type=st_extension_eligible_setup,
        entry_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        direction=st_direction,
    )
    @settings(max_examples=200)
    def test_none_stop_and_none_invalidators_validates_false(
        self, setup_type, entry_price, direction
    ):
        """_validate_invalidation_criteria returns is_valid=False when both are missing."""
        policy = get_policy(setup_type)

        trade = _make_trade(
            direction=direction,
            entry_price=entry_price,
            stop_price=None,
            invalidators=None,
            setup_type=setup_type,
        )

        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is False, (
            f"Expected is_valid=False for {setup_type} with no stop/invalidators, "
            f"got is_valid=True, reason='{reason}'"
        )

    @given(
        setup_type=st_extension_eligible_setup,
        entry_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        direction=st_direction,
    )
    @settings(max_examples=200)
    def test_empty_invalidators_dict_validates_false(
        self, setup_type, entry_price, direction
    ):
        """_validate_invalidation_criteria returns is_valid=False with empty invalidators dict."""
        policy = get_policy(setup_type)

        trade = _make_trade(
            direction=direction,
            entry_price=entry_price,
            stop_price=None,
            invalidators={},
            setup_type=setup_type,
        )

        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is False, (
            f"Expected is_valid=False for {setup_type} with empty invalidators, "
            f"got is_valid=True, reason='{reason}'"
        )


# ---------------------------------------------------------------------------
# Property 6: Invalid stop geometry denies extension eligibility
# ---------------------------------------------------------------------------


class TestProperty6InvalidStopGeometryDeniesExtension:
    """
    For any LONG trade where stop_price >= entry_price (wrong side), or
    stop_price == 0, the invalidation criteria validation returns is_valid=False.

    **Validates: Requirements 2.5**
    """

    @given(
        setup_type=st_extension_eligible_setup,
        entry_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        stop_above_pct=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_long_stop_at_or_above_entry_invalid(
        self, setup_type, entry_price, stop_above_pct
    ):
        """For LONG trades, stop_price >= entry_price is invalid (wrong side)."""
        policy = get_policy(setup_type)

        # Stop at or above entry for LONG
        stop_price = entry_price * (1 + stop_above_pct)
        assume(stop_price >= entry_price)

        trade = _make_trade(
            direction="LONG",
            entry_price=entry_price,
            stop_price=stop_price,
            invalidators=None,
            setup_type=setup_type,
        )

        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is False, (
            f"Expected is_valid=False for LONG with stop={stop_price} >= entry={entry_price}, "
            f"got is_valid=True, reason='{reason}'"
        )

    @given(
        setup_type=st_extension_eligible_setup,
        entry_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        stop_below_pct=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_short_stop_at_or_below_entry_invalid(
        self, setup_type, entry_price, stop_below_pct
    ):
        """For SHORT trades, stop_price <= entry_price is invalid (wrong side)."""
        policy = get_policy(setup_type)

        # Stop at or below entry for SHORT
        stop_price = entry_price * (1 - stop_below_pct)
        assume(stop_price <= entry_price)
        assume(stop_price > 0)  # Zero is tested separately

        trade = _make_trade(
            direction="SHORT",
            entry_price=entry_price,
            stop_price=stop_price,
            invalidators=None,
            setup_type=setup_type,
        )

        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is False, (
            f"Expected is_valid=False for SHORT with stop={stop_price} <= entry={entry_price}, "
            f"got is_valid=True, reason='{reason}'"
        )

    @given(
        setup_type=st_extension_eligible_setup,
        entry_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        direction=st_direction,
    )
    @settings(max_examples=200)
    def test_zero_stop_price_invalid(self, setup_type, entry_price, direction):
        """Stop price of zero is always invalid regardless of direction."""
        policy = get_policy(setup_type)

        trade = _make_trade(
            direction=direction,
            entry_price=entry_price,
            stop_price=0.0,
            invalidators=None,
            setup_type=setup_type,
        )

        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is False, (
            f"Expected is_valid=False for {direction} with stop=0.0, "
            f"got is_valid=True, reason='{reason}'"
        )

    @given(
        setup_type=st_extension_eligible_setup,
        entry_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        stop_above_pct=st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_long_wrong_side_stop_denies_extension_in_evaluator(
        self, setup_type, entry_price, stop_above_pct
    ):
        """Full evaluator denies extension when LONG stop is on wrong side."""
        policy = get_policy(setup_type)
        assert policy.revalidate_minutes is not None

        stop_price = entry_price * (1 + stop_above_pct)
        assume(stop_price > entry_price)

        entry_time = datetime(2026, 5, 26, 10, 0, 0, tzinfo=timezone.utc)
        now_utc = entry_time + timedelta(minutes=policy.revalidate_minutes)
        now_et = datetime(2026, 5, 26, 12, 0, 0)

        trade = _make_trade(
            direction="LONG",
            entry_price=entry_price,
            stop_price=stop_price,
            invalidators=None,
            setup_type=setup_type,
            entry_time=entry_time,
        )

        result = evaluate_setup_aware_lifecycle(
            trade,
            [],
            now_utc=now_utc,
            now_et=now_et,
            current_price=entry_price * 1.01,
            market_data_timestamp=now_utc - timedelta(seconds=5),
        )

        assert result["decision"] == "close", (
            f"Expected close for LONG with wrong-side stop, "
            f"got decision={result['decision']}"
        )
        assert result["reason_type"] == "setup_revalidation_denied_missing_criteria", (
            f"Expected reason_type='setup_revalidation_denied_missing_criteria', "
            f"got '{result['reason_type']}'"
        )


# ---------------------------------------------------------------------------
# Property 7: Qualitative-only criteria are insufficient for extension
# ---------------------------------------------------------------------------


class TestProperty7QualitativeOnlyCriteriaInsufficient:
    """
    For any trade where the only "invalidation" data is confidence scores,
    target_price, or sentiment (no numeric stop or structural level),
    `_validate_invalidation_criteria` returns is_valid=False.

    **Validates: Requirements 2.6**
    """

    @given(
        setup_type=st_extension_eligible_setup,
        entry_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        direction=st_direction,
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_confidence_score_only_insufficient(
        self, setup_type, entry_price, direction, confidence
    ):
        """Confidence score alone does not satisfy invalidation criteria."""
        policy = get_policy(setup_type)

        trade = _make_trade(
            direction=direction,
            entry_price=entry_price,
            stop_price=None,
            invalidators=None,
            setup_type=setup_type,
            confidence=confidence,
        )

        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is False, (
            f"Expected is_valid=False with only confidence={confidence}, "
            f"got is_valid=True, reason='{reason}'"
        )

    @given(
        setup_type=st_extension_eligible_setup,
        entry_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        direction=st_direction,
        target_price=st.floats(min_value=10.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_target_price_only_insufficient(
        self, setup_type, entry_price, direction, target_price
    ):
        """Target price alone does not satisfy invalidation criteria."""
        policy = get_policy(setup_type)

        trade = _make_trade(
            direction=direction,
            entry_price=entry_price,
            stop_price=None,
            target_price=target_price,
            invalidators=None,
            setup_type=setup_type,
        )

        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is False, (
            f"Expected is_valid=False with only target_price={target_price}, "
            f"got is_valid=True, reason='{reason}'"
        )

    @given(
        setup_type=st_extension_eligible_setup,
        entry_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        direction=st_direction,
        sentiment=st.sampled_from(["bullish", "bearish", "neutral", "very bullish", "strongly bearish"]),
    )
    @settings(max_examples=200)
    def test_sentiment_only_insufficient(
        self, setup_type, entry_price, direction, sentiment
    ):
        """Sentiment alone does not satisfy invalidation criteria."""
        policy = get_policy(setup_type)

        trade = _make_trade(
            direction=direction,
            entry_price=entry_price,
            stop_price=None,
            invalidators=None,
            setup_type=setup_type,
            sentiment=sentiment,
        )

        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is False, (
            f"Expected is_valid=False with only sentiment='{sentiment}', "
            f"got is_valid=True, reason='{reason}'"
        )

    @given(
        setup_type=st_extension_eligible_setup,
        entry_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        direction=st_direction,
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        target_price=st.floats(min_value=10.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        sentiment=st.sampled_from(["bullish", "bearish", "neutral"]),
    )
    @settings(max_examples=200)
    def test_all_qualitative_combined_insufficient(
        self, setup_type, entry_price, direction, confidence, target_price, sentiment
    ):
        """All qualitative data combined (confidence + target + sentiment) is insufficient."""
        policy = get_policy(setup_type)

        trade = _make_trade(
            direction=direction,
            entry_price=entry_price,
            stop_price=None,
            target_price=target_price,
            invalidators=None,
            setup_type=setup_type,
            confidence=confidence,
            sentiment=sentiment,
        )

        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is False, (
            f"Expected is_valid=False with confidence={confidence}, "
            f"target_price={target_price}, sentiment='{sentiment}', "
            f"got is_valid=True, reason='{reason}'"
        )

    @given(
        setup_type=st_extension_eligible_setup,
        entry_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        direction=st_direction,
        confidence_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_confidence_score_field_variant_insufficient(
        self, setup_type, entry_price, direction, confidence_score
    ):
        """confidence_score field (variant of confidence) alone is insufficient."""
        policy = get_policy(setup_type)

        trade = _make_trade(
            direction=direction,
            entry_price=entry_price,
            stop_price=None,
            invalidators=None,
            setup_type=setup_type,
            confidence_score=confidence_score,
        )

        is_valid, reason, criteria = _validate_invalidation_criteria(trade, policy)
        assert is_valid is False, (
            f"Expected is_valid=False with only confidence_score={confidence_score}, "
            f"got is_valid=True, reason='{reason}'"
        )
