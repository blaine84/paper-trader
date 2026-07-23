"""
Property-based tests for force-close payload, legacy override prevention,
priority chain, and pre-wall buffer logic.

Tests Properties 15, 16, 17, 18 from the design document using Hypothesis.
Feature: setup-aware-exit-governance

**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 7.1, 7.6**
"""

from datetime import datetime, timedelta, timezone

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from utils.setup_aware_evaluator import evaluate_setup_aware_lifecycle
from utils.setup_time_policy import (
    get_policy,
    SETUP_TIME_POLICY_REGISTRY,
)
from utils.position_lifecycle_governance import evaluate_position_lifecycle


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

FIXED_NOW_UTC = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
# now_et well before EOD hard wall (15:45 ET) to avoid pre-wall buffer interference
FIXED_NOW_ET = datetime(2026, 5, 26, 12, 0, 0)

st_price = st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
st_direction = st.sampled_from(["LONG", "SHORT"])

# Non-extension-eligible setup types (force close at their limit, no revalidation)
NON_EXTENSION_SETUPS = [
    k for k, v in SETUP_TIME_POLICY_REGISTRY.items() if not v.extension_eligible
]
st_non_extension_setup = st.sampled_from(NON_EXTENSION_SETUPS)

# All known setup types
st_known_setup = st.sampled_from(list(SETUP_TIME_POLICY_REGISTRY.keys()))

# Extension-eligible setups
EXTENSION_ELIGIBLE_SETUPS = [
    k for k, v in SETUP_TIME_POLICY_REGISTRY.items() if v.extension_eligible
]
st_extension_eligible_setup = st.sampled_from(EXTENSION_ELIGIBLE_SETUPS)

# News setups specifically (news_breakout, news_catalyst)
st_news_setup = st.sampled_from(["news_breakout", "news_catalyst"])


# ---------------------------------------------------------------------------
# Property 15: Force close includes complete payload
# Feature: setup-aware-exit-governance, Property 15: Force close includes complete payload
# ---------------------------------------------------------------------------


class TestProperty15ForceCloseIncludesCompletePayload:
    """
    For any non-extension-eligible trade past its force_close_minutes, the
    evaluator produces a close decision with metadata containing: setup_type,
    force_close_minutes, minutes_held, revalidation_attempted=False.

    **Validates: Requirements 4.1, 4.2, 4.7**
    """

    @given(
        setup_type=st_non_extension_setup,
        direction=st_direction,
        entry_price=st_price,
        extra_minutes=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_force_close_payload_contains_required_fields(
        self, setup_type, direction, entry_price, extra_minutes,
    ):
        """Non-extension-eligible trade past force_close_minutes has complete payload."""
        policy = get_policy(setup_type)
        minutes_held = policy.force_close_minutes + extra_minutes

        entry_time = FIXED_NOW_UTC - timedelta(minutes=minutes_held)

        if direction == "LONG":
            stop_price = round(entry_price * 0.95, 2)
        else:
            stop_price = round(entry_price * 1.05, 2)

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
            current_price=entry_price,
            market_data_timestamp=FIXED_NOW_UTC - timedelta(seconds=5),
            shadow_mode=False,
        )

        # Must be a close decision
        assert result["decision"] == "close", (
            f"Expected close for {setup_type} at {minutes_held:.0f} min "
            f"(limit: {policy.force_close_minutes}), got '{result['decision']}'"
        )

        # Metadata must contain required force-close payload fields
        metadata = result.get("metadata", {})
        assert "setup_type" in metadata, "metadata missing 'setup_type'"
        assert metadata["setup_type"] == setup_type
        assert "force_close_minutes" in metadata, "metadata missing 'force_close_minutes'"
        assert metadata["force_close_minutes"] == policy.force_close_minutes
        assert "minutes_held" in metadata, "metadata missing 'minutes_held'"
        assert metadata["minutes_held"] == round(minutes_held)
        assert "revalidation_attempted" in metadata, "metadata missing 'revalidation_attempted'"
        assert metadata["revalidation_attempted"] is False, (
            f"Non-extension-eligible trade should have revalidation_attempted=False, "
            f"got {metadata['revalidation_attempted']}"
        )

        # The state should indicate setup_time_limit_exceeded
        assert result["state"] == "setup_time_limit_exceeded", (
            f"Expected state 'setup_time_limit_exceeded', got '{result['state']}'"
        )

        # requires_event should be True (for setup_exit_force_close event emission)
        assert result["requires_event"] is True, (
            "Force close decision must have requires_event=True for event emission"
        )


# ---------------------------------------------------------------------------
# Property 16: News_breakout not closed at legacy 90-minute threshold
# Feature: setup-aware-exit-governance, Property 16: News_breakout not closed at legacy 90-minute threshold
# ---------------------------------------------------------------------------


class TestProperty16NewsBreakoutNotClosedAtLegacy90MinThreshold:
    """
    For any news_breakout or news_catalyst trade at 91 minutes (past legacy
    90-min threshold) with valid stop_price and price above entry, the evaluator
    does NOT produce a close decision. It should produce a hold (revalidation
    passed).

    **Validates: Requirements 4.3**
    """

    @given(
        setup_type=st_news_setup,
        direction=st_direction,
        entry_price=st_price,
        stop_offset_pct=st.floats(min_value=0.01, max_value=0.05, allow_nan=False, allow_infinity=False),
        price_above_pct=st.floats(min_value=0.001, max_value=0.10, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_news_trade_at_91_minutes_not_force_closed(
        self, setup_type, direction, entry_price, stop_offset_pct, price_above_pct,
    ):
        """News trade at 91 min with valid stop and price above entry → NOT closed."""
        policy = get_policy(setup_type)
        # 91 minutes: past legacy 90-min threshold but within news_breakout's
        # revalidation window (revalidate_minutes=90, force_close=120)
        minutes_held = 91.0

        entry_time = FIXED_NOW_UTC - timedelta(minutes=minutes_held)

        # Build valid stop on loss side and current price on profit side
        if direction == "LONG":
            stop_price = round(entry_price * (1 - stop_offset_pct), 2)
            current_price = round(entry_price * (1 + price_above_pct), 2)
            target_price = round(entry_price * 1.15, 2)
        else:
            stop_price = round(entry_price * (1 + stop_offset_pct), 2)
            current_price = round(entry_price * (1 - price_above_pct), 2)
            target_price = round(entry_price * 0.85, 2)

        assume(stop_price > 0)
        assume(current_price > 0)
        # Rounding entry/current to 2 decimals can collapse the intended
        # "price above entry" margin at low entry prices (e.g. entry=1.0000...02,
        # current rounds to 1.00 which is BELOW entry). That destroys the test's
        # own premise ("price above entry") and correctly yields a force-close
        # (no positive indicator), so exclude those knife-edge inputs and only
        # exercise genuine positive-indicator cases.
        if direction == "LONG":
            assume(current_price > entry_price)
        else:
            assume(current_price < entry_price)

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

        market_ts = FIXED_NOW_UTC - timedelta(seconds=5)
        result = evaluate_setup_aware_lifecycle(
            trade, [],
            now_utc=FIXED_NOW_UTC,
            now_et=FIXED_NOW_ET,
            current_price=current_price,
            market_data_timestamp=market_ts,
            shadow_mode=False,
        )

        # The evaluator should NOT close this trade at 91 minutes
        # It should produce a hold decision (revalidation passed with positive indicators)
        assert result["decision"] != "close" or result.get("reason_type") not in (
            "setup_time_limit_exceeded",
            "intraday_time_limit_exceeded",
        ), (
            f"News trade ({setup_type}) at 91 min should NOT be force-closed at "
            f"legacy 90-min threshold. Got decision='{result['decision']}', "
            f"reason_type='{result.get('reason_type')}', state='{result.get('state')}'. "
            f"direction={direction}, entry={entry_price}, current={current_price}, "
            f"stop={stop_price}"
        )


# ---------------------------------------------------------------------------
# Property 17: Priority chain prevents setup-aware logic when hard controls decide
# Feature: setup-aware-exit-governance, Property 17: Priority chain prevents setup-aware logic when hard controls decide
# ---------------------------------------------------------------------------


class TestProperty17PriorityChainPreventsSetupAwareLogicWhenHardControlsDecide:
    """
    For the evaluate_position_lifecycle function, when a trade is
    closed/cancelled (status="closed"), the result is always "skip"
    regardless of setup-aware logic. This tests that priority 1 (skip)
    takes precedence.

    **Validates: Requirements 4.4, 4.5, 4.6, 7.1**
    """

    @given(
        setup_type=st_known_setup,
        direction=st_direction,
        entry_price=st_price,
        minutes_held=st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
        current_price=st.one_of(st_price, st.none()),
    )
    @settings(max_examples=200)
    def test_closed_trade_always_skipped_regardless_of_setup(
        self, setup_type, direction, entry_price, minutes_held, current_price,
    ):
        """Closed/cancelled trade always returns 'skip' — priority 1 overrides all."""
        entry_time = FIXED_NOW_UTC - timedelta(minutes=minutes_held)

        if direction == "LONG":
            stop_price = round(entry_price * 0.95, 2)
        else:
            stop_price = round(entry_price * 1.05, 2)

        trade = {
            "id": 1,
            "symbol": "AMD",
            "profile": "moderate",
            "direction": direction,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": entry_price * 1.1 if direction == "LONG" else entry_price * 0.9,
            "setup_type": setup_type,
            "status": "closed",  # Already closed
            "quantity": 100,
            "entry_time": entry_time.isoformat(),
        }

        market_ts = FIXED_NOW_UTC - timedelta(seconds=5) if current_price else None

        result = evaluate_position_lifecycle(
            trade, [],
            now_utc=FIXED_NOW_UTC,
            now_et=FIXED_NOW_ET,
            current_price=current_price,
            market_data_timestamp=market_ts,
            shadow_mode=False,
        )

        assert result["decision"] == "skip", (
            f"Expected 'skip' for closed trade, got '{result['decision']}'. "
            f"setup={setup_type}, minutes={minutes_held}, state='{result.get('state')}'"
        )
        assert result["state"] == "skipped", (
            f"Expected state 'skipped', got '{result['state']}'"
        )


# ---------------------------------------------------------------------------
# Property 18: Pre-wall buffer revokes extension
# Feature: setup-aware-exit-governance, Property 18: Pre-wall buffer revokes extension
# ---------------------------------------------------------------------------


class TestProperty18PreWallBufferRevokesExtension:
    """
    For any trade where now_et time >= 15:30 (3:30 PM ET, which is 15 min
    before the 3:45 PM hard wall), the evaluator produces a close decision
    with reason_type="setup_pre_wall_buffer_close" regardless of setup type
    or extension status.

    **Validates: Requirements 7.6**
    """

    @given(
        setup_type=st_known_setup,
        direction=st_direction,
        entry_price=st_price,
        minutes_held=st.floats(min_value=10.0, max_value=200.0, allow_nan=False, allow_infinity=False),
        # Generate times from 15:30 to 15:59 (at or past the pre-wall buffer)
        hour=st.just(15),
        minute=st.integers(min_value=30, max_value=59),
    )
    @settings(max_examples=200)
    def test_pre_wall_buffer_produces_close_decision(
        self, setup_type, direction, entry_price, minutes_held, hour, minute,
    ):
        """Trade at or past 3:30 PM ET → close with setup_pre_wall_buffer_close."""
        entry_time = FIXED_NOW_UTC - timedelta(minutes=minutes_held)

        if direction == "LONG":
            stop_price = round(entry_price * 0.95, 2)
            current_price = round(entry_price * 1.05, 2)
        else:
            stop_price = round(entry_price * 1.05, 2)
            current_price = round(entry_price * 0.95, 2)

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

        # Set now_et to be at or past 3:30 PM ET (pre-wall buffer threshold)
        now_et = datetime(2026, 5, 26, hour, minute, 0)

        market_ts = FIXED_NOW_UTC - timedelta(seconds=5)
        result = evaluate_setup_aware_lifecycle(
            trade, [],
            now_utc=FIXED_NOW_UTC,
            now_et=now_et,
            current_price=current_price,
            market_data_timestamp=market_ts,
            shadow_mode=False,
        )

        assert result["decision"] == "close", (
            f"Expected close at pre-wall buffer time {hour}:{minute:02d} ET, "
            f"got '{result['decision']}'. setup={setup_type}, "
            f"state='{result.get('state')}', reason='{result.get('reason_type')}'"
        )
        assert result["reason_type"] == "setup_pre_wall_buffer_close", (
            f"Expected reason_type 'setup_pre_wall_buffer_close', "
            f"got '{result['reason_type']}'. setup={setup_type}, "
            f"time={hour}:{minute:02d} ET"
        )

        # Metadata should indicate extension was revoked
        metadata = result.get("metadata", {})
        assert metadata.get("extension_revoked") is True, (
            f"Expected metadata.extension_revoked=True, got {metadata.get('extension_revoked')}"
        )
