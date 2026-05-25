"""
Property-based tests for Entry Contract Validation and Event Payload construction.

Tests Properties 19, 20, 21 from the design document using Hypothesis.
Feature: setup-aware-exit-governance

**Validates: Requirements 3.9, 3.10, 5.2, 5.3, 5.4, 6.1, 6.3**
"""

from datetime import datetime, timedelta, timezone

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from utils.setup_aware_evaluator import (
    SETUP_EXIT_EVENT_TYPES,
    build_setup_exit_event_payload,
    generate_dedupe_key,
)
from utils.entry_contract_validator import validate_entry_for_exit_governance


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

FIXED_NOW_UTC = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)

st_price = st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
st_trade_id = st.integers(min_value=1, max_value=100000)
st_event_type = st.sampled_from(sorted(SETUP_EXIT_EVENT_TYPES))
st_setup_type = st.sampled_from([
    "news_breakout", "news_catalyst", "trend_pullback",
    "momentum_fade", "orb", "short_squeeze", "gap_and_go", "vwap_reclaim",
])
st_minutes_held = st.floats(min_value=0.0, max_value=300.0, allow_nan=False, allow_infinity=False)
st_decision_outcome = st.sampled_from([
    "hold_valid_until_next_window",
    "close_time_expired",
    "close_thesis_invalidated",
    "warn_revalidation_due",
    "insufficient_data_close_at_base_limit",
])
st_base_force_close = st.integers(min_value=60, max_value=180)
st_next_limit = st.one_of(st.none(), st.integers(min_value=90, max_value=240))
st_revalidation_attempted = st.booleans()
st_extension_active = st.booleans()
st_close_reason = st.one_of(st.none(), st.text(min_size=5, max_size=100))
st_invalidation_criteria = st.one_of(
    st.none(),
    st.fixed_dictionaries({"stop_price": st_price}),
    st.fixed_dictionaries({"vwap_level": st_price}),
    st.fixed_dictionaries({"support_level": st_price, "stop_price": st_price}),
)

# Timestamp strategy: random UTC timestamps within a reasonable range
st_timestamp_utc = st.builds(
    lambda offset_minutes: FIXED_NOW_UTC + timedelta(minutes=offset_minutes),
    st.integers(min_value=-500, max_value=500),
)

# Thesis-development setup types
st_thesis_development_setup = st.sampled_from(["news_breakout", "news_catalyst", "trend_pullback"])


# ---------------------------------------------------------------------------
# Property 19: Revalidation events include all required fields
# Feature: setup-aware-exit-governance, Property 19: Revalidation events include all required fields
# ---------------------------------------------------------------------------


class TestProperty19RevalidationEventsIncludeAllRequiredFields:
    """
    For any call to build_setup_exit_event_payload, the returned dict contains
    ALL required fields: trade_id, setup_type, event_type, minutes_held,
    decision_outcome, invalidation_criteria_used, timestamp_utc,
    base_force_close_limit, next_limit_if_extended, revalidation_attempted,
    extension_active, close_reason, dedupe_key.

    **Validates: Requirements 3.9, 3.10, 6.1**
    """

    REQUIRED_FIELDS = {
        "trade_id",
        "setup_type",
        "event_type",
        "minutes_held",
        "decision_outcome",
        "invalidation_criteria_used",
        "timestamp_utc",
        "base_force_close_limit",
        "next_limit_if_extended",
        "revalidation_attempted",
        "extension_active",
        "close_reason",
        "dedupe_key",
    }

    @given(
        trade_id=st_trade_id,
        setup_type=st_setup_type,
        event_type=st_event_type,
        minutes_held=st_minutes_held,
        decision_outcome=st_decision_outcome,
        invalidation_criteria_used=st_invalidation_criteria,
        timestamp_utc=st_timestamp_utc,
        base_force_close_limit=st_base_force_close,
        next_limit_if_extended=st_next_limit,
        revalidation_attempted=st_revalidation_attempted,
        extension_active=st_extension_active,
        close_reason=st_close_reason,
    )
    @settings(max_examples=200)
    def test_payload_contains_all_required_fields(
        self,
        trade_id,
        setup_type,
        event_type,
        minutes_held,
        decision_outcome,
        invalidation_criteria_used,
        timestamp_utc,
        base_force_close_limit,
        next_limit_if_extended,
        revalidation_attempted,
        extension_active,
        close_reason,
    ):
        """build_setup_exit_event_payload always returns a dict with all required fields."""
        payload = build_setup_exit_event_payload(
            trade_id=trade_id,
            setup_type=setup_type,
            event_type=event_type,
            minutes_held=minutes_held,
            decision_outcome=decision_outcome,
            invalidation_criteria_used=invalidation_criteria_used,
            timestamp_utc=timestamp_utc,
            base_force_close_limit=base_force_close_limit,
            next_limit_if_extended=next_limit_if_extended,
            revalidation_attempted=revalidation_attempted,
            extension_active=extension_active,
            close_reason=close_reason,
        )

        missing_fields = self.REQUIRED_FIELDS - set(payload.keys())
        assert not missing_fields, (
            f"Payload missing required fields: {missing_fields}. "
            f"Got keys: {set(payload.keys())}"
        )

    @given(
        trade_id=st_trade_id,
        setup_type=st_setup_type,
        event_type=st_event_type,
        minutes_held=st_minutes_held,
        decision_outcome=st_decision_outcome,
        invalidation_criteria_used=st_invalidation_criteria,
        timestamp_utc=st_timestamp_utc,
        base_force_close_limit=st_base_force_close,
        next_limit_if_extended=st_next_limit,
        revalidation_attempted=st_revalidation_attempted,
        extension_active=st_extension_active,
        close_reason=st_close_reason,
    )
    @settings(max_examples=200)
    def test_payload_field_values_match_inputs(
        self,
        trade_id,
        setup_type,
        event_type,
        minutes_held,
        decision_outcome,
        invalidation_criteria_used,
        timestamp_utc,
        base_force_close_limit,
        next_limit_if_extended,
        revalidation_attempted,
        extension_active,
        close_reason,
    ):
        """Payload field values faithfully reflect the inputs provided."""
        payload = build_setup_exit_event_payload(
            trade_id=trade_id,
            setup_type=setup_type,
            event_type=event_type,
            minutes_held=minutes_held,
            decision_outcome=decision_outcome,
            invalidation_criteria_used=invalidation_criteria_used,
            timestamp_utc=timestamp_utc,
            base_force_close_limit=base_force_close_limit,
            next_limit_if_extended=next_limit_if_extended,
            revalidation_attempted=revalidation_attempted,
            extension_active=extension_active,
            close_reason=close_reason,
        )

        assert payload["trade_id"] == trade_id
        assert payload["setup_type"] == setup_type
        assert payload["event_type"] == event_type
        assert payload["minutes_held"] == minutes_held
        assert payload["decision_outcome"] == decision_outcome
        assert payload["invalidation_criteria_used"] == invalidation_criteria_used
        assert payload["timestamp_utc"] == timestamp_utc.isoformat()
        assert payload["base_force_close_limit"] == base_force_close_limit
        assert payload["next_limit_if_extended"] == next_limit_if_extended
        assert payload["revalidation_attempted"] == revalidation_attempted
        assert payload["extension_active"] == extension_active
        assert payload["close_reason"] == close_reason
        assert isinstance(payload["dedupe_key"], str)
        assert len(payload["dedupe_key"]) > 0


# ---------------------------------------------------------------------------
# Property 20: Entry metadata validation for thesis-development setups
# Feature: setup-aware-exit-governance, Property 20: Entry metadata validation for thesis-development setups
# ---------------------------------------------------------------------------


class TestProperty20EntryMetadataValidationThesisDevelopment:
    """
    For any thesis-development setup (news_breakout, news_catalyst, trend_pullback)
    with all required fields present (entry_price, stop_price, target_price, thesis,
    invalidation_basis), validate_entry_for_exit_governance returns full_eligibility.
    For the same setups missing both stop_price and invalidation_basis, it returns
    "reject".

    **Validates: Requirements 5.2, 5.3, 5.4**
    """

    @given(
        setup_type=st_thesis_development_setup,
        entry_price=st_price,
        stop_price=st_price,
        target_price=st_price,
        thesis=st.text(min_size=5, max_size=200),
        invalidation_basis=st.text(min_size=5, max_size=200),
    )
    @settings(max_examples=200)
    def test_full_metadata_yields_full_eligibility(
        self, setup_type, entry_price, stop_price, target_price, thesis, invalidation_basis,
    ):
        """Thesis-development entry with all required fields → full_eligibility."""
        # Ensure stop_price is non-zero (validator requires valid number)
        assume(stop_price != 0.0)
        assume(entry_price != 0.0)
        assume(target_price != 0.0)

        entry = {
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "thesis": thesis,
            "invalidation_basis": invalidation_basis,
        }

        is_valid, eligibility_status, reason = validate_entry_for_exit_governance(
            entry, setup_type
        )

        assert eligibility_status == "full_eligibility", (
            f"Expected full_eligibility for {setup_type} with all fields present, "
            f"got '{eligibility_status}'. Reason: {reason}. "
            f"Entry: entry_price={entry_price}, stop_price={stop_price}, "
            f"target_price={target_price}, thesis='{thesis[:30]}...', "
            f"invalidation_basis='{invalidation_basis[:30]}...'"
        )
        assert is_valid is True

    @given(
        setup_type=st.sampled_from(["news_breakout", "news_catalyst"]),
        entry_price=st_price,
        target_price=st.one_of(st_price, st.none()),
        thesis=st.one_of(st.text(min_size=5, max_size=200), st.none()),
    )
    @settings(max_examples=200)
    def test_news_setup_missing_both_stop_and_invalidation_yields_reject(
        self, setup_type, entry_price, target_price, thesis,
    ):
        """news_breakout/news_catalyst missing both stop_price AND invalidation_basis → reject."""
        assume(entry_price != 0.0)

        entry = {
            "entry_price": entry_price,
            # No stop_price, no stop_loss
            # No invalidation_basis
        }
        if target_price is not None:
            entry["target_price"] = target_price
        if thesis is not None:
            entry["thesis"] = thesis

        is_valid, eligibility_status, reason = validate_entry_for_exit_governance(
            entry, setup_type
        )

        assert eligibility_status == "reject", (
            f"Expected reject for {setup_type} missing both stop_price and "
            f"invalidation_basis, got '{eligibility_status}'. Reason: {reason}"
        )
        assert is_valid is False

    @given(
        entry_price=st_price,
        target_price=st.one_of(st_price, st.none()),
        thesis=st.one_of(st.text(min_size=5, max_size=200), st.none()),
    )
    @settings(max_examples=200)
    def test_trend_pullback_missing_stop_yields_reject(
        self, entry_price, target_price, thesis,
    ):
        """trend_pullback missing stop_price → reject."""
        assume(entry_price != 0.0)

        entry = {
            "entry_price": entry_price,
            # No stop_price, no stop_loss
        }
        if target_price is not None:
            entry["target_price"] = target_price
        if thesis is not None:
            entry["thesis"] = thesis

        is_valid, eligibility_status, reason = validate_entry_for_exit_governance(
            entry, "trend_pullback"
        )

        assert eligibility_status == "reject", (
            f"Expected reject for trend_pullback missing stop_price, "
            f"got '{eligibility_status}'. Reason: {reason}"
        )
        assert is_valid is False


# ---------------------------------------------------------------------------
# Property 21: Event dedupe via deterministic key
# Feature: setup-aware-exit-governance, Property 21: Event dedupe via deterministic key
# ---------------------------------------------------------------------------


class TestProperty21EventDedupeViaDeterministicKey:
    """
    For any two calls to generate_dedupe_key with the same event_type, trade_id,
    and timestamp within the same minute, the keys are identical. For calls with
    different minutes, the keys differ.

    **Validates: Requirements 6.3**
    """

    @given(
        event_type=st_event_type,
        trade_id=st_trade_id,
        base_timestamp=st_timestamp_utc,
        seconds_offset=st.integers(min_value=0, max_value=59),
    )
    @settings(max_examples=200)
    def test_same_minute_produces_identical_keys(
        self, event_type, trade_id, base_timestamp, seconds_offset,
    ):
        """Two timestamps within the same minute produce the same dedupe key."""
        # Normalize base_timestamp to start of minute
        ts1 = base_timestamp.replace(second=0, microsecond=0)
        # ts2 is within the same minute (different seconds)
        ts2 = ts1.replace(second=seconds_offset, microsecond=123456)

        key1 = generate_dedupe_key(event_type, trade_id, ts1)
        key2 = generate_dedupe_key(event_type, trade_id, ts2)

        assert key1 == key2, (
            f"Expected identical dedupe keys for same minute. "
            f"ts1={ts1}, ts2={ts2}, key1='{key1}', key2='{key2}'"
        )

    @given(
        event_type=st_event_type,
        trade_id=st_trade_id,
        base_timestamp=st_timestamp_utc,
        minute_offset=st.integers(min_value=1, max_value=500),
    )
    @settings(max_examples=200)
    def test_different_minutes_produce_different_keys(
        self, event_type, trade_id, base_timestamp, minute_offset,
    ):
        """Two timestamps in different minutes produce different dedupe keys."""
        ts1 = base_timestamp
        ts2 = base_timestamp + timedelta(minutes=minute_offset)

        # Ensure they are actually in different minutes after truncation
        ts1_truncated = ts1.replace(second=0, microsecond=0)
        ts2_truncated = ts2.replace(second=0, microsecond=0)
        assume(ts1_truncated != ts2_truncated)

        key1 = generate_dedupe_key(event_type, trade_id, ts1)
        key2 = generate_dedupe_key(event_type, trade_id, ts2)

        assert key1 != key2, (
            f"Expected different dedupe keys for different minutes. "
            f"ts1={ts1}, ts2={ts2}, key1='{key1}', key2='{key2}'"
        )

    @given(
        event_type=st_event_type,
        trade_id=st_trade_id,
        timestamp_utc=st_timestamp_utc,
    )
    @settings(max_examples=200)
    def test_dedupe_key_contains_expected_components(
        self, event_type, trade_id, timestamp_utc,
    ):
        """Dedupe key format is '{event_type}:{trade_id}:{timestamp_truncated_to_minute}'."""
        key = generate_dedupe_key(event_type, trade_id, timestamp_utc)

        # Key should contain the event_type and trade_id
        parts = key.split(":")
        # Format: "event_type:trade_id:YYYY-MM-DDTHH:MM"
        # Note: the timestamp contains a colon in HH:MM, so we expect 4 parts
        assert len(parts) >= 3, (
            f"Dedupe key should have at least 3 colon-separated parts, "
            f"got {len(parts)}: '{key}'"
        )
        assert parts[0] == event_type, (
            f"First part of dedupe key should be event_type '{event_type}', "
            f"got '{parts[0]}'"
        )
        assert parts[1] == str(trade_id), (
            f"Second part of dedupe key should be trade_id '{trade_id}', "
            f"got '{parts[1]}'"
        )
