"""
Property-based tests for StopAuthority using Hypothesis.

Tests universal correctness properties that must hold across all valid inputs.
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from utils.stop_authority import (
    validate_stop_geometry,
    should_stop_trigger,
    StopValidationResult,
    StopTriggerResult,
    _is_in_profit,
    _compute_buffer,
)


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

st_side = st.sampled_from(["long", "short"])
st_price = st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False)
st_stop_role = st.sampled_from(["initial", "breakeven", "trail", "manual", "maintenance_tighten"])
st_protective_role = st.sampled_from(["initial", "manual"])
st_profit_role = st.sampled_from(["breakeven", "trail"])


# ---------------------------------------------------------------------------
# Property 1: Protective stop geometry invariant
# Feature: stop-geometry-authority, Property 1: Protective stop geometry invariant
# ---------------------------------------------------------------------------


class TestProperty1ProtectiveStopGeometryInvariant:
    """
    For any side and stop with role initial/manual, stop is valid iff strictly
    on loss side of both entry and current.

    **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
    """

    @given(
        side=st_side,
        entry_price=st_price,
        current_price=st_price,
        stop_price=st_price,
        stop_role=st_protective_role,
    )
    @settings(max_examples=200)
    def test_protective_stop_valid_iff_on_loss_side(
        self, side, entry_price, current_price, stop_price, stop_role
    ):
        """Protective stop is valid iff strictly on loss side of both entry and current."""
        result = validate_stop_geometry(
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            stop_price=stop_price,
            stop_role=stop_role,
        )

        if side == "long":
            # Valid iff stop < entry AND stop < current
            expected_valid = (stop_price < entry_price) and (stop_price < current_price)
        else:
            # Valid iff stop > entry AND stop > current
            expected_valid = (stop_price > entry_price) and (stop_price > current_price)

        assert result.valid == expected_valid, (
            f"side={side}, entry={entry_price}, current={current_price}, "
            f"stop={stop_price}, role={stop_role}: "
            f"expected valid={expected_valid}, got valid={result.valid}"
        )

    @given(
        side=st_side,
        entry_price=st_price,
        stop_price=st_price,
        stop_role=st_protective_role,
    )
    @settings(max_examples=200)
    def test_protective_stop_without_current_price(
        self, side, entry_price, stop_price, stop_role
    ):
        """Protective stop without current_price: valid iff on loss side of entry."""
        result = validate_stop_geometry(
            side=side,
            entry_price=entry_price,
            current_price=None,
            stop_price=stop_price,
            stop_role=stop_role,
        )

        if side == "long":
            expected_valid = stop_price < entry_price
        else:
            expected_valid = stop_price > entry_price

        assert result.valid == expected_valid, (
            f"side={side}, entry={entry_price}, stop={stop_price}, role={stop_role}: "
            f"expected valid={expected_valid}, got valid={result.valid}"
        )


# ---------------------------------------------------------------------------
# Property 2: Profit-protecting stop geometry invariant
# Feature: stop-geometry-authority, Property 2: Profit-protecting stop geometry invariant
# ---------------------------------------------------------------------------


class TestProperty2ProfitProtectingStopGeometryInvariant:
    """
    For any side and stop with role breakeven/trail, stop is valid iff trade is
    in profit AND stop maintains buffer_pct distance from current.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
    """

    @given(
        side=st_side,
        entry_price=st_price,
        current_price=st_price,
        stop_price=st_price,
        stop_role=st_profit_role,
    )
    @settings(max_examples=200)
    def test_profit_protecting_stop_valid_iff_in_profit_and_buffer_maintained(
        self, side, entry_price, current_price, stop_price, stop_role
    ):
        """Profit-protecting stop valid iff in profit AND buffer distance maintained."""
        result = validate_stop_geometry(
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            stop_price=stop_price,
            stop_role=stop_role,
            buffer_pct=0.001,
        )

        in_profit = _is_in_profit(side, entry_price, current_price)
        buffer = _compute_buffer(current_price, 0.001)

        if side == "long":
            # Valid iff in profit AND stop < current - buffer
            expected_valid = in_profit and (stop_price < current_price - buffer)
        else:
            # Valid iff in profit AND stop > current + buffer
            expected_valid = in_profit and (stop_price > current_price + buffer)

        assert result.valid == expected_valid, (
            f"side={side}, entry={entry_price}, current={current_price}, "
            f"stop={stop_price}, role={stop_role}, in_profit={in_profit}, "
            f"buffer={buffer}: expected valid={expected_valid}, got valid={result.valid}"
        )


# ---------------------------------------------------------------------------
# Property 3: maintenance_tighten conditional rule application
# Feature: stop-geometry-authority, Property 3: maintenance_tighten conditional rule application
# ---------------------------------------------------------------------------


class TestProperty3MaintenanceTightenConditionalRules:
    """
    For any trade with maintenance_tighten role, rules applied are profit-protecting
    when in profit, protective otherwise. Result must be identical to calling
    validate_stop_geometry with role breakeven (in profit) or initial (not in profit).

    **Validates: Requirements 5.8**
    """

    @given(
        side=st_side,
        entry_price=st_price,
        current_price=st_price,
        stop_price=st_price,
    )
    @settings(max_examples=200)
    def test_maintenance_tighten_matches_conditional_role(
        self, side, entry_price, current_price, stop_price
    ):
        """maintenance_tighten result matches breakeven (in profit) or initial (not in profit)."""
        # Get maintenance_tighten result
        mt_result = validate_stop_geometry(
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            stop_price=stop_price,
            stop_role="maintenance_tighten",
            buffer_pct=0.001,
        )

        # Determine which role should be equivalent
        in_profit = _is_in_profit(side, entry_price, current_price)

        if in_profit:
            equivalent_role = "breakeven"
        else:
            equivalent_role = "initial"

        equiv_result = validate_stop_geometry(
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            stop_price=stop_price,
            stop_role=equivalent_role,
            buffer_pct=0.001,
        )

        # The validity decision must be identical
        assert mt_result.valid == equiv_result.valid, (
            f"side={side}, entry={entry_price}, current={current_price}, "
            f"stop={stop_price}, in_profit={in_profit}: "
            f"maintenance_tighten valid={mt_result.valid} != "
            f"{equivalent_role} valid={equiv_result.valid}"
        )


# ---------------------------------------------------------------------------
# Property 4: Trigger correctness
# Feature: stop-geometry-authority, Property 4: Trigger correctness
# ---------------------------------------------------------------------------


class TestProperty4TriggerCorrectness:
    """
    should_stop_trigger returns triggered=True iff geometry valid AND price
    breaches buffered stop level. If geometry invalid, triggered is always False.

    **Validates: Requirements 4.2, 4.4, 4.5, 4.6**
    """

    @given(
        side=st_side,
        entry_price=st_price,
        current_price=st_price,
        stop_price=st_price,
        stop_role=st_stop_role,
    )
    @settings(max_examples=200)
    def test_trigger_correctness(
        self, side, entry_price, current_price, stop_price, stop_role
    ):
        """Trigger fires iff geometry valid AND price breaches buffered stop level."""
        result = should_stop_trigger(
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            stop_price=stop_price,
            stop_role=stop_role,
            buffer_pct=0.001,
        )

        if not result.geometry_valid:
            # If geometry invalid, triggered must always be False
            assert result.triggered is False, (
                f"Geometry invalid but triggered=True: side={side}, "
                f"entry={entry_price}, current={current_price}, stop={stop_price}"
            )
        else:
            # If geometry valid, check trigger condition
            buffer = _compute_buffer(stop_price, 0.001)
            if side == "long":
                expected_triggered = current_price <= (stop_price - buffer)
            else:
                expected_triggered = current_price >= (stop_price + buffer)

            assert result.triggered == expected_triggered, (
                f"side={side}, entry={entry_price}, current={current_price}, "
                f"stop={stop_price}, buffer={buffer}: "
                f"expected triggered={expected_triggered}, got triggered={result.triggered}"
            )

    @given(
        side=st_side,
        entry_price=st_price,
        current_price=st_price,
        stop_price=st_price,
    )
    @settings(max_examples=200)
    def test_invalid_geometry_never_triggers(
        self, side, entry_price, current_price, stop_price
    ):
        """When geometry is invalid (initial role), triggered is always False."""
        # Use initial role which has strict protective geometry
        result = should_stop_trigger(
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            stop_price=stop_price,
            stop_role="initial",
            buffer_pct=0.001,
        )

        if not result.geometry_valid:
            assert result.triggered is False


# ---------------------------------------------------------------------------
# Property 5: Role does not affect trigger direction
# Feature: stop-geometry-authority, Property 5: Role does not affect trigger direction
# ---------------------------------------------------------------------------


class TestProperty5RoleDoesNotAffectTriggerDirection:
    """
    For any valid geometry and two different stop_roles that both produce valid
    geometry for the same price configuration, trigger decision is identical.

    **Validates: Requirements 4.7**
    """

    @given(
        side=st_side,
        entry_price=st_price,
        current_price=st_price,
        stop_price=st_price,
        role_a=st_stop_role,
        role_b=st_stop_role,
    )
    @settings(max_examples=200)
    def test_role_does_not_affect_trigger_direction(
        self, side, entry_price, current_price, stop_price, role_a, role_b
    ):
        """Two roles with valid geometry produce identical trigger decisions."""
        assume(role_a != role_b)

        result_a = should_stop_trigger(
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            stop_price=stop_price,
            stop_role=role_a,
            buffer_pct=0.001,
        )

        result_b = should_stop_trigger(
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            stop_price=stop_price,
            stop_role=role_b,
            buffer_pct=0.001,
        )

        # Only compare when both have valid geometry
        if result_a.geometry_valid and result_b.geometry_valid:
            assert result_a.triggered == result_b.triggered, (
                f"side={side}, entry={entry_price}, current={current_price}, "
                f"stop={stop_price}: role_a={role_a} triggered={result_a.triggered} "
                f"!= role_b={role_b} triggered={result_b.triggered}"
            )


# ---------------------------------------------------------------------------
# Property 10: Validation result completeness
# Feature: stop-geometry-authority, Property 10: Validation result completeness
# ---------------------------------------------------------------------------


class TestProperty10ValidationResultCompleteness:
    """
    For any inputs, returned StopValidationResult always has non-None valid,
    reason_type, reason, side, stop_role; repair_price non-None when
    reason_type is "repair".

    **Validates: Requirements 1.4**
    """

    @given(
        side=st_side,
        entry_price=st_price,
        current_price=st_price,
        stop_price=st_price,
        stop_role=st_stop_role,
    )
    @settings(max_examples=200)
    def test_validation_result_completeness(
        self, side, entry_price, current_price, stop_price, stop_role
    ):
        """StopValidationResult always has required non-None fields."""
        result = validate_stop_geometry(
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            stop_price=stop_price,
            stop_role=stop_role,
        )

        # Required fields must be non-None
        assert result.valid is not None, "valid must not be None"
        assert result.reason_type is not None, "reason_type must not be None"
        assert result.reason is not None, "reason must not be None"
        assert result.reason != "", "reason must not be empty"
        assert result.side is not None, "side must not be None"
        assert result.side != "", "side must not be empty"
        assert result.stop_role is not None, "stop_role must not be None"
        assert result.stop_role != "", "stop_role must not be empty"

        # repair_price must be non-None when reason_type is "repair"
        if result.reason_type == "repair":
            assert result.repair_price is not None, (
                f"repair_price must not be None when reason_type='repair': "
                f"side={side}, entry={entry_price}, current={current_price}, "
                f"stop={stop_price}, role={stop_role}"
            )

    @given(
        side=st_side,
        entry_price=st_price,
        stop_price=st_price,
        stop_role=st_stop_role,
    )
    @settings(max_examples=200)
    def test_validation_result_completeness_without_current(
        self, side, entry_price, stop_price, stop_role
    ):
        """StopValidationResult completeness holds even without current_price."""
        result = validate_stop_geometry(
            side=side,
            entry_price=entry_price,
            current_price=None,
            stop_price=stop_price,
            stop_role=stop_role,
        )

        # Required fields must be non-None
        assert result.valid is not None, "valid must not be None"
        assert result.reason_type is not None, "reason_type must not be None"
        assert result.reason is not None, "reason must not be None"
        assert result.reason != "", "reason must not be empty"
        assert result.side is not None, "side must not be None"
        assert result.side != "", "side must not be empty"
        assert result.stop_role is not None, "stop_role must not be None"
        assert result.stop_role != "", "stop_role must not be empty"

        # repair_price must be non-None when reason_type is "repair"
        if result.reason_type == "repair":
            assert result.repair_price is not None, (
                f"repair_price must not be None when reason_type='repair'"
            )


# ---------------------------------------------------------------------------
# DB-based property tests (Properties 6-9)
# These tests require an in-memory SQLite database for each test example.
# ---------------------------------------------------------------------------

import json
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, Trade, TradeEvent
from utils.stop_authority import apply_stop_update, should_stop_trigger


def _fresh_db():
    """Create a fresh in-memory SQLite database with all tables."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    Session = sessionmaker(bind=eng)
    return Session()


# ---------------------------------------------------------------------------
# Property 6: Invalid stop never overwrites valid stop
# Feature: stop-geometry-authority, Property 6: Invalid stop never overwrites valid stop
# ---------------------------------------------------------------------------


class TestProperty6InvalidStopNeverOverwritesValidStop:
    """
    For any trade with a valid existing stop (per its current role and price state),
    calling apply_stop_update() with an invalid proposed stop SHALL leave
    trade.stop_price unchanged.

    **Validates: Requirements 6.3, 6.6**
    """

    @given(
        entry_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        current_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        stop_offset_pct=st.floats(min_value=0.005, max_value=0.20, allow_nan=False, allow_infinity=False),
        invalid_offset_pct=st.floats(min_value=0.001, max_value=0.50, allow_nan=False, allow_infinity=False),
        side=st_side,
    )
    @settings(max_examples=100)
    def test_invalid_proposed_stop_preserves_valid_existing_long_short(
        self, entry_price, current_price, stop_offset_pct, invalid_offset_pct, side
    ):
        """Invalid proposed stop does not overwrite a valid existing stop."""
        # Construct a valid existing stop for the trade
        if side == "long":
            # Valid protective stop: below both entry and current
            min_ref = min(entry_price, current_price)
            valid_stop = min_ref * (1 - stop_offset_pct)
            # Ensure current > entry so we have a sensible long trade
            assume(current_price > entry_price * 0.5)
            assume(valid_stop > 0.01)
            # Invalid proposed stop: above entry (violates long protective geometry)
            invalid_stop = entry_price * (1 + invalid_offset_pct)
        else:
            # Valid protective stop: above both entry and current
            max_ref = max(entry_price, current_price)
            valid_stop = max_ref * (1 + stop_offset_pct)
            # Ensure current < entry so we have a sensible short trade
            assume(current_price < entry_price * 1.5)
            assume(valid_stop < 100000)
            # Invalid proposed stop: below entry (violates short protective geometry)
            invalid_stop = entry_price * (1 - invalid_offset_pct)

        assume(valid_stop > 0)
        assume(invalid_stop > 0)

        # Verify the existing stop is actually valid
        existing_validation = validate_stop_geometry(
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            stop_price=valid_stop,
            stop_role="initial",
        )
        assume(existing_validation.valid)

        # Verify the proposed stop is actually invalid
        proposed_validation = validate_stop_geometry(
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            stop_price=invalid_stop,
            stop_role="initial",
        )
        assume(not proposed_validation.valid)

        # Set up DB and trade
        db = _fresh_db()
        trade = Trade(
            symbol="TEST",
            direction=side.upper(),
            quantity=100,
            entry_price=entry_price,
            stop_price=valid_stop,
            status="open",
            profile="moderate",
        )
        db.add(trade)
        db.flush()

        # Attempt the invalid stop update
        result = apply_stop_update(
            db,
            trade=trade,
            new_stop=invalid_stop,
            source_agent="test_agent",
            stop_role="initial",
            reason="Test invalid update",
            current_price=current_price,
        )

        # The result must be invalid
        assert result.valid is False, (
            f"Expected rejection but got valid=True: side={side}, "
            f"entry={entry_price}, current={current_price}, "
            f"valid_stop={valid_stop}, invalid_stop={invalid_stop}"
        )

        # The trade's stop_price must be unchanged
        assert trade.stop_price == valid_stop, (
            f"Trade stop_price changed from {valid_stop} to {trade.stop_price} "
            f"despite invalid proposed stop"
        )

        db.close()


# ---------------------------------------------------------------------------
# Property 7: Double-invalid emits review event
# Feature: stop-geometry-authority, Property 7: Double-invalid emits review event
# ---------------------------------------------------------------------------


class TestProperty7DoubleInvalidEmitsReviewEvent:
    """
    When both the existing stop and the proposed new stop are geometrically
    invalid, apply_stop_update() SHALL emit a stop_review_required event.

    **Validates: Requirements 6.4**
    """

    @given(
        entry_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        current_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        existing_offset_pct=st.floats(min_value=0.001, max_value=0.50, allow_nan=False, allow_infinity=False),
        proposed_offset_pct=st.floats(min_value=0.001, max_value=0.50, allow_nan=False, allow_infinity=False),
        side=st_side,
    )
    @settings(max_examples=100)
    def test_double_invalid_emits_review_required(
        self, entry_price, current_price, existing_offset_pct, proposed_offset_pct, side
    ):
        """Both existing and proposed stops invalid → stop_review_required event emitted."""
        # Construct invalid stops for both existing and proposed
        if side == "long":
            # Invalid for long: stop above entry
            existing_invalid_stop = entry_price * (1 + existing_offset_pct)
            proposed_invalid_stop = entry_price * (1 + proposed_offset_pct)
            # Ensure current is also above the stops so they're invalid vs current too
            assume(current_price <= existing_invalid_stop)
        else:
            # Invalid for short: stop below entry
            existing_invalid_stop = entry_price * (1 - existing_offset_pct)
            proposed_invalid_stop = entry_price * (1 - proposed_offset_pct)
            # Ensure current is also below the stops so they're invalid vs current too
            assume(current_price >= existing_invalid_stop)

        assume(existing_invalid_stop > 0)
        assume(proposed_invalid_stop > 0)

        # Verify both stops are actually invalid
        existing_val = validate_stop_geometry(
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            stop_price=existing_invalid_stop,
            stop_role="initial",
        )
        assume(not existing_val.valid)

        proposed_val = validate_stop_geometry(
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            stop_price=proposed_invalid_stop,
            stop_role="initial",
        )
        assume(not proposed_val.valid)

        # Set up DB and trade with invalid existing stop
        db = _fresh_db()
        trade = Trade(
            symbol="TEST",
            direction=side.upper(),
            quantity=100,
            entry_price=entry_price,
            stop_price=existing_invalid_stop,
            status="open",
            profile="moderate",
        )
        db.add(trade)
        db.flush()

        # Attempt the invalid stop update
        result = apply_stop_update(
            db,
            trade=trade,
            new_stop=proposed_invalid_stop,
            source_agent="test_agent",
            stop_role="initial",
            reason="Test double-invalid",
            current_price=current_price,
        )

        # Result should indicate review_required
        assert result.reason_type == "review_required", (
            f"Expected reason_type='review_required', got '{result.reason_type}': "
            f"side={side}, entry={entry_price}, current={current_price}, "
            f"existing_stop={existing_invalid_stop}, proposed_stop={proposed_invalid_stop}"
        )

        # Check that a stop_review_required event was emitted
        review_events = (
            db.query(TradeEvent)
            .filter(
                TradeEvent.trade_id == trade.id,
                TradeEvent.event_type == "stop_review_required",
            )
            .all()
        )
        assert len(review_events) == 1, (
            f"Expected exactly 1 stop_review_required event, got {len(review_events)}"
        )

        # Verify event payload contains required fields
        event = review_events[0]
        payload = json.loads(event.payload_json) if event.payload_json else {}
        assert payload.get("source_agent") == "test_agent"
        assert payload.get("proposed_stop") == proposed_invalid_stop
        assert payload.get("existing_stop") == existing_invalid_stop
        assert "side" in payload

        db.close()


# ---------------------------------------------------------------------------
# Property 8: Audit event completeness
# Feature: stop-geometry-authority, Property 8: Audit event completeness
# ---------------------------------------------------------------------------


class TestProperty8AuditEventCompleteness:
    """
    For any call to apply_stop_update(), exactly one stop_update_requested event
    is written, followed by exactly one of: stop_update_accepted, stop_update_rejected,
    or stop_review_required.

    **Validates: Requirements 7.1, 7.2, 7.3**
    """

    @given(
        entry_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        current_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        proposed_stop=st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
        side=st_side,
        stop_role=st_stop_role,
    )
    @settings(max_examples=100)
    def test_audit_event_completeness(
        self, entry_price, current_price, proposed_stop, side, stop_role
    ):
        """Every apply_stop_update call produces exactly 1 requested + 1 outcome event."""
        # Set up a trade with some existing stop (may be valid or invalid)
        if side == "long":
            existing_stop = entry_price * 0.95  # likely valid for long
        else:
            existing_stop = entry_price * 1.05  # likely valid for short

        assume(existing_stop > 0)

        db = _fresh_db()
        trade = Trade(
            symbol="TEST",
            direction=side.upper(),
            quantity=100,
            entry_price=entry_price,
            stop_price=existing_stop,
            status="open",
            profile="moderate",
        )
        db.add(trade)
        db.flush()

        # Call apply_stop_update
        apply_stop_update(
            db,
            trade=trade,
            new_stop=proposed_stop,
            source_agent="test_agent",
            stop_role=stop_role,
            reason="Test audit completeness",
            current_price=current_price,
        )

        # Query all events for this trade
        events = (
            db.query(TradeEvent)
            .filter(TradeEvent.trade_id == trade.id)
            .order_by(TradeEvent.id)
            .all()
        )

        event_types = [e.event_type for e in events]

        # Must have exactly one stop_update_requested
        requested_count = event_types.count("stop_update_requested")
        assert requested_count == 1, (
            f"Expected exactly 1 stop_update_requested, got {requested_count}. "
            f"Events: {event_types}"
        )

        # Must have exactly one outcome event
        outcome_types = {"stop_update_accepted", "stop_update_rejected", "stop_review_required"}
        outcome_events = [et for et in event_types if et in outcome_types]
        assert len(outcome_events) == 1, (
            f"Expected exactly 1 outcome event (accepted/rejected/review_required), "
            f"got {len(outcome_events)}: {outcome_events}. All events: {event_types}"
        )

        # The requested event should come before the outcome event
        requested_idx = event_types.index("stop_update_requested")
        outcome_idx = event_types.index(outcome_events[0])
        assert requested_idx < outcome_idx, (
            f"stop_update_requested (idx={requested_idx}) should precede "
            f"outcome event (idx={outcome_idx})"
        )

        # Verify source_agent is present in all events
        for event in events:
            if event.event_type in (outcome_types | {"stop_update_requested"}):
                payload = json.loads(event.payload_json) if event.payload_json else {}
                assert payload.get("source_agent") == "test_agent", (
                    f"Event {event.event_type} missing source_agent in payload"
                )

        db.close()


# ---------------------------------------------------------------------------
# Property 9: Geometry-invalid event dedupe
# Feature: stop-geometry-authority, Property 9: Geometry-invalid event dedupe
# ---------------------------------------------------------------------------


class TestProperty9GeometryInvalidEventDedupe:
    """
    For repeated should_stop_trigger() calls with the same (trade_id, source_agent,
    stop_value) within a 15-minute window, at most one stop_geometry_invalid event
    is written to trade_events.

    **Validates: Requirements 7.5**
    """

    @given(
        entry_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        current_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        invalid_offset_pct=st.floats(min_value=0.001, max_value=0.50, allow_nan=False, allow_infinity=False),
        num_calls=st.integers(min_value=2, max_value=10),
        side=st_side,
    )
    @settings(max_examples=100)
    def test_dedupe_within_window(
        self, entry_price, current_price, invalid_offset_pct, num_calls, side
    ):
        """Repeated calls with same params within 15min produce at most 1 event."""
        # Construct an invalid stop for the given side
        if side == "long":
            # Invalid for long initial: stop above entry
            invalid_stop = entry_price * (1 + invalid_offset_pct)
        else:
            # Invalid for short initial: stop below entry
            invalid_stop = entry_price * (1 - invalid_offset_pct)

        assume(invalid_stop > 0)

        # Verify the stop is actually invalid for initial role
        val = validate_stop_geometry(
            side=side,
            entry_price=entry_price,
            current_price=None,  # Use None to check against entry only
            stop_price=invalid_stop,
            stop_role="initial",
        )
        assume(not val.valid)

        # Set up DB and trade
        db = _fresh_db()
        trade = Trade(
            symbol="TEST",
            direction=side.upper(),
            quantity=100,
            entry_price=entry_price,
            stop_price=invalid_stop,
            status="open",
            profile="moderate",
        )
        db.add(trade)
        db.flush()

        # Call should_stop_trigger multiple times with same params
        for _ in range(num_calls):
            should_stop_trigger(
                side=side,
                entry_price=entry_price,
                current_price=current_price,
                stop_price=invalid_stop,
                stop_role="initial",
                db=db,
                trade_id=trade.id,
                symbol="TEST",
                profile="moderate",
                source_agent="price_monitor",
            )

        # Count stop_geometry_invalid events
        geometry_events = (
            db.query(TradeEvent)
            .filter(
                TradeEvent.trade_id == trade.id,
                TradeEvent.event_type == "stop_geometry_invalid",
            )
            .all()
        )

        assert len(geometry_events) <= 1, (
            f"Expected at most 1 stop_geometry_invalid event within 15min window, "
            f"got {len(geometry_events)} after {num_calls} calls"
        )

        # Verify at least one event was written (first call should produce one)
        assert len(geometry_events) == 1, (
            f"Expected exactly 1 stop_geometry_invalid event (first call), "
            f"got {len(geometry_events)}"
        )

        db.close()

    @given(
        entry_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        current_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        offset_pct_1=st.floats(min_value=0.01, max_value=0.30, allow_nan=False, allow_infinity=False),
        offset_pct_2=st.floats(min_value=0.01, max_value=0.30, allow_nan=False, allow_infinity=False),
        side=st_side,
    )
    @settings(max_examples=100)
    def test_different_stop_values_produce_separate_events(
        self, entry_price, current_price, offset_pct_1, offset_pct_2, side
    ):
        """Different stop values produce separate geometry_invalid events (no false dedupe)."""
        # Construct two different invalid stops
        if side == "long":
            invalid_stop_1 = entry_price * (1 + offset_pct_1)
            invalid_stop_2 = entry_price * (1 + offset_pct_2)
        else:
            invalid_stop_1 = entry_price * (1 - offset_pct_1)
            invalid_stop_2 = entry_price * (1 - offset_pct_2)

        # Ensure they are actually different values
        assume(abs(invalid_stop_1 - invalid_stop_2) > 0.001)
        assume(invalid_stop_1 > 0)
        assume(invalid_stop_2 > 0)

        # Verify both are invalid
        val1 = validate_stop_geometry(
            side=side, entry_price=entry_price, current_price=None,
            stop_price=invalid_stop_1, stop_role="initial",
        )
        val2 = validate_stop_geometry(
            side=side, entry_price=entry_price, current_price=None,
            stop_price=invalid_stop_2, stop_role="initial",
        )
        assume(not val1.valid)
        assume(not val2.valid)

        # Set up DB and trade
        db = _fresh_db()
        trade = Trade(
            symbol="TEST",
            direction=side.upper(),
            quantity=100,
            entry_price=entry_price,
            stop_price=invalid_stop_1,
            status="open",
            profile="moderate",
        )
        db.add(trade)
        db.flush()

        # Call with first stop value
        should_stop_trigger(
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            stop_price=invalid_stop_1,
            stop_role="initial",
            db=db,
            trade_id=trade.id,
            symbol="TEST",
            profile="moderate",
            source_agent="price_monitor",
        )

        # Call with second (different) stop value
        should_stop_trigger(
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            stop_price=invalid_stop_2,
            stop_role="initial",
            db=db,
            trade_id=trade.id,
            symbol="TEST",
            profile="moderate",
            source_agent="price_monitor",
        )

        # Should have 2 separate events (different stop values)
        geometry_events = (
            db.query(TradeEvent)
            .filter(
                TradeEvent.trade_id == trade.id,
                TradeEvent.event_type == "stop_geometry_invalid",
            )
            .all()
        )

        assert len(geometry_events) == 2, (
            f"Expected 2 stop_geometry_invalid events for different stop values, "
            f"got {len(geometry_events)}. stop_1={invalid_stop_1}, stop_2={invalid_stop_2}"
        )

        db.close()
