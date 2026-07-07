"""Property-based tests for utils/swing_geometry_builder.py.

Uses Hypothesis to validate universal correctness properties of the swing
geometry builder's pure-function validation and construction logic.
"""

from __future__ import annotations

from decimal import Decimal

from hypothesis import assume, given, settings, strategies as st

from utils.swing_geometry_builder import (
    GeometryRejection,
    SwingGeometry,
    build_swing_geometry,
    SETUP_HOLDING_HORIZONS,
    SWING_GEOMETRY_MIN_RR,
)

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

profile_st = st.sampled_from(["conservative", "moderate", "aggressive"])

setup_type_st = st.sampled_from(list(SETUP_HOLDING_HORIZONS.keys()))

price_st = st.decimals(min_value=Decimal("1.00"), max_value=Decimal("10000.00"), places=2)


# ---------------------------------------------------------------------------
# Property 12: Stop Too Tight Rejection
# Validates: Requirements 3.5
# ---------------------------------------------------------------------------


@given(
    entry=st.decimals(min_value=Decimal("10.00"), max_value=Decimal("1000.00"), places=2),
    profile_id=profile_st,
    setup_type=setup_type_st,
)
@settings(max_examples=200)
def test_stop_too_tight_rejection(entry, profile_id, setup_type):
    """Property 12: Stop within 1.5% of entry → stop_too_tight_for_swing.

    **Validates: Requirements 3.5**
    """
    # Generate a stop that's too close (within 1.0% of entry for safety margin)
    tight_offset = entry * Decimal("0.005")  # 0.5% — definitely < 1.5%
    if tight_offset == 0:
        tight_offset = Decimal("0.01")
    stop = entry - tight_offset  # LONG direction, stop just below entry
    target = entry + entry * Decimal("0.10")  # 10% target (doesn't matter, won't reach this check)

    assume(stop > 0)
    assume(stop != entry)

    result = build_swing_geometry(
        symbol="TEST",
        direction="LONG",
        normalized_setup_type=setup_type,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        source_signal_id="test-signal",
        profile_id=profile_id,
    )

    assert isinstance(result, GeometryRejection)
    assert result.reason_code == "stop_too_tight_for_swing"



# ---------------------------------------------------------------------------
# Property 10: Geometry Holding Horizon Bounds
# Validates: Requirements 3.4
# ---------------------------------------------------------------------------


@given(
    symbol=st.text(min_size=1, max_size=5, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    direction=st.sampled_from(["LONG", "SHORT"]),
    setup_type=setup_type_st,
    entry=price_st,
    stop=price_st,
    target=price_st,
    profile_id=profile_st,
)
@settings(max_examples=200)
def test_geometry_holding_horizon_bounds(
    symbol, direction, setup_type, entry, stop, target, profile_id
):
    """Property 10: holding_horizon within [2, 10] and setup-specific range.

    **Validates: Requirements 3.4**
    """
    assume(entry != stop)
    assume(entry != target)
    assume(stop != target)

    result = build_swing_geometry(
        symbol=symbol,
        direction=direction,
        normalized_setup_type=setup_type,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        source_signal_id="test-signal",
        profile_id=profile_id,
    )

    if isinstance(result, SwingGeometry):
        # Global bounds
        assert 2 <= result.holding_horizon <= 10
        # Check setup-specific sub-range
        if setup_type in SETUP_HOLDING_HORIZONS:
            min_h, max_h = SETUP_HOLDING_HORIZONS[setup_type]
            assert min_h <= result.holding_horizon <= max_h
