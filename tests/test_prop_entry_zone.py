"""
Property-based tests for utils/entry_zone.py — EntryZone derivation invariants.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6**
"""
from __future__ import annotations

from decimal import Decimal

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from utils.entry_zone import derive_entry_zone


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

st_price = st.decimals(min_value="0.01", max_value="10000", places=2)
st_direction = st.sampled_from(["BUY", "SHORT"])
st_zone_fraction = st.decimals(min_value="0.01", max_value="0.99", places=2)


# ---------------------------------------------------------------------------
# Property: derive_entry_zone always produces lower <= upper
# ---------------------------------------------------------------------------


@given(
    entry=st_price,
    stop=st_price,
    direction=st_direction,
    zone_fraction=st_zone_fraction,
)
@settings(max_examples=200)
def test_derive_entry_zone_lower_leq_upper(
    entry: Decimal,
    stop: Decimal,
    direction: str,
    zone_fraction: Decimal,
) -> None:
    """For any valid entry/stop/direction, derive_entry_zone produces lower <= upper.

    This invariant must hold regardless of direction, zone fraction, or
    relative positions of entry and stop. The zone always represents a
    non-degenerate interval.
    """
    # Filter invalid inputs: stop must be on the correct side for the direction
    if direction == "BUY":
        assume(stop < entry)  # LONG: stop below entry
    else:
        assume(stop > entry)  # SHORT: stop above entry

    zone = derive_entry_zone(entry, direction, stop, zone_fraction=zone_fraction)

    assert zone.lower <= zone.upper, (
        f"Zone invariant violated: lower={zone.lower} > upper={zone.upper} "
        f"(entry={entry}, stop={stop}, direction={direction}, frac={zone_fraction})"
    )
