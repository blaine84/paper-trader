"""Tests for utils/entry_zone.py — EntryZone derivation and price helpers.

Validates: derive_entry_zone(), is_price_in_zone(), is_price_past_target()
Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from utils.entry_zone import (
    EntryZone,
    derive_entry_zone,
    is_price_in_zone,
    is_price_past_target,
    DEFAULT_ZONE_FRACTION,
)


# ---------------------------------------------------------------------------
# derive_entry_zone — LONG
# ---------------------------------------------------------------------------


class TestDeriveEntryZoneLong:
    """LONG zone: upper = entry_reference, lower = entry - (entry - stop) * fraction."""

    def test_basic_long_zone(self):
        """LONG with entry=100, stop=95 → upper=100, lower=99 (fraction=0.20)."""
        zone = derive_entry_zone(Decimal("100"), "BUY", Decimal("95"))
        assert zone.upper == Decimal("100")
        # lower = 100 - (100 - 95) * 0.20 = 100 - 1 = 99
        assert zone.lower == Decimal("99.00")
        assert zone.reference == Decimal("100")

    def test_long_zone_wider_stop(self):
        """LONG with wider stop distance produces wider zone."""
        zone = derive_entry_zone(Decimal("200"), "BUY", Decimal("180"))
        # lower = 200 - (200 - 180) * 0.20 = 200 - 4 = 196
        assert zone.upper == Decimal("200")
        assert zone.lower == Decimal("196.00")

    def test_long_zone_tight_stop(self):
        """LONG with very tight stop produces narrow zone."""
        zone = derive_entry_zone(Decimal("50"), "BUY", Decimal("49"))
        # lower = 50 - (50 - 49) * 0.20 = 50 - 0.20 = 49.80
        assert zone.upper == Decimal("50")
        assert zone.lower == Decimal("49.80")

    def test_long_zone_custom_fraction(self):
        """LONG with custom zone_fraction=0.50."""
        zone = derive_entry_zone(
            Decimal("100"), "BUY", Decimal("90"), zone_fraction=Decimal("0.50")
        )
        # lower = 100 - (100 - 90) * 0.50 = 100 - 5 = 95
        assert zone.upper == Decimal("100")
        assert zone.lower == Decimal("95.00")


# ---------------------------------------------------------------------------
# derive_entry_zone — SHORT
# ---------------------------------------------------------------------------


class TestDeriveEntryZoneShort:
    """SHORT zone: lower = entry_reference, upper = entry + (stop - entry) * fraction."""

    def test_basic_short_zone(self):
        """SHORT with entry=100, stop=105 → lower=100, upper=101."""
        zone = derive_entry_zone(Decimal("100"), "SHORT", Decimal("105"))
        assert zone.lower == Decimal("100")
        # upper = 100 + (105 - 100) * 0.20 = 100 + 1 = 101
        assert zone.upper == Decimal("101.00")
        assert zone.reference == Decimal("100")

    def test_short_zone_wider_stop(self):
        """SHORT with wider stop distance produces wider zone."""
        zone = derive_entry_zone(Decimal("200"), "SHORT", Decimal("220"))
        # upper = 200 + (220 - 200) * 0.20 = 200 + 4 = 204
        assert zone.lower == Decimal("200")
        assert zone.upper == Decimal("204.00")

    def test_short_zone_custom_fraction(self):
        """SHORT with custom zone_fraction=0.30."""
        zone = derive_entry_zone(
            Decimal("100"), "SHORT", Decimal("110"), zone_fraction=Decimal("0.30")
        )
        # upper = 100 + (110 - 100) * 0.30 = 100 + 3 = 103
        assert zone.lower == Decimal("100")
        assert zone.upper == Decimal("103.00")


# ---------------------------------------------------------------------------
# derive_entry_zone — tolerance_pct
# ---------------------------------------------------------------------------


class TestDeriveEntryZoneTolerance:
    """Tolerance is stored on the zone but NOT baked into bounds."""

    def test_default_tolerance_from_gate_config(self):
        """Default tolerance_pct comes from PLAN_ENTRY_ZONE_TOLERANCE_PCT (0.005)."""
        zone = derive_entry_zone(Decimal("100"), "BUY", Decimal("95"))
        assert zone.tolerance_pct == Decimal("0.005")

    def test_custom_tolerance_override(self):
        """Custom tolerance_pct is stored on the zone."""
        zone = derive_entry_zone(
            Decimal("100"), "BUY", Decimal("95"), tolerance_pct=Decimal("0.01")
        )
        assert zone.tolerance_pct == Decimal("0.01")

    def test_tolerance_not_in_bounds(self):
        """Zone bounds are the same regardless of tolerance_pct."""
        zone_default = derive_entry_zone(Decimal("100"), "BUY", Decimal("95"))
        zone_custom = derive_entry_zone(
            Decimal("100"), "BUY", Decimal("95"), tolerance_pct=Decimal("0.05")
        )
        assert zone_default.upper == zone_custom.upper
        assert zone_default.lower == zone_custom.lower


# ---------------------------------------------------------------------------
# is_price_in_zone
# ---------------------------------------------------------------------------


class TestIsPriceInZone:
    """Price-in-zone check applies tolerance at evaluation time."""

    @pytest.fixture
    def long_zone(self):
        """LONG zone: upper=100, lower=99, reference=100, tolerance=0.005."""
        return derive_entry_zone(Decimal("100"), "BUY", Decimal("95"))

    @pytest.fixture
    def short_zone(self):
        """SHORT zone: lower=100, upper=101, reference=100, tolerance=0.005."""
        return derive_entry_zone(Decimal("100"), "SHORT", Decimal("105"))

    def test_price_inside_zone_long(self, long_zone):
        """Price clearly inside zone returns True."""
        assert is_price_in_zone(Decimal("99.5"), long_zone, "BUY") is True

    def test_price_at_upper_bound_long(self, long_zone):
        """Price at upper bound is inside."""
        assert is_price_in_zone(Decimal("100"), long_zone, "BUY") is True

    def test_price_at_lower_bound_long(self, long_zone):
        """Price at lower bound is inside."""
        assert is_price_in_zone(Decimal("99"), long_zone, "BUY") is True

    def test_price_within_tolerance_above_upper(self, long_zone):
        """Price above upper but within tolerance is inside.
        tolerance = 100 * 0.005 = 0.5 → effective upper = 100.5
        """
        assert is_price_in_zone(Decimal("100.4"), long_zone, "BUY") is True

    def test_price_within_tolerance_below_lower(self, long_zone):
        """Price below lower but within tolerance is inside.
        tolerance = 100 * 0.005 = 0.5 → effective lower = 98.5
        """
        assert is_price_in_zone(Decimal("98.6"), long_zone, "BUY") is True

    def test_price_at_tolerance_boundary_lower(self, long_zone):
        """Price exactly at effective lower boundary (98.5) is inside."""
        assert is_price_in_zone(Decimal("98.5"), long_zone, "BUY") is True

    def test_price_at_tolerance_boundary_upper(self, long_zone):
        """Price exactly at effective upper boundary (100.5) is inside."""
        assert is_price_in_zone(Decimal("100.5"), long_zone, "BUY") is True

    def test_price_below_tolerance_long(self, long_zone):
        """Price below effective lower boundary is outside."""
        assert is_price_in_zone(Decimal("98.4"), long_zone, "BUY") is False

    def test_price_above_tolerance_long(self, long_zone):
        """Price above effective upper boundary is outside."""
        assert is_price_in_zone(Decimal("100.6"), long_zone, "BUY") is False

    def test_price_inside_zone_short(self, short_zone):
        """Short: price inside zone returns True."""
        assert is_price_in_zone(Decimal("100.5"), short_zone, "SHORT") is True

    def test_price_above_tolerance_short(self, short_zone):
        """Short: price above effective upper is outside.
        upper=101, tolerance=0.5 → effective upper = 101.5
        """
        assert is_price_in_zone(Decimal("101.6"), short_zone, "SHORT") is False

    def test_price_below_tolerance_short(self, short_zone):
        """Short: price below effective lower is outside.
        lower=100, tolerance=0.5 → effective lower = 99.5
        """
        assert is_price_in_zone(Decimal("99.4"), short_zone, "SHORT") is False

    def test_zero_tolerance(self):
        """With tolerance_pct=0, zone uses exact bounds."""
        zone = derive_entry_zone(
            Decimal("100"), "BUY", Decimal("95"), tolerance_pct=Decimal("0")
        )
        # Exact bounds: [99, 100]
        assert is_price_in_zone(Decimal("99"), zone, "BUY") is True
        assert is_price_in_zone(Decimal("100"), zone, "BUY") is True
        assert is_price_in_zone(Decimal("98.99"), zone, "BUY") is False
        assert is_price_in_zone(Decimal("100.01"), zone, "BUY") is False


# ---------------------------------------------------------------------------
# is_price_past_target
# ---------------------------------------------------------------------------


class TestIsPricePastTarget:
    """Price-past-target for LONG and SHORT directions."""

    def test_long_price_above_target(self):
        """LONG: price >= target is past target."""
        assert is_price_past_target(Decimal("110"), Decimal("105"), "BUY") is True

    def test_long_price_at_target(self):
        """LONG: price exactly at target is past target."""
        assert is_price_past_target(Decimal("105"), Decimal("105"), "BUY") is True

    def test_long_price_below_target(self):
        """LONG: price below target is NOT past target."""
        assert is_price_past_target(Decimal("104.99"), Decimal("105"), "BUY") is False

    def test_short_price_below_target(self):
        """SHORT: price <= target is past target."""
        assert is_price_past_target(Decimal("90"), Decimal("95"), "SHORT") is True

    def test_short_price_at_target(self):
        """SHORT: price exactly at target is past target."""
        assert is_price_past_target(Decimal("95"), Decimal("95"), "SHORT") is True

    def test_short_price_above_target(self):
        """SHORT: price above target is NOT past target."""
        assert is_price_past_target(Decimal("95.01"), Decimal("95"), "SHORT") is False


# ---------------------------------------------------------------------------
# EntryZone dataclass
# ---------------------------------------------------------------------------


class TestEntryZoneDataclass:
    """EntryZone is a frozen (immutable) dataclass."""

    def test_frozen(self):
        """Cannot mutate EntryZone fields."""
        zone = derive_entry_zone(Decimal("100"), "BUY", Decimal("95"))
        with pytest.raises(AttributeError):
            zone.upper = Decimal("999")

    def test_equality(self):
        """Two zones with same values are equal."""
        zone1 = derive_entry_zone(Decimal("100"), "BUY", Decimal("95"))
        zone2 = derive_entry_zone(Decimal("100"), "BUY", Decimal("95"))
        assert zone1 == zone2

    def test_default_zone_fraction_is_020(self):
        """Default zone fraction is 0.20."""
        assert DEFAULT_ZONE_FRACTION == Decimal("0.20")
