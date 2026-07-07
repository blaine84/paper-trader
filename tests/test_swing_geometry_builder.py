"""Unit tests for swing geometry builder.

Validates Requirements 3.5, 3.8, 3.9, 3.10, 3.11.
"""
from __future__ import annotations

import pytest
from decimal import Decimal

from utils.swing_geometry_builder import (
    build_swing_geometry,
    GeometryRejection,
    SwingGeometry,
    SETUP_HOLDING_HORIZONS,
)


def _build(
    entry="100",
    stop="95",
    target="110",
    direction="LONG",
    setup_type="breakout_retest",
    profile_id="moderate",
    symbol="AAPL",
    signal_id="sig-1",
    stop_none=False,
):
    """Helper to build geometry with Decimal conversion and sensible defaults."""
    return build_swing_geometry(
        symbol=symbol,
        direction=direction,
        normalized_setup_type=setup_type,
        entry_price=Decimal(entry),
        stop_price=None if stop_none else Decimal(stop),
        target_price=Decimal(target),
        source_signal_id=signal_id,
        profile_id=profile_id,
    )


class TestZeroRiskDistance:
    """Requirement 3.11: stop == entry produces zero_risk_distance rejection."""

    def test_stop_equals_entry(self):
        result = build_swing_geometry(
            symbol="AAPL",
            direction="LONG",
            normalized_setup_type="breakout_retest",
            entry_price=Decimal("100"),
            stop_price=Decimal("100"),
            target_price=Decimal("110"),
            source_signal_id="sig-1",
            profile_id="moderate",
        )
        assert isinstance(result, GeometryRejection)
        assert result.reason_code == "zero_risk_distance"


class TestNonFinitePrices:
    """Requirement 3.8: NaN and Infinity produce non_finite_price rejection."""

    def test_nan_entry(self):
        result = build_swing_geometry(
            symbol="AAPL",
            direction="LONG",
            normalized_setup_type="breakout_retest",
            entry_price=Decimal("NaN"),
            stop_price=Decimal("95"),
            target_price=Decimal("110"),
            source_signal_id="sig-1",
            profile_id="moderate",
        )
        assert isinstance(result, GeometryRejection)
        assert result.reason_code == "non_finite_price"

    def test_infinity_stop(self):
        result = build_swing_geometry(
            symbol="AAPL",
            direction="LONG",
            normalized_setup_type="breakout_retest",
            entry_price=Decimal("100"),
            stop_price=Decimal("Infinity"),
            target_price=Decimal("110"),
            source_signal_id="sig-1",
            profile_id="moderate",
        )
        assert isinstance(result, GeometryRejection)
        assert result.reason_code == "non_finite_price"


class TestZeroPrice:
    """Requirement 3.8: zero price produces zero_price rejection."""

    def test_zero_entry(self):
        result = build_swing_geometry(
            symbol="AAPL",
            direction="LONG",
            normalized_setup_type="breakout_retest",
            entry_price=Decimal("0"),
            stop_price=Decimal("95"),
            target_price=Decimal("110"),
            source_signal_id="sig-1",
            profile_id="moderate",
        )
        assert isinstance(result, GeometryRejection)
        assert result.reason_code == "zero_price"


class TestNegativePrice:
    """Requirement 3.8: negative price produces negative_price rejection."""

    def test_negative_target(self):
        result = build_swing_geometry(
            symbol="AAPL",
            direction="LONG",
            normalized_setup_type="breakout_retest",
            entry_price=Decimal("100"),
            stop_price=Decimal("95"),
            target_price=Decimal("-5"),
            source_signal_id="sig-1",
            profile_id="moderate",
        )
        assert isinstance(result, GeometryRejection)
        assert result.reason_code == "negative_price"


class TestMissingGeometry:
    """Requirement 5.6: missing stop produces missing_geometry rejection."""

    def test_stop_is_none(self):
        result = build_swing_geometry(
            symbol="AAPL",
            direction="LONG",
            normalized_setup_type="breakout_retest",
            entry_price=Decimal("100"),
            stop_price=None,
            target_price=Decimal("110"),
            source_signal_id="sig-1",
            profile_id="moderate",
        )
        assert isinstance(result, GeometryRejection)
        assert result.reason_code == "missing_geometry"


class TestKnownRiskReward:
    """Requirement 3.6: R:R = abs(target - entry) / abs(entry - stop)."""

    def test_long_rr_3_to_1(self):
        # entry=100, stop=95, target=115 → reward=15, risk=5, R:R=3.0
        result = build_swing_geometry(
            symbol="AAPL",
            direction="LONG",
            normalized_setup_type="breakout_retest",
            entry_price=Decimal("100"),
            stop_price=Decimal("95"),
            target_price=Decimal("115"),
            source_signal_id="sig-1",
            profile_id="moderate",
        )
        assert isinstance(result, SwingGeometry)
        assert result.risk_reward == Decimal("3")

    def test_short_rr_3_to_1(self):
        # entry=100, stop=105, target=85 → reward=15, risk=5, R:R=3.0
        result = build_swing_geometry(
            symbol="AAPL",
            direction="SHORT",
            normalized_setup_type="risk_off_macro_short",
            entry_price=Decimal("100"),
            stop_price=Decimal("105"),
            target_price=Decimal("85"),
            source_signal_id="sig-1",
            profile_id="moderate",
        )
        assert isinstance(result, SwingGeometry)
        assert result.risk_reward == Decimal("3")


class TestInvalidDirectionalOrder:
    """Requirement 3.9: directional ordering must hold (LONG: stop < entry < target)."""

    def test_long_stop_above_entry(self):
        # LONG with stop > entry → invalid_directional_order
        result = build_swing_geometry(
            symbol="AAPL",
            direction="LONG",
            normalized_setup_type="breakout_retest",
            entry_price=Decimal("100"),
            stop_price=Decimal("105"),
            target_price=Decimal("115"),
            source_signal_id="sig-1",
            profile_id="moderate",
        )
        assert isinstance(result, GeometryRejection)
        assert result.reason_code == "invalid_directional_order"

    def test_short_stop_below_entry(self):
        # SHORT with stop < entry → invalid_directional_order
        result = build_swing_geometry(
            symbol="AAPL",
            direction="SHORT",
            normalized_setup_type="risk_off_macro_short",
            entry_price=Decimal("100"),
            stop_price=Decimal("95"),
            target_price=Decimal("85"),
            source_signal_id="sig-1",
            profile_id="moderate",
        )
        assert isinstance(result, GeometryRejection)
        assert result.reason_code == "invalid_directional_order"


class TestHoldingHorizonPerSetupType:
    """Requirement 3.4: holding horizon is midpoint of per-setup range."""

    @pytest.mark.parametrize(
        "setup_type,expected_horizon",
        [
            ("sector_rotation_swing", 5),       # (3+8)//2 = 5
            ("risk_off_macro_short", 3),        # (2+5)//2 = 3
            ("breakout_retest", 3),             # (2+5)//2 = 3
            ("pullback_continuation", 5),       # (3+7)//2 = 5
            ("relative_strength_swing", 7),     # (4+10)//2 = 7
            ("support_bounce_swing", 3),        # (2+5)//2 = 3
            ("failed_breakdown_reclaim", 3),    # (2+4)//2 = 3
        ],
    )
    def test_holding_horizon_midpoint(self, setup_type, expected_horizon):
        # Use direction and prices that satisfy directional ordering.
        # For risk_off_macro_short, direction is SHORT.
        if setup_type == "risk_off_macro_short":
            direction = "SHORT"
            entry, stop, target = "100", "105", "85"
        else:
            direction = "LONG"
            entry, stop, target = "100", "95", "115"

        result = build_swing_geometry(
            symbol="AAPL",
            direction=direction,
            normalized_setup_type=setup_type,
            entry_price=Decimal(entry),
            stop_price=Decimal(stop),
            target_price=Decimal(target),
            source_signal_id="sig-1",
            profile_id="moderate",
        )
        assert isinstance(result, SwingGeometry), (
            f"Expected SwingGeometry for {setup_type}, got {result}"
        )
        assert result.holding_horizon == expected_horizon


class TestInsufficientRiskReward:
    """Requirement 3.7: R:R below profile minimum → insufficient_risk_reward."""

    def test_aggressive_profile_rr_below_minimum(self):
        # entry=100, stop=95, target=102 → reward=2, risk=5, R:R=0.4
        # aggressive minimum is 1.25
        result = build_swing_geometry(
            symbol="AAPL",
            direction="LONG",
            normalized_setup_type="breakout_retest",
            entry_price=Decimal("100"),
            stop_price=Decimal("95"),
            target_price=Decimal("102"),
            source_signal_id="sig-1",
            profile_id="aggressive",
        )
        assert isinstance(result, GeometryRejection)
        assert result.reason_code == "insufficient_risk_reward"
        assert "risk_reward" in result.computed_values
        assert Decimal(result.computed_values["risk_reward"]) < Decimal("1.25")
