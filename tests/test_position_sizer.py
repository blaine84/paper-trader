"""
Unit tests for utils/position_sizer.py

Tests the deterministic position sizing logic extracted from execute_trade().
Validates Requirements 4.1, 4.2, 4.3, 4.4.
"""

import math

import pytest

from utils.position_sizer import SizingResult, calculate_position_size


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class MockResolvedOrder:
    """Duck-typed resolved order for testing."""

    def __init__(
        self,
        entry_price: float = 100.0,
        stop_price: float = 95.0,
        target_price: float = 110.0,
        action: str = "BUY",
        risk_multiplier: float = 1.0,
        # Extra field that should be ignored
        quantity: int = 9999,
    ):
        self.entry_price = entry_price
        self.stop_price = stop_price
        self.target_price = target_price
        self.action = action
        self.risk_multiplier = risk_multiplier
        self.quantity = quantity


def make_portfolio(cash: float = 50_000.0, total_equity: float = 100_000.0) -> dict:
    return {"cash": cash, "total_equity": total_equity}


def make_profile(
    max_position_pct: float = 0.25,
    risk_per_trade_pct: float = 0.01,
) -> dict:
    return {
        "max_position_pct": max_position_pct,
        "risk_per_trade_pct": risk_per_trade_pct,
    }


# ---------------------------------------------------------------------------
# Standard sizing tests
# ---------------------------------------------------------------------------


class TestCalculatePositionSize:
    """Core tests for calculate_position_size."""

    def test_standard_long_sizing(self):
        """Basic BUY sizing: 1% of 100k equity / $5 stop distance = 200 shares."""
        order = MockResolvedOrder(entry_price=100.0, stop_price=95.0)
        portfolio = make_portfolio(total_equity=100_000.0)
        profile = make_profile(risk_per_trade_pct=0.01, max_position_pct=0.25)

        result = calculate_position_size(order, portfolio, profile, "moderate")

        assert result.rejection_reason is None
        # max_dollar_risk = 100000 * 0.01 = 1000
        # stop_distance = 5
        # quantity = floor(1000 / 5) = 200
        assert result.quantity == 200
        assert result.dollar_risk == 200 * 5.0  # 1000.0
        assert result.position_value == 200 * 100.0  # 20000.0
        assert result.sizing_method == "standard"
        assert result.applied_multiplier == 1.0

    def test_standard_short_sizing(self):
        """SHORT sizing uses abs(entry - stop) for stop distance."""
        order = MockResolvedOrder(
            entry_price=100.0, stop_price=105.0, action="SHORT"
        )
        portfolio = make_portfolio(total_equity=100_000.0)
        profile = make_profile(risk_per_trade_pct=0.01)

        result = calculate_position_size(order, portfolio, profile, "moderate")

        assert result.rejection_reason is None
        # stop_distance = abs(100 - 105) = 5
        # quantity = floor(1000 / 5) = 200
        assert result.quantity == 200
        assert result.dollar_risk == 1000.0

    def test_risk_multiplier_reduces_size(self):
        """risk_multiplier=0.5 halves the dollar risk, halving quantity."""
        order = MockResolvedOrder(
            entry_price=100.0, stop_price=95.0, risk_multiplier=0.5
        )
        portfolio = make_portfolio(total_equity=100_000.0)
        profile = make_profile(risk_per_trade_pct=0.01)

        result = calculate_position_size(order, portfolio, profile, "moderate")

        assert result.rejection_reason is None
        # max_dollar_risk = 1000 * 0.5 = 500
        # quantity = floor(500 / 5) = 100
        assert result.quantity == 100
        assert result.applied_multiplier == 0.5

    def test_recovery_multiplier_reduces_size(self):
        """recovery_multiplier < 1.0 further reduces dollar risk."""
        order = MockResolvedOrder(entry_price=100.0, stop_price=95.0)
        portfolio = make_portfolio(total_equity=100_000.0)
        profile = make_profile(risk_per_trade_pct=0.01)

        result = calculate_position_size(
            order, portfolio, profile, "moderate", recovery_multiplier=0.5
        )

        assert result.rejection_reason is None
        # max_dollar_risk = 1000 * 1.0 * 0.5 = 500
        # quantity = floor(500 / 5) = 100
        assert result.quantity == 100
        assert result.sizing_method == "recovery"
        assert result.applied_multiplier == 0.5

    def test_combined_risk_and_recovery_multipliers(self):
        """Both multipliers compound: risk_multiplier * recovery_multiplier."""
        order = MockResolvedOrder(
            entry_price=100.0, stop_price=95.0, risk_multiplier=0.5
        )
        portfolio = make_portfolio(total_equity=100_000.0)
        profile = make_profile(risk_per_trade_pct=0.01)

        result = calculate_position_size(
            order, portfolio, profile, "moderate", recovery_multiplier=0.5
        )

        assert result.rejection_reason is None
        # max_dollar_risk = 1000 * 0.5 * 0.5 = 250
        # quantity = floor(250 / 5) = 50
        assert result.quantity == 50
        assert result.sizing_method == "recovery"
        assert result.applied_multiplier == 0.25

    def test_max_position_pct_reduces_quantity(self):
        """When position value exceeds max_position_pct, quantity is capped."""
        # With default: 200 shares * $100 = $20,000 (20% of $100k) < 25% max
        # With tight max_position_pct=0.10: max = $10,000, so quantity = floor(10000/100) = 100
        order = MockResolvedOrder(entry_price=100.0, stop_price=95.0)
        portfolio = make_portfolio(total_equity=100_000.0)
        profile = make_profile(risk_per_trade_pct=0.01, max_position_pct=0.10)

        result = calculate_position_size(order, portfolio, profile, "moderate")

        assert result.rejection_reason is None
        # raw_quantity = floor(1000 / 5) = 200
        # position_value = 200 * 100 = 20000 > 10000
        # reduced quantity = floor(10000 / 100) = 100
        assert result.quantity == 100
        assert result.position_value == 10_000.0


# ---------------------------------------------------------------------------
# Rejection tests
# ---------------------------------------------------------------------------


class TestSizingRejections:
    """Tests for rejection conditions."""

    def test_rejects_risk_multiplier_above_one(self):
        """risk_multiplier > 1.0 is rejected (downward only)."""
        order = MockResolvedOrder(risk_multiplier=1.5)
        portfolio = make_portfolio()
        profile = make_profile()

        result = calculate_position_size(order, portfolio, profile, "moderate")

        assert result.rejected
        assert "exceeds 1.0" in result.rejection_reason

    def test_rejects_risk_multiplier_zero(self):
        """risk_multiplier == 0.0 is rejected."""
        order = MockResolvedOrder(risk_multiplier=0.0)
        portfolio = make_portfolio()
        profile = make_profile()

        result = calculate_position_size(order, portfolio, profile, "moderate")

        assert result.rejected
        assert "non-positive" in result.rejection_reason

    def test_rejects_risk_multiplier_negative(self):
        """risk_multiplier < 0.0 is rejected."""
        order = MockResolvedOrder(risk_multiplier=-0.5)
        portfolio = make_portfolio()
        profile = make_profile()

        result = calculate_position_size(order, portfolio, profile, "moderate")

        assert result.rejected
        assert "non-positive" in result.rejection_reason

    def test_rejects_zero_stop_distance(self):
        """stop_distance == 0 when entry == stop."""
        order = MockResolvedOrder(entry_price=100.0, stop_price=100.0)
        portfolio = make_portfolio()
        profile = make_profile()

        result = calculate_position_size(order, portfolio, profile, "moderate")

        assert result.rejected
        assert "stop distance" in result.rejection_reason

    def test_rejects_zero_equity(self):
        """Zero total_equity means no sizing possible."""
        order = MockResolvedOrder()
        portfolio = {"cash": 0.0, "total_equity": 0.0}
        profile = make_profile()

        result = calculate_position_size(order, portfolio, profile, "moderate")

        assert result.rejected
        assert "total_equity" in result.rejection_reason

    def test_rejects_when_quantity_rounds_to_zero(self):
        """Tiny risk budget relative to stop distance → 0 shares."""
        # Very tight risk budget: 100000 * 0.0001 = $10 max risk
        # stop_distance = $50 → floor(10/50) = 0
        order = MockResolvedOrder(entry_price=100.0, stop_price=50.0)
        portfolio = make_portfolio(total_equity=100_000.0)
        profile = make_profile(risk_per_trade_pct=0.0001)

        result = calculate_position_size(order, portfolio, profile, "moderate")

        assert result.rejected
        assert "quantity" in result.rejection_reason

    def test_rejects_when_position_cap_reduces_to_zero(self):
        """Very expensive stock with tight position cap → 0 shares."""
        # entry = $1,000,000 per share
        # max_position_value = 100000 * 0.01 = $1000
        # floor(1000 / 1000000) = 0
        order = MockResolvedOrder(entry_price=1_000_000.0, stop_price=999_000.0)
        portfolio = make_portfolio(total_equity=100_000.0)
        profile = make_profile(risk_per_trade_pct=0.01, max_position_pct=0.01)

        result = calculate_position_size(order, portfolio, profile, "moderate")

        assert result.rejected
        assert "quantity" in result.rejection_reason or "reduced" in (result.rejection_reason or "")


# ---------------------------------------------------------------------------
# PM quantity isolation tests (Requirement 4.2)
# ---------------------------------------------------------------------------


class TestPMQuantityIsolation:
    """Verify PM-supplied quantity is never used."""

    def test_ignores_quantity_on_resolved_order(self):
        """Even if resolved_order has a .quantity attribute, it's never used."""
        order = MockResolvedOrder(
            entry_price=100.0, stop_price=95.0, quantity=9999
        )
        portfolio = make_portfolio(total_equity=100_000.0)
        profile = make_profile(risk_per_trade_pct=0.01)

        result = calculate_position_size(order, portfolio, profile, "moderate")

        assert result.rejection_reason is None
        # If PM quantity (9999) were used, the result would be 9999
        assert result.quantity == 200
        assert result.quantity != 9999


# ---------------------------------------------------------------------------
# Edge case and property-like tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for position sizing."""

    def test_risk_multiplier_exactly_one(self):
        """risk_multiplier=1.0 is valid (no reduction)."""
        order = MockResolvedOrder(risk_multiplier=1.0)
        portfolio = make_portfolio(total_equity=100_000.0)
        profile = make_profile(risk_per_trade_pct=0.01)

        result = calculate_position_size(order, portfolio, profile, "moderate")

        assert result.rejection_reason is None
        assert result.applied_multiplier == 1.0

    def test_risk_multiplier_none_defaults_to_one(self):
        """None risk_multiplier defaults to 1.0."""
        order = MockResolvedOrder()
        order.risk_multiplier = None
        portfolio = make_portfolio(total_equity=100_000.0)
        profile = make_profile(risk_per_trade_pct=0.01)

        result = calculate_position_size(order, portfolio, profile, "moderate")

        assert result.rejection_reason is None
        assert result.applied_multiplier == 1.0

    def test_recovery_multiplier_at_one_is_standard(self):
        """recovery_multiplier=1.0 → sizing_method='standard'."""
        order = MockResolvedOrder()
        portfolio = make_portfolio(total_equity=100_000.0)
        profile = make_profile()

        result = calculate_position_size(
            order, portfolio, profile, "moderate", recovery_multiplier=1.0
        )

        assert result.sizing_method == "standard"

    def test_very_small_risk_multiplier(self):
        """Very small but valid risk_multiplier (0.01) reduces size greatly."""
        order = MockResolvedOrder(
            entry_price=100.0, stop_price=95.0, risk_multiplier=0.01
        )
        portfolio = make_portfolio(total_equity=100_000.0)
        profile = make_profile(risk_per_trade_pct=0.01)

        result = calculate_position_size(order, portfolio, profile, "moderate")

        # max_dollar_risk = 1000 * 0.01 = 10
        # quantity = floor(10 / 5) = 2
        assert result.rejection_reason is None
        assert result.quantity == 2

    def test_rejected_property(self):
        """SizingResult.rejected property reflects rejection state."""
        order = MockResolvedOrder(risk_multiplier=2.0)
        portfolio = make_portfolio()
        profile = make_profile()

        result = calculate_position_size(order, portfolio, profile, "moderate")

        assert result.rejected is True

    def test_not_rejected_property(self):
        """SizingResult.rejected is False for valid results."""
        order = MockResolvedOrder()
        portfolio = make_portfolio(total_equity=100_000.0)
        profile = make_profile()

        result = calculate_position_size(order, portfolio, profile, "moderate")

        assert result.rejected is False

    def test_profile_defaults_used(self):
        """Missing profile keys fall back to defaults (1% risk, 25% position)."""
        order = MockResolvedOrder(entry_price=100.0, stop_price=95.0)
        portfolio = make_portfolio(total_equity=100_000.0)
        profile = {}  # empty profile, defaults apply

        result = calculate_position_size(order, portfolio, profile, "moderate")

        assert result.rejection_reason is None
        # defaults: risk_per_trade_pct=0.01, max_position_pct=0.25
        assert result.quantity == 200
