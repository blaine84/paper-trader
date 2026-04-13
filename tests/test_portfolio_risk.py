"""
Tests for core.portfolio_risk — property-based and unit tests.

Covers Properties 7–8 from the design document and unit tests for
portfolio risk engine functions.
"""

import math

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from core.portfolio_risk import (
    BUCKETS,
    MAX_BUCKET_EXPOSURE_PCT,
    DEFAULT_MAX_TOTAL_EXPOSURE,
    compute_portfolio_risk,
    validate_portfolio_risk,
    adaptive_risk_throttle,
    compute_risk_score,
)


# ===================================================================
# Task 3.4 — Property 7: Adaptive risk throttling
# ===================================================================

@given(
    base_size=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
    recent_losses=st.integers(min_value=3, max_value=100),
)
@settings(max_examples=100)
def test_adaptive_risk_throttle(base_size, recent_losses):
    """
    **Validates: Requirements 3.7**

    For any base_size > 0 and recent_losses >= 3, adaptive_risk_throttle
    returns a value between base_size * 0.50 and base_size * 0.75 inclusive.
    """
    result = adaptive_risk_throttle(base_size, recent_losses)
    lower = base_size * 0.50
    upper = base_size * 0.75
    assert lower - 1e-9 <= result <= upper + 1e-9, (
        f"base_size={base_size}, recent_losses={recent_losses}: "
        f"result {result} not in [{lower}, {upper}]"
    )


# ===================================================================
# Task 3.4 — Property 8: Multi-bucket exposure accounting
# ===================================================================

@given(
    quantity=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    avg_cost=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_multi_bucket_exposure_accounting(quantity, avg_cost):
    """
    **Validates: Requirements 3.1**

    For positions with symbols in multiple buckets, verify each bucket
    counts the symbol independently. NVDA is in both 'semis' and
    'mega_growth', so both buckets must reflect its exposure.
    """
    total_equity = 100_000.0
    positions = [
        {"symbol": "NVDA", "quantity": quantity, "avg_cost": avg_cost, "side": "long"},
    ]
    result = compute_portfolio_risk(positions, total_equity)

    expected_exposure = abs(quantity * avg_cost) / total_equity

    assert math.isclose(
        result["bucket_exposure"]["semis"], expected_exposure, rel_tol=1e-9
    ), f"semis exposure mismatch"
    assert math.isclose(
        result["bucket_exposure"]["mega_growth"], expected_exposure, rel_tol=1e-9
    ), f"mega_growth exposure mismatch"


# ===================================================================
# Task 3.5 — Unit tests
# ===================================================================

class TestMegaGrowthBucketClassification:
    def test_meta_in_mega_growth(self):
        """META is classified in the mega_growth bucket."""
        assert "META" in BUCKETS["mega_growth"]

    def test_mega_growth_bucket_exposure(self):
        """META position shows up in mega_growth bucket exposure."""
        positions = [
            {"symbol": "META", "quantity": 10, "avg_cost": 500.0, "side": "long"},
        ]
        result = compute_portfolio_risk(positions, 100_000.0)
        assert result["bucket_exposure"]["mega_growth"] == pytest.approx(
            10 * 500.0 / 100_000.0
        )


class TestMultiBucketMembership:
    def test_nvda_in_semis_and_mega_growth(self):
        """NVDA is counted in both semis and mega_growth buckets."""
        positions = [
            {"symbol": "NVDA", "quantity": 20, "avg_cost": 800.0, "side": "long"},
        ]
        result = compute_portfolio_risk(positions, 100_000.0)
        expected = 20 * 800.0 / 100_000.0
        assert result["bucket_exposure"]["semis"] == pytest.approx(expected)
        assert result["bucket_exposure"]["mega_growth"] == pytest.approx(expected)


class TestConfigurableTotalExposureThreshold:
    def test_lower_threshold_rejects(self):
        """A trade within default 1.5 but exceeding 1.2 threshold is rejected."""
        # Position that puts total exposure at ~1.3
        positions = [
            {"symbol": "AAPL", "quantity": 100, "avg_cost": 1000.0, "side": "long"},
        ]
        total_equity = 100_000.0
        new_trade = {"symbol": "GOOG", "quantity": 100, "price": 350.0}
        # Existing exposure: 100*1000/100000 = 1.0
        # New trade adds: 100*350/100000 = 0.35 → total = 1.35
        ok, msg = validate_portfolio_risk(
            new_trade, positions, total_equity, max_total_exposure=1.2
        )
        assert not ok
        assert "total exposure exceeded threshold" in msg

    def test_higher_threshold_accepts(self):
        """Same trade passes with a higher threshold."""
        positions = [
            {"symbol": "AAPL", "quantity": 100, "avg_cost": 1000.0, "side": "long"},
        ]
        total_equity = 100_000.0
        new_trade = {"symbol": "GOOG", "quantity": 100, "price": 350.0}
        ok, msg = validate_portfolio_risk(
            new_trade, positions, total_equity, max_total_exposure=1.5
        )
        assert ok
        assert msg == "OK"


class TestOverexposureRejection:
    def test_bucket_overexposure_rejected(self):
        """Trade that pushes a single bucket over 50% is rejected."""
        # Put heavy weight in semis bucket
        positions = [
            {"symbol": "AMD", "quantity": 200, "avg_cost": 150.0, "side": "long"},
        ]
        total_equity = 100_000.0
        # Existing semis: 200*150/100000 = 0.30
        # New trade adds: 100*250/100000 = 0.25 → semis total = 0.55 > 0.50
        new_trade = {"symbol": "INTC", "quantity": 100, "price": 250.0}
        ok, msg = validate_portfolio_risk(new_trade, positions, total_equity)
        assert not ok
        assert "semis exposure exceeded 50%" in msg


class TestSafeTradeAcceptance:
    def test_within_limits_accepted(self):
        """Trade within all limits is accepted."""
        positions = [
            {"symbol": "SPY", "quantity": 10, "avg_cost": 450.0, "side": "long"},
        ]
        total_equity = 100_000.0
        new_trade = {"symbol": "AAPL", "quantity": 5, "price": 180.0}
        ok, msg = validate_portfolio_risk(new_trade, positions, total_equity)
        assert ok
        assert msg == "OK"


class TestEmptyPositions:
    def test_empty_positions_all_zero(self):
        """Empty positions → all exposures 0.0."""
        result = compute_portfolio_risk([], 100_000.0)
        assert result["total_exposure"] == 0.0
        for v in result["bucket_exposure"].values():
            assert v == 0.0

    def test_empty_positions_trade_allowed(self):
        """With empty positions, a small trade is allowed."""
        new_trade = {"symbol": "SPY", "quantity": 10, "price": 450.0}
        ok, msg = validate_portfolio_risk(new_trade, [], 100_000.0)
        assert ok
        assert msg == "OK"


class TestZeroEquityGuard:
    def test_zero_equity_returns_max_risk(self):
        """Zero equity → risk_score = 1.0."""
        result = compute_portfolio_risk([], 0.0)
        assert result["risk_score"] == 1.0
        assert result["total_exposure"] == 0.0

    def test_zero_equity_rejects_trade(self):
        """Zero equity → trade rejected."""
        new_trade = {"symbol": "SPY", "quantity": 10, "price": 450.0}
        ok, msg = validate_portfolio_risk(new_trade, [], 0.0)
        assert not ok
        assert "zero equity" in msg


class TestRiskScoreOutput:
    def test_risk_score_present(self):
        """risk_score is present in compute_portfolio_risk output."""
        result = compute_portfolio_risk([], 100_000.0)
        assert "risk_score" in result
        assert isinstance(result["risk_score"], float)
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_risk_score_increases_with_exposure(self):
        """Higher exposure → higher risk_score."""
        low = compute_portfolio_risk(
            [{"symbol": "SPY", "quantity": 10, "avg_cost": 100.0, "side": "long"}],
            100_000.0,
        )
        high = compute_portfolio_risk(
            [{"symbol": "SPY", "quantity": 500, "avg_cost": 100.0, "side": "long"}],
            100_000.0,
        )
        assert high["risk_score"] > low["risk_score"]


class TestAdaptiveThrottleReducesSize:
    def test_three_losses_reduces(self):
        """3 losses → 25% reduction (returns 75% of base)."""
        result = adaptive_risk_throttle(100.0, 3)
        assert result == pytest.approx(75.0)

    def test_six_losses_reduces_max(self):
        """6 losses → 50% reduction (returns 50% of base)."""
        result = adaptive_risk_throttle(100.0, 6)
        assert result == pytest.approx(50.0)

    def test_four_losses_intermediate(self):
        """4 losses → between 50% and 75% of base."""
        result = adaptive_risk_throttle(100.0, 4)
        assert 50.0 <= result <= 75.0

    def test_large_losses_capped_at_50_pct(self):
        """10+ losses still returns 50% (capped at 6)."""
        result = adaptive_risk_throttle(100.0, 10)
        assert result == pytest.approx(50.0)


class TestAdaptiveThrottleNoReductionBelow3:
    def test_zero_losses_no_change(self):
        """0 losses → no reduction."""
        assert adaptive_risk_throttle(100.0, 0) == 100.0

    def test_one_loss_no_change(self):
        """1 loss → no reduction."""
        assert adaptive_risk_throttle(100.0, 1) == 100.0

    def test_two_losses_no_change(self):
        """2 losses → no reduction."""
        assert adaptive_risk_throttle(100.0, 2) == 100.0
