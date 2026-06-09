"""
Unit tests for utils/shadow_quantity.py

Tests the shadow quantity calculator that derives a non-executable quantity
from profile risk parameters and candidate geometry only.

Validates Requirement 10.3.
"""

import math
from datetime import datetime, timezone

import pytest

from utils.candidate_registry import CandidateRecord
from utils.shadow_quantity import compute_shadow_quantity


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_candidate(
    entry_price: float = 100.0,
    stop_price: float = 95.0,
    target_price: float = 110.0,
) -> CandidateRecord:
    """Create a minimal CandidateRecord for shadow quantity testing."""
    now = datetime.now(tz=timezone.utc)
    return CandidateRecord(
        candidate_id="test-uuid-1234",
        cycle_id="cycle-001",
        profile_id="moderate",
        symbol="AAPL",
        direction="BUY",
        setup_type="breakout",
        geometry_name="primary",
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        risk_reward=2.0,
        trigger="price_above_resistance",
        invalidation_basis="below_support",
        target_basis="measured_move",
        source_signal_id="signal-001",
        signal_snapshot_json="{}",
        created_at=now,
        expires_at=now,
        integrity_hash="abc123",
    )


def _make_profile(risk_per_trade_pct: float = 0.01) -> dict:
    return {"risk_per_trade_pct": risk_per_trade_pct}


# ---------------------------------------------------------------------------
# Core logic tests
# ---------------------------------------------------------------------------


class TestComputeShadowQuantity:
    """Core tests for compute_shadow_quantity."""

    def test_standard_calculation(self):
        """Basic: 1% of 100k equity / $5 stop distance = 200 shares."""
        candidate = _make_candidate(entry_price=100.0, stop_price=95.0)
        profile = _make_profile(risk_per_trade_pct=0.01)

        result = compute_shadow_quantity(candidate, profile, "moderate", 100_000.0)

        # max_dollar_risk = 100000 * 0.01 = 1000
        # stop_distance = abs(100 - 95) = 5
        # quantity = floor(1000 / 5) = 200
        assert result == 200

    def test_short_direction_uses_abs_stop_distance(self):
        """SHORT candidate: stop is above entry, abs gives correct distance."""
        candidate = _make_candidate(entry_price=50.0, stop_price=52.0)
        profile = _make_profile(risk_per_trade_pct=0.02)

        result = compute_shadow_quantity(candidate, profile, "aggressive", 50_000.0)

        # max_dollar_risk = 50000 * 0.02 = 1000
        # stop_distance = abs(50 - 52) = 2
        # quantity = floor(1000 / 2) = 500
        assert result == 500

    def test_floors_fractional_quantity(self):
        """Quantity is always floored to integer."""
        candidate = _make_candidate(entry_price=100.0, stop_price=97.0)
        profile = _make_profile(risk_per_trade_pct=0.01)

        result = compute_shadow_quantity(candidate, profile, "moderate", 100_000.0)

        # max_dollar_risk = 100000 * 0.01 = 1000
        # stop_distance = 3
        # quantity = floor(1000 / 3) = floor(333.33) = 333
        assert result == 333

    def test_defaults_risk_pct_when_missing(self):
        """When profile lacks risk_per_trade_pct, defaults to 0.01."""
        candidate = _make_candidate(entry_price=100.0, stop_price=95.0)
        profile = {}  # No risk_per_trade_pct key

        result = compute_shadow_quantity(candidate, profile, "moderate", 100_000.0)

        # Uses default 0.01: 100000 * 0.01 / 5 = 200
        assert result == 200


# ---------------------------------------------------------------------------
# Geometry incomplete (returns None)
# ---------------------------------------------------------------------------


class TestIncompleteGeometry:
    """Tests that incomplete or invalid geometry returns None."""

    def test_entry_price_zero(self):
        candidate = _make_candidate(entry_price=0.0, stop_price=95.0)
        assert compute_shadow_quantity(candidate, _make_profile(), "x", 100_000.0) is None

    def test_stop_price_zero(self):
        candidate = _make_candidate(entry_price=100.0, stop_price=0.0)
        assert compute_shadow_quantity(candidate, _make_profile(), "x", 100_000.0) is None

    def test_entry_price_negative(self):
        candidate = _make_candidate(entry_price=-10.0, stop_price=95.0)
        assert compute_shadow_quantity(candidate, _make_profile(), "x", 100_000.0) is None

    def test_stop_price_negative(self):
        candidate = _make_candidate(entry_price=100.0, stop_price=-5.0)
        assert compute_shadow_quantity(candidate, _make_profile(), "x", 100_000.0) is None

    def test_entry_equals_stop(self):
        """Zero stop distance means geometry is incomplete."""
        candidate = _make_candidate(entry_price=100.0, stop_price=100.0)
        assert compute_shadow_quantity(candidate, _make_profile(), "x", 100_000.0) is None

    def test_entry_price_infinity(self):
        candidate = _make_candidate(entry_price=float("inf"), stop_price=95.0)
        assert compute_shadow_quantity(candidate, _make_profile(), "x", 100_000.0) is None

    def test_stop_price_infinity(self):
        candidate = _make_candidate(entry_price=100.0, stop_price=float("inf"))
        assert compute_shadow_quantity(candidate, _make_profile(), "x", 100_000.0) is None

    def test_entry_price_nan(self):
        candidate = _make_candidate(entry_price=float("nan"), stop_price=95.0)
        assert compute_shadow_quantity(candidate, _make_profile(), "x", 100_000.0) is None

    def test_portfolio_equity_zero(self):
        candidate = _make_candidate(entry_price=100.0, stop_price=95.0)
        assert compute_shadow_quantity(candidate, _make_profile(), "x", 0.0) is None

    def test_portfolio_equity_negative(self):
        candidate = _make_candidate(entry_price=100.0, stop_price=95.0)
        assert compute_shadow_quantity(candidate, _make_profile(), "x", -1000.0) is None

    def test_tiny_risk_budget_yields_zero_quantity(self):
        """When risk budget is too small for even 1 share, returns None."""
        candidate = _make_candidate(entry_price=100.0, stop_price=95.0)
        # risk = 100 * 0.01 = 1.0, stop_distance = 5, floor(1/5) = 0
        assert compute_shadow_quantity(candidate, _make_profile(), "x", 100.0) is None
