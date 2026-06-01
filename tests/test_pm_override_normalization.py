"""Unit tests for override_confidence_score and override_reason normalization.

Validates that normalize_pm_entry_decisions() correctly preserves, coerces,
and drops override metadata fields per the range validation rules.

Requirements: 3.7
"""

import math

import pytest

from agents.portfolio_manager import normalize_pm_entry_decisions


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_valid_decision(**overrides) -> dict:
    """Build a minimal valid BUY decision dict with optional overrides."""
    base = {
        "action": "BUY",
        "symbol": "AAPL",
        "quantity": 10,
        "entry_price": 150.0,
        "stop": 145.0,
        "target": 160.0,
        "setup_type": "breakout",
        "rationale": "test",
    }
    base.update(overrides)
    return base


def _entry_signals():
    return {"AAPL": {"setup_type": "breakout"}}


def _normalize_single(decision: dict) -> dict | None:
    """Normalize a single decision and return the order dict (or None if rejected)."""
    result = normalize_pm_entry_decisions([decision], _entry_signals())
    if result.orders:
        return result.orders[0].order
    return None


# ── override_confidence_score: valid numeric values preserved as float ────────

class TestOverrideConfidenceScorePreservation:
    """Test that valid numeric override_confidence_score is preserved as float."""

    def test_int_value_preserved_as_float(self):
        order = _normalize_single(_make_valid_decision(override_confidence_score=8))
        assert order is not None
        assert order["override_confidence_score"] == 8.0
        assert isinstance(order["override_confidence_score"], float)

    def test_float_value_preserved(self):
        order = _normalize_single(_make_valid_decision(override_confidence_score=8.0))
        assert order is not None
        assert order["override_confidence_score"] == 8.0
        assert isinstance(order["override_confidence_score"], float)

    def test_numeric_string_parsed_and_preserved(self):
        order = _normalize_single(_make_valid_decision(override_confidence_score="8.5"))
        assert order is not None
        assert order["override_confidence_score"] == 8.5
        assert isinstance(order["override_confidence_score"], float)


# ── override_confidence_score: boundary values ────────────────────────────────

class TestOverrideConfidenceScoreBoundaries:
    """Test boundary values for override_confidence_score range [0.0, 10.0]."""

    def test_just_below_threshold_valid(self):
        """7.999 is valid (within range, just below the 8.0 gate threshold)."""
        order = _normalize_single(_make_valid_decision(override_confidence_score=7.999))
        assert order is not None
        assert order["override_confidence_score"] == 7.999

    def test_exactly_at_threshold_valid(self):
        """8.0 is valid (at the gate threshold)."""
        order = _normalize_single(_make_valid_decision(override_confidence_score=8.0))
        assert order is not None
        assert order["override_confidence_score"] == 8.0

    def test_upper_bound_valid(self):
        """10.0 is valid (upper bound of range)."""
        order = _normalize_single(_make_valid_decision(override_confidence_score=10.0))
        assert order is not None
        assert order["override_confidence_score"] == 10.0

    def test_lower_bound_valid(self):
        """0.0 is valid (lower bound of range)."""
        order = _normalize_single(_make_valid_decision(override_confidence_score=0.0))
        assert order is not None
        assert order["override_confidence_score"] == 0.0


# ── override_confidence_score: invalid values dropped ─────────────────────────

class TestOverrideConfidenceScoreInvalidDropped:
    """Test that invalid override_confidence_score values are silently dropped."""

    def test_above_range_dropped(self):
        """10.1 exceeds the valid range and is dropped."""
        order = _normalize_single(_make_valid_decision(override_confidence_score=10.1))
        assert order is not None
        assert "override_confidence_score" not in order

    def test_negative_dropped(self):
        """-1.0 is below the valid range and is dropped."""
        order = _normalize_single(_make_valid_decision(override_confidence_score=-1.0))
        assert order is not None
        assert "override_confidence_score" not in order

    def test_nan_dropped(self):
        """NaN is not finite and is dropped."""
        order = _normalize_single(_make_valid_decision(override_confidence_score=float("nan")))
        assert order is not None
        assert "override_confidence_score" not in order

    def test_inf_dropped(self):
        """Positive infinity is not finite and is dropped."""
        order = _normalize_single(_make_valid_decision(override_confidence_score=float("inf")))
        assert order is not None
        assert "override_confidence_score" not in order

    def test_negative_inf_dropped(self):
        """Negative infinity is not finite and is dropped."""
        order = _normalize_single(_make_valid_decision(override_confidence_score=float("-inf")))
        assert order is not None
        assert "override_confidence_score" not in order

    def test_non_numeric_string_dropped(self):
        """Non-numeric string 'high' cannot be coerced and is dropped."""
        order = _normalize_single(_make_valid_decision(override_confidence_score="high"))
        assert order is not None
        assert "override_confidence_score" not in order

    def test_list_dropped(self):
        """A list value cannot be coerced to float and is dropped."""
        order = _normalize_single(_make_valid_decision(override_confidence_score=[8, 9]))
        assert order is not None
        assert "override_confidence_score" not in order

    def test_dict_dropped(self):
        """A dict value cannot be coerced to float and is dropped."""
        order = _normalize_single(_make_valid_decision(override_confidence_score={"score": 8}))
        assert order is not None
        assert "override_confidence_score" not in order

    def test_none_dropped(self):
        """Explicit None is treated as absent (field not added to order)."""
        order = _normalize_single(_make_valid_decision(override_confidence_score=None))
        assert order is not None
        assert "override_confidence_score" not in order


# ── override_reason: preservation and dropping ────────────────────────────────

class TestOverrideReasonPreservation:
    """Test that override_reason is preserved when non-empty string, dropped otherwise."""

    def test_non_empty_string_preserved(self):
        order = _normalize_single(_make_valid_decision(override_reason="Strong catalyst alignment"))
        assert order is not None
        assert order["override_reason"] == "Strong catalyst alignment"

    def test_whitespace_trimmed(self):
        order = _normalize_single(_make_valid_decision(override_reason="  padded reason  "))
        assert order is not None
        assert order["override_reason"] == "padded reason"

    def test_empty_string_dropped(self):
        order = _normalize_single(_make_valid_decision(override_reason=""))
        assert order is not None
        assert "override_reason" not in order

    def test_whitespace_only_dropped(self):
        order = _normalize_single(_make_valid_decision(override_reason="   "))
        assert order is not None
        assert "override_reason" not in order

    def test_none_dropped(self):
        order = _normalize_single(_make_valid_decision(override_reason=None))
        assert order is not None
        assert "override_reason" not in order

    def test_integer_dropped(self):
        """Non-string types are dropped."""
        order = _normalize_single(_make_valid_decision(override_reason=42))
        assert order is not None
        assert "override_reason" not in order

    def test_list_dropped(self):
        """Non-string types are dropped."""
        order = _normalize_single(_make_valid_decision(override_reason=["reason"]))
        assert order is not None
        assert "override_reason" not in order


# ── Fail-closed: absent field means gate receives None ────────────────────────

class TestFailClosedAbsentField:
    """Test that gate receives None when override_confidence_score is absent."""

    def test_absent_field_not_in_order(self):
        """When override_confidence_score is not in the decision, it is not in the order."""
        decision = _make_valid_decision()
        assert "override_confidence_score" not in decision
        order = _normalize_single(decision)
        assert order is not None
        assert "override_confidence_score" not in order

    def test_gate_receives_none_via_get(self):
        """The gate uses order.get('override_confidence_score') which returns None when absent."""
        decision = _make_valid_decision()
        order = _normalize_single(decision)
        assert order is not None
        # This is how the gate pipeline extracts the value — .get() returns None
        assert order.get("override_confidence_score") is None
