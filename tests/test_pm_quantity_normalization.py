"""Regression coverage for nullable LLM quantities in PM decisions."""

from agents.portfolio_manager import _coerce_quantity


def test_coerce_quantity_treats_none_as_zero():
    assert _coerce_quantity(None, symbol="NVDA") == 0


def test_coerce_quantity_handles_numeric_strings():
    assert _coerce_quantity("1,234", symbol="NVDA") == 1234


def test_coerce_quantity_rejects_invalid_values():
    assert _coerce_quantity("ten shares", symbol="NVDA") == 0
