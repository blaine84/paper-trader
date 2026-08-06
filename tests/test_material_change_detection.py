"""Tests for _detect_material_change() in utils.alert_intent_store.

Validates Requirements 3.1–3.7: Material Change Detection rules.
"""
from __future__ import annotations

from utils.alert_intent_store import _detect_material_change


# ─── Not material: same source_level, direction, price within 0.5% ──────────


def test_same_fields_price_within_threshold_not_material():
    """Same source_level, direction, price within 0.5% → NOT material."""
    # 155.00 → 155.77 is 0.496% which is below 0.5% threshold
    result = _detect_material_change(
        existing_source_level="resistance_155",
        existing_direction="long",
        existing_trigger_price="155.00",
        new_source_level="resistance_155",
        new_direction="long",
        new_trigger_price="155.77",
    )
    assert result is False


# ─── Different source_level → material ──────────────────────────────────────


def test_different_source_level_is_material():
    """Different source_level → material."""
    result = _detect_material_change(
        existing_source_level="resistance_155",
        existing_direction="long",
        existing_trigger_price="155.00",
        new_source_level="resistance_157",
        new_direction="long",
        new_trigger_price="155.00",
    )
    assert result is True


# ─── Different direction → material ─────────────────────────────────────────


def test_different_direction_is_material():
    """Different direction → material."""
    result = _detect_material_change(
        existing_source_level="resistance_155",
        existing_direction="long",
        existing_trigger_price="155.00",
        new_source_level="resistance_155",
        new_direction="short",
        new_trigger_price="155.00",
    )
    assert result is True


# ─── Price change > 0.5% → material ─────────────────────────────────────────


def test_price_change_above_threshold_is_material():
    """Price change > 0.5% → material. 155.00 → 155.78 is 0.503%."""
    result = _detect_material_change(
        existing_source_level="resistance_155",
        existing_direction="long",
        existing_trigger_price="155.00",
        new_source_level="resistance_155",
        new_direction="long",
        new_trigger_price="155.78",
    )
    assert result is True


# ─── Price change exactly 0.5% → NOT material (strictly greater required) ───


def test_price_change_exactly_threshold_not_material():
    """Price change exactly 0.5% → NOT material (strictly greater required).

    0.5% of 100.00 is 0.50, so 100.00 → 100.50 is exactly 0.5%.
    """
    result = _detect_material_change(
        existing_source_level="resistance_100",
        existing_direction="long",
        existing_trigger_price="100.00",
        new_source_level="resistance_100",
        new_direction="long",
        new_trigger_price="100.50",
    )
    assert result is False


# ─── None == None for source_level → not material ───────────────────────────


def test_none_equals_none_source_level_not_material():
    """None == None for source_level → NOT material."""
    result = _detect_material_change(
        existing_source_level=None,
        existing_direction="long",
        existing_trigger_price="155.00",
        new_source_level=None,
        new_direction="long",
        new_trigger_price="155.00",
    )
    assert result is False


# ─── None != "resistance_155" for source_level → material ───────────────────


def test_none_to_value_source_level_is_material():
    """None != 'resistance_155' for source_level → material."""
    result = _detect_material_change(
        existing_source_level=None,
        existing_direction="long",
        existing_trigger_price="155.00",
        new_source_level="resistance_155",
        new_direction="long",
        new_trigger_price="155.00",
    )
    assert result is True


# ─── "resistance_155" != None for source_level → material ───────────────────


def test_value_to_none_source_level_is_material():
    """'resistance_155' != None for source_level → material."""
    result = _detect_material_change(
        existing_source_level="resistance_155",
        existing_direction="long",
        existing_trigger_price="155.00",
        new_source_level=None,
        new_direction="long",
        new_trigger_price="155.00",
    )
    assert result is True


# ─── None trigger_price (either side) → material (fail-open) ────────────────


def test_none_existing_trigger_price_is_material():
    """None existing trigger_price → material (fail-open)."""
    result = _detect_material_change(
        existing_source_level="resistance_155",
        existing_direction="long",
        existing_trigger_price=None,
        new_source_level="resistance_155",
        new_direction="long",
        new_trigger_price="155.00",
    )
    assert result is True


def test_none_new_trigger_price_is_material():
    """None new trigger_price → material (fail-open)."""
    result = _detect_material_change(
        existing_source_level="resistance_155",
        existing_direction="long",
        existing_trigger_price="155.00",
        new_source_level="resistance_155",
        new_direction="long",
        new_trigger_price=None,
    )
    assert result is True


# ─── Zero trigger_price (existing) → material (fail-open) ───────────────────


def test_zero_existing_trigger_price_is_material():
    """Zero existing trigger_price → material (fail-open, cannot compute %)."""
    result = _detect_material_change(
        existing_source_level="resistance_155",
        existing_direction="long",
        existing_trigger_price="0",
        new_source_level="resistance_155",
        new_direction="long",
        new_trigger_price="155.00",
    )
    assert result is True
