"""Deterministic Geometry Calculator.

Pure, side-effect-free module. Single source of truth for all geometry arithmetic.
Used at every provenance stage and in Decision Replay.

Requirements: 4.1, 4.2, 4.3, 4.5, 4.6, 4.7
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, Context, InvalidOperation
from enum import Enum
from typing import List


class ValidationStatus(Enum):
    """Typed validation status enum."""
    VALID = "valid"
    INVALID = "invalid"
    INCOMPLETE = "incomplete"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class GeometryValidationError:
    """Structured validation error for a specific field."""
    field_name: str
    reason: str
    value: str  # string representation of the invalid value


# Fixed Decimal context — 28 digits precision, preserving canonical unrounded strings
GEOMETRY_DECIMAL_CONTEXT = Context(prec=28, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class GeometryResult:
    """Immutable result of geometry computation."""
    direction: str                              # "BUY" or "SHORT"
    entry_price: Decimal
    stop_price: Decimal
    target_price: Decimal
    quantity: Decimal
    risk_distance: Decimal                      # always positive when valid
    reward_distance: Decimal                    # always positive when valid
    reward_to_risk: Decimal                     # reward_distance / risk_distance
    per_unit_risk: Decimal                      # risk_distance * 1
    total_dollar_risk: Decimal                  # per_unit_risk * quantity
    stop_direction_valid: bool                  # stop on correct side of entry
    target_direction_valid: bool                # target on correct side of entry
    is_valid: bool                              # all fields finite, positive, direction-correct
    validation_errors: List[GeometryValidationError]  # empty when valid
    validation_status: ValidationStatus


# Sentinel zero for invalid/incomplete results
_ZERO = Decimal("0")


def _to_decimal(value, field_name: str, errors: List[GeometryValidationError]) -> Decimal | None:
    """Convert a value to Decimal within GEOMETRY_DECIMAL_CONTEXT.

    Returns the Decimal if successful, or None if the value is invalid.
    Appends a GeometryValidationError to errors on failure.
    """
    if value is None:
        errors.append(GeometryValidationError(
            field_name=field_name,
            reason="missing",
            value="None",
        ))
        return None

    try:
        if isinstance(value, Decimal):
            d = GEOMETRY_DECIMAL_CONTEXT.create_decimal(value)
        else:
            d = GEOMETRY_DECIMAL_CONTEXT.create_decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        errors.append(GeometryValidationError(
            field_name=field_name,
            reason="non_finite",
            value=str(value),
        ))
        return None

    if not d.is_finite():
        errors.append(GeometryValidationError(
            field_name=field_name,
            reason="non_finite",
            value=str(value),
        ))
        return None

    return d


def _validate_direction(direction: str | None) -> tuple[str | None, GeometryValidationError | None]:
    """Validate and normalize the direction field.

    Returns (normalized_direction, error_or_None).
    """
    if direction is None:
        return None, GeometryValidationError(
            field_name="direction",
            reason="missing",
            value="None",
        )

    normalized = str(direction).strip().upper()
    if normalized not in ("BUY", "SHORT"):
        return None, GeometryValidationError(
            field_name="direction",
            reason="invalid_direction",
            value=str(direction),
        )

    return normalized, None


def compute_geometry(
    direction: str | None,
    entry_price: Decimal | str | float | None,
    stop_price: Decimal | str | float | None,
    target_price: Decimal | str | float | None,
    quantity: Decimal | str | float | int | None,
) -> GeometryResult:
    """Compute deterministic geometry for a trade intent.

    Pure function. No side effects. Same inputs always produce same outputs.
    Used at every provenance stage and in Decision Replay.

    Accepts None for any field — produces validation errors identifying missing fields.
    All arithmetic uses GEOMETRY_DECIMAL_CONTEXT (28-digit precision).
    Never substitutes defaults — always reports the error.
    """
    errors: List[GeometryValidationError] = []

    # Validate direction
    norm_direction, dir_error = _validate_direction(direction)
    if dir_error is not None:
        errors.append(dir_error)

    # Convert numeric fields
    entry_dec = _to_decimal(entry_price, "entry_price", errors)
    stop_dec = _to_decimal(stop_price, "stop_price", errors)
    target_dec = _to_decimal(target_price, "target_price", errors)
    qty_dec = _to_decimal(quantity, "quantity", errors)

    # Validate zero values for prices and quantity
    if entry_dec is not None and entry_dec == _ZERO:
        errors.append(GeometryValidationError(
            field_name="entry_price",
            reason="zero",
            value=str(entry_price),
        ))
        entry_dec = None

    if stop_dec is not None and stop_dec == _ZERO:
        errors.append(GeometryValidationError(
            field_name="stop_price",
            reason="zero",
            value=str(stop_price),
        ))
        stop_dec = None

    if target_dec is not None and target_dec == _ZERO:
        errors.append(GeometryValidationError(
            field_name="target_price",
            reason="zero",
            value=str(target_price),
        ))
        target_dec = None

    if qty_dec is not None and qty_dec == _ZERO:
        errors.append(GeometryValidationError(
            field_name="quantity",
            reason="zero",
            value=str(quantity),
        ))
        qty_dec = None

    # If any core field is missing/invalid at this point, produce incomplete result
    if norm_direction is None or entry_dec is None or stop_dec is None or target_dec is None or qty_dec is None:
        return GeometryResult(
            direction=norm_direction or "UNKNOWN",
            entry_price=entry_dec or _ZERO,
            stop_price=stop_dec or _ZERO,
            target_price=target_dec or _ZERO,
            quantity=qty_dec or _ZERO,
            risk_distance=_ZERO,
            reward_distance=_ZERO,
            reward_to_risk=_ZERO,
            per_unit_risk=_ZERO,
            total_dollar_risk=_ZERO,
            stop_direction_valid=False,
            target_direction_valid=False,
            is_valid=False,
            validation_errors=errors,
            validation_status=ValidationStatus.INCOMPLETE,
        )

    # Directional validation and geometry computation
    ctx = GEOMETRY_DECIMAL_CONTEXT

    if norm_direction == "BUY":
        # BUY: stop < entry < target
        stop_direction_valid = ctx.compare(stop_dec, entry_dec) < 0  # stop < entry
        target_direction_valid = ctx.compare(target_dec, entry_dec) > 0  # target > entry
        risk_distance = ctx.subtract(entry_dec, stop_dec)
        reward_distance = ctx.subtract(target_dec, entry_dec)
    else:
        # SHORT: target < entry < stop
        stop_direction_valid = ctx.compare(stop_dec, entry_dec) > 0  # stop > entry
        target_direction_valid = ctx.compare(target_dec, entry_dec) < 0  # target < entry
        risk_distance = ctx.subtract(stop_dec, entry_dec)
        reward_distance = ctx.subtract(entry_dec, target_dec)

    # Check directional validity
    if not stop_direction_valid:
        if norm_direction == "BUY":
            errors.append(GeometryValidationError(
                field_name="stop_price",
                reason="directionally_invalid",
                value=str(stop_price),
            ))
        else:
            errors.append(GeometryValidationError(
                field_name="stop_price",
                reason="directionally_invalid",
                value=str(stop_price),
            ))

    if not target_direction_valid:
        if norm_direction == "BUY":
            errors.append(GeometryValidationError(
                field_name="target_price",
                reason="directionally_invalid",
                value=str(target_price),
            ))
        else:
            errors.append(GeometryValidationError(
                field_name="target_price",
                reason="directionally_invalid",
                value=str(target_price),
            ))

    # Compute derived values using Decimal context
    if risk_distance > _ZERO:
        reward_to_risk = ctx.divide(reward_distance, risk_distance)
    else:
        reward_to_risk = _ZERO

    per_unit_risk = risk_distance
    total_dollar_risk = ctx.multiply(per_unit_risk, qty_dec)

    # Determine overall validity
    is_valid = (
        len(errors) == 0
        and stop_direction_valid
        and target_direction_valid
        and risk_distance > _ZERO
        and reward_distance > _ZERO
    )

    if is_valid:
        validation_status = ValidationStatus.VALID
    else:
        validation_status = ValidationStatus.INVALID

    return GeometryResult(
        direction=norm_direction,
        entry_price=entry_dec,
        stop_price=stop_dec,
        target_price=target_dec,
        quantity=qty_dec,
        risk_distance=risk_distance,
        reward_distance=reward_distance,
        reward_to_risk=reward_to_risk,
        per_unit_risk=per_unit_risk,
        total_dollar_risk=total_dollar_risk,
        stop_direction_valid=stop_direction_valid,
        target_direction_valid=target_direction_valid,
        is_valid=is_valid,
        validation_errors=errors,
        validation_status=validation_status,
    )


# Fields to compare in compare_geometry (excludes validation_errors list)
_COMPARABLE_FIELDS = (
    "direction",
    "entry_price",
    "stop_price",
    "target_price",
    "quantity",
    "risk_distance",
    "reward_distance",
    "reward_to_risk",
    "per_unit_risk",
    "total_dollar_risk",
    "stop_direction_valid",
    "target_direction_valid",
    "is_valid",
    "validation_status",
)


def compare_geometry(before: GeometryResult, after: GeometryResult) -> dict:
    """Compare two geometry results and return changed fields.

    Returns dict with field names as keys and (before_value, after_value) tuples.
    Used to populate the `fields_changed` array in provenance events.

    Compares all numeric fields and boolean fields.
    Skips validation_errors list comparison (compares is_valid and validation_status instead).
    """
    changes: dict = {}

    for field_name in _COMPARABLE_FIELDS:
        before_val = getattr(before, field_name)
        after_val = getattr(after, field_name)
        if before_val != after_val:
            changes[field_name] = (before_val, after_val)

    return changes
