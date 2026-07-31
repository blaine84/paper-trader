"""Deterministic entry zone derivation and price-zone evaluation helpers.

Pure, side-effect-free module. Computes entry zones from reference price
and stop geometry, and evaluates whether current price is inside a zone
or past a target.

Used by plan_monitor and plan_executor — shared helpers avoid circular imports.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, Context

from utils.gate_config import PLAN_ENTRY_ZONE_TOLERANCE_PCT

logger = logging.getLogger(__name__)

# Fixed Decimal context — 28 digits precision, consistent with geometry_calculator
_CTX = Context(prec=28, rounding=ROUND_HALF_UP)

# Default zone fraction: the entry zone spans this fraction of the
# entry_reference-to-stop distance on the entry side.
DEFAULT_ZONE_FRACTION = Decimal("0.20")


@dataclass(frozen=True)
class EntryZone:
    """Computed entry zone for a trade plan.

    upper/lower define the raw zone bounds (no tolerance applied here).
    Tolerance is applied at evaluation time by is_price_in_zone().
    """

    upper: Decimal
    lower: Decimal
    reference: Decimal
    tolerance_pct: Decimal


def derive_entry_zone(
    entry_reference: Decimal,
    direction: str,
    stop_price: Decimal,
    tolerance_pct: Decimal | None = None,
    zone_fraction: Decimal | None = None,
) -> EntryZone:
    """Derive entry zone bounds from reference price, direction, and stop.

    The zone is derived deterministically using a fraction of the distance
    between entry_reference and stop_price. Tolerance is NOT baked into the
    zone bounds — it is applied separately at evaluation time by
    is_price_in_zone().

    Args:
        entry_reference: The analyst's intended entry price.
        direction: "BUY" (long) or "SHORT".
        stop_price: The protective stop price for the trade.
        tolerance_pct: Optional override for zone tolerance (fraction of
            reference). Defaults to PLAN_ENTRY_ZONE_TOLERANCE_PCT from
            gate_config.
        zone_fraction: Fraction of the entry-to-stop distance used to
            compute the zone width on the entry side. Default 0.20.

    Returns:
        EntryZone with upper/lower bounds (tolerance applied separately).
    """
    entry_reference = _CTX.create_decimal(entry_reference)
    stop_price = _CTX.create_decimal(stop_price)

    if tolerance_pct is None:
        tol = _CTX.create_decimal(Decimal(str(PLAN_ENTRY_ZONE_TOLERANCE_PCT)))
    else:
        tol = _CTX.create_decimal(tolerance_pct)

    frac = _CTX.create_decimal(zone_fraction) if zone_fraction is not None else DEFAULT_ZONE_FRACTION

    if direction == "BUY":
        # LONG: upper = entry_reference
        #       lower = entry_reference - (entry_reference - stop_price) * zone_fraction
        distance = _CTX.subtract(entry_reference, stop_price)
        offset = _CTX.multiply(distance, frac)
        upper = entry_reference
        lower = _CTX.subtract(entry_reference, offset)
    else:
        # SHORT: lower = entry_reference
        #        upper = entry_reference + (stop_price - entry_reference) * zone_fraction
        distance = _CTX.subtract(stop_price, entry_reference)
        offset = _CTX.multiply(distance, frac)
        lower = entry_reference
        upper = _CTX.add(entry_reference, offset)

    return EntryZone(
        upper=upper,
        lower=lower,
        reference=entry_reference,
        tolerance_pct=tol,
    )


def is_price_in_zone(price: Decimal, zone: EntryZone, direction: str) -> bool:
    """Check if price is within entry zone bounds (including tolerance).

    Tolerance is applied HERE (once) — not in derive_entry_zone().
    This is the single place where tolerance extends the zone for trigger
    evaluation.

    For both LONG and SHORT:
        lower - tolerance <= price <= upper + tolerance

    Args:
        price: Current market price to evaluate.
        zone: The EntryZone to check against.
        direction: "BUY" or "SHORT" (included for interface symmetry and
            potential future direction-specific tolerance logic).

    Returns:
        True if price is within the toleranced zone bounds.
    """
    price = _CTX.create_decimal(price)
    tolerance = _CTX.multiply(zone.reference, zone.tolerance_pct)

    effective_lower = _CTX.subtract(zone.lower, tolerance)
    effective_upper = _CTX.add(zone.upper, tolerance)

    return effective_lower <= price <= effective_upper


def is_price_past_target(price: Decimal, target_price: Decimal, direction: str) -> bool:
    """Check if price has already crossed the intended target.

    Args:
        price: Current market price.
        target_price: The trade plan's target price.
        direction: "BUY" or "SHORT".

    Returns:
        True if the price has moved past (or exactly at) the target.
    """
    price = _CTX.create_decimal(price)
    target_price = _CTX.create_decimal(target_price)

    if direction == "BUY":
        return price >= target_price
    else:  # SHORT
        return price <= target_price
