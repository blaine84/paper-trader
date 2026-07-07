"""Swing Geometry Builder — constructs swing candidate geometry or produces rejections.

Pure module: no side effects, no logging, no DB, no network.
All price/ratio arithmetic uses Decimal with 28-digit precision and ROUND_HALF_UP.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_HALF_UP
from typing import Literal


DECIMAL_CTX = Context(prec=28, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class SwingGeometry:
    """Immutable swing candidate geometry."""

    symbol: str
    direction: Literal["LONG", "SHORT"]
    normalized_setup_type: str
    entry_price: Decimal
    stop_price: Decimal
    target_price: Decimal
    risk_reward: Decimal
    holding_horizon: int  # trading days, 2-10 inclusive
    invalidation_basis: str
    source_signal_id: str


@dataclass(frozen=True)
class GeometryRejection:
    """Immutable rejection record for failed geometry construction."""

    symbol: str
    direction: str
    normalized_setup_type: str
    reason_code: str
    computed_values: dict
    source_signal_id: str


GEOMETRY_REJECTION_CODES = frozenset({
    "stop_too_tight_for_swing",
    "insufficient_risk_reward",
    "zero_risk_distance",
    "missing_geometry",
    "non_finite_price",
    "zero_price",
    "negative_price",
    "invalid_directional_order",
})


# Per-setup holding horizon ranges (narrower sub-ranges within [2, 10])
SETUP_HOLDING_HORIZONS: dict[str, tuple[int, int]] = {
    "sector_rotation_swing": (3, 8),
    "risk_off_macro_short": (2, 5),
    "breakout_retest": (2, 5),
    "pullback_continuation": (3, 7),
    "relative_strength_swing": (4, 10),
    "support_bounce_swing": (2, 5),
    "failed_breakdown_reclaim": (2, 4),
}


# Geometry-level minimum R:R thresholds per profile.
SWING_GEOMETRY_MIN_RR: dict[str, Decimal] = {
    "conservative": Decimal("2.0"),
    "moderate": Decimal("1.5"),
    "aggressive": Decimal("1.25"),
}


def build_swing_geometry(
    symbol: str,
    direction: Literal["LONG", "SHORT"],
    normalized_setup_type: str,
    entry_price: Decimal,
    stop_price: Decimal | None,
    target_price: Decimal | None,
    source_signal_id: str,
    profile_id: str,
    *,
    atr: Decimal | None = None,
) -> SwingGeometry | GeometryRejection:
    """Construct swing geometry or produce a rejection.

    Pure function — no side effects. Uses Decimal arithmetic with 28-digit
    precision and ROUND_HALF_UP rounding for all price/ratio computations.

    Validation order:
    1. Missing geometry check (stop or target is None)
    2. Non-finite / zero / negative price validation
    3. Zero risk distance check (stop == entry)
    4. Stop too tight check (< 1.5% of entry)
    5. Directional ordering validation
    6. Risk/reward computation and threshold check
    7. Holding horizon assignment
    """

    def _reject(reason_code: str, computed_values: dict) -> GeometryRejection:
        return GeometryRejection(
            symbol=symbol,
            direction=direction,
            normalized_setup_type=normalized_setup_type,
            reason_code=reason_code,
            computed_values=computed_values,
            source_signal_id=source_signal_id,
        )

    # 1. Missing geometry
    if stop_price is None or target_price is None:
        return _reject("missing_geometry", {})

    # 2. Non-finite / zero / negative price validation
    for field_name, price in [
        ("entry_price", entry_price),
        ("stop_price", stop_price),
        ("target_price", target_price),
    ]:
        price_float = float(price)
        if math.isnan(price_float) or math.isinf(price_float):
            return _reject("non_finite_price", {"field": field_name, "value": str(price)})
        if price == 0:
            return _reject("zero_price", {"field": field_name})
        if price < 0:
            return _reject("negative_price", {"field": field_name, "value": str(price)})

    # 3. Zero risk distance
    if stop_price == entry_price:
        return _reject("zero_risk_distance", {})

    # 4. Stop too tight (< 1.5% of entry)
    stop_distance_pct = DECIMAL_CTX.divide(
        abs(entry_price - stop_price), entry_price
    )
    if stop_distance_pct < Decimal("0.015"):
        return _reject(
            "stop_too_tight_for_swing",
            {"stop_distance_pct": str(stop_distance_pct)},
        )

    # 5. Directional ordering validation
    if direction == "LONG":
        if not (stop_price < entry_price < target_price):
            return _reject(
                "invalid_directional_order",
                {
                    "direction": direction,
                    "entry": str(entry_price),
                    "stop": str(stop_price),
                    "target": str(target_price),
                },
            )
    else:  # SHORT
        if not (target_price < entry_price < stop_price):
            return _reject(
                "invalid_directional_order",
                {
                    "direction": direction,
                    "entry": str(entry_price),
                    "stop": str(stop_price),
                    "target": str(target_price),
                },
            )

    # 6. R:R computation and threshold check
    risk = abs(entry_price - stop_price)
    reward = abs(target_price - entry_price)
    risk_reward = DECIMAL_CTX.divide(reward, risk)

    min_rr = SWING_GEOMETRY_MIN_RR.get(profile_id)
    if min_rr is not None and risk_reward < min_rr:
        return _reject(
            "insufficient_risk_reward",
            {"risk_reward": str(risk_reward), "min_required": str(min_rr)},
        )

    # 7. Holding horizon assignment
    horizon_range = SETUP_HOLDING_HORIZONS.get(normalized_setup_type, (2, 10))
    holding_horizon = (horizon_range[0] + horizon_range[1]) // 2

    # Build invalidation basis description
    if direction == "LONG":
        invalidation_basis = f"Below stop at {stop_price}"
    else:
        invalidation_basis = f"Above stop at {stop_price}"

    return SwingGeometry(
        symbol=symbol,
        direction=direction,
        normalized_setup_type=normalized_setup_type,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        risk_reward=risk_reward,
        holding_horizon=holding_horizon,
        invalidation_basis=invalidation_basis,
        source_signal_id=source_signal_id,
    )
