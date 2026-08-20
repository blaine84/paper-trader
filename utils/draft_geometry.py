"""Draft Geometry — compute entry zones and draft geometry from trusted key levels.

Pure functions for deriving draft entry/stop/target from analyst signal
key_levels using per-setup-type rules. Uses Decimal arithmetic throughout.
Draft geometry is non-executable context for the PM; actual trade geometry
is always rebuilt via build_entry_geometry_scaffold() at promotion time.

Requirements: 2.1-2.9
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, Context, ROUND_HALF_UP
from typing import Any

from utils.setup_watch_evaluator import _safe_decimal

logger = logging.getLogger(__name__)

_CTX = Context(prec=28, rounding=ROUND_HALF_UP)

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

# Setup types with defined draft-geometry derivation rules
SWING_SETUP_TYPES: frozenset[str] = frozenset({
    "momentum_fade",
    "pullback_continuation",
    "support_bounce_swing",
    "breakout_retest",
    "failed_breakdown_reclaim",
})

# ────────────────────────────────────────────────────────────────────────────
# Data classes
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DraftGeometry:
    """Non-executable draft geometry computed from key levels."""

    entry: Decimal
    stop: Decimal
    target: Decimal
    risk_reward: Decimal  # rounded to 2 decimal places


@dataclass(frozen=True)
class EntryZone:
    """Price zone within which a setup becomes actionable."""

    low: Decimal
    high: Decimal


# ────────────────────────────────────────────────────────────────────────────
# Helper: extract all valid Decimal values from key_levels dict
# ────────────────────────────────────────────────────────────────────────────


def _extract_all_decimals(key_levels: dict) -> list[Decimal]:
    """Extract all numeric values from key_levels, flattening lists."""
    values: list[Decimal] = []
    for v in key_levels.values():
        if isinstance(v, list):
            for item in v:
                parsed = _safe_decimal(item)
                if parsed is not None:
                    values.append(parsed)
        else:
            parsed = _safe_decimal(v)
            if parsed is not None:
                values.append(parsed)
    return values


def _get_level(key_levels: dict, *field_names: str) -> Decimal | None:
    """Get the first valid Decimal from key_levels trying multiple field names.

    If the value is a list, returns the first valid element.
    """
    for name in field_names:
        raw = key_levels.get(name)
        if raw is None:
            continue
        if isinstance(raw, list):
            for item in raw:
                parsed = _safe_decimal(item)
                if parsed is not None:
                    return parsed
        else:
            parsed = _safe_decimal(raw)
            if parsed is not None:
                return parsed
    return None


def _get_all_levels(key_levels: dict, *field_names: str) -> list[Decimal]:
    """Get all valid Decimals for given field names from key_levels.

    Flattens list-valued fields.
    """
    results: list[Decimal] = []
    for name in field_names:
        raw = key_levels.get(name)
        if raw is None:
            continue
        if isinstance(raw, list):
            for item in raw:
                parsed = _safe_decimal(item)
                if parsed is not None:
                    results.append(parsed)
        else:
            parsed = _safe_decimal(raw)
            if parsed is not None:
                results.append(parsed)
    return results


def _compute_risk_reward(entry: Decimal, stop: Decimal, target: Decimal) -> Decimal:
    """Compute risk/reward ratio rounded to 2 decimal places."""
    risk = abs(entry - stop)
    if risk == 0:
        return Decimal("0")
    reward = abs(target - entry)
    rr = _CTX.divide(reward, risk)
    return rr.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────


def compute_entry_zone(
    key_levels: dict,
    side: str,
) -> EntryZone | None:
    """Compute entry zone from at least two trusted numeric key levels.

    Extracts all numeric values from key_levels dict (flattening lists),
    parses them via _safe_decimal(), takes min/max of valid values as
    the zone bounds.

    Returns None if fewer than two valid levels are available.

    Parameters
    ----------
    key_levels : dict
        Signal key_levels dict (vwap, support, resistance, moving_average, etc.)
    side : str
        "BUY" or "SHORT"

    Returns
    -------
    EntryZone | None
    """
    values = _extract_all_decimals(key_levels)
    if len(values) < 2:
        return None
    low = min(values)
    high = max(values)
    if low == high:
        return None
    return EntryZone(low=low, high=high)


def compute_draft_geometry(
    key_levels: dict,
    setup_type: str,
    side: str,
    current_price: Decimal | None = None,
) -> DraftGeometry | None:
    """Compute draft geometry using per-setup-type derivation rules.

    Dispatches to the appropriate derivation function based on setup_type.
    Returns None if:
      - setup_type is not in SWING_SETUP_TYPES
      - required levels for the setup type are not present
      - directional consistency fails

    Parameters
    ----------
    key_levels : dict
        Combined dict from the signal containing key_levels sub-dict fields
        AND top-level signal fields (prior_swing_high, broken_level, etc.)
    setup_type : str
        Must be in SWING_SETUP_TYPES for geometry computation
    side : str
        "BUY" or "SHORT"
    current_price : Decimal | None
        Used by some derivation rules as a reference

    Returns
    -------
    DraftGeometry | None
    """
    if setup_type not in SWING_SETUP_TYPES:
        return None

    side_upper = side.upper()

    derivation_map = {
        "support_bounce_swing": _derive_support_bounce_swing,
        "pullback_continuation": _derive_pullback_continuation,
        "momentum_fade": _derive_momentum_fade,
        "breakout_retest": _derive_breakout_retest,
        "failed_breakdown_reclaim": _derive_failed_breakdown_reclaim,
    }

    derive_fn = derivation_map.get(setup_type)
    if derive_fn is None:
        return None

    geom = derive_fn(key_levels, side_upper)
    if geom is None:
        return None

    if not _validate_directional_consistency(geom, side_upper):
        logger.debug(
            "Draft geometry failed directional consistency for %s %s: "
            "entry=%s stop=%s target=%s",
            side_upper,
            setup_type,
            geom.entry,
            geom.stop,
            geom.target,
        )
        return None

    return geom


# ────────────────────────────────────────────────────────────────────────────
# Per-setup-type derivation functions
# ────────────────────────────────────────────────────────────────────────────


def _derive_support_bounce_swing(levels: dict, side: str) -> DraftGeometry | None:
    """support_bounce_swing: entry = support + 0.2%, stop = support - 0.5%, target = next resistance above entry.

    Required signal fields: key_levels.support, key_levels.resistance
    """
    support = _get_level(levels, "support")
    if support is None:
        return None

    resistance_levels = _get_all_levels(levels, "resistance")
    if not resistance_levels:
        return None

    # Entry = support + 0.2% buffer
    buffer_entry = _CTX.multiply(support, Decimal("0.002"))
    entry = _CTX.add(support, buffer_entry)

    # Stop = support - 0.5% buffer
    buffer_stop = _CTX.multiply(support, Decimal("0.005"))
    stop = _CTX.subtract(support, buffer_stop)

    # Target = first resistance above entry
    resistances_above = sorted([r for r in resistance_levels if r > entry])
    if not resistances_above:
        return None
    target = resistances_above[0]

    risk_reward = _compute_risk_reward(entry, stop, target)

    return DraftGeometry(entry=entry, stop=stop, target=target, risk_reward=risk_reward)


def _derive_pullback_continuation(levels: dict, side: str) -> DraftGeometry | None:
    """pullback_continuation: entry = moving_average, stop = below next support, target = prior_swing_high (BUY) / prior_swing_low (SHORT).

    Required signal fields: key_levels.moving_average OR key_levels.ma_20,
    key_levels.support, AND signal.prior_swing_high (BUY) or signal.prior_swing_low (SHORT)
    """
    # Entry = moving average
    entry = _get_level(levels, "moving_average", "ma_20")
    if entry is None:
        return None

    # Stop = below next lower support
    support_levels = _get_all_levels(levels, "support")
    supports_below = sorted([s for s in support_levels if s < entry], reverse=True)
    if not supports_below:
        return None
    next_support = supports_below[0]
    # Stop is placed below the support level by 0.5%
    buffer_stop = _CTX.multiply(next_support, Decimal("0.005"))
    stop = _CTX.subtract(next_support, buffer_stop)

    # Target = prior_swing_high (BUY) or prior_swing_low (SHORT)
    if side == "BUY":
        target = _get_level(levels, "prior_swing_high")
    else:
        target = _get_level(levels, "prior_swing_low")

    if target is None:
        return None

    risk_reward = _compute_risk_reward(entry, stop, target)

    return DraftGeometry(entry=entry, stop=stop, target=target, risk_reward=risk_reward)


def _derive_momentum_fade(levels: dict, side: str) -> DraftGeometry | None:
    """momentum_fade: entry = faded level, stop = beyond by 1%, target = VWAP.

    For SHORT: entry = resistance, stop = resistance + 1%, target = vwap
    For BUY: entry = support, stop = support - 1%, target = vwap

    Required signal fields: key_levels.resistance (SHORT) or key_levels.support (BUY),
    key_levels.vwap
    """
    # Determine the faded level based on side
    if side == "SHORT":
        faded_level = _get_level(levels, "resistance")
    else:
        faded_level = _get_level(levels, "support")

    if faded_level is None:
        return None

    # Target = VWAP
    target = _get_level(levels, "vwap")
    if target is None:
        return None

    entry = faded_level

    # Stop = beyond the level by 1%
    buffer = _CTX.multiply(faded_level, Decimal("0.01"))
    if side == "SHORT":
        # SHORT: stop is above the resistance
        stop = _CTX.add(faded_level, buffer)
    else:
        # BUY: stop is below the support
        stop = _CTX.subtract(faded_level, buffer)

    risk_reward = _compute_risk_reward(entry, stop, target)

    return DraftGeometry(entry=entry, stop=stop, target=target, risk_reward=risk_reward)


def _derive_breakout_retest(levels: dict, side: str) -> DraftGeometry | None:
    """breakout_retest: entry = broken level, stop = below 0.5%, target = measured_move_target.

    Required signal fields: signal.broken_level OR key_levels.breakout_level,
    signal.measured_move_target
    """
    # Entry = broken level on retest
    entry = _get_level(levels, "broken_level", "breakout_level")
    if entry is None:
        return None

    # Target = measured move target
    target = _get_level(levels, "measured_move_target")
    if target is None:
        return None

    # Stop = below the level by 0.5%
    buffer = _CTX.multiply(entry, Decimal("0.005"))
    if side == "BUY":
        stop = _CTX.subtract(entry, buffer)
    else:
        # SHORT: stop above the level
        stop = _CTX.add(entry, buffer)

    risk_reward = _compute_risk_reward(entry, stop, target)

    return DraftGeometry(entry=entry, stop=stop, target=target, risk_reward=risk_reward)


def _derive_failed_breakdown_reclaim(levels: dict, side: str) -> DraftGeometry | None:
    """failed_breakdown_reclaim: entry = reclaimed + 0.2%, stop = below breakdown - 0.3%, target = prior resistance.

    Required signal fields: signal.reclaimed_level OR key_levels.breakdown_level,
    signal.breakdown_low, key_levels.resistance
    """
    # Entry = reclaimed level + 0.2% buffer
    reclaimed = _get_level(levels, "reclaimed_level", "breakdown_level")
    if reclaimed is None:
        return None

    # Stop = below breakdown low - 0.3%
    breakdown_low = _get_level(levels, "breakdown_low")
    if breakdown_low is None:
        return None

    # Target = prior resistance (first element above entry)
    resistance_levels = _get_all_levels(levels, "resistance")
    if not resistance_levels:
        return None

    # Entry = reclaimed + 0.2%
    buffer_entry = _CTX.multiply(reclaimed, Decimal("0.002"))
    entry = _CTX.add(reclaimed, buffer_entry)

    # Stop = breakdown_low - 0.3%
    buffer_stop = _CTX.multiply(breakdown_low, Decimal("0.003"))
    stop = _CTX.subtract(breakdown_low, buffer_stop)

    # Target = first resistance above entry
    resistances_above = sorted([r for r in resistance_levels if r > entry])
    if not resistances_above:
        return None
    target = resistances_above[0]

    risk_reward = _compute_risk_reward(entry, stop, target)

    return DraftGeometry(entry=entry, stop=stop, target=target, risk_reward=risk_reward)


# ────────────────────────────────────────────────────────────────────────────
# Validation
# ────────────────────────────────────────────────────────────────────────────


def _validate_directional_consistency(geom: DraftGeometry, side: str) -> bool:
    """Validate directional consistency of geometry.

    BUY: stop < entry < target
    SHORT: stop > entry > target
    """
    if side == "BUY":
        return geom.stop < geom.entry < geom.target
    elif side == "SHORT":
        return geom.stop > geom.entry > geom.target
    return False


# ────────────────────────────────────────────────────────────────────────────
# Zone comparison
# ────────────────────────────────────────────────────────────────────────────


def should_replace_entry_zone(
    existing_zone_json: str | None,
    new_zone: EntryZone,
) -> bool:
    """Return True if new_zone is strictly tighter (smaller high - low) than existing.

    Returns True (replace) if:
      - existing_zone_json is None or empty
      - existing_zone_json fails to parse
      - new zone width is strictly less than existing zone width
    """
    if not existing_zone_json:
        return True

    try:
        existing = json.loads(existing_zone_json)
    except (json.JSONDecodeError, TypeError):
        return True

    existing_low = _safe_decimal(existing.get("low"))
    existing_high = _safe_decimal(existing.get("high"))

    if existing_low is None or existing_high is None:
        return True

    existing_width = existing_high - existing_low
    new_width = new_zone.high - new_zone.low

    return new_width < existing_width


# ────────────────────────────────────────────────────────────────────────────
# Serialization
# ────────────────────────────────────────────────────────────────────────────


def geometry_to_json(geom: DraftGeometry) -> str:
    """Serialize DraftGeometry to JSON string for storage."""
    return json.dumps({
        "entry": str(geom.entry),
        "stop": str(geom.stop),
        "target": str(geom.target),
        "risk_reward": str(geom.risk_reward),
    })


def entry_zone_to_json(zone: EntryZone) -> str:
    """Serialize EntryZone to JSON string for storage."""
    return json.dumps({
        "low": str(zone.low),
        "high": str(zone.high),
    })
