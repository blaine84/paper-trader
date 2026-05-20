"""
Entry Geometry Scaffold

Deterministic, side-effect-free module that generates candidate trade plans
from Analyst signals and quote context. Sits between the Analyst agent and
the Portfolio Manager (PM) agent to resolve the contract conflict where the
PM prompt displays placeholder geometry the Analyst was never supposed to produce.

Public API:
    build_entry_geometry_scaffold(signal, profile_id, profile_context) -> dict
"""

from __future__ import annotations

import copy
import math
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import TypedDict


# ---------------------------------------------------------------------------
# Constants (in-memory, no I/O)
# ---------------------------------------------------------------------------

DEFAULT_RULES: dict = {
    "stop_buffer_pct": 0.002,           # 0.2% buffer below/above level
    "target_multiplier": 2.0,           # default R:R target multiplier
    "min_risk_reward": 1.0,             # minimum R:R to include candidate
    "max_entry_distance_pct": 0.05,     # 5% max distance from current price
}

PROFILE_RULES: dict[str, dict] = {
    "conservative": {
        "min_risk_reward": 1.5,
        "target_multiplier": 2.5,
        "max_entry_distance_pct": 0.03,
    },
    "moderate": {
        "min_risk_reward": 1.0,
        "target_multiplier": 2.0,
        "max_entry_distance_pct": 0.05,
    },
    "aggressive": {
        "min_risk_reward": 1.0,
        "target_multiplier": 1.5,
        "max_entry_distance_pct": 0.07,
    },
}


# ---------------------------------------------------------------------------
# Type Definitions
# ---------------------------------------------------------------------------

class CandidateDict(TypedDict):
    """Schema for a single candidate trade plan."""

    candidate_id: str
    name: str
    entry_price: float
    stop_loss: float
    target: float
    risk_reward: float
    trigger: str
    invalidation_basis: str
    target_basis: str


class ScaffoldResult(TypedDict):
    """Schema for the scaffold function return value."""

    symbol: str
    direction: str
    source: str
    status: str
    reason: str
    levels_used: list[str]
    candidates: list[CandidateDict]


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

_MAX_STRING_LENGTH = 2000
_MAX_CANDIDATES = 10
_VALID_DIRECTIONS = {"LONG", "SHORT", "HOLD"}
_SIGNAL_TO_DIRECTION = {"LONG": "LONG", "SHORT": "SHORT", "HOLD": "HOLD"}


def _truncate_str(value: str) -> str:
    """Truncate a string to the maximum allowed length."""
    if len(value) > _MAX_STRING_LENGTH:
        return value[:_MAX_STRING_LENGTH]
    return value


def _is_finite_number(value) -> bool:
    """Check if a value is a finite numeric value (int or float, not NaN/Inf)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return False


def _is_valid_tick_size(tick_size) -> bool:
    """Check if tick_size is a valid positive numeric value."""
    if tick_size is None:
        return False
    if isinstance(tick_size, bool):
        return False
    if not isinstance(tick_size, (int, float)):
        return False
    if not math.isfinite(tick_size):
        return False
    return tick_size > 0


def _round_price(value: float, tick_size=None) -> float:
    """Round a price value using Decimal ROUND_HALF_UP.

    If a valid tick_size is provided, rounds to the nearest multiple of tick_size.
    Otherwise, rounds to 2 decimal places.

    Python's built-in round() uses banker's rounding and MUST NOT be used.
    """
    try:
        d = Decimal(str(value))
        if _is_valid_tick_size(tick_size):
            tick_d = Decimal(str(tick_size))
            # Round to nearest multiple of tick_size
            rounded = (d / tick_d).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * tick_d
        else:
            rounded = d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return float(rounded)
    except (InvalidOperation, ValueError, OverflowError):
        # Fallback: return value as-is if Decimal conversion fails
        return value


def _round_rr(value: float) -> float:
    """Round risk_reward to 2 decimal places using Decimal ROUND_HALF_UP.

    Always rounds to 2dp regardless of tick_size.
    """
    try:
        d = Decimal(str(value))
        rounded = d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return float(rounded)
    except (InvalidOperation, ValueError, OverflowError):
        return value


def _sanitize_output(obj):
    """Recursively ensure output contains only dict, list, int, float, str, bool, or None.

    Also enforces string truncation to _MAX_STRING_LENGTH characters.
    Returns a sanitized copy of the object.
    """
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj
    if isinstance(obj, str):
        return obj[:_MAX_STRING_LENGTH] if len(obj) > _MAX_STRING_LENGTH else obj
    if isinstance(obj, dict):
        return {
            _sanitize_output(k): _sanitize_output(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_sanitize_output(item) for item in obj]
    # Unsupported type: convert to string representation, truncated
    s = str(obj)
    return s[:_MAX_STRING_LENGTH] if len(s) > _MAX_STRING_LENGTH else s


def _make_insufficient_data(symbol: str, direction: str, reason: str) -> ScaffoldResult:
    """Build a scaffold result with status 'insufficient_data'."""
    return {
        "symbol": _truncate_str(symbol),
        "direction": direction,
        "source": "deterministic_geometry_scaffold",
        "status": "insufficient_data",
        "reason": _truncate_str(reason[:500] if len(reason) > 500 else reason),
        "levels_used": [],
        "candidates": [],
    }


def _make_not_tradeable(symbol: str, direction: str, reason: str) -> ScaffoldResult:
    """Build a scaffold result with status 'not_tradeable_signal'."""
    return {
        "symbol": _truncate_str(symbol),
        "direction": direction,
        "source": "deterministic_geometry_scaffold",
        "status": "not_tradeable_signal",
        "reason": _truncate_str(reason[:500] if len(reason) > 500 else reason),
        "levels_used": [],
        "candidates": [],
    }


def _resolve_rules(profile_id: str | None, profile_context: dict | None) -> dict:
    """Resolve effective rules from profile_id and profile_context.

    Priority:
    1. Start with DEFAULT_RULES as base.
    2. If profile_id matches a PROFILE_RULES key, overlay those values.
    3. If profile_context is provided, overlay matching keys (unknown keys ignored).
    """
    rules = dict(DEFAULT_RULES)

    if profile_id is not None and profile_id in PROFILE_RULES:
        rules.update(PROFILE_RULES[profile_id])

    if profile_context is not None and isinstance(profile_context, dict):
        for key in DEFAULT_RULES:
            if key in profile_context:
                val = profile_context[key]
                if _is_finite_number(val) and val > 0:
                    rules[key] = val

    return rules


def _extract_levels(signal: dict) -> dict[str, float]:
    """Extract numeric levels from the signal dict.

    Priority order:
    1. key_levels_sanitized (if present and is a dict)
    2. key_levels (if present and is a dict)
    3. Top-level fallback fields (vwap, support, resistance)

    Returns a dict mapping level names to their numeric values.
    Only includes levels that are finite numbers.
    """
    levels: dict[str, float] = {}

    # Determine the source dict for key levels
    source = None
    if "key_levels_sanitized" in signal and isinstance(signal["key_levels_sanitized"], dict):
        source = signal["key_levels_sanitized"]
    elif "key_levels" in signal and isinstance(signal["key_levels"], dict):
        source = signal["key_levels"]

    # Extract from the chosen source
    if source is not None:
        for key, val in source.items():
            if isinstance(key, str) and _is_finite_number(val):
                levels[key.lower()] = float(val)

    # Top-level fallback for standard level names not already found
    for field in ("vwap", "support", "resistance"):
        if field not in levels:
            val = signal.get(field)
            if _is_finite_number(val):
                levels[field] = float(val)

    return levels


def _derive_direction(signal: dict) -> tuple[str | None, str | None]:
    """Derive direction from the signal dict.

    Returns (direction, error_reason).
    If direction can be derived, returns (direction, None).
    If there's a conflict or missing data, returns (None, reason).
    """
    has_direction = "direction" in signal
    has_signal_field = "signal" in signal

    direction_val = signal.get("direction")
    signal_val = signal.get("signal")

    # Normalize to uppercase strings for comparison
    dir_str = str(direction_val).upper().strip() if direction_val is not None else None
    sig_str = str(signal_val).upper().strip() if signal_val is not None else None

    if has_direction and has_signal_field:
        # Both present — check for conflict
        if dir_str in _VALID_DIRECTIONS and sig_str in _VALID_DIRECTIONS:
            if dir_str != sig_str:
                return None, (
                    f"direction/signal conflict: direction='{direction_val}' "
                    f"vs signal='{signal_val}'"
                )
            return dir_str, None
        # If direction is valid, use it
        if dir_str in _VALID_DIRECTIONS:
            return dir_str, None
        # If signal is valid, use it
        if sig_str in _VALID_DIRECTIONS:
            return sig_str, None
        # Neither is valid
        return None, (
            f"invalid direction value: direction='{direction_val}', "
            f"signal='{signal_val}' — expected LONG, SHORT, or HOLD"
        )

    if has_direction:
        if dir_str in _VALID_DIRECTIONS:
            return dir_str, None
        return None, (
            f"invalid direction value: '{direction_val}' — expected LONG, SHORT, or HOLD"
        )

    if has_signal_field:
        if sig_str in _VALID_DIRECTIONS:
            return sig_str, None
        return None, (
            f"invalid signal value: '{signal_val}' — expected LONG, SHORT, or HOLD"
        )

    # Neither field present
    return None, "missing direction/signal field — cannot determine trade direction"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_entry_geometry_scaffold(
    signal: dict,
    profile_id: str | None = None,
    profile_context: dict | None = None,
) -> dict:
    """Generate candidate trade plans from an Analyst signal.

    This function is deterministic and side-effect free. It deep-copies the
    input signal before processing and never mutates the original.

    Args:
        signal: Analyst signal dict (deep-copied internally, never mutated).
        profile_id: Optional profile identifier for rule selection from
                    in-memory constants. Falls back to defaults if unknown.
        profile_context: Optional dict of profile rules (in-memory only).
                         Overrides matching values from PROFILE_RULES.

    Returns:
        A dict conforming to ScaffoldResult schema with keys:
            symbol, direction, source, status, reason, levels_used, candidates
    """
    # --- Input validation: None and type check ---
    if signal is None:
        return _make_insufficient_data("", "", "signal is None — expected a dict")

    if not isinstance(signal, dict):
        return _make_insufficient_data(
            "", "", f"signal is not a dict — got {type(signal).__name__}"
        )

    # --- Deep copy to ensure immutability ---
    sig = copy.deepcopy(signal)

    # --- Extract symbol ---
    raw_symbol = sig.get("symbol", "")
    symbol = _truncate_str(str(raw_symbol)) if raw_symbol is not None else ""

    # --- Derive direction ---
    direction, dir_error = _derive_direction(sig)
    if direction is None:
        return _make_insufficient_data(symbol, "", dir_error)

    # --- HOLD handling (before current_price validation) ---
    if direction == "HOLD":
        return _make_not_tradeable(
            symbol, "HOLD", "signal direction is HOLD — no executable candidates"
        )

    # --- current_price validation (required for LONG/SHORT) ---
    current_price = sig.get("current_price")
    if current_price is None:
        return _make_insufficient_data(
            symbol, direction,
            "current_price is missing — required for non-HOLD signals"
        )
    if not _is_finite_number(current_price):
        return _make_insufficient_data(
            symbol, direction,
            f"current_price is not a finite number — got {current_price!r}"
        )
    current_price = float(current_price)
    if current_price <= 0:
        return _make_insufficient_data(
            symbol, direction,
            f"current_price must be positive — got {current_price}"
        )

    # --- Resolve rules from profile ---
    rules = _resolve_rules(profile_id, profile_context)

    # --- Extract tick_size from signal ---
    raw_tick_size = sig.get("tick_size")
    tick_size = raw_tick_size if _is_valid_tick_size(raw_tick_size) else None

    # --- Extract levels ---
    levels = _extract_levels(sig)
    levels_used = sorted(levels.keys())

    # --- Candidate generation placeholder ---
    # LONG and SHORT candidate generation will be implemented in tasks 1.2 and 1.3.
    # For now, return empty candidates with appropriate status.
    candidates: list[CandidateDict] = []

    if direction == "LONG":
        candidates = _generate_long_candidates(
            levels, current_price, rules, symbol
        )
    elif direction == "SHORT":
        candidates = _generate_short_candidates(
            levels, current_price, rules, symbol
        )

    # --- Apply rounding to all candidates ---
    rounded_candidates: list[CandidateDict] = []
    for candidate in candidates:
        # Round price fields
        entry_rounded = _round_price(candidate["entry_price"], tick_size)
        stop_rounded = _round_price(candidate["stop_loss"], tick_size)
        target_rounded = _round_price(candidate["target"], tick_size)

        # Re-verify directional geometry invariants after rounding
        if direction == "LONG":
            # LONG: stop < entry < target
            if not (stop_rounded < entry_rounded < target_rounded):
                continue  # Discard candidate where rounding broke geometry
        elif direction == "SHORT":
            # SHORT: target < entry < stop
            if not (target_rounded < entry_rounded < stop_rounded):
                continue  # Discard candidate where rounding broke geometry

        # Recompute R:R from rounded values
        if direction == "LONG":
            if entry_rounded == stop_rounded:
                continue  # entry == stop after rounding, discard
            rr = (target_rounded - entry_rounded) / (entry_rounded - stop_rounded)
        else:  # SHORT
            if stop_rounded == entry_rounded:
                continue  # entry == stop after rounding, discard
            rr = (entry_rounded - target_rounded) / (stop_rounded - entry_rounded)

        rr_rounded = _round_rr(rr)

        # Re-check min_risk_reward after rounding
        min_risk_reward = rules.get("min_risk_reward", DEFAULT_RULES["min_risk_reward"])
        if rr_rounded < min_risk_reward:
            continue

        # Update candidate with rounded values
        candidate["entry_price"] = entry_rounded
        candidate["stop_loss"] = stop_rounded
        candidate["target"] = target_rounded
        candidate["risk_reward"] = rr_rounded

        rounded_candidates.append(candidate)

    # --- Cap candidates at 10 ---
    candidates = rounded_candidates[:_MAX_CANDIDATES]

    # --- Determine status based on candidates ---
    if candidates:
        result = {
            "symbol": _truncate_str(symbol),
            "direction": direction,
            "source": "deterministic_geometry_scaffold",
            "status": "ok",
            "reason": "",
            "levels_used": levels_used,
            "candidates": candidates,
        }
    else:
        # Determine which levels were missing
        if direction == "LONG":
            needed = ["vwap", "support", "resistance"]
        else:
            needed = ["resistance", "support", "vwap"]
        missing = [lvl for lvl in needed if lvl not in levels]
        if missing:
            reason = (
                f"no candidates generated — missing required levels: "
                f"{', '.join(missing)}"
            )
        else:
            reason = (
                "no candidates generated — all candidates excluded by "
                "price-distance or risk-reward filters"
            )
        result = _make_insufficient_data(symbol, direction, reason)

    # --- Apply type safety sanitization ---
    return _sanitize_output(result)


# ---------------------------------------------------------------------------
# Candidate Generation Stubs (to be implemented in tasks 1.2 and 1.3)
# ---------------------------------------------------------------------------

def _find_nearest_lower_level(
    entry_price: float,
    levels: dict[str, float],
    current_price: float,
) -> float | None:
    """Find the nearest level below entry_price for LONG stop derivation.

    Searches valid numeric support, VWAP, and current_price — whichever is
    closest below entry. Returns None if no level exists below entry.
    """
    candidates_below: list[float] = []

    for name in ("support", "vwap"):
        if name in levels:
            val = levels[name]
            if _is_finite_number(val) and val < entry_price:
                candidates_below.append(val)

    if _is_finite_number(current_price) and current_price < entry_price:
        candidates_below.append(current_price)

    if not candidates_below:
        return None

    # Return the closest one below entry (i.e., the maximum of those below)
    return max(candidates_below)


def _passes_long_sanity(
    name: str,
    entry_price: float,
    current_price: float,
    max_entry_distance_pct: float,
) -> bool:
    """Check current-price sanity rules for LONG candidates.

    - pullback_to_vwap and support_bounce: entry at or below current_price
      and within max_entry_distance_pct of current_price.
    - breakout_continuation: entry at or above current_price and within
      max_entry_distance_pct of current_price.
    """
    if current_price <= 0:
        return False

    distance_pct = abs(entry_price - current_price) / current_price

    if name in ("pullback_to_vwap", "support_bounce"):
        # Entry must be at or below current price, within distance
        return entry_price <= current_price and distance_pct <= max_entry_distance_pct
    elif name == "breakout_continuation":
        # Entry must be at or above current price, within distance
        return entry_price >= current_price and distance_pct <= max_entry_distance_pct
    return False


def _generate_long_candidates(
    levels: dict[str, float],
    current_price: float,
    rules: dict,
    symbol: str,
) -> list[CandidateDict]:
    """Generate LONG candidate trade plans.

    Generates up to 3 candidate types:
    - pullback_to_vwap: entry from VWAP level
    - support_bounce: entry from support level
    - breakout_continuation: entry from resistance level

    Each candidate is validated against current-price sanity rules,
    entry != stop check, and minimum risk-reward threshold.
    """
    stop_buffer_pct = rules.get("stop_buffer_pct", DEFAULT_RULES["stop_buffer_pct"])
    target_multiplier = rules.get("target_multiplier", DEFAULT_RULES["target_multiplier"])
    min_risk_reward = rules.get("min_risk_reward", DEFAULT_RULES["min_risk_reward"])
    max_entry_distance_pct = rules.get("max_entry_distance_pct", DEFAULT_RULES["max_entry_distance_pct"])

    candidates: list[CandidateDict] = []
    ordinal = 0
    norm_symbol = symbol.lower().strip() if symbol else ""

    # --- pullback_to_vwap ---
    if "vwap" in levels:
        vwap = levels["vwap"]
        entry_price = vwap
        buffer = entry_price * stop_buffer_pct
        stop_loss = entry_price - buffer

        if _passes_long_sanity("pullback_to_vwap", entry_price, current_price, max_entry_distance_pct):
            if entry_price != stop_loss:
                risk = entry_price - stop_loss
                target = entry_price + risk * target_multiplier
                risk_reward = (target - entry_price) / (entry_price - stop_loss)

                if risk_reward >= min_risk_reward:
                    ordinal += 1
                    candidates.append({
                        "candidate_id": f"{norm_symbol}_long_pullback_to_vwap_{ordinal}",
                        "name": "pullback_to_vwap",
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "target": target,
                        "risk_reward": risk_reward,
                        "trigger": "Price pulls back to VWAP level",
                        "invalidation_basis": "Price breaks below VWAP minus buffer",
                        "target_basis": f"Entry + (entry - stop) x {target_multiplier} target multiplier",
                    })

    # --- support_bounce ---
    if "support" in levels:
        support = levels["support"]
        entry_price = support
        buffer = entry_price * stop_buffer_pct
        stop_loss = entry_price - buffer

        if _passes_long_sanity("support_bounce", entry_price, current_price, max_entry_distance_pct):
            if entry_price != stop_loss:
                risk = entry_price - stop_loss
                target = entry_price + risk * target_multiplier
                risk_reward = (target - entry_price) / (entry_price - stop_loss)

                if risk_reward >= min_risk_reward:
                    ordinal += 1
                    candidates.append({
                        "candidate_id": f"{norm_symbol}_long_support_bounce_{ordinal}",
                        "name": "support_bounce",
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "target": target,
                        "risk_reward": risk_reward,
                        "trigger": "Price bounces off support level",
                        "invalidation_basis": "Price breaks below support minus buffer",
                        "target_basis": f"Entry + (entry - stop) x {target_multiplier} target multiplier",
                    })

    # --- breakout_continuation ---
    if "resistance" in levels:
        resistance = levels["resistance"]
        entry_price = resistance
        buffer = entry_price * stop_buffer_pct

        # Stop derivation: entry - (resistance - nearest_lower_level) or entry - buffer
        nearest_lower = _find_nearest_lower_level(entry_price, levels, current_price)
        if nearest_lower is not None:
            stop_distance = entry_price - nearest_lower
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price - buffer

        if _passes_long_sanity("breakout_continuation", entry_price, current_price, max_entry_distance_pct):
            if entry_price != stop_loss:
                risk = entry_price - stop_loss
                target = entry_price + risk * target_multiplier
                risk_reward = (target - entry_price) / (entry_price - stop_loss)

                if risk_reward >= min_risk_reward:
                    ordinal += 1
                    candidates.append({
                        "candidate_id": f"{norm_symbol}_long_breakout_continuation_{ordinal}",
                        "name": "breakout_continuation",
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "target": target,
                        "risk_reward": risk_reward,
                        "trigger": "Price breaks above resistance level",
                        "invalidation_basis": "Price falls back below resistance minus stop distance",
                        "target_basis": f"Entry + (entry - stop) x {target_multiplier} target multiplier",
                    })

    return candidates


def _resolve_nearest_upper_level(
    entry_price: float,
    levels: dict[str, float],
    current_price: float,
    buffer: float,
) -> float:
    """Find the nearest level above entry_price for SHORT stop derivation.

    Searches valid numeric resistance, VWAP, and current_price — whichever
    is closest above entry. If none exists, falls back to entry + buffer.
    """
    candidates_above: list[float] = []

    # Consider resistance, vwap, and current_price as potential upper levels
    for level_name in ("resistance", "vwap"):
        val = levels.get(level_name)
        if val is not None and val > entry_price:
            candidates_above.append(val)

    if current_price > entry_price:
        candidates_above.append(current_price)

    if candidates_above:
        return min(candidates_above)  # nearest above = smallest value above entry
    return entry_price + buffer


def _generate_short_candidates(
    levels: dict[str, float],
    current_price: float,
    rules: dict,
    symbol: str,
) -> list[CandidateDict]:
    """Generate SHORT candidate trade plans.

    Candidate types:
    - resistance_rejection: entry from resistance, stop = resistance + buffer
    - breakdown_continuation: entry from support, stop = entry + (nearest_upper - entry) or buffer
    - fade: entry from VWAP, stop = VWAP + buffer

    All targets derived as: entry - (stop - entry) * target_multiplier
    For SHORT: stop_loss > entry_price > target must hold.
    """
    stop_buffer_pct = rules["stop_buffer_pct"]
    target_multiplier = rules["target_multiplier"]
    min_risk_reward = rules["min_risk_reward"]
    max_entry_distance_pct = rules["max_entry_distance_pct"]

    norm_symbol = symbol.lower()
    candidates: list[CandidateDict] = []
    ordinal = 0

    # --- resistance_rejection ---
    # Required level: resistance
    # Entry: resistance level
    # Sanity: entry >= current_price and within max_entry_distance_pct
    if "resistance" in levels:
        entry_price = levels["resistance"]
        buffer = entry_price * stop_buffer_pct
        stop_loss = entry_price + buffer

        # Current-price sanity: entry must be at or above current_price
        # and within max_entry_distance_pct of current_price
        if entry_price >= current_price:
            distance_pct = (entry_price - current_price) / current_price if current_price > 0 else float("inf")
            if distance_pct <= max_entry_distance_pct:
                # entry == stop discard
                if stop_loss != entry_price:
                    risk = stop_loss - entry_price  # positive for SHORT
                    target = entry_price - risk * target_multiplier

                    # Verify SHORT geometry: stop > entry > target
                    if stop_loss > entry_price and entry_price > target:
                        risk_reward = (entry_price - target) / (stop_loss - entry_price)

                        if risk_reward >= min_risk_reward:
                            ordinal += 1
                            candidates.append({
                                "candidate_id": f"{norm_symbol}_short_resistance_rejection_{ordinal}",
                                "name": "resistance_rejection",
                                "entry_price": entry_price,
                                "stop_loss": stop_loss,
                                "target": target,
                                "risk_reward": risk_reward,
                                "trigger": f"Price rejected at resistance {entry_price}",
                                "invalidation_basis": f"Price closes above stop {stop_loss}",
                                "target_basis": f"Entry - (stop - entry) × {target_multiplier} target multiplier",
                            })

    # --- breakdown_continuation ---
    # Required level: support
    # Entry: support level
    # Sanity: entry <= current_price and within max_entry_distance_pct
    if "support" in levels:
        entry_price = levels["support"]
        buffer = entry_price * stop_buffer_pct

        # Stop derivation: entry + (nearest_upper_level - entry) or entry + buffer
        nearest_upper = _resolve_nearest_upper_level(
            entry_price, levels, current_price, buffer
        )
        stop_distance = nearest_upper - entry_price
        # If nearest_upper == entry (shouldn't happen but defensive), use buffer
        if stop_distance <= 0:
            stop_distance = buffer
        stop_loss = entry_price + stop_distance

        # Current-price sanity: entry must be at or below current_price
        # and within max_entry_distance_pct of current_price
        if entry_price <= current_price:
            distance_pct = (current_price - entry_price) / current_price if current_price > 0 else float("inf")
            if distance_pct <= max_entry_distance_pct:
                # entry == stop discard
                if stop_loss != entry_price:
                    risk = stop_loss - entry_price  # positive for SHORT
                    target = entry_price - risk * target_multiplier

                    # Verify SHORT geometry: stop > entry > target
                    if stop_loss > entry_price and entry_price > target:
                        risk_reward = (entry_price - target) / (stop_loss - entry_price)

                        if risk_reward >= min_risk_reward:
                            ordinal += 1
                            candidates.append({
                                "candidate_id": f"{norm_symbol}_short_breakdown_continuation_{ordinal}",
                                "name": "breakdown_continuation",
                                "entry_price": entry_price,
                                "stop_loss": stop_loss,
                                "target": target,
                                "risk_reward": risk_reward,
                                "trigger": f"Price breaks below support {entry_price}",
                                "invalidation_basis": f"Price recovers above stop {stop_loss}",
                                "target_basis": f"Entry - (stop - entry) × {target_multiplier} target multiplier",
                            })

    # --- fade ---
    # Required level: VWAP
    # Entry: VWAP level
    # Sanity: entry >= current_price and within max_entry_distance_pct
    if "vwap" in levels:
        entry_price = levels["vwap"]
        buffer = entry_price * stop_buffer_pct
        stop_loss = entry_price + buffer

        # Current-price sanity: entry must be at or above current_price
        # and within max_entry_distance_pct of current_price
        if entry_price >= current_price:
            distance_pct = (entry_price - current_price) / current_price if current_price > 0 else float("inf")
            if distance_pct <= max_entry_distance_pct:
                # entry == stop discard
                if stop_loss != entry_price:
                    risk = stop_loss - entry_price  # positive for SHORT
                    target = entry_price - risk * target_multiplier

                    # Verify SHORT geometry: stop > entry > target
                    if stop_loss > entry_price and entry_price > target:
                        risk_reward = (entry_price - target) / (stop_loss - entry_price)

                        if risk_reward >= min_risk_reward:
                            ordinal += 1
                            candidates.append({
                                "candidate_id": f"{norm_symbol}_short_fade_{ordinal}",
                                "name": "fade",
                                "entry_price": entry_price,
                                "stop_loss": stop_loss,
                                "target": target,
                                "risk_reward": risk_reward,
                                "trigger": f"Price fades at VWAP {entry_price}",
                                "invalidation_basis": f"Price closes above stop {stop_loss}",
                                "target_basis": f"Entry - (stop - entry) × {target_multiplier} target multiplier",
                            })

    return candidates
