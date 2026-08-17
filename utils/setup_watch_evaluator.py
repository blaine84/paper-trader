"""Setup Watch Evaluator — condition evaluation and maturity scoring.

Pure evaluation functions that take condition definitions and current market
context, returning evaluation results. Condition definitions are never mutated.
The caller (setup_watch_manager) handles state transitions and persistence.

All price comparisons use ``decimal.Decimal`` for exactness.  No database
access, no side effects — every function is referentially transparent given
identical inputs.

Requirements: 4.2, 4.2.1, 4.3, 4.12, 5.1-5.3, 5.3.1, 5.7, 12.8
"""
from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Data classes
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConditionResult:
    """Result of evaluating a single condition."""

    condition_type: str
    met: bool
    detail: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate result of evaluating all conditions for a watch."""

    invalidated: bool
    invalidation_reason: str | None
    maturity_score: float
    condition_results: list[ConditionResult]
    evaluation_timestamp: str  # ISO8601 UTC


# ────────────────────────────────────────────────────────────────────────────
# Supported condition types
# ────────────────────────────────────────────────────────────────────────────

_SUPPORTED_MATURATION_TYPES: frozenset[str] = frozenset({
    "price_zone",
    "regime_aligned",
    "catalyst_fresh",
    "time_window",
    "key_level_proximity",
})

_SUPPORTED_INVALIDATION_TYPES: frozenset[str] = frozenset({
    "price_breach",
    "regime_flip",
    "catalyst_expired",
    "exposure_conflict",
})


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────


def evaluate_watch(
    maturation_conditions_json: str,
    invalidation_conditions_json: str,
    market_context: dict,
) -> EvaluationResult:
    """Evaluate a watch's conditions against current market state.

    Evaluates invalidation FIRST (short-circuits on first triggered condition),
    then maturation. Never mutates the input JSON strings or parsed structures.

    Parameters
    ----------
    maturation_conditions_json : str
        JSON array of condition definitions with type, params, weight.
    invalidation_conditions_json : str
        JSON array of invalidation condition definitions with type, params.
    market_context : dict
        Current market state built by the manager.

    Returns
    -------
    EvaluationResult
        Aggregate evaluation with invalidation status and maturity score.
    """
    # Deep-copy market_context to guarantee no mutation of caller's data
    ctx = copy.deepcopy(market_context)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # --- Invalidation (priority: evaluate first, short-circuit) ---
    invalidated, invalidation_reason = evaluate_invalidation_conditions(
        invalidation_conditions_json, ctx
    )

    if invalidated:
        return EvaluationResult(
            invalidated=True,
            invalidation_reason=invalidation_reason,
            maturity_score=0.0,
            condition_results=[],
            evaluation_timestamp=timestamp,
        )

    # --- Maturation ---
    score, condition_results = evaluate_maturation_conditions(
        maturation_conditions_json, ctx
    )

    return EvaluationResult(
        invalidated=False,
        invalidation_reason=None,
        maturity_score=score,
        condition_results=condition_results,
        evaluation_timestamp=timestamp,
    )


def evaluate_invalidation_conditions(
    conditions_json: str,
    market_context: dict,
) -> tuple[bool, str | None]:
    """Evaluate invalidation conditions — short-circuit on first triggered.

    Parameters
    ----------
    conditions_json : str
        JSON array of invalidation condition definitions.
    market_context : dict
        Current market state.

    Returns
    -------
    tuple[bool, str | None]
        (True, reason) if invalidated, (False, None) otherwise.
    """
    conditions = json.loads(conditions_json)

    for cond in conditions:
        cond_type = cond.get("type", "")
        params = cond.get("params", {})

        if cond_type not in _SUPPORTED_INVALIDATION_TYPES:
            logger.debug(
                "Unknown invalidation condition type %r — treating as NOT triggered",
                cond_type,
            )
            continue

        triggered, detail = _evaluate_single_invalidation(cond_type, params, market_context)
        if triggered:
            reason = f"{cond_type}: {detail}" if detail else cond_type
            return True, reason

    return False, None


def evaluate_maturation_conditions(
    conditions_json: str,
    market_context: dict,
) -> tuple[float, list[ConditionResult]]:
    """Evaluate maturation conditions and compute weighted score.

    Unknown condition types are treated as unmet AND excluded from the
    score denominator (their weight does not count against the watch).

    Parameters
    ----------
    conditions_json : str
        JSON array of maturation condition definitions.
    market_context : dict
        Current market state.

    Returns
    -------
    tuple[float, list[ConditionResult]]
        (maturity_score clamped [0.0, 1.0], per-condition results).
    """
    conditions = json.loads(conditions_json)

    results: list[ConditionResult] = []
    total_applicable_weight = Decimal("0")
    met_weight = Decimal("0")

    for cond in conditions:
        cond_type = cond.get("type", "")
        params = cond.get("params", {})
        weight_raw = cond.get("weight", 1.0)

        try:
            weight = Decimal(str(weight_raw))
        except (InvalidOperation, TypeError, ValueError):
            weight = Decimal("1")

        if cond_type not in _SUPPORTED_MATURATION_TYPES:
            # Unknown type: unmet, excluded from denominator
            logger.debug(
                "Unknown maturation condition type %r — treating as unmet, "
                "excluding from denominator",
                cond_type,
            )
            results.append(ConditionResult(
                condition_type=cond_type,
                met=False,
                detail="unknown condition type — excluded from scoring",
            ))
            continue

        # Supported type: evaluate
        met, detail = _evaluate_single_maturation(cond_type, params, market_context)
        total_applicable_weight += weight
        if met:
            met_weight += weight

        results.append(ConditionResult(
            condition_type=cond_type,
            met=met,
            detail=detail,
        ))

    # Compute score
    if total_applicable_weight == Decimal("0"):
        logger.debug(
            "All maturation conditions unknown/excluded — applicable weight is zero, "
            "returning score 0.0"
        )
        score = 0.0
    else:
        raw_score = float(met_weight / total_applicable_weight)
        score = max(0.0, min(1.0, raw_score))

    return score, results


def validate_draft_geometry(geometry_json: str | None, side: str) -> bool:
    """Validate draft geometry ordering using Decimal comparison.

    For BUY: stop < entry < target.
    For SHORT: stop > entry > target.

    Parameters
    ----------
    geometry_json : str | None
        JSON object with "entry", "stop", "target" keys, or None.
    side : str
        "BUY" or "SHORT".

    Returns
    -------
    bool
        True if valid or no geometry provided, False if inverted/invalid.
    """
    if not geometry_json:
        return True

    try:
        geom = json.loads(geometry_json)
    except (json.JSONDecodeError, TypeError):
        return True  # Unparseable treated as absent

    entry_raw = geom.get("entry")
    stop_raw = geom.get("stop")
    target_raw = geom.get("target")

    if entry_raw is None or stop_raw is None or target_raw is None:
        return True  # Incomplete geometry is not invalid

    try:
        entry = Decimal(str(entry_raw))
        stop = Decimal(str(stop_raw))
        target = Decimal(str(target_raw))
    except (InvalidOperation, TypeError, ValueError):
        return False  # Non-numeric values are invalid

    side_upper = side.upper()
    if side_upper == "BUY":
        return stop < entry < target
    elif side_upper == "SHORT":
        return stop > entry > target
    else:
        # Unknown side — can't validate ordering
        return False


# ────────────────────────────────────────────────────────────────────────────
# Maturation condition handlers
# ────────────────────────────────────────────────────────────────────────────


def _evaluate_single_maturation(
    cond_type: str, params: dict, market_context: dict
) -> tuple[bool, str | None]:
    """Dispatch a single maturation condition to its handler.

    Returns (met, detail).
    """
    handler = _MATURATION_HANDLERS.get(cond_type)
    if handler is None:
        # Should not reach here since we filter above, but just in case
        return False, "no handler"
    return handler(params, market_context)


def _handle_price_zone(params: dict, ctx: dict) -> tuple[bool, str | None]:
    """price_zone: price within [low, high] range (Decimal)."""
    current_price_raw = ctx.get("current_price")
    if current_price_raw is None:
        return False, "current_price not available"

    try:
        price = Decimal(str(current_price_raw))
        low = Decimal(str(params.get("low", 0)))
        high = Decimal(str(params.get("high", 0)))
    except (InvalidOperation, TypeError, ValueError):
        return False, "invalid numeric values"

    met = low <= price <= high
    detail = f"price {price} {'within' if met else 'outside'} [{low}, {high}]"
    return met, detail


def _handle_regime_aligned(params: dict, ctx: dict) -> tuple[bool, str | None]:
    """regime_aligned: current regime matches required regime."""
    current_regime = str(ctx.get("market_regime", "")).lower()
    required_regime = str(params.get("required_regime", "")).lower()

    if not required_regime:
        return False, "no required_regime specified"

    met = current_regime == required_regime
    detail = f"regime={current_regime} {'matches' if met else 'does not match'} required={required_regime}"
    return met, detail


def _handle_catalyst_fresh(params: dict, ctx: dict) -> tuple[bool, str | None]:
    """catalyst_fresh: catalyst age within max_age_minutes."""
    max_age_minutes = params.get("max_age_minutes")
    if max_age_minutes is None:
        return False, "no max_age_minutes specified"

    catalyst_ts = ctx.get("catalyst_timestamp")
    if not catalyst_ts:
        return False, "no catalyst_timestamp available"

    try:
        if isinstance(catalyst_ts, datetime):
            cat_dt = catalyst_ts if catalyst_ts.tzinfo else catalyst_ts.replace(tzinfo=timezone.utc)
        else:
            # Parse ISO8601 string
            ts_str = str(catalyst_ts)
            # Handle both Z suffix and +00:00
            ts_str = ts_str.replace("Z", "+00:00")
            cat_dt = datetime.fromisoformat(ts_str)
            if cat_dt.tzinfo is None:
                cat_dt = cat_dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return False, "unparseable catalyst_timestamp"

    now = datetime.now(timezone.utc)
    age_minutes = (now - cat_dt).total_seconds() / 60.0

    met = age_minutes <= float(max_age_minutes)
    detail = f"catalyst_age={age_minutes:.1f}min {'<=' if met else '>'} max={max_age_minutes}min"
    return met, detail


def _handle_time_window(params: dict, ctx: dict) -> tuple[bool, str | None]:
    """time_window: current ET hour within [start_hour, end_hour] range."""
    start_hour = params.get("start_hour")
    end_hour = params.get("end_hour")

    if start_hour is None or end_hour is None:
        return False, "start_hour or end_hour not specified"

    current_hour = ctx.get("current_hour_et")
    if current_hour is None:
        return False, "current_hour_et not available"

    try:
        current_h = int(current_hour)
        start_h = int(start_hour)
        end_h = int(end_hour)
    except (ValueError, TypeError):
        return False, "invalid hour values"

    # Support wrapping (e.g., start=22, end=6 means 22-23 and 0-6)
    if start_h <= end_h:
        met = start_h <= current_h <= end_h
    else:
        # Wraps around midnight
        met = current_h >= start_h or current_h <= end_h

    detail = f"hour={current_h} {'within' if met else 'outside'} [{start_h}, {end_h}]"
    return met, detail


def _handle_key_level_proximity(params: dict, ctx: dict) -> tuple[bool, str | None]:
    """key_level_proximity: price within N% of a key level (support/resistance)."""
    level_type = str(params.get("level_type", "")).lower()
    within_pct = params.get("within_pct")

    if not level_type or within_pct is None:
        return False, "level_type or within_pct not specified"

    current_price_raw = ctx.get("current_price")
    if current_price_raw is None:
        return False, "current_price not available"

    key_levels = ctx.get("key_levels")
    if not key_levels or not isinstance(key_levels, dict):
        return False, "key_levels not available"

    levels_raw = key_levels.get(level_type, [])
    if not levels_raw:
        return False, f"no {level_type} levels available"

    try:
        price = Decimal(str(current_price_raw))
        threshold_pct = Decimal(str(within_pct))
        levels = [Decimal(str(lv)) for lv in levels_raw]
    except (InvalidOperation, TypeError, ValueError):
        return False, "invalid numeric values"

    # Check if price is within threshold_pct of any level
    for level in levels:
        if level == Decimal("0"):
            continue
        distance_pct = abs(price - level) / level * Decimal("100")
        if distance_pct <= threshold_pct:
            detail = f"price {price} within {distance_pct:.2f}% of {level_type} {level} (threshold {threshold_pct}%)"
            return True, detail

    closest = min(levels, key=lambda lv: abs(price - lv)) if levels else Decimal("0")
    if closest != Decimal("0"):
        closest_pct = abs(price - closest) / closest * Decimal("100")
        detail = f"price {price} is {closest_pct:.2f}% from nearest {level_type} {closest} (threshold {threshold_pct}%)"
    else:
        detail = f"no valid {level_type} levels to compare"
    return False, detail


# ────────────────────────────────────────────────────────────────────────────
# Invalidation condition handlers
# ────────────────────────────────────────────────────────────────────────────


def _evaluate_single_invalidation(
    cond_type: str, params: dict, market_context: dict
) -> tuple[bool, str | None]:
    """Dispatch a single invalidation condition to its handler.

    Returns (triggered, detail).
    """
    handler = _INVALIDATION_HANDLERS.get(cond_type)
    if handler is None:
        return False, "no handler"
    return handler(params, market_context)


def _handle_price_breach(params: dict, ctx: dict) -> tuple[bool, str | None]:
    """price_breach: price crossed level in specified direction (Decimal)."""
    level_raw = params.get("level")
    direction = str(params.get("direction", "")).lower()

    if level_raw is None or direction not in ("above", "below"):
        return False, "invalid price_breach params"

    current_price_raw = ctx.get("current_price")
    if current_price_raw is None:
        return False, "current_price not available"

    try:
        price = Decimal(str(current_price_raw))
        level = Decimal(str(level_raw))
    except (InvalidOperation, TypeError, ValueError):
        return False, "invalid numeric values"

    if direction == "above":
        triggered = price > level
        detail = f"price {price} {'>' if triggered else '<='} level {level} (direction=above)"
    else:  # below
        triggered = price < level
        detail = f"price {price} {'<' if triggered else '>='} level {level} (direction=below)"

    return triggered, detail


def _handle_regime_flip(params: dict, ctx: dict) -> tuple[bool, str | None]:
    """regime_flip: current regime in blocked set."""
    blocked_regimes = params.get("blocked_regimes", [])
    if not blocked_regimes or not isinstance(blocked_regimes, list):
        return False, "no blocked_regimes specified"

    current_regime = str(ctx.get("market_regime", "")).lower()
    blocked_lower = [str(r).lower() for r in blocked_regimes]

    triggered = current_regime in blocked_lower
    detail = f"regime={current_regime} {'in' if triggered else 'not in'} blocked={blocked_lower}"
    return triggered, detail


def _handle_catalyst_expired(params: dict, ctx: dict) -> tuple[bool, str | None]:
    """catalyst_expired: catalyst age exceeds max_age_minutes."""
    max_age_minutes = params.get("max_age_minutes")
    if max_age_minutes is None:
        return False, "no max_age_minutes specified"

    catalyst_ts = ctx.get("catalyst_timestamp")
    if not catalyst_ts:
        # No catalyst timestamp — cannot be expired, not triggered
        return False, "no catalyst_timestamp available"

    try:
        if isinstance(catalyst_ts, datetime):
            cat_dt = catalyst_ts if catalyst_ts.tzinfo else catalyst_ts.replace(tzinfo=timezone.utc)
        else:
            ts_str = str(catalyst_ts)
            ts_str = ts_str.replace("Z", "+00:00")
            cat_dt = datetime.fromisoformat(ts_str)
            if cat_dt.tzinfo is None:
                cat_dt = cat_dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return False, "unparseable catalyst_timestamp"

    now = datetime.now(timezone.utc)
    age_minutes = (now - cat_dt).total_seconds() / 60.0

    triggered = age_minutes > float(max_age_minutes)
    detail = f"catalyst_age={age_minutes:.1f}min {'>' if triggered else '<='} max={max_age_minutes}min"
    return triggered, detail


def _handle_exposure_conflict(params: dict, ctx: dict) -> tuple[bool, str | None]:
    """exposure_conflict: symbol already has an open position."""
    symbol = ctx.get("symbol", "")
    held_symbols = ctx.get("held_symbols")

    if not held_symbols or not isinstance(held_symbols, (set, list, frozenset)):
        return False, "no held_symbols available"

    # Normalize for comparison
    held_upper = {str(s).upper() for s in held_symbols}
    triggered = str(symbol).upper() in held_upper
    detail = f"symbol={symbol} {'in' if triggered else 'not in'} held_symbols"
    return triggered, detail


# ────────────────────────────────────────────────────────────────────────────
# Handler dispatch tables
# ────────────────────────────────────────────────────────────────────────────

_MATURATION_HANDLERS: dict[str, callable] = {
    "price_zone": _handle_price_zone,
    "regime_aligned": _handle_regime_aligned,
    "catalyst_fresh": _handle_catalyst_fresh,
    "time_window": _handle_time_window,
    "key_level_proximity": _handle_key_level_proximity,
}

_INVALIDATION_HANDLERS: dict[str, callable] = {
    "price_breach": _handle_price_breach,
    "regime_flip": _handle_regime_flip,
    "catalyst_expired": _handle_catalyst_expired,
    "exposure_conflict": _handle_exposure_conflict,
}
