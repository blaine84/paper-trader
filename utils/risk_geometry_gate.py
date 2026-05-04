"""Risk Geometry Gate — validates full trade geometry before order execution.

Deterministic pre-execution gate that validates stop distance, position size,
dollar risk, reward-to-risk ratio, and target feasibility. Follows the same
pure-evaluator pattern as setup_quality_gate.py and pre_trade_quality_gate.py.

See: .kiro/specs/risk-geometry-gate/design.md
"""

import logging
import math
from datetime import datetime, timedelta, timezone

from utils.gate_config import (
    DEFAULT_STOP_DISTANCE_RULE,
    HIGH_BETA_CLUSTER,
    STOP_DISTANCE_RULES,
)
from utils.symbol_class import classify_symbol
from utils.trade_events import log_trade_event

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _normalize_direction(direction: str) -> str:
    """Normalize BUY/LONG/SHORT to canonical LONG/SHORT."""
    upper = direction.upper()
    if upper in ("BUY", "LONG"):
        return "LONG"
    return "SHORT"


def _resolve_rule(symbol: str, setup_type: str | None) -> tuple[dict, str, str]:
    """Resolve the applicable stop-distance rule.

    Priority:
    1. HIGH_BETA_CLUSTER membership → high_beta_mega_cap_intraday
    2. classify_symbol() == "broad_etf" → etf_intraday
    3. {symbol_class}_{setup_type} override in STOP_DISTANCE_RULES
    4. DEFAULT_STOP_DISTANCE_RULE

    Returns:
        (rule_dict, rule_name, rule_source)
    """
    upper_symbol = symbol.upper()

    # Priority 1: HIGH_BETA_CLUSTER
    if upper_symbol in HIGH_BETA_CLUSTER:
        rule_name = "high_beta_mega_cap_intraday"
        rule = STOP_DISTANCE_RULES.get(rule_name, DEFAULT_STOP_DISTANCE_RULE)
        return rule, rule_name, "HIGH_BETA_CLUSTER"

    # Priority 2: broad_etf classification
    symbol_class = classify_symbol(symbol)
    if symbol_class == "broad_etf":
        rule_name = "etf_intraday"
        rule = STOP_DISTANCE_RULES.get(rule_name, DEFAULT_STOP_DISTANCE_RULE)
        return rule, rule_name, "broad_etf"

    # Priority 3: {symbol_class}_{setup_type} override
    if setup_type:
        override_key = f"{symbol_class}_{setup_type}"
        if override_key in STOP_DISTANCE_RULES:
            return STOP_DISTANCE_RULES[override_key], override_key, override_key

    # Priority 4: Default
    return DEFAULT_STOP_DISTANCE_RULE, "default", "default"


def _compute_min_stop_distance(
    entry_price: float,
    rule: dict,
    atr_5min: float | None,
    atr_timestamp: datetime | None,
    trade_timestamp: datetime | None,
) -> tuple[float | None, bool]:
    """Compute minimum stop distance from pct floor and ATR floor.

    Returns:
        (min_distance, atr_fallback_used)
        - (float, False) when both floors computed normally
        - (float, True) when ATR unavailable/stale and pct-only fallback used
        - (None, True) when ATR unavailable/stale and fallback disabled (rejection needed)
    """
    if trade_timestamp is None:
        trade_timestamp = datetime.now(timezone.utc)

    pct_floor = entry_price * rule["min_pct"]
    atr_max_age_minutes = rule.get("atr_max_age_minutes", 15)
    allow_pct_only_fallback = rule.get("allow_pct_only_fallback", True)

    # Determine if ATR is valid and fresh
    atr_valid = (
        atr_5min is not None
        and atr_5min > 0
    )

    atr_fresh = True
    if atr_valid and atr_timestamp is not None:
        age = trade_timestamp - atr_timestamp
        if age > timedelta(minutes=atr_max_age_minutes):
            atr_fresh = False
    elif atr_valid and atr_timestamp is None:
        # ATR value present but no timestamp — treat as stale
        atr_fresh = False

    if atr_valid and atr_fresh:
        atr_floor = atr_5min * rule["atr_multiplier"]
        min_distance = max(pct_floor, atr_floor)
        return min_distance, False
    else:
        # ATR unavailable or stale
        if allow_pct_only_fallback:
            return pct_floor, True
        else:
            return None, True


def _apply_quantity_policy(
    raw_quantity: float,
    policy: str,
    precision: int,
) -> int | float:
    """Apply quantity truncation policy. Never rounds up.

    Args:
        raw_quantity: The unrounded quantity.
        policy: "whole_share" or "fractional".
        precision: Decimal precision for fractional policy.

    Returns:
        Truncated quantity (int for whole_share, float for fractional).
    """
    if policy == "whole_share":
        return int(math.floor(raw_quantity))
    else:
        # fractional: truncate to given decimal precision
        factor = 10 ** precision
        return math.floor(raw_quantity * factor) / factor


def _reconstruct_trade(
    entry_price: float,
    target_price: float,
    min_stop_distance: float,
    direction: str,
    max_dollar_risk: float,
    quantity_policy: str,
    fractional_precision: int,
) -> dict:
    """Reconstruct trade with valid geometry.

    Returns dict with adjusted parameters:
        adjusted_stop_price, adjusted_quantity, adjusted_dollar_risk,
        adjusted_rr, target_distance
    """
    # Compute adjusted stop
    if direction == "LONG":
        adjusted_stop_price = entry_price - min_stop_distance
        target_distance = target_price - entry_price
    else:
        adjusted_stop_price = entry_price + min_stop_distance
        target_distance = entry_price - target_price

    # Compute adjusted quantity: floor(max_dollar_risk / min_stop_distance)
    raw_quantity = max_dollar_risk / min_stop_distance
    adjusted_quantity = _apply_quantity_policy(
        raw_quantity, quantity_policy, fractional_precision
    )

    # Compute adjusted dollar risk
    adjusted_dollar_risk = adjusted_quantity * min_stop_distance

    # Compute adjusted R:R
    adjusted_rr = target_distance / min_stop_distance if min_stop_distance > 0 else 0.0

    return {
        "adjusted_stop_price": adjusted_stop_price,
        "adjusted_quantity": adjusted_quantity,
        "adjusted_dollar_risk": adjusted_dollar_risk,
        "adjusted_rr": adjusted_rr,
        "target_distance": target_distance,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_risk_geometry(
    *,
    entry_price: float,
    stop_price: float,
    target_price: float,
    quantity: int | float,
    direction: str,
    symbol: str,
    setup_type: str | None = None,
    atr_5min: float | None = None,
    atr_timestamp: datetime | None = None,
    atr_source: str | None = None,
    trade_timestamp: datetime | None = None,
    max_dollar_risk: float,
    quantity_policy: str = "whole_share",
    fractional_precision: int = 2,
    db=None,
    profile: str | None = None,
    agent: str = "portfolio_manager",
) -> dict:
    """Evaluate trade geometry and return gate decision.

    Full evaluation flow:
    1. Normalize direction
    2. Validate stop direction (BEFORE min stop distance computation)
    3. Validate target geometry (BEFORE stop distance comparison)
    4. Compute min stop distance (may signal ATR rejection)
    5. Branch on stop adequacy
    6. Validate unchanged or reconstructed trade
    7. Log exactly one TradeEvent
    8. Return result dict

    On unexpected exception: catch, log error, return fail-open result.
    """
    try:
        return _evaluate_risk_geometry_inner(
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=quantity,
            direction=direction,
            symbol=symbol,
            setup_type=setup_type,
            atr_5min=atr_5min,
            atr_timestamp=atr_timestamp,
            atr_source=atr_source,
            trade_timestamp=trade_timestamp,
            max_dollar_risk=max_dollar_risk,
            quantity_policy=quantity_policy,
            fractional_precision=fractional_precision,
            db=db,
            profile=profile,
            agent=agent,
        )
    except Exception as exc:
        log.error("RiskGeometryGate unexpected error (fail-open): %s", exc)
        # Fail-open: log error event and allow trade to proceed
        fail_open_payload = {
            "gate_name": "risk_geometry_gate",
            "risk_geometry_gate_failed_open": True,
            "error": str(exc),
            "decision": "passed_unchanged",
            "symbol": symbol,
        }
        try:
            if db is not None:
                log_trade_event(
                    db,
                    "risk_geometry_gate_evaluated",
                    agent=agent,
                    symbol=symbol,
                    profile=profile,
                    message=f"Gate failed open due to unexpected error: {exc}",
                    payload=fail_open_payload,
                )
        except Exception:
            log.warning("Failed to log fail-open event for risk_geometry_gate")

        return {
            "decision": "passed_unchanged",
            "reason": f"Gate failed open due to unexpected error: {exc}",
            "reason_code": "GATE_ERROR_FAIL_OPEN",
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "quantity": quantity,
            "stop_distance": 0.0,
            "min_stop_distance": 0.0,
            "adjusted_stop_price": None,
            "adjusted_quantity": None,
            "original_dollar_risk": 0.0,
            "adjusted_dollar_risk": None,
            "original_rr": 0.0,
            "adjusted_rr": None,
            "target_distance": 0.0,
            "atr_value": atr_5min,
            "atr_source": atr_source,
            "atr_timestamp": atr_timestamp,
            "atr_fallback": False,
            "rule_name": "unknown",
            "rule_source": "unknown",
            "quantity_policy": quantity_policy,
            "risk_geometry_gate_failed_open": True,
        }


def _evaluate_risk_geometry_inner(
    *,
    entry_price: float,
    stop_price: float,
    target_price: float,
    quantity: int | float,
    direction: str,
    symbol: str,
    setup_type: str | None,
    atr_5min: float | None,
    atr_timestamp: datetime | None,
    atr_source: str | None,
    trade_timestamp: datetime | None,
    max_dollar_risk: float,
    quantity_policy: str,
    fractional_precision: int,
    db,
    profile: str | None,
    agent: str,
) -> dict:
    """Core evaluation logic (may raise on truly unexpected errors)."""

    if trade_timestamp is None:
        trade_timestamp = datetime.now(timezone.utc)

    # Step 1: Normalize direction
    norm_direction = _normalize_direction(direction)

    # Step 2: Resolve rule
    rule, rule_name, rule_source = _resolve_rule(symbol, setup_type)

    # Step 3: Validate stop direction (BEFORE min stop distance computation)
    if norm_direction == "LONG" and stop_price >= entry_price:
        return _build_rejection(
            reason="Stop price must be below entry price for LONG trades",
            reason_code="INVALID_STOP_DIRECTION",
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=quantity,
            direction=norm_direction,
            symbol=symbol,
            atr_5min=atr_5min,
            atr_source=atr_source,
            atr_timestamp=atr_timestamp,
            rule_name=rule_name,
            rule_source=rule_source,
            quantity_policy=quantity_policy,
            db=db,
            profile=profile,
            agent=agent,
        )
    if norm_direction == "SHORT" and stop_price <= entry_price:
        return _build_rejection(
            reason="Stop price must be above entry price for SHORT trades",
            reason_code="INVALID_STOP_DIRECTION",
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=quantity,
            direction=norm_direction,
            symbol=symbol,
            atr_5min=atr_5min,
            atr_source=atr_source,
            atr_timestamp=atr_timestamp,
            rule_name=rule_name,
            rule_source=rule_source,
            quantity_policy=quantity_policy,
            db=db,
            profile=profile,
            agent=agent,
        )

    # Step 4: Validate target geometry (BEFORE stop distance comparison)
    if target_price is None or target_price == 0:
        return _build_rejection(
            reason="Target price is missing or zero",
            reason_code="INVALID_TARGET_GEOMETRY",
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=quantity,
            direction=norm_direction,
            symbol=symbol,
            atr_5min=atr_5min,
            atr_source=atr_source,
            atr_timestamp=atr_timestamp,
            rule_name=rule_name,
            rule_source=rule_source,
            quantity_policy=quantity_policy,
            db=db,
            profile=profile,
            agent=agent,
        )

    if norm_direction == "LONG" and target_price <= entry_price:
        return _build_rejection(
            reason="Target must be above entry price for LONG trades",
            reason_code="INVALID_TARGET_GEOMETRY",
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=quantity,
            direction=norm_direction,
            symbol=symbol,
            atr_5min=atr_5min,
            atr_source=atr_source,
            atr_timestamp=atr_timestamp,
            rule_name=rule_name,
            rule_source=rule_source,
            quantity_policy=quantity_policy,
            db=db,
            profile=profile,
            agent=agent,
        )
    if norm_direction == "SHORT" and target_price >= entry_price:
        return _build_rejection(
            reason="Target must be below entry price for SHORT trades",
            reason_code="INVALID_TARGET_GEOMETRY",
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=quantity,
            direction=norm_direction,
            symbol=symbol,
            atr_5min=atr_5min,
            atr_source=atr_source,
            atr_timestamp=atr_timestamp,
            rule_name=rule_name,
            rule_source=rule_source,
            quantity_policy=quantity_policy,
            db=db,
            profile=profile,
            agent=agent,
        )

    # Step 5: Compute min stop distance
    min_stop_result = _compute_min_stop_distance(
        entry_price=entry_price,
        rule=rule,
        atr_5min=atr_5min,
        atr_timestamp=atr_timestamp,
        trade_timestamp=trade_timestamp,
    )
    min_stop_distance, atr_fallback_used = min_stop_result

    # ATR rejection case
    if min_stop_distance is None:
        return _build_rejection(
            reason="ATR data unavailable or stale and pct-only fallback disabled",
            reason_code="ATR_UNAVAILABLE_FOR_STOP_VALIDATION",
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=quantity,
            direction=norm_direction,
            symbol=symbol,
            atr_5min=atr_5min,
            atr_source=atr_source,
            atr_timestamp=atr_timestamp,
            rule_name=rule_name,
            rule_source=rule_source,
            quantity_policy=quantity_policy,
            db=db,
            profile=profile,
            agent=agent,
        )

    # Compute proposed stop distance and target distance
    stop_distance = abs(entry_price - stop_price)
    if norm_direction == "LONG":
        target_distance = target_price - entry_price
    else:
        target_distance = entry_price - target_price

    # Step 6: Branch on stop adequacy
    if stop_distance >= min_stop_distance:
        # Unchanged trade — validate R:R, dollar risk, position size
        original_rr = target_distance / stop_distance if stop_distance > 0 else 0.0
        original_dollar_risk = quantity * stop_distance

        # Validate R:R
        min_rr = rule.get("min_reward_to_risk", 2.0)
        if original_rr < min_rr:
            return _build_rejection(
                reason=f"Reward-to-risk ratio {original_rr:.2f} below minimum {min_rr:.2f}",
                reason_code="RISK_REWARD_BELOW_THRESHOLD",
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                quantity=quantity,
                direction=norm_direction,
                symbol=symbol,
                atr_5min=atr_5min,
                atr_source=atr_source,
                atr_timestamp=atr_timestamp,
                rule_name=rule_name,
                rule_source=rule_source,
                quantity_policy=quantity_policy,
                db=db,
                profile=profile,
                agent=agent,
                stop_distance=stop_distance,
                min_stop_distance=min_stop_distance,
                target_distance=target_distance,
                original_rr=original_rr,
                original_dollar_risk=original_dollar_risk,
                atr_fallback=atr_fallback_used,
            )

        # Validate dollar risk
        if original_dollar_risk > max_dollar_risk:
            return _build_rejection(
                reason=f"Dollar risk ${original_dollar_risk:.2f} exceeds max ${max_dollar_risk:.2f}",
                reason_code="MAX_DOLLAR_RISK_EXCEEDED",
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                quantity=quantity,
                direction=norm_direction,
                symbol=symbol,
                atr_5min=atr_5min,
                atr_source=atr_source,
                atr_timestamp=atr_timestamp,
                rule_name=rule_name,
                rule_source=rule_source,
                quantity_policy=quantity_policy,
                db=db,
                profile=profile,
                agent=agent,
                stop_distance=stop_distance,
                min_stop_distance=min_stop_distance,
                target_distance=target_distance,
                original_rr=original_rr,
                original_dollar_risk=original_dollar_risk,
                atr_fallback=atr_fallback_used,
            )

        # Passed unchanged
        result = {
            "decision": "passed_unchanged",
            "reason": "Trade geometry validated — stop distance adequate",
            "reason_code": "PASSED",
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "quantity": quantity,
            "stop_distance": stop_distance,
            "min_stop_distance": min_stop_distance,
            "adjusted_stop_price": None,
            "adjusted_quantity": None,
            "original_dollar_risk": original_dollar_risk,
            "adjusted_dollar_risk": None,
            "original_rr": original_rr,
            "adjusted_rr": None,
            "target_distance": target_distance,
            "atr_value": atr_5min,
            "atr_source": atr_source,
            "atr_timestamp": atr_timestamp,
            "atr_fallback": atr_fallback_used,
            "rule_name": rule_name,
            "rule_source": rule_source,
            "quantity_policy": quantity_policy,
        }
        _log_gate_event(result, symbol=symbol, db=db, profile=profile, agent=agent)
        return result

    else:
        # Stop distance < min — reconstruct trade
        reconstructed = _reconstruct_trade(
            entry_price=entry_price,
            target_price=target_price,
            min_stop_distance=min_stop_distance,
            direction=norm_direction,
            max_dollar_risk=max_dollar_risk,
            quantity_policy=quantity_policy,
            fractional_precision=fractional_precision,
        )

        adjusted_quantity = reconstructed["adjusted_quantity"]
        adjusted_dollar_risk = reconstructed["adjusted_dollar_risk"]
        adjusted_rr = reconstructed["adjusted_rr"]
        adjusted_stop_price = reconstructed["adjusted_stop_price"]
        recon_target_distance = reconstructed["target_distance"]

        original_rr = target_distance / stop_distance if stop_distance > 0 else 0.0
        original_dollar_risk = quantity * stop_distance

        # Determine minimum tradable unit
        if quantity_policy == "whole_share":
            min_tradable_unit = 1
        else:
            min_tradable_unit = 10 ** (-fractional_precision)

        # Validate: position size minimum
        if adjusted_quantity < min_tradable_unit:
            return _build_rejection(
                reason=f"Adjusted quantity {adjusted_quantity} below minimum tradable unit",
                reason_code="POSITION_SIZE_BELOW_MINIMUM",
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                quantity=quantity,
                direction=norm_direction,
                symbol=symbol,
                atr_5min=atr_5min,
                atr_source=atr_source,
                atr_timestamp=atr_timestamp,
                rule_name=rule_name,
                rule_source=rule_source,
                quantity_policy=quantity_policy,
                db=db,
                profile=profile,
                agent=agent,
                stop_distance=stop_distance,
                min_stop_distance=min_stop_distance,
                target_distance=target_distance,
                original_rr=original_rr,
                original_dollar_risk=original_dollar_risk,
                adjusted_stop_price=adjusted_stop_price,
                adjusted_quantity=adjusted_quantity,
                adjusted_dollar_risk=adjusted_dollar_risk,
                adjusted_rr=adjusted_rr,
                atr_fallback=atr_fallback_used,
            )

        # Validate: adjusted dollar risk
        if adjusted_dollar_risk > max_dollar_risk:
            return _build_rejection(
                reason=f"Adjusted dollar risk ${adjusted_dollar_risk:.2f} exceeds max ${max_dollar_risk:.2f}",
                reason_code="STOP_DISTANCE_VIOLATION",
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                quantity=quantity,
                direction=norm_direction,
                symbol=symbol,
                atr_5min=atr_5min,
                atr_source=atr_source,
                atr_timestamp=atr_timestamp,
                rule_name=rule_name,
                rule_source=rule_source,
                quantity_policy=quantity_policy,
                db=db,
                profile=profile,
                agent=agent,
                stop_distance=stop_distance,
                min_stop_distance=min_stop_distance,
                target_distance=target_distance,
                original_rr=original_rr,
                original_dollar_risk=original_dollar_risk,
                adjusted_stop_price=adjusted_stop_price,
                adjusted_quantity=adjusted_quantity,
                adjusted_dollar_risk=adjusted_dollar_risk,
                adjusted_rr=adjusted_rr,
                atr_fallback=atr_fallback_used,
            )

        # Validate: adjusted R:R
        min_rr = rule.get("min_reward_to_risk", 2.0)
        if adjusted_rr < min_rr:
            return _build_rejection(
                reason=f"Adjusted R:R {adjusted_rr:.2f} below minimum {min_rr:.2f} after stop adjustment",
                reason_code="RISK_REWARD_AFTER_STOP_ADJUSTMENT",
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                quantity=quantity,
                direction=norm_direction,
                symbol=symbol,
                atr_5min=atr_5min,
                atr_source=atr_source,
                atr_timestamp=atr_timestamp,
                rule_name=rule_name,
                rule_source=rule_source,
                quantity_policy=quantity_policy,
                db=db,
                profile=profile,
                agent=agent,
                stop_distance=stop_distance,
                min_stop_distance=min_stop_distance,
                target_distance=target_distance,
                original_rr=original_rr,
                original_dollar_risk=original_dollar_risk,
                adjusted_stop_price=adjusted_stop_price,
                adjusted_quantity=adjusted_quantity,
                adjusted_dollar_risk=adjusted_dollar_risk,
                adjusted_rr=adjusted_rr,
                atr_fallback=atr_fallback_used,
            )

        # Adjusted trade allowed
        result = {
            "decision": "adjusted_allowed",
            "reason": "Trade reconstructed with valid geometry",
            "reason_code": "ADJUSTED",
            "entry_price": entry_price,
            "stop_price": adjusted_stop_price,
            "target_price": target_price,
            "quantity": adjusted_quantity,
            "stop_distance": stop_distance,
            "min_stop_distance": min_stop_distance,
            "adjusted_stop_price": adjusted_stop_price,
            "adjusted_quantity": adjusted_quantity,
            "original_dollar_risk": original_dollar_risk,
            "adjusted_dollar_risk": adjusted_dollar_risk,
            "original_rr": original_rr,
            "adjusted_rr": adjusted_rr,
            "target_distance": target_distance,
            "atr_value": atr_5min,
            "atr_source": atr_source,
            "atr_timestamp": atr_timestamp,
            "atr_fallback": atr_fallback_used,
            "rule_name": rule_name,
            "rule_source": rule_source,
            "quantity_policy": quantity_policy,
        }
        _log_gate_event(result, symbol=symbol, db=db, profile=profile, agent=agent)
        return result


# ---------------------------------------------------------------------------
# Result Builders & Logging
# ---------------------------------------------------------------------------


def _build_rejection(
    *,
    reason: str,
    reason_code: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
    quantity: int | float,
    direction: str,
    symbol: str,
    atr_5min: float | None,
    atr_source: str | None,
    atr_timestamp: datetime | None,
    rule_name: str,
    rule_source: str,
    quantity_policy: str,
    db,
    profile: str | None,
    agent: str,
    stop_distance: float = 0.0,
    min_stop_distance: float = 0.0,
    target_distance: float = 0.0,
    original_rr: float = 0.0,
    original_dollar_risk: float = 0.0,
    adjusted_stop_price: float | None = None,
    adjusted_quantity: int | float | None = None,
    adjusted_dollar_risk: float | None = None,
    adjusted_rr: float | None = None,
    atr_fallback: bool = False,
) -> dict:
    """Build a rejection result dict and log the event."""
    # Compute stop_distance if not provided
    if stop_distance == 0.0 and entry_price and stop_price:
        stop_distance = abs(entry_price - stop_price)

    # Compute target_distance if not provided
    if target_distance == 0.0 and target_price and entry_price:
        if direction == "LONG":
            target_distance = target_price - entry_price
        else:
            target_distance = entry_price - target_price

    # Compute original_dollar_risk if not provided
    if original_dollar_risk == 0.0 and quantity and stop_distance:
        original_dollar_risk = quantity * stop_distance

    # Compute original_rr if not provided
    if original_rr == 0.0 and stop_distance > 0 and target_distance > 0:
        original_rr = target_distance / stop_distance

    result = {
        "decision": "rejected",
        "reason": reason,
        "reason_code": reason_code,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "quantity": quantity,
        "stop_distance": stop_distance,
        "min_stop_distance": min_stop_distance,
        "adjusted_stop_price": adjusted_stop_price,
        "adjusted_quantity": adjusted_quantity,
        "original_dollar_risk": original_dollar_risk,
        "adjusted_dollar_risk": adjusted_dollar_risk,
        "original_rr": original_rr,
        "adjusted_rr": adjusted_rr,
        "target_distance": target_distance,
        "atr_value": atr_5min,
        "atr_source": atr_source,
        "atr_timestamp": atr_timestamp,
        "atr_fallback": atr_fallback,
        "rule_name": rule_name,
        "rule_source": rule_source,
        "quantity_policy": quantity_policy,
    }

    _log_gate_event(result, symbol=symbol, db=db, profile=profile, agent=agent)
    return result


def _log_gate_event(
    result: dict,
    *,
    symbol: str,
    db,
    profile: str | None,
    agent: str,
) -> None:
    """Log exactly one TradeEvent for the gate evaluation."""
    if db is None:
        return

    payload = {
        "gate_name": "risk_geometry_gate",
        **result,
    }

    try:
        log_trade_event(
            db,
            "risk_geometry_gate_evaluated",
            agent=agent,
            symbol=symbol,
            profile=profile,
            price=result.get("entry_price"),
            message=result.get("reason", ""),
            payload=payload,
        )
    except Exception as exc:
        log.warning("Failed to log risk_geometry_gate event: %s", exc)
