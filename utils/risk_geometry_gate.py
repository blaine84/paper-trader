"""Risk Geometry Gate — validates full trade geometry before order execution.

Deterministic pre-execution gate that validates stop distance, position size,
dollar risk, reward-to-risk ratio, and target feasibility. Follows the same
pure-evaluator pattern as setup_quality_gate.py and pre_trade_quality_gate.py.

See: .kiro/specs/risk-geometry-gate/design.md
"""

import logging
import math
import os
import re
from datetime import datetime, timedelta, timezone

from utils.gate_config import (
    DEFAULT_STOP_DISTANCE_RULE,
    GATE_EVENT_TYPES,
    HIGH_BETA_CLUSTER,
    QUALIFYING_MIN_SIGNAL_STRENGTH,
    QUALIFYING_SETUP_TYPES,
    REDUCED_RR_THRESHOLDS_BY_PROFILE,
    ROLLING_RECOVERY_PROBE_SIZE_MULTIPLIER,
    STOP_DISTANCE_RULES,
    is_moderate_near_miss_pilot_active,
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


def _is_feature_flag_enabled() -> bool:
    """Check if the setup-specific R:R thresholds feature flag is enabled.

    Reads os.environ on each call so that flag changes take effect without
    process restart. Returns True only if the value is case-insensitive "true".
    """
    return os.environ.get("SETUP_SPECIFIC_RR_THRESHOLDS", "").strip().lower() == "true"


def _is_qualifying_setup(
    setup_type: str | None,
    signal_strength: float | None,
    confidence_level: str | None,
) -> bool:
    """Determine if a trade signal qualifies for reduced R:R thresholds.

    Returns True only when ALL criteria are met:
    1. setup_type (lowercased) is in QUALIFYING_SETUP_TYPES
    2. signal_strength is a numeric value in [0, 10] and >= QUALIFYING_MIN_SIGNAL_STRENGTH
    3. confidence_level (lowercased) equals "high"

    Handles None values, non-numeric signal_strength, and out-of-range values
    gracefully by returning False.
    """
    # Check setup_type
    if setup_type is None or not isinstance(setup_type, str):
        return False
    if setup_type.lower() not in QUALIFYING_SETUP_TYPES:
        return False

    # Check signal_strength — must be numeric, in [0, 10], and >= minimum
    if signal_strength is None:
        return False
    try:
        strength = float(signal_strength)
    except (TypeError, ValueError):
        return False
    if math.isnan(strength) or math.isinf(strength):
        return False
    if strength < 0 or strength > 10:
        return False
    if strength < QUALIFYING_MIN_SIGNAL_STRENGTH:
        return False

    # Check confidence_level
    if confidence_level is None or not isinstance(confidence_level, str):
        return False
    if confidence_level.lower() != "high":
        return False

    return True



def _resolve_rule_from_policy(symbol: str, setup_type: str | None, policy) -> tuple[dict, str, str]:
    """Resolve the applicable stop-distance rule from a GatePolicyConfig.

    Same logic as _resolve_rule but reads from policy instead of module constants.
    """
    upper_symbol = symbol.upper()

    # Priority 1: HIGH_BETA_CLUSTER from policy
    if upper_symbol in policy.high_beta_cluster:
        rule_name = "high_beta_mega_cap_intraday"
        rule = policy.stop_distance_rules.get(rule_name, policy.default_stop_distance_rule)
        return rule, rule_name, "HIGH_BETA_CLUSTER"

    # Priority 2: broad_etf classification
    symbol_class = classify_symbol(symbol)
    if symbol_class == "broad_etf":
        rule_name = "etf_intraday"
        rule = policy.stop_distance_rules.get(rule_name, policy.default_stop_distance_rule)
        return rule, rule_name, "broad_etf"

    # Priority 3: {symbol_class}_{setup_type} override
    if setup_type:
        override_key = f"{symbol_class}_{setup_type}"
        if override_key in policy.stop_distance_rules:
            return policy.stop_distance_rules[override_key], override_key, override_key

    # Priority 4: Default
    return policy.default_stop_distance_rule, "default", "default"


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


def _build_setup_specific_rr_audit_fields(
    *,
    setup_specific_rr_applied: bool,
    setup_type: str | None,
    signal_strength: float | None,
    confidence_level: str | None,
    reduced_threshold: float | None,
    default_threshold: float,
) -> dict:
    """Build the audit payload fields for setup-specific R:R decisions.

    Only called when the feature flag is enabled. Returns a dict of fields
    to merge into the gate event payload.
    """
    missing_fields = []
    if setup_type is None:
        missing_fields.append("setup_type")
    if signal_strength is None:
        missing_fields.append("signal_strength")
    if confidence_level is None:
        missing_fields.append("confidence_level")

    return {
        "setup_specific_rr_applied": setup_specific_rr_applied,
        "setup_specific_rr_qualifying_criteria": {
            "setup_type": setup_type,
            "signal_strength": signal_strength,
            "confidence_level": confidence_level,
            "missing_fields": missing_fields,
        },
        "setup_specific_rr_reduced_threshold": reduced_threshold if setup_specific_rr_applied else None,
        "setup_specific_rr_default_threshold": default_threshold,
    }


def _compute_default_rr_threshold(rule: dict, profile: str | None) -> float:
    """Compute the default R:R threshold that would apply without setup-specific override."""
    by_profile = rule.get("min_reward_to_risk_by_profile") or {}
    profile_key = profile.lower() if isinstance(profile, str) else None
    if profile_key and profile_key in by_profile:
        return float(by_profile[profile_key])
    return float(rule.get("min_reward_to_risk", 2.0))


def _min_reward_to_risk(
    rule: dict,
    profile: str | None,
    *,
    setup_type: str | None = None,
    signal_strength: float | None = None,
    confidence_level: str | None = None,
) -> tuple[float, bool]:
    """Resolve risk-geometry R:R floor, allowing profile-specific overrides.

    When the setup-specific R:R feature flag is enabled and the trade qualifies
    (correct setup_type + high signal_strength + high confidence), returns the
    reduced threshold from REDUCED_RR_THRESHOLDS_BY_PROFILE. Otherwise returns
    the rule's legacy min_reward_to_risk value.

    Returns:
        (threshold, setup_specific_rr_applied) — the R:R floor and whether
        the reduced threshold was used.
    """
    # Resolve the default threshold first (always needed as fallback)
    by_profile = rule.get("min_reward_to_risk_by_profile") or {}
    profile_key = profile.lower() if isinstance(profile, str) else None
    if profile_key and profile_key in by_profile:
        default_threshold = float(by_profile[profile_key])
    else:
        default_threshold = float(rule.get("min_reward_to_risk", 2.0))

    # Attempt qualifying override when feature flag is enabled
    try:
        if _is_feature_flag_enabled() and _is_qualifying_setup(
            setup_type, signal_strength, confidence_level
        ):
            if profile_key and profile_key in REDUCED_RR_THRESHOLDS_BY_PROFILE:
                return (REDUCED_RR_THRESHOLDS_BY_PROFILE[profile_key], True)
            # Profile not in reduced thresholds dict — fall back to default
            return (default_threshold, False)
    except Exception:
        # Fail-open: if anything unexpected happens in qualifying check,
        # fall through to default threshold without blocking the trade.
        log.warning(
            "Unexpected error in setup-specific R:R qualifying check; "
            "falling back to default threshold",
            exc_info=True,
        )

    return (default_threshold, False)


def _min_reward_to_risk_di(
    rule: dict,
    profile: str | None,
    *,
    setup_type: str | None = None,
    signal_strength: float | None = None,
    confidence_level: str | None = None,
    policy=None,
    feature_flag_enabled: bool = False,
) -> tuple[float, bool]:
    """Policy-aware version of _min_reward_to_risk.

    When policy is provided, reads reduced R:R thresholds and qualifying
    criteria from policy instead of module constants. When policy is None,
    delegates to the original _min_reward_to_risk.
    """
    if policy is None:
        return _min_reward_to_risk(
            rule, profile,
            setup_type=setup_type,
            signal_strength=signal_strength,
            confidence_level=confidence_level,
        )

    # Resolve default threshold from rule (same logic as original)
    by_profile = rule.get("min_reward_to_risk_by_profile") or {}
    profile_key = profile.lower() if isinstance(profile, str) else None
    if profile_key and profile_key in by_profile:
        default_threshold = float(by_profile[profile_key])
    else:
        default_threshold = float(rule.get("min_reward_to_risk", 2.0))

    # Attempt qualifying override when feature flag is enabled
    try:
        if feature_flag_enabled and _is_qualifying_setup_di(
            setup_type, signal_strength, confidence_level, policy
        ):
            reduced = policy.reduced_rr_thresholds_by_profile
            if profile_key and profile_key in reduced:
                return (reduced[profile_key], True)
            return (default_threshold, False)
    except Exception:
        log.warning(
            "Unexpected error in policy-aware setup-specific R:R qualifying check; "
            "falling back to default threshold",
            exc_info=True,
        )

    return (default_threshold, False)


def _is_qualifying_setup_di(
    setup_type: str | None,
    signal_strength: float | None,
    confidence_level: str | None,
    policy,
) -> bool:
    """Policy-aware version of _is_qualifying_setup.

    Reads qualifying_min_signal_strength and qualifying_setup_types from policy.
    """
    if setup_type is None or not isinstance(setup_type, str):
        return False
    if setup_type.lower() not in policy.qualifying_setup_types:
        return False

    if signal_strength is None:
        return False
    try:
        strength = float(signal_strength)
    except (TypeError, ValueError):
        return False
    if math.isnan(strength) or math.isinf(strength):
        return False
    if strength < 0 or strength > 10:
        return False
    if strength < policy.qualifying_min_signal_strength:
        return False

    if confidence_level is None or not isinstance(confidence_level, str):
        return False
    if confidence_level.lower() != "high":
        return False

    return True



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


def _compute_tactical_min_stop(
    entry_price: float,
    atr_5min: float,
    min_pct: float,
    atr_multiplier: float,
) -> float:
    """Compute tactical minimum stop distance.

    Returns max(entry_price * min_pct, atr_5min * atr_multiplier)
    """
    pct_floor = entry_price * min_pct
    atr_floor = atr_5min * atr_multiplier
    return max(pct_floor, atr_floor)


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


def _has_tactical_context(
    indicators: list[str],
    metadata: str | None,
    rationale: str | None,
) -> bool:
    """Case-insensitive word-boundary match for tactical context indicators.

    Uses \\b word boundaries to prevent false positives (e.g., "support" must not
    match "unsupported"). Returns True if any indicator is found as a whole word
    in metadata or rationale.
    """
    combined = ""
    if metadata:
        combined += metadata.lower()
    if rationale:
        combined += " " + rationale.lower()
    if not combined.strip():
        return False
    return any(
        re.search(r"\b" + re.escape(ind.lower()) + r"\b", combined)
        for ind in indicators
    )


def _evaluate_tactical_stop_exception(
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
    trade_timestamp: datetime | None,
    max_dollar_risk: float,
    profile: str | None,
    rule: dict,
    rule_name: str,
    trade_metadata: str | None,
    trade_rationale: str | None,
    atr_source: str | None,
    rule_source: str | None,
    quantity_policy: str,
) -> dict | None:
    """Evaluate tactical stop exception for high-beta aggressive trades.

    Returns:
        - Result dict with decision="passed_unchanged" + tactical metadata if
          exception applies and passes all validation.
        - None if exception does not apply or tactical validation fails
          (caller should continue to global path).

    Error handling:
        Wraps entire body in try/except. Any unexpected exception is logged as
        a warning and returns None (fail-closed fallback to global path).
    """
    try:
        # --- Normalize inputs ---
        profile_key = profile.lower() if isinstance(profile, str) else None
        setup_key = setup_type.lower() if isinstance(setup_type, str) else None

        # --- Check profile exists in tactical config ---
        tactical_config = rule.get("tactical_stop_by_profile")
        if not tactical_config or not isinstance(tactical_config, dict):
            return None

        if profile_key is None or profile_key not in tactical_config:
            return None

        profile_cfg = tactical_config[profile_key]

        # --- Check enabled flag ---
        if not profile_cfg.get("enabled", False):
            return None

        # --- Validate all required fields present ---
        required_fields = [
            "enabled",
            "qualifying_setups",
            "conditional_setups",
            "tactical_context_indicators",
            "min_pct",
            "atr_multiplier",
            "min_reward_to_risk",
        ]
        for field in required_fields:
            if field not in profile_cfg:
                return None

        # --- Check setup eligibility ---
        if setup_key is None:
            return None

        qualifying_setups = [s.lower() for s in profile_cfg["qualifying_setups"]]
        conditional_setups = [s.lower() for s in profile_cfg["conditional_setups"]]

        if setup_key in qualifying_setups:
            # Unconditional qualification
            pass
        elif setup_key in conditional_setups:
            # Conditional: requires tactical context indicator match
            indicators = profile_cfg["tactical_context_indicators"]
            if not _has_tactical_context(indicators, trade_metadata, trade_rationale):
                return None
        else:
            # Not in either list
            return None

        # --- Validate ATR freshness ---
        if atr_5min is None or atr_5min <= 0:
            return None
        if atr_timestamp is None:
            return None

        effective_trade_ts = trade_timestamp if trade_timestamp is not None else datetime.now(timezone.utc)
        atr_max_age_minutes = rule.get("atr_max_age_minutes", 15)
        atr_age = effective_trade_ts - atr_timestamp
        if atr_age > timedelta(minutes=atr_max_age_minutes):
            return None

        # --- Compute tactical minimum stop ---
        tactical_min_stop = _compute_tactical_min_stop(
            entry_price=entry_price,
            atr_5min=atr_5min,
            min_pct=profile_cfg["min_pct"],
            atr_multiplier=profile_cfg["atr_multiplier"],
        )

        # --- Compute stop and target distances ---
        stop_distance = abs(entry_price - stop_price)
        target_distance = abs(target_price - entry_price)

        # --- Validate stop_distance >= tactical_min_stop ---
        if stop_distance < tactical_min_stop:
            return None

        # --- Validate original R:R ---
        original_rr = target_distance / stop_distance if stop_distance > 0 else 0.0
        min_rr = profile_cfg["min_reward_to_risk"]
        if original_rr < min_rr:
            return None

        # --- Validate dollar risk ---
        dollar_risk = quantity * stop_distance
        if dollar_risk > max_dollar_risk:
            return None

        # --- All checks passed: return tactical pass result ---
        return {
            "decision": "passed_unchanged",
            "canonical_decision": "allow",
            "reason": "Trade geometry validated \u2014 aggressive tactical stop accepted",
            "reason_code": "PASSED_TACTICAL",
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "quantity": quantity,
            "stop_distance": stop_distance,
            "target_distance": target_distance,
            "min_stop_distance": tactical_min_stop,
            "adjusted_stop_price": None,
            "adjusted_quantity": None,
            "original_dollar_risk": dollar_risk,
            "adjusted_dollar_risk": None,
            "original_rr": original_rr,
            "adjusted_rr": None,
            "atr_value": atr_5min,
            "atr_source": atr_source,
            "atr_timestamp": atr_timestamp,
            "atr_fallback": False,
            "rule_name": rule_name,
            "rule_source": rule_source,
            "quantity_policy": quantity_policy,
            "min_reward_to_risk": min_rr,
            "tactical_stop_applied": True,
            "tactical_min_stop_distance": tactical_min_stop,
        }

    except Exception as exc:
        log.warning(
            "Tactical stop exception evaluation failed (fail-closed): %s "
            "[symbol=%s, profile=%s, setup_type=%s, entry_price=%s]",
            exc,
            symbol,
            profile,
            setup_type,
            entry_price,
        )
        return None


def _evaluate_pilot_rr_override(
    *,
    original_rr: float,
    adjusted_rr: float,
    min_rr: float,
    profile: str,
) -> bool:
    """Determine if a risk geometry rejection qualifies for pilot override.

    Qualifying conditions:
    1. Profile is 'moderate'
    2. Pilot is active
    3. original_rr >= min_rr (pre-adjustment was valid)
    4. adjusted_rr < min_rr (post-adjustment degraded below threshold)

    Returns True if pilot override should be applied.
    """
    if profile != "moderate":
        return False

    if not is_moderate_near_miss_pilot_active():
        return False

    # If original R:R is already below threshold, no override (Req 6.5)
    if original_rr < min_rr:
        return False

    # Override applies when original was valid but adjustment degraded it
    if adjusted_rr < min_rr:
        return True

    return False


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
    trade_metadata: str | None = None,
    trade_rationale: str | None = None,
    signal_strength: float | None = None,
    confidence_level: str | None = None,
    policy=None,
    event_sink=None,
    clock=None,
    id_provider=None,
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
            trade_metadata=trade_metadata,
            trade_rationale=trade_rationale,
            signal_strength=signal_strength,
            confidence_level=confidence_level,
            policy=policy,
            event_sink=event_sink,
            clock=clock,
            id_provider=id_provider,
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
                sink = event_sink if event_sink is not None else log_trade_event
                sink(
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
            "canonical_decision": "allow",
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
    trade_metadata: str | None = None,
    trade_rationale: str | None = None,
    signal_strength: float | None = None,
    confidence_level: str | None = None,
    policy=None,
    event_sink=None,
    clock=None,
    id_provider=None,
) -> dict:
    """Core evaluation logic (may raise on truly unexpected errors)."""

    if trade_timestamp is None:
        trade_timestamp = clock() if clock is not None else datetime.now(timezone.utc)

    # Step 1: Normalize direction
    norm_direction = _normalize_direction(direction)

    # Step 2: Resolve rule (policy-aware)
    if policy is not None:
        rule, rule_name, rule_source = _resolve_rule_from_policy(symbol, setup_type, policy)
    else:
        rule, rule_name, rule_source = _resolve_rule(symbol, setup_type)

    # Resolve feature flag (policy-aware)
    if policy is not None:
        _ff_setup_specific_rr = policy.feature_flags.get("SETUP_SPECIFIC_RR_THRESHOLDS", False)
    else:
        _ff_setup_specific_rr = _is_feature_flag_enabled()

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
            event_sink=event_sink,
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
            event_sink=event_sink,
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
            event_sink=event_sink,
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
            event_sink=event_sink,
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
            event_sink=event_sink,
        )

    # Step 4.5: Tactical stop exception check (high-beta aggressive only)
    if rule_name == "high_beta_mega_cap_intraday":
        tactical_result = _evaluate_tactical_stop_exception(
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=quantity,
            direction=norm_direction,
            symbol=symbol,
            setup_type=setup_type,
            atr_5min=atr_5min,
            atr_timestamp=atr_timestamp,
            trade_timestamp=trade_timestamp,
            max_dollar_risk=max_dollar_risk,
            profile=profile,
            rule=rule,
            rule_name=rule_name,
            trade_metadata=trade_metadata,
            trade_rationale=trade_rationale,
            atr_source=atr_source,
            rule_source=rule_source,
            quantity_policy=quantity_policy,
        )
        if tactical_result is not None:
            _log_gate_event(tactical_result, symbol=symbol, db=db, profile=profile, agent=agent, event_sink=event_sink)
            return tactical_result

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
            event_sink=event_sink,
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

        # Validate R:R (policy-aware)
        min_rr, _setup_specific_rr_applied = _min_reward_to_risk_di(
            rule, profile,
            setup_type=setup_type,
            signal_strength=signal_strength,
            confidence_level=confidence_level,
            policy=policy,
            feature_flag_enabled=_ff_setup_specific_rr,
        )
        if original_rr < min_rr:
            # Build setup-specific R:R audit fields for rejection payload
            _rr_extra_payload = None
            if _ff_setup_specific_rr:
                default_threshold = _compute_default_rr_threshold(rule, profile)
                _rr_extra_payload = _build_setup_specific_rr_audit_fields(
                    setup_specific_rr_applied=_setup_specific_rr_applied,
                    setup_type=setup_type,
                    signal_strength=signal_strength,
                    confidence_level=confidence_level,
                    reduced_threshold=min_rr if _setup_specific_rr_applied else None,
                    default_threshold=default_threshold,
                )
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
                extra_payload=_rr_extra_payload,
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
            "canonical_decision": "allow",
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
            "min_reward_to_risk": min_rr,
        }

        # Add setup-specific R:R audit fields when feature flag is enabled
        if _ff_setup_specific_rr:
            default_threshold = _compute_default_rr_threshold(rule, profile)
            result.update(_build_setup_specific_rr_audit_fields(
                setup_specific_rr_applied=_setup_specific_rr_applied,
                setup_type=setup_type,
                signal_strength=signal_strength,
                confidence_level=confidence_level,
                reduced_threshold=min_rr if _setup_specific_rr_applied else None,
                default_threshold=default_threshold,
            ))

        _log_gate_event(result, symbol=symbol, db=db, profile=profile, agent=agent, event_sink=event_sink)
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
        min_rr, _setup_specific_rr_applied = _min_reward_to_risk_di(
            rule, profile,
            setup_type=setup_type,
            signal_strength=signal_strength,
            confidence_level=confidence_level,
            policy=policy,
            feature_flag_enabled=_ff_setup_specific_rr,
        )
        if adjusted_rr < min_rr:
            # --- Pilot R:R override check ---
            # If the original R:R was valid but adjustment degraded it,
            # moderate profiles with active pilot get a reduced-size pass.
            if _evaluate_pilot_rr_override(
                original_rr=original_rr,
                adjusted_rr=adjusted_rr,
                min_rr=min_rr,
                profile=profile or "",
            ):
                size_multiplier = ROLLING_RECOVERY_PROBE_SIZE_MULTIPLIER
                # Log separate pilot override audit event
                pilot_payload = {
                    "gate_name": "risk_geometry_gate",
                    "canonical_decision": "reduce_size",
                    "reason_type": "pilot_rr_override",
                    "original_rr": original_rr,
                    "adjusted_rr": adjusted_rr,
                    "min_rr": min_rr,
                    "size_multiplier": size_multiplier,
                    "pilot_override": True,
                }
                if db is not None:
                    try:
                        _pilot_sink = event_sink if event_sink is not None else log_trade_event
                        _pilot_sink(
                            db,
                            GATE_EVENT_TYPES["pilot_override"],
                            agent=agent,
                            symbol=symbol,
                            profile=profile,
                            price=entry_price,
                            message=f"Pilot R:R override: adjusted_rr={adjusted_rr:.2f} < min_rr={min_rr:.2f}, original_rr={original_rr:.2f}",
                            payload=pilot_payload,
                        )
                    except Exception as exc:
                        log.warning("Failed to log pilot_rr_override event: %s", exc)

                # Return adjusted_allowed with reduced size multiplier
                result = {
                    "decision": "adjusted_allowed",
                    "canonical_decision": "reduce_size",
                    "reason": "Trade reconstructed with pilot R:R override",
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
                    "min_reward_to_risk": min_rr,
                    "size_multiplier": size_multiplier,
                    "pilot_override": True,
                }
                _log_gate_event(result, symbol=symbol, db=db, profile=profile, agent=agent, event_sink=event_sink)
                return result

            # --- Standard rejection (no pilot override) ---
            # Build setup-specific R:R audit fields for rejection payload
            _rr_extra_payload = None
            if _ff_setup_specific_rr:
                default_threshold = _compute_default_rr_threshold(rule, profile)
                _rr_extra_payload = _build_setup_specific_rr_audit_fields(
                    setup_specific_rr_applied=_setup_specific_rr_applied,
                    setup_type=setup_type,
                    signal_strength=signal_strength,
                    confidence_level=confidence_level,
                    reduced_threshold=min_rr if _setup_specific_rr_applied else None,
                    default_threshold=default_threshold,
                )
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
                extra_payload=_rr_extra_payload,
            )

        # Adjusted trade allowed
        result = {
            "decision": "adjusted_allowed",
            "canonical_decision": "allow",
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
            "min_reward_to_risk": min_rr,
        }

        # Add setup-specific R:R audit fields when feature flag is enabled
        if _ff_setup_specific_rr:
            default_threshold = _compute_default_rr_threshold(rule, profile)
            result.update(_build_setup_specific_rr_audit_fields(
                setup_specific_rr_applied=_setup_specific_rr_applied,
                setup_type=setup_type,
                signal_strength=signal_strength,
                confidence_level=confidence_level,
                reduced_threshold=min_rr if _setup_specific_rr_applied else None,
                default_threshold=default_threshold,
            ))

        _log_gate_event(result, symbol=symbol, db=db, profile=profile, agent=agent, event_sink=event_sink)
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
    extra_payload: dict | None = None,
    event_sink=None,
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
        "canonical_decision": "reject",
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

    if extra_payload:
        result.update(extra_payload)

    _log_gate_event(result, symbol=symbol, db=db, profile=profile, agent=agent, event_sink=event_sink)
    return result


def _log_gate_event(
    result: dict,
    *,
    symbol: str,
    db,
    profile: str | None,
    agent: str,
    event_sink=None,
) -> None:
    """Log exactly one TradeEvent for the gate evaluation.

    When *event_sink* is provided, uses it instead of ``log_trade_event``
    (dependency injection for replay).
    """
    if db is None:
        return

    payload = {
        "gate_name": "risk_geometry_gate",
        **result,
    }

    try:
        sink = event_sink if event_sink is not None else log_trade_event
        sink(
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
