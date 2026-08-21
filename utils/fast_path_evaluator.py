"""Fast-path outcome evaluator — deterministic trigger evaluation logic.

Evaluates registered triggers against fresh market data to produce explicit
outcome types without LLM involvement.  The evaluator follows a strict
priority order: data quality → target status → geometry → cooldown →
exposure → trigger condition → confirmation → limit eligibility →
entry deviation → gate pipeline → execution.

Fail mode: fail-closed (evaluation error → stand_down).

See: .kiro/specs/fast-path-deterministic-execution/design.md
Requirements: 3.1, 3.10, 9.1, 2.1–2.11, 5.1–5.7, 10.1–10.8
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from utils.gate_config import (
    FAST_PATH_MAX_ENTRY_DEVIATION_PCT,
    FAST_PATH_MAX_TRIGGER_AGE_SECONDS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FastPathOutcome — immutable value object representing a single evaluation
# result.  Carries all fields needed for event persistence, stream narration,
# and audit.
# (Requirements: 3.1, 3.10, 9.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FastPathOutcome:
    """Immutable outcome produced by deterministic trigger evaluation.

    Every trigger evaluation that resolves (fires) produces exactly one
    FastPathOutcome.  The outcome captures what happened, why, and enough
    context for downstream event persistence and public stream narration.

    Required fields identify the trigger and market state at evaluation time.
    Optional fields carry geometry, blocking-rule details, and freeform
    metadata for audit enrichment.
    """

    # --- Required fields ---

    outcome_type: str
    """One of the six fast-path outcome types defined in fast_path_config.OUTCOME_TYPES."""

    outcome_reason_code: str
    """Structured code explaining the specific determination (e.g. 'target_already_crossed')."""

    trigger_id: str
    """UUID of the trigger that produced this outcome."""

    symbol: str
    """Ticker symbol evaluated."""

    profile_id: str
    """Trading profile that owns the trigger."""

    direction: str
    """Trade direction: 'BUY' or 'SHORT'."""

    setup_type: str
    """Setup classification (e.g. 'momentum_fade', 'gap_and_go')."""

    current_price: float
    """Market price at evaluation time."""

    # --- Optional fields (geometry, blocking info, metadata) ---

    entry_price: float | None = None
    """Frozen entry price from the trigger's registered geometry."""

    stop_price: float | None = None
    """Frozen stop price from the trigger's registered geometry."""

    target_price: float | None = None
    """Frozen target price from the trigger's registered geometry."""

    reward_to_risk: float | None = None
    """Computed reward-to-risk ratio at evaluation time."""

    blocking_rule_name: str | None = None
    """Name of the rule that blocked the setup (gate, cooldown, exposure)."""

    blocking_rule_threshold: str | None = None
    """Threshold value of the blocking rule, for audit context."""

    metadata: dict[str, Any] | None = None
    """Freeform additional context for event persistence and diagnostics."""


# ---------------------------------------------------------------------------
# Helper: Build FastPathOutcome with DRY pattern
# ---------------------------------------------------------------------------


def _make_outcome(
    trigger,
    quote,
    outcome_type: str,
    reason_code: str,
    **kwargs,
) -> FastPathOutcome:
    """Construct a FastPathOutcome from a trigger and quote with common fields populated."""
    return FastPathOutcome(
        outcome_type=outcome_type,
        outcome_reason_code=reason_code,
        trigger_id=trigger.trigger_id,
        symbol=trigger.symbol,
        profile_id=trigger.profile_id,
        direction=trigger.direction,
        setup_type=trigger.setup_type,
        current_price=quote.price if hasattr(quote, "price") else quote.get("price", 0.0),
        entry_price=trigger.entry_price,
        stop_price=trigger.stop_price,
        target_price=trigger.target_price,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Stub helper functions — will be fully implemented in tasks 4.3–4.6.
# Return conservative defaults so the code compiles and evaluates safely.
# ---------------------------------------------------------------------------


def target_crossed(direction: str, current_price: float, target_price: float) -> bool:
    """Check whether the target has already been crossed.

    BUY: target crossed when current_price >= target_price.
    SHORT: target crossed when current_price <= target_price.
    """
    if direction.upper() == "BUY":
        return current_price >= target_price
    elif direction.upper() == "SHORT":
        return current_price <= target_price
    return False


def trigger_condition_met(trigger, quote) -> bool:
    """Evaluate whether the trigger's specific condition has fired.

    Covers five trigger types:
      - entry_zone: price within [trigger_zone_lower, trigger_zone_upper]
      - level_break: price crossed trigger_level in the trigger direction
      - level_reject: price approached trigger_level (within 0.5% proximity)
      - vwap_cross: price crossed VWAP (trigger_level) in the trigger direction
      - price_target: price reached trigger_level in the trigger direction

    Returns True when the condition is met, False otherwise.
    """
    price = quote.price if hasattr(quote, "price") else quote.get("price", 0.0)

    if trigger.trigger_type == "entry_zone":
        # Price within entry zone bounds
        if trigger.trigger_zone_lower is None or trigger.trigger_zone_upper is None:
            return False
        return trigger.trigger_zone_lower <= price <= trigger.trigger_zone_upper

    elif trigger.trigger_type == "level_break":
        # Price crossed trigger_level in the trigger direction
        if trigger.direction.upper() == "BUY":
            return price >= trigger.trigger_level
        else:  # SHORT
            return price <= trigger.trigger_level

    elif trigger.trigger_type == "level_reject":
        # Price approached trigger_level and reversed.
        # Simplified check: price within 0.5% of trigger_level counts as
        # "at the level" — real rejection detection needs bar data.
        if trigger.trigger_level == 0:
            return False
        proximity = abs(price - trigger.trigger_level) / trigger.trigger_level
        return proximity <= 0.005

    elif trigger.trigger_type == "vwap_cross":
        # Price crossed VWAP (trigger_level) in the trigger direction
        if trigger.direction.upper() == "BUY":
            return price >= trigger.trigger_level
        else:  # SHORT
            return price <= trigger.trigger_level

    elif trigger.trigger_type == "price_target":
        # Price reached trigger_level in the trigger direction
        if trigger.direction.upper() == "BUY":
            return price >= trigger.trigger_level
        else:  # SHORT
            return price <= trigger.trigger_level

    return False


def requires_confirmation(trigger) -> bool:
    """Determine if the trigger needs post-level confirmation before action.

    Returns True for:
    - trigger_type == "level_reject" (needs confirmation that price reversed)
    - invalidation_basis containing "retest" or "confirmation" keywords
    """
    # level_reject always needs confirmation of reversal
    if trigger.trigger_type == "level_reject":
        return True

    # Check invalidation_basis for confirmation/retest keywords
    if trigger.invalidation_basis:
        basis_lower = trigger.invalidation_basis.lower()
        if "retest" in basis_lower or "confirmation" in basis_lower:
            return True

    return False


def price_away_but_limit_valid(trigger, quote) -> bool:
    """Check if price is away from entry but a valid pending limit order can be created.

    Conditions that must ALL be true:
    1. Price has run past the intended entry in the trade direction
       - BUY: current_price > entry_price (ran past entry, but target still ahead)
       - SHORT: current_price < entry_price (ran past entry, but target still ahead)
    2. Target has NOT been crossed (still room for profit)
    3. Geometry is valid at the limit price (entry) — using compute_geometry
    4. R:R is acceptable at the limit price (>= 1.0 as minimum threshold)

    (Requirements: 4.1, 4.2)
    """
    price = quote.price if hasattr(quote, "price") else quote.get("price", 0.0)
    direction = trigger.direction.upper()

    # Condition 1: Price has run past entry in trade direction
    if direction == "BUY":
        if price <= trigger.entry_price:
            return False  # Price hasn't run past entry
    elif direction == "SHORT":
        if price >= trigger.entry_price:
            return False  # Price hasn't run past entry
    else:
        return False

    # Condition 2: Target NOT crossed (still room for profit)
    if target_crossed(direction, price, trigger.target_price):
        return False

    # Condition 3 & 4: Geometry valid and R:R acceptable at entry (limit) price
    from utils.geometry_calculator import compute_geometry

    geometry = compute_geometry(
        direction,
        trigger.entry_price,
        trigger.stop_price,
        trigger.target_price,
        1,  # quantity=1 for validation
    )
    if not geometry.is_valid:
        return False

    # R:R must be at least 1.0 for the limit order to be worthwhile
    if geometry.reward_to_risk is not None:
        if float(geometry.reward_to_risk) < 1.0:
            return False

    return True


# ---------------------------------------------------------------------------
# Stub external system wrappers — thin wrappers that will be wired to real
# implementations later.  Placeholders return permissive defaults.
# ---------------------------------------------------------------------------


def _check_cooldown(trigger, profile_state: dict) -> tuple[bool, str | None]:
    """Check cooldown rules for this trigger.

    Returns:
        (blocked, reason): blocked=True means the trigger is cooldown-blocked,
        reason is the reason_code string (e.g. 'recent_trade', 'churn').

    Placeholder: returns (False, None) — no cooldown blocking.
    """
    return (False, None)


def _check_exposure(trigger, profile_state: dict) -> tuple[bool, str | None]:
    """Check exposure limits for this trigger.

    Returns:
        (blocked, reason): blocked=True means exposure/concentration would be
        violated, reason is the reason_code string.

    Placeholder: returns (False, None) — no exposure blocking.
    """
    return (False, None)


def _run_gates(trigger, quote, profile_state: dict) -> tuple[bool, str | None]:
    """Run the gate pipeline for a trigger that would produce trade_executed.

    Returns:
        (proceed, blocking_gate): proceed=True means all gates passed,
        blocking_gate is the name of the gate that rejected (when proceed=False).

    Placeholder: returns (True, None) — all gates pass.
    """
    return (True, None)


# ---------------------------------------------------------------------------
# Main evaluation function
# (Requirements: 2.1–2.11, 3.1–3.10, 5.1–5.7, 10.1–10.8)
# ---------------------------------------------------------------------------


def evaluate_trigger(trigger, quote, profile_state: dict) -> FastPathOutcome | None:
    """Evaluate a single trigger against fresh market data.

    Deterministic, priority-ordered evaluation.  Returns a FastPathOutcome when
    the trigger resolves (fire or stand-down), or None when the trigger condition
    is not met and the trigger stays ACTIVE.

    Parameters:
        trigger: TriggerRecord from utils.fast_path_registry.
        quote: Object or dict with fields: price (float), age_ms (int),
               reliable (bool).
        profile_state: Dict with profile context (profile_id, exposure info, etc.)

    Returns:
        FastPathOutcome when trigger resolves, or None if trigger stays active.

    Fail mode: fail-closed — any unexpected error produces stand_down.
    """
    try:
        return _evaluate_trigger_inner(trigger, quote, profile_state)
    except Exception as exc:
        logger.error(
            "fast_path_evaluator: unexpected error evaluating trigger %s for %s: %s",
            trigger.trigger_id,
            trigger.symbol,
            exc,
        )
        return _make_outcome(
            trigger,
            quote,
            "stand_down",
            "evaluation_error",
            metadata={"error": str(exc)},
        )


def _evaluate_trigger_inner(trigger, quote, profile_state: dict) -> FastPathOutcome | None:
    """Core evaluation logic — priority-ordered, first-match-wins.

    Separated from evaluate_trigger so the outer function provides the
    fail-closed try/except wrapper.
    """
    # Extract quote fields (support both object and dict)
    quote_price = quote.price if hasattr(quote, "price") else quote.get("price", 0.0)
    quote_age_ms = quote.age_ms if hasattr(quote, "age_ms") else quote.get("age_ms", 0)
    quote_reliable = quote.reliable if hasattr(quote, "reliable") else quote.get("reliable", True)

    # -----------------------------------------------------------------------
    # Priority 1: Market data freshness and reliability
    # (Requirements 10.1, 10.6, 2.10)
    # -----------------------------------------------------------------------
    max_age_ms = FAST_PATH_MAX_TRIGGER_AGE_SECONDS * 1000
    if quote_age_ms > max_age_ms:
        return _make_outcome(
            trigger,
            quote,
            "stand_down",
            "stale_market_data",
            metadata={"quote_age_ms": quote_age_ms, "max_age_ms": max_age_ms},
        )

    if not quote_reliable:
        return _make_outcome(
            trigger,
            quote,
            "stand_down",
            "market_data_unreliable",
        )

    # -----------------------------------------------------------------------
    # Priority 2: Target already crossed → missed_move
    # (Requirements 5.2, 2.4)
    # -----------------------------------------------------------------------
    if target_crossed(trigger.direction, quote_price, trigger.target_price):
        return _make_outcome(
            trigger,
            quote,
            "missed_move",
            "target_already_crossed",
        )

    # -----------------------------------------------------------------------
    # Priority 3: Geometry validation via compute_geometry()
    # (Requirements 10.7, 2.5)
    # -----------------------------------------------------------------------
    from utils.geometry_calculator import compute_geometry

    geometry = compute_geometry(
        trigger.direction,
        trigger.entry_price,
        trigger.stop_price,
        trigger.target_price,
        1,  # quantity=1 for geometry validation
    )
    if not geometry.is_valid:
        return _make_outcome(
            trigger,
            quote,
            "stand_down",
            "invalid_geometry",
            metadata={
                "validation_errors": [
                    {"field": e.field_name, "reason": e.reason}
                    for e in geometry.validation_errors
                ]
            },
        )

    # Capture reward_to_risk for downstream use
    reward_to_risk = float(geometry.reward_to_risk) if geometry.reward_to_risk else None

    # -----------------------------------------------------------------------
    # Priority 4: Cooldown check — fail-closed for execution path
    # (Requirements 6.1–6.8)
    # -----------------------------------------------------------------------
    cooldown_blocked, cooldown_reason = _check_cooldown(trigger, profile_state)
    if cooldown_blocked:
        return _make_outcome(
            trigger,
            quote,
            "stand_down",
            f"cooldown:{cooldown_reason}",
            blocking_rule_name="cooldown",
            blocking_rule_threshold=cooldown_reason,
        )

    # -----------------------------------------------------------------------
    # Priority 5: Exposure check
    # (Requirement 2.7)
    # -----------------------------------------------------------------------
    exposure_blocked, exposure_reason = _check_exposure(trigger, profile_state)
    if exposure_blocked:
        return _make_outcome(
            trigger,
            quote,
            "stand_down",
            f"exposure:{exposure_reason}",
            blocking_rule_name="exposure",
            blocking_rule_threshold=exposure_reason,
        )

    # -----------------------------------------------------------------------
    # Priority 6: Trigger condition evaluation
    # If trigger has not fired, return None — trigger stays ACTIVE.
    # -----------------------------------------------------------------------
    if not trigger_condition_met(trigger, quote):
        return None  # Trigger not fired, stays ACTIVE for next tick

    # -----------------------------------------------------------------------
    # Priority 7: Confirmation requirement → watch_created
    # (Requirements 4.3, 5.5)
    # -----------------------------------------------------------------------
    if requires_confirmation(trigger):
        return _make_outcome(
            trigger,
            quote,
            "watch_created",
            "awaiting_confirmation",
            reward_to_risk=reward_to_risk,
        )

    # -----------------------------------------------------------------------
    # Priority 8: Price away but valid limit → pending_order_created
    # Entry deviation MUST NOT preclude pending_order_created — pending orders
    # exist specifically for price-away scenarios.
    # (Requirements 4.1, 4.2, 5.3)
    # -----------------------------------------------------------------------
    if price_away_but_limit_valid(trigger, quote):
        return _make_outcome(
            trigger,
            quote,
            "pending_order_created",
            "limit_order_valid",
            reward_to_risk=reward_to_risk,
        )

    # -----------------------------------------------------------------------
    # Priority 9: Entry deviation check (only if NOT limit-eligible)
    # If price is far from entry and NOT eligible for pending order, stand down.
    # (Requirements 5.3)
    # -----------------------------------------------------------------------
    if trigger.entry_price and trigger.entry_price != 0.0:
        deviation = abs(quote_price - trigger.entry_price) / trigger.entry_price
        if deviation > FAST_PATH_MAX_ENTRY_DEVIATION_PCT:
            return _make_outcome(
                trigger,
                quote,
                "stand_down",
                "entry_too_far_from_price",
                metadata={
                    "deviation_pct": round(deviation, 4),
                    "max_deviation_pct": FAST_PATH_MAX_ENTRY_DEVIATION_PCT,
                },
                blocking_rule_name="entry_deviation",
                blocking_rule_threshold=str(FAST_PATH_MAX_ENTRY_DEVIATION_PCT),
            )

    # -----------------------------------------------------------------------
    # Priority 10/11: Gate pipeline → trade_executed or stand_down
    # (Requirements 10.2–10.5)
    # -----------------------------------------------------------------------
    gates_proceed, blocking_gate = _run_gates(trigger, quote, profile_state)
    if not gates_proceed:
        return _make_outcome(
            trigger,
            quote,
            "stand_down",
            f"gate_rejected:{blocking_gate}",
            reward_to_risk=reward_to_risk,
            blocking_rule_name=blocking_gate,
        )

    # All checks pass — trade_executed
    return _make_outcome(
        trigger,
        quote,
        "trade_executed",
        "all_gates_passed",
        reward_to_risk=reward_to_risk,
    )
