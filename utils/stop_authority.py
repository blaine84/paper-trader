"""
StopAuthority – Centralized stop-price validation, mutation, and trigger evaluation.

This module is the single source of truth for all stop-price operations in the
trading system. It enforces side-aware geometry rules, role-based validation,
repair-or-reject policy, and full audit logging.

All stop mutations MUST go through this module. Direct trade.stop_price
assignments are prohibited outside of this module, migrations, and test fixtures.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from db.schema import TradeEvent
from utils.trade_events import log_trade_event

log = logging.getLogger(__name__)


@dataclass
class StopValidationResult:
    """Result of stop geometry validation or stop update attempt.

    Returned by both ``validate_stop_geometry()`` and ``apply_stop_update()``
    to communicate the outcome of a stop validation or mutation request.

    Attributes:
        valid: Whether the stop passed geometry validation.
        reason_type: Classification of the outcome –
            "accepted", "rejected", "repair", or "review_required".
        reason: Human-readable explanation of the validation decision.
        old_stop: The previous stop price (populated by apply_stop_update).
        new_stop: The proposed/applied stop price.
        side: Trade side – "long" or "short".
        stop_role: The role classification of the stop –
            "initial", "breakeven", "trail", "manual", or "maintenance_tighten".
        repair_price: Suggested valid stop level when repair is possible.
    """

    valid: bool
    reason_type: str  # "accepted" | "rejected" | "repair" | "review_required"
    reason: str  # Human-readable explanation
    old_stop: Optional[float] = None
    new_stop: Optional[float] = None
    side: str = ""  # "long" | "short"
    stop_role: str = ""  # "initial" | "breakeven" | "trail" | "manual" | "maintenance_tighten"
    repair_price: Optional[float] = None  # Suggested valid stop if repair is possible


@dataclass
class StopTriggerResult:
    """Result of stop trigger evaluation.

    Returned by ``should_stop_trigger()`` to indicate whether a stop should
    fire and provide trigger metadata for downstream processing.

    Attributes:
        triggered: Whether the stop condition was met.
        trigger_price: The current price that caused (or was evaluated for) trigger.
        stop_price: The stop level being evaluated.
        buffered_level: The effective comparison level (stop ± buffer).
        side: Trade side – "long" or "short".
        geometry_valid: Whether the stop geometry passed validation.
        reason: Human-readable explanation of the trigger decision.
    """

    triggered: bool
    trigger_price: Optional[float] = None  # Current price that caused trigger
    stop_price: Optional[float] = None  # The stop level
    buffered_level: Optional[float] = None  # Stop ± buffer (actual comparison level)
    side: str = ""
    geometry_valid: bool = True
    reason: str = ""


# ---------------------------------------------------------------------------
# Internal helper functions
# ---------------------------------------------------------------------------

# Default distance used for repair price computation (percentage of entry)
_DEFAULT_REPAIR_DISTANCE = 0.02  # 2% from entry


def _compute_buffer(price: float, buffer_pct: float) -> float:
    """Compute execution buffer as percentage of price.

    Usage:
    - For validation (geometry checks): pass current_price → buffer_pct * current_price
    - For trigger (breach checks): pass stop_price → buffer_pct * stop_price
    """
    return price * buffer_pct


def _is_in_profit(side: str, entry_price: float, current_price: float) -> bool:
    """Determine if trade is currently in profit."""
    if side == "long":
        return current_price > entry_price
    else:
        return current_price < entry_price


def _effective_role_rules(
    stop_role: str, side: str, entry_price: float, current_price: Optional[float]
) -> str:
    """Determine which geometry rules to apply based on role and profit state.

    Returns "protective" or "profit_protecting".

    - initial/manual → always "protective"
    - breakeven/trail → always "profit_protecting"
    - maintenance_tighten → "profit_protecting" if in profit, else "protective"
    """
    if stop_role in ("initial", "manual"):
        return "protective"
    if stop_role in ("breakeven", "trail"):
        return "profit_protecting"
    # maintenance_tighten: conditional on profit state
    if current_price is not None and _is_in_profit(side, entry_price, current_price):
        return "profit_protecting"
    return "protective"


def _compute_repair_price(
    side: str,
    entry_price: float,
    current_price: Optional[float],
    buffer_pct: float,
) -> Optional[float]:
    """Compute a valid repair price for an invalid initial stop.

    For long: places stop at entry_price - (entry_price * default_distance)
    For short: places stop at entry_price + (entry_price * default_distance)

    Returns None if no valid repair is possible (e.g., no current_price available
    for profit-protecting roles, though for initial/protective repairs we only
    need entry_price).
    """
    if side == "long":
        repair = entry_price - (entry_price * _DEFAULT_REPAIR_DISTANCE)
    else:
        repair = entry_price + (entry_price * _DEFAULT_REPAIR_DISTANCE)

    # Sanity check: repair must be positive
    if repair <= 0:
        return None

    return round(repair, 6)


def _should_dedupe_geometry_event(
    db,
    trade_id: int,
    source_agent: str,
    stop_value: float,
    window_minutes: int = 15,
) -> bool:
    """Check if a stop_geometry_invalid event was already logged for this
    trade/source_agent/stop_value within the dedupe window.

    Queries recent events from DB, filters stop_value in Python
    (avoids fragile JSON LIKE patterns).

    Returns True if event should be suppressed (duplicate exists).
    """
    cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
    recent = (
        db.query(TradeEvent)
        .filter(
            TradeEvent.trade_id == trade_id,
            TradeEvent.event_type == "stop_geometry_invalid",
            TradeEvent.agent == source_agent,
            TradeEvent.timestamp > cutoff,
        )
        .all()
    )
    for event in recent:
        payload = json.loads(event.payload_json) if event.payload_json else {}
        if payload.get("stop_price") == stop_value:
            return True  # suppress duplicate
    return False


def _log_stop_event(
    db,
    event_type: str,
    *,
    trade_id: int,
    source_agent: str,
    symbol: Optional[str] = None,
    profile: Optional[str] = None,
    price: Optional[float] = None,
    message: Optional[str] = None,
    payload: Optional[dict] = None,
) -> None:
    """Wrapper around log_trade_event for stop-specific events."""
    log_trade_event(
        db,
        event_type,
        trade_id=trade_id,
        agent=source_agent,
        symbol=symbol,
        profile=profile,
        price=price,
        message=message,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_VALID_SIDES = ("long", "short")
_VALID_STOP_ROLES = ("initial", "breakeven", "trail", "manual", "maintenance_tighten")


def validate_stop_geometry(
    *,
    side: str,
    entry_price: float,
    current_price: Optional[float],
    stop_price: float,
    target_price: Optional[float] = None,
    stop_role: str,
    min_distance_pct: Optional[float] = None,
    max_distance_pct: Optional[float] = None,
    buffer_pct: float = 0.001,
) -> StopValidationResult:
    """Validate stop geometry for a given trade configuration.

    Pure function — no DB access, no side effects.

    Rules by role:
    - initial/manual: stop must be on loss side of entry (and current if known)
    - breakeven/trail: stop may be past entry only if current is favorable;
      must maintain buffer_pct distance from current
    - maintenance_tighten: uses profit-protecting rules if in profit
      (current favorable vs entry), protective rules otherwise

    Raises:
        ValueError: If side, entry_price, stop_price, or stop_role are invalid.
    """
    # --- Input validation ---
    if side not in _VALID_SIDES:
        raise ValueError(f"Invalid side: {side!r}. Must be one of {_VALID_SIDES}.")
    if entry_price <= 0:
        raise ValueError(f"entry_price must be positive, got {entry_price}")
    if stop_price <= 0:
        raise ValueError(f"stop_price must be positive, got {stop_price}")
    if stop_role not in _VALID_STOP_ROLES:
        raise ValueError(
            f"Invalid stop_role: {stop_role!r}. Must be one of {_VALID_STOP_ROLES}."
        )

    # Determine which rule set applies
    rule_set = _effective_role_rules(stop_role, side, entry_price, current_price)

    # --- Protective rules (initial / manual, or maintenance_tighten when not in profit) ---
    if rule_set == "protective":
        return _validate_protective(
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            stop_price=stop_price,
            stop_role=stop_role,
            buffer_pct=buffer_pct,
        )

    # --- Profit-protecting rules (breakeven / trail, or maintenance_tighten when in profit) ---
    return _validate_profit_protecting(
        side=side,
        entry_price=entry_price,
        current_price=current_price,
        stop_price=stop_price,
        stop_role=stop_role,
        buffer_pct=buffer_pct,
    )


def _validate_protective(
    *,
    side: str,
    entry_price: float,
    current_price: Optional[float],
    stop_price: float,
    stop_role: str,
    buffer_pct: float,
) -> StopValidationResult:
    """Apply protective-stop geometry rules.

    Long: stop must be strictly < entry AND strictly < current (if known).
    Short: stop must be strictly > entry AND strictly > current (if known).
    """
    if side == "long":
        # Stop must be below entry
        if stop_price >= entry_price:
            repair = _compute_repair_price(side, entry_price, current_price, buffer_pct)
            return StopValidationResult(
                valid=False,
                reason_type="rejected" if repair is None else "repair",
                reason=(
                    f"Long protective stop ({stop_price}) must be below "
                    f"entry ({entry_price})"
                ),
                new_stop=stop_price,
                side=side,
                stop_role=stop_role,
                repair_price=repair,
            )
        # Stop must be below current (if known)
        if current_price is not None and stop_price >= current_price:
            repair = _compute_repair_price(side, entry_price, current_price, buffer_pct)
            return StopValidationResult(
                valid=False,
                reason_type="rejected" if repair is None else "repair",
                reason=(
                    f"Long protective stop ({stop_price}) must be below "
                    f"current price ({current_price})"
                ),
                new_stop=stop_price,
                side=side,
                stop_role=stop_role,
                repair_price=repair,
            )
    else:  # short
        # Stop must be above entry
        if stop_price <= entry_price:
            repair = _compute_repair_price(side, entry_price, current_price, buffer_pct)
            return StopValidationResult(
                valid=False,
                reason_type="rejected" if repair is None else "repair",
                reason=(
                    f"Short protective stop ({stop_price}) must be above "
                    f"entry ({entry_price})"
                ),
                new_stop=stop_price,
                side=side,
                stop_role=stop_role,
                repair_price=repair,
            )
        # Stop must be above current (if known)
        if current_price is not None and stop_price <= current_price:
            repair = _compute_repair_price(side, entry_price, current_price, buffer_pct)
            return StopValidationResult(
                valid=False,
                reason_type="rejected" if repair is None else "repair",
                reason=(
                    f"Short protective stop ({stop_price}) must be above "
                    f"current price ({current_price})"
                ),
                new_stop=stop_price,
                side=side,
                stop_role=stop_role,
                repair_price=repair,
            )

    # All checks passed
    return StopValidationResult(
        valid=True,
        reason_type="accepted",
        reason="Protective stop geometry valid",
        new_stop=stop_price,
        side=side,
        stop_role=stop_role,
    )


def _validate_profit_protecting(
    *,
    side: str,
    entry_price: float,
    current_price: Optional[float],
    stop_price: float,
    stop_role: str,
    buffer_pct: float,
) -> StopValidationResult:
    """Apply profit-protecting stop geometry rules.

    Requires current_price to be known and trade to be in profit.
    Long: stop must be < current_price - buffer.
    Short: stop must be > current_price + buffer.
    """
    # Cannot validate profit-protecting without current price
    if current_price is None:
        return StopValidationResult(
            valid=False,
            reason_type="rejected",
            reason=(
                "Cannot validate profit-protecting stop without current_price"
            ),
            new_stop=stop_price,
            side=side,
            stop_role=stop_role,
        )

    # Trade must be in profit for profit-protecting stops
    if not _is_in_profit(side, entry_price, current_price):
        return StopValidationResult(
            valid=False,
            reason_type="rejected",
            reason=(
                f"Profit-protecting stop requires trade to be in profit "
                f"(side={side}, entry={entry_price}, current={current_price})"
            ),
            new_stop=stop_price,
            side=side,
            stop_role=stop_role,
        )

    # Compute buffer based on current_price (for validation)
    buffer = _compute_buffer(current_price, buffer_pct)

    if side == "long":
        # Stop must be below current - buffer
        max_allowed = current_price - buffer
        if stop_price >= max_allowed:
            return StopValidationResult(
                valid=False,
                reason_type="rejected",
                reason=(
                    f"Long profit-protecting stop ({stop_price}) must be below "
                    f"current ({current_price}) minus buffer ({buffer:.6f}), "
                    f"max allowed: {max_allowed:.6f}"
                ),
                new_stop=stop_price,
                side=side,
                stop_role=stop_role,
            )
    else:  # short
        # Stop must be above current + buffer
        min_allowed = current_price + buffer
        if stop_price <= min_allowed:
            return StopValidationResult(
                valid=False,
                reason_type="rejected",
                reason=(
                    f"Short profit-protecting stop ({stop_price}) must be above "
                    f"current ({current_price}) plus buffer ({buffer:.6f}), "
                    f"min allowed: {min_allowed:.6f}"
                ),
                new_stop=stop_price,
                side=side,
                stop_role=stop_role,
            )

    # All checks passed
    return StopValidationResult(
        valid=True,
        reason_type="accepted",
        reason="Profit-protecting stop geometry valid",
        new_stop=stop_price,
        side=side,
        stop_role=stop_role,
    )


def should_stop_trigger(
    *,
    side: str,
    entry_price: float,
    current_price: float,
    stop_price: float,
    stop_role: str | None = None,
    buffer_pct: float = 0.001,
    # Optional context for audit logging
    db=None,
    trade_id: int | None = None,
    symbol: str | None = None,
    profile: str | None = None,
    source_agent: str | None = None,
) -> StopTriggerResult:
    """Evaluate whether a stop should trigger for the given trade state.

    Used by both bookkeeper and price_monitor for identical trigger decisions.

    Logic:
    1. Validate geometry (using stop_role or defaulting to "initial")
    2. If geometry invalid:
       - Return non-trigger result (triggered=False, geometry_valid=False)
       - If db/trade context provided: emit stop_geometry_invalid event (subject to dedupe)
    3. If geometry valid:
       - Long: trigger if current_price <= stop_price - buffer
       - Short: trigger if current_price >= stop_price + buffer
       - Buffer = buffer_pct * stop_price
    4. Trigger direction and buffer are INDEPENDENT of stop_role

    Buffer basis distinction:
    - Validation (geometry checks): buffer = buffer_pct * current_price
    - Trigger (breach checks): buffer = buffer_pct * stop_price
    """
    # Default stop_role to "initial" if not provided
    effective_role = stop_role if stop_role is not None else "initial"

    # Step 1: Validate fundamental geometry for trigger evaluation
    # The geometry check for triggers is simpler than for mutations:
    # - Protective roles (initial/manual): stop must be on loss side of entry
    #   (long: stop < entry, short: stop > entry). We pass current_price=None
    #   to skip the proximity check which is a mutation guard, not a trigger gate.
    # - Profit-protecting roles (breakeven/trail/maintenance_tighten): the stop
    #   was already validated when set via apply_stop_update(). For trigger
    #   evaluation, we only need to verify fundamental directional validity.
    #   A trailing stop above entry for a long is valid (it was moved up in profit).
    #   We use protective validation without current_price — if stop is below entry
    #   for a long trail, that's still valid (it just hasn't been moved above yet).
    #
    # Per Req 4.7: stop_role affects geometry validation rules only, not trigger
    # direction or buffer application.
    if effective_role in ("initial", "manual"):
        # Protective: stop must be on loss side of entry
        validation = validate_stop_geometry(
            side=side,
            entry_price=entry_price,
            current_price=None,
            stop_price=stop_price,
            stop_role=effective_role,
            buffer_pct=buffer_pct,
        )
    else:
        # Profit-protecting roles: stop was validated at set time.
        # For trigger purposes, geometry is valid as long as the stop is positive
        # and the trade configuration is sensible. We skip the full validation
        # which would reject stops near current (exactly the trigger scenario).
        # The only truly invalid case is a stop on the completely wrong side
        # (e.g., short stop below entry with initial role), but profit-protecting
        # stops are allowed on either side of entry by design.
        validation = StopValidationResult(
            valid=True,
            reason_type="accepted",
            reason="Profit-protecting stop geometry accepted for trigger evaluation",
            new_stop=stop_price,
            side=side,
            stop_role=effective_role,
        )

    # Step 2: If geometry invalid, return non-trigger
    if not validation.valid:
        # Emit dedupe-controlled event if db context provided
        if db is not None and trade_id is not None and source_agent is not None:
            if not _should_dedupe_geometry_event(db, trade_id, source_agent, stop_price):
                _log_stop_event(
                    db,
                    "stop_geometry_invalid",
                    trade_id=trade_id,
                    source_agent=source_agent,
                    symbol=symbol,
                    profile=profile,
                    price=current_price,
                    message=validation.reason,
                    payload={
                        "source_agent": source_agent,
                        "stop_price": stop_price,
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "side": side,
                    },
                )

        return StopTriggerResult(
            triggered=False,
            trigger_price=current_price,
            stop_price=stop_price,
            buffered_level=None,
            side=side,
            geometry_valid=False,
            reason=f"Geometry invalid: {validation.reason}",
        )

    # Step 3: Geometry valid — evaluate trigger condition
    # Buffer for trigger uses stop_price as basis
    buffer = _compute_buffer(stop_price, buffer_pct)

    if side == "long":
        buffered_level = stop_price - buffer
        triggered = current_price <= buffered_level
    else:  # short
        buffered_level = stop_price + buffer
        triggered = current_price >= buffered_level

    if triggered:
        reason = (
            f"Stop triggered: {side} current_price={current_price} "
            f"{'<=' if side == 'long' else '>='} buffered_level={buffered_level:.6f}"
        )
    else:
        reason = (
            f"Stop not triggered: {side} current_price={current_price} "
            f"{'>' if side == 'long' else '<'} buffered_level={buffered_level:.6f}"
        )

    return StopTriggerResult(
        triggered=triggered,
        trigger_price=current_price,
        stop_price=stop_price,
        buffered_level=buffered_level,
        side=side,
        geometry_valid=True,
        reason=reason,
    )


def apply_stop_update(
    db,
    *,
    trade,
    new_stop: float,
    source_agent: str,
    stop_role: str,
    reason: str,
    current_price: float | None = None,
    buffer_pct: float = 0.001,
) -> StopValidationResult:
    """Validate, persist, and audit a stop mutation.

    Transaction ownership: ALWAYS flushes only. Caller commits.

    Caller commit policy:
    - If result is valid: caller commits (mutation + audit events persisted).
    - If result is invalid: caller SHOULD commit to persist the audit-only events
      (stop_update_requested + stop_update_rejected/stop_review_required), unless
      prior caller-side mutations require rollback.

    Behavior:
    1. Logs stop_update_requested event
    2. Calls validate_stop_geometry() for the proposed stop
    3. If valid: updates trade.stop_price, trade.stop_role, trade.stop_updated_by,
       trade.stop_updated_at; logs stop_update_accepted; flushes
    4. If invalid: validates existing stop using trade.stop_role (not proposed role)
       a. If existing stop is valid: rejects proposed; logs stop_update_rejected
       b. If existing stop is also invalid: logs stop_review_required event
    5. If repairable: sets repair_price in result but does NOT auto-apply.
       Caller decides whether to call apply_stop_update() again with repair_price.

    Returns StopValidationResult with full context.
    """
    # Derive side from trade.direction ("LONG" → "long", "SHORT" → "short")
    direction = getattr(trade, "direction", None) or ""
    side = direction.lower() if direction else "long"

    entry_price = getattr(trade, "entry_price", None) or 0.0
    old_stop = getattr(trade, "stop_price", None)
    trade_id = getattr(trade, "id", None)
    symbol = getattr(trade, "symbol", None)
    profile = getattr(trade, "profile", None)

    # Step 1: Log stop_update_requested event
    _log_stop_event(
        db,
        "stop_update_requested",
        trade_id=trade_id,
        source_agent=source_agent,
        symbol=symbol,
        profile=profile,
        price=new_stop,
        message=reason,
        payload={
            "source_agent": source_agent,
            "proposed_stop": new_stop,
            "stop_role": stop_role,
            "reason": reason,
            "old_stop": old_stop,
            "entry_price": entry_price,
            "current_price": current_price,
            "side": side,
        },
    )

    # Step 2: Validate proposed stop geometry
    validation = validate_stop_geometry(
        side=side,
        entry_price=entry_price,
        current_price=current_price,
        stop_price=new_stop,
        stop_role=stop_role,
        buffer_pct=buffer_pct,
    )

    # Step 3: If valid — apply the update
    if validation.valid:
        trade.stop_price = new_stop

        # Update optional metadata columns using setattr with hasattr guard
        if hasattr(trade, "stop_role"):
            trade.stop_role = stop_role
        if hasattr(trade, "stop_updated_by"):
            trade.stop_updated_by = source_agent
        if hasattr(trade, "stop_updated_at"):
            trade.stop_updated_at = datetime.utcnow()

        # Log stop_update_accepted event
        _log_stop_event(
            db,
            "stop_update_accepted",
            trade_id=trade_id,
            source_agent=source_agent,
            symbol=symbol,
            profile=profile,
            price=new_stop,
            message=f"Stop updated: {old_stop} → {new_stop} ({reason})",
            payload={
                "source_agent": source_agent,
                "old_stop": old_stop,
                "new_stop": new_stop,
                "stop_role": stop_role,
                "reason": reason,
            },
        )

        # Flush only — caller owns the transaction
        db.flush()

        return StopValidationResult(
            valid=True,
            reason_type="accepted",
            reason=f"Stop updated: {old_stop} → {new_stop} ({reason})",
            old_stop=old_stop,
            new_stop=new_stop,
            side=side,
            stop_role=stop_role,
            repair_price=None,
        )

    # Step 4: Invalid proposed stop — check existing stop validity
    existing_stop_role = getattr(trade, "stop_role", None) or "initial"

    if old_stop is not None and old_stop > 0:
        existing_validation = validate_stop_geometry(
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            stop_price=old_stop,
            stop_role=existing_stop_role,
            buffer_pct=buffer_pct,
        )
    else:
        # No existing stop or invalid value — treat as invalid
        existing_validation = StopValidationResult(
            valid=False,
            reason_type="rejected",
            reason="No valid existing stop price",
            side=side,
            stop_role=existing_stop_role,
        )

    if existing_validation.valid:
        # Existing stop is valid — reject proposed, preserve existing
        _log_stop_event(
            db,
            "stop_update_rejected",
            trade_id=trade_id,
            source_agent=source_agent,
            symbol=symbol,
            profile=profile,
            price=new_stop,
            message=f"Rejected: {validation.reason}",
            payload={
                "source_agent": source_agent,
                "proposed_stop": new_stop,
                "existing_stop": old_stop,
                "rejection_reason": validation.reason,
                "stop_role": stop_role,
            },
        )

        return StopValidationResult(
            valid=False,
            reason_type="rejected",
            reason=validation.reason,
            old_stop=old_stop,
            new_stop=new_stop,
            side=side,
            stop_role=stop_role,
            repair_price=validation.repair_price,
        )
    else:
        # Both existing and proposed are invalid — emit review_required
        _log_stop_event(
            db,
            "stop_review_required",
            trade_id=trade_id,
            source_agent=source_agent,
            symbol=symbol,
            profile=profile,
            price=new_stop,
            message=(
                f"Both proposed ({new_stop}) and existing ({old_stop}) stops "
                f"are geometrically invalid. Manual review required."
            ),
            payload={
                "source_agent": source_agent,
                "proposed_stop": new_stop,
                "existing_stop": old_stop,
                "proposed_reason": validation.reason,
                "existing_reason": existing_validation.reason,
                "side": side,
            },
        )

        return StopValidationResult(
            valid=False,
            reason_type="review_required",
            reason=(
                f"Both proposed ({new_stop}) and existing ({old_stop}) stops "
                f"are invalid. Review required."
            ),
            old_stop=old_stop,
            new_stop=new_stop,
            side=side,
            stop_role=stop_role,
            repair_price=validation.repair_price,
        )
