"""
Setup-Aware Lifecycle Evaluator — Deterministic exit governance by setup type.

Replaces the generic ~90-minute force-close timer with setup-specific lifecycle
logic. Consults the Setup_Time_Policy registry for per-setup timing, distinguishes
thesis-development setups from fast tactical setups, and requires explicit
invalidation criteria for any extension beyond base force-close.

This module is a pure function evaluator: no DB writes, no side effects.
Event logging is handled by the caller (position_timer executor).
"""

from __future__ import annotations

import os
from datetime import datetime, time, timedelta, timezone
from typing import Any

from utils.setup_time_policy import SetupTimePolicy, get_policy


# ---------------------------------------------------------------------------
# Constants: Revalidation Decision Outcomes
# ---------------------------------------------------------------------------

REVALIDATION_DECISIONS = {
    "hold_valid_until_next_window",
    "close_time_expired",
    "close_thesis_invalidated",
    "warn_revalidation_due",
    "insufficient_data_close_at_base_limit",
}

# ---------------------------------------------------------------------------
# Constants: Setup-Aware Event Types
# ---------------------------------------------------------------------------

SETUP_EXIT_EVENT_TYPES = {
    "setup_exit_alert",
    "setup_exit_revalidation_due",
    "setup_exit_revalidated_hold",
    "setup_exit_revalidation_failed",
    "setup_exit_force_close",
    "setup_exit_thesis_invalidated",
}

# ---------------------------------------------------------------------------
# Constants: Setup-Aware Lifecycle States
# ---------------------------------------------------------------------------

SETUP_AWARE_STATES = {
    "setup_exit_alert",
    "setup_revalidation_hold",
    "setup_revalidation_failed",
    "setup_time_limit_exceeded",
    "setup_thesis_invalidated",
}

# ---------------------------------------------------------------------------
# Constants: Setup-Aware Reason Types
# ---------------------------------------------------------------------------

SETUP_AWARE_REASON_TYPES = {
    "setup_alert_approaching_revalidation",
    "setup_revalidation_hold_valid",
    "setup_revalidation_denied_missing_criteria",
    "setup_revalidation_denied_stale_data",
    "setup_revalidation_denied_invalid_stop",
    "setup_time_limit_exceeded",
    "setup_thesis_invalidated",
    "setup_max_extension_reached",
    "setup_pre_wall_buffer_close",
}

# ---------------------------------------------------------------------------
# Pre-wall buffer: 15 minutes before EOD hard wall
# ---------------------------------------------------------------------------

PRE_WALL_BUFFER_MINUTES = 15

# ---------------------------------------------------------------------------
# Lifecycle State → Event Type Mapping
# ---------------------------------------------------------------------------

LIFECYCLE_STATE_TO_EVENT_TYPE: dict[str, str] = {
    "setup_exit_alert": "setup_exit_alert",
    "setup_revalidation_hold": "setup_exit_revalidated_hold",
    "setup_revalidation_failed": "setup_exit_revalidation_failed",
    "setup_time_limit_exceeded": "setup_exit_force_close",
    "setup_thesis_invalidated": "setup_exit_thesis_invalidated",
}


# ---------------------------------------------------------------------------
# Public API: Dedupe Key Generation
# ---------------------------------------------------------------------------


def generate_dedupe_key(event_type: str, trade_id: int, timestamp_utc: datetime) -> str:
    """Generate a dedupe key for setup-aware exit events.

    Format: "{event_type}:{trade_id}:{decision_timestamp_truncated_to_minute}"
    Example: "setup_exit_revalidated_hold:42:2026-05-26T14:30"

    The timestamp is truncated to the minute (no seconds/microseconds) to allow
    deduplication of events emitted within the same minute for the same trade
    and event type.

    Args:
        event_type: One of SETUP_EXIT_EVENT_TYPES.
        trade_id: The trade's unique identifier.
        timestamp_utc: The decision timestamp (will be truncated to minute).

    Returns:
        A string dedupe key.
    """
    # Truncate to minute: remove seconds and microseconds
    truncated = timestamp_utc.replace(second=0, microsecond=0)
    # Format as ISO-8601 without seconds: YYYY-MM-DDTHH:MM
    ts_str = truncated.strftime("%Y-%m-%dT%H:%M")
    return f"{event_type}:{trade_id}:{ts_str}"


# ---------------------------------------------------------------------------
# Public API: Event Payload Construction
# ---------------------------------------------------------------------------


def build_setup_exit_event_payload(
    trade_id: int,
    setup_type: str,
    event_type: str,
    minutes_held: float,
    decision_outcome: str,
    invalidation_criteria_used: dict | None,
    timestamp_utc: datetime,
    base_force_close_limit: int,
    next_limit_if_extended: int | None,
    revalidation_attempted: bool,
    extension_active: bool,
    close_reason: str | None,
) -> dict:
    """Construct a SetupExitEventPayload with all required fields.

    This payload is used by the position_timer executor when logging
    setup-aware lifecycle events to trade_events.

    The dedupe_key is automatically generated from event_type, trade_id,
    and timestamp_utc.

    Args:
        trade_id: The trade's unique identifier.
        setup_type: The trade's setup type classification.
        event_type: One of SETUP_EXIT_EVENT_TYPES.
        minutes_held: Minutes the trade has been held since entry.
        decision_outcome: The revalidation/lifecycle decision outcome string.
        invalidation_criteria_used: Dict of validated criteria, or None.
        timestamp_utc: UTC timestamp of the decision.
        base_force_close_limit: The policy's force_close_minutes value.
        next_limit_if_extended: Next revalidation boundary if extended, or None.
        revalidation_attempted: Whether revalidation was attempted this cycle.
        extension_active: Whether an extension is currently active.
        close_reason: Human-readable close reason, or None if not closing.

    Returns:
        A dict matching the SetupExitEventPayload schema with dedupe_key included.
    """
    dedupe_key = generate_dedupe_key(event_type, trade_id, timestamp_utc)

    return {
        "trade_id": trade_id,
        "setup_type": setup_type,
        "event_type": event_type,
        "minutes_held": minutes_held,
        "decision_outcome": decision_outcome,
        "invalidation_criteria_used": invalidation_criteria_used,
        "timestamp_utc": timestamp_utc.isoformat(),
        "base_force_close_limit": base_force_close_limit,
        "next_limit_if_extended": next_limit_if_extended,
        "revalidation_attempted": revalidation_attempted,
        "extension_active": extension_active,
        "close_reason": close_reason,
        "dedupe_key": dedupe_key,
    }


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _validate_invalidation_criteria(
    trade: dict,
    policy: SetupTimePolicy,
) -> tuple[bool, str, dict]:
    """Validate that trade has sufficient invalidation criteria for extension.

    Returns:
        (is_valid, reason, criteria_used)
        - is_valid: True if criteria meet minimum requirements
        - reason: Human-readable explanation
        - criteria_used: Dict of validated criteria for event payload

    Validation rules:
    - At minimum: one numeric stop price on loss side of entry, OR
      one structural invalidation level (named support/resistance/VWAP as numeric price)
    - Stop price must be non-zero, on loss side of entry (below for long, above for short)
    - Structural levels must be parseable as numeric prices
    - Confidence scores, target prices, or qualitative sentiment alone are NOT sufficient
    - For news_breakout: also accepts catalyst-specific invalidation with deterministic trigger
    """
    direction = (trade.get("direction") or "").upper()
    entry_price = trade.get("entry_price")
    criteria_used: dict = {}

    # Must have a valid entry price to evaluate anything
    if entry_price is None or not _is_numeric(entry_price) or float(entry_price) <= 0:
        return (False, "missing or invalid entry_price", {})

    entry_price = float(entry_price)

    # --- Check stop price validity ---
    stop_price = trade.get("stop_price") or trade.get("stop_loss")
    stop_valid = False

    if stop_price is not None and _is_numeric(stop_price):
        stop_val = float(stop_price)
        if stop_val != 0.0:
            # Validate stop is on loss side of entry
            if direction == "LONG" and stop_val < entry_price:
                stop_valid = True
                criteria_used["stop_price"] = stop_val
            elif direction == "SHORT" and stop_val > entry_price:
                stop_valid = True
                criteria_used["stop_price"] = stop_val

    if stop_valid:
        return (True, "valid stop price on loss side of entry", criteria_used)

    # --- Check structural invalidation levels ---
    structural_valid = False
    invalidators = trade.get("invalidators")

    # invalidators may be a dict or a list
    structural_keys = {"vwap", "support", "resistance", "vwap_level", "support_level", "resistance_level"}

    if isinstance(invalidators, dict):
        for key, value in invalidators.items():
            key_lower = key.lower()
            if key_lower in structural_keys and _is_numeric(value):
                level_val = float(value)
                if level_val > 0:
                    structural_valid = True
                    criteria_used[f"{key_lower}_level"] = level_val
                    break
    elif isinstance(invalidators, list):
        for item in invalidators:
            if isinstance(item, dict):
                for key, value in item.items():
                    key_lower = key.lower()
                    if key_lower in structural_keys and _is_numeric(value):
                        level_val = float(value)
                        if level_val > 0:
                            structural_valid = True
                            criteria_used[f"{key_lower}_level"] = level_val
                            break
                if structural_valid:
                    break
            elif _is_numeric(item):
                # A bare numeric in the list — treat as structural level
                level_val = float(item)
                if level_val > 0:
                    structural_valid = True
                    criteria_used["structural_level"] = level_val
                    break

    if structural_valid:
        return (True, "valid structural invalidation level", criteria_used)

    # --- For news_breakout: accept catalyst-specific invalidation with deterministic trigger ---
    if policy.setup_type == "news_breakout":
        catalyst_valid = _check_catalyst_invalidation(trade)
        if catalyst_valid:
            criteria_used["catalyst_invalidation"] = True
            return (True, "catalyst-specific invalidation with deterministic trigger", criteria_used)

    # --- Reject insufficient criteria ---
    # Check if only qualitative/insufficient data is present
    has_confidence = trade.get("confidence") is not None or trade.get("confidence_score") is not None
    has_target_only = trade.get("target_price") is not None
    has_sentiment = trade.get("sentiment") is not None

    if has_confidence or has_target_only or has_sentiment:
        return (
            False,
            "confidence scores, target price, or qualitative sentiment alone are insufficient for extension",
            {},
        )

    return (False, "no valid numeric stop price or structural invalidation level found", {})


def _is_numeric(value: Any) -> bool:
    """Check if a value can be parsed as a numeric float."""
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    return False


def _check_catalyst_invalidation(trade: dict) -> bool:
    """Check if a news_breakout trade has catalyst-specific invalidation with a deterministic trigger.

    A deterministic trigger is one that references a measurable market condition:
    price level, volume threshold, VWAP failure, headline reversal flag, or explicit time boundary.
    """
    invalidators = trade.get("invalidators")
    thesis = trade.get("thesis") or ""

    # Check invalidators dict/list for catalyst-specific entries
    if isinstance(invalidators, dict):
        # Look for catalyst_invalidation key with a deterministic trigger
        catalyst_inv = invalidators.get("catalyst_invalidation") or invalidators.get("catalyst")
        if catalyst_inv:
            if isinstance(catalyst_inv, dict):
                # Must have a trigger field with a numeric or measurable condition
                trigger = catalyst_inv.get("trigger") or catalyst_inv.get("price_level") or catalyst_inv.get("level")
                if trigger is not None and _is_numeric(trigger):
                    return True
                # Accept string triggers that reference measurable conditions
                trigger_str = str(catalyst_inv.get("trigger", ""))
                if _has_deterministic_trigger_language(trigger_str):
                    return True
            elif isinstance(catalyst_inv, str):
                if _has_deterministic_trigger_language(catalyst_inv):
                    return True

    elif isinstance(invalidators, list):
        for item in invalidators:
            if isinstance(item, dict):
                item_type = (item.get("type") or item.get("name") or "").lower()
                if "catalyst" in item_type:
                    trigger = item.get("trigger") or item.get("price_level") or item.get("level")
                    if trigger is not None and _is_numeric(trigger):
                        return True
                    trigger_str = str(item.get("trigger", ""))
                    if _has_deterministic_trigger_language(trigger_str):
                        return True

    return False


def _has_deterministic_trigger_language(text: str) -> bool:
    """Check if text references a measurable market condition (deterministic trigger).

    Deterministic triggers include: price levels, volume thresholds, VWAP failure,
    headline reversal flags, or explicit time boundaries.
    """
    if not text:
        return False
    text_lower = text.lower()
    # Keywords indicating deterministic/measurable triggers
    deterministic_indicators = [
        "price", "level", "below", "above", "break",
        "volume", "vwap", "support", "resistance",
        "reversal", "headline", "time", "minute", "hour",
        "threshold", "breach", "fail",
    ]
    return any(indicator in text_lower for indicator in deterministic_indicators)


def _evaluate_revalidation_decision(
    trade: dict,
    policy: SetupTimePolicy,
    current_price: float,
    minutes_held: float,
    criteria: dict,
) -> tuple[str, str, dict]:
    """Produce exactly one revalidation decision.

    Returns:
        (decision_outcome, reason, metadata)

    Decision logic (deterministic, in order):
    1. If current_price has breached stop/invalidation level → close_thesis_invalidated
    2. If minutes_held >= max_extension_minutes → close_time_expired
    3. If price above stop AND (price >= entry OR price >= VWAP/support OR
       target_progress >= 25%) → hold_valid_until_next_window
    4. Otherwise → close_time_expired (no positive indicator)

    Target progress = (current_price - entry_price) / (target_price - entry_price) * 100
    for longs; inverted for shorts. Division-by-zero yields 0%.
    """
    direction = (trade.get("direction") or "LONG").upper()
    entry_price = trade.get("entry_price", 0.0) or 0.0
    target_price = trade.get("target_price") or None

    # Determine the effective stop/invalidation level from criteria dict
    stop_price = criteria.get("stop_price")
    # Structural levels from criteria (optional)
    vwap_level = criteria.get("vwap_level")
    support_level = criteria.get("support_level")
    resistance_level = criteria.get("resistance_level")

    # --- Step 1: Check if current_price has breached stop/invalidation level ---
    breached = False
    breach_level = None
    breach_type = None

    if direction == "LONG":
        # For LONG: breached if current_price <= stop_price (or structural level)
        if stop_price is not None and current_price <= stop_price:
            breached = True
            breach_level = stop_price
            breach_type = "stop_price"
        elif support_level is not None and current_price <= support_level:
            breached = True
            breach_level = support_level
            breach_type = "support_level"
        elif vwap_level is not None and current_price <= vwap_level:
            breached = True
            breach_level = vwap_level
            breach_type = "vwap_level"
    else:
        # For SHORT: breached if current_price >= stop_price (or structural level)
        if stop_price is not None and current_price >= stop_price:
            breached = True
            breach_level = stop_price
            breach_type = "stop_price"
        elif resistance_level is not None and current_price >= resistance_level:
            breached = True
            breach_level = resistance_level
            breach_type = "resistance_level"
        elif vwap_level is not None and current_price >= vwap_level:
            breached = True
            breach_level = vwap_level
            breach_type = "vwap_level"

    if breached:
        return (
            "close_thesis_invalidated",
            f"price {current_price} breached {breach_type} at {breach_level} ({direction})",
            {
                "current_price": current_price,
                "breach_level": breach_level,
                "breach_type": breach_type,
                "direction": direction,
                "entry_price": entry_price,
            },
        )

    # --- Step 2: Check if minutes_held >= max_extension_minutes ---
    if (
        policy.max_extension_minutes is not None
        and minutes_held >= policy.max_extension_minutes
    ):
        return (
            "close_time_expired",
            f"max extension reached: held {minutes_held:.0f} min (max {policy.max_extension_minutes} min)",
            {
                "current_price": current_price,
                "minutes_held": minutes_held,
                "max_extension_minutes": policy.max_extension_minutes,
                "direction": direction,
                "entry_price": entry_price,
            },
        )

    # --- Step 3: Check for positive indicators ---
    # First verify price is above stop (not breached, already checked, but confirm
    # price is on the safe side of the stop for the hold decision)
    price_above_stop = True  # Already passed breach check above
    if direction == "LONG" and stop_price is not None:
        price_above_stop = current_price > stop_price
    elif direction == "SHORT" and stop_price is not None:
        price_above_stop = current_price < stop_price

    # Compute target_progress with division-by-zero protection
    target_progress = 0.0
    if target_price is not None and entry_price != 0.0:
        if direction == "LONG":
            denominator = target_price - entry_price
            if denominator != 0.0:
                target_progress = (current_price - entry_price) / denominator * 100
        else:
            denominator = entry_price - target_price
            if denominator != 0.0:
                target_progress = (entry_price - current_price) / denominator * 100

    # Check positive indicators
    positive_indicators = []

    if direction == "LONG":
        if current_price >= entry_price:
            positive_indicators.append("price_at_or_above_entry")
        if vwap_level is not None and current_price >= vwap_level:
            positive_indicators.append("price_at_or_above_vwap")
        if support_level is not None and current_price >= support_level:
            positive_indicators.append("price_at_or_above_support")
    else:
        # SHORT direction
        if current_price <= entry_price:
            positive_indicators.append("price_at_or_below_entry")
        if resistance_level is not None and current_price <= resistance_level:
            positive_indicators.append("price_at_or_below_resistance")
        if vwap_level is not None and current_price <= vwap_level:
            positive_indicators.append("price_at_or_below_vwap")

    if target_progress >= 25.0:
        positive_indicators.append("target_progress_above_25pct")

    if price_above_stop and len(positive_indicators) > 0:
        return (
            "hold_valid_until_next_window",
            f"positive indicators: {', '.join(positive_indicators)}",
            {
                "current_price": current_price,
                "entry_price": entry_price,
                "target_progress": round(target_progress, 2),
                "positive_indicators": positive_indicators,
                "direction": direction,
                "minutes_held": minutes_held,
            },
        )

    # --- Step 4: No positive indicator found → close_time_expired ---
    return (
        "close_time_expired",
        "no positive indicator found",
        {
            "current_price": current_price,
            "entry_price": entry_price,
            "target_progress": round(target_progress, 2),
            "direction": direction,
            "minutes_held": minutes_held,
            "price_above_stop": price_above_stop,
        },
    )


def _check_market_data_freshness(
    current_price: float | None,
    market_data_timestamp: datetime | None,
    now_utc: datetime,
    max_staleness_seconds: int | None = None,
) -> tuple[bool, str]:
    """Check if market data is fresh enough for revalidation.

    Args:
        current_price: Latest market price (None if unavailable).
        market_data_timestamp: When current_price was last updated (None if unavailable).
        now_utc: Current UTC time for staleness comparison.
        max_staleness_seconds: Override threshold. If None, reads from
            SETUP_AWARE_MAX_MARKET_DATA_STALENESS_SECONDS env var (default 30).

    Returns:
        (is_fresh, reason)
        - is_fresh: True if data is available and within staleness threshold
        - reason: Human-readable explanation of freshness status
    """
    if current_price is None:
        return (False, "current_price is None")
    if market_data_timestamp is None:
        return (False, "market_data_timestamp is None")

    if max_staleness_seconds is None:
        max_staleness_seconds = int(
            os.environ.get("SETUP_AWARE_MAX_MARKET_DATA_STALENESS_SECONDS", "30")
        )

    now_aware = now_utc if now_utc.tzinfo is not None else now_utc.replace(tzinfo=timezone.utc)
    ts_aware = (
        market_data_timestamp
        if market_data_timestamp.tzinfo is not None
        else market_data_timestamp.replace(tzinfo=timezone.utc)
    )
    staleness = (now_aware - ts_aware).total_seconds()

    if staleness > max_staleness_seconds:
        return (False, f"market data stale by {staleness:.0f}s (max {max_staleness_seconds}s)")

    return (True, "market data fresh")


def _compute_next_revalidation_boundary(
    entry_time: datetime,
    policy: SetupTimePolicy,
    minutes_held: float,
) -> float | None:
    """Compute the next revalidation boundary in minutes from entry.

    For extension-eligible trades, the boundary sequence is:
    - First boundary: revalidate_minutes (e.g., 90 for news_breakout)
    - Second boundary: force_close_minutes (e.g., 120 for news_breakout)
      — this is NOT a hard close for extension-eligible trades
    - Subsequent boundaries: every revalidation_interval_minutes after
      force_close_minutes (e.g., 150, 180 for news_breakout)
    - Stops at max_extension_minutes

    Returns the NEXT boundary strictly greater than minutes_held.
    Returns None if not extension-eligible or past max_extension_minutes.

    Example for news_breakout (revalidate=90, force_close=120, interval=30, max=180):
        Boundaries: [90, 120, 150, 180]
        minutes_held=85  → 90
        minutes_held=91  → 120
        minutes_held=121 → 150
        minutes_held=151 → 180
        minutes_held=181 → None
    """
    if not policy.extension_eligible:
        return None
    if policy.revalidate_minutes is None or policy.revalidation_interval_minutes is None:
        return None
    if policy.max_extension_minutes is not None and minutes_held >= policy.max_extension_minutes:
        return None

    # Build the boundary sequence:
    # 1. revalidate_minutes
    # 2. force_close_minutes
    # 3. force_close_minutes + N * revalidation_interval_minutes (until max_extension)
    boundaries: list[float] = [policy.revalidate_minutes]

    # Add force_close_minutes as second boundary if it's distinct from revalidate_minutes
    if policy.force_close_minutes > policy.revalidate_minutes:
        boundaries.append(policy.force_close_minutes)

    # Add subsequent boundaries after force_close_minutes at interval cadence
    next_after_force = policy.force_close_minutes + policy.revalidation_interval_minutes
    while True:
        if policy.max_extension_minutes is not None and next_after_force > policy.max_extension_minutes:
            break
        # Only add if not already in the list (avoid duplicates)
        if next_after_force not in boundaries:
            boundaries.append(next_after_force)
        next_after_force += policy.revalidation_interval_minutes

    # Include max_extension_minutes as the final boundary if not already present
    if (
        policy.max_extension_minutes is not None
        and policy.max_extension_minutes not in boundaries
    ):
        boundaries.append(policy.max_extension_minutes)

    # Find the first boundary strictly greater than minutes_held
    for boundary in boundaries:
        if boundary > minutes_held:
            return boundary

    return None


def _is_at_revalidation_boundary(
    minutes_held: float,
    entry_time: datetime,
    policy: SetupTimePolicy,
    tolerance_minutes: float = 1.0,
) -> bool:
    """Determine if the trade is at or past a revalidation boundary.

    A trade is "at" a boundary if minutes_held >= the first boundary in the
    revalidation sequence (revalidate_minutes). Once past the first boundary,
    the trade is considered at a revalidation boundary for the evaluator to
    produce a decisive outcome.

    The main evaluator handles max_extension checks separately, so this
    function simply returns True if minutes_held >= revalidate_minutes
    for extension-eligible trades.

    Args:
        minutes_held: Minutes since trade entry.
        entry_time: Trade entry datetime.
        policy: The setup time policy for this trade's setup type.
        tolerance_minutes: Not used in current implementation but reserved
            for future boundary-proximity checks.

    Returns:
        True if the trade has reached or passed the first revalidation boundary.
    """
    if not policy.extension_eligible:
        return False
    if policy.revalidate_minutes is None:
        return False

    # A trade is at a revalidation boundary if it has reached or passed
    # the first revalidation point (revalidate_minutes)
    return minutes_held >= policy.revalidate_minutes


# ---------------------------------------------------------------------------
# Internal Helper: Parse datetime from trade fields
# ---------------------------------------------------------------------------


def _parse_dt(value: Any) -> datetime | None:
    """Parse a datetime value from trade dict (handles str and datetime)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return None
    return None


# ---------------------------------------------------------------------------
# Internal Helper: Build LifecycleDecision dict
# ---------------------------------------------------------------------------


def _build_decision(
    trade: dict,
    now_utc: datetime,
    *,
    decision: str,
    state: str,
    reason_type: str,
    requires_event: bool,
    close_reason: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Construct a LifecycleDecision result dict matching existing schema."""
    entry_time = _parse_dt(trade.get("entry_time"))
    if entry_time is not None:
        now_aware = now_utc if now_utc.tzinfo is not None else now_utc.replace(tzinfo=timezone.utc)
        entry_aware = entry_time if entry_time.tzinfo is not None else entry_time.replace(tzinfo=timezone.utc)
        hours_held = (now_aware - entry_aware).total_seconds() / 3600
    else:
        hours_held = 0.0

    return {
        "decision": decision,
        "state": state,
        "reason_type": reason_type,
        "trade_id": trade.get("id"),
        "symbol": trade.get("symbol", ""),
        "profile": trade.get("profile", ""),
        "setup_type": trade.get("setup_type", ""),
        "entry_time": entry_time,
        "hours_held": hours_held,
        "requires_event": requires_event,
        "close_reason": close_reason,
        "metadata": metadata or {},
    }


# ---------------------------------------------------------------------------
# Internal Helper: Compute legacy decision for shadow mode
# ---------------------------------------------------------------------------


def _compute_legacy_decision(
    trade: dict,
    now_utc: datetime,
    minutes_held: float,
) -> dict:
    """Compute what the legacy generic 90-minute force-close policy would do.

    Used in shadow mode to compare setup-aware decisions against legacy behavior.
    Legacy policy: alert at 60 min, force close at 90 min for all setups.
    """
    legacy_force_close = 90
    legacy_alert = 60

    if minutes_held > legacy_force_close:
        return {
            "decision": "close",
            "state": "intraday_expired",
            "reason_type": "intraday_time_limit_exceeded",
            "minutes_held": round(minutes_held),
            "force_close_limit": legacy_force_close,
        }
    elif minutes_held > legacy_alert:
        return {
            "decision": "warn",
            "state": "intraday_warning",
            "reason_type": "intraday_time_warning",
            "minutes_held": round(minutes_held),
            "alert_limit": legacy_alert,
            "force_limit": legacy_force_close,
        }
    else:
        return {
            "decision": "hold",
            "state": "intraday_ok",
            "reason_type": "within_time_limits",
            "minutes_held": round(minutes_held),
        }


# ---------------------------------------------------------------------------
# Public API: Setup-Aware Lifecycle Evaluator
# ---------------------------------------------------------------------------


def evaluate_setup_aware_lifecycle(
    trade: dict,
    events: list[dict],
    *,
    now_utc: datetime,
    now_et: datetime,
    current_price: float | None = None,
    market_data_timestamp: datetime | None = None,
    shadow_mode: bool = False,
) -> dict:
    """Deterministic setup-aware lifecycle evaluator.

    Pure function — no DB writes, no side effects.

    Replaces _check_intraday_force_close and _check_intraday_warning
    in the priority chain. Called only when priorities 1-5 have not
    produced a decisive result.

    Args:
        trade: Dict with keys: id, symbol, profile, direction, entry_price,
               entry_time, stop_price, target_price, setup_type, status,
               quantity, thesis, invalidators.
        events: Pre-fetched trade events list.
        now_utc: Current UTC time.
        now_et: Current US/Eastern time (for EOD pre-wall buffer).
        current_price: Latest market price (None if unavailable).
        market_data_timestamp: When current_price was last updated.
        shadow_mode: If True, returns decision but marks it as shadow-only.

    Returns:
        LifecycleDecision dict (same schema as existing evaluator).
        When shadow_mode=True, metadata includes "shadow_mode": True and
        "legacy_decision" with what the old policy would have produced.
    """
    # --- Setup: parse entry time and get policy ---
    setup_type = trade.get("setup_type", "") or ""
    policy = get_policy(setup_type)

    entry_time = _parse_dt(trade.get("entry_time"))
    if entry_time is None:
        # Cannot evaluate without entry time — return intraday_ok (no action)
        return _build_decision(
            trade,
            now_utc,
            decision="hold",
            state="intraday_ok",
            reason_type="missing_entry_time",
            requires_event=False,
        )

    # Compute minutes held
    now_aware = now_utc if now_utc.tzinfo is not None else now_utc.replace(tzinfo=timezone.utc)
    entry_aware = entry_time if entry_time.tzinfo is not None else entry_time.replace(tzinfo=timezone.utc)
    minutes_held = (now_aware - entry_aware).total_seconds() / 60

    # --- EOD Pre-Wall Buffer Check (15 minutes before eod_hard_wall) ---
    # If now_et time is within 15 minutes of eod_hard_wall, revoke any extension
    # and produce close decision with reason setup_pre_wall_buffer_close.
    eod_hard_wall = policy.eod_hard_wall
    now_et_time = now_et.time()
    # Compute the pre-wall buffer threshold: eod_hard_wall minus 15 minutes
    wall_dt = datetime.combine(now_et.date(), eod_hard_wall)
    buffer_dt = wall_dt - timedelta(minutes=PRE_WALL_BUFFER_MINUTES)
    buffer_time = buffer_dt.time()

    if now_et_time >= buffer_time:
        result = _build_decision(
            trade,
            now_utc,
            decision="close",
            state="setup_time_limit_exceeded",
            reason_type="setup_pre_wall_buffer_close",
            requires_event=True,
            close_reason=(
                f"EOD pre-wall buffer: {setup_type or 'unknown'} within "
                f"{PRE_WALL_BUFFER_MINUTES} min of hard wall "
                f"({eod_hard_wall.strftime('%H:%M')} ET), "
                f"held {round(minutes_held)} min"
            ),
            metadata={
                "minutes_held": round(minutes_held),
                "setup_type": setup_type,
                "eod_hard_wall": eod_hard_wall.isoformat(),
                "pre_wall_buffer_minutes": PRE_WALL_BUFFER_MINUTES,
                "extension_revoked": True,
            },
        )
        if shadow_mode:
            result["metadata"]["shadow_mode"] = True
            result["metadata"]["legacy_decision"] = _compute_legacy_decision(
                trade, now_utc, minutes_held
            )
        return result

    # --- Below alert threshold → intraday_ok ---
    if minutes_held < policy.alert_minutes:
        result = _build_decision(
            trade,
            now_utc,
            decision="hold",
            state="intraday_ok",
            reason_type="within_time_limits",
            requires_event=False,
            metadata={
                "minutes_held": round(minutes_held),
                "alert_minutes": policy.alert_minutes,
                "setup_type": setup_type,
            },
        )
        if shadow_mode:
            result["metadata"]["shadow_mode"] = True
            result["metadata"]["legacy_decision"] = _compute_legacy_decision(
                trade, now_utc, minutes_held
            )
        return result

    # --- Past alert, determine if at revalidation boundary ---
    if policy.extension_eligible:
        # Extension-eligible path: check if at revalidation boundary
        at_boundary = _is_at_revalidation_boundary(
            minutes_held, entry_aware, policy
        )

        if at_boundary:
            # Check max extension first
            if (
                policy.max_extension_minutes is not None
                and minutes_held >= policy.max_extension_minutes
            ):
                result = _build_decision(
                    trade,
                    now_utc,
                    decision="close",
                    state="setup_time_limit_exceeded",
                    reason_type="setup_max_extension_reached",
                    requires_event=True,
                    close_reason=(
                        f"Max extension reached: {setup_type} held {round(minutes_held)} min "
                        f"(max: {policy.max_extension_minutes} min)"
                    ),
                    metadata={
                        "minutes_held": round(minutes_held),
                        "setup_type": setup_type,
                        "max_extension_minutes": policy.max_extension_minutes,
                        "force_close_minutes": policy.force_close_minutes,
                    },
                )
                if shadow_mode:
                    result["metadata"]["shadow_mode"] = True
                    result["metadata"]["legacy_decision"] = _compute_legacy_decision(
                        trade, now_utc, minutes_held
                    )
                return result

            # Validate invalidation criteria
            criteria_valid, criteria_reason, criteria_used = _validate_invalidation_criteria(
                trade, policy
            )

            if not criteria_valid:
                # Deny extension — close at base force-close limit
                result = _build_decision(
                    trade,
                    now_utc,
                    decision="close",
                    state="setup_revalidation_failed",
                    reason_type="setup_revalidation_denied_missing_criteria",
                    requires_event=True,
                    close_reason=(
                        f"Extension denied (missing criteria): {setup_type} "
                        f"held {round(minutes_held)} min — {criteria_reason}"
                    ),
                    metadata={
                        "minutes_held": round(minutes_held),
                        "setup_type": setup_type,
                        "force_close_minutes": policy.force_close_minutes,
                        "criteria_reason": criteria_reason,
                        "revalidation_attempted": True,
                    },
                )
                if shadow_mode:
                    result["metadata"]["shadow_mode"] = True
                    result["metadata"]["legacy_decision"] = _compute_legacy_decision(
                        trade, now_utc, minutes_held
                    )
                return result

            # Check market data freshness
            is_fresh, freshness_reason = _check_market_data_freshness(
                current_price, market_data_timestamp, now_utc
            )

            if not is_fresh:
                # Fail closed — deny extension due to stale data
                result = _build_decision(
                    trade,
                    now_utc,
                    decision="close",
                    state="setup_revalidation_failed",
                    reason_type="setup_revalidation_denied_stale_data",
                    requires_event=True,
                    close_reason=(
                        f"Extension denied (stale data): {setup_type} "
                        f"held {round(minutes_held)} min — {freshness_reason}"
                    ),
                    metadata={
                        "minutes_held": round(minutes_held),
                        "setup_type": setup_type,
                        "force_close_minutes": policy.force_close_minutes,
                        "freshness_reason": freshness_reason,
                        "revalidation_attempted": True,
                    },
                )
                if shadow_mode:
                    result["metadata"]["shadow_mode"] = True
                    result["metadata"]["legacy_decision"] = _compute_legacy_decision(
                        trade, now_utc, minutes_held
                    )
                return result

            # Run revalidation decision engine
            decision_outcome, decision_reason, decision_meta = _evaluate_revalidation_decision(
                trade, policy, current_price, minutes_held, criteria_used
            )

            # Map revalidation outcome to lifecycle decision
            if decision_outcome == "hold_valid_until_next_window":
                next_boundary = _compute_next_revalidation_boundary(
                    entry_aware, policy, minutes_held
                )
                result = _build_decision(
                    trade,
                    now_utc,
                    decision="hold",
                    state="setup_revalidation_hold",
                    reason_type="setup_revalidation_hold_valid",
                    requires_event=True,
                    metadata={
                        "minutes_held": round(minutes_held),
                        "setup_type": setup_type,
                        "decision_outcome": decision_outcome,
                        "decision_reason": decision_reason,
                        "next_revalidation_boundary": next_boundary,
                        "force_close_minutes": policy.force_close_minutes,
                        "max_extension_minutes": policy.max_extension_minutes,
                        "invalidation_criteria_used": criteria_used,
                        "revalidation_attempted": True,
                        **decision_meta,
                    },
                )
            elif decision_outcome == "close_thesis_invalidated":
                result = _build_decision(
                    trade,
                    now_utc,
                    decision="close",
                    state="setup_thesis_invalidated",
                    reason_type="setup_thesis_invalidated",
                    requires_event=True,
                    close_reason=(
                        f"Thesis invalidated: {setup_type} held {round(minutes_held)} min "
                        f"— {decision_reason}"
                    ),
                    metadata={
                        "minutes_held": round(minutes_held),
                        "setup_type": setup_type,
                        "decision_outcome": decision_outcome,
                        "decision_reason": decision_reason,
                        "invalidation_criteria_used": criteria_used,
                        "revalidation_attempted": True,
                        **decision_meta,
                    },
                )
            else:
                # close_time_expired or insufficient_data_close_at_base_limit
                result = _build_decision(
                    trade,
                    now_utc,
                    decision="close",
                    state="setup_time_limit_exceeded",
                    reason_type="setup_time_limit_exceeded",
                    requires_event=True,
                    close_reason=(
                        f"Time limit exceeded: {setup_type} held {round(minutes_held)} min "
                        f"— {decision_reason}"
                    ),
                    metadata={
                        "minutes_held": round(minutes_held),
                        "setup_type": setup_type,
                        "decision_outcome": decision_outcome,
                        "decision_reason": decision_reason,
                        "force_close_minutes": policy.force_close_minutes,
                        "invalidation_criteria_used": criteria_used,
                        "revalidation_attempted": True,
                        **decision_meta,
                    },
                )

            if shadow_mode:
                result["metadata"]["shadow_mode"] = True
                result["metadata"]["legacy_decision"] = _compute_legacy_decision(
                    trade, now_utc, minutes_held
                )
            return result

        else:
            # Past alert but not yet at revalidation boundary → alert state
            result = _build_decision(
                trade,
                now_utc,
                decision="hold",
                state="setup_exit_alert",
                reason_type="setup_alert_approaching_revalidation",
                requires_event=True,
                metadata={
                    "minutes_held": round(minutes_held),
                    "setup_type": setup_type,
                    "alert_minutes": policy.alert_minutes,
                    "revalidate_minutes": policy.revalidate_minutes,
                    "force_close_minutes": policy.force_close_minutes,
                },
            )
            if shadow_mode:
                result["metadata"]["shadow_mode"] = True
                result["metadata"]["legacy_decision"] = _compute_legacy_decision(
                    trade, now_utc, minutes_held
                )
            return result

    else:
        # --- Non-extension-eligible path ---
        # Close at force_close_minutes with setup_time_limit_exceeded
        if minutes_held >= policy.force_close_minutes:
            result = _build_decision(
                trade,
                now_utc,
                decision="close",
                state="setup_time_limit_exceeded",
                reason_type="setup_time_limit_exceeded",
                requires_event=True,
                close_reason=(
                    f"Setup time limit exceeded: {setup_type or 'unknown'} "
                    f"held {round(minutes_held)} min "
                    f"(limit: {policy.force_close_minutes} min)"
                ),
                metadata={
                    "minutes_held": round(minutes_held),
                    "setup_type": setup_type,
                    "force_close_minutes": policy.force_close_minutes,
                    "extension_eligible": False,
                    "revalidation_attempted": False,
                },
            )
            if shadow_mode:
                result["metadata"]["shadow_mode"] = True
                result["metadata"]["legacy_decision"] = _compute_legacy_decision(
                    trade, now_utc, minutes_held
                )
            return result

        # Past alert but before force close → alert state
        result = _build_decision(
            trade,
            now_utc,
            decision="hold",
            state="setup_exit_alert",
            reason_type="setup_alert_approaching_revalidation",
            requires_event=True,
            metadata={
                "minutes_held": round(minutes_held),
                "setup_type": setup_type,
                "alert_minutes": policy.alert_minutes,
                "force_close_minutes": policy.force_close_minutes,
                "extension_eligible": False,
            },
        )
        if shadow_mode:
            result["metadata"]["shadow_mode"] = True
            result["metadata"]["legacy_decision"] = _compute_legacy_decision(
                trade, now_utc, minutes_held
            )
        return result
