"""
Case-Memory Exit Classification — Classifies closed trades into exactly one
exit category for case-memory learning and review.

Each closed trade is assigned one of four discrete categories based on its
lifecycle events. This classification is stored in the case-memory update
process and used by the reviewer, daily review, and CEO memo to distinguish
genuine entry failures from policy-induced exits.

Requirements: 6.6
"""

from utils.setup_time_policy import is_thesis_development_setup

# Classification categories — exactly one per closed trade
CASE_MEMORY_EXIT_CATEGORIES = {
    "bad_entry",                          # Entry quality was poor
    "valid_entry_bad_exit_policy",        # Good entry, timer killed it prematurely
    "valid_exit_thesis_invalidated",      # Correct exit by thesis invalidation
    "forced_exit_missing_metadata",       # Extension denied due to missing data
}


def classify_trade_exit(
    trade: dict,
    events: list[dict],
) -> str:
    """Classify a closed trade into exactly one exit category.

    Args:
        trade: The closed trade dict with keys: id, symbol, setup_type, direction,
               entry_price, stop_price, exit_price, pnl, status, etc.
        events: List of trade_events for this trade (dicts with at minimum
                an 'event_type' key, and optionally 'reason' or 'message').

    Returns:
        One of CASE_MEMORY_EXIT_CATEGORIES.

    Classification logic (priority order):
    1. If any event has event_type == "setup_exit_thesis_invalidated"
       → "valid_exit_thesis_invalidated"
    2. If any event has event_type == "setup_exit_revalidation_failed" with
       reason/message containing "missing criteria" or "stale data"
       → "forced_exit_missing_metadata"
    3. If any event has event_type == "setup_exit_force_close" AND the trade
       was a thesis-development setup with positive price action at close
       → "valid_entry_bad_exit_policy"
    4. Otherwise → "bad_entry" (default: entry quality was the issue)
    """
    # Priority 1: Thesis invalidation — correct exit
    if _has_thesis_invalidation_event(events):
        return "valid_exit_thesis_invalidated"

    # Priority 2: Missing metadata forced exit
    if _has_missing_metadata_failure(events):
        return "forced_exit_missing_metadata"

    # Priority 3: Good entry killed by timer (bad exit policy)
    if _is_valid_entry_bad_exit_policy(trade, events):
        return "valid_entry_bad_exit_policy"

    # Priority 4: Default — entry quality was the issue
    return "bad_entry"


def _has_thesis_invalidation_event(events: list[dict]) -> bool:
    """Check if any event indicates thesis invalidation exit."""
    return any(
        e.get("event_type") == "setup_exit_thesis_invalidated"
        for e in events
    )


def _has_missing_metadata_failure(events: list[dict]) -> bool:
    """Check if any revalidation failure was due to missing/stale data."""
    for e in events:
        if e.get("event_type") != "setup_exit_revalidation_failed":
            continue
        # Check reason or message for missing criteria / stale data indicators
        reason_text = _get_event_reason_text(e).lower()
        if "missing criteria" in reason_text or "stale data" in reason_text:
            return True
    return False


def _is_valid_entry_bad_exit_policy(trade: dict, events: list[dict]) -> bool:
    """Check if a thesis-development trade was profitable but force-closed by timer.

    Conditions:
    - Trade has a setup_exit_force_close event
    - Trade is a thesis-development setup (news_breakout, news_catalyst, trend_pullback)
    - Trade had positive price action at close (profitable direction)
    """
    has_force_close = any(
        e.get("event_type") == "setup_exit_force_close"
        for e in events
    )
    if not has_force_close:
        return False

    setup_type = trade.get("setup_type", "")
    if not is_thesis_development_setup(setup_type):
        return False

    return _had_positive_price_action(trade)


def _had_positive_price_action(trade: dict) -> bool:
    """Check if exit_price was on the profitable side of entry_price.

    For LONG: exit_price > entry_price
    For SHORT: exit_price < entry_price
    """
    entry_price = trade.get("entry_price")
    exit_price = trade.get("exit_price")

    if entry_price is None or exit_price is None:
        return False

    try:
        entry_price = float(entry_price)
        exit_price = float(exit_price)
    except (TypeError, ValueError):
        return False

    if entry_price <= 0:
        return False

    direction = (trade.get("direction") or "").upper()
    if direction == "SHORT":
        return exit_price < entry_price
    # Default to LONG
    return exit_price > entry_price


def _get_event_reason_text(event: dict) -> str:
    """Extract reason text from an event, checking multiple possible fields."""
    # Check 'reason' field directly
    reason = event.get("reason", "")
    if reason:
        return str(reason)

    # Check 'message' field
    message = event.get("message", "")
    if message:
        return str(message)

    # Check payload for reason
    payload = event.get("payload") or event.get("payload_json")
    if isinstance(payload, dict):
        return str(payload.get("reason", "") or payload.get("close_reason", ""))
    if isinstance(payload, str):
        try:
            import json
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                return str(parsed.get("reason", "") or parsed.get("close_reason", ""))
        except (json.JSONDecodeError, TypeError):
            pass

    return ""
