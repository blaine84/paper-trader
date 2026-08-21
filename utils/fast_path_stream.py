"""Fast-path public stream event generation and template narration.

Transforms fast_path_events into structured events suitable for an
audience-facing AI trading desk stream.  Provides deterministic template
narration (always available, no LLM needed) and a structured event builder
that includes LLM-enriched narration when annotation is present.

Fail mode: fail-open — stream/narration errors never block or invalidate
fast-path outcomes.  All functions return sensible defaults on error.

See: .kiro/specs/fast-path-deterministic-execution/design.md
Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 9.6
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 10.1 — Deterministic template narration
# (Requirements: 8.1, 8.2, 8.3, 8.4)
# ---------------------------------------------------------------------------


def generate_template_narration(outcome: Any) -> str:
    """Generate a deterministic plain-English narration from outcome fields.

    Produces a template-based narration for each of the six fast-path outcome
    types.  The narration is educational and experimental, never framed as
    financial advice or copy-trade instructions.

    Accepts either a FastPathOutcome dataclass or a dict with the relevant
    fields (outcome_type, symbol, entry_price, stop_price, target_price,
    current_price, reward_to_risk, outcome_reason_code, trigger_level or
    entry_price as level proxy).

    Args:
        outcome: A FastPathOutcome instance or dict with outcome fields.

    Returns:
        A plain-English sentence describing what happened and why.
        Returns a generic fallback string if outcome_type is unrecognized
        or required fields are missing.
    """
    try:
        # Normalize access — support both dataclass attrs and dict keys
        if isinstance(outcome, dict):
            get = outcome.get
        else:
            get = lambda key, default=None: getattr(outcome, key, default)

        outcome_type = get("outcome_type", "")
        symbol = get("symbol", "UNKNOWN")
        entry_price = get("entry_price")
        target_price = get("target_price")
        current_price = get("current_price")
        reward_to_risk = get("reward_to_risk")
        reason_code = get("outcome_reason_code", "")

        # Use entry_price as the primary "level" reference; fall back to
        # trigger_level if available (for watch_created context).
        level = entry_price or get("trigger_level") or ""

        if outcome_type == "missed_move":
            level_str = _format_price(level)
            return (
                f"{symbol} broke {level_str}, but the move had already "
                f"crossed target; no order created."
            )

        elif outcome_type == "watch_created":
            level_str = _format_price(level)
            return (
                f"{symbol} is approaching {level_str}; watch created, "
                f"waiting for confirmation."
            )

        elif outcome_type == "pending_order_created":
            entry_str = _format_price(entry_price)
            return (
                f"{symbol} ran past intended entry but target remains ahead; "
                f"pending limit order resting at {entry_str}."
            )

        elif outcome_type == "trade_executed":
            price_str = _format_price(current_price or entry_price)
            rr_str = _format_rr(reward_to_risk)
            return (
                f"{symbol} setup triggered at {price_str}; trade executed "
                f"with {rr_str} reward/risk."
            )

        elif outcome_type == "stand_down":
            reason = _humanize_reason(reason_code)
            return f"{symbol} setup blocked by {reason}."

        elif outcome_type == "watch_promoted":
            return (
                f"{symbol} watched setup matured; evaluating for entry."
            )

        else:
            # Unrecognized outcome type — generic fallback
            return f"{symbol}: {reason_code or outcome_type}."

    except Exception as exc:
        # Fail-open: narration errors never propagate
        logger.warning(
            "fast_path_stream: narration generation failed: %s", exc
        )
        symbol = ""
        try:
            symbol = outcome.get("symbol", "") if isinstance(outcome, dict) else getattr(outcome, "symbol", "")
        except Exception:
            pass
        return f"{symbol or 'UNKNOWN'} fast-path event recorded."


# ---------------------------------------------------------------------------
# 10.2 — Structured stream event builder
# (Requirements: 8.1, 8.5)
# ---------------------------------------------------------------------------


def build_stream_event(event_row: dict) -> dict:
    """Build a structured stream event dict from a fast_path_events row.

    Produces a clean event dict with all fields required by Requirement 8.1:
    symbol, outcome_type, outcome_reason_code, entry_price, stop_price,
    target_price, current_price, timestamp, profile_id, setup_type, narration.

    The narration field is populated from the event row's existing narration
    if available.  If the annotation_status indicates LLM enrichment is present,
    the LLM-enriched narration is preferred.  Otherwise, template narration is
    regenerated from the event fields.

    Args:
        event_row: A dict representing a row from fast_path_events
                   (column names as keys).

    Returns:
        A structured dict suitable for downstream consumers (dashboard,
        websocket, log aggregator).  Returns a minimal dict with available
        fields on error — never raises.
    """
    try:
        # Determine best narration source
        narration = _resolve_narration(event_row)

        return {
            "symbol": event_row.get("symbol", ""),
            "outcome_type": event_row.get("outcome_type", ""),
            "outcome_reason_code": event_row.get("outcome_reason_code", ""),
            "entry_price": event_row.get("entry_price"),
            "stop_price": event_row.get("stop_price"),
            "target_price": event_row.get("target_price"),
            "current_price": event_row.get("current_price"),
            "timestamp": event_row.get("evaluated_at", ""),
            "profile_id": event_row.get("profile_id", ""),
            "setup_type": event_row.get("setup_type", ""),
            "narration": narration,
        }

    except Exception as exc:
        # Fail-open: return whatever we can
        logger.warning(
            "fast_path_stream: build_stream_event failed: %s", exc
        )
        return {
            "symbol": event_row.get("symbol", "") if isinstance(event_row, dict) else "",
            "outcome_type": event_row.get("outcome_type", "") if isinstance(event_row, dict) else "",
            "outcome_reason_code": "",
            "entry_price": None,
            "stop_price": None,
            "target_price": None,
            "current_price": None,
            "timestamp": "",
            "profile_id": "",
            "setup_type": "",
            "narration": "Event details unavailable.",
        }


# ---------------------------------------------------------------------------
# 10.3 — Per-cycle summary event generation
# (Requirement: 9.6)
# ---------------------------------------------------------------------------


def generate_cycle_summary(engine: Any, cycle_start: str, cycle_end: str) -> dict:
    """Generate a per-cycle summary counting events by outcome_type.

    Queries fast_path_events within the time window [cycle_start, cycle_end]
    and fast_path_triggers for trigger lifecycle counts.

    Args:
        engine: SQLAlchemy engine for database access.
        cycle_start: ISO UTC timestamp string — start of the cycle window.
        cycle_end: ISO UTC timestamp string — end of the cycle window.

    Returns:
        A dict with:
          - outcome_counts: dict mapping outcome_type → count
          - total_events: total fast-path events in the window
          - total_triggers_evaluated: triggers that were evaluated (fired)
          - total_triggers_fired: triggers that reached FIRED state
          - total_triggers_expired: triggers that expired in the window
          - cycle_start: echo of the input start
          - cycle_end: echo of the input end
        Returns a zeroed-out summary dict on error — never raises.
    """
    empty_summary = {
        "outcome_counts": {},
        "total_events": 0,
        "total_triggers_evaluated": 0,
        "total_triggers_fired": 0,
        "total_triggers_expired": 0,
        "cycle_start": cycle_start,
        "cycle_end": cycle_end,
    }

    try:
        with engine.connect() as conn:
            # Count events by outcome_type within the time window
            rows = conn.execute(
                text(
                    """
                    SELECT outcome_type, COUNT(*) as cnt
                    FROM fast_path_events
                    WHERE evaluated_at >= :cycle_start
                      AND evaluated_at <= :cycle_end
                    GROUP BY outcome_type
                    """
                ),
                {"cycle_start": cycle_start, "cycle_end": cycle_end},
            ).fetchall()

            outcome_counts: dict[str, int] = {}
            total_events = 0
            for row in rows:
                otype = row[0]
                count = row[1]
                outcome_counts[otype] = count
                total_events += count

            # Count triggers that fired within the window
            fired_row = conn.execute(
                text(
                    """
                    SELECT COUNT(*) FROM fast_path_triggers
                    WHERE fired_at >= :cycle_start
                      AND fired_at <= :cycle_end
                    """
                ),
                {"cycle_start": cycle_start, "cycle_end": cycle_end},
            ).fetchone()
            total_fired = fired_row[0] if fired_row else 0

            # Count triggers that expired within the window
            # (resolved_at is set when expired — but state column is 'expired')
            expired_row = conn.execute(
                text(
                    """
                    SELECT COUNT(*) FROM fast_path_triggers
                    WHERE state = 'expired'
                      AND expires_at >= :cycle_start
                      AND expires_at <= :cycle_end
                    """
                ),
                {"cycle_start": cycle_start, "cycle_end": cycle_end},
            ).fetchone()
            total_expired = expired_row[0] if expired_row else 0

            # Total triggers evaluated = fired + expired (all that reached
            # a terminal state during this window). This is an approximation;
            # a trigger is "evaluated" every tick it is active, but the
            # meaningful count is those that resolved.
            total_evaluated = total_fired + total_expired

            return {
                "outcome_counts": outcome_counts,
                "total_events": total_events,
                "total_triggers_evaluated": total_evaluated,
                "total_triggers_fired": total_fired,
                "total_triggers_expired": total_expired,
                "cycle_start": cycle_start,
                "cycle_end": cycle_end,
            }

    except Exception as exc:
        # Fail-open: summary generation errors never propagate
        logger.warning(
            "fast_path_stream: generate_cycle_summary failed: %s", exc
        )
        return empty_summary


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _format_price(price: Any) -> str:
    """Format a price value for narration display."""
    if price is None:
        return "N/A"
    try:
        return f"{float(price):.2f}"
    except (TypeError, ValueError):
        return str(price)


def _format_rr(rr: Any) -> str:
    """Format reward-to-risk ratio for narration display."""
    if rr is None:
        return "N/A"
    try:
        return f"{float(rr):.1f}:1"
    except (TypeError, ValueError):
        return str(rr)


def _humanize_reason(reason_code: str) -> str:
    """Convert an outcome_reason_code to a human-readable fragment.

    Replaces underscores with spaces and strips known prefixes like
    'cooldown:' or 'exposure:' into readable phrases.
    """
    if not reason_code:
        return "an unspecified rule"

    # Strip prefixes and convert underscores
    for prefix in ("cooldown:", "exposure:", "gate_rejected:"):
        if reason_code.startswith(prefix):
            remainder = reason_code[len(prefix):]
            category = prefix.rstrip(":")
            return f"{category} — {remainder.replace('_', ' ')}"

    return reason_code.replace("_", " ")


def _resolve_narration(event_row: dict) -> str:
    """Determine the best narration for a stream event.

    Prefers LLM-enriched narration when annotation is complete.
    Falls back to stored template narration or regenerates from fields.
    """
    annotation_status = event_row.get("annotation_status", "")
    narration_source = event_row.get("narration_source", "")

    # If LLM enrichment is available and annotation succeeded, use stored narration
    # (it was already updated by the annotation system)
    if annotation_status == "annotated" and narration_source == "llm_enriched":
        stored = event_row.get("narration")
        if stored:
            return stored

    # If template narration is stored, use it
    stored = event_row.get("narration")
    if stored:
        return stored

    # Regenerate template narration from event fields
    return generate_template_narration(event_row)
