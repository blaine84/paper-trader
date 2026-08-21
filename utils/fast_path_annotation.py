"""Fast-path LLM annotation system.

Provides asynchronous, non-blocking annotation of fast-path events by LLM
agents.  The annotation layer is strictly fail-open: annotation failures are
logged but never modify, invalidate, or block a fast-path outcome.

Key functions:

- ``get_unannotated_events`` — fetch events awaiting PM annotation.
- ``annotate_event`` — write PM annotation payload to an event.
- ``process_pm_veto`` — PM structured veto on pending_order or watch outcomes.

The PM agent calls these during its cycle to enrich events with thesis context,
narration, and (optionally) vetoes.  Vetoes are the ONE path where LLM judgment
overrides a deterministic outcome — limited to ``pending_order_created`` and
``watch_created`` outcomes that have not yet been executed.

All operations use ``engine.connect()`` + ``text()`` for raw SQL against the
``fast_path_events`` table.  The immutability trigger on that table is expected
to be relaxed (task 9.5) to allow annotation-column-only updates.

See: .kiro/specs/fast-path-deterministic-execution/design.md
Requirements: 7.1, 7.2, 7.3, 7.4, 7.6, 7.7
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from sqlalchemy import text

logger = logging.getLogger(__name__)

# Outcome types eligible for PM veto.  Only outcomes with pending side-effects
# (an active pending order or an active watch) can be vetoed.
_VETOABLE_OUTCOME_TYPES: frozenset[str] = frozenset(
    {"pending_order_created", "watch_created"}
)

# Outcome types that cannot be vetoed — either already executed, purely
# informational, or represent no action.
_NON_VETOABLE_OUTCOME_TYPES: frozenset[str] = frozenset(
    {"trade_executed", "missed_move", "stand_down", "watch_promoted"}
)


# ---------------------------------------------------------------------------
# 9.1 — get_unannotated_events
# Requirements: 7.1, 7.3
# ---------------------------------------------------------------------------


def get_unannotated_events(
    engine,
    profile_id: str,
    limit: int = 20,
) -> list[dict]:
    """Fetch fast-path events awaiting LLM annotation.

    Returns events with ``annotation_status='annotation_pending'`` for the
    given profile, ordered by ``evaluated_at ASC`` (oldest first) so that
    earlier events are annotated before recent ones.

    Parameters
    ----------
    engine
        SQLAlchemy engine (or compatible connection source).
    profile_id
        Profile whose events to fetch.
    limit
        Maximum events to return (default 20).

    Returns
    -------
    list[dict]
        List of event rows as dictionaries.  Empty list on error (fail-open).
    """
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT event_id, trigger_id, symbol, profile_id, "
                    "setup_type, direction, entry_price, stop_price, "
                    "target_price, current_price, reward_to_risk, "
                    "outcome_type, outcome_reason_code, outcome_metadata_json, "
                    "narration, narration_source, evaluated_at, "
                    "annotation_status "
                    "FROM fast_path_events "
                    "WHERE annotation_status = 'annotation_pending' "
                    "  AND profile_id = :profile_id "
                    "ORDER BY evaluated_at ASC "
                    "LIMIT :limit"
                ),
                {"profile_id": profile_id, "limit": limit},
            ).mappings().all()
            return [dict(row) for row in rows]
    except Exception as exc:
        logger.error(
            "get_unannotated_events: failed to query events for profile %s: %s "
            "(fail-open, returning empty list)",
            profile_id,
            exc,
        )
        return []


# ---------------------------------------------------------------------------
# 9.2 — annotate_event
# Requirements: 7.1, 7.2, 7.4
# ---------------------------------------------------------------------------


def annotate_event(
    engine,
    event_id: str,
    annotation_json: dict | str,
) -> None:
    """Write an LLM annotation to a fast-path event.

    Updates the annotation columns on the event row:
        - ``annotation_status`` → ``'annotated'``
        - ``annotation_json`` → serialized payload
        - ``annotation_timestamp`` → current UTC ISO timestamp

    This is the ONE allowed update on ``fast_path_events`` — annotation fields
    only.  The immutability trigger must be relaxed (task 9.5) to permit this.

    The operation is fail-open: errors are logged but never raised to the
    caller.  A failed annotation does not invalidate the event outcome.

    Parameters
    ----------
    engine
        SQLAlchemy engine.
    event_id
        The ``event_id`` (UUID) of the event to annotate.
    annotation_json
        Annotation payload — either a dict (will be JSON-serialized) or a
        pre-serialized JSON string.
    """
    try:
        if isinstance(annotation_json, dict):
            payload = json.dumps(annotation_json, separators=(",", ":"))
        else:
            payload = annotation_json

        now_iso = datetime.now(timezone.utc).isoformat()

        with engine.connect() as conn:
            conn.execute(
                text(
                    "UPDATE fast_path_events "
                    "SET annotation_status = 'annotated', "
                    "    annotation_json = :payload, "
                    "    annotation_timestamp = :ts "
                    "WHERE event_id = :event_id"
                ),
                {
                    "payload": payload,
                    "ts": now_iso,
                    "event_id": event_id,
                },
            )
            conn.commit()
    except Exception as exc:
        logger.error(
            "annotate_event: failed to annotate event %s: %s "
            "(fail-open, event outcome unchanged)",
            event_id,
            exc,
        )


# ---------------------------------------------------------------------------
# 9.3 — process_pm_veto
# Requirements: 7.6, 7.7
# ---------------------------------------------------------------------------


def process_pm_veto(
    engine,
    event_id: str,
    veto_rationale: str,
) -> bool:
    """Process a PM structured veto on a fast-path outcome.

    Vetoes are the one path where LLM judgment overrides a deterministic
    outcome.  Only ``pending_order_created`` and ``watch_created`` outcomes
    may be vetoed — trades that are already executed, missed moves, and
    stand-downs cannot be retroactively overridden.

    On a valid veto:
        - If a pending order was created: cancel it via pending order registry.
        - If a watch was created: invalidate it via watch candidates.
        - Record the veto as an annotation with a ``veto`` flag in the payload.

    Parameters
    ----------
    engine
        SQLAlchemy engine.
    event_id
        The ``event_id`` (UUID) of the event being vetoed.
    veto_rationale
        Free-text rationale from the PM explaining the veto.

    Returns
    -------
    bool
        True if the veto was successfully processed, False otherwise.
        Failure is non-fatal (fail-open): the event outcome remains valid
        regardless.
    """
    try:
        # 1. Fetch the event to validate veto eligibility.
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT event_id, outcome_type, trigger_id, symbol, "
                    "profile_id, outcome_metadata_json "
                    "FROM fast_path_events "
                    "WHERE event_id = :event_id"
                ),
                {"event_id": event_id},
            ).mappings().first()

        if row is None:
            logger.warning(
                "process_pm_veto: event %s not found, cannot veto",
                event_id,
            )
            return False

        outcome_type = row["outcome_type"]

        # 2. Reject vetoes on non-vetoable outcomes.
        if outcome_type not in _VETOABLE_OUTCOME_TYPES:
            logger.warning(
                "process_pm_veto: cannot veto outcome_type=%r for event %s "
                "(only pending_order_created and watch_created are vetoable)",
                outcome_type,
                event_id,
            )
            return False

        # 3. Execute the veto side-effects.
        symbol = row["symbol"]
        trigger_id = row["trigger_id"]

        if outcome_type == "pending_order_created":
            _cancel_pending_order_for_event(engine, event_id, trigger_id, symbol)

        elif outcome_type == "watch_created":
            _invalidate_watch_for_event(engine, event_id, trigger_id, symbol)

        # 4. Record veto as annotation.
        veto_payload = json.dumps(
            {
                "veto": True,
                "veto_rationale": veto_rationale,
                "vetoed_outcome_type": outcome_type,
                "vetoed_at": datetime.now(timezone.utc).isoformat(),
            },
            separators=(",", ":"),
        )
        annotate_event(engine, event_id, veto_payload)

        logger.info(
            "process_pm_veto: veto applied to event %s (outcome_type=%s, "
            "symbol=%s, rationale=%s)",
            event_id,
            outcome_type,
            symbol,
            veto_rationale[:80],
        )
        return True

    except Exception as exc:
        logger.error(
            "process_pm_veto: failed to process veto for event %s: %s "
            "(fail-open, event outcome unchanged)",
            event_id,
            exc,
        )
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cancel_pending_order_for_event(
    engine,
    event_id: str,
    trigger_id: str,
    symbol: str,
) -> None:
    """Cancel pending orders associated with a vetoed fast-path event.

    Lazily imports from pending_order modules to avoid circular dependencies.
    Fail-open: logs errors and returns without raising.
    """
    try:
        from utils.pending_order_registry import PendingOrderRegistry

        registry = PendingOrderRegistry(engine)
        active_orders = registry.get_active_orders(symbol=symbol)

        # Find orders that were created by this fast-path trigger.
        # The trigger_id is stored in the order's stale_reason or metadata.
        cancelled_count = 0
        for order in active_orders:
            # Match by stale_reason containing the fast-path reference,
            # or by symbol if only one active order exists for the symbol.
            if (
                order.stale_reason == "fast_path_pending_order"
                or len(active_orders) == 1
            ):
                try:
                    registry.mark_canceled(order.order_id, "pm_veto")
                    cancelled_count += 1
                    logger.info(
                        "_cancel_pending_order_for_event: cancelled order %s "
                        "for event %s (symbol=%s)",
                        order.order_id,
                        event_id,
                        symbol,
                    )
                except Exception as cancel_exc:
                    logger.warning(
                        "_cancel_pending_order_for_event: failed to cancel "
                        "order %s: %s",
                        order.order_id,
                        cancel_exc,
                    )

        if cancelled_count == 0:
            logger.info(
                "_cancel_pending_order_for_event: no active pending orders "
                "found for symbol %s (event %s) — may already be "
                "filled/expired",
                symbol,
                event_id,
            )
    except Exception as exc:
        logger.error(
            "_cancel_pending_order_for_event: failed to cancel pending order "
            "for event %s: %s (fail-open)",
            event_id,
            exc,
        )


def _invalidate_watch_for_event(
    engine,
    event_id: str,
    trigger_id: str,
    symbol: str,
) -> None:
    """Invalidate watch candidates associated with a vetoed fast-path event.

    Lazily imports from watch_candidates to avoid circular dependencies.
    Searches for active watches matching the symbol and transitions them to
    invalidated state.  Fail-open: logs errors and returns without raising.
    """
    try:
        from utils.watch_candidates import _transition_watch_state

        # Find the watch created by this fast-path event.
        # The trigger's source_watch_id or the watch created in response to
        # the fast-path outcome may be linked via trigger_id in the events table.
        # We query active watches for the symbol and invalidate the most recent.
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT watch_id FROM watch_candidates "
                    "WHERE symbol = :symbol AND state = 'active' "
                    "ORDER BY created_at DESC LIMIT 1"
                ),
                {"symbol": symbol},
            ).mappings().all()

        if not rows:
            logger.info(
                "_invalidate_watch_for_event: no active watch found for "
                "symbol %s (event %s) — may already be invalidated/expired",
                symbol,
                event_id,
            )
            return

        watch_id = rows[0]["watch_id"]
        outcome_json = json.dumps(
            {
                "terminal_state": "invalidated",
                "terminal_reason": "pm_veto",
                "veto_event_id": event_id,
            }
        )
        success = _transition_watch_state(engine, watch_id, "invalidated", outcome_json)
        if success:
            logger.info(
                "_invalidate_watch_for_event: invalidated watch %s for "
                "event %s (symbol=%s)",
                watch_id,
                event_id,
                symbol,
            )
        else:
            logger.warning(
                "_invalidate_watch_for_event: CAS transition failed for "
                "watch %s — may already be in terminal state",
                watch_id,
            )
    except Exception as exc:
        logger.error(
            "_invalidate_watch_for_event: failed to invalidate watch for "
            "event %s: %s (fail-open)",
            event_id,
            exc,
        )
