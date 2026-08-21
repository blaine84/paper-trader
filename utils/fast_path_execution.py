"""Fast-path execution delegation — trade and pending order execution from fast-path outcomes.

Handles delegation of `trade_executed` and `pending_order_created` outcomes
to existing execution infrastructure. Builds decision dicts from the
trigger's frozen geometry (entry, stop, target from registration time) —
MUST NOT substitute fresh geometry from newer analyst signals.

Trade execution paths:
  - PM_CANDIDATE_MODE == "enabled": synthetic CandidateRecord → candidate pipeline
  - PM_CANDIDATE_MODE != "enabled": execute_trade(normalized=True) directly

Pending order delegation:
  - PENDING_ORDER_MODE == "enabled": delegate to maybe_create_pending_order()
  - PENDING_ORDER_MODE != "enabled": record intent only (no actual order)

Fail mode: fail-open for event metadata updates (log error, never blocks
pipeline). Fail-closed for execution validation (provenance check rejects
invalid triggers before any trade attempt).

See: .kiro/specs/fast-path-deterministic-execution/design.md
Requirements: 10.2, 10.3, 10.4, 10.5, 10.9, 10.10, 10.11, 4.4, 4.5
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from utils.gate_config import FAST_PATH_MODE, PENDING_ORDER_MODE, PM_CANDIDATE_MODE

logger = logging.getLogger(__name__)


def execute_fast_path_trade(
    outcome: Any,
    trigger: Any,
    db: Any,
    engine: Any,
) -> bool:
    """Execute a fast-path trade_executed outcome via existing infrastructure.

    Delegates to the candidate pipeline (when PM_CANDIDATE_MODE == "enabled")
    or directly to execute_trade (otherwise). Uses the trigger's frozen
    geometry — never substitutes fresh geometry from a newer signal.

    Args:
        outcome: FastPathOutcome from utils.fast_path_evaluator.
        trigger: TriggerRecord from utils.fast_path_registry.
        db: SQLAlchemy engine instance.
        engine: SQLAlchemy engine instance (same as db in this project).

    Returns:
        True on successful execution, False on failure.

    Guards:
        - Only callable when FAST_PATH_MODE == "enabled"
        - Rejects triggers without strategy provenance (source_signal_id
          or source_watch_id)
    """
    # Guard: only execute when fast path is enabled
    if FAST_PATH_MODE != "enabled":
        logger.warning(
            "execute_fast_path_trade called but FAST_PATH_MODE=%s, skipping",
            FAST_PATH_MODE,
        )
        return False

    # Provenance check: trigger must have valid source linkage
    if trigger.source_signal_id is None and trigger.source_watch_id is None:
        logger.warning(
            "execute_fast_path_trade: trigger %s for %s has no strategy "
            "provenance (source_signal_id=None, source_watch_id=None) — "
            "rejecting with stand_down(no_strategy_provenance)",
            trigger.trigger_id,
            trigger.symbol,
        )
        _update_event_failure_metadata(
            engine,
            outcome.trigger_id,
            reason="no_strategy_provenance",
            details="Trigger lacks source_signal_id and source_watch_id",
        )
        return False

    try:
        # Build decision dict from trigger's FROZEN geometry
        # (Requirements 10.9, 10.10: use signal_snapshot_json, not fresh data)
        decision = _build_decision_from_trigger(trigger, outcome)

        if PM_CANDIDATE_MODE == "enabled":
            return _execute_via_candidate_pipeline(
                decision, trigger, outcome, db, engine
            )
        else:
            return _execute_via_trade_direct(decision, trigger, outcome, db)

    except Exception as exc:
        logger.error(
            "execute_fast_path_trade: unexpected error for trigger %s (%s): %s",
            trigger.trigger_id,
            trigger.symbol,
            exc,
            exc_info=True,
        )
        # Fail-open: update event with failure metadata, don't crash monitor
        _update_event_failure_metadata(
            engine,
            outcome.trigger_id,
            reason="execution_exception",
            details=str(exc),
        )
        return False


def _build_decision_from_trigger(trigger: Any, outcome: Any) -> dict:
    """Build a trade decision dict from the trigger's frozen geometry.

    Uses entry_price, stop_price, target_price from the trigger registration
    (frozen at registration time). The current_price from the outcome is used
    as the price field for execution.

    MUST NOT substitute fresh geometry from a newer analyst signal —
    signal_snapshot_json from the trigger row is authoritative.
    (Requirements: 10.9, 10.10)
    """
    decision = {
        "action": trigger.direction.upper(),  # "BUY" or "SHORT"
        "symbol": trigger.symbol,
        "price": outcome.current_price,
        "entry_price": trigger.entry_price,
        "stop_price": trigger.stop_price,
        "target_price": trigger.target_price,
        "setup_type": trigger.setup_type,
        "quantity": 0,  # Position sizer will determine quantity
        "source": "fast_path",
        "trigger_id": trigger.trigger_id,
    }

    # Include geometry name if available
    if trigger.geometry_name:
        decision["geometry_name"] = trigger.geometry_name

    # Include invalidation and target basis for downstream context
    if trigger.invalidation_basis:
        decision["invalidation_basis"] = trigger.invalidation_basis
    if trigger.target_basis:
        decision["target_basis"] = trigger.target_basis

    return decision


def _execute_via_candidate_pipeline(
    decision: dict,
    trigger: Any,
    outcome: Any,
    db: Any,
    engine: Any,
) -> bool:
    """Execute via the candidate pipeline (PM_CANDIDATE_MODE == "enabled").

    Creates a synthetic CandidateRecord from the trigger's frozen geometry,
    registers it in the candidate registry, and calls
    execute_candidate_pipeline().

    (Requirements: 10.2, 10.3, 10.4, 10.5)
    """
    # Lazy imports to avoid circular dependencies
    from utils.candidate_pipeline import execute_candidate_pipeline
    from utils.candidate_registry import (
        CandidateRecord,
        CandidateRegistry,
    )
    from utils.decision_contract import CandidateDecision

    # Generate synthetic candidate and cycle IDs
    candidate_id = str(uuid.uuid4())
    cycle_id = f"fast_path_{trigger.trigger_id}"

    # Build frozen signal snapshot for the candidate
    signal_snapshot_json = trigger.signal_snapshot_json or json.dumps(
        {
            "source": "fast_path",
            "trigger_id": trigger.trigger_id,
            "entry_price": trigger.entry_price,
            "stop_price": trigger.stop_price,
            "target_price": trigger.target_price,
            "direction": trigger.direction,
            "setup_type": trigger.setup_type,
        },
        separators=(",", ":"),
    )

    now = datetime.now(timezone.utc)

    # Create synthetic CandidateRecord with frozen geometry
    candidate = CandidateRecord(
        candidate_id=candidate_id,
        cycle_id=cycle_id,
        profile_id=trigger.profile_id,
        symbol=trigger.symbol,
        direction=trigger.direction,
        setup_type=trigger.setup_type,
        geometry_name=trigger.geometry_name or "fast_path",
        entry_price=trigger.entry_price,
        stop_price=trigger.stop_price,
        target_price=trigger.target_price,
        risk_reward=float(outcome.reward_to_risk) if outcome.reward_to_risk else 0.0,
        trigger="fast_path_trigger",
        invalidation_basis=trigger.invalidation_basis or "",
        target_basis=trigger.target_basis or "",
        source_signal_id=trigger.source_signal_id or trigger.source_watch_id or "",
        signal_snapshot_json=signal_snapshot_json,
        created_at=now,
        expires_at=now + timedelta(minutes=5),
        integrity_hash="",  # Will be computed by registry
    )

    # Register the synthetic candidate
    registry = CandidateRegistry(
        db=db, cycle_id=cycle_id, profile_id=trigger.profile_id
    )
    registry.register(candidate)

    # Build the accept decision for the pipeline
    candidate_decision = CandidateDecision(
        candidate_id=candidate_id,
        decision="accept",
        rationale=f"Fast-path trade_executed: {outcome.outcome_reason_code}",
    )

    # Execute the pipeline — this runs: resolve → reserve → size → gates → execute
    result = execute_candidate_pipeline(
        db=db,
        engine=engine,
        registry=registry,
        decision=candidate_decision,
        portfolio={},  # Pipeline will fetch current portfolio state
        profile={"profile_id": trigger.profile_id},
        profile_id=trigger.profile_id,
    )

    success = result.outcome == "executed"
    if not success:
        logger.warning(
            "execute_fast_path_trade: candidate pipeline returned outcome=%s "
            "for trigger %s (%s), error=%s",
            result.outcome,
            trigger.trigger_id,
            trigger.symbol,
            result.error,
        )
        _update_event_failure_metadata(
            engine,
            outcome.trigger_id,
            reason=f"pipeline_{result.outcome}",
            details=result.error or "",
        )

    return success


def _execute_via_trade_direct(
    decision: dict,
    trigger: Any,
    outcome: Any,
    db: Any,
) -> bool:
    """Execute via execute_trade(normalized=True) directly.

    Used when PM_CANDIDATE_MODE != "enabled". Calls the portfolio manager's
    execute_trade with the frozen geometry decision.

    (Requirements: 10.5)
    """
    # Lazy import to avoid circular dependency with portfolio_manager
    from agents.portfolio_manager import execute_trade

    success, message = execute_trade(
        db=db,
        decision=decision,
        profile_id=trigger.profile_id,
        normalized=True,
    )

    if not success:
        logger.warning(
            "execute_fast_path_trade: execute_trade returned failure for "
            "trigger %s (%s): %s",
            trigger.trigger_id,
            trigger.symbol,
            message,
        )
        _update_event_failure_metadata(
            db,
            outcome.trigger_id,
            reason="execute_trade_failed",
            details=message or "",
        )

    return success


def _update_event_failure_metadata(
    engine: Any,
    trigger_id: str,
    reason: str,
    details: str,
) -> None:
    """Update the fast_path_event with failure metadata (fail-open).

    Attempts to update the outcome_metadata_json field on the fast_path_events
    row for this trigger. This is a fail-open operation: if the update fails,
    it logs the error and continues without raising.

    Note: fast_path_events has immutability triggers that block most updates.
    This uses the annotation columns (outcome_metadata_json) which are
    update-allowed. If the schema does not allow this update, the error is
    logged silently.
    """
    try:
        from sqlalchemy import text

        metadata = json.dumps(
            {"execution_failure_reason": reason, "details": details},
            separators=(",", ":"),
        )

        with engine.connect() as conn:
            conn.execute(
                text(
                    """
                    UPDATE fast_path_events
                    SET outcome_metadata_json = :metadata
                    WHERE trigger_id = :trigger_id
                    """
                ),
                {"metadata": metadata, "trigger_id": trigger_id},
            )
            conn.commit()
    except Exception as exc:
        # Fail-open: log and continue, never block the pipeline
        logger.debug(
            "execute_fast_path_trade: failed to update event failure metadata "
            "for trigger %s: %s (fail-open, continuing)",
            trigger_id,
            exc,
        )


def execute_fast_path_watch(
    outcome: Any,
    trigger: Any,
    db: Any,
    engine: Any,
) -> bool:
    """Delegate a fast-path watch outcome (watch_created or watch_promoted).

    When outcome_type == "watch_created":
        Imports create_watch_candidate from utils.watch_candidates lazily,
        builds watch parameters from the trigger's frozen geometry, and
        creates a watch candidate row.

    When outcome_type == "watch_promoted":
        This is a promotion case — the watch was already promoted through
        the normal watch flow. Logs that promotion was triggered from
        fast path and returns True.

    No provenance check needed for watches (watches are benign non-executing
    outcomes). However, FAST_PATH_MODE must be "enabled" — observe mode does
    NOT create watches from fast-path evaluation.

    Args:
        outcome: FastPathOutcome from utils.fast_path_evaluator.
        trigger: TriggerRecord from utils.fast_path_registry.
        db: SQLAlchemy engine instance.
        engine: SQLAlchemy engine instance (same as db in this project).

    Returns:
        True on successful watch creation/promotion, False on failure.

    Requirements: 3.2, 3.3, 4.3
    """
    # Guard: watches are only created/promoted when fast path is enabled
    if FAST_PATH_MODE != "enabled":
        logger.info(
            "execute_fast_path_watch: FAST_PATH_MODE=%s, skipping watch "
            "delegation for trigger %s (%s)",
            FAST_PATH_MODE,
            trigger.trigger_id,
            trigger.symbol,
        )
        return False

    try:
        if outcome.outcome_type == "watch_created":
            return _delegate_watch_creation(outcome, trigger, engine)
        elif outcome.outcome_type == "watch_promoted":
            return _delegate_watch_promotion(outcome, trigger, engine)
        else:
            logger.warning(
                "execute_fast_path_watch: unexpected outcome_type=%s for "
                "trigger %s (%s)",
                outcome.outcome_type,
                trigger.trigger_id,
                trigger.symbol,
            )
            return False

    except Exception as exc:
        logger.error(
            "execute_fast_path_watch: unexpected error for trigger %s (%s): %s",
            trigger.trigger_id,
            trigger.symbol,
            exc,
            exc_info=True,
        )
        return False


def _delegate_watch_creation(
    outcome: Any,
    trigger: Any,
    engine: Any,
) -> bool:
    """Create a watch candidate from the trigger's frozen geometry.

    Lazily imports create_watch_candidate from utils.watch_candidates and
    builds watch parameters from the trigger data.

    Requirements: 3.2
    """
    # Lazy import to avoid circular dependencies
    from utils.watch_candidates import create_watch_candidate

    # Parse signal snapshot for additional context
    signal_snapshot = None
    if trigger.signal_snapshot_json:
        try:
            signal_snapshot = json.loads(trigger.signal_snapshot_json)
        except (json.JSONDecodeError, TypeError):
            signal_snapshot = None

    # Build key levels from trigger geometry
    key_levels = {}
    if trigger.entry_price:
        key_levels["entry"] = trigger.entry_price
    if trigger.stop_price:
        key_levels["stop"] = trigger.stop_price
    if trigger.target_price:
        key_levels["target"] = trigger.target_price
    if trigger.trigger_level:
        key_levels["trigger_level"] = trigger.trigger_level

    watch_id = create_watch_candidate(
        engine,
        symbol=trigger.symbol,
        profile_id=trigger.profile_id,
        direction=trigger.direction,
        entry_price=trigger.entry_price,
        stop_price=trigger.stop_price,
        target_price=trigger.target_price,
        setup_type=trigger.setup_type,
        reason=f"Fast-path watch_created: {outcome.outcome_reason_code}",
        source_cycle_id=f"fast_path_{trigger.trigger_id}",
        key_levels=key_levels,
        signal_snapshot=signal_snapshot,
    )

    if watch_id:
        logger.info(
            "execute_fast_path_watch: watch candidate %s created for "
            "trigger %s (%s, %s)",
            watch_id,
            trigger.trigger_id,
            trigger.symbol,
            trigger.setup_type,
        )
        return True
    else:
        logger.warning(
            "execute_fast_path_watch: failed to create watch candidate for "
            "trigger %s (%s)",
            trigger.trigger_id,
            trigger.symbol,
        )
        return False


def _delegate_watch_promotion(
    outcome: Any,
    trigger: Any,
    engine: Any,
) -> bool:
    """Handle watch_promoted outcome from fast path.

    The promotion was already tracked through the normal watch flow
    (evaluate_active_watch_candidates → promoted state). The fast path
    simply acknowledges this and logs it.

    Requirements: 3.3
    """
    logger.info(
        "execute_fast_path_watch: watch promotion triggered from fast path "
        "for trigger %s (%s), source_watch_id=%s",
        trigger.trigger_id,
        trigger.symbol,
        trigger.source_watch_id,
    )
    return True


def execute_fast_path_pending_order(
    outcome: Any,
    trigger: Any,
    db: Any,
    engine: Any,
) -> bool:
    """Delegate a fast-path pending_order_created outcome to pending order infrastructure.

    When PENDING_ORDER_MODE == "enabled", calls maybe_create_pending_order()
    from utils.pending_order_creation with the trigger's frozen geometry.
    When PENDING_ORDER_MODE != "enabled", records the intent in event metadata
    only — no actual order is created.

    Args:
        outcome: FastPathOutcome with outcome_type == "pending_order_created".
        trigger: TriggerRecord from utils.fast_path_registry.
        db: SQLAlchemy engine instance.
        engine: SQLAlchemy engine instance (same as db in this project).

    Returns:
        True on successful delegation (or successful intent recording),
        False on failure.

    Guards:
        - Only callable when FAST_PATH_MODE == "enabled"
        - Rejects triggers without strategy provenance (source_signal_id
          or source_watch_id)

    Requirements: 4.4, 4.5
    """
    # Guard: only execute when fast path is enabled
    if FAST_PATH_MODE != "enabled":
        logger.warning(
            "execute_fast_path_pending_order called but FAST_PATH_MODE=%s, skipping",
            FAST_PATH_MODE,
        )
        return False

    # Provenance check: trigger must have valid source linkage
    if trigger.source_signal_id is None and trigger.source_watch_id is None:
        logger.warning(
            "execute_fast_path_pending_order: trigger %s for %s has no strategy "
            "provenance (source_signal_id=None, source_watch_id=None) — "
            "rejecting with stand_down(no_strategy_provenance)",
            trigger.trigger_id,
            trigger.symbol,
        )
        _update_event_failure_metadata(
            engine,
            outcome.trigger_id,
            reason="no_strategy_provenance",
            details="Trigger lacks source_signal_id and source_watch_id",
        )
        return False

    try:
        if PENDING_ORDER_MODE == "enabled":
            return _delegate_pending_order(outcome, trigger, db, engine)
        else:
            # Record intent only — no actual order created
            logger.info(
                "execute_fast_path_pending_order: PENDING_ORDER_MODE=%s, "
                "recording intent only for trigger %s (%s) — no actual order created",
                PENDING_ORDER_MODE,
                trigger.trigger_id,
                trigger.symbol,
            )
            _record_pending_order_intent(engine, outcome, trigger)
            return True

    except Exception as exc:
        logger.error(
            "execute_fast_path_pending_order: unexpected error for trigger %s (%s): %s",
            trigger.trigger_id,
            trigger.symbol,
            exc,
            exc_info=True,
        )
        _update_event_failure_metadata(
            engine,
            outcome.trigger_id,
            reason="pending_order_exception",
            details=str(exc),
        )
        return False


def _delegate_pending_order(
    outcome: Any,
    trigger: Any,
    db: Any,
    engine: Any,
) -> bool:
    """Delegate to maybe_create_pending_order() with trigger's frozen geometry.

    Builds the pending order parameters from the trigger's frozen geometry
    (entry, stop, target from registration time). The entry_price from the
    trigger becomes the limit price (intended_entry).

    Requirements: 4.4
    """
    # Lazy import to avoid circular dependencies
    from utils.pending_order_creation import maybe_create_pending_order

    # Build decision dict from trigger's frozen geometry
    decision = _build_decision_from_trigger(trigger, outcome)

    result = maybe_create_pending_order(
        db=db,
        decision=decision,
        profile_id=trigger.profile_id,
        action=trigger.direction.upper(),
        symbol=trigger.symbol,
        intended_entry=trigger.entry_price,
        fresh_price=outcome.current_price,
        stop=trigger.stop_price,
        target=trigger.target_price,
        stale_reason="fast_path_pending_order",
        engine=engine,
    )

    if result.created:
        logger.info(
            "execute_fast_path_pending_order: pending order created for "
            "trigger %s (%s), order_id=%s",
            trigger.trigger_id,
            trigger.symbol,
            result.order_id,
        )
        return True
    else:
        logger.warning(
            "execute_fast_path_pending_order: pending order declined for "
            "trigger %s (%s), reason=%s",
            trigger.trigger_id,
            trigger.symbol,
            result.decline_reason,
        )
        _update_event_failure_metadata(
            engine,
            outcome.trigger_id,
            reason="pending_order_declined",
            details=result.decline_reason or "",
        )
        return False


def _record_pending_order_intent(
    engine: Any,
    outcome: Any,
    trigger: Any,
) -> None:
    """Record pending order intent in event metadata when mode is not enabled.

    The outcome event was already persisted by the monitor. This updates the
    outcome_metadata_json with the intent details for audit/telemetry purposes.
    Fail-open: if the update fails, log and continue.

    Requirements: 4.5
    """
    try:
        from sqlalchemy import text

        intent_metadata = json.dumps(
            {
                "pending_order_intent": True,
                "pending_order_mode": PENDING_ORDER_MODE,
                "intended_entry": float(trigger.entry_price) if trigger.entry_price else None,
                "current_price": float(outcome.current_price) if outcome.current_price else None,
                "stop_price": float(trigger.stop_price) if trigger.stop_price else None,
                "target_price": float(trigger.target_price) if trigger.target_price else None,
                "direction": trigger.direction,
                "symbol": trigger.symbol,
                "note": "Intent recorded but no actual order created (PENDING_ORDER_MODE != enabled)",
            },
            separators=(",", ":"),
        )

        with engine.connect() as conn:
            conn.execute(
                text(
                    """
                    UPDATE fast_path_events
                    SET outcome_metadata_json = :metadata
                    WHERE trigger_id = :trigger_id
                    """
                ),
                {"metadata": intent_metadata, "trigger_id": trigger.trigger_id},
            )
            conn.commit()
    except Exception as exc:
        # Fail-open: log and continue, never block the pipeline
        logger.debug(
            "execute_fast_path_pending_order: failed to record intent metadata "
            "for trigger %s: %s (fail-open, continuing)",
            trigger.trigger_id,
            exc,
        )
