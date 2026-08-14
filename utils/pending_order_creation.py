"""Creation path for pending limit orders.

Lives outside ``agents/portfolio_manager.py`` (already ~5000 lines) so that
branch classification, decline emission, and the cap/duplicate checks stay out
of the live trading module. The hook there is a handful of lines.

Two things here are load-bearing and easy to get wrong:

**The limit price is the ORIGINAL intended entry.** ``execute_trade()``'s Tier 2
repair (deviation above 5%, at or below 10%) overwrites ``price`` with the live
quote and rewrites ``decision["price"]``/``["entry_price"]`` plus the rescaled
stop and target. A limit sourced from the post-repair decision would rest at the
chased price, which is the exact opposite of this feature's purpose. The caller
snapshots the entry before the tier block and passes it in.

**Only the runaway branch creates an order.** The other stale-entry branch fires
when the fresh price has already crossed the *profit target*, meaning the market
traded through the whole reward leg. Resting a limit there yields a position
whose entry equals its target — zero reward, and ``validate_trade()`` step 5
would reject it at fill time anyway. That branch is recorded as a decline so its
counterfactual rate is measurable, not converted into an order.

Everything in this module is fail-open. Order creation is additive to the
stale-entry rejection: ``execute_trade()`` returns ``(False, reason)`` either
way, so nothing about the trading outcome depends on this succeeding.

Requirements: 1.1-1.17, 2.x, 3.4, 8.5, 10.1, 10.3, 14.3-14.7
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from utils.gate_config import (
    PENDING_ORDER_MAX_ACTIVE_PER_PROFILE,
    PENDING_ORDER_MAX_RUNAWAY_PCT,
    PENDING_ORDER_MODE,
    TRIGGERED_PLAN_MODE,
)
from utils.pending_order_expiry import resolve_expiry
from utils.pending_order_registry import (
    OrderState,
    PendingOrder,
    PendingOrderRegistry,
    PendingOrderRegistryError,
)
from utils.pending_order_time import now_utc, to_iso

logger = logging.getLogger(__name__)

# Branch labels returned by classify_stale_entry_branch().
BRANCH_RUNAWAY = "runaway"
BRANCH_TARGET_EXCEEDED = "target_exceeded"
BRANCH_NEITHER = "neither"

__all__ = [
    "BRANCH_NEITHER",
    "BRANCH_RUNAWAY",
    "BRANCH_TARGET_EXCEEDED",
    "BranchClassification",
    "CreationOutcome",
    "classify_stale_entry_branch",
    "emit_repair_band_decline",
    "maybe_create_pending_order",
]


@dataclass(frozen=True)
class BranchClassification:
    """Which stale-entry branch fired, and how far price ran."""

    branch: str
    runaway_pct: float | None = None

    @property
    def is_runaway(self) -> bool:
        return self.branch == BRANCH_RUNAWAY


@dataclass(frozen=True)
class CreationOutcome:
    """Result of a creation attempt. ``created`` and ``decline_reason`` are exclusive."""

    created: bool
    order_id: str | None = None
    decline_reason: str | None = None
    superseded: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Branch classification
# ---------------------------------------------------------------------------


def classify_stale_entry_branch(
    *,
    action: str,
    intended_entry: Any,
    fresh_price: Any,
    target: Any,
) -> BranchClassification:
    """Recompute which branch of ``_fresh_price_stale_entry_check()`` fired.

    Deliberately recomputes rather than parsing the rejection message string, so
    a wording change in that function cannot silently reroute orders. Reuses
    ``_try_positive_float`` and ``STALE_ENTRY_MAX_FAVORABLE_MOVE_PCT`` from
    ``portfolio_manager`` via lazy import, so the classification cannot drift
    from the check it mirrors.

    Args:
        action: "BUY", "SHORT", or anything else.
        intended_entry: The ORIGINAL intended entry, pre-repair.
        fresh_price: The live quote the check compared against.
        target: The profit target.

    Returns:
        BranchClassification. ``runaway_pct`` is populated whenever the
        favorable-move percentage is computable, including when the branch is
        ``neither``, so callers can log it.
    """
    try:
        from agents.portfolio_manager import (
            STALE_ENTRY_MAX_FAVORABLE_MOVE_PCT,
            _try_positive_float,
        )
    except Exception:  # pragma: no cover - defensive
        logger.warning(
            "Could not import stale-entry helpers; treating as 'neither'",
            exc_info=True,
        )
        return BranchClassification(BRANCH_NEITHER)

    normalized_action = str(action).strip().upper()
    if normalized_action not in {"BUY", "SHORT"}:
        return BranchClassification(BRANCH_NEITHER)

    entry = _try_positive_float(intended_entry)
    fresh = _try_positive_float(fresh_price)
    target_price = _try_positive_float(target)

    if entry is None or fresh is None:
        return BranchClassification(BRANCH_NEITHER)

    if normalized_action == "BUY":
        if target_price is not None and fresh >= target_price:
            return BranchClassification(BRANCH_TARGET_EXCEEDED)
        favorable_move_pct = (fresh - entry) / entry
    else:
        if target_price is not None and fresh <= target_price:
            return BranchClassification(BRANCH_TARGET_EXCEEDED)
        favorable_move_pct = (entry - fresh) / entry

    if favorable_move_pct > STALE_ENTRY_MAX_FAVORABLE_MOVE_PCT:
        return BranchClassification(BRANCH_RUNAWAY, favorable_move_pct)

    return BranchClassification(BRANCH_NEITHER, favorable_move_pct)


# ---------------------------------------------------------------------------
# Repair-band observability
# ---------------------------------------------------------------------------


def emit_repair_band_decline(
    *,
    db: Any,
    profile_id: str,
    symbol: str,
    action: str,
    original_intended_entry: Any,
    live_price: Any,
    deviation: float,
    original_stop: Any,
    original_target: Any,
    repaired_stop: Any,
    repaired_target: Any,
) -> None:
    """Record that the Tier 2 repair pre-empted the stale-entry check.

    The deviation tiers run BEFORE ``_fresh_price_stale_entry_check()``. Tier 2
    sets ``price = live_price`` and rewrites the geometry, so by the time the
    check runs ``intended_entry == fresh_price``: the runaway branch computes a
    favorable move of zero and the repaired target sits beyond the live price, so
    *neither* stale-entry branch can fire. Runaway coverage is therefore bounded
    to roughly the 1%-5% range.

    v1 does not change that. It measures it: this event sizes how much
    missed-entry volume the repair band is absorbing and chasing, which is the
    evidence needed before implementing PENDING_ORDER_DIVERT_REPAIR_BAND.

    Creates NO order. A repair-band decision still executes its repaired trade,
    and resting an order alongside it would double-book the same intent.

    Fail-open — never raises.
    """
    if PENDING_ORDER_MODE == "disabled":
        return

    try:
        _emit_trade_event(
            engine=_engine_from(db),
            event_type="pending_order_declined",
            profile_id=profile_id,
            symbol=symbol,
            price=_as_float(original_intended_entry),
            message=(
                f"{symbol}: pending order declined - Tier 2 price repair "
                f"pre-empted the stale-entry check "
                f"({deviation * 100:.2f}% deviation)"
            ),
            payload={
                "reason": "repaired_before_check",
                "action": str(action).upper(),
                "original_intended_entry": _as_float(original_intended_entry),
                "live_price": _as_float(live_price),
                "deviation_pct": round(float(deviation), 6),
                "original_stop": _as_float(original_stop),
                "original_target": _as_float(original_target),
                "repaired_stop": _as_float(repaired_stop),
                "repaired_target": _as_float(repaired_target),
                "note": (
                    "The repaired trade still executes. No pending order was "
                    "created, because resting one alongside a repaired fill "
                    "would double-book the intent."
                ),
            },
        )
    except Exception:
        logger.error(
            "Failed to emit repair-band decline for %s (non-fatal)",
            symbol, exc_info=True,
        )


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


def maybe_create_pending_order(
    *,
    db: Any,
    decision: dict,
    profile_id: str,
    action: str,
    symbol: str,
    intended_entry: Any,
    fresh_price: Any,
    stop: Any,
    target: Any,
    stale_reason: str | None = None,
    engine: Any = None,
) -> CreationOutcome:
    """Create a resting limit order for a runaway-entry rejection, if eligible.

    Read-only gates run first, side effects last, so a decline can never leave a
    valid order cancelled for nothing.

    Args:
        db: The caller's SQLAlchemy session (used only to reach the engine).
        decision: The PM decision dict. Read for metadata only; never mutated.
        profile_id: Profile the decision belongs to.
        action: "BUY" or "SHORT". Anything else returns immediately.
        symbol: Ticker.
        intended_entry: The ORIGINAL intended entry, captured before the
            deviation tiers could overwrite it. Becomes the limit price.
        fresh_price: The live quote at rejection time.
        stop: Stop price for the trade.
        target: Profit target.
        stale_reason: The rejection message, recorded for audit only — never
            parsed for control flow.
        engine: Optional engine override, for tests.

    Returns:
        CreationOutcome describing what happened. Never raises; the caller wraps
        this in try/except as a second layer.
    """
    if PENDING_ORDER_MODE == "disabled":
        return CreationOutcome(created=False, decline_reason=None)

    normalized_action = str(action).strip().upper()
    if normalized_action not in {"BUY", "SHORT"}:
        return CreationOutcome(created=False, decline_reason=None)

    resolved_engine = engine if engine is not None else _engine_from(db)
    setup_type = _resolve_setup_type(decision)

    def decline(reason: str, **extra: Any) -> CreationOutcome:
        _emit_decline(
            engine=resolved_engine,
            reason=reason,
            profile_id=profile_id,
            symbol=symbol,
            action=normalized_action,
            intended_entry=intended_entry,
            fresh_price=fresh_price,
            stop=stop,
            target=target,
            setup_type=setup_type,
            stale_reason=stale_reason,
            extra=extra,
        )
        return CreationOutcome(created=False, decline_reason=reason)

    # ── Gate 1: only the runaway branch is eligible ──
    classification = classify_stale_entry_branch(
        action=normalized_action,
        intended_entry=intended_entry,
        fresh_price=fresh_price,
        target=target,
    )

    if classification.branch == BRANCH_TARGET_EXCEEDED:
        return decline("target_already_exceeded")

    if classification.branch != BRANCH_RUNAWAY:
        # The caller only invokes this on a stale-entry rejection, so landing
        # here means the classification disagreed with the check — worth a log,
        # but not an event, since no intent was discarded.
        logger.debug(
            "%s: stale-entry rejection did not classify as runaway "
            "(branch=%s); no pending order",
            symbol, classification.branch,
        )
        return CreationOutcome(created=False, decline_reason=None)

    # ── Gate 2: runaway magnitude ceiling ──
    runaway_pct = classification.runaway_pct or 0.0
    if runaway_pct > PENDING_ORDER_MAX_RUNAWAY_PCT:
        return decline("runaway_exceeds_max", runaway_pct=round(runaway_pct, 6))

    # ── Gate 3: geometry must be complete ──
    limit_price = _as_positive_float(intended_entry)
    stop_price = _as_positive_float(stop)
    target_price = _as_positive_float(target)
    if limit_price is None or stop_price is None or target_price is None:
        return decline(
            "incomplete_geometry",
            has_entry=limit_price is not None,
            has_stop=stop_price is not None,
            has_target=target_price is not None,
        )

    # ── Gate 4: geometry must be valid AT THE LIMIT ──
    if not _geometry_valid_at_limit(
        normalized_action, limit_price, stop_price, target_price
    ):
        return decline(
            "invalid_geometry_at_limit",
            limit_price=limit_price,
            stop_price=stop_price,
            target_price=target_price,
        )

    # ── Gate 5: the active window must be usable ──
    created_at = now_utc()
    expires_at = resolve_expiry(created_at_utc=created_at, setup_type=setup_type)
    if expires_at is None:
        return decline("window_too_short", setup_type=setup_type)

    registry = PendingOrderRegistry(resolved_engine)

    # ── Gate 6: an active trade plan already covers this idea ──
    if TRIGGERED_PLAN_MODE != "disabled" and _has_active_trade_plan(
        resolved_engine, profile_id, symbol, normalized_action
    ):
        return decline("active_trade_plan_exists")

    # ── Gate 7: per-profile cap, discounting orders we are about to supersede ──
    try:
        duplicates = registry.find_duplicate_active(
            profile_id, symbol, normalized_action, setup_type
        )
        active_count = registry.count_active_for_profile(profile_id)
    except Exception:
        logger.error(
            "Could not evaluate pending order caps for %s (non-fatal)",
            symbol, exc_info=True,
        )
        return CreationOutcome(created=False, decline_reason=None)

    # Duplicates are checked BEFORE the cap and subtracted from it. Checking the
    # cap first would spuriously decline an at-cap profile whose only blocker is
    # an order about to be replaced; superseding first would cancel a valid
    # resting order and then decline anyway.
    effective_active = active_count - len(duplicates)
    if effective_active >= PENDING_ORDER_MAX_ACTIVE_PER_PROFILE:
        return decline(
            "active_order_cap_reached",
            active_count=active_count,
            cap=PENDING_ORDER_MAX_ACTIVE_PER_PROFILE,
        )

    # ── Side effects begin here ──
    superseded: tuple[str, ...] = ()
    if duplicates:
        try:
            superseded = tuple(
                registry.supersede_duplicates(
                    profile_id, symbol, normalized_action, setup_type
                )
            )
        except Exception:
            logger.error(
                "Could not supersede duplicate pending orders for %s",
                symbol, exc_info=True,
            )

    # Build and persist inside one broad guard. This function promises never to
    # raise: creation is additive to a rejection that has already been decided,
    # so no failure here should be able to change the trading outcome. The
    # caller wraps this in try/except as a second layer, but the contract holds
    # here rather than depending on that.
    try:
        snapshot = _capture_signal_snapshot(db, symbol, decision)
    except Exception:
        logger.debug(
            "Signal snapshot failed for %s; continuing without it",
            symbol, exc_info=True,
        )
        snapshot = None

    try:
        order = PendingOrder(
            order_id=str(uuid.uuid4()),
            profile_id=profile_id,
            symbol=symbol,
            side=normalized_action,
            setup_type=setup_type,
            geometry_name=decision.get("geometry_name"),
            candidate_id=(
                decision.get("pm_candidate_id") or decision.get("candidate_id")
            ),
            cycle_id=decision.get("cycle_id"),
            source_signal_id=decision.get("source_signal_id"),
            plan_id=decision.get("plan_id"),
            limit_price=limit_price,      # the ORIGINAL entry, never the target
            stop_price=stop_price,
            target_price=target_price,
            risk_reward=_compute_risk_reward(
                normalized_action, limit_price, stop_price, target_price
            ),
            intended_quantity=_as_int(decision.get("quantity")),
            fresh_price_at_creation=_as_float(fresh_price) or 0.0,
            runaway_pct_at_creation=round(runaway_pct, 6),
            pm_rationale=decision.get("rationale"),
            signal_snapshot_json=snapshot,
            state=OrderState.PENDING,
            created_at=created_at,
            expires_at=expires_at,
        )
        order_id = registry.create_order(order)
    except Exception:
        logger.error(
            "Could not persist pending order for %s (non-fatal)",
            symbol, exc_info=True,
        )
        return CreationOutcome(
            created=False, decline_reason=None, superseded=superseded
        )

    logger.info(
        "PENDING_ORDER_CREATED: %s %s limit=%.2f stop=%.2f target=%.2f "
        "fresh=%.2f runaway=%.2f%% expires=%s order_id=%s",
        normalized_action, symbol, limit_price, stop_price, target_price,
        _as_float(fresh_price) or 0.0, runaway_pct * 100,
        expires_at.isoformat(), order_id,
    )

    # Both of these are internally fail-open, but the order is already persisted
    # at this point — nothing after it may turn a created order into a raised
    # exception, so guard them here too.
    try:
        _emit_created_event(resolved_engine, order, stale_reason)
        _link_candidate(resolved_engine, order)
    except Exception:
        logger.error(
            "Post-creation bookkeeping failed for order %s (order stands)",
            order_id, exc_info=True,
        )

    return CreationOutcome(
        created=True, order_id=order_id, superseded=superseded
    )


# ---------------------------------------------------------------------------
# Candidate linkage
# ---------------------------------------------------------------------------


def _link_candidate(engine: Any, order: PendingOrder) -> None:
    """Record the order on the source candidate's event trail.

    Only meaningful when PM_CANDIDATE_MODE is enabled; the live legacy path
    produces no candidate_id and this is a no-op.

    NOTE (known semantic compromise): the candidate itself still reaches
    EXECUTION_FAILED with rejection_reason="pending_order_created", because
    execute_trade() returned False and utils/candidate_pipeline.py maps that to
    mark_execution_failed(). EXECUTION_FAILED does not describe "a resting order
    was created". Adding a proper CandidateState would mean touching
    CandidateRegistry and finalize_cycle(), and RESERVED -> NOT_SELECTED is not a
    permitted transition. PM_CANDIDATE_MODE is disabled in production, so no live
    data is affected; this row plus the rejection_reason discriminator make the
    affected candidates trivially findable when the proper state lands.

    Fail-open.
    """
    if not order.candidate_id:
        return

    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO pm_candidate_events
                        (candidate_id, cycle_id, profile_id, event_type,
                         event_data, created_at)
                    VALUES
                        (:candidate_id, :cycle_id, :profile_id, :event_type,
                         :event_data, :created_at)
                    """
                ),
                {
                    "candidate_id": order.candidate_id,
                    "cycle_id": order.cycle_id or "",
                    "profile_id": order.profile_id,
                    "event_type": "pending_order_created",
                    "event_data": json.dumps({
                        "order_id": order.order_id,
                        "limit_price": order.limit_price,
                        "expires_at": to_iso(order.expires_at),
                    }),
                    "created_at": to_iso(order.created_at),
                },
            )
            conn.commit()
    except Exception:
        logger.warning(
            "Could not link pending order %s to candidate %s (non-fatal)",
            order.order_id, order.candidate_id, exc_info=True,
        )


# ---------------------------------------------------------------------------
# Event emission
# ---------------------------------------------------------------------------


def _emit_created_event(
    engine: Any, order: PendingOrder, stale_reason: str | None
) -> None:
    """Emit pending_order_created to trade_events. Fail-open."""
    try:
        _emit_trade_event(
            engine=engine,
            event_type="pending_order_created",
            profile_id=order.profile_id,
            symbol=order.symbol,
            price=order.limit_price,
            message=(
                f"{order.symbol}: pending {order.side} limit order resting at "
                f"{order.limit_price:.2f} (fresh price "
                f"{order.fresh_price_at_creation:.2f}) until "
                f"{order.expires_at.isoformat()}"
            ),
            payload={
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "setup_type": order.setup_type,
                "limit_price": order.limit_price,
                "stop_price": order.stop_price,
                "target_price": order.target_price,
                "risk_reward": order.risk_reward,
                "intended_quantity": order.intended_quantity,
                "fresh_price_at_creation": order.fresh_price_at_creation,
                "runaway_pct_at_creation": order.runaway_pct_at_creation,
                "expires_at": to_iso(order.expires_at),
                "candidate_id": order.candidate_id,
                "cycle_id": order.cycle_id,
                "profile_id": order.profile_id,
                "stale_entry_reason": stale_reason,
            },
            pm_candidate_id=order.candidate_id,
        )
    except Exception:
        logger.error(
            "Failed to emit pending_order_created for %s (non-fatal)",
            order.symbol, exc_info=True,
        )


def _emit_decline(
    *,
    engine: Any,
    reason: str,
    profile_id: str,
    symbol: str,
    action: str,
    intended_entry: Any,
    fresh_price: Any,
    stop: Any,
    target: Any,
    setup_type: str,
    stale_reason: str | None,
    extra: dict[str, Any],
) -> None:
    """Emit pending_order_declined. Fail-open."""
    payload = {
        "reason": reason,
        "symbol": symbol,
        "side": action,
        "setup_type": setup_type,
        "intended_entry": _as_float(intended_entry),
        "fresh_price": _as_float(fresh_price),
        "stop_price": _as_float(stop),
        "target_price": _as_float(target),
        "profile_id": profile_id,
        "stale_entry_reason": stale_reason,
    }
    payload.update(extra)

    try:
        _emit_trade_event(
            engine=engine,
            event_type="pending_order_declined",
            profile_id=profile_id,
            symbol=symbol,
            price=_as_float(intended_entry),
            message=f"{symbol}: pending order declined - {reason}",
            payload=payload,
        )
    except Exception:
        logger.error(
            "Failed to emit pending_order_declined(%s) for %s (non-fatal)",
            reason, symbol, exc_info=True,
        )


def _emit_trade_event(
    *,
    engine: Any,
    event_type: str,
    profile_id: str,
    symbol: str,
    price: float | None,
    message: str,
    payload: dict,
    pm_candidate_id: str | None = None,
) -> None:
    """Write a trade_events row using a DEDICATED session.

    Deliberately not the caller's session. ``execute_trade()`` returns
    ``(False, reason)`` on the stale-entry path without committing, so an event
    added to that session would either be discarded on rollback or ride along on
    an unrelated later commit. Same reasoning as
    ``plan_executor._record_missed_setup_event()``.
    """
    from db.schema import get_session
    from utils.trade_events import log_trade_event

    session = get_session(engine)
    try:
        log_trade_event(
            session,
            event_type,
            agent="pending_order_creation",
            symbol=symbol,
            profile=profile_id,
            price=price,
            message=message,
            payload=payload,
            pm_candidate_id=pm_candidate_id,
        )
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _engine_from(db: Any) -> Any:
    """Reach the engine from a session, matching execute_trade's own `db.bind`."""
    bind = getattr(db, "bind", None)
    if bind is not None:
        return bind
    get_bind = getattr(db, "get_bind", None)
    if callable(get_bind):
        return get_bind()
    # Already an engine.
    return db


def _resolve_setup_type(decision: dict) -> str:
    """Best-effort setup type; the empty string is acceptable for the key tuple."""
    for key in ("setup_type", "normalized_setup_type", "setup"):
        value = decision.get(key)
        if value:
            return str(value)
    return "unknown"


def _geometry_valid_at_limit(
    action: str, limit: float, stop: float, target: float
) -> bool:
    """Whether stop/limit/target are correctly ordered for the side.

    Checked at the limit, not at the fresh price, because the limit is where the
    fill would occur. Uses Decimal so the comparison matches the fill path.
    """
    try:
        limit_d = Decimal(str(limit))
        stop_d = Decimal(str(stop))
        target_d = Decimal(str(target))
    except (InvalidOperation, ValueError, TypeError):
        return False

    if action == "BUY":
        return stop_d < limit_d < target_d
    return target_d < limit_d < stop_d


def _compute_risk_reward(
    action: str, limit: float, stop: float, target: float
) -> float:
    """Reward-to-risk at the limit price, in Decimal. Zero when risk is degenerate."""
    try:
        limit_d = Decimal(str(limit))
        stop_d = Decimal(str(stop))
        target_d = Decimal(str(target))
    except (InvalidOperation, ValueError, TypeError):
        return 0.0

    if action == "BUY":
        risk = limit_d - stop_d
        reward = target_d - limit_d
    else:
        risk = stop_d - limit_d
        reward = limit_d - target_d

    if risk <= 0:
        return 0.0
    return round(float(reward / risk), 4)


def _capture_signal_snapshot(db: Any, symbol: str, decision: dict) -> str | None:
    """Snapshot the Analyst signal for audit. Fail-open, returns None on error.

    The filler rebuilds the signal at fill time (fresher, and what the gates
    should see); this snapshot exists so review can reconstruct what the PM was
    looking at when the order was created.
    """
    try:
        from agents.portfolio_manager import _build_signal_for_symbol

        signal = _build_signal_for_symbol(db, symbol, decision)
        if not signal:
            return None
        return json.dumps(signal, default=str)
    except Exception:
        logger.debug(
            "Could not capture signal snapshot for %s", symbol, exc_info=True
        )
        return None


def _has_active_trade_plan(
    engine: Any, profile_id: str, symbol: str, side: str
) -> bool:
    """Whether an active trade plan already covers this symbol and side.

    Read-only. Prevents two resting intents for one idea when the triggered-plan
    subsystem is also enabled. Fail-open: on error, assume no plan, since
    blocking creation on a failed read would be the wrong direction.
    """
    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT 1 FROM trade_plans
                    WHERE profile_id = :profile_id
                      AND symbol = :symbol
                      AND direction = :direction
                      AND state IN ('planned', 'watching', 'triggered')
                    LIMIT 1
                    """
                ),
                {
                    "profile_id": profile_id,
                    "symbol": symbol,
                    "direction": side,
                },
            ).fetchone()
        return row is not None
    except Exception:
        logger.debug(
            "Could not check for an active trade plan for %s (assuming none)",
            symbol, exc_info=True,
        )
        return False


def _as_float(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_positive_float(value: Any) -> float | None:
    result = _as_float(value)
    if result is None or result <= 0:
        return None
    return result


def _as_int(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        result = int(float(value))
        return result if result > 0 else None
    except (TypeError, ValueError):
        return None
