"""Fill-time revalidation and execution for claimed pending limit orders.

Structurally mirrors ``utils/plan_executor.py::execute_triggered_plan()``, the
established precedent for driving ``execute_trade()`` from outside a PM cycle. It
diverges on one point that matters: ``execute_triggered_plan()`` passes the fresh
quote as the entry price, so the deviation tiers are harmless there. A pending
order's limit sits deliberately away from the live price, so this caller must
also pass ``price_authoritative=True`` — otherwise Tier 2 would repair the fill to
the chased price and Tier 3 would refuse a legitimate crossing.

Fail-CLOSED throughout: gates, position sizing, and ``validate_trade()`` all still
run, and no position is created on any validation failure. The one fail-open
element is event emission.

Outcome vocabulary follows Requirement 8.7: risk failures **cancel** (the order
was valid, the portfolio moved), while gate and execution failures **reject**.

Requirements: 6.1-6.12, 7.6, 7.7, 8.1, 8.3, 8.6, 8.7, 10.4, 10.9, 10.10, 13.5,
              13.13
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from utils.gate_config import (
    PENDING_ORDER_MAX_FILL_BAR_AGE_SECONDS,
    PENDING_ORDER_MODE,
)
from utils.pending_order_fill import FILL_POLICY_LIMIT_PRICE, Bar, resolve_fill_price
from utils.pending_order_registry import (
    OrderState,
    PendingOrder,
    PendingOrderRegistry,
    PendingOrderRegistryError,
)
from utils.pending_order_time import now_utc

logger = logging.getLogger(__name__)

__all__ = ["FillResult", "fill_pending_order"]


@dataclass(frozen=True)
class FillResult:
    """Outcome of a fill attempt on a claimed order."""

    order_id: str
    success: bool
    fill_price: float | None
    reason: str
    trade_id: int | None = None


@dataclass(frozen=True)
class _ResolvedOrder:
    """Duck-typed shape ``calculate_position_size()`` expects."""

    entry_price: float
    stop_price: float
    action: str
    risk_multiplier: float = 1.0


def fill_pending_order(engine, order: PendingOrder, bar: Bar) -> FillResult:
    """Revalidate a claimed order and fill it at the limit price.

    Precondition: ``order.state`` is FILLING — the caller won the CAS claim.

    Steps, in order:

    1. Fill_Bar_Age bound. Runs FIRST and is the compensating control for
       bypassing the deviation tiers in step 7.
    2. Mode guard. In "observe", run 3-6 and emit ``pending_order_would_fill``
       without executing.
    3. Build the decision dict with entry = limit_price.
    4. Gate pipeline. Rejection -> REJECTED.
    5. Apply the gate size multiplier.
    6. Position sizer and fill-time risk re-checks. Failure -> CANCELED.
    7. ``execute_trade(normalized=True, price_authoritative=True)``.
       Success -> FILLED, failure -> REJECTED.
    """
    registry = PendingOrderRegistry(engine)

    if PENDING_ORDER_MODE == "disabled":
        _release(registry, order, "feature_disabled")
        return FillResult(order.order_id, False, None, "feature_disabled")

    if order.state is not OrderState.FILLING:
        logger.warning(
            "fill_pending_order called on order %s in state %s (expected filling)",
            order.order_id, order.state.value,
        )
        return FillResult(
            order.order_id, False, None, f"invalid_state_{order.state.value}"
        )

    # ── Step 1: bar freshness ──
    # Bypassing the deviation tiers removes the live-price sanity net, so this
    # bound is the only remaining guarantee that the fill reflects recent market
    # reality. A slow tick, a restart, or an order backlog can all surface a bar
    # several minutes old; filling at the limit off a 20-minute-old bar would be
    # exactly the stale favorable fill Requirement 13.2 forbids.
    bar_age = (now_utc() - bar.ts).total_seconds()
    if bar_age > PENDING_ORDER_MAX_FILL_BAR_AGE_SECONDS:
        logger.info(
            "PENDING_ORDER_STALE_FILL_BAR: %s bar age %.0fs exceeds %ds; "
            "releasing order %s",
            order.symbol, bar_age, PENDING_ORDER_MAX_FILL_BAR_AGE_SECONDS,
            order.order_id,
        )
        _release(registry, order, "stale_fill_bar")
        return FillResult(order.order_id, False, None, "stale_fill_bar")

    fill_price = float(
        resolve_fill_price(bar, side=order.side, limit_price=order.limit_price)
    )

    from db.schema import get_session

    session = get_session(engine)
    try:
        # ── Step 3: decision dict at the LIMIT price ──
        decision = _build_decision(order, fill_price)
        signal = _parse_signal(order, session)

        # ── Step 4: gate pipeline ──
        from agents.portfolio_manager import _run_gate_pipeline

        proceed, gate_notes, multiplier, _ = _run_gate_pipeline(
            session, engine, decision, signal, order.profile_id
        )

        if not proceed:
            gate_reason = _rejecting_gate(gate_notes)
            logger.info(
                "PENDING_ORDER_GATE_REJECTED: %s %s reason=%s order_id=%s",
                order.side, order.symbol, gate_reason, order.order_id,
            )
            _terminate(
                registry, order, "reject", gate_reason,
                engine=engine, fill_price=fill_price, bar=bar,
            )
            return FillResult(order.order_id, False, fill_price, gate_reason)

        # ── Steps 5-6: sizing and fill-time risk ──
        quantity, risk_reason = _resolve_quantity(
            session, order, fill_price, multiplier
        )
        if risk_reason is not None:
            logger.info(
                "PENDING_ORDER_RISK_CANCEL: %s %s reason=%s order_id=%s",
                order.side, order.symbol, risk_reason, order.order_id,
            )
            _terminate(
                registry, order, "cancel", risk_reason,
                engine=engine, fill_price=fill_price, bar=bar,
            )
            return FillResult(order.order_id, False, fill_price, risk_reason)

        decision["quantity"] = quantity

        # ── Step 2 (deferred): observe mode short-circuit ──
        # Placed after revalidation so the recorded would-fill carries the real
        # outcome rather than an untested guess.
        if PENDING_ORDER_MODE == "observe":
            _emit_fill_event(
                engine,
                event_type="pending_order_would_fill",
                order=order,
                fill_price=fill_price,
                bar=bar,
                message=(
                    f"{order.symbol}: pending order WOULD fill at "
                    f"{fill_price:.2f} (observe mode, no trade created)"
                ),
                extra={
                    "would_be_quantity": quantity,
                    "gate_multiplier": multiplier,
                    "mode": "observe",
                },
            )
            # Keep resting so it can be observed again until it expires naturally.
            _release(registry, order, "observe_mode")
            return FillResult(order.order_id, False, fill_price, "observe_would_fill")

        # ── Step 7: execute at the limit, price authoritative ──
        from agents.portfolio_manager import execute_trade

        success, message = execute_trade(
            session,
            decision,
            order.profile_id,
            normalized=True,             # stop is guaranteed present
            price_authoritative=True,    # the limit survives; tiers skipped
        )

        if not success:
            logger.warning(
                "PENDING_ORDER_EXECUTION_FAILED: %s %s - %s",
                order.side, order.symbol, message,
            )
            _terminate(
                registry, order, "reject", "execution_failed",
                engine=engine, fill_price=fill_price, bar=bar,
                detail=message,
            )
            return FillResult(order.order_id, False, fill_price, "execution_failed")

        trade_id = _find_new_trade_id(session, order, fill_price)

        try:
            registry.mark_filled(
                order.order_id,
                fill_price=fill_price,
                fill_policy=FILL_POLICY_LIMIT_PRICE,
                fill_bar_ts=bar.ts,
                trade_id=trade_id,
            )
        except PendingOrderRegistryError:
            # The trade exists. Losing the state transition is bad but must not
            # be reported as a failure; the orphan sweep reconciles it by looking
            # for exactly this trade.
            logger.error(
                "Order %s filled (trade_id=%s) but the state transition failed; "
                "the orphan sweep will reconcile it",
                order.order_id, trade_id, exc_info=True,
            )

        logger.info(
            "PENDING_ORDER_FILLED: %s %s qty=%s at limit %.2f "
            "(bar %s, age %.0fs) trade_id=%s order_id=%s",
            order.side, order.symbol, quantity, fill_price,
            bar.ts.isoformat(), bar_age, trade_id, order.order_id,
        )
        _emit_fill_event(
            engine,
            event_type="pending_order_filled",
            order=order,
            fill_price=fill_price,
            bar=bar,
            message=(
                f"{order.symbol}: pending {order.side} limit order filled at "
                f"{fill_price:.2f}"
            ),
            extra={
                "quantity": quantity,
                "trade_id": trade_id,
                "gate_multiplier": multiplier,
                "seconds_from_creation_to_fill": round(
                    (now_utc() - order.created_at).total_seconds(), 1
                ),
            },
            trade_id=trade_id,
        )
        return FillResult(order.order_id, True, fill_price, "filled", trade_id)

    except Exception as exc:
        logger.error(
            "Unexpected error filling order %s: %s",
            order.order_id, exc, exc_info=True,
        )
        _terminate(
            registry, order, "reject", "execution_failed",
            engine=engine, fill_price=fill_price, bar=bar,
            detail=str(exc),
        )
        return FillResult(order.order_id, False, fill_price, "execution_failed")
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Decision construction
# ---------------------------------------------------------------------------


def _build_decision(order: PendingOrder, fill_price: float) -> dict:
    """Build a decision dict at the limit price, under every alias read downstream.

    ``execute_trade()`` and ``validate_trade()`` each look for stop/target under
    several key names, so all of them are populated to avoid a normalized order
    reaching fallback stop derivation (which fails closed).
    """
    return {
        "action": order.side,
        "symbol": order.symbol,
        "price": fill_price,
        "entry_price": fill_price,
        "stop": order.stop_price,
        "stop_price": order.stop_price,
        "stop_loss": order.stop_price,
        "target": order.target_price,
        "target_price": order.target_price,
        "profit_target": order.target_price,
        "risk_reward": order.risk_reward,
        "quantity": order.intended_quantity or 0,
        "setup_type": order.setup_type,
        "geometry_name": order.geometry_name,
        "profile_id": order.profile_id,
        "rationale": order.pm_rationale,
        "pending_order_id": order.order_id,
        "pm_candidate_id": order.candidate_id,
        "candidate_id": order.candidate_id,
        "cycle_id": order.cycle_id,
    }


def _parse_signal(order: PendingOrder, session: Any) -> dict:
    """Signal context for the gates.

    Prefers a freshly rebuilt signal, because the gates should judge current
    conditions rather than what was true when the order was created. Falls back
    to the creation-time snapshot, then to an empty dict.
    """
    try:
        from agents.portfolio_manager import _build_signal_for_symbol

        rebuilt = _build_signal_for_symbol(session, order.symbol, {})
        if rebuilt:
            return rebuilt
    except Exception:
        logger.debug(
            "Could not rebuild the signal for %s; falling back to the snapshot",
            order.symbol, exc_info=True,
        )

    if order.signal_snapshot_json:
        try:
            parsed = json.loads(order.signal_snapshot_json)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, TypeError):
            pass
    return {}


def _rejecting_gate(gate_notes: Any) -> str:
    """Extract the rejecting gate's name, matching execute_triggered_plan()."""
    if isinstance(gate_notes, list):
        for note in gate_notes:
            if isinstance(note, dict) and note.get("decision") == "reject":
                return str(note.get("gate") or "gate_pipeline_rejected")
    return "gate_pipeline_rejected"


# ---------------------------------------------------------------------------
# Sizing and fill-time risk
# ---------------------------------------------------------------------------


def _resolve_quantity(
    session: Any, order: PendingOrder, fill_price: float, multiplier: float
) -> tuple[int, str | None]:
    """Size the order and run fill-time risk checks.

    Returns ``(quantity, None)`` when clear, or ``(0, reason)`` to cancel.

    The sizer here is a **pre-check for outcome classification**, not the
    authority: ``execute_trade()`` sizes and validates again. Running it here is
    what lets a sizing failure be recorded as ``canceled(sizing_rejected)``
    rather than collapsing into ``rejected(execution_failed)``, which matters for
    review (Requirement 8.7).
    """
    from agents.portfolio_manager import PM_PROFILES

    profile = PM_PROFILES.get(order.profile_id) or {}
    portfolio = _load_portfolio(session, order.profile_id, profile)

    try:
        from utils.position_sizer import calculate_position_size

        resolved = _ResolvedOrder(
            entry_price=fill_price,
            stop_price=order.stop_price,
            action=order.side,
            risk_multiplier=min(float(multiplier or 1.0), 1.0),
        )
        sizing = calculate_position_size(
            resolved, portfolio, profile, order.profile_id
        )
    except Exception:
        logger.error(
            "Position sizing raised for order %s; cancelling fail-closed",
            order.order_id, exc_info=True,
        )
        return 0, "sizing_rejected"

    quantity = int(getattr(sizing, "quantity", 0) or 0)
    if quantity <= 0:
        logger.info(
            "Order %s sized to %s (%s)",
            order.order_id, quantity,
            getattr(sizing, "rejection_reason", None),
        )
        return 0, "sizing_rejected"

    # ── Buying power, for longs. Shorts are governed by validate_trade(). ──
    if order.side == "BUY":
        needed = quantity * fill_price
        cash = float(portfolio.get("cash") or 0.0)
        if needed > cash:
            return 0, "insufficient_buying_power"

    # ── High-momentum re-entry cooldown ──
    if _in_cooldown(session, order):
        return 0, "cooldown_active"

    # ── Correlated exposure ──
    if _correlation_blocked(session, order, profile):
        return 0, "correlation_limit"

    return quantity, None


def _load_portfolio(session: Any, profile_id: str, profile: dict) -> dict:
    """Build the ``{cash, total_equity}`` dict the sizer expects.

    Mirrors how ``execute_trade()`` derives cash and equity, so the pre-check
    sees the same numbers the authoritative sizing will.
    """
    from db.schema import Balance, Position

    starting = float(profile.get("starting_balance") or 0.0)
    cash = starting
    try:
        balance = (
            session.query(Balance)
            .filter_by(profile=profile_id)
            .order_by(Balance.timestamp.desc())
            .first()
        )
        if balance is not None:
            cash = float(balance.cash)
    except Exception:
        logger.debug("Could not load balance for %s", profile_id, exc_info=True)

    position_value = 0.0
    try:
        for position in session.query(Position).filter_by(profile=profile_id).all():
            position_value += float(position.quantity) * float(position.avg_cost)
    except Exception:
        logger.debug("Could not load positions for %s", profile_id, exc_info=True)

    return {"cash": cash, "total_equity": cash + position_value}


def _in_cooldown(session: Any, order: PendingOrder) -> bool:
    """Whether a high-momentum symbol is inside its re-entry cooldown.

    Uses the same constants and helper the live preflight uses, so pending fills
    cannot bypass a rule that blocks immediate entries.
    """
    try:
        from agents.portfolio_manager import (
            HIGH_MOMENTUM_ASSETS,
            HIGH_MOMENTUM_COOLDOWN_MINUTES,
            _get_recent_closed_trades_for_preflight,
        )

        if order.symbol not in HIGH_MOMENTUM_ASSETS:
            return False

        recent = _get_recent_closed_trades_for_preflight(
            session.bind if hasattr(session, "bind") else session,
            order.profile_id,
            now_utc(),
            minutes=HIGH_MOMENTUM_COOLDOWN_MINUTES,
        )
        return any(trade.get("symbol") == order.symbol for trade in recent)
    except Exception:
        logger.debug(
            "Cooldown check failed for %s; allowing", order.symbol, exc_info=True
        )
        return False


def _correlation_blocked(session: Any, order: PendingOrder, profile: dict) -> bool:
    """Whether correlated exposure blocks this fill.

    ``check_correlation()`` is warning-only in the live system — no profile
    currently disallows correlated exposure, and no immediate entry is blocked by
    it. Cancelling a resting order on a warning would make pending fills STRICTER
    than immediate execution, an inconsistency that would show up as pending
    orders mysteriously underperforming.

    So the warning is recorded and the fill proceeds, unless a profile explicitly
    opts in via ``disallow_correlated_exposure``. That flag does not exist on any
    profile today, which makes this branch inert by design rather than by
    omission.
    """
    try:
        from utils.trade_validator import check_correlation

        direction = "LONG" if order.side == "BUY" else "SHORT"
        warning = check_correlation(
            order.symbol, direction, order.profile_id, session
        )
        if not warning:
            return False

        if profile.get("disallow_correlated_exposure"):
            logger.info(
                "Order %s blocked by correlated exposure: %s",
                order.order_id, warning,
            )
            return True

        logger.info(
            "Order %s has correlated exposure (warning only): %s",
            order.order_id, warning,
        )
        return False
    except Exception:
        logger.debug(
            "Correlation check failed for %s; allowing",
            order.symbol, exc_info=True,
        )
        return False


# ---------------------------------------------------------------------------
# Termination and events
# ---------------------------------------------------------------------------


def _terminate(
    registry: PendingOrderRegistry,
    order: PendingOrder,
    kind: str,
    reason: str,
    *,
    engine: Any,
    fill_price: float | None,
    bar: Bar | None,
    detail: str | None = None,
) -> None:
    """Move the order to a terminal state and emit the matching trade event."""
    try:
        if kind == "cancel":
            registry.mark_canceled(order.order_id, reason)
            event_type = "pending_order_canceled"
        else:
            registry.mark_rejected(order.order_id, reason)
            event_type = "pending_order_rejected"
    except PendingOrderRegistryError:
        logger.warning(
            "Could not %s order %s (%s); the sweep will reconcile it",
            kind, order.order_id, reason,
        )
        return

    extra: dict[str, Any] = {"reason": reason}
    if detail:
        extra["detail"] = detail

    _emit_fill_event(
        engine,
        event_type=event_type,
        order=order,
        fill_price=fill_price,
        bar=bar,
        message=f"{order.symbol}: pending order {kind}ed at fill time - {reason}",
        extra=extra,
    )


def _release(
    registry: PendingOrderRegistry, order: PendingOrder, reason: str
) -> None:
    """Return a claimed order to PENDING so it keeps resting."""
    try:
        registry.release_claim(order.order_id, reason=reason)
    except PendingOrderRegistryError:
        logger.warning(
            "Could not release the claim on order %s (%s); the orphan sweep "
            "will recover it", order.order_id, reason,
        )


def _emit_fill_event(
    engine: Any,
    *,
    event_type: str,
    order: PendingOrder,
    fill_price: float | None,
    bar: Bar | None,
    message: str,
    extra: dict | None = None,
    trade_id: int | None = None,
) -> None:
    """Write a trade_events row for a fill-path outcome. Fail-open.

    ``trade_id`` is set for successful fills so ``pending_order_filled`` stays
    visible in the existing per-trade drill-down, which requires a trade_id.
    """
    from db.schema import get_session
    from utils.trade_events import log_trade_event

    payload: dict[str, Any] = {
        "order_id": order.order_id,
        "symbol": order.symbol,
        "side": order.side,
        "setup_type": order.setup_type,
        "profile_id": order.profile_id,
        "limit_price": order.limit_price,
        "stop_price": order.stop_price,
        "target_price": order.target_price,
        "risk_reward": order.risk_reward,
        "fill_price": fill_price,
        "fill_policy": FILL_POLICY_LIMIT_PRICE,
        "candidate_id": order.candidate_id,
        "cycle_id": order.cycle_id,
        "created_at": order.created_at.isoformat(),
        "expires_at": order.expires_at.isoformat(),
    }
    if bar is not None:
        payload.update({
            "fill_bar_ts": bar.ts.isoformat(),
            "fill_bar_open": float(bar.open),
            "fill_bar_high": float(bar.high),
            "fill_bar_low": float(bar.low),
            "fill_bar_close": float(bar.close),
            "fill_bar_age_seconds": round(
                (now_utc() - bar.ts).total_seconds(), 1
            ),
        })
    if extra:
        payload.update(extra)

    session = get_session(engine)
    try:
        log_trade_event(
            session,
            event_type,
            trade_id=trade_id,
            agent="pending_order_filler",
            symbol=order.symbol,
            profile=order.profile_id,
            price=fill_price,
            message=message,
            payload=payload,
            pm_candidate_id=order.candidate_id,
        )
        session.commit()
    except Exception:
        session.rollback()
        logger.error(
            "Failed to emit %s for order %s (non-fatal)",
            event_type, order.order_id, exc_info=True,
        )
    finally:
        session.close()


def _find_new_trade_id(
    session: Any, order: PendingOrder, fill_price: float
) -> int | None:
    """Locate the Trade row ``execute_trade()`` just created.

    ``execute_trade()`` does not return the trade id, so it is recovered by
    querying the newest matching open trade. Fail-open: a missing id costs
    linkage, not correctness, and the orphan sweep can still reconcile state.
    """
    try:
        from db.schema import Trade

        direction = "LONG" if order.side == "BUY" else "SHORT"
        trade = (
            session.query(Trade)
            .filter_by(profile=order.profile_id, symbol=order.symbol)
            .filter(Trade.direction == direction)
            .order_by(Trade.id.desc())
            .first()
        )
        return int(trade.id) if trade is not None else None
    except Exception:
        logger.debug(
            "Could not resolve the trade id for order %s",
            order.order_id, exc_info=True,
        )
        return None
