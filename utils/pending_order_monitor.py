"""Deterministic monitor tick for pending limit orders.

Runs on its own APScheduler interval, independent of PM cycles, guarded by the
regular-market-hours check. Makes **no LLM calls** — every decision here is a
function of persisted order state and OHLC bars.

The tick order is fixed and load-bearing:

1. Sweep orphans (expire past-window PENDING, recover stranded FILLING).
2. Load PENDING orders.
3. Evaluate cancellation conditions **before** fill detection, so an order whose
   thesis just died is never filled in the same tick that would have killed it.
4. Fetch bars **once per unique symbol**, not once per order.
5. Detect crossings and advance the watermark — the watermark advances whether or
   not a crossing was found.
6. Gap-through cancels; a genuine crossing claims the order via CAS and hands it
   to the filler.

Failure direction: fail-OPEN throughout. A provider outage, a ragged payload, or
an evaluation error leaves orders PENDING for the next tick and never produces a
terminal state. Only explicit conditions terminate an order.

Requirements: 4.1, 4.2, 4.3, 4.9, 4.12, 4.13, 4.14, 4.15, 7.1, 7.2, 7.5, 7.10,
              3.7, 3.10, 9.12
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from utils.gate_config import (
    PENDING_ORDER_BAR_RESOLUTION,
    PENDING_ORDER_MAX_GAP_THROUGH_PCT,
    PENDING_ORDER_MODE,
)
from utils.pending_order_fill import (
    Bar,
    bars_from_candles,
    detect_crossing,
    eligible_bars,
)
from utils.pending_order_registry import (
    OrderState,
    PendingOrder,
    PendingOrderRegistry,
    PendingOrderRegistryError,
)
from utils.pending_order_time import now_utc

logger = logging.getLogger(__name__)

# Lease after which a FILLING order is presumed stranded by a crashed tick.
FILLING_LEASE_MINUTES = 5

__all__ = ["MonitorTickResult", "PendingOrderMonitor", "run"]


@dataclass(frozen=True)
class MonitorTickResult:
    """Per-tick telemetry."""

    orders_checked: int = 0
    orders_filled: int = 0
    orders_expired: int = 0
    orders_canceled: int = 0
    bars_fetched: int = 0
    symbols_fetched: int = 0
    tick_duration_ms: float = 0.0

    @property
    def had_activity(self) -> bool:
        return bool(
            self.orders_filled or self.orders_expired or self.orders_canceled
        )


def run(engine) -> MonitorTickResult:
    """Execute one monitor tick. Never raises."""
    return PendingOrderMonitor(engine).tick()


class PendingOrderMonitor:
    """Evaluates resting orders against fresh OHLC bars."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._registry = PendingOrderRegistry(engine)

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self) -> MonitorTickResult:
        started = time.perf_counter()

        if PENDING_ORDER_MODE == "disabled":
            return MonitorTickResult()

        expired = 0
        canceled = 0
        filled = 0
        checked = 0
        bars_fetched = 0
        symbols_fetched = 0

        try:
            # ── Step 1: load PENDING orders ──
            # Deliberately before the orphan sweep: expiring orders are handled
            # here so their expiry event can carry near-miss telemetry computed
            # from the same bars the survivors use. The sweep still runs at the
            # end as the Layer-2 hard guarantee.
            try:
                orders = self._registry.get_pending_orders()
            except Exception:
                logger.error(
                    "Could not load pending orders this tick (non-fatal)",
                    exc_info=True,
                )
                return self._result(started)

            checked = len(orders)
            now = now_utc()
            expiring = [o for o in orders if o.expires_at <= now]
            live = [o for o in orders if o.expires_at > now]

            # ── Step 2: cancellation checks, BEFORE any fill detection ──
            survivors: list[PendingOrder] = []
            for order in live:
                reason = self._cancellation_reason(order)
                if reason is None:
                    survivors.append(order)
                    continue
                if self._cancel(order, reason):
                    canceled += 1

            # ── Step 3: fetch bars once per unique symbol ──
            needed = {o.symbol for o in expiring} | {o.symbol for o in survivors}
            if needed:
                bars_by_symbol, bars_fetched, symbols_fetched = self._fetch_bars(
                    needed
                )
            else:
                bars_by_symbol = {}

            # ── Step 4: expire elapsed windows, with near-miss telemetry ──
            for order in expiring:
                if self._expire(order, bars_by_symbol.get(order.symbol)):
                    expired += 1

            # ── Steps 5-6: crossing detection and dispatch ──
            for order in survivors:
                bars = bars_by_symbol.get(order.symbol)
                if not bars:
                    # Provider unavailable or empty for this symbol. Leave the
                    # order PENDING; retry next tick.
                    continue

                outcome = self._evaluate(order, bars)
                if outcome == "filled":
                    filled += 1
                elif outcome == "canceled":
                    canceled += 1

            # ── Step 7: Layer-2 sweep ──
            # Recovers orders stranded in FILLING by a crashed tick, and catches
            # any expiry the loop above could not resolve. Normally a no-op.
            try:
                resolved = self._registry.finalize_orphaned_orders(
                    filling_lease_minutes=FILLING_LEASE_MINUTES
                )
                expired += sum(
                    1 for s in resolved.values() if s is OrderState.EXPIRED
                )
            except Exception:
                logger.error(
                    "Pending order orphan sweep failed this tick (non-fatal)",
                    exc_info=True,
                )

        except Exception:
            # Belt and braces: the scheduler job also catches, but a monitor
            # tick must never propagate.
            logger.error("Pending order monitor tick failed", exc_info=True)

        return self._result(
            started,
            orders_checked=checked,
            orders_filled=filled,
            orders_expired=expired,
            orders_canceled=canceled,
            bars_fetched=bars_fetched,
            symbols_fetched=symbols_fetched,
        )

    # ------------------------------------------------------------------
    # Expiry
    # ------------------------------------------------------------------

    def _expire(self, order: PendingOrder, bars: list[Bar] | None) -> bool:
        """Expire an elapsed order, recording how close price came to the limit.

        The near-miss figures are recomputed over the order's whole active
        window rather than tracked incrementally, so no extra column is needed
        and the number is exact rather than a running approximation.
        """
        closest_price = None
        closest_distance = None

        if bars:
            try:
                windowed = eligible_bars(
                    bars,
                    created_at=order.created_at,
                    expires_at=order.expires_at,
                    watermark=None,  # whole window, not just unseen bars
                )
                if windowed:
                    result = detect_crossing(
                        windowed,
                        side=order.side,
                        limit_price=Decimal(str(order.limit_price)),
                        gap_through_pct=Decimal(
                            str(PENDING_ORDER_MAX_GAP_THROUGH_PCT)
                        ),
                    )
                    closest_price = result.closest_approach_price
                    closest_distance = result.closest_approach_distance
            except Exception:
                logger.debug(
                    "Could not compute near-miss telemetry for order %s",
                    order.order_id, exc_info=True,
                )

        try:
            self._registry.mark_expired(order.order_id, reason="window_elapsed")
        except PendingOrderRegistryError:
            logger.warning(
                "Could not expire order %s; the sweep will retry",
                order.order_id,
            )
            return False

        logger.info(
            "PENDING_ORDER_EXPIRED: %s %s limit=%.2f closest=%s order_id=%s",
            order.side, order.symbol, order.limit_price,
            f"{closest_price:.2f}" if closest_price is not None else "unknown",
            order.order_id,
        )
        _emit_order_trade_event(
            self._engine,
            event_type="pending_order_expired",
            order=order,
            price=order.limit_price,
            message=(
                f"{order.symbol}: pending order expired without filling "
                f"(limit {order.limit_price:.2f}, closest approach "
                f"{closest_price if closest_price is not None else 'unknown'})"
            ),
            extra={
                "reason": "window_elapsed",
                "expires_at": order.expires_at.isoformat(),
                "created_at": order.created_at.isoformat(),
                "closest_approach_price": (
                    float(closest_price) if closest_price is not None else None
                ),
                "closest_approach_distance": (
                    float(closest_distance)
                    if closest_distance is not None else None
                ),
            },
        )
        return True

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def _cancellation_reason(self, order: PendingOrder) -> str | None:
        """Whether this order's thesis or portfolio state has invalidated it.

        Evaluated before fill detection (Requirement 7.10). Fail-open: any
        lookup error returns None so the order keeps resting — a failed read is
        not evidence against the thesis.

        Cooldown, correlation, and buying-power checks deliberately live in the
        filler instead, because they are only meaningful against the concrete
        quantity and cash state at fill time.
        """
        try:
            signal = self._latest_signal(order.symbol)
            if signal is not None:
                if self._signal_flipped(order.side, signal):
                    return "signal_flipped"
                if self._signal_invalidated(signal):
                    return "signal_invalidated"

            if self._position_already_open(order):
                return "position_already_open"
        except Exception:
            logger.debug(
                "Cancellation checks failed for order %s; leaving it pending",
                order.order_id, exc_info=True,
            )
            return None

        return None

    def _latest_signal(self, symbol: str) -> dict | None:
        """Newest Analyst signal for a symbol, or None.

        Same source ``execute_trade()`` uses for fallback stop derivation:
        AgentMemory(agent="analyst", key="signal"), newest first.
        """
        from db.schema import AgentMemory, get_session

        session = get_session(self._engine)
        try:
            row = (
                session.query(AgentMemory)
                .filter_by(agent="analyst", symbol=symbol, key="signal")
                .order_by(AgentMemory.timestamp.desc())
                .first()
            )
            if row is None or not row.value:
                return None
            parsed = json.loads(row.value)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            # A missing or unparseable signal is NOT a flip. Fail-open.
            logger.debug(
                "Could not read the latest analyst signal for %s", symbol,
                exc_info=True,
            )
            return None
        finally:
            session.close()

    @staticmethod
    def _signal_flipped(side: str, signal: dict) -> bool:
        """Whether the Analyst now calls the opposite direction.

        ``signal["signal"]`` is "LONG" | "SHORT" | "HOLD"; order side is
        "BUY" | "SHORT".

        HOLD is deliberately NOT treated as a flip. The Analyst emits HOLD as its
        default for any symbol without an active directional call, so treating it
        as invalidation would cancel nearly every resting order within one
        analyst cycle and defeat the feature. HOLD is a withdrawal of conviction,
        not opposition.

        Quarantine markers ("error", "malformed_analyst_output",
        "unclear_direction") are also not flips — they signal a data or parsing
        problem, and fail-open on evaluation is the rule.
        """
        direction = str(signal.get("signal") or "").strip().upper()
        if direction not in {"LONG", "SHORT"}:
            return False

        # A data-quality quarantine is not a directional opinion.
        setup_type = str(signal.get("setup_type") or "").strip().lower()
        if setup_type in {
            "error", "malformed_analyst_output", "unclear_direction"
        }:
            return False
        if signal.get("data_unavailable"):
            return False

        if side == "BUY":
            return direction == "SHORT"
        return direction == "LONG"

    @staticmethod
    def _signal_invalidated(signal: dict) -> bool:
        """Whether the signal carries an explicit invalidation marker.

        A forward-looking hook: no current Analyst field sets these, so this is
        effectively inert today. It exists so that when an explicit invalidation
        flag is introduced, the cancellation path already honors it rather than
        needing a change here.
        """
        for key in ("invalidated", "thesis_invalidated", "signal_invalidated"):
            if signal.get(key):
                return True
        return False

    def _position_already_open(self, order: PendingOrder) -> bool:
        """Whether a position for this symbol and direction is already open.

        ``Position.side`` is lowercase "long"/"short"; order side is
        "BUY"/"SHORT".
        """
        from db.schema import Position, get_session

        expected_side = "long" if order.side == "BUY" else "short"

        session = get_session(self._engine)
        try:
            existing = (
                session.query(Position)
                .filter_by(profile=order.profile_id, symbol=order.symbol)
                .all()
            )
            for position in existing:
                side = str(getattr(position, "side", "") or "").lower()
                quantity = getattr(position, "quantity", 0) or 0
                if side == expected_side and quantity > 0:
                    return True
            return False
        except Exception:
            logger.debug(
                "Could not check open positions for %s", order.symbol,
                exc_info=True,
            )
            return False
        finally:
            session.close()

    def _cancel(self, order: PendingOrder, reason: str) -> bool:
        """Cancel an order and emit its trade event. Returns True on success."""
        try:
            self._registry.mark_canceled(order.order_id, reason)
        except PendingOrderRegistryError:
            logger.warning(
                "Could not cancel order %s (%s); it may have already "
                "transitioned", order.order_id, reason,
            )
            return False

        logger.info(
            "PENDING_ORDER_CANCELED: %s %s reason=%s order_id=%s",
            order.side, order.symbol, reason, order.order_id,
        )
        _emit_order_trade_event(
            self._engine,
            event_type="pending_order_canceled",
            order=order,
            price=order.limit_price,
            message=f"{order.symbol}: pending order canceled - {reason}",
            extra={"reason": reason},
        )
        return True

    # ------------------------------------------------------------------
    # Bar fetching
    # ------------------------------------------------------------------

    def _fetch_bars(
        self, symbols: set[str]
    ) -> tuple[dict[str, list[Bar]], int, int]:
        """Fetch bars once per unique symbol into a tick-local cache.

        Routes through ``FinnhubClient.get_candles()``, which tries Alpaca first
        for sub-daily resolutions. Only the Finnhub fallback consumes the shared
        quote budget, so the common path does not compete with the price monitor.

        Fail-open per symbol: one symbol's failure never affects the others.
        """
        results: dict[str, list[Bar]] = {}
        total_bars = 0
        fetched = 0

        try:
            from utils.finnhub_client import FinnhubClient

            client = FinnhubClient()
        except Exception:
            logger.error(
                "Could not construct a market data client this tick (non-fatal)",
                exc_info=True,
            )
            return results, 0, 0

        for symbol in sorted(symbols):
            try:
                candles = client.get_candles(
                    symbol, resolution=PENDING_ORDER_BAR_RESOLUTION, days=1
                )
                fetched += 1
                bars = bars_from_candles(candles)
                if bars:
                    results[symbol] = bars
                    total_bars += len(bars)
                else:
                    logger.debug("No usable bars for %s this tick", symbol)
            except Exception:
                logger.warning(
                    "Bar fetch failed for %s this tick; order(s) stay pending",
                    symbol, exc_info=True,
                )

        return results, total_bars, fetched

    # ------------------------------------------------------------------
    # Evaluation and dispatch
    # ------------------------------------------------------------------

    def _evaluate(self, order: PendingOrder, bars: list[Bar]) -> str | None:
        """Evaluate one order against a symbol's bars.

        Returns "filled", "canceled", or None.
        """
        try:
            windowed = eligible_bars(
                bars,
                created_at=order.created_at,
                expires_at=order.expires_at,
                watermark=order.last_evaluated_bar_ts,
            )
        except Exception:
            logger.warning(
                "Window filtering failed for order %s; leaving it pending",
                order.order_id, exc_info=True,
            )
            return None

        if not windowed:
            return None

        try:
            result = detect_crossing(
                windowed,
                side=order.side,
                limit_price=Decimal(str(order.limit_price)),
                gap_through_pct=Decimal(str(PENDING_ORDER_MAX_GAP_THROUGH_PCT)),
            )
        except Exception:
            logger.warning(
                "Crossing detection failed for order %s; leaving it pending",
                order.order_id, exc_info=True,
            )
            return None

        # Advance the watermark regardless of outcome, so the next tick does not
        # rescan these bars.
        if result.newest_bar_ts is not None:
            self._registry.advance_watermark(order.order_id, result.newest_bar_ts)

        if not result.crossed:
            return None

        if result.gap_through:
            # The market jumped past the level rather than trading down to it,
            # which invalidates the stop and target derived from the pre-gap
            # structure. Cancel rather than fill at a stale geometry.
            logger.info(
                "PENDING_ORDER_GAP_THROUGH: %s %s limit=%.2f bar_open=%s",
                order.side, order.symbol, order.limit_price,
                result.bar.open if result.bar else None,
            )
            return "canceled" if self._cancel(order, "gap_through") else None

        # ── CAS claim: at most one fill attempt per order ──
        claimed, claim_reason = self._registry.claim_for_fill(order.order_id)
        if not claimed:
            logger.debug(
                "Order %s was claimed by another tick (%s)",
                order.order_id, claim_reason,
            )
            return None

        # Re-load after the claim. The in-memory record still says PENDING, and
        # the filler asserts it has been handed a FILLING order — handing over
        # the stale copy would trip that precondition and silently skip the fill.
        claimed_order = self._registry.get_order(order.order_id) or order
        return self._dispatch_fill(claimed_order, result.bar)

    def _dispatch_fill(self, order: PendingOrder, bar: Bar | None) -> str | None:
        """Hand a claimed order to the filler.

        If dispatch itself explodes, release the claim so the order is not
        stranded in FILLING until the lease expires.
        """
        if bar is None:  # pragma: no cover - defensive
            self._safe_release(order, "missing_crossing_bar")
            return None

        try:
            from utils.pending_order_filler import fill_pending_order

            result = fill_pending_order(self._engine, order, bar)
        except Exception:
            logger.error(
                "Fill dispatch failed for order %s; releasing the claim",
                order.order_id, exc_info=True,
            )
            self._safe_release(order, "fill_dispatch_error")
            return None

        if getattr(result, "success", False):
            return "filled"
        if getattr(result, "reason", None) in {"canceled", "sizing_rejected"}:
            return "canceled"
        return None

    def _safe_release(self, order: PendingOrder, reason: str) -> None:
        try:
            self._registry.release_claim(order.order_id, reason=reason)
        except PendingOrderRegistryError:
            logger.warning(
                "Could not release the claim on order %s; the orphan sweep "
                "will recover it", order.order_id,
            )

    # ------------------------------------------------------------------
    # Result assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _result(started: float, **counts: Any) -> MonitorTickResult:
        return MonitorTickResult(
            tick_duration_ms=(time.perf_counter() - started) * 1000.0,
            **counts,
        )


# ---------------------------------------------------------------------------
# Shared event helper
# ---------------------------------------------------------------------------


def _emit_order_trade_event(
    engine: Any,
    *,
    event_type: str,
    order: PendingOrder,
    price: float | None,
    message: str,
    extra: dict | None = None,
) -> None:
    """Write a trade_events row for an order lifecycle transition.

    Uses a dedicated session, because the monitor runs outside any PM
    transaction. Fail-open.
    """
    from db.schema import get_session
    from utils.trade_events import log_trade_event

    payload = {
        "order_id": order.order_id,
        "symbol": order.symbol,
        "side": order.side,
        "setup_type": order.setup_type,
        "profile_id": order.profile_id,
        "limit_price": order.limit_price,
        "stop_price": order.stop_price,
        "target_price": order.target_price,
        "candidate_id": order.candidate_id,
        "cycle_id": order.cycle_id,
    }
    if extra:
        payload.update(extra)

    session = get_session(engine)
    try:
        log_trade_event(
            session,
            event_type,
            agent="pending_order_monitor",
            symbol=order.symbol,
            profile=order.profile_id,
            price=price,
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
