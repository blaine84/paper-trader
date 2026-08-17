"""Pending limit order persistence and CAS state machine.

Structural sibling of ``utils/trade_plan_registry.py``: every state transition
is a compare-and-swap (``UPDATE ... WHERE state = :expected``) with rowcount
verification, so two concurrent monitor ticks can never both act on one order.

Permitted transitions:

    PENDING  -> FILLING     (CAS claim for a fill attempt)
    PENDING  -> EXPIRED     (active window elapsed)
    PENDING  -> CANCELED    (thesis died, superseded, risk changed)
    FILLING  -> FILLED      (execute_trade succeeded)
    FILLING  -> REJECTED    (gate rejection or execution failure)
    FILLING  -> CANCELED    (fill-time risk re-check failed)
    FILLING  -> PENDING     (release on transient failure)

Terminal states are final: FILLED, EXPIRED, CANCELED, REJECTED.

``FILLING`` extends the source spec's five-state model. It exists solely to make
fill claiming atomic — without it, two overlapping ticks could both fill one
order. It is swept on a lease timeout by :meth:`finalize_orphaned_orders`.

Failure direction: state transitions fail CLOSED (raise
``PendingOrderRegistryError``), while event emission fails OPEN so an audit
write can never block a state change. :meth:`claim_for_fill` returns a bool
rather than raising, because a lost CAS race is an expected outcome in a
polling monitor rather than an error.

Requirements: 2.1-2.7, 3.9, 3.10, 4.9, 7.3, 7.4, 8.4, 9.1-9.12, 10.8
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from sqlalchemy import text

from utils.db_retry import with_lock_retry
from utils.pending_order_time import now_utc, to_iso, to_utc

logger = logging.getLogger(__name__)


class OrderState(Enum):
    """Lifecycle states for a pending limit order."""

    PENDING = "pending"       # resting, awaiting a crossing
    FILLING = "filling"       # transient CAS claim for one fill attempt
    FILLED = "filled"         # paper fill created
    EXPIRED = "expired"       # active window elapsed without a crossing
    CANCELED = "canceled"     # thesis or risk state invalidated it
    REJECTED = "rejected"     # hard gate failure or execution failure


TERMINAL_STATES = frozenset({
    OrderState.FILLED,
    OrderState.EXPIRED,
    OrderState.CANCELED,
    OrderState.REJECTED,
})

TRANSIENT_STATES = frozenset({OrderState.PENDING, OrderState.FILLING})

# The exact set of transitions the machine allows. Anything else is a bug.
PERMITTED_TRANSITIONS: frozenset[tuple[OrderState, OrderState]] = frozenset({
    (OrderState.PENDING, OrderState.FILLING),
    (OrderState.PENDING, OrderState.EXPIRED),
    (OrderState.PENDING, OrderState.CANCELED),
    (OrderState.FILLING, OrderState.FILLED),
    (OrderState.FILLING, OrderState.REJECTED),
    (OrderState.FILLING, OrderState.CANCELED),
    (OrderState.FILLING, OrderState.PENDING),
})

# Closed vocabulary — cancellation reasons are machine-readable, never free text.
CANCEL_REASONS = frozenset({
    "signal_flipped",
    "signal_invalidated",
    "superseded",
    "position_already_open",
    "cooldown_active",
    "correlation_limit",
    "insufficient_buying_power",
    "gap_through",
    "sizing_rejected",
})

# Reasons an order was never created in the first place.
DECLINE_REASONS = frozenset({
    "target_already_exceeded",
    "runaway_exceeds_max",
    "incomplete_geometry",
    "invalid_geometry_at_limit",
    "window_too_short",
    "active_order_cap_reached",
    "duplicate_active_order",
    "repaired_before_check",
})

# Columns on pending_orders, in declaration order. Single source of truth for
# INSERT and SELECT so the two can never drift apart.
_ORDER_COLUMNS: tuple[str, ...] = (
    "order_id",
    "profile_id",
    "symbol",
    "side",
    "setup_type",
    "geometry_name",
    "candidate_id",
    "cycle_id",
    "source_signal_id",
    "plan_id",
    "limit_price",
    "stop_price",
    "target_price",
    "risk_reward",
    "intended_quantity",
    "fresh_price_at_creation",
    "runaway_pct_at_creation",
    "pm_rationale",
    "signal_snapshot_json",
    "state",
    "created_at",
    "expires_at",
    "last_evaluated_bar_ts",
    "filled_at",
    "terminal_at",
    "fill_price",
    "fill_policy",
    "fill_bar_ts",
    "terminal_reason",
    "trade_id",
    "integrity_hash",
)

_DATETIME_FIELDS = frozenset({
    "created_at", "expires_at", "last_evaluated_bar_ts",
    "filled_at", "terminal_at", "fill_bar_ts",
})

__all__ = [
    "CANCEL_REASONS",
    "DECLINE_REASONS",
    "OrderState",
    "PERMITTED_TRANSITIONS",
    "PendingOrder",
    "PendingOrderRegistry",
    "PendingOrderRegistryError",
    "TERMINAL_STATES",
    "TRANSIENT_STATES",
    "compute_order_integrity_hash",
]


@dataclass(frozen=True)
class PendingOrder:
    """Immutable pending limit order record.

    All linkage fields default to None: the live PM path runs with
    PM_CANDIDATE_MODE disabled and therefore produces no candidate_id or cycle_id.
    plan_id is permanently None after the triggered-plan retirement.
    """

    order_id: str
    profile_id: str
    symbol: str
    side: str                       # "BUY" | "SHORT"
    setup_type: str

    limit_price: float              # the intended ENTRY, never the target
    stop_price: float
    target_price: float
    risk_reward: float

    fresh_price_at_creation: float
    runaway_pct_at_creation: float

    created_at: datetime
    expires_at: datetime

    state: OrderState = OrderState.PENDING

    geometry_name: str | None = None

    # Linkage — all nullable
    candidate_id: str | None = None
    cycle_id: str | None = None
    source_signal_id: str | None = None
    # Retained for historical/linkage compatibility only. Always None for orders
    # created after the triggered-plan architecture was retired (2025-01).
    plan_id: str | None = None

    intended_quantity: int | None = None
    pm_rationale: str | None = None
    signal_snapshot_json: str | None = None

    # Lifecycle bookkeeping
    last_evaluated_bar_ts: datetime | None = None
    filled_at: datetime | None = None
    terminal_at: datetime | None = None
    fill_price: float | None = None
    fill_policy: str | None = None
    fill_bar_ts: datetime | None = None
    terminal_reason: str | None = None
    trade_id: int | None = None

    integrity_hash: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def active_key(self) -> tuple[str, str, str, str]:
        """The tuple constrained to one active order at a time."""
        return (self.profile_id, self.symbol, self.side, self.setup_type)


def compute_order_integrity_hash(order: PendingOrder) -> str:
    """SHA-256 over canonical JSON of identity and geometry fields.

    Mirrors ``_compute_plan_integrity_hash`` and ``_compute_integrity_hash``:
    covers exactly the fields that must never change after creation.
    """
    identity = {
        "order_id": order.order_id,
        "profile_id": order.profile_id,
        "symbol": order.symbol,
        "side": order.side,
        "setup_type": order.setup_type,
        "limit_price": order.limit_price,
        "stop_price": order.stop_price,
        "target_price": order.target_price,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class PendingOrderRegistryError(Exception):
    """Raised when a pending order registry operation fails closed."""


class PendingOrderRegistry:
    """Manages pending order lifecycle with CAS state transitions."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def create_order(self, order: PendingOrder) -> str:
        """INSERT a new order with state=PENDING. Returns the order_id.

        Fails closed. A duplicate active key raises via the partial UNIQUE
        index ``idx_pending_orders_active_key``, which is the correct outcome:
        the caller supersedes existing orders before inserting, so a collision
        means a genuine race.
        """
        if order.state is not OrderState.PENDING:
            raise PendingOrderRegistryError(
                f"New orders must start PENDING, got {order.state.value}"
            )

        stamped = order
        if not stamped.integrity_hash:
            stamped = replace(
                order, integrity_hash=compute_order_integrity_hash(order)
            )

        try:
            self._execute_create_write(stamped)
        except PendingOrderRegistryError:
            raise
        except Exception as e:
            logger.error("Failed to create pending order %s: %s", order.order_id, e)
            raise PendingOrderRegistryError(
                f"Failed to create pending order {order.order_id}: {e}"
            ) from e

        self._emit_order_event(
            order_id=stamped.order_id,
            profile_id=stamped.profile_id,
            symbol=stamped.symbol,
            event_type="state_pending",
            to_state=OrderState.PENDING,
            reference_price=stamped.fresh_price_at_creation,
            payload={
                "limit_price": stamped.limit_price,
                "stop_price": stamped.stop_price,
                "target_price": stamped.target_price,
                "expires_at": to_iso(stamped.expires_at),
                "runaway_pct_at_creation": stamped.runaway_pct_at_creation,
            },
        )
        return stamped.order_id

    @with_lock_retry
    def _execute_create_write(self, order: PendingOrder) -> None:
        columns = ", ".join(_ORDER_COLUMNS)
        placeholders = ", ".join(f":{c}" for c in _ORDER_COLUMNS)
        with self._engine.connect() as conn:
            conn.execute(
                text(
                    f"INSERT INTO pending_orders ({columns}) "
                    f"VALUES ({placeholders})"
                ),
                _order_to_params(order),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def claim_for_fill(self, order_id: str) -> tuple[bool, str | None]:
        """CAS PENDING -> FILLING, claiming the order for one fill attempt.

        Returns ``(True, None)`` on success and ``(False, reason)`` when the CAS
        lost — deliberately not an exception, because losing a race to another
        tick is an expected outcome in a polling monitor. Mirrors
        ``CandidateRegistry.reserve()``.

        Raises:
            PendingOrderRegistryError: On an actual persistence error.
        """
        try:
            rowcount = self._execute_state_write(
                order_id,
                OrderState.PENDING,
                OrderState.FILLING,
            )
        except Exception as e:
            reason = f"Persistence error claiming order {order_id}: {e}"
            logger.error(reason)
            raise PendingOrderRegistryError(reason) from e

        if rowcount == 1:
            self._emit_order_event_for(
                order_id,
                event_type="state_filling",
                from_state=OrderState.PENDING,
                to_state=OrderState.FILLING,
            )
            return (True, None)

        reason = (
            f"CAS failed claiming order {order_id}: expected state=pending, "
            f"rowcount={rowcount}"
        )
        logger.debug(reason)
        return (False, reason)

    def release_claim(self, order_id: str, reason: str) -> None:
        """FILLING -> PENDING. Used when a fill attempt cannot proceed yet.

        Applies to a stale crossing bar, an unavailable quote, or observe mode —
        conditions where the order should keep resting rather than terminate.
        """
        self._transition(
            order_id,
            OrderState.FILLING,
            OrderState.PENDING,
            reason=reason,
        )

    def mark_filled(
        self,
        order_id: str,
        *,
        fill_price: float,
        fill_policy: str,
        fill_bar_ts: datetime,
        trade_id: int | None = None,
        filled_at: datetime | None = None,
    ) -> None:
        """FILLING -> FILLED, recording how and when the fill happened."""
        stamp = to_utc(filled_at) or now_utc()
        self._transition(
            order_id,
            OrderState.FILLING,
            OrderState.FILLED,
            reason="filled",
            reference_price=fill_price,
            extra_fields={
                "filled_at": to_iso(stamp),
                "terminal_at": to_iso(stamp),
                "fill_price": float(fill_price),
                "fill_policy": fill_policy,
                "fill_bar_ts": to_iso(fill_bar_ts),
                "trade_id": trade_id,
            },
        )

    def mark_rejected(self, order_id: str, reason: str) -> None:
        """FILLING -> REJECTED. Reserved for hard gate and execution failures.

        Never used for "price moved away" — that is an order's normal resting
        state, not a rejection (Requirement 9.6).
        """
        stamp = now_utc()
        self._transition(
            order_id,
            OrderState.FILLING,
            OrderState.REJECTED,
            reason=reason,
            extra_fields={
                "terminal_at": to_iso(stamp),
                "terminal_reason": reason,
            },
        )

    def mark_canceled(self, order_id: str, reason: str) -> None:
        """PENDING|FILLING -> CANCELED.

        Tries PENDING first, then FILLING, following the ``mark_missed()``
        cascade in TradePlanRegistry. Raises if the order is already terminal.
        """
        if reason not in CANCEL_REASONS:
            logger.warning(
                "Cancel reason %r is outside the closed vocabulary; recording "
                "it anyway but reporting will not group it",
                reason,
            )

        stamp = now_utc()
        extra = {
            "terminal_at": to_iso(stamp),
            "terminal_reason": reason,
        }

        try:
            self._transition(
                order_id, OrderState.PENDING, OrderState.CANCELED,
                reason=reason, extra_fields=extra,
            )
            return
        except PendingOrderRegistryError:
            pass

        self._transition(
            order_id, OrderState.FILLING, OrderState.CANCELED,
            reason=reason, extra_fields=extra,
        )

    def mark_expired(self, order_id: str, reason: str = "window_elapsed") -> None:
        """PENDING -> EXPIRED. The active window elapsed without a crossing."""
        stamp = now_utc()
        self._transition(
            order_id,
            OrderState.PENDING,
            OrderState.EXPIRED,
            reason=reason,
            extra_fields={
                "terminal_at": to_iso(stamp),
                "terminal_reason": reason,
            },
        )

    # ------------------------------------------------------------------
    # Watermark bookkeeping
    # ------------------------------------------------------------------

    def advance_watermark(self, order_id: str, bar_ts: datetime) -> None:
        """Record the newest bar evaluated for this order.

        Deliberately NOT a state transition and emits no event: this runs on
        every tick that observes bars, so an event per call would swamp the
        audit trail. Never moves the watermark backwards.
        """
        normalized = to_utc(bar_ts)
        if normalized is None:
            return
        try:
            self._execute_watermark_write(order_id, to_iso(normalized))
        except Exception as e:
            # Fail-open: a lost watermark costs a redundant rescan next tick,
            # which detect_crossing() handles idempotently. It is not worth
            # failing a tick over.
            logger.warning(
                "Could not advance watermark for order %s: %s", order_id, e
            )

    @with_lock_retry
    def _execute_watermark_write(self, order_id: str, bar_ts_iso: str) -> int:
        with self._engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    UPDATE pending_orders
                    SET last_evaluated_bar_ts = :bar_ts
                    WHERE order_id = :order_id
                      AND (last_evaluated_bar_ts IS NULL
                           OR last_evaluated_bar_ts < :bar_ts)
                    """
                ),
                {"order_id": order_id, "bar_ts": bar_ts_iso},
            )
            conn.commit()
            return result.rowcount

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_order(self, order_id: str) -> PendingOrder | None:
        """Look up one order by ID, or None."""
        columns = ", ".join(_ORDER_COLUMNS)
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    f"SELECT {columns} FROM pending_orders "
                    f"WHERE order_id = :order_id"
                ),
                {"order_id": order_id},
            ).mappings().fetchone()
        return _row_to_order(row) if row else None

    def get_pending_orders(self, symbol: str | None = None) -> list[PendingOrder]:
        """Orders in PENDING only — what the monitor evaluates each tick."""
        return self._select_by_states([OrderState.PENDING], symbol=symbol)

    def get_active_orders(self, symbol: str | None = None) -> list[PendingOrder]:
        """Orders in any non-terminal state (PENDING or FILLING)."""
        return self._select_by_states(
            [OrderState.PENDING, OrderState.FILLING], symbol=symbol
        )

    def get_orders_for_profile(
        self, profile_id: str, *, include_terminal: bool = False, limit: int = 200
    ) -> list[PendingOrder]:
        """Orders for one profile, newest first. Powers the dashboard."""
        columns = ", ".join(_ORDER_COLUMNS)
        clause = "" if include_terminal else " AND state IN ('pending', 'filling')"
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT {columns} FROM pending_orders "
                    f"WHERE profile_id = :profile_id{clause} "
                    f"ORDER BY created_at DESC LIMIT :limit"
                ),
                {"profile_id": profile_id, "limit": limit},
            ).mappings().all()
        return [_row_to_order(r) for r in rows]

    def count_active_for_profile(self, profile_id: str) -> int:
        """Active (non-terminal) order count, for the per-profile cap."""
        with self._engine.connect() as conn:
            return int(
                conn.execute(
                    text(
                        "SELECT COUNT(*) FROM pending_orders "
                        "WHERE profile_id = :profile_id "
                        "AND state IN ('pending', 'filling')"
                    ),
                    {"profile_id": profile_id},
                ).scalar()
                or 0
            )

    def find_duplicate_active(
        self, profile_id: str, symbol: str, side: str, setup_type: str
    ) -> list[str]:
        """Active order IDs sharing the given active key."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT order_id FROM pending_orders
                    WHERE profile_id = :profile_id
                      AND symbol = :symbol
                      AND side = :side
                      AND setup_type = :setup_type
                      AND state IN ('pending', 'filling')
                    ORDER BY created_at ASC
                    """
                ),
                {
                    "profile_id": profile_id,
                    "symbol": symbol,
                    "side": side,
                    "setup_type": setup_type,
                },
            ).fetchall()
        return [r[0] for r in rows]

    def _select_by_states(
        self, states: list[OrderState], *, symbol: str | None = None
    ) -> list[PendingOrder]:
        columns = ", ".join(_ORDER_COLUMNS)
        params: dict[str, Any] = {
            f"s{i}": s.value for i, s in enumerate(states)
        }
        placeholders = ", ".join(f":{k}" for k in params)
        symbol_clause = ""
        if symbol:
            symbol_clause = " AND symbol = :symbol"
            params["symbol"] = symbol

        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT {columns} FROM pending_orders "
                    f"WHERE state IN ({placeholders}){symbol_clause} "
                    f"ORDER BY created_at ASC"
                ),
                params,
            ).mappings().all()
        return [_row_to_order(r) for r in rows]

    # ------------------------------------------------------------------
    # Supersession and sweeps
    # ------------------------------------------------------------------

    def supersede_duplicates(
        self,
        profile_id: str,
        symbol: str,
        side: str,
        setup_type: str,
        *,
        reason: str = "superseded",
    ) -> list[str]:
        """Cancel every active order sharing the active key.

        Called before creating a new order for the same key, so the partial
        UNIQUE index is satisfied. Mirrors ``expire_duplicate_plans()``.
        """
        superseded: list[str] = []
        for order_id in self.find_duplicate_active(
            profile_id, symbol, side, setup_type
        ):
            try:
                self.mark_canceled(order_id, reason)
                superseded.append(order_id)
            except PendingOrderRegistryError:
                logger.warning(
                    "Could not supersede order %s (may have already "
                    "transitioned)", order_id,
                )
        return superseded

    def finalize_orphaned_orders(
        self, *, filling_lease_minutes: int = 5
    ) -> dict[str, OrderState]:
        """Sweep transient orders to terminal states. Layer-2 hard guarantee.

        - PENDING past ``expires_at`` -> EXPIRED
        - FILLING older than the lease -> resolved by looking for the trade
          first, mirroring ``recover_stale_reservations()``:
            * a trade exists  -> FILLED (execution completed during a crash)
            * still in window -> released to PENDING for another attempt
            * window elapsed  -> released to PENDING then EXPIRED

        Runs on orchestrator startup and inside every monitor tick, so no order
        can leak a transient state across a restart.

        Returns:
            ``{order_id: terminal_state}`` for everything it resolved. Orders
            released back to PENDING are omitted, since they are not terminal.
        """
        now = now_utc()
        resolved: dict[str, OrderState] = {}

        # ── PENDING past expiry ──
        for order in self.get_pending_orders():
            expires = to_utc(order.expires_at)
            if expires is not None and expires <= now:
                try:
                    self.mark_expired(order.order_id, reason="window_elapsed")
                    resolved[order.order_id] = OrderState.EXPIRED
                except PendingOrderRegistryError:
                    logger.warning(
                        "Orphan sweep: could not expire order %s", order.order_id
                    )

        # ── FILLING past the lease ──
        lease_cutoff = now - timedelta(minutes=filling_lease_minutes)
        for order in self._select_by_states([OrderState.FILLING]):
            claimed_at = to_utc(order.last_evaluated_bar_ts) or to_utc(
                order.created_at
            )
            if claimed_at is not None and claimed_at > lease_cutoff:
                continue  # lease still valid; a live tick may own it

            try:
                trade_id = self._find_trade_for_order(order)
                if trade_id is not None:
                    # Crashed between execute_trade() success and mark_filled().
                    self.mark_filled(
                        order.order_id,
                        fill_price=order.limit_price,
                        fill_policy="limit_price",
                        fill_bar_ts=order.fill_bar_ts or order.created_at,
                        trade_id=trade_id,
                    )
                    resolved[order.order_id] = OrderState.FILLED
                    logger.warning(
                        "Orphan sweep: order %s had a trade (%s) but was stuck "
                        "FILLING; marked filled",
                        order.order_id, trade_id,
                    )
                    continue

                self.release_claim(order.order_id, reason="lease_expired")
                expires = to_utc(order.expires_at)
                if expires is not None and expires <= now:
                    self.mark_expired(order.order_id, reason="window_elapsed")
                    resolved[order.order_id] = OrderState.EXPIRED
            except PendingOrderRegistryError:
                logger.warning(
                    "Orphan sweep: could not resolve stranded order %s",
                    order.order_id,
                )

        if resolved:
            logger.info(
                "Pending order orphan sweep resolved %d order(s): %s",
                len(resolved),
                {k: v.value for k, v in resolved.items()},
            )
        return resolved

    def _find_trade_for_order(self, order: PendingOrder) -> int | None:
        """Look for a trade already recorded for this order.

        Checks the order's own ``trade_id`` first, then any trade_events row
        carrying the order_id in its payload. Fail-open: on error, return None
        so the sweep treats the order as unfilled, which is the safe direction
        (it may retry, but it will not fabricate a fill).
        """
        if order.trade_id is not None:
            return order.trade_id

        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    text(
                        """
                        SELECT trade_id FROM trade_events
                        WHERE event_type = 'pending_order_filled'
                          AND trade_id IS NOT NULL
                          AND payload_json LIKE :needle
                        ORDER BY id DESC LIMIT 1
                        """
                    ),
                    {"needle": f'%{order.order_id}%'},
                ).fetchone()
            return int(row[0]) if row and row[0] is not None else None
        except Exception as e:
            logger.warning(
                "Could not check for an existing trade for order %s: %s",
                order.order_id, e,
            )
            return None

    # ------------------------------------------------------------------
    # Internal — transitions
    # ------------------------------------------------------------------

    def _transition(
        self,
        order_id: str,
        from_state: OrderState,
        to_state: OrderState,
        *,
        reason: str | None = None,
        reference_price: float | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        """Execute a CAS transition. Fails closed if rowcount != 1."""
        if (from_state, to_state) not in PERMITTED_TRANSITIONS:
            raise PendingOrderRegistryError(
                f"Illegal transition {from_state.value} -> {to_state.value} "
                f"for order {order_id}"
            )

        try:
            rowcount = self._execute_state_write(
                order_id, from_state, to_state, extra_fields
            )
            if rowcount != 1:
                msg = (
                    f"CAS transition failed for order {order_id}: "
                    f"{from_state.value} -> {to_state.value}, "
                    f"rowcount={rowcount}"
                )
                logger.error(msg)
                raise PendingOrderRegistryError(msg)
        except PendingOrderRegistryError:
            raise
        except Exception as e:
            msg = (
                f"Persistence error during transition for {order_id} "
                f"({from_state.value} -> {to_state.value}): {e}"
            )
            logger.error(msg)
            raise PendingOrderRegistryError(msg) from e

        self._emit_order_event_for(
            order_id,
            event_type=f"state_{to_state.value}",
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            reference_price=reference_price,
        )

    @with_lock_retry
    def _execute_state_write(
        self,
        order_id: str,
        from_state: OrderState,
        to_state: OrderState,
        extra_fields: dict[str, Any] | None = None,
    ) -> int:
        """DB write for a state transition. Returns rowcount for CAS checking."""
        set_parts = ["state = :new_state"]
        params: dict[str, Any] = {
            "new_state": to_state.value,
            "order_id": order_id,
            "expected_state": from_state.value,
        }

        if extra_fields:
            for key, value in extra_fields.items():
                if key not in _ORDER_COLUMNS:
                    raise PendingOrderRegistryError(
                        f"Unknown column {key!r} in transition for {order_id}"
                    )
                set_parts.append(f"{key} = :{key}")
                params[key] = value

        sql = (
            f"UPDATE pending_orders SET {', '.join(set_parts)} "
            f"WHERE order_id = :order_id AND state = :expected_state"
        )

        with self._engine.connect() as conn:
            result = conn.execute(text(sql), params)
            conn.commit()
            return result.rowcount

    # ------------------------------------------------------------------
    # Internal — event emission (fail-open)
    # ------------------------------------------------------------------

    def _emit_order_event_for(
        self,
        order_id: str,
        *,
        event_type: str,
        from_state: OrderState | None = None,
        to_state: OrderState | None = None,
        reason: str | None = None,
        reference_price: float | None = None,
        payload: dict | None = None,
    ) -> None:
        """Emit an event, resolving profile/symbol from the order row."""
        profile_id = ""
        symbol = ""
        try:
            order = self.get_order(order_id)
            if order is not None:
                profile_id = order.profile_id
                symbol = order.symbol
        except Exception:  # pragma: no cover - defensive
            logger.debug("Could not resolve order %s for event metadata", order_id)

        self._emit_order_event(
            order_id=order_id,
            profile_id=profile_id,
            symbol=symbol,
            event_type=event_type,
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            reference_price=reference_price,
            payload=payload,
        )

    def _emit_order_event(
        self,
        *,
        order_id: str,
        profile_id: str,
        symbol: str,
        event_type: str,
        from_state: OrderState | None = None,
        to_state: OrderState | None = None,
        reason: str | None = None,
        reference_price: float | None = None,
        payload: dict | None = None,
    ) -> None:
        """INSERT into pending_order_events (append-only).

        Fail-open: logs and returns on error. Event emission must never block a
        state transition, which is already committed by the time this runs.
        """
        event_data: dict[str, Any] = {}
        if reason:
            event_data["reason"] = reason
        if payload:
            event_data.update(payload)

        try:
            self._execute_event_write(
                order_id=order_id,
                profile_id=profile_id,
                symbol=symbol,
                event_type=event_type,
                event_data=json.dumps(event_data) if event_data else None,
                from_state=from_state.value if from_state else None,
                to_state=to_state.value if to_state else None,
                reference_price=(
                    float(reference_price) if reference_price is not None else None
                ),
                created_at=to_iso(now_utc()),
            )
        except Exception as e:
            logger.error(
                "Failed to emit pending order event %s for %s: %s",
                event_type, order_id, e,
            )

    @with_lock_retry
    def _execute_event_write(self, **kwargs: Any) -> None:
        with self._engine.connect() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO pending_order_events
                        (order_id, profile_id, symbol, event_type, event_data,
                         from_state, to_state, reference_price, created_at)
                    VALUES
                        (:order_id, :profile_id, :symbol, :event_type,
                         :event_data, :from_state, :to_state, :reference_price,
                         :created_at)
                    """
                ),
                kwargs,
            )
            conn.commit()

    def get_events(self, order_id: str) -> list[dict]:
        """Full lifecycle history for one order, oldest first.

        Powers the order-scoped dashboard endpoint, which exists because
        ``api_trade_events()`` requires a trade_id and pending-order events are
        mostly pre-trade.
        """
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT id, order_id, profile_id, symbol, event_type,
                           event_data, from_state, to_state, reference_price,
                           created_at
                    FROM pending_order_events
                    WHERE order_id = :order_id
                    ORDER BY id ASC
                    """
                ),
                {"order_id": order_id},
            ).mappings().all()

        events = []
        for row in rows:
            record = dict(row)
            raw = record.get("event_data")
            if raw:
                try:
                    record["event_data"] = json.loads(raw)
                except (ValueError, TypeError):
                    pass
            events.append(record)
        return events


# ---------------------------------------------------------------------------
# Row mapping
# ---------------------------------------------------------------------------


def _order_to_params(order: PendingOrder) -> dict[str, Any]:
    """Flatten a PendingOrder into INSERT parameters."""
    params: dict[str, Any] = {}
    for column in _ORDER_COLUMNS:
        value = getattr(order, column)
        if column == "state":
            params[column] = value.value
        elif column in _DATETIME_FIELDS:
            params[column] = to_iso(value) if value is not None else None
        else:
            params[column] = value
    return params


def _row_to_order(row: Any) -> PendingOrder:
    """Build a PendingOrder from a database row mapping."""
    data = dict(row)
    kwargs: dict[str, Any] = {}
    for column in _ORDER_COLUMNS:
        value = data.get(column)
        if column == "state":
            kwargs[column] = OrderState(value)
        elif column in _DATETIME_FIELDS:
            kwargs[column] = to_utc(value)
        else:
            kwargs[column] = value
    return PendingOrder(**kwargs)
