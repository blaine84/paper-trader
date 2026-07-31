"""Trade Plan Registry — persistent lifecycle management for triggered trade plans.

Provides PlanState enum, frozen TradePlan dataclass, integrity hashing,
and the TradePlanRegistry class backed by the trade_plans SQLite table.
All state transitions use database compare-and-set (UPDATE ... WHERE
state = :expected) with rowcount verification. Persistence or
state-transition failures fail closed (raise TradePlanRegistryError).

See: design.md §utils/trade_plan_registry.py
Requirements: 1.1–1.10, 8.1–8.6, 9.1–9.6
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy import text

from utils.db_retry import with_lock_retry

logger = logging.getLogger(__name__)


class PlanState(Enum):
    """Lifecycle states for a trade plan."""
    PLANNED = "planned"        # PM accepted, awaiting monitor pickup
    WATCHING = "watching"      # Monitor actively evaluating trigger conditions
    TRIGGERED = "triggered"    # Trigger conditions met, execution in progress
    ENTERED = "entered"        # Execution succeeded — paper fill created
    MISSED = "missed"          # Valid plan became stale before execution
    REJECTED = "rejected"      # Invalidation logic or gate failure
    EXPIRED = "expired"        # TTL exceeded without triggering


TERMINAL_STATES = frozenset({
    PlanState.ENTERED,
    PlanState.MISSED,
    PlanState.REJECTED,
    PlanState.EXPIRED,
})


@dataclass(frozen=True)
class TradePlan:
    """Immutable trade plan record.

    Represents a PM-approved setup with entry zone, trigger conditions,
    and geometry — distinct from an executable order.
    """
    plan_id: str                    # UUID4
    candidate_id: str               # Link to source CandidateRecord
    cycle_id: str
    profile_id: str
    symbol: str
    direction: str                  # "BUY" or "SHORT"
    setup_type: str
    geometry_name: str | None

    # Entry zone
    entry_reference: float          # Analyst's intended entry price
    entry_zone_upper: float
    entry_zone_lower: float

    # Geometry (from original candidate — recalculated on trigger)
    stop_price: float
    target_price: float
    risk_reward: float

    # Trigger
    trigger_type: str               # "price_in_zone" | "level_breach" | "volume_confirmed"
    trigger_condition_json: str     # Structured JSON describing evaluation logic
    trigger_confirmation_required: bool

    # Invalidation
    invalidation_logic_json: str | None

    # Provenance
    analyst_reasoning: str | None
    pm_rationale: str | None
    source_signal_id: str | None
    signal_snapshot_json: str | None

    # Lifecycle
    state: PlanState
    created_at: datetime
    expires_at: datetime
    triggered_at: datetime | None
    executed_at: datetime | None
    missed_at: datetime | None

    # Integrity
    integrity_hash: str


def _compute_plan_integrity_hash(plan: TradePlan) -> str:
    """Compute SHA-256 over canonical JSON of identity + geometry fields.

    Fields included: plan_id, candidate_id, symbol, direction, setup_type,
    entry_reference, stop_price, target_price, profile_id, cycle_id.
    """
    identity_fields = {
        "plan_id": plan.plan_id,
        "candidate_id": plan.candidate_id,
        "symbol": plan.symbol,
        "direction": plan.direction,
        "setup_type": plan.setup_type,
        "entry_reference": plan.entry_reference,
        "stop_price": plan.stop_price,
        "target_price": plan.target_price,
        "profile_id": plan.profile_id,
        "cycle_id": plan.cycle_id,
    }
    canonical = json.dumps(identity_fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class TradePlanRegistryError(Exception):
    """Raised when a plan registry operation fails closed."""


class TradePlanRegistry:
    """Manages trade plan lifecycle with CAS state transitions.

    Follows the same pattern as CandidateRegistry: all state transitions
    use UPDATE ... WHERE state = :expected with rowcount verification.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    # Public API — Plan Creation
    # ------------------------------------------------------------------

    def create_plan(self, plan: TradePlan) -> str:
        """INSERT a new trade plan with state=PLANNED.

        Computes integrity_hash if not already set. Returns the plan_id.
        Fails closed on DB error (raises TradePlanRegistryError).
        """
        integrity_hash = plan.integrity_hash or _compute_plan_integrity_hash(plan)
        try:
            self._execute_create_write(plan, integrity_hash)
        except TradePlanRegistryError:
            raise
        except Exception as e:
            logger.error("Failed to create plan %s: %s", plan.plan_id, e)
            raise TradePlanRegistryError(
                f"Failed to create plan {plan.plan_id}: {e}"
            ) from e

        self._emit_plan_event(
            plan_id=plan.plan_id,
            cycle_id=plan.cycle_id,
            profile_id=plan.profile_id,
            event_type="plan_created",
            from_state=None,
            to_state=PlanState.PLANNED,
            payload={"candidate_id": plan.candidate_id, "symbol": plan.symbol},
        )
        return plan.plan_id

    @with_lock_retry
    def _execute_create_write(self, plan: TradePlan, integrity_hash: str) -> None:
        """Execute the DB INSERT for plan creation. Retried on lock contention."""
        with self._engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO trade_plans (
                        plan_id, candidate_id, cycle_id, profile_id,
                        symbol, direction, setup_type, geometry_name,
                        entry_reference, entry_zone_upper, entry_zone_lower,
                        stop_price, target_price, risk_reward,
                        trigger_type, trigger_condition_json,
                        trigger_confirmation_required,
                        invalidation_logic_json,
                        analyst_reasoning, pm_rationale,
                        source_signal_id, signal_snapshot_json,
                        state, created_at, expires_at,
                        triggered_at, executed_at, missed_at,
                        miss_reason, rejection_reason,
                        integrity_hash
                    ) VALUES (
                        :plan_id, :candidate_id, :cycle_id, :profile_id,
                        :symbol, :direction, :setup_type, :geometry_name,
                        :entry_reference, :entry_zone_upper, :entry_zone_lower,
                        :stop_price, :target_price, :risk_reward,
                        :trigger_type, :trigger_condition_json,
                        :trigger_confirmation_required,
                        :invalidation_logic_json,
                        :analyst_reasoning, :pm_rationale,
                        :source_signal_id, :signal_snapshot_json,
                        :state, :created_at, :expires_at,
                        :triggered_at, :executed_at, :missed_at,
                        :miss_reason, :rejection_reason,
                        :integrity_hash
                    )
                """),
                {
                    "plan_id": plan.plan_id,
                    "candidate_id": plan.candidate_id,
                    "cycle_id": plan.cycle_id,
                    "profile_id": plan.profile_id,
                    "symbol": plan.symbol,
                    "direction": plan.direction,
                    "setup_type": plan.setup_type,
                    "geometry_name": plan.geometry_name,
                    "entry_reference": plan.entry_reference,
                    "entry_zone_upper": plan.entry_zone_upper,
                    "entry_zone_lower": plan.entry_zone_lower,
                    "stop_price": plan.stop_price,
                    "target_price": plan.target_price,
                    "risk_reward": plan.risk_reward,
                    "trigger_type": plan.trigger_type,
                    "trigger_condition_json": plan.trigger_condition_json,
                    "trigger_confirmation_required": (
                        1 if plan.trigger_confirmation_required else 0
                    ),
                    "invalidation_logic_json": plan.invalidation_logic_json,
                    "analyst_reasoning": plan.analyst_reasoning,
                    "pm_rationale": plan.pm_rationale,
                    "source_signal_id": plan.source_signal_id,
                    "signal_snapshot_json": plan.signal_snapshot_json,
                    "state": PlanState.PLANNED.value,
                    "created_at": plan.created_at.isoformat(),
                    "expires_at": plan.expires_at.isoformat(),
                    "triggered_at": None,
                    "executed_at": None,
                    "missed_at": None,
                    "miss_reason": None,
                    "rejection_reason": None,
                    "integrity_hash": integrity_hash,
                },
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Public API — State Transitions
    # ------------------------------------------------------------------

    def activate(self, plan_id: str) -> None:
        """Transition PLANNED → WATCHING (monitor pickup)."""
        self._transition_state(plan_id, PlanState.PLANNED, PlanState.WATCHING)

    def trigger(self, plan_id: str, triggered_at: datetime | None = None) -> None:
        """Transition WATCHING → TRIGGERED (trigger conditions met)."""
        ts = triggered_at or datetime.now(timezone.utc)
        self._transition_state(
            plan_id, PlanState.WATCHING, PlanState.TRIGGERED,
            extra_fields={"triggered_at": ts.isoformat()},
        )

    def mark_entered(self, plan_id: str, executed_at: datetime | None = None) -> None:
        """Transition TRIGGERED → ENTERED (execution succeeded)."""
        ts = executed_at or datetime.now(timezone.utc)
        self._transition_state(
            plan_id, PlanState.TRIGGERED, PlanState.ENTERED,
            extra_fields={"executed_at": ts.isoformat()},
        )

    def mark_executed(self, plan_id: str, executed_at: datetime | None = None) -> None:
        """Alias for mark_entered — transition TRIGGERED → ENTERED."""
        self.mark_entered(plan_id, executed_at)

    def mark_missed(self, plan_id: str, reason: str, fresh_price: float | None = None) -> None:
        """Transition WATCHING|TRIGGERED → MISSED.

        Tries WATCHING → MISSED first; if CAS fails (plan may be TRIGGERED),
        tries TRIGGERED → MISSED.
        """
        now = datetime.now(timezone.utc)
        extra = {"missed_at": now.isoformat(), "miss_reason": reason}

        # Try from WATCHING first
        try:
            self._transition_state(
                plan_id, PlanState.WATCHING, PlanState.MISSED,
                reason=reason,
                fresh_price=fresh_price,
                extra_fields=extra,
            )
            return
        except TradePlanRegistryError:
            pass

        # Try from TRIGGERED
        try:
            self._transition_state(
                plan_id, PlanState.TRIGGERED, PlanState.MISSED,
                reason=reason,
                fresh_price=fresh_price,
                extra_fields=extra,
            )
            return
        except TradePlanRegistryError:
            pass

        # Try from PLANNED (for plans that never got picked up)
        self._transition_state(
            plan_id, PlanState.PLANNED, PlanState.MISSED,
            reason=reason,
            fresh_price=fresh_price,
            extra_fields=extra,
        )

    def mark_rejected(self, plan_id: str, reason: str) -> None:
        """Transition WATCHING|TRIGGERED → REJECTED.

        Tries WATCHING → REJECTED first; if CAS fails, tries TRIGGERED → REJECTED.
        """
        extra = {"rejection_reason": reason}

        try:
            self._transition_state(
                plan_id, PlanState.WATCHING, PlanState.REJECTED,
                reason=reason,
                extra_fields=extra,
            )
            return
        except TradePlanRegistryError:
            pass

        # Try from TRIGGERED
        try:
            self._transition_state(
                plan_id, PlanState.TRIGGERED, PlanState.REJECTED,
                reason=reason,
                extra_fields=extra,
            )
            return
        except TradePlanRegistryError:
            pass

        # Try from PLANNED
        self._transition_state(
            plan_id, PlanState.PLANNED, PlanState.REJECTED,
            reason=reason,
            extra_fields=extra,
        )

    def mark_expired(self, plan_id: str, reason: str | None = None) -> None:
        """Transition PLANNED|WATCHING → EXPIRED.

        Tries PLANNED → EXPIRED first; if CAS fails, tries WATCHING → EXPIRED.
        """
        extra: dict[str, Any] = {}
        if reason:
            extra["miss_reason"] = reason

        try:
            self._transition_state(
                plan_id, PlanState.PLANNED, PlanState.EXPIRED,
                reason=reason,
                extra_fields=extra if extra else None,
            )
            return
        except TradePlanRegistryError:
            pass

        self._transition_state(
            plan_id, PlanState.WATCHING, PlanState.EXPIRED,
            reason=reason,
            extra_fields=extra if extra else None,
        )

    def mark_triggered(self, plan_id: str) -> None:
        """Convenience: transition WATCHING → TRIGGERED (alias for trigger)."""
        self.trigger(plan_id)

    # ------------------------------------------------------------------
    # Public API — Queries
    # ------------------------------------------------------------------

    def get_active_plans(self, symbol: str | None = None) -> list[TradePlan]:
        """Return all plans in PLANNED or WATCHING state.

        If symbol is provided, filters to that symbol only.
        """
        with self._engine.connect() as conn:
            if symbol:
                result = conn.execute(
                    text("""
                        SELECT * FROM trade_plans
                        WHERE state IN (:s1, :s2) AND symbol = :symbol
                        ORDER BY created_at ASC
                    """),
                    {
                        "s1": PlanState.PLANNED.value,
                        "s2": PlanState.WATCHING.value,
                        "symbol": symbol,
                    },
                )
            else:
                result = conn.execute(
                    text("""
                        SELECT * FROM trade_plans
                        WHERE state IN (:s1, :s2)
                        ORDER BY created_at ASC
                    """),
                    {
                        "s1": PlanState.PLANNED.value,
                        "s2": PlanState.WATCHING.value,
                    },
                )
            rows = result.mappings().all()
        return [self._row_to_plan(row) for row in rows]

    def get_triggered_plans(self) -> list[TradePlan]:
        """Return all plans in TRIGGERED state (pending execution)."""
        with self._engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT * FROM trade_plans
                    WHERE state = :state
                    ORDER BY triggered_at ASC
                """),
                {"state": PlanState.TRIGGERED.value},
            )
            rows = result.mappings().all()
        return [self._row_to_plan(row) for row in rows]

    def get_plan(self, plan_id: str) -> TradePlan | None:
        """Return a single plan by ID, or None if not found."""
        with self._engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM trade_plans WHERE plan_id = :plan_id"),
                {"plan_id": plan_id},
            )
            row = result.mappings().first()
        if row is None:
            return None
        return self._row_to_plan(row)

    def has_active_plan_for_candidate(self, candidate_id: str) -> bool:
        """Return True if there is an active (PLANNED/WATCHING) plan for this candidate."""
        with self._engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT 1 FROM trade_plans
                    WHERE candidate_id = :candidate_id
                      AND state IN (:s1, :s2)
                    LIMIT 1
                """),
                {
                    "candidate_id": candidate_id,
                    "s1": PlanState.PLANNED.value,
                    "s2": PlanState.WATCHING.value,
                },
            )
            return result.first() is not None

    # ------------------------------------------------------------------
    # Public API — Deduplication
    # ------------------------------------------------------------------

    def expire_duplicate_plans(
        self,
        profile_id: str,
        symbol: str,
        direction: str,
        setup_type: str,
        reason: str = "superseded",
    ) -> list[str]:
        """Expire any active PLANNED/WATCHING plans for the same key.

        Returns list of expired plan_ids. Called before creating a new
        plan for the same (profile_id, symbol, direction, setup_type).
        """
        with self._engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT plan_id FROM trade_plans
                    WHERE profile_id = :profile_id
                      AND symbol = :symbol
                      AND direction = :direction
                      AND setup_type = :setup_type
                      AND state IN (:s1, :s2)
                """),
                {
                    "profile_id": profile_id,
                    "symbol": symbol,
                    "direction": direction,
                    "setup_type": setup_type,
                    "s1": PlanState.PLANNED.value,
                    "s2": PlanState.WATCHING.value,
                },
            )
            plan_ids = [row[0] for row in result.fetchall()]

        expired = []
        for pid in plan_ids:
            try:
                self.mark_expired(pid, reason=reason)
                expired.append(pid)
            except TradePlanRegistryError:
                logger.warning(
                    "Could not expire duplicate plan %s (may have already transitioned)", pid
                )
        return expired

    # ------------------------------------------------------------------
    # Public API — Orphan Sweep
    # ------------------------------------------------------------------

    def finalize_orphaned_plans(self) -> dict[str, PlanState]:
        """Sweep PLANNED/WATCHING/TRIGGERED plans past expiration to terminal states.

        - PLANNED/WATCHING past expires_at → EXPIRED
        - TRIGGERED past expires_at → MISSED (reason: "execution_timeout")

        Called on startup and periodically as a safety net.
        Returns dict of {plan_id: terminal_state} for all swept plans.
        """
        now = datetime.now(timezone.utc)
        swept: dict[str, PlanState] = {}

        with self._engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT plan_id, state FROM trade_plans
                    WHERE state IN (:s1, :s2, :s3)
                      AND expires_at <= :now
                """),
                {
                    "s1": PlanState.PLANNED.value,
                    "s2": PlanState.WATCHING.value,
                    "s3": PlanState.TRIGGERED.value,
                    "now": now.isoformat(),
                },
            )
            rows = result.fetchall()

        for plan_id, state_val in rows:
            try:
                if state_val == PlanState.TRIGGERED.value:
                    self.mark_missed(plan_id, reason="execution_timeout")
                    swept[plan_id] = PlanState.MISSED
                    # Record missed_setup event for TRIGGERED plans that never
                    # obtained a fresh execution quote before expiration.
                    self._record_orphan_missed_event(plan_id, "no_fresh_price_available")
                else:
                    self.mark_expired(plan_id, reason="orphan_sweep")
                    swept[plan_id] = PlanState.EXPIRED
                    # Record missed_setup event for plans that expired without triggering.
                    self._record_orphan_missed_event(plan_id, "plan_expired")
            except TradePlanRegistryError:
                logger.warning(
                    "Orphan sweep: could not finalize plan %s (state=%s)",
                    plan_id, state_val,
                )

        if swept:
            logger.info("Orphan sweep finalized %d plans: %s", len(swept), swept)
        return swept

    def _record_orphan_missed_event(self, plan_id: str, reason: str) -> None:
        """Record a missed_setup trade event for orphan-swept plans.

        Fail-open: logs error but never raises.
        """
        try:
            plan = self.get_plan(plan_id)
            if plan is None:
                return

            from utils.trade_events import log_trade_event
            from db.schema import get_session

            db = get_session(self._engine)
            try:
                log_trade_event(
                    db,
                    "missed_setup",
                    agent="plan_registry_sweep",
                    symbol=plan.symbol,
                    profile=plan.profile_id,
                    message=(
                        f"Plan {plan.plan_id} missed (orphan sweep): {reason}. "
                        f"Entry zone [{plan.entry_zone_lower}-{plan.entry_zone_upper}], "
                        f"target={plan.target_price}"
                    ),
                    payload={
                        "plan_id": plan.plan_id,
                        "candidate_id": plan.candidate_id,
                        "cycle_id": plan.cycle_id,
                        "symbol": plan.symbol,
                        "direction": plan.direction,
                        "setup_type": plan.setup_type,
                        "geometry_name": plan.geometry_name,
                        "profile_id": plan.profile_id,
                        "entry_reference": plan.entry_reference,
                        "entry_zone_upper": plan.entry_zone_upper,
                        "entry_zone_lower": plan.entry_zone_lower,
                        "fresh_price_at_miss": None,
                        "quote_timestamp": None,
                        "quote_age_seconds": None,
                        "intended_target": plan.target_price,
                        "original_stop": plan.stop_price,
                        "original_risk_reward": plan.risk_reward,
                        "reason_for_miss": reason,
                    },
                )
                db.commit()
            except Exception as e:
                logger.error(
                    "Failed to record orphan missed_setup event for plan %s: %s",
                    plan.plan_id, e,
                )
            finally:
                db.close()
        except Exception as e:
            logger.error(
                "Failed to record orphan missed_setup event for plan %s: %s",
                plan_id, e,
            )

    # ------------------------------------------------------------------
    # Internal — CAS State Transition
    # ------------------------------------------------------------------

    def _transition_state(
        self,
        plan_id: str,
        from_state: PlanState,
        to_state: PlanState,
        *,
        reason: str | None = None,
        fresh_price: float | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        """Execute a CAS state transition. Fails closed if rowcount != 1."""
        try:
            rowcount = self._execute_state_write(
                plan_id, from_state, to_state, extra_fields
            )
            if rowcount != 1:
                msg = (
                    f"CAS transition failed for plan {plan_id}: "
                    f"{from_state.value} → {to_state.value}, rowcount={rowcount}"
                )
                logger.error(msg)
                raise TradePlanRegistryError(msg)
        except TradePlanRegistryError:
            raise
        except Exception as e:
            msg = (
                f"Persistence error during state transition for "
                f"{plan_id} ({from_state.value} → {to_state.value}): {e}"
            )
            logger.error(msg)
            raise TradePlanRegistryError(msg) from e

        # Emit audit event on successful transition
        # Need plan metadata for event — fetch cycle_id/profile_id
        plan = self.get_plan(plan_id)
        cycle_id = plan.cycle_id if plan else ""
        profile_id = plan.profile_id if plan else ""

        self._emit_plan_event(
            plan_id=plan_id,
            cycle_id=cycle_id,
            profile_id=profile_id,
            event_type=f"state_{to_state.value}",
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            fresh_price=fresh_price,
        )

    @with_lock_retry
    def _execute_state_write(
        self,
        plan_id: str,
        from_state: PlanState,
        to_state: PlanState,
        extra_fields: dict[str, Any] | None = None,
    ) -> int:
        """Execute the DB write for a state transition. Retried on lock contention.

        Returns rowcount so the caller can verify CAS success.
        """
        with self._engine.connect() as conn:
            # Build SET clause dynamically for extra fields
            set_parts = ["state = :new_state"]
            params: dict[str, Any] = {
                "new_state": to_state.value,
                "plan_id": plan_id,
                "expected_state": from_state.value,
            }

            if extra_fields:
                for key, value in extra_fields.items():
                    set_parts.append(f"{key} = :{key}")
                    params[key] = value

            set_clause = ", ".join(set_parts)
            sql = f"""
                UPDATE trade_plans
                SET {set_clause}
                WHERE plan_id = :plan_id
                  AND state = :expected_state
            """

            result = conn.execute(text(sql), params)
            conn.commit()
            return result.rowcount

    # ------------------------------------------------------------------
    # Internal — Event Emission
    # ------------------------------------------------------------------

    def _emit_plan_event(
        self,
        plan_id: str,
        cycle_id: str,
        profile_id: str,
        event_type: str,
        from_state: PlanState | None = None,
        to_state: PlanState | None = None,
        reason: str | None = None,
        fresh_price: float | None = None,
        payload: dict | None = None,
    ) -> None:
        """Insert into trade_plan_events (append-only audit).

        Fail-open: logs error but does not raise — event emission
        must never block a state transition.
        """
        now = datetime.now(timezone.utc)
        event_data = {}
        if reason:
            event_data["reason"] = reason
        if payload:
            event_data.update(payload)

        try:
            self._execute_event_write(
                plan_id=plan_id,
                cycle_id=cycle_id,
                profile_id=profile_id,
                event_type=event_type,
                event_data=json.dumps(event_data) if event_data else None,
                fresh_price=fresh_price,
                from_state=from_state.value if from_state else None,
                to_state=to_state.value if to_state else None,
                created_at=now.isoformat(),
            )
        except Exception as e:
            logger.error(
                "Failed to emit plan event %s for plan %s: %s",
                event_type, plan_id, e,
            )

    @with_lock_retry
    def _execute_event_write(
        self,
        plan_id: str,
        cycle_id: str,
        profile_id: str,
        event_type: str,
        event_data: str | None,
        fresh_price: float | None,
        from_state: str | None,
        to_state: str | None,
        created_at: str,
    ) -> None:
        """Execute the INSERT for a plan event. Retried on lock contention."""
        with self._engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO trade_plan_events (
                        plan_id, cycle_id, profile_id, event_type,
                        event_data, fresh_price, from_state, to_state,
                        created_at
                    ) VALUES (
                        :plan_id, :cycle_id, :profile_id, :event_type,
                        :event_data, :fresh_price, :from_state, :to_state,
                        :created_at
                    )
                """),
                {
                    "plan_id": plan_id,
                    "cycle_id": cycle_id,
                    "profile_id": profile_id,
                    "event_type": event_type,
                    "event_data": event_data,
                    "fresh_price": fresh_price,
                    "from_state": from_state,
                    "to_state": to_state,
                    "created_at": created_at,
                },
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Internal — Row Mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_plan(row: Any) -> TradePlan:
        """Convert a database row (mapping) to a TradePlan dataclass."""
        return TradePlan(
            plan_id=row["plan_id"],
            candidate_id=row["candidate_id"],
            cycle_id=row["cycle_id"],
            profile_id=row["profile_id"],
            symbol=row["symbol"],
            direction=row["direction"],
            setup_type=row["setup_type"],
            geometry_name=row["geometry_name"],
            entry_reference=float(row["entry_reference"]),
            entry_zone_upper=float(row["entry_zone_upper"]),
            entry_zone_lower=float(row["entry_zone_lower"]),
            stop_price=float(row["stop_price"]),
            target_price=float(row["target_price"]),
            risk_reward=float(row["risk_reward"]),
            trigger_type=row["trigger_type"],
            trigger_condition_json=row["trigger_condition_json"],
            trigger_confirmation_required=bool(row["trigger_confirmation_required"]),
            invalidation_logic_json=row["invalidation_logic_json"],
            analyst_reasoning=row["analyst_reasoning"],
            pm_rationale=row["pm_rationale"],
            source_signal_id=row["source_signal_id"],
            signal_snapshot_json=row["signal_snapshot_json"],
            state=PlanState(row["state"]),
            created_at=_parse_datetime(row["created_at"]),
            expires_at=_parse_datetime(row["expires_at"]),
            triggered_at=_parse_datetime(row["triggered_at"]) if row["triggered_at"] else None,
            executed_at=_parse_datetime(row["executed_at"]) if row["executed_at"] else None,
            missed_at=_parse_datetime(row["missed_at"]) if row["missed_at"] else None,
            integrity_hash=row["integrity_hash"],
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _parse_datetime(value: Any) -> datetime:
    """Parse a datetime value from the database.

    Handles ISO format strings (with or without timezone) and returns
    a timezone-aware datetime in UTC.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, str):
        # Handle ISO format with timezone
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            # Fallback: strip microseconds if format is unusual
            dt = datetime.strptime(value[:19], "%Y-%m-%dT%H:%M:%S")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    raise ValueError(f"Cannot parse datetime from: {value!r}")
