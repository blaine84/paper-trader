"""Setup Watch Registry — persistent lifecycle management for setup watches.

Structural sibling of ``utils/pending_order_registry.py``: every state transition
is a compare-and-swap (``UPDATE ... WHERE state = :expected``) with rowcount
verification, so two concurrent cycles can never both act on one watch.

CAS state machine for the setup watch layer. All state transitions use
compare-and-swap (UPDATE ... WHERE state = :expected) with rowcount
verification. Event emission fails open; state transitions fail closed.

Schema DDL is owned by db/schema.py (init_setup_watch_schema).
This module contains only business logic.

Permitted transitions:

    WATCHING  -> MATURING    (first maturation condition met)
    MATURING  -> READY       (score >= threshold)
    READY     -> PROMOTED    (promoted to PM candidate pipeline)
    PROMOTED  -> ORDERED     (PM executed or created pending order)
    MATURING  -> WATCHING    (conditions faded)
    READY     -> MATURING    (score regressed below threshold)
    WATCHING  -> REJECTED    (invalidation triggered)
    WATCHING  -> EXPIRED     (TTL elapsed or superseded)
    MATURING  -> REJECTED    (invalidation triggered)
    MATURING  -> EXPIRED     (TTL elapsed or superseded)
    READY     -> REJECTED    (invalidation triggered)
    READY     -> EXPIRED     (TTL elapsed or superseded)
    PROMOTED  -> REJECTED    (PM rejected / gate failure)
    PROMOTED  -> EXPIRED     (promotion not consumed)

Terminal states are final: ORDERED, EXPIRED, REJECTED.

Failure direction: state transitions fail CLOSED (raise
``SetupWatchRegistryError``), while event emission fails OPEN so an audit
write can never block a state change.

Requirements: 1.9, 2.4-2.10, 3.1-3.10, 8.1, 11.1-11.3, 11.8, 12.6, 12.10
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import text

from db.schema import is_sqlite
from utils.db_retry import with_lock_retry
from utils.gate_config import SETUP_WATCH_MIN_CONDITION_COUNT
from utils.pending_order_time import now_utc, to_iso, to_utc

logger = logging.getLogger(__name__)


class WatchState(Enum):
    """Lifecycle states for a setup watch."""

    WATCHING = "watching"     # initial: resting, no conditions met yet
    MATURING = "maturing"    # at least one condition met, score < threshold
    READY = "ready"          # score >= threshold, eligible for promotion
    PROMOTED = "promoted"    # promoted to PM candidate pipeline
    ORDERED = "ordered"      # PM executed or created a pending order
    EXPIRED = "expired"      # TTL elapsed, superseded, or promotion not consumed
    REJECTED = "rejected"    # invalidation triggered or PM/gate rejection


TERMINAL_STATES: frozenset[WatchState] = frozenset({
    WatchState.ORDERED,
    WatchState.EXPIRED,
    WatchState.REJECTED,
})

ACTIVE_STATES: frozenset[WatchState] = frozenset({
    WatchState.WATCHING,
    WatchState.MATURING,
    WatchState.READY,
    WatchState.PROMOTED,
})

# The exact set of transitions the machine allows. Anything else is a bug.
PERMITTED_TRANSITIONS: frozenset[tuple[WatchState, WatchState]] = frozenset({
    # Forward maturation
    (WatchState.WATCHING, WatchState.MATURING),
    (WatchState.MATURING, WatchState.READY),
    (WatchState.READY, WatchState.PROMOTED),
    (WatchState.PROMOTED, WatchState.ORDERED),
    # Regression (conditions faded)
    (WatchState.MATURING, WatchState.WATCHING),
    (WatchState.READY, WatchState.MATURING),
    # Terminal from any active state
    (WatchState.WATCHING, WatchState.REJECTED),
    (WatchState.WATCHING, WatchState.EXPIRED),
    (WatchState.MATURING, WatchState.REJECTED),
    (WatchState.MATURING, WatchState.EXPIRED),
    (WatchState.READY, WatchState.REJECTED),
    (WatchState.READY, WatchState.EXPIRED),
    (WatchState.PROMOTED, WatchState.REJECTED),
    (WatchState.PROMOTED, WatchState.EXPIRED),
})

# Columns on setup_watches, in declaration order. Single source of truth for
# INSERT and SELECT so the two can never drift apart.
_WATCH_COLUMNS: tuple[str, ...] = (
    "watch_id",
    "profile_id",
    "symbol",
    "side",
    "setup_type",
    "state",
    "thesis",
    "source_type",
    "source_id",
    "source_cycle_id",
    "maturation_conditions_json",
    "invalidation_conditions_json",
    "last_evaluation_json",
    "entry_zone_json",
    "draft_geometry_json",
    "maturity_score",
    "created_at",
    "updated_at",
    "expires_at",
    "state_changed_at",
    "observed_cycles",
    "ready_at",
    "ready_reference_price",
    "terminal_reason",
    "promoted_cycle_id",
    "execution_ref_type",
    "execution_ref_id",
    "integrity_hash",
)

_DATETIME_FIELDS = frozenset({
    "created_at", "updated_at", "expires_at", "state_changed_at", "ready_at",
})

# Terminal state values for SQL IN clauses
_TERMINAL_STATE_VALUES = tuple(s.value for s in TERMINAL_STATES)

__all__ = [
    "ACTIVE_STATES",
    "PERMITTED_TRANSITIONS",
    "SetupWatch",
    "SetupWatchRegistry",
    "SetupWatchRegistryError",
    "TERMINAL_STATES",
    "WatchState",
    "compute_watch_integrity_hash",
]


@dataclass(frozen=True)
class SetupWatch:
    """Immutable setup watch record."""

    watch_id: str
    profile_id: str
    symbol: str
    side: str                           # "BUY" | "SHORT"
    setup_type: str
    state: WatchState

    # Thesis and source
    thesis: str
    source_type: str                    # "analyst"|"scout"|"market_state"|"candidate_reject"|"pm_defer"
    source_id: str | None
    source_cycle_id: str

    # Condition definitions (IMMUTABLE after INSERT — never UPDATEd)
    maturation_conditions_json: str
    invalidation_conditions_json: str

    # Evaluation results (mutable, updated each cycle)
    last_evaluation_json: str | None

    # Optional context
    entry_zone_json: str | None
    draft_geometry_json: str | None

    # Maturity tracking
    maturity_score: float

    # Timestamps
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    state_changed_at: datetime | None

    # Promotion gating (cycle-based, not wall-clock)
    observed_cycles: int

    # Outcome measurement (set once on first ready transition)
    ready_at: datetime | None
    ready_reference_price: float | None

    # Terminal / promotion metadata
    terminal_reason: str | None
    promoted_cycle_id: str | None

    # Execution linkage (set on ordered transition)
    execution_ref_type: str | None      # "trade" | "pending_order"
    execution_ref_id: str | None        # trade_id or order_id

    # Integrity
    integrity_hash: str

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def active_key(self) -> tuple[str, str, str, str]:
        """The tuple constrained to one active watch at a time."""
        return (self.profile_id, self.symbol, self.side, self.setup_type)


def compute_watch_integrity_hash(watch: SetupWatch) -> str:
    """SHA-256 over canonical JSON of identity fields.

    Covers exactly the fields that must never change after creation.
    """
    identity = {
        "watch_id": watch.watch_id,
        "profile_id": watch.profile_id,
        "symbol": watch.symbol,
        "side": watch.side,
        "setup_type": watch.setup_type,
        "thesis": watch.thesis,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _ready_window_elapsed_sql(engine: Any) -> str:
    """Dialect-aware predicate for outcome scoring window eligibility."""
    if is_sqlite(engine):
        return (
            "datetime(w.ready_at, '+' || :window_minutes || ' minutes') "
            "<= datetime(:cutoff)"
        )
    return (
        "(w.ready_at::timestamptz + (CAST(:window_minutes AS integer) * INTERVAL '1 minute')) "
        "<= CAST(:cutoff AS timestamptz)"
    )


class SetupWatchRegistryError(Exception):
    """Raised when a registry operation fails closed."""


class SetupWatchRegistry:
    """Manages setup watch lifecycle with CAS state transitions.

    All state transitions use CAS (UPDATE ... WHERE state = :expected).
    Event emission uses separate connection and fails open.
    Condition definition columns are never UPDATEd.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def create_watch(self, watch: SetupWatch) -> str:
        """INSERT a new watch with state=WATCHING. Returns the watch_id.

        Validates:
          - maturation_conditions_json has >= SETUP_WATCH_MIN_CONDITION_COUNT items
          - invalidation_conditions_json has >= 1 item
          - side is normalized to uppercase ("BUY" | "SHORT")

        Supersedes existing active watch for the same key before inserting.
        Emits a ``watch_created`` event on success.
        """
        if watch.state is not WatchState.WATCHING:
            raise SetupWatchRegistryError(
                f"New watches must start WATCHING, got {watch.state.value}"
            )

        # Validate condition minimums
        try:
            maturation = json.loads(watch.maturation_conditions_json)
        except (json.JSONDecodeError, TypeError):
            raise SetupWatchRegistryError(
                "maturation_conditions_json is not valid JSON"
            )
        if len(maturation) < SETUP_WATCH_MIN_CONDITION_COUNT:
            raise SetupWatchRegistryError(
                f"maturation_conditions_json must have >= "
                f"{SETUP_WATCH_MIN_CONDITION_COUNT} conditions, "
                f"got {len(maturation)}"
            )

        try:
            invalidation = json.loads(watch.invalidation_conditions_json)
        except (json.JSONDecodeError, TypeError):
            raise SetupWatchRegistryError(
                "invalidation_conditions_json is not valid JSON"
            )
        if len(invalidation) < 1:
            raise SetupWatchRegistryError(
                "invalidation_conditions_json must have >= 1 condition"
            )

        # Normalize side to uppercase
        normalized_side = watch.side.upper()
        if normalized_side not in ("BUY", "SHORT"):
            raise SetupWatchRegistryError(
                f"side must be 'BUY' or 'SHORT', got {watch.side!r}"
            )

        # Build the final watch with normalized side and integrity hash
        final_watch = SetupWatch(
            watch_id=watch.watch_id,
            profile_id=watch.profile_id,
            symbol=watch.symbol,
            side=normalized_side,
            setup_type=watch.setup_type,
            state=watch.state,
            thesis=watch.thesis,
            source_type=watch.source_type,
            source_id=watch.source_id,
            source_cycle_id=watch.source_cycle_id,
            maturation_conditions_json=watch.maturation_conditions_json,
            invalidation_conditions_json=watch.invalidation_conditions_json,
            last_evaluation_json=watch.last_evaluation_json,
            entry_zone_json=watch.entry_zone_json,
            draft_geometry_json=watch.draft_geometry_json,
            maturity_score=watch.maturity_score,
            created_at=watch.created_at,
            updated_at=watch.updated_at,
            expires_at=watch.expires_at,
            state_changed_at=watch.state_changed_at,
            observed_cycles=watch.observed_cycles,
            ready_at=watch.ready_at,
            ready_reference_price=watch.ready_reference_price,
            terminal_reason=watch.terminal_reason,
            promoted_cycle_id=watch.promoted_cycle_id,
            execution_ref_type=watch.execution_ref_type,
            execution_ref_id=watch.execution_ref_id,
            integrity_hash=watch.integrity_hash or compute_watch_integrity_hash(watch),
        )

        # Supersede existing active watch for the same key
        self._supersede_active(
            profile_id=final_watch.profile_id,
            symbol=final_watch.symbol,
            side=final_watch.side,
            setup_type=final_watch.setup_type,
        )

        # INSERT the new watch
        try:
            self._execute_create_write(final_watch)
        except SetupWatchRegistryError:
            raise
        except Exception as e:
            logger.error("Failed to create setup watch %s: %s", watch.watch_id, e)
            raise SetupWatchRegistryError(
                f"Failed to create setup watch {watch.watch_id}: {e}"
            ) from e

        # Emit event (fail-open)
        self._emit_event(
            watch_id=final_watch.watch_id,
            profile_id=final_watch.profile_id,
            symbol=final_watch.symbol,
            event_type="watch_created",
            to_state=WatchState.WATCHING,
            maturity_score=final_watch.maturity_score,
            event_data=json.dumps({
                "source_type": final_watch.source_type,
                "source_id": final_watch.source_id,
                "setup_type": final_watch.setup_type,
                "side": final_watch.side,
            }),
        )

        return final_watch.watch_id

    @with_lock_retry
    def _execute_create_write(self, watch: SetupWatch) -> None:
        """INSERT the watch row. Wrapped with @with_lock_retry."""
        columns = ", ".join(_WATCH_COLUMNS)
        placeholders = ", ".join(f":{c}" for c in _WATCH_COLUMNS)
        with self._engine.connect() as conn:
            conn.execute(
                text(
                    f"INSERT INTO setup_watches ({columns}) "
                    f"VALUES ({placeholders})"
                ),
                _watch_to_params(watch),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # State transitions (CAS, fail-closed)
    # ------------------------------------------------------------------

    def transition_state(
        self,
        watch_id: str,
        from_state: WatchState,
        to_state: WatchState,
        *,
        terminal_reason: str | None = None,
        promoted_cycle_id: str | None = None,
        execution_ref_type: str | None = None,
        execution_ref_id: str | None = None,
        ready_reference_price: float | None = None,
    ) -> None:
        """Execute a CAS state transition. Fails closed if rowcount != 1.

        On transition to READY, sets ready_at and ready_reference_price using
        COALESCE so re-entry after regression never overwrites the first
        maturation timestamp/price.

        Emits a state-change event (fail-open) after the transition commits.
        """
        if (from_state, to_state) not in PERMITTED_TRANSITIONS:
            raise SetupWatchRegistryError(
                f"Illegal transition {from_state.value} -> {to_state.value} "
                f"for watch {watch_id}"
            )

        try:
            rowcount = self._execute_transition_write(
                watch_id=watch_id,
                from_state=from_state,
                to_state=to_state,
                terminal_reason=terminal_reason,
                promoted_cycle_id=promoted_cycle_id,
                execution_ref_type=execution_ref_type,
                execution_ref_id=execution_ref_id,
                ready_reference_price=ready_reference_price,
            )
            if rowcount != 1:
                msg = (
                    f"CAS transition failed for watch {watch_id}: "
                    f"{from_state.value} -> {to_state.value}, "
                    f"rowcount={rowcount}"
                )
                logger.error(msg)
                raise SetupWatchRegistryError(msg)
        except SetupWatchRegistryError:
            raise
        except Exception as e:
            msg = (
                f"Persistence error during transition for {watch_id} "
                f"({from_state.value} -> {to_state.value}): {e}"
            )
            logger.error(msg)
            raise SetupWatchRegistryError(msg) from e

        # Emit event (fail-open)
        self._emit_event(
            watch_id=watch_id,
            profile_id="",  # resolved below
            symbol="",
            event_type=f"state_{to_state.value}",
            from_state=from_state,
            to_state=to_state,
            maturity_score=None,
            event_data=json.dumps({
                k: v for k, v in {
                    "terminal_reason": terminal_reason,
                    "promoted_cycle_id": promoted_cycle_id,
                    "execution_ref_type": execution_ref_type,
                    "execution_ref_id": execution_ref_id,
                }.items() if v is not None
            }) or None,
            resolve_from_watch=True,
        )

    @with_lock_retry
    def _execute_transition_write(
        self,
        *,
        watch_id: str,
        from_state: WatchState,
        to_state: WatchState,
        terminal_reason: str | None = None,
        promoted_cycle_id: str | None = None,
        execution_ref_type: str | None = None,
        execution_ref_id: str | None = None,
        ready_reference_price: float | None = None,
    ) -> int:
        """DB write for a state transition. Returns rowcount for CAS checking."""
        now = now_utc()
        now_iso = to_iso(now)

        params: dict[str, Any] = {
            "new_state": to_state.value,
            "state_changed_at": now_iso,
            "updated_at": now_iso,
            "watch_id": watch_id,
            "expected_state": from_state.value,
        }

        # Build SET clause
        if to_state == WatchState.READY:
            # COALESCE: never overwrite the first ready timestamp/price
            set_clause = (
                "state = :new_state, "
                "state_changed_at = :state_changed_at, "
                "updated_at = :updated_at, "
                "ready_at = COALESCE(ready_at, :ready_at), "
                "ready_reference_price = COALESCE(ready_reference_price, :ready_reference_price)"
            )
            params["ready_at"] = now_iso
            params["ready_reference_price"] = ready_reference_price
        else:
            set_clause = (
                "state = :new_state, "
                "state_changed_at = :state_changed_at, "
                "updated_at = :updated_at"
            )

        # Optional fields
        if terminal_reason is not None:
            set_clause += ", terminal_reason = :terminal_reason"
            params["terminal_reason"] = terminal_reason

        if promoted_cycle_id is not None:
            set_clause += ", promoted_cycle_id = :promoted_cycle_id"
            params["promoted_cycle_id"] = promoted_cycle_id

        if execution_ref_type is not None:
            set_clause += ", execution_ref_type = :execution_ref_type"
            params["execution_ref_type"] = execution_ref_type

        if execution_ref_id is not None:
            set_clause += ", execution_ref_id = :execution_ref_id"
            params["execution_ref_id"] = execution_ref_id

        sql = (
            f"UPDATE setup_watches SET {set_clause} "
            f"WHERE watch_id = :watch_id AND state = :expected_state"
        )

        with self._engine.connect() as conn:
            result = conn.execute(text(sql), params)
            conn.commit()
            return result.rowcount

    # ------------------------------------------------------------------
    # Evaluation updates (non-CAS, fail-open)
    # ------------------------------------------------------------------

    def update_evaluation(
        self,
        watch_id: str,
        maturity_score: float,
        last_evaluation_json: str,
    ) -> None:
        """Update maturity_score and last_evaluation_json for a watch.

        Non-CAS UPDATE — deliberately does NOT touch maturation_conditions_json
        or invalidation_conditions_json (condition immutability invariant).

        Fail-open: logs and returns on error. Evaluation persistence must never
        block the pipeline.
        """
        try:
            self._execute_evaluation_write(
                watch_id, maturity_score, last_evaluation_json
            )
        except Exception as e:
            logger.warning(
                "Failed to update evaluation for watch %s (fail-open): %s",
                watch_id, e,
            )

    @with_lock_retry
    def _execute_evaluation_write(
        self, watch_id: str, maturity_score: float, last_evaluation_json: str
    ) -> None:
        now_iso = to_iso(now_utc())
        with self._engine.connect() as conn:
            conn.execute(
                text(
                    "UPDATE setup_watches "
                    "SET maturity_score = :score, "
                    "    last_evaluation_json = :eval_json, "
                    "    updated_at = :updated_at "
                    "WHERE watch_id = :watch_id"
                ),
                {
                    "score": maturity_score,
                    "eval_json": last_evaluation_json,
                    "updated_at": now_iso,
                    "watch_id": watch_id,
                },
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Cycle observation counter (non-CAS, fail-open)
    # ------------------------------------------------------------------

    def increment_observed_cycles(self, watch_ids: list[str]) -> None:
        """Batch increment observed_cycles for the given watch IDs.

        Fail-open: logs and returns on error. Observation counting must never
        block the pipeline.
        """
        if not watch_ids:
            return
        try:
            self._execute_increment_write(watch_ids)
        except Exception as e:
            logger.warning(
                "Failed to increment observed_cycles for %d watches (fail-open): %s",
                len(watch_ids), e,
            )

    @with_lock_retry
    def _execute_increment_write(self, watch_ids: list[str]) -> None:
        now_iso = to_iso(now_utc())
        # Build parameterized IN clause
        params: dict[str, Any] = {"updated_at": now_iso}
        placeholders = []
        for i, wid in enumerate(watch_ids):
            key = f"id_{i}"
            params[key] = wid
            placeholders.append(f":{key}")

        in_clause = ", ".join(placeholders)
        with self._engine.connect() as conn:
            conn.execute(
                text(
                    f"UPDATE setup_watches "
                    f"SET observed_cycles = observed_cycles + 1, "
                    f"    updated_at = :updated_at "
                    f"WHERE watch_id IN ({in_clause})"
                ),
                params,
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_watch(self, watch_id: str) -> SetupWatch | None:
        """Look up one watch by ID, or None."""
        columns = ", ".join(_WATCH_COLUMNS)
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    f"SELECT {columns} FROM setup_watches "
                    f"WHERE watch_id = :watch_id"
                ),
                {"watch_id": watch_id},
            ).mappings().fetchone()
        return _row_to_watch(row) if row else None

    def get_active_watches(self, profile_id: str) -> list[SetupWatch]:
        """All non-terminal watches for a profile, oldest first."""
        columns = ", ".join(_WATCH_COLUMNS)
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT {columns} FROM setup_watches "
                    f"WHERE profile_id = :profile_id "
                    f"  AND state NOT IN ('ordered', 'expired', 'rejected') "
                    f"ORDER BY created_at ASC"
                ),
                {"profile_id": profile_id},
            ).mappings().all()
        return [_row_to_watch(r) for r in rows]

    def get_promoted_watches(
        self, profile_id: str, cycle_id: str
    ) -> list[SetupWatch]:
        """Watches in PROMOTED state for a given profile and cycle."""
        columns = ", ".join(_WATCH_COLUMNS)
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT {columns} FROM setup_watches "
                    f"WHERE profile_id = :profile_id "
                    f"  AND state = 'promoted' "
                    f"  AND promoted_cycle_id = :cycle_id "
                    f"ORDER BY created_at ASC"
                ),
                {"profile_id": profile_id, "cycle_id": cycle_id},
            ).mappings().all()
        return [_row_to_watch(r) for r in rows]

    def count_active(self, profile_id: str) -> int:
        """Count non-terminal watches for a profile."""
        with self._engine.connect() as conn:
            return int(
                conn.execute(
                    text(
                        "SELECT COUNT(*) FROM setup_watches "
                        "WHERE profile_id = :profile_id "
                        "  AND state NOT IN ('ordered', 'expired', 'rejected')"
                    ),
                    {"profile_id": profile_id},
                ).scalar()
                or 0
            )

    def count_active_for_symbol(self, symbol: str) -> int:
        """Count non-terminal watches for a symbol across all profiles."""
        with self._engine.connect() as conn:
            return int(
                conn.execute(
                    text(
                        "SELECT COUNT(*) FROM setup_watches "
                        "WHERE symbol = :symbol "
                        "  AND state NOT IN ('ordered', 'expired', 'rejected')"
                    ),
                    {"symbol": symbol},
                ).scalar()
                or 0
            )

    # ------------------------------------------------------------------
    # Outcome scoring support
    # ------------------------------------------------------------------

    def get_watches_awaiting_scoring(
        self, window_label: str, window_minutes: int
    ) -> list[SetupWatch]:
        """Select watches with ready_at older than window_minutes that lack
        a row in setup_watch_outcomes for the given window_label.

        Watches are scored regardless of terminal state (counterfactual
        independence).
        """
        cutoff = to_iso(now_utc())
        columns = ", ".join(f"w.{c}" for c in _WATCH_COLUMNS)
        ready_window_elapsed = _ready_window_elapsed_sql(self._engine)
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT {columns} FROM setup_watches w "
                    f"WHERE w.ready_at IS NOT NULL "
                    f"  AND {ready_window_elapsed} "
                    f"  AND NOT EXISTS ("
                    f"    SELECT 1 FROM setup_watch_outcomes o "
                    f"    WHERE o.watch_id = w.watch_id "
                    f"      AND o.window_label = :window_label"
                    f"  )"
                ),
                {
                    "cutoff": cutoff,
                    "window_label": window_label,
                    "window_minutes": window_minutes,
                },
            ).mappings().all()
        return [_row_to_watch(r) for r in rows]

    def record_outcome(self, outcome: dict) -> bool:
        """INSERT into setup_watch_outcomes. Returns False on IntegrityError
        (benign duplicate via the unique index on watch_id + window_label).
        """
        try:
            self._execute_outcome_write(outcome)
            return True
        except Exception as e:
            # Check for unique constraint violation (benign duplicate)
            err_msg = str(e).lower()
            if "unique" in err_msg or "integrity" in err_msg:
                logger.debug(
                    "Duplicate outcome for watch %s window %s (benign race)",
                    outcome.get("watch_id"), outcome.get("window_label"),
                )
                return False
            logger.warning(
                "Failed to record outcome for watch %s: %s",
                outcome.get("watch_id"), e,
            )
            return False

    @with_lock_retry
    def _execute_outcome_write(self, outcome: dict) -> None:
        with self._engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO setup_watch_outcomes "
                    "  (watch_id, profile_id, symbol, side, window_label, "
                    "   window_minutes, reference_price, evaluated_at, "
                    "   mfe_pct, mae_pct, entry_zone_touched, "
                    "   would_have_hit_target, would_have_hit_stop, "
                    "   scorable, unscorable_reason, created_at) "
                    "VALUES "
                    "  (:watch_id, :profile_id, :symbol, :side, :window_label, "
                    "   :window_minutes, :reference_price, :evaluated_at, "
                    "   :mfe_pct, :mae_pct, :entry_zone_touched, "
                    "   :would_have_hit_target, :would_have_hit_stop, "
                    "   :scorable, :unscorable_reason, :created_at)"
                ),
                outcome,
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def expire_elapsed(self, profile_id: str) -> int:
        """Batch expire watches whose TTL has elapsed.

        Returns the number of watches expired.
        """
        now_iso = to_iso(now_utc())
        try:
            with self._engine.connect() as conn:
                result = conn.execute(
                    text(
                        "UPDATE setup_watches "
                        "SET state = 'expired', "
                        "    terminal_reason = 'ttl_elapsed', "
                        "    state_changed_at = :now, "
                        "    updated_at = :now "
                        "WHERE profile_id = :profile_id "
                        "  AND state NOT IN ('ordered', 'expired', 'rejected') "
                        "  AND expires_at < :now"
                    ),
                    {"profile_id": profile_id, "now": now_iso},
                )
                conn.commit()
                count = result.rowcount
        except Exception as e:
            logger.warning(
                "Failed to expire elapsed watches for profile %s: %s",
                profile_id, e,
            )
            return 0

        if count > 0:
            logger.info(
                "Expired %d TTL-elapsed watches for profile %s", count, profile_id
            )
        return count

    def expire_stale_promoted(self, profile_id: str, current_cycle_id: str) -> int:
        """Expire promoted watches from prior cycles that were not consumed.

        Returns the number of watches expired.
        """
        now_iso = to_iso(now_utc())
        try:
            with self._engine.connect() as conn:
                result = conn.execute(
                    text(
                        "UPDATE setup_watches "
                        "SET state = 'expired', "
                        "    terminal_reason = 'promotion_not_consumed', "
                        "    state_changed_at = :now, "
                        "    updated_at = :now "
                        "WHERE profile_id = :profile_id "
                        "  AND state = 'promoted' "
                        "  AND promoted_cycle_id != :current_cycle_id"
                    ),
                    {
                        "profile_id": profile_id,
                        "now": now_iso,
                        "current_cycle_id": current_cycle_id,
                    },
                )
                conn.commit()
                count = result.rowcount
        except Exception as e:
            logger.warning(
                "Failed to expire stale promoted watches for profile %s: %s",
                profile_id, e,
            )
            return 0

        if count > 0:
            logger.info(
                "Expired %d stale promoted watches for profile %s",
                count, profile_id,
            )
        return count

    # ------------------------------------------------------------------
    # Internal — supersession
    # ------------------------------------------------------------------

    def _supersede_active(
        self,
        profile_id: str,
        symbol: str,
        side: str,
        setup_type: str,
    ) -> None:
        """Expire any existing active watch for the same key before inserting.

        CAS transition: only watches in non-terminal states are affected.
        """
        now_iso = to_iso(now_utc())
        try:
            with self._engine.connect() as conn:
                result = conn.execute(
                    text(
                        "UPDATE setup_watches "
                        "SET state = 'expired', "
                        "    terminal_reason = 'superseded', "
                        "    state_changed_at = :now, "
                        "    updated_at = :now "
                        "WHERE profile_id = :profile_id "
                        "  AND symbol = :symbol "
                        "  AND side = :side "
                        "  AND setup_type = :setup_type "
                        "  AND state NOT IN ('ordered', 'expired', 'rejected')"
                    ),
                    {
                        "profile_id": profile_id,
                        "symbol": symbol,
                        "side": side,
                        "setup_type": setup_type,
                        "now": now_iso,
                    },
                )
                conn.commit()
                if result.rowcount > 0:
                    logger.debug(
                        "Superseded %d existing watch(es) for %s/%s/%s/%s",
                        result.rowcount, profile_id, symbol, side, setup_type,
                    )
        except Exception as e:
            logger.warning(
                "Failed to supersede active watches for %s/%s (non-fatal): %s",
                profile_id, symbol, e,
            )

    # ------------------------------------------------------------------
    # Internal — event emission (fail-open)
    # ------------------------------------------------------------------

    def _emit_event(
        self,
        *,
        watch_id: str,
        profile_id: str,
        symbol: str,
        event_type: str,
        from_state: WatchState | None = None,
        to_state: WatchState | None = None,
        maturity_score: float | None = None,
        event_data: str | None = None,
        resolve_from_watch: bool = False,
    ) -> None:
        """INSERT into setup_watch_events (append-only).

        Fail-open: logs and returns on error. Event emission must never block a
        state transition, which is already committed by the time this runs.
        """
        # Resolve profile_id and symbol from the watch if needed
        if resolve_from_watch:
            try:
                watch = self.get_watch(watch_id)
                if watch is not None:
                    profile_id = watch.profile_id
                    symbol = watch.symbol
                    if maturity_score is None:
                        maturity_score = watch.maturity_score
            except Exception:
                logger.debug(
                    "Could not resolve watch %s for event metadata", watch_id
                )

        try:
            self._execute_event_write(
                watch_id=watch_id,
                profile_id=profile_id,
                symbol=symbol,
                event_type=event_type,
                event_data=event_data,
                from_state=from_state.value if from_state else None,
                to_state=to_state.value if to_state else None,
                maturity_score=maturity_score,
                created_at=to_iso(now_utc()),
            )
        except Exception as e:
            logger.warning(
                "Failed to emit setup watch event %s for %s (fail-open): %s",
                event_type, watch_id, e,
            )

    @with_lock_retry
    def _execute_event_write(self, **kwargs: Any) -> None:
        with self._engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO setup_watch_events "
                    "  (watch_id, profile_id, symbol, event_type, event_data, "
                    "   from_state, to_state, maturity_score, created_at) "
                    "VALUES "
                    "  (:watch_id, :profile_id, :symbol, :event_type, "
                    "   :event_data, :from_state, :to_state, :maturity_score, "
                    "   :created_at)"
                ),
                kwargs,
            )
            conn.commit()


# ---------------------------------------------------------------------------
# Row mapping
# ---------------------------------------------------------------------------


def _watch_to_params(watch: SetupWatch) -> dict[str, Any]:
    """Flatten a SetupWatch into INSERT parameters."""
    params: dict[str, Any] = {}
    for column in _WATCH_COLUMNS:
        value = getattr(watch, column)
        if column == "state":
            params[column] = value.value
        elif column in _DATETIME_FIELDS:
            params[column] = to_iso(value) if value is not None else None
        else:
            params[column] = value
    return params


def _row_to_watch(row: Any) -> SetupWatch:
    """Build a SetupWatch from a database row mapping."""
    data = dict(row)
    kwargs: dict[str, Any] = {}
    for column in _WATCH_COLUMNS:
        value = data.get(column)
        if column == "state":
            kwargs[column] = WatchState(value)
        elif column in _DATETIME_FIELDS:
            kwargs[column] = to_utc(value)
        else:
            kwargs[column] = value
    return SetupWatch(**kwargs)
