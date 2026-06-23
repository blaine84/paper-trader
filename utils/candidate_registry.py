"""Candidate Registry — persistent lifecycle management for PM trade candidates.

Provides CandidateState enum, frozen CandidateRecord dataclass, integrity
hashing, and the CandidateRegistry class backed by the pm_candidates SQLite
table. All state transitions use database compare-and-set (UPDATE ... WHERE
state = :expected) with rowcount verification. In authoritative mode,
persistence or state-transition failures fail closed (raise).

See: design.md §utils/candidate_registry.py
Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.4, 6.1–6.10
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)


class CandidateState(Enum):
    """Lifecycle states for a registered candidate."""

    REGISTERED = "registered"
    RESERVED = "reserved"  # CAS-claimed for execution pipeline
    EXECUTED = "executed"  # Trade successfully created
    SIZING_REJECTED = "sizing_rejected"  # Position sizer rejected
    GATE_REJECTED = "gate_rejected"  # Gate pipeline rejected
    REJECTED = "rejected"  # PM explicitly rejected
    NOT_SELECTED = "not_selected"  # Omitted from PM response
    EXPIRED = "expired"  # Past expiration or cycle boundary


@dataclass(frozen=True)
class CandidateRecord:
    """Immutable record for a single trade candidate.

    All dict fields (signal_snapshot, context_snapshot, benchmark_mapping)
    are stored as deep-copied, JSON-serializable canonical strings. The
    integrity_hash covers the canonical JSON of all identity and geometry fields.
    """

    candidate_id: str  # Full UUID4
    cycle_id: str
    profile_id: str
    symbol: str
    direction: str  # "BUY" or "SHORT"
    setup_type: str
    geometry_name: str
    entry_price: float
    stop_price: float
    target_price: float
    risk_reward: float
    trigger: str
    invalidation_basis: str
    target_basis: str
    source_signal_id: str
    signal_snapshot_json: str  # Canonical JSON string (not mutable dict)
    created_at: datetime
    expires_at: datetime
    integrity_hash: str  # SHA-256 of canonical identity+geometry fields
    # P1 fields (None when P1 disabled)
    context_snapshot_json: str | None = None
    benchmark_mapping_json: str | None = None


def _compute_integrity_hash(record_dict: dict) -> str:
    """Compute SHA-256 over canonical JSON of identity and geometry fields.

    Fields included in the hash (sorted-key canonical JSON):
        candidate_id, symbol, direction, entry_price, stop_price,
        target_price, setup_type, profile_id, cycle_id
    """
    identity_fields = {
        "candidate_id": record_dict["candidate_id"],
        "symbol": record_dict["symbol"],
        "direction": record_dict["direction"],
        "entry_price": record_dict["entry_price"],
        "stop_price": record_dict["stop_price"],
        "target_price": record_dict["target_price"],
        "setup_type": record_dict["setup_type"],
        "profile_id": record_dict["profile_id"],
        "cycle_id": record_dict["cycle_id"],
    }
    canonical = json.dumps(identity_fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class CandidateRegistryError(Exception):
    """Raised when a registry operation fails closed in authoritative mode."""


class CandidateRegistry:
    """Per-cycle candidate registry backed by the pm_candidates table.

    All state transitions use database CAS operations (UPDATE ... WHERE
    state = :expected). In authoritative mode, persistence failures fail
    closed (raise CandidateRegistryError).
    """

    def __init__(self, db: Any, cycle_id: str, profile_id: str) -> None:
        """Initialize registry for a specific cycle and profile.

        Args:
            db: SQLAlchemy engine instance.
            cycle_id: Unique identifier for this PM cycle.
            profile_id: Profile identifier for this cycle.
        """
        self._db = db
        self.cycle_id = cycle_id
        self.profile_id = profile_id

    def register(self, candidate: CandidateRecord) -> None:
        """INSERT candidate into pm_candidates with state=REGISTERED.

        Fails closed if INSERT fails (raises CandidateRegistryError).
        """
        try:
            with self._db.connect() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO pm_candidates (
                            candidate_id, cycle_id, profile_id, symbol, direction,
                            setup_type, geometry_name, entry_price, stop_price,
                            target_price, risk_reward, trigger, invalidation_basis,
                            target_basis, source_signal_id, signal_snapshot_json,
                            state, integrity_hash, created_at, expires_at,
                            context_snapshot_json, benchmark_mapping_json
                        ) VALUES (
                            :candidate_id, :cycle_id, :profile_id, :symbol, :direction,
                            :setup_type, :geometry_name, :entry_price, :stop_price,
                            :target_price, :risk_reward, :trigger, :invalidation_basis,
                            :target_basis, :source_signal_id, :signal_snapshot_json,
                            :state, :integrity_hash, :created_at, :expires_at,
                            :context_snapshot_json, :benchmark_mapping_json
                        )
                        """
                    ),
                    {
                        "candidate_id": candidate.candidate_id,
                        "cycle_id": candidate.cycle_id,
                        "profile_id": candidate.profile_id,
                        "symbol": candidate.symbol,
                        "direction": candidate.direction,
                        "setup_type": candidate.setup_type,
                        "geometry_name": candidate.geometry_name,
                        "entry_price": candidate.entry_price,
                        "stop_price": candidate.stop_price,
                        "target_price": candidate.target_price,
                        "risk_reward": candidate.risk_reward,
                        "trigger": candidate.trigger,
                        "invalidation_basis": candidate.invalidation_basis,
                        "target_basis": candidate.target_basis,
                        "source_signal_id": candidate.source_signal_id,
                        "signal_snapshot_json": candidate.signal_snapshot_json,
                        "state": CandidateState.REGISTERED.value,
                        "integrity_hash": candidate.integrity_hash,
                        "created_at": candidate.created_at.isoformat(),
                        "expires_at": candidate.expires_at.isoformat(),
                        "context_snapshot_json": candidate.context_snapshot_json,
                        "benchmark_mapping_json": candidate.benchmark_mapping_json,
                    },
                )
                conn.commit()
        except Exception as e:
            logger.error(
                "Failed to register candidate %s: %s",
                candidate.candidate_id,
                e,
            )
            raise CandidateRegistryError(
                f"Failed to register candidate {candidate.candidate_id}: {e}"
            ) from e

    def reserve(
        self, candidate_id: str, execution_key: str
    ) -> tuple[bool, str | None]:
        """Atomic CAS: transition REGISTERED → RESERVED with execution_key.

        UPDATE pm_candidates
        SET state='reserved', reserved_at=:now, execution_key=:key
        WHERE candidate_id=:id AND state='registered'

        Returns:
            (True, None) if exactly 1 row updated (reservation succeeded).
            (False, reason) otherwise — fail closed, no order produced.
        """
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._db.connect() as conn:
                result = conn.execute(
                    text(
                        """
                        UPDATE pm_candidates
                        SET state = :new_state,
                            reserved_at = :reserved_at,
                            execution_key = :execution_key
                        WHERE candidate_id = :candidate_id
                          AND state = :expected_state
                        """
                    ),
                    {
                        "new_state": CandidateState.RESERVED.value,
                        "reserved_at": now,
                        "execution_key": execution_key,
                        "candidate_id": candidate_id,
                        "expected_state": CandidateState.REGISTERED.value,
                    },
                )
                conn.commit()

                if result.rowcount == 1:
                    return (True, None)
                else:
                    reason = (
                        f"CAS failed for candidate {candidate_id}: "
                        f"expected state=registered, rowcount={result.rowcount}"
                    )
                    logger.warning(reason)
                    return (False, reason)
        except Exception as e:
            reason = f"Persistence error during reserve for {candidate_id}: {e}"
            logger.error(reason)
            raise CandidateRegistryError(reason) from e

    def mark_executed(self, candidate_id: str) -> None:
        """Transition RESERVED → EXECUTED.

        Fails closed if rowcount != 1 (raises CandidateRegistryError).
        """
        self._transition_state(
            candidate_id=candidate_id,
            from_state=CandidateState.RESERVED,
            to_state=CandidateState.EXECUTED,
        )

    def mark_sizing_rejected(self, candidate_id: str, reason: str) -> None:
        """Transition RESERVED → SIZING_REJECTED with rejection reason."""
        self._transition_state(
            candidate_id=candidate_id,
            from_state=CandidateState.RESERVED,
            to_state=CandidateState.SIZING_REJECTED,
            rejection_reason=reason,
        )

    def mark_gate_rejected(self, candidate_id: str, reason: str) -> None:
        """Transition RESERVED → GATE_REJECTED with rejection reason."""
        self._transition_state(
            candidate_id=candidate_id,
            from_state=CandidateState.RESERVED,
            to_state=CandidateState.GATE_REJECTED,
            rejection_reason=reason,
        )

    def mark_rejected(self, candidate_id: str, reason: str) -> None:
        """Transition REGISTERED → REJECTED with rejection reason (PM explicit)."""
        self._transition_state(
            candidate_id=candidate_id,
            from_state=CandidateState.REGISTERED,
            to_state=CandidateState.REJECTED,
            rejection_reason=reason,
        )

    def finalize_cycle(self) -> dict[str, CandidateState]:
        """Assign terminal states to all remaining REGISTERED candidates.

        - Past expires_at → EXPIRED
        - Still REGISTERED (not expired) → NOT_SELECTED

        Fails closed if any UPDATE fails.

        Returns:
            Dict mapping candidate_id → assigned terminal state.
        """
        now = datetime.now(timezone.utc).isoformat()
        terminal_assignments: dict[str, CandidateState] = {}

        try:
            with self._db.connect() as conn:
                # First: mark expired candidates
                result = conn.execute(
                    text(
                        """
                        UPDATE pm_candidates
                        SET state = :expired_state
                        WHERE cycle_id = :cycle_id
                          AND profile_id = :profile_id
                          AND state = :registered_state
                          AND expires_at <= :now
                        """
                    ),
                    {
                        "expired_state": CandidateState.EXPIRED.value,
                        "registered_state": CandidateState.REGISTERED.value,
                        "cycle_id": self.cycle_id,
                        "profile_id": self.profile_id,
                        "now": now,
                    },
                )

                # Retrieve expired candidate IDs
                if result.rowcount > 0:
                    rows = conn.execute(
                        text(
                            """
                            SELECT candidate_id FROM pm_candidates
                            WHERE cycle_id = :cycle_id
                              AND profile_id = :profile_id
                              AND state = :expired_state
                            """
                        ),
                        {
                            "cycle_id": self.cycle_id,
                            "profile_id": self.profile_id,
                            "expired_state": CandidateState.EXPIRED.value,
                        },
                    ).fetchall()
                    for row in rows:
                        terminal_assignments[row[0]] = CandidateState.EXPIRED

                # Second: mark remaining registered as NOT_SELECTED
                result = conn.execute(
                    text(
                        """
                        UPDATE pm_candidates
                        SET state = :not_selected_state
                        WHERE cycle_id = :cycle_id
                          AND profile_id = :profile_id
                          AND state = :registered_state
                        """
                    ),
                    {
                        "not_selected_state": CandidateState.NOT_SELECTED.value,
                        "registered_state": CandidateState.REGISTERED.value,
                        "cycle_id": self.cycle_id,
                        "profile_id": self.profile_id,
                    },
                )

                # Retrieve not-selected candidate IDs
                if result.rowcount > 0:
                    rows = conn.execute(
                        text(
                            """
                            SELECT candidate_id FROM pm_candidates
                            WHERE cycle_id = :cycle_id
                              AND profile_id = :profile_id
                              AND state = :not_selected_state
                            """
                        ),
                        {
                            "cycle_id": self.cycle_id,
                            "profile_id": self.profile_id,
                            "not_selected_state": CandidateState.NOT_SELECTED.value,
                        },
                    ).fetchall()
                    for row in rows:
                        if row[0] not in terminal_assignments:
                            terminal_assignments[row[0]] = CandidateState.NOT_SELECTED

                conn.commit()
        except Exception as e:
            logger.error("Failed to finalize cycle %s: %s", self.cycle_id, e)
            raise CandidateRegistryError(
                f"Failed to finalize cycle {self.cycle_id}: {e}"
            ) from e

        return terminal_assignments

    def get(self, candidate_id: str) -> CandidateRecord | None:
        """Look up candidate by ID. Returns CandidateRecord or None."""
        try:
            with self._db.connect() as conn:
                row = conn.execute(
                    text(
                        """
                        SELECT candidate_id, cycle_id, profile_id, symbol, direction,
                               setup_type, geometry_name, entry_price, stop_price,
                               target_price, risk_reward, trigger, invalidation_basis,
                               target_basis, source_signal_id, signal_snapshot_json,
                               created_at, expires_at, integrity_hash,
                               context_snapshot_json, benchmark_mapping_json
                        FROM pm_candidates
                        WHERE candidate_id = :candidate_id
                        """
                    ),
                    {"candidate_id": candidate_id},
                ).fetchone()

                if row is None:
                    return None

                return CandidateRecord(
                    candidate_id=row[0],
                    cycle_id=row[1],
                    profile_id=row[2],
                    symbol=row[3],
                    direction=row[4],
                    setup_type=row[5],
                    geometry_name=row[6],
                    entry_price=row[7],
                    stop_price=row[8],
                    target_price=row[9],
                    risk_reward=row[10],
                    trigger=row[11],
                    invalidation_basis=row[12],
                    target_basis=row[13],
                    source_signal_id=row[14],
                    signal_snapshot_json=row[15],
                    created_at=_parse_datetime(row[16]),
                    expires_at=_parse_datetime(row[17]),
                    integrity_hash=row[18],
                    context_snapshot_json=row[19],
                    benchmark_mapping_json=row[20],
                )
        except Exception as e:
            logger.error("Failed to get candidate %s: %s", candidate_id, e)
            raise CandidateRegistryError(
                f"Failed to get candidate {candidate_id}: {e}"
            ) from e

    def get_registered_ids(self) -> set[str]:
        """Return set of candidate_ids in REGISTERED state for this cycle."""
        try:
            with self._db.connect() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT candidate_id FROM pm_candidates
                        WHERE cycle_id = :cycle_id
                          AND profile_id = :profile_id
                          AND state = :state
                        """
                    ),
                    {
                        "cycle_id": self.cycle_id,
                        "profile_id": self.profile_id,
                        "state": CandidateState.REGISTERED.value,
                    },
                ).fetchall()
                return {row[0] for row in rows}
        except Exception as e:
            logger.error(
                "Failed to get registered IDs for cycle %s: %s", self.cycle_id, e
            )
            raise CandidateRegistryError(
                f"Failed to get registered IDs for cycle {self.cycle_id}: {e}"
            ) from e

    def get_offered_summary(self) -> list[dict]:
        """Return summary dicts for all candidates in this cycle.

        Each dict contains: candidate_id, symbol, direction, setup_type,
        entry_price, stop_price, target_price, risk_reward, geometry_name,
        trigger, state.
        """
        try:
            with self._db.connect() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT candidate_id, symbol, direction, setup_type,
                               entry_price, stop_price, target_price, risk_reward,
                               geometry_name, trigger, state
                        FROM pm_candidates
                        WHERE cycle_id = :cycle_id
                          AND profile_id = :profile_id
                        """
                    ),
                    {
                        "cycle_id": self.cycle_id,
                        "profile_id": self.profile_id,
                    },
                ).fetchall()

                return [
                    {
                        "candidate_id": row[0],
                        "symbol": row[1],
                        "direction": row[2],
                        "setup_type": row[3],
                        "entry_price": row[4],
                        "stop_price": row[5],
                        "target_price": row[6],
                        "risk_reward": row[7],
                        "geometry_name": row[8],
                        "trigger": row[9],
                        "state": row[10],
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(
                "Failed to get offered summary for cycle %s: %s", self.cycle_id, e
            )
            raise CandidateRegistryError(
                f"Failed to get offered summary for cycle {self.cycle_id}: {e}"
            ) from e

    def get_candidate_metadata(self) -> dict[str, dict]:
        """Return immutable metadata map keyed by candidate_id.

        Each entry contains: symbol, source_signal_id, profile_id.
        Used by parse_decision_contract for deduplication enforcement.
        """
        try:
            with self._db.connect() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT candidate_id, symbol, source_signal_id, profile_id
                        FROM pm_candidates
                        WHERE cycle_id = :cycle_id
                          AND profile_id = :profile_id
                        """
                    ),
                    {
                        "cycle_id": self.cycle_id,
                        "profile_id": self.profile_id,
                    },
                ).fetchall()

                return {
                    row[0]: {
                        "symbol": row[1],
                        "source_signal_id": row[2],
                        "profile_id": row[3],
                    }
                    for row in rows
                }
        except Exception as e:
            logger.error(
                "Failed to get candidate metadata for cycle %s: %s",
                self.cycle_id,
                e,
            )
            raise CandidateRegistryError(
                f"Failed to get candidate metadata for cycle {self.cycle_id}: {e}"
            ) from e

    @property
    def is_empty(self) -> bool:
        """True if no candidates registered for this cycle."""
        try:
            with self._db.connect() as conn:
                row = conn.execute(
                    text(
                        """
                        SELECT COUNT(*) FROM pm_candidates
                        WHERE cycle_id = :cycle_id
                          AND profile_id = :profile_id
                        """
                    ),
                    {
                        "cycle_id": self.cycle_id,
                        "profile_id": self.profile_id,
                    },
                ).fetchone()
                return row[0] == 0
        except Exception as e:
            logger.error(
                "Failed to check is_empty for cycle %s: %s", self.cycle_id, e
            )
            raise CandidateRegistryError(
                f"Failed to check is_empty for cycle {self.cycle_id}: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transition_state(
        self,
        candidate_id: str,
        from_state: CandidateState,
        to_state: CandidateState,
        rejection_reason: str | None = None,
    ) -> None:
        """Execute a CAS state transition. Fails closed if rowcount != 1."""
        try:
            with self._db.connect() as conn:
                params: dict[str, Any] = {
                    "new_state": to_state.value,
                    "candidate_id": candidate_id,
                    "expected_state": from_state.value,
                }

                if rejection_reason is not None:
                    result = conn.execute(
                        text(
                            """
                            UPDATE pm_candidates
                            SET state = :new_state,
                                rejection_reason = :rejection_reason
                            WHERE candidate_id = :candidate_id
                              AND state = :expected_state
                            """
                        ),
                        {**params, "rejection_reason": rejection_reason},
                    )
                else:
                    result = conn.execute(
                        text(
                            """
                            UPDATE pm_candidates
                            SET state = :new_state
                            WHERE candidate_id = :candidate_id
                              AND state = :expected_state
                            """
                        ),
                        params,
                    )
                conn.commit()

                if result.rowcount != 1:
                    msg = (
                        f"CAS transition failed for candidate {candidate_id}: "
                        f"{from_state.value} → {to_state.value}, "
                        f"rowcount={result.rowcount}"
                    )
                    logger.error(msg)
                    raise CandidateRegistryError(msg)
        except CandidateRegistryError:
            raise
        except Exception as e:
            msg = (
                f"Persistence error during state transition for "
                f"{candidate_id} ({from_state.value} → {to_state.value}): {e}"
            )
            logger.error(msg)
            raise CandidateRegistryError(msg) from e


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
        # Handle ISO format with or without timezone
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            # Fallback for formats like "2024-01-01 12:00:00"
            dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    raise ValueError(f"Cannot parse datetime from {type(value)}: {value}")


def recover_stale_reservations(
    engine,
    *,
    lease_timeout_minutes: int = 5,
) -> list[dict]:
    """Recover stale reservations at startup or cycle begin.

    Finds RESERVED candidates whose reserved_at is past the lease timeout.
    For each stale reservation:
    1. Check if a trade exists with the candidate's execution_key
       (in the trades table or trade_events table)
    2. If trade exists → mark EXECUTED (execution completed during crash)
    3. If no trade AND cycle expired (expires_at < now) → mark EXPIRED
    4. If no trade AND cycle active → mark REGISTERED (release for re-selection)

    Args:
        engine: SQLAlchemy engine instance.
        lease_timeout_minutes: Minutes after reservation before it's considered stale.
            Default is 5 minutes.

    Returns:
        List of recovery action dicts: [{candidate_id, action, reason}]
    """
    now = datetime.now(timezone.utc)
    stale_cutoff = now - timedelta(minutes=lease_timeout_minutes)

    recoveries = []

    with engine.connect() as conn:
        # Find all stale reserved candidates
        stale_rows = conn.execute(
            text("""
                SELECT candidate_id, execution_key, cycle_id, profile_id, expires_at
                FROM pm_candidates
                WHERE state = :state
                  AND reserved_at <= :cutoff
            """),
            {"state": CandidateState.RESERVED.value, "cutoff": stale_cutoff.isoformat()},
        ).fetchall()

        for row in stale_rows:
            candidate_id = row[0]
            execution_key = row[1]
            cycle_id = row[2]
            profile_id = row[3]
            expires_at_str = row[4]

            # Check if trade exists for this execution_key
            trade_exists = False
            if execution_key:
                # Check trade_events for an entry_filled with this key
                trade_row = conn.execute(
                    text("""
                        SELECT 1 FROM trade_events
                        WHERE event_type = 'entry_filled'
                          AND payload_json LIKE :key_pattern
                        LIMIT 1
                    """),
                    {"key_pattern": f"%{execution_key}%"},
                ).fetchone()
                trade_exists = trade_row is not None

            if trade_exists:
                # Trade was created during crash — mark as EXECUTED
                conn.execute(
                    text("""
                        UPDATE pm_candidates
                        SET state = :new_state
                        WHERE candidate_id = :cid AND state = :expected_state
                    """),
                    {
                        "new_state": CandidateState.EXECUTED.value,
                        "cid": candidate_id,
                        "expected_state": CandidateState.RESERVED.value,
                    },
                )
                recoveries.append({
                    "candidate_id": candidate_id,
                    "action": "mark_executed",
                    "reason": "trade_found_for_execution_key",
                })
            else:
                # No trade — check if cycle expired
                expires_at = _parse_datetime(expires_at_str) if expires_at_str else now
                if expires_at <= now:
                    # Cycle expired → mark EXPIRED
                    conn.execute(
                        text("""
                            UPDATE pm_candidates
                            SET state = :new_state
                            WHERE candidate_id = :cid AND state = :expected_state
                        """),
                        {
                            "new_state": CandidateState.EXPIRED.value,
                            "cid": candidate_id,
                            "expected_state": CandidateState.RESERVED.value,
                        },
                    )
                    recoveries.append({
                        "candidate_id": candidate_id,
                        "action": "mark_expired",
                        "reason": "no_trade_cycle_expired",
                    })
                else:
                    # Cycle still active → release back to REGISTERED
                    conn.execute(
                        text("""
                            UPDATE pm_candidates
                            SET state = :new_state, reserved_at = NULL, execution_key = NULL
                            WHERE candidate_id = :cid AND state = :expected_state
                        """),
                        {
                            "new_state": CandidateState.REGISTERED.value,
                            "cid": candidate_id,
                            "expected_state": CandidateState.RESERVED.value,
                        },
                    )
                    recoveries.append({
                        "candidate_id": candidate_id,
                        "action": "release_to_registered",
                        "reason": "no_trade_cycle_active",
                    })

            # Append recovery event to pm_candidate_events
            conn.execute(
                text("""
                    INSERT INTO pm_candidate_events
                    (candidate_id, cycle_id, profile_id, event_type, event_data, created_at)
                    VALUES (:cid, :cycle_id, :profile_id, :event_type, :event_data, :created_at)
                """),
                {
                    "cid": candidate_id,
                    "cycle_id": cycle_id,
                    "profile_id": profile_id,
                    "event_type": "recovery_released",
                    "event_data": json.dumps(recoveries[-1]),
                    "created_at": now.isoformat(),
                },
            )

        conn.commit()

    if recoveries:
        logger.warning(
            "Recovered %d stale reservations: %s",
            len(recoveries),
            ", ".join(f"{r['candidate_id'][:8]}→{r['action']}" for r in recoveries),
        )

    return recoveries
