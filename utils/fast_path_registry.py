"""Fast-path trigger registry — registration, state management, and expiry.

Provides the TriggerRecord frozen dataclass representing a single registered
trigger, and the FastPathRegistry class for persistent lifecycle management
backed by the fast_path_triggers SQLite table.

All state transitions use database compare-and-set (UPDATE ... WHERE
state = :expected) with rowcount verification. Registration is fail-closed:
INSERT failure raises rather than silently skipping.

Fail mode: fail-closed (no trigger = no evaluation).

See: .kiro/specs/fast-path-deterministic-execution/design.md
Requirements: 2.1, 2.2, 2.3, 2.11, 1.9
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text

from utils.db_retry import with_lock_retry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TriggerRecord:
    """Immutable record for a single registered fast-path trigger.

    Represents one row in the fast_path_triggers table. All geometry fields
    are frozen at registration time and must not be substituted with fresh
    data during evaluation (Requirement 10.10).
    """

    trigger_id: str
    symbol: str
    profile_id: str
    direction: str  # "BUY" or "SHORT"
    setup_type: str

    # Trigger condition
    trigger_type: str  # entry_zone | level_break | level_reject | vwap_cross | price_target
    trigger_level: float  # Primary price level to watch

    # Entry zone (optional, only for zone-based triggers)
    trigger_zone_upper: float | None = None
    trigger_zone_lower: float | None = None

    # Frozen geometry
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    geometry_name: str | None = None

    # Source linkage
    source_signal_id: str | None = None
    source_watch_id: str | None = None

    # Basis descriptions
    invalidation_basis: str | None = None
    target_basis: str | None = None

    # Lifecycle
    state: str = "active"
    registered_at: str = ""
    expires_at: str = ""

    # Context snapshots (JSON strings)
    signal_snapshot_json: str | None = None
    context_json: str | None = None


class FastPathRegistryError(Exception):
    """Raised when a registry operation fails closed."""


class FastPathRegistry:
    """Per-profile trigger registry backed by the fast_path_triggers table.

    All state transitions use database CAS operations (UPDATE ... WHERE
    state = :expected). Registration is fail-closed: INSERT failure raises
    FastPathRegistryError. Deduplication prevents multiple active triggers
    for the same (symbol, direction, profile_id, setup_type).
    """

    def __init__(self, db: Any, profile_id: str) -> None:
        """Initialize registry for a specific profile.

        Args:
            db: SQLAlchemy engine instance.
            profile_id: Profile identifier for this registry.
        """
        self._db = db
        self.profile_id = profile_id

    def register_trigger(self, trigger: TriggerRecord) -> None:
        """INSERT trigger into fast_path_triggers with state=active.

        Performs deduplication check first: if an active trigger already
        exists for the same (symbol, direction, profile_id, setup_type),
        logs a warning and skips registration (fail-closed — no silent
        duplicates).

        Raises:
            FastPathRegistryError: If INSERT fails or deduplication rejects.
        """
        # Deduplication: check for existing active trigger with same identity
        try:
            with self._db.connect() as conn:
                result = conn.execute(
                    text(
                        """
                        SELECT trigger_id FROM fast_path_triggers
                        WHERE symbol = :symbol
                          AND direction = :direction
                          AND profile_id = :profile_id
                          AND setup_type = :setup_type
                          AND state = 'active'
                        LIMIT 1
                        """
                    ),
                    {
                        "symbol": trigger.symbol,
                        "direction": trigger.direction,
                        "profile_id": trigger.profile_id,
                        "setup_type": trigger.setup_type,
                    },
                )
                existing = result.fetchone()
        except Exception as e:
            logger.error(
                "Deduplication check failed for trigger %s: %s",
                trigger.trigger_id,
                e,
            )
            raise FastPathRegistryError(
                f"Deduplication check failed for trigger {trigger.trigger_id}: {e}"
            ) from e

        if existing:
            logger.warning(
                "Duplicate trigger rejected: symbol=%s direction=%s profile=%s "
                "setup_type=%s — active trigger %s already exists",
                trigger.symbol,
                trigger.direction,
                trigger.profile_id,
                trigger.setup_type,
                existing[0],
            )
            return

        # INSERT the trigger
        try:
            self._execute_register_write(trigger)
        except FastPathRegistryError:
            raise
        except Exception as e:
            logger.error(
                "Failed to register trigger %s: %s",
                trigger.trigger_id,
                e,
            )
            raise FastPathRegistryError(
                f"Failed to register trigger {trigger.trigger_id}: {e}"
            ) from e

    @with_lock_retry
    def _execute_register_write(self, trigger: TriggerRecord) -> None:
        """Execute the DB INSERT for trigger registration. Retried on lock contention."""
        with self._db.connect() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO fast_path_triggers (
                        trigger_id, symbol, profile_id, direction, setup_type,
                        trigger_type, trigger_level, trigger_zone_upper,
                        trigger_zone_lower, entry_price, stop_price, target_price,
                        geometry_name, source_signal_id, source_watch_id,
                        invalidation_basis, target_basis, state, registered_at,
                        expires_at, signal_snapshot_json, context_json
                    ) VALUES (
                        :trigger_id, :symbol, :profile_id, :direction, :setup_type,
                        :trigger_type, :trigger_level, :trigger_zone_upper,
                        :trigger_zone_lower, :entry_price, :stop_price, :target_price,
                        :geometry_name, :source_signal_id, :source_watch_id,
                        :invalidation_basis, :target_basis, :state, :registered_at,
                        :expires_at, :signal_snapshot_json, :context_json
                    )
                    """
                ),
                {
                    "trigger_id": trigger.trigger_id,
                    "symbol": trigger.symbol,
                    "profile_id": trigger.profile_id,
                    "direction": trigger.direction,
                    "setup_type": trigger.setup_type,
                    "trigger_type": trigger.trigger_type,
                    "trigger_level": trigger.trigger_level,
                    "trigger_zone_upper": trigger.trigger_zone_upper,
                    "trigger_zone_lower": trigger.trigger_zone_lower,
                    "entry_price": trigger.entry_price,
                    "stop_price": trigger.stop_price,
                    "target_price": trigger.target_price,
                    "geometry_name": trigger.geometry_name,
                    "source_signal_id": trigger.source_signal_id,
                    "source_watch_id": trigger.source_watch_id,
                    "invalidation_basis": trigger.invalidation_basis,
                    "target_basis": trigger.target_basis,
                    "state": "active",
                    "registered_at": trigger.registered_at,
                    "expires_at": trigger.expires_at,
                    "signal_snapshot_json": trigger.signal_snapshot_json,
                    "context_json": trigger.context_json,
                },
            )
            conn.commit()

    def get_active_triggers(self) -> list[TriggerRecord]:
        """SELECT active triggers that have not expired.

        Returns triggers WHERE state='active' AND expires_at > now for
        this registry's profile_id.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._db.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT trigger_id, symbol, profile_id, direction, setup_type,
                           trigger_type, trigger_level, trigger_zone_upper,
                           trigger_zone_lower, entry_price, stop_price, target_price,
                           geometry_name, source_signal_id, source_watch_id,
                           invalidation_basis, target_basis, state, registered_at,
                           expires_at, signal_snapshot_json, context_json
                    FROM fast_path_triggers
                    WHERE state = 'active'
                      AND expires_at > :now
                      AND profile_id = :profile_id
                    """
                ),
                {"now": now, "profile_id": self.profile_id},
            )
            rows = result.fetchall()

        return [
            TriggerRecord(
                trigger_id=row[0],
                symbol=row[1],
                profile_id=row[2],
                direction=row[3],
                setup_type=row[4],
                trigger_type=row[5],
                trigger_level=row[6],
                trigger_zone_upper=row[7],
                trigger_zone_lower=row[8],
                entry_price=row[9],
                stop_price=row[10],
                target_price=row[11],
                geometry_name=row[12],
                source_signal_id=row[13],
                source_watch_id=row[14],
                invalidation_basis=row[15],
                target_basis=row[16],
                state=row[17],
                registered_at=row[18],
                expires_at=row[19],
                signal_snapshot_json=row[20],
                context_json=row[21],
            )
            for row in rows
        ]

    def mark_fired(self, trigger_id: str, event_id: str) -> None:
        """CAS update: active → fired with fired_at, resolved_at, and resolution_event_id.

        Both fired_at and resolved_at are set atomically (same moment) per design.

        Raises:
            FastPathRegistryError: If CAS fails (rowcount != 1).
        """
        now = datetime.now(timezone.utc).isoformat()
        try:
            rowcount = self._execute_mark_fired(trigger_id, event_id, now)
        except FastPathRegistryError:
            raise
        except Exception as e:
            logger.error("mark_fired failed for trigger %s: %s", trigger_id, e)
            raise FastPathRegistryError(
                f"mark_fired failed for trigger {trigger_id}: {e}"
            ) from e

        if rowcount != 1:
            raise FastPathRegistryError(
                f"CAS failed for mark_fired: trigger_id={trigger_id}, "
                f"rowcount={rowcount} (expected 1 — state already changed)"
            )

    @with_lock_retry
    def _execute_mark_fired(self, trigger_id: str, event_id: str, now: str) -> int:
        """Execute the CAS UPDATE for mark_fired. Retried on lock contention."""
        with self._db.connect() as conn:
            result = conn.execute(
                text(
                    """
                    UPDATE fast_path_triggers
                    SET state = 'fired',
                        fired_at = :now,
                        resolved_at = :now,
                        resolution_event_id = :event_id
                    WHERE trigger_id = :trigger_id
                      AND state = 'active'
                    """
                ),
                {
                    "now": now,
                    "event_id": event_id,
                    "trigger_id": trigger_id,
                },
            )
            conn.commit()
            return result.rowcount

    def mark_expired(self, trigger_id: str) -> None:
        """CAS update: active → expired.

        Raises:
            FastPathRegistryError: If CAS fails (rowcount != 1).
        """
        try:
            rowcount = self._execute_mark_expired(trigger_id)
        except FastPathRegistryError:
            raise
        except Exception as e:
            logger.error("mark_expired failed for trigger %s: %s", trigger_id, e)
            raise FastPathRegistryError(
                f"mark_expired failed for trigger {trigger_id}: {e}"
            ) from e

        if rowcount != 1:
            raise FastPathRegistryError(
                f"CAS failed for mark_expired: trigger_id={trigger_id}, "
                f"rowcount={rowcount} (expected 1 — state already changed)"
            )

    @with_lock_retry
    def _execute_mark_expired(self, trigger_id: str) -> int:
        """Execute the CAS UPDATE for mark_expired. Retried on lock contention."""
        with self._db.connect() as conn:
            result = conn.execute(
                text(
                    """
                    UPDATE fast_path_triggers
                    SET state = 'expired'
                    WHERE trigger_id = :trigger_id
                      AND state = 'active'
                    """
                ),
                {"trigger_id": trigger_id},
            )
            conn.commit()
            return result.rowcount

    def mark_invalidated(self, trigger_id: str, reason: str) -> None:
        """CAS update: active → invalidated with invalidation reason in context_json.

        Raises:
            FastPathRegistryError: If CAS fails (rowcount != 1).
        """
        try:
            rowcount = self._execute_mark_invalidated(trigger_id, reason)
        except FastPathRegistryError:
            raise
        except Exception as e:
            logger.error(
                "mark_invalidated failed for trigger %s: %s", trigger_id, e
            )
            raise FastPathRegistryError(
                f"mark_invalidated failed for trigger {trigger_id}: {e}"
            ) from e

        if rowcount != 1:
            raise FastPathRegistryError(
                f"CAS failed for mark_invalidated: trigger_id={trigger_id}, "
                f"rowcount={rowcount} (expected 1 — state already changed)"
            )

    @with_lock_retry
    def _execute_mark_invalidated(self, trigger_id: str, reason: str) -> int:
        """Execute the CAS UPDATE for mark_invalidated. Retried on lock contention."""
        import json

        with self._db.connect() as conn:
            # Read current context_json to merge the invalidation reason
            row = conn.execute(
                text(
                    """
                    SELECT context_json FROM fast_path_triggers
                    WHERE trigger_id = :trigger_id AND state = 'active'
                    """
                ),
                {"trigger_id": trigger_id},
            ).fetchone()

            if row is None:
                return 0  # Trigger not in active state; CAS will fail at caller

            existing_context = {}
            if row[0]:
                try:
                    existing_context = json.loads(row[0])
                except (TypeError, ValueError):
                    existing_context = {}

            existing_context["invalidation_reason"] = reason
            updated_context = json.dumps(existing_context, separators=(",", ":"))

            result = conn.execute(
                text(
                    """
                    UPDATE fast_path_triggers
                    SET state = 'invalidated',
                        context_json = :context_json
                    WHERE trigger_id = :trigger_id
                      AND state = 'active'
                    """
                ),
                {
                    "trigger_id": trigger_id,
                    "context_json": updated_context,
                },
            )
            conn.commit()
            return result.rowcount

    def expire_stale_triggers(self) -> int:
        """Bulk UPDATE: active triggers past expires_at → expired.

        Returns the number of triggers transitioned to expired.
        """
        now = datetime.now(timezone.utc).isoformat()
        try:
            return self._execute_expire_stale(now)
        except Exception as e:
            logger.error("expire_stale_triggers failed: %s", e)
            raise FastPathRegistryError(
                f"expire_stale_triggers failed: {e}"
            ) from e

    @with_lock_retry
    def _execute_expire_stale(self, now: str) -> int:
        """Execute the bulk UPDATE for stale trigger expiry. Retried on lock contention."""
        with self._db.connect() as conn:
            result = conn.execute(
                text(
                    """
                    UPDATE fast_path_triggers
                    SET state = 'expired'
                    WHERE state = 'active'
                      AND expires_at < :now
                      AND profile_id = :profile_id
                    """
                ),
                {"now": now, "profile_id": self.profile_id},
            )
            conn.commit()
            return result.rowcount


def _determine_trigger_type(signal: dict) -> str:
    """Determine trigger_type from signal characteristics.

    Priority:
      1. If signal has entry_zone_upper and entry_zone_lower → entry_zone
      2. If signal references VWAP (setup_type or keywords) → vwap_cross
      3. If signal mentions breakout/break in setup_type → level_break
      4. Default: level_break

    Args:
        signal: Signal dict with setup metadata.

    Returns:
        One of: entry_zone, vwap_cross, level_break.
    """
    # Zone-based trigger
    if signal.get("entry_zone_upper") is not None and signal.get("entry_zone_lower") is not None:
        return "entry_zone"

    setup_type = (signal.get("setup_type") or "").lower()

    # VWAP-based trigger
    if "vwap" in setup_type:
        return "vwap_cross"

    # Level-break trigger
    if "breakout" in setup_type or "break" in setup_type:
        return "level_break"

    # Default
    return "level_break"


def register_triggers_from_signals(
    signals: dict[str, dict], profile_id: str, engine: Any
) -> list[str]:
    """Convert analyst signals to fast-path triggers.

    Filters signals to FAST_PATH_ELIGIBLE_SETUP_TYPES, extracts geometry
    (entry, stop, target) from each signal, determines trigger_type from
    signal characteristics, and registers valid signals as triggers via
    FastPathRegistry.

    Signals with incomplete geometry (missing stop_price or target_price)
    are skipped. Deduplication is handled by the registry (same
    symbol+direction+profile+setup_type with state=active).

    Args:
        signals: Dict mapping symbol names to signal dicts. Each signal
            dict typically has keys: setup_type, direction/side,
            entry_price, stop_price, target_price, signal_id,
            entry_zone_upper, entry_zone_lower, etc.
        profile_id: Profile identifier for trigger registration.
        engine: SQLAlchemy engine instance.

    Returns:
        List of trigger_ids that were successfully registered.

    Requirements: 2.1, 2.2, 2.5, 2.11, 10.7
    """
    from utils.gate_config import (
        FAST_PATH_ELIGIBLE_SETUP_TYPES,
        FAST_PATH_MAX_TRIGGER_AGE_SECONDS,
    )

    registry = FastPathRegistry(db=engine, profile_id=profile_id)
    registered_ids: list[str] = []

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=FAST_PATH_MAX_TRIGGER_AGE_SECONDS)
    now_iso = now.isoformat()
    expires_iso = expires_at.isoformat()

    for symbol, signal in signals.items():
        # Filter: only eligible setup types
        setup_type = signal.get("setup_type", "")
        if setup_type not in FAST_PATH_ELIGIBLE_SETUP_TYPES:
            logger.debug(
                "Skipping signal for %s: setup_type=%r not in eligible set",
                symbol,
                setup_type,
            )
            continue

        # Extract geometry — entry, stop, target
        entry_price = signal.get("entry_price")
        stop_price = signal.get("stop_price")
        target_price = signal.get("target_price")

        # Skip signals with incomplete geometry (missing stop or target)
        if stop_price is None or target_price is None:
            logger.debug(
                "Skipping signal for %s: incomplete geometry "
                "(stop_price=%s, target_price=%s)",
                symbol,
                stop_price,
                target_price,
            )
            continue

        # Entry price defaults to trigger_level; skip if also missing
        if entry_price is None:
            logger.debug(
                "Skipping signal for %s: missing entry_price",
                symbol,
            )
            continue

        # Direction: accept "direction" or "side" key
        direction = signal.get("direction") or signal.get("side", "BUY")
        direction = direction.upper()

        # Determine trigger type from signal characteristics
        trigger_type = _determine_trigger_type(signal)

        # Extract entry zone bounds (if present)
        trigger_zone_upper = signal.get("entry_zone_upper")
        trigger_zone_lower = signal.get("entry_zone_lower")

        # trigger_level is the entry_price
        trigger_level = float(entry_price)

        # Source signal ID
        source_signal_id = signal.get("signal_id")

        # Geometry name (if provided)
        geometry_name = signal.get("geometry_name")

        # Invalidation and target basis
        invalidation_basis = signal.get("invalidation_basis")
        target_basis = signal.get("target_basis")

        # Freeze the full signal as JSON snapshot
        try:
            signal_snapshot_json = json.dumps(signal, default=str, separators=(",", ":"))
        except (TypeError, ValueError) as e:
            logger.warning(
                "Failed to serialize signal snapshot for %s: %s",
                symbol,
                e,
            )
            signal_snapshot_json = None

        # Generate trigger_id
        trigger_id = str(uuid.uuid4())

        # Build TriggerRecord
        trigger_record = TriggerRecord(
            trigger_id=trigger_id,
            symbol=symbol,
            profile_id=profile_id,
            direction=direction,
            setup_type=setup_type,
            trigger_type=trigger_type,
            trigger_level=trigger_level,
            trigger_zone_upper=float(trigger_zone_upper) if trigger_zone_upper is not None else None,
            trigger_zone_lower=float(trigger_zone_lower) if trigger_zone_lower is not None else None,
            entry_price=float(entry_price),
            stop_price=float(stop_price),
            target_price=float(target_price),
            geometry_name=geometry_name,
            source_signal_id=source_signal_id,
            source_watch_id=None,
            invalidation_basis=invalidation_basis,
            target_basis=target_basis,
            state="active",
            registered_at=now_iso,
            expires_at=expires_iso,
            signal_snapshot_json=signal_snapshot_json,
            context_json=None,
        )

        # Register the trigger (deduplication handled by registry)
        try:
            registry.register_trigger(trigger_record)
            registered_ids.append(trigger_id)
            logger.info(
                "Registered fast-path trigger %s for %s (%s, %s, type=%s)",
                trigger_id,
                symbol,
                direction,
                setup_type,
                trigger_type,
            )
        except FastPathRegistryError as e:
            logger.warning(
                "Failed to register trigger for %s: %s",
                symbol,
                e,
            )
        except Exception as e:
            logger.error(
                "Unexpected error registering trigger for %s: %s",
                symbol,
                e,
            )

    logger.info(
        "register_triggers_from_signals: registered %d triggers from %d signals "
        "(profile=%s)",
        len(registered_ids),
        len(signals),
        profile_id,
    )
    return registered_ids


def register_triggers_from_watches(
    promoted_watches: list[dict], profile_id: str, engine: Any
) -> list[str]:
    """Convert promoted watch candidates to fast-path triggers.

    Similar to register_triggers_from_signals but takes promoted watch
    candidates as input. Each watch is linked to its trigger via
    source_watch_id (from the watch's watch_id field). If the watch also
    carries a signal_id, that is preserved as source_signal_id.

    Filters watches to FAST_PATH_ELIGIBLE_SETUP_TYPES. Watches with
    incomplete geometry (missing stop_price or target_price) are skipped.
    Deduplication is handled by the registry (same symbol+direction+
    profile+setup_type with state=active).

    Args:
        promoted_watches: List of dicts representing promoted watch
            candidates. Expected keys: watch_id, symbol, profile_id,
            side/direction, setup_type, entry_price, stop_price,
            target_price, and optionally entry_zone_upper,
            entry_zone_lower, signal_id, geometry_name,
            invalidation_basis, target_basis.
        profile_id: Profile identifier for trigger registration.
        engine: SQLAlchemy engine instance.

    Returns:
        List of trigger_ids that were successfully registered.

    Requirements: 2.11, 3.3
    """
    from utils.gate_config import (
        FAST_PATH_ELIGIBLE_SETUP_TYPES,
        FAST_PATH_MAX_TRIGGER_AGE_SECONDS,
    )

    registry = FastPathRegistry(db=engine, profile_id=profile_id)
    registered_ids: list[str] = []

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=FAST_PATH_MAX_TRIGGER_AGE_SECONDS)
    now_iso = now.isoformat()
    expires_iso = expires_at.isoformat()

    for watch in promoted_watches:
        # Filter: only eligible setup types
        setup_type = watch.get("setup_type", "")
        if setup_type not in FAST_PATH_ELIGIBLE_SETUP_TYPES:
            logger.debug(
                "Skipping promoted watch %s: setup_type=%r not in eligible set",
                watch.get("watch_id", "unknown"),
                setup_type,
            )
            continue

        # Extract geometry — entry, stop, target
        entry_price = watch.get("entry_price")
        stop_price = watch.get("stop_price")
        target_price = watch.get("target_price")

        # Skip watches with incomplete geometry (missing stop or target)
        if stop_price is None or target_price is None:
            logger.debug(
                "Skipping promoted watch %s: incomplete geometry "
                "(stop_price=%s, target_price=%s)",
                watch.get("watch_id", "unknown"),
                stop_price,
                target_price,
            )
            continue

        # Entry price is required
        if entry_price is None:
            logger.debug(
                "Skipping promoted watch %s: missing entry_price",
                watch.get("watch_id", "unknown"),
            )
            continue

        # Symbol
        symbol = watch.get("symbol")
        if not symbol:
            logger.debug(
                "Skipping promoted watch %s: missing symbol",
                watch.get("watch_id", "unknown"),
            )
            continue

        # Direction: accept "direction" or "side" key
        direction = watch.get("direction") or watch.get("side", "BUY")
        direction = direction.upper()

        # Determine trigger type from watch characteristics
        trigger_type = _determine_trigger_type(watch)

        # Extract entry zone bounds (if present)
        trigger_zone_upper = watch.get("entry_zone_upper")
        trigger_zone_lower = watch.get("entry_zone_lower")

        # trigger_level is the entry_price
        trigger_level = float(entry_price)

        # Source linkage: watch_id is the primary source, signal_id if carried
        source_watch_id = watch.get("watch_id")
        source_signal_id = watch.get("signal_id")

        # Geometry name (if provided)
        geometry_name = watch.get("geometry_name")

        # Invalidation and target basis
        invalidation_basis = watch.get("invalidation_basis")
        target_basis = watch.get("target_basis")

        # Freeze the full watch data as JSON snapshot
        try:
            signal_snapshot_json = json.dumps(watch, default=str, separators=(",", ":"))
        except (TypeError, ValueError) as e:
            logger.warning(
                "Failed to serialize watch snapshot for %s: %s",
                symbol,
                e,
            )
            signal_snapshot_json = None

        # Generate trigger_id
        trigger_id = str(uuid.uuid4())

        # Build TriggerRecord
        trigger_record = TriggerRecord(
            trigger_id=trigger_id,
            symbol=symbol,
            profile_id=profile_id,
            direction=direction,
            setup_type=setup_type,
            trigger_type=trigger_type,
            trigger_level=trigger_level,
            trigger_zone_upper=float(trigger_zone_upper) if trigger_zone_upper is not None else None,
            trigger_zone_lower=float(trigger_zone_lower) if trigger_zone_lower is not None else None,
            entry_price=float(entry_price),
            stop_price=float(stop_price),
            target_price=float(target_price),
            geometry_name=geometry_name,
            source_signal_id=source_signal_id,
            source_watch_id=source_watch_id,
            invalidation_basis=invalidation_basis,
            target_basis=target_basis,
            state="active",
            registered_at=now_iso,
            expires_at=expires_iso,
            signal_snapshot_json=signal_snapshot_json,
            context_json=None,
        )

        # Register the trigger (deduplication handled by registry)
        try:
            registry.register_trigger(trigger_record)
            registered_ids.append(trigger_id)
            logger.info(
                "Registered fast-path trigger %s from watch %s for %s "
                "(%s, %s, type=%s)",
                trigger_id,
                source_watch_id,
                symbol,
                direction,
                setup_type,
                trigger_type,
            )
        except FastPathRegistryError as e:
            logger.warning(
                "Failed to register trigger from watch %s for %s: %s",
                source_watch_id,
                symbol,
                e,
            )
        except Exception as e:
            logger.error(
                "Unexpected error registering trigger from watch %s for %s: %s",
                source_watch_id,
                symbol,
                e,
            )

    logger.info(
        "register_triggers_from_watches: registered %d triggers from %d watches "
        "(profile=%s)",
        len(registered_ids),
        len(promoted_watches),
        profile_id,
    )
    return registered_ids
