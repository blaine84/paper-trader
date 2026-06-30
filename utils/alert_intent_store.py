"""Alert Intent Store — durable CRUD for alert intent lifecycle.

Provides record_intent(), update_intent(), query_pending(), mark_dispatched(),
mark_consumed(), mark_expired(), and cooldown management functions.
All writes use short transactions with the project's db_retry pattern.

Requirements: 1.1, 2.1–2.5, 3.1–3.4, 5.1–5.7, 11.1–11.5
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

from sqlalchemy import text

from utils.db_retry import with_lock_retry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlertIntent:
    """Immutable value object for an alert intent record."""

    id: int
    alert_intent_id: str           # UUID4 for cross-reference
    symbol: str                    # max 10 chars
    alert_type: str                # "entry_alert" | "rapid_move"
    direction: Optional[str]       # "long" | "short" | None
    trigger_price: Decimal
    source_level: Optional[str]    # trigger condition when known
    urgency: str                   # "high" | "medium" | "low"
    reason: Optional[str]          # max 500 chars
    dedupe_key: str                # max 128 chars, composite
    filter_status: str             # "unclassified" | "passed" | "failed"
    first_seen_at: datetime        # UTC
    last_seen_at: datetime         # UTC
    occurrence_count: int          # >= 1
    expiration_at: datetime        # UTC
    dispatch_status: str           # "pending"|"dispatched"|"consumed"|"expired"|"suppressed"|"dispatch_failed"|"claimed_by_scheduled"
    dispatch_reason: Optional[str]  # suppression reason
    dispatched_at: Optional[datetime]
    deferred_until: Optional[datetime]  # cooldown re-eligibility time
    occurrence_count_at_deferral: int  # snapshot at deferral time for material change detection
    dispatch_attempt_count: int    # default 0, max 3
    last_dispatch_error: Optional[str]  # error text from last PM failure


@dataclass(frozen=True)
class CooldownRecord:
    """Immutable value object for a persisted cooldown."""

    id: int
    symbol: str                    # symbol or "__GLOBAL__" sentinel
    expiry_at: datetime            # UTC
    started_by_dispatch_id: Optional[int]
    created_at: datetime


def build_dedupe_key(symbol: str, alert_type: str, setup_condition: str) -> str:
    """Deterministic composite key for intent deduplication.

    Normalizes setup_condition by lowercasing and stripping whitespace
    before hashing, so keys remain stable regardless of formatting.

    Format: "{SYMBOL}:{alert_type}:{normalized_hash}"
    Example: "NVDA:entry_alert:a3f2b1c4..."
    """
    normalized = setup_condition.lower().strip()
    condition_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
    return f"{symbol.upper()}:{alert_type}:{condition_hash}"


_GLOBAL_SENTINEL = "__GLOBAL__"

# Terminal lifecycle states — once reached, no further transitions are permitted.
# Requirements: 9.6
TERMINAL_STATES = frozenset({"consumed", "expired", "suppressed", "dispatch_failed"})

# ISO8601 format for writing datetimes to SQLite.
# SQLite strftime('%Y-%m-%dT%H:%M:%fZ', 'now') produces 'YYYY-MM-DDTHH:MM:SS.SSSZ'
# We write in a compatible format using Python's strftime.
_ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _parse_iso_dt(val: Optional[str]) -> Optional[datetime]:
    """Parse ISO8601 datetime strings from SQLite (various formats).

    Handles:
    - SQLite strftime output: '2024-06-24T14:30:00.123Z' (3-digit ms)
    - Python-generated: '2024-06-24T14:30:00.123000Z' (6-digit us)
    - No fractional: '2024-06-24T14:30:00Z'
    """
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    # Try common formats
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",   # Python microseconds (6 digits)
        "%Y-%m-%dT%H:%M:%SZ",       # No fractional
        "%Y-%m-%dT%H:%M:%S.%f",     # Without trailing Z
        "%Y-%m-%dT%H:%M:%S",        # Plain
    ):
        try:
            return datetime.strptime(val, fmt)
        except ValueError:
            continue
    # SQLite strftime '%f' produces 3-digit fractional seconds (e.g., '.571')
    # Python's %f expects 6 digits, so pad if needed
    if "." in val and val.endswith("Z"):
        base, frac = val.rsplit(".", 1)
        frac = frac.rstrip("Z")
        frac_padded = frac.ljust(6, "0")
        try:
            return datetime.strptime(f"{base}.{frac_padded}", "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError:
            pass
    # Last resort: fromisoformat
    return datetime.fromisoformat(val.replace("Z", "+00:00")).replace(tzinfo=None)


class AlertIntentStore:
    """Encapsulates all alert intent and cooldown DB operations."""

    def __init__(self, engine):
        self._engine = engine

    # ─── Audit log ─────────────────────────────────────────────────────────

    def record_audit_log(
        self,
        *,
        alert_intent_id: str,
        symbol: str,
        alert_type: str,
        urgency: str,
        dispatch_status: str,
        reason: Optional[str] = None,
        cooldown_remaining_seconds: Optional[float] = None,
        cycle_trigger_type: Optional[str] = None,
        dispatch_attempt_count: Optional[int] = None,
        freshness_age_seconds: Optional[float] = None,
        first_seen_age_seconds: Optional[float] = None,
        configured_mode: Optional[str] = None,
        dedupe_key: Optional[str] = None,
        trigger_price=None,
        occurrence_count: Optional[int] = None,
        dispatch_batch_symbols: Optional[str] = None,
    ) -> None:
        """Append an audit record to alert_dispatch_log. Fail-open on write failure.

        Requirements: 7.1, 7.2, 7.5
        """
        try:
            with self._engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO alert_dispatch_log
                        (alert_intent_id, symbol, alert_type, urgency, dispatch_status,
                         reason, cooldown_remaining_seconds, cycle_trigger_type,
                         dispatch_attempt_count, freshness_age_seconds,
                         first_seen_age_seconds, configured_mode,
                         dedupe_key, trigger_price, occurrence_count,
                         dispatch_batch_symbols)
                    VALUES (:alert_intent_id, :symbol, :alert_type, :urgency, :dispatch_status,
                            :reason, :cooldown_remaining_seconds, :cycle_trigger_type,
                            :dispatch_attempt_count, :freshness_age_seconds,
                            :first_seen_age_seconds, :configured_mode,
                            :dedupe_key, :trigger_price, :occurrence_count,
                            :dispatch_batch_symbols)
                """), {
                    "alert_intent_id": alert_intent_id,
                    "symbol": symbol,
                    "alert_type": alert_type,
                    "urgency": urgency,
                    "dispatch_status": dispatch_status,
                    "reason": reason,
                    "cooldown_remaining_seconds": cooldown_remaining_seconds,
                    "cycle_trigger_type": cycle_trigger_type,
                    "dispatch_attempt_count": dispatch_attempt_count,
                    "freshness_age_seconds": freshness_age_seconds,
                    "first_seen_age_seconds": first_seen_age_seconds,
                    "configured_mode": configured_mode,
                    "dedupe_key": dedupe_key,
                    "trigger_price": str(trigger_price) if trigger_price is not None else None,
                    "occurrence_count": occurrence_count,
                    "dispatch_batch_symbols": dispatch_batch_symbols,
                })
            # Structured log at INFO level (Requirement 10.1)
            payload = {
                "symbol": symbol,
                "alert_type": alert_type,
                "urgency": urgency,
                "dispatch_status": dispatch_status,
                "reason": reason,
            }
            if cooldown_remaining_seconds is not None:
                payload["cooldown_remaining_seconds"] = cooldown_remaining_seconds
            if cycle_trigger_type is not None:
                payload["cycle_trigger_type"] = cycle_trigger_type
            logger.info("alert_dispatch_event: %s", json.dumps(payload))
        except Exception as exc:
            logger.error(
                "Failed to write audit log: alert_intent_id=%s symbol=%s error=%s",
                alert_intent_id, symbol, str(exc),
            )

    # ------------------------------------------------------------------
    # Cooldown management
    # ------------------------------------------------------------------

    @with_lock_retry
    def is_symbol_cooled(self, symbol: str, *, now: datetime) -> bool:
        """Check if unexpired cooldown exists for the given symbol."""
        with self._engine.begin() as conn:
            row = conn.execute(
                text(
                    "SELECT 1 FROM alert_cooldowns "
                    "WHERE symbol = :symbol AND expiry_at > :now LIMIT 1"
                ),
                {"symbol": symbol, "now": now.strftime(_ISO_FMT)},
            ).fetchone()
            return row is not None

    @with_lock_retry
    def is_global_cooled(self, *, now: datetime) -> bool:
        """Check if unexpired cooldown exists for '__GLOBAL__' sentinel."""
        with self._engine.begin() as conn:
            row = conn.execute(
                text(
                    "SELECT 1 FROM alert_cooldowns "
                    "WHERE symbol = :symbol AND expiry_at > :now LIMIT 1"
                ),
                {"symbol": _GLOBAL_SENTINEL, "now": now.strftime(_ISO_FMT)},
            ).fetchone()
            return row is not None

    @with_lock_retry
    def record_cooldown(self, symbol: str, expiry_at: datetime, dispatch_id: Optional[int] = None) -> None:
        """INSERT into alert_cooldowns with symbol, expiry_at, started_by_dispatch_id."""
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO alert_cooldowns (symbol, expiry_at, started_by_dispatch_id) "
                    "VALUES (:symbol, :expiry_at, :dispatch_id)"
                ),
                {
                    "symbol": symbol,
                    "expiry_at": expiry_at.strftime(_ISO_FMT),
                    "dispatch_id": dispatch_id,
                },
            )

    @with_lock_retry
    def load_active_cooldowns(self, *, now: datetime) -> list[CooldownRecord]:
        """SELECT unexpired cooldowns (expiry_at > now)."""
        with self._engine.begin() as conn:
            rows = conn.execute(
                text(
                    "SELECT id, symbol, expiry_at, started_by_dispatch_id, created_at "
                    "FROM alert_cooldowns WHERE expiry_at > :now"
                ),
                {"now": now.strftime(_ISO_FMT)},
            ).fetchall()
            return [
                CooldownRecord(
                    id=row[0],
                    symbol=row[1],
                    expiry_at=_parse_iso_dt(row[2]),
                    started_by_dispatch_id=row[3],
                    created_at=_parse_iso_dt(row[4]),
                )
                for row in rows
            ]

    @with_lock_retry
    def record_suppression(self, symbol: str, intent_id: int, reason: str, cooldown_expiry: datetime) -> None:
        """Record suppression in alert_cooldowns for audit purposes.
        Also logs structured event."""
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO alert_cooldowns (symbol, expiry_at, started_by_dispatch_id) "
                    "VALUES (:symbol, :expiry_at, :intent_id)"
                ),
                {
                    "symbol": symbol,
                    "expiry_at": cooldown_expiry.strftime(_ISO_FMT),
                    "intent_id": intent_id,
                },
            )
        logger.info(
            "alert_intent_suppressed_by_cooldown: %s",
            {"symbol": symbol, "intent_id": intent_id, "reason": reason, "cooldown_expiry": cooldown_expiry.isoformat()},
        )

    # ------------------------------------------------------------------
    # Crash recovery
    # ------------------------------------------------------------------

    @with_lock_retry
    def recover_stale_active_intents(
        self,
        *,
        dispatch_stale_minutes: int,
        scheduled_max_runtime_minutes: int,
        now: datetime,
    ) -> int:
        """Startup/periodic sweep for crash recovery.

        - dispatched intents older than dispatch_stale_minutes → pending
          (or dispatch_failed if attempt_count >= 3)
        - claimed_by_scheduled intents older than scheduled_max_runtime_minutes → pending

        Returns count of recovered intents.
        Logs recovered_stale_dispatch / recovered_stale_scheduled_claim events.
        """
        now_str = now.strftime(_ISO_FMT)
        recovered = 0

        with self._engine.begin() as conn:
            # Calculate the cutoff for dispatched intents
            dispatch_cutoff = (now - timedelta(minutes=dispatch_stale_minutes)).strftime(_ISO_FMT)
            scheduled_cutoff = (now - timedelta(minutes=scheduled_max_runtime_minutes)).strftime(_ISO_FMT)

            # Recover dispatched intents that are stale and have attempt_count >= 3 → dispatch_failed
            result = conn.execute(
                text(
                    "UPDATE alert_intents SET dispatch_status = 'dispatch_failed' "
                    "WHERE dispatch_status = 'dispatched' "
                    "AND dispatched_at < :cutoff "
                    "AND dispatch_attempt_count >= 3"
                ),
                {"cutoff": dispatch_cutoff},
            )
            failed_count = result.rowcount
            if failed_count > 0:
                logger.warning(
                    "recovered_stale_dispatch: %s",
                    {"count": failed_count, "action": "dispatch_failed", "now": now_str},
                )
            recovered += failed_count

            # Recover dispatched intents that are stale and have attempt_count < 3 → pending
            result = conn.execute(
                text(
                    "UPDATE alert_intents SET dispatch_status = 'pending', dispatched_at = NULL "
                    "WHERE dispatch_status = 'dispatched' "
                    "AND dispatched_at < :cutoff "
                    "AND dispatch_attempt_count < 3"
                ),
                {"cutoff": dispatch_cutoff},
            )
            pending_count = result.rowcount
            if pending_count > 0:
                logger.warning(
                    "recovered_stale_dispatch: %s",
                    {"count": pending_count, "action": "revert_to_pending", "now": now_str},
                )
            recovered += pending_count

            # Recover claimed_by_scheduled intents stuck beyond timeout → pending
            result = conn.execute(
                text(
                    "UPDATE alert_intents SET dispatch_status = 'pending', dispatched_at = NULL "
                    "WHERE dispatch_status = 'claimed_by_scheduled' "
                    "AND dispatched_at < :cutoff"
                ),
                {"cutoff": scheduled_cutoff},
            )
            claimed_count = result.rowcount
            if claimed_count > 0:
                logger.warning(
                    "recovered_stale_scheduled_claim: %s",
                    {"count": claimed_count, "action": "revert_to_pending", "now": now_str},
                )
            recovered += claimed_count

        return recovered

    def _row_to_intent(self, row) -> AlertIntent:
        """Convert a SQLAlchemy Row to an AlertIntent dataclass.

        Parses ISO8601 datetime strings to datetime objects and
        Decimal from text. Reused by all query methods.
        """
        return AlertIntent(
            id=row.id,
            alert_intent_id=row.alert_intent_id,
            symbol=row.symbol,
            alert_type=row.alert_type,
            direction=row.direction,
            trigger_price=Decimal(row.trigger_price) if row.trigger_price is not None else Decimal("0"),
            source_level=row.source_level,
            urgency=row.urgency,
            reason=row.reason,
            dedupe_key=row.dedupe_key,
            filter_status=row.filter_status,
            first_seen_at=_parse_iso_dt(row.first_seen_at),
            last_seen_at=_parse_iso_dt(row.last_seen_at),
            occurrence_count=row.occurrence_count,
            expiration_at=_parse_iso_dt(row.expiration_at),
            dispatch_status=row.dispatch_status,
            dispatch_reason=row.dispatch_reason,
            dispatched_at=_parse_iso_dt(row.dispatched_at),
            deferred_until=_parse_iso_dt(row.deferred_until),
            occurrence_count_at_deferral=row.occurrence_count_at_deferral if hasattr(row, 'occurrence_count_at_deferral') else 0,
            dispatch_attempt_count=row.dispatch_attempt_count,
            last_dispatch_error=row.last_dispatch_error,
        )

    @with_lock_retry
    def record_or_update_intent(self, intent_data: dict) -> AlertIntent:
        """UPSERT via INSERT ... ON CONFLICT against partial unique index.

        Inserts a new alert intent or updates an existing one matching the
        dedupe_key with active dispatch_status. On conflict: updates last_seen_at,
        increments occurrence_count, and extends expiration_at via MAX.

        Args:
            intent_data: dict with keys: symbol, alert_type, direction,
                trigger_price, source_level, urgency, reason, dedupe_key,
                filter_status, first_seen_at, last_seen_at, expiration_at

        Returns:
            The resulting AlertIntent (newly inserted or updated).
        """
        alert_intent_id = str(uuid.uuid4())

        # Apply defaults
        filter_status = intent_data.get("filter_status", "unclassified")
        urgency = intent_data.get("urgency", "medium")

        with self._engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO alert_intents (
                    alert_intent_id, symbol, alert_type, direction, trigger_price,
                    source_level, urgency, reason, dedupe_key, filter_status,
                    first_seen_at, last_seen_at, occurrence_count, expiration_at,
                    dispatch_status, dispatch_attempt_count
                )
                VALUES (
                    :alert_intent_id, :symbol, :alert_type, :direction, :trigger_price,
                    :source_level, :urgency, :reason, :dedupe_key, :filter_status,
                    :first_seen_at, :last_seen_at, 1, :expiration_at,
                    'pending', 0
                )
                ON CONFLICT(dedupe_key)
                    WHERE dispatch_status IN ('pending', 'dispatched', 'claimed_by_scheduled')
                DO UPDATE SET
                    last_seen_at = excluded.last_seen_at,
                    occurrence_count = occurrence_count + 1,
                    expiration_at = MAX(expiration_at, excluded.expiration_at),
                    trigger_price = excluded.trigger_price,
                    direction = excluded.direction,
                    source_level = excluded.source_level
            """), {
                "alert_intent_id": alert_intent_id,
                "symbol": intent_data["symbol"],
                "alert_type": intent_data["alert_type"],
                "direction": intent_data.get("direction"),
                "trigger_price": str(intent_data["trigger_price"]),
                "source_level": intent_data.get("source_level"),
                "urgency": urgency,
                "reason": intent_data.get("reason"),
                "dedupe_key": intent_data["dedupe_key"],
                "filter_status": filter_status,
                "first_seen_at": intent_data["first_seen_at"],
                "last_seen_at": intent_data["last_seen_at"],
                "expiration_at": intent_data["expiration_at"],
            })

            # SELECT the row back to return a full AlertIntent
            row = conn.execute(text("""
                SELECT id, alert_intent_id, symbol, alert_type, direction,
                    trigger_price, source_level, urgency, reason, dedupe_key,
                    filter_status, first_seen_at, last_seen_at, occurrence_count,
                    expiration_at, dispatch_status, dispatch_reason, dispatched_at,
                    deferred_until, occurrence_count_at_deferral,
                    dispatch_attempt_count, last_dispatch_error
                FROM alert_intents
                WHERE dedupe_key = :dedupe_key
                    AND dispatch_status IN ('pending', 'dispatched', 'claimed_by_scheduled')
            """), {"dedupe_key": intent_data["dedupe_key"]}).fetchone()

        return self._row_to_intent(row)

    # ------------------------------------------------------------------
    # State transition methods
    # ------------------------------------------------------------------

    @with_lock_retry
    def mark_dispatched(self, intent_ids: list[int], dispatch_ts: datetime) -> int:
        """Mark intents as dispatched with timestamp. Returns count of rows updated.

        Terminal-state intents (consumed, expired, suppressed, dispatch_failed) are
        excluded from the update. A WARNING is logged if any intents were skipped.

        Requirements: 9.3, 9.6
        """
        if not intent_ids:
            return 0
        with self._engine.begin() as conn:
            placeholders = ", ".join(f":id_{i}" for i in range(len(intent_ids)))
            params = {f"id_{i}": id_ for i, id_ in enumerate(intent_ids)}
            params["dispatch_ts"] = dispatch_ts.strftime(_ISO_FMT)

            # Query symbol/alert_type for logging (only pending rows will transition)
            info_rows = conn.execute(text(f"""
                SELECT id, symbol, alert_type FROM alert_intents
                WHERE id IN ({placeholders}) AND dispatch_status = 'pending'
            """), params).fetchall()

            result = conn.execute(text(f"""
                UPDATE alert_intents
                SET dispatch_status = 'dispatched', dispatched_at = :dispatch_ts
                WHERE id IN ({placeholders})
                    AND dispatch_status = 'pending'
            """), params)
        updated = result.rowcount
        skipped = len(intent_ids) - updated
        if skipped > 0:
            logger.warning(
                "mark_dispatched: %d of %d intents skipped (already in terminal or "
                "non-pending state, attempted transition to dispatched)",
                skipped, len(intent_ids),
            )

        # Emit structured state transition logs (Requirement 9.3)
        for info_row in info_rows:
            logger.info(
                "ALERT_STATE_TRANSITION: symbol=%s alert_type=%s old_status=%s new_status=%s reason=%s",
                info_row[1], info_row[2], "pending", "dispatched", "dispatch_batch",
            )

        return updated

    @with_lock_retry
    def mark_consumed(self, intent_ids: list[int]) -> int:
        """Mark intents as consumed (terminal state). Returns count of rows updated.

        Only transitions intents in dispatched or claimed_by_scheduled states.
        Terminal-state intents (consumed, expired, suppressed, dispatch_failed) are
        excluded from the update. A WARNING is logged if any intents were skipped.

        Requirements: 9.3, 9.6
        """
        if not intent_ids:
            return 0
        with self._engine.begin() as conn:
            placeholders = ", ".join(f":id_{i}" for i in range(len(intent_ids)))
            params = {f"id_{i}": id_ for i, id_ in enumerate(intent_ids)}

            # Query symbol/alert_type/old_status for logging before UPDATE
            info_rows = conn.execute(text(f"""
                SELECT id, symbol, alert_type, dispatch_status FROM alert_intents
                WHERE id IN ({placeholders})
                    AND dispatch_status IN ('dispatched', 'claimed_by_scheduled')
            """), params).fetchall()

            result = conn.execute(text(f"""
                UPDATE alert_intents
                SET dispatch_status = 'consumed'
                WHERE id IN ({placeholders})
                    AND dispatch_status IN ('dispatched', 'claimed_by_scheduled')
            """), params)
        updated = result.rowcount
        skipped = len(intent_ids) - updated
        if skipped > 0:
            logger.warning(
                "mark_consumed: %d of %d intents skipped (already in terminal or "
                "ineligible state, attempted transition to consumed)",
                skipped, len(intent_ids),
            )

        # Emit structured state transition logs (Requirement 9.3)
        for info_row in info_rows:
            logger.info(
                "ALERT_STATE_TRANSITION: symbol=%s alert_type=%s old_status=%s new_status=%s reason=%s",
                info_row[1], info_row[2], info_row[3], "consumed", "pm_processing_complete",
            )

        return updated

    @with_lock_retry
    def mark_expired(self, *, now: datetime) -> int:
        """Mark expired intents where expiration_at < now and in active states. Returns count."""
        with self._engine.begin() as conn:
            result = conn.execute(text("""
                UPDATE alert_intents
                SET dispatch_status = 'expired'
                WHERE expiration_at < :now
                    AND dispatch_status IN ('pending', 'dispatched', 'claimed_by_scheduled')
            """), {"now": now.strftime(_ISO_FMT)})
        return result.rowcount

    @with_lock_retry
    def query_active_past_expiration(self, *, now: datetime) -> list[AlertIntent]:
        """Query active intents whose expiration_at has passed.

        Returns intents in active states (pending, dispatched, claimed_by_scheduled)
        where expiration_at < now. Used by the dispatcher to individually expire
        intents with audit records and PM-cycle protection.

        Requirements: 4.1, 4.3
        """
        with self._engine.begin() as conn:
            rows = conn.execute(text("""
                SELECT id, alert_intent_id, symbol, alert_type, direction,
                    trigger_price, source_level, urgency, reason, dedupe_key,
                    filter_status, first_seen_at, last_seen_at, occurrence_count,
                    expiration_at, dispatch_status, dispatch_reason, dispatched_at,
                    deferred_until, occurrence_count_at_deferral,
                    dispatch_attempt_count, last_dispatch_error
                FROM alert_intents
                WHERE expiration_at < :now
                    AND dispatch_status IN ('pending', 'dispatched', 'claimed_by_scheduled')
                ORDER BY expiration_at ASC
            """), {"now": now.strftime(_ISO_FMT)}).fetchall()
        return [self._row_to_intent(row) for row in rows]

    @with_lock_retry
    def mark_suppressed(self, intent_id: int, reason: str) -> None:
        """Mark intent as suppressed (terminal state) with reason.

        Only transitions intents in pending state. Terminal-state intents
        (consumed, expired, suppressed, dispatch_failed) are excluded.
        A WARNING is logged if the intent was skipped due to terminal state.

        Requirements: 9.3, 9.6
        """
        with self._engine.begin() as conn:
            # Query symbol/alert_type/old_status for logging before UPDATE
            info_row = conn.execute(text("""
                SELECT symbol, alert_type, dispatch_status FROM alert_intents
                WHERE id = :intent_id
            """), {"intent_id": intent_id}).fetchone()

            result = conn.execute(text("""
                UPDATE alert_intents
                SET dispatch_status = 'suppressed', dispatch_reason = :reason
                WHERE id = :intent_id
                    AND dispatch_status NOT IN ('consumed', 'expired', 'suppressed', 'dispatch_failed')
            """), {"intent_id": intent_id, "reason": reason})

        if result.rowcount == 0:
            logger.warning(
                "mark_suppressed: intent_id=%d skipped — already in terminal state, "
                "attempted transition to suppressed (reason=%s)",
                intent_id, reason,
            )
            return

        # Emit structured state transition log (Requirement 9.3)
        symbol = info_row[0] if info_row else "unknown"
        alert_type = info_row[1] if info_row else "unknown"
        old_status = info_row[2] if info_row else "unknown"
        truncated_reason = reason[:200] if reason else ""
        logger.info(
            "ALERT_STATE_TRANSITION: symbol=%s alert_type=%s old_status=%s new_status=%s reason=%s",
            symbol, alert_type, old_status, "suppressed", truncated_reason,
        )

    @with_lock_retry
    def transition_to_expired(self, intent_id: int, reason: str) -> bool:
        """Transition an active intent to expired with a reason.

        Only transitions intents whose dispatch_status is in the active set:
        ('pending', 'dispatched', 'claimed_by_scheduled').

        If the intent is already in a terminal state (consumed, expired,
        suppressed, dispatch_failed), logs a WARNING and returns False.

        Returns True on successful transition, False if intent was already terminal.

        Requirements: 3.3, 4.1, 9.3, 9.6
        """
        with self._engine.begin() as conn:
            # Query old status, symbol, alert_type before transition for logging
            row = conn.execute(text("""
                SELECT dispatch_status, symbol, alert_type FROM alert_intents
                WHERE id = :intent_id
            """), {"intent_id": intent_id}).fetchone()

            old_status = row[0] if row else "unknown"
            symbol = row[1] if row else "unknown"
            alert_type = row[2] if row else "unknown"

            result = conn.execute(text("""
                UPDATE alert_intents
                SET dispatch_status = 'expired', dispatch_reason = :reason
                WHERE id = :intent_id
                    AND dispatch_status IN ('pending', 'dispatched', 'claimed_by_scheduled')
            """), {"intent_id": intent_id, "reason": reason})

        if result.rowcount == 0:
            logger.warning(
                "transition_to_expired: intent_id=%d already in terminal state, "
                "cannot transition to expired (reason=%s)",
                intent_id,
                reason,
            )
            return False

        truncated_reason = reason[:200] if reason else ""
        logger.info(
            "ALERT_STATE_TRANSITION: symbol=%s alert_type=%s old_status=%s new_status=%s reason=%s",
            symbol, alert_type, old_status, "expired", truncated_reason,
        )
        return True

    @with_lock_retry
    def mark_dispatch_failed(self, intent_ids: list[int], error: str) -> int:
        """Increment dispatch_attempt_count, set last_dispatch_error.
        If count >= 3, transition to 'dispatch_failed' (terminal).
        Otherwise, back to 'pending'. Returns count of updated rows.

        Terminal-state intents (consumed, expired, suppressed, dispatch_failed) are
        excluded from all updates. A WARNING is logged if any intents were skipped.

        Requirements: 9.3, 9.6
        """
        if not intent_ids:
            return 0
        with self._engine.begin() as conn:
            placeholders = ", ".join(f":id_{i}" for i in range(len(intent_ids)))
            params = {f"id_{i}": id_ for i, id_ in enumerate(intent_ids)}
            params["error"] = error

            # Query symbol/alert_type/old_status for logging before UPDATE
            info_rows = conn.execute(text(f"""
                SELECT id, symbol, alert_type, dispatch_status FROM alert_intents
                WHERE id IN ({placeholders})
                    AND dispatch_status NOT IN ('consumed', 'expired', 'suppressed', 'dispatch_failed')
            """), params).fetchall()

            # Increment attempt count and set error — only for non-terminal intents
            conn.execute(text(f"""
                UPDATE alert_intents
                SET dispatch_attempt_count = dispatch_attempt_count + 1,
                    last_dispatch_error = :error
                WHERE id IN ({placeholders})
                    AND dispatch_status NOT IN ('consumed', 'expired', 'suppressed', 'dispatch_failed')
            """), params)

            # Then set terminal state for those at/above threshold
            conn.execute(text(f"""
                UPDATE alert_intents
                SET dispatch_status = 'dispatch_failed'
                WHERE id IN ({placeholders})
                    AND dispatch_attempt_count >= 3
                    AND dispatch_status NOT IN ('consumed', 'expired', 'suppressed', 'dispatch_failed')
            """), params)

            # Revert remaining to pending
            result = conn.execute(text(f"""
                UPDATE alert_intents
                SET dispatch_status = 'pending', dispatched_at = NULL
                WHERE id IN ({placeholders})
                    AND dispatch_attempt_count < 3
                    AND dispatch_status NOT IN ('dispatch_failed', 'consumed', 'expired', 'suppressed')
            """), params)

            # Query final states for accurate transition logging
            final_rows = conn.execute(text(f"""
                SELECT id, dispatch_status FROM alert_intents
                WHERE id IN ({placeholders})
            """), params).fetchall()

        # Build lookup for final states
        final_status_map = {row[0]: row[1] for row in final_rows}
        truncated_error = error[:200] if error else ""

        # Emit structured state transition logs (Requirement 9.3)
        for info_row in info_rows:
            intent_id_val = info_row[0]
            symbol = info_row[1]
            alert_type = info_row[2]
            old_status = info_row[3]
            final_status = final_status_map.get(intent_id_val, "unknown")
            # Only log if status actually changed
            if old_status != final_status:
                logger.info(
                    "ALERT_STATE_TRANSITION: symbol=%s alert_type=%s old_status=%s new_status=%s reason=%s",
                    symbol, alert_type, old_status, final_status, truncated_error,
                )

        # Check if any intents were potentially in terminal state
        # The first UPDATE only touches non-terminal intents; if rowcount < len(intent_ids),
        # some were skipped (already terminal)
        total_processed = result.rowcount + len(intent_ids)  # approximate
        if result.rowcount == 0 and len(intent_ids) > 0:
            logger.warning(
                "mark_dispatch_failed: some intents may have been skipped due to "
                "terminal state (attempted transition for %d intents, error=%s)",
                len(intent_ids), error,
            )
        return total_processed

    @with_lock_retry
    def mark_claimed_by_scheduled(self, intent_ids: list[int]) -> int:
        """Mark pending intents as claimed_by_scheduled. Returns count of rows updated."""
        if not intent_ids:
            return 0
        with self._engine.begin() as conn:
            placeholders = ", ".join(f":id_{i}" for i in range(len(intent_ids)))
            params = {f"id_{i}": id_ for i, id_ in enumerate(intent_ids)}
            params["dispatch_ts"] = datetime.utcnow().strftime(_ISO_FMT)
            result = conn.execute(text(f"""
                UPDATE alert_intents
                SET dispatch_status = 'claimed_by_scheduled', dispatched_at = :dispatch_ts
                WHERE id IN ({placeholders})
                    AND dispatch_status = 'pending'
            """), params)
        return result.rowcount

    @with_lock_retry
    def mark_claimed_back_to_pending(self, intent_ids: list[int], error: str) -> int:
        """Revert claimed_by_scheduled → pending on scheduled PM failure. Returns count."""
        if not intent_ids:
            return 0
        with self._engine.begin() as conn:
            placeholders = ", ".join(f":id_{i}" for i in range(len(intent_ids)))
            params = {f"id_{i}": id_ for i, id_ in enumerate(intent_ids)}
            params["error"] = error
            result = conn.execute(text(f"""
                UPDATE alert_intents
                SET dispatch_status = 'pending', dispatched_at = NULL, last_dispatch_error = :error
                WHERE id IN ({placeholders})
                    AND dispatch_status = 'claimed_by_scheduled'
            """), params)
        return result.rowcount

    @with_lock_retry
    def query_pending(self, *, now: datetime) -> list[AlertIntent]:
        """SELECT pending classified intents that haven't expired.

        Returns intents where:
        - dispatch_status = 'pending'
        - filter_status != 'unclassified' (already classified)
        - expiration_at > now (not expired)

        Ordered by urgency ASC (high < low < medium alphabetically,
        so high-urgency intents come first), then first_seen_at ASC for FIFO.
        """
        with self._engine.begin() as conn:
            rows = conn.execute(text("""
                SELECT id, alert_intent_id, symbol, alert_type, direction,
                    trigger_price, source_level, urgency, reason, dedupe_key,
                    filter_status, first_seen_at, last_seen_at, occurrence_count,
                    expiration_at, dispatch_status, dispatch_reason, dispatched_at,
                    deferred_until, occurrence_count_at_deferral,
                    dispatch_attempt_count, last_dispatch_error
                FROM alert_intents
                WHERE dispatch_status = 'pending'
                    AND filter_status != 'unclassified'
                    AND expiration_at > :now
                ORDER BY urgency ASC, first_seen_at ASC
            """), {"now": now.strftime(_ISO_FMT)}).fetchall()
        return [self._row_to_intent(row) for row in rows]

    @with_lock_retry
    def query_unclassified(self) -> list[AlertIntent]:
        """SELECT unclassified intents, limited by PM_ALERT_CLASSIFY_MAX_PER_PASS."""
        from utils.gate_config import PM_ALERT_CLASSIFY_MAX_PER_PASS

        with self._engine.begin() as conn:
            rows = conn.execute(text("""
                SELECT id, alert_intent_id, symbol, alert_type, direction,
                    trigger_price, source_level, urgency, reason, dedupe_key,
                    filter_status, first_seen_at, last_seen_at, occurrence_count,
                    expiration_at, dispatch_status, dispatch_reason, dispatched_at,
                    deferred_until, occurrence_count_at_deferral,
                    dispatch_attempt_count, last_dispatch_error
                FROM alert_intents
                WHERE filter_status = 'unclassified'
                    AND dispatch_status IN ('pending', 'dispatched', 'claimed_by_scheduled')
                ORDER BY first_seen_at ASC
                LIMIT :limit
            """), {"limit": PM_ALERT_CLASSIFY_MAX_PER_PASS}).fetchall()
        return [self._row_to_intent(row) for row in rows]

    @with_lock_retry
    def update_classification(self, intent_id: int, filter_status: str, urgency: str) -> None:
        """UPDATE filter_status and urgency for a given intent_id."""
        with self._engine.begin() as conn:
            conn.execute(text("""
                UPDATE alert_intents
                SET filter_status = :filter_status, urgency = :urgency
                WHERE id = :intent_id
            """), {
                "intent_id": intent_id,
                "filter_status": filter_status,
                "urgency": urgency,
            })

    @with_lock_retry
    def query_by_lifecycle(
        self,
        *,
        status_list: Optional[list[str]] = None,
        alert_type: Optional[str] = None,
        symbol: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> list[AlertIntent]:
        """Query alert intents by lifecycle filters.

        Supports filtering by any combination of dispatch_status, alert_type,
        symbol, and a time range (start/end in ISO format against first_seen_at).
        Returns matching records ordered by first_seen_at DESC, limited to 1000 rows.
        Returns empty list when no matches found (never raises for empty results).

        Requirements: 9.4, 9.5
        """
        clauses: list[str] = []
        params: dict = {}

        if status_list:
            # Expand list into individual named params for SQLite compatibility
            status_placeholders = ", ".join(
                f":status_{i}" for i in range(len(status_list))
            )
            clauses.append(f"dispatch_status IN ({status_placeholders})")
            for i, status in enumerate(status_list):
                params[f"status_{i}"] = status

        if alert_type is not None:
            clauses.append("alert_type = :alert_type")
            params["alert_type"] = alert_type

        if symbol is not None:
            clauses.append("symbol = :symbol")
            params["symbol"] = symbol

        if start_time is not None:
            clauses.append("first_seen_at >= :start_time")
            params["start_time"] = start_time

        if end_time is not None:
            clauses.append("first_seen_at <= :end_time")
            params["end_time"] = end_time

        where_clause = ""
        if clauses:
            where_clause = "WHERE " + " AND ".join(clauses)

        sql = f"""
            SELECT id, alert_intent_id, symbol, alert_type, direction,
                trigger_price, source_level, urgency, reason, dedupe_key,
                filter_status, first_seen_at, last_seen_at, occurrence_count,
                expiration_at, dispatch_status, dispatch_reason, dispatched_at,
                deferred_until, occurrence_count_at_deferral,
                dispatch_attempt_count, last_dispatch_error
            FROM alert_intents
            {where_clause}
            ORDER BY first_seen_at DESC
            LIMIT 1000
        """

        with self._engine.begin() as conn:
            rows = conn.execute(text(sql), params).fetchall()

        return [self._row_to_intent(row) for row in rows]

    @with_lock_retry
    def set_deferred_until(
        self, intent_id: int, deferred_until: datetime, occurrence_count_at_deferral: int
    ) -> None:
        """Set deferred_until and snapshot of occurrence_count at deferral time.

        After any dispatch evaluation outcome, the dispatcher calls this to prevent
        re-evaluation until the cooldown expires or occurrence_count changes.

        Args:
            intent_id: SQLite row id of the alert intent.
            deferred_until: Cooldown expiry timestamp (stored as ISO 8601 TEXT).
            occurrence_count_at_deferral: Snapshot of occurrence_count at deferral time,
                used to detect material changes that should break the deferral.
        """
        with self._engine.begin() as conn:
            conn.execute(text("""
                UPDATE alert_intents
                SET deferred_until = :deferred_until,
                    occurrence_count_at_deferral = :occ_count
                WHERE id = :intent_id
            """), {
                "intent_id": intent_id,
                "deferred_until": deferred_until.strftime(_ISO_FMT),
                "occ_count": occurrence_count_at_deferral,
            })

    @with_lock_retry
    def has_would_dispatch_for_occurrence(
        self,
        alert_intent_id: str,
        dedupe_key: str,
        trigger_price,
        occurrence_count: int,
    ) -> bool:
        """Check if a would_dispatch audit row exists for this exact occurrence state.

        Used for observation deduplication — if we already observed this exact
        occurrence (same alert_intent_id, dedupe_key, trigger_price, and
        occurrence_count), we don't write another would_dispatch row.

        Args:
            alert_intent_id: UUID4 string identifier of the alert intent.
            dedupe_key: Composite deduplication key for the intent.
            trigger_price: The trigger price (Decimal, str, or float).
            occurrence_count: The current occurrence count of the intent.

        Returns:
            True if a matching would_dispatch row exists, False otherwise.

        Requirements: 2.3, 2.4, 8.2
        """
        with self._engine.begin() as conn:
            row = conn.execute(text("""
                SELECT 1 FROM alert_dispatch_log
                WHERE alert_intent_id = :aid
                    AND dispatch_status = 'would_dispatch'
                    AND dedupe_key = :dk
                    AND trigger_price = :tp
                    AND occurrence_count = :oc
                LIMIT 1
            """), {
                "aid": alert_intent_id,
                "dk": dedupe_key,
                "tp": str(trigger_price),
                "oc": occurrence_count,
            }).fetchone()
        return row is not None

    @with_lock_retry
    def get_latest_would_dispatch_trigger_price(
        self,
        alert_intent_id: str,
    ) -> Optional[Decimal]:
        """Get the trigger_price from the most recent would_dispatch row for an intent.

        Used for material change detection — comparing the current trigger_price
        against the last observed price to determine if the change exceeds the
        0.5% threshold.

        Args:
            alert_intent_id: UUID4 string identifier of the alert intent.

        Returns:
            The trigger_price as Decimal from the most recent would_dispatch row,
            or None if no would_dispatch row exists for this intent.

        Requirements: 8.3
        """
        with self._engine.begin() as conn:
            row = conn.execute(text("""
                SELECT trigger_price FROM alert_dispatch_log
                WHERE alert_intent_id = :aid
                    AND dispatch_status = 'would_dispatch'
                    AND trigger_price IS NOT NULL
                ORDER BY id DESC
                LIMIT 1
            """), {"aid": alert_intent_id}).fetchone()
        if row is None or row[0] is None:
            return None
        try:
            return Decimal(row[0])
        except Exception:
            return None
