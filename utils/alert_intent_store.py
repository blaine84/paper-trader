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
    ) -> None:
        """Append an audit record to alert_dispatch_log. Fail-open on write failure."""
        try:
            with self._engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO alert_dispatch_log
                        (alert_intent_id, symbol, alert_type, urgency, dispatch_status,
                         reason, cooldown_remaining_seconds, cycle_trigger_type, dispatch_attempt_count)
                    VALUES (:alert_intent_id, :symbol, :alert_type, :urgency, :dispatch_status,
                            :reason, :cooldown_remaining_seconds, :cycle_trigger_type, :dispatch_attempt_count)
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
                    expiration_at = MAX(expiration_at, excluded.expiration_at)
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
                    deferred_until, dispatch_attempt_count, last_dispatch_error
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
        """Mark intents as dispatched with timestamp. Returns count of rows updated."""
        if not intent_ids:
            return 0
        with self._engine.begin() as conn:
            placeholders = ", ".join(f":id_{i}" for i in range(len(intent_ids)))
            params = {f"id_{i}": id_ for i, id_ in enumerate(intent_ids)}
            params["dispatch_ts"] = dispatch_ts.strftime(_ISO_FMT)
            result = conn.execute(text(f"""
                UPDATE alert_intents
                SET dispatch_status = 'dispatched', dispatched_at = :dispatch_ts
                WHERE id IN ({placeholders})
                    AND dispatch_status = 'pending'
            """), params)
        return result.rowcount

    @with_lock_retry
    def mark_consumed(self, intent_ids: list[int]) -> int:
        """Mark intents as consumed (terminal state). Returns count of rows updated."""
        if not intent_ids:
            return 0
        with self._engine.begin() as conn:
            placeholders = ", ".join(f":id_{i}" for i in range(len(intent_ids)))
            params = {f"id_{i}": id_ for i, id_ in enumerate(intent_ids)}
            result = conn.execute(text(f"""
                UPDATE alert_intents
                SET dispatch_status = 'consumed'
                WHERE id IN ({placeholders})
                    AND dispatch_status IN ('dispatched', 'claimed_by_scheduled')
            """), params)
        return result.rowcount

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
    def mark_suppressed(self, intent_id: int, reason: str) -> None:
        """Mark intent as suppressed (terminal state) with reason."""
        with self._engine.begin() as conn:
            conn.execute(text("""
                UPDATE alert_intents
                SET dispatch_status = 'suppressed', dispatch_reason = :reason
                WHERE id = :intent_id
                    AND dispatch_status = 'pending'
            """), {"intent_id": intent_id, "reason": reason})

    @with_lock_retry
    def mark_dispatch_failed(self, intent_ids: list[int], error: str) -> int:
        """Increment dispatch_attempt_count, set last_dispatch_error.
        If count >= 3, transition to 'dispatch_failed' (terminal).
        Otherwise, back to 'pending'. Returns count of updated rows."""
        if not intent_ids:
            return 0
        with self._engine.begin() as conn:
            placeholders = ", ".join(f":id_{i}" for i in range(len(intent_ids)))
            params = {f"id_{i}": id_ for i, id_ in enumerate(intent_ids)}
            params["error"] = error

            # First increment attempt count and set error for all
            conn.execute(text(f"""
                UPDATE alert_intents
                SET dispatch_attempt_count = dispatch_attempt_count + 1,
                    last_dispatch_error = :error
                WHERE id IN ({placeholders})
            """), params)

            # Then set terminal state for those at/above threshold
            conn.execute(text(f"""
                UPDATE alert_intents
                SET dispatch_status = 'dispatch_failed'
                WHERE id IN ({placeholders})
                    AND dispatch_attempt_count >= 3
                    AND dispatch_status != 'dispatch_failed'
            """), params)

            # Revert remaining to pending
            result = conn.execute(text(f"""
                UPDATE alert_intents
                SET dispatch_status = 'pending', dispatched_at = NULL
                WHERE id IN ({placeholders})
                    AND dispatch_attempt_count < 3
                    AND dispatch_status NOT IN ('dispatch_failed', 'consumed', 'expired', 'suppressed')
            """), params)
        return result.rowcount + len(intent_ids)  # approximate

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
                    deferred_until, dispatch_attempt_count, last_dispatch_error
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
                    deferred_until, dispatch_attempt_count, last_dispatch_error
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
