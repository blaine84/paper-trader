"""Shadow ledger for blocked trade candidates — Phase 0 capture layer."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

log = logging.getLogger(__name__)


def ensure_shadow_ledger_schema(engine) -> None:
    """Create blocked_trade_candidates table and indexes if they don't exist.

    Called once at system startup. Logs errors but never raises —
    a schema failure must not prevent the system from starting.
    """
    try:
        with engine.connect() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS blocked_trade_candidates (
                        id INTEGER PRIMARY KEY,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        symbol VARCHAR(10),
                        action VARCHAR(16) NOT NULL,
                        direction VARCHAR(10),
                        profile VARCHAR(16),
                        setup_type VARCHAR(64),
                        entry_price REAL,
                        stop_price REAL,
                        target_price REAL,
                        quantity REAL,
                        blocked_by VARCHAR(64) NOT NULL,
                        block_reason TEXT NOT NULL,
                        reason_code VARCHAR(64),
                        gate_notes_json TEXT,
                        decision_snapshot_json TEXT,
                        signal_snapshot_json TEXT,
                        source VARCHAR(64),
                        agent VARCHAR(64),
                        dedupe_key VARCHAR(255),
                        trade_event_id INTEGER REFERENCES trade_events(id) ON DELETE SET NULL
                    )
                    """
                )
            )

            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_blocked_created_at
                    ON blocked_trade_candidates (created_at)
                    """
                )
            )

            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_blocked_symbol_created
                    ON blocked_trade_candidates (symbol, created_at)
                    """
                )
            )

            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_blocked_by_created
                    ON blocked_trade_candidates (blocked_by, created_at)
                    """
                )
            )

            conn.execute(
                text(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS uq_blocked_dedupe
                    ON blocked_trade_candidates (dedupe_key)
                    WHERE dedupe_key IS NOT NULL
                    """
                )
            )

            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS blocked_trade_candidate_outcomes (
                        id INTEGER PRIMARY KEY,
                        blocked_candidate_id INTEGER NOT NULL REFERENCES blocked_trade_candidates(id) ON DELETE CASCADE,
                        eval_window VARCHAR(16) NOT NULL,
                        evaluated_at DATETIME NOT NULL,
                        eval_price REAL,
                        pnl_pct REAL,
                        mfe_pct REAL,
                        mae_pct REAL,
                        stop_hit BOOLEAN DEFAULT 0,
                        target_hit BOOLEAN DEFAULT 0,
                        first_hit VARCHAR(16),
                        first_hit_at DATETIME,
                        outcome_label VARCHAR(64),
                        gate_verdict VARCHAR(64),
                        notes_json TEXT,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(blocked_candidate_id, eval_window)
                    )
                    """
                )
            )

            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_blocked_outcomes_candidate
                    ON blocked_trade_candidate_outcomes (blocked_candidate_id)
                    """
                )
            )

            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_blocked_outcomes_verdict_created
                    ON blocked_trade_candidate_outcomes (gate_verdict, created_at)
                    """
                )
            )

            conn.commit()
            log.info("Shadow ledger schema ensured successfully.")
    except Exception as exc:
        log.error("Failed to ensure shadow ledger schema: %s", exc)


_ET = ZoneInfo("America/New_York")


def compute_dedupe_key(
    *,
    profile: str | None,
    symbol: str | None,
    action: str,
    setup_type: str | None,
    blocked_by: str,
    entry_price: float | None,
    stop_price: float | None,
    target_price: float | None,
    block_reason: str,
) -> str:
    """Compute a deterministic deduplication key for a blocked candidate.

    Uses the current market session date (America/New_York) to prevent
    ambiguity near UTC midnight.

    Components: date (market session date in America/New_York), profile, symbol,
    action, setup_type, blocked_by, round(entry_price, 2), round(stop_price, 2),
    round(target_price, 2), sha256(block_reason)[:16].

    Returns a pipe-delimited string ≤255 chars.
    """
    # Market session date in America/New_York
    session_date = datetime.now(_ET).strftime("%Y-%m-%d")

    # Substitute "none" for None values
    prof = profile if profile is not None else "none"
    sym = symbol.upper() if symbol is not None else "none"
    act = action.upper()
    setup = setup_type if setup_type is not None else "none"

    # Round prices to 2 decimal places, "none" if null
    entry = f"{round(entry_price, 2)}" if entry_price is not None else "none"
    stop = f"{round(stop_price, 2)}" if stop_price is not None else "none"
    target = f"{round(target_price, 2)}" if target_price is not None else "none"

    # SHA-256 hash of block_reason, first 16 chars
    reason_hash = hashlib.sha256(block_reason.encode("utf-8")).hexdigest()[:16]

    key = "|".join([
        session_date,
        prof,
        sym,
        act,
        setup,
        blocked_by,
        entry,
        stop,
        target,
        reason_hash,
    ])

    # Ensure output ≤255 chars by truncating if necessary
    return key[:255]


def _count_scaffold_entries_this_cycle(db, symbol: str | None, signal_id: str | None) -> int:
    """Count scaffold-aware entries already recorded for this signal in the current cycle.

    Uses the current market session date (America/New_York) and checks for entries
    that have geometry_candidate_id in their decision_snapshot_json.

    Returns the count, or 0 on error.
    """
    try:
        session_date = datetime.now(_ET).strftime("%Y-%m-%d")
        # Query entries for this symbol today that contain geometry_candidate_id
        result = db.execute(
            text(
                """
                SELECT COUNT(*) FROM blocked_trade_candidates
                WHERE symbol = :symbol
                  AND created_at >= :session_start
                  AND decision_snapshot_json LIKE '%geometry_candidate_id%'
                """
            ),
            {
                "symbol": symbol,
                "session_start": f"{session_date} 00:00:00",
            },
        )
        return result.scalar() or 0
    except Exception as exc:
        log.debug("_count_scaffold_entries_this_cycle: error counting entries: %s", exc)
        return 0


# Maximum scaffold-aware rejection entries per signal per cycle
_MAX_SCAFFOLD_ENTRIES_PER_SIGNAL_PER_CYCLE = 10


def record_blocked_candidate(
    db,
    symbol: str | None,
    action: str,
    blocked_by: str,
    block_reason: str,
    *,
    direction: str | None = None,
    profile: str | None = None,
    setup_type: str | None = None,
    entry_price: float | None = None,
    stop_price: float | None = None,
    target_price: float | None = None,
    quantity: float | None = None,
    reason_code: str | None = None,
    decision_snapshot: dict | str | None = None,
    signal_snapshot: dict | str | None = None,
    gate_notes: list | str | None = None,
    source: str | None = None,
    agent: str | None = None,
    trade_event_id: int | None = None,
    geometry_candidate_id: str | None = None,
    geometry_candidate_name: str | None = None,
    scaffold_snapshot: dict | None = None,
) -> int | None:
    """Record a blocked trade candidate in the shadow ledger.

    Args:
        direction: Explicit trade direction ("long" or "short"). If not provided,
            derived from action: BUY → "long", SHORT → "short". Callers may pass
            it explicitly when the signal carries a direction that differs from
            the simple action mapping.
        geometry_candidate_id: Optional scaffold candidate_id for scaffold-aware
            rejection recording.
        geometry_candidate_name: Optional scaffold candidate name (informational).
        scaffold_snapshot: Optional full scaffold output dict. When provided, it is
            stored in the signal_snapshot_json column alongside the signal data.

    Returns:
        int: The new row's primary key on successful insert.
        None: On deduplication skip, validation failure, or error.

    This function is guaranteed not to raise — all exceptions are caught and logged.
    """
    try:
        # --- Validate required fields ---
        if not action or not action.strip():
            log.warning("record_blocked_candidate: missing required field 'action'")
            return None
        if not blocked_by or not blocked_by.strip():
            log.warning("record_blocked_candidate: missing required field 'blocked_by'")
            return None
        if not block_reason or not block_reason.strip():
            log.warning("record_blocked_candidate: missing required field 'block_reason'")
            return None

        # --- Validate symbol (nullable only for pm_normalizer) ---
        if symbol is None and blocked_by != "pm_normalizer":
            log.warning(
                "record_blocked_candidate: symbol is None and blocked_by is '%s' (not pm_normalizer)",
                blocked_by,
            )
            return None

        # --- Derive direction from action if not explicitly provided ---
        if direction is None:
            action_upper = action.strip().upper()
            if action_upper == "BUY":
                direction = "long"
            elif action_upper == "SHORT":
                direction = "short"

        # --- Scaffold-aware rejection recording ---
        # When geometry_candidate_id is provided, this is a scaffold-aware rejection.
        # Cap at 10 entries per signal per cycle to prevent noise from bulk rejections.
        if geometry_candidate_id is not None:
            current_count = _count_scaffold_entries_this_cycle(db, symbol, None)
            if current_count >= _MAX_SCAFFOLD_ENTRIES_PER_SIGNAL_PER_CYCLE:
                log.debug(
                    "record_blocked_candidate: scaffold entry cap reached (%d) for symbol=%s, skipping",
                    current_count,
                    symbol,
                )
                return None

        # Enrich decision_snapshot with scaffold candidate geometry fields
        if geometry_candidate_id is not None or geometry_candidate_name is not None:
            # Parse existing decision_snapshot if it's a string
            if isinstance(decision_snapshot, str):
                try:
                    decision_snapshot = json.loads(decision_snapshot)
                except (json.JSONDecodeError, TypeError):
                    decision_snapshot = {}
            elif decision_snapshot is None:
                decision_snapshot = {}
            elif not isinstance(decision_snapshot, dict):
                decision_snapshot = {}

            # Add scaffold candidate fields to decision snapshot
            if geometry_candidate_id is not None:
                decision_snapshot["geometry_candidate_id"] = geometry_candidate_id
            if geometry_candidate_name is not None:
                decision_snapshot["geometry_candidate_name"] = geometry_candidate_name

            # Record candidate geometry fields (defensive — include whatever is available)
            # Try to get risk_reward from scaffold_snapshot candidate data if available
            candidate_risk_reward = None
            if scaffold_snapshot and isinstance(scaffold_snapshot, dict):
                candidates = scaffold_snapshot.get("candidates", [])
                if isinstance(candidates, list):
                    for candidate in candidates:
                        if isinstance(candidate, dict) and candidate.get("candidate_id") == geometry_candidate_id:
                            candidate_risk_reward = candidate.get("risk_reward")
                            break

            decision_snapshot["candidate_geometry"] = {
                "entry_price": entry_price,
                "stop_loss": stop_price,
                "target": target_price,
                "risk_reward": candidate_risk_reward,
            }

            # Record rejection reason and gate/profile identifier
            decision_snapshot["rejection_reason"] = block_reason
            decision_snapshot["rejected_by"] = blocked_by
            if profile is not None:
                decision_snapshot["rejection_profile"] = profile

        # Enrich signal_snapshot with scaffold_snapshot when provided
        if scaffold_snapshot is not None:
            # Parse existing signal_snapshot if it's a string
            if isinstance(signal_snapshot, str):
                try:
                    signal_snapshot = json.loads(signal_snapshot)
                except (json.JSONDecodeError, TypeError):
                    signal_snapshot = {}
            elif signal_snapshot is None:
                signal_snapshot = {}
            elif not isinstance(signal_snapshot, dict):
                signal_snapshot = {}

            # Store the full scaffold snapshot alongside the signal data
            signal_snapshot["scaffold_snapshot"] = scaffold_snapshot

        # --- Serialize dict/list arguments to JSON ---
        def _serialize(val: dict | list | str | None) -> str | None:
            if val is None:
                return None
            if isinstance(val, str):
                return val
            return json.dumps(val, default=str)

        decision_snapshot_json = _serialize(decision_snapshot)
        signal_snapshot_json = _serialize(signal_snapshot)
        gate_notes_json = _serialize(gate_notes)

        # --- Compute dedupe_key ---
        dedupe_key = None
        try:
            dedupe_key = compute_dedupe_key(
                profile=profile,
                symbol=symbol,
                action=action,
                setup_type=setup_type,
                blocked_by=blocked_by,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                block_reason=block_reason,
            )
        except Exception as exc:
            log.error(
                "record_blocked_candidate: failed to compute dedupe_key for symbol=%s, blocked_by=%s: %s",
                symbol,
                blocked_by,
                exc,
            )

        # --- Pre-check dedupe_key ---
        if dedupe_key is not None:
            result = db.execute(
                text("SELECT EXISTS(SELECT 1 FROM blocked_trade_candidates WHERE dedupe_key = :dk)"),
                {"dk": dedupe_key},
            )
            if result.scalar():
                log.debug(
                    "record_blocked_candidate: dedupe skip for symbol=%s, blocked_by=%s",
                    symbol,
                    blocked_by,
                )
                return None

        # --- Execute INSERT ---
        result = db.execute(
            text(
                """
                INSERT INTO blocked_trade_candidates (
                    symbol, action, direction, profile, setup_type,
                    entry_price, stop_price, target_price, quantity,
                    blocked_by, block_reason, reason_code,
                    gate_notes_json, decision_snapshot_json, signal_snapshot_json,
                    source, agent, dedupe_key, trade_event_id
                ) VALUES (
                    :symbol, :action, :direction, :profile, :setup_type,
                    :entry_price, :stop_price, :target_price, :quantity,
                    :blocked_by, :block_reason, :reason_code,
                    :gate_notes_json, :decision_snapshot_json, :signal_snapshot_json,
                    :source, :agent, :dedupe_key, :trade_event_id
                )
                """
            ),
            {
                "symbol": symbol,
                "action": action,
                "direction": direction,
                "profile": profile,
                "setup_type": setup_type,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "quantity": quantity,
                "blocked_by": blocked_by,
                "block_reason": block_reason,
                "reason_code": reason_code,
                "gate_notes_json": gate_notes_json,
                "decision_snapshot_json": decision_snapshot_json,
                "signal_snapshot_json": signal_snapshot_json,
                "source": source,
                "agent": agent,
                "dedupe_key": dedupe_key,
                "trade_event_id": trade_event_id,
            },
        )
        db.flush()
        return result.lastrowid

    except IntegrityError as exc:
        exc_msg = str(exc).lower()
        if "dedupe_key" in exc_msg or "uq_blocked_dedupe" in exc_msg:
            log.debug(
                "record_blocked_candidate: dedupe constraint hit (race) for symbol=%s, blocked_by=%s",
                symbol,
                blocked_by,
            )
            return None
        else:
            log.error(
                "record_blocked_candidate: IntegrityError for symbol=%s, blocked_by=%s: %s",
                symbol,
                blocked_by,
                exc,
            )
            return None
    except Exception as exc:
        log.error(
            "record_blocked_candidate: unexpected error for symbol=%s, blocked_by=%s: %s",
            symbol,
            blocked_by,
            exc,
        )
        return None


def write_pilot_counterfactual_row(
    db,
    *,
    symbol: str | None,
    action: str,
    blocked_by: str,
    block_reason: str,
    direction: str | None = None,
    profile: str | None = None,
    setup_type: str | None = None,
    entry_price: float | None = None,
    stop_price: float | None = None,
    target_price: float | None = None,
    quantity: float | None = None,
    gate_result: dict | None = None,
    gate_notes: list | None = None,
    signal_snapshot: dict | str | None = None,
    agent: str | None = None,
) -> int | None:
    """Write a counterfactual shadow ledger row for a pilot override.

    When a gate converts a rejection to `reduce_size` under the pilot,
    this function records what *would have happened* if the trade had been
    fully blocked. This enables post-experiment analysis comparing
    counterfactual outcomes to actual reduced-size trade results.

    The row is written to `blocked_trade_candidates` with
    `pilot_override_applied: true` and `pilot_trade_link: null` in the
    `decision_snapshot_json`. The `pilot_trade_link` is filled in later
    (Task 9.2) when the trade executes.

    Args:
        db: SQLAlchemy session (caller owns commit).
        symbol: Trade symbol.
        action: Trade action (BUY, SHORT, etc.).
        blocked_by: Gate name that would have blocked the trade.
        block_reason: Original rejection reason.
        direction: Trade direction (long/short).
        profile: PM profile (moderate, aggressive, conservative).
        setup_type: Setup classification string.
        entry_price: Entry price of the trade.
        stop_price: Stop price of the trade.
        target_price: Target price of the trade.
        quantity: Original (pre-reduction) quantity.
        gate_result: Full gate evaluation result dict for context.
        gate_notes: Gate pipeline notes list.
        signal_snapshot: Signal data dict for context.
        agent: Agent name.

    Returns:
        int: The new row's primary key on successful insert.
        None: On error or deduplication skip.

    This function is guaranteed not to raise — all exceptions are caught and logged.
    """
    try:
        # Build decision_snapshot with pilot override metadata
        decision_snapshot = {
            "pilot_override_applied": True,
            "pilot_trade_link": None,
        }
        if gate_result and isinstance(gate_result, dict):
            # Include relevant gate result fields for post-experiment analysis
            decision_snapshot["gate_result_snapshot"] = {
                k: v for k, v in gate_result.items()
                if k in (
                    "decision", "reason_type", "reason", "win_rate", "threshold",
                    "size_multiplier", "confirming_signals", "near_miss_margin",
                    "original_rr", "adjusted_rr", "min_reward_to_risk",
                )
            }

        # Delegate to the existing record_blocked_candidate function
        return record_blocked_candidate(
            db,
            symbol=symbol,
            action=action,
            blocked_by=blocked_by,
            block_reason=block_reason,
            direction=direction,
            profile=profile,
            setup_type=setup_type,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=quantity,
            reason_code="pilot_counterfactual",
            decision_snapshot=decision_snapshot,
            signal_snapshot=signal_snapshot,
            gate_notes=gate_notes,
            source="pilot_counterfactual",
            agent=agent,
        )

    except Exception as exc:
        log.error(
            "write_pilot_counterfactual_row: unexpected error for symbol=%s, blocked_by=%s: %s",
            symbol,
            blocked_by,
            exc,
        )
        return None
