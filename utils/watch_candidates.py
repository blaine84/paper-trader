"""Watch candidate lifecycle management.

Creates, evaluates, and transitions watch candidates that track symbols
approaching actionable trade setups. Watch candidates bridge the gap between
market-state analysis and PM candidate creation.

See: design.md §9, §10, requirements.md §8–§10
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from utils.gate_config import MARKET_STATE_MODE, CANDIDATE_EXECUTABLE_SETUP_TYPES
from utils.market_state import WATCHABLE_LIFECYCLE_STATES

logger = logging.getLogger(__name__)

# Default watch candidate TTL
DEFAULT_WATCH_TTL_HOURS: int = 7


@dataclass
class WatchCandidate:
    """Mutable dataclass representing a watch candidate DB row."""
    watch_id: str
    symbol: str
    created_at: str
    updated_at: str
    expires_at: str
    source_cycle_id: str
    profile_id: str
    market_state: str
    setup_lifecycle_state: str
    timeframe_authority_json: str
    direction_watch: str
    trade_posture: str
    activation_conditions_json: str
    invalidation_conditions_json: str
    key_levels_json: str
    trigger_status_json: str
    reason: str
    source_signal_snapshot_json: str
    state: str = "active"
    state_changed_at: str | None = None
    outcome_json: str | None = None

    @property
    def activation_conditions(self) -> list[dict]:
        try:
            return json.loads(self.activation_conditions_json)
        except (json.JSONDecodeError, TypeError):
            return []

    @property
    def invalidation_conditions(self) -> list[dict]:
        try:
            return json.loads(self.invalidation_conditions_json)
        except (json.JSONDecodeError, TypeError):
            return []


def _insert_watch_candidate(engine, watch: WatchCandidate) -> bool:
    """Insert a watch candidate, handling deduplication.

    1. Expire any existing active watch for the same (symbol, profile_id, direction_watch)
    2. INSERT the new watch candidate
    3. Catch IntegrityError from unique index race → log and return False

    Returns True if inserted successfully, False on dedup conflict.
    """
    now = datetime.now(timezone.utc).isoformat()
    try:
        with engine.connect() as conn:
            # Expire existing active watch for same tuple
            conn.execute(
                text(
                    "UPDATE watch_candidates "
                    "SET state = 'expired', state_changed_at = :now, updated_at = :now, "
                    "    outcome_json = :outcome "
                    "WHERE symbol = :symbol AND profile_id = :profile_id "
                    "  AND direction_watch = :direction AND state = 'active'"
                ),
                {
                    "now": now,
                    "outcome": json.dumps({"terminal_state": "expired", "terminal_reason": "replaced_by_newer_watch"}),
                    "symbol": watch.symbol,
                    "profile_id": watch.profile_id,
                    "direction": watch.direction_watch,
                },
            )
            # INSERT new watch
            conn.execute(
                text(
                    "INSERT INTO watch_candidates "
                    "(watch_id, symbol, created_at, updated_at, expires_at, source_cycle_id, "
                    " profile_id, market_state, setup_lifecycle_state, timeframe_authority_json, "
                    " direction_watch, trade_posture, activation_conditions_json, "
                    " invalidation_conditions_json, key_levels_json, trigger_status_json, "
                    " reason, source_signal_snapshot_json, state) "
                    "VALUES (:watch_id, :symbol, :created_at, :updated_at, :expires_at, "
                    " :source_cycle_id, :profile_id, :market_state, :setup_lifecycle_state, "
                    " :timeframe_authority_json, :direction_watch, :trade_posture, "
                    " :activation_conditions_json, :invalidation_conditions_json, "
                    " :key_levels_json, :trigger_status_json, :reason, "
                    " :source_signal_snapshot_json, :state)"
                ),
                {
                    "watch_id": watch.watch_id,
                    "symbol": watch.symbol,
                    "created_at": watch.created_at,
                    "updated_at": watch.updated_at,
                    "expires_at": watch.expires_at,
                    "source_cycle_id": watch.source_cycle_id,
                    "profile_id": watch.profile_id,
                    "market_state": watch.market_state,
                    "setup_lifecycle_state": watch.setup_lifecycle_state,
                    "timeframe_authority_json": watch.timeframe_authority_json,
                    "direction_watch": watch.direction_watch,
                    "trade_posture": watch.trade_posture,
                    "activation_conditions_json": watch.activation_conditions_json,
                    "invalidation_conditions_json": watch.invalidation_conditions_json,
                    "key_levels_json": watch.key_levels_json,
                    "trigger_status_json": watch.trigger_status_json,
                    "reason": watch.reason,
                    "source_signal_snapshot_json": watch.source_signal_snapshot_json,
                    "state": watch.state,
                },
            )
            conn.commit()
        return True
    except Exception as exc:
        # IntegrityError from unique index race condition — safe to swallow
        if "UNIQUE constraint" in str(exc) or "IntegrityError" in type(exc).__name__:
            logger.info("Watch candidate dedup conflict for %s/%s — other cycle won", watch.symbol, watch.profile_id)
            return False
        logger.warning("Failed to insert watch candidate for %s: %s", watch.symbol, exc)
        return False


def _transition_watch_state(engine, watch_id: str, new_state: str, outcome_json: str | None = None) -> bool:
    """Transition a watch candidate to a terminal state.

    Uses CAS pattern: only transitions if current state is 'active'.
    Returns True if transition succeeded, False if already transitioned.
    """
    now = datetime.now(timezone.utc).isoformat()
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "UPDATE watch_candidates "
                    "SET state = :new_state, state_changed_at = :now, "
                    "    updated_at = :now, outcome_json = :outcome "
                    "WHERE watch_id = :watch_id AND state = 'active'"
                ),
                {
                    "new_state": new_state,
                    "now": now,
                    "outcome": outcome_json,
                    "watch_id": watch_id,
                },
            )
            conn.commit()
            return result.rowcount == 1
    except Exception as exc:
        logger.warning("Watch state transition failed for %s: %s", watch_id, exc)
        return False


def _threshold_crossed(cond: dict, current_price: float) -> bool:
    """Check if a trigger condition threshold has been crossed."""
    threshold = cond.get("threshold")
    if threshold is None:
        return False
    direction = cond.get("condition", "")
    if "above" in direction or ">" in direction:
        return current_price > threshold
    if "below" in direction or "<" in direction:
        return current_price < threshold
    return False


def _evaluate_watch(watch: WatchCandidate, current_price: float) -> str | None:
    """Return 'invalidated' or 'promotion_eligible' or None (still watching).

    Invalidation is checked FIRST (conservative — per Req 10.5).
    """
    # Check invalidation FIRST
    for cond in watch.invalidation_conditions:
        if _threshold_crossed(cond, current_price):
            return "invalidated"

    # Check activation
    for cond in watch.activation_conditions:
        if _threshold_crossed(cond, current_price):
            return "promotion_eligible"

    return None  # Still active


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_and_create_watch_candidates(
    engine,
    signals: dict[str, dict],
    cycle_id: str,
    profile_id: str,
) -> int:
    """Create watch candidates for eligible symbols.

    Criteria (all must be true):
    1. MARKET_STATE_MODE != "disabled"
    2. setup_lifecycle_state is in WATCHABLE_LIFECYCLE_STATES
    3. At least one if_then_trigger has a non-null threshold

    Returns count of watch candidates created.
    """
    if MARKET_STATE_MODE == "disabled":
        return 0

    created = 0
    now = datetime.now(timezone.utc)
    expires_at = (now + timedelta(hours=DEFAULT_WATCH_TTL_HOURS)).isoformat()

    for symbol, signal in signals.items():
        try:
            lifecycle_state = signal.get("setup_lifecycle_state", "no_setup")
            if lifecycle_state not in WATCHABLE_LIFECYCLE_STATES:
                continue

            triggers = signal.get("if_then_triggers", [])
            if not any(t.get("threshold") is not None for t in triggers):
                continue

            # Derive direction from signal direction + timeframe authority
            direction = signal.get("signal", "").upper()
            if direction not in ("LONG", "SHORT"):
                direction = "LONG"  # default watch direction

            # Build activation/invalidation conditions from triggers
            activation_conditions = []
            invalidation_conditions = []
            for t in triggers:
                if isinstance(t, dict):
                    posture = t.get("trade_posture", "")
                    if posture in ("watch_long_trigger", "watch_short_trigger", "watch_retest", "eligible_for_pm_review"):
                        activation_conditions.append(t)
                    elif posture in ("veto_long", "veto_short"):
                        invalidation_conditions.append(t)

            watch = WatchCandidate(
                watch_id=str(uuid.uuid4()),
                symbol=symbol,
                created_at=now.isoformat(),
                updated_at=now.isoformat(),
                expires_at=expires_at,
                source_cycle_id=cycle_id,
                profile_id=profile_id,
                market_state=signal.get("market_state", "confounded"),
                setup_lifecycle_state=lifecycle_state,
                timeframe_authority_json=json.dumps(signal.get("timeframe_authority", {})),
                direction_watch=direction,
                trade_posture=signal.get("setup_reclassification", {}).get("trade_posture", "flat") if signal.get("setup_reclassification") else "flat",
                activation_conditions_json=json.dumps(activation_conditions),
                invalidation_conditions_json=json.dumps(invalidation_conditions),
                key_levels_json=json.dumps(signal.get("key_levels", {})),
                trigger_status_json=json.dumps(signal.get("trigger_status", {})),
                reason=f"Lifecycle {lifecycle_state} with {len(triggers)} triggers",
                source_signal_snapshot_json=json.dumps({
                    "signal": signal.get("signal"),
                    "setup_type": signal.get("setup_type"),
                    "strength": signal.get("strength"),
                }),
                state="active",
            )

            if _insert_watch_candidate(engine, watch):
                created += 1
                try:
                    from utils.trade_events import log_trade_event
                    log_trade_event(
                        None,
                        "watch_candidate_created",
                        symbol=symbol,
                        payload={
                            "watch_id": watch.watch_id,
                            "lifecycle_state": lifecycle_state,
                            "direction": direction,
                        },
                    )
                except Exception:
                    pass  # fail-open: logging never blocks

        except Exception as exc:
            logger.warning("Watch candidate creation failed for %s: %s", symbol, exc)
            continue

    return created


def evaluate_active_watch_candidates(
    engine,
    signals: dict[str, dict],
    profile_id: str,
) -> dict[str, int]:
    """Evaluate all active watch candidates for a profile.

    Returns dict of counts: {"invalidated": N, "promotion_eligible": N, "still_active": N}
    """
    counts = {"invalidated": 0, "promotion_eligible": 0, "still_active": 0}

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT watch_id, symbol, direction_watch, "
                    "       activation_conditions_json, invalidation_conditions_json, "
                    "       source_signal_snapshot_json "
                    "FROM watch_candidates "
                    "WHERE profile_id = :profile_id AND state = 'active'"
                ),
                {"profile_id": profile_id},
            ).fetchall()

        for row in rows:
            watch_id = row[0]
            symbol = row[1]
            signal = signals.get(symbol, {})
            current_price = signal.get("current_price") or signal.get("key_levels", {}).get("vwap")

            if current_price is None:
                counts["still_active"] += 1
                continue

            # Build minimal WatchCandidate for evaluation
            watch = WatchCandidate(
                watch_id=watch_id,
                symbol=symbol,
                created_at="",
                updated_at="",
                expires_at="",
                source_cycle_id="",
                profile_id=profile_id,
                market_state="",
                setup_lifecycle_state="",
                timeframe_authority_json="",
                direction_watch=row[2],
                trade_posture="",
                activation_conditions_json=row[3],
                invalidation_conditions_json=row[4],
                key_levels_json="",
                trigger_status_json="",
                reason="",
                source_signal_snapshot_json=row[5] or "{}",
            )

            result = _evaluate_watch(watch, current_price)

            if result == "invalidated":
                outcome = json.dumps({
                    "terminal_state": "invalidated",
                    "terminal_reason": "invalidation_threshold_crossed",
                    "price_at_invalidation": current_price,
                })
                _transition_watch_state(engine, watch_id, "invalidated", outcome)
                counts["invalidated"] += 1
                try:
                    from utils.trade_events import log_trade_event
                    log_trade_event(None, "watch_candidate_invalidated", symbol=symbol, payload={"watch_id": watch_id})
                except Exception:
                    pass

            elif result == "promotion_eligible":
                if MARKET_STATE_MODE == "enforcing":
                    # Attempt promotion
                    signal_direction = signal.get("signal", "").upper()
                    setup_type = signal.get("setup_type", "")

                    if signal_direction in ("LONG", "SHORT") and setup_type in CANDIDATE_EXECUTABLE_SETUP_TYPES:
                        outcome = json.dumps({
                            "terminal_state": "promoted",
                            "terminal_reason": "activation_threshold_crossed_and_promoted",
                            "price_at_activation": current_price,
                        })
                        _transition_watch_state(engine, watch_id, "promoted", outcome)
                        counts["promotion_eligible"] += 1
                    else:
                        # Cannot promote — signal not directional or setup not executable
                        outcome = json.dumps({
                            "terminal_state": "expired",
                            "terminal_reason": "activation_detected_but_promotion_blocked",
                            "block_reason": f"signal={signal_direction}, setup={setup_type}",
                            "price_at_activation": current_price,
                        })
                        _transition_watch_state(engine, watch_id, "expired", outcome)
                        counts["promotion_eligible"] += 1
                else:
                    # Observe mode — record activation but expire
                    outcome = json.dumps({
                        "terminal_state": "expired",
                        "terminal_reason": "activation_observed_in_observe_mode",
                        "price_at_activation": current_price,
                    })
                    _transition_watch_state(engine, watch_id, "expired", outcome)
                    counts["promotion_eligible"] += 1

                try:
                    from utils.trade_events import log_trade_event
                    log_trade_event(None, "watch_candidate_activation_observed", symbol=symbol, payload={"watch_id": watch_id})
                except Exception:
                    pass
            else:
                counts["still_active"] += 1

    except Exception as exc:
        logger.warning("Watch candidate evaluation failed for profile %s: %s", profile_id, exc)

    return counts


def expire_session_watch_candidates(engine) -> int:
    """Expire all active watch candidates past their expires_at time.

    Idempotent — safe to call multiple times. Used for:
    - Session-end cleanup
    - Startup recovery sweep

    Returns count of expired candidates.
    """
    now = datetime.now(timezone.utc).isoformat()
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "UPDATE watch_candidates "
                    "SET state = 'expired', state_changed_at = :now, updated_at = :now, "
                    "    outcome_json = :outcome "
                    "WHERE state = 'active' AND expires_at < :now"
                ),
                {
                    "now": now,
                    "outcome": json.dumps({"terminal_state": "expired", "terminal_reason": "session_expired"}),
                },
            )
            conn.commit()
            expired_count = result.rowcount
            if expired_count > 0:
                logger.info("Expired %d session watch candidates", expired_count)
            return expired_count
    except Exception as exc:
        logger.warning("Session watch candidate expiration failed: %s", exc)
        return 0


def get_promotable_candidates(
    engine,
    signals: dict[str, dict],
    profile_id: str,
) -> list[dict]:
    """Get watch candidates eligible for promotion to PM candidates.

    Returns candidates that:
    1. Have been promoted in the current cycle (state='promoted')
    2. Signal is directional (LONG/SHORT)
    3. Setup type is in CANDIDATE_EXECUTABLE_SETUP_TYPES

    Note: This function queries promoted candidates that were already
    processed by evaluate_active_watch_candidates(). It's used by the
    candidate builder for the actual PM candidate creation.
    """
    promotable = []
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT watch_id, symbol, direction_watch, "
                    "       source_signal_snapshot_json, outcome_json "
                    "FROM watch_candidates "
                    "WHERE profile_id = :profile_id AND state = 'promoted'"
                ),
                {"profile_id": profile_id},
            ).fetchall()

        for row in rows:
            symbol = row[1]
            signal = signals.get(symbol, {})
            signal_direction = signal.get("signal", "").upper()
            setup_type = signal.get("setup_type", "")

            if signal_direction in ("LONG", "SHORT") and setup_type in CANDIDATE_EXECUTABLE_SETUP_TYPES:
                promotable.append({
                    "watch_id": row[0],
                    "symbol": symbol,
                    "direction": row[2],
                    "signal": signal,
                    "source_signal_snapshot": json.loads(row[3] or "{}"),
                })

    except Exception as exc:
        logger.warning("get_promotable_candidates failed for profile %s: %s", profile_id, exc)

    return promotable
