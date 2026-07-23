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

from utils.gate_config import (
    MARKET_STATE_MODE,
    CANDIDATE_EXECUTABLE_SETUP_TYPES,
    WATCH_KEY_LEVEL_DRIFT_PCT,
    WATCH_SAME_CYCLE_PROMOTION_POLICY,
)
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


def _transition_watch_state(
    engine,
    watch_id: str,
    new_state: str,
    outcome_json: str | None = None,
    *,
    expected_state: str = "active",
    promoted_cycle_id: str | None = None,
) -> bool:
    """Transition a watch candidate state via CAS.

    Uses CAS pattern: only transitions if current state matches expected_state.
    Default expected_state is 'active' for backward compatibility. For a
    promoted -> registered (or promoted -> expired) transition, the caller
    passes expected_state='promoted'.

    When promoted_cycle_id is provided (typically for active -> promoted), it is
    written atomically with the state change. COALESCE preserves any existing
    value when the parameter is None, so non-promotion transitions never clear
    it and a promoted row is never observable with a NULL/stale promoted_cycle_id
    relative to the state transition.

    Returns True if transition succeeded, False if already transitioned.
    """
    now = datetime.now(timezone.utc).isoformat()
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "UPDATE watch_candidates "
                    "SET state = :new_state, state_changed_at = :now, "
                    "    updated_at = :now, outcome_json = :outcome, "
                    "    promoted_cycle_id = COALESCE(:promoted_cycle_id, promoted_cycle_id) "
                    "WHERE watch_id = :watch_id AND state = :expected_state"
                ),
                {
                    "new_state": new_state,
                    "now": now,
                    "outcome": outcome_json,
                    "promoted_cycle_id": promoted_cycle_id,
                    "watch_id": watch_id,
                    "expected_state": expected_state,
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


def _is_positive_numeric(value) -> bool:
    """Return True if value is a real number (not bool) strictly greater than zero."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    try:
        return value > 0
    except TypeError:
        return False


def _check_structural_invalidation(
    watch: WatchCandidate,
    current_signal: dict,
) -> str | None:
    """Return a terminal_reason if the watch is structurally invalid, else None.

    Checks (in order):
    1. Lifecycle state regression: the current signal's ``setup_lifecycle_state``
       is present but NOT in WATCHABLE_LIFECYCLE_STATES -> ``"structural_degradation"``.
    2. Key-level drift: any stored support/resistance level whose current value has
       drifted more than ``WATCH_KEY_LEVEL_DRIFT_PCT`` percent, computed as
       ``abs(current - stored) / stored * 100`` -> ``"key_level_drift"``.

    Fail-open: returns None when the current signal is missing/empty, has no
    lifecycle data, or has no comparable key levels. A level whose stored or
    current value is non-numeric or <= 0 is skipped from the drift comparison and
    logged at DEBUG with the reason.

    See design.md §5 (Structural Invalidation Algorithm).
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.6, 3.7, 3.8.
    """
    # Fail-open: no signal to compare against.
    if not current_signal:
        return None

    # --- Check 1: lifecycle state regression (only when lifecycle data present) ---
    lifecycle_state = current_signal.get("setup_lifecycle_state")
    if lifecycle_state:
        if lifecycle_state not in WATCHABLE_LIFECYCLE_STATES:
            logger.debug(
                "Structural degradation for watch %s: lifecycle_state=%r not watchable",
                watch.watch_id, lifecycle_state,
            )
            return "structural_degradation"

    # --- Check 2: key-level drift for support / resistance ---
    try:
        stored_levels = json.loads(watch.key_levels_json) if watch.key_levels_json else {}
    except (json.JSONDecodeError, TypeError):
        stored_levels = {}
    if not isinstance(stored_levels, dict):
        stored_levels = {}

    current_levels = current_signal.get("key_levels", {})
    if not isinstance(current_levels, dict):
        current_levels = {}

    for level_name in ("support", "resistance"):
        stored_value = stored_levels.get(level_name)
        current_value = current_levels.get(level_name)

        if not _is_positive_numeric(stored_value):
            logger.debug(
                "Skipping drift check for %s on watch %s: stored value non-numeric or <= 0 (%r)",
                level_name, watch.watch_id, stored_value,
            )
            continue
        if not _is_positive_numeric(current_value):
            logger.debug(
                "Skipping drift check for %s on watch %s: current value non-numeric or <= 0 (%r)",
                level_name, watch.watch_id, current_value,
            )
            continue

        drift_pct = abs(current_value - stored_value) / stored_value * 100
        if drift_pct > WATCH_KEY_LEVEL_DRIFT_PCT:
            logger.debug(
                "Key-level drift for %s on watch %s: %.2f%% > %.2f%% threshold",
                level_name, watch.watch_id, drift_pct, WATCH_KEY_LEVEL_DRIFT_PCT,
            )
            return "key_level_drift"

    # Fail-open: no structural issue detected (or no comparable levels).
    return None


def _check_same_cycle_policy(
    watch: WatchCandidate,
    current_signal: dict,
    cycle_id: str | None,
) -> bool:
    """Return True if promotion is BLOCKED by the same-cycle policy, else False.

    Invoked only AFTER price activation has been detected for a watch. It governs
    the residual edge case where an already-activated watch's ``source_cycle_id``
    equals the current ``cycle_id``.

    Returns False (allow) when:
    - ``cycle_id`` is None (backward compat — skip policy)
    - ``watch.source_cycle_id`` != ``cycle_id`` (not same-cycle)
    - policy is ``"always"``
    - policy is ``"activation_pending_only"`` AND current signal's
      ``setup_lifecycle_state`` == ``"activation_pending"``

    Returns True (block) when:
    - policy is ``"never"`` AND same-cycle
    - policy is ``"activation_pending_only"`` AND same-cycle AND the current
      signal's lifecycle is not ``"activation_pending"`` (including missing data)

    Blocked promotions are logged at DEBUG with the policy name and watch_id.

    See design.md §6 (Same-Cycle Promotion Policy).
    Requirements: 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.12, 4.13.
    """
    # Backward compat: no cycle context → skip policy entirely.
    if cycle_id is None:
        return False

    # Not a same-cycle promotion → policy does not apply.
    if watch.source_cycle_id != cycle_id:
        return False

    policy = WATCH_SAME_CYCLE_PROMOTION_POLICY

    if policy == "always":
        return False

    if policy == "never":
        logger.debug(
            "Same-cycle promotion blocked by policy=%r for watch %s",
            policy, watch.watch_id,
        )
        return True

    # policy == "activation_pending_only": allow only when lifecycle matches.
    lifecycle_state = (current_signal or {}).get("setup_lifecycle_state")
    if lifecycle_state == "activation_pending":
        return False

    logger.debug(
        "Same-cycle promotion blocked by policy=%r (lifecycle=%r) for watch %s",
        policy, lifecycle_state, watch.watch_id,
    )
    return True


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
    *,
    cycle_id: str | None = None,
) -> dict[str, int]:
    """Evaluate all active watch candidates for a profile.

    Per-watch evaluation order (corrected — same-cycle policy runs AFTER price
    activation detection, gating only the final ``active -> promoted`` transition):

    1. Structural invalidation (lifecycle regression + key-level drift) — runs
       before any price logic; on invalidation the watch transitions to
       ``invalidated`` immediately.
    2. Price-threshold invalidation (existing behavior).
    3. Price-threshold activation detection (existing behavior).
    4. Same-cycle promotion policy — only for a watch whose activation threshold
       has been crossed; when blocked the watch is left in ``active`` state.
    5. Promote / block transition.

    ``cycle_id`` (keyword-only, defaults to ``None`` for backward compatibility):
    - On promotion, ``promoted_cycle_id = cycle_id`` is recorded atomically via
      ``_transition_watch_state()``.
    - Enforcing mode + ``cycle_id is None``: all invalidation checks (structural
      + price-threshold) still run, but NO watch is transitioned to ``promoted``
      (promotion blocked entirely) and a WARNING is logged. This prevents leaving
      a promoted row with a NULL ``promoted_cycle_id`` that
      ``get_promotable_candidates()`` could never consume.
    - Non-enforcing mode + ``cycle_id is None``: same-cycle policy is skipped and
      promotion proceeds as normal (backward-compatible for tests / observe mode).

    Structural and same-cycle checks are wrapped in try/except and fail-open
    (log WARNING, continue). All hardening logic is skipped while
    ``MARKET_STATE_MODE`` is ``"disabled"``.

    Returns dict of counts: {"invalidated": N, "promotion_eligible": N, "still_active": N}

    See design.md §4 (Modified ``evaluate_active_watch_candidates()``).
    Requirements: 1.6, 1.12, 3.1, 3.2, 3.3, 3.4, 3.5, 4.3, 4.9, 4.10, 4.11, 4.13, 5.1, 5.4.
    """
    counts = {"invalidated": 0, "promotion_eligible": 0, "still_active": 0}
    hardening_enabled = MARKET_STATE_MODE != "disabled"

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT watch_id, symbol, direction_watch, "
                    "       activation_conditions_json, invalidation_conditions_json, "
                    "       source_signal_snapshot_json, source_cycle_id, key_levels_json "
                    "FROM watch_candidates "
                    "WHERE profile_id = :profile_id AND state = 'active'"
                ),
                {"profile_id": profile_id},
            ).fetchall()

        for row in rows:
            watch_id = row[0]
            symbol = row[1]
            signal = signals.get(symbol, {})

            # Build WatchCandidate for evaluation. source_cycle_id and
            # key_levels_json are populated so the hardening helpers work.
            watch = WatchCandidate(
                watch_id=watch_id,
                symbol=symbol,
                created_at="",
                updated_at="",
                expires_at="",
                source_cycle_id=row[6] or "",
                profile_id=profile_id,
                market_state="",
                setup_lifecycle_state="",
                timeframe_authority_json="",
                direction_watch=row[2],
                trade_posture="",
                activation_conditions_json=row[3],
                invalidation_conditions_json=row[4],
                key_levels_json=row[7] or "",
                trigger_status_json="",
                reason="",
                source_signal_snapshot_json=row[5] or "{}",
            )

            # --- Step 1: Structural invalidation (before any price logic) ---
            structural_reason = None
            if hardening_enabled:
                try:
                    structural_reason = _check_structural_invalidation(watch, signal)
                except Exception as exc:
                    # Fail-open: log and continue to price-threshold evaluation.
                    logger.warning(
                        "Structural invalidation check failed for watch %s: %s (fail-open)",
                        watch_id, exc,
                    )
                    structural_reason = None

            if structural_reason:
                outcome = json.dumps({
                    "terminal_state": "invalidated",
                    "terminal_reason": structural_reason,
                })
                _transition_watch_state(engine, watch_id, "invalidated", outcome)
                counts["invalidated"] += 1
                try:
                    from utils.trade_events import log_trade_event
                    log_trade_event(
                        None, "watch_candidate_invalidated", symbol=symbol,
                        payload={"watch_id": watch_id, "reason": structural_reason},
                    )
                except Exception:
                    pass
                continue

            # --- Price-threshold evaluation (Steps 2 & 3) ---
            current_price = signal.get("current_price") or signal.get("key_levels", {}).get("vwap")
            if current_price is None:
                counts["still_active"] += 1
                continue

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
                # --- Step 4: Same-cycle promotion policy (only after activation) ---
                blocked_same_cycle = False
                if hardening_enabled:
                    try:
                        blocked_same_cycle = _check_same_cycle_policy(watch, signal, cycle_id)
                    except Exception as exc:
                        # Fail-open: allow promotion to proceed.
                        logger.warning(
                            "Same-cycle policy check failed for watch %s: %s (fail-open, allowing)",
                            watch_id, exc,
                        )
                        blocked_same_cycle = False

                if blocked_same_cycle:
                    # Leave in active state so it can be evaluated next cycle.
                    counts["still_active"] += 1
                    continue

                # --- Step 5: Promote / block transition ---
                if MARKET_STATE_MODE == "enforcing":
                    if cycle_id is None:
                        # Promotion blocked entirely: cannot leave a promoted row
                        # with a NULL promoted_cycle_id that could never be consumed.
                        logger.warning(
                            "evaluate_active_watch_candidates: cycle_id is required "
                            "for promotion in enforcing mode — watch %s left active "
                            "(promotion blocked)",
                            watch_id,
                        )
                        counts["still_active"] += 1
                        continue

                    # Attempt promotion
                    signal_direction = signal.get("signal", "").upper()
                    setup_type = signal.get("setup_type", "")

                    if signal_direction in ("LONG", "SHORT") and setup_type in CANDIDATE_EXECUTABLE_SETUP_TYPES:
                        outcome = json.dumps({
                            "terminal_state": "promoted",
                            "terminal_reason": "activation_threshold_crossed_and_promoted",
                            "price_at_activation": current_price,
                        })
                        _transition_watch_state(
                            engine, watch_id, "promoted", outcome,
                            promoted_cycle_id=cycle_id,
                        )
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


def expire_session_watch_candidates(engine, profile_id: str | None = None) -> int:
    """Expire watch candidates past their TTL (``expires_at``).

    Sweeps BOTH stale ``active`` and stale ``promoted`` rows whose ``expires_at``
    is earlier than now, transitioning them to ``expired``. Previously only
    ``active`` rows were swept; including ``promoted`` guarantees a promoted watch
    that was never consumed (crash between promotion and the promotion loop) still
    reaches a terminal state via TTL rather than lingering as transient.

    Existing ``outcome_json`` is preserved via COALESCE — only rows with no prior
    outcome are stamped with ``{"terminal_reason": "ttl_expired"}``.

    ``profile_id`` (backward-compatible, defaults to ``None``):
    - When provided, the sweep is scoped to that profile (cycle-scoped usage).
    - When ``None``, the sweep spans all profiles (session-end / startup recovery).

    Idempotent — safe to call multiple times. Returns count of expired candidates.

    See design.md §11 (Modified ``expire_session_watch_candidates()``).
    Requirements: 1.13.
    """
    now = datetime.now(timezone.utc).isoformat()
    outcome = json.dumps({"terminal_reason": "ttl_expired"})
    try:
        with engine.connect() as conn:
            if profile_id is not None:
                result = conn.execute(
                    text(
                        "UPDATE watch_candidates "
                        "SET state = 'expired', "
                        "    outcome_json = COALESCE(outcome_json, :outcome), "
                        "    state_changed_at = :now, updated_at = :now "
                        "WHERE profile_id = :profile_id "
                        "  AND state IN ('active', 'promoted') "
                        "  AND expires_at < :now"
                    ),
                    {"now": now, "outcome": outcome, "profile_id": profile_id},
                )
            else:
                result = conn.execute(
                    text(
                        "UPDATE watch_candidates "
                        "SET state = 'expired', "
                        "    outcome_json = COALESCE(outcome_json, :outcome), "
                        "    state_changed_at = :now, updated_at = :now "
                        "WHERE state IN ('active', 'promoted') "
                        "  AND expires_at < :now"
                    ),
                    {"now": now, "outcome": outcome},
                )
            conn.commit()
            expired_count = result.rowcount
            if expired_count > 0:
                logger.info("Expired %d session watch candidates", expired_count)
            return expired_count
    except Exception as exc:
        logger.warning("Session watch candidate expiration failed: %s", exc)
        return 0


def expire_stale_promoted_watches(
    engine,
    profile_id: str,
    cycle_id: str,
) -> int:
    """Expire promoted watches whose ``promoted_cycle_id`` != current ``cycle_id``.

    Runs as the FIRST step of the candidate_builder evaluation order, BEFORE
    evaluating any active watch. Any ``promoted`` watch left over from a prior
    cycle that crashed before consumption is decisively expired here: it would
    otherwise be un-consumable (``get_promotable_candidates()`` filters by the
    current ``promoted_cycle_id``) yet still occupy a transient state.

    Each matched row is transitioned to ``expired`` via CAS
    (``expected_state='promoted'``) with outcome_json containing
    ``{"terminal_reason": "promotion_expired_stale_cycle"}``. This is a decisive
    expire — NOT a re-consume. Rows whose CAS fails (already consumed
    concurrently) are logged at WARNING and skipped harmlessly.

    Fail-open: catches OperationalError on the missing ``promoted_cycle_id``
    column (pre-migration rolling deploy), logs a WARNING, and returns 0.

    Returns the count of expired stale promoted rows.

    See design.md §3 (New ``expire_stale_promoted_watches()``).
    Requirements: 1.11.
    """
    from sqlalchemy.exc import OperationalError

    expired = 0
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT watch_id FROM watch_candidates "
                    "WHERE profile_id = :profile_id "
                    "  AND state = 'promoted' "
                    "  AND (promoted_cycle_id IS NULL OR promoted_cycle_id != :cycle_id)"
                ),
                {"profile_id": profile_id, "cycle_id": cycle_id},
            ).fetchall()
    except OperationalError as exc:
        # Pre-migration: promoted_cycle_id column does not exist yet.
        logger.warning(
            "expire_stale_promoted_watches: promoted_cycle_id column unavailable "
            "(pre-migration?) for profile %s: %s",
            profile_id, exc,
        )
        return 0
    except Exception as exc:
        logger.warning(
            "expire_stale_promoted_watches query failed for profile %s: %s",
            profile_id, exc,
        )
        return 0

    outcome = json.dumps({
        "terminal_state": "expired",
        "terminal_reason": "promotion_expired_stale_cycle",
    })

    for row in rows:
        watch_id = row[0]
        # Decisive expire via CAS (expected_state='promoted'). A concurrent
        # consumer may have already transitioned the row — CAS failure is benign.
        if _transition_watch_state(
            engine, watch_id, "expired", outcome, expected_state="promoted"
        ):
            expired += 1
        else:
            logger.warning(
                "expire_stale_promoted_watches: CAS failed for watch %s "
                "(already consumed concurrently?) — skipping",
                watch_id,
            )

    if expired > 0:
        logger.info(
            "Expired %d stale promoted watch candidates for profile %s (cycle %s)",
            expired, profile_id, cycle_id,
        )
    return expired


def get_promotable_candidates(
    engine,
    signals: dict[str, dict],
    profile_id: str,
    *,
    cycle_id: str | None = None,
) -> list[dict]:
    """Get watch candidates eligible for promotion to PM candidates.

    Returns candidates that:
    1. Have been promoted in the current cycle (state='promoted')
    2. Signal is directional (LONG/SHORT)
    3. Setup type is in CANDIDATE_EXECUTABLE_SETUP_TYPES

    Note: This function queries promoted candidates that were already
    processed by evaluate_active_watch_candidates(). It's used by the
    candidate builder for the actual PM candidate creation.

    ``cycle_id`` (keyword-only, defaults to ``None`` for backward compatibility):
    - Enforcing mode + ``cycle_id is None``: returns an empty list and logs a
      WARNING. This prevents un-scoped queries from polluting the candidate
      pipeline in enforcing mode.
    - Non-enforcing mode + ``cycle_id is None``: falls back to existing behavior
      (all promoted rows for the profile) — backward-compatible for tests and
      observe mode.
    - When ``cycle_id`` is provided: the query is scoped to
      ``promoted_cycle_id = :cycle_id`` so only watches promoted in the current
      cycle are returned.

    See design.md §7 (Modified ``get_promotable_candidates()``).
    Requirements: 1.1, 1.8, 1.9.
    """
    # Enforcing mode requires cycle_id — refuse un-scoped queries.
    if MARKET_STATE_MODE == "enforcing" and cycle_id is None:
        logger.warning(
            "get_promotable_candidates called without cycle_id in enforcing mode "
            "— returning empty"
        )
        return []

    promotable = []
    try:
        with engine.connect() as conn:
            if cycle_id is not None:
                rows = conn.execute(
                    text(
                        "SELECT watch_id, symbol, direction_watch, "
                        "       source_signal_snapshot_json, outcome_json "
                        "FROM watch_candidates "
                        "WHERE profile_id = :profile_id AND state = 'promoted' "
                        "  AND promoted_cycle_id = :cycle_id"
                    ),
                    {"profile_id": profile_id, "cycle_id": cycle_id},
                ).fetchall()
            else:
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
