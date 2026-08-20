"""Setup Watch Manager — orchestrates setup watch lifecycle within a cycle.

Single entry point for the candidate builder. Handles the full evaluation
sequence: expire → invalidate → evaluate maturity → promote.
Also provides the noise-filtered creation interface for upstream sources.

Requirements: 2.1-2.10, 4.1, 4.9-4.11, 5.1-5.8, 6.1-6.7, 6.7-6.8,
              7.6, 8.2-8.3, 9.5-9.8, 10.1-10.8, 11.1-11.8, 12.1-12.7
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text

from utils.gate_config import (
    CANDIDATE_EXECUTABLE_SETUP_TYPES,
    MARKET_DATA_RELIABILITY_MODE,
    SETUP_WATCH_MATURITY_THRESHOLD,
    SETUP_WATCH_MAX_ACTIVE_PER_PROFILE,
    SETUP_WATCH_MAX_PER_SYMBOL,
    SETUP_WATCH_MAX_TTL_HOURS,
    SETUP_WATCH_MIN_CONDITION_COUNT,
    SETUP_WATCH_MIN_CREATION_STRENGTH,
    SETUP_WATCH_MISSED_MOVE_ENABLED,
    SETUP_WATCH_MODE,
    SETUP_WATCH_PROMOTION_MIN_CYCLES,
    SETUP_WATCH_REALTIME_MODE,
    SWING_EXECUTABLE_SETUP_TYPES,
)
from utils.pending_order_time import now_utc, to_iso
from utils.setup_watch_evaluator import evaluate_watch, validate_draft_geometry
from utils.setup_watch_registry import (
    ACTIVE_STATES,
    TERMINAL_STATES,
    SetupWatch,
    SetupWatchRegistry,
    SetupWatchRegistryError,
    WatchState,
    compute_watch_integrity_hash,
)

logger = logging.getLogger(__name__)

# Signal strength ordering — mirrors candidate_builder.STRENGTH_ORDER
STRENGTH_ORDER: dict[str, int] = {"weak": 1, "moderate": 2, "strong": 3}
CONFIDENCE_ORDER: dict[str, int] = {"low": 1, "medium": 2, "high": 3}
WATCHABLE_SETUP_TYPES = CANDIDATE_EXECUTABLE_SETUP_TYPES | SWING_EXECUTABLE_SETUP_TYPES

# Known source types for provenance validation (Req 11.6)
_KNOWN_SOURCE_TYPES: frozenset[str] = frozenset({
    "analyst",
    "candidate_reject",
    "market_state",
    "pm_defer",
    "price_monitor",
    "scout",
})

# Legacy source types that tolerate null source_id (Req 11.8)
_LEGACY_SOURCE_TYPES: frozenset[str] = frozenset({
    "analyst",
    "candidate_reject",
    "market_state",
    "pm_defer",
})


# ────────────────────────────────────────────────────────────────────────────
# Data classes
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CreationStats:
    """Summary of watch creation attempts for one cycle.

    No ``rejected_non_executable_type`` counter: the setup-type allowlist is
    enforced at promotion, not creation (Req 10.6), so it can never reject
    here.
    """

    attempted: int
    created: int
    rejected_weak_signal: int
    rejected_insufficient_conditions: int
    rejected_per_symbol_cap: int
    rejected_exposure_conflict: int
    rejected_short_thesis: int
    evicted_for_capacity: int


@dataclass(frozen=True)
class CycleEvaluationResult:
    """Summary of setup watch evaluation for one cycle."""

    expired_ttl: int
    expired_stale_promoted: int
    invalidated: int
    matured: int         # state changed (watching→maturing or maturing→ready)
    regressed: int       # state regressed (ready→maturing or maturing→watching)
    promoted: int        # state changed to promoted (enabled mode only)
    still_active: int
    created: int


# ────────────────────────────────────────────────────────────────────────────
# Internal — source provenance validation (Req 11.6, 11.8)
# ────────────────────────────────────────────────────────────────────────────


def _validate_source_provenance(source_type: str, source_id: str | None) -> bool:
    """Validate source provenance for a new watch.

    Rules:
      - Unknown source_type → reject (return False)
      - New types (price_monitor, scout) require source_id → reject if null
      - Legacy types (analyst, candidate_reject, market_state, pm_defer)
        tolerate null source_id with DEBUG log

    Returns True if valid, False if rejected.
    """
    if source_type not in _KNOWN_SOURCE_TYPES:
        logger.warning(
            "Watch creation rejected: unknown source_type=%r", source_type
        )
        return False

    if source_type not in _LEGACY_SOURCE_TYPES:
        # New type — require source_id
        if not source_id:
            logger.warning(
                "Watch creation rejected: source_type=%r requires source_id",
                source_type,
            )
            return False
    else:
        # Legacy type — tolerate null source_id
        if not source_id:
            logger.debug(
                "Legacy source_type=%r with null source_id (tolerated)",
                source_type,
            )

    return True


# ────────────────────────────────────────────────────────────────────────────
# Internal — shared promotion function (Req 5.1-5.8, 6.1-6.7, 12.1-12.6)
# ────────────────────────────────────────────────────────────────────────────


def _promote_ready_watch(
    registry: SetupWatchRegistry,
    watch: SetupWatch,
    cycle_id: str | None = None,
    signals: dict[str, dict] | None = None,
) -> bool:
    """Single promotion path shared by scheduled evaluator and realtime bridge.

    Steps:
      1. Idempotency check — skip if already promoted this cycle
      2. Missed-move check — if target crossed, CAS → MISSED, skip
      3. Market-data reliability gate — if unreliable, defer, emit event
      4. CAS READY→PROMOTED with promoted_cycle_id
      5. Emit promotion event with enhanced evidence package

    Returns True if promotion succeeded, False otherwise.
    """
    # --- Step 1: Idempotency check (Req 12.1-12.6) ---
    if cycle_id and watch.promoted_cycle_id == cycle_id:
        logger.warning(
            "Promotion skipped (idempotency): watch %s already promoted "
            "in cycle %s",
            watch.watch_id, cycle_id,
        )
        return False

    # --- Step 2: Missed-move pre-promotion check (Req 4.2-4.5) ---
    if SETUP_WATCH_MISSED_MOVE_ENABLED and watch.draft_geometry_json:
        try:
            from utils.missed_move_detector import (
                apply_missed_move_transition,
                check_missed_move,
            )

            # Get current price from signals
            current_price = None
            if signals and watch.symbol in signals:
                current_price = signals[watch.symbol].get("current_price")

            if current_price is not None:
                missed_result = check_missed_move(watch, current_price)
                if missed_result.missed:
                    apply_missed_move_transition(registry, watch, missed_result)
                    logger.warning(
                        "Promotion blocked (missed move): watch %s symbol=%s "
                        "side=%s target=%s current=%s",
                        watch.watch_id, watch.symbol, missed_result.side,
                        missed_result.target_price, missed_result.current_price,
                    )
                    return False
        except Exception as exc:
            # Missed-move is safety — log but continue with promotion
            # (if we can't determine missed state, it's safer to allow promotion
            # and let the pending-order guard catch it)
            logger.warning(
                "Missed-move check failed for watch %s (continuing): %s",
                watch.watch_id, exc,
            )

    # --- Step 3: Market-data reliability gate (Req 5.1-5.8) ---
    if MARKET_DATA_RELIABILITY_MODE != "disabled":
        try:
            from utils.market_data_reliability.pipeline_integration import (
                check_market_data_readiness,
            )

            readiness = check_market_data_readiness(watch.symbol)

            if not readiness.proceed:
                # In "observe" realtime mode: evaluate for logging but do NOT block
                if SETUP_WATCH_REALTIME_MODE == "observe":
                    logger.info(
                        "Market-data reliability would defer watch %s "
                        "(observe mode, not blocking): reason_codes=%s",
                        watch.watch_id, readiness.reason_codes,
                    )
                else:
                    # Defer promotion, emit event
                    _emit_promotion_deferred_event(
                        registry, watch, readiness.reason_codes,
                        readiness.missing_data_types,
                    )
                    logger.info(
                        "Promotion deferred (market data): watch %s symbol=%s "
                        "reason_codes=%s",
                        watch.watch_id, watch.symbol, readiness.reason_codes,
                    )
                    return False
        except Exception as exc:
            # Exception → treat as unreliable, defer (Req 5.6)
            if SETUP_WATCH_REALTIME_MODE == "observe":
                logger.warning(
                    "Market-data reliability check failed for watch %s "
                    "(observe mode, not blocking): %s",
                    watch.watch_id, exc,
                )
            else:
                _emit_promotion_deferred_event(
                    registry, watch,
                    reason_codes=("reliability_check_error",),
                    missing_data_types=(),
                )
                logger.warning(
                    "Promotion deferred (reliability check error): watch %s "
                    "symbol=%s error=%s",
                    watch.watch_id, watch.symbol, exc,
                )
                return False

    # --- Step 4: CAS READY→PROMOTED with promoted_cycle_id ---
    try:
        registry.transition_state(
            watch.watch_id,
            WatchState.READY,
            WatchState.PROMOTED,
            promoted_cycle_id=cycle_id,
        )
    except SetupWatchRegistryError as e:
        logger.warning(
            "CAS promotion failed for watch %s (race condition): %s",
            watch.watch_id, e,
        )
        return False

    # --- Step 5: Emit promotion event with evidence package ---
    try:
        evidence = _build_evidence_package(watch, cycle_id, signals)
        registry._emit_event(
            watch_id=watch.watch_id,
            profile_id=watch.profile_id,
            symbol=watch.symbol,
            event_type="state_transition",
            from_state=WatchState.READY,
            to_state=WatchState.PROMOTED,
            maturity_score=watch.maturity_score,
            event_data=json.dumps({
                "cycle_id": cycle_id,
                "evidence_package": evidence,
                "summary": _build_event_summary(
                    watch.symbol, watch.side, "promoted to PM",
                    watch.maturity_score,
                ),
            }),
        )
    except Exception as exc:
        # Event emission is fail-open (Req 8.7)
        logger.warning(
            "Failed to emit promotion event for watch %s (fail-open): %s",
            watch.watch_id, exc,
        )

    return True


def _emit_promotion_deferred_event(
    registry: SetupWatchRegistry,
    watch: SetupWatch,
    reason_codes: tuple[str, ...],
    missing_data_types: tuple[str, ...],
) -> None:
    """Emit promotion_deferred_market_data event (Req 5.5)."""
    try:
        registry._emit_event(
            watch_id=watch.watch_id,
            profile_id=watch.profile_id,
            symbol=watch.symbol,
            event_type="promotion_deferred_market_data",
            maturity_score=watch.maturity_score,
            event_data=json.dumps({
                "reason_codes": list(reason_codes),
                "missing_data_types": list(missing_data_types),
                "evaluation_timestamp": to_iso(now_utc()),
                "summary": _build_event_summary(
                    watch.symbol, watch.side, "promotion deferred",
                    watch.maturity_score,
                ),
            }),
        )
    except Exception as exc:
        logger.warning(
            "Failed to emit promotion_deferred event for watch %s: %s",
            watch.watch_id, exc,
        )


def _build_evidence_package(
    watch: SetupWatch,
    cycle_id: str | None,
    signals: dict[str, dict] | None,
) -> dict:
    """Build enhanced evidence package for promoted candidate (Req 6.1-6.7).

    All fields present — NULL values as JSON null, never omitted.
    """
    # Parse stored JSON fields (may be None)
    last_eval = None
    if watch.last_evaluation_json:
        try:
            last_eval = json.loads(watch.last_evaluation_json)
        except (json.JSONDecodeError, TypeError):
            pass

    entry_zone = None
    if watch.entry_zone_json:
        try:
            entry_zone = json.loads(watch.entry_zone_json)
        except (json.JSONDecodeError, TypeError):
            pass

    draft_geometry = None
    if watch.draft_geometry_json:
        try:
            draft_geometry = json.loads(watch.draft_geometry_json)
        except (json.JSONDecodeError, TypeError):
            pass

    # Current signal data
    current_price = None
    key_levels = None
    if signals and watch.symbol in signals:
        signal = signals[watch.symbol]
        current_price = signal.get("current_price")
        key_levels = signal.get("key_levels")

    # Ready_at as ISO string or null
    ready_at_str = to_iso(watch.ready_at) if watch.ready_at else None

    return {
        "source_type": "setup_watch",
        "watch_id": watch.watch_id,
        "setup_type": watch.setup_type,
        "thesis": watch.thesis,
        "maturity_score": watch.maturity_score,
        "observed_cycles": watch.observed_cycles,
        "last_evaluation_json": last_eval,
        "ready_reference_price": watch.ready_reference_price,
        "ready_at": ready_at_str,
        "source_trigger": cycle_id,
        "entry_zone": entry_zone,
        "draft_geometry": draft_geometry,
        "current_price": current_price,
        "key_levels": key_levels,
        "source_provenance": {
            "source_type": watch.source_type,
            "source_id": watch.source_id,
            "source_cycle_id": watch.source_cycle_id,
        },
    }


def _build_event_summary(
    symbol: str, side: str, action: str, score: float | None
) -> str:
    """Build human-readable event summary (max 120 chars, Req 8.4)."""
    score_str = f" score={score:.2f}" if score is not None else ""
    raw = f"{symbol} {side} {action}{score_str}"
    return raw[:120]


# ────────────────────────────────────────────────────────────────────────────
# Public API — cycle evaluation
# ────────────────────────────────────────────────────────────────────────────


def evaluate_cycle(
    engine: Any,
    profile_id: str,
    cycle_id: str,
    signals: dict[str, dict],
    portfolio: dict,
) -> CycleEvaluationResult:
    """Run full setup watch evaluation for one PM cycle.

    Evaluation sequence:
      1. Expire TTL-elapsed watches
      2. Expire stale promoted watches from prior cycles
      3. Per active watch: evaluate invalidation → reject if triggered
      4. Per surviving watch: evaluate maturation → update_evaluation()
         → transition on threshold crossing
      5. increment_observed_cycles() for surviving active watches
      6. Enabled mode only: promote ready watches meeting cycle gate
      7. Create new watches from eligible signals
      8. Return summary

    Top-level try/except ensures the candidate pipeline is never blocked.
    Per-watch try/except inside evaluation loops ensures one failing watch
    does not abort processing of the rest.

    Logs a WARNING if execution exceeds 2 seconds.
    """
    start_time = time.monotonic()

    try:
        result = _evaluate_cycle_inner(
            engine, profile_id, cycle_id, signals, portfolio
        )
    except Exception as exc:
        logger.error(
            "Setup watch evaluate_cycle failed (fail-open): %s", exc, exc_info=True
        )
        result = CycleEvaluationResult(
            expired_ttl=0,
            expired_stale_promoted=0,
            invalidated=0,
            matured=0,
            regressed=0,
            promoted=0,
            still_active=0,
            created=0,
        )

    elapsed = time.monotonic() - start_time
    if elapsed > 2.0:
        logger.warning(
            "Setup watch evaluate_cycle took %.2fs (threshold 2s) "
            "for profile=%s cycle=%s",
            elapsed, profile_id, cycle_id,
        )

    return result


def _evaluate_cycle_inner(
    engine: Any,
    profile_id: str,
    cycle_id: str,
    signals: dict[str, dict],
    portfolio: dict,
) -> CycleEvaluationResult:
    """Inner implementation without the top-level safety net."""
    registry = SetupWatchRegistry(engine)

    # --- Step 1: Expire TTL-elapsed ---
    expired_ttl = registry.expire_elapsed(profile_id)

    # --- Step 2: Expire stale promoted from prior cycles ---
    expired_stale_promoted = registry.expire_stale_promoted(profile_id, cycle_id)

    # --- Step 3 & 4: Evaluate active watches ---
    active_watches = registry.get_active_watches(profile_id)

    invalidated = 0
    matured = 0
    regressed = 0
    surviving_ids: list[str] = []

    for watch in active_watches:
        # Skip promoted watches — they are waiting for PM resolution
        if watch.state == WatchState.PROMOTED:
            surviving_ids.append(watch.watch_id)
            continue

        try:
            market_ctx = _build_market_context_for_watch(
                watch.symbol, signals, portfolio
            )

            eval_result = evaluate_watch(
                watch.maturation_conditions_json,
                watch.invalidation_conditions_json,
                market_ctx,
            )

            # Persist evaluation result (fail-open)
            eval_json = json.dumps({
                "evaluated_at": eval_result.evaluation_timestamp,
                "maturity_score": eval_result.maturity_score,
                "conditions": [
                    {
                        "type": cr.condition_type,
                        "met": cr.met,
                        "detail": cr.detail,
                    }
                    for cr in eval_result.condition_results
                ],
                "invalidation_checked": True,
                "invalidation_triggered": eval_result.invalidated,
            })

            registry.update_evaluation(
                watch.watch_id,
                eval_result.maturity_score,
                eval_json,
            )

            # --- Entry zone tightening (Req 2.5) ---
            try:
                _maybe_tighten_entry_zone(engine, watch, signals)
            except Exception as ez_exc:
                logger.debug(
                    "Entry zone tightening failed for watch %s (non-critical): %s",
                    watch.watch_id, ez_exc,
                )

            # --- Invalidation check ---
            if eval_result.invalidated:
                try:
                    registry.transition_state(
                        watch.watch_id,
                        watch.state,
                        WatchState.REJECTED,
                        terminal_reason=f"invalidated: {eval_result.invalidation_reason}",
                    )
                    invalidated += 1
                except SetupWatchRegistryError as e:
                    logger.warning(
                        "Failed to reject invalidated watch %s: %s", watch.watch_id, e
                    )
                continue

            # --- Maturity state transitions ---
            score = eval_result.maturity_score
            current_state = watch.state

            # Determine desired state from score
            if score >= SETUP_WATCH_MATURITY_THRESHOLD:
                desired_state = WatchState.READY
            elif score > 0.0:
                desired_state = WatchState.MATURING
            else:
                desired_state = WatchState.WATCHING

            # Transition if needed
            if desired_state != current_state:
                # Forward maturation
                if (
                    (current_state == WatchState.WATCHING and desired_state == WatchState.MATURING)
                    or (current_state == WatchState.MATURING and desired_state == WatchState.READY)
                ):
                    try:
                        # Capture reference price on → READY
                        ref_price = None
                        if desired_state == WatchState.READY:
                            ref_price = market_ctx.get("current_price")

                        registry.transition_state(
                            watch.watch_id,
                            current_state,
                            desired_state,
                            ready_reference_price=ref_price,
                        )
                        matured += 1
                    except SetupWatchRegistryError as e:
                        logger.warning(
                            "Failed to transition watch %s %s→%s: %s",
                            watch.watch_id, current_state.value,
                            desired_state.value, e,
                        )

                # Skip-level forward: watching → ready (score jumped)
                elif current_state == WatchState.WATCHING and desired_state == WatchState.READY:
                    try:
                        # Must go watching → maturing first
                        registry.transition_state(
                            watch.watch_id,
                            WatchState.WATCHING,
                            WatchState.MATURING,
                        )
                        ref_price = market_ctx.get("current_price")
                        registry.transition_state(
                            watch.watch_id,
                            WatchState.MATURING,
                            WatchState.READY,
                            ready_reference_price=ref_price,
                        )
                        matured += 1
                    except SetupWatchRegistryError as e:
                        logger.warning(
                            "Failed skip-level transition for watch %s: %s",
                            watch.watch_id, e,
                        )

                # Regression
                elif (
                    (current_state == WatchState.READY and desired_state in (WatchState.MATURING, WatchState.WATCHING))
                    or (current_state == WatchState.MATURING and desired_state == WatchState.WATCHING)
                ):
                    try:
                        if current_state == WatchState.READY and desired_state == WatchState.WATCHING:
                            # Must go ready → maturing first
                            registry.transition_state(
                                watch.watch_id,
                                WatchState.READY,
                                WatchState.MATURING,
                            )
                            registry.transition_state(
                                watch.watch_id,
                                WatchState.MATURING,
                                WatchState.WATCHING,
                            )
                        else:
                            registry.transition_state(
                                watch.watch_id,
                                current_state,
                                desired_state,
                            )
                        regressed += 1
                    except SetupWatchRegistryError as e:
                        logger.warning(
                            "Failed regression transition for watch %s: %s",
                            watch.watch_id, e,
                        )

            # Emit maturity_evaluated event (at most once per watch per cycle)
            registry._emit_event(
                watch_id=watch.watch_id,
                profile_id=watch.profile_id,
                symbol=watch.symbol,
                event_type="maturity_evaluated",
                maturity_score=eval_result.maturity_score,
                event_data=json.dumps({
                    "cycle_id": cycle_id,
                    "score": eval_result.maturity_score,
                    "summary": _build_event_summary(
                        watch.symbol, watch.side, "maturity evaluated",
                        eval_result.maturity_score,
                    ),
                }),
            )

            surviving_ids.append(watch.watch_id)

        except Exception as exc:
            logger.warning(
                "Error evaluating watch %s (per-watch fail-open): %s",
                watch.watch_id, exc,
            )
            surviving_ids.append(watch.watch_id)

    # --- Step 5: Increment observed_cycles for surviving active watches ---
    if surviving_ids:
        registry.increment_observed_cycles(surviving_ids)

    # --- Step 6: Promotion (enabled mode only) ---
    promoted = 0
    if SETUP_WATCH_MODE == "enabled":
        # Re-fetch active watches to get updated state after transitions
        ready_watches = [
            w for w in registry.get_active_watches(profile_id)
            if w.state == WatchState.READY
            and w.observed_cycles >= SETUP_WATCH_PROMOTION_MIN_CYCLES
        ]
        for watch in ready_watches:
            try:
                success = _promote_ready_watch(
                    registry, watch, cycle_id=cycle_id, signals=signals
                )
                if success:
                    promoted += 1
            except Exception as e:
                logger.warning(
                    "Failed to promote watch %s: %s", watch.watch_id, e
                )

    # --- Step 7: Create new watches from eligible signals ---
    creation_stats = _create_watches_from_signals(
        engine, profile_id, cycle_id, signals, portfolio
    )

    # Count still-active
    still_active = registry.count_active(profile_id)

    return CycleEvaluationResult(
        expired_ttl=expired_ttl,
        expired_stale_promoted=expired_stale_promoted,
        invalidated=invalidated,
        matured=matured,
        regressed=regressed,
        promoted=promoted,
        still_active=still_active,
        created=creation_stats.created,
    )


# ────────────────────────────────────────────────────────────────────────────
# Public API — watch creation
# ────────────────────────────────────────────────────────────────────────────


def create_setup_watch(
    engine: Any,
    *,
    symbol: str,
    profile_id: str,
    side: str,
    setup_type: str,
    thesis: str,
    source_type: str,
    source_id: str | None,
    source_cycle_id: str,
    maturation_conditions: list[dict],
    invalidation_conditions: list[dict],
    entry_zone: dict | None = None,
    draft_geometry: dict | None = None,
    expires_at: datetime | None = None,
    signal_strength: str | None = None,
    portfolio: dict | None = None,
    key_levels: dict | None = None,
) -> str | None:
    """Create a new setup watch after applying all noise filters.

    Noise filters are applied in this order:
      0. Source provenance validation
      1. Categorical strength via STRENGTH_ORDER
      2. Maturation condition count
      3. Invalidation condition count >= 1
      4. Thesis length >= 10
      5. Per-symbol cap
      6. Exposure conflict
      7. Per-profile cap (evict oldest, not reject)
      8. Clamp expires_at to max TTL

    Returns the watch_id on success, None if rejected by a noise filter.

    NOTE: The CANDIDATE_EXECUTABLE_SETUP_TYPES allowlist is deliberately NOT
    applied here — it belongs at promotion time (Req 10.6).
    """
    registry = SetupWatchRegistry(engine)

    # --- Filter 0: Source provenance validation (Req 11.6, 11.8) ---
    if not _validate_source_provenance(source_type, source_id):
        return None

    # --- Filter 1: Signal strength check (CATEGORICAL) ---
    if signal_strength is not None:
        sig_val = STRENGTH_ORDER.get(str(signal_strength).lower(), 0)
        thr_val = STRENGTH_ORDER.get(SETUP_WATCH_MIN_CREATION_STRENGTH, 0)
        if sig_val < thr_val:
            logger.debug(
                "Watch creation rejected for %s: strength %s < threshold %s",
                symbol, signal_strength, SETUP_WATCH_MIN_CREATION_STRENGTH,
            )
            return None

    # --- Filter 2: Minimum maturation condition count ---
    if len(maturation_conditions) < SETUP_WATCH_MIN_CONDITION_COUNT:
        logger.debug(
            "Watch creation rejected for %s: %d maturation conditions < min %d",
            symbol, len(maturation_conditions), SETUP_WATCH_MIN_CONDITION_COUNT,
        )
        return None

    # --- Filter 3: At least one invalidation condition ---
    if len(invalidation_conditions) < 1:
        logger.debug(
            "Watch creation rejected for %s: no invalidation conditions",
            symbol,
        )
        return None

    # --- Filter 4: Thesis length ---
    if len(thesis.strip()) < 10:
        logger.debug(
            "Watch creation rejected for %s: thesis too short (%d chars)",
            symbol, len(thesis.strip()),
        )
        return None

    # --- Filter 5: Per-symbol cap ---
    if registry.count_active_for_symbol(symbol) >= SETUP_WATCH_MAX_PER_SYMBOL:
        logger.debug(
            "Watch creation rejected for %s: per-symbol cap reached (%d)",
            symbol, SETUP_WATCH_MAX_PER_SYMBOL,
        )
        return None

    # --- Filter 6: Exposure conflict ---
    if _has_exposure_conflict(engine, profile_id, symbol, side, portfolio):
        logger.debug(
            "Watch creation rejected for %s: exposure conflict", symbol
        )
        return None

    # --- Filter 7: Per-profile cap (evict oldest, don't reject) ---
    evicted = 0
    if registry.count_active(profile_id) >= SETUP_WATCH_MAX_ACTIVE_PER_PROFILE:
        _expire_oldest_active(registry, profile_id)
        evicted = 1

    # --- Clamp expires_at to max TTL ---
    now = now_utc()
    max_expiry = now + timedelta(hours=SETUP_WATCH_MAX_TTL_HOURS)
    if expires_at is None:
        final_expires_at = max_expiry
    else:
        final_expires_at = min(expires_at, max_expiry)

    # --- Build and persist the watch ---
    watch_id = str(uuid.uuid4())
    normalized_side = side.upper()

    # --- Draft geometry computation at creation (Req 2.1-2.4, 7.6) ---
    computed_entry_zone_json = json.dumps(entry_zone) if entry_zone else None
    computed_draft_geometry_json = json.dumps(draft_geometry) if draft_geometry else None

    # If key_levels provided and no explicit geometry passed, compute from levels
    if key_levels and isinstance(key_levels, dict):
        try:
            from utils.draft_geometry import (
                compute_draft_geometry,
                compute_entry_zone,
                entry_zone_to_json,
                geometry_to_json,
            )

            # Compute entry zone if not explicitly provided
            if not computed_entry_zone_json:
                ez = compute_entry_zone(key_levels, normalized_side)
                if ez is not None:
                    computed_entry_zone_json = entry_zone_to_json(ez)

            # Compute draft geometry if not explicitly provided
            if not computed_draft_geometry_json:
                dg = compute_draft_geometry(
                    key_levels, setup_type, normalized_side
                )
                if dg is not None:
                    computed_draft_geometry_json = geometry_to_json(dg)
        except Exception as exc:
            logger.debug(
                "Draft geometry computation failed for %s (non-critical): %s",
                symbol, exc,
            )

    # Build the SetupWatch object
    watch = SetupWatch(
        watch_id=watch_id,
        profile_id=profile_id,
        symbol=symbol,
        side=normalized_side,
        setup_type=setup_type,
        state=WatchState.WATCHING,
        thesis=thesis,
        source_type=source_type,
        source_id=source_id,
        source_cycle_id=source_cycle_id,
        maturation_conditions_json=json.dumps(maturation_conditions),
        invalidation_conditions_json=json.dumps(invalidation_conditions),
        last_evaluation_json=None,
        entry_zone_json=computed_entry_zone_json,
        draft_geometry_json=computed_draft_geometry_json,
        maturity_score=0.0,
        created_at=now,
        updated_at=now,
        expires_at=final_expires_at,
        state_changed_at=None,
        observed_cycles=0,
        ready_at=None,
        ready_reference_price=None,
        terminal_reason=None,
        promoted_cycle_id=None,
        execution_ref_type=None,
        execution_ref_id=None,
        integrity_hash="",  # computed inside create_watch
    )

    try:
        return registry.create_watch(watch)
    except SetupWatchRegistryError as e:
        logger.warning(
            "Watch creation failed for %s (registry error): %s", symbol, e
        )
        return None
    except Exception as e:
        logger.warning(
            "Watch creation failed for %s (unexpected): %s", symbol, e
        )
        return None


# ────────────────────────────────────────────────────────────────────────────
# Public API — promotion support
# ────────────────────────────────────────────────────────────────────────────


def get_promotable_watches(
    engine: Any, profile_id: str, cycle_id: str
) -> list[SetupWatch]:
    """Get watches in PROMOTED state for the given profile and cycle.

    These are the watches ready to be consumed by the candidate builder
    in enabled mode.
    """
    registry = SetupWatchRegistry(engine)
    return registry.get_promoted_watches(profile_id, cycle_id)


# ────────────────────────────────────────────────────────────────────────────
# Public API — post-PM result propagation
# ────────────────────────────────────────────────────────────────────────────


def propagate_candidate_results(
    engine: Any,
    cycle_id: str,
    profile_id: str,
    candidate_results: list[dict],
) -> None:
    """Propagate PM decisions back to originating setup watches.

    Branch order matters: the pending-order check precedes the rejection
    branches, because a candidate that produced a pending order may also carry
    a non-executed terminal state.

    Fail-open per candidate: one failure never aborts the rest.

    Requirements: 6.7, 6.7.1, 6.7.2, 6.8
    """
    registry = SetupWatchRegistry(engine)

    for result in candidate_results:
        try:
            snapshot_raw = result.get("signal_snapshot_json") or "{}"
            try:
                snapshot = json.loads(snapshot_raw)
            except (json.JSONDecodeError, TypeError):
                continue

            if snapshot.get("source_type") != "setup_watch":
                continue

            watch_id = snapshot.get("watch_id")
            if not watch_id:
                continue

            candidate_id = result.get("candidate_id")
            terminal_state = str(result.get("terminal_state", "")).upper()

            # Branch 1: Executed → ordered with trade ref
            if terminal_state == "EXECUTED":
                trade_id = result.get("trade_id", "")
                registry.transition_state(
                    watch_id,
                    WatchState.PROMOTED,
                    WatchState.ORDERED,
                    execution_ref_type="trade",
                    execution_ref_id=str(trade_id) if trade_id else "",
                )
                continue

            # Branch 2: Pending order lookup (takes precedence over rejection)
            order_id = _lookup_pending_order_id(engine, candidate_id)
            if order_id:
                registry.transition_state(
                    watch_id,
                    WatchState.PROMOTED,
                    WatchState.ORDERED,
                    execution_ref_type="pending_order",
                    execution_ref_id=order_id,
                )
                continue

            # Branch 3: Rejection states
            if terminal_state in ("REJECTED", "GATE_REJECTED", "SIZING_REJECTED"):
                registry.transition_state(
                    watch_id,
                    WatchState.PROMOTED,
                    WatchState.REJECTED,
                    terminal_reason=f"candidate_{terminal_state.lower()}",
                )
                continue

            # Branch 4: Non-consumed states → expired
            if terminal_state in ("NOT_SELECTED", "EXPIRED", "EXECUTION_FAILED"):
                registry.transition_state(
                    watch_id,
                    WatchState.PROMOTED,
                    WatchState.EXPIRED,
                    terminal_reason="promotion_not_consumed",
                )
                continue

            # Unknown terminal state — log and skip
            logger.debug(
                "Unknown terminal_state %r for watch %s candidate %s",
                terminal_state, watch_id, candidate_id,
            )

        except Exception as exc:
            logger.warning(
                "Failed to propagate result to watch (candidate %s): %s",
                result.get("candidate_id", "?"), exc,
            )


# ────────────────────────────────────────────────────────────────────────────
# Internal — pending order lookup
# ────────────────────────────────────────────────────────────────────────────


def _lookup_pending_order_id(engine: Any, candidate_id: str | None) -> str | None:
    """Resolve an active pending order for a candidate, if one exists.

    Fail-open: returns None on any error.
    """
    if not candidate_id:
        return None
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT order_id FROM pending_orders "
                    "WHERE candidate_id = :cid "
                    "  AND state IN ('pending', 'filling') "
                    "LIMIT 1"
                ),
                {"cid": candidate_id},
            ).fetchone()
        return row[0] if row else None
    except Exception as exc:
        logger.warning(
            "Pending order lookup failed for candidate %s: %s",
            candidate_id, exc,
        )
        return None


# ────────────────────────────────────────────────────────────────────────────
# Internal — market context builder
# ────────────────────────────────────────────────────────────────────────────


def _build_market_context_for_watch(
    symbol: str,
    signals: dict[str, dict],
    portfolio: dict,
) -> dict:
    """Build market_context dict for a specific symbol's watch evaluation.

    Returns dict with: current_price, market_regime, catalyst_timestamp,
    held_symbols, key_levels, symbol, current_hour_et.
    """
    signal = signals.get(symbol, {})

    # Held symbols from portfolio
    held_symbols: set[str] = set()
    positions = portfolio.get("positions", {})
    if isinstance(positions, dict):
        held_symbols = set(positions.keys())
    elif isinstance(positions, (list, set)):
        held_symbols = set(positions)

    # Current ET hour
    try:
        from datetime import timezone as tz
        import zoneinfo
        et_tz = zoneinfo.ZoneInfo("America/New_York")
        current_hour_et = datetime.now(et_tz).hour
    except Exception:
        # Fallback: approximate ET from UTC (offset -4 or -5)
        current_hour_et = (datetime.now(timezone.utc).hour - 4) % 24

    return {
        "symbol": symbol,
        "current_price": signal.get("current_price"),
        "market_regime": signal.get("market_regime"),
        "catalyst_timestamp": signal.get("catalyst_timestamp"),
        "held_symbols": held_symbols,
        "key_levels": signal.get("key_levels"),
        "current_hour_et": current_hour_et,
    }


# ────────────────────────────────────────────────────────────────────────────
# Internal — exposure conflict check
# ────────────────────────────────────────────────────────────────────────────


def _has_exposure_conflict(
    engine: Any,
    profile_id: str,
    symbol: str,
    side: str,
    portfolio: dict | None,
) -> bool:
    """Check if there's an existing exposure conflict for this symbol.

    Conflict exists if:
      - Portfolio has an open position for the symbol, OR
      - An active pending order exists for the same profile/symbol/side
    """
    # Check portfolio positions
    if portfolio:
        positions = portfolio.get("positions", {})
        if isinstance(positions, dict):
            if symbol in positions or symbol.upper() in positions:
                return True
        elif isinstance(positions, (list, set)):
            if symbol in positions or symbol.upper() in positions:
                return True

    # Check active pending orders
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT COUNT(*) FROM pending_orders "
                    "WHERE profile_id = :profile_id "
                    "  AND symbol = :symbol "
                    "  AND state IN ('pending', 'filling')"
                ),
                {"profile_id": profile_id, "symbol": symbol},
            ).fetchone()
            if row and row[0] > 0:
                return True
    except Exception as exc:
        logger.debug(
            "Pending order conflict check failed for %s/%s (assuming no conflict): %s",
            profile_id, symbol, exc,
        )

    return False


# ────────────────────────────────────────────────────────────────────────────
# Internal — entry zone tightening (Req 2.5)
# ────────────────────────────────────────────────────────────────────────────


def _maybe_tighten_entry_zone(
    engine: Any,
    watch: SetupWatch,
    signals: dict[str, dict],
) -> None:
    """If current signal provides a narrower entry zone, replace the stored one.

    Non-critical: logs and returns on error.
    """
    if not signals or watch.symbol not in signals:
        return

    signal = signals[watch.symbol]
    key_levels = signal.get("key_levels")
    if not key_levels or not isinstance(key_levels, dict):
        return

    try:
        from utils.draft_geometry import (
            compute_entry_zone,
            entry_zone_to_json,
            should_replace_entry_zone,
        )

        new_zone = compute_entry_zone(key_levels, watch.side)
        if new_zone is None:
            return

        if should_replace_entry_zone(watch.entry_zone_json, new_zone):
            new_json = entry_zone_to_json(new_zone)
            _update_entry_zone_json(engine, watch.watch_id, new_json)
            logger.debug(
                "Entry zone tightened for watch %s: %s",
                watch.watch_id, new_json,
            )
    except Exception as exc:
        logger.debug(
            "Entry zone tightening check failed for watch %s: %s",
            watch.watch_id, exc,
        )


def _update_entry_zone_json(engine: Any, watch_id: str, entry_zone_json: str) -> None:
    """Update entry_zone_json for a watch (non-CAS, fail-open)."""
    try:
        with engine.connect() as conn:
            conn.execute(
                text(
                    "UPDATE setup_watches "
                    "SET entry_zone_json = :ez_json, "
                    "    updated_at = :updated_at "
                    "WHERE watch_id = :watch_id"
                ),
                {
                    "ez_json": entry_zone_json,
                    "updated_at": to_iso(now_utc()),
                    "watch_id": watch_id,
                },
            )
            conn.commit()
    except Exception as exc:
        logger.warning(
            "Failed to update entry_zone_json for watch %s (fail-open): %s",
            watch_id, exc,
        )


# ────────────────────────────────────────────────────────────────────────────
# Internal — eviction
# ────────────────────────────────────────────────────────────────────────────


def _expire_oldest_active(registry: SetupWatchRegistry, profile_id: str) -> None:
    """Evict the oldest active watch for a profile to make room for a new one.

    Transitions the oldest active watch to EXPIRED with
    terminal_reason="capacity_evicted".
    """
    active = registry.get_active_watches(profile_id)
    if not active:
        return

    # get_active_watches returns oldest first
    oldest = active[0]
    try:
        registry.transition_state(
            oldest.watch_id,
            oldest.state,
            WatchState.EXPIRED,
            terminal_reason="capacity_evicted",
        )
        logger.debug(
            "Evicted oldest watch %s for profile %s (capacity)",
            oldest.watch_id, profile_id,
        )
    except SetupWatchRegistryError as e:
        logger.warning(
            "Failed to evict oldest watch %s: %s", oldest.watch_id, e
        )


# ────────────────────────────────────────────────────────────────────────────
# Internal — condition derivation (v1 heuristics)
# ────────────────────────────────────────────────────────────────────────────


def _derive_maturation_conditions(signal: dict) -> list[dict]:
    """Derive maturation conditions from a signal using v1 heuristics.

    Sources conditions from:
      - current_price + key_levels → price_zone (±2%)
      - market_regime → regime_aligned
      - Always: time_window for market hours (9-15 ET)
      - catalyst_timestamp → catalyst_fresh (120 min)
      - key_levels support/resistance → key_level_proximity (1.5%)
    """
    conditions: list[dict] = []
    num_conditions = 0

    current_price = signal.get("current_price")
    key_levels = signal.get("key_levels")
    market_regime = signal.get("market_regime")
    catalyst_ts = signal.get("catalyst_timestamp")

    # price_zone: ±2% of current price around key levels or current price
    if current_price is not None:
        try:
            price = float(current_price)
            low = round(price * 0.98, 2)
            high = round(price * 1.02, 2)
            conditions.append({
                "type": "price_zone",
                "params": {"low": low, "high": high},
                "weight": 0.3,
            })
            num_conditions += 1
        except (ValueError, TypeError):
            pass

    # regime_aligned
    if market_regime:
        conditions.append({
            "type": "regime_aligned",
            "params": {"required_regime": str(market_regime).lower()},
            "weight": 0.25,
        })
        num_conditions += 1

    # time_window: market hours (9 to 15 ET)
    conditions.append({
        "type": "time_window",
        "params": {"start_hour": 9, "end_hour": 15},
        "weight": 0.15,
    })
    num_conditions += 1

    # catalyst_fresh
    if catalyst_ts:
        conditions.append({
            "type": "catalyst_fresh",
            "params": {"max_age_minutes": 120},
            "weight": 0.15,
        })
        num_conditions += 1

    # key_level_proximity
    if key_levels and isinstance(key_levels, dict):
        # Check for support or resistance levels
        for level_type in ("support", "resistance"):
            level = _nearest_level(
                key_levels.get(level_type),
                prefer="support" if level_type == "support" else "resistance",
            )
            if level is not None:
                conditions.append({
                    "type": "key_level_proximity",
                    "params": {"level_type": level_type, "within_pct": 1.5},
                    "weight": 0.15,
                })
                num_conditions += 1
                break  # Only add one proximity condition

    return conditions


def _derive_invalidation_conditions(signal: dict) -> list[dict]:
    """Derive invalidation conditions from a signal using v1 heuristics.

    Sources conditions from:
      - key_levels support → price_breach below nearest support
      - market_regime → regime_flip with blocked regimes based on context
      - catalyst_timestamp → catalyst_expired (240 min max)
    """
    conditions: list[dict] = []

    key_levels = signal.get("key_levels")
    market_regime = signal.get("market_regime")
    catalyst_ts = signal.get("catalyst_timestamp")
    side = _infer_watch_side(signal) or "BUY"

    # price_breach below nearest support (for BUY) or above resistance (for SHORT)
    if key_levels and isinstance(key_levels, dict):
        if side == "BUY":
            nearest_support = _nearest_level(key_levels.get("support"), prefer="support")
            if nearest_support is not None:
                conditions.append({
                    "type": "price_breach",
                    "params": {"level": nearest_support, "direction": "below"},
                })
        else:  # SHORT
            nearest_resistance = _nearest_level(key_levels.get("resistance"), prefer="resistance")
            if nearest_resistance is not None:
                conditions.append({
                    "type": "price_breach",
                    "params": {"level": nearest_resistance, "direction": "above"},
                })

    # regime_flip — blocked regimes depend on side
    if market_regime:
        if side == "BUY":
            blocked = ["bearish", "crisis"]
        else:
            blocked = ["bullish", "euphoric"]
        conditions.append({
            "type": "regime_flip",
            "params": {"blocked_regimes": blocked},
        })

    # catalyst_expired
    if catalyst_ts:
        conditions.append({
            "type": "catalyst_expired",
            "params": {"max_age_minutes": 240},
        })

    return conditions


def _nearest_level(raw_level: Any, *, prefer: str) -> float | None:
    """Return one numeric support/resistance level from scalar or list input."""
    if isinstance(raw_level, list):
        values = []
        for item in raw_level:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                continue
        if not values:
            return None
        return max(values) if prefer == "support" else min(values)

    try:
        return float(raw_level)
    except (TypeError, ValueError):
        return None


def _infer_watch_side(signal: dict) -> str | None:
    """Infer a non-executable watch side without making HOLD executable."""
    for field in ("direction", "side", "signal", "original_signal"):
        value = str(signal.get(field) or "").upper()
        if value in {"BUY", "LONG"}:
            return "BUY"
        if value == "SHORT":
            return "SHORT"

    sanity = signal.get("deterministic_sanity")
    if isinstance(sanity, dict):
        bias = str(sanity.get("bias") or "").upper()
        if bias == "LONG":
            return "BUY"
        if bias == "SHORT":
            return "SHORT"

    return None


def _watch_thesis(signal: dict) -> str:
    """Extract an analyst thesis from current live payload shapes."""
    for field in ("thesis", "reason", "setup_reasoning", "reasoning", "invalidation"):
        value = signal.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _is_watchable_signal(signal: dict) -> bool:
    """Return True when a signal is worth tracking as non-executable setup state."""
    setup_type = signal.get("setup_type", "unknown")
    if setup_type not in WATCHABLE_SETUP_TYPES:
        return False

    direction = str(signal.get("signal", signal.get("direction", "")) or "").upper()
    if direction == "BUY":
        direction = "LONG"
    strength = str(signal.get("signal_strength", signal.get("strength", "")) or "").lower()
    confidence = str(signal.get("confidence", "") or "").lower()

    if direction in {"LONG", "SHORT"}:
        return STRENGTH_ORDER.get(strength, 0) >= STRENGTH_ORDER.get(
            SETUP_WATCH_MIN_CREATION_STRENGTH, 0
        )

    if direction == "HOLD":
        return (
            setup_type in SWING_EXECUTABLE_SETUP_TYPES
            and STRENGTH_ORDER.get(strength, 0) >= STRENGTH_ORDER.get(SETUP_WATCH_MIN_CREATION_STRENGTH, 0)
            and CONFIDENCE_ORDER.get(confidence, 0) >= CONFIDENCE_ORDER["medium"]
            and _infer_watch_side(signal) is not None
        )

    return False


# ────────────────────────────────────────────────────────────────────────────
# Internal — batch watch creation from signals
# ────────────────────────────────────────────────────────────────────────────


def _create_watches_from_signals(
    engine: Any,
    profile_id: str,
    cycle_id: str,
    signals: dict[str, dict],
    portfolio: dict,
) -> CreationStats:
    """Create new watches from current cycle signals using noise filters.

    Only creates watches from signals that are not already covered by
    an active watch for the same (profile, symbol, side, setup_type) key.

    Returns creation statistics.
    """
    registry = SetupWatchRegistry(engine)
    active_watches = registry.get_active_watches(profile_id)

    # Build set of already-watched keys
    watched_keys: set[tuple[str, str, str, str]] = set()
    for w in active_watches:
        watched_keys.add(w.active_key)

    attempted = 0
    created = 0
    rejected_weak_signal = 0
    rejected_insufficient_conditions = 0
    rejected_per_symbol_cap = 0
    rejected_exposure_conflict = 0
    rejected_short_thesis = 0
    evicted_for_capacity = 0

    for symbol, signal in signals.items():
        if not _is_watchable_signal(signal):
            continue

        # Skip signals already watched
        side = _infer_watch_side(signal)
        if side is None:
            continue
        setup_type = signal.get("setup_type", "unknown")
        key = (profile_id, symbol, side, setup_type)
        if key in watched_keys:
            continue

        # Skip signals without a thesis or reason
        thesis = _watch_thesis(signal)
        if not thesis:
            continue

        attempted += 1

        # Derive conditions
        maturation_conds = _derive_maturation_conditions(signal)
        invalidation_conds = _derive_invalidation_conditions(signal)

        signal_strength = signal.get("signal_strength", signal.get("strength"))

        # Use create_setup_watch which applies noise filters
        # But we need to track individual rejection reasons here
        # So we replicate the filter logic inline for stats tracking

        # Filter 1: Strength
        if signal_strength is not None:
            sig_val = STRENGTH_ORDER.get(str(signal_strength).lower(), 0)
            thr_val = STRENGTH_ORDER.get(SETUP_WATCH_MIN_CREATION_STRENGTH, 0)
            if sig_val < thr_val:
                rejected_weak_signal += 1
                continue

        # Filter 2: Maturation condition count
        if len(maturation_conds) < SETUP_WATCH_MIN_CONDITION_COUNT:
            rejected_insufficient_conditions += 1
            continue

        # Filter 3: Invalidation condition count
        if len(invalidation_conds) < 1:
            rejected_insufficient_conditions += 1
            continue

        # Filter 4: Thesis length
        if len(thesis.strip()) < 10:
            rejected_short_thesis += 1
            continue

        # Filter 5: Per-symbol cap
        if registry.count_active_for_symbol(symbol) >= SETUP_WATCH_MAX_PER_SYMBOL:
            rejected_per_symbol_cap += 1
            continue

        # Filter 6: Exposure conflict
        if _has_exposure_conflict(engine, profile_id, symbol, side, portfolio):
            rejected_exposure_conflict += 1
            continue

        # Filter 7: Per-profile cap (evict oldest)
        if registry.count_active(profile_id) >= SETUP_WATCH_MAX_ACTIVE_PER_PROFILE:
            _expire_oldest_active(registry, profile_id)
            evicted_for_capacity += 1

        # Clamp expires_at
        now = now_utc()
        final_expires_at = now + timedelta(hours=SETUP_WATCH_MAX_TTL_HOURS)

        # Build watch
        watch_id = str(uuid.uuid4())
        entry_zone = signal.get("entry_zone")
        draft_geometry = _extract_draft_geometry(signal)

        watch = SetupWatch(
            watch_id=watch_id,
            profile_id=profile_id,
            symbol=symbol,
            side=side,
            setup_type=setup_type,
            state=WatchState.WATCHING,
            thesis=thesis,
            source_type="analyst",
            source_id=signal.get("signal_id"),
            source_cycle_id=cycle_id,
            maturation_conditions_json=json.dumps(maturation_conds),
            invalidation_conditions_json=json.dumps(invalidation_conds),
            last_evaluation_json=None,
            entry_zone_json=json.dumps(entry_zone) if entry_zone else None,
            draft_geometry_json=json.dumps(draft_geometry) if draft_geometry else None,
            maturity_score=0.0,
            created_at=now,
            updated_at=now,
            expires_at=final_expires_at,
            state_changed_at=None,
            observed_cycles=0,
            ready_at=None,
            ready_reference_price=None,
            terminal_reason=None,
            promoted_cycle_id=None,
            execution_ref_type=None,
            execution_ref_id=None,
            integrity_hash="",
        )

        try:
            registry.create_watch(watch)
            created += 1
            watched_keys.add(key)
        except SetupWatchRegistryError as e:
            logger.warning(
                "Watch creation failed for %s in cycle creation: %s", symbol, e
            )
        except Exception as e:
            logger.warning(
                "Unexpected error creating watch for %s: %s", symbol, e
            )

    stats = CreationStats(
        attempted=attempted,
        created=created,
        rejected_weak_signal=rejected_weak_signal,
        rejected_insufficient_conditions=rejected_insufficient_conditions,
        rejected_per_symbol_cap=rejected_per_symbol_cap,
        rejected_exposure_conflict=rejected_exposure_conflict,
        rejected_short_thesis=rejected_short_thesis,
        evicted_for_capacity=evicted_for_capacity,
    )

    if attempted > 0:
        logger.info(
            "Setup watch creation: attempted=%d created=%d "
            "rejected_weak=%d rejected_conditions=%d rejected_symbol_cap=%d "
            "rejected_exposure=%d rejected_thesis=%d evicted=%d",
            stats.attempted, stats.created,
            stats.rejected_weak_signal, stats.rejected_insufficient_conditions,
            stats.rejected_per_symbol_cap, stats.rejected_exposure_conflict,
            stats.rejected_short_thesis, stats.evicted_for_capacity,
        )

    return stats


def _extract_draft_geometry(signal: dict) -> dict | None:
    """Extract draft geometry from a signal if available.

    Looks for explicit geometry fields or constructs from entry/stop/target.
    """
    # Check for explicit draft geometry
    geom = signal.get("draft_geometry")
    if geom and isinstance(geom, dict):
        return geom

    # Try to construct from individual fields
    entry = signal.get("entry_price", signal.get("entry"))
    stop = signal.get("stop_price", signal.get("stop"))
    target = signal.get("target_price", signal.get("target"))

    if entry is not None and stop is not None and target is not None:
        try:
            entry_f = float(entry)
            stop_f = float(stop)
            target_f = float(target)
            rr = abs(target_f - entry_f) / abs(entry_f - stop_f) if abs(entry_f - stop_f) > 0 else 0
            return {
                "entry": entry_f,
                "stop": stop_f,
                "target": target_f,
                "risk_reward": round(rr, 2),
            }
        except (ValueError, TypeError, ZeroDivisionError):
            pass

    return None
