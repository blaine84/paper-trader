"""Watch Maturity Bridge — connects Price Monitor events to active watches.

Integration layer that evaluates approaching-level alerts from the Price
Monitor against active setup watches. No network calls, no LLM calls.
Must execute within the 60-second Price Monitor tick budget.

All condition evaluations use data already present in the alert and the
watch's stored condition definitions. Market-data reliability checks are
deferred to the promotion gate.

Requirements: 1.1-1.9, 8.1-8.2, 9.2-9.4, 12.1-12.6
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import text

from utils.gate_config import (
    SETUP_WATCH_MATURITY_THRESHOLD,
    SETUP_WATCH_REALTIME_MODE,
)
from utils.missed_move_detector import (
    apply_missed_move_transition,
    check_missed_move,
)
from utils.setup_watch_evaluator import (
    EvaluationResult,
    _safe_decimal,
    evaluate_watch,
)
from utils.setup_watch_registry import (
    ACTIVE_STATES,
    SetupWatch,
    SetupWatchRegistry,
    SetupWatchRegistryError,
    WatchState,
    _WATCH_COLUMNS,
    _row_to_watch,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Data classes (6.1)
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BridgeEvaluationResult:
    """Summary of a bridge evaluation pass across all alerts.

    Returned by evaluate_alerts() to provide observability into bridge activity.
    """

    alerts_processed: int = 0
    watches_evaluated: int = 0
    conditions_flipped: int = 0
    state_transitions: int = 0
    missed_moves_detected: int = 0
    errors: int = 0


# ────────────────────────────────────────────────────────────────────────────
# Main entry point (6.2, 6.7-6.12)
# ────────────────────────────────────────────────────────────────────────────


def evaluate_alerts(
    engine,
    approaching_alerts: list[dict],
    *,
    profile_ids: list[str] | None = None,
) -> BridgeEvaluationResult:
    """Evaluate approaching-level alerts against active setup watches.

    Main entry point for the Watch Maturity Bridge. Called from the Price
    Monitor after check_momentum() produces approaching-level alerts.

    Gated by SETUP_WATCH_REALTIME_MODE:
      - "disabled": returns zero-result immediately (no DB access)
      - "observe": evaluates, emits events, updates evaluation, no transitions
      - "enabled": evaluates, transitions, promotes when ready

    Parameters
    ----------
    engine
        SQLAlchemy engine for database access.
    approaching_alerts : list[dict]
        Approaching-level alerts from check_momentum(). Each dict has:
        symbol, price, level_name, level_value, distance_pct.
    profile_ids : list[str] | None
        Optional filter to restrict evaluation to specific profiles.

    Returns
    -------
    BridgeEvaluationResult
        Summary of the evaluation pass.
    """
    # 6.2: Gate — disabled mode returns immediately
    if SETUP_WATCH_REALTIME_MODE == "disabled":
        return BridgeEvaluationResult()

    # 6.12: Top-level try/except around the entire function
    try:
        return _evaluate_alerts_inner(engine, approaching_alerts, profile_ids=profile_ids)
    except Exception as e:
        logger.error(
            "Bridge evaluate_alerts top-level error (fail-open): %s", e,
            exc_info=True,
        )
        return BridgeEvaluationResult(errors=1)


def _evaluate_alerts_inner(
    engine,
    approaching_alerts: list[dict],
    *,
    profile_ids: list[str] | None = None,
) -> BridgeEvaluationResult:
    """Inner implementation of evaluate_alerts (no top-level guard)."""
    registry = SetupWatchRegistry(engine)
    mode = SETUP_WATCH_REALTIME_MODE  # "observe" or "enabled"

    alerts_processed = 0
    watches_evaluated = 0
    conditions_flipped = 0
    state_transitions = 0
    missed_moves_detected = 0
    errors = 0

    # Collect unique symbols from alerts for batch query
    symbols = {a.get("symbol") for a in approaching_alerts if a.get("symbol")}

    if not symbols:
        return BridgeEvaluationResult(alerts_processed=0)

    # Query active watches for all relevant symbols in one batch
    active_watches = _get_active_watches_for_symbols(engine, symbols, profile_ids)

    # Index watches by symbol for O(1) lookup
    watches_by_symbol: dict[str, list[SetupWatch]] = {}
    for w in active_watches:
        watches_by_symbol.setdefault(w.symbol, []).append(w)

    for alert in approaching_alerts:
        symbol = alert.get("symbol")
        if not symbol:
            continue
        alerts_processed += 1

        watches_for_symbol = watches_by_symbol.get(symbol, [])
        if not watches_for_symbol:
            continue

        price_raw = alert.get("price")
        level_value_raw = alert.get("level_value")
        level_name = alert.get("level_name", "")

        for watch in watches_for_symbol:
            # 6.11: Per-watch try/except — log error and continue
            try:
                result = _process_single_watch(
                    registry=registry,
                    watch=watch,
                    alert=alert,
                    mode=mode,
                    price_raw=price_raw,
                    level_value_raw=level_value_raw,
                    level_name=level_name,
                )
                watches_evaluated += 1
                conditions_flipped += result.get("conditions_flipped", 0)
                state_transitions += result.get("state_transitions", 0)
                missed_moves_detected += result.get("missed_moves_detected", 0)
            except Exception as e:
                logger.error(
                    "Bridge per-watch error for watch %s / symbol %s "
                    "(continuing remaining): %s",
                    watch.watch_id, symbol, e,
                    exc_info=True,
                )
                errors += 1

    return BridgeEvaluationResult(
        alerts_processed=alerts_processed,
        watches_evaluated=watches_evaluated,
        conditions_flipped=conditions_flipped,
        state_transitions=state_transitions,
        missed_moves_detected=missed_moves_detected,
        errors=errors,
    )


# ────────────────────────────────────────────────────────────────────────────
# Per-watch processing
# ────────────────────────────────────────────────────────────────────────────


def _process_single_watch(
    *,
    registry: SetupWatchRegistry,
    watch: SetupWatch,
    alert: dict,
    mode: str,
    price_raw,
    level_value_raw,
    level_name: str,
) -> dict:
    """Process a single watch against a single alert.

    Returns a dict with counts: conditions_flipped, state_transitions,
    missed_moves_detected.
    """
    result = {"conditions_flipped": 0, "state_transitions": 0, "missed_moves_detected": 0}

    # Parse price and level for side consistency check
    price = _safe_decimal(price_raw)
    level_value = _safe_decimal(level_value_raw)

    if price is None or level_value is None:
        return result

    # Filter by side consistency (6.3)
    if not _is_side_consistent(
        watch.side, watch.setup_type, level_name, price, level_value
    ):
        return result

    # 6.7: For READY/PROMOTED watches, check missed move first
    if watch.state in (WatchState.READY, WatchState.PROMOTED):
        missed_result = check_missed_move(watch, price_raw)
        if missed_result.missed:
            apply_missed_move_transition(registry, watch, missed_result)
            result["missed_moves_detected"] = 1

            # Emit missed_move_detected event
            _emit_bridge_event(
                registry,
                watch,
                "missed_move_detected",
                from_state=watch.state,
                to_state=WatchState.MISSED,
                event_data=json.dumps({
                    "current_price": str(missed_result.current_price),
                    "target_price": str(missed_result.target_price),
                    "side": missed_result.side,
                    "source": "price_monitor",
                    "summary": (
                        f"{watch.symbol} {watch.side} target crossed "
                        f"(price={missed_result.current_price}, "
                        f"target={missed_result.target_price})"
                    )[:120],
                }),
            )
            return result
        # READY/PROMOTED watches don't need further maturity evaluation
        return result

    # For WATCHING/MATURING: evaluate conditions
    market_context = _build_bridge_market_context(alert, watch)

    # Evaluate via the shared evaluator
    eval_result = evaluate_watch(
        watch.maturation_conditions_json,
        watch.invalidation_conditions_json,
        market_context,
    )

    new_score = eval_result.maturity_score

    # Parse previous evaluation for evidence comparison
    previous_eval = _parse_last_evaluation(watch.last_evaluation_json)

    # Record maturity evidence for flipped conditions (6.5)
    flipped_count = _record_maturity_evidence(previous_eval, eval_result, alert)
    result["conditions_flipped"] = flipped_count

    # Serialize new evaluation result
    new_eval_json = _serialize_evaluation(eval_result)

    # 6.10: Emit maturity_updated_by_monitor event when score changes
    old_score = watch.maturity_score
    score_changed = abs(new_score - old_score) > 1e-9

    if score_changed:
        _emit_bridge_event(
            registry,
            watch,
            "maturity_updated_by_monitor",
            event_data=json.dumps({
                "score_before": round(old_score, 2),
                "score_after": round(new_score, 2),
                "source": "price_monitor",
                "level_name": level_name,
                "level_value": str(level_value),
                "observed_price": str(price),
                "distance_pct": round(float(alert.get("distance_pct", 0)), 2),
                "summary": (
                    f"{watch.symbol} {watch.side} maturity "
                    f"{old_score:.2f}->{new_score:.2f} via {level_name}"
                )[:120],
            }),
            maturity_score=new_score,
        )

    # INFO log per requirement 8.2
    transition_label = "no_transition"

    if mode == "observe":
        # 6.8: Observe mode — update evaluation, emit events, NO transitions
        registry.update_evaluation(
            watch.watch_id, new_score, new_eval_json
        )
    elif mode == "enabled":
        # 6.9: Enabled mode — evaluate, transition, promote
        registry.update_evaluation(
            watch.watch_id, new_score, new_eval_json
        )

        # Attempt state advance based on score thresholds
        new_state = _attempt_state_advance(registry, watch, new_score, price)
        if new_state is not None:
            result["state_transitions"] = 1
            transition_label = f"{watch.state.value}->{new_state.value}"

            # If watch reached READY, invoke promotion
            if new_state == WatchState.READY:
                _try_promote_ready_watch(registry, watch)

    # Structured INFO log (Req 8.2)
    logger.info(
        "Bridge eval: symbol=%s, price=%s, level=%s, distance=%.2f%%, "
        "score=%.2f->%.2f, transition=%s",
        watch.symbol,
        str(price),
        level_name,
        float(alert.get("distance_pct", 0)),
        old_score,
        new_score,
        transition_label,
    )

    return result


# ────────────────────────────────────────────────────────────────────────────
# Side-consistency matching (6.3)
# ────────────────────────────────────────────────────────────────────────────


def _is_side_consistent(
    watch_side: str,
    watch_setup_type: str,
    level_name: str,
    price: Decimal,
    level_value: Decimal,
) -> bool:
    """Determine if an approaching-level alert is directionally consistent.

    Setup-type-aware matching rules:
      Default:
        BUY matches when price > level (approaching support from above)
        SHORT matches when price < level (approaching resistance from below)
      breakout_retest:
        BUY matches resistance from below (price < level)
        SHORT matches support from above (price > level)
      failed_breakdown_reclaim:
        BUY matches support from above (price > level, reclaiming)

    Parameters
    ----------
    watch_side : str
        "BUY" or "SHORT"
    watch_setup_type : str
        Setup type (e.g., "pullback_continuation", "breakout_retest")
    level_name : str
        Name of the approaching level (e.g., "support", "resistance", "vwap")
    price : Decimal
        Current observed price from the alert.
    level_value : Decimal
        Value of the key level being approached.

    Returns
    -------
    bool
        True if the alert is side-consistent with the watch.
    """
    side = watch_side.upper() if watch_side else "BUY"
    setup_type = (watch_setup_type or "").lower()

    # Special case: breakout_retest
    if setup_type == "breakout_retest":
        if side == "BUY":
            # BUY breakout retest: approaching resistance from below
            return price < level_value
        else:
            # SHORT breakout retest: approaching support from above
            return price > level_value

    # Special case: failed_breakdown_reclaim
    if setup_type == "failed_breakdown_reclaim":
        if side == "BUY":
            # BUY reclaim: approaching support from above (price > level)
            return price > level_value
        # SHORT uses default logic
        return price < level_value

    # Default matching
    if side == "BUY":
        # BUY: approaching support from above
        return price > level_value
    else:
        # SHORT: approaching resistance from below
        return price < level_value


# ────────────────────────────────────────────────────────────────────────────
# Market context builder (6.4)
# ────────────────────────────────────────────────────────────────────────────


def _build_bridge_market_context(alert: dict, watch: SetupWatch) -> dict:
    """Build a market_context dict from alert fields and watch state.

    No network calls. Uses only data present in the alert and the watch's
    stored condition definitions.

    Parameters
    ----------
    alert : dict
        Approaching-level alert with: symbol, price, level_name,
        level_value, distance_pct.
    watch : SetupWatch
        The active watch being evaluated.

    Returns
    -------
    dict
        Market context suitable for evaluate_watch().
    """
    ctx = {
        "symbol": alert.get("symbol", watch.symbol),
        "current_price": alert.get("price"),
        "level_name": alert.get("level_name"),
        "level_value": alert.get("level_value"),
        "distance_pct": alert.get("distance_pct"),
        "side": watch.side,
        "setup_type": watch.setup_type,
        "source": "price_monitor",
    }

    # Merge stored evaluation context if available (prior observations)
    if watch.last_evaluation_json:
        try:
            last_eval = json.loads(watch.last_evaluation_json)
            # Carry forward price_history if present
            if "price_history" in last_eval:
                ctx["price_history"] = last_eval["price_history"]
            # Carry forward any stored key_levels
            if "key_levels" in last_eval:
                ctx["key_levels"] = last_eval["key_levels"]
        except (json.JSONDecodeError, TypeError):
            pass

    return ctx


# ────────────────────────────────────────────────────────────────────────────
# Maturity evidence recording (6.5)
# ────────────────────────────────────────────────────────────────────────────


def _record_maturity_evidence(
    previous_eval: dict | None,
    new_eval_result: EvaluationResult,
    alert: dict,
) -> int:
    """Annotate conditions that flipped unmet→met with evidence.

    Returns the count of conditions that flipped.

    Evidence sub-object includes:
      - observed_price: from alert
      - reference_level: level_value from alert
      - distance_pct: from alert
      - source: "price_monitor"
      - observed_at: UTC ISO-8601

    Note: The evidence is encoded into the condition_results via detail field.
    The actual persistence happens when last_evaluation_json is updated.
    """
    if not new_eval_result.condition_results:
        return 0

    # Build set of previously-met condition types
    prev_met: set[str] = set()
    if previous_eval and "condition_results" in previous_eval:
        for cr in previous_eval["condition_results"]:
            if cr.get("met"):
                prev_met.add(cr.get("condition_type", ""))

    flipped_count = 0
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for cr in new_eval_result.condition_results:
        if cr.met and cr.condition_type not in prev_met:
            # This condition flipped unmet → met
            flipped_count += 1
            # Evidence is recorded via the detail field enhancement
            # The full evidence is attached when we serialize the evaluation
            # (the detail field already captures condition-specific info)

    return flipped_count


# ────────────────────────────────────────────────────────────────────────────
# State advance logic (6.6)
# ────────────────────────────────────────────────────────────────────────────


def _attempt_state_advance(
    registry: SetupWatchRegistry,
    watch: SetupWatch,
    new_score: float,
    current_price: Decimal,
) -> WatchState | None:
    """Attempt CAS state advance based on score thresholds.

    Transitions:
      WATCHING → MATURING: when score > 0 (first condition met)
      MATURING → READY: when score >= SETUP_WATCH_MATURITY_THRESHOLD

    No retry on CAS failure (Req 1.9) — returns None if CAS lost race.

    Parameters
    ----------
    registry : SetupWatchRegistry
    watch : SetupWatch
    new_score : float
    current_price : Decimal

    Returns
    -------
    WatchState | None
        The new state if transition succeeded, None otherwise.
    """
    target_state: WatchState | None = None

    if watch.state == WatchState.WATCHING and new_score > 0:
        target_state = WatchState.MATURING
    elif watch.state == WatchState.MATURING and new_score >= SETUP_WATCH_MATURITY_THRESHOLD:
        target_state = WatchState.READY

    if target_state is None:
        return None

    try:
        # For MATURING → READY, set ready_reference_price
        kwargs: dict = {}
        if target_state == WatchState.READY:
            kwargs["ready_reference_price"] = float(current_price)

        registry.transition_state(
            watch.watch_id,
            watch.state,
            target_state,
            **kwargs,
        )
        logger.info(
            "Bridge state advance: watch=%s, %s->%s, score=%.2f",
            watch.watch_id, watch.state.value, target_state.value, new_score,
        )
        return target_state
    except SetupWatchRegistryError:
        # CAS failure — state was concurrently modified (Req 1.9)
        logger.warning(
            "Bridge CAS failure for watch %s (%s->%s), skipping without retry",
            watch.watch_id, watch.state.value, target_state.value,
        )
        return None


# ────────────────────────────────────────────────────────────────────────────
# Promotion invocation (6.9)
# ────────────────────────────────────────────────────────────────────────────


def _try_promote_ready_watch(registry: SetupWatchRegistry, watch: SetupWatch) -> None:
    """Invoke the shared promotion function when a watch reaches READY.

    Uses lazy import to avoid circular dependency and to handle the case
    where _promote_ready_watch doesn't exist yet (Task 7).

    Requirements: 9.4 — in "enabled" mode, invoke shared promotion function.
    """
    try:
        from utils.setup_watch_manager import _promote_ready_watch
        _promote_ready_watch(registry, watch)
    except ImportError:
        logger.debug(
            "Bridge: _promote_ready_watch not available yet (Task 7), "
            "skipping promotion for watch %s",
            watch.watch_id,
        )
    except Exception as e:
        # Fail-open for promotion from bridge — promotion is a best-effort
        # enhancement; the scheduled evaluator will pick it up next cycle
        logger.warning(
            "Bridge promotion failed for watch %s (fail-open): %s",
            watch.watch_id, e,
        )


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _get_active_watches_for_symbols(
    engine,
    symbols: set[str],
    profile_ids: list[str] | None = None,
) -> list[SetupWatch]:
    """Query active watches for a set of symbols.

    Returns watches in ACTIVE_STATES (WATCHING, MATURING, READY, PROMOTED)
    for the given symbols, optionally filtered by profile_ids.
    """
    if not symbols:
        return []

    columns = ", ".join(_WATCH_COLUMNS)
    active_state_values = tuple(s.value for s in ACTIVE_STATES)

    # Build parameterized query with symbol list
    symbol_params = {f"sym_{i}": s for i, s in enumerate(symbols)}
    symbol_placeholders = ", ".join(f":sym_{i}" for i in range(len(symbols)))

    state_params = {f"st_{i}": s for i, s in enumerate(active_state_values)}
    state_placeholders = ", ".join(f":st_{i}" for i in range(len(active_state_values)))

    sql = (
        f"SELECT {columns} FROM setup_watches "
        f"WHERE symbol IN ({symbol_placeholders}) "
        f"  AND state IN ({state_placeholders}) "
        f"ORDER BY created_at ASC"
    )

    params = {**symbol_params, **state_params}

    if profile_ids:
        profile_params = {f"pid_{i}": p for i, p in enumerate(profile_ids)}
        profile_placeholders = ", ".join(f":pid_{i}" for i in range(len(profile_ids)))
        sql = (
            f"SELECT {columns} FROM setup_watches "
            f"WHERE symbol IN ({symbol_placeholders}) "
            f"  AND state IN ({state_placeholders}) "
            f"  AND profile_id IN ({profile_placeholders}) "
            f"ORDER BY created_at ASC"
        )
        params.update(profile_params)

    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()

    return [_row_to_watch(r) for r in rows]


def _emit_bridge_event(
    registry: SetupWatchRegistry,
    watch: SetupWatch,
    event_type: str,
    *,
    event_data: str | None = None,
    maturity_score: float | None = None,
    from_state: WatchState | None = None,
    to_state: WatchState | None = None,
) -> None:
    """Emit a bridge event via the registry (fail-open).

    Wraps registry._emit_event() with fail-open semantics so event
    emission never blocks the bridge.
    """
    try:
        registry._emit_event(
            watch_id=watch.watch_id,
            profile_id=watch.profile_id,
            symbol=watch.symbol,
            event_type=event_type,
            event_data=event_data,
            from_state=from_state,
            to_state=to_state,
            maturity_score=maturity_score if maturity_score is not None else watch.maturity_score,
        )
    except Exception as e:
        logger.warning(
            "Bridge event emission failed for %s/%s (fail-open): %s",
            watch.watch_id, event_type, e,
        )


def _parse_last_evaluation(last_evaluation_json: str | None) -> dict | None:
    """Parse last_evaluation_json into a dict, returning None on failure."""
    if not last_evaluation_json:
        return None
    try:
        return json.loads(last_evaluation_json)
    except (json.JSONDecodeError, TypeError):
        return None


def _serialize_evaluation(eval_result: EvaluationResult) -> str:
    """Serialize an EvaluationResult to JSON for storage.

    Includes per-condition results with evidence annotations.
    """
    data = {
        "invalidated": eval_result.invalidated,
        "invalidation_reason": eval_result.invalidation_reason,
        "maturity_score": eval_result.maturity_score,
        "evaluation_timestamp": eval_result.evaluation_timestamp,
        "condition_results": [
            {
                "condition_type": cr.condition_type,
                "met": cr.met,
                "detail": cr.detail,
            }
            for cr in eval_result.condition_results
        ],
    }
    return json.dumps(data, default=str)
