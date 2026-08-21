"""Fast-path monitor — APScheduler job evaluating triggers against live quotes.

Runs on a fast cadence (default 15s) independently of the LLM-mediated PM
cycle.  Each tick acquires a non-blocking lock, expires stale triggers,
fetches fresh quotes (batched/deduplicated), evaluates active triggers, and
persists outcomes to ``fast_path_events``.

Execution-path outcomes (trade_executed, pending_order_created) are capped at
``FAST_PATH_MAX_OUTCOMES_PER_TICK`` per tick to prevent gate-pipeline stampede.
Excess outcomes are deferred and processed first-in-queue on the next tick.

Fail mode: fail-open at the monitor level — individual tick errors are logged
and partial results returned.  Per-trigger evaluation is fail-closed (handled
by evaluate_trigger's own wrapper).

See: .kiro/specs/fast-path-deterministic-execution/design.md
Requirements: 1.8, 2.8, 2.9, 2.10, 9.6
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text

from utils.fast_path_config import OUTCOME_TYPES
from utils.fast_path_evaluator import FastPathOutcome, evaluate_trigger
from utils.fast_path_registry import FastPathRegistry
from utils.gate_config import (
    FAST_PATH_ENABLED_SETUP_TYPES,
    FAST_PATH_MAX_OUTCOMES_PER_TICK,
    FAST_PATH_MODE,
)

logger = logging.getLogger(__name__)

# Outcome types that count against the per-tick execution cap.
_EXECUTION_PATH_OUTCOMES: frozenset[str] = frozenset(
    {"trade_executed", "pending_order_created"}
)

# Per-trigger evaluation timeout in seconds.
_TRIGGER_EVALUATION_TIMEOUT_SECONDS: float = 3.0


# ---------------------------------------------------------------------------
# Stubs for operations wired in later tasks
# ---------------------------------------------------------------------------


# Module-level quote cache: symbol → (fetch_time_monotonic, quote_dict)
_quote_cache: dict[str, tuple[float, dict]] = {}

# Cache TTL in seconds — reuse a cached quote if it was fetched less than 5s ago.
_QUOTE_CACHE_TTL_SECONDS: float = 5.0


def _fetch_quotes(symbols: set[str]) -> dict[str, dict]:
    """Fetch fresh quotes for the given symbols (batch, deduplicated).

    Uses a simple time-based cache: if a quote for a symbol was fetched less
    than 5 seconds ago, the cached value is reused.  Otherwise, calls
    ``FinnhubClient().get_quote(symbol)`` to obtain a fresh quote.

    Returns:
        Dict mapping symbol → quote dict with keys:
            - price (float): current price
            - age_ms (int): milliseconds since the quote was fetched
            - reliable (bool): True by default (market data reliability
              integration wired separately)
    """
    from utils.finnhub_client import FinnhubClient

    now = time.monotonic()
    results: dict[str, dict] = {}
    client = None  # lazy-init to avoid ValueError if key missing when all cached

    for symbol in symbols:
        # Check cache first
        cached = _quote_cache.get(symbol)
        if cached is not None:
            cached_time, cached_quote = cached
            age_seconds = now - cached_time
            if age_seconds < _QUOTE_CACHE_TTL_SECONDS:
                results[symbol] = {
                    "price": cached_quote.get("price", 0.0),
                    "age_ms": int(age_seconds * 1000),
                    "reliable": True,
                }
                continue

        # Cache miss or stale — fetch fresh quote
        try:
            if client is None:
                client = FinnhubClient()
            quote = client.get_quote(symbol)
            fetch_time = time.monotonic()
            _quote_cache[symbol] = (fetch_time, quote)
            results[symbol] = {
                "price": quote.get("price", 0.0),
                "age_ms": 0,
                "reliable": True,
            }
        except Exception as e:
            logger.warning(
                "fast_path_monitor: quote fetch failed for %s: %s",
                symbol,
                e,
            )
            # Skip this symbol — caller will handle missing quotes gracefully

    return results


def _generate_simple_narration(outcome: FastPathOutcome) -> str:
    """Generate a minimal template narration for the event.

    Produces a deterministic plain-English sentence from the outcome fields,
    suitable for display without additional context.  Will be replaced by
    ``generate_template_narration`` from ``utils.fast_path_stream`` once
    task 10.1 is implemented.

    Args:
        outcome: The evaluated fast-path outcome.

    Returns:
        A short narration string.
    """
    templates = {
        "missed_move": f"{outcome.symbol} target already crossed; no order created.",
        "stand_down": f"{outcome.symbol} setup blocked: {outcome.outcome_reason_code}.",
        "trade_executed": f"{outcome.symbol} trade executed at {outcome.current_price}.",
        "pending_order_created": f"{outcome.symbol} pending limit order created.",
        "watch_created": f"{outcome.symbol} watch created: {outcome.outcome_reason_code}.",
        "watch_promoted": f"{outcome.symbol} watch promoted to actionable.",
    }
    return templates.get(
        outcome.outcome_type,
        f"{outcome.symbol}: {outcome.outcome_reason_code}",
    )


def _persist_event(outcome: FastPathOutcome, trigger: Any, engine: Any) -> str:
    """Persist a fast-path event to the fast_path_events table.

    Generates a UUID4 event_id and INSERTs a row with all fields from the
    FastPathOutcome, plus audit metadata (market_data_age_ms,
    evaluation_duration_ms from outcome.metadata), a deterministic template
    narration, and annotation_status='annotation_pending'.

    Fail-open: on persistence error, logs the failure and returns the
    event_id anyway — event persistence never blocks the monitor tick.

    Args:
        outcome: The evaluated fast-path outcome to persist.
        trigger: The TriggerRecord that produced this outcome.
        engine: SQLAlchemy engine for database access.

    Returns:
        The generated event_id (UUID4 string).
    """
    event_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    # Generate deterministic template narration
    narration = _generate_simple_narration(outcome)

    # Extract audit timing fields from outcome metadata
    metadata = outcome.metadata or {}
    market_data_age_ms = metadata.get("market_data_age_ms")
    evaluation_duration_ms = metadata.get("evaluation_duration_ms")

    # Serialize metadata to JSON (None if empty)
    outcome_metadata_json: str | None = None
    if metadata:
        try:
            outcome_metadata_json = json.dumps(metadata)
        except (TypeError, ValueError) as e:
            logger.warning(
                "fast_path_monitor: failed to serialize outcome metadata "
                "for trigger %s: %s",
                outcome.trigger_id,
                e,
            )

    # Extract source_signal_id from trigger if available
    source_signal_id = getattr(trigger, "source_signal_id", None)

    try:
        with engine.connect() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO fast_path_events (
                        event_id, trigger_id, source_signal_id,
                        symbol, profile_id, setup_type,
                        direction, entry_price, stop_price, target_price,
                        current_price, reward_to_risk, outcome_type,
                        outcome_reason_code, outcome_metadata_json,
                        blocking_rule_name, blocking_rule_threshold,
                        annotation_status, narration, narration_source,
                        market_data_age_ms, evaluation_duration_ms,
                        evaluated_at, created_at
                    ) VALUES (
                        :event_id, :trigger_id, :source_signal_id,
                        :symbol, :profile_id, :setup_type,
                        :direction, :entry_price, :stop_price, :target_price,
                        :current_price, :reward_to_risk, :outcome_type,
                        :outcome_reason_code, :outcome_metadata_json,
                        :blocking_rule_name, :blocking_rule_threshold,
                        :annotation_status, :narration, :narration_source,
                        :market_data_age_ms, :evaluation_duration_ms,
                        :evaluated_at, :created_at
                    )
                    """
                ),
                {
                    "event_id": event_id,
                    "trigger_id": outcome.trigger_id,
                    "source_signal_id": source_signal_id,
                    "symbol": outcome.symbol,
                    "profile_id": outcome.profile_id,
                    "setup_type": outcome.setup_type,
                    "direction": outcome.direction,
                    "entry_price": outcome.entry_price,
                    "stop_price": outcome.stop_price,
                    "target_price": outcome.target_price,
                    "current_price": outcome.current_price,
                    "reward_to_risk": outcome.reward_to_risk,
                    "outcome_type": outcome.outcome_type,
                    "outcome_reason_code": outcome.outcome_reason_code,
                    "outcome_metadata_json": outcome_metadata_json,
                    "blocking_rule_name": outcome.blocking_rule_name,
                    "blocking_rule_threshold": outcome.blocking_rule_threshold,
                    "annotation_status": "annotation_pending",
                    "narration": narration,
                    "narration_source": "template",
                    "market_data_age_ms": market_data_age_ms,
                    "evaluation_duration_ms": evaluation_duration_ms,
                    "evaluated_at": now_iso,
                    "created_at": now_iso,
                },
            )
            conn.commit()
    except Exception as e:
        logger.error(
            "fast_path_monitor: failed to persist event for trigger %s: %s",
            outcome.trigger_id,
            e,
        )
    return event_id


def _delegate_execution(outcome: FastPathOutcome, trigger: Any, engine: Any) -> None:
    """Delegate execution-path outcome to existing infrastructure.

    Stub implementation — will be wired to execution delegation in task 8.
    Covers trade_executed (gate pipeline + execute_trade) and
    pending_order_created (maybe_create_pending_order).
    """
    logger.info(
        "fast_path_monitor: execution delegation stub called for %s "
        "(outcome=%s, trigger=%s)",
        outcome.symbol,
        outcome.outcome_type,
        outcome.trigger_id,
    )


# ---------------------------------------------------------------------------
# FastPathMonitor
# ---------------------------------------------------------------------------


class FastPathMonitor:
    """Orchestrates per-tick trigger evaluation across all profiles.

    Maintains per-profile FastPathRegistry instances, a non-blocking tick
    lock, and a deferred-execution queue for outcomes that exceed the
    per-tick cap.

    Usage:
        monitor = FastPathMonitor(engine, ["conservative", "moderate"])
        summary = monitor.run_tick()

    The monitor is designed to be called by APScheduler on a fixed interval
    (FAST_PATH_MONITOR_INTERVAL_SECONDS).  If a tick is still running when
    the next interval fires, the new invocation skips immediately.
    """

    def __init__(self, engine: Any, profile_ids: list[str]) -> None:
        """Initialize monitor with engine and profile registries.

        Args:
            engine: SQLAlchemy engine for database access.
            profile_ids: List of profile identifiers to evaluate.
        """
        self._engine = engine
        self._profile_ids = profile_ids
        self._registries: dict[str, FastPathRegistry] = {
            pid: FastPathRegistry(db=engine, profile_id=pid)
            for pid in profile_ids
        }
        self._tick_lock = threading.Lock()
        self._deferred_queue: list[tuple[FastPathOutcome, Any]] = []

    def run_tick(self) -> dict:
        """Execute a single evaluation pass across all profiles.

        Steps:
          1. Acquire tick lock (non-blocking) — skip if already held
          2. Process deferred queue first (from prior ticks)
          3. Expire stale triggers per registry
          4. Fetch quotes for all unique symbols with active triggers
          5. Evaluate each trigger with per-trigger timeout watchdog
          6. Persist outcomes and mark triggers as fired
          7. Delegate execution-path outcomes up to cap, defer remainder
          8. Release tick lock in finally block
          9. Return tick summary dict

        Returns:
            Dict with keys: evaluated, fired, expired, skipped, deferred,
            outcomes (dict of outcome_type → count).
        """
        # Step 1: Non-blocking lock acquisition
        acquired = self._tick_lock.acquire(blocking=False)
        if not acquired:
            logger.warning(
                "fast_path_monitor: tick still running, skipping"
            )
            return {
                "evaluated": 0,
                "fired": 0,
                "expired": 0,
                "skipped": 1,
                "deferred": 0,
                "outcomes": {},
            }

        try:
            return self._execute_tick()
        except Exception as e:
            logger.error(
                "fast_path_monitor: unexpected error during tick: %s", e
            )
            return {
                "evaluated": 0,
                "fired": 0,
                "expired": 0,
                "skipped": 0,
                "deferred": 0,
                "outcomes": {},
                "error": str(e),
            }
        finally:
            # Step 8: Always release lock
            self._tick_lock.release()

    def _execute_tick(self) -> dict:
        """Internal tick logic — called while holding the tick lock."""
        summary: dict[str, int] = {
            "evaluated": 0,
            "fired": 0,
            "expired": 0,
            "skipped": 0,
            "deferred": 0,
        }
        outcome_counts: dict[str, int] = {}
        execution_count = 0

        # Step 2: Process deferred queue from prior ticks
        deferred_to_process = list(self._deferred_queue)
        self._deferred_queue.clear()

        for deferred_outcome, deferred_trigger in deferred_to_process:
            if execution_count >= FAST_PATH_MAX_OUTCOMES_PER_TICK:
                # Still over cap — re-defer
                self._deferred_queue.append((deferred_outcome, deferred_trigger))
                summary["deferred"] += 1
                continue
            try:
                _delegate_execution(deferred_outcome, deferred_trigger, self._engine)
                execution_count += 1
            except Exception as e:
                logger.error(
                    "fast_path_monitor: deferred execution failed for %s: %s",
                    deferred_outcome.trigger_id,
                    e,
                )

        # Step 3: Expire stale triggers per registry
        for profile_id, registry in self._registries.items():
            try:
                expired_count = registry.expire_stale_triggers()
                summary["expired"] += expired_count
                if expired_count > 0:
                    logger.debug(
                        "fast_path_monitor: expired %d triggers for profile %s",
                        expired_count,
                        profile_id,
                    )
            except Exception as e:
                logger.error(
                    "fast_path_monitor: expire_stale_triggers failed for profile %s: %s",
                    profile_id,
                    e,
                )

        # Step 4: Gather active triggers and deduplicate symbols for quote fetch
        all_triggers: list[tuple[str, Any]] = []  # (profile_id, trigger)
        symbols_needed: set[str] = set()

        for profile_id, registry in self._registries.items():
            try:
                triggers = registry.get_active_triggers()
                for trigger in triggers:
                    all_triggers.append((profile_id, trigger))
                    symbols_needed.add(trigger.symbol)
            except Exception as e:
                logger.error(
                    "fast_path_monitor: get_active_triggers failed for profile %s: %s",
                    profile_id,
                    e,
                )

        # Fetch quotes in batch (deduplicated by symbol)
        quotes: dict[str, dict] = {}
        if symbols_needed:
            try:
                quotes = _fetch_quotes(symbols_needed)
            except Exception as e:
                logger.error(
                    "fast_path_monitor: quote fetch failed: %s", e
                )
                # No quotes means all triggers get stale_market_data or skip
                quotes = {}

        # Step 5 & 6: Evaluate each trigger
        for profile_id, trigger in all_triggers:
            summary["evaluated"] += 1

            # Get quote for this trigger's symbol
            quote = quotes.get(trigger.symbol)
            if quote is None:
                logger.debug(
                    "fast_path_monitor: no quote available for %s, skipping trigger %s",
                    trigger.symbol,
                    trigger.trigger_id,
                )
                continue

            # Step 5: Per-trigger evaluation with 3s timeout watchdog
            outcome = self._evaluate_with_timeout(trigger, quote, profile_id)

            if outcome is None:
                # Trigger condition not met — stays active
                continue

            # Trigger fired — record outcome
            summary["fired"] += 1
            outcome_counts[outcome.outcome_type] = (
                outcome_counts.get(outcome.outcome_type, 0) + 1
            )

            # Step 6: Persist event and mark trigger as fired
            event_id = _persist_event(outcome, trigger, self._engine)

            registry = self._registries[profile_id]
            try:
                registry.mark_fired(trigger.trigger_id, event_id)
            except Exception as e:
                # Fail-open at monitor level: log, continue with next trigger
                logger.error(
                    "fast_path_monitor: mark_fired failed for trigger %s: %s",
                    trigger.trigger_id,
                    e,
                )

            # Step 7: Execution delegation with cap
            if outcome.outcome_type in _EXECUTION_PATH_OUTCOMES:
                if FAST_PATH_MODE == "enabled":
                    # Setup-type gate: only delegate for enabled setup types
                    if outcome.setup_type not in FAST_PATH_ENABLED_SETUP_TYPES:
                        logger.debug(
                            "fast_path_monitor: setup_type %s not in "
                            "FAST_PATH_ENABLED_SETUP_TYPES, skipping delegation "
                            "for trigger %s (observe-equivalent)",
                            outcome.setup_type,
                            trigger.trigger_id,
                        )
                        continue
                    if execution_count < FAST_PATH_MAX_OUTCOMES_PER_TICK:
                        try:
                            _delegate_execution(outcome, trigger, self._engine)
                            execution_count += 1
                        except Exception as e:
                            logger.error(
                                "fast_path_monitor: execution delegation failed "
                                "for trigger %s: %s",
                                trigger.trigger_id,
                                e,
                            )
                    else:
                        # Cap reached — defer to next tick
                        self._deferred_queue.append((outcome, trigger))
                        summary["deferred"] += 1
                        logger.info(
                            "fast_path_monitor: execution cap reached, deferring "
                            "trigger %s (%s)",
                            trigger.trigger_id,
                            outcome.outcome_type,
                        )

        summary["outcomes"] = outcome_counts
        return summary

    def _evaluate_with_timeout(
        self, trigger: Any, quote: dict, profile_id: str
    ) -> FastPathOutcome | None:
        """Evaluate a single trigger with a 3-second watchdog.

        If evaluation exceeds _TRIGGER_EVALUATION_TIMEOUT_SECONDS, produces
        a stand_down("evaluation_timeout") outcome instead of blocking the
        entire tick.

        Args:
            trigger: TriggerRecord to evaluate.
            quote: Dict with price, age_ms, reliable fields.
            profile_id: Owning profile.

        Returns:
            FastPathOutcome or None if trigger condition not met.
        """
        profile_state = {"profile_id": profile_id}
        start = time.monotonic()

        # Run evaluation — the evaluator itself is fail-closed (returns
        # stand_down on error), so we only need to watch for excessive duration.
        outcome = evaluate_trigger(trigger, quote, profile_state)

        elapsed = time.monotonic() - start
        if elapsed > _TRIGGER_EVALUATION_TIMEOUT_SECONDS:
            logger.warning(
                "fast_path_monitor: trigger %s evaluation took %.2fs "
                "(exceeds %.1fs watchdog), forcing stand_down",
                trigger.trigger_id,
                elapsed,
                _TRIGGER_EVALUATION_TIMEOUT_SECONDS,
            )
            return FastPathOutcome(
                outcome_type="stand_down",
                outcome_reason_code="evaluation_timeout",
                trigger_id=trigger.trigger_id,
                symbol=trigger.symbol,
                profile_id=trigger.profile_id,
                direction=trigger.direction,
                setup_type=trigger.setup_type,
                current_price=quote.get("price", 0.0),
                entry_price=trigger.entry_price,
                stop_price=trigger.stop_price,
                target_price=trigger.target_price,
                metadata={"evaluation_duration_s": round(elapsed, 3)},
            )

        return outcome


# ---------------------------------------------------------------------------
# APScheduler job registration
# ---------------------------------------------------------------------------


def register_fast_path_job(scheduler, engine, profile_ids: list[str]) -> None:
    """Register the fast-path monitor as an APScheduler interval job.

    Only registers when FAST_PATH_MODE is not "disabled". The job runs on a
    fixed interval (FAST_PATH_MONITOR_INTERVAL_SECONDS) and uses the
    orchestrator's market-hours guard to ensure evaluation only occurs during
    US equity regular session (9:30-16:00 ET).

    Follows the same registration pattern as the pending_order_monitor:
    IntervalTrigger, max_instances=1, coalesce=True, replace_existing=True.

    Args:
        scheduler: APScheduler BlockingScheduler instance.
        engine: SQLAlchemy engine for database access.
        profile_ids: List of profile identifiers to evaluate.

    Requirements: 1.4, 1.8
    """
    from utils.gate_config import FAST_PATH_MODE, FAST_PATH_MONITOR_INTERVAL_SECONDS

    if FAST_PATH_MODE == "disabled":
        return

    from apscheduler.triggers.interval import IntervalTrigger as _FastPathIntervalTrigger

    monitor = FastPathMonitor(engine, profile_ids)

    def _run_fast_path_monitor():
        """Evaluate registered triggers against fresh market data."""
        # Import here to avoid circular imports — mirrors orchestrator pattern
        from orchestrator import _skip_outside_regular_market_job

        if _skip_outside_regular_market_job("fast_path_monitor"):
            return
        try:
            summary = monitor.run_tick()
            if summary.get("fired", 0) > 0 or summary.get("expired", 0) > 0:
                logger.info(
                    "FAST_PATH_TICK: evaluated=%d fired=%d expired=%d "
                    "deferred=%d outcomes=%s",
                    summary.get("evaluated", 0),
                    summary.get("fired", 0),
                    summary.get("expired", 0),
                    summary.get("deferred", 0),
                    summary.get("outcomes", {}),
                )
        except Exception as e:
            logger.error("Fast-path monitor error: %s", e, exc_info=True)

    scheduler.add_job(
        _run_fast_path_monitor,
        _FastPathIntervalTrigger(seconds=FAST_PATH_MONITOR_INTERVAL_SECONDS),
        id="fast_path_monitor",
        max_instances=1,
        replace_existing=True,
        coalesce=True,
    )

    logger.info(
        "Fast-path monitor job registered (interval=%ds, profiles=%s)",
        FAST_PATH_MONITOR_INTERVAL_SECONDS,
        profile_ids,
    )
