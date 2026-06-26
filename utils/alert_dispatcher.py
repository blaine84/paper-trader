"""Alert Dispatcher — classifies intents and dispatches PM cycles.

Runs on a configurable interval (default 30s), separate from the 60s price
monitor. Classifies unclassified intents via LLM, checks eligibility
(cooldown, expiry, mutex, market hours), and triggers narrow-scope PM
evaluation when conditions are met.

Requirements: 6.1–6.5, 7.1–7.4, 8.1–8.5, 9.2–9.5, 12.1–12.5
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Optional

from utils.alert_intent_store import AlertIntentStore, AlertIntent

logger = logging.getLogger(__name__)


class AlertDispatcher:
    """Evaluates pending alert intents and dispatches narrow-scope PM cycles.

    Runs as a separate APScheduler job on a 30-second interval. Receives
    PM-cycle mutex callbacks via constructor injection to avoid circular
    imports from orchestrator.
    """

    def __init__(
        self,
        engine,
        intent_store: AlertIntentStore,
        begin_pm_cycle: Callable[[str], bool],
        end_pm_cycle: Callable[[str], None],
    ):
        self._engine = engine
        self._store = intent_store
        self._begin_pm_cycle = begin_pm_cycle
        self._end_pm_cycle = end_pm_cycle
        self._running = False  # overlap guard
        self._filter_failure_counts: dict[str, list[datetime]] = {}
        # dict maps symbol → list of failure timestamps (for consecutive failure tracking)

    def evaluate_and_dispatch(self) -> Optional[dict]:
        """Main entry point called by APScheduler every 30s.

        Steps:
        0. Check overlap guard and feature flag
        1. Recover stale dispatched/claimed intents (crash recovery)
        2. Classify unclassified intents (LLM filter sub-step)
        3. Mark expired intents
        4. Query pending classified intents
        5. Check eligibility (cooldown, expiry, market hours)
        6. Select dispatch set (urgency-based timing, 10-symbol cap)
        7. Dispatch or observe

        Returns dispatch result dict or None if no dispatch occurred.
        """
        from utils.gate_config import PM_ALERT_DISPATCH_MODE

        # Feature flag check
        if PM_ALERT_DISPATCH_MODE == "disabled":
            return None

        # Overlap guard: skip if already running
        if self._running:
            logger.debug("ALERT_DISPATCHER_SKIP: already running, skipping overlapping invocation")
            return None

        self._running = True
        try:
            # Step 1: Crash recovery
            self._recover_stale_intents()

            # Step 2: Classify unclassified intents via LLM filter
            self._classify_unclassified()

            # Step 3: Mark expired intents
            now = datetime.utcnow()
            expired_count = self._store.mark_expired(now=now)
            if expired_count > 0:
                logger.info("ALERT_DISPATCH_EXPIRED: marked %d intents as expired", expired_count)

            # Step 4: Query pending classified intents
            pending = self._store.query_pending(now=now)
            if not pending:
                return None

            # Step 5: Check eligibility
            eligible = self._check_eligibility(pending)
            if not eligible:
                return None

            # Step 6: Select dispatch set
            dispatch_set = self._select_dispatch_set(eligible)
            if not dispatch_set:
                return None

            # Extract symbols and intent IDs
            symbols = list(dict.fromkeys(intent.symbol for intent in dispatch_set))
            intent_ids = [intent.id for intent in dispatch_set]

            # Step 7: Dispatch or observe
            if PM_ALERT_DISPATCH_MODE == "observe":
                # Log would_dispatch events, do NOT call PM
                for intent in dispatch_set:
                    self._store.record_audit_log(
                        alert_intent_id=intent.alert_intent_id,
                        symbol=intent.symbol,
                        alert_type=intent.alert_type,
                        urgency=intent.urgency,
                        dispatch_status="would_dispatch",
                        reason="observe_mode",
                    )
                logger.info(
                    "ALERT_DISPATCH_OBSERVE: would dispatch %d intents for %s",
                    len(dispatch_set), symbols,
                )
                return {"mode": "observe", "would_dispatch": len(dispatch_set), "symbols": symbols}

            # Mode is "enabled": trigger actual PM cycle
            return self._trigger_pm_cycle(symbols, intent_ids)

        finally:
            self._running = False

    def _recover_stale_intents(self) -> None:
        """Crash recovery: sweep dispatched/claimed intents stuck beyond timeout.
        Called at startup and at beginning of each evaluate_and_dispatch() pass."""
        from utils.gate_config import (
            PM_ALERT_DISPATCH_STALE_MINUTES,
            PM_ALERT_SCHEDULED_MAX_RUNTIME_MINUTES,
        )

        now = datetime.utcnow()
        count = self._store.recover_stale_active_intents(
            dispatch_stale_minutes=PM_ALERT_DISPATCH_STALE_MINUTES,
            scheduled_max_runtime_minutes=PM_ALERT_SCHEDULED_MAX_RUNTIME_MINUTES,
            now=now,
        )
        if count > 0:
            logger.warning("STALE_INTENT_RECOVERY: recovered %d stale intents", count)

    def _classify_unclassified(self) -> None:
        """Run LLM filter on up to PM_ALERT_CLASSIFY_MAX_PER_PASS intents.

        Each call has a hard timeout (PM_ALERT_CLASSIFY_TIMEOUT_SECONDS).
        Updates filter_status to 'passed' or 'failed' and adjusts urgency.
        DB writes committed before LLM calls (short transactions per Req 11.2).
        """
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

        from utils.gate_config import PM_ALERT_CLASSIFY_TIMEOUT_SECONDS

        intents = self._store.query_unclassified()
        if not intents:
            return

        for intent in intents:
            # Build alert dict from intent fields for filter_alert_with_llm
            alert_dict = {
                "symbol": intent.symbol,
                "alert_type": intent.alert_type,
                "direction": intent.direction,
                "trigger_price": str(intent.trigger_price),
                "source_level": intent.source_level,
                "reason": intent.reason,
            }

            try:
                # Lazy import to avoid circular imports
                from agents.price_monitor import filter_alert_with_llm

                # Call LLM filter with timeout (ThreadPoolExecutor for Windows compat)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(filter_alert_with_llm, alert_dict, self._engine)
                    result = future.result(timeout=PM_ALERT_CLASSIFY_TIMEOUT_SECONDS)

                # On success: determine filter_status and urgency
                if result.get("actionable"):
                    # Passed filter — use urgency from result if provided, else keep current
                    new_urgency = result.get("urgency", intent.urgency)
                    if new_urgency not in ("high", "medium", "low"):
                        new_urgency = "medium"
                    self._store.update_classification(intent.id, "passed", new_urgency)
                else:
                    # LLM says not actionable — mark as failed
                    self._store.update_classification(intent.id, "failed", "medium")

                # Reset failure tracking on success for this symbol
                if intent.symbol in self._filter_failure_counts:
                    del self._filter_failure_counts[intent.symbol]

            except (FuturesTimeout, Exception) as exc:
                logger.warning(
                    "LLM filter failed for %s: %s", intent.symbol, str(exc)
                )
                self._handle_filter_failure(intent)

    def _handle_filter_failure(self, intent: AlertIntent) -> None:
        """Handle filter failure: set filter_status='failed', cap urgency.

        Track consecutive failures per symbol. If 3+ failures in 10 minutes,
        degrade urgency to 'low' (Requirement 4.5).
        """
        from datetime import timedelta

        now = datetime.utcnow()
        symbol = intent.symbol

        # Track failure timestamp
        if symbol not in self._filter_failure_counts:
            self._filter_failure_counts[symbol] = []
        self._filter_failure_counts[symbol].append(now)

        # Prune old failures (older than 10 minutes)
        cutoff = now - timedelta(minutes=10)
        self._filter_failure_counts[symbol] = [
            ts for ts in self._filter_failure_counts[symbol] if ts > cutoff
        ]

        # Determine urgency: cap at medium, degrade to low after 3+ failures
        if len(self._filter_failure_counts[symbol]) >= 3:
            new_urgency = "low"
        else:
            new_urgency = "medium"

        # Update classification as failed with capped urgency
        self._store.update_classification(intent.id, "failed", new_urgency)

    def _check_eligibility(self, intents: list[AlertIntent]) -> list[AlertIntent]:
        """Filter intents by cooldown, expiry, deferred_until, and market hours.

        Cooldown-deferred intents remain pending with deferred_until set.
        Only true stale duplicates are suppressed terminally.
        """
        now = datetime.utcnow()
        eligible = []

        # Market hours check: 09:30-16:00 ET on weekdays
        if not self._is_market_hours(now):
            return []

        # Check global cooldown once
        if self._store.is_global_cooled(now=now):
            return []

        for intent in intents:
            # Already expired? Skip (will be cleaned up by mark_expired)
            if intent.expiration_at <= now:
                continue

            # Deferred? Skip until deferred_until passes
            if intent.deferred_until is not None and intent.deferred_until > now:
                continue

            # Per-symbol cooldown check
            if self._store.is_symbol_cooled(intent.symbol, now=now):
                continue

            # Stale-duplicate check: suppress if same symbol+direction+trigger_price was recently dispatched
            if self._is_stale_duplicate(intent):
                self._store.mark_suppressed(intent.id, "stale_duplicate")
                self._store.record_audit_log(
                    alert_intent_id=intent.alert_intent_id,
                    symbol=intent.symbol,
                    alert_type=intent.alert_type,
                    urgency=intent.urgency,
                    dispatch_status="suppressed",
                    reason="stale_duplicate",
                )
                continue

            eligible.append(intent)

        return eligible

    def _is_market_hours(self, now: datetime) -> bool:
        """Check if current time is within US market hours (09:30-16:00 ET on weekdays)."""
        from pytz import timezone as pytz_timezone, utc as pytz_utc

        et = pytz_timezone("America/New_York")
        now_et = now.replace(tzinfo=pytz_utc).astimezone(et) if now.tzinfo is None else now.astimezone(et)

        # Must be a weekday (Mon=0, Fri=4)
        if now_et.weekday() > 4:
            return False

        # Must be between 09:30 and 16:00 ET
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= now_et <= market_close

    def _is_stale_duplicate(self, intent: AlertIntent) -> bool:
        """Check if intent matches the most recently dispatched intent for same symbol+direction+trigger_price.

        Returns True if an intent with the same symbol, direction, and trigger_price
        was recently dispatched (status='dispatched' or 'consumed').
        """
        from sqlalchemy import text

        with self._engine.begin() as conn:
            row = conn.execute(text("""
                SELECT 1 FROM alert_intents
                WHERE symbol = :symbol
                    AND direction = :direction
                    AND trigger_price = :trigger_price
                    AND dispatch_status IN ('dispatched', 'consumed')
                    AND id != :current_id
                ORDER BY dispatched_at DESC
                LIMIT 1
            """), {
                "symbol": intent.symbol,
                "direction": intent.direction,
                "trigger_price": str(intent.trigger_price),
                "current_id": intent.id,
            }).fetchone()
        return row is not None

    def _select_dispatch_set(self, eligible: list[AlertIntent]) -> list[AlertIntent]:
        """Apply urgency-based dispatch timing and 10-symbol cap.

        - High urgency: dispatch immediately (within this pass)
        - Medium/Low urgency: batch (include if 15 min since first_seen or scheduled PM due)
        - 10-symbol cap: highest urgency first, remainder stays pending
        """
        from datetime import timedelta

        now = datetime.utcnow()
        dispatch_set = []

        # Sort by urgency priority: high > medium > low, then by first_seen_at (FIFO)
        urgency_order = {"high": 0, "medium": 1, "low": 2}
        sorted_intents = sorted(
            eligible,
            key=lambda i: (urgency_order.get(i.urgency, 1), i.first_seen_at),
        )

        for intent in sorted_intents:
            if intent.urgency == "high":
                # High urgency: always include
                dispatch_set.append(intent)
            else:
                # Medium/Low: include if 15 minutes have passed since first_seen
                max_batch_wait = timedelta(minutes=15)
                if (now - intent.first_seen_at) >= max_batch_wait:
                    dispatch_set.append(intent)

        # Apply 10-symbol cap (deduplicate by symbol, keep highest urgency)
        seen_symbols: set[str] = set()
        capped_set: list[AlertIntent] = []
        for intent in dispatch_set:
            if intent.symbol not in seen_symbols:
                if len(seen_symbols) >= 10:
                    break
                seen_symbols.add(intent.symbol)
            capped_set.append(intent)

        return capped_set

    def _trigger_pm_cycle(self, symbols: list[str], intent_ids: list[int]) -> Optional[dict]:
        """Acquire PM mutex via injected callback and run narrow-scope PM.

        On PM success: mark intents consumed, record cooldowns.
        On PM failure: revert intents to pending with incremented attempt count.
        Always release mutex in finally block.
        """
        from datetime import timedelta
        from utils.gate_config import (
            PM_ALERT_SYMBOL_COOLDOWN_MINUTES,
            PM_ALERT_GLOBAL_COOLDOWN_MINUTES,
        )

        now = datetime.utcnow()

        # Acquire PM mutex
        if not self._begin_pm_cycle("alert_dispatcher"):
            # PM cycle already active — log skip, intents stay pending
            logger.info("ALERT_DISPATCH_SKIP: PM cycle already active, %d intents remain pending", len(intent_ids))
            return None

        try:
            # Mark intents as dispatched
            self._store.mark_dispatched(intent_ids, now)

            # Record per-symbol cooldowns
            symbol_cooldown_expiry = now + timedelta(minutes=PM_ALERT_SYMBOL_COOLDOWN_MINUTES)
            for sym in symbols:
                self._store.record_cooldown(sym, symbol_cooldown_expiry)

            # Record global cooldown
            global_cooldown_expiry = now + timedelta(minutes=PM_ALERT_GLOBAL_COOLDOWN_MINUTES)
            from utils.alert_intent_store import _GLOBAL_SENTINEL
            self._store.record_cooldown(_GLOBAL_SENTINEL, global_cooldown_expiry)

            # Get UUID strings from DB for PM lineage tracking
            from sqlalchemy import text as sa_text
            with self._engine.begin() as conn:
                placeholders = ", ".join(f":id_{i}" for i in range(len(intent_ids)))
                params = {f"id_{i}": id_ for i, id_ in enumerate(intent_ids)}
                rows = conn.execute(sa_text(f"""
                    SELECT alert_intent_id FROM alert_intents WHERE id IN ({placeholders})
                """), params).fetchall()
                alert_intent_id_list = [row[0] for row in rows]

            # Run PM for each active profile
            from agents.portfolio_manager import run_profile
            from models.pm_profiles import ACTIVE_PROFILES

            for profile_id in ACTIVE_PROFILES:
                run_profile(
                    self._engine,
                    symbols,
                    profile_id,
                    tier="medium",
                )

            # PM success: mark intents consumed
            self._store.mark_consumed(intent_ids)

            # Log dispatch to audit
            for sym in symbols:
                self._store.record_audit_log(
                    alert_intent_id=alert_intent_id_list[0] if alert_intent_id_list else "",
                    symbol=sym,
                    alert_type="entry_alert",
                    urgency="medium",
                    dispatch_status="dispatched",
                    cycle_trigger_type="alert",
                )

            logger.info("ALERT_DISPATCH_SUCCESS: dispatched %d intents for %s", len(intent_ids), symbols)
            return {"dispatched": len(intent_ids), "symbols": symbols}

        except Exception as exc:
            # PM failed: revert intents
            logger.error("ALERT_DISPATCH_FAILED: %s", str(exc))
            self._store.mark_dispatch_failed(intent_ids, str(exc))
            return None
        finally:
            self._end_pm_cycle("alert_dispatcher")

    def consume_for_scheduled_cycle(self) -> tuple[list[str], list[int]]:
        """Called by scheduled PM to claim pending intents.

        Returns (extra_symbols, claimed_intent_ids).
        Marks intents as 'claimed_by_scheduled'.
        Caller must call confirm_scheduled_consumption(ids) on PM success
        or revert_scheduled_claim(ids, error) on PM failure.
        """
        from utils.gate_config import PM_ALERT_DISPATCH_MODE

        if PM_ALERT_DISPATCH_MODE == "disabled":
            return [], []

        now = datetime.utcnow()
        # Query pending classified intents (not expired)
        intents = self._store.query_pending(now=now)
        if not intents:
            return [], []

        # Extract intent IDs and deduplicated symbols (preserve order)
        intent_ids = [intent.id for intent in intents]
        symbols = list(dict.fromkeys(intent.symbol for intent in intents))

        # Mark as claimed
        self._store.mark_claimed_by_scheduled(intent_ids)

        logger.info(
            "SCHEDULED_PM_CLAIM: claimed %d intents for symbols %s",
            len(intent_ids), symbols,
        )

        return symbols, intent_ids

    def confirm_scheduled_consumption(self, intent_ids: list[int]) -> None:
        """Transition claimed_by_scheduled → consumed after PM success."""
        if not intent_ids:
            return
        self._store.mark_consumed(intent_ids)
        logger.info(
            "SCHEDULED_PM_CONSUMED: confirmed consumption of %d intents",
            len(intent_ids),
        )

    def revert_scheduled_claim(self, intent_ids: list[int], error: str) -> None:
        """Transition claimed_by_scheduled → pending on PM failure."""
        if not intent_ids:
            return
        self._store.mark_claimed_back_to_pending(intent_ids, error)
        logger.warning(
            "SCHEDULED_PM_REVERT: reverted %d claimed intents to pending: %s",
            len(intent_ids), error,
        )
