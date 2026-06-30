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
        5. Per-alert mode routing: mode → freshness → deferral pipeline
        6. Check eligibility on dispatch candidates (cooldown, expiry, market hours)
        7. Select dispatch set (urgency-based timing, 10-symbol cap)
        8. Trigger PM cycle for dispatch candidates

        Returns dispatch result dict or None if no dispatch occurred.

        Requirements: 1.2, 1.3, 1.4, 1.5, 10.2
        """
        from utils.dispatch_mode import build_dispatch_mode_config
        from utils.gate_config import (
            PM_ALERT_DISPATCH_MODE,
            PM_ALERT_MODE_ENTRY_ALERT,
            PM_ALERT_MODE_BREAKOUT,
            PM_ALERT_MODE_RAPID_MOVE,
            PM_ALERT_MODE_TARGET_HIT,
        )

        # Build resolved mode config (handles "enabled" → "dispatch" alias)
        mode_config = build_dispatch_mode_config(
            global_mode_raw=PM_ALERT_DISPATCH_MODE,
            env_values={
                "entry_alert": PM_ALERT_MODE_ENTRY_ALERT,
                "breakout": PM_ALERT_MODE_BREAKOUT,
                "rapid_move": PM_ALERT_MODE_RAPID_MOVE,
                "target_hit": PM_ALERT_MODE_TARGET_HIT,
            },
        )

        # If global resolves to disabled, short-circuit
        if mode_config.global_mode == "disabled":
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

            # Step 3: Expire active intents past their expiration_at (with audit records)
            now = datetime.utcnow()
            expired_count = self._expire_active_intents(now)
            if expired_count > 0:
                logger.info("ALERT_DISPATCH_EXPIRED: marked %d intents as expired", expired_count)

            # Step 4: Query pending classified intents
            pending = self._store.query_pending(now=now)
            if not pending:
                return None

            # Step 5: Per-alert mode routing — mode → freshness → deferral pipeline
            dispatch_candidates = []
            observe_count = 0
            for intent in pending:
                effective = mode_config.effective_mode(intent.alert_type)

                if effective == "disabled":
                    # Lightweight structured log only — no DB audit row
                    logger.debug(
                        "ALERT_DISPATCH_DISABLED_SKIP: symbol=%s alert_type=%s",
                        intent.symbol, intent.alert_type,
                    )
                    continue

                if not self._check_freshness(intent, now):
                    # Stale: transition to expired, audit
                    self._expire_for_freshness(intent, now, configured_mode=effective)
                    continue

                if self._is_deferred(intent, now):
                    continue  # Silently skip deferred

                if effective == "observe":
                    self._handle_observe(intent, now)
                    observe_count += 1
                    continue

                # effective == "dispatch": add to dispatch candidates
                dispatch_candidates.append(intent)

            # If only observations occurred (no dispatch candidates), return observe summary
            if not dispatch_candidates:
                if observe_count > 0:
                    return {"mode": "observe", "would_dispatch": observe_count, "symbols": []}
                return None

            # Step 6: Check eligibility on dispatch candidates only
            eligible = self._check_eligibility(dispatch_candidates)
            if not eligible:
                return None

            # Step 7: Select dispatch set
            dispatch_set = self._select_dispatch_set(eligible)
            if not dispatch_set:
                return None

            # Extract symbols and intent IDs
            symbols = list(dict.fromkeys(intent.symbol for intent in dispatch_set))
            intent_ids = [intent.id for intent in dispatch_set]

            # Step 8: Trigger PM cycle
            return self._trigger_pm_cycle(dispatch_set, symbols, intent_ids)

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

        # Recover stale PM claims (stuck in 'claimed' status beyond timeout)
        from agents.portfolio_manager import recover_stale_claims
        from utils.gate_config import PM_ALERT_CLAIM_STALE_MINUTES
        recovered = recover_stale_claims(self._engine, stale_minutes=PM_ALERT_CLAIM_STALE_MINUTES)
        if recovered > 0:
            logger.info("STALE_CLAIM_RECOVERY: recovered %d stale PM claims", recovered)

    def _expire_for_freshness(self, intent: AlertIntent, now: datetime, configured_mode: str = "dispatch") -> None:
        """Transition stale intent to expired with structured audit record.

        Audit records BOTH ages for analysis:
        - freshness_age_seconds: now - last_seen_at (dispatch freshness)
        - first_seen_age_seconds: now - first_seen_at (total intent lifetime)

        Requirements: 3.3, 3.4, 4.4, 7.1, 7.2
        """
        last_seen_age = (now - intent.last_seen_at).total_seconds() if intent.last_seen_at else -1
        first_seen_age = (now - intent.first_seen_at).total_seconds() if intent.first_seen_at else -1
        reason = "freshness_expired" if intent.last_seen_at else "undetermined_freshness"

        self._store.transition_to_expired(intent.id, reason=reason)
        self._store.record_audit_log(
            alert_intent_id=intent.alert_intent_id,
            symbol=intent.symbol,
            alert_type=intent.alert_type,
            urgency=intent.urgency,
            dispatch_status="expired",
            reason=reason,
            freshness_age_seconds=last_seen_age,
            first_seen_age_seconds=first_seen_age,
            configured_mode=configured_mode,
            dedupe_key=intent.dedupe_key,
            trigger_price=intent.trigger_price,
            occurrence_count=intent.occurrence_count,
        )
        logger.info(
            "ALERT_FRESHNESS_EXPIRED: symbol=%s alert_type=%s last_seen_age=%.1f first_seen_age=%.1f reason=%s",
            intent.symbol, intent.alert_type, last_seen_age, first_seen_age, reason,
        )

    def _is_pm_cycle_active(self) -> bool:
        """Non-destructive probe: check if a PM cycle is currently active.

        Attempts to acquire the PM cycle mutex and immediately releases it.
        Returns True if a PM cycle is active (mutex busy), False otherwise.

        Used to protect dispatched intents from expiration while their PM cycle
        is still in progress (Requirement 4.5).
        """
        if self._begin_pm_cycle("_expiration_probe_"):
            self._end_pm_cycle("_expiration_probe_")
            return False
        return True

    def _expire_active_intents(self, now: datetime) -> int:
        """Expire active intents whose expiration_at has passed, with audit records.

        Queries all active intents (pending, dispatched, claimed_by_scheduled) where
        expiration_at < now and transitions them to expired with structured audit
        records. Respects the PM cycle guard: dispatched intents are NOT expired
        while a PM cycle is actively processing them.

        Returns count of intents expired.

        Requirements: 4.1, 4.2, 4.3, 4.5
        """
        expired_intents = self._store.query_active_past_expiration(now=now)
        if not expired_intents:
            return 0

        pm_cycle_active = self._is_pm_cycle_active()
        expired_count = 0

        for intent in expired_intents:
            # Requirement 4.5: Do NOT expire dispatched intents while a PM cycle
            # is actively processing. Only expire after the PM cycle releases them.
            if intent.dispatch_status == "dispatched" and pm_cycle_active:
                continue

            # Transition to expired with reason
            success = self._store.transition_to_expired(intent.id, reason="age_limit_reached")
            if not success:
                continue  # Already in terminal state (race condition)

            # Write audit record (Requirement 4.4, 7.2)
            first_seen_age = (now - intent.first_seen_at).total_seconds() if intent.first_seen_at else -1
            freshness_age = (now - intent.last_seen_at).total_seconds() if intent.last_seen_at else -1
            self._store.record_audit_log(
                alert_intent_id=intent.alert_intent_id,
                symbol=intent.symbol,
                alert_type=intent.alert_type,
                urgency=intent.urgency,
                dispatch_status="expired",
                reason="age_limit_reached",
                first_seen_age_seconds=first_seen_age,
                freshness_age_seconds=freshness_age,
                configured_mode="dispatch",
                dedupe_key=intent.dedupe_key,
                trigger_price=intent.trigger_price,
                occurrence_count=intent.occurrence_count,
            )
            expired_count += 1

        if expired_count > 0:
            logger.info(
                "ALERT_EXPIRATION_CHECK: expired %d intents (reason=age_limit_reached)",
                expired_count,
            )

        return expired_count

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

        Writes audit records for ALL rejection reasons so every non-disabled
        intent that enters evaluation produces exactly one audit record.

        Requirements: 7.1, 7.2
        """
        from datetime import timedelta
        from utils.gate_config import PM_ALERT_SYMBOL_COOLDOWN_MINUTES

        now = datetime.utcnow()
        eligible = []

        # Market hours check: 09:30-16:00 ET on weekdays
        if not self._is_market_hours(now):
            # Write audit record for each intent rejected by market hours
            for intent in intents:
                first_seen_age = (now - intent.first_seen_at).total_seconds() if intent.first_seen_at else -1
                freshness_age = (now - intent.last_seen_at).total_seconds() if intent.last_seen_at else -1
                self._store.record_audit_log(
                    alert_intent_id=intent.alert_intent_id,
                    symbol=intent.symbol,
                    alert_type=intent.alert_type,
                    urgency=intent.urgency,
                    dispatch_status="skipped",
                    reason="market_closed",
                    configured_mode="dispatch",
                    dedupe_key=intent.dedupe_key,
                    trigger_price=intent.trigger_price,
                    occurrence_count=intent.occurrence_count,
                    freshness_age_seconds=freshness_age,
                    first_seen_age_seconds=first_seen_age,
                )
            return []

        # Check global cooldown once
        if self._store.is_global_cooled(now=now):
            # Write audit record for each intent rejected by global cooldown
            for intent in intents:
                first_seen_age = (now - intent.first_seen_at).total_seconds() if intent.first_seen_at else -1
                freshness_age = (now - intent.last_seen_at).total_seconds() if intent.last_seen_at else -1
                self._store.record_audit_log(
                    alert_intent_id=intent.alert_intent_id,
                    symbol=intent.symbol,
                    alert_type=intent.alert_type,
                    urgency=intent.urgency,
                    dispatch_status="skipped",
                    reason="cooldown_active",
                    configured_mode="dispatch",
                    dedupe_key=intent.dedupe_key,
                    trigger_price=intent.trigger_price,
                    occurrence_count=intent.occurrence_count,
                    freshness_age_seconds=freshness_age,
                    first_seen_age_seconds=first_seen_age,
                )
            return []

        for intent in intents:
            first_seen_age = (now - intent.first_seen_at).total_seconds() if intent.first_seen_at else -1
            freshness_age = (now - intent.last_seen_at).total_seconds() if intent.last_seen_at else -1

            # Already expired? Skip (will be cleaned up by mark_expired)
            if intent.expiration_at <= now:
                self._store.record_audit_log(
                    alert_intent_id=intent.alert_intent_id,
                    symbol=intent.symbol,
                    alert_type=intent.alert_type,
                    urgency=intent.urgency,
                    dispatch_status="skipped",
                    reason="already_expired",
                    configured_mode="dispatch",
                    dedupe_key=intent.dedupe_key,
                    trigger_price=intent.trigger_price,
                    occurrence_count=intent.occurrence_count,
                    freshness_age_seconds=freshness_age,
                    first_seen_age_seconds=first_seen_age,
                )
                continue

            # Deferred? Skip until deferred_until passes
            if intent.deferred_until is not None and intent.deferred_until > now:
                self._store.record_audit_log(
                    alert_intent_id=intent.alert_intent_id,
                    symbol=intent.symbol,
                    alert_type=intent.alert_type,
                    urgency=intent.urgency,
                    dispatch_status="deferred",
                    reason="cooldown_active",
                    configured_mode="dispatch",
                    dedupe_key=intent.dedupe_key,
                    trigger_price=intent.trigger_price,
                    occurrence_count=intent.occurrence_count,
                    cooldown_remaining_seconds=(intent.deferred_until - now).total_seconds(),
                    freshness_age_seconds=freshness_age,
                    first_seen_age_seconds=first_seen_age,
                )
                continue

            # Per-symbol cooldown check
            if self._store.is_symbol_cooled(intent.symbol, now=now):
                self._store.record_audit_log(
                    alert_intent_id=intent.alert_intent_id,
                    symbol=intent.symbol,
                    alert_type=intent.alert_type,
                    urgency=intent.urgency,
                    dispatch_status="skipped",
                    reason="cooldown_active",
                    configured_mode="dispatch",
                    dedupe_key=intent.dedupe_key,
                    trigger_price=intent.trigger_price,
                    occurrence_count=intent.occurrence_count,
                    freshness_age_seconds=freshness_age,
                    first_seen_age_seconds=first_seen_age,
                )
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
                    configured_mode="dispatch",
                    dedupe_key=intent.dedupe_key,
                    trigger_price=intent.trigger_price,
                    occurrence_count=intent.occurrence_count,
                    freshness_age_seconds=freshness_age,
                    first_seen_age_seconds=first_seen_age,
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

    def _is_deferred(self, intent: AlertIntent, now: datetime) -> bool:
        """Check if intent is deferred and unchanged.

        Returns True (skip) if:
        - deferred_until > now AND
        - occurrence_count has not incremented since deferral was set

        Returns False (re-evaluate) if:
        - deferred_until is None (not deferred)
        - deferred_until <= now (deferral expired)
        - occurrence_count > occurrence_count_at_deferral (material change)

        Requirements: 2.5
        """
        if intent.deferred_until is None:
            return False
        if intent.deferred_until <= now:
            return False  # Deferral expired — re-evaluate
        # Check if occurrence changed since deferral
        if intent.occurrence_count > intent.occurrence_count_at_deferral:
            return False  # Material change — re-evaluate despite deferral
        return True

    def _set_deferred(self, intent_id: int, cooldown_expiry: datetime, occurrence_count: int) -> None:
        """Set deferred_until and snapshot occurrence_count at deferral time.

        Requirements: 2.1
        """
        self._store.set_deferred_until(
            intent_id=intent_id,
            deferred_until=cooldown_expiry,
            occurrence_count_at_deferral=occurrence_count,
        )

    @staticmethod
    def _is_material_price_change(current_price, previous_price) -> bool:
        """Determine if a trigger_price change exceeds the 0.5% threshold.

        A price change is material if the absolute percentage difference
        between current and previous exceeds 0.5%. This prevents trivial
        price fluctuations from generating new observation records.

        If either price is zero or None, the change is considered material
        (fail-open: write the observation rather than silently suppress).

        Args:
            current_price: The current trigger_price (Decimal, str, or numeric).
            previous_price: The previously observed trigger_price (Decimal, str, or numeric).

        Returns:
            True if the price change exceeds 0.5%, False otherwise.

        Requirements: 8.3
        """
        from decimal import Decimal, InvalidOperation

        try:
            current = Decimal(str(current_price)) if current_price is not None else None
            previous = Decimal(str(previous_price)) if previous_price is not None else None
        except (InvalidOperation, ValueError, TypeError):
            return True  # Cannot compare — treat as material (fail-open)

        if current is None or previous is None:
            return True  # Missing price — treat as material (fail-open)

        if previous == 0:
            # Cannot compute percentage change from zero — treat as material
            return True

        pct_change = abs((current - previous) / previous) * 100
        return pct_change > Decimal("0.5")

    def _handle_observe(self, intent: AlertIntent, now: datetime) -> None:
        """Record would-dispatch for first observation, dedup subsequent unchanged.

        Material change detection (Requirement 8.3, 8.4):
        - dedupe_key refresh → new observation (handled by exact-match dedup)
        - trigger_price change >0.5% → new observation
        - occurrence_count increment → new observation (handled by exact-match dedup)
        - cooldown expiry → new observation (deferral system handles this)

        If the exact-match dedup passes (unchanged dedupe_key, trigger_price,
        occurrence_count), skip silently. If exact match fails (any field changed),
        apply the 0.5% threshold for trigger_price: if only the price changed and
        the change is <=0.5%, suppress the new observation.

        Requirements: 2.1, 2.3, 8.1, 8.2, 8.3, 8.4
        """
        from datetime import timedelta
        from utils.gate_config import PM_ALERT_SYMBOL_COOLDOWN_MINUTES

        # Check if we already observed this exact occurrence (exact match on all fields)
        if self._store.has_would_dispatch_for_occurrence(
            alert_intent_id=intent.alert_intent_id,
            dedupe_key=intent.dedupe_key,
            trigger_price=intent.trigger_price,
            occurrence_count=intent.occurrence_count,
        ):
            return  # Already observed this exact state — skip silently

        # Exact match failed — something changed. Apply material change threshold.
        # Get the latest observed trigger_price to check if the price change is material.
        last_observed_price = self._store.get_latest_would_dispatch_trigger_price(
            alert_intent_id=intent.alert_intent_id,
        )

        if last_observed_price is not None:
            # A prior observation exists. Check what changed:
            # - If dedupe_key or occurrence_count changed, that's always material
            #   (the exact-match already failed, so at least one field differs).
            #   We need to check if it's ONLY a price change within threshold.
            # Query whether a would_dispatch exists with same dedupe_key & occurrence_count
            # but different price. If so, only price changed — apply threshold.
            has_same_key_and_count = self._store.has_would_dispatch_for_occurrence(
                alert_intent_id=intent.alert_intent_id,
                dedupe_key=intent.dedupe_key,
                trigger_price=last_observed_price,  # Use the stored price for exact match
                occurrence_count=intent.occurrence_count,
            )
            if has_same_key_and_count:
                # Only price changed (dedupe_key and occurrence_count match prior observation)
                # Apply 0.5% threshold
                if not self._is_material_price_change(intent.trigger_price, last_observed_price):
                    return  # Price change <=0.5% — not material, skip silently

        # Material change confirmed (or first observation): record
        age_seconds = (now - intent.last_seen_at).total_seconds() if intent.last_seen_at else -1
        first_seen_age = (now - intent.first_seen_at).total_seconds() if intent.first_seen_at else -1
        cooldown_expiry = now + timedelta(minutes=PM_ALERT_SYMBOL_COOLDOWN_MINUTES)

        self._store.record_audit_log(
            alert_intent_id=intent.alert_intent_id,
            symbol=intent.symbol,
            alert_type=intent.alert_type,
            urgency=intent.urgency,
            dispatch_status="would_dispatch",
            reason="observe_mode",
            freshness_age_seconds=age_seconds,
            first_seen_age_seconds=first_seen_age,
            configured_mode="observe",
            dedupe_key=intent.dedupe_key,
            trigger_price=intent.trigger_price,
            occurrence_count=intent.occurrence_count,
        )
        self._set_deferred(intent.id, cooldown_expiry, intent.occurrence_count)

        logger.info(
            "ALERT_OBSERVE_WOULD_DISPATCH: symbol=%s alert_type=%s urgency=%s age_seconds=%.1f",
            intent.symbol, intent.alert_type, intent.urgency, age_seconds,
        )

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

    def _trigger_pm_cycle(self, dispatch_set: list[AlertIntent], symbols: list[str], intent_ids: list[int]) -> Optional[dict]:
        """Acquire PM mutex via injected callback and run narrow-scope PM.

        On PM success: mark intents consumed, record cooldowns, write dispatched audit records.
        On PM failure: revert intents to pending with incremented attempt count, write dispatch_failed audit records.
        Always release mutex in finally block.

        Requirements: 7.1, 7.2, 7.3
        """
        from datetime import timedelta
        from utils.gate_config import (
            PM_ALERT_SYMBOL_COOLDOWN_MINUTES,
            PM_ALERT_GLOBAL_COOLDOWN_MINUTES,
        )

        now = datetime.utcnow()
        batch_symbols_str = ",".join(symbols)

        # Acquire PM mutex
        if not self._begin_pm_cycle("alert_dispatcher"):
            # PM cycle already active — write audit records for each skipped intent
            for intent in dispatch_set:
                first_seen_age = (now - intent.first_seen_at).total_seconds() if intent.first_seen_at else -1
                freshness_age = (now - intent.last_seen_at).total_seconds() if intent.last_seen_at else -1
                self._store.record_audit_log(
                    alert_intent_id=intent.alert_intent_id,
                    symbol=intent.symbol,
                    alert_type=intent.alert_type,
                    urgency=intent.urgency,
                    dispatch_status="skipped",
                    reason="pm_cycle_active",
                    configured_mode="dispatch",
                    dedupe_key=intent.dedupe_key,
                    trigger_price=intent.trigger_price,
                    occurrence_count=intent.occurrence_count,
                    freshness_age_seconds=freshness_age,
                    first_seen_age_seconds=first_seen_age,
                )
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

            # Build structured alert_contexts for PM-side idempotency (Req 6.4, 10.2, 10.3)
            alert_contexts = [
                {
                    "alert_intent_id": intent.alert_intent_id,
                    "symbol": intent.symbol,
                    "alert_type": intent.alert_type,
                }
                for intent in dispatch_set
            ]

            # Run PM for each active profile
            from agents.portfolio_manager import run_profile
            from models.pm_profiles import ACTIVE_PROFILES

            for profile_id in ACTIVE_PROFILES:
                run_profile(
                    self._engine,
                    symbols,
                    profile_id,
                    tier="medium",
                    cycle_trigger_type="alert",
                    alert_contexts=alert_contexts,
                )

            # PM success: mark intents consumed
            self._store.mark_consumed(intent_ids)

            # Write audit record per intent with full schema (Requirement 7.2, 7.3)
            for intent in dispatch_set:
                first_seen_age = (now - intent.first_seen_at).total_seconds() if intent.first_seen_at else -1
                freshness_age = (now - intent.last_seen_at).total_seconds() if intent.last_seen_at else -1
                self._store.record_audit_log(
                    alert_intent_id=intent.alert_intent_id,
                    symbol=intent.symbol,
                    alert_type=intent.alert_type,
                    urgency=intent.urgency,
                    dispatch_status="dispatched",
                    reason="eligible",
                    configured_mode="dispatch",
                    cycle_trigger_type="alert",
                    dispatch_attempt_count=intent.dispatch_attempt_count + 1,
                    dispatch_batch_symbols=batch_symbols_str,
                    dedupe_key=intent.dedupe_key,
                    trigger_price=intent.trigger_price,
                    occurrence_count=intent.occurrence_count,
                    freshness_age_seconds=freshness_age,
                    first_seen_age_seconds=first_seen_age,
                )

            logger.info("ALERT_DISPATCH_SUCCESS: dispatched %d intents for %s", len(intent_ids), symbols)
            return {"dispatched": len(intent_ids), "symbols": symbols}

        except Exception as exc:
            # PM failed: revert intents, write dispatch_failed audit records
            logger.error("ALERT_DISPATCH_FAILED: %s", str(exc))
            self._store.mark_dispatch_failed(intent_ids, str(exc))

            # Write audit record per intent for failure
            for intent in dispatch_set:
                first_seen_age = (now - intent.first_seen_at).total_seconds() if intent.first_seen_at else -1
                freshness_age = (now - intent.last_seen_at).total_seconds() if intent.last_seen_at else -1
                self._store.record_audit_log(
                    alert_intent_id=intent.alert_intent_id,
                    symbol=intent.symbol,
                    alert_type=intent.alert_type,
                    urgency=intent.urgency,
                    dispatch_status="dispatch_failed",
                    reason=str(exc)[:200],
                    configured_mode="dispatch",
                    cycle_trigger_type="alert",
                    dispatch_attempt_count=intent.dispatch_attempt_count + 1,
                    dispatch_batch_symbols=batch_symbols_str,
                    dedupe_key=intent.dedupe_key,
                    trigger_price=intent.trigger_price,
                    occurrence_count=intent.occurrence_count,
                    freshness_age_seconds=freshness_age,
                    first_seen_age_seconds=first_seen_age,
                )

            return None
        finally:
            self._end_pm_cycle("alert_dispatcher")

    def _check_freshness(self, intent: AlertIntent, now: datetime) -> bool:
        """Return True if intent is fresh enough for dispatch consideration.

        Dispatch freshness uses last_seen_at (most recent observation time).
        This reflects whether the market condition is still being observed.

        Returns False (stale / excluded) if:
        - alert_type is "target_hit" (excluded from dispatch entirely)
        - alert_type is not recognized (unknown type → stale)
        - last_seen_at is None (fail-closed: can't determine freshness)
        - age (now - last_seen_at) exceeds configured freshness limit

        Requirements: 3.1, 3.2, 3.6
        """
        from utils.gate_config import (
            PM_ALERT_FRESHNESS_ENTRY_ALERT_MINUTES,
            PM_ALERT_FRESHNESS_BREAKOUT_MINUTES,
            PM_ALERT_FRESHNESS_RAPID_MOVE_MINUTES,
        )

        freshness_limits = {
            "entry_alert": PM_ALERT_FRESHNESS_ENTRY_ALERT_MINUTES,
            "breakout": PM_ALERT_FRESHNESS_BREAKOUT_MINUTES,
            "rapid_move": PM_ALERT_FRESHNESS_RAPID_MOVE_MINUTES,
            # target_hit excluded from dispatch entirely — no entry here
        }

        limit_minutes = freshness_limits.get(intent.alert_type)
        if limit_minutes is None:
            return False  # Unknown alert type or target_hit → stale

        if intent.last_seen_at is None:
            return False  # Cannot determine freshness → stale (fail-closed)

        age = now - intent.last_seen_at
        return age.total_seconds() < (limit_minutes * 60)

    def consume_for_scheduled_cycle(self) -> tuple[list[str], list[int]]:
        """Called by scheduled PM to claim pending intents.

        Returns (extra_symbols, claimed_intent_ids).
        Marks intents as 'claimed_by_scheduled'.
        Caller must call confirm_scheduled_consumption(ids) on PM success
        or revert_scheduled_claim(ids, error) on PM failure.
        """
        from utils.gate_config import PM_ALERT_DISPATCH_MODE

        if PM_ALERT_DISPATCH_MODE != "enabled":
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
