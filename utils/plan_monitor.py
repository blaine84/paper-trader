"""Plan Monitor — real-time trigger evaluation for trade plans.

Runs on a configurable cadence (default 30s) independent of PM cycles.
Evaluates active trade plans against current prices — deterministic,
no LLM calls. Transitions plans to TRIGGERED state when conditions
are met and hands off to plan_executor for execution.

Quote fetching is cache-first with rate-limited provider fallback.
Plans never transition to MISSED due to transient provider issues.

See: design.md §utils/plan_monitor.py
Requirements: 3.1–3.7, 4.1–4.14
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

from utils.entry_zone import is_price_in_zone, is_price_past_target, EntryZone
from utils.gate_config import (
    TRIGGERED_PLAN_MODE,
    PLAN_ENTRY_ZONE_TOLERANCE_PCT,
    PLAN_TRIGGER_CONFIRMATION_TICKS,
    PLAN_TRIGGER_QUOTE_MAX_AGE_SECONDS,
    QUOTE_PROVIDER_MIN_SECONDS_PER_SYMBOL,
    QUOTE_PROVIDER_MAX_CALLS_PER_MINUTE,
)
from utils.trade_plan_registry import (
    TradePlanRegistry,
    TradePlan,
    TradePlanRegistryError,
    PlanState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriggerEvaluation:
    """Result of evaluating a plan's trigger conditions."""

    triggered: bool
    reason: str
    fresh_price: float
    evaluation_timestamp: datetime
    missed: bool = False
    miss_reason: str | None = None
    invalidated: bool = False
    invalidation_reason: str | None = None


@dataclass
class MonitorTickResult:
    """Summary of a single plan monitor tick."""

    plans_checked: int = 0
    plans_triggered: int = 0
    plans_expired: int = 0
    plans_invalidated: int = 0
    plans_missed: int = 0
    quotes_fetched: int = 0
    quotes_from_cache: int = 0
    provider_calls_made: int = 0
    tick_duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Module-level quote rate state (persists across PlanMonitor instances)
# ---------------------------------------------------------------------------


@dataclass
class _QuoteRateState:
    """Process-level quote rate limiter state.

    Lives at module scope so it survives PlanMonitor re-instantiation
    per tick. Reset only on process restart.
    """

    symbol_last_fetched: dict[str, float] = field(default_factory=dict)
    calls_this_window: list[float] = field(default_factory=list)

    def can_fetch_symbol(self, symbol: str, now: float, min_interval: int) -> bool:
        """True if per-symbol interval has elapsed."""
        last = self.symbol_last_fetched.get(symbol, 0.0)
        return (now - last) >= min_interval

    def can_fetch_global(self, now: float, max_per_minute: int) -> bool:
        """True if rolling 60s call count is under budget."""
        self.calls_this_window = [t for t in self.calls_this_window if now - t < 60.0]
        return len(self.calls_this_window) < max_per_minute

    def record_fetch(self, symbol: str, now: float) -> None:
        """Record a provider call for rate tracking."""
        self.symbol_last_fetched[symbol] = now
        self.calls_this_window.append(now)


_quote_rate_state = _QuoteRateState()


# ---------------------------------------------------------------------------
# Quote provider protocol
# ---------------------------------------------------------------------------


class QuoteResult:
    """Simple container for a quote result used by the monitor."""

    def __init__(self, symbol: str, price: float, timestamp: float):
        self.symbol = symbol
        self.price = price
        self.timestamp = timestamp  # time.time() when fetched/cached


# ---------------------------------------------------------------------------
# PlanMonitor class
# ---------------------------------------------------------------------------


class PlanMonitor:
    """Evaluates active trade plans against current prices.

    Runs on a fast cadence (default 30s) independent of PM cycles.
    Deterministic — no LLM calls. Quote fetching is rate-limited.

    Rate state is MODULE-LEVEL (not per-instance) because PlanMonitor is
    recreated each tick.
    """

    def __init__(self, engine) -> None:
        self._engine = engine
        self._registry = TradePlanRegistry(engine)
        # Confirmation ticks tracked per plan across the monitor lifecycle.
        # Since PlanMonitor is re-created per tick, we use module-level storage.
        # (see _confirmation_ticks module-level dict)

    def run(self) -> MonitorTickResult:
        """Execute one monitor tick.

        Steps:
        1. Load active plans (PLANNED, WATCHING)
        2. Transition PLANNED → WATCHING on first pickup
        3. Check expiration for all plans
        4. Collect unique symbols across all active plans
        5. Fetch quotes: cache-first, rate-limited provider fallback
        6. Evaluate trigger conditions using available quotes
        7. Handle triggered/missed/invalidated plans
        8. Return tick summary
        """
        tick_start = time.time()
        result = MonitorTickResult()

        # Load active plans
        active_plans = self._registry.get_active_plans()
        if not active_plans:
            result.tick_duration_ms = (time.time() - tick_start) * 1000
            return result

        # Step 1: Activate PLANNED → WATCHING
        watching_plans: list[TradePlan] = []
        now = datetime.now(timezone.utc)

        for plan in active_plans:
            if plan.state == PlanState.PLANNED:
                try:
                    self._registry.activate(plan.plan_id)
                    # Update local reference to reflect new state
                    watching_plans.append(plan)
                except TradePlanRegistryError:
                    logger.warning(
                        "Could not activate plan %s (may already be activated)",
                        plan.plan_id,
                    )
            else:
                watching_plans.append(plan)

        # Step 2: Check expiration — expired plans don't need quote evaluation
        plans_to_evaluate: list[TradePlan] = []
        for plan in watching_plans:
            if plan.expires_at <= now:
                try:
                    self._registry.mark_expired(plan.plan_id)
                    result.plans_expired += 1
                    logger.info("Plan %s expired (symbol=%s)", plan.plan_id, plan.symbol)
                    _record_missed_setup_event(
                        self._engine, plan, 0.0,
                        now,
                        0.0,
                        "plan_expired",
                    )
                except TradePlanRegistryError:
                    logger.warning(
                        "Could not expire plan %s (may have already transitioned)",
                        plan.plan_id,
                    )
            else:
                plans_to_evaluate.append(plan)

        result.plans_checked = len(plans_to_evaluate)

        if not plans_to_evaluate:
            result.tick_duration_ms = (time.time() - tick_start) * 1000
            return result

        # Step 3: Collect unique symbols
        symbols = list({p.symbol for p in plans_to_evaluate})

        # Step 4: Fetch quotes (cache-first, rate-limited)
        quotes = self._get_rate_limited_quotes(symbols)
        result.quotes_fetched = quotes.get("_meta_provider_calls", 0)
        result.quotes_from_cache = quotes.get("_meta_cache_hits", 0)
        result.provider_calls_made = quotes.get("_meta_provider_calls", 0)

        # Remove meta keys
        quotes.pop("_meta_provider_calls", None)
        quotes.pop("_meta_cache_hits", None)

        # Step 5: Evaluate trigger conditions for each plan
        for plan in plans_to_evaluate:
            price = quotes.get(plan.symbol)
            if price is None:
                # No quote available — plan stays WATCHING (not MISSED)
                logger.debug(
                    "No quote available for %s — plan %s stays WATCHING",
                    plan.symbol, plan.plan_id,
                )
                continue

            evaluation = self._evaluate_trigger(plan, price)

            if evaluation.missed:
                try:
                    self._registry.mark_missed(
                        plan.plan_id,
                        evaluation.miss_reason or "price_past_target",
                        evaluation.fresh_price,
                    )
                    result.plans_missed += 1
                    _record_missed_setup_event(
                        self._engine, plan, evaluation.fresh_price,
                        evaluation.evaluation_timestamp,
                        0.0,  # quote_age not tracked for trigger eval
                        evaluation.miss_reason or "price_past_target",
                    )
                except TradePlanRegistryError:
                    logger.warning(
                        "Could not mark plan %s as missed", plan.plan_id
                    )
            elif evaluation.invalidated:
                try:
                    self._registry.mark_rejected(
                        plan.plan_id,
                        evaluation.invalidation_reason or "invalidation_triggered",
                    )
                    result.plans_invalidated += 1
                    _record_missed_setup_event(
                        self._engine, plan, evaluation.fresh_price,
                        evaluation.evaluation_timestamp,
                        0.0,  # quote_age not tracked for trigger eval
                        "invalidation_triggered",
                    )
                except TradePlanRegistryError:
                    logger.warning(
                        "Could not mark plan %s as rejected", plan.plan_id
                    )
            elif evaluation.triggered:
                try:
                    self._registry.trigger(plan.plan_id)
                    result.plans_triggered += 1
                    logger.info(
                        "Plan %s TRIGGERED (symbol=%s, price=%.2f, reason=%s)",
                        plan.plan_id, plan.symbol,
                        evaluation.fresh_price, evaluation.reason,
                    )
                except TradePlanRegistryError:
                    logger.warning(
                        "Could not trigger plan %s (may have already transitioned)",
                        plan.plan_id,
                    )

        result.tick_duration_ms = (time.time() - tick_start) * 1000
        logger.info(
            "Plan monitor tick: checked=%d, triggered=%d, expired=%d, "
            "invalidated=%d, missed=%d, provider_calls=%d, cache_hits=%d, "
            "duration_ms=%.1f",
            result.plans_checked, result.plans_triggered, result.plans_expired,
            result.plans_invalidated, result.plans_missed,
            result.provider_calls_made, result.quotes_from_cache,
            result.tick_duration_ms,
        )
        return result

    def _evaluate_trigger(
        self, plan: TradePlan, fresh_price: float
    ) -> TriggerEvaluation:
        """Evaluate whether a plan's trigger conditions are met.

        Checks in order:
        1. Has the price already moved past the target? → missed
        2. Is the invalidation condition met? → invalidated
        3. Is the price inside the entry zone? → potential trigger
        4. Is additional confirmation required? → check tick count
        5. Final trigger decision
        """
        now = datetime.now(timezone.utc)
        price_decimal = Decimal(str(fresh_price))
        target_decimal = Decimal(str(plan.target_price))

        # 1. Check price past target → missed
        if is_price_past_target(price_decimal, target_decimal, plan.direction):
            return TriggerEvaluation(
                triggered=False,
                reason="price_past_target",
                fresh_price=fresh_price,
                evaluation_timestamp=now,
                missed=True,
                miss_reason="price_past_target",
            )

        # 2. Check invalidation
        invalidated, invalidation_reason = self._check_invalidation(plan, fresh_price)
        if invalidated:
            return TriggerEvaluation(
                triggered=False,
                reason="invalidation_triggered",
                fresh_price=fresh_price,
                evaluation_timestamp=now,
                invalidated=True,
                invalidation_reason=invalidation_reason,
            )

        # 3. Check price in zone
        zone = EntryZone(
            upper=Decimal(str(plan.entry_zone_upper)),
            lower=Decimal(str(plan.entry_zone_lower)),
            reference=Decimal(str(plan.entry_reference)),
            tolerance_pct=Decimal(str(PLAN_ENTRY_ZONE_TOLERANCE_PCT)),
        )
        in_zone = is_price_in_zone(price_decimal, zone, plan.direction)

        if not in_zone:
            # Price is outside zone — reset confirmation counter
            _confirmation_ticks.pop(plan.plan_id, None)
            return TriggerEvaluation(
                triggered=False,
                reason="price_outside_zone",
                fresh_price=fresh_price,
                evaluation_timestamp=now,
            )

        # 4. Check confirmation if required
        if plan.trigger_confirmation_required:
            current_count = _confirmation_ticks.get(plan.plan_id, 0) + 1
            _confirmation_ticks[plan.plan_id] = current_count

            required = PLAN_TRIGGER_CONFIRMATION_TICKS
            if current_count < required:
                return TriggerEvaluation(
                    triggered=False,
                    reason=f"confirmation_pending ({current_count}/{required})",
                    fresh_price=fresh_price,
                    evaluation_timestamp=now,
                )
        else:
            # No confirmation needed — increment for tracking but trigger immediately
            _confirmation_ticks[plan.plan_id] = _confirmation_ticks.get(plan.plan_id, 0) + 1

        # 5. Trigger!
        # Clean up confirmation tracking
        _confirmation_ticks.pop(plan.plan_id, None)
        return TriggerEvaluation(
            triggered=True,
            reason="price_in_zone",
            fresh_price=fresh_price,
            evaluation_timestamp=now,
        )

    def _check_invalidation(
        self, plan: TradePlan, price: float
    ) -> tuple[bool, str | None]:
        """Evaluate structured invalidation logic against current price.

        The invalidation_logic_json field contains structured conditions.
        Currently supports:
        - {"type": "price_below", "level": <float>} — for longs
        - {"type": "price_above", "level": <float>} — for shorts
        - None or empty — no invalidation logic configured
        """
        if not plan.invalidation_logic_json:
            return False, None

        try:
            logic = json.loads(plan.invalidation_logic_json)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "Plan %s has unparseable invalidation_logic_json: %s",
                plan.plan_id, plan.invalidation_logic_json,
            )
            return False, None

        if not isinstance(logic, dict):
            return False, None

        inv_type = logic.get("type")
        level = logic.get("level")

        if level is None:
            return False, None

        try:
            level_f = float(level)
        except (ValueError, TypeError):
            return False, None

        if inv_type == "price_below" and price < level_f:
            return True, f"price {price} below invalidation level {level_f}"
        elif inv_type == "price_above" and price > level_f:
            return True, f"price {price} above invalidation level {level_f}"

        return False, None

    def _get_rate_limited_quotes(self, symbols: list[str]) -> dict:
        """Fetch quotes for symbols: cache-first with rate-limited provider fallback.

        Uses the shared quote cache from price_monitor (agents.price_monitor._quote_cache).
        If cache is stale (> PLAN_TRIGGER_QUOTE_MAX_AGE_SECONDS), attempts provider
        call if rate budget permits. On 429 or circuit-break, uses stale cache.

        Returns dict of {symbol: price} plus metadata keys for reporting.
        """
        from agents.price_monitor import _quote_cache, get_batch_quotes

        now = time.time()
        quotes: dict[str, float] = {}
        provider_calls = 0
        cache_hits = 0
        circuit_broken = False

        for symbol in symbols:
            # Check cache first
            cached = _quote_cache.get(symbol)
            if cached:
                cache_ts, cache_price = cached
                cache_age = now - cache_ts
                if cache_age <= PLAN_TRIGGER_QUOTE_MAX_AGE_SECONDS:
                    quotes[symbol] = cache_price
                    cache_hits += 1
                    continue

            # Cache is stale or missing — check rate limits before provider call
            if circuit_broken:
                # Provider is circuit-broken for this tick — use stale cache if available
                if cached:
                    quotes[symbol] = cached[1]
                    cache_hits += 1
                continue

            if not _quote_rate_state.can_fetch_symbol(
                symbol, now, QUOTE_PROVIDER_MIN_SECONDS_PER_SYMBOL
            ):
                # Per-symbol rate limit — use stale cache
                if cached:
                    quotes[symbol] = cached[1]
                    cache_hits += 1
                continue

            if not _quote_rate_state.can_fetch_global(
                now, QUOTE_PROVIDER_MAX_CALLS_PER_MINUTE
            ):
                # Global budget exhausted — use stale cache for rest of tick
                if cached:
                    quotes[symbol] = cached[1]
                    cache_hits += 1
                continue

            # Budget available — fetch from provider
            try:
                batch = get_batch_quotes([symbol], prefer_finnhub=False)
                price = batch.get(symbol)
                if price and price > 0:
                    quotes[symbol] = price
                    _quote_rate_state.record_fetch(symbol, now)
                    provider_calls += 1
                elif cached:
                    # Provider returned no price — use stale cache
                    quotes[symbol] = cached[1]
                    cache_hits += 1
            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    # Circuit-break: stop provider calls for rest of tick
                    logger.warning(
                        "Provider 429 for %s — circuit-breaking for this tick", symbol
                    )
                    circuit_broken = True
                    if cached:
                        quotes[symbol] = cached[1]
                        cache_hits += 1
                else:
                    logger.warning(
                        "Quote fetch failed for %s: %s — using stale cache", symbol, e
                    )
                    if cached:
                        quotes[symbol] = cached[1]
                        cache_hits += 1

        # Attach metadata for reporting
        quotes["_meta_provider_calls"] = provider_calls
        quotes["_meta_cache_hits"] = cache_hits
        return quotes


# ---------------------------------------------------------------------------
# Module-level confirmation tracking (persists across PlanMonitor instances)
# ---------------------------------------------------------------------------

_confirmation_ticks: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Module-level entry point for APScheduler
# ---------------------------------------------------------------------------


def run(engine) -> MonitorTickResult:
    """Entry point called by APScheduler every PLAN_MONITOR_INTERVAL_SECONDS.

    Returns immediately with zeros when feature is disabled.
    """
    if TRIGGERED_PLAN_MODE == "disabled":
        return MonitorTickResult()

    monitor = PlanMonitor(engine)
    return monitor.run()


# ---------------------------------------------------------------------------
# Missed setup event recording
# ---------------------------------------------------------------------------


def _record_missed_setup_event(
    engine,
    plan: TradePlan,
    fresh_price: float,
    evaluation_timestamp: datetime,
    quote_age_seconds: float,
    reason: str,
) -> None:
    """Record a missed_setup trade event for audit/review.

    Fail-open: logs error but does not raise.
    """
    from utils.trade_events import log_trade_event
    from db.schema import get_session

    db = get_session(engine)
    try:
        log_trade_event(
            db,
            "missed_setup",
            agent="plan_monitor",
            symbol=plan.symbol,
            profile=plan.profile_id,
            message=(
                f"Plan {plan.plan_id} missed: {reason}. "
                f"Entry zone [{plan.entry_zone_lower}-{plan.entry_zone_upper}], "
                f"fresh price={fresh_price}, target={plan.target_price}"
            ),
            payload={
                "plan_id": plan.plan_id,
                "candidate_id": plan.candidate_id,
                "cycle_id": plan.cycle_id,
                "symbol": plan.symbol,
                "direction": plan.direction,
                "setup_type": plan.setup_type,
                "geometry_name": plan.geometry_name,
                "profile_id": plan.profile_id,
                "entry_reference": plan.entry_reference,
                "entry_zone_upper": plan.entry_zone_upper,
                "entry_zone_lower": plan.entry_zone_lower,
                "fresh_price_at_miss": fresh_price,
                "quote_timestamp": evaluation_timestamp.isoformat() if evaluation_timestamp else None,
                "quote_age_seconds": round(quote_age_seconds, 1),
                "intended_target": plan.target_price,
                "original_stop": plan.stop_price,
                "original_risk_reward": plan.risk_reward,
                "reason_for_miss": reason,
            },
        )
        db.commit()
    except Exception as e:
        logger.error(
            "Failed to record missed_setup event for plan %s: %s", plan.plan_id, e
        )
    finally:
        db.close()
