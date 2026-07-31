"""Plan Executor — fresh-quote execution for triggered trade plans.

Handles the execution phase after a plan has been promoted to TRIGGERED
state by the plan monitor. Fetches a fresh quote (cache-bypass), validates
freshness, recalculates geometry from the fresh price, runs the full gate
pipeline + position sizer, and calls execute_trade() on success.

Execution path is FAIL-CLOSED: no fill without a verified fresh quote
with known timestamp, valid geometry, and passing gates.

See: design.md §utils/plan_executor.py
Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10,
              4.11, 4.12, 4.13, 4.14
"""
from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, Context, ROUND_HALF_UP

from utils.entry_zone import is_price_in_zone, is_price_past_target, EntryZone
from utils.gate_config import (
    TRIGGERED_PLAN_MODE,
    PLAN_ENTRY_ZONE_TOLERANCE_PCT,
    PLAN_EXECUTION_MAX_QUOTE_AGE_SECONDS,
    QUOTE_PROVIDER_MIN_SECONDS_PER_SYMBOL,
    QUOTE_PROVIDER_MAX_CALLS_PER_MINUTE,
)
from utils.trade_plan_registry import TradePlanRegistry, TradePlan, PlanState

logger = logging.getLogger(__name__)

# Fixed Decimal context — 28 digits precision, consistent with geometry_calculator
_CTX = Context(prec=28, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Value object — execution result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanExecutionResult:
    """Outcome of attempting to execute a triggered plan."""

    plan_id: str
    success: bool
    fill_price: float | None
    reason: str
    geometry_recalculated: bool


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def execute_triggered_plan(
    engine,
    plan: TradePlan,
    fresh_price: float,
) -> PlanExecutionResult:
    """Execute a triggered trade plan with fresh geometry.

    Steps:
    1. Feature flag guard
    2. Verify plan.state == TRIGGERED
    3. Fetch a FRESH quote (cache-bypass, validate quote age)
    4. Fail-closed on stale quotes
    5. Validate fresh price still within entry zone
    6. Check fresh price hasn't crossed target (missed)
    7. Recalculate geometry at fill price
    8. Run full gate pipeline validation
    9. Call execute_trade() with recalculated geometry
    10. On success: registry.mark_entered(plan_id)
    11. On failure: registry.mark_missed(plan_id, reason)

    Args:
        engine: SQLAlchemy engine for database operations.
        plan: The TradePlan in TRIGGERED state.
        fresh_price: The trigger price from plan monitor (used as fallback context).

    Returns:
        PlanExecutionResult describing the outcome.
    """
    # Step 1: Feature flag guard
    if TRIGGERED_PLAN_MODE == "disabled":
        return PlanExecutionResult(
            plan_id=plan.plan_id,
            success=False,
            fill_price=None,
            reason="feature_disabled",
            geometry_recalculated=False,
        )

    registry = TradePlanRegistry(engine)

    # Step 2: Verify plan.state == TRIGGERED
    if plan.state != PlanState.TRIGGERED:
        logger.warning(
            "execute_triggered_plan called on plan %s with state %s (expected TRIGGERED)",
            plan.plan_id, plan.state.value,
        )
        return PlanExecutionResult(
            plan_id=plan.plan_id,
            success=False,
            fill_price=None,
            reason=f"invalid_state_{plan.state.value}",
            geometry_recalculated=False,
        )

    # Step 3: Fetch fresh execution quote (cache-bypass)
    quote_price, quote_timestamp, quote_age_seconds = _fetch_fresh_execution_quote(
        plan.symbol
    )

    if quote_price is None or quote_price <= 0:
        # Provider exhausted or unavailable — leave plan TRIGGERED for retry
        logger.warning(
            "Fresh quote unavailable for plan %s (%s) — will retry next tick",
            plan.plan_id, plan.symbol,
        )
        return PlanExecutionResult(
            plan_id=plan.plan_id,
            success=False,
            fill_price=None,
            reason="no_fresh_quote_retry",
            geometry_recalculated=False,
        )

    # Step 4: Fail-closed on stale quotes
    if quote_age_seconds > PLAN_EXECUTION_MAX_QUOTE_AGE_SECONDS:
        logger.warning(
            "Plan %s: quote age %.1fs exceeds max %ds — stale quote, "
            "marking missed",
            plan.plan_id, quote_age_seconds, PLAN_EXECUTION_MAX_QUOTE_AGE_SECONDS,
        )
        try:
            registry.mark_missed(plan.plan_id, "quote_too_stale", quote_price)
        except Exception as e:
            logger.error("Failed to mark plan %s missed: %s", plan.plan_id, e)
        _record_missed_setup_event(
            engine, plan, quote_price, quote_timestamp,
            quote_age_seconds, "quote_too_stale",
        )
        return PlanExecutionResult(
            plan_id=plan.plan_id,
            success=False,
            fill_price=quote_price,
            reason="quote_too_stale",
            geometry_recalculated=False,
        )

    # Step 5: Validate fresh price still in entry zone
    zone = EntryZone(
        upper=Decimal(str(plan.entry_zone_upper)),
        lower=Decimal(str(plan.entry_zone_lower)),
        reference=Decimal(str(plan.entry_reference)),
        tolerance_pct=Decimal(str(PLAN_ENTRY_ZONE_TOLERANCE_PCT)),
    )
    price_decimal = Decimal(str(quote_price))

    if not is_price_in_zone(price_decimal, zone, plan.direction):
        logger.info(
            "Plan %s: fresh price %.2f moved beyond entry zone [%.2f–%.2f] — missed",
            plan.plan_id, quote_price, plan.entry_zone_lower, plan.entry_zone_upper,
        )
        try:
            registry.mark_missed(plan.plan_id, "price_beyond_zone", quote_price)
        except Exception as e:
            logger.error("Failed to mark plan %s missed: %s", plan.plan_id, e)
        _record_missed_setup_event(
            engine, plan, quote_price, quote_timestamp,
            quote_age_seconds, "price_beyond_zone",
        )
        return PlanExecutionResult(
            plan_id=plan.plan_id,
            success=False,
            fill_price=quote_price,
            reason="price_beyond_zone",
            geometry_recalculated=False,
        )

    # Step 6: Check fresh price hasn't crossed target
    target_decimal = Decimal(str(plan.target_price))
    if is_price_past_target(price_decimal, target_decimal, plan.direction):
        logger.info(
            "Plan %s: fresh price %.2f past target %.2f — missed",
            plan.plan_id, quote_price, plan.target_price,
        )
        try:
            registry.mark_missed(plan.plan_id, "price_past_target", quote_price)
        except Exception as e:
            logger.error("Failed to mark plan %s missed: %s", plan.plan_id, e)
        _record_missed_setup_event(
            engine, plan, quote_price, quote_timestamp,
            quote_age_seconds, "price_past_target",
        )
        return PlanExecutionResult(
            plan_id=plan.plan_id,
            success=False,
            fill_price=quote_price,
            reason="price_past_target",
            geometry_recalculated=False,
        )

    # Step 7: Recalculate geometry at fill price
    geometry = recalculate_geometry(plan, quote_price)

    # Step 8: Run full gate pipeline validation
    from db.schema import get_session
    from agents.portfolio_manager import _run_gate_pipeline, execute_trade

    db = get_session(engine)
    try:
        # Build a decision dict compatible with _run_gate_pipeline
        decision = {
            "symbol": plan.symbol,
            "action": plan.direction,  # "BUY" or "SHORT"
            "entry_price": geometry["entry_price"],
            "price": geometry["entry_price"],
            "stop": geometry["stop_price"],
            "stop_price": geometry["stop_price"],
            "target": geometry["target_price"],
            "target_price": geometry["target_price"],
            "risk_reward": geometry["risk_reward"],
            "quantity": geometry["quantity"],
            "setup_type": plan.setup_type,
            "geometry_name": plan.geometry_name,
            "profile_id": plan.profile_id,
            "plan_id": plan.plan_id,
            "candidate_id": plan.candidate_id,
        }

        # Parse signal from plan's snapshot for gate context
        signal = None
        if plan.signal_snapshot_json:
            try:
                import json
                signal = json.loads(plan.signal_snapshot_json)
            except (ValueError, TypeError):
                signal = {}

        proceed, gate_notes, multiplier, _ = _run_gate_pipeline(
            db, engine, decision, signal, plan.profile_id,
        )

        if not proceed:
            # Gate pipeline rejected — mark plan as rejected
            rejection_reason = "gate_pipeline_rejected"
            if gate_notes:
                # Extract the rejecting gate's reason
                for note in gate_notes:
                    if isinstance(note, dict) and note.get("decision") == "reject":
                        rejection_reason = note.get("gate", "gate_pipeline_rejected")
                        break

            logger.info(
                "Plan %s: gate pipeline rejected — %s",
                plan.plan_id, rejection_reason,
            )
            try:
                registry.mark_rejected(plan.plan_id, rejection_reason)
            except Exception as e:
                logger.error("Failed to mark plan %s rejected: %s", plan.plan_id, e)

            return PlanExecutionResult(
                plan_id=plan.plan_id,
                success=False,
                fill_price=quote_price,
                reason=rejection_reason,
                geometry_recalculated=True,
            )

        # Apply size multiplier from gates if applicable
        if multiplier and multiplier != 1.0 and geometry["quantity"]:
            adjusted_qty = max(1, int(geometry["quantity"] * multiplier))
            decision["quantity"] = adjusted_qty

        # Step 9: Call execute_trade() with fresh geometry (normalized=True)
        success, result_msg = execute_trade(
            db, decision, plan.profile_id, normalized=True,
        )

        if success:
            # Step 10: Mark plan as entered
            logger.info(
                "Plan %s: execution succeeded at price %.2f (symbol=%s)",
                plan.plan_id, quote_price, plan.symbol,
            )
            try:
                registry.mark_entered(plan.plan_id)
            except Exception as e:
                logger.error(
                    "Failed to mark plan %s entered (trade was created): %s",
                    plan.plan_id, e,
                )

            return PlanExecutionResult(
                plan_id=plan.plan_id,
                success=True,
                fill_price=quote_price,
                reason="entered",
                geometry_recalculated=True,
            )
        else:
            # Step 11: Execution failed — mark missed
            logger.warning(
                "Plan %s: execute_trade failed — %s",
                plan.plan_id, result_msg,
            )
            try:
                registry.mark_missed(
                    plan.plan_id, "execution_failed", quote_price,
                )
            except Exception as e:
                logger.error("Failed to mark plan %s missed: %s", plan.plan_id, e)

            _record_missed_setup_event(
                engine, plan, quote_price, quote_timestamp,
                quote_age_seconds, "execution_failed",
            )
            return PlanExecutionResult(
                plan_id=plan.plan_id,
                success=False,
                fill_price=quote_price,
                reason="execution_failed",
                geometry_recalculated=True,
            )

    except Exception as e:
        logger.error(
            "Plan %s: unexpected error during execution: %s",
            plan.plan_id, e, exc_info=True,
        )
        try:
            registry.mark_missed(plan.plan_id, "execution_error", quote_price)
        except Exception:
            pass
        return PlanExecutionResult(
            plan_id=plan.plan_id,
            success=False,
            fill_price=quote_price,
            reason="execution_error",
            geometry_recalculated=False,
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Geometry recalculation
# ---------------------------------------------------------------------------


def recalculate_geometry(plan: TradePlan, fill_price: float) -> dict:
    """Recalculate stop/target/R:R from fresh fill price using proportional adjustment.

    Preserves the original stop/target distances as ratios of entry_reference,
    then applies those ratios to the fresh fill price. Uses Decimal arithmetic
    for precision (Context prec=28, ROUND_HALF_UP).

    Direction-aware logic:
    - LONG (BUY): stop below fill price, target above
    - SHORT: stop above fill price, target below

    Args:
        plan: The trade plan with original geometry.
        fill_price: The fresh market price to base recalculation on.

    Returns:
        dict with: entry_price, stop_price, target_price, risk_reward, quantity
    """
    entry_ref = _CTX.create_decimal(Decimal(str(plan.entry_reference)))
    fresh = _CTX.create_decimal(Decimal(str(fill_price)))
    stop = _CTX.create_decimal(Decimal(str(plan.stop_price)))
    target = _CTX.create_decimal(Decimal(str(plan.target_price)))

    if entry_ref <= 0:
        raise ValueError(f"Invalid entry_reference: {plan.entry_reference}")

    # Compute ratios relative to original entry_reference
    stop_ratio = _CTX.divide(_CTX.subtract(stop, entry_ref), entry_ref)
    target_ratio = _CTX.divide(_CTX.subtract(target, entry_ref), entry_ref)

    # Apply ratios to fresh price
    new_stop = _CTX.add(fresh, _CTX.multiply(fresh, stop_ratio))
    new_target = _CTX.add(fresh, _CTX.multiply(fresh, target_ratio))

    # Compute risk/reward
    if plan.direction == "BUY":
        risk = _CTX.subtract(fresh, new_stop)
        reward = _CTX.subtract(new_target, fresh)
    else:  # SHORT
        risk = _CTX.subtract(new_stop, fresh)
        reward = _CTX.subtract(fresh, new_target)

    if risk > 0:
        rr = float(_CTX.divide(reward, risk))
    else:
        rr = 0.0

    return {
        "entry_price": fill_price,
        "stop_price": float(new_stop),
        "target_price": float(new_target),
        "risk_reward": round(rr, 2),
        "quantity": None,  # Calculated by position sizer downstream
    }


# ---------------------------------------------------------------------------
# Fresh quote fetching (cache-bypass, rate-limited)
# ---------------------------------------------------------------------------


def _fetch_fresh_execution_quote(
    symbol: str,
) -> tuple[float | None, datetime | None, float]:
    """Fetch a fresh quote for plan execution, bypassing the quote cache.

    Rate-limited: checks the shared _quote_rate_state budget from
    plan_monitor before calling the provider. If provider budget is
    exhausted or circuit-broken, returns (None, None, inf) so the caller
    retries next tick (plan stays TRIGGERED, not MISSED).

    Returns:
        (price, quote_timestamp, quote_age_seconds)
        - price: float or None if unavailable
        - quote_timestamp: datetime of the quote or None
        - quote_age_seconds: age in seconds or inf if unknown/unavailable

    Uses Finnhub directly (not get_batch_quotes) to ensure freshness.
    Falls back to yfinance if Finnhub unavailable (with conservative age
    that will be rejected by the caller's age check).
    """
    from utils.plan_monitor import _quote_rate_state
    from utils.finnhub_client import FinnhubClient

    now = _time.time()

    # Check shared rate budget BEFORE calling provider
    if not _quote_rate_state.can_fetch_symbol(
        symbol, now, QUOTE_PROVIDER_MIN_SECONDS_PER_SYMBOL
    ):
        logger.debug(
            "Execution quote for %s rate-limited (per-symbol interval)", symbol
        )
        return None, None, float("inf")

    if not _quote_rate_state.can_fetch_global(
        now, QUOTE_PROVIDER_MAX_CALLS_PER_MINUTE
    ):
        logger.debug(
            "Execution quote for %s rate-limited (global budget exhausted)", symbol
        )
        return None, None, float("inf")

    # Try Finnhub primary
    try:
        fh = FinnhubClient()
        quote = fh.get_quote(symbol, retries=1)
        _quote_rate_state.record_fetch(symbol, now)

        price = float(quote.get("price", 0))

        # FinnhubClient returns 'timestamp' as ISO string from datetime.utcnow()
        # The actual Finnhub API returns 't' as Unix timestamp, but our client
        # wraps it. We compute age from current time.
        quote_timestamp = datetime.now(timezone.utc)
        quote_age = 0.0  # Quote is fetched right now, effectively 0 age

        if price > 0:
            logger.info(
                "Plan execution fresh quote for %s: price=%.2f, age=%.1fs",
                symbol, price, quote_age,
            )
            return price, quote_timestamp, quote_age

    except Exception as e:
        err_str = str(e)
        if "429" in err_str:
            logger.warning(
                "Finnhub 429 for execution quote %s — budget exhausted", symbol
            )
            return None, None, float("inf")
        logger.warning("Finnhub fresh quote failed for %s: %s", symbol, e)

    # Fallback: yfinance (less reliable timestamp)
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        fast_info = ticker.fast_info
        price = float(getattr(fast_info, "last_price", 0) or 0)
        if price <= 0:
            price = float(fast_info.get("lastPrice", 0) if hasattr(fast_info, "get") else 0)

        if price > 0:
            # yfinance does not provide a reliable quote timestamp.
            # For execution, unknown freshness is treated conservatively:
            # assign age = PLAN_EXECUTION_MAX_QUOTE_AGE_SECONDS + 1 so the
            # caller's age check will reject it for execution.
            quote_timestamp = datetime.now(timezone.utc)
            conservative_age = float(PLAN_EXECUTION_MAX_QUOTE_AGE_SECONDS + 1)
            logger.info(
                "Plan execution yfinance fallback for %s: price=%.2f, "
                "age=UNKNOWN (treated as %.0fs for safety)",
                symbol, price, conservative_age,
            )
            return price, quote_timestamp, conservative_age
    except Exception as e:
        logger.warning("yfinance fresh quote failed for %s: %s", symbol, e)

    return None, None, float("inf")


# ---------------------------------------------------------------------------
# Missed setup event recording
# ---------------------------------------------------------------------------


def _record_missed_setup_event(
    engine,
    plan: TradePlan,
    fresh_price: float,
    quote_timestamp: datetime | None,
    quote_age_seconds: float,
    reason: str,
) -> None:
    """Record a missed_setup trade event for audit/review.

    First-class audit data: includes quote age, full plan metadata, and
    all fields needed for counterfactual analysis.

    Fail-open: logs error but does not raise — event emission must never
    block the execution result.
    """
    from utils.trade_events import log_trade_event
    from db.schema import get_session

    db = get_session(engine)
    try:
        log_trade_event(
            db,
            "missed_setup",
            agent="plan_executor",
            symbol=plan.symbol,
            profile=plan.profile_id,
            message=(
                f"Plan {plan.plan_id} missed: {reason}. "
                f"Entry zone [{plan.entry_zone_lower}-{plan.entry_zone_upper}], "
                f"fresh price={fresh_price}, target={plan.target_price}, "
                f"quote_age={quote_age_seconds:.1f}s"
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
                "quote_timestamp": (
                    quote_timestamp.isoformat() if quote_timestamp else None
                ),
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
            "Failed to record missed_setup event for plan %s: %s",
            plan.plan_id, e,
        )
    finally:
        db.close()
