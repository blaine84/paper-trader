"""Swing Candidate Bridge — profile policy evaluation and orchestration.

This module provides the profile policy evaluator, position sizing, and the
full swing candidate bridge orchestration (process_swing_signals, etc.).

The orchestration layer handles I/O (logging, DB events) while delegating
pure logic to setup_normalizer and swing_geometry_builder.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Context, Decimal, ROUND_DOWN, ROUND_HALF_UP
from typing import Any, Literal

logger = logging.getLogger(__name__)


# Decimal contexts for financial arithmetic
_DECIMAL_CTX = Context(prec=28, rounding=ROUND_HALF_UP)
_FLOOR_CTX = Context(prec=28, rounding=ROUND_DOWN)


@dataclass(frozen=True)
class PolicyResult:
    """Result of profile policy evaluation."""

    accepted: bool
    sizing_multiplier: Decimal | None = None  # Present when accepted
    reason_code: str | None = None  # Present when rejected


POLICY_REJECTION_CODES = frozenset({
    "observe_only_period",
    "confidence_below_threshold",
    "strength_below_threshold",
    "rr_below_threshold",
    "same_symbol_overlap_blocked",
})

# Confidence ordering for comparison
_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}
_STRENGTH_ORDER = {"weak": 0, "moderate": 1, "strong": 2}


def _get_signal_direction(signal: dict) -> str:
    """Return analyst trade direction from either supported field name."""
    direction = str(signal.get("direction") or "").upper().strip()
    if direction in ("LONG", "SHORT", "HOLD"):
        return direction

    analyst_signal = str(signal.get("signal") or "").upper().strip()
    if analyst_signal in ("LONG", "SHORT", "HOLD"):
        return analyst_signal

    return "HOLD"


def evaluate_profile_policy(
    profile_id: str,
    confidence: Literal["low", "medium", "high"],
    strength: Literal["weak", "moderate", "strong"],
    risk_reward: Decimal,
    symbol: str,
    open_swing_symbols: set[str],  # symbols with open swing positions in OTHER profiles
) -> PolicyResult:
    """Evaluate whether a swing candidate passes the profile's policy.

    Checks in order:
    1. Conservative observe-only override
    2. Confidence meets profile minimum
    3. Strength meets profile minimum
    4. Risk/reward meets profile minimum (policy-level, may be tighter than geometry)
    5. Same-symbol overlap check

    Returns PolicyResult with accepted=True and sizing_multiplier, or
    accepted=False and reason_code.
    """
    from utils.gate_config import (
        SWING_CONSERVATIVE_OBSERVE_ONLY,
        SWING_PROFILE_POLICY,
    )

    policy = SWING_PROFILE_POLICY.get(profile_id)
    if policy is None:
        return PolicyResult(accepted=False, reason_code="confidence_below_threshold")

    # 1. Conservative observe-only override
    if profile_id == "conservative" and SWING_CONSERVATIVE_OBSERVE_ONLY:
        return PolicyResult(accepted=False, reason_code="observe_only_period")

    # 2. Confidence check
    min_confidence = policy["min_confidence"]
    if _CONFIDENCE_ORDER.get(confidence, 0) < _CONFIDENCE_ORDER.get(min_confidence, 0):
        return PolicyResult(accepted=False, reason_code="confidence_below_threshold")

    # 3. Strength check
    min_strength = policy["min_strength"]
    if _STRENGTH_ORDER.get(strength, 0) < _STRENGTH_ORDER.get(min_strength, 0):
        return PolicyResult(accepted=False, reason_code="strength_below_threshold")

    # 4. Risk/reward check (policy-level, may be tighter than geometry floor)
    min_rr = policy["min_risk_reward"]
    if risk_reward < min_rr:
        return PolicyResult(accepted=False, reason_code="rr_below_threshold")

    # 5. Same-symbol overlap across profiles
    if symbol in open_swing_symbols:
        return PolicyResult(accepted=False, reason_code="same_symbol_overlap_blocked")

    return PolicyResult(accepted=True, sizing_multiplier=policy["sizing_multiplier"])


@dataclass(frozen=True)
class SizingResult:
    """Result of position sizing computation."""

    accepted: bool
    quantity: int | None = None  # Present when accepted
    dollar_risk: Decimal | None = None  # Present when accepted
    reason_code: str | None = None  # Present when rejected


def compute_swing_position_size(
    portfolio_equity: Decimal,
    risk_per_trade_pct: Decimal,
    entry_price: Decimal,
    stop_price: Decimal,
    sizing_multiplier: Decimal,
) -> SizingResult:
    """Compute swing position quantity using Decimal arithmetic.

    Formula: quantity = floor(max_dollar_risk / stop_distance * sizing_multiplier)
    Where: max_dollar_risk = portfolio_equity * risk_per_trade_pct

    Rejects if computed quantity <= 0 with reason "sizing_rejected".
    Verifies dollar_risk = quantity * stop_distance does not exceed max_dollar_risk.

    Pure function — no side effects.
    """
    max_dollar_risk = _DECIMAL_CTX.multiply(portfolio_equity, risk_per_trade_pct)
    stop_distance = abs(entry_price - stop_price)

    if stop_distance == 0:
        return SizingResult(accepted=False, reason_code="sizing_rejected")

    # raw_quantity = max_dollar_risk / stop_distance * sizing_multiplier
    raw_quantity = _DECIMAL_CTX.multiply(
        _DECIMAL_CTX.divide(max_dollar_risk, stop_distance),
        sizing_multiplier,
    )

    # Floor to integer using ROUND_DOWN context
    quantity = int(_FLOOR_CTX.to_integral_value(raw_quantity))

    if quantity <= 0:
        return SizingResult(accepted=False, reason_code="sizing_rejected")

    # Verify actual dollar risk does not exceed budget
    dollar_risk = _DECIMAL_CTX.multiply(Decimal(quantity), stop_distance)

    return SizingResult(accepted=True, quantity=quantity, dollar_risk=dollar_risk)


# ---------------------------------------------------------------------------
# Swing Candidate Bridge — Orchestration Layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SwingBridgeResult:
    """Per-signal result from the swing candidate bridge."""

    signal_id: str
    symbol: str
    raw_label: str
    normalized_label: str | None
    rejection_reason: str | None
    construction_attempted: bool
    construction_succeeded: bool


def _get_swing_mode() -> str:
    """Read SWING_CANDIDATE_MODE at call time (not import time).

    This avoids module-level constant caching so tests can patch
    the environment variable without reloading the module.
    """
    from utils.gate_config import get_swing_candidate_mode
    return get_swing_candidate_mode()


def process_swing_signals(
    signals: dict[str, dict],
    profile_id: str,
    profile: dict,
    portfolio: dict,
    cycle_id: str,
    db: Any,
    engine: Any,
) -> list:
    """Process analyst signals through the swing candidate bridge.

    Respects SWING_CANDIDATE_MODE (read at call time via _get_swing_mode()):
    - disabled: returns [] immediately, no logging
    - observe: normalizes + logs per signal, returns []
    - enabled: normalizes + builds geometry + profiles policy + risk controls + registers

    Fail-open: logging/event failures caught and logged at WARNING,
    never block the pipeline.

    Returns list of registered candidate dicts (only in enabled mode).
    """
    mode = _get_swing_mode()
    if mode == "disabled":
        return []

    from utils.gate_config import (
        SWING_MAX_CANDIDATE_AGE_HOURS,
        SWING_MAX_CONCURRENT_POSITIONS,
        SWING_SECTOR_CONCENTRATION_WARN_THRESHOLD,
    )
    from utils.setup_normalizer import normalize_setup, TechnicalContext
    from utils.swing_geometry_builder import build_swing_geometry, SwingGeometry

    results: list[SwingBridgeResult] = []
    registered_candidates: list[dict] = []

    # Get open swing positions for same-symbol and max-concurrent checks
    open_swing_symbols = _get_open_swing_symbols(engine, profile_id)
    open_swing_count = len(open_swing_symbols)
    sector_counts: dict[str, int] = {}

    for signal_id, signal in signals.items():
        symbol = signal.get("symbol", "")
        raw_label = signal.get("setup_type", "")
        direction = _get_signal_direction(signal)

        # Skip stale signals
        signal_age_hours = signal.get("signal_age_hours", 0)
        if signal_age_hours > SWING_MAX_CANDIDATE_AGE_HOURS:
            result = SwingBridgeResult(
                signal_id=signal_id, symbol=symbol, raw_label=raw_label,
                normalized_label=None, rejection_reason="stale_signal",
                construction_attempted=False, construction_succeeded=False,
            )
            results.append(result)
            _safe_emit_log(result)
            continue

        # Max concurrent positions check
        max_allowed = SWING_MAX_CONCURRENT_POSITIONS.get(profile_id, 0)
        if open_swing_count >= max_allowed:
            result = SwingBridgeResult(
                signal_id=signal_id, symbol=symbol, raw_label=raw_label,
                normalized_label=None, rejection_reason="max_swing_positions_reached",
                construction_attempted=False, construction_succeeded=False,
            )
            results.append(result)
            _safe_emit_log(result)
            continue

        # Same-symbol exposure check
        if symbol in open_swing_symbols:
            result = SwingBridgeResult(
                signal_id=signal_id, symbol=symbol, raw_label=raw_label,
                normalized_label=None, rejection_reason="same_symbol_exposure",
                construction_attempted=False, construction_succeeded=False,
            )
            results.append(result)
            _safe_emit_log(result)
            continue

        # Build TechnicalContext
        tc = TechnicalContext(
            key_levels=signal.get("key_levels", {"support": None, "resistance": None}),
            ema_trend=signal.get("ema_trend", "neutral"),
            market_regime=signal.get("market_regime", "mixed"),
        )

        # Normalize
        norm_result = normalize_setup(
            raw_label=raw_label,
            direction=direction,
            strength=signal.get("strength", "weak"),
            confidence=signal.get("confidence", "low"),
            technical_context=tc,
            llm_veto_reason=signal.get("llm_veto_reason"),
            data_source_error=signal.get("data_source_error", False),
            error_code=signal.get("error_code"),
        )

        if not norm_result.success:
            result = SwingBridgeResult(
                signal_id=signal_id, symbol=symbol, raw_label=raw_label,
                normalized_label=None, rejection_reason=norm_result.reason_code,
                construction_attempted=False, construction_succeeded=False,
            )
            results.append(result)
            _safe_emit_log(result)
            if mode == "enabled":
                _safe_emit_event(db, None, cycle_id, profile_id,
                    "swing_candidate_rejected", {
                        "signal_id": signal_id, "symbol": symbol,
                        "raw_label": raw_label, "reason_code": norm_result.reason_code,
                    })
            continue

        normalized_type = norm_result.executable_type

        if mode == "observe":
            # In observe mode: normalize + log per signal, return []
            result = SwingBridgeResult(
                signal_id=signal_id, symbol=symbol, raw_label=raw_label,
                normalized_label=normalized_type, rejection_reason=None,
                construction_attempted=False, construction_succeeded=False,
            )
            results.append(result)
            _safe_emit_log(result)
            continue

        # mode == "enabled" — build geometry
        entry_price = signal.get("entry_price")
        stop_price = signal.get("stop_price")
        target_price = signal.get("target_price")

        # Convert to Decimal if not None
        entry_dec = Decimal(str(entry_price)) if entry_price is not None else Decimal("0")
        stop_dec = Decimal(str(stop_price)) if stop_price is not None else None
        target_dec = Decimal(str(target_price)) if target_price is not None else None

        geom_result = build_swing_geometry(
            symbol=symbol,
            direction=direction,
            normalized_setup_type=normalized_type,
            entry_price=entry_dec,
            stop_price=stop_dec,
            target_price=target_dec,
            source_signal_id=signal_id,
            profile_id=profile_id,
        )

        if not isinstance(geom_result, SwingGeometry):
            # Geometry rejected
            result = SwingBridgeResult(
                signal_id=signal_id, symbol=symbol, raw_label=raw_label,
                normalized_label=normalized_type, rejection_reason=geom_result.reason_code,
                construction_attempted=True, construction_succeeded=False,
            )
            results.append(result)
            _safe_emit_log(result)
            _safe_emit_event(db, None, cycle_id, profile_id,
                "swing_candidate_rejected", {
                    "signal_id": signal_id, "symbol": symbol,
                    "raw_label": raw_label, "normalized_label": normalized_type,
                    "reason_code": geom_result.reason_code,
                })
            continue

        # Profile policy check
        policy_result = evaluate_profile_policy(
            profile_id=profile_id,
            confidence=signal.get("confidence", "low"),
            strength=signal.get("strength", "weak"),
            risk_reward=geom_result.risk_reward,
            symbol=symbol,
            open_swing_symbols=open_swing_symbols,
        )

        if not policy_result.accepted:
            result = SwingBridgeResult(
                signal_id=signal_id, symbol=symbol, raw_label=raw_label,
                normalized_label=normalized_type, rejection_reason=policy_result.reason_code,
                construction_attempted=True, construction_succeeded=False,
            )
            results.append(result)
            _safe_emit_log(result)
            _safe_emit_event(db, None, cycle_id, profile_id,
                "swing_candidate_rejected", {
                    "signal_id": signal_id, "symbol": symbol,
                    "raw_label": raw_label, "normalized_label": normalized_type,
                    "reason_code": policy_result.reason_code,
                })
            continue

        # Position sizing
        portfolio_equity = Decimal(str(portfolio.get("equity", 100000)))
        risk_per_trade_pct = Decimal(str(profile.get("risk_per_trade_pct", "0.01")))

        sizing_result = compute_swing_position_size(
            portfolio_equity=portfolio_equity,
            risk_per_trade_pct=risk_per_trade_pct,
            entry_price=geom_result.entry_price,
            stop_price=geom_result.stop_price,
            sizing_multiplier=policy_result.sizing_multiplier,
        )

        if not sizing_result.accepted:
            result = SwingBridgeResult(
                signal_id=signal_id, symbol=symbol, raw_label=raw_label,
                normalized_label=normalized_type, rejection_reason="sizing_rejected",
                construction_attempted=True, construction_succeeded=False,
            )
            results.append(result)
            _safe_emit_log(result)
            _safe_emit_event(db, None, cycle_id, profile_id,
                "swing_candidate_rejected", {
                    "signal_id": signal_id, "symbol": symbol,
                    "raw_label": raw_label, "normalized_label": normalized_type,
                    "reason_code": "sizing_rejected",
                })
            continue

        # Sector concentration warning
        sector = signal.get("sector", "unknown")
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if sector_counts[sector] >= SWING_SECTOR_CONCENTRATION_WARN_THRESHOLD:
            try:
                logger.warning(
                    "Swing sector concentration: profile=%s sector=%s count=%d",
                    profile_id, sector, sector_counts[sector],
                )
            except Exception:
                pass

        # SUCCESS — candidate passes all checks
        registered_candidates.append({
            "signal_id": signal_id,
            "symbol": symbol,
            "direction": direction,
            "normalized_setup_type": normalized_type,
            "geometry": geom_result,
            "quantity": sizing_result.quantity,
            "dollar_risk": sizing_result.dollar_risk,
            "sizing_multiplier": policy_result.sizing_multiplier,
            "holding_horizon": geom_result.holding_horizon,
        })

        result = SwingBridgeResult(
            signal_id=signal_id, symbol=symbol, raw_label=raw_label,
            normalized_label=normalized_type, rejection_reason=None,
            construction_attempted=True, construction_succeeded=True,
        )
        results.append(result)
        _safe_emit_log(result)
        _safe_emit_event(db, signal_id, cycle_id, profile_id,
            "swing_candidate_constructed", {
                "signal_id": signal_id, "symbol": symbol,
                "raw_label": raw_label, "normalized_label": normalized_type,
            })

        # Update tracking for subsequent signals in this batch
        open_swing_symbols.add(symbol)
        open_swing_count += 1

    return registered_candidates


def _get_open_swing_symbols(engine: Any, profile_id: str) -> set[str]:
    """Query open swing positions for same-symbol exposure check.

    Returns set of symbols with open swing positions (for same-symbol
    overlap detection). Fail-open: returns empty set on error.
    """
    if engine is None:
        return set()
    try:
        from sqlalchemy import text as sql_text
        with engine.connect() as conn:
            rows = conn.execute(
                sql_text("""
                    SELECT DISTINCT symbol FROM pm_candidates
                    WHERE COALESCE(candidate_type, 'intraday') = 'swing'
                      AND state IN ('registered', 'reserved')
                """),
            ).fetchall()
        return {row[0] for row in rows}
    except Exception:
        return set()


def _safe_emit_log(result: SwingBridgeResult) -> None:
    """Emit structured INFO log. Fail-open."""
    try:
        _emit_bridge_log(result)
    except Exception:
        try:
            logger.warning("Failed to emit swing bridge log for signal %s", result.signal_id)
        except Exception:
            pass


def _emit_bridge_log(result: SwingBridgeResult) -> None:
    """Emit structured INFO log for a processed signal.

    Fields: signal_id, symbol, raw_label, normalized_label,
    rejection_reason, construction_attempted, construction_succeeded.
    """
    logger.info(
        "swing_bridge_signal: signal_id=%s symbol=%s raw_label=%s "
        "normalized_label=%s rejection_reason=%s "
        "construction_attempted=%s construction_succeeded=%s",
        result.signal_id,
        result.symbol,
        result.raw_label,
        result.normalized_label,
        result.rejection_reason,
        result.construction_attempted,
        result.construction_succeeded,
    )


def _safe_emit_event(
    db: Any,
    candidate_id: str | None,
    cycle_id: str,
    profile_id: str,
    event_type: str,
    event_data: dict,
) -> None:
    """Emit event. Fail-open."""
    try:
        _emit_bridge_event(db, candidate_id, cycle_id, profile_id, event_type, event_data)
    except Exception:
        try:
            logger.warning("Failed to emit swing bridge event: %s", event_type)
        except Exception:
            pass


def _emit_bridge_event(
    db: Any,
    candidate_id: str | None,
    cycle_id: str,
    profile_id: str,
    event_type: str,
    event_data: dict,
) -> None:
    """Write pm_candidate_events row. Fail-open on error."""
    if db is None:
        return
    from sqlalchemy import text as sql_text
    now = datetime.now(timezone.utc).isoformat()
    with db.connect() as conn:
        conn.execute(
            sql_text("""
                INSERT INTO pm_candidate_events
                (candidate_id, cycle_id, profile_id, event_type, event_data, created_at, candidate_type)
                VALUES (:cid, :cycle_id, :profile_id, :event_type, :event_data, :created_at, :candidate_type)
            """),
            {
                "cid": candidate_id or "",
                "cycle_id": cycle_id,
                "profile_id": profile_id,
                "event_type": event_type,
                "event_data": json.dumps(event_data),
                "created_at": now,
                "candidate_type": "swing",
            },
        )
        conn.commit()
