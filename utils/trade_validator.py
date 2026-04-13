"""
Trade Validation Layer
Validates every trade before it hits the database.
Prevents null stops, bad R:R, and oversized positions.
"""

import logging
from models.pm_profiles import PM_PROFILES

log = logging.getLogger(__name__)


class TradeValidationError(Exception):
    """Raised when a trade fails validation checks."""
    pass


def validate_trade(decision: dict, profile_id: str, cash: float, total_equity: float, direction: str):
    """
    Validate a trade decision before execution.
    Raises TradeValidationError if any check fails.
    """
    symbol = decision.get("symbol", "?")
    price = decision.get("price") or decision.get("entry_price") or 0
    stop = decision.get("stop") or decision.get("stop_price") or decision.get("stop_loss")
    target = decision.get("target") or decision.get("target_price") or decision.get("profit_target")
    quantity = decision.get("quantity", 0)
    action = decision.get("action", "")

    if action == "CLOSE":
        return  # no validation needed for closes

    # 1. Price must be valid
    if not price or not isinstance(price, (int, float)) or price <= 0:
        raise TradeValidationError(f"{symbol}: invalid entry price ({price})")

    # 2. Stop must be valid
    if not stop or not isinstance(stop, (int, float)) or stop <= 0:
        raise TradeValidationError(f"{symbol}: stop_price is null or invalid ({stop})")

    # 3. Target must be valid
    if not target or not isinstance(target, (int, float)) or target <= 0:
        raise TradeValidationError(f"{symbol}: target_price is null or invalid ({target})")

    # 4. Stop must be on the correct side
    if direction == "LONG" and stop >= price:
        raise TradeValidationError(f"{symbol}: LONG stop ({stop}) must be below entry ({price})")
    if direction == "SHORT" and stop <= price:
        raise TradeValidationError(f"{symbol}: SHORT stop ({stop}) must be above entry ({price})")

    # 5. Target must be on the correct side
    if direction == "LONG" and target <= price:
        raise TradeValidationError(f"{symbol}: LONG target ({target}) must be above entry ({price})")
    if direction == "SHORT" and target >= price:
        raise TradeValidationError(f"{symbol}: SHORT target ({target}) must be below entry ({price})")

    # 6. R:R must be at least 1:1
    if direction == "LONG":
        risk = price - stop
        reward = target - price
    else:
        risk = stop - price
        reward = price - target

    if risk <= 0:
        raise TradeValidationError(f"{symbol}: zero or negative risk ({risk})")

    rr_ratio = reward / risk
    if rr_ratio < 1.0:
        raise TradeValidationError(
            f"{symbol}: R:R ratio {rr_ratio:.2f} is below minimum 1:1 "
            f"(risk={risk:.2f}, reward={reward:.2f})"
        )

    # 7. Position size must not exceed profile max allocation
    profile = PM_PROFILES.get(profile_id)
    if profile and total_equity > 0:
        position_value = quantity * price
        max_pct = profile.get("max_position_pct", 0.35)
        max_value = total_equity * max_pct
        if position_value > max_value:
            raise TradeValidationError(
                f"{symbol}: position ${position_value:,.0f} exceeds "
                f"{max_pct*100:.0f}% max (${max_value:,.0f}) for {profile_id}"
            )

    # 8. Quantity must be positive
    if not quantity or quantity <= 0:
        raise TradeValidationError(f"{symbol}: invalid quantity ({quantity})")

    log.info(f"Trade validated: {direction} {quantity} {symbol} @ {price} | stop={stop} target={target} R:R={rr_ratio:.1f}")


# Correlated pairs — don't hold the same direction simultaneously
CORRELATED_PAIRS = {
    frozenset({"SPY", "IWM"}),
    frozenset({"SPY", "QQQ"}),
    frozenset({"QQQ", "IWM"}),
    frozenset({"SPY", "DIA"}),
}


def check_correlation(symbol: str, direction: str, profile_id: str, db) -> str:
    """
    Check if opening this position would create correlated exposure.
    Returns warning message or empty string.
    """
    from db.schema import Position
    positions = db.query(Position).filter_by(profile=profile_id).all()

    for pos in positions:
        if pos.side == direction.lower() or (direction == "LONG" and pos.side == "long") or \
           (direction == "SHORT" and pos.side == "short"):
            pair = frozenset({symbol, pos.symbol})
            if pair in CORRELATED_PAIRS:
                return (f"Correlated exposure: already {pos.side} {pos.symbol}, "
                        f"adding {direction} {symbol} compounds regime risk")
    return ""


# ─── CONFIDENCE ADJUSTMENT BASED ON CASE LIBRARY ─────────────────────────────

WIN_RATE_BLOCK_THRESHOLD = 0.35    # block trade if win rate below this
WIN_RATE_DOWNGRADE_THRESHOLD = 0.50  # downgrade confidence if below this
MIN_CASES_FOR_ADJUSTMENT = 5        # need at least this many cases to adjust


def adjust_confidence(engine, setup_type: str, market_regime: str = None) -> dict:
    """
    Query case library for this setup_type + regime.
    Returns confidence modifier and whether trade should be blocked.

    Returns:
        {
            "modifier": 0.0-1.0 (multiply against base confidence),
            "block": True/False,
            "reason": "explanation",
            "win_rate": float or None,
            "total_cases": int,
        }
    """
    from models.case import Case
    from db.schema import get_session

    db = get_session(engine)

    # Query cases matching setup_type
    query = db.query(Case).filter_by(setup_type=setup_type)
    if market_regime:
        regime_cases = query.filter_by(market_regime=market_regime).all()
    else:
        regime_cases = []

    # Fall back to all cases for this setup if regime-specific data is sparse
    all_setup_cases = query.all()
    db.close()

    # Use regime-specific if enough data, otherwise all setup cases
    if len(regime_cases) >= MIN_CASES_FOR_ADJUSTMENT:
        cases = regime_cases
        context = f"{setup_type} in {market_regime}"
    elif len(all_setup_cases) >= MIN_CASES_FOR_ADJUSTMENT:
        cases = all_setup_cases
        context = f"{setup_type} (all regimes)"
    else:
        return {
            "modifier": 1.0,
            "block": False,
            "reason": f"Insufficient data ({len(all_setup_cases)} cases) — no adjustment",
            "win_rate": None,
            "total_cases": len(all_setup_cases),
        }

    total = len(cases)
    wins = sum(1 for c in cases if c.outcome == "success")
    win_rate = wins / total

    # Block if win rate is terrible
    if win_rate < WIN_RATE_BLOCK_THRESHOLD:
        return {
            "modifier": 0.0,
            "block": True,
            "reason": f"{context}: win rate {win_rate:.0%} ({wins}/{total}) below {WIN_RATE_BLOCK_THRESHOLD:.0%} threshold — BLOCKED",
            "win_rate": round(win_rate, 3),
            "total_cases": total,
        }

    # Downgrade if win rate is mediocre
    if win_rate < WIN_RATE_DOWNGRADE_THRESHOLD:
        modifier = win_rate / WIN_RATE_DOWNGRADE_THRESHOLD  # scales 0.7-1.0
        return {
            "modifier": round(modifier, 2),
            "block": False,
            "reason": f"{context}: win rate {win_rate:.0%} ({wins}/{total}) — confidence downgraded to {modifier:.0%}",
            "win_rate": round(win_rate, 3),
            "total_cases": total,
        }

    # Good win rate — no adjustment
    return {
        "modifier": 1.0,
        "block": False,
        "reason": f"{context}: win rate {win_rate:.0%} ({wins}/{total}) — no adjustment needed",
        "win_rate": round(win_rate, 3),
        "total_cases": total,
    }
