"""
Portfolio Risk Engine — deterministic, LLM-free module for portfolio-level
exposure tracking, risk validation, and adaptive risk throttling.

Symbols can belong to multiple buckets (e.g., NVDA in both semis and mega_growth).
"""

BUCKETS = {
    "index": ["SPY", "QQQ", "IWM", "DIA"],
    "semis": ["NVDA", "AMD", "INTC", "TSM"],
    "ev":    ["TSLA", "LCID", "RIVN"],
    "mega_growth": ["NVDA", "TSLA", "META", "AMZN"],
}

MAX_BUCKET_EXPOSURE_PCT = 0.50
DEFAULT_MAX_TOTAL_EXPOSURE = 1.5


def compute_risk_score(
    total_exposure: float,
    bucket_exposure: dict,
    recent_losses: int = 0,
) -> float:
    """
    Composite risk score 0.0–1.0.  Higher = more risk.

    Components (equal-ish weighting):
      - max bucket exposure (0–1 scaled against 50% cap)
      - total exposure (0–1 scaled against 1.5 threshold)
      - loss streak penalty (0–1 scaled: 0 at 0 losses, 1.0 at 6+)
    """
    # Max bucket component
    max_bucket = max(bucket_exposure.values()) if bucket_exposure else 0.0
    bucket_component = min(1.0, max_bucket / MAX_BUCKET_EXPOSURE_PCT)

    # Total exposure component
    exposure_component = min(1.0, total_exposure / DEFAULT_MAX_TOTAL_EXPOSURE)

    # Loss streak component
    clamped_losses = max(0, min(recent_losses, 6))
    loss_component = clamped_losses / 6.0

    score = 0.4 * bucket_component + 0.4 * exposure_component + 0.2 * loss_component
    return max(0.0, min(1.0, score))


def compute_portfolio_risk(
    positions: list[dict],
    total_equity: float,
    max_total_exposure: float = DEFAULT_MAX_TOTAL_EXPOSURE,
) -> dict:
    """
    Compute current portfolio exposure by bucket.

    Args:
        positions: list of dicts with symbol, quantity, avg_cost, side
        total_equity: current total portfolio value
        max_total_exposure: configurable total exposure threshold

    Returns:
        {"total_exposure": float, "bucket_exposure": dict, "risk_score": float}
    """
    if total_equity == 0:
        bucket_exp = {name: 0.0 for name in BUCKETS}
        bucket_exp["other"] = 0.0
        return {
            "total_exposure": 0.0,
            "bucket_exposure": bucket_exp,
            "risk_score": 1.0,
        }

    # Build reverse lookup: symbol → list of bucket names
    symbol_to_buckets: dict[str, list[str]] = {}
    for bucket_name, symbols in BUCKETS.items():
        for sym in symbols:
            symbol_to_buckets.setdefault(sym, []).append(bucket_name)

    # Compute position values
    bucket_values: dict[str, float] = {name: 0.0 for name in BUCKETS}
    bucket_values["other"] = 0.0
    total_value = 0.0

    for pos in positions:
        sym = pos.get("symbol", "")
        qty = abs(pos.get("quantity", 0))
        avg_cost = abs(pos.get("avg_cost", 0))
        pos_value = qty * avg_cost
        total_value += pos_value

        buckets_for_sym = symbol_to_buckets.get(sym)
        if buckets_for_sym:
            for b in buckets_for_sym:
                bucket_values[b] += pos_value
        else:
            bucket_values["other"] += pos_value

    total_exposure = total_value / total_equity
    bucket_exposure = {k: v / total_equity for k, v in bucket_values.items()}
    risk_score = compute_risk_score(total_exposure, bucket_exposure)

    return {
        "total_exposure": total_exposure,
        "bucket_exposure": bucket_exposure,
        "risk_score": risk_score,
    }


def adaptive_risk_throttle(base_size: float, recent_losses: int) -> float:
    """
    Reduce position size based on recent loss streak.

    If recent_losses < 3: return base_size unchanged.
    If recent_losses >= 3: reduce by 25–50% (linear scale).
      - 3 losses → 25% reduction (return 75% of base)
      - 6+ losses → 50% reduction (return 50% of base)
    """
    if recent_losses < 3:
        return base_size

    # Linear interpolation: 3 → 0.75, 6 → 0.50
    clamped = min(recent_losses, 6)
    # fraction goes from 0.75 at 3 losses to 0.50 at 6 losses
    fraction = 0.75 - (clamped - 3) * (0.25 / 3)
    return base_size * fraction


def validate_portfolio_risk(
    new_trade: dict,
    positions: list[dict],
    total_equity: float,
    max_total_exposure: float = DEFAULT_MAX_TOTAL_EXPOSURE,
    recent_losses: int = 0,
) -> tuple[bool, str]:
    """
    Check if adding new_trade would exceed risk limits.

    Args:
        new_trade: dict with symbol, quantity, price
        positions: current open positions
        total_equity: current portfolio value
        max_total_exposure: configurable threshold (1.2–1.5)
        recent_losses: count of consecutive recent losing trades

    Returns:
        (True, "OK") if trade is allowed
        (False, "reason") if trade would exceed limits
    """
    if total_equity == 0:
        return False, "zero equity"

    # Simulate adding the new trade to positions
    sim_positions = list(positions)
    sim_positions.append({
        "symbol": new_trade.get("symbol", ""),
        "quantity": new_trade.get("quantity", 0),
        "avg_cost": new_trade.get("price", 0),
        "side": new_trade.get("side", "long"),
    })

    risk = compute_portfolio_risk(sim_positions, total_equity, max_total_exposure)

    # Check per-bucket exposure
    for bucket_name, exposure in risk["bucket_exposure"].items():
        if bucket_name == "other":
            continue
        if exposure > MAX_BUCKET_EXPOSURE_PCT:
            return False, f"{bucket_name} exposure exceeded 50%"

    # Check total exposure
    if risk["total_exposure"] > max_total_exposure:
        return False, "total exposure exceeded threshold"

    return True, "OK"
