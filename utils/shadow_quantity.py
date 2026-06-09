"""Shadow Quantity Calculator.

Derives a non-executable deterministic quantity from profile risk parameters
and candidate geometry only. Used for shadow-outcome scoring when a candidate
has not passed execution-path position sizing.

This is intentionally simplified compared to position_sizer.py: no
risk_multiplier, no recovery_multiplier, no position cap.

Requirements: 10.3
"""

from __future__ import annotations

import math

from utils.candidate_registry import CandidateRecord


def compute_shadow_quantity(
    candidate: CandidateRecord,
    profile: dict,
    profile_id: str,
    portfolio_equity: float,
) -> int | None:
    """Derive a non-executable quantity for shadow scoring.

    Uses profile risk parameters and candidate geometry only.
    Returns None if candidate geometry is incomplete.

    Args:
        candidate: The CandidateRecord with entry_price and stop_price.
        profile: PM profile dict containing risk_per_trade_pct.
        profile_id: Profile identifier (for context, not used in calculation).
        portfolio_equity: Total portfolio equity for risk budget computation.

    Returns:
        Computed integer quantity, or None if geometry is incomplete or
        the derived quantity is non-positive.
    """
    # Step 1: Validate candidate geometry — entry_price and stop_price
    # must be finite positive numbers.
    entry_price = candidate.entry_price
    stop_price = candidate.stop_price

    if not _is_finite_positive(entry_price) or not _is_finite_positive(stop_price):
        return None

    # Step 2: Compute stop distance
    stop_distance = abs(entry_price - stop_price)

    # Step 3: If stop distance is zero, geometry is incomplete
    if stop_distance <= 0:
        return None

    # Step 4: If portfolio equity is non-positive, cannot size
    if portfolio_equity <= 0:
        return None

    # Step 5: Compute max dollar risk from profile risk parameters
    risk_per_trade_pct = profile.get("risk_per_trade_pct", 0.01)
    max_dollar_risk = portfolio_equity * risk_per_trade_pct

    # Step 6: Compute quantity (floor of risk budget / stop distance)
    quantity = math.floor(max_dollar_risk / stop_distance)

    # Step 7: If quantity is non-positive, return None
    if quantity <= 0:
        return None

    # Step 8: Return computed quantity
    return quantity


def _is_finite_positive(value: float) -> bool:
    """Check that a value is a finite positive number."""
    if value is None:
        return False
    try:
        return math.isfinite(value) and value > 0
    except (TypeError, ValueError):
        return False
