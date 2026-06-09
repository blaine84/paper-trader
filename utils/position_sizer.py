"""
Deterministic Position Sizer

Extracted from execute_trade() into an isolated module for the candidate-ID
selection pipeline. Calculates absolute order quantity from profile risk limits,
portfolio constraints, and candidate geometry. Never accepts PM-supplied quantities.

Requirements: 4.1, 4.2, 4.3, 4.4
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SizingResult:
    """Result of deterministic position sizing."""

    quantity: int
    dollar_risk: float
    position_value: float
    sizing_method: str  # "standard" | "recovery" | "pilot"
    applied_multiplier: float  # cumulative (risk_multiplier × recovery_multiplier)
    rejection_reason: str | None = None  # set if sizing rejects

    @property
    def rejected(self) -> bool:
        """True if this sizing result is a rejection."""
        return self.rejection_reason is not None


def _make_rejection(reason: str) -> SizingResult:
    """Create a rejected SizingResult with zero values."""
    return SizingResult(
        quantity=0,
        dollar_risk=0.0,
        position_value=0.0,
        sizing_method="standard",
        applied_multiplier=0.0,
        rejection_reason=reason,
    )


def calculate_position_size(
    resolved_order,
    portfolio: dict,
    profile: dict,
    profile_id: str,
    *,
    recovery_multiplier: float = 1.0,
) -> SizingResult:
    """Calculate absolute order quantity deterministically.

    Inputs:
    - resolved_order: duck-typed object with entry_price, stop_price, action,
      risk_multiplier attributes (from candidate pipeline)
    - portfolio: dict with 'cash' and 'total_equity' keys
    - profile: PM_PROFILES entry with max_position_pct, risk_per_trade_pct, etc.
    - profile_id: profile identifier (for logging context)
    - recovery_multiplier: strategy recovery multiplier (< 1.0 reduces size)

    Returns SizingResult with computed quantity or rejection reason.

    This function NEVER accepts PM-supplied quantity values. The quantity is
    always derived from stop distance, risk limits, and portfolio constraints.
    """
    # Extract geometry from resolved order
    entry_price = resolved_order.entry_price
    stop_price = resolved_order.stop_price
    risk_multiplier = resolved_order.risk_multiplier

    # --- Validate risk_multiplier (must be > 0.0 and <= 1.0, downward only) ---
    if risk_multiplier is None:
        risk_multiplier = 1.0

    if risk_multiplier > 1.0:
        return _make_rejection(
            f"risk_multiplier {risk_multiplier} exceeds 1.0 — upward adjustment rejected"
        )

    if risk_multiplier <= 0.0:
        return _make_rejection(
            f"risk_multiplier {risk_multiplier} is non-positive — invalid"
        )

    # --- Step 1: Compute stop distance ---
    stop_distance = abs(entry_price - stop_price)

    # --- Step 2: Reject if stop distance is zero or negative ---
    if stop_distance <= 0:
        return _make_rejection("stop distance is zero or negative")

    # --- Extract portfolio state ---
    total_equity = portfolio.get("total_equity", 0.0)
    if total_equity <= 0:
        return _make_rejection("total_equity is zero or negative")

    # --- Step 3: Compute max dollar risk from profile ---
    risk_per_trade_pct = profile.get("risk_per_trade_pct", 0.01)
    max_dollar_risk = total_equity * risk_per_trade_pct

    # --- Step 4: Apply risk_multiplier (PM-requested, validated above) ---
    max_dollar_risk *= risk_multiplier

    # --- Step 5: Apply recovery_multiplier if < 1.0 ---
    if recovery_multiplier < 1.0:
        max_dollar_risk *= recovery_multiplier

    # --- Step 6: Compute raw quantity ---
    raw_quantity = math.floor(max_dollar_risk / stop_distance)

    # --- Step 7: Reject if raw quantity is zero or negative ---
    if raw_quantity <= 0:
        return _make_rejection(
            f"calculated quantity is {raw_quantity} — insufficient risk budget "
            f"(max_dollar_risk={max_dollar_risk:.2f}, stop_distance={stop_distance:.2f})"
        )

    quantity = raw_quantity

    # --- Step 8-9: Check position value against max ---
    position_value = quantity * entry_price
    max_position_pct = profile.get("max_position_pct", 0.25)
    max_position_value = total_equity * max_position_pct

    # --- Step 10: Reduce quantity to fit max position value ---
    if position_value > max_position_value:
        quantity = math.floor(max_position_value / entry_price)

    # --- Step 13: Final quantity check after reduction ---
    if quantity <= 0:
        return _make_rejection(
            f"quantity reduced to {quantity} after max_position_pct constraint "
            f"(max_position_value={max_position_value:.2f}, entry_price={entry_price:.2f})"
        )

    # --- Step 11: Compute actual dollar risk ---
    actual_dollar_risk = quantity * stop_distance

    # --- Step 12: Reject if actual dollar risk exceeds max (after multipliers) ---
    if actual_dollar_risk > max_dollar_risk:
        return _make_rejection(
            f"actual dollar risk {actual_dollar_risk:.2f} exceeds max "
            f"{max_dollar_risk:.2f} after multipliers"
        )

    # --- Step 14: Determine sizing method ---
    if recovery_multiplier < 1.0:
        sizing_method = "recovery"
    else:
        sizing_method = "standard"

    # --- Step 15: Compute applied multiplier ---
    applied_multiplier = risk_multiplier * recovery_multiplier

    # --- Compute final position value ---
    final_position_value = quantity * entry_price

    return SizingResult(
        quantity=quantity,
        dollar_risk=actual_dollar_risk,
        position_value=final_position_value,
        sizing_method=sizing_method,
        applied_multiplier=applied_multiplier,
        rejection_reason=None,
    )
