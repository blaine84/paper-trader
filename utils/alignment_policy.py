"""Alignment Policy — deterministic market-alignment evaluation for candidates.

Evaluates a candidate against its frozen context snapshot and registered
benchmarks to produce one normalized outcome: allow, reduce_size, reject,
or not_evaluated.

Requirements: 18.1–18.7
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from utils.gate_config import PM_ALIGNMENT_POLICY_MODE

logger = logging.getLogger(__name__)


class AlignmentOutcome(Enum):
    """Possible alignment evaluation outcomes (Requirement 18.2)."""
    ALLOW = "allow"
    REDUCE_SIZE = "reduce_size"
    REJECT = "reject"
    NOT_EVALUATED = "not_evaluated"


@dataclass
class AlignmentResult:
    """Result of alignment policy evaluation.

    Fields:
        outcome: One of the four deterministic outcomes.
        size_multiplier: Downward only — always in (0.0, 1.0]. 1.0 means no reduction.
        rule_triggered: Exact rule name that produced the outcome (Requirement 18.6).
        measurements_used: Dict of measurements consulted during evaluation.
        benchmark_evaluated: Which registered benchmark was used (not universal ETF).
        mode: Current policy mode (disabled/log_only/enforcing).
        version: Policy configuration version for auditability (Requirement 18.4).
        timestamp: UTC evaluation time.
    """
    outcome: AlignmentOutcome
    size_multiplier: float = 1.0  # Only downward: (0.0, 1.0]
    rule_triggered: str | None = None
    measurements_used: dict = field(default_factory=dict)
    benchmark_evaluated: str | None = None
    mode: str = "disabled"
    version: str = "1.0.0"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# Configurable, versioned thresholds (Requirement 18.4)
DEFAULT_ALIGNMENT_THRESHOLDS: dict = {
    "version": "1.0.0",
    # Relative strength thresholds (symbol change % - benchmark change %)
    "reject_below_relative_strength": -3.0,  # Reject if symbol lags sector benchmark by >3%
    "reduce_below_relative_strength": -1.5,  # Reduce size if lagging by >1.5%
    "reduce_size_multiplier": 0.5,  # Apply 0.5x when reducing (bounded downward)
    # Momentum alignment
    "reject_on_counter_momentum": True,  # Reject if buying into bearish or shorting into bullish
}


def evaluate_alignment(
    context_snapshot_json: str | None,
    direction: str,
    *,
    thresholds: dict | None = None,
    mode_override: str | None = None,
) -> AlignmentResult:
    """Evaluate candidate alignment against its registered benchmarks.

    One deterministic alignment policy producing exactly one normalized
    outcome (Requirement 18.2). Evaluates against registered benchmarks,
    NOT a universal ETF (Req 18.3). Only bounded downward size adjustments
    — never increases position size (Req 18.5). Records measurements and
    exact rule that produced the outcome (Req 18.6). No duplicate penalty
    stacking — returns after the first matching rule (Req 18.7).

    Args:
        context_snapshot_json: Frozen context snapshot JSON string from
            the candidate's CandidateRecord.context_snapshot_json.
        direction: Trade direction ("BUY" or "SHORT").
        thresholds: Override thresholds dict (or use DEFAULT_ALIGNMENT_THRESHOLDS).
        mode_override: Override mode instead of reading PM_ALIGNMENT_POLICY_MODE.

    Returns:
        AlignmentResult with outcome, audit trail, and mode.
    """
    mode = mode_override if mode_override is not None else PM_ALIGNMENT_POLICY_MODE
    config = thresholds or DEFAULT_ALIGNMENT_THRESHOLDS
    version = config.get("version", "1.0.0")

    # If disabled, return not_evaluated immediately
    if mode == "disabled":
        return AlignmentResult(
            outcome=AlignmentOutcome.NOT_EVALUATED,
            mode="disabled",
            version=version,
            rule_triggered="policy_disabled",
        )

    # If no context snapshot provided, cannot evaluate
    if not context_snapshot_json:
        return AlignmentResult(
            outcome=AlignmentOutcome.NOT_EVALUATED,
            mode=mode,
            version=version,
            rule_triggered="no_context_snapshot",
        )

    # Parse context snapshot
    try:
        ctx = json.loads(context_snapshot_json)
    except (json.JSONDecodeError, TypeError):
        return AlignmentResult(
            outcome=AlignmentOutcome.NOT_EVALUATED,
            mode=mode,
            version=version,
            rule_triggered="invalid_context_json",
        )

    # If context is excluded, cannot evaluate
    if ctx.get("context_state") == "excluded":
        return AlignmentResult(
            outcome=AlignmentOutcome.NOT_EVALUATED,
            mode=mode,
            version=version,
            rule_triggered="context_excluded",
        )

    # Extract registered sector benchmark (Requirement 18.3 — not universal ETF)
    sector_bm = ctx.get("sector_benchmark")
    rs_data = ctx.get("relative_strength", {})
    sector_rs = rs_data.get(sector_bm) if sector_bm else None

    measurements = {
        "sector_benchmark": sector_bm,
        "sector_relative_strength": sector_rs,
        "symbol_momentum": ctx.get("symbol_momentum"),
        "direction": direction,
    }

    # --- Rule evaluation (first match wins — no stacking, Req 18.7) ---

    # Rule 1: Counter-momentum rejection
    # Reject if buying into bearish momentum or shorting into bullish momentum
    if config.get("reject_on_counter_momentum"):
        symbol_momentum = ctx.get("symbol_momentum")
        if direction == "BUY" and symbol_momentum == "bearish":
            return AlignmentResult(
                outcome=AlignmentOutcome.REJECT,
                mode=mode,
                version=version,
                rule_triggered="counter_momentum_buy_into_bearish",
                measurements_used=measurements,
                benchmark_evaluated=sector_bm,
            )
        if direction == "SHORT" and symbol_momentum == "bullish":
            return AlignmentResult(
                outcome=AlignmentOutcome.REJECT,
                mode=mode,
                version=version,
                rule_triggered="counter_momentum_short_into_bullish",
                measurements_used=measurements,
                benchmark_evaluated=sector_bm,
            )

    # Rule 2: Relative strength rejection (severely lagging sector benchmark)
    reject_threshold = config.get("reject_below_relative_strength", -3.0)
    if sector_rs is not None and direction == "BUY":
        if sector_rs < reject_threshold:
            return AlignmentResult(
                outcome=AlignmentOutcome.REJECT,
                mode=mode,
                version=version,
                rule_triggered=f"relative_strength_below_reject ({sector_rs:.2f} < {reject_threshold})",
                measurements_used=measurements,
                benchmark_evaluated=sector_bm,
            )

    # Rule 3: Relative strength size reduction (moderately lagging)
    reduce_threshold = config.get("reduce_below_relative_strength", -1.5)
    reduce_multiplier = config.get("reduce_size_multiplier", 0.5)
    # Ensure multiplier is bounded downward only (Requirement 18.5)
    reduce_multiplier = max(0.01, min(reduce_multiplier, 1.0))

    if sector_rs is not None and direction == "BUY":
        if sector_rs < reduce_threshold:
            return AlignmentResult(
                outcome=AlignmentOutcome.REDUCE_SIZE,
                size_multiplier=reduce_multiplier,
                mode=mode,
                version=version,
                rule_triggered=f"relative_strength_below_reduce ({sector_rs:.2f} < {reduce_threshold})",
                measurements_used=measurements,
                benchmark_evaluated=sector_bm,
            )

    # No rule triggered — allow
    return AlignmentResult(
        outcome=AlignmentOutcome.ALLOW,
        mode=mode,
        version=version,
        rule_triggered="no_rule_triggered",
        measurements_used=measurements,
        benchmark_evaluated=sector_bm,
    )


# ============================================================
# Alignment Observation Mode (Requirements 19.1–19.5)
# ============================================================


def record_alignment_observation(
    engine,
    candidate_id: str,
    cycle_id: str,
    profile_id: str,
    alignment_result: AlignmentResult,
    candidate_fate: str,  # "executed" | "gate_rejected" | "sizing_rejected" | "not_selected"
    realized_outcome: dict | None = None,  # After scoring: {pnl_pct, r_multiple, win}
) -> None:
    """Record alignment observation for log_only analysis period.

    Records proposed outcome, measurements, and candidate fate without
    enforcing the outcome. Used during the minimum 5-session observation
    period (Requirement 19.1).

    Each record captures:
      - proposed_outcome: what the alignment policy would have decided
      - proposed_multiplier: what size adjustment would have been applied
      - rule_triggered: exact rule that fired
      - measurements_used: benchmark measurements consulted
      - benchmark_evaluated: which registered benchmark was used
      - mode: current policy mode (should be log_only)
      - candidate_fate: what actually happened to the candidate
      - realized_outcome: post-trade outcome if available (filled later)
      - false_positive / false_negative placeholders for shadow outcome scorer

    Args:
        engine: SQLAlchemy engine.
        candidate_id: The candidate being evaluated.
        cycle_id: Current PM cycle.
        profile_id: Active profile.
        alignment_result: The result from evaluate_alignment().
        candidate_fate: What actually happened to the candidate.
        realized_outcome: Post-trade outcome if available (filled later).
    """
    from sqlalchemy import text

    observation_data = {
        "proposed_outcome": alignment_result.outcome.value,
        "proposed_multiplier": alignment_result.size_multiplier,
        "rule_triggered": alignment_result.rule_triggered,
        "measurements_used": alignment_result.measurements_used,
        "benchmark_evaluated": alignment_result.benchmark_evaluated,
        "mode": alignment_result.mode,
        "version": alignment_result.version,
        "candidate_fate": candidate_fate,
        "realized_outcome": realized_outcome,
        # False positive: alignment would reject/reduce, but candidate succeeded
        # False negative: alignment would allow, but candidate failed
        # These get filled in later by the shadow outcome scorer
        "false_positive": None,
        "false_negative": None,
    }

    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO pm_candidate_events
                    (candidate_id, cycle_id, profile_id, event_type, event_data, created_at)
                    VALUES (:cid, :cycle_id, :profile_id, :event_type, :event_data, :created_at)
                """),
                {
                    "cid": candidate_id,
                    "cycle_id": cycle_id,
                    "profile_id": profile_id,
                    "event_type": "alignment_observation",
                    "event_data": json.dumps(observation_data, default=str),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            conn.commit()
    except Exception as exc:
        logger.warning("Failed to record alignment observation for %s: %s", candidate_id, exc)


def should_enforce_alignment(engine) -> bool:
    """Check if alignment policy has completed minimum observation period.

    Enforcement requires (Requirements 19.3, 19.4, 19.5):
    1. Minimum 5 trading sessions of observation data
    2. Explicit operator approval (checked via environment variable)

    A "trading session" is defined as a distinct calendar date (UTC) on which
    at least one alignment_observation event was recorded.

    Returns True only if BOTH conditions are met:
    - PM_ALIGNMENT_ENFORCEMENT_APPROVED env var is "true"
    - At least 5 distinct dates have alignment_observation events
    """
    import os
    from sqlalchemy import text

    # Condition 1: Check explicit operator approval (Requirement 19.5)
    # Enforcement requires reviewed evidence and explicit operator approval.
    approval = os.environ.get("PM_ALIGNMENT_ENFORCEMENT_APPROVED", "false")
    if approval.lower() != "true":
        return False

    # Condition 2: Check minimum sessions (Requirement 19.3)
    # Count distinct calendar dates (UTC) with alignment_observation events.
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT COUNT(DISTINCT DATE(created_at)) as session_count
                    FROM pm_candidate_events
                    WHERE event_type = 'alignment_observation'
                """)
            ).fetchone()
            session_count = row[0] if row else 0
            return session_count >= 5
    except Exception as exc:
        logger.warning("Failed to check alignment enforcement readiness: %s", exc)
        return False
