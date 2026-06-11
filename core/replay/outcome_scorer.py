"""Counterfactual Outcome Scoring for the Decision Replay Agent.

Scores the market outcome of changed decisions using post-cutoff candle data.
Includes explicit fill model with gap handling, direction-aware MFE/MAE,
stop/target hit detection, ambiguous candle handling, and shadow outcome reuse.

Scoring Triggers:
1. Decision direction changes (reject→allow or allow→reject)
2. Replay geometry differs from original geometry (beyond tick equivalence)
3. Fill model produces a different simulated_fill_price than original entry_price

Decision-Specific Scoring Branches:
- reject → allow: counterfactual fill + candle scoring
- allow → reject: actual realized outcome from trades table
- allow → allow with changed geometry: counterfactual path using replay geometry
- reject → reject with changed geometry: NO economic scoring

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.9
See: design.md §core/replay/outcome_scorer.py
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from utils.decision_snapshot import compute_geometry_hash

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FillModel:
    """Explicit fill model for counterfactual scoring.

    Represents how the simulated entry would have been filled:
    market order at next 1-min candle open after the replay cutoff.
    """

    proposed_entry_price: Decimal  # the geometry's intended entry
    simulated_fill_price: Decimal  # actual price used for scoring (first candle open after cutoff)
    fill_timestamp: datetime  # when the simulated fill occurs
    fill_rule: str  # "market_order_next_candle_open"
    max_permitted_fill_delay_seconds: int  # configurable (default: 300s / 5 min)
    fill_gap_seconds: int  # actual gap between cutoff and fill candle
    slippage_estimate: Decimal  # zero unless operator configures per-symbol


@dataclass(frozen=True)
class CandleEvidence:
    """Persisted candle data used for scoring (prevents re-fetch issues).

    The content hash ensures reproducibility: if the same candles are
    fetched later, the hash will match, confirming identical data.
    """

    candles_json: str  # normalized 1-min candles actually used
    candle_source: str  # provider name (yfinance, polygon, etc.)
    candle_fetch_timestamp: datetime  # when candles were retrieved
    candle_content_hash: str  # SHA-256 of normalized candle data


@dataclass(frozen=True)
class CounterfactualOutcome:
    """The scored market outcome of a counterfactual (changed) decision.

    Contains return measurements at 15/30/60-min windows, MFE/MAE,
    stop/target hit indicators, and the fill model + candle evidence
    used to produce the score.
    """

    return_15m: float | None
    return_30m: float | None
    return_60m: float | None
    mfe: float  # max favorable excursion (direction-aware)
    mae: float  # max adverse excursion (direction-aware)
    stop_hit: bool
    target_hit: bool
    first_hit: str  # "stop" | "target" | "ambiguous" | "neither"
    first_hit_candle_time: datetime | None
    fill_model: FillModel
    candle_evidence: CandleEvidence
    status: str  # "scored" | "ambiguous" | "unscorable_outcome" | "unscorable_fill_delay"
    unscorable_reason: str | None


# ---------------------------------------------------------------------------
# Fill Model Construction
# ---------------------------------------------------------------------------


def build_fill_model(
    proposed_entry_price: Decimal,
    cutoff: datetime,
    candles_1m: list[dict],
    *,
    max_permitted_fill_delay_seconds: int = 300,
    slippage_estimate: Decimal = Decimal("0"),
) -> FillModel:
    """Construct fill model from cutoff and available candles.

    Fill rule: market_order_next_candle_open
    - simulated_fill_price = first candle open after cutoff + slippage
    - If fill_gap_seconds > max_permitted_fill_delay_seconds,
      downstream scoring marks outcome unscorable_fill_delay

    Args:
        proposed_entry_price: The geometry's intended entry price
        cutoff: Replay cutoff timestamp
        candles_1m: List of 1-min candle dicts with at least 'open' and 'timestamp' keys
        max_permitted_fill_delay_seconds: Max allowed gap (default 300s / 5 min)
        slippage_estimate: Per-symbol slippage (default 0)

    Returns:
        FillModel with computed fill price and gap information

    Raises:
        ValueError: If candles_1m is empty (caller should handle unscorable_outcome)
    """
    if not candles_1m:
        raise ValueError("No candles available to build fill model")

    # Sort candles by timestamp to find first after cutoff
    sorted_candles = sorted(candles_1m, key=lambda c: _parse_candle_timestamp(c["timestamp"]))

    # Find first candle with timestamp strictly after cutoff
    fill_candle = None
    for candle in sorted_candles:
        candle_ts = _parse_candle_timestamp(candle["timestamp"])
        if candle_ts > cutoff:
            fill_candle = candle
            break

    if fill_candle is None:
        raise ValueError("No candle available after cutoff to determine fill price")

    fill_ts = _parse_candle_timestamp(fill_candle["timestamp"])
    fill_gap = int((fill_ts - cutoff).total_seconds())

    # Simulated fill price = first candle open + slippage
    fill_price = Decimal(str(fill_candle["open"])) + slippage_estimate

    return FillModel(
        proposed_entry_price=proposed_entry_price,
        simulated_fill_price=fill_price,
        fill_timestamp=fill_ts,
        fill_rule="market_order_next_candle_open",
        max_permitted_fill_delay_seconds=max_permitted_fill_delay_seconds,
        fill_gap_seconds=fill_gap,
        slippage_estimate=slippage_estimate,
    )


# ---------------------------------------------------------------------------
# Candle Evidence Construction
# ---------------------------------------------------------------------------


def build_candle_evidence(
    candles_1m: list[dict],
    source: str,
    fetch_timestamp: datetime,
) -> CandleEvidence:
    """Build evidence record with SHA-256 content hash for reproducibility.

    Normalization: sort by timestamp, round OHLC to 4 decimals,
    then SHA-256 the canonical JSON.

    Args:
        candles_1m: Raw 1-min candle dicts
        source: Provider name (yfinance, polygon, etc.)
        fetch_timestamp: When candles were retrieved

    Returns:
        CandleEvidence with normalized JSON and content hash
    """
    # Normalize: sort by timestamp, round OHLC to 4 decimals
    normalized_candles = []
    for candle in sorted(candles_1m, key=lambda c: str(c.get("timestamp", ""))):
        normalized = {}
        for key, value in sorted(candle.items()):
            if key in ("open", "high", "low", "close") and value is not None:
                normalized[key] = round(float(value), 4)
            else:
                normalized[key] = value
        normalized_candles.append(normalized)

    canonical_json = json.dumps(normalized_candles, sort_keys=True, default=str)
    content_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    return CandleEvidence(
        candles_json=canonical_json,
        candle_source=source,
        candle_fetch_timestamp=fetch_timestamp,
        candle_content_hash=content_hash,
    )


# ---------------------------------------------------------------------------
# Scoring Branch Determination
# ---------------------------------------------------------------------------


def should_score_counterfactual(
    original_decision: str,
    replay_decision: str,
    original_geometry: dict,
    replay_geometry: dict,
    original_entry_price: Decimal,
    simulated_fill_price: Decimal,
    tick_size: Decimal = Decimal("0.01"),
) -> str | None:
    """Determine scoring branch based on delta type.

    Returns:
        "counterfactual_fill" — reject→allow (use fill model + candles)
        "realized_outcome" — allow→reject (use actual trade P&L)
        "counterfactual_geometry" — allow→allow with changed geometry
        None — reject→reject (no economic scoring) or same decision + same geometry

    Args:
        original_decision: Normalized original decision ("allow" or "reject")
        replay_decision: Normalized replay decision ("allow" or "reject")
        original_geometry: Dict with entry_price, stop_price, target_price
        replay_geometry: Dict with entry_price, stop_price, target_price
        original_entry_price: Original proposed entry as Decimal
        simulated_fill_price: Fill model's simulated price as Decimal
        tick_size: Tick size for geometry comparison (default 0.01)
    """
    orig = _normalize_decision(original_decision)
    replay = _normalize_decision(replay_decision)

    # reject → allow: counterfactual fill scoring
    if orig == "reject" and replay == "allow":
        return "counterfactual_fill"

    # allow → reject: use actual realized trade P&L
    if orig == "allow" and replay == "reject":
        return "realized_outcome"

    # reject → reject: no economic scoring regardless of geometry change
    if orig == "reject" and replay == "reject":
        return None

    # allow → allow: check if geometry changed
    if orig == "allow" and replay == "allow":
        if _geometry_differs_beyond_tick(original_geometry, replay_geometry, tick_size):
            return "counterfactual_geometry"

        # Check if fill price differs from original entry
        entry_norm = _tick_normalize(original_entry_price, tick_size)
        fill_norm = _tick_normalize(simulated_fill_price, tick_size)
        if entry_norm != fill_norm:
            return "counterfactual_geometry"

        # Same decision, same geometry, same fill → no scoring needed
        return None

    return None


# ---------------------------------------------------------------------------
# Counterfactual Scoring (Full)
# ---------------------------------------------------------------------------


def score_counterfactual(
    fill_model: FillModel,
    stop_price: Decimal,
    target_price: Decimal,
    direction: str,
    candles_1m: list[dict],
    candle_evidence: CandleEvidence,
) -> CounterfactualOutcome:
    """Score using post-cutoff 1-min candles.

    Entry at simulated_fill_price from fill_model. Scoring windows (15/30/60 min)
    begin at the simulated fill time.

    If fill_model.fill_gap_seconds > fill_model.max_permitted_fill_delay_seconds,
    returns status="unscorable_fill_delay".

    If no candles are available in scoring windows, returns status="unscorable_outcome".

    MFE/MAE are direction-aware:
    - Long: MFE = highest_high - fill_price, MAE = lowest_low - fill_price
    - Short: MFE = fill_price - lowest_low, MAE = fill_price - highest_high

    Ambiguous candle: when the FIRST candle in the scoring window to satisfy
    either boundary condition has high/low that together satisfy BOTH stop
    and target conditions.

    Args:
        fill_model: The computed fill model
        stop_price: Replay stop price
        target_price: Replay target price
        direction: "LONG" or "SHORT"
        candles_1m: Post-cutoff 1-min candle data
        candle_evidence: Persisted evidence record

    Returns:
        CounterfactualOutcome with all scored fields
    """
    # Check for unscorable fill delay
    if fill_model.fill_gap_seconds > fill_model.max_permitted_fill_delay_seconds:
        return CounterfactualOutcome(
            return_15m=None,
            return_30m=None,
            return_60m=None,
            mfe=0.0,
            mae=0.0,
            stop_hit=False,
            target_hit=False,
            first_hit="neither",
            first_hit_candle_time=None,
            fill_model=fill_model,
            candle_evidence=candle_evidence,
            status="unscorable_fill_delay",
            unscorable_reason=(
                f"Fill gap {fill_model.fill_gap_seconds}s exceeds "
                f"max permitted {fill_model.max_permitted_fill_delay_seconds}s"
            ),
        )

    # Filter candles to scoring window (after fill time)
    fill_time = fill_model.fill_timestamp
    fill_price = float(fill_model.simulated_fill_price)
    stop_f = float(stop_price)
    target_f = float(target_price)
    is_long = direction.upper() in ("LONG", "BUY")

    # Sort and filter candles after fill time
    scoring_candles = _get_scoring_candles(candles_1m, fill_time)

    if not scoring_candles:
        return CounterfactualOutcome(
            return_15m=None,
            return_30m=None,
            return_60m=None,
            mfe=0.0,
            mae=0.0,
            stop_hit=False,
            target_hit=False,
            first_hit="neither",
            first_hit_candle_time=None,
            fill_model=fill_model,
            candle_evidence=candle_evidence,
            status="unscorable_outcome",
            unscorable_reason="No candles available within scoring window after fill time",
        )

    # Compute time-windowed returns
    return_15m = _compute_window_return(scoring_candles, fill_time, fill_price, 15, is_long)
    return_30m = _compute_window_return(scoring_candles, fill_time, fill_price, 30, is_long)
    return_60m = _compute_window_return(scoring_candles, fill_time, fill_price, 60, is_long)

    # Compute MFE/MAE across the full 60-min window
    window_60_candles = _filter_candles_in_window(scoring_candles, fill_time, 60)
    mfe, mae = _compute_mfe_mae(window_60_candles, fill_price, is_long)

    # Detect stop/target hits and first hit (with ambiguous candle handling)
    stop_hit, target_hit, first_hit, first_hit_time = _detect_hits(
        scoring_candles, fill_price, stop_f, target_f, is_long
    )

    # Determine status
    status = "scored"
    if first_hit == "ambiguous":
        status = "ambiguous"

    return CounterfactualOutcome(
        return_15m=return_15m,
        return_30m=return_30m,
        return_60m=return_60m,
        mfe=mfe,
        mae=mae,
        stop_hit=stop_hit,
        target_hit=target_hit,
        first_hit=first_hit,
        first_hit_candle_time=first_hit_time,
        fill_model=fill_model,
        candle_evidence=candle_evidence,
        status=status,
        unscorable_reason=None,
    )


# ---------------------------------------------------------------------------
# Shadow Outcome Reuse
# ---------------------------------------------------------------------------


def should_reuse_shadow_outcome(
    replay_stop: Decimal,
    replay_target: Decimal,
    shadow_stop: Decimal,
    shadow_target: Decimal,
    tick_size: Decimal = Decimal("0.01"),
) -> bool:
    """Determine if a stored shadow outcome can be reused.

    Returns True if the replay stop and target each match the shadow stop
    and target within Geometry_Hash tick-normalized equivalence.

    When geometry matches within tick equivalence, the shadow outcome already
    scored the same scenario — no need to rescore.

    Args:
        replay_stop: Replay's stop price
        replay_target: Replay's target price
        shadow_stop: Stored shadow outcome's stop price
        shadow_target: Stored shadow outcome's target price
        tick_size: Tick size for normalization (default 0.01)

    Returns:
        True if geometry matches (reuse shadow), False if it differs (rescore needed)
    """
    replay_stop_norm = _tick_normalize(replay_stop, tick_size)
    replay_target_norm = _tick_normalize(replay_target, tick_size)
    shadow_stop_norm = _tick_normalize(shadow_stop, tick_size)
    shadow_target_norm = _tick_normalize(shadow_target, tick_size)

    return replay_stop_norm == shadow_stop_norm and replay_target_norm == shadow_target_norm


# ---------------------------------------------------------------------------
# Allowed-to-Rejected Scoring
# ---------------------------------------------------------------------------


def score_allowed_to_rejected(
    realized_pnl: float,
    entry_price: Decimal,
    exit_price: Decimal,
) -> dict:
    """Calculate avoided loss or forgone gain from an allowed-to-rejected delta.

    When the original decision was "allow" and replay says "reject",
    this uses the actual realized trade outcome to show what the proposed
    policy would have avoided or missed.

    Returns exclude commission (recorded separately when configured).

    Args:
        realized_pnl: Actual P&L from the trades table
        entry_price: Original entry price
        exit_price: Actual exit price from the trade

    Returns:
        Dict with:
        - classification: "avoided_loss" | "forgone_gain"
        - realized_pnl: The actual P&L value
        - entry_price: Original entry (str for precision)
        - exit_price: Actual exit (str for precision)
        - return_pct: Percentage return (exit - entry) / entry
    """
    entry_f = float(entry_price)
    exit_f = float(exit_price)

    if entry_f != 0:
        return_pct = (exit_f - entry_f) / entry_f
    else:
        return_pct = 0.0

    if realized_pnl < 0:
        classification = "avoided_loss"
    else:
        classification = "forgone_gain"

    return {
        "classification": classification,
        "realized_pnl": realized_pnl,
        "entry_price": str(entry_price),
        "exit_price": str(exit_price),
        "return_pct": return_pct,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_decision(decision: str) -> str:
    """Normalize decision string to canonical 'allow' or 'reject'."""
    normalized = decision.lower().strip()
    if normalized in ("allow", "allowed", "adjusted_allowed"):
        return "allow"
    if normalized in ("reject", "rejected", "block", "blocked"):
        return "reject"
    return normalized


def _tick_normalize(price: Decimal, tick_size: Decimal) -> Decimal:
    """Normalize a price to the nearest tick using ROUND_HALF_UP."""
    if tick_size <= 0:
        return price
    return price.quantize(tick_size, rounding=ROUND_HALF_UP)


def _geometry_differs_beyond_tick(
    original_geometry: dict,
    replay_geometry: dict,
    tick_size: Decimal,
) -> bool:
    """Check if two geometries differ beyond tick equivalence.

    Uses compute_geometry_hash for canonical comparison.
    """
    orig_entry = original_geometry.get("entry_price")
    orig_stop = original_geometry.get("stop_price")
    orig_target = original_geometry.get("target_price")

    replay_entry = replay_geometry.get("entry_price")
    replay_stop = replay_geometry.get("stop_price")
    replay_target = replay_geometry.get("target_price")

    # Convert to Decimal for hashing
    def to_decimal(val: Any) -> Decimal | None:
        if val is None:
            return None
        return Decimal(str(val))

    orig_hash = compute_geometry_hash(
        to_decimal(orig_entry),
        to_decimal(orig_stop),
        to_decimal(orig_target),
        tick_size=tick_size,
    )
    replay_hash = compute_geometry_hash(
        to_decimal(replay_entry),
        to_decimal(replay_stop),
        to_decimal(replay_target),
        tick_size=tick_size,
    )

    # Empty hashes (incomplete geometry) → treat as not differing
    if not orig_hash or not replay_hash:
        return False

    return orig_hash != replay_hash


def _parse_candle_timestamp(ts: Any) -> datetime:
    """Parse a candle timestamp to datetime.

    Handles datetime objects directly or ISO-format strings.
    """
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        # Handle common ISO formats
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                continue
        # Fallback: try fromisoformat (handles most variants)
        return datetime.fromisoformat(ts.replace("Z", "+00:00").replace("+00:00", ""))
    raise ValueError(f"Cannot parse candle timestamp: {ts!r}")


def _get_scoring_candles(candles_1m: list[dict], fill_time: datetime) -> list[dict]:
    """Get candles at or after fill_time, sorted by timestamp."""
    result = []
    for candle in candles_1m:
        candle_ts = _parse_candle_timestamp(candle["timestamp"])
        if candle_ts >= fill_time:
            result.append(candle)
    return sorted(result, key=lambda c: _parse_candle_timestamp(c["timestamp"]))


def _filter_candles_in_window(
    scoring_candles: list[dict],
    fill_time: datetime,
    window_minutes: int,
) -> list[dict]:
    """Filter candles within a time window from fill_time."""
    window_end = fill_time + timedelta(minutes=window_minutes)
    return [
        c for c in scoring_candles
        if _parse_candle_timestamp(c["timestamp"]) <= window_end
    ]


def _compute_window_return(
    scoring_candles: list[dict],
    fill_time: datetime,
    fill_price: float,
    window_minutes: int,
    is_long: bool,
) -> float | None:
    """Compute return at the end of a time window.

    Uses the close of the last candle within the window.
    Returns None if no candles exist within the window.
    """
    window_candles = _filter_candles_in_window(scoring_candles, fill_time, window_minutes)
    if not window_candles:
        return None

    # Use close of last candle in window
    last_candle = window_candles[-1]
    close_price = float(last_candle["close"])

    if fill_price == 0:
        return 0.0

    if is_long:
        return (close_price - fill_price) / fill_price
    else:
        # Short: profit when price goes down
        return (fill_price - close_price) / fill_price


def _compute_mfe_mae(
    candles: list[dict],
    fill_price: float,
    is_long: bool,
) -> tuple[float, float]:
    """Compute direction-aware MFE and MAE.

    Long:
        MFE = highest_high - fill_price
        MAE = lowest_low - fill_price

    Short:
        MFE = fill_price - lowest_low
        MAE = fill_price - highest_high

    Returns (mfe, mae). If no candles, returns (0.0, 0.0).
    """
    if not candles:
        return 0.0, 0.0

    highest_high = max(float(c["high"]) for c in candles)
    lowest_low = min(float(c["low"]) for c in candles)

    if is_long:
        mfe = highest_high - fill_price
        mae = lowest_low - fill_price
    else:
        mfe = fill_price - lowest_low
        mae = fill_price - highest_high

    return mfe, mae


def _detect_hits(
    scoring_candles: list[dict],
    fill_price: float,
    stop_price: float,
    target_price: float,
    is_long: bool,
) -> tuple[bool, bool, str, datetime | None]:
    """Detect stop/target hits with ambiguous candle handling.

    Walks candles in time order. For each candle, checks if stop or target
    conditions are met:
    - Long stop: candle low <= stop_price
    - Long target: candle high >= target_price
    - Short stop: candle high >= stop_price
    - Short target: candle low <= target_price

    Ambiguous: the FIRST candle to satisfy either boundary condition has
    both stop AND target conditions met simultaneously.

    Returns:
        (stop_hit, target_hit, first_hit, first_hit_candle_time)
        first_hit is one of: "stop", "target", "ambiguous", "neither"
    """
    stop_hit = False
    target_hit = False
    first_hit = "neither"
    first_hit_time: datetime | None = None

    for candle in scoring_candles:
        candle_ts = _parse_candle_timestamp(candle["timestamp"])
        high = float(candle["high"])
        low = float(candle["low"])

        # Determine boundary conditions for this candle
        if is_long:
            hits_stop = low <= stop_price
            hits_target = high >= target_price
        else:
            hits_stop = high >= stop_price
            hits_target = low <= target_price

        if hits_stop or hits_target:
            # Record that boundaries were hit
            if hits_stop:
                stop_hit = True
            if hits_target:
                target_hit = True

            # Only set first_hit on the FIRST boundary-touching candle
            if first_hit == "neither":
                if hits_stop and hits_target:
                    # Ambiguous: single candle spans both stop and target
                    first_hit = "ambiguous"
                    first_hit_time = candle_ts
                elif hits_stop:
                    first_hit = "stop"
                    first_hit_time = candle_ts
                else:
                    first_hit = "target"
                    first_hit_time = candle_ts

            # Once first hit is determined and isn't ambiguous,
            # continue scanning to see if other boundary is also hit later
            # (but first_hit is already locked in)

    return stop_hit, target_hit, first_hit, first_hit_time
