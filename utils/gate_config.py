"""Shared constants for all trade safety gates.

Single source of truth for gate thresholds, symbol role sets, event type
mappings, and rejection reason codes.  All gate modules import from here
to prevent divergent defaults.

See: requirements.md §Default Configuration, design.md §utils/gate_config.py
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta

logger = logging.getLogger(__name__)

# Module-level flag to ensure pilot expiration is logged only once per process.
_pilot_expiration_logged: bool = False

# ---------------------------------------------------------------------------
# Setup Quality Gate
# ---------------------------------------------------------------------------

MIN_WIN_RATE_BY_SETUP: dict[str, float] = {
    "momentum_fade": 0.35,
    "news_breakout": 0.40,
    "gap_and_go": 0.45,
    "technical_breakout": 0.40,
}
DEFAULT_MIN_WIN_RATE: float = 0.40

# Profile-aware setup quality floors. The base setup thresholds remain the
# conservative default; moderate/aggressive can take more experimental flow
# without letting truly broken setups run unchecked.
MIN_WIN_RATE_BY_SETUP_PROFILE: dict[str, dict[str, float]] = {
    "momentum_fade": {
        "conservative": 0.35,
        "moderate": 0.30,
        "aggressive": 0.20,
    },
    "news_breakout": {
        "conservative": 0.40,
        "moderate": 0.35,
        "aggressive": 0.25,
    },
    "gap_and_go": {
        "conservative": 0.45,
        "moderate": 0.40,
        "aggressive": 0.30,
    },
    "technical_breakout": {
        "conservative": 0.40,
        "moderate": 0.35,
        "aggressive": 0.25,
    },
}
DEFAULT_MIN_WIN_RATE_BY_PROFILE: dict[str, float] = {
    "conservative": DEFAULT_MIN_WIN_RATE,
    "moderate": 0.35,
    "aggressive": 0.25,
}

ROLLING_WINDOW: int = 5
MIN_CASES_FOR_BLOCK: int = 5
MIN_ROLLING_CASES: int = 3
CONSECUTIVE_LOSS_PAUSE_THRESHOLD: int = 3
CONSECUTIVE_LOSS_PAUSE_EXEMPT_SETUPS: set[str] = {"gap_and_go"}

# Recovery override
# Recovery evaluates the configured rolling window, so this minimum must be
# attainable within that window.
RECOVERY_MIN_ROLLING_CASES: int = ROLLING_WINDOW
RECOVERY_WIN_RATE_MARGIN: float = 0.15
REQUIRE_POSITIVE_ROLLING_AVG_PNL_FOR_RECOVERY: bool = True

# Recovery probe sizing — applied when a profile permits bounded recovery
# probes under rolling underperformance (aggressive unconditional, moderate
# with high-confirmation only).
ROLLING_RECOVERY_PROBE_SIZE_MULTIPLIER: float = 0.25

# Shadow scoring — maximum allowable entry price deviation percentage.
# Candidates with entry-price-to-first-candle deviation above this threshold
# are marked unscorable.
SHADOW_SCORE_MAX_ENTRY_DEVIATION_PCT: float = 0.20

# Near-miss margin for threshold softening — moderate-profile candidates
# within this margin below the rejection threshold qualify for pilot override.
NEAR_MISS_MARGIN_PCT: float = 0.05

if not (0 < ROLLING_RECOVERY_PROBE_SIZE_MULTIPLIER < 1.0):
    raise ValueError(
        f"ROLLING_RECOVERY_PROBE_SIZE_MULTIPLIER must be in (0, 1.0), "
        f"got {ROLLING_RECOVERY_PROBE_SIZE_MULTIPLIER}"
    )

# ---------------------------------------------------------------------------
# Pre-Trade Quality Gate
# ---------------------------------------------------------------------------

OVERRIDE_MIN_CONFIDENCE_SCORE: float = 8.0

# ---------------------------------------------------------------------------
# Concentration Gate
# ---------------------------------------------------------------------------

RECENT_DUPLICATE_WINDOW_MINUTES: int = 30

# ---------------------------------------------------------------------------
# Symbol Role Sets
# ---------------------------------------------------------------------------

CONTEXT_ONLY: set[str] = {"SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "XLK", "XLF", "XLE"}
HIGH_BETA_CLUSTER: set[str] = {"AMD", "NVDA", "TSLA"}
SEMI_CLUSTER: set[str] = {"AMD", "NVDA", "AVGO", "SMCI", "ARM", "MU", "INTC"}
CRYPTO_PROXY_CLUSTER: set[str] = {"COIN", "MSTR"}

# ---------------------------------------------------------------------------
# Risk Geometry Gate — Stop Distance Rules
# ---------------------------------------------------------------------------

STOP_DISTANCE_RULES: dict[str, dict] = {
    "high_beta_mega_cap_intraday": {
        "min_pct": 0.015,
        "atr_multiplier": 1.5,
        "min_reward_to_risk": 2.0,
        "min_reward_to_risk_by_profile": {
            "conservative": 2.0,
            "moderate": 1.5,
            "aggressive": 1.25,
        },
        "allow_pct_only_fallback": True,
        "atr_max_age_minutes": 15,
        "tactical_stop_by_profile": {
            "aggressive": {
                "enabled": True,
                "qualifying_setups": [
                    "support_bounce",
                    "vwap_pullback",
                    "pullback_continuation",
                ],
                "conditional_setups": ["news_breakout"],
                "tactical_context_indicators": ["support", "bounce", "vwap", "pullback"],
                "min_pct": 0.002,
                "atr_multiplier": 1.0,
                "min_reward_to_risk": 1.25,
            }
        },
    },
    "etf_intraday": {
        "min_pct": 0.008,
        "atr_multiplier": 1.2,
        "min_reward_to_risk": 1.8,
        "min_reward_to_risk_by_profile": {
            "conservative": 1.8,
            "moderate": 1.5,
            "aggressive": 1.25,
        },
        "allow_pct_only_fallback": True,
        "atr_max_age_minutes": 15,
    },
    "small_cap_momentum": {
        "min_pct": 0.025,
        "atr_multiplier": 2.0,
        "min_reward_to_risk": 2.5,
        "min_reward_to_risk_by_profile": {
            "conservative": 2.5,
            "moderate": 2.0,
            "aggressive": 1.5,
        },
        "allow_pct_only_fallback": False,
        "atr_max_age_minutes": 10,
    },
}

DEFAULT_STOP_DISTANCE_RULE: dict = {
    "min_pct": 0.012,
    "atr_multiplier": 1.3,
    "min_reward_to_risk": 2.0,
    "min_reward_to_risk_by_profile": {
        "conservative": 2.0,
        "moderate": 1.5,
        "aggressive": 1.25,
    },
    "allow_pct_only_fallback": True,
    "atr_max_age_minutes": 15,
}

# ---------------------------------------------------------------------------
# Feedback Rule Registry
# ---------------------------------------------------------------------------

DEFAULT_RULE_TTL_DAYS: int = 14
MAX_ACTIVE_RULES: int = 25
MAX_EVIDENCE_REFS_PER_RULE: int = 10
RULE_REGISTRY_LOOKBACK_DAYS: int = 30

# ---------------------------------------------------------------------------
# Gate Event Types — maps gate decision to TradeEvent event_type
# ---------------------------------------------------------------------------

GATE_EVENT_TYPES: dict[str, str] = {
    "allow": "gate_allowed",
    "warn": "gate_warned",
    "downgrade": "gate_downgraded",
    "reject": "gate_rejected",
    "reduce_size": "gate_size_reduced",
    "override_required": "gate_override_required",
    "override_approved": "gate_override_approved",
    "pilot_override": "gate_pilot_override",
    "risk_geometry_gate_evaluated": "risk_geometry_gate_evaluated",
    "catalyst_specificity_gate_evaluated": "catalyst_specificity_gate_evaluated",
}

# ---------------------------------------------------------------------------
# Rejection Reasons — canonical set of rejection category codes
# ---------------------------------------------------------------------------

REJECTION_REASONS: set[str] = {
    "setup_quality_gate",
    "pre_trade_quality_gate",
    "catalyst_specificity_gate",
    "catalyst_timing_risk",
    "concentration_limit",
    "correlation_limit",
    "invalid_stop_target",
    "insufficient_cash",
    "price_target_missed",
    "signal_invalidated",
    "timeout_expired",
    "pm_override_missing",
}

# ---------------------------------------------------------------------------
# Catalyst Specificity Gate — Profile Thresholds
# ---------------------------------------------------------------------------

CATALYST_SPECIFICITY_PROFILE_THRESHOLDS: dict[str, dict[str, int]] = {
    "conservative": {"allow": 8, "warn": 6},
    "moderate": {"allow": 7, "warn": 5},
    "aggressive": {"allow": 6, "warn": 4},
}

CATALYST_SPECIFICITY_SECTOR_SYMPATHY_SIZE_MULTIPLIER: dict[str, float] = {
    "conservative": 0.0,
    "moderate": 0.5,
    "aggressive": 0.5,
}

# ---------------------------------------------------------------------------
# Setup-Specific R:R Threshold Overrides
# ---------------------------------------------------------------------------

QUALIFYING_SETUP_TYPES: list[str] = [
    "news_breakout",
    "technical_breakout",
    "sector_move",
]

QUALIFYING_MIN_SIGNAL_STRENGTH: float = 7.5

REDUCED_RR_THRESHOLDS_BY_PROFILE: dict[str, float] = {
    "aggressive": 0.5,
    "moderate": 0.75,
    "conservative": 1.0,
}

# ---------------------------------------------------------------------------
# Candidate-ID Selection Feature Flags
# ---------------------------------------------------------------------------

# Values: "disabled" | "shadow" | "enabled"
PM_CANDIDATE_MODE: str = os.environ.get("PM_CANDIDATE_MODE", "disabled")

# P1 Benchmark Context (independent of P0)
PM_BENCHMARK_CONTEXT_ENABLED: bool = os.environ.get(
    "PM_BENCHMARK_CONTEXT_ENABLED", "false"
).lower() == "true"

# P1 Alignment Policy
# Values: "disabled" | "log_only" | "enforcing"
PM_ALIGNMENT_POLICY_MODE: str = os.environ.get("PM_ALIGNMENT_POLICY_MODE", "disabled")


# ---------------------------------------------------------------------------
# Pilot Controller
# ---------------------------------------------------------------------------


def is_moderate_near_miss_pilot_active(now: date | None = None) -> bool:
    """Check if the moderate near-miss pilot is currently active.

    Args:
        now: Override for current date (for testability). Defaults to date.today().

    Conditions for active:
    1. MODERATE_NEAR_MISS_PILOT env var == 'true' (case-insensitive)
    2. MODERATE_NEAR_MISS_PILOT_START_DATE is parseable ISO date
    3. Current date <= start_date + duration_days

    Logs warning if flag enabled but start date missing/unparseable.
    Logs info ONCE per process lifecycle if pilot has expired (uses a
    module-level flag ``_pilot_expiration_logged`` to avoid log spam on
    every gate evaluation).
    """
    global _pilot_expiration_logged

    flag = os.environ.get("MODERATE_NEAR_MISS_PILOT", "")
    if flag.lower() != "true":
        return False

    # Flag is enabled — parse start date
    start_date_raw = os.environ.get("MODERATE_NEAR_MISS_PILOT_START_DATE", "")
    if not start_date_raw:
        logger.warning(
            "MODERATE_NEAR_MISS_PILOT is enabled but MODERATE_NEAR_MISS_PILOT_START_DATE "
            "is missing; treating pilot as disabled."
        )
        return False

    try:
        start_date = date.fromisoformat(start_date_raw)
    except (ValueError, TypeError):
        logger.warning(
            "MODERATE_NEAR_MISS_PILOT_START_DATE is unparseable ('%s'); "
            "treating pilot as disabled.",
            start_date_raw,
        )
        return False

    # Parse duration (default 7 days)
    duration_raw = os.environ.get("MODERATE_NEAR_MISS_PILOT_DURATION_DAYS", "7")
    try:
        duration_days = int(duration_raw)
    except (ValueError, TypeError):
        logger.warning(
            "MODERATE_NEAR_MISS_PILOT_DURATION_DAYS is not a valid integer ('%s'); "
            "using default of 7 days.",
            duration_raw,
        )
        duration_days = 7

    current_date = now if now is not None else date.today()
    expiration_date = start_date + timedelta(days=duration_days)

    if current_date <= expiration_date:
        return True

    # Pilot has expired — log once
    if not _pilot_expiration_logged:
        logger.info(
            "Moderate near-miss pilot has expired. start_date=%s, duration_days=%d, "
            "expiration_date=%s.",
            start_date.isoformat(),
            duration_days,
            expiration_date.isoformat(),
        )
        _pilot_expiration_logged = True

    return False
