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
from decimal import Decimal

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
    "vwap_reclaim": {
        "conservative": 0.35,
        "moderate": 0.30,
        "aggressive": 0.20,
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
    # ── Pending limit order cancellations ──
    # Added alongside rather than reusing price_target_missed /
    # signal_invalidated / timeout_expired, so pending-order outcomes stay
    # distinguishable from triggered-plan outcomes in queries.
    "signal_flipped",
    "superseded",
    "position_already_open",
    "cooldown_active",
    "insufficient_buying_power",
    "gap_through",
    "sizing_rejected",
    # ── Pending limit order creation declines ──
    "target_already_exceeded",
    "runaway_exceeds_max",
    "incomplete_geometry",
    "invalid_geometry_at_limit",
    "window_too_short",
    "active_order_cap_reached",
    "duplicate_active_order",
    "active_trade_plan_exists",
    "repaired_before_check",
    # ── Pending limit order fill-path observation ──
    "stale_fill_bar",
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

# When false, candidate shadow mode records candidate-path telemetry and skips
# the expensive legacy freeform PM entry call. Set true only for deliberate
# short A/B comparison windows.
PM_SHADOW_RUN_LEGACY_ENTRY: bool = os.environ.get(
    "PM_SHADOW_RUN_LEGACY_ENTRY", "false"
).lower() == "true"

# Controls whether missing rejection codes produce violations or just warnings.
# Values: "warn" | "enforcing"
PM_REJECTION_CODE_MODE: str = os.environ.get("PM_REJECTION_CODE_MODE", "warn")

# Controls whether preflight-failed candidates are excluded or shown in observe mode.
# Values: "disabled" | "observe" | "enabled"
PM_PREFLIGHT_OBSERVE_MODE: str = os.environ.get("PM_PREFLIGHT_OBSERVE_MODE", "disabled")

# Closed set of setup types eligible for candidate-ID pipeline execution.
# Only these types will be offered to PM in candidate mode.
# Other setup types may still flow through legacy entry or shadow mode.
CANDIDATE_EXECUTABLE_SETUP_TYPES: frozenset[str] = frozenset({
    "momentum_fade",
    "news_breakout",
    "gap_and_go",
    "technical_breakout",
    "vwap_reclaim",
})

# P1 Benchmark Context (independent of P0)
PM_BENCHMARK_CONTEXT_ENABLED: bool = os.environ.get(
    "PM_BENCHMARK_CONTEXT_ENABLED", "false"
).lower() == "true"

# P1 Alignment Policy
# Values: "disabled" | "log_only" | "enforcing"
PM_ALIGNMENT_POLICY_MODE: str = os.environ.get("PM_ALIGNMENT_POLICY_MODE", "disabled")

# ---------------------------------------------------------------------------
# Market Data Reliability Layer Feature Flag
# ---------------------------------------------------------------------------

# Values: "disabled" | "observe" | "enforcing"
MARKET_DATA_RELIABILITY_MODE: str = os.environ.get("MARKET_DATA_RELIABILITY_MODE", "disabled")

# ---------------------------------------------------------------------------
# PM Decision Provenance Feature Flags
# ---------------------------------------------------------------------------

# Values: "disabled" | "observe" | "enforcing"
PM_PROVENANCE_MODE: str = os.environ.get("PM_PROVENANCE_MODE", "disabled")

# Controls detail level for provenance payload storage.
# "full" persists raw response body, full input bundles, and all geometry snapshots.
# "minimal" disables nonessential payload detail while preserving deterministic
# geometry validation, first-invalid-stage attribution, and coverage metrics.
# Values: "full" | "minimal"
PM_PROVENANCE_DETAIL: str = os.environ.get("PM_PROVENANCE_DETAIL", "full")

# Maximum allowed end-to-end latency (in milliseconds) for provenance persistence
# within the market-hours PM candidate processing cycle. Actual added latency is
# recorded per candidate for monitoring.
PM_PROVENANCE_LATENCY_BUDGET_MS: int = int(
    os.environ.get("PM_PROVENANCE_LATENCY_BUDGET_MS", "200")
)


# ---------------------------------------------------------------------------
# LLM Queue and Backpressure Feature Flag
# ---------------------------------------------------------------------------

# Values: "disabled" | "observe" | "enforcing"
LLM_QUEUE_MODE: str = os.environ.get("LLM_QUEUE_MODE", "disabled")

# Startup log reporting active mode
if LLM_QUEUE_MODE != "disabled":
    logger.info(
        "LLM Queue Mode: %s (concurrency=%s, max_queue=%s)",
        LLM_QUEUE_MODE,
        os.environ.get("LLM_QUEUE_GLOBAL_CONCURRENCY", "1"),
        os.environ.get("LLM_QUEUE_MAX_SIZE", "10"),
    )


# ---------------------------------------------------------------------------
# Market State Trigger Contract Feature Flag
# ---------------------------------------------------------------------------

# Values: "disabled" | "observe" | "enforcing"
_raw_market_state_mode = os.environ.get("MARKET_STATE_MODE", "disabled")
if _raw_market_state_mode not in ("disabled", "observe", "enforcing"):
    logger.warning(
        "Unrecognized MARKET_STATE_MODE=%r, defaulting to 'disabled'",
        _raw_market_state_mode,
    )
    _raw_market_state_mode = "disabled"
MARKET_STATE_MODE: str = _raw_market_state_mode

# ---------------------------------------------------------------------------
# Watch Candidate Hardening Constants
# ---------------------------------------------------------------------------

# Key-level drift threshold (percentage). Active watches with support/resistance
# drift exceeding this value are structurally invalidated.
# Default: 2.0% (tighter than a naive 5% — catches meaningful structural shifts
# on intraday key levels where 2% already represents a broken level).
WATCH_KEY_LEVEL_DRIFT_PCT: float = float(
    os.environ.get("WATCH_KEY_LEVEL_DRIFT_PCT", "2.0")
)

# Same-cycle promotion policy.
# Values: "never" | "activation_pending_only" | "always"
_raw_same_cycle_policy = os.environ.get(
    "WATCH_SAME_CYCLE_PROMOTION_POLICY", "activation_pending_only"
)
if _raw_same_cycle_policy not in ("never", "activation_pending_only", "always"):
    logger.warning(
        "Unrecognized WATCH_SAME_CYCLE_PROMOTION_POLICY=%r, defaulting to 'activation_pending_only'",
        _raw_same_cycle_policy,
    )
    _raw_same_cycle_policy = "activation_pending_only"
WATCH_SAME_CYCLE_PROMOTION_POLICY: str = _raw_same_cycle_policy

# ---------------------------------------------------------------------------
# Swing Candidate Pipeline Feature Flags
# ---------------------------------------------------------------------------

# Values: "disabled" | "observe" | "enabled"
_raw_swing_mode = os.environ.get("SWING_CANDIDATE_MODE", "disabled")
if _raw_swing_mode not in ("disabled", "observe", "enabled"):
    logger.warning(
        "Unrecognized SWING_CANDIDATE_MODE=%r, defaulting to 'disabled'",
        _raw_swing_mode,
    )
    _raw_swing_mode = "disabled"
SWING_CANDIDATE_MODE: str = _raw_swing_mode


def get_swing_candidate_mode() -> str:
    """Return current SWING_CANDIDATE_MODE value.

    Exposed as a function so callers (e.g., swing_candidate_bridge) can
    read the flag at call time rather than import time, making tests simpler
    (patch this function instead of reloading the module).
    """
    return SWING_CANDIDATE_MODE


# Closed set of executable swing setup types
SWING_EXECUTABLE_SETUP_TYPES: frozenset[str] = frozenset({
    "sector_rotation_swing",
    "risk_off_macro_short",
    "breakout_retest",
    "pullback_continuation",
    "relative_strength_swing",
    "support_bounce_swing",
    "failed_breakdown_reclaim",
})

# Per-profile swing policy configuration
SWING_PROFILE_POLICY: dict[str, dict] = {
    "conservative": {
        "min_confidence": "high",
        "min_strength": "strong",
        "min_risk_reward": Decimal("3.0"),
        "sizing_multiplier": Decimal("0.5"),
    },
    "moderate": {
        "min_confidence": "medium",
        "min_strength": "moderate",
        "min_risk_reward": Decimal("1.5"),
        "sizing_multiplier": Decimal("0.5"),
    },
    "aggressive": {
        "min_confidence": "low",
        "min_strength": "moderate",
        "min_risk_reward": Decimal("1.25"),
        "sizing_multiplier": Decimal("1.0"),
    },
}

# Maximum concurrent swing positions per profile
SWING_MAX_CONCURRENT_POSITIONS: dict[str, int] = {
    "conservative": 2,
    "moderate": 4,
    "aggressive": 6,
}

# Swing candidate expiration
SWING_MAX_CANDIDATE_AGE_HOURS: int = int(
    os.environ.get("SWING_MAX_CANDIDATE_AGE_HOURS", "24")
)

# Price deviation threshold for expiration (percentage)
SWING_PRICE_DEVIATION_THRESHOLD_PCT: Decimal = Decimal(
    os.environ.get("SWING_PRICE_DEVIATION_THRESHOLD_PCT", "3.0")
)

# Sector concentration warning threshold
SWING_SECTOR_CONCENTRATION_WARN_THRESHOLD: int = int(
    os.environ.get("SWING_SECTOR_CONCENTRATION_WARN_THRESHOLD", "3")
)

# Conservative observe-only flag
SWING_CONSERVATIVE_OBSERVE_ONLY: bool = os.environ.get(
    "SWING_CONSERVATIVE_OBSERVE_ONLY", "false"
).lower() == "true"

# ---------------------------------------------------------------------------
# Swing Freshness Thresholds
# ---------------------------------------------------------------------------

# Signal freshness: maximum age in hours before a signal is considered stale.
# Read from the SWING_SIGNAL_FRESHNESS_HOURS environment variable.
# Bounded to [1, 168] hours (1 hour minimum, 1 week maximum).
# Default: 24 hours (matches SWING_MAX_CANDIDATE_AGE_HOURS).
_raw_signal_freshness = os.environ.get("SWING_SIGNAL_FRESHNESS_HOURS", "24")
try:
    _signal_freshness_val = int(_raw_signal_freshness)
except (ValueError, TypeError):
    logger.warning(
        "SWING_SIGNAL_FRESHNESS_HOURS has non-numeric value '%s'; using default 24.",
        _raw_signal_freshness,
    )
    _signal_freshness_val = 24

SWING_SIGNAL_FRESHNESS_HOURS: int = max(1, min(168, _signal_freshness_val))

# ---------------------------------------------------------------------------
# Price Alert PM Dispatcher
# ---------------------------------------------------------------------------

# Values: "disabled" | "observe" | "enabled"
PM_ALERT_DISPATCH_MODE: str = os.environ.get("PM_ALERT_DISPATCH_MODE", "disabled")

# Cooldown configuration
PM_ALERT_SYMBOL_COOLDOWN_MINUTES: int = int(
    os.environ.get("PM_ALERT_SYMBOL_COOLDOWN_MINUTES", "15")
)
PM_ALERT_GLOBAL_COOLDOWN_MINUTES: int = int(
    os.environ.get("PM_ALERT_GLOBAL_COOLDOWN_MINUTES", "10")
)
PM_ALERT_DISPATCHER_INTERVAL_SECONDS: int = int(
    os.environ.get("PM_ALERT_DISPATCHER_INTERVAL_SECONDS", "30")
)

# Classification batch limits
PM_ALERT_CLASSIFY_MAX_PER_PASS: int = int(
    os.environ.get("PM_ALERT_CLASSIFY_MAX_PER_PASS", "5")
)
PM_ALERT_CLASSIFY_TIMEOUT_SECONDS: int = int(
    os.environ.get("PM_ALERT_CLASSIFY_TIMEOUT_SECONDS", "10")
)

# Crash recovery: stale dispatch timeout
PM_ALERT_DISPATCH_STALE_MINUTES: int = int(
    os.environ.get("PM_ALERT_DISPATCH_STALE_MINUTES", "10")
)
PM_ALERT_SCHEDULED_MAX_RUNTIME_MINUTES: int = int(
    os.environ.get("PM_ALERT_SCHEDULED_MAX_RUNTIME_MINUTES", "15")
)

# ---------------------------------------------------------------------------
# Per-Alert-Type Dispatch Mode Overrides
# ---------------------------------------------------------------------------

# Per-alert-type dispatch mode overrides.
# Values: "dispatch" | "observe" | "disabled" | "" (fall back to global mode)
PM_ALERT_MODE_ENTRY_ALERT: str = os.environ.get("PM_ALERT_MODE_ENTRY_ALERT", "")
PM_ALERT_MODE_BREAKOUT: str = os.environ.get("PM_ALERT_MODE_BREAKOUT", "")
PM_ALERT_MODE_RAPID_MOVE: str = os.environ.get("PM_ALERT_MODE_RAPID_MOVE", "")
PM_ALERT_MODE_TARGET_HIT: str = os.environ.get("PM_ALERT_MODE_TARGET_HIT", "")

# ---------------------------------------------------------------------------
# Alert Material Occurrence Feature Flag
# ---------------------------------------------------------------------------

# Values: "disabled" | "enabled"
# Controls which counter the dispatcher uses for deferral/dedup decisions.
# "disabled": occurrence_count drives decisions (existing behavior)
# "enabled": material_occurrence_count drives decisions (fixed behavior)
#
# Invariant (both modes):
#   - occurrence_count always increments on every upsert
#   - material_occurrence_count always populates best-effort
#   - Only dispatcher READS are gated by this flag
ALERT_MATERIAL_OCCURRENCE_MODE: str = os.environ.get(
    "ALERT_MATERIAL_OCCURRENCE_MODE", "disabled"
)

# Material price change threshold (decimal fraction, e.g. 0.005 = 0.5%)
# Price changes at or below this threshold are NOT material.
ALERT_MATERIAL_PRICE_THRESHOLD_PCT: float = float(
    os.environ.get("ALERT_MATERIAL_PRICE_THRESHOLD_PCT", "0.005")
)


# ---------------------------------------------------------------------------
# Alert Freshness Limits
# ---------------------------------------------------------------------------


def _clamp_freshness(raw_value: str, alert_type: str) -> int:
    """Parse and clamp a freshness value to [1, 120] minutes.

    Args:
        raw_value: The raw string value from the environment variable.
        alert_type: The alert type name (for warning messages).

    Returns:
        The clamped integer value in the range [1, 120].
    """
    try:
        value = int(raw_value)
    except (ValueError, TypeError):
        logger.warning(
            "PM_ALERT_FRESHNESS_%s_MINUTES has non-numeric value '%s'; "
            "using default of 15 minutes.",
            alert_type,
            raw_value,
        )
        value = 15

    if value < 1:
        logger.warning(
            "PM_ALERT_FRESHNESS_%s_MINUTES value %d is below minimum; "
            "clamping to 1 minute.",
            alert_type,
            value,
        )
        return 1
    if value > 120:
        logger.warning(
            "PM_ALERT_FRESHNESS_%s_MINUTES value %d exceeds maximum; "
            "clamping to 120 minutes.",
            alert_type,
            value,
        )
        return 120

    return value


# Freshness limits per alert type (minutes), clamped to [1, 120]
PM_ALERT_FRESHNESS_ENTRY_ALERT_MINUTES: int = _clamp_freshness(
    os.environ.get("PM_ALERT_FRESHNESS_ENTRY_ALERT_MINUTES", "15"), "ENTRY_ALERT"
)
PM_ALERT_FRESHNESS_BREAKOUT_MINUTES: int = _clamp_freshness(
    os.environ.get("PM_ALERT_FRESHNESS_BREAKOUT_MINUTES", "10"), "BREAKOUT"
)
PM_ALERT_FRESHNESS_RAPID_MOVE_MINUTES: int = _clamp_freshness(
    os.environ.get("PM_ALERT_FRESHNESS_RAPID_MOVE_MINUTES", "5"), "RAPID_MOVE"
)

# Stale claim recovery timeout (minutes)
PM_ALERT_CLAIM_STALE_MINUTES: int = int(
    os.environ.get("PM_ALERT_CLAIM_STALE_MINUTES", "10")
)

# ---------------------------------------------------------------------------
# Coordinated Market Cycle Scheduler Feature Flags
# ---------------------------------------------------------------------------

# Values: "disabled" | "enabled"
PM_CYCLE_COORDINATOR_MODE: str = os.environ.get("PM_CYCLE_COORDINATOR_MODE", "disabled")

# Phase timeout budgets (seconds)
CYCLE_ANALYST_TIMEOUT_SECONDS: int = int(
    os.environ.get("CYCLE_ANALYST_TIMEOUT_SECONDS", "180")
)
CYCLE_PM_TIMEOUT_SECONDS: int = int(
    os.environ.get("CYCLE_PM_TIMEOUT_SECONDS", "300")
)

# Signal freshness window — signals older than this are excluded from the PM phase
PM_SIGNAL_FRESHNESS_WINDOW_SECONDS: int = int(
    os.environ.get("PM_SIGNAL_FRESHNESS_WINDOW_SECONDS", "120")
)

# Decision window — maximum seconds PM has to render a decision after receiving candidates
CYCLE_DECISION_WINDOW_SECONDS: int = int(
    os.environ.get("CYCLE_DECISION_WINDOW_SECONDS", "90")
)

# P1: Market snapshot TTL — cached snapshot expires after this many seconds
CYCLE_SNAPSHOT_TTL_SECONDS: int = int(
    os.environ.get("CYCLE_SNAPSHOT_TTL_SECONDS", "90")
)

# P1: Finnhub API call budget per cycle
CYCLE_FINNHUB_BUDGET: int = int(
    os.environ.get("CYCLE_FINNHUB_BUDGET", "40")
)

# ---------------------------------------------------------------------------
# Triggered Trade Plans Feature Flags
# ---------------------------------------------------------------------------


def _int_env(name: str, default: int, minimum: int | None = None) -> int:
    """Read an int env var, falling back to default on non-numeric input.

    Args:
        name: Environment variable name.
        default: Value used when unset or unparseable.
        minimum: Optional inclusive lower bound; values below are clamped.
    """
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except (ValueError, TypeError):
        logger.warning(
            "%s has non-numeric value '%s'; using default %d.", name, raw, default
        )
        value = default
    if minimum is not None and value < minimum:
        logger.warning(
            "%s value %d is below minimum %d; clamping.", name, value, minimum
        )
        value = minimum
    return value


def _float_env(name: str, default: float, minimum: float | None = None) -> float:
    """Read a float env var, falling back to default on non-numeric input."""
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except (ValueError, TypeError):
        logger.warning(
            "%s has non-numeric value '%s'; using default %s.", name, raw, default
        )
        value = default
    if minimum is not None and value < minimum:
        logger.warning(
            "%s value %s is below minimum %s; clamping.", name, value, minimum
        )
        value = minimum
    return value


# Values: "disabled" | "observe" | "enabled"
# disabled: existing single-pass candidate pipeline unchanged (zero behavior change)
# observe:  trade plans created and trigger evaluations logged, execution unchanged
# enabled:  PM acceptance creates plans; fills require plan trigger confirmation
_raw_triggered_plan_mode = os.environ.get("TRIGGERED_PLAN_MODE", "disabled")
if _raw_triggered_plan_mode not in ("disabled", "observe", "enabled"):
    logger.warning(
        "Unrecognized TRIGGERED_PLAN_MODE=%r, defaulting to 'disabled'",
        _raw_triggered_plan_mode,
    )
    _raw_triggered_plan_mode = "disabled"
TRIGGERED_PLAN_MODE: str = _raw_triggered_plan_mode

# Plan monitor cadence (seconds) — independent of PM cycle scheduling.
PLAN_MONITOR_INTERVAL_SECONDS: int = _int_env(
    "PLAN_MONITOR_INTERVAL_SECONDS", 30, minimum=1
)

# Default plan TTL (minutes) — untriggered plans expire after this window.
PLAN_DEFAULT_EXPIRATION_MINUTES: int = _int_env(
    "PLAN_DEFAULT_EXPIRATION_MINUTES", 60, minimum=1
)

# Fractional tolerance applied to entry zone bounds at trigger evaluation time
# only (never baked into derived zone bounds). 0.005 == 0.5% of entry reference.
PLAN_ENTRY_ZONE_TOLERANCE_PCT: float = _float_env(
    "PLAN_ENTRY_ZONE_TOLERANCE_PCT", 0.005, minimum=0.0
)

# Consecutive in-zone monitor ticks required when a plan demands confirmation.
PLAN_TRIGGER_CONFIRMATION_TICKS: int = _int_env(
    "PLAN_TRIGGER_CONFIRMATION_TICKS", 2, minimum=1
)

# Maximum acceptable quote age (seconds) for plan execution. Older quotes are
# treated as unavailable — execution is fail-closed on quote freshness.
PLAN_EXECUTION_MAX_QUOTE_AGE_SECONDS: int = _int_env(
    "PLAN_EXECUTION_MAX_QUOTE_AGE_SECONDS", 5, minimum=1
)

# Maximum acceptable quote age (seconds) for trigger evaluation. Cached quotes
# within this age are used instead of calling a provider.
PLAN_TRIGGER_QUOTE_MAX_AGE_SECONDS: int = _int_env(
    "PLAN_TRIGGER_QUOTE_MAX_AGE_SECONDS", 30, minimum=1
)

# Minimum interval (seconds) between provider calls for any single symbol,
# enforced across monitor ticks.
QUOTE_PROVIDER_MIN_SECONDS_PER_SYMBOL: int = _int_env(
    "QUOTE_PROVIDER_MIN_SECONDS_PER_SYMBOL", 30, minimum=0
)

# Global cap on outbound quote provider calls per rolling minute from the
# plan monitor / plan executor.
QUOTE_PROVIDER_MAX_CALLS_PER_MINUTE: int = _int_env(
    "QUOTE_PROVIDER_MAX_CALLS_PER_MINUTE", 40, minimum=1
)

# Startup log reporting active mode
if TRIGGERED_PLAN_MODE != "disabled":
    logger.info(
        "Triggered Plan Mode: %s (monitor_interval=%ss, plan_ttl=%smin, "
        "zone_tolerance=%s, confirmation_ticks=%s)",
        TRIGGERED_PLAN_MODE,
        PLAN_MONITOR_INTERVAL_SECONDS,
        PLAN_DEFAULT_EXPIRATION_MINUTES,
        PLAN_ENTRY_ZONE_TOLERANCE_PCT,
        PLAN_TRIGGER_CONFIRMATION_TICKS,
    )


# ---------------------------------------------------------------------------
# Pending Limit Orders Feature Flags
#
# Converts the runaway-entry branch of _fresh_price_stale_entry_check() into a
# resting paper limit order instead of discarding the intent. Independent of
# TRIGGERED_PLAN_MODE by design — pending orders must be rollable without
# enabling the (currently dormant) triggered-plan subsystem.
# ---------------------------------------------------------------------------

# Values: "disabled" | "observe" | "enabled"
# disabled: execute_trade() behavior unchanged; no orders, no monitor job
# observe:  orders created and evaluated; fills are logged, never executed
# enabled:  detected crossings execute through execute_trade() after full
#           fill-time revalidation
_raw_pending_order_mode = os.environ.get("PENDING_ORDER_MODE", "disabled")
if _raw_pending_order_mode not in ("disabled", "observe", "enabled"):
    logger.warning(
        "Unrecognized PENDING_ORDER_MODE=%r, defaulting to 'disabled'",
        _raw_pending_order_mode,
    )
    _raw_pending_order_mode = "disabled"
PENDING_ORDER_MODE: str = _raw_pending_order_mode

# Monitor cadence (seconds) — independent of PM cycles. Bars are 1-minute by
# default, so polling faster than the bar interval gains nothing.
PENDING_ORDER_MONITOR_INTERVAL_SECONDS: int = _int_env(
    "PENDING_ORDER_MONITOR_INTERVAL_SECONDS", 60, minimum=15
)

# Fallback active-window length (minutes) for setup types absent from
# PENDING_ORDER_EXPIRY_MINUTES_BY_SETUP. Always clamped to the session close.
PENDING_ORDER_DEFAULT_EXPIRY_MINUTES: int = _int_env(
    "PENDING_ORDER_DEFAULT_EXPIRY_MINUTES", 120, minimum=5
)

# Upper bound on how far price may have run beyond the intended entry and still
# justify resting an order.
#
# NOTE: non-binding under current execute_trade() ordering. The live-quote
# deviation tiers run BEFORE _fresh_price_stale_entry_check(): Tier 2 (>5%,
# <=10%) overwrites price with the live quote and rewrites the geometry back
# into the decision dict, and Tier 3 (>10%) returns outright. So the
# runaway-entry branch only ever observes roughly the 1%-5% range and nothing
# above ~5% reaches this guard. It becomes meaningful only if
# PENDING_ORDER_DIVERT_REPAIR_BAND is implemented.
PENDING_ORDER_MAX_RUNAWAY_PCT: float = _float_env(
    "PENDING_ORDER_MAX_RUNAWAY_PCT", 0.05, minimum=0.0
)

# Gap-through threshold. When the crossing bar's OPEN is already beyond the
# limit by more than this fraction, the market gapped past the level rather
# than trading down to it, which invalidates the stop and target derived from
# the pre-gap structure. Such orders cancel rather than fill.
PENDING_ORDER_MAX_GAP_THROUGH_PCT: float = _float_env(
    "PENDING_ORDER_MAX_GAP_THROUGH_PCT", 0.015, minimum=0.0
)

# Cap on concurrent active orders per profile. Buying power is NOT reserved at
# creation (it is re-checked at fill), so this cap is what bounds the resulting
# overcommitment risk.
PENDING_ORDER_MAX_ACTIVE_PER_PROFILE: int = _int_env(
    "PENDING_ORDER_MAX_ACTIVE_PER_PROFILE", 5, minimum=1
)

# OHLC resolution used for crossing detection. Passed straight to
# FinnhubClient.get_candles(), which routes Alpaca -> yfinance -> Finnhub for
# sub-daily resolutions.
PENDING_ORDER_BAR_RESOLUTION: str = os.environ.get(
    "PENDING_ORDER_BAR_RESOLUTION", "1"
)

# Maximum age of a crossing bar at the moment of the fill attempt. This is the
# compensating control for bypassing execute_trade()'s deviation tiers on the
# price-authoritative fill path: it is the only remaining guarantee that the
# fill price reflects recent market reality. A slow tick, a restart, or an
# order backlog can all surface a bar several minutes old.
PENDING_ORDER_MAX_FILL_BAR_AGE_SECONDS: int = _int_env(
    "PENDING_ORDER_MAX_FILL_BAR_AGE_SECONDS", 180, minimum=30
)

# Phase 3 lever, deliberately unimplemented in v1. Would divert Repair_Band
# (>5%, <=10% deviation) decisions to a pending order at the original intended
# entry instead of repairing to the live price and executing. That is a
# behavior change to live execution rather than an additive one, so it needs
# the pending_order_declined(repaired_before_check) evidence first.
PENDING_ORDER_DIVERT_REPAIR_BAND: bool = os.environ.get(
    "PENDING_ORDER_DIVERT_REPAIR_BAND", "false"
).lower() == "true"

# Setup-specific active-window lengths (minutes). Biased short because
# intraday setup premises decay fast. Absent setup types fall back to
# PENDING_ORDER_DEFAULT_EXPIRY_MINUTES. All values are further clamped by
# ENTRY_WINDOW_LIMITS and by the regular session close.
PENDING_ORDER_EXPIRY_MINUTES_BY_SETUP: dict[str, int] = {
    "gap_and_go": 30,
    "orb": 30,
    "short_squeeze": 30,
    "momentum_fade": 45,
    "vwap_reclaim": 60,
    "news_breakout": 60,
    "technical_breakout": 120,
}

# Canonical pending-order event vocabulary. Shared so that web/app.py filters
# against this constant instead of a second hardcoded list.
PENDING_ORDER_EVENT_TYPES: frozenset[str] = frozenset({
    "pending_order_created",
    "pending_order_filled",
    "pending_order_expired",
    "pending_order_canceled",
    "pending_order_rejected",
    "pending_order_declined",
    "pending_order_would_fill",
})

# Startup log reporting active mode
if PENDING_ORDER_MODE != "disabled":
    logger.info(
        "Pending Limit Order Mode: %s (monitor_interval=%ss, bar_resolution=%s, "
        "default_expiry=%smin, max_active_per_profile=%s, max_fill_bar_age=%ss)",
        PENDING_ORDER_MODE,
        PENDING_ORDER_MONITOR_INTERVAL_SECONDS,
        PENDING_ORDER_BAR_RESOLUTION,
        PENDING_ORDER_DEFAULT_EXPIRY_MINUTES,
        PENDING_ORDER_MAX_ACTIVE_PER_PROFILE,
        PENDING_ORDER_MAX_FILL_BAR_AGE_SECONDS,
    )


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
