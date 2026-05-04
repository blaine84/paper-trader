"""Shared constants for all trade safety gates.

Single source of truth for gate thresholds, symbol role sets, event type
mappings, and rejection reason codes.  All gate modules import from here
to prevent divergent defaults.

See: requirements.md §Default Configuration, design.md §utils/gate_config.py
"""

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

ROLLING_WINDOW: int = 5
MIN_CASES_FOR_BLOCK: int = 5
MIN_ROLLING_CASES: int = 3
CONSECUTIVE_LOSS_PAUSE_THRESHOLD: int = 3

# Recovery override
RECOVERY_MIN_ROLLING_CASES: int = 10
RECOVERY_WIN_RATE_MARGIN: float = 0.15
REQUIRE_POSITIVE_ROLLING_AVG_PNL_FOR_RECOVERY: bool = True

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
        "allow_pct_only_fallback": True,
        "atr_max_age_minutes": 15,
    },
    "etf_intraday": {
        "min_pct": 0.008,
        "atr_multiplier": 1.2,
        "min_reward_to_risk": 1.8,
        "allow_pct_only_fallback": True,
        "atr_max_age_minutes": 15,
    },
    "small_cap_momentum": {
        "min_pct": 0.025,
        "atr_multiplier": 2.0,
        "min_reward_to_risk": 2.5,
        "allow_pct_only_fallback": False,
        "atr_max_age_minutes": 10,
    },
}

DEFAULT_STOP_DISTANCE_RULE: dict = {
    "min_pct": 0.012,
    "atr_multiplier": 1.3,
    "min_reward_to_risk": 2.0,
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
    "risk_geometry_gate_evaluated": "risk_geometry_gate_evaluated",
}

# ---------------------------------------------------------------------------
# Rejection Reasons — canonical set of rejection category codes
# ---------------------------------------------------------------------------

REJECTION_REASONS: set[str] = {
    "setup_quality_gate",
    "pre_trade_quality_gate",
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
