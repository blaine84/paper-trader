"""
Setup Time Policy Registry — Single source of truth for setup-specific exit timing.

Defines per-setup-type alert/revalidate/force-close/extension timing used by
the Setup-Aware Lifecycle Evaluator. Each setup type maps to an immutable
SetupTimePolicy that governs how long a position may remain open and under
what conditions extensions are granted.

Thesis-development setups (news_breakout, news_catalyst, trend_pullback) have
revalidation windows and extension eligibility. Fast tactical setups
(momentum_fade, orb, short_squeeze, gap_and_go, vwap_reclaim) use shorter
timers with no extension path.
"""

from dataclasses import dataclass
from datetime import time


# ---------------------------------------------------------------------------
# SetupTimePolicy Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SetupTimePolicy:
    """Immutable policy for a single setup type's exit timing."""

    setup_type: str
    alert_minutes: int
    revalidate_minutes: int | None  # None for non-revalidation setups
    force_close_minutes: int
    extension_eligible: bool
    max_extension_minutes: int | None  # None when not extension-eligible
    revalidation_interval_minutes: int | None  # None when not extension-eligible
    eod_hard_wall: time  # Latest absolute close time (default 15:45 ET)
    fallback_behavior: str  # "reject" | "execute_without_extension"


# ---------------------------------------------------------------------------
# Thesis-Development Setup Types
# ---------------------------------------------------------------------------

THESIS_DEVELOPMENT_SETUPS = frozenset({
    "news_breakout",
    "news_catalyst",
    "trend_pullback",
})


# ---------------------------------------------------------------------------
# Setup Time Policy Registry
# ---------------------------------------------------------------------------

SETUP_TIME_POLICY_REGISTRY: dict[str, SetupTimePolicy] = {
    "news_breakout": SetupTimePolicy(
        setup_type="news_breakout",
        alert_minutes=60,
        revalidate_minutes=90,
        force_close_minutes=120,
        extension_eligible=True,
        max_extension_minutes=180,
        revalidation_interval_minutes=30,
        eod_hard_wall=time(15, 45),
        fallback_behavior="execute_without_extension",
    ),
    "news_catalyst": SetupTimePolicy(
        setup_type="news_catalyst",
        alert_minutes=60,
        revalidate_minutes=90,
        force_close_minutes=120,
        extension_eligible=True,
        max_extension_minutes=180,
        revalidation_interval_minutes=30,
        eod_hard_wall=time(15, 45),
        fallback_behavior="execute_without_extension",
    ),
    "trend_pullback": SetupTimePolicy(
        setup_type="trend_pullback",
        alert_minutes=90,
        revalidate_minutes=120,
        force_close_minutes=150,
        extension_eligible=True,
        max_extension_minutes=180,
        revalidation_interval_minutes=30,
        eod_hard_wall=time(15, 45),
        fallback_behavior="execute_without_extension",
    ),
    "momentum_fade": SetupTimePolicy(
        setup_type="momentum_fade",
        alert_minutes=35,
        revalidate_minutes=None,
        force_close_minutes=75,
        extension_eligible=False,
        max_extension_minutes=None,
        revalidation_interval_minutes=None,
        eod_hard_wall=time(15, 45),
        fallback_behavior="execute_without_extension",
    ),
    "orb": SetupTimePolicy(
        setup_type="orb",
        alert_minutes=45,
        revalidate_minutes=None,
        force_close_minutes=75,
        extension_eligible=False,
        max_extension_minutes=None,
        revalidation_interval_minutes=None,
        eod_hard_wall=time(15, 45),
        fallback_behavior="execute_without_extension",
    ),
    "short_squeeze": SetupTimePolicy(
        setup_type="short_squeeze",
        alert_minutes=30,
        revalidate_minutes=None,
        force_close_minutes=60,
        extension_eligible=False,
        max_extension_minutes=None,
        revalidation_interval_minutes=None,
        eod_hard_wall=time(15, 45),
        fallback_behavior="execute_without_extension",
    ),
    "gap_and_go": SetupTimePolicy(
        setup_type="gap_and_go",
        alert_minutes=60,
        revalidate_minutes=None,
        force_close_minutes=90,
        extension_eligible=False,
        max_extension_minutes=None,
        revalidation_interval_minutes=None,
        eod_hard_wall=time(15, 45),
        fallback_behavior="execute_without_extension",
    ),
    "vwap_reclaim": SetupTimePolicy(
        setup_type="vwap_reclaim",
        alert_minutes=60,
        revalidate_minutes=None,
        force_close_minutes=90,
        extension_eligible=False,
        max_extension_minutes=None,
        revalidation_interval_minutes=None,
        eod_hard_wall=time(15, 45),
        fallback_behavior="execute_without_extension",
    ),
}


# ---------------------------------------------------------------------------
# Default Policy (unknown/unrecognized setup types)
# ---------------------------------------------------------------------------

DEFAULT_POLICY = SetupTimePolicy(
    setup_type="unknown",
    alert_minutes=60,
    revalidate_minutes=None,
    force_close_minutes=90,
    extension_eligible=False,
    max_extension_minutes=None,
    revalidation_interval_minutes=None,
    eod_hard_wall=time(15, 45),
    fallback_behavior="execute_without_extension",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_policy(setup_type: str) -> SetupTimePolicy:
    """Look up policy for a setup type. Returns DEFAULT_POLICY for unknown types."""
    return SETUP_TIME_POLICY_REGISTRY.get(setup_type, DEFAULT_POLICY)


def is_thesis_development_setup(setup_type: str) -> bool:
    """Return True if setup_type is classified as thesis-development."""
    return setup_type in THESIS_DEVELOPMENT_SETUPS


def is_extension_eligible(setup_type: str) -> bool:
    """Return True if setup_type allows time extensions via revalidation."""
    policy = get_policy(setup_type)
    return policy.extension_eligible
