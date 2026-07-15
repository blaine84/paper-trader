"""Deterministic preflight validator for candidate execution prerequisites.

Evaluates whether a candidate can proceed to PM offering based on persisted
state and deterministic computation. No LLM judgment. Does not modify
candidate state in the registry (observational only).

See: design.md §Preflight Validator
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from utils.candidate_registry import CandidateRecord
from utils.gate_config import (
    CANDIDATE_EXECUTABLE_SETUP_TYPES,
    SWING_EXECUTABLE_SETUP_TYPES,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreflightSummary:
    """Deterministic preflight evaluation result for a single candidate."""

    candidate_id: str
    has_entry_stop_target: bool
    min_risk_reward_met: bool
    direction_valid: bool
    profile_allowed: bool
    candidate_not_expired: bool
    cash_available: bool
    sizing_possible: bool
    max_positions_available: bool
    same_symbol_allowed: bool
    blocking_reason_codes: list[str]

    @property
    def passed(self) -> bool:
        """True when no blocking reason codes exist."""
        return len(self.blocking_reason_codes) == 0


def compute_preflight(
    candidate: CandidateRecord,
    profile: dict,
    portfolio: dict,
    open_positions: list[dict],
    now_utc: datetime,
) -> PreflightSummary:
    """Evaluate deterministic execution prerequisites.

    Uses only persisted state and deterministic computation.
    No LLM judgment. Does not modify candidate state.

    Args:
        candidate: The CandidateRecord to evaluate.
        profile: Profile configuration dict (expects keys like
            "min_risk_reward", "max_positions").
        portfolio: Portfolio state dict (expects key "available_cash").
        open_positions: List of open position dicts (each with "symbol" key).
        now_utc: Current UTC datetime for expiration check.

    Returns:
        PreflightSummary with boolean check results and blocking reason codes.
    """
    blocking_reason_codes: list[str] = []

    # 1. has_entry_stop_target: all three geometry prices are non-null and non-zero
    has_entry_stop_target = (
        candidate.entry_price is not None
        and candidate.stop_price is not None
        and candidate.target_price is not None
        and candidate.entry_price != 0
        and candidate.stop_price != 0
        and candidate.target_price != 0
    )
    if not has_entry_stop_target:
        blocking_reason_codes.append("missing_geometry")

    # 2. min_risk_reward_met: candidate's risk_reward >= profile threshold (default 1.5)
    min_rr_threshold = profile.get("min_risk_reward", 1.5)
    candidate_rr = candidate.risk_reward if candidate.risk_reward is not None else 0.0
    min_risk_reward_met = candidate_rr >= min_rr_threshold
    if not min_risk_reward_met:
        blocking_reason_codes.append("min_risk_reward_not_met")

    # 3. direction_valid: direction is "BUY" or "SHORT"
    direction_valid = candidate.direction in ("BUY", "SHORT")
    if not direction_valid:
        blocking_reason_codes.append("invalid_direction")

    # 4. profile_allowed: setup_type is in executable setup types
    all_executable = CANDIDATE_EXECUTABLE_SETUP_TYPES | SWING_EXECUTABLE_SETUP_TYPES
    profile_allowed = candidate.setup_type in all_executable
    if not profile_allowed:
        blocking_reason_codes.append("profile_not_allowed")

    # 5. candidate_not_expired: expires_at > now_utc
    candidate_not_expired = candidate.expires_at > now_utc
    if not candidate_not_expired:
        blocking_reason_codes.append("candidate_expired")

    # 6. cash_available: portfolio's available cash > 0
    available_cash = portfolio.get("available_cash", 0)
    cash_available = available_cash > 0
    if not cash_available:
        blocking_reason_codes.append("insufficient_cash")

    # 7. sizing_possible: entry_price and stop_price allow non-zero quantity
    #    (dollar risk per share > 0, meaning entry and stop differ)
    if has_entry_stop_target:
        dollar_risk_per_share = abs(candidate.entry_price - candidate.stop_price)
        sizing_possible = dollar_risk_per_share > 0
    else:
        # If geometry is missing, sizing is impossible
        sizing_possible = False
    if not sizing_possible:
        blocking_reason_codes.append("sizing_impossible")

    # 8. max_positions_available: open position count < profile's max_positions
    max_positions = profile.get("max_positions", 10)
    max_positions_available = len(open_positions) < max_positions
    if not max_positions_available:
        blocking_reason_codes.append("max_positions_reached")

    # 9. same_symbol_allowed: no existing position for candidate's symbol
    open_symbols = {pos.get("symbol") for pos in open_positions}
    same_symbol_allowed = candidate.symbol not in open_symbols
    if not same_symbol_allowed:
        blocking_reason_codes.append("same_symbol_exists")

    return PreflightSummary(
        candidate_id=candidate.candidate_id,
        has_entry_stop_target=has_entry_stop_target,
        min_risk_reward_met=min_risk_reward_met,
        direction_valid=direction_valid,
        profile_allowed=profile_allowed,
        candidate_not_expired=candidate_not_expired,
        cash_available=cash_available,
        sizing_possible=sizing_possible,
        max_positions_available=max_positions_available,
        same_symbol_allowed=same_symbol_allowed,
        blocking_reason_codes=blocking_reason_codes,
    )


def _make_passing_preflight(candidate_id: str) -> PreflightSummary:
    """Create a PreflightSummary with all checks passing (fail-open default).

    Used when a database read error or computation error prevents normal
    preflight evaluation. Returns a summary that allows the candidate to
    proceed to PM offering without blocking.
    """
    return PreflightSummary(
        candidate_id=candidate_id,
        has_entry_stop_target=True,
        min_risk_reward_met=True,
        direction_valid=True,
        profile_allowed=True,
        candidate_not_expired=True,
        cash_available=True,
        sizing_possible=True,
        max_positions_available=True,
        same_symbol_allowed=True,
        blocking_reason_codes=[],
    )


def compute_preflight_safe(
    candidate: CandidateRecord,
    profile: dict,
    portfolio: dict,
    open_positions: list[dict],
    now_utc: datetime,
) -> PreflightSummary:
    """Fail-open wrapper around compute_preflight().

    On any exception (including database read errors propagated from the
    caller), logs ERROR and returns a passing PreflightSummary so that
    the candidate proceeds to PM offering without blocking the pipeline.

    Args:
        candidate: The CandidateRecord to evaluate.
        profile: Profile configuration dict.
        portfolio: Portfolio state dict.
        open_positions: List of open position dicts.
        now_utc: Current UTC datetime for expiration check.

    Returns:
        PreflightSummary — either the real evaluation or a passing summary
        on error.
    """
    try:
        return compute_preflight(candidate, profile, portfolio, open_positions, now_utc)
    except Exception:
        logger.error(
            "Preflight computation failed for %s, failing open",
            candidate.candidate_id,
            exc_info=True,
        )
        return _make_passing_preflight(candidate.candidate_id)
