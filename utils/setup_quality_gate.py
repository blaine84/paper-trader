"""Setup-type circuit breaker gate.

Evaluates whether a setup type should be allowed based on case-library
performance.  Uses a deterministic first-match-wins evaluation chain:

1. Insufficient data → allow
2. Consecutive losses → reject
3. Historical underperformance (with recovery override check) → reject / allow
4. Rolling underperformance → profile-aware (reduce_size / reject)
5. Weak but allowed → downgrade
6. Otherwise → allow

All thresholds are imported from ``utils/gate_config`` — no hardcoded
constants in this module.

See: requirements.md §1, design.md §utils/setup_quality_gate.py
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from models.case import Case
from db.schema import get_session
from utils.gate_config import (
    CONSECUTIVE_LOSS_PAUSE_THRESHOLD,
    DEFAULT_MIN_WIN_RATE,
    DEFAULT_MIN_WIN_RATE_BY_PROFILE,
    GATE_EVENT_TYPES,
    MIN_CASES_FOR_BLOCK,
    MIN_ROLLING_CASES,
    MIN_WIN_RATE_BY_SETUP,
    MIN_WIN_RATE_BY_SETUP_PROFILE,
    OVERRIDE_MIN_CONFIDENCE_SCORE,
    RECOVERY_MIN_ROLLING_CASES,
    RECOVERY_WIN_RATE_MARGIN,
    REQUIRE_POSITIVE_ROLLING_AVG_PNL_FOR_RECOVERY,
    ROLLING_RECOVERY_PROBE_SIZE_MULTIPLIER,
    ROLLING_WINDOW,
)
from utils.trade_events import log_trade_event

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_cases_for_setup(engine, setup_type: str) -> list[Any]:
    """Query all cases for *setup_type*, most-recent first.

    Returns a list of Case ORM objects.  On any query failure the caller
    receives an empty list so the gate falls through to the
    ``insufficient_data`` path.
    """
    try:
        session = get_session(engine)
        cases = (
            session.query(Case)
            .filter(Case.setup_type == setup_type)
            .order_by(Case.created_at.desc())
            .all()
        )
        session.close()
        return cases
    except Exception:
        log.exception("Failed to query cases for setup_type=%s", setup_type)
        return []


def _compute_rolling_stats(
    cases: list[Any], window: int
) -> tuple[float | None, float | None, int]:
    """Compute win rate and average PnL for the most recent *window* cases.

    Returns ``(rolling_win_rate, avg_pnl, rolling_sample_size)``.
    If there are no cases, returns ``(None, None, 0)``.
    """
    rolling = cases[:window]
    if not rolling:
        return None, None, 0

    wins = sum(1 for c in rolling if c.outcome == "success")
    rolling_wr = wins / len(rolling)

    pnl_values = [c.pnl_pct for c in rolling if c.pnl_pct is not None]
    avg_pnl = sum(pnl_values) / len(pnl_values) if pnl_values else 0.0

    return rolling_wr, avg_pnl, len(rolling)


def _check_consecutive_losses(cases: list[Any], threshold: int) -> bool:
    """Return ``True`` if the most recent *threshold* cases are all losses.

    A profitable partial outcome breaks a loss streak without being promoted
    to a win for win-rate calculations. Other non-success outcomes still count
    toward the consecutive-loss pause.
    """
    if len(cases) < threshold:
        return False
    return all(
        c.outcome != "success"
        and not (c.outcome == "partial" and c.pnl_pct is not None and c.pnl_pct > 0)
        for c in cases[:threshold]
    )


def _resolve_threshold(setup_type: str, profile: str | None) -> float:
    """Resolve setup-quality win-rate floor with profile-aware overrides."""
    profile_key = profile.lower() if isinstance(profile, str) else None
    setup_profile_thresholds = MIN_WIN_RATE_BY_SETUP_PROFILE.get(setup_type, {})
    if profile_key and profile_key in setup_profile_thresholds:
        return float(setup_profile_thresholds[profile_key])
    if profile_key and profile_key in DEFAULT_MIN_WIN_RATE_BY_PROFILE:
        return float(DEFAULT_MIN_WIN_RATE_BY_PROFILE[profile_key])
    return float(MIN_WIN_RATE_BY_SETUP.get(setup_type, DEFAULT_MIN_WIN_RATE))


def _check_recovery_override(
    all_time_wr: float,
    threshold: float,
    rolling_wr: float | None,
    avg_rolling_pnl: float | None,
    rolling_sample_size: int,
) -> bool:
    """Evaluate whether recovery override criteria are met.

    Recovery is granted when:
    - rolling sample size >= ``RECOVERY_MIN_ROLLING_CASES``
    - rolling win rate > threshold + ``RECOVERY_WIN_RATE_MARGIN``
    - avg rolling PnL > 0 (if ``REQUIRE_POSITIVE_ROLLING_AVG_PNL_FOR_RECOVERY``)
    """
    if rolling_sample_size < RECOVERY_MIN_ROLLING_CASES:
        return False
    if rolling_wr is None:
        return False
    if rolling_wr <= threshold + RECOVERY_WIN_RATE_MARGIN:
        return False
    if REQUIRE_POSITIVE_ROLLING_AVG_PNL_FOR_RECOVERY:
        if avg_rolling_pnl is None or avg_rolling_pnl <= 0:
            return False
    return True


def _evaluate_rolling_underperformance(
    profile_key: str | None,
    confidence_score: float | None,
    setup_type: str,
    rolling_wr: float,
    threshold: float,
    rolling_sample_size: int,
    win_rate: float | None,
    sample_size: int,
) -> dict:
    """Profile-aware rolling underperformance evaluation.

    Called only when:
    - All-time WR >= profile floor (step 3 passed)
    - Rolling WR < profile floor
    - Rolling sample >= MIN_ROLLING_CASES

    Returns a result dict with decision, reason_type, optional size_multiplier,
    and profile field. For moderate probes, includes override_confidence_score.
    For aggressive probes, does NOT include confidence_score (unused input).

    All branches include a gate_decision_id (UUID4) for correlation.
    """
    base = {
        "setup_type": setup_type,
        "win_rate": win_rate,
        "rolling_win_rate": rolling_wr,
        "sample_size": sample_size,
        "rolling_sample_size": rolling_sample_size,
        "threshold": threshold,
    }
    gate_decision_id = str(uuid.uuid4())

    if profile_key == "aggressive":
        return {
            **base,
            "decision": "reduce_size",
            "reason_type": "rolling_underperformance_recovery_probe",
            "size_multiplier": ROLLING_RECOVERY_PROBE_SIZE_MULTIPLIER,
            "profile": "aggressive",
            "gate_decision_id": gate_decision_id,
            "reason": (
                f"Setup '{setup_type}' rolling WR {rolling_wr:.1%} < {threshold:.0%} "
                f"over last {rolling_sample_size} cases; aggressive recovery probe "
                f"at {ROLLING_RECOVERY_PROBE_SIZE_MULTIPLIER:.0%} size."
            ),
        }

    elif profile_key == "moderate":
        if (
            confidence_score is not None
            and confidence_score >= OVERRIDE_MIN_CONFIDENCE_SCORE
        ):
            return {
                **base,
                "decision": "reduce_size",
                "reason_type": "rolling_underperformance_recovery_probe",
                "size_multiplier": ROLLING_RECOVERY_PROBE_SIZE_MULTIPLIER,
                "profile": "moderate",
                "override_confidence_score": confidence_score,
                "gate_decision_id": gate_decision_id,
                "reason": (
                    f"Setup '{setup_type}' rolling WR {rolling_wr:.1%} < {threshold:.0%} "
                    f"over last {rolling_sample_size} cases; moderate recovery probe "
                    f"at {ROLLING_RECOVERY_PROBE_SIZE_MULTIPLIER:.0%} size "
                    f"(override_confidence_score={confidence_score:.1f} >= "
                    f"{OVERRIDE_MIN_CONFIDENCE_SCORE})."
                ),
            }
        else:
            return {
                **base,
                "decision": "reject",
                "reason_type": "rolling_underperformance_confirmation_required",
                "profile": "moderate",
                "gate_decision_id": gate_decision_id,
                "reason": (
                    f"Setup '{setup_type}' rolling WR {rolling_wr:.1%} < {threshold:.0%}; "
                    f"moderate profile requires override_confidence_score >= "
                    f"{OVERRIDE_MIN_CONFIDENCE_SCORE} for recovery probe "
                    f"(got {confidence_score})."
                ),
            }

    elif profile_key == "conservative":
        return {
            **base,
            "decision": "reject",
            "reason_type": "rolling_underperformance_conservative_reject",
            "profile": "conservative",
            "gate_decision_id": gate_decision_id,
            "reason": (
                f"Setup '{setup_type}' rolling WR {rolling_wr:.1%} < {threshold:.0%} "
                f"over last {rolling_sample_size} cases; rejecting trade."
            ),
        }

    else:
        # Unknown/None profile — hard reject with existing reason_type
        return {
            **base,
            "decision": "reject",
            "reason_type": "rolling_underperformance",
            "gate_decision_id": gate_decision_id,
            "reason": (
                f"Setup '{setup_type}' rolling WR {rolling_wr:.1%} < {threshold:.0%} "
                f"over last {rolling_sample_size} cases; rejecting trade."
            ),
        }


def _log_gate_event(
    db,
    decision_dict: dict,
    *,
    symbol: str | None = None,
    profile: str | None = None,
    agent: str = "portfolio_manager",
) -> None:
    """Log a single TradeEvent for this gate evaluation.

    Uses ``GATE_EVENT_TYPES`` to map the decision to an ``event_type``.
    The full *decision_dict* is included in the payload alongside
    gate-specific audit fields.
    """
    decision = decision_dict.get("decision", "allow")
    event_type = GATE_EVENT_TYPES.get(decision, GATE_EVENT_TYPES["allow"])

    payload = {
        **decision_dict,
        "gate_name": "setup_quality_gate",
        "rejection_category": "setup_quality_gate",
        "reason_type": decision_dict.get("reason_type", ""),
    }

    log_trade_event(
        db,
        event_type,
        agent=agent,
        symbol=symbol,
        profile=profile,
        message=decision_dict.get("reason", ""),
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_setup_type(
    decision: dict | None, signal: dict | None
) -> str | None:
    """Resolve and normalize setup type from decision/signal.

    Priority order:
      1. decision.setup_type
      2. decision.setup
      3. signal.setup_type
      4. signal.setup

    Only string values are considered. Non-string values (int, float, list, dict,
    bool, etc.) are treated as missing — they are never coerced via str().

    Returns:
        Stripped non-empty string, or None if missing/blank/non-string.
    """
    candidates = []
    for source in (decision or {}, signal or {}):
        for key in ("setup_type", "setup"):
            val = source.get(key)
            if isinstance(val, str):
                stripped = val.strip()
                if stripped:
                    candidates.append(stripped)

    return candidates[0] if candidates else None


def evaluate_setup_quality(
    engine,
    db,
    setup_type: str,
    market_regime: str | None = None,
    *,
    symbol: str | None = None,
    profile: str | None = None,
    agent: str = "portfolio_manager",
    confidence_score: float | None = None,
) -> dict:
    """Evaluate whether a setup type should be allowed based on case-library
    performance.

    Evaluation order (first match wins):
    1. Insufficient data (< MIN_CASES_FOR_BLOCK) → allow
    2. Consecutive losses (>= CONSECUTIVE_LOSS_PAUSE_THRESHOLD) → reject
    3. Historical underperformance (all-time WR < threshold) → reject
       UNLESS recovery override criteria are met → allow
    4. Rolling underperformance (rolling WR < threshold, >= MIN_ROLLING_CASES) → profile-aware
    5. Weak but allowed (WR >= threshold but < 50%) → downgrade
    6. Otherwise → allow

    Args:
        engine: SQLAlchemy engine for case library queries
        db: SQLAlchemy session for trade event logging (caller owns commit)
        setup_type: The setup classification string
        market_regime: Optional current market regime
        symbol: Optional symbol for event logging context
        profile: Optional PM profile for event logging context
        agent: Agent name for event logging (default: "portfolio_manager")

    Returns:
        Structured dict with decision, reason_type, stats, and threshold.

    Side effects:
        Logs exactly one TradeEvent to the provided *db* session.
    """
    threshold = _resolve_threshold(setup_type, profile)

    # Fetch case history --------------------------------------------------
    cases = _get_cases_for_setup(engine, setup_type)
    sample_size = len(cases)

    # Compute stats -------------------------------------------------------
    if sample_size > 0:
        wins = sum(1 for c in cases if c.outcome == "success")
        win_rate: float | None = wins / sample_size
    else:
        win_rate = None

    rolling_wr, avg_rolling_pnl, rolling_sample_size = _compute_rolling_stats(
        cases, ROLLING_WINDOW
    )

    # Build base result dict (fields filled in by each branch) ------------
    def _result(
        decision: str,
        reason_type: str,
        reason: str,
    ) -> dict:
        return {
            "decision": decision,
            "reason_type": reason_type,
            "setup_type": setup_type,
            "win_rate": win_rate,
            "rolling_win_rate": rolling_wr,
            "sample_size": sample_size,
            "rolling_sample_size": rolling_sample_size,
            "threshold": threshold,
            "reason": reason,
        }

    # 1. Insufficient data ------------------------------------------------
    if sample_size < MIN_CASES_FOR_BLOCK:
        result = _result(
            "allow",
            "insufficient_data",
            f"Only {sample_size} cases for setup '{setup_type}' "
            f"(need {MIN_CASES_FOR_BLOCK}); allowing trade.",
        )
        _log_gate_event(db, result, symbol=symbol, profile=profile, agent=agent)
        return result

    # 2. Consecutive losses -----------------------------------------------
    if _check_consecutive_losses(cases, CONSECUTIVE_LOSS_PAUSE_THRESHOLD):
        result = _result(
            "reject",
            "consecutive_losses",
            f"Last {CONSECUTIVE_LOSS_PAUSE_THRESHOLD} cases for "
            f"'{setup_type}' are all losses; pausing setup.",
        )
        _log_gate_event(db, result, symbol=symbol, profile=profile, agent=agent)
        return result

    # 3. Historical underperformance (with recovery check) ----------------
    if win_rate is not None and win_rate < threshold:
        # Check recovery override
        if _check_recovery_override(
            win_rate, threshold, rolling_wr, avg_rolling_pnl, rolling_sample_size
        ):
            result = _result(
                "allow",
                "recovery_override",
                f"Setup '{setup_type}' all-time WR "
                f"{win_rate:.1%} < {threshold:.0%} but recovery criteria met "
                f"(rolling WR {rolling_wr:.1%}, avg PnL {avg_rolling_pnl:+.2f}%).",
            )
            _log_gate_event(db, result, symbol=symbol, profile=profile, agent=agent)
            return result

        result = _result(
            "reject",
            "historical_underperformance",
            f"Setup '{setup_type}' all-time WR "
            f"{win_rate:.1%} < {threshold:.0%}; rejecting trade.",
        )
        _log_gate_event(db, result, symbol=symbol, profile=profile, agent=agent)
        return result

    # 4. Rolling underperformance — PROFILE-AWARE
    if (
        rolling_wr is not None
        and rolling_sample_size >= MIN_ROLLING_CASES
        and rolling_wr < threshold
    ):
        result = _evaluate_rolling_underperformance(
            profile_key=(profile.lower() if isinstance(profile, str) else None),
            confidence_score=confidence_score,
            setup_type=setup_type,
            rolling_wr=rolling_wr,
            threshold=threshold,
            rolling_sample_size=rolling_sample_size,
            win_rate=win_rate,
            sample_size=sample_size,
        )
        _log_gate_event(db, result, symbol=symbol, profile=profile, agent=agent)
        return result

    # 5. Weak but allowed -------------------------------------------------
    if win_rate is not None and win_rate < 0.50:
        result = _result(
            "downgrade",
            "weak_but_allowed",
            f"Setup '{setup_type}' WR {win_rate:.1%} is above "
            f"block threshold ({threshold:.0%}) but below 50%; downgrading.",
        )
        _log_gate_event(db, result, symbol=symbol, profile=profile, agent=agent)
        return result

    # 6. Pass — all checks cleared ----------------------------------------
    result = _result(
        "allow",
        "pass",
        f"Setup '{setup_type}' passes all quality checks "
        f"(WR {win_rate:.1%}, {sample_size} cases).",
    )
    _log_gate_event(db, result, symbol=symbol, profile=profile, agent=agent)
    return result
