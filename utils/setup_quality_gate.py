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
    CONSECUTIVE_LOSS_PAUSE_EXEMPT_SETUPS,
    CONSECUTIVE_LOSS_PAUSE_THRESHOLD,
    DEFAULT_MIN_WIN_RATE,
    DEFAULT_MIN_WIN_RATE_BY_PROFILE,
    GATE_EVENT_TYPES,
    MIN_CASES_FOR_BLOCK,
    MIN_ROLLING_CASES,
    MIN_WIN_RATE_BY_SETUP,
    MIN_WIN_RATE_BY_SETUP_PROFILE,
    NEAR_MISS_MARGIN_PCT,
    OVERRIDE_MIN_CONFIDENCE_SCORE,
    RECOVERY_MIN_ROLLING_CASES,
    RECOVERY_WIN_RATE_MARGIN,
    REQUIRE_POSITIVE_ROLLING_AVG_PNL_FOR_RECOVERY,
    ROLLING_RECOVERY_PROBE_SIZE_MULTIPLIER,
    ROLLING_WINDOW,
    is_moderate_near_miss_pilot_active,
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
    id_provider=None,
    override_min_confidence: float | None = None,
    probe_size_multiplier: float | None = None,
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
    gate_decision_id = id_provider() if id_provider is not None else str(uuid.uuid4())
    p_override_min_confidence = override_min_confidence if override_min_confidence is not None else OVERRIDE_MIN_CONFIDENCE_SCORE
    p_probe_size_mult = probe_size_multiplier if probe_size_multiplier is not None else ROLLING_RECOVERY_PROBE_SIZE_MULTIPLIER

    if profile_key == "aggressive":
        return {
            **base,
            "decision": "reduce_size",
            "canonical_decision": "reduce_size",
            "reason_type": "rolling_underperformance_recovery_probe",
            "size_multiplier": p_probe_size_mult,
            "profile": "aggressive",
            "gate_decision_id": gate_decision_id,
            "reason": (
                f"Setup '{setup_type}' rolling WR {rolling_wr:.1%} < {threshold:.0%} "
                f"over last {rolling_sample_size} cases; aggressive recovery probe "
                f"at {p_probe_size_mult:.0%} size."
            ),
        }

    elif profile_key == "moderate":
        if (
            confidence_score is not None
            and confidence_score >= p_override_min_confidence
        ):
            return {
                **base,
                "decision": "reduce_size",
                "canonical_decision": "reduce_size",
                "reason_type": "rolling_underperformance_recovery_probe",
                "size_multiplier": p_probe_size_mult,
                "profile": "moderate",
                "override_confidence_score": confidence_score,
                "gate_decision_id": gate_decision_id,
                "reason": (
                    f"Setup '{setup_type}' rolling WR {rolling_wr:.1%} < {threshold:.0%} "
                    f"over last {rolling_sample_size} cases; moderate recovery probe "
                    f"at {p_probe_size_mult:.0%} size "
                    f"(override_confidence_score={confidence_score:.1f} >= "
                    f"{p_override_min_confidence})."
                ),
            }
        else:
            return {
                **base,
                "decision": "reject",
                "canonical_decision": "reject",
                "reason_type": "rolling_underperformance_confirmation_required",
                "profile": "moderate",
                "gate_decision_id": gate_decision_id,
                "reason": (
                    f"Setup '{setup_type}' rolling WR {rolling_wr:.1%} < {threshold:.0%}; "
                    f"moderate profile requires override_confidence_score >= "
                    f"{p_override_min_confidence} for recovery probe "
                    f"(got {confidence_score})."
                ),
            }

    elif profile_key == "conservative":
        return {
            **base,
            "decision": "reject",
            "canonical_decision": "reject",
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
            "canonical_decision": "reject",
            "reason_type": "rolling_underperformance",
            "gate_decision_id": gate_decision_id,
            "reason": (
                f"Setup '{setup_type}' rolling WR {rolling_wr:.1%} < {threshold:.0%} "
                f"over last {rolling_sample_size} cases; rejecting trade."
            ),
        }


def _evaluate_near_miss_pilot(
    *,
    win_rate: float,
    threshold: float,
    near_miss_margin: float,
    profile: str,
    confidence_score: float | None,
    catalyst_type: str | None,
    price_above_vwap: bool | None,
    volume_ratio: float | None,
) -> dict | None:
    """Check if a rejection qualifies for near-miss pilot override.

    Returns:
        Result dict with decision='reduce_size' if qualifies, None otherwise.

    Qualifying conditions:
    1. Profile is 'moderate'
    2. Pilot is active (via is_moderate_near_miss_pilot_active())
    3. Win rate is within margin below threshold: threshold - margin <= win_rate < threshold
    4. At least one confirming signal present
    """
    # 1. Profile must be 'moderate'
    if not isinstance(profile, str) or profile.lower() != "moderate":
        return None

    # 2. Pilot must be active
    if not is_moderate_near_miss_pilot_active():
        return None

    # 3. Win rate must be within [threshold - near_miss_margin, threshold)
    lower_bound = threshold - near_miss_margin
    if not (lower_bound <= win_rate < threshold):
        return None

    # 4. At least one confirming signal must be present
    confirming_signals: list[str] = []

    if confidence_score is not None and confidence_score >= 7.0:
        confirming_signals.append("confidence_score >= 7.0")

    if catalyst_type is not None and catalyst_type.strip():
        confirming_signals.append(f"catalyst_type: {catalyst_type}")

    if (
        price_above_vwap is True
        and volume_ratio is not None
        and volume_ratio >= 1.5
    ):
        confirming_signals.append("price_above_vwap AND volume_ratio >= 1.5")

    if not confirming_signals:
        return None

    return {
        "decision": "reduce_size",
        "canonical_decision": "reduce_size",
        "size_multiplier": ROLLING_RECOVERY_PROBE_SIZE_MULTIPLIER,
        "reason_type": "near_miss_pilot_override",
        "win_rate": win_rate,
        "threshold": threshold,
        "margin": near_miss_margin,
        "confirming_signals": confirming_signals,
    }


def _log_gate_event(
    db,
    decision_dict: dict,
    *,
    symbol: str | None = None,
    profile: str | None = None,
    agent: str = "portfolio_manager",
    event_sink=None,
) -> None:
    """Log a single TradeEvent for this gate evaluation.

    Uses ``GATE_EVENT_TYPES`` to map the decision to an ``event_type``.
    The full *decision_dict* is included in the payload alongside
    gate-specific audit fields.

    When *event_sink* is provided, uses it instead of ``log_trade_event``
    (dependency injection for replay).
    """
    decision = decision_dict.get("decision", "allow")
    event_type = GATE_EVENT_TYPES.get(decision, GATE_EVENT_TYPES["allow"])

    payload = {
        **decision_dict,
        "gate_name": "setup_quality_gate",
        "rejection_category": "setup_quality_gate",
        "reason_type": decision_dict.get("reason_type", ""),
    }

    sink = event_sink if event_sink is not None else log_trade_event
    sink(
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
    catalyst_type: str | None = None,
    price_above_vwap: bool | None = None,
    volume_ratio: float | None = None,
    policy=None,
    event_sink=None,
    clock=None,
    id_provider=None,
) -> dict:
    """Evaluate whether a setup type should be allowed based on case-library
    performance.

    Evaluation order (first match wins):
    1. Insufficient data (< MIN_CASES_FOR_BLOCK) → allow
    2. Consecutive losses (>= CONSECUTIVE_LOSS_PAUSE_THRESHOLD) → reject
    3. Historical underperformance (all-time WR < threshold) → reject
       UNLESS recovery override criteria are met → allow
       UNLESS near-miss pilot override applies (moderate profile only) → reduce_size
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
        confidence_score: Optional confidence score for near-miss pilot evaluation
        catalyst_type: Optional catalyst type for near-miss pilot evaluation
        price_above_vwap: Optional VWAP indicator for near-miss pilot evaluation
        volume_ratio: Optional volume ratio for near-miss pilot evaluation
        policy: Optional GatePolicyConfig; when provided, thresholds are read
            from it instead of module constants. None = use production defaults.
        event_sink: Optional callable replacing log_trade_event for logging.
            None = use production log_trade_event.
        clock: Optional callable returning datetime; replaces datetime.utcnow().
            None = use production clock. (Not currently used by this gate.)
        id_provider: Optional callable returning str; replaces uuid.uuid4().
            None = use production uuid.uuid4().

    Returns:
        Structured dict with decision, reason_type, stats, and threshold.

    Side effects:
        Logs exactly one TradeEvent to the provided *db* session.
        May log an additional pilot override event if near-miss pilot fires.
    """
    # --- Resolve thresholds from policy or module constants ---
    if policy is not None:
        p_min_cases_for_block = policy.min_cases_for_block
        p_consecutive_loss_pause = policy.consecutive_loss_pause_threshold
        p_rolling_window = policy.rolling_window
        p_min_rolling_cases = policy.min_rolling_cases
        p_recovery_min_rolling_cases = policy.recovery_min_rolling_cases
        p_recovery_wr_margin = policy.recovery_win_rate_margin
        p_require_positive_pnl = policy.require_positive_rolling_avg_pnl_for_recovery
        p_probe_size_mult = policy.rolling_recovery_probe_size_multiplier
        p_near_miss_margin = policy.near_miss_margin_pct
        p_override_min_confidence = policy.override_min_confidence_score
    else:
        p_min_cases_for_block = MIN_CASES_FOR_BLOCK
        p_consecutive_loss_pause = CONSECUTIVE_LOSS_PAUSE_THRESHOLD
        p_rolling_window = ROLLING_WINDOW
        p_min_rolling_cases = MIN_ROLLING_CASES
        p_recovery_min_rolling_cases = RECOVERY_MIN_ROLLING_CASES
        p_recovery_wr_margin = RECOVERY_WIN_RATE_MARGIN
        p_require_positive_pnl = REQUIRE_POSITIVE_ROLLING_AVG_PNL_FOR_RECOVERY
        p_probe_size_mult = ROLLING_RECOVERY_PROBE_SIZE_MULTIPLIER
        p_near_miss_margin = NEAR_MISS_MARGIN_PCT
        p_override_min_confidence = OVERRIDE_MIN_CONFIDENCE_SCORE

    # --- Resolve threshold (profile-aware) ---
    if policy is not None:
        profile_key = profile.lower() if isinstance(profile, str) else None
        setup_profile_thresholds = policy.min_win_rate_by_setup_profile.get(setup_type, {})
        if profile_key and profile_key in setup_profile_thresholds:
            threshold = float(setup_profile_thresholds[profile_key])
        elif profile_key and profile_key in policy.default_min_win_rate_by_profile:
            threshold = float(policy.default_min_win_rate_by_profile[profile_key])
        else:
            threshold = float(policy.min_win_rate_by_setup.get(setup_type, policy.default_min_win_rate))
    else:
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
        cases, p_rolling_window
    )

    # Build base result dict (fields filled in by each branch) ------------
    def _result(
        decision: str,
        reason_type: str,
        reason: str,
    ) -> dict:
        return {
            "decision": decision,
            "canonical_decision": decision,
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
    if sample_size < p_min_cases_for_block:
        result = _result(
            "allow",
            "insufficient_data",
            f"Only {sample_size} cases for setup '{setup_type}' "
            f"(need {p_min_cases_for_block}); allowing trade.",
        )
        _log_gate_event(db, result, symbol=symbol, profile=profile, agent=agent, event_sink=event_sink)
        return result

    # 2. Consecutive losses -----------------------------------------------
    if (
        setup_type not in CONSECUTIVE_LOSS_PAUSE_EXEMPT_SETUPS
        and _check_consecutive_losses(cases, p_consecutive_loss_pause)
    ):
        result = _result(
            "reject",
            "consecutive_losses",
            f"Last {p_consecutive_loss_pause} cases for "
            f"'{setup_type}' are all losses; pausing setup.",
        )
        _log_gate_event(db, result, symbol=symbol, profile=profile, agent=agent, event_sink=event_sink)
        return result

    # 3. Historical underperformance (with recovery check) ----------------
    if win_rate is not None and win_rate < threshold:
        # Check recovery override (policy-aware)
        recovery_met = False
        if rolling_sample_size >= p_recovery_min_rolling_cases:
            if rolling_wr is not None and rolling_wr > threshold + p_recovery_wr_margin:
                if p_require_positive_pnl:
                    if avg_rolling_pnl is not None and avg_rolling_pnl > 0:
                        recovery_met = True
                else:
                    recovery_met = True

        if recovery_met:
            result = _result(
                "allow",
                "recovery_override",
                f"Setup '{setup_type}' all-time WR "
                f"{win_rate:.1%} < {threshold:.0%} but recovery criteria met "
                f"(rolling WR {rolling_wr:.1%}, avg PnL {avg_rolling_pnl:+.2f}%).",
            )
            _log_gate_event(db, result, symbol=symbol, profile=profile, agent=agent, event_sink=event_sink)
            return result

        # Check near-miss pilot override before rejecting
        near_miss_result = _evaluate_near_miss_pilot(
            win_rate=win_rate,
            threshold=threshold,
            near_miss_margin=p_near_miss_margin,
            profile=profile or "",
            confidence_score=confidence_score,
            catalyst_type=catalyst_type,
            price_above_vwap=price_above_vwap,
            volume_ratio=volume_ratio,
        )
        if near_miss_result is not None:
            # Build the full gate result with standard fields + pilot override fields
            pilot_result = {
                **_result(
                    "reduce_size",
                    "near_miss_pilot_override",
                    f"Setup '{setup_type}' all-time WR "
                    f"{win_rate:.1%} < {threshold:.0%} but within near-miss margin "
                    f"({p_near_miss_margin:.0%}); pilot override with "
                    f"{near_miss_result['size_multiplier']:.0%} size.",
                ),
                "size_multiplier": near_miss_result["size_multiplier"],
                "confirming_signals": near_miss_result["confirming_signals"],
                "near_miss_margin": near_miss_result["margin"],
            }
            # Log standard gate decision event
            _log_gate_event(db, pilot_result, symbol=symbol, profile=profile, agent=agent, event_sink=event_sink)
            # Log SEPARATE pilot override audit event
            pilot_event_payload = {
                "gate_name": "setup_quality_gate",
                "canonical_decision": "reduce_size",
                "event_type": GATE_EVENT_TYPES["pilot_override"],
                "reason_type": "near_miss_pilot_override",
                "win_rate": win_rate,
                "threshold": threshold,
                "margin": p_near_miss_margin,
                "confirming_signals": near_miss_result["confirming_signals"],
                "size_multiplier": near_miss_result["size_multiplier"],
                "pilot_override": True,
                "pilot_size_multiplier": near_miss_result["size_multiplier"],
                "setup_type": setup_type,
            }
            sink = event_sink if event_sink is not None else log_trade_event
            sink(
                db,
                GATE_EVENT_TYPES["pilot_override"],
                agent=agent,
                symbol=symbol,
                profile=profile,
                message=(
                    f"Near-miss pilot override for '{setup_type}': WR {win_rate:.1%} "
                    f"within {p_near_miss_margin:.0%} of threshold {threshold:.0%}; "
                    f"signals: {near_miss_result['confirming_signals']}"
                ),
                payload=pilot_event_payload,
            )
            return pilot_result

        result = _result(
            "reject",
            "historical_underperformance",
            f"Setup '{setup_type}' all-time WR "
            f"{win_rate:.1%} < {threshold:.0%}; rejecting trade.",
        )
        _log_gate_event(db, result, symbol=symbol, profile=profile, agent=agent, event_sink=event_sink)
        return result

    # 4. Rolling underperformance — PROFILE-AWARE
    if (
        rolling_wr is not None
        and rolling_sample_size >= p_min_rolling_cases
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
            id_provider=id_provider,
            override_min_confidence=p_override_min_confidence,
            probe_size_multiplier=p_probe_size_mult,
        )
        _log_gate_event(db, result, symbol=symbol, profile=profile, agent=agent, event_sink=event_sink)
        return result

    # 5. Weak but allowed -------------------------------------------------
    if win_rate is not None and win_rate < 0.50:
        result = _result(
            "downgrade",
            "weak_but_allowed",
            f"Setup '{setup_type}' WR {win_rate:.1%} is above "
            f"block threshold ({threshold:.0%}) but below 50%; downgrading.",
        )
        _log_gate_event(db, result, symbol=symbol, profile=profile, agent=agent, event_sink=event_sink)
        return result

    # 6. Pass — all checks cleared ----------------------------------------
    result = _result(
        "allow",
        "pass",
        f"Setup '{setup_type}' passes all quality checks "
        f"(WR {win_rate:.1%}, {sample_size} cases).",
    )
    _log_gate_event(db, result, symbol=symbol, profile=profile, agent=agent, event_sink=event_sink)
    return result
