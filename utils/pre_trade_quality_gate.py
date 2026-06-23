"""PM pre-trade quality gate.

Evaluates trade quality based on Reviewer selection and execution scores.
The gate is a **pure evaluator** — the PM is responsible for passing all
available data (decision dict, signal metadata).  The gate does NOT query
AgentMemory or the database itself.

Score lookup priority (first non-None wins):
1. Fields on PM decision dict (``selection_score``, ``execution_score``)
2. Latest Analyst signal metadata (passed via *signal* parameter)
3. Relevant case/reviewer context explicitly attached to the signal
4. Treat as missing (``None``)

Evaluation order (first match wins):
1. Either score missing → warn
2. Both scores < 7.0 → reject
3. execution_score < 6.0 AND selection_score < 8.5 → reject
4. execution_score < 7.0 AND selection_score < 9.0 → override_required
5. Otherwise → allow

Override handling:
If the initial decision is ``override_required`` and the PM decision
contains valid override metadata (``override_confidence_score >= 8.0``
and non-empty ``override_reason``), the gate converts the decision to
``allow`` internally and logs a **single** ``gate_override_approved``
event (not two events).

See: requirements.md §2, design.md §utils/pre_trade_quality_gate.py
"""

from __future__ import annotations

import logging
from typing import Any

from utils.gate_config import (
    GATE_EVENT_TYPES,
    OVERRIDE_MIN_CONFIDENCE_SCORE,
)
from utils.trade_events import log_trade_event

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_score(
    field: str,
    decision: dict,
    signal: dict | None,
) -> float | None:
    """Resolve a score value using the priority chain.

    1. PM decision dict  (``decision[field]``)
    2. Analyst signal metadata  (``signal[field]``)
    3. Case/reviewer context attached to signal
       (``signal["case_context"][field]``)
    4. ``None``
    """
    # 1. PM decision dict
    value = decision.get(field)
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass

    if signal is not None:
        # 2. Signal metadata
        value = signal.get(field)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

        # 3. Case/reviewer context attached to signal
        case_ctx = signal.get("case_context")
        if isinstance(case_ctx, dict):
            value = case_ctx.get(field)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass

    # 4. Missing
    return None


def _check_override(decision: dict) -> bool:
    """Return ``True`` if the PM decision contains valid override metadata.

    Valid override requires:
    - ``override_confidence_score >= OVERRIDE_MIN_CONFIDENCE_SCORE``
    - non-empty ``override_reason``
    """
    confidence = decision.get("override_confidence_score")
    reason = decision.get("override_reason")

    if confidence is None or reason is None:
        return False

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return False

    if not isinstance(reason, str) or not reason.strip():
        return False

    return confidence >= OVERRIDE_MIN_CONFIDENCE_SCORE


def _check_override_di(decision: dict, min_confidence: float) -> bool:
    """Return ``True`` if the PM decision contains valid override metadata.

    Policy-aware variant that accepts the minimum confidence threshold
    as a parameter (dependency injection for replay).

    Valid override requires:
    - ``override_confidence_score >= min_confidence``
    - non-empty ``override_reason``
    """
    confidence = decision.get("override_confidence_score")
    reason = decision.get("override_reason")

    if confidence is None or reason is None:
        return False

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return False

    if not isinstance(reason, str) or not reason.strip():
        return False

    return confidence >= min_confidence


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

    payload: dict[str, Any] = {
        **decision_dict,
        "gate_name": "pre_trade_quality_gate",
        "rejection_category": "pre_trade_quality_gate",
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


def evaluate_pre_trade_quality(
    db,
    decision: dict,
    signal: dict | None = None,
    *,
    symbol: str | None = None,
    profile: str | None = None,
    agent: str = "portfolio_manager",
    policy=None,
    event_sink=None,
    clock=None,
    id_provider=None,
) -> dict:
    """Evaluate trade quality based on Reviewer selection and execution scores.

    The gate is a pure evaluator — the PM passes all available data and the
    gate evaluates what it receives without performing its own lookups.

    Args:
        db: SQLAlchemy session (caller owns commit)
        decision: PM decision dict with scores and optional override metadata
        signal: Optional Analyst signal dict for fallback score lookup
        symbol: Optional symbol for event logging
        profile: Optional PM profile for event logging
        agent: Agent name for event logging
        policy: Optional GatePolicyConfig; when provided, thresholds are read
            from it instead of module constants. None = use production defaults.
        event_sink: Optional callable replacing log_trade_event for logging.
            None = use production log_trade_event.
        clock: Optional callable returning datetime; replaces datetime.utcnow().
            None = use production clock. (Not currently used by this gate.)
        id_provider: Optional callable returning str; replaces uuid.uuid4().
            None = use production uuid.uuid4(). (Not currently used by this gate.)

    Returns:
        {
            "decision": "allow|warn|reject|override_required",
            "reason_type": str,
            "selection_score": float | None,
            "execution_score": float | None,
            "reason": str,
        }

    Side effects:
        Logs exactly one TradeEvent per evaluation.
    """
    # --- Resolve scores via priority chain --------------------------------
    selection_score = _resolve_score("selection_score", decision, signal)
    execution_score = _resolve_score("execution_score", decision, signal)

    # --- Resolve override threshold from policy or module constant ---
    p_override_min_confidence = (
        policy.override_min_confidence_score if policy is not None
        else OVERRIDE_MIN_CONFIDENCE_SCORE
    )

    # --- Helper to build result dict -------------------------------------
    def _result(gate_decision: str, reason_type: str, reason: str) -> dict:
        return {
            "decision": gate_decision,
            "canonical_decision": gate_decision,
            "reason_type": reason_type,
            "selection_score": selection_score,
            "execution_score": execution_score,
            "reason": reason,
        }

    # --- Evaluation chain (first match wins) -----------------------------

    # 1. Either score missing → warn
    if selection_score is None or execution_score is None:
        result = _result(
            "warn",
            "missing_quality_scores",
            "One or both quality scores are missing; allowing trade with warning.",
        )
        _log_gate_event(db, result, symbol=symbol, profile=profile, agent=agent, event_sink=event_sink)
        return result

    # 2. Both scores < 7.0 → reject
    if selection_score < 7.0 and execution_score < 7.0:
        result = _result(
            "reject",
            "both_scores_below_minimum",
            f"Both selection ({selection_score:.1f}) and execution "
            f"({execution_score:.1f}) scores are below 7.0; rejecting trade.",
        )
        _log_gate_event(db, result, symbol=symbol, profile=profile, agent=agent, event_sink=event_sink)
        return result

    # 3. execution_score < 6.0 AND selection_score < 8.5 → reject
    if execution_score < 6.0 and selection_score < 8.5:
        result = _result(
            "reject",
            "execution_low_selection_not_elite",
            f"Execution score ({execution_score:.1f}) is below 6.0 and "
            f"selection score ({selection_score:.1f}) is below 8.5; "
            f"rejecting trade.",
        )
        _log_gate_event(db, result, symbol=symbol, profile=profile, agent=agent, event_sink=event_sink)
        return result

    # 4. execution_score < 7.0 AND selection_score < 9.0 → override_required
    if execution_score < 7.0 and selection_score < 9.0:
        initial_result = _result(
            "override_required",
            "execution_below_auto_approval",
            f"Execution score ({execution_score:.1f}) is below 7.0 and "
            f"selection score ({selection_score:.1f}) is below 9.0; "
            f"override required.",
        )

        # --- Override handling (policy-aware threshold) -------------------
        if _check_override_di(decision, p_override_min_confidence):
            override_confidence = float(decision["override_confidence_score"])
            override_reason = decision["override_reason"]

            result = _result(
                "allow",
                "override_approved",
                f"Override approved: confidence {override_confidence:.1f}, "
                f"reason: {override_reason}",
            )
            # Log a single gate_override_approved event with full audit trail
            override_payload: dict[str, Any] = {
                **result,
                "original_decision": "override_required",
                "override_confidence_score": override_confidence,
                "override_reason": override_reason,
                "gate_name": "pre_trade_quality_gate",
                "rejection_category": "pre_trade_quality_gate",
                "reason_type": "override_approved",
            }
            sink = event_sink if event_sink is not None else log_trade_event
            sink(
                db,
                GATE_EVENT_TYPES["override_approved"],
                agent=agent,
                symbol=symbol,
                profile=profile,
                message=result["reason"],
                payload=override_payload,
            )
            return result

        # No valid override — keep override_required
        _log_gate_event(
            db, initial_result, symbol=symbol, profile=profile, agent=agent, event_sink=event_sink
        )
        return initial_result

    # 5. Otherwise → allow
    result = _result(
        "allow",
        "pass",
        f"Quality scores pass all checks "
        f"(selection={selection_score:.1f}, execution={execution_score:.1f}).",
    )
    _log_gate_event(db, result, symbol=symbol, profile=profile, agent=agent, event_sink=event_sink)
    return result
