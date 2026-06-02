"""Analyst Funnel Analysis — authoritative setup classification for promoted candidates.

Builds authoritative setup type, key levels, invalidation conditions, and
signal direction/strength for each FunnelCandidate that reached awaiting_analysis.
Overrides Scout's preliminary_setup_type with an authoritative classification
used by downstream setup-quality gates.

See: design.md §Component 4, requirements.md §5
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from db.schema import AgentMemory, FunnelCandidate, get_session
from utils.llm import call_llm, parse_json_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class AnalysisDecision:
    """Result of evaluating a single funnel candidate for setup classification."""

    candidate_id: str
    decision: str  # "promoted" | "rejected" | "needs_confirmation"
    authoritative_setup_type: str | None = None
    signal_direction: str | None = None  # "LONG" | "SHORT" | "HOLD"
    signal_strength: str | None = None  # "weak" | "moderate" | "strong"
    key_levels: dict | None = None  # support, resistance, vwap, etc.
    invalidation: str | None = None
    volume_requirements: str | None = None
    catalyst_dependence: str | None = None
    reasoning: str = ""


# ---------------------------------------------------------------------------
# LLM Prompt for Analyst Setup Classification
# ---------------------------------------------------------------------------

_ANALYSIS_SYSTEM_PROMPT = """You are a technical analyst agent classifying day-trading setups for candidates that passed catalyst quality screening.

For the given candidate, determine:
1. Setup type: Classify the authoritative setup type. Valid types include: gap_and_go, breakout, reversal, momentum_continuation, mean_reversion, opening_range_breakout, vwap_reclaim, earnings_drift, news_catalyst, sector_rotation, or another recognized day-trading setup type.
2. Signal direction: LONG, SHORT, or HOLD
3. Signal strength: weak, moderate, or strong
4. Key levels: Identify support, resistance, VWAP, and other critical price levels for entry/exit
5. Invalidation condition: What specific price action or condition would invalidate this setup
6. Volume/VWAP requirements: What volume or VWAP conditions must be met for valid entry
7. Catalyst dependence: How dependent is this setup on the catalyst remaining fresh? (high/medium/low)
8. Catalyst freshness state: Is the catalyst still actionable? (fresh/aging/stale)

Decision criteria:
- PROMOTE if: clear setup type identified, directional signal is LONG or SHORT with moderate+ strength, key levels are identifiable, and invalidation is specific
- REJECT if: no actionable setup type, HOLD signal, weak strength with no catalysts, or contradictory technical picture that cannot be resolved
- NEEDS_CONFIRMATION if: setup exists but key levels need live market data confirmation, or signal is moderate with mixed evidence

Respond in JSON format:
{
  "setup_type": "gap_and_go|breakout|reversal|momentum_continuation|mean_reversion|opening_range_breakout|vwap_reclaim|earnings_drift|news_catalyst|sector_rotation|other",
  "signal_direction": "LONG|SHORT|HOLD",
  "signal_strength": "weak|moderate|strong",
  "confidence": "low|medium|high",
  "key_levels": {
    "support": null,
    "resistance": null,
    "vwap": null,
    "entry_zone": null,
    "stop_level": null,
    "target_1": null,
    "target_2": null
  },
  "invalidation": "specific condition that invalidates the setup",
  "volume_requirements": "description of volume/VWAP conditions required for entry",
  "catalyst_dependence": "high|medium|low",
  "catalyst_freshness": "fresh|aging|stale",
  "reasoning": "explanation of setup classification and signal decision",
  "decision": "promoted|rejected|needs_confirmation"
}
"""


# ---------------------------------------------------------------------------
# Core Analysis Logic
# ---------------------------------------------------------------------------


def run_funnel_analysis(
    engine,
    candidates: list[FunnelCandidate],
) -> list[AnalysisDecision]:
    """Build authoritative setup for each promoted candidate.

    Evaluates only candidates with stage_status=="awaiting_analysis".
    Determines authoritative_setup_type (overrides Scout preliminary_setup_type).
    Records signal direction, strength, key levels, invalidation, volume/VWAP
    requirements, catalyst dependence, and a promote/reject/needs_confirmation
    decision with reasoning.

    Failure on one candidate does NOT affect others — failed candidates receive
    a not_evaluated decision and remain in awaiting_analysis.

    Args:
        engine: SQLAlchemy engine for DB access.
        candidates: FunnelCandidate rows to evaluate.

    Returns:
        List of AnalysisDecision for each candidate evaluated
        (including not_evaluated decisions on failure).

    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.7
    """
    decisions: list[AnalysisDecision] = []

    # Only process candidates with stage_status=awaiting_analysis (Req 5.1)
    eligible = [c for c in candidates if c.stage_status == "awaiting_analysis"]

    for candidate in eligible:
        try:
            # Per-candidate LLM evaluation
            analysis = _evaluate_candidate_setup(engine, candidate)

            # Decision logic based on setup classification
            decision, reasoning, next_stage = _make_analysis_decision(analysis)

            # Extract fields from LLM response
            authoritative_setup_type = analysis.get("setup_type")
            signal_direction = _normalize_direction(analysis.get("signal_direction"))
            signal_strength = _normalize_strength(analysis.get("signal_strength"))
            key_levels = analysis.get("key_levels") if isinstance(analysis.get("key_levels"), dict) else None
            invalidation = analysis.get("invalidation")
            volume_requirements = analysis.get("volume_requirements")
            catalyst_dependence = analysis.get("catalyst_dependence")

            # If promoted, set authoritative_setup_type on the candidate (Req 5.3)
            if decision == "promoted":
                _set_authoritative_setup_type(engine, candidate, authoritative_setup_type)
                _write_signal_memory(engine, candidate.symbol, analysis, authoritative_setup_type)

            # If needs_confirmation, also set authoritative_setup_type since it's determined
            if decision == "needs_confirmation" and authoritative_setup_type:
                _set_authoritative_setup_type(engine, candidate, authoritative_setup_type)
                _write_signal_memory(engine, candidate.symbol, analysis, authoritative_setup_type)

            # Build evidence payload for stage decision (Req 5.5)
            evidence_payload = {
                "authoritative_setup_type": authoritative_setup_type,
                "signal_direction": signal_direction,
                "signal_strength": signal_strength,
                "confidence": analysis.get("confidence", "low"),
                "key_levels": key_levels,
                "invalidation": invalidation,
                "volume_requirements": volume_requirements,
                "catalyst_dependence": catalyst_dependence,
                "catalyst_freshness": analysis.get("catalyst_freshness"),
            }

            # Append stage decision to candidate history (never overwrite) (Req 5.5)
            _append_stage_decision(
                engine=engine,
                candidate=candidate,
                decision=decision,
                reasoning=reasoning,
                evidence=evidence_payload,
                next_stage=next_stage,
            )

            # Update stage_status
            _update_stage_status(engine, candidate, next_stage)

            decisions.append(AnalysisDecision(
                candidate_id=candidate.candidate_id,
                decision=decision,
                authoritative_setup_type=authoritative_setup_type,
                signal_direction=signal_direction,
                signal_strength=signal_strength,
                key_levels=key_levels,
                invalidation=invalidation,
                volume_requirements=volume_requirements,
                catalyst_dependence=catalyst_dependence,
                reasoning=reasoning,
            ))

        except Exception as e:
            # Failure on one candidate does NOT erase others (Req 5.7)
            logger.error(
                "Funnel analysis failed for %s: %s",
                candidate.symbol, e,
            )
            _append_stage_decision(
                engine=engine,
                candidate=candidate,
                decision="not_evaluated",
                reasoning=f"Analysis error: {str(e)[:200]}",
                evidence={},
                next_stage="awaiting_analysis",  # Stays in current stage
            )

            decisions.append(AnalysisDecision(
                candidate_id=candidate.candidate_id,
                decision="not_evaluated",
                reasoning=f"Analysis error: {str(e)[:200]}",
            ))

    return decisions


# ---------------------------------------------------------------------------
# Per-Candidate LLM Evaluation
# ---------------------------------------------------------------------------


def _evaluate_candidate_setup(engine, candidate: FunnelCandidate) -> dict:
    """Call LLM to classify setup type and determine signal for a single candidate.

    Builds a focused prompt with the candidate's catalyst evidence, sector
    context, researcher sentiment, direction bias, and selection reason, then
    asks the LLM to determine the authoritative setup classification.

    Args:
        engine: SQLAlchemy engine for reading researcher context.
        candidate: The FunnelCandidate to evaluate.

    Returns:
        Parsed JSON dict with analysis fields.

    Raises:
        ValueError: If LLM response cannot be parsed.
        Exception: On LLM call failure.
    """
    # Parse catalyst evidence JSON
    try:
        catalyst_data = json.loads(candidate.catalyst_evidence)
    except (json.JSONDecodeError, TypeError):
        catalyst_data = {"raw": candidate.catalyst_evidence}

    # Parse sector context JSON
    try:
        sector_data = json.loads(candidate.sector_context or "{}")
    except (json.JSONDecodeError, TypeError):
        sector_data = {}

    # Retrieve researcher sentiment from AgentMemory for context
    researcher_context = _get_researcher_sentiment(engine, candidate.symbol)

    # Get prior stage decisions for context
    try:
        stage_decisions = json.loads(candidate.stage_decisions or "[]")
    except (json.JSONDecodeError, TypeError):
        stage_decisions = []

    # Extract researcher decision evidence if available
    researcher_evidence = {}
    for sd in stage_decisions:
        if sd.get("agent") == "researcher" and sd.get("decision") in ("promoted", "needs_confirmation"):
            researcher_evidence = sd.get("evidence", {})
            break

    user_prompt = f"""Classify the trading setup and determine signal for this promoted candidate:

SYMBOL: {candidate.symbol}
DIRECTION BIAS (from Scout): {candidate.direction_bias or 'unknown'}
PRELIMINARY SETUP TYPE (from Scout - advisory only): {candidate.preliminary_setup_type or 'none'}
SCOUT SCORE: {candidate.scout_score}
SELECTION REASON: {candidate.selection_reason}
PRIMARY RISK: {candidate.primary_risk}

CATALYST EVIDENCE:
{json.dumps(catalyst_data, indent=2)}

SECTOR CONTEXT:
{json.dumps(sector_data, indent=2)}

RESEARCHER SENTIMENT:
{json.dumps(researcher_context, indent=2)}

RESEARCHER EVIDENCE:
{json.dumps(researcher_evidence, indent=2)}

Current time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

Determine the authoritative setup type, signal direction/strength, key levels, invalidation condition, volume/VWAP requirements, and catalyst dependence. The preliminary_setup_type from Scout is advisory only — you must determine the actual setup type based on the full evidence.
"""

    raw_response = call_llm(
        _ANALYSIS_SYSTEM_PROMPT,
        user_prompt,
        json_mode=True,
        tier="finance",
        purpose="funnel_analyst_analysis",
    )
    result = parse_json_response(raw_response)
    return result


# ---------------------------------------------------------------------------
# Decision Logic
# ---------------------------------------------------------------------------


def _make_analysis_decision(analysis: dict) -> tuple[str, str, str]:
    """Determine promote/reject/needs_confirmation based on LLM analysis.

    Enforces:
    - HOLD signal → reject (no actionable direction)
    - Weak strength with no clear setup → reject
    - Clear setup + directional signal + moderate/strong strength → promote
    - Setup identified but needs live data confirmation → needs_confirmation

    Args:
        analysis: Parsed LLM analysis dict.

    Returns:
        Tuple of (decision, reasoning, next_stage).
    """
    signal_direction = _normalize_direction(analysis.get("signal_direction"))
    signal_strength = _normalize_strength(analysis.get("signal_strength"))
    setup_type = analysis.get("setup_type", "unknown")
    llm_decision = analysis.get("decision", "rejected")
    llm_reasoning = analysis.get("reasoning", "No reasoning provided")

    # If the LLM explicitly says HOLD, reject — no actionable setup
    if signal_direction == "HOLD":
        return (
            "rejected",
            f"HOLD signal — no actionable setup: {llm_reasoning}",
            "rejected_analysis",
        )

    # If the LLM says reject, honor it
    if llm_decision == "rejected":
        return ("rejected", llm_reasoning, "rejected_analysis")

    # If the LLM says needs_confirmation, honor it
    if llm_decision == "needs_confirmation":
        return ("needs_confirmation", llm_reasoning, "awaiting_confirmation")

    # Weak strength with unknown/no setup → reject
    if signal_strength == "weak" and setup_type in ("unknown", "other", None, ""):
        return (
            "rejected",
            f"Weak signal with unclassifiable setup ({setup_type}): {llm_reasoning}",
            "rejected_analysis",
        )

    # LLM says promote — validate we have a real directional signal
    if signal_direction in ("LONG", "SHORT") and signal_strength in ("moderate", "strong"):
        return ("promoted", llm_reasoning, "awaiting_confirmation")

    # Moderate conditions that don't clearly promote — needs_confirmation
    if signal_direction in ("LONG", "SHORT") and signal_strength == "weak":
        return (
            "needs_confirmation",
            f"Weak directional signal — needs live confirmation: {llm_reasoning}",
            "awaiting_confirmation",
        )

    # Default: promote if LLM said promoted and there's a directional signal
    if llm_decision == "promoted" and signal_direction in ("LONG", "SHORT"):
        return ("promoted", llm_reasoning, "awaiting_confirmation")

    # Fallback: needs_confirmation
    return ("needs_confirmation", llm_reasoning, "awaiting_confirmation")


# ---------------------------------------------------------------------------
# Normalization Helpers
# ---------------------------------------------------------------------------


def _normalize_direction(direction: str | None) -> str:
    """Normalize signal direction to LONG/SHORT/HOLD."""
    if direction is None:
        return "HOLD"
    normalized = str(direction).upper().strip()
    if normalized in ("LONG", "SHORT", "HOLD"):
        return normalized
    # Handle alternate formats
    if normalized in ("BUY", "BULLISH"):
        return "LONG"
    if normalized in ("SELL", "BEARISH"):
        return "SHORT"
    return "HOLD"


def _normalize_strength(strength: str | None) -> str:
    """Normalize signal strength to weak/moderate/strong."""
    if strength is None:
        return "weak"
    normalized = str(strength).lower().strip()
    if normalized in ("weak", "moderate", "strong"):
        return normalized
    # Handle alternate formats
    if normalized in ("low", "minimal"):
        return "weak"
    if normalized in ("medium", "average"):
        return "moderate"
    if normalized in ("high", "very_strong"):
        return "strong"
    return "weak"


# ---------------------------------------------------------------------------
# Database Helpers
# ---------------------------------------------------------------------------


def _get_researcher_sentiment(engine, symbol: str) -> dict:
    """Retrieve the latest researcher sentiment record for context.

    Args:
        engine: SQLAlchemy engine.
        symbol: The candidate symbol.

    Returns:
        Parsed sentiment dict, or empty dict if not found.
    """
    session = get_session(engine)
    try:
        mem = (
            session.query(AgentMemory)
            .filter_by(agent="researcher", symbol=symbol, key="sentiment")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if mem and mem.value:
            return json.loads(mem.value)
        return {}
    except Exception:
        logger.exception("Failed to read researcher sentiment for %s", symbol)
        return {}
    finally:
        session.close()


def _write_signal_memory(
    engine,
    symbol: str,
    analysis: dict,
    authoritative_setup_type: str | None,
) -> None:
    """Write a standard AgentMemory analyst signal record for a promoted candidate.

    Writes the same JSON schema that the existing Analyst produces
    (agent="analyst", key="signal"), so that downstream PM agents can
    read funnel candidate signals via the existing AgentMemory query path
    without modification.

    The value JSON matches the expected schema:
        {
            "symbol": "...",
            "signal": "LONG|SHORT|HOLD",
            "strength": "weak|moderate|strong",
            "confidence": "low|medium|high",
            "setup_type": "...",
            "reasoning": "...",
            "key_levels": {...},
            "invalidation": "...",
        }

    Args:
        engine: SQLAlchemy engine.
        symbol: The candidate symbol.
        analysis: Parsed LLM analysis dict from _evaluate_candidate_setup().
        authoritative_setup_type: The determined setup type.

    Requirements: 5.6
    """
    signal_data = {
        "symbol": symbol,
        "signal": _normalize_direction(analysis.get("signal_direction")),
        "strength": _normalize_strength(analysis.get("signal_strength")),
        "confidence": analysis.get("confidence", "low"),
        "setup_type": authoritative_setup_type or analysis.get("setup_type", "unknown"),
        "reasoning": analysis.get("reasoning", ""),
        "key_levels": analysis.get("key_levels", {}),
        "invalidation": analysis.get("invalidation", ""),
    }

    session = get_session(engine)
    try:
        mem = AgentMemory(
            agent="analyst",
            symbol=symbol,
            key="signal",
            value=json.dumps(signal_data),
        )
        session.add(mem)
        session.commit()
        logger.info(
            "Wrote AgentMemory signal record for funnel candidate %s", symbol
        )
    except Exception:
        session.rollback()
        # Log but don't fail the promotion — memory write is secondary to stage decision
        logger.exception(
            "Failed to write AgentMemory signal for %s", symbol
        )
    finally:
        session.close()


def _set_authoritative_setup_type(
    engine,
    candidate: FunnelCandidate,
    setup_type: str | None,
) -> None:
    """Set the authoritative_setup_type field on the FunnelCandidate upon promotion.

    The authoritative_setup_type overrides Scout's preliminary_setup_type
    and is used by downstream setup-quality gates. Once set by Analyst,
    it SHALL NOT be changed except by a new Analyst stage decision (Req 5.5).

    Args:
        engine: SQLAlchemy engine.
        candidate: The FunnelCandidate to update.
        setup_type: The authoritative setup type determined by Analyst.
    """
    if not setup_type:
        return

    session = get_session(engine)
    try:
        db_candidate = (
            session.query(FunnelCandidate)
            .filter(FunnelCandidate.candidate_id == candidate.candidate_id)
            .first()
        )
        if db_candidate is None:
            logger.error(
                "Cannot set authoritative_setup_type: candidate %s not found",
                candidate.candidate_id,
            )
            return

        db_candidate.authoritative_setup_type = setup_type
        db_candidate.updated_at = datetime.now(timezone.utc)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _append_stage_decision(
    engine,
    candidate: FunnelCandidate,
    decision: str,
    reasoning: str,
    evidence: dict,
    next_stage: str,
) -> None:
    """Append an analyst stage decision to the candidate's history.

    Stage decisions are append-only — prior entries are never removed.
    This function reads the current stage_decisions JSON array, appends
    the new decision, and writes back atomically.

    Args:
        engine: SQLAlchemy engine.
        candidate: The FunnelCandidate to update.
        decision: One of "promoted", "rejected", "needs_confirmation", "not_evaluated".
        reasoning: Human-readable explanation.
        evidence: Agent-specific evidence payload dict.
        next_stage: Target stage_status value after this decision.
    """
    session = get_session(engine)
    try:
        # Re-fetch within session for transactional safety
        db_candidate = (
            session.query(FunnelCandidate)
            .filter(FunnelCandidate.candidate_id == candidate.candidate_id)
            .first()
        )
        if db_candidate is None:
            logger.error(
                "Cannot append stage decision: candidate %s not found",
                candidate.candidate_id,
            )
            return

        # Parse existing decisions
        try:
            current_decisions = json.loads(db_candidate.stage_decisions or "[]")
        except (json.JSONDecodeError, TypeError):
            current_decisions = []

        # Build new decision record
        stage_decision = {
            "agent": "analyst",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "reasoning": reasoning,
            "evidence": evidence,
            "next_stage": next_stage,
        }

        current_decisions.append(stage_decision)
        db_candidate.stage_decisions = json.dumps(current_decisions)
        db_candidate.updated_at = datetime.now(timezone.utc)

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _update_stage_status(
    engine,
    candidate: FunnelCandidate,
    next_stage: str,
) -> None:
    """Update the candidate's stage_status to the next stage.

    Only updates if next_stage is different from current stage_status.
    Does not regress stage_status (checked upstream by decision logic).

    Args:
        engine: SQLAlchemy engine.
        candidate: The FunnelCandidate to update.
        next_stage: Target stage_status value.
    """
    session = get_session(engine)
    try:
        db_candidate = (
            session.query(FunnelCandidate)
            .filter(FunnelCandidate.candidate_id == candidate.candidate_id)
            .first()
        )
        if db_candidate is None:
            logger.error(
                "Cannot update stage_status: candidate %s not found",
                candidate.candidate_id,
            )
            return

        db_candidate.stage_status = next_stage
        db_candidate.updated_at = datetime.now(timezone.utc)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
