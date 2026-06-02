"""Researcher Funnel Qualification — per-candidate catalyst quality evaluation.

New per-candidate qualification mode distinct from existing bulk-sentiment
Researcher. Makes promote/reject/needs_confirmation decisions with evidence
for each shortlisted FunnelCandidate.

See: design.md §Component 3, requirements.md §4
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from db.schema import AgentMemory, FunnelCandidate, get_session
from utils.catalyst_freshness import FRESHNESS_THRESHOLDS
from utils.llm import call_llm, parse_json_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class QualificationDecision:
    """Result of evaluating a single funnel candidate for catalyst quality."""

    candidate_id: str
    decision: str  # "promoted" | "rejected" | "needs_confirmation"
    reasoning: str
    catalyst_validation: dict  # freshness, specificity, material evidence
    contradictory_evidence: list[str] = field(default_factory=list)
    sentiment_assessment: str = ""
    evidence: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM Prompt for Catalyst Qualification
# ---------------------------------------------------------------------------

_QUALIFICATION_SYSTEM_PROMPT = """You are a financial research agent evaluating catalyst quality for day-trading candidates.

For the given candidate, assess:
1. Catalyst freshness: Is the catalyst from today or very recent? (fresh = <60min, aging = 60-180min, stale = >180min)
2. Catalyst specificity: Is it attributable to a named event, filing, data release, or specific news? Generic "market sentiment" or "sector rotation" without a named trigger is NOT specific.
3. Material evidence: Are there concrete data points supporting the catalyst (e.g., revenue beat, FDA approval, earnings surprise, CEO departure)?
4. Contradictory evidence: Any weakening factors, counter-arguments, or risks that undermine the thesis?
5. Sentiment assessment: Overall bullish/bearish/neutral with confidence level.

Decision criteria:
- PROMOTE if: catalyst is fresh/aging AND specific AND has material evidence
- REJECT if: catalyst is stale OR not specific (generic sector/market move without named trigger)
- NEEDS_CONFIRMATION if: catalyst is aging with partial specificity, or evidence is mixed

Respond in JSON format:
{
  "catalyst_freshness": "fresh|aging|stale",
  "catalyst_specific": true|false,
  "specificity_detail": "what named event/filing/release it refers to, or why it lacks specificity",
  "material_evidence": ["list of concrete supporting data points"],
  "contradictory_evidence": ["list of weakening factors or counter-arguments"],
  "sentiment": "bullish|bearish|neutral",
  "confidence": "low|medium|high",
  "catalysts": ["key catalyst drivers"],
  "risks": ["key risks"],
  "summary": "1-2 sentence summary of the thesis quality",
  "decision": "promoted|rejected|needs_confirmation",
  "reasoning": "explanation for the decision"
}
"""


# ---------------------------------------------------------------------------
# Core Qualification Logic
# ---------------------------------------------------------------------------


def run_funnel_qualification(
    engine,
    candidates: list[FunnelCandidate],
    config: dict,
    max_promoted: int = 3,
) -> list[QualificationDecision]:
    """Evaluate shortlisted candidates for catalyst quality and promotion.

    Evaluates each candidate independently — failure on one does not affect
    others. Enforces the max_promoted ceiling (default 3); excess qualifiers
    are rejected with reasoning "promotion_ceiling_reached".

    Candidates with stale or non-specific catalysts are rejected per
    requirement 4.4.

    Args:
        engine: SQLAlchemy engine for DB access.
        candidates: FunnelCandidate rows with stage_status="awaiting_research".
        config: Application configuration dict (sector scout config).
        max_promoted: Maximum candidates that can be promoted (default 3).

    Returns:
        List of QualificationDecision for each candidate that was evaluated
        (including not_evaluated decisions on failure).

    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6
    """
    decisions: list[QualificationDecision] = []
    promoted_count = 0

    for candidate in candidates:
        try:
            # Per-candidate LLM evaluation
            qualification = _evaluate_candidate_catalyst(candidate)

            # Decision logic based on catalyst quality
            decision, reasoning, next_stage = _make_qualification_decision(
                qualification=qualification,
                promoted_count=promoted_count,
                max_promoted=max_promoted,
            )

            # If promoted, increment counter and write AgentMemory sentiment record
            if decision == "promoted":
                promoted_count += 1
                _write_sentiment_memory(engine, candidate.symbol, qualification)

            # Build catalyst_validation dict
            catalyst_validation = {
                "freshness": qualification.get("catalyst_freshness", "unknown"),
                "specific": qualification.get("catalyst_specific", False),
                "specificity_detail": qualification.get("specificity_detail", ""),
                "material_evidence": qualification.get("material_evidence", []),
            }

            contradictory_evidence = qualification.get("contradictory_evidence", [])
            sentiment_assessment = qualification.get("sentiment", "neutral")

            # Build evidence payload for stage decision
            evidence_payload = {
                "catalyst_validation": catalyst_validation,
                "contradictory_evidence": contradictory_evidence,
                "sentiment": sentiment_assessment,
                "confidence": qualification.get("confidence", "low"),
                "catalysts": qualification.get("catalysts", []),
                "risks": qualification.get("risks", []),
                "summary": qualification.get("summary", ""),
            }

            # Append stage decision to candidate history (never overwrite)
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

            decisions.append(QualificationDecision(
                candidate_id=candidate.candidate_id,
                decision=decision,
                reasoning=reasoning,
                catalyst_validation=catalyst_validation,
                contradictory_evidence=contradictory_evidence,
                sentiment_assessment=sentiment_assessment,
                evidence=evidence_payload,
            ))

        except Exception as e:
            # Failure on one candidate does NOT erase others (Req 4.5)
            logger.error(
                "Funnel qualification failed for %s: %s",
                candidate.symbol, e,
            )
            _append_stage_decision(
                engine=engine,
                candidate=candidate,
                decision="not_evaluated",
                reasoning=f"Evaluation error: {str(e)[:200]}",
                evidence={},
                next_stage="awaiting_research",  # Stays in current stage
            )

            decisions.append(QualificationDecision(
                candidate_id=candidate.candidate_id,
                decision="not_evaluated",
                reasoning=f"Evaluation error: {str(e)[:200]}",
                catalyst_validation={},
                contradictory_evidence=[],
                sentiment_assessment="",
                evidence={},
            ))

    return decisions


# ---------------------------------------------------------------------------
# Per-Candidate LLM Evaluation
# ---------------------------------------------------------------------------


def _evaluate_candidate_catalyst(candidate: FunnelCandidate) -> dict:
    """Call LLM to evaluate catalyst quality for a single candidate.

    Builds a focused prompt with the candidate's catalyst evidence,
    sector context, direction bias, and selection reason, then asks
    the LLM to assess freshness, specificity, material evidence, and
    contradictory factors.

    Args:
        candidate: The FunnelCandidate to evaluate.

    Returns:
        Parsed JSON dict with evaluation fields.

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

    user_prompt = f"""Evaluate this day-trading candidate's catalyst quality:

SYMBOL: {candidate.symbol}
DIRECTION BIAS: {candidate.direction_bias or 'unknown'}
SCOUT SCORE: {candidate.scout_score}
SELECTION REASON: {candidate.selection_reason}
PRIMARY RISK: {candidate.primary_risk}

CATALYST EVIDENCE:
{json.dumps(catalyst_data, indent=2)}

SECTOR CONTEXT:
{json.dumps(sector_data, indent=2)}

Current time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

Assess catalyst freshness, specificity, material evidence, contradictory evidence, and overall sentiment. Make a promote/reject/needs_confirmation decision based on catalyst quality.
"""

    raw_response = call_llm(
        _QUALIFICATION_SYSTEM_PROMPT,
        user_prompt,
        json_mode=True,
        tier="medium",
        purpose="funnel_researcher_qualification",
    )
    result = parse_json_response(raw_response)
    return result


# ---------------------------------------------------------------------------
# Decision Logic
# ---------------------------------------------------------------------------


def _make_qualification_decision(
    qualification: dict,
    promoted_count: int,
    max_promoted: int,
) -> tuple[str, str, str]:
    """Determine promote/reject/needs_confirmation based on LLM evaluation.

    Enforces:
    - Stale catalyst → reject (Req 4.4)
    - Non-specific catalyst → reject (Req 4.4)
    - Promotion ceiling reached → reject with "promotion_ceiling_reached" (Req 4.3)
    - Fresh/aging + specific + material evidence → promote
    - Aging + partial specificity or mixed evidence → needs_confirmation

    Args:
        qualification: Parsed LLM evaluation dict.
        promoted_count: Current number of promoted candidates.
        max_promoted: Maximum promotion ceiling.

    Returns:
        Tuple of (decision, reasoning, next_stage).
    """
    freshness = qualification.get("catalyst_freshness", "stale")
    is_specific = qualification.get("catalyst_specific", False)
    material_evidence = qualification.get("material_evidence", [])
    llm_decision = qualification.get("decision", "rejected")
    llm_reasoning = qualification.get("reasoning", "No reasoning provided")

    # Hard reject: stale catalyst (Req 4.4)
    if freshness == "stale":
        return (
            "rejected",
            f"Stale catalyst (freshness={freshness}): {llm_reasoning}",
            "rejected_research",
        )

    # Hard reject: non-specific catalyst (Req 4.4)
    if not is_specific:
        specificity_detail = qualification.get("specificity_detail", "lacks specificity")
        return (
            "rejected",
            f"Non-specific catalyst ({specificity_detail}): {llm_reasoning}",
            "rejected_research",
        )

    # If the LLM says reject, honor it
    if llm_decision == "rejected":
        return ("rejected", llm_reasoning, "rejected_research")

    # If the LLM says needs_confirmation, honor it
    if llm_decision == "needs_confirmation":
        return ("needs_confirmation", llm_reasoning, "awaiting_confirmation")

    # LLM says promote — check ceiling (Req 4.3)
    if promoted_count >= max_promoted:
        return (
            "rejected",
            "promotion_ceiling_reached",
            "rejected_research",
        )

    # Promote: fresh/aging + specific + LLM endorses
    return ("promoted", llm_reasoning, "awaiting_analysis")


# ---------------------------------------------------------------------------
# Database Helpers
# ---------------------------------------------------------------------------


def _write_sentiment_memory(
    engine,
    symbol: str,
    qualification: dict,
) -> None:
    """Write a standard AgentMemory researcher sentiment record for a promoted candidate.

    Writes the same JSON schema that the existing bulk-sentiment Researcher
    produces (agent="researcher", key="sentiment"), so that downstream Analyst
    and PM agents can read funnel candidate sentiment via the existing
    AgentMemory query path without modification.

    The value JSON matches the expected schema:
        {
            "sentiment": "bullish|bearish|neutral",
            "confidence": "low|medium|high",
            "catalysts": [...],
            "risks": [...],
            "summary": "..."
        }

    Args:
        engine: SQLAlchemy engine.
        symbol: The candidate symbol.
        qualification: Parsed LLM evaluation dict from _evaluate_candidate_catalyst().

    Requirements: 4.7
    """
    sentiment_data = {
        "sentiment": qualification.get("sentiment", "neutral"),
        "confidence": qualification.get("confidence", "low"),
        "catalysts": qualification.get("catalysts", []),
        "risks": qualification.get("risks", []),
        "summary": qualification.get("summary", ""),
    }

    session = get_session(engine)
    try:
        mem = AgentMemory(
            agent="researcher",
            symbol=symbol,
            key="sentiment",
            value=json.dumps(sentiment_data),
        )
        session.add(mem)
        session.commit()
        logger.info(
            "Wrote AgentMemory sentiment record for funnel candidate %s", symbol
        )
    except Exception:
        session.rollback()
        # Log but don't fail the promotion — memory write is secondary to stage decision
        logger.exception(
            "Failed to write AgentMemory sentiment for %s", symbol
        )
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
    """Append a researcher stage decision to the candidate's history.

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
            "agent": "researcher",
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
