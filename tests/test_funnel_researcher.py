"""Unit tests for funnel researcher qualification (Task 5.1).

Tests:
- run_funnel_qualification() — per-candidate evaluation, promotion ceiling,
  rejection of stale/non-specific catalysts, failure isolation
- _make_qualification_decision() — decision logic
- _append_stage_decision() — append-only stage history
- _update_stage_status() — stage status transitions
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import create_engine

from db.schema import Base, FunnelCandidate, AgentMemory, get_session
from utils.funnel_researcher import (
    QualificationDecision,
    run_funnel_qualification,
    _make_qualification_decision,
    _append_stage_decision,
    _update_stage_status,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_engine():
    """Create an in-memory SQLite engine with all tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def _make_funnel_candidate(
    engine,
    symbol: str = "AAPL",
    scout_score: float = 80.0,
    direction_bias: str = "bullish",
    catalyst_evidence: dict | None = None,
    selection_reason: str = "Strong earnings beat",
    primary_risk: str = "Valuation stretched",
    stage_status: str = "awaiting_research",
) -> FunnelCandidate:
    """Create and persist a FunnelCandidate in the DB for testing."""
    if catalyst_evidence is None:
        catalyst_evidence = {
            "news_headlines": [{"headline": f"{symbol} beats earnings expectations"}],
            "news_freshness_minutes": 30.0,
            "move_pct": 3.5,
            "relative_volume": 2.1,
            "dollar_volume": 500_000_000,
        }

    ny_tz = ZoneInfo("America/New_York")
    today_ny = datetime.now(timezone.utc).astimezone(ny_tz).date()

    scout_decision = {
        "agent": "scout",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": "promoted",
        "reasoning": selection_reason,
        "evidence": {
            "scout_score": scout_score,
            "direction_bias": direction_bias,
            "selection_mode": "deterministic_fallback",
        },
        "next_stage": "awaiting_research",
    }

    candidate = FunnelCandidate(
        candidate_id=str(uuid.uuid4()),
        date=today_ny,
        symbol=symbol,
        discovered_at=datetime.now(timezone.utc),
        source_run="premarket",
        selection_mode="deterministic_fallback",
        scout_rank=1,
        scout_score=scout_score,
        direction_bias=direction_bias,
        catalyst_evidence=json.dumps(catalyst_evidence),
        selection_reason=selection_reason,
        primary_risk=primary_risk,
        sector_context=json.dumps({"sector": "tech", "sector_name": "Technology"}),
        preliminary_setup_type=None,
        stage_status=stage_status,
        stage_decisions=json.dumps([scout_decision]),
        expired=False,
    )

    session = get_session(engine)
    session.add(candidate)
    session.commit()
    session.refresh(candidate)
    session.expunge(candidate)
    session.close()
    return candidate


def _mock_llm_response(
    freshness: str = "fresh",
    specific: bool = True,
    decision: str = "promoted",
    reasoning: str = "Strong catalyst with named event",
    material_evidence: list | None = None,
    contradictory_evidence: list | None = None,
) -> dict:
    """Build a mock LLM response dict for qualification."""
    return {
        "catalyst_freshness": freshness,
        "catalyst_specific": specific,
        "specificity_detail": "Q4 earnings beat by 15%" if specific else "generic market sentiment",
        "material_evidence": material_evidence or ["Revenue +15% YoY", "EPS beat by $0.20"],
        "contradictory_evidence": contradictory_evidence or [],
        "sentiment": "bullish",
        "confidence": "high",
        "catalysts": ["Earnings beat"],
        "risks": ["Valuation"],
        "summary": "Strong catalyst with clear named event",
        "decision": decision,
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# _make_qualification_decision tests
# ---------------------------------------------------------------------------


class TestMakeQualificationDecision:
    """Tests for the decision logic function."""

    def test_stale_catalyst_rejected(self):
        """Stale catalyst always leads to rejection."""
        qual = _mock_llm_response(freshness="stale", specific=True, decision="promoted")
        decision, reasoning, next_stage = _make_qualification_decision(qual, 0, 3)

        assert decision == "rejected"
        assert "stale" in reasoning.lower() or "Stale" in reasoning
        assert next_stage == "rejected_research"

    def test_non_specific_catalyst_rejected(self):
        """Non-specific catalyst always leads to rejection."""
        qual = _mock_llm_response(freshness="fresh", specific=False, decision="promoted")
        decision, reasoning, next_stage = _make_qualification_decision(qual, 0, 3)

        assert decision == "rejected"
        assert "specific" in reasoning.lower() or "specificity" in reasoning.lower()
        assert next_stage == "rejected_research"

    def test_promotion_ceiling_enforced(self):
        """When ceiling is reached, qualified candidates are rejected."""
        qual = _mock_llm_response(freshness="fresh", specific=True, decision="promoted")
        decision, reasoning, next_stage = _make_qualification_decision(qual, 3, 3)

        assert decision == "rejected"
        assert "promotion_ceiling_reached" in reasoning
        assert next_stage == "rejected_research"

    def test_fresh_specific_promoted_below_ceiling(self):
        """Fresh + specific catalyst is promoted when below ceiling."""
        qual = _mock_llm_response(freshness="fresh", specific=True, decision="promoted")
        decision, reasoning, next_stage = _make_qualification_decision(qual, 0, 3)

        assert decision == "promoted"
        assert next_stage == "awaiting_analysis"

    def test_aging_specific_promoted(self):
        """Aging + specific catalyst can be promoted."""
        qual = _mock_llm_response(freshness="aging", specific=True, decision="promoted")
        decision, reasoning, next_stage = _make_qualification_decision(qual, 0, 3)

        assert decision == "promoted"
        assert next_stage == "awaiting_analysis"

    def test_llm_needs_confirmation_honored(self):
        """LLM needs_confirmation decision is respected."""
        qual = _mock_llm_response(freshness="fresh", specific=True, decision="needs_confirmation")
        decision, reasoning, next_stage = _make_qualification_decision(qual, 0, 3)

        assert decision == "needs_confirmation"
        assert next_stage == "awaiting_confirmation"

    def test_llm_rejected_honored(self):
        """LLM rejection is respected even with fresh/specific catalyst."""
        qual = _mock_llm_response(freshness="fresh", specific=True, decision="rejected")
        decision, reasoning, next_stage = _make_qualification_decision(qual, 0, 3)

        assert decision == "rejected"
        assert next_stage == "rejected_research"

    def test_stale_overrides_llm_promote(self):
        """Stale check takes priority over LLM promote decision."""
        qual = _mock_llm_response(freshness="stale", specific=True, decision="promoted")
        decision, _, _ = _make_qualification_decision(qual, 0, 3)
        assert decision == "rejected"

    def test_non_specific_overrides_llm_promote(self):
        """Specificity check takes priority over LLM promote decision."""
        qual = _mock_llm_response(freshness="fresh", specific=False, decision="promoted")
        decision, _, _ = _make_qualification_decision(qual, 0, 3)
        assert decision == "rejected"


# ---------------------------------------------------------------------------
# _append_stage_decision tests
# ---------------------------------------------------------------------------


class TestAppendStageDecision:
    """Tests for stage decision persistence."""

    def test_appends_to_existing_decisions(self, in_memory_engine):
        """New decision is appended to existing stage_decisions array."""
        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")

        _append_stage_decision(
            engine=in_memory_engine,
            candidate=candidate,
            decision="promoted",
            reasoning="Strong catalyst",
            evidence={"test": True},
            next_stage="awaiting_analysis",
        )

        session = get_session(in_memory_engine)
        row = session.query(FunnelCandidate).filter_by(symbol="AAPL").first()
        decisions = json.loads(row.stage_decisions)
        assert len(decisions) == 2  # scout + researcher
        assert decisions[1]["agent"] == "researcher"
        assert decisions[1]["decision"] == "promoted"
        assert decisions[1]["reasoning"] == "Strong catalyst"
        assert decisions[1]["evidence"] == {"test": True}
        assert decisions[1]["next_stage"] == "awaiting_analysis"
        assert "timestamp" in decisions[1]
        session.close()

    def test_preserves_prior_decisions(self, in_memory_engine):
        """Prior decisions remain unchanged when appending."""
        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")

        # Get original scout decision
        session = get_session(in_memory_engine)
        row = session.query(FunnelCandidate).filter_by(symbol="AAPL").first()
        original_decisions = json.loads(row.stage_decisions)
        session.close()

        _append_stage_decision(
            engine=in_memory_engine,
            candidate=candidate,
            decision="promoted",
            reasoning="Test",
            evidence={},
            next_stage="awaiting_analysis",
        )

        session = get_session(in_memory_engine)
        row = session.query(FunnelCandidate).filter_by(symbol="AAPL").first()
        decisions = json.loads(row.stage_decisions)
        # Original scout decision unchanged
        assert decisions[0] == original_decisions[0]
        session.close()

    def test_not_evaluated_decision_recorded(self, in_memory_engine):
        """not_evaluated decision is properly recorded."""
        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")

        _append_stage_decision(
            engine=in_memory_engine,
            candidate=candidate,
            decision="not_evaluated",
            reasoning="Evaluation error: LLM timeout",
            evidence={},
            next_stage="awaiting_research",
        )

        session = get_session(in_memory_engine)
        row = session.query(FunnelCandidate).filter_by(symbol="AAPL").first()
        decisions = json.loads(row.stage_decisions)
        assert decisions[-1]["decision"] == "not_evaluated"
        assert "LLM timeout" in decisions[-1]["reasoning"]
        session.close()


# ---------------------------------------------------------------------------
# _update_stage_status tests
# ---------------------------------------------------------------------------


class TestUpdateStageStatus:
    """Tests for stage status transitions."""

    def test_updates_to_awaiting_analysis(self, in_memory_engine):
        """Promoted candidates move to awaiting_analysis."""
        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")

        _update_stage_status(in_memory_engine, candidate, "awaiting_analysis")

        session = get_session(in_memory_engine)
        row = session.query(FunnelCandidate).filter_by(symbol="AAPL").first()
        assert row.stage_status == "awaiting_analysis"
        session.close()

    def test_updates_to_rejected_research(self, in_memory_engine):
        """Rejected candidates move to rejected_research."""
        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")

        _update_stage_status(in_memory_engine, candidate, "rejected_research")

        session = get_session(in_memory_engine)
        row = session.query(FunnelCandidate).filter_by(symbol="AAPL").first()
        assert row.stage_status == "rejected_research"
        session.close()

    def test_updates_to_awaiting_confirmation(self, in_memory_engine):
        """needs_confirmation candidates move to awaiting_confirmation."""
        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")

        _update_stage_status(in_memory_engine, candidate, "awaiting_confirmation")

        session = get_session(in_memory_engine)
        row = session.query(FunnelCandidate).filter_by(symbol="AAPL").first()
        assert row.stage_status == "awaiting_confirmation"
        session.close()


# ---------------------------------------------------------------------------
# run_funnel_qualification integration tests
# ---------------------------------------------------------------------------


class TestRunFunnelQualification:
    """Integration tests for the full qualification pipeline."""

    @patch("utils.funnel_researcher._evaluate_candidate_catalyst")
    def test_promotes_qualified_candidates(self, mock_eval, in_memory_engine):
        """Candidates with fresh specific catalysts are promoted."""
        mock_eval.return_value = _mock_llm_response(
            freshness="fresh", specific=True, decision="promoted"
        )

        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")
        decisions = run_funnel_qualification(
            in_memory_engine, [candidate], config={}, max_promoted=3
        )

        assert len(decisions) == 1
        assert decisions[0].decision == "promoted"
        assert decisions[0].candidate_id == candidate.candidate_id

        # Verify DB state
        session = get_session(in_memory_engine)
        row = session.query(FunnelCandidate).filter_by(symbol="AAPL").first()
        assert row.stage_status == "awaiting_analysis"
        stage_decisions = json.loads(row.stage_decisions)
        assert len(stage_decisions) == 2  # scout + researcher
        assert stage_decisions[1]["agent"] == "researcher"
        assert stage_decisions[1]["decision"] == "promoted"
        session.close()

    @patch("utils.funnel_researcher._evaluate_candidate_catalyst")
    def test_rejects_stale_catalyst(self, mock_eval, in_memory_engine):
        """Candidates with stale catalysts are rejected."""
        mock_eval.return_value = _mock_llm_response(
            freshness="stale", specific=True, decision="promoted"
        )

        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")
        decisions = run_funnel_qualification(
            in_memory_engine, [candidate], config={}, max_promoted=3
        )

        assert len(decisions) == 1
        assert decisions[0].decision == "rejected"
        assert "stale" in decisions[0].reasoning.lower() or "Stale" in decisions[0].reasoning

        # Verify DB state
        session = get_session(in_memory_engine)
        row = session.query(FunnelCandidate).filter_by(symbol="AAPL").first()
        assert row.stage_status == "rejected_research"
        session.close()

    @patch("utils.funnel_researcher._evaluate_candidate_catalyst")
    def test_rejects_non_specific_catalyst(self, mock_eval, in_memory_engine):
        """Candidates with non-specific catalysts are rejected."""
        mock_eval.return_value = _mock_llm_response(
            freshness="fresh", specific=False, decision="promoted"
        )

        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")
        decisions = run_funnel_qualification(
            in_memory_engine, [candidate], config={}, max_promoted=3
        )

        assert len(decisions) == 1
        assert decisions[0].decision == "rejected"

        session = get_session(in_memory_engine)
        row = session.query(FunnelCandidate).filter_by(symbol="AAPL").first()
        assert row.stage_status == "rejected_research"
        session.close()

    @patch("utils.funnel_researcher._evaluate_candidate_catalyst")
    def test_enforces_promotion_ceiling(self, mock_eval, in_memory_engine):
        """No more than max_promoted candidates are promoted."""
        mock_eval.return_value = _mock_llm_response(
            freshness="fresh", specific=True, decision="promoted"
        )

        # Create 5 candidates
        candidates = []
        for i, sym in enumerate(["AAPL", "MSFT", "GOOGL", "AMZN", "META"]):
            c = _make_funnel_candidate(in_memory_engine, sym, scout_score=90.0 - i)
            candidates.append(c)

        decisions = run_funnel_qualification(
            in_memory_engine, candidates, config={}, max_promoted=3
        )

        promoted = [d for d in decisions if d.decision == "promoted"]
        rejected_ceiling = [d for d in decisions if "promotion_ceiling_reached" in d.reasoning]

        assert len(promoted) == 3
        assert len(rejected_ceiling) == 2

        # Verify rejected-ceiling candidates have correct stage_status
        session = get_session(in_memory_engine)
        for sym in ["AMZN", "META"]:
            row = session.query(FunnelCandidate).filter_by(symbol=sym).first()
            assert row.stage_status == "rejected_research"
        session.close()

    @patch("utils.funnel_researcher._evaluate_candidate_catalyst")
    def test_failure_isolation(self, mock_eval, in_memory_engine):
        """Failure on one candidate doesn't affect others."""
        # First call succeeds, second raises, third succeeds
        mock_eval.side_effect = [
            _mock_llm_response(freshness="fresh", specific=True, decision="promoted"),
            RuntimeError("LLM service unavailable"),
            _mock_llm_response(freshness="fresh", specific=True, decision="promoted"),
        ]

        candidates = [
            _make_funnel_candidate(in_memory_engine, "AAPL"),
            _make_funnel_candidate(in_memory_engine, "MSFT"),
            _make_funnel_candidate(in_memory_engine, "GOOGL"),
        ]

        decisions = run_funnel_qualification(
            in_memory_engine, candidates, config={}, max_promoted=3
        )

        assert len(decisions) == 3
        assert decisions[0].decision == "promoted"
        assert decisions[1].decision == "not_evaluated"
        assert decisions[2].decision == "promoted"

        # MSFT should stay in awaiting_research
        session = get_session(in_memory_engine)
        msft = session.query(FunnelCandidate).filter_by(symbol="MSFT").first()
        assert msft.stage_status == "awaiting_research"

        # MSFT still has not_evaluated decision appended
        msft_decisions = json.loads(msft.stage_decisions)
        assert msft_decisions[-1]["decision"] == "not_evaluated"
        assert "LLM service unavailable" in msft_decisions[-1]["reasoning"]

        # Others are promoted
        aapl = session.query(FunnelCandidate).filter_by(symbol="AAPL").first()
        assert aapl.stage_status == "awaiting_analysis"
        googl = session.query(FunnelCandidate).filter_by(symbol="GOOGL").first()
        assert googl.stage_status == "awaiting_analysis"
        session.close()

    @patch("utils.funnel_researcher._evaluate_candidate_catalyst")
    def test_needs_confirmation_decision(self, mock_eval, in_memory_engine):
        """needs_confirmation sets stage to awaiting_confirmation."""
        mock_eval.return_value = _mock_llm_response(
            freshness="aging", specific=True, decision="needs_confirmation",
            reasoning="Catalyst is aging, needs live confirmation"
        )

        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")
        decisions = run_funnel_qualification(
            in_memory_engine, [candidate], config={}, max_promoted=3
        )

        assert decisions[0].decision == "needs_confirmation"

        session = get_session(in_memory_engine)
        row = session.query(FunnelCandidate).filter_by(symbol="AAPL").first()
        assert row.stage_status == "awaiting_confirmation"
        session.close()

    @patch("utils.funnel_researcher._evaluate_candidate_catalyst")
    def test_empty_candidate_list(self, mock_eval, in_memory_engine):
        """Empty candidate list returns empty decisions."""
        decisions = run_funnel_qualification(
            in_memory_engine, [], config={}, max_promoted=3
        )

        assert decisions == []
        mock_eval.assert_not_called()

    @patch("utils.funnel_researcher._evaluate_candidate_catalyst")
    def test_all_rejected_no_promotions(self, mock_eval, in_memory_engine):
        """If all candidates fail qualification, zero promotions (Req 4.3)."""
        mock_eval.return_value = _mock_llm_response(
            freshness="stale", specific=True, decision="promoted"
        )

        candidates = [
            _make_funnel_candidate(in_memory_engine, "AAPL"),
            _make_funnel_candidate(in_memory_engine, "MSFT"),
        ]

        decisions = run_funnel_qualification(
            in_memory_engine, candidates, config={}, max_promoted=3
        )

        promoted = [d for d in decisions if d.decision == "promoted"]
        assert len(promoted) == 0

    @patch("utils.funnel_researcher._evaluate_candidate_catalyst")
    def test_catalyst_validation_recorded(self, mock_eval, in_memory_engine):
        """QualificationDecision includes catalyst_validation details."""
        mock_eval.return_value = _mock_llm_response(
            freshness="fresh",
            specific=True,
            decision="promoted",
            material_evidence=["Revenue +15%", "EPS beat"],
            contradictory_evidence=["High valuation"],
        )

        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")
        decisions = run_funnel_qualification(
            in_memory_engine, [candidate], config={}, max_promoted=3
        )

        d = decisions[0]
        assert d.catalyst_validation["freshness"] == "fresh"
        assert d.catalyst_validation["specific"] is True
        assert "Revenue +15%" in d.catalyst_validation["material_evidence"]
        assert d.contradictory_evidence == ["High valuation"]
        assert d.sentiment_assessment == "bullish"

    @patch("utils.funnel_researcher._evaluate_candidate_catalyst")
    def test_evidence_payload_in_stage_decision(self, mock_eval, in_memory_engine):
        """Stage decision evidence includes full evaluation payload."""
        mock_eval.return_value = _mock_llm_response(
            freshness="fresh", specific=True, decision="promoted"
        )

        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")
        run_funnel_qualification(
            in_memory_engine, [candidate], config={}, max_promoted=3
        )

        session = get_session(in_memory_engine)
        row = session.query(FunnelCandidate).filter_by(symbol="AAPL").first()
        stage_decisions = json.loads(row.stage_decisions)
        researcher_decision = stage_decisions[-1]

        assert researcher_decision["agent"] == "researcher"
        evidence = researcher_decision["evidence"]
        assert "catalyst_validation" in evidence
        assert "contradictory_evidence" in evidence
        assert "sentiment" in evidence
        assert "confidence" in evidence
        session.close()

    @patch("utils.funnel_researcher._evaluate_candidate_catalyst")
    def test_max_promoted_zero_rejects_all(self, mock_eval, in_memory_engine):
        """max_promoted=0 means all qualified candidates get ceiling rejection."""
        mock_eval.return_value = _mock_llm_response(
            freshness="fresh", specific=True, decision="promoted"
        )

        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")
        decisions = run_funnel_qualification(
            in_memory_engine, [candidate], config={}, max_promoted=0
        )

        assert decisions[0].decision == "rejected"
        assert "promotion_ceiling_reached" in decisions[0].reasoning


# ---------------------------------------------------------------------------
# AgentMemory sentiment record tests (Task 5.2)
# ---------------------------------------------------------------------------


class TestAgentMemorySentimentRecord:
    """Tests for writing AgentMemory researcher sentiment records for promoted candidates."""

    @patch("utils.funnel_researcher._evaluate_candidate_catalyst")
    def test_promoted_candidate_gets_sentiment_record(self, mock_eval, in_memory_engine):
        """Promoted candidates get an AgentMemory sentiment record written."""
        mock_eval.return_value = _mock_llm_response(
            freshness="fresh", specific=True, decision="promoted"
        )

        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")
        run_funnel_qualification(
            in_memory_engine, [candidate], config={}, max_promoted=3
        )

        # Verify AgentMemory record exists
        session = get_session(in_memory_engine)
        mem = (
            session.query(AgentMemory)
            .filter_by(agent="researcher", key="sentiment", symbol="AAPL")
            .first()
        )
        assert mem is not None
        assert mem.agent == "researcher"
        assert mem.key == "sentiment"
        assert mem.symbol == "AAPL"
        session.close()

    @patch("utils.funnel_researcher._evaluate_candidate_catalyst")
    def test_sentiment_record_matches_expected_schema(self, mock_eval, in_memory_engine):
        """AgentMemory sentiment value matches the schema used by existing Researcher."""
        mock_eval.return_value = _mock_llm_response(
            freshness="fresh",
            specific=True,
            decision="promoted",
            material_evidence=["Revenue +15% YoY"],
            contradictory_evidence=["High PE ratio"],
        )

        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")
        run_funnel_qualification(
            in_memory_engine, [candidate], config={}, max_promoted=3
        )

        session = get_session(in_memory_engine)
        mem = (
            session.query(AgentMemory)
            .filter_by(agent="researcher", key="sentiment", symbol="AAPL")
            .first()
        )
        data = json.loads(mem.value)

        # Must have all required fields matching existing Researcher output schema
        assert "sentiment" in data
        assert data["sentiment"] in {"bullish", "bearish", "neutral"}
        assert "confidence" in data
        assert data["confidence"] in {"low", "medium", "high"}
        assert "catalysts" in data
        assert isinstance(data["catalysts"], list)
        assert "risks" in data
        assert isinstance(data["risks"], list)
        assert "summary" in data
        assert isinstance(data["summary"], str)
        session.close()

    @patch("utils.funnel_researcher._evaluate_candidate_catalyst")
    def test_sentiment_record_contains_correct_values(self, mock_eval, in_memory_engine):
        """AgentMemory sentiment value contains correct values from LLM evaluation."""
        mock_eval.return_value = _mock_llm_response(
            freshness="fresh",
            specific=True,
            decision="promoted",
            material_evidence=["Revenue +15% YoY"],
            contradictory_evidence=["High PE ratio"],
        )

        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")
        run_funnel_qualification(
            in_memory_engine, [candidate], config={}, max_promoted=3
        )

        session = get_session(in_memory_engine)
        mem = (
            session.query(AgentMemory)
            .filter_by(agent="researcher", key="sentiment", symbol="AAPL")
            .first()
        )
        data = json.loads(mem.value)

        assert data["sentiment"] == "bullish"
        assert data["confidence"] == "high"
        assert data["catalysts"] == ["Earnings beat"]
        assert data["risks"] == ["Valuation"]
        assert data["summary"] == "Strong catalyst with clear named event"
        session.close()

    @patch("utils.funnel_researcher._evaluate_candidate_catalyst")
    def test_rejected_candidate_no_sentiment_record(self, mock_eval, in_memory_engine):
        """Rejected candidates do NOT get an AgentMemory sentiment record."""
        mock_eval.return_value = _mock_llm_response(
            freshness="stale", specific=True, decision="promoted"
        )

        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")
        run_funnel_qualification(
            in_memory_engine, [candidate], config={}, max_promoted=3
        )

        session = get_session(in_memory_engine)
        mem = (
            session.query(AgentMemory)
            .filter_by(agent="researcher", key="sentiment", symbol="AAPL")
            .first()
        )
        assert mem is None
        session.close()

    @patch("utils.funnel_researcher._evaluate_candidate_catalyst")
    def test_needs_confirmation_no_sentiment_record(self, mock_eval, in_memory_engine):
        """needs_confirmation candidates do NOT get an AgentMemory sentiment record."""
        mock_eval.return_value = _mock_llm_response(
            freshness="fresh", specific=True, decision="needs_confirmation"
        )

        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")
        run_funnel_qualification(
            in_memory_engine, [candidate], config={}, max_promoted=3
        )

        session = get_session(in_memory_engine)
        mem = (
            session.query(AgentMemory)
            .filter_by(agent="researcher", key="sentiment", symbol="AAPL")
            .first()
        )
        assert mem is None
        session.close()

    @patch("utils.funnel_researcher._evaluate_candidate_catalyst")
    def test_multiple_promoted_get_separate_records(self, mock_eval, in_memory_engine):
        """Each promoted candidate gets its own AgentMemory sentiment record."""
        mock_eval.return_value = _mock_llm_response(
            freshness="fresh", specific=True, decision="promoted"
        )

        candidates = [
            _make_funnel_candidate(in_memory_engine, "AAPL"),
            _make_funnel_candidate(in_memory_engine, "MSFT"),
            _make_funnel_candidate(in_memory_engine, "GOOGL"),
        ]
        run_funnel_qualification(
            in_memory_engine, candidates, config={}, max_promoted=3
        )

        session = get_session(in_memory_engine)
        records = (
            session.query(AgentMemory)
            .filter_by(agent="researcher", key="sentiment")
            .all()
        )
        symbols_with_records = {r.symbol for r in records}
        assert symbols_with_records == {"AAPL", "MSFT", "GOOGL"}
        session.close()

    @patch("utils.funnel_researcher._evaluate_candidate_catalyst")
    def test_ceiling_rejected_no_sentiment_record(self, mock_eval, in_memory_engine):
        """Candidates rejected due to promotion ceiling do NOT get sentiment records."""
        mock_eval.return_value = _mock_llm_response(
            freshness="fresh", specific=True, decision="promoted"
        )

        candidates = [
            _make_funnel_candidate(in_memory_engine, "AAPL"),
            _make_funnel_candidate(in_memory_engine, "MSFT"),
            _make_funnel_candidate(in_memory_engine, "GOOGL"),
            _make_funnel_candidate(in_memory_engine, "AMZN"),
        ]
        run_funnel_qualification(
            in_memory_engine, candidates, config={}, max_promoted=2
        )

        session = get_session(in_memory_engine)
        records = (
            session.query(AgentMemory)
            .filter_by(agent="researcher", key="sentiment")
            .all()
        )
        symbols_with_records = {r.symbol for r in records}
        # Only first 2 promoted, GOOGL and AMZN rejected due to ceiling
        assert symbols_with_records == {"AAPL", "MSFT"}
        session.close()

    @patch("utils.funnel_researcher._evaluate_candidate_catalyst")
    def test_sentiment_record_readable_by_analyst_query(self, mock_eval, in_memory_engine):
        """Verify the record is readable by the same query pattern the Analyst uses."""
        mock_eval.return_value = _mock_llm_response(
            freshness="fresh", specific=True, decision="promoted"
        )

        candidate = _make_funnel_candidate(in_memory_engine, "AAPL")
        run_funnel_qualification(
            in_memory_engine, [candidate], config={}, max_promoted=3
        )

        # Simulate the Analyst's query pattern (from agents/analyst.py)
        session = get_session(in_memory_engine)
        sentiments = (
            session.query(AgentMemory)
            .filter_by(agent="researcher", key="sentiment")
            .order_by(AgentMemory.timestamp.desc())
            .all()
        )

        # Should find our record
        assert len(sentiments) >= 1
        recent_sentiment = {}
        seen = set()
        for s in sentiments:
            if s.symbol not in seen:
                recent_sentiment[s.symbol] = json.loads(s.value)
                seen.add(s.symbol)

        assert "AAPL" in recent_sentiment
        aapl_sentiment = recent_sentiment["AAPL"]
        assert aapl_sentiment["sentiment"] in {"bullish", "bearish", "neutral"}
        assert isinstance(aapl_sentiment.get("catalysts"), list)
        session.close()
