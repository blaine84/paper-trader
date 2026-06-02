"""
Property-based test for researcher failure isolation (Property 14).

For any set of candidates, if evaluation fails for one candidate, all
previously completed decisions remain persisted and unchanged. Failed
candidate receives not_evaluated decision.

Feature: premarket-candidate-funnel

**Validates: Requirements 4.5**
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from unittest.mock import patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, FunnelCandidate
from utils.funnel_researcher import run_funnel_qualification


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

# Number of candidates: 2 to 10 (need at least 2 to test isolation)
st_num_candidates = st.integers(min_value=2, max_value=10)

# Failure position index (will be bounded by num_candidates via assume)
st_failure_position = st.integers(min_value=0, max_value=9)

# Max promoted ceiling: 1 to 10
st_max_promoted = st.integers(min_value=1, max_value=10)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_engine_and_tables():
    """Create a fresh in-memory SQLite engine with all tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def _make_funnel_candidate(
    engine, symbol: str, rank: int, score: float
) -> FunnelCandidate:
    """Create and persist a FunnelCandidate in awaiting_research status."""
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        candidate = FunnelCandidate(
            candidate_id=str(uuid.uuid4()),
            date=date.today(),
            symbol=symbol,
            discovered_at=datetime.now(timezone.utc),
            source_run="premarket",
            selection_mode="deterministic_fallback",
            scout_rank=rank,
            scout_score=score,
            direction_bias="bullish",
            catalyst_evidence=json.dumps({
                "type": "earnings_beat",
                "detail": f"Strong Q4 earnings for {symbol}",
                "age_minutes": 30,
            }),
            selection_reason=f"Strong momentum candidate: {symbol}",
            primary_risk="Sector rotation risk",
            sector_context=json.dumps({"sector": "technology", "etf": "XLK"}),
            preliminary_setup_type="momentum_breakout",
            stage_status="awaiting_research",
            stage_decisions=json.dumps([{
                "agent": "scout",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "decision": "promoted",
                "reasoning": "High scout score with catalyst",
                "evidence": {"scout_score": score},
                "next_stage": "awaiting_research",
            }]),
        )
        session.add(candidate)
        session.commit()
        session.refresh(candidate)
        return candidate
    finally:
        session.close()


def _get_candidate_from_db(engine, candidate_id: str) -> FunnelCandidate | None:
    """Fetch a FunnelCandidate by candidate_id."""
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        return (
            session.query(FunnelCandidate)
            .filter(FunnelCandidate.candidate_id == candidate_id)
            .first()
        )
    finally:
        session.close()


def _make_success_response() -> str:
    """Generate a valid LLM JSON response for a successful evaluation."""
    return json.dumps({
        "catalyst_freshness": "fresh",
        "catalyst_specific": True,
        "specificity_detail": "Q4 earnings beat with 15% revenue growth",
        "material_evidence": ["Revenue beat by 15%", "Raised guidance"],
        "contradictory_evidence": [],
        "sentiment": "bullish",
        "confidence": "high",
        "catalysts": ["earnings beat"],
        "risks": ["sector rotation"],
        "summary": "Strong catalyst with material evidence",
        "decision": "promoted",
        "reasoning": "Fresh, specific catalyst with strong material evidence supports promotion",
    })


# ---------------------------------------------------------------------------
# Property 14: Researcher failure isolation
# Feature: premarket-candidate-funnel
# ---------------------------------------------------------------------------


class TestProperty14ResearcherFailureIsolation:
    """
    For any set of candidates, if evaluation fails for one candidate, all
    previously completed decisions remain persisted and unchanged. Failed
    candidate receives not_evaluated decision.

    **Validates: Requirements 4.5**
    """

    @given(
        num_candidates=st_num_candidates,
        failure_position=st_failure_position,
        max_promoted=st_max_promoted,
    )
    @settings(max_examples=100)
    def test_failure_isolation_preserves_prior_decisions(
        self,
        num_candidates: int,
        failure_position: int,
        max_promoted: int,
    ):
        """
        When evaluation fails for one candidate at any position, all
        previously completed decisions remain persisted and unchanged in
        the database, and the failed candidate gets not_evaluated.
        """
        # Ensure failure position is within bounds
        assume(failure_position < num_candidates)

        engine = _create_engine_and_tables()

        # Create candidates
        candidates = []
        for i in range(num_candidates):
            symbol = f"T{i:03d}"[:5]
            score = 90.0 - i
            candidate = _make_funnel_candidate(engine, symbol, rank=i + 1, score=score)
            candidates.append(candidate)

        # Track which call we're on to inject failure at the right position
        call_counter = {"count": 0}
        fail_symbol = candidates[failure_position].symbol

        def mock_call_llm(system_prompt, user_prompt, **kwargs):
            """Mock LLM that fails for the candidate at failure_position."""
            call_counter["count"] += 1
            # Check if this call is for the failing symbol
            if fail_symbol in user_prompt:
                raise RuntimeError(f"Simulated LLM failure for {fail_symbol}")
            return _make_success_response()

        # Run qualification with mocked LLM
        with patch("utils.funnel_researcher.call_llm", side_effect=mock_call_llm):
            decisions = run_funnel_qualification(
                engine=engine,
                candidates=candidates,
                config={},
                max_promoted=max_promoted,
            )

        # --- Assertions ---

        # 1. We should get a decision for every candidate
        assert len(decisions) == num_candidates, (
            f"Expected {num_candidates} decisions but got {len(decisions)}"
        )

        # 2. The failed candidate must receive not_evaluated decision
        failed_decision = decisions[failure_position]
        assert failed_decision.decision == "not_evaluated", (
            f"Failed candidate at position {failure_position} should have "
            f"'not_evaluated' but got '{failed_decision.decision}'"
        )
        assert failed_decision.candidate_id == candidates[failure_position].candidate_id

        # 3. All other candidates must have a valid decision (not not_evaluated)
        for i, dec in enumerate(decisions):
            if i == failure_position:
                continue
            assert dec.decision in ("promoted", "rejected", "needs_confirmation"), (
                f"Candidate at position {i} should have a valid decision "
                f"but got '{dec.decision}'"
            )

        # 4. Previously completed decisions (before failure) are persisted in DB
        for i in range(num_candidates):
            db_candidate = _get_candidate_from_db(engine, candidates[i].candidate_id)
            assert db_candidate is not None, (
                f"Candidate {i} should still exist in DB"
            )

            # Parse stage decisions
            stage_decisions = json.loads(db_candidate.stage_decisions)

            if i == failure_position:
                # Failed candidate: should have original scout decision + not_evaluated
                researcher_decisions = [
                    d for d in stage_decisions if d["agent"] == "researcher"
                ]
                assert len(researcher_decisions) == 1, (
                    f"Failed candidate should have exactly 1 researcher decision "
                    f"but got {len(researcher_decisions)}"
                )
                assert researcher_decisions[0]["decision"] == "not_evaluated"
                # Stage status should remain awaiting_research
                assert db_candidate.stage_status == "awaiting_research", (
                    f"Failed candidate should remain in awaiting_research "
                    f"but is in '{db_candidate.stage_status}'"
                )
            else:
                # Non-failed candidates should have a researcher stage decision
                researcher_decisions = [
                    d for d in stage_decisions if d["agent"] == "researcher"
                ]
                assert len(researcher_decisions) == 1, (
                    f"Candidate {i} should have exactly 1 researcher decision "
                    f"but got {len(researcher_decisions)}"
                )
                assert researcher_decisions[0]["decision"] in (
                    "promoted", "rejected", "needs_confirmation"
                ), (
                    f"Candidate {i} researcher decision should be valid "
                    f"but got '{researcher_decisions[0]['decision']}'"
                )

        # 5. The original scout decision is always preserved for all candidates
        for i in range(num_candidates):
            db_candidate = _get_candidate_from_db(engine, candidates[i].candidate_id)
            stage_decisions = json.loads(db_candidate.stage_decisions)
            scout_decisions = [d for d in stage_decisions if d["agent"] == "scout"]
            assert len(scout_decisions) == 1, (
                f"Candidate {i} should retain exactly 1 scout decision "
                f"but got {len(scout_decisions)}"
            )
            # Scout decision evidence should be unchanged
            assert scout_decisions[0]["decision"] == "promoted"
            assert scout_decisions[0]["reasoning"] == "High scout score with catalyst"
