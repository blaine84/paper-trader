"""
Property-based test for researcher promotion ceiling (Property 2).

For any qualification run with any number of input candidates, promoted count
never exceeds max_researcher_promoted.

Feature: premarket-candidate-funnel

**Validates: Requirements 4.3**
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, FunnelCandidate
from utils.funnel_researcher import run_funnel_qualification, QualificationDecision


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

# max_promoted ceiling: 1 to 10
st_max_promoted = st.integers(min_value=1, max_value=10)

# Number of input candidates: 1 to 30 (can greatly exceed ceiling)
st_num_candidates = st.integers(min_value=1, max_value=30)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_engine_and_tables():
    """Create a fresh in-memory SQLite engine with all tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def _make_funnel_candidate(engine, index: int) -> FunnelCandidate:
    """Create and persist a FunnelCandidate in awaiting_research status."""
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        candidate = FunnelCandidate(
            candidate_id=str(uuid.uuid4()),
            date=date.today(),
            symbol=f"T{index:03d}"[:5],
            discovered_at=datetime.now(timezone.utc),
            source_run="premarket",
            selection_mode="deterministic_fallback",
            scout_rank=index + 1,
            scout_score=80.0 - index,
            direction_bias="bullish",
            catalyst_evidence=json.dumps({
                "headline": f"Test catalyst for T{index:03d}",
                "age_minutes": 30,
            }),
            selection_reason=f"Strong catalyst for testing candidate {index}",
            primary_risk="Market risk",
            sector_context=json.dumps({"sector": "technology", "etf": "XLK"}),
            preliminary_setup_type="momentum_breakout",
            stage_status="awaiting_research",
            stage_decisions=json.dumps([{
                "agent": "scout",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "decision": "promoted",
                "reasoning": "Discovery promotion",
                "evidence": {},
                "next_stage": "awaiting_research",
            }]),
            expired=False,
        )
        session.add(candidate)
        session.commit()
        session.refresh(candidate)
        return candidate
    finally:
        session.close()


def _mock_llm_always_promote(*args, **kwargs) -> str:
    """Mock LLM response that always returns fresh + specific + promoted."""
    return json.dumps({
        "catalyst_freshness": "fresh",
        "catalyst_specific": True,
        "specificity_detail": "FDA approval announcement",
        "material_evidence": ["Revenue beat by 20%", "Guidance raised"],
        "contradictory_evidence": [],
        "sentiment": "bullish",
        "confidence": "high",
        "catalysts": ["FDA approval"],
        "risks": ["Market risk"],
        "summary": "Strong fresh catalyst with specific named event.",
        "decision": "promoted",
        "reasoning": "Fresh, specific catalyst with material evidence supports promotion.",
    })


# ---------------------------------------------------------------------------
# Property 2: Researcher promotion ceiling enforced
# Feature: premarket-candidate-funnel
# ---------------------------------------------------------------------------


class TestProperty2ResearcherPromotionCeiling:
    """
    For any qualification run with any number of input candidates, promoted
    count never exceeds max_researcher_promoted.

    **Validates: Requirements 4.3**
    """

    @given(
        num_candidates=st_num_candidates,
        max_promoted=st_max_promoted,
    )
    @settings(max_examples=100)
    @patch("utils.funnel_researcher._write_sentiment_memory")
    @patch("utils.funnel_researcher.call_llm")
    def test_promoted_count_never_exceeds_max_promoted(
        self,
        mock_call_llm,
        mock_write_sentiment,
        num_candidates: int,
        max_promoted: int,
    ):
        """
        run_funnel_qualification() with any number of candidates where all
        would qualify (mocked LLM always returns fresh+specific+promoted)
        never promotes more than max_promoted candidates.

        **Validates: Requirements 4.3**
        """
        # Mock LLM to always return "promote" decision
        mock_call_llm.side_effect = _mock_llm_always_promote
        # Mock sentiment memory write (task 5.2, not relevant to ceiling property)
        mock_write_sentiment.return_value = None

        engine = _create_engine_and_tables()

        # Create N candidates in the database
        candidates = [
            _make_funnel_candidate(engine, i)
            for i in range(num_candidates)
        ]

        # Run qualification with the ceiling
        decisions = run_funnel_qualification(
            engine=engine,
            candidates=candidates,
            config={},
            max_promoted=max_promoted,
        )

        # Count promoted decisions
        promoted_decisions = [
            d for d in decisions if d.decision == "promoted"
        ]

        # PROPERTY: promoted count NEVER exceeds max_promoted
        assert len(promoted_decisions) <= max_promoted, (
            f"run_funnel_qualification promoted {len(promoted_decisions)} candidates "
            f"but max_promoted ceiling is {max_promoted} "
            f"(input candidates: {num_candidates})"
        )

        # Additional invariant: promoted count equals min(num_candidates, max_promoted)
        # because ALL candidates would qualify (LLM always says promote)
        expected_promoted = min(num_candidates, max_promoted)
        assert len(promoted_decisions) == expected_promoted, (
            f"Expected {expected_promoted} promotions "
            f"(min({num_candidates}, {max_promoted})) "
            f"but got {len(promoted_decisions)}"
        )

        # Verify excess candidates are rejected with "promotion_ceiling_reached"
        if num_candidates > max_promoted:
            ceiling_rejections = [
                d for d in decisions
                if d.decision == "rejected"
                and "promotion_ceiling_reached" in d.reasoning
            ]
            expected_ceiling_rejections = num_candidates - max_promoted
            assert len(ceiling_rejections) == expected_ceiling_rejections, (
                f"Expected {expected_ceiling_rejections} ceiling rejections "
                f"but got {len(ceiling_rejections)}"
            )
