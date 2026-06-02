"""
Property-based test for stage decisions append-only (Property 3).

For any candidate and any stage decision append operation, resulting array
length equals prior length plus one, and all prior elements are byte-identical.

Feature: premarket-candidate-funnel

**Validates: Requirements 3.3, 3.5, 10.1**
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, FunnelCandidate, get_session
from utils.funnel_researcher import (
    _append_stage_decision as researcher_append_stage_decision,
)
from utils.funnel_analyst import (
    _append_stage_decision as analyst_append_stage_decision,
)


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

# Agent types that produce stage decisions
st_agent_source = st.sampled_from(["researcher", "analyst"])

# Valid decision values per the spec
st_decision = st.sampled_from([
    "promoted", "rejected", "needs_confirmation", "timed_out", "not_evaluated",
])

# Valid next_stage values
st_next_stage = st.sampled_from([
    "awaiting_research",
    "awaiting_analysis",
    "awaiting_confirmation",
    "pm_eligible",
    "rejected_research",
    "rejected_analysis",
    "rejected_confirmation",
])

# Reasoning text
st_reasoning = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=200,
)

# Evidence dict (simple but non-trivial)
st_evidence = st.fixed_dictionaries({
    "score": st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    "notes": st.text(min_size=0, max_size=50),
})

# Number of sequential appends to test (1 to 10)
st_num_appends = st.integers(min_value=1, max_value=10)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_engine_and_tables():
    """Create a fresh in-memory SQLite engine with all tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def _make_funnel_candidate(engine) -> FunnelCandidate:
    """Create and persist a FunnelCandidate with an initial scout decision."""
    ny_tz = ZoneInfo("America/New_York")
    today_ny = datetime.now(timezone.utc).astimezone(ny_tz).date()

    scout_decision = {
        "agent": "scout",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": "promoted",
        "reasoning": "Discovery promotion",
        "evidence": {"scout_score": 75.0, "direction_bias": "bullish"},
        "next_stage": "awaiting_research",
    }

    session = get_session(engine)
    try:
        candidate = FunnelCandidate(
            candidate_id=str(uuid.uuid4()),
            date=today_ny,
            symbol=f"S{uuid.uuid4().hex[:4].upper()}",
            discovered_at=datetime.now(timezone.utc),
            source_run="premarket",
            selection_mode="deterministic_fallback",
            scout_rank=1,
            scout_score=75.0,
            direction_bias="bullish",
            catalyst_evidence=json.dumps({"headline": "Test catalyst"}),
            selection_reason="Test selection reason",
            primary_risk="Market risk",
            sector_context=json.dumps({"sector": "technology"}),
            preliminary_setup_type="momentum_breakout",
            stage_status="awaiting_research",
            stage_decisions=json.dumps([scout_decision]),
            expired=False,
        )
        session.add(candidate)
        session.commit()
        session.refresh(candidate)
        return candidate
    finally:
        session.close()


def _get_stage_decisions(engine, candidate_id: str) -> list[dict]:
    """Read current stage_decisions from DB for a candidate."""
    session = get_session(engine)
    try:
        row = (
            session.query(FunnelCandidate)
            .filter(FunnelCandidate.candidate_id == candidate_id)
            .first()
        )
        return json.loads(row.stage_decisions) if row else []
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Property 3: Stage decisions are append-only
# Feature: premarket-candidate-funnel
# ---------------------------------------------------------------------------


class TestProperty3StageDecisionsAppendOnly:
    """
    For any candidate and any stage decision append operation, resulting array
    length equals prior length plus one, and all prior elements are
    byte-identical.

    **Validates: Requirements 3.3, 3.5, 10.1**
    """

    @given(
        decision=st_decision,
        reasoning=st_reasoning,
        evidence=st_evidence,
        next_stage=st_next_stage,
        agent_source=st_agent_source,
    )
    @settings(max_examples=100)
    def test_single_append_increases_length_by_one(
        self,
        decision: str,
        reasoning: str,
        evidence: dict,
        next_stage: str,
        agent_source: str,
    ):
        """
        A single append operation increases stage_decisions array length by
        exactly one regardless of the decision content.

        **Validates: Requirements 3.3, 3.5, 10.1**
        """
        engine = _create_engine_and_tables()
        candidate = _make_funnel_candidate(engine)

        # Get prior state
        prior_decisions = _get_stage_decisions(engine, candidate.candidate_id)
        prior_length = len(prior_decisions)

        # Perform append using the appropriate agent's function
        append_fn = (
            researcher_append_stage_decision
            if agent_source == "researcher"
            else analyst_append_stage_decision
        )
        append_fn(
            engine=engine,
            candidate=candidate,
            decision=decision,
            reasoning=reasoning,
            evidence=evidence,
            next_stage=next_stage,
        )

        # Get post state
        post_decisions = _get_stage_decisions(engine, candidate.candidate_id)

        # PROPERTY: length increases by exactly 1
        assert len(post_decisions) == prior_length + 1, (
            f"Expected length {prior_length + 1} after append, "
            f"got {len(post_decisions)}"
        )

    @given(
        decision=st_decision,
        reasoning=st_reasoning,
        evidence=st_evidence,
        next_stage=st_next_stage,
        agent_source=st_agent_source,
    )
    @settings(max_examples=100)
    def test_single_append_preserves_prior_elements(
        self,
        decision: str,
        reasoning: str,
        evidence: dict,
        next_stage: str,
        agent_source: str,
    ):
        """
        After appending a stage decision, all prior elements remain
        byte-identical (serialized JSON comparison).

        **Validates: Requirements 3.3, 3.5, 10.1**
        """
        engine = _create_engine_and_tables()
        candidate = _make_funnel_candidate(engine)

        # Get prior state as serialized JSON for byte-identical comparison
        prior_decisions = _get_stage_decisions(engine, candidate.candidate_id)
        prior_serialized = [json.dumps(d, sort_keys=True) for d in prior_decisions]

        # Perform append
        append_fn = (
            researcher_append_stage_decision
            if agent_source == "researcher"
            else analyst_append_stage_decision
        )
        append_fn(
            engine=engine,
            candidate=candidate,
            decision=decision,
            reasoning=reasoning,
            evidence=evidence,
            next_stage=next_stage,
        )

        # Get post state
        post_decisions = _get_stage_decisions(engine, candidate.candidate_id)
        post_serialized = [json.dumps(d, sort_keys=True) for d in post_decisions]

        # PROPERTY: all prior elements are byte-identical
        for i, (prior, post) in enumerate(zip(prior_serialized, post_serialized)):
            assert prior == post, (
                f"Prior decision at index {i} was modified after append.\n"
                f"Before: {prior}\n"
                f"After:  {post}"
            )

    @given(
        num_appends=st_num_appends,
        decisions=st.lists(st_decision, min_size=10, max_size=10),
        reasonings=st.lists(st_reasoning, min_size=10, max_size=10),
        evidences=st.lists(st_evidence, min_size=10, max_size=10),
        next_stages=st.lists(st_next_stage, min_size=10, max_size=10),
        agent_sources=st.lists(st_agent_source, min_size=10, max_size=10),
    )
    @settings(max_examples=50)
    def test_multiple_appends_preserve_all_prior_entries(
        self,
        num_appends: int,
        decisions: list,
        reasonings: list,
        evidences: list,
        next_stages: list,
        agent_sources: list,
    ):
        """
        After N sequential appends, the array has exactly initial_length + N
        elements and all elements from before each append are unchanged.

        **Validates: Requirements 3.3, 3.5, 10.1**
        """
        engine = _create_engine_and_tables()
        candidate = _make_funnel_candidate(engine)

        # Get initial state
        initial_decisions = _get_stage_decisions(engine, candidate.candidate_id)
        initial_length = len(initial_decisions)
        initial_serialized = [json.dumps(d, sort_keys=True) for d in initial_decisions]

        # Perform N sequential appends
        for i in range(num_appends):
            append_fn = (
                researcher_append_stage_decision
                if agent_sources[i] == "researcher"
                else analyst_append_stage_decision
            )
            append_fn(
                engine=engine,
                candidate=candidate,
                decision=decisions[i],
                reasoning=reasonings[i],
                evidence=evidences[i],
                next_stage=next_stages[i],
            )

        # Get final state
        final_decisions = _get_stage_decisions(engine, candidate.candidate_id)

        # PROPERTY: final length equals initial_length + num_appends
        assert len(final_decisions) == initial_length + num_appends, (
            f"Expected {initial_length + num_appends} decisions after "
            f"{num_appends} appends (starting from {initial_length}), "
            f"got {len(final_decisions)}"
        )

        # PROPERTY: all initial elements remain byte-identical
        final_serialized = [json.dumps(d, sort_keys=True) for d in final_decisions]
        for i, initial in enumerate(initial_serialized):
            assert final_serialized[i] == initial, (
                f"Initial decision at index {i} was modified after "
                f"{num_appends} appends.\n"
                f"Original: {initial}\n"
                f"Current:  {final_serialized[i]}"
            )

        # PROPERTY: no prior entry is deleted or modified between appends
        # Each appended entry appears in order after initial entries
        for j in range(num_appends):
            appended_entry = final_decisions[initial_length + j]
            expected_agent = agent_sources[j]
            assert appended_entry["agent"] == expected_agent, (
                f"Appended entry at position {initial_length + j} has agent "
                f"'{appended_entry['agent']}' but expected '{expected_agent}'"
            )
            assert appended_entry["decision"] == decisions[j], (
                f"Appended entry at position {initial_length + j} has decision "
                f"'{appended_entry['decision']}' but expected '{decisions[j]}'"
            )
