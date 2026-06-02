"""
Property-based test for timeout never clears existing state (Property 4).

For any timeout event, all FunnelCandidate rows persisted before timeout remain
with stage_decisions intact and stage_status unchanged. No candidate is deleted
or cleared.

Feature: premarket-candidate-funnel

**Validates: Requirements 2.2, 6.5, 6.6, 7.3**
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

from hypothesis import given, settings, assume
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, FunnelCandidate, get_session
from utils.funnel_confirmation import run_opening_confirmation


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

# Generate between 1 and 8 candidates to simulate various shortlist sizes
st_num_candidates = st.integers(min_value=1, max_value=8)

# Budget in seconds — use very small budgets to force timeout during test
# 0 means immediate timeout before any candidate is processed
st_budget_seconds = st.integers(min_value=0, max_value=2)

# Valid stage_status values for candidates already in the pipeline
st_prior_stage_status = st.just("awaiting_confirmation")

# Scout scores for candidates
st_scout_score = st.floats(min_value=0.0, max_value=100.0, allow_nan=False)

# Scout ranks (1-based)
st_scout_rank = st.integers(min_value=1, max_value=5)

# Direction bias
st_direction_bias = st.sampled_from(["bullish", "bearish", "neutral", None])

# Generate prior stage decisions (1-3 existing decisions per candidate)
st_prior_decision = st.fixed_dictionaries({
    "agent": st.sampled_from(["scout", "researcher", "analyst"]),
    "timestamp": st.just(datetime.now(timezone.utc).isoformat()),
    "decision": st.sampled_from(["promoted", "needs_confirmation"]),
    "reasoning": st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        min_size=1,
        max_size=100,
    ),
    "evidence": st.fixed_dictionaries({
        "score": st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    }),
    "next_stage": st.sampled_from([
        "awaiting_research", "awaiting_analysis", "awaiting_confirmation",
    ]),
})

st_prior_decisions_list = st.lists(st_prior_decision, min_size=1, max_size=3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_engine_and_tables():
    """Create a fresh in-memory SQLite engine with all tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def _make_candidates(
    engine,
    num: int,
    scout_scores: list[float],
    scout_ranks: list[int],
    direction_biases: list[str | None],
    prior_decisions_lists: list[list[dict]],
) -> list[FunnelCandidate]:
    """Create and persist multiple FunnelCandidate rows with prior decisions."""
    ny_tz = ZoneInfo("America/New_York")
    today_ny = datetime.now(timezone.utc).astimezone(ny_tz).date()
    candidates = []

    session = get_session(engine)
    try:
        for i in range(num):
            candidate = FunnelCandidate(
                candidate_id=str(uuid.uuid4()),
                date=today_ny,
                symbol=f"T{i:03d}",
                discovered_at=datetime.now(timezone.utc),
                source_run="premarket",
                selection_mode="deterministic_fallback",
                scout_rank=scout_ranks[i],
                scout_score=scout_scores[i],
                direction_bias=direction_biases[i],
                catalyst_evidence=json.dumps({"headline": f"Catalyst {i}"}),
                selection_reason=f"Selection reason {i}",
                primary_risk=f"Risk {i}",
                sector_context=json.dumps({"sector": "technology"}),
                preliminary_setup_type="momentum_breakout",
                stage_status="awaiting_confirmation",
                stage_decisions=json.dumps(prior_decisions_lists[i]),
                expired=False,
            )
            session.add(candidate)
            candidates.append(candidate)
        session.commit()
        for c in candidates:
            session.refresh(c)
        return candidates
    finally:
        session.close()


def _get_all_candidates(engine) -> list[FunnelCandidate]:
    """Fetch all FunnelCandidate rows from DB."""
    session = get_session(engine)
    try:
        return session.query(FunnelCandidate).all()
    finally:
        session.close()


def _get_candidate_by_id(engine, candidate_id: str) -> FunnelCandidate | None:
    """Fetch a specific FunnelCandidate by candidate_id."""
    session = get_session(engine)
    try:
        return (
            session.query(FunnelCandidate)
            .filter(FunnelCandidate.candidate_id == candidate_id)
            .first()
        )
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Property 4: Timeout never clears existing state
# Feature: premarket-candidate-funnel
# ---------------------------------------------------------------------------


class TestProperty4TimeoutNeverClearsExistingState:
    """
    For any timeout event, all FunnelCandidate rows persisted before timeout
    remain with stage_decisions intact and stage_status unchanged. No candidate
    is deleted or cleared.

    **Validates: Requirements 2.2, 6.5, 6.6, 7.3**
    """

    @given(
        num_candidates=st_num_candidates,
        scout_scores=st.lists(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
            min_size=8, max_size=8,
        ),
        scout_ranks=st.lists(
            st.integers(min_value=1, max_value=5),
            min_size=8, max_size=8,
        ),
        direction_biases=st.lists(st_direction_bias, min_size=8, max_size=8),
        prior_decisions_lists=st.lists(
            st_prior_decisions_list, min_size=8, max_size=8,
        ),
    )
    @settings(max_examples=50)
    def test_all_candidates_still_exist_after_timeout(
        self,
        num_candidates: int,
        scout_scores: list[float],
        scout_ranks: list[int],
        direction_biases: list[str | None],
        prior_decisions_lists: list[list[dict]],
    ):
        """
        After a timeout occurs during confirmation, all FunnelCandidate rows
        that existed before the timeout still exist in the database. No rows
        are deleted.

        **Validates: Requirements 2.2, 6.5, 6.6, 7.3**
        """
        engine = _create_engine_and_tables()

        candidates = _make_candidates(
            engine, num_candidates,
            scout_scores[:num_candidates],
            scout_ranks[:num_candidates],
            direction_biases[:num_candidates],
            prior_decisions_lists[:num_candidates],
        )

        candidate_ids_before = {c.candidate_id for c in candidates}

        # Run with budget=0 to force immediate timeout
        with patch(
            "utils.funnel_confirmation._get_live_data",
            side_effect=lambda sym: {"price": 100.0, "volume": 1000000, "open": 99.0},
        ):
            run_opening_confirmation(engine, candidates, budget_seconds=0)

        # PROPERTY: All candidates still exist
        db_candidates = _get_all_candidates(engine)
        candidate_ids_after = {c.candidate_id for c in db_candidates}

        assert candidate_ids_before == candidate_ids_after, (
            f"Candidates were deleted during timeout!\n"
            f"Before: {candidate_ids_before}\n"
            f"After:  {candidate_ids_after}\n"
            f"Missing: {candidate_ids_before - candidate_ids_after}"
        )

    @given(
        num_candidates=st_num_candidates,
        scout_scores=st.lists(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
            min_size=8, max_size=8,
        ),
        scout_ranks=st.lists(
            st.integers(min_value=1, max_value=5),
            min_size=8, max_size=8,
        ),
        direction_biases=st.lists(st_direction_bias, min_size=8, max_size=8),
        prior_decisions_lists=st.lists(
            st_prior_decisions_list, min_size=8, max_size=8,
        ),
    )
    @settings(max_examples=50)
    def test_stage_decisions_not_lost_after_timeout(
        self,
        num_candidates: int,
        scout_scores: list[float],
        scout_ranks: list[int],
        direction_biases: list[str | None],
        prior_decisions_lists: list[list[dict]],
    ):
        """
        After a timeout, stage_decisions arrays for all candidates have NOT
        lost any prior entries. The array length is >= prior length and all
        prior entries are preserved.

        **Validates: Requirements 2.2, 6.5, 6.6, 7.3**
        """
        engine = _create_engine_and_tables()

        candidates = _make_candidates(
            engine, num_candidates,
            scout_scores[:num_candidates],
            scout_ranks[:num_candidates],
            direction_biases[:num_candidates],
            prior_decisions_lists[:num_candidates],
        )

        # Record prior state for each candidate
        prior_state = {}
        for c in candidates:
            prior_decisions = json.loads(c.stage_decisions)
            prior_state[c.candidate_id] = {
                "decisions": prior_decisions,
                "serialized": [json.dumps(d, sort_keys=True) for d in prior_decisions],
            }

        # Run with budget=0 to force immediate timeout
        with patch(
            "utils.funnel_confirmation._get_live_data",
            side_effect=lambda sym: {"price": 100.0, "volume": 1000000, "open": 99.0},
        ):
            run_opening_confirmation(engine, candidates, budget_seconds=0)

        # PROPERTY: No candidate lost any prior stage_decisions entry
        for cid, prior in prior_state.items():
            db_candidate = _get_candidate_by_id(engine, cid)
            assert db_candidate is not None, f"Candidate {cid} was deleted!"

            post_decisions = json.loads(db_candidate.stage_decisions)
            post_serialized = [json.dumps(d, sort_keys=True) for d in post_decisions]

            # Array length must be >= prior (append-only, may have not_evaluated added)
            assert len(post_decisions) >= len(prior["decisions"]), (
                f"Candidate {cid}: stage_decisions lost entries!\n"
                f"Prior length: {len(prior['decisions'])}\n"
                f"Post length: {len(post_decisions)}"
            )

            # All prior entries must be preserved byte-identical
            for i, prior_entry in enumerate(prior["serialized"]):
                assert post_serialized[i] == prior_entry, (
                    f"Candidate {cid}: prior decision at index {i} was modified!\n"
                    f"Before: {prior_entry}\n"
                    f"After:  {post_serialized[i]}"
                )

    @given(
        num_candidates=st.integers(min_value=2, max_value=8),
        scout_scores=st.lists(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
            min_size=8, max_size=8,
        ),
        scout_ranks=st.lists(
            st.integers(min_value=1, max_value=5),
            min_size=8, max_size=8,
        ),
        direction_biases=st.lists(st_direction_bias, min_size=8, max_size=8),
        prior_decisions_lists=st.lists(
            st_prior_decisions_list, min_size=8, max_size=8,
        ),
    )
    @settings(max_examples=50)
    def test_unreached_candidates_stage_status_unchanged(
        self,
        num_candidates: int,
        scout_scores: list[float],
        scout_ranks: list[int],
        direction_biases: list[str | None],
        prior_decisions_lists: list[list[dict]],
    ):
        """
        Candidates that are not reached due to budget exhaustion still have
        their original stage_status (awaiting_confirmation) unchanged. The
        timeout does not change their state to any other value.

        **Validates: Requirements 2.2, 6.5, 6.6, 7.3**
        """
        engine = _create_engine_and_tables()

        candidates = _make_candidates(
            engine, num_candidates,
            scout_scores[:num_candidates],
            scout_ranks[:num_candidates],
            direction_biases[:num_candidates],
            prior_decisions_lists[:num_candidates],
        )

        # Run with budget=0 so ALL candidates hit the timeout path
        with patch(
            "utils.funnel_confirmation._get_live_data",
            side_effect=lambda sym: {"price": 100.0, "volume": 1000000, "open": 99.0},
        ):
            run_opening_confirmation(engine, candidates, budget_seconds=0)

        # PROPERTY: All candidates retain stage_status=awaiting_confirmation
        for c in candidates:
            db_candidate = _get_candidate_by_id(engine, c.candidate_id)
            assert db_candidate is not None, f"Candidate {c.candidate_id} was deleted!"
            assert db_candidate.stage_status == "awaiting_confirmation", (
                f"Candidate {c.candidate_id} ({c.symbol}): stage_status changed "
                f"from 'awaiting_confirmation' to '{db_candidate.stage_status}' "
                f"after timeout!"
            )
