"""
Property-based test for deterministic fallback non-empty (Property 12).

For any discovery run where at least one finalist passes screening and
Chief Scout fails/times out, persisted candidate list is non-empty.

Feature: premarket-candidate-funnel

**Validates: Requirements 2.1, 2.2**
"""

from __future__ import annotations

from dataclasses import field
from unittest.mock import patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from utils.funnel_discovery import (
    _apply_deterministic_fallback,
    run_chief_scout_curation,
)
from utils.sector_scout_models import CandidateRow


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

# Generate a valid scout_score (0.0 to 100.0, no NaN/Inf)
st_scout_score = st.floats(
    min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
)

# Generate a ticker symbol: 1-5 uppercase letters
st_symbol = st.text(
    alphabet=st.sampled_from("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    min_size=1,
    max_size=5,
)

# Generate a sector name
st_sector = st.sampled_from([
    "technology", "healthcare", "financials", "energy",
    "consumer_discretionary", "industrials", "materials",
])

# Max shortlist ceiling (1 to 10)
st_max_shortlist = st.integers(min_value=1, max_value=10)


# Strategy for a single CandidateRow with required fields
@st.composite
def st_candidate_row(draw):
    """Generate a valid CandidateRow with randomized scoring and identity."""
    symbol = draw(st_symbol)
    sector = draw(st_sector)
    score = draw(st_scout_score)
    return CandidateRow(
        symbol=symbol,
        sector=sector,
        sector_name=sector.replace("_", " ").title(),
        scout_score=score,
        hard_gate_passed=True,
    )


# Strategy for a non-empty list of finalist CandidateRows (at least 1)
st_finalists = st.lists(st_candidate_row(), min_size=1, max_size=20)


# ---------------------------------------------------------------------------
# Property 12: Deterministic fallback produces non-empty results when
#              finalists exist
# Feature: premarket-candidate-funnel
# ---------------------------------------------------------------------------


class TestProperty12DeterministicFallbackNonEmpty:
    """
    For any discovery run where at least one finalist passes screening
    and Chief Scout fails/times out, persisted candidate list is non-empty.

    **Validates: Requirements 2.1, 2.2**
    """

    @given(
        finalists=st_finalists,
        max_shortlist=st_max_shortlist,
    )
    @settings(max_examples=200)
    def test_deterministic_fallback_always_non_empty(
        self, finalists: list[CandidateRow], max_shortlist: int
    ):
        """_apply_deterministic_fallback returns non-empty list when finalists exist."""
        result = _apply_deterministic_fallback(finalists, max_shortlist)

        # Core property: when finalists exist, fallback is never empty
        assert len(result) > 0, (
            f"Deterministic fallback returned empty list with {len(finalists)} "
            f"finalists and max_shortlist={max_shortlist}"
        )
        # Additional: result is bounded by both input size and max_shortlist
        assert len(result) <= min(len(finalists), max_shortlist)

    @given(
        finalists=st_finalists,
        max_shortlist=st_max_shortlist,
    )
    @settings(max_examples=200)
    def test_run_chief_scout_curation_fallback_non_empty_on_timeout(
        self, finalists: list[CandidateRow], max_shortlist: int
    ):
        """When Chief Scout times out but finalists exist, result is non-empty."""
        # Simulate timeout by patching _invoke_chief_scout_with_timeout to raise
        with patch(
            "utils.funnel_discovery._invoke_chief_scout_with_timeout",
            side_effect=TimeoutError("Simulated timeout"),
        ):
            curated_list, selection_mode, curation_error = run_chief_scout_curation(
                finalists=finalists,
                remaining_budget=30.0,  # >10s so curation is attempted
                config={},
                engine=None,
                core_watchlist=[],
                max_shortlist=max_shortlist,
            )

        # Core property: non-empty when finalists exist and curation fails
        assert len(curated_list) > 0, (
            f"run_chief_scout_curation returned empty list on timeout with "
            f"{len(finalists)} finalists and max_shortlist={max_shortlist}"
        )
        assert selection_mode == "deterministic_fallback"
        assert curation_error is not None
        assert len(curated_list) <= min(len(finalists), max_shortlist)

    @given(
        finalists=st_finalists,
        max_shortlist=st_max_shortlist,
    )
    @settings(max_examples=200)
    def test_run_chief_scout_curation_fallback_non_empty_on_error(
        self, finalists: list[CandidateRow], max_shortlist: int
    ):
        """When Chief Scout raises an exception but finalists exist, result is non-empty."""
        # Simulate LLM error by patching _invoke_chief_scout_with_timeout to raise
        with patch(
            "utils.funnel_discovery._invoke_chief_scout_with_timeout",
            side_effect=RuntimeError("LLM service unavailable"),
        ):
            curated_list, selection_mode, curation_error = run_chief_scout_curation(
                finalists=finalists,
                remaining_budget=30.0,  # >10s so curation is attempted
                config={},
                engine=None,
                core_watchlist=[],
                max_shortlist=max_shortlist,
            )

        # Core property: non-empty when finalists exist and curation fails
        assert len(curated_list) > 0, (
            f"run_chief_scout_curation returned empty list on error with "
            f"{len(finalists)} finalists and max_shortlist={max_shortlist}"
        )
        assert selection_mode == "deterministic_fallback"
        assert curation_error is not None
        assert len(curated_list) <= min(len(finalists), max_shortlist)

    @given(
        finalists=st_finalists,
        max_shortlist=st_max_shortlist,
        remaining_budget=st.floats(
            min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=200)
    def test_run_chief_scout_curation_fallback_non_empty_insufficient_budget(
        self,
        finalists: list[CandidateRow],
        max_shortlist: int,
        remaining_budget: float,
    ):
        """When budget is insufficient for curation (<=10s), fallback is non-empty."""
        curated_list, selection_mode, curation_error = run_chief_scout_curation(
            finalists=finalists,
            remaining_budget=remaining_budget,
            config={},
            engine=None,
            core_watchlist=[],
            max_shortlist=max_shortlist,
        )

        # Core property: non-empty when finalists exist regardless of budget
        assert len(curated_list) > 0, (
            f"run_chief_scout_curation returned empty list with insufficient budget "
            f"({remaining_budget}s) with {len(finalists)} finalists"
        )
        assert selection_mode == "deterministic_fallback"
        assert len(curated_list) <= min(len(finalists), max_shortlist)
