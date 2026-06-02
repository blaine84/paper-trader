"""Unit tests for Chief Scout curation with deterministic fallback (Task 3.2).

Tests the run_chief_scout_curation() function and _apply_deterministic_fallback()
to verify:
- Deterministic fallback is used when budget <= 10s
- Deterministic fallback is used when Chief Scout fails/times out
- Chief Scout curation is used when budget > 10s and LLM succeeds
- Fallback ranking: scout_score DESC, symbol ASC for ties
- selection_mode is set correctly in each scenario
- Max shortlist ceiling is enforced

See: requirements 1.10, 1.11, 2.1, 2.2, 2.3
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from utils.funnel_discovery import (
    run_chief_scout_curation,
    _apply_deterministic_fallback,
    _deterministic_fallback_sort_key,
)
from utils.sector_scout_models import CandidateRow, ChiefScoutPick


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_candidate(
    symbol: str = "AAPL",
    sector: str = "tech",
    scout_score: float = 50.0,
) -> CandidateRow:
    """Create a minimal CandidateRow for curation tests."""
    return CandidateRow(
        symbol=symbol,
        sector=sector,
        sector_name="Technology",
        scout_score=scout_score,
    )


@pytest.fixture
def finalists() -> list[CandidateRow]:
    """A set of finalists with varying scores."""
    return [
        _make_candidate(symbol="NVDA", sector="ai_semi", scout_score=85.0),
        _make_candidate(symbol="AVGO", sector="ai_semi", scout_score=72.0),
        _make_candidate(symbol="TSLA", sector="ev", scout_score=68.0),
        _make_candidate(symbol="AMD", sector="ai_semi", scout_score=72.0),  # tie with AVGO
        _make_candidate(symbol="AAPL", sector="tech", scout_score=55.0),
        _make_candidate(symbol="META", sector="tech", scout_score=60.0),
    ]


@pytest.fixture
def mock_engine():
    """Mock SQLAlchemy engine."""
    return MagicMock()


@pytest.fixture
def default_config() -> dict:
    """Minimal sector scout config."""
    return {
        "chief_scout": {
            "max_picks": 8,
        },
        "budget_ceilings": {
            "max_expanded_watchlist": 12,
        },
    }


# ---------------------------------------------------------------------------
# Tests: _apply_deterministic_fallback
# ---------------------------------------------------------------------------


class TestDeterministicFallback:
    """Tests for deterministic fallback ranking logic."""

    def test_ranks_by_score_descending(self, finalists):
        """Finalists should be ranked by scout_score descending."""
        ranked = _apply_deterministic_fallback(finalists, max_shortlist=10)

        scores = [r.scout_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_ties_broken_by_symbol_ascending(self):
        """When scores are equal, symbol name ascending breaks the tie."""
        tied = [
            _make_candidate(symbol="TSLA", scout_score=72.0),
            _make_candidate(symbol="AMD", scout_score=72.0),
            _make_candidate(symbol="AVGO", scout_score=72.0),
        ]
        ranked = _apply_deterministic_fallback(tied, max_shortlist=10)

        symbols = [r.symbol for r in ranked]
        assert symbols == ["AMD", "AVGO", "TSLA"]

    def test_max_shortlist_ceiling_enforced(self, finalists):
        """Should return at most max_shortlist candidates."""
        ranked = _apply_deterministic_fallback(finalists, max_shortlist=3)

        assert len(ranked) == 3
        # Top 3 by score: NVDA(85), AMD(72), AVGO(72) — ties: AMD < AVGO
        assert ranked[0].symbol == "NVDA"
        assert ranked[1].symbol == "AMD"
        assert ranked[2].symbol == "AVGO"

    def test_empty_finalists_returns_empty(self):
        """Empty input produces empty output."""
        ranked = _apply_deterministic_fallback([], max_shortlist=5)
        assert ranked == []

    def test_fewer_than_max_returns_all(self):
        """When fewer candidates than max_shortlist, return all."""
        two = [
            _make_candidate(symbol="A", scout_score=80.0),
            _make_candidate(symbol="B", scout_score=60.0),
        ]
        ranked = _apply_deterministic_fallback(two, max_shortlist=5)
        assert len(ranked) == 2

    def test_deterministic_sort_key(self):
        """Sort key produces correct tuple for ranking."""
        c = _make_candidate(symbol="NVDA", scout_score=85.0)
        key = _deterministic_fallback_sort_key(c)
        assert key == (-85.0, "NVDA")


# ---------------------------------------------------------------------------
# Tests: run_chief_scout_curation — budget checks
# ---------------------------------------------------------------------------


class TestCurationBudgetChecks:
    """Tests for budget-based curation decisions."""

    def test_insufficient_budget_uses_fallback(self, finalists, mock_engine, default_config):
        """When remaining_budget <= 10s, should skip LLM and use fallback."""
        curated, mode, curation_error = run_chief_scout_curation(
            finalists=finalists,
            remaining_budget=10.0,  # exactly 10 — not > 10
            config=default_config,
            engine=mock_engine,
            max_shortlist=5,
        )

        assert mode == "deterministic_fallback"
        assert len(curated) <= 5
        # First should be highest score
        assert curated[0].symbol == "NVDA"

    def test_zero_budget_uses_fallback(self, finalists, mock_engine, default_config):
        """Zero budget should use fallback without attempting LLM."""
        curated, mode, curation_error = run_chief_scout_curation(
            finalists=finalists,
            remaining_budget=0.0,
            config=default_config,
            engine=mock_engine,
            max_shortlist=5,
        )

        assert mode == "deterministic_fallback"

    def test_negative_budget_uses_fallback(self, finalists, mock_engine, default_config):
        """Negative budget should use fallback."""
        curated, mode, curation_error = run_chief_scout_curation(
            finalists=finalists,
            remaining_budget=-5.0,
            config=default_config,
            engine=mock_engine,
            max_shortlist=5,
        )

        assert mode == "deterministic_fallback"

    def test_no_finalists_returns_empty_fallback(self, mock_engine, default_config):
        """Empty finalists list returns empty with fallback mode."""
        curated, mode, curation_error = run_chief_scout_curation(
            finalists=[],
            remaining_budget=60.0,
            config=default_config,
            engine=mock_engine,
            max_shortlist=5,
        )

        assert mode == "deterministic_fallback"
        assert curated == []


# ---------------------------------------------------------------------------
# Tests: run_chief_scout_curation — LLM success
# ---------------------------------------------------------------------------


class TestCurationLLMSuccess:
    """Tests for successful Chief Scout LLM curation."""

    @patch("utils.funnel_discovery._invoke_chief_scout_with_timeout")
    def test_chief_scout_success_uses_curated_picks(
        self, mock_invoke, finalists, mock_engine, default_config
    ):
        """When LLM succeeds, should use curated picks and set chief_scout mode."""
        mock_picks = [
            ChiefScoutPick(
                symbol="NVDA",
                sector="ai_semi",
                direction_bias="bullish",
                conviction="high",
                catalyst_summary="AI demand surge",
                reason="Strong momentum with catalyst",
                risk="Valuation stretched",
                source_candidate_score=85.0,
            ),
            ChiefScoutPick(
                symbol="TSLA",
                sector="ev",
                direction_bias="bullish",
                conviction="medium",
                catalyst_summary="Delivery numbers beat",
                reason="Positive momentum shift",
                risk="Competition pressure",
                source_candidate_score=68.0,
            ),
        ]
        mock_invoke.return_value = mock_picks

        curated, mode, curation_error = run_chief_scout_curation(
            finalists=finalists,
            remaining_budget=60.0,
            config=default_config,
            engine=mock_engine,
            max_shortlist=5,
        )

        assert mode == "chief_scout"
        assert len(curated) == 2
        assert curated[0]["symbol"] == "NVDA"
        assert curated[1]["symbol"] == "TSLA"

    @patch("utils.funnel_discovery._invoke_chief_scout_with_timeout")
    def test_chief_scout_max_shortlist_enforced(
        self, mock_invoke, finalists, mock_engine, default_config
    ):
        """When LLM returns more than max_shortlist, should truncate."""
        # Return 6 picks but max_shortlist is 3
        mock_picks = [
            ChiefScoutPick(
                symbol=f"SYM{i}",
                sector="tech",
                direction_bias="neutral",
                conviction="medium",
                catalyst_summary="test",
                reason="test",
                risk="test",
                source_candidate_score=float(90 - i * 5),
            )
            for i in range(6)
        ]
        mock_invoke.return_value = mock_picks

        curated, mode, curation_error = run_chief_scout_curation(
            finalists=finalists,
            remaining_budget=60.0,
            config=default_config,
            engine=mock_engine,
            max_shortlist=3,
        )

        assert mode == "chief_scout"
        assert len(curated) == 3


# ---------------------------------------------------------------------------
# Tests: run_chief_scout_curation — LLM failure/timeout
# ---------------------------------------------------------------------------


class TestCurationLLMFailure:
    """Tests for Chief Scout failure and timeout scenarios."""

    @patch("utils.funnel_discovery._invoke_chief_scout_with_timeout")
    def test_timeout_uses_fallback(
        self, mock_invoke, finalists, mock_engine, default_config
    ):
        """When Chief Scout times out, should use deterministic fallback."""
        mock_invoke.side_effect = TimeoutError("LLM timeout")

        curated, mode, curation_error = run_chief_scout_curation(
            finalists=finalists,
            remaining_budget=60.0,
            config=default_config,
            engine=mock_engine,
            max_shortlist=5,
        )

        assert mode == "deterministic_fallback"
        assert len(curated) <= 5
        assert curated[0].symbol == "NVDA"  # highest score

    @patch("utils.funnel_discovery._invoke_chief_scout_with_timeout")
    def test_exception_uses_fallback(
        self, mock_invoke, finalists, mock_engine, default_config
    ):
        """When Chief Scout raises any exception, should use deterministic fallback."""
        mock_invoke.side_effect = RuntimeError("LLM connection error")

        curated, mode, curation_error = run_chief_scout_curation(
            finalists=finalists,
            remaining_budget=60.0,
            config=default_config,
            engine=mock_engine,
            max_shortlist=5,
        )

        assert mode == "deterministic_fallback"
        assert len(curated) > 0

    @patch("utils.funnel_discovery._invoke_chief_scout_with_timeout")
    def test_empty_picks_uses_fallback(
        self, mock_invoke, finalists, mock_engine, default_config
    ):
        """When Chief Scout returns empty picks, should use deterministic fallback."""
        mock_invoke.return_value = None

        curated, mode, curation_error = run_chief_scout_curation(
            finalists=finalists,
            remaining_budget=60.0,
            config=default_config,
            engine=mock_engine,
            max_shortlist=5,
        )

        assert mode == "deterministic_fallback"

    @patch("utils.funnel_discovery._invoke_chief_scout_with_timeout")
    def test_empty_list_picks_uses_fallback(
        self, mock_invoke, finalists, mock_engine, default_config
    ):
        """When Chief Scout returns an empty list, should use deterministic fallback."""
        mock_invoke.return_value = []

        curated, mode, curation_error = run_chief_scout_curation(
            finalists=finalists,
            remaining_budget=60.0,
            config=default_config,
            engine=mock_engine,
            max_shortlist=5,
        )

        assert mode == "deterministic_fallback"

    @patch("utils.funnel_discovery._invoke_chief_scout_with_timeout")
    def test_fallback_preserves_finalists(
        self, mock_invoke, finalists, mock_engine, default_config
    ):
        """When fallback is used after failure, finalists are preserved (not empty)."""
        mock_invoke.side_effect = TimeoutError("timeout")

        curated, mode, curation_error = run_chief_scout_curation(
            finalists=finalists,
            remaining_budget=60.0,
            config=default_config,
            engine=mock_engine,
            max_shortlist=5,
        )

        assert mode == "deterministic_fallback"
        assert len(curated) == 5  # 6 finalists capped to max_shortlist=5
        # Verify ordering: score desc, symbol asc for ties
        assert curated[0].symbol == "NVDA"  # 85.0
        assert curated[1].symbol == "AMD"   # 72.0 (AMD < AVGO alphabetically)
        assert curated[2].symbol == "AVGO"  # 72.0
        assert curated[3].symbol == "TSLA"  # 68.0
        assert curated[4].symbol == "META"  # 60.0
