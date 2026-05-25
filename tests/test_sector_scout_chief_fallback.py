"""Unit tests for chief_scout_fallback() deterministic fallback logic.

Tests cover:
- Basic fallback returns top N by scout_score
- Respects max_expanded_watchlist ceiling
- Returns empty picks when watchlist is full
- Logs "no expanded candidates" when no finalists exist
- Stable tie-breaking matches rank_candidates sort key
- Converts candidates to ChiefScoutPick format correctly
- fallback_used is always True
"""

from __future__ import annotations

import pytest

from utils.sector_scout_chief import chief_scout_fallback
from utils.sector_scout_models import CandidateRow, ChiefScoutPick


def _make_candidate(
    symbol: str,
    sector: str = "ai_semi",
    scout_score: float = 50.0,
    relative_volume: float | None = 2.0,
    dollar_volume: float | None = 10_000_000.0,
) -> CandidateRow:
    """Helper to create a minimal CandidateRow for testing."""
    return CandidateRow(
        symbol=symbol,
        sector=sector,
        sector_name="Test Sector",
        scout_score=scout_score,
        relative_volume=relative_volume,
        dollar_volume=dollar_volume,
        hard_gate_passed=True,
    )


def _make_config(
    fallback_limit: int = 3,
    max_expanded_watchlist: int = 12,
) -> dict:
    """Helper to create a minimal config dict for testing."""
    return {
        "chief_scout": {
            "max_picks": 8,
            "fallback_limit": fallback_limit,
        },
        "budget_ceilings": {
            "max_expanded_watchlist": max_expanded_watchlist,
        },
    }


class TestChiefScoutFallbackBasic:
    """Basic fallback behavior tests."""

    def test_returns_top_n_by_scout_score(self):
        """Fallback returns top N candidates sorted by scout_score descending."""
        finalists = {
            "ai_semi": [
                _make_candidate("AVGO", scout_score=80.0),
                _make_candidate("SMCI", scout_score=60.0),
                _make_candidate("ARM", scout_score=70.0),
                _make_candidate("MU", scout_score=50.0),
            ]
        }
        config = _make_config(fallback_limit=3)

        result = chief_scout_fallback(finalists, config, current_watchlist_size=0)

        assert result["fallback_used"] is True
        assert len(result["picks"]) == 3
        # Should be sorted by score descending
        symbols = [p["symbol"] for p in result["picks"]]
        assert symbols == ["AVGO", "ARM", "SMCI"]

    def test_fallback_used_always_true(self):
        """fallback_used flag is always True in fallback results."""
        finalists = {"ai_semi": [_make_candidate("AVGO", scout_score=80.0)]}
        config = _make_config(fallback_limit=3)

        result = chief_scout_fallback(finalists, config, current_watchlist_size=0)

        assert result["fallback_used"] is True

    def test_returns_fewer_than_limit_when_fewer_finalists(self):
        """When fewer finalists than fallback_limit, returns all available."""
        finalists = {
            "ai_semi": [
                _make_candidate("AVGO", scout_score=80.0),
                _make_candidate("SMCI", scout_score=60.0),
            ]
        }
        config = _make_config(fallback_limit=5)

        result = chief_scout_fallback(finalists, config, current_watchlist_size=0)

        assert len(result["picks"]) == 2

    def test_collects_from_multiple_sectors(self):
        """Fallback collects finalists from all sectors and ranks globally."""
        finalists = {
            "ai_semi": [
                _make_candidate("AVGO", sector="ai_semi", scout_score=80.0),
                _make_candidate("SMCI", sector="ai_semi", scout_score=50.0),
            ],
            "financials": [
                _make_candidate("GS", sector="financials", scout_score=75.0),
                _make_candidate("MS", sector="financials", scout_score=55.0),
            ],
        }
        config = _make_config(fallback_limit=3)

        result = chief_scout_fallback(finalists, config, current_watchlist_size=0)

        symbols = [p["symbol"] for p in result["picks"]]
        assert symbols == ["AVGO", "GS", "MS"]


class TestChiefScoutFallbackWatchlistCeiling:
    """Tests for max_expanded_watchlist ceiling enforcement."""

    def test_respects_watchlist_ceiling(self):
        """Fallback adds at most min(fallback_limit, max - current) symbols."""
        finalists = {
            "ai_semi": [
                _make_candidate("AVGO", scout_score=80.0),
                _make_candidate("SMCI", scout_score=70.0),
                _make_candidate("ARM", scout_score=60.0),
            ]
        }
        # max=12, current=10 → only 2 slots available
        config = _make_config(fallback_limit=3, max_expanded_watchlist=12)

        result = chief_scout_fallback(finalists, config, current_watchlist_size=10)

        assert len(result["picks"]) == 2
        symbols = [p["symbol"] for p in result["picks"]]
        assert symbols == ["AVGO", "SMCI"]

    def test_returns_empty_when_watchlist_full(self):
        """When watchlist is at capacity, fallback returns empty picks."""
        finalists = {
            "ai_semi": [
                _make_candidate("AVGO", scout_score=80.0),
            ]
        }
        config = _make_config(fallback_limit=3, max_expanded_watchlist=12)

        result = chief_scout_fallback(finalists, config, current_watchlist_size=12)

        assert result["picks"] == []
        assert result["fallback_used"] is True

    def test_returns_empty_when_watchlist_over_capacity(self):
        """When current_watchlist_size > max, fallback returns empty picks."""
        finalists = {
            "ai_semi": [
                _make_candidate("AVGO", scout_score=80.0),
            ]
        }
        config = _make_config(fallback_limit=3, max_expanded_watchlist=12)

        result = chief_scout_fallback(finalists, config, current_watchlist_size=15)

        assert result["picks"] == []
        assert result["fallback_used"] is True


class TestChiefScoutFallbackEmptyInput:
    """Tests for empty/no finalists scenarios."""

    def test_returns_empty_when_no_finalists(self):
        """When no finalists exist, returns empty picks."""
        finalists: dict = {}
        config = _make_config(fallback_limit=3)

        result = chief_scout_fallback(finalists, config, current_watchlist_size=0)

        assert result["picks"] == []
        assert result["fallback_used"] is True

    def test_returns_empty_when_all_sectors_empty(self):
        """When all sector lists are empty, returns empty picks."""
        finalists = {"ai_semi": [], "financials": []}
        config = _make_config(fallback_limit=3)

        result = chief_scout_fallback(finalists, config, current_watchlist_size=0)

        assert result["picks"] == []
        assert result["fallback_used"] is True


class TestChiefScoutFallbackPickFormat:
    """Tests for ChiefScoutPick format conversion."""

    def test_pick_has_correct_format(self):
        """Each pick has all required ChiefScoutPick fields."""
        finalists = {
            "ai_semi": [
                _make_candidate("AVGO", sector="ai_semi", scout_score=80.0),
            ]
        }
        config = _make_config(fallback_limit=3)

        result = chief_scout_fallback(finalists, config, current_watchlist_size=0)

        pick = result["picks"][0]
        assert pick["symbol"] == "AVGO"
        assert pick["sector"] == "ai_semi"
        assert pick["direction_bias"] == "neutral"
        assert pick["conviction"] == "low"
        assert pick["catalyst_summary"] == "Deterministic fallback - top by scout score"
        assert pick["source_candidate_score"] == 80.0
        assert "reason" in pick
        assert "risk" in pick

    def test_pick_source_candidate_score_matches_scout_score(self):
        """source_candidate_score reflects the candidate's scout_score."""
        finalists = {
            "ai_semi": [
                _make_candidate("AVGO", scout_score=72.5),
            ]
        }
        config = _make_config(fallback_limit=3)

        result = chief_scout_fallback(finalists, config, current_watchlist_size=0)

        assert result["picks"][0]["source_candidate_score"] == 72.5


class TestChiefScoutFallbackTieBreaking:
    """Tests for stable tie-breaking in fallback sorting."""

    def test_tie_breaking_by_relative_volume(self):
        """When scores are equal, higher relative_volume wins."""
        finalists = {
            "ai_semi": [
                _make_candidate("SMCI", scout_score=70.0, relative_volume=3.0),
                _make_candidate("AVGO", scout_score=70.0, relative_volume=5.0),
            ]
        }
        config = _make_config(fallback_limit=2)

        result = chief_scout_fallback(finalists, config, current_watchlist_size=0)

        symbols = [p["symbol"] for p in result["picks"]]
        assert symbols == ["AVGO", "SMCI"]

    def test_tie_breaking_by_dollar_volume(self):
        """When score and rvol are equal, higher dollar_volume wins."""
        finalists = {
            "ai_semi": [
                _make_candidate("SMCI", scout_score=70.0, relative_volume=3.0, dollar_volume=5_000_000.0),
                _make_candidate("AVGO", scout_score=70.0, relative_volume=3.0, dollar_volume=10_000_000.0),
            ]
        }
        config = _make_config(fallback_limit=2)

        result = chief_scout_fallback(finalists, config, current_watchlist_size=0)

        symbols = [p["symbol"] for p in result["picks"]]
        assert symbols == ["AVGO", "SMCI"]

    def test_tie_breaking_by_symbol_ascending(self):
        """When all numeric fields are equal, symbol ascending breaks tie."""
        finalists = {
            "ai_semi": [
                _make_candidate("SMCI", scout_score=70.0, relative_volume=3.0, dollar_volume=10_000_000.0),
                _make_candidate("AVGO", scout_score=70.0, relative_volume=3.0, dollar_volume=10_000_000.0),
            ]
        }
        config = _make_config(fallback_limit=2)

        result = chief_scout_fallback(finalists, config, current_watchlist_size=0)

        symbols = [p["symbol"] for p in result["picks"]]
        assert symbols == ["AVGO", "SMCI"]

    def test_none_relative_volume_treated_as_zero(self):
        """None relative_volume is treated as 0 for tie-breaking."""
        finalists = {
            "ai_semi": [
                _make_candidate("SMCI", scout_score=70.0, relative_volume=None),
                _make_candidate("AVGO", scout_score=70.0, relative_volume=1.0),
            ]
        }
        config = _make_config(fallback_limit=2)

        result = chief_scout_fallback(finalists, config, current_watchlist_size=0)

        symbols = [p["symbol"] for p in result["picks"]]
        assert symbols == ["AVGO", "SMCI"]
