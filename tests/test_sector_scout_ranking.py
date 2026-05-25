"""Unit tests for rank_candidates() in the sector scout pipeline."""

from __future__ import annotations

import pytest

from utils.sector_scout import rank_candidates
from utils.sector_scout_models import CandidateRow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> dict:
    """Minimal config dict with budget_ceilings section."""
    return {
        "budget_ceilings": {
            "max_finalists_per_sector": 5,
        }
    }


def _make_row(
    symbol: str = "AAPL",
    sector: str = "tech",
    scout_score: float = 50.0,
    relative_volume: float | None = 2.0,
    dollar_volume: float | None = 10_000_000.0,
) -> CandidateRow:
    """Create a minimal CandidateRow for ranking tests."""
    return CandidateRow(
        symbol=symbol,
        sector=sector,
        sector_name="Technology",
        scout_score=scout_score,
        relative_volume=relative_volume,
        dollar_volume=dollar_volume,
    )


# ---------------------------------------------------------------------------
# Tests: Basic sorting
# ---------------------------------------------------------------------------


class TestRankCandidatesSorting:
    """Tests for stable tie-breaking sort order."""

    def test_sorts_by_scout_score_descending(self, default_config):
        candidates = {
            "tech": [
                _make_row(symbol="LOW", scout_score=30.0),
                _make_row(symbol="MID", scout_score=60.0),
                _make_row(symbol="HIGH", scout_score=90.0),
            ]
        }
        finalists, global_list = rank_candidates(candidates, default_config)
        assert [r.symbol for r in finalists["tech"]] == ["HIGH", "MID", "LOW"]
        assert [r.symbol for r in global_list] == ["HIGH", "MID", "LOW"]

    def test_tie_breaks_by_relative_volume_descending(self, default_config):
        candidates = {
            "tech": [
                _make_row(symbol="A", scout_score=50.0, relative_volume=1.0),
                _make_row(symbol="B", scout_score=50.0, relative_volume=3.0),
                _make_row(symbol="C", scout_score=50.0, relative_volume=2.0),
            ]
        }
        finalists, global_list = rank_candidates(candidates, default_config)
        assert [r.symbol for r in finalists["tech"]] == ["B", "C", "A"]

    def test_tie_breaks_by_dollar_volume_descending(self, default_config):
        candidates = {
            "tech": [
                _make_row(symbol="A", scout_score=50.0, relative_volume=2.0, dollar_volume=5_000_000.0),
                _make_row(symbol="B", scout_score=50.0, relative_volume=2.0, dollar_volume=20_000_000.0),
                _make_row(symbol="C", scout_score=50.0, relative_volume=2.0, dollar_volume=10_000_000.0),
            ]
        }
        finalists, global_list = rank_candidates(candidates, default_config)
        assert [r.symbol for r in finalists["tech"]] == ["B", "C", "A"]

    def test_tie_breaks_by_symbol_ascending(self, default_config):
        candidates = {
            "tech": [
                _make_row(symbol="ZZZ", scout_score=50.0, relative_volume=2.0, dollar_volume=10_000_000.0),
                _make_row(symbol="AAA", scout_score=50.0, relative_volume=2.0, dollar_volume=10_000_000.0),
                _make_row(symbol="MMM", scout_score=50.0, relative_volume=2.0, dollar_volume=10_000_000.0),
            ]
        }
        finalists, global_list = rank_candidates(candidates, default_config)
        assert [r.symbol for r in finalists["tech"]] == ["AAA", "MMM", "ZZZ"]

    def test_none_relative_volume_treated_as_zero(self, default_config):
        candidates = {
            "tech": [
                _make_row(symbol="A", scout_score=50.0, relative_volume=None),
                _make_row(symbol="B", scout_score=50.0, relative_volume=1.0),
            ]
        }
        finalists, _ = rank_candidates(candidates, default_config)
        # B has rvol=1.0 > 0.0 (None treated as 0), so B comes first
        assert [r.symbol for r in finalists["tech"]] == ["B", "A"]

    def test_none_dollar_volume_treated_as_zero(self, default_config):
        candidates = {
            "tech": [
                _make_row(symbol="A", scout_score=50.0, relative_volume=2.0, dollar_volume=None),
                _make_row(symbol="B", scout_score=50.0, relative_volume=2.0, dollar_volume=5_000_000.0),
            ]
        }
        finalists, _ = rank_candidates(candidates, default_config)
        # B has dollar_volume > 0 (None treated as 0), so B comes first
        assert [r.symbol for r in finalists["tech"]] == ["B", "A"]


# ---------------------------------------------------------------------------
# Tests: Finalist truncation (max_finalists_per_sector)
# ---------------------------------------------------------------------------


class TestRankCandidatesFinalistCeiling:
    """Tests for max_finalists_per_sector enforcement."""

    def test_truncates_to_max_finalists_per_sector(self, default_config):
        # Create 8 candidates, ceiling is 5
        candidates = {
            "tech": [
                _make_row(symbol=f"SYM{i}", scout_score=float(100 - i))
                for i in range(8)
            ]
        }
        finalists, global_list = rank_candidates(candidates, default_config)
        assert len(finalists["tech"]) == 5
        assert len(global_list) == 5

    def test_fewer_than_ceiling_keeps_all(self, default_config):
        candidates = {
            "tech": [
                _make_row(symbol="A", scout_score=80.0),
                _make_row(symbol="B", scout_score=70.0),
            ]
        }
        finalists, global_list = rank_candidates(candidates, default_config)
        assert len(finalists["tech"]) == 2
        assert len(global_list) == 2

    def test_custom_ceiling_from_config(self):
        config = {"budget_ceilings": {"max_finalists_per_sector": 2}}
        candidates = {
            "tech": [
                _make_row(symbol="A", scout_score=90.0),
                _make_row(symbol="B", scout_score=80.0),
                _make_row(symbol="C", scout_score=70.0),
            ]
        }
        finalists, global_list = rank_candidates(candidates, config)
        assert len(finalists["tech"]) == 2
        assert [r.symbol for r in finalists["tech"]] == ["A", "B"]

    def test_truncation_keeps_top_scoring(self):
        config = {"budget_ceilings": {"max_finalists_per_sector": 3}}
        candidates = {
            "tech": [
                _make_row(symbol="E", scout_score=50.0),
                _make_row(symbol="D", scout_score=60.0),
                _make_row(symbol="C", scout_score=70.0),
                _make_row(symbol="B", scout_score=80.0),
                _make_row(symbol="A", scout_score=90.0),
            ]
        }
        finalists, _ = rank_candidates(candidates, config)
        assert [r.symbol for r in finalists["tech"]] == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Tests: Multi-sector and global ranking
# ---------------------------------------------------------------------------


class TestRankCandidatesMultiSector:
    """Tests for multi-sector merging and global re-sort."""

    def test_global_list_merges_all_sectors(self, default_config):
        candidates = {
            "tech": [_make_row(symbol="NVDA", sector="tech", scout_score=85.0)],
            "energy": [_make_row(symbol="XOM", sector="energy", scout_score=75.0)],
            "finance": [_make_row(symbol="GS", sector="finance", scout_score=90.0)],
        }
        _, global_list = rank_candidates(candidates, default_config)
        assert len(global_list) == 3
        # Global should be sorted by score desc
        assert [r.symbol for r in global_list] == ["GS", "NVDA", "XOM"]

    def test_global_list_respects_tie_breaking_across_sectors(self, default_config):
        candidates = {
            "tech": [_make_row(symbol="NVDA", sector="tech", scout_score=80.0, relative_volume=3.0)],
            "energy": [_make_row(symbol="XOM", sector="energy", scout_score=80.0, relative_volume=2.0)],
        }
        _, global_list = rank_candidates(candidates, default_config)
        # Same score, NVDA has higher rvol
        assert [r.symbol for r in global_list] == ["NVDA", "XOM"]

    def test_empty_sector_produces_empty_finalists(self, default_config):
        candidates = {
            "tech": [],
            "energy": [_make_row(symbol="XOM", sector="energy", scout_score=75.0)],
        }
        finalists, global_list = rank_candidates(candidates, default_config)
        assert finalists["tech"] == []
        assert len(global_list) == 1

    def test_empty_input_returns_empty(self, default_config):
        finalists, global_list = rank_candidates({}, default_config)
        assert finalists == {}
        assert global_list == []

    def test_per_sector_ceiling_applied_independently(self):
        config = {"budget_ceilings": {"max_finalists_per_sector": 2}}
        candidates = {
            "tech": [
                _make_row(symbol="A", sector="tech", scout_score=90.0),
                _make_row(symbol="B", sector="tech", scout_score=80.0),
                _make_row(symbol="C", sector="tech", scout_score=70.0),
            ],
            "energy": [
                _make_row(symbol="X", sector="energy", scout_score=85.0),
                _make_row(symbol="Y", sector="energy", scout_score=65.0),
                _make_row(symbol="Z", sector="energy", scout_score=55.0),
            ],
        }
        finalists, global_list = rank_candidates(candidates, config)
        assert len(finalists["tech"]) == 2
        assert len(finalists["energy"]) == 2
        # Global has 4 total (2 per sector), sorted globally
        assert len(global_list) == 4
        assert [r.symbol for r in global_list] == ["A", "X", "B", "Y"]
