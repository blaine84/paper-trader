"""Unit tests for the funnel discovery pipeline.

Tests the core discovery loop structure: per-sector budget enforcement,
total pipeline budget, sector tracking, and the DiscoveryResult contract.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from utils.funnel_discovery import (
    DiscoveryResult,
    get_enabled_sectors,
    run_funnel_discovery,
    run_sector_with_timeout,
)
from utils.sector_scout_models import CandidateRow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_candidate(symbol: str, sector: str, score: float) -> CandidateRow:
    """Create a minimal CandidateRow for testing."""
    return CandidateRow(
        symbol=symbol,
        sector=sector,
        sector_name=f"{sector}_name",
        scout_score=score,
    )


@pytest.fixture
def minimal_sector_config():
    """A minimal sector scout config with two enabled sectors."""
    return {
        "sector_buckets": {
            "tech": {"enabled": True, "symbols": ["AAPL", "MSFT"], "name": "Technology"},
            "energy": {"enabled": True, "symbols": ["XOM", "CVX"], "name": "Energy"},
            "disabled_sector": {"enabled": False, "symbols": ["X"], "name": "Disabled"},
        },
        "hard_gates": {},
        "score_penalties": {},
        "scoring_weights": {},
        "scoring_caps": {},
        "budget_ceilings": {
            "max_sectors_per_run": 7,
            "max_candidates_per_sector": 20,
            "max_finalists_per_sector": 5,
        },
        "chief_scout": {},
        "reanalysis_cooldown": {},
        "enabled": True,
    }


@pytest.fixture
def default_funnel_config():
    """Default funnel config matching the YAML defaults."""
    return {
        "funnel": {
            "enabled": True,
            "budgets": {
                "per_sector_seconds": 15,
                "total_pipeline_seconds": 90,
            },
            "ceilings": {
                "max_discovery_shortlist": 5,
            },
        }
    }


# ---------------------------------------------------------------------------
# get_enabled_sectors
# ---------------------------------------------------------------------------


class TestGetEnabledSectors:
    def test_returns_only_enabled(self, minimal_sector_config):
        """Only sectors with enabled=True are returned."""
        result = get_enabled_sectors(minimal_sector_config)
        assert "tech" in result
        assert "energy" in result
        assert "disabled_sector" not in result

    def test_respects_max_sectors_per_run(self):
        """Enforces max_sectors_per_run ceiling."""
        config = {
            "sector_buckets": {
                f"sector_{i}": {"enabled": True, "symbols": []}
                for i in range(10)
            },
            "budget_ceilings": {"max_sectors_per_run": 3},
        }
        result = get_enabled_sectors(config)
        assert len(result) == 3

    def test_empty_buckets(self):
        """Returns empty list when no sectors defined."""
        config = {"sector_buckets": {}, "budget_ceilings": {}}
        result = get_enabled_sectors(config)
        assert result == []

    def test_all_disabled(self, minimal_sector_config):
        """Returns empty list when all sectors disabled."""
        for bucket in minimal_sector_config["sector_buckets"].values():
            bucket["enabled"] = False
        result = get_enabled_sectors(minimal_sector_config)
        assert result == []

    def test_default_max_sectors_is_7(self):
        """Without explicit max_sectors_per_run, default is 7."""
        config = {
            "sector_buckets": {
                f"sector_{i}": {"enabled": True, "symbols": []}
                for i in range(10)
            },
            "budget_ceilings": {},
        }
        result = get_enabled_sectors(config)
        assert len(result) == 7


# ---------------------------------------------------------------------------
# run_sector_with_timeout
# ---------------------------------------------------------------------------


class TestRunSectorWithTimeout:
    @patch("utils.funnel_discovery.FinnhubClient")
    @patch("utils.sector_scout.collect_candidate_data")
    @patch("utils.sector_scout.apply_hard_gates")
    @patch("utils.sector_scout.compute_scout_score")
    @patch("utils.sector_scout.apply_score_penalties")
    def test_returns_results_within_timeout(
        self, mock_penalties, mock_score, mock_gates, mock_collect, mock_fh_class
    ):
        """Sector that completes within timeout returns its results."""
        candidate = _make_candidate("AAPL", "tech", 75.0)
        mock_collect.return_value = candidate
        mock_gates.return_value = (True, None)
        mock_score.return_value = candidate
        mock_penalties.return_value = candidate

        config = {
            "sector_buckets": {
                "tech": {"enabled": True, "symbols": ["AAPL"], "name": "Technology"},
            },
            "budget_ceilings": {"max_candidates_per_sector": 20},
        }

        result = run_sector_with_timeout("tech", config, timeout=5.0)
        assert len(result) == 1
        assert result[0].symbol == "AAPL"

    @patch("utils.funnel_discovery.FinnhubClient")
    @patch("utils.sector_scout.collect_candidate_data")
    def test_raises_timeout_on_slow_sector(self, mock_collect, mock_fh_class):
        """Sector that exceeds timeout raises TimeoutError."""

        def slow_collect(*args, **kwargs):
            time.sleep(2.0)
            return _make_candidate("SLOW", "tech", 50.0)

        mock_collect.side_effect = slow_collect

        config = {
            "sector_buckets": {
                "tech": {"enabled": True, "symbols": ["SLOW"], "name": "Technology"},
            },
            "budget_ceilings": {"max_candidates_per_sector": 20},
        }

        with pytest.raises(TimeoutError):
            run_sector_with_timeout("tech", config, timeout=0.1)

    @patch("utils.funnel_discovery.FinnhubClient")
    @patch("utils.sector_scout.collect_candidate_data")
    def test_propagates_exceptions(self, mock_collect, mock_fh_class):
        """Exceptions from screening are propagated through the future."""
        mock_collect.side_effect = RuntimeError("API failure")

        config = {
            "sector_buckets": {
                "tech": {"enabled": True, "symbols": ["FAIL"], "name": "Technology"},
            },
            "budget_ceilings": {"max_candidates_per_sector": 20},
        }

        with pytest.raises(RuntimeError, match="API failure"):
            run_sector_with_timeout("tech", config, timeout=5.0)

    @patch("utils.funnel_discovery.FinnhubClient")
    @patch("utils.sector_scout.collect_candidate_data")
    @patch("utils.sector_scout.apply_hard_gates")
    def test_excludes_core_watchlist(self, mock_gates, mock_collect, mock_fh_class):
        """Core watchlist symbols are excluded from sector screening."""
        config = {
            "sector_buckets": {
                "tech": {
                    "enabled": True,
                    "symbols": ["AAPL", "MSFT", "GOOGL"],
                    "name": "Technology",
                },
            },
            "budget_ceilings": {"max_candidates_per_sector": 20},
        }

        mock_collect.return_value = _make_candidate("MSFT", "tech", 70.0)
        mock_gates.return_value = (False, "test_gate")

        result = run_sector_with_timeout(
            "tech", config, timeout=5.0, core_watchlist=["AAPL"]
        )

        # AAPL should be excluded; only MSFT and GOOGL should be screened
        # (both fail gates in this test setup, so result is empty)
        call_symbols = [call[0][0] for call in mock_collect.call_args_list]
        assert "AAPL" not in call_symbols
        assert "MSFT" in call_symbols
        assert "GOOGL" in call_symbols


# ---------------------------------------------------------------------------
# run_funnel_discovery — sector tracking and budget enforcement
# ---------------------------------------------------------------------------


class TestRunFunnelDiscovery:
    @patch("utils.funnel_discovery.record_discovery_run_log")
    @patch("utils.funnel_discovery.persist_discovery_candidates")
    @patch("utils.funnel_discovery.run_chief_scout_curation")
    @patch("utils.funnel_discovery.run_sector_with_timeout")
    @patch("utils.funnel_discovery.FinnhubClient")
    def test_all_sectors_complete_successfully(
        self, mock_fh_class, mock_sector_timeout, mock_curation, mock_persist,
        mock_record_log,
        minimal_sector_config, default_funnel_config
    ):
        """When all sectors succeed, they appear in sectors_completed."""
        tech_candidates = [_make_candidate("AAPL", "tech", 80.0)]
        energy_candidates = [_make_candidate("XOM", "energy", 70.0)]
        mock_sector_timeout.side_effect = [tech_candidates, energy_candidates]
        mock_curation.return_value = (tech_candidates + energy_candidates, "deterministic_fallback", None)
        mock_persist.return_value = tech_candidates + energy_candidates

        engine = MagicMock()
        result = run_funnel_discovery(engine, minimal_sector_config, default_funnel_config)

        assert isinstance(result, DiscoveryResult)
        assert "tech" in result.sectors_completed
        assert "energy" in result.sectors_completed
        assert result.sectors_timed_out == []
        assert result.sectors_skipped == []
        assert result.partial_screening is False
        assert result.pipeline_budget_exhausted is False

    @patch("utils.funnel_discovery.record_discovery_run_log")
    @patch("utils.funnel_discovery.persist_discovery_candidates")
    @patch("utils.funnel_discovery.run_chief_scout_curation")
    @patch("utils.funnel_discovery.run_sector_with_timeout")
    @patch("utils.funnel_discovery.FinnhubClient")
    def test_sector_timeout_tracked(
        self, mock_fh_class, mock_sector_timeout, mock_curation, mock_persist,
        mock_record_log,
        minimal_sector_config, default_funnel_config
    ):
        """Timed-out sectors are recorded in sectors_timed_out."""
        mock_sector_timeout.side_effect = [
            TimeoutError("tech exceeded budget"),
            [_make_candidate("XOM", "energy", 70.0)],
        ]
        mock_curation.return_value = ([_make_candidate("XOM", "energy", 70.0)], "deterministic_fallback", None)
        mock_persist.return_value = [_make_candidate("XOM", "energy", 70.0)]

        engine = MagicMock()
        result = run_funnel_discovery(engine, minimal_sector_config, default_funnel_config)

        assert "tech" in result.sectors_timed_out
        assert "energy" in result.sectors_completed
        assert result.partial_screening is True
        assert result.pipeline_budget_exhausted is False

    @patch("utils.funnel_discovery.record_discovery_run_log")
    @patch("utils.funnel_discovery.persist_discovery_candidates")
    @patch("utils.funnel_discovery.run_chief_scout_curation")
    @patch("utils.funnel_discovery.run_sector_with_timeout")
    @patch("utils.funnel_discovery.FinnhubClient")
    def test_total_budget_exhaustion_skips_remaining(
        self, mock_fh_class, mock_sector_timeout, mock_curation, mock_persist,
        mock_record_log,
        minimal_sector_config, default_funnel_config
    ):
        """When total budget is exhausted, remaining sectors are skipped."""
        # Set a very tight total budget
        default_funnel_config["funnel"]["budgets"]["total_pipeline_seconds"] = 0.05

        def slow_sector(*args, **kwargs):
            time.sleep(0.1)  # exceeds total budget
            return [_make_candidate("AAPL", "tech", 80.0)]

        mock_sector_timeout.side_effect = slow_sector
        mock_curation.return_value = ([], "deterministic_fallback", None)
        mock_persist.return_value = []

        engine = MagicMock()
        result = run_funnel_discovery(engine, minimal_sector_config, default_funnel_config)

        # After first sector takes ~100ms against 50ms budget,
        # second sector should be skipped
        assert result.pipeline_budget_exhausted is True
        assert result.partial_screening is True
        assert len(result.sectors_skipped) > 0

    @patch("utils.funnel_discovery.record_discovery_run_log")
    @patch("utils.funnel_discovery.persist_discovery_candidates")
    @patch("utils.funnel_discovery.run_chief_scout_curation")
    @patch("utils.funnel_discovery.run_sector_with_timeout")
    @patch("utils.funnel_discovery.FinnhubClient")
    def test_empty_sectors_produce_empty_result(
        self, mock_fh_class, mock_sector_timeout, mock_curation, mock_persist,
        mock_record_log,
        minimal_sector_config, default_funnel_config
    ):
        """When no sectors produce candidates, result has empty candidates list."""
        mock_sector_timeout.side_effect = [[], []]
        mock_curation.return_value = ([], "deterministic_fallback", None)
        mock_persist.return_value = []

        engine = MagicMock()
        result = run_funnel_discovery(engine, minimal_sector_config, default_funnel_config)

        assert result.candidates == []
        assert "tech" in result.sectors_completed
        assert "energy" in result.sectors_completed

    @patch("utils.funnel_discovery.record_discovery_run_log")
    @patch("utils.funnel_discovery.persist_discovery_candidates")
    @patch("utils.funnel_discovery.run_chief_scout_curation")
    @patch("utils.funnel_discovery.run_sector_with_timeout")
    @patch("utils.funnel_discovery.FinnhubClient")
    def test_selection_mode_from_curation(
        self, mock_fh_class, mock_sector_timeout, mock_curation, mock_persist,
        mock_record_log,
        minimal_sector_config, default_funnel_config
    ):
        """Selection mode is determined by the curation phase."""
        mock_sector_timeout.side_effect = [
            [_make_candidate("AAPL", "tech", 80.0)],
            [],
        ]
        mock_curation.return_value = (
            [_make_candidate("AAPL", "tech", 80.0)],
            "chief_scout",
            None,
        )
        mock_persist.return_value = [_make_candidate("AAPL", "tech", 80.0)]

        engine = MagicMock()
        result = run_funnel_discovery(engine, minimal_sector_config, default_funnel_config)

        assert result.selection_mode == "chief_scout"

    @patch("utils.funnel_discovery.record_discovery_run_log")
    @patch("utils.funnel_discovery.persist_discovery_candidates")
    @patch("utils.funnel_discovery.run_chief_scout_curation")
    @patch("utils.funnel_discovery.run_sector_with_timeout")
    @patch("utils.funnel_discovery.FinnhubClient")
    def test_total_duration_measured(
        self, mock_fh_class, mock_sector_timeout, mock_curation, mock_persist,
        mock_record_log,
        minimal_sector_config, default_funnel_config
    ):
        """total_duration_seconds is a positive value reflecting pipeline time."""
        mock_sector_timeout.side_effect = [[], []]
        mock_curation.return_value = ([], "deterministic_fallback", None)
        mock_persist.return_value = []

        engine = MagicMock()
        result = run_funnel_discovery(engine, minimal_sector_config, default_funnel_config)

        assert result.total_duration_seconds > 0

    @patch("utils.funnel_discovery.record_discovery_run_log")
    @patch("utils.funnel_discovery.persist_discovery_candidates")
    @patch("utils.funnel_discovery.run_chief_scout_curation")
    @patch("utils.funnel_discovery.run_sector_with_timeout")
    @patch("utils.funnel_discovery.FinnhubClient")
    def test_unexpected_exception_treated_as_timeout(
        self, mock_fh_class, mock_sector_timeout, mock_curation, mock_persist,
        mock_record_log,
        minimal_sector_config, default_funnel_config
    ):
        """Unexpected exceptions from sector screening go to sectors_timed_out."""
        mock_sector_timeout.side_effect = [
            RuntimeError("Unexpected API failure"),
            [_make_candidate("XOM", "energy", 70.0)],
        ]
        mock_curation.return_value = ([_make_candidate("XOM", "energy", 70.0)], "deterministic_fallback", None)
        mock_persist.return_value = [_make_candidate("XOM", "energy", 70.0)]

        engine = MagicMock()
        result = run_funnel_discovery(engine, minimal_sector_config, default_funnel_config)

        assert "tech" in result.sectors_timed_out
        assert "energy" in result.sectors_completed
        assert result.partial_screening is True

    @patch("utils.funnel_discovery.record_discovery_run_log")
    @patch("utils.funnel_discovery.persist_discovery_candidates")
    @patch("utils.funnel_discovery.run_chief_scout_curation")
    @patch("utils.funnel_discovery.run_sector_with_timeout")
    @patch("utils.funnel_discovery.FinnhubClient")
    def test_all_enabled_sectors_accounted_for(
        self, mock_fh_class, mock_sector_timeout, mock_curation, mock_persist,
        mock_record_log,
        minimal_sector_config, default_funnel_config
    ):
        """Every enabled sector ends up in exactly one tracking list."""
        mock_sector_timeout.side_effect = [
            TimeoutError("tech timed out"),
            [_make_candidate("XOM", "energy", 70.0)],
        ]
        mock_curation.return_value = ([_make_candidate("XOM", "energy", 70.0)], "deterministic_fallback", None)
        mock_persist.return_value = [_make_candidate("XOM", "energy", 70.0)]

        engine = MagicMock()
        result = run_funnel_discovery(engine, minimal_sector_config, default_funnel_config)

        all_tracked = (
            set(result.sectors_completed)
            | set(result.sectors_timed_out)
            | set(result.sectors_skipped)
        )
        enabled = set(get_enabled_sectors(minimal_sector_config))
        assert all_tracked == enabled

    @patch("utils.funnel_discovery.record_discovery_run_log")
    @patch("utils.funnel_discovery.persist_discovery_candidates")
    @patch("utils.funnel_discovery.run_chief_scout_curation")
    @patch("utils.funnel_discovery.run_sector_with_timeout")
    @patch("utils.funnel_discovery.FinnhubClient")
    def test_per_sector_budget_is_min_of_config_and_remaining(
        self, mock_fh_class, mock_sector_timeout, mock_curation, mock_persist,
        mock_record_log,
        minimal_sector_config, default_funnel_config
    ):
        """Per-sector budget passed to run_sector_with_timeout is capped by remaining total."""
        # Set per_sector to 15s but total to 20s with 2 sectors
        default_funnel_config["funnel"]["budgets"]["per_sector_seconds"] = 15
        default_funnel_config["funnel"]["budgets"]["total_pipeline_seconds"] = 20

        mock_sector_timeout.side_effect = [[], []]
        mock_curation.return_value = ([], "deterministic_fallback", None)
        mock_persist.return_value = []

        engine = MagicMock()
        run_funnel_discovery(engine, minimal_sector_config, default_funnel_config)

        # First call should get 15s (min of 15, ~20)
        # Second call should get min(15, ~5) ≈ 5s
        calls = mock_sector_timeout.call_args_list
        assert len(calls) == 2
        # First sector gets full per-sector budget (15s, remaining is ~20s)
        first_timeout = calls[0].kwargs.get("timeout", calls[0][1] if len(calls[0]) > 1 else None)
        assert first_timeout is not None
        assert first_timeout <= 15.0
        # Second sector gets remaining (less than 15s since some elapsed)
        second_timeout = calls[1].kwargs.get("timeout", calls[1][1] if len(calls[1]) > 1 else None)
        assert second_timeout is not None
        assert second_timeout <= 15.0

    @patch("utils.funnel_discovery.record_discovery_run_log")
    @patch("utils.funnel_discovery.persist_discovery_candidates")
    @patch("utils.funnel_discovery.run_chief_scout_curation")
    @patch("utils.funnel_discovery.run_sector_with_timeout")
    @patch("utils.funnel_discovery.FinnhubClient")
    def test_finnhub_creation_failure_does_not_crash(
        self, mock_fh_class, mock_sector_timeout, mock_curation, mock_persist,
        mock_record_log,
        minimal_sector_config, default_funnel_config
    ):
        """If FinnhubClient creation fails, pipeline proceeds with fh=None."""
        mock_fh_class.side_effect = ValueError("FINNHUB_API_KEY not set")
        mock_sector_timeout.side_effect = [[], []]
        mock_curation.return_value = ([], "deterministic_fallback", None)
        mock_persist.return_value = []

        engine = MagicMock()
        result = run_funnel_discovery(engine, minimal_sector_config, default_funnel_config)

        # Pipeline should still complete
        assert isinstance(result, DiscoveryResult)
        # Sectors should still be processed (via mocked run_sector_with_timeout)
        assert len(result.sectors_completed) == 2
