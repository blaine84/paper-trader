"""Tests for utils/funnel_transition.py — expanded_watchlist → FunnelCandidate migration.

Tests the transition logic (Req 13.1–13.4, 13.7):
- Uses get_pm_eligible_candidates() as primary source when funnel ran today
- Falls back to get_expanded_watchlist() when no valid FunnelRunLog exists
- Does NOT fall back when funnel ran successfully but returned empty
- Deduplicates symbols from multiple sources
- Applies max_pm_handoff ceiling to legacy fallback
"""

import json
import uuid
from datetime import datetime, date as date_type, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import create_engine

from db.schema import Base, FunnelCandidate, FunnelRunLog, get_session
from utils.funnel_transition import (
    _has_valid_discovery_run_today,
    build_deduplicated_watchlist,
    get_funnel_or_fallback_candidates,
)

_NY_TZ = ZoneInfo("America/New_York")


@pytest.fixture
def engine():
    """Create an in-memory SQLite engine with all tables."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _today_ny() -> date_type:
    return datetime.now(_NY_TZ).date()


def _create_funnel_run_log(engine, stage="discovery", result_status="completed"):
    """Helper to create a FunnelRunLog entry for today."""
    session = get_session(engine)
    try:
        log = FunnelRunLog(
            date=_today_ny(),
            stage=stage,
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
            duration_seconds=45.0,
            budget_seconds=90.0,
            result_status=result_status,
            candidates_input=10,
            candidates_promoted=3,
            candidates_rejected=7,
        )
        session.add(log)
        session.commit()
    finally:
        session.close()


def _create_pm_eligible_candidate(engine, symbol: str, scout_rank: int = 1):
    """Helper to create a pm_eligible FunnelCandidate for today."""
    session = get_session(engine)
    try:
        candidate = FunnelCandidate(
            candidate_id=str(uuid.uuid4()),
            date=_today_ny(),
            symbol=symbol,
            discovered_at=datetime.now(timezone.utc),
            source_run="premarket",
            selection_mode="deterministic_fallback",
            scout_rank=scout_rank,
            scout_score=75.0,
            catalyst_evidence=json.dumps({"catalyst": "test"}),
            selection_reason="test reason",
            primary_risk="test risk",
            stage_status="pm_eligible",
            stage_decisions=json.dumps([
                {
                    "agent": "scout",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "decision": "promoted",
                    "reasoning": "test",
                    "evidence": {},
                    "next_stage": "awaiting_research",
                },
                {
                    "agent": "confirmation",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "decision": "promoted",
                    "reasoning": "confirmed",
                    "evidence": {},
                    "next_stage": "pm_eligible",
                },
            ]),
            expired=False,
        )
        session.add(candidate)
        session.commit()
    finally:
        session.close()


class TestHasValidDiscoveryRunToday:
    """Tests for _has_valid_discovery_run_today()."""

    def test_returns_false_when_no_logs_exist(self, engine):
        """No FunnelRunLog → False."""
        assert _has_valid_discovery_run_today(engine) is False

    def test_returns_true_for_completed_discovery(self, engine):
        """FunnelRunLog with result_status='completed' → True."""
        _create_funnel_run_log(engine, stage="discovery", result_status="completed")
        assert _has_valid_discovery_run_today(engine) is True

    def test_returns_true_for_degraded_discovery(self, engine):
        """FunnelRunLog with result_status='degraded' → True."""
        _create_funnel_run_log(engine, stage="discovery", result_status="degraded")
        assert _has_valid_discovery_run_today(engine) is True

    def test_returns_false_for_error_discovery(self, engine):
        """FunnelRunLog with result_status='error' does NOT count (Req 13.2 note)."""
        _create_funnel_run_log(engine, stage="discovery", result_status="error")
        assert _has_valid_discovery_run_today(engine) is False

    def test_returns_false_for_timed_out_discovery(self, engine):
        """FunnelRunLog with result_status='timed_out' does NOT count."""
        _create_funnel_run_log(engine, stage="discovery", result_status="timed_out")
        assert _has_valid_discovery_run_today(engine) is False

    def test_ignores_non_discovery_stages(self, engine):
        """A research or analysis FunnelRunLog doesn't satisfy discovery check."""
        _create_funnel_run_log(engine, stage="research", result_status="completed")
        assert _has_valid_discovery_run_today(engine) is False


class TestGetFunnelOrFallbackCandidates:
    """Tests for get_funnel_or_fallback_candidates()."""

    def test_funnel_primary_source_when_discovery_ran(self, engine):
        """When discovery ran and pm_eligible candidates exist, use funnel."""
        _create_funnel_run_log(engine, stage="discovery", result_status="completed")
        _create_pm_eligible_candidate(engine, "AAPL", scout_rank=1)
        _create_pm_eligible_candidate(engine, "TSLA", scout_rank=2)

        symbols, source = get_funnel_or_fallback_candidates(engine, max_pm_handoff=3)

        assert source == "funnel"
        assert set(symbols) == {"AAPL", "TSLA"}

    def test_valid_empty_funnel_no_fallback(self, engine):
        """When discovery ran successfully but no pm_eligible → empty, no fallback (Req 13.3)."""
        _create_funnel_run_log(engine, stage="discovery", result_status="completed")
        # No pm_eligible candidates created

        symbols, source = get_funnel_or_fallback_candidates(engine, max_pm_handoff=3)

        assert source == "funnel_empty"
        assert symbols == []

    def test_degraded_discovery_still_valid(self, engine):
        """Degraded discovery (deterministic fallback) is still valid — no legacy fallback."""
        _create_funnel_run_log(engine, stage="discovery", result_status="degraded")
        _create_pm_eligible_candidate(engine, "NVDA")

        symbols, source = get_funnel_or_fallback_candidates(engine, max_pm_handoff=3)

        assert source == "funnel"
        assert symbols == ["NVDA"]

    @patch("utils.funnel_transition.get_expanded_watchlist")
    def test_fallback_to_legacy_when_no_discovery_log(self, mock_legacy, engine):
        """No FunnelRunLog for today → fall back to legacy expanded_watchlist."""
        mock_legacy.return_value = ["AMZN", "GOOG", "META", "MSFT"]

        symbols, source = get_funnel_or_fallback_candidates(engine, max_pm_handoff=3)

        assert source == "legacy_fallback"
        # max_pm_handoff ceiling applied
        assert symbols == ["AMZN", "GOOG", "META"]
        mock_legacy.assert_called_once_with(engine)

    @patch("utils.funnel_transition.get_expanded_watchlist")
    def test_fallback_caps_at_max_pm_handoff(self, mock_legacy, engine):
        """Legacy fallback respects max_pm_handoff ceiling (Req 13.2)."""
        mock_legacy.return_value = ["A", "B", "C", "D", "E"]

        symbols, source = get_funnel_or_fallback_candidates(engine, max_pm_handoff=2)

        assert symbols == ["A", "B"]
        assert source == "legacy_fallback"

    @patch("utils.funnel_transition.get_expanded_watchlist")
    def test_fallback_on_error_discovery_log(self, mock_legacy, engine):
        """FunnelRunLog with result_status='error' triggers fallback (missed job)."""
        _create_funnel_run_log(engine, stage="discovery", result_status="error")
        mock_legacy.return_value = ["FB", "SNAP"]

        symbols, source = get_funnel_or_fallback_candidates(engine, max_pm_handoff=3)

        assert source == "legacy_fallback"
        assert symbols == ["FB", "SNAP"]

    @patch("utils.funnel_transition.get_expanded_watchlist")
    def test_fallback_returns_empty_when_legacy_empty(self, mock_legacy, engine):
        """Legacy fallback returns empty list when no legacy data."""
        mock_legacy.return_value = []

        symbols, source = get_funnel_or_fallback_candidates(engine, max_pm_handoff=3)

        assert source == "legacy_fallback"
        assert symbols == []

    @patch("utils.funnel_transition.get_expanded_watchlist")
    def test_fallback_handles_legacy_exception(self, mock_legacy, engine):
        """If legacy get_expanded_watchlist() raises, return empty gracefully."""
        mock_legacy.side_effect = RuntimeError("DB error")

        symbols, source = get_funnel_or_fallback_candidates(engine, max_pm_handoff=3)

        assert source == "legacy_fallback"
        assert symbols == []

    def test_funnel_respects_max_handoff(self, engine):
        """Funnel source respects max_pm_handoff ceiling."""
        _create_funnel_run_log(engine, stage="discovery", result_status="completed")
        _create_pm_eligible_candidate(engine, "AAPL", scout_rank=1)
        _create_pm_eligible_candidate(engine, "TSLA", scout_rank=2)
        _create_pm_eligible_candidate(engine, "NVDA", scout_rank=3)
        _create_pm_eligible_candidate(engine, "AMZN", scout_rank=4)

        symbols, source = get_funnel_or_fallback_candidates(engine, max_pm_handoff=2)

        assert source == "funnel"
        assert len(symbols) == 2


class TestBuildDeduplicatedWatchlist:
    """Tests for build_deduplicated_watchlist()."""

    def test_basic_deduplication(self):
        """Symbols appear only once regardless of source."""
        core = ["AAPL", "MSFT"]
        scout = ["TSLA", "AAPL"]  # AAPL duplicate
        expanded = ["NVDA", "MSFT"]  # MSFT duplicate

        result = build_deduplicated_watchlist(core, scout, expanded)

        assert result == ["AAPL", "MSFT", "TSLA", "NVDA"]

    def test_priority_order_core_first(self):
        """Core watchlist symbols come first."""
        core = ["MSFT", "AAPL"]
        scout = ["GOOG"]
        expanded = ["META"]

        result = build_deduplicated_watchlist(core, scout, expanded)

        assert result == ["MSFT", "AAPL", "GOOG", "META"]

    def test_empty_sources(self):
        """Empty sources don't break anything."""
        result = build_deduplicated_watchlist(["AAPL"], [], [])
        assert result == ["AAPL"]

        result = build_deduplicated_watchlist([], [], ["NVDA"])
        assert result == ["NVDA"]

    def test_all_duplicates(self):
        """All sources have same symbol → appears once."""
        result = build_deduplicated_watchlist(["AAPL"], ["AAPL"], ["AAPL"])
        assert result == ["AAPL"]

    def test_all_empty(self):
        """All empty → empty result."""
        result = build_deduplicated_watchlist([], [], [])
        assert result == []
