"""Unit tests for funnel discovery persistence and adapter functions (Task 3.3).

Tests:
- adapt_candidate_row_to_funnel() — deterministic fallback path
- adapt_chief_scout_pick_to_funnel() — Chief Scout curated path
- persist_discovery_candidates() — FunnelCandidate row creation and upsert
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import create_engine

from db.schema import Base, FunnelCandidate, get_session
from utils.funnel_discovery import (
    adapt_candidate_row_to_funnel,
    adapt_chief_scout_pick_to_funnel,
    persist_discovery_candidates,
)
from utils.sector_scout_models import CandidateRow, ChiefScoutPick


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_engine():
    """Create an in-memory SQLite engine with FunnelCandidate table."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def _make_candidate_row(
    symbol: str = "AAPL",
    sector: str = "tech",
    score: float = 75.0,
    move_pct: float = 3.5,
    relative_volume: float = 2.1,
) -> CandidateRow:
    """Create a realistic CandidateRow for testing."""
    return CandidateRow(
        symbol=symbol,
        sector=sector,
        sector_name="Technology",
        current_price=150.0,
        prev_close=145.0,
        move_pct=move_pct,
        current_volume=5_000_000,
        average_volume=2_500_000,
        relative_volume=relative_volume,
        dollar_volume=750_000_000,
        news_headlines=[{"headline": "AAPL beats earnings", "datetime": 1700000000}],
        news_freshness_minutes=30.0,
        sector_etf="XLK",
        sector_etf_move_pct=1.2,
        sector_confirmed=True,
        bid=149.90,
        ask=150.10,
        spread_pct=0.13,
        spread_status="known",
        scout_score=score,
    )


def _make_chief_scout_pick(
    symbol: str = "NVDA",
    sector: str = "tech",
    score: float = 82.0,
) -> ChiefScoutPick:
    """Create a realistic ChiefScoutPick dict for testing."""
    return {
        "symbol": symbol,
        "sector": sector,
        "direction_bias": "bullish",
        "conviction": "high",
        "catalyst_summary": "Strong AI demand driving GPU sales",
        "reason": "AI infrastructure capex cycle accelerating",
        "risk": "Valuation stretched at 35x forward PE",
        "source_candidate_score": score,
    }


# ---------------------------------------------------------------------------
# adapt_candidate_row_to_funnel
# ---------------------------------------------------------------------------


class TestAdaptCandidateRowToFunnel:
    def test_basic_fields_mapped(self):
        """Core fields are correctly mapped from CandidateRow."""
        row = _make_candidate_row(symbol="AAPL", score=75.0, move_pct=3.5)
        result = adapt_candidate_row_to_funnel(row)

        assert result["symbol"] == "AAPL"
        assert result["scout_score"] == 75.0
        assert result["direction_bias"] == "bullish"  # move_pct > 0.5

    def test_catalyst_evidence_is_valid_json(self):
        """catalyst_evidence is a JSON string with expected keys."""
        row = _make_candidate_row()
        result = adapt_candidate_row_to_funnel(row)

        evidence = json.loads(result["catalyst_evidence"])
        assert "news_headlines" in evidence
        assert "news_freshness_minutes" in evidence
        assert "move_pct" in evidence
        assert "relative_volume" in evidence
        assert "dollar_volume" in evidence

    def test_sector_context_is_valid_json(self):
        """sector_context is a JSON string with sector info."""
        row = _make_candidate_row()
        result = adapt_candidate_row_to_funnel(row)

        ctx = json.loads(result["sector_context"])
        assert ctx["sector"] == "tech"
        assert ctx["sector_name"] == "Technology"
        assert ctx["sector_etf"] == "XLK"
        assert ctx["sector_confirmed"] is True

    def test_direction_bias_bearish(self):
        """Negative move produces bearish bias."""
        row = _make_candidate_row(move_pct=-2.5)
        result = adapt_candidate_row_to_funnel(row)
        assert result["direction_bias"] == "bearish"

    def test_direction_bias_neutral(self):
        """Small move produces neutral bias."""
        row = _make_candidate_row(move_pct=0.1)
        result = adapt_candidate_row_to_funnel(row)
        assert result["direction_bias"] == "neutral"

    def test_direction_bias_none_when_no_move(self):
        """None move_pct produces None direction_bias."""
        row = _make_candidate_row()
        row.move_pct = None
        result = adapt_candidate_row_to_funnel(row)
        assert result["direction_bias"] is None

    def test_selection_reason_includes_context(self):
        """selection_reason includes move, volume, news info."""
        row = _make_candidate_row(move_pct=3.5, relative_volume=2.1)
        row.news_freshness_minutes = 30.0
        result = adapt_candidate_row_to_funnel(row)

        assert "Move" in result["selection_reason"]
        assert "RelVol" in result["selection_reason"]
        assert "News" in result["selection_reason"]

    def test_primary_risk_wide_spread(self):
        """Wide spread is noted in primary_risk."""
        row = _make_candidate_row()
        row.spread_pct = 1.5
        result = adapt_candidate_row_to_funnel(row)
        assert "spread" in result["primary_risk"].lower()

    def test_primary_risk_no_sector_confirmation(self):
        """Missing sector confirmation appears in risk."""
        row = _make_candidate_row()
        row.sector_confirmed = False
        result = adapt_candidate_row_to_funnel(row)
        assert "sector" in result["primary_risk"].lower()


# ---------------------------------------------------------------------------
# adapt_chief_scout_pick_to_funnel
# ---------------------------------------------------------------------------


class TestAdaptChiefScoutPickToFunnel:
    def test_basic_fields_mapped(self):
        """Core fields from ChiefScoutPick are correctly mapped."""
        pick = _make_chief_scout_pick(symbol="NVDA", score=82.0)
        result = adapt_chief_scout_pick_to_funnel(pick)

        assert result["symbol"] == "NVDA"
        assert result["scout_score"] == 82.0
        assert result["direction_bias"] == "bullish"

    def test_catalyst_evidence_contains_summary(self):
        """catalyst_evidence JSON wraps the catalyst_summary."""
        pick = _make_chief_scout_pick()
        result = adapt_chief_scout_pick_to_funnel(pick)

        evidence = json.loads(result["catalyst_evidence"])
        assert evidence["catalyst_summary"] == "Strong AI demand driving GPU sales"
        assert evidence["conviction"] == "high"
        assert evidence["curated_by"] == "chief_scout"

    def test_selection_reason_is_reason_field(self):
        """selection_reason maps from pick's reason."""
        pick = _make_chief_scout_pick()
        result = adapt_chief_scout_pick_to_funnel(pick)
        assert result["selection_reason"] == "AI infrastructure capex cycle accelerating"

    def test_primary_risk_is_risk_field(self):
        """primary_risk maps from pick's risk."""
        pick = _make_chief_scout_pick()
        result = adapt_chief_scout_pick_to_funnel(pick)
        assert result["primary_risk"] == "Valuation stretched at 35x forward PE"

    def test_sector_context_has_sector(self):
        """sector_context JSON includes the sector."""
        pick = _make_chief_scout_pick(sector="semiconductors")
        result = adapt_chief_scout_pick_to_funnel(pick)

        ctx = json.loads(result["sector_context"])
        assert ctx["sector"] == "semiconductors"

    def test_invalid_direction_bias_normalized_to_none(self):
        """Invalid direction_bias values become None."""
        pick = _make_chief_scout_pick()
        pick["direction_bias"] = "sideways"  # invalid
        result = adapt_chief_scout_pick_to_funnel(pick)
        assert result["direction_bias"] is None

    def test_missing_score_defaults_to_zero(self):
        """None source_candidate_score maps to 0.0."""
        pick = _make_chief_scout_pick()
        pick["source_candidate_score"] = None
        result = adapt_chief_scout_pick_to_funnel(pick)
        assert result["scout_score"] == 0.0


# ---------------------------------------------------------------------------
# persist_discovery_candidates
# ---------------------------------------------------------------------------


class TestPersistDiscoveryCandidates:
    def test_creates_new_candidates(self, in_memory_engine):
        """Persists new FunnelCandidate rows for each finalist."""
        finalists = [
            _make_candidate_row("AAPL", "tech", 80.0),
            _make_candidate_row("MSFT", "tech", 75.0),
        ]

        result = persist_discovery_candidates(
            in_memory_engine, finalists, "deterministic_fallback", "premarket"
        )

        assert len(result) == 2
        # Verify in database
        session = get_session(in_memory_engine)
        rows = session.query(FunnelCandidate).all()
        assert len(rows) == 2
        symbols = {r.symbol for r in rows}
        assert symbols == {"AAPL", "MSFT"}
        session.close()

    def test_stage_status_is_awaiting_research(self, in_memory_engine):
        """New candidates have stage_status='awaiting_research'."""
        finalists = [_make_candidate_row("AAPL", "tech", 80.0)]

        result = persist_discovery_candidates(
            in_memory_engine, finalists, "deterministic_fallback", "premarket"
        )

        assert result[0].stage_status == "awaiting_research"

    def test_initial_scout_decision_appended(self, in_memory_engine):
        """New candidates have an initial Scout stage decision."""
        finalists = [_make_candidate_row("AAPL", "tech", 80.0)]

        result = persist_discovery_candidates(
            in_memory_engine, finalists, "deterministic_fallback", "premarket"
        )

        decisions = json.loads(result[0].stage_decisions)
        assert len(decisions) == 1
        assert decisions[0]["agent"] == "scout"
        assert decisions[0]["decision"] == "promoted"
        assert decisions[0]["next_stage"] == "awaiting_research"

    def test_scout_decision_evidence_payload(self, in_memory_engine):
        """Scout decision evidence includes required fields."""
        finalists = [_make_candidate_row("AAPL", "tech", 80.0)]

        result = persist_discovery_candidates(
            in_memory_engine, finalists, "chief_scout", "premarket"
        )

        decisions = json.loads(result[0].stage_decisions)
        evidence = decisions[0]["evidence"]
        assert "scout_score" in evidence
        assert "direction_bias" in evidence
        assert "catalyst_evidence" in evidence
        assert "selection_mode" in evidence
        assert evidence["selection_mode"] == "chief_scout"
        assert "source_run" in evidence
        assert "scout_rank" in evidence

    def test_enforces_max_shortlist_ceiling(self, in_memory_engine):
        """Only max_shortlist candidates are persisted."""
        finalists = [
            _make_candidate_row(f"SYM{i}", "tech", 90.0 - i)
            for i in range(10)
        ]

        result = persist_discovery_candidates(
            in_memory_engine, finalists, "deterministic_fallback", "premarket",
            max_shortlist=5,
        )

        assert len(result) == 5
        session = get_session(in_memory_engine)
        count = session.query(FunnelCandidate).count()
        assert count == 5
        session.close()

    def test_scout_rank_assigned_sequentially(self, in_memory_engine):
        """scout_rank is 1-based sequential per position."""
        finalists = [
            _make_candidate_row("AAPL", "tech", 90.0),
            _make_candidate_row("MSFT", "tech", 85.0),
            _make_candidate_row("GOOGL", "tech", 80.0),
        ]

        result = persist_discovery_candidates(
            in_memory_engine, finalists, "deterministic_fallback", "premarket"
        )

        ranks = [r.scout_rank for r in result]
        assert ranks == [1, 2, 3]

    def test_selection_mode_stored(self, in_memory_engine):
        """selection_mode is stored on the FunnelCandidate."""
        finalists = [_make_candidate_row("AAPL", "tech", 80.0)]

        result = persist_discovery_candidates(
            in_memory_engine, finalists, "chief_scout", "premarket"
        )

        assert result[0].selection_mode == "chief_scout"

    def test_source_run_stored(self, in_memory_engine):
        """source_run is stored on the FunnelCandidate."""
        finalists = [_make_candidate_row("AAPL", "tech", 80.0)]

        result = persist_discovery_candidates(
            in_memory_engine, finalists, "deterministic_fallback", "confirmation"
        )

        assert result[0].source_run == "confirmation"

    def test_chief_scout_picks_persisted(self, in_memory_engine):
        """ChiefScoutPick dicts are correctly adapted and persisted."""
        picks = [
            _make_chief_scout_pick("NVDA", "tech", 82.0),
            _make_chief_scout_pick("AMD", "tech", 78.0),
        ]

        result = persist_discovery_candidates(
            in_memory_engine, picks, "chief_scout", "premarket"
        )

        assert len(result) == 2
        symbols = {r.symbol for r in result}
        assert symbols == {"NVDA", "AMD"}
        assert result[0].selection_mode == "chief_scout"

    def test_empty_finalists_returns_empty(self, in_memory_engine):
        """Empty finalists list returns empty result."""
        result = persist_discovery_candidates(
            in_memory_engine, [], "deterministic_fallback", "premarket"
        )
        assert result == []

    def test_same_day_duplicate_updates_mutable_fields(self, in_memory_engine):
        """Re-discovery on same day updates mutable fields."""
        # First discovery
        finalists_v1 = [_make_candidate_row("AAPL", "tech", 70.0)]
        persist_discovery_candidates(
            in_memory_engine, finalists_v1, "deterministic_fallback", "premarket"
        )

        # Second discovery with updated evidence
        finalists_v2 = [_make_candidate_row("AAPL", "tech", 60.0, move_pct=5.0)]
        persist_discovery_candidates(
            in_memory_engine, finalists_v2, "chief_scout", "premarket"
        )

        # Should still be one row
        session = get_session(in_memory_engine)
        rows = session.query(FunnelCandidate).all()
        assert len(rows) == 1

        # selection_mode updated to latest
        assert rows[0].selection_mode == "chief_scout"
        # catalyst_evidence updated
        evidence = json.loads(rows[0].catalyst_evidence)
        assert evidence["move_pct"] == 5.0
        session.close()

    def test_same_day_duplicate_score_updated_only_if_higher(self, in_memory_engine):
        """scout_score is only updated when new score is higher."""
        # First with score=80
        finalists_v1 = [_make_candidate_row("AAPL", "tech", 80.0)]
        persist_discovery_candidates(
            in_memory_engine, finalists_v1, "deterministic_fallback", "premarket"
        )

        # Second with lower score=60
        finalists_v2 = [_make_candidate_row("AAPL", "tech", 60.0)]
        persist_discovery_candidates(
            in_memory_engine, finalists_v2, "deterministic_fallback", "premarket"
        )

        session = get_session(in_memory_engine)
        row = session.query(FunnelCandidate).first()
        assert row.scout_score == 80.0  # Not regressed
        session.close()

    def test_same_day_duplicate_score_updated_when_higher(self, in_memory_engine):
        """scout_score IS updated when new score is higher."""
        # First with score=70
        finalists_v1 = [_make_candidate_row("AAPL", "tech", 70.0)]
        persist_discovery_candidates(
            in_memory_engine, finalists_v1, "deterministic_fallback", "premarket"
        )

        # Second with higher score=90
        finalists_v2 = [_make_candidate_row("AAPL", "tech", 90.0)]
        persist_discovery_candidates(
            in_memory_engine, finalists_v2, "deterministic_fallback", "premarket"
        )

        session = get_session(in_memory_engine)
        row = session.query(FunnelCandidate).first()
        assert row.scout_score == 90.0  # Updated
        session.close()

    def test_same_day_duplicate_appends_scout_decision(self, in_memory_engine):
        """Re-discovery appends a new Scout decision."""
        finalists = [_make_candidate_row("AAPL", "tech", 80.0)]
        persist_discovery_candidates(
            in_memory_engine, finalists, "deterministic_fallback", "premarket"
        )
        persist_discovery_candidates(
            in_memory_engine, finalists, "chief_scout", "premarket"
        )

        session = get_session(in_memory_engine)
        row = session.query(FunnelCandidate).first()
        decisions = json.loads(row.stage_decisions)
        assert len(decisions) == 2
        assert decisions[0]["agent"] == "scout"
        assert decisions[1]["agent"] == "scout"
        # Second decision reflects new selection_mode
        assert decisions[1]["evidence"]["selection_mode"] == "chief_scout"
        session.close()

    def test_same_day_duplicate_never_regresses_advanced_status(self, in_memory_engine):
        """Re-discovery does NOT regress stage_status beyond awaiting_research."""
        # Create initial candidate
        finalists = [_make_candidate_row("AAPL", "tech", 80.0)]
        persist_discovery_candidates(
            in_memory_engine, finalists, "deterministic_fallback", "premarket"
        )

        # Manually advance the candidate
        session = get_session(in_memory_engine)
        row = session.query(FunnelCandidate).first()
        row.stage_status = "awaiting_analysis"
        session.commit()
        session.close()

        # Re-discover — should NOT regress to awaiting_research
        persist_discovery_candidates(
            in_memory_engine, finalists, "chief_scout", "premarket"
        )

        session = get_session(in_memory_engine)
        row = session.query(FunnelCandidate).first()
        assert row.stage_status == "awaiting_analysis"
        session.close()

    def test_candidate_id_is_valid_uuid(self, in_memory_engine):
        """candidate_id is a valid UUID4 string."""
        finalists = [_make_candidate_row("AAPL", "tech", 80.0)]
        result = persist_discovery_candidates(
            in_memory_engine, finalists, "deterministic_fallback", "premarket"
        )

        # Should not raise
        parsed = uuid.UUID(result[0].candidate_id, version=4)
        assert str(parsed) == result[0].candidate_id

    def test_discovered_at_is_utc(self, in_memory_engine):
        """discovered_at timestamp is set."""
        finalists = [_make_candidate_row("AAPL", "tech", 80.0)]
        result = persist_discovery_candidates(
            in_memory_engine, finalists, "deterministic_fallback", "premarket"
        )

        assert result[0].discovered_at is not None

    def test_date_is_ny_trading_date(self, in_memory_engine):
        """date field uses New York timezone."""
        finalists = [_make_candidate_row("AAPL", "tech", 80.0)]
        result = persist_discovery_candidates(
            in_memory_engine, finalists, "deterministic_fallback", "premarket"
        )

        # The date should be today in NY timezone
        ny_tz = ZoneInfo("America/New_York")
        expected_date = datetime.now(timezone.utc).astimezone(ny_tz).date()
        assert result[0].date == expected_date

    def test_justification_payload_complete(self, in_memory_engine):
        """All required justification fields are populated."""
        finalists = [_make_candidate_row("AAPL", "tech", 80.0)]
        result = persist_discovery_candidates(
            in_memory_engine, finalists, "deterministic_fallback", "premarket"
        )

        candidate = result[0]
        # Required justification payload fields
        assert candidate.catalyst_evidence is not None
        assert candidate.selection_reason is not None
        assert candidate.primary_risk is not None
        assert candidate.sector_context is not None
        assert candidate.direction_bias is not None
        assert candidate.scout_rank is not None
        assert candidate.scout_score is not None

        # All should be non-empty
        assert len(candidate.catalyst_evidence) > 0
        assert len(candidate.selection_reason) > 0
        assert len(candidate.primary_risk) > 0
