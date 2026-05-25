"""Unit tests for utils/sector_scout_persistence.py.

Tests cover:
- persist_run_summary writes RunSummary to AgentMemory with correct key pattern
- persist_run_summary upserts on repeated calls for same run_type/date
- persist_run_summary includes all required fields
- persist_candidate_rows writes per-symbol CandidateRows with correct key pattern
- persist_candidate_rows handles both CandidateRow dataclass and dict inputs
- persist_candidate_rows upserts on repeated calls for same symbol
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine

from db.schema import AgentMemory, Base, get_session
from utils.sector_scout_models import CandidateRow
from utils.sector_scout_persistence import (
    persist_candidate_rows,
    persist_run_summary,
)


@pytest.fixture
def engine():
    """Create an in-memory SQLite database for testing."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _make_screener_result(
    sectors_scanned: int = 3,
    finalists: dict | None = None,
    candidates: dict | None = None,
    rejections: list | None = None,
) -> dict:
    """Helper to create a minimal screener result dict."""
    return {
        "sectors_scanned": sectors_scanned,
        "finalists_by_sector": finalists or {},
        "candidates_by_sector": candidates or {"ai_semi": ["A", "B", "C"]},
        "global_finalists": [],
        "rejections": rejections or [{"symbol": "X", "reason_code": "penny_stock"}],
        "reason_counts": {"penny_stock": 1},
        "budget_hits": [],
    }


def _make_chief_result(
    picks: list | None = None,
    fallback_used: bool = False,
) -> dict:
    """Helper to create a minimal chief scout result dict."""
    return {
        "picks": picks or [
            {"symbol": "AVGO", "sector": "ai_semi", "conviction": "high"},
            {"symbol": "SMCI", "sector": "ai_semi", "conviction": "medium"},
        ],
        "fallback_used": fallback_used,
        "llm_error": None,
    }


def _make_candidate_row(symbol: str, sector: str = "ai_semi", score: float = 75.0) -> CandidateRow:
    """Helper to create a CandidateRow dataclass instance."""
    return CandidateRow(
        symbol=symbol,
        sector=sector,
        sector_name="AI / Semiconductors",
        current_price=150.0,
        move_pct=3.5,
        relative_volume=2.1,
        dollar_volume=50_000_000.0,
        scout_score=score,
        component_scores={"move_pct": 20.0, "relative_volume": 15.0},
        penalties_applied=[{"type": "stale_news", "deduction": 15.0}],
        hard_gate_passed=True,
        run_type="premarket",
    )


class TestPersistRunSummary:
    """Tests for persist_run_summary()."""

    def test_writes_run_summary_to_memory(self, engine):
        """Run summary is written to AgentMemory with correct key pattern."""
        screener_result = _make_screener_result()
        chief_result = _make_chief_result()

        persist_run_summary(engine, screener_result, chief_result, "premarket", 5.2)

        db = get_session(engine)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        record = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout", key=f"run_summary:{today}:premarket")
            .first()
        )
        assert record is not None
        data = json.loads(record.value)
        assert data["run_type"] == "premarket"
        assert data["sectors_scanned"] == 3
        assert data["duration_seconds"] == 5.2
        db.close()

    def test_includes_all_required_fields(self, engine):
        """Persisted run summary includes all RunSummary fields."""
        screener_result = _make_screener_result(
            sectors_scanned=5,
            candidates={"ai_semi": ["A", "B"], "ev": ["C", "D", "E"]},
            rejections=[
                {"symbol": "X", "reason_code": "penny_stock"},
                {"symbol": "Y", "reason_code": "missing_price"},
            ],
        )
        chief_result = _make_chief_result(
            picks=[{"symbol": "AVGO", "sector": "ai_semi"}],
            fallback_used=True,
        )

        persist_run_summary(engine, screener_result, chief_result, "midday", 12.5)

        db = get_session(engine)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        record = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout", key=f"run_summary:{today}:midday")
            .first()
        )
        data = json.loads(record.value)

        # Verify all required fields
        assert data["run_type"] == "midday"
        assert "timestamp" in data
        assert data["sectors_scanned"] == 5
        assert data["total_candidates_evaluated"] == 5  # 2 + 3
        assert data["hard_gate_rejections"] == 2
        assert data["finalists_count"] == 0  # empty finalists_by_sector
        assert data["chief_scout_picks"] == [{"symbol": "AVGO", "sector": "ai_semi"}]
        assert data["fallback_used"] is True
        assert data["expanded_watchlist_symbols"] == ["AVGO"]
        assert data["expanded_watchlist_size"] == 1
        assert data["reason_counts"] == {"penny_stock": 1}
        assert data["budget_hits"] == []
        assert data["duration_seconds"] == 12.5
        db.close()

    def test_upserts_on_repeated_calls(self, engine):
        """Repeated calls for same date/run_type update rather than duplicate."""
        screener_result = _make_screener_result()
        chief_result = _make_chief_result()

        persist_run_summary(engine, screener_result, chief_result, "premarket", 5.0)
        persist_run_summary(engine, screener_result, chief_result, "premarket", 8.0)

        db = get_session(engine)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        records = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout", key=f"run_summary:{today}:premarket")
            .all()
        )
        # Should only have one record (upsert, not duplicate)
        assert len(records) == 1
        data = json.loads(records[0].value)
        assert data["duration_seconds"] == 8.0  # Updated value
        db.close()

    def test_different_run_types_stored_separately(self, engine):
        """Different run_types get separate keys."""
        screener_result = _make_screener_result()
        chief_result = _make_chief_result()

        persist_run_summary(engine, screener_result, chief_result, "premarket", 5.0)
        persist_run_summary(engine, screener_result, chief_result, "confirmation", 7.0)

        db = get_session(engine)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        premarket = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout", key=f"run_summary:{today}:premarket")
            .first()
        )
        confirmation = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout", key=f"run_summary:{today}:confirmation")
            .first()
        )
        assert premarket is not None
        assert confirmation is not None
        assert json.loads(premarket.value)["duration_seconds"] == 5.0
        assert json.loads(confirmation.value)["duration_seconds"] == 7.0
        db.close()


class TestPersistCandidateRows:
    """Tests for persist_candidate_rows()."""

    def test_writes_candidate_rows_to_memory(self, engine):
        """CandidateRows are written with correct key pattern."""
        finalists = {
            "ai_semi": [
                _make_candidate_row("AVGO", score=80.0),
                _make_candidate_row("SMCI", score=70.0),
            ],
        }

        persist_candidate_rows(engine, finalists, "premarket")

        db = get_session(engine)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        avgo_record = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout", key=f"candidate_row:{today}:premarket:AVGO")
            .first()
        )
        smci_record = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout", key=f"candidate_row:{today}:premarket:SMCI")
            .first()
        )

        assert avgo_record is not None
        assert smci_record is not None

        avgo_data = json.loads(avgo_record.value)
        assert avgo_data["symbol"] == "AVGO"
        assert avgo_data["scout_score"] == 80.0
        assert avgo_data["sector"] == "ai_semi"

        smci_data = json.loads(smci_record.value)
        assert smci_data["symbol"] == "SMCI"
        assert smci_data["scout_score"] == 70.0
        db.close()

    def test_includes_scoring_details(self, engine):
        """Persisted candidate rows include component scores and penalties."""
        candidate = _make_candidate_row("AVGO", score=75.0)
        finalists = {"ai_semi": [candidate]}

        persist_candidate_rows(engine, finalists, "premarket")

        db = get_session(engine)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        record = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout", key=f"candidate_row:{today}:premarket:AVGO")
            .first()
        )
        data = json.loads(record.value)

        assert data["component_scores"] == {"move_pct": 20.0, "relative_volume": 15.0}
        assert data["penalties_applied"] == [{"type": "stale_news", "deduction": 15.0}]
        assert data["hard_gate_passed"] is True
        assert data["move_pct"] == 3.5
        assert data["relative_volume"] == 2.1
        db.close()

    def test_handles_dict_candidates(self, engine):
        """Works with dict-based candidates (not just dataclass instances)."""
        finalists = {
            "financials": [
                {
                    "symbol": "GS",
                    "sector": "financials",
                    "scout_score": 65.0,
                    "move_pct": 2.0,
                    "component_scores": {"move_pct": 10.0},
                    "penalties_applied": [],
                },
            ],
        }

        persist_candidate_rows(engine, finalists, "confirmation")

        db = get_session(engine)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        record = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout", key=f"candidate_row:{today}:confirmation:GS")
            .first()
        )
        assert record is not None
        data = json.loads(record.value)
        assert data["symbol"] == "GS"
        assert data["scout_score"] == 65.0
        db.close()

    def test_upserts_on_repeated_calls(self, engine):
        """Repeated calls for same symbol/date/run_type update rather than duplicate."""
        finalists1 = {"ai_semi": [_make_candidate_row("AVGO", score=70.0)]}
        finalists2 = {"ai_semi": [_make_candidate_row("AVGO", score=85.0)]}

        persist_candidate_rows(engine, finalists1, "premarket")
        persist_candidate_rows(engine, finalists2, "premarket")

        db = get_session(engine)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        records = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout", key=f"candidate_row:{today}:premarket:AVGO")
            .all()
        )
        assert len(records) == 1
        data = json.loads(records[0].value)
        assert data["scout_score"] == 85.0  # Updated value
        db.close()

    def test_multiple_sectors(self, engine):
        """Handles finalists from multiple sectors."""
        finalists = {
            "ai_semi": [_make_candidate_row("AVGO", "ai_semi", 80.0)],
            "financials": [_make_candidate_row("GS", "financials", 65.0)],
            "ev_high_beta": [_make_candidate_row("RIVN", "ev_high_beta", 55.0)],
        }

        persist_candidate_rows(engine, finalists, "midday")

        db = get_session(engine)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        for symbol in ["AVGO", "GS", "RIVN"]:
            record = (
                db.query(AgentMemory)
                .filter_by(agent="sector_scout", key=f"candidate_row:{today}:midday:{symbol}")
                .first()
            )
            assert record is not None, f"Missing record for {symbol}"
        db.close()

    def test_sets_symbol_field_on_memory_record(self, engine):
        """AgentMemory record has symbol field set for per-symbol tracking."""
        finalists = {"ai_semi": [_make_candidate_row("AVGO", score=80.0)]}

        persist_candidate_rows(engine, finalists, "premarket")

        db = get_session(engine)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        record = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout", key=f"candidate_row:{today}:premarket:AVGO")
            .first()
        )
        assert record.symbol == "AVGO"
        db.close()

    def test_empty_finalists_does_nothing(self, engine):
        """Empty finalists dict doesn't create any records."""
        persist_candidate_rows(engine, {}, "premarket")

        db = get_session(engine)
        records = db.query(AgentMemory).filter_by(agent="sector_scout").all()
        assert len(records) == 0
        db.close()
