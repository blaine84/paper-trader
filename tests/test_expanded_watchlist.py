"""Unit tests for utils/expanded_watchlist.py.

Tests cover:
- update_expanded_watchlist merges picks and deduplicates by symbol
- update_expanded_watchlist enforces max_expanded_watchlist ceiling
- update_expanded_watchlist keeps highest scout_score on duplicate
- get_expanded_watchlist returns today's symbols
- get_expanded_watchlist returns empty list when no data
- expire_expanded_watchlist marks yesterday's watchlist as expired
- Never mutates Core_Watchlist
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine

from db.schema import AgentMemory, Base, get_session
from utils.expanded_watchlist import (
    expire_expanded_watchlist,
    get_expanded_watchlist,
    update_expanded_watchlist,
)


@pytest.fixture
def engine():
    """Create an in-memory SQLite database for testing."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _make_config(max_expanded_watchlist: int = 12) -> dict:
    """Helper to create a minimal config dict."""
    return {
        "budget_ceilings": {
            "max_expanded_watchlist": max_expanded_watchlist,
        },
    }


def _make_pick(symbol: str, score: float = 50.0, sector: str = "ai_semi") -> dict:
    """Helper to create a minimal ChiefScoutPick-like dict."""
    return {
        "symbol": symbol,
        "sector": sector,
        "direction_bias": "bullish",
        "conviction": "medium",
        "catalyst_summary": f"Test catalyst for {symbol}",
        "reason": f"Test reason for {symbol}",
        "risk": "Test risk",
        "source_candidate_score": score,
    }


class TestUpdateExpandedWatchlist:
    """Tests for update_expanded_watchlist()."""

    def test_writes_picks_to_memory(self, engine):
        """New picks are written to AgentMemory."""
        picks = [_make_pick("AVGO", 80.0), _make_pick("SMCI", 70.0)]
        config = _make_config()

        result = update_expanded_watchlist(engine, picks, "premarket", config)

        assert "AVGO" in result
        assert "SMCI" in result
        assert len(result) == 2

    def test_merges_with_existing_picks(self, engine):
        """New picks are merged with existing picks from earlier runs."""
        config = _make_config()

        # First run
        picks1 = [_make_pick("AVGO", 80.0), _make_pick("SMCI", 70.0)]
        update_expanded_watchlist(engine, picks1, "premarket", config)

        # Second run with different symbols
        picks2 = [_make_pick("ARM", 75.0), _make_pick("MU", 65.0)]
        result = update_expanded_watchlist(engine, picks2, "confirmation", config)

        assert len(result) == 4
        assert set(result) == {"AVGO", "SMCI", "ARM", "MU"}

    def test_deduplicates_keeps_highest_score(self, engine):
        """When same symbol appears in multiple runs, keeps highest score."""
        config = _make_config()

        # First run: AVGO with score 60
        picks1 = [_make_pick("AVGO", 60.0)]
        update_expanded_watchlist(engine, picks1, "premarket", config)

        # Second run: AVGO with higher score 85
        picks2 = [_make_pick("AVGO", 85.0)]
        result = update_expanded_watchlist(engine, picks2, "confirmation", config)

        assert result == ["AVGO"]

        # Verify the stored score is the higher one
        db = get_session(engine)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        record = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout", key=f"expanded_watchlist:{today}")
            .first()
        )
        data = json.loads(record.value)
        assert data["picks"][0]["source_candidate_score"] == 85.0
        db.close()

    def test_deduplicates_keeps_existing_when_higher(self, engine):
        """When existing score is higher, keeps existing version."""
        config = _make_config()

        # First run: AVGO with score 90
        picks1 = [_make_pick("AVGO", 90.0)]
        update_expanded_watchlist(engine, picks1, "premarket", config)

        # Second run: AVGO with lower score 60
        picks2 = [_make_pick("AVGO", 60.0)]
        result = update_expanded_watchlist(engine, picks2, "confirmation", config)

        assert result == ["AVGO"]

        # Verify the stored score is still the higher one
        db = get_session(engine)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        record = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout", key=f"expanded_watchlist:{today}")
            .first()
        )
        data = json.loads(record.value)
        assert data["picks"][0]["source_candidate_score"] == 90.0
        db.close()

    def test_enforces_max_ceiling(self, engine):
        """Stops adding when max_expanded_watchlist cap is reached."""
        config = _make_config(max_expanded_watchlist=3)

        picks = [
            _make_pick("AVGO", 80.0),
            _make_pick("SMCI", 70.0),
            _make_pick("ARM", 60.0),
            _make_pick("MU", 50.0),
            _make_pick("INTC", 40.0),
        ]
        result = update_expanded_watchlist(engine, picks, "premarket", config)

        assert len(result) == 3
        # Should keep the top 3 by score
        assert set(result) == {"AVGO", "SMCI", "ARM"}

    def test_ceiling_across_multiple_runs(self, engine):
        """Ceiling is enforced across multiple runs in a day."""
        config = _make_config(max_expanded_watchlist=4)

        # First run fills 3 slots
        picks1 = [_make_pick("AVGO", 80.0), _make_pick("SMCI", 70.0), _make_pick("ARM", 60.0)]
        update_expanded_watchlist(engine, picks1, "premarket", config)

        # Second run tries to add 3 more, but only 1 slot available
        picks2 = [_make_pick("MU", 90.0), _make_pick("INTC", 50.0), _make_pick("QCOM", 40.0)]
        result = update_expanded_watchlist(engine, picks2, "confirmation", config)

        assert len(result) == 4
        # MU has highest score (90), so it should be included along with top existing
        assert "MU" in result

    def test_empty_picks_returns_existing(self, engine):
        """Passing empty picks returns existing watchlist unchanged."""
        config = _make_config()

        picks1 = [_make_pick("AVGO", 80.0)]
        update_expanded_watchlist(engine, picks1, "premarket", config)

        result = update_expanded_watchlist(engine, [], "confirmation", config)
        assert result == ["AVGO"]

    def test_returns_symbols_sorted_by_score(self, engine):
        """Returned symbols are ordered by score descending."""
        config = _make_config()

        picks = [
            _make_pick("LOW", 30.0),
            _make_pick("HIGH", 90.0),
            _make_pick("MID", 60.0),
        ]
        result = update_expanded_watchlist(engine, picks, "premarket", config)

        assert result == ["HIGH", "MID", "LOW"]


class TestGetExpandedWatchlist:
    """Tests for get_expanded_watchlist()."""

    def test_returns_todays_symbols(self, engine):
        """Returns symbols from today's watchlist."""
        config = _make_config()
        picks = [_make_pick("AVGO", 80.0), _make_pick("SMCI", 70.0)]
        update_expanded_watchlist(engine, picks, "premarket", config)

        result = get_expanded_watchlist(engine)
        assert set(result) == {"AVGO", "SMCI"}

    def test_returns_empty_when_no_data(self, engine):
        """Returns empty list when no watchlist exists."""
        result = get_expanded_watchlist(engine)
        assert result == []

    def test_returns_empty_for_stale_date(self, engine):
        """Returns empty list if stored date doesn't match today."""
        # Manually insert a record with yesterday's date
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        db = get_session(engine)
        db.add(AgentMemory(
            agent="sector_scout",
            symbol=None,
            key=f"expanded_watchlist:{yesterday}",
            value=json.dumps({
                "date": yesterday,
                "picks": [_make_pick("AVGO", 80.0)],
                "symbols": ["AVGO"],
                "size": 1,
            }),
        ))
        db.commit()
        db.close()

        # Today's key won't match yesterday's record
        result = get_expanded_watchlist(engine)
        assert result == []


class TestExpireExpandedWatchlist:
    """Tests for expire_expanded_watchlist()."""

    def test_marks_yesterday_as_expired(self, engine):
        """Yesterday's watchlist is marked as expired."""
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        memory_key = f"expanded_watchlist:{yesterday}"

        # Insert yesterday's watchlist
        db = get_session(engine)
        db.add(AgentMemory(
            agent="sector_scout",
            symbol=None,
            key=memory_key,
            value=json.dumps({
                "date": yesterday,
                "picks": [_make_pick("AVGO", 80.0)],
                "symbols": ["AVGO"],
                "size": 1,
            }),
        ))
        db.commit()
        db.close()

        # Expire it
        expire_expanded_watchlist(engine)

        # Verify it's marked as expired
        db = get_session(engine)
        record = db.query(AgentMemory).filter_by(agent="sector_scout", key=memory_key).first()
        assert record is not None
        data = json.loads(record.value)
        assert data["expired"] is True
        assert "expired_at" in data
        db.close()

    def test_does_not_affect_today(self, engine):
        """Today's watchlist is not affected by expire call."""
        config = _make_config()
        picks = [_make_pick("AVGO", 80.0)]
        update_expanded_watchlist(engine, picks, "premarket", config)

        expire_expanded_watchlist(engine)

        # Today's watchlist should still be accessible
        result = get_expanded_watchlist(engine)
        assert result == ["AVGO"]

    def test_no_error_when_nothing_to_expire(self, engine):
        """No error when there's nothing to expire."""
        # Should not raise
        expire_expanded_watchlist(engine)

    def test_never_mutates_core_watchlist(self, engine):
        """Expiring expanded watchlist never touches core watchlist data."""
        # Insert a core watchlist record (different agent/key pattern)
        db = get_session(engine)
        db.add(AgentMemory(
            agent="scout",
            symbol=None,
            key="daily_picks",
            value=json.dumps({
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "picks": [{"symbol": "NVDA"}, {"symbol": "AMD"}],
            }),
        ))
        db.commit()
        db.close()

        expire_expanded_watchlist(engine)

        # Core watchlist record should be untouched
        db = get_session(engine)
        record = db.query(AgentMemory).filter_by(agent="scout", key="daily_picks").first()
        assert record is not None
        data = json.loads(record.value)
        assert len(data["picks"]) == 2
        db.close()
