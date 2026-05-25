"""Unit tests for utils/sector_scout_outcomes.py.

Tests cover:
- record_analyst_outcome stores signal for expanded candidates
- record_analyst_outcome ignores non-expanded symbols
- record_pm_outcome stores PM status for expanded candidates
- record_pm_outcome ignores non-expanded symbols
- record_trade_outcome stores trade result for expanded candidates
- get_candidate_outcomes returns all outcomes for a date
- Incremental updates preserve existing fields
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine

from db.schema import AgentMemory, Base, get_session
from utils.sector_scout_outcomes import (
    get_candidate_outcomes,
    record_analyst_outcome,
    record_pm_outcome,
    record_trade_outcome,
)


@pytest.fixture
def engine():
    """Create an in-memory SQLite database for testing."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _seed_expanded_watchlist(engine, date: str, symbols: list[str]) -> None:
    """Seed an expanded watchlist record for the given date."""
    db = get_session(engine)
    db.add(AgentMemory(
        agent="sector_scout",
        symbol=None,
        key=f"expanded_watchlist:{date}",
        value=json.dumps({
            "date": date,
            "symbols": symbols,
            "picks": [{"symbol": s, "source_candidate_score": 50.0} for s in symbols],
            "size": len(symbols),
        }),
    ))
    db.commit()
    db.close()


class TestRecordAnalystOutcome:
    """Tests for record_analyst_outcome()."""

    def test_records_signal_for_expanded_candidate(self, engine):
        """Records analyst signal when symbol is in expanded watchlist."""
        _seed_expanded_watchlist(engine, "2025-01-15", ["AVGO", "SMCI"])

        record_analyst_outcome(engine, "AVGO", "2025-01-15", "LONG")

        outcomes = get_candidate_outcomes(engine, "2025-01-15")
        assert len(outcomes) == 1
        assert outcomes[0]["symbol"] == "AVGO"
        assert outcomes[0]["analyst_signal"] == "LONG"
        assert outcomes[0]["analyst_recorded_at"] is not None

    def test_ignores_non_expanded_symbol(self, engine):
        """Does not record when symbol is not in expanded watchlist."""
        _seed_expanded_watchlist(engine, "2025-01-15", ["AVGO", "SMCI"])

        record_analyst_outcome(engine, "AAPL", "2025-01-15", "LONG")

        outcomes = get_candidate_outcomes(engine, "2025-01-15")
        assert len(outcomes) == 0

    def test_ignores_when_no_watchlist_exists(self, engine):
        """Does not record when no expanded watchlist exists for date."""
        record_analyst_outcome(engine, "AVGO", "2025-01-15", "LONG")

        outcomes = get_candidate_outcomes(engine, "2025-01-15")
        assert len(outcomes) == 0


class TestRecordPmOutcome:
    """Tests for record_pm_outcome()."""

    def test_records_pm_status_for_expanded_candidate(self, engine):
        """Records PM status when symbol is in expanded watchlist."""
        _seed_expanded_watchlist(engine, "2025-01-15", ["AVGO", "SMCI"])

        record_pm_outcome(engine, "SMCI", "2025-01-15", "eligible")

        outcomes = get_candidate_outcomes(engine, "2025-01-15")
        assert len(outcomes) == 1
        assert outcomes[0]["symbol"] == "SMCI"
        assert outcomes[0]["pm_status"] == "eligible"
        assert outcomes[0]["pm_recorded_at"] is not None

    def test_ignores_non_expanded_symbol(self, engine):
        """Does not record when symbol is not in expanded watchlist."""
        _seed_expanded_watchlist(engine, "2025-01-15", ["AVGO"])

        record_pm_outcome(engine, "TSLA", "2025-01-15", "executed")

        outcomes = get_candidate_outcomes(engine, "2025-01-15")
        assert len(outcomes) == 0


class TestRecordTradeOutcome:
    """Tests for record_trade_outcome()."""

    def test_records_trade_outcome_for_expanded_candidate(self, engine):
        """Records trade outcome when symbol is in expanded watchlist."""
        _seed_expanded_watchlist(engine, "2025-01-15", ["AVGO"])

        outcome = {
            "pnl_pct": 2.5,
            "direction": "LONG",
            "entry_price": 150.0,
            "exit_price": 153.75,
        }
        record_trade_outcome(engine, "AVGO", "2025-01-15", outcome)

        outcomes = get_candidate_outcomes(engine, "2025-01-15")
        assert len(outcomes) == 1
        assert outcomes[0]["trade_outcome"]["pnl_pct"] == 2.5
        assert outcomes[0]["trade_outcome"]["direction"] == "LONG"
        assert outcomes[0]["trade_recorded_at"] is not None

    def test_ignores_non_expanded_symbol(self, engine):
        """Does not record when symbol is not in expanded watchlist."""
        _seed_expanded_watchlist(engine, "2025-01-15", ["AVGO"])

        record_trade_outcome(engine, "MSFT", "2025-01-15", {"pnl_pct": 1.0})

        outcomes = get_candidate_outcomes(engine, "2025-01-15")
        assert len(outcomes) == 0


class TestIncrementalUpdates:
    """Tests for incremental outcome record updates."""

    def test_preserves_existing_fields_on_update(self, engine):
        """Updating one field preserves previously recorded fields."""
        _seed_expanded_watchlist(engine, "2025-01-15", ["AVGO"])

        # Record analyst signal first
        record_analyst_outcome(engine, "AVGO", "2025-01-15", "LONG")

        # Then record PM status — analyst_signal should be preserved
        record_pm_outcome(engine, "AVGO", "2025-01-15", "executed")

        outcomes = get_candidate_outcomes(engine, "2025-01-15")
        assert len(outcomes) == 1
        assert outcomes[0]["analyst_signal"] == "LONG"
        assert outcomes[0]["pm_status"] == "executed"

    def test_full_lifecycle_tracking(self, engine):
        """Records full lifecycle: analyst → PM → trade."""
        _seed_expanded_watchlist(engine, "2025-01-15", ["AVGO"])

        record_analyst_outcome(engine, "AVGO", "2025-01-15", "LONG")
        record_pm_outcome(engine, "AVGO", "2025-01-15", "executed")
        record_trade_outcome(engine, "AVGO", "2025-01-15", {
            "pnl_pct": 3.2,
            "direction": "LONG",
            "entry_price": 150.0,
            "exit_price": 154.80,
        })

        outcomes = get_candidate_outcomes(engine, "2025-01-15")
        assert len(outcomes) == 1
        record = outcomes[0]
        assert record["analyst_signal"] == "LONG"
        assert record["pm_status"] == "executed"
        assert record["trade_outcome"]["pnl_pct"] == 3.2


class TestGetCandidateOutcomes:
    """Tests for get_candidate_outcomes()."""

    def test_returns_all_outcomes_for_date(self, engine):
        """Returns all outcome records for the given date."""
        _seed_expanded_watchlist(engine, "2025-01-15", ["AVGO", "SMCI", "ARM"])

        record_analyst_outcome(engine, "AVGO", "2025-01-15", "LONG")
        record_analyst_outcome(engine, "SMCI", "2025-01-15", "SHORT")
        record_analyst_outcome(engine, "ARM", "2025-01-15", "HOLD")

        outcomes = get_candidate_outcomes(engine, "2025-01-15")
        assert len(outcomes) == 3
        symbols = {o["symbol"] for o in outcomes}
        assert symbols == {"AVGO", "SMCI", "ARM"}

    def test_returns_empty_for_no_outcomes(self, engine):
        """Returns empty list when no outcomes exist for date."""
        outcomes = get_candidate_outcomes(engine, "2025-01-15")
        assert outcomes == []

    def test_does_not_return_other_dates(self, engine):
        """Only returns outcomes for the specified date."""
        _seed_expanded_watchlist(engine, "2025-01-15", ["AVGO"])
        _seed_expanded_watchlist(engine, "2025-01-16", ["SMCI"])

        record_analyst_outcome(engine, "AVGO", "2025-01-15", "LONG")
        record_analyst_outcome(engine, "SMCI", "2025-01-16", "SHORT")

        outcomes = get_candidate_outcomes(engine, "2025-01-15")
        assert len(outcomes) == 1
        assert outcomes[0]["symbol"] == "AVGO"
