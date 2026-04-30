"""Tests for compute_confidence_regime (Task 2.4)."""

import json
from datetime import datetime, date
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine

from db.schema import (
    Base, Trade, Position, AgentMemory, get_session,
)
from agents.narrator import compute_confidence_regime


@pytest.fixture
def engine():
    """In-memory SQLite engine with all tables created."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _add_trade(db, symbol, edge_score, entry_time=None):
    """Helper to insert a trade with an edge score for today."""
    if entry_time is None:
        entry_time = datetime.utcnow()
    db.add(Trade(
        symbol=symbol, direction="LONG", quantity=10,
        entry_price=100.0, edge_score=edge_score,
        entry_time=entry_time,
    ))


def _add_analyst_signal(db, symbol, signal):
    """Helper to insert an analyst signal into AgentMemory."""
    db.add(AgentMemory(
        agent="analyst", key="signal", symbol=symbol,
        value=json.dumps({"signal": signal}),
    ))


def _add_position(db, symbol):
    """Helper to insert an open position."""
    db.add(Position(
        symbol=symbol, side="long", quantity=10, avg_cost=100.0,
    ))


# --- Return structure tests ---

def test_returns_required_keys(engine):
    """Returned dict always has the six required keys."""
    result = compute_confidence_regime(engine)
    for key in ("overall", "avg_edge_score", "hold_ratio",
                "stale_positions", "total_positions", "tape_noisy"):
        assert key in result, f"Missing key: {key}"


def test_overall_label_is_valid(engine):
    """Overall label is one of the four valid values or 'unknown'."""
    result = compute_confidence_regime(engine)
    valid = {"high conviction", "moderate conviction", "low conviction",
             "deteriorating", "unknown"}
    assert result["overall"] in valid


# --- Edge quality dimension ---

def test_avg_edge_none_when_no_trades(engine):
    """avg_edge_score is None when no trades exist today."""
    result = compute_confidence_regime(engine)
    assert result["avg_edge_score"] is None


def test_avg_edge_computed_from_todays_trades(engine):
    """avg_edge_score is the mean of today's trade edge scores."""
    db = get_session(engine)
    _add_trade(db, "TSLA", 0.8)
    _add_trade(db, "NVDA", 0.6)
    db.commit()
    db.close()

    result = compute_confidence_regime(engine)
    assert result["avg_edge_score"] == 0.7


# --- Signal disagreement dimension ---

def test_hold_ratio_zero_when_no_signals(engine):
    """hold_ratio is 0.0 when no analyst signals exist."""
    result = compute_confidence_regime(engine)
    assert result["hold_ratio"] == 0.0


def test_hold_ratio_computed_correctly(engine):
    """hold_ratio reflects the fraction of HOLD signals."""
    db = get_session(engine)
    _add_analyst_signal(db, "TSLA", "LONG")
    _add_analyst_signal(db, "NVDA", "HOLD")
    _add_analyst_signal(db, "AMD", "HOLD")
    _add_analyst_signal(db, "AAPL", "SHORT")
    db.commit()
    db.close()

    result = compute_confidence_regime(engine)
    assert result["hold_ratio"] == 0.5  # 2 HOLD out of 4


# --- Tape noise dimension ---

def test_tape_noisy_when_hold_ratio_above_half(engine):
    """tape_noisy is True when hold_ratio > 0.5."""
    db = get_session(engine)
    _add_analyst_signal(db, "TSLA", "HOLD")
    _add_analyst_signal(db, "NVDA", "HOLD")
    _add_analyst_signal(db, "AMD", "LONG")
    db.commit()
    db.close()

    result = compute_confidence_regime(engine)
    assert result["hold_ratio"] == pytest.approx(0.67, abs=0.01)
    assert result["tape_noisy"] is True


def test_tape_not_noisy_when_hold_ratio_at_half(engine):
    """tape_noisy is False when hold_ratio == 0.5 (not strictly greater)."""
    db = get_session(engine)
    _add_analyst_signal(db, "TSLA", "HOLD")
    _add_analyst_signal(db, "NVDA", "LONG")
    db.commit()
    db.close()

    result = compute_confidence_regime(engine)
    assert result["hold_ratio"] == 0.5
    assert result["tape_noisy"] is False


# --- Positions and stale count ---

def test_total_positions_reflects_open_positions(engine):
    """total_positions counts all open positions."""
    db = get_session(engine)
    _add_position(db, "TSLA")
    _add_position(db, "NVDA")
    db.commit()
    db.close()

    result = compute_confidence_regime(engine)
    assert result["total_positions"] == 2


def test_stale_positions_zero_when_no_positions(engine):
    """stale_positions is 0 when there are no open positions."""
    result = compute_confidence_regime(engine)
    assert result["stale_positions"] == 0


# --- Overall label logic ---

def test_high_conviction_when_edge_high_no_noise_no_stale(engine):
    """Overall is 'high conviction' when avg_edge >= 0.6, not noisy, no stale."""
    db = get_session(engine)
    _add_trade(db, "TSLA", 0.8)
    _add_trade(db, "NVDA", 0.7)
    _add_analyst_signal(db, "TSLA", "LONG")
    _add_analyst_signal(db, "NVDA", "LONG")
    db.commit()
    db.close()

    result = compute_confidence_regime(engine)
    assert result["overall"] == "high conviction"


def test_low_conviction_when_edge_low(engine):
    """Overall is 'low conviction' when avg_edge < 0.4."""
    db = get_session(engine)
    _add_trade(db, "TSLA", 0.2)
    _add_trade(db, "NVDA", 0.3)
    _add_analyst_signal(db, "TSLA", "LONG")
    _add_analyst_signal(db, "NVDA", "LONG")
    db.commit()
    db.close()

    result = compute_confidence_regime(engine)
    assert result["overall"] == "low conviction"


def test_deteriorating_when_tape_noisy(engine):
    """Overall is 'deteriorating' when tape is noisy and edge is moderate."""
    db = get_session(engine)
    _add_trade(db, "TSLA", 0.5)
    # 3 HOLD out of 4 signals → hold_ratio = 0.75 > 0.5
    _add_analyst_signal(db, "TSLA", "HOLD")
    _add_analyst_signal(db, "NVDA", "HOLD")
    _add_analyst_signal(db, "AMD", "HOLD")
    _add_analyst_signal(db, "AAPL", "LONG")
    db.commit()
    db.close()

    result = compute_confidence_regime(engine)
    assert result["overall"] == "deteriorating"


def test_moderate_conviction_default(engine):
    """Overall is 'moderate conviction' when no extreme conditions apply."""
    db = get_session(engine)
    _add_trade(db, "TSLA", 0.5)
    _add_analyst_signal(db, "TSLA", "LONG")
    _add_analyst_signal(db, "NVDA", "LONG")
    db.commit()
    db.close()

    result = compute_confidence_regime(engine)
    assert result["overall"] == "moderate conviction"


# --- Error resilience ---

def test_returns_unknown_on_catastrophic_failure(engine):
    """Returns {'overall': 'unknown'} if a query inside the computation fails."""
    from unittest.mock import MagicMock
    # Create a session whose .query() always raises
    mock_db = MagicMock()
    mock_db.query.side_effect = RuntimeError("boom")
    with patch("agents.narrator.get_session", return_value=mock_db):
        result = compute_confidence_regime(engine)
    assert result == {"overall": "unknown"}


def test_empty_db_returns_moderate_conviction(engine):
    """Empty database produces a sensible default (moderate conviction)."""
    result = compute_confidence_regime(engine)
    assert result["overall"] == "moderate conviction"
    assert result["avg_edge_score"] is None
    assert result["hold_ratio"] == 0.0
    assert result["stale_positions"] == 0
    assert result["total_positions"] == 0
    assert result["tape_noisy"] is False
