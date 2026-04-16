"""
Unit tests for DRIFTING label in get_portfolio_for_profile() output.

Verifies that each open position in the portfolio snapshot includes
a "drifting" key reflecting whether the position has received analyst
signals since entry.

Requirements: 3.5
"""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, AgentMemory, Trade, Position, Balance
from models.case import Case  # noqa: F401 — registers with Base
from agents.portfolio_manager import get_portfolio_for_profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine():
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    return engine


def _make_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()


def _seed_position(db, symbol, profile_id, side="long", quantity=10, avg_cost=150.0):
    pos = Position(
        symbol=symbol, profile=profile_id, side=side,
        quantity=quantity, avg_cost=avg_cost,
    )
    db.add(pos)
    db.commit()
    return pos


def _seed_open_trade(db, symbol, profile_id, entry_time=None, stop=145.0, target=160.0):
    trade = Trade(
        symbol=symbol, direction="LONG", quantity=10,
        entry_price=150.0, status="open", profile=profile_id,
        entry_time=entry_time or datetime.utcnow(),
        stop_price=stop, target_price=target,
    )
    db.add(trade)
    db.commit()
    return trade


def _seed_balance(db, profile_id, cash=100_000.0):
    db.add(Balance(profile=profile_id, cash=cash))
    db.commit()


def _add_analyst_signal(db, symbol, timestamp, signal_data=None):
    if signal_data is None:
        signal_data = {"bias": "LONG", "strength": "strong", "confidence": "high"}
    db.add(AgentMemory(
        agent="analyst", symbol=symbol, key="signal",
        timestamp=timestamp, value=json.dumps(signal_data),
    ))
    db.commit()


def _mock_finnhub(price=150.0):
    fh = MagicMock()
    fh.get_quote.return_value = {"price": price}
    return fh


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDriftingLabelInPortfolio:
    """Verify 'drifting' key appears in get_portfolio_for_profile() output."""

    def test_drifting_true_when_no_signal_after_entry(self):
        """Position with no analyst signal after entry → drifting: True."""
        engine = _make_engine()
        db = _make_session(engine)
        profile_id = "moderate"
        _seed_balance(db, profile_id)

        entry_time = datetime.utcnow() - timedelta(hours=2)
        _seed_position(db, "AMD", profile_id)
        _seed_open_trade(db, "AMD", profile_id, entry_time=entry_time)

        result = get_portfolio_for_profile(db, _mock_finnhub(), profile_id)

        assert len(result["positions"]) == 1
        assert "drifting" in result["positions"][0]
        assert result["positions"][0]["drifting"] is True
        db.close()

    def test_drifting_false_when_signal_after_entry(self):
        """Position with analyst signal after entry → drifting: False."""
        engine = _make_engine()
        db = _make_session(engine)
        profile_id = "moderate"
        _seed_balance(db, profile_id)

        entry_time = datetime.utcnow() - timedelta(hours=2)
        _seed_position(db, "AMD", profile_id)
        _seed_open_trade(db, "AMD", profile_id, entry_time=entry_time)
        _add_analyst_signal(db, "AMD", entry_time + timedelta(hours=1))

        result = get_portfolio_for_profile(db, _mock_finnhub(), profile_id)

        assert len(result["positions"]) == 1
        assert result["positions"][0]["drifting"] is False
        db.close()

    def test_drifting_true_when_no_open_trade(self):
        """Position with no matching open trade → drifting: True (conservative)."""
        engine = _make_engine()
        db = _make_session(engine)
        profile_id = "moderate"
        _seed_balance(db, profile_id)

        # Position exists but no open trade record
        _seed_position(db, "AMD", profile_id)

        result = get_portfolio_for_profile(db, _mock_finnhub(), profile_id)

        assert len(result["positions"]) == 1
        assert result["positions"][0]["drifting"] is True
        db.close()

    def test_multiple_positions_mixed_drifting(self):
        """Two positions: one drifting, one not."""
        engine = _make_engine()
        db = _make_session(engine)
        profile_id = "moderate"
        _seed_balance(db, profile_id)

        entry_time = datetime.utcnow() - timedelta(hours=2)

        # AMD: no signal after entry → drifting
        _seed_position(db, "AMD", profile_id)
        _seed_open_trade(db, "AMD", profile_id, entry_time=entry_time)

        # AAPL: signal after entry → not drifting
        _seed_position(db, "AAPL", profile_id)
        _seed_open_trade(db, "AAPL", profile_id, entry_time=entry_time)
        _add_analyst_signal(db, "AAPL", entry_time + timedelta(hours=1))

        result = get_portfolio_for_profile(db, _mock_finnhub(), profile_id)

        assert len(result["positions"]) == 2
        pos_by_symbol = {p["symbol"]: p for p in result["positions"]}
        assert pos_by_symbol["AMD"]["drifting"] is True
        assert pos_by_symbol["AAPL"]["drifting"] is False
        db.close()
