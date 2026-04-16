"""
Unit tests for detect_drifting() in agents/portfolio_manager.py.

Tests verify that DRIFTING state is correctly computed by comparing
the trade's entry_time against analyst signal timestamps in AgentMemory.

Requirements: 3.1, 3.4
"""

import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, AgentMemory, Trade
from models.case import Case  # noqa: F401 — registers with Base
from agents.portfolio_manager import detect_drifting


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine():
    """Create an in-memory SQLite engine with all tables."""
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    return engine


def _make_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()


def _make_trade(symbol: str = "AMD", entry_time: datetime = None) -> Trade:
    """Create a Trade object (not persisted — just needs symbol and entry_time)."""
    trade = Trade(
        symbol=symbol,
        direction="LONG",
        quantity=10,
        entry_price=150.0,
        status="open",
        profile="moderate",
        entry_time=entry_time or datetime.utcnow(),
    )
    return trade


def _add_analyst_signal(db, symbol: str, timestamp: datetime, signal_data: dict = None):
    """Insert an analyst signal into AgentMemory with a specific timestamp."""
    if signal_data is None:
        signal_data = {"bias": "LONG", "strength": "strong", "confidence": "high"}
    db.add(AgentMemory(
        agent="analyst",
        symbol=symbol,
        key="signal",
        timestamp=timestamp,
        value=json.dumps(signal_data),
    ))
    db.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDetectDrifting:
    """Tests for detect_drifting(db, trade)."""

    def test_no_signals_at_all_returns_true(self):
        """No analyst signals for the symbol → drifting."""
        engine = _make_engine()
        db = _make_session(engine)
        trade = _make_trade(symbol="AMD", entry_time=datetime.utcnow())

        assert detect_drifting(db, trade) is True
        db.close()

    def test_signal_before_entry_returns_true(self):
        """Signal exists but before entry_time → drifting."""
        engine = _make_engine()
        db = _make_session(engine)
        entry_time = datetime.utcnow()
        trade = _make_trade(symbol="AMD", entry_time=entry_time)

        # Signal 1 hour before entry
        _add_analyst_signal(db, "AMD", entry_time - timedelta(hours=1))

        assert detect_drifting(db, trade) is True
        db.close()

    def test_signal_at_exact_entry_time_returns_true(self):
        """Signal at exactly entry_time → drifting (must be strictly after)."""
        engine = _make_engine()
        db = _make_session(engine)
        entry_time = datetime(2025, 1, 15, 14, 30, 0)
        trade = _make_trade(symbol="AMD", entry_time=entry_time)

        _add_analyst_signal(db, "AMD", entry_time)

        assert detect_drifting(db, trade) is True
        db.close()

    def test_signal_after_entry_returns_false(self):
        """Signal after entry_time → not drifting."""
        engine = _make_engine()
        db = _make_session(engine)
        entry_time = datetime.utcnow() - timedelta(hours=2)
        trade = _make_trade(symbol="AMD", entry_time=entry_time)

        # Signal 1 hour after entry
        _add_analyst_signal(db, "AMD", entry_time + timedelta(hours=1))

        assert detect_drifting(db, trade) is False
        db.close()

    def test_multiple_signals_latest_after_entry_returns_false(self):
        """Multiple signals — latest one is after entry → not drifting."""
        engine = _make_engine()
        db = _make_session(engine)
        entry_time = datetime.utcnow() - timedelta(hours=3)
        trade = _make_trade(symbol="AMD", entry_time=entry_time)

        # Signal before entry
        _add_analyst_signal(db, "AMD", entry_time - timedelta(hours=1))
        # Signal after entry
        _add_analyst_signal(db, "AMD", entry_time + timedelta(hours=1))

        assert detect_drifting(db, trade) is False
        db.close()

    def test_multiple_signals_all_before_entry_returns_true(self):
        """Multiple signals — all before entry → drifting."""
        engine = _make_engine()
        db = _make_session(engine)
        entry_time = datetime.utcnow()
        trade = _make_trade(symbol="AMD", entry_time=entry_time)

        _add_analyst_signal(db, "AMD", entry_time - timedelta(hours=2))
        _add_analyst_signal(db, "AMD", entry_time - timedelta(hours=1))

        assert detect_drifting(db, trade) is True
        db.close()

    def test_signal_for_different_symbol_ignored(self):
        """Signal for a different symbol doesn't affect drifting detection."""
        engine = _make_engine()
        db = _make_session(engine)
        entry_time = datetime.utcnow() - timedelta(hours=1)
        trade = _make_trade(symbol="AMD", entry_time=entry_time)

        # Signal after entry but for AAPL, not AMD
        _add_analyst_signal(db, "AAPL", entry_time + timedelta(minutes=30))

        assert detect_drifting(db, trade) is True
        db.close()

    def test_no_entry_time_returns_true(self):
        """Trade with no entry_time → drifting (conservative default)."""
        engine = _make_engine()
        db = _make_session(engine)
        trade = _make_trade(symbol="AMD", entry_time=None)
        trade.entry_time = None  # Explicitly set to None

        assert detect_drifting(db, trade) is True
        db.close()

    def test_new_signal_removes_drifting(self):
        """
        Requirement 3.4: When a new signal arrives after entry,
        drifting state is removed.
        """
        engine = _make_engine()
        db = _make_session(engine)
        entry_time = datetime.utcnow() - timedelta(hours=2)
        trade = _make_trade(symbol="AMD", entry_time=entry_time)

        # Initially drifting — no signals after entry
        assert detect_drifting(db, trade) is True

        # New signal arrives after entry
        _add_analyst_signal(db, "AMD", datetime.utcnow())

        # No longer drifting
        assert detect_drifting(db, trade) is False
        db.close()
