"""
Integration tests for thesis invalidator evaluation in check_stops_and_targets().
Task 5.3 — verifies that invalidator breaches produce thesis_invalidation triggers
and store them in AgentMemory for PM to read during Reversal/Close Review.
"""

import json
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, Trade, Position, AgentMemory, Balance, get_session
from models.case import Case  # noqa: F401 — registers with Base
from agents.price_monitor import check_stops_and_targets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine():
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    return engine


def _add_open_trade(db, symbol="AMD", direction="LONG", entry_price=165.0,
                    stop_price=160.0, target_price=175.0, profile="moderate",
                    invalidators=None):
    """Insert an open trade with optional invalidators JSON."""
    trade = Trade(
        symbol=symbol,
        direction=direction,
        quantity=50,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        status="open",
        profile=profile,
        entry_time=datetime(2025, 1, 15, 10, 0, 0),
        invalidators=json.dumps(invalidators) if invalidators else None,
    )
    db.add(trade)
    db.commit()
    return trade


def _mock_quotes(quotes_dict):
    """Patch get_batch_quotes to return a fixed dict."""
    return patch("agents.price_monitor.get_batch_quotes", return_value=quotes_dict)


def _mock_profit_manager():
    """Patch profit manager run to no-op."""
    return patch("agents.profit_manager.run", return_value=None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInvalidatorIntegration:
    """Thesis invalidation triggers emitted from check_stops_and_targets."""

    def test_thesis_invalidation_trigger_emitted(self):
        """When an invalidator is breached, a thesis_invalidation trigger is returned."""
        engine = _make_engine()
        db = get_session(engine)
        _add_open_trade(db, symbol="AMD", entry_price=165.0,
                        stop_price=160.0, target_price=175.0,
                        invalidators=[
                            {"type": "price_below_level", "reference": "162.50",
                             "confirmation": "tick", "lookback_bars": 0}
                        ])
        db.close()

        with _mock_quotes({"AMD": 161.0}), _mock_profit_manager():
            triggers = check_stops_and_targets(engine)

        invalidation_triggers = [t for t in triggers if t["type"] == "thesis_invalidation"]
        assert len(invalidation_triggers) == 1
        t = invalidation_triggers[0]
        assert t["symbol"] == "AMD"
        assert t["profile"] == "moderate"
        assert t["price"] == 161.0
        assert t["invalidator"]["type"] == "price_below_level"
        assert t["invalidator"]["reference"] == "162.50"
        assert "timestamp" in t
        assert "trade_id" in t

    def test_thesis_invalidation_stored_in_agent_memory(self):
        """Breached invalidators are persisted to AgentMemory for PM to read."""
        engine = _make_engine()
        db = get_session(engine)
        _add_open_trade(db, symbol="AMD", entry_price=165.0,
                        stop_price=160.0, target_price=175.0,
                        invalidators=[
                            {"type": "price_below_level", "reference": "162.50",
                             "confirmation": "tick", "lookback_bars": 0}
                        ])
        db.close()

        with _mock_quotes({"AMD": 161.0}), _mock_profit_manager():
            check_stops_and_targets(engine)

        # Check AgentMemory for the stored trigger
        db = get_session(engine)
        mem = db.query(AgentMemory).filter_by(
            agent="price_monitor", key="thesis_invalidation", symbol="AMD"
        ).first()
        assert mem is not None
        stored = json.loads(mem.value)
        assert stored["type"] == "thesis_invalidation"
        assert stored["symbol"] == "AMD"
        assert stored["invalidator"]["type"] == "price_below_level"
        db.close()

    def test_no_trigger_when_invalidator_not_breached(self):
        """When price doesn't breach the invalidator, no thesis_invalidation trigger."""
        engine = _make_engine()
        db = get_session(engine)
        _add_open_trade(db, symbol="AMD", entry_price=165.0,
                        stop_price=160.0, target_price=175.0,
                        invalidators=[
                            {"type": "price_below_level", "reference": "162.50",
                             "confirmation": "tick", "lookback_bars": 0}
                        ])
        db.close()

        # Price is above the invalidator reference — no breach
        with _mock_quotes({"AMD": 164.0}), _mock_profit_manager():
            triggers = check_stops_and_targets(engine)

        invalidation_triggers = [t for t in triggers if t["type"] == "thesis_invalidation"]
        assert len(invalidation_triggers) == 0

    def test_no_trigger_when_trade_has_no_invalidators(self):
        """Trades without invalidators produce no thesis_invalidation triggers."""
        engine = _make_engine()
        db = get_session(engine)
        _add_open_trade(db, symbol="AMD", entry_price=165.0,
                        stop_price=160.0, target_price=175.0,
                        invalidators=None)
        db.close()

        with _mock_quotes({"AMD": 161.0}), _mock_profit_manager():
            triggers = check_stops_and_targets(engine)

        invalidation_triggers = [t for t in triggers if t["type"] == "thesis_invalidation"]
        assert len(invalidation_triggers) == 0

    def test_invalidation_alongside_stop_loss(self):
        """Both stop_loss and thesis_invalidation can fire for the same trade."""
        engine = _make_engine()
        db = get_session(engine)
        _add_open_trade(db, symbol="AMD", entry_price=165.0,
                        stop_price=160.0, target_price=175.0,
                        invalidators=[
                            {"type": "price_below_level", "reference": "162.50",
                             "confirmation": "tick", "lookback_bars": 0}
                        ])
        db.close()

        # Price below both stop (with buffer) and invalidator reference
        with _mock_quotes({"AMD": 159.0}), _mock_profit_manager():
            triggers = check_stops_and_targets(engine)

        types = [t["type"] for t in triggers]
        assert "stop_loss" in types
        assert "thesis_invalidation" in types

    def test_multiple_invalidators_or_logic(self):
        """If multiple invalidators are defined, any single breach produces a trigger."""
        engine = _make_engine()
        db = get_session(engine)
        _add_open_trade(db, symbol="AMD", entry_price=165.0,
                        stop_price=155.0, target_price=175.0,
                        invalidators=[
                            {"type": "price_below_level", "reference": "162.50",
                             "confirmation": "tick", "lookback_bars": 0},
                            {"type": "price_below_level", "reference": "158.00",
                             "confirmation": "tick", "lookback_bars": 0},
                        ])
        db.close()

        # Price 160 breaches first invalidator (162.50) but not second (158.00)
        with _mock_quotes({"AMD": 160.0}), _mock_profit_manager():
            triggers = check_stops_and_targets(engine)

        invalidation_triggers = [t for t in triggers if t["type"] == "thesis_invalidation"]
        assert len(invalidation_triggers) == 1
        assert invalidation_triggers[0]["invalidator"]["reference"] == "162.50"

    def test_trigger_schema_matches_design(self):
        """Verify the thesis_invalidation trigger matches the design doc schema."""
        engine = _make_engine()
        db = get_session(engine)
        _add_open_trade(db, symbol="AMD", entry_price=165.0,
                        stop_price=160.0, target_price=175.0,
                        profile="moderate",
                        invalidators=[
                            {"type": "price_below_level", "reference": "VWAP",
                             "confirmation": "5m_close", "lookback_bars": 1}
                        ])
        db.close()

        # For tick-based test, use a simple numeric invalidator
        engine2 = _make_engine()
        db2 = get_session(engine2)
        _add_open_trade(db2, symbol="AMD", entry_price=165.0,
                        stop_price=160.0, target_price=175.0,
                        profile="moderate",
                        invalidators=[
                            {"type": "price_below_level", "reference": "162.50",
                             "confirmation": "tick", "lookback_bars": 0}
                        ])
        db2.close()

        with _mock_quotes({"AMD": 161.80}), _mock_profit_manager():
            triggers = check_stops_and_targets(engine2)

        inv_triggers = [t for t in triggers if t["type"] == "thesis_invalidation"]
        assert len(inv_triggers) == 1
        t = inv_triggers[0]

        # Verify all required fields from design doc schema
        assert t["type"] == "thesis_invalidation"
        assert t["symbol"] == "AMD"
        assert t["profile"] == "moderate"
        assert isinstance(t["trade_id"], int)
        assert isinstance(t["price"], float)
        assert isinstance(t["invalidator"], dict)
        assert "timestamp" in t
        # Invalidator object has required fields
        inv = t["invalidator"]
        assert "type" in inv
        assert "reference" in inv
        assert "confirmation" in inv
        assert "lookback_bars" in inv
