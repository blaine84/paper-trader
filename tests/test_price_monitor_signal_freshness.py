"""Regression tests for stale analyst signal entry triggers."""

import json
from datetime import datetime, timedelta
from unittest.mock import patch

from sqlalchemy import create_engine

from agents.price_monitor import check_entry_triggers
from db.schema import Base, AgentMemory, get_session


def _engine():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    return eng


def _insert_signal(engine, *, symbol="CLF", timestamp=None):
    db = get_session(engine)
    db.add(AgentMemory(
        agent="analyst",
        symbol=symbol,
        key="signal",
        timestamp=timestamp or datetime.utcnow(),
        value=json.dumps({
            "symbol": symbol,
            "signal": "SHORT",
            "strength": "strong",
            "confidence": "high",
            "setup_type": "news_catalyst",
            "key_levels": {"support": 10.20, "resistance": 11.0},
        }),
    ))
    db.commit()
    db.close()


def test_stale_analyst_signal_does_not_trigger_entry():
    engine = _engine()
    _insert_signal(engine, timestamp=datetime.utcnow() - timedelta(days=23))

    with patch("agents.price_monitor.get_batch_quotes", return_value={"CLF": 10.15}):
        triggers = check_entry_triggers(engine)

    assert triggers == []


def test_fresh_analyst_signal_can_trigger_entry():
    engine = _engine()
    _insert_signal(engine, timestamp=datetime.utcnow() - timedelta(minutes=5))

    with patch("agents.price_monitor.get_batch_quotes", return_value={"CLF": 10.15}):
        triggers = check_entry_triggers(engine)

    assert triggers == [{
        "type": "breakdown",
        "symbol": "CLF",
        "signal": "SHORT",
        "price": 10.15,
        "level": 10.20,
        "level_name": "support",
        "strength": "strong",
        "setup_type": "news_catalyst",
    }]
