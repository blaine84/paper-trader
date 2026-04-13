"""
Tests for gather_agent_context, load_previous_review, and gather_cases_today
in agents/daily_review.py.
"""

import json
from datetime import datetime

from sqlalchemy import create_engine

from db.schema import Base, AgentMemory, get_session
from models.case import Case
from agents.daily_review import (
    gather_agent_context,
    load_previous_review,
    gather_cases_today,
)


def _make_engine():
    """Create an in-memory SQLite engine with all tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


# ---------------------------------------------------------------------------
# gather_agent_context tests
# ---------------------------------------------------------------------------


class TestGatherAgentContext:
    def test_all_sources_present(self):
        engine = _make_engine()
        db = get_session(engine)
        db.add(AgentMemory(agent="researcher", key="market_context", value="risk_on"))
        db.add(AgentMemory(agent="reviewer", key="selection_feedback", value="good picks"))
        db.add(AgentMemory(agent="reviewer", key="execution_feedback", value='{"moderate": "ok"}'))
        db.add(AgentMemory(agent="analyst", symbol="AAPL", key="signal", value='{"bias": "LONG"}'))
        db.commit()
        db.close()

        ctx = gather_agent_context(engine)
        assert ctx["market_context"] == "risk_on"
        assert ctx["selection_feedback"] == "good picks"
        assert ctx["execution_feedback"] == {"moderate": "ok"}
        assert ctx["analyst_signals"] == {"AAPL": {"bias": "LONG"}}
        assert ctx["missing_sources"] == []

    def test_all_sources_missing(self):
        engine = _make_engine()
        ctx = gather_agent_context(engine)
        assert ctx["market_context"] is None
        assert ctx["selection_feedback"] is None
        assert ctx["execution_feedback"] is None
        assert ctx["analyst_signals"] is None
        assert "researcher_context" in ctx["missing_sources"]
        assert "selection_feedback" in ctx["missing_sources"]
        assert "execution_feedback" in ctx["missing_sources"]
        assert "analyst_signals" in ctx["missing_sources"]

    def test_partial_sources(self):
        engine = _make_engine()
        db = get_session(engine)
        db.add(AgentMemory(agent="researcher", key="market_context", value="mixed"))
        db.commit()
        db.close()

        ctx = gather_agent_context(engine)
        assert ctx["market_context"] == "mixed"
        assert ctx["selection_feedback"] is None
        assert "selection_feedback" in ctx["missing_sources"]
        assert "researcher_context" not in ctx["missing_sources"]

    def test_latest_market_context_returned(self):
        engine = _make_engine()
        db = get_session(engine)
        db.add(AgentMemory(
            agent="researcher", key="market_context", value="old",
            timestamp=datetime(2025, 1, 1, 10, 0, 0),
        ))
        db.add(AgentMemory(
            agent="researcher", key="market_context", value="latest",
            timestamp=datetime(2025, 1, 2, 10, 0, 0),
        ))
        db.commit()
        db.close()

        ctx = gather_agent_context(engine)
        assert ctx["market_context"] == "latest"

    def test_execution_feedback_non_json_fallback(self):
        engine = _make_engine()
        db = get_session(engine)
        db.add(AgentMemory(agent="reviewer", key="execution_feedback", value="plain text"))
        db.commit()
        db.close()

        ctx = gather_agent_context(engine)
        assert ctx["execution_feedback"] == "plain text"

    def test_analyst_signals_multiple_symbols(self):
        engine = _make_engine()
        db = get_session(engine)
        db.add(AgentMemory(
            agent="analyst", symbol="AAPL", key="signal",
            value='{"bias": "LONG"}',
            timestamp=datetime(2025, 1, 1, 10, 0, 0),
        ))
        db.add(AgentMemory(
            agent="analyst", symbol="TSLA", key="signal",
            value='{"bias": "SHORT"}',
            timestamp=datetime(2025, 1, 1, 11, 0, 0),
        ))
        db.commit()
        db.close()

        ctx = gather_agent_context(engine)
        assert "AAPL" in ctx["analyst_signals"]
        assert "TSLA" in ctx["analyst_signals"]
        assert ctx["analyst_signals"]["TSLA"]["bias"] == "SHORT"

    def test_analyst_signals_latest_per_symbol(self):
        """When multiple signals exist for the same symbol, only the latest is kept."""
        engine = _make_engine()
        db = get_session(engine)
        db.add(AgentMemory(
            agent="analyst", symbol="AAPL", key="signal",
            value='{"bias": "SHORT"}',
            timestamp=datetime(2025, 1, 1, 9, 0, 0),
        ))
        db.add(AgentMemory(
            agent="analyst", symbol="AAPL", key="signal",
            value='{"bias": "LONG"}',
            timestamp=datetime(2025, 1, 1, 14, 0, 0),
        ))
        db.commit()
        db.close()

        ctx = gather_agent_context(engine)
        # Ordered desc by timestamp, so the first seen for AAPL is the latest
        assert ctx["analyst_signals"]["AAPL"]["bias"] == "LONG"


# ---------------------------------------------------------------------------
# load_previous_review tests
# ---------------------------------------------------------------------------


class TestLoadPreviousReview:
    def test_returns_previous_review(self):
        engine = _make_engine()
        db = get_session(engine)
        review = {"date": "2025-01-01", "market_summary": "Good day"}
        db.add(AgentMemory(
            agent="daily_review", symbol="2025-01-01", key="daily_review",
            value=json.dumps(review),
        ))
        db.commit()
        db.close()

        result = load_previous_review(engine, "2025-01-02")
        assert result is not None
        assert result["date"] == "2025-01-01"

    def test_returns_none_when_no_previous(self):
        engine = _make_engine()
        result = load_previous_review(engine, "2025-01-01")
        assert result is None

    def test_skips_same_date(self):
        engine = _make_engine()
        db = get_session(engine)
        db.add(AgentMemory(
            agent="daily_review", symbol="2025-01-02", key="daily_review",
            value=json.dumps({"date": "2025-01-02"}),
        ))
        db.commit()
        db.close()

        result = load_previous_review(engine, "2025-01-02")
        assert result is None

    def test_returns_most_recent_before_today(self):
        engine = _make_engine()
        db = get_session(engine)
        db.add(AgentMemory(
            agent="daily_review", symbol="2025-01-01", key="daily_review",
            value=json.dumps({"date": "2025-01-01"}),
        ))
        db.add(AgentMemory(
            agent="daily_review", symbol="2025-01-03", key="daily_review",
            value=json.dumps({"date": "2025-01-03"}),
        ))
        db.commit()
        db.close()

        result = load_previous_review(engine, "2025-01-05")
        assert result["date"] == "2025-01-03"

    def test_handles_invalid_json_gracefully(self):
        engine = _make_engine()
        db = get_session(engine)
        db.add(AgentMemory(
            agent="daily_review", symbol="2025-01-01", key="daily_review",
            value="not valid json {{{",
        ))
        db.commit()
        db.close()

        result = load_previous_review(engine, "2025-01-02")
        assert result is None


# ---------------------------------------------------------------------------
# gather_cases_today tests
# ---------------------------------------------------------------------------


class TestGatherCasesToday:
    def test_returns_cases_for_today(self):
        engine = _make_engine()
        db = get_session(engine)
        db.add(Case(
            symbol="AAPL", date="2025-01-02", outcome="success",
            lesson="Gap and go works above resistance",
            setup_type="gap_and_go", pnl_pct=2.1,
        ))
        db.commit()
        db.close()

        cases = gather_cases_today(engine, "2025-01-02")
        assert len(cases) == 1
        assert cases[0]["symbol"] == "AAPL"
        assert cases[0]["setup_type"] == "gap_and_go"
        assert cases[0]["pnl_pct"] == 2.1

    def test_returns_empty_when_no_cases(self):
        engine = _make_engine()
        cases = gather_cases_today(engine, "2025-01-02")
        assert cases == []

    def test_filters_by_date(self):
        engine = _make_engine()
        db = get_session(engine)
        db.add(Case(symbol="AAPL", date="2025-01-01", outcome="success", lesson="old"))
        db.add(Case(symbol="TSLA", date="2025-01-02", outcome="failure", lesson="today"))
        db.commit()
        db.close()

        cases = gather_cases_today(engine, "2025-01-02")
        assert len(cases) == 1
        assert cases[0]["symbol"] == "TSLA"

    def test_multiple_cases_same_day(self):
        engine = _make_engine()
        db = get_session(engine)
        db.add(Case(symbol="AAPL", date="2025-01-02", outcome="success", lesson="l1"))
        db.add(Case(symbol="TSLA", date="2025-01-02", outcome="failure", lesson="l2"))
        db.commit()
        db.close()

        cases = gather_cases_today(engine, "2025-01-02")
        assert len(cases) == 2
        symbols = {c["symbol"] for c in cases}
        assert symbols == {"AAPL", "TSLA"}
