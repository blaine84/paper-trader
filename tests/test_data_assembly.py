"""Smoke tests for assemble_morning_data and assemble_afternoon_data."""

import json
from datetime import date

import pytest
from sqlalchemy import create_engine

from db.schema import Base, AgentMemory, get_session
from utils.slack_notifier import assemble_morning_data, assemble_afternoon_data


@pytest.fixture
def engine():
    """In-memory SQLite engine with schema created."""
    eng = create_engine("sqlite://")
    Base.metadata.create_all(eng)
    return eng


class TestAssembleMorningData:
    def test_empty_db_returns_all_unavailable(self, engine):
        result = assemble_morning_data(engine)
        assert result["regime"] == "Data unavailable"
        assert result["strategies"]["recommended"] == "Data unavailable"
        assert result["strategies"]["avoid"] == "Data unavailable"
        assert result["scout_picks"] == "Data unavailable"
        assert result["analyst_signals"] == "Data unavailable"
        assert result["date"] == date.today().isoformat()

    def test_seeded_data_assembled_correctly(self, engine):
        db = get_session(engine)
        db.add(AgentMemory(
            agent="quant_researcher", key="regime",
            value=json.dumps("risk_on"),
        ))
        db.add(AgentMemory(
            agent="quant_researcher", key="strategy_recommendation",
            value=json.dumps({
                "recommended_strategies": ["gap_and_go", "breakout"],
                "strategies_to_avoid": ["momentum_fade"],
            }),
        ))
        db.add(AgentMemory(
            agent="scout", key="daily_picks",
            value=json.dumps({
                "date": "2025-01-01",
                "picks": [{"symbol": "AAPL", "reasoning": "Strong gap up"}],
            }),
        ))
        db.add(AgentMemory(
            agent="analyst", symbol="SPY", key="signal",
            value=json.dumps({
                "symbol": "SPY", "signal": "BULLISH", "strength": "strong",
            }),
        ))
        db.commit()
        db.close()

        result = assemble_morning_data(engine)
        assert result["regime"] == "risk_on"
        assert result["strategies"]["recommended"] == ["gap_and_go", "breakout"]
        assert result["strategies"]["avoid"] == ["momentum_fade"]
        assert len(result["scout_picks"]) == 1
        assert result["scout_picks"][0]["symbol"] == "AAPL"
        assert len(result["analyst_signals"]) == 1
        assert result["analyst_signals"][0]["symbol"] == "SPY"

    def test_bad_json_regime_uses_raw_value(self, engine):
        db = get_session(engine)
        db.add(AgentMemory(
            agent="quant_researcher", key="regime",
            value="not valid json",
        ))
        db.commit()
        db.close()

        result = assemble_morning_data(engine)
        assert result["regime"] == "not valid json"

    def test_multiple_analyst_signals_deduped_by_symbol(self, engine):
        db = get_session(engine)
        # Two signals for SPY — only the latest should be kept
        db.add(AgentMemory(
            agent="analyst", symbol="SPY", key="signal",
            value=json.dumps({"symbol": "SPY", "signal": "BULLISH", "strength": "strong"}),
        ))
        db.add(AgentMemory(
            agent="analyst", symbol="AAPL", key="signal",
            value=json.dumps({"symbol": "AAPL", "signal": "BEARISH", "strength": "moderate"}),
        ))
        db.commit()
        db.close()

        result = assemble_morning_data(engine)
        assert isinstance(result["analyst_signals"], list)
        assert len(result["analyst_signals"]) == 2
        symbols = {s["symbol"] for s in result["analyst_signals"]}
        assert symbols == {"SPY", "AAPL"}

    def test_partial_data_fills_unavailable(self, engine):
        """Only regime is present — other fields should be Data unavailable."""
        db = get_session(engine)
        db.add(AgentMemory(
            agent="quant_researcher", key="regime",
            value=json.dumps({"regime": "risk_off"}),
        ))
        db.commit()
        db.close()

        result = assemble_morning_data(engine)
        assert result["regime"] == "risk_off"
        assert result["scout_picks"] == "Data unavailable"
        assert result["analyst_signals"] == "Data unavailable"


class TestAssembleAfternoonData:
    def test_empty_db_returns_fallback(self, engine):
        result = assemble_afternoon_data(engine)
        assert result["missing"] is True
        assert result["date"] == date.today().isoformat()

    def test_seeded_review_returned(self, engine):
        today = date.today().isoformat()
        review = {
            "date": today,
            "day_classification": "modest_win",
            "executive_summary": "Good day",
            "trade_performance": {
                "total_pnl": 150.0, "total_pnl_pct": 1.5,
                "wins": 3, "losses": 1, "total_trades": 4,
                "per_profile": {},
            },
            "what_worked": ["Gap trades"],
            "what_failed": ["Late entries"],
            "highest_leverage_fix": "Enter earlier",
            "tomorrows_focus": ["Watch AAPL"],
        }
        db = get_session(engine)
        db.add(AgentMemory(
            agent="daily_review", symbol=today, key="daily_review",
            value=json.dumps(review),
        ))
        db.commit()
        db.close()

        result = assemble_afternoon_data(engine)
        assert result["day_classification"] == "modest_win"
        assert result["trade_performance"]["total_pnl"] == 150.0

    def test_bad_json_returns_fallback(self, engine):
        today = date.today().isoformat()
        db = get_session(engine)
        db.add(AgentMemory(
            agent="daily_review", symbol=today, key="daily_review",
            value="not valid json{{{",
        ))
        db.commit()
        db.close()

        result = assemble_afternoon_data(engine)
        assert result["missing"] is True
        assert result["date"] == today

    def test_wrong_date_returns_fallback(self, engine):
        """A review for a different date should not be returned."""
        db = get_session(engine)
        db.add(AgentMemory(
            agent="daily_review", symbol="1999-01-01", key="daily_review",
            value=json.dumps({"day_classification": "bad_day"}),
        ))
        db.commit()
        db.close()

        result = assemble_afternoon_data(engine)
        assert result["missing"] is True
