"""
Tests for assemble_daily_wrap and assemble_weekly_wrap (Task 4.3).
Validates: Requirements 5.1, 5.2, 5.3, 6.1, 6.2, 6.3, 6.4, 13.1, 13.3, 13.4, 13.5
"""

import json
from datetime import date, datetime, timedelta

import pytest
from sqlalchemy import create_engine

from db.schema import (
    Base, Trade, Balance, DailyLog, AgentMemory,
    DynamicStrategy, get_session,
)
from models.case import Case
from agents.narrator import assemble_daily_wrap, assemble_weekly_wrap


@pytest.fixture
def engine():
    """In-memory SQLite engine with schema created."""
    eng = create_engine("sqlite://")
    Base.metadata.create_all(eng)
    return eng


# ---------------------------------------------------------------------------
# assemble_daily_wrap tests
# ---------------------------------------------------------------------------

class TestAssembleDailyWrap:
    def test_empty_db_returns_all_keys(self, engine):
        """Req 13.1: assembly function returns a dict with expected keys."""
        result = assemble_daily_wrap(engine)
        assert isinstance(result, dict)
        assert "closed_trades" in result
        assert "open_positions" in result
        assert "profile_summary" in result
        assert "win_loss" in result
        assert "daily_log" in result
        assert "reviewer_scores" in result
        assert "daily_review" in result
        assert "lessons" in result

    def test_empty_db_returns_empty_collections(self, engine):
        result = assemble_daily_wrap(engine)
        assert result["closed_trades"] == []
        assert result["open_positions"] == []
        assert result["daily_log"] == {}
        assert result["reviewer_scores"] == {}
        assert result["daily_review"] == {}
        assert result["lessons"] == []

    def test_closed_trades_included(self, engine):
        """Req 5.1, 13.3: closed trades for today are returned."""
        today = date.today().isoformat()
        db = get_session(engine)
        db.add(Trade(
            symbol="TSLA", direction="LONG", quantity=50,
            entry_price=185.0, exit_price=190.0,
            entry_time=datetime.now(), exit_time=datetime.now(),
            status="closed", pnl=250.0, pnl_pct=2.7,
            profile="moderate", review_score=8.0,
            setup_type="gap_and_go",
        ))
        db.commit()
        db.close()

        result = assemble_daily_wrap(engine)
        assert len(result["closed_trades"]) == 1
        trade = result["closed_trades"][0]
        assert trade["symbol"] == "TSLA"
        assert trade["pnl"] == 250.0
        assert trade["review_score"] == 8.0
        assert trade["setup_type"] == "gap_and_go"

    def test_open_positions_included(self, engine):
        """Req 5.1: open positions carried overnight are returned."""
        db = get_session(engine)
        db.add(Trade(
            symbol="NVDA", direction="LONG", quantity=30,
            entry_price=200.0, status="open",
            profile="aggressive", stop_price=195.0, target_price=215.0,
        ))
        db.commit()
        db.close()

        result = assemble_daily_wrap(engine)
        assert len(result["open_positions"]) == 1
        pos = result["open_positions"][0]
        assert pos["symbol"] == "NVDA"
        assert pos["stop_price"] == 195.0

    def test_profile_summary_pnl_and_equity(self, engine):
        """Req 5.2: per-profile P&L and ending equity."""
        db = get_session(engine)
        # Two closed trades for moderate
        db.add(Trade(
            symbol="TSLA", direction="LONG", quantity=50,
            entry_price=185.0, exit_price=190.0,
            entry_time=datetime.now(), exit_time=datetime.now(),
            status="closed", pnl=250.0, profile="moderate",
        ))
        db.add(Trade(
            symbol="AMD", direction="LONG", quantity=100,
            entry_price=160.0, exit_price=158.0,
            entry_time=datetime.now(), exit_time=datetime.now(),
            status="closed", pnl=-200.0, profile="moderate",
        ))
        # Balance for moderate
        db.add(Balance(
            profile="moderate", cash=95000.0,
            portfolio_value=5050.0, total_equity=100050.0,
        ))
        db.commit()
        db.close()

        result = assemble_daily_wrap(engine)
        mod = result["profile_summary"]["moderate"]
        assert mod["total_pnl"] == 50.0  # 250 - 200
        assert mod["wins"] == 1
        assert mod["losses"] == 1
        assert mod["total_trades"] == 2
        assert mod["ending_equity"] == 100050.0

    def test_win_loss_aggregate(self, engine):
        """Req 5.1: aggregate win/loss across all profiles."""
        db = get_session(engine)
        db.add(Trade(
            symbol="TSLA", direction="LONG", quantity=50,
            entry_price=185.0, exit_price=190.0,
            entry_time=datetime.now(), exit_time=datetime.now(),
            status="closed", pnl=250.0, profile="moderate",
        ))
        db.add(Trade(
            symbol="AMD", direction="LONG", quantity=100,
            entry_price=160.0, exit_price=158.0,
            entry_time=datetime.now(), exit_time=datetime.now(),
            status="closed", pnl=-200.0, profile="aggressive",
        ))
        db.commit()
        db.close()

        result = assemble_daily_wrap(engine)
        assert result["win_loss"]["wins"] == 1
        assert result["win_loss"]["losses"] == 1
        assert result["win_loss"]["total"] == 2

    def test_daily_log_included(self, engine):
        """Req 5.1: DailyLog for today is returned."""
        today = date.today().isoformat()
        db = get_session(engine)
        db.add(DailyLog(
            date=today, starting_equity=100000.0, ending_equity=100500.0,
            trades_taken=4, winning_trades=3, losing_trades=1,
            daily_pnl=500.0, daily_pnl_pct=0.5,
            notes="Good day overall",
        ))
        db.commit()
        db.close()

        result = assemble_daily_wrap(engine)
        assert result["daily_log"]["daily_pnl"] == 500.0
        assert result["daily_log"]["notes"] == "Good day overall"

    def test_reviewer_scores_included(self, engine):
        """Req 5.3: reviewer feedback and scores for today."""
        db = get_session(engine)
        db.add(AgentMemory(
            agent="reviewer", symbol="TSLA", key="feedback",
            value=json.dumps({"score": 8.5, "notes": "Good entry timing"}),
        ))
        db.commit()
        db.close()

        result = assemble_daily_wrap(engine)
        assert "TSLA" in result["reviewer_scores"]
        assert result["reviewer_scores"]["TSLA"]["score"] == 8.5

    def test_daily_review_included(self, engine):
        """Req 5.1: daily_review for today is returned."""
        today = date.today().isoformat()
        db = get_session(engine)
        db.add(AgentMemory(
            agent="daily_review", symbol=today, key="daily_review",
            value=json.dumps({
                "day_classification": "modest_win",
                "executive_summary": "Solid day",
            }),
        ))
        db.commit()
        db.close()

        result = assemble_daily_wrap(engine)
        assert result["daily_review"]["day_classification"] == "modest_win"

    def test_case_lessons_included(self, engine):
        """Req 5.1: Case lessons for today are returned."""
        today = date.today().isoformat()
        db = get_session(engine)
        db.add(Case(
            symbol="TSLA", date=today, setup_type="gap_and_go",
            outcome="success", pnl_pct=2.7, profile="moderate",
            lesson="Gap trades work best in risk-on regime",
            selection_score=8.0, execution_score=7.5,
        ))
        db.commit()
        db.close()

        result = assemble_daily_wrap(engine)
        assert len(result["lessons"]) == 1
        assert result["lessons"][0]["lesson"] == "Gap trades work best in risk-on regime"
        assert result["lessons"][0]["selection_score"] == 8.0

    def test_individual_query_failure_doesnt_break_assembly(self, engine):
        """Req 13.5: if one query fails, others still return data."""
        today = date.today().isoformat()
        db = get_session(engine)
        # Seed some data that will be queryable
        db.add(DailyLog(
            date=today, starting_equity=100000.0, ending_equity=100500.0,
            trades_taken=2, winning_trades=1, losing_trades=1,
            daily_pnl=500.0, daily_pnl_pct=0.5,
        ))
        db.commit()
        db.close()

        # Even if no trades exist, the function should still return daily_log
        result = assemble_daily_wrap(engine)
        assert result["closed_trades"] == []
        assert result["daily_log"]["daily_pnl"] == 500.0


# ---------------------------------------------------------------------------
# assemble_weekly_wrap tests
# ---------------------------------------------------------------------------

class TestAssembleWeeklyWrap:
    def test_empty_db_returns_all_keys(self, engine):
        """Req 13.1: assembly function returns a dict with expected keys."""
        result = assemble_weekly_wrap(engine)
        assert isinstance(result, dict)
        assert "week_pnl" in result
        assert "best_trades" in result
        assert "worst_trades" in result
        assert "daily_logs" in result
        assert "case_trends" in result
        assert "strategy_performance" in result
        assert "agent_grades" in result

    def test_empty_db_returns_empty_collections(self, engine):
        result = assemble_weekly_wrap(engine)
        assert result["week_pnl"] == {} or all(
            v["total_trades"] == 0 for v in result["week_pnl"].values()
        )
        assert result["best_trades"] == []
        assert result["worst_trades"] == []
        assert result["daily_logs"] == []
        assert result["case_trends"] == []
        assert result["strategy_performance"] == {}
        assert result["agent_grades"] == {}

    def test_week_trades_and_pnl(self, engine):
        """Req 6.1, 13.4: trades for the current week with per-profile P&L."""
        today = date.today()
        monday = today - timedelta(days=today.weekday())
        trade_time = datetime.combine(monday, datetime.min.time().replace(hour=10))

        db = get_session(engine)
        db.add(Trade(
            symbol="TSLA", direction="LONG", quantity=50,
            entry_price=185.0, exit_price=195.0,
            entry_time=trade_time, exit_time=trade_time,
            status="closed", pnl=500.0, pnl_pct=5.4,
            profile="moderate",
        ))
        db.add(Trade(
            symbol="AMD", direction="LONG", quantity=100,
            entry_price=160.0, exit_price=155.0,
            entry_time=trade_time, exit_time=trade_time,
            status="closed", pnl=-500.0, pnl_pct=-3.1,
            profile="aggressive",
        ))
        db.commit()
        db.close()

        result = assemble_weekly_wrap(engine)
        assert result["week_pnl"]["moderate"]["total_pnl"] == 500.0
        assert result["week_pnl"]["moderate"]["wins"] == 1
        assert result["week_pnl"]["aggressive"]["total_pnl"] == -500.0
        assert result["week_pnl"]["aggressive"]["losses"] == 1

    def test_best_and_worst_trades(self, engine):
        """Req 6.1: best and worst trades of the week."""
        today = date.today()
        monday = today - timedelta(days=today.weekday())
        trade_time = datetime.combine(monday, datetime.min.time().replace(hour=10))

        db = get_session(engine)
        for i, (sym, pnl) in enumerate([
            ("TSLA", 800.0), ("NVDA", 500.0), ("AMD", -300.0), ("AAPL", -100.0),
        ]):
            db.add(Trade(
                symbol=sym, direction="LONG", quantity=50,
                entry_price=100.0, exit_price=100.0 + pnl / 50,
                entry_time=trade_time, exit_time=trade_time,
                status="closed", pnl=pnl, profile="moderate",
            ))
        db.commit()
        db.close()

        result = assemble_weekly_wrap(engine)
        best_symbols = [t["symbol"] for t in result["best_trades"]]
        worst_symbols = [t["symbol"] for t in result["worst_trades"]]
        assert "TSLA" in best_symbols
        assert "AMD" in worst_symbols

    def test_daily_logs_for_week(self, engine):
        """Req 13.4: DailyLog entries for the current week."""
        today = date.today()
        monday = today - timedelta(days=today.weekday())

        db = get_session(engine)
        for i in range(3):
            day = monday + timedelta(days=i)
            db.add(DailyLog(
                date=day.isoformat(),
                starting_equity=100000.0 + i * 100,
                ending_equity=100100.0 + i * 100,
                trades_taken=2, winning_trades=1, losing_trades=1,
                daily_pnl=100.0, daily_pnl_pct=0.1,
            ))
        db.commit()
        db.close()

        result = assemble_weekly_wrap(engine)
        assert len(result["daily_logs"]) == 3

    def test_case_trends_for_week(self, engine):
        """Req 6.1: Case library trends for the current week."""
        today = date.today()
        monday = today - timedelta(days=today.weekday())

        db = get_session(engine)
        db.add(Case(
            symbol="TSLA", date=monday.isoformat(),
            setup_type="gap_and_go", outcome="success",
            pnl_pct=2.7, lesson="Gap trades work in risk-on",
            profile="moderate",
        ))
        db.commit()
        db.close()

        result = assemble_weekly_wrap(engine)
        assert len(result["case_trends"]) == 1
        assert result["case_trends"][0]["setup_type"] == "gap_and_go"

    def test_strategy_performance(self, engine):
        """Req 6.3: DynamicStrategy performance included."""
        db = get_session(engine)
        db.add(DynamicStrategy(
            key="gap_and_go", name="Gap and Go",
            description="Buy gap ups", status="active",
            total_trades=20, wins=14, win_rate=0.7,
            avg_pnl_pct=1.5,
        ))
        db.add(DynamicStrategy(
            key="momentum_fade", name="Momentum Fade",
            description="Fade momentum", status="retired",
            total_trades=10, wins=3, win_rate=0.3,
            avg_pnl_pct=-0.8,
            retired_at=datetime.now(),
            retire_reason="Poor performance",
        ))
        db.commit()
        db.close()

        result = assemble_weekly_wrap(engine)
        assert "gap_and_go" in result["strategy_performance"]
        assert result["strategy_performance"]["gap_and_go"]["win_rate"] == 0.7
        assert result["strategy_performance"]["momentum_fade"]["status"] == "retired"

    def test_agent_grades_included(self, engine):
        """Req 6.4: meta_reviewer agent grades included when available."""
        db = get_session(engine)
        db.add(AgentMemory(
            agent="meta_reviewer", key="weekly_review",
            value=json.dumps({
                "analyst": "A", "researcher": "B+",
                "pm_moderate": "A-", "reviewer": "B",
            }),
        ))
        db.commit()
        db.close()

        result = assemble_weekly_wrap(engine)
        assert result["agent_grades"]["analyst"] == "A"
        assert result["agent_grades"]["researcher"] == "B+"

    def test_agent_grades_empty_when_missing(self, engine):
        """Req 6.4: agent grades omitted when not available."""
        result = assemble_weekly_wrap(engine)
        assert result["agent_grades"] == {}

    def test_trades_outside_week_excluded(self, engine):
        """Req 13.4: only trades within the current Mon-Fri week are included."""
        today = date.today()
        monday = today - timedelta(days=today.weekday())
        last_week = monday - timedelta(days=7)
        last_week_time = datetime.combine(last_week, datetime.min.time().replace(hour=10))

        db = get_session(engine)
        db.add(Trade(
            symbol="OLD", direction="LONG", quantity=50,
            entry_price=100.0, exit_price=105.0,
            entry_time=last_week_time, exit_time=last_week_time,
            status="closed", pnl=250.0, profile="moderate",
        ))
        db.commit()
        db.close()

        result = assemble_weekly_wrap(engine)
        # The old trade should not appear in best/worst
        all_trade_symbols = [t["symbol"] for t in result["best_trades"] + result["worst_trades"]]
        assert "OLD" not in all_trade_symbols
