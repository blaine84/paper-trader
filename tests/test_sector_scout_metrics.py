"""Tests for utils/sector_scout_metrics.py — success metrics tracking and reporting."""

import json
from datetime import date, datetime, timezone

import pytest
from sqlalchemy import create_engine

from db.schema import AgentMemory, Base, Trade, get_session
from utils.sector_scout_metrics import compute_daily_metrics, format_metrics_for_review


@pytest.fixture
def engine():
    """Create an in-memory SQLite engine with schema."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def today():
    return date.today().isoformat()


def _insert_expanded_watchlist(engine, target_date: str, symbols: list[str]):
    """Helper to insert an expanded watchlist record."""
    db = get_session(engine)
    picks = [{"symbol": s, "source_candidate_score": 50.0 + i} for i, s in enumerate(symbols)]
    data = {
        "date": target_date,
        "run_type": "premarket",
        "picks": picks,
        "symbols": symbols,
        "size": len(symbols),
    }
    db.add(AgentMemory(
        agent="sector_scout",
        symbol=None,
        key=f"expanded_watchlist:{target_date}",
        value=json.dumps(data),
    ))
    db.commit()
    db.close()


def _insert_run_summary(engine, target_date: str, run_type: str, reason_counts: dict, symbols: list[str]):
    """Helper to insert a run summary record."""
    db = get_session(engine)
    summary = {
        "run_type": run_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sectors_scanned": 3,
        "total_candidates_evaluated": 20,
        "hard_gate_rejections": 5,
        "finalists_count": 8,
        "chief_scout_picks": [{"symbol": s} for s in symbols],
        "fallback_used": False,
        "expanded_watchlist_symbols": symbols,
        "expanded_watchlist_size": len(symbols),
        "reason_counts": reason_counts,
        "budget_hits": [],
        "duration_seconds": 12.5,
    }
    db.add(AgentMemory(
        agent="sector_scout",
        symbol=None,
        key=f"run_summary:{target_date}:{run_type}",
        value=json.dumps(summary),
    ))
    db.commit()
    db.close()


def _insert_candidate_row(engine, target_date: str, run_type: str, symbol: str, penalties: list[dict], reason_codes: list[str]):
    """Helper to insert a candidate row record."""
    db = get_session(engine)
    candidate = {
        "symbol": symbol,
        "sector": "ai_semi",
        "sector_name": "AI / Semiconductors",
        "move_pct": 3.5,
        "scout_score": 65.0,
        "hard_gate_passed": True,
        "reason_codes": reason_codes,
        "penalties_applied": penalties,
    }
    db.add(AgentMemory(
        agent="sector_scout",
        symbol=symbol,
        key=f"candidate_row:{target_date}:{run_type}:{symbol}",
        value=json.dumps(candidate),
    ))
    db.commit()
    db.close()


def _insert_analyst_signal(engine, symbol: str, signal: str):
    """Helper to insert an analyst signal."""
    db = get_session(engine)
    db.add(AgentMemory(
        agent="analyst",
        symbol=symbol,
        key="signal",
        value=json.dumps({"signal": signal, "strength": "strong"}),
    ))
    db.commit()
    db.close()


def _insert_trade(engine, symbol: str, pnl: float, pnl_pct: float, status: str = "closed"):
    """Helper to insert a trade."""
    db = get_session(engine)
    db.add(Trade(
        symbol=symbol,
        direction="LONG",
        quantity=100,
        entry_price=50.0,
        exit_price=55.0 if status == "closed" else None,
        entry_time=datetime.now(),
        exit_time=datetime.now() if status == "closed" else None,
        status=status,
        pnl=pnl,
        pnl_pct=pnl_pct,
    ))
    db.commit()
    db.close()


class TestComputeDailyMetrics:
    """Tests for compute_daily_metrics()."""

    def test_empty_database_returns_zeros(self, engine, today):
        """When no data exists, all metrics should be zero/empty."""
        metrics = compute_daily_metrics(engine, today)

        assert metrics["date"] == today
        assert metrics["expanded_candidates_surfaced"] == 0
        assert metrics["pct_reaching_analyst_long_short"] == 0.0
        assert metrics["pct_reaching_pm_eligible"] == 0.0
        assert metrics["executed_trade_outcomes"] == []
        assert metrics["top_rejection_reason_codes"] == []
        assert metrics["top_penalty_reason_codes"] == []
        assert metrics["follow_through_rejected"] == []

    def test_counts_expanded_candidates(self, engine, today):
        """Should count the number of expanded candidates surfaced."""
        _insert_expanded_watchlist(engine, today, ["AVGO", "SMCI", "ARM"])

        metrics = compute_daily_metrics(engine, today)
        assert metrics["expanded_candidates_surfaced"] == 3

    def test_pct_reaching_analyst_long_short(self, engine, today):
        """Should compute percentage of candidates reaching Analyst LONG/SHORT."""
        _insert_expanded_watchlist(engine, today, ["AVGO", "SMCI", "ARM", "MU"])
        _insert_analyst_signal(engine, "AVGO", "LONG")
        _insert_analyst_signal(engine, "SMCI", "SHORT")
        _insert_analyst_signal(engine, "ARM", "HOLD")
        # MU has no signal

        metrics = compute_daily_metrics(engine, today)
        # 2 out of 4 reached LONG/SHORT
        assert metrics["pct_reaching_analyst_long_short"] == 50.0

    def test_pct_reaching_pm_eligible(self, engine, today):
        """Should compute percentage of candidates reaching PM eligible."""
        _insert_expanded_watchlist(engine, today, ["AVGO", "SMCI", "ARM"])
        # AVGO has a trade (PM acted on it)
        _insert_trade(engine, "AVGO", pnl=100.0, pnl_pct=2.0)

        metrics = compute_daily_metrics(engine, today)
        # 1 out of 3 reached PM eligible
        assert metrics["pct_reaching_pm_eligible"] == pytest.approx(33.3, abs=0.1)

    def test_executed_trade_outcomes(self, engine, today):
        """Should return trade outcomes for expanded candidates."""
        _insert_expanded_watchlist(engine, today, ["AVGO", "SMCI"])
        _insert_trade(engine, "AVGO", pnl=150.0, pnl_pct=3.0, status="closed")

        metrics = compute_daily_metrics(engine, today)
        assert len(metrics["executed_trade_outcomes"]) == 1
        assert metrics["executed_trade_outcomes"][0]["symbol"] == "AVGO"
        assert metrics["executed_trade_outcomes"][0]["pnl_pct"] == 3.0

    def test_top_rejection_reason_codes(self, engine, today):
        """Should aggregate rejection reason codes from run summaries."""
        reason_counts = {
            "price_below_min": 3,
            "spread_too_wide": 2,
            "missing_price": 1,
        }
        _insert_run_summary(engine, today, "premarket", reason_counts, ["AVGO"])

        metrics = compute_daily_metrics(engine, today)
        codes = metrics["top_rejection_reason_codes"]
        assert len(codes) == 3
        assert codes[0]["code"] == "price_below_min"
        assert codes[0]["count"] == 3

    def test_top_penalty_reason_codes(self, engine, today):
        """Should aggregate penalty reason codes from candidate rows."""
        _insert_run_summary(engine, today, "premarket", {}, ["AVGO"])
        _insert_candidate_row(
            engine, today, "premarket", "SMCI",
            penalties=[
                {"type": "stale_news", "deduction": 15.0},
                {"type": "low_rvol", "deduction": 12.0},
            ],
            reason_codes=["stale_news:-15.0", "low_rvol:-12.0"],
        )
        _insert_candidate_row(
            engine, today, "premarket", "ARM",
            penalties=[
                {"type": "stale_news", "deduction": 15.0},
            ],
            reason_codes=["stale_news:-15.0"],
        )

        metrics = compute_daily_metrics(engine, today)
        codes = metrics["top_penalty_reason_codes"]
        assert len(codes) >= 1
        # stale_news should be the top penalty code (count=2)
        assert codes[0]["code"] == "stale_news"
        assert codes[0]["count"] == 2

    def test_follow_through_rejected(self, engine, today):
        """Should track finalists that were not picked."""
        _insert_run_summary(engine, today, "premarket", {}, ["AVGO"])
        # SMCI is a finalist but not in the picks
        _insert_candidate_row(
            engine, today, "premarket", "SMCI",
            penalties=[],
            reason_codes=[],
        )

        metrics = compute_daily_metrics(engine, today)
        follow = metrics["follow_through_rejected"]
        assert len(follow) == 1
        assert follow[0]["symbol"] == "SMCI"

    def test_defaults_to_today_when_no_date(self, engine):
        """Should default to today's date when no date is provided."""
        metrics = compute_daily_metrics(engine)
        assert metrics["date"] == date.today().isoformat()

    def test_aggregates_across_run_types(self, engine, today):
        """Should aggregate reason codes across premarket, confirmation, midday."""
        _insert_run_summary(engine, today, "premarket", {"price_below_min": 2}, ["AVGO"])
        _insert_run_summary(engine, today, "confirmation", {"price_below_min": 1, "spread_too_wide": 3}, ["SMCI"])

        metrics = compute_daily_metrics(engine, today)
        codes = metrics["top_rejection_reason_codes"]
        code_map = {c["code"]: c["count"] for c in codes}
        assert code_map["price_below_min"] == 3
        assert code_map["spread_too_wide"] == 3


class TestFormatMetricsForReview:
    """Tests for format_metrics_for_review()."""

    def test_formats_empty_metrics(self):
        """Should produce a readable summary even with zero data."""
        metrics = {
            "date": "2025-01-15",
            "expanded_candidates_surfaced": 0,
            "pct_reaching_analyst_long_short": 0.0,
            "pct_reaching_pm_eligible": 0.0,
            "executed_trade_outcomes": [],
            "top_rejection_reason_codes": [],
            "top_penalty_reason_codes": [],
            "follow_through_rejected": [],
        }
        result = format_metrics_for_review(metrics)
        assert "Sector Scout Metrics" in result
        assert "2025-01-15" in result
        assert "Expanded Candidates Surfaced: 0" in result

    def test_formats_full_metrics(self):
        """Should format all sections when data is present."""
        metrics = {
            "date": "2025-01-15",
            "expanded_candidates_surfaced": 5,
            "pct_reaching_analyst_long_short": 40.0,
            "pct_reaching_pm_eligible": 20.0,
            "executed_trade_outcomes": [
                {"symbol": "AVGO", "status": "closed", "pnl_pct": 2.5},
            ],
            "top_rejection_reason_codes": [
                {"code": "price_below_min", "count": 8},
                {"code": "spread_too_wide", "count": 3},
            ],
            "top_penalty_reason_codes": [
                {"code": "stale_news", "count": 5},
            ],
            "follow_through_rejected": [
                {"symbol": "SMCI", "subsequent_move_pct": 4.2},
            ],
        }
        result = format_metrics_for_review(metrics)
        assert "Expanded Candidates Surfaced: 5" in result
        assert "40.0%" in result
        assert "20.0%" in result
        assert "AVGO" in result
        assert "+2.50%" in result
        assert "price_below_min" in result
        assert "stale_news" in result
        assert "SMCI" in result
        assert "+4.20%" in result

    def test_contains_no_urgency_language(self):
        """Verify no trade pressure or urgency language in output."""
        metrics = {
            "date": "2025-01-15",
            "expanded_candidates_surfaced": 3,
            "pct_reaching_analyst_long_short": 66.7,
            "pct_reaching_pm_eligible": 33.3,
            "executed_trade_outcomes": [],
            "top_rejection_reason_codes": [],
            "top_penalty_reason_codes": [],
            "follow_through_rejected": [],
        }
        result = format_metrics_for_review(metrics)
        # No urgency language per requirement 7.2
        urgency_words = ["urgent", "immediately", "must trade", "buy now", "sell now", "opportunity missed"]
        for word in urgency_words:
            assert word.lower() not in result.lower()
