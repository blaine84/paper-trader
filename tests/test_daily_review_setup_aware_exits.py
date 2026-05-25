"""
Tests for gather_setup_aware_exit_summary and its integration into
build_deterministic_summary in agents/daily_review.py.
"""

import json
from datetime import date, datetime

from sqlalchemy import create_engine

from db.schema import Base, TradeEvent, get_session
from agents.daily_review import (
    gather_setup_aware_exit_summary,
    build_deterministic_summary,
)
from utils.setup_aware_evaluator import SETUP_EXIT_EVENT_TYPES


def _make_engine():
    """Create an in-memory SQLite engine with all tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


class TestGatherSetupAwareExitSummary:
    def test_no_events_returns_empty_summary(self):
        engine = _make_engine()
        today = date.today().isoformat()
        result = gather_setup_aware_exit_summary(engine, today)

        assert result["total_setup_aware_events"] == 0
        assert result["by_setup_type"] == {}
        assert result["timer_based_closes"] == 0
        assert result["thesis_invalidation_closes"] == 0
        assert result["revalidated_holds"] == 0

    def test_single_force_close_event(self):
        engine = _make_engine()
        today = date.today().isoformat()
        db = get_session(engine)
        db.add(TradeEvent(
            trade_id=1,
            event_type="setup_exit_force_close",
            timestamp=datetime.now(),
            payload_json=json.dumps({"setup_type": "news_breakout"}),
        ))
        db.commit()
        db.close()

        result = gather_setup_aware_exit_summary(engine, today)

        assert result["total_setup_aware_events"] == 1
        assert result["timer_based_closes"] == 1
        assert result["thesis_invalidation_closes"] == 0
        assert result["revalidated_holds"] == 0
        assert "news_breakout" in result["by_setup_type"]
        assert result["by_setup_type"]["news_breakout"]["setup_exit_force_close"] == 1

    def test_thesis_invalidation_event(self):
        engine = _make_engine()
        today = date.today().isoformat()
        db = get_session(engine)
        db.add(TradeEvent(
            trade_id=2,
            event_type="setup_exit_thesis_invalidated",
            timestamp=datetime.now(),
            payload_json=json.dumps({"setup_type": "news_catalyst"}),
        ))
        db.commit()
        db.close()

        result = gather_setup_aware_exit_summary(engine, today)

        assert result["total_setup_aware_events"] == 1
        assert result["timer_based_closes"] == 0
        assert result["thesis_invalidation_closes"] == 1
        assert "news_catalyst" in result["by_setup_type"]
        assert result["by_setup_type"]["news_catalyst"]["setup_exit_thesis_invalidated"] == 1

    def test_revalidated_hold_event(self):
        engine = _make_engine()
        today = date.today().isoformat()
        db = get_session(engine)
        db.add(TradeEvent(
            trade_id=3,
            event_type="setup_exit_revalidated_hold",
            timestamp=datetime.now(),
            payload_json=json.dumps({"setup_type": "trend_pullback"}),
        ))
        db.commit()
        db.close()

        result = gather_setup_aware_exit_summary(engine, today)

        assert result["total_setup_aware_events"] == 1
        assert result["revalidated_holds"] == 1
        assert "trend_pullback" in result["by_setup_type"]
        assert result["by_setup_type"]["trend_pullback"]["setup_exit_revalidated_hold"] == 1

    def test_multiple_events_grouped_by_setup_type(self):
        engine = _make_engine()
        today = date.today().isoformat()
        db = get_session(engine)

        # news_breakout events
        db.add(TradeEvent(
            trade_id=1,
            event_type="setup_exit_alert",
            timestamp=datetime.now(),
            payload_json=json.dumps({"setup_type": "news_breakout"}),
        ))
        db.add(TradeEvent(
            trade_id=1,
            event_type="setup_exit_revalidated_hold",
            timestamp=datetime.now(),
            payload_json=json.dumps({"setup_type": "news_breakout"}),
        ))
        db.add(TradeEvent(
            trade_id=1,
            event_type="setup_exit_force_close",
            timestamp=datetime.now(),
            payload_json=json.dumps({"setup_type": "news_breakout"}),
        ))

        # momentum_fade events
        db.add(TradeEvent(
            trade_id=2,
            event_type="setup_exit_force_close",
            timestamp=datetime.now(),
            payload_json=json.dumps({"setup_type": "momentum_fade"}),
        ))

        db.commit()
        db.close()

        result = gather_setup_aware_exit_summary(engine, today)

        assert result["total_setup_aware_events"] == 4
        assert result["timer_based_closes"] == 2
        assert result["revalidated_holds"] == 1

        nb = result["by_setup_type"]["news_breakout"]
        assert nb["setup_exit_alert"] == 1
        assert nb["setup_exit_revalidated_hold"] == 1
        assert nb["setup_exit_force_close"] == 1
        assert nb["setup_exit_thesis_invalidated"] == 0

        mf = result["by_setup_type"]["momentum_fade"]
        assert mf["setup_exit_force_close"] == 1
        assert mf["setup_exit_alert"] == 0

    def test_missing_payload_defaults_to_unknown(self):
        engine = _make_engine()
        today = date.today().isoformat()
        db = get_session(engine)
        db.add(TradeEvent(
            trade_id=4,
            event_type="setup_exit_alert",
            timestamp=datetime.now(),
            payload_json=None,
        ))
        db.commit()
        db.close()

        result = gather_setup_aware_exit_summary(engine, today)

        assert result["total_setup_aware_events"] == 1
        assert "unknown" in result["by_setup_type"]
        assert result["by_setup_type"]["unknown"]["setup_exit_alert"] == 1

    def test_events_from_other_days_excluded(self):
        engine = _make_engine()
        today = "2025-06-15"
        db = get_session(engine)
        # Event from a different day
        db.add(TradeEvent(
            trade_id=5,
            event_type="setup_exit_force_close",
            timestamp=datetime(2025, 6, 14, 12, 0, 0),
            payload_json=json.dumps({"setup_type": "news_breakout"}),
        ))
        db.commit()
        db.close()

        result = gather_setup_aware_exit_summary(engine, today)

        assert result["total_setup_aware_events"] == 0
        assert result["by_setup_type"] == {}

    def test_non_setup_event_types_excluded(self):
        engine = _make_engine()
        today = date.today().isoformat()
        db = get_session(engine)
        # A non-setup-aware event type
        db.add(TradeEvent(
            trade_id=6,
            event_type="news_expiry_force_close",
            timestamp=datetime.now(),
            payload_json=json.dumps({"setup_type": "news_breakout"}),
        ))
        db.commit()
        db.close()

        result = gather_setup_aware_exit_summary(engine, today)

        assert result["total_setup_aware_events"] == 0


class TestBuildDeterministicSummarySetupAwareExits:
    def test_setup_aware_exits_included_in_summary(self):
        exits = {
            "total_setup_aware_events": 3,
            "by_setup_type": {"news_breakout": {"setup_exit_force_close": 1}},
            "timer_based_closes": 1,
            "thesis_invalidation_closes": 1,
            "revalidated_holds": 1,
        }
        summary = build_deterministic_summary(
            None, None, None, None, None, setup_aware_exits=exits
        )
        assert summary["setup_aware_exits"] == exits

    def test_none_setup_aware_exits_defaults_to_empty(self):
        summary = build_deterministic_summary(None, None, None, None, None)
        assert summary["setup_aware_exits"]["total_setup_aware_events"] == 0
        assert summary["setup_aware_exits"]["by_setup_type"] == {}
        assert summary["setup_aware_exits"]["timer_based_closes"] == 0
        assert summary["setup_aware_exits"]["thesis_invalidation_closes"] == 0
        assert summary["setup_aware_exits"]["revalidated_holds"] == 0
