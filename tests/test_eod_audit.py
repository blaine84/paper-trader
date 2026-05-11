"""Tests for EOD audit exception handling and alerting."""

import json
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, Trade, TradeEvent, AgentMemory
from utils.eod_audit import handle_eod_exceptions


def _make_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def _make_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()


class TestHandleEodExceptions:
    """Tests for handle_eod_exceptions function."""

    def test_no_unauthorized_positions_no_error_no_memory(self):
        """When no unauthorized positions remain, no ERROR or AgentMemory is written."""
        engine = _make_engine()
        db = _make_session(engine)
        now_utc = datetime(2025, 6, 15, 21, 0, 0, tzinfo=timezone.utc)

        audit_result = {
            "open_positions_total": 2,
            "by_profile": {"moderate": {"count": 2, "gross_exposure_pct": 0.05}},
            "positions": [
                {"trade_id": 1, "symbol": "AAPL", "authorization_status": "valid",
                 "profile": "moderate", "setup_type": "", "hours_held": 5.0,
                 "stop_status": "valid", "required_action": "allow"},
                {"trade_id": 2, "symbol": "MSFT", "authorization_status": "valid",
                 "profile": "moderate", "setup_type": "", "hours_held": 3.0,
                 "stop_status": "valid", "required_action": "allow"},
            ],
            "unauthorized_positions_remaining": 0,
            "force_closed_by_governance": 1,
        }

        handle_eod_exceptions(db, audit_result, now_utc)

        # No AgentMemory for eod_exposure_exception
        memories = db.query(AgentMemory).filter_by(key="eod_exposure_exception").all()
        assert len(memories) == 0

        # But audit event should still be written
        events = db.query(TradeEvent).filter_by(event_type="eod_exposure_audit").all()
        assert len(events) == 1
        assert events[0].dedupe_key == "eod_exposure_audit:2025-06-15"

        db.close()

    def test_unauthorized_positions_writes_error_and_memory(self):
        """When unauthorized positions remain, ERROR is logged and AgentMemory is written."""
        engine = _make_engine()
        db = _make_session(engine)
        now_utc = datetime(2025, 6, 15, 21, 0, 0, tzinfo=timezone.utc)

        audit_result = {
            "open_positions_total": 2,
            "by_profile": {"moderate": {"count": 2, "gross_exposure_pct": 0.08}},
            "positions": [
                {"trade_id": 1, "symbol": "AAPL", "authorization_status": "missing",
                 "profile": "moderate", "setup_type": "", "hours_held": 8.0,
                 "stop_status": "valid", "required_action": "close"},
                {"trade_id": 2, "symbol": "MSFT", "authorization_status": "valid",
                 "profile": "moderate", "setup_type": "", "hours_held": 3.0,
                 "stop_status": "valid", "required_action": "allow"},
            ],
            "unauthorized_positions_remaining": 1,
            "force_closed_by_governance": 0,
        }

        handle_eod_exceptions(db, audit_result, now_utc)

        # AgentMemory for eod_exposure_exception should exist
        memories = db.query(AgentMemory).filter_by(key="eod_exposure_exception").all()
        assert len(memories) == 1
        mem = memories[0]
        assert mem.agent == "position_timer"
        assert mem.key == "eod_exposure_exception"
        value = json.loads(mem.value)
        assert value["unauthorized_count"] == 1
        assert len(value["positions"]) == 1
        assert value["positions"][0]["symbol"] == "AAPL"

        # Audit event should also be written
        events = db.query(TradeEvent).filter_by(event_type="eod_exposure_audit").all()
        assert len(events) == 1

        db.close()

    def test_daily_deduplication_prevents_duplicate_audit_event(self):
        """Calling handle_eod_exceptions twice on the same day writes only one audit event."""
        engine = _make_engine()
        db = _make_session(engine)
        now_utc = datetime(2025, 6, 15, 21, 0, 0, tzinfo=timezone.utc)

        audit_result = {
            "open_positions_total": 1,
            "by_profile": {"moderate": {"count": 1, "gross_exposure_pct": 0.03}},
            "positions": [
                {"trade_id": 1, "symbol": "AAPL", "authorization_status": "valid",
                 "profile": "moderate", "setup_type": "", "hours_held": 2.0,
                 "stop_status": "valid", "required_action": "allow"},
            ],
            "unauthorized_positions_remaining": 0,
            "force_closed_by_governance": 0,
        }

        # Call twice
        handle_eod_exceptions(db, audit_result, now_utc)
        handle_eod_exceptions(db, audit_result, now_utc)

        # Only one audit event should exist
        events = db.query(TradeEvent).filter_by(event_type="eod_exposure_audit").all()
        assert len(events) == 1

        db.close()

    def test_different_days_write_separate_audit_events(self):
        """Audit events on different days are not deduplicated."""
        engine = _make_engine()
        db = _make_session(engine)
        day1 = datetime(2025, 6, 15, 21, 0, 0, tzinfo=timezone.utc)
        day2 = datetime(2025, 6, 16, 21, 0, 0, tzinfo=timezone.utc)

        audit_result = {
            "open_positions_total": 0,
            "by_profile": {},
            "positions": [],
            "unauthorized_positions_remaining": 0,
            "force_closed_by_governance": 0,
        }

        handle_eod_exceptions(db, audit_result, day1)
        handle_eod_exceptions(db, audit_result, day2)

        events = db.query(TradeEvent).filter_by(event_type="eod_exposure_audit").all()
        assert len(events) == 2

        dedupe_keys = {e.dedupe_key for e in events}
        assert "eod_exposure_audit:2025-06-15" in dedupe_keys
        assert "eod_exposure_audit:2025-06-16" in dedupe_keys

        db.close()

    def test_audit_event_message_includes_force_closed_count(self):
        """The audit event message reports the count of force-closed positions."""
        engine = _make_engine()
        db = _make_session(engine)
        now_utc = datetime(2025, 6, 15, 21, 0, 0, tzinfo=timezone.utc)

        audit_result = {
            "open_positions_total": 3,
            "by_profile": {},
            "positions": [],
            "unauthorized_positions_remaining": 1,
            "force_closed_by_governance": 2,
        }

        handle_eod_exceptions(db, audit_result, now_utc)

        events = db.query(TradeEvent).filter_by(event_type="eod_exposure_audit").all()
        assert len(events) == 1
        assert "2 force-closed" in events[0].message
        assert "3 open" in events[0].message
        assert "1 unauthorized" in events[0].message

        db.close()
