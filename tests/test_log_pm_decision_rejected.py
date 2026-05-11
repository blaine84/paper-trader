"""
Unit tests for _log_pm_decision_rejected helper.

Validates:
- Logs pm_decision_rejected trade event with correct payload (Req 9.1)
- Includes normalized symbol if parseable, null otherwise (Req 9.2)
- Truncates raw JSON to 2000 chars with "..." suffix (Req 9.4)
- Uses default=str for non-serializable values (Req 9.1)
"""

from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, TradeEvent
from agents.portfolio_manager import (
    _log_pm_decision_rejected,
    NormalizedOrder,
)


@pytest.fixture
def db_session():
    """In-memory SQLite session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestLogPmDecisionRejected:
    """Tests for _log_pm_decision_rejected helper."""

    def test_basic_rejection_logged(self, db_session):
        """Logs pm_decision_rejected with reason_code, reason, and profile."""
        rejection = NormalizedOrder(
            ok=False,
            reason_code="unsupported_symbol",
            reason="Symbol 'XYZ' not in allowed entry symbols",
            details={"symbol": "XYZ"},
            raw_decision={"action": "BUY", "symbol": "XYZ", "quantity": 10},
        )

        _log_pm_decision_rejected(db_session, rejection, "aggressive")
        db_session.flush()

        events = db_session.query(TradeEvent).all()
        assert len(events) == 1
        event = events[0]
        assert event.event_type == "pm_decision_rejected"
        assert event.agent == "portfolio_manager"
        assert event.symbol == "XYZ"
        assert event.profile == "aggressive"

    def test_symbol_from_details(self, db_session):
        """Uses symbol from rejection.details when available."""
        rejection = NormalizedOrder(
            ok=False,
            reason_code="invalid_quantity",
            reason="Quantity rejected",
            details={"symbol": "NVDA"},
            raw_decision={"action": "BUY", "symbol": "NVDA", "quantity": 0},
        )

        _log_pm_decision_rejected(db_session, rejection, "moderate")
        db_session.flush()

        event = db_session.query(TradeEvent).first()
        assert event.symbol == "NVDA"

    def test_symbol_from_raw_decision_fallback(self, db_session):
        """Falls back to raw_decision symbol when details has no symbol."""
        rejection = NormalizedOrder(
            ok=False,
            reason_code="invalid_quantity",
            reason="Quantity rejected",
            details={"quantity": -1},
            raw_decision={"action": "BUY", "symbol": "AMD", "quantity": -1},
        )

        _log_pm_decision_rejected(db_session, rejection, "conservative")
        db_session.flush()

        event = db_session.query(TradeEvent).first()
        assert event.symbol == "AMD"

    def test_symbol_null_when_unparseable(self, db_session):
        """Symbol is null when neither details nor raw_decision has a string symbol."""
        rejection = NormalizedOrder(
            ok=False,
            reason_code="missing_action",
            reason="Action missing",
            details={},
            raw_decision={"quantity": 10},
        )

        _log_pm_decision_rejected(db_session, rejection, "aggressive")
        db_session.flush()

        event = db_session.query(TradeEvent).first()
        assert event.symbol is None

    def test_truncation_at_2000_chars(self, db_session):
        """Raw JSON is truncated to 2000 chars with '...' suffix."""
        # Create a raw_decision that serializes to > 2000 chars
        large_rationale = "x" * 3000
        rejection = NormalizedOrder(
            ok=False,
            reason_code="invalid_geometry",
            reason="Geometry rejected",
            details={"symbol": "TSLA"},
            raw_decision={
                "action": "BUY",
                "symbol": "TSLA",
                "rationale": large_rationale,
            },
        )

        _log_pm_decision_rejected(db_session, rejection, "aggressive")
        db_session.flush()

        event = db_session.query(TradeEvent).first()
        import json
        payload = json.loads(event.payload_json)
        raw = payload["raw_decision"]
        assert len(raw) == 2000
        assert raw.endswith("...")

    def test_non_serializable_values_handled(self, db_session):
        """Non-serializable values (e.g., datetime) are handled via default=str."""
        rejection = NormalizedOrder(
            ok=False,
            reason_code="invalid_price",
            reason="Price rejected",
            details={"symbol": "AAPL"},
            raw_decision={
                "action": "BUY",
                "symbol": "AAPL",
                "timestamp": datetime(2024, 1, 15, 10, 30),
            },
        )

        # Should not raise
        _log_pm_decision_rejected(db_session, rejection, "moderate")
        db_session.flush()

        event = db_session.query(TradeEvent).first()
        assert event.event_type == "pm_decision_rejected"

    def test_payload_contains_all_required_fields(self, db_session):
        """Payload includes reason_code, reason, symbol, raw_decision, profile."""
        rejection = NormalizedOrder(
            ok=False,
            reason_code="missing_stop_loss",
            reason="Stop rejected: missing_stop_loss",
            details={"symbol": "GOOG"},
            raw_decision={"action": "BUY", "symbol": "GOOG", "quantity": 5},
        )

        _log_pm_decision_rejected(db_session, rejection, "conservative")
        db_session.flush()

        event = db_session.query(TradeEvent).first()
        import json
        payload = json.loads(event.payload_json)
        assert payload["reason_code"] == "missing_stop_loss"
        assert payload["reason"] == "Stop rejected: missing_stop_loss"
        assert payload["symbol"] == "GOOG"
        assert payload["profile"] == "conservative"
        assert "raw_decision" in payload
