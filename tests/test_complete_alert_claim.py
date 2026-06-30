"""Tests for complete_alert_claim() — Task 10.2.

Verifies:
- Updates pm_alert_claims status to 'completed' for non-error outcomes
- Updates pm_alert_claims status to 'error' for error outcomes
- Sets completed_at timestamp
- Records outcome event in pm_alert_events via record_alert_event()

Requirements: 5.3
"""

import json
import pytest
from sqlalchemy import create_engine, text
from utils.alert_dispatch_schema import init_alert_dispatch_schema
from agents.portfolio_manager import complete_alert_claim, record_alert_event


@pytest.fixture
def engine():
    """Create an in-memory SQLite engine with alert dispatch schema."""
    eng = create_engine("sqlite:///:memory:")
    init_alert_dispatch_schema(eng)
    return eng


@pytest.fixture
def claimed_row(engine):
    """Insert a claimed row into pm_alert_claims for testing completion."""
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO pm_alert_claims
                (alert_intent_id, symbol, alert_type, profile_id, status, claimed_at)
            VALUES ('intent-abc', 'NVDA', 'entry_alert', 'aggressive', 'claimed', '2025-01-15T10:00:00.000000Z')
        """))
    return "intent-abc"


class TestCompleteAlertClaimSuccess:
    """complete_alert_claim with non-error outcome sets status='completed'."""

    def test_sets_completed_status(self, engine, claimed_row):
        """Non-error outcome → claim status becomes 'completed'."""
        complete_alert_claim(engine, "intent-abc", "accepted", "aggressive")

        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT status, completed_at FROM pm_alert_claims WHERE alert_intent_id = 'intent-abc'"
            )).fetchone()

        assert row[0] == "completed"
        assert row[1] is not None  # completed_at set

    def test_rejected_outcome_is_completed(self, engine, claimed_row):
        """'rejected' outcome still results in 'completed' status (not 'error')."""
        complete_alert_claim(engine, "intent-abc", "rejected", "aggressive")

        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT status FROM pm_alert_claims WHERE alert_intent_id = 'intent-abc'"
            )).fetchone()

        assert row[0] == "completed"

    def test_gate_rejected_outcome_is_completed(self, engine, claimed_row):
        """'gate_rejected' outcome still results in 'completed' status."""
        complete_alert_claim(engine, "intent-abc", "gate_rejected", "aggressive")

        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT status FROM pm_alert_claims WHERE alert_intent_id = 'intent-abc'"
            )).fetchone()

        assert row[0] == "completed"


class TestCompleteAlertClaimError:
    """complete_alert_claim with 'error' outcome sets status='error'."""

    def test_error_outcome_sets_error_status(self, engine, claimed_row):
        """'error' outcome → claim status becomes 'error'."""
        complete_alert_claim(engine, "intent-abc", "error", "aggressive")

        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT status, completed_at FROM pm_alert_claims WHERE alert_intent_id = 'intent-abc'"
            )).fetchone()

        assert row[0] == "error"
        assert row[1] is not None  # completed_at still set


class TestCompleteAlertClaimEvent:
    """complete_alert_claim records a 'completed' event in pm_alert_events."""

    def test_event_recorded_with_outcome(self, engine, claimed_row):
        """A 'completed' event is inserted into pm_alert_events with the outcome."""
        complete_alert_claim(engine, "intent-abc", "accepted", "aggressive")

        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT event_type, profile_id, event_data FROM pm_alert_events
                WHERE alert_intent_id = 'intent-abc' AND event_type = 'completed'
            """)).fetchone()

        assert row is not None
        assert row[0] == "completed"
        assert row[1] == "aggressive"
        data = json.loads(row[2])
        assert data["outcome"] == "accepted"

    def test_error_event_recorded(self, engine, claimed_row):
        """Error outcome is recorded in event_data."""
        complete_alert_claim(engine, "intent-abc", "error", "aggressive")

        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT event_data FROM pm_alert_events
                WHERE alert_intent_id = 'intent-abc' AND event_type = 'completed'
            """)).fetchone()

        data = json.loads(row[0])
        assert data["outcome"] == "error"


class TestRecordAlertEventFailOpen:
    """record_alert_event is fail-open — doesn't raise on DB errors."""

    def test_basic_insert(self, engine):
        """record_alert_event inserts an event row successfully."""
        record_alert_event(engine, "intent-x", "claimed", "AAPL", "breakout", "moderate")

        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT alert_intent_id, event_type, symbol FROM pm_alert_events WHERE alert_intent_id = 'intent-x'"
            )).fetchone()

        assert row[0] == "intent-x"
        assert row[1] == "claimed"
        assert row[2] == "AAPL"

    def test_extra_data_serialized(self, engine):
        """Extra kwargs are serialized to event_data as JSON."""
        record_alert_event(
            engine, "intent-y", "duplicate_noop", "TSLA", "rapid_move", "aggressive",
            original_status="completed", original_claimed_at="2025-01-15T09:00:00Z"
        )

        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT event_data FROM pm_alert_events WHERE alert_intent_id = 'intent-y'"
            )).fetchone()

        data = json.loads(row[0])
        assert data["original_status"] == "completed"
        assert data["original_claimed_at"] == "2025-01-15T09:00:00Z"

    def test_no_extra_data_means_null(self, engine):
        """No extra kwargs → event_data is NULL."""
        record_alert_event(engine, "intent-z", "claimed", "MSFT", "entry_alert", "moderate")

        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT event_data FROM pm_alert_events WHERE alert_intent_id = 'intent-z'"
            )).fetchone()

        assert row[0] is None
