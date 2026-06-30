"""Tests for claim_alert_for_processing() and record_alert_event() — Task 10.1.

Verifies:
- Atomic INSERT-or-reject via UNIQUE constraint on (alert_intent_id, profile_id)
- IntegrityError handling: check existing row status, attempt reclaim if 'error'
- Return None (proceed) on success, dict with duplicate_noop on failure
- Events recorded in pm_alert_events for all outcomes (claimed, reclaimed, duplicate_noop)

Requirements: 5.1, 5.2, 5.4
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from utils.alert_dispatch_schema import init_alert_dispatch_schema
from agents.portfolio_manager import claim_alert_for_processing, record_alert_event


@pytest.fixture
def engine():
    """In-memory SQLite engine with alert dispatch schema initialized."""
    eng = create_engine("sqlite://", echo=False)
    init_alert_dispatch_schema(eng)
    return eng


class TestRecordAlertEvent:
    """record_alert_event() — fail-open INSERT into pm_alert_events."""

    def test_basic_insert(self, engine):
        """Inserts event row with all fields."""
        record_alert_event(engine, "intent-1", "claimed", "NVDA", "entry_alert", "aggressive")

        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT alert_intent_id, event_type, symbol, alert_type, profile_id, event_at "
                "FROM pm_alert_events WHERE alert_intent_id = 'intent-1'"
            )).fetchone()

        assert row is not None
        assert row[0] == "intent-1"
        assert row[1] == "claimed"
        assert row[2] == "NVDA"
        assert row[3] == "entry_alert"
        assert row[4] == "aggressive"
        assert row[5] is not None  # event_at timestamp

    def test_extra_data_serialized_as_json(self, engine):
        """Extra keyword args are serialized as JSON in event_data."""
        record_alert_event(
            engine, "intent-2", "duplicate_noop", "AAPL", "breakout", "moderate",
            original_status="claimed", original_claimed_at="2025-01-15T10:00:00Z",
        )

        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT event_data FROM pm_alert_events WHERE alert_intent_id = 'intent-2'"
            )).fetchone()

        assert row is not None
        import json
        data = json.loads(row[0])
        assert data["original_status"] == "claimed"
        assert data["original_claimed_at"] == "2025-01-15T10:00:00Z"

    def test_fail_open_on_error(self, engine):
        """Does not raise on failure (fail-open pattern)."""
        # Pass a broken engine to trigger an error
        broken_engine = create_engine("sqlite://", echo=False)
        # Don't init schema — table doesn't exist
        # Should not raise
        record_alert_event(broken_engine, "intent-x", "claimed", "TSLA", "rapid_move", "aggressive")


class TestClaimAlertForProcessing:
    """claim_alert_for_processing() — atomic INSERT-or-reject idempotency."""

    def test_first_claim_succeeds_returns_none(self, engine):
        """First claim for an (alert_intent_id, profile_id) succeeds → returns None."""
        result = claim_alert_for_processing(
            engine, "intent-100", "NVDA", "entry_alert", "aggressive"
        )
        assert result is None

    def test_first_claim_inserts_row(self, engine):
        """Successful claim creates a row in pm_alert_claims with status='claimed'."""
        claim_alert_for_processing(engine, "intent-101", "AAPL", "breakout", "moderate")

        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT alert_intent_id, symbol, alert_type, profile_id, status "
                "FROM pm_alert_claims WHERE alert_intent_id = 'intent-101' AND profile_id = 'moderate'"
            )).fetchone()

        assert row is not None
        assert row[0] == "intent-101"
        assert row[1] == "AAPL"
        assert row[2] == "breakout"
        assert row[3] == "moderate"
        assert row[4] == "claimed"

    def test_first_claim_records_claimed_event(self, engine):
        """Successful claim records a 'claimed' event in pm_alert_events."""
        claim_alert_for_processing(engine, "intent-102", "MSFT", "entry_alert", "conservative")

        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT event_type, symbol, alert_type, profile_id "
                "FROM pm_alert_events WHERE alert_intent_id = 'intent-102'"
            )).fetchone()

        assert row is not None
        assert row[0] == "claimed"
        assert row[1] == "MSFT"
        assert row[2] == "entry_alert"
        assert row[3] == "conservative"

    def test_duplicate_claim_returns_noop(self, engine):
        """Second claim for same (alert_intent_id, profile_id) returns duplicate_noop."""
        claim_alert_for_processing(engine, "intent-200", "TSLA", "entry_alert", "aggressive")
        result = claim_alert_for_processing(engine, "intent-200", "TSLA", "entry_alert", "aggressive")

        assert result is not None
        assert result["outcome"] == "duplicate_noop"
        assert result["original_status"] == "claimed"
        assert "original_claimed_at" in result

    def test_duplicate_claim_records_noop_event(self, engine):
        """Duplicate claim records a 'duplicate_noop' event."""
        claim_alert_for_processing(engine, "intent-201", "NVDA", "entry_alert", "moderate")
        claim_alert_for_processing(engine, "intent-201", "NVDA", "entry_alert", "moderate")

        with engine.connect() as conn:
            events = conn.execute(text(
                "SELECT event_type FROM pm_alert_events "
                "WHERE alert_intent_id = 'intent-201' ORDER BY id"
            )).fetchall()

        event_types = [e[0] for e in events]
        assert "claimed" in event_types
        assert "duplicate_noop" in event_types

    def test_different_profiles_can_claim_same_intent(self, engine):
        """Different profiles independently claim the same alert_intent_id."""
        r1 = claim_alert_for_processing(engine, "intent-300", "AAPL", "entry_alert", "aggressive")
        r2 = claim_alert_for_processing(engine, "intent-300", "AAPL", "entry_alert", "moderate")
        r3 = claim_alert_for_processing(engine, "intent-300", "AAPL", "entry_alert", "conservative")

        assert r1 is None
        assert r2 is None
        assert r3 is None

    def test_reclaim_error_status_succeeds(self, engine):
        """Claim with status='error' (stale-recovered) can be reclaimed."""
        # Insert a claim row directly with status='error' to simulate stale recovery
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO pm_alert_claims
                    (alert_intent_id, symbol, alert_type, profile_id, status, claimed_at, completed_at)
                VALUES ('intent-400', 'GOOGL', 'entry_alert', 'aggressive', 'error',
                        '2025-01-15T08:00:00Z', '2025-01-15T08:10:00Z')
            """))

        # Now attempt to claim — should reclaim successfully
        result = claim_alert_for_processing(
            engine, "intent-400", "GOOGL", "entry_alert", "aggressive"
        )
        assert result is None  # reclaim succeeded

        # Verify status is now 'claimed' again
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT status, completed_at FROM pm_alert_claims "
                "WHERE alert_intent_id = 'intent-400' AND profile_id = 'aggressive'"
            )).fetchone()
        assert row[0] == "claimed"
        assert row[1] is None  # completed_at cleared on reclaim

    def test_reclaim_records_reclaimed_event(self, engine):
        """Successful reclaim records a 'reclaimed' event with previous_status."""
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO pm_alert_claims
                    (alert_intent_id, symbol, alert_type, profile_id, status, claimed_at)
                VALUES ('intent-401', 'AMZN', 'breakout', 'moderate', 'error', '2025-01-15T07:00:00Z')
            """))

        claim_alert_for_processing(engine, "intent-401", "AMZN", "breakout", "moderate")

        with engine.connect() as conn:
            events = conn.execute(text(
                "SELECT event_type, event_data FROM pm_alert_events "
                "WHERE alert_intent_id = 'intent-401' ORDER BY id"
            )).fetchall()

        reclaimed_events = [e for e in events if e[0] == "reclaimed"]
        assert len(reclaimed_events) == 1

        import json
        data = json.loads(reclaimed_events[0][1])
        assert data["previous_status"] == "error"

    def test_completed_claim_returns_noop(self, engine):
        """Claim with status='completed' returns duplicate_noop (not reclaimable)."""
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO pm_alert_claims
                    (alert_intent_id, symbol, alert_type, profile_id, status, claimed_at, completed_at)
                VALUES ('intent-500', 'META', 'entry_alert', 'aggressive', 'completed',
                        '2025-01-15T09:00:00Z', '2025-01-15T09:05:00Z')
            """))

        result = claim_alert_for_processing(
            engine, "intent-500", "META", "entry_alert", "aggressive"
        )
        assert result is not None
        assert result["outcome"] == "duplicate_noop"
        assert result["original_status"] == "completed"
