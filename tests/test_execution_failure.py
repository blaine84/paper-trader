"""
Unit tests for execution failure classification functions.

Tests the three new functions added to utils/candidate_pipeline.py:
- _write_execution_failed_event (Requirements 6.1, 6.2)
- _write_execution_fallback_blocked_event (Requirement 6.3)
- _attempt_execution_failed_transition (Requirements 6.4, 6.5)
"""
from __future__ import annotations

import json
import uuid

from sqlalchemy import create_engine, text
from unittest.mock import MagicMock, patch

from utils.candidate_pipeline import (
    _write_execution_failed_event,
    _write_execution_fallback_blocked_event,
    _attempt_execution_failed_transition,
)
from utils.candidate_registry import CandidateRecord


def _create_test_engine():
    """Create an in-memory SQLite database with needed tables."""
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        conn.execute(text("""
            CREATE TABLE pm_candidates (
                id INTEGER PRIMARY KEY,
                candidate_id VARCHAR(36) NOT NULL UNIQUE,
                cycle_id VARCHAR(64) NOT NULL,
                profile_id VARCHAR(64) NOT NULL,
                symbol VARCHAR(10) NOT NULL,
                direction VARCHAR(10) NOT NULL,
                setup_type VARCHAR(64) NOT NULL,
                geometry_name VARCHAR(64) NOT NULL,
                entry_price REAL NOT NULL,
                stop_price REAL NOT NULL,
                target_price REAL NOT NULL,
                risk_reward REAL NOT NULL,
                source_signal_id VARCHAR(64) NOT NULL,
                signal_snapshot_json TEXT NOT NULL,
                state VARCHAR(32) DEFAULT 'registered',
                integrity_hash VARCHAR(64) NOT NULL,
                expires_at DATETIME NOT NULL,
                rejection_reason TEXT,
                rejection_reason_code VARCHAR(64)
            )
        """))
        conn.execute(text("""
            CREATE TABLE pm_candidate_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                cycle_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_data TEXT,
                created_at TEXT NOT NULL,
                candidate_type TEXT
            )
        """))
    return eng


def _make_candidate(candidate_id=None, symbol="AAPL", candidate_type="intraday"):
    """Create a CandidateRecord for testing."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    cid = candidate_id or str(uuid.uuid4())
    return CandidateRecord(
        candidate_id=cid,
        cycle_id="cycle-001",
        profile_id="moderate",
        symbol=symbol,
        direction="BUY",
        setup_type="momentum_breakout",
        geometry_name="analyst_geometry",
        entry_price=150.0,
        stop_price=148.0,
        target_price=155.0,
        risk_reward=2.5,
        trigger="breakout above resistance",
        invalidation_basis="below 148.0",
        target_basis="measured move to 155.0",
        source_signal_id="sig-001",
        signal_snapshot_json='{"test": true}',
        created_at=now,
        expires_at=now,
        integrity_hash="abc123",
        candidate_type=candidate_type,
    )


class TestWriteExecutionFailedEvent:
    """Tests for _write_execution_failed_event (Req 6.1, 6.2)."""

    def test_writes_execution_failed_event_with_correct_fields(self):
        """Verify the event is written with all required fields."""
        engine = _create_test_engine()
        candidate = _make_candidate()
        cycle_id = "cycle-001"
        profile_id = "moderate"

        _write_execution_failed_event(
            engine, candidate, cycle_id, profile_id,
            intended_action="BUY",
            attempted_quantity=50,
            failure_reason="execute_trade returned False: broker error",
        )

        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT * FROM pm_candidate_events WHERE event_type = 'execution_failed'"
            )).mappings().first()

        assert row is not None
        assert row["candidate_id"] == candidate.candidate_id
        assert row["cycle_id"] == cycle_id
        assert row["profile_id"] == profile_id
        assert row["event_type"] == "execution_failed"
        assert row["candidate_type"] == "intraday"

        event_data = json.loads(row["event_data"])
        assert event_data["candidate_id"] == candidate.candidate_id
        assert event_data["profile"] == profile_id
        assert event_data["symbol"] == "AAPL"
        assert event_data["intended_action"] == "BUY"
        assert event_data["attempted_quantity"] == 50
        assert event_data["failure_reason"] == "execute_trade returned False: broker error"

    def test_failure_reason_truncated_to_1024_chars(self):
        """Verify failure_reason is truncated to max 1024 characters."""
        engine = _create_test_engine()
        candidate = _make_candidate()

        long_reason = "x" * 2000

        _write_execution_failed_event(
            engine, candidate, "cycle-001", "moderate",
            intended_action="BUY",
            attempted_quantity=100,
            failure_reason=long_reason,
        )

        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT event_data FROM pm_candidate_events WHERE event_type = 'execution_failed'"
            )).mappings().first()

        event_data = json.loads(row["event_data"])
        assert len(event_data["failure_reason"]) == 1024

    def test_event_type_distinct_from_pm_reject_and_gate_fail(self):
        """Verify the event_type is 'execution_failed', not pm_reject or gate_fail."""
        engine = _create_test_engine()
        candidate = _make_candidate()

        _write_execution_failed_event(
            engine, candidate, "cycle-001", "moderate",
            intended_action="SHORT",
            attempted_quantity=25,
            failure_reason="timeout",
        )

        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT event_type FROM pm_candidate_events LIMIT 1"
            )).mappings().first()

        assert row["event_type"] == "execution_failed"
        assert row["event_type"] != "pm_reject"
        assert row["event_type"] != "gate_fail"

    def test_fail_open_on_db_error(self):
        """Verify the function logs and continues on database error (fail-open)."""
        candidate = _make_candidate()
        broken_engine = MagicMock()
        broken_engine.connect.side_effect = Exception("DB connection lost")

        # Should not raise
        _write_execution_failed_event(
            broken_engine, candidate, "cycle-001", "moderate",
            intended_action="BUY",
            attempted_quantity=10,
            failure_reason="some error",
        )

    def test_swing_candidate_type_preserved(self):
        """Verify candidate_type is correctly set for swing candidates."""
        engine = _create_test_engine()
        candidate = _make_candidate(candidate_type="swing")

        _write_execution_failed_event(
            engine, candidate, "cycle-001", "moderate",
            intended_action="BUY",
            attempted_quantity=30,
            failure_reason="execution timeout",
        )

        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT candidate_type FROM pm_candidate_events LIMIT 1"
            )).mappings().first()

        assert row["candidate_type"] == "swing"


class TestWriteExecutionFallbackBlockedEvent:
    """Tests for _write_execution_fallback_blocked_event (Req 6.3)."""

    def test_writes_fallback_blocked_event(self):
        """Verify the event is written with candidate_id and blocked_path."""
        engine = _create_test_engine()
        candidate = _make_candidate()

        _write_execution_fallback_blocked_event(
            engine, candidate, "cycle-001", "moderate",
            blocked_path="legacy_free_form_order",
        )

        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT * FROM pm_candidate_events WHERE event_type = 'execution_fallback_blocked'"
            )).mappings().first()

        assert row is not None
        assert row["event_type"] == "execution_fallback_blocked"

        event_data = json.loads(row["event_data"])
        assert event_data["candidate_id"] == candidate.candidate_id
        assert event_data["blocked_path"] == "legacy_free_form_order"

    def test_fail_open_on_db_error(self):
        """Verify the function logs and continues on database error (fail-open)."""
        candidate = _make_candidate()
        broken_engine = MagicMock()
        broken_engine.connect.side_effect = Exception("DB connection lost")

        # Should not raise
        _write_execution_fallback_blocked_event(
            broken_engine, candidate, "cycle-001", "moderate",
            blocked_path="legacy_free_form_order",
        )


class TestAttemptExecutionFailedTransition:
    """Tests for _attempt_execution_failed_transition (Req 6.4, 6.5)."""

    def test_successful_cas_transition_from_reserved(self):
        """Verify successful CAS transition from reserved to execution_failed."""
        engine = _create_test_engine()
        candidate_id = str(uuid.uuid4())

        # Insert a candidate in 'reserved' state
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO pm_candidates
                (candidate_id, cycle_id, profile_id, symbol, direction, setup_type,
                 geometry_name, entry_price, stop_price, target_price, risk_reward,
                 source_signal_id, signal_snapshot_json, state, integrity_hash, expires_at)
                VALUES (:cid, 'cycle-001', 'moderate', 'AAPL', 'BUY', 'momentum',
                        'analyst', 150.0, 148.0, 155.0, 2.5,
                        'sig-001', '{}', 'reserved', 'hash123', '2099-01-01')
            """), {"cid": candidate_id})

        _attempt_execution_failed_transition(engine, candidate_id)

        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT state FROM pm_candidates WHERE candidate_id = :cid"
            ), {"cid": candidate_id}).mappings().first()

        assert row["state"] == "execution_failed"

    def test_cas_fails_when_not_in_reserved_state(self):
        """Verify CAS fails gracefully (rowcount=0) when state is not reserved."""
        engine = _create_test_engine()
        candidate_id = str(uuid.uuid4())

        # Insert a candidate in 'executed' state (already terminal)
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO pm_candidates
                (candidate_id, cycle_id, profile_id, symbol, direction, setup_type,
                 geometry_name, entry_price, stop_price, target_price, risk_reward,
                 source_signal_id, signal_snapshot_json, state, integrity_hash, expires_at)
                VALUES (:cid, 'cycle-001', 'moderate', 'AAPL', 'BUY', 'momentum',
                        'analyst', 150.0, 148.0, 155.0, 2.5,
                        'sig-001', '{}', 'executed', 'hash123', '2099-01-01')
            """), {"cid": candidate_id})

        # Should not raise — logs and continues
        _attempt_execution_failed_transition(engine, candidate_id)

        # State should remain unchanged
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT state FROM pm_candidates WHERE candidate_id = :cid"
            ), {"cid": candidate_id}).mappings().first()

        assert row["state"] == "executed"

    def test_cas_fails_when_candidate_not_found(self):
        """Verify CAS fails gracefully when candidate doesn't exist (rowcount=0)."""
        engine = _create_test_engine()
        nonexistent_id = str(uuid.uuid4())

        # Should not raise
        _attempt_execution_failed_transition(engine, nonexistent_id)

    def test_fail_open_on_db_exception(self):
        """Verify the function logs and continues on database exception (fail-open)."""
        broken_engine = MagicMock()
        broken_engine.connect.side_effect = Exception("DB connection lost")

        # Should not raise
        _attempt_execution_failed_transition(broken_engine, "some-candidate-id")
