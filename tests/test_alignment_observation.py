"""Tests for alignment observation mode (Requirements 19.1–19.5).

Validates:
- record_alignment_observation() persists observation events correctly
- should_enforce_alignment() requires both 5+ sessions AND explicit approval
- log_only mode records but does not enforce
"""

import json
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text

from utils.alignment_policy import (
    AlignmentOutcome,
    AlignmentResult,
    record_alignment_observation,
    should_enforce_alignment,
)


@pytest.fixture
def engine():
    """Create an in-memory SQLite engine with pm_candidate_events table."""
    eng = create_engine("sqlite:///:memory:")
    with eng.connect() as conn:
        conn.execute(text("""
            CREATE TABLE pm_candidate_events (
                id INTEGER PRIMARY KEY,
                candidate_id TEXT NOT NULL,
                cycle_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_data TEXT,
                created_at TEXT NOT NULL
            )
        """))
        conn.commit()
    return eng


@pytest.fixture
def sample_alignment_result():
    """A sample AlignmentResult for testing."""
    return AlignmentResult(
        outcome=AlignmentOutcome.REDUCE_SIZE,
        size_multiplier=0.5,
        rule_triggered="relative_strength_below_reduce (-2.10 < -1.5)",
        measurements_used={
            "sector_benchmark": "XLK",
            "sector_relative_strength": -2.1,
            "symbol_momentum": "neutral",
            "direction": "BUY",
        },
        benchmark_evaluated="XLK",
        mode="log_only",
        version="1.0.0",
    )


class TestRecordAlignmentObservation:
    """Tests for record_alignment_observation()."""

    def test_records_observation_event(self, engine, sample_alignment_result):
        """Observation is persisted to pm_candidate_events with correct event_type."""
        record_alignment_observation(
            engine=engine,
            candidate_id="cand-001",
            cycle_id="cycle-abc",
            profile_id="profile-1",
            alignment_result=sample_alignment_result,
            candidate_fate="executed",
        )

        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT * FROM pm_candidate_events WHERE event_type = 'alignment_observation'"
            )).fetchall()

        assert len(rows) == 1
        row = rows[0]
        assert row[1] == "cand-001"  # candidate_id
        assert row[2] == "cycle-abc"  # cycle_id
        assert row[3] == "profile-1"  # profile_id
        assert row[4] == "alignment_observation"  # event_type

    def test_observation_data_contains_all_required_fields(self, engine, sample_alignment_result):
        """Observation data includes proposed outcome, multiplier, measurements,
        candidate fate, and placeholders for false-positive/negative (Req 19.2)."""
        record_alignment_observation(
            engine=engine,
            candidate_id="cand-002",
            cycle_id="cycle-xyz",
            profile_id="profile-2",
            alignment_result=sample_alignment_result,
            candidate_fate="gate_rejected",
            realized_outcome={"pnl_pct": -1.5, "r_multiple": -0.5, "win": False},
        )

        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT event_data FROM pm_candidate_events WHERE candidate_id = 'cand-002'"
            )).fetchone()

        data = json.loads(row[0])
        assert data["proposed_outcome"] == "reduce_size"
        assert data["proposed_multiplier"] == 0.5
        assert data["rule_triggered"] == "relative_strength_below_reduce (-2.10 < -1.5)"
        assert data["measurements_used"]["sector_benchmark"] == "XLK"
        assert data["benchmark_evaluated"] == "XLK"
        assert data["mode"] == "log_only"
        assert data["version"] == "1.0.0"
        assert data["candidate_fate"] == "gate_rejected"
        assert data["realized_outcome"] == {"pnl_pct": -1.5, "r_multiple": -0.5, "win": False}
        assert "false_positive" in data
        assert "false_negative" in data

    def test_observation_without_realized_outcome(self, engine, sample_alignment_result):
        """Observation can be recorded before post-trade outcome is available."""
        record_alignment_observation(
            engine=engine,
            candidate_id="cand-003",
            cycle_id="cycle-1",
            profile_id="profile-1",
            alignment_result=sample_alignment_result,
            candidate_fate="not_selected",
            realized_outcome=None,
        )

        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT event_data FROM pm_candidate_events WHERE candidate_id = 'cand-003'"
            )).fetchone()

        data = json.loads(row[0])
        assert data["realized_outcome"] is None

    def test_observation_gracefully_handles_db_error(self, sample_alignment_result):
        """DB write failure is logged as warning, does not raise (best-effort)."""
        # Use a broken engine that will fail on connect
        broken_engine = create_engine("sqlite:///nonexistent_dir/fake.db")

        # Should not raise — logs a warning instead
        record_alignment_observation(
            engine=broken_engine,
            candidate_id="cand-004",
            cycle_id="cycle-1",
            profile_id="profile-1",
            alignment_result=sample_alignment_result,
            candidate_fate="executed",
        )


class TestShouldEnforceAlignment:
    """Tests for should_enforce_alignment()."""

    def test_returns_false_without_approval(self, engine):
        """Without explicit operator approval, enforcement is blocked (Req 19.5)."""
        with patch.dict(os.environ, {"PM_ALIGNMENT_ENFORCEMENT_APPROVED": "false"}):
            assert should_enforce_alignment(engine) is False

    def test_returns_false_with_approval_but_insufficient_sessions(self, engine):
        """With approval but <5 sessions, enforcement is blocked (Req 19.3)."""
        # Insert observations for only 3 distinct dates
        with engine.connect() as conn:
            for i in range(3):
                date = (datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)).isoformat()
                conn.execute(text("""
                    INSERT INTO pm_candidate_events
                    (candidate_id, cycle_id, profile_id, event_type, event_data, created_at)
                    VALUES (:cid, :cycle_id, :profile_id, :event_type, :event_data, :created_at)
                """), {
                    "cid": f"cand-{i}",
                    "cycle_id": "cycle-1",
                    "profile_id": "profile-1",
                    "event_type": "alignment_observation",
                    "event_data": "{}",
                    "created_at": date,
                })
            conn.commit()

        with patch.dict(os.environ, {"PM_ALIGNMENT_ENFORCEMENT_APPROVED": "true"}):
            assert should_enforce_alignment(engine) is False

    def test_returns_true_with_approval_and_five_sessions(self, engine):
        """With approval AND 5+ distinct session dates, enforcement is allowed."""
        with engine.connect() as conn:
            for i in range(5):
                date = (datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)).isoformat()
                conn.execute(text("""
                    INSERT INTO pm_candidate_events
                    (candidate_id, cycle_id, profile_id, event_type, event_data, created_at)
                    VALUES (:cid, :cycle_id, :profile_id, :event_type, :event_data, :created_at)
                """), {
                    "cid": f"cand-{i}",
                    "cycle_id": "cycle-1",
                    "profile_id": "profile-1",
                    "event_type": "alignment_observation",
                    "event_data": "{}",
                    "created_at": date,
                })
            conn.commit()

        with patch.dict(os.environ, {"PM_ALIGNMENT_ENFORCEMENT_APPROVED": "true"}):
            assert should_enforce_alignment(engine) is True

    def test_multiple_observations_same_day_count_as_one_session(self, engine):
        """Multiple observations on the same date = 1 trading session."""
        with engine.connect() as conn:
            # Insert 10 observations but all on the same date
            for i in range(10):
                conn.execute(text("""
                    INSERT INTO pm_candidate_events
                    (candidate_id, cycle_id, profile_id, event_type, event_data, created_at)
                    VALUES (:cid, :cycle_id, :profile_id, :event_type, :event_data, :created_at)
                """), {
                    "cid": f"cand-{i}",
                    "cycle_id": "cycle-1",
                    "profile_id": "profile-1",
                    "event_type": "alignment_observation",
                    "event_data": "{}",
                    "created_at": "2025-01-01T10:00:00+00:00",
                })
            conn.commit()

        with patch.dict(os.environ, {"PM_ALIGNMENT_ENFORCEMENT_APPROVED": "true"}):
            # Only 1 distinct date, so should not enforce
            assert should_enforce_alignment(engine) is False

    def test_returns_false_when_approval_env_missing(self, engine):
        """Missing PM_ALIGNMENT_ENFORCEMENT_APPROVED defaults to not approved."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure the var is not set
            os.environ.pop("PM_ALIGNMENT_ENFORCEMENT_APPROVED", None)
            assert should_enforce_alignment(engine) is False

    def test_ignores_non_observation_events(self, engine):
        """Only alignment_observation events count toward session threshold."""
        with engine.connect() as conn:
            for i in range(10):
                date = (datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)).isoformat()
                conn.execute(text("""
                    INSERT INTO pm_candidate_events
                    (candidate_id, cycle_id, profile_id, event_type, event_data, created_at)
                    VALUES (:cid, :cycle_id, :profile_id, :event_type, :event_data, :created_at)
                """), {
                    "cid": f"cand-{i}",
                    "cycle_id": "cycle-1",
                    "profile_id": "profile-1",
                    "event_type": "pm_accept",  # Not an observation event
                    "event_data": "{}",
                    "created_at": date,
                })
            conn.commit()

        with patch.dict(os.environ, {"PM_ALIGNMENT_ENFORCEMENT_APPROVED": "true"}):
            assert should_enforce_alignment(engine) is False
