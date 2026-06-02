"""Unit tests for FunnelRunLog recording in discovery (Task 3.4).

Tests record_discovery_run_log() and _determine_discovery_result_status()
to verify:
- FunnelRunLog entry is always created for discovery executions
- result_status is correctly determined based on outcome
- Empty discovery is recorded as "completed" (Requirement 1.8)
- Deterministic fallback is recorded as "degraded" with error_message (Requirement 2.3)
- sectors_completed and sectors_timed_out are stored as JSON arrays
- All required fields are populated

See: requirements 1.8, 1.12, 2.3, 2.5, 7.6
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db.schema import Base, FunnelRunLog, get_session
from utils.funnel_discovery import (
    DiscoveryResult,
    record_discovery_run_log,
    _determine_discovery_result_status,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_engine():
    """Create an in-memory SQLite engine with FunnelRunLog table."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def started_at() -> datetime:
    """A fixed started_at timestamp for tests."""
    return datetime(2026, 5, 27, 10, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Tests: _determine_discovery_result_status
# ---------------------------------------------------------------------------


class TestDetermineDiscoveryResultStatus:
    """Tests for result_status determination logic."""

    def test_chief_scout_success_is_completed(self):
        """Chief Scout successful curation → 'completed'."""
        result = DiscoveryResult(
            candidates=["NVDA", "TSLA"],
            sectors_completed=["tech", "energy"],
            selection_mode="chief_scout",
        )
        status = _determine_discovery_result_status(result, error_message=None)
        assert status == "completed"

    def test_empty_discovery_is_completed(self):
        """Empty discovery with no candidates but pipeline ran → 'completed'.

        Requirement 1.8: completed empty discovery = result_status='completed'.
        """
        result = DiscoveryResult(
            candidates=[],
            sectors_completed=["tech", "energy"],
            sectors_timed_out=[],
            selection_mode="deterministic_fallback",
            pipeline_budget_exhausted=False,
        )
        status = _determine_discovery_result_status(result, error_message=None)
        assert status == "completed"

    def test_deterministic_fallback_with_error_and_candidates_is_degraded(self):
        """Deterministic fallback used due to curation failure with candidates → 'degraded'.

        Requirement 2.3: deterministic fallback = 'degraded' with error_message.
        """
        result = DiscoveryResult(
            candidates=["NVDA", "TSLA"],
            sectors_completed=["tech"],
            selection_mode="deterministic_fallback",
        )
        status = _determine_discovery_result_status(
            result,
            error_message="Chief Scout curation timed out after 60.0s budget",
        )
        assert status == "degraded"

    def test_deterministic_fallback_without_error_is_completed(self):
        """Deterministic fallback used because budget < 10s (not a failure) → 'completed'.

        When no error occurred (budget was simply insufficient to attempt LLM),
        this is a normal completion path.
        """
        result = DiscoveryResult(
            candidates=["NVDA"],
            sectors_completed=["tech"],
            selection_mode="deterministic_fallback",
        )
        status = _determine_discovery_result_status(result, error_message=None)
        assert status == "completed"

    def test_all_sectors_skipped_no_candidates_is_timed_out(self):
        """All sectors skipped (budget exhausted before any complete) → 'timed_out'."""
        result = DiscoveryResult(
            candidates=[],
            sectors_completed=[],
            sectors_timed_out=[],
            sectors_skipped=["tech", "energy"],
            selection_mode="deterministic_fallback",
            pipeline_budget_exhausted=True,
        )
        status = _determine_discovery_result_status(
            result,
            error_message="Total pipeline budget exhausted before any sector completed",
        )
        assert status == "timed_out"

    def test_error_with_no_sectors_no_candidates_is_error(self):
        """Unexpected error with no sectors completed or timed out → 'error'."""
        result = DiscoveryResult(
            candidates=[],
            sectors_completed=[],
            sectors_timed_out=[],
            sectors_skipped=[],
            selection_mode="deterministic_fallback",
            pipeline_budget_exhausted=False,
        )
        status = _determine_discovery_result_status(
            result,
            error_message="FinnhubClient initialization failed: no API key",
        )
        assert status == "error"

    def test_partial_sectors_with_candidates_and_error_is_degraded(self):
        """Some sectors completed, curation failed, but candidates exist → 'degraded'."""
        result = DiscoveryResult(
            candidates=["NVDA"],
            sectors_completed=["tech"],
            sectors_timed_out=["energy"],
            selection_mode="deterministic_fallback",
            partial_screening=True,
        )
        status = _determine_discovery_result_status(
            result,
            error_message="Chief Scout curation failed: RuntimeError: API error",
        )
        assert status == "degraded"

    def test_chief_scout_with_empty_candidates_is_completed(self):
        """Chief Scout mode but zero candidates (curated to empty) → 'completed'."""
        result = DiscoveryResult(
            candidates=[],
            sectors_completed=["tech"],
            selection_mode="chief_scout",
        )
        status = _determine_discovery_result_status(result, error_message=None)
        assert status == "completed"


# ---------------------------------------------------------------------------
# Tests: record_discovery_run_log — integration with DB
# ---------------------------------------------------------------------------


class TestRecordDiscoveryRunLog:
    """Tests for FunnelRunLog persistence."""

    def test_creates_log_entry_for_successful_discovery(self, in_memory_engine, started_at):
        """A completed discovery creates a FunnelRunLog with correct fields."""
        result = DiscoveryResult(
            candidates=["NVDA", "TSLA", "AMD"],
            sectors_completed=["tech", "energy"],
            sectors_timed_out=[],
            sectors_skipped=[],
            selection_mode="chief_scout",
            total_duration_seconds=45.2,
            pipeline_budget_exhausted=False,
        )

        log = record_discovery_run_log(
            engine=in_memory_engine,
            result=result,
            started_at=started_at,
            budget_seconds=90.0,
            error_message=None,
        )

        assert log.stage == "discovery"
        assert log.result_status == "completed"
        assert log.duration_seconds == 45.2
        assert log.budget_seconds == 90.0
        assert log.candidates_input == 3
        assert log.candidates_promoted == 3
        assert log.error_message is None
        assert log.sectors_completed is not None
        assert json.loads(log.sectors_completed) == ["tech", "energy"]
        assert log.sectors_timed_out is None  # empty list → None

    def test_creates_log_entry_for_empty_discovery(self, in_memory_engine, started_at):
        """Empty discovery (no candidates) is 'completed' not 'timed_out'.

        Requirement 1.8: persist a completed empty discovery result.
        """
        result = DiscoveryResult(
            candidates=[],
            sectors_completed=["tech", "energy"],
            sectors_timed_out=[],
            sectors_skipped=[],
            selection_mode="deterministic_fallback",
            total_duration_seconds=12.0,
            pipeline_budget_exhausted=False,
        )

        log = record_discovery_run_log(
            engine=in_memory_engine,
            result=result,
            started_at=started_at,
            budget_seconds=90.0,
            error_message=None,
        )

        assert log.result_status == "completed"
        assert log.candidates_input == 0
        assert log.candidates_promoted == 0

    def test_creates_log_entry_for_degraded_fallback(self, in_memory_engine, started_at):
        """Deterministic fallback due to curation failure → 'degraded' with error.

        Requirement 2.3: record result_status='degraded', error_message with
        timeout duration or exception.
        """
        result = DiscoveryResult(
            candidates=["NVDA", "AMD"],
            sectors_completed=["tech"],
            sectors_timed_out=["energy"],
            sectors_skipped=[],
            selection_mode="deterministic_fallback",
            total_duration_seconds=91.3,
            pipeline_budget_exhausted=False,
            partial_screening=True,
        )

        error_msg = "Chief Scout curation timed out after 60.0s budget"
        log = record_discovery_run_log(
            engine=in_memory_engine,
            result=result,
            started_at=started_at,
            budget_seconds=90.0,
            error_message=error_msg,
        )

        assert log.result_status == "degraded"
        assert log.error_message == error_msg
        assert log.candidates_input == 2
        assert json.loads(log.sectors_timed_out) == ["energy"]
        assert json.loads(log.sectors_completed) == ["tech"]

    def test_sectors_stored_as_json_arrays(self, in_memory_engine, started_at):
        """sectors_completed and sectors_timed_out stored as JSON arrays in Text columns."""
        result = DiscoveryResult(
            candidates=["AAPL"],
            sectors_completed=["tech", "ev", "ai_semi"],
            sectors_timed_out=["energy", "crypto"],
            selection_mode="deterministic_fallback",
            total_duration_seconds=85.0,
        )

        log = record_discovery_run_log(
            engine=in_memory_engine,
            result=result,
            started_at=started_at,
            budget_seconds=90.0,
            error_message="Chief Scout failed: timeout",
        )

        # Parse JSON and verify arrays
        completed = json.loads(log.sectors_completed)
        timed_out = json.loads(log.sectors_timed_out)
        assert completed == ["tech", "ev", "ai_semi"]
        assert timed_out == ["energy", "crypto"]

    def test_log_persists_to_database(self, in_memory_engine, started_at):
        """FunnelRunLog entry is actually committed to the database."""
        result = DiscoveryResult(
            candidates=["NVDA"],
            sectors_completed=["tech"],
            selection_mode="chief_scout",
            total_duration_seconds=30.0,
        )

        record_discovery_run_log(
            engine=in_memory_engine,
            result=result,
            started_at=started_at,
            budget_seconds=90.0,
        )

        # Query the DB to verify persistence
        session = get_session(in_memory_engine)
        try:
            logs = session.query(FunnelRunLog).all()
            assert len(logs) == 1
            assert logs[0].stage == "discovery"
            assert logs[0].result_status == "completed"
            assert logs[0].candidates_input == 1
        finally:
            session.close()

    def test_started_at_and_ended_at_populated(self, in_memory_engine, started_at):
        """started_at and ended_at timestamps are recorded."""
        result = DiscoveryResult(
            candidates=[],
            sectors_completed=["tech"],
            selection_mode="deterministic_fallback",
            total_duration_seconds=5.0,
        )

        log = record_discovery_run_log(
            engine=in_memory_engine,
            result=result,
            started_at=started_at,
            budget_seconds=90.0,
        )

        assert log.started_at == started_at
        assert log.ended_at is not None
        assert log.ended_at >= started_at

    def test_date_is_ny_trading_date(self, in_memory_engine, started_at):
        """date field uses New York trading date."""
        result = DiscoveryResult(
            candidates=[],
            sectors_completed=[],
            selection_mode="deterministic_fallback",
            total_duration_seconds=1.0,
        )

        log = record_discovery_run_log(
            engine=in_memory_engine,
            result=result,
            started_at=started_at,
            budget_seconds=90.0,
        )

        # Date should be set (we can't assert exact value since it depends
        # on when the test runs, but it should be non-null)
        assert log.date is not None

    def test_budget_seconds_recorded(self, in_memory_engine, started_at):
        """budget_seconds reflects the configured total_pipeline_seconds."""
        result = DiscoveryResult(
            candidates=[],
            sectors_completed=["tech"],
            selection_mode="deterministic_fallback",
            total_duration_seconds=10.0,
        )

        log = record_discovery_run_log(
            engine=in_memory_engine,
            result=result,
            started_at=started_at,
            budget_seconds=120.0,  # custom budget
        )

        assert log.budget_seconds == 120.0
