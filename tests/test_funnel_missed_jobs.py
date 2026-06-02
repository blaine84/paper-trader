"""Tests for funnel missed-job detection and error handling in orchestrator.

Validates:
- Missed funnel jobs create FunnelRunLog entries with result_status="error"
  and error_message="missed scheduled start at {time}"
- Unhandled exceptions in funnel jobs log to FunnelRunLog and don't propagate
- PM cycle blocks new funnel jobs but allows in-progress ones to complete
- _check_missed_funnel_jobs() correctly identifies which jobs were missed

Requirements: 12.7, 12.8, 7.5
"""

import pytest
from unittest.mock import patch, MagicMock, call
from datetime import datetime, date

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_funnel_config():
    """Standard funnel config for testing."""
    return {
        "funnel": {
            "enabled": True,
            "schedule": {
                "discovery_time": "06:00",
                "research_time": "06:30",
                "analysis_time": "07:15",
                "confirmation_time": "09:35",
            },
            "ceilings": {
                "max_discovery_shortlist": 5,
                "max_researcher_promoted": 3,
                "max_pm_handoff": 3,
            },
            "budgets": {
                "per_sector_seconds": 15,
                "total_pipeline_seconds": 90,
                "confirmation_budget_seconds": 45,
                "market_hours_confirmation_budget_seconds": 60,
            },
        }
    }


class TestLogMissedFunnelJob:
    """Tests for _log_missed_funnel_job helper."""

    def test_creates_error_log_with_missed_message(self):
        """Creates FunnelRunLog with result_status='error' and correct error_message."""
        from orchestrator import _log_missed_funnel_job

        mock_engine = MagicMock()
        mock_session = MagicMock()

        with patch("db.schema.get_session", return_value=mock_session):
            _log_missed_funnel_job(
                mock_engine, stage="discovery",
                scheduled_time_str="06:00 ET", budget_seconds=90
            )

        # Verify the FunnelRunLog was added with correct fields
        mock_session.add.assert_called_once()
        log_entry = mock_session.add.call_args[0][0]
        assert log_entry.stage == "discovery"
        assert log_entry.result_status == "error"
        assert log_entry.error_message == "missed scheduled start at 06:00 ET"
        assert log_entry.budget_seconds == 90
        assert log_entry.candidates_input == 0
        assert log_entry.candidates_promoted == 0
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    def test_does_not_propagate_db_errors(self):
        """If DB write fails, exception is swallowed (never propagates)."""
        from orchestrator import _log_missed_funnel_job

        mock_engine = MagicMock()

        with patch("db.schema.get_session", side_effect=Exception("DB locked")):
            # Must NOT raise
            _log_missed_funnel_job(
                mock_engine, stage="research",
                scheduled_time_str="06:30 ET", budget_seconds=90
            )


class TestLogFunnelJobError:
    """Tests for _log_funnel_job_error helper."""

    def test_creates_error_log_with_exception_info(self):
        """Creates FunnelRunLog with exception type and message."""
        from orchestrator import _log_funnel_job_error

        mock_engine = MagicMock()
        mock_session = MagicMock()

        with patch("db.schema.get_session", return_value=mock_session):
            error = RuntimeError("LLM connection refused")
            _log_funnel_job_error(
                mock_engine, stage="discovery",
                budget_seconds=90, error=error
            )

        mock_session.add.assert_called_once()
        log_entry = mock_session.add.call_args[0][0]
        assert log_entry.stage == "discovery"
        assert log_entry.result_status == "error"
        assert "RuntimeError" in log_entry.error_message
        assert "LLM connection refused" in log_entry.error_message
        assert log_entry.budget_seconds == 90
        mock_session.commit.assert_called_once()

    def test_truncates_long_error_messages(self):
        """Error messages longer than 500 chars are truncated."""
        from orchestrator import _log_funnel_job_error

        mock_engine = MagicMock()
        mock_session = MagicMock()

        with patch("db.schema.get_session", return_value=mock_session):
            error = RuntimeError("x" * 1000)
            _log_funnel_job_error(
                mock_engine, stage="analysis",
                budget_seconds=90, error=error
            )

        log_entry = mock_session.add.call_args[0][0]
        # "RuntimeError: " + 500 chars max of the error text
        assert len(log_entry.error_message) <= len("RuntimeError: ") + 500

    def test_does_not_propagate_db_errors(self):
        """If DB write fails, exception is swallowed (never propagates)."""
        from orchestrator import _log_funnel_job_error

        mock_engine = MagicMock()

        with patch("db.schema.get_session", side_effect=Exception("DB locked")):
            error = RuntimeError("original error")
            # Must NOT raise
            _log_funnel_job_error(
                mock_engine, stage="confirmation",
                budget_seconds=45, error=error
            )


class TestCheckMissedFunnelJobs:
    """Tests for _check_missed_funnel_jobs startup detection."""

    @patch("orchestrator.is_trading_day", return_value=True)
    @patch("orchestrator._log_missed_funnel_job")
    def test_detects_missed_discovery_job(
        self, mock_log_missed, mock_trading_day, sample_funnel_config
    ):
        """When orchestrator starts at 07:00 ET, discovery (06:00) is detected as missed."""
        from orchestrator import _check_missed_funnel_jobs
        from zoneinfo import ZoneInfo

        mock_engine = MagicMock()
        mock_session = MagicMock()

        # No existing logs for today
        mock_session.query.return_value.filter.return_value.all.return_value = []

        now_ny = datetime(2025, 6, 2, 7, 0, 0, tzinfo=ZoneInfo("America/New_York"))

        with patch("orchestrator.datetime") as mock_dt:
            mock_dt.now.return_value = now_ny
            with patch("db.schema.get_session", return_value=mock_session):
                _check_missed_funnel_jobs(mock_engine, sample_funnel_config)

        # Should log missed discovery (06:00 < 07:00) and research (06:30 < 07:00)
        assert mock_log_missed.called
        # Find the call for discovery
        stages_logged = [c[0][1] for c in mock_log_missed.call_args_list]
        assert "discovery" in stages_logged
        assert "research" in stages_logged
        # analysis (07:15) should NOT be logged (07:00 < 07:15)
        assert "analysis" not in stages_logged

    @patch("orchestrator.is_trading_day", return_value=True)
    @patch("orchestrator._log_missed_funnel_job")
    def test_detects_all_missed_jobs_when_started_late(
        self, mock_log_missed, mock_trading_day, sample_funnel_config
    ):
        """When orchestrator starts at 10:00 ET, all premarket jobs are missed."""
        from orchestrator import _check_missed_funnel_jobs
        from zoneinfo import ZoneInfo

        mock_engine = MagicMock()
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.all.return_value = []

        now_ny = datetime(2025, 6, 2, 10, 0, 0, tzinfo=ZoneInfo("America/New_York"))

        with patch("orchestrator.datetime") as mock_dt:
            mock_dt.now.return_value = now_ny
            with patch("db.schema.get_session", return_value=mock_session):
                _check_missed_funnel_jobs(mock_engine, sample_funnel_config)

        # All 4 jobs should be missed (06:00, 06:30, 07:15, 09:35 < 10:00)
        assert mock_log_missed.call_count == 4

    @patch("orchestrator.is_trading_day", return_value=True)
    @patch("orchestrator._log_missed_funnel_job")
    def test_does_not_log_future_jobs(
        self, mock_log_missed, mock_trading_day, sample_funnel_config
    ):
        """When orchestrator starts at 05:45 ET, no jobs are missed."""
        from orchestrator import _check_missed_funnel_jobs
        from zoneinfo import ZoneInfo

        mock_engine = MagicMock()
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.all.return_value = []

        now_ny = datetime(2025, 6, 2, 5, 45, 0, tzinfo=ZoneInfo("America/New_York"))

        with patch("orchestrator.datetime") as mock_dt:
            mock_dt.now.return_value = now_ny
            with patch("db.schema.get_session", return_value=mock_session):
                _check_missed_funnel_jobs(mock_engine, sample_funnel_config)

        # No jobs missed yet
        mock_log_missed.assert_not_called()

    @patch("orchestrator.is_trading_day", return_value=True)
    @patch("orchestrator._log_missed_funnel_job")
    def test_skips_jobs_with_existing_logs(
        self, mock_log_missed, mock_trading_day, sample_funnel_config
    ):
        """Jobs that already have FunnelRunLog entries are not logged as missed."""
        from orchestrator import _check_missed_funnel_jobs
        from zoneinfo import ZoneInfo

        mock_engine = MagicMock()
        mock_session = MagicMock()

        # Discovery already has a log entry
        mock_row = MagicMock()
        mock_row.stage = "discovery"
        mock_session.query.return_value.filter.return_value.all.return_value = [mock_row]

        now_ny = datetime(2025, 6, 2, 7, 0, 0, tzinfo=ZoneInfo("America/New_York"))

        with patch("orchestrator.datetime") as mock_dt:
            mock_dt.now.return_value = now_ny
            with patch("db.schema.get_session", return_value=mock_session):
                _check_missed_funnel_jobs(mock_engine, sample_funnel_config)

        # Discovery should NOT be logged as missed (it already has a log)
        stages_logged = [c[0][1] for c in mock_log_missed.call_args_list]
        assert "discovery" not in stages_logged
        # Research (06:30 < 07:00) should be logged since it has no log
        assert "research" in stages_logged

    @patch("orchestrator.is_trading_day", return_value=False)
    @patch("orchestrator._log_missed_funnel_job")
    def test_skips_non_trading_days(
        self, mock_log_missed, mock_trading_day, sample_funnel_config
    ):
        """Does not log missed jobs on weekends/holidays."""
        from orchestrator import _check_missed_funnel_jobs
        from zoneinfo import ZoneInfo

        mock_engine = MagicMock()

        now_ny = datetime(2025, 6, 1, 10, 0, 0, tzinfo=ZoneInfo("America/New_York"))

        with patch("orchestrator.datetime") as mock_dt:
            mock_dt.now.return_value = now_ny
            _check_missed_funnel_jobs(mock_engine, sample_funnel_config)

        mock_log_missed.assert_not_called()


class TestPMCycleBlocking:
    """Tests for PM cycle blocking funnel jobs.

    Requirement 7.5: In-progress premarket job completes within its budget
    when first PM cycle is scheduled, but block new funnel jobs until PM
    cycle completes.
    """

    def test_pm_cycle_blocks_funnel_discovery(self):
        """When PM cycle is active, funnel_discovery is skipped."""
        from orchestrator import (
            run_funnel_discovery_job,
            _set_pm_cycle_active,
            _is_pm_cycle_blocking_funnel,
        )

        _set_pm_cycle_active(True)
        try:
            assert _is_pm_cycle_blocking_funnel() is True

            # Patch the market-closed check to allow execution
            with patch("orchestrator._skip_closed_market_job", return_value=False):
                with patch("orchestrator.get_engine") as mock_eng:
                    with patch("orchestrator.run_funnel_discovery") as mock_disc:
                        run_funnel_discovery_job()
                        # Discovery should NOT be called because PM is blocking
                        mock_disc.assert_not_called()
        finally:
            _set_pm_cycle_active(False)

    def test_pm_cycle_blocks_funnel_research(self):
        """When PM cycle is active, funnel_research is skipped."""
        from orchestrator import run_funnel_research_job, _set_pm_cycle_active

        _set_pm_cycle_active(True)
        try:
            with patch("orchestrator._skip_closed_market_job", return_value=False):
                with patch("orchestrator.run_funnel_qualification") as mock_qual:
                    run_funnel_research_job()
                    mock_qual.assert_not_called()
        finally:
            _set_pm_cycle_active(False)

    def test_pm_cycle_blocks_funnel_analysis(self):
        """When PM cycle is active, funnel_analysis is skipped."""
        from orchestrator import run_funnel_analysis_job, _set_pm_cycle_active

        _set_pm_cycle_active(True)
        try:
            with patch("orchestrator._skip_closed_market_job", return_value=False):
                with patch("orchestrator.run_funnel_analysis") as mock_anal:
                    run_funnel_analysis_job()
                    mock_anal.assert_not_called()
        finally:
            _set_pm_cycle_active(False)

    def test_pm_cycle_blocks_funnel_confirmation(self):
        """When PM cycle is active, funnel_confirmation is skipped."""
        from orchestrator import run_funnel_confirmation_job, _set_pm_cycle_active

        _set_pm_cycle_active(True)
        try:
            with patch("orchestrator._skip_closed_market_job", return_value=False):
                with patch("orchestrator.run_opening_confirmation") as mock_conf:
                    run_funnel_confirmation_job()
                    mock_conf.assert_not_called()
        finally:
            _set_pm_cycle_active(False)

    def test_pm_cycle_blocks_funnel_confirmation_retry(self):
        """When PM cycle is active, funnel_confirmation_retry is skipped."""
        from orchestrator import run_funnel_confirmation_retry_job, _set_pm_cycle_active

        _set_pm_cycle_active(True)
        try:
            with patch("orchestrator._skip_closed_market_job", return_value=False):
                with patch("orchestrator.run_confirmation_retry") as mock_retry:
                    run_funnel_confirmation_retry_job()
                    mock_retry.assert_not_called()
        finally:
            _set_pm_cycle_active(False)

    def test_funnel_runs_when_pm_not_active(self):
        """When PM cycle is not active, funnel jobs are not blocked."""
        from orchestrator import _set_pm_cycle_active, _is_pm_cycle_blocking_funnel

        _set_pm_cycle_active(False)
        assert _is_pm_cycle_blocking_funnel() is False

    def test_run_intraday_sets_and_clears_pm_flag(self):
        """run_intraday() sets _pm_cycle_active during execution and clears after."""
        from orchestrator import run_intraday, _is_pm_cycle_blocking_funnel

        flag_during_execution = []

        def capture_flag(*args, **kwargs):
            flag_during_execution.append(_is_pm_cycle_blocking_funnel())

        with patch("orchestrator._skip_outside_regular_market_job", return_value=False):
            with patch("orchestrator.get_engine", return_value=MagicMock()):
                with patch("orchestrator._run_intraday_inner", side_effect=capture_flag):
                    run_intraday()

        # Flag should have been True during execution
        assert flag_during_execution == [True]
        # Flag should be False after
        assert _is_pm_cycle_blocking_funnel() is False

    def test_run_intraday_clears_flag_on_exception(self):
        """run_intraday() clears _pm_cycle_active even if inner function raises."""
        from orchestrator import run_intraday, _is_pm_cycle_blocking_funnel

        with patch("orchestrator._skip_outside_regular_market_job", return_value=False):
            with patch("orchestrator.get_engine", return_value=MagicMock()):
                with patch("orchestrator._run_intraday_inner", side_effect=RuntimeError("crash")):
                    with pytest.raises(RuntimeError):
                        run_intraday()

        # Flag must be cleared even on exception
        assert _is_pm_cycle_blocking_funnel() is False


class TestFunnelJobErrorLogging:
    """Tests that funnel jobs log to FunnelRunLog on unhandled exceptions.

    Requirement 12.8: Unhandled exceptions are caught at the job level,
    logged to FunnelRunLog, and other jobs continue.
    """

    @patch("orchestrator._log_funnel_job_error")
    @patch("orchestrator._is_pm_cycle_blocking_funnel", return_value=False)
    @patch("orchestrator._skip_closed_market_job", return_value=False)
    def test_discovery_job_logs_error_on_exception(
        self, mock_skip, mock_pm, mock_log_error
    ):
        """Discovery job catches exceptions and logs to FunnelRunLog."""
        from orchestrator import run_funnel_discovery_job

        with patch("orchestrator.get_engine", return_value=MagicMock()):
            with patch("utils.funnel_orchestrator.is_discovery_allowed", return_value=True):
                with patch(
                    "utils.sector_scout.load_sector_scout_config",
                    side_effect=RuntimeError("config error"),
                ):
                    # Must NOT raise
                    run_funnel_discovery_job()

        mock_log_error.assert_called_once()
        call_kwargs = mock_log_error.call_args
        # Check stage is "discovery"
        assert "discovery" in str(call_kwargs)

    @patch("orchestrator._log_funnel_job_error")
    @patch("orchestrator._is_pm_cycle_blocking_funnel", return_value=False)
    @patch("orchestrator._skip_closed_market_job", return_value=False)
    def test_confirmation_retry_logs_error_on_exception(
        self, mock_skip, mock_pm, mock_log_error
    ):
        """Confirmation retry job catches exceptions and logs to FunnelRunLog."""
        from orchestrator import run_funnel_confirmation_retry_job

        with patch("orchestrator.get_engine", return_value=MagicMock()):
            with patch(
                "orchestrator.run_confirmation_retry",
                side_effect=ConnectionError("data feed down"),
            ):
                run_funnel_confirmation_retry_job()

        mock_log_error.assert_called_once()

    @patch("orchestrator._is_pm_cycle_blocking_funnel", return_value=False)
    @patch("orchestrator._skip_closed_market_job", return_value=False)
    def test_funnel_jobs_never_propagate_exceptions(self, mock_skip, mock_pm):
        """All funnel job functions catch exceptions — none propagate to caller."""
        from orchestrator import (
            run_funnel_discovery_job,
            run_funnel_research_job,
            run_funnel_analysis_job,
            run_funnel_confirmation_job,
            run_funnel_confirmation_retry_job,
        )

        with patch("orchestrator.get_engine", return_value=MagicMock()):
            with patch("orchestrator._log_funnel_job_error"):
                # Each should NOT raise
                with patch("utils.funnel_orchestrator.is_discovery_allowed", return_value=True):
                    with patch(
                        "utils.sector_scout.load_sector_scout_config",
                        side_effect=Exception("fatal"),
                    ):
                        run_funnel_discovery_job()  # no raise

                with patch("orchestrator.load_funnel_config", side_effect=Exception("fatal")):
                    run_funnel_research_job()  # no raise

                with patch("orchestrator.load_funnel_config", return_value={"funnel": {"budgets": {}, "ceilings": {}}}):
                    with patch("db.schema.get_session", side_effect=Exception("fatal")):
                        run_funnel_analysis_job()  # no raise

                with patch("orchestrator.load_funnel_config", side_effect=Exception("fatal")):
                    run_funnel_confirmation_job()  # no raise

                with patch(
                    "orchestrator.run_confirmation_retry",
                    side_effect=Exception("fatal"),
                ):
                    run_funnel_confirmation_retry_job()  # no raise
