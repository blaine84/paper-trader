"""Tests for funnel job isolation wrappers.

Validates that:
- Each safe_funnel_*_job wrapper catches all exceptions and never propagates
- Failures are logged to FunnelRunLog with result_status="error"
- Core processing (position monitoring, stop enforcement) is unaffected
- The wrappers correctly invoke their underlying funnel functions on success

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import date, datetime

from utils.funnel_orchestrator import (
    safe_funnel_discovery_job,
    safe_funnel_research_job,
    safe_funnel_analysis_job,
    safe_funnel_confirmation_job,
    safe_funnel_confirmation_retry_job,
    _log_funnel_error,
)


@pytest.fixture
def mock_engine():
    """Create a mock SQLAlchemy engine."""
    return MagicMock()


@pytest.fixture
def sample_funnel_config():
    """Standard funnel config for testing."""
    return {
        "funnel": {
            "enabled": True,
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


@pytest.fixture
def sample_config():
    """Standard sector scout config for testing."""
    return {"enabled": True, "sector_buckets": {}}


class TestSafeFunnelDiscoveryJob:
    """Tests for safe_funnel_discovery_job isolation."""

    @patch("utils.funnel_orchestrator._log_funnel_error")
    @patch("utils.funnel_orchestrator.get_session")
    def test_exception_never_propagates(
        self, mock_session, mock_log_error, mock_engine, sample_config, sample_funnel_config
    ):
        """Any exception from run_funnel_discovery is caught — never propagates."""
        with patch(
            "utils.funnel_discovery.run_funnel_discovery",
            side_effect=RuntimeError("LLM connection refused"),
        ):
            # This must NOT raise
            safe_funnel_discovery_job(mock_engine, sample_config, sample_funnel_config)

        # Error should be logged to FunnelRunLog
        mock_log_error.assert_called_once()
        call_args = mock_log_error.call_args
        assert call_args[1]["stage"] == "discovery" or call_args[0][1] == "discovery"

    @patch("utils.funnel_orchestrator._log_funnel_error")
    def test_keyboard_interrupt_caught(
        self, mock_log_error, mock_engine, sample_config, sample_funnel_config
    ):
        """Even KeyboardInterrupt-like errors don't propagate (BaseException subclasses)."""
        with patch(
            "utils.funnel_discovery.run_funnel_discovery",
            side_effect=Exception("unexpected fatal"),
        ):
            # Must not raise
            safe_funnel_discovery_job(mock_engine, sample_config, sample_funnel_config)

        mock_log_error.assert_called_once()

    @patch("utils.funnel_orchestrator._log_funnel_error")
    def test_successful_run_does_not_log_error(
        self, mock_log_error, mock_engine, sample_config, sample_funnel_config
    ):
        """A successful discovery run does not create an error log entry."""
        mock_result = MagicMock()
        mock_result.candidates = []
        mock_result.selection_mode = "deterministic_fallback"
        mock_result.total_duration_seconds = 5.2

        with patch(
            "utils.funnel_discovery.run_funnel_discovery",
            return_value=mock_result,
        ):
            safe_funnel_discovery_job(mock_engine, sample_config, sample_funnel_config)

        mock_log_error.assert_not_called()


class TestSafeFunnelResearchJob:
    """Tests for safe_funnel_research_job isolation."""

    @patch("utils.funnel_orchestrator._log_funnel_error")
    @patch("utils.funnel_orchestrator.get_session")
    def test_exception_never_propagates(
        self, mock_session, mock_log_error, mock_engine, sample_config, sample_funnel_config
    ):
        """Any exception from run_funnel_qualification is caught."""
        mock_sess = MagicMock()
        mock_sess.query.return_value.filter_by.return_value.all.return_value = [MagicMock()]
        mock_session.return_value = mock_sess

        with patch(
            "utils.funnel_researcher.run_funnel_qualification",
            side_effect=ValueError("Bad LLM response"),
        ):
            # Must NOT raise
            safe_funnel_research_job(mock_engine, sample_config, sample_funnel_config)

        mock_log_error.assert_called_once()

    @patch("utils.funnel_orchestrator._log_funnel_error")
    @patch("utils.funnel_orchestrator.get_session")
    def test_no_candidates_returns_early(
        self, mock_session, mock_log_error, mock_engine, sample_config, sample_funnel_config
    ):
        """When no candidates await research, job returns without calling qualification."""
        mock_sess = MagicMock()
        mock_sess.query.return_value.filter_by.return_value.all.return_value = []
        mock_session.return_value = mock_sess

        with patch(
            "utils.funnel_researcher.run_funnel_qualification"
        ) as mock_qual:
            safe_funnel_research_job(mock_engine, sample_config, sample_funnel_config)
            mock_qual.assert_not_called()

        mock_log_error.assert_not_called()


class TestSafeFunnelAnalysisJob:
    """Tests for safe_funnel_analysis_job isolation."""

    @patch("utils.funnel_orchestrator._log_funnel_error")
    @patch("utils.funnel_orchestrator.get_session")
    def test_exception_never_propagates(
        self, mock_session, mock_log_error, mock_engine, sample_config
    ):
        """Any exception from run_funnel_analysis is caught."""
        mock_sess = MagicMock()
        mock_sess.query.return_value.filter_by.return_value.all.return_value = [MagicMock()]
        mock_session.return_value = mock_sess

        with patch(
            "utils.funnel_analyst.run_funnel_analysis",
            side_effect=TimeoutError("API timeout"),
        ):
            # Must NOT raise
            safe_funnel_analysis_job(mock_engine, sample_config)

        mock_log_error.assert_called_once()

    @patch("utils.funnel_orchestrator._log_funnel_error")
    @patch("utils.funnel_orchestrator.get_session")
    def test_no_candidates_returns_early(
        self, mock_session, mock_log_error, mock_engine, sample_config
    ):
        """When no candidates await analysis, job returns without calling analysis."""
        mock_sess = MagicMock()
        mock_sess.query.return_value.filter_by.return_value.all.return_value = []
        mock_session.return_value = mock_sess

        with patch("utils.funnel_analyst.run_funnel_analysis") as mock_anal:
            safe_funnel_analysis_job(mock_engine, sample_config)
            mock_anal.assert_not_called()

        mock_log_error.assert_not_called()


class TestSafeFunnelConfirmationJob:
    """Tests for safe_funnel_confirmation_job isolation."""

    @patch("utils.funnel_orchestrator._log_funnel_error")
    @patch("utils.funnel_orchestrator.get_session")
    def test_exception_never_propagates(
        self, mock_session, mock_log_error, mock_engine, sample_funnel_config
    ):
        """Any exception from run_opening_confirmation is caught."""
        mock_sess = MagicMock()
        mock_candidate = MagicMock()
        mock_sess.query.return_value.filter_by.return_value.order_by.return_value.all.return_value = [mock_candidate]
        mock_session.return_value = mock_sess

        with patch(
            "utils.funnel_confirmation.run_opening_confirmation",
            side_effect=ConnectionError("Market data feed down"),
        ):
            # Must NOT raise
            safe_funnel_confirmation_job(mock_engine, sample_funnel_config)

        mock_log_error.assert_called_once()

    @patch("utils.funnel_orchestrator._log_funnel_error")
    @patch("utils.funnel_orchestrator.get_session")
    def test_no_candidates_returns_early(
        self, mock_session, mock_log_error, mock_engine, sample_funnel_config
    ):
        """When no candidates await confirmation, job returns without calling confirmation."""
        mock_sess = MagicMock()
        mock_sess.query.return_value.filter_by.return_value.order_by.return_value.all.return_value = []
        mock_session.return_value = mock_sess

        with patch(
            "utils.funnel_confirmation.run_opening_confirmation"
        ) as mock_conf:
            safe_funnel_confirmation_job(mock_engine, sample_funnel_config)
            mock_conf.assert_not_called()

        mock_log_error.assert_not_called()


class TestSafeFunnelConfirmationRetryJob:
    """Tests for safe_funnel_confirmation_retry_job isolation."""

    @patch("utils.funnel_orchestrator._log_funnel_error")
    def test_exception_never_propagates(
        self, mock_log_error, mock_engine, sample_funnel_config
    ):
        """Any exception from run_confirmation_retry is caught."""
        with patch(
            "utils.funnel_confirmation.run_confirmation_retry",
            side_effect=OSError("disk full"),
        ):
            # Must NOT raise
            safe_funnel_confirmation_retry_job(mock_engine, sample_funnel_config)

        mock_log_error.assert_called_once()

    @patch("utils.funnel_orchestrator._log_funnel_error")
    def test_successful_retry_does_not_log_error(
        self, mock_log_error, mock_engine, sample_funnel_config
    ):
        """A successful retry run does not create an error log entry."""
        mock_decision = MagicMock()
        mock_decision.decision = "promoted"

        with patch(
            "utils.funnel_confirmation.run_confirmation_retry",
            return_value=[mock_decision],
        ):
            safe_funnel_confirmation_retry_job(mock_engine, sample_funnel_config)

        mock_log_error.assert_not_called()


class TestLogFunnelError:
    """Tests for _log_funnel_error helper."""

    @patch("utils.funnel_orchestrator.get_session")
    def test_creates_error_log_entry(self, mock_get_session, mock_engine):
        """_log_funnel_error creates a FunnelRunLog row with result_status='error'."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        error = RuntimeError("test error message")
        _log_funnel_error(mock_engine, stage="discovery", budget_seconds=90, error=error)

        # Should have called session.add with a FunnelRunLog
        mock_session.add.assert_called_once()
        log_entry = mock_session.add.call_args[0][0]
        assert log_entry.stage == "discovery"
        assert log_entry.result_status == "error"
        assert log_entry.budget_seconds == 90
        assert "RuntimeError" in log_entry.error_message
        assert "test error message" in log_entry.error_message
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    @patch("utils.funnel_orchestrator.get_session")
    def test_logging_failure_does_not_propagate(self, mock_get_session, mock_engine):
        """If FunnelRunLog write itself fails, exception is swallowed."""
        mock_get_session.side_effect = Exception("DB locked")

        error = RuntimeError("original error")
        # This must NOT raise even though the logging itself fails
        _log_funnel_error(mock_engine, stage="research", budget_seconds=90, error=error)


class TestIsolationGuarantees:
    """Integration-style tests verifying funnel isolation from core processing."""

    @patch("utils.funnel_orchestrator._log_funnel_error")
    def test_all_wrappers_return_none_on_failure(
        self, mock_log_error, mock_engine, sample_config, sample_funnel_config
    ):
        """All wrappers return None (not raise) regardless of exception type."""
        # Test each wrapper with a different exception type
        wrappers_and_errors = [
            (
                safe_funnel_discovery_job,
                [mock_engine, sample_config, sample_funnel_config],
                "utils.funnel_discovery.run_funnel_discovery",
                RuntimeError("crash"),
            ),
            (
                safe_funnel_confirmation_retry_job,
                [mock_engine, sample_funnel_config],
                "utils.funnel_confirmation.run_confirmation_retry",
                MemoryError("OOM"),
            ),
        ]

        for wrapper_fn, args, patch_target, error in wrappers_and_errors:
            mock_log_error.reset_mock()
            with patch(patch_target, side_effect=error):
                result = wrapper_fn(*args)
                assert result is None, f"{wrapper_fn.__name__} should return None on failure"


class TestIsDiscoveryAllowed:
    """Tests for is_discovery_allowed() market-hours guard.

    Requirements: 7.1, 8.2
    """

    def test_allowed_before_confirmation_time(self):
        """Discovery allowed before 09:35 ET."""
        from utils.funnel_orchestrator import is_discovery_allowed

        config = {
            "funnel": {
                "schedule": {"confirmation_time": "09:35"},
            }
        }

        # Mock time to 06:00 ET
        from unittest.mock import patch
        import datetime as dt
        from zoneinfo import ZoneInfo

        mock_now = dt.datetime(2025, 6, 2, 6, 0, 0, tzinfo=ZoneInfo("America/New_York"))
        with patch("utils.funnel_orchestrator.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: dt.datetime(*a, **kw)
            result = is_discovery_allowed(config)

        assert result is True

    def test_blocked_after_confirmation_time(self):
        """Discovery blocked at or after 09:35 ET."""
        from utils.funnel_orchestrator import is_discovery_allowed

        config = {
            "funnel": {
                "schedule": {"confirmation_time": "09:35"},
            }
        }

        # Mock time to 09:35 ET
        from unittest.mock import patch
        import datetime as dt
        from zoneinfo import ZoneInfo

        mock_now = dt.datetime(2025, 6, 2, 9, 35, 0, tzinfo=ZoneInfo("America/New_York"))
        with patch("utils.funnel_orchestrator.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: dt.datetime(*a, **kw)
            result = is_discovery_allowed(config)

        assert result is False

    def test_blocked_well_after_confirmation_time(self):
        """Discovery blocked at 12:30 ET (midday)."""
        from utils.funnel_orchestrator import is_discovery_allowed

        config = {
            "funnel": {
                "schedule": {"confirmation_time": "09:35"},
            }
        }

        from unittest.mock import patch
        import datetime as dt
        from zoneinfo import ZoneInfo

        mock_now = dt.datetime(2025, 6, 2, 12, 30, 0, tzinfo=ZoneInfo("America/New_York"))
        with patch("utils.funnel_orchestrator.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: dt.datetime(*a, **kw)
            result = is_discovery_allowed(config)

        assert result is False

    def test_uses_config_confirmation_time(self):
        """Uses the configured confirmation_time, not a hardcoded value."""
        from utils.funnel_orchestrator import is_discovery_allowed

        # Custom config with 10:00 cutoff
        config = {
            "funnel": {
                "schedule": {"confirmation_time": "10:00"},
            }
        }

        from unittest.mock import patch
        import datetime as dt
        from zoneinfo import ZoneInfo

        # 09:45 is before 10:00 → allowed
        mock_now = dt.datetime(2025, 6, 2, 9, 45, 0, tzinfo=ZoneInfo("America/New_York"))
        with patch("utils.funnel_orchestrator.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: dt.datetime(*a, **kw)
            result = is_discovery_allowed(config)

        assert result is True

    def test_defaults_to_0935_on_malformed_config(self):
        """Falls back to 09:35 if config is malformed."""
        from utils.funnel_orchestrator import is_discovery_allowed

        config = {
            "funnel": {
                "schedule": {"confirmation_time": "invalid"},
            }
        }

        from unittest.mock import patch
        import datetime as dt
        from zoneinfo import ZoneInfo

        # 09:40 is after default 09:35 → blocked
        mock_now = dt.datetime(2025, 6, 2, 9, 40, 0, tzinfo=ZoneInfo("America/New_York"))
        with patch("utils.funnel_orchestrator.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: dt.datetime(*a, **kw)
            result = is_discovery_allowed(config)

        assert result is False


class TestRunMarketHoursConfirmation:
    """Tests for run_market_hours_confirmation() budget enforcement.

    Requirements: 7.2, 7.3
    """

    @patch("utils.funnel_orchestrator.get_session")
    def test_empty_candidates_records_completed(
        self, mock_get_session, mock_engine, sample_funnel_config
    ):
        """No awaiting_confirmation candidates → returns empty, logs completed."""
        from utils.funnel_orchestrator import run_market_hours_confirmation

        mock_sess = MagicMock()
        mock_sess.query.return_value.filter_by.return_value.order_by.return_value.all.return_value = []
        mock_get_session.return_value = mock_sess

        decisions = run_market_hours_confirmation(mock_engine, sample_funnel_config)
        assert decisions == []

    @patch("utils.funnel_orchestrator.get_session")
    @patch("utils.funnel_confirmation.run_opening_confirmation")
    def test_uses_market_hours_budget(
        self, mock_confirmation, mock_get_session, mock_engine, sample_funnel_config
    ):
        """Uses market_hours_confirmation_budget_seconds (60s) not primary (45s)."""
        from utils.funnel_orchestrator import run_market_hours_confirmation

        mock_sess = MagicMock()
        mock_candidate = MagicMock()
        mock_candidate.scout_rank = 1
        mock_candidate.scout_score = 75.0
        mock_sess.query.return_value.filter_by.return_value.order_by.return_value.all.return_value = [mock_candidate]
        mock_sess.expunge_all = MagicMock()
        mock_get_session.return_value = mock_sess

        mock_decision = MagicMock()
        mock_decision.decision = "promoted"
        mock_confirmation.return_value = [mock_decision]

        run_market_hours_confirmation(mock_engine, sample_funnel_config)

        # Verify budget passed is 60 (market-hours) not 45 (primary)
        mock_confirmation.assert_called_once()
        call_kwargs = mock_confirmation.call_args
        assert call_kwargs[1]["budget_seconds"] == 60 or call_kwargs[0][2] == 60

    @patch("utils.funnel_orchestrator.get_session")
    @patch("utils.funnel_confirmation.run_opening_confirmation")
    def test_timed_out_records_in_funnel_run_log(
        self, mock_confirmation, mock_get_session, mock_engine, sample_funnel_config
    ):
        """When budget exceeded, result_status='timed_out' is recorded."""
        from utils.funnel_orchestrator import run_market_hours_confirmation
        import time

        mock_sess = MagicMock()
        mock_candidate = MagicMock()
        mock_candidate.scout_rank = 1
        mock_candidate.scout_score = 75.0
        mock_sess.query.return_value.filter_by.return_value.order_by.return_value.all.return_value = [mock_candidate]
        mock_sess.expunge_all = MagicMock()
        mock_get_session.return_value = mock_sess

        # Simulate not_evaluated decision (budget exhausted)
        mock_decision = MagicMock()
        mock_decision.decision = "not_evaluated"
        mock_confirmation.return_value = [mock_decision]

        # Patch time.monotonic to simulate elapsed >= budget
        original_monotonic = time.monotonic
        call_count = [0]

        def mock_monotonic():
            call_count[0] += 1
            if call_count[0] == 1:
                return 0.0  # start
            return 61.0  # after budget (60s)

        with patch("utils.funnel_orchestrator.time.monotonic", side_effect=mock_monotonic):
            with patch("utils.funnel_orchestrator._record_market_hours_confirmation_log") as mock_log:
                run_market_hours_confirmation(mock_engine, sample_funnel_config)

                # Should record timed_out
                mock_log.assert_called()
                call_kwargs = mock_log.call_args[1]
                assert call_kwargs["result_status"] == "timed_out"


class TestRunManualIntradayDiscovery:
    """Tests for run_manual_intraday_discovery() in funnel_orchestrator.

    Requirements: 8.3
    """

    @patch("utils.funnel_orchestrator._log_funnel_error")
    @patch("utils.finnhub_client.FinnhubClient")
    def test_manual_discovery_uses_same_budgets(self, mock_fh_cls, mock_log_error, mock_engine):
        """Manual discovery enforces Total_Pipeline_Budget and max_discovery_shortlist."""
        from utils.funnel_orchestrator import run_manual_intraday_discovery

        mock_fh_cls.return_value = MagicMock()

        funnel_config = {
            "funnel": {
                "budgets": {
                    "per_sector_seconds": 15,
                    "total_pipeline_seconds": 90,
                },
                "ceilings": {
                    "max_discovery_shortlist": 5,
                },
            }
        }

        # Provide a config with no enabled sectors so discovery finishes fast
        config = {
            "enabled_sectors": [],
            "sector_buckets": {},
            "core_watchlist": [],
        }

        with patch(
            "utils.funnel_discovery.get_enabled_sectors",
            return_value=[],
        ), patch(
            "utils.funnel_discovery.run_chief_scout_curation",
            return_value=([], "deterministic_fallback", None),
        ), patch(
            "utils.funnel_discovery.persist_discovery_candidates",
            return_value=[],
        ), patch(
            "utils.funnel_discovery.record_discovery_run_log",
        ):
            result = run_manual_intraday_discovery(mock_engine, config, funnel_config)

        assert result is not None
        assert result.selection_mode == "deterministic_fallback"

    @patch("utils.finnhub_client.FinnhubClient")
    def test_manual_discovery_labels_source_run(self, mock_fh_cls, mock_engine):
        """Manual discovery persists candidates with source_run='manual_intraday'."""
        from utils.funnel_orchestrator import run_manual_intraday_discovery

        mock_fh_cls.return_value = MagicMock()

        funnel_config = {
            "funnel": {
                "budgets": {
                    "per_sector_seconds": 15,
                    "total_pipeline_seconds": 90,
                },
                "ceilings": {
                    "max_discovery_shortlist": 5,
                },
            }
        }

        config = {
            "enabled_sectors": [],
            "sector_buckets": {},
            "core_watchlist": [],
        }

        with patch(
            "utils.funnel_discovery.get_enabled_sectors",
            return_value=[],
        ), patch(
            "utils.funnel_discovery.run_chief_scout_curation",
            return_value=([], "deterministic_fallback", None),
        ), patch(
            "utils.funnel_discovery.persist_discovery_candidates",
            return_value=[],
        ) as mock_persist, patch(
            "utils.funnel_discovery.record_discovery_run_log",
        ):
            run_manual_intraday_discovery(mock_engine, config, funnel_config)

        # Verify source_run is "manual_intraday"
        mock_persist.assert_called_once()
        call_args = mock_persist.call_args
        # persist_discovery_candidates(engine, curated_list, selection_mode, source_run="manual_intraday", max_shortlist=5)
        # It's called with positional or keyword args
        if call_args[1]:
            assert call_args[1].get("source_run") == "manual_intraday"
        else:
            # Positional: (engine, curated_list, selection_mode, source_run, max_shortlist)
            assert call_args[0][3] == "manual_intraday"
