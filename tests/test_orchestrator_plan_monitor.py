"""Integration tests for orchestrator plan monitor registration and startup behavior.

Validates that:
- Plan monitor job is registered when TRIGGERED_PLAN_MODE != "disabled"
- Plan monitor job is NOT registered when TRIGGERED_PLAN_MODE == "disabled"
- finalize_orphaned_plans() is called at startup
- Plan monitor respects the market hours guard

Requirements: 4.1, 8.6, 11.6
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock, call
from datetime import datetime

import pytest


@pytest.fixture
def _mock_main_infrastructure():
    """Patch all heavyweight orchestrator infrastructure so main() can run cheaply.

    Yields the mock scheduler instance for job registration assertions.
    """
    mock_scheduler = MagicMock()
    mock_scheduler.start.side_effect = SystemExit(0)

    with patch("orchestrator.BlockingScheduler", return_value=mock_scheduler), \
         patch("orchestrator.check_llm_connectivity"), \
         patch("orchestrator.ensure_shadow_ledger_schema"), \
         patch("orchestrator.check_schema"), \
         patch("orchestrator.ensure_initial_balance"), \
         patch("orchestrator.get_engine", return_value=MagicMock()), \
         patch("orchestrator.load_funnel_config", return_value={}), \
         patch("orchestrator._check_missed_funnel_jobs"), \
         patch("utils.alert_dispatch_schema.init_alert_dispatch_schema"), \
         patch("utils.gate_config.MARKET_STATE_MODE", "disabled"), \
         patch("utils.gate_config.PM_ALERT_DISPATCH_MODE", "disabled"), \
         patch("utils.gate_config.PM_ALERT_DISPATCHER_INTERVAL_SECONDS", 60):
        yield mock_scheduler


class TestPlanMonitorJobRegistration:
    """Plan monitor APScheduler job is conditionally registered based on flag."""

    def test_plan_monitor_registered_when_flag_enabled(self, _mock_main_infrastructure):
        """Validates: Requirements 4.1, 11.6 — job registered when mode != disabled."""
        mock_scheduler = _mock_main_infrastructure

        with patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled"), \
             patch("utils.gate_config.PLAN_MONITOR_INTERVAL_SECONDS", 30), \
             patch("utils.trade_plan_registry.TradePlanRegistry") as mock_registry_class:

            mock_registry = MagicMock()
            mock_registry.finalize_orphaned_plans.return_value = {}
            mock_registry_class.return_value = mock_registry

            from orchestrator import main
            main()

            # Verify plan_monitor job was added to the scheduler
            add_job_calls = mock_scheduler.add_job.call_args_list
            plan_monitor_jobs = [
                c for c in add_job_calls
                if c.kwargs.get("id") == "plan_monitor"
            ]
            assert len(plan_monitor_jobs) == 1, (
                f"Expected plan_monitor job registered once, got {len(plan_monitor_jobs)}"
            )

    def test_plan_monitor_registered_when_flag_observe(self, _mock_main_infrastructure):
        """Validates: Requirements 4.1, 11.6 — job registered in observe mode too."""
        mock_scheduler = _mock_main_infrastructure

        with patch("utils.gate_config.TRIGGERED_PLAN_MODE", "observe"), \
             patch("utils.gate_config.PLAN_MONITOR_INTERVAL_SECONDS", 30), \
             patch("utils.trade_plan_registry.TradePlanRegistry") as mock_registry_class:

            mock_registry = MagicMock()
            mock_registry.finalize_orphaned_plans.return_value = {}
            mock_registry_class.return_value = mock_registry

            from orchestrator import main
            main()

            add_job_calls = mock_scheduler.add_job.call_args_list
            plan_monitor_jobs = [
                c for c in add_job_calls
                if c.kwargs.get("id") == "plan_monitor"
            ]
            assert len(plan_monitor_jobs) == 1

    def test_plan_monitor_not_registered_when_flag_disabled(self, _mock_main_infrastructure):
        """Validates: Requirements 4.1, 11.6 — no job when mode == disabled."""
        mock_scheduler = _mock_main_infrastructure

        with patch("utils.gate_config.TRIGGERED_PLAN_MODE", "disabled"), \
             patch("utils.gate_config.PLAN_MONITOR_INTERVAL_SECONDS", 30):

            from orchestrator import main
            main()

            add_job_calls = mock_scheduler.add_job.call_args_list
            plan_monitor_jobs = [
                c for c in add_job_calls
                if c.kwargs.get("id") == "plan_monitor"
            ]
            assert len(plan_monitor_jobs) == 0, (
                "plan_monitor job should NOT be registered when TRIGGERED_PLAN_MODE=disabled"
            )

    def test_plan_monitor_job_has_max_instances_1(self, _mock_main_infrastructure):
        """Validates: Requirement 4.1 — only one instance at a time."""
        mock_scheduler = _mock_main_infrastructure

        with patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled"), \
             patch("utils.gate_config.PLAN_MONITOR_INTERVAL_SECONDS", 30), \
             patch("utils.trade_plan_registry.TradePlanRegistry") as mock_registry_class:

            mock_registry = MagicMock()
            mock_registry.finalize_orphaned_plans.return_value = {}
            mock_registry_class.return_value = mock_registry

            from orchestrator import main
            main()

            add_job_calls = mock_scheduler.add_job.call_args_list
            plan_monitor_jobs = [
                c for c in add_job_calls
                if c.kwargs.get("id") == "plan_monitor"
            ]
            assert plan_monitor_jobs[0].kwargs.get("max_instances") == 1


class TestFinalizeOrphanedPlansAtStartup:
    """finalize_orphaned_plans() is called during orchestrator startup."""

    def test_finalize_orphaned_plans_called_at_startup(self, _mock_main_infrastructure):
        """Validates: Requirement 8.6 — orphan sweep runs on startup."""
        with patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled"), \
             patch("utils.gate_config.PLAN_MONITOR_INTERVAL_SECONDS", 30), \
             patch("utils.trade_plan_registry.TradePlanRegistry") as mock_registry_class:

            mock_registry = MagicMock()
            mock_registry.finalize_orphaned_plans.return_value = {"plan_123": MagicMock(value="expired")}
            mock_registry_class.return_value = mock_registry

            from orchestrator import main
            main()

            mock_registry.finalize_orphaned_plans.assert_called_once()

    def test_finalize_orphaned_plans_not_called_when_disabled(self, _mock_main_infrastructure):
        """Validates: Requirement 8.6 — no orphan sweep when flag disabled."""
        with patch("utils.gate_config.TRIGGERED_PLAN_MODE", "disabled"), \
             patch("utils.gate_config.PLAN_MONITOR_INTERVAL_SECONDS", 30), \
             patch("utils.trade_plan_registry.TradePlanRegistry") as mock_registry_class:

            mock_registry = MagicMock()
            mock_registry_class.return_value = mock_registry

            from orchestrator import main
            main()

            mock_registry.finalize_orphaned_plans.assert_not_called()

    def test_finalize_orphaned_plans_failure_does_not_block_startup(self, _mock_main_infrastructure):
        """Validates: Requirement 8.6 — orphan sweep is fail-open."""
        mock_scheduler = _mock_main_infrastructure

        with patch("utils.gate_config.TRIGGERED_PLAN_MODE", "enabled"), \
             patch("utils.gate_config.PLAN_MONITOR_INTERVAL_SECONDS", 30), \
             patch("utils.trade_plan_registry.TradePlanRegistry") as mock_registry_class:

            mock_registry = MagicMock()
            mock_registry.finalize_orphaned_plans.side_effect = RuntimeError("DB locked")
            mock_registry_class.return_value = mock_registry

            from orchestrator import main
            # Should NOT raise — startup continues despite sweep failure
            main()

            # The sweep was attempted
            mock_registry.finalize_orphaned_plans.assert_called_once()
            # Scheduler still started (scheduler.start() was called)
            mock_scheduler.start.assert_called_once()


class TestPlanMonitorMarketHoursGuard:
    """Plan monitor respects the market hours guard (_skip_outside_regular_market_job)."""

    def test_plan_monitor_skipped_outside_market_hours(self):
        """Validates: Requirement 4.1 — monitor does not run outside 9:30-16:00 ET."""
        from pytz import timezone as pytz_timezone
        from orchestrator import _skip_outside_regular_market_job

        # Simulate 8:00 AM ET (before market open)
        before_open = datetime(2026, 7, 30, 8, 0, 0, tzinfo=pytz_timezone("America/New_York"))

        with patch("orchestrator.is_trading_day", return_value=True):
            result = _skip_outside_regular_market_job("plan_monitor", now_et=before_open)
            assert result is True, "Plan monitor should skip before market open"

    def test_plan_monitor_runs_during_market_hours(self):
        """Validates: Requirement 4.1 — monitor runs during 9:30-16:00 ET."""
        from pytz import timezone as pytz_timezone
        from orchestrator import _skip_outside_regular_market_job

        # Simulate 10:30 AM ET (during market hours)
        during_market = datetime(2026, 7, 30, 10, 30, 0, tzinfo=pytz_timezone("America/New_York"))

        with patch("orchestrator.is_trading_day", return_value=True):
            result = _skip_outside_regular_market_job("plan_monitor", now_et=during_market)
            assert result is False, "Plan monitor should run during market hours"

    def test_plan_monitor_skipped_after_market_close(self):
        """Validates: Requirement 4.1 — monitor does not run after 16:00 ET."""
        from pytz import timezone as pytz_timezone
        from orchestrator import _skip_outside_regular_market_job

        # Simulate 4:30 PM ET (after market close)
        after_close = datetime(2026, 7, 30, 16, 30, 0, tzinfo=pytz_timezone("America/New_York"))

        with patch("orchestrator.is_trading_day", return_value=True):
            result = _skip_outside_regular_market_job("plan_monitor", now_et=after_close)
            assert result is True, "Plan monitor should skip after market close"

    def test_plan_monitor_skipped_on_non_trading_day(self):
        """Validates: Requirement 4.1 — monitor does not run on weekends/holidays."""
        from pytz import timezone as pytz_timezone
        from orchestrator import _skip_outside_regular_market_job

        # Simulate Saturday 10:30 AM ET
        saturday = datetime(2026, 7, 25, 10, 30, 0, tzinfo=pytz_timezone("America/New_York"))

        with patch("orchestrator.is_trading_day", return_value=False):
            result = _skip_outside_regular_market_job("plan_monitor", now_et=saturday)
            assert result is True, "Plan monitor should skip on non-trading days"

    def test_plan_monitor_module_noop_when_disabled(self):
        """Validates: Requirement 4.1 — plan_monitor.run() returns immediately when disabled."""
        with patch("utils.gate_config.TRIGGERED_PLAN_MODE", "disabled"):
            import utils.plan_monitor as plan_monitor_mod

            result = plan_monitor_mod.run(MagicMock())

            assert result.plans_checked == 0
            assert result.plans_triggered == 0
            assert result.tick_duration_ms == 0.0
