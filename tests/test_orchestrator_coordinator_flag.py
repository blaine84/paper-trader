"""Integration tests for orchestrator coordinator flag behavior.

Validates that PM_CYCLE_COORDINATOR_MODE controls routing in:
- run_coordinated_market_cycle()
- run_once()

Requirements: 1.4, 1.5, 6.1, 6.2, 9.1, 9.2
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest


class TestRunCoordinatedMarketCycleEnabled:
    """When PM_CYCLE_COORDINATOR_MODE=enabled, coordinator path is used."""

    @patch("orchestrator.get_engine", return_value=MagicMock())
    @patch("orchestrator._skip_outside_regular_market_job", return_value=False)
    @patch("utils.cycle_coordinator.CycleCoordinator")
    @patch("utils.gate_config.PM_CYCLE_COORDINATOR_MODE", "enabled")
    def test_enabled_calls_coordinator(
        self, mock_coordinator_class, mock_skip, mock_engine
    ):
        """Validates: Requirements 1.4, 6.1"""
        from orchestrator import run_coordinated_market_cycle

        mock_coordinator = MagicMock()
        mock_coordinator.run_market_cycle.return_value = MagicMock(
            cycle_id="20260723_1030_scheduled_a3f2",
            total_duration_seconds=5.0,
            symbols_fresh=3,
            symbols_stale_skipped=1,
        )
        mock_coordinator_class.return_value = mock_coordinator

        run_coordinated_market_cycle()

        mock_coordinator.run_market_cycle.assert_called_once_with(
            trigger_source="scheduled"
        )

    @patch("orchestrator.get_engine", return_value=MagicMock())
    @patch("orchestrator._skip_outside_regular_market_job", return_value=False)
    @patch("utils.cycle_coordinator.CycleCoordinator")
    @patch("utils.gate_config.PM_CYCLE_COORDINATOR_MODE", "enabled")
    def test_enabled_does_not_call_legacy_jobs(
        self, mock_coordinator_class, mock_skip, mock_engine
    ):
        """Validates: Requirements 1.4, 6.2"""
        from orchestrator import run_coordinated_market_cycle

        mock_coordinator = MagicMock()
        mock_coordinator.run_market_cycle.return_value = MagicMock(
            cycle_id="test_123",
            total_duration_seconds=2.0,
            symbols_fresh=1,
            symbols_stale_skipped=0,
        )
        mock_coordinator_class.return_value = mock_coordinator

        with patch("orchestrator.run_analyst_refresh") as mock_analyst, \
             patch("orchestrator.run_intraday") as mock_intraday:
            run_coordinated_market_cycle()

            mock_analyst.assert_not_called()
            mock_intraday.assert_not_called()


class TestRunCoordinatedMarketCycleDisabled:
    """When PM_CYCLE_COORDINATOR_MODE=disabled, legacy path is used."""

    @patch("orchestrator.run_intraday")
    @patch("orchestrator.run_analyst_refresh")
    @patch("utils.gate_config.PM_CYCLE_COORDINATOR_MODE", "disabled")
    def test_disabled_calls_legacy_jobs(self, mock_analyst, mock_intraday):
        """Validates: Requirements 1.5, 9.2"""
        from orchestrator import run_coordinated_market_cycle

        run_coordinated_market_cycle()

        mock_analyst.assert_called_once()
        mock_intraday.assert_called_once()

    @patch("orchestrator.run_intraday")
    @patch("orchestrator.run_analyst_refresh")
    @patch("utils.gate_config.PM_CYCLE_COORDINATOR_MODE", "disabled")
    def test_disabled_does_not_instantiate_coordinator(
        self, mock_analyst, mock_intraday
    ):
        """Validates: Requirements 1.5"""
        from orchestrator import run_coordinated_market_cycle

        with patch("utils.cycle_coordinator.CycleCoordinator") as mock_coord_class:
            run_coordinated_market_cycle()

            mock_coord_class.assert_not_called()


class TestRunOnceEnabled:
    """run_once() routes to coordinator when PM_CYCLE_COORDINATOR_MODE=enabled."""

    @patch("orchestrator.ensure_shadow_ledger_schema")
    @patch("orchestrator.check_schema")
    @patch("orchestrator.ensure_initial_balance")
    @patch("orchestrator.get_engine", return_value=MagicMock())
    @patch("utils.cycle_coordinator.CycleCoordinator")
    @patch("utils.gate_config.PM_CYCLE_COORDINATOR_MODE", "enabled")
    def test_run_once_enabled_uses_coordinator(
        self,
        mock_coord_class,
        mock_engine,
        mock_balance,
        mock_schema,
        mock_shadow,
    ):
        """Validates: Requirements 1.4, 9.1"""
        from orchestrator import run_once

        mock_coordinator = MagicMock()
        mock_coordinator.run_market_cycle.return_value = MagicMock(
            cycle_id="20260723_1030_manual_b1c2",
            total_duration_seconds=3.0,
            symbols_fresh=2,
            symbols_stale_skipped=0,
        )
        mock_coord_class.return_value = mock_coordinator

        run_once()

        mock_coord_class.assert_called_once()
        mock_coordinator.run_market_cycle.assert_called_once_with(
            trigger_source="manual"
        )

    @patch("orchestrator.ensure_shadow_ledger_schema")
    @patch("orchestrator.check_schema")
    @patch("orchestrator.ensure_initial_balance")
    @patch("orchestrator.get_engine", return_value=MagicMock())
    @patch("utils.cycle_coordinator.CycleCoordinator")
    @patch("utils.gate_config.PM_CYCLE_COORDINATOR_MODE", "enabled")
    def test_run_once_enabled_does_not_call_legacy(
        self,
        mock_coord_class,
        mock_engine,
        mock_balance,
        mock_schema,
        mock_shadow,
    ):
        """Validates: Requirements 6.1, 6.2"""
        from orchestrator import run_once

        mock_coordinator = MagicMock()
        mock_coordinator.run_market_cycle.return_value = MagicMock(
            cycle_id="test_456",
            total_duration_seconds=1.0,
            symbols_fresh=1,
            symbols_stale_skipped=0,
        )
        mock_coord_class.return_value = mock_coordinator

        with patch("orchestrator.run_pre_market") as mock_pre, \
             patch("orchestrator.run_intraday") as mock_intraday:
            run_once()

            mock_pre.assert_not_called()
            mock_intraday.assert_not_called()


class TestRunOnceDisabled:
    """run_once() routes to legacy path when PM_CYCLE_COORDINATOR_MODE=disabled."""

    @patch("orchestrator.run_intraday")
    @patch("orchestrator.run_pre_market")
    @patch("orchestrator.ensure_shadow_ledger_schema")
    @patch("orchestrator.check_schema")
    @patch("orchestrator.ensure_initial_balance")
    @patch("orchestrator.get_engine", return_value=MagicMock())
    @patch("utils.gate_config.PM_CYCLE_COORDINATOR_MODE", "disabled")
    def test_run_once_disabled_uses_legacy(
        self,
        mock_engine,
        mock_balance,
        mock_schema,
        mock_shadow,
        mock_pre,
        mock_intraday,
    ):
        """Validates: Requirements 1.5, 9.2"""
        from orchestrator import run_once

        run_once()

        mock_pre.assert_called_once()
        mock_intraday.assert_called_once()

    @patch("orchestrator.run_intraday")
    @patch("orchestrator.run_pre_market")
    @patch("orchestrator.ensure_shadow_ledger_schema")
    @patch("orchestrator.check_schema")
    @patch("orchestrator.ensure_initial_balance")
    @patch("orchestrator.get_engine", return_value=MagicMock())
    @patch("utils.gate_config.PM_CYCLE_COORDINATOR_MODE", "disabled")
    def test_run_once_disabled_does_not_instantiate_coordinator(
        self,
        mock_engine,
        mock_balance,
        mock_schema,
        mock_shadow,
        mock_pre,
        mock_intraday,
    ):
        """Validates: Requirements 1.5"""
        from orchestrator import run_once

        with patch("utils.cycle_coordinator.CycleCoordinator") as mock_coord_class:
            run_once()

            mock_coord_class.assert_not_called()
