"""End-to-end test: manual market-cycle command.

Verifies that _run_manual_coordinated_cycle():
1. Calls CycleCoordinator with trigger_source="manual"
2. Works regardless of PM_CYCLE_COORDINATOR_MODE flag value (always uses coordinator)
3. Bypasses market-hours guard (can be run anytime)

Requirements: 1.5, 6.2, 9.3
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest


class TestManualMarketCycleCommand:
    """Tests for _run_manual_coordinated_cycle() — the manual market-cycle CLI command."""

    @patch("orchestrator.get_engine", return_value=MagicMock())
    @patch("utils.cycle_coordinator.CycleCoordinator")
    def test_manual_cycle_uses_coordinator_with_manual_trigger(
        self, mock_coord_class, mock_engine
    ):
        """_run_manual_coordinated_cycle() uses coordinator with trigger_source='manual'.

        Validates: Requirements 9.3
        """
        mock_coordinator = MagicMock()
        mock_coordinator.run_market_cycle.return_value = MagicMock(
            cycle_id="20260723_1030_manual_a3f2",
            total_duration_seconds=3.0,
            symbols_fresh=2,
            symbols_stale_skipped=0,
        )
        mock_coord_class.return_value = mock_coordinator

        from orchestrator import _run_manual_coordinated_cycle

        _run_manual_coordinated_cycle()

        mock_coordinator.run_market_cycle.assert_called_once_with(trigger_source="manual")

    @patch("orchestrator.get_engine", return_value=MagicMock())
    @patch("utils.cycle_coordinator.CycleCoordinator")
    @patch("utils.gate_config.PM_CYCLE_COORDINATOR_MODE", "disabled")
    def test_manual_cycle_works_when_flag_disabled(
        self, mock_coord_class, mock_engine
    ):
        """Manual cycle command always uses coordinator even when flag is disabled.

        Validates: Requirements 1.5, 9.3
        """
        mock_coordinator = MagicMock()
        mock_coordinator.run_market_cycle.return_value = MagicMock(
            cycle_id="test_123",
            total_duration_seconds=1.0,
            symbols_fresh=0,
            symbols_stale_skipped=0,
        )
        mock_coord_class.return_value = mock_coordinator

        from orchestrator import _run_manual_coordinated_cycle

        _run_manual_coordinated_cycle()

        mock_coord_class.assert_called_once()
        mock_coordinator.run_market_cycle.assert_called_once_with(trigger_source="manual")

    @patch("orchestrator.get_engine", return_value=MagicMock())
    @patch("utils.cycle_coordinator.CycleCoordinator")
    @patch("utils.gate_config.PM_CYCLE_COORDINATOR_MODE", "enabled")
    def test_manual_cycle_works_when_flag_enabled(
        self, mock_coord_class, mock_engine
    ):
        """Manual cycle command works when flag is enabled too.

        Validates: Requirements 1.5, 9.3
        """
        mock_coordinator = MagicMock()
        mock_coordinator.run_market_cycle.return_value = MagicMock(
            cycle_id="test_456",
            total_duration_seconds=2.0,
            symbols_fresh=1,
            symbols_stale_skipped=1,
        )
        mock_coord_class.return_value = mock_coordinator

        from orchestrator import _run_manual_coordinated_cycle

        _run_manual_coordinated_cycle()

        mock_coord_class.assert_called_once()
        mock_coordinator.run_market_cycle.assert_called_once_with(trigger_source="manual")

    @patch("orchestrator.get_engine", return_value=MagicMock())
    @patch("utils.cycle_coordinator.CycleCoordinator")
    def test_manual_cycle_does_not_check_market_hours(
        self, mock_coord_class, mock_engine
    ):
        """Manual cycle bypasses market-hours guard (can be run anytime).

        Validates: Requirements 6.2, 9.3
        """
        mock_coordinator = MagicMock()
        mock_coordinator.run_market_cycle.return_value = MagicMock(
            cycle_id="test_789",
            total_duration_seconds=1.0,
            symbols_fresh=0,
            symbols_stale_skipped=0,
        )
        mock_coord_class.return_value = mock_coordinator

        with patch("orchestrator._skip_outside_regular_market_job") as mock_skip:
            from orchestrator import _run_manual_coordinated_cycle

            _run_manual_coordinated_cycle()

            mock_skip.assert_not_called()
