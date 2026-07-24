"""End-to-end test: backward compatibility with coordinator flag disabled.

Verifies that when PM_CYCLE_COORDINATOR_MODE="disabled", the legacy path is
completely unchanged:
1. run_coordinated_market_cycle() delegates to run_analyst_refresh() + run_intraday()
2. No coordinator code executes (CycleCoordinator is never instantiated)
3. No freshness gate is invoked
4. analyst.run() accepts cycle_id=None without error (backward-compatible signature)
5. pm.run_profile() accepts cycle_id=None without error (backward-compatible signature)

This validates that the entire feature can be disabled with zero impact on production.

Requirements: 1.5, 9.1, 9.2, 9.5
"""

from __future__ import annotations

import inspect
from unittest.mock import patch, MagicMock

import pytest


class TestBackwardCompatibilityFlagDisabled:
    """When PM_CYCLE_COORDINATOR_MODE=disabled, legacy behavior is preserved."""

    @patch("orchestrator.run_intraday")
    @patch("orchestrator.run_analyst_refresh")
    @patch("utils.gate_config.PM_CYCLE_COORDINATOR_MODE", "disabled")
    def test_coordinated_cycle_delegates_to_legacy_when_disabled(
        self, mock_analyst_refresh, mock_intraday
    ):
        """When flag disabled, run_coordinated_market_cycle() just calls legacy jobs.

        Validates: Requirements 1.5, 9.2
        """
        from orchestrator import run_coordinated_market_cycle

        run_coordinated_market_cycle()

        mock_analyst_refresh.assert_called_once()
        mock_intraday.assert_called_once()

    @patch("orchestrator.run_intraday")
    @patch("orchestrator.run_analyst_refresh")
    @patch("utils.gate_config.PM_CYCLE_COORDINATOR_MODE", "disabled")
    def test_no_coordinator_instantiated_when_disabled(
        self, mock_analyst_refresh, mock_intraday
    ):
        """CycleCoordinator is never instantiated when flag is disabled.

        Validates: Requirements 1.5, 9.5
        """
        from orchestrator import run_coordinated_market_cycle

        with patch("utils.cycle_coordinator.CycleCoordinator") as mock_coord:
            run_coordinated_market_cycle()
            mock_coord.assert_not_called()

    @patch("orchestrator.run_intraday")
    @patch("orchestrator.run_analyst_refresh")
    @patch("utils.gate_config.PM_CYCLE_COORDINATOR_MODE", "disabled")
    def test_no_freshness_gate_in_legacy_path(
        self, mock_analyst_refresh, mock_intraday
    ):
        """Freshness gate is never called in the legacy path.

        Validates: Requirements 1.5, 9.5
        """
        from orchestrator import run_coordinated_market_cycle

        with patch("utils.signal_freshness.check_signal_freshness") as mock_freshness:
            run_coordinated_market_cycle()
            mock_freshness.assert_not_called()

    def test_analyst_run_accepts_cycle_id_none(self):
        """analyst.run() accepts cycle_id=None without TypeError (backward-compatible).

        Validates: Requirements 9.1, 9.5
        """
        from agents.analyst import run as analyst_run

        sig = inspect.signature(analyst_run)
        assert "cycle_id" in sig.parameters
        assert sig.parameters["cycle_id"].default is None

    def test_pm_run_profile_accepts_cycle_id_none(self):
        """pm.run_profile() accepts cycle_id=None without TypeError (backward-compatible).

        Validates: Requirements 9.2, 9.5
        """
        from agents.portfolio_manager import run_profile

        sig = inspect.signature(run_profile)
        assert "cycle_id" in sig.parameters
        assert sig.parameters["cycle_id"].default is None

    @patch("orchestrator.run_intraday")
    @patch("orchestrator.run_analyst_refresh")
    @patch("utils.gate_config.PM_CYCLE_COORDINATOR_MODE", "disabled")
    def test_legacy_path_calls_in_correct_order(
        self, mock_analyst_refresh, mock_intraday
    ):
        """Legacy path calls analyst refresh before intraday (sequential delegation).

        Validates: Requirements 1.5, 9.5
        """
        from orchestrator import run_coordinated_market_cycle

        call_order = []
        mock_analyst_refresh.side_effect = lambda: call_order.append("analyst_refresh")
        mock_intraday.side_effect = lambda: call_order.append("intraday")

        run_coordinated_market_cycle()

        assert call_order == ["analyst_refresh", "intraday"]
