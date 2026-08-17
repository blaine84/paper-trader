"""Verify the triggered-plan scheduler jobs are no longer registered.

After the limit-order-mode-cleanup, the orchestrator must not register
``plan_monitor`` or ``plan_orphan_sweep`` APScheduler jobs under any
PENDING_ORDER_MODE setting.

Requirements: 9.5 (spec: limit-order-mode-cleanup)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def _mock_main_infrastructure():
    """Patch heavyweight orchestrator infrastructure so main() runs cheaply.

    Mirrors the fixture in test_orchestrator_pending_orders.py.
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


def _run_main(mode: str):
    """Run main() with the given PENDING_ORDER_MODE, returning the mock scheduler."""
    with patch("utils.gate_config.PENDING_ORDER_MODE", mode), \
         patch("utils.gate_config.PENDING_ORDER_MONITOR_INTERVAL_SECONDS", 60), \
         patch("utils.pending_order_registry.PendingOrderRegistry") as registry_class:

        registry = MagicMock()
        registry.finalize_orphaned_orders.return_value = {}
        registry_class.return_value = registry

        from orchestrator import main

        main()


def _jobs_with_id(mock_scheduler, job_id: str):
    return [
        c for c in mock_scheduler.add_job.call_args_list
        if c.kwargs.get("id") == job_id
    ]


# ---------------------------------------------------------------------------
# plan_monitor and plan_orphan_sweep must not be registered
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["disabled", "observe", "enabled"])
def test_plan_monitor_job_is_not_registered(_mock_main_infrastructure, mode):
    """No PENDING_ORDER_MODE value should cause plan_monitor registration."""
    mock_scheduler = _mock_main_infrastructure
    _run_main(mode)

    assert _jobs_with_id(mock_scheduler, "plan_monitor") == [], (
        f"plan_monitor job was unexpectedly registered in mode={mode!r}"
    )


@pytest.mark.parametrize("mode", ["disabled", "observe", "enabled"])
def test_plan_orphan_sweep_job_is_not_registered(_mock_main_infrastructure, mode):
    """No PENDING_ORDER_MODE value should cause plan_orphan_sweep registration."""
    mock_scheduler = _mock_main_infrastructure
    _run_main(mode)

    assert _jobs_with_id(mock_scheduler, "plan_orphan_sweep") == [], (
        f"plan_orphan_sweep job was unexpectedly registered in mode={mode!r}"
    )
