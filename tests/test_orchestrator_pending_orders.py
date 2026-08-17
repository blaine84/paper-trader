"""Orchestrator wiring for pending limit orders.

Validates by actually importing and running orchestrator.main(), which became
possible once utils/resource_telemetry.py stopped hard-failing its `import
resource` on non-POSIX platforms. Previously this could only be checked by
inspecting the source.

Covers:
- the monitor job is registered when PENDING_ORDER_MODE != "disabled"
- it is NOT registered when the mode is "disabled"
- it is configured with max_instances=1 and coalesce=True
- the startup orphan sweep runs, and a failure in it does not block startup
- check_schema() creates the pending-order tables

Requirements: 2.10, 3.9, 4.1, 9.12, 14.8, 0.4
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, inspect


@pytest.fixture
def _mock_main_infrastructure():
    """Patch heavyweight orchestrator infrastructure so main() runs cheaply.

    Mirrors the fixture in test_orchestrator_plan_monitor.py.
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


def _run_main(mode: str, interval: int = 60):
    """Run main() with the pending-order mode set, returning the mock registry."""
    with patch("utils.gate_config.PENDING_ORDER_MODE", mode), \
         patch("utils.gate_config.PENDING_ORDER_MONITOR_INTERVAL_SECONDS", interval), \
         patch("utils.pending_order_registry.PendingOrderRegistry") as registry_class:

        registry = MagicMock()
        registry.finalize_orphaned_orders.return_value = {}
        registry_class.return_value = registry

        from orchestrator import main

        main()
        return registry


def _jobs(mock_scheduler, job_id: str):
    return [
        c for c in mock_scheduler.add_job.call_args_list
        if c.kwargs.get("id") == job_id
    ]


# ---------------------------------------------------------------------------
# Job registration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["observe", "enabled"])
def test_monitor_job_is_registered_when_active(_mock_main_infrastructure, mode):
    mock_scheduler = _mock_main_infrastructure
    _run_main(mode)

    jobs = _jobs(mock_scheduler, "pending_order_monitor")
    assert len(jobs) == 1, f"expected exactly one job in {mode} mode, got {len(jobs)}"


def test_monitor_job_is_not_registered_when_disabled(_mock_main_infrastructure):
    """Requirement 0.4 — the disabled path registers nothing."""
    mock_scheduler = _mock_main_infrastructure
    _run_main("disabled")

    assert _jobs(mock_scheduler, "pending_order_monitor") == []


def test_monitor_job_uses_max_instances_one_and_coalesce(_mock_main_infrastructure):
    """Requirement 14.8 — ticks must not overlap or pile up."""
    mock_scheduler = _mock_main_infrastructure
    _run_main("enabled")

    job = _jobs(mock_scheduler, "pending_order_monitor")[0]
    assert job.kwargs.get("max_instances") == 1
    assert job.kwargs.get("coalesce") is True
    assert job.kwargs.get("replace_existing") is True


def test_monitor_job_honors_the_configured_interval(_mock_main_infrastructure):
    mock_scheduler = _mock_main_infrastructure
    _run_main("enabled", interval=45)

    job = _jobs(mock_scheduler, "pending_order_monitor")[0]
    trigger = job.args[1] if len(job.args) > 1 else job.kwargs.get("trigger")
    assert "45" in str(trigger), f"interval not reflected in trigger: {trigger}"


def test_no_separate_orphan_sweep_job_is_registered(_mock_main_infrastructure):
    """The tick runs the sweep as its own final step, so a second job would be
    redundant work against the same rows."""
    mock_scheduler = _mock_main_infrastructure
    _run_main("enabled")

    assert _jobs(mock_scheduler, "pending_order_orphan_sweep") == []


# ---------------------------------------------------------------------------
# Startup orphan sweep
# ---------------------------------------------------------------------------


def test_startup_sweep_runs_when_active(_mock_main_infrastructure):
    """Requirement 9.12 — no order survives a restart in a transient state."""
    registry = _run_main("enabled")
    registry.finalize_orphaned_orders.assert_called_once()


def test_startup_sweep_is_skipped_when_disabled(_mock_main_infrastructure):
    registry = _run_main("disabled")
    registry.finalize_orphaned_orders.assert_not_called()


def test_startup_sweep_failure_does_not_block_startup(_mock_main_infrastructure):
    """Fail-open: observability must never prevent the orchestrator booting."""
    mock_scheduler = _mock_main_infrastructure

    with patch("utils.gate_config.PENDING_ORDER_MODE", "enabled"), \
         patch("utils.gate_config.PENDING_ORDER_MONITOR_INTERVAL_SECONDS", 60), \
         patch("utils.pending_order_registry.PendingOrderRegistry") as registry_class:

        registry = MagicMock()
        registry.finalize_orphaned_orders.side_effect = RuntimeError("db locked")
        registry_class.return_value = registry

        from orchestrator import main

        main()  # must not raise

    # Startup continued far enough to register the monitor.
    assert len(_jobs(mock_scheduler, "pending_order_monitor")) == 1


# ---------------------------------------------------------------------------
# Market-hours guard
# ---------------------------------------------------------------------------


def test_monitor_tick_is_skipped_outside_regular_hours(_mock_main_infrastructure):
    """The registered callable must consult the market-hours guard."""
    mock_scheduler = _mock_main_infrastructure
    _run_main("enabled")

    job_fn = _jobs(mock_scheduler, "pending_order_monitor")[0].args[0]

    with patch("orchestrator._skip_outside_regular_market_job", return_value=True), \
         patch("utils.pending_order_monitor.run") as run_monitor:
        job_fn()

    run_monitor.assert_not_called()


def test_monitor_tick_runs_during_regular_hours(_mock_main_infrastructure):
    mock_scheduler = _mock_main_infrastructure
    _run_main("enabled")

    job_fn = _jobs(mock_scheduler, "pending_order_monitor")[0].args[0]

    with patch("orchestrator._skip_outside_regular_market_job", return_value=False), \
         patch("orchestrator.get_engine", return_value=MagicMock()), \
         patch("utils.pending_order_monitor.run") as run_monitor:
        run_monitor.return_value = MagicMock(had_activity=False)
        job_fn()

    run_monitor.assert_called_once()


def test_monitor_tick_swallows_errors(_mock_main_infrastructure):
    """A failing tick must not propagate into the scheduler."""
    mock_scheduler = _mock_main_infrastructure
    _run_main("enabled")

    job_fn = _jobs(mock_scheduler, "pending_order_monitor")[0].args[0]

    with patch("orchestrator._skip_outside_regular_market_job", return_value=False), \
         patch("orchestrator.get_engine", return_value=MagicMock()), \
         patch("utils.pending_order_monitor.run", side_effect=RuntimeError("boom")):
        job_fn()  # must not raise


# ---------------------------------------------------------------------------
# check_schema wiring (Requirement 2.10)
# ---------------------------------------------------------------------------


def test_check_schema_creates_the_pending_order_tables():
    """Verified by execution now, not source inspection.

    Runs unconditionally rather than behind the feature flag, so flipping the
    flag on a live system needs no schema step.
    """
    from db.schema import Base
    from orchestrator import check_schema

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    check_schema(engine)

    inspector = inspect(engine)
    assert inspector.has_table("pending_orders")
    assert inspector.has_table("pending_order_events")

    indexes = {ix["name"] for ix in inspector.get_indexes("pending_orders")}
    assert "idx_pending_orders_active_key" in indexes


def test_check_schema_is_idempotent_for_pending_orders():
    from db.schema import Base
    from orchestrator import check_schema

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    check_schema(engine)
    check_schema(engine)  # must not raise

    assert inspect(engine).has_table("pending_orders")


def test_check_schema_creates_tables_even_when_feature_disabled():
    """So the dashboard can query them while the feature is off."""
    from db.schema import Base
    from orchestrator import check_schema

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with patch("utils.gate_config.PENDING_ORDER_MODE", "disabled"):
        check_schema(engine)

    assert inspect(engine).has_table("pending_orders")
