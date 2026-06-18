import logging
from unittest.mock import MagicMock, patch

import orchestrator


def _reset_pm_cycle_state():
    if orchestrator._pm_cycle_lock.locked():
        orchestrator._pm_cycle_lock.release()
    orchestrator._set_pm_cycle_active(False)
    orchestrator._pm_cycle_owner = None


def setup_function():
    _reset_pm_cycle_state()


def teardown_function():
    _reset_pm_cycle_state()


def test_run_intraday_skips_when_pm_cycle_is_already_active(caplog):
    assert orchestrator._try_begin_pm_cycle("price_monitor") is True

    with (
        patch.object(orchestrator, "_skip_outside_regular_market_job", return_value=False),
        patch.object(orchestrator, "get_engine", return_value=MagicMock()) as get_engine,
        patch.object(orchestrator, "_run_intraday_inner") as run_inner,
        caplog.at_level(logging.INFO),
    ):
        orchestrator.run_intraday()

    get_engine.assert_not_called()
    run_inner.assert_not_called()
    assert orchestrator._is_pm_cycle_blocking_funnel() is True
    assert any(
        "PM_CYCLE_SKIP: owner=intraday" in record.message
        and "active_owner=price_monitor" in record.message
        for record in caplog.records
    )


def test_pm_cycle_guard_clears_after_intraday_exception():
    with (
        patch.object(orchestrator, "_skip_outside_regular_market_job", return_value=False),
        patch.object(orchestrator, "get_engine", return_value=MagicMock()),
        patch.object(orchestrator, "_run_intraday_inner", side_effect=RuntimeError("boom")),
    ):
        try:
            orchestrator.run_intraday()
        except RuntimeError:
            pass

    assert orchestrator._is_pm_cycle_blocking_funnel() is False
    assert orchestrator._pm_cycle_owner is None
    assert not orchestrator._pm_cycle_lock.locked()


def test_candidate_mode_runs_pm_profiles_serially(monkeypatch):
    monkeypatch.setattr(
        orchestrator, "_candidate_mode_requires_serial_pm_profiles", lambda: True
    )

    active = 0
    max_active = 0
    calls = []

    def run_one_profile(profile_id):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        calls.append(profile_id)
        active -= 1

    orchestrator._run_pm_profile_jobs(["conservative", "moderate", "aggressive"], run_one_profile)

    assert calls == ["conservative", "moderate", "aggressive"]
    assert max_active == 1
