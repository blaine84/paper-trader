"""Tests for utils/cycle_coordinator.py — cycle_id generation utility."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from hypothesis import given, settings, strategies as st

from utils.cycle_coordinator import generate_cycle_id


# --- Unit tests ---


class TestGenerateCycleId:
    """Unit tests for generate_cycle_id()."""

    def test_format_matches_spec(self):
        """Cycle ID follows {YYYYMMDD}_{HHMM}_{source}_{4-char-hex} format."""
        cycle_id = generate_cycle_id("scheduled")
        pattern = r"^\d{8}_\d{4}_scheduled_[0-9a-f]{4}$"
        assert re.match(pattern, cycle_id), f"Unexpected format: {cycle_id}"

    def test_manual_trigger_source(self):
        """Cycle ID includes the trigger source verbatim."""
        cycle_id = generate_cycle_id("manual")
        parts = cycle_id.split("_")
        # Format: date_time_source_hex → 4 parts for single-word source
        assert parts[2] == "manual"

    def test_alert_dispatch_trigger_source(self):
        """Multi-word trigger sources with underscores are preserved."""
        cycle_id = generate_cycle_id("alert_dispatch")
        assert "alert_dispatch" in cycle_id

    def test_hex_suffix_is_4_chars(self):
        """The hex suffix is exactly 4 lowercase hex characters."""
        cycle_id = generate_cycle_id("scheduled")
        suffix = cycle_id.rsplit("_", 1)[-1]
        assert len(suffix) == 4
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_uniqueness_across_calls(self):
        """Multiple calls in quick succession produce different IDs."""
        ids = {generate_cycle_id("scheduled") for _ in range(50)}
        # With 4 hex chars (65536 possibilities), 50 calls should all be unique
        assert len(ids) == 50

    def test_uses_utc_time(self):
        """Cycle ID date/time components reflect UTC."""
        fake_utc = datetime(2026, 7, 23, 10, 30, 0, tzinfo=timezone.utc)
        with patch("utils.cycle_coordinator.datetime") as mock_dt:
            mock_dt.now.return_value = fake_utc
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            cycle_id = generate_cycle_id("scheduled")

        assert cycle_id.startswith("20260723_1030_scheduled_")

    def test_date_part_is_8_digits(self):
        """Date component is exactly YYYYMMDD (8 digits)."""
        cycle_id = generate_cycle_id("scheduled")
        date_part = cycle_id.split("_")[0]
        assert len(date_part) == 8
        assert date_part.isdigit()

    def test_time_part_is_4_digits(self):
        """Time component is exactly HHMM (4 digits)."""
        cycle_id = generate_cycle_id("scheduled")
        time_part = cycle_id.split("_")[1]
        assert len(time_part) == 4
        assert time_part.isdigit()


# --- Property-based tests ---


# Validates: Requirements 2.1, 2.5
@given(trigger_source=st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
    min_size=1,
    max_size=30,
))
@settings(max_examples=200)
def test_prop_cycle_id_format_invariant(trigger_source: str):
    """Property: Every generated cycle_id has the correct structure regardless of trigger source.

    **Validates: Requirements 2.1, 2.5**
    """
    cycle_id = generate_cycle_id(trigger_source)

    # Must contain the trigger source
    assert trigger_source in cycle_id

    # Must end with _XXXX where X is a hex char
    suffix = cycle_id.rsplit("_", 1)[-1]
    assert len(suffix) == 4
    assert all(c in "0123456789abcdef" for c in suffix)

    # Must start with YYYYMMDD_HHMM_
    parts_before_source = cycle_id.split(f"_{trigger_source}_")[0]
    date_time_parts = parts_before_source.split("_")
    assert len(date_time_parts) == 2
    assert len(date_time_parts[0]) == 8 and date_time_parts[0].isdigit()
    assert len(date_time_parts[1]) == 4 and date_time_parts[1].isdigit()


# Validates: Requirements 2.5
@given(trigger_source=st.sampled_from(["scheduled", "manual", "alert_dispatch"]))
@settings(max_examples=200)
def test_prop_cycle_id_uniqueness(trigger_source: str):
    """Property: Two cycle_ids generated with the same source are (practically) always unique.

    **Validates: Requirements 2.5**
    """
    id1 = generate_cycle_id(trigger_source)
    id2 = generate_cycle_id(trigger_source)
    assert id1 != id2


# ---------------------------------------------------------------------------
# CycleCoordinator Tests (Task 4.2)
# ---------------------------------------------------------------------------

import time
import threading
from datetime import timedelta
from unittest.mock import MagicMock, patch, call

from sqlalchemy import create_engine

from utils.cycle_coordinator import (
    CycleCoordinator,
    CycleContext,
    get_decision_window_end,
    _cycle_decision_window_end,
)
from utils.signal_freshness import FreshnessResult, StaleSignalSkip


@pytest.fixture
def mem_engine():
    """In-memory SQLite engine for coordinator tests."""
    return create_engine("sqlite:///:memory:")


def _make_freshness_result(
    fresh: tuple[str, ...] = (),
    stale: tuple[str, ...] = (),
    missing: tuple[str, ...] = (),
    error: tuple[str, ...] = (),
) -> FreshnessResult:
    """Helper to build a FreshnessResult with empty skip_events."""
    return FreshnessResult(
        fresh_symbols=fresh,
        stale_symbols=stale,
        missing_symbols=missing,
        error_symbols=error,
        skip_events=(),
    )


# --- Phase ordering tests ---


class TestPhaseOrdering:
    """Validates: Requirements 1.1, 1.2, 6.1, 6.2"""

    @patch("agents.bookkeeper.check_stop_losses", return_value=[])
    @patch("orchestrator._end_pm_cycle")
    @patch("orchestrator._try_begin_pm_cycle", return_value=True)
    @patch("agents.portfolio_manager.run_profile", return_value={"decisions": []})
    @patch("utils.signal_freshness.check_signal_freshness")
    @patch("agents.analyst.run", return_value={})
    def test_phases_execute_in_correct_order(
        self,
        mock_analyst_run,
        mock_freshness,
        mock_pm_run_profile,
        mock_try_begin,
        mock_end_cycle,
        mock_bookkeeper,
        mem_engine,
    ):
        """Phases execute in order: analyst → freshness gate → PM → safety checks.

        Validates: Requirements 1.1, 1.2, 6.1
        """
        call_order = []

        mock_analyst_run.side_effect = lambda *a, **kw: (
            call_order.append("analyst"),
            {},
        )[1]
        mock_freshness.side_effect = lambda *a, **kw: (
            call_order.append("freshness_gate"),
            _make_freshness_result(fresh=("AAPL",)),
        )[1]
        mock_pm_run_profile.side_effect = lambda *a, **kw: (
            call_order.append("pm"),
            {"decisions": []},
        )[1]
        mock_bookkeeper.side_effect = lambda *a, **kw: (
            call_order.append("safety_checks"),
            [],
        )[1]

        coordinator = CycleCoordinator(mem_engine)
        coordinator.run_market_cycle(
            trigger_source="scheduled",
            override_symbols=["AAPL"],
        )

        assert call_order == ["analyst", "freshness_gate", "pm", "safety_checks"]


# --- PM receives only fresh symbols ---


class TestFreshnessGating:
    """Validates: Requirements 3.1, 3.5"""

    @patch("agents.bookkeeper.check_stop_losses", return_value=[])
    @patch("orchestrator._end_pm_cycle")
    @patch("orchestrator._try_begin_pm_cycle", return_value=True)
    @patch("agents.portfolio_manager.run_profile", return_value={"decisions": []})
    @patch("utils.signal_freshness.check_signal_freshness")
    @patch("agents.analyst.run", return_value={})
    def test_pm_receives_only_fresh_symbols(
        self,
        mock_analyst_run,
        mock_freshness,
        mock_pm_run_profile,
        mock_try_begin,
        mock_end_cycle,
        mock_bookkeeper,
        mem_engine,
    ):
        """PM receives only fresh symbols — stale, missing, and error are excluded.

        Validates: Requirements 3.1
        """
        mock_freshness.return_value = _make_freshness_result(
            fresh=("AAPL", "MSFT"),
            stale=("TSLA",),
            missing=("GOOG",),
            error=("AMZN",),
        )

        coordinator = CycleCoordinator(mem_engine)
        coordinator.run_market_cycle(
            trigger_source="scheduled",
            override_symbols=["AAPL", "MSFT", "TSLA", "GOOG", "AMZN"],
        )

        # PM should have been called with only fresh symbols
        mock_pm_run_profile.assert_called_once()
        pm_call_args = mock_pm_run_profile.call_args
        # run_profile(engine, fresh_symbols, profile_id, *, cycle_id=...)
        symbols_passed_to_pm = pm_call_args[0][1]
        assert set(symbols_passed_to_pm) == {"AAPL", "MSFT"}
        assert "TSLA" not in symbols_passed_to_pm
        assert "GOOG" not in symbols_passed_to_pm
        assert "AMZN" not in symbols_passed_to_pm


# --- Timeout tests ---


class TestTimeoutAdvancement:
    """Validates: Requirements 10.1, 10.2, 10.3, 10.4"""

    @patch("agents.bookkeeper.check_stop_losses", return_value=[])
    @patch("orchestrator._end_pm_cycle")
    @patch("orchestrator._try_begin_pm_cycle", return_value=True)
    @patch("agents.portfolio_manager.run_profile", return_value={"decisions": []})
    @patch("utils.signal_freshness.check_signal_freshness")
    @patch("agents.analyst.run")
    @patch("utils.gate_config.CYCLE_ANALYST_TIMEOUT_SECONDS", 1)
    def test_analyst_timeout_advances_to_freshness_gate(
        self,
        mock_analyst_run,
        mock_freshness,
        mock_pm_run_profile,
        mock_try_begin,
        mock_end_cycle,
        mock_bookkeeper,
        mem_engine,
    ):
        """When analyst times out, cycle advances to freshness gate phase.

        Validates: Requirements 10.1, 10.2
        """

        def slow_analyst(*args, **kwargs):
            time.sleep(3)  # Longer than the 1s timeout
            return {}

        mock_analyst_run.side_effect = slow_analyst
        mock_freshness.return_value = _make_freshness_result(fresh=("AAPL",))

        coordinator = CycleCoordinator(mem_engine)
        summary = coordinator.run_market_cycle(
            trigger_source="scheduled",
            override_symbols=["AAPL"],
        )

        # Freshness gate and PM should still have been called despite analyst timeout
        mock_freshness.assert_called_once()
        mock_pm_run_profile.assert_called_once()

        # Analyst phase should be recorded as timeout
        analyst_phase = next(
            (p for p in summary.phases if p.phase_name == "analyst_refresh"), None
        )
        assert analyst_phase is not None
        assert analyst_phase.status == "timeout"

    @patch("agents.bookkeeper.check_stop_losses", return_value=[])
    @patch("orchestrator._end_pm_cycle")
    @patch("orchestrator._try_begin_pm_cycle", return_value=True)
    @patch("agents.portfolio_manager.run_profile")
    @patch("utils.signal_freshness.check_signal_freshness")
    @patch("agents.analyst.run", return_value={})
    @patch("utils.gate_config.CYCLE_PM_TIMEOUT_SECONDS", 1)
    def test_pm_timeout_advances_to_safety_checks(
        self,
        mock_analyst_run,
        mock_freshness,
        mock_pm_run_profile,
        mock_try_begin,
        mock_end_cycle,
        mock_bookkeeper,
        mem_engine,
    ):
        """When PM times out, cycle advances to safety checks.

        Validates: Requirements 10.3, 10.4
        """

        def slow_pm(*args, **kwargs):
            time.sleep(3)  # Longer than the 1s timeout
            return {"decisions": []}

        mock_pm_run_profile.side_effect = slow_pm
        mock_freshness.return_value = _make_freshness_result(fresh=("AAPL",))

        coordinator = CycleCoordinator(mem_engine)
        summary = coordinator.run_market_cycle(
            trigger_source="scheduled",
            override_symbols=["AAPL"],
        )

        # Safety checks should still run despite PM timeout
        mock_bookkeeper.assert_called_once()

        # PM phase should be recorded as timeout
        pm_phase = next(
            (p for p in summary.phases if p.phase_name == "pm_decisioning"), None
        )
        assert pm_phase is not None
        assert pm_phase.status == "timeout"


# --- Cycle ID propagation ---


class TestCycleIdPropagation:
    """Validates: Requirements 2.1, 2.2, 2.3"""

    @patch("agents.bookkeeper.check_stop_losses", return_value=[])
    @patch("orchestrator._end_pm_cycle")
    @patch("orchestrator._try_begin_pm_cycle", return_value=True)
    @patch("agents.portfolio_manager.run_profile", return_value={"decisions": []})
    @patch("utils.signal_freshness.check_signal_freshness")
    @patch("agents.analyst.run", return_value={})
    def test_cycle_id_propagates_to_analyst_and_pm(
        self,
        mock_analyst_run,
        mock_freshness,
        mock_pm_run_profile,
        mock_try_begin,
        mock_end_cycle,
        mock_bookkeeper,
        mem_engine,
    ):
        """Cycle ID propagates to both analyst.run() and pm.run_profile() calls.

        Validates: Requirements 2.2, 2.3
        """
        mock_freshness.return_value = _make_freshness_result(fresh=("AAPL",))

        coordinator = CycleCoordinator(mem_engine)
        coordinator.run_market_cycle(
            trigger_source="scheduled",
            override_symbols=["AAPL"],
        )

        # analyst.run() receives cycle_id kwarg
        analyst_call_kwargs = mock_analyst_run.call_args[1]
        assert "cycle_id" in analyst_call_kwargs
        analyst_cycle_id = analyst_call_kwargs["cycle_id"]
        assert analyst_cycle_id is not None

        # pm.run_profile() receives same cycle_id kwarg
        pm_call_kwargs = mock_pm_run_profile.call_args[1]
        assert "cycle_id" in pm_call_kwargs
        assert pm_call_kwargs["cycle_id"] == analyst_cycle_id


# --- Decision window state ---


class TestDecisionWindow:
    """Validates: Requirements 1.6, 10.8"""

    @patch("agents.bookkeeper.check_stop_losses", return_value=[])
    @patch("orchestrator._end_pm_cycle")
    @patch("orchestrator._try_begin_pm_cycle", return_value=True)
    @patch("agents.portfolio_manager.run_profile", return_value={"decisions": []})
    @patch("utils.signal_freshness.check_signal_freshness")
    @patch("agents.analyst.run", return_value={})
    def test_decision_window_set_during_cycle_cleared_after(
        self,
        mock_analyst_run,
        mock_freshness,
        mock_pm_run_profile,
        mock_try_begin,
        mock_end_cycle,
        mock_bookkeeper,
        mem_engine,
    ):
        """decision_window_end is set at cycle start and cleared at cycle end.

        Validates: Requirements 1.6
        """
        import utils.cycle_coordinator as cc_module

        window_values_during_cycle = []

        def capture_window_in_analyst(*args, **kwargs):
            window_values_during_cycle.append(get_decision_window_end())
            return {}

        mock_analyst_run.side_effect = capture_window_in_analyst
        mock_freshness.return_value = _make_freshness_result(fresh=("AAPL",))

        # Before cycle: should be None
        assert get_decision_window_end() is None

        coordinator = CycleCoordinator(mem_engine)
        coordinator.run_market_cycle(
            trigger_source="scheduled",
            override_symbols=["AAPL"],
        )

        # During cycle: should have been set (captured in analyst side_effect)
        assert len(window_values_during_cycle) == 1
        assert window_values_during_cycle[0] is not None

        # After cycle: should be cleared back to None
        assert get_decision_window_end() is None


# --- CycleSummary accuracy ---


class TestCycleSummary:
    """Validates: Requirements 8.1, 8.2, 8.3"""

    @patch("agents.bookkeeper.check_stop_losses", return_value=[])
    @patch("orchestrator._end_pm_cycle")
    @patch("orchestrator._try_begin_pm_cycle", return_value=True)
    @patch("agents.portfolio_manager.run_profile", return_value={"decisions": []})
    @patch("utils.signal_freshness.check_signal_freshness")
    @patch("agents.analyst.run", return_value={})
    def test_cycle_summary_contains_accurate_phase_durations_and_counts(
        self,
        mock_analyst_run,
        mock_freshness,
        mock_pm_run_profile,
        mock_try_begin,
        mock_end_cycle,
        mock_bookkeeper,
        mem_engine,
    ):
        """CycleSummary contains accurate phase durations and counts.

        Validates: Requirements 8.2, 8.3
        """
        mock_freshness.return_value = _make_freshness_result(
            fresh=("AAPL", "MSFT"),
            stale=("TSLA",),
        )

        coordinator = CycleCoordinator(mem_engine)
        summary = coordinator.run_market_cycle(
            trigger_source="scheduled",
            override_symbols=["AAPL", "MSFT", "TSLA"],
        )

        # Summary should have correct cycle info
        assert summary.cycle_id is not None
        assert summary.trigger_source == "scheduled"
        assert summary.total_duration_seconds >= 0

        # Phase logs should exist and have non-negative durations
        assert len(summary.phases) >= 3  # at least analyst, freshness, pm
        for phase in summary.phases:
            assert phase.duration_seconds >= 0
            assert phase.started_at is not None
            assert phase.ended_at is not None
            assert phase.ended_at >= phase.started_at

        # Freshness counts should be accurate
        assert summary.symbols_fresh == 2
        assert summary.symbols_stale_skipped >= 0


# --- Exception resilience ---


class TestExceptionResilience:
    """Validates: Requirements 1.1, 10.5"""

    @patch("agents.bookkeeper.check_stop_losses", return_value=[])
    @patch("orchestrator._end_pm_cycle")
    @patch("orchestrator._try_begin_pm_cycle", return_value=True)
    @patch("agents.portfolio_manager.run_profile", return_value={"decisions": []})
    @patch("utils.signal_freshness.check_signal_freshness")
    @patch("agents.analyst.run")
    def test_exception_in_one_phase_does_not_prevent_subsequent_phases(
        self,
        mock_analyst_run,
        mock_freshness,
        mock_pm_run_profile,
        mock_try_begin,
        mock_end_cycle,
        mock_bookkeeper,
        mem_engine,
    ):
        """Exception in analyst phase does not prevent freshness gate and PM from running.

        Validates: Requirements 1.1
        """
        mock_analyst_run.side_effect = RuntimeError("LLM provider error")
        mock_freshness.return_value = _make_freshness_result(fresh=("AAPL",))

        coordinator = CycleCoordinator(mem_engine)
        summary = coordinator.run_market_cycle(
            trigger_source="scheduled",
            override_symbols=["AAPL"],
        )

        # Freshness gate and PM should still execute
        mock_freshness.assert_called_once()
        mock_pm_run_profile.assert_called_once()
        mock_bookkeeper.assert_called_once()

        # Analyst phase recorded as error
        analyst_phase = next(
            (p for p in summary.phases if p.phase_name == "analyst_refresh"), None
        )
        assert analyst_phase is not None
        assert analyst_phase.status == "error"

    @patch("agents.bookkeeper.check_stop_losses", return_value=[])
    @patch("orchestrator._end_pm_cycle")
    @patch("orchestrator._try_begin_pm_cycle", return_value=True)
    @patch("agents.portfolio_manager.run_profile")
    @patch("utils.signal_freshness.check_signal_freshness")
    @patch("agents.analyst.run", return_value={})
    def test_exception_in_pm_does_not_prevent_safety_checks(
        self,
        mock_analyst_run,
        mock_freshness,
        mock_pm_run_profile,
        mock_try_begin,
        mock_end_cycle,
        mock_bookkeeper,
        mem_engine,
    ):
        """Exception in PM phase does not prevent safety checks from running.

        Validates: Requirements 1.1
        """
        mock_freshness.return_value = _make_freshness_result(fresh=("AAPL",))
        mock_pm_run_profile.side_effect = RuntimeError("PM crash")

        coordinator = CycleCoordinator(mem_engine)
        summary = coordinator.run_market_cycle(
            trigger_source="scheduled",
            override_symbols=["AAPL"],
        )

        # Safety checks should still run
        mock_bookkeeper.assert_called_once()

        # PM phase recorded as error
        pm_phase = next(
            (p for p in summary.phases if p.phase_name == "pm_decisioning"), None
        )
        assert pm_phase is not None
        assert pm_phase.status == "error"


# --- Late write protection ---


class TestLateWriteProtection:
    """Validates: Requirements 10.6, 10.7, 10.8"""

    @patch("agents.bookkeeper.check_stop_losses", return_value=[])
    @patch("orchestrator._end_pm_cycle")
    @patch("orchestrator._try_begin_pm_cycle", return_value=True)
    @patch("agents.portfolio_manager.run_profile", return_value={"decisions": []})
    @patch("utils.signal_freshness.check_signal_freshness")
    @patch("agents.analyst.run", return_value={})
    def test_late_analyst_write_after_pm_completes_does_not_retrigger_pm(
        self,
        mock_analyst_run,
        mock_freshness,
        mock_pm_run_profile,
        mock_try_begin,
        mock_end_cycle,
        mock_bookkeeper,
        mem_engine,
    ):
        """After PM phase completes, pm_phase_completed is True — late writes cannot re-trigger PM.

        Validates: Requirements 10.6, 10.7, 10.8
        """
        mock_freshness.return_value = _make_freshness_result(fresh=("AAPL",))

        coordinator = CycleCoordinator(mem_engine)

        # Before cycle, PM is not marked complete
        assert coordinator.pm_phase_completed is False

        coordinator.run_market_cycle(
            trigger_source="scheduled",
            override_symbols=["AAPL"],
        )

        # After cycle, PM phase is marked complete — the authoritative marker
        # that prevents late analyst writes from reviving the PM phase
        assert coordinator.pm_phase_completed is True

    @patch("agents.bookkeeper.check_stop_losses", return_value=[])
    @patch("orchestrator._end_pm_cycle")
    @patch("orchestrator._try_begin_pm_cycle", return_value=True)
    @patch("agents.portfolio_manager.run_profile")
    @patch("utils.signal_freshness.check_signal_freshness")
    @patch("agents.analyst.run", return_value={})
    @patch("utils.gate_config.CYCLE_PM_TIMEOUT_SECONDS", 1)
    def test_pm_phase_completed_set_even_on_timeout(
        self,
        mock_analyst_run,
        mock_freshness,
        mock_pm_run_profile,
        mock_try_begin,
        mock_end_cycle,
        mock_bookkeeper,
        mem_engine,
    ):
        """pm_phase_completed is set True even when PM times out.

        Validates: Requirements 10.8
        """

        def slow_pm(*args, **kwargs):
            time.sleep(3)
            return {"decisions": []}

        mock_pm_run_profile.side_effect = slow_pm
        mock_freshness.return_value = _make_freshness_result(fresh=("AAPL",))

        coordinator = CycleCoordinator(mem_engine)
        coordinator.run_market_cycle(
            trigger_source="scheduled",
            override_symbols=["AAPL"],
        )

        # Even on timeout, PM phase is marked complete
        assert coordinator.pm_phase_completed is True
