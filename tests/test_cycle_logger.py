"""Tests for utils/cycle_logger.py — PhaseLog, CycleSummary, CycleLogger."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from utils.cycle_logger import CycleLogger, CycleSummary, PhaseLog


# --- Stub types for testing ---


@dataclass(frozen=True)
class FakeCycleContext:
    cycle_id: str
    trigger_source: str
    focused_symbols: tuple[str, ...]


@dataclass(frozen=True)
class FakeStaleSignalSkip:
    cycle_id: str
    symbol: str
    signal_age_seconds: float
    freshness_threshold_seconds: int
    signal_timestamp: datetime
    reason: str


# --- PhaseLog dataclass tests ---


class TestPhaseLog:
    def test_frozen(self):
        now = datetime.now(timezone.utc)
        pl = PhaseLog(
            phase_name="analyst_refresh",
            started_at=now,
            ended_at=now,
            duration_seconds=0.0,
            symbols_processed=5,
            status="completed",
            details={"key": "value"},
        )
        with pytest.raises(Exception):
            pl.phase_name = "other"  # type: ignore[misc]

    def test_fields(self):
        now = datetime.now(timezone.utc)
        pl = PhaseLog(
            phase_name="pm_decisioning",
            started_at=now,
            ended_at=now,
            duration_seconds=12.5,
            symbols_processed=3,
            status="timeout",
            details={"timed_out_symbols": ["TSLA"]},
        )
        assert pl.phase_name == "pm_decisioning"
        assert pl.duration_seconds == 12.5
        assert pl.status == "timeout"
        assert pl.details == {"timed_out_symbols": ["TSLA"]}


# --- CycleSummary dataclass tests ---


class TestCycleSummary:
    def test_frozen(self):
        summary = CycleSummary(
            cycle_id="20260723_1030_scheduled_a3f2",
            trigger_source="scheduled",
            total_duration_seconds=45.0,
            phases=(),
            symbols_fresh=5,
            symbols_stale_skipped=2,
            symbols_data_unavailable=1,
            provider_calls_total=10,
            provider_calls_by_phase={"analyst_refresh": 8, "pm_decisioning": 2},
            rate_limit_events=0,
        )
        with pytest.raises(Exception):
            summary.cycle_id = "other"  # type: ignore[misc]

    def test_tuple_phases(self):
        """Phases field uses tuple for hashability in frozen dataclass."""
        summary = CycleSummary(
            cycle_id="test",
            trigger_source="manual",
            total_duration_seconds=0.0,
            phases=(),
            symbols_fresh=0,
            symbols_stale_skipped=0,
            symbols_data_unavailable=0,
            provider_calls_total=0,
            provider_calls_by_phase={},
            rate_limit_events=0,
        )
        assert isinstance(summary.phases, tuple)


# --- CycleLogger tests ---


class TestCycleLogger:
    def test_log_cycle_start_sets_internal_state(self):
        cl = CycleLogger("test_cycle_001")
        ctx = FakeCycleContext(
            cycle_id="test_cycle_001",
            trigger_source="scheduled",
            focused_symbols=("TSLA", "AAPL", "NVDA"),
        )
        cl.log_cycle_start(ctx)
        assert cl._trigger_source == "scheduled"
        assert cl._cycle_started_at is not None

    def test_log_phase_start_and_complete(self):
        cl = CycleLogger("test_cycle_002")
        ctx = FakeCycleContext(
            cycle_id="test_cycle_002",
            trigger_source="manual",
            focused_symbols=("TSLA",),
        )
        cl.log_cycle_start(ctx)
        cl.log_phase_start("analyst_refresh")
        cl.log_phase_complete("analyst_refresh", symbols_processed=1, status="completed")

        assert len(cl._phases) == 1
        assert cl._phases[0].phase_name == "analyst_refresh"
        assert cl._phases[0].symbols_processed == 1
        assert cl._phases[0].status == "completed"
        assert cl._phases[0].duration_seconds >= 0.0

    def test_log_phase_complete_with_provider_calls(self):
        cl = CycleLogger("test_cycle_003")
        ctx = FakeCycleContext(
            cycle_id="test_cycle_003",
            trigger_source="scheduled",
            focused_symbols=("TSLA", "AAPL"),
        )
        cl.log_cycle_start(ctx)
        cl.log_phase_start("analyst_refresh")
        cl.log_phase_complete("analyst_refresh", symbols_processed=2, provider_calls=8)
        cl.log_phase_start("pm_decisioning")
        cl.log_phase_complete("pm_decisioning", symbols_processed=2, provider_calls=3)

        assert cl._provider_calls_total == 11
        assert cl._provider_calls_by_phase == {"analyst_refresh": 8, "pm_decisioning": 3}

    def test_log_freshness_skip_stale(self):
        cl = CycleLogger("test_cycle_004")
        skip = FakeStaleSignalSkip(
            cycle_id="test_cycle_004",
            symbol="TSLA",
            signal_age_seconds=250.0,
            freshness_threshold_seconds=120,
            signal_timestamp=datetime(2026, 7, 23, 14, 25, 0, tzinfo=timezone.utc),
            reason="stale_signal",
        )
        cl.log_freshness_skip(skip)
        assert cl._symbols_stale_skipped == 1
        assert cl._symbols_data_unavailable == 0

    def test_log_freshness_skip_missing(self):
        cl = CycleLogger("test_cycle_005")
        skip = FakeStaleSignalSkip(
            cycle_id="test_cycle_005",
            symbol="NVDA",
            signal_age_seconds=0.0,
            freshness_threshold_seconds=120,
            signal_timestamp=datetime(2026, 7, 23, 14, 0, 0, tzinfo=timezone.utc),
            reason="missing_signal",
        )
        cl.log_freshness_skip(skip)
        assert cl._symbols_stale_skipped == 0
        assert cl._symbols_data_unavailable == 1

    def test_log_freshness_skip_error(self):
        cl = CycleLogger("test_cycle_006")
        skip = FakeStaleSignalSkip(
            cycle_id="test_cycle_006",
            symbol="AMD",
            signal_age_seconds=0.0,
            freshness_threshold_seconds=120,
            signal_timestamp=datetime(2026, 7, 23, 14, 0, 0, tzinfo=timezone.utc),
            reason="freshness_gate_error",
        )
        cl.log_freshness_skip(skip)
        assert cl._symbols_data_unavailable == 1

    def test_log_cycle_end_returns_summary(self):
        cl = CycleLogger("test_cycle_007")
        ctx = FakeCycleContext(
            cycle_id="test_cycle_007",
            trigger_source="scheduled",
            focused_symbols=("TSLA", "AAPL", "NVDA"),
        )
        cl.log_cycle_start(ctx)

        cl.log_phase_start("focus_selection")
        cl.log_phase_complete("focus_selection", symbols_processed=3)

        cl.log_phase_start("analyst_refresh")
        cl.log_phase_complete("analyst_refresh", symbols_processed=3, provider_calls=6)

        cl.set_symbols_fresh(2)

        skip = FakeStaleSignalSkip(
            cycle_id="test_cycle_007",
            symbol="NVDA",
            signal_age_seconds=300.0,
            freshness_threshold_seconds=120,
            signal_timestamp=datetime(2026, 7, 23, 14, 0, 0, tzinfo=timezone.utc),
            reason="stale_signal",
        )
        cl.log_freshness_skip(skip)

        cl.log_phase_start("pm_decisioning")
        cl.log_phase_complete("pm_decisioning", symbols_processed=2, provider_calls=2)

        summary = cl.log_cycle_end()

        assert isinstance(summary, CycleSummary)
        assert summary.cycle_id == "test_cycle_007"
        assert summary.trigger_source == "scheduled"
        assert summary.total_duration_seconds >= 0.0
        assert len(summary.phases) == 3
        assert summary.symbols_fresh == 2
        assert summary.symbols_stale_skipped == 1
        assert summary.symbols_data_unavailable == 0
        assert summary.provider_calls_total == 8
        assert summary.provider_calls_by_phase == {"analyst_refresh": 6, "pm_decisioning": 2}
        assert summary.rate_limit_events == 0

    def test_log_rate_limit_event(self):
        cl = CycleLogger("test_cycle_008")
        cl.log_rate_limit_event()
        cl.log_rate_limit_event()
        assert cl._rate_limit_events == 2

    def test_summary_phases_are_tuple(self):
        """CycleSummary.phases is a tuple (immutable, hashable)."""
        cl = CycleLogger("test_cycle_009")
        ctx = FakeCycleContext(
            cycle_id="test_cycle_009",
            trigger_source="manual",
            focused_symbols=(),
        )
        cl.log_cycle_start(ctx)
        summary = cl.log_cycle_end()
        assert isinstance(summary.phases, tuple)

    def test_logging_output(self, caplog):
        """Verify log messages are emitted via cycle_coordinator logger."""
        cl = CycleLogger("test_cycle_010")
        ctx = FakeCycleContext(
            cycle_id="test_cycle_010",
            trigger_source="manual",
            focused_symbols=("TSLA",),
        )

        with caplog.at_level(logging.INFO, logger="cycle_coordinator"):
            cl.log_cycle_start(ctx)
            cl.log_phase_start("analyst_refresh")
            cl.log_phase_complete("analyst_refresh", symbols_processed=1)
            cl.log_cycle_end()

        messages = [r.message for r in caplog.records]
        assert any("Cycle started" in m for m in messages)
        assert any("Phase started: analyst_refresh" in m for m in messages)
        assert any("Phase completed: analyst_refresh" in m for m in messages)
        assert any("Cycle completed" in m for m in messages)

    def test_phase_complete_without_prior_start(self):
        """log_phase_complete works even without a prior log_phase_start call."""
        cl = CycleLogger("test_cycle_011")
        ctx = FakeCycleContext(
            cycle_id="test_cycle_011",
            trigger_source="scheduled",
            focused_symbols=("AAPL",),
        )
        cl.log_cycle_start(ctx)
        # Complete without calling log_phase_start — should not crash
        cl.log_phase_complete("safety_checks", symbols_processed=0)
        assert len(cl._phases) == 1
        assert cl._phases[0].phase_name == "safety_checks"
        assert cl._phases[0].duration_seconds == 0.0
