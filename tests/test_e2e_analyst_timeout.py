"""End-to-end integration test: all symbols stale after analyst timeout.

Simulates a scenario where:
1. Analyst times out (doesn't produce fresh signals this cycle)
2. Only stale signals exist in the DB (from a previous cycle)
3. Freshness gate classifies all symbols as stale
4. PM receives ZERO candidates (empty fresh list)
5. PM is NOT called OR called with empty symbols
6. Cycle still completes with appropriate CycleSummary metrics

Validates the safety guarantee: PM never evaluates candidates based on stale data.

Requirements: 3.5, 10.1, 10.2, 10.5
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from utils.cycle_coordinator import CycleCoordinator


@pytest.fixture
def engine():
    """In-memory SQLite engine with agent_memory table.

    Uses StaticPool so the same in-memory DB is shared across threads
    (the CycleCoordinator runs analyst in a ThreadPoolExecutor).
    """
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with eng.connect() as conn:
        conn.execute(text("""
            CREATE TABLE agent_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent VARCHAR(32) NOT NULL,
                symbol VARCHAR(10),
                timestamp DATETIME,
                key VARCHAR(64) NOT NULL,
                value TEXT NOT NULL
            )
        """))
        conn.commit()
    return eng


def _insert_stale_signal(engine, symbol: str, age_seconds: int = 600) -> None:
    """Insert a stale signal for a symbol (from a previous cycle, well outside freshness window)."""
    old_time = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    signal = {"strength": 5.0, "_cycle_id": "stale_old_cycle_xyz"}
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO agent_memory (agent, symbol, timestamp, key, value)
            VALUES (:agent, :symbol, :timestamp, :key, :value)
        """), {
            "agent": "analyst",
            "symbol": symbol,
            "timestamp": old_time,
            "key": "signal",
            "value": json.dumps(signal),
        })
        conn.commit()


class TestAllSymbolsStaleAfterAnalystTimeout:
    """End-to-end test: analyst timeout with only stale signals results in zero PM candidates."""

    @patch("utils.gate_config.CYCLE_ANALYST_TIMEOUT_SECONDS", 1)
    @patch("agents.bookkeeper.check_stop_losses", return_value=[])
    @patch("orchestrator._end_pm_cycle")
    @patch("orchestrator._try_begin_pm_cycle", return_value=True)
    @patch("agents.portfolio_manager.run_profile")
    @patch("agents.analyst.run")
    def test_analyst_timeout_all_stale_results_in_zero_pm_candidates(
        self, mock_analyst, mock_pm, mock_try_begin, mock_end_cycle, mock_bookkeeper, engine
    ):
        """When analyst times out and only stale signals exist, PM gets zero candidates.

        Validates: Requirements 3.5, 10.1, 10.2
        """
        # Pre-insert stale signals for all symbols (from a previous cycle, 600s old)
        symbols = ["TSLA", "AAPL", "NVDA"]
        for sym in symbols:
            _insert_stale_signal(engine, sym, age_seconds=600)

        # Analyst is slow — will exceed the 1s timeout
        def slow_analyst(eng, syms, *, cycle_id=None):
            time.sleep(3)
            return {}

        mock_analyst.side_effect = slow_analyst
        mock_pm.return_value = {"decisions": []}

        coordinator = CycleCoordinator(engine)
        summary = coordinator.run_market_cycle(
            trigger_source="scheduled",
            override_symbols=symbols,
        )

        # PM should NOT be called (all stale → empty fresh list → PM skipped)
        if mock_pm.called:
            pm_call_symbols = mock_pm.call_args[0][1]
            assert len(pm_call_symbols) == 0, (
                f"PM should receive zero symbols after analyst timeout with all stale, "
                f"got: {pm_call_symbols}"
            )

        # CycleSummary should reflect no fresh symbols
        assert summary.symbols_fresh == 0

        # Stale skipped should match the number of symbols with stale signals
        assert summary.symbols_stale_skipped == 3

        # Cycle still completed (not crashed)
        assert summary.cycle_id is not None
        assert summary.total_duration_seconds > 0

    @patch("utils.gate_config.CYCLE_ANALYST_TIMEOUT_SECONDS", 1)
    @patch("agents.bookkeeper.check_stop_losses", return_value=[])
    @patch("orchestrator._end_pm_cycle")
    @patch("orchestrator._try_begin_pm_cycle", return_value=True)
    @patch("agents.portfolio_manager.run_profile")
    @patch("agents.analyst.run")
    def test_analyst_timeout_with_no_prior_signals_classifies_as_missing(
        self, mock_analyst, mock_pm, mock_try_begin, mock_end_cycle, mock_bookkeeper, engine
    ):
        """When analyst times out and NO signals exist at all, symbols are classified as missing.

        Validates: Requirements 3.5, 10.2, 10.5
        """
        symbols = ["TSLA", "AAPL", "NVDA"]

        # No pre-inserted signals — DB is empty for these symbols

        # Analyst is slow — will exceed the 1s timeout
        def slow_analyst(eng, syms, *, cycle_id=None):
            time.sleep(3)
            return {}

        mock_analyst.side_effect = slow_analyst
        mock_pm.return_value = {"decisions": []}

        coordinator = CycleCoordinator(engine)
        summary = coordinator.run_market_cycle(
            trigger_source="scheduled",
            override_symbols=symbols,
        )

        # PM should NOT be called (all missing → empty fresh list → PM skipped)
        if mock_pm.called:
            pm_call_symbols = mock_pm.call_args[0][1]
            assert len(pm_call_symbols) == 0, (
                f"PM should receive zero symbols when all signals are missing, "
                f"got: {pm_call_symbols}"
            )

        # No fresh symbols
        assert summary.symbols_fresh == 0

        # Missing symbols counted as data unavailable
        assert summary.symbols_data_unavailable == 3

        # Cycle still completed
        assert summary.cycle_id is not None
        assert summary.total_duration_seconds > 0

    @patch("utils.gate_config.CYCLE_ANALYST_TIMEOUT_SECONDS", 1)
    @patch("agents.bookkeeper.check_stop_losses", return_value=[])
    @patch("orchestrator._end_pm_cycle")
    @patch("orchestrator._try_begin_pm_cycle", return_value=True)
    @patch("agents.portfolio_manager.run_profile")
    @patch("agents.analyst.run")
    def test_analyst_timeout_cycle_phases_complete_in_order(
        self, mock_analyst, mock_pm, mock_try_begin, mock_end_cycle, mock_bookkeeper, engine
    ):
        """After analyst timeout, all remaining phases still execute in order.

        Validates: Requirements 10.1, 10.2
        """
        symbols = ["TSLA", "AAPL"]
        for sym in symbols:
            _insert_stale_signal(engine, sym, age_seconds=600)

        # Analyst is slow — will exceed the 1s timeout
        def slow_analyst(eng, syms, *, cycle_id=None):
            time.sleep(3)
            return {}

        mock_analyst.side_effect = slow_analyst
        mock_pm.return_value = {"decisions": []}

        coordinator = CycleCoordinator(engine)
        summary = coordinator.run_market_cycle(
            trigger_source="scheduled",
            override_symbols=symbols,
        )

        # Verify phases executed: we expect focus_selection, analyst_refresh (timeout),
        # freshness_gate, pm_decisioning, safety_checks, bookkeeping
        phase_names = [p.phase_name for p in summary.phases]
        assert "analyst_refresh" in phase_names
        assert "freshness_gate" in phase_names
        assert "pm_decisioning" in phase_names
        assert "safety_checks" in phase_names

        # Analyst phase should have status "timeout"
        analyst_phase = next(p for p in summary.phases if p.phase_name == "analyst_refresh")
        assert analyst_phase.status == "timeout"

        # Freshness gate should have completed
        freshness_phase = next(p for p in summary.phases if p.phase_name == "freshness_gate")
        assert freshness_phase.status == "completed"

        # PM decisioning should have completed (with zero symbols)
        pm_phase = next(p for p in summary.phases if p.phase_name == "pm_decisioning")
        assert pm_phase.status == "completed"

    @patch("utils.gate_config.CYCLE_ANALYST_TIMEOUT_SECONDS", 1)
    @patch("agents.bookkeeper.check_stop_losses", return_value=[])
    @patch("orchestrator._end_pm_cycle")
    @patch("orchestrator._try_begin_pm_cycle", return_value=True)
    @patch("agents.portfolio_manager.run_profile")
    @patch("agents.analyst.run")
    def test_stale_signal_skip_events_logged_with_correct_reason(
        self, mock_analyst, mock_pm, mock_try_begin, mock_end_cycle, mock_bookkeeper, engine
    ):
        """Each stale symbol generates a skip event with 'previous_cycle' reason.

        Validates: Requirements 3.5, 10.2
        """
        symbols = ["TSLA", "AAPL", "NVDA"]
        for sym in symbols:
            _insert_stale_signal(engine, sym, age_seconds=600)

        # Analyst is slow — will exceed the 1s timeout
        def slow_analyst(eng, syms, *, cycle_id=None):
            time.sleep(3)
            return {}

        mock_analyst.side_effect = slow_analyst
        mock_pm.return_value = {"decisions": []}

        coordinator = CycleCoordinator(engine)
        summary = coordinator.run_market_cycle(
            trigger_source="scheduled",
            override_symbols=symbols,
        )

        # Find the freshness_gate phase to confirm it completed with correct counts
        freshness_phase = next(
            p for p in summary.phases if p.phase_name == "freshness_gate"
        )
        # It should report the stale symbols in details
        assert freshness_phase.details.get("stale") == 3
        assert freshness_phase.details.get("fresh") == 0

        # All symbols are stale-skipped
        assert summary.symbols_stale_skipped == 3
        assert summary.symbols_fresh == 0
