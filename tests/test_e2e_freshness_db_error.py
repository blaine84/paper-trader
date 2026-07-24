"""End-to-end integration test: freshness gate DB error is fail-closed.

Simulates a database error during the freshness gate and verifies:
1. All symbols are classified as error (fail-closed)
2. PM receives ZERO candidates (error symbols never in fresh_symbols)
3. CycleSummary records appropriate error metrics
4. Cycle still completes (fail-open at the phase level, fail-closed at the gate level)

The freshness gate is the ONE fail-closed component in the coordinator.
If we cannot verify freshness, we do not evaluate.

Requirements: 3.8, 3.9, 10.5
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

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


def _insert_fresh_signal(engine, symbol: str, cycle_id: str) -> None:
    """Insert a fresh signal for a symbol that matches the current cycle_id."""
    now = datetime.now(timezone.utc)
    signal = {"strength": 8.0, "setup_type": "momentum_fade", "_cycle_id": cycle_id}
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO agent_memory (agent, symbol, timestamp, key, value)
            VALUES (:agent, :symbol, :timestamp, :key, :value)
        """), {
            "agent": "analyst",
            "symbol": symbol,
            "timestamp": now,
            "key": "signal",
            "value": json.dumps(signal),
        })
        conn.commit()


class TestFreshnessGateDbErrorFailClosed:
    """End-to-end test: DB error in freshness gate results in fail-closed behavior."""

    @patch("utils.signal_freshness._fetch_latest_signals")
    @patch("agents.bookkeeper.check_stop_losses", return_value=[])
    @patch("orchestrator._end_pm_cycle")
    @patch("orchestrator._try_begin_pm_cycle", return_value=True)
    @patch("agents.portfolio_manager.run_profile")
    @patch("agents.analyst.run")
    def test_db_error_classifies_all_symbols_as_error(
        self,
        mock_analyst,
        mock_pm,
        mock_try_begin,
        mock_end_cycle,
        mock_bookkeeper,
        mock_fetch,
        engine,
    ):
        """When DB fails during freshness gate, ALL symbols classified as error.

        Validates: Requirements 3.8, 3.9, 10.5
        """
        # Simulate a DB connection failure during freshness gate
        mock_fetch.side_effect = RuntimeError("Simulated DB connection failure")

        # Analyst "completes" but freshness gate can't read DB
        mock_analyst.return_value = {"TSLA": {}, "AAPL": {}, "NVDA": {}}
        mock_pm.return_value = {"decisions": []}

        symbols = ["TSLA", "AAPL", "NVDA"]

        coordinator = CycleCoordinator(engine)
        summary = coordinator.run_market_cycle(
            trigger_source="scheduled",
            override_symbols=symbols,
        )

        # PM should receive ZERO symbols (all classified as error, fail-closed)
        if mock_pm.called:
            pm_call_symbols = mock_pm.call_args[0][1]
            assert len(pm_call_symbols) == 0, (
                f"PM should receive zero symbols when DB error occurs in freshness gate, "
                f"got: {pm_call_symbols}"
            )

        # CycleSummary: no fresh symbols
        assert summary.symbols_fresh == 0

        # Error symbols counted as data_unavailable in the logger
        assert summary.symbols_data_unavailable == len(symbols)

        # Cycle still completed (not crashed)
        assert summary.cycle_id is not None
        assert summary.total_duration_seconds > 0

    @patch("utils.signal_freshness._fetch_latest_signals")
    @patch("agents.bookkeeper.check_stop_losses", return_value=[])
    @patch("orchestrator._end_pm_cycle")
    @patch("orchestrator._try_begin_pm_cycle", return_value=True)
    @patch("agents.portfolio_manager.run_profile")
    @patch("agents.analyst.run")
    def test_db_error_pm_receives_zero_candidates(
        self,
        mock_analyst,
        mock_pm,
        mock_try_begin,
        mock_end_cycle,
        mock_bookkeeper,
        mock_fetch,
        engine,
    ):
        """PM is either not called or called with empty symbol list on DB error.

        Validates: Requirements 3.8, 3.9
        """
        mock_fetch.side_effect = RuntimeError("Simulated DB connection failure")

        mock_analyst.return_value = {"TSLA": {}, "AAPL": {}}
        mock_pm.return_value = {"decisions": []}

        coordinator = CycleCoordinator(engine)
        summary = coordinator.run_market_cycle(
            trigger_source="scheduled",
            override_symbols=["TSLA", "AAPL"],
        )

        # Either PM not called at all (preferred when fresh_symbols is empty)
        # or called with zero symbols
        if mock_pm.called:
            pm_call_symbols = mock_pm.call_args[0][1]
            assert pm_call_symbols == [], (
                f"PM should not receive any candidates on DB error, "
                f"got: {pm_call_symbols}"
            )
        # If PM was not called, that's also correct (empty fresh list skips PM)

    @patch("utils.signal_freshness._fetch_latest_signals")
    @patch("agents.bookkeeper.check_stop_losses", return_value=[])
    @patch("orchestrator._end_pm_cycle")
    @patch("orchestrator._try_begin_pm_cycle", return_value=True)
    @patch("agents.portfolio_manager.run_profile")
    @patch("agents.analyst.run")
    def test_db_error_cycle_summary_records_error_metrics(
        self,
        mock_analyst,
        mock_pm,
        mock_try_begin,
        mock_end_cycle,
        mock_bookkeeper,
        mock_fetch,
        engine,
    ):
        """CycleSummary correctly records error metrics from freshness gate DB failure.

        Validates: Requirements 3.8, 10.5
        """
        mock_fetch.side_effect = RuntimeError("Simulated DB connection failure")
        mock_analyst.return_value = {"TSLA": {}, "AAPL": {}, "NVDA": {}}
        mock_pm.return_value = {"decisions": []}

        symbols = ["TSLA", "AAPL", "NVDA"]

        coordinator = CycleCoordinator(engine)
        summary = coordinator.run_market_cycle(
            trigger_source="scheduled",
            override_symbols=symbols,
        )

        # Freshness gate phase should still complete (fail-closed at gate level,
        # fail-open at phase level — the phase doesn't crash)
        freshness_phase = next(
            (p for p in summary.phases if p.phase_name == "freshness_gate"),
            None,
        )
        assert freshness_phase is not None
        assert freshness_phase.status == "completed"

        # The error count should reflect all symbols as error
        assert freshness_phase.details.get("error") == 3
        assert freshness_phase.details.get("fresh") == 0
        assert freshness_phase.details.get("stale") == 0
        assert freshness_phase.details.get("missing") == 0

        # Summary-level metrics
        assert summary.symbols_fresh == 0
        assert summary.symbols_data_unavailable == 3

    @patch("utils.signal_freshness._fetch_latest_signals")
    @patch("agents.bookkeeper.check_stop_losses", return_value=[])
    @patch("orchestrator._end_pm_cycle")
    @patch("orchestrator._try_begin_pm_cycle", return_value=True)
    @patch("agents.portfolio_manager.run_profile")
    @patch("agents.analyst.run")
    def test_db_error_cycle_still_completes_all_phases(
        self,
        mock_analyst,
        mock_pm,
        mock_try_begin,
        mock_end_cycle,
        mock_bookkeeper,
        mock_fetch,
        engine,
    ):
        """Cycle completes all phases even when freshness gate encounters DB error.

        This verifies the key distinction: fail-closed at the gate level
        (no symbols pass through) but fail-open at the phase level (cycle
        continues to safety checks and bookkeeping).

        Validates: Requirements 3.8, 10.5
        """
        mock_fetch.side_effect = RuntimeError("Simulated DB connection failure")
        mock_analyst.return_value = {"TSLA": {}}
        mock_pm.return_value = {"decisions": []}

        coordinator = CycleCoordinator(engine)
        summary = coordinator.run_market_cycle(
            trigger_source="scheduled",
            override_symbols=["TSLA"],
        )

        # All phases should be present in the summary
        phase_names = [p.phase_name for p in summary.phases]
        assert "analyst_refresh" in phase_names
        assert "freshness_gate" in phase_names
        assert "pm_decisioning" in phase_names
        assert "safety_checks" in phase_names
        assert "bookkeeping" in phase_names

        # Safety checks still ran (bookkeeper was called)
        mock_bookkeeper.assert_called_once()

        # Cycle completed with valid metadata
        assert summary.cycle_id is not None
        assert summary.total_duration_seconds > 0
        assert summary.trigger_source == "scheduled"

    @patch("utils.signal_freshness._fetch_latest_signals")
    @patch("agents.bookkeeper.check_stop_losses", return_value=[])
    @patch("orchestrator._end_pm_cycle")
    @patch("orchestrator._try_begin_pm_cycle", return_value=True)
    @patch("agents.portfolio_manager.run_profile")
    @patch("agents.analyst.run")
    def test_db_error_skip_events_have_correct_reason(
        self,
        mock_analyst,
        mock_pm,
        mock_try_begin,
        mock_end_cycle,
        mock_bookkeeper,
        mock_fetch,
        engine,
    ):
        """Skip events from DB error have reason='freshness_gate_error'.

        Validates: Requirements 3.8, 3.9
        """
        mock_fetch.side_effect = RuntimeError("Simulated DB connection failure")
        mock_analyst.return_value = {"TSLA": {}, "AAPL": {}}
        mock_pm.return_value = {"decisions": []}

        symbols = ["TSLA", "AAPL"]

        coordinator = CycleCoordinator(engine)
        # Access the freshness result directly to inspect skip events
        from utils.signal_freshness import check_signal_freshness

        with patch("utils.signal_freshness._fetch_latest_signals", side_effect=RuntimeError("DB failure")):
            result = check_signal_freshness(
                engine,
                symbols,
                "test_cycle_123",
                freshness_window_seconds=120,
            )

        # All symbols in error_symbols
        assert set(result.error_symbols) == {"TSLA", "AAPL"}

        # No symbols in fresh (fail-closed)
        assert len(result.fresh_symbols) == 0

        # Each skip event has the correct reason
        assert len(result.skip_events) == 2
        for event in result.skip_events:
            assert event.reason == "freshness_gate_error"
            assert event.cycle_id == "test_cycle_123"
            assert event.symbol in ("TSLA", "AAPL")
