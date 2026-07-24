"""Tests verifying cycle_id propagation through analyst and PM.

Validates: Requirements 2.2, 2.3, 2.4, 9.5
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy import create_engine

from db.schema import Base, AgentMemory, get_session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mem_engine():
    """In-memory SQLite engine with full schema for analyst/PM tests."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


# ---------------------------------------------------------------------------
# Analyst cycle_id propagation tests
# ---------------------------------------------------------------------------


class TestAnalystCycleIdPropagation:
    """Validates: Requirements 2.2, 9.5"""

    @patch("agents.analyst.process_pending_feedback")
    @patch("agents.analyst.write_feedback_health_status")
    @patch("agents.analyst.build_strategy_context", return_value="")
    @patch("agents.analyst.build_feedback_prompt_context", return_value="")
    @patch("agents.analyst.get_active_mitigations", return_value=[])
    @patch("utils.strategy_store.get_all_setup_types", return_value=set())
    @patch("agents.analyst.FinnhubClient")
    def test_analyst_stamps_cycle_id_into_signal(
        self,
        mock_fh_class,
        mock_setup_types,
        mock_mitigations,
        mock_feedback_ctx,
        mock_strategy_ctx,
        mock_write_health,
        mock_process_feedback,
        mem_engine,
    ):
        """When cycle_id is provided, analyst saves _cycle_id in signal JSON.

        Validates: Requirements 2.2
        """
        # FinnhubClient.get_quote raises → _analyze_symbol catches, produces error signal
        mock_fh = MagicMock()
        mock_fh.get_quote.side_effect = RuntimeError("test: no real API")
        mock_fh_class.return_value = mock_fh

        from agents.analyst import run as analyst_run
        signals = analyst_run(mem_engine, ["TSLA"], cycle_id="test_cycle_abc123")

        # Verify _cycle_id is stamped in the returned signal dict
        assert "TSLA" in signals
        assert signals["TSLA"]["_cycle_id"] == "test_cycle_abc123"

        # Verify _cycle_id is persisted in AgentMemory
        db = get_session(mem_engine)
        row = (
            db.query(AgentMemory)
            .filter_by(agent="analyst", symbol="TSLA", key="signal")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        assert row is not None
        signal_data = json.loads(row.value)
        assert signal_data["_cycle_id"] == "test_cycle_abc123"
        db.close()

    @patch("agents.analyst.process_pending_feedback")
    @patch("agents.analyst.write_feedback_health_status")
    @patch("agents.analyst.build_strategy_context", return_value="")
    @patch("agents.analyst.build_feedback_prompt_context", return_value="")
    @patch("agents.analyst.get_active_mitigations", return_value=[])
    @patch("utils.strategy_store.get_all_setup_types", return_value=set())
    @patch("agents.analyst.FinnhubClient")
    def test_analyst_stamps_null_cycle_id_when_none(
        self,
        mock_fh_class,
        mock_setup_types,
        mock_mitigations,
        mock_feedback_ctx,
        mock_strategy_ctx,
        mock_write_health,
        mock_process_feedback,
        mem_engine,
    ):
        """When cycle_id is None (legacy path), signal JSON has _cycle_id: null.

        Validates: Requirements 9.5 (backward compat)
        """
        mock_fh = MagicMock()
        mock_fh.get_quote.side_effect = RuntimeError("test: no real API")
        mock_fh_class.return_value = mock_fh

        from agents.analyst import run as analyst_run
        signals = analyst_run(mem_engine, ["AAPL"], cycle_id=None)

        # Verify _cycle_id is None in the returned signal
        assert signals["AAPL"]["_cycle_id"] is None

        # Verify _cycle_id is null in persisted JSON
        db = get_session(mem_engine)
        row = (
            db.query(AgentMemory)
            .filter_by(agent="analyst", symbol="AAPL", key="signal")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        assert row is not None
        signal_data = json.loads(row.value)
        assert signal_data["_cycle_id"] is None
        db.close()


# ---------------------------------------------------------------------------
# PM cycle_id propagation tests
# ---------------------------------------------------------------------------


class TestPMCycleIdPropagation:
    """Validates: Requirements 2.3, 9.5"""

    _FAKE_PORTFOLIO = {
        "starting_balance": 100000.0,
        "daily_pnl": 0.0,
        "cash": 50000.0,
        "positions": [],
    }

    @patch("agents.portfolio_manager.FinnhubClient")
    @patch("agents.portfolio_manager.get_portfolio_for_profile")
    def test_pm_logs_cycle_id_when_provided(
        self,
        mock_portfolio,
        mock_fh_class,
        mem_engine,
        caplog,
    ):
        """PM run_profile() logs the cycle_id when provided.

        Validates: Requirements 2.3, 2.4
        """
        import logging

        mock_fh_class.return_value = MagicMock()
        mock_portfolio.return_value = self._FAKE_PORTFOLIO.copy()

        with (
            patch("agents.portfolio_manager.PM_PROFILES", {"test_profile": {"direction": "long", "max_daily_loss_pct": 0.05}}),
            patch("utils.gate_config.PM_CANDIDATE_MODE", "disabled"),
        ):
            with caplog.at_level(logging.INFO, logger="agents.portfolio_manager"):
                from agents.portfolio_manager import run_profile

                # The function will fail downstream (no open positions, etc.),
                # but the cycle_id logging happens early. Wrap in try/except to capture log.
                try:
                    run_profile(
                        mem_engine, ["TSLA"], "test_profile",
                        cycle_id="coordinated_cycle_xyz",
                    )
                except Exception:
                    pass  # Expected — we only care about the log output

        # Verify cycle_id appears in log messages
        assert any(
            "coordinated_cycle_xyz" in record.message
            for record in caplog.records
        ), f"cycle_id not found in logs: {[r.message for r in caplog.records]}"

    @patch("agents.portfolio_manager.FinnhubClient")
    @patch("agents.portfolio_manager.get_portfolio_for_profile")
    def test_pm_uses_external_cycle_id_for_candidate_registry(
        self,
        mock_portfolio,
        mock_fh_class,
        mem_engine,
    ):
        """When cycle_id is provided, PM passes it to build_candidate_set instead of generating its own.

        Validates: Requirements 2.3
        """
        mock_fh_class.return_value = MagicMock()
        mock_portfolio.return_value = self._FAKE_PORTFOLIO.copy()

        external_cycle_id = "20260723_1030_scheduled_a3f2"

        with (
            patch("agents.portfolio_manager.PM_PROFILES", {"test_profile": {"direction": "long", "max_daily_loss_pct": 0.05}}),
            patch("utils.gate_config.PM_CANDIDATE_MODE", "enabled"),
            patch("utils.gate_config.PM_SHADOW_RUN_LEGACY_ENTRY", "disabled"),
            patch("utils.candidate_registry.recover_stale_reservations"),
            patch("utils.candidate_builder.build_candidate_set") as mock_build,
        ):
            # build_candidate_set returns an empty registry mock
            mock_registry = MagicMock()
            mock_registry.is_empty = True
            mock_build.return_value = mock_registry

            from agents.portfolio_manager import run_profile

            try:
                run_profile(
                    mem_engine, ["TSLA"], "test_profile",
                    cycle_id=external_cycle_id,
                )
            except Exception:
                pass

            # Verify build_candidate_set was called with the external cycle_id
            assert mock_build.called, "build_candidate_set should have been called"
            call_args = mock_build.call_args
            # cycle_id is the 6th positional arg (engine, signals, profile_id, profile, portfolio, cycle_id)
            actual_cycle_id = call_args[0][5] if len(call_args[0]) > 5 else call_args[1].get("cycle_id")
            assert actual_cycle_id == external_cycle_id, (
                f"Expected cycle_id={external_cycle_id}, got {actual_cycle_id}"
            )

    @patch("agents.portfolio_manager.FinnhubClient")
    @patch("agents.portfolio_manager.get_portfolio_for_profile")
    def test_pm_generates_own_cycle_id_when_none(
        self,
        mock_portfolio,
        mock_fh_class,
        mem_engine,
    ):
        """When cycle_id is None (legacy mode), PM generates its own uuid-based cycle_id.

        Validates: Requirements 9.5
        """
        mock_fh_class.return_value = MagicMock()
        mock_portfolio.return_value = self._FAKE_PORTFOLIO.copy()

        with (
            patch("agents.portfolio_manager.PM_PROFILES", {"test_profile": {"direction": "long", "max_daily_loss_pct": 0.05}}),
            patch("utils.gate_config.PM_CANDIDATE_MODE", "enabled"),
            patch("utils.gate_config.PM_SHADOW_RUN_LEGACY_ENTRY", "disabled"),
            patch("utils.candidate_registry.recover_stale_reservations"),
            patch("utils.candidate_builder.build_candidate_set") as mock_build,
        ):
            mock_registry = MagicMock()
            mock_registry.is_empty = True
            mock_build.return_value = mock_registry

            from agents.portfolio_manager import run_profile

            try:
                run_profile(
                    mem_engine, ["TSLA"], "test_profile",
                    cycle_id=None,  # legacy mode
                )
            except Exception:
                pass

            # Verify build_candidate_set was called with a generated cycle_id
            # (not None — PM should generate one internally)
            assert mock_build.called, "build_candidate_set should have been called"
            call_args = mock_build.call_args
            actual_cycle_id = call_args[0][5] if len(call_args[0]) > 5 else call_args[1].get("cycle_id")
            assert actual_cycle_id is not None, "PM should generate cycle_id when None provided"
            assert actual_cycle_id.startswith("cycle_test_profile_"), (
                f"Expected auto-generated cycle_id pattern, got: {actual_cycle_id}"
            )
