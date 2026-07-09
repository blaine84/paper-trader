"""Unit tests for candidate builder swing integration.

Tests the integration between build_candidate_set / _build_swing_candidates
and the swing candidate bridge: candidate_type assignment, non-executable
label filtering, PM notes explanation, expiration timestamp computation,
evaluation summary production, and fail-open exception handling.

Validates: Requirements 1.3, 1.4, 6.1, 6.5, 9.2, 16.1, 20.1, 20.2, 21.1
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text

from utils.candidate_builder import build_candidate_set, _build_swing_candidates
from utils.candidate_registry import CandidateRegistry
from utils.gate_config import SWING_MAX_CANDIDATE_AGE_HOURS


def _create_tables(engine):
    """Create in-memory pm_candidates and pm_candidate_events tables."""
    with engine.begin() as conn:
        conn.execute(text('''
            CREATE TABLE pm_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL UNIQUE,
                cycle_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                setup_type TEXT NOT NULL,
                geometry_name TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_price REAL NOT NULL,
                target_price REAL NOT NULL,
                risk_reward REAL NOT NULL,
                trigger TEXT,
                invalidation_basis TEXT,
                target_basis TEXT,
                source_signal_id TEXT NOT NULL,
                signal_snapshot_json TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'registered',
                integrity_hash TEXT NOT NULL,
                execution_key TEXT,
                reserved_at DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                expires_at DATETIME NOT NULL,
                context_snapshot_json TEXT,
                benchmark_mapping_json TEXT,
                rejection_reason TEXT,
                candidate_type TEXT DEFAULT 'intraday',
                holding_horizon INTEGER,
                normalized_setup_type TEXT
            )
        '''))
        conn.execute(text('''
            CREATE TABLE pm_candidate_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                cycle_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_data TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                candidate_type TEXT DEFAULT 'intraday'
            )
        '''))


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    _create_tables(eng)
    return eng


def _make_swing_candidate_result(symbol="MSFT"):
    """Create a candidate dict as returned by process_swing_signals."""
    from utils.swing_geometry_builder import SwingGeometry
    geom = SwingGeometry(
        symbol=symbol,
        direction="LONG",
        normalized_setup_type="sector_rotation_swing",
        entry_price=Decimal("400.00"),
        stop_price=Decimal("388.00"),
        target_price=Decimal("430.00"),
        risk_reward=Decimal("2.50"),
        holding_horizon=5,
        invalidation_basis="Below 388 support",
        source_signal_id=f"sig-{symbol.lower()}-1",
    )
    return {
        "signal_id": f"sig-{symbol.lower()}-1",
        "symbol": symbol,
        "direction": "LONG",
        "normalized_setup_type": "sector_rotation_swing",
        "geometry": geom,
        "quantity": 10,
        "dollar_risk": Decimal("120.00"),
        "sizing_multiplier": Decimal("0.5"),
        "holding_horizon": 5,
    }


# ---------------------------------------------------------------------------
# Test Case 1: Swing candidates registered with candidate_type="swing"
# Validates: Requirement 6.1
# ---------------------------------------------------------------------------


class TestSwingCandidateTypeRegistration:
    """Swing candidates built via _build_swing_candidates have candidate_type='swing'."""

    @patch("utils.swing_candidate_bridge.process_swing_signals")
    @patch("utils.gate_config.get_swing_candidate_mode", return_value="enabled")
    def test_swing_candidate_registered_with_type_swing(self, mock_mode, mock_bridge, engine):
        """A candidate from the swing bridge is written with candidate_type='swing'."""
        mock_bridge.return_value = [_make_swing_candidate_result()]

        registry = CandidateRegistry(engine, "cycle-1", "moderate")
        signals = {"MSFT": {"symbol": "MSFT", "setup_type": "sector_rotation", "signal": "BUY", "strength": "strong"}}

        _build_swing_candidates(
            db=engine,
            signals=signals,
            profile_id="moderate",
            profile={"risk_per_trade_pct": "0.01"},
            portfolio={"equity": 100000},
            cycle_id="cycle-1",
            registry=registry,
        )

        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT candidate_type, symbol FROM pm_candidates")
            ).fetchall()

        assert len(rows) == 1
        assert rows[0][0] == "swing"
        assert rows[0][1] == "MSFT"

    @patch("utils.swing_candidate_bridge.process_swing_signals")
    @patch("utils.gate_config.get_swing_candidate_mode", return_value="enabled")
    def test_multiple_swing_candidates_all_type_swing(self, mock_mode, mock_bridge, engine):
        """Multiple swing candidates all have candidate_type='swing'."""
        mock_bridge.return_value = [
            _make_swing_candidate_result("MSFT"),
            _make_swing_candidate_result("AAPL"),
        ]

        registry = CandidateRegistry(engine, "cycle-1", "moderate")
        signals = {
            "MSFT": {"symbol": "MSFT", "setup_type": "sector_rotation", "signal": "BUY", "strength": "strong"},
            "AAPL": {"symbol": "AAPL", "setup_type": "sector_rotation", "signal": "BUY", "strength": "strong"},
        }

        _build_swing_candidates(
            db=engine, signals=signals, profile_id="moderate",
            profile={"risk_per_trade_pct": "0.01"}, portfolio={"equity": 100000},
            cycle_id="cycle-1", registry=registry,
        )

        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT candidate_type FROM pm_candidates")
            ).fetchall()

        assert len(rows) == 2
        assert all(r[0] == "swing" for r in rows)


# ---------------------------------------------------------------------------
# Test Case 2: Intraday candidates still registered with candidate_type="intraday"
# Validates: Requirement 6.1
# ---------------------------------------------------------------------------


class TestIntradayCandidateTypePreserved:
    """Existing intraday candidates still have candidate_type='intraday'."""

    @patch("utils.candidate_builder.build_entry_geometry_scaffold")
    @patch("utils.gate_config.get_swing_candidate_mode", return_value="disabled")
    def test_intraday_candidate_has_type_intraday(self, mock_mode, mock_scaffold, engine):
        """The standard intraday path sets candidate_type='intraday'."""
        mock_scaffold.return_value = {
            "symbol": "XLF",
            "direction": "SHORT",
            "status": "ok",
            "candidates": [
                {
                    "name": "momentum_fade",
                    "entry_price": 53.88,
                    "stop_loss": 53.99,
                    "target": 53.66,
                    "risk_reward": 2.0,
                    "trigger": "Price breaks below support",
                    "invalidation_basis": "Price recovers above stop",
                    "target_basis": "Entry - risk x target multiplier",
                }
            ],
        }

        from models.pm_profiles import PM_PROFILES

        registry = build_candidate_set(
            engine,
            {
                "XLF": {
                    "symbol": "XLF",
                    "signal": "SHORT",
                    "strength": "moderate",
                    "setup_type": "momentum_fade",
                    "current_price": 53.9,
                }
            },
            "moderate",
            PM_PROFILES["moderate"],
            {"positions": {}},
            "cycle-intraday",
        )

        assert not registry.is_empty

        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT candidate_type FROM pm_candidates")
            ).fetchall()

        assert len(rows) >= 1
        assert all(r[0] == "intraday" for r in rows)


# ---------------------------------------------------------------------------
# Test Case 3: Non-executable label excluded from PM prompt with DEBUG log
# Validates: Requirement 1.4
# ---------------------------------------------------------------------------


class TestNonExecutableLabelExcluded:
    """Setup types not in either frozenset produce DEBUG log and no candidate."""

    @patch("utils.candidate_builder.build_entry_geometry_scaffold")
    @patch("utils.gate_config.get_swing_candidate_mode", return_value="disabled")
    def test_non_executable_label_excluded_with_debug_log(self, mock_mode, mock_scaffold, engine, caplog):
        """A label not in CANDIDATE_EXECUTABLE_SETUP_TYPES or SWING_EXECUTABLE_SETUP_TYPES is excluded."""
        mock_scaffold.return_value = {
            "symbol": "AAPL",
            "direction": "LONG",
            "status": "ok",
            "candidates": [
                {
                    "name": "base_breakout",
                    "entry_price": 150.0,
                    "stop_loss": 148.0,
                    "target": 154.0,
                    "risk_reward": 2.0,
                    "trigger": "Price breaks above",
                    "invalidation_basis": "Falls below stop",
                    "target_basis": "Entry + RR * risk",
                }
            ],
        }

        from models.pm_profiles import PM_PROFILES

        with caplog.at_level(logging.DEBUG, logger="utils.candidate_builder"):
            registry = build_candidate_set(
                engine,
                {
                    "AAPL": {
                        "symbol": "AAPL",
                        "signal": "BUY",
                        "strength": "strong",
                        "setup_type": "imaginary_unknown_type",
                        "current_price": 150.0,
                    }
                },
                "moderate",
                PM_PROFILES["moderate"],
                {"positions": {}},
                "cycle-exclude",
            )

        assert registry.is_empty

        # Verify DEBUG log mentions symbol, raw label, and reason
        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        found = any(
            "AAPL" in msg and "imaginary_unknown_type" in msg and "non_executable_type" in msg
            for msg in debug_msgs
        )
        assert found, f"Expected DEBUG log with AAPL, imaginary_unknown_type, non_executable_type. Got: {debug_msgs}"

    @patch("utils.candidate_builder.build_entry_geometry_scaffold")
    @patch("utils.gate_config.get_swing_candidate_mode", return_value="disabled")
    def test_swing_only_type_excluded_from_intraday_with_debug(self, mock_mode, mock_scaffold, engine, caplog):
        """A setup type in SWING_EXECUTABLE_SETUP_TYPES but not intraday is excluded from intraday build."""
        mock_scaffold.return_value = {
            "symbol": "TSLA",
            "direction": "LONG",
            "status": "ok",
            "candidates": [
                {
                    "name": "breakout",
                    "entry_price": 250.0,
                    "stop_loss": 245.0,
                    "target": 270.0,
                    "risk_reward": 4.0,
                    "trigger": "Breakout above resistance",
                    "invalidation_basis": "Below 245",
                    "target_basis": "Prior swing high",
                }
            ],
        }

        from models.pm_profiles import PM_PROFILES

        with caplog.at_level(logging.DEBUG, logger="utils.candidate_builder"):
            registry = build_candidate_set(
                engine,
                {
                    "TSLA": {
                        "symbol": "TSLA",
                        "signal": "BUY",
                        "strength": "strong",
                        "setup_type": "breakout_retest",
                        "current_price": 250.0,
                    }
                },
                "moderate",
                PM_PROFILES["moderate"],
                {"positions": {}},
                "cycle-swing-only",
            )

        # breakout_retest is in SWING_EXECUTABLE_SETUP_TYPES, not in intraday set
        assert registry.is_empty

        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        found = any("TSLA" in msg and "breakout_retest" in msg for msg in debug_msgs)
        assert found, f"Expected DEBUG log for TSLA/breakout_retest. Got: {debug_msgs}"


# ---------------------------------------------------------------------------
# Test Case 4: JSON explanation in PM notes when no swing candidates built
# Validates: Requirement 6.5
# ---------------------------------------------------------------------------


class TestNoSwingExplanation:
    """When mode != disabled, swing_evaluation_summary (inside process_swing_signals)
    supersedes the old swing_no_candidates event. candidate_builder should NOT
    produce a separate swing_no_candidates row."""

    @patch("utils.swing_candidate_bridge.process_swing_signals", return_value=[])
    @patch("utils.gate_config.get_swing_candidate_mode", return_value="enabled")
    def test_no_swing_no_candidates_event_when_mode_enabled(self, mock_mode, mock_bridge, engine):
        """When process_swing_signals returns [], no swing_no_candidates event is written
        (observability is handled inside process_swing_signals via swing_evaluation_summary)."""
        registry = CandidateRegistry(engine, "cycle-1", "moderate")
        signals = {"MSFT": {"symbol": "MSFT", "setup_type": "sector_rotation"}}

        _build_swing_candidates(
            db=engine, signals=signals, profile_id="moderate",
            profile={"risk_per_trade_pct": "0.01"}, portfolio={"equity": 100000},
            cycle_id="cycle-1", registry=registry,
        )

        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT event_type FROM pm_candidate_events WHERE event_type = 'swing_no_candidates'")
            ).fetchall()

        # No swing_no_candidates event — summary is produced inside process_swing_signals
        assert len(rows) == 0

    @patch("utils.swing_candidate_bridge.process_swing_signals", return_value=[])
    @patch("utils.gate_config.get_swing_candidate_mode", return_value="observe")
    def test_no_swing_no_candidates_event_when_mode_observe(self, mock_mode, mock_bridge, engine):
        """Observe mode also does not produce swing_no_candidates from candidate_builder."""
        registry = CandidateRegistry(engine, "cycle-2", "moderate")

        _build_swing_candidates(
            db=engine, signals={}, profile_id="moderate",
            profile={"risk_per_trade_pct": "0.01"}, portfolio={"equity": 100000},
            cycle_id="cycle-2", registry=registry,
        )

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT event_data FROM pm_candidate_events WHERE event_type = 'swing_no_candidates' AND cycle_id = 'cycle-2'")
            ).fetchone()

        assert row is None

    @patch("utils.swing_candidate_bridge.process_swing_signals", return_value=[])
    @patch("utils.gate_config.get_swing_candidate_mode", return_value="enabled")
    def test_no_swing_no_candidates_event_no_executable_mapping(self, mock_mode, mock_bridge, engine):
        """Even with non-swing-eligible signals, no swing_no_candidates event is written."""
        registry = CandidateRegistry(engine, "cycle-3", "moderate")
        signals = {"AAPL": {"symbol": "AAPL", "setup_type": "totally_random_label"}}

        _build_swing_candidates(
            db=engine, signals=signals, profile_id="moderate",
            profile={"risk_per_trade_pct": "0.01"}, portfolio={"equity": 100000},
            cycle_id="cycle-3", registry=registry,
        )

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT event_data FROM pm_candidate_events WHERE event_type = 'swing_no_candidates' AND cycle_id = 'cycle-3'")
            ).fetchone()

        assert row is None


# ---------------------------------------------------------------------------
# Test Case 5: Swing expiration timestamp computed correctly
# Validates: Requirement 9.2
# ---------------------------------------------------------------------------


class TestSwingExpirationTimestamp:
    """Swing candidates have expires_at = created_at + SWING_MAX_CANDIDATE_AGE_HOURS."""

    @patch("utils.swing_candidate_bridge.process_swing_signals")
    @patch("utils.gate_config.get_swing_candidate_mode", return_value="enabled")
    def test_swing_expires_at_uses_max_age_hours(self, mock_mode, mock_bridge, engine):
        """expires_at should be approximately created_at + SWING_MAX_CANDIDATE_AGE_HOURS."""
        mock_bridge.return_value = [_make_swing_candidate_result()]

        registry = CandidateRegistry(engine, "cycle-exp", "moderate")
        signals = {"MSFT": {"symbol": "MSFT", "setup_type": "sector_rotation"}}

        _build_swing_candidates(
            db=engine, signals=signals, profile_id="moderate",
            profile={"risk_per_trade_pct": "0.01"}, portfolio={"equity": 100000},
            cycle_id="cycle-exp", registry=registry,
        )

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT created_at, expires_at FROM pm_candidates WHERE cycle_id = 'cycle-exp'")
            ).fetchone()

        assert row is not None
        created = datetime.fromisoformat(row[0])
        expires = datetime.fromisoformat(row[1])

        expected_delta = timedelta(hours=SWING_MAX_CANDIDATE_AGE_HOURS)
        actual_delta = expires - created

        # Allow small timing variance (within 5 seconds) due to datetime.now() calls
        assert abs((actual_delta - expected_delta).total_seconds()) < 5, (
            f"Expected delta ~{expected_delta}, got {actual_delta}"
        )

    @patch("utils.swing_candidate_bridge.process_swing_signals")
    @patch("utils.gate_config.get_swing_candidate_mode", return_value="enabled")
    def test_swing_expires_at_not_intraday_window(self, mock_mode, mock_bridge, engine):
        """Swing expiration is based on SWING_MAX_CANDIDATE_AGE_HOURS, not cycle_expires_at."""
        mock_bridge.return_value = [_make_swing_candidate_result()]

        registry = CandidateRegistry(engine, "cycle-noexp", "moderate")
        signals = {"MSFT": {"symbol": "MSFT", "setup_type": "sector_rotation"}}

        _build_swing_candidates(
            db=engine, signals=signals, profile_id="moderate",
            profile={"risk_per_trade_pct": "0.01"}, portfolio={"equity": 100000},
            cycle_id="cycle-noexp", registry=registry,
        )

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT created_at, expires_at FROM pm_candidates WHERE cycle_id = 'cycle-noexp'")
            ).fetchone()

        assert row is not None
        created = datetime.fromisoformat(row[0])
        expires = datetime.fromisoformat(row[1])
        delta_hours = (expires - created).total_seconds() / 3600

        # Should be approximately SWING_MAX_CANDIDATE_AGE_HOURS (default 24)
        assert abs(delta_hours - SWING_MAX_CANDIDATE_AGE_HOURS) < 0.01


# ---------------------------------------------------------------------------
# Test Case 6: Disabled mode produces no swing candidates
# Validates: Requirement 10.7 (backward compat)
# ---------------------------------------------------------------------------


class TestDisabledModeNoop:
    """When SWING_CANDIDATE_MODE=disabled, _build_swing_candidates does nothing."""

    @patch("utils.gate_config.get_swing_candidate_mode", return_value="disabled")
    def test_disabled_mode_no_registration(self, mock_mode, engine):
        """No candidates or events are written when mode is disabled."""
        registry = CandidateRegistry(engine, "cycle-dis", "moderate")
        signals = {"MSFT": {"symbol": "MSFT", "setup_type": "sector_rotation"}}

        _build_swing_candidates(
            db=engine, signals=signals, profile_id="moderate",
            profile={"risk_per_trade_pct": "0.01"}, portfolio={"equity": 100000},
            cycle_id="cycle-dis", registry=registry,
        )

        with engine.connect() as conn:
            cand_count = conn.execute(
                text("SELECT COUNT(*) FROM pm_candidates")
            ).scalar()
            event_count = conn.execute(
                text("SELECT COUNT(*) FROM pm_candidate_events")
            ).scalar()

        assert cand_count == 0
        assert event_count == 0


# ---------------------------------------------------------------------------
# Test Case 7: Integration — process_swing_signals candidates registered in registry
# Validates: Requirements 20.1, 21.1
# ---------------------------------------------------------------------------


class TestSwingCandidateRegistration:
    """Candidates returned by process_swing_signals are properly registered in the CandidateRegistry."""

    @patch("utils.swing_candidate_bridge.process_swing_signals")
    @patch("utils.gate_config.get_swing_candidate_mode", return_value="enabled")
    def test_candidates_from_bridge_are_in_registry(self, mock_mode, mock_bridge, engine):
        """Candidates from process_swing_signals are registered with correct state."""
        mock_bridge.return_value = [
            _make_swing_candidate_result("MSFT"),
            _make_swing_candidate_result("AAPL"),
        ]

        registry = CandidateRegistry(engine, "cycle-reg", "moderate")
        signals = {
            "MSFT": {"symbol": "MSFT", "setup_type": "sector_rotation", "signal": "BUY", "strength": "strong"},
            "AAPL": {"symbol": "AAPL", "setup_type": "sector_rotation", "signal": "BUY", "strength": "strong"},
        }

        _build_swing_candidates(
            db=engine, signals=signals, profile_id="moderate",
            profile={"risk_per_trade_pct": "0.01"}, portfolio={"equity": 100000},
            cycle_id="cycle-reg", registry=registry,
        )

        # Registry should not be empty
        assert not registry.is_empty

        # Verify candidates are in the database with correct state
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT symbol, state, candidate_type FROM pm_candidates WHERE cycle_id = 'cycle-reg' ORDER BY symbol")
            ).fetchall()

        assert len(rows) == 2
        assert rows[0][0] == "AAPL"
        assert rows[0][1] == "registered"
        assert rows[0][2] == "swing"
        assert rows[1][0] == "MSFT"
        assert rows[1][1] == "registered"
        assert rows[1][2] == "swing"

    @patch("utils.swing_candidate_bridge.process_swing_signals")
    @patch("utils.gate_config.get_swing_candidate_mode", return_value="enabled")
    def test_registered_candidates_have_correct_geometry(self, mock_mode, mock_bridge, engine):
        """Registered swing candidates carry correct entry/stop/target from geometry."""
        mock_bridge.return_value = [_make_swing_candidate_result("TSLA")]

        registry = CandidateRegistry(engine, "cycle-geom", "moderate")
        signals = {"TSLA": {"symbol": "TSLA", "setup_type": "sector_rotation", "signal": "BUY", "strength": "strong"}}

        _build_swing_candidates(
            db=engine, signals=signals, profile_id="moderate",
            profile={"risk_per_trade_pct": "0.01"}, portfolio={"equity": 100000},
            cycle_id="cycle-geom", registry=registry,
        )

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT entry_price, stop_price, target_price, risk_reward FROM pm_candidates WHERE cycle_id = 'cycle-geom'")
            ).fetchone()

        assert row is not None
        assert row[0] == 400.00  # entry_price
        assert row[1] == 388.00  # stop_price
        assert row[2] == 430.00  # target_price
        assert row[3] == 2.50    # risk_reward


# ---------------------------------------------------------------------------
# Test Case 8: Fail-open — process_swing_signals exception does not block pipeline
# Validates: Requirements 20.1, 20.2, 21.1
# ---------------------------------------------------------------------------


class TestSwingFailOpen:
    """Exceptions from process_swing_signals are caught and logged, never block pipeline."""

    @patch("utils.swing_candidate_bridge.process_swing_signals", side_effect=RuntimeError("DB connection lost"))
    @patch("utils.gate_config.get_swing_candidate_mode", return_value="enabled")
    def test_exception_logged_pipeline_continues(self, mock_mode, mock_bridge, engine, caplog):
        """When process_swing_signals raises, _build_swing_candidates logs warning and returns."""
        registry = CandidateRegistry(engine, "cycle-fail", "moderate")
        signals = {"MSFT": {"symbol": "MSFT", "setup_type": "sector_rotation"}}

        with caplog.at_level(logging.WARNING, logger="utils.candidate_builder"):
            _build_swing_candidates(
                db=engine, signals=signals, profile_id="moderate",
                profile={"risk_per_trade_pct": "0.01"}, portfolio={"equity": 100000},
                cycle_id="cycle-fail", registry=registry,
            )

        # No crash, registry is empty (nothing registered), and warning logged
        assert registry.is_empty

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        found = any("fail-open" in msg.lower() or "DB connection lost" in msg for msg in warning_msgs)
        assert found, f"Expected WARNING about fail-open exception. Got: {warning_msgs}"

    @patch("utils.swing_candidate_bridge.process_swing_signals", side_effect=Exception("Unexpected error"))
    @patch("utils.gate_config.get_swing_candidate_mode", return_value="observe")
    def test_exception_in_observe_mode_does_not_crash(self, mock_mode, mock_bridge, engine, caplog):
        """Observe mode exceptions are also caught — fail-open applies to all modes."""
        registry = CandidateRegistry(engine, "cycle-obs-fail", "moderate")
        signals = {"AAPL": {"symbol": "AAPL", "setup_type": "risk_off_macro_short"}}

        with caplog.at_level(logging.WARNING, logger="utils.candidate_builder"):
            _build_swing_candidates(
                db=engine, signals=signals, profile_id="moderate",
                profile={"risk_per_trade_pct": "0.01"}, portfolio={"equity": 100000},
                cycle_id="cycle-obs-fail", registry=registry,
            )

        # No crash, registry is empty
        assert registry.is_empty

        # No candidates or events in database
        with engine.connect() as conn:
            cand_count = conn.execute(text("SELECT COUNT(*) FROM pm_candidates")).scalar()
            event_count = conn.execute(text("SELECT COUNT(*) FROM pm_candidate_events")).scalar()

        assert cand_count == 0
        assert event_count == 0


# ---------------------------------------------------------------------------
# Test Case 9: Disabled mode does not produce swing_evaluation_summary
# Validates: Requirement 16.1
# ---------------------------------------------------------------------------


class TestDisabledModeNoSummary:
    """When mode=disabled, no swing_evaluation_summary event is produced."""

    @patch("utils.gate_config.get_swing_candidate_mode", return_value="disabled")
    def test_disabled_mode_no_evaluation_summary(self, mock_mode, engine):
        """Mode=disabled returns immediately — no swing_evaluation_summary event."""
        registry = CandidateRegistry(engine, "cycle-dis-sum", "moderate")
        signals = {"MSFT": {"symbol": "MSFT", "setup_type": "sector_rotation"}}

        _build_swing_candidates(
            db=engine, signals=signals, profile_id="moderate",
            profile={"risk_per_trade_pct": "0.01"}, portfolio={"equity": 100000},
            cycle_id="cycle-dis-sum", registry=registry,
        )

        with engine.connect() as conn:
            summary_row = conn.execute(
                text("SELECT * FROM pm_candidate_events WHERE event_type = 'swing_evaluation_summary'")
            ).fetchone()

        assert summary_row is None

    @patch("utils.gate_config.get_swing_candidate_mode", return_value="disabled")
    def test_disabled_mode_no_events_at_all(self, mock_mode, engine):
        """Mode=disabled produces zero events of any type."""
        registry = CandidateRegistry(engine, "cycle-dis-ev", "moderate")
        signals = {"AAPL": {"symbol": "AAPL", "setup_type": "risk_off_macro_short"}}

        _build_swing_candidates(
            db=engine, signals=signals, profile_id="moderate",
            profile={"risk_per_trade_pct": "0.01"}, portfolio={"equity": 100000},
            cycle_id="cycle-dis-ev", registry=registry,
        )

        with engine.connect() as conn:
            total_events = conn.execute(
                text("SELECT COUNT(*) FROM pm_candidate_events")
            ).scalar()

        assert total_events == 0


# ---------------------------------------------------------------------------
# Test Case 10: Observe mode — no swing_no_candidates, evaluation handled in bridge
# Validates: Requirements 1.1, 16.1
# ---------------------------------------------------------------------------


class TestObserveModeIntegration:
    """Observe mode: process_swing_signals handles observability internally,
    candidate_builder does not write swing_no_candidates."""

    @patch("utils.swing_candidate_bridge.process_swing_signals", return_value=[])
    @patch("utils.gate_config.get_swing_candidate_mode", return_value="observe")
    def test_observe_mode_no_swing_no_candidates_event(self, mock_mode, mock_bridge, engine):
        """Observe mode: no swing_no_candidates written by candidate_builder."""
        registry = CandidateRegistry(engine, "cycle-obs", "moderate")
        signals = {"MSFT": {"symbol": "MSFT", "setup_type": "sector_rotation"}}

        _build_swing_candidates(
            db=engine, signals=signals, profile_id="moderate",
            profile={"risk_per_trade_pct": "0.01"}, portfolio={"equity": 100000},
            cycle_id="cycle-obs", registry=registry,
        )

        with engine.connect() as conn:
            no_cand_rows = conn.execute(
                text("SELECT * FROM pm_candidate_events WHERE event_type = 'swing_no_candidates'")
            ).fetchall()

        assert len(no_cand_rows) == 0

    @patch("utils.swing_candidate_bridge.process_swing_signals", return_value=[])
    @patch("utils.gate_config.get_swing_candidate_mode", return_value="observe")
    def test_observe_mode_no_candidates_registered(self, mock_mode, mock_bridge, engine):
        """Observe mode returns [] from bridge — no candidates registered."""
        registry = CandidateRegistry(engine, "cycle-obs-empty", "moderate")
        signals = {"MSFT": {"symbol": "MSFT", "setup_type": "sector_rotation"}}

        _build_swing_candidates(
            db=engine, signals=signals, profile_id="moderate",
            profile={"risk_per_trade_pct": "0.01"}, portfolio={"equity": 100000},
            cycle_id="cycle-obs-empty", registry=registry,
        )

        assert registry.is_empty

        with engine.connect() as conn:
            cand_count = conn.execute(text("SELECT COUNT(*) FROM pm_candidates")).scalar()

        assert cand_count == 0

    @patch("utils.swing_candidate_bridge.process_swing_signals", return_value=[])
    @patch("utils.gate_config.get_swing_candidate_mode", return_value="observe")
    def test_observe_mode_bridge_called_with_correct_args(self, mock_mode, mock_bridge, engine):
        """Verify process_swing_signals is called with correct arguments in observe mode."""
        registry = CandidateRegistry(engine, "cycle-obs-args", "aggressive")
        signals = {"NVDA": {"symbol": "NVDA", "setup_type": "sector_rotation"}}
        profile = {"risk_per_trade_pct": "0.02"}
        portfolio = {"equity": 200000}

        _build_swing_candidates(
            db=engine, signals=signals, profile_id="aggressive",
            profile=profile, portfolio=portfolio,
            cycle_id="cycle-obs-args", registry=registry,
        )

        mock_bridge.assert_called_once_with(
            signals=signals,
            profile_id="aggressive",
            profile=profile,
            portfolio=portfolio,
            cycle_id="cycle-obs-args",
            db=engine,
            engine=engine,
        )
