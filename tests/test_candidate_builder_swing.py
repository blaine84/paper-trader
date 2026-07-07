"""Unit tests for candidate builder swing integration.

Tests the integration between build_candidate_set / _build_swing_candidates
and the swing candidate bridge: candidate_type assignment, non-executable
label filtering, PM notes explanation, and expiration timestamp computation.

Validates: Requirements 1.3, 1.4, 6.1, 6.5, 9.2
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
    """When no swing candidates are produced, a JSON explanation is written to pm_candidate_events."""

    @patch("utils.swing_candidate_bridge.process_swing_signals", return_value=[])
    @patch("utils.gate_config.get_swing_candidate_mode", return_value="enabled")
    def test_no_swing_explanation_recorded(self, mock_mode, mock_bridge, engine):
        """When process_swing_signals returns [], a 'swing_no_candidates' event is written."""
        registry = CandidateRegistry(engine, "cycle-1", "moderate")
        signals = {"MSFT": {"symbol": "MSFT", "setup_type": "sector_rotation"}}

        _build_swing_candidates(
            db=engine, signals=signals, profile_id="moderate",
            profile={"risk_per_trade_pct": "0.01"}, portfolio={"equity": 100000},
            cycle_id="cycle-1", registry=registry,
        )

        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT event_type, event_data, candidate_type FROM pm_candidate_events")
            ).fetchall()

        assert len(rows) == 1
        assert rows[0][0] == "swing_no_candidates"
        event_data = json.loads(rows[0][1])
        assert "reason" in event_data
        valid_reasons = {
            "no_fresh_signals", "no_executable_mapping", "missing_geometry",
            "failed_risk_gates", "stale_data", "same_symbol_exposure", "profile_policy",
        }
        assert event_data["reason"] in valid_reasons
        assert rows[0][2] == "swing"

    @patch("utils.swing_candidate_bridge.process_swing_signals", return_value=[])
    @patch("utils.gate_config.get_swing_candidate_mode", return_value="enabled")
    def test_no_swing_explanation_empty_signals(self, mock_mode, mock_bridge, engine):
        """Empty signals dict produces reason 'no_fresh_signals'."""
        registry = CandidateRegistry(engine, "cycle-2", "moderate")

        _build_swing_candidates(
            db=engine, signals={}, profile_id="moderate",
            profile={"risk_per_trade_pct": "0.01"}, portfolio={"equity": 100000},
            cycle_id="cycle-2", registry=registry,
        )

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT event_data FROM pm_candidate_events WHERE cycle_id = 'cycle-2'")
            ).fetchone()

        assert row is not None
        event_data = json.loads(row[0])
        assert event_data["reason"] == "no_fresh_signals"

    @patch("utils.swing_candidate_bridge.process_swing_signals", return_value=[])
    @patch("utils.gate_config.get_swing_candidate_mode", return_value="enabled")
    def test_no_swing_explanation_no_executable_mapping(self, mock_mode, mock_bridge, engine):
        """Signals with no swing-eligible setup types produce reason 'no_executable_mapping'."""
        registry = CandidateRegistry(engine, "cycle-3", "moderate")
        signals = {"AAPL": {"symbol": "AAPL", "setup_type": "totally_random_label"}}

        _build_swing_candidates(
            db=engine, signals=signals, profile_id="moderate",
            profile={"risk_per_trade_pct": "0.01"}, portfolio={"equity": 100000},
            cycle_id="cycle-3", registry=registry,
        )

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT event_data FROM pm_candidate_events WHERE cycle_id = 'cycle-3'")
            ).fetchone()

        assert row is not None
        event_data = json.loads(row[0])
        assert event_data["reason"] == "no_executable_mapping"


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
