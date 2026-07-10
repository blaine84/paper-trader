"""Unit tests for the swing candidate bridge orchestration layer.

Tests disabled/observe/enabled mode behavior, fail-open logging,
stale signal rejection, and same-symbol exposure rejection.

Validates: Requirements 8.5, 9.4, 10.2, 10.3, 10.4
"""

from __future__ import annotations

import logging
from unittest.mock import patch, MagicMock

import pytest

from utils.swing_candidate_bridge import process_swing_signals


def _make_signal(**overrides):
    """Create a well-formed analyst signal dict for testing."""
    base = {
        "symbol": "AAPL",
        "setup_type": "sector_rotation",
        "direction": "LONG",
        "strength": "strong",
        "confidence": "high",
        "key_levels": {"support": 95.0, "resistance": 110.0},
        "ema_trend": "bullish",
        "market_regime": "risk_on",
        "entry_price": 100.0,
        "stop_price": 95.0,
        "target_price": 115.0,
        "signal_age_hours": 0,
        "sector": "technology",
    }
    base.update(overrides)
    return base


def _call_bridge(signals=None, profile_id="moderate", **kwargs):
    """Helper to call process_swing_signals with sensible defaults."""
    if signals is None:
        signals = {"sig-1": _make_signal()}
    defaults = {
        "profile_id": profile_id,
        "profile": {"risk_per_trade_pct": "0.01"},
        "portfolio": {"equity": 100000},
        "cycle_id": "cycle-1",
        "db": None,
        "engine": None,
    }
    defaults.update(kwargs)
    return process_swing_signals(signals=signals, **defaults)


# ---------------------------------------------------------------------------
# Test Case 1: Disabled mode — zero results, no logging
# Validates: Requirement 10.2
# ---------------------------------------------------------------------------


class TestDisabledMode:
    """SWING_CANDIDATE_MODE=disabled → returns [] and emits no swing logs."""

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="disabled")
    def test_returns_empty_list(self, mock_mode):
        result = _call_bridge()
        assert result == []

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="disabled")
    def test_no_log_entries(self, mock_mode, caplog):
        with caplog.at_level(logging.INFO, logger="utils.swing_candidate_bridge"):
            _call_bridge()
        assert not any("swing_bridge_signal" in r.message for r in caplog.records)

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="disabled")
    def test_multiple_signals_still_empty(self, mock_mode):
        signals = {
            "sig-1": _make_signal(symbol="AAPL"),
            "sig-2": _make_signal(symbol="MSFT"),
        }
        result = _call_bridge(signals=signals)
        assert result == []


# ---------------------------------------------------------------------------
# Test Case 2: Observe mode — returns [] but emits INFO log per signal
# Validates: Requirement 10.3
# ---------------------------------------------------------------------------


class TestObserveMode:
    """SWING_CANDIDATE_MODE=observe → normalizes + logs per signal, returns []."""

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_returns_empty_list(self, mock_symbols, mock_mode):
        result = _call_bridge()
        assert result == []

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_emits_log_per_signal(self, mock_symbols, mock_mode, caplog):
        signals = {
            "sig-1": _make_signal(symbol="AAPL"),
            "sig-2": _make_signal(symbol="MSFT"),
        }
        with caplog.at_level(logging.INFO, logger="utils.swing_candidate_bridge"):
            _call_bridge(signals=signals)
        swing_logs = [r for r in caplog.records if "swing_bridge_signal" in r.message]
        assert len(swing_logs) >= 2

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_log_contains_signal_fields(self, mock_symbols, mock_mode, caplog):
        with caplog.at_level(logging.INFO, logger="utils.swing_candidate_bridge"):
            _call_bridge(signals={"sig-1": _make_signal(symbol="AAPL")})
        swing_logs = [r for r in caplog.records if "swing_bridge_signal" in r.message]
        assert len(swing_logs) == 1
        msg = swing_logs[0].message
        assert "sig-1" in msg
        assert "AAPL" in msg


# ---------------------------------------------------------------------------
# Test Case 3: Enabled mode with valid signal — registers candidate
# Validates: Requirement 10.4
# ---------------------------------------------------------------------------


class TestEnabledModeValidSignal:
    """SWING_CANDIDATE_MODE=enabled → registers valid candidates."""

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="enabled")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_returns_registered_candidate(self, mock_symbols, mock_mode):
        result = _call_bridge()
        assert len(result) == 1
        candidate = result[0]
        assert candidate["symbol"] == "AAPL"
        assert candidate["signal_id"] == "sig-1"
        assert candidate["direction"] == "LONG"
        assert candidate["normalized_setup_type"] == "sector_rotation_swing"

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="enabled")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_candidate_has_expected_fields(self, mock_symbols, mock_mode):
        result = _call_bridge()
        assert len(result) == 1
        candidate = result[0]
        expected_fields = {
            "signal_id", "symbol", "direction", "normalized_setup_type",
            "geometry", "quantity", "dollar_risk", "sizing_multiplier",
            "holding_horizon",
        }
        assert expected_fields.issubset(set(candidate.keys()))

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="enabled")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_candidate_quantity_is_positive(self, mock_symbols, mock_mode):
        result = _call_bridge()
        assert len(result) == 1
        assert result[0]["quantity"] > 0

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="enabled")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_uses_analyst_signal_field_when_direction_missing(self, mock_symbols, mock_mode):
        signals = {"sig-1": _make_signal(signal="LONG")}
        signals["sig-1"].pop("direction")

        result = _call_bridge(signals=signals)

        assert len(result) == 1
        assert result[0]["direction"] == "LONG"


# ---------------------------------------------------------------------------
# Test Case 4: Enabled mode with invalid signal (normalization fails)
# Validates: Requirement 10.4
# ---------------------------------------------------------------------------


class TestEnabledModeInvalidSignal:
    """Normalization failure → returns [] with rejection event."""

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="enabled")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_unmapped_label_returns_empty(self, mock_symbols, mock_mode):
        signals = {"sig-1": _make_signal(setup_type="unknown_garbage_label")}
        result = _call_bridge(signals=signals)
        assert result == []

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="enabled")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_unmapped_label_logs_rejection(self, mock_symbols, mock_mode, caplog):
        signals = {"sig-1": _make_signal(setup_type="unknown_garbage_label")}
        with caplog.at_level(logging.INFO, logger="utils.swing_candidate_bridge"):
            _call_bridge(signals=signals)
        swing_logs = [r for r in caplog.records if "swing_bridge_signal" in r.message]
        assert len(swing_logs) == 1
        assert "unmapped_label" in swing_logs[0].message


# ---------------------------------------------------------------------------
# Test Case 5: Fail-open logging — _emit_bridge_log raises, pipeline continues
# Validates: Requirement 8.5
# ---------------------------------------------------------------------------


class TestFailOpenLogging:
    """Fail-open: if _emit_bridge_log raises, pipeline continues processing."""

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="enabled")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    @patch("utils.swing_candidate_bridge._emit_bridge_log", side_effect=RuntimeError("logging exploded"))
    def test_pipeline_continues_when_log_raises(self, mock_log, mock_symbols, mock_mode):
        result = _call_bridge()
        # Pipeline should still produce a result despite logging failure
        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    @patch("utils.swing_candidate_bridge._emit_bridge_log", side_effect=RuntimeError("logging exploded"))
    def test_observe_mode_continues_when_log_raises(self, mock_log, mock_symbols, mock_mode):
        signals = {
            "sig-1": _make_signal(symbol="AAPL"),
            "sig-2": _make_signal(symbol="MSFT"),
        }
        # Should not raise — fail-open means pipeline continues
        result = _call_bridge(signals=signals)
        assert result == []


# ---------------------------------------------------------------------------
# Test Case 6: Stale signal — signal_age_hours > 24 → skipped
# Validates: Requirement 9.4
# ---------------------------------------------------------------------------


class TestStaleSignalRejection:
    """Stale signals (signal_age_hours > SWING_MAX_CANDIDATE_AGE_HOURS) are skipped."""

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="enabled")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_stale_signal_returns_empty(self, mock_symbols, mock_mode):
        signals = {"sig-1": _make_signal(signal_age_hours=25)}
        result = _call_bridge(signals=signals)
        assert result == []

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="enabled")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_stale_signal_logs_rejection(self, mock_symbols, mock_mode, caplog):
        signals = {"sig-1": _make_signal(signal_age_hours=48)}
        with caplog.at_level(logging.INFO, logger="utils.swing_candidate_bridge"):
            _call_bridge(signals=signals)
        swing_logs = [r for r in caplog.records if "swing_bridge_signal" in r.message]
        assert len(swing_logs) == 1
        assert "stale_signal" in swing_logs[0].message

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="enabled")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_boundary_signal_not_stale(self, mock_symbols, mock_mode):
        """A signal exactly at 24 hours is not stale (> 24 check, not >=)."""
        signals = {"sig-1": _make_signal(signal_age_hours=24)}
        result = _call_bridge(signals=signals)
        # Should process normally (not rejected as stale)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Test Case 7: Same-symbol exposure — symbol already has open swing position
# Validates: Requirement 10.4 (same-symbol exposure check)
# ---------------------------------------------------------------------------


class TestSameSymbolExposure:
    """Symbol with existing open swing position → rejected."""

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="enabled")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value={"AAPL"})
    def test_same_symbol_rejected(self, mock_symbols, mock_mode):
        signals = {"sig-1": _make_signal(symbol="AAPL")}
        result = _call_bridge(signals=signals)
        assert result == []

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="enabled")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value={"AAPL"})
    def test_same_symbol_logs_rejection(self, mock_symbols, mock_mode, caplog):
        signals = {"sig-1": _make_signal(symbol="AAPL")}
        with caplog.at_level(logging.INFO, logger="utils.swing_candidate_bridge"):
            _call_bridge(signals=signals)
        swing_logs = [r for r in caplog.records if "swing_bridge_signal" in r.message]
        assert len(swing_logs) == 1
        assert "same_symbol_exposure" in swing_logs[0].message

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="enabled")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value={"AAPL"})
    def test_different_symbol_still_passes(self, mock_symbols, mock_mode):
        """A different symbol not in open_swing_symbols should still register."""
        signals = {"sig-1": _make_signal(symbol="MSFT")}
        result = _call_bridge(signals=signals)
        assert len(result) == 1
        assert result[0]["symbol"] == "MSFT"
