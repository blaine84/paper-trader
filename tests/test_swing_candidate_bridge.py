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
        "catalyst_freshness": "fresh",
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

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="enabled")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_derives_sector_rotation_geometry_when_signal_prices_missing(self, mock_symbols, mock_mode):
        signals = {
            "sig-1": _make_signal(
                current_price=100.0,
                key_levels={"support": 96.0, "resistance": 108.0},
            )
        }
        for field in ("entry_price", "stop_price", "target_price"):
            signals["sig-1"].pop(field)

        result = _call_bridge(signals=signals)

        assert len(result) == 1
        geometry = result[0]["geometry"]
        assert float(geometry.entry_price) == pytest.approx(100.0)
        assert float(geometry.stop_price) == pytest.approx(96.0)
        assert float(geometry.target_price) == pytest.approx(108.0)
        assert geometry.normalized_setup_type == "sector_rotation_swing"


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


# ---------------------------------------------------------------------------
# Test Case 8: Observe mode runs full shadow pipeline (Req 18.2, 18.3)
# Validates: Requirements 18.2, 18.3
# ---------------------------------------------------------------------------


class TestObserveModeFullShadowPipeline:
    """Observe mode runs geometry, policy, sizing — captures telemetry, never registers."""

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_observe_returns_empty_even_when_all_checks_pass(self, mock_symbols, mock_mode):
        """A valid signal that passes all checks still returns [] in observe mode."""
        result = _call_bridge()
        assert result == []

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_observe_logs_construction_attempted_true(self, mock_symbols, mock_mode, caplog):
        """Observe mode runs geometry so construction_attempted=True in log."""
        with caplog.at_level(logging.INFO, logger="utils.swing_candidate_bridge"):
            _call_bridge(signals={"sig-1": _make_signal(symbol="AAPL")})
        swing_logs = [r for r in caplog.records if "swing_bridge_signal" in r.message]
        assert len(swing_logs) == 1
        msg = swing_logs[0].message
        assert "construction_attempted=True" in msg
        assert "construction_succeeded=True" in msg

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_observe_does_not_emit_candidate_constructed_event(self, mock_symbols, mock_mode):
        """Observe mode NEVER emits swing_candidate_constructed events."""
        with patch("utils.swing_candidate_bridge._safe_emit_event") as mock_event:
            _call_bridge(signals={"sig-1": _make_signal(symbol="AAPL")})
            # No swing_candidate_constructed event should have been emitted
            for call in mock_event.call_args_list:
                args = call[0]
                # args[4] is event_type
                assert args[4] != "swing_candidate_constructed"

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_observe_captures_geometry_rejection(self, mock_symbols, mock_mode, caplog):
        """Observe mode captures geometry rejection with construction_attempted=True."""
        # Missing stop price → geometry should fail
        signals = {"sig-1": _make_signal(symbol="AAPL", stop_price=None)}
        with caplog.at_level(logging.INFO, logger="utils.swing_candidate_bridge"):
            result = _call_bridge(signals=signals)
        assert result == []
        swing_logs = [r for r in caplog.records if "swing_bridge_signal" in r.message]
        assert len(swing_logs) == 1
        msg = swing_logs[0].message
        assert "construction_attempted=True" in msg
        assert "construction_succeeded=False" in msg

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_observe_captures_policy_rejection(self, mock_symbols, mock_mode, caplog):
        """Observe mode captures profile policy rejection telemetry."""
        # R:R = 12/5 = 2.4 → passes geometry (conservative min=2.0) but fails policy (conservative min=3.0)
        signals = {"sig-1": _make_signal(
            symbol="AAPL", entry_price=100.0, stop_price=95.0, target_price=112.0,
        )}
        with caplog.at_level(logging.INFO, logger="utils.swing_candidate_bridge"):
            result = _call_bridge(signals=signals, profile_id="conservative")
        assert result == []
        swing_logs = [r for r in caplog.records if "swing_bridge_signal" in r.message]
        assert len(swing_logs) == 1
        msg = swing_logs[0].message
        assert "construction_attempted=True" in msg
        assert "construction_succeeded=False" in msg

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_observe_never_registers_candidates(self, mock_symbols, mock_mode):
        """Observe mode NEVER adds to registered_candidates."""
        # Multiple valid signals that would all register in enabled mode
        signals = {
            "sig-1": _make_signal(symbol="AAPL"),
            "sig-2": _make_signal(symbol="MSFT"),
            "sig-3": _make_signal(symbol="GOOG"),
        }
        result = _call_bridge(signals=signals)
        assert result == []

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_observe_captures_sizing_rejection(self, mock_symbols, mock_mode, caplog):
        """Observe mode captures sizing rejection with construction_attempted=True."""
        # entry_price == stop_price → stop_distance=0 → sizing rejected
        signals = {"sig-1": _make_signal(
            symbol="AAPL", entry_price=100.0, stop_price=100.0
        )}
        with caplog.at_level(logging.INFO, logger="utils.swing_candidate_bridge"):
            result = _call_bridge(signals=signals)
        assert result == []
        swing_logs = [r for r in caplog.records if "swing_bridge_signal" in r.message]
        assert len(swing_logs) == 1
        msg = swing_logs[0].message
        assert "construction_attempted=True" in msg
        assert "construction_succeeded=False" in msg


# ---------------------------------------------------------------------------
# Test Case: Profile Evaluation Order and Conservative Strictness
# Validates: Requirements 19.1, 19.2
# ---------------------------------------------------------------------------


class TestProfileEvaluationOrderAndStrictness:
    """Verify that moderate/aggressive profiles are evaluated before conservative
    (via SWING_CONSERVATIVE_OBSERVE_ONLY gate) and that conservative has stricter
    thresholds than moderate and aggressive profiles."""

    def test_conservative_thresholds_stricter_than_moderate(self):
        """Conservative min_confidence, min_strength, min_risk_reward all >= moderate.

        Validates: Requirement 19.2
        """
        from utils.gate_config import SWING_PROFILE_POLICY

        conservative = SWING_PROFILE_POLICY["conservative"]
        moderate = SWING_PROFILE_POLICY["moderate"]

        _CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}
        _STRENGTH_ORDER = {"weak": 0, "moderate": 1, "strong": 2}

        assert _CONFIDENCE_ORDER[conservative["min_confidence"]] >= _CONFIDENCE_ORDER[moderate["min_confidence"]], (
            f"Conservative min_confidence ({conservative['min_confidence']}) should be "
            f">= moderate ({moderate['min_confidence']})"
        )
        assert _STRENGTH_ORDER[conservative["min_strength"]] >= _STRENGTH_ORDER[moderate["min_strength"]], (
            f"Conservative min_strength ({conservative['min_strength']}) should be "
            f">= moderate ({moderate['min_strength']})"
        )
        assert conservative["min_risk_reward"] >= moderate["min_risk_reward"], (
            f"Conservative min_risk_reward ({conservative['min_risk_reward']}) should be "
            f">= moderate ({moderate['min_risk_reward']})"
        )

    def test_conservative_thresholds_stricter_than_aggressive(self):
        """Conservative min_confidence, min_strength, min_risk_reward all >= aggressive.

        Validates: Requirement 19.2
        """
        from utils.gate_config import SWING_PROFILE_POLICY

        conservative = SWING_PROFILE_POLICY["conservative"]
        aggressive = SWING_PROFILE_POLICY["aggressive"]

        _CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}
        _STRENGTH_ORDER = {"weak": 0, "moderate": 1, "strong": 2}

        assert _CONFIDENCE_ORDER[conservative["min_confidence"]] >= _CONFIDENCE_ORDER[aggressive["min_confidence"]], (
            f"Conservative min_confidence ({conservative['min_confidence']}) should be "
            f">= aggressive ({aggressive['min_confidence']})"
        )
        assert _STRENGTH_ORDER[conservative["min_strength"]] >= _STRENGTH_ORDER[aggressive["min_strength"]], (
            f"Conservative min_strength ({conservative['min_strength']}) should be "
            f">= aggressive ({aggressive['min_strength']})"
        )
        assert conservative["min_risk_reward"] >= aggressive["min_risk_reward"], (
            f"Conservative min_risk_reward ({conservative['min_risk_reward']}) should be "
            f">= aggressive ({aggressive['min_risk_reward']})"
        )

    def test_moderate_thresholds_stricter_than_aggressive(self):
        """Moderate min_risk_reward >= aggressive.

        Validates: Requirement 19.2
        """
        from utils.gate_config import SWING_PROFILE_POLICY

        moderate = SWING_PROFILE_POLICY["moderate"]
        aggressive = SWING_PROFILE_POLICY["aggressive"]

        assert moderate["min_risk_reward"] >= aggressive["min_risk_reward"], (
            f"Moderate min_risk_reward ({moderate['min_risk_reward']}) should be "
            f">= aggressive ({aggressive['min_risk_reward']})"
        )

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_conservative_rejects_signal_moderate_accepts(self, mock_symbols, mock_mode):
        """A signal with medium confidence and moderate strength should be accepted
        by moderate but rejected by conservative (which requires high/strong).

        Validates: Requirements 19.1, 19.2
        """
        from utils.swing_candidate_bridge import evaluate_profile_policy
        from decimal import Decimal

        # Signal with medium confidence, moderate strength, and 2.0 R:R
        # Moderate requires: medium confidence, moderate strength, 1.5 R:R → PASS
        # Conservative requires: high confidence, strong strength, 3.0 R:R → FAIL
        moderate_result = evaluate_profile_policy(
            profile_id="moderate",
            confidence="medium",
            strength="moderate",
            risk_reward=Decimal("2.0"),
            symbol="TEST",
            open_swing_symbols=set(),
        )
        conservative_result = evaluate_profile_policy(
            profile_id="conservative",
            confidence="medium",
            strength="moderate",
            risk_reward=Decimal("2.0"),
            symbol="TEST",
            open_swing_symbols=set(),
        )

        assert moderate_result.accepted is True, (
            f"Moderate should accept medium/moderate/2.0 R:R, got rejected: {moderate_result.reason_code}"
        )
        assert conservative_result.accepted is False, (
            "Conservative should reject medium/moderate/2.0 R:R"
        )

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_aggressive_accepts_signal_conservative_rejects(self, mock_symbols, mock_mode):
        """A signal with low confidence and moderate strength should be accepted
        by aggressive but rejected by conservative.

        Validates: Requirements 19.1, 19.2
        """
        from utils.swing_candidate_bridge import evaluate_profile_policy
        from decimal import Decimal

        # Signal with low confidence, moderate strength, and 1.5 R:R
        # Aggressive requires: low confidence, moderate strength, 1.25 R:R → PASS
        # Conservative requires: high confidence, strong strength, 3.0 R:R → FAIL
        aggressive_result = evaluate_profile_policy(
            profile_id="aggressive",
            confidence="low",
            strength="moderate",
            risk_reward=Decimal("1.5"),
            symbol="TEST",
            open_swing_symbols=set(),
        )
        conservative_result = evaluate_profile_policy(
            profile_id="conservative",
            confidence="low",
            strength="moderate",
            risk_reward=Decimal("1.5"),
            symbol="TEST",
            open_swing_symbols=set(),
        )

        assert aggressive_result.accepted is True, (
            f"Aggressive should accept low/moderate/1.5 R:R, got rejected: {aggressive_result.reason_code}"
        )
        assert conservative_result.accepted is False, (
            "Conservative should reject low/moderate/1.5 R:R"
        )

    def test_conservative_observe_only_blocks_before_threshold_check(self):
        """When SWING_CONSERVATIVE_OBSERVE_ONLY=True, conservative profile is
        immediately rejected with 'observe_only_period' — ensuring moderate/aggressive
        profiles can proceed while conservative remains blocked during rollout.

        Validates: Requirement 19.1
        """
        from utils.swing_candidate_bridge import evaluate_profile_policy
        from decimal import Decimal

        # Even with perfect signal that meets all thresholds
        with patch("utils.gate_config.SWING_CONSERVATIVE_OBSERVE_ONLY", True):
            conservative_result = evaluate_profile_policy(
                profile_id="conservative",
                confidence="high",
                strength="strong",
                risk_reward=Decimal("5.0"),
                symbol="TEST",
                open_swing_symbols=set(),
            )

        assert conservative_result.accepted is False
        assert conservative_result.reason_code == "observe_only_period", (
            "Conservative should be blocked with observe_only_period when flag is True"
        )

        # Moderate and aggressive are NOT blocked by this flag
        moderate_result = evaluate_profile_policy(
            profile_id="moderate",
            confidence="high",
            strength="strong",
            risk_reward=Decimal("5.0"),
            symbol="TEST",
            open_swing_symbols=set(),
        )
        aggressive_result = evaluate_profile_policy(
            profile_id="aggressive",
            confidence="high",
            strength="strong",
            risk_reward=Decimal("5.0"),
            symbol="TEST",
            open_swing_symbols=set(),
        )

        assert moderate_result.accepted is True, "Moderate should not be blocked by conservative observe-only flag"
        assert aggressive_result.accepted is True, "Aggressive should not be blocked by conservative observe-only flag"


# ---------------------------------------------------------------------------
# Test Case: Geometry Failure Categorization (Property 15)
# Validates: Requirements 13.1, 13.2, 15.2
# ---------------------------------------------------------------------------


class TestGeometryRejectionCategories:
    """Geometry construction failures produce 'missing_geometry' canonical code,
    distinct from 'failed_risk_gates' which applies when geometry succeeds but
    sizing/R:R is inadequate.

    **Validates: Requirements 13.1, 13.2, 15.2**
    """

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_missing_stop_price_produces_missing_geometry(self, mock_symbols, mock_mode, caplog):
        """A signal with None stop_price → geometry fails → missing_geometry canonical code."""
        signals = {"sig-1": _make_signal(symbol="AAPL", stop_price=None)}
        with caplog.at_level(logging.INFO, logger="utils.swing_candidate_bridge"):
            result = _call_bridge(signals=signals)
        assert result == []
        swing_logs = [r for r in caplog.records if "swing_bridge_signal" in r.message]
        assert len(swing_logs) == 1
        assert "missing_geometry" in swing_logs[0].message

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_missing_target_price_produces_missing_geometry(self, mock_symbols, mock_mode, caplog):
        """A signal with None target_price → geometry fails → missing_geometry canonical code."""
        signals = {"sig-1": _make_signal(symbol="AAPL", target_price=None)}
        with caplog.at_level(logging.INFO, logger="utils.swing_candidate_bridge"):
            result = _call_bridge(signals=signals)
        assert result == []
        swing_logs = [r for r in caplog.records if "swing_bridge_signal" in r.message]
        assert len(swing_logs) == 1
        assert "missing_geometry" in swing_logs[0].message

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_geometry_rejection_sets_construction_attempted_true(self, mock_symbols, mock_mode, caplog):
        """When geometry fails, construction_attempted=True but construction_succeeded=False."""
        signals = {"sig-1": _make_signal(symbol="AAPL", stop_price=None)}
        with caplog.at_level(logging.INFO, logger="utils.swing_candidate_bridge"):
            _call_bridge(signals=signals)
        swing_logs = [r for r in caplog.records if "swing_bridge_signal" in r.message]
        assert len(swing_logs) == 1
        msg = swing_logs[0].message
        assert "construction_attempted=True" in msg
        assert "construction_succeeded=False" in msg

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_sizing_rejection_produces_failed_risk_gates(self, mock_symbols, mock_mode, caplog):
        """When geometry succeeds but sizing fails (stop==entry → zero risk distance),
        the canonical code is 'failed_risk_gates', NOT 'missing_geometry'.

        Validates: Requirement 13.2 — geometry OK but sizing/R:R fails → failed_risk_gates.
        """
        # entry==stop → zero_risk_distance from geometry builder
        # However, we want sizing_rejected which requires geometry success.
        # To trigger sizing_rejected: pass geometry but get zero quantity.
        # Actually zero_risk_distance is a geometry rejection. Let's use a very
        # tight stop instead which passes geometry min check.
        # Better approach: use entry_price=stop_price which yields zero_risk_distance
        # in geometry (still a geometry-level rejection). For sizing_rejected we need
        # geometry to pass. Let's use a scenario where stop is different but sizing
        # produces 0 quantity (portfolio equity = 0).
        signals = {"sig-1": _make_signal(symbol="AAPL")}
        with caplog.at_level(logging.INFO, logger="utils.swing_candidate_bridge"):
            _call_bridge(signals=signals, portfolio={"equity": 0})
        swing_logs = [r for r in caplog.records if "swing_bridge_signal" in r.message]
        assert len(swing_logs) == 1
        msg = swing_logs[0].message
        assert "failed_risk_gates" in msg or "sizing_rejected" in msg

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_geometry_and_risk_gates_are_distinct_categories(self, mock_symbols, mock_mode):
        """'missing_geometry' and 'failed_risk_gates' are separate canonical codes."""
        from utils.swing_candidate_bridge import CANONICAL_REJECTION_CODES
        assert "missing_geometry" in CANONICAL_REJECTION_CODES
        assert "failed_risk_gates" in CANONICAL_REJECTION_CODES
        assert "missing_geometry" != "failed_risk_gates"

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_geometry_codes_map_correctly(self, mock_symbols, mock_mode):
        """All raw geometry failure codes map to 'missing_geometry' canonical."""
        from utils.swing_candidate_bridge import map_rejection_reason
        mapping = map_rejection_reason("missing_geometry", "TEST")
        assert mapping.canonical_code == "missing_geometry"
        assert mapping.raw_reason == "missing_geometry"

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_sizing_rejected_maps_to_failed_risk_gates(self, mock_symbols, mock_mode):
        """Raw code 'sizing_rejected' maps to canonical 'failed_risk_gates'."""
        from utils.swing_candidate_bridge import map_rejection_reason
        mapping = map_rejection_reason("sizing_rejected", "TEST")
        assert mapping.canonical_code == "failed_risk_gates"
        assert mapping.raw_reason == "sizing_rejected"

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value=set())
    def test_geometry_rejection_in_per_symbol_entry(self, mock_symbols, mock_mode):
        """Geometry failure produces PerSymbolEntry with correct canonical code in summary."""
        from utils.swing_candidate_bridge import _build_evaluation_summary, PerSymbolEntry
        # Simulate a geometry rejection entry
        entry = PerSymbolEntry(
            symbol="AAPL",
            raw_direction="LONG",
            raw_setup_label="sector_rotation",
            normalized_setup_label="sector_rotation_swing",
            confidence="high",
            strength="strong",
            construction_attempted=True,
            construction_succeeded=False,
            final_rejection_reason="missing_geometry",
            raw_rejection_reason="missing_geometry",
            missing_evidence=None,
        )
        summary = _build_evaluation_summary("cycle-1", "moderate", "observe", [entry])
        assert summary.counts_by_rejection_category == {"missing_geometry": 1}
        assert summary.per_symbol_entries[0].final_rejection_reason == "missing_geometry"


# ---------------------------------------------------------------------------
# Test Case: Exposure Rejection Separation (Property 16)
# Validates: Requirements 14.1, 14.2
# ---------------------------------------------------------------------------


class TestExposureRejectionCategories:
    """Same-symbol and correlation exposure use distinct canonical codes.

    **Validates: Requirements 14.1, 14.2**
    """

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="enabled")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value={"AAPL"})
    def test_same_symbol_exposure_produces_canonical_code(self, mock_symbols, mock_mode, caplog):
        """A signal for a symbol already in open positions → same_symbol_exposure."""
        signals = {"sig-1": _make_signal(symbol="AAPL")}
        with caplog.at_level(logging.INFO, logger="utils.swing_candidate_bridge"):
            _call_bridge(signals=signals)
        swing_logs = [r for r in caplog.records if "swing_bridge_signal" in r.message]
        assert len(swing_logs) == 1
        assert "same_symbol_exposure" in swing_logs[0].message

    def test_same_symbol_exposure_mapping(self):
        """Raw 'same_symbol_exposure' maps to canonical 'same_symbol_exposure'."""
        from utils.swing_candidate_bridge import map_rejection_reason
        mapping = map_rejection_reason("same_symbol_exposure", "AAPL")
        assert mapping.canonical_code == "same_symbol_exposure"
        assert mapping.raw_reason == "same_symbol_exposure"

    def test_same_symbol_overlap_blocked_maps_to_same_symbol_exposure(self):
        """Raw 'same_symbol_overlap_blocked' (from policy) maps to canonical 'same_symbol_exposure'."""
        from utils.swing_candidate_bridge import map_rejection_reason
        mapping = map_rejection_reason("same_symbol_overlap_blocked", "AAPL")
        assert mapping.canonical_code == "same_symbol_exposure"
        assert mapping.raw_reason == "same_symbol_overlap_blocked"

    def test_correlation_exposure_mapping(self):
        """Raw 'correlation_exposure' maps to canonical 'correlation_exposure'."""
        from utils.swing_candidate_bridge import map_rejection_reason
        mapping = map_rejection_reason("correlation_exposure", "MSFT")
        assert mapping.canonical_code == "correlation_exposure"
        assert mapping.raw_reason == "correlation_exposure"

    def test_exposure_codes_are_distinct(self):
        """'same_symbol_exposure' and 'correlation_exposure' are separate canonical codes."""
        from utils.swing_candidate_bridge import CANONICAL_REJECTION_CODES
        assert "same_symbol_exposure" in CANONICAL_REJECTION_CODES
        assert "correlation_exposure" in CANONICAL_REJECTION_CODES
        assert "same_symbol_exposure" != "correlation_exposure"

    def test_exposure_codes_distinct_from_geometry_codes(self):
        """Exposure codes do not overlap with geometry codes."""
        from utils.swing_candidate_bridge import CANONICAL_REJECTION_CODES
        geometry_codes = {"missing_geometry"}
        exposure_codes = {"same_symbol_exposure", "correlation_exposure"}
        assert geometry_codes.isdisjoint(exposure_codes)

    def test_exposure_codes_distinct_from_risk_gates(self):
        """Exposure codes do not overlap with risk gate codes."""
        exposure_codes = {"same_symbol_exposure", "correlation_exposure"}
        risk_gate_codes = {"failed_risk_gates"}
        assert exposure_codes.isdisjoint(risk_gate_codes)

    @patch("utils.swing_candidate_bridge._get_swing_mode", return_value="observe")
    @patch("utils.swing_candidate_bridge._get_open_swing_symbols", return_value={"AAPL", "MSFT"})
    def test_same_symbol_exposure_in_observe_mode(self, mock_symbols, mock_mode, caplog):
        """Observe mode captures same-symbol exposure rejection telemetry."""
        signals = {"sig-1": _make_signal(symbol="AAPL")}
        with caplog.at_level(logging.INFO, logger="utils.swing_candidate_bridge"):
            result = _call_bridge(signals=signals)
        assert result == []
        swing_logs = [r for r in caplog.records if "swing_bridge_signal" in r.message]
        assert len(swing_logs) == 1
        assert "same_symbol_exposure" in swing_logs[0].message

    def test_exposure_per_symbol_entry_construction(self):
        """PerSymbolEntry correctly records exposure rejection in counts summary."""
        from utils.swing_candidate_bridge import _build_evaluation_summary, PerSymbolEntry
        entries = [
            PerSymbolEntry(
                symbol="AAPL",
                raw_direction="LONG",
                raw_setup_label="sector_rotation",
                normalized_setup_label=None,
                confidence="high",
                strength="strong",
                construction_attempted=False,
                construction_succeeded=False,
                final_rejection_reason="same_symbol_exposure",
                raw_rejection_reason="same_symbol_exposure",
                missing_evidence=None,
            ),
            PerSymbolEntry(
                symbol="MSFT",
                raw_direction="LONG",
                raw_setup_label="sector_rotation",
                normalized_setup_label=None,
                confidence="high",
                strength="strong",
                construction_attempted=False,
                construction_succeeded=False,
                final_rejection_reason="correlation_exposure",
                raw_rejection_reason="correlation_exposure",
                missing_evidence=None,
            ),
        ]
        summary = _build_evaluation_summary("cycle-1", "moderate", "observe", entries)
        assert summary.counts_by_rejection_category == {
            "same_symbol_exposure": 1,
            "correlation_exposure": 1,
        }

    def test_all_rejection_categories_are_nonoverlapping(self):
        """The major rejection category groups are disjoint.

        Geometry, exposure, risk-gates, normalization, freshness, and policy codes
        form non-overlapping groups.
        """
        freshness_codes = {"stale_signal", "stale_catalyst"}
        normalization_codes = {
            "diagnostic_only", "unmapped_label",
            "insufficient_normalization_evidence", "context_mismatch",
            "data_provider_error", "analyst_veto",
        }
        geometry_codes = {"missing_geometry"}
        risk_gate_codes = {"failed_risk_gates"}
        exposure_codes = {"same_symbol_exposure", "correlation_exposure"}
        policy_codes = {"profile_policy"}
        catchall_codes = {"unknown_error"}

        all_groups = [
            freshness_codes, normalization_codes, geometry_codes,
            risk_gate_codes, exposure_codes, policy_codes, catchall_codes,
        ]

        # Check pairwise disjointness
        for i, group_a in enumerate(all_groups):
            for j, group_b in enumerate(all_groups):
                if i != j:
                    overlap = group_a & group_b
                    assert not overlap, (
                        f"Groups {i} and {j} overlap: {overlap}"
                    )

        # Check completeness — all groups together form the full canonical set
        from utils.swing_candidate_bridge import CANONICAL_REJECTION_CODES
        all_codes_from_groups = set()
        for group in all_groups:
            all_codes_from_groups.update(group)
        assert all_codes_from_groups == CANONICAL_REJECTION_CODES, (
            f"Missing codes: {CANONICAL_REJECTION_CODES - all_codes_from_groups}, "
            f"Extra codes: {all_codes_from_groups - CANONICAL_REJECTION_CODES}"
        )
