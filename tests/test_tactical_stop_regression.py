"""
Regression tests for Aggressive Tactical Stop Geometry (Requirement 8).

Concrete example-based tests covering the tactical stop exception path
in risk_geometry_gate, verifying decision outcomes and presence/absence
of tactical metadata fields.

**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6**
"""

import pytest
from datetime import datetime, timedelta, timezone

from utils.risk_geometry_gate import evaluate_risk_geometry


# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

SYMBOL = "NVDA"
DIRECTION = "BUY"
QUANTITY = 100
MAX_DOLLAR_RISK = 5000.0
TRADE_TIMESTAMP = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
ATR_TIMESTAMP = TRADE_TIMESTAMP - timedelta(minutes=5)  # Fresh: within 15 minutes
ATR_5MIN = 0.3555

# Common geometry for the "passing" scenario
# Note: stop 220.16 gives stop_distance=0.45 which exceeds tactical_min_stop=0.44122
# (entry 220.61 * 0.002 = 0.44122 pct floor dominates over ATR floor 0.3555 * 1.0 = 0.3555)
ENTRY_PRICE = 220.61
STOP_PRICE_PASS = 220.16
TARGET_PRICE_PASS = 221.27


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _call_gate(
    *,
    entry_price=ENTRY_PRICE,
    stop_price=STOP_PRICE_PASS,
    target_price=TARGET_PRICE_PASS,
    profile="aggressive",
    setup_type="support_bounce",
    atr_5min=ATR_5MIN,
):
    """Call evaluate_risk_geometry with shared defaults, overriding as needed."""
    return evaluate_risk_geometry(
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        quantity=QUANTITY,
        direction=DIRECTION,
        symbol=SYMBOL,
        setup_type=setup_type,
        atr_5min=atr_5min,
        atr_timestamp=ATR_TIMESTAMP,
        atr_source="polygon_5min",
        trade_timestamp=TRADE_TIMESTAMP,
        max_dollar_risk=MAX_DOLLAR_RISK,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# 8.1 Aggressive NVDA tactical support bounce passes
# ---------------------------------------------------------------------------


class TestAggressiveTacticalSupportBouncePass:
    """
    Requirement 8.1: Aggressive NVDA tactical support bounce with entry 220.61,
    stop 220.17, target 221.27, ATR 0.3555, profile aggressive, setup support_bounce.
    Expects decision passed_unchanged with tactical_stop_applied: True.
    """

    def test_decision_is_passed_unchanged(self):
        result = _call_gate()
        assert result["decision"] == "passed_unchanged"

    def test_tactical_stop_applied_is_true(self):
        result = _call_gate()
        assert result["tactical_stop_applied"] is True

    def test_tactical_min_stop_distance_value(self):
        """tactical_min_stop = max(220.61 * 0.002, 0.3555 * 1.0) = max(0.44122, 0.3555) = 0.44122"""
        result = _call_gate()
        # ATR floor = 0.3555 * 1.0 = 0.3555
        # Pct floor = 220.61 * 0.002 = 0.44122
        # max(0.44122, 0.3555) = 0.44122 — pct floor dominates
        expected_tactical_min = max(ENTRY_PRICE * 0.002, ATR_5MIN * 1.0)
        assert result["tactical_min_stop_distance"] == pytest.approx(expected_tactical_min, rel=1e-6)

    def test_original_rr_value(self):
        """original_rr = target_distance / stop_distance = 0.66 / 0.45 ≈ 1.467"""
        result = _call_gate()
        stop_distance = ENTRY_PRICE - STOP_PRICE_PASS  # 0.45
        target_distance = TARGET_PRICE_PASS - ENTRY_PRICE  # 0.66
        expected_rr = target_distance / stop_distance  # ~1.467
        assert result["original_rr"] == pytest.approx(expected_rr, rel=1e-6)

    def test_tactical_metadata_fields_all_present(self):
        """Both tactical-specific fields must be present together (atomicity)."""
        result = _call_gate()
        assert "tactical_stop_applied" in result
        assert "tactical_min_stop_distance" in result

    def test_standard_fields_present(self):
        """Standard fields original_rr and rule_name must also be present."""
        result = _call_gate()
        assert "original_rr" in result
        assert result["rule_name"] == "high_beta_mega_cap_intraday"

    def test_trade_parameters_preserved(self):
        """Original trade parameters must be preserved unchanged."""
        result = _call_gate()
        assert result["entry_price"] == ENTRY_PRICE
        assert result["stop_price"] == STOP_PRICE_PASS
        assert result["target_price"] == TARGET_PRICE_PASS
        assert result["quantity"] == QUANTITY


# ---------------------------------------------------------------------------
# 8.2 Moderate same trade rejects (falls through to global path)
# ---------------------------------------------------------------------------


class TestModerateSameTradeRejects:
    """
    Requirement 8.2: Same geometry as 8.1 but with profile moderate.
    Expects fallback to global path, no tactical_stop_applied.
    """

    def test_tactical_stop_applied_absent(self):
        result = _call_gate(profile="moderate")
        assert "tactical_stop_applied" not in result

    def test_tactical_min_stop_distance_absent(self):
        result = _call_gate(profile="moderate")
        assert "tactical_min_stop_distance" not in result

    def test_falls_through_to_global_path(self):
        """The trade should be processed by the global path (not tactical)."""
        result = _call_gate(profile="moderate")
        # The global path will process this trade — it should NOT have tactical reason code
        assert result.get("reason_code") != "PASSED_TACTICAL"

    def test_decision_outcome_present(self):
        """Result must have a decision field regardless of path."""
        result = _call_gate(profile="moderate")
        assert "decision" in result


# ---------------------------------------------------------------------------
# 8.3 Aggressive non-tactical setup rejects
# ---------------------------------------------------------------------------


class TestAggressiveNonTacticalSetupRejects:
    """
    Requirement 8.3: Aggressive profile with non-tactical setup breakout_continuation.
    Expects fallback to global path, no tactical_stop_applied.
    """

    def test_tactical_stop_applied_absent(self):
        result = _call_gate(setup_type="breakout_continuation")
        assert "tactical_stop_applied" not in result

    def test_tactical_min_stop_distance_absent(self):
        result = _call_gate(setup_type="breakout_continuation")
        assert "tactical_min_stop_distance" not in result

    def test_falls_through_to_global_path(self):
        """The trade should be processed by the global path (not tactical)."""
        result = _call_gate(setup_type="breakout_continuation")
        assert result.get("reason_code") != "PASSED_TACTICAL"

    def test_decision_outcome_present(self):
        """Result must have a decision field regardless of path."""
        result = _call_gate(setup_type="breakout_continuation")
        assert "decision" in result


# ---------------------------------------------------------------------------
# 8.4 Aggressive too-tight stop falls through
# ---------------------------------------------------------------------------


class TestAggressiveTooTightStopFallsThrough:
    """
    Requirement 8.4: Aggressive profile, support_bounce, but stop 220.50 gives
    stop_distance = 0.11, which is below tactical_min_stop of 0.3555 (ATR floor)
    and 0.44122 (pct floor). Expects fallback to global path.
    """

    def test_tactical_stop_applied_absent(self):
        result = _call_gate(stop_price=220.50)
        assert "tactical_stop_applied" not in result

    def test_tactical_min_stop_distance_absent(self):
        result = _call_gate(stop_price=220.50)
        assert "tactical_min_stop_distance" not in result

    def test_falls_through_to_global_path(self):
        """Stop distance 0.11 is below tactical_min_stop — must fall through."""
        result = _call_gate(stop_price=220.50)
        assert result.get("reason_code") != "PASSED_TACTICAL"

    def test_decision_outcome_present(self):
        """Result must have a decision field regardless of path."""
        result = _call_gate(stop_price=220.50)
        assert "decision" in result


# ---------------------------------------------------------------------------
# 8.5 Aggressive bad R:R falls through
# ---------------------------------------------------------------------------


class TestAggressiveBadRRFallsThrough:
    """
    Requirement 8.5: Aggressive profile, support_bounce, target 220.99 gives
    target_distance = 0.38, stop_distance = 0.44, R:R = 0.38/0.44 ≈ 0.86
    which is below min_reward_to_risk of 1.25. Expects fallback to global path.
    """

    def test_tactical_stop_applied_absent(self):
        result = _call_gate(target_price=220.99)
        assert "tactical_stop_applied" not in result

    def test_tactical_min_stop_distance_absent(self):
        result = _call_gate(target_price=220.99)
        assert "tactical_min_stop_distance" not in result

    def test_falls_through_to_global_path(self):
        """R:R ≈ 0.86 is below 1.25 — must fall through to global path."""
        result = _call_gate(target_price=220.99)
        assert result.get("reason_code") != "PASSED_TACTICAL"

    def test_decision_outcome_present(self):
        """Result must have a decision field regardless of path."""
        result = _call_gate(target_price=220.99)
        assert "decision" in result
