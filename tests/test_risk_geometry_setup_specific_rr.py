"""Backtest replay tests for setup-specific R:R threshold overrides.

Validates Requirements 7.1, 7.2, 7.3, 7.4:
- June 1 MSFT trade: qualifying + flag enabled → pass; flag disabled → reject
- June 2 AMD trade: qualifying + flag enabled → pass; flag disabled → reject

Both trades have:
  - signal_strength >= 7.5
  - confidence_level = "high"
  - setup_type in QUALIFYING_SETUP_TYPES
  - R:R ratio >= Reduced_Threshold for profile but < Default_Threshold for profile
  - All other gate checks (stop distance, dollar risk, position sizing) pass
"""

import pytest

from utils.risk_geometry_gate import evaluate_risk_geometry


# ---------------------------------------------------------------------------
# June 1 MSFT trade replay — moderate profile, default rule
# ---------------------------------------------------------------------------
# MSFT uses DEFAULT_STOP_DISTANCE_RULE (min_pct=0.012, moderate default R:R=1.50)
# Trade params: entry=420.00, stop=414.80, target=425.20
#   stop_distance = 5.20, pct = 1.24% > 1.2% min_pct ✓
#   target_distance = 5.20, R:R = 1.0
#   R:R 1.0 >= reduced moderate threshold 0.75 but < default moderate threshold 1.50
#   dollar_risk = 10 * 5.20 = $52 < max_dollar_risk $1000 ✓


class TestJune1MsftReplay:
    """Backtest replay of June 1 MSFT high-conviction news_breakout rejection."""

    TRADE_PARAMS = dict(
        entry_price=420.00,
        stop_price=414.80,
        target_price=425.20,
        quantity=10,
        direction="BUY",
        symbol="MSFT",
        setup_type="news_breakout",
        atr_5min=None,
        atr_timestamp=None,
        max_dollar_risk=1_000,
        profile="moderate",
        signal_strength=8.2,
        confidence_level="high",
    )

    def test_flag_enabled_passes(self, monkeypatch):
        """Validates Requirement 7.1: qualifying MSFT trade passes with flag enabled."""
        monkeypatch.setenv("SETUP_SPECIFIC_RR_THRESHOLDS", "true")

        result = evaluate_risk_geometry(**self.TRADE_PARAMS)

        assert result["decision"] in ("passed_unchanged", "adjusted_allowed")
        # Verify the reduced threshold was used
        assert result["min_reward_to_risk"] == pytest.approx(0.75)
        # Verify audit fields
        assert result["setup_specific_rr_applied"] is True
        assert result["setup_specific_rr_reduced_threshold"] == pytest.approx(0.75)
        assert result["setup_specific_rr_default_threshold"] == pytest.approx(1.50)

    def test_flag_disabled_rejects(self, monkeypatch):
        """Same trade warns with flag disabled while risk geometry is soft."""
        monkeypatch.delenv("SETUP_SPECIFIC_RR_THRESHOLDS", raising=False)

        result = evaluate_risk_geometry(**self.TRADE_PARAMS)

        assert result["decision"] == "warn"
        assert result["canonical_decision"] == "warn"
        assert result["reason_code"] == "RISK_REWARD_BELOW_THRESHOLD"
        assert result["risk_geometry_soft_gate"] is True

    def test_flag_disabled_explicit_false(self, monkeypatch):
        """Explicit 'false' flag also warns while risk geometry is soft."""
        monkeypatch.setenv("SETUP_SPECIFIC_RR_THRESHOLDS", "false")

        result = evaluate_risk_geometry(**self.TRADE_PARAMS)

        assert result["decision"] == "warn"
        assert result["canonical_decision"] == "warn"
        assert result["reason_code"] == "RISK_REWARD_BELOW_THRESHOLD"
        assert result["risk_geometry_soft_gate"] is True


# ---------------------------------------------------------------------------
# June 2 AMD trade replay — aggressive profile, high_beta_mega_cap_intraday rule
# ---------------------------------------------------------------------------
# AMD is in HIGH_BETA_CLUSTER → high_beta_mega_cap_intraday rule
#   (min_pct=0.015, aggressive default R:R=1.25)
# Trade params: entry=160.00, stop=157.50, target=162.00
#   stop_distance = 2.50, pct = 1.5625% > 1.5% min_pct ✓
#   target_distance = 2.00, R:R = 0.8
#   R:R 0.8 >= reduced aggressive threshold 0.5 but < default aggressive threshold 1.25
#   dollar_risk = 10 * 2.50 = $25 < max_dollar_risk $1000 ✓
# setup_type="technical_breakout" — NOT in tactical stop qualifying/conditional setups,
#   so tactical stop exception does not trigger.


class TestJune2AmdReplay:
    """Backtest replay of June 2 AMD high-conviction technical_breakout rejection."""

    TRADE_PARAMS = dict(
        entry_price=160.00,
        stop_price=157.50,
        target_price=162.00,
        quantity=10,
        direction="BUY",
        symbol="AMD",
        setup_type="technical_breakout",
        atr_5min=None,
        atr_timestamp=None,
        max_dollar_risk=1_000,
        profile="aggressive",
        signal_strength=8.5,
        confidence_level="high",
    )

    def test_flag_enabled_passes(self, monkeypatch):
        """Validates Requirement 7.2: qualifying AMD trade passes with flag enabled."""
        monkeypatch.setenv("SETUP_SPECIFIC_RR_THRESHOLDS", "true")

        result = evaluate_risk_geometry(**self.TRADE_PARAMS)

        assert result["decision"] in ("passed_unchanged", "adjusted_allowed")
        # Verify the reduced threshold was used
        assert result["min_reward_to_risk"] == pytest.approx(0.5)
        # Verify audit fields
        assert result["setup_specific_rr_applied"] is True
        assert result["setup_specific_rr_reduced_threshold"] == pytest.approx(0.5)
        assert result["setup_specific_rr_default_threshold"] == pytest.approx(1.25)

    def test_flag_disabled_rejects(self, monkeypatch):
        """Same trade warns with flag disabled while risk geometry is soft."""
        monkeypatch.delenv("SETUP_SPECIFIC_RR_THRESHOLDS", raising=False)

        result = evaluate_risk_geometry(**self.TRADE_PARAMS)

        assert result["decision"] == "warn"
        assert result["canonical_decision"] == "warn"
        assert result["reason_code"] == "RISK_REWARD_BELOW_THRESHOLD"
        assert result["risk_geometry_soft_gate"] is True

    def test_flag_disabled_explicit_false(self, monkeypatch):
        """Explicit 'false' flag also warns while risk geometry is soft."""
        monkeypatch.setenv("SETUP_SPECIFIC_RR_THRESHOLDS", "false")

        result = evaluate_risk_geometry(**self.TRADE_PARAMS)

        assert result["decision"] == "warn"
        assert result["canonical_decision"] == "warn"
        assert result["reason_code"] == "RISK_REWARD_BELOW_THRESHOLD"
        assert result["risk_geometry_soft_gate"] is True


# ---------------------------------------------------------------------------
# Profile consistency validation (Requirement 7.4)
# ---------------------------------------------------------------------------


class TestProfileConsistency:
    """Validates Requirement 7.4: trades use the same profile as original rejection."""

    def test_msft_moderate_profile_applied(self, monkeypatch):
        """MSFT replay uses moderate profile — matching original rejection."""
        monkeypatch.setenv("SETUP_SPECIFIC_RR_THRESHOLDS", "true")

        result = evaluate_risk_geometry(
            entry_price=420.00,
            stop_price=414.80,
            target_price=425.20,
            quantity=10,
            direction="BUY",
            symbol="MSFT",
            setup_type="news_breakout",
            atr_5min=None,
            atr_timestamp=None,
            max_dollar_risk=1_000,
            profile="moderate",
            signal_strength=8.2,
            confidence_level="high",
        )

        # The moderate reduced threshold (0.75) was applied, not aggressive (0.5)
        assert result["min_reward_to_risk"] == pytest.approx(0.75)
        assert result["decision"] in ("passed_unchanged", "adjusted_allowed")

    def test_amd_aggressive_profile_applied(self, monkeypatch):
        """AMD replay uses aggressive profile — matching original rejection."""
        monkeypatch.setenv("SETUP_SPECIFIC_RR_THRESHOLDS", "true")

        result = evaluate_risk_geometry(
            entry_price=160.00,
            stop_price=157.50,
            target_price=162.00,
            quantity=10,
            direction="BUY",
            symbol="AMD",
            setup_type="technical_breakout",
            atr_5min=None,
            atr_timestamp=None,
            max_dollar_risk=1_000,
            profile="aggressive",
            signal_strength=8.5,
            confidence_level="high",
        )

        # The aggressive reduced threshold (0.5) was applied, not moderate (0.75)
        assert result["min_reward_to_risk"] == pytest.approx(0.5)
        assert result["decision"] in ("passed_unchanged", "adjusted_allowed")
