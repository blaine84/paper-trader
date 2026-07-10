"""
Unit tests for tactical stop configuration validation and edge cases.

Tests that the tactical stop exception correctly handles:
- Missing `tactical_stop_by_profile` key (disables exception for all profiles)
- Profile entry with missing required fields (disables exception for that profile)
- `enabled: False` flag (disables exception for that profile)
- Case normalization for profile names ("Aggressive", "AGGRESSIVE")
- Case normalization for setup types ("Support_Bounce", "VWAP_PULLBACK")

Requirements: 1.1, 1.2, 1.3, 1.5
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from utils.gate_config import STOP_DISTANCE_RULES
from utils.risk_geometry_gate import evaluate_risk_geometry


# ---------------------------------------------------------------------------
# Base trade geometry (same as regression tests — NVDA aggressive support bounce)
# ---------------------------------------------------------------------------

_BASE_TRADE_TIMESTAMP = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
_BASE_ATR_TIMESTAMP = _BASE_TRADE_TIMESTAMP - timedelta(minutes=2)

_BASE_KWARGS = {
    "entry_price": 220.61,
    "stop_price": 220.16,
    "target_price": 221.27,
    "quantity": 100,
    "direction": "BUY",
    "symbol": "NVDA",
    "setup_type": "support_bounce",
    "atr_5min": 0.3555,
    "atr_timestamp": _BASE_ATR_TIMESTAMP,
    "trade_timestamp": _BASE_TRADE_TIMESTAMP,
    "max_dollar_risk": 5000.0,
    "profile": "aggressive",
}


def _make_rules_without_tactical_key():
    """Return STOP_DISTANCE_RULES with tactical_stop_by_profile removed entirely."""
    rule = STOP_DISTANCE_RULES["high_beta_mega_cap_intraday"]
    rule_without_tactical = {k: v for k, v in rule.items() if k != "tactical_stop_by_profile"}
    return {
        **STOP_DISTANCE_RULES,
        "high_beta_mega_cap_intraday": rule_without_tactical,
    }


def _make_rules_with_modified_profile(profile_cfg):
    """Return STOP_DISTANCE_RULES with a custom aggressive profile config."""
    rule = dict(STOP_DISTANCE_RULES["high_beta_mega_cap_intraday"])
    rule["tactical_stop_by_profile"] = {"aggressive": profile_cfg}
    return {
        **STOP_DISTANCE_RULES,
        "high_beta_mega_cap_intraday": rule,
    }


# ---------------------------------------------------------------------------
# Test: Missing tactical_stop_by_profile key disables exception for all profiles
# ---------------------------------------------------------------------------


class TestMissingTacticalStopByProfileKey:
    """When tactical_stop_by_profile is absent from the rule, the exception is disabled."""

    def test_missing_key_disables_exception_for_aggressive(self):
        """Aggressive profile with qualifying trade falls through to global path."""
        patched_rules = _make_rules_without_tactical_key()

        with patch.dict(
            "utils.risk_geometry_gate.STOP_DISTANCE_RULES",
            patched_rules,
            clear=True,
        ):
            result = evaluate_risk_geometry(**_BASE_KWARGS)

        assert "tactical_stop_applied" not in result, (
            f"tactical_stop_applied should NOT be present when tactical_stop_by_profile "
            f"key is missing. Got decision={result.get('decision')}, "
            f"reason_code={result.get('reason_code')}"
        )


# ---------------------------------------------------------------------------
# Test: Profile entry with missing required field disables exception
# ---------------------------------------------------------------------------


class TestMissingRequiredFieldDisablesException:
    """When a required field is missing from the profile config, exception is disabled."""

    _FULL_PROFILE_CFG = {
        "enabled": True,
        "qualifying_setups": ["support_bounce", "vwap_pullback", "pullback_continuation"],
        "conditional_setups": ["news_breakout"],
        "tactical_context_indicators": ["support", "bounce", "vwap", "pullback"],
        "min_pct": 0.002,
        "atr_multiplier": 1.0,
        "min_reward_to_risk": 1.25,
    }

    _REQUIRED_FIELDS = [
        "enabled",
        "qualifying_setups",
        "conditional_setups",
        "tactical_context_indicators",
        "min_pct",
        "atr_multiplier",
        "min_reward_to_risk",
    ]

    def test_missing_qualifying_setups_disables_exception(self):
        """Missing qualifying_setups field disables tactical exception."""
        cfg = {k: v for k, v in self._FULL_PROFILE_CFG.items() if k != "qualifying_setups"}
        patched_rules = _make_rules_with_modified_profile(cfg)

        with patch.dict(
            "utils.risk_geometry_gate.STOP_DISTANCE_RULES",
            patched_rules,
            clear=True,
        ):
            result = evaluate_risk_geometry(**_BASE_KWARGS)

        assert "tactical_stop_applied" not in result, (
            f"tactical_stop_applied should NOT be present when qualifying_setups is missing. "
            f"Got decision={result.get('decision')}, reason_code={result.get('reason_code')}"
        )

    def test_missing_min_pct_disables_exception(self):
        """Missing min_pct field disables tactical exception."""
        cfg = {k: v for k, v in self._FULL_PROFILE_CFG.items() if k != "min_pct"}
        patched_rules = _make_rules_with_modified_profile(cfg)

        with patch.dict(
            "utils.risk_geometry_gate.STOP_DISTANCE_RULES",
            patched_rules,
            clear=True,
        ):
            result = evaluate_risk_geometry(**_BASE_KWARGS)

        assert "tactical_stop_applied" not in result, (
            f"tactical_stop_applied should NOT be present when min_pct is missing. "
            f"Got decision={result.get('decision')}, reason_code={result.get('reason_code')}"
        )

    def test_missing_atr_multiplier_disables_exception(self):
        """Missing atr_multiplier field disables tactical exception."""
        cfg = {k: v for k, v in self._FULL_PROFILE_CFG.items() if k != "atr_multiplier"}
        patched_rules = _make_rules_with_modified_profile(cfg)

        with patch.dict(
            "utils.risk_geometry_gate.STOP_DISTANCE_RULES",
            patched_rules,
            clear=True,
        ):
            result = evaluate_risk_geometry(**_BASE_KWARGS)

        assert "tactical_stop_applied" not in result, (
            f"tactical_stop_applied should NOT be present when atr_multiplier is missing. "
            f"Got decision={result.get('decision')}, reason_code={result.get('reason_code')}"
        )

    def test_missing_min_reward_to_risk_disables_exception(self):
        """Missing min_reward_to_risk field disables tactical exception."""
        cfg = {k: v for k, v in self._FULL_PROFILE_CFG.items() if k != "min_reward_to_risk"}
        patched_rules = _make_rules_with_modified_profile(cfg)

        with patch.dict(
            "utils.risk_geometry_gate.STOP_DISTANCE_RULES",
            patched_rules,
            clear=True,
        ):
            result = evaluate_risk_geometry(**_BASE_KWARGS)

        assert "tactical_stop_applied" not in result, (
            f"tactical_stop_applied should NOT be present when min_reward_to_risk is missing. "
            f"Got decision={result.get('decision')}, reason_code={result.get('reason_code')}"
        )

    def test_missing_conditional_setups_disables_exception(self):
        """Missing conditional_setups field disables tactical exception."""
        cfg = {k: v for k, v in self._FULL_PROFILE_CFG.items() if k != "conditional_setups"}
        patched_rules = _make_rules_with_modified_profile(cfg)

        with patch.dict(
            "utils.risk_geometry_gate.STOP_DISTANCE_RULES",
            patched_rules,
            clear=True,
        ):
            result = evaluate_risk_geometry(**_BASE_KWARGS)

        assert "tactical_stop_applied" not in result, (
            f"tactical_stop_applied should NOT be present when conditional_setups is missing. "
            f"Got decision={result.get('decision')}, reason_code={result.get('reason_code')}"
        )

    def test_missing_tactical_context_indicators_disables_exception(self):
        """Missing tactical_context_indicators field disables tactical exception."""
        cfg = {k: v for k, v in self._FULL_PROFILE_CFG.items() if k != "tactical_context_indicators"}
        patched_rules = _make_rules_with_modified_profile(cfg)

        with patch.dict(
            "utils.risk_geometry_gate.STOP_DISTANCE_RULES",
            patched_rules,
            clear=True,
        ):
            result = evaluate_risk_geometry(**_BASE_KWARGS)

        assert "tactical_stop_applied" not in result, (
            f"tactical_stop_applied should NOT be present when tactical_context_indicators "
            f"is missing. Got decision={result.get('decision')}, "
            f"reason_code={result.get('reason_code')}"
        )


# ---------------------------------------------------------------------------
# Test: enabled: False disables exception for that profile
# ---------------------------------------------------------------------------


class TestEnabledFalseDisablesException:
    """When enabled is False, the tactical exception is disabled for that profile."""

    def test_enabled_false_disables_exception(self):
        """Profile with enabled=False falls through to global path."""
        cfg = {
            "enabled": False,
            "qualifying_setups": ["support_bounce", "vwap_pullback", "pullback_continuation"],
            "conditional_setups": ["news_breakout"],
            "tactical_context_indicators": ["support", "bounce", "vwap", "pullback"],
            "min_pct": 0.002,
            "atr_multiplier": 1.0,
            "min_reward_to_risk": 1.25,
        }
        patched_rules = _make_rules_with_modified_profile(cfg)

        with patch.dict(
            "utils.risk_geometry_gate.STOP_DISTANCE_RULES",
            patched_rules,
            clear=True,
        ):
            result = evaluate_risk_geometry(**_BASE_KWARGS)

        assert "tactical_stop_applied" not in result, (
            f"tactical_stop_applied should NOT be present when enabled=False. "
            f"Got decision={result.get('decision')}, reason_code={result.get('reason_code')}"
        )


# ---------------------------------------------------------------------------
# Test: Case normalization for profile names
# ---------------------------------------------------------------------------


class TestProfileCaseNormalization:
    """Profile names are normalized to lowercase before config lookup."""

    def test_mixed_case_aggressive_matches_config(self):
        """Profile 'Aggressive' (mixed case) matches the 'aggressive' config entry."""
        kwargs = {**_BASE_KWARGS, "profile": "Aggressive"}

        result = evaluate_risk_geometry(**kwargs)

        assert result.get("tactical_stop_applied") is True, (
            f"Expected tactical_stop_applied=True for profile='Aggressive' (mixed case). "
            f"Got decision={result.get('decision')}, reason_code={result.get('reason_code')}"
        )

    def test_upper_case_aggressive_matches_config(self):
        """Profile 'AGGRESSIVE' (all caps) matches the 'aggressive' config entry."""
        kwargs = {**_BASE_KWARGS, "profile": "AGGRESSIVE"}

        result = evaluate_risk_geometry(**kwargs)

        assert result.get("tactical_stop_applied") is True, (
            f"Expected tactical_stop_applied=True for profile='AGGRESSIVE' (all caps). "
            f"Got decision={result.get('decision')}, reason_code={result.get('reason_code')}"
        )


# ---------------------------------------------------------------------------
# Test: Case normalization for setup types
# ---------------------------------------------------------------------------


class TestSetupTypeCaseNormalization:
    """Setup types are normalized to lowercase before config lookup."""

    def test_mixed_case_support_bounce_matches_config(self):
        """Setup type 'Support_Bounce' (mixed case) matches qualifying setups."""
        kwargs = {**_BASE_KWARGS, "setup_type": "Support_Bounce"}

        result = evaluate_risk_geometry(**kwargs)

        assert result.get("tactical_stop_applied") is True, (
            f"Expected tactical_stop_applied=True for setup_type='Support_Bounce' (mixed case). "
            f"Got decision={result.get('decision')}, reason_code={result.get('reason_code')}"
        )

    def test_upper_case_vwap_pullback_matches_config(self):
        """Setup type 'VWAP_PULLBACK' (all caps) matches qualifying setups."""
        kwargs = {**_BASE_KWARGS, "setup_type": "VWAP_PULLBACK"}

        result = evaluate_risk_geometry(**kwargs)

        assert result.get("tactical_stop_applied") is True, (
            f"Expected tactical_stop_applied=True for setup_type='VWAP_PULLBACK' (all caps). "
            f"Got decision={result.get('decision')}, reason_code={result.get('reason_code')}"
        )


class TestGeometryNameEligibility:
    """Candidate geometry can qualify a tactical stop when setup_type is broad."""

    def test_geometry_name_support_bounce_matches_config(self):
        """Broad setup_type with support_bounce geometry uses the tactical stop exception."""
        kwargs = {
            **_BASE_KWARGS,
            "setup_type": "technical_breakout",
            "geometry_name": "support_bounce",
        }

        result = evaluate_risk_geometry(**kwargs)

        assert result["decision"] == "passed_unchanged"
        assert result.get("tactical_stop_applied") is True
        assert result["tactical_stop_match_field"] == "geometry_name"
        assert result["tactical_stop_match_value"] == "support_bounce"
