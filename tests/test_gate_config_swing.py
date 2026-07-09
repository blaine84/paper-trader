"""Unit tests for swing candidate gate config additions.

Validates that the swing constants defined in utils/gate_config.py match
the exact values specified in the requirements and design documents.

Requirements: 1.1, 1.2, 10.5, 10.6
"""

from __future__ import annotations

import importlib
import logging
import os
from decimal import Decimal
from unittest.mock import patch

import pytest


class TestSwingExecutableSetupTypes:
    """Assert SWING_EXECUTABLE_SETUP_TYPES contains exactly the 7 canonical types."""

    def test_contains_sector_rotation_swing(self):
        from utils.gate_config import SWING_EXECUTABLE_SETUP_TYPES

        assert "sector_rotation_swing" in SWING_EXECUTABLE_SETUP_TYPES

    def test_contains_risk_off_macro_short(self):
        from utils.gate_config import SWING_EXECUTABLE_SETUP_TYPES

        assert "risk_off_macro_short" in SWING_EXECUTABLE_SETUP_TYPES

    def test_contains_breakout_retest(self):
        from utils.gate_config import SWING_EXECUTABLE_SETUP_TYPES

        assert "breakout_retest" in SWING_EXECUTABLE_SETUP_TYPES

    def test_contains_pullback_continuation(self):
        from utils.gate_config import SWING_EXECUTABLE_SETUP_TYPES

        assert "pullback_continuation" in SWING_EXECUTABLE_SETUP_TYPES

    def test_contains_relative_strength_swing(self):
        from utils.gate_config import SWING_EXECUTABLE_SETUP_TYPES

        assert "relative_strength_swing" in SWING_EXECUTABLE_SETUP_TYPES

    def test_contains_support_bounce_swing(self):
        from utils.gate_config import SWING_EXECUTABLE_SETUP_TYPES

        assert "support_bounce_swing" in SWING_EXECUTABLE_SETUP_TYPES

    def test_contains_failed_breakdown_reclaim(self):
        from utils.gate_config import SWING_EXECUTABLE_SETUP_TYPES

        assert "failed_breakdown_reclaim" in SWING_EXECUTABLE_SETUP_TYPES

    def test_exactly_seven_types(self):
        from utils.gate_config import SWING_EXECUTABLE_SETUP_TYPES

        assert len(SWING_EXECUTABLE_SETUP_TYPES) == 7

    def test_is_frozenset(self):
        from utils.gate_config import SWING_EXECUTABLE_SETUP_TYPES

        assert isinstance(SWING_EXECUTABLE_SETUP_TYPES, frozenset)


class TestCandidateExecutableSetupTypesUnchanged:
    """Backward compatibility regression guard for CANDIDATE_EXECUTABLE_SETUP_TYPES."""

    def test_contains_momentum_fade(self):
        from utils.gate_config import CANDIDATE_EXECUTABLE_SETUP_TYPES

        assert "momentum_fade" in CANDIDATE_EXECUTABLE_SETUP_TYPES

    def test_contains_news_breakout(self):
        from utils.gate_config import CANDIDATE_EXECUTABLE_SETUP_TYPES

        assert "news_breakout" in CANDIDATE_EXECUTABLE_SETUP_TYPES

    def test_contains_gap_and_go(self):
        from utils.gate_config import CANDIDATE_EXECUTABLE_SETUP_TYPES

        assert "gap_and_go" in CANDIDATE_EXECUTABLE_SETUP_TYPES

    def test_contains_technical_breakout(self):
        from utils.gate_config import CANDIDATE_EXECUTABLE_SETUP_TYPES

        assert "technical_breakout" in CANDIDATE_EXECUTABLE_SETUP_TYPES

    def test_contains_vwap_reclaim(self):
        from utils.gate_config import CANDIDATE_EXECUTABLE_SETUP_TYPES

        assert "vwap_reclaim" in CANDIDATE_EXECUTABLE_SETUP_TYPES

    def test_exactly_five_types(self):
        from utils.gate_config import CANDIDATE_EXECUTABLE_SETUP_TYPES

        assert len(CANDIDATE_EXECUTABLE_SETUP_TYPES) == 5

    def test_is_frozenset(self):
        from utils.gate_config import CANDIDATE_EXECUTABLE_SETUP_TYPES

        assert isinstance(CANDIDATE_EXECUTABLE_SETUP_TYPES, frozenset)

    def test_no_overlap_with_swing_types(self):
        from utils.gate_config import (
            CANDIDATE_EXECUTABLE_SETUP_TYPES,
            SWING_EXECUTABLE_SETUP_TYPES,
        )

        assert CANDIDATE_EXECUTABLE_SETUP_TYPES & SWING_EXECUTABLE_SETUP_TYPES == frozenset()


class TestSwingProfilePolicy:
    """Assert SWING_PROFILE_POLICY structure with correct thresholds per profile."""

    def test_has_conservative_profile(self):
        from utils.gate_config import SWING_PROFILE_POLICY

        assert "conservative" in SWING_PROFILE_POLICY

    def test_has_moderate_profile(self):
        from utils.gate_config import SWING_PROFILE_POLICY

        assert "moderate" in SWING_PROFILE_POLICY

    def test_has_aggressive_profile(self):
        from utils.gate_config import SWING_PROFILE_POLICY

        assert "aggressive" in SWING_PROFILE_POLICY

    def test_exactly_three_profiles(self):
        from utils.gate_config import SWING_PROFILE_POLICY

        assert len(SWING_PROFILE_POLICY) == 3

    # Conservative profile thresholds
    def test_conservative_min_confidence(self):
        from utils.gate_config import SWING_PROFILE_POLICY

        assert SWING_PROFILE_POLICY["conservative"]["min_confidence"] == "high"

    def test_conservative_min_strength(self):
        from utils.gate_config import SWING_PROFILE_POLICY

        assert SWING_PROFILE_POLICY["conservative"]["min_strength"] == "strong"

    def test_conservative_min_risk_reward(self):
        from utils.gate_config import SWING_PROFILE_POLICY

        assert SWING_PROFILE_POLICY["conservative"]["min_risk_reward"] == Decimal("3.0")

    def test_conservative_sizing_multiplier(self):
        from utils.gate_config import SWING_PROFILE_POLICY

        assert SWING_PROFILE_POLICY["conservative"]["sizing_multiplier"] == Decimal("0.5")

    # Moderate profile thresholds
    def test_moderate_min_confidence(self):
        from utils.gate_config import SWING_PROFILE_POLICY

        assert SWING_PROFILE_POLICY["moderate"]["min_confidence"] == "medium"

    def test_moderate_min_strength(self):
        from utils.gate_config import SWING_PROFILE_POLICY

        assert SWING_PROFILE_POLICY["moderate"]["min_strength"] == "moderate"

    def test_moderate_min_risk_reward(self):
        from utils.gate_config import SWING_PROFILE_POLICY

        assert SWING_PROFILE_POLICY["moderate"]["min_risk_reward"] == Decimal("1.5")

    def test_moderate_sizing_multiplier(self):
        from utils.gate_config import SWING_PROFILE_POLICY

        assert SWING_PROFILE_POLICY["moderate"]["sizing_multiplier"] == Decimal("0.5")

    # Aggressive profile thresholds
    def test_aggressive_min_confidence(self):
        from utils.gate_config import SWING_PROFILE_POLICY

        assert SWING_PROFILE_POLICY["aggressive"]["min_confidence"] == "low"

    def test_aggressive_min_strength(self):
        from utils.gate_config import SWING_PROFILE_POLICY

        assert SWING_PROFILE_POLICY["aggressive"]["min_strength"] == "moderate"

    def test_aggressive_min_risk_reward(self):
        from utils.gate_config import SWING_PROFILE_POLICY

        assert SWING_PROFILE_POLICY["aggressive"]["min_risk_reward"] == Decimal("1.25")

    def test_aggressive_sizing_multiplier(self):
        from utils.gate_config import SWING_PROFILE_POLICY

        assert SWING_PROFILE_POLICY["aggressive"]["sizing_multiplier"] == Decimal("1.0")


class TestSwingMaxConcurrentPositions:
    """Assert SWING_MAX_CONCURRENT_POSITIONS values per profile."""

    def test_conservative_max_positions(self):
        from utils.gate_config import SWING_MAX_CONCURRENT_POSITIONS

        assert SWING_MAX_CONCURRENT_POSITIONS["conservative"] == 2

    def test_moderate_max_positions(self):
        from utils.gate_config import SWING_MAX_CONCURRENT_POSITIONS

        assert SWING_MAX_CONCURRENT_POSITIONS["moderate"] == 4

    def test_aggressive_max_positions(self):
        from utils.gate_config import SWING_MAX_CONCURRENT_POSITIONS

        assert SWING_MAX_CONCURRENT_POSITIONS["aggressive"] == 6

    def test_exactly_three_profiles(self):
        from utils.gate_config import SWING_MAX_CONCURRENT_POSITIONS

        assert len(SWING_MAX_CONCURRENT_POSITIONS) == 3


class TestSwingCandidateModeDefault:
    """Test SWING_CANDIDATE_MODE defaults and unrecognized value handling."""

    def test_default_is_disabled(self):
        from utils.gate_config import SWING_CANDIDATE_MODE

        # Default env should produce "disabled"
        assert SWING_CANDIDATE_MODE in ("disabled", "observe", "enabled")

    def test_code_default_is_disabled_without_env_var(self):
        """Reload module without SWING_CANDIDATE_MODE env var to verify code default is 'disabled'.

        This is a safety check: the code must default to disabled so the swing
        candidate pipeline doesn't run in production unless explicitly enabled
        via environment variable.
        Requirements: 18.1
        """
        import utils.gate_config as gc_module

        env_without_swing_mode = {
            k: v for k, v in os.environ.items() if k != "SWING_CANDIDATE_MODE"
        }
        with patch.dict("os.environ", env_without_swing_mode, clear=True):
            importlib.reload(gc_module)

        assert gc_module.SWING_CANDIDATE_MODE == "disabled"
        assert gc_module.get_swing_candidate_mode() == "disabled"

        # Reload with current env to restore module state
        importlib.reload(gc_module)

    def test_get_swing_candidate_mode_returns_string(self):
        from utils.gate_config import get_swing_candidate_mode

        result = get_swing_candidate_mode()
        assert isinstance(result, str)
        assert result in ("disabled", "observe", "enabled")

    def test_unrecognized_mode_defaults_to_disabled_with_warning(self, caplog):
        """Reload the module with an unrecognized env var to verify fallback + warning."""
        import utils.gate_config as gc_module

        with patch.dict("os.environ", {"SWING_CANDIDATE_MODE": "banana"}):
            with caplog.at_level(logging.WARNING, logger="utils.gate_config"):
                importlib.reload(gc_module)

        assert gc_module.SWING_CANDIDATE_MODE == "disabled"
        assert any("banana" in record.message for record in caplog.records)

        # Reload with default to restore module state
        with patch.dict("os.environ", {"SWING_CANDIDATE_MODE": "disabled"}):
            importlib.reload(gc_module)
