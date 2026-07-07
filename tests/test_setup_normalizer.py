"""Unit tests for setup normalizer edge cases.

Tests error label rejection, 429 fallback detection, data_source_error,
raw label length boundary, and llm_veto_reason handling.

Validates: Requirements 1.5, 1.6, 2.12, 7.5
"""

from __future__ import annotations

import pytest

from utils.setup_normalizer import NormalizationResult, TechnicalContext, normalize_setup


@pytest.fixture
def default_context():
    return TechnicalContext(
        key_levels={"support": 100.0, "resistance": 110.0},
        ema_trend="bullish",
        market_regime="risk_on",
    )


class TestErrorLabelRejection:
    """Requirement 1.5: error label → error_setup_blocked."""

    def test_error_label_rejected(self, default_context):
        result = normalize_setup("error", "LONG", "strong", "high", default_context)
        assert not result.success
        assert result.reason_code == "error_setup_blocked"

    def test_error_label_rejected_regardless_of_direction(self, default_context):
        result = normalize_setup("error", "SHORT", "moderate", "medium", default_context)
        assert not result.success
        assert result.reason_code == "error_setup_blocked"


class TestDataProviderErrorDetection:
    """Requirement 1.6: 429 fallback and data_source_error → data_provider_error_blocked."""

    def test_429_in_error_code(self, default_context):
        result = normalize_setup(
            "sector_rotation", "LONG", "strong", "high", default_context,
            error_code="429",
        )
        assert not result.success
        assert result.reason_code == "data_provider_error_blocked"

    def test_429_in_longer_error_code(self, default_context):
        result = normalize_setup(
            "sector_rotation", "LONG", "strong", "high", default_context,
            error_code="rate_limit_429_exceeded",
        )
        assert not result.success
        assert result.reason_code == "data_provider_error_blocked"

    def test_data_source_error_true(self, default_context):
        result = normalize_setup(
            "sector_rotation", "LONG", "strong", "high", default_context,
            data_source_error=True,
        )
        assert not result.success
        assert result.reason_code == "data_provider_error_blocked"


class TestRawLabelBoundary:
    """Requirement 2.12: raw label up to 64 characters."""

    def test_64_char_label_returns_unmapped(self, default_context):
        label_64 = "a" * 64
        result = normalize_setup(label_64, "LONG", "strong", "high", default_context)
        # A 64-char unknown label should produce unmapped_label, not an error
        assert not result.success
        assert result.reason_code == "unmapped_label"


class TestLlmVetoReason:
    """Requirement 7.5: llm_veto_reason handling."""

    def test_empty_string_does_not_trigger_veto(self, default_context):
        """Empty string is not a non-empty string per the spec."""
        result = normalize_setup(
            "sector_rotation", "LONG", "strong", "high", default_context,
            llm_veto_reason="",
        )
        # Should NOT be rejected with analyst_veto — empty string is not non-empty
        assert result.success
        assert result.executable_type == "sector_rotation_swing"

    def test_whitespace_only_does_not_trigger_veto(self, default_context):
        """Whitespace-only string stripped produces empty → no veto."""
        result = normalize_setup(
            "sector_rotation", "LONG", "strong", "high", default_context,
            llm_veto_reason="   ",
        )
        assert result.success
        assert result.executable_type == "sector_rotation_swing"

    def test_non_empty_veto_reason_triggers_rejection(self, default_context):
        """Non-empty llm_veto_reason → reject with analyst_veto."""
        result = normalize_setup(
            "sector_rotation", "LONG", "strong", "high", default_context,
            llm_veto_reason="Price action unclear",
        )
        assert not result.success
        assert result.reason_code == "analyst_veto"


class TestPriorityOrdering:
    """Requirement 2.12: Priority ordering — error > data_provider > veto."""

    def test_error_label_takes_priority_over_veto(self, default_context):
        """Error label check runs before veto check."""
        result = normalize_setup(
            "error", "LONG", "strong", "high", default_context,
            llm_veto_reason="veto",
        )
        assert not result.success
        assert result.reason_code == "error_setup_blocked"

    def test_data_provider_error_takes_priority_over_veto(self, default_context):
        """Data provider error check runs before veto check."""
        result = normalize_setup(
            "sector_rotation", "LONG", "strong", "high", default_context,
            data_source_error=True,
            llm_veto_reason="veto",
        )
        assert not result.success
        assert result.reason_code == "data_provider_error_blocked"
