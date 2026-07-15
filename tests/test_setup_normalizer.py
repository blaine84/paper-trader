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
    """Requirement 9.2: error label → data_provider_error."""

    def test_error_label_rejected(self, default_context):
        result = normalize_setup("error", "LONG", "strong", "high", default_context)
        assert not result.success
        assert result.reason_code == "data_provider_error"
        assert result.raw_label == "error"

    def test_error_label_rejected_regardless_of_direction(self, default_context):
        result = normalize_setup("error", "SHORT", "moderate", "medium", default_context)
        assert not result.success
        assert result.reason_code == "data_provider_error"
        assert result.raw_label == "error"


class TestUnclearDirectionRejection:
    """Ambiguous direction labels are diagnostic-only and never executable."""

    def test_unclear_direction_rejected_even_when_directional(self, default_context):
        result = normalize_setup("unclear_direction", "LONG", "strong", "high", default_context)
        assert not result.success
        assert result.reason_code == "unclear_direction"


class TestDataProviderErrorDetection:
    """Requirement 9.2: 429 fallback and data_source_error → data_provider_error."""

    def test_429_in_error_code(self, default_context):
        result = normalize_setup(
            "sector_rotation", "LONG", "strong", "high", default_context,
            error_code="429",
        )
        assert not result.success
        assert result.reason_code == "data_provider_error"
        assert result.raw_label == "sector_rotation"

    def test_429_in_longer_error_code(self, default_context):
        result = normalize_setup(
            "sector_rotation", "LONG", "strong", "high", default_context,
            error_code="rate_limit_429_exceeded",
        )
        assert not result.success
        assert result.reason_code == "data_provider_error"
        assert result.raw_label == "sector_rotation"

    def test_data_source_error_true(self, default_context):
        result = normalize_setup(
            "sector_rotation", "LONG", "strong", "high", default_context,
            data_source_error=True,
        )
        assert not result.success
        assert result.reason_code == "data_provider_error"
        assert result.raw_label == "sector_rotation"


class TestRawLabelBoundary:
    """Requirement 9.1: raw label up to 64 characters, unknown labels preserve raw_label."""

    def test_64_char_label_returns_unmapped(self, default_context):
        label_64 = "a" * 64
        result = normalize_setup(label_64, "LONG", "strong", "high", default_context)
        # A 64-char unknown label should produce unmapped_label, not an error
        assert not result.success
        assert result.reason_code == "unmapped_label"
        assert result.raw_label == label_64

    def test_unknown_label_preserves_raw_label(self, default_context):
        """Requirement 9.1: unknown labels preserve raw_label."""
        result = normalize_setup("some_unknown_setup", "LONG", "strong", "high", default_context)
        assert not result.success
        assert result.reason_code == "unmapped_label"
        assert result.raw_label == "some_unknown_setup"


class TestInsufficientEvidenceMissingEvidenceList:
    """Requirement 9.3: insufficient_normalization_evidence populates missing_evidence."""

    def test_sector_rotation_missing_evidence_populated(self, default_context):
        """When sector_rotation rejected, missing_evidence lists what's missing."""
        # HOLD direction will fail the direction check
        result = normalize_setup("sector_rotation", "HOLD", "strong", "high", default_context)
        assert not result.success
        assert result.reason_code == "insufficient_normalization_evidence"
        assert result.missing_evidence is not None
        assert len(result.missing_evidence) > 0
        assert "direction_not_directional" in result.missing_evidence

    def test_sector_rotation_multiple_missing_evidence(self, default_context):
        """Multiple missing evidence fields are all reported."""
        ctx = TechnicalContext(
            key_levels={"support": None, "resistance": None},
            ema_trend="neutral",
            market_regime="risk_on",
        )
        result = normalize_setup("sector_rotation", "HOLD", "weak", "low", ctx)
        assert not result.success
        assert result.reason_code == "insufficient_normalization_evidence"
        assert result.missing_evidence is not None
        assert "direction_not_directional" in result.missing_evidence
        assert "confidence_below_medium" in result.missing_evidence
        assert "strength_below_moderate" in result.missing_evidence
        assert "no_key_levels_and_neutral_ema" in result.missing_evidence


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
        assert result.reason_code == "data_provider_error"

    def test_data_provider_error_takes_priority_over_veto(self, default_context):
        """Data provider error check runs before veto check."""
        result = normalize_setup(
            "sector_rotation", "LONG", "strong", "high", default_context,
            data_source_error=True,
            llm_veto_reason="veto",
        )
        assert not result.success
        assert result.reason_code == "data_provider_error"
