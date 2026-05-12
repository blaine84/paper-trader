"""
Unit tests for apply_gate_decision() gate decision engine (task 4.1).

Tests profile-aware thresholds, sector sympathy size multipliers,
log-only mode behavior, and unknown profile fallback.

Requirements: 8.1, 8.2, 8.3, 8.4, 9.1, 9.2, 9.3, 11.1, 11.2, 11.3, 11.4
"""

import pytest

from utils.catalyst_specificity import apply_gate_decision


# ---------------------------------------------------------------------------
# Requirement 8.1: Conservative profile thresholds
# ---------------------------------------------------------------------------


class TestConservativeProfile:
    """Conservative: allow >= 8, warn 6-7, block < 6."""

    def test_score_8_allows(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=8, reason_type="direct_symbol", profile="conservative", mode="enforce"
        )
        assert decision == "allow"
        assert mult == 1.0

    def test_score_9_allows(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=9, reason_type="direct_symbol", profile="conservative", mode="enforce"
        )
        assert decision == "allow"
        assert mult == 1.0

    def test_score_10_allows(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=10, reason_type="direct_symbol", profile="conservative", mode="enforce"
        )
        assert decision == "allow"
        assert mult == 1.0

    def test_score_7_warns(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=7, reason_type="direct_symbol", profile="conservative", mode="enforce"
        )
        assert decision == "warn"
        assert mult == 1.0

    def test_score_6_warns(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=6, reason_type="direct_symbol", profile="conservative", mode="enforce"
        )
        assert decision == "warn"
        assert mult == 1.0

    def test_score_5_blocks_conservative(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=5, reason_type="direct_symbol", profile="conservative", mode="enforce"
        )
        assert decision == "block"
        assert mult == 0.0

    def test_score_0_blocks_conservative(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=0, reason_type="unknown", profile="conservative", mode="enforce"
        )
        assert decision == "block"
        assert mult == 0.0


# ---------------------------------------------------------------------------
# Requirement 8.2: Moderate profile thresholds
# ---------------------------------------------------------------------------


class TestModerateProfile:
    """Moderate: allow >= 7, warn 5-6, below-threshold rules < 5."""

    def test_score_7_allows(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=7, reason_type="direct_symbol", profile="moderate", mode="enforce"
        )
        assert decision == "allow"
        assert mult == 1.0

    def test_score_8_allows(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=8, reason_type="direct_symbol", profile="moderate", mode="enforce"
        )
        assert decision == "allow"
        assert mult == 1.0

    def test_score_6_warns(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=6, reason_type="named_readthrough", profile="moderate", mode="enforce"
        )
        assert decision == "warn"
        assert mult == 1.0

    def test_score_5_warns(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=5, reason_type="named_readthrough", profile="moderate", mode="enforce"
        )
        assert decision == "warn"
        assert mult == 1.0

    def test_score_4_reduces_non_sector(self):
        """Moderate reduces (not blocks) non-sector below warn threshold."""
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=4, reason_type="named_readthrough", profile="moderate", mode="enforce"
        )
        assert decision == "reduce_size"
        assert mult == 0.5

    def test_score_3_reduces_non_sector(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=3, reason_type="unknown", profile="moderate", mode="enforce"
        )
        assert decision == "reduce_size"
        assert mult == 0.5


# ---------------------------------------------------------------------------
# Requirement 8.3: Aggressive profile thresholds
# ---------------------------------------------------------------------------


class TestAggressiveProfile:
    """Aggressive: allow >= 6, warn 4-5, below-threshold rules < 4."""

    def test_score_6_allows(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=6, reason_type="direct_symbol", profile="aggressive", mode="enforce"
        )
        assert decision == "allow"
        assert mult == 1.0

    def test_score_7_allows(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=7, reason_type="direct_symbol", profile="aggressive", mode="enforce"
        )
        assert decision == "allow"
        assert mult == 1.0

    def test_score_5_warns(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=5, reason_type="named_readthrough", profile="aggressive", mode="enforce"
        )
        assert decision == "warn"
        assert mult == 1.0

    def test_score_4_warns(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=4, reason_type="named_readthrough", profile="aggressive", mode="enforce"
        )
        assert decision == "warn"
        assert mult == 1.0

    def test_score_3_reduces_non_sector(self):
        """Aggressive reduces (not blocks) non-sector below warn threshold."""
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=3, reason_type="named_readthrough", profile="aggressive", mode="enforce"
        )
        assert decision == "reduce_size"
        assert mult == 0.5

    def test_score_0_reduces_non_sector(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=0, reason_type="unknown", profile="aggressive", mode="enforce"
        )
        assert decision == "reduce_size"
        assert mult == 0.5


# ---------------------------------------------------------------------------
# Requirement 8.4: Unknown profile defaults to moderate
# ---------------------------------------------------------------------------


class TestUnknownProfile:
    """Unknown profile names default to moderate thresholds."""

    def test_unknown_profile_uses_moderate_allow(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=7, reason_type="direct_symbol", profile="unknown_profile", mode="enforce"
        )
        assert decision == "allow"
        assert mult == 1.0

    def test_unknown_profile_uses_moderate_warn(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=5, reason_type="direct_symbol", profile="nonexistent", mode="enforce"
        )
        assert decision == "warn"
        assert mult == 1.0

    def test_unknown_profile_uses_moderate_below_warn(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=4, reason_type="direct_symbol", profile="xyz", mode="enforce"
        )
        assert decision == "reduce_size"
        assert mult == 0.5


# ---------------------------------------------------------------------------
# Requirements 8.5, 8.6: Below-threshold rules for sector_sympathy
# ---------------------------------------------------------------------------


class TestBelowThresholdSectorSympathy:
    """Sector sympathy below warn threshold always blocks (all profiles)."""

    def test_conservative_sector_below_warn_blocks(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=5, reason_type="sector_sympathy", profile="conservative", mode="enforce"
        )
        assert decision == "block"
        assert mult == 0.0

    def test_moderate_sector_below_warn_blocks(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=4, reason_type="sector_sympathy", profile="moderate", mode="enforce"
        )
        assert decision == "block"
        assert mult == 0.0

    def test_aggressive_sector_below_warn_blocks(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=3, reason_type="sector_sympathy", profile="aggressive", mode="enforce"
        )
        assert decision == "block"
        assert mult == 0.0


# ---------------------------------------------------------------------------
# Requirements 9.1, 9.2, 9.3: Sector sympathy size multiplier in warn range
# ---------------------------------------------------------------------------


class TestSectorSympathyWarnRange:
    """Sector sympathy in warn range applies profile-specific multiplier."""

    def test_conservative_sector_warn_range_blocks(self):
        """Conservative: sector sympathy in warn range → size_multiplier=0.0 (effective block)."""
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=7, reason_type="sector_sympathy", profile="conservative", mode="enforce"
        )
        assert decision == "reduce_size"
        assert mult == 0.0

    def test_conservative_sector_warn_range_score_6(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=6, reason_type="sector_sympathy", profile="conservative", mode="enforce"
        )
        assert decision == "reduce_size"
        assert mult == 0.0

    def test_moderate_sector_warn_range_reduces_half(self):
        """Moderate: sector sympathy in warn range → size_multiplier=0.5."""
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=6, reason_type="sector_sympathy", profile="moderate", mode="enforce"
        )
        assert decision == "reduce_size"
        assert mult == 0.5

    def test_moderate_sector_warn_range_score_5(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=5, reason_type="sector_sympathy", profile="moderate", mode="enforce"
        )
        assert decision == "reduce_size"
        assert mult == 0.5

    def test_aggressive_sector_warn_range_reduces_half(self):
        """Aggressive: sector sympathy in warn range → size_multiplier=0.5."""
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=5, reason_type="sector_sympathy", profile="aggressive", mode="enforce"
        )
        assert decision == "reduce_size"
        assert mult == 0.5

    def test_aggressive_sector_warn_range_score_4(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=4, reason_type="sector_sympathy", profile="aggressive", mode="enforce"
        )
        assert decision == "reduce_size"
        assert mult == 0.5


# ---------------------------------------------------------------------------
# Requirements 11.1, 11.2, 11.3, 11.4: Log-only mode
# ---------------------------------------------------------------------------


class TestLogOnlyMode:
    """Log-only mode always returns allow/1.0 but preserves intended decision."""

    def test_log_only_returns_allow_when_would_block(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=0, reason_type="unknown", profile="conservative", mode="log_only"
        )
        assert decision == "allow"
        assert mult == 1.0
        assert intended == "block"
        assert intended_mult == 0.0

    def test_log_only_returns_allow_when_would_warn(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=6, reason_type="direct_symbol", profile="conservative", mode="log_only"
        )
        assert decision == "allow"
        assert mult == 1.0
        assert intended == "warn"
        assert intended_mult == 1.0

    def test_log_only_returns_allow_when_would_reduce(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=5, reason_type="sector_sympathy", profile="moderate", mode="log_only"
        )
        assert decision == "allow"
        assert mult == 1.0
        assert intended == "reduce_size"
        assert intended_mult == 0.5

    def test_log_only_returns_allow_when_would_allow(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=10, reason_type="direct_symbol", profile="moderate", mode="log_only"
        )
        assert decision == "allow"
        assert mult == 1.0
        assert intended == "allow"
        assert intended_mult == 1.0

    def test_log_only_preserves_sector_sympathy_block(self):
        """Log-only preserves intended block for sector sympathy below warn."""
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=3, reason_type="sector_sympathy", profile="moderate", mode="log_only"
        )
        assert decision == "allow"
        assert mult == 1.0
        assert intended == "block"
        assert intended_mult == 0.0

    def test_log_only_preserves_aggressive_reduce(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=2, reason_type="named_readthrough", profile="aggressive", mode="log_only"
        )
        assert decision == "allow"
        assert mult == 1.0
        assert intended == "reduce_size"
        assert intended_mult == 0.5


# ---------------------------------------------------------------------------
# Enforce mode: intended matches actual
# ---------------------------------------------------------------------------


class TestEnforceModeIntended:
    """In enforce mode, intended_decision == decision."""

    def test_enforce_allow_intended_matches(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=8, reason_type="direct_symbol", profile="moderate", mode="enforce"
        )
        assert decision == intended
        assert mult == intended_mult

    def test_enforce_warn_intended_matches(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=5, reason_type="direct_symbol", profile="moderate", mode="enforce"
        )
        assert decision == intended
        assert mult == intended_mult

    def test_enforce_block_intended_matches(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=3, reason_type="sector_sympathy", profile="moderate", mode="enforce"
        )
        assert decision == intended
        assert mult == intended_mult

    def test_enforce_reduce_intended_matches(self):
        decision, mult, intended, intended_mult = apply_gate_decision(
            score=4, reason_type="named_readthrough", profile="moderate", mode="enforce"
        )
        assert decision == intended
        assert mult == intended_mult


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_score_at_exact_allow_boundary(self):
        """Score exactly at allow threshold should allow."""
        decision, mult, _, _ = apply_gate_decision(
            score=7, reason_type="direct_symbol", profile="moderate", mode="enforce"
        )
        assert decision == "allow"

    def test_score_one_below_allow_boundary(self):
        """Score one below allow threshold should warn (non-sector)."""
        decision, mult, _, _ = apply_gate_decision(
            score=6, reason_type="direct_symbol", profile="moderate", mode="enforce"
        )
        assert decision == "warn"

    def test_score_at_exact_warn_boundary(self):
        """Score exactly at warn threshold should warn (non-sector)."""
        decision, mult, _, _ = apply_gate_decision(
            score=5, reason_type="direct_symbol", profile="moderate", mode="enforce"
        )
        assert decision == "warn"

    def test_score_one_below_warn_boundary(self):
        """Score one below warn threshold triggers below-threshold rules."""
        decision, mult, _, _ = apply_gate_decision(
            score=4, reason_type="direct_symbol", profile="moderate", mode="enforce"
        )
        assert decision == "reduce_size"
        assert mult == 0.5

    def test_sector_sympathy_at_allow_threshold_allows(self):
        """Sector sympathy at or above allow threshold still allows."""
        decision, mult, _, _ = apply_gate_decision(
            score=7, reason_type="sector_sympathy", profile="moderate", mode="enforce"
        )
        assert decision == "allow"
        assert mult == 1.0

    def test_mismatch_reason_type_treated_as_non_sector(self):
        """Mismatch reason_type follows non-sector rules below warn."""
        decision, mult, _, _ = apply_gate_decision(
            score=3, reason_type="mismatch", profile="moderate", mode="enforce"
        )
        assert decision == "reduce_size"
        assert mult == 0.5

    def test_unknown_reason_type_treated_as_non_sector(self):
        """Unknown reason_type follows non-sector rules."""
        decision, mult, _, _ = apply_gate_decision(
            score=4, reason_type="unknown", profile="moderate", mode="enforce"
        )
        assert decision == "reduce_size"
        assert mult == 0.5
