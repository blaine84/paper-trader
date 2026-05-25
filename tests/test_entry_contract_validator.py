"""Unit tests for validate_entry_for_exit_governance() in entry_contract_validator.

Tests cover:
- Thesis-development setups (news_breakout, news_catalyst, trend_pullback):
  - Full eligibility when all metadata present
  - Fallback behavior per setup type
  - Reject when critical fields missing
- Fast tactical setups (momentum_fade, orb, short_squeeze, gap_and_go, vwap_reclaim):
  - Full eligibility with entry_price + stop_price
  - Reject when stop_price missing
- Unknown setup types fallback to fast tactical validation
- stop_loss field accepted as alternative to stop_price
"""

import pytest

from utils.entry_contract_validator import validate_entry_for_exit_governance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    entry_price=150.0,
    stop_price=145.0,
    target_price=165.0,
    thesis="Breakout above resistance on catalyst",
    invalidation_basis="Close below VWAP at 148.50",
    vwap=149.0,
    support=147.0,
    resistance=155.0,
    **kwargs,
):
    """Build a complete thesis-development entry dict."""
    entry = {
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "thesis": thesis,
        "invalidation_basis": invalidation_basis,
        "vwap": vwap,
        "support": support,
        "resistance": resistance,
    }
    entry.update(kwargs)
    return entry


def _make_tactical_entry(
    entry_price=150.0,
    stop_price=148.0,
    **kwargs,
):
    """Build a minimal fast tactical entry dict."""
    entry = {
        "entry_price": entry_price,
        "stop_price": stop_price,
    }
    entry.update(kwargs)
    return entry


# ---------------------------------------------------------------------------
# Tests: news_breakout full eligibility
# ---------------------------------------------------------------------------


class TestNewsBreakoutFullEligibility:
    """news_breakout with all required metadata → full_eligibility."""

    def test_all_fields_present(self):
        entry = _make_entry()
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "news_breakout")
        assert is_valid is True
        assert status == "full_eligibility"

    def test_structural_levels_not_required_for_full_eligibility(self):
        """Structural levels are checked when available but not required."""
        entry = _make_entry(vwap=None, support=None, resistance=None)
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "news_breakout")
        assert is_valid is True
        assert status == "full_eligibility"


# ---------------------------------------------------------------------------
# Tests: news_breakout fallback behavior
# ---------------------------------------------------------------------------


class TestNewsBreakoutFallback:
    """news_breakout fallback: reject if both stop+invalidation missing,
    execute_without_extension if only invalidation missing."""

    def test_missing_both_stop_and_invalidation_basis_rejects(self):
        entry = _make_entry(stop_price=None, invalidation_basis=None)
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "news_breakout")
        assert is_valid is False
        assert status == "reject"
        assert "stop_price" in reason
        assert "invalidation_basis" in reason

    def test_has_stop_missing_invalidation_basis(self):
        entry = _make_entry(invalidation_basis=None)
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "news_breakout")
        assert is_valid is False
        assert status == "execute_without_extension"
        assert "invalidation_basis" in reason

    def test_has_invalidation_basis_missing_stop(self):
        entry = _make_entry(stop_price=None)
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "news_breakout")
        assert is_valid is False
        assert status == "execute_without_extension"
        assert "stop_price" in reason

    def test_missing_target_price(self):
        entry = _make_entry(target_price=None)
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "news_breakout")
        assert is_valid is False
        assert status == "execute_without_extension"
        assert "target_price" in reason

    def test_missing_thesis(self):
        entry = _make_entry(thesis=None)
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "news_breakout")
        assert is_valid is False
        assert status == "execute_without_extension"
        assert "thesis" in reason

    def test_empty_thesis_string(self):
        entry = _make_entry(thesis="   ")
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "news_breakout")
        assert is_valid is False
        assert status == "execute_without_extension"


# ---------------------------------------------------------------------------
# Tests: news_catalyst (same fallback as news_breakout)
# ---------------------------------------------------------------------------


class TestNewsCatalystFallback:
    """news_catalyst follows same fallback rules as news_breakout."""

    def test_full_eligibility(self):
        entry = _make_entry()
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "news_catalyst")
        assert is_valid is True
        assert status == "full_eligibility"

    def test_missing_both_stop_and_invalidation_rejects(self):
        entry = _make_entry(stop_price=None, invalidation_basis=None)
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "news_catalyst")
        assert is_valid is False
        assert status == "reject"

    def test_has_stop_missing_invalidation(self):
        entry = _make_entry(invalidation_basis=None)
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "news_catalyst")
        assert is_valid is False
        assert status == "execute_without_extension"


# ---------------------------------------------------------------------------
# Tests: trend_pullback fallback behavior
# ---------------------------------------------------------------------------


class TestTrendPullbackFallback:
    """trend_pullback: reject if stop missing, execute_without_extension otherwise."""

    def test_full_eligibility(self):
        entry = _make_entry()
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "trend_pullback")
        assert is_valid is True
        assert status == "full_eligibility"

    def test_missing_stop_price_rejects(self):
        entry = _make_entry(stop_price=None)
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "trend_pullback")
        assert is_valid is False
        assert status == "reject"
        assert "stop_price" in reason

    def test_has_stop_missing_thesis_fields(self):
        entry = _make_entry(thesis=None, invalidation_basis=None, target_price=None)
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "trend_pullback")
        assert is_valid is False
        assert status == "execute_without_extension"

    def test_has_stop_missing_only_invalidation_basis(self):
        entry = _make_entry(invalidation_basis=None)
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "trend_pullback")
        assert is_valid is False
        assert status == "execute_without_extension"
        assert "invalidation_basis" in reason


# ---------------------------------------------------------------------------
# Tests: Fast tactical setups
# ---------------------------------------------------------------------------


class TestFastTacticalSetups:
    """Fast tactical setups require only entry_price + stop_price."""

    @pytest.mark.parametrize("setup_type", [
        "momentum_fade", "orb", "short_squeeze", "gap_and_go", "vwap_reclaim",
    ])
    def test_full_eligibility_with_stop(self, setup_type):
        entry = _make_tactical_entry()
        is_valid, status, reason = validate_entry_for_exit_governance(entry, setup_type)
        assert is_valid is True
        assert status == "full_eligibility"

    @pytest.mark.parametrize("setup_type", [
        "momentum_fade", "orb", "short_squeeze", "gap_and_go", "vwap_reclaim",
    ])
    def test_missing_stop_rejects(self, setup_type):
        entry = _make_tactical_entry(stop_price=None)
        is_valid, status, reason = validate_entry_for_exit_governance(entry, setup_type)
        assert is_valid is False
        assert status == "reject"

    @pytest.mark.parametrize("setup_type", [
        "momentum_fade", "orb", "short_squeeze", "gap_and_go", "vwap_reclaim",
    ])
    def test_missing_entry_price_rejects(self, setup_type):
        entry = _make_tactical_entry(entry_price=None)
        is_valid, status, reason = validate_entry_for_exit_governance(entry, setup_type)
        assert is_valid is False
        assert status == "reject"


# ---------------------------------------------------------------------------
# Tests: stop_loss field as alternative to stop_price
# ---------------------------------------------------------------------------


class TestStopLossAlternative:
    """stop_loss field accepted as alternative to stop_price."""

    def test_stop_loss_accepted_for_tactical(self):
        entry = {"entry_price": 150.0, "stop_loss": 148.0}
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "momentum_fade")
        assert is_valid is True
        assert status == "full_eligibility"

    def test_stop_loss_accepted_for_thesis_development(self):
        entry = _make_entry(stop_price=None)
        entry["stop_loss"] = 145.0
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "news_breakout")
        assert is_valid is True
        assert status == "full_eligibility"

    def test_stop_price_preferred_over_stop_loss(self):
        """When both present, stop_price is used (it's checked first)."""
        entry = {"entry_price": 150.0, "stop_price": 145.0, "stop_loss": 144.0}
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "orb")
        assert is_valid is True
        assert status == "full_eligibility"


# ---------------------------------------------------------------------------
# Tests: Unknown setup types
# ---------------------------------------------------------------------------


class TestUnknownSetupType:
    """Unknown setup types fall back to fast tactical validation."""

    def test_unknown_with_stop_full_eligibility(self):
        entry = _make_tactical_entry()
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "unknown_setup")
        assert is_valid is True
        assert status == "full_eligibility"

    def test_unknown_missing_stop_rejects(self):
        entry = _make_tactical_entry(stop_price=None)
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "unknown_setup")
        assert is_valid is False
        assert status == "reject"

    def test_empty_string_setup_type(self):
        entry = _make_tactical_entry()
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "")
        assert is_valid is True
        assert status == "full_eligibility"


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: zero values, empty dicts, non-numeric values."""

    def test_zero_stop_price_treated_as_missing(self):
        entry = _make_tactical_entry(stop_price=0.0)
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "orb")
        assert is_valid is False
        assert status == "reject"

    def test_zero_entry_price_treated_as_missing(self):
        entry = _make_tactical_entry(entry_price=0.0)
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "orb")
        assert is_valid is False
        assert status == "reject"

    def test_empty_entry_dict(self):
        is_valid, status, reason = validate_entry_for_exit_governance({}, "news_breakout")
        assert is_valid is False
        assert status == "reject"

    def test_non_numeric_stop_price(self):
        entry = _make_tactical_entry(stop_price="not_a_number")
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "orb")
        assert is_valid is False
        assert status == "reject"

    def test_empty_invalidation_basis_string(self):
        entry = _make_entry(invalidation_basis="")
        is_valid, status, reason = validate_entry_for_exit_governance(entry, "news_breakout")
        assert is_valid is False
        assert status == "execute_without_extension"


# ---------------------------------------------------------------------------
# Tests: Return tuple structure
# ---------------------------------------------------------------------------


class TestReturnStructure:
    """Validate return tuple structure (is_valid, eligibility_status, reason)."""

    def test_returns_three_element_tuple(self):
        entry = _make_entry()
        result = validate_entry_for_exit_governance(entry, "news_breakout")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_is_valid_is_boolean(self):
        entry = _make_entry()
        is_valid, _, _ = validate_entry_for_exit_governance(entry, "news_breakout")
        assert isinstance(is_valid, bool)

    def test_eligibility_status_is_valid_value(self):
        valid_statuses = {"reject", "execute_without_extension", "full_eligibility"}
        entry = _make_entry()
        _, status, _ = validate_entry_for_exit_governance(entry, "news_breakout")
        assert status in valid_statuses

    def test_reason_is_non_empty_string(self):
        entry = _make_entry()
        _, _, reason = validate_entry_for_exit_governance(entry, "news_breakout")
        assert isinstance(reason, str)
        assert len(reason) > 0
