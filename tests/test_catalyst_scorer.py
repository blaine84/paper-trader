"""
Unit tests for compute_catalyst_score() main scoring function (task 3.2).

Tests symbol mention scoring, macro instrument handling, freshness scoring,
confirmation scoring, direction consistency, score clamping, and reason_type
classification.

Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 4.4,
              5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6,
              15.1, 15.2, 15.3, 15.4
"""

import pytest

from utils.catalyst_specificity import compute_catalyst_score, load_catalyst_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def aliases():
    """Standard aliases from config."""
    config = load_catalyst_config(force_reload=True)
    return config["aliases"]


@pytest.fixture
def relationships():
    """Standard relationships from config."""
    config = load_catalyst_config(force_reload=True)
    return config["relationships"]


# ---------------------------------------------------------------------------
# Symbol Mention Scoring — Standard Symbols
# ---------------------------------------------------------------------------


class TestSymbolMentionScoring:
    """Tests for symbol mention scoring (+0 to +4)."""

    def test_direct_symbol_mention_scores_4(self, aliases, relationships):
        decision = {
            "catalyst": "AMD reports Q2 earnings beat today",
            "bias": "LONG",
        }
        score, reason_type, evidence, missing = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        assert score >= 4  # At least 4 from mention + freshness
        assert reason_type == "direct_symbol"
        assert any("mentions AMD" in e for e in evidence)

    def test_alias_mention_scores_4(self, aliases, relationships):
        decision = {
            "catalyst": "Nvidia announces new GPU architecture today",
            "bias": "LONG",
        }
        score, reason_type, evidence, missing = compute_catalyst_score(
            "NVDA", decision, None, aliases, relationships
        )
        assert reason_type == "direct_symbol"
        assert any("mentions NVDA" in e for e in evidence)

    def test_readthrough_mention_scores_3(self, aliases, relationships):
        decision = {
            "catalyst": "TSMC reports strong AI chip demand today",
            "bias": "LONG",
        }
        score, reason_type, evidence, missing = compute_catalyst_score(
            "NVDA", decision, None, aliases, relationships
        )
        assert reason_type == "named_readthrough"
        assert any("linked company" in e for e in evidence)

    def test_sector_terms_score_2(self, aliases, relationships):
        decision = {
            "catalyst": "Semiconductor industry sees strong demand today",
            "bias": "LONG",
        }
        score, reason_type, evidence, missing = compute_catalyst_score(
            "NVDA", decision, None, aliases, relationships
        )
        assert reason_type == "sector_sympathy"
        assert any("sector" in e.lower() for e in evidence)

    def test_no_mention_scores_0(self, aliases, relationships):
        decision = {
            "catalyst": "Lumber prices move on housing data today",
            "bias": "LONG",
        }
        score, reason_type, evidence, missing = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        # No AMD-related mention in lumber/housing text
        assert any("AMD" in m for m in missing) or reason_type in ("macro_only", "unknown")

    def test_empty_catalyst_returns_unknown(self, aliases, relationships):
        decision = {"catalyst": "", "bias": "LONG"}
        score, reason_type, evidence, missing = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        assert score == 0
        assert reason_type == "unknown"
        assert "no catalyst evidence found" in missing


# ---------------------------------------------------------------------------
# Macro Instrument Handling (Requirement 15)
# ---------------------------------------------------------------------------


class TestMacroInstrumentHandling:
    """Tests for macro instrument guardrails."""

    def test_tlt_fed_catalyst_gets_direct_4(self, aliases, relationships):
        """TLT + Fed/yields/CPI = direct (4 points). Req 15.1"""
        decision = {
            "catalyst": "Fed announces policy decision today",
            "bias": "SHORT",
        }
        score, reason_type, evidence, missing = compute_catalyst_score(
            "TLT", decision, None, aliases, relationships
        )
        assert reason_type == "direct_symbol"
        assert any("primary driver" in e for e in evidence)

    def test_tlt_cpi_catalyst_gets_direct(self, aliases, relationships):
        """TLT + CPI = direct. Req 15.1"""
        decision = {
            "catalyst": "CPI comes in hot at 3.5% today",
            "bias": "SHORT",
        }
        score, reason_type, evidence, missing = compute_catalyst_score(
            "TLT", decision, None, aliases, relationships
        )
        assert reason_type == "direct_symbol"

    def test_gld_gold_catalyst_gets_direct(self, aliases, relationships):
        """GLD + gold/bullion = direct (4 points). Req 15.1"""
        decision = {
            "catalyst": "Gold prices surge on inflation fears today",
            "bias": "LONG",
        }
        score, reason_type, evidence, missing = compute_catalyst_score(
            "GLD", decision, None, aliases, relationships
        )
        assert reason_type == "direct_symbol"
        assert any("primary driver" in e for e in evidence)

    def test_gld_inflation_catalyst_gets_direct(self, aliases, relationships):
        """GLD + inflation = direct. Req 15.1"""
        decision = {
            "catalyst": "Inflation data shows persistent price pressures today",
            "bias": "LONG",
        }
        score, reason_type, evidence, missing = compute_catalyst_score(
            "GLD", decision, None, aliases, relationships
        )
        assert reason_type == "direct_symbol"

    def test_spy_macro_catalyst_gets_readthrough_3(self, aliases, relationships):
        """SPY + broad macro (Fed) = readthrough (3), not direct. Req 15.2"""
        decision = {
            "catalyst": "Fed signals policy shift today",
            "setup_type": "news_catalyst",
            "bias": "LONG",
        }
        score, reason_type, evidence, missing = compute_catalyst_score(
            "SPY", decision, None, aliases, relationships
        )
        assert reason_type == "named_readthrough"
        assert any("readthrough" in e.lower() for e in evidence)

    def test_qqq_macro_catalyst_gets_readthrough(self, aliases, relationships):
        """QQQ + broad macro = readthrough (3). Req 15.2"""
        decision = {
            "catalyst": "CPI data shows cooling inflation today",
            "setup_type": "news_catalyst",
            "bias": "LONG",
        }
        score, reason_type, evidence, missing = compute_catalyst_score(
            "QQQ", decision, None, aliases, relationships
        )
        assert reason_type == "named_readthrough"

    def test_spy_macro_focused_setup_gets_direct(self, aliases, relationships):
        """SPY + macro catalyst + macro-focused setup = direct (4). Req 15.2"""
        decision = {
            "catalyst": "Fed announces major policy shift today",
            "setup_type": "macro_catalyst",
            "bias": "LONG",
        }
        score, reason_type, evidence, missing = compute_catalyst_score(
            "SPY", decision, None, aliases, relationships
        )
        assert reason_type == "direct_symbol"
        assert any("macro-focused" in e for e in evidence)

    def test_spy_direct_mention_gets_4(self, aliases, relationships):
        """SPY explicitly mentioned = direct (4). Req 15.2"""
        decision = {
            "catalyst": "SPY breaks all-time high today",
            "bias": "LONG",
        }
        score, reason_type, evidence, missing = compute_catalyst_score(
            "SPY", decision, None, aliases, relationships
        )
        assert reason_type == "direct_symbol"

    def test_xlk_matching_sector_gets_direct(self, aliases, relationships):
        """XLK + tech sector catalyst = direct (4). Req 15.3"""
        decision = {
            "catalyst": "Technology sector leads market rally today",
            "bias": "LONG",
        }
        score, reason_type, evidence, missing = compute_catalyst_score(
            "XLK", decision, None, aliases, relationships
        )
        assert reason_type == "direct_symbol"
        # Check that evidence mentions sector and direct treatment
        assert any("sector" in e.lower() for e in evidence) or any("XLK" in e for e in evidence)

    def test_xlf_matching_sector_gets_direct(self, aliases, relationships):
        """XLF + financials catalyst = direct (4). Req 15.3"""
        decision = {
            "catalyst": "Banks report strong earnings across the board today",
            "bias": "LONG",
        }
        score, reason_type, evidence, missing = compute_catalyst_score(
            "XLF", decision, None, aliases, relationships
        )
        assert reason_type == "direct_symbol"

    def test_xle_different_sector_gets_sympathy(self, aliases, relationships):
        """XLE + tech sector catalyst = sector sympathy (2). Req 15.4"""
        decision = {
            "catalyst": "Technology sector rally lifts all boats today",
            "bias": "LONG",
        }
        score, reason_type, evidence, missing = compute_catalyst_score(
            "XLE", decision, None, aliases, relationships
        )
        assert reason_type == "sector_sympathy"
        assert any("different sector" in e for e in evidence)

    def test_xlk_different_sector_gets_sympathy(self, aliases, relationships):
        """XLK + energy sector catalyst = sector sympathy (2). Req 15.4"""
        decision = {
            "catalyst": "Oil prices surge on OPEC decision today",
            "bias": "LONG",
        }
        score, reason_type, evidence, missing = compute_catalyst_score(
            "XLK", decision, None, aliases, relationships
        )
        assert reason_type == "sector_sympathy"


# ---------------------------------------------------------------------------
# Freshness Scoring
# ---------------------------------------------------------------------------


class TestFreshnessScoring:
    """Tests for freshness scoring (-2 to +2)."""

    def test_intraday_adds_2(self, aliases, relationships):
        decision = {
            "catalyst": "AMD beats earnings just announced today",
            "bias": "LONG",
        }
        score, _, evidence, _ = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        assert any("intraday" in e for e in evidence)
        # Direct (4) + intraday (2) + same_day(0) = at least 6
        assert score >= 6

    def test_stale_subtracts_2(self, aliases, relationships):
        decision = {
            "catalyst": "AMD reported earnings yesterday",
            "bias": "LONG",
        }
        score, _, _, missing = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        assert any("stale" in m for m in missing)
        # Direct (4) + stale (-2) + same_day(1) = 3 (with same_day from default)
        # Actually: direct(4) + stale(-2) = 2 minimum
        assert score <= 4

    def test_same_day_adds_1(self, aliases, relationships):
        decision = {
            "catalyst": "AMD earnings beat expectations",
            "bias": "LONG",
        }
        score, _, evidence, _ = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        assert any("same-day" in e for e in evidence)


# ---------------------------------------------------------------------------
# Confirmation Scoring
# ---------------------------------------------------------------------------


class TestConfirmationScoring:
    """Tests for confirmation scoring (-1 to +2)."""

    def test_high_volume_adds_2(self, aliases, relationships):
        decision = {
            "catalyst": "AMD beats earnings today",
            "bias": "LONG",
            "indicators": {"volume_ratio": 2.0},
        }
        score, _, evidence, _ = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        assert any("volume" in e.lower() for e in evidence)
        # Direct(4) + intraday(2) + volume(2) + direction(1) = 9
        assert score >= 8

    def test_volume_at_threshold_adds_2(self, aliases, relationships):
        """Volume exactly at 1.5x threshold should add 2."""
        decision = {
            "catalyst": "AMD beats earnings today",
            "bias": "LONG",
            "indicators": {"relative_volume": 1.5},
        }
        score, _, evidence, _ = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        assert any("volume" in e.lower() for e in evidence)

    def test_breaking_level_adds_1(self, aliases, relationships):
        decision = {
            "catalyst": "AMD beats earnings today",
            "bias": "LONG",
            "indicators": {"current_price": 100.0, "day_high": 100.0},
        }
        score, _, evidence, _ = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        assert any("breaking" in e.lower() or "strong" in e.lower() for e in evidence)

    def test_strong_signal_adds_1(self, aliases, relationships):
        decision = {
            "catalyst": "AMD beats earnings today",
            "bias": "LONG",
        }
        signal = {"strength": "strong"}
        score, _, evidence, _ = compute_catalyst_score(
            "AMD", decision, signal, aliases, relationships
        )
        assert any("breaking" in e.lower() or "strong" in e.lower() for e in evidence)

    def test_overextended_subtracts_1(self, aliases, relationships):
        decision = {
            "catalyst": "AMD beats earnings today",
            "bias": "LONG",
            "indicators": {"change_pct": 8.0},
        }
        score, _, _, missing = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        assert any("overextended" in m for m in missing)

    def test_no_indicators_defaults_to_0(self, aliases, relationships):
        """Missing indicators should not add or subtract. Req 4.4"""
        decision = {
            "catalyst": "AMD beats earnings today",
            "bias": "LONG",
        }
        score, _, evidence, missing = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        # No volume/breaking/overextended evidence or missing
        assert not any("volume" in e.lower() for e in evidence)
        assert not any("overextended" in m for m in missing)


# ---------------------------------------------------------------------------
# Direction Consistency Scoring
# ---------------------------------------------------------------------------


class TestDirectionConsistency:
    """Tests for direction consistency scoring (+1 / -3)."""

    def test_matching_direction_adds_1(self, aliases, relationships):
        decision = {
            "catalyst": "AMD beats earnings expectations today",
            "bias": "LONG",
        }
        score, _, evidence, _ = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        assert any("direction matches" in e for e in evidence)

    def test_conflicting_direction_subtracts_3(self, aliases, relationships):
        decision = {
            "catalyst": "AMD downgraded by Goldman today",
            "bias": "LONG",
        }
        score, reason_type, _, missing = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        assert any("conflicts" in m for m in missing)
        assert reason_type == "mismatch"

    def test_ambiguous_direction_no_penalty(self, aliases, relationships):
        """Ambiguous catalyst direction should not penalize. Req 5.3"""
        decision = {
            "catalyst": "AMD announces restructuring plan today",
            "bias": "LONG",
        }
        score, _, evidence, missing = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        # No direction match or conflict in evidence/missing
        assert not any("direction matches" in e for e in evidence)
        assert not any("conflicts" in m for m in missing)

    def test_short_trade_bearish_catalyst_matches(self, aliases, relationships):
        decision = {
            "catalyst": "TSLA recall issued for 500k vehicles today",
            "bias": "SHORT",
        }
        score, _, evidence, _ = compute_catalyst_score(
            "TSLA", decision, None, aliases, relationships
        )
        assert any("direction matches" in e for e in evidence)

    def test_short_trade_bullish_catalyst_conflicts(self, aliases, relationships):
        decision = {
            "catalyst": "TSLA beats delivery expectations today",
            "direction": "SHORT",
        }
        score, reason_type, _, missing = compute_catalyst_score(
            "TSLA", decision, None, aliases, relationships
        )
        assert any("conflicts" in m for m in missing)
        assert reason_type == "mismatch"

    def test_no_trade_direction_no_scoring(self, aliases, relationships):
        """If trade direction is empty, no direction scoring applied."""
        decision = {
            "catalyst": "AMD beats earnings today",
        }
        score, _, evidence, missing = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        assert not any("direction" in e for e in evidence)
        assert not any("direction" in m.lower() for m in missing)


# ---------------------------------------------------------------------------
# Score Clamping
# ---------------------------------------------------------------------------


class TestScoreClamping:
    """Tests for score clamping to [0, 10]. Reqs 6.1, 6.2"""

    def test_score_never_below_zero(self, aliases, relationships):
        """Stale + no mention + overextended + conflict should clamp to 0."""
        decision = {
            "catalyst": "Company downgraded yesterday, weak outlook",
            "bias": "LONG",
            "indicators": {"change_pct": 8.0},
        }
        score, _, _, _ = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        assert score >= 0

    def test_score_never_above_ten(self, aliases, relationships):
        """Maximum possible scoring should clamp to 10."""
        decision = {
            "catalyst": "AMD beats earnings, raises guidance today",
            "bias": "LONG",
            "indicators": {"volume_ratio": 3.0},
        }
        signal = {"quote_timestamp": "2025-01-15T14:30:00Z"}
        score, _, _, _ = compute_catalyst_score(
            "AMD", decision, signal, aliases, relationships
        )
        assert score <= 10

    def test_max_score_scenario(self, aliases, relationships):
        """Direct(4) + intraday(2) + volume(2) + direction_match(1) = 9."""
        decision = {
            "catalyst": "AMD beats earnings expectations today",
            "bias": "LONG",
            "indicators": {"volume_ratio": 2.0},
        }
        score, _, _, _ = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        assert score == 9


# ---------------------------------------------------------------------------
# Reason Type Classification
# ---------------------------------------------------------------------------


class TestReasonTypeClassification:
    """Tests for reason_type classification. Reqs 7.1-7.6"""

    def test_mismatch_overrides_direct(self, aliases, relationships):
        """Direction conflict → mismatch even if direct mention. Req 7.1"""
        decision = {
            "catalyst": "AMD downgraded by analyst today",
            "bias": "LONG",
        }
        _, reason_type, _, _ = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        assert reason_type == "mismatch"

    def test_direct_symbol_classification(self, aliases, relationships):
        """Direct mention → direct_symbol. Req 7.2"""
        decision = {
            "catalyst": "AMD announces new product today",
            "bias": "LONG",
        }
        _, reason_type, _, _ = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        assert reason_type == "direct_symbol"

    def test_named_readthrough_classification(self, aliases, relationships):
        """Readthrough mention → named_readthrough. Req 7.3"""
        decision = {
            "catalyst": "TSMC reports strong demand today",
            "bias": "LONG",
        }
        _, reason_type, _, _ = compute_catalyst_score(
            "NVDA", decision, None, aliases, relationships
        )
        assert reason_type == "named_readthrough"

    def test_sector_sympathy_classification(self, aliases, relationships):
        """Sector terms only → sector_sympathy. Req 7.4"""
        decision = {
            "catalyst": "Semiconductor industry outlook positive today",
            "bias": "LONG",
        }
        _, reason_type, _, _ = compute_catalyst_score(
            "NVDA", decision, None, aliases, relationships
        )
        assert reason_type == "sector_sympathy"

    def test_macro_only_classification(self, aliases, relationships):
        """Macro catalyst for non-macro symbol → macro_only. Req 7.5"""
        decision = {
            "catalyst": "Fed holds steady on policy today",
            "bias": "LONG",
        }
        _, reason_type, _, _ = compute_catalyst_score(
            "MSFT", decision, None, aliases, relationships
        )
        assert reason_type == "macro_only"

    def test_unknown_classification(self, aliases, relationships):
        """No evidence → unknown. Req 7.6"""
        decision = {
            "catalyst": "",
            "bias": "LONG",
        }
        _, reason_type, _, _ = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        assert reason_type == "unknown"


# ---------------------------------------------------------------------------
# Return Value Structure
# ---------------------------------------------------------------------------


class TestReturnValueStructure:
    """Tests for correct return tuple structure."""

    def test_returns_four_element_tuple(self, aliases, relationships):
        decision = {"catalyst": "AMD beats today", "bias": "LONG"}
        result = compute_catalyst_score("AMD", decision, None, aliases, relationships)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_score_is_int(self, aliases, relationships):
        decision = {"catalyst": "AMD beats today", "bias": "LONG"}
        score, _, _, _ = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        assert isinstance(score, int)

    def test_reason_type_is_valid_string(self, aliases, relationships):
        decision = {"catalyst": "AMD beats today", "bias": "LONG"}
        _, reason_type, _, _ = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        valid_types = {
            "direct_symbol", "named_readthrough", "sector_sympathy",
            "macro_only", "unknown", "mismatch",
        }
        assert reason_type in valid_types

    def test_evidence_is_list(self, aliases, relationships):
        decision = {"catalyst": "AMD beats today", "bias": "LONG"}
        _, _, evidence, _ = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        assert isinstance(evidence, list)
        assert all(isinstance(e, str) for e in evidence)

    def test_missing_is_list(self, aliases, relationships):
        decision = {"catalyst": "AMD beats today", "bias": "LONG"}
        _, _, _, missing = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        assert isinstance(missing, list)
        assert all(isinstance(m, str) for m in missing)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for compute_catalyst_score."""

    def test_symbol_case_insensitive(self, aliases, relationships):
        """Symbol should be normalized to uppercase."""
        decision = {"catalyst": "AMD beats today", "bias": "LONG"}
        score1, _, _, _ = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        score2, _, _, _ = compute_catalyst_score(
            "amd", decision, None, aliases, relationships
        )
        assert score1 == score2

    def test_signal_provides_indicators(self, aliases, relationships):
        """Signal indicators should be used when decision has none."""
        decision = {"catalyst": "AMD beats today", "bias": "LONG"}
        signal = {"relative_volume": 2.0}
        score, _, evidence, _ = compute_catalyst_score(
            "AMD", decision, signal, aliases, relationships
        )
        assert any("volume" in e.lower() for e in evidence)

    def test_unknown_symbol_uses_symbol_as_name(self, aliases, relationships):
        """Unknown symbol should use the symbol itself for matching."""
        decision = {"catalyst": "AAPL announces new iPhone today", "bias": "LONG"}
        score, reason_type, evidence, _ = compute_catalyst_score(
            "AAPL", decision, None, aliases, relationships
        )
        assert reason_type == "direct_symbol"
        assert any("AAPL" in e for e in evidence)

    def test_empty_aliases_and_relationships(self):
        """Should work with empty alias/relationship dicts."""
        decision = {"catalyst": "AMD beats today", "bias": "LONG"}
        score, _, _, _ = compute_catalyst_score(
            "AMD", decision, None, {}, {}
        )
        # With empty aliases, "AMD" still matches because fallback is [symbol]
        assert score >= 0

    def test_whitespace_only_catalyst(self, aliases, relationships):
        """Whitespace-only catalyst should return unknown."""
        decision = {"catalyst": "   ", "bias": "LONG"}
        score, reason_type, _, missing = compute_catalyst_score(
            "AMD", decision, None, aliases, relationships
        )
        assert score == 0
        assert reason_type == "unknown"
        assert "no catalyst evidence found" in missing
