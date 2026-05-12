"""
Unit tests for catalyst specificity helper functions (task 3.1).

Tests: extract_catalyst_text, any_mention, contains_sector_terms,
assess_freshness, extract_indicators, infer_catalyst_direction,
directions_match, directions_conflict, is_breaking_level,
is_strong_signal, is_overextended, is_macro_catalyst.

Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3,
              5.1, 5.2, 5.3, 5.4, 15.1, 15.2
"""

import pytest

from utils.catalyst_specificity import (
    any_mention,
    assess_freshness,
    contains_sector_terms,
    directions_conflict,
    directions_match,
    extract_catalyst_text,
    extract_indicators,
    infer_catalyst_direction,
    is_breaking_level,
    is_macro_catalyst,
    is_overextended,
    is_strong_signal,
)


# ===================================================================
# Tests for extract_catalyst_text
# ===================================================================


class TestExtractCatalystText:
    """Tests for extract_catalyst_text — gathers catalyst text from merged context."""

    def test_extracts_from_decision_fields(self):
        decision = {
            "catalyst": "AMD beats earnings",
            "rationale": "Strong Q2 results",
            "thesis": "AI demand growing",
        }
        result = extract_catalyst_text(decision)
        assert "AMD beats earnings" in result
        assert "Strong Q2 results" in result
        assert "AI demand growing" in result

    def test_decision_overrides_signal(self):
        decision = {"catalyst": "AMD beats earnings"}
        signal = {"catalyst": "Old catalyst from signal"}
        result = extract_catalyst_text(decision, signal)
        assert "AMD beats earnings" in result
        assert "Old catalyst from signal" not in result

    def test_falls_back_to_signal_when_decision_empty(self):
        decision = {}
        signal = {"catalyst": "Signal catalyst text", "rationale": "Signal rationale"}
        result = extract_catalyst_text(decision, signal)
        assert "Signal catalyst text" in result
        assert "Signal rationale" in result

    def test_includes_news_catalyst_field(self):
        decision = {"news_catalyst": "Breaking: FDA approval for drug X"}
        result = extract_catalyst_text(decision)
        assert "FDA approval" in result

    def test_empty_decision_and_signal(self):
        result = extract_catalyst_text({}, None)
        assert result == ""

    def test_ignores_none_values(self):
        decision = {"catalyst": None, "rationale": "Some rationale"}
        result = extract_catalyst_text(decision)
        assert "Some rationale" in result
        assert "None" not in result

    def test_strips_whitespace(self):
        decision = {"catalyst": "  AMD earnings  "}
        result = extract_catalyst_text(decision)
        assert result == "AMD earnings"

    def test_signal_none_handled(self):
        decision = {"catalyst": "Test"}
        result = extract_catalyst_text(decision, None)
        assert "Test" in result


# ===================================================================
# Tests for any_mention
# ===================================================================


class TestAnyMention:
    """Tests for any_mention — case-insensitive substring match."""

    def test_exact_match(self):
        assert any_mention("AMD reports earnings", ["AMD"]) is True

    def test_case_insensitive(self):
        assert any_mention("nvidia announces new GPU", ["NVDA", "Nvidia"]) is True

    def test_no_match(self):
        assert any_mention("Intel announces new chip", ["AMD", "NVDA"]) is False

    def test_empty_text(self):
        assert any_mention("", ["AMD"]) is False

    def test_empty_names(self):
        assert any_mention("AMD reports earnings", []) is False

    def test_substring_match(self):
        assert any_mention("Advanced Micro Devices beats", ["Advanced Micro Devices"]) is True

    def test_multiple_names_first_matches(self):
        assert any_mention("Tesla deliveries up", ["Tesla", "TSLA"]) is True

    def test_partial_word_match(self):
        # "AMD" appears in text even as part of larger context
        assert any_mention("The AMD chip is fast", ["AMD"]) is True


# ===================================================================
# Tests for contains_sector_terms
# ===================================================================


class TestContainsSectorTerms:
    """Tests for contains_sector_terms — sector/theme keyword detection."""

    def test_tech_sector_for_xlk(self):
        assert contains_sector_terms("technology sector rally today", "XLK") is True

    def test_energy_sector_for_xle(self):
        assert contains_sector_terms("oil prices surge on OPEC news", "XLE") is True

    def test_financials_for_xlf(self):
        assert contains_sector_terms("banks report strong earnings", "XLF") is True

    def test_no_match_wrong_sector(self):
        assert contains_sector_terms("oil prices surge", "XLK") is False

    def test_empty_text(self):
        assert contains_sector_terms("", "XLK") is False

    def test_unknown_symbol(self):
        assert contains_sector_terms("some text", "UNKNOWN") is False

    def test_case_insensitive(self):
        assert contains_sector_terms("SEMICONDUCTOR demand rising", "NVDA") is True

    def test_ev_terms_for_tsla(self):
        assert contains_sector_terms("electric vehicle sales growing", "TSLA") is True

    def test_bond_terms_for_tlt(self):
        assert contains_sector_terms("Treasury yields rising sharply", "TLT") is True

    def test_gold_terms_for_gld(self):
        assert contains_sector_terms("gold prices hit new highs", "GLD") is True


# ===================================================================
# Tests for assess_freshness
# ===================================================================


class TestAssessFreshness:
    """Tests for assess_freshness — determine intraday/same_day/stale."""

    def test_intraday_keyword(self):
        decision = {"catalyst": "AMD reports intraday earnings beat"}
        assert assess_freshness(decision) == "intraday"

    def test_today_keyword(self):
        decision = {"catalyst": "Tesla announced today new factory"}
        assert assess_freshness(decision) == "intraday"

    def test_breaking_keyword(self):
        decision = {"catalyst": "Breaking: FDA approves new drug"}
        assert assess_freshness(decision) == "intraday"

    def test_stale_yesterday(self):
        decision = {"catalyst": "AMD reported yesterday after close"}
        assert assess_freshness(decision) == "stale"

    def test_stale_last_week(self):
        decision = {"catalyst": "Earnings from last week still moving"}
        assert assess_freshness(decision) == "stale"

    def test_same_day_no_indicators(self):
        decision = {"catalyst": "AMD earnings beat expectations"}
        assert assess_freshness(decision) == "same_day"

    def test_quote_timestamp_in_signal_means_intraday(self):
        decision = {"catalyst": "AMD earnings"}
        signal = {"quote_timestamp": "2025-01-15T14:30:00Z"}
        assert assess_freshness(decision, signal) == "intraday"

    def test_empty_decision(self):
        assert assess_freshness({}) == "same_day"

    def test_stale_overridden_by_intraday(self):
        # If both stale and intraday terms present, intraday wins
        # (quote_timestamp is a strong intraday signal)
        decision = {"catalyst": "yesterday's news still moving"}
        signal = {"quote_timestamp": "2025-01-15T14:30:00Z"}
        assert assess_freshness(decision, signal) == "intraday"


# ===================================================================
# Tests for extract_indicators
# ===================================================================


class TestExtractIndicators:
    """Tests for extract_indicators — merge indicator dicts."""

    def test_decision_indicators_override_signal(self):
        decision = {"indicators": {"relative_volume": 2.5}}
        signal = {"indicators": {"relative_volume": 1.2, "change_pct": 3.0}}
        result = extract_indicators(decision, signal)
        assert result["relative_volume"] == 2.5
        assert result["change_pct"] == 3.0

    def test_signal_top_level_fields(self):
        decision = {}
        signal = {"relative_volume": 1.8, "current_price": 150.0}
        result = extract_indicators(decision, signal)
        assert result["relative_volume"] == 1.8
        assert result["current_price"] == 150.0

    def test_empty_both(self):
        result = extract_indicators({}, None)
        assert result == {}

    def test_graceful_with_non_dict_indicators(self):
        decision = {"indicators": "not a dict"}
        signal = {"indicators": None}
        result = extract_indicators(decision, signal)
        assert result == {}

    def test_signal_none(self):
        decision = {"indicators": {"volume_ratio": 2.0}}
        result = extract_indicators(decision, None)
        assert result["volume_ratio"] == 2.0

    def test_missing_volume_data(self):
        decision = {"indicators": {"change_pct": 1.5}}
        result = extract_indicators(decision)
        assert "relative_volume" not in result
        assert "volume_ratio" not in result
        assert result["change_pct"] == 1.5


# ===================================================================
# Tests for infer_catalyst_direction
# ===================================================================


class TestInferCatalystDirection:
    """Tests for infer_catalyst_direction — CONSERVATIVE classification."""

    def test_bullish_beat(self):
        assert infer_catalyst_direction("AMD beat earnings expectations") == "BULLISH"

    def test_bullish_raises(self):
        assert infer_catalyst_direction("Company raises guidance for Q3") == "BULLISH"

    def test_bullish_upgrade(self):
        assert infer_catalyst_direction("Goldman upgrade to buy") == "BULLISH"

    def test_bullish_approval(self):
        assert infer_catalyst_direction("FDA approval for new drug") == "BULLISH"

    def test_bullish_contract_win(self):
        assert infer_catalyst_direction("Major contract win with DoD") == "BULLISH"

    def test_bearish_downgrade(self):
        assert infer_catalyst_direction("Analyst downgrade to sell") == "BEARISH"

    def test_bearish_miss(self):
        assert infer_catalyst_direction("Company miss on revenue") == "BEARISH"

    def test_bearish_cut(self):
        assert infer_catalyst_direction("Dividend cut announced") == "BEARISH"

    def test_bearish_probe(self):
        assert infer_catalyst_direction("SEC probe into accounting") == "BEARISH"

    def test_bearish_recall(self):
        assert infer_catalyst_direction("Product recall issued") == "BEARISH"

    def test_none_for_ambiguous(self):
        assert infer_catalyst_direction("Company restructuring plans") is None

    def test_none_for_mixed_signals(self):
        # Both bullish and bearish keywords → None
        assert infer_catalyst_direction("Beat on earnings but downgrade from analyst") is None

    def test_none_for_empty_text(self):
        assert infer_catalyst_direction("") is None

    def test_none_for_no_keywords(self):
        assert infer_catalyst_direction("Stock moved on volume") is None

    def test_case_insensitive(self):
        assert infer_catalyst_direction("UPGRADE from Morgan Stanley") == "BULLISH"


# ===================================================================
# Tests for directions_match and directions_conflict
# ===================================================================


class TestDirectionsMatch:
    """Tests for directions_match."""

    def test_long_bullish_match(self):
        assert directions_match("LONG", "BULLISH") is True

    def test_short_bearish_match(self):
        assert directions_match("SHORT", "BEARISH") is True

    def test_buy_bullish_match(self):
        assert directions_match("BUY", "BULLISH") is True

    def test_sell_bearish_match(self):
        assert directions_match("SELL", "BEARISH") is True

    def test_long_bearish_no_match(self):
        assert directions_match("LONG", "BEARISH") is False

    def test_none_catalyst_dir(self):
        assert directions_match("LONG", None) is False

    def test_empty_trade_dir(self):
        assert directions_match("", "BULLISH") is False

    def test_case_insensitive_trade_dir(self):
        assert directions_match("long", "BULLISH") is True


class TestDirectionsConflict:
    """Tests for directions_conflict."""

    def test_long_bearish_conflict(self):
        assert directions_conflict("LONG", "BEARISH") is True

    def test_short_bullish_conflict(self):
        assert directions_conflict("SHORT", "BULLISH") is True

    def test_long_bullish_no_conflict(self):
        assert directions_conflict("LONG", "BULLISH") is False

    def test_none_catalyst_no_conflict(self):
        assert directions_conflict("LONG", None) is False

    def test_empty_trade_no_conflict(self):
        assert directions_conflict("", "BEARISH") is False

    def test_buy_bearish_conflict(self):
        assert directions_conflict("BUY", "BEARISH") is True

    def test_sell_bullish_conflict(self):
        assert directions_conflict("SELL", "BULLISH") is True


# ===================================================================
# Tests for is_breaking_level
# ===================================================================


class TestIsBreakingLevel:
    """Tests for is_breaking_level."""

    def test_at_day_high(self):
        indicators = {"current_price": 100.0, "day_high": 100.0}
        assert is_breaking_level(indicators) is True

    def test_near_day_high(self):
        # Within 0.5% of day high
        indicators = {"current_price": 99.6, "day_high": 100.0}
        assert is_breaking_level(indicators) is True

    def test_below_day_high(self):
        indicators = {"current_price": 95.0, "day_high": 100.0}
        assert is_breaking_level(indicators) is False

    def test_at_day_low(self):
        indicators = {"current_price": 90.0, "day_low": 90.0}
        assert is_breaking_level(indicators) is True

    def test_explicit_breaking_level_flag(self):
        indicators = {"current_price": 50.0, "breaking_level": True}
        assert is_breaking_level(indicators) is True

    def test_empty_indicators(self):
        assert is_breaking_level({}) is False

    def test_no_current_price(self):
        indicators = {"day_high": 100.0}
        assert is_breaking_level(indicators) is False


# ===================================================================
# Tests for is_strong_signal
# ===================================================================


class TestIsStrongSignal:
    """Tests for is_strong_signal."""

    def test_strong_strength(self):
        signal = {"strength": "strong"}
        assert is_strong_signal(signal) is True

    def test_high_strength(self):
        signal = {"strength": "high"}
        assert is_strong_signal(signal) is True

    def test_high_conviction(self):
        signal = {"conviction": "high"}
        assert is_strong_signal(signal) is True

    def test_weak_signal(self):
        signal = {"strength": "weak"}
        assert is_strong_signal(signal) is False

    def test_none_signal(self):
        assert is_strong_signal(None) is False

    def test_empty_signal(self):
        assert is_strong_signal({}) is False

    def test_medium_strength_not_strong(self):
        signal = {"strength": "medium"}
        assert is_strong_signal(signal) is False


# ===================================================================
# Tests for is_overextended
# ===================================================================


class TestIsOverextended:
    """Tests for is_overextended."""

    def test_large_move_no_support(self):
        indicators = {"change_pct": 6.0}
        assert is_overextended(indicators) is True

    def test_large_negative_move_no_support(self):
        indicators = {"change_pct": -7.0}
        assert is_overextended(indicators) is True

    def test_small_move(self):
        indicators = {"change_pct": 2.0}
        assert is_overextended(indicators) is False

    def test_large_move_with_support(self):
        indicators = {"change_pct": 6.0, "has_support": True}
        assert is_overextended(indicators) is False

    def test_large_move_retested(self):
        indicators = {"change_pct": 6.0, "retested": True}
        assert is_overextended(indicators) is False

    def test_empty_indicators(self):
        assert is_overextended({}) is False

    def test_uses_price_change_pct_field(self):
        indicators = {"price_change_pct": 8.0}
        assert is_overextended(indicators) is True


# ===================================================================
# Tests for is_macro_catalyst
# ===================================================================


class TestIsMacroCatalyst:
    """Tests for is_macro_catalyst — detect broad macro terms."""

    def test_fed(self):
        assert is_macro_catalyst("Fed raises rates by 25bps") is True

    def test_cpi(self):
        assert is_macro_catalyst("CPI comes in hot at 3.5%") is True

    def test_pce(self):
        assert is_macro_catalyst("PCE data shows cooling inflation") is True

    def test_jobs_report(self):
        assert is_macro_catalyst("Jobs report beats expectations") is True

    def test_fomc(self):
        assert is_macro_catalyst("FOMC meeting minutes released") is True

    def test_gdp(self):
        assert is_macro_catalyst("GDP growth slows to 1.5%") is True

    def test_inflation(self):
        assert is_macro_catalyst("Inflation concerns mount") is True

    def test_interest_rate(self):
        assert is_macro_catalyst("Interest rate decision tomorrow") is True

    def test_treasury(self):
        assert is_macro_catalyst("Treasury yields spike") is True

    def test_yields(self):
        assert is_macro_catalyst("10-year yields hit 5%") is True

    def test_not_macro(self):
        assert is_macro_catalyst("AMD beats earnings") is False

    def test_empty_text(self):
        assert is_macro_catalyst("") is False

    def test_case_insensitive(self):
        assert is_macro_catalyst("the FOMC decision was hawkish") is True
