"""
Unit tests for the Similarity Matching Engine (core/similarity.py).
"""

import pytest
from core.similarity import (
    compute_similarity_score,
    compute_similarity_stats,
    SIMILARITY_WEIGHTS,
)


# ---------------------------------------------------------------------------
# test_empty_case_list_returns_skip_similarity
# ---------------------------------------------------------------------------
class TestEmptyCaseList:
    def test_empty_case_list_returns_skip_similarity(self):
        """Empty input → skip_similarity=True with zeroed stats."""
        result = compute_similarity_stats([])
        assert result["skip_similarity"] is True
        assert result["similarity_winrate"] == 0.0
        assert result["similarity_avg_r"] == 0.0
        assert result["sample_size"] == 0
        assert result["similarity_confidence"] == 0.0


# ---------------------------------------------------------------------------
# test_stat_aggregation_with_confidence
# ---------------------------------------------------------------------------
class TestStatAggregation:
    def test_stat_aggregation_with_confidence(self):
        """Known data → correct winrate, avg_r, sample_size, confidence."""
        cases = [
            {"outcome": "success", "pnl_pct": 3.0},
            {"outcome": "success", "pnl_pct": 5.0},
            {"outcome": "failure", "pnl_pct": -2.0},
            {"outcome": "failure", "pnl_pct": -1.0},
            {"outcome": "success", "pnl_pct": 4.0},
        ]
        result = compute_similarity_stats(cases)

        assert result["sample_size"] == 5
        assert result["similarity_winrate"] == pytest.approx(3 / 5)
        assert result["similarity_avg_r"] == pytest.approx(
            (3.0 + 5.0 - 2.0 - 1.0 + 4.0) / 5
        )
        assert result["similarity_confidence"] == pytest.approx(0.5)
        assert "skip_similarity" not in result

    def test_null_pnl_excluded_from_avg_r(self):
        """Cases with None pnl_pct are excluded from avg_r calculation."""
        cases = [
            {"outcome": "success", "pnl_pct": 4.0},
            {"outcome": "failure", "pnl_pct": None},
        ]
        result = compute_similarity_stats(cases)
        assert result["similarity_avg_r"] == pytest.approx(4.0)
        assert result["sample_size"] == 2

    def test_confidence_caps_at_one(self):
        """sample_size >= 10 → similarity_confidence == 1.0."""
        cases = [{"outcome": "success", "pnl_pct": 1.0}] * 15
        result = compute_similarity_stats(cases)
        assert result["similarity_confidence"] == 1.0


# ---------------------------------------------------------------------------
# test_weighted_scoring_partial_matches
# ---------------------------------------------------------------------------
class TestWeightedScoringPartialMatches:
    def test_partial_match_nonzero_score(self):
        """Cases matching on some criteria still get non-zero scores."""
        signal = {
            "setup_type": "gap_and_go",
            "market_regime": "risk_on",
            "rsi": 55.0,
            "above_vwap": True,
            "ema_trend": "bullish",
        }
        # Only setup_type matches
        case = {
            "setup_type": "gap_and_go",
            "market_regime": "risk_off",
            "rsi_at_entry": 30.0,
            "above_vwap": "false",
            "ema_trend": "bearish",
        }
        score = compute_similarity_score(case, signal)
        assert score > 0.0, "Partial match should produce a non-zero score"

    def test_more_matches_higher_score(self):
        """More matching criteria → higher score."""
        signal = {
            "setup_type": "gap_and_go",
            "market_regime": "risk_on",
            "rsi": 55.0,
            "above_vwap": True,
            "ema_trend": "bullish",
        }
        case_one_match = {
            "setup_type": "gap_and_go",
            "market_regime": "risk_off",
            "rsi_at_entry": 10.0,
            "above_vwap": "false",
            "ema_trend": "bearish",
        }
        case_three_matches = {
            "setup_type": "gap_and_go",
            "market_regime": "risk_on",
            "rsi_at_entry": 55.0,
            "above_vwap": "false",
            "ema_trend": "bearish",
        }
        score_one = compute_similarity_score(case_one_match, signal)
        score_three = compute_similarity_score(case_three_matches, signal)
        assert score_three > score_one


# ---------------------------------------------------------------------------
# test_rsi_distance_scoring
# ---------------------------------------------------------------------------
class TestRsiDistanceScoring:
    def test_closer_rsi_higher_score(self):
        """Closer RSI → higher similarity score (continuous, not bucketed)."""
        signal = {
            "setup_type": "gap_and_go",
            "market_regime": "risk_on",
            "rsi": 50.0,
            "above_vwap": True,
            "ema_trend": "bullish",
        }
        case_close = {
            "setup_type": "gap_and_go",
            "market_regime": "risk_on",
            "rsi_at_entry": 52.0,
            "above_vwap": "true",
            "ema_trend": "bullish",
        }
        case_far = {
            "setup_type": "gap_and_go",
            "market_regime": "risk_on",
            "rsi_at_entry": 90.0,
            "above_vwap": "true",
            "ema_trend": "bullish",
        }
        score_close = compute_similarity_score(case_close, signal)
        score_far = compute_similarity_score(case_far, signal)
        assert score_close > score_far

    def test_exact_rsi_gives_full_rsi_weight(self):
        """Identical RSI → rsi_distance component is 1.0 (full contribution)."""
        signal = {
            "setup_type": "gap_and_go",
            "market_regime": "risk_on",
            "rsi": 60.0,
            "above_vwap": True,
            "ema_trend": "bullish",
        }
        case_exact_rsi = {
            "setup_type": "gap_and_go",
            "market_regime": "risk_on",
            "rsi_at_entry": 60.0,
            "above_vwap": "true",
            "ema_trend": "bullish",
        }
        case_far_rsi = {
            "setup_type": "gap_and_go",
            "market_regime": "risk_on",
            "rsi_at_entry": 10.0,
            "above_vwap": "true",
            "ema_trend": "bullish",
        }
        score_exact = compute_similarity_score(case_exact_rsi, signal)
        score_far = compute_similarity_score(case_far_rsi, signal)
        # The only difference is RSI distance, so exact RSI should score higher
        assert score_exact > score_far
        # Exact RSI match with all other criteria matching → perfect score
        assert score_exact == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# test_compute_similarity_score_exact_match
# ---------------------------------------------------------------------------
class TestExactMatch:
    def test_all_criteria_match_score_near_one(self):
        """All criteria match → score near 1.0."""
        signal = {
            "setup_type": "gap_and_go",
            "market_regime": "risk_on",
            "rsi": 55.0,
            "above_vwap": True,
            "ema_trend": "bullish",
        }
        case = {
            "setup_type": "gap_and_go",
            "market_regime": "risk_on",
            "rsi_at_entry": 55.0,
            "above_vwap": "true",
            "ema_trend": "bullish",
        }
        score = compute_similarity_score(case, signal)
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# test_compute_similarity_score_no_match
# ---------------------------------------------------------------------------
class TestNoMatch:
    def test_no_criteria_match_score_near_zero(self):
        """No criteria match → score near 0.0."""
        signal = {
            "setup_type": "gap_and_go",
            "market_regime": "risk_on",
            "rsi": 50.0,
            "above_vwap": True,
            "ema_trend": "bullish",
        }
        case = {
            "setup_type": "momentum_fade",
            "market_regime": "risk_off",
            "rsi_at_entry": 0.0,
            "above_vwap": "false",
            "ema_trend": "bearish",
        }
        score = compute_similarity_score(case, signal)
        # RSI distance: 1.0 - 50/100 = 0.5, so there's a small RSI contribution
        # but setup, regime, vwap, trend all miss → score should be low
        assert score < 0.15
