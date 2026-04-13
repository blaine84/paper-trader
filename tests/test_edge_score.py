"""
Tests for core.edge_score — property-based and unit tests.

Covers Properties 1–6 from the design document and unit tests for
helper mappings and the no-LLM-calls guarantee.
"""

import math
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from core.edge_score import (
    WEIGHTS,
    compute_edge_score,
    normalize_winrate,
    map_strength,
    map_confidence,
    confluence_score,
    similarity_quality,
    check_hard_rejection,
    cap_position_size,
)


# ---------------------------------------------------------------------------
# Shared Hypothesis strategies
# ---------------------------------------------------------------------------

strength_st = st.sampled_from(["weak", "moderate", "strong"])
confidence_st = st.sampled_from(["low", "medium", "high"])
bias_st = st.sampled_from(["LONG", "SHORT"])

indicators_st = st.fixed_dictionaries({
    "above_vwap": st.booleans(),
    "ema_trend": st.sampled_from(["bullish", "bearish"]),
    "rsi": st.floats(min_value=0.0, max_value=100.0),
    "macd_bias": st.sampled_from(["bullish", "bearish"]),
    "bb_position": st.sampled_from(["upper", "lower"]),
})

valid_signal_st = st.fixed_dictionaries({
    "strength": strength_st,
    "confidence": confidence_st,
    "bias": bias_st,
    "indicators": indicators_st,
})

case_stats_st = st.fixed_dictionaries({
    "win_rate": st.floats(min_value=0.0, max_value=1.0),
    "sample_size": st.integers(min_value=0, max_value=100),
})

similarity_stats_st = st.fixed_dictionaries({
    "similarity_winrate": st.floats(min_value=0.0, max_value=1.0),
    "sample_size": st.integers(min_value=0, max_value=100),
})


# ===================================================================
# Task 1.3 — Property 1: Edge score formula correctness (v2)
# ===================================================================

@given(signal=valid_signal_st, case_stats=case_stats_st, sim_stats=similarity_stats_st)
@settings(max_examples=100)
def test_edge_score_formula_v2(signal, case_stats, sim_stats):
    """
    **Validates: Requirements 1.1, 1.9, 1.10**

    For any valid signal, case_stats, and similarity_stats the output of
    compute_edge_score must equal the 6-component weighted formula clamped
    to [0.0, 1.0].
    """
    result = compute_edge_score(signal, case_stats, sim_stats)

    # Manually compute expected value
    setup_wr = normalize_winrate(case_stats["win_rate"])
    sim_wr = normalize_winrate(sim_stats["similarity_winrate"])
    strength = map_strength(signal["strength"])
    confidence = map_confidence(signal["confidence"])
    confluence = confluence_score(signal["indicators"], signal["bias"])
    sim_qual = similarity_quality(sim_stats["sample_size"])

    expected_raw = (
        WEIGHTS["setup_winrate"] * setup_wr
        + WEIGHTS["similarity_winrate"] * sim_wr
        + WEIGHTS["signal_strength"] * strength
        + WEIGHTS["signal_confidence"] * confidence
        + WEIGHTS["confluence"] * confluence
        + WEIGHTS["similarity_quality"] * sim_qual
    )
    expected = max(0.0, min(1.0, expected_raw))

    assert math.isclose(result, expected, abs_tol=1e-9), (
        f"Expected {expected}, got {result}"
    )


# ===================================================================
# Task 1.3 — Property 2: Edge score output range invariant
# ===================================================================

# Strategy that generates adversarial / arbitrary inputs
_any_value = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=20),
)

adversarial_signal_st = st.one_of(
    st.fixed_dictionaries({}),
    st.fixed_dictionaries({
        "strength": _any_value,
        "confidence": _any_value,
        "bias": _any_value,
        "indicators": st.one_of(
            st.fixed_dictionaries({}),
            st.dictionaries(st.text(max_size=10), _any_value, max_size=6),
        ),
    }),
    st.dictionaries(st.text(max_size=10), _any_value, max_size=6),
)

adversarial_case_st = st.one_of(
    st.fixed_dictionaries({}),
    st.fixed_dictionaries({
        "win_rate": st.one_of(st.floats(allow_nan=False, allow_infinity=False), st.integers()),
        "sample_size": st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False)),
    }),
    st.dictionaries(st.text(max_size=10), _any_value, max_size=4),
)

adversarial_sim_st = st.one_of(
    st.fixed_dictionaries({}),
    st.fixed_dictionaries({
        "similarity_winrate": st.one_of(st.floats(allow_nan=False, allow_infinity=False), st.integers()),
        "sample_size": st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False)),
    }),
    st.dictionaries(st.text(max_size=10), _any_value, max_size=4),
)


@given(signal=adversarial_signal_st, case_stats=adversarial_case_st, sim_stats=adversarial_sim_st)
@settings(max_examples=100)
def test_edge_score_range_invariant(signal, case_stats, sim_stats):
    """
    **Validates: Requirements 1.2**

    For ANY input (including adversarial), compute_edge_score must return
    a float in [0.0, 1.0] without raising an unhandled exception.
    """
    result = compute_edge_score(signal, case_stats, sim_stats)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0, f"Out of range: {result}"


# ===================================================================
# Task 1.4 — Property 3: Confluence score correctness
# ===================================================================

@given(indicators=indicators_st, bias=bias_st)
@settings(max_examples=100)
def test_confluence_score_correctness(indicators, bias):
    """
    **Validates: Requirements 1.7**

    confluence_score must equal aligned_count / 5 where alignment is
    computed per the design spec.
    """
    result = confluence_score(indicators, bias)

    # Manually count aligned indicators
    is_long = bias == "LONG"
    aligned = 0

    # 1. above_vwap
    if (is_long and indicators["above_vwap"]) or (not is_long and not indicators["above_vwap"]):
        aligned += 1

    # 2. ema_trend
    ema = indicators["ema_trend"].lower()
    if (is_long and ema == "bullish") or (not is_long and ema == "bearish"):
        aligned += 1

    # 3. rsi
    rsi = indicators["rsi"]
    if is_long and 30 <= rsi <= 70:
        aligned += 1
    elif not is_long and (rsi > 70 or rsi < 30):
        aligned += 1

    # 4. macd_bias
    macd = indicators["macd_bias"].lower()
    if (is_long and macd == "bullish") or (not is_long and macd == "bearish"):
        aligned += 1

    # 5. bb_position
    bb = indicators["bb_position"].lower()
    if (is_long and bb == "upper") or (not is_long and bb == "lower"):
        aligned += 1

    expected = aligned / 5.0
    assert math.isclose(result, expected, abs_tol=1e-9), (
        f"Expected {expected}, got {result}"
    )


# ===================================================================
# Task 1.5 — Property 4: Hard rejection rule correctness
# ===================================================================

@given(
    win_rate=st.floats(min_value=-1.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    sample_size=st.integers(min_value=0, max_value=200),
)
@settings(max_examples=100)
def test_hard_rejection_rule(win_rate, sample_size):
    """
    **Validates: Requirements 1.11**

    check_hard_rejection returns True iff sample_size >= 10 AND win_rate < 0.35.
    """
    case_stats = {"win_rate": win_rate, "sample_size": sample_size}
    result = check_hard_rejection(case_stats)
    expected = sample_size >= 10 and win_rate < 0.35
    assert result == expected, (
        f"win_rate={win_rate}, sample_size={sample_size}: expected {expected}, got {result}"
    )


# ===================================================================
# Task 1.5 — Property 5: Position sizing cap invariant
# ===================================================================

@given(
    scaled_size=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
    base_size=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_position_sizing_cap_invariant(scaled_size, base_size):
    """
    **Validates: Requirements 1.12, 4.3**

    cap_position_size result must always be <= base_size * 1.2.
    """
    result = cap_position_size(scaled_size, base_size)
    assert result <= base_size * 1.2 + 1e-9, (
        f"scaled={scaled_size}, base={base_size}: result {result} > cap {base_size * 1.2}"
    )


# ===================================================================
# Task 1.6 — Property 6: Similarity confidence computation
# ===================================================================

@given(sample_size=st.integers(min_value=0, max_value=1000))
@settings(max_examples=100)
def test_similarity_confidence_computation(sample_size):
    """
    **Validates: Requirements 2.3**

    similarity_quality must equal min(1.0, sample_size / 10).
    """
    result = similarity_quality(sample_size)
    expected = min(1.0, sample_size / 10.0)
    assert math.isclose(result, expected, abs_tol=1e-9), (
        f"sample_size={sample_size}: expected {expected}, got {result}"
    )


# ===================================================================
# Task 1.7 — Unit tests
# ===================================================================


class TestMapStrength:
    def test_strong(self):
        assert map_strength("strong") == 1.0

    def test_moderate(self):
        assert map_strength("moderate") == 0.6

    def test_weak(self):
        assert map_strength("weak") == 0.3

    def test_unknown(self):
        assert map_strength("unknown") == 0.0


class TestMapConfidence:
    def test_high(self):
        assert map_confidence("high") == 1.0

    def test_medium(self):
        assert map_confidence("medium") == 0.6

    def test_low(self):
        assert map_confidence("low") == 0.3

    def test_unknown(self):
        assert map_confidence("unknown") == 0.0


class TestNormalizeWinrate:
    def test_zero(self):
        assert normalize_winrate(0.0) == 0.0

    def test_half(self):
        assert normalize_winrate(0.5) == 0.5

    def test_one(self):
        assert normalize_winrate(1.0) == 1.0

    def test_negative_clamps(self):
        assert normalize_winrate(-0.5) == 0.0

    def test_above_one_clamps(self):
        assert normalize_winrate(1.5) == 1.0


class TestSimilarityQualityExamples:
    def test_zero(self):
        assert similarity_quality(0) == 0.0

    def test_five(self):
        assert similarity_quality(5) == 0.5

    def test_ten(self):
        assert similarity_quality(10) == 1.0

    def test_twenty(self):
        assert similarity_quality(20) == 1.0


def test_no_llm_calls():
    """Verify compute_edge_score makes zero LLM API calls."""
    signal = {
        "strength": "strong",
        "confidence": "high",
        "bias": "LONG",
        "indicators": {
            "above_vwap": True,
            "ema_trend": "bullish",
            "rsi": 55.0,
            "macd_bias": "bullish",
            "bb_position": "upper",
        },
    }
    case_stats = {"win_rate": 0.6, "sample_size": 15}
    sim_stats = {"similarity_winrate": 0.55, "sample_size": 8}

    with patch("utils.llm.call_llm", new_callable=MagicMock) as mock_llm:
        result = compute_edge_score(signal, case_stats, sim_stats)
        mock_llm.assert_not_called()

    assert 0.0 <= result <= 1.0
