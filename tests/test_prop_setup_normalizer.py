"""Property-based tests for utils/setup_normalizer.py.

Uses Hypothesis to validate universal correctness properties of the setup
normalizer's pure-function rule engine.
"""

from __future__ import annotations

from hypothesis import given, settings, strategies as st, assume

from utils.gate_config import SWING_EXECUTABLE_SETUP_TYPES
from utils.setup_normalizer import (
    REJECTION_REASON_CODES,
    NormalizationResult,
    TechnicalContext,
    normalize_setup,
)

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

technical_context_st = st.builds(
    TechnicalContext,
    key_levels=st.fixed_dictionaries(
        {
            "support": st.one_of(st.none(), st.floats(min_value=1.0, max_value=500.0)),
            "resistance": st.one_of(st.none(), st.floats(min_value=1.0, max_value=500.0)),
        }
    ),
    ema_trend=st.sampled_from(["bullish", "bearish", "neutral"]),
    market_regime=st.sampled_from(["risk_on", "risk_off", "mixed"]),
)

# Labels that trigger specific normalizer code paths
_KNOWN_LABELS = [
    "sector_rotation",
    "risk_off_macro_short",
    "directional_confusion_breakout",
    "error",
]

# Strategy for diverse raw labels: known labels + executable types + unknown strings
raw_label_st = st.one_of(
    st.sampled_from(list(SWING_EXECUTABLE_SETUP_TYPES) + _KNOWN_LABELS),
    st.text(min_size=1, max_size=40, alphabet=st.characters(whitelist_categories=("L", "N", "P"))),
)


# ---------------------------------------------------------------------------
# Property 2: Sector Rotation Normalization Correctness
# Validates: Requirements 2.3, 2.4
# ---------------------------------------------------------------------------


@given(
    direction=st.sampled_from(["LONG", "SHORT", "HOLD"]),
    strength=st.sampled_from(["weak", "moderate", "strong"]),
    confidence=st.sampled_from(["low", "medium", "high"]),
    technical_context=technical_context_st,
)
@settings(max_examples=200)
def test_sector_rotation_normalization_correctness(
    direction, strength, confidence, technical_context
):
    """**Validates: Requirements 2.3, 2.4**

    For raw_label "sector_rotation", returns "sector_rotation_swing" iff
    direction in {LONG, SHORT} AND confidence in {medium, high} AND
    strength in {moderate, strong} AND (key_levels has non-null value OR
    ema_trend != neutral). Otherwise rejects with
    "insufficient_normalization_evidence".
    """
    result = normalize_setup(
        "sector_rotation", direction, strength, confidence, technical_context
    )

    # Determine if all conditions are met
    direction_ok = direction in ("LONG", "SHORT")
    confidence_ok = confidence in ("medium", "high")
    strength_ok = strength in ("moderate", "strong")
    has_key_level = any(v is not None for v in technical_context.key_levels.values())
    context_ok = has_key_level or technical_context.ema_trend != "neutral"

    should_accept = direction_ok and confidence_ok and strength_ok and context_ok

    if should_accept:
        assert result.success is True
        assert result.executable_type == "sector_rotation_swing"
    else:
        assert result.success is False
        assert result.reason_code == "insufficient_normalization_evidence"


# ---------------------------------------------------------------------------
# Property 3: Risk-Off Macro Short Normalization Correctness
# Validates: Requirements 2.5, 2.6
# ---------------------------------------------------------------------------


@given(
    direction=st.sampled_from(["LONG", "SHORT", "HOLD"]),
    strength=st.sampled_from(["weak", "moderate", "strong"]),
    confidence=st.sampled_from(["low", "medium", "high"]),
    technical_context=technical_context_st,
)
@settings(max_examples=200)
def test_risk_off_macro_short_normalization_correctness(
    direction, strength, confidence, technical_context
):
    """**Validates: Requirements 2.5, 2.6**

    For raw_label "risk_off_macro_short", the normalizer returns
    "risk_off_macro_short" iff direction=SHORT AND ema_trend=bearish
    AND market_regime=risk_off. Otherwise rejects with "context_mismatch".
    """
    result = normalize_setup(
        "risk_off_macro_short", direction, strength, confidence, technical_context
    )

    should_accept = (
        direction == "SHORT"
        and technical_context.ema_trend == "bearish"
        and technical_context.market_regime == "risk_off"
    )

    if should_accept:
        assert result.success is True
        assert result.executable_type == "risk_off_macro_short"
    else:
        assert result.success is False
        assert result.reason_code == "context_mismatch"


# ---------------------------------------------------------------------------
# Property 4: Directional Confusion Breakout Resolution
# Validates: Requirements 2.7, 2.8
# ---------------------------------------------------------------------------


@given(
    direction=st.sampled_from(["LONG", "SHORT", "HOLD"]),
    strength=st.sampled_from(["weak", "moderate", "strong"]),
    confidence=st.sampled_from(["low", "medium", "high"]),
    technical_context=technical_context_st,
)
@settings(max_examples=200)
def test_directional_confusion_breakout_resolution(
    direction: str,
    strength: str,
    confidence: str,
    technical_context: TechnicalContext,
) -> None:
    """**Validates: Requirements 2.7, 2.8**

    For any input where raw_label is "directional_confusion_breakout", the normalizer
    SHALL return "breakout_retest" or "failed_breakdown_reclaim" if and only if
    ema_trend is bullish or bearish AND key_levels contains both non-null support
    AND resistance values. If bullish → "breakout_retest"; if bearish →
    "failed_breakdown_reclaim". Otherwise, reject with "diagnostic_only".
    """
    result = normalize_setup(
        "directional_confusion_breakout", direction, strength, confidence, technical_context
    )

    trend = technical_context.ema_trend
    has_both_levels = (
        technical_context.key_levels.get("support") is not None
        and technical_context.key_levels.get("resistance") is not None
    )
    resolvable = trend in ("bullish", "bearish") and has_both_levels

    if resolvable:
        assert result.success is True
        if trend == "bullish":
            assert result.executable_type == "breakout_retest"
        else:
            assert result.executable_type == "failed_breakdown_reclaim"
    else:
        assert result.success is False
        assert result.reason_code == "diagnostic_only"


# ---------------------------------------------------------------------------
# Property 8: Raw Label Preservation
# Validates: Requirements 5.1, 5.2
# ---------------------------------------------------------------------------


@given(
    raw_label=raw_label_st,
    direction=st.sampled_from(["LONG", "SHORT", "HOLD"]),
    strength=st.sampled_from(["weak", "moderate", "strong"]),
    confidence=st.sampled_from(["low", "medium", "high"]),
    technical_context=technical_context_st,
)
@settings(max_examples=200)
def test_raw_label_always_preserved_in_result(
    raw_label: str,
    direction: str,
    strength: str,
    confidence: str,
    technical_context: TechnicalContext,
) -> None:
    """**Validates: Requirements 5.1, 5.2**

    For any valid inputs to normalize_setup, the NormalizationResult always has
    raw_label equal to the input raw_label parameter. The raw analyst label is
    preserved regardless of whether normalization succeeds or fails.
    """
    result = normalize_setup(
        raw_label, direction, strength, confidence, technical_context
    )

    assert result.raw_label == raw_label


# ---------------------------------------------------------------------------
# Property 9: Sector Rotation Biconditional
# Validates: Requirements 6.1, 6.2, 6.3
# ---------------------------------------------------------------------------


@given(
    direction=st.sampled_from(["LONG", "SHORT", "HOLD"]),
    strength=st.sampled_from(["weak", "moderate", "strong"]),
    confidence=st.sampled_from(["low", "medium", "high"]),
    technical_context=technical_context_st,
)
@settings(max_examples=200)
def test_sector_rotation_biconditional(
    direction: str,
    strength: str,
    confidence: str,
    technical_context: TechnicalContext,
) -> None:
    """**Validates: Requirements 6.1, 6.2, 6.3**

    sector_rotation normalizes to sector_rotation_swing IFF:
    - direction in {LONG, SHORT} AND
    - confidence in {medium, high} AND
    - strength in {moderate, strong} AND
    - (has_key_levels OR ema_trend != neutral)

    When rejected, reason_code is "insufficient_normalization_evidence" and
    missing_evidence is a non-empty list describing which conditions failed.
    """
    result = normalize_setup(
        "sector_rotation", direction, strength, confidence, technical_context
    )

    direction_ok = direction in ("LONG", "SHORT")
    confidence_ok = confidence in ("medium", "high")
    strength_ok = strength in ("moderate", "strong")
    has_key_level = any(v is not None for v in technical_context.key_levels.values())
    context_ok = has_key_level or technical_context.ema_trend != "neutral"

    all_evidence_present = direction_ok and confidence_ok and strength_ok and context_ok

    if all_evidence_present:
        # Normalizes successfully
        assert result.success is True
        assert result.executable_type == "sector_rotation_swing"
        assert result.raw_label == "sector_rotation"
    else:
        # Rejects with evidence reporting
        assert result.success is False
        assert result.reason_code == "insufficient_normalization_evidence"
        assert result.raw_label == "sector_rotation"
        assert result.missing_evidence is not None
        assert len(result.missing_evidence) > 0


# ---------------------------------------------------------------------------
# Property 10: Near-Miss Evidence Reporting
# Validates: Requirements 9.3, 17.3
# ---------------------------------------------------------------------------


@given(
    direction=st.sampled_from(["LONG", "SHORT", "HOLD"]),
    strength=st.sampled_from(["weak", "moderate", "strong"]),
    confidence=st.sampled_from(["low", "medium", "high"]),
    technical_context=technical_context_st,
)
@settings(max_examples=200)
def test_insufficient_normalization_evidence_has_missing_evidence(
    direction: str,
    strength: str,
    confidence: str,
    technical_context: TechnicalContext,
) -> None:
    """**Validates: Requirements 9.3, 17.3**

    Whenever normalize_setup returns reason_code == "insufficient_normalization_evidence",
    the missing_evidence field is not None and has length > 0. This ensures near-miss
    rejections always identify what was missing.
    """
    result = normalize_setup(
        "sector_rotation", direction, strength, confidence, technical_context
    )

    if result.reason_code == "insufficient_normalization_evidence":
        assert result.missing_evidence is not None
        assert len(result.missing_evidence) > 0


# ---------------------------------------------------------------------------
# Property 11: Risk-Off Macro Short Biconditional
# Validates: Requirements 7.1, 7.2
# ---------------------------------------------------------------------------


@given(
    direction=st.sampled_from(["LONG", "SHORT", "HOLD"]),
    strength=st.sampled_from(["weak", "moderate", "strong"]),
    confidence=st.sampled_from(["low", "medium", "high"]),
    technical_context=technical_context_st,
)
@settings(max_examples=200)
def test_risk_off_macro_short_biconditional(
    direction: str,
    strength: str,
    confidence: str,
    technical_context: TechnicalContext,
) -> None:
    """**Validates: Requirements 7.1, 7.2**

    risk_off_macro_short normalizes IFF:
    - direction == SHORT AND
    - ema_trend == bearish AND
    - market_regime == risk_off

    When rejected, reason_code is "context_mismatch".
    """
    result = normalize_setup(
        "risk_off_macro_short", direction, strength, confidence, technical_context
    )

    should_normalize = (
        direction == "SHORT"
        and technical_context.ema_trend == "bearish"
        and technical_context.market_regime == "risk_off"
    )

    if should_normalize:
        assert result.success is True
        assert result.executable_type == "risk_off_macro_short"
        assert result.raw_label == "risk_off_macro_short"
    else:
        assert result.success is False
        assert result.reason_code == "context_mismatch"
        assert result.raw_label == "risk_off_macro_short"


# ---------------------------------------------------------------------------
# Property 12: Directional Confusion Breakout Resolution
# Validates: Requirements 8.1, 8.2, 8.3
# ---------------------------------------------------------------------------


@given(
    direction=st.sampled_from(["LONG", "SHORT", "HOLD"]),
    strength=st.sampled_from(["weak", "moderate", "strong"]),
    confidence=st.sampled_from(["low", "medium", "high"]),
    technical_context=technical_context_st,
)
@settings(max_examples=200)
def test_directional_confusion_breakout_resolution_biconditional(
    direction: str,
    strength: str,
    confidence: str,
    technical_context: TechnicalContext,
) -> None:
    """**Validates: Requirements 8.1, 8.2, 8.3**

    directional_confusion_breakout resolution:
    - neutral EMA OR missing both levels -> diagnostic_only
    - bullish EMA AND both levels -> breakout_retest
    - bearish EMA AND both levels -> failed_breakdown_reclaim

    raw_label is always "directional_confusion_breakout".
    """
    result = normalize_setup(
        "directional_confusion_breakout", direction, strength, confidence, technical_context
    )

    trend = technical_context.ema_trend
    has_both_levels = (
        technical_context.key_levels.get("support") is not None
        and technical_context.key_levels.get("resistance") is not None
    )

    # Always preserves raw_label
    assert result.raw_label == "directional_confusion_breakout"

    if trend == "neutral" or not has_both_levels:
        # Diagnostic only — cannot resolve
        assert result.success is False
        assert result.reason_code == "diagnostic_only"
    elif trend == "bullish" and has_both_levels:
        # Resolves to breakout_retest
        assert result.success is True
        assert result.executable_type == "breakout_retest"
    elif trend == "bearish" and has_both_levels:
        # Resolves to failed_breakdown_reclaim
        assert result.success is True
        assert result.executable_type == "failed_breakdown_reclaim"
