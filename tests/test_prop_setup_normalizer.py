"""Property-based tests for utils/setup_normalizer.py.

Uses Hypothesis to validate universal correctness properties of the setup
normalizer's pure-function rule engine.
"""

from __future__ import annotations

from hypothesis import given, settings, strategies as st

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
# Property 4: Directional Confusion Labels Are Never Executable
# Validates: Requirements 2.7, 2.8
# ---------------------------------------------------------------------------


@given(
    direction=st.sampled_from(["LONG", "SHORT", "HOLD"]),
    strength=st.sampled_from(["weak", "moderate", "strong"]),
    confidence=st.sampled_from(["low", "medium", "high"]),
    technical_context=technical_context_st,
)
@settings(max_examples=200)
def test_directional_confusion_breakout_never_executes(
    direction: str,
    strength: str,
    confidence: str,
    technical_context: TechnicalContext,
) -> None:
    """**Validates: Requirements 2.7, 2.8**

    For any input where raw_label is "directional_confusion_breakout", the
    normalizer SHALL reject it as "unclear_direction". This label means the
    analyst has no clean directional setup and must not be promoted into a
    tradeable setup by normalization.
    """
    result = normalize_setup(
        "directional_confusion_breakout", direction, strength, confidence, technical_context
    )

    assert result.success is False
    assert result.reason_code == "unclear_direction"
