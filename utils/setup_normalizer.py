from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from utils.gate_config import SWING_EXECUTABLE_SETUP_TYPES


@dataclass(frozen=True)
class TechnicalContext:
    """Immutable technical context from analyst signal."""

    key_levels: dict[str, float | None]  # {"support": float|None, "resistance": float|None}
    ema_trend: Literal["bullish", "bearish", "neutral"]
    market_regime: Literal["risk_on", "risk_off", "mixed"]


@dataclass(frozen=True)
class NormalizationResult:
    """Result of setup normalization — either success or rejection."""

    success: bool
    executable_type: str | None = None  # Present only when success=True
    reason_code: str | None = None  # Present only when success=False
    raw_label: str | None = None  # Preserved raw analyst label (Req 5.2)
    missing_evidence: list[str] | None = None  # Which fields were missing (Req 6.3, 9.3)


REJECTION_REASON_CODES: frozenset[str] = frozenset({
    "insufficient_normalization_evidence",
    "context_mismatch",
    "diagnostic_only",
    "unmapped_label",
    "data_provider_error",
    "analyst_veto",
})


def normalize_setup(
    raw_label: str,
    direction: Literal["LONG", "SHORT", "HOLD"],
    strength: Literal["weak", "moderate", "strong"],
    confidence: Literal["low", "medium", "high"],
    technical_context: TechnicalContext,
    *,
    llm_veto_reason: str | None = None,
    data_source_error: bool = False,
    error_code: str | None = None,
) -> NormalizationResult:
    """Deterministic, pure normalization of analyst setup label.

    Returns NormalizationResult with exactly one of:
    - success=True, executable_type=<string in SWING_EXECUTABLE_SETUP_TYPES>
    - success=False, reason_code=<string in REJECTION_REASON_CODES>

    No side effects, no database access, no network calls.
    """
    # 1. Error label check
    if raw_label == "error":
        return NormalizationResult(
            success=False, reason_code="data_provider_error", raw_label=raw_label
        )

    # 2. Data provider error check
    if data_source_error or "429" in (error_code or ""):
        return NormalizationResult(
            success=False, reason_code="data_provider_error", raw_label=raw_label
        )

    # 3. Analyst veto check
    if llm_veto_reason is not None and llm_veto_reason.strip():
        return NormalizationResult(
            success=False, reason_code="analyst_veto", raw_label=raw_label
        )

    # 5. Sector rotation mapping
    if raw_label == "sector_rotation":
        return _normalize_sector_rotation(direction, strength, confidence, technical_context)

    # 6. Risk-off macro short mapping
    if raw_label == "risk_off_macro_short":
        return _normalize_risk_off_macro_short(direction, technical_context)

    # 7. Directional confusion breakout mapping
    if raw_label == "directional_confusion_breakout":
        return _normalize_directional_confusion_breakout(technical_context)

    # 4. Pass-through for labels already in SWING_EXECUTABLE_SETUP_TYPES
    if raw_label in SWING_EXECUTABLE_SETUP_TYPES:
        return NormalizationResult(success=True, executable_type=raw_label, raw_label=raw_label)

    # 8. Fallback — unmapped label
    return NormalizationResult(
        success=False, reason_code="unmapped_label", raw_label=raw_label
    )


def _has_any_non_null_key_level(technical_context: TechnicalContext) -> bool:
    """Return True if key_levels has at least one non-null numeric value."""
    for value in technical_context.key_levels.values():
        if value is not None:
            return True
    return False


def _has_both_support_and_resistance(technical_context: TechnicalContext) -> bool:
    """Return True if key_levels has both non-null support AND resistance."""
    support = technical_context.key_levels.get("support")
    resistance = technical_context.key_levels.get("resistance")
    return support is not None and resistance is not None


def _normalize_sector_rotation(
    direction: Literal["LONG", "SHORT", "HOLD"],
    strength: Literal["weak", "moderate", "strong"],
    confidence: Literal["low", "medium", "high"],
    technical_context: TechnicalContext,
) -> NormalizationResult:
    """Sector rotation: reports missing evidence fields on rejection.

    Accepts iff direction in {LONG, SHORT}, confidence in {medium, high},
    strength in {moderate, strong}, and (at least one non-null key level OR
    ema_trend != neutral). On rejection, reports which evidence fields were
    missing via missing_evidence list.
    """
    missing: list[str] = []
    if direction not in ("LONG", "SHORT"):
        missing.append("direction_not_directional")
    if confidence not in ("medium", "high"):
        missing.append("confidence_below_medium")
    if strength not in ("moderate", "strong"):
        missing.append("strength_below_moderate")

    context_ok = (
        _has_any_non_null_key_level(technical_context)
        or technical_context.ema_trend != "neutral"
    )
    if not context_ok:
        missing.append("no_key_levels_and_neutral_ema")

    if not missing:
        return NormalizationResult(
            success=True,
            executable_type="sector_rotation_swing",
            raw_label="sector_rotation",
        )

    return NormalizationResult(
        success=False,
        reason_code="insufficient_normalization_evidence",
        raw_label="sector_rotation",
        missing_evidence=missing,
    )


def _normalize_risk_off_macro_short(
    direction: Literal["LONG", "SHORT", "HOLD"],
    technical_context: TechnicalContext,
) -> NormalizationResult:
    """Risk-off macro short: accept iff direction == SHORT, ema_trend == bearish,
    market_regime == risk_off. Reports context mismatch with missing_evidence on failure.
    """
    missing: list[str] = []
    if direction != "SHORT":
        missing.append("direction_not_short")
    if technical_context.ema_trend != "bearish":
        missing.append("ema_trend_not_bearish")
    if technical_context.market_regime != "risk_off":
        missing.append("market_regime_not_risk_off")

    if not missing:
        return NormalizationResult(
            success=True,
            executable_type="risk_off_macro_short",
            raw_label="risk_off_macro_short",
        )

    return NormalizationResult(
        success=False,
        reason_code="context_mismatch",
        raw_label="risk_off_macro_short",
        missing_evidence=missing,
    )


def _normalize_directional_confusion_breakout(
    technical_context: TechnicalContext,
) -> NormalizationResult:
    """Directional confusion breakout: resolve or reject based on evidence.

    - If ema_trend is neutral OR key_levels lacks both support and resistance
      → reject with "diagnostic_only"
    - If ema_trend is bullish AND both support and resistance present
      → normalize to "breakout_retest"
    - If ema_trend is bearish AND both support and resistance present
      → normalize to "failed_breakdown_reclaim"

    Preserves raw_label="directional_confusion_breakout" in all return paths.
    """
    trend = technical_context.ema_trend
    has_both_levels = _has_both_support_and_resistance(technical_context)

    # Reject: ema_trend neutral OR key_levels lacks both support and resistance
    if trend == "neutral" or not has_both_levels:
        return NormalizationResult(
            success=False,
            reason_code="diagnostic_only",
            raw_label="directional_confusion_breakout",
        )

    # Resolve: bullish with both levels → breakout_retest
    if trend == "bullish":
        return NormalizationResult(
            success=True,
            executable_type="breakout_retest",
            raw_label="directional_confusion_breakout",
        )

    # Resolve: bearish with both levels → failed_breakdown_reclaim
    return NormalizationResult(
        success=True,
        executable_type="failed_breakdown_reclaim",
        raw_label="directional_confusion_breakout",
    )
