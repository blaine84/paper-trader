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


REJECTION_REASON_CODES: frozenset[str] = frozenset({
    "insufficient_normalization_evidence",
    "context_mismatch",
    "diagnostic_only",
    "unmapped_label",
    "unclear_direction",
    "error_setup_blocked",
    "data_provider_error_blocked",
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
        return NormalizationResult(success=False, reason_code="error_setup_blocked")

    # 2. Data provider error check
    if data_source_error or "429" in (error_code or ""):
        return NormalizationResult(success=False, reason_code="data_provider_error_blocked")

    # 3. Analyst veto check
    if llm_veto_reason is not None and llm_veto_reason.strip():
        return NormalizationResult(success=False, reason_code="analyst_veto")

    # 5. Sector rotation mapping
    if raw_label == "sector_rotation":
        return _normalize_sector_rotation(direction, strength, confidence, technical_context)

    # 6. Risk-off macro short mapping
    if raw_label == "risk_off_macro_short":
        return _normalize_risk_off_macro_short(direction, technical_context)

    # 7. Ambiguous directional labels are never executable.
    if raw_label in ("directional_confusion_breakout", "unclear_direction"):
        return NormalizationResult(success=False, reason_code="unclear_direction")

    # 4. Pass-through for labels already in SWING_EXECUTABLE_SETUP_TYPES
    if raw_label in SWING_EXECUTABLE_SETUP_TYPES:
        return NormalizationResult(success=True, executable_type=raw_label)

    # 8. Fallback — unmapped label
    return NormalizationResult(success=False, reason_code="unmapped_label")


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
    """Sector rotation: accept iff direction in {LONG, SHORT}, confidence in {medium, high},
    strength in {moderate, strong}, and (at least one non-null key level OR ema_trend != neutral).
    """
    direction_ok = direction in ("LONG", "SHORT")
    confidence_ok = confidence in ("medium", "high")
    strength_ok = strength in ("moderate", "strong")
    context_ok = (
        _has_any_non_null_key_level(technical_context)
        or technical_context.ema_trend != "neutral"
    )

    if direction_ok and confidence_ok and strength_ok and context_ok:
        return NormalizationResult(success=True, executable_type="sector_rotation_swing")

    return NormalizationResult(success=False, reason_code="insufficient_normalization_evidence")


def _normalize_risk_off_macro_short(
    direction: Literal["LONG", "SHORT", "HOLD"],
    technical_context: TechnicalContext,
) -> NormalizationResult:
    """Risk-off macro short: accept iff direction == SHORT, ema_trend == bearish,
    market_regime == risk_off.
    """
    if (
        direction == "SHORT"
        and technical_context.ema_trend == "bearish"
        and technical_context.market_regime == "risk_off"
    ):
        return NormalizationResult(success=True, executable_type="risk_off_macro_short")

    return NormalizationResult(success=False, reason_code="context_mismatch")

