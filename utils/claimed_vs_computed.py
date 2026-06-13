"""Claimed-versus-Computed Reward-to-Risk Comparison.

Detects mismatches between narrative/claimed reward-to-risk values in PM output
and the deterministic computed reward-to-risk from the Geometry Calculator.

The comparison function determines numeric mismatch. Full classification
(including "correct claim later invalidated by mutation") requires the complete
provenance chain and is performed by the attribution engine, not here.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.6
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Any


MISMATCH_THRESHOLD = Decimal("0.10")  # absolute ratio units


@dataclass(frozen=True)
class ClaimedComputedComparison:
    """Result of comparing claimed vs computed reward-to-risk.

    This performs the NUMERIC comparison only. Full classification
    (e.g., correct_claim_invalidated_by_mutation) requires the
    complete provenance chain and is performed by the attribution engine.
    """
    claimed_value: Decimal | None       # parsed numeric claim, or None
    claimed_phrase: str | None           # original text if categorical/qualitative
    computed_value: Decimal              # from GeometryCalculator
    absolute_difference: Decimal | None  # |claimed - computed| when both numeric
    is_numeric_mismatch: bool           # absolute_difference > MISMATCH_THRESHOLD
    is_categorical: bool                # True when phrase present but no numeric
    claim_absent: bool                  # True when no claim found at all
    source_field: str | None            # field name where claim was found
    source_offset: int | None           # character offset in unstructured text


# Full classification enum — assigned by attribution engine using provenance chain
CLASSIFICATIONS = {
    "correct_narrative_correct_geometry",
    "incorrect_narrative_valid_geometry",
    "correct_claim_invalidated_by_mutation",
    "invalid_geometry_from_pm_response",
    "unverifiable_categorical_claim",
    "claimed_reward_risk_not_stated",
}


# --- Structured field keys to search (in priority order) ---
_STRUCTURED_FIELD_KEYS = [
    "risk_reward",
    "reward_to_risk",
    "rr",
    "reward_risk",
    "r_r",
]


# --- Unstructured text patterns ---
# Pattern: "X.X:1" or "X:1" (e.g., "2.5:1", "3:1")
_RATIO_COLON_ONE = re.compile(
    r"(\d+(?:\.\d+)?)\s*:\s*1(?!\d)",
)

# Pattern: "R:R of X.X" or "R/R of X.X"
_RR_OF_PATTERN = re.compile(
    r"[Rr][:/][Rr]\s+of\s+(\d+(?:\.\d+)?)",
)

# Pattern: "reward-to-risk of X.X" or "reward to risk X.X"
_REWARD_TO_RISK_PATTERN = re.compile(
    r"reward[\s\-]+to[\s\-]+risk\s+(?:of\s+)?(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Pattern: "risk/reward X.X" or "risk-reward X.X"
_RISK_REWARD_PATTERN = re.compile(
    r"risk[/\-\s]+reward\s+(?:of\s+)?(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Categorical phrases that indicate a qualitative claim without a numeric value
_CATEGORICAL_PATTERNS = re.compile(
    r"(?:favorable|strong|clear|good|excellent|poor|weak|marginal|attractive|solid)"
    r"\s+(?:risk[/\-\s]*reward|reward[/\-\s]*risk|[Rr][:/][Rr]|R:R)",
    re.IGNORECASE,
)

# Reverse form: "risk/reward is favorable" etc.
_CATEGORICAL_PATTERNS_REVERSE = re.compile(
    r"(?:risk[/\-\s]*reward|reward[/\-\s]*risk|[Rr][:/][Rr]|R:R)"
    r"\s+(?:is\s+|looks\s+)?(?:favorable|strong|clear|good|excellent|poor|weak|marginal|attractive|solid)",
    re.IGNORECASE,
)


def _try_parse_decimal(value: Any) -> Decimal | None:
    """Attempt to parse a value as a Decimal. Returns None on failure."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        d = Decimal(str(value))
        if d.is_finite():
            return d
        return None
    except (InvalidOperation, ValueError, TypeError):
        return None


def _search_structured_fields(
    pm_output: dict,
) -> tuple[Decimal | None, str | None, str | None]:
    """Search structured fields for a numeric R:R claim.

    Returns (numeric_value, raw_phrase, source_field_name).
    """
    for key in _STRUCTURED_FIELD_KEYS:
        if key in pm_output:
            raw_value = pm_output[key]
            numeric = _try_parse_decimal(raw_value)
            if numeric is not None:
                return numeric, str(raw_value), key
            # Value exists but isn't parseable as numeric — treat as phrase
            if raw_value is not None and str(raw_value).strip():
                return None, str(raw_value).strip(), key
    return None, None, None


def _search_unstructured_text(
    text: str,
) -> tuple[Decimal | None, str | None, int | None]:
    """Search unstructured rationale text for R:R patterns.

    Returns (numeric_value, raw_phrase, character_offset).
    """
    # Try numeric patterns first (in priority order)
    numeric_patterns = [
        _RATIO_COLON_ONE,
        _RR_OF_PATTERN,
        _REWARD_TO_RISK_PATTERN,
        _RISK_REWARD_PATTERN,
    ]

    for pattern in numeric_patterns:
        match = pattern.search(text)
        if match:
            numeric = _try_parse_decimal(match.group(1))
            if numeric is not None:
                return numeric, match.group(0), match.start()

    # Try categorical patterns
    for cat_pattern in [_CATEGORICAL_PATTERNS, _CATEGORICAL_PATTERNS_REVERSE]:
        match = cat_pattern.search(text)
        if match:
            return None, match.group(0), match.start()

    return None, None, None


def extract_claimed_reward_risk(
    pm_output: dict,
    rationale_text: str | None = None,
) -> tuple[Decimal | None, str | None]:
    """Extract numeric or categorical R:R claim from PM output.

    Searches structured fields first (e.g., "risk_reward", "reward_to_risk"),
    then unstructured rationale text for patterns like "2.0:1", "R:R of 2.0".

    Returns (numeric_value, raw_phrase) — one or both may be None.
    If no claim is found anywhere, returns (None, None).
    """
    if not isinstance(pm_output, dict):
        pm_output = {}

    # Step 1: Search structured fields
    numeric, phrase, _field = _search_structured_fields(pm_output)
    if numeric is not None or phrase is not None:
        return numeric, phrase

    # Step 2: Search unstructured text
    if rationale_text and isinstance(rationale_text, str):
        numeric, phrase, _offset = _search_unstructured_text(rationale_text)
        if numeric is not None or phrase is not None:
            return numeric, phrase

    return None, None


def extract_claimed_reward_risk_detailed(
    pm_output: dict,
    rationale_text: str | None = None,
) -> tuple[Decimal | None, str | None, str | None, int | None]:
    """Extract R:R claim with source location details.

    Like extract_claimed_reward_risk but also returns:
    - source_field: the field name if found in structured fields
    - source_offset: character offset if found in unstructured text

    Returns (numeric_value, raw_phrase, source_field, source_offset).
    """
    if not isinstance(pm_output, dict):
        pm_output = {}

    # Step 1: Search structured fields
    numeric, phrase, field = _search_structured_fields(pm_output)
    if numeric is not None or phrase is not None:
        return numeric, phrase, field, None

    # Step 2: Search unstructured text
    if rationale_text and isinstance(rationale_text, str):
        numeric, phrase, offset = _search_unstructured_text(rationale_text)
        if numeric is not None or phrase is not None:
            return numeric, phrase, None, offset

    return None, None, None, None


def compare_claimed_vs_computed(
    claimed_value: Decimal | None,
    claimed_phrase: str | None,
    computed_rr: Decimal,
) -> ClaimedComputedComparison:
    """Compare claimed and computed reward-to-risk (numeric comparison only).

    Applies the 0.10 absolute-difference threshold for mismatch detection.
    Does NOT perform full classification — that requires chain context.

    Cases:
    1. claimed_value is None AND claimed_phrase is None → claim_absent=True
    2. claimed_value is None AND claimed_phrase is not None → is_categorical=True
    3. claimed_value is not None → compute absolute_difference, check threshold
    """
    # Case 1: No claim found at all
    if claimed_value is None and claimed_phrase is None:
        return ClaimedComputedComparison(
            claimed_value=None,
            claimed_phrase=None,
            computed_value=computed_rr,
            absolute_difference=None,
            is_numeric_mismatch=False,
            is_categorical=False,
            claim_absent=True,
            source_field=None,
            source_offset=None,
        )

    # Case 2: Categorical claim only (phrase but no numeric)
    if claimed_value is None and claimed_phrase is not None:
        return ClaimedComputedComparison(
            claimed_value=None,
            claimed_phrase=claimed_phrase,
            computed_value=computed_rr,
            absolute_difference=None,
            is_numeric_mismatch=False,
            is_categorical=True,
            claim_absent=False,
            source_field=None,
            source_offset=None,
        )

    # Case 3: Numeric claim — compute absolute difference
    abs_diff = abs(claimed_value - computed_rr)
    is_mismatch = abs_diff > MISMATCH_THRESHOLD

    return ClaimedComputedComparison(
        claimed_value=claimed_value,
        claimed_phrase=claimed_phrase,
        computed_value=computed_rr,
        absolute_difference=abs_diff,
        is_numeric_mismatch=is_mismatch,
        is_categorical=False,
        claim_absent=False,
        source_field=None,
        source_offset=None,
    )
