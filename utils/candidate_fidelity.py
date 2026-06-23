"""Candidate/Scaffold Fidelity Check.

Compares PM-returned fields against trusted candidate/scaffold values to detect
unsupported reconstruction, permitted adjustments, or identity mismatches.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


# Fidelity tolerance: 5% RELATIVE difference for numeric price fields
FIDELITY_TOLERANCE_PCT = Decimal("0.05")  # 5% relative

# Fields requiring exact match (categorical)
EXACT_MATCH_FIELDS = ("symbol", "direction", "setup_type")

# Fields using relative numeric tolerance
NUMERIC_TOLERANCE_FIELDS = ("entry_price", "stop_price", "target_price")


@dataclass(frozen=True)
class FieldDifference:
    """A single field difference between PM output and trusted candidate."""
    field_name: str
    trusted_value: str
    pm_value: str
    classification: str               # per-field classification
    tolerance_pct: Decimal | None     # relative % difference (for numeric)
    tolerance_exceeded: bool


@dataclass(frozen=True)
class FidelityResult:
    """Result of comparing PM decision against trusted candidate fields."""
    candidate_id_matched: bool
    classification: str               # overall classification
    field_differences: list[FieldDifference]


def _to_decimal(value) -> Decimal | None:
    """Convert a value to Decimal, returning None if conversion fails."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _compute_relative_pct(pm_value: Decimal, trusted_value: Decimal) -> Decimal | None:
    """Compute relative percentage difference: |pm - trusted| / trusted.

    Returns None if trusted_value is zero (cannot compute relative difference).
    """
    if trusted_value == 0:
        return None
    return abs(pm_value - trusted_value) / abs(trusted_value)


def check_candidate_fidelity(
    pm_decision: dict,
    trusted_candidate: dict,
    supplied_candidate_ids: list[str],
    numeric_tolerance_pct: Decimal = FIDELITY_TOLERANCE_PCT,
) -> FidelityResult:
    """Compare PM-returned fields against trusted candidate/scaffold.

    Exact match for: symbol, direction, setup_type
    Relative percentage tolerance for: entry_price, stop_price, target_price
    (tolerance_pct = |pm_value - trusted_value| / trusted_value)

    Classifications per field:
    - permitted_selection: PM chose a different supplied candidate
    - permitted_bounded_adjustment: numeric deviation within tolerance
    - unsupported_reconstruction: numeric deviation exceeds tolerance or categorical mismatch
    - missing_candidate_identity: no candidate_id in PM output
    - candidate_mismatch: candidate_id not in supplied set
    """
    # Step 1: Check candidate_id presence and matching
    pm_candidate_id = pm_decision.get("candidate_id")

    if pm_candidate_id is None:
        return FidelityResult(
            candidate_id_matched=False,
            classification="missing_candidate_identity",
            field_differences=[],
        )

    if pm_candidate_id not in supplied_candidate_ids:
        return FidelityResult(
            candidate_id_matched=False,
            classification="candidate_mismatch",
            field_differences=[],
        )

    # Step 2: candidate_id is in the supplied set — compare fields
    field_differences: list[FieldDifference] = []

    # Check exact match fields (symbol, direction, setup_type)
    for field_name in EXACT_MATCH_FIELDS:
        pm_val = pm_decision.get(field_name)
        trusted_val = trusted_candidate.get(field_name)

        if pm_val is None and trusted_val is None:
            continue
        if str(pm_val) != str(trusted_val):
            field_differences.append(FieldDifference(
                field_name=field_name,
                trusted_value=str(trusted_val) if trusted_val is not None else "",
                pm_value=str(pm_val) if pm_val is not None else "",
                classification="unsupported_reconstruction",
                tolerance_pct=None,
                tolerance_exceeded=True,
            ))

    # Check numeric tolerance fields (entry_price, stop_price, target_price)
    for field_name in NUMERIC_TOLERANCE_FIELDS:
        pm_raw = pm_decision.get(field_name)
        trusted_raw = trusted_candidate.get(field_name)

        pm_val = _to_decimal(pm_raw)
        trusted_val = _to_decimal(trusted_raw)

        # Skip if both are None or not convertible
        if pm_val is None and trusted_val is None:
            continue

        # If one is None but not the other, treat as unsupported reconstruction
        if pm_val is None or trusted_val is None:
            field_differences.append(FieldDifference(
                field_name=field_name,
                trusted_value=str(trusted_raw) if trusted_raw is not None else "",
                pm_value=str(pm_raw) if pm_raw is not None else "",
                classification="unsupported_reconstruction",
                tolerance_pct=None,
                tolerance_exceeded=True,
            ))
            continue

        # Both are valid Decimals — compare
        if pm_val == trusted_val:
            continue

        # Compute relative percentage difference
        if trusted_val == 0:
            # Cannot compute relative difference when trusted is 0
            # Any non-zero PM value is unsupported reconstruction
            field_differences.append(FieldDifference(
                field_name=field_name,
                trusted_value=str(trusted_val),
                pm_value=str(pm_val),
                classification="unsupported_reconstruction",
                tolerance_pct=None,
                tolerance_exceeded=True,
            ))
            continue

        relative_pct = _compute_relative_pct(pm_val, trusted_val)

        if relative_pct is not None and relative_pct <= numeric_tolerance_pct:
            # Within tolerance
            field_differences.append(FieldDifference(
                field_name=field_name,
                trusted_value=str(trusted_val),
                pm_value=str(pm_val),
                classification="permitted_bounded_adjustment",
                tolerance_pct=relative_pct,
                tolerance_exceeded=False,
            ))
        else:
            # Exceeds tolerance
            field_differences.append(FieldDifference(
                field_name=field_name,
                trusted_value=str(trusted_val),
                pm_value=str(pm_val),
                classification="unsupported_reconstruction",
                tolerance_pct=relative_pct,
                tolerance_exceeded=True,
            ))

    # Step 3: Determine overall classification
    if not field_differences:
        # No differences at all — PM selected and faithfully reproduced the candidate
        return FidelityResult(
            candidate_id_matched=True,
            classification="permitted_selection",
            field_differences=[],
        )

    # Check if any field is unsupported_reconstruction
    has_unsupported = any(
        fd.classification == "unsupported_reconstruction"
        for fd in field_differences
    )

    if has_unsupported:
        overall_classification = "unsupported_reconstruction"
    else:
        # All differences are permitted_bounded_adjustment
        overall_classification = "permitted_bounded_adjustment"

    return FidelityResult(
        candidate_id_matched=True,
        classification=overall_classification,
        field_differences=field_differences,
    )
