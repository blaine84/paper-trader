"""First-Invalid-Stage Attribution Engine.

Deterministic attribution of geometry defects to the earliest pipeline stage
that introduced them. No LLM judgment involved — purely rule-based traversal
of the provenance chain.

Also provides:
- Gate reconstruction outcome classification
- Full claimed-vs-computed classification using chain context

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 11.2, 11.3, 11.4, 5.7
"""
from __future__ import annotations

from decimal import Decimal

from utils.claimed_vs_computed import CLASSIFICATIONS, ClaimedComputedComparison
from utils.provenance_capture import ProvenanceChain, STAGE_TO_ATTRIBUTION


ATTRIBUTION_CATEGORIES = [
    "trusted_input_invalid",
    "raw_pm_output_invalid",
    "parse_or_normalization_invalid",
    "candidate_resolution_invalid",
    "price_repair_invalid",
    "behavioral_adjustment_invalid",
    "pre_gate_contract_invalid",
    "gate_reconstruction_invalid",
    "policy_rejection_of_valid_contract",
    "incomplete_provenance",
    "unknown",
]


def attribute_first_invalid_stage(chain: ProvenanceChain) -> str:
    """Traverse provenance chain and return the attribution category.

    Rules:
    - Walks events in sequence order (earliest first)
    - Skips terminal events (is_terminal=True) and not_applicable stages
    - Returns the attribution category for the first stage whose
      validation_after == "invalid"
    - If all stages valid but candidate rejected by gate policy:
      returns "policy_rejection_of_valid_contract"
    - If chain is missing expected stages (accounting for mode and terminals):
      returns "incomplete_provenance"
    - No LLM judgment involved — purely deterministic

    This function delegates to chain.get_attribution_category() which already
    implements the traversal logic. Provided as a standalone function for
    external callers.
    """
    return chain.get_attribution_category()


def classify_reconstruction_outcome(
    pre_reconstruction_valid: bool,
    post_reconstruction_valid: bool,
    rr_before: Decimal,
    rr_after: Decimal,
) -> str:
    """Classify gate reconstruction outcome.

    Returns one of:
    - "valid_geometry_preserved" (valid -> valid, RR not degraded)
    - "valid_geometry_degraded" (valid -> valid, RR decreased)
    - "invalid_geometry_rejected" (invalid -> invalid/rejected)
    - "invalid_geometry_repaired" (invalid -> valid)
    - "reconstruction_introduced_defect" (valid -> invalid)
    """
    if pre_reconstruction_valid and post_reconstruction_valid:
        if rr_after >= rr_before:
            return "valid_geometry_preserved"
        return "valid_geometry_degraded"

    if not pre_reconstruction_valid and not post_reconstruction_valid:
        return "invalid_geometry_rejected"

    if not pre_reconstruction_valid and post_reconstruction_valid:
        return "invalid_geometry_repaired"

    # pre_reconstruction_valid=True and post_reconstruction_valid=False
    return "reconstruction_introduced_defect"


def classify_claimed_vs_computed_full(
    comparison: ClaimedComputedComparison,
    chain: ProvenanceChain,
) -> str:
    """Full classification using chain context.

    This function handles "correct_claim_invalidated_by_mutation" which
    requires checking whether the claim was valid at the time it was made
    but later stages degraded the geometry.

    Returns one of CLASSIFICATIONS:
    - "claimed_reward_risk_not_stated"
    - "unverifiable_categorical_claim"
    - "correct_claim_invalidated_by_mutation"
    - "incorrect_narrative_valid_geometry"
    - "invalid_geometry_from_pm_response"
    - "correct_narrative_correct_geometry"
    """
    # Case 1: No claim found at all
    if comparison.claim_absent:
        return "claimed_reward_risk_not_stated"

    # Case 2: Categorical/qualitative claim without numeric value
    if comparison.is_categorical:
        return "unverifiable_categorical_claim"

    # Case 3: Numeric mismatch present
    if comparison.is_numeric_mismatch:
        # Check if the claim was valid at the PM stage but later stages
        # degraded the geometry (correct_claim_invalidated_by_mutation)
        if _claim_was_valid_then_degraded(chain):
            return "correct_claim_invalidated_by_mutation"

        # Check if overall geometry is still valid
        if _final_geometry_valid(chain):
            return "incorrect_narrative_valid_geometry"

        # Geometry is invalid — attribute to PM response
        return "invalid_geometry_from_pm_response"

    # Case 4: Claim matches computed (no mismatch)
    if _final_geometry_valid(chain):
        return "correct_narrative_correct_geometry"

    return "invalid_geometry_from_pm_response"


def _claim_was_valid_then_degraded(chain: ProvenanceChain) -> bool:
    """Check if claim was valid at early PM stages but later stages degraded geometry.

    Looks at early events (raw_pm_output, parsed_pm_decision) — if their
    geometry was valid, then checks if any later event has geometry going invalid.
    This indicates the PM's original claim was correct but downstream mutations
    invalidated it.
    """
    early_stages = {"raw_pm_output", "parsed_pm_decision"}
    sorted_events = sorted(chain.events, key=lambda e: e.sequence_number)

    # Check if early stages had valid geometry
    early_valid = False
    for event in sorted_events:
        if event.is_terminal:
            continue
        if event.stage_name in early_stages:
            if event.validation_after == "valid":
                early_valid = True
                break

    if not early_valid:
        return False

    # Check if any later stage (after early stages) has geometry going invalid
    past_early = False
    for event in sorted_events:
        if event.is_terminal:
            continue
        if event.stage_name in early_stages:
            past_early = True
            continue
        if past_early and event.validation_after == "invalid":
            return True

    return False


def _final_geometry_valid(chain: ProvenanceChain) -> bool:
    """Check if the final (latest non-terminal) event has valid geometry.

    Traverses events in reverse sequence order to find the last non-terminal
    event and checks its validation_after status.
    """
    sorted_events = sorted(
        chain.events, key=lambda e: e.sequence_number, reverse=True,
    )

    for event in sorted_events:
        if event.is_terminal:
            continue
        if event.validation_after == "not_applicable":
            continue
        return event.validation_after == "valid"

    # No non-terminal events found — treat as invalid
    return False
