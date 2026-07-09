"""Property-based tests for rejection mapping (Properties 5, 6, 20).

Tests that:
- Property 5: Any non-None final_rejection_reason is in CANONICAL_REJECTION_CODES
- Property 6: map_rejection_reason always returns canonical_code ∈ CANONICAL_REJECTION_CODES
           and raw_reason == input string
- Property 20: raw_reason is always the exact input string (mapping layer preserves raw fidelity)

Also includes a deterministic check that all values in _RAW_TO_CANONICAL are in
CANONICAL_REJECTION_CODES.
"""

from __future__ import annotations

from hypothesis import given, settings, strategies as st

from utils.swing_candidate_bridge import (
    CANONICAL_REJECTION_CODES,
    RejectionMapping,
    map_rejection_reason,
    _RAW_TO_CANONICAL,
)


# ---------------------------------------------------------------------------
# Strategy: arbitrary symbol strings for the symbol parameter
# ---------------------------------------------------------------------------

symbol_st = st.text(min_size=1, max_size=5, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# ---------------------------------------------------------------------------
# Property 5: Canonical Rejection Code Membership
# Validates: Requirements 3.1, 3.3, 2.1
#
# For any arbitrary string input, map_rejection_reason produces a canonical_code
# that is always a member of the frozen CANONICAL_REJECTION_CODES set.
# This guarantees that no non-None final_rejection_reason can escape the closed set.
# ---------------------------------------------------------------------------


@given(raw_reason=st.text(), symbol=symbol_st)
@settings(max_examples=200)
def test_canonical_rejection_code_membership(raw_reason: str, symbol: str):
    """Property 5: Any canonical_code from map_rejection_reason is in CANONICAL_REJECTION_CODES.

    **Validates: Requirements 3.1, 3.3, 2.1**
    """
    result = map_rejection_reason(raw_reason, symbol)
    assert result.canonical_code in CANONICAL_REJECTION_CODES, (
        f"canonical_code {result.canonical_code!r} not in CANONICAL_REJECTION_CODES"
    )


# ---------------------------------------------------------------------------
# Property 6: Rejection Mapping Consistency
# Validates: Requirements 3.2, 3.5
#
# map_rejection_reason always returns a RejectionMapping where:
# - canonical_code ∈ CANONICAL_REJECTION_CODES
# - raw_reason == input string
# ---------------------------------------------------------------------------


@given(raw_reason=st.text(), symbol=symbol_st)
@settings(max_examples=200)
def test_rejection_mapping_consistency(raw_reason: str, symbol: str):
    """Property 6: map_rejection_reason returns consistent RejectionMapping.

    The returned object always has:
    - canonical_code that is a member of the canonical set
    - raw_reason that exactly equals the input string

    **Validates: Requirements 3.2, 3.5**
    """
    result = map_rejection_reason(raw_reason, symbol)

    # Must be a RejectionMapping instance
    assert isinstance(result, RejectionMapping)

    # canonical_code must be in the closed set
    assert result.canonical_code in CANONICAL_REJECTION_CODES

    # raw_reason must exactly equal the input
    assert result.raw_reason == raw_reason


# ---------------------------------------------------------------------------
# Property 20: Mapping Layer Preserves Raw Fidelity
# Validates: Requirements 17.1, 3.5
#
# raw_reason is always the exact input string — no transformation, truncation,
# or normalization is applied to the raw reason.
# ---------------------------------------------------------------------------


@given(raw_reason=st.text(), symbol=symbol_st)
@settings(max_examples=200)
def test_mapping_layer_preserves_raw_fidelity(raw_reason: str, symbol: str):
    """Property 20: raw_reason is always the exact input string.

    The mapping layer MUST preserve the original rejection reason string without
    any modification, ensuring full audit fidelity for reviewers.

    **Validates: Requirements 17.1, 3.5**
    """
    result = map_rejection_reason(raw_reason, symbol)
    assert result.raw_reason is raw_reason or result.raw_reason == raw_reason
    # Verify exact string equality (not just equivalence) — same length, same chars
    assert len(result.raw_reason) == len(raw_reason)
    assert result.raw_reason == raw_reason


# ---------------------------------------------------------------------------
# Deterministic: All _RAW_TO_CANONICAL values are in CANONICAL_REJECTION_CODES
# Validates: Requirements 3.1, 3.3
#
# This is not a property test but a structural invariant check that ensures
# the mapping dict only targets valid canonical codes.
# ---------------------------------------------------------------------------


def test_raw_to_canonical_values_are_all_canonical():
    """All mapped values in _RAW_TO_CANONICAL must be in CANONICAL_REJECTION_CODES.

    This ensures no mapping can produce a code outside the closed canonical set.

    **Validates: Requirements 3.1, 3.3**
    """
    for raw_key, canonical_value in _RAW_TO_CANONICAL.items():
        assert canonical_value in CANONICAL_REJECTION_CODES, (
            f"_RAW_TO_CANONICAL[{raw_key!r}] = {canonical_value!r} "
            f"is not in CANONICAL_REJECTION_CODES"
        )


def test_canonical_rejection_codes_is_frozen():
    """CANONICAL_REJECTION_CODES must be a frozenset (immutable).

    **Validates: Requirements 3.3**
    """
    assert isinstance(CANONICAL_REJECTION_CODES, frozenset)


def test_known_raw_codes_map_to_expected_canonical():
    """Known raw codes from the mapping produce expected canonical codes.

    Verifies identity mappings and legacy-to-canonical mappings work correctly.

    **Validates: Requirements 3.2**
    """
    # Identity mappings
    for code in CANONICAL_REJECTION_CODES:
        if code in _RAW_TO_CANONICAL:
            result = map_rejection_reason(code, "TEST")
            assert result.canonical_code == _RAW_TO_CANONICAL[code]
            assert result.raw_reason == code

    # Legacy mappings
    legacy_cases = {
        "error_setup_blocked": "data_provider_error",
        "data_provider_error_blocked": "data_provider_error",
        "observe_only_period": "profile_policy",
        "confidence_below_threshold": "profile_policy",
        "strength_below_threshold": "profile_policy",
        "rr_below_threshold": "profile_policy",
        "same_symbol_overlap_blocked": "same_symbol_exposure",
        "max_swing_positions_reached": "profile_policy",
        "sizing_rejected": "failed_risk_gates",
    }
    for raw, expected_canonical in legacy_cases.items():
        result = map_rejection_reason(raw, "TEST")
        assert result.canonical_code == expected_canonical, (
            f"Expected {raw!r} → {expected_canonical!r}, got {result.canonical_code!r}"
        )
        assert result.raw_reason == raw


def test_unknown_raw_reason_maps_to_unknown_error():
    """Unrecognized raw reasons always map to 'unknown_error'.

    **Validates: Requirements 3.2, 3.5**
    """
    result = map_rejection_reason("completely_made_up_reason_xyz", "AAPL")
    assert result.canonical_code == "unknown_error"
    assert result.raw_reason == "completely_made_up_reason_xyz"
