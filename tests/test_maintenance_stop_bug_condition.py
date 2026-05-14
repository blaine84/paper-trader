"""
Bug Condition Exploration Test — Invalid Maintenance Stop Proposals Are Not Suppressed.

**Validates: Requirements 1.1, 1.2, 1.3, 1.5**

This test encodes the EXPECTED (correct) behavior. It is designed to FAIL on
unfixed code, confirming the bug exists. After the fix is applied, these tests
should PASS, confirming the bug is resolved.

Bug conditions tested:
- Non-monotonic proposals: LONG new_stop <= old_stop, SHORT new_stop >= old_stop
- No-op proposals: new_stop == old_stop for any side
- Invalid values: null, zero, negative, non-numeric new_stop values

On UNFIXED code these tests are EXPECTED TO FAIL because:
- No `should_suppress_maintenance_stop` function exists
- Proposals go directly to `apply_stop_update()` logging noisy event pairs
"""

import pytest
from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Concrete bug condition cases from the specification
# ---------------------------------------------------------------------------

CONCRETE_BUG_CASES = [
    # (side, old_stop, new_stop_raw, expected_reason)
    ("long", 56.95, 56.90, "non_monotonic_or_noop"),   # LONG non-monotonic
    ("long", 56.95, 56.95, "non_monotonic_or_noop"),   # LONG no-op
    ("short", 457.00, 460.00, "non_monotonic_or_noop"),  # SHORT non-monotonic
    ("long", 56.95, None, "invalid_stop_value"),        # invalid: null
    ("long", 56.95, 0, "invalid_stop_value"),           # invalid: zero
    ("long", 56.95, -1.0, "invalid_stop_value"),        # invalid: negative
    ("long", 56.95, "abc", "invalid_stop_value"),       # invalid: non-numeric
    ("long", 56.95, "NaN", "invalid_stop_value"),       # invalid: NaN string
    ("long", 56.95, float("inf"), "invalid_stop_value"),  # invalid: +inf
    ("long", 56.95, float("-inf"), "invalid_stop_value"),  # invalid: -inf
    ("short", 457.00, float("nan"), "invalid_stop_value"),  # invalid: NaN float
]


# ---------------------------------------------------------------------------
# Hypothesis strategies for random bug-condition inputs
# ---------------------------------------------------------------------------

def long_non_monotonic_strategy():
    """Generate LONG proposals where new_stop <= old_stop (non-monotonic/no-op)."""
    return st.tuples(
        st.just("long"),
        st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
    ).flatmap(
        lambda t: st.tuples(
            st.just(t[0]),
            st.just(t[1]),
            # new_stop <= old_stop for LONG is non-monotonic
            st.floats(min_value=0.01, max_value=t[1], allow_nan=False, allow_infinity=False),
            st.just("non_monotonic_or_noop"),
        )
    )


def short_non_monotonic_strategy():
    """Generate SHORT proposals where new_stop >= old_stop (non-monotonic/no-op)."""
    return st.tuples(
        st.just("short"),
        st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
    ).flatmap(
        lambda t: st.tuples(
            st.just(t[0]),
            st.just(t[1]),
            # new_stop >= old_stop for SHORT is non-monotonic
            st.floats(min_value=t[1], max_value=t[1] + 10000.0, allow_nan=False, allow_infinity=False),
            st.just("non_monotonic_or_noop"),
        )
    )


def invalid_value_strategy():
    """Generate proposals with invalid new_stop values (None, 0, negative, non-numeric)."""
    invalid_values = st.one_of(
        st.none(),
        st.just(0),
        st.just(0.0),
        st.floats(max_value=-0.01, allow_nan=False, allow_infinity=False),
        st.text(min_size=1, max_size=10).filter(lambda s: not _is_positive_numeric(s)),
    )
    sides = st.sampled_from(["long", "short"])
    old_stops = st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False)

    return st.tuples(sides, old_stops, invalid_values, st.just("invalid_stop_value"))


def _is_positive_numeric(s: str) -> bool:
    """Check if a string represents a positive number."""
    try:
        return float(s) > 0
    except (TypeError, ValueError):
        return False


# Combined strategy for all bug-condition inputs
bug_condition_strategy = st.one_of(
    long_non_monotonic_strategy(),
    short_non_monotonic_strategy(),
    invalid_value_strategy(),
)


# ---------------------------------------------------------------------------
# Property Test: Bug Condition — Invalid Proposals Are Suppressed
# ---------------------------------------------------------------------------

@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    deadline=None,
)
@given(case=bug_condition_strategy)
def test_bug_condition_random_inputs_are_suppressed(case):
    """
    **Validates: Requirements 1.1, 1.2, 1.3, 1.5**

    Property 1: Bug Condition — Invalid Maintenance Stop Proposals Are Not Suppressed

    For any tighten_stop proposal where old_stop is valid, side is recognized,
    and new_stop_raw is non-monotonic, no-op, or invalid, the suppressor SHALL
    return (True, reason) with the correct reason.

    On UNFIXED code, this will fail with:
      ImportError: cannot import name 'should_suppress_maintenance_stop'
      (confirming the suppression layer does not exist)
    """
    from agents.portfolio_manager import should_suppress_maintenance_stop

    side, old_stop, new_stop_raw, expected_reason = case

    result = should_suppress_maintenance_stop(side, old_stop, new_stop_raw)

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"

    should_suppress, reason = result
    assert should_suppress is True, (
        f"Expected suppression for side={side}, old_stop={old_stop}, "
        f"new_stop_raw={new_stop_raw!r}, but got should_suppress=False"
    )
    assert reason == expected_reason, (
        f"Expected reason='{expected_reason}' for side={side}, old_stop={old_stop}, "
        f"new_stop_raw={new_stop_raw!r}, but got reason='{reason}'"
    )


# ---------------------------------------------------------------------------
# Parametrized Test: Concrete bug condition cases from specification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "side,old_stop,new_stop_raw,expected_reason",
    CONCRETE_BUG_CASES,
    ids=[
        "LONG_non_monotonic_56.95_to_56.90",
        "LONG_noop_56.95_to_56.95",
        "SHORT_non_monotonic_457_to_460",
        "LONG_invalid_null",
        "LONG_invalid_zero",
        "LONG_invalid_negative",
        "LONG_invalid_non_numeric",
        "LONG_invalid_NaN_string",
        "LONG_invalid_pos_inf",
        "LONG_invalid_neg_inf",
        "SHORT_invalid_NaN_float",
    ],
)
def test_bug_condition_concrete_cases_are_suppressed(side, old_stop, new_stop_raw, expected_reason):
    """
    **Validates: Requirements 1.1, 1.2, 1.3, 1.5**

    Concrete test cases from the bug condition specification.
    Each case represents a known bug scenario that should be suppressed.

    On UNFIXED code, this will fail with:
      ImportError: cannot import name 'should_suppress_maintenance_stop'
      (confirming the suppression layer does not exist)
    """
    from agents.portfolio_manager import should_suppress_maintenance_stop

    result = should_suppress_maintenance_stop(side, old_stop, new_stop_raw)

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"

    should_suppress, reason = result
    assert should_suppress is True, (
        f"Expected suppression for side={side}, old_stop={old_stop}, "
        f"new_stop_raw={new_stop_raw!r}, but got should_suppress=False"
    )
    assert reason == expected_reason, (
        f"Expected reason='{expected_reason}' for side={side}, old_stop={old_stop}, "
        f"new_stop_raw={new_stop_raw!r}, but got reason='{reason}'"
    )
