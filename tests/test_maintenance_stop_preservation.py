"""
Preservation Property Tests — Maintenance Stop Discipline

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

These tests verify that the maintenance stop suppressor does NOT intercept
valid proposals or delegate-to-stop-authority cases. They encode the
preservation behavior: inputs where the bug condition does NOT hold must
pass through unchanged.

Preservation cases tested:
- Valid monotonic LONG proposals (new_stop > old_stop > 0) → (False, None)
- Valid monotonic SHORT proposals (0 < new_stop < old_stop) → (False, None)
- Invalid old_stop (null, zero, negative, non-numeric) → (False, None) delegates to stop authority
- Unknown/missing side → (False, None) delegates to stop authority
- Boundary: new_stop = old_stop + epsilon for LONG (barely valid)
- Boundary: new_stop = old_stop - epsilon for SHORT (barely valid)

Observation-first methodology:
- On UNFIXED code, valid monotonic proposals currently pass to apply_stop_update()
  unchanged (no suppressor exists to block them).
- The suppressor, once implemented, must preserve this behavior by returning
  (False, None) for all preservation inputs.
- These tests will PASS after task 3.1 implements should_suppress_maintenance_stop().

NOTE: These tests will fail with ImportError on unfixed code because
should_suppress_maintenance_stop() does not exist yet. This is expected.
They will pass after the fix is implemented (task 3.1).
"""

import pytest
from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Hypothesis strategies for preservation inputs
# ---------------------------------------------------------------------------

def valid_long_monotonic_strategy():
    """
    Generate LONG proposals where new_stop > old_stop > 0 (valid monotonic).
    These must NOT be suppressed — they pass through to apply_stop_update().
    """
    return st.tuples(
        st.sampled_from(["long", "LONG", "Long", "buy", "BUY", "Buy"]),
        st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
    ).flatmap(
        lambda t: st.tuples(
            st.just(t[0]),  # side
            st.just(t[1]),  # old_stop
            # new_stop strictly greater than old_stop
            st.floats(
                min_value=t[1] + 0.0001,
                max_value=t[1] + 10000.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )


def valid_short_monotonic_strategy():
    """
    Generate SHORT proposals where 0 < new_stop < old_stop (valid monotonic).
    These must NOT be suppressed — they pass through to apply_stop_update().
    """
    return st.tuples(
        st.sampled_from(["short", "SHORT", "Short", "sell", "SELL", "Sell"]),
        st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    ).flatmap(
        lambda t: st.tuples(
            st.just(t[0]),  # side
            st.just(t[1]),  # old_stop
            # new_stop strictly less than old_stop but still positive
            st.floats(
                min_value=0.01,
                max_value=t[1] - 0.0001,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )


def invalid_old_stop_strategy():
    """
    Generate proposals where old_stop is null, zero, negative, or non-numeric.
    Suppressor must return (False, None) — delegates to stop authority.
    """
    invalid_old_stops = st.one_of(
        st.none(),
        st.just(0),
        st.just(0.0),
        st.floats(max_value=-0.01, allow_nan=False, allow_infinity=False),
        st.text(min_size=1, max_size=10).filter(lambda s: not _is_positive_numeric(s)),
    )
    sides = st.sampled_from(["long", "short", "LONG", "SHORT", "buy", "sell"])
    new_stops = st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False)

    return st.tuples(sides, invalid_old_stops, new_stops)


def unknown_side_strategy():
    """
    Generate proposals where side is None or unrecognized.
    Suppressor must return (False, None) — delegates to stop authority.
    """
    unknown_sides = st.one_of(
        st.none(),
        st.just(""),
        st.just("unknown"),
        st.just("neutral"),
        st.just("flat"),
        st.text(min_size=1, max_size=10).filter(
            lambda s: s.lower() not in {"long", "short", "buy", "sell"}
        ),
    )
    old_stops = st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False)
    new_stops = st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False)

    return st.tuples(unknown_sides, old_stops, new_stops)


def _is_positive_numeric(s: str) -> bool:
    """Check if a string represents a positive number."""
    try:
        return float(s) > 0
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Property Test: Valid LONG Monotonic Proposals Pass Through
# ---------------------------------------------------------------------------

@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    deadline=None,
)
@given(case=valid_long_monotonic_strategy())
def test_valid_long_monotonic_proposals_pass_through(case):
    """
    **Validates: Requirements 3.1**

    Property 2 (Preservation): For all LONG positions with new_stop > old_stop > 0,
    the suppressor returns (False, None) — proposal passes through to apply_stop_update().

    Observed on unfixed code: LONG, old_stop=56.95, new_stop=57.10 passes to
    apply_stop_update() (eligible for pass-through). The suppressor must preserve
    this behavior.
    """
    from agents.portfolio_manager import should_suppress_maintenance_stop

    side, old_stop, new_stop = case

    result = should_suppress_maintenance_stop(side, old_stop, new_stop)

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"

    should_suppress, reason = result
    assert should_suppress is False, (
        f"Valid LONG monotonic proposal should NOT be suppressed: "
        f"side={side}, old_stop={old_stop}, new_stop={new_stop}"
    )
    assert reason is None, (
        f"Reason should be None for valid proposals, got: {reason}"
    )


# ---------------------------------------------------------------------------
# Property Test: Valid SHORT Monotonic Proposals Pass Through
# ---------------------------------------------------------------------------

@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    deadline=None,
)
@given(case=valid_short_monotonic_strategy())
def test_valid_short_monotonic_proposals_pass_through(case):
    """
    **Validates: Requirements 3.2**

    Property 2 (Preservation): For all SHORT positions with 0 < new_stop < old_stop,
    the suppressor returns (False, None) — proposal passes through to apply_stop_update().

    Observed on unfixed code: SHORT, old_stop=457.00, new_stop=455.00 passes to
    apply_stop_update() (eligible for pass-through). The suppressor must preserve
    this behavior.
    """
    from agents.portfolio_manager import should_suppress_maintenance_stop

    side, old_stop, new_stop = case

    result = should_suppress_maintenance_stop(side, old_stop, new_stop)

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"

    should_suppress, reason = result
    assert should_suppress is False, (
        f"Valid SHORT monotonic proposal should NOT be suppressed: "
        f"side={side}, old_stop={old_stop}, new_stop={new_stop}"
    )
    assert reason is None, (
        f"Reason should be None for valid proposals, got: {reason}"
    )


# ---------------------------------------------------------------------------
# Property Test: Invalid old_stop Delegates to Stop Authority
# ---------------------------------------------------------------------------

@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    deadline=None,
)
@given(case=invalid_old_stop_strategy())
def test_invalid_old_stop_delegates_to_stop_authority(case):
    """
    **Validates: Requirements 3.3, 3.4**

    Property 2 (Preservation): For all proposals where old_stop is null, zero,
    negative, or non-numeric, the suppressor returns (False, None) — delegates
    to stop authority unchanged.

    Observed on unfixed code: old_stop=null, new_stop=57.10 delegates to stop
    authority unchanged. The suppressor must preserve this behavior.
    """
    from agents.portfolio_manager import should_suppress_maintenance_stop

    side, old_stop, new_stop = case

    result = should_suppress_maintenance_stop(side, old_stop, new_stop)

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"

    should_suppress, reason = result
    assert should_suppress is False, (
        f"Invalid old_stop should delegate to stop authority (not suppress): "
        f"side={side}, old_stop={old_stop!r}, new_stop={new_stop}"
    )
    assert reason is None, (
        f"Reason should be None when delegating to stop authority, got: {reason}"
    )


# ---------------------------------------------------------------------------
# Property Test: Unknown/Missing Side Delegates to Stop Authority
# ---------------------------------------------------------------------------

@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    deadline=None,
)
@given(case=unknown_side_strategy())
def test_unknown_side_delegates_to_stop_authority(case):
    """
    **Validates: Requirements 3.5, 3.6**

    Property 2 (Preservation): For all proposals where side is None or unrecognized,
    the suppressor returns (False, None) — delegates to stop authority unchanged.

    Observed on unfixed code: side=None, old_stop=56.95, new_stop=56.90 delegates
    to stop authority unchanged. The suppressor must preserve this behavior.
    """
    from agents.portfolio_manager import should_suppress_maintenance_stop

    side, old_stop, new_stop = case

    result = should_suppress_maintenance_stop(side, old_stop, new_stop)

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"

    should_suppress, reason = result
    assert should_suppress is False, (
        f"Unknown/missing side should delegate to stop authority (not suppress): "
        f"side={side!r}, old_stop={old_stop}, new_stop={new_stop}"
    )
    assert reason is None, (
        f"Reason should be None when delegating to stop authority, got: {reason}"
    )


# ---------------------------------------------------------------------------
# Boundary Tests: Barely Valid Proposals
# ---------------------------------------------------------------------------

@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    deadline=None,
)
@given(
    old_stop=st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
    epsilon=st.floats(min_value=1e-10, max_value=0.01, allow_nan=False, allow_infinity=False),
)
def test_boundary_long_barely_valid(old_stop, epsilon):
    """
    **Validates: Requirements 3.1**

    Boundary testing: new_stop = old_stop + epsilon for LONG (barely valid).
    Even the smallest positive increment must pass through.
    """
    from agents.portfolio_manager import should_suppress_maintenance_stop

    new_stop = old_stop + epsilon
    assume(new_stop > old_stop)  # guard against floating point collapse

    result = should_suppress_maintenance_stop("long", old_stop, new_stop)

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"

    should_suppress, reason = result
    assert should_suppress is False, (
        f"Barely valid LONG proposal should NOT be suppressed: "
        f"old_stop={old_stop}, new_stop={new_stop}, epsilon={epsilon}"
    )
    assert reason is None, (
        f"Reason should be None for barely valid proposals, got: {reason}"
    )


@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
    deadline=None,
)
@given(
    old_stop=st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    epsilon=st.floats(min_value=1e-10, max_value=0.01, allow_nan=False, allow_infinity=False),
)
def test_boundary_short_barely_valid(old_stop, epsilon):
    """
    **Validates: Requirements 3.2**

    Boundary testing: new_stop = old_stop - epsilon for SHORT (barely valid).
    Even the smallest decrement must pass through, as long as new_stop > 0.
    """
    from agents.portfolio_manager import should_suppress_maintenance_stop

    new_stop = old_stop - epsilon
    assume(new_stop < old_stop)  # guard against floating point collapse
    assume(new_stop > 0)  # must remain positive

    result = should_suppress_maintenance_stop("short", old_stop, new_stop)

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"

    should_suppress, reason = result
    assert should_suppress is False, (
        f"Barely valid SHORT proposal should NOT be suppressed: "
        f"old_stop={old_stop}, new_stop={new_stop}, epsilon={epsilon}"
    )
    assert reason is None, (
        f"Reason should be None for barely valid proposals, got: {reason}"
    )


# ---------------------------------------------------------------------------
# Parametrized Test: Concrete preservation cases from specification
# ---------------------------------------------------------------------------

CONCRETE_PRESERVATION_CASES = [
    # (side, old_stop, new_stop, description)
    ("long", 56.95, 57.10, "LONG valid monotonic — passes through"),
    ("short", 457.00, 455.00, "SHORT valid monotonic — passes through"),
    (None, 56.95, 56.90, "side=None — delegates to stop authority"),
    ("long", None, 57.10, "old_stop=None — delegates to stop authority"),
    ("long", 0, 57.10, "old_stop=0 — delegates to stop authority"),
    ("long", -5.0, 57.10, "old_stop=negative — delegates to stop authority"),
    ("unknown", 56.95, 56.90, "side=unknown — delegates to stop authority"),
    ("", 56.95, 56.90, "side=empty string — delegates to stop authority"),
    ("LONG", 56.95, 57.10, "LONG uppercase — valid monotonic passes through"),
    ("buy", 56.95, 57.10, "buy alias — valid monotonic passes through"),
    ("SHORT", 457.00, 455.00, "SHORT uppercase — valid monotonic passes through"),
    ("sell", 457.00, 455.00, "sell alias — valid monotonic passes through"),
    ("long", float("nan"), 57.10, "old_stop=NaN — delegates to stop authority"),
    ("long", float("inf"), 57.10, "old_stop=+inf — delegates to stop authority"),
    ("short", float("-inf"), 455.00, "old_stop=-inf — delegates to stop authority"),
]


@pytest.mark.parametrize(
    "side,old_stop,new_stop,description",
    CONCRETE_PRESERVATION_CASES,
    ids=[c[3] for c in CONCRETE_PRESERVATION_CASES],
)
def test_preservation_concrete_cases(side, old_stop, new_stop, description):
    """
    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

    Concrete preservation cases from the specification observations.
    Each case represents a scenario where the suppressor must NOT intercept.
    """
    from agents.portfolio_manager import should_suppress_maintenance_stop

    result = should_suppress_maintenance_stop(side, old_stop, new_stop)

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"

    should_suppress, reason = result
    assert should_suppress is False, (
        f"Preservation case should NOT be suppressed ({description}): "
        f"side={side!r}, old_stop={old_stop!r}, new_stop={new_stop!r}"
    )
    assert reason is None, (
        f"Reason should be None for preservation cases, got: {reason}"
    )
