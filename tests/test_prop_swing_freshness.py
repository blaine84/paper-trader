"""Property-based tests for freshness checks (Properties 13, 14).

Tests that:
- Property 13: Signal Freshness Precedes Normalization — signals exceeding threshold
  or lacking valid age field get stale_signal regardless of other checks
- Property 14: Catalyst Freshness Rejection — signals with catalyst_freshness="stale"
  or absent/null get stale_catalyst when signal freshness passes

Validates: Requirements 10.1, 10.3, 10.4, 11.1, 11.3
"""

from __future__ import annotations

from hypothesis import given, settings, strategies as st, assume
from unittest.mock import patch

from utils.swing_candidate_bridge import (
    _check_signal_freshness,
    _check_catalyst_freshness,
)
from utils.gate_config import SWING_SIGNAL_FRESHNESS_HOURS


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Ages that exceed the threshold (stale signals)
stale_age_st = st.floats(
    min_value=SWING_SIGNAL_FRESHNESS_HOURS + 0.001,
    max_value=10000.0,
    allow_nan=False,
    allow_infinity=False,
)

# Ages within the threshold (fresh signals)
fresh_age_st = st.floats(
    min_value=0.0,
    max_value=float(SWING_SIGNAL_FRESHNESS_HOURS),
    allow_nan=False,
    allow_infinity=False,
)

# Non-numeric values that cannot be parsed as a float
non_numeric_st = st.one_of(
    st.text(min_size=1).filter(lambda s: not _is_numeric(s)),
    st.just([1, 2, 3]),
    st.just({}),
    st.just(True),  # booleans are tricky — True == 1.0, but test the code
)


def _is_numeric(s: str) -> bool:
    """Check if a string can be parsed as a float."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Property 13: Signal Freshness Precedes Normalization
# Validates: Requirements 10.1, 10.3, 10.4
#
# Signals exceeding the freshness threshold or lacking a valid age field
# must return "stale_signal" regardless of any other signal properties.
# ---------------------------------------------------------------------------


@given(age=stale_age_st)
@settings(max_examples=200)
def test_signal_freshness_stale_when_age_exceeds_threshold(age: float):
    """Property 13a: signal_age_hours > threshold → stale_signal.

    Any signal whose age exceeds SWING_SIGNAL_FRESHNESS_HOURS must be
    rejected as stale, regardless of other fields on the signal.

    **Validates: Requirements 10.1**
    """
    signal = {"signal_age_hours": age, "symbol": "TEST"}
    result = _check_signal_freshness(signal)
    assert result == "stale_signal", (
        f"Expected stale_signal for age={age} > threshold={SWING_SIGNAL_FRESHNESS_HOURS}"
    )


@given(
    extra_fields=st.fixed_dictionaries({
        "symbol": st.text(min_size=1, max_size=5),
        "catalyst_freshness": st.sampled_from(["fresh", "aging", "stale"]),
        "direction": st.sampled_from(["LONG", "SHORT", "HOLD"]),
    })
)
@settings(max_examples=200)
def test_signal_freshness_stale_when_age_missing(extra_fields: dict):
    """Property 13b: missing signal_age_hours → stale_signal.

    Signals that lack a signal_age_hours field entirely must be rejected
    as stale — the absence of age information means we cannot verify freshness.

    **Validates: Requirements 10.4**
    """
    # signal dict WITHOUT signal_age_hours key
    signal = {**extra_fields}
    assert "signal_age_hours" not in signal
    result = _check_signal_freshness(signal)
    assert result == "stale_signal"


@given(
    extra_fields=st.fixed_dictionaries({
        "symbol": st.text(min_size=1, max_size=5),
        "catalyst_freshness": st.sampled_from(["fresh", "aging", "stale"]),
    })
)
@settings(max_examples=200)
def test_signal_freshness_stale_when_age_none(extra_fields: dict):
    """Property 13c: signal_age_hours=None → stale_signal.

    Signals with an explicit None age field must be treated as stale.

    **Validates: Requirements 10.4**
    """
    signal = {"signal_age_hours": None, **extra_fields}
    result = _check_signal_freshness(signal)
    assert result == "stale_signal"


@given(
    bad_age=st.text(min_size=1).filter(lambda s: not _is_numeric(s))
)
@settings(max_examples=200)
def test_signal_freshness_stale_when_age_non_numeric(bad_age: str):
    """Property 13d: non-numeric signal_age_hours → stale_signal.

    Signals with a non-parseable age value must be rejected as stale
    since we cannot determine freshness from invalid data.

    **Validates: Requirements 10.4**
    """
    signal = {"signal_age_hours": bad_age, "symbol": "TEST"}
    result = _check_signal_freshness(signal)
    assert result == "stale_signal"


@given(age=fresh_age_st)
@settings(max_examples=200)
def test_signal_freshness_passes_when_age_within_threshold(age: float):
    """Property 13e: signal_age_hours <= threshold → None (pass).

    Fresh signals with age at or below the threshold must pass the
    freshness check (return None), allowing them to proceed to further checks.

    **Validates: Requirements 10.1, 10.3**
    """
    signal = {"signal_age_hours": age, "symbol": "TEST"}
    result = _check_signal_freshness(signal)
    assert result is None, (
        f"Expected None (fresh) for age={age} <= threshold={SWING_SIGNAL_FRESHNESS_HOURS}"
    )


# ---------------------------------------------------------------------------
# Property 14: Catalyst Freshness Rejection
# Validates: Requirements 11.1, 11.3
#
# Signals with catalyst_freshness="stale" or absent/null get stale_catalyst.
# Signals with catalyst_freshness="fresh" or "aging" pass (return None).
# ---------------------------------------------------------------------------


@given(
    extra_fields=st.fixed_dictionaries({
        "symbol": st.text(min_size=1, max_size=5),
        "signal_age_hours": fresh_age_st,
    })
)
@settings(max_examples=200)
def test_catalyst_freshness_rejects_stale(extra_fields: dict):
    """Property 14a: catalyst_freshness="stale" → stale_catalyst.

    When catalyst context is explicitly marked stale, the signal must be
    rejected with stale_catalyst to prevent overnight holds based on
    outdated news/catalyst context.

    **Validates: Requirements 11.1, 11.3**
    """
    signal = {"catalyst_freshness": "stale", **extra_fields}
    result = _check_catalyst_freshness(signal)
    assert result == "stale_catalyst"


@given(
    extra_fields=st.fixed_dictionaries({
        "symbol": st.text(min_size=1, max_size=5),
        "signal_age_hours": fresh_age_st,
    })
)
@settings(max_examples=200)
def test_catalyst_freshness_rejects_absent(extra_fields: dict):
    """Property 14b: catalyst_freshness absent → stale_catalyst.

    When the catalyst_freshness field is missing entirely, the signal must
    be treated as having stale catalyst context and rejected.

    **Validates: Requirements 11.1**
    """
    signal = {**extra_fields}
    assert "catalyst_freshness" not in signal
    result = _check_catalyst_freshness(signal)
    assert result == "stale_catalyst"


@given(
    extra_fields=st.fixed_dictionaries({
        "symbol": st.text(min_size=1, max_size=5),
        "signal_age_hours": fresh_age_st,
    })
)
@settings(max_examples=200)
def test_catalyst_freshness_rejects_none(extra_fields: dict):
    """Property 14c: catalyst_freshness=None → stale_catalyst.

    Explicit None is treated as absent/stale per Req 11.1.

    **Validates: Requirements 11.1**
    """
    signal = {"catalyst_freshness": None, **extra_fields}
    result = _check_catalyst_freshness(signal)
    assert result == "stale_catalyst"


@given(
    freshness_value=st.sampled_from(["fresh", "aging"]),
    extra_fields=st.fixed_dictionaries({
        "symbol": st.text(min_size=1, max_size=5),
        "signal_age_hours": fresh_age_st,
    }),
)
@settings(max_examples=200)
def test_catalyst_freshness_passes_fresh_or_aging(freshness_value: str, extra_fields: dict):
    """Property 14d: catalyst_freshness="fresh" or "aging" → None (pass).

    Fresh and aging catalyst contexts are acceptable for swing evaluation.
    Only "stale" or absent values trigger rejection.

    **Validates: Requirements 11.1, 11.3**
    """
    signal = {"catalyst_freshness": freshness_value, **extra_fields}
    result = _check_catalyst_freshness(signal)
    assert result is None, (
        f"Expected None (pass) for catalyst_freshness={freshness_value!r}"
    )
