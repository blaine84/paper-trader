"""
Property-based tests for freshness clamping using Hypothesis.

Validates that _clamp_freshness() always produces a value in [1, 120] regardless
of input, with correct clamping and default behaviors.

**Validates: Requirements 3.5**
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from utils.gate_config import _clamp_freshness


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Alert types that use freshness limits
alert_type_strategy = st.sampled_from(["entry_alert", "breakout", "rapid_move"])

# Numeric integers as strings (broad range to test clamping)
numeric_string_strategy = st.integers().map(str)

# Non-numeric strings that cannot be parsed as int
non_numeric_strategy = st.text(min_size=1).filter(
    lambda s: not _is_numeric(s)
)


def _is_numeric(s: str) -> bool:
    """Check if a string can be parsed as an integer."""
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Property 6 (partial): Freshness limit clamped to [1, 120] minutes
# **Validates: Requirements 3.5**
# ---------------------------------------------------------------------------


class TestProperty6FreshnessClamping:
    """
    _clamp_freshness() always returns an integer in [1, 120] regardless of input.
    Values below 1 are clamped to 1, values above 120 are clamped to 120,
    values in [1, 120] pass through unchanged, and non-numeric strings default to 15.

    **Validates: Requirements 3.5**
    """

    @given(
        raw_value=numeric_string_strategy,
        alert_type=alert_type_strategy,
    )
    @settings(max_examples=200)
    def test_result_always_int_in_range(self, raw_value: str, alert_type: str):
        """_clamp_freshness() always returns an int in [1, 120]."""
        result = _clamp_freshness(raw_value, alert_type)

        assert isinstance(result, int), (
            f"Expected int, got {type(result).__name__} for input '{raw_value}'"
        )
        assert 1 <= result <= 120, (
            f"Result {result} out of [1, 120] for input '{raw_value}'"
        )

    @given(
        raw_value=st.integers(max_value=0).map(str),
        alert_type=alert_type_strategy,
    )
    @settings(max_examples=200)
    def test_values_below_1_clamped_to_1(self, raw_value: str, alert_type: str):
        """Values < 1 are clamped to 1."""
        result = _clamp_freshness(raw_value, alert_type)

        assert result == 1, (
            f"Expected 1 for input '{raw_value}' (below minimum), got {result}"
        )

    @given(
        raw_value=st.integers(min_value=121).map(str),
        alert_type=alert_type_strategy,
    )
    @settings(max_examples=200)
    def test_values_above_120_clamped_to_120(self, raw_value: str, alert_type: str):
        """Values > 120 are clamped to 120."""
        result = _clamp_freshness(raw_value, alert_type)

        assert result == 120, (
            f"Expected 120 for input '{raw_value}' (above maximum), got {result}"
        )

    @given(
        raw_value=st.integers(min_value=1, max_value=120).map(str),
        alert_type=alert_type_strategy,
    )
    @settings(max_examples=200)
    def test_values_in_range_pass_through_unchanged(self, raw_value: str, alert_type: str):
        """Values in [1, 120] pass through unchanged."""
        result = _clamp_freshness(raw_value, alert_type)

        assert result == int(raw_value), (
            f"Expected {int(raw_value)} for in-range input '{raw_value}', got {result}"
        )

    @given(
        raw_value=non_numeric_strategy,
        alert_type=alert_type_strategy,
    )
    @settings(max_examples=200)
    def test_non_numeric_defaults_to_15(self, raw_value: str, alert_type: str):
        """Non-numeric strings default to 15 (then clamped normally, so result is 15)."""
        result = _clamp_freshness(raw_value, alert_type)

        assert result == 15, (
            f"Expected 15 for non-numeric input '{raw_value}', got {result}"
        )
