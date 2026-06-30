"""
Property-based tests for dispatch mode resolution using Hypothesis.

Tests universal correctness properties that must hold across all valid inputs.
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from utils.dispatch_mode import (
    resolve_per_alert_mode,
    normalize_global_mode,
)


# Valid canonical modes (after normalization)
_VALID_MODES = {"dispatch", "observe", "disabled"}
# Valid raw inputs that resolve to canonical modes (including "enabled" alias)
_VALID_RAW_INPUTS = {"dispatch", "observe", "disabled", "enabled"}


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Invalid non-empty strings: text that is NOT a valid mode and NOT whitespace-only
invalid_mode_strategy = st.text(min_size=1).filter(
    lambda s: s.strip().lower() not in _VALID_RAW_INPUTS and s.strip() != ""
)

# Whitespace-only or empty strings: these indicate "unset/inherit"
whitespace_strategy = st.sampled_from(["", " ", "  ", "\t", "\n", "  \t  "])

# Valid global modes (canonical form only — global mode is already normalized)
global_mode_strategy = st.sampled_from(["dispatch", "observe", "disabled"])

# Alert types
alert_type_strategy = st.sampled_from(["entry_alert", "breakout", "rapid_move", "target_hit"])


# ---------------------------------------------------------------------------
# Property 2: Invalid mode values are treated as disabled; empty/whitespace
#              inherits global
# ---------------------------------------------------------------------------


class TestProperty2InvalidModeValuesDisabledEmptyInherits:
    """
    Any non-empty string that is NOT one of (dispatch, observe, disabled, enabled)
    is treated as "disabled" (fail-closed). Empty string or whitespace-only inherits
    the global mode exactly (after normalization).

    The key distinction:
    - Invalid non-empty = always "disabled" (fail-closed for unknown values)
    - Empty/whitespace = inherit global (could be dispatch, observe, or disabled)

    **Validates: Requirements 1.6, 1.7, 10.5**
    """

    @given(
        invalid_value=invalid_mode_strategy,
        global_mode=global_mode_strategy,
        alert_type=alert_type_strategy,
    )
    @settings(max_examples=200)
    def test_invalid_non_empty_values_resolve_to_disabled(
        self, invalid_value: str, global_mode: str, alert_type: str
    ):
        """Any non-empty invalid mode string always resolves to 'disabled'."""
        result = resolve_per_alert_mode(invalid_value, global_mode, alert_type)

        # Invalid values are treated as disabled (fail-closed).
        # Since disabled is the MOST restrictive mode, the effective result
        # is always "disabled" regardless of global_mode (disabled >= everything).
        assert result == "disabled", (
            f"Invalid value '{invalid_value}' with global_mode='{global_mode}' "
            f"should resolve to 'disabled', got '{result}'"
        )

    @given(
        whitespace_value=whitespace_strategy,
        global_mode=global_mode_strategy,
        alert_type=alert_type_strategy,
    )
    @settings(max_examples=200)
    def test_empty_or_whitespace_inherits_global_mode(
        self, whitespace_value: str, global_mode: str, alert_type: str
    ):
        """Empty string or whitespace-only inherits global mode exactly."""
        result = resolve_per_alert_mode(whitespace_value, global_mode, alert_type)

        # Empty/whitespace means "unset" → inherit from global.
        # Since per-alert inherits global, and restrictiveness precedence
        # compares global vs per-alert (which are the same), result = global.
        assert result == global_mode, (
            f"Whitespace value '{repr(whitespace_value)}' with global_mode='{global_mode}' "
            f"should inherit global mode '{global_mode}', got '{result}'"
        )

    @given(
        invalid_value=invalid_mode_strategy,
    )
    @settings(max_examples=200)
    def test_normalize_global_mode_invalid_returns_disabled(
        self, invalid_value: str,
    ):
        """normalize_global_mode with invalid input returns 'disabled' (fail-closed)."""
        result = normalize_global_mode(invalid_value)

        assert result == "disabled", (
            f"normalize_global_mode('{invalid_value}') should return 'disabled', "
            f"got '{result}'"
        )

    @given(
        whitespace_value=whitespace_strategy,
    )
    @settings(max_examples=200)
    def test_normalize_global_mode_empty_returns_disabled(
        self, whitespace_value: str,
    ):
        """normalize_global_mode with empty/whitespace returns 'disabled' (fail-closed)."""
        result = normalize_global_mode(whitespace_value)

        assert result == "disabled", (
            f"normalize_global_mode('{repr(whitespace_value)}') should return 'disabled', "
            f"got '{result}'"
        )

    @given(
        invalid_value=invalid_mode_strategy,
        global_mode=global_mode_strategy,
        alert_type=alert_type_strategy,
        whitespace_value=whitespace_strategy,
    )
    @settings(max_examples=200)
    def test_invalid_vs_empty_distinction(
        self, invalid_value: str, global_mode: str, alert_type: str, whitespace_value: str
    ):
        """Invalid non-empty and empty/whitespace are distinct behaviors."""
        invalid_result = resolve_per_alert_mode(invalid_value, global_mode, alert_type)
        whitespace_result = resolve_per_alert_mode(whitespace_value, global_mode, alert_type)

        # Invalid always → disabled
        assert invalid_result == "disabled", (
            f"Invalid '{invalid_value}' should be 'disabled', got '{invalid_result}'"
        )

        # Whitespace always → global_mode
        assert whitespace_result == global_mode, (
            f"Whitespace '{repr(whitespace_value)}' should inherit '{global_mode}', "
            f"got '{whitespace_result}'"
        )

        # When global_mode != "disabled", the two behaviors diverge
        if global_mode != "disabled":
            assert invalid_result != whitespace_result, (
                f"When global_mode='{global_mode}', invalid and whitespace should differ: "
                f"invalid='{invalid_result}', whitespace='{whitespace_result}'"
            )


# ---------------------------------------------------------------------------
# Property 16: Mode configuration round-trip
# **Validates: Requirements 10.6**
# ---------------------------------------------------------------------------

from utils.dispatch_mode import (
    build_dispatch_mode_config,
    serialize_mode_config,
)

_ALERT_TYPES = ["entry_alert", "breakout", "rapid_move", "target_hit"]
_CANONICAL_MODES_LIST = ["dispatch", "observe", "disabled"]

st_global_raw = st.sampled_from(["dispatch", "observe", "disabled", "enabled", "", "invalid"])
st_per_alert_value = st.sampled_from(["dispatch", "observe", "disabled", "enabled", "", "bogus"])


@given(
    global_raw=st_global_raw,
    env_entry_alert=st_per_alert_value,
    env_breakout=st_per_alert_value,
    env_rapid_move=st_per_alert_value,
    env_target_hit=st_per_alert_value,
)
@settings(max_examples=200)
def test_mode_config_round_trip(
    global_raw, env_entry_alert, env_breakout, env_rapid_move, env_target_hit
):
    """Property 16: Mode configuration round-trip.

    Serializing a built config produces canonical modes for all 4 alert types,
    never contains "enabled", and re-parsing produces identical effective modes.

    **Validates: Requirements 10.6**
    """
    env_values = {
        "entry_alert": env_entry_alert,
        "breakout": env_breakout,
        "rapid_move": env_rapid_move,
        "target_hit": env_target_hit,
    }

    # Build config from raw inputs
    config = build_dispatch_mode_config(global_raw, env_values)

    # Serialize to dict
    serialized = serialize_mode_config(config)

    # 1. Keys are exactly the 4 alert types
    assert set(serialized.keys()) == set(_ALERT_TYPES), (
        f"Expected keys {set(_ALERT_TYPES)}, got {set(serialized.keys())}"
    )

    # 2. All values are valid canonical modes (never "enabled")
    for alert_type, mode_val in serialized.items():
        assert mode_val in _CANONICAL_MODES_LIST, (
            f"Non-canonical mode '{mode_val}' for {alert_type}"
        )

    # 3. "enabled" never appears in serialized values
    assert "enabled" not in serialized.values(), (
        f"'enabled' found in serialized values: {serialized}"
    )

    # 4. Re-building from serialized produces same effective modes
    for alert_type in _ALERT_TYPES:
        re_resolved = resolve_per_alert_mode(
            serialized[alert_type], config.global_mode, alert_type
        )
        assert re_resolved == config.effective_mode(alert_type), (
            f"Round-trip mismatch for {alert_type}: "
            f"re_resolved={re_resolved}, original={config.effective_mode(alert_type)}"
        )
