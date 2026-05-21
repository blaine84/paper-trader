"""
Property-based tests for Aggressive Tactical Stop Geometry using Hypothesis.

Tests universal correctness properties that must hold across all valid inputs
for the tactical stop exception path in risk_geometry_gate.
"""

from datetime import datetime, timedelta, timezone

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from utils.gate_config import HIGH_BETA_CLUSTER, STOP_DISTANCE_RULES
from utils.risk_geometry_gate import evaluate_risk_geometry, _has_tactical_context


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

# Qualifying setups from config
_tactical_cfg = STOP_DISTANCE_RULES["high_beta_mega_cap_intraday"]["tactical_stop_by_profile"]["aggressive"]
QUALIFYING_SETUPS = _tactical_cfg["qualifying_setups"]
CONDITIONAL_SETUPS = _tactical_cfg["conditional_setups"]
ALL_TACTICAL_SETUPS = QUALIFYING_SETUPS + CONDITIONAL_SETUPS

# Tactical context indicators that must be AVOIDED in metadata/rationale for Property 4
TACTICAL_INDICATORS = _tactical_cfg["tactical_context_indicators"]

st_high_beta_symbol = st.sampled_from(sorted(HIGH_BETA_CLUSTER))
st_entry_price = st.floats(min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False)
st_atr = st.floats(min_value=0.05, max_value=5.0, allow_nan=False, allow_infinity=False)
st_qualifying_setup = st.sampled_from(QUALIFYING_SETUPS)

# Non-aggressive profiles: conservative, moderate, or random non-aggressive strings
st_non_aggressive_profile = st.one_of(
    st.sampled_from(["conservative", "moderate"]),
    st.text(min_size=1, max_size=20).filter(lambda s: s.lower() != "aggressive"),
)

# Characters safe for generating random text that won't accidentally contain indicators
SAFE_ALPHABET = "abcdefghijklmnqrtwxyz0123456789 "


def _safe_text_strategy():
    """Generate random text that does NOT contain any tactical context indicators."""
    return st.text(
        alphabet=SAFE_ALPHABET,
        min_size=0,
        max_size=100,
    ).filter(
        lambda t: not any(ind in t.lower() for ind in TACTICAL_INDICATORS)
    )


# ---------------------------------------------------------------------------
# Property 1: Non-aggressive profiles never trigger tactical exception
# Feature: aggressive-tactical-stop-geometry, Property 1: Non-aggressive profiles never trigger tactical exception
# ---------------------------------------------------------------------------


class TestProperty1NonAggressiveProfilesNeverTriggerTacticalException:
    """
    For any trade on a HIGH_BETA_CLUSTER symbol with a qualifying tactical setup,
    if the profile is not "aggressive" (including "conservative", "moderate", or
    any unknown string), the result SHALL NOT contain `tactical_stop_applied` in
    the result dictionary.

    **Validates: Requirements 1.3, 2.5, 6.1, 6.2**
    """

    @given(
        profile=st_non_aggressive_profile,
        symbol=st_high_beta_symbol,
        setup_type=st_qualifying_setup,
        entry_price=st_entry_price,
        atr_5min=st_atr,
        stop_offset_pct=st.floats(min_value=0.003, max_value=0.05, allow_nan=False, allow_infinity=False),
        target_offset_pct=st.floats(min_value=0.01, max_value=0.10, allow_nan=False, allow_infinity=False),
        quantity=st.integers(min_value=1, max_value=500),
    )
    @settings(max_examples=100)
    def test_non_aggressive_profiles_never_get_tactical_stop(
        self,
        profile,
        symbol,
        setup_type,
        entry_price,
        atr_5min,
        stop_offset_pct,
        target_offset_pct,
        quantity,
    ):
        """Non-aggressive profiles never produce tactical_stop_applied in result."""
        # Construct valid LONG trade geometry
        stop_price = entry_price * (1 - stop_offset_pct)
        target_price = entry_price * (1 + target_offset_pct)

        # Ensure valid geometry
        assume(stop_price > 0)
        assume(target_price > entry_price)
        assume(stop_price < entry_price)

        # Use a fresh ATR timestamp (within 15 minutes)
        now = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        atr_timestamp = now - timedelta(minutes=2)

        result = evaluate_risk_geometry(
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=quantity,
            direction="BUY",
            symbol=symbol,
            setup_type=setup_type,
            atr_5min=atr_5min,
            atr_timestamp=atr_timestamp,
            trade_timestamp=now,
            max_dollar_risk=50_000.0,
            profile=profile,
            trade_metadata="support bounce near VWAP",
            trade_rationale="Bouncing off support level with pullback",
        )

        assert "tactical_stop_applied" not in result, (
            f"Non-aggressive profile '{profile}' should never trigger tactical exception. "
            f"Got result with tactical_stop_applied={result.get('tactical_stop_applied')}. "
            f"symbol={symbol}, setup_type={setup_type}, entry={entry_price}"
        )


# ---------------------------------------------------------------------------
# Hypothesis Strategies for Property 3
# ---------------------------------------------------------------------------

_ASCII_LOWER = "abcdefghijklmnopqrstuvwxyz"

# Generate alphabetic indicator strings between 3-10 characters (ASCII lowercase)
st_indicator = st.text(
    alphabet=st.sampled_from(list(_ASCII_LOWER)),
    min_size=3,
    max_size=10,
)

# Generate random filler words (ASCII lowercase)
st_filler_word = st.text(
    alphabet=st.sampled_from(list(_ASCII_LOWER)),
    min_size=3,
    max_size=8,
)


# ---------------------------------------------------------------------------
# Property 3: Word-boundary indicator matching
# Feature: aggressive-tactical-stop-geometry, Property 3: Word-boundary indicator matching
# ---------------------------------------------------------------------------


class TestProperty3WordBoundaryIndicatorMatching:
    """
    For any tactical context indicator string and any case variation,
    when that variation appears as a whole word (word-boundary delimited)
    in trade metadata or rationale, _has_tactical_context SHALL return True.
    Conversely, when the indicator appears only as a substring of a larger
    word (e.g., "support" within "unsupported"), the function SHALL return False.

    **Validates: Requirements 1.4**
    """

    @given(
        indicator=st_indicator,
        prefix_text=st_filler_word,
        suffix_text=st_filler_word,
    )
    @settings(max_examples=100)
    def test_whole_word_match_returns_true(self, indicator, prefix_text, suffix_text):
        """Indicator as a whole word (surrounded by spaces) returns True."""
        # Embed indicator as a whole word surrounded by spaces
        text = f"{prefix_text} {indicator} {suffix_text}"
        result = _has_tactical_context([indicator], text, None)
        assert result is True, (
            f"Expected True for whole-word match: indicator={indicator!r}, text={text!r}"
        )

    @given(
        indicator=st_indicator,
        prefix_text=st_filler_word,
        suffix_text=st_filler_word,
    )
    @settings(max_examples=100)
    def test_whole_word_match_in_rationale_returns_true(self, indicator, prefix_text, suffix_text):
        """Indicator as a whole word in rationale (second param) returns True."""
        text = f"{prefix_text} {indicator} {suffix_text}"
        result = _has_tactical_context([indicator], None, text)
        assert result is True, (
            f"Expected True for whole-word match in rationale: indicator={indicator!r}, text={text!r}"
        )

    @given(
        indicator=st_indicator,
        prefix_addition=st.text(
            alphabet=st.sampled_from(list(_ASCII_LOWER)),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(max_examples=100)
    def test_substring_only_as_suffix_returns_false(self, indicator, prefix_addition):
        """Indicator embedded as suffix of a larger word returns False (e.g., 'unsupported')."""
        # Create a larger word with indicator as suffix: e.g., "un" + "support" = "unsupport"
        larger_word = prefix_addition + indicator
        # Ensure the larger word is different from the indicator itself
        assume(larger_word.lower() != indicator.lower())
        text = f"some text {larger_word} more text"
        result = _has_tactical_context([indicator], text, None)
        assert result is False, (
            f"Expected False for substring-only (suffix): indicator={indicator!r}, "
            f"larger_word={larger_word!r}, text={text!r}"
        )

    @given(
        indicator=st_indicator,
        suffix_addition=st.text(
            alphabet=st.sampled_from(list(_ASCII_LOWER)),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(max_examples=100)
    def test_substring_only_as_prefix_returns_false(self, indicator, suffix_addition):
        """Indicator embedded as prefix of a larger word returns False (e.g., 'supported')."""
        # Create a larger word with indicator as prefix: e.g., "support" + "ed" = "supported"
        larger_word = indicator + suffix_addition
        # Ensure the larger word is different from the indicator itself
        assume(larger_word.lower() != indicator.lower())
        text = f"some text {larger_word} more text"
        result = _has_tactical_context([indicator], text, None)
        assert result is False, (
            f"Expected False for substring-only (prefix): indicator={indicator!r}, "
            f"larger_word={larger_word!r}, text={text!r}"
        )

    @given(
        indicator=st_indicator,
        prefix_text=st_filler_word,
        suffix_text=st_filler_word,
    )
    @settings(max_examples=100)
    def test_case_insensitive_upper(self, indicator, prefix_text, suffix_text):
        """Upper-case variation of indicator as whole word returns True."""
        text = f"{prefix_text} {indicator.upper()} {suffix_text}"
        result = _has_tactical_context([indicator], text, None)
        assert result is True, (
            f"Expected True for upper-case whole-word: indicator={indicator!r}, text={text!r}"
        )

    @given(
        indicator=st_indicator,
        prefix_text=st_filler_word,
        suffix_text=st_filler_word,
    )
    @settings(max_examples=100)
    def test_case_insensitive_lower(self, indicator, prefix_text, suffix_text):
        """Lower-case variation of indicator as whole word returns True (indicator given in upper)."""
        text = f"{prefix_text} {indicator.lower()} {suffix_text}"
        result = _has_tactical_context([indicator.upper()], text, None)
        assert result is True, (
            f"Expected True for lower-case text with upper indicator: "
            f"indicator={indicator.upper()!r}, text={text!r}"
        )

    @given(
        indicator=st_indicator,
        prefix_text=st_filler_word,
        suffix_text=st_filler_word,
    )
    @settings(max_examples=100)
    def test_case_insensitive_mixed(self, indicator, prefix_text, suffix_text):
        """Mixed-case variation of indicator as whole word returns True."""
        # Create a mixed-case version: alternate upper/lower
        mixed = "".join(
            c.upper() if i % 2 == 0 else c.lower()
            for i, c in enumerate(indicator)
        )
        text = f"{prefix_text} {mixed} {suffix_text}"
        result = _has_tactical_context([indicator], text, None)
        assert result is True, (
            f"Expected True for mixed-case whole-word: indicator={indicator!r}, "
            f"mixed={mixed!r}, text={text!r}"
        )


# Non-qualifying setup types — these should never trigger tactical exception
NON_QUALIFYING_SETUPS = [
    "momentum_fade",
    "breakout_continuation",
    "gap_and_go",
    "technical_breakout",
    "mean_reversion",
    "swing_trade",
    "range_bound",
    "reversal",
    "scalp",
    "opening_drive",
]

st_non_qualifying_setup = st.sampled_from(NON_QUALIFYING_SETUPS)


# ---------------------------------------------------------------------------
# Property 2: Non-qualifying setups never trigger tactical exception
# Feature: aggressive-tactical-stop-geometry, Property 2: Non-qualifying setups never trigger tactical exception
# ---------------------------------------------------------------------------


class TestProperty2NonQualifyingSetupsNeverTriggerTacticalException:
    """
    For any trade on a HIGH_BETA_CLUSTER symbol with profile "aggressive",
    if the setup type is not in the configured qualifying_setups or
    conditional_setups lists, the result SHALL NOT contain
    `tactical_stop_applied` in the result dictionary.

    **Validates: Requirements 2.6, 6.3, 6.6**
    """

    @given(
        symbol=st_high_beta_symbol,
        entry_price=st_entry_price,
        stop_offset_pct=st.floats(min_value=0.001, max_value=0.05, allow_nan=False, allow_infinity=False),
        target_offset_pct=st.floats(min_value=0.002, max_value=0.10, allow_nan=False, allow_infinity=False),
        atr_5min=st_atr,
        quantity=st.integers(min_value=1, max_value=500),
        max_dollar_risk=st.floats(min_value=100.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        setup_type=st_non_qualifying_setup,
    )
    @settings(max_examples=100)
    def test_non_qualifying_setup_never_triggers_tactical_exception(
        self,
        symbol,
        entry_price,
        stop_offset_pct,
        target_offset_pct,
        atr_5min,
        quantity,
        max_dollar_risk,
        setup_type,
    ):
        """Non-qualifying setup types with aggressive profile never get tactical_stop_applied."""
        # Ensure setup_type is truly not in qualifying or conditional lists
        assume(setup_type.lower() not in [s.lower() for s in ALL_TACTICAL_SETUPS])

        # Compute valid trade geometry (LONG direction)
        stop_price = entry_price * (1 - stop_offset_pct)
        target_price = entry_price * (1 + target_offset_pct)

        # Ensure valid geometry
        assume(stop_price > 0)
        assume(target_price > entry_price)
        assume(stop_price < entry_price)

        # Fresh ATR timestamp (within 15 minutes)
        trade_timestamp = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        atr_timestamp = trade_timestamp - timedelta(minutes=5)

        result = evaluate_risk_geometry(
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=quantity,
            direction="BUY",
            symbol=symbol,
            setup_type=setup_type,
            atr_5min=atr_5min,
            atr_timestamp=atr_timestamp,
            trade_timestamp=trade_timestamp,
            max_dollar_risk=max_dollar_risk,
            profile="aggressive",
        )

        # The tactical exception should never be applied for non-qualifying setups
        assert "tactical_stop_applied" not in result, (
            f"tactical_stop_applied found in result for non-qualifying setup "
            f"'{setup_type}' on {symbol} with aggressive profile. "
            f"Result decision: {result.get('decision')}, "
            f"reason_code: {result.get('reason_code')}"
        )

    @given(
        symbol=st_high_beta_symbol,
        entry_price=st_entry_price,
        stop_offset_pct=st.floats(min_value=0.001, max_value=0.05, allow_nan=False, allow_infinity=False),
        target_offset_pct=st.floats(min_value=0.002, max_value=0.10, allow_nan=False, allow_infinity=False),
        atr_5min=st_atr,
        quantity=st.integers(min_value=1, max_value=500),
        max_dollar_risk=st.floats(min_value=100.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        random_setup=st.text(
            alphabet=st.characters(whitelist_categories=("L", "Nd"), whitelist_characters="_-"),
            min_size=3,
            max_size=30,
        ),
    )
    @settings(max_examples=100)
    def test_random_setup_string_never_triggers_tactical_exception(
        self,
        symbol,
        entry_price,
        stop_offset_pct,
        target_offset_pct,
        atr_5min,
        quantity,
        max_dollar_risk,
        random_setup,
    ):
        """Randomly generated setup type strings never get tactical_stop_applied."""
        # Ensure the random string is not accidentally a qualifying or conditional setup
        assume(random_setup.lower() not in [s.lower() for s in ALL_TACTICAL_SETUPS])
        assume(len(random_setup.strip()) > 0)

        # Compute valid trade geometry (LONG direction)
        stop_price = entry_price * (1 - stop_offset_pct)
        target_price = entry_price * (1 + target_offset_pct)

        # Ensure valid geometry
        assume(stop_price > 0)
        assume(target_price > entry_price)
        assume(stop_price < entry_price)

        # Fresh ATR timestamp (within 15 minutes)
        trade_timestamp = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        atr_timestamp = trade_timestamp - timedelta(minutes=5)

        result = evaluate_risk_geometry(
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=quantity,
            direction="BUY",
            symbol=symbol,
            setup_type=random_setup,
            atr_5min=atr_5min,
            atr_timestamp=atr_timestamp,
            trade_timestamp=trade_timestamp,
            max_dollar_risk=max_dollar_risk,
            profile="aggressive",
        )

        # The tactical exception should never be applied for non-qualifying setups
        assert "tactical_stop_applied" not in result, (
            f"tactical_stop_applied found in result for random setup "
            f"'{random_setup}' on {symbol} with aggressive profile. "
            f"Result decision: {result.get('decision')}, "
            f"reason_code: {result.get('reason_code')}"
        )


# ---------------------------------------------------------------------------
# Property 5: Tactical minimum stop computation correctness
# Feature: aggressive-tactical-stop-geometry, Property 5: Tactical minimum stop computation correctness
# ---------------------------------------------------------------------------

import pytest
from utils.risk_geometry_gate import _compute_tactical_min_stop


class TestProperty5TacticalMinStopComputationCorrectness:
    """
    For any positive entry price and positive ATR value, the computed
    tactical_min_stop SHALL equal max(entry_price * min_pct, atr_5min * atr_multiplier)
    where min_pct and atr_multiplier are the configured values for the active profile.

    **Validates: Requirements 3.1, 3.2**
    """

    @given(
        entry_price=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        atr_5min=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_tactical_min_stop_with_configured_values(self, entry_price: float, atr_5min: float):
        """With configured min_pct=0.002 and atr_multiplier=1.0, result equals max of both floors."""
        min_pct = 0.002
        atr_multiplier = 1.0

        result = _compute_tactical_min_stop(
            entry_price=entry_price,
            atr_5min=atr_5min,
            min_pct=min_pct,
            atr_multiplier=atr_multiplier,
        )

        expected = max(entry_price * min_pct, atr_5min * atr_multiplier)
        assert result == pytest.approx(expected), (
            f"Expected max({entry_price} * {min_pct}, {atr_5min} * {atr_multiplier}) = {expected}, got {result}"
        )

    @given(
        entry_price=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        atr_5min=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_pct=st.floats(min_value=0.001, max_value=0.05, allow_nan=False, allow_infinity=False),
        atr_multiplier=st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_tactical_min_stop_with_random_params(
        self, entry_price: float, atr_5min: float, min_pct: float, atr_multiplier: float
    ):
        """With random min_pct and atr_multiplier, result equals max of both floors."""
        result = _compute_tactical_min_stop(
            entry_price=entry_price,
            atr_5min=atr_5min,
            min_pct=min_pct,
            atr_multiplier=atr_multiplier,
        )

        expected = max(entry_price * min_pct, atr_5min * atr_multiplier)
        assert result == pytest.approx(expected), (
            f"Expected max({entry_price} * {min_pct}, {atr_5min} * {atr_multiplier}) = {expected}, got {result}"
        )


# ---------------------------------------------------------------------------
# Property 6: Invalid or stale ATR skips tactical path
# Feature: aggressive-tactical-stop-geometry, Property 6: Invalid or stale ATR skips tactical path
# ---------------------------------------------------------------------------


class TestProperty6InvalidOrStaleATRSkipsTacticalPath:
    """
    For any trade that would otherwise qualify for the tactical exception,
    if the ATR value is None, zero, or negative, OR atr_timestamp is None,
    OR atr_timestamp is older than the parent rule's atr_max_age_minutes
    (15 minutes), the result SHALL NOT contain `tactical_stop_applied` in
    the result dictionary.

    **Validates: Requirements 3.3, 3.4**
    """

    @given(
        symbol=st_high_beta_symbol,
        setup_type=st_qualifying_setup,
        entry_price=st_entry_price,
        stop_offset_pct=st.floats(min_value=0.003, max_value=0.05, allow_nan=False, allow_infinity=False),
        target_offset_pct=st.floats(min_value=0.01, max_value=0.10, allow_nan=False, allow_infinity=False),
        quantity=st.integers(min_value=1, max_value=500),
        atr_value=st.one_of(
            st.none(),
            st.just(0.0),
            st.floats(min_value=-100.0, max_value=-0.001, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=100)
    def test_invalid_atr_value_skips_tactical_path(
        self,
        symbol,
        setup_type,
        entry_price,
        stop_offset_pct,
        target_offset_pct,
        quantity,
        atr_value,
    ):
        """ATR value that is None, zero, or negative prevents tactical exception."""
        # Construct valid LONG trade geometry that would otherwise qualify
        stop_price = entry_price * (1 - stop_offset_pct)
        target_price = entry_price * (1 + target_offset_pct)

        # Ensure valid geometry
        assume(stop_price > 0)
        assume(target_price > entry_price)
        assume(stop_price < entry_price)

        # Use a fresh ATR timestamp (within 15 minutes) — only ATR value is invalid
        trade_timestamp = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        atr_timestamp = trade_timestamp - timedelta(minutes=2)

        result = evaluate_risk_geometry(
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=quantity,
            direction="BUY",
            symbol=symbol,
            setup_type=setup_type,
            atr_5min=atr_value,
            atr_timestamp=atr_timestamp,
            trade_timestamp=trade_timestamp,
            max_dollar_risk=50_000.0,
            profile="aggressive",
            trade_metadata="support bounce near VWAP",
            trade_rationale="Bouncing off support level",
        )

        assert "tactical_stop_applied" not in result, (
            f"tactical_stop_applied should NOT be present when ATR value is "
            f"{atr_value!r}. symbol={symbol}, setup={setup_type}, "
            f"decision={result.get('decision')}, reason_code={result.get('reason_code')}"
        )

    @given(
        symbol=st_high_beta_symbol,
        setup_type=st_qualifying_setup,
        entry_price=st_entry_price,
        atr_5min=st_atr,
        stop_offset_pct=st.floats(min_value=0.003, max_value=0.05, allow_nan=False, allow_infinity=False),
        target_offset_pct=st.floats(min_value=0.01, max_value=0.10, allow_nan=False, allow_infinity=False),
        quantity=st.integers(min_value=1, max_value=500),
    )
    @settings(max_examples=100)
    def test_none_atr_timestamp_skips_tactical_path(
        self,
        symbol,
        setup_type,
        entry_price,
        atr_5min,
        stop_offset_pct,
        target_offset_pct,
        quantity,
    ):
        """atr_timestamp=None prevents tactical exception even with valid ATR value."""
        # Construct valid LONG trade geometry that would otherwise qualify
        stop_price = entry_price * (1 - stop_offset_pct)
        target_price = entry_price * (1 + target_offset_pct)

        # Ensure valid geometry
        assume(stop_price > 0)
        assume(target_price > entry_price)
        assume(stop_price < entry_price)

        trade_timestamp = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)

        result = evaluate_risk_geometry(
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=quantity,
            direction="BUY",
            symbol=symbol,
            setup_type=setup_type,
            atr_5min=atr_5min,
            atr_timestamp=None,  # None timestamp
            trade_timestamp=trade_timestamp,
            max_dollar_risk=50_000.0,
            profile="aggressive",
            trade_metadata="support bounce near VWAP",
            trade_rationale="Bouncing off support level",
        )

        assert "tactical_stop_applied" not in result, (
            f"tactical_stop_applied should NOT be present when atr_timestamp is None. "
            f"symbol={symbol}, setup={setup_type}, atr_5min={atr_5min}, "
            f"decision={result.get('decision')}, reason_code={result.get('reason_code')}"
        )

    @given(
        symbol=st_high_beta_symbol,
        setup_type=st_qualifying_setup,
        entry_price=st_entry_price,
        atr_5min=st_atr,
        stop_offset_pct=st.floats(min_value=0.003, max_value=0.05, allow_nan=False, allow_infinity=False),
        target_offset_pct=st.floats(min_value=0.01, max_value=0.10, allow_nan=False, allow_infinity=False),
        quantity=st.integers(min_value=1, max_value=500),
        stale_minutes=st.floats(min_value=15.01, max_value=120.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_stale_atr_timestamp_skips_tactical_path(
        self,
        symbol,
        setup_type,
        entry_price,
        atr_5min,
        stop_offset_pct,
        target_offset_pct,
        quantity,
        stale_minutes,
    ):
        """atr_timestamp older than 15 minutes prevents tactical exception."""
        # Construct valid LONG trade geometry that would otherwise qualify
        stop_price = entry_price * (1 - stop_offset_pct)
        target_price = entry_price * (1 + target_offset_pct)

        # Ensure valid geometry
        assume(stop_price > 0)
        assume(target_price > entry_price)
        assume(stop_price < entry_price)

        trade_timestamp = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        # ATR timestamp is stale (>15 minutes old relative to trade_timestamp)
        atr_timestamp = trade_timestamp - timedelta(minutes=stale_minutes)

        result = evaluate_risk_geometry(
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=quantity,
            direction="BUY",
            symbol=symbol,
            setup_type=setup_type,
            atr_5min=atr_5min,
            atr_timestamp=atr_timestamp,
            trade_timestamp=trade_timestamp,
            max_dollar_risk=50_000.0,
            profile="aggressive",
            trade_metadata="support bounce near VWAP",
            trade_rationale="Bouncing off support level",
        )

        assert "tactical_stop_applied" not in result, (
            f"tactical_stop_applied should NOT be present when atr_timestamp is stale "
            f"({stale_minutes:.1f} min old, max allowed is 15 min). "
            f"symbol={symbol}, setup={setup_type}, atr_5min={atr_5min}, "
            f"decision={result.get('decision')}, reason_code={result.get('reason_code')}"
        )


# ---------------------------------------------------------------------------
# Property 7: Eligible trades meeting all criteria pass unchanged
# Feature: aggressive-tactical-stop-geometry, Property 7: Eligible trades meeting all criteria pass unchanged
# ---------------------------------------------------------------------------


class TestProperty7EligibleTradesMeetingAllCriteriaPassUnchanged:
    """
    For any trade where: symbol is in HIGH_BETA_CLUSTER, profile is "aggressive",
    setup is in qualifying setups, ATR is valid and fresh, stop_distance >= tactical_min_stop,
    original_rr >= configured min_reward_to_risk (1.25), and dollar_risk <= max_dollar_risk,
    the gate SHALL return decision: "passed_unchanged" with the original entry_price,
    stop_price, target_price, and quantity preserved exactly.

    **Validates: Requirements 2.1, 4.1**
    """

    @given(
        symbol=st_high_beta_symbol,
        setup_type=st_qualifying_setup,
        entry_price=st.floats(min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        atr_5min=st.floats(min_value=0.05, max_value=5.0, allow_nan=False, allow_infinity=False),
        stop_multiplier=st.floats(min_value=1.01, max_value=3.0, allow_nan=False, allow_infinity=False),
        target_multiplier=st.floats(min_value=1.30, max_value=5.0, allow_nan=False, allow_infinity=False),
        quantity=st.integers(min_value=1, max_value=200),
        atr_age_minutes=st.floats(min_value=0.0, max_value=14.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_eligible_trades_pass_unchanged_with_original_geometry(
        self,
        symbol,
        setup_type,
        entry_price,
        atr_5min,
        stop_multiplier,
        target_multiplier,
        quantity,
        atr_age_minutes,
    ):
        """Trades meeting all tactical criteria return passed_unchanged with original values."""
        # Compute tactical_min_stop = max(entry * 0.002, atr * 1.0)
        tactical_min_stop = max(entry_price * 0.002, atr_5min * 1.0)

        # Set stop_distance >= tactical_min_stop (multiplier > 1.0 ensures strict >=)
        stop_distance = tactical_min_stop * stop_multiplier

        # Set target_distance >= stop_distance * 1.25 (use 1.30 min to avoid float edge)
        target_distance = stop_distance * target_multiplier

        # Compute prices for LONG direction
        stop_price = entry_price - stop_distance
        target_price = entry_price + target_distance

        # Ensure valid geometry
        assume(stop_price > 0)
        assume(target_price > entry_price)
        assume(stop_price < entry_price)
        assume(stop_distance > 0)

        # Verify the R:R is truly >= 1.25 after floating point computation
        actual_stop_distance = abs(entry_price - stop_price)
        actual_target_distance = abs(target_price - entry_price)
        assume(actual_stop_distance > 0)
        actual_rr = actual_target_distance / actual_stop_distance
        assume(actual_rr >= 1.25)

        # Verify stop_distance >= tactical_min_stop after float computation
        recomputed_tmin = max(entry_price * 0.002, atr_5min * 1.0)
        assume(actual_stop_distance >= recomputed_tmin)

        # Set max_dollar_risk high enough so dollar_risk <= max_dollar_risk
        dollar_risk = quantity * stop_distance
        max_dollar_risk = dollar_risk * 2.0  # Generous headroom

        assume(max_dollar_risk > 0)

        # Fresh ATR timestamp (within 15 minutes)
        trade_timestamp = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        atr_timestamp = trade_timestamp - timedelta(minutes=atr_age_minutes)

        result = evaluate_risk_geometry(
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=quantity,
            direction="BUY",
            symbol=symbol,
            setup_type=setup_type,
            atr_5min=atr_5min,
            atr_timestamp=atr_timestamp,
            trade_timestamp=trade_timestamp,
            max_dollar_risk=max_dollar_risk,
            profile="aggressive",
        )

        # Assert decision is passed_unchanged
        assert result["decision"] == "passed_unchanged", (
            f"Expected 'passed_unchanged' but got '{result['decision']}'. "
            f"reason_code={result.get('reason_code')}, reason={result.get('reason')}, "
            f"symbol={symbol}, setup={setup_type}, entry={entry_price}, "
            f"stop_distance={stop_distance}, tactical_min={tactical_min_stop}, "
            f"target_distance={target_distance}, rr={target_distance/stop_distance:.3f}"
        )

        # Assert original entry_price preserved
        assert result["entry_price"] == entry_price, (
            f"entry_price changed: expected {entry_price}, got {result['entry_price']}"
        )

        # Assert original stop_price preserved
        assert result["stop_price"] == stop_price, (
            f"stop_price changed: expected {stop_price}, got {result['stop_price']}"
        )

        # Assert original target_price preserved
        assert result["target_price"] == target_price, (
            f"target_price changed: expected {target_price}, got {result['target_price']}"
        )

        # Assert original quantity preserved
        assert result["quantity"] == quantity, (
            f"quantity changed: expected {quantity}, got {result['quantity']}"
        )

        # Assert tactical_stop_applied is True
        assert result.get("tactical_stop_applied") is True, (
            f"Expected tactical_stop_applied=True but got {result.get('tactical_stop_applied')}. "
            f"reason_code={result.get('reason_code')}"
        )


# ---------------------------------------------------------------------------
# Property 8: Tactical metadata atomicity
# Feature: aggressive-tactical-stop-geometry, Property 8: Tactical metadata atomicity
# ---------------------------------------------------------------------------

# Symbols: HIGH_BETA_CLUSTER + some non-cluster symbols
NON_CLUSTER_SYMBOLS = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "SPY", "QQQ"]

st_any_symbol = st.one_of(
    st.sampled_from(sorted(HIGH_BETA_CLUSTER)),
    st.sampled_from(NON_CLUSTER_SYMBOLS),
)

# Profiles: aggressive, conservative, moderate, random strings
st_any_profile = st.one_of(
    st.sampled_from(["aggressive", "conservative", "moderate"]),
    st.text(min_size=1, max_size=20),
    st.none(),
)

# Setup types: qualifying, conditional, non-qualifying, random
st_any_setup = st.one_of(
    st.sampled_from(QUALIFYING_SETUPS),
    st.sampled_from(CONDITIONAL_SETUPS),
    st.sampled_from(NON_QUALIFYING_SETUPS),
    st.text(min_size=1, max_size=30),
    st.none(),
)

# ATR values: valid, None, zero, negative
st_any_atr = st.one_of(
    st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
    st.none(),
    st.just(0.0),
    st.just(-1.0),
)


class TestProperty8TacticalMetadataAtomicity:
    """
    For any gate evaluation result, either all tactical-specific metadata fields
    (tactical_stop_applied, tactical_min_stop_distance) are present in the result,
    or none are present. No partial subset SHALL exist.

    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**
    """

    @given(
        profile=st_any_profile,
        symbol=st_any_symbol,
        setup_type=st_any_setup,
        entry_price=st.floats(min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        stop_offset_pct=st.floats(min_value=0.001, max_value=0.08, allow_nan=False, allow_infinity=False),
        target_offset_pct=st.floats(min_value=0.002, max_value=0.15, allow_nan=False, allow_infinity=False),
        quantity=st.integers(min_value=1, max_value=1000),
        atr_5min=st_any_atr,
        max_dollar_risk=st.floats(min_value=100.0, max_value=100_000.0, allow_nan=False, allow_infinity=False),
        use_fresh_atr=st.booleans(),
        metadata=st.one_of(st.none(), st.text(min_size=0, max_size=100)),
        rationale=st.one_of(st.none(), st.text(min_size=0, max_size=100)),
    )
    @settings(max_examples=100)
    def test_tactical_metadata_atomicity(
        self,
        profile,
        symbol,
        setup_type,
        entry_price,
        stop_offset_pct,
        target_offset_pct,
        quantity,
        atr_5min,
        max_dollar_risk,
        use_fresh_atr,
        metadata,
        rationale,
    ):
        """Either BOTH tactical_stop_applied AND tactical_min_stop_distance are present, or NEITHER is."""
        # Construct trade geometry (LONG direction)
        stop_price = entry_price * (1 - stop_offset_pct)
        target_price = entry_price * (1 + target_offset_pct)

        # Ensure valid geometry basics
        assume(stop_price > 0)
        assume(target_price > entry_price)
        assume(stop_price < entry_price)

        # ATR timestamp: fresh or stale/None
        trade_timestamp = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        if use_fresh_atr and atr_5min is not None and atr_5min > 0:
            atr_timestamp = trade_timestamp - timedelta(minutes=2)
        elif atr_5min is not None and atr_5min > 0:
            # Stale ATR (older than 15 minutes)
            atr_timestamp = trade_timestamp - timedelta(minutes=30)
        else:
            atr_timestamp = None

        result = evaluate_risk_geometry(
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=quantity,
            direction="BUY",
            symbol=symbol,
            setup_type=setup_type,
            atr_5min=atr_5min,
            atr_timestamp=atr_timestamp,
            trade_timestamp=trade_timestamp,
            max_dollar_risk=max_dollar_risk,
            profile=profile,
            trade_metadata=metadata,
            trade_rationale=rationale,
        )

        has_applied = "tactical_stop_applied" in result
        has_min_stop = "tactical_min_stop_distance" in result
        assert has_applied == has_min_stop, (
            "Tactical metadata fields must be atomic (both present or both absent). "
            f"tactical_stop_applied present={has_applied}, "
            f"tactical_min_stop_distance present={has_min_stop}. "
            f"profile={profile}, symbol={symbol}, setup_type={setup_type}, "
            f"decision={result.get('decision')}, reason_code={result.get('reason_code')}"
        )


# ---------------------------------------------------------------------------
# Property 10: Dollar risk and position size constraints apply on tactical path
# Feature: aggressive-tactical-stop-geometry, Property 10: Dollar risk constraints apply on tactical path
# ---------------------------------------------------------------------------


class TestProperty10DollarRiskConstraintsApplyOnTacticalPath:
    """
    For any trade that qualifies for the tactical exception where
    quantity * stop_distance > max_dollar_risk, the gate SHALL NOT return
    tactical_stop_applied: True (the trade must fall through to global
    validation which will reject or adjust).

    **Validates: Requirements 6.5**
    """

    @given(
        symbol=st_high_beta_symbol,
        setup_type=st_qualifying_setup,
        entry_price=st.floats(min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        atr_5min=st.floats(min_value=0.10, max_value=5.0, allow_nan=False, allow_infinity=False),
        quantity=st.integers(min_value=10, max_value=500),
        dollar_risk_fraction=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_dollar_risk_violation_prevents_tactical_pass(
        self,
        symbol,
        setup_type,
        entry_price,
        atr_5min,
        quantity,
        dollar_risk_fraction,
    ):
        """When quantity * stop_distance > max_dollar_risk, tactical_stop_applied must NOT be True."""
        # Compute tactical min stop to ensure stop_distance meets tactical floor
        min_pct = _tactical_cfg["min_pct"]  # 0.002
        atr_multiplier = _tactical_cfg["atr_multiplier"]  # 1.0
        min_rr = _tactical_cfg["min_reward_to_risk"]  # 1.25

        tactical_min_stop = max(entry_price * min_pct, atr_5min * atr_multiplier)

        # Set stop_distance to be at least tactical_min_stop (use 1.1x to ensure it passes)
        stop_distance = tactical_min_stop * 1.1

        # Set target_distance to satisfy R:R >= 1.25 (use 1.5x stop_distance)
        target_distance = stop_distance * 1.5
        assume(target_distance > 0)

        # Construct LONG trade geometry
        stop_price = entry_price - stop_distance
        target_price = entry_price + target_distance

        # Ensure valid geometry
        assume(stop_price > 0)
        assume(target_price > entry_price)
        assume(stop_price < entry_price)

        # Compute actual dollar risk
        actual_dollar_risk = quantity * stop_distance

        # Set max_dollar_risk to be LESS than actual dollar risk (guaranteed violation)
        max_dollar_risk = actual_dollar_risk * dollar_risk_fraction
        assume(max_dollar_risk > 0)
        assume(actual_dollar_risk > max_dollar_risk)

        # Fresh ATR timestamp (within 15 minutes)
        trade_timestamp = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        atr_timestamp = trade_timestamp - timedelta(minutes=2)

        result = evaluate_risk_geometry(
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=quantity,
            direction="BUY",
            symbol=symbol,
            setup_type=setup_type,
            atr_5min=atr_5min,
            atr_timestamp=atr_timestamp,
            trade_timestamp=trade_timestamp,
            max_dollar_risk=max_dollar_risk,
            profile="aggressive",
            trade_metadata="support bounce near VWAP",
            trade_rationale="Bouncing off support level with pullback",
        )

        # The tactical exception should NOT be applied when dollar risk is violated
        assert result.get("tactical_stop_applied") is not True, (
            f"tactical_stop_applied should NOT be True when dollar risk is violated. "
            f"quantity={quantity}, stop_distance={stop_distance:.4f}, "
            f"actual_dollar_risk={actual_dollar_risk:.2f}, "
            f"max_dollar_risk={max_dollar_risk:.2f}, "
            f"symbol={symbol}, setup={setup_type}, "
            f"decision={result.get('decision')}, reason_code={result.get('reason_code')}"
        )


# ---------------------------------------------------------------------------
# Property 9: Tactical failures produce equivalent results to global path
# Feature: aggressive-tactical-stop-geometry, Property 9: Tactical failures produce equivalent results to global path
# ---------------------------------------------------------------------------

from unittest.mock import patch


class TestProperty9TacticalFailuresProduceEquivalentResultsToGlobalPath:
    """
    For any trade that is eligible for the tactical exception but fails tactical
    validation (stop too tight, R:R too low, or dollar risk too high), the gate
    result SHALL produce the same decision, reason_code, adjusted prices/quantity,
    risk fields, and absence of tactical metadata as the global path would produce
    for the same trade inputs.

    **Validates: Requirements 4.2, 4.3**
    """

    @given(
        symbol=st_high_beta_symbol,
        setup_type=st_qualifying_setup,
        entry_price=st.floats(min_value=100.0, max_value=400.0, allow_nan=False, allow_infinity=False),
        atr_5min=st.floats(min_value=0.10, max_value=3.0, allow_nan=False, allow_infinity=False),
        quantity=st.integers(min_value=1, max_value=300),
        max_dollar_risk=st.floats(min_value=50.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        failure_mode=st.sampled_from(["stop_too_tight", "rr_too_low", "dollar_risk_too_high"]),
    )
    @settings(max_examples=100)
    def test_tactical_failure_equivalent_to_global_path(
        self,
        symbol,
        setup_type,
        entry_price,
        atr_5min,
        quantity,
        max_dollar_risk,
        failure_mode,
    ):
        """Trades eligible for tactical exception but failing validation produce same result as global path."""
        # Compute tactical_min_stop = max(entry * 0.002, atr * 1.0)
        tactical_min_stop = max(entry_price * 0.002, atr_5min * 1.0)

        # Generate geometry that is eligible for tactical exception but FAILS validation
        if failure_mode == "stop_too_tight":
            # Stop distance below tactical_min_stop
            stop_distance = tactical_min_stop * 0.5
            # Use a good R:R and dollar risk to isolate the failure
            target_distance = stop_distance * 2.0
            # Ensure dollar risk is within limits
            assume(quantity * stop_distance <= max_dollar_risk)
        elif failure_mode == "rr_too_low":
            # Stop distance is adequate but R:R below 1.25
            stop_distance = tactical_min_stop * 1.5
            # Target distance gives R:R < 1.25 (e.g., 0.8)
            target_distance = stop_distance * 0.8
            assume(quantity * stop_distance <= max_dollar_risk)
        else:  # dollar_risk_too_high
            # Stop distance and R:R are adequate but dollar risk exceeds max
            stop_distance = tactical_min_stop * 1.5
            target_distance = stop_distance * 2.0
            # Force dollar_risk > max_dollar_risk
            dollar_risk = quantity * stop_distance
            assume(dollar_risk > max_dollar_risk)

        # Compute prices for LONG direction
        stop_price = entry_price - stop_distance
        target_price = entry_price + target_distance

        # Ensure valid geometry
        assume(stop_price > 0)
        assume(target_price > entry_price)
        assume(stop_price < entry_price)
        assume(stop_distance > 0)
        assume(target_distance > 0)

        # Fresh ATR timestamp (within 15 minutes)
        trade_timestamp = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        atr_timestamp = trade_timestamp - timedelta(minutes=2)

        # Common kwargs for both calls
        common_kwargs = dict(
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=quantity,
            direction="BUY",
            symbol=symbol,
            setup_type=setup_type,
            atr_5min=atr_5min,
            atr_timestamp=atr_timestamp,
            trade_timestamp=trade_timestamp,
            max_dollar_risk=max_dollar_risk,
            profile="aggressive",
            trade_metadata="support bounce near VWAP",
            trade_rationale="Bouncing off support level with pullback",
        )

        # Run with tactical config present (tactical failure should fall through to global)
        result_with_tactical = evaluate_risk_geometry(**common_kwargs)

        # Run with tactical config removed (pure global path)
        rule = STOP_DISTANCE_RULES["high_beta_mega_cap_intraday"]
        rule_without_tactical = {k: v for k, v in rule.items() if k != "tactical_stop_by_profile"}
        patched_rules = {
            **STOP_DISTANCE_RULES,
            "high_beta_mega_cap_intraday": rule_without_tactical,
        }
        with patch.dict("utils.gate_config.STOP_DISTANCE_RULES", patched_rules):
            result_without_tactical = evaluate_risk_geometry(**common_kwargs)

        # Assert tactical_stop_applied is NOT in either result
        assert "tactical_stop_applied" not in result_with_tactical, (
            f"tactical_stop_applied should NOT be present when tactical validation fails. "
            f"failure_mode={failure_mode}, decision={result_with_tactical.get('decision')}, "
            f"reason_code={result_with_tactical.get('reason_code')}"
        )
        assert "tactical_stop_applied" not in result_without_tactical, (
            f"tactical_stop_applied should NOT be present in global-only path. "
            f"decision={result_without_tactical.get('decision')}"
        )

        # Compare key result fields for equivalence
        assert result_with_tactical["decision"] == result_without_tactical["decision"], (
            f"decision mismatch: tactical_failure={result_with_tactical['decision']}, "
            f"global={result_without_tactical['decision']}, failure_mode={failure_mode}"
        )

        assert result_with_tactical["reason_code"] == result_without_tactical["reason_code"], (
            f"reason_code mismatch: tactical_failure={result_with_tactical['reason_code']}, "
            f"global={result_without_tactical['reason_code']}, failure_mode={failure_mode}"
        )

        assert result_with_tactical["entry_price"] == result_without_tactical["entry_price"], (
            f"entry_price mismatch: {result_with_tactical['entry_price']} vs "
            f"{result_without_tactical['entry_price']}"
        )

        assert result_with_tactical["stop_price"] == result_without_tactical["stop_price"], (
            f"stop_price mismatch: {result_with_tactical['stop_price']} vs "
            f"{result_without_tactical['stop_price']}"
        )

        assert result_with_tactical["target_price"] == result_without_tactical["target_price"], (
            f"target_price mismatch: {result_with_tactical['target_price']} vs "
            f"{result_without_tactical['target_price']}"
        )

        assert result_with_tactical["quantity"] == result_without_tactical["quantity"], (
            f"quantity mismatch: {result_with_tactical['quantity']} vs "
            f"{result_without_tactical['quantity']}"
        )

        # Compare adjusted fields (may be None or numeric)
        assert result_with_tactical.get("adjusted_stop_price") == result_without_tactical.get("adjusted_stop_price"), (
            f"adjusted_stop_price mismatch: {result_with_tactical.get('adjusted_stop_price')} vs "
            f"{result_without_tactical.get('adjusted_stop_price')}"
        )

        assert result_with_tactical.get("adjusted_quantity") == result_without_tactical.get("adjusted_quantity"), (
            f"adjusted_quantity mismatch: {result_with_tactical.get('adjusted_quantity')} vs "
            f"{result_without_tactical.get('adjusted_quantity')}"
        )

        # Verify absence of tactical metadata in both
        assert "tactical_min_stop_distance" not in result_with_tactical, (
            "tactical_min_stop_distance should not be present in tactical failure result"
        )
        assert "tactical_min_stop_distance" not in result_without_tactical, (
            "tactical_min_stop_distance should not be present in global-only result"
        )
