"""Property-based tests for market state module.

Uses Hypothesis to verify universal invariants:
- All classification functions return valid enum members
- compute_market_state() never raises (fail-open guarantee)
- Trigger lists are always well-formed

**Validates: Requirements 17.7**
"""

from __future__ import annotations

from hypothesis import given, strategies as st, settings, assume
from utils.market_state import (
    TimeframeAuthority,
    IfThenTrigger,
    MarketStateResult,
    VALID_MARKET_STATES,
    VALID_LIFECYCLE_STATES,
    VALID_TRADE_POSTURES,
    _compute_timeframe_authority,
    _classify_market_state,
    _compute_if_then_triggers,
    _classify_setup_lifecycle,
    compute_market_state,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

trend_values = st.sampled_from(["bullish", "bearish", None, "neutral", "unknown", ""])
authority_values = st.sampled_from(["aligned", "higher_timeframe", "intraday", "confounded"])
market_state_values = st.sampled_from(list(VALID_MARKET_STATES) + ["unknown", "", None])
direction_values = st.sampled_from(["LONG", "SHORT", "HOLD", "", None, "long", "BUY"])

price_values = st.one_of(
    st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
    st.none(),
)


def mtf_context_strategy():
    """Generate arbitrary multi-timeframe context dicts."""
    return st.one_of(
        st.fixed_dictionaries({
            "timeframes": st.fixed_dictionaries({
                "daily": st.fixed_dictionaries({"trend": trend_values}),
                "5m": st.fixed_dictionaries({"trend": trend_values}),
            })
        }),
        st.just({}),
        st.just({"timeframes": "invalid"}),
        st.just({"timeframes": {}}),
    )


def signal_strategy():
    """Generate arbitrary signal dicts."""
    return st.fixed_dictionaries({
        "signal": direction_values,
        "setup_type": st.sampled_from(["technical_breakout", "gap_and_go", "vwap_reclaim", "momentum_fade", "news_breakout", "", None]),
        "current_price": price_values,
        "key_levels": st.fixed_dictionaries({
            "resistance": price_values,
            "support": price_values,
            "vwap": price_values,
        }),
        "trigger_status": st.fixed_dictionaries({
            "breakout": st.fixed_dictionaries({"status": st.sampled_from(["confirmed", "approaching", "none", "", None])}),
            "pullback": st.fixed_dictionaries({"status": st.sampled_from(["at_level", "holding_above_level", "failed", "none", "", None])}),
            "status": st.sampled_from(["active", "no_trigger", "", None]),
        }),
        "multitimeframe_context": mtf_context_strategy(),
        "llm_veto_reason": st.one_of(st.none(), st.text(max_size=100)),
    })


def timeframe_authority_strategy():
    """Generate arbitrary TimeframeAuthority instances."""
    return st.builds(
        TimeframeAuthority,
        higher_timeframe_trend=st.sampled_from(["bullish", "bearish", "neutral"]),
        intraday_trend=st.sampled_from(["bullish", "bearish", "neutral"]),
        authority=authority_values,
        conflict=st.booleans(),
        reason=st.text(max_size=50),
    )


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


@given(mtf_context=mtf_context_strategy())
@settings(max_examples=200)
def test_prop_timeframe_authority_always_valid(mtf_context):
    """_compute_timeframe_authority always returns valid authority and valid conflict.

    **Validates: Requirements 17.7**
    """
    result = _compute_timeframe_authority(mtf_context)

    assert isinstance(result, TimeframeAuthority)
    assert result.authority in ("aligned", "higher_timeframe", "intraday", "confounded")
    assert result.higher_timeframe_trend in ("bullish", "bearish", "neutral")
    assert result.intraday_trend in ("bullish", "bearish", "neutral")
    assert isinstance(result.conflict, bool)


@given(
    signal=signal_strategy(),
    quote_price=price_values,
    authority=timeframe_authority_strategy(),
    regime=st.sampled_from(["risk_off", None, "", "bullish"]),
)
@settings(max_examples=200)
def test_prop_classify_market_state_always_valid_enum(signal, quote_price, authority, regime):
    """_classify_market_state always returns a valid market state enum member.

    **Validates: Requirements 17.7**
    """
    quote = {"price": quote_price} if quote_price else {}
    result = _classify_market_state(signal, quote, {}, authority, market_regime=regime)

    assert result in VALID_MARKET_STATES


@given(
    signal=signal_strategy(),
    current_price=price_values,
    authority=timeframe_authority_strategy(),
)
@settings(max_examples=200)
def test_prop_if_then_triggers_always_well_formed(signal, current_price, authority):
    """_compute_if_then_triggers always returns a list of well-formed IfThenTrigger objects.

    **Validates: Requirements 17.7**
    """
    result = _compute_if_then_triggers(signal, current_price, authority)

    assert isinstance(result, list)
    assert len(result) <= 4
    for trigger in result:
        assert isinstance(trigger, IfThenTrigger)
        assert trigger.id in ("long_breakout", "pullback_hold", "vwap_veto", "short_rejection")
        assert trigger.trade_posture in VALID_TRADE_POSTURES


@given(signal=signal_strategy(), quote_price=price_values)
@settings(max_examples=200)
def test_prop_compute_market_state_never_raises(signal, quote_price):
    """compute_market_state() never raises - always returns MarketStateResult.

    **Validates: Requirements 17.7**
    """
    quote = {"price": quote_price} if quote_price else {}
    result = compute_market_state(signal, quote, {})

    assert isinstance(result, MarketStateResult)
    assert result.market_state in VALID_MARKET_STATES
    assert result.setup_lifecycle_state in VALID_LIFECYCLE_STATES
    assert isinstance(result.if_then_triggers, list)
    # to_dict() must not raise
    d = result.to_dict()
    assert isinstance(d, dict)
