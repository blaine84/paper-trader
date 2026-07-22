"""Unit tests for timeframe authority computation in utils/market_state.py.

Covers: aligned (bullish/bearish), higher_timeframe control with conflict,
intraday leads, confounded, and missing/malformed data edge cases.

Requirements: 17.2
"""

from __future__ import annotations

from utils.market_state import (
    TimeframeAuthority,
    SetupReclassification,
    IfThenTrigger,
    _compute_timeframe_authority,
    _classify_market_state,
    _compute_setup_reclassification,
    _categorize_veto_reason,
    _compute_if_then_triggers,
    _classify_setup_lifecycle,
)


# ─── Timeframe Authority Tests ──────────────────────────────────────────────


def test_timeframe_authority_aligned_bullish():
    """Daily=bullish, 5m=bullish → aligned, no conflict."""
    mtf = {
        "timeframes": {
            "daily": {"trend": "bullish"},
            "5m": {"trend": "bullish"},
        }
    }
    result = _compute_timeframe_authority(mtf)

    assert isinstance(result, TimeframeAuthority)
    assert result.authority == "aligned"
    assert result.higher_timeframe_trend == "bullish"
    assert result.intraday_trend == "bullish"
    assert result.conflict is False
    assert "bullish" in result.reason.lower()


def test_timeframe_authority_aligned_bearish():
    """Daily=bearish, 5m=bearish → aligned, no conflict."""
    mtf = {
        "timeframes": {
            "daily": {"trend": "bearish"},
            "5m": {"trend": "bearish"},
        }
    }
    result = _compute_timeframe_authority(mtf)

    assert result.authority == "aligned"
    assert result.higher_timeframe_trend == "bearish"
    assert result.intraday_trend == "bearish"
    assert result.conflict is False


def test_timeframe_authority_ht_controls():
    """Daily=bearish, 5m=bullish → higher_timeframe, conflict=True."""
    mtf = {
        "timeframes": {
            "daily": {"trend": "bearish"},
            "5m": {"trend": "bullish"},
        }
    }
    result = _compute_timeframe_authority(mtf)

    assert result.authority == "higher_timeframe"
    assert result.higher_timeframe_trend == "bearish"
    assert result.intraday_trend == "bullish"
    assert result.conflict is True
    assert "daily" in result.reason.lower() or "bearish" in result.reason.lower()


def test_timeframe_authority_intraday_leads():
    """Daily=neutral, 5m=bullish → intraday authority, no conflict."""
    mtf = {
        "timeframes": {
            "daily": {"trend": None},
            "5m": {"trend": "bullish"},
        }
    }
    result = _compute_timeframe_authority(mtf)

    assert result.authority == "intraday"
    assert result.higher_timeframe_trend == "neutral"
    assert result.intraday_trend == "bullish"
    assert result.conflict is False


def test_timeframe_authority_confounded():
    """Daily=neutral, 5m=neutral → confounded, no conflict."""
    mtf = {
        "timeframes": {
            "daily": {"trend": None},
            "5m": {"trend": None},
        }
    }
    result = _compute_timeframe_authority(mtf)

    assert result.authority == "confounded"
    assert result.higher_timeframe_trend == "neutral"
    assert result.intraday_trend == "neutral"
    assert result.conflict is False


def test_timeframe_authority_missing_data():
    """Empty dict → confounded, conflict=False (safe default)."""
    result = _compute_timeframe_authority({})

    assert result.authority == "confounded"
    assert result.conflict is False
    assert result.higher_timeframe_trend == "neutral"
    assert result.intraday_trend == "neutral"


# ─── Edge Cases ─────────────────────────────────────────────────────────────


def test_timeframe_authority_none_trends_normalize_to_neutral():
    """None-valued trends normalize to neutral → confounded."""
    mtf = {
        "timeframes": {
            "daily": {"trend": None},
            "5m": {"trend": None},
        }
    }
    result = _compute_timeframe_authority(mtf)

    assert result.authority == "confounded"
    assert result.higher_timeframe_trend == "neutral"
    assert result.intraday_trend == "neutral"
    assert result.conflict is False


def test_timeframe_authority_missing_timeframes_key():
    """Missing 'timeframes' key → confounded, no conflict."""
    mtf = {"some_other_key": "value"}
    result = _compute_timeframe_authority(mtf)

    assert result.authority == "confounded"
    assert result.conflict is False
    assert result.higher_timeframe_trend == "neutral"
    assert result.intraday_trend == "neutral"


def test_timeframe_authority_non_dict_timeframes():
    """Non-dict value for timeframes → confounded via exception handler."""
    mtf = {"timeframes": "not_a_dict"}
    result = _compute_timeframe_authority(mtf)

    assert result.authority == "confounded"
    assert result.conflict is False
    assert result.higher_timeframe_trend == "neutral"
    assert result.intraday_trend == "neutral"


def test_timeframe_authority_ht_directional_intraday_neutral():
    """Daily=bullish, 5m=neutral → higher_timeframe authority, no conflict."""
    mtf = {
        "timeframes": {
            "daily": {"trend": "bullish"},
            "5m": {"trend": None},
        }
    }
    result = _compute_timeframe_authority(mtf)

    assert result.authority == "higher_timeframe"
    assert result.higher_timeframe_trend == "bullish"
    assert result.intraday_trend == "neutral"
    assert result.conflict is False


# ─── Market State Classification Tests ──────────────────────────────────────
# Requirements: 17.1


def _make_signal(
    breakout_status="none",
    pullback_status="none",
    trigger_main_status="active",
    vwap=100.0,
    current_price=100.0,
    bias="bullish",
    agreement="aligned",
):
    """Helper to build a minimal signal dict for classification tests."""
    return {
        "trigger_status": {
            "breakout": {"status": breakout_status},
            "pullback": {"status": pullback_status},
            "status": trigger_main_status,
        },
        "key_levels": {"vwap": vwap},
        "current_price": current_price,
        "multitimeframe_context": {
            "directional_alignment": {
                "bias": bias,
                "agreement": agreement,
            }
        },
    }


def _aligned_authority(direction="bullish"):
    """Aligned timeframe authority (no conflict)."""
    return TimeframeAuthority(
        higher_timeframe_trend=direction,
        intraday_trend=direction,
        authority="aligned",
        conflict=False,
        reason=f"Both timeframes agree: {direction}",
    )


def _ht_conflict_authority():
    """Higher timeframe controls with conflict (daily bearish, intraday bullish)."""
    return TimeframeAuthority(
        higher_timeframe_trend="bearish",
        intraday_trend="bullish",
        authority="higher_timeframe",
        conflict=True,
        reason="Intraday bullish conflicts with daily bearish; daily controls",
    )


def _confounded_authority():
    """Confounded authority — no clear direction."""
    return TimeframeAuthority(
        higher_timeframe_trend="neutral",
        intraday_trend="neutral",
        authority="confounded",
        conflict=False,
        reason="Neither timeframe provides clear direction",
    )


def test_classify_risk_off_suppression():
    """regime=risk_off + bearish alignment → risk_off_suppression."""
    signal = _make_signal(bias="bearish")
    quote = {"price": 100.0}
    result = _classify_market_state(
        signal, quote, {}, _aligned_authority("bearish"), market_regime="risk_off"
    )
    assert result == "risk_off_suppression"


def test_classify_trend_aligned_breakout():
    """Breakout confirmed + aligned authority → trend_aligned_breakout."""
    signal = _make_signal(breakout_status="confirmed", vwap=99.5, current_price=100.0)
    quote = {"price": 100.0}
    result = _classify_market_state(signal, quote, {}, _aligned_authority())
    assert result == "trend_aligned_breakout"


def test_classify_breakout_extended():
    """Breakout confirmed + extended > 1.5% from VWAP → breakout_extended."""
    # Price 102.0, VWAP 100.0 → 2% distance > 1.5% threshold
    signal = _make_signal(breakout_status="confirmed", vwap=100.0, current_price=102.0)
    quote = {"price": 102.0}
    result = _classify_market_state(signal, quote, {}, _aligned_authority())
    assert result == "breakout_extended"


def test_classify_counter_trend_under_resistance():
    """Breakout confirmed + HT conflict → counter_trend_retracement_under_resistance."""
    signal = _make_signal(breakout_status="confirmed", vwap=99.5, current_price=100.0)
    quote = {"price": 100.0}
    result = _classify_market_state(signal, quote, {}, _ht_conflict_authority())
    assert result == "counter_trend_retracement_under_resistance"


def test_classify_compression_under_resistance():
    """Breakout approaching + not aligned → compression_under_resistance."""
    signal = _make_signal(breakout_status="approaching")
    quote = {"price": 100.0}
    result = _classify_market_state(signal, quote, {}, _ht_conflict_authority())
    assert result == "compression_under_resistance"


def test_classify_breakout_retest_watch():
    """Breakout approaching + aligned authority → breakout_retest_watch."""
    signal = _make_signal(breakout_status="approaching")
    quote = {"price": 100.0}
    result = _classify_market_state(signal, quote, {}, _aligned_authority())
    assert result == "breakout_retest_watch"


def test_classify_pullback_validating():
    """Pullback at_level + aligned authority → pullback_validating."""
    signal = _make_signal(pullback_status="at_level")
    quote = {"price": 100.0}
    result = _classify_market_state(signal, quote, {}, _aligned_authority())
    assert result == "pullback_validating"


def test_classify_pullback_failed():
    """Pullback failed → pullback_failed."""
    signal = _make_signal(pullback_status="failed")
    quote = {"price": 100.0}
    result = _classify_market_state(signal, quote, {}, _aligned_authority())
    assert result == "pullback_failed"


def test_classify_confounded_conflicted():
    """Conflicted agreement → confounded."""
    signal = _make_signal(agreement="conflicted")
    quote = {"price": 100.0}
    result = _classify_market_state(signal, quote, {}, _confounded_authority())
    assert result == "confounded"


def test_classify_range_bound_churn():
    """no_trigger + mixed bias → range_bound_churn."""
    signal = _make_signal(trigger_main_status="no_trigger", bias="mixed")
    quote = {"price": 100.0}
    result = _classify_market_state(signal, quote, {}, _confounded_authority())
    assert result == "range_bound_churn"


def test_classify_default_confounded():
    """Fallthrough with no matching conditions → confounded."""
    signal = _make_signal(
        breakout_status="none",
        pullback_status="none",
        trigger_main_status="active",
        bias="bullish",
        agreement="aligned",
    )
    quote = {"price": 100.0}
    result = _classify_market_state(signal, quote, {}, _confounded_authority())
    assert result == "confounded"


def test_classify_malformed_inputs_fails_safe():
    """Missing/None inputs → confounded (fail-safe via exception handler)."""
    # Completely empty inputs
    result = _classify_market_state({}, {}, {}, _confounded_authority())
    assert result == "confounded"

    # None-like signal fields
    result = _classify_market_state(
        {"trigger_status": None}, {}, {}, _confounded_authority()
    )
    assert result == "confounded"

    # Missing nested keys
    result = _classify_market_state(
        {"trigger_status": {"breakout": None}}, {}, {}, _confounded_authority()
    )
    assert result == "confounded"


# ─── Setup Reclassification Tests ───────────────────────────────────────────
# Requirements: 17.3


def _intraday_authority():
    """Intraday authority — bearish HT, bullish intraday, authority=intraday."""
    return TimeframeAuthority(
        higher_timeframe_trend="bearish",
        intraday_trend="bullish",
        authority="intraday",
        conflict=False,
        reason="Intraday bullish with no daily trend",
    )


def test_reclassify_technical_breakout_under_ht_bearish():
    """technical_breakout + HT bearish → counter_trend_retracement_under_resistance, watch_retest."""
    signal = {"signal": "LONG", "setup_type": "technical_breakout"}
    auth = _ht_conflict_authority()
    result = _compute_setup_reclassification(signal, "trend_aligned_breakout", auth)

    assert isinstance(result, SetupReclassification)
    assert result.original_setup_type == "technical_breakout"
    assert result.reclassified_setup_type == "counter_trend_retracement_under_resistance"
    assert result.trade_posture == "watch_retest"


def test_reclassify_breakout_extended_from_vwap():
    """breakout_* prefix + breakout_extended market state → breakout_extended, watch_retest."""
    signal = {"signal": "LONG", "setup_type": "breakout_continuation"}
    auth = _aligned_authority()
    result = _compute_setup_reclassification(signal, "breakout_extended", auth)

    assert isinstance(result, SetupReclassification)
    assert result.original_setup_type == "breakout_continuation"
    assert result.reclassified_setup_type == "breakout_extended"
    assert result.trade_posture == "watch_retest"


def test_reclassify_vwap_reclaim_ht_bearish():
    """vwap_reclaim + HT bearish + non-breakout state → compression_under_resistance, watch_long_trigger."""
    signal = {"signal": "LONG", "setup_type": "vwap_reclaim"}
    auth = _ht_conflict_authority()
    result = _compute_setup_reclassification(signal, "compression_under_resistance", auth)

    assert isinstance(result, SetupReclassification)
    assert result.original_setup_type == "vwap_reclaim"
    assert result.reclassified_setup_type == "compression_under_resistance"
    assert result.trade_posture == "watch_long_trigger"


def test_reclassify_gap_and_go_counter_trend():
    """gap_and_go + HT bearish + intraday authority → counter_trend, veto_long."""
    signal = {"signal": "LONG", "setup_type": "gap_and_go"}
    auth = _intraday_authority()
    result = _compute_setup_reclassification(signal, "trend_aligned_breakout", auth)

    assert isinstance(result, SetupReclassification)
    assert result.original_setup_type == "gap_and_go"
    assert result.reclassified_setup_type == "counter_trend_retracement_under_resistance"
    assert result.trade_posture == "veto_long"


def test_no_reclassification_when_aligned():
    """Aligned authority + non-contradictory setup → no reclassification (None)."""
    signal = {"signal": "LONG", "setup_type": "technical_breakout"}
    auth = _aligned_authority()
    result = _compute_setup_reclassification(signal, "trend_aligned_breakout", auth)

    assert result is None


def test_no_reclassification_when_hold():
    """HOLD signal → no reclassification regardless of state/authority."""
    signal = {"signal": "HOLD", "setup_type": "technical_breakout"}
    auth = _ht_conflict_authority()
    result = _compute_setup_reclassification(signal, "counter_trend_retracement_under_resistance", auth)

    assert result is None


def test_no_reclassification_malformed_inputs():
    """Missing/None fields → None (fail-open via exception handler)."""
    # Missing signal key
    result = _compute_setup_reclassification({}, "confounded", _aligned_authority())
    assert result is None

    # None setup_type
    result = _compute_setup_reclassification(
        {"signal": "LONG", "setup_type": None}, "confounded", _aligned_authority()
    )
    assert result is None


# ─── Veto Reason Categorization Tests ───────────────────────────────────────
# Requirements: 17.3


def test_categorize_veto_higher_timeframe():
    """Veto reason with 'higher timeframe' keyword → higher_timeframe_resistance."""
    signal = {"llm_veto_reason": "higher timeframe resistance is strong"}
    result = _categorize_veto_reason(signal)
    assert result == "higher_timeframe_resistance"


def test_categorize_veto_extended_from_vwap():
    """Veto reason with 'overextended' keyword → extended_from_vwap."""
    signal = {"llm_veto_reason": "stock is overextended from VWAP"}
    result = _categorize_veto_reason(signal)
    assert result == "extended_from_vwap"


def test_categorize_veto_no_match_returns_none():
    """Veto reason with no matching keywords → None."""
    signal = {"llm_veto_reason": "no clear reason"}
    result = _categorize_veto_reason(signal)
    assert result is None


# ─── If-Then Trigger Computation Tests ──────────────────────────────────────
# Requirements: 17.4


def test_triggers_long_breakout_when_below_resistance():
    """Resistance exists + authority != higher_timeframe → long_breakout trigger fires."""
    signal = {"key_levels": {"resistance": 110.0, "support": None, "vwap": None}}
    result = _compute_if_then_triggers(signal, current_price=105.0, timeframe_authority=_aligned_authority())

    assert len(result) >= 1
    breakout = next(t for t in result if t.id == "long_breakout")
    assert isinstance(breakout, IfThenTrigger)
    assert breakout.trade_posture == "watch_long_trigger"
    assert breakout.threshold == 110.0


def test_triggers_pullback_hold_when_near_support():
    """Support exists + aligned authority + price within 3% of support → pullback_hold fires."""
    # Price 102.0, support 100.0 → 2% distance (within 3% threshold)
    signal = {"key_levels": {"resistance": None, "support": 100.0, "vwap": None}}
    result = _compute_if_then_triggers(signal, current_price=102.0, timeframe_authority=_aligned_authority())

    assert len(result) >= 1
    pullback = next(t for t in result if t.id == "pullback_hold")
    assert isinstance(pullback, IfThenTrigger)
    assert pullback.trade_posture == "watch_long_trigger"
    assert pullback.threshold == 100.0


def test_triggers_vwap_veto_when_above_vwap():
    """VWAP exists + price > 1.5% above VWAP → vwap_veto fires."""
    # Price 102.0, VWAP 100.0 → 2% above (exceeds 1.5% threshold)
    signal = {"key_levels": {"resistance": None, "support": None, "vwap": 100.0}}
    result = _compute_if_then_triggers(signal, current_price=102.0, timeframe_authority=_aligned_authority())

    assert len(result) >= 1
    veto = next(t for t in result if t.id == "vwap_veto")
    assert isinstance(veto, IfThenTrigger)
    assert veto.trade_posture == "veto_long"
    assert veto.threshold == 100.0


def test_triggers_short_rejection_when_ht_bearish():
    """HT trend bearish + resistance exists → short_rejection fires."""
    signal = {"key_levels": {"resistance": 110.0, "support": None, "vwap": None}}
    result = _compute_if_then_triggers(signal, current_price=105.0, timeframe_authority=_ht_conflict_authority())

    assert len(result) >= 1
    short = next(t for t in result if t.id == "short_rejection")
    assert isinstance(short, IfThenTrigger)
    assert short.trade_posture == "watch_short_trigger"
    assert short.threshold == 110.0


def test_triggers_empty_when_no_levels():
    """No key levels populated → empty trigger list."""
    signal = {"key_levels": {"resistance": None, "support": None, "vwap": None}}
    result = _compute_if_then_triggers(signal, current_price=100.0, timeframe_authority=_aligned_authority())

    assert result == []


def test_triggers_skip_pullback_when_far_from_support():
    """Support exists but price > 3% away → pullback_hold does NOT fire."""
    # Price 110.0, support 100.0 → 10% distance (exceeds 3% threshold)
    signal = {"key_levels": {"resistance": None, "support": 100.0, "vwap": None}}
    result = _compute_if_then_triggers(signal, current_price=110.0, timeframe_authority=_aligned_authority())

    pullback_ids = [t.id for t in result if t.id == "pullback_hold"]
    assert pullback_ids == []


def test_triggers_null_threshold_acceptable():
    """None current_price → returns empty list gracefully."""
    signal = {"key_levels": {"resistance": 110.0, "support": 100.0, "vwap": 105.0}}
    result = _compute_if_then_triggers(signal, current_price=None, timeframe_authority=_aligned_authority())

    assert result == []


def test_triggers_capped_at_four():
    """All 4 triggers fire simultaneously → len(result) == 4."""
    # Setup: resistance exists (long_breakout + short_rejection),
    # support within 3% (pullback_hold), price > 1.5% above VWAP (vwap_veto)
    # Need: authority in (aligned, intraday) for pullback, ht_trend bearish for short_rejection
    # Use intraday authority with bearish HT to satisfy both pullback and short_rejection
    auth = _intraday_authority()  # bearish HT, bullish intraday, authority=intraday

    # Price 102, support 100 (2% away, within 3%), VWAP 100 (2% above, > 1.5%)
    # Resistance 110 (authority != higher_timeframe → long_breakout; ht bearish → short_rejection)
    signal = {"key_levels": {"resistance": 110.0, "support": 100.0, "vwap": 100.0}}
    result = _compute_if_then_triggers(signal, current_price=102.0, timeframe_authority=auth)

    assert len(result) == 4
    trigger_ids = {t.id for t in result}
    assert trigger_ids == {"long_breakout", "pullback_hold", "vwap_veto", "short_rejection"}


# ─── Setup Lifecycle Classification Tests ───────────────────────────────────
# Requirements: 17.5


def test_lifecycle_confounded_no_setup():
    """market_state=confounded, any direction → no_setup."""
    result = _classify_setup_lifecycle("confounded", "LONG", _aligned_authority())
    assert result == "no_setup"

    result = _classify_setup_lifecycle("confounded", "SHORT", _confounded_authority())
    assert result == "no_setup"

    result = _classify_setup_lifecycle("confounded", "HOLD", _ht_conflict_authority())
    assert result == "no_setup"


def test_lifecycle_compression_watch():
    """market_state=compression_under_resistance, direction=LONG → compression_watch."""
    result = _classify_setup_lifecycle(
        "compression_under_resistance", "LONG", _aligned_authority()
    )
    assert result == "compression_watch"


def test_lifecycle_breakout_watch():
    """market_state=breakout_retest_watch, direction=LONG, authority=aligned → breakout_watch."""
    result = _classify_setup_lifecycle(
        "breakout_retest_watch", "LONG", _aligned_authority()
    )
    assert result == "breakout_watch"


def test_lifecycle_breakout_extended_wait_retest():
    """market_state=breakout_extended → breakout_confirmed_wait_retest."""
    result = _classify_setup_lifecycle(
        "breakout_extended", "LONG", _aligned_authority()
    )
    assert result == "breakout_confirmed_wait_retest"


def test_lifecycle_trend_aligned_directional_activation_pending():
    """market_state=trend_aligned_breakout, direction=LONG, authority=aligned → activation_pending."""
    result = _classify_setup_lifecycle(
        "trend_aligned_breakout", "LONG", _aligned_authority()
    )
    assert result == "activation_pending"


def test_lifecycle_trend_aligned_hold_breakout_watch():
    """market_state=trend_aligned_breakout, direction=HOLD → breakout_watch."""
    result = _classify_setup_lifecycle(
        "trend_aligned_breakout", "HOLD", _aligned_authority()
    )
    assert result == "breakout_watch"


def test_lifecycle_pullback_validating():
    """market_state=pullback_validating, direction=LONG, authority=aligned → pullback_validating."""
    result = _classify_setup_lifecycle(
        "pullback_validating", "LONG", _aligned_authority()
    )
    assert result == "pullback_validating"


def test_lifecycle_pullback_failed_invalidated():
    """market_state=pullback_failed → invalidated."""
    result = _classify_setup_lifecycle(
        "pullback_failed", "LONG", _aligned_authority()
    )
    assert result == "invalidated"


# ─── Fail-Open & Happy-Path Integration Tests ──────────────────────────────
# Requirements: 17.6

from unittest.mock import patch

from utils.market_state import (
    compute_market_state,
    MarketStateResult,
    VALID_MARKET_STATES,
    VALID_LIFECYCLE_STATES,
)


def test_compute_market_state_fails_open_on_exception():
    """Injected error in internal function → safe defaults returned (never raises)."""
    signal = {"signal": "LONG", "setup_type": "technical_breakout", "multitimeframe_context": {}}
    with patch("utils.market_state._compute_timeframe_authority", side_effect=RuntimeError("test error")):
        result = compute_market_state(signal, {"price": 100.0}, {})

    assert isinstance(result, MarketStateResult)
    assert result.market_state == "confounded"
    assert result.setup_lifecycle_state == "no_setup"
    assert result.if_then_triggers == []
    assert result.setup_reclassification is None
    assert result.veto_reason_category is None
    assert result.timeframe_authority.authority == "confounded"
    assert result.timeframe_authority.conflict is False


def test_compute_market_state_returns_valid_result():
    """Happy path — full signal produces valid MarketStateResult with all fields."""
    signal = {
        "signal": "LONG",
        "setup_type": "technical_breakout",
        "current_price": 100.0,
        "key_levels": {"resistance": 101.0, "support": 99.0, "vwap": 99.5},
        "trigger_status": {
            "breakout": {"status": "confirmed"},
            "pullback": {"status": "none"},
            "status": "active",
        },
        "multitimeframe_context": {
            "timeframes": {
                "daily": {"trend": "bullish"},
                "5m": {"trend": "bullish"},
            },
            "directional_alignment": {"bias": "bullish", "agreement": "aligned"},
        },
    }
    quote = {"price": 100.0}

    result = compute_market_state(signal, quote, {})

    assert isinstance(result, MarketStateResult)
    assert result.market_state in VALID_MARKET_STATES
    assert result.setup_lifecycle_state in VALID_LIFECYCLE_STATES
    assert result.timeframe_authority.authority == "aligned"
    assert isinstance(result.if_then_triggers, list)
    # to_dict() should not raise
    d = result.to_dict()
    assert "market_state" in d
    assert "timeframe_authority" in d


# ─── Signal Enrichment Integration Test ─────────────────────────────────────
# Requirements: 9.3 — Verifies signal enrichment adds market-state fields


def test_signal_enrichment_adds_market_state_fields():
    """When MARKET_STATE_MODE != disabled, signal is enriched with market-state fields."""
    signal = {
        "signal": "LONG",
        "setup_type": "technical_breakout",
        "current_price": 100.0,
        "key_levels": {"resistance": 101.0, "support": 99.0, "vwap": 99.5},
        "trigger_status": {
            "breakout": {"status": "confirmed"},
            "pullback": {"status": "none"},
            "status": "active",
        },
        "multitimeframe_context": {
            "timeframes": {
                "daily": {"trend": "bullish"},
                "5m": {"trend": "bullish"},
            },
            "directional_alignment": {"bias": "bullish", "agreement": "aligned"},
        },
    }
    quote = {"price": 100.0}
    indicators = {}

    # Simulate what the analyst does
    result = compute_market_state(signal, quote, indicators)

    # Enrich signal dict (same as analyst.py does)
    signal["market_state"] = result.market_state
    signal["timeframe_authority"] = result.timeframe_authority.to_dict()
    signal["setup_reclassification"] = (
        result.setup_reclassification.to_dict() if result.setup_reclassification else None
    )
    signal["if_then_triggers"] = [t.to_dict() for t in result.if_then_triggers]
    signal["setup_lifecycle_state"] = result.setup_lifecycle_state
    signal["veto_reason_category"] = result.veto_reason_category

    # Verify all expected fields present
    assert "market_state" in signal
    assert "timeframe_authority" in signal
    assert "setup_reclassification" in signal
    assert "if_then_triggers" in signal
    assert "setup_lifecycle_state" in signal
    assert "veto_reason_category" in signal

    # Verify types
    assert isinstance(signal["market_state"], str)
    assert isinstance(signal["timeframe_authority"], dict)
    assert isinstance(signal["if_then_triggers"], list)
    assert isinstance(signal["setup_lifecycle_state"], str)

    # Verify valid values
    assert signal["market_state"] in VALID_MARKET_STATES
    assert signal["setup_lifecycle_state"] in VALID_LIFECYCLE_STATES

    # Verify timeframe_authority has expected keys
    assert "authority" in signal["timeframe_authority"]
    assert "conflict" in signal["timeframe_authority"]

    # Verify signal["signal"] is NOT modified (safety invariant)
    assert signal["signal"] == "LONG"


# ─── Schema Creation Tests ───────────────────────────────────────────────────


def test_ensure_watch_candidate_tables_creates_schema():
    """In-memory SQLite → run _ensure_watch_candidate_tables → table and indexes exist."""
    from sqlalchemy import create_engine, inspect as sa_inspect
    from orchestrator import _ensure_watch_candidate_tables

    eng = create_engine("sqlite:///:memory:")
    inspector = sa_inspect(eng)

    _ensure_watch_candidate_tables(eng, inspector)

    # Refresh inspector to pick up new table
    inspector = sa_inspect(eng)
    assert inspector.has_table("watch_candidates")

    # Verify columns
    columns = {col["name"] for col in inspector.get_columns("watch_candidates")}
    expected_cols = {
        "watch_id", "symbol", "created_at", "updated_at", "expires_at",
        "source_cycle_id", "profile_id", "market_state", "setup_lifecycle_state",
        "timeframe_authority_json", "direction_watch", "trade_posture",
        "activation_conditions_json", "invalidation_conditions_json",
        "key_levels_json", "trigger_status_json", "reason",
        "source_signal_snapshot_json", "state", "state_changed_at", "outcome_json",
    }
    assert expected_cols.issubset(columns)

    # Verify indexes exist
    indexes = {idx["name"] for idx in inspector.get_indexes("watch_candidates")}
    assert "idx_watch_candidates_active" in indexes
    assert "idx_watch_candidates_expires" in indexes
    assert "idx_watch_candidates_active_unique" in indexes


def test_ensure_watch_candidate_tables_idempotent():
    """Calling _ensure_watch_candidate_tables twice does not raise."""
    from sqlalchemy import create_engine, inspect as sa_inspect
    from orchestrator import _ensure_watch_candidate_tables

    eng = create_engine("sqlite:///:memory:")
    inspector = sa_inspect(eng)

    _ensure_watch_candidate_tables(eng, inspector)
    # Refresh inspector and call again — should be a no-op
    inspector = sa_inspect(eng)
    _ensure_watch_candidate_tables(eng, inspector)

    assert inspector.has_table("watch_candidates")
