from agents.analyst import (
    build_deterministic_sanity_prompt_context,
    compute_deterministic_signal_sanity,
    enforce_veto_accountability,
)
from agents.portfolio_manager import (
    format_entry_signal_filter_summary,
    summarize_entry_signal_filter,
)


def test_deterministic_sanity_flags_hold_conflict_on_clean_long_setup():
    signal = {"signal": "HOLD", "relative_volume": 1.8}
    quote = {"price": 101.0, "change_pct": 0.8}
    indicators = {
        "vwap": 100.0,
        "trend": "bullish",
        "macd_bias": "bullish",
        "rsi": 58,
    }

    sanity = compute_deterministic_signal_sanity(signal, quote, indicators)

    assert sanity["bias"] == "LONG"
    assert sanity["score"] >= 3
    assert sanity["conflict"] is True
    assert "relative_volume_confirming_1.80x" in sanity["reasons"]


def test_deterministic_sanity_does_not_flag_ambiguous_hold():
    signal = {"signal": "HOLD"}
    quote = {"price": 100.01, "change_pct": 0.02}
    indicators = {"vwap": 100.0, "trend": "neutral", "macd_bias": "neutral", "rsi": 49}

    sanity = compute_deterministic_signal_sanity(signal, quote, indicators)

    assert sanity["bias"] == "HOLD"
    assert sanity["conflict"] is False


def test_enforce_veto_accountability_marks_missing_veto_reason():
    signal = {
        "signal": "HOLD",
        "deterministic_sanity": {
            "conflict": True,
            "llm_signal": "HOLD",
            "bias": "SHORT",
            "score": -4,
            "reasons": ["price_below_vwap_-0.50%"],
        },
    }

    enforced = enforce_veto_accountability(signal)

    assert enforced["llm_veto_required"] is True
    assert enforced["llm_veto_present"] is False
    assert enforced["llm_veto_missing"] is True
    assert "MISSING_LLM_VETO_REASON" in enforced["llm_veto_reason"]


def test_enforce_veto_accountability_accepts_concrete_veto_reason():
    signal = {
        "signal": "HOLD",
        "llm_veto_reason": "VWAP break is occurring on thin relative volume and directly into prior-day support.",
        "veto_evidence": ["relative_volume=0.42", "prior_day_support=nearby"],
        "deterministic_sanity": {
            "conflict": True,
            "llm_signal": "HOLD",
            "bias": "SHORT",
            "score": -4,
        },
    }

    enforced = enforce_veto_accountability(signal)

    assert enforced["llm_veto_required"] is True
    assert enforced["llm_veto_present"] is True
    assert enforced["llm_veto_missing"] is False


def test_deterministic_sanity_prompt_context_demands_veto_for_directional_precheck():
    text = build_deterministic_sanity_prompt_context({
        "bias": "SHORT",
        "score": -5,
        "reasons": ["price_below_vwap_-0.60%"],
    })

    assert "Deterministic sanity favors SHORT" in text
    assert "llm_veto_reason" in text


def test_pm_filter_summary_explains_all_hold_batch_and_sanity_conflicts():
    signals = {
        "SPY": {
            "signal": "HOLD",
            "strength": "weak",
            "confidence": "low",
            "setup_type": "trend_pullback",
            "deterministic_sanity": {
                "conflict": True,
                "llm_signal": "HOLD",
                "bias": "LONG",
                "score": 4,
                "reasons": ["price_above_vwap_0.40%"],
            },
        },
        "QQQ": {
            "signal": "HOLD",
            "strength": "weak",
            "confidence": "low",
            "setup_type": "momentum_fade",
        },
    }

    summary = summarize_entry_signal_filter(signals, held_symbols=set(), min_signal_strength="moderate")
    text = format_entry_signal_filter_summary(summary)

    assert summary["eligible"] == 0
    assert summary["hold"] == 2
    assert summary["direction_counts"] == {"HOLD": 2}
    assert summary["strength_counts"] == {"weak": 2}
    assert summary["confidence_counts"] == {"low": 2}
    assert summary["setup_counts"] == {"trend_pullback": 1, "momentum_fade": 1}
    assert summary["sanity_conflicts"][0]["symbol"] == "SPY"
    assert "eligible=0" in text
    assert "sanity_conflicts" in text


def test_pm_filter_summary_counts_below_threshold_separately():
    signals = {
        "AMD": {"signal": "LONG", "strength": "weak", "confidence": "medium", "setup_type": "vwap_reclaim"},
        "NVDA": {"signal": "SHORT", "strength": "moderate", "confidence": "medium", "setup_type": "momentum_fade"},
    }

    summary = summarize_entry_signal_filter(signals, held_symbols=set(), min_signal_strength="moderate")

    assert summary["eligible"] == 1
    assert summary["below_threshold"] == 1
    assert summary["hold"] == 0
