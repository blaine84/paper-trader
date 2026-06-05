from agents.analyst import (
    build_deterministic_sanity_prompt_context,
    compute_deterministic_signal_sanity,
    enforce_veto_accountability,
    repair_missing_veto_contract,
    sanitize_analyst_key_levels,
    validate_candle_indicator_alignment,
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


def test_enforce_veto_accountability_requires_evidence_with_reason():
    signal = {
        "signal": "HOLD",
        "llm_veto_reason": "Price is extended above resistance.",
        "veto_evidence": [],
        "deterministic_sanity": {
            "conflict": True,
            "llm_signal": "HOLD",
            "bias": "LONG",
            "score": 4,
        },
    }

    enforced = enforce_veto_accountability(signal)

    assert enforced["llm_veto_required"] is True
    assert enforced["llm_veto_present"] is False
    assert enforced["llm_veto_missing"] is True
    assert enforced["llm_veto_contract_error"] == "missing_veto_evidence"


def test_repair_missing_veto_contract_accepts_deterministic_direction(monkeypatch):
    monkeypatch.setattr(
        "agents.analyst.call_llm",
        lambda *args, **kwargs: """
        {
          "signal": "LONG",
          "strength": "moderate",
          "confidence": "medium",
          "llm_veto_reason": null,
          "veto_evidence": []
        }
        """,
    )
    signal = {
        "symbol": "TSLA",
        "signal": "HOLD",
        "strength": "weak",
        "confidence": "low",
        "deterministic_sanity": {
            "conflict": True,
            "llm_signal": "HOLD",
            "bias": "LONG",
            "score": 5,
            "reasons": ["price_above_vwap_0.20%", "relative_volume_confirming_3.00x"],
        },
        "llm_veto_required": True,
        "llm_veto_present": False,
        "llm_veto_missing": True,
    }

    repaired = repair_missing_veto_contract(signal, "TSLA")

    assert repaired["signal"] == "LONG"
    assert repaired["strength"] == "moderate"
    assert repaired["confidence"] == "medium"
    assert repaired["deterministic_sanity"]["conflict"] is False
    assert repaired["llm_veto_required"] is False
    assert repaired["veto_contract_repaired"] is True
    assert repaired["veto_repair_method"] == "primary_llm"


def test_repair_missing_veto_contract_accepts_justified_hold(monkeypatch):
    monkeypatch.setattr(
        "agents.analyst.call_llm",
        lambda *args, **kwargs: """
        {
          "signal": "HOLD",
          "strength": "weak",
          "confidence": "medium",
          "llm_veto_reason": "Price is directly below prior-day resistance on thin volume.",
          "veto_evidence": ["relative_volume=0.42", "distance_to_prior_high=0.10%"]
        }
        """,
    )
    signal = {
        "symbol": "SPY",
        "signal": "HOLD",
        "strength": "weak",
        "confidence": "low",
        "deterministic_sanity": {
            "conflict": True,
            "llm_signal": "HOLD",
            "bias": "LONG",
            "score": 4,
            "reasons": ["price_above_vwap_0.30%"],
        },
        "llm_veto_required": True,
        "llm_veto_present": False,
        "llm_veto_missing": True,
    }

    repaired = repair_missing_veto_contract(signal, "SPY")

    assert repaired["signal"] == "HOLD"
    assert repaired["llm_veto_present"] is True
    assert repaired["llm_veto_missing"] is False
    assert repaired["veto_contract_repaired"] is True


def test_repair_missing_veto_contract_quarantines_invalid_retry(monkeypatch):
    monkeypatch.setattr(
        "agents.analyst.call_llm",
        lambda *args, **kwargs: """
        {
          "signal": "HOLD",
          "strength": "weak",
          "confidence": "low",
          "llm_veto_reason": "Still uncertain.",
          "veto_evidence": []
        }
        """,
    )
    signal = {
        "symbol": "MU",
        "signal": "HOLD",
        "strength": "weak",
        "confidence": "low",
        "deterministic_sanity": {
            "conflict": True,
            "llm_signal": "HOLD",
            "bias": "SHORT",
            "score": -4,
            "reasons": ["price_below_vwap_-0.40%"],
        },
        "llm_veto_required": True,
        "llm_veto_present": False,
        "llm_veto_missing": True,
    }

    repaired = repair_missing_veto_contract(signal, "MU")

    assert repaired["signal"] == "HOLD"
    assert repaired["veto_contract_repair_failed"] is True
    assert repaired["analyst_contract_failure"] == "missing_veto_after_primary_retry"
    assert repaired["llm_veto_missing"] is True


def test_repair_missing_veto_contract_does_not_override_mitigation(monkeypatch):
    def unexpected_call(*args, **kwargs):
        raise AssertionError("repair LLM should not be called for mitigated signals")

    monkeypatch.setattr("agents.analyst.call_llm", unexpected_call)
    signal = {
        "symbol": "AMD",
        "signal": "HOLD",
        "original_signal": "LONG",
        "mitigation": {"setup_type": "news_breakout", "level": 2},
        "deterministic_sanity": {
            "conflict": True,
            "llm_signal": "HOLD",
            "bias": "LONG",
            "score": 5,
        },
        "llm_veto_required": True,
        "llm_veto_present": False,
        "llm_veto_missing": True,
    }

    repaired = repair_missing_veto_contract(signal, "AMD")

    assert repaired["signal"] == "HOLD"
    assert repaired["veto_contract_repair_skipped"] == "active_reviewer_mitigation"


def test_deterministic_sanity_prompt_context_demands_veto_for_directional_precheck():
    text = build_deterministic_sanity_prompt_context({
        "bias": "SHORT",
        "score": -5,
        "reasons": ["price_below_vwap_-0.60%"],
    })

    assert "Deterministic sanity favors SHORT" in text
    assert "llm_veto_reason" in text


def test_validate_candle_indicator_alignment_rejects_cross_symbol_candles():
    quote = {"price": 280.0}
    candles = {"close": [738.0]}
    indicators = {"vwap": 738.0}

    try:
        validate_candle_indicator_alignment("IWM", quote, candles, indicators)
    except ValueError as exc:
        assert "candle_quote_mismatch" in str(exc)
    else:
        raise AssertionError("expected cross-symbol candle mismatch to be rejected")


def test_sanitize_key_levels_drops_implausible_fallback_vwap():
    signal = {"key_levels": {}}
    quote = {"price": 280.0, "low": 278.0, "high": 282.0}
    indicators = {"vwap": 738.0}

    sanitized = sanitize_analyst_key_levels(signal, quote, indicators)

    assert sanitized["key_levels"]["support"] == 278.0
    assert sanitized["key_levels"]["resistance"] == 282.0
    assert "vwap" not in sanitized["key_levels"]
    assert sanitized["removed_key_levels"]["fallback.vwap"] == 738.0


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
            "llm_veto_required": True,
            "llm_veto_present": False,
            "llm_veto_missing": True,
            "llm_veto_reason": "MISSING_LLM_VETO_REASON: Analyst output HOLD despite deterministic LONG sanity.",
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
    assert summary["veto_required"] == 1
    assert summary["veto_missing"] == 1
    assert summary["sanity_conflicts"][0]["veto_missing"] is True
    assert "eligible=0" in text
    assert "veto_missing=1" in text
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
