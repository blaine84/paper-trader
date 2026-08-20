from agents.analyst import (
    annotate_unregistered_setup,
    build_technical_data_quality_context,
    compute_intraday_indicators_with_warmup,
    normalize_analyst_signal_shape,
    sanitize_historical_context_for_prompt,
    sanitize_reasoning_quality,
    sanitize_historical_feedback_bleed,
)


def test_registered_setup_type_has_no_warning():
    signal = {"symbol": "AAPL", "setup_type": "technical_breakout"}

    result = annotate_unregistered_setup(signal, ["technical_breakout", "orb"])

    assert "setup_validation_warning" not in result
    assert "needs_setup_type_review" not in result


def test_canonical_swing_setup_type_has_no_warning_when_registry_list_is_stale():
    signal = {"symbol": "META", "setup_type": "pullback_continuation"}

    result = annotate_unregistered_setup(signal, ["technical_breakout", "orb"])

    assert "setup_validation_warning" not in result
    assert "needs_setup_type_review" not in result


def test_unregistered_setup_type_is_preserved_with_warning():
    signal = {"symbol": "AAPL", "setup_type": "liquidity_sweep"}

    result = annotate_unregistered_setup(signal, ["technical_breakout", "orb"])

    assert result["setup_type"] == "liquidity_sweep"
    assert result["needs_setup_type_review"] is True
    assert "liquidity_sweep" in result["setup_validation_warning"]


def test_unregistered_setup_annotation_preserves_existing_warning():
    signal = {
        "symbol": "AAPL",
        "setup_type": "unclear_direction",
        "setup_validation_warning": "forced to HOLD by normalizer",
        "needs_setup_type_review": True,
    }

    result = annotate_unregistered_setup(signal, ["technical_breakout", "orb"])

    assert result["setup_validation_warning"] == "forced to HOLD by normalizer"
    assert result["needs_setup_type_review"] is True


def test_directional_confusion_breakout_is_rewritten_to_hold():
    signal = {
        "symbol": "AAPL",
        "signal": "LONG",
        "strength": "moderate",
        "confidence": "medium",
        "setup_type": "directional_confusion_breakout",
        "normalized_setup_suggestion": "breakout_retest",
    }

    result = normalize_analyst_signal_shape(signal, "AAPL")

    assert result["setup_type"] == "unclear_direction"
    assert result["signal"] == "HOLD"
    assert result["strength"] == "weak"
    assert result["confidence"] == "low"
    assert result["normalized_setup_suggestion"] is None
    assert result["needs_setup_type_review"] is True


def test_actionable_unclear_direction_uses_valid_swing_suggestion():
    signal = {
        "symbol": "AAPL",
        "signal": "SHORT",
        "strength": "strong",
        "confidence": "high",
        "setup_type": "unclear_direction",
        "normalized_setup_suggestion": "risk_off_macro_short",
    }

    result = normalize_analyst_signal_shape(signal, "AAPL")

    assert result["setup_type"] == "risk_off_macro_short"
    assert result["signal"] == "SHORT"
    assert result["strength"] == "strong"
    assert result["confidence"] == "high"
    assert result["normalized_setup_suggestion"] == "risk_off_macro_short"
    assert result["original_setup_type"] == "unclear_direction"
    assert result["needs_setup_type_review"] is True


def test_unclear_direction_without_actionable_suggestion_is_forced_to_hold():
    signal = {
        "symbol": "AAPL",
        "signal": "SHORT",
        "strength": "strong",
        "confidence": "high",
        "setup_type": "unclear_direction",
        "normalized_setup_suggestion": None,
    }

    result = normalize_analyst_signal_shape(signal, "AAPL")

    assert result["setup_type"] == "unclear_direction"
    assert result["signal"] == "HOLD"
    assert result["strength"] == "weak"
    assert result["confidence"] == "low"
    assert result["normalized_setup_suggestion"] is None
    assert result["needs_setup_type_review"] is True


def test_actionable_unclear_direction_infers_long_breakout_setup():
    signal = {
        "symbol": "TSLA",
        "signal": "LONG",
        "strength": "moderate",
        "confidence": "medium",
        "setup_type": "unclear_direction",
        "normalized_setup_suggestion": None,
        "setup_reasoning": "Bullish trend near VWAP with a potential breakout.",
        "reasoning": "MACD bullish and price is holding above VWAP.",
        "key_levels": {"support": 404.91, "resistance": 412.49, "vwap": 408.97},
        "indicators": {
            "rsi": 54.54,
            "macd_bias": "bullish",
            "ema_trend": "bullish",
            "above_vwap": True,
        },
    }

    result = normalize_analyst_signal_shape(signal, "TSLA")

    assert result["setup_type"] == "breakout_retest"
    assert result["signal"] == "LONG"
    assert result["normalized_setup_suggestion"] == "breakout_retest"


def test_oversold_unclear_short_without_suggestion_stays_hold():
    signal = {
        "symbol": "GLD",
        "signal": "SHORT",
        "strength": "strong",
        "confidence": "high",
        "setup_type": "unclear_direction",
        "normalized_setup_suggestion": None,
        "setup_reasoning": "Risk-off but RSI is oversold and price may bounce.",
        "reasoning": "Bearish MACD, below VWAP, but oversold near support.",
        "key_levels": {"support": 374.97, "resistance": 378.44, "vwap": 375.75},
        "indicators": {
            "rsi": 36.47,
            "macd_bias": "bearish",
            "ema_trend": "bearish",
            "above_vwap": False,
        },
    }

    result = normalize_analyst_signal_shape(signal, "GLD")

    assert result["setup_type"] == "unclear_direction"
    assert result["signal"] == "HOLD"
    assert result["normalized_setup_suggestion"] is None


def test_technical_confusion_breakout_is_rewritten_to_hold():
    signal = {
        "symbol": "TSLA",
        "signal": "SHORT",
        "strength": "moderate",
        "confidence": "medium",
        "setup_type": "technical_confusion_breakout",
        "setup_reasoning": "Bearish read, but no clean technical setup.",
        "normalized_setup_suggestion": None,
    }

    result = normalize_analyst_signal_shape(signal, "TSLA")

    assert result["setup_type"] == "unclear_direction"
    assert result["signal"] == "HOLD"
    assert result["strength"] == "weak"
    assert result["confidence"] == "low"
    assert result["normalized_setup_suggestion"] is None
    assert result["needs_setup_type_review"] is True


def test_directional_unknown_setup_is_forced_to_hold():
    signal = {
        "symbol": "MSFT",
        "signal": "SHORT",
        "strength": "strong",
        "confidence": "high",
        "setup_type": "liquidity_sweep",
        "setup_reasoning": "Unregistered bearish label from the LLM.",
        "normalized_setup_suggestion": None,
    }

    result = normalize_analyst_signal_shape(signal, "MSFT")

    assert result["setup_type"] == "unclear_direction"
    assert result["signal"] == "HOLD"
    assert result["strength"] == "weak"
    assert result["confidence"] == "low"
    assert result["original_signal"] == "SHORT"
    assert result["original_setup_type"] == "liquidity_sweep"
    assert result["needs_setup_type_review"] is True


def test_directional_sector_rotation_remains_mappable():
    signal = {
        "symbol": "AMD",
        "signal": "LONG",
        "strength": "strong",
        "confidence": "high",
        "setup_type": "sector_rotation",
        "setup_reasoning": "Semis leading with strong sector breadth.",
        "normalized_setup_suggestion": "sector_rotation_swing",
    }

    result = normalize_analyst_signal_shape(signal, "AMD")

    assert result["setup_type"] == "sector_rotation"
    assert result["signal"] == "LONG"
    assert result["strength"] == "strong"
    assert result["confidence"] == "high"


def test_historical_feedback_bleed_redacts_stale_cpi_sentences():
    signal = {
        "symbol": "XLK",
        "signal": "HOLD",
        "strength": "weak",
        "confidence": "low",
        "setup_type": "unclear_direction",
        "setup_reasoning": (
            "The setup is unclear due to the presence of a scheduled macro catalyst "
            "(CPI inflation print) on July 15, which directly contradicts the intraday "
            "rotation setup mandate. Price is below VWAP with bearish sector breadth."
        ),
        "reasoning": (
            "The economic calendar intersection was either missed or consciously ignored. "
            "Current tape is weak and volume is elevated."
        ),
        "invalidation": "Price reclaims VWAP or July 15 CPI risk resolves.",
        "llm_veto_reason": "Scheduled major macro catalyst blocks the setup.",
    }

    result = sanitize_historical_feedback_bleed(signal)
    combined = " ".join(
        result.get(field) or ""
        for field in ("setup_reasoning", "reasoning", "invalidation", "llm_veto_reason")
    )

    assert result["historical_feedback_redacted"] is True
    assert "CPI" not in combined
    assert "July 15" not in combined
    assert "scheduled macro catalyst" not in combined.lower()
    assert "economic calendar intersection" not in combined.lower()
    assert "Price is below VWAP" in result["setup_reasoning"]
    assert result["llm_veto_reason"] is None
    assert result["veto_evidence"] == []


def test_historical_feedback_bleed_redacts_trade_review_crossover():
    signal = {
        "symbol": "AMD",
        "signal": "HOLD",
        "strength": "weak",
        "confidence": "low",
        "setup_type": "unclear_direction",
        "setup_reasoning": (
            "The Analyst failed to flag this critical macro event in the past, "
            "violating discipline-profitable exit masks execution risk: moderate "
            "PM should NOT hold intraday rotation trades 1400+ minutes overnight "
            "into major macro catalysts without explicit multi-day authorization "
            "and hard contingency exit rule tied to catalyst outcome."
        ),
        "reasoning": (
            "The setup is bearish due to price below VWAP (-0.92%), negative "
            "change (-8.87%), and relative volume confirming (2.10x)."
        ),
        "llm_veto_reason": (
            "Moderate PM should not hold 1400+ minutes overnight into a macro event."
        ),
        "veto_evidence": ["historical AMD review"],
    }

    result = sanitize_historical_feedback_bleed(signal)

    assert result["historical_feedback_redacted"] is True
    assert "profitable exit" not in result["setup_reasoning"].lower()
    assert "1400" not in result["setup_reasoning"]
    assert "price below VWAP" in result["reasoning"]
    assert result["llm_veto_reason"] is None
    assert result["veto_evidence"] == []


def test_historical_feedback_bleed_redacts_process_review_language():
    signal = {
        "symbol": "MSFT",
        "signal": "LONG",
        "strength": "moderate",
        "confidence": "medium",
        "setup_type": "technical_breakout",
        "setup_reasoning": "The Analyst failed to flag this critical conflict in prior setups.",
        "reasoning": (
            "This is a Scout/Analyst failure to cross-reference economic calendar "
            "against holding duration and setup type. Rotation setups are inherently "
            "intraday (480min or less) and require hard exit discipline."
        ),
        "key_levels": {"support": 396.0, "resistance": 400.24, "vwap": 396.32},
        "indicators": {"rsi": 54.2, "macd_bias": "bullish", "ema_trend": "bullish"},
        "market_state": "breakout_retest_watch",
        "sentiment": "bullish",
    }

    result = sanitize_historical_feedback_bleed(signal)
    combined = " ".join(
        result.get(field) or ""
        for field in ("setup_reasoning", "reasoning", "invalidation", "llm_veto_reason")
    )

    assert result["historical_feedback_redacted"] is True
    assert "Analyst failed" not in combined
    assert "Scout/Analyst failure" not in combined
    assert "holding duration" not in combined
    assert "hard exit discipline" not in combined
    assert "MSFT current setup is LONG / technical_breakout" in result["setup_reasoning"]
    assert "resistance 400.24" in result["setup_reasoning"]
    assert "VWAP 396.32" in result["setup_reasoning"]


def test_historical_prompt_context_redacts_process_review_language():
    raw_context = """
    The Analyst failed to flag this critical conflict in prior setups.
    This is a Scout/Analyst failure to cross-reference economic calendar against holding duration.
    Strongest when price holds above VWAP with aligned 5m/60m trend.
    """

    result = sanitize_historical_context_for_prompt(raw_context)

    assert "Analyst failed" not in result
    assert "Scout/Analyst failure" not in result
    assert "holding duration" not in result
    assert "Strongest when price holds above VWAP" in result
    assert "Historical advisory lessons only" in result


def test_intraday_indicators_use_warmup_when_session_is_thin(monkeypatch):
    class FakeClient:
        def get_candles(self, symbol, resolution, days):
            assert symbol == "NVDA"
            assert resolution == "5"
            assert days == 20
            return {
                "open": list(range(100, 130)),
                "high": list(range(101, 131)),
                "low": list(range(99, 129)),
                "close": list(range(100, 130)),
                "volume": [1000] * 30,
                "timestamps": [1_750_000_000 + i * 300 for i in range(30)],
            }

    thin_session = {
        "open": [100, 101],
        "high": [101, 102],
        "low": [99, 100],
        "close": [100, 101],
        "volume": [1000, 1200],
        "timestamps": [1_750_000_000, 1_750_000_300],
    }

    result = compute_intraday_indicators_with_warmup(FakeClient(), "NVDA", thin_session)

    assert result["indicator_warmup_used"] is True
    assert result["indicator_warmup_reason"] == "Not enough candles for indicators"
    assert "rsi" in result
    assert result.get("error") is None


def test_technical_data_quality_distinguishes_partial_mtf_availability():
    context = {
        "timeframes": {
            "5m": {"price": 202.28, "trend": None, "macd_bias": None, "rsi": None},
            "60m": {"price": 202.28, "trend": "bearish", "macd_bias": "bearish", "rsi": 34.99},
            "daily": {"price": 202.28, "trend": "bullish", "macd_bias": "bullish", "rsi": 46.9},
        }
    }

    prompt_block = build_technical_data_quality_context(
        {"indicator_warmup_used": True},
        context,
    )

    assert "Do not say technical indicators are unavailable" in prompt_block
    assert '"5m"' in prompt_block
    assert '"60m"' in prompt_block
    assert '"trend_available": true' in prompt_block


def test_reasoning_quality_rewrites_confluent_macd_ema_and_key_level_context():
    signal = {
        "setup_reasoning": (
            "The setup is unclear. The MACD is bearish, but the EMA trend is also bearish. "
            "Additionally, there are conflicting key levels: the prior high is 574.01, "
            "and the VWAP is at 498.98."
        ),
        "reasoning": (
            "The technical indicators also support a bearish view: the MACD is bearish, "
            "and the EMA trend is bearish. Additionally, there are conflicting key levels: "
            "the prior high is 574.01, and the VWAP is at 498.98."
        ),
        "key_levels": {
            "support": 476.93,
            "resistance": 527.49,
            "vwap": 498.98,
            "prior_high": 574.01,
            "prior_low": 460.29,
        },
        "indicators": {
            "macd_bias": "bearish",
            "ema_trend": "bearish",
        },
    }

    result = sanitize_reasoning_quality(signal)
    combined = result["setup_reasoning"] + " " + result["reasoning"]

    assert "but the EMA trend is also bearish" not in combined
    assert "conflicting key levels" not in combined.lower()
    assert "trend signals are confluent" in result["setup_reasoning"]
    assert result["trend_confluence"]["direction"] == "bearish"
    assert result["key_level_context"]["structural_levels"]["prior_high"] == 574.01
    assert result["key_level_context"]["dynamic_intraday_indicators"]["vwap"] == 498.98


def test_reasoning_quality_removes_duplicate_reasoning_sentences():
    signal = {
        "setup_reasoning": (
            "MACD and EMA trend are both bearish, so trend signals are confluent. "
            "RSI is oversold near support."
        ),
        "reasoning": (
            "MACD and EMA trend are both bearish, so trend signals are confluent. "
            "Volume is elevated versus the same-time baseline."
        ),
        "key_levels": {},
        "indicators": {},
    }

    result = sanitize_reasoning_quality(signal)

    assert result["reasoning"] == "Volume is elevated versus the same-time baseline."
    assert result["reasoning_quality_sanitized"] is True
