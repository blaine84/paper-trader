from agents.analyst import annotate_unregistered_setup, normalize_analyst_signal_shape


def test_registered_setup_type_has_no_warning():
    signal = {"symbol": "AAPL", "setup_type": "technical_breakout"}

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
