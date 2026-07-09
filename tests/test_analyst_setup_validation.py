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


def test_unclear_direction_is_forced_to_hold():
    signal = {
        "symbol": "AAPL",
        "signal": "SHORT",
        "strength": "strong",
        "confidence": "high",
        "setup_type": "unclear_direction",
        "normalized_setup_suggestion": "risk_off_macro_short",
    }

    result = normalize_analyst_signal_shape(signal, "AAPL")

    assert result["setup_type"] == "unclear_direction"
    assert result["signal"] == "HOLD"
    assert result["strength"] == "weak"
    assert result["confidence"] == "low"
    assert result["normalized_setup_suggestion"] is None
    assert result["needs_setup_type_review"] is True
