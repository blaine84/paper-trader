from agents.quant_researcher import _normalize_quant_result, _normalize_recommendation


def test_malformed_enum_literal_defaults_to_caution():
    assert _normalize_recommendation("lean_into|use_with_caution|avoid") == "use_with_caution"


def test_unknown_regime_without_researcher_context_does_not_invent_bias():
    result = {
        "market_conditions_summary": "Risk-off regime due to geopolitical tensions.",
        "strategies": [
            {
                "strategy_key": "trend_pullback",
                "strategy_name": "Trend Pullback",
                "fit_score": 8,
                "recommendation": "lean_into|use_with_caution|avoid",
                "conditions_met": ["risk_off regime", "low premarket volume"],
                "analyst_guidance": "Monitor trend pullback signals in risk-off conditions.",
                "pm_guidance": "Watch for risk-off execution.",
            }
        ],
    }

    normalized = _normalize_quant_result(result, market_regime=None, market_context_text="")

    assert normalized["market_regime"] == "unknown"
    assert "No deterministic current market regime" in normalized["market_conditions_summary"]
    assert normalized["strategies"][0]["recommendation"] == "use_with_caution"
    assert normalized["strategies"][0]["conditions_met"] == ["low premarket volume"]
    assert "no current regime bias" in normalized["strategies"][0]["analyst_guidance"]
