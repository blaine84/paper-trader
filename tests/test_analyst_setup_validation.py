from agents.analyst import annotate_unregistered_setup


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
