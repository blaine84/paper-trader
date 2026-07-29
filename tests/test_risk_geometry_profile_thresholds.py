import pytest

from utils.risk_geometry_gate import evaluate_risk_geometry


def _evaluate(profile: str, *, stop_price: float = 99.8):
    return evaluate_risk_geometry(
        entry_price=100.0,
        stop_price=stop_price,
        target_price=101.95,
        quantity=10,
        direction="BUY",
        symbol="NVDA",
        setup_type="momentum_fade",
        atr_5min=None,
        atr_timestamp=None,
        max_dollar_risk=1_000,
        profile=profile,
    )


def test_high_beta_adjusted_rr_is_profile_aware():
    # NVDA high-beta floor widens stop to 1.5%, making adjusted R:R 1.30.
    # Moderate is warned/allowed while risk geometry is running as a soft gate;
    # aggressive should be allowed outright.
    moderate = _evaluate("moderate")
    aggressive = _evaluate("aggressive")

    assert moderate["decision"] == "adjusted_allowed"
    assert moderate["canonical_decision"] == "warn"
    assert moderate["reason_code"] == "RISK_REWARD_AFTER_STOP_ADJUSTMENT"
    assert "below minimum 1.50" in moderate["reason"]
    assert moderate["risk_geometry_soft_gate"] is True
    assert moderate["adjusted_rr"] == pytest.approx(1.3)

    assert aggressive["decision"] == "adjusted_allowed"
    assert aggressive["adjusted_rr"] == pytest.approx(1.3)
    assert aggressive["min_reward_to_risk"] == 1.25


def test_high_beta_unchanged_rr_is_profile_aware():
    # Stop already meets the 1.5% high-beta stop floor, so this exercises the
    # unchanged branch rather than the reconstructed/adjusted branch.
    moderate = _evaluate("moderate", stop_price=98.5)
    aggressive = _evaluate("aggressive", stop_price=98.5)

    assert moderate["decision"] == "warn"
    assert moderate["canonical_decision"] == "warn"
    assert moderate["reason_code"] == "RISK_REWARD_BELOW_THRESHOLD"
    assert "below minimum 1.50" in moderate["reason"]
    assert moderate["risk_geometry_soft_gate"] is True

    assert aggressive["decision"] == "passed_unchanged"
    assert aggressive["original_rr"] == pytest.approx(1.3)
    assert aggressive["min_reward_to_risk"] == 1.25


def test_aggressive_adjusts_target_to_execution_floor_after_stop_widening():
    result = evaluate_risk_geometry(
        entry_price=802.12,
        stop_price=803.72,
        target_price=799.71,
        quantity=127,
        direction="SHORT",
        symbol="MU",
        setup_type="momentum_fade",
        atr_5min=None,
        atr_timestamp=None,
        max_dollar_risk=1_000,
        profile="aggressive",
    )

    assert result["decision"] == "adjusted_allowed"
    assert result["canonical_decision"] == "warn"
    assert result["reason_code"] == "RISK_REWARD_AFTER_STOP_ADJUSTMENT"
    assert result["risk_geometry_soft_gate"] is True
    assert result["target_recalculated"] is True
    assert result["original_target_price"] == 799.71
    assert result["stop_price"] == pytest.approx(811.74544)
    assert result["target_price"] == pytest.approx(797.30728)
    assert result["adjusted_rr"] == pytest.approx(0.5)
    assert result["min_reward_to_risk"] == 1.25
    assert result["execution_min_reward_to_risk"] == 0.5
