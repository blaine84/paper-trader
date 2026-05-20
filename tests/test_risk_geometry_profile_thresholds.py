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
    # Moderate should still reject, but aggressive should be allowed.
    moderate = _evaluate("moderate")
    aggressive = _evaluate("aggressive")

    assert moderate["decision"] == "rejected"
    assert moderate["reason_code"] == "RISK_REWARD_AFTER_STOP_ADJUSTMENT"
    assert "below minimum 1.50" in moderate["reason"]

    assert aggressive["decision"] == "adjusted_allowed"
    assert aggressive["adjusted_rr"] == pytest.approx(1.3)
    assert aggressive["min_reward_to_risk"] == 1.25


def test_high_beta_unchanged_rr_is_profile_aware():
    # Stop already meets the 1.5% high-beta stop floor, so this exercises the
    # unchanged branch rather than the reconstructed/adjusted branch.
    moderate = _evaluate("moderate", stop_price=98.5)
    aggressive = _evaluate("aggressive", stop_price=98.5)

    assert moderate["decision"] == "rejected"
    assert moderate["reason_code"] == "RISK_REWARD_BELOW_THRESHOLD"
    assert "below minimum 1.50" in moderate["reason"]

    assert aggressive["decision"] == "passed_unchanged"
    assert aggressive["original_rr"] == pytest.approx(1.3)
    assert aggressive["min_reward_to_risk"] == 1.25
