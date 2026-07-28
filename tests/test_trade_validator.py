import pytest

from utils.trade_validator import TradeValidationError, validate_trade


def _decision(rr: float) -> dict:
    price = 100.0
    risk = 10.0
    return {
        "symbol": "MSFT",
        "action": "BUY",
        "price": price,
        "stop": price - risk,
        "target": price + (risk * rr),
        "quantity": 1,
    }


def test_aggressive_final_validator_allows_reduced_rr_floor():
    validate_trade(
        _decision(0.57),
        profile_id="aggressive",
        cash=10_000,
        total_equity=10_000,
        direction="LONG",
    )


def test_aggressive_final_validator_blocks_below_reduced_rr_floor():
    with pytest.raises(TradeValidationError, match="below minimum 0.50:1"):
        validate_trade(
            _decision(0.49),
            profile_id="aggressive",
            cash=10_000,
            total_equity=10_000,
            direction="LONG",
        )


def test_unknown_profile_keeps_one_to_one_floor():
    with pytest.raises(TradeValidationError, match="below minimum 1.00:1"):
        validate_trade(
            _decision(0.99),
            profile_id="unknown",
            cash=10_000,
            total_equity=10_000,
            direction="LONG",
        )
