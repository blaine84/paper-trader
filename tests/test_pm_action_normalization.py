from agents.portfolio_manager import (
    _apply_scaffold_geometry_defaults,
    _validate_symbol,
    normalize_pm_entry_decisions,
)


def test_skip_is_normalized_to_pass_non_order():
    result = normalize_pm_entry_decisions(
        [{"action": "skip", "symbol": "XLK", "reason": "wait for confirmation"}],
        {"XLK": {}},
    )

    assert result.orders == []
    assert result.rejections == []
    assert len(result.non_orders) == 1
    assert result.non_orders[0].action == "PASS"
    assert result.non_orders[0].symbol == "XLK"


def test_decision_type_reject_is_non_order_not_malformed_buy():
    result = normalize_pm_entry_decisions(
        [
            {
                "symbol": "AMD",
                "action": "BUY",
                "decision_type": "reject",
                "rationale": "R:R below threshold",
            }
        ],
        {"AMD": {}},
    )

    assert result.orders == []
    assert result.rejections == []
    assert len(result.non_orders) == 1
    assert result.non_orders[0].action == "REJECT"
    assert result.non_orders[0].symbol == "AMD"


def test_real_ticker_outside_entry_signals_has_specific_reason_code():
    result = normalize_pm_entry_decisions(
        [
            {
                "action": "BUY",
                "symbol": "XLU",
                "quantity": 10,
                "entry_price": 50.0,
                "stop": 49.0,
                "target": 52.0,
            }
        ],
        {"XLK": {}, "XLF": {}},
    )

    assert result.orders == []
    assert len(result.rejections) == 1
    rejection = result.rejections[0]
    assert rejection.reason_code == "symbol_not_in_entry_signals"
    assert rejection.details["symbol"] == "XLU"
    assert rejection.details["allowed_symbols"] == ["XLF", "XLK"]


def test_concept_symbol_is_still_unsupported_symbol():
    valid, canonical, reason, details = _validate_symbol("XLE|XLB", {"XLE": {}})

    assert valid is False
    assert canonical is None
    assert reason == "unsupported_symbol"
    assert details["symbol"] == "XLE|XLB"


def test_scaffold_geometry_defaults_fill_missing_candidate_fields():
    decisions = [
        {
            "symbol": "AMD",
            "action": "BUY",
            "decision_type": "accept",
            "geometry_candidate_id": "amd_long_breakout_continuation_1",
            "quantity": 10,
            "entry_price": 162.0,
            "rationale": "Breakout continuation",
        }
    ]
    scaffold_results = {
        "AMD": {
            "status": "ok",
            "candidates": [
                {
                    "candidate_id": "amd_long_breakout_continuation_1",
                    "name": "breakout_continuation",
                    "entry_price": 161.5,
                    "stop_loss": 158.0,
                    "target": 168.0,
                }
            ],
        }
    }

    repaired = _apply_scaffold_geometry_defaults(decisions, scaffold_results)

    assert repaired[0]["entry_price"] == 162.0
    assert repaired[0]["stop_loss"] == 158.0
    assert repaired[0]["target"] == 168.0
    assert repaired[0]["geometry_candidate_name"] == "breakout_continuation"
    assert "stop_loss" not in decisions[0]
