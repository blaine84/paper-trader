from agents.portfolio_manager import normalize_pm_entry_decisions


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
