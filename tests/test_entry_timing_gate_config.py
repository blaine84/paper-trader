from agents.portfolio_manager import ENTRY_WINDOW_LIMITS


def test_momentum_fade_is_not_first_hour_only():
    assert "momentum_fade" not in ENTRY_WINDOW_LIMITS


def test_open_only_setups_remain_first_hour_limited():
    assert ENTRY_WINDOW_LIMITS["gap_and_go"] == 60
    assert ENTRY_WINDOW_LIMITS["orb"] == 60
    assert ENTRY_WINDOW_LIMITS["short_squeeze"] == 60
