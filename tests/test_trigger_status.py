from utils.trigger_status import compute_trigger_status
from utils.prompt_compaction import compact_signal_for_pm


def test_breakout_approaching_when_price_near_resistance():
    result = compute_trigger_status(
        {"key_levels": {"resistance": 100, "vwap": 96, "support": 94}},
        {"price": 99.5},
    )

    assert result["status"] == "waiting_for_breakout"
    assert result["entry_trigger"] == "breakout_approaching"
    assert result["breakout"]["status"] == "approaching"
    assert result["breakout"]["level"] == 100


def test_breakout_confirmed_when_price_above_resistance():
    result = compute_trigger_status(
        {"key_levels": {"resistance": 100, "vwap": 97, "support": 95}},
        {"price": 100.25},
    )

    assert result["status"] == "breakout_confirmed"
    assert result["entry_trigger"] == "breakout_confirmed"
    assert result["breakout"]["status"] == "confirmed"


def test_pullback_validating_when_price_tests_vwap():
    result = compute_trigger_status(
        {"key_levels": {"resistance": 105, "vwap": 100, "support": 96}},
        {"price": 100.2},
    )

    assert result["status"] == "pullback_validating"
    assert result["entry_trigger"] == "pullback_validating"
    assert result["pullback"]["status"] == "at_level"
    assert result["pullback"]["level_name"] == "vwap"


def test_trigger_failed_when_price_loses_support():
    result = compute_trigger_status(
        {"key_levels": {"resistance": 105, "vwap": 100, "support": 96}},
        {"price": 95.5},
    )

    assert result["status"] == "trigger_failed"
    assert result["entry_trigger"] == "pullback_failed"
    assert result["pullback"]["status"] == "failed"


def test_pm_compact_signal_includes_trigger_status():
    text = compact_signal_for_pm(
        "AMD",
        {
            "signal": "LONG",
            "strength": "moderate",
            "setup_type": "technical_breakout",
            "trigger_status": {
                "status": "waiting_for_breakout",
                "entry_trigger": "breakout_approaching",
                "breakout": {"status": "approaching", "level": 100},
                "pullback": {"status": "holding_above_level", "level": 97},
            },
        },
        scaffold_result={"status": "insufficient_data", "reason": "test"},
    )

    assert "trigger_status:" in text
    assert "entry_trigger: breakout_approaching" in text
    assert "breakout: approaching @ 100" in text
