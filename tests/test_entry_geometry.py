"""Tests for deterministic PM entry geometry scaffold."""

from utils.entry_geometry import build_entry_geometry_scaffold


def test_long_scaffold_produces_valid_executable_candidates():
    signal = {
        "symbol": "XLE",
        "signal": "LONG",
        "current_price": 61.32,
        "key_levels": {
            "support": 60.295,
            "resistance": 61.33,
            "vwap": 60.76,
            "day_high": 61.33,
            "day_low": 60.295,
        },
    }

    result = build_entry_geometry_scaffold(signal, profile_id="moderate")

    assert result["source"] == "deterministic_geometry_scaffold"
    assert result["status"] == "ok"
    assert result["direction"] == "LONG"
    assert result["candidates"]
    for candidate in result["candidates"]:
        assert candidate["stop_loss"] < candidate["entry_price"] < candidate["target"]
        assert candidate["risk_reward"] >= 1.0
        assert candidate["name"] in {
            "pullback_to_vwap",
            "support_bounce",
            "breakout_continuation",
        }
        assert candidate["trigger"]
        assert candidate["invalidation_basis"]
        assert candidate["target_basis"]


def test_short_scaffold_produces_valid_executable_candidates():
    signal = {
        "symbol": "XYZ",
        "signal": "SHORT",
        "current_price": 100.0,
        "key_levels": {
            "support": 98.0,
            "resistance": 101.0,
            "vwap": 100.5,
            "day_high": 101.0,
            "day_low": 98.0,
        },
    }

    result = build_entry_geometry_scaffold(signal, profile_id="moderate")

    assert result["status"] == "ok"
    assert result["direction"] == "SHORT"
    assert result["candidates"]
    for candidate in result["candidates"]:
        assert candidate["target"] < candidate["entry_price"] < candidate["stop_loss"]
        assert candidate["risk_reward"] >= 1.0
        assert candidate["name"] in {
            "resistance_rejection",
            "breakdown_continuation",
            "fade",
        }
        assert candidate["trigger"]
        assert candidate["invalidation_basis"]
        assert candidate["target_basis"]


def test_hold_scaffold_is_not_tradeable_and_has_no_candidates():
    result = build_entry_geometry_scaffold(
        {
            "symbol": "ABC",
            "signal": "HOLD",
            "current_price": 10.0,
            "key_levels": {"support": 9.5, "resistance": 10.5, "vwap": 10.0},
        }
    )

    assert result["status"] == "not_tradeable_signal"
    assert result["direction"] == "HOLD"
    assert result["candidates"] == []
    assert "HOLD" in result["reason"]


def test_missing_current_price_fails_closed():
    result = build_entry_geometry_scaffold(
        {
            "symbol": "MISS",
            "signal": "LONG",
            "key_levels": {"support": 49.0, "resistance": 51.0, "vwap": 50.0},
        }
    )

    assert result["status"] == "insufficient_data"
    assert result["candidates"] == []
    assert "current_price" in result["reason"]


def test_missing_tradeable_levels_fails_closed_with_reason():
    result = build_entry_geometry_scaffold(
        {
            "symbol": "MISS",
            "signal": "LONG",
            "current_price": 50.0,
            "key_levels": {},
        }
    )

    assert result["status"] == "insufficient_data"
    assert result["candidates"] == []
    assert "missing required levels" in result["reason"]


def test_risk_reward_is_recomputed_after_rounding():
    result = build_entry_geometry_scaffold(
        {
            "symbol": "TICK",
            "signal": "LONG",
            "current_price": 100.0,
            "tick_size": 0.05,
            "key_levels": {"vwap": 99.0, "support": 98.5, "resistance": 100.5},
        },
        profile_context={"target_multiplier": 2.0, "min_risk_reward": 1.0},
    )

    assert result["status"] == "ok"
    for candidate in result["candidates"]:
        entry = candidate["entry_price"]
        stop = candidate["stop_loss"]
        target = candidate["target"]
        expected_rr = round((target - entry) / (entry - stop), 2)
        assert candidate["risk_reward"] == expected_rr
