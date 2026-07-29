from datetime import datetime, timedelta, timezone

from utils.candidate_registry import CandidateRecord
from utils.entry_geometry import build_entry_geometry_scaffold
from utils.preflight_validator import compute_preflight


def _candidate(
    *,
    symbol: str = "MU",
    direction: str = "SHORT",
    profile_id: str = "aggressive",
) -> CandidateRecord:
    now = datetime.now(timezone.utc)
    return CandidateRecord(
        candidate_id="candidate-1",
        cycle_id="cycle-1",
        profile_id=profile_id,
        symbol=symbol,
        direction=direction,
        setup_type="momentum_fade",
        geometry_name="breakdown_continuation",
        entry_price=758.64,
        stop_price=772.58,
        target_price=737.73,
        risk_reward=1.5,
        trigger="Price breaks below support 758.64",
        invalidation_basis="Price recovers above stop 772.58",
        target_basis="Entry - risk x target multiplier",
        source_signal_id="signal-1",
        signal_snapshot_json='{"symbol": "MU"}',
        created_at=now,
        expires_at=now + timedelta(hours=1),
        integrity_hash="hash",
    )


def test_recent_same_direction_stop_loss_blocks_reentry():
    now = datetime.now(timezone.utc)
    candidate = _candidate()
    recent_closed_trades = [
        {
            "profile": "aggressive",
            "symbol": "MU",
            "direction": "SHORT",
            "exit_time": now - timedelta(minutes=20),
            "pnl": -437.58,
            "reason_exit": "Price monitor: stop_loss at 775.47",
        }
    ]

    result = compute_preflight(
        candidate,
        {"min_risk_reward": 1.0, "max_positions": 10},
        {"available_cash": 100_000},
        [],
        now,
        recent_closed_trades,
    )

    assert result.passed is False
    assert "recent_same_symbol_direction_stop_loss" in result.blocking_reason_codes


def test_old_or_opposite_direction_closed_trade_does_not_block_reentry():
    now = datetime.now(timezone.utc)
    candidate = _candidate(direction="SHORT")
    recent_closed_trades = [
        {
            "profile": "aggressive",
            "symbol": "MU",
            "direction": "LONG",
            "exit_time": now - timedelta(minutes=10),
            "pnl": -100.0,
            "reason_exit": "Price monitor: stop_loss",
        },
        {
            "profile": "aggressive",
            "symbol": "MU",
            "direction": "SHORT",
            "exit_time": now - timedelta(minutes=45),
            "pnl": -100.0,
            "reason_exit": "Price monitor: stop_loss",
        },
    ]

    result = compute_preflight(
        candidate,
        {"min_risk_reward": 1.0, "max_positions": 10},
        {"available_cash": 100_000},
        [],
        now,
        recent_closed_trades,
    )

    assert result.passed is True
    assert "recent_same_symbol_direction_stop_loss" not in result.blocking_reason_codes


def test_breakdown_short_requires_current_price_below_support():
    signal = {
        "symbol": "MU",
        "signal": "SHORT",
        "current_price": 772.58,
        "key_levels": {
            "support": 758.64,
            "resistance": 783.66,
        },
    }

    scaffold = build_entry_geometry_scaffold(signal, profile_id="aggressive")

    assert all(
        candidate["name"] != "breakdown_continuation"
        for candidate in scaffold.get("candidates", [])
    )


def test_confirmed_breakdown_short_still_generates_candidate():
    signal = {
        "symbol": "MU",
        "signal": "SHORT",
        "current_price": 757.90,
        "key_levels": {
            "support": 758.64,
            "resistance": 783.66,
        },
    }

    scaffold = build_entry_geometry_scaffold(signal, profile_id="aggressive")

    assert scaffold["status"] == "ok"
    assert any(
        candidate["name"] == "breakdown_continuation"
        for candidate in scaffold["candidates"]
    )
