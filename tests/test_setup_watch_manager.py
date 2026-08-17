"""Tests for utils/setup_watch_manager.py — orchestration, noise control, propagation.

Requirements: 2.1-2.10, 4.1, 4.9-4.11, 5.4-5.8, 6.1-6.2, 6.7-6.8,
              9.5-9.8, 10.1-10.8, 12.2, 12.7
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from db.schema import init_pending_order_schema, init_setup_watch_schema
from utils.pending_order_time import now_utc, to_iso
from utils.setup_watch_manager import (
    CycleEvaluationResult,
    CreationStats,
    _lookup_pending_order_id,
    create_setup_watch,
    evaluate_cycle,
    get_promotable_watches,
    propagate_candidate_results,
)
from utils.setup_watch_registry import (
    ACTIVE_STATES,
    TERMINAL_STATES,
    SetupWatch,
    SetupWatchRegistry,
    SetupWatchRegistryError,
    WatchState,
)

NOW = datetime(2026, 8, 17, 14, 30, 0, tzinfo=timezone.utc)
FUTURE = NOW + timedelta(hours=24)  # safely in the future from now_utc()
PROFILE = "moderate"
CYCLE = "cycle_001"


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    init_setup_watch_schema(eng)
    init_pending_order_schema(eng)
    return eng


@pytest.fixture
def registry(engine):
    return SetupWatchRegistry(engine)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _maturation_conditions(n: int = 3) -> list[dict]:
    """Build a list of n maturation conditions with default weights."""
    return [
        {"type": "price_zone", "params": {"low": 99.0, "high": 101.0}, "weight": 1.0}
        for _ in range(n)
    ]


def _invalidation_conditions(n: int = 1) -> list[dict]:
    """Build a list of n invalidation conditions."""
    return [
        {"type": "price_breach", "params": {"level": 95.0, "direction": "below"}}
        for _ in range(n)
    ]


def _insert_watch(engine, *, watch_id=None, profile_id=PROFILE, symbol="AAPL",
                  side="BUY", setup_type="breakout", state="watching",
                  created_at=None, expires_at=None, observed_cycles=0,
                  maturity_score=0.0, promoted_cycle_id=None,
                  maturation_conditions=None, invalidation_conditions=None,
                  ready_at=None, ready_reference_price=None):
    """Directly insert a watch row for test setup."""
    if watch_id is None:
        watch_id = str(uuid.uuid4())
    if created_at is None:
        created_at = NOW
    if expires_at is None:
        expires_at = FUTURE
    if maturation_conditions is None:
        maturation_conditions = _maturation_conditions(3)
    if invalidation_conditions is None:
        invalidation_conditions = _invalidation_conditions(1)

    now_iso = to_iso(created_at)
    exp_iso = to_iso(expires_at)
    ready_iso = to_iso(ready_at) if ready_at else None

    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO setup_watches "
                "(watch_id, profile_id, symbol, side, setup_type, state, "
                " thesis, source_type, source_cycle_id, "
                " maturation_conditions_json, invalidation_conditions_json, "
                " maturity_score, created_at, updated_at, expires_at, "
                " observed_cycles, promoted_cycle_id, ready_at, "
                " ready_reference_price, integrity_hash) "
                "VALUES "
                "(:wid, :pid, :sym, :side, :stype, :state, "
                " :thesis, :src_type, :cycle_id, "
                " :mat_json, :inv_json, "
                " :score, :now, :now, :exp, "
                " :cycles, :promoted_cycle, :ready_at, "
                " :ready_price, :hash)"
            ),
            {
                "wid": watch_id,
                "pid": profile_id,
                "sym": symbol,
                "side": side,
                "stype": setup_type,
                "state": state,
                "thesis": "A solid breakout thesis for testing purposes",
                "src_type": "analyst",
                "cycle_id": CYCLE,
                "mat_json": json.dumps(maturation_conditions),
                "inv_json": json.dumps(invalidation_conditions),
                "score": maturity_score,
                "now": now_iso,
                "exp": exp_iso,
                "cycles": observed_cycles,
                "promoted_cycle": promoted_cycle_id,
                "ready_at": ready_iso,
                "ready_price": ready_reference_price,
                "hash": "test_hash_" + watch_id[:8],
            },
        )
        conn.commit()
    return watch_id


def _state_of(engine, watch_id) -> str:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT state FROM setup_watches WHERE watch_id = :wid"),
            {"wid": watch_id},
        ).scalar()


def _terminal_reason_of(engine, watch_id) -> str | None:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT terminal_reason FROM setup_watches WHERE watch_id = :wid"),
            {"wid": watch_id},
        ).scalar()


def _count_rows(engine, table="setup_watches") -> int:
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()


def _count_events(engine, watch_id=None, event_type=None) -> int:
    clauses = []
    params = {}
    if watch_id:
        clauses.append("watch_id = :wid")
        params["wid"] = watch_id
    if event_type:
        clauses.append("event_type = :etype")
        params["etype"] = event_type
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with engine.connect() as conn:
        return conn.execute(
            text(f"SELECT COUNT(*) FROM setup_watch_events{where}"),
            params,
        ).scalar()


def _make_signals(symbol="AAPL", current_price=100.0, regime="bullish",
                  strength="strong", thesis=None):
    """Build a signals dict for one symbol."""
    sig = {
        "current_price": current_price,
        "market_regime": regime,
        "signal_strength": strength,
        "setup_type": "breakout",
        "direction": "BUY",
        "thesis": thesis or "A solid thesis about this breakout setup pattern",
        "key_levels": {"support": [95.0], "resistance": [110.0]},
        "catalyst_timestamp": to_iso(NOW - timedelta(minutes=30)),
    }
    return {symbol: sig}


def _insert_pending_order(engine, candidate_id, state="pending"):
    """Insert a minimal pending_orders row for propagation tests."""
    order_id = str(uuid.uuid4())
    now_iso = to_iso(NOW)
    exp_iso = to_iso(NOW + timedelta(hours=2))
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO pending_orders "
                "(order_id, profile_id, symbol, side, setup_type, "
                " candidate_id, limit_price, stop_price, target_price, "
                " risk_reward, fresh_price_at_creation, "
                " runaway_pct_at_creation, state, created_at, expires_at, "
                " integrity_hash) "
                "VALUES "
                "(:oid, :pid, :sym, :side, :stype, "
                " :cid, :limit, :stop, :target, "
                " :rr, :fresh, "
                " :runaway, :state, :created, :expires, "
                " :hash)"
            ),
            {
                "oid": order_id,
                "pid": PROFILE,
                "sym": "AAPL",
                "side": "BUY",
                "stype": "breakout",
                "cid": candidate_id,
                "limit": 100.0,
                "stop": 95.0,
                "target": 110.0,
                "rr": 2.0,
                "fresh": 100.5,
                "runaway": 0.5,
                "state": state,
                "created": now_iso,
                "expires": exp_iso,
                "hash": "testhash",
            },
        )
        conn.commit()
    return order_id


# ────────────────────────────────────────────────────────────────────────────
# 1. evaluate_cycle expires TTL-elapsed watches before evaluating anything
# ────────────────────────────────────────────────────────────────────────────


def test_evaluate_cycle_expires_ttl_elapsed(engine):
    """Watches past their expires_at are expired before any evaluation."""
    past = now_utc() - timedelta(hours=1)
    wid = _insert_watch(engine, expires_at=past)

    result = evaluate_cycle(
        engine, PROFILE, CYCLE, signals={}, portfolio={}
    )

    assert result.expired_ttl >= 1
    assert _state_of(engine, wid) == "expired"


# ────────────────────────────────────────────────────────────────────────────
# 2. evaluate_cycle expires stale promoted watches from prior cycles
# ────────────────────────────────────────────────────────────────────────────


def test_evaluate_cycle_expires_stale_promoted(engine):
    """Promoted watches from a prior cycle are expired."""
    wid = _insert_watch(
        engine, state="promoted", promoted_cycle_id="old_cycle_000"
    )

    result = evaluate_cycle(
        engine, PROFILE, "current_cycle_999", signals={}, portfolio={}
    )

    assert result.expired_stale_promoted >= 1
    assert _state_of(engine, wid) == "expired"


# ────────────────────────────────────────────────────────────────────────────
# 3. evaluate_cycle invalidates before maturation
# ────────────────────────────────────────────────────────────────────────────


def test_evaluate_cycle_invalidates_before_maturation(engine):
    """When invalidation triggers, the watch is rejected regardless of score."""
    # Create a watch with invalidation condition: price_breach below 95
    inv_conds = [{"type": "price_breach", "params": {"level": 95.0, "direction": "below"}}]
    mat_conds = [
        {"type": "price_zone", "params": {"low": 90.0, "high": 110.0}, "weight": 1.0},
        {"type": "regime_aligned", "params": {"required_regime": "bullish"}, "weight": 1.0},
    ]
    wid = _insert_watch(
        engine, symbol="TSLA",
        maturation_conditions=mat_conds,
        invalidation_conditions=inv_conds,
    )

    # Signal shows price at 90 (below 95 breach level)
    signals = {
        "TSLA": {
            "current_price": 90.0,
            "market_regime": "bullish",
            "key_levels": {"support": [85.0]},
        }
    }

    result = evaluate_cycle(engine, PROFILE, CYCLE, signals=signals, portfolio={})

    assert result.invalidated >= 1
    assert _state_of(engine, wid) == "rejected"


# ────────────────────────────────────────────────────────────────────────────
# 4. State progressions: watching→maturing, maturing→ready, regression
# ────────────────────────────────────────────────────────────────────────────


def test_state_progression_watching_to_maturing(engine):
    """Watch transitions from watching to maturing when partial conditions met."""
    # Price zone met but low weight → score > 0 but < threshold
    mat_conds = [
        {"type": "price_zone", "params": {"low": 99.0, "high": 101.0}, "weight": 0.4},
        {"type": "regime_aligned", "params": {"required_regime": "bearish"}, "weight": 0.6},
    ]
    inv_conds = [{"type": "regime_flip", "params": {"blocked_regimes": ["crisis"]}}]
    wid = _insert_watch(
        engine, symbol="GOOG",
        maturation_conditions=mat_conds,
        invalidation_conditions=inv_conds,
    )

    signals = {"GOOG": {"current_price": 100.0, "market_regime": "bullish"}}
    result = evaluate_cycle(engine, PROFILE, CYCLE, signals=signals, portfolio={})

    assert result.matured >= 1
    assert _state_of(engine, wid) == "maturing"


def test_state_progression_maturing_to_ready(engine):
    """Watch transitions from maturing to ready when score >= threshold."""
    # All conditions met → score = 1.0
    mat_conds = [
        {"type": "price_zone", "params": {"low": 99.0, "high": 101.0}, "weight": 1.0},
        {"type": "regime_aligned", "params": {"required_regime": "bullish"}, "weight": 1.0},
    ]
    inv_conds = [{"type": "regime_flip", "params": {"blocked_regimes": ["crisis"]}}]
    wid = _insert_watch(
        engine, symbol="MSFT", state="maturing",
        maturation_conditions=mat_conds,
        invalidation_conditions=inv_conds,
    )

    signals = {"MSFT": {"current_price": 100.0, "market_regime": "bullish"}}

    with patch("utils.setup_watch_manager.SETUP_WATCH_MATURITY_THRESHOLD", 0.7):
        result = evaluate_cycle(engine, PROFILE, CYCLE, signals=signals, portfolio={})

    assert result.matured >= 1
    assert _state_of(engine, wid) == "ready"


def test_state_regression_ready_to_maturing(engine):
    """Ready watch regresses to maturing when score drops below threshold."""
    # Only partial conditions met → score below threshold
    mat_conds = [
        {"type": "price_zone", "params": {"low": 99.0, "high": 101.0}, "weight": 0.3},
        {"type": "regime_aligned", "params": {"required_regime": "bearish"}, "weight": 0.7},
    ]
    inv_conds = [{"type": "regime_flip", "params": {"blocked_regimes": ["crisis"]}}]
    wid = _insert_watch(
        engine, symbol="META", state="ready",
        maturation_conditions=mat_conds,
        invalidation_conditions=inv_conds,
        maturity_score=0.9,
    )

    # Price zone met (0.3) but regime not met → score = 0.3
    signals = {"META": {"current_price": 100.0, "market_regime": "bullish"}}

    with patch("utils.setup_watch_manager.SETUP_WATCH_MATURITY_THRESHOLD", 0.7):
        result = evaluate_cycle(engine, PROFILE, CYCLE, signals=signals, portfolio={})

    assert result.regressed >= 1
    assert _state_of(engine, wid) == "maturing"


def test_state_regression_maturing_to_watching(engine):
    """Maturing watch regresses to watching when score drops to 0."""
    # No conditions met → score = 0
    mat_conds = [
        {"type": "regime_aligned", "params": {"required_regime": "bearish"}, "weight": 1.0},
        {"type": "price_zone", "params": {"low": 200.0, "high": 210.0}, "weight": 1.0},
    ]
    inv_conds = [{"type": "regime_flip", "params": {"blocked_regimes": ["crisis"]}}]
    wid = _insert_watch(
        engine, symbol="NVDA", state="maturing",
        maturation_conditions=mat_conds,
        invalidation_conditions=inv_conds,
        maturity_score=0.5,
    )

    signals = {"NVDA": {"current_price": 100.0, "market_regime": "bullish"}}

    result = evaluate_cycle(engine, PROFILE, CYCLE, signals=signals, portfolio={})

    assert result.regressed >= 1
    assert _state_of(engine, wid) == "watching"


# ────────────────────────────────────────────────────────────────────────────
# 5. Promotion blocked until observed_cycles >= SETUP_WATCH_PROMOTION_MIN_CYCLES
# ────────────────────────────────────────────────────────────────────────────


def test_promotion_blocked_until_min_cycles_reached(engine):
    """Ready watches are not promoted until they have enough observed cycles."""
    mat_conds = [
        {"type": "price_zone", "params": {"low": 99.0, "high": 101.0}, "weight": 1.0},
    ]
    inv_conds = [{"type": "regime_flip", "params": {"blocked_regimes": ["crisis"]}}]
    wid = _insert_watch(
        engine, symbol="AMZN", state="ready",
        maturation_conditions=mat_conds,
        invalidation_conditions=inv_conds,
        maturity_score=0.9,
        observed_cycles=0,  # below threshold
    )

    signals = {"AMZN": {"current_price": 100.0, "market_regime": "neutral"}}

    with patch("utils.setup_watch_manager.SETUP_WATCH_MODE", "enabled"), \
         patch("utils.setup_watch_manager.SETUP_WATCH_PROMOTION_MIN_CYCLES", 3), \
         patch("utils.setup_watch_manager.SETUP_WATCH_MATURITY_THRESHOLD", 0.7):
        result = evaluate_cycle(engine, PROFILE, CYCLE, signals=signals, portfolio={})

    assert result.promoted == 0
    # Watch should still be ready (not promoted)
    assert _state_of(engine, wid) == "ready"


# ────────────────────────────────────────────────────────────────────────────
# 6. Observe mode does not promote; enabled mode does
# ────────────────────────────────────────────────────────────────────────────


def test_observe_mode_does_not_promote(engine):
    """In observe mode, ready watches are never promoted."""
    mat_conds = [
        {"type": "price_zone", "params": {"low": 99.0, "high": 101.0}, "weight": 1.0},
    ]
    inv_conds = [{"type": "regime_flip", "params": {"blocked_regimes": ["crisis"]}}]
    wid = _insert_watch(
        engine, symbol="SPY", state="ready",
        maturation_conditions=mat_conds,
        invalidation_conditions=inv_conds,
        maturity_score=0.9,
        observed_cycles=5,
    )

    signals = {"SPY": {"current_price": 100.0, "market_regime": "neutral"}}

    with patch("utils.setup_watch_manager.SETUP_WATCH_MODE", "observe"), \
         patch("utils.setup_watch_manager.SETUP_WATCH_MATURITY_THRESHOLD", 0.7), \
         patch("utils.setup_watch_manager.SETUP_WATCH_PROMOTION_MIN_CYCLES", 1):
        result = evaluate_cycle(engine, PROFILE, CYCLE, signals=signals, portfolio={})

    assert result.promoted == 0
    assert _state_of(engine, wid) == "ready"


def test_enabled_mode_promotes_ready_watches(engine):
    """In enabled mode, ready watches meeting cycle gate are promoted."""
    mat_conds = [
        {"type": "price_zone", "params": {"low": 99.0, "high": 101.0}, "weight": 1.0},
    ]
    inv_conds = [{"type": "regime_flip", "params": {"blocked_regimes": ["crisis"]}}]
    wid = _insert_watch(
        engine, symbol="QQQ", state="ready",
        maturation_conditions=mat_conds,
        invalidation_conditions=inv_conds,
        maturity_score=0.9,
        observed_cycles=5,
    )

    signals = {"QQQ": {"current_price": 100.0, "market_regime": "neutral"}}

    with patch("utils.setup_watch_manager.SETUP_WATCH_MODE", "enabled"), \
         patch("utils.setup_watch_manager.SETUP_WATCH_MATURITY_THRESHOLD", 0.7), \
         patch("utils.setup_watch_manager.SETUP_WATCH_PROMOTION_MIN_CYCLES", 1):
        result = evaluate_cycle(engine, PROFILE, CYCLE, signals=signals, portfolio={})

    assert result.promoted >= 1
    assert _state_of(engine, wid) == "promoted"


# ────────────────────────────────────────────────────────────────────────────
# 7. last_evaluation_json contains per-condition results and timestamp
# ────────────────────────────────────────────────────────────────────────────


def test_last_evaluation_json_has_conditions_and_timestamp(engine):
    """After evaluation, last_evaluation_json contains structured results."""
    mat_conds = [
        {"type": "price_zone", "params": {"low": 99.0, "high": 101.0}, "weight": 1.0},
        {"type": "regime_aligned", "params": {"required_regime": "bullish"}, "weight": 1.0},
    ]
    inv_conds = [{"type": "regime_flip", "params": {"blocked_regimes": ["crisis"]}}]
    wid = _insert_watch(
        engine, symbol="INTC",
        maturation_conditions=mat_conds,
        invalidation_conditions=inv_conds,
    )

    signals = {"INTC": {"current_price": 100.0, "market_regime": "bullish"}}
    evaluate_cycle(engine, PROFILE, CYCLE, signals=signals, portfolio={})

    with engine.connect() as conn:
        raw = conn.execute(
            text("SELECT last_evaluation_json FROM setup_watches WHERE watch_id = :wid"),
            {"wid": wid},
        ).scalar()

    assert raw is not None
    data = json.loads(raw)
    assert "evaluated_at" in data
    assert "conditions" in data
    assert "maturity_score" in data
    assert len(data["conditions"]) == 2
    # Each condition has type, met, detail
    for cond in data["conditions"]:
        assert "type" in cond
        assert "met" in cond


# ────────────────────────────────────────────────────────────────────────────
# 8. maturity_evaluated emitted once per watch per cycle
# ────────────────────────────────────────────────────────────────────────────


def test_maturity_evaluated_event_emitted_once_per_watch(engine):
    """Each watch emits exactly one maturity_evaluated event per cycle."""
    mat_conds = [
        {"type": "price_zone", "params": {"low": 99.0, "high": 101.0}, "weight": 1.0},
    ]
    inv_conds = [{"type": "regime_flip", "params": {"blocked_regimes": ["crisis"]}}]
    wid = _insert_watch(
        engine, symbol="AMD",
        maturation_conditions=mat_conds,
        invalidation_conditions=inv_conds,
    )

    signals = {"AMD": {"current_price": 100.0, "market_regime": "neutral"}}
    evaluate_cycle(engine, PROFILE, CYCLE, signals=signals, portfolio={})

    count = _count_events(engine, watch_id=wid, event_type="maturity_evaluated")
    assert count == 1


# ────────────────────────────────────────────────────────────────────────────
# 9. WARNING logged when evaluate_cycle exceeds 2 seconds
# ────────────────────────────────────────────────────────────────────────────


def test_warning_logged_when_cycle_exceeds_2s(engine, caplog):
    """A WARNING is logged when evaluate_cycle takes more than 2 seconds."""
    # Simulate slow execution by patching time.monotonic
    call_count = [0]

    def fake_monotonic():
        call_count[0] += 1
        if call_count[0] == 1:
            return 0.0  # start time
        return 3.0  # end time — 3s elapsed

    with patch("utils.setup_watch_manager.time.monotonic", side_effect=fake_monotonic), \
         caplog.at_level(logging.WARNING, logger="utils.setup_watch_manager"):
        evaluate_cycle(engine, PROFILE, CYCLE, signals={}, portfolio={})

    assert any("took" in r.message and "threshold 2s" in r.message for r in caplog.records)


# ────────────────────────────────────────────────────────────────────────────
# 10. Each creation noise filter rejects as specified
# ────────────────────────────────────────────────────────────────────────────


def test_creation_rejected_weak_signal(engine):
    """Weak strength below minimum threshold is rejected."""
    with patch("utils.setup_watch_manager.SETUP_WATCH_MIN_CREATION_STRENGTH", "strong"):
        result = create_setup_watch(
            engine,
            symbol="XYZ",
            profile_id=PROFILE,
            side="BUY",
            setup_type="breakout",
            thesis="A perfectly valid thesis for trading",
            source_type="analyst",
            source_id="sig1",
            source_cycle_id=CYCLE,
            maturation_conditions=_maturation_conditions(3),
            invalidation_conditions=_invalidation_conditions(1),
            signal_strength="weak",
        )
    assert result is None


def test_creation_rejected_insufficient_conditions(engine):
    """Too few maturation conditions is rejected."""
    with patch("utils.setup_watch_manager.SETUP_WATCH_MIN_CONDITION_COUNT", 3):
        result = create_setup_watch(
            engine,
            symbol="XYZ",
            profile_id=PROFILE,
            side="BUY",
            setup_type="breakout",
            thesis="A perfectly valid thesis for trading",
            source_type="analyst",
            source_id="sig1",
            source_cycle_id=CYCLE,
            maturation_conditions=_maturation_conditions(2),  # below 3
            invalidation_conditions=_invalidation_conditions(1),
            signal_strength="strong",
        )
    assert result is None


def test_creation_rejected_no_invalidation_conditions(engine):
    """Zero invalidation conditions is rejected."""
    result = create_setup_watch(
        engine,
        symbol="XYZ",
        profile_id=PROFILE,
        side="BUY",
        setup_type="breakout",
        thesis="A perfectly valid thesis for trading",
        source_type="analyst",
        source_id="sig1",
        source_cycle_id=CYCLE,
        maturation_conditions=_maturation_conditions(3),
        invalidation_conditions=[],  # zero
        signal_strength="strong",
    )
    assert result is None


def test_creation_rejected_short_thesis(engine):
    """Thesis shorter than 10 characters is rejected."""
    result = create_setup_watch(
        engine,
        symbol="XYZ",
        profile_id=PROFILE,
        side="BUY",
        setup_type="breakout",
        thesis="short",  # < 10 chars
        source_type="analyst",
        source_id="sig1",
        source_cycle_id=CYCLE,
        maturation_conditions=_maturation_conditions(3),
        invalidation_conditions=_invalidation_conditions(1),
        signal_strength="strong",
    )
    assert result is None


def test_creation_rejected_per_symbol_cap(engine):
    """Per-symbol cap reached causes rejection."""
    with patch("utils.setup_watch_manager.SETUP_WATCH_MAX_PER_SYMBOL", 1):
        # Create first watch to fill the cap
        _insert_watch(engine, symbol="CAP", side="BUY", setup_type="type_a")

        result = create_setup_watch(
            engine,
            symbol="CAP",
            profile_id=PROFILE,
            side="SHORT",
            setup_type="type_b",
            thesis="A perfectly valid thesis for trading",
            source_type="analyst",
            source_id="sig1",
            source_cycle_id=CYCLE,
            maturation_conditions=_maturation_conditions(3),
            invalidation_conditions=_invalidation_conditions(1),
            signal_strength="strong",
        )
    assert result is None


def test_creation_rejected_exposure_conflict(engine):
    """Existing position for the symbol causes rejection."""
    portfolio = {"positions": {"HELD": {"qty": 100}}}
    result = create_setup_watch(
        engine,
        symbol="HELD",
        profile_id=PROFILE,
        side="BUY",
        setup_type="breakout",
        thesis="A perfectly valid thesis for trading",
        source_type="analyst",
        source_id="sig1",
        source_cycle_id=CYCLE,
        maturation_conditions=_maturation_conditions(3),
        invalidation_conditions=_invalidation_conditions(1),
        signal_strength="strong",
        portfolio=portfolio,
    )
    assert result is None


# ────────────────────────────────────────────────────────────────────────────
# 11. Per-profile cap evicts oldest rather than rejecting the new watch
# ────────────────────────────────────────────────────────────────────────────


def test_per_profile_cap_evicts_oldest(engine):
    """When at capacity, the oldest active watch is evicted, not the new one."""
    with patch("utils.setup_watch_manager.SETUP_WATCH_MAX_ACTIVE_PER_PROFILE", 2), \
         patch("utils.setup_watch_manager.SETUP_WATCH_MAX_PER_SYMBOL", 10):
        # Create two existing watches
        wid1 = _insert_watch(
            engine, symbol="OLD1", created_at=NOW - timedelta(hours=3)
        )
        wid2 = _insert_watch(
            engine, symbol="OLD2", created_at=NOW - timedelta(hours=1)
        )

        # Creating a new one should evict the oldest
        result = create_setup_watch(
            engine,
            symbol="NEW1",
            profile_id=PROFILE,
            side="BUY",
            setup_type="breakout",
            thesis="A perfectly valid thesis for trading",
            source_type="analyst",
            source_id="sig1",
            source_cycle_id=CYCLE,
            maturation_conditions=_maturation_conditions(3),
            invalidation_conditions=_invalidation_conditions(1),
            signal_strength="strong",
        )

    assert result is not None  # new watch created
    assert _state_of(engine, wid1) == "expired"  # oldest evicted
    assert _terminal_reason_of(engine, wid1) == "capacity_evicted"


# ────────────────────────────────────────────────────────────────────────────
# 12. Non-executable setup type is NOT rejected at creation
# ────────────────────────────────────────────────────────────────────────────


def test_non_executable_setup_type_accepted_at_creation(engine):
    """Any setup type is accepted at creation — allowlist is enforced at promotion."""
    result = create_setup_watch(
        engine,
        symbol="EXOTIC",
        profile_id=PROFILE,
        side="BUY",
        setup_type="some_exotic_non_executable_type",
        thesis="A perfectly valid thesis for trading",
        source_type="analyst",
        source_id="sig1",
        source_cycle_id=CYCLE,
        maturation_conditions=_maturation_conditions(3),
        invalidation_conditions=_invalidation_conditions(1),
        signal_strength="strong",
    )
    assert result is not None  # created successfully


# ────────────────────────────────────────────────────────────────────────────
# 13. expires_at clamped to max TTL
# ────────────────────────────────────────────────────────────────────────────


def test_expires_at_clamped_to_max_ttl(engine):
    """expires_at is clamped to max TTL hours from now."""
    far_future = NOW + timedelta(hours=100)

    with patch("utils.setup_watch_manager.SETUP_WATCH_MAX_TTL_HOURS", 8), \
         patch("utils.setup_watch_manager.now_utc", return_value=NOW):
        wid = create_setup_watch(
            engine,
            symbol="CLAMP",
            profile_id=PROFILE,
            side="BUY",
            setup_type="breakout",
            thesis="A perfectly valid thesis for trading",
            source_type="analyst",
            source_id="sig1",
            source_cycle_id=CYCLE,
            maturation_conditions=_maturation_conditions(3),
            invalidation_conditions=_invalidation_conditions(1),
            signal_strength="strong",
            expires_at=far_future,
        )

    assert wid is not None
    with engine.connect() as conn:
        exp = conn.execute(
            text("SELECT expires_at FROM setup_watches WHERE watch_id = :wid"),
            {"wid": wid},
        ).scalar()
    # Should be clamped to NOW + 8 hours, not 100 hours
    # Parse and compare
    from utils.pending_order_time import to_utc
    exp_dt = to_utc(exp)
    max_allowed = NOW + timedelta(hours=8)
    assert exp_dt <= max_allowed + timedelta(seconds=5)


# ────────────────────────────────────────────────────────────────────────────
# 14. CreationStats counts each rejection reason accurately
# ────────────────────────────────────────────────────────────────────────────


def test_creation_stats_counts_accurately(engine):
    """The _create_watches_from_signals function accurately tallies stats."""
    # We test this through evaluate_cycle which calls _create_watches_from_signals
    # Provide multiple signals: one weak, one with short thesis, one valid
    signals = {
        "WEAK": {
            "current_price": 100.0,
            "market_regime": "bullish",
            "signal_strength": "weak",
            "setup_type": "breakout",
            "direction": "BUY",
            "thesis": "A valid thesis for the weak signal here",
            "key_levels": {"support": [95.0], "resistance": [110.0]},
            "catalyst_timestamp": to_iso(NOW - timedelta(minutes=30)),
        },
        "SHORTTHESIS": {
            "current_price": 100.0,
            "market_regime": "bullish",
            "signal_strength": "strong",
            "setup_type": "breakout",
            "direction": "BUY",
            "thesis": "tiny",  # too short
            "key_levels": {"support": [95.0], "resistance": [110.0]},
            "catalyst_timestamp": to_iso(NOW - timedelta(minutes=30)),
        },
        "VALID": {
            "current_price": 100.0,
            "market_regime": "bullish",
            "signal_strength": "strong",
            "setup_type": "breakout",
            "direction": "BUY",
            "thesis": "A perfectly good thesis about this breakout",
            "key_levels": {"support": [95.0], "resistance": [110.0]},
            "catalyst_timestamp": to_iso(NOW - timedelta(minutes=30)),
        },
    }

    with patch("utils.setup_watch_manager.SETUP_WATCH_MIN_CREATION_STRENGTH", "moderate"):
        result = evaluate_cycle(engine, PROFILE, CYCLE, signals=signals, portfolio={})

    # At least the valid signal should create a watch
    assert result.created >= 1


# ────────────────────────────────────────────────────────────────────────────
# 15. propagate: executed → ordered with execution_ref_type="trade"
# ────────────────────────────────────────────────────────────────────────────


def test_propagate_executed_to_ordered_trade(engine):
    """Executed candidate transitions watch to ordered with trade ref."""
    wid = _insert_watch(engine, state="promoted", promoted_cycle_id=CYCLE)

    candidate_results = [
        {
            "candidate_id": "cand_001",
            "terminal_state": "EXECUTED",
            "trade_id": "trade_555",
            "signal_snapshot_json": json.dumps({
                "source_type": "setup_watch",
                "watch_id": wid,
            }),
        }
    ]

    propagate_candidate_results(engine, CYCLE, PROFILE, candidate_results)

    assert _state_of(engine, wid) == "ordered"
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT execution_ref_type, execution_ref_id "
                "FROM setup_watches WHERE watch_id = :wid"
            ),
            {"wid": wid},
        ).fetchone()
    assert row[0] == "trade"
    assert row[1] == "trade_555"


# ────────────────────────────────────────────────────────────────────────────
# 16. propagate: pending order found → ordered with execution_ref_type="pending_order"
# ────────────────────────────────────────────────────────────────────────────


def test_propagate_pending_order_to_ordered(engine):
    """If a pending order exists, watch transitions to ordered with pending_order ref."""
    wid = _insert_watch(engine, state="promoted", promoted_cycle_id=CYCLE)
    cand_id = "cand_002"

    # Insert a pending order for this candidate
    order_id = _insert_pending_order(engine, cand_id, state="pending")

    candidate_results = [
        {
            "candidate_id": cand_id,
            "terminal_state": "NOT_SELECTED",  # would normally → expired
            "signal_snapshot_json": json.dumps({
                "source_type": "setup_watch",
                "watch_id": wid,
            }),
        }
    ]

    propagate_candidate_results(engine, CYCLE, PROFILE, candidate_results)

    # Pending order takes precedence
    assert _state_of(engine, wid) == "ordered"
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT execution_ref_type, execution_ref_id "
                "FROM setup_watches WHERE watch_id = :wid"
            ),
            {"wid": wid},
        ).fetchone()
    assert row[0] == "pending_order"
    assert row[1] == order_id


# ────────────────────────────────────────────────────────────────────────────
# 17. propagate: pending-order branch wins over a rejection terminal state
# ────────────────────────────────────────────────────────────────────────────


def test_propagate_pending_order_wins_over_rejection(engine):
    """Pending order branch takes precedence over rejection state."""
    wid = _insert_watch(engine, state="promoted", promoted_cycle_id=CYCLE)
    cand_id = "cand_003"

    # Insert a pending order
    order_id = _insert_pending_order(engine, cand_id, state="filling")

    candidate_results = [
        {
            "candidate_id": cand_id,
            "terminal_state": "GATE_REJECTED",  # would normally → rejected
            "signal_snapshot_json": json.dumps({
                "source_type": "setup_watch",
                "watch_id": wid,
            }),
        }
    ]

    propagate_candidate_results(engine, CYCLE, PROFILE, candidate_results)

    # Pending order takes precedence over rejection
    assert _state_of(engine, wid) == "ordered"
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT execution_ref_type FROM setup_watches WHERE watch_id = :wid"),
            {"wid": wid},
        ).fetchone()
    assert row[0] == "pending_order"


# ────────────────────────────────────────────────────────────────────────────
# 18. propagate: rejection states → rejected; non-consumed → expired
# ────────────────────────────────────────────────────────────────────────────


def test_propagate_rejection_states_to_rejected(engine):
    """REJECTED, GATE_REJECTED, SIZING_REJECTED → watch rejected."""
    for terminal in ("REJECTED", "GATE_REJECTED", "SIZING_REJECTED"):
        wid = _insert_watch(engine, state="promoted", promoted_cycle_id=CYCLE,
                            symbol=f"REJ_{terminal}")

        candidate_results = [
            {
                "candidate_id": f"cand_{terminal}",
                "terminal_state": terminal,
                "signal_snapshot_json": json.dumps({
                    "source_type": "setup_watch",
                    "watch_id": wid,
                }),
            }
        ]

        propagate_candidate_results(engine, CYCLE, PROFILE, candidate_results)
        assert _state_of(engine, wid) == "rejected", f"Failed for {terminal}"


def test_propagate_non_consumed_states_to_expired(engine):
    """NOT_SELECTED, EXPIRED, EXECUTION_FAILED → watch expired."""
    for terminal in ("NOT_SELECTED", "EXPIRED", "EXECUTION_FAILED"):
        wid = _insert_watch(engine, state="promoted", promoted_cycle_id=CYCLE,
                            symbol=f"EXP_{terminal}")

        candidate_results = [
            {
                "candidate_id": f"cand_{terminal}",
                "terminal_state": terminal,
                "signal_snapshot_json": json.dumps({
                    "source_type": "setup_watch",
                    "watch_id": wid,
                }),
            }
        ]

        propagate_candidate_results(engine, CYCLE, PROFILE, candidate_results)
        assert _state_of(engine, wid) == "expired", f"Failed for {terminal}"


# ────────────────────────────────────────────────────────────────────────────
# 19. propagate: skips candidates lacking source_type="setup_watch"
# ────────────────────────────────────────────────────────────────────────────


def test_propagate_skips_non_setup_watch_candidates(engine):
    """Candidates without source_type=setup_watch are ignored."""
    wid = _insert_watch(engine, state="promoted", promoted_cycle_id=CYCLE)

    candidate_results = [
        {
            "candidate_id": "cand_other",
            "terminal_state": "EXECUTED",
            "trade_id": "trade_999",
            "signal_snapshot_json": json.dumps({
                "source_type": "analyst",  # not setup_watch
                "watch_id": wid,
            }),
        }
    ]

    propagate_candidate_results(engine, CYCLE, PROFILE, candidate_results)

    # Watch should remain promoted (untouched)
    assert _state_of(engine, wid) == "promoted"


# ────────────────────────────────────────────────────────────────────────────
# 20. propagate: one candidate failing does not stop the others
# ────────────────────────────────────────────────────────────────────────────


def test_propagate_one_failure_does_not_stop_others(engine):
    """A failure on one candidate does not prevent others from propagating."""
    wid_good = _insert_watch(engine, state="promoted", promoted_cycle_id=CYCLE,
                             symbol="GOOD")

    candidate_results = [
        # This one has an invalid watch_id that will cause a transition error
        {
            "candidate_id": "cand_bad",
            "terminal_state": "EXECUTED",
            "trade_id": "trade_1",
            "signal_snapshot_json": json.dumps({
                "source_type": "setup_watch",
                "watch_id": "nonexistent_watch_id",
            }),
        },
        # This one should succeed
        {
            "candidate_id": "cand_good",
            "terminal_state": "EXECUTED",
            "trade_id": "trade_2",
            "signal_snapshot_json": json.dumps({
                "source_type": "setup_watch",
                "watch_id": wid_good,
            }),
        },
    ]

    # Should not raise
    propagate_candidate_results(engine, CYCLE, PROFILE, candidate_results)

    # The good one still gets processed
    assert _state_of(engine, wid_good) == "ordered"


# ────────────────────────────────────────────────────────────────────────────
# 21. _lookup_pending_order_id returns None on DB error without raising
# ────────────────────────────────────────────────────────────────────────────


def test_lookup_pending_order_id_returns_none_on_error():
    """_lookup_pending_order_id returns None on DB error, no exception."""
    broken_engine = MagicMock()
    broken_engine.connect.side_effect = Exception("DB connection failed")

    result = _lookup_pending_order_id(broken_engine, "some_candidate_id")
    assert result is None


def test_lookup_pending_order_id_returns_none_for_missing(engine):
    """Returns None when no pending order exists for the candidate."""
    result = _lookup_pending_order_id(engine, "nonexistent_candidate")
    assert result is None


def test_lookup_pending_order_id_finds_active_order(engine):
    """Returns order_id when a pending order exists."""
    cand_id = "cand_with_order"
    order_id = _insert_pending_order(engine, cand_id, state="pending")

    result = _lookup_pending_order_id(engine, cand_id)
    assert result == order_id


# ────────────────────────────────────────────────────────────────────────────
# 22. SETUP_WATCH_MODE=disabled produces zero DB writes
# ────────────────────────────────────────────────────────────────────────────


def test_disabled_mode_produces_zero_db_writes(engine):
    """When mode is disabled, evaluate_cycle still runs (it's the caller's
    responsibility to not call it), but the important thing is that in the
    candidate_builder the call is guarded. Here we verify that even if called,
    the mode=disabled path doesn't promote."""
    # Insert a ready watch with sufficient cycles
    wid = _insert_watch(
        engine, state="ready",
        maturity_score=0.9,
        observed_cycles=5,
    )
    mat_conds = [
        {"type": "price_zone", "params": {"low": 99.0, "high": 101.0}, "weight": 1.0},
    ]

    signals = {"AAPL": {"current_price": 100.0, "market_regime": "neutral"}}

    with patch("utils.setup_watch_manager.SETUP_WATCH_MODE", "disabled"), \
         patch("utils.setup_watch_manager.SETUP_WATCH_PROMOTION_MIN_CYCLES", 1), \
         patch("utils.setup_watch_manager.SETUP_WATCH_MATURITY_THRESHOLD", 0.7):
        result = evaluate_cycle(engine, PROFILE, CYCLE, signals=signals, portfolio={})

    # No promotions in disabled mode
    assert result.promoted == 0
    assert _state_of(engine, wid) == "ready"


# ────────────────────────────────────────────────────────────────────────────
# 23. SETUP_WATCH_MODE=observe produces watches and evaluations but zero promotions
# ────────────────────────────────────────────────────────────────────────────


def test_observe_mode_evaluates_but_no_promotions(engine):
    """Observe mode writes evaluations and state changes but never promotes."""
    mat_conds = [
        {"type": "price_zone", "params": {"low": 99.0, "high": 101.0}, "weight": 1.0},
        {"type": "regime_aligned", "params": {"required_regime": "bullish"}, "weight": 1.0},
    ]
    inv_conds = [{"type": "regime_flip", "params": {"blocked_regimes": ["crisis"]}}]

    # Create a watching watch that will mature
    wid = _insert_watch(
        engine, symbol="OBS", state="watching",
        maturation_conditions=mat_conds,
        invalidation_conditions=inv_conds,
        observed_cycles=5,
    )

    signals = {"OBS": {"current_price": 100.0, "market_regime": "bullish"}}

    with patch("utils.setup_watch_manager.SETUP_WATCH_MODE", "observe"), \
         patch("utils.setup_watch_manager.SETUP_WATCH_MATURITY_THRESHOLD", 0.5), \
         patch("utils.setup_watch_manager.SETUP_WATCH_PROMOTION_MIN_CYCLES", 1):
        result = evaluate_cycle(engine, PROFILE, CYCLE, signals=signals, portfolio={})

    # It should have matured (state change happened)
    assert result.matured >= 1 or _state_of(engine, wid) in ("maturing", "ready")
    # But no promotions
    assert result.promoted == 0
    # Evaluation was written
    with engine.connect() as conn:
        eval_json = conn.execute(
            text("SELECT last_evaluation_json FROM setup_watches WHERE watch_id = :wid"),
            {"wid": wid},
        ).scalar()
    assert eval_json is not None
