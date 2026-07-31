"""Tests for PlanMonitor — trigger evaluation, rate limiting, and plan transitions.

Validates: plan activation, trigger evaluation, confirmation tracking, expiration,
invalidation, missed detection, rate limiting, and idempotency.

Requirements: 3.1–3.7, 4.1–4.14
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy import create_engine, text

from utils.trade_plan_registry import (
    PlanState,
    TradePlan,
    TradePlanRegistry,
)
from utils.plan_monitor import (
    PlanMonitor,
    TriggerEvaluation,
    MonitorTickResult,
    _QuoteRateState,
    _quote_rate_state,
    _confirmation_ticks,
    run,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_tables(engine):
    """Create trade_plans and trade_plan_events tables in memory."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS trade_plans (
                plan_id TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL,
                cycle_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                setup_type TEXT NOT NULL,
                geometry_name TEXT,
                entry_reference REAL NOT NULL,
                entry_zone_upper REAL NOT NULL,
                entry_zone_lower REAL NOT NULL,
                stop_price REAL NOT NULL,
                target_price REAL NOT NULL,
                risk_reward REAL NOT NULL,
                trigger_type TEXT NOT NULL,
                trigger_condition_json TEXT NOT NULL,
                trigger_confirmation_required INTEGER NOT NULL DEFAULT 0,
                invalidation_logic_json TEXT,
                analyst_reasoning TEXT,
                pm_rationale TEXT,
                source_signal_id TEXT,
                signal_snapshot_json TEXT,
                state TEXT NOT NULL DEFAULT 'planned',
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                triggered_at TEXT,
                executed_at TEXT,
                missed_at TEXT,
                miss_reason TEXT,
                rejection_reason TEXT,
                integrity_hash TEXT NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS trade_plan_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id TEXT NOT NULL,
                cycle_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_data TEXT,
                fresh_price REAL,
                from_state TEXT,
                to_state TEXT,
                created_at TEXT NOT NULL
            )
        """))


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    _create_tables(eng)
    return eng


@pytest.fixture
def registry(engine):
    return TradePlanRegistry(engine)


@pytest.fixture(autouse=True)
def reset_module_state():
    """Reset module-level state between tests."""
    _confirmation_ticks.clear()
    _quote_rate_state.symbol_last_fetched.clear()
    _quote_rate_state.calls_this_window.clear()
    yield
    _confirmation_ticks.clear()
    _quote_rate_state.symbol_last_fetched.clear()
    _quote_rate_state.calls_this_window.clear()


def _make_plan(
    plan_id: str = "plan-001",
    candidate_id: str = "cand-001",
    symbol: str = "TSLA",
    direction: str = "BUY",
    setup_type: str = "momentum_fade",
    state: PlanState = PlanState.PLANNED,
    expires_at: datetime | None = None,
    entry_reference: float = 250.0,
    entry_zone_upper: float = 252.0,
    entry_zone_lower: float = 248.0,
    stop_price: float = 245.0,
    target_price: float = 260.0,
    trigger_confirmation_required: bool = False,
    invalidation_logic_json: str | None = None,
) -> TradePlan:
    """Build a TradePlan with reasonable defaults for testing."""
    now = datetime.now(timezone.utc)
    if expires_at is None:
        expires_at = now + timedelta(minutes=60)
    return TradePlan(
        plan_id=plan_id,
        candidate_id=candidate_id,
        cycle_id="cycle-001",
        profile_id="aggressive",
        symbol=symbol,
        direction=direction,
        setup_type=setup_type,
        geometry_name="standard",
        entry_reference=entry_reference,
        entry_zone_upper=entry_zone_upper,
        entry_zone_lower=entry_zone_lower,
        stop_price=stop_price,
        target_price=target_price,
        risk_reward=2.0,
        trigger_type="price_in_zone",
        trigger_condition_json=json.dumps({"type": "price_in_zone"}),
        trigger_confirmation_required=trigger_confirmation_required,
        invalidation_logic_json=invalidation_logic_json,
        analyst_reasoning="Strong setup",
        pm_rationale="Approved",
        source_signal_id="sig-001",
        signal_snapshot_json=json.dumps({"symbol": symbol}),
        state=state,
        created_at=now,
        expires_at=expires_at,
        triggered_at=None,
        executed_at=None,
        missed_at=None,
        integrity_hash="",
    )


def _insert_plan(registry, plan):
    """Insert a plan directly into DB."""
    registry.create_plan(plan)


def _patch_quotes(quote_dict):
    """Patch the price_monitor imports used inside PlanMonitor._get_rate_limited_quotes.

    Returns a context manager that patches both _quote_cache and get_batch_quotes.
    """
    cache = {}
    for sym, price in quote_dict.items():
        cache[sym] = (time.time() - 5, price)  # Fresh (5s old)

    return patch(
        "agents.price_monitor._quote_cache", cache
    )


def _patch_quotes_stale(quote_dict, age_seconds=60):
    """Patch with stale cache entries."""
    cache = {}
    for sym, price in quote_dict.items():
        cache[sym] = (time.time() - age_seconds, price)
    return patch("agents.price_monitor._quote_cache", cache)


# ---------------------------------------------------------------------------
# Test: PLANNED plans transition to WATCHING on first tick
# ---------------------------------------------------------------------------


def test_planned_plans_transition_to_watching_on_first_tick(engine, registry):
    plan = _make_plan()
    _insert_plan(registry, plan)

    with patch("agents.price_monitor._quote_cache", {}), \
         patch("agents.price_monitor.get_batch_quotes", return_value={}):
        monitor = PlanMonitor(engine)
        monitor.run()

    updated = registry.get_plan("plan-001")
    assert updated.state == PlanState.WATCHING


# ---------------------------------------------------------------------------
# Test: price inside entry zone triggers plan
# ---------------------------------------------------------------------------


def test_price_in_zone_triggers_plan(engine, registry):
    plan = _make_plan(state=PlanState.PLANNED)
    _insert_plan(registry, plan)
    registry.activate("plan-001")

    # Price 250 is inside zone [248, 252]
    now = time.time()
    cache = {"TSLA": (now - 5, 250.0)}

    with patch("agents.price_monitor._quote_cache", cache), \
         patch("agents.price_monitor.get_batch_quotes", return_value={"TSLA": 250.0}):
        monitor = PlanMonitor(engine)
        result = monitor.run()

    updated = registry.get_plan("plan-001")
    assert updated.state == PlanState.TRIGGERED
    assert result.plans_triggered == 1


# ---------------------------------------------------------------------------
# Test: price outside entry zone does NOT trigger
# ---------------------------------------------------------------------------


def test_price_outside_zone_does_not_trigger(engine, registry):
    plan = _make_plan(state=PlanState.PLANNED)
    _insert_plan(registry, plan)
    registry.activate("plan-001")

    # Price 240 is below zone [248, 252] (well outside even with tolerance)
    now = time.time()
    cache = {"TSLA": (now - 5, 240.0)}

    with patch("agents.price_monitor._quote_cache", cache):
        monitor = PlanMonitor(engine)
        result = monitor.run()

    updated = registry.get_plan("plan-001")
    assert updated.state == PlanState.WATCHING
    assert result.plans_triggered == 0


# ---------------------------------------------------------------------------
# Test: price past target (long) marks MISSED
# ---------------------------------------------------------------------------


def test_price_past_target_long_marks_missed(engine, registry):
    plan = _make_plan(direction="BUY", target_price=260.0)
    _insert_plan(registry, plan)
    registry.activate("plan-001")

    # Price 265 is past target 260 for a long
    now = time.time()
    cache = {"TSLA": (now - 5, 265.0)}

    with patch("agents.price_monitor._quote_cache", cache):
        monitor = PlanMonitor(engine)
        result = monitor.run()

    updated = registry.get_plan("plan-001")
    assert updated.state == PlanState.MISSED
    assert result.plans_missed == 1


# ---------------------------------------------------------------------------
# Test: price past target (short) marks MISSED
# ---------------------------------------------------------------------------


def test_price_past_target_short_marks_missed(engine, registry):
    # SHORT: entry_ref=100, zone=[100, 102], target=95, stop=105
    plan = _make_plan(
        direction="SHORT",
        entry_reference=100.0,
        entry_zone_upper=102.0,
        entry_zone_lower=100.0,
        target_price=95.0,
        stop_price=105.0,
    )
    _insert_plan(registry, plan)
    registry.activate("plan-001")

    # Price 93 is past target 95 for a short
    now = time.time()
    cache = {"TSLA": (now - 5, 93.0)}

    with patch("agents.price_monitor._quote_cache", cache):
        monitor = PlanMonitor(engine)
        result = monitor.run()

    updated = registry.get_plan("plan-001")
    assert updated.state == PlanState.MISSED
    assert result.plans_missed == 1


# ---------------------------------------------------------------------------
# Test: expired plan transitions to EXPIRED without trigger evaluation
# ---------------------------------------------------------------------------


def test_expired_plan_transitions_to_expired(engine, registry):
    expired_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    plan = _make_plan(expires_at=expired_time)
    _insert_plan(registry, plan)
    registry.activate("plan-001")

    with patch("agents.price_monitor._quote_cache", {}), \
         patch("agents.price_monitor.get_batch_quotes", return_value={"TSLA": 250.0}) as mock_batch:
        monitor = PlanMonitor(engine)
        result = monitor.run()

    updated = registry.get_plan("plan-001")
    assert updated.state == PlanState.EXPIRED
    assert result.plans_expired == 1
    # Provider should NOT be called since plan was expired before quote fetch
    mock_batch.assert_not_called()


# ---------------------------------------------------------------------------
# Test: invalidation condition met transitions to REJECTED
# ---------------------------------------------------------------------------


def test_invalidation_condition_transitions_to_rejected(engine, registry):
    plan = _make_plan(
        invalidation_logic_json=json.dumps({"type": "price_below", "level": 244.0}),
    )
    _insert_plan(registry, plan)
    registry.activate("plan-001")

    # Price 242 is below invalidation level 244
    now = time.time()
    cache = {"TSLA": (now - 5, 242.0)}

    with patch("agents.price_monitor._quote_cache", cache):
        monitor = PlanMonitor(engine)
        result = monitor.run()

    updated = registry.get_plan("plan-001")
    assert updated.state == PlanState.REJECTED
    assert result.plans_invalidated == 1


# ---------------------------------------------------------------------------
# Test: confirmation_required=True requires consecutive ticks in zone
# ---------------------------------------------------------------------------


def test_confirmation_requires_consecutive_ticks(engine, registry):
    plan = _make_plan(trigger_confirmation_required=True)
    _insert_plan(registry, plan)
    registry.activate("plan-001")

    now = time.time()
    cache = {"TSLA": (now - 5, 250.0)}

    # Tick 1: in zone but confirmation pending (need 2 ticks)
    with patch("agents.price_monitor._quote_cache", cache):
        monitor = PlanMonitor(engine)
        result = monitor.run()

    updated = registry.get_plan("plan-001")
    assert updated.state == PlanState.WATCHING
    assert result.plans_triggered == 0

    # Tick 2: still in zone → should trigger
    cache2 = {"TSLA": (time.time() - 5, 250.0)}
    with patch("agents.price_monitor._quote_cache", cache2):
        monitor2 = PlanMonitor(engine)
        result2 = monitor2.run()

    updated2 = registry.get_plan("plan-001")
    assert updated2.state == PlanState.TRIGGERED
    assert result2.plans_triggered == 1


# ---------------------------------------------------------------------------
# Test: confirmation resets if price leaves zone between ticks
# ---------------------------------------------------------------------------


def test_confirmation_resets_on_zone_exit(engine, registry):
    plan = _make_plan(trigger_confirmation_required=True)
    _insert_plan(registry, plan)
    registry.activate("plan-001")

    # Tick 1: in zone
    cache1 = {"TSLA": (time.time() - 5, 250.0)}
    with patch("agents.price_monitor._quote_cache", cache1):
        PlanMonitor(engine).run()

    updated = registry.get_plan("plan-001")
    assert updated.state == PlanState.WATCHING

    # Tick 2: price leaves zone (resets confirmation)
    cache2 = {"TSLA": (time.time() - 5, 240.0)}
    with patch("agents.price_monitor._quote_cache", cache2):
        PlanMonitor(engine).run()

    # Tick 3: back in zone — count resets, need 2 more consecutive
    cache3 = {"TSLA": (time.time() - 5, 250.0)}
    with patch("agents.price_monitor._quote_cache", cache3):
        result3 = PlanMonitor(engine).run()

    updated3 = registry.get_plan("plan-001")
    assert updated3.state == PlanState.WATCHING
    assert result3.plans_triggered == 0


# ---------------------------------------------------------------------------
# Test: already triggered plan is not re-triggered
# ---------------------------------------------------------------------------


def test_already_triggered_plan_not_retriggered(engine, registry):
    plan = _make_plan()
    _insert_plan(registry, plan)
    registry.activate("plan-001")
    registry.trigger("plan-001")

    now = time.time()
    cache = {"TSLA": (now - 5, 250.0)}

    with patch("agents.price_monitor._quote_cache", cache):
        monitor = PlanMonitor(engine)
        result = monitor.run()

    # Already triggered plans are not in active_plans (PLANNED/WATCHING only)
    assert result.plans_triggered == 0
    updated = registry.get_plan("plan-001")
    assert updated.state == PlanState.TRIGGERED


# ---------------------------------------------------------------------------
# Test: monitor tick with no active plans returns zeros
# ---------------------------------------------------------------------------


def test_no_active_plans_returns_zeros(engine):
    with patch("agents.price_monitor._quote_cache", {}), \
         patch("agents.price_monitor.get_batch_quotes", return_value={}) as mock_batch:
        monitor = PlanMonitor(engine)
        result = monitor.run()

    assert result.plans_checked == 0
    assert result.plans_triggered == 0
    assert result.plans_expired == 0
    assert result.plans_missed == 0
    assert result.plans_invalidated == 0
    mock_batch.assert_not_called()


# ---------------------------------------------------------------------------
# Test: cached quote within tolerance used without provider call
# ---------------------------------------------------------------------------


def test_cached_quote_within_tolerance_uses_cache(engine, registry):
    plan = _make_plan()
    _insert_plan(registry, plan)
    registry.activate("plan-001")

    # Fresh cache (< 30s old)
    now = time.time()
    cache = {"TSLA": (now - 5, 250.0)}

    with patch("agents.price_monitor._quote_cache", cache), \
         patch("agents.price_monitor.get_batch_quotes") as mock_batch:
        monitor = PlanMonitor(engine)
        result = monitor.run()

    # Should use cache, not call provider
    mock_batch.assert_not_called()
    assert result.quotes_from_cache == 1
    assert result.provider_calls_made == 0


# ---------------------------------------------------------------------------
# Test: stale cached quote triggers provider fetch if budget permits
# ---------------------------------------------------------------------------


def test_stale_cache_triggers_provider_fetch(engine, registry):
    plan = _make_plan()
    _insert_plan(registry, plan)
    registry.activate("plan-001")

    # Stale cache (> 30s old)
    now = time.time()
    cache = {"TSLA": (now - 60, 250.0)}

    with patch("agents.price_monitor._quote_cache", cache), \
         patch("agents.price_monitor.get_batch_quotes", return_value={"TSLA": 251.0}) as mock_batch:
        monitor = PlanMonitor(engine)
        result = monitor.run()

    mock_batch.assert_called_once()
    assert result.provider_calls_made == 1


# ---------------------------------------------------------------------------
# Test: QUOTE_PROVIDER_MAX_CALLS_PER_MINUTE cap honored
# ---------------------------------------------------------------------------


def test_global_rate_cap_honored(engine, registry):
    plan = _make_plan()
    _insert_plan(registry, plan)
    registry.activate("plan-001")

    # Exhaust global budget
    now = time.time()
    _quote_rate_state.calls_this_window = [now - i for i in range(40)]

    # Stale cache
    cache = {"TSLA": (now - 60, 249.0)}

    with patch("agents.price_monitor._quote_cache", cache), \
         patch("agents.price_monitor.get_batch_quotes") as mock_batch:
        monitor = PlanMonitor(engine)
        result = monitor.run()

    # Should NOT call provider, should use stale cache
    mock_batch.assert_not_called()
    assert result.quotes_from_cache == 1
    assert result.provider_calls_made == 0


# ---------------------------------------------------------------------------
# Test: QUOTE_PROVIDER_MIN_SECONDS_PER_SYMBOL prevents re-fetching too soon
# ---------------------------------------------------------------------------


def test_per_symbol_rate_limit_prevents_refetch(engine, registry):
    plan = _make_plan()
    _insert_plan(registry, plan)
    registry.activate("plan-001")

    # Symbol was fetched recently (< 30s ago)
    now = time.time()
    _quote_rate_state.symbol_last_fetched["TSLA"] = now - 10  # 10s ago

    # Stale cache
    cache = {"TSLA": (now - 60, 249.0)}

    with patch("agents.price_monitor._quote_cache", cache), \
         patch("agents.price_monitor.get_batch_quotes") as mock_batch:
        monitor = PlanMonitor(engine)
        result = monitor.run()

    # Should NOT call provider (per-symbol limit), use stale cache
    mock_batch.assert_not_called()
    assert result.quotes_from_cache == 1


# ---------------------------------------------------------------------------
# Test: provider 429 stops further calls, plans stay WATCHING
# ---------------------------------------------------------------------------


def test_provider_429_circuit_breaks_plans_stay_watching(engine, registry):
    # Create two plans for different symbols — prices OUTSIDE zone
    # so we can verify plans stay WATCHING (not transitioned to MISSED)
    plan1 = _make_plan(plan_id="plan-001", symbol="TSLA")
    plan2 = _make_plan(plan_id="plan-002", symbol="AAPL",
                       entry_reference=150.0, entry_zone_upper=152.0,
                       entry_zone_lower=148.0, stop_price=145.0,
                       target_price=160.0)
    _insert_plan(registry, plan1)
    _insert_plan(registry, plan2)
    registry.activate("plan-001")
    registry.activate("plan-002")

    # Stale cache with prices outside zone (won't trigger, won't miss)
    now = time.time()
    cache = {
        "TSLA": (now - 60, 255.0),  # Outside zone [248, 252] but not past target
        "AAPL": (now - 60, 153.0),  # Outside zone [148, 152] but not past target
    }

    def raise_429(symbols, **kwargs):
        raise Exception("429 Too Many Requests")

    with patch("agents.price_monitor._quote_cache", cache), \
         patch("agents.price_monitor.get_batch_quotes", side_effect=raise_429):
        monitor = PlanMonitor(engine)
        result = monitor.run()

    # Both plans should stay WATCHING (not MISSED) — 429 doesn't kill plans
    p1 = registry.get_plan("plan-001")
    p2 = registry.get_plan("plan-002")
    assert p1.state == PlanState.WATCHING
    assert p2.state == PlanState.WATCHING
    assert result.plans_missed == 0
    assert result.plans_triggered == 0


# ---------------------------------------------------------------------------
# Test: no quote available for symbol → plan stays WATCHING
# ---------------------------------------------------------------------------


def test_no_quote_available_plan_stays_watching(engine, registry):
    plan = _make_plan()
    _insert_plan(registry, plan)
    registry.activate("plan-001")

    with patch("agents.price_monitor._quote_cache", {}), \
         patch("agents.price_monitor.get_batch_quotes", return_value={}):
        monitor = PlanMonitor(engine)
        result = monitor.run()

    updated = registry.get_plan("plan-001")
    assert updated.state == PlanState.WATCHING
    assert result.plans_missed == 0


# ---------------------------------------------------------------------------
# Test: run() returns immediately when TRIGGERED_PLAN_MODE=disabled
# ---------------------------------------------------------------------------


def test_run_returns_immediately_when_disabled(engine):
    with patch("utils.plan_monitor.TRIGGERED_PLAN_MODE", "disabled"):
        result = run(engine)

    assert result.plans_checked == 0
    assert result.tick_duration_ms == 0.0


# ---------------------------------------------------------------------------
# Test: tick is idempotent — re-running with same state produces same result
# ---------------------------------------------------------------------------


def test_tick_is_idempotent(engine, registry):
    plan = _make_plan()
    _insert_plan(registry, plan)
    registry.activate("plan-001")

    # Price outside zone — no transition
    now = time.time()
    cache = {"TSLA": (now - 5, 240.0)}

    with patch("agents.price_monitor._quote_cache", cache):
        result1 = PlanMonitor(engine).run()

    cache2 = {"TSLA": (time.time() - 5, 240.0)}
    with patch("agents.price_monitor._quote_cache", cache2):
        result2 = PlanMonitor(engine).run()

    assert result1.plans_triggered == result2.plans_triggered == 0
    assert result1.plans_missed == result2.plans_missed == 0
    updated = registry.get_plan("plan-001")
    assert updated.state == PlanState.WATCHING


# ---------------------------------------------------------------------------
# Test: MonitorTickResult includes correct quote metrics
# ---------------------------------------------------------------------------


def test_monitor_tick_result_includes_quote_metrics(engine, registry):
    plan = _make_plan()
    _insert_plan(registry, plan)
    registry.activate("plan-001")

    now = time.time()
    cache = {"TSLA": (now - 5, 250.0)}

    with patch("agents.price_monitor._quote_cache", cache), \
         patch("agents.price_monitor.get_batch_quotes") as mock_batch:
        monitor = PlanMonitor(engine)
        result = monitor.run()

    assert result.quotes_from_cache == 1
    assert result.provider_calls_made == 0
    assert result.tick_duration_ms >= 0


# ---------------------------------------------------------------------------
# Test: _quote_rate_state is module-level — persists across instances
# ---------------------------------------------------------------------------


def test_quote_rate_state_persists_across_instances(engine):
    """_quote_rate_state is module-level and survives PlanMonitor re-instantiation."""
    _quote_rate_state.record_fetch("TSLA", time.time())

    # Create new PlanMonitor instances — state should persist
    PlanMonitor(engine)
    PlanMonitor(engine)

    assert "TSLA" in _quote_rate_state.symbol_last_fetched
    assert len(_quote_rate_state.calls_this_window) == 1


# ---------------------------------------------------------------------------
# Test: _QuoteRateState.can_fetch_global prunes old entries
# ---------------------------------------------------------------------------


def test_quote_rate_state_prunes_old_entries():
    state = _QuoteRateState()
    now = time.time()
    # Add calls from >60s ago
    state.calls_this_window = [now - 120, now - 90, now - 70]

    assert state.can_fetch_global(now, 40) is True
    # Old entries should be pruned
    assert len(state.calls_this_window) == 0


# ---------------------------------------------------------------------------
# Test: TriggerEvaluation frozen dataclass
# ---------------------------------------------------------------------------


def test_trigger_evaluation_is_frozen():
    now = datetime.now(timezone.utc)
    ev = TriggerEvaluation(
        triggered=True,
        reason="price_in_zone",
        fresh_price=250.0,
        evaluation_timestamp=now,
    )
    assert ev.triggered is True
    assert ev.reason == "price_in_zone"
    assert ev.fresh_price == 250.0
    assert ev.missed is False
    assert ev.invalidated is False

    with pytest.raises(Exception):  # FrozenInstanceError
        ev.triggered = False


# ---------------------------------------------------------------------------
# Test: MonitorTickResult dataclass defaults
# ---------------------------------------------------------------------------


def test_monitor_tick_result_defaults():
    result = MonitorTickResult()
    assert result.plans_checked == 0
    assert result.plans_triggered == 0
    assert result.plans_expired == 0
    assert result.plans_invalidated == 0
    assert result.plans_missed == 0
    assert result.quotes_fetched == 0
    assert result.quotes_from_cache == 0
    assert result.provider_calls_made == 0
    assert result.tick_duration_ms == 0.0
