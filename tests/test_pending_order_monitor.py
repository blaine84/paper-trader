"""Tests for utils/pending_order_monitor.py.

The tick order is the thing under test as much as the individual checks:
cancellation must precede fill detection, expiry must precede filling, and bars
must be fetched once per symbol rather than once per order.

Requirements: 3.7, 3.8, 3.10, 4.1, 4.3, 4.9, 4.12, 4.14, 4.15, 7.1, 7.2, 7.5,
              7.10, 9.12, 10.5
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from db.schema import AgentMemory, Base, Position, get_session, init_pending_order_schema
from utils.pending_order_monitor import (
    MonitorTickResult,
    PendingOrderMonitor,
    run,
)
from utils.pending_order_registry import (
    OrderState,
    PendingOrder,
    PendingOrderRegistry,
)

LIMIT = 593.87
STOP = 585.00
TARGET = 620.00


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    init_pending_order_schema(eng)
    return eng


@pytest.fixture
def registry(engine):
    return PendingOrderRegistry(engine)


@pytest.fixture
def enabled():
    with patch("utils.pending_order_monitor.PENDING_ORDER_MODE", "enabled"):
        yield


def make_order(registry, **overrides) -> PendingOrder:
    """Create a live PENDING order whose window is open."""
    now = datetime.now(timezone.utc)
    defaults = dict(
        order_id=str(uuid.uuid4()),
        profile_id="moderate",
        symbol="META",
        side="BUY",
        setup_type="technical_breakout",
        limit_price=LIMIT,
        stop_price=STOP,
        target_price=TARGET,
        risk_reward=2.9,
        fresh_price_at_creation=601.24,
        runaway_pct_at_creation=0.0124,
        created_at=now - timedelta(minutes=30),
        expires_at=now + timedelta(hours=1),
    )
    defaults.update(overrides)
    order = PendingOrder(**defaults)
    registry.create_order(order)
    return registry.get_order(order.order_id)


def candles(*, lows, symbol="META", minutes_ago_start=20):
    """Build a get_candles() payload whose bars sit inside the active window."""
    base = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago_start)
    timestamps = [
        int((base + timedelta(minutes=i)).timestamp()) for i in range(len(lows))
    ]
    return {
        "symbol": symbol,
        "resolution": "1",
        "timestamps": timestamps,
        "open": [low + 3.0 for low in lows],
        "high": [low + 4.0 for low in lows],
        "low": list(lows),
        "close": [low + 2.0 for low in lows],
        "volume": [1000] * len(lows),
        "source": "alpaca",
    }


def patch_candles(payload_or_fn):
    """Patch FinnhubClient so get_candles returns a fixed payload."""
    client = MagicMock()
    if callable(payload_or_fn):
        client.get_candles.side_effect = payload_or_fn
    else:
        client.get_candles.return_value = payload_or_fn
    factory = MagicMock(return_value=client)
    return patch("utils.finnhub_client.FinnhubClient", factory), client


def stub_filler(success=True, reason="filled"):
    """Patch the filler so monitor tests do not exercise execution.

    Asserts the real filler's precondition: the order handed over must already
    be in FILLING. A plain MagicMock would silently accept a stale PENDING copy,
    which is exactly the bug this guard exists to catch.
    """
    result = MagicMock()
    result.success = success
    result.reason = reason

    def _checked(engine, order, bar):
        assert order.state is OrderState.FILLING, (
            f"filler was handed an order in state {order.state.value}; the "
            f"monitor must re-load after the CAS claim"
        )
        return result

    return patch(
        "utils.pending_order_filler.fill_pending_order", side_effect=_checked
    )


def write_signal(engine, symbol, direction, **extra):
    payload = {"symbol": symbol, "signal": direction, "setup_type": "technical_breakout"}
    payload.update(extra)
    session = get_session(engine)
    try:
        session.add(
            AgentMemory(
                agent="analyst", symbol=symbol, key="signal",
                value=json.dumps(payload),
                timestamp=datetime.utcnow(),
            )
        )
        session.commit()
    finally:
        session.close()


def trade_events(engine, event_type=None):
    clause = " WHERE event_type = :et" if event_type else ""
    params = {"et": event_type} if event_type else {}
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"SELECT event_type, payload_json FROM trade_events{clause} "
                f"ORDER BY id ASC"
            ),
            params,
        ).mappings().all()
    out = []
    for row in rows:
        record = dict(row)
        if record.get("payload_json"):
            record["payload"] = json.loads(record["payload_json"])
        out.append(record)
    return out


# ---------------------------------------------------------------------------
# Mode guard
# ---------------------------------------------------------------------------


def test_disabled_mode_makes_no_provider_call_and_no_transition(engine, registry):
    order = make_order(registry)
    ctx, client = patch_candles(candles(lows=[593.0]))

    with patch("utils.pending_order_monitor.PENDING_ORDER_MODE", "disabled"):
        with ctx:
            result = run(engine)

    assert result == MonitorTickResult()
    client.get_candles.assert_not_called()
    assert registry.get_order(order.order_id).state is OrderState.PENDING


def test_empty_registry_tick_is_harmless(engine, enabled):
    ctx, client = patch_candles(candles(lows=[593.0]))
    with ctx:
        result = run(engine)

    assert result.orders_checked == 0
    client.get_candles.assert_not_called()


# ---------------------------------------------------------------------------
# Fill detection
# ---------------------------------------------------------------------------


def test_crossing_bar_drives_a_fill_attempt(engine, registry, enabled):
    order = make_order(registry)
    ctx, _ = patch_candles(candles(lows=[599.0, 593.0]))

    with ctx, stub_filler(success=True) as filler:
        result = run(engine)

    assert result.orders_filled == 1
    filler.assert_called_once()
    # The order was claimed before dispatch.
    dispatched_order = filler.call_args[0][1]
    assert dispatched_order.order_id == order.order_id


def test_no_crossing_leaves_the_order_pending_and_advances_the_watermark(
    engine, registry, enabled
):
    order = make_order(registry)
    ctx, _ = patch_candles(candles(lows=[599.0, 598.0, 597.0]))

    with ctx, stub_filler() as filler:
        result = run(engine)

    assert result.orders_filled == 0
    filler.assert_not_called()

    reloaded = registry.get_order(order.order_id)
    assert reloaded.state is OrderState.PENDING
    assert reloaded.last_evaluated_bar_ts is not None


def test_short_order_fills_on_a_high_crossing(engine, registry, enabled):
    make_order(
        registry, symbol="AMD", side="SHORT",
        limit_price=100.0, stop_price=105.0, target_price=90.0,
    )
    payload = candles(lows=[96.0, 99.0], symbol="AMD")
    # high = low + 4, so the second bar reaches 103 and crosses a SHORT limit of 100.
    ctx, _ = patch_candles(payload)

    with ctx, stub_filler(success=True) as filler:
        result = run(engine)

    assert result.orders_filled == 1
    filler.assert_called_once()


def test_bar_predating_creation_does_not_fill(engine, registry, enabled):
    """End-to-end anti-stale-fill guarantee through the monitor path."""
    now = datetime.now(timezone.utc)
    make_order(registry, created_at=now - timedelta(minutes=2))

    # A crossing bar 10 minutes before the order existed.
    old = now - timedelta(minutes=10)
    payload = {
        "symbol": "META", "resolution": "1",
        "timestamps": [int(old.timestamp())],
        "open": [596.0], "high": [597.0], "low": [590.0], "close": [591.0],
    }
    ctx, _ = patch_candles(payload)

    with ctx, stub_filler() as filler:
        result = run(engine)

    assert result.orders_filled == 0
    filler.assert_not_called()


def test_watermark_prevents_a_second_fill_attempt(engine, registry, enabled):
    """Two identical ticks must produce exactly one fill attempt."""
    make_order(registry)
    ctx, _ = patch_candles(candles(lows=[599.0, 593.0]))

    with ctx, stub_filler(success=True) as filler:
        run(engine)
        run(engine)

    assert filler.call_count == 1


def test_claim_is_atomic_across_ticks(engine, registry, enabled):
    """A second tick cannot claim an order already in FILLING."""
    order = make_order(registry)
    registry.claim_for_fill(order.order_id)  # simulate an in-flight tick

    ctx, _ = patch_candles(candles(lows=[593.0]))
    with ctx, stub_filler() as filler:
        result = run(engine)

    # FILLING orders are not returned by get_pending_orders.
    assert result.orders_checked == 0
    filler.assert_not_called()


# ---------------------------------------------------------------------------
# Gap-through
# ---------------------------------------------------------------------------


def test_gap_through_cancels_instead_of_filling(engine, registry, enabled):
    order = make_order(registry)
    now = datetime.now(timezone.utc)
    # Opens at 570, far below the 593.87 limit (1.5% threshold == 8.91).
    payload = {
        "symbol": "META", "resolution": "1",
        "timestamps": [int((now - timedelta(minutes=5)).timestamp())],
        "open": [570.0], "high": [575.0], "low": [565.0], "close": [568.0],
    }
    ctx, _ = patch_candles(payload)

    with ctx, stub_filler() as filler:
        result = run(engine)

    assert result.orders_canceled == 1
    assert result.orders_filled == 0
    filler.assert_not_called()

    reloaded = registry.get_order(order.order_id)
    assert reloaded.state is OrderState.CANCELED
    assert reloaded.terminal_reason == "gap_through"


def test_gap_through_emits_a_cancel_event(engine, registry, enabled):
    make_order(registry)
    now = datetime.now(timezone.utc)
    payload = {
        "symbol": "META", "resolution": "1",
        "timestamps": [int((now - timedelta(minutes=5)).timestamp())],
        "open": [570.0], "high": [575.0], "low": [565.0], "close": [568.0],
    }
    ctx, _ = patch_candles(payload)

    with ctx, stub_filler():
        run(engine)

    events = trade_events(engine, "pending_order_canceled")
    assert len(events) == 1
    assert events[0]["payload"]["reason"] == "gap_through"


# ---------------------------------------------------------------------------
# Cancellation checks precede fill detection
# ---------------------------------------------------------------------------


def test_flipped_signal_cancels_before_any_fill(engine, registry, enabled):
    """The crossing bar is present, but the thesis died first."""
    order = make_order(registry)
    write_signal(engine, "META", "SHORT")  # order is BUY

    ctx, _ = patch_candles(candles(lows=[593.0]))
    with ctx, stub_filler(success=True) as filler:
        result = run(engine)

    assert result.orders_canceled == 1
    assert result.orders_filled == 0
    filler.assert_not_called()

    reloaded = registry.get_order(order.order_id)
    assert reloaded.state is OrderState.CANCELED
    assert reloaded.terminal_reason == "signal_flipped"


def test_short_order_cancels_on_a_long_flip(engine, registry, enabled):
    order = make_order(
        registry, side="SHORT", limit_price=100.0,
        stop_price=105.0, target_price=90.0,
    )
    write_signal(engine, "META", "LONG")

    ctx, _ = patch_candles(candles(lows=[96.0]))
    with ctx, stub_filler():
        run(engine)

    assert registry.get_order(order.order_id).terminal_reason == "signal_flipped"


def test_agreeing_signal_does_not_cancel(engine, registry, enabled):
    order = make_order(registry)
    write_signal(engine, "META", "LONG")  # agrees with BUY

    ctx, _ = patch_candles(candles(lows=[599.0]))
    with ctx, stub_filler():
        result = run(engine)

    assert result.orders_canceled == 0
    assert registry.get_order(order.order_id).state is OrderState.PENDING


def test_hold_signal_does_not_cancel(engine, registry, enabled):
    """HOLD is the Analyst's default for symbols without an active call.

    Treating it as invalidation would cancel nearly every resting order within
    one analyst cycle and defeat the feature. It is a withdrawal of conviction,
    not opposition.
    """
    order = make_order(registry)
    write_signal(engine, "META", "HOLD")

    ctx, _ = patch_candles(candles(lows=[599.0]))
    with ctx, stub_filler():
        result = run(engine)

    assert result.orders_canceled == 0
    assert registry.get_order(order.order_id).state is OrderState.PENDING


@pytest.mark.parametrize(
    "setup_type", ["error", "malformed_analyst_output", "unclear_direction"]
)
def test_quarantined_signals_do_not_cancel(engine, registry, enabled, setup_type):
    """A data or parsing problem is not a directional opinion."""
    order = make_order(registry)
    write_signal(engine, "META", "SHORT", setup_type=setup_type)

    ctx, _ = patch_candles(candles(lows=[599.0]))
    with ctx, stub_filler():
        result = run(engine)

    assert result.orders_canceled == 0
    assert registry.get_order(order.order_id).state is OrderState.PENDING


def test_data_unavailable_signal_does_not_cancel(engine, registry, enabled):
    order = make_order(registry)
    write_signal(engine, "META", "SHORT", data_unavailable=True)

    ctx, _ = patch_candles(candles(lows=[599.0]))
    with ctx, stub_filler():
        run(engine)

    assert registry.get_order(order.order_id).state is OrderState.PENDING


def test_missing_signal_does_not_cancel(engine, registry, enabled):
    """Fail-open: an absent signal is not evidence against the thesis."""
    order = make_order(registry)

    ctx, _ = patch_candles(candles(lows=[599.0]))
    with ctx, stub_filler():
        result = run(engine)

    assert result.orders_canceled == 0
    assert registry.get_order(order.order_id).state is OrderState.PENDING


def test_unparseable_signal_does_not_cancel(engine, registry, enabled):
    order = make_order(registry)
    session = get_session(engine)
    try:
        session.add(
            AgentMemory(
                agent="analyst", symbol="META", key="signal",
                value="not json at all", timestamp=datetime.utcnow(),
            )
        )
        session.commit()
    finally:
        session.close()

    ctx, _ = patch_candles(candles(lows=[599.0]))
    with ctx, stub_filler():
        run(engine)

    assert registry.get_order(order.order_id).state is OrderState.PENDING


def test_explicit_invalidation_marker_cancels(engine, registry, enabled):
    order = make_order(registry)
    write_signal(engine, "META", "LONG", invalidated=True)

    ctx, _ = patch_candles(candles(lows=[599.0]))
    with ctx, stub_filler():
        run(engine)

    reloaded = registry.get_order(order.order_id)
    assert reloaded.state is OrderState.CANCELED
    assert reloaded.terminal_reason == "signal_invalidated"


def test_open_position_cancels_the_order(engine, registry, enabled):
    order = make_order(registry)
    session = get_session(engine)
    try:
        session.add(
            Position(
                profile="moderate", symbol="META", side="long",
                quantity=10, avg_cost=590.0,
            )
        )
        session.commit()
    finally:
        session.close()

    ctx, _ = patch_candles(candles(lows=[599.0]))
    with ctx, stub_filler():
        run(engine)

    reloaded = registry.get_order(order.order_id)
    assert reloaded.state is OrderState.CANCELED
    assert reloaded.terminal_reason == "position_already_open"


def test_opposite_side_position_does_not_cancel(engine, registry, enabled):
    order = make_order(registry)
    session = get_session(engine)
    try:
        session.add(
            Position(
                profile="moderate", symbol="META", side="short",
                quantity=10, avg_cost=610.0,
            )
        )
        session.commit()
    finally:
        session.close()

    ctx, _ = patch_candles(candles(lows=[599.0]))
    with ctx, stub_filler():
        run(engine)

    assert registry.get_order(order.order_id).state is OrderState.PENDING


def test_position_for_another_profile_does_not_cancel(engine, registry, enabled):
    order = make_order(registry)
    session = get_session(engine)
    try:
        session.add(
            Position(
                profile="aggressive", symbol="META", side="long",
                quantity=10, avg_cost=590.0,
            )
        )
        session.commit()
    finally:
        session.close()

    ctx, _ = patch_candles(candles(lows=[599.0]))
    with ctx, stub_filler():
        run(engine)

    assert registry.get_order(order.order_id).state is OrderState.PENDING


# ---------------------------------------------------------------------------
# Expiry
# ---------------------------------------------------------------------------


def test_elapsed_window_expires_the_order(engine, registry, enabled):
    now = datetime.now(timezone.utc)
    order = make_order(
        registry,
        created_at=now - timedelta(hours=2),
        expires_at=now - timedelta(minutes=1),
    )

    ctx, _ = patch_candles(candles(lows=[599.0]))
    with ctx, stub_filler() as filler:
        result = run(engine)

    assert result.orders_expired == 1
    filler.assert_not_called()
    assert registry.get_order(order.order_id).state is OrderState.EXPIRED


def test_expired_order_is_not_filled_even_with_a_crossing_bar(
    engine, registry, enabled
):
    now = datetime.now(timezone.utc)
    order = make_order(
        registry,
        created_at=now - timedelta(hours=2),
        expires_at=now - timedelta(minutes=1),
    )

    ctx, _ = patch_candles(candles(lows=[593.0]))
    with ctx, stub_filler(success=True) as filler:
        result = run(engine)

    assert result.orders_filled == 0
    filler.assert_not_called()
    assert registry.get_order(order.order_id).state is OrderState.EXPIRED


def test_expiry_emits_an_event_with_near_miss_telemetry(engine, registry, enabled):
    """Requirement 10.5 — review must be able to tell a near-miss from a rout."""
    now = datetime.now(timezone.utc)
    make_order(
        registry,
        created_at=now - timedelta(minutes=30),
        expires_at=now - timedelta(minutes=1),
    )

    # Closest low is 595.00, i.e. 1.13 above the 593.87 limit.
    ctx, _ = patch_candles(candles(lows=[599.0, 595.0, 597.0], minutes_ago_start=20))
    with ctx, stub_filler():
        run(engine)

    events = trade_events(engine, "pending_order_expired")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["reason"] == "window_elapsed"
    assert payload["closest_approach_price"] == pytest.approx(595.0)
    assert payload["closest_approach_distance"] == pytest.approx(1.13)


def test_expiry_still_happens_when_bars_are_unavailable(engine, registry, enabled):
    """Telemetry is best-effort; the terminal state is not."""
    now = datetime.now(timezone.utc)
    order = make_order(
        registry,
        created_at=now - timedelta(hours=2),
        expires_at=now - timedelta(minutes=1),
    )

    ctx, _ = patch_candles({})
    with ctx, stub_filler():
        result = run(engine)

    assert result.orders_expired == 1
    assert registry.get_order(order.order_id).state is OrderState.EXPIRED

    payload = trade_events(engine, "pending_order_expired")[0]["payload"]
    assert payload["closest_approach_price"] is None


# ---------------------------------------------------------------------------
# Provider failures are fail-open
# ---------------------------------------------------------------------------


def test_empty_candles_leave_the_order_pending(engine, registry, enabled):
    order = make_order(registry)
    ctx, _ = patch_candles({})

    with ctx, stub_filler() as filler:
        result = run(engine)

    assert result.orders_filled == 0
    assert result.orders_canceled == 0
    assert result.orders_expired == 0
    filler.assert_not_called()
    assert registry.get_order(order.order_id).state is OrderState.PENDING


def test_raising_provider_leaves_the_order_pending(engine, registry, enabled):
    order = make_order(registry)
    ctx, _ = patch_candles(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("429")))

    with ctx, stub_filler() as filler:
        result = run(engine)

    filler.assert_not_called()
    assert registry.get_order(order.order_id).state is OrderState.PENDING
    assert result.orders_checked == 1


def test_client_construction_failure_is_survivable(engine, registry, enabled):
    order = make_order(registry)

    with patch(
        "utils.finnhub_client.FinnhubClient", side_effect=RuntimeError("no key")
    ):
        result = run(engine)

    assert result.orders_checked == 1
    assert registry.get_order(order.order_id).state is OrderState.PENDING


def test_one_symbol_failure_does_not_block_another(engine, registry, enabled):
    make_order(registry, symbol="META")
    make_order(registry, symbol="AMD", limit_price=100.0,
               stop_price=95.0, target_price=110.0)

    def selective(symbol, **kwargs):
        if symbol == "META":
            raise RuntimeError("provider down for META")
        return candles(lows=[99.0], symbol="AMD")

    ctx, _ = patch_candles(selective)
    with ctx, stub_filler(success=True) as filler:
        result = run(engine)

    # AMD crossed its 100.0 limit and was dispatched; META simply stayed pending.
    assert result.orders_filled == 1
    assert filler.call_count == 1


def test_registry_load_failure_returns_an_empty_result(engine, enabled):
    with patch.object(
        PendingOrderRegistry, "get_pending_orders",
        side_effect=RuntimeError("locked"),
    ):
        result = run(engine)

    assert result.orders_checked == 0


def test_tick_never_raises(engine, registry, enabled):
    make_order(registry)
    with patch.object(
        PendingOrderMonitor, "_fetch_bars", side_effect=RuntimeError("boom")
    ):
        result = run(engine)  # must not raise
    assert isinstance(result, MonitorTickResult)


# ---------------------------------------------------------------------------
# Symbol deduplication and telemetry
# ---------------------------------------------------------------------------


def test_bars_are_fetched_once_per_symbol_not_once_per_order(
    engine, registry, enabled
):
    """Three orders on one symbol must cost exactly one provider call."""
    for setup in ("technical_breakout", "vwap_reclaim", "news_breakout"):
        make_order(registry, symbol="META", setup_type=setup)

    ctx, client = patch_candles(candles(lows=[599.0]))
    with ctx, stub_filler():
        result = run(engine)

    assert client.get_candles.call_count == 1
    assert result.symbols_fetched == 1
    assert result.orders_checked == 3


def test_distinct_symbols_each_get_one_fetch(engine, registry, enabled):
    make_order(registry, symbol="META")
    make_order(registry, symbol="AMD", limit_price=100.0,
               stop_price=95.0, target_price=110.0)

    ctx, client = patch_candles(candles(lows=[999.0]))
    with ctx, stub_filler():
        result = run(engine)

    assert client.get_candles.call_count == 2
    assert result.symbols_fetched == 2


def test_tick_records_duration_and_counts(engine, registry, enabled):
    make_order(registry)
    ctx, _ = patch_candles(candles(lows=[599.0, 598.0]))

    with ctx, stub_filler():
        result = run(engine)

    assert result.tick_duration_ms > 0
    assert result.orders_checked == 1
    assert result.bars_fetched == 2
    assert result.had_activity is False


def test_had_activity_flags_real_outcomes(engine, registry, enabled):
    make_order(registry)
    ctx, _ = patch_candles(candles(lows=[593.0]))

    with ctx, stub_filler(success=True):
        result = run(engine)

    assert result.had_activity is True


def test_monitor_makes_no_llm_calls(engine, registry, enabled):
    """The monitor must be purely deterministic."""
    make_order(registry)
    ctx, _ = patch_candles(candles(lows=[593.0]))

    with patch("utils.llm.call_llm") as llm:
        with ctx, stub_filler(success=True):
            run(engine)

    llm.assert_not_called()


# ---------------------------------------------------------------------------
# Layer-2 sweep and dispatch failures
# ---------------------------------------------------------------------------


def test_stranded_filling_order_is_recovered_by_the_sweep(
    engine, registry, enabled
):
    """Requirement 9.12 — no transient state survives."""
    now = datetime.now(timezone.utc)
    order = make_order(
        registry,
        created_at=now - timedelta(hours=3),
        expires_at=now - timedelta(minutes=5),
    )
    registry.claim_for_fill(order.order_id)

    ctx, _ = patch_candles({})
    with ctx, stub_filler():
        run(engine)

    assert registry.get_order(order.order_id).state is OrderState.EXPIRED


def test_fill_dispatch_failure_releases_the_claim(engine, registry, enabled):
    """A crashed dispatch must not strand the order in FILLING."""
    order = make_order(registry)
    ctx, _ = patch_candles(candles(lows=[593.0]))

    with ctx:
        with patch(
            "utils.pending_order_filler.fill_pending_order",
            side_effect=RuntimeError("filler exploded"),
        ):
            result = run(engine)

    assert result.orders_filled == 0
    assert registry.get_order(order.order_id).state is OrderState.PENDING


def test_sweep_failure_does_not_abort_the_tick(engine, registry, enabled):
    make_order(registry)
    ctx, _ = patch_candles(candles(lows=[593.0]))

    with ctx, stub_filler(success=True):
        with patch.object(
            PendingOrderRegistry, "finalize_orphaned_orders",
            side_effect=RuntimeError("sweep failed"),
        ):
            result = run(engine)

    assert result.orders_filled == 1
