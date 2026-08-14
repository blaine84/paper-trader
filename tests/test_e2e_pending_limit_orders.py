"""End-to-end tests for pending limit orders.

These drive the real path: execute_trade() rejects a runaway entry and creates a
resting order, then a monitor tick detects a crossing and fills it through the
real filler and a real execute_trade() call that creates an actual Trade row.

The gate pipeline is patched to proceed. Gate behavior is not what these tests
are for — it is covered in test_pending_order_filler.py — and a real gate run here
would depend on case-library win rates that a fresh fixture database does not
have.

Requirements: 1.4, 3.7, 3.8, 4.4, 4.7, 5.1, 5.3, 6.6a, 6.6b, 6.6f, 6.9, 6.10,
              7.1, 7.9, 7.10, 9.12, 10.5, 13.2, 13.4, 13.13
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

import models.case  # noqa: F401  — registers `cases` on Base.metadata
from db.schema import (
    AgentMemory,
    Balance,
    Base,
    Position,
    Trade,
    get_session,
    init_pending_order_schema,
    init_trade_plan_schema,
)
from utils.pending_order_registry import (
    TERMINAL_STATES,
    OrderState,
    PendingOrderRegistry,
)

# The corrected META scenario: intended entry BELOW the fresh price by ~2.7%
# (trips the 1% runaway threshold, stays under the 5% ceiling), with the profit
# target well above the fresh price so the target branch stays quiet.
INTENDED_ENTRY = 593.87
FRESH_PRICE = 610.00
STOP = 585.00
TARGET = 650.00

PROFILE = "moderate"


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    init_pending_order_schema(eng)
    init_trade_plan_schema(eng)

    session = get_session(eng)
    try:
        session.add(
            Balance(
                profile=PROFILE, cash=100_000.0,
                portfolio_value=0.0, total_equity=100_000.0,
            )
        )
        session.commit()
    finally:
        session.close()
    return eng


@pytest.fixture
def registry(engine):
    return PendingOrderRegistry(engine)


@pytest.fixture
def active_mode():
    """Enable the feature across every module that reads the flag."""
    with patch("agents.portfolio_manager.PENDING_ORDER_MODE", "enabled"), \
         patch("utils.pending_order_creation.PENDING_ORDER_MODE", "enabled"), \
         patch("utils.pending_order_monitor.PENDING_ORDER_MODE", "enabled"), \
         patch("utils.pending_order_filler.PENDING_ORDER_MODE", "enabled"):
        yield


@pytest.fixture
def gates_pass():
    """Neutralize the two history-dependent gates inside execute_trade().

    Both `_run_gate_pipeline` and `compute_edge_score` read the case library for
    win rates and similarity statistics. A fresh fixture database has no case
    history, so the edge score lands at 0.15 against a 0.4 floor and every trade
    is refused — correctly, but for a reason that has nothing to do with pending
    orders.

    Their real behavior is covered elsewhere: gate outcome mapping in
    test_pending_order_filler.py, and Requirement 13.8 (gates decide identically
    regardless of whether geometry arrived from a pending order or a direct
    candidate) by the fact that the filler calls the unmodified pipeline.
    """
    with patch(
        "agents.portfolio_manager._run_gate_pipeline",
        return_value=(True, [], 1.0, []),
    ), patch(
        "agents.portfolio_manager.compute_edge_score", return_value=0.75
    ):
        yield


def decision(**overrides) -> dict:
    base = {
        "action": "BUY",
        "symbol": "META",
        "price": INTENDED_ENTRY,
        "entry_price": INTENDED_ENTRY,
        "stop": STOP,
        "target": TARGET,
        "quantity": 10,
        "setup_type": "technical_breakout",
        "rationale": "buy the pullback to 593.87",
    }
    base.update(overrides)
    return base


def reject_the_entry(engine, dec=None, fresh=FRESH_PRICE):
    """Drive execute_trade() into the runaway rejection, creating an order."""
    from agents.portfolio_manager import execute_trade

    quote = MagicMock()
    quote.get_quote.return_value = {"price": fresh}

    session = get_session(engine)
    try:
        with patch("agents.portfolio_manager.FinnhubClient", return_value=quote):
            return execute_trade(session, dec or decision(), PROFILE)
    finally:
        session.close()


# 11:00 ET on Friday 2026-08-14 — a trading day, comfortably mid-session.
FROZEN_CREATION = datetime(2026, 8, 14, 15, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def frozen_creation_clock():
    """Pin CREATION time so the tests do not depend on the wall clock.

    resolve_expiry() correctly refuses to open a window after 16:00 ET or on a
    non-trading day, so without pinning this file only passes while US markets
    happen to be open — and fails as a silent `window_too_short` decline rather
    than an obvious error.

    Only creation is frozen. The monitor phase stays on the real clock, because
    it compares bar timestamps against now for the Fill_Bar_Age bound; the
    ``normalize_window()`` helper bridges the two.
    """
    with patch("utils.pending_order_creation.now_utc", return_value=FROZEN_CREATION):
        yield


def normalize_window(engine, order_id: str, minutes: int = 45) -> None:
    """Re-anchor an order's window to the REAL clock.

    Two reasons this is needed:

    1. eligible_bars() requires `ts > created_at` strictly. An order created at
       "now" can never see a real bar, because every bar timestamp is already in
       the past. Production has the opposite shape — the order rests and future
       bars arrive — so created_at is moved back to reproduce that.
    2. Creation ran on the frozen clock, so its expires_at is pinned to a fixed
       date. Moving it forward relative to real now keeps the order in the
       monitor's "live" partition instead of being swept as expired.
    """
    now = datetime.now(timezone.utc)
    with engine.connect() as conn:
        conn.execute(
            text(
                "UPDATE pending_orders SET created_at = :c, expires_at = :e "
                "WHERE order_id = :oid"
            ),
            {
                "c": (now - timedelta(minutes=minutes)).isoformat(),
                "e": (now + timedelta(hours=1)).isoformat(),
                "oid": order_id,
            },
        )
        conn.commit()


# Kept as an alias so the intent reads clearly at each call site.
backdate = normalize_window


def bar_payload(*, lows, symbol="META", start_minutes_ago=2, opens=None):
    """Bars ending near `now`.

    The default keeps every bar inside PENDING_ORDER_MAX_FILL_BAR_AGE_SECONDS
    (180s), which the filler enforces before filling. Tests that want a stale bar
    pass a larger offset explicitly.
    """
    base = datetime.now(timezone.utc) - timedelta(minutes=start_minutes_ago)
    timestamps = [
        int((base + timedelta(minutes=i)).timestamp()) for i in range(len(lows))
    ]
    return {
        "symbol": symbol,
        "resolution": "1",
        "timestamps": timestamps,
        "open": opens or [low + 4.0 for low in lows],
        "high": [low + 5.0 for low in lows],
        "low": list(lows),
        "close": [low + 2.0 for low in lows],
        "volume": [1000] * len(lows),
        "source": "alpaca",
    }


def run_monitor(engine, payload):
    """One monitor tick against a fixed bar payload."""
    from utils.pending_order_monitor import run

    client = MagicMock()
    client.get_candles.return_value = payload
    with patch("utils.finnhub_client.FinnhubClient", return_value=client):
        return run(engine)


def trades(engine):
    session = get_session(engine)
    try:
        return session.query(Trade).all()
    finally:
        session.close()


def events(engine, event_type):
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT payload_json FROM trade_events "
                "WHERE event_type = :et ORDER BY id ASC"
            ),
            {"et": event_type},
        ).fetchall()
    return [json.loads(r[0]) for r in rows if r[0]]


# ===========================================================================
# 11.1 — the corrected motivating scenario, all the way to a filled trade
# ===========================================================================


def test_e2e_runaway_rests_then_fills_at_the_limit(
    engine, registry, active_mode, gates_pass
):
    """The whole feature in one test.

    A valid entry the market ran away from becomes a resting order, and a later
    pullback fills it AT THE LIMIT — not at the chased price, and not at the
    profit target.
    """
    # ── Act 1: the entry is rejected, but the intent survives ──
    success, reason = reject_the_entry(engine)

    assert success is False, "the trade must NOT execute in the rejecting cycle"
    assert "stale entry rejected" in reason

    resting = registry.get_active_orders()
    assert len(resting) == 1
    order = resting[0]
    assert order.state is OrderState.PENDING
    assert order.limit_price == pytest.approx(INTENDED_ENTRY)

    # The regression the source spec would have introduced.
    assert order.limit_price != pytest.approx(order.target_price)
    assert order.target_price == pytest.approx(TARGET)
    assert trades(engine) == []

    # ── Act 2: price pulls back and the order fills ──
    backdate(engine, order.order_id)
    # The live quote at fill time is deliberately far from the limit. Without
    # price_authoritative=True, Tier 2 would repair the fill to 640 or Tier 3
    # would refuse it outright.
    live_at_fill = MagicMock()
    live_at_fill.get_quote.return_value = {"price": 640.00}

    with patch("agents.portfolio_manager.FinnhubClient", return_value=live_at_fill):
        result = run_monitor(engine, bar_payload(lows=[599.0, 593.0]))

    assert result.orders_filled == 1

    # ── Assert: filled at the limit, linked, and terminal ──
    filled = registry.get_order(order.order_id)
    assert filled.state is OrderState.FILLED
    assert filled.fill_price == pytest.approx(INTENDED_ENTRY)
    assert filled.fill_policy == "limit_price"
    assert filled.trade_id is not None

    created = trades(engine)
    assert len(created) == 1
    trade = created[0]
    assert trade.symbol == "META"
    assert trade.direction == "LONG"
    assert trade.entry_price == pytest.approx(INTENDED_ENTRY), (
        "the fill must be at the limit, not the chased live price"
    )
    assert trade.entry_price != pytest.approx(640.00)
    assert trade.id == filled.trade_id

    # ── Assert: the audit trail is complete and joinable ──
    fill_events = events(engine, "pending_order_filled")
    assert len(fill_events) == 1
    assert fill_events[0]["order_id"] == order.order_id
    assert fill_events[0]["fill_price"] == pytest.approx(INTENDED_ENTRY)
    assert fill_events[0]["trade_id"] == trade.id

    assert len(events(engine, "pending_order_created")) == 1


def test_e2e_short_runaway_fills_at_the_limit(engine, registry, active_mode, gates_pass):
    dec = decision(
        action="SHORT", symbol="AMD",
        price=100.0, entry_price=100.0, stop=110.0, target=80.0,
    )
    reject_the_entry(engine, dec, fresh=97.0)  # 3% runaway for a short

    order = registry.get_active_orders()[0]
    assert order.side == "SHORT"
    assert order.limit_price == pytest.approx(100.0)
    backdate(engine, order.order_id)

    # highs = low + 5, so a low of 96 reaches 101 and crosses the SHORT limit.
    result = run_monitor(engine, bar_payload(lows=[93.0, 96.0], symbol="AMD"))
    assert result.orders_filled == 1

    filled = registry.get_order(order.order_id)
    assert filled.state is OrderState.FILLED
    assert filled.fill_price == pytest.approx(100.0)

    created = trades(engine)
    assert len(created) == 1
    assert created[0].direction == "SHORT"
    assert created[0].entry_price == pytest.approx(100.0)


# ===========================================================================
# 11.2 — expiry with no crossing
# ===========================================================================


def test_e2e_expiry_records_a_near_miss(engine, registry, active_mode, gates_pass):
    reject_the_entry(engine)
    order = registry.get_active_orders()[0]

    # Backdate creation and close the window, so the bars below land inside it.
    backdate(engine, order.order_id, minutes=45)
    with engine.connect() as conn:
        conn.execute(
            text("UPDATE pending_orders SET expires_at = :ts WHERE order_id = :oid"),
            {
                "ts": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
                "oid": order.order_id,
            },
        )
        conn.commit()

    # Closest low is 595.00 — 1.13 above the limit, never reached. Bars sit at
    # now-10/-9/-8 minutes so they fall inside the (now-45min, now-1min] window.
    result = run_monitor(
        engine, bar_payload(lows=[600.0, 595.0, 598.0], start_minutes_ago=10)
    )

    assert result.orders_expired == 1
    assert result.orders_filled == 0
    assert trades(engine) == []

    expired = registry.get_order(order.order_id)
    assert expired.state is OrderState.EXPIRED
    assert expired.terminal_reason == "window_elapsed"

    payload = events(engine, "pending_order_expired")[0]
    assert payload["closest_approach_price"] == pytest.approx(595.0)
    assert payload["closest_approach_distance"] == pytest.approx(1.13)


# ===========================================================================
# 11.3 — cancellation before fill
# ===========================================================================


def test_e2e_signal_flip_cancels_even_with_a_crossing_bar(
    engine, registry, active_mode, gates_pass
):
    """Cancellation runs before fill detection, so a dead thesis never fills."""
    reject_the_entry(engine)
    order = registry.get_active_orders()[0]
    normalize_window(engine, order.order_id)

    session = get_session(engine)
    try:
        session.add(
            AgentMemory(
                agent="analyst", symbol="META", key="signal",
                value=json.dumps({
                    "symbol": "META", "signal": "SHORT",
                    "setup_type": "momentum_fade",
                }),
                timestamp=datetime.utcnow(),
            )
        )
        session.commit()
    finally:
        session.close()

    result = run_monitor(engine, bar_payload(lows=[593.0]))

    assert result.orders_canceled == 1
    assert result.orders_filled == 0
    assert trades(engine) == []

    canceled = registry.get_order(order.order_id)
    assert canceled.state is OrderState.CANCELED
    assert canceled.terminal_reason == "signal_flipped"

    payload = events(engine, "pending_order_canceled")[0]
    assert payload["reason"] == "signal_flipped"


# ===========================================================================
# 11.4 — no stale favorable fill is reachable
# ===========================================================================


def test_e2e_bar_predating_creation_never_fills(
    engine, registry, active_mode, gates_pass
):
    """The anti-stale-fill guarantee, through the full monitor path."""
    reject_the_entry(engine)
    order = registry.get_active_orders()[0]

    # Backdate creation by 20 minutes, then offer a deeply crossing bar from
    # 60 minutes ago — comfortably inside the active window's end but BEFORE the
    # order existed. Without the strict lower bound this would fill, because the
    # bar's low of 560 is far through the 593.87 limit.
    backdate(engine, order.order_id, minutes=20)
    old = datetime.now(timezone.utc) - timedelta(minutes=60)
    payload = {
        "symbol": "META", "resolution": "1",
        "timestamps": [int(old.timestamp())],
        "open": [596.0], "high": [597.0], "low": [560.0], "close": [565.0],
    }

    result = run_monitor(engine, payload)

    assert result.orders_filled == 0
    assert trades(engine) == []
    assert registry.get_order(order.order_id).state is OrderState.PENDING


def test_e2e_gap_through_cancels_and_creates_no_trade(
    engine, registry, active_mode, gates_pass
):
    reject_the_entry(engine)
    order = registry.get_active_orders()[0]
    backdate(engine, order.order_id)

    # Opens at 570, more than 1.5% below the 593.87 limit.
    payload = bar_payload(lows=[565.0], opens=[570.0], start_minutes_ago=5)

    result = run_monitor(engine, payload)

    assert result.orders_filled == 0
    assert trades(engine) == []
    canceled = registry.get_order(order.order_id)
    assert canceled.state is OrderState.CANCELED
    assert canceled.terminal_reason == "gap_through"


def test_e2e_stale_crossing_bar_does_not_fill(
    engine, registry, active_mode, gates_pass
):
    """The Fill_Bar_Age bound is what replaces the live-quote sanity net."""
    reject_the_entry(engine)
    order = registry.get_active_orders()[0]
    backdate(engine, order.order_id, minutes=60)

    # A crossing bar 30 minutes old — in-window, but well past the 180s bound.
    payload = bar_payload(lows=[593.0], start_minutes_ago=30)

    result = run_monitor(engine, payload)

    assert result.orders_filled == 0
    assert trades(engine) == []
    # Released back to PENDING rather than terminated: the order is still valid.
    assert registry.get_order(order.order_id).state is OrderState.PENDING


def test_e2e_no_fill_is_ever_better_than_the_limit(
    engine, registry, active_mode, gates_pass
):
    """Across a bar that traded far through the limit, the fill is still the limit."""
    reject_the_entry(engine)
    order = registry.get_active_orders()[0]
    backdate(engine, order.order_id)

    # Dips to 588 — through the limit, but not far enough to be a gap-through
    # (open stays inside the 1.5% band).
    payload = bar_payload(lows=[588.0], opens=[592.0], start_minutes_ago=2)
    run_monitor(engine, payload)

    filled = registry.get_order(order.order_id)
    assert filled.state is OrderState.FILLED
    assert filled.fill_price == pytest.approx(INTENDED_ENTRY)
    assert filled.fill_price > 588.0, "a better-than-limit fill would be a gift"

    assert trades(engine)[0].entry_price == pytest.approx(INTENDED_ENTRY)


# ===========================================================================
# 11.5 — observe mode is non-behavioral
# ===========================================================================


def test_e2e_observe_mode_creates_no_trade_but_records_would_fill(
    engine, registry, gates_pass
):
    with patch("agents.portfolio_manager.PENDING_ORDER_MODE", "observe"), \
         patch("utils.pending_order_creation.PENDING_ORDER_MODE", "observe"), \
         patch("utils.pending_order_monitor.PENDING_ORDER_MODE", "observe"), \
         patch("utils.pending_order_filler.PENDING_ORDER_MODE", "observe"):

        reject_the_entry(engine)
        order = registry.get_active_orders()[0]
        backdate(engine, order.order_id)

        result = run_monitor(engine, bar_payload(lows=[599.0, 593.0]))

    # Nothing was executed.
    assert result.orders_filled == 0
    assert trades(engine) == []

    session = get_session(engine)
    try:
        assert session.query(Position).count() == 0
        assert session.query(Balance).count() == 1  # only the fixture row
    finally:
        session.close()

    # But the would-be fill was recorded with its real revalidated outcome.
    would = events(engine, "pending_order_would_fill")
    assert len(would) == 1
    assert would[0]["fill_price"] == pytest.approx(INTENDED_ENTRY)
    assert would[0]["mode"] == "observe"
    assert would[0]["would_be_quantity"] > 0

    # And the order keeps resting until it expires naturally.
    assert registry.get_order(order.order_id).state is OrderState.PENDING


def test_e2e_disabled_mode_is_completely_inert(engine, registry, gates_pass):
    with patch("agents.portfolio_manager.PENDING_ORDER_MODE", "disabled"), \
         patch("utils.pending_order_creation.PENDING_ORDER_MODE", "disabled"), \
         patch("utils.pending_order_monitor.PENDING_ORDER_MODE", "disabled"):

        success, reason = reject_the_entry(engine)
        result = run_monitor(engine, bar_payload(lows=[593.0]))

    assert success is False
    assert "stale entry rejected" in reason
    assert registry.get_active_orders() == []
    assert trades(engine) == []
    assert result.orders_checked == 0
    assert events(engine, "pending_order_created") == []


# ===========================================================================
# 11.6 — transient states never survive a restart
# ===========================================================================


def test_e2e_orphan_sweep_terminalizes_every_transient_state(
    engine, registry, active_mode, gates_pass
):
    """Requirement 9.12 — nothing leaks PENDING or FILLING across a restart."""
    reject_the_entry(engine)
    stranded = registry.get_active_orders()[0]

    # Strand it in FILLING with an elapsed window and an expired lease.
    registry.claim_for_fill(stranded.order_id)
    past = datetime.now(timezone.utc) - timedelta(hours=3)
    with engine.connect() as conn:
        conn.execute(
            text(
                "UPDATE pending_orders SET created_at = :c, expires_at = :e "
                "WHERE order_id = :oid"
            ),
            {
                "c": past.isoformat(),
                "e": (past + timedelta(minutes=30)).isoformat(),
                "oid": stranded.order_id,
            },
        )
        conn.commit()

    resolved = registry.finalize_orphaned_orders(filling_lease_minutes=5)

    assert resolved[stranded.order_id] is OrderState.EXPIRED
    assert registry.get_order(stranded.order_id).state in TERMINAL_STATES
    assert registry.get_active_orders() == []


def test_e2e_crash_after_execution_is_reconciled_to_filled(
    engine, registry, active_mode, gates_pass
):
    """The worst crash window: the trade exists but the state write was lost.

    Fabricating a fill would be wrong; failing to record a real one is worse,
    because the position exists either way.
    """
    reject_the_entry(engine)
    order = registry.get_active_orders()[0]
    registry.claim_for_fill(order.order_id)

    past = datetime.now(timezone.utc) - timedelta(hours=3)
    with engine.connect() as conn:
        conn.execute(
            text("UPDATE pending_orders SET created_at = :c WHERE order_id = :oid"),
            {"c": past.isoformat(), "oid": order.order_id},
        )
        conn.commit()

    with patch.object(
        PendingOrderRegistry, "_find_trade_for_order", return_value=4242
    ):
        resolved = registry.finalize_orphaned_orders(filling_lease_minutes=5)

    assert resolved[order.order_id] is OrderState.FILLED
    reconciled = registry.get_order(order.order_id)
    assert reconciled.state is OrderState.FILLED
    assert reconciled.trade_id == 4242


# ===========================================================================
# Supersession and idempotence across cycles
# ===========================================================================


def test_e2e_repeated_cycles_keep_one_resting_order(
    engine, registry, active_mode, gates_pass
):
    for _ in range(3):
        reject_the_entry(engine)

    active = registry.get_active_orders()
    assert len(active) == 1

    with engine.connect() as conn:
        superseded = conn.execute(
            text(
                "SELECT COUNT(*) FROM pending_orders "
                "WHERE state = 'canceled' AND terminal_reason = 'superseded'"
            )
        ).scalar()
    assert superseded == 2


def test_e2e_two_identical_ticks_fill_once(engine, registry, active_mode, gates_pass):
    reject_the_entry(engine)
    backdate(engine, registry.get_active_orders()[0].order_id)
    payload = bar_payload(lows=[599.0, 593.0])

    first = run_monitor(engine, payload)
    second = run_monitor(engine, payload)

    assert first.orders_filled == 1
    assert second.orders_filled == 0
    assert len(trades(engine)) == 1


def test_e2e_target_exceeded_never_rests_an_order(
    engine, registry, active_mode, gates_pass
):
    """The branch the source spec proposed resting. It stays a rejection."""
    dec = decision(target=600.00)
    success, reason = reject_the_entry(engine, dec, fresh=601.24)

    assert success is False
    assert "already crossed BUY target" in reason
    assert registry.get_active_orders() == []
    assert trades(engine) == []

    declines = events(engine, "pending_order_declined")
    assert len(declines) == 1
    assert declines[0]["reason"] == "target_already_exceeded"
