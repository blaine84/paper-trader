"""Integration tests for the pending-limit-order hooks inside execute_trade().

These assert the property the whole design rests on: order creation is
**additive** to the stale-entry rejection. `execute_trade()` returns the same
`(False, reason)` in every mode, and nothing about the trading outcome depends on
the pending-order machinery succeeding.

Also pins the coverage boundary. Because the live-quote deviation tiers run
BEFORE `_fresh_price_stale_entry_check()`, the runaway branch only ever observes
roughly the 1%-5% range:

    <=1%          no rejection at all
    1% to ~5%     runaway  -> pending order created
    ~5% to 10%    Tier 2 repairs and executes -> declined(repaired_before_check)
    >10%          Tier 3 rejects outright     -> no order, no decline

Requirements: 0.4, 0.5, 1.11, 1.12, 1.13, 1.16, 1.17, 13.1, 14.1, 14.8
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from db.schema import (
    Balance,
    Base,
    get_session,
    init_pending_order_schema,
    init_trade_plan_schema,
)
from utils.pending_order_registry import OrderState, PendingOrderRegistry

INTENDED_ENTRY = 593.87
STOP = 585.00
TARGET = 650.00


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
                profile="moderate", cash=100_000.0,
                portfolio_value=0.0, total_equity=100_000.0,
            )
        )
        session.commit()
    finally:
        session.close()
    return eng


# 11:00 ET on Friday 2026-08-14 — a trading day, comfortably mid-session.
FROZEN_NOW = datetime(2026, 8, 14, 15, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def frozen_clock():
    """Pin creation time so these tests do not depend on the wall clock.

    resolve_expiry() correctly returns None after 16:00 ET and on non-trading
    days, which makes creation decline with `window_too_short`. Without pinning,
    this file only passes while US markets happen to be open, and it fails as a
    silent decline rather than an obvious error.
    """
    with patch("utils.pending_order_creation.now_utc", return_value=FROZEN_NOW):
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
        "rationale": "waiting for the pullback",
    }
    base.update(overrides)
    return base


def run_execute_trade(engine, live_price, dec=None):
    """Drive execute_trade() with a mocked live quote."""
    from agents.portfolio_manager import execute_trade

    dec = dec if dec is not None else decision()
    quote = MagicMock()
    quote.get_quote.return_value = {"price": live_price}

    session = get_session(engine)
    try:
        with patch("agents.portfolio_manager.FinnhubClient", return_value=quote):
            return execute_trade(session, dec, "moderate"), dec
    finally:
        session.close()


def orders(engine):
    return PendingOrderRegistry(engine).get_active_orders()


def declines(engine, reason=None):
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT payload_json FROM trade_events "
                "WHERE event_type = 'pending_order_declined' ORDER BY id ASC"
            )
        ).fetchall()
    payloads = [json.loads(r[0]) for r in rows if r[0]]
    if reason:
        payloads = [p for p in payloads if p.get("reason") == reason]
    return payloads


def created_events(engine):
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT payload_json FROM trade_events "
                "WHERE event_type = 'pending_order_created' ORDER BY id ASC"
            )
        ).fetchall()
    return [json.loads(r[0]) for r in rows if r[0]]


# ---------------------------------------------------------------------------
# Disabled mode: byte-for-byte unchanged behavior
# ---------------------------------------------------------------------------


def test_disabled_mode_creates_nothing_and_returns_the_same_rejection(engine):
    live = INTENDED_ENTRY * 1.03  # 3% runaway

    with patch("agents.portfolio_manager.PENDING_ORDER_MODE", "disabled"):
        (success, reason), _ = run_execute_trade(engine, live)

    assert success is False
    assert "stale entry rejected" in reason
    assert orders(engine) == []
    assert declines(engine) == []
    assert created_events(engine) == []


def test_disabled_mode_emits_no_repair_band_decline(engine):
    live = INTENDED_ENTRY * 1.07  # Tier 2 territory

    with patch("agents.portfolio_manager.PENDING_ORDER_MODE", "disabled"):
        run_execute_trade(engine, live)

    assert declines(engine) == []


# ---------------------------------------------------------------------------
# Observe / enabled mode: rejection is unchanged, order is additive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["observe", "enabled"])
def test_rejection_message_is_identical_with_the_feature_active(engine, mode):
    """The safety decision must be untouched in every mode."""
    live = INTENDED_ENTRY * 1.03

    with patch("agents.portfolio_manager.PENDING_ORDER_MODE", "disabled"):
        (baseline_success, baseline_reason), _ = run_execute_trade(engine, live)

    with patch("agents.portfolio_manager.PENDING_ORDER_MODE", mode):
        with patch("utils.pending_order_creation.PENDING_ORDER_MODE", mode):
            (success, reason), _ = run_execute_trade(engine, live)

    assert success is baseline_success is False
    assert reason == baseline_reason


def test_runaway_creates_an_order_and_still_rejects(engine):
    live = INTENDED_ENTRY * 1.03

    with patch("agents.portfolio_manager.PENDING_ORDER_MODE", "observe"):
        with patch("utils.pending_order_creation.PENDING_ORDER_MODE", "observe"):
            (success, reason), _ = run_execute_trade(engine, live)

    # The trade did NOT execute this cycle.
    assert success is False
    assert "stale entry rejected" in reason

    # But the intent survives.
    active = orders(engine)
    assert len(active) == 1
    assert active[0].state is OrderState.PENDING
    assert active[0].limit_price == pytest.approx(INTENDED_ENTRY)
    assert len(created_events(engine)) == 1


def test_created_order_uses_the_pre_repair_entry(engine):
    """A 3% deviation does not trigger repair, but assert the wiring anyway."""
    live = INTENDED_ENTRY * 1.03

    with patch("agents.portfolio_manager.PENDING_ORDER_MODE", "observe"):
        with patch("utils.pending_order_creation.PENDING_ORDER_MODE", "observe"):
            _, dec = run_execute_trade(engine, live)

    active = orders(engine)
    assert active[0].limit_price == pytest.approx(INTENDED_ENTRY)
    assert active[0].limit_price != pytest.approx(live)
    # The decision's price was untouched at this tier, so they agree here.
    assert dec["price"] == pytest.approx(INTENDED_ENTRY)


def test_creation_failure_does_not_change_the_return_value(engine):
    """Layer-1 fail-open, verified through the real call path."""
    live = INTENDED_ENTRY * 1.03

    with patch("agents.portfolio_manager.PENDING_ORDER_MODE", "observe"):
        with patch(
            "utils.pending_order_creation.maybe_create_pending_order",
            side_effect=RuntimeError("catastrophe"),
        ):
            (success, reason), _ = run_execute_trade(engine, live)

    assert success is False
    assert "stale entry rejected" in reason
    assert orders(engine) == []


# ---------------------------------------------------------------------------
# Target-exceeded branch
# ---------------------------------------------------------------------------


def test_target_exceeded_declines_and_creates_no_order(engine):
    """The market traded through the whole reward leg; the idea is spent."""
    dec = decision(target=600.00)
    live = 601.24  # above the 600.00 target

    with patch("agents.portfolio_manager.PENDING_ORDER_MODE", "observe"):
        with patch("utils.pending_order_creation.PENDING_ORDER_MODE", "observe"):
            (success, reason), _ = run_execute_trade(engine, live, dec)

    assert success is False
    assert "already crossed BUY target" in reason
    assert orders(engine) == []
    assert len(declines(engine, "target_already_exceeded")) == 1


# ---------------------------------------------------------------------------
# The coverage boundary
# ---------------------------------------------------------------------------


def test_three_percent_runaway_creates_an_order(engine):
    live = INTENDED_ENTRY * 1.03

    with patch("agents.portfolio_manager.PENDING_ORDER_MODE", "observe"):
        with patch("utils.pending_order_creation.PENDING_ORDER_MODE", "observe"):
            run_execute_trade(engine, live)

    assert len(orders(engine)) == 1


def test_seven_percent_deviation_is_repaired_and_declined_not_rested(engine):
    """Tier 2 territory: the repair pre-empts the stale-entry check entirely.

    This is the Repair_Band blind spot. The repaired trade proceeds (it does not
    reach the stale-entry rejection at all), and no order is rested — resting one
    alongside a repaired fill would double-book the same intent.
    """
    live = INTENDED_ENTRY * 1.07

    with patch("agents.portfolio_manager.PENDING_ORDER_MODE", "observe"):
        with patch("utils.pending_order_creation.PENDING_ORDER_MODE", "observe"):
            (success, reason), dec = run_execute_trade(engine, live)

    # No pending order, but the band is now measurable.
    assert orders(engine) == []
    band = declines(engine, "repaired_before_check")
    assert len(band) == 1
    assert band[0]["original_intended_entry"] == pytest.approx(INTENDED_ENTRY)
    assert band[0]["live_price"] == pytest.approx(live)
    assert band[0]["deviation_pct"] == pytest.approx(0.0654, abs=1e-3)

    # And the repair genuinely re-anchored the decision to the live price.
    assert dec["price"] == pytest.approx(live)
    assert dec["price"] != pytest.approx(INTENDED_ENTRY)


def test_fifteen_percent_deviation_rejects_without_order_or_decline(engine):
    """Tier 3 returns before the stale-entry check and before any hook."""
    live = INTENDED_ENTRY * 1.15

    with patch("agents.portfolio_manager.PENDING_ORDER_MODE", "observe"):
        with patch("utils.pending_order_creation.PENDING_ORDER_MODE", "observe"):
            (success, reason), _ = run_execute_trade(engine, live)

    assert success is False
    assert "deviation" in reason.lower()
    assert orders(engine) == []
    assert declines(engine) == []


def test_tier_two_repair_behavior_is_unchanged_across_all_modes(engine):
    """The feature must not alter the repair outcome in any mode."""
    live = INTENDED_ENTRY * 1.07
    results = {}

    for mode in ("disabled", "observe", "enabled"):
        with patch("agents.portfolio_manager.PENDING_ORDER_MODE", mode):
            with patch("utils.pending_order_creation.PENDING_ORDER_MODE", mode):
                (success, reason), dec = run_execute_trade(engine, live)
        results[mode] = (success, dec["price"], dec["stop"], dec["target"])

    assert results["disabled"] == results["observe"] == results["enabled"]


# ---------------------------------------------------------------------------
# Action guards
# ---------------------------------------------------------------------------


def test_close_action_creates_no_order(engine):
    dec = decision(action="CLOSE")

    with patch("agents.portfolio_manager.PENDING_ORDER_MODE", "observe"):
        with patch("utils.pending_order_creation.PENDING_ORDER_MODE", "observe"):
            run_execute_trade(engine, INTENDED_ENTRY * 1.03, dec)

    assert orders(engine) == []


def test_short_runaway_creates_a_short_order(engine):
    dec = decision(
        action="SHORT", symbol="AMD", price=100.0, entry_price=100.0,
        stop=105.0, target=90.0,
    )
    live = 97.0  # 3% runaway for a short

    with patch("agents.portfolio_manager.PENDING_ORDER_MODE", "observe"):
        with patch("utils.pending_order_creation.PENDING_ORDER_MODE", "observe"):
            (success, reason), _ = run_execute_trade(engine, live, dec)

    assert success is False
    active = orders(engine)
    assert len(active) == 1
    assert active[0].side == "SHORT"
    assert active[0].limit_price == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Legacy path compatibility (Requirement 14.1)
# ---------------------------------------------------------------------------


def test_order_is_created_with_no_candidate_on_the_legacy_path(engine):
    """PM_CANDIDATE_MODE is disabled in production, so there is no candidate."""
    live = INTENDED_ENTRY * 1.03

    with patch("agents.portfolio_manager.PENDING_ORDER_MODE", "observe"):
        with patch("utils.pending_order_creation.PENDING_ORDER_MODE", "observe"):
            run_execute_trade(engine, live)

    active = orders(engine)
    assert len(active) == 1
    assert active[0].candidate_id is None
    assert active[0].cycle_id is None


def test_repeated_rejections_keep_only_one_active_order(engine):
    """Supersession keeps the resting set clean across PM cycles."""
    live = INTENDED_ENTRY * 1.03

    with patch("agents.portfolio_manager.PENDING_ORDER_MODE", "observe"):
        with patch("utils.pending_order_creation.PENDING_ORDER_MODE", "observe"):
            for _ in range(3):
                run_execute_trade(engine, live)

    assert len(orders(engine)) == 1


# ---------------------------------------------------------------------------
# Orchestrator job registration (Requirement 14.8)
# ---------------------------------------------------------------------------


def test_orchestrator_wiring_is_covered_by_execution():
    """Orchestrator wiring is verified by running main(), not by inspecting source.

    That became possible once utils/resource_telemetry.py stopped hard-failing its
    `import resource` on non-POSIX platforms. The real coverage — job
    registration, the flag guard, max_instances/coalesce, the market-hours guard,
    the startup sweep, and check_schema creating the tables — lives in
    tests/test_orchestrator_pending_orders.py.

    This test just asserts orchestrator imports, since everything above depends
    on it and a regression there would be confusing to diagnose from a dozen
    downstream failures.
    """
    import orchestrator

    assert callable(orchestrator.check_schema)
    assert callable(orchestrator._ensure_pending_order_events_identity_default)
