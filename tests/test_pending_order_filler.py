"""Tests for utils/pending_order_filler.py and execute_trade's price_authoritative.

Two groups:

A. ``execute_trade(price_authoritative=True)`` — proves the deviation tiers are
   genuinely skipped. Without this, a limit deliberately away from the live price
   would be repaired to the chased price (5-10%) or refused outright (>10%).
B. Filler outcome mapping — risk failures cancel, gate and execution failures
   reject, observe mode creates nothing.

Requirements: 6.1-6.12, 7.6, 8.1, 8.3, 8.7, 13.5, 13.12, 13.13
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from db.schema import (
    Balance,
    Base,
    Position,
    Trade,
    get_session,
    init_pending_order_schema,
)
from utils.pending_order_fill import Bar
from utils.pending_order_filler import FillResult, fill_pending_order
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
    session = get_session(eng)
    try:
        session.add(
            Balance(
                profile="moderate",
                cash=100_000.0,
                portfolio_value=0.0,
                total_equity=100_000.0,
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
def enabled():
    with patch("utils.pending_order_filler.PENDING_ORDER_MODE", "enabled"):
        yield


def claimed_order(registry, **overrides) -> PendingOrder:
    """A PENDING order already claimed into FILLING, as the monitor leaves it."""
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
        intended_quantity=10,
        created_at=now - timedelta(minutes=20),
        expires_at=now + timedelta(hours=1),
    )
    defaults.update(overrides)
    order = PendingOrder(**defaults)
    registry.create_order(order)
    registry.claim_for_fill(order.order_id)
    return registry.get_order(order.order_id)


def fresh_bar(*, minutes_ago=1, low=593.0, open_=596.0) -> Bar:
    ts = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return Bar(
        ts=ts,
        open=Decimal(str(open_)),
        high=Decimal(str(open_ + 1)),
        low=Decimal(str(low)),
        close=Decimal(str(low + 1)),
    )


def patch_gates(proceed=True, notes=None, multiplier=1.0):
    return patch(
        "agents.portfolio_manager._run_gate_pipeline",
        return_value=(proceed, notes or [], multiplier, []),
    )


def patch_sizer(quantity=10, rejection_reason=None):
    result = MagicMock()
    result.quantity = quantity
    result.rejection_reason = rejection_reason
    return patch(
        "utils.position_sizer.calculate_position_size", return_value=result
    )


def patch_execute(success=True, message="ok"):
    return patch(
        "agents.portfolio_manager.execute_trade",
        return_value=(success, message),
    )


def trade_events(engine, event_type=None):
    clause = " WHERE event_type = :et" if event_type else ""
    params = {"et": event_type} if event_type else {}
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"SELECT event_type, trade_id, price, payload_json "
                f"FROM trade_events{clause} ORDER BY id ASC"
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


def count_rows(engine, table):
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()


# ===========================================================================
# A. execute_trade(price_authoritative=True)
# ===========================================================================


def test_price_authoritative_makes_no_quote_call(engine):
    """The whole point: no live quote is fetched, so no tier can run."""
    from agents.portfolio_manager import execute_trade

    session = get_session(engine)
    decision = {
        "action": "BUY", "symbol": "META", "price": LIMIT, "entry_price": LIMIT,
        "stop": STOP, "target": TARGET, "quantity": 10,
        "setup_type": "technical_breakout",
    }
    try:
        with patch("agents.portfolio_manager.FinnhubClient") as client:
            try:
                execute_trade(
                    session, decision, "moderate",
                    normalized=True, price_authoritative=True,
                )
            except Exception:
                # The outcome is irrelevant here — this fixture's schema does not
                # carry every table the gate pipeline touches. What matters is
                # whether a quote was requested before reaching that point.
                pass
        client.assert_not_called()
    finally:
        session.close()


@pytest.mark.parametrize(
    "live_price,label",
    [
        (LIMIT * 1.07, "7% above — Tier 2 would have repaired"),
        (LIMIT * 1.15, "15% above — Tier 3 would have rejected"),
        (LIMIT * 0.80, "20% below — Tier 3 would have rejected"),
    ],
)
def test_price_authoritative_never_rewrites_the_entry(engine, live_price, label):
    """The decision's entry price must survive untouched.

    Asserted on the decision dict rather than on the trade outcome, because that
    is precisely what Tier 2 mutates: `price`, `entry_price`, and the rescaled
    stop/target.
    """
    from agents.portfolio_manager import execute_trade

    session = get_session(engine)
    decision = {
        "action": "BUY", "symbol": "META", "price": LIMIT, "entry_price": LIMIT,
        "stop": STOP, "target": TARGET, "quantity": 10,
        "setup_type": "technical_breakout",
    }
    try:
        quote = MagicMock()
        quote.get_quote.return_value = {"price": live_price}
        with patch(
            "agents.portfolio_manager.FinnhubClient", return_value=quote
        ):
            execute_trade(
                session, decision, "moderate",
                normalized=True, price_authoritative=True,
            )
    finally:
        session.close()

    assert decision["price"] == pytest.approx(LIMIT), label
    assert decision["entry_price"] == pytest.approx(LIMIT), label
    assert decision["stop"] == pytest.approx(STOP), label
    assert decision["target"] == pytest.approx(TARGET), label


def test_default_still_applies_the_deviation_tiers(engine):
    """Backward compatibility: every existing caller keeps the old behavior."""
    from agents.portfolio_manager import execute_trade

    session = get_session(engine)
    decision = {
        "action": "BUY", "symbol": "META", "price": LIMIT, "entry_price": LIMIT,
        "stop": STOP, "target": TARGET, "quantity": 10,
        "setup_type": "technical_breakout",
    }
    try:
        quote = MagicMock()
        # 15% above the decision price trips Tier 3 (extreme deviation).
        quote.get_quote.return_value = {"price": LIMIT * 1.15}
        with patch(
            "agents.portfolio_manager.FinnhubClient", return_value=quote
        ):
            success, message = execute_trade(
                session, decision, "moderate", normalized=True
            )
    finally:
        session.close()

    assert success is False
    assert "deviation" in message.lower()
    quote.get_quote.assert_called_once()


def test_price_authoritative_still_rejects_a_zero_price(engine):
    """Skipping the tier block must not let a bad price through."""
    from agents.portfolio_manager import execute_trade

    session = get_session(engine)
    try:
        success, message = execute_trade(
            session,
            {
                "action": "BUY", "symbol": "META", "price": 0,
                "entry_price": 0, "stop": STOP, "target": TARGET,
                "quantity": 10,
            },
            "moderate",
            normalized=True,
            price_authoritative=True,
        )
    finally:
        session.close()

    assert success is False
    assert "price" in message.lower()


def test_stale_entry_check_is_inert_on_the_authoritative_path(engine):
    """No live price is fetched, so _fresh_price_stale_entry_check no-ops.

    Its own guard returns (True, None) when fresh_price is not positive, so the
    function needed no modification.
    """
    from agents.portfolio_manager import _fresh_price_stale_entry_check

    ok, reason = _fresh_price_stale_entry_check(
        action="BUY", symbol="META", intended_entry=LIMIT,
        fresh_price=0.0,  # what live_price stays at on this path
        target=TARGET,
    )
    assert ok is True
    assert reason is None


def test_no_existing_caller_passes_price_authoritative():
    """Only the filler may use it in v1 (Requirement 13.12)."""
    import subprocess

    result = subprocess.run(
        ["git", "grep", "-n", "price_authoritative"],
        capture_output=True, text=True,
    )
    call_sites = [
        line for line in result.stdout.splitlines()
        if "price_authoritative=True" in line
    ]
    offenders = [
        line for line in call_sites
        if not line.startswith(("utils/pending_order_filler.py", "tests/"))
    ]
    assert offenders == [], f"unexpected call sites: {offenders}"


# ===========================================================================
# B. Filler outcome mapping
# ===========================================================================


def test_successful_fill_records_everything(engine, registry, enabled):
    order = claimed_order(registry)
    bar = fresh_bar()

    with patch_gates(), patch_sizer(quantity=10), patch_execute(success=True):
        result = fill_pending_order(engine, order, bar)

    assert result.success is True
    assert result.fill_price == pytest.approx(LIMIT)
    assert result.reason == "filled"

    reloaded = registry.get_order(order.order_id)
    assert reloaded.state is OrderState.FILLED
    assert reloaded.fill_price == pytest.approx(LIMIT)
    assert reloaded.fill_policy == "limit_price"
    assert reloaded.fill_bar_ts == bar.ts
    assert reloaded.filled_at is not None


def test_fill_price_is_the_limit_not_the_bar_low(engine, registry, enabled):
    order = claimed_order(registry)
    bar = fresh_bar(low=550.0)  # traded far below the limit

    with patch_gates(), patch_sizer(), patch_execute():
        result = fill_pending_order(engine, order, bar)

    assert result.fill_price == pytest.approx(LIMIT)


def test_fill_emits_an_event_with_bar_and_policy_detail(engine, registry, enabled):
    order = claimed_order(registry)
    bar = fresh_bar()

    with patch_gates(), patch_sizer(quantity=7), patch_execute():
        fill_pending_order(engine, order, bar)

    events = trade_events(engine, "pending_order_filled")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["fill_price"] == pytest.approx(LIMIT)
    assert payload["fill_policy"] == "limit_price"
    assert payload["fill_bar_ts"] == bar.ts.isoformat()
    assert payload["fill_bar_low"] == pytest.approx(float(bar.low))
    assert payload["quantity"] == 7
    assert "seconds_from_creation_to_fill" in payload
    assert "fill_bar_age_seconds" in payload


def test_fill_passes_the_limit_to_execute_trade(engine, registry, enabled):
    order = claimed_order(registry)

    with patch_gates(), patch_sizer(), patch_execute() as execute:
        fill_pending_order(engine, order, fresh_bar())

    decision = execute.call_args[0][1]
    assert decision["price"] == pytest.approx(LIMIT)
    assert decision["entry_price"] == pytest.approx(LIMIT)
    assert execute.call_args[1]["normalized"] is True
    assert execute.call_args[1]["price_authoritative"] is True


def test_fill_supplies_stop_under_every_alias(engine, registry, enabled):
    """normalized=True fails closed without a stop, so all aliases are set."""
    order = claimed_order(registry)

    with patch_gates(), patch_sizer(), patch_execute() as execute:
        fill_pending_order(engine, order, fresh_bar())

    decision = execute.call_args[0][1]
    for key in ("stop", "stop_price", "stop_loss"):
        assert decision[key] == pytest.approx(STOP)
    for key in ("target", "target_price", "profit_target"):
        assert decision[key] == pytest.approx(TARGET)


# ---------------------------------------------------------------------------
# Fill_Bar_Age bound
# ---------------------------------------------------------------------------


def test_stale_crossing_bar_releases_instead_of_filling(engine, registry, enabled):
    """The compensating control for skipping the deviation tiers."""
    order = claimed_order(registry)
    stale = fresh_bar(minutes_ago=10)  # 600s > the 180s default

    with patch_gates(), patch_sizer(), patch_execute() as execute:
        result = fill_pending_order(engine, order, stale)

    assert result.success is False
    assert result.reason == "stale_fill_bar"
    execute.assert_not_called()
    assert registry.get_order(order.order_id).state is OrderState.PENDING


def test_bar_just_inside_the_age_bound_fills(engine, registry, enabled):
    order = claimed_order(registry)
    bar = fresh_bar(minutes_ago=2)  # 120s < 180s

    with patch_gates(), patch_sizer(), patch_execute():
        result = fill_pending_order(engine, order, bar)

    assert result.success is True


def test_age_bound_is_configurable(engine, registry, enabled):
    order = claimed_order(registry)
    bar = fresh_bar(minutes_ago=2)

    with patch(
        "utils.pending_order_filler.PENDING_ORDER_MAX_FILL_BAR_AGE_SECONDS", 30
    ):
        with patch_gates(), patch_sizer(), patch_execute():
            result = fill_pending_order(engine, order, bar)

    assert result.reason == "stale_fill_bar"


# ---------------------------------------------------------------------------
# Gate rejection -> REJECTED
# ---------------------------------------------------------------------------


def test_gate_rejection_marks_rejected_with_the_gate_name(engine, registry, enabled):
    order = claimed_order(registry)
    notes = [{"gate": "setup_quality_gate", "decision": "reject", "reason": "low WR"}]

    with patch_gates(proceed=False, notes=notes), patch_execute() as execute:
        result = fill_pending_order(engine, order, fresh_bar())

    assert result.success is False
    assert result.reason == "setup_quality_gate"
    execute.assert_not_called()

    reloaded = registry.get_order(order.order_id)
    assert reloaded.state is OrderState.REJECTED
    assert reloaded.terminal_reason == "setup_quality_gate"
    assert count_rows(engine, "trades") == 0


def test_gate_rejection_without_notes_uses_a_generic_reason(engine, registry, enabled):
    order = claimed_order(registry)

    with patch_gates(proceed=False, notes=[]):
        result = fill_pending_order(engine, order, fresh_bar())

    assert result.reason == "gate_pipeline_rejected"


def test_gate_multiplier_is_applied_to_the_quantity(engine, registry, enabled):
    order = claimed_order(registry)

    with patch_gates(multiplier=0.5), patch_sizer(quantity=4) as sizer:
        with patch_execute():
            fill_pending_order(engine, order, fresh_bar())

    resolved = sizer.call_args[0][0]
    assert resolved.risk_multiplier == pytest.approx(0.5)


def test_multiplier_above_one_is_clamped(engine, registry, enabled):
    """The sizer rejects risk_multiplier > 1.0, so it must be clamped first."""
    order = claimed_order(registry)

    with patch_gates(multiplier=1.5), patch_sizer() as sizer, patch_execute():
        fill_pending_order(engine, order, fresh_bar())

    assert sizer.call_args[0][0].risk_multiplier == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Risk failures -> CANCELED
# ---------------------------------------------------------------------------


def test_zero_quantity_cancels_as_sizing_rejected(engine, registry, enabled):
    """Risk failures cancel; 'rejected' stays reserved for gates and execution."""
    order = claimed_order(registry)

    with patch_gates(), patch_sizer(quantity=0, rejection_reason="risk too small"):
        with patch_execute() as execute:
            result = fill_pending_order(engine, order, fresh_bar())

    assert result.reason == "sizing_rejected"
    execute.assert_not_called()

    reloaded = registry.get_order(order.order_id)
    assert reloaded.state is OrderState.CANCELED
    assert reloaded.terminal_reason == "sizing_rejected"
    assert count_rows(engine, "trades") == 0


def test_sizer_raising_cancels_fail_closed(engine, registry, enabled):
    order = claimed_order(registry)

    with patch_gates():
        with patch(
            "utils.position_sizer.calculate_position_size",
            side_effect=RuntimeError("sizer exploded"),
        ):
            with patch_execute() as execute:
                result = fill_pending_order(engine, order, fresh_bar())

    assert result.reason == "sizing_rejected"
    execute.assert_not_called()
    assert registry.get_order(order.order_id).state is OrderState.CANCELED


def test_insufficient_cash_cancels(engine, registry, enabled):
    order = claimed_order(registry)
    # 1000 shares at 593.87 needs ~$594k against a $100k balance.
    with patch_gates(), patch_sizer(quantity=1000), patch_execute() as execute:
        result = fill_pending_order(engine, order, fresh_bar())

    assert result.reason == "insufficient_buying_power"
    execute.assert_not_called()

    reloaded = registry.get_order(order.order_id)
    assert reloaded.state is OrderState.CANCELED
    assert reloaded.terminal_reason == "insufficient_buying_power"


def test_cooldown_cancels(engine, registry, enabled):
    """AMD is a high-momentum asset with a 30-minute re-entry cooldown."""
    order = claimed_order(
        registry, symbol="AMD", limit_price=100.0,
        stop_price=95.0, target_price=115.0,
    )

    with patch_gates(), patch_sizer(quantity=10):
        with patch(
            "agents.portfolio_manager._get_recent_closed_trades_for_preflight",
            return_value=[{"symbol": "AMD", "profile": "moderate"}],
        ):
            with patch_execute() as execute:
                result = fill_pending_order(engine, order, fresh_bar(low=99.0))

    assert result.reason == "cooldown_active"
    execute.assert_not_called()
    assert registry.get_order(order.order_id).terminal_reason == "cooldown_active"


def test_cooldown_ignores_non_momentum_symbols(engine, registry, enabled):
    """META is not in HIGH_MOMENTUM_ASSETS, so recent trades do not block it."""
    order = claimed_order(registry)

    with patch_gates(), patch_sizer(quantity=10):
        with patch(
            "agents.portfolio_manager._get_recent_closed_trades_for_preflight",
            return_value=[{"symbol": "META"}],
        ):
            with patch_execute(success=True):
                result = fill_pending_order(engine, order, fresh_bar())

    assert result.success is True


def test_correlation_warning_does_not_cancel_by_default(engine, registry, enabled):
    """check_correlation is warning-only live; cancelling here would make
    pending fills stricter than immediate execution."""
    order = claimed_order(registry)

    with patch_gates(), patch_sizer(quantity=10):
        with patch(
            "utils.trade_validator.check_correlation",
            return_value="Correlated exposure: already long QQQ",
        ):
            with patch_execute(success=True):
                result = fill_pending_order(engine, order, fresh_bar())

    assert result.success is True


# ---------------------------------------------------------------------------
# Execution failures -> REJECTED
# ---------------------------------------------------------------------------


def test_execute_trade_returning_false_marks_rejected(engine, registry, enabled):
    order = claimed_order(registry)

    with patch_gates(), patch_sizer(), patch_execute(success=False, message="nope"):
        result = fill_pending_order(engine, order, fresh_bar())

    assert result.reason == "execution_failed"
    reloaded = registry.get_order(order.order_id)
    assert reloaded.state is OrderState.REJECTED
    assert reloaded.terminal_reason == "execution_failed"
    assert count_rows(engine, "trades") == 0


def test_execute_trade_raising_marks_rejected(engine, registry, enabled):
    order = claimed_order(registry)

    with patch_gates(), patch_sizer():
        with patch(
            "agents.portfolio_manager.execute_trade",
            side_effect=RuntimeError("boom"),
        ):
            result = fill_pending_order(engine, order, fresh_bar())

    assert result.reason == "execution_failed"
    assert registry.get_order(order.order_id).state is OrderState.REJECTED
    assert count_rows(engine, "trades") == 0


def test_rejection_event_carries_the_detail(engine, registry, enabled):
    order = claimed_order(registry)

    with patch_gates(), patch_sizer(), patch_execute(success=False, message="why"):
        fill_pending_order(engine, order, fresh_bar())

    events = trade_events(engine, "pending_order_rejected")
    assert len(events) == 1
    assert events[0]["payload"]["reason"] == "execution_failed"
    assert events[0]["payload"]["detail"] == "why"


# ---------------------------------------------------------------------------
# Observe mode
# ---------------------------------------------------------------------------


def test_observe_mode_creates_nothing_and_releases(engine, registry):
    order = claimed_order(registry)
    before_balances = count_rows(engine, "balance")

    with patch("utils.pending_order_filler.PENDING_ORDER_MODE", "observe"):
        with patch_gates(), patch_sizer(quantity=10), patch_execute() as execute:
            result = fill_pending_order(engine, order, fresh_bar())

    assert result.success is False
    assert result.reason == "observe_would_fill"
    execute.assert_not_called()

    assert count_rows(engine, "trades") == 0
    assert count_rows(engine, "positions") == 0
    assert count_rows(engine, "balance") == before_balances

    # Keeps resting so it can be observed again until it expires naturally.
    assert registry.get_order(order.order_id).state is OrderState.PENDING


def test_observe_mode_emits_would_fill_with_the_real_outcome(engine, registry):
    order = claimed_order(registry)

    with patch("utils.pending_order_filler.PENDING_ORDER_MODE", "observe"):
        with patch_gates(multiplier=0.5), patch_sizer(quantity=6), patch_execute():
            fill_pending_order(engine, order, fresh_bar())

    events = trade_events(engine, "pending_order_would_fill")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["fill_price"] == pytest.approx(LIMIT)
    assert payload["would_be_quantity"] == 6
    assert payload["mode"] == "observe"


def test_observe_mode_still_records_a_gate_rejection(engine, registry):
    """Revalidation runs in observe mode, so blockers are measurable."""
    order = claimed_order(registry)
    notes = [{"gate": "pre_trade_quality_gate", "decision": "reject"}]

    with patch("utils.pending_order_filler.PENDING_ORDER_MODE", "observe"):
        with patch_gates(proceed=False, notes=notes):
            result = fill_pending_order(engine, order, fresh_bar())

    assert result.reason == "pre_trade_quality_gate"
    assert registry.get_order(order.order_id).state is OrderState.REJECTED


def test_disabled_mode_releases_without_acting(engine, registry):
    order = claimed_order(registry)

    with patch("utils.pending_order_filler.PENDING_ORDER_MODE", "disabled"):
        with patch_execute() as execute:
            result = fill_pending_order(engine, order, fresh_bar())

    assert result.reason == "feature_disabled"
    execute.assert_not_called()
    assert registry.get_order(order.order_id).state is OrderState.PENDING


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


def test_unclaimed_order_is_refused(engine, registry, enabled):
    """The caller must have won the CAS claim first."""
    now = datetime.now(timezone.utc)
    order = PendingOrder(
        order_id=str(uuid.uuid4()),
        profile_id="moderate", symbol="META", side="BUY",
        setup_type="technical_breakout",
        limit_price=LIMIT, stop_price=STOP, target_price=TARGET,
        risk_reward=2.9, fresh_price_at_creation=601.24,
        runaway_pct_at_creation=0.0124,
        created_at=now - timedelta(minutes=10),
        expires_at=now + timedelta(hours=1),
    )
    registry.create_order(order)
    pending = registry.get_order(order.order_id)  # still PENDING

    with patch_execute() as execute:
        result = fill_pending_order(engine, pending, fresh_bar())

    assert result.success is False
    assert "invalid_state" in result.reason
    execute.assert_not_called()


def test_result_is_a_frozen_dataclass():
    result = FillResult("id", True, 1.0, "filled", 5)
    with pytest.raises(Exception):
        result.success = False
