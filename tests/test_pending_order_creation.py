"""Tests for utils/pending_order_creation.py.

Two regression guards in here encode the review findings that shaped the design
and must never be reintroduced:

- ``test_limit_price_is_never_the_profit_target`` — the source spec's motivating
  example proposed resting a buy limit at META's profit target, which yields a
  position whose entry equals its target.
- ``test_repaired_decision_rests_at_the_original_entry_not_the_chased_price`` —
  Tier 2 rewrites ``decision["price"]`` to the live quote, so sourcing the limit
  from the decision would rest the order at the chased price.

Requirements: 1.1-1.17, 3.4, 8.5, 10.1, 10.3, 14.1, 14.3, 14.7
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text

from db.schema import Base, init_pending_order_schema, init_trade_plan_schema
from utils.pending_order_creation import (
    BRANCH_NEITHER,
    BRANCH_RUNAWAY,
    BRANCH_TARGET_EXCEEDED,
    classify_stale_entry_branch,
    emit_repair_band_decline,
    maybe_create_pending_order,
)
from utils.pending_order_registry import OrderState, PendingOrderRegistry

# The corrected META scenario. Intended entry 593.87 sits BELOW the fresh price
# 601.24 by ~1.24%, which trips the runaway branch (threshold 1%). The profit
# target is 620.00, well above the fresh price, so the target branch stays quiet.
META_ENTRY = 593.87
META_FRESH = 601.24
META_STOP = 585.00
META_TARGET = 620.00


@pytest.fixture
def engine():
    """Full schema: pending orders, trade plans, and the ORM tables."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    init_pending_order_schema(eng)
    init_trade_plan_schema(eng)
    with eng.connect() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS pm_candidate_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT NOT NULL,
                    cycle_id TEXT,
                    profile_id TEXT,
                    event_type TEXT NOT NULL,
                    event_data TEXT,
                    created_at TEXT NOT NULL,
                    candidate_type TEXT
                )
                """
            )
        )
        conn.commit()
    return eng


@pytest.fixture
def registry(engine):
    return PendingOrderRegistry(engine)


@pytest.fixture
def enabled():
    """PENDING_ORDER_MODE must be non-disabled for creation to run at all."""
    with patch("utils.pending_order_creation.PENDING_ORDER_MODE", "enabled"):
        yield


# 11:00 ET on Friday 2026-08-14 — a trading day, comfortably mid-session.
FROZEN_NOW = datetime(2026, 8, 14, 15, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def frozen_clock():
    """Pin creation time so these tests do not depend on the wall clock.

    resolve_expiry() legitimately returns None outside the regular session and on
    non-trading days, which would make every creation decline with
    `window_too_short`. Without this, the suite would only pass while US markets
    happen to be open — a real trap, since it fails silently as a decline rather
    than as an obvious error.
    """
    with patch("utils.pending_order_creation.now_utc", return_value=FROZEN_NOW):
        yield


def decision(**overrides) -> dict:
    base = {
        "action": "BUY",
        "symbol": "META",
        "price": META_ENTRY,
        "entry_price": META_ENTRY,
        "stop": META_STOP,
        "target": META_TARGET,
        "quantity": 10,
        "setup_type": "technical_breakout",
        "rationale": "waiting for the pullback to 593.87",
    }
    base.update(overrides)
    return base


def create(engine, **overrides):
    kwargs = dict(
        db=engine,
        decision=decision(),
        profile_id="moderate",
        action="BUY",
        symbol="META",
        intended_entry=META_ENTRY,
        fresh_price=META_FRESH,
        stop=META_STOP,
        target=META_TARGET,
        stale_reason=(
            f"META: stale entry rejected - fresh price {META_FRESH:.2f} moved "
            f"1.24% beyond intended BUY entry {META_ENTRY:.2f}"
        ),
        engine=engine,
    )
    kwargs.update(overrides)
    return maybe_create_pending_order(**kwargs)


def trade_events(engine, event_type=None) -> list[dict]:
    clause = " WHERE event_type = :et" if event_type else ""
    params = {"et": event_type} if event_type else {}
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"SELECT event_type, symbol, profile, price, message, "
                f"payload_json FROM trade_events{clause} ORDER BY id ASC"
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
# Branch classification
# ---------------------------------------------------------------------------


def test_runaway_branch_is_classified():
    result = classify_stale_entry_branch(
        action="BUY", intended_entry=META_ENTRY,
        fresh_price=META_FRESH, target=META_TARGET,
    )
    assert result.branch == BRANCH_RUNAWAY
    assert result.is_runaway
    assert result.runaway_pct == pytest.approx(0.01241, abs=1e-4)


def test_target_exceeded_branch_is_classified():
    """The original spec's misread scenario: fresh price past the profit target."""
    result = classify_stale_entry_branch(
        action="BUY", intended_entry=588.00,
        fresh_price=601.24, target=593.87,
    )
    assert result.branch == BRANCH_TARGET_EXCEEDED


def test_short_runaway_branch_is_classified():
    """SHORT runs away when the fresh price drops BELOW the intended entry."""
    result = classify_stale_entry_branch(
        action="SHORT", intended_entry=100.00, fresh_price=98.00, target=90.00,
    )
    assert result.branch == BRANCH_RUNAWAY
    assert result.runaway_pct == pytest.approx(0.02)


def test_short_target_exceeded_branch_is_classified():
    result = classify_stale_entry_branch(
        action="SHORT", intended_entry=100.00, fresh_price=89.00, target=90.00,
    )
    assert result.branch == BRANCH_TARGET_EXCEEDED


def test_move_inside_the_threshold_is_neither():
    """0.5% is under the 1% threshold, so the check would not have rejected."""
    result = classify_stale_entry_branch(
        action="BUY", intended_entry=100.00, fresh_price=100.50, target=110.00,
    )
    assert result.branch == BRANCH_NEITHER
    assert result.runaway_pct == pytest.approx(0.005)


@pytest.mark.parametrize("action", ["CLOSE", "SELL", "", None])
def test_non_entry_actions_are_neither(action):
    result = classify_stale_entry_branch(
        action=action, intended_entry=100.0, fresh_price=105.0, target=110.0,
    )
    assert result.branch == BRANCH_NEITHER


@pytest.mark.parametrize(
    "entry,fresh", [(None, 100.0), (100.0, None), (0, 100.0), (-5, 100.0)]
)
def test_missing_or_nonpositive_prices_are_neither(entry, fresh):
    result = classify_stale_entry_branch(
        action="BUY", intended_entry=entry, fresh_price=fresh, target=110.0,
    )
    assert result.branch == BRANCH_NEITHER


def test_absent_target_still_allows_runaway_classification():
    """The check computes favorable move even when target is None."""
    result = classify_stale_entry_branch(
        action="BUY", intended_entry=100.0, fresh_price=105.0, target=None,
    )
    assert result.branch == BRANCH_RUNAWAY


def test_classification_matches_the_live_check_exactly():
    """Cross-check against _fresh_price_stale_entry_check on the same inputs.

    Guards the recompute-don't-parse decision: if the two ever disagree, orders
    would be created for cases the check allowed, or skipped for cases it blocked.
    """
    from agents.portfolio_manager import _fresh_price_stale_entry_check

    cases = [
        ("BUY", META_ENTRY, META_FRESH, META_TARGET),
        ("BUY", 588.0, 601.24, 593.87),
        ("BUY", 100.0, 100.5, 110.0),
        ("SHORT", 100.0, 98.0, 90.0),
        ("SHORT", 100.0, 89.0, 90.0),
        ("SHORT", 100.0, 99.8, 90.0),
        ("CLOSE", 100.0, 105.0, 110.0),
    ]

    for action, entry, fresh, target in cases:
        ok, _ = _fresh_price_stale_entry_check(
            action=action, symbol="X", intended_entry=entry,
            fresh_price=fresh, target=target,
        )
        branch = classify_stale_entry_branch(
            action=action, intended_entry=entry, fresh_price=fresh, target=target,
        ).branch

        rejected = not ok
        classified_as_rejection = branch in (BRANCH_RUNAWAY, BRANCH_TARGET_EXCEEDED)
        assert rejected == classified_as_rejection, (
            f"disagreement for {action} entry={entry} fresh={fresh} "
            f"target={target}: check rejected={rejected}, branch={branch}"
        )


# ---------------------------------------------------------------------------
# The regression guards
# ---------------------------------------------------------------------------


def test_limit_price_is_never_the_profit_target(engine, enabled):
    """REGRESSION GUARD for the source spec's misread.

    The original document proposed resting a buy limit at 593.87 while reading
    that number as the profit target. An order whose limit equals its target has
    zero reward and validate_trade() step 5 would reject it at fill time.
    """
    outcome = create(engine)
    assert outcome.created is True

    order = PendingOrderRegistry(engine).get_order(outcome.order_id)
    assert order.limit_price == pytest.approx(META_ENTRY)
    assert order.limit_price != pytest.approx(order.target_price)
    assert order.target_price == pytest.approx(META_TARGET)
    # And the geometry is genuinely tradable at the limit.
    assert order.stop_price < order.limit_price < order.target_price
    assert order.risk_reward > 0


def test_repaired_decision_rests_at_the_original_entry_not_the_chased_price(
    engine, enabled
):
    """REGRESSION GUARD for the Tier 2 repair interaction.

    Tier 2 sets decision["price"] = live_price and rescales stop/target. The
    limit must come from the caller's pre-repair snapshot, so a decision that
    went through repair can never rest an order at the chased price.
    """
    # A 2.7% runaway: above the 1% stale threshold, below the 5% ceiling, and
    # with the target still above the fresh price so the target branch stays
    # quiet. Tier 2 would have rewritten decision["price"] to the chased value.
    chased = 610.00
    repaired = decision(
        price=chased, entry_price=chased, stop=600.0, target=667.0
    )

    outcome = create(
        engine,
        decision=repaired,
        intended_entry=META_ENTRY,   # the pre-repair snapshot
        fresh_price=chased,
        stop=META_STOP,
        target=650.00,
    )
    assert outcome.created is True, outcome.decline_reason

    order = PendingOrderRegistry(engine).get_order(outcome.order_id)
    assert order.limit_price == pytest.approx(META_ENTRY)
    assert order.limit_price != pytest.approx(chased)
    assert order.limit_price != pytest.approx(repaired["price"])
    # The stop and target also come from the pre-repair values, not the rescaled ones.
    assert order.stop_price == pytest.approx(META_STOP)
    assert order.target_price == pytest.approx(650.00)


# ---------------------------------------------------------------------------
# Successful creation
# ---------------------------------------------------------------------------


def test_runaway_creates_a_pending_order(engine, enabled):
    outcome = create(engine)

    assert outcome.created is True
    assert outcome.order_id is not None
    assert outcome.decline_reason is None

    order = PendingOrderRegistry(engine).get_order(outcome.order_id)
    assert order.state is OrderState.PENDING
    assert order.symbol == "META"
    assert order.side == "BUY"
    assert order.setup_type == "technical_breakout"
    assert order.fresh_price_at_creation == pytest.approx(META_FRESH)
    assert order.runaway_pct_at_creation == pytest.approx(0.01241, abs=1e-4)
    assert order.intended_quantity == 10


def test_created_order_records_the_pm_rationale(engine, enabled):
    outcome = create(engine)
    order = PendingOrderRegistry(engine).get_order(outcome.order_id)
    assert "pullback" in order.pm_rationale


def test_creation_emits_a_trade_event_with_full_payload(engine, enabled):
    outcome = create(engine)
    events = trade_events(engine, "pending_order_created")

    assert len(events) == 1
    payload = events[0]["payload"]
    for key in (
        "order_id", "symbol", "side", "setup_type", "limit_price", "stop_price",
        "target_price", "risk_reward", "intended_quantity",
        "fresh_price_at_creation", "runaway_pct_at_creation", "expires_at",
        "candidate_id", "cycle_id", "profile_id",
    ):
        assert key in payload, f"missing {key}"
    assert payload["order_id"] == outcome.order_id
    assert payload["limit_price"] == pytest.approx(META_ENTRY)


def test_creation_event_uses_its_own_session_and_survives(engine, enabled):
    """execute_trade() returns without committing on this path, so the event
    must be committed by a dedicated session or it would be lost."""
    create(engine)
    assert len(trade_events(engine, "pending_order_created")) == 1


def test_short_runaway_creates_a_short_order(engine, enabled):
    outcome = create(
        engine,
        decision=decision(action="SHORT", symbol="AMD"),
        action="SHORT",
        symbol="AMD",
        intended_entry=100.00,
        fresh_price=98.00,
        stop=105.00,
        target=90.00,
    )
    assert outcome.created is True

    order = PendingOrderRegistry(engine).get_order(outcome.order_id)
    assert order.side == "SHORT"
    assert order.target_price < order.limit_price < order.stop_price


def test_risk_reward_is_computed_at_the_limit(engine, enabled):
    outcome = create(engine)
    order = PendingOrderRegistry(engine).get_order(outcome.order_id)

    # risk = 593.87 - 585.00 = 8.87 ; reward = 620.00 - 593.87 = 26.13
    assert order.risk_reward == pytest.approx(26.13 / 8.87, abs=1e-3)


# ---------------------------------------------------------------------------
# Decline gates
# ---------------------------------------------------------------------------


def test_target_exceeded_declines_and_creates_nothing(engine, enabled):
    """v1 keeps rejecting this branch, but records it for measurement."""
    outcome = create(
        engine, intended_entry=588.00, fresh_price=601.24, target=593.87,
    )

    assert outcome.created is False
    assert outcome.decline_reason == "target_already_exceeded"
    assert PendingOrderRegistry(engine).get_active_orders() == []

    events = trade_events(engine, "pending_order_declined")
    assert len(events) == 1
    assert events[0]["payload"]["reason"] == "target_already_exceeded"


def test_runaway_over_the_ceiling_declines(engine, enabled):
    """Non-binding in production today, but the guard must still work."""
    with patch("utils.pending_order_creation.PENDING_ORDER_MAX_RUNAWAY_PCT", 0.01):
        outcome = create(engine)

    assert outcome.decline_reason == "runaway_exceeds_max"
    assert PendingOrderRegistry(engine).get_active_orders() == []


@pytest.mark.parametrize(
    "field,value",
    [("stop", None), ("stop", 0), ("target", None), ("target", 0)],
)
def test_incomplete_geometry_declines(engine, enabled, field, value):
    outcome = create(engine, **{field: value})

    assert outcome.decline_reason == "incomplete_geometry"
    assert PendingOrderRegistry(engine).get_active_orders() == []


def test_missing_intended_entry_declines(engine, enabled):
    """Classification returns 'neither' without an entry, so nothing is created."""
    outcome = create(engine, intended_entry=None)
    assert outcome.created is False
    assert PendingOrderRegistry(engine).get_active_orders() == []


@pytest.mark.parametrize(
    "stop,target,label",
    [
        (600.0, 620.0, "BUY stop above the limit"),
        (593.87, 620.0, "BUY stop equal to the limit"),
    ],
)
def test_invalid_geometry_at_the_limit_declines(engine, enabled, stop, target, label):
    """Validated AT THE LIMIT, because that is where the fill would occur.

    Only stop-side violations are reachable through this path. A BUY runaway
    requires fresh > limit, so a target at or below the limit would also be at or
    below the fresh price — which the target-exceeded gate catches first. For a
    BUY runaway, `limit < target` is therefore structurally guaranteed. The
    unreachable combinations are covered directly in
    ``test_geometry_validation_matrix``.
    """
    outcome = create(engine, stop=stop, target=target)

    assert outcome.decline_reason == "invalid_geometry_at_limit", label
    assert PendingOrderRegistry(engine).get_active_orders() == []


def test_target_at_or_below_the_limit_is_caught_as_target_exceeded(engine, enabled):
    """Documents why the geometry gate cannot see these cases.

    They are not unvalidated — they are intercepted earlier by a gate that
    describes the situation more accurately.
    """
    for target in (590.00, META_ENTRY):
        outcome = create(engine, stop=585.0, target=target)
        assert outcome.decline_reason == "target_already_exceeded"
        assert PendingOrderRegistry(engine).get_active_orders() == []


@pytest.mark.parametrize(
    "action,limit,stop,target,expected,label",
    [
        ("BUY", 100.0, 95.0, 110.0, True, "BUY well-formed"),
        ("BUY", 100.0, 100.0, 110.0, False, "BUY stop == limit"),
        ("BUY", 100.0, 105.0, 110.0, False, "BUY stop above limit"),
        ("BUY", 100.0, 95.0, 100.0, False, "BUY target == limit"),
        ("BUY", 100.0, 95.0, 90.0, False, "BUY target below limit"),
        ("SHORT", 100.0, 105.0, 90.0, True, "SHORT well-formed"),
        ("SHORT", 100.0, 100.0, 90.0, False, "SHORT stop == limit"),
        ("SHORT", 100.0, 95.0, 90.0, False, "SHORT stop below limit"),
        ("SHORT", 100.0, 105.0, 100.0, False, "SHORT target == limit"),
        ("SHORT", 100.0, 105.0, 110.0, False, "SHORT target above limit"),
    ],
)
def test_geometry_validation_matrix(action, limit, stop, target, expected, label):
    """Exhaustive check of the geometry predicate, including states that are
    unreachable through the creation path but must still be handled."""
    from utils.pending_order_creation import _geometry_valid_at_limit

    assert _geometry_valid_at_limit(action, limit, stop, target) is expected, label


def test_window_too_short_declines(engine, enabled):
    with patch(
        "utils.pending_order_creation.resolve_expiry", return_value=None
    ):
        outcome = create(engine)

    assert outcome.decline_reason == "window_too_short"
    assert PendingOrderRegistry(engine).get_active_orders() == []


def test_active_order_cap_declines(engine, enabled):
    with patch(
        "utils.pending_order_creation.PENDING_ORDER_MAX_ACTIVE_PER_PROFILE", 1
    ):
        first = create(engine, decision=decision(symbol="AMD"), symbol="AMD")
        assert first.created is True

        second = create(engine, decision=decision(symbol="NVDA"), symbol="NVDA")

    assert second.decline_reason == "active_order_cap_reached"
    assert len(PendingOrderRegistry(engine).get_active_orders()) == 1


def test_cap_does_not_spuriously_decline_when_the_blocker_is_a_duplicate(
    engine, enabled
):
    """Duplicates are subtracted from the cap before it is applied.

    Checking the cap with the duplicate counted would decline an order whose only
    blocker is about to be superseded; superseding first would cancel a valid
    order and then decline anyway. Neither is acceptable.
    """
    with patch(
        "utils.pending_order_creation.PENDING_ORDER_MAX_ACTIVE_PER_PROFILE", 1
    ):
        first = create(engine)
        assert first.created is True

        # Same active key, so the existing order will be superseded.
        second = create(engine)

    assert second.created is True, "the duplicate should free the slot"
    assert second.decline_reason is None
    assert first.order_id in second.superseded



# ---------------------------------------------------------------------------
# Supersession
# ---------------------------------------------------------------------------


def test_duplicate_key_supersedes_the_old_order(engine, enabled):
    first = create(engine)
    second = create(engine)

    assert second.created is True
    assert first.order_id in second.superseded

    reg = PendingOrderRegistry(engine)
    assert reg.get_order(first.order_id).state is OrderState.CANCELED
    assert reg.get_order(first.order_id).terminal_reason == "superseded"
    assert reg.get_order(second.order_id).state is OrderState.PENDING


def test_only_one_active_order_survives_repeated_creation(engine, enabled):
    for _ in range(4):
        create(engine)

    assert len(PendingOrderRegistry(engine).get_active_orders()) == 1


def test_different_side_does_not_supersede(engine, enabled):
    long_order = create(engine)
    short_order = create(
        engine,
        decision=decision(action="SHORT"),
        action="SHORT",
        intended_entry=100.0, fresh_price=98.0, stop=105.0, target=90.0,
    )

    assert short_order.created is True
    assert short_order.superseded == ()
    assert len(PendingOrderRegistry(engine).get_active_orders()) == 2
    assert long_order.order_id is not None


# ---------------------------------------------------------------------------
# Action and mode guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action", ["CLOSE", "SELL", "", "buy_maybe"])
def test_non_entry_actions_create_nothing(engine, enabled, action):
    outcome = create(engine, action=action)

    assert outcome.created is False
    assert outcome.decline_reason is None
    assert PendingOrderRegistry(engine).get_active_orders() == []
    assert trade_events(engine) == []


def test_disabled_mode_creates_nothing_and_emits_nothing(engine):
    with patch("utils.pending_order_creation.PENDING_ORDER_MODE", "disabled"):
        outcome = create(engine)

    assert outcome.created is False
    assert PendingOrderRegistry(engine).get_active_orders() == []
    assert trade_events(engine) == []


def test_observe_mode_still_creates_orders(engine):
    """Observe suppresses fills, not creation — that is what makes it measurable."""
    with patch("utils.pending_order_creation.PENDING_ORDER_MODE", "observe"):
        outcome = create(engine)

    assert outcome.created is True
    assert len(PendingOrderRegistry(engine).get_active_orders()) == 1


def test_lowercase_action_is_normalized(engine, enabled):
    outcome = create(engine, action="buy")
    assert outcome.created is True
    order = PendingOrderRegistry(engine).get_order(outcome.order_id)
    assert order.side == "BUY"


# ---------------------------------------------------------------------------
# Fail-open behavior
# ---------------------------------------------------------------------------


def test_registry_failure_does_not_raise(engine, enabled):
    """Creation is additive; a persistence failure must not escape."""
    with patch.object(
        PendingOrderRegistry, "create_order",
        side_effect=Exception("db exploded"),
    ):
        outcome = create(engine)

    assert outcome.created is False


def test_event_emission_failure_does_not_prevent_the_order(engine, enabled):
    with patch(
        "utils.pending_order_creation._emit_trade_event",
        side_effect=RuntimeError("no session"),
    ):
        outcome = create(engine)

    assert outcome.created is True
    assert PendingOrderRegistry(engine).get_order(outcome.order_id) is not None


def test_cap_lookup_failure_declines_quietly(engine, enabled):
    with patch.object(
        PendingOrderRegistry, "count_active_for_profile",
        side_effect=Exception("locked"),
    ):
        outcome = create(engine)

    assert outcome.created is False
    assert outcome.decline_reason is None


def test_signal_snapshot_failure_is_tolerated(engine, enabled):
    with patch(
        "utils.pending_order_creation._capture_signal_snapshot",
        side_effect=Exception("no memory"),
    ):
        # The helper itself is fail-open, but assert the caller survives too.
        try:
            outcome = create(engine)
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"creation should not raise: {exc}")
    assert outcome.created in (True, False)


# ---------------------------------------------------------------------------
# Candidate linkage
# ---------------------------------------------------------------------------


def test_creation_without_a_candidate_succeeds(engine, enabled):
    """The live legacy PM path has PM_CANDIDATE_MODE disabled and no candidate."""
    outcome = create(engine)

    order = PendingOrderRegistry(engine).get_order(outcome.order_id)
    assert order.candidate_id is None
    assert order.cycle_id is None
    assert order.plan_id is None


def test_candidate_linkage_is_recorded_when_present(engine, enabled):
    outcome = create(
        engine,
        decision=decision(pm_candidate_id="cand-42", cycle_id="cycle-7"),
    )
    assert outcome.created is True

    order = PendingOrderRegistry(engine).get_order(outcome.order_id)
    assert order.candidate_id == "cand-42"
    assert order.cycle_id == "cycle-7"

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT event_type, event_data FROM pm_candidate_events "
                "WHERE candidate_id = 'cand-42'"
            )
        ).fetchone()
    assert row is not None
    assert row[0] == "pending_order_created"
    assert json.loads(row[1])["order_id"] == outcome.order_id


def test_candidate_linkage_failure_does_not_undo_the_order(engine, enabled):
    """Nothing after persistence may turn a created order into an exception."""
    with patch(
        "utils.pending_order_creation._link_candidate",
        side_effect=RuntimeError("no table"),
    ):
        outcome = create(engine, decision=decision(pm_candidate_id="cand-1"))

    assert outcome.created is True
    assert len(PendingOrderRegistry(engine).get_active_orders()) == 1


# ---------------------------------------------------------------------------
# Repair-band decline
# ---------------------------------------------------------------------------


def test_repair_band_decline_emits_an_event(engine):
    with patch("utils.pending_order_creation.PENDING_ORDER_MODE", "observe"):
        emit_repair_band_decline(
            db=engine,
            profile_id="moderate",
            symbol="META",
            action="BUY",
            original_intended_entry=META_ENTRY,
            live_price=640.00,
            deviation=0.0725,
            original_stop=META_STOP,
            original_target=META_TARGET,
            repaired_stop=630.00,
            repaired_target=668.00,
        )

    events = trade_events(engine, "pending_order_declined")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["reason"] == "repaired_before_check"
    assert payload["original_intended_entry"] == pytest.approx(META_ENTRY)
    assert payload["live_price"] == pytest.approx(640.00)
    assert payload["deviation_pct"] == pytest.approx(0.0725)
    assert payload["repaired_stop"] == pytest.approx(630.00)


def test_repair_band_decline_creates_no_order(engine):
    """The repaired trade still executes; resting an order too would double-book."""
    with patch("utils.pending_order_creation.PENDING_ORDER_MODE", "enabled"):
        emit_repair_band_decline(
            db=engine, profile_id="moderate", symbol="META", action="BUY",
            original_intended_entry=META_ENTRY, live_price=640.0,
            deviation=0.0725, original_stop=META_STOP,
            original_target=META_TARGET, repaired_stop=630.0,
            repaired_target=668.0,
        )

    assert PendingOrderRegistry(engine).get_active_orders() == []


def test_repair_band_decline_is_silent_when_disabled(engine):
    with patch("utils.pending_order_creation.PENDING_ORDER_MODE", "disabled"):
        emit_repair_band_decline(
            db=engine, profile_id="moderate", symbol="META", action="BUY",
            original_intended_entry=META_ENTRY, live_price=640.0,
            deviation=0.0725, original_stop=META_STOP,
            original_target=META_TARGET, repaired_stop=630.0,
            repaired_target=668.0,
        )

    assert trade_events(engine) == []


def test_repair_band_decline_never_raises(engine):
    with patch("utils.pending_order_creation.PENDING_ORDER_MODE", "observe"):
        with patch(
            "utils.pending_order_creation._emit_trade_event",
            side_effect=RuntimeError("boom"),
        ):
            emit_repair_band_decline(
                db=engine, profile_id="moderate", symbol="META", action="BUY",
                original_intended_entry=META_ENTRY, live_price=640.0,
                deviation=0.0725, original_stop=META_STOP,
                original_target=META_TARGET, repaired_stop=630.0,
                repaired_target=668.0,
            )
    # Reaching here without an exception is the assertion.


# ---------------------------------------------------------------------------
# Time-of-day dependence
#
# These pin the behavior that made the rest of this file wall-clock dependent
# before FROZEN_NOW was introduced. resolve_expiry() refusing to open a window
# outside the session is correct, but it surfaces as a silent decline rather than
# an error — so it is worth asserting explicitly rather than leaving implicit.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "frozen_at,label",
    [
        (datetime(2026, 8, 14, 20, 30, tzinfo=timezone.utc), "16:30 ET, after close"),
        (datetime(2026, 8, 14, 23, 0, tzinfo=timezone.utc), "19:00 ET, evening"),
        (datetime(2026, 8, 15, 15, 0, tzinfo=timezone.utc), "Saturday"),
        (datetime(2026, 8, 16, 15, 0, tzinfo=timezone.utc), "Sunday"),
    ],
)
def test_outside_the_session_creation_declines_as_window_too_short(
    engine, enabled, frozen_at, label
):
    with patch("utils.pending_order_creation.now_utc", return_value=frozen_at):
        outcome = create(engine)

    assert outcome.created is False, label
    assert outcome.decline_reason == "window_too_short", label
    assert PendingOrderRegistry(engine).get_active_orders() == []

    declines = trade_events(engine, "pending_order_declined")
    assert len(declines) == 1
    assert declines[0]["payload"]["reason"] == "window_too_short"


def test_inside_the_session_creation_succeeds(engine, enabled):
    """The counterpart, so the parametrized cases above cannot pass vacuously."""
    mid_session = datetime(2026, 8, 14, 15, 0, tzinfo=timezone.utc)  # 11:00 ET Fri

    with patch("utils.pending_order_creation.now_utc", return_value=mid_session):
        outcome = create(engine)

    assert outcome.created is True
    assert outcome.decline_reason is None
