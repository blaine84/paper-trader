"""Tests for utils/pending_order_registry.py — CAS state machine.

Requirements: 2.1-2.7, 3.9, 3.10, 4.9, 7.3, 7.4, 8.4, 9.1-9.12, 10.8
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text

from db.schema import init_pending_order_schema
from utils.pending_order_registry import (
    CANCEL_REASONS,
    PERMITTED_TRANSITIONS,
    TERMINAL_STATES,
    TRANSIENT_STATES,
    OrderState,
    PendingOrder,
    PendingOrderRegistry,
    PendingOrderRegistryError,
    compute_order_integrity_hash,
)
from utils.pending_order_time import to_iso

NOW = datetime(2026, 8, 14, 14, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    init_pending_order_schema(eng)
    return eng


@pytest.fixture
def registry(engine):
    return PendingOrderRegistry(engine)


def make_order(**overrides) -> PendingOrder:
    defaults = dict(
        order_id=str(uuid.uuid4()),
        profile_id="moderate",
        symbol="META",
        side="BUY",
        setup_type="technical_breakout",
        limit_price=593.87,
        stop_price=588.00,
        target_price=605.00,
        risk_reward=1.9,
        fresh_price_at_creation=601.24,
        runaway_pct_at_creation=0.0124,
        created_at=NOW,
        expires_at=NOW + timedelta(hours=2),
    )
    defaults.update(overrides)
    return PendingOrder(**defaults)


def create(registry, **overrides) -> str:
    return registry.create_order(make_order(**overrides))


def state_of(registry, order_id) -> OrderState:
    return registry.get_order(order_id).state


# ---------------------------------------------------------------------------
# Value object and hashing
# ---------------------------------------------------------------------------


def test_state_sets_are_coherent():
    assert TERMINAL_STATES | TRANSIENT_STATES == set(OrderState)
    assert not (TERMINAL_STATES & TRANSIENT_STATES)


def test_permitted_transitions_match_the_spec():
    assert PERMITTED_TRANSITIONS == frozenset({
        (OrderState.PENDING, OrderState.FILLING),
        (OrderState.PENDING, OrderState.EXPIRED),
        (OrderState.PENDING, OrderState.CANCELED),
        (OrderState.FILLING, OrderState.FILLED),
        (OrderState.FILLING, OrderState.REJECTED),
        (OrderState.FILLING, OrderState.CANCELED),
        (OrderState.FILLING, OrderState.PENDING),
    })


def test_no_transition_originates_from_a_terminal_state():
    """Terminal states are final (Requirement 9.2)."""
    for source, _ in PERMITTED_TRANSITIONS:
        assert source not in TERMINAL_STATES


def test_pending_to_rejected_is_not_permitted():
    """'rejected' is reserved for the fill path, reached only via FILLING."""
    assert (OrderState.PENDING, OrderState.REJECTED) not in PERMITTED_TRANSITIONS


def test_integrity_hash_is_stable_and_covers_geometry():
    order = make_order()
    assert compute_order_integrity_hash(order) == compute_order_integrity_hash(order)

    moved = make_order(order_id=order.order_id, limit_price=590.00)
    assert compute_order_integrity_hash(moved) != compute_order_integrity_hash(order)


def test_integrity_hash_ignores_lifecycle_fields():
    """Only identity and geometry are covered, so lifecycle updates don't void it."""
    base = make_order()
    later = make_order(
        order_id=base.order_id,
        state=OrderState.FILLING,
        trade_id=42,
        fill_price=593.87,
    )
    assert compute_order_integrity_hash(base) == compute_order_integrity_hash(later)


def test_active_key_is_the_constrained_tuple():
    order = make_order()
    assert order.active_key == (
        "moderate", "META", "BUY", "technical_breakout",
    )


# ---------------------------------------------------------------------------
# Creation and round-trip
# ---------------------------------------------------------------------------


def test_create_then_get_preserves_every_field(registry):
    order = make_order(
        geometry_name="pullback",
        candidate_id="cand-1",
        cycle_id="cycle-1",
        source_signal_id="sig-1",
        plan_id="plan-1",
        intended_quantity=10,
        pm_rationale="waiting for the pullback",
        signal_snapshot_json='{"strength": 8}',
    )
    registry.create_order(order)

    loaded = registry.get_order(order.order_id)
    assert loaded is not None
    assert loaded.order_id == order.order_id
    assert loaded.profile_id == order.profile_id
    assert loaded.symbol == order.symbol
    assert loaded.side == order.side
    assert loaded.setup_type == order.setup_type
    assert loaded.geometry_name == "pullback"
    assert loaded.candidate_id == "cand-1"
    assert loaded.cycle_id == "cycle-1"
    assert loaded.source_signal_id == "sig-1"
    assert loaded.plan_id == "plan-1"
    assert loaded.limit_price == pytest.approx(593.87)
    assert loaded.stop_price == pytest.approx(588.00)
    assert loaded.target_price == pytest.approx(605.00)
    assert loaded.risk_reward == pytest.approx(1.9)
    assert loaded.intended_quantity == 10
    assert loaded.fresh_price_at_creation == pytest.approx(601.24)
    assert loaded.runaway_pct_at_creation == pytest.approx(0.0124)
    assert loaded.pm_rationale == "waiting for the pullback"
    assert loaded.signal_snapshot_json == '{"strength": 8}'
    assert loaded.state is OrderState.PENDING
    assert loaded.created_at == NOW
    assert loaded.expires_at == NOW + timedelta(hours=2)


def test_created_order_has_an_integrity_hash(registry):
    order_id = create(registry)
    assert registry.get_order(order_id).integrity_hash


def test_supplied_integrity_hash_is_preserved(registry):
    order = make_order(integrity_hash="explicit-hash")
    registry.create_order(order)
    assert registry.get_order(order.order_id).integrity_hash == "explicit-hash"


def test_datetimes_round_trip_as_aware_utc(registry):
    order_id = create(registry)
    loaded = registry.get_order(order_id)
    assert loaded.created_at.tzinfo is not None
    assert loaded.expires_at.tzinfo is not None


def test_get_order_returns_none_for_unknown_id(registry):
    assert registry.get_order("does-not-exist") is None


def test_creating_in_a_non_pending_state_is_rejected(registry):
    with pytest.raises(PendingOrderRegistryError, match="must start PENDING"):
        registry.create_order(make_order(state=OrderState.FILLED))


def test_create_emits_a_pending_event(registry):
    order_id = create(registry)
    events = registry.get_events(order_id)

    assert len(events) == 1
    assert events[0]["event_type"] == "state_pending"
    assert events[0]["to_state"] == "pending"
    assert events[0]["event_data"]["limit_price"] == pytest.approx(593.87)


# ---------------------------------------------------------------------------
# claim_for_fill — the atomicity guarantee
# ---------------------------------------------------------------------------


def test_claim_succeeds_from_pending(registry):
    order_id = create(registry)
    claimed, reason = registry.claim_for_fill(order_id)

    assert claimed is True
    assert reason is None
    assert state_of(registry, order_id) is OrderState.FILLING


def test_only_one_of_two_concurrent_claims_wins(registry):
    """The reason FILLING exists: two ticks must not both fill one order."""
    order_id = create(registry)

    first, _ = registry.claim_for_fill(order_id)
    second, second_reason = registry.claim_for_fill(order_id)

    assert first is True
    assert second is False
    assert "CAS failed" in second_reason


def test_lost_claim_returns_false_instead_of_raising(registry):
    """A lost race is expected in a polling monitor, not an error."""
    order_id = create(registry)
    registry.claim_for_fill(order_id)

    claimed, reason = registry.claim_for_fill(order_id)
    assert claimed is False
    assert reason is not None


def test_claim_on_unknown_order_returns_false(registry):
    claimed, reason = registry.claim_for_fill("nope")
    assert claimed is False
    assert reason is not None


@pytest.mark.parametrize(
    "terminal", [OrderState.FILLED, OrderState.EXPIRED, OrderState.CANCELED]
)
def test_claim_fails_on_a_terminal_order(registry, terminal):
    order_id = create(registry)
    _force_state(registry, order_id, terminal)

    claimed, _ = registry.claim_for_fill(order_id)
    assert claimed is False


def _force_state(registry, order_id, state: OrderState) -> None:
    """Set state directly, bypassing CAS — for constructing test fixtures."""
    with registry._engine.connect() as conn:
        conn.execute(
            text("UPDATE pending_orders SET state = :s WHERE order_id = :oid"),
            {"s": state.value, "oid": order_id},
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Legal transitions
# ---------------------------------------------------------------------------


def test_release_claim_returns_to_pending(registry):
    order_id = create(registry)
    registry.claim_for_fill(order_id)
    registry.release_claim(order_id, reason="stale_fill_bar")

    assert state_of(registry, order_id) is OrderState.PENDING


def test_mark_filled_records_the_fill_details(registry):
    order_id = create(registry)
    registry.claim_for_fill(order_id)

    bar_ts = NOW + timedelta(minutes=5)
    registry.mark_filled(
        order_id,
        fill_price=593.87,
        fill_policy="limit_price",
        fill_bar_ts=bar_ts,
        trade_id=77,
    )

    loaded = registry.get_order(order_id)
    assert loaded.state is OrderState.FILLED
    assert loaded.fill_price == pytest.approx(593.87)
    assert loaded.fill_policy == "limit_price"
    assert loaded.fill_bar_ts == bar_ts
    assert loaded.trade_id == 77
    assert loaded.filled_at is not None
    assert loaded.terminal_at is not None


def test_mark_rejected_records_the_reason(registry):
    order_id = create(registry)
    registry.claim_for_fill(order_id)
    registry.mark_rejected(order_id, "setup_quality_gate")

    loaded = registry.get_order(order_id)
    assert loaded.state is OrderState.REJECTED
    assert loaded.terminal_reason == "setup_quality_gate"
    assert loaded.terminal_at is not None


def test_mark_canceled_from_pending(registry):
    order_id = create(registry)
    registry.mark_canceled(order_id, "signal_flipped")

    loaded = registry.get_order(order_id)
    assert loaded.state is OrderState.CANCELED
    assert loaded.terminal_reason == "signal_flipped"


def test_mark_canceled_cascades_from_filling(registry):
    """The PENDING attempt fails, so it falls through to FILLING."""
    order_id = create(registry)
    registry.claim_for_fill(order_id)
    registry.mark_canceled(order_id, "insufficient_buying_power")

    loaded = registry.get_order(order_id)
    assert loaded.state is OrderState.CANCELED
    assert loaded.terminal_reason == "insufficient_buying_power"


def test_mark_expired_from_pending(registry):
    order_id = create(registry)
    registry.mark_expired(order_id)

    loaded = registry.get_order(order_id)
    assert loaded.state is OrderState.EXPIRED
    assert loaded.terminal_reason == "window_elapsed"


def test_every_cancel_reason_in_the_vocabulary_is_accepted(registry):
    for reason in sorted(CANCEL_REASONS):
        order_id = create(registry, symbol=f"SYM{hash(reason) % 1000}")
        registry.mark_canceled(order_id, reason)
        assert registry.get_order(order_id).terminal_reason == reason


def test_off_vocabulary_cancel_reason_warns_but_still_records(registry, caplog):
    order_id = create(registry)
    with caplog.at_level("WARNING"):
        registry.mark_canceled(order_id, "improvised_reason")

    assert registry.get_order(order_id).state is OrderState.CANCELED
    assert "closed vocabulary" in caplog.text


# ---------------------------------------------------------------------------
# Illegal transitions
# ---------------------------------------------------------------------------


def test_mark_filled_from_pending_is_rejected(registry):
    """A fill must be claimed first — no PENDING -> FILLED shortcut."""
    order_id = create(registry)
    with pytest.raises(PendingOrderRegistryError):
        registry.mark_filled(
            order_id, fill_price=593.87, fill_policy="limit_price",
            fill_bar_ts=NOW,
        )
    assert state_of(registry, order_id) is OrderState.PENDING


def test_mark_rejected_from_pending_is_rejected(registry):
    order_id = create(registry)
    with pytest.raises(PendingOrderRegistryError):
        registry.mark_rejected(order_id, "setup_quality_gate")


def test_mark_expired_from_filling_is_rejected(registry):
    """Must be released to PENDING first — the sweep does exactly that."""
    order_id = create(registry)
    registry.claim_for_fill(order_id)
    with pytest.raises(PendingOrderRegistryError):
        registry.mark_expired(order_id)


@pytest.mark.parametrize(
    "terminal",
    [OrderState.FILLED, OrderState.EXPIRED, OrderState.CANCELED, OrderState.REJECTED],
)
def test_no_transition_out_of_a_terminal_state_succeeds(registry, terminal):
    order_id = create(registry)
    _force_state(registry, order_id, terminal)

    with pytest.raises(PendingOrderRegistryError):
        registry.mark_canceled(order_id, "signal_flipped")

    assert state_of(registry, order_id) is terminal


def test_release_claim_from_pending_is_rejected(registry):
    order_id = create(registry)
    with pytest.raises(PendingOrderRegistryError):
        registry.release_claim(order_id, reason="whatever")


def test_unknown_column_in_extra_fields_is_rejected(registry):
    """Guards against a typo silently producing a no-op UPDATE."""
    order_id = create(registry)
    registry.claim_for_fill(order_id)

    with pytest.raises(PendingOrderRegistryError, match="Unknown column"):
        registry._transition(
            order_id, OrderState.FILLING, OrderState.FILLED,
            extra_fields={"not_a_column": 1},
        )


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def test_every_transition_emits_an_event(registry):
    order_id = create(registry)
    registry.claim_for_fill(order_id)
    registry.mark_filled(
        order_id, fill_price=593.87, fill_policy="limit_price", fill_bar_ts=NOW,
    )

    types = [e["event_type"] for e in registry.get_events(order_id)]
    assert types == ["state_pending", "state_filling", "state_filled"]


def test_events_record_from_and_to_state(registry):
    order_id = create(registry)
    registry.claim_for_fill(order_id)

    claim_event = registry.get_events(order_id)[1]
    assert claim_event["from_state"] == "pending"
    assert claim_event["to_state"] == "filling"


def test_events_are_ordered_oldest_first(registry):
    order_id = create(registry)
    registry.claim_for_fill(order_id)
    registry.release_claim(order_id, reason="stale_fill_bar")
    registry.mark_canceled(order_id, "signal_flipped")

    ids = [e["id"] for e in registry.get_events(order_id)]
    assert ids == sorted(ids)


def test_advance_watermark_emits_no_event(registry):
    """It runs every tick; an event per call would swamp the audit trail."""
    order_id = create(registry)
    before = len(registry.get_events(order_id))

    registry.advance_watermark(order_id, NOW + timedelta(minutes=1))

    assert len(registry.get_events(order_id)) == before


def test_event_write_failure_does_not_block_the_transition(registry):
    """Event emission is fail-open — the state change is already committed."""
    order_id = create(registry)

    with patch.object(
        registry, "_execute_event_write", side_effect=RuntimeError("disk full")
    ):
        registry.claim_for_fill(order_id)

    assert state_of(registry, order_id) is OrderState.FILLING


def test_create_succeeds_even_if_its_event_fails(registry):
    order = make_order()
    with patch.object(
        registry, "_execute_event_write", side_effect=RuntimeError("boom")
    ):
        order_id = registry.create_order(order)

    assert registry.get_order(order_id) is not None


# ---------------------------------------------------------------------------
# Watermark
# ---------------------------------------------------------------------------


def test_advance_watermark_sets_the_value(registry):
    order_id = create(registry)
    bar_ts = NOW + timedelta(minutes=3)
    registry.advance_watermark(order_id, bar_ts)

    assert registry.get_order(order_id).last_evaluated_bar_ts == bar_ts


def test_watermark_never_moves_backwards(registry):
    """A backwards watermark would let an evaluated bar be rescanned."""
    order_id = create(registry)
    later = NOW + timedelta(minutes=10)
    earlier = NOW + timedelta(minutes=2)

    registry.advance_watermark(order_id, later)
    registry.advance_watermark(order_id, earlier)

    assert registry.get_order(order_id).last_evaluated_bar_ts == later


def test_advance_watermark_tolerates_none(registry):
    order_id = create(registry)
    registry.advance_watermark(order_id, None)
    assert registry.get_order(order_id).last_evaluated_bar_ts is None


def test_advance_watermark_is_fail_open(registry):
    """A lost watermark costs a redundant rescan, not a failed tick."""
    order_id = create(registry)
    with patch.object(
        registry, "_execute_watermark_write", side_effect=RuntimeError("locked")
    ):
        registry.advance_watermark(order_id, NOW + timedelta(minutes=1))

    assert state_of(registry, order_id) is OrderState.PENDING


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def test_get_pending_orders_excludes_filling(registry):
    pending_id = create(registry, symbol="META")
    filling_id = create(registry, symbol="AMD")
    registry.claim_for_fill(filling_id)

    ids = [o.order_id for o in registry.get_pending_orders()]
    assert pending_id in ids
    assert filling_id not in ids


def test_get_active_orders_includes_both_transient_states(registry):
    pending_id = create(registry, symbol="META")
    filling_id = create(registry, symbol="AMD")
    registry.claim_for_fill(filling_id)

    ids = {o.order_id for o in registry.get_active_orders()}
    assert {pending_id, filling_id} <= ids


def test_get_active_orders_excludes_terminal(registry):
    active_id = create(registry, symbol="META")
    done_id = create(registry, symbol="AMD")
    registry.mark_canceled(done_id, "signal_flipped")

    ids = {o.order_id for o in registry.get_active_orders()}
    assert active_id in ids
    assert done_id not in ids


def test_get_active_orders_filters_by_symbol(registry):
    create(registry, symbol="META")
    amd_id = create(registry, symbol="AMD")

    result = registry.get_active_orders(symbol="AMD")
    assert [o.order_id for o in result] == [amd_id]


def test_count_active_for_profile_counts_pending_and_filling(registry):
    create(registry, symbol="META")
    filling_id = create(registry, symbol="AMD")
    registry.claim_for_fill(filling_id)
    done_id = create(registry, symbol="NVDA")
    registry.mark_canceled(done_id, "signal_flipped")

    assert registry.count_active_for_profile("moderate") == 2
    assert registry.count_active_for_profile("aggressive") == 0


def test_get_orders_for_profile_can_include_terminal(registry):
    active_id = create(registry, symbol="META")
    done_id = create(registry, symbol="AMD")
    registry.mark_canceled(done_id, "signal_flipped")

    active_only = {o.order_id for o in registry.get_orders_for_profile("moderate")}
    everything = {
        o.order_id
        for o in registry.get_orders_for_profile("moderate", include_terminal=True)
    }

    assert active_only == {active_id}
    assert everything == {active_id, done_id}


# ---------------------------------------------------------------------------
# Duplicates and supersession
# ---------------------------------------------------------------------------


def test_find_duplicate_active_matches_the_full_key(registry):
    order_id = create(registry)

    assert registry.find_duplicate_active(
        "moderate", "META", "BUY", "technical_breakout"
    ) == [order_id]
    assert registry.find_duplicate_active(
        "moderate", "META", "SHORT", "technical_breakout"
    ) == []


def test_supersede_cancels_only_the_matching_key(registry):
    same_key = create(registry)
    other_symbol = create(registry, symbol="AMD")
    other_side = create(registry, side="SHORT")

    superseded = registry.supersede_duplicates(
        "moderate", "META", "BUY", "technical_breakout"
    )

    assert superseded == [same_key]
    assert state_of(registry, same_key) is OrderState.CANCELED
    assert state_of(registry, other_symbol) is OrderState.PENDING
    assert state_of(registry, other_side) is OrderState.PENDING


def test_superseded_orders_carry_the_superseded_reason(registry):
    order_id = create(registry)
    registry.supersede_duplicates("moderate", "META", "BUY", "technical_breakout")

    assert registry.get_order(order_id).terminal_reason == "superseded"


def test_supersede_then_create_satisfies_the_unique_index(registry):
    """The real creation sequence: supersede, then insert."""
    first = create(registry)
    registry.supersede_duplicates("moderate", "META", "BUY", "technical_breakout")
    second = create(registry)

    assert state_of(registry, first) is OrderState.CANCELED
    assert state_of(registry, second) is OrderState.PENDING


def test_creating_a_duplicate_active_order_fails_closed(registry):
    """The partial UNIQUE index enforces Requirement 7.4 at the storage layer."""
    create(registry)

    with pytest.raises(PendingOrderRegistryError):
        create(registry)


def test_supersede_on_an_empty_key_is_a_noop(registry):
    assert registry.supersede_duplicates(
        "moderate", "NOTHING", "BUY", "technical_breakout"
    ) == []


# ---------------------------------------------------------------------------
# Orphan sweep
# ---------------------------------------------------------------------------


def test_sweep_expires_pending_orders_past_their_window(registry):
    expired_id = create(registry, symbol="META", expires_at=NOW - timedelta(minutes=1))
    live_id = create(
        registry, symbol="AMD",
        created_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )

    resolved = registry.finalize_orphaned_orders()

    assert resolved[expired_id] is OrderState.EXPIRED
    assert live_id not in resolved
    assert state_of(registry, live_id) is OrderState.PENDING


def test_sweep_leaves_a_recently_claimed_filling_order_alone(registry):
    """The lease is still valid, so a live tick may own it."""
    order_id = create(
        registry,
        created_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    registry.claim_for_fill(order_id)

    registry.finalize_orphaned_orders(filling_lease_minutes=5)

    assert state_of(registry, order_id) is OrderState.FILLING


def test_sweep_releases_a_stranded_filling_order_still_in_window(registry):
    order_id = create(
        registry,
        created_at=NOW,  # far in the past -> lease expired
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    registry.claim_for_fill(order_id)

    registry.finalize_orphaned_orders(filling_lease_minutes=5)

    assert state_of(registry, order_id) is OrderState.PENDING


def test_sweep_expires_a_stranded_filling_order_past_its_window(registry):
    order_id = create(
        registry, created_at=NOW, expires_at=NOW + timedelta(minutes=1),
    )
    registry.claim_for_fill(order_id)

    resolved = registry.finalize_orphaned_orders(filling_lease_minutes=5)

    assert resolved[order_id] is OrderState.EXPIRED
    assert state_of(registry, order_id) is OrderState.EXPIRED


def test_sweep_marks_filled_when_a_trade_already_exists(registry):
    """Crash between execute_trade() success and mark_filled().

    Fabricating a fill would be wrong; failing to record a real one would be
    worse, because the position exists either way.
    """
    order_id = create(registry, created_at=NOW, expires_at=NOW + timedelta(hours=1))
    registry.claim_for_fill(order_id)

    with patch.object(registry, "_find_trade_for_order", return_value=999):
        resolved = registry.finalize_orphaned_orders(filling_lease_minutes=5)

    assert resolved[order_id] is OrderState.FILLED
    loaded = registry.get_order(order_id)
    assert loaded.state is OrderState.FILLED
    assert loaded.trade_id == 999


def test_sweep_is_idempotent(registry):
    order_id = create(registry, expires_at=NOW - timedelta(minutes=1))

    first = registry.finalize_orphaned_orders()
    second = registry.finalize_orphaned_orders()

    assert first[order_id] is OrderState.EXPIRED
    assert second == {}


def test_sweep_leaves_no_transient_state_behind(registry):
    """Requirement 9.12 — nothing survives a restart in a transient state."""
    a = create(registry, symbol="META", expires_at=NOW - timedelta(minutes=1))
    b = create(registry, symbol="AMD", expires_at=NOW - timedelta(minutes=1))
    registry.claim_for_fill(b)

    registry.finalize_orphaned_orders(filling_lease_minutes=0)

    for order_id in (a, b):
        assert state_of(registry, order_id) in TERMINAL_STATES


def test_sweep_survives_a_failure_on_one_order(registry):
    """One unresolvable order must not abort the whole sweep."""
    good = create(registry, symbol="META", expires_at=NOW - timedelta(minutes=1))
    bad = create(registry, symbol="AMD", expires_at=NOW - timedelta(minutes=1))

    real_mark_expired = registry.mark_expired

    def selective(order_id, reason="window_elapsed"):
        if order_id == bad:
            raise PendingOrderRegistryError("simulated failure")
        return real_mark_expired(order_id, reason)

    with patch.object(registry, "mark_expired", side_effect=selective):
        resolved = registry.finalize_orphaned_orders()

    assert resolved.get(good) is OrderState.EXPIRED
    assert bad not in resolved


def test_find_trade_for_order_prefers_the_recorded_trade_id(registry):
    order_id = create(registry)
    registry.claim_for_fill(order_id)
    registry.mark_filled(
        order_id, fill_price=593.87, fill_policy="limit_price",
        fill_bar_ts=NOW, trade_id=123,
    )

    order = registry.get_order(order_id)
    assert registry._find_trade_for_order(order) == 123


def test_find_trade_for_order_is_fail_open_on_a_missing_table(registry):
    """trade_events does not exist in this minimal fixture engine."""
    order_id = create(registry)
    order = registry.get_order(order_id)
    assert registry._find_trade_for_order(order) is None
