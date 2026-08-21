"""End-to-end integration tests for the fast-path deterministic execution layer.

Exercises the full stack — trigger registry → monitor tick → outcome
evaluator → event persistence — against an in-memory SQLite database, wiring
the real modules together rather than mocking their interactions.

Covers cross-cutting acceptance tests:
    12.1 Observe-mode missed_move (target already crossed) — no execution_failed
    12.2 Enabled-mode trade_executed — gate pipeline invoked, delegation happens
    12.3 Cooldown isolation — a prior missed_move does NOT block a fresh trigger
    12.4 Annotation failure isolation — annotation errors never touch the outcome
    12.5 Performance — 20 triggers across 10 symbols evaluated in one tick < 5s

Setup notes:
    - In-memory SQLite via create_engine("sqlite:///:memory:")
    - init_fast_path_triggers_schema + init_fast_path_events_schema
    - FastPathRegistry to register triggers, FastPathMonitor to run ticks
    - _fetch_quotes is patched to return controlled quotes
    - FAST_PATH_MODE is patched per test for observe/enabled behaviour
    - Timestamps use datetime.now(timezone.utc) so trigger expiry stays ahead
      of the registry's `expires_at > now` ISO-string filter

Requirements: 6.2, 6.3, 7.3, 7.5, 10.2, 10.3, 2.8,
              cross-cutting acceptance tests 1, 2, 4, 5, 8
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text

from db.schema import (
    Base,
    init_fast_path_events_schema,
    init_fast_path_triggers_schema,
    init_pending_order_schema,
)
from utils.fast_path_annotation import annotate_event, get_unannotated_events
from utils.fast_path_cooldown import check_fast_path_cooldown
from utils.fast_path_monitor import FastPathMonitor
from utils.fast_path_registry import FastPathRegistry, TriggerRecord


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """In-memory SQLite engine with the full fast-path + cooldown schema.

    The cooldown check queries the ``trades`` and ``pending_orders`` tables,
    so those are created here too (via the ORM metadata and the pending-order
    DDL) to let the cooldown path run against real, empty tables rather than
    failing closed on a missing-table error.
    """
    eng = create_engine("sqlite:///:memory:")
    # ORM-backed tables (trades, etc.) needed by the cooldown check.
    Base.metadata.create_all(eng)
    # Raw-DDL fast-path and pending-order tables.
    init_fast_path_triggers_schema(eng)
    init_fast_path_events_schema(eng)
    init_pending_order_schema(eng)
    return eng


def _future_expiry(seconds: int = 300) -> tuple[str, str]:
    """Return (registered_at, expires_at) ISO strings with expiry in the future."""
    now = datetime.now(timezone.utc)
    return now.isoformat(), (now + timedelta(seconds=seconds)).isoformat()


def _make_trigger(
    *,
    symbol: str,
    profile_id: str = "moderate",
    direction: str = "SHORT",
    setup_type: str = "momentum_fade",
    trigger_type: str = "entry_zone",
    trigger_level: float = 351.0,
    trigger_zone_lower: float | None = None,
    trigger_zone_upper: float | None = None,
    entry_price: float = 351.61,
    stop_price: float = 355.00,
    target_price: float = 348.97,
    source_signal_id: str | None = "signal-abc",
    source_watch_id: str | None = None,
    trigger_id: str | None = None,
) -> TriggerRecord:
    """Build a TriggerRecord with future expiry and sensible SHORT geometry."""
    registered_at, expires_at = _future_expiry()
    return TriggerRecord(
        trigger_id=trigger_id or str(uuid.uuid4()),
        symbol=symbol,
        profile_id=profile_id,
        direction=direction,
        setup_type=setup_type,
        trigger_type=trigger_type,
        trigger_level=trigger_level,
        trigger_zone_lower=trigger_zone_lower,
        trigger_zone_upper=trigger_zone_upper,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        geometry_name="test_geometry",
        source_signal_id=source_signal_id,
        source_watch_id=source_watch_id,
        invalidation_basis=None,
        target_basis=None,
        state="active",
        registered_at=registered_at,
        expires_at=expires_at,
        signal_snapshot_json=None,
        context_json=None,
    )


def _quote(price: float, age_ms: int = 50, reliable: bool = True) -> dict:
    return {"price": price, "age_ms": age_ms, "reliable": reliable}


def _fetch_events(engine, symbol: str) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT * FROM fast_path_events WHERE symbol = :s"),
            {"s": symbol},
        ).mappings().all()
    return [dict(r) for r in rows]


def _insert_missed_move_event(engine, symbol: str, profile_id: str, setup_type: str) -> str:
    """Insert a missed_move event directly into fast_path_events."""
    event_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    with engine.connect() as conn:
        conn.execute(
            text(
                """
                INSERT INTO fast_path_events (
                    event_id, trigger_id, symbol, profile_id, setup_type,
                    direction, current_price, outcome_type, outcome_reason_code,
                    annotation_status, narration_source, evaluated_at, created_at
                ) VALUES (
                    :event_id, :trigger_id, :symbol, :profile_id, :setup_type,
                    :direction, :current_price, :outcome_type, :outcome_reason_code,
                    :annotation_status, :narration_source, :evaluated_at, :created_at
                )
                """
            ),
            {
                "event_id": event_id,
                "trigger_id": str(uuid.uuid4()),
                "symbol": symbol,
                "profile_id": profile_id,
                "setup_type": setup_type,
                "direction": "SHORT",
                "current_price": 342.08,
                "outcome_type": "missed_move",
                "outcome_reason_code": "target_already_crossed",
                "annotation_status": "annotation_pending",
                "narration_source": "template",
                "evaluated_at": now_iso,
                "created_at": now_iso,
            },
        )
        conn.commit()
    return event_id


# ---------------------------------------------------------------------------
# 12.1 — End-to-end observe mode test (missed_move, not execution_failed)
# ---------------------------------------------------------------------------


class TestObserveModeMissedMove:
    """Cross-cutting acceptance test 1: target-crossed stale entry → missed_move."""

    def test_target_already_crossed_produces_missed_move_event(self, engine):
        """A SHORT trigger whose target is already crossed produces a
        missed_move event (NOT execution_failed) in observe mode."""
        registry = FastPathRegistry(db=engine, profile_id="moderate")
        # SHORT geometry: target(348.97) < entry(351.61) < stop(355.00)
        trigger = _make_trigger(
            symbol="TSLA",
            direction="SHORT",
            entry_price=351.61,
            stop_price=355.00,
            target_price=348.97,
        )
        registry.register_trigger(trigger)

        monitor = FastPathMonitor(engine, ["moderate"])

        # Current price 342.08 is below the SHORT target 348.97 → target crossed.
        delegate_calls = []

        with patch(
            "utils.fast_path_monitor._fetch_quotes",
            return_value={"TSLA": _quote(342.08)},
        ), patch(
            "utils.fast_path_monitor._delegate_execution",
            side_effect=lambda o, t, e: delegate_calls.append(o.trigger_id),
        ), patch(
            "utils.fast_path_monitor.FAST_PATH_MODE", "observe"
        ):
            summary = monitor.run_tick()

        assert summary["fired"] == 1

        events = _fetch_events(engine, "TSLA")
        assert len(events) == 1
        event = events[0]
        assert event["outcome_type"] == "missed_move"
        assert event["outcome_reason_code"] == "target_already_crossed"
        # Critically: it is NOT an execution failure
        assert event["outcome_type"] != "execution_failed"

        # Observe mode: no execution delegation occurs
        assert delegate_calls == []

    def test_missed_move_marks_trigger_fired_and_records_outcome(self, engine):
        """The trigger transitions to fired and links to the resolution event."""
        registry = FastPathRegistry(db=engine, profile_id="moderate")
        trigger = _make_trigger(symbol="TSLA", trigger_id="trig-missed-001")
        registry.register_trigger(trigger)

        monitor = FastPathMonitor(engine, ["moderate"])

        with patch(
            "utils.fast_path_monitor._fetch_quotes",
            return_value={"TSLA": _quote(342.08)},
        ), patch(
            "utils.fast_path_monitor.FAST_PATH_MODE", "observe"
        ):
            monitor.run_tick()

        # Trigger should be terminal (fired), no longer active.
        assert registry.get_active_triggers() == []

        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT state, resolution_event_id FROM fast_path_triggers "
                    "WHERE trigger_id = :tid"
                ),
                {"tid": "trig-missed-001"},
            ).mappings().first()
        assert row["state"] == "fired"
        assert row["resolution_event_id"] is not None

        # The linked event is the missed_move outcome (PM/annotation can process it).
        events = _fetch_events(engine, "TSLA")
        assert events[0]["event_id"] == row["resolution_event_id"]
        assert events[0]["outcome_type"] == "missed_move"


# ---------------------------------------------------------------------------
# 12.2 — End-to-end enabled mode test (trade_executed, gates + delegation)
# ---------------------------------------------------------------------------


class TestEnabledModeTradeExecuted:
    """Cross-cutting acceptance test 2: valid geometry + price in zone → trade_executed."""

    def test_valid_geometry_in_zone_produces_trade_executed(self, engine):
        """A BUY trigger with price inside the entry zone and passing gates
        produces a trade_executed event and delegates execution."""
        registry = FastPathRegistry(db=engine, profile_id="moderate")
        # BUY geometry: stop(98.0) < entry(100.0) < target(110.0)
        # entry_zone spans 99.5..100.5; current price 100.0 sits inside.
        trigger = _make_trigger(
            symbol="AAPL",
            direction="BUY",
            setup_type="momentum_fade",  # in FAST_PATH_ENABLED_SETUP_TYPES
            trigger_type="entry_zone",
            trigger_level=100.0,
            trigger_zone_lower=99.5,
            trigger_zone_upper=100.5,
            entry_price=100.0,
            stop_price=98.0,
            target_price=110.0,
            trigger_id="trig-exec-001",
        )
        registry.register_trigger(trigger)

        monitor = FastPathMonitor(engine, ["moderate"])

        delegate_calls = []
        gate_calls = []

        def track_gates(trig, quote, profile_state):
            gate_calls.append(trig.trigger_id)
            return (True, None)

        with patch(
            "utils.fast_path_monitor._fetch_quotes",
            return_value={"AAPL": _quote(100.0)},
        ), patch(
            "utils.fast_path_monitor._delegate_execution",
            side_effect=lambda o, t, e: delegate_calls.append((o.trigger_id, o.outcome_type)),
        ), patch(
            "utils.fast_path_evaluator._run_gates", side_effect=track_gates
        ), patch(
            "utils.fast_path_monitor.FAST_PATH_MODE", "enabled"
        ):
            summary = monitor.run_tick()

        assert summary["fired"] == 1

        events = _fetch_events(engine, "AAPL")
        assert len(events) == 1
        assert events[0]["outcome_type"] == "trade_executed"

        # Gate pipeline was invoked for this trigger
        assert gate_calls == ["trig-exec-001"]

        # Execution was delegated with the correct outcome type
        assert delegate_calls == [("trig-exec-001", "trade_executed")]

    def test_gate_rejection_stands_down_without_delegation(self, engine):
        """When the gate pipeline rejects, the outcome is stand_down and no
        execution is delegated even in enabled mode."""
        registry = FastPathRegistry(db=engine, profile_id="moderate")
        trigger = _make_trigger(
            symbol="AAPL",
            direction="BUY",
            setup_type="momentum_fade",
            trigger_type="entry_zone",
            trigger_level=100.0,
            trigger_zone_lower=99.5,
            trigger_zone_upper=100.5,
            entry_price=100.0,
            stop_price=98.0,
            target_price=110.0,
            trigger_id="trig-exec-002",
        )
        registry.register_trigger(trigger)

        monitor = FastPathMonitor(engine, ["moderate"])
        delegate_calls = []

        with patch(
            "utils.fast_path_monitor._fetch_quotes",
            return_value={"AAPL": _quote(100.0)},
        ), patch(
            "utils.fast_path_monitor._delegate_execution",
            side_effect=lambda o, t, e: delegate_calls.append(o.trigger_id),
        ), patch(
            "utils.fast_path_evaluator._run_gates",
            return_value=(False, "risk_geometry"),
        ), patch(
            "utils.fast_path_monitor.FAST_PATH_MODE", "enabled"
        ):
            monitor.run_tick()

        events = _fetch_events(engine, "AAPL")
        assert len(events) == 1
        assert events[0]["outcome_type"] == "stand_down"
        assert events[0]["outcome_reason_code"] == "gate_rejected:risk_geometry"
        # No delegation on a stand_down
        assert delegate_calls == []


# ---------------------------------------------------------------------------
# 12.3 — Cooldown isolation test
# ---------------------------------------------------------------------------


class TestCooldownIsolation:
    """Cross-cutting acceptance test 5: a missed_move must not block a fresh trigger."""

    def test_missed_move_does_not_block_subsequent_trigger(self, engine):
        """After a missed_move for TSLA, check_fast_path_cooldown returns None
        (the missed move does not suppress a later fresh same-symbol trigger)."""
        # Produce a missed_move for TSLA.
        _insert_missed_move_event(
            engine, symbol="TSLA", profile_id="moderate", setup_type="momentum_fade"
        )

        # A new signal for TSLA with different geometry arrives — cooldown check
        # must allow it (missed_move never blocks).
        block = check_fast_path_cooldown(
            symbol="TSLA",
            setup_type="momentum_fade",
            profile_id="moderate",
            db=engine,
            execution_path=True,
        )
        assert block is None

    def test_second_trigger_fires_normally_after_prior_missed_move(self, engine):
        """End-to-end: a prior missed_move event does not prevent a fresh
        trigger for the same symbol from being evaluated and firing."""
        _insert_missed_move_event(
            engine, symbol="TSLA", profile_id="moderate", setup_type="momentum_fade"
        )

        registry = FastPathRegistry(db=engine, profile_id="moderate")
        # New trigger with different geometry, price still below its target
        # so it too resolves as missed_move — the point is it is NOT suppressed.
        trigger = _make_trigger(
            symbol="TSLA",
            entry_price=345.00,
            stop_price=348.00,
            target_price=340.00,
            trigger_id="trig-second-001",
        )
        registry.register_trigger(trigger)

        monitor = FastPathMonitor(engine, ["moderate"])
        with patch(
            "utils.fast_path_monitor._fetch_quotes",
            return_value={"TSLA": _quote(338.0)},
        ), patch(
            "utils.fast_path_monitor.FAST_PATH_MODE", "observe"
        ):
            summary = monitor.run_tick()

        # The second trigger fired (was evaluated, not cooldown-suppressed).
        assert summary["fired"] == 1
        events = _fetch_events(engine, "TSLA")
        # Two events now: the seeded one + the newly fired trigger.
        assert len(events) == 2
        trigger_ids = {e["trigger_id"] for e in events}
        assert "trig-second-001" in trigger_ids


# ---------------------------------------------------------------------------
# 12.4 — Annotation failure isolation test
# ---------------------------------------------------------------------------


class TestAnnotationFailureIsolation:
    """Cross-cutting acceptance test 4: annotation errors never touch outcomes."""

    def test_annotation_exception_does_not_modify_event_outcome(self, engine):
        """If PM annotation raises, the fast-path event outcome and geometry
        remain unchanged and no execution state is modified."""
        # Seed a fast-path event by running a tick.
        registry = FastPathRegistry(db=engine, profile_id="moderate")
        trigger = _make_trigger(symbol="TSLA", trigger_id="trig-annot-001")
        registry.register_trigger(trigger)

        monitor = FastPathMonitor(engine, ["moderate"])
        with patch(
            "utils.fast_path_monitor._fetch_quotes",
            return_value={"TSLA": _quote(342.08)},
        ), patch(
            "utils.fast_path_monitor.FAST_PATH_MODE", "observe"
        ):
            monitor.run_tick()

        events_before = _fetch_events(engine, "TSLA")
        assert len(events_before) == 1
        event = events_before[0]
        event_id = event["event_id"]
        assert event["annotation_status"] == "annotation_pending"

        # The event is available for annotation.
        pending = get_unannotated_events(engine, "moderate")
        assert any(e["event_id"] == event_id for e in pending)

        # Simulate PM annotation raising inside the connection layer.
        # annotate_event is fail-open: it must swallow the error and leave the
        # event outcome untouched.
        with patch.object(engine, "connect", side_effect=RuntimeError("db boom")):
            # Must not raise
            annotate_event(engine, event_id, {"thesis": "should not persist"})

        # Event outcome, geometry, and annotation status are unchanged.
        events_after = _fetch_events(engine, "TSLA")
        assert len(events_after) == 1
        after = events_after[0]
        assert after["outcome_type"] == event["outcome_type"] == "missed_move"
        assert after["outcome_reason_code"] == event["outcome_reason_code"]
        assert after["current_price"] == event["current_price"]
        assert after["entry_price"] == event["entry_price"]
        assert after["stop_price"] == event["stop_price"]
        assert after["target_price"] == event["target_price"]
        # Annotation never succeeded → status stays pending (unchanged / not annotated).
        assert after["annotation_status"] == "annotation_pending"
        assert after["annotation_json"] is None

    def test_annotation_pipeline_failure_leaves_event_visible(self, engine):
        """A raising annotation callback in a PM-style loop does not remove the
        event from the queue nor block subsequent processing."""
        registry = FastPathRegistry(db=engine, profile_id="moderate")
        trigger = _make_trigger(symbol="TSLA", trigger_id="trig-annot-002")
        registry.register_trigger(trigger)

        monitor = FastPathMonitor(engine, ["moderate"])
        with patch(
            "utils.fast_path_monitor._fetch_quotes",
            return_value={"TSLA": _quote(342.08)},
        ), patch(
            "utils.fast_path_monitor.FAST_PATH_MODE", "observe"
        ):
            monitor.run_tick()

        pending = get_unannotated_events(engine, "moderate")
        assert len(pending) == 1
        event_id = pending[0]["event_id"]

        # Simulate a PM annotation loop where generating the annotation raises.
        def failing_pm_annotation(_event):
            raise ValueError("LLM annotation failed")

        annotation_error_caught = False
        try:
            for ev in pending:
                try:
                    payload = failing_pm_annotation(ev)
                    annotate_event(engine, ev["event_id"], payload)
                except Exception:
                    # Fail-open in the caller: log & continue, event stays valid.
                    annotation_error_caught = True
        except Exception:  # pragma: no cover - loop itself must not propagate
            pytest.fail("annotation failure propagated out of the PM loop")

        assert annotation_error_caught is True

        # Event still present and still pending — outcome untouched.
        still_pending = get_unannotated_events(engine, "moderate")
        assert any(e["event_id"] == event_id for e in still_pending)
        events = _fetch_events(engine, "TSLA")
        assert events[0]["outcome_type"] == "missed_move"


# ---------------------------------------------------------------------------
# 12.5 — Performance test
# ---------------------------------------------------------------------------


class TestPerformance:
    """Cross-cutting acceptance test 8: 20 triggers / 10 symbols in one tick < 5s."""

    def test_twenty_triggers_ten_symbols_single_tick_under_five_seconds(self, engine):
        """Register 20 triggers across 10 symbols and assert a single monitor
        tick completes within 5 seconds."""
        registry = FastPathRegistry(db=engine, profile_id="moderate")

        symbols = [f"SYM{i:02d}" for i in range(10)]
        # 20 triggers total: 2 per symbol, differing setup_type so dedup allows both.
        setup_types = ["momentum_fade", "gap_and_go"]
        for i, symbol in enumerate(symbols):
            for j, setup_type in enumerate(setup_types):
                trigger = _make_trigger(
                    symbol=symbol,
                    setup_type=setup_type,
                    direction="BUY",
                    trigger_type="entry_zone",
                    trigger_level=100.0,
                    trigger_zone_lower=99.5,
                    trigger_zone_upper=100.5,
                    entry_price=100.0,
                    stop_price=98.0,
                    target_price=110.0,
                    trigger_id=f"trig-{i:02d}-{j}",
                )
                registry.register_trigger(trigger)

        assert len(registry.get_active_triggers()) == 20

        monitor = FastPathMonitor(engine, ["moderate"])

        # Controlled quotes for every symbol; price inside the entry zone.
        quotes = {symbol: _quote(100.0) for symbol in symbols}

        with patch(
            "utils.fast_path_monitor._fetch_quotes", return_value=quotes
        ), patch(
            "utils.fast_path_monitor._delegate_execution"
        ), patch(
            "utils.fast_path_monitor.FAST_PATH_MODE", "observe"
        ):
            start = time.monotonic()
            summary = monitor.run_tick()
            elapsed = time.monotonic() - start

        assert summary["evaluated"] == 20
        assert elapsed < 5.0, f"tick took {elapsed:.3f}s (>= 5s budget)"
