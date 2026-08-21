"""Tests for the fast-path trigger registry.

Validates:
- Registration creates a row with state=active
- Duplicate registration (same symbol+direction+profile+setup_type) is rejected
- mark_fired transitions active -> fired with event_id linkage
- expire_stale_triggers sweeps past-expiry triggers
- Incomplete geometry signals are skipped
- register_triggers_from_signals skips ineligible setup types
- register_triggers_from_signals registers valid signals

Requirements: 2.1, 2.11
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text

from db.schema import init_fast_path_triggers_schema
from utils.fast_path_registry import (
    FastPathRegistry,
    FastPathRegistryError,
    TriggerRecord,
    register_triggers_from_signals,
)


NOW = datetime(2026, 8, 20, 14, 30, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    init_fast_path_triggers_schema(eng)
    return eng


def _make_trigger(**overrides) -> TriggerRecord:
    """Build a minimally valid TriggerRecord with sensible defaults."""
    defaults = {
        "trigger_id": str(uuid.uuid4()),
        "symbol": "TSLA",
        "profile_id": "moderate",
        "direction": "SHORT",
        "setup_type": "momentum_fade",
        "trigger_type": "entry_zone",
        "trigger_level": 351.61,
        "trigger_zone_upper": 352.00,
        "trigger_zone_lower": 350.50,
        "entry_price": 351.61,
        "stop_price": 355.00,
        "target_price": 348.00,
        "geometry_name": "momentum_fade_short",
        "source_signal_id": "sig-001",
        "source_watch_id": None,
        "invalidation_basis": "close above 355",
        "target_basis": "prior support",
        "state": "active",
        "registered_at": _iso(NOW),
        "expires_at": _iso(NOW + timedelta(seconds=300)),
        "signal_snapshot_json": '{"setup_type":"momentum_fade"}',
        "context_json": None,
    }
    defaults.update(overrides)
    return TriggerRecord(**defaults)


# ---------------------------------------------------------------------------
# Test 1: Registration creates a row with state=active
# ---------------------------------------------------------------------------


def test_register_creates_active_row(engine):
    """Registering a trigger creates a DB row with state='active'."""
    registry = FastPathRegistry(db=engine, profile_id="moderate")
    trigger = _make_trigger()

    registry.register_trigger(trigger)

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT state, symbol, direction FROM fast_path_triggers WHERE trigger_id = :tid"),
            {"tid": trigger.trigger_id},
        ).fetchone()

    assert row is not None
    assert row[0] == "active"
    assert row[1] == "TSLA"
    assert row[2] == "SHORT"


# ---------------------------------------------------------------------------
# Test 2: Duplicate registration rejected
# ---------------------------------------------------------------------------


def test_duplicate_registration_rejected(engine):
    """Second registration with same symbol+direction+profile+setup_type is silently rejected."""
    registry = FastPathRegistry(db=engine, profile_id="moderate")

    trigger1 = _make_trigger(trigger_id=str(uuid.uuid4()))
    trigger2 = _make_trigger(trigger_id=str(uuid.uuid4()))

    registry.register_trigger(trigger1)
    # Second registration: same identity fields, different trigger_id
    registry.register_trigger(trigger2)

    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM fast_path_triggers WHERE state = 'active'")
        ).scalar()

    assert count == 1


# ---------------------------------------------------------------------------
# Test 3: mark_fired transitions state
# ---------------------------------------------------------------------------


def test_mark_fired_transitions_state(engine):
    """mark_fired transitions active -> fired with fired_at, resolved_at, and resolution_event_id."""
    registry = FastPathRegistry(db=engine, profile_id="moderate")
    trigger = _make_trigger()
    registry.register_trigger(trigger)

    event_id = str(uuid.uuid4())
    registry.mark_fired(trigger.trigger_id, event_id)

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT state, fired_at, resolved_at, resolution_event_id "
                "FROM fast_path_triggers WHERE trigger_id = :tid"
            ),
            {"tid": trigger.trigger_id},
        ).fetchone()

    assert row[0] == "fired"
    assert row[1] is not None  # fired_at set
    assert row[2] is not None  # resolved_at set
    assert row[3] == event_id


# ---------------------------------------------------------------------------
# Test 4: mark_fired CAS fails on non-active
# ---------------------------------------------------------------------------


def test_mark_fired_cas_fails_on_non_active(engine):
    """mark_fired on an already-fired trigger raises FastPathRegistryError."""
    registry = FastPathRegistry(db=engine, profile_id="moderate")
    trigger = _make_trigger()
    registry.register_trigger(trigger)

    event_id_1 = str(uuid.uuid4())
    registry.mark_fired(trigger.trigger_id, event_id_1)

    # Second mark_fired should fail — state is now 'fired', not 'active'
    event_id_2 = str(uuid.uuid4())
    with pytest.raises(FastPathRegistryError):
        registry.mark_fired(trigger.trigger_id, event_id_2)


# ---------------------------------------------------------------------------
# Test 5: expire_stale_triggers sweeps past-expiry triggers
# ---------------------------------------------------------------------------


def test_expire_stale_triggers(engine):
    """expire_stale_triggers transitions active triggers with past expires_at to expired."""
    registry = FastPathRegistry(db=engine, profile_id="moderate")

    # Register a trigger that already expired (expires_at in the past)
    past_expires = _iso(NOW - timedelta(seconds=60))
    trigger = _make_trigger(expires_at=past_expires, registered_at=_iso(NOW - timedelta(seconds=360)))
    registry.register_trigger(trigger)

    expired_count = registry.expire_stale_triggers()

    assert expired_count == 1

    with engine.connect() as conn:
        state = conn.execute(
            text("SELECT state FROM fast_path_triggers WHERE trigger_id = :tid"),
            {"tid": trigger.trigger_id},
        ).scalar()

    assert state == "expired"


# ---------------------------------------------------------------------------
# Test 6: get_active_triggers excludes expired
# ---------------------------------------------------------------------------


def test_get_active_triggers_excludes_expired(engine):
    """get_active_triggers returns empty when trigger has past expires_at."""
    registry = FastPathRegistry(db=engine, profile_id="moderate")

    # Register a trigger with expires_at in the past
    past_expires = _iso(NOW - timedelta(seconds=60))
    trigger = _make_trigger(expires_at=past_expires, registered_at=_iso(NOW - timedelta(seconds=360)))
    registry.register_trigger(trigger)

    active = registry.get_active_triggers()
    assert len(active) == 0


# ---------------------------------------------------------------------------
# Test 7: register_triggers_from_signals skips incomplete geometry
# ---------------------------------------------------------------------------


def test_register_triggers_from_signals_skips_incomplete_geometry(engine):
    """Signals missing stop_price are not registered."""
    signals = {
        "TSLA": {
            "setup_type": "momentum_fade",
            "direction": "SHORT",
            "entry_price": 351.61,
            "stop_price": None,  # missing
            "target_price": 348.00,
            "signal_id": "sig-001",
        }
    }

    registered = register_triggers_from_signals(signals, "moderate", engine)
    assert len(registered) == 0

    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM fast_path_triggers")
        ).scalar()

    assert count == 0


# ---------------------------------------------------------------------------
# Test 8: register_triggers_from_signals skips ineligible setup type
# ---------------------------------------------------------------------------


def test_register_triggers_from_signals_skips_ineligible_setup_type(engine):
    """Signals with setup_type not in FAST_PATH_ELIGIBLE_SETUP_TYPES are skipped."""
    signals = {
        "AAPL": {
            "setup_type": "unusual_options_activity",
            "direction": "BUY",
            "entry_price": 180.00,
            "stop_price": 175.00,
            "target_price": 190.00,
            "signal_id": "sig-002",
        }
    }

    registered = register_triggers_from_signals(signals, "moderate", engine)
    assert len(registered) == 0

    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM fast_path_triggers")
        ).scalar()

    assert count == 0


# ---------------------------------------------------------------------------
# Test 9: register_triggers_from_signals registers valid signal
# ---------------------------------------------------------------------------


def test_register_triggers_from_signals_registers_valid(engine):
    """A valid signal with eligible setup_type and complete geometry is registered."""
    signals = {
        "TSLA": {
            "setup_type": "momentum_fade",
            "direction": "SHORT",
            "entry_price": 351.61,
            "stop_price": 355.00,
            "target_price": 348.00,
            "signal_id": "sig-003",
        }
    }

    registered = register_triggers_from_signals(signals, "moderate", engine)
    assert len(registered) == 1

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT state, symbol, direction, setup_type, entry_price, stop_price, target_price "
                "FROM fast_path_triggers WHERE trigger_id = :tid"
            ),
            {"tid": registered[0]},
        ).fetchone()

    assert row is not None
    assert row[0] == "active"
    assert row[1] == "TSLA"
    assert row[2] == "SHORT"
    assert row[3] == "momentum_fade"
    assert row[4] == 351.61
    assert row[5] == 355.00
    assert row[6] == 348.00
