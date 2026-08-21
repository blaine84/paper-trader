"""Tests for the fast-path cooldown system.

Validates:
- missed_move does not block subsequent trigger
- trade_executed blocks within cooldown window
- Active pending order blocks same-symbol trigger
- Churn protection fires on 3+ stand_downs
- stand_down alone does not block
- Cooldown DB error on execution-path outcome → returns CooldownBlock (fail-closed)
- Cooldown DB error on watch-only outcome → returns None (fail-open)

Requirements: 6.1-6.8, cross-cutting acceptance test 5
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, text

from db.schema import Base, init_fast_path_events_schema, init_pending_order_schema
from utils.fast_path_cooldown import CooldownBlock, check_fast_path_cooldown


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    """Return the real current UTC time so tests are relative to system clock."""
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _insert_fast_path_event(conn, *, symbol="TSLA", profile_id="moderate",
                            setup_type="momentum_fade", outcome_type="stand_down",
                            evaluated_at=None, event_id=None):
    """Insert a minimal fast_path_events row for testing."""
    if evaluated_at is None:
        evaluated_at = _iso(_now())
    if event_id is None:
        event_id = str(uuid.uuid4())
    conn.execute(
        text(
            "INSERT INTO fast_path_events "
            "(event_id, trigger_id, symbol, profile_id, setup_type, direction, "
            "outcome_type, outcome_reason_code, current_price, evaluated_at, "
            "annotation_status) "
            "VALUES (:event_id, :trigger_id, :symbol, :profile_id, :setup_type, "
            "'SHORT', :outcome_type, :reason, 350.0, :evaluated_at, "
            "'annotation_pending')"
        ),
        {
            "event_id": event_id,
            "trigger_id": str(uuid.uuid4()),
            "symbol": symbol,
            "profile_id": profile_id,
            "setup_type": setup_type,
            "outcome_type": outcome_type,
            "reason": f"test_{outcome_type}",
            "evaluated_at": evaluated_at,
        },
    )


def _insert_trade(conn, *, symbol="TSLA", profile="moderate", entry_time=None):
    """Insert a minimal trade row."""
    if entry_time is None:
        entry_time = _iso(_now())
    conn.execute(
        text(
            "INSERT INTO trades (symbol, profile, direction, quantity, "
            "entry_price, entry_time, status) "
            "VALUES (:symbol, :profile, 'SHORT', 100, 351.0, :entry_time, 'open')"
        ),
        {"symbol": symbol, "profile": profile, "entry_time": entry_time},
    )


def _insert_pending_order(conn, *, symbol="TSLA", profile_id="moderate",
                          state="pending", order_id=None):
    """Insert a minimal pending_orders row."""
    if order_id is None:
        order_id = str(uuid.uuid4())
    now = _now()
    conn.execute(
        text(
            "INSERT INTO pending_orders "
            "(order_id, profile_id, symbol, side, setup_type, limit_price, "
            "stop_price, target_price, risk_reward, fresh_price_at_creation, "
            "runaway_pct_at_creation, state, created_at, expires_at, integrity_hash) "
            "VALUES (:order_id, :profile_id, :symbol, 'SHORT', 'momentum_fade', "
            "351.0, 355.0, 348.0, 2.1, 350.0, 0.003, :state, :created_at, "
            ":expires_at, 'hash123')"
        ),
        {
            "order_id": order_id,
            "profile_id": profile_id,
            "symbol": symbol,
            "state": state,
            "created_at": _iso(now),
            "expires_at": _iso(now + timedelta(hours=4)),
        },
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """In-memory SQLite with all tables needed for cooldown checks."""
    eng = create_engine("sqlite:///:memory:")
    # Create ORM-based tables (trades, etc.)
    Base.metadata.create_all(eng)
    # Create raw-DDL tables
    init_fast_path_events_schema(eng)
    init_pending_order_schema(eng)

    # Drop the immutability triggers so tests can insert freely without
    # being blocked on UPDATE (the cooldown module only reads, so this is safe).
    # Actually, the immutability triggers only block UPDATE/DELETE, not INSERT,
    # so we leave them. Tests only INSERT.
    return eng


# ---------------------------------------------------------------------------
# Test 1: missed_move does NOT block subsequent trigger
# (Requirement 6.2 — cross-cutting acceptance test 5)
# ---------------------------------------------------------------------------


def test_missed_move_does_not_block(engine):
    """A prior missed_move event should not suppress a fresh trigger for the same symbol."""
    with engine.begin() as conn:
        _insert_fast_path_event(
            conn,
            symbol="TSLA",
            profile_id="moderate",
            outcome_type="missed_move",
            evaluated_at=_iso(_now() - timedelta(minutes=2)),
        )

    result = check_fast_path_cooldown(
        symbol="TSLA",
        setup_type="momentum_fade",
        profile_id="moderate",
        db=engine,
        execution_path=True,
    )

    assert result is None


# ---------------------------------------------------------------------------
# Test 2: trade_executed blocks within cooldown window
# (Requirement 6.1)
# ---------------------------------------------------------------------------


def test_trade_executed_blocks_within_cooldown(engine):
    """A recent trade_executed event blocks new triggers for the same symbol."""
    with engine.begin() as conn:
        _insert_fast_path_event(
            conn,
            symbol="TSLA",
            profile_id="moderate",
            outcome_type="trade_executed",
            evaluated_at=_iso(_now() - timedelta(minutes=5)),
        )

    result = check_fast_path_cooldown(
        symbol="TSLA",
        setup_type="momentum_fade",
        profile_id="moderate",
        db=engine,
        execution_path=True,
    )

    assert result is not None
    assert isinstance(result, CooldownBlock)
    assert result.reason_code == "recent_trade"
    assert result.blocking_outcome_type == "trade_executed"


# ---------------------------------------------------------------------------
# Test 3: Active pending order blocks same-symbol trigger
# (Requirement 6.1)
# ---------------------------------------------------------------------------


def test_active_pending_order_blocks(engine):
    """An active pending order for the same symbol blocks new triggers."""
    with engine.begin() as conn:
        _insert_pending_order(conn, symbol="TSLA", profile_id="moderate", state="pending")

    result = check_fast_path_cooldown(
        symbol="TSLA",
        setup_type="momentum_fade",
        profile_id="moderate",
        db=engine,
        execution_path=True,
    )

    assert result is not None
    assert isinstance(result, CooldownBlock)
    assert result.reason_code == "active_pending_order"
    assert result.blocking_outcome_type == "pending_order_created"


# ---------------------------------------------------------------------------
# Test 4: Churn protection fires on 3+ stand_downs
# (Requirement 6.4)
# ---------------------------------------------------------------------------


def test_churn_protection_fires_on_3_stand_downs(engine):
    """3+ stand_down events for same symbol+setup_type within churn window triggers block."""
    now = _now()
    with engine.begin() as conn:
        for i in range(3):
            _insert_fast_path_event(
                conn,
                symbol="TSLA",
                profile_id="moderate",
                setup_type="momentum_fade",
                outcome_type="stand_down",
                evaluated_at=_iso(now - timedelta(minutes=i + 1)),
            )

    result = check_fast_path_cooldown(
        symbol="TSLA",
        setup_type="momentum_fade",
        profile_id="moderate",
        db=engine,
        execution_path=True,
    )

    assert result is not None
    assert isinstance(result, CooldownBlock)
    assert result.reason_code == "churn_protection"


# ---------------------------------------------------------------------------
# Test 5: stand_down alone does NOT block
# (Requirement 6.1 — stand_down does not block unless churn protection triggers)
# ---------------------------------------------------------------------------


def test_stand_down_alone_does_not_block(engine):
    """A single stand_down event does not block subsequent triggers."""
    with engine.begin() as conn:
        _insert_fast_path_event(
            conn,
            symbol="TSLA",
            profile_id="moderate",
            setup_type="momentum_fade",
            outcome_type="stand_down",
            evaluated_at=_iso(_now() - timedelta(minutes=2)),
        )

    result = check_fast_path_cooldown(
        symbol="TSLA",
        setup_type="momentum_fade",
        profile_id="moderate",
        db=engine,
        execution_path=True,
    )

    assert result is None


# ---------------------------------------------------------------------------
# Test 6: Cooldown DB error on execution-path → returns CooldownBlock (fail-closed)
# (Requirement 6.7)
# ---------------------------------------------------------------------------


def test_db_error_execution_path_returns_cooldown_block():
    """When DB raises on execution path, check returns CooldownBlock (fail-closed)."""
    mock_engine = MagicMock()
    mock_engine.connect.side_effect = RuntimeError("DB unavailable")

    result = check_fast_path_cooldown(
        symbol="TSLA",
        setup_type="momentum_fade",
        profile_id="moderate",
        db=mock_engine,
        execution_path=True,
    )

    assert result is not None
    assert isinstance(result, CooldownBlock)
    assert result.reason_code == "cooldown_check_failed"


# ---------------------------------------------------------------------------
# Test 7: Cooldown DB error on watch-only outcome → returns None (fail-open)
# (Requirement 6.8)
# ---------------------------------------------------------------------------


def test_db_error_watch_path_returns_none():
    """When DB raises on non-execution path, check returns None (fail-open)."""
    mock_engine = MagicMock()
    mock_engine.connect.side_effect = RuntimeError("DB unavailable")

    result = check_fast_path_cooldown(
        symbol="TSLA",
        setup_type="momentum_fade",
        profile_id="moderate",
        db=mock_engine,
        execution_path=False,
    )

    assert result is None
