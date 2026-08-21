"""Tests for fast-path candidate builder exclusion (Task 8.4).

Verifies that when FAST_PATH_MODE == "enabled", signals with confirmed active
triggers in fast_path_triggers are suppressed from the PM candidate set, and
that failure of the suppression query is fail-open (all signals included).

Requirements: 11.3, 11.5
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from sqlalchemy import create_engine, text

from models.pm_profiles import PM_PROFILES
from utils.candidate_builder import build_candidate_set


def _init_tables(engine):
    """Create minimal tables needed for candidate builder + fast path triggers."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE pm_candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT NOT NULL,
                    cycle_id TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    setup_type TEXT NOT NULL,
                    geometry_name TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_price REAL NOT NULL,
                    target_price REAL NOT NULL,
                    risk_reward REAL NOT NULL,
                    trigger TEXT,
                    invalidation_basis TEXT,
                    target_basis TEXT,
                    source_signal_id TEXT NOT NULL,
                    signal_snapshot_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    integrity_hash TEXT NOT NULL,
                    execution_key TEXT,
                    reserved_at TEXT,
                    created_at TEXT,
                    expires_at TEXT NOT NULL,
                    context_snapshot_json TEXT,
                    benchmark_mapping_json TEXT,
                    rejection_reason TEXT,
                    candidate_lineage_id TEXT,
                    candidate_type TEXT DEFAULT 'intraday',
                    holding_horizon INTEGER,
                    normalized_setup_type TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS fast_path_triggers (
                    trigger_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    setup_type TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    trigger_level REAL NOT NULL,
                    trigger_zone_upper REAL,
                    trigger_zone_lower REAL,
                    entry_price REAL NOT NULL,
                    stop_price REAL NOT NULL,
                    target_price REAL NOT NULL,
                    geometry_name TEXT,
                    source_signal_id TEXT,
                    source_watch_id TEXT,
                    invalidation_basis TEXT,
                    target_basis TEXT,
                    state TEXT NOT NULL DEFAULT 'active',
                    registered_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    fired_at TEXT,
                    resolution_event_id TEXT,
                    signal_snapshot_json TEXT,
                    context_json TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_fpt_state "
                "ON fast_path_triggers(state)"
            )
        )


def _insert_active_trigger(engine, symbol, direction, profile_id, setup_type="momentum_fade"):
    """Insert an active fast-path trigger for the given symbol/direction/profile."""
    import uuid

    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=5)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO fast_path_triggers
                (trigger_id, symbol, profile_id, direction, setup_type,
                 trigger_type, trigger_level, entry_price, stop_price, target_price,
                 state, registered_at, expires_at)
                VALUES
                (:tid, :symbol, :profile_id, :direction, :setup_type,
                 'entry_zone', 100.0, 100.0, 98.0, 104.0,
                 'active', :registered_at, :expires_at)
                """
            ),
            {
                "tid": str(uuid.uuid4()),
                "symbol": symbol,
                "profile_id": profile_id,
                "direction": direction,
                "setup_type": setup_type,
                "registered_at": now.isoformat(),
                "expires_at": expires.isoformat(),
            },
        )


def _fake_scaffold(signal, profile_id=None, profile_context=None):
    """Return a valid scaffold for any signal."""
    direction_raw = signal.get("signal", "BUY").upper()
    scaffold_dir = "LONG" if direction_raw in ("LONG", "BUY") else "SHORT"
    return {
        "symbol": signal["symbol"],
        "direction": scaffold_dir,
        "status": "ok",
        "candidates": [
            {
                "name": "test_geometry",
                "entry_price": 100.0,
                "stop_loss": 98.0,
                "target": 104.0,
                "risk_reward": 2.0,
                "trigger": "Price breaks above",
                "invalidation_basis": "Falls below stop",
                "target_basis": "Entry + RR * risk",
            }
        ],
    }


@patch("utils.candidate_builder.build_entry_geometry_scaffold", _fake_scaffold)
@patch("utils.gate_config.FAST_PATH_MODE", "enabled")
def test_signal_with_active_trigger_is_suppressed():
    """When FAST_PATH_MODE==enabled and active trigger exists, signal is excluded."""
    engine = create_engine("sqlite:///:memory:")
    _init_tables(engine)

    # Insert active trigger for TSLA SHORT in moderate profile
    _insert_active_trigger(engine, "TSLA", "SHORT", "moderate")

    signals = {
        "TSLA": {
            "symbol": "TSLA",
            "signal": "SHORT",
            "strength": "strong",
            "setup_type": "momentum_fade",
            "current_price": 350.0,
        },
    }

    registry = build_candidate_set(
        engine, signals, "moderate", PM_PROFILES["moderate"],
        {"positions": {}}, "cycle_test",
    )

    assert registry.is_empty


@patch("utils.candidate_builder.build_entry_geometry_scaffold", _fake_scaffold)
@patch("utils.gate_config.FAST_PATH_MODE", "enabled")
def test_signal_without_active_trigger_is_included():
    """When FAST_PATH_MODE==enabled but NO active trigger exists, signal passes through."""
    engine = create_engine("sqlite:///:memory:")
    _init_tables(engine)

    # No trigger inserted for AAPL
    signals = {
        "AAPL": {
            "symbol": "AAPL",
            "signal": "BUY",
            "strength": "strong",
            "setup_type": "momentum_fade",
            "current_price": 150.0,
        },
    }

    registry = build_candidate_set(
        engine, signals, "moderate", PM_PROFILES["moderate"],
        {"positions": {}}, "cycle_test",
    )

    assert not registry.is_empty


@patch("utils.candidate_builder.build_entry_geometry_scaffold", _fake_scaffold)
@patch("utils.gate_config.FAST_PATH_MODE", "enabled")
def test_only_matching_direction_suppressed():
    """Active trigger for SHORT does not suppress a BUY signal for same symbol."""
    engine = create_engine("sqlite:///:memory:")
    _init_tables(engine)

    # Active trigger for TSLA SHORT only
    _insert_active_trigger(engine, "TSLA", "SHORT", "moderate")

    signals = {
        "TSLA": {
            "symbol": "TSLA",
            "signal": "BUY",  # BUY direction — trigger is SHORT
            "strength": "strong",
            "setup_type": "momentum_fade",
            "current_price": 350.0,
        },
    }

    registry = build_candidate_set(
        engine, signals, "moderate", PM_PROFILES["moderate"],
        {"positions": {}}, "cycle_test",
    )

    assert not registry.is_empty


@patch("utils.candidate_builder.build_entry_geometry_scaffold", _fake_scaffold)
@patch("utils.gate_config.FAST_PATH_MODE", "enabled")
def test_mixed_signals_only_triggered_suppressed():
    """Only the signal with an active trigger is excluded; others pass through."""
    engine = create_engine("sqlite:///:memory:")
    _init_tables(engine)

    # Active trigger for TSLA SHORT
    _insert_active_trigger(engine, "TSLA", "SHORT", "moderate")

    signals = {
        "TSLA": {
            "symbol": "TSLA",
            "signal": "SHORT",
            "strength": "strong",
            "setup_type": "momentum_fade",
            "current_price": 350.0,
        },
        "AAPL": {
            "symbol": "AAPL",
            "signal": "BUY",
            "strength": "strong",
            "setup_type": "momentum_fade",
            "current_price": 150.0,
        },
    }

    registry = build_candidate_set(
        engine, signals, "moderate", PM_PROFILES["moderate"],
        {"positions": {}}, "cycle_test",
    )

    assert not registry.is_empty
    offered = registry.get_offered_summary()
    symbols = [c["symbol"] for c in offered]
    assert "AAPL" in symbols
    assert "TSLA" not in symbols


@patch("utils.candidate_builder.build_entry_geometry_scaffold", _fake_scaffold)
@patch("utils.gate_config.FAST_PATH_MODE", "disabled")
def test_no_suppression_when_mode_disabled():
    """When FAST_PATH_MODE==disabled, no signals are suppressed even with active trigger."""
    engine = create_engine("sqlite:///:memory:")
    _init_tables(engine)

    _insert_active_trigger(engine, "TSLA", "SHORT", "moderate")

    signals = {
        "TSLA": {
            "symbol": "TSLA",
            "signal": "SHORT",
            "strength": "strong",
            "setup_type": "momentum_fade",
            "current_price": 350.0,
        },
    }

    registry = build_candidate_set(
        engine, signals, "moderate", PM_PROFILES["moderate"],
        {"positions": {}}, "cycle_test",
    )

    assert not registry.is_empty


@patch("utils.candidate_builder.build_entry_geometry_scaffold", _fake_scaffold)
@patch("utils.gate_config.FAST_PATH_MODE", "observe")
def test_no_suppression_when_mode_observe():
    """When FAST_PATH_MODE==observe, no signals are suppressed — observe is read-only."""
    engine = create_engine("sqlite:///:memory:")
    _init_tables(engine)

    _insert_active_trigger(engine, "TSLA", "SHORT", "moderate")

    signals = {
        "TSLA": {
            "symbol": "TSLA",
            "signal": "SHORT",
            "strength": "strong",
            "setup_type": "momentum_fade",
            "current_price": 350.0,
        },
    }

    registry = build_candidate_set(
        engine, signals, "moderate", PM_PROFILES["moderate"],
        {"positions": {}}, "cycle_test",
    )

    assert not registry.is_empty


@patch("utils.candidate_builder.build_entry_geometry_scaffold", _fake_scaffold)
@patch("utils.gate_config.FAST_PATH_MODE", "enabled")
def test_fail_open_when_trigger_query_errors(monkeypatch):
    """If the fast-path trigger query fails, all signals pass through (fail-open)."""
    engine = create_engine("sqlite:///:memory:")
    _init_tables(engine)

    # Drop the table to force a query error
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE fast_path_triggers"))

    signals = {
        "TSLA": {
            "symbol": "TSLA",
            "signal": "SHORT",
            "strength": "strong",
            "setup_type": "momentum_fade",
            "current_price": 350.0,
        },
    }

    registry = build_candidate_set(
        engine, signals, "moderate", PM_PROFILES["moderate"],
        {"positions": {}}, "cycle_test",
    )

    # Fail-open: signal should still be processed
    assert not registry.is_empty


@patch("utils.candidate_builder.build_entry_geometry_scaffold", _fake_scaffold)
@patch("utils.gate_config.FAST_PATH_MODE", "enabled")
def test_direction_normalization_long_to_buy():
    """Signal direction 'LONG' maps to trigger direction 'BUY' for matching."""
    engine = create_engine("sqlite:///:memory:")
    _init_tables(engine)

    # Active trigger stored with direction BUY (normalized from LONG)
    _insert_active_trigger(engine, "AAPL", "BUY", "moderate")

    signals = {
        "AAPL": {
            "symbol": "AAPL",
            "signal": "LONG",  # Analyst uses LONG, trigger stores BUY
            "strength": "strong",
            "setup_type": "momentum_fade",
            "current_price": 150.0,
        },
    }

    registry = build_candidate_set(
        engine, signals, "moderate", PM_PROFILES["moderate"],
        {"positions": {}}, "cycle_test",
    )

    # LONG in signal matches BUY in trigger → suppressed
    assert registry.is_empty
