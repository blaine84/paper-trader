from sqlalchemy import create_engine, text

from models.pm_profiles import PM_PROFILES
from utils.candidate_builder import build_candidate_set


def _create_pm_candidates_table(engine):
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
                    candidate_lineage_id TEXT
                )
                """
            )
        )


def test_candidate_builder_does_not_pass_pm_profile_as_geometry_overrides(monkeypatch):
    """PM profile fields like target_multiplier must not erase scaffold defaults."""
    engine = create_engine("sqlite:///:memory:")
    _create_pm_candidates_table(engine)
    calls = []

    def fake_scaffold(signal, profile_id=None, profile_context=None):
        calls.append(
            {
                "signal": signal,
                "profile_id": profile_id,
                "profile_context": profile_context,
            }
        )
        return {
            "symbol": signal["symbol"],
            "direction": "SHORT",
            "status": "ok",
            "candidates": [
                {
                    "name": "breakdown_continuation",
                    "entry_price": 53.88,
                    "stop_loss": 53.99,
                    "target": 53.66,
                    "risk_reward": 2.0,
                    "trigger": "Price breaks below support",
                    "invalidation_basis": "Price recovers above stop",
                    "target_basis": "Entry - risk x target multiplier",
                }
            ],
        }

    monkeypatch.setattr("utils.candidate_builder.build_entry_geometry_scaffold", fake_scaffold)

    registry = build_candidate_set(
        engine,
        {
            "XLF": {
                "symbol": "XLF",
                "signal": "SHORT",
                "strength": "moderate",
                "setup_type": "momentum_fade",
                "current_price": 53.9,
            }
        },
        "moderate",
        PM_PROFILES["moderate"],
        {"positions": {}},
        "cycle_test",
    )

    assert not registry.is_empty
    assert calls == [
        {
            "signal": {
                "symbol": "XLF",
                "signal": "SHORT",
                "strength": "moderate",
                "setup_type": "momentum_fade",
                "current_price": 53.9,
            },
            "profile_id": "moderate",
            "profile_context": None,
        }
    ]


def test_moderate_profile_candidate_builder_produces_candidates_with_live_shape():
    engine = create_engine("sqlite:///:memory:")
    _create_pm_candidates_table(engine)

    registry = build_candidate_set(
        engine,
        {
            "XLF": {
                "symbol": "XLF",
                "signal": "SHORT",
                "strength": "moderate",
                "setup_type": "momentum_fade",
                "current_price": 53.9,
                "key_levels": {
                    "support": 53.88,
                    "resistance": 54.61,
                    "vwap": 54.16,
                    "prior_high": 54.89,
                    "prior_low": 53.9,
                    "day_high": 54.61,
                    "day_low": 53.87,
                },
            }
        },
        "moderate",
        PM_PROFILES["moderate"],
        {"positions": {}},
        "cycle_test",
    )

    assert not registry.is_empty
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT symbol, direction, risk_reward, state FROM pm_candidates")
        ).fetchall()

    assert rows
    assert all(row.symbol == "XLF" for row in rows)
    assert all(row.direction == "SHORT" for row in rows)
    assert all(row.risk_reward >= 1.0 for row in rows)
    assert all(row.state == "registered" for row in rows)

    offered = registry.get_offered_summary()
    assert offered
    assert all(candidate["invalidation_basis"] for candidate in offered)
    assert all(candidate["target_basis"] for candidate in offered)


def test_non_executable_setup_type_excluded(monkeypatch):
    """Signals with setup types not in CANDIDATE_EXECUTABLE_SETUP_TYPES are excluded."""
    engine = create_engine("sqlite:///:memory:")
    _create_pm_candidates_table(engine)

    def fake_scaffold(signal, profile_id=None, profile_context=None):
        return {
            "symbol": signal["symbol"],
            "direction": "LONG",
            "status": "ok",
            "candidates": [
                {
                    "name": "base_breakout",
                    "entry_price": 150.0,
                    "stop_loss": 148.0,
                    "target": 154.0,
                    "risk_reward": 2.0,
                    "trigger": "Price breaks above",
                    "invalidation_basis": "Falls below stop",
                    "target_basis": "Entry + RR * risk",
                }
            ],
        }

    monkeypatch.setattr("utils.candidate_builder.build_entry_geometry_scaffold", fake_scaffold)

    registry = build_candidate_set(
        engine,
        {
            "AAPL": {
                "symbol": "AAPL",
                "signal": "BUY",
                "strength": "strong",
                "setup_type": "sector_rotation",
                "current_price": 150.0,
            }
        },
        "moderate",
        PM_PROFILES["moderate"],
        {"positions": {}},
        "cycle_test",
    )

    assert registry.is_empty


def test_executable_setup_type_registered(monkeypatch):
    """Signals with setup types in CANDIDATE_EXECUTABLE_SETUP_TYPES are registered."""
    engine = create_engine("sqlite:///:memory:")
    _create_pm_candidates_table(engine)

    def fake_scaffold(signal, profile_id=None, profile_context=None):
        return {
            "symbol": signal["symbol"],
            "direction": "LONG",
            "status": "ok",
            "candidates": [
                {
                    "name": "base_breakout",
                    "entry_price": 150.0,
                    "stop_loss": 148.0,
                    "target": 154.0,
                    "risk_reward": 2.0,
                    "trigger": "Price breaks above",
                    "invalidation_basis": "Falls below stop",
                    "target_basis": "Entry + RR * risk",
                }
            ],
        }

    monkeypatch.setattr("utils.candidate_builder.build_entry_geometry_scaffold", fake_scaffold)

    registry = build_candidate_set(
        engine,
        {
            "AAPL": {
                "symbol": "AAPL",
                "signal": "BUY",
                "strength": "strong",
                "setup_type": "momentum_fade",
                "current_price": 150.0,
            }
        },
        "moderate",
        PM_PROFILES["moderate"],
        {"positions": {}},
        "cycle_test",
    )

    assert not registry.is_empty
