from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, text

from utils.candidate_registry import CandidateState, recover_stale_reservations


def _create_recovery_tables(engine):
    with engine.begin() as conn:
        conn.execute(text("""
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
                candidate_type TEXT DEFAULT 'intraday'
            )
        """))
        conn.execute(text("""
            CREATE TABLE pm_candidate_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                cycle_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_data TEXT,
                created_at TEXT NOT NULL,
                candidate_type TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE trade_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                payload_json TEXT
            )
        """))


def _insert_reserved_candidate(engine, *, candidate_id="cand-1", execution_key="exec-1"):
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO pm_candidates (
                    candidate_id, cycle_id, profile_id, symbol, direction,
                    setup_type, geometry_name, entry_price, stop_price,
                    target_price, risk_reward, source_signal_id,
                    signal_snapshot_json, state, integrity_hash, execution_key,
                    reserved_at, created_at, expires_at
                ) VALUES (
                    :candidate_id, 'cycle-1', 'moderate', 'AMD', 'BUY',
                    'trend_pullback', 'support_bounce', 100, 98,
                    104, 2, 'sig-1', '{}', :state, 'hash-1',
                    :execution_key, :reserved_at, :created_at, :expires_at
                )
            """),
            {
                "candidate_id": candidate_id,
                "execution_key": execution_key,
                "state": CandidateState.RESERVED.value,
                "reserved_at": (now - timedelta(minutes=10)).isoformat(),
                "created_at": (now - timedelta(minutes=10)).isoformat(),
                "expires_at": (now + timedelta(minutes=20)).isoformat(),
            },
        )


def test_recover_stale_reservations_uses_trade_event_payload_json():
    engine = create_engine("sqlite:///:memory:")
    _create_recovery_tables(engine)
    _insert_reserved_candidate(engine, execution_key="exec-found")

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO trade_events (event_type, payload_json)
                VALUES ('entry_filled', :payload_json)
            """),
            {"payload_json": '{"execution_key": "exec-found"}'},
        )

    recoveries = recover_stale_reservations(engine, lease_timeout_minutes=5)

    assert recoveries == [{
        "candidate_id": "cand-1",
        "action": "mark_executed",
        "reason": "trade_found_for_execution_key",
    }]
    with engine.connect() as conn:
        state = conn.execute(
            text("SELECT state FROM pm_candidates WHERE candidate_id = 'cand-1'")
        ).scalar_one()
        event_count = conn.execute(
            text("SELECT COUNT(*) FROM pm_candidate_events WHERE event_type = 'recovery_released'")
        ).scalar_one()

    assert state == CandidateState.EXECUTED.value
    assert event_count == 1
