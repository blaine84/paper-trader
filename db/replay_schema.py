"""
Replay namespace database schema and initialization.

This module defines the replay-specific tables stored in a SEPARATE namespace
from production trade data. Production gate logic, case-library queries, and
position-tracking queries SHALL NOT read from the replay namespace.

Also defines the `decision_snapshots` table in the PRODUCTION schema (audit
inputs written during live PM cycles) with immutability triggers.

Requirements: 3.3, 3.4, 9.4, 13.3
"""

import logging
from sqlalchemy import text, event, inspect as sa_inspect

from db.schema import is_sqlite

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Production schema: decision_snapshots (immutable after gate evaluation)
# ---------------------------------------------------------------------------

_DECISION_SNAPSHOTS_DDL = """
CREATE TABLE IF NOT EXISTS decision_snapshots (
    id INTEGER PRIMARY KEY,
    snapshot_id VARCHAR(36) NOT NULL UNIQUE,
    schema_version VARCHAR(10) NOT NULL,
    candidate_lineage_id VARCHAR(36) NOT NULL,
    timestamp DATETIME NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    profile VARCHAR(16) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    setup_type VARCHAR(64),
    analyst_signal_json TEXT,
    signal_strength REAL,
    confidence_value REAL,
    decision_payload_json TEXT NOT NULL,
    entry_price TEXT NOT NULL,
    stop_price TEXT NOT NULL,
    target_price TEXT NOT NULL,
    quantity TEXT NOT NULL,
    atr_value REAL,
    atr_bar_timestamps_json TEXT,
    account_equity TEXT NOT NULL,
    available_cash TEXT NOT NULL,
    open_position_context_json TEXT,
    case_library_stats_json TEXT,
    gate_config_json TEXT NOT NULL,
    feature_flags_json TEXT NOT NULL,
    policy_version_id VARCHAR(128) NOT NULL,
    geometry_hash VARCHAR(64),
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

_DECISION_SNAPSHOTS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_snapshot_lineage ON decision_snapshots(candidate_lineage_id)",
    "CREATE INDEX IF NOT EXISTS ix_snapshot_symbol_ts ON decision_snapshots(symbol, timestamp)",
    "CREATE INDEX IF NOT EXISTS ix_snapshot_profile_ts ON decision_snapshots(profile, timestamp)",
]

_DECISION_SNAPSHOTS_IMMUTABILITY_TRIGGERS = [
    """
    CREATE TRIGGER IF NOT EXISTS tr_decision_snapshots_no_update
        BEFORE UPDATE ON decision_snapshots
    BEGIN
        SELECT RAISE(ABORT, 'decision_snapshots is immutable: UPDATE prohibited');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS tr_decision_snapshots_no_delete
        BEFORE DELETE ON decision_snapshots
    BEGIN
        SELECT RAISE(ABORT, 'decision_snapshots is immutable: DELETE prohibited');
    END
    """,
]


# ---------------------------------------------------------------------------
# Replay namespace: replay_audit_records
# ---------------------------------------------------------------------------

_REPLAY_AUDIT_RECORDS_DDL = """
CREATE TABLE IF NOT EXISTS replay_audit_records (
    id INTEGER PRIMARY KEY,
    replay_id VARCHAR(36) NOT NULL UNIQUE,
    batch_run_id VARCHAR(36),
    candidate_id VARCHAR(36) NOT NULL,
    source_candidate_ids_json TEXT NOT NULL,
    snapshot_id VARCHAR(36),
    replay_cutoff DATETIME NOT NULL,
    input_sources_json TEXT NOT NULL,
    policy_version_json TEXT NOT NULL,
    replay_status VARCHAR(16) NOT NULL,
    gate_trace_json TEXT,
    decision_delta_classification VARCHAR(64),
    decision_delta_json TEXT,
    counterfactual_outcome_json TEXT,
    divergence_cause VARCHAR(64),
    divergence_evidence_json TEXT,
    code_revision VARCHAR(128),
    era VARCHAR(16) NOT NULL,
    diagnostic_mode BOOLEAN DEFAULT false,
    failure_reason_code VARCHAR(64),
    failure_details TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

_REPLAY_AUDIT_RECORDS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_replay_candidate ON replay_audit_records(candidate_id)",
    "CREATE INDEX IF NOT EXISTS ix_replay_batch ON replay_audit_records(batch_run_id)",
    "CREATE INDEX IF NOT EXISTS ix_replay_status ON replay_audit_records(replay_status)",
    "CREATE INDEX IF NOT EXISTS ix_replay_delta ON replay_audit_records(decision_delta_classification)",
    "CREATE INDEX IF NOT EXISTS ix_replay_era ON replay_audit_records(era)",
    "CREATE INDEX IF NOT EXISTS ix_replay_created ON replay_audit_records(created_at)",
]

_REPLAY_AUDIT_RECORDS_IMMUTABILITY_TRIGGERS = [
    """
    CREATE TRIGGER IF NOT EXISTS tr_replay_audit_no_update
        BEFORE UPDATE ON replay_audit_records
    BEGIN
        SELECT RAISE(ABORT, 'replay_audit_records is immutable: UPDATE prohibited');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS tr_replay_audit_no_delete
        BEFORE DELETE ON replay_audit_records
    BEGIN
        SELECT RAISE(ABORT, 'replay_audit_records is immutable: DELETE prohibited');
    END
    """,
]


# ---------------------------------------------------------------------------
# Replay namespace: replay_batch_runs
# ---------------------------------------------------------------------------

_REPLAY_BATCH_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS replay_batch_runs (
    id INTEGER PRIMARY KEY,
    batch_run_id VARCHAR(36) NOT NULL UNIQUE,
    started_at DATETIME NOT NULL,
    ended_at DATETIME,
    mode VARCHAR(16) NOT NULL,
    policy_version_json TEXT NOT NULL,
    filters_json TEXT,
    candidates_total INTEGER DEFAULT 0,
    candidates_processed INTEGER DEFAULT 0,
    candidates_failed INTEGER DEFAULT 0,
    exact_count INTEGER DEFAULT 0,
    partial_count INTEGER DEFAULT 0,
    unscorable_count INTEGER DEFAULT 0,
    delta_counts_json TEXT,
    duration_seconds REAL,
    status VARCHAR(16) NOT NULL,
    watermark_start DATETIME,
    watermark_end DATETIME,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


# ---------------------------------------------------------------------------
# Replay namespace: replay_batch_items
# ---------------------------------------------------------------------------

_REPLAY_BATCH_ITEMS_DDL = """
CREATE TABLE IF NOT EXISTS replay_batch_items (
    id INTEGER PRIMARY KEY,
    batch_run_id VARCHAR(36) NOT NULL REFERENCES replay_batch_runs(batch_run_id),
    candidate_id VARCHAR(36) NOT NULL,
    processing_order INTEGER NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'pending',
    replay_id VARCHAR(36),
    failure_reason_code VARCHAR(64),
    failure_details TEXT,
    started_at DATETIME,
    ended_at DATETIME,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

_REPLAY_BATCH_ITEMS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_batch_items_batch ON replay_batch_items(batch_run_id, processing_order)",
    "CREATE INDEX IF NOT EXISTS ix_batch_items_status ON replay_batch_items(batch_run_id, status)",
]


# ---------------------------------------------------------------------------
# Replay namespace: replay_annotations (append-only)
# ---------------------------------------------------------------------------

_REPLAY_ANNOTATIONS_DDL = """
CREATE TABLE IF NOT EXISTS replay_annotations (
    id INTEGER PRIMARY KEY,
    replay_id VARCHAR(36) NOT NULL REFERENCES replay_audit_records(replay_id),
    author VARCHAR(64) NOT NULL,
    annotation_timestamp DATETIME NOT NULL,
    content TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

_REPLAY_ANNOTATIONS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_annotation_replay ON replay_annotations(replay_id)",
]


# ---------------------------------------------------------------------------
# Replay namespace: replay_counterfactual_outcomes
# ---------------------------------------------------------------------------

_REPLAY_COUNTERFACTUAL_OUTCOMES_DDL = """
CREATE TABLE IF NOT EXISTS replay_counterfactual_outcomes (
    id INTEGER PRIMARY KEY,
    replay_id VARCHAR(36) NOT NULL REFERENCES replay_audit_records(replay_id),
    candidate_id VARCHAR(36) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    proposed_entry_price TEXT NOT NULL,
    simulated_fill_price TEXT NOT NULL,
    fill_timestamp DATETIME,
    fill_rule VARCHAR(64) NOT NULL,
    max_permitted_fill_delay_seconds INTEGER NOT NULL DEFAULT 300,
    fill_gap_seconds INTEGER,
    slippage_estimate TEXT DEFAULT '0',
    stop_price TEXT NOT NULL,
    target_price TEXT NOT NULL,
    return_15m REAL,
    return_30m REAL,
    return_60m REAL,
    mfe REAL,
    mae REAL,
    stop_hit BOOLEAN DEFAULT false,
    target_hit BOOLEAN DEFAULT false,
    first_hit VARCHAR(16),
    first_hit_candle_time DATETIME,
    status VARCHAR(32) NOT NULL,
    unscorable_reason TEXT,
    candles_json TEXT,
    candle_source VARCHAR(64),
    candle_fetch_timestamp DATETIME,
    candle_content_hash VARCHAR(64),
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

_REPLAY_COUNTERFACTUAL_OUTCOMES_INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_outcome_replay ON replay_counterfactual_outcomes(replay_id)",
    "CREATE INDEX IF NOT EXISTS ix_outcome_candidate ON replay_counterfactual_outcomes(candidate_id)",
]


# ---------------------------------------------------------------------------
# Candidate lineage ID migrations (add column to existing tables)
# ---------------------------------------------------------------------------

_LINEAGE_MIGRATIONS = [
    # (table, column, type, index_name)
    ("blocked_trade_candidates", "candidate_lineage_id", "VARCHAR(36)", "ix_blocked_lineage"),
    ("funnel_candidates", "candidate_lineage_id", "VARCHAR(36)", "ix_funnel_lineage"),
    ("trade_events", "candidate_lineage_id", "VARCHAR(36)", "ix_trade_events_lineage"),
    ("trades", "candidate_lineage_id", "VARCHAR(36)", "ix_trades_lineage"),
    ("pm_candidates", "candidate_lineage_id", "VARCHAR(36)", "ix_pm_candidates_lineage"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_replay_db(engine) -> None:
    """Initialize replay schema with correct pragmas and all replay tables.

    Sets up:
    - WAL mode for concurrent read/write performance (SQLite only)
    - busy_timeout=30000ms to handle lock contention (SQLite only)
    - foreign_keys=ON to enforce FK constraints in replay namespace (SQLite only)
    - All replay namespace tables with triggers and indexes
    - decision_snapshots table in production schema with immutability triggers
    - candidate_lineage_id columns on existing tables (non-destructive)

    Safe to call multiple times (all DDL uses IF NOT EXISTS / IF NOT EXISTS triggers).
    """
    # Register pragma listener only for SQLite engines
    if is_sqlite(engine):
        _register_pragmas(engine)

    with engine.connect() as conn:
        # --- Production schema: decision_snapshots ---
        conn.execute(text(_DECISION_SNAPSHOTS_DDL))
        for idx_sql in _DECISION_SNAPSHOTS_INDEXES:
            conn.execute(text(idx_sql))

        # --- Replay namespace tables ---
        conn.execute(text(_REPLAY_AUDIT_RECORDS_DDL))
        for idx_sql in _REPLAY_AUDIT_RECORDS_INDEXES:
            conn.execute(text(idx_sql))

        conn.execute(text(_REPLAY_BATCH_RUNS_DDL))

        conn.execute(text(_REPLAY_BATCH_ITEMS_DDL))
        for idx_sql in _REPLAY_BATCH_ITEMS_INDEXES:
            conn.execute(text(idx_sql))

        conn.execute(text(_REPLAY_ANNOTATIONS_DDL))
        for idx_sql in _REPLAY_ANNOTATIONS_INDEXES:
            conn.execute(text(idx_sql))

        conn.execute(text(_REPLAY_COUNTERFACTUAL_OUTCOMES_DDL))
        for idx_sql in _REPLAY_COUNTERFACTUAL_OUTCOMES_INDEXES:
            conn.execute(text(idx_sql))

        # --- Immutability triggers (dialect-specific) ---
        if is_sqlite(engine):
            for trigger_sql in _DECISION_SNAPSHOTS_IMMUTABILITY_TRIGGERS:
                conn.execute(text(trigger_sql))
            for trigger_sql in _REPLAY_AUDIT_RECORDS_IMMUTABILITY_TRIGGERS:
                conn.execute(text(trigger_sql))
        else:
            # Postgres: use shared PL/pgSQL trigger function
            conn.execute(text("""
                CREATE OR REPLACE FUNCTION raise_immutable() RETURNS trigger AS $$
                BEGIN
                    RAISE EXCEPTION '% is immutable: % prohibited', TG_TABLE_NAME, TG_OP;
                END;
                $$ LANGUAGE plpgsql
            """))
            for table in ["decision_snapshots", "replay_audit_records"]:
                conn.execute(text(f"DROP TRIGGER IF EXISTS tr_{table}_no_update ON {table}"))
                conn.execute(text(f"""
                    CREATE TRIGGER tr_{table}_no_update BEFORE UPDATE ON {table}
                    FOR EACH ROW EXECUTE FUNCTION raise_immutable()
                """))
                conn.execute(text(f"DROP TRIGGER IF EXISTS tr_{table}_no_delete ON {table}"))
                conn.execute(text(f"""
                    CREATE TRIGGER tr_{table}_no_delete BEFORE DELETE ON {table}
                    FOR EACH ROW EXECUTE FUNCTION raise_immutable()
                """))

        # --- Add candidate_lineage_id to existing tables ---
        _migrate_lineage_columns(conn, engine)

        conn.commit()

    log.info("Replay schema initialized successfully")


def _register_pragmas(engine) -> None:
    """Register SQLite pragmas for replay database connections.

    Sets WAL mode, busy_timeout, and foreign_keys=ON on every new connection.
    """
    @event.listens_for(engine, "connect")
    def _set_replay_pragmas(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def _migrate_lineage_columns(conn, engine) -> None:
    """Add candidate_lineage_id column to existing tables if not already present.

    Non-destructive: skips tables/columns that already exist.
    Uses SQLAlchemy inspector for dialect-neutral column introspection.
    Logs each migration for audit trail.
    """
    inspector = sa_inspect(engine)
    for table_name, column_name, col_type, index_name in _LINEAGE_MIGRATIONS:
        # Check if table exists
        if not inspector.has_table(table_name):
            log.debug(f"Table {table_name} not found, skipping lineage migration")
            continue

        # Check if column already exists
        existing_columns = {col["name"] for col in inspector.get_columns(table_name)}
        if column_name in existing_columns:
            continue

        try:
            conn.execute(text(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {col_type}"
            ))
            conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({column_name})"
            ))
            log.warning(f"Schema migration: added {table_name}.{column_name} ({col_type})")
        except Exception as e:
            log.error(f"Failed to add {table_name}.{column_name}: {e}")
