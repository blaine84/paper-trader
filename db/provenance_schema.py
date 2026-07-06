"""
Provenance namespace database schema and initialization.

This module defines the provenance-specific tables for the PM Decision Geometry
Integrity system. All tables are immutable (append-only) with BEFORE UPDATE and
BEFORE DELETE triggers that RAISE ABORT.

Tables:
- pm_raw_responses: Raw PM LLM responses captured before parsing
- response_lineage_links: Join table linking responses to candidate lineages
- provenance_events: Stage-by-stage mutation audit trail
- provenance_findings: Classified integrity findings (mismatches, gaps, etc.)

Requirements: 3.4, 14.1
"""

import logging
from sqlalchemy import text, event

from db.schema import is_sqlite

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# pm_raw_responses (immutable, append-only)
# ---------------------------------------------------------------------------

_PM_RAW_RESPONSES_DDL = """
CREATE TABLE IF NOT EXISTS pm_raw_responses (
    id INTEGER PRIMARY KEY,
    response_id VARCHAR(36) NOT NULL UNIQUE,
    pm_cycle_id VARCHAR(36) NOT NULL,
    profile VARCHAR(16) NOT NULL,
    model_id VARCHAR(128) NOT NULL,
    timestamp DATETIME NOT NULL,
    prompt_version_id VARCHAR(128) NOT NULL,
    candidate_ids_supplied_json TEXT NOT NULL,
    raw_payload TEXT,
    original_payload_hash VARCHAR(64) NOT NULL,
    stored_payload_hash VARCHAR(64),
    parse_status VARCHAR(48) NOT NULL,
    attempt_ordinal INTEGER NOT NULL DEFAULT 1,
    payload_size_bytes INTEGER NOT NULL,
    payload_truncated BOOLEAN DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

_PM_RAW_RESPONSES_INDEXES = [
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_raw_resp_cycle_attempt ON pm_raw_responses(pm_cycle_id, attempt_ordinal)",
    "CREATE INDEX IF NOT EXISTS ix_raw_resp_profile_ts ON pm_raw_responses(profile, timestamp)",
]

_PM_RAW_RESPONSES_IMMUTABILITY_TRIGGERS = [
    """
    CREATE TRIGGER IF NOT EXISTS tr_pm_raw_responses_no_update
        BEFORE UPDATE ON pm_raw_responses
    BEGIN
        SELECT RAISE(ABORT, 'pm_raw_responses is immutable: UPDATE prohibited');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS tr_pm_raw_responses_no_delete
        BEFORE DELETE ON pm_raw_responses
    BEGIN
        SELECT RAISE(ABORT, 'pm_raw_responses is immutable: DELETE prohibited');
    END
    """,
]


# ---------------------------------------------------------------------------
# response_lineage_links (immutable, append-only)
# ---------------------------------------------------------------------------

_RESPONSE_LINEAGE_LINKS_DDL = """
CREATE TABLE IF NOT EXISTS response_lineage_links (
    id INTEGER PRIMARY KEY,
    response_id VARCHAR(36) NOT NULL REFERENCES pm_raw_responses(response_id),
    lineage_id VARCHAR(36) NOT NULL,
    candidate_id VARCHAR(36),
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

_RESPONSE_LINEAGE_LINKS_INDEXES = [
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_resp_link_response_lineage ON response_lineage_links(response_id, lineage_id)",
    "CREATE INDEX IF NOT EXISTS ix_resp_link_lineage ON response_lineage_links(lineage_id)",
]

_RESPONSE_LINEAGE_LINKS_IMMUTABILITY_TRIGGERS = [
    """
    CREATE TRIGGER IF NOT EXISTS tr_response_lineage_links_no_update
        BEFORE UPDATE ON response_lineage_links
    BEGIN
        SELECT RAISE(ABORT, 'response_lineage_links is immutable: UPDATE prohibited');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS tr_response_lineage_links_no_delete
        BEFORE DELETE ON response_lineage_links
    BEGIN
        SELECT RAISE(ABORT, 'response_lineage_links is immutable: DELETE prohibited');
    END
    """,
]


# ---------------------------------------------------------------------------
# provenance_events (immutable, append-only)
# ---------------------------------------------------------------------------

_PROVENANCE_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS provenance_events (
    id INTEGER PRIMARY KEY,
    lineage_id VARCHAR(36) NOT NULL,
    stage_name VARCHAR(64) NOT NULL,
    stage_version VARCHAR(32) NOT NULL,
    sequence_number INTEGER NOT NULL,
    mutation_ordinal INTEGER NOT NULL DEFAULT 1,
    timestamp DATETIME NOT NULL,
    input_contract_json TEXT,
    output_contract_json TEXT,
    fields_changed_json TEXT,
    mutation_reason_code VARCHAR(64) NOT NULL,
    rule_id VARCHAR(128),
    geometry_before_json TEXT NOT NULL,
    geometry_after_json TEXT NOT NULL,
    validation_before VARCHAR(16) NOT NULL,
    validation_after VARCHAR(16) NOT NULL,
    attempt_ordinal INTEGER NOT NULL DEFAULT 1,
    is_terminal BOOLEAN DEFAULT 0,
    payload_truncated BOOLEAN DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

_PROVENANCE_EVENTS_INDEXES = [
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_prov_lineage_seq ON provenance_events(lineage_id, sequence_number)",
    "CREATE INDEX IF NOT EXISTS ix_prov_stage ON provenance_events(stage_name)",
    "CREATE INDEX IF NOT EXISTS ix_prov_validation ON provenance_events(validation_after)",
    "CREATE INDEX IF NOT EXISTS ix_prov_lineage ON provenance_events(lineage_id)",
]

_PROVENANCE_EVENTS_IMMUTABILITY_TRIGGERS = [
    """
    CREATE TRIGGER IF NOT EXISTS tr_provenance_events_no_update
        BEFORE UPDATE ON provenance_events
    BEGIN
        SELECT RAISE(ABORT, 'provenance_events is immutable: UPDATE prohibited');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS tr_provenance_events_no_delete
        BEFORE DELETE ON provenance_events
    BEGIN
        SELECT RAISE(ABORT, 'provenance_events is immutable: DELETE prohibited');
    END
    """,
]


# ---------------------------------------------------------------------------
# provenance_findings (immutable, append-only)
# ---------------------------------------------------------------------------

_PROVENANCE_FINDINGS_DDL = """
CREATE TABLE IF NOT EXISTS provenance_findings (
    id INTEGER PRIMARY KEY,
    finding_id VARCHAR(36) NOT NULL UNIQUE,
    lineage_id VARCHAR(36) NOT NULL,
    finding_type VARCHAR(64) NOT NULL,
    stage_name VARCHAR(64) NOT NULL,
    severity VARCHAR(16) NOT NULL,
    details_json TEXT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

_PROVENANCE_FINDINGS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_findings_lineage ON provenance_findings(lineage_id)",
    "CREATE INDEX IF NOT EXISTS ix_findings_type ON provenance_findings(finding_type)",
]

_PROVENANCE_FINDINGS_IMMUTABILITY_TRIGGERS = [
    """
    CREATE TRIGGER IF NOT EXISTS tr_provenance_findings_no_update
        BEFORE UPDATE ON provenance_findings
    BEGIN
        SELECT RAISE(ABORT, 'provenance_findings is immutable: UPDATE prohibited');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS tr_provenance_findings_no_delete
        BEFORE DELETE ON provenance_findings
    BEGIN
        SELECT RAISE(ABORT, 'provenance_findings is immutable: DELETE prohibited');
    END
    """,
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_provenance_schema(engine) -> None:
    """Initialize provenance schema with correct pragmas and all provenance tables.

    Sets up:
    - WAL mode for concurrent read/write performance (SQLite only)
    - busy_timeout=30000ms to handle lock contention (SQLite only)
    - foreign_keys=ON to enforce FK constraints (SQLite only)
    - All provenance tables with immutability triggers and indexes

    Safe to call multiple times (all DDL uses IF NOT EXISTS / CREATE OR REPLACE).
    """
    # Only register SQLite pragmas when engine is SQLite
    if is_sqlite(engine):
        _register_pragmas(engine)

    with engine.connect() as conn:
        # --- pm_raw_responses ---
        conn.execute(text(_PM_RAW_RESPONSES_DDL))
        for idx_sql in _PM_RAW_RESPONSES_INDEXES:
            conn.execute(text(idx_sql))

        # --- response_lineage_links ---
        conn.execute(text(_RESPONSE_LINEAGE_LINKS_DDL))
        for idx_sql in _RESPONSE_LINEAGE_LINKS_INDEXES:
            conn.execute(text(idx_sql))

        # --- provenance_events ---
        conn.execute(text(_PROVENANCE_EVENTS_DDL))
        for idx_sql in _PROVENANCE_EVENTS_INDEXES:
            conn.execute(text(idx_sql))

        # --- provenance_findings ---
        conn.execute(text(_PROVENANCE_FINDINGS_DDL))
        for idx_sql in _PROVENANCE_FINDINGS_INDEXES:
            conn.execute(text(idx_sql))

        # --- Immutability triggers (dialect-specific) ---
        if is_sqlite(engine):
            for trigger_sql in _PM_RAW_RESPONSES_IMMUTABILITY_TRIGGERS:
                conn.execute(text(trigger_sql))
            for trigger_sql in _RESPONSE_LINEAGE_LINKS_IMMUTABILITY_TRIGGERS:
                conn.execute(text(trigger_sql))
            for trigger_sql in _PROVENANCE_EVENTS_IMMUTABILITY_TRIGGERS:
                conn.execute(text(trigger_sql))
            for trigger_sql in _PROVENANCE_FINDINGS_IMMUTABILITY_TRIGGERS:
                conn.execute(text(trigger_sql))
        else:
            # Postgres: create shared trigger function (idempotent)
            conn.execute(text("""
                CREATE OR REPLACE FUNCTION raise_immutable() RETURNS trigger AS $$
                BEGIN
                    RAISE EXCEPTION '%% is immutable: %% prohibited', TG_TABLE_NAME, TG_OP;
                END;
                $$ LANGUAGE plpgsql
            """))
            # Create triggers for each immutable table
            _immutable_tables = [
                "pm_raw_responses",
                "response_lineage_links",
                "provenance_events",
                "provenance_findings",
            ]
            for table in _immutable_tables:
                conn.execute(text(
                    f"DROP TRIGGER IF EXISTS tr_{table}_no_update ON {table}"
                ))
                conn.execute(text(
                    f"CREATE TRIGGER tr_{table}_no_update "
                    f"BEFORE UPDATE ON {table} "
                    f"FOR EACH ROW EXECUTE FUNCTION raise_immutable()"
                ))
                conn.execute(text(
                    f"DROP TRIGGER IF EXISTS tr_{table}_no_delete ON {table}"
                ))
                conn.execute(text(
                    f"CREATE TRIGGER tr_{table}_no_delete "
                    f"BEFORE DELETE ON {table} "
                    f"FOR EACH ROW EXECUTE FUNCTION raise_immutable()"
                ))

        conn.commit()

    log.info("Provenance schema initialized successfully")


def _register_pragmas(engine) -> None:
    """Register SQLite pragmas for provenance database connections.

    Sets WAL mode, busy_timeout, and foreign_keys=ON on every new connection.
    """
    @event.listens_for(engine, "connect")
    def _set_provenance_pragmas(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
