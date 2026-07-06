"""Alert Dispatch Schema — DDL init for alert_intents, alert_cooldowns, alert_dispatch_log.

Creates all tables, indexes, and immutability triggers required by the
price-alert PM dispatcher feature. Uses IF NOT EXISTS DDL matching the
existing project pattern for non-destructive schema migrations.

Requirements: 1.1, 2.1, 5.1, 10.2, 11.3
"""

from __future__ import annotations

import logging

from sqlalchemy import event, inspect as sa_inspect, text

from db.schema import is_sqlite
from utils.dialect_sql import _default_timestamp, _pk_column

logger = logging.getLogger(__name__)


def init_alert_dispatch_schema(engine) -> None:
    """Create alert dispatch tables, indexes, and triggers via raw DDL.

    Applies WAL mode and busy_timeout=30000 via a SQLAlchemy event listener
    (SQLite only), then executes IF NOT EXISTS DDL for all required objects.

    Safe to call multiple times — all statements are idempotent.
    """
    # Set WAL mode and busy_timeout on every new connection (SQLite only)
    if is_sqlite(engine):
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

    pk = _pk_column(engine)
    ts_default = _default_timestamp(engine)

    with engine.begin() as conn:
        # --- alert_intents table -----------------------------------------------
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS alert_intents (
                {pk},
                alert_intent_id       TEXT NOT NULL UNIQUE,
                symbol                TEXT NOT NULL,
                alert_type            TEXT NOT NULL,
                direction             TEXT,
                trigger_price         TEXT NOT NULL,
                source_level          TEXT,
                urgency               TEXT NOT NULL DEFAULT 'medium',
                reason                TEXT,
                dedupe_key            TEXT NOT NULL,
                filter_status         TEXT NOT NULL DEFAULT 'unclassified',
                first_seen_at         TEXT NOT NULL,
                last_seen_at          TEXT NOT NULL,
                occurrence_count      INTEGER NOT NULL DEFAULT 1,
                expiration_at         TEXT NOT NULL,
                dispatch_status       TEXT NOT NULL DEFAULT 'pending',
                dispatch_reason       TEXT,
                dispatched_at         TEXT,
                deferred_until        TEXT,
                dispatch_attempt_count INTEGER NOT NULL DEFAULT 0,
                last_dispatch_error   TEXT,
                created_at            TEXT NOT NULL {ts_default}
            )
        """))

        # Partial unique index: only one active intent per dedupe_key
        conn.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS ix_alert_intents_dedupe_active
                ON alert_intents(dedupe_key)
                WHERE dispatch_status IN ('pending', 'dispatched', 'claimed_by_scheduled')
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_alert_intents_status_expiry
                ON alert_intents(dispatch_status, expiration_at)
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_alert_intents_symbol
                ON alert_intents(symbol, dispatch_status)
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_alert_intents_filter_status
                ON alert_intents(filter_status)
                WHERE filter_status = 'unclassified'
        """))

        # --- alert_cooldowns table ---------------------------------------------
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS alert_cooldowns (
                {pk},
                symbol              TEXT NOT NULL,
                expiry_at           TEXT NOT NULL,
                started_by_dispatch_id INTEGER,
                created_at          TEXT NOT NULL {ts_default}
            )
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_alert_cooldowns_symbol_expiry
                ON alert_cooldowns(symbol, expiry_at)
        """))

        # --- alert_dispatch_log table (immutable append-only) ------------------
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS alert_dispatch_log (
                {pk},
                alert_intent_id   TEXT NOT NULL,
                symbol            TEXT NOT NULL,
                alert_type        TEXT NOT NULL,
                urgency           TEXT NOT NULL,
                dispatch_status   TEXT NOT NULL,
                reason            TEXT,
                cooldown_remaining_seconds REAL,
                cycle_trigger_type TEXT,
                dispatch_attempt_count INTEGER,
                dispatched_at     TEXT NOT NULL {ts_default}
            )
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_dispatch_log_time
                ON alert_dispatch_log(dispatched_at)
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_dispatch_log_symbol
                ON alert_dispatch_log(symbol, dispatched_at)
        """))

        # --- pm_alert_claims table (mutable, PM-side idempotency) --------------
        # Requirements: 5.1, 5.4
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS pm_alert_claims (
                {pk},
                alert_intent_id       TEXT NOT NULL,
                symbol                TEXT NOT NULL,
                alert_type            TEXT NOT NULL,
                profile_id            TEXT NOT NULL,
                status                TEXT NOT NULL DEFAULT 'claimed',
                claimed_at            TEXT NOT NULL,
                completed_at          TEXT,
                UNIQUE(alert_intent_id, profile_id)
            )
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_pm_alert_claims_status_claimed
                ON pm_alert_claims(status, claimed_at)
                WHERE status = 'claimed'
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_pm_alert_claims_symbol_claimed
                ON pm_alert_claims(symbol, claimed_at)
        """))

        # --- pm_alert_events table (immutable, append-only audit log) ----------
        # Requirements: 5.3, 7.4
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS pm_alert_events (
                {pk},
                alert_intent_id       TEXT NOT NULL,
                event_type            TEXT NOT NULL,
                symbol                TEXT NOT NULL DEFAULT '',
                alert_type            TEXT NOT NULL DEFAULT '',
                profile_id            TEXT NOT NULL DEFAULT '',
                event_at              TEXT NOT NULL,
                event_data            TEXT
            )
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_pm_alert_events_intent_type
                ON pm_alert_events(alert_intent_id, event_type)
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_pm_alert_events_event_at
                ON pm_alert_events(event_at)
        """))

        # --- Immutability triggers (dialect-specific) --------------------------
        if is_sqlite(engine):
            conn.execute(text("""
                CREATE TRIGGER IF NOT EXISTS trg_alert_dispatch_log_no_update
                    BEFORE UPDATE ON alert_dispatch_log
                BEGIN
                    SELECT RAISE(ABORT, 'alert_dispatch_log is immutable');
                END
            """))

            conn.execute(text("""
                CREATE TRIGGER IF NOT EXISTS trg_alert_dispatch_log_no_delete
                    BEFORE DELETE ON alert_dispatch_log
                BEGIN
                    SELECT RAISE(ABORT, 'alert_dispatch_log is immutable');
                END
            """))

            conn.execute(text("""
                CREATE TRIGGER IF NOT EXISTS trg_pm_alert_events_no_update
                    BEFORE UPDATE ON pm_alert_events
                BEGIN
                    SELECT RAISE(ABORT, 'pm_alert_events is immutable: UPDATE blocked');
                END
            """))

            conn.execute(text("""
                CREATE TRIGGER IF NOT EXISTS trg_pm_alert_events_no_delete
                    BEFORE DELETE ON pm_alert_events
                BEGIN
                    SELECT RAISE(ABORT, 'pm_alert_events is immutable: DELETE blocked');
                END
            """))
        else:
            # Postgres: use PL/pgSQL trigger function for immutability
            conn.execute(text("""
                CREATE OR REPLACE FUNCTION raise_immutable() RETURNS trigger AS $$
                BEGIN
                    RAISE EXCEPTION '% is immutable: % prohibited', TG_TABLE_NAME, TG_OP;
                END;
                $$ LANGUAGE plpgsql
            """))
            for table in ["alert_dispatch_log", "pm_alert_events"]:
                conn.execute(text(f"DROP TRIGGER IF EXISTS trg_{table}_no_update ON {table}"))
                conn.execute(text(f"""
                    CREATE TRIGGER trg_{table}_no_update BEFORE UPDATE ON {table}
                    FOR EACH ROW EXECUTE FUNCTION raise_immutable()
                """))
                conn.execute(text(f"DROP TRIGGER IF EXISTS trg_{table}_no_delete ON {table}"))
                conn.execute(text(f"""
                    CREATE TRIGGER trg_{table}_no_delete BEFORE DELETE ON {table}
                    FOR EACH ROW EXECUTE FUNCTION raise_immutable()
                """))

    # --- Non-destructive column migrations -----------------------------------
    # Add columns to existing tables using inspector check.
    # Safe to call multiple times — skips if column already exists.
    _migrate_alert_intents_columns(engine)

    # --- Enhanced audit columns for alert_dispatch_log (non-destructive) -----
    # Requirements: 7.1, 7.2 — complete audit trail for dispatch decisions
    _add_dispatch_log_audit_columns(engine)

    logger.info("alert_dispatch_schema: all tables and indexes initialized")


def _migrate_alert_intents_columns(engine) -> None:
    """Add new columns to alert_intents if not already present.

    Non-destructive: checks inspector.get_columns() before ALTER TABLE.
    Required for dispatch-once semantics (occurrence_count_at_deferral stores
    the occurrence_count at the time deferral was set, enabling material change detection).

    Requirements: 2.1, 2.5
    """
    inspector = sa_inspect(engine)
    existing_columns = {col["name"] for col in inspector.get_columns("alert_intents")}

    if "occurrence_count_at_deferral" not in existing_columns:
        with engine.begin() as conn:
            conn.execute(text(
                "ALTER TABLE alert_intents ADD COLUMN occurrence_count_at_deferral INTEGER DEFAULT 0"
            ))
            logger.warning(
                "Schema migration: added alert_intents.occurrence_count_at_deferral (INTEGER DEFAULT 0)"
            )


def _add_dispatch_log_audit_columns(engine) -> None:
    """Add enhanced audit columns to alert_dispatch_log via non-destructive ALTER TABLE.

    Uses inspector.get_columns() to check for existing columns before adding.
    Safe to call multiple times — skips columns that already exist.

    Requirements: 7.1, 7.2
    """
    columns_to_add = [
        ("dedupe_key", "TEXT"),
        ("configured_mode", "TEXT"),
        ("freshness_age_seconds", "REAL"),
        ("first_seen_age_seconds", "REAL"),
        ("dispatch_batch_symbols", "TEXT"),
        ("trigger_price", "REAL"),
        ("occurrence_count", "INTEGER"),
    ]

    inspector = sa_inspect(engine)
    existing_columns = {col["name"] for col in inspector.get_columns("alert_dispatch_log")}

    with engine.begin() as conn:
        for col_name, col_type in columns_to_add:
            if col_name in existing_columns:
                continue
            try:
                conn.execute(text(
                    f"ALTER TABLE alert_dispatch_log ADD COLUMN {col_name} {col_type}"
                ))
                logger.warning(
                    "Schema migration: added alert_dispatch_log.%s (%s)",
                    col_name, col_type,
                )
            except Exception as e:
                logger.error(
                    "Failed to add alert_dispatch_log.%s: %s", col_name, e
                )
