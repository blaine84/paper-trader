"""
Blocker mitigation database schema and initialization.

This module defines the tables for the Candidate Blocker Mitigation feature:
- candidate_lifecycle_checklists: Post-entry trade lifecycle verification records
- daily_loss_summaries: Aggregated candidate attrition for a single trading day

All tables use IF NOT EXISTS for idempotent creation. No DROP, UPDATE, or DELETE
statements against existing tables.

Requirements: 1.3, 5.6, 7.1, 8.4, 9.5
"""

import logging
from sqlalchemy import text

from db.schema import is_sqlite

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# candidate_lifecycle_checklists
# ---------------------------------------------------------------------------

_LIFECYCLE_CHECKLISTS_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS candidate_lifecycle_checklists (
    id INTEGER PRIMARY KEY,
    candidate_id VARCHAR(36) NOT NULL,
    trade_id VARCHAR(64) NOT NULL,
    cycle_id VARCHAR(64) NOT NULL,
    profile_id VARCHAR(64) NOT NULL,
    trade_row_created BOOLEAN NOT NULL DEFAULT 0,
    position_row_created_or_updated BOOLEAN NOT NULL DEFAULT 0,
    stop_registered BOOLEAN NOT NULL DEFAULT 0,
    target_registered BOOLEAN NOT NULL DEFAULT 0,
    thesis_invalidation_recorded BOOLEAN NOT NULL DEFAULT 0,
    position_monitor_armed BOOLEAN NOT NULL DEFAULT 0,
    review_lineage_linked BOOLEAN NOT NULL DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

_LIFECYCLE_CHECKLISTS_POSTGRES_DDL = """
CREATE TABLE IF NOT EXISTS candidate_lifecycle_checklists (
    id SERIAL PRIMARY KEY,
    candidate_id VARCHAR(36) NOT NULL,
    trade_id VARCHAR(64) NOT NULL,
    cycle_id VARCHAR(64) NOT NULL,
    profile_id VARCHAR(64) NOT NULL,
    trade_row_created BOOLEAN NOT NULL DEFAULT FALSE,
    position_row_created_or_updated BOOLEAN NOT NULL DEFAULT FALSE,
    stop_registered BOOLEAN NOT NULL DEFAULT FALSE,
    target_registered BOOLEAN NOT NULL DEFAULT FALSE,
    thesis_invalidation_recorded BOOLEAN NOT NULL DEFAULT FALSE,
    position_monitor_armed BOOLEAN NOT NULL DEFAULT FALSE,
    review_lineage_linked BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_LIFECYCLE_CHECKLISTS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_lifecycle_checklists_candidate ON candidate_lifecycle_checklists (candidate_id)",
    "CREATE INDEX IF NOT EXISTS ix_lifecycle_checklists_trade ON candidate_lifecycle_checklists (trade_id)",
]


# ---------------------------------------------------------------------------
# daily_loss_summaries
# ---------------------------------------------------------------------------

_DAILY_LOSS_SUMMARIES_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS daily_loss_summaries (
    id INTEGER PRIMARY KEY,
    trade_date VARCHAR(10) NOT NULL,
    profile_id VARCHAR(64) NOT NULL,
    signals_seen INTEGER NOT NULL DEFAULT 0,
    candidates_built INTEGER NOT NULL DEFAULT 0,
    preflight_failed INTEGER NOT NULL DEFAULT 0,
    offered_to_pm INTEGER NOT NULL DEFAULT 0,
    pm_rejected INTEGER NOT NULL DEFAULT 0,
    pm_rejected_by_reason_json TEXT,
    pm_accepted INTEGER NOT NULL DEFAULT 0,
    gate_sizing_rejected INTEGER NOT NULL DEFAULT 0,
    execution_failed INTEGER NOT NULL DEFAULT 0,
    executed INTEGER NOT NULL DEFAULT 0,
    lifecycle_incomplete INTEGER NOT NULL DEFAULT 0,
    top_blocking_reasons_json TEXT,
    dominant_blocker_stage VARCHAR(64),
    error_indication TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(trade_date, profile_id)
)
"""

_DAILY_LOSS_SUMMARIES_POSTGRES_DDL = """
CREATE TABLE IF NOT EXISTS daily_loss_summaries (
    id SERIAL PRIMARY KEY,
    trade_date VARCHAR(10) NOT NULL,
    profile_id VARCHAR(64) NOT NULL,
    signals_seen INTEGER NOT NULL DEFAULT 0,
    candidates_built INTEGER NOT NULL DEFAULT 0,
    preflight_failed INTEGER NOT NULL DEFAULT 0,
    offered_to_pm INTEGER NOT NULL DEFAULT 0,
    pm_rejected INTEGER NOT NULL DEFAULT 0,
    pm_rejected_by_reason_json TEXT,
    pm_accepted INTEGER NOT NULL DEFAULT 0,
    gate_sizing_rejected INTEGER NOT NULL DEFAULT 0,
    execution_failed INTEGER NOT NULL DEFAULT 0,
    executed INTEGER NOT NULL DEFAULT 0,
    lifecycle_incomplete INTEGER NOT NULL DEFAULT 0,
    top_blocking_reasons_json TEXT,
    dominant_blocker_stage VARCHAR(64),
    error_indication TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(trade_date, profile_id)
)
"""

_DAILY_LOSS_SUMMARIES_INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_daily_loss_summaries_date ON daily_loss_summaries (trade_date)",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_blocker_mitigation_schema(engine) -> None:
    """Initialize blocker mitigation schema tables.

    Creates candidate_lifecycle_checklists and daily_loss_summaries tables
    with their indexes. Safe to call multiple times (all DDL uses IF NOT EXISTS).

    Requirements: 7.1, 8.4, 9.5
    """
    lifecycle_ddl = (
        _LIFECYCLE_CHECKLISTS_SQLITE_DDL
        if is_sqlite(engine)
        else _LIFECYCLE_CHECKLISTS_POSTGRES_DDL
    )
    daily_loss_ddl = (
        _DAILY_LOSS_SUMMARIES_SQLITE_DDL
        if is_sqlite(engine)
        else _DAILY_LOSS_SUMMARIES_POSTGRES_DDL
    )

    with engine.connect() as conn:
        # --- candidate_lifecycle_checklists ---
        conn.execute(text(lifecycle_ddl))
        for idx_sql in _LIFECYCLE_CHECKLISTS_INDEXES:
            conn.execute(text(idx_sql))

        # --- daily_loss_summaries ---
        conn.execute(text(daily_loss_ddl))
        for idx_sql in _DAILY_LOSS_SUMMARIES_INDEXES:
            conn.execute(text(idx_sql))

        conn.commit()

    log.info("Blocker mitigation schema initialized successfully")
