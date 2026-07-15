"""
Lifecycle Checklist Writer — Post-entry trade setup completeness verification.

After a candidate reaches EXECUTED state, this module queries current state
for each lifecycle component and persists a structured checklist record.

Fail-open: The entire write_lifecycle_checklist() function is wrapped in
try/except. On any failure, it logs at ERROR and returns None — never blocks
the pipeline or alters candidate terminal state.

Requirements: 7.1, 7.2, 7.3, 7.4
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LifecycleChecklist:
    """Post-entry trade lifecycle verification record.

    Each boolean field is true if the corresponding row or registration
    exists at checklist-write time, and false otherwise.
    """

    candidate_id: str
    trade_id: str
    trade_row_created: bool
    position_row_created_or_updated: bool
    stop_registered: bool
    target_registered: bool
    thesis_invalidation_recorded: bool
    position_monitor_armed: bool
    review_lineage_linked: bool

    @property
    def complete(self) -> bool:
        """True when all lifecycle components are present."""
        return all([
            self.trade_row_created,
            self.position_row_created_or_updated,
            self.stop_registered,
            self.target_registered,
            self.thesis_invalidation_recorded,
            self.position_monitor_armed,
            self.review_lineage_linked,
        ])

    @property
    def missing_components(self) -> list[str]:
        """List field names where value is false."""
        missing = []
        for field_name in [
            "trade_row_created",
            "position_row_created_or_updated",
            "stop_registered",
            "target_registered",
            "thesis_invalidation_recorded",
            "position_monitor_armed",
            "review_lineage_linked",
        ]:
            if not getattr(self, field_name):
                missing.append(field_name)
        return missing


# ---------------------------------------------------------------------------
# SQL queries for component checks
# ---------------------------------------------------------------------------

_CHECK_TRADE_ROW = text("""
    SELECT COUNT(*) FROM trades
    WHERE candidate_lineage_id = :candidate_id
""")

_CHECK_POSITION_ROW = text("""
    SELECT COUNT(*) FROM positions
    WHERE symbol = :symbol AND profile = :profile_id
""")

_CHECK_STOP_REGISTERED = text("""
    SELECT COUNT(*) FROM trades
    WHERE candidate_lineage_id = :candidate_id
      AND stop_price IS NOT NULL AND stop_price != 0
""")

_CHECK_TARGET_REGISTERED = text("""
    SELECT COUNT(*) FROM trades
    WHERE candidate_lineage_id = :candidate_id
      AND target_price IS NOT NULL AND target_price != 0
""")

_CHECK_THESIS_INVALIDATION = text("""
    SELECT COUNT(*) FROM trades
    WHERE candidate_lineage_id = :candidate_id
      AND invalidators IS NOT NULL AND invalidators != '' AND invalidators != '[]'
""")

_CHECK_POSITION_MONITOR = text("""
    SELECT COUNT(*) FROM trades
    WHERE candidate_lineage_id = :candidate_id
      AND status = 'open'
""")

_CHECK_REVIEW_LINEAGE = text("""
    SELECT COUNT(*) FROM response_lineage_links
    WHERE candidate_id = :candidate_id
""")

_INSERT_CHECKLIST = text("""
    INSERT INTO candidate_lifecycle_checklists (
        candidate_id, trade_id, cycle_id, profile_id,
        trade_row_created, position_row_created_or_updated,
        stop_registered, target_registered,
        thesis_invalidation_recorded, position_monitor_armed,
        review_lineage_linked, created_at
    ) VALUES (
        :candidate_id, :trade_id, :cycle_id, :profile_id,
        :trade_row_created, :position_row_created_or_updated,
        :stop_registered, :target_registered,
        :thesis_invalidation_recorded, :position_monitor_armed,
        :review_lineage_linked, :created_at
    )
""")


def _get_candidate_symbol(conn, candidate_id: str) -> str | None:
    """Look up candidate symbol from pm_candidates table."""
    row = conn.execute(
        text("SELECT symbol, profile_id FROM pm_candidates WHERE candidate_id = :cid"),
        {"cid": candidate_id},
    ).fetchone()
    if row:
        return row[0]
    return None


def _get_candidate_profile(conn, candidate_id: str) -> str | None:
    """Look up candidate profile_id from pm_candidates table."""
    row = conn.execute(
        text("SELECT profile_id FROM pm_candidates WHERE candidate_id = :cid"),
        {"cid": candidate_id},
    ).fetchone()
    if row:
        return row[0]
    return None


def write_lifecycle_checklist(
    engine,
    candidate_id: str,
    trade_id: str,
    cycle_id: str,
    profile_id: str,
) -> LifecycleChecklist | None:
    """Query current state and persist lifecycle checklist. Fail-open.

    Queries the database for each lifecycle component's existence, creates
    a LifecycleChecklist dataclass, persists it to the
    candidate_lifecycle_checklists table, and returns the checklist.

    On any exception, logs at ERROR level and returns None. Never raises.

    Args:
        engine: SQLAlchemy engine for database access.
        candidate_id: The candidate UUID linking to pm_candidates.
        trade_id: The trade identifier (string representation of trades.id
                  or candidate_lineage_id depending on caller context).
        cycle_id: The PM cycle identifier.
        profile_id: The active trading profile.

    Returns:
        LifecycleChecklist if successful, None on any failure.
    """
    try:
        with engine.connect() as conn:
            # Look up the candidate's symbol for position check
            symbol = _get_candidate_symbol(conn, candidate_id)

            # 1. trade_row_created: check if candidate_lineage_id exists in trades
            trade_row_created = (
                conn.execute(_CHECK_TRADE_ROW, {"candidate_id": candidate_id}).scalar() or 0
            ) > 0

            # 2. position_row_created_or_updated: check active position for symbol
            if symbol:
                position_row_created = (
                    conn.execute(
                        _CHECK_POSITION_ROW,
                        {"symbol": symbol, "profile_id": profile_id},
                    ).scalar() or 0
                ) > 0
            else:
                position_row_created = False

            # 3. stop_registered: check if trade has a stop_price set
            stop_registered = (
                conn.execute(_CHECK_STOP_REGISTERED, {"candidate_id": candidate_id}).scalar() or 0
            ) > 0

            # 4. target_registered: check if trade has a target_price set
            target_registered = (
                conn.execute(_CHECK_TARGET_REGISTERED, {"candidate_id": candidate_id}).scalar() or 0
            ) > 0

            # 5. thesis_invalidation_recorded: check if invalidators are present
            thesis_invalidation_recorded = (
                conn.execute(_CHECK_THESIS_INVALIDATION, {"candidate_id": candidate_id}).scalar() or 0
            ) > 0

            # 6. position_monitor_armed: trade exists in open status
            #    (price_monitor agent monitors all open trades automatically)
            position_monitor_armed = (
                conn.execute(_CHECK_POSITION_MONITOR, {"candidate_id": candidate_id}).scalar() or 0
            ) > 0

            # 7. review_lineage_linked: response_lineage_links has entry
            #    Fail-open for this specific check if table doesn't exist
            try:
                review_lineage_linked = (
                    conn.execute(_CHECK_REVIEW_LINEAGE, {"candidate_id": candidate_id}).scalar() or 0
                ) > 0
            except Exception:
                # Table may not exist if provenance schema not initialized
                review_lineage_linked = False

            # Build the checklist dataclass
            checklist = LifecycleChecklist(
                candidate_id=candidate_id,
                trade_id=trade_id,
                trade_row_created=trade_row_created,
                position_row_created_or_updated=position_row_created,
                stop_registered=stop_registered,
                target_registered=target_registered,
                thesis_invalidation_recorded=thesis_invalidation_recorded,
                position_monitor_armed=position_monitor_armed,
                review_lineage_linked=review_lineage_linked,
            )

            # Persist to candidate_lifecycle_checklists table
            conn.execute(_INSERT_CHECKLIST, {
                "candidate_id": candidate_id,
                "trade_id": trade_id,
                "cycle_id": cycle_id,
                "profile_id": profile_id,
                "trade_row_created": checklist.trade_row_created,
                "position_row_created_or_updated": checklist.position_row_created_or_updated,
                "stop_registered": checklist.stop_registered,
                "target_registered": checklist.target_registered,
                "thesis_invalidation_recorded": checklist.thesis_invalidation_recorded,
                "position_monitor_armed": checklist.position_monitor_armed,
                "review_lineage_linked": checklist.review_lineage_linked,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            conn.commit()

            logger.info(
                "Lifecycle checklist written for candidate=%s trade=%s complete=%s missing=%s",
                candidate_id,
                trade_id,
                checklist.complete,
                checklist.missing_components if not checklist.complete else "none",
            )

            return checklist

    except Exception:
        logger.error(
            "Lifecycle checklist write failed for candidate=%s trade=%s",
            candidate_id,
            trade_id,
            exc_info=True,
        )
        return None
