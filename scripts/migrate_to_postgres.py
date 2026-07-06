"""Migration script: Copy data from SQLite to Postgres.

Pre-flight checks ensure services are stopped, a clean SQLite snapshot
is created, and all tables exist in Postgres before data transfer begins.

The actual data copy (table-by-table transfer within a single Postgres
transaction) is implemented separately in _copy_table / migrate().
"""

import hashlib
import logging
import os
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, inspect as sa_inspect, text

log = logging.getLogger(__name__)

# LaunchAgent service labels that must be stopped before migration.
_SERVICES = [
    "com.paper-trader.orchestrator",
    "com.paper-trader.web",
]

# Default snapshot directory relative to project root.
_SNAPSHOTS_DIR = Path(__file__).resolve().parent.parent / "db" / "snapshots"


@dataclass
class ValidationReport:
    """Result of post-copy in-transaction validation checks.

    status is "PASS" if all checks passed, "FAIL" if any critical check failed.
    On FAIL the caller should raise MigrationValidationError to trigger rollback.
    """

    status: str  # "PASS" or "FAIL"
    row_counts: dict[str, tuple[int, int]]  # table -> (sqlite_count, pg_count)
    mismatched_tables: list[str] = field(default_factory=list)
    fk_integrity_errors: list[str] = field(default_factory=list)
    stale_reservations: int = 0
    duplicate_candidate_ids: int = 0
    snapshot_hash_match: bool = True
    checks: list[dict] = field(default_factory=list)  # individual check results


class MigrationValidationError(Exception):
    """Raised when in-transaction validation fails, triggering rollback."""

    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__(f"Migration validation failed: {report.mismatched_tables}")


def _verify_services_stopped() -> list[str]:
    """Confirm production services are not running.

    Checks macOS LaunchAgent status via ``launchctl list``.
    Returns list of blocking errors (empty = pass).
    """
    errors: list[str] = []
    for svc in _SERVICES:
        try:
            result = subprocess.run(
                ["launchctl", "list", svc],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                errors.append(
                    f"BLOCKING: service '{svc}' is still loaded. "
                    f"Run 'launchctl unload ~/Library/LaunchAgents/{svc}.plist' first."
                )
        except FileNotFoundError:
            # launchctl not available (e.g. running on Windows/Linux for dev).
            # This is only relevant on macOS production.
            log.warning(
                "launchctl not found — skipping service check for '%s'. "
                "This is expected on non-macOS systems.",
                svc,
            )
    return errors


def _create_snapshot(sqlite_path: str) -> tuple[str, str]:
    """Create a timestamped snapshot copy and compute its SHA-256 hash.

    Steps:
        1. Copy the SQLite file (and any -wal/-shm files) to
           db/snapshots/paper_trader_YYYYMMDD_HHMMSS.db
        2. Open the snapshot with sqlite3, run PRAGMA wal_checkpoint(TRUNCATE)
           to fold any WAL data into the main DB file, then close.
        3. Delete the snapshot's -wal and -shm files if present.
        4. Compute SHA-256 of the clean snapshot file.
        5. Return (snapshot_path, sha256_hex).

    The migration reads ONLY from the snapshot. The production file
    is never opened by the migration script, avoiding WAL/SHM issues.

    Args:
        sqlite_path: Path to the production SQLite database file.

    Returns:
        Tuple of (snapshot_path, sha256_hex_digest).

    Raises:
        FileNotFoundError: If the source SQLite file does not exist.
        OSError: If the snapshot directory cannot be created or files
            cannot be copied.
    """
    src = Path(sqlite_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"SQLite database not found: {src}")

    # Ensure snapshot directory exists.
    _SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build timestamped snapshot filename.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_name = f"paper_trader_{timestamp}.db"
    snapshot_path = _SNAPSHOTS_DIR / snapshot_name

    log.info("Creating snapshot: %s -> %s", src, snapshot_path)

    # Step 1: Copy main DB file and any WAL/SHM sidecar files.
    shutil.copy2(str(src), str(snapshot_path))

    wal_src = src.with_suffix(".db-wal")
    shm_src = src.with_suffix(".db-shm")
    if wal_src.exists():
        shutil.copy2(str(wal_src), str(snapshot_path.with_suffix(".db-wal")))
    if shm_src.exists():
        shutil.copy2(str(shm_src), str(snapshot_path.with_suffix(".db-shm")))

    # Step 2: Checkpoint WAL into the snapshot so all data is in the main file.
    conn = sqlite3.connect(str(snapshot_path))
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()

    # Step 3: Remove snapshot WAL/SHM files (data is now in main file).
    snapshot_wal = snapshot_path.with_suffix(".db-wal")
    snapshot_shm = snapshot_path.with_suffix(".db-shm")
    if snapshot_wal.exists():
        snapshot_wal.unlink()
    if snapshot_shm.exists():
        snapshot_shm.unlink()

    # Step 4: Compute SHA-256 of the clean snapshot.
    sha256_hex = _compute_sha256(str(snapshot_path))
    log.info("Snapshot SHA-256: %s", sha256_hex)

    return str(snapshot_path), sha256_hex


def _compute_sha256(file_path: str) -> str:
    """Compute the SHA-256 hex digest of a file.

    Reads in 64KB chunks to handle large files without excessive memory use.

    Args:
        file_path: Path to the file to hash.

    Returns:
        Lowercase hex digest string.
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _verify_table_coverage(snapshot_engine, postgres_engine) -> list[str]:
    """Verify all tables in the SQLite snapshot are accounted for on Postgres.

    Compares the actual snapshot's sqlite_master against the live Postgres
    catalog. Returns list of blocking errors (empty = pass).

    Args:
        snapshot_engine: SQLAlchemy engine connected to the SQLite snapshot.
        postgres_engine: SQLAlchemy engine connected to the target Postgres DB.

    Returns:
        List of blocking error messages. Empty list means all tables are
        present in Postgres.
    """
    # Get all user tables from SQLite snapshot.
    with snapshot_engine.connect() as conn:
        sqlite_tables = {
            row[0]
            for row in conn.execute(
                text(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            ).fetchall()
        }

    # Get all tables from Postgres.
    pg_inspector = sa_inspect(postgres_engine)
    pg_tables = set(pg_inspector.get_table_names())

    missing = sqlite_tables - pg_tables
    errors: list[str] = []
    for table in sorted(missing):
        errors.append(
            f"BLOCKING: table '{table}' exists in SQLite snapshot but not in "
            f"Postgres. Ensure its schema init module creates it on Postgres."
        )

    if not errors:
        log.info(
            "Table coverage check passed: %d tables present in both.",
            len(sqlite_tables),
        )
    else:
        log.error(
            "Table coverage check FAILED: %d table(s) missing from Postgres.",
            len(missing),
        )

    return errors


# Table copy order — respects FK dependencies so child rows are inserted
# after their parent rows.  TRUNCATE happens in reversed order with CASCADE.
TABLE_ORDER = [
    "trades",
    "positions",
    "balance",
    "agent_memory",
    "cases",
    "daily_log",
    "dynamic_strategies",
    "analyst_feedback_queue",
    "analyst_mitigations",
    "review_queue",
    "trade_events",
    "funnel_candidates",
    "funnel_run_logs",
    "pm_candidates",
    "pm_candidate_events",
    "checkpoint_events",
    "blocked_trade_candidates",
    "blocked_trade_candidate_outcomes",
    "pm_raw_responses",
    "response_lineage_links",
    "provenance_events",
    "provenance_findings",
    "decision_snapshots",
    "replay_audit_records",
    "replay_batch_runs",
    "replay_batch_items",
    "replay_annotations",
    "replay_counterfactual_outcomes",
    "alert_intents",
    "alert_cooldowns",
    "alert_dispatch_log",
    "pm_alert_claims",
    "pm_alert_events",
    "candidate_shadow_comparison",
]


def _normalize_row(row: dict) -> dict:
    """Normalize SQLite values for Postgres insertion.

    Minimal normalization — Postgres handles most conversions natively:
    - Boolean 0/1 → Python bool (SQLAlchemy/psycopg maps bool correctly
      when the target column is typed BOOLEAN in Postgres)
    - Datetime TEXT → kept as string (Postgres TIMESTAMPTZ parses ISO 8601)
    - None values pass through unchanged
    - JSON text passes through unchanged (Postgres TEXT/JSONB accepts strings)
    """
    normalized = {}
    for key, value in row.items():
        # SQLite stores booleans as 0/1 integers.  Convert to Python bool
        # so psycopg sends them as proper Postgres BOOLEAN values.
        # Heuristic: integer 0 or 1 in columns whose name suggests boolean
        # semantics.  We keep it simple — column-name based detection covers
        # the known schema (stop_hit, target_hit, first_hit, etc.).
        if isinstance(value, int) and value in (0, 1):
            # Only convert columns that are clearly boolean by naming.
            bool_prefixes = (
                "is_", "has_", "was_", "should_", "can_",
            )
            bool_suffixes = (
                "_hit", "_flag", "_active", "_enabled", "_disabled",
                "_completed", "_resolved",
            )
            bool_exact = (
                "stop_hit", "target_hit", "first_hit",
                "active", "resolved", "acknowledged",
                "diagnostic_mode", "expired", "no_data_reject",
                "payload_truncated",
            )
            lower_key = key.lower()
            if (
                any(lower_key.startswith(p) for p in bool_prefixes)
                or any(lower_key.endswith(s) for s in bool_suffixes)
                or lower_key in bool_exact
            ):
                normalized[key] = bool(value)
                continue
        normalized[key] = value
    return normalized


def _copy_table(sqlite_conn, pg_conn, table: str) -> int:
    """Copy all rows from a SQLite table into the corresponding Postgres table.

    Reads every row from the SQLite snapshot and bulk-inserts them into
    Postgres with data-type normalization applied per-row.

    Args:
        sqlite_conn: An active SQLAlchemy connection to the SQLite snapshot.
        pg_conn: An active SQLAlchemy connection within a Postgres transaction.
        table: The table name to copy.

    Returns:
        The number of rows copied (0 if source table is empty).
    """
    rows = sqlite_conn.execute(text(f"SELECT * FROM {table}")).mappings().all()
    if not rows:
        log.info("  %s: 0 rows (empty)", table)
        return 0

    # Build parameterized INSERT from the column names in the first row.
    columns = list(rows[0].keys())
    col_list = ", ".join(columns)
    placeholders = ", ".join(f":{col}" for col in columns)
    insert_sql = text(f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})")

    # Normalize each row for Postgres type compatibility.
    normalized_rows = [_normalize_row(dict(row)) for row in rows]

    pg_conn.execute(insert_sql, normalized_rows)
    log.info("  %s: %d rows copied", table, len(normalized_rows))
    return len(normalized_rows)


def _reset_sequence(pg_conn, table: str) -> None:
    """Reset the Postgres sequence for a table's ``id`` column to MAX(id).

    Uses ``pg_get_serial_sequence()`` to discover whether the table has a
    sequence attached to the ``id`` column.  If no sequence exists (e.g.
    the table uses UUID/string keys or has no ``id`` column), the function
    returns silently.

    If the table is empty (MAX(id) is NULL), the sequence is left unchanged.

    Args:
        pg_conn: An active SQLAlchemy connection within a Postgres transaction.
        table: The table name whose sequence should be reset.
    """
    # Check if the table has a sequence on the 'id' column.
    seq_result = pg_conn.execute(
        text("SELECT pg_get_serial_sequence(:table, 'id')"),
        {"table": table},
    ).scalar()

    if seq_result is None:
        # No sequence for this table — skip (UUID/string PK or no id column).
        return

    # Get the current max id value.
    max_id = pg_conn.execute(text(f"SELECT MAX(id) FROM {table}")).scalar()
    if max_id is None:
        # Table is empty — leave sequence at its current value.
        return

    pg_conn.execute(
        text(f"SELECT setval(:seq, :max_id)"),
        {"seq": seq_result, "max_id": max_id},
    )
    log.info("  %s: sequence reset to %d", table, max_id)


def _validate(sqlite_conn, pg_conn, snapshot_hash: str) -> ValidationReport:
    """Run all validation queries within the active Postgres transaction.

    Checks row counts, FK integrity, stale reservations, candidate_id
    uniqueness, and snapshot hash integrity. Returns a ValidationReport.
    If status is FAIL, the caller should raise MigrationValidationError
    to trigger rollback.

    Args:
        sqlite_conn: An active SQLAlchemy connection to the SQLite snapshot.
        pg_conn: An active SQLAlchemy connection within a Postgres transaction.
        snapshot_hash: The SHA-256 hex digest of the snapshot file computed
            before the migration started.

    Returns:
        ValidationReport with status "PASS" or "FAIL".
    """
    report = ValidationReport(status="PASS", row_counts={})

    # 1. Row count comparison for all tables.
    log.info("Validation: checking row counts...")
    for table in TABLE_ORDER:
        sqlite_count = sqlite_conn.execute(
            text(f"SELECT COUNT(*) FROM {table}")
        ).scalar()
        pg_count = pg_conn.execute(
            text(f"SELECT COUNT(*) FROM {table}")
        ).scalar()
        report.row_counts[table] = (sqlite_count, pg_count)
        if sqlite_count != pg_count:
            report.mismatched_tables.append(table)

    if report.mismatched_tables:
        report.checks.append({
            "name": "row_count_match",
            "status": "FAIL",
            "detail": f"Mismatched tables: {report.mismatched_tables}",
        })
        log.error(
            "Validation FAIL: row count mismatch in %s",
            report.mismatched_tables,
        )
    else:
        report.checks.append({
            "name": "row_count_match",
            "status": "PASS",
            "detail": f"All {len(TABLE_ORDER)} tables match.",
        })

    # 2. FK integrity: trade_events → trades (trade_id)
    log.info("Validation: checking FK integrity (trade_events → trades)...")
    orphan_trade_events = pg_conn.execute(text("""
        SELECT COUNT(*) FROM trade_events te
        WHERE te.trade_id IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM trades t WHERE t.id = te.trade_id)
    """)).scalar()
    if orphan_trade_events > 0:
        report.fk_integrity_errors.append(
            f"trade_events: {orphan_trade_events} rows reference non-existent trades"
        )

    # 3. FK integrity: trade_events → pm_candidates (candidate_lineage_id)
    log.info("Validation: checking FK integrity (trade_events → pm_candidates)...")
    orphan_candidates = pg_conn.execute(text("""
        SELECT COUNT(*) FROM trade_events te
        WHERE te.candidate_lineage_id IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM pm_candidates pc
              WHERE pc.candidate_id = te.candidate_lineage_id
          )
    """)).scalar()
    if orphan_candidates > 0:
        report.fk_integrity_errors.append(
            f"trade_events: {orphan_candidates} rows reference non-existent pm_candidates"
        )

    if report.fk_integrity_errors:
        report.checks.append({
            "name": "fk_integrity",
            "status": "FAIL",
            "detail": "; ".join(report.fk_integrity_errors),
        })
        log.error(
            "Validation FAIL: FK integrity errors: %s",
            report.fk_integrity_errors,
        )
    else:
        report.checks.append({
            "name": "fk_integrity",
            "status": "PASS",
            "detail": "All FK references are valid.",
        })

    # 4. Stale reservation detection.
    log.info("Validation: checking stale reservations...")
    stale = pg_conn.execute(text("""
        SELECT COUNT(*) FROM pm_candidates WHERE state = 'RESERVED'
    """)).scalar()
    report.stale_reservations = stale or 0

    if report.stale_reservations > 0:
        report.checks.append({
            "name": "stale_reservations",
            "status": "WARNING",
            "detail": (
                f"{report.stale_reservations} candidates in RESERVED state. "
                "Expected 0 when services are stopped."
            ),
        })
        log.warning(
            "Validation WARNING: %d candidates in RESERVED state.",
            report.stale_reservations,
        )
    else:
        report.checks.append({
            "name": "stale_reservations",
            "status": "PASS",
            "detail": "No stale reservations found.",
        })

    # 5. candidate_id uniqueness.
    log.info("Validation: checking candidate_id uniqueness...")
    duplicates = pg_conn.execute(text("""
        SELECT COUNT(*) FROM (
            SELECT candidate_id FROM pm_candidates
            GROUP BY candidate_id HAVING COUNT(*) > 1
        ) dups
    """)).scalar()
    report.duplicate_candidate_ids = duplicates or 0

    if report.duplicate_candidate_ids > 0:
        report.checks.append({
            "name": "candidate_id_uniqueness",
            "status": "FAIL",
            "detail": (
                f"{report.duplicate_candidate_ids} duplicate candidate_id groups found."
            ),
        })
        log.error(
            "Validation FAIL: %d duplicate candidate_id groups.",
            report.duplicate_candidate_ids,
        )
    else:
        report.checks.append({
            "name": "candidate_id_uniqueness",
            "status": "PASS",
            "detail": "All candidate_ids are unique.",
        })

    # 6. SHA-256 post-migration snapshot hash comparison.
    # Re-hash the snapshot file to confirm it was not modified during migration.
    log.info("Validation: verifying snapshot hash integrity...")
    # The sqlite_conn URL contains the snapshot path. Extract it.
    snapshot_url = str(sqlite_conn.engine.url)
    # URL format: sqlite:///path/to/file.db
    snapshot_path = snapshot_url.replace("sqlite:///", "")
    post_hash = _compute_sha256(snapshot_path)
    report.snapshot_hash_match = (post_hash == snapshot_hash)

    if not report.snapshot_hash_match:
        report.checks.append({
            "name": "snapshot_hash_integrity",
            "status": "FAIL",
            "detail": (
                f"Snapshot hash mismatch: pre={snapshot_hash[:16]}... "
                f"post={post_hash[:16]}... — file was modified during migration!"
            ),
        })
        log.error(
            "Validation FAIL: snapshot file modified during migration! "
            "pre=%s post=%s",
            snapshot_hash,
            post_hash,
        )
    else:
        report.checks.append({
            "name": "snapshot_hash_integrity",
            "status": "PASS",
            "detail": "Snapshot hash matches pre-migration value.",
        })

    # Determine overall status.
    if report.mismatched_tables:
        report.status = "FAIL"
    if report.fk_integrity_errors:
        report.status = "FAIL"
    if report.duplicate_candidate_ids > 0:
        report.status = "FAIL"
    if not report.snapshot_hash_match:
        report.status = "FAIL"
    # Note: stale_reservations is a WARNING, not a failure condition.

    log.info("Validation complete: status=%s", report.status)
    return report


def copy_all_tables(
    snapshot_path: str,
    postgres_url: str,
    snapshot_hash: str = "",
) -> tuple[dict[str, int], ValidationReport]:
    """Copy all tables from a SQLite snapshot into Postgres in a single transaction.

    The entire operation — truncation, row copies, sequence resets, and
    validation — is wrapped in one Postgres transaction.  If any step fails
    or validation reports FAIL, the transaction rolls back and Postgres is
    left unchanged.

    Args:
        snapshot_path: File path to the SQLite snapshot database.
        postgres_url: Postgres connection URL (e.g. from DATABASE_URL).
        snapshot_hash: SHA-256 hex digest of the snapshot file computed
            before migration. Used for post-migration integrity verification.

    Returns:
        Tuple of (row_counts dict, ValidationReport).

    Raises:
        MigrationValidationError: If validation fails (triggers rollback).
        Exception: Any failure during copy causes the transaction to roll back
            and the original exception propagates.
    """
    snapshot_engine = create_engine(f"sqlite:///{snapshot_path}")
    postgres_engine = create_engine(postgres_url, pool_pre_ping=True)

    row_counts: dict[str, int] = {}

    try:
        with snapshot_engine.connect() as sqlite_conn:
            with postgres_engine.begin() as pg_conn:
                # 1. Truncate all tables in reverse FK order for safety.
                log.info("Truncating %d Postgres tables...", len(TABLE_ORDER))
                for table in reversed(TABLE_ORDER):
                    pg_conn.execute(text(f"TRUNCATE TABLE {table} CASCADE"))

                # 2. Copy rows table-by-table in FK-dependency order.
                log.info("Copying rows from SQLite snapshot...")
                for table in TABLE_ORDER:
                    count = _copy_table(sqlite_conn, pg_conn, table)
                    row_counts[table] = count

                # 3. Reset sequences for tables with integer auto-increment PKs.
                log.info("Resetting Postgres sequences...")
                for table in TABLE_ORDER:
                    _reset_sequence(pg_conn, table)

                # 4. Validate within the same transaction.
                log.info("Running in-transaction validation...")
                report = _validate(sqlite_conn, pg_conn, snapshot_hash)
                if report.status == "FAIL":
                    raise MigrationValidationError(report)

                log.info(
                    "All tables copied and validated successfully. "
                    "Total rows: %d across %d tables.",
                    sum(row_counts.values()),
                    len(TABLE_ORDER),
                )
                # Transaction commits when pg_conn context exits normally.
    finally:
        snapshot_engine.dispose()
        postgres_engine.dispose()

    return row_counts, report


def run_preflight(
    sqlite_path: str = "db/paper_trader.db",
    postgres_url: str | None = None,
    skip_service_check: bool = False,
) -> tuple[str, str, list[str]]:
    """Execute all pre-flight checks before migration.

    Args:
        sqlite_path: Path to the production SQLite database.
        postgres_url: Postgres connection URL. If None, reads DATABASE_URL
            from the environment.
        skip_service_check: If True, skip the LaunchAgent service check.
            Useful for development/testing on non-macOS systems.

    Returns:
        Tuple of (snapshot_path, sha256_hex, errors) where errors is a list
        of blocking error messages. If errors is non-empty, migration must
        not proceed.

    Raises:
        FileNotFoundError: If the SQLite database file does not exist.
        ValueError: If no Postgres URL is provided or found in environment.
    """
    errors: list[str] = []

    # 1. Service check (unless skipped).
    if not skip_service_check:
        service_errors = _verify_services_stopped()
        errors.extend(service_errors)
        if service_errors:
            log.error("Service check failed: %s", service_errors)
    else:
        log.info("Service check skipped (skip_service_check=True).")

    # 2. Resolve Postgres URL.
    pg_url = postgres_url or os.environ.get("DATABASE_URL", "").strip()
    if not pg_url:
        raise ValueError(
            "No Postgres URL provided. Set DATABASE_URL or pass postgres_url."
        )

    # 3. Create snapshot.
    snapshot_path, sha256_hex = _create_snapshot(sqlite_path)

    # 4. Table coverage check.
    snapshot_engine = create_engine(f"sqlite:///{snapshot_path}")
    postgres_engine = create_engine(pg_url, pool_pre_ping=True)
    try:
        table_errors = _verify_table_coverage(snapshot_engine, postgres_engine)
        errors.extend(table_errors)
    finally:
        snapshot_engine.dispose()
        postgres_engine.dispose()

    if errors:
        log.error("Pre-flight checks FAILED with %d error(s).", len(errors))
    else:
        log.info("All pre-flight checks passed.")

    return snapshot_path, sha256_hex, errors


def _print_report(report: ValidationReport, row_counts: dict[str, int]) -> None:
    """Print a human-readable validation report to stdout."""
    print(f"\n{'='*60}")
    print(f"  Migration Validation Report: {report.status}")
    print(f"{'='*60}")

    total_rows = sum(row_counts.values())
    print(f"\n  Tables migrated: {len(row_counts)}")
    print(f"  Total rows: {total_rows:,}")

    if report.mismatched_tables:
        print(f"\n  \u2717 Row count mismatches:")
        for t in report.mismatched_tables:
            sqlite_c, pg_c = report.row_counts[t]
            print(f"      {t}: SQLite={sqlite_c} Postgres={pg_c}")

    if report.fk_integrity_errors:
        print(f"\n  \u2717 FK integrity errors:")
        for err in report.fk_integrity_errors:
            print(f"      {err}")

    if report.stale_reservations > 0:
        print(f"\n  \u26a0 Stale reservations: {report.stale_reservations}")

    if report.duplicate_candidate_ids > 0:
        print(f"\n  \u2717 Duplicate candidate_ids: {report.duplicate_candidate_ids}")

    snapshot_symbol = "\u2713" if report.snapshot_hash_match else "\u2717"
    print(f"\n  Snapshot hash match: {snapshot_symbol}")

    print(f"\n  Checks:")
    for check in report.checks:
        symbol = "\u2713" if check["status"] == "PASS" else ("\u26a0" if check["status"] == "WARNING" else "\u2717")
        print(f"    {symbol} {check['name']}: {check['detail']}")

    print(f"\n{'='*60}\n")


def migrate(
    sqlite_path: str = "db/paper_trader.db",
    postgres_url: str | None = None,
    skip_service_check: bool = False,
) -> ValidationReport:
    """Run the full migration: pre-flight \u2192 snapshot \u2192 copy \u2192 validate.

    Args:
        sqlite_path: Path to the production SQLite database.
        postgres_url: Postgres connection URL. If None, reads DATABASE_URL env var.
        skip_service_check: Skip LaunchAgent service checks (for dev/testing).

    Returns:
        ValidationReport on success.

    Raises:
        MigrationValidationError: If validation fails (Postgres is rolled back).
        ValueError: If no Postgres URL is available.
        FileNotFoundError: If SQLite database doesn't exist.
    """
    # 1. Pre-flight checks
    snapshot_path, snapshot_hash, errors = run_preflight(
        sqlite_path=sqlite_path,
        postgres_url=postgres_url,
        skip_service_check=skip_service_check,
    )
    if errors:
        print("Pre-flight checks FAILED:")
        for e in errors:
            print(f"  \u2717 {e}")
        raise RuntimeError(f"Pre-flight failed with {len(errors)} error(s)")

    # 2. Copy data (single transaction with validation)
    pg_url = postgres_url or os.environ.get("DATABASE_URL", "").strip()
    row_counts, report = copy_all_tables(
        snapshot_path=snapshot_path,
        postgres_url=pg_url,
        snapshot_hash=snapshot_hash,
    )

    # 3. Print report
    _print_report(report, row_counts)
    return report


if __name__ == "__main__":
    import argparse
    import sys

    # Add project root to path for imports when run directly.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    parser = argparse.ArgumentParser(
        description="Migrate paper-trader from SQLite to Postgres"
    )
    parser.add_argument(
        "--sqlite-path",
        default="db/paper_trader.db",
        help="Path to the production SQLite database (default: db/paper_trader.db)",
    )
    parser.add_argument(
        "--postgres-url",
        default=None,
        help="Postgres connection URL (default: reads DATABASE_URL from environment)",
    )
    parser.add_argument(
        "--skip-service-check",
        action="store_true",
        help="Skip LaunchAgent service checks (for development/testing)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        report = migrate(
            sqlite_path=args.sqlite_path,
            postgres_url=args.postgres_url,
            skip_service_check=args.skip_service_check,
        )
        sys.exit(0 if report.status == "PASS" else 1)
    except MigrationValidationError as e:
        print(f"\nMigration ROLLED BACK: validation failed")
        _print_report(e.report, {})
        sys.exit(1)
    except Exception as e:
        print(f"\nMigration FAILED: {e}")
        sys.exit(2)
