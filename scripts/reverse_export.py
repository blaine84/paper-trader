"""Reverse-export: selectively copy post-cutover data from Postgres back to SQLite.

Used during rollback when new data has been written to Postgres after cutover.
Only copies rows from specific tables that may have accumulated new data:
- trades (new trades executed while on Postgres)
- trade_events (trade lifecycle events)
- pm_candidate_events (candidate lifecycle events)
- checkpoint_events (checkpoint log entries)
- pm_alert_events (alert audit events)

Usage:
    python scripts/reverse_export.py --sqlite-path db/paper_trader.db
    python scripts/reverse_export.py --sqlite-path db/paper_trader.db --tables trades,trade_events
    python scripts/reverse_export.py --sqlite-path db/paper_trader.db --postgres-url postgresql+psycopg://...
"""

import logging
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect as sa_inspect, text

log = logging.getLogger(__name__)

# Tables that may accumulate new data after cutover to Postgres.
DEFAULT_TABLES = [
    "trades",
    "trade_events",
    "pm_candidate_events",
    "checkpoint_events",
    "pm_alert_events",
]


def _get_primary_key_columns(engine, table: str) -> list[str]:
    """Get the primary key column(s) for a table using SQLAlchemy inspector.

    Args:
        engine: SQLAlchemy engine to inspect.
        table: Table name.

    Returns:
        List of primary key column names. Empty list if table not found.
    """
    inspector = sa_inspect(engine)
    try:
        pk = inspector.get_pk_constraint(table)
        return pk.get("constrained_columns", [])
    except Exception:
        return []


def _get_common_columns(pg_engine, sqlite_engine, table: str) -> list[str]:
    """Find columns that exist in both Postgres and SQLite for a given table.

    This handles the case where Postgres may have columns that SQLite doesn't
    (e.g., added after cutover). Only columns present in both are exported.

    Args:
        pg_engine: SQLAlchemy engine connected to Postgres.
        sqlite_engine: SQLAlchemy engine connected to SQLite.
        table: Table name.

    Returns:
        List of column names present in both databases.
    """
    pg_inspector = sa_inspect(pg_engine)
    sqlite_inspector = sa_inspect(sqlite_engine)

    pg_columns = {col["name"] for col in pg_inspector.get_columns(table)}
    sqlite_columns = {col["name"] for col in sqlite_inspector.get_columns(table)}

    common = sorted(pg_columns & sqlite_columns)
    if pg_columns - sqlite_columns:
        log.warning(
            "  %s: skipping Postgres-only columns: %s",
            table,
            sorted(pg_columns - sqlite_columns),
        )
    return common


def _export_table(pg_engine, sqlite_engine, table: str) -> int:
    """Export rows from Postgres that don't exist in SQLite for a single table.

    Finds rows in Postgres whose primary key values are NOT present in SQLite,
    then inserts them into SQLite using INSERT OR IGNORE for conflict safety.

    Args:
        pg_engine: SQLAlchemy engine connected to Postgres.
        sqlite_engine: SQLAlchemy engine connected to SQLite.
        table: Table name to export.

    Returns:
        Number of rows exported.

    Raises:
        ValueError: If the table has no primary key or doesn't exist.
    """
    # Get primary key columns from Postgres (authoritative schema).
    pk_cols = _get_primary_key_columns(pg_engine, table)
    if not pk_cols:
        raise ValueError(
            f"Table '{table}' has no primary key in Postgres — cannot determine "
            f"which rows are new. Skipping."
        )

    # Get columns common to both databases.
    common_columns = _get_common_columns(pg_engine, sqlite_engine, table)
    if not common_columns:
        raise ValueError(
            f"Table '{table}' has no common columns between Postgres and SQLite."
        )

    # Ensure all PK columns are in the common set.
    for pk_col in pk_cols:
        if pk_col not in common_columns:
            raise ValueError(
                f"Primary key column '{pk_col}' of table '{table}' is not present "
                f"in SQLite. Cannot perform reverse export."
            )

    # Get existing PK values from SQLite.
    pk_select = ", ".join(pk_cols)
    with sqlite_engine.connect() as sqlite_conn:
        existing_pks = set()
        rows = sqlite_conn.execute(
            text(f"SELECT {pk_select} FROM {table}")
        ).fetchall()
        for row in rows:
            existing_pks.add(tuple(row))

    log.info("  %s: %d existing rows in SQLite", table, len(existing_pks))

    # Read all rows from Postgres for the common columns.
    col_list = ", ".join(common_columns)
    with pg_engine.connect() as pg_conn:
        pg_rows = pg_conn.execute(
            text(f"SELECT {col_list} FROM {table}")
        ).mappings().all()

    # Filter to only rows whose PK is NOT in SQLite.
    new_rows = []
    for row in pg_rows:
        pk_value = tuple(row[col] for col in pk_cols)
        if pk_value not in existing_pks:
            new_rows.append(dict(row))

    if not new_rows:
        log.info("  %s: no new rows to export", table)
        return 0

    # Insert new rows into SQLite using INSERT OR IGNORE for safety.
    placeholders = ", ".join(f":{col}" for col in common_columns)
    insert_sql = text(
        f"INSERT OR IGNORE INTO {table} ({col_list}) VALUES ({placeholders})"
    )

    with sqlite_engine.begin() as sqlite_conn:
        sqlite_conn.execute(insert_sql, new_rows)

    log.info("  %s: exported %d new rows", table, len(new_rows))
    return len(new_rows)


def reverse_export(
    sqlite_path: str,
    postgres_url: str | None = None,
    tables: list[str] | None = None,
) -> dict[str, int]:
    """Export post-cutover data from Postgres back to SQLite.

    Selectively copies rows that exist in Postgres but not in SQLite for
    the specified tables. Uses primary key comparison to identify new rows
    and INSERT OR IGNORE for conflict-safe insertion.

    Args:
        sqlite_path: Path to the SQLite database file.
        postgres_url: Postgres connection URL. If None, reads DATABASE_URL
            from the environment.
        tables: List of table names to export. If None, uses DEFAULT_TABLES.

    Returns:
        Dictionary mapping table name to number of rows exported.

    Raises:
        FileNotFoundError: If the SQLite database file does not exist.
        ValueError: If no Postgres URL is provided or found in environment.
    """
    # Validate SQLite path.
    sqlite_file = Path(sqlite_path).resolve()
    if not sqlite_file.exists():
        raise FileNotFoundError(f"SQLite database not found: {sqlite_file}")

    # Resolve Postgres URL.
    pg_url = postgres_url or os.environ.get("DATABASE_URL", "").strip()
    if not pg_url:
        raise ValueError(
            "No Postgres URL provided. Set DATABASE_URL or pass --postgres-url."
        )

    # Resolve table list.
    export_tables = tables if tables is not None else DEFAULT_TABLES

    log.info("Reverse export: Postgres → SQLite (%s)", sqlite_path)
    log.info("Tables to export: %s", export_tables)

    # Create engines.
    pg_engine = create_engine(pg_url, pool_pre_ping=True)
    sqlite_engine = create_engine(
        f"sqlite:///{sqlite_file}",
        connect_args={"timeout": 30},
    )

    # Verify Postgres is reachable.
    try:
        with pg_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        raise ConnectionError(f"Cannot connect to Postgres: {e}") from e

    # Verify tables exist in both databases.
    pg_inspector = sa_inspect(pg_engine)
    sqlite_inspector = sa_inspect(sqlite_engine)
    pg_tables = set(pg_inspector.get_table_names())
    sqlite_tables = set(sqlite_inspector.get_table_names())

    results: dict[str, int] = {}

    for table in export_tables:
        if table not in pg_tables:
            log.warning("  %s: not found in Postgres — skipping", table)
            results[table] = 0
            continue
        if table not in sqlite_tables:
            log.warning("  %s: not found in SQLite — skipping", table)
            results[table] = 0
            continue

        try:
            count = _export_table(pg_engine, sqlite_engine, table)
            results[table] = count
        except ValueError as e:
            log.warning("  %s: %s", table, e)
            results[table] = 0
        except Exception as e:
            log.error("  %s: export failed — %s", table, e)
            results[table] = 0

    # Cleanup.
    pg_engine.dispose()
    sqlite_engine.dispose()

    # Summary.
    total = sum(results.values())
    log.info("Reverse export complete: %d total rows exported", total)
    return results


def _print_summary(results: dict[str, int]) -> None:
    """Print a human-readable summary of the reverse export."""
    print(f"\n{'='*50}")
    print("  Reverse Export Summary")
    print(f"{'='*50}")
    total = 0
    for table, count in results.items():
        status = f"{count} rows" if count > 0 else "no new rows"
        symbol = "\u2713" if count > 0 else "-"
        print(f"  {symbol} {table}: {status}")
        total += count
    print(f"\n  Total rows exported: {total}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    import argparse

    # Add project root to path for imports when run directly.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    parser = argparse.ArgumentParser(
        description=(
            "Reverse-export post-cutover data from Postgres back to SQLite. "
            "Copies only rows that exist in Postgres but not in SQLite."
        )
    )
    parser.add_argument(
        "--sqlite-path",
        required=True,
        help="Path to the SQLite database file",
    )
    parser.add_argument(
        "--postgres-url",
        default=None,
        help="Postgres connection URL (default: reads DATABASE_URL from environment)",
    )
    parser.add_argument(
        "--tables",
        default=None,
        help=(
            "Comma-separated list of tables to export "
            f"(default: {','.join(DEFAULT_TABLES)})"
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging output",
    )

    args = parser.parse_args()

    # Configure logging.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Parse tables argument.
    table_list = None
    if args.tables:
        table_list = [t.strip() for t in args.tables.split(",") if t.strip()]

    try:
        results = reverse_export(
            sqlite_path=args.sqlite_path,
            postgres_url=args.postgres_url,
            tables=table_list,
        )
        _print_summary(results)
    except (FileNotFoundError, ValueError, ConnectionError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        log.exception("Unexpected error during reverse export")
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        sys.exit(2)
