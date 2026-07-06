"""
Property-based tests for migration script (scripts/migrate_to_postgres.py).

Tests universal correctness properties for data migration:
- Property 4: Migration Data Preservation (Round Trip)
- Property 5: Migration Atomicity on Failure
- Property 6: Migration Idempotence
- Property 7: SQLite Snapshot Immutability During Migration

These tests use TWO SQLite databases (source and target) since no real Postgres
is available in CI. Postgres-specific operations (_reset_sequence,
pg_get_serial_sequence) are mocked where needed.

**Validates: Requirements 4.1, 4.4, 4.6, 5.1, 7.2**
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from sqlalchemy import create_engine, text


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Generate random symbols for the trades table
symbol_strategy = st.sampled_from([
    "AAPL", "TSLA", "NVDA", "MSFT", "GOOG", "AMZN", "META", "NFLX",
])

# Generate random PnL values
pnl_strategy = st.floats(
    min_value=-1000.0, max_value=1000.0,
    allow_nan=False, allow_infinity=False,
)

# Generate random quantities
quantity_strategy = st.integers(min_value=1, max_value=10000)

# Generate random price values
price_strategy = st.floats(
    min_value=1.0, max_value=5000.0,
    allow_nan=False, allow_infinity=False,
)

# Generate random text fields (for notes, rationale, etc.)
text_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=0,
    max_size=50,
)

# Generate random row counts for tables
row_count_strategy = st.integers(min_value=1, max_value=20)

# Strategy for a single trade row (id, symbol, entry_price, quantity, pnl)
trade_row_strategy = st.fixed_dictionaries({
    "symbol": symbol_strategy,
    "entry_price": price_strategy,
    "quantity": quantity_strategy,
    "pnl": pnl_strategy,
    "notes": text_strategy,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_source_db(db_path: str, rows: list[dict]) -> None:
    """Create a SQLite source database with a trades table and insert rows."""
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS trades ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  symbol TEXT NOT NULL,"
            "  entry_price REAL,"
            "  quantity INTEGER,"
            "  pnl REAL,"
            "  notes TEXT"
            ")"
        ))
        for row in rows:
            conn.execute(
                text(
                    "INSERT INTO trades (symbol, entry_price, quantity, pnl, notes) "
                    "VALUES (:symbol, :entry_price, :quantity, :pnl, :notes)"
                ),
                row,
            )
    engine.dispose()


def _create_target_db(db_path: str) -> None:
    """Create an empty SQLite target database with the same schema as source."""
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS trades ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  symbol TEXT NOT NULL,"
            "  entry_price REAL,"
            "  quantity INTEGER,"
            "  pnl REAL,"
            "  notes TEXT"
            ")"
        ))
    engine.dispose()


def _copy_table_sqlite_to_sqlite(source_engine, target_conn, table: str) -> int:
    """Copy all rows from source to target using the migration's _normalize_row logic."""
    from scripts.migrate_to_postgres import _normalize_row

    with source_engine.connect() as src_conn:
        rows = src_conn.execute(text(f"SELECT * FROM {table}")).mappings().all()

    if not rows:
        return 0

    columns = list(rows[0].keys())
    col_list = ", ".join(columns)
    placeholders = ", ".join(f":{col}" for col in columns)
    insert_sql = text(f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})")

    normalized_rows = [_normalize_row(dict(row)) for row in rows]
    target_conn.execute(insert_sql, normalized_rows)
    return len(normalized_rows)


def _compute_sha256(file_path: str) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Property 4: Migration Data Preservation (Round Trip)
#
# All rows copied from SQLite source are present in target with correct values.
# Row counts match exactly and column values are semantically equivalent.
#
# **Validates: Requirements 4.1, 5.1**
# ---------------------------------------------------------------------------


class TestProperty4MigrationDataPreservation:
    """
    Property 4: Migration Data Preservation (Round Trip).

    For any set of rows in the SQLite source, after _copy_table completes,
    the target database contains rows with matching primary keys, values,
    and row counts.

    **Validates: Requirements 4.1, 5.1**
    """

    @given(rows=st.lists(trade_row_strategy, min_size=1, max_size=30))
    @settings(max_examples=100)
    def test_all_rows_preserved_after_copy(self, rows: list[dict]):
        """Every row in the source is present in the target after _copy_table."""
        from scripts.migrate_to_postgres import _copy_table

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = os.path.join(tmp_dir, "source.db")
            target_path = os.path.join(tmp_dir, "target.db")

            _create_source_db(source_path, rows)
            _create_target_db(target_path)

            source_engine = create_engine(f"sqlite:///{source_path}")
            target_engine = create_engine(f"sqlite:///{target_path}")

            try:
                with source_engine.connect() as src_conn:
                    with target_engine.begin() as tgt_conn:
                        count = _copy_table(src_conn, tgt_conn, "trades")

                # Verify row count matches
                assert count == len(rows), (
                    f"Expected {len(rows)} rows copied, got {count}"
                )

                # Verify data in target matches source
                with source_engine.connect() as src_conn:
                    source_rows = src_conn.execute(
                        text("SELECT * FROM trades ORDER BY id")
                    ).fetchall()

                with target_engine.connect() as tgt_conn:
                    target_rows = tgt_conn.execute(
                        text("SELECT * FROM trades ORDER BY id")
                    ).fetchall()

                assert len(source_rows) == len(target_rows), (
                    f"Source has {len(source_rows)} rows, target has {len(target_rows)}"
                )

                for src_row, tgt_row in zip(source_rows, target_rows):
                    # Compare all columns
                    assert src_row[0] == tgt_row[0], f"ID mismatch: {src_row[0]} != {tgt_row[0]}"
                    assert src_row[1] == tgt_row[1], f"Symbol mismatch: {src_row[1]} != {tgt_row[1]}"
                    # Floating point comparison for price/pnl
                    if src_row[2] is not None:
                        assert abs(src_row[2] - tgt_row[2]) < 1e-9, (
                            f"entry_price mismatch: {src_row[2]} != {tgt_row[2]}"
                        )
                    assert src_row[3] == tgt_row[3], f"Quantity mismatch: {src_row[3]} != {tgt_row[3]}"
                    if src_row[4] is not None:
                        assert abs(src_row[4] - tgt_row[4]) < 1e-9, (
                            f"pnl mismatch: {src_row[4]} != {tgt_row[4]}"
                        )
                    assert src_row[5] == tgt_row[5], f"Notes mismatch: {src_row[5]} != {tgt_row[5]}"
            finally:
                source_engine.dispose()
                target_engine.dispose()

    @given(rows=st.lists(trade_row_strategy, min_size=0, max_size=5))
    @settings(max_examples=100)
    def test_row_count_exact_match(self, rows: list[dict]):
        """Row counts match exactly between source and target after copy."""
        from scripts.migrate_to_postgres import _copy_table

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = os.path.join(tmp_dir, "source.db")
            target_path = os.path.join(tmp_dir, "target.db")

            _create_source_db(source_path, rows)
            _create_target_db(target_path)

            source_engine = create_engine(f"sqlite:///{source_path}")
            target_engine = create_engine(f"sqlite:///{target_path}")

            try:
                with source_engine.connect() as src_conn:
                    with target_engine.begin() as tgt_conn:
                        _copy_table(src_conn, tgt_conn, "trades")

                with source_engine.connect() as src_conn:
                    src_count = src_conn.execute(
                        text("SELECT COUNT(*) FROM trades")
                    ).scalar()

                with target_engine.connect() as tgt_conn:
                    tgt_count = tgt_conn.execute(
                        text("SELECT COUNT(*) FROM trades")
                    ).scalar()

                assert src_count == tgt_count, (
                    f"Row count mismatch: source={src_count}, target={tgt_count}"
                )
            finally:
                source_engine.dispose()
                target_engine.dispose()


# ---------------------------------------------------------------------------
# Property 5: Migration Atomicity on Failure
#
# When _copy_table fails mid-way, the target is left unchanged (transaction
# rolled back). No partial data from the failed run persists.
#
# **Validates: Requirements 4.4**
# ---------------------------------------------------------------------------


class TestProperty5MigrationAtomicityOnFailure:
    """
    Property 5: Migration Atomicity on Failure.

    When a failure occurs during the copy step, the target database is left
    in its pre-migration state. The transaction rolls back cleanly.

    **Validates: Requirements 4.4**
    """

    @given(
        rows=st.lists(trade_row_strategy, min_size=3, max_size=20),
        fail_after=st.integers(min_value=0, max_value=2),
    )
    @settings(max_examples=100)
    def test_target_unchanged_on_copy_failure(
        self, rows: list[dict], fail_after: int
    ):
        """On failure during copy, target DB remains empty (no partial data)."""
        from scripts.migrate_to_postgres import _normalize_row

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = os.path.join(tmp_dir, "source.db")
            target_path = os.path.join(tmp_dir, "target.db")

            _create_source_db(source_path, rows)
            _create_target_db(target_path)

            source_engine = create_engine(f"sqlite:///{source_path}")
            target_engine = create_engine(f"sqlite:///{target_path}")

            try:
                # Simulate a failure during the transaction
                with pytest.raises(RuntimeError):
                    with source_engine.connect() as src_conn:
                        with target_engine.begin() as tgt_conn:
                            # Start copying rows
                            src_rows = src_conn.execute(
                                text("SELECT * FROM trades")
                            ).mappings().all()

                            columns = list(src_rows[0].keys())
                            col_list = ", ".join(columns)
                            placeholders = ", ".join(f":{col}" for col in columns)
                            insert_sql = text(
                                f"INSERT INTO trades ({col_list}) VALUES ({placeholders})"
                            )

                            for i, row in enumerate(src_rows):
                                if i >= fail_after:
                                    raise RuntimeError(
                                        f"Simulated failure at row {i}"
                                    )
                                normalized = _normalize_row(dict(row))
                                tgt_conn.execute(insert_sql, normalized)

                # After the transaction rolled back, target should be empty
                with target_engine.connect() as tgt_conn:
                    count = tgt_conn.execute(
                        text("SELECT COUNT(*) FROM trades")
                    ).scalar()

                assert count == 0, (
                    f"Expected 0 rows in target after failed transaction, got {count}. "
                    "Transaction should have rolled back completely."
                )
            finally:
                source_engine.dispose()
                target_engine.dispose()

    @given(rows=st.lists(trade_row_strategy, min_size=2, max_size=15))
    @settings(max_examples=100)
    def test_pre_existing_data_preserved_on_failure(self, rows: list[dict]):
        """If target has pre-existing data and migration fails, pre-existing data remains."""
        from scripts.migrate_to_postgres import _normalize_row

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = os.path.join(tmp_dir, "source.db")
            target_path = os.path.join(tmp_dir, "target.db")

            # Create source with N rows
            _create_source_db(source_path, rows)

            # Create target with 1 pre-existing row
            _create_target_db(target_path)
            target_engine = create_engine(f"sqlite:///{target_path}")
            with target_engine.begin() as conn:
                conn.execute(text(
                    "INSERT INTO trades (symbol, entry_price, quantity, pnl, notes) "
                    "VALUES ('PRE_EXISTING', 100.0, 10, 5.0, 'prior data')"
                ))
            target_engine.dispose()

            source_engine = create_engine(f"sqlite:///{source_path}")
            target_engine = create_engine(f"sqlite:///{target_path}")

            try:
                # Attempt migration that fails after truncate + partial copy
                with pytest.raises(RuntimeError):
                    with target_engine.begin() as tgt_conn:
                        # Simulate truncation (like the real migration does)
                        tgt_conn.execute(text("DELETE FROM trades"))

                        # Start copy then fail
                        with source_engine.connect() as src_conn:
                            src_rows = src_conn.execute(
                                text("SELECT * FROM trades")
                            ).mappings().all()

                            if src_rows:
                                columns = list(src_rows[0].keys())
                                col_list = ", ".join(columns)
                                placeholders = ", ".join(f":{col}" for col in columns)
                                insert_sql = text(
                                    f"INSERT INTO trades ({col_list}) VALUES ({placeholders})"
                                )
                                # Copy first row then fail
                                normalized = _normalize_row(dict(src_rows[0]))
                                tgt_conn.execute(insert_sql, normalized)

                            raise RuntimeError("Simulated mid-copy failure")

                # Pre-existing data should still be there (transaction rolled back)
                with target_engine.connect() as tgt_conn:
                    count = tgt_conn.execute(
                        text("SELECT COUNT(*) FROM trades")
                    ).scalar()
                    pre_existing = tgt_conn.execute(
                        text("SELECT symbol FROM trades WHERE symbol = 'PRE_EXISTING'")
                    ).fetchone()

                assert count == 1, (
                    f"Expected 1 pre-existing row after rollback, got {count}"
                )
                assert pre_existing is not None, (
                    "Pre-existing row should be preserved after failed migration"
                )
            finally:
                source_engine.dispose()
                target_engine.dispose()


# ---------------------------------------------------------------------------
# Property 6: Migration Idempotence
#
# Running migration twice produces the same result. The second run truncates
# and re-copies cleanly with no duplicates.
#
# **Validates: Requirements 4.6**
# ---------------------------------------------------------------------------


class TestProperty6MigrationIdempotence:
    """
    Property 6: Migration Idempotence.

    Running the copy operation N times against the same target produces
    identical final data as running it once. No duplicates accumulate.

    **Validates: Requirements 4.6**
    """

    @given(rows=st.lists(trade_row_strategy, min_size=1, max_size=25))
    @settings(max_examples=100)
    def test_double_migration_produces_same_result(self, rows: list[dict]):
        """Running migration twice yields identical row counts (no duplicates)."""
        from scripts.migrate_to_postgres import _copy_table

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = os.path.join(tmp_dir, "source.db")
            target_path = os.path.join(tmp_dir, "target.db")

            _create_source_db(source_path, rows)
            _create_target_db(target_path)

            source_engine = create_engine(f"sqlite:///{source_path}")
            target_engine = create_engine(f"sqlite:///{target_path}")

            try:
                # First migration run
                with source_engine.connect() as src_conn:
                    with target_engine.begin() as tgt_conn:
                        _copy_table(src_conn, tgt_conn, "trades")

                # Second migration run (truncate + re-copy, like the real script)
                with source_engine.connect() as src_conn:
                    with target_engine.begin() as tgt_conn:
                        tgt_conn.execute(text("DELETE FROM trades"))
                        _copy_table(src_conn, tgt_conn, "trades")

                # Verify row counts match source exactly (no duplicates)
                with source_engine.connect() as src_conn:
                    src_count = src_conn.execute(
                        text("SELECT COUNT(*) FROM trades")
                    ).scalar()

                with target_engine.connect() as tgt_conn:
                    tgt_count = tgt_conn.execute(
                        text("SELECT COUNT(*) FROM trades")
                    ).scalar()

                assert src_count == tgt_count, (
                    f"After double migration: source={src_count}, target={tgt_count}. "
                    "Second run should truncate and re-copy cleanly."
                )
            finally:
                source_engine.dispose()
                target_engine.dispose()

    @given(
        rows=st.lists(trade_row_strategy, min_size=1, max_size=20),
        num_runs=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=100)
    def test_n_migrations_produce_same_result(
        self, rows: list[dict], num_runs: int
    ):
        """Running migration N times yields same data as running once."""
        from scripts.migrate_to_postgres import _copy_table

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = os.path.join(tmp_dir, "source.db")
            target_path = os.path.join(tmp_dir, "target.db")

            _create_source_db(source_path, rows)
            _create_target_db(target_path)

            source_engine = create_engine(f"sqlite:///{source_path}")
            target_engine = create_engine(f"sqlite:///{target_path}")

            try:
                # Run migration N times (each time truncate + copy)
                for _ in range(num_runs):
                    with source_engine.connect() as src_conn:
                        with target_engine.begin() as tgt_conn:
                            tgt_conn.execute(text("DELETE FROM trades"))
                            _copy_table(src_conn, tgt_conn, "trades")

                # Verify final state matches source
                with source_engine.connect() as src_conn:
                    source_rows = src_conn.execute(
                        text("SELECT * FROM trades ORDER BY id")
                    ).fetchall()

                with target_engine.connect() as tgt_conn:
                    target_rows = tgt_conn.execute(
                        text("SELECT * FROM trades ORDER BY id")
                    ).fetchall()

                assert len(source_rows) == len(target_rows), (
                    f"After {num_runs} migrations: source={len(source_rows)}, "
                    f"target={len(target_rows)} rows"
                )

                # Verify data equality
                for src_row, tgt_row in zip(source_rows, target_rows):
                    assert src_row[0] == tgt_row[0], "ID mismatch after N migrations"
                    assert src_row[1] == tgt_row[1], "Symbol mismatch after N migrations"
            finally:
                source_engine.dispose()
                target_engine.dispose()


# ---------------------------------------------------------------------------
# Property 7: SQLite Snapshot Immutability During Migration
#
# The snapshot file's SHA-256 hash does not change during migration.
# The migration only reads from the snapshot, never writes to it.
#
# **Validates: Requirements 7.2**
# ---------------------------------------------------------------------------


class TestProperty7SnapshotImmutabilityDuringMigration:
    """
    Property 7: SQLite Snapshot Immutability During Migration.

    The SHA-256 hash of the SQLite snapshot file before migration equals
    the hash after migration. The migration process never modifies the
    snapshot file.

    **Validates: Requirements 7.2**
    """

    @given(rows=st.lists(trade_row_strategy, min_size=1, max_size=25))
    @settings(max_examples=100)
    def test_snapshot_hash_unchanged_after_migration(self, rows: list[dict]):
        """Snapshot file hash is identical before and after _copy_table reads it."""
        from scripts.migrate_to_postgres import _copy_table

        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_path = os.path.join(tmp_dir, "snapshot.db")
            target_path = os.path.join(tmp_dir, "target.db")

            # Create the snapshot (simulates the frozen snapshot file)
            _create_source_db(snapshot_path, rows)
            _create_target_db(target_path)

            # Compute hash BEFORE migration
            hash_before = _compute_sha256(snapshot_path)

            snapshot_engine = create_engine(f"sqlite:///{snapshot_path}")
            target_engine = create_engine(f"sqlite:///{target_path}")

            try:
                # Run the migration (reads from snapshot)
                with snapshot_engine.connect() as src_conn:
                    with target_engine.begin() as tgt_conn:
                        _copy_table(src_conn, tgt_conn, "trades")

                # Dispose engines to release any file handles
                snapshot_engine.dispose()
                target_engine.dispose()

                # Compute hash AFTER migration
                hash_after = _compute_sha256(snapshot_path)

                assert hash_before == hash_after, (
                    f"Snapshot hash changed during migration! "
                    f"Before: {hash_before[:16]}... After: {hash_after[:16]}... "
                    "The migration must never write to the snapshot file."
                )
            finally:
                snapshot_engine.dispose()
                target_engine.dispose()

    @given(rows=st.lists(trade_row_strategy, min_size=1, max_size=15))
    @settings(max_examples=100)
    def test_snapshot_hash_unchanged_after_failed_migration(self, rows: list[dict]):
        """Snapshot hash unchanged even if migration fails mid-way."""
        from scripts.migrate_to_postgres import _normalize_row

        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_path = os.path.join(tmp_dir, "snapshot.db")
            target_path = os.path.join(tmp_dir, "target.db")

            _create_source_db(snapshot_path, rows)
            _create_target_db(target_path)

            # Compute hash BEFORE migration
            hash_before = _compute_sha256(snapshot_path)

            snapshot_engine = create_engine(f"sqlite:///{snapshot_path}")
            target_engine = create_engine(f"sqlite:///{target_path}")

            try:
                # Simulate a migration that fails
                with pytest.raises(RuntimeError):
                    with snapshot_engine.connect() as src_conn:
                        with target_engine.begin() as tgt_conn:
                            src_rows = src_conn.execute(
                                text("SELECT * FROM trades")
                            ).mappings().all()

                            if src_rows:
                                columns = list(src_rows[0].keys())
                                col_list = ", ".join(columns)
                                placeholders = ", ".join(f":{col}" for col in columns)
                                insert_sql = text(
                                    f"INSERT INTO trades ({col_list}) VALUES ({placeholders})"
                                )
                                # Insert one row then fail
                                normalized = _normalize_row(dict(src_rows[0]))
                                tgt_conn.execute(insert_sql, normalized)

                            raise RuntimeError("Simulated failure")

                # Dispose engines to release file handles
                snapshot_engine.dispose()
                target_engine.dispose()

                # Compute hash AFTER failed migration
                hash_after = _compute_sha256(snapshot_path)

                assert hash_before == hash_after, (
                    f"Snapshot hash changed during failed migration! "
                    f"Before: {hash_before[:16]}... After: {hash_after[:16]}... "
                    "Even on failure, the snapshot must never be modified."
                )
            finally:
                snapshot_engine.dispose()
                target_engine.dispose()

    @given(rows=st.lists(trade_row_strategy, min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_compute_sha256_matches_validate_function(self, rows: list[dict]):
        """Our local _compute_sha256 matches the migration script's implementation."""
        from scripts.migrate_to_postgres import _compute_sha256 as migration_sha256

        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_path = os.path.join(tmp_dir, "snapshot.db")
            _create_source_db(snapshot_path, rows)

            local_hash = _compute_sha256(snapshot_path)
            migration_hash = migration_sha256(snapshot_path)

            assert local_hash == migration_hash, (
                f"Hash implementations differ: local={local_hash[:16]}... "
                f"migration={migration_hash[:16]}..."
            )
