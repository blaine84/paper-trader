"""Tests for db/replay_schema.py — replay namespace tables, triggers, and migrations."""

import pytest
from datetime import datetime
from sqlalchemy import create_engine, text, event, inspect

from db.replay_schema import init_replay_db


@pytest.fixture
def engine():
    """In-memory SQLite engine with existing production tables pre-created."""
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        # Create minimal versions of existing production tables so migrations can run
        conn.execute(text("""
            CREATE TABLE blocked_trade_candidates (
                id INTEGER PRIMARY KEY,
                symbol VARCHAR(10),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE funnel_candidates (
                id INTEGER PRIMARY KEY,
                candidate_id VARCHAR(36),
                symbol VARCHAR(10)
            )
        """))
        conn.execute(text("""
            CREATE TABLE trade_events (
                id INTEGER PRIMARY KEY,
                event_type VARCHAR(64),
                symbol VARCHAR(10)
            )
        """))
        conn.execute(text("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY,
                symbol VARCHAR(10),
                direction VARCHAR(5)
            )
        """))
        conn.execute(text("""
            CREATE TABLE pm_candidates (
                id INTEGER PRIMARY KEY,
                candidate_id VARCHAR(36),
                symbol VARCHAR(10)
            )
        """))
    return eng


@pytest.fixture
def initialized_engine(engine):
    """Engine with replay schema fully initialized."""
    init_replay_db(engine)
    return engine


class TestReplaySchemaCreation:
    """Verify all replay namespace tables are created."""

    def test_decision_snapshots_table_created(self, initialized_engine):
        inspector = inspect(initialized_engine)
        assert inspector.has_table("decision_snapshots")

    def test_replay_audit_records_table_created(self, initialized_engine):
        inspector = inspect(initialized_engine)
        assert inspector.has_table("replay_audit_records")

    def test_replay_batch_runs_table_created(self, initialized_engine):
        inspector = inspect(initialized_engine)
        assert inspector.has_table("replay_batch_runs")

    def test_replay_batch_items_table_created(self, initialized_engine):
        inspector = inspect(initialized_engine)
        assert inspector.has_table("replay_batch_items")

    def test_replay_annotations_table_created(self, initialized_engine):
        inspector = inspect(initialized_engine)
        assert inspector.has_table("replay_annotations")

    def test_replay_counterfactual_outcomes_table_created(self, initialized_engine):
        inspector = inspect(initialized_engine)
        assert inspector.has_table("replay_counterfactual_outcomes")


class TestDecisionSnapshotsImmutability:
    """Verify immutability triggers on decision_snapshots."""

    def test_insert_succeeds(self, initialized_engine):
        with initialized_engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO decision_snapshots (
                    snapshot_id, schema_version, candidate_lineage_id, timestamp,
                    symbol, profile, direction, decision_payload_json,
                    entry_price, stop_price, target_price, quantity,
                    account_equity, available_cash,
                    gate_config_json, feature_flags_json, policy_version_id
                ) VALUES (
                    'snap-001', '1.0', 'lin-001', '2024-01-15 10:00:00',
                    'TSLA', 'aggressive', 'LONG', '{}',
                    '185.50', '184.00', '190.00', '10',
                    '100000', '50000',
                    '{}', '{}', 'policy-v1'
                )
            """))

        with initialized_engine.connect() as conn:
            row = conn.execute(text(
                "SELECT snapshot_id FROM decision_snapshots WHERE snapshot_id = 'snap-001'"
            )).fetchone()
            assert row is not None
            assert row[0] == "snap-001"

    def test_update_prohibited(self, initialized_engine):
        with initialized_engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO decision_snapshots (
                    snapshot_id, schema_version, candidate_lineage_id, timestamp,
                    symbol, profile, direction, decision_payload_json,
                    entry_price, stop_price, target_price, quantity,
                    account_equity, available_cash,
                    gate_config_json, feature_flags_json, policy_version_id
                ) VALUES (
                    'snap-002', '1.0', 'lin-002', '2024-01-15 10:00:00',
                    'AAPL', 'moderate', 'LONG', '{}',
                    '150.00', '148.00', '155.00', '5',
                    '100000', '50000',
                    '{}', '{}', 'policy-v1'
                )
            """))

        with pytest.raises(Exception, match="immutable.*UPDATE prohibited"):
            with initialized_engine.begin() as conn:
                conn.execute(text("""
                    UPDATE decision_snapshots SET symbol = 'MSFT'
                    WHERE snapshot_id = 'snap-002'
                """))

    def test_delete_prohibited(self, initialized_engine):
        with initialized_engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO decision_snapshots (
                    snapshot_id, schema_version, candidate_lineage_id, timestamp,
                    symbol, profile, direction, decision_payload_json,
                    entry_price, stop_price, target_price, quantity,
                    account_equity, available_cash,
                    gate_config_json, feature_flags_json, policy_version_id
                ) VALUES (
                    'snap-003', '1.0', 'lin-003', '2024-01-15 10:00:00',
                    'GOOG', 'conservative', 'SHORT', '{}',
                    '140.00', '142.00', '135.00', '8',
                    '100000', '50000',
                    '{}', '{}', 'policy-v1'
                )
            """))

        with pytest.raises(Exception, match="immutable.*DELETE prohibited"):
            with initialized_engine.begin() as conn:
                conn.execute(text("""
                    DELETE FROM decision_snapshots WHERE snapshot_id = 'snap-003'
                """))


class TestReplayAuditRecordsImmutability:
    """Verify immutability triggers on replay_audit_records."""

    def _insert_audit_record(self, conn, replay_id="replay-001"):
        conn.execute(text(f"""
            INSERT INTO replay_audit_records (
                replay_id, candidate_id, source_candidate_ids_json,
                replay_cutoff, input_sources_json, policy_version_json,
                replay_status, era
            ) VALUES (
                '{replay_id}', 'cand-001', '[]',
                '2024-01-15 10:00:00', '{{}}', '{{}}',
                'exact', 'post-snapshot'
            )
        """))

    def test_insert_succeeds(self, initialized_engine):
        with initialized_engine.begin() as conn:
            self._insert_audit_record(conn)

        with initialized_engine.connect() as conn:
            row = conn.execute(text(
                "SELECT replay_id FROM replay_audit_records WHERE replay_id = 'replay-001'"
            )).fetchone()
            assert row is not None

    def test_update_prohibited(self, initialized_engine):
        with initialized_engine.begin() as conn:
            self._insert_audit_record(conn, "replay-upd")

        with pytest.raises(Exception, match="immutable.*UPDATE prohibited"):
            with initialized_engine.begin() as conn:
                conn.execute(text("""
                    UPDATE replay_audit_records SET replay_status = 'failed'
                    WHERE replay_id = 'replay-upd'
                """))

    def test_delete_prohibited(self, initialized_engine):
        with initialized_engine.begin() as conn:
            self._insert_audit_record(conn, "replay-del")

        with pytest.raises(Exception, match="immutable.*DELETE prohibited"):
            with initialized_engine.begin() as conn:
                conn.execute(text("""
                    DELETE FROM replay_audit_records WHERE replay_id = 'replay-del'
                """))


class TestLineageMigrations:
    """Verify candidate_lineage_id columns are added to existing tables."""

    def test_blocked_trade_candidates_has_lineage_column(self, initialized_engine):
        inspector = inspect(initialized_engine)
        columns = {col["name"] for col in inspector.get_columns("blocked_trade_candidates")}
        assert "candidate_lineage_id" in columns

    def test_funnel_candidates_has_lineage_column(self, initialized_engine):
        inspector = inspect(initialized_engine)
        columns = {col["name"] for col in inspector.get_columns("funnel_candidates")}
        assert "candidate_lineage_id" in columns

    def test_trade_events_has_lineage_column(self, initialized_engine):
        inspector = inspect(initialized_engine)
        columns = {col["name"] for col in inspector.get_columns("trade_events")}
        assert "candidate_lineage_id" in columns

    def test_trades_has_lineage_column(self, initialized_engine):
        inspector = inspect(initialized_engine)
        columns = {col["name"] for col in inspector.get_columns("trades")}
        assert "candidate_lineage_id" in columns

    def test_pm_candidates_has_lineage_column(self, initialized_engine):
        inspector = inspect(initialized_engine)
        columns = {col["name"] for col in inspector.get_columns("pm_candidates")}
        assert "candidate_lineage_id" in columns


class TestPragmas:
    """Verify SQLite pragmas are applied correctly.

    Uses a fresh engine (no prior connections) so the pragma event listener
    fires on the first connection. Uses a file-based temp DB since in-memory
    SQLite doesn't support WAL mode.
    """

    @pytest.fixture
    def fresh_file_engine(self, tmp_path):
        """Engine backed by a temp file, no prior connections."""
        db_path = tmp_path / "test_pragmas.db"
        eng = create_engine(f"sqlite:///{db_path}")
        # Create minimal tables needed for lineage migrations
        with eng.begin() as conn:
            conn.execute(text("CREATE TABLE blocked_trade_candidates (id INTEGER PRIMARY KEY)"))
            conn.execute(text("CREATE TABLE funnel_candidates (id INTEGER PRIMARY KEY)"))
            conn.execute(text("CREATE TABLE trade_events (id INTEGER PRIMARY KEY)"))
            conn.execute(text("CREATE TABLE trades (id INTEGER PRIMARY KEY)"))
            conn.execute(text("CREATE TABLE pm_candidates (id INTEGER PRIMARY KEY)"))
        # Dispose to clear connection pool so pragmas fire on next connect
        eng.dispose()
        # Re-create engine to register pragmas fresh
        eng2 = create_engine(f"sqlite:///{db_path}")
        init_replay_db(eng2)
        return eng2

    def test_wal_mode_enabled(self, fresh_file_engine):
        with fresh_file_engine.connect() as conn:
            result = conn.execute(text("PRAGMA journal_mode")).fetchone()
            assert result[0] == "wal"

    def test_busy_timeout_set(self, fresh_file_engine):
        with fresh_file_engine.connect() as conn:
            result = conn.execute(text("PRAGMA busy_timeout")).fetchone()
            assert result[0] == 30000

    def test_foreign_keys_enabled(self, fresh_file_engine):
        with fresh_file_engine.connect() as conn:
            result = conn.execute(text("PRAGMA foreign_keys")).fetchone()
            assert result[0] == 1


class TestIdempotency:
    """Verify init_replay_db can be called multiple times safely."""

    def test_double_init_does_not_raise(self, engine):
        init_replay_db(engine)
        # Second call should not raise
        init_replay_db(engine)

        inspector = inspect(engine)
        assert inspector.has_table("decision_snapshots")
        assert inspector.has_table("replay_audit_records")

    def test_lineage_migration_idempotent(self, engine):
        init_replay_db(engine)
        init_replay_db(engine)

        inspector = inspect(engine)
        columns = {col["name"] for col in inspector.get_columns("trades")}
        assert "candidate_lineage_id" in columns
