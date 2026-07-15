"""
Tests for utils/lifecycle_checklist.py — LifecycleChecklist dataclass and
write_lifecycle_checklist() function.

Requirements: 7.1, 7.2, 7.3, 7.4
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text

from utils.lifecycle_checklist import LifecycleChecklist, write_lifecycle_checklist


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """Create an in-memory SQLite database with required tables."""
    eng = create_engine("sqlite:///:memory:")
    with eng.connect() as conn:
        # Create trades table
        conn.execute(text("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY,
                symbol VARCHAR(10) NOT NULL,
                direction VARCHAR(5) NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                stop_price REAL,
                target_price REAL,
                status VARCHAR(8) DEFAULT 'open',
                profile VARCHAR(16) DEFAULT 'moderate',
                invalidators TEXT,
                candidate_lineage_id VARCHAR(36)
            )
        """))
        # Create positions table
        conn.execute(text("""
            CREATE TABLE positions (
                id INTEGER PRIMARY KEY,
                profile VARCHAR(16) DEFAULT 'moderate',
                symbol VARCHAR(10) NOT NULL,
                side VARCHAR(5) DEFAULT 'long',
                quantity REAL NOT NULL,
                avg_cost REAL NOT NULL
            )
        """))
        # Create pm_candidates table
        conn.execute(text("""
            CREATE TABLE pm_candidates (
                id INTEGER PRIMARY KEY,
                candidate_id VARCHAR(36) NOT NULL UNIQUE,
                cycle_id VARCHAR(64) NOT NULL,
                profile_id VARCHAR(64) NOT NULL,
                symbol VARCHAR(10) NOT NULL,
                direction VARCHAR(10) NOT NULL,
                setup_type VARCHAR(64) NOT NULL,
                geometry_name VARCHAR(64) NOT NULL,
                entry_price REAL NOT NULL,
                stop_price REAL NOT NULL,
                target_price REAL NOT NULL,
                risk_reward REAL NOT NULL,
                trigger TEXT,
                invalidation_basis TEXT,
                target_basis TEXT,
                source_signal_id VARCHAR(64) NOT NULL,
                signal_snapshot_json TEXT NOT NULL,
                state VARCHAR(32) DEFAULT 'registered',
                integrity_hash VARCHAR(64) NOT NULL,
                execution_key VARCHAR(128),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                expires_at DATETIME NOT NULL
            )
        """))
        # Create response_lineage_links table
        conn.execute(text("""
            CREATE TABLE response_lineage_links (
                id INTEGER PRIMARY KEY,
                response_id VARCHAR(36) NOT NULL,
                lineage_id VARCHAR(36) NOT NULL,
                candidate_id VARCHAR(36),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))
        # Create candidate_lifecycle_checklists table
        conn.execute(text("""
            CREATE TABLE candidate_lifecycle_checklists (
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
        """))
        conn.commit()
    return eng


def _insert_candidate(engine, candidate_id: str, symbol: str = "AAPL", profile_id: str = "moderate"):
    """Helper to insert a candidate record."""
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO pm_candidates (
                candidate_id, cycle_id, profile_id, symbol, direction,
                setup_type, geometry_name, entry_price, stop_price,
                target_price, risk_reward, source_signal_id,
                signal_snapshot_json, integrity_hash, expires_at
            ) VALUES (
                :cid, 'cycle-1', :pid, :symbol, 'BUY',
                'breakout', 'analyst_geometry', 150.0, 145.0,
                160.0, 2.0, 'signal-1',
                '{}', 'hash123', '2099-01-01T00:00:00'
            )
        """), {"cid": candidate_id, "pid": profile_id, "symbol": symbol})
        conn.commit()


def _insert_trade(engine, candidate_id: str, symbol: str = "AAPL",
                  stop_price=145.0, target_price=160.0,
                  invalidators='[{"type":"price_level"}]',
                  status="open"):
    """Helper to insert a trade linked to a candidate."""
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO trades (
                symbol, direction, quantity, entry_price, stop_price,
                target_price, status, profile, invalidators, candidate_lineage_id
            ) VALUES (
                :symbol, 'LONG', 100, 150.0, :stop_price,
                :target_price, :status, 'moderate', :invalidators, :candidate_id
            )
        """), {
            "symbol": symbol,
            "stop_price": stop_price,
            "target_price": target_price,
            "invalidators": invalidators,
            "status": status,
            "candidate_id": candidate_id,
        })
        conn.commit()


def _insert_position(engine, symbol: str = "AAPL", profile: str = "moderate"):
    """Helper to insert a position."""
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO positions (profile, symbol, side, quantity, avg_cost)
            VALUES (:profile, :symbol, 'long', 100, 150.0)
        """), {"profile": profile, "symbol": symbol})
        conn.commit()


def _insert_lineage_link(engine, candidate_id: str):
    """Helper to insert a response lineage link."""
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO response_lineage_links (response_id, lineage_id, candidate_id)
            VALUES (:rid, :lid, :cid)
        """), {"rid": str(uuid.uuid4()), "lid": str(uuid.uuid4()), "cid": candidate_id})
        conn.commit()


# ---------------------------------------------------------------------------
# Dataclass Tests
# ---------------------------------------------------------------------------

class TestLifecycleChecklistDataclass:
    """Tests for the LifecycleChecklist frozen dataclass."""

    def test_complete_when_all_true(self):
        checklist = LifecycleChecklist(
            candidate_id="c-1",
            trade_id="t-1",
            trade_row_created=True,
            position_row_created_or_updated=True,
            stop_registered=True,
            target_registered=True,
            thesis_invalidation_recorded=True,
            position_monitor_armed=True,
            review_lineage_linked=True,
        )
        assert checklist.complete is True
        assert checklist.missing_components == []

    def test_not_complete_when_some_false(self):
        checklist = LifecycleChecklist(
            candidate_id="c-1",
            trade_id="t-1",
            trade_row_created=True,
            position_row_created_or_updated=True,
            stop_registered=True,
            target_registered=False,
            thesis_invalidation_recorded=True,
            position_monitor_armed=False,
            review_lineage_linked=True,
        )
        assert checklist.complete is False
        assert checklist.missing_components == [
            "target_registered",
            "position_monitor_armed",
        ]

    def test_all_missing_when_all_false(self):
        checklist = LifecycleChecklist(
            candidate_id="c-1",
            trade_id="t-1",
            trade_row_created=False,
            position_row_created_or_updated=False,
            stop_registered=False,
            target_registered=False,
            thesis_invalidation_recorded=False,
            position_monitor_armed=False,
            review_lineage_linked=False,
        )
        assert checklist.complete is False
        assert len(checklist.missing_components) == 7

    def test_frozen_dataclass_immutable(self):
        checklist = LifecycleChecklist(
            candidate_id="c-1",
            trade_id="t-1",
            trade_row_created=True,
            position_row_created_or_updated=True,
            stop_registered=True,
            target_registered=True,
            thesis_invalidation_recorded=True,
            position_monitor_armed=True,
            review_lineage_linked=True,
        )
        with pytest.raises(AttributeError):
            checklist.trade_row_created = False  # type: ignore


# ---------------------------------------------------------------------------
# write_lifecycle_checklist Tests
# ---------------------------------------------------------------------------

class TestWriteLifecycleChecklist:
    """Tests for the write_lifecycle_checklist() function."""

    def test_all_components_present(self, engine):
        """All components exist → complete checklist."""
        cid = str(uuid.uuid4())
        _insert_candidate(engine, cid)
        _insert_trade(engine, cid)
        _insert_position(engine)
        _insert_lineage_link(engine, cid)

        checklist = write_lifecycle_checklist(
            engine, cid, "trade-1", "cycle-1", "moderate"
        )

        assert checklist is not None
        assert checklist.complete is True
        assert checklist.trade_row_created is True
        assert checklist.position_row_created_or_updated is True
        assert checklist.stop_registered is True
        assert checklist.target_registered is True
        assert checklist.thesis_invalidation_recorded is True
        assert checklist.position_monitor_armed is True
        assert checklist.review_lineage_linked is True

    def test_no_trade_row(self, engine):
        """No trade exists → trade_row_created is False."""
        cid = str(uuid.uuid4())
        _insert_candidate(engine, cid)

        checklist = write_lifecycle_checklist(
            engine, cid, "trade-1", "cycle-1", "moderate"
        )

        assert checklist is not None
        assert checklist.trade_row_created is False

    def test_no_position(self, engine):
        """No position for symbol → position_row_created_or_updated is False."""
        cid = str(uuid.uuid4())
        _insert_candidate(engine, cid)
        _insert_trade(engine, cid)

        checklist = write_lifecycle_checklist(
            engine, cid, "trade-1", "cycle-1", "moderate"
        )

        assert checklist is not None
        assert checklist.position_row_created_or_updated is False

    def test_no_stop(self, engine):
        """Trade has null stop_price → stop_registered is False."""
        cid = str(uuid.uuid4())
        _insert_candidate(engine, cid)
        _insert_trade(engine, cid, stop_price=None)

        checklist = write_lifecycle_checklist(
            engine, cid, "trade-1", "cycle-1", "moderate"
        )

        assert checklist is not None
        assert checklist.stop_registered is False

    def test_zero_stop(self, engine):
        """Trade has stop_price=0 → stop_registered is False."""
        cid = str(uuid.uuid4())
        _insert_candidate(engine, cid)
        _insert_trade(engine, cid, stop_price=0)

        checklist = write_lifecycle_checklist(
            engine, cid, "trade-1", "cycle-1", "moderate"
        )

        assert checklist is not None
        assert checklist.stop_registered is False

    def test_no_target(self, engine):
        """Trade has null target_price → target_registered is False."""
        cid = str(uuid.uuid4())
        _insert_candidate(engine, cid)
        _insert_trade(engine, cid, target_price=None)

        checklist = write_lifecycle_checklist(
            engine, cid, "trade-1", "cycle-1", "moderate"
        )

        assert checklist is not None
        assert checklist.target_registered is False

    def test_no_invalidators(self, engine):
        """Trade has empty invalidators → thesis_invalidation_recorded is False."""
        cid = str(uuid.uuid4())
        _insert_candidate(engine, cid)
        _insert_trade(engine, cid, invalidators=None)

        checklist = write_lifecycle_checklist(
            engine, cid, "trade-1", "cycle-1", "moderate"
        )

        assert checklist is not None
        assert checklist.thesis_invalidation_recorded is False

    def test_empty_list_invalidators(self, engine):
        """Trade has invalidators='[]' → thesis_invalidation_recorded is False."""
        cid = str(uuid.uuid4())
        _insert_candidate(engine, cid)
        _insert_trade(engine, cid, invalidators="[]")

        checklist = write_lifecycle_checklist(
            engine, cid, "trade-1", "cycle-1", "moderate"
        )

        assert checklist is not None
        assert checklist.thesis_invalidation_recorded is False

    def test_closed_trade_monitor_not_armed(self, engine):
        """Trade with status='closed' → position_monitor_armed is False."""
        cid = str(uuid.uuid4())
        _insert_candidate(engine, cid)
        _insert_trade(engine, cid, status="closed")

        checklist = write_lifecycle_checklist(
            engine, cid, "trade-1", "cycle-1", "moderate"
        )

        assert checklist is not None
        assert checklist.position_monitor_armed is False

    def test_no_lineage_link(self, engine):
        """No response_lineage_links row → review_lineage_linked is False."""
        cid = str(uuid.uuid4())
        _insert_candidate(engine, cid)
        _insert_trade(engine, cid)
        _insert_position(engine)

        checklist = write_lifecycle_checklist(
            engine, cid, "trade-1", "cycle-1", "moderate"
        )

        assert checklist is not None
        assert checklist.review_lineage_linked is False

    def test_persists_to_database(self, engine):
        """Checklist is persisted to candidate_lifecycle_checklists table."""
        cid = str(uuid.uuid4())
        _insert_candidate(engine, cid)
        _insert_trade(engine, cid)
        _insert_position(engine)

        write_lifecycle_checklist(engine, cid, "trade-1", "cycle-1", "moderate")

        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT * FROM candidate_lifecycle_checklists WHERE candidate_id = :cid"
            ), {"cid": cid}).fetchone()

        assert row is not None

    def test_fail_open_returns_none_on_error(self, engine):
        """On database error, returns None without raising."""
        # Use an engine that will fail (bad table name scenario)
        # We'll mock the engine.connect to raise
        with patch.object(engine, "connect", side_effect=RuntimeError("DB down")):
            result = write_lifecycle_checklist(
                engine, "bad-id", "trade-1", "cycle-1", "moderate"
            )
        assert result is None

    def test_missing_candidate_symbol_graceful(self, engine):
        """When candidate doesn't exist in pm_candidates, position check returns False."""
        cid = str(uuid.uuid4())
        # Don't insert candidate — symbol lookup will return None

        checklist = write_lifecycle_checklist(
            engine, cid, "trade-1", "cycle-1", "moderate"
        )

        assert checklist is not None
        assert checklist.position_row_created_or_updated is False
