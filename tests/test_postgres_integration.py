"""Postgres integration tests — skipped unless DATABASE_URL is set.

These tests verify the paper-trader runtime works correctly against a real
Postgres database. They are part of the validation subset described in
Requirement 5.7.

Run with:
    DATABASE_URL=postgresql+psycopg://... pytest tests/test_postgres_integration.py -x -q
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text, inspect as sa_inspect

# Skip entire module if DATABASE_URL is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL", "").strip(),
    reason="DATABASE_URL not set — Postgres integration tests skipped",
)


@pytest.fixture(scope="module")
def pg_engine():
    """Create a Postgres engine from DATABASE_URL and initialize full schema."""
    url = os.environ["DATABASE_URL"]
    engine = create_engine(url, pool_pre_ping=True)

    # Verify connectivity
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))

    # Initialize all schemas (same as production startup)
    from db.schema import Base
    from models.case import Case  # noqa: F401

    Base.metadata.create_all(engine)

    from orchestrator import check_schema
    check_schema(engine)

    from utils.alert_dispatch_schema import init_alert_dispatch_schema
    init_alert_dispatch_schema(engine)

    from utils.shadow_ledger import ensure_shadow_ledger_schema
    ensure_shadow_ledger_schema(engine)

    yield engine
    engine.dispose()


@pytest.fixture()
def pg_conn(pg_engine):
    """Provide a connection wrapped in a SAVEPOINT that rolls back after each test.

    This isolates tests so they do not leave residual data in the database.
    """
    conn = pg_engine.connect()
    trans = conn.begin()
    # Use a nested transaction (SAVEPOINT) so test writes are visible within
    # the test but rolled back at the end.
    nested = conn.begin_nested()
    yield conn
    nested.rollback()
    trans.rollback()
    conn.close()


# ---------------------------------------------------------------------------
# Candidate Lifecycle CAS Operations
# ---------------------------------------------------------------------------


class TestCandidateLifecycle:
    """Test CAS state transitions against Postgres."""

    def _create_test_candidate(self, conn, candidate_id: str, cycle_id: str, profile_id: str):
        """Insert a test candidate in REGISTERED state."""
        now = datetime.now(timezone.utc).isoformat()
        expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        integrity_hash = "a" * 64
        conn.execute(
            text("""
                INSERT INTO pm_candidates (
                    candidate_id, cycle_id, profile_id, symbol, direction,
                    setup_type, geometry_name, entry_price, stop_price,
                    target_price, risk_reward, trigger, invalidation_basis,
                    target_basis, source_signal_id, signal_snapshot_json,
                    state, integrity_hash, created_at, expires_at
                ) VALUES (
                    :candidate_id, :cycle_id, :profile_id, :symbol, :direction,
                    :setup_type, :geometry_name, :entry_price, :stop_price,
                    :target_price, :risk_reward, :trigger, :invalidation_basis,
                    :target_basis, :source_signal_id, :signal_snapshot_json,
                    :state, :integrity_hash, :created_at, :expires_at
                )
            """),
            {
                "candidate_id": candidate_id,
                "cycle_id": cycle_id,
                "profile_id": profile_id,
                "symbol": "TEST",
                "direction": "BUY",
                "setup_type": "breakout",
                "geometry_name": "standard",
                "entry_price": 100.0,
                "stop_price": 95.0,
                "target_price": 110.0,
                "risk_reward": 2.0,
                "trigger": "price above resistance",
                "invalidation_basis": "break below support",
                "target_basis": "measured move",
                "source_signal_id": str(uuid.uuid4()),
                "signal_snapshot_json": json.dumps({"signal": "test"}),
                "state": "registered",
                "integrity_hash": integrity_hash,
                "created_at": now,
                "expires_at": expires,
            },
        )

    def test_register_and_reserve_cas(self, pg_conn):
        """CAS reserve: REGISTERED → RESERVED succeeds with rowcount=1."""
        cid = str(uuid.uuid4())
        cycle_id = f"test-cycle-{uuid.uuid4().hex[:8]}"
        profile_id = "moderate"

        self._create_test_candidate(pg_conn, cid, cycle_id, profile_id)

        # CAS: reserve
        now = datetime.now(timezone.utc).isoformat()
        exec_key = f"exec-{uuid.uuid4().hex[:8]}"
        result = pg_conn.execute(
            text("""
                UPDATE pm_candidates
                SET state = :new_state,
                    reserved_at = :reserved_at,
                    execution_key = :execution_key
                WHERE candidate_id = :candidate_id
                  AND state = :expected_state
            """),
            {
                "new_state": "reserved",
                "reserved_at": now,
                "execution_key": exec_key,
                "candidate_id": cid,
                "expected_state": "registered",
            },
        )
        assert result.rowcount == 1

        # Verify state
        row = pg_conn.execute(
            text("SELECT state, execution_key FROM pm_candidates WHERE candidate_id = :cid"),
            {"cid": cid},
        ).mappings().one()
        assert row["state"] == "reserved"
        assert row["execution_key"] == exec_key

    def test_double_reserve_cas_fails(self, pg_conn):
        """CAS reserve on already-reserved candidate returns rowcount=0."""
        cid = str(uuid.uuid4())
        cycle_id = f"test-cycle-{uuid.uuid4().hex[:8]}"
        profile_id = "moderate"

        self._create_test_candidate(pg_conn, cid, cycle_id, profile_id)

        # First reserve succeeds
        now = datetime.now(timezone.utc).isoformat()
        pg_conn.execute(
            text("""
                UPDATE pm_candidates
                SET state = 'reserved', reserved_at = :now, execution_key = :key
                WHERE candidate_id = :cid AND state = 'registered'
            """),
            {"now": now, "key": "key-1", "cid": cid},
        )

        # Second reserve fails (CAS — state is no longer 'registered')
        result = pg_conn.execute(
            text("""
                UPDATE pm_candidates
                SET state = 'reserved', reserved_at = :now, execution_key = :key
                WHERE candidate_id = :cid AND state = 'registered'
            """),
            {"now": now, "key": "key-2", "cid": cid},
        )
        assert result.rowcount == 0

    def test_reserve_to_executed_transition(self, pg_conn):
        """CAS: RESERVED → EXECUTED succeeds."""
        cid = str(uuid.uuid4())
        cycle_id = f"test-cycle-{uuid.uuid4().hex[:8]}"

        self._create_test_candidate(pg_conn, cid, cycle_id, "moderate")

        # Reserve first
        now = datetime.now(timezone.utc).isoformat()
        pg_conn.execute(
            text("""
                UPDATE pm_candidates
                SET state = 'reserved', reserved_at = :now, execution_key = :key
                WHERE candidate_id = :cid AND state = 'registered'
            """),
            {"now": now, "key": "exec-key", "cid": cid},
        )

        # Mark executed
        result = pg_conn.execute(
            text("""
                UPDATE pm_candidates
                SET state = 'executed'
                WHERE candidate_id = :cid AND state = 'reserved'
            """),
            {"cid": cid},
        )
        assert result.rowcount == 1

        row = pg_conn.execute(
            text("SELECT state FROM pm_candidates WHERE candidate_id = :cid"),
            {"cid": cid},
        ).mappings().one()
        assert row["state"] == "executed"


# ---------------------------------------------------------------------------
# Alert Dispatch Intent Creation and Claim Flow
# ---------------------------------------------------------------------------


class TestAlertDispatch:
    """Test alert intent creation and claim flow against Postgres."""

    def test_create_alert_intent(self, pg_conn):
        """INSERT into alert_intents succeeds and is queryable."""
        intent_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        expiration = (datetime.now(timezone.utc) + timedelta(hours=4)).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        dedupe_key = f"test-dedupe-{uuid.uuid4().hex[:8]}"

        pg_conn.execute(
            text("""
                INSERT INTO alert_intents (
                    alert_intent_id, symbol, alert_type, direction,
                    trigger_price, urgency, reason, dedupe_key,
                    first_seen_at, last_seen_at, expiration_at
                ) VALUES (
                    :alert_intent_id, :symbol, :alert_type, :direction,
                    :trigger_price, :urgency, :reason, :dedupe_key,
                    :first_seen_at, :last_seen_at, :expiration_at
                )
            """),
            {
                "alert_intent_id": intent_id,
                "symbol": "NVDA",
                "alert_type": "entry_alert",
                "direction": "long",
                "trigger_price": "145.50",
                "urgency": "medium",
                "reason": "Price crossed resistance",
                "dedupe_key": dedupe_key,
                "first_seen_at": now,
                "last_seen_at": now,
                "expiration_at": expiration,
            },
        )

        row = pg_conn.execute(
            text("SELECT symbol, dispatch_status FROM alert_intents WHERE alert_intent_id = :id"),
            {"id": intent_id},
        ).mappings().one()
        assert row["symbol"] == "NVDA"
        assert row["dispatch_status"] == "pending"

    def test_claim_alert_intent(self, pg_conn):
        """Claim flow: insert intent, create claim, verify UNIQUE constraint."""
        intent_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        expiration = (datetime.now(timezone.utc) + timedelta(hours=4)).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        dedupe_key = f"claim-test-{uuid.uuid4().hex[:8]}"

        # Create intent
        pg_conn.execute(
            text("""
                INSERT INTO alert_intents (
                    alert_intent_id, symbol, alert_type, direction,
                    trigger_price, urgency, dedupe_key,
                    first_seen_at, last_seen_at, expiration_at
                ) VALUES (
                    :id, :symbol, :alert_type, :direction,
                    :trigger_price, :urgency, :dedupe_key,
                    :first_seen_at, :last_seen_at, :expiration_at
                )
            """),
            {
                "id": intent_id,
                "symbol": "AMD",
                "alert_type": "entry_alert",
                "direction": "long",
                "trigger_price": "160.00",
                "urgency": "high",
                "dedupe_key": dedupe_key,
                "first_seen_at": now,
                "last_seen_at": now,
                "expiration_at": expiration,
            },
        )

        # Create claim
        profile_id = "moderate"
        pg_conn.execute(
            text("""
                INSERT INTO pm_alert_claims (
                    alert_intent_id, symbol, alert_type, profile_id, claimed_at
                ) VALUES (:intent_id, :symbol, :alert_type, :profile_id, :claimed_at)
            """),
            {
                "intent_id": intent_id,
                "symbol": "AMD",
                "alert_type": "entry_alert",
                "profile_id": profile_id,
                "claimed_at": now,
            },
        )

        # Verify claim exists
        row = pg_conn.execute(
            text("""
                SELECT status FROM pm_alert_claims
                WHERE alert_intent_id = :id AND profile_id = :pid
            """),
            {"id": intent_id, "pid": profile_id},
        ).mappings().one()
        assert row["status"] == "claimed"

        # Duplicate claim should fail (UNIQUE constraint on alert_intent_id + profile_id)
        with pytest.raises(Exception):
            pg_conn.execute(
                text("""
                    INSERT INTO pm_alert_claims (
                        alert_intent_id, symbol, alert_type, profile_id, claimed_at
                    ) VALUES (:intent_id, :symbol, :alert_type, :profile_id, :claimed_at)
                """),
                {
                    "intent_id": intent_id,
                    "symbol": "AMD",
                    "alert_type": "entry_alert",
                    "profile_id": profile_id,
                    "claimed_at": now,
                },
            )


# ---------------------------------------------------------------------------
# Checkpoint Logging Writes
# ---------------------------------------------------------------------------


class TestCheckpointLogging:
    """Test checkpoint_events INSERT against Postgres."""

    def test_insert_checkpoint_event(self, pg_conn):
        """INSERT into checkpoint_events succeeds and returns correct data."""
        cycle_id = str(uuid.uuid4())
        candidate_id = str(uuid.uuid4())

        pg_conn.execute(
            text("""
                INSERT INTO checkpoint_events (
                    stage, outcome_category, cycle_id, candidate_id,
                    profile, symbol, setup_type, decision, reason_code,
                    metadata_json
                ) VALUES (
                    :stage, :outcome_category, :cycle_id, :candidate_id,
                    :profile, :symbol, :setup_type, :decision, :reason_code,
                    :metadata_json
                )
            """),
            {
                "stage": "gate_pipeline",
                "outcome_category": "blocked",
                "cycle_id": cycle_id,
                "candidate_id": candidate_id,
                "profile": "moderate",
                "symbol": "TSLA",
                "setup_type": "breakout",
                "decision": "reject",
                "reason_code": "concentration_limit",
                "metadata_json": json.dumps({"gate": "concentration", "pct": 0.12}),
            },
        )

        row = pg_conn.execute(
            text("""
                SELECT stage, symbol, decision, metadata_json
                FROM checkpoint_events
                WHERE cycle_id = :cycle_id AND candidate_id = :cid
            """),
            {"cycle_id": cycle_id, "cid": candidate_id},
        ).mappings().one()
        assert row["stage"] == "gate_pipeline"
        assert row["symbol"] == "TSLA"
        assert row["decision"] == "reject"
        meta = json.loads(row["metadata_json"])
        assert meta["gate"] == "concentration"

    def test_checkpoint_created_at_default(self, pg_conn):
        """Checkpoint created_at defaults to CURRENT_TIMESTAMP on Postgres."""
        cycle_id = str(uuid.uuid4())

        pg_conn.execute(
            text("""
                INSERT INTO checkpoint_events (
                    stage, cycle_id, profile, decision
                ) VALUES (:stage, :cycle_id, :profile, :decision)
            """),
            {
                "stage": "sizing",
                "cycle_id": cycle_id,
                "profile": "aggressive",
                "decision": "pass",
            },
        )

        row = pg_conn.execute(
            text("SELECT created_at FROM checkpoint_events WHERE cycle_id = :cid"),
            {"cid": cycle_id},
        ).mappings().one()
        # created_at should be auto-populated (not NULL)
        assert row["created_at"] is not None


# ---------------------------------------------------------------------------
# Web API /api/data Health Check
# ---------------------------------------------------------------------------


class TestWebApiHealth:
    """Test /api/data returns 200 against Postgres.

    This test uses the Flask test client with the app configured against
    the real Postgres engine. It verifies the endpoint responds without
    crashing when queries run against Postgres.
    """

    def test_api_data_returns_200(self, pg_engine):
        """The /api/data endpoint returns HTTP 200 when backed by Postgres."""
        import sys
        import importlib

        # Import web app module — it initializes its own engine from DATABASE_URL
        # Since DATABASE_URL is set (we wouldn't be here otherwise), it will
        # connect to Postgres.
        from web.app import app

        with app.test_client() as client:
            resp = client.get("/api/data")
            # Accept 200 (success) — the endpoint may return partial data
            # if market data APIs are unavailable, but the DB queries should work.
            assert resp.status_code == 200
            data = resp.get_json()
            assert "timestamp" in data
            assert "watchlist" in data


# ---------------------------------------------------------------------------
# Dialect-Conditional Queries (Shadow Outcomes, Gate Effectiveness, Dashboard)
# ---------------------------------------------------------------------------


class TestDialectQueries:
    """Test shadow outcomes and gate effectiveness queries against Postgres.

    These validate that the dialect-conditional SQL helpers generate valid
    Postgres SQL that executes without syntax errors.
    """

    def test_shadow_outcomes_insert_on_conflict(self, pg_conn, pg_engine):
        """ON CONFLICT DO NOTHING syntax works for blocked_trade_candidate_outcomes."""
        # First, insert a parent blocked_trade_candidates row
        pg_conn.execute(
            text("""
                INSERT INTO blocked_trade_candidates (
                    symbol, action, direction, profile, setup_type,
                    entry_price, stop_price, target_price,
                    blocked_by, block_reason, created_at
                ) VALUES (
                    :symbol, :action, :direction, :profile, :setup_type,
                    :entry_price, :stop_price, :target_price,
                    :blocked_by, :block_reason, :created_at
                )
            """),
            {
                "symbol": "AAPL",
                "action": "BUY",
                "direction": "long",
                "profile": "moderate",
                "setup_type": "breakout",
                "entry_price": 180.0,
                "stop_price": 175.0,
                "target_price": 195.0,
                "blocked_by": "concentration_gate",
                "block_reason": "Exceeds sector limit",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        # Get the inserted ID
        row = pg_conn.execute(
            text("SELECT id FROM blocked_trade_candidates WHERE symbol = 'AAPL' ORDER BY id DESC LIMIT 1")
        ).mappings().one()
        parent_id = row["id"]

        # Insert outcome using ON CONFLICT DO NOTHING (the dialect-neutral pattern)
        from utils.dialect_sql import _upsert_outcome_sql

        outcome_params = {
            "blocked_candidate_id": parent_id,
            "eval_window": "60m",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "eval_price": 182.0,
            "pnl_pct": 1.1,
            "mfe_pct": 2.5,
            "mae_pct": -0.8,
            "stop_hit": False,
            "target_hit": False,
            "first_hit": None,
            "first_hit_at": None,
            "outcome_label": "neutral",
            "gate_verdict": "saved_us",
            "notes_json": json.dumps({"candidate_source_classification": "valid"}),
        }
        pg_conn.execute(text(_upsert_outcome_sql()), outcome_params)

        # Verify insert
        count = pg_conn.execute(
            text("""
                SELECT COUNT(*) FROM blocked_trade_candidate_outcomes
                WHERE blocked_candidate_id = :pid AND eval_window = '60m'
            """),
            {"pid": parent_id},
        ).scalar()
        assert count == 1

        # Second insert with same (blocked_candidate_id, eval_window) should be ignored
        pg_conn.execute(text(_upsert_outcome_sql()), outcome_params)
        count = pg_conn.execute(
            text("""
                SELECT COUNT(*) FROM blocked_trade_candidate_outcomes
                WHERE blocked_candidate_id = :pid AND eval_window = '60m'
            """),
            {"pid": parent_id},
        ).scalar()
        assert count == 1  # Still 1 — ON CONFLICT DO NOTHING

    def test_date_cutoff_filter_postgres(self, pg_engine):
        """_date_cutoff_filter produces valid Postgres SQL."""
        from utils.dialect_sql import _date_cutoff_filter

        fragment = _date_cutoff_filter(pg_engine, "b.created_at")
        # Postgres branch should use NOW() + interval
        assert "NOW()" in fragment
        assert "interval" in fragment

        # Verify it executes without syntax error in a real query
        with pg_engine.connect() as conn:
            # This won't find rows but should not raise a SQL syntax error
            conn.execute(
                text(f"""
                    SELECT COUNT(*) FROM blocked_trade_candidates b
                    WHERE {fragment}
                """),
                {"cutoff": "-7 days"},
            )

    def test_json_field_postgres(self, pg_engine):
        """_json_field produces valid Postgres SQL for JSONB extraction."""
        from utils.dialect_sql import _json_field

        fragment = _json_field(pg_engine, "o.notes_json", "candidate_source_classification")
        # Postgres branch uses ::jsonb->>
        assert "jsonb" in fragment
        assert "candidate_source_classification" in fragment

    def test_gate_effectiveness_query_executes(self, pg_engine):
        """get_gate_effectiveness_summary runs without SQL errors on Postgres."""
        from utils.shadow_outcomes import get_gate_effectiveness_summary

        # Should execute cleanly even with no data
        result = get_gate_effectiveness_summary(pg_engine, lookback_days=7)
        assert result["gate_name"] == "all"
        assert result["blocked_winners"] == 0
        assert result["saved_us"] == 0
        assert result["period_days"] == 7

    def test_shadow_outcomes_dashboard_query(self, pg_engine):
        """Dashboard shadow outcomes query executes without SQL errors on Postgres."""
        from utils.dialect_sql import _date_cutoff_filter

        # Simulate the dashboard query pattern from web/app.py
        cutoff_filter = _date_cutoff_filter(pg_engine, "b.created_at")
        query = f"""
            SELECT b.symbol, b.blocked_by, b.block_reason, b.created_at
            FROM blocked_trade_candidates b
            WHERE {cutoff_filter}
            ORDER BY b.created_at DESC
            LIMIT 20
        """
        with pg_engine.connect() as conn:
            rows = conn.execute(text(query), {"cutoff": "-7 days"}).mappings().all()
            # No rows expected in a clean DB, but query should not fail
            assert isinstance(rows, list)
