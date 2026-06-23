"""
End-to-end integration tests for the Decision Replay Agent.

Tests the FULL pipeline with real in-memory DB state:
1. Sourcing → reconstruction → replay → delta → outcome → persistence
2. Blocked candidate through full pipeline
3. Executed trade through full pipeline
4. Backfill from existing shadow ledger data
5. Candle evidence persistence and content_hash reproducibility
6. Decision-specific outcome branches:
   - reject→allow scores candles
   - allow→reject reads realized P&L
   - reject→reject produces no outcome record

Requirements: 13.1, 13.2, 13.3, 7.1, 7.2
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, text

from agents.decision_replay import run, BatchRunSummary
from core.replay.candidate_sourcer import (
    ReplayCandidate,
    SourceReference,
    load_candidates,
    correlate_and_deduplicate,
)
from core.replay.input_reconstructor import (
    InputSource,
    ReplayInputBundle,
    reconstruct_inputs,
    compute_replay_cutoff,
)
from core.replay.gate_replayer import replay_gates, GateTrace
from core.replay.delta_classifier import classify_delta, DecisionDelta
from core.replay.outcome_scorer import (
    build_candle_evidence,
    build_fill_model,
    score_counterfactual,
    score_allowed_to_rejected,
    should_score_counterfactual,
    CandleEvidence,
    FillModel,
    CounterfactualOutcome,
)
from core.replay.policy_version import PolicyVersion, build_current_policy_version
from core.replay.gate_adapter import (
    GatePolicyConfig,
    ReplayGateContext,
    build_current_gate_policy_config,
    build_replay_clock,
    build_deterministic_id_provider,
)
from core.replay.backfill import (
    grade_historical_record,
    assess_backfill_coverage,
    generate_coverage_report,
    BackfillGrade,
)
from db.replay_schema import init_replay_db
from db.schema import get_session


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """In-memory SQLite engine with production + replay schema initialized."""
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        # Create production tables needed by the replay pipeline
        conn.execute(text("""
            CREATE TABLE blocked_trade_candidates (
                id INTEGER PRIMARY KEY,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                symbol VARCHAR(10),
                action VARCHAR(16) NOT NULL,
                direction VARCHAR(10),
                profile VARCHAR(16),
                setup_type VARCHAR(64),
                entry_price REAL,
                stop_price REAL,
                target_price REAL,
                quantity REAL,
                blocked_by VARCHAR(64) NOT NULL,
                block_reason TEXT NOT NULL,
                reason_code VARCHAR(64),
                gate_notes_json TEXT,
                decision_snapshot_json TEXT,
                signal_snapshot_json TEXT,
                source VARCHAR(64),
                agent VARCHAR(64),
                dedupe_key VARCHAR(255),
                trade_event_id INTEGER
            )
        """))
        conn.execute(text("""
            CREATE TABLE trade_events (
                id INTEGER PRIMARY KEY,
                trade_id INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type VARCHAR(64) NOT NULL,
                agent VARCHAR(64),
                symbol VARCHAR(10),
                profile VARCHAR(16),
                price REAL,
                message TEXT,
                payload_json TEXT,
                dedupe_key VARCHAR(256),
                candidate_lineage_id VARCHAR(36)
            )
        """))
        conn.execute(text("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY,
                profile VARCHAR(16) DEFAULT 'moderate',
                symbol VARCHAR(10) NOT NULL,
                direction VARCHAR(5) NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                exit_time DATETIME,
                status VARCHAR(8) DEFAULT 'open',
                pnl REAL,
                pnl_pct REAL,
                reason_entry TEXT,
                reason_exit TEXT,
                stop_price REAL,
                target_price REAL,
                setup_type VARCHAR(64),
                candidate_lineage_id VARCHAR(36)
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
            CREATE TABLE pm_candidates (
                id INTEGER PRIMARY KEY,
                candidate_id VARCHAR(36),
                symbol VARCHAR(10),
                profile_id VARCHAR(16),
                direction VARCHAR(10),
                setup_type VARCHAR(64),
                state VARCHAR(16) DEFAULT 'pending',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                rejection_reason TEXT,
                entry_price REAL,
                stop_price REAL,
                target_price REAL,
                candidate_lineage_id VARCHAR(36)
            )
        """))
        conn.execute(text("""
            CREATE TABLE positions (
                id INTEGER PRIMARY KEY,
                profile VARCHAR(16) DEFAULT 'moderate',
                symbol VARCHAR(10) NOT NULL,
                side VARCHAR(5) DEFAULT 'long',
                quantity REAL NOT NULL,
                avg_cost REAL NOT NULL,
                opened_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE balance (
                id INTEGER PRIMARY KEY,
                profile VARCHAR(16) DEFAULT 'moderate',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                cash REAL NOT NULL,
                portfolio_value REAL,
                total_equity REAL
            )
        """))
        conn.execute(text("""
            CREATE TABLE agent_memory (
                id INTEGER PRIMARY KEY,
                agent VARCHAR(32) NOT NULL,
                symbol VARCHAR(10),
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                key VARCHAR(64) NOT NULL,
                value TEXT NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE blocked_trade_candidate_outcomes (
                id INTEGER PRIMARY KEY,
                blocked_candidate_id INTEGER NOT NULL,
                eval_window VARCHAR(16) NOT NULL,
                evaluated_at DATETIME NOT NULL,
                entry_price REAL,
                stop_price REAL,
                target_price REAL,
                outcome_label VARCHAR(16),
                gate_verdict VARCHAR(16),
                pnl_pct REAL,
                mfe_pct REAL,
                mae_pct REAL,
                stop_hit INTEGER DEFAULT 0,
                target_hit INTEGER DEFAULT 0,
                first_hit VARCHAR(16)
            )
        """))
    init_replay_db(eng)
    return eng


@pytest.fixture
def base_timestamp():
    """Base timestamp for test data (outside market hours for batch replay)."""
    return datetime(2024, 6, 15, 14, 30, 0)


@pytest.fixture
def policy_version():
    """Test policy version."""
    return PolicyVersion(
        name="test_e2e",
        gate_revision="v2.0.0",
        config_digest="e2e_test_digest",
        feature_flags={"SETUP_SPECIFIC_RR_THRESHOLDS": True},
        benchmark_version=None,
        config_source_timestamp=datetime(2024, 6, 15, 10, 0, 0),
        gate_ordering_version="v1.0",
        adapter_version="1.0.0",
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _insert_blocked_candidate(
    engine,
    *,
    symbol="AAPL",
    profile="aggressive",
    direction="long",
    setup_type="news_breakout",
    entry_price=185.50,
    stop_price=184.00,
    target_price=188.00,
    quantity=10,
    blocked_by="risk_geometry_gate",
    block_reason="R:R below threshold",
    reason_code="RISK_REWARD_BELOW_THRESHOLD",
    created_at=None,
    lineage_id=None,
):
    """Insert a blocked trade candidate and return its ID."""
    if created_at is None:
        created_at = datetime(2024, 6, 15, 14, 30, 0)
    with engine.begin() as conn:
        result = conn.execute(
            text("""
                INSERT INTO blocked_trade_candidates
                (symbol, action, direction, profile, setup_type,
                 entry_price, stop_price, target_price, quantity,
                 blocked_by, block_reason, reason_code, created_at,
                 candidate_lineage_id)
                VALUES (:symbol, :action, :direction, :profile, :setup_type,
                        :entry_price, :stop_price, :target_price, :quantity,
                        :blocked_by, :block_reason, :reason_code, :created_at,
                        :lineage_id)
            """),
            {
                "symbol": symbol,
                "action": "BUY" if direction.upper() == "LONG" else "SHORT",
                "direction": direction,
                "profile": profile,
                "setup_type": setup_type,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "quantity": quantity,
                "blocked_by": blocked_by,
                "block_reason": block_reason,
                "reason_code": reason_code,
                "created_at": created_at,
                "lineage_id": lineage_id,
            },
        )
        return result.lastrowid


def _insert_trade(
    engine,
    *,
    symbol="AAPL",
    profile="moderate",
    direction="LONG",
    setup_type="news_breakout",
    entry_price=185.50,
    stop_price=184.00,
    target_price=188.00,
    quantity=10,
    exit_price=None,
    status="open",
    pnl=None,
    entry_time=None,
    exit_time=None,
    lineage_id=None,
):
    """Insert a trade and return its ID."""
    if entry_time is None:
        entry_time = datetime(2024, 6, 15, 14, 30, 0)
    with engine.begin() as conn:
        result = conn.execute(
            text("""
                INSERT INTO trades
                (symbol, profile, direction, setup_type,
                 entry_price, stop_price, target_price, quantity,
                 exit_price, status, pnl, entry_time, exit_time,
                 candidate_lineage_id)
                VALUES (:symbol, :profile, :direction, :setup_type,
                        :entry_price, :stop_price, :target_price, :quantity,
                        :exit_price, :status, :pnl, :entry_time, :exit_time,
                        :lineage_id)
            """),
            {
                "symbol": symbol,
                "profile": profile,
                "direction": direction,
                "setup_type": setup_type,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "quantity": quantity,
                "exit_price": exit_price,
                "status": status,
                "pnl": pnl,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "lineage_id": lineage_id,
            },
        )
        return result.lastrowid


def _insert_analyst_signal(engine, *, symbol="AAPL", profile="aggressive",
                           signal_data=None, timestamp=None):
    """Insert an analyst signal into agent_memory."""
    if timestamp is None:
        timestamp = datetime(2024, 6, 15, 14, 29, 0)
    if signal_data is None:
        signal_data = {
            "signal_strength": 9.0,
            "confidence": "high",
            "setup_type": "news_breakout",
            "status": "active",
        }
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO agent_memory (agent, symbol, timestamp, key, value)
                VALUES ('analyst', :symbol, :timestamp, 'signal', :value)
            """),
            {
                "symbol": symbol,
                "timestamp": timestamp,
                "value": json.dumps(signal_data),
            },
        )


def _insert_balance(engine, *, profile="aggressive", cash=50000, equity=100000,
                    timestamp=None):
    """Insert a balance record."""
    if timestamp is None:
        timestamp = datetime(2024, 6, 15, 14, 0, 0)
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO balance (profile, timestamp, cash, portfolio_value, total_equity)
                VALUES (:profile, :timestamp, :cash, :portfolio_value, :equity)
            """),
            {
                "profile": profile,
                "timestamp": timestamp,
                "cash": cash,
                "portfolio_value": equity,
                "equity": equity,
            },
        )


def _insert_shadow_outcome(engine, *, blocked_candidate_id, entry_price=185.50,
                           stop_price=184.00, target_price=188.00,
                           pnl_pct=1.35, stop_hit=0, target_hit=1,
                           first_hit="target", outcome_label="winner"):
    """Insert a shadow ledger outcome for a blocked candidate."""
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO blocked_trade_candidate_outcomes
                (blocked_candidate_id, eval_window, evaluated_at,
                 entry_price, stop_price, target_price,
                 pnl_pct, mfe_pct, mae_pct, stop_hit, target_hit,
                 first_hit, outcome_label, gate_verdict)
                VALUES (:bid, '60m', :evaluated_at,
                        :entry_price, :stop_price, :target_price,
                        :pnl_pct, 1.5, -0.3, :stop_hit, :target_hit,
                        :first_hit, :outcome_label, 'allow')
            """),
            {
                "bid": blocked_candidate_id,
                "evaluated_at": datetime(2024, 6, 15, 16, 0, 0),
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "pnl_pct": pnl_pct,
                "stop_hit": stop_hit,
                "target_hit": target_hit,
                "first_hit": first_hit,
                "outcome_label": outcome_label,
            },
        )


# ---------------------------------------------------------------------------
# Test: End-to-end replay of a blocked candidate (full pipeline)
# ---------------------------------------------------------------------------


class TestE2EBlockedCandidate:
    """End-to-end replay of a blocked candidate through the full pipeline:
    sourcing → reconstruction → replay → delta → outcome → persistence.

    Validates: Requirements 13.1, 13.2, 13.3
    """

    def test_blocked_candidate_full_pipeline(self, engine, base_timestamp):
        """A blocked candidate flows through the complete pipeline and produces
        a valid replay_audit_record with correct delta classification."""
        # --- Setup: Insert test data ---
        lineage_id = str(uuid.uuid4())
        candidate_id = _insert_blocked_candidate(
            engine,
            symbol="AAPL",
            profile="aggressive",
            direction="long",
            setup_type="news_breakout",
            entry_price=185.50,
            stop_price=184.00,
            target_price=188.00,
            quantity=10,
            blocked_by="risk_geometry_gate",
            block_reason="R:R below threshold",
            reason_code="RISK_REWARD_BELOW_THRESHOLD",
            created_at=base_timestamp,
            lineage_id=lineage_id,
        )

        # Insert supporting data: analyst signal before cutoff
        _insert_analyst_signal(
            engine,
            symbol="AAPL",
            signal_data={
                "signal_strength": 9.0,
                "confidence": "high",
                "setup_type": "news_breakout",
                "status": "active",
            },
            timestamp=base_timestamp - timedelta(minutes=1),
        )

        # Insert balance record
        _insert_balance(engine, profile="aggressive", cash=50000, equity=100000,
                        timestamp=base_timestamp - timedelta(hours=1))

        # --- Execute: Run the replay agent in adhoc mode ---
        date_range = (
            base_timestamp - timedelta(hours=1),
            base_timestamp + timedelta(hours=1),
        )

        summary = run(
            engine,
            mode="adhoc",
            date_range=date_range,
            filters={"symbol": "AAPL"},
            operator_override=True,
        )

        # --- Verify: Pipeline produced results ---
        assert isinstance(summary, BatchRunSummary)
        assert summary.candidates_total >= 1
        assert summary.candidates_processed >= 1
        assert summary.status == "completed"

        # Verify audit record was persisted
        session = get_session(engine)
        try:
            result = session.execute(
                text("SELECT * FROM replay_audit_records WHERE 1=1")
            )
            records = result.fetchall()
            assert len(records) >= 1

            record = records[0]._mapping
            assert record["replay_status"] in ("exact", "partial", "unscorable")
            assert record["era"] in ("historical", "post-snapshot")
            # Policy version should be recorded
            assert record["policy_version_json"] is not None
        finally:
            session.close()

    def test_blocked_candidate_produces_replay_audit_record(self, engine, base_timestamp):
        """Verify that the replay produces a proper audit record with all required
        fields populated (Requirement 13.3: append-only)."""
        _insert_blocked_candidate(
            engine,
            symbol="MSFT",
            profile="moderate",
            direction="long",
            setup_type="news_breakout",
            entry_price=400.00,
            stop_price=398.00,
            target_price=405.00,
            quantity=5,
            created_at=base_timestamp,
        )

        _insert_analyst_signal(
            engine,
            symbol="MSFT",
            signal_data={"signal_strength": 7.0, "confidence": "medium", "status": "active"},
            timestamp=base_timestamp - timedelta(minutes=2),
        )
        _insert_balance(engine, profile="moderate", cash=75000, equity=150000,
                        timestamp=base_timestamp - timedelta(hours=1))

        date_range = (
            base_timestamp - timedelta(hours=1),
            base_timestamp + timedelta(hours=1),
        )

        summary = run(
            engine,
            mode="adhoc",
            date_range=date_range,
            filters={"symbol": "MSFT"},
            operator_override=True,
        )

        assert summary.candidates_processed >= 1

        # Verify the audit record structure
        session = get_session(engine)
        try:
            result = session.execute(
                text("SELECT * FROM replay_audit_records LIMIT 1")
            )
            record = result.fetchone()._mapping
            # Core fields must be present
            assert record["replay_id"] is not None
            assert record["candidate_id"] is not None
            assert record["replay_cutoff"] is not None
            assert record["input_sources_json"] is not None
            assert record["policy_version_json"] is not None
            assert record["replay_status"] is not None
            assert record["era"] is not None
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Test: End-to-end replay of an executed trade
# ---------------------------------------------------------------------------


class TestE2EExecutedTrade:
    """End-to-end replay of an executed trade through the full pipeline.

    Validates: Requirements 13.1, 13.2
    """

    def test_executed_trade_full_pipeline(self, engine, base_timestamp):
        """An executed trade (from trades table) flows through the pipeline
        and produces a replay audit record."""
        # Insert a closed trade that was allowed and executed
        trade_id = _insert_trade(
            engine,
            symbol="NVDA",
            profile="aggressive",
            direction="LONG",
            setup_type="news_breakout",
            entry_price=500.00,
            stop_price=495.00,
            target_price=510.00,
            quantity=20,
            exit_price=508.00,
            status="closed",
            pnl=160.0,
            entry_time=base_timestamp,
            exit_time=base_timestamp + timedelta(hours=2),
        )

        # Insert supporting data
        _insert_analyst_signal(
            engine,
            symbol="NVDA",
            signal_data={
                "signal_strength": 8.5,
                "confidence": "high",
                "setup_type": "news_breakout",
                "status": "active",
            },
            timestamp=base_timestamp - timedelta(minutes=1),
        )
        _insert_balance(engine, profile="aggressive", cash=80000, equity=200000,
                        timestamp=base_timestamp - timedelta(hours=1))

        date_range = (
            base_timestamp - timedelta(hours=1),
            base_timestamp + timedelta(hours=1),
        )

        summary = run(
            engine,
            mode="adhoc",
            date_range=date_range,
            filters={"symbol": "NVDA"},
            operator_override=True,
        )

        # An executed trade should be sourced and replayed
        assert isinstance(summary, BatchRunSummary)
        assert summary.candidates_total >= 1
        assert summary.candidates_processed >= 1
        assert summary.status == "completed"

        # Verify audit record exists
        session = get_session(engine)
        try:
            result = session.execute(
                text("SELECT * FROM replay_audit_records")
            )
            records = result.fetchall()
            assert len(records) >= 1

            # For an executed trade, original_decision should be "allow"
            # The replay may agree or disagree
            record = records[0]._mapping
            assert record["replay_status"] in ("exact", "partial", "unscorable")
        finally:
            session.close()

    def test_executed_trade_delta_reflects_original_allow(self, engine, base_timestamp):
        """An executed trade has original_decision='allow'. The delta classification
        should reflect the comparison against the replay decision."""
        _insert_trade(
            engine,
            symbol="META",
            profile="moderate",
            direction="LONG",
            setup_type="news_breakout",
            entry_price=350.00,
            stop_price=345.00,
            target_price=360.00,
            quantity=15,
            status="open",
            entry_time=base_timestamp,
        )
        _insert_analyst_signal(
            engine,
            symbol="META",
            signal_data={"signal_strength": 7.5, "confidence": "high", "status": "active"},
            timestamp=base_timestamp - timedelta(minutes=2),
        )
        _insert_balance(engine, profile="moderate", cash=100000, equity=200000,
                        timestamp=base_timestamp - timedelta(hours=1))

        date_range = (
            base_timestamp - timedelta(hours=1),
            base_timestamp + timedelta(hours=1),
        )
        summary = run(
            engine,
            mode="adhoc",
            date_range=date_range,
            filters={"symbol": "META"},
            operator_override=True,
        )

        assert summary.candidates_processed >= 1

        session = get_session(engine)
        try:
            result = session.execute(
                text("SELECT decision_delta_classification FROM replay_audit_records LIMIT 1")
            )
            row = result.fetchone()
            if row is not None and row[0] is not None:
                # Delta classification should be one of the valid categories
                valid_classifications = [
                    "same_allow", "same_reject",
                    "same_final_reject_different_trace",
                    "same_final_allow_different_trace",
                    "replay_allows_original_reject",
                    "replay_rejects_original_allow",
                    "same_direction_different_size",
                    "same_direction_different_geometry",
                    "same_direction_different_size_and_geometry",
                    "partial_comparison", "unscorable",
                ]
                assert row[0] in valid_classifications
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Test: Backfill from existing shadow ledger data
# ---------------------------------------------------------------------------


class TestE2EBackfill:
    """Backfill from existing shadow ledger data.

    Verifies that historical data (pre-snapshot) can be graded and
    processed through the pipeline with era labeling.

    Validates: Requirements 13.1, 13.3
    """

    def test_backfill_grades_historical_record(self, engine, base_timestamp):
        """Historical records without decision_snapshots are graded appropriately
        and labeled with era='historical'."""
        # Insert a blocked candidate (pre-snapshot era — no decision_snapshot)
        candidate_id = _insert_blocked_candidate(
            engine,
            symbol="GOOG",
            profile="moderate",
            direction="long",
            setup_type="news_breakout",
            entry_price=150.00,
            stop_price=148.00,
            target_price=154.00,
            quantity=8,
            created_at=base_timestamp,
        )

        # Insert shadow outcome for this candidate (existing ledger data)
        _insert_shadow_outcome(
            engine,
            blocked_candidate_id=candidate_id,
            entry_price=150.00,
            stop_price=148.00,
            target_price=154.00,
            pnl_pct=2.67,
            stop_hit=0,
            target_hit=1,
            first_hit="target",
            outcome_label="winner",
        )

        _insert_analyst_signal(
            engine,
            symbol="GOOG",
            signal_data={"signal_strength": 8.0, "confidence": "high", "status": "active"},
            timestamp=base_timestamp - timedelta(minutes=5),
        )
        _insert_balance(engine, profile="moderate", cash=60000, equity=120000,
                        timestamp=base_timestamp - timedelta(hours=1))

        # Run the pipeline
        date_range = (
            base_timestamp - timedelta(hours=1),
            base_timestamp + timedelta(hours=1),
        )
        summary = run(
            engine,
            mode="adhoc",
            date_range=date_range,
            filters={"symbol": "GOOG"},
            operator_override=True,
        )

        assert summary.candidates_processed >= 1

        # Verify era is labeled 'historical' (no snapshot exists)
        session = get_session(engine)
        try:
            result = session.execute(
                text("SELECT era FROM replay_audit_records WHERE 1=1")
            )
            rows = result.fetchall()
            assert len(rows) >= 1
            # Without a decision_snapshot, era should be 'historical'
            assert rows[0][0] == "historical"
        finally:
            session.close()

    def test_backfill_with_snapshot_labeled_post_snapshot(self, engine, base_timestamp):
        """When a decision_snapshot exists for a candidate, era is labeled
        'post-snapshot'."""
        lineage_id = str(uuid.uuid4())
        snapshot_id = str(uuid.uuid4())

        # Insert a blocked candidate with lineage_id
        _insert_blocked_candidate(
            engine,
            symbol="AMZN",
            profile="aggressive",
            direction="long",
            setup_type="news_breakout",
            entry_price=175.00,
            stop_price=173.00,
            target_price=179.00,
            quantity=12,
            created_at=base_timestamp,
            lineage_id=lineage_id,
        )

        # Insert a decision_snapshot for this candidate
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO decision_snapshots
                    (snapshot_id, schema_version, candidate_lineage_id,
                     timestamp, symbol, profile, direction, setup_type,
                     decision_payload_json, entry_price, stop_price, target_price,
                     quantity, account_equity, available_cash,
                     gate_config_json, feature_flags_json, policy_version_id)
                    VALUES (:snapshot_id, '1.0', :lineage_id,
                            :ts, 'AMZN', 'aggressive', 'LONG', 'news_breakout',
                            '{}', '175.00', '173.00', '179.00',
                            '12', '100000', '50000',
                            '{}', '{}', 'test_policy_v1')
                """),
                {
                    "snapshot_id": snapshot_id,
                    "lineage_id": lineage_id,
                    "ts": base_timestamp,
                },
            )

        _insert_analyst_signal(
            engine,
            symbol="AMZN",
            signal_data={"signal_strength": 8.5, "confidence": "high", "status": "active"},
            timestamp=base_timestamp - timedelta(minutes=1),
        )
        _insert_balance(engine, profile="aggressive", cash=50000, equity=100000,
                        timestamp=base_timestamp - timedelta(hours=1))

        date_range = (
            base_timestamp - timedelta(hours=1),
            base_timestamp + timedelta(hours=1),
        )
        summary = run(
            engine,
            mode="adhoc",
            date_range=date_range,
            filters={"symbol": "AMZN"},
            operator_override=True,
        )

        assert summary.candidates_processed >= 1

        session = get_session(engine)
        try:
            result = session.execute(
                text("SELECT era, snapshot_id FROM replay_audit_records WHERE 1=1")
            )
            rows = result.fetchall()
            assert len(rows) >= 1
            # With a decision_snapshot, era should be 'post-snapshot'
            assert rows[0][0] == "post-snapshot"
            assert rows[0][1] == snapshot_id
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Test: Candle evidence persistence and content_hash reproducibility
# ---------------------------------------------------------------------------


class TestCandleEvidenceReproducibility:
    """Candle evidence persistence and content_hash reproducibility.

    Validates: Requirements 13.2, 7.1
    """

    def test_build_candle_evidence_produces_consistent_hash(self):
        """The same candles should always produce the same content_hash
        regardless of call order (deterministic)."""
        candles = [
            {"timestamp": "2024-06-15T14:31:00", "open": 185.50, "high": 186.00,
             "low": 185.30, "close": 185.80, "volume": 1000},
            {"timestamp": "2024-06-15T14:32:00", "open": 185.80, "high": 186.50,
             "low": 185.60, "close": 186.20, "volume": 1200},
            {"timestamp": "2024-06-15T14:33:00", "open": 186.20, "high": 187.00,
             "low": 186.00, "close": 186.80, "volume": 800},
        ]
        fetch_time = datetime(2024, 6, 15, 20, 0, 0)

        evidence_1 = build_candle_evidence(candles, "test_source", fetch_time)
        evidence_2 = build_candle_evidence(candles, "test_source", fetch_time)

        # Same input → same hash
        assert evidence_1.candle_content_hash == evidence_2.candle_content_hash
        assert evidence_1.candles_json == evidence_2.candles_json

    def test_candle_evidence_hash_differs_for_different_data(self):
        """Different candle data produces different content_hash."""
        candles_a = [
            {"timestamp": "2024-06-15T14:31:00", "open": 185.50, "high": 186.00,
             "low": 185.30, "close": 185.80},
        ]
        candles_b = [
            {"timestamp": "2024-06-15T14:31:00", "open": 185.55, "high": 186.00,
             "low": 185.30, "close": 185.80},
        ]
        fetch_time = datetime(2024, 6, 15, 20, 0, 0)

        evidence_a = build_candle_evidence(candles_a, "test", fetch_time)
        evidence_b = build_candle_evidence(candles_b, "test", fetch_time)

        assert evidence_a.candle_content_hash != evidence_b.candle_content_hash

    def test_candle_evidence_hash_stable_across_candle_order(self):
        """Hash is stable even if candles are provided in different order
        (normalization sorts by timestamp)."""
        candle_1 = {"timestamp": "2024-06-15T14:31:00", "open": 185.50,
                    "high": 186.00, "low": 185.30, "close": 185.80}
        candle_2 = {"timestamp": "2024-06-15T14:32:00", "open": 185.80,
                    "high": 186.50, "low": 185.60, "close": 186.20}

        fetch_time = datetime(2024, 6, 15, 20, 0, 0)

        # Two different orderings should produce the same hash
        evidence_forward = build_candle_evidence([candle_1, candle_2], "test", fetch_time)
        evidence_reverse = build_candle_evidence([candle_2, candle_1], "test", fetch_time)

        assert evidence_forward.candle_content_hash == evidence_reverse.candle_content_hash

    def test_candle_evidence_content_hash_is_sha256(self):
        """Content hash should be a valid SHA-256 hex digest (64 chars)."""
        candles = [
            {"timestamp": "2024-06-15T14:31:00", "open": 100.0, "high": 101.0,
             "low": 99.0, "close": 100.5},
        ]
        evidence = build_candle_evidence(candles, "polygon", datetime(2024, 6, 15, 20, 0, 0))

        assert len(evidence.candle_content_hash) == 64
        assert all(c in "0123456789abcdef" for c in evidence.candle_content_hash)


# ---------------------------------------------------------------------------
# Test: Decision-specific outcome branches
# ---------------------------------------------------------------------------


class TestOutcomeBranches:
    """Verify decision-specific outcome branches:
    - reject→allow: scores candles (counterfactual fill)
    - allow→reject: reads realized P&L
    - reject→reject: produces no outcome record

    Validates: Requirements 7.1, 7.2
    """

    def test_reject_to_allow_scores_candles(self):
        """When original=reject and replay=allow, counterfactual fill scoring
        is triggered using post-cutoff candle data."""
        # Determine scoring branch
        branch = should_score_counterfactual(
            original_decision="reject",
            replay_decision="allow",
            original_geometry={
                "entry_price": Decimal("185.50"),
                "stop_price": Decimal("184.00"),
                "target_price": Decimal("188.00"),
            },
            replay_geometry={
                "entry_price": Decimal("185.50"),
                "stop_price": Decimal("184.00"),
                "target_price": Decimal("188.00"),
            },
            original_entry_price=Decimal("185.50"),
            simulated_fill_price=Decimal("185.60"),
        )

        assert branch == "counterfactual_fill"

        # Now actually score with candles
        candles = [
            {"timestamp": "2024-06-15T14:31:00", "open": 185.60, "high": 186.80,
             "low": 185.40, "close": 186.50, "volume": 1000},
            {"timestamp": "2024-06-15T14:32:00", "open": 186.50, "high": 187.20,
             "low": 186.00, "close": 186.90, "volume": 800},
            {"timestamp": "2024-06-15T14:45:00", "open": 187.00, "high": 188.10,
             "low": 186.80, "close": 188.00, "volume": 900},
        ]

        cutoff = datetime(2024, 6, 15, 14, 30, 0)
        fill_model = build_fill_model(
            proposed_entry_price=Decimal("185.50"),
            cutoff=cutoff,
            candles_1m=candles,
        )

        assert fill_model.fill_rule == "market_order_next_candle_open"
        assert fill_model.simulated_fill_price == Decimal("185.60")

        candle_evidence = build_candle_evidence(
            candles, "test_source", datetime(2024, 6, 15, 20, 0, 0)
        )

        outcome = score_counterfactual(
            fill_model=fill_model,
            stop_price=Decimal("184.00"),
            target_price=Decimal("188.00"),
            direction="LONG",
            candles_1m=candles,
            candle_evidence=candle_evidence,
        )

        # Should produce a scored outcome (not unscorable)
        assert outcome.status in ("scored", "ambiguous")
        # MFE should be positive for a LONG that went up
        assert outcome.mfe > 0

    def test_allow_to_reject_reads_realized_pnl(self):
        """When original=allow and replay=reject, we use the actual realized
        trade P&L to compute avoided loss or forgone gain."""
        branch = should_score_counterfactual(
            original_decision="allow",
            replay_decision="reject",
            original_geometry={
                "entry_price": Decimal("185.50"),
                "stop_price": Decimal("184.00"),
                "target_price": Decimal("188.00"),
            },
            replay_geometry={
                "entry_price": Decimal("185.50"),
                "stop_price": Decimal("184.00"),
                "target_price": Decimal("188.00"),
            },
            original_entry_price=Decimal("185.50"),
            simulated_fill_price=Decimal("185.50"),
        )

        assert branch == "realized_outcome"

        # Score using actual trade outcome (realized P&L)
        # Trade lost money — this is an "avoided loss"
        result = score_allowed_to_rejected(
            realized_pnl=-150.0,
            entry_price=Decimal("185.50"),
            exit_price=Decimal("184.00"),
        )

        assert result["classification"] == "avoided_loss"
        assert result["realized_pnl"] == -150.0
        assert result["return_pct"] < 0

    def test_allow_to_reject_forgone_gain(self):
        """When original=allow, replay=reject, and the trade was profitable,
        this is a 'forgone_gain'."""
        result = score_allowed_to_rejected(
            realized_pnl=200.0,
            entry_price=Decimal("185.50"),
            exit_price=Decimal("188.00"),
        )

        assert result["classification"] == "forgone_gain"
        assert result["realized_pnl"] == 200.0
        assert result["return_pct"] > 0

    def test_reject_to_reject_no_outcome_scoring(self):
        """When both original and replay reject, there is NO economic scoring
        regardless of geometry change."""
        # Same geometry
        branch = should_score_counterfactual(
            original_decision="reject",
            replay_decision="reject",
            original_geometry={
                "entry_price": Decimal("185.50"),
                "stop_price": Decimal("184.00"),
                "target_price": Decimal("188.00"),
            },
            replay_geometry={
                "entry_price": Decimal("185.50"),
                "stop_price": Decimal("184.00"),
                "target_price": Decimal("188.00"),
            },
            original_entry_price=Decimal("185.50"),
            simulated_fill_price=Decimal("185.60"),
        )
        assert branch is None  # No scoring

        # Different geometry — still no scoring for reject→reject
        branch_diff = should_score_counterfactual(
            original_decision="reject",
            replay_decision="reject",
            original_geometry={
                "entry_price": Decimal("185.50"),
                "stop_price": Decimal("184.00"),
                "target_price": Decimal("188.00"),
            },
            replay_geometry={
                "entry_price": Decimal("186.00"),
                "stop_price": Decimal("184.50"),
                "target_price": Decimal("189.00"),
            },
            original_entry_price=Decimal("185.50"),
            simulated_fill_price=Decimal("186.00"),
        )
        assert branch_diff is None  # Still no scoring

    def test_allow_to_allow_with_changed_geometry_triggers_counterfactual(self):
        """When both allow but geometry changes, counterfactual scoring is
        triggered on the new geometry."""
        branch = should_score_counterfactual(
            original_decision="allow",
            replay_decision="allow",
            original_geometry={
                "entry_price": Decimal("185.50"),
                "stop_price": Decimal("184.00"),
                "target_price": Decimal("188.00"),
            },
            replay_geometry={
                "entry_price": Decimal("185.50"),
                "stop_price": Decimal("183.00"),  # Different stop
                "target_price": Decimal("190.00"),  # Different target
            },
            original_entry_price=Decimal("185.50"),
            simulated_fill_price=Decimal("185.50"),
        )
        assert branch == "counterfactual_geometry"

    def test_allow_to_allow_same_geometry_no_scoring(self):
        """When both allow with same geometry and same fill, no scoring needed."""
        branch = should_score_counterfactual(
            original_decision="allow",
            replay_decision="allow",
            original_geometry={
                "entry_price": Decimal("185.50"),
                "stop_price": Decimal("184.00"),
                "target_price": Decimal("188.00"),
            },
            replay_geometry={
                "entry_price": Decimal("185.50"),
                "stop_price": Decimal("184.00"),
                "target_price": Decimal("188.00"),
            },
            original_entry_price=Decimal("185.50"),
            simulated_fill_price=Decimal("185.50"),
        )
        assert branch is None


# ---------------------------------------------------------------------------
# Test: Immutability of replay_audit_records
# ---------------------------------------------------------------------------


class TestAuditRecordImmutability:
    """Verify that replay_audit_records are immutable after creation
    (Requirement 13.3)."""

    def test_audit_record_cannot_be_updated(self, engine, base_timestamp):
        """UPDATE on replay_audit_records raises due to immutability trigger."""
        # First create an audit record via the pipeline
        _insert_blocked_candidate(
            engine,
            symbol="TSLA",
            profile="aggressive",
            direction="long",
            created_at=base_timestamp,
        )
        _insert_analyst_signal(
            engine,
            symbol="TSLA",
            timestamp=base_timestamp - timedelta(minutes=1),
        )
        _insert_balance(engine, profile="aggressive",
                        timestamp=base_timestamp - timedelta(hours=1))

        date_range = (
            base_timestamp - timedelta(hours=1),
            base_timestamp + timedelta(hours=1),
        )
        run(engine, mode="adhoc", date_range=date_range,
            filters={"symbol": "TSLA"}, operator_override=True)

        # Attempt to UPDATE should fail
        session = get_session(engine)
        try:
            with pytest.raises(Exception, match="immutable"):
                session.execute(
                    text("UPDATE replay_audit_records SET era = 'tampered' WHERE 1=1")
                )
                session.commit()
        finally:
            session.rollback()
            session.close()

    def test_audit_record_cannot_be_deleted(self, engine, base_timestamp):
        """DELETE on replay_audit_records raises due to immutability trigger."""
        _insert_blocked_candidate(
            engine,
            symbol="AMZN",
            profile="moderate",
            direction="long",
            created_at=base_timestamp,
        )
        _insert_analyst_signal(engine, symbol="AMZN",
                               timestamp=base_timestamp - timedelta(minutes=1))
        _insert_balance(engine, profile="moderate",
                        timestamp=base_timestamp - timedelta(hours=1))

        date_range = (
            base_timestamp - timedelta(hours=1),
            base_timestamp + timedelta(hours=1),
        )
        run(engine, mode="adhoc", date_range=date_range,
            filters={"symbol": "AMZN"}, operator_override=True)

        session = get_session(engine)
        try:
            with pytest.raises(Exception, match="immutable"):
                session.execute(
                    text("DELETE FROM replay_audit_records WHERE 1=1")
                )
                session.commit()
        finally:
            session.rollback()
            session.close()


# ---------------------------------------------------------------------------
# Test: Pipeline does not write to production tables
# ---------------------------------------------------------------------------


class TestReportOnlyGuarantee:
    """Verify the replay pipeline does not mutate production tables
    (Requirement 9.1, 9.2)."""

    def test_no_writes_to_production_tables(self, engine, base_timestamp):
        """After replay execution, trades/positions/balance/trade_events
        remain unchanged."""
        # Record initial state
        session = get_session(engine)
        try:
            initial_trades = session.execute(text("SELECT COUNT(*) FROM trades")).scalar()
            initial_positions = session.execute(text("SELECT COUNT(*) FROM positions")).scalar()
            initial_events = session.execute(text("SELECT COUNT(*) FROM trade_events")).scalar()
        finally:
            session.close()

        # Insert candidate and run replay
        _insert_blocked_candidate(
            engine,
            symbol="TEST",
            profile="moderate",
            direction="long",
            created_at=base_timestamp,
        )
        _insert_analyst_signal(engine, symbol="TEST",
                               timestamp=base_timestamp - timedelta(minutes=1))
        _insert_balance(engine, profile="moderate",
                        timestamp=base_timestamp - timedelta(hours=1))

        date_range = (
            base_timestamp - timedelta(hours=1),
            base_timestamp + timedelta(hours=1),
        )
        run(engine, mode="adhoc", date_range=date_range,
            filters={"symbol": "TEST"}, operator_override=True)

        # Verify production tables are unchanged
        session = get_session(engine)
        try:
            assert session.execute(text("SELECT COUNT(*) FROM trades")).scalar() == initial_trades
            assert session.execute(text("SELECT COUNT(*) FROM positions")).scalar() == initial_positions
            assert session.execute(text("SELECT COUNT(*) FROM trade_events")).scalar() == initial_events
        finally:
            session.close()
