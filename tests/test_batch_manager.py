"""Tests for core/replay/batch_manager.py — Batch state management.

Tests cover:
- Creating batch_items rows (status='pending') for each candidate at batch start
- Atomic persistence: audit record INSERT + batch_item status → 'completed'
- Failure handling: persist failure audit record + batch_item → 'failed' with reason code
- Batch resumption: pick up from first 'pending' item in processing_order
- Idempotent replay: same candidate + same policy_version → return cached result
- Completion statistics

Requirements: 10.5, 10.6, 10.7, 10.8, 10.9
"""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, text

from core.replay.batch_manager import (
    BatchManager,
    BatchRunSummary,
    BatchItemResult,
    FAILURE_REASON_CODES,
)
from core.replay.policy_version import PolicyVersion
from db.replay_schema import init_replay_db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """In-memory SQLite engine with replay schema initialized."""
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        # Create minimal versions of existing production tables for migrations
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
    init_replay_db(eng)
    return eng


@pytest.fixture
def session(engine):
    """SQLAlchemy session for the test engine."""
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine)
    sess = Session()
    yield sess
    sess.close()


@pytest.fixture
def policy_version():
    """A test PolicyVersion."""
    return PolicyVersion(
        name="test_policy",
        gate_revision="abc123",
        config_digest="digest_aaa",
        feature_flags={"SETUP_SPECIFIC_RR_THRESHOLDS": True},
        benchmark_version=None,
        config_source_timestamp=datetime(2024, 1, 1, 12, 0, 0),
        gate_ordering_version="v1.0",
        adapter_version="1.0.0",
    )


@pytest.fixture
def alt_policy_version():
    """A different PolicyVersion (different config_digest)."""
    return PolicyVersion(
        name="alt_policy",
        gate_revision="def456",
        config_digest="digest_bbb",
        feature_flags={"SETUP_SPECIFIC_RR_THRESHOLDS": False},
        benchmark_version=None,
        config_source_timestamp=datetime(2024, 2, 1, 12, 0, 0),
        gate_ordering_version="v1.0",
        adapter_version="1.0.0",
    )


@dataclass(frozen=True)
class FakeCandidate:
    """Minimal candidate for testing batch creation."""
    candidate_id: str
    symbol: str = "TSLA"
    profile: str = "aggressive"
    direction: str = "LONG"
    entry_timestamp: datetime = datetime(2024, 1, 15, 14, 30, 0)


def make_candidates(n: int) -> list[FakeCandidate]:
    """Create n fake candidates with unique IDs."""
    return [
        FakeCandidate(candidate_id=str(uuid.uuid4()))
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests: Batch creation
# ---------------------------------------------------------------------------


class TestBatchCreation:
    """Tests for creating batch_runs and batch_items."""

    def test_create_batch_run_inserts_run_record(
        self, session, policy_version
    ):
        manager = BatchManager(session)
        candidates = make_candidates(3)

        batch_run_id = manager.create_batch_run(
            mode="batch",
            policy_version=policy_version,
            candidates=candidates,
        )

        # Verify batch_run record exists
        result = session.execute(
            text("SELECT * FROM replay_batch_runs WHERE batch_run_id = :id"),
            {"id": batch_run_id},
        )
        row = result.fetchone()
        assert row is not None
        # Columns: id, batch_run_id, started_at, ended_at, mode, policy_version_json, ...
        assert row[4] == "batch"  # mode
        assert row[7] == 3  # candidates_total

    def test_create_batch_items_all_pending(self, session, policy_version):
        manager = BatchManager(session)
        candidates = make_candidates(5)

        batch_run_id = manager.create_batch_run(
            mode="batch",
            policy_version=policy_version,
            candidates=candidates,
        )

        # Verify all batch items are pending
        result = session.execute(
            text(
                "SELECT candidate_id, processing_order, status "
                "FROM replay_batch_items "
                "WHERE batch_run_id = :id ORDER BY processing_order"
            ),
            {"id": batch_run_id},
        )
        rows = result.fetchall()
        assert len(rows) == 5
        for i, row in enumerate(rows, start=1):
            assert row[1] == i  # processing_order
            assert row[2] == "pending"  # status

    def test_create_batch_preserves_candidate_order(
        self, session, policy_version
    ):
        manager = BatchManager(session)
        candidates = make_candidates(3)
        expected_ids = [c.candidate_id for c in candidates]

        batch_run_id = manager.create_batch_run(
            mode="batch",
            policy_version=policy_version,
            candidates=candidates,
        )

        result = session.execute(
            text(
                "SELECT candidate_id FROM replay_batch_items "
                "WHERE batch_run_id = :id ORDER BY processing_order"
            ),
            {"id": batch_run_id},
        )
        actual_ids = [row[0] for row in result.fetchall()]
        assert actual_ids == expected_ids

    def test_create_batch_stores_watermarks(self, session, policy_version):
        manager = BatchManager(session)
        candidates = make_candidates(2)
        ws = datetime(2024, 1, 10, 0, 0, 0)
        we = datetime(2024, 1, 15, 23, 59, 59)

        batch_run_id = manager.create_batch_run(
            mode="batch",
            policy_version=policy_version,
            candidates=candidates,
            watermark_start=ws,
            watermark_end=we,
        )

        result = session.execute(
            text(
                "SELECT watermark_start, watermark_end "
                "FROM replay_batch_runs WHERE batch_run_id = :id"
            ),
            {"id": batch_run_id},
        )
        row = result.fetchone()
        assert row[0] == ws.isoformat()
        assert row[1] == we.isoformat()


# ---------------------------------------------------------------------------
# Tests: Atomic persistence (complete_item_atomic)
# ---------------------------------------------------------------------------


class TestAtomicPersistence:
    """Tests for atomic audit record INSERT + batch_item completion."""

    def _setup_batch(self, session, policy_version):
        """Create a batch with one candidate in 'processing' state."""
        manager = BatchManager(session)
        candidates = make_candidates(1)
        batch_run_id = manager.create_batch_run(
            mode="batch",
            policy_version=policy_version,
            candidates=candidates,
        )
        candidate_id = candidates[0].candidate_id
        manager.mark_item_processing(batch_run_id, candidate_id)
        return manager, batch_run_id, candidate_id

    def test_complete_item_inserts_audit_record(
        self, session, policy_version
    ):
        manager, batch_run_id, candidate_id = self._setup_batch(
            session, policy_version
        )
        replay_id = str(uuid.uuid4())
        audit_record = {
            "replay_id": replay_id,
            "source_candidate_ids_json": json.dumps([candidate_id]),
            "snapshot_id": None,
            "replay_cutoff": datetime(2024, 1, 15, 14, 30, 0).isoformat(),
            "input_sources_json": "{}",
            "policy_version_json": json.dumps({"config_digest": "digest_aaa"}),
            "replay_status": "exact",
            "gate_trace_json": json.dumps({"entries": []}),
            "decision_delta_classification": "same_allow",
            "decision_delta_json": json.dumps({}),
            "counterfactual_outcome_json": None,
            "divergence_cause": None,
            "divergence_evidence_json": None,
            "code_revision": "abc123",
            "era": "post-snapshot",
            "diagnostic_mode": 0,
        }

        manager.complete_item_atomic(
            batch_run_id=batch_run_id,
            candidate_id=candidate_id,
            replay_id=replay_id,
            audit_record=audit_record,
        )

        # Verify audit record exists
        result = session.execute(
            text(
                "SELECT replay_id, replay_status "
                "FROM replay_audit_records WHERE replay_id = :id"
            ),
            {"id": replay_id},
        )
        row = result.fetchone()
        assert row is not None
        assert row[0] == replay_id
        assert row[1] == "exact"

    def test_complete_item_updates_batch_item_status(
        self, session, policy_version
    ):
        manager, batch_run_id, candidate_id = self._setup_batch(
            session, policy_version
        )
        replay_id = str(uuid.uuid4())
        audit_record = {
            "replay_id": replay_id,
            "replay_cutoff": datetime(2024, 1, 15, 14, 30, 0).isoformat(),
            "input_sources_json": "{}",
            "policy_version_json": json.dumps({"config_digest": "digest_aaa"}),
            "replay_status": "exact",
        }

        manager.complete_item_atomic(
            batch_run_id=batch_run_id,
            candidate_id=candidate_id,
            replay_id=replay_id,
            audit_record=audit_record,
        )

        # Verify batch item is completed
        result = session.execute(
            text(
                "SELECT status, replay_id FROM replay_batch_items "
                "WHERE batch_run_id = :bid AND candidate_id = :cid"
            ),
            {"bid": batch_run_id, "cid": candidate_id},
        )
        row = result.fetchone()
        assert row[0] == "completed"
        assert row[1] == replay_id


# ---------------------------------------------------------------------------
# Tests: Failure handling
# ---------------------------------------------------------------------------


class TestFailureHandling:
    """Tests for failure handling: item marked failed, batch continues."""

    def test_fail_item_marks_batch_item_failed(self, session, policy_version):
        manager = BatchManager(session)
        candidates = make_candidates(3)
        batch_run_id = manager.create_batch_run(
            mode="batch",
            policy_version=policy_version,
            candidates=candidates,
        )
        failing_id = candidates[1].candidate_id
        manager.mark_item_processing(batch_run_id, failing_id)

        manager.fail_item(
            batch_run_id=batch_run_id,
            candidate_id=failing_id,
            reason_code="gate_execution_error",
            details="Gate raised ValueError",
        )

        # Verify the item is failed
        result = session.execute(
            text(
                "SELECT status, failure_reason_code, failure_details "
                "FROM replay_batch_items "
                "WHERE batch_run_id = :bid AND candidate_id = :cid"
            ),
            {"bid": batch_run_id, "cid": failing_id},
        )
        row = result.fetchone()
        assert row[0] == "failed"
        assert row[1] == "gate_execution_error"
        assert row[2] == "Gate raised ValueError"

    def test_fail_item_other_items_remain_pending(
        self, session, policy_version
    ):
        """Failure of one item does NOT affect other pending items."""
        manager = BatchManager(session)
        candidates = make_candidates(3)
        batch_run_id = manager.create_batch_run(
            mode="batch",
            policy_version=policy_version,
            candidates=candidates,
        )
        failing_id = candidates[0].candidate_id
        manager.mark_item_processing(batch_run_id, failing_id)

        manager.fail_item(
            batch_run_id=batch_run_id,
            candidate_id=failing_id,
            reason_code="missing_input",
        )

        # Other items are still pending
        result = session.execute(
            text(
                "SELECT status FROM replay_batch_items "
                "WHERE batch_run_id = :bid AND candidate_id != :cid"
            ),
            {"bid": batch_run_id, "cid": failing_id},
        )
        rows = result.fetchall()
        assert all(row[0] == "pending" for row in rows)

    def test_fail_item_persists_failure_audit_record(
        self, session, policy_version
    ):
        manager = BatchManager(session)
        candidates = make_candidates(1)
        batch_run_id = manager.create_batch_run(
            mode="batch",
            policy_version=policy_version,
            candidates=candidates,
        )
        candidate_id = candidates[0].candidate_id
        manager.mark_item_processing(batch_run_id, candidate_id)

        fail_replay_id = str(uuid.uuid4())
        manager.fail_item(
            batch_run_id=batch_run_id,
            candidate_id=candidate_id,
            reason_code="snapshot_corrupt",
            details="JSON decode error in snapshot",
            audit_record={
                "replay_id": fail_replay_id,
                "source_candidate_ids_json": json.dumps([candidate_id]),
                "replay_cutoff": datetime(2024, 1, 15).isoformat(),
                "input_sources_json": "{}",
                "policy_version_json": json.dumps(
                    {"config_digest": "digest_aaa"}
                ),
            },
        )

        # Verify failure audit record exists
        result = session.execute(
            text(
                "SELECT replay_status, failure_reason_code "
                "FROM replay_audit_records WHERE replay_id = :id"
            ),
            {"id": fail_replay_id},
        )
        row = result.fetchone()
        assert row is not None
        assert row[0] == "failed"
        assert row[1] == "snapshot_corrupt"

    def test_fail_item_normalizes_unknown_reason_code(
        self, session, policy_version
    ):
        manager = BatchManager(session)
        candidates = make_candidates(1)
        batch_run_id = manager.create_batch_run(
            mode="batch",
            policy_version=policy_version,
            candidates=candidates,
        )
        candidate_id = candidates[0].candidate_id
        manager.mark_item_processing(batch_run_id, candidate_id)

        manager.fail_item(
            batch_run_id=batch_run_id,
            candidate_id=candidate_id,
            reason_code="some_invalid_code",
        )

        result = session.execute(
            text(
                "SELECT failure_reason_code FROM replay_batch_items "
                "WHERE batch_run_id = :bid AND candidate_id = :cid"
            ),
            {"bid": batch_run_id, "cid": candidate_id},
        )
        row = result.fetchone()
        assert row[0] == "unknown_error"


# ---------------------------------------------------------------------------
# Tests: Batch resumption
# ---------------------------------------------------------------------------


class TestBatchResumption:
    """Tests for batch resumption from first pending item."""

    def test_find_resumable_batch_returns_existing(
        self, session, policy_version
    ):
        manager = BatchManager(session)
        candidates = make_candidates(3)
        batch_run_id = manager.create_batch_run(
            mode="batch",
            policy_version=policy_version,
            candidates=candidates,
        )

        # Should find the running batch
        found = manager.find_resumable_batch(
            mode="batch", policy_version=policy_version
        )
        assert found == batch_run_id

    def test_find_resumable_batch_returns_none_when_all_done(
        self, session, policy_version
    ):
        manager = BatchManager(session)
        candidates = make_candidates(1)
        batch_run_id = manager.create_batch_run(
            mode="batch",
            policy_version=policy_version,
            candidates=candidates,
        )
        candidate_id = candidates[0].candidate_id
        manager.mark_item_processing(batch_run_id, candidate_id)

        # Complete the only item
        replay_id = str(uuid.uuid4())
        manager.complete_item_atomic(
            batch_run_id=batch_run_id,
            candidate_id=candidate_id,
            replay_id=replay_id,
            audit_record={
                "replay_id": replay_id,
                "replay_cutoff": datetime(2024, 1, 15).isoformat(),
                "input_sources_json": "{}",
                "policy_version_json": json.dumps(
                    {"config_digest": "digest_aaa"}
                ),
                "replay_status": "exact",
            },
        )

        # No pending items → no resumable batch
        found = manager.find_resumable_batch(
            mode="batch", policy_version=policy_version
        )
        assert found is None

    def test_find_resumable_batch_wrong_policy_returns_none(
        self, session, policy_version, alt_policy_version
    ):
        manager = BatchManager(session)
        candidates = make_candidates(2)
        manager.create_batch_run(
            mode="batch",
            policy_version=policy_version,
            candidates=candidates,
        )

        # Different policy digest → not resumable
        found = manager.find_resumable_batch(
            mode="batch", policy_version=alt_policy_version
        )
        assert found is None

    def test_get_pending_items_returns_from_first_pending(
        self, session, policy_version
    ):
        manager = BatchManager(session)
        candidates = make_candidates(5)
        batch_run_id = manager.create_batch_run(
            mode="batch",
            policy_version=policy_version,
            candidates=candidates,
        )

        # Complete first two items
        for c in candidates[:2]:
            manager.mark_item_processing(batch_run_id, c.candidate_id)
            replay_id = str(uuid.uuid4())
            manager.complete_item_atomic(
                batch_run_id=batch_run_id,
                candidate_id=c.candidate_id,
                replay_id=replay_id,
                audit_record={
                    "replay_id": replay_id,
                    "replay_cutoff": datetime(2024, 1, 15).isoformat(),
                    "input_sources_json": "{}",
                    "policy_version_json": json.dumps(
                        {"config_digest": "digest_aaa"}
                    ),
                    "replay_status": "exact",
                },
            )

        # Should return items 3, 4, 5 as pending
        pending = manager.get_pending_items(batch_run_id)
        assert len(pending) == 3
        assert pending[0]["processing_order"] == 3
        assert pending[0]["candidate_id"] == candidates[2].candidate_id

    def test_resume_suspended_batch(self, session, policy_version):
        manager = BatchManager(session)
        candidates = make_candidates(3)
        batch_run_id = manager.create_batch_run(
            mode="batch",
            policy_version=policy_version,
            candidates=candidates,
        )

        # Suspend the batch
        manager.suspend_batch(batch_run_id)

        # Should still be found as resumable
        found = manager.find_resumable_batch(
            mode="batch", policy_version=policy_version
        )
        assert found == batch_run_id


# ---------------------------------------------------------------------------
# Tests: Idempotent replay
# ---------------------------------------------------------------------------


class TestIdempotentReplay:
    """Tests for idempotent replay: same candidate + policy → cached result."""

    def test_cached_result_returned_for_same_policy(
        self, session, policy_version
    ):
        manager = BatchManager(session)
        candidates = make_candidates(1)
        batch_run_id = manager.create_batch_run(
            mode="batch",
            policy_version=policy_version,
            candidates=candidates,
        )
        candidate_id = candidates[0].candidate_id
        manager.mark_item_processing(batch_run_id, candidate_id)

        replay_id = str(uuid.uuid4())
        audit_record = {
            "replay_id": replay_id,
            "replay_cutoff": datetime(2024, 1, 15).isoformat(),
            "input_sources_json": "{}",
            "policy_version_json": json.dumps(
                {"config_digest": "digest_aaa"}
            ),
            "replay_status": "exact",
            "gate_trace_json": json.dumps({"entries": []}),
            "decision_delta_classification": "same_allow",
            "decision_delta_json": json.dumps({"classification": "same_allow"}),
        }
        manager.complete_item_atomic(
            batch_run_id=batch_run_id,
            candidate_id=candidate_id,
            replay_id=replay_id,
            audit_record=audit_record,
        )

        # Check idempotent replay
        cached = manager.check_idempotent_replay(candidate_id, policy_version)
        assert cached is not None
        assert cached["replay_id"] == replay_id
        assert cached["replay_status"] == "exact"
        assert cached["decision_delta_classification"] == "same_allow"

    def test_no_cache_hit_for_different_policy(
        self, session, policy_version, alt_policy_version
    ):
        manager = BatchManager(session)
        candidates = make_candidates(1)
        batch_run_id = manager.create_batch_run(
            mode="batch",
            policy_version=policy_version,
            candidates=candidates,
        )
        candidate_id = candidates[0].candidate_id
        manager.mark_item_processing(batch_run_id, candidate_id)

        replay_id = str(uuid.uuid4())
        manager.complete_item_atomic(
            batch_run_id=batch_run_id,
            candidate_id=candidate_id,
            replay_id=replay_id,
            audit_record={
                "replay_id": replay_id,
                "replay_cutoff": datetime(2024, 1, 15).isoformat(),
                "input_sources_json": "{}",
                "policy_version_json": json.dumps(
                    {"config_digest": "digest_aaa"}
                ),
                "replay_status": "exact",
            },
        )

        # Different policy → no cache hit
        cached = manager.check_idempotent_replay(
            candidate_id, alt_policy_version
        )
        assert cached is None

    def test_no_cache_hit_for_unknown_candidate(
        self, session, policy_version
    ):
        manager = BatchManager(session)
        cached = manager.check_idempotent_replay(
            "nonexistent-id", policy_version
        )
        assert cached is None


# ---------------------------------------------------------------------------
# Tests: Completion statistics
# ---------------------------------------------------------------------------


class TestCompletionStatistics:
    """Tests for finalize_batch and statistics reporting."""

    def test_finalize_reports_processed_and_failed_counts(
        self, session, policy_version
    ):
        manager = BatchManager(session)
        candidates = make_candidates(4)
        batch_run_id = manager.create_batch_run(
            mode="batch",
            policy_version=policy_version,
            candidates=candidates,
        )

        # Complete 2, fail 1, leave 1 pending
        for c in candidates[:2]:
            manager.mark_item_processing(batch_run_id, c.candidate_id)
            replay_id = str(uuid.uuid4())
            manager.complete_item_atomic(
                batch_run_id=batch_run_id,
                candidate_id=c.candidate_id,
                replay_id=replay_id,
                audit_record={
                    "replay_id": replay_id,
                    "replay_cutoff": datetime(2024, 1, 15).isoformat(),
                    "input_sources_json": "{}",
                    "policy_version_json": json.dumps(
                        {"config_digest": "digest_aaa"}
                    ),
                    "replay_status": "exact",
                    "decision_delta_classification": "same_allow",
                },
            )

        fail_candidate = candidates[2]
        manager.mark_item_processing(batch_run_id, fail_candidate.candidate_id)
        manager.fail_item(
            batch_run_id=batch_run_id,
            candidate_id=fail_candidate.candidate_id,
            reason_code="missing_input",
            details="Signal not found",
        )

        summary = manager.finalize_batch(batch_run_id)

        assert isinstance(summary, BatchRunSummary)
        assert summary.candidates_total == 4
        assert summary.candidates_processed == 2
        assert summary.candidates_failed == 1
        assert summary.status == "completed"
        assert summary.duration_seconds is not None
        assert summary.duration_seconds >= 0

    def test_finalize_reports_coverage_breakdown(
        self, session, policy_version
    ):
        manager = BatchManager(session)
        candidates = make_candidates(3)
        batch_run_id = manager.create_batch_run(
            mode="batch",
            policy_version=policy_version,
            candidates=candidates,
        )

        statuses = ["exact", "partial", "unscorable"]
        for i, c in enumerate(candidates):
            manager.mark_item_processing(batch_run_id, c.candidate_id)
            replay_id = str(uuid.uuid4())
            manager.complete_item_atomic(
                batch_run_id=batch_run_id,
                candidate_id=c.candidate_id,
                replay_id=replay_id,
                audit_record={
                    "replay_id": replay_id,
                    "replay_cutoff": datetime(2024, 1, 15).isoformat(),
                    "input_sources_json": "{}",
                    "policy_version_json": json.dumps(
                        {"config_digest": "digest_aaa"}
                    ),
                    "replay_status": statuses[i],
                },
            )

        summary = manager.finalize_batch(batch_run_id)
        assert summary.exact_count == 1
        assert summary.partial_count == 1
        assert summary.unscorable_count == 1

    def test_finalize_reports_delta_classification_counts(
        self, session, policy_version
    ):
        manager = BatchManager(session)
        candidates = make_candidates(3)
        batch_run_id = manager.create_batch_run(
            mode="batch",
            policy_version=policy_version,
            candidates=candidates,
        )

        deltas = [
            "same_allow",
            "replay_allows_original_reject",
            "same_allow",
        ]
        for i, c in enumerate(candidates):
            manager.mark_item_processing(batch_run_id, c.candidate_id)
            replay_id = str(uuid.uuid4())
            manager.complete_item_atomic(
                batch_run_id=batch_run_id,
                candidate_id=c.candidate_id,
                replay_id=replay_id,
                audit_record={
                    "replay_id": replay_id,
                    "replay_cutoff": datetime(2024, 1, 15).isoformat(),
                    "input_sources_json": "{}",
                    "policy_version_json": json.dumps(
                        {"config_digest": "digest_aaa"}
                    ),
                    "replay_status": "exact",
                    "decision_delta_classification": deltas[i],
                },
            )

        summary = manager.finalize_batch(batch_run_id)
        assert summary.delta_counts == {
            "same_allow": 2,
            "replay_allows_original_reject": 1,
        }

    def test_finalize_reports_failure_reason_codes(
        self, session, policy_version
    ):
        manager = BatchManager(session)
        candidates = make_candidates(3)
        batch_run_id = manager.create_batch_run(
            mode="batch",
            policy_version=policy_version,
            candidates=candidates,
        )

        # Fail all with different reasons
        reasons = ["missing_input", "gate_execution_error", "missing_input"]
        for i, c in enumerate(candidates):
            manager.mark_item_processing(batch_run_id, c.candidate_id)
            manager.fail_item(
                batch_run_id=batch_run_id,
                candidate_id=c.candidate_id,
                reason_code=reasons[i],
            )

        summary = manager.finalize_batch(batch_run_id)
        assert summary.failure_reasons == {
            "missing_input": 2,
            "gate_execution_error": 1,
        }

    def test_finalize_batch_not_found_raises(self, session):
        manager = BatchManager(session)
        with pytest.raises(ValueError, match="not found"):
            manager.finalize_batch("nonexistent-batch-id")
