"""
Batch state management for the Decision Replay Agent.

Manages the lifecycle of replay_batch_runs and replay_batch_items:
- Creating batch_items rows (status='pending') for each candidate at batch start
- Atomic persistence: audit record INSERT + batch_item status update in one transaction
- Failure handling: persist failure + mark batch_item 'failed', continue to next candidate
- Batch resumption: resume from first 'pending' item in processing_order
- Idempotent replay: same candidate + same policy_version → return cached result
- Completion statistics: processed, failed, coverage breakdown, delta counts

Requirements: 10.5, 10.6, 10.7, 10.8, 10.9
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import text

from core.replay.policy_version import PolicyVersion

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class BatchRunSummary:
    """Summary statistics for a completed or resumed batch run.

    Fields:
        batch_run_id: Unique identifier for this batch run
        started_at: When the batch run started
        ended_at: When the batch run completed (None if still running)
        mode: "batch" or "adhoc"
        policy_version: Policy version used for this run
        candidates_total: Total number of candidates in the batch
        candidates_processed: Number successfully processed
        candidates_failed: Number that failed processing
        exact_count: Candidates classified as 'exact'
        partial_count: Candidates classified as 'partial'
        unscorable_count: Candidates classified as 'unscorable'
        delta_counts: Dict mapping delta classification → count
        duration_seconds: Total run duration
        status: "running", "completed", "suspended", "failed"
        watermark_start: Earliest candidate timestamp in this batch
        watermark_end: Latest candidate timestamp in this batch
        failure_reasons: Dict mapping reason_code → count
    """

    batch_run_id: str
    started_at: datetime
    ended_at: datetime | None = None
    mode: str = "batch"
    policy_version: PolicyVersion | None = None
    candidates_total: int = 0
    candidates_processed: int = 0
    candidates_failed: int = 0
    exact_count: int = 0
    partial_count: int = 0
    unscorable_count: int = 0
    delta_counts: dict[str, int] = field(default_factory=dict)
    duration_seconds: float | None = None
    status: str = "running"
    watermark_start: datetime | None = None
    watermark_end: datetime | None = None
    failure_reasons: dict[str, int] = field(default_factory=dict)


@dataclass
class BatchItemResult:
    """Result of processing a single batch item.

    Fields:
        candidate_id: The candidate that was processed
        replay_id: ID of the replay_audit_record (None if failed or cached)
        status: "completed" or "failed"
        failure_reason_code: Reason code if failed (None if successful)
        failure_details: Human-readable failure details (None if successful)
        classification: "exact", "partial", or "unscorable" (None if failed)
        delta_classification: Decision delta category (None if failed/unscorable)
        cached: Whether this result was returned from cache (idempotent replay)
    """

    candidate_id: str
    replay_id: str | None = None
    status: str = "completed"
    failure_reason_code: str | None = None
    failure_details: str | None = None
    classification: str | None = None
    delta_classification: str | None = None
    cached: bool = False


# ---------------------------------------------------------------------------
# Failure reason codes (Requirement 13.5)
# ---------------------------------------------------------------------------

FAILURE_REASON_CODES = frozenset({
    "missing_input",
    "policy_not_found",
    "snapshot_corrupt",
    "gate_execution_error",
    "timeout",
    "reconstruction_error",
    "unknown_error",
})


# ---------------------------------------------------------------------------
# Batch Manager
# ---------------------------------------------------------------------------


class BatchManager:
    """Manages batch state for the Decision Replay Agent.

    Handles batch lifecycle: creation, item processing, failure handling,
    resumption, idempotent replay, and completion statistics.
    """

    def __init__(self, session):
        """Initialize with a SQLAlchemy session bound to the replay DB."""
        self._session = session

    # -----------------------------------------------------------------------
    # Batch creation / resumption
    # -----------------------------------------------------------------------

    def create_batch_run(
        self,
        *,
        mode: str = "batch",
        policy_version: PolicyVersion,
        filters: dict | None = None,
        candidates: list[Any],
        watermark_start: datetime | None = None,
        watermark_end: datetime | None = None,
    ) -> str:
        """Create a new batch_run and pending batch_items for each candidate.

        Args:
            mode: "batch" or "adhoc"
            policy_version: The PolicyVersion for this replay run
            filters: Optional filter criteria used to select candidates
            candidates: List of ReplayCandidate objects to process
            watermark_start: Earliest candidate timestamp boundary
            watermark_end: Latest candidate timestamp boundary

        Returns:
            batch_run_id: The new batch run's unique identifier
        """
        batch_run_id = str(uuid.uuid4())
        now = datetime.utcnow()

        policy_json = json.dumps({
            "name": policy_version.name,
            "gate_revision": policy_version.gate_revision,
            "config_digest": policy_version.config_digest,
            "feature_flags": policy_version.feature_flags,
            "benchmark_version": policy_version.benchmark_version,
            "config_source_timestamp": (
                policy_version.config_source_timestamp.isoformat()
                if policy_version.config_source_timestamp
                else None
            ),
            "gate_ordering_version": policy_version.gate_ordering_version,
            "adapter_version": policy_version.adapter_version,
        })

        filters_json = json.dumps(filters) if filters else None

        # Insert the batch_run record
        self._session.execute(
            text("""
                INSERT INTO replay_batch_runs (
                    batch_run_id, started_at, mode, policy_version_json,
                    filters_json, candidates_total, candidates_processed,
                    candidates_failed, exact_count, partial_count,
                    unscorable_count, delta_counts_json, status,
                    watermark_start, watermark_end
                ) VALUES (
                    :batch_run_id, :started_at, :mode, :policy_version_json,
                    :filters_json, :candidates_total, 0,
                    0, 0, 0,
                    0, :delta_counts_json, 'running',
                    :watermark_start, :watermark_end
                )
            """),
            {
                "batch_run_id": batch_run_id,
                "started_at": now.isoformat(),
                "mode": mode,
                "policy_version_json": policy_json,
                "filters_json": filters_json,
                "candidates_total": len(candidates),
                "delta_counts_json": json.dumps({}),
                "watermark_start": (
                    watermark_start.isoformat() if watermark_start else None
                ),
                "watermark_end": (
                    watermark_end.isoformat() if watermark_end else None
                ),
            },
        )

        # Insert batch_items (status='pending') for each candidate
        for order, candidate in enumerate(candidates, start=1):
            self._session.execute(
                text("""
                    INSERT INTO replay_batch_items (
                        batch_run_id, candidate_id, processing_order,
                        status, created_at
                    ) VALUES (
                        :batch_run_id, :candidate_id, :processing_order,
                        'pending', :created_at
                    )
                """),
                {
                    "batch_run_id": batch_run_id,
                    "candidate_id": candidate.candidate_id,
                    "processing_order": order,
                    "created_at": now.isoformat(),
                },
            )

        self._session.commit()
        log.info(
            f"Created batch run {batch_run_id} with {len(candidates)} candidates"
        )
        return batch_run_id

    def find_resumable_batch(
        self,
        *,
        mode: str = "batch",
        policy_version: PolicyVersion,
    ) -> str | None:
        """Find an existing batch_run with pending items that can be resumed.

        A batch is resumable if:
        - status is 'running' or 'suspended'
        - mode matches
        - policy_version config_digest matches
        - There are still pending items

        Returns:
            batch_run_id if a resumable batch exists, else None
        """
        result = self._session.execute(
            text("""
                SELECT br.batch_run_id
                FROM replay_batch_runs br
                WHERE br.status IN ('running', 'suspended')
                  AND br.mode = :mode
                  AND EXISTS (
                      SELECT 1 FROM replay_batch_items bi
                      WHERE bi.batch_run_id = br.batch_run_id
                        AND bi.status = 'pending'
                  )
                ORDER BY br.started_at DESC
                LIMIT 1
            """),
            {"mode": mode},
        )
        row = result.fetchone()
        if row is None:
            return None

        batch_run_id = row[0]

        # Verify policy version matches
        pv_result = self._session.execute(
            text("""
                SELECT policy_version_json FROM replay_batch_runs
                WHERE batch_run_id = :batch_run_id
            """),
            {"batch_run_id": batch_run_id},
        )
        pv_row = pv_result.fetchone()
        if pv_row:
            stored_policy = json.loads(pv_row[0])
            if stored_policy.get("config_digest") == policy_version.config_digest:
                log.info(
                    f"Found resumable batch {batch_run_id} with pending items"
                )
                return batch_run_id

        return None

    def get_pending_items(self, batch_run_id: str) -> list[dict]:
        """Get pending batch items in processing_order.

        Returns list of dicts with keys: candidate_id, processing_order.
        Items are ordered by processing_order ascending (resume from first pending).
        """
        result = self._session.execute(
            text("""
                SELECT candidate_id, processing_order
                FROM replay_batch_items
                WHERE batch_run_id = :batch_run_id
                  AND status = 'pending'
                ORDER BY processing_order ASC
            """),
            {"batch_run_id": batch_run_id},
        )
        return [
            {"candidate_id": row[0], "processing_order": row[1]}
            for row in result.fetchall()
        ]

    # -----------------------------------------------------------------------
    # Item processing — atomic persistence
    # -----------------------------------------------------------------------

    def mark_item_processing(
        self, batch_run_id: str, candidate_id: str
    ) -> None:
        """Mark a batch_item as 'processing' with started_at timestamp."""
        now = datetime.utcnow()
        self._session.execute(
            text("""
                UPDATE replay_batch_items
                SET status = 'processing', started_at = :started_at
                WHERE batch_run_id = :batch_run_id
                  AND candidate_id = :candidate_id
                  AND status = 'pending'
            """),
            {
                "batch_run_id": batch_run_id,
                "candidate_id": candidate_id,
                "started_at": now.isoformat(),
            },
        )
        self._session.commit()

    def complete_item_atomic(
        self,
        *,
        batch_run_id: str,
        candidate_id: str,
        replay_id: str,
        audit_record: dict,
    ) -> None:
        """Atomically persist audit record + mark batch_item completed.

        This is a SINGLE TRANSACTION: if either the audit record INSERT or
        the batch_item status update fails, both are rolled back — leaving
        the item in 'processing' status for retry.

        Args:
            batch_run_id: The batch run this item belongs to
            candidate_id: The candidate being completed
            replay_id: Unique replay identifier for the audit record
            audit_record: Dict with all fields for replay_audit_records INSERT
        """
        now = datetime.utcnow()

        try:
            # INSERT audit record
            self._session.execute(
                text("""
                    INSERT INTO replay_audit_records (
                        replay_id, batch_run_id, candidate_id,
                        source_candidate_ids_json, snapshot_id,
                        replay_cutoff, input_sources_json,
                        policy_version_json, replay_status,
                        gate_trace_json, decision_delta_classification,
                        decision_delta_json, counterfactual_outcome_json,
                        divergence_cause, divergence_evidence_json,
                        code_revision, era, diagnostic_mode,
                        created_at
                    ) VALUES (
                        :replay_id, :batch_run_id, :candidate_id,
                        :source_candidate_ids_json, :snapshot_id,
                        :replay_cutoff, :input_sources_json,
                        :policy_version_json, :replay_status,
                        :gate_trace_json, :decision_delta_classification,
                        :decision_delta_json, :counterfactual_outcome_json,
                        :divergence_cause, :divergence_evidence_json,
                        :code_revision, :era, :diagnostic_mode,
                        :created_at
                    )
                """),
                {
                    "replay_id": audit_record["replay_id"],
                    "batch_run_id": batch_run_id,
                    "candidate_id": candidate_id,
                    "source_candidate_ids_json": audit_record.get(
                        "source_candidate_ids_json", "[]"
                    ),
                    "snapshot_id": audit_record.get("snapshot_id"),
                    "replay_cutoff": audit_record["replay_cutoff"],
                    "input_sources_json": audit_record.get(
                        "input_sources_json", "{}"
                    ),
                    "policy_version_json": audit_record["policy_version_json"],
                    "replay_status": audit_record["replay_status"],
                    "gate_trace_json": audit_record.get("gate_trace_json"),
                    "decision_delta_classification": audit_record.get(
                        "decision_delta_classification"
                    ),
                    "decision_delta_json": audit_record.get(
                        "decision_delta_json"
                    ),
                    "counterfactual_outcome_json": audit_record.get(
                        "counterfactual_outcome_json"
                    ),
                    "divergence_cause": audit_record.get("divergence_cause"),
                    "divergence_evidence_json": audit_record.get(
                        "divergence_evidence_json"
                    ),
                    "code_revision": audit_record.get("code_revision"),
                    "era": audit_record.get("era", "post-snapshot"),
                    "diagnostic_mode": audit_record.get("diagnostic_mode", 0),
                    "created_at": now.isoformat(),
                },
            )

            # UPDATE batch_item to 'completed'
            self._session.execute(
                text("""
                    UPDATE replay_batch_items
                    SET status = 'completed',
                        replay_id = :replay_id,
                        ended_at = :ended_at
                    WHERE batch_run_id = :batch_run_id
                      AND candidate_id = :candidate_id
                      AND status = 'processing'
                """),
                {
                    "replay_id": replay_id,
                    "ended_at": now.isoformat(),
                    "batch_run_id": batch_run_id,
                    "candidate_id": candidate_id,
                },
            )

            # Commit both in one transaction
            self._session.commit()
            log.debug(
                f"Atomically completed item {candidate_id} "
                f"with replay {replay_id}"
            )

        except Exception:
            self._session.rollback()
            raise

    def fail_item(
        self,
        *,
        batch_run_id: str,
        candidate_id: str,
        reason_code: str,
        details: str | None = None,
        audit_record: dict | None = None,
    ) -> None:
        """Mark a batch_item as 'failed' and optionally persist failure audit record.

        Failure handling: persist failure audit record + mark batch_item 'failed',
        then CONTINUE to next candidate (caller is responsible for not stopping).

        Args:
            batch_run_id: The batch run this item belongs to
            candidate_id: The candidate that failed
            reason_code: Structured failure reason code
            details: Optional human-readable failure details
            audit_record: Optional dict for failure audit record persistence
        """
        now = datetime.utcnow()

        # Normalize reason_code
        if reason_code not in FAILURE_REASON_CODES:
            reason_code = "unknown_error"

        try:
            # Optionally persist a failure audit record
            if audit_record:
                self._session.execute(
                    text("""
                        INSERT INTO replay_audit_records (
                            replay_id, batch_run_id, candidate_id,
                            source_candidate_ids_json, snapshot_id,
                            replay_cutoff, input_sources_json,
                            policy_version_json, replay_status,
                            failure_reason_code, failure_details,
                            era, diagnostic_mode, created_at
                        ) VALUES (
                            :replay_id, :batch_run_id, :candidate_id,
                            :source_candidate_ids_json, :snapshot_id,
                            :replay_cutoff, :input_sources_json,
                            :policy_version_json, :replay_status,
                            :failure_reason_code, :failure_details,
                            :era, :diagnostic_mode, :created_at
                        )
                    """),
                    {
                        "replay_id": audit_record.get(
                            "replay_id", str(uuid.uuid4())
                        ),
                        "batch_run_id": batch_run_id,
                        "candidate_id": candidate_id,
                        "source_candidate_ids_json": audit_record.get(
                            "source_candidate_ids_json", "[]"
                        ),
                        "snapshot_id": audit_record.get("snapshot_id"),
                        "replay_cutoff": audit_record.get(
                            "replay_cutoff", now.isoformat()
                        ),
                        "input_sources_json": audit_record.get(
                            "input_sources_json", "{}"
                        ),
                        "policy_version_json": audit_record.get(
                            "policy_version_json", "{}"
                        ),
                        "replay_status": "failed",
                        "failure_reason_code": reason_code,
                        "failure_details": details,
                        "era": audit_record.get("era", "post-snapshot"),
                        "diagnostic_mode": audit_record.get(
                            "diagnostic_mode", 0
                        ),
                        "created_at": now.isoformat(),
                    },
                )

            # UPDATE batch_item to 'failed'
            self._session.execute(
                text("""
                    UPDATE replay_batch_items
                    SET status = 'failed',
                        failure_reason_code = :reason_code,
                        failure_details = :details,
                        ended_at = :ended_at
                    WHERE batch_run_id = :batch_run_id
                      AND candidate_id = :candidate_id
                      AND status IN ('pending', 'processing')
                """),
                {
                    "reason_code": reason_code,
                    "details": details,
                    "ended_at": now.isoformat(),
                    "batch_run_id": batch_run_id,
                    "candidate_id": candidate_id,
                },
            )

            self._session.commit()
            log.warning(
                f"Batch item {candidate_id} failed: "
                f"{reason_code} — {details or 'no details'}"
            )

        except Exception as e:
            self._session.rollback()
            log.error(
                f"Failed to record failure for {candidate_id}: {e}"
            )
            raise

    # -----------------------------------------------------------------------
    # Idempotent replay — cached result lookup
    # -----------------------------------------------------------------------

    def check_idempotent_replay(
        self,
        candidate_id: str,
        policy_version: PolicyVersion,
    ) -> dict | None:
        """Check if same candidate + same policy_version has been replayed.

        If a completed replay_audit_record exists for this candidate with
        a matching policy config_digest, return the cached result.

        Returns:
            Dict with cached audit record fields if found, else None
        """
        result = self._session.execute(
            text("""
                SELECT replay_id, replay_status, gate_trace_json,
                       decision_delta_classification, decision_delta_json,
                       counterfactual_outcome_json, divergence_cause,
                       divergence_evidence_json
                FROM replay_audit_records
                WHERE candidate_id = :candidate_id
                  AND replay_status != 'failed'
                ORDER BY created_at DESC
                LIMIT 1
            """),
            {"candidate_id": candidate_id},
        )
        row = result.fetchone()
        if row is None:
            return None

        # Verify the policy_version matches
        # We need to check the policy_version_json in the record
        pv_result = self._session.execute(
            text("""
                SELECT policy_version_json
                FROM replay_audit_records
                WHERE replay_id = :replay_id
            """),
            {"replay_id": row[0]},
        )
        pv_row = pv_result.fetchone()
        if pv_row is None:
            return None

        stored_policy = json.loads(pv_row[0])
        if stored_policy.get("config_digest") != policy_version.config_digest:
            return None

        log.debug(
            f"Idempotent cache hit for candidate {candidate_id} "
            f"(replay_id={row[0]})"
        )
        return {
            "replay_id": row[0],
            "replay_status": row[1],
            "gate_trace_json": row[2],
            "decision_delta_classification": row[3],
            "decision_delta_json": row[4],
            "counterfactual_outcome_json": row[5],
            "divergence_cause": row[6],
            "divergence_evidence_json": row[7],
        }

    # -----------------------------------------------------------------------
    # Batch completion and statistics
    # -----------------------------------------------------------------------

    def finalize_batch(self, batch_run_id: str) -> BatchRunSummary:
        """Finalize a batch run and compute completion statistics.

        Computes:
        - Total processed, failed counts
        - Coverage breakdown (exact/partial/unscorable)
        - Delta classification counts
        - Failure reason code counts
        - Duration

        Updates the batch_run record with final stats and status='completed'.

        Returns:
            BatchRunSummary with all computed statistics
        """
        now = datetime.utcnow()

        # Get batch run metadata
        run_result = self._session.execute(
            text("""
                SELECT started_at, mode, policy_version_json, filters_json,
                       candidates_total, watermark_start, watermark_end
                FROM replay_batch_runs
                WHERE batch_run_id = :batch_run_id
            """),
            {"batch_run_id": batch_run_id},
        )
        run_row = run_result.fetchone()
        if run_row is None:
            raise ValueError(f"Batch run {batch_run_id} not found")

        started_at_str = run_row[0]
        mode = run_row[1]
        candidates_total = run_row[4]

        # Parse started_at
        if isinstance(started_at_str, str):
            started_at = datetime.fromisoformat(started_at_str)
        else:
            started_at = started_at_str

        duration_seconds = (now - started_at).total_seconds()

        # Count completed and failed items
        status_result = self._session.execute(
            text("""
                SELECT status, COUNT(*) as cnt
                FROM replay_batch_items
                WHERE batch_run_id = :batch_run_id
                GROUP BY status
            """),
            {"batch_run_id": batch_run_id},
        )
        status_counts = {row[0]: row[1] for row in status_result.fetchall()}
        candidates_processed = status_counts.get("completed", 0)
        candidates_failed = status_counts.get("failed", 0)

        # Coverage breakdown from audit records
        coverage_result = self._session.execute(
            text("""
                SELECT replay_status, COUNT(*) as cnt
                FROM replay_audit_records
                WHERE batch_run_id = :batch_run_id
                  AND replay_status != 'failed'
                GROUP BY replay_status
            """),
            {"batch_run_id": batch_run_id},
        )
        coverage_counts = {
            row[0]: row[1] for row in coverage_result.fetchall()
        }
        exact_count = coverage_counts.get("exact", 0)
        partial_count = coverage_counts.get("partial", 0)
        unscorable_count = coverage_counts.get("unscorable", 0)

        # Delta classification counts
        delta_result = self._session.execute(
            text("""
                SELECT decision_delta_classification, COUNT(*) as cnt
                FROM replay_audit_records
                WHERE batch_run_id = :batch_run_id
                  AND decision_delta_classification IS NOT NULL
                GROUP BY decision_delta_classification
            """),
            {"batch_run_id": batch_run_id},
        )
        delta_counts = {row[0]: row[1] for row in delta_result.fetchall()}

        # Failure reason code counts
        failure_result = self._session.execute(
            text("""
                SELECT failure_reason_code, COUNT(*) as cnt
                FROM replay_batch_items
                WHERE batch_run_id = :batch_run_id
                  AND status = 'failed'
                  AND failure_reason_code IS NOT NULL
                GROUP BY failure_reason_code
            """),
            {"batch_run_id": batch_run_id},
        )
        failure_reasons = {row[0]: row[1] for row in failure_result.fetchall()}

        # Parse watermarks
        watermark_start = None
        watermark_end = None
        if run_row[5]:
            watermark_start = (
                datetime.fromisoformat(run_row[5])
                if isinstance(run_row[5], str)
                else run_row[5]
            )
        if run_row[6]:
            watermark_end = (
                datetime.fromisoformat(run_row[6])
                if isinstance(run_row[6], str)
                else run_row[6]
            )

        # Parse policy version
        policy_version = None
        if run_row[2]:
            pv_data = json.loads(run_row[2])
            policy_version = PolicyVersion(
                name=pv_data.get("name", "unknown"),
                gate_revision=pv_data.get("gate_revision", "unknown"),
                config_digest=pv_data.get("config_digest", ""),
                feature_flags=pv_data.get("feature_flags", {}),
                benchmark_version=pv_data.get("benchmark_version"),
                config_source_timestamp=(
                    datetime.fromisoformat(pv_data["config_source_timestamp"])
                    if pv_data.get("config_source_timestamp")
                    else None
                ),
                gate_ordering_version=pv_data.get(
                    "gate_ordering_version", "v1.0"
                ),
                adapter_version=pv_data.get("adapter_version", "1.0.0"),
            )

        # Update the batch_run record with final stats
        self._session.execute(
            text("""
                UPDATE replay_batch_runs
                SET ended_at = :ended_at,
                    candidates_processed = :candidates_processed,
                    candidates_failed = :candidates_failed,
                    exact_count = :exact_count,
                    partial_count = :partial_count,
                    unscorable_count = :unscorable_count,
                    delta_counts_json = :delta_counts_json,
                    duration_seconds = :duration_seconds,
                    status = 'completed'
                WHERE batch_run_id = :batch_run_id
            """),
            {
                "ended_at": now.isoformat(),
                "candidates_processed": candidates_processed,
                "candidates_failed": candidates_failed,
                "exact_count": exact_count,
                "partial_count": partial_count,
                "unscorable_count": unscorable_count,
                "delta_counts_json": json.dumps(delta_counts),
                "duration_seconds": duration_seconds,
                "batch_run_id": batch_run_id,
            },
        )
        self._session.commit()

        summary = BatchRunSummary(
            batch_run_id=batch_run_id,
            started_at=started_at,
            ended_at=now,
            mode=mode,
            policy_version=policy_version,
            candidates_total=candidates_total,
            candidates_processed=candidates_processed,
            candidates_failed=candidates_failed,
            exact_count=exact_count,
            partial_count=partial_count,
            unscorable_count=unscorable_count,
            delta_counts=delta_counts,
            duration_seconds=duration_seconds,
            status="completed",
            watermark_start=watermark_start,
            watermark_end=watermark_end,
            failure_reasons=failure_reasons,
        )

        log.info(
            f"Batch {batch_run_id} finalized: "
            f"{candidates_processed} processed, "
            f"{candidates_failed} failed, "
            f"exact={exact_count}, partial={partial_count}, "
            f"unscorable={unscorable_count}"
        )
        return summary

    def suspend_batch(self, batch_run_id: str) -> None:
        """Suspend a batch run (e.g., for market-hour checkpoint).

        Sets status to 'suspended' — can be resumed later via
        find_resumable_batch().
        """
        self._session.execute(
            text("""
                UPDATE replay_batch_runs
                SET status = 'suspended'
                WHERE batch_run_id = :batch_run_id
                  AND status = 'running'
            """),
            {"batch_run_id": batch_run_id},
        )
        self._session.commit()
        log.info(f"Batch {batch_run_id} suspended")

    def get_batch_summary(self, batch_run_id: str) -> BatchRunSummary | None:
        """Get current statistics for a batch run without finalizing.

        Useful for progress reporting during long-running batches.
        """
        run_result = self._session.execute(
            text("""
                SELECT started_at, ended_at, mode, policy_version_json,
                       candidates_total, candidates_processed,
                       candidates_failed, exact_count, partial_count,
                       unscorable_count, delta_counts_json,
                       duration_seconds, status,
                       watermark_start, watermark_end
                FROM replay_batch_runs
                WHERE batch_run_id = :batch_run_id
            """),
            {"batch_run_id": batch_run_id},
        )
        row = run_result.fetchone()
        if row is None:
            return None

        started_at = (
            datetime.fromisoformat(row[0])
            if isinstance(row[0], str)
            else row[0]
        )
        ended_at = (
            datetime.fromisoformat(row[1])
            if row[1] and isinstance(row[1], str)
            else row[1]
        )
        delta_counts = json.loads(row[10]) if row[10] else {}

        watermark_start = None
        watermark_end = None
        if row[13]:
            watermark_start = (
                datetime.fromisoformat(row[13])
                if isinstance(row[13], str)
                else row[13]
            )
        if row[14]:
            watermark_end = (
                datetime.fromisoformat(row[14])
                if isinstance(row[14], str)
                else row[14]
            )

        return BatchRunSummary(
            batch_run_id=batch_run_id,
            started_at=started_at,
            ended_at=ended_at,
            mode=row[2],
            candidates_total=row[4] or 0,
            candidates_processed=row[5] or 0,
            candidates_failed=row[6] or 0,
            exact_count=row[7] or 0,
            partial_count=row[8] or 0,
            unscorable_count=row[9] or 0,
            delta_counts=delta_counts,
            duration_seconds=row[11],
            status=row[12],
            watermark_start=watermark_start,
            watermark_end=watermark_end,
        )
