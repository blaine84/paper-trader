"""
Tests for core/replay/reporter.py — CEO and Reviewer integration.

Validates Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6
"""

import json
import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine, text, event

from db.replay_schema import init_replay_db
from core.replay.reporter import (
    build_daily_ceo_input,
    build_weekly_ceo_input,
    get_replay_for_reviewer,
    get_replay_by_candidate_id,
    format_findings_with_examples,
    CEOFinding,
    ReplayExample,
    DailyCEOInput,
    WeeklyCEOInput,
    ReviewerReplayResult,
    DEFAULT_MATERIALITY_THRESHOLD_PCT,
    MAX_EXAMPLES_PER_FINDING,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """In-memory SQLite engine with production tables pre-created."""
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        # Minimal production tables needed by init_replay_db
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


@pytest.fixture
def session(initialized_engine):
    """SQLAlchemy connection acting as a session for queries."""
    from sqlalchemy.orm import Session
    with Session(initialized_engine) as sess:
        yield sess


def _insert_audit_record(session, **overrides):
    """Helper to insert a replay_audit_record with sensible defaults."""
    defaults = {
        "replay_id": str(uuid.uuid4()),
        "batch_run_id": str(uuid.uuid4()),
        "candidate_id": str(uuid.uuid4()),
        "source_candidate_ids_json": json.dumps([{"source_table": "blocked_trade_candidates", "source_id": 1}]),
        "snapshot_id": str(uuid.uuid4()),
        "replay_cutoff": datetime(2024, 6, 15, 10, 0, 0).isoformat(),
        "input_sources_json": json.dumps({}),
        "policy_version_json": json.dumps({"name": "current", "gate_revision": "abc123"}),
        "replay_status": "exact",
        "gate_trace_json": json.dumps([{"gate_name": "risk_geometry_gate", "decision": "allow", "missing_fields": []}]),
        "decision_delta_classification": "same_allow",
        "decision_delta_json": json.dumps({"classification": "same_allow"}),
        "counterfactual_outcome_json": None,
        "divergence_cause": None,
        "divergence_evidence_json": None,
        "code_revision": "abc123",
        "era": "post-snapshot",
        "diagnostic_mode": 0,
        "failure_reason_code": None,
        "failure_details": None,
        "created_at": datetime(2024, 6, 15, 12, 0, 0).isoformat(),
    }
    defaults.update(overrides)

    session.execute(
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
                failure_reason_code, failure_details, created_at
            ) VALUES (
                :replay_id, :batch_run_id, :candidate_id,
                :source_candidate_ids_json, :snapshot_id,
                :replay_cutoff, :input_sources_json,
                :policy_version_json, :replay_status,
                :gate_trace_json, :decision_delta_classification,
                :decision_delta_json, :counterfactual_outcome_json,
                :divergence_cause, :divergence_evidence_json,
                :code_revision, :era, :diagnostic_mode,
                :failure_reason_code, :failure_details, :created_at
            )
        """),
        defaults,
    )
    session.commit()
    return defaults


# ---------------------------------------------------------------------------
# Test: Daily CEO Input (Requirement 11.1)
# ---------------------------------------------------------------------------


class TestDailyCEOInput:
    """Test build_daily_ceo_input — Requirement 11.1."""

    def test_empty_day_returns_empty_findings(self, session):
        """No records for the day → empty findings."""
        result = build_daily_ceo_input(session, datetime(2024, 6, 15))
        assert result.findings == []
        assert result.total_candidates_evaluated == 0

    def test_defects_included_in_daily_input(self, session):
        """Defects (metadata_wiring_defect) are surfaced in daily CEO input."""
        _insert_audit_record(
            session,
            replay_status="exact",
            divergence_cause="metadata_wiring_defect",
            decision_delta_classification="replay_allows_original_reject",
            counterfactual_outcome_json=json.dumps({"return_60m": 2.5}),
            created_at=datetime(2024, 6, 15, 14, 0, 0).isoformat(),
        )

        result = build_daily_ceo_input(session, datetime(2024, 6, 15))

        assert result.total_candidates_evaluated == 1
        assert result.exact_coverage_count == 1
        # Should have at least the defect finding
        defect_findings = [f for f in result.findings if f.finding_type == "defect"]
        assert len(defect_findings) >= 1
        assert defect_findings[0].affected_candidate_count == 1
        assert defect_findings[0].replay_confidence == "exact"

    def test_code_defect_included(self, session):
        """code_defect divergence cause is also surfaced."""
        _insert_audit_record(
            session,
            replay_status="exact",
            divergence_cause="code_defect",
            decision_delta_classification="replay_rejects_original_allow",
            created_at=datetime(2024, 6, 15, 14, 0, 0).isoformat(),
        )

        result = build_daily_ceo_input(session, datetime(2024, 6, 15))
        defect_findings = [f for f in result.findings if f.finding_type == "defect"]
        assert len(defect_findings) == 1
        assert defect_findings[0].divergence_cause == "code_defect"

    def test_material_deltas_included(self, session):
        """Decision_Deltas exceeding materiality threshold are included."""
        _insert_audit_record(
            session,
            replay_status="exact",
            divergence_cause="configuration_change",
            decision_delta_classification="replay_allows_original_reject",
            counterfactual_outcome_json=json.dumps({"return_60m": 1.5}),
            created_at=datetime(2024, 6, 15, 14, 0, 0).isoformat(),
        )

        result = build_daily_ceo_input(
            session, datetime(2024, 6, 15), materiality_threshold_pct=1.0
        )

        delta_findings = [f for f in result.findings if f.finding_type == "delta"]
        assert len(delta_findings) == 1
        assert delta_findings[0].affected_candidate_count == 1

    def test_below_materiality_excluded(self, session):
        """Deltas below materiality threshold are NOT included as delta findings."""
        _insert_audit_record(
            session,
            replay_status="exact",
            divergence_cause="configuration_change",
            decision_delta_classification="replay_allows_original_reject",
            counterfactual_outcome_json=json.dumps({"return_60m": 0.1}),
            created_at=datetime(2024, 6, 15, 14, 0, 0).isoformat(),
        )

        result = build_daily_ceo_input(
            session, datetime(2024, 6, 15), materiality_threshold_pct=0.5
        )

        delta_findings = [f for f in result.findings if f.finding_type == "delta"]
        assert len(delta_findings) == 0

    def test_findings_labeled_with_confidence(self, session):
        """All findings are labeled with replay confidence (Req 11.1)."""
        _insert_audit_record(
            session,
            replay_status="partial",
            divergence_cause="metadata_wiring_defect",
            decision_delta_classification="replay_allows_original_reject",
            created_at=datetime(2024, 6, 15, 14, 0, 0).isoformat(),
        )

        result = build_daily_ceo_input(session, datetime(2024, 6, 15))
        for finding in result.findings:
            assert finding.replay_confidence in ("exact", "partial")

    def test_unscorable_records_excluded_from_findings(self, session):
        """Unscorable replays are NOT included in defect or delta findings."""
        _insert_audit_record(
            session,
            replay_status="unscorable",
            divergence_cause="metadata_wiring_defect",
            decision_delta_classification="unscorable",
            created_at=datetime(2024, 6, 15, 14, 0, 0).isoformat(),
        )

        result = build_daily_ceo_input(session, datetime(2024, 6, 15))
        # The record is counted in total but not in findings
        assert result.total_candidates_evaluated == 1
        assert result.unscorable_count == 1
        defect_findings = [f for f in result.findings if f.finding_type == "defect"]
        assert len(defect_findings) == 0

    def test_coverage_counts_accurate(self, session):
        """Coverage breakdown reflects actual record statuses."""
        _insert_audit_record(
            session,
            replay_status="exact",
            created_at=datetime(2024, 6, 15, 10, 0, 0).isoformat(),
        )
        _insert_audit_record(
            session,
            replay_status="partial",
            created_at=datetime(2024, 6, 15, 11, 0, 0).isoformat(),
        )
        _insert_audit_record(
            session,
            replay_status="unscorable",
            created_at=datetime(2024, 6, 15, 12, 0, 0).isoformat(),
        )

        result = build_daily_ceo_input(session, datetime(2024, 6, 15))
        assert result.total_candidates_evaluated == 3
        assert result.exact_coverage_count == 1
        assert result.partial_coverage_count == 1
        assert result.unscorable_count == 1


# ---------------------------------------------------------------------------
# Test: Repeated Omission Surfacing (Requirement 11.4)
# ---------------------------------------------------------------------------


class TestRepeatedOmissionSurfacing:
    """Test repeated-omission detection and surfacing — Requirement 11.4."""

    def test_three_omissions_surfaced_as_defect(self, session):
        """≥3 same metadata omission in 7-day window → engineering defect."""
        base_time = datetime(2024, 6, 15, 10, 0, 0)
        for i in range(3):
            _insert_audit_record(
                session,
                replay_status="exact",
                gate_trace_json=json.dumps([{
                    "gate_name": "risk_geometry_gate",
                    "decision": "reject",
                    "missing_fields": ["signal_strength"],
                }]),
                replay_cutoff=(base_time + timedelta(hours=i)).isoformat(),
                created_at=(base_time + timedelta(hours=i)).isoformat(),
            )

        result = build_daily_ceo_input(session, datetime(2024, 6, 15))
        omission_findings = [
            f for f in result.findings if f.finding_type == "repeated_omission"
        ]
        assert len(omission_findings) >= 1

        finding = omission_findings[0]
        assert finding.affected_candidate_count >= 3
        assert "signal_strength" in finding.title
        assert "risk_geometry_gate" in finding.title

    def test_two_omissions_not_surfaced(self, session):
        """Fewer than 3 omissions do NOT trigger engineering defect."""
        base_time = datetime(2024, 6, 15, 10, 0, 0)
        for i in range(2):
            _insert_audit_record(
                session,
                replay_status="exact",
                gate_trace_json=json.dumps([{
                    "gate_name": "risk_geometry_gate",
                    "decision": "reject",
                    "missing_fields": ["signal_strength"],
                }]),
                replay_cutoff=(base_time + timedelta(hours=i)).isoformat(),
                created_at=(base_time + timedelta(hours=i)).isoformat(),
            )

        result = build_daily_ceo_input(session, datetime(2024, 6, 15))
        omission_findings = [
            f for f in result.findings if f.finding_type == "repeated_omission"
        ]
        assert len(omission_findings) == 0

    def test_omission_includes_field_and_first_occurrence(self, session):
        """Engineering defect includes specific field and first occurrence date."""
        base_time = datetime(2024, 6, 12, 10, 0, 0)  # Within 7-day lookback
        for i in range(4):
            _insert_audit_record(
                session,
                replay_status="exact",
                gate_trace_json=json.dumps([{
                    "gate_name": "catalyst_specificity_gate",
                    "decision": "reject",
                    "missing_fields": ["catalyst_type"],
                }]),
                replay_cutoff=(base_time + timedelta(days=i)).isoformat(),
                created_at=(base_time + timedelta(days=i)).isoformat(),
            )

        result = build_daily_ceo_input(session, datetime(2024, 6, 15))
        omission_findings = [
            f for f in result.findings if f.finding_type == "repeated_omission"
        ]
        assert len(omission_findings) >= 1
        finding = omission_findings[0]
        assert "catalyst_type" in finding.description
        assert "catalyst_specificity_gate" in finding.description
        assert "First occurrence" in finding.description


# ---------------------------------------------------------------------------
# Test: Weekly CEO Input (Requirement 11.2)
# ---------------------------------------------------------------------------


class TestWeeklyCEOInput:
    """Test build_weekly_ceo_input — Requirement 11.2."""

    def test_empty_week_returns_empty_summaries(self, session):
        """No records for the week → empty summaries."""
        result = build_weekly_ceo_input(session, datetime(2024, 6, 10))
        assert result.gate_summaries == []
        assert result.total_candidates_evaluated == 0

    def test_includes_sample_size_and_coverage(self, session):
        """Weekly input states exact-replay coverage % and sample size."""
        base_time = datetime(2024, 6, 10, 10, 0, 0)
        # Insert enough records to potentially meet sample size
        for i in range(35):
            _insert_audit_record(
                session,
                replay_status="exact" if i < 30 else "partial",
                gate_trace_json=json.dumps([{
                    "gate_name": "risk_geometry_gate",
                    "decision": "allow",
                    "missing_fields": [],
                }]),
                decision_delta_classification="same_allow",
                created_at=(base_time + timedelta(hours=i)).isoformat(),
            )

        result = build_weekly_ceo_input(
            session, datetime(2024, 6, 10), min_sample_size=30
        )

        assert result.total_candidates_evaluated == 35
        assert result.exact_coverage_pct > 0
        assert result.min_sample_size == 30

    def test_below_sample_size_not_included(self, session):
        """Groups below minimum sample size are excluded from summaries."""
        base_time = datetime(2024, 6, 10, 10, 0, 0)
        # Insert too few records
        for i in range(5):
            _insert_audit_record(
                session,
                replay_status="exact",
                gate_trace_json=json.dumps([{
                    "gate_name": "risk_geometry_gate",
                    "decision": "allow",
                    "missing_fields": [],
                }]),
                decision_delta_classification="same_allow",
                created_at=(base_time + timedelta(hours=i)).isoformat(),
            )

        result = build_weekly_ceo_input(
            session, datetime(2024, 6, 10), min_sample_size=30
        )

        # Gate summaries should be empty because sample size not met
        assert result.gate_summaries == []
        assert result.meets_sample_size is False


# ---------------------------------------------------------------------------
# Test: Reviewer Access (Requirement 11.3)
# ---------------------------------------------------------------------------


class TestReviewerAccess:
    """Test get_replay_for_reviewer — Requirement 11.3."""

    def test_get_replay_by_id(self, session):
        """Reviewer can retrieve replay result by replay_id."""
        record = _insert_audit_record(
            session,
            replay_id="test-replay-001",
            replay_status="exact",
            decision_delta_classification="replay_allows_original_reject",
            divergence_cause="metadata_wiring_defect",
        )

        result = get_replay_for_reviewer(session, "test-replay-001")

        assert result is not None
        assert result.replay_id == "test-replay-001"
        assert result.replay_status == "exact"
        assert result.decision_delta_classification == "replay_allows_original_reject"
        assert result.divergence_cause == "metadata_wiring_defect"

    def test_nonexistent_replay_returns_none(self, session):
        """Non-existent replay_id returns None."""
        result = get_replay_for_reviewer(session, "nonexistent-id")
        assert result is None

    def test_get_replay_by_candidate_id(self, session):
        """Reviewer can retrieve replay results by candidate audit ID."""
        candidate_id = "candidate-123"
        _insert_audit_record(session, candidate_id=candidate_id, replay_id="r1")
        _insert_audit_record(session, candidate_id=candidate_id, replay_id="r2")

        results = get_replay_by_candidate_id(session, candidate_id)
        assert len(results) == 2
        replay_ids = {r.replay_id for r in results}
        assert "r1" in replay_ids
        assert "r2" in replay_ids

    def test_reviewer_access_is_read_only(self, session):
        """Reviewer retrieval does not mutate any records."""
        record = _insert_audit_record(
            session,
            replay_id="readonly-test",
            replay_status="exact",
        )

        # Read the record
        result = get_replay_for_reviewer(session, "readonly-test")
        assert result is not None

        # Verify original record is unchanged
        row = session.execute(
            text("SELECT replay_status FROM replay_audit_records WHERE replay_id = :rid"),
            {"rid": "readonly-test"},
        ).fetchone()
        assert row[0] == "exact"

    def test_reviewer_result_includes_policy_version(self, session):
        """Reviewer result includes parsed policy version."""
        policy = {"name": "current", "gate_revision": "abc123", "config_digest": "def456"}
        _insert_audit_record(
            session,
            replay_id="policy-test",
            policy_version_json=json.dumps(policy),
        )

        result = get_replay_for_reviewer(session, "policy-test")
        assert result.policy_version == policy

    def test_reviewer_result_includes_gate_trace(self, session):
        """Reviewer result includes parsed gate trace."""
        trace = [
            {"gate_name": "setup_quality_gate", "decision": "allow", "missing_fields": []},
            {"gate_name": "risk_geometry_gate", "decision": "reject", "missing_fields": ["signal_strength"]},
        ]
        _insert_audit_record(
            session,
            replay_id="trace-test",
            gate_trace_json=json.dumps(trace),
        )

        result = get_replay_for_reviewer(session, "trace-test")
        assert result.gate_trace is not None
        assert len(result.gate_trace) == 2
        assert result.gate_trace[0]["gate_name"] == "setup_quality_gate"


# ---------------------------------------------------------------------------
# Test: Representative Examples (Requirement 11.5)
# ---------------------------------------------------------------------------


class TestRepresentativeExamples:
    """Test format_findings_with_examples — Requirements 11.5, 11.6."""

    def test_examples_capped_at_three(self):
        """At most 3 examples per finding."""
        examples = [
            ReplayExample(
                replay_id=f"r{i}",
                candidate_id=f"c{i}",
                symbol="TSLA",
                setup_type="news_breakout",
                profile="aggressive",
                delta_classification="replay_allows_original_reject",
                divergence_cause="metadata_wiring_defect",
                replay_confidence="exact",
                summary=f"Example {i}",
            )
            for i in range(5)
        ]

        finding = CEOFinding(
            finding_type="defect",
            title="Test finding",
            description="Test",
            replay_confidence="exact",
            affected_candidate_count=5,
            estimated_return_impact_pct=2.0,
            divergence_cause="metadata_wiring_defect",
            examples_exact=examples,
            examples_partial=[],
        )

        formatted = format_findings_with_examples([finding], max_examples=3)
        assert len(formatted) == 1
        assert len(formatted[0]["examples"]["exact"]) == 3

    def test_at_least_one_example_when_available(self):
        """At least 1 example per finding when records exist."""
        examples = [
            ReplayExample(
                replay_id="r1",
                candidate_id="c1",
                symbol="AAPL",
                setup_type="vwap_reclaim",
                profile="moderate",
                delta_classification="same_reject",
                divergence_cause=None,
                replay_confidence="exact",
                summary="Example 1",
            )
        ]

        finding = CEOFinding(
            finding_type="delta",
            title="Test",
            description="Test",
            replay_confidence="exact",
            affected_candidate_count=1,
            estimated_return_impact_pct=1.0,
            divergence_cause=None,
            examples_exact=examples,
            examples_partial=[],
        )

        formatted = format_findings_with_examples([finding])
        assert len(formatted[0]["examples"]["exact"]) >= 1

    def test_partial_and_exact_separated_structurally(self):
        """Partial results are structurally separated from exact (Req 11.6)."""
        exact_examples = [
            ReplayExample(
                replay_id="r1", candidate_id="c1", symbol="TSLA",
                setup_type="news_breakout", profile="aggressive",
                delta_classification="replay_allows_original_reject",
                divergence_cause="metadata_wiring_defect",
                replay_confidence="exact", summary="Exact example",
            )
        ]
        partial_examples = [
            ReplayExample(
                replay_id="r2", candidate_id="c2", symbol="AAPL",
                setup_type="vwap_reclaim", profile="moderate",
                delta_classification="replay_allows_original_reject",
                divergence_cause="metadata_wiring_defect",
                replay_confidence="partial", summary="Partial example",
            )
        ]

        finding = CEOFinding(
            finding_type="defect",
            title="Test",
            description="Test",
            replay_confidence="exact",
            affected_candidate_count=2,
            estimated_return_impact_pct=1.5,
            divergence_cause="metadata_wiring_defect",
            examples_exact=exact_examples,
            examples_partial=partial_examples,
        )

        formatted = format_findings_with_examples([finding])
        assert "exact" in formatted[0]["examples"]
        assert "partial" in formatted[0]["examples"]
        assert len(formatted[0]["examples"]["exact"]) == 1
        assert len(formatted[0]["examples"]["partial"]) == 1

    def test_examples_include_audit_identifiers(self):
        """Each example is linked to its original audit identifier (Req 11.5)."""
        examples = [
            ReplayExample(
                replay_id="replay-abc",
                candidate_id="candidate-xyz",
                symbol="TSLA",
                setup_type="news_breakout",
                profile="aggressive",
                delta_classification="replay_allows_original_reject",
                divergence_cause="metadata_wiring_defect",
                replay_confidence="exact",
                summary="Test",
            )
        ]

        finding = CEOFinding(
            finding_type="defect",
            title="Test",
            description="Test",
            replay_confidence="exact",
            affected_candidate_count=1,
            estimated_return_impact_pct=1.0,
            divergence_cause="metadata_wiring_defect",
            examples_exact=examples,
            examples_partial=[],
        )

        formatted = format_findings_with_examples([finding])
        example = formatted[0]["examples"]["exact"][0]
        assert example["replay_id"] == "replay-abc"
        assert example["candidate_id"] == "candidate-xyz"

    def test_examples_labeled_with_confidence(self):
        """Each example is labeled with replay confidence (Req 11.5)."""
        examples = [
            ReplayExample(
                replay_id="r1",
                candidate_id="c1",
                symbol="TSLA",
                setup_type="news_breakout",
                profile="aggressive",
                delta_classification="replay_allows_original_reject",
                divergence_cause="metadata_wiring_defect",
                replay_confidence="exact",
                summary="Test",
            )
        ]

        finding = CEOFinding(
            finding_type="defect",
            title="Test",
            description="Test",
            replay_confidence="exact",
            affected_candidate_count=1,
            estimated_return_impact_pct=1.0,
            divergence_cause="metadata_wiring_defect",
            examples_exact=examples,
            examples_partial=[],
        )

        formatted = format_findings_with_examples([finding])
        example = formatted[0]["examples"]["exact"][0]
        assert example["replay_confidence"] == "exact"


# ---------------------------------------------------------------------------
# Test: Confidence Labeling (Requirement 11.6)
# ---------------------------------------------------------------------------


class TestConfidenceLabeling:
    """Test that results are labeled with confidence — Requirement 11.6."""

    def test_daily_findings_have_confidence(self, session):
        """Every finding in daily CEO input has replay_confidence label."""
        _insert_audit_record(
            session,
            replay_status="exact",
            divergence_cause="metadata_wiring_defect",
            decision_delta_classification="replay_allows_original_reject",
            counterfactual_outcome_json=json.dumps({"return_60m": 3.0}),
            created_at=datetime(2024, 6, 15, 14, 0, 0).isoformat(),
        )

        result = build_daily_ceo_input(session, datetime(2024, 6, 15))
        for finding in result.findings:
            assert finding.replay_confidence in ("exact", "partial")

    def test_reviewer_result_has_status(self, session):
        """Reviewer replay result includes replay_status as confidence."""
        _insert_audit_record(
            session,
            replay_id="confidence-test",
            replay_status="partial",
        )

        result = get_replay_for_reviewer(session, "confidence-test")
        assert result.replay_status == "partial"
