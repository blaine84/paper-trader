"""Tests for core/replay/backfill.py — Historical backfill module.

Tests cover:
- Grading logic: exact, partial, unscorable classification
- Era labeling: historical (pre-snapshot) vs post-snapshot
- Coverage report generation: per-field presence %, coverage gap flagging
- Missing fields explicitly marked (never inferred/interpolated/substituted)
- Backfill from multiple data sources

Requirements: 12.1, 12.2, 12.3, 12.4, 12.6
"""

import json
import uuid
from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, text

from core.replay.backfill import (
    BackfillCoverageReport,
    BackfillGrade,
    FieldCoverage,
    DECISION_SNAPSHOT_FIELDS,
    assess_backfill_coverage,
    generate_coverage_report,
    grade_historical_record,
)
from core.replay.candidate_sourcer import ReplayCandidate, SourceReference
from core.replay.input_reconstructor import InputSource, ReplayInputBundle
from core.replay.policy_version import PolicyVersion
from db.replay_schema import init_replay_db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def policy_version():
    """A test policy version."""
    return PolicyVersion(
        name="test_policy",
        gate_revision="abc123",
        config_digest="sha256:test",
        feature_flags={"SETUP_SPECIFIC_RR_THRESHOLDS": True},
        benchmark_version="2024-01-01",
        config_source_timestamp=datetime(2024, 1, 1),
        gate_ordering_version="v1.0",
        adapter_version="1.0.0",
    )


@pytest.fixture
def base_candidate():
    """A minimal ReplayCandidate for testing."""
    return ReplayCandidate(
        candidate_id=str(uuid.uuid4()),
        lineage_id=None,
        symbol="TSLA",
        profile="aggressive",
        direction="LONG",
        setup_type="news_breakout",
        entry_price=Decimal("185.50"),
        stop_price=Decimal("184.00"),
        target_price=Decimal("188.00"),
        quantity=Decimal("10"),
        entry_timestamp=datetime(2024, 6, 15, 14, 30, 0),
        original_decision="reject",
        original_gate="risk_geometry_gate",
        original_reason_code="RISK_REWARD_BELOW_THRESHOLD",
        source_records=(
            SourceReference(
                source_table="blocked_trade_candidates",
                source_id=1,
                field_contributions=("symbol", "entry_price"),
            ),
        ),
        geometry_complete=True,
    )


def _make_input_source(field_name: str, value, available: bool = True) -> InputSource:
    """Helper to create an InputSource."""
    return InputSource(
        field_name=field_name,
        value=value if available else None,
        source_timestamp=datetime(2024, 6, 15, 14, 29, 0) if available else None,
        source_record_id="test:1" if available else None,
        status="available" if available else "unavailable",
    )


def _build_full_inputs(candidate, snapshot_id=None):
    """Build a ReplayInputBundle with all fields present (exact)."""
    inputs = {}
    field_values = {
        "symbol": "TSLA",
        "profile": "aggressive",
        "direction": "LONG",
        "setup_type": "news_breakout",
        "entry_price": Decimal("185.50"),
        "stop_price": Decimal("184.00"),
        "target_price": Decimal("188.00"),
        "quantity": Decimal("10"),
        "signal_strength": 9.0,
        "confidence_value": 8.5,
        "atr_value": 1.25,
        "atr_timestamp": datetime(2024, 6, 15, 14, 25, 0),
        "account_equity": Decimal("50000"),
        "available_cash": Decimal("25000"),
        "open_positions": [{"symbol": "AAPL", "side": "LONG", "quantity": 5}],
        "case_library_stats": {"win_rate": 0.65, "total_cases": 40},
        "selection_score": 7.5,
        "execution_score": 8.0,
        "override_confidence_score": 0.85,
        "override_reason": None,
        "catalyst_type": "earnings_surprise",
        "max_dollar_risk": 500.0,
        "trade_metadata": "test metadata",
        "trade_rationale": "strong breakout setup",
        "atr_source": "5min",
        "rationale": "Price breaking above resistance with volume",
        "thesis": "Bullish continuation",
        "indicators": ["volume_spike", "momentum"],
        "quote_timestamp": datetime(2024, 6, 15, 14, 28, 0),
        "strength": "strong",
        "conviction": "high",
    }
    for field_name in DECISION_SNAPSHOT_FIELDS:
        value = field_values.get(field_name)
        inputs[field_name] = _make_input_source(field_name, value, available=value is not None)

    # override_reason is legitimately None but still "available" as a non-critical field
    inputs["override_reason"] = InputSource(
        field_name="override_reason",
        value="none",
        source_timestamp=datetime(2024, 6, 15, 14, 29, 0),
        source_record_id="test:1",
        status="available",
    )

    return ReplayInputBundle(
        candidate=candidate,
        cutoff=datetime(2024, 6, 15, 14, 30, 0),
        classification="exact",
        inputs=inputs,
        missing_inputs=[],
        snapshot_id=snapshot_id,
    )


def _build_partial_inputs(candidate, missing_non_critical_fields: list[str]):
    """Build a ReplayInputBundle with some non-critical fields missing (partial)."""
    inputs = {}
    field_values = {
        "symbol": "TSLA",
        "profile": "aggressive",
        "direction": "LONG",
        "setup_type": "news_breakout",
        "entry_price": Decimal("185.50"),
        "stop_price": Decimal("184.00"),
        "target_price": Decimal("188.00"),
        "quantity": Decimal("10"),
        "signal_strength": 9.0,
        "confidence_value": 8.5,
        "atr_value": 1.25,
        "atr_timestamp": datetime(2024, 6, 15, 14, 25, 0),
        "account_equity": Decimal("50000"),
        "available_cash": Decimal("25000"),
        "open_positions": [],
        "case_library_stats": {"win_rate": 0.65},
        "selection_score": 7.5,
        "execution_score": 8.0,
        "override_confidence_score": 0.85,
        "override_reason": "high_conviction",
        "catalyst_type": "earnings",
        "max_dollar_risk": 500.0,
        "trade_metadata": "metadata",
        "trade_rationale": "rationale",
        "atr_source": "5min",
        "rationale": "test rationale",
        "thesis": "test thesis",
        "indicators": ["vol"],
        "quote_timestamp": datetime(2024, 6, 15, 14, 28, 0),
        "strength": "strong",
        "conviction": "high",
    }
    missing_inputs = []
    for field_name in DECISION_SNAPSHOT_FIELDS:
        if field_name in missing_non_critical_fields:
            inputs[field_name] = _make_input_source(field_name, None, available=False)
            missing_inputs.append({
                "field": field_name,
                "reason": f"No source for '{field_name}'",
                "is_critical": False,
            })
        else:
            value = field_values.get(field_name, "test_value")
            inputs[field_name] = _make_input_source(field_name, value, available=True)

    return ReplayInputBundle(
        candidate=candidate,
        cutoff=datetime(2024, 6, 15, 14, 30, 0),
        classification="partial",
        inputs=inputs,
        missing_inputs=missing_inputs,
        snapshot_id=None,
    )


def _build_unscorable_inputs(candidate, missing_critical_fields: list[str]):
    """Build a ReplayInputBundle with critical fields missing (unscorable)."""
    inputs = {}
    field_values = {
        "symbol": "TSLA",
        "profile": "aggressive",
        "direction": "LONG",
        "setup_type": "news_breakout",
        "entry_price": Decimal("185.50"),
        "stop_price": Decimal("184.00"),
        "target_price": Decimal("188.00"),
        "quantity": Decimal("10"),
        "signal_strength": 9.0,
        "confidence_value": 8.5,
        "atr_value": 1.25,
        "atr_timestamp": datetime(2024, 6, 15, 14, 25, 0),
        "account_equity": Decimal("50000"),
        "available_cash": Decimal("25000"),
        "open_positions": [],
        "case_library_stats": {"win_rate": 0.65},
        "selection_score": 7.5,
        "execution_score": 8.0,
        "override_confidence_score": 0.85,
        "override_reason": "high",
        "catalyst_type": "earnings",
        "max_dollar_risk": 500.0,
        "trade_metadata": "metadata",
        "trade_rationale": "rationale",
        "atr_source": "5min",
        "rationale": "test",
        "thesis": "test",
        "indicators": ["vol"],
        "quote_timestamp": datetime(2024, 6, 15, 14, 28),
        "strength": "strong",
        "conviction": "high",
    }
    missing_inputs = []
    for field_name in DECISION_SNAPSHOT_FIELDS:
        if field_name in missing_critical_fields:
            inputs[field_name] = _make_input_source(field_name, None, available=False)
            missing_inputs.append({
                "field": field_name,
                "reason": f"Critical: no source for '{field_name}'",
                "is_critical": True,
            })
        else:
            value = field_values.get(field_name, "test_value")
            inputs[field_name] = _make_input_source(field_name, value, available=True)

    return ReplayInputBundle(
        candidate=candidate,
        cutoff=datetime(2024, 6, 15, 14, 30, 0),
        classification="unscorable",
        inputs=inputs,
        missing_inputs=missing_inputs,
        snapshot_id=None,
    )


# ---------------------------------------------------------------------------
# Tests: grade_historical_record
# ---------------------------------------------------------------------------


class TestGradeHistoricalRecord:
    """Tests for grade_historical_record function."""

    def test_exact_grade_when_all_fields_present(self, base_candidate):
        """Exact grade when all Decision_Snapshot fields are present."""
        inputs = _build_full_inputs(base_candidate)
        grade = grade_historical_record(base_candidate, inputs)

        assert grade.grade == "exact"
        assert grade.candidate_id == base_candidate.candidate_id
        assert len(grade.missing_fields) == 0
        assert len(grade.critical_missing) == 0

    def test_partial_grade_when_non_critical_missing(self, base_candidate):
        """Partial grade when only non-critical inputs are missing."""
        # account_equity and available_cash are NOT in any GATE_REQUIRED_FIELDS
        inputs = _build_partial_inputs(
            base_candidate, missing_non_critical_fields=["account_equity", "available_cash"]
        )
        grade = grade_historical_record(base_candidate, inputs)

        assert grade.grade == "partial"
        assert "account_equity" in grade.missing_fields
        assert "available_cash" in grade.missing_fields
        assert len(grade.critical_missing) == 0

    def test_unscorable_grade_when_critical_missing(self, base_candidate):
        """Unscorable grade when critical inputs cannot be determined."""
        # entry_price is critical for risk_geometry_gate
        inputs = _build_unscorable_inputs(
            base_candidate, missing_critical_fields=["entry_price"]
        )
        grade = grade_historical_record(base_candidate, inputs)

        assert grade.grade == "unscorable"
        assert "entry_price" in grade.missing_fields
        assert "entry_price" in grade.critical_missing

    def test_era_historical_when_no_snapshot(self, base_candidate):
        """Era is 'historical' when no snapshot_id is present."""
        inputs = _build_full_inputs(base_candidate, snapshot_id=None)
        grade = grade_historical_record(base_candidate, inputs)

        assert grade.era == "historical"

    def test_era_post_snapshot_when_snapshot_exists(self, base_candidate):
        """Era is 'post-snapshot' when snapshot_id is present."""
        snapshot_id = str(uuid.uuid4())
        inputs = _build_full_inputs(base_candidate, snapshot_id=snapshot_id)
        grade = grade_historical_record(base_candidate, inputs)

        assert grade.era == "post-snapshot"

    def test_missing_fields_explicitly_marked(self, base_candidate):
        """Missing fields are explicitly tracked, never silently omitted."""
        inputs = _build_unscorable_inputs(
            base_candidate,
            missing_critical_fields=["signal_strength", "confidence_value"],
        )
        grade = grade_historical_record(base_candidate, inputs)

        # Both missing fields should appear in the grade
        assert "signal_strength" in grade.missing_fields
        assert "confidence_value" in grade.missing_fields

    def test_present_fields_tracked(self, base_candidate):
        """Present fields are correctly reported."""
        inputs = _build_full_inputs(base_candidate)
        grade = grade_historical_record(base_candidate, inputs)

        # All snapshot fields should be present
        for field in DECISION_SNAPSHOT_FIELDS:
            assert field in grade.present_fields


# ---------------------------------------------------------------------------
# Tests: Coverage report generation
# ---------------------------------------------------------------------------


class TestCoverageReport:
    """Tests for coverage report generation logic."""

    def test_field_coverage_percentage_calculation(self, base_candidate):
        """Per-field presence percentage is correctly calculated."""
        # Create 4 grades: 3 with signal_strength present, 1 without
        grades = []
        for i in range(3):
            inputs = _build_full_inputs(base_candidate)
            grades.append(grade_historical_record(base_candidate, inputs))

        # One with signal_strength missing
        inputs = _build_unscorable_inputs(
            base_candidate, missing_critical_fields=["signal_strength"]
        )
        grades.append(grade_historical_record(base_candidate, inputs))

        # Build coverage report from grades
        from core.replay.backfill import _build_coverage_report

        report = _build_coverage_report(
            (datetime(2024, 6, 1), datetime(2024, 6, 30)),
            grades,
        )

        # Find signal_strength coverage
        ss_coverage = next(
            fc for fc in report.field_coverage if fc.field_name == "signal_strength"
        )
        assert ss_coverage.present_count == 3
        assert ss_coverage.total_count == 4
        assert ss_coverage.presence_pct == 75.0
        assert ss_coverage.is_coverage_gap is False

    def test_coverage_gap_flagged_below_50_pct(self, base_candidate):
        """Fields with < 50% presence are flagged as coverage gaps."""
        from core.replay.backfill import _build_coverage_report

        # 4 grades: 1 with case_library_stats, 3 without
        grades = []

        # 1 with all fields
        inputs = _build_full_inputs(base_candidate)
        grades.append(grade_historical_record(base_candidate, inputs))

        # 3 with case_library_stats missing
        for _ in range(3):
            inputs = _build_partial_inputs(
                base_candidate, missing_non_critical_fields=["case_library_stats"]
            )
            grades.append(grade_historical_record(base_candidate, inputs))

        report = _build_coverage_report(
            (datetime(2024, 6, 1), datetime(2024, 6, 30)),
            grades,
        )

        # case_library_stats should be a coverage gap (25% presence)
        cls_coverage = next(
            fc for fc in report.field_coverage if fc.field_name == "case_library_stats"
        )
        assert cls_coverage.presence_pct == 25.0
        assert cls_coverage.is_coverage_gap is True
        assert "case_library_stats" in report.coverage_gaps

    def test_grade_breakdown_counted_correctly(self, base_candidate):
        """Exact, partial, and unscorable counts are correct."""
        from core.replay.backfill import _build_coverage_report

        grades = []
        # 2 exact
        for _ in range(2):
            inputs = _build_full_inputs(base_candidate)
            grades.append(grade_historical_record(base_candidate, inputs))
        # 1 partial
        inputs = _build_partial_inputs(
            base_candidate, missing_non_critical_fields=["trade_metadata"]
        )
        grades.append(grade_historical_record(base_candidate, inputs))
        # 1 unscorable
        inputs = _build_unscorable_inputs(
            base_candidate, missing_critical_fields=["entry_price"]
        )
        grades.append(grade_historical_record(base_candidate, inputs))

        report = _build_coverage_report(
            (datetime(2024, 6, 1), datetime(2024, 6, 30)),
            grades,
        )

        assert report.total_candidates == 4
        assert report.exact_count == 2
        assert report.partial_count == 1
        assert report.unscorable_count == 1

    def test_era_breakdown_in_report(self, base_candidate):
        """Era breakdown correctly separates historical vs post-snapshot."""
        from core.replay.backfill import _build_coverage_report

        grades = []
        # 2 historical (no snapshot)
        for _ in range(2):
            inputs = _build_full_inputs(base_candidate, snapshot_id=None)
            grades.append(grade_historical_record(base_candidate, inputs))
        # 1 post-snapshot
        inputs = _build_full_inputs(base_candidate, snapshot_id=str(uuid.uuid4()))
        grades.append(grade_historical_record(base_candidate, inputs))

        report = _build_coverage_report(
            (datetime(2024, 6, 1), datetime(2024, 6, 30)),
            grades,
        )

        assert report.era_breakdown["historical"] == 2
        assert report.era_breakdown["post-snapshot"] == 1

    def test_empty_backfill_produces_zero_report(self, base_candidate):
        """Empty candidate set produces a report with zero counts."""
        from core.replay.backfill import _build_coverage_report

        report = _build_coverage_report(
            (datetime(2024, 6, 1), datetime(2024, 6, 30)),
            [],
        )

        assert report.total_candidates == 0
        assert report.exact_count == 0
        assert report.partial_count == 0
        assert report.unscorable_count == 0
        # All fields have 0% presence → all are coverage gaps
        for fc in report.field_coverage:
            assert fc.presence_pct == 0.0
            assert fc.is_coverage_gap is True


# ---------------------------------------------------------------------------
# Tests: Integration with session (assess_backfill_coverage)
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """In-memory SQLite engine with all required tables."""
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        # Create minimal production tables needed by candidate_sourcer
        conn.execute(text("""
            CREATE TABLE blocked_trade_candidates (
                id INTEGER PRIMARY KEY,
                symbol VARCHAR(10),
                profile VARCHAR(16),
                direction VARCHAR(10),
                setup_type VARCHAR(64),
                entry_price TEXT,
                stop_price TEXT,
                target_price TEXT,
                quantity TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                blocked_by VARCHAR(64),
                reason_code VARCHAR(64),
                action VARCHAR(64),
                candidate_lineage_id VARCHAR(36),
                decision_snapshot_json TEXT,
                signal_snapshot_json TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE trade_events (
                id INTEGER PRIMARY KEY,
                symbol VARCHAR(10),
                profile VARCHAR(16),
                timestamp DATETIME,
                event_type VARCHAR(64),
                agent VARCHAR(64),
                payload_json TEXT,
                candidate_lineage_id VARCHAR(36)
            )
        """))
        conn.execute(text("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY,
                symbol VARCHAR(10),
                profile VARCHAR(16),
                direction VARCHAR(10),
                setup_type VARCHAR(64),
                entry_price TEXT,
                stop_price TEXT,
                target_price TEXT,
                quantity TEXT,
                entry_time DATETIME,
                status VARCHAR(16) DEFAULT 'open',
                candidate_lineage_id VARCHAR(36)
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
                entry_price TEXT,
                stop_price TEXT,
                target_price TEXT,
                state VARCHAR(16),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                rejection_reason TEXT,
                candidate_lineage_id VARCHAR(36)
            )
        """))
        conn.execute(text("""
            CREATE TABLE balance (
                id INTEGER PRIMARY KEY,
                profile VARCHAR(16),
                timestamp DATETIME,
                cash REAL,
                total_equity REAL
            )
        """))
        conn.execute(text("""
            CREATE TABLE positions (
                id INTEGER PRIMARY KEY,
                profile VARCHAR(16),
                symbol VARCHAR(10),
                side VARCHAR(10),
                quantity REAL,
                avg_cost REAL,
                opened_at DATETIME
            )
        """))
        # Agent memory table for analyst signals
        conn.execute(text("""
            CREATE TABLE agent_memory (
                id INTEGER PRIMARY KEY,
                agent VARCHAR(64),
                symbol VARCHAR(10),
                key VARCHAR(64),
                value TEXT,
                timestamp DATETIME
            )
        """))
    # Initialize replay schema (decision_snapshots, replay_audit_records, etc.)
    init_replay_db(eng)
    return eng


@pytest.fixture
def session(engine):
    """SQLAlchemy session from the test engine."""
    from sqlalchemy.orm import Session
    with Session(engine) as sess:
        yield sess


class TestAssessBackfillCoverage:
    """Integration tests for assess_backfill_coverage with real DB."""

    def test_backfill_from_blocked_trade_candidates(self, session, policy_version):
        """Backfill loads candidates from blocked_trade_candidates."""
        # Insert a blocked candidate
        session.execute(text("""
            INSERT INTO blocked_trade_candidates
            (symbol, profile, direction, setup_type, entry_price, stop_price,
             target_price, quantity, created_at, blocked_by, reason_code)
            VALUES ('TSLA', 'aggressive', 'LONG', 'news_breakout',
                    '185.50', '184.00', '188.00', '10',
                    '2024-06-15 14:30:00', 'risk_geometry_gate',
                    'RISK_REWARD_BELOW_THRESHOLD')
        """))
        session.commit()

        date_range = (datetime(2024, 6, 1), datetime(2024, 6, 30))
        report = assess_backfill_coverage(
            session, date_range, policy_version=policy_version
        )

        assert report.total_candidates >= 1
        # Should have at least one grade
        assert len(report.grades) >= 1

    def test_backfill_from_trades(self, session, policy_version):
        """Backfill loads candidates from trades table."""
        session.execute(text("""
            INSERT INTO trades
            (symbol, profile, direction, setup_type, entry_price, stop_price,
             target_price, quantity, entry_time, status)
            VALUES ('AAPL', 'moderate', 'LONG', 'breakout',
                    '150.00', '148.00', '155.00', '5',
                    '2024-06-10 10:00:00', 'closed')
        """))
        session.commit()

        date_range = (datetime(2024, 6, 1), datetime(2024, 6, 30))
        report = assess_backfill_coverage(
            session, date_range, policy_version=policy_version
        )

        assert report.total_candidates >= 1

    def test_era_label_historical_when_no_snapshot(self, session, policy_version):
        """Candidates without snapshots get era='historical'."""
        session.execute(text("""
            INSERT INTO blocked_trade_candidates
            (symbol, profile, direction, setup_type, entry_price, stop_price,
             target_price, quantity, created_at, blocked_by, reason_code)
            VALUES ('MSFT', 'moderate', 'LONG', 'breakout',
                    '350.00', '348.00', '355.00', '3',
                    '2024-06-12 11:00:00', 'setup_quality_gate', 'LOW_WIN_RATE')
        """))
        session.commit()

        date_range = (datetime(2024, 6, 1), datetime(2024, 6, 30))
        report = assess_backfill_coverage(
            session, date_range, policy_version=policy_version
        )

        # All candidates should be historical (no snapshots in DB)
        for grade in report.grades:
            assert grade.era == "historical"

    def test_era_label_post_snapshot_when_snapshot_exists(self, session, policy_version):
        """Candidates with snapshots get era='post-snapshot'."""
        lineage_id = str(uuid.uuid4())

        # Insert a blocked candidate with lineage_id
        session.execute(text("""
            INSERT INTO blocked_trade_candidates
            (symbol, profile, direction, setup_type, entry_price, stop_price,
             target_price, quantity, created_at, blocked_by, reason_code,
             candidate_lineage_id)
            VALUES ('NVDA', 'aggressive', 'LONG', 'momentum',
                    '500.00', '495.00', '510.00', '2',
                    '2024-06-20 09:35:00', 'risk_geometry_gate',
                    'STOP_DISTANCE', :lineage_id)
        """), {"lineage_id": lineage_id})

        # Insert a matching decision_snapshot
        session.execute(text("""
            INSERT INTO decision_snapshots
            (snapshot_id, schema_version, candidate_lineage_id, timestamp,
             symbol, profile, direction, setup_type,
             decision_payload_json, entry_price, stop_price, target_price,
             quantity, account_equity, available_cash, gate_config_json,
             feature_flags_json, policy_version_id)
            VALUES (:snapshot_id, '1.0', :lineage_id, '2024-06-20 09:35:00',
                    'NVDA', 'aggressive', 'LONG', 'momentum',
                    '{"selection_score": 7.0, "execution_score": 8.0}',
                    '500.00', '495.00', '510.00', '2',
                    '50000', '25000', '{}', '{}', 'test_v1')
        """), {"snapshot_id": str(uuid.uuid4()), "lineage_id": lineage_id})
        session.commit()

        date_range = (datetime(2024, 6, 1), datetime(2024, 6, 30))
        report = assess_backfill_coverage(
            session, date_range, policy_version=policy_version
        )

        # Find the NVDA candidate grade
        nvda_grades = [g for g in report.grades if "NVDA" in str(g.present_fields)
                       or g.era == "post-snapshot"]
        # At least one should be post-snapshot
        assert any(g.era == "post-snapshot" for g in report.grades)

    def test_coverage_gaps_reported(self, session, policy_version):
        """Fields below 50% presence are flagged as coverage gaps."""
        # Insert multiple blocked candidates with minimal data
        for i in range(4):
            session.execute(text("""
                INSERT INTO blocked_trade_candidates
                (symbol, profile, direction, setup_type, entry_price, stop_price,
                 target_price, quantity, created_at, blocked_by, reason_code)
                VALUES (:symbol, 'moderate', 'LONG', 'breakout',
                        '100.00', '98.00', '105.00', '5',
                        :ts, 'risk_geometry_gate', 'TEST')
            """), {
                "symbol": f"SYM{i}",
                "ts": f"2024-06-{10 + i:02d} 10:00:00",
            })
        session.commit()

        date_range = (datetime(2024, 6, 1), datetime(2024, 6, 30))
        report = assess_backfill_coverage(
            session, date_range, policy_version=policy_version
        )

        # Historical data from blocked_trade_candidates won't have many fields
        # (no signal_strength, no case_library_stats, etc.)
        # So there should be coverage gaps
        assert len(report.coverage_gaps) > 0

    def test_unavailable_fields_not_inferred(self, session, policy_version):
        """Missing fields are marked unavailable, never inferred or interpolated."""
        session.execute(text("""
            INSERT INTO blocked_trade_candidates
            (symbol, profile, direction, setup_type, entry_price, stop_price,
             target_price, quantity, created_at, blocked_by, reason_code)
            VALUES ('AMD', 'moderate', 'LONG', 'breakout',
                    '120.00', '118.00', '125.00', '8',
                    '2024-06-18 13:00:00', 'risk_geometry_gate', 'TEST')
        """))
        session.commit()

        date_range = (datetime(2024, 6, 1), datetime(2024, 6, 30))
        report = assess_backfill_coverage(
            session, date_range, policy_version=policy_version
        )

        # Find the AMD candidate's grade
        assert report.total_candidates >= 1
        grade = report.grades[0]

        # Fields like signal_strength, case_library_stats should be missing
        # They should NOT be inferred with default values
        assert len(grade.missing_fields) > 0

    def test_generate_coverage_report_matches_assess(self, session, policy_version):
        """generate_coverage_report returns the same result as assess_backfill_coverage."""
        session.execute(text("""
            INSERT INTO blocked_trade_candidates
            (symbol, profile, direction, setup_type, entry_price, stop_price,
             target_price, quantity, created_at, blocked_by, reason_code)
            VALUES ('GOOG', 'moderate', 'LONG', 'breakout',
                    '140.00', '138.00', '145.00', '4',
                    '2024-06-14 12:00:00', 'setup_quality_gate', 'TEST')
        """))
        session.commit()

        date_range = (datetime(2024, 6, 1), datetime(2024, 6, 30))

        report_assess = assess_backfill_coverage(
            session, date_range, policy_version=policy_version
        )
        report_gen = generate_coverage_report(
            session, date_range, policy_version=policy_version
        )

        assert report_assess.total_candidates == report_gen.total_candidates
        assert report_assess.exact_count == report_gen.exact_count
        assert report_assess.partial_count == report_gen.partial_count
        assert report_assess.unscorable_count == report_gen.unscorable_count
