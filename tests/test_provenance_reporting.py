"""Unit tests for utils/provenance_reporting.py.

Tests coverage metrics computation, daily/weekly report generation,
CEO summary ordering, and all requirement constraints.

Requirements: 1.6, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 11.6, 11.7
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, text

from db.provenance_schema import init_provenance_schema
from utils.provenance_reporting import (
    EXPLORATORY_THRESHOLD,
    MAX_EXAMPLES_PER_CATEGORY,
    REPEATED_DEFECT_THRESHOLD,
    CoverageMetrics,
    EconomicOutcomes,
    FindingCategory,
    ProvenanceReport,
    build_ceo_summary,
    compute_lineage_coverage,
    generate_daily_report,
    generate_weekly_report,
)


@pytest.fixture
def engine():
    """Create an in-memory SQLite engine with provenance schema."""
    eng = create_engine("sqlite:///:memory:")
    init_provenance_schema(eng)
    return eng


def _insert_provenance_event(
    conn,
    lineage_id: str,
    stage_name: str,
    sequence_number: int,
    *,
    timestamp: datetime | None = None,
    validation_before: str = "valid",
    validation_after: str = "valid",
    is_terminal: bool = False,
    mutation_reason_code: str = "passthrough",
    profile: str = "aggressive",
    symbol: str = "AAPL",
    setup_type: str = "gap_and_go",
    total_dollar_risk: str = "100.00",
    reward_distance: str = "2.00",
    quantity: str = "10",
):
    """Helper to insert a provenance event for testing."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    input_contract = {
        "profile": profile,
        "symbol": symbol,
        "setup_type": setup_type,
    }
    geometry_after = {
        "total_dollar_risk": total_dollar_risk,
        "reward_distance": reward_distance,
        "quantity": quantity,
    }

    conn.execute(
        text("""
            INSERT INTO provenance_events (
                lineage_id, stage_name, stage_version, sequence_number,
                mutation_ordinal, timestamp, input_contract_json,
                output_contract_json, fields_changed_json,
                mutation_reason_code, rule_id, geometry_before_json,
                geometry_after_json, validation_before, validation_after,
                attempt_ordinal, is_terminal, payload_truncated
            ) VALUES (
                :lineage_id, :stage_name, '1.0', :sequence_number,
                1, :timestamp, :input_contract_json,
                '{}', '[]',
                :mutation_reason_code, NULL, '{}',
                :geometry_after_json, :validation_before, :validation_after,
                1, :is_terminal, 0
            )
        """),
        {
            "lineage_id": lineage_id,
            "stage_name": stage_name,
            "sequence_number": sequence_number,
            "timestamp": timestamp.isoformat(),
            "input_contract_json": json.dumps(input_contract),
            "geometry_after_json": json.dumps(geometry_after),
            "validation_before": validation_before,
            "validation_after": validation_after,
            "is_terminal": 1 if is_terminal else 0,
            "mutation_reason_code": mutation_reason_code,
        },
    )


def _insert_complete_lineage(
    conn,
    lineage_id: str,
    *,
    timestamp: datetime | None = None,
    profile: str = "aggressive",
    symbol: str = "AAPL",
    setup_type: str = "gap_and_go",
    mode: str = "candidate_id",
):
    """Insert a complete set of provenance events for a lineage."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    if mode == "candidate_id":
        stages = [
            "trusted_input", "raw_pm_output", "parsed_pm_decision",
            "candidate_resolution", "behavioral_adjustment", "pre_gate_snapshot",
        ]
    else:
        stages = [
            "trusted_input", "raw_pm_output", "parsed_pm_decision",
            "price_repair", "behavioral_adjustment", "pre_gate_snapshot",
        ]

    for i, stage in enumerate(stages, start=1):
        _insert_provenance_event(
            conn, lineage_id, stage, i,
            timestamp=timestamp + timedelta(seconds=i),
            profile=profile,
            symbol=symbol,
            setup_type=setup_type,
        )


# ─── Coverage Metrics Tests ───────────────────────────────────────────────────


class TestComputeLineageCoverage:
    """Tests for compute_lineage_coverage (Requirement 1.6)."""

    def test_empty_database_returns_zero_coverage(self, engine):
        """No events → 0 total, 0% coverage."""
        metrics = compute_lineage_coverage(engine)
        assert metrics.total_initiated == 0
        assert metrics.coverage_pct == 0.0

    def test_complete_lineage_gives_100_percent(self, engine):
        """A lineage with all expected stages → 100% coverage."""
        lid = str(uuid.uuid4())
        with engine.begin() as conn:
            _insert_complete_lineage(conn, lid)

        metrics = compute_lineage_coverage(engine)
        assert metrics.total_initiated == 1
        assert metrics.complete_provenance == 1
        assert metrics.coverage_pct == 100.0

    def test_incomplete_lineage_gives_partial_coverage(self, engine):
        """A lineage missing stages → incomplete."""
        lid = str(uuid.uuid4())
        ts = datetime.now(timezone.utc)
        with engine.begin() as conn:
            # Only insert 3 of 6 expected stages
            _insert_provenance_event(conn, lid, "trusted_input", 1, timestamp=ts)
            _insert_provenance_event(conn, lid, "raw_pm_output", 2, timestamp=ts)
            _insert_provenance_event(conn, lid, "parsed_pm_decision", 3, timestamp=ts)

        metrics = compute_lineage_coverage(engine)
        assert metrics.total_initiated == 1
        assert metrics.incomplete_provenance == 1
        assert metrics.coverage_pct == 0.0

    def test_terminal_lineage_counts_as_complete(self, engine):
        """A lineage that terminated early is considered complete."""
        lid = str(uuid.uuid4())
        ts = datetime.now(timezone.utc)
        with engine.begin() as conn:
            _insert_provenance_event(conn, lid, "trusted_input", 1, timestamp=ts)
            _insert_provenance_event(
                conn, lid, "raw_pm_output", 2,
                timestamp=ts, is_terminal=True,
            )

        metrics = compute_lineage_coverage(engine)
        assert metrics.total_initiated == 1
        assert metrics.complete_provenance == 1
        assert metrics.coverage_pct == 100.0

    def test_mixed_lineages_coverage(self, engine):
        """Mix of complete and incomplete → correct percentage."""
        lid_complete = str(uuid.uuid4())
        lid_incomplete = str(uuid.uuid4())
        ts = datetime.now(timezone.utc)

        with engine.begin() as conn:
            _insert_complete_lineage(conn, lid_complete, timestamp=ts)
            # Incomplete lineage — only 2 stages
            _insert_provenance_event(
                conn, lid_incomplete, "trusted_input", 1, timestamp=ts
            )
            _insert_provenance_event(
                conn, lid_incomplete, "raw_pm_output", 2, timestamp=ts
            )

        metrics = compute_lineage_coverage(engine)
        assert metrics.total_initiated == 2
        assert metrics.complete_provenance == 1
        assert metrics.incomplete_provenance == 1
        assert metrics.coverage_pct == 50.0

    def test_profile_breakdown(self, engine):
        """Coverage is broken down by PM profile."""
        lid1 = str(uuid.uuid4())
        lid2 = str(uuid.uuid4())
        ts = datetime.now(timezone.utc)

        with engine.begin() as conn:
            _insert_complete_lineage(
                conn, lid1, timestamp=ts, profile="aggressive"
            )
            _insert_complete_lineage(
                conn, lid2, timestamp=ts, profile="conservative"
            )

        metrics = compute_lineage_coverage(engine)
        assert "aggressive" in metrics.by_profile
        assert "conservative" in metrics.by_profile
        assert metrics.by_profile["aggressive"]["coverage_pct"] == 100.0
        assert metrics.by_profile["conservative"]["coverage_pct"] == 100.0

    def test_stage_breakdown_shows_missing_stages(self, engine):
        """by_stage shows which stages are most commonly missing."""
        lid = str(uuid.uuid4())
        ts = datetime.now(timezone.utc)

        with engine.begin() as conn:
            # Insert all except pre_gate_snapshot and behavioral_adjustment
            _insert_provenance_event(conn, lid, "trusted_input", 1, timestamp=ts)
            _insert_provenance_event(conn, lid, "raw_pm_output", 2, timestamp=ts)
            _insert_provenance_event(conn, lid, "parsed_pm_decision", 3, timestamp=ts)
            _insert_provenance_event(conn, lid, "candidate_resolution", 4, timestamp=ts)

        metrics = compute_lineage_coverage(engine)
        assert "behavioral_adjustment" in metrics.by_stage
        assert "pre_gate_snapshot" in metrics.by_stage
        assert metrics.by_stage["behavioral_adjustment"] == 1
        assert metrics.by_stage["pre_gate_snapshot"] == 1

    def test_time_window_filtering(self, engine):
        """Only events within time window are counted."""
        lid_in = str(uuid.uuid4())
        lid_out = str(uuid.uuid4())
        ts_in = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        ts_out = datetime(2024, 6, 10, 12, 0, 0, tzinfo=timezone.utc)

        with engine.begin() as conn:
            _insert_complete_lineage(conn, lid_in, timestamp=ts_in)
            _insert_complete_lineage(conn, lid_out, timestamp=ts_out)

        start = datetime(2024, 6, 14, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 6, 16, 0, 0, 0, tzinfo=timezone.utc)

        metrics = compute_lineage_coverage(engine, start_time=start, end_time=end)
        assert metrics.total_initiated == 1
        assert metrics.complete_provenance == 1

    def test_profile_filter(self, engine):
        """Profile filter only counts matching lineages."""
        lid1 = str(uuid.uuid4())
        lid2 = str(uuid.uuid4())
        ts = datetime.now(timezone.utc)

        with engine.begin() as conn:
            _insert_complete_lineage(conn, lid1, timestamp=ts, profile="aggressive")
            _insert_complete_lineage(conn, lid2, timestamp=ts, profile="conservative")

        metrics = compute_lineage_coverage(engine, profile="aggressive")
        assert metrics.total_initiated == 1
        assert metrics.complete_provenance == 1


# ─── Report Generation Tests ─────────────────────────────────────────────────


class TestGenerateDailyReport:
    """Tests for generate_daily_report (Requirement 13.1)."""

    def test_empty_day_returns_empty_report(self, engine):
        """No data → empty report with zero metrics."""
        report = generate_daily_report(engine, report_date=datetime(2024, 6, 15, tzinfo=timezone.utc))
        assert report.total_candidates == 0
        assert report.period_type == "daily"

    def test_daily_report_covers_one_day(self, engine):
        """Daily report spans exactly one calendar day."""
        report_date = datetime(2024, 6, 15, tzinfo=timezone.utc)
        report = generate_daily_report(engine, report_date=report_date)

        assert report.period_start == datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
        assert report.period_end == datetime(2024, 6, 16, 0, 0, 0, tzinfo=timezone.utc)
        assert report.period_type == "daily"

    def test_report_with_data(self, engine):
        """Report correctly counts candidates and classifications."""
        ts = datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        lid1 = str(uuid.uuid4())
        lid2 = str(uuid.uuid4())

        with engine.begin() as conn:
            _insert_complete_lineage(conn, lid1, timestamp=ts, profile="aggressive")
            # Second lineage with invalid geometry at raw_pm_output stage
            _insert_provenance_event(
                conn, lid2, "trusted_input", 1, timestamp=ts, profile="conservative"
            )
            _insert_provenance_event(
                conn, lid2, "raw_pm_output", 2,
                timestamp=ts + timedelta(seconds=1),
                validation_after="invalid",
                mutation_reason_code="geometry_invalid",
                profile="conservative",
            )
            _insert_provenance_event(
                conn, lid2, "parsed_pm_decision", 3,
                timestamp=ts + timedelta(seconds=2),
                is_terminal=True,
                profile="conservative",
            )

        report = generate_daily_report(
            engine, report_date=datetime(2024, 6, 15, tzinfo=timezone.utc)
        )
        assert report.total_candidates == 2
        assert report.malformed_at_pm_stage >= 1


class TestGenerateWeeklyReport:
    """Tests for generate_weekly_report (Requirement 13.1)."""

    def test_weekly_report_covers_seven_days(self, engine):
        """Weekly report spans exactly 7 calendar days."""
        end_date = datetime(2024, 6, 15, tzinfo=timezone.utc)
        report = generate_weekly_report(engine, end_date=end_date)

        assert report.period_start == datetime(2024, 6, 8, 0, 0, 0, tzinfo=timezone.utc)
        assert report.period_end == datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
        assert report.period_type == "weekly"

    def test_weekly_aggregates_across_days(self, engine):
        """Weekly report aggregates data from multiple days."""
        lid1 = str(uuid.uuid4())
        lid2 = str(uuid.uuid4())
        ts1 = datetime(2024, 6, 9, 14, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 6, 13, 14, 0, 0, tzinfo=timezone.utc)

        with engine.begin() as conn:
            _insert_complete_lineage(conn, lid1, timestamp=ts1)
            _insert_complete_lineage(conn, lid2, timestamp=ts2)

        report = generate_weekly_report(
            engine, end_date=datetime(2024, 6, 15, tzinfo=timezone.utc)
        )
        assert report.total_candidates == 2


# ─── CEO Summary Tests ────────────────────────────────────────────────────────


class TestBuildCEOSummary:
    """Tests for build_ceo_summary (Requirements 13.5, 13.6, 13.7, 11.7)."""

    def _make_report_with_defects(
        self, defect_counts: dict[str, int]
    ) -> ProvenanceReport:
        """Helper to create a report with specific defect counts."""
        report = ProvenanceReport(
            period_start=datetime(2024, 6, 15, tzinfo=timezone.utc),
            period_end=datetime(2024, 6, 16, tzinfo=timezone.utc),
            period_type="daily",
            total_candidates=100,
        )
        report.coverage = CoverageMetrics(
            total_initiated=100,
            complete_provenance=90,
            incomplete_provenance=10,
            coverage_pct=90.0,
        )

        for cat, count in defect_counts.items():
            report.defects_by_attribution[cat] = FindingCategory(
                category=cat,
                count=count,
                is_exploratory=count < EXPLORATORY_THRESHOLD,
                representative_examples=[
                    str(uuid.uuid4())
                    for _ in range(min(count, MAX_EXAMPLES_PER_CATEGORY))
                ],
                economic_outcomes=EconomicOutcomes(
                    total_dollar_risk=Decimal("500") * count,
                    total_potential_reward=Decimal("1000") * count,
                    count=count,
                ),
            )

        return report

    def test_repeated_defects_appear_before_recommendations(self):
        """CEO summary orders repeated upstream defects before threshold tuning (Req 13.6)."""
        report = self._make_report_with_defects({
            "raw_pm_output_invalid": 5,
            "policy_rejection_of_valid_contract": 10,
        })

        summary = build_ceo_summary(report)

        # Repeated defects section should be non-empty
        assert len(summary["repeated_upstream_defects"]) >= 1
        assert summary["repeated_upstream_defects"][0]["category"] == "raw_pm_output_invalid"

        # Threshold tuning recommendations come after
        assert "threshold_tuning_recommendations" in summary

    def test_exploratory_label_under_20_occurrences(self):
        """Categories with <20 occurrences are labeled exploratory (Req 13.5)."""
        report = self._make_report_with_defects({
            "raw_pm_output_invalid": 5,  # < 20 → exploratory
            "behavioral_adjustment_invalid": 25,  # >= 20 → not exploratory
        })

        summary = build_ceo_summary(report)

        # Find the raw_pm_output_invalid category
        categories = {c["category"]: c for c in summary["defect_categories"]}
        assert categories["raw_pm_output_invalid"]["is_exploratory"] is True
        assert categories["behavioral_adjustment_invalid"]["is_exploratory"] is False

    def test_does_not_recommend_loosening_gate_for_malformed_upstream(self):
        """Never recommends loosening a gate that blocked malformed contracts (Req 13.7)."""
        report = self._make_report_with_defects({
            "raw_pm_output_invalid": 30,
            "policy_rejection_of_valid_contract": 10,
        })

        summary = build_ceo_summary(report)

        # Check that recommendations have the safety note
        for rec in summary["threshold_tuning_recommendations"]:
            assert "Do NOT loosen" in rec.get("note", "")

    def test_reconstruction_is_report_only(self):
        """Reconstruction analysis marked as report-only (Req 11.7)."""
        report = ProvenanceReport(
            period_start=datetime(2024, 6, 15, tzinfo=timezone.utc),
            period_end=datetime(2024, 6, 16, tzinfo=timezone.utc),
            period_type="daily",
        )
        report.coverage = CoverageMetrics()
        report.reconstruction_outcomes = {
            "valid_geometry_preserved": 5,
            "valid_geometry_degraded": 3,
        }

        summary = build_ceo_summary(report)

        assert "No automated policy changes" in summary["reconstruction_analysis"]["note"]
        assert summary["reconstruction_analysis"]["outcomes"] == {
            "valid_geometry_preserved": 5,
            "valid_geometry_degraded": 3,
        }

    def test_separates_counts_from_economic_outcomes(self):
        """Counts and economic outcomes are separate (Req 13.4)."""
        report = self._make_report_with_defects({
            "raw_pm_output_invalid": 5,
        })

        summary = build_ceo_summary(report)

        # Each category has both count AND economic_impact as separate fields
        cat = summary["defect_categories"][0]
        assert "count" in cat
        assert "economic_impact" in cat
        assert "total_dollar_risk" in cat["economic_impact"]
        assert "total_potential_reward" in cat["economic_impact"]
        assert "affected_candidates" in cat["economic_impact"]

    def test_representative_examples_capped_at_5(self):
        """At most 5 representative examples per category (Req 13.3)."""
        report = self._make_report_with_defects({
            "raw_pm_output_invalid": 50,
        })

        summary = build_ceo_summary(report)

        cat = next(
            c for c in summary["defect_categories"]
            if c["category"] == "raw_pm_output_invalid"
        )
        assert len(cat["representative_examples"]) <= MAX_EXAMPLES_PER_CATEGORY

    def test_defects_below_threshold_not_in_repeated(self):
        """Defects with < 3 occurrences are NOT in repeated_upstream_defects."""
        report = self._make_report_with_defects({
            "raw_pm_output_invalid": 2,  # Below threshold
        })

        summary = build_ceo_summary(report)
        assert len(summary["repeated_upstream_defects"]) == 0

    def test_non_exploratory_sorted_before_exploratory(self):
        """Non-exploratory categories appear before exploratory in listing."""
        report = self._make_report_with_defects({
            "raw_pm_output_invalid": 5,  # exploratory
            "behavioral_adjustment_invalid": 25,  # not exploratory
        })

        summary = build_ceo_summary(report)

        categories = summary["defect_categories"]
        # Non-exploratory should come first
        first_non_exp = next(
            (i for i, c in enumerate(categories) if not c["is_exploratory"]),
            None,
        )
        first_exp = next(
            (i for i, c in enumerate(categories) if c["is_exploratory"]),
            None,
        )
        if first_non_exp is not None and first_exp is not None:
            assert first_non_exp < first_exp

    def test_headline_metrics_present(self):
        """CEO summary includes headline metrics."""
        report = self._make_report_with_defects({
            "raw_pm_output_invalid": 5,
        })
        report.policy_rejections = 3
        report.integrity_rejections = 7

        summary = build_ceo_summary(report)

        headline = summary["headline_metrics"]
        assert headline["total_candidates"] == 100
        assert headline["policy_rejections"] == 3
        assert headline["integrity_rejections"] == 7
        assert headline["coverage_pct"] == 90.0

    def test_breakdowns_included(self):
        """CEO summary includes profile, setup_type, and symbol_class breakdowns."""
        report = ProvenanceReport(
            period_start=datetime(2024, 6, 15, tzinfo=timezone.utc),
            period_end=datetime(2024, 6, 16, tzinfo=timezone.utc),
            period_type="daily",
        )
        report.coverage = CoverageMetrics()
        report.by_profile = {"aggressive": {"total": 10, "defects": 3}}
        report.by_setup_type = {"gap_and_go": {"total": 5, "defects": 1}}
        report.by_symbol_class = {"single_stock": {"total": 8, "defects": 2}}

        summary = build_ceo_summary(report)

        assert summary["breakdowns"]["by_profile"] == {"aggressive": {"total": 10, "defects": 3}}
        assert summary["breakdowns"]["by_setup_type"] == {"gap_and_go": {"total": 5, "defects": 1}}
        assert summary["breakdowns"]["by_symbol_class"] == {"single_stock": {"total": 8, "defects": 2}}
