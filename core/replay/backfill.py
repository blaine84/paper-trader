"""Historical backfill from existing data sources for the Decision Replay Agent.

Reconstructs replay candidates from pre-snapshot era data, grades each record's
replay readiness, generates coverage reports (per-field presence %), and labels
results with era ("historical" vs "post-snapshot").

Key principles:
- Never infers, interpolates, or substitutes — missing = explicitly marked
- Era is determined by whether a decision_snapshot exists for the candidate
- Coverage gaps flagged when field presence < 50%

Requirements: 12.1, 12.2, 12.3, 12.4, 12.6
See: design.md, tasks.md §13.2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import text

from core.replay.candidate_sourcer import (
    ReplayCandidate,
    load_candidates,
    correlate_and_deduplicate,
)
from core.replay.gate_adapter import GATE_REQUIRED_FIELDS
from core.replay.input_reconstructor import (
    InputSource,
    ReplayInputBundle,
    reconstruct_inputs,
    compute_replay_cutoff,
    CORE_GATE_SEQUENCE,
)
from core.replay.policy_version import PolicyVersion, build_current_policy_version

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# All Decision_Snapshot fields required for exact replay
# ---------------------------------------------------------------------------

DECISION_SNAPSHOT_FIELDS: list[str] = [
    "symbol",
    "profile",
    "direction",
    "setup_type",
    "entry_price",
    "stop_price",
    "target_price",
    "quantity",
    "signal_strength",
    "confidence_value",
    "atr_value",
    "atr_timestamp",
    "account_equity",
    "available_cash",
    "open_positions",
    "case_library_stats",
    "selection_score",
    "execution_score",
    "override_confidence_score",
    "override_reason",
    "catalyst_type",
    "max_dollar_risk",
    "trade_metadata",
    "trade_rationale",
    "atr_source",
    "rationale",
    "thesis",
    "indicators",
    "quote_timestamp",
    "strength",
    "conviction",
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackfillGrade:
    """Grade for a single historical record's replay readiness.

    Attributes:
        candidate_id: ID of the replay candidate.
        grade: "exact" | "partial" | "unscorable"
        era: "historical" | "post-snapshot"
        present_fields: Fields that are available with values.
        missing_fields: Fields that are unavailable (explicitly marked).
        critical_missing: Critical fields that are missing (cause unscorable).
    """

    candidate_id: str
    grade: str  # "exact" | "partial" | "unscorable"
    era: str  # "historical" | "post-snapshot"
    present_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]
    critical_missing: tuple[str, ...]


@dataclass(frozen=True)
class FieldCoverage:
    """Coverage statistics for a single field across backfill candidates.

    Attributes:
        field_name: The Decision_Snapshot field name.
        present_count: Number of candidates where this field is available.
        total_count: Total number of candidates assessed.
        presence_pct: Percentage of candidates with this field present (0.0-100.0).
        is_coverage_gap: True if presence_pct < 50%.
    """

    field_name: str
    present_count: int
    total_count: int
    presence_pct: float
    is_coverage_gap: bool


@dataclass(frozen=True)
class BackfillCoverageReport:
    """Complete backfill coverage report.

    Attributes:
        date_range: The date range assessed.
        total_candidates: Total number of candidates in the backfill set.
        exact_count: Number graded "exact".
        partial_count: Number graded "partial".
        unscorable_count: Number graded "unscorable".
        field_coverage: Per-field presence statistics.
        coverage_gaps: Fields with < 50% presence.
        grades: Individual grades per candidate.
        era_breakdown: Count of candidates per era.
    """

    date_range: tuple[datetime, datetime]
    total_candidates: int
    exact_count: int
    partial_count: int
    unscorable_count: int
    field_coverage: tuple[FieldCoverage, ...]
    coverage_gaps: tuple[str, ...]
    grades: tuple[BackfillGrade, ...]
    era_breakdown: dict[str, int]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def grade_historical_record(
    candidate: ReplayCandidate,
    inputs: ReplayInputBundle,
) -> BackfillGrade:
    """Grade a single historical record's replay readiness.

    Grading logic (Requirement 12.2):
    - exact: all Decision_Snapshot fields present with timestamp-correct values
    - partial: sufficient geometry/timing but non-critical inputs missing
    - unscorable: critical inputs cannot be determined

    The era is determined by whether a snapshot_id exists for the candidate
    (Requirement 12.6):
    - "post-snapshot" if the input bundle was sourced from an immutable snapshot
    - "historical" if reconstructed from pre-snapshot era data

    Missing fields are explicitly marked rather than inferred (Requirement 12.4).
    """
    # Determine era based on snapshot availability
    era = "post-snapshot" if inputs.snapshot_id else "historical"

    # Assess field presence from the input bundle
    present_fields: list[str] = []
    missing_fields: list[str] = []

    for field_name in DECISION_SNAPSHOT_FIELDS:
        source = inputs.inputs.get(field_name)
        if source is not None and source.status == "available" and source.value is not None:
            present_fields.append(field_name)
        else:
            missing_fields.append(field_name)

    # Determine which missing fields are critical
    active_gates = CORE_GATE_SEQUENCE
    critical_fields = _get_critical_fields_for_gates(active_gates)
    critical_missing = [f for f in missing_fields if f in critical_fields]

    # Classification follows input_reconstructor's classification
    # but we also respect the grading from the input bundle itself
    grade = inputs.classification

    return BackfillGrade(
        candidate_id=candidate.candidate_id,
        grade=grade,
        era=era,
        present_fields=tuple(present_fields),
        missing_fields=tuple(missing_fields),
        critical_missing=tuple(critical_missing),
    )


def assess_backfill_coverage(
    session,
    date_range: tuple[datetime, datetime],
    *,
    policy_version: PolicyVersion | None = None,
    filters: dict | None = None,
) -> BackfillCoverageReport:
    """Assess backfill coverage across all historical data sources.

    Scans trade_events, blocked_trade_candidates, shadow_outcomes, trades,
    cases, and analyst signal memory to reconstruct candidates and grade
    their replay readiness.

    Args:
        session: SQLAlchemy session.
        date_range: (start, end) datetime range to assess.
        policy_version: Policy to evaluate critical fields against.
            Defaults to current policy if None.
        filters: Optional filters (profile, symbol, setup_type, etc.).

    Returns:
        BackfillCoverageReport with per-field presence % and coverage gaps.
    """
    if policy_version is None:
        try:
            policy_version = build_current_policy_version()
        except Exception:
            # Fallback: use a minimal policy version for testing
            policy_version = _build_fallback_policy_version()

    # Load and deduplicate candidates from all source tables
    candidates = load_candidates(session, date_range=date_range, filters=filters)
    candidates = correlate_and_deduplicate(candidates)

    # Grade each candidate
    grades: list[BackfillGrade] = []
    for candidate in candidates:
        try:
            inputs = reconstruct_inputs(session, candidate, policy_version)
            grade = grade_historical_record(candidate, inputs)
            grades.append(grade)
        except Exception as e:
            log.warning(
                "Failed to reconstruct inputs for candidate %s: %s",
                candidate.candidate_id,
                str(e),
            )
            # Mark as unscorable when reconstruction fails
            grades.append(
                BackfillGrade(
                    candidate_id=candidate.candidate_id,
                    grade="unscorable",
                    era="historical",
                    present_fields=(),
                    missing_fields=tuple(DECISION_SNAPSHOT_FIELDS),
                    critical_missing=tuple(
                        _get_critical_fields_for_gates(CORE_GATE_SEQUENCE)
                    ),
                )
            )

    # Build the coverage report
    return _build_coverage_report(date_range, grades)


def generate_coverage_report(
    session,
    date_range: tuple[datetime, datetime],
    *,
    policy_version: PolicyVersion | None = None,
    filters: dict | None = None,
) -> BackfillCoverageReport:
    """Generate a backfill coverage report: per-field presence %, coverage gaps.

    This is the primary reporting entrypoint. Delegates to assess_backfill_coverage
    and returns the complete report.

    Args:
        session: SQLAlchemy session.
        date_range: (start, end) datetime range to report on.
        policy_version: Policy to evaluate against. Defaults to current.
        filters: Optional filters.

    Returns:
        BackfillCoverageReport with:
        - Per-field presence percentage
        - Fields flagged as coverage gaps (< 50% presence)
        - Grade breakdown (exact/partial/unscorable)
        - Era breakdown (historical/post-snapshot)
    """
    return assess_backfill_coverage(
        session,
        date_range,
        policy_version=policy_version,
        filters=filters,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_critical_fields_for_gates(active_gates: list[str]) -> set[str]:
    """Get all fields consumed by the active gate set (Critical_Inputs).

    A field is critical when the selected gate/policy consumes it and its
    absence could change the replay decision (Requirement 2.7).
    """
    critical: set[str] = set()
    for gate_name in active_gates:
        fields = GATE_REQUIRED_FIELDS.get(gate_name, [])
        critical.update(fields)
    return critical


def _build_coverage_report(
    date_range: tuple[datetime, datetime],
    grades: list[BackfillGrade],
) -> BackfillCoverageReport:
    """Build a BackfillCoverageReport from individual grades.

    Computes per-field presence percentages and flags coverage gaps (< 50%).
    """
    total = len(grades)
    exact_count = sum(1 for g in grades if g.grade == "exact")
    partial_count = sum(1 for g in grades if g.grade == "partial")
    unscorable_count = sum(1 for g in grades if g.grade == "unscorable")

    # Era breakdown
    era_breakdown: dict[str, int] = {"historical": 0, "post-snapshot": 0}
    for g in grades:
        era_breakdown[g.era] = era_breakdown.get(g.era, 0) + 1

    # Per-field presence
    field_coverage_list: list[FieldCoverage] = []
    coverage_gaps: list[str] = []

    for field_name in DECISION_SNAPSHOT_FIELDS:
        present_count = sum(
            1 for g in grades if field_name in g.present_fields
        )
        presence_pct = (present_count / total * 100.0) if total > 0 else 0.0
        is_gap = presence_pct < 50.0

        fc = FieldCoverage(
            field_name=field_name,
            present_count=present_count,
            total_count=total,
            presence_pct=round(presence_pct, 2),
            is_coverage_gap=is_gap,
        )
        field_coverage_list.append(fc)

        if is_gap:
            coverage_gaps.append(field_name)

    return BackfillCoverageReport(
        date_range=date_range,
        total_candidates=total,
        exact_count=exact_count,
        partial_count=partial_count,
        unscorable_count=unscorable_count,
        field_coverage=tuple(field_coverage_list),
        coverage_gaps=tuple(coverage_gaps),
        grades=tuple(grades),
        era_breakdown=era_breakdown,
    )


def _build_fallback_policy_version() -> PolicyVersion:
    """Build a minimal PolicyVersion for testing when current cannot be built."""
    return PolicyVersion(
        name="fallback",
        gate_revision="unknown",
        config_digest="unknown",
        feature_flags={},
        benchmark_version=None,
        config_source_timestamp=None,
        gate_ordering_version="v1.0",
        adapter_version="1.0.0",
    )
