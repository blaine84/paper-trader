"""
CEO and Reviewer integration module for the Decision Replay Agent.

Formats replay findings for consumption by the CEO agent and reviewer agent:

- Daily CEO input: defects and Decision_Deltas exceeding materiality threshold,
  labeled with replay confidence and candidate count.
- Weekly CEO input: gate-effectiveness summaries meeting minimum sample size
  with exact-replay coverage percentage.
- Reviewer access: reference replay results by replay_id and candidate audit_id
  without mutating original records.
- Repeated-omission surfacing: ≥3 same metadata omission in rolling 7-day
  window → engineering defect with affected count, first occurrence, field.
- Representative examples: 1–3 replay examples per finding, linked to audit IDs.
- Confidence labeling: all results labeled exact/partial/unscorable; partial
  separated from exact structurally.

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import text

from core.replay.aggregator import (
    GateEffectivenessMetrics,
    RepeatedOmission,
    aggregate_by_group,
    detect_repeated_omissions,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

# Materiality threshold for estimated_return_impact_pct (Requirement 11.1)
DEFAULT_MATERIALITY_THRESHOLD_PCT = 0.5

# Minimum sample size for gate-effectiveness to be included in weekly CEO input
DEFAULT_MIN_SAMPLE_SIZE = 30

# Maximum representative examples per finding (Requirement 11.5)
MAX_EXAMPLES_PER_FINDING = 3
MIN_EXAMPLES_PER_FINDING = 1


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayExample:
    """A representative replay example linked to its audit identifier.

    Requirement 11.5: Each example is linked to its original audit
    identifier and labeled with replay confidence.
    """

    replay_id: str
    candidate_id: str
    symbol: str
    setup_type: str | None
    profile: str
    delta_classification: str
    divergence_cause: str | None
    replay_confidence: str  # "exact" | "partial" | "unscorable"
    summary: str  # brief human-readable description


@dataclass(frozen=True)
class CEOFinding:
    """A single finding for inclusion in CEO input.

    Requirement 11.6: Each result is labeled with replay confidence
    (exact/partial/unscorable). Partial results are structurally
    separated from exact results.
    """

    finding_type: str  # "defect" | "delta" | "repeated_omission"
    title: str
    description: str
    replay_confidence: str  # "exact" | "partial"
    affected_candidate_count: int
    estimated_return_impact_pct: float | None
    divergence_cause: str | None
    examples_exact: list[ReplayExample]
    examples_partial: list[ReplayExample]


@dataclass(frozen=True)
class DailyCEOInput:
    """Structured daily CEO input from replay findings.

    Requirement 11.1: Include defects and Decision_Deltas exceeding
    materiality threshold, each labeled with replay confidence and
    affected candidate count.
    """

    date: datetime
    findings: list[CEOFinding]
    total_candidates_evaluated: int
    exact_coverage_count: int
    partial_coverage_count: int
    unscorable_count: int
    materiality_threshold_pct: float


@dataclass(frozen=True)
class WeeklyCEOInput:
    """Structured weekly CEO input with gate-effectiveness summaries.

    Requirement 11.2: Include gate-effectiveness summaries meeting
    minimum sample size with exact-replay coverage percentage and
    sample size stated.
    """

    week_start: datetime
    week_end: datetime
    gate_summaries: list[dict]  # list of gate effectiveness summary dicts
    total_candidates_evaluated: int
    exact_coverage_pct: float
    meets_sample_size: bool
    min_sample_size: int


@dataclass(frozen=True)
class ReviewerReplayResult:
    """Read-only replay result for reviewer reference.

    Requirement 11.3: Reviewer can reference replay results by replay_id
    and candidate audit_id without mutating original records.
    """

    replay_id: str
    candidate_id: str
    replay_status: str  # "exact" | "partial" | "unscorable" | "failed"
    replay_cutoff: datetime
    policy_version: dict
    gate_trace: list[dict] | None
    decision_delta_classification: str | None
    decision_delta: dict | None
    counterfactual_outcome: dict | None
    divergence_cause: str | None
    divergence_evidence: dict | None
    era: str
    created_at: datetime


@dataclass(frozen=True)
class EngineeringDefect:
    """A repeated-omission engineering defect for surfacing.

    Requirement 11.4: ≥3 same metadata omission in rolling 7-day window
    surfaced as engineering defect with affected count, first occurrence
    date, and the specific field or rule involved.
    """

    field_name: str
    gate_name: str
    affected_candidate_count: int
    first_occurrence_date: datetime
    last_occurrence_date: datetime
    affected_candidate_ids: list[str]
    examples: list[ReplayExample]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_daily_ceo_input(
    session,
    date: datetime,
    *,
    materiality_threshold_pct: float = DEFAULT_MATERIALITY_THRESHOLD_PCT,
) -> DailyCEOInput:
    """Build structured findings for daily CEO input.

    Includes:
    - Replay-detected defects (metadata_wiring_defect, code_defect)
    - Decision_Deltas with estimated_return_impact_pct exceeding threshold
    - Repeated omissions surfaced as engineering defects

    Each finding is labeled with replay confidence (exact/partial) and
    affected candidate count.

    Requirement 11.1: WHEN the daily CEO input is assembled AND at least
    one replay-detected defect or Decision_Delta with
    estimated_return_impact_pct exceeding the configured materiality
    threshold exists, include those defects and deltas.

    Args:
        session: SQLAlchemy session for querying replay namespace.
        date: The date for which to build the CEO input (UTC).
        materiality_threshold_pct: Minimum estimated_return_impact_pct
            to include a delta finding. Default 0.5%.

    Returns:
        DailyCEOInput with all qualifying findings.
    """
    day_start = datetime(date.year, date.month, date.day)
    day_end = day_start + timedelta(days=1)

    # Load replay audit records for the day
    records = _load_audit_records_for_range(session, day_start, day_end)

    if not records:
        return DailyCEOInput(
            date=date,
            findings=[],
            total_candidates_evaluated=0,
            exact_coverage_count=0,
            partial_coverage_count=0,
            unscorable_count=0,
            materiality_threshold_pct=materiality_threshold_pct,
        )

    # Coverage statistics
    exact_count = sum(1 for r in records if r["replay_status"] == "exact")
    partial_count = sum(1 for r in records if r["replay_status"] == "partial")
    unscorable_count = sum(1 for r in records if r["replay_status"] == "unscorable")

    findings: list[CEOFinding] = []

    # 1. Surface defects (metadata_wiring_defect, code_defect)
    defect_records = [
        r for r in records
        if r.get("divergence_cause") in ("metadata_wiring_defect", "code_defect")
        and r["replay_status"] in ("exact", "partial")
    ]
    if defect_records:
        defect_finding = _build_defect_finding(defect_records)
        findings.append(defect_finding)

    # 2. Surface Decision_Deltas exceeding materiality threshold
    delta_records = [
        r for r in records
        if r.get("estimated_return_impact_pct") is not None
        and abs(r["estimated_return_impact_pct"]) >= materiality_threshold_pct
        and r["replay_status"] in ("exact", "partial")
    ]
    if delta_records:
        delta_finding = _build_delta_finding(
            delta_records, materiality_threshold_pct
        )
        findings.append(delta_finding)

    # 3. Surface repeated omissions as engineering defects
    # Use a 7-day lookback window ending on the given date
    lookback_start = day_start - timedelta(days=7)
    lookback_records = _load_audit_records_for_range(
        session, lookback_start, day_end
    )
    omission_records = _prepare_omission_records(lookback_records)
    repeated_omissions = detect_repeated_omissions(
        omission_records, window_days=7, threshold=3
    )

    for omission in repeated_omissions:
        omission_finding = _build_omission_finding(omission, lookback_records)
        findings.append(omission_finding)

    return DailyCEOInput(
        date=date,
        findings=findings,
        total_candidates_evaluated=len(records),
        exact_coverage_count=exact_count,
        partial_coverage_count=partial_count,
        unscorable_count=unscorable_count,
        materiality_threshold_pct=materiality_threshold_pct,
    )


def build_weekly_ceo_input(
    session,
    week_start: datetime,
    *,
    min_sample_size: int = DEFAULT_MIN_SAMPLE_SIZE,
) -> WeeklyCEOInput:
    """Build gate-effectiveness summaries for weekly CEO input.

    Includes gate-effectiveness summaries that meet the configured minimum
    sample size, with exact-replay coverage percentage and sample size stated.

    Requirement 11.2: WHEN the weekly CEO input is assembled AND the
    gate-effectiveness summary meets the configured minimum sample size,
    include gate-effectiveness summaries with coverage % and sample size.

    Args:
        session: SQLAlchemy session for querying replay namespace.
        week_start: Start date of the week (Monday, UTC).
        min_sample_size: Minimum sample size for inclusion. Default 30.

    Returns:
        WeeklyCEOInput with qualifying gate-effectiveness summaries.
    """
    week_end = week_start + timedelta(days=7)

    # Load replay audit records for the week
    records = _load_audit_records_for_range(session, week_start, week_end)

    if not records:
        return WeeklyCEOInput(
            week_start=week_start,
            week_end=week_end,
            gate_summaries=[],
            total_candidates_evaluated=0,
            exact_coverage_pct=0.0,
            meets_sample_size=False,
            min_sample_size=min_sample_size,
        )

    total_evaluated = len(records)
    exact_count = sum(1 for r in records if r["replay_status"] == "exact")
    exact_coverage_pct = (exact_count / total_evaluated * 100) if total_evaluated > 0 else 0.0

    # Aggregate gate effectiveness by gate
    metrics_list = aggregate_by_group(
        records,
        group_by=["gate"],
        min_sample_size=min_sample_size,
    )

    # Only include groups meeting minimum sample size
    qualifying_summaries = []
    for metrics in metrics_list:
        if metrics.meets_sample_size:
            summary = _format_gate_effectiveness_summary(
                metrics, exact_coverage_pct
            )
            qualifying_summaries.append(summary)

    meets_overall = total_evaluated >= min_sample_size

    return WeeklyCEOInput(
        week_start=week_start,
        week_end=week_end,
        gate_summaries=qualifying_summaries,
        total_candidates_evaluated=total_evaluated,
        exact_coverage_pct=exact_coverage_pct,
        meets_sample_size=meets_overall,
        min_sample_size=min_sample_size,
    )


def get_replay_for_reviewer(
    session,
    replay_id: str,
) -> ReviewerReplayResult | None:
    """Retrieve a read-only replay result by replay_id for reviewer reference.

    Returns the replay audit record without mutating any original trade
    outcome, shadow outcome, or case record.

    Requirement 11.3: Reviewer SHALL be able to reference replay results
    by replay identifier and candidate audit identifier without mutating
    the original records.

    Args:
        session: SQLAlchemy session for querying replay namespace.
        replay_id: The unique replay identifier.

    Returns:
        ReviewerReplayResult if found, None otherwise.
    """
    result = session.execute(
        text("""
            SELECT replay_id, candidate_id, replay_status, replay_cutoff,
                   policy_version_json, gate_trace_json,
                   decision_delta_classification, decision_delta_json,
                   counterfactual_outcome_json, divergence_cause,
                   divergence_evidence_json, era, created_at
            FROM replay_audit_records
            WHERE replay_id = :replay_id
        """),
        {"replay_id": replay_id},
    )
    row = result.fetchone()
    if row is None:
        return None

    return _row_to_reviewer_result(row)


def get_replay_by_candidate_id(
    session,
    candidate_id: str,
) -> list[ReviewerReplayResult]:
    """Retrieve read-only replay results by candidate audit ID.

    Returns all replay audit records for a given candidate without
    mutating any original records.

    Requirement 11.3: Reviewer SHALL be able to reference replay results
    by candidate audit identifier.

    Args:
        session: SQLAlchemy session for querying replay namespace.
        candidate_id: The candidate identifier.

    Returns:
        List of ReviewerReplayResult for the candidate (may be empty).
    """
    result = session.execute(
        text("""
            SELECT replay_id, candidate_id, replay_status, replay_cutoff,
                   policy_version_json, gate_trace_json,
                   decision_delta_classification, decision_delta_json,
                   counterfactual_outcome_json, divergence_cause,
                   divergence_evidence_json, era, created_at
            FROM replay_audit_records
            WHERE candidate_id = :candidate_id
            ORDER BY created_at DESC
        """),
        {"candidate_id": candidate_id},
    )
    rows = result.fetchall()
    return [_row_to_reviewer_result(row) for row in rows]


def format_findings_with_examples(
    findings: list[CEOFinding],
    max_examples: int = MAX_EXAMPLES_PER_FINDING,
) -> list[dict]:
    """Format findings with representative examples for presentation.

    Each finding includes 1–3 representative replay examples, linked to
    audit identifiers and labeled with replay confidence.

    Requirement 11.5: Include at least 1 and at most 3 representative
    replay examples per surfaced finding.

    Requirement 11.6: Label each result with replay confidence and
    structurally separate partial from exact results.

    Args:
        findings: List of CEOFinding objects to format.
        max_examples: Maximum examples per finding (capped at 3).

    Returns:
        List of formatted finding dicts with examples.
    """
    max_examples = min(max_examples, MAX_EXAMPLES_PER_FINDING)

    formatted = []
    for finding in findings:
        # Structurally separate exact and partial examples (Requirement 11.6)
        exact_examples = finding.examples_exact[:max_examples]
        partial_examples = finding.examples_partial[:max_examples]

        # Ensure at least 1 example total if any exist
        total_examples = len(exact_examples) + len(partial_examples)
        if total_examples == 0 and finding.affected_candidate_count > 0:
            # No examples available — still include the finding
            pass

        formatted_finding = {
            "finding_type": finding.finding_type,
            "title": finding.title,
            "description": finding.description,
            "replay_confidence": finding.replay_confidence,
            "affected_candidate_count": finding.affected_candidate_count,
            "estimated_return_impact_pct": finding.estimated_return_impact_pct,
            "divergence_cause": finding.divergence_cause,
            "examples": {
                "exact": [
                    _format_example(ex) for ex in exact_examples
                ],
                "partial": [
                    _format_example(ex) for ex in partial_examples
                ],
            },
        }
        formatted.append(formatted_finding)

    return formatted


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_audit_records_for_range(
    session,
    start: datetime,
    end: datetime,
) -> list[dict]:
    """Load replay audit records within a date range.

    Returns dicts with all fields needed for CEO/reviewer reporting.
    """
    result = session.execute(
        text("""
            SELECT r.replay_id, r.candidate_id, r.replay_status,
                   r.replay_cutoff, r.policy_version_json, r.gate_trace_json,
                   r.decision_delta_classification, r.decision_delta_json,
                   r.counterfactual_outcome_json, r.divergence_cause,
                   r.divergence_evidence_json, r.era, r.created_at,
                   r.source_candidate_ids_json
            FROM replay_audit_records r
            WHERE r.created_at >= :start_dt
              AND r.created_at < :end_dt
              AND r.replay_status != 'failed'
            ORDER BY r.created_at
        """),
        {"start_dt": start, "end_dt": end},
    )
    rows = result.fetchall()

    records = []
    for row in rows:
        record = {
            "replay_id": row[0],
            "candidate_id": row[1],
            "replay_status": row[2],
            "replay_cutoff": _parse_datetime(row[3]),
            "policy_version_json": row[4],
            "gate_trace_json": row[5],
            "decision_delta_classification": row[6],
            "decision_delta_json": row[7],
            "counterfactual_outcome_json": row[8],
            "divergence_cause": row[9],
            "divergence_evidence_json": row[10],
            "era": row[11],
            "created_at": _parse_datetime(row[12]),
            "source_candidate_ids_json": row[13],
        }

        # Parse counterfactual outcome for return impact
        if record["counterfactual_outcome_json"]:
            try:
                outcome = json.loads(record["counterfactual_outcome_json"])
                # Use 60-min return as the materiality metric
                record["estimated_return_impact_pct"] = outcome.get("return_60m")
            except (json.JSONDecodeError, TypeError):
                record["estimated_return_impact_pct"] = None
        else:
            record["estimated_return_impact_pct"] = None

        # Parse gate trace for missing fields (needed for omission detection)
        if record["gate_trace_json"]:
            try:
                trace = json.loads(record["gate_trace_json"])
                missing_fields = []
                gate_name = None
                if isinstance(trace, list):
                    for entry in trace:
                        if entry.get("missing_fields"):
                            missing_fields.extend(entry["missing_fields"])
                            if not gate_name:
                                gate_name = entry.get("gate_name")
                elif isinstance(trace, dict) and trace.get("entries"):
                    for entry in trace["entries"]:
                        if entry.get("missing_fields"):
                            missing_fields.extend(entry["missing_fields"])
                            if not gate_name:
                                gate_name = entry.get("gate_name")
                record["missing_fields"] = missing_fields
                record["gate_name"] = gate_name
            except (json.JSONDecodeError, TypeError):
                record["missing_fields"] = []
                record["gate_name"] = None
        else:
            record["missing_fields"] = []
            record["gate_name"] = None

        # Parse entry_timestamp from replay_cutoff
        record["entry_timestamp"] = record["replay_cutoff"]

        records.append(record)

    return records


def _prepare_omission_records(records: list[dict]) -> list[dict]:
    """Prepare records for detect_repeated_omissions().

    The aggregator's detect_repeated_omissions() expects records with:
    - candidate_id
    - gate_name
    - missing_fields: list of field names
    - entry_timestamp: datetime
    """
    prepared = []
    for record in records:
        if not record.get("missing_fields"):
            continue

        # If gate trace has per-gate missing fields, expand them
        gate_trace_json = record.get("gate_trace_json")
        if gate_trace_json:
            try:
                trace = json.loads(gate_trace_json) if isinstance(gate_trace_json, str) else gate_trace_json
                entries = trace if isinstance(trace, list) else trace.get("entries", [])
                for entry in entries:
                    gate_missing = entry.get("missing_fields", [])
                    if gate_missing:
                        prepared.append({
                            "candidate_id": record["candidate_id"],
                            "gate_name": entry.get("gate_name", "unknown"),
                            "missing_fields": gate_missing,
                            "entry_timestamp": record["entry_timestamp"],
                        })
            except (json.JSONDecodeError, TypeError):
                # Fallback: use top-level missing_fields
                prepared.append({
                    "candidate_id": record["candidate_id"],
                    "gate_name": record.get("gate_name", "unknown"),
                    "missing_fields": record["missing_fields"],
                    "entry_timestamp": record["entry_timestamp"],
                })
        else:
            prepared.append({
                "candidate_id": record["candidate_id"],
                "gate_name": record.get("gate_name", "unknown"),
                "missing_fields": record["missing_fields"],
                "entry_timestamp": record["entry_timestamp"],
            })

    return prepared


def _build_defect_finding(defect_records: list[dict]) -> CEOFinding:
    """Build a CEOFinding for detected defects."""
    # Group by divergence cause for description
    causes = {}
    for r in defect_records:
        cause = r.get("divergence_cause", "unknown")
        causes.setdefault(cause, []).append(r)

    cause_summary = ", ".join(
        f"{cause}: {len(recs)} candidate(s)"
        for cause, recs in causes.items()
    )

    # Separate exact and partial examples
    exact_records = [r for r in defect_records if r["replay_status"] == "exact"]
    partial_records = [r for r in defect_records if r["replay_status"] == "partial"]

    examples_exact = _select_representative_examples(exact_records)
    examples_partial = _select_representative_examples(partial_records)

    # Determine overall confidence label
    confidence = "exact" if exact_records else "partial"

    return CEOFinding(
        finding_type="defect",
        title="Replay-detected defects",
        description=f"Defects detected in replay: {cause_summary}",
        replay_confidence=confidence,
        affected_candidate_count=len(defect_records),
        estimated_return_impact_pct=_compute_aggregate_impact(defect_records),
        divergence_cause=list(causes.keys())[0] if len(causes) == 1 else "multiple",
        examples_exact=examples_exact,
        examples_partial=examples_partial,
    )


def _build_delta_finding(
    delta_records: list[dict],
    materiality_threshold: float,
) -> CEOFinding:
    """Build a CEOFinding for material Decision_Deltas."""
    # Separate exact and partial examples
    exact_records = [r for r in delta_records if r["replay_status"] == "exact"]
    partial_records = [r for r in delta_records if r["replay_status"] == "partial"]

    examples_exact = _select_representative_examples(exact_records)
    examples_partial = _select_representative_examples(partial_records)

    confidence = "exact" if exact_records else "partial"
    aggregate_impact = _compute_aggregate_impact(delta_records)

    # Group by delta classification for description
    classifications = {}
    for r in delta_records:
        cls = r.get("decision_delta_classification", "unknown")
        classifications.setdefault(cls, []).append(r)

    cls_summary = ", ".join(
        f"{cls}: {len(recs)}" for cls, recs in classifications.items()
    )

    return CEOFinding(
        finding_type="delta",
        title=f"Material Decision_Deltas (≥{materiality_threshold}% impact)",
        description=f"Decision deltas exceeding materiality threshold: {cls_summary}",
        replay_confidence=confidence,
        affected_candidate_count=len(delta_records),
        estimated_return_impact_pct=aggregate_impact,
        divergence_cause=None,
        examples_exact=examples_exact,
        examples_partial=examples_partial,
    )


def _build_omission_finding(
    omission: RepeatedOmission,
    all_records: list[dict],
) -> CEOFinding:
    """Build a CEOFinding from a RepeatedOmission engineering defect.

    Requirement 11.4: Surface as engineering defect with affected
    candidate count, first occurrence date, and specific field.
    """
    # Find matching records to build examples
    matching_records = [
        r for r in all_records
        if r["candidate_id"] in omission.affected_candidate_ids
    ]

    exact_records = [r for r in matching_records if r.get("replay_status") == "exact"]
    partial_records = [r for r in matching_records if r.get("replay_status") == "partial"]

    examples_exact = _select_representative_examples(exact_records)
    examples_partial = _select_representative_examples(partial_records)

    confidence = "exact" if exact_records else "partial"

    return CEOFinding(
        finding_type="repeated_omission",
        title=f"Repeated omission: {omission.field_name} at {omission.gate_name}",
        description=(
            f"Field '{omission.field_name}' missing at gate '{omission.gate_name}' "
            f"in {omission.occurrence_count} candidates within rolling 7-day window. "
            f"First occurrence: {omission.first_occurrence.isoformat()}. "
            f"Classified as engineering defect ({omission.defect_type})."
        ),
        replay_confidence=confidence,
        affected_candidate_count=omission.occurrence_count,
        estimated_return_impact_pct=_compute_aggregate_impact(matching_records),
        divergence_cause=omission.defect_type,
        examples_exact=examples_exact,
        examples_partial=examples_partial,
    )


def _select_representative_examples(
    records: list[dict],
    max_count: int = MAX_EXAMPLES_PER_FINDING,
) -> list[ReplayExample]:
    """Select 1–3 representative examples from a set of records.

    Requirement 11.5: At least 1 and at most 3 representative replay
    examples per surfaced finding.

    Selection strategy:
    - Prefer diverse symbols/setup_types for representativeness
    - Cap at max_count (default 3)
    - Minimum 1 if any records exist
    """
    if not records:
        return []

    # Select diverse examples by (symbol, setup_type) if possible
    seen_keys: set[tuple] = set()
    selected: list[dict] = []

    for record in records:
        key = (
            record.get("candidate_id", ""),
            record.get("divergence_cause", ""),
        )
        if key not in seen_keys:
            seen_keys.add(key)
            selected.append(record)
            if len(selected) >= max_count:
                break

    # If we have fewer than max due to diversity, just take first N
    if not selected and records:
        selected = records[:max_count]

    examples = []
    for record in selected:
        # Parse source candidate info for symbol/profile/setup_type
        symbol, profile, setup_type = _extract_candidate_metadata(record)

        examples.append(
            ReplayExample(
                replay_id=record.get("replay_id", ""),
                candidate_id=record.get("candidate_id", ""),
                symbol=symbol,
                setup_type=setup_type,
                profile=profile,
                delta_classification=record.get("decision_delta_classification", "unknown"),
                divergence_cause=record.get("divergence_cause"),
                replay_confidence=record.get("replay_status", "unknown"),
                summary=_build_example_summary(record),
            )
        )

    return examples


def _format_gate_effectiveness_summary(
    metrics: GateEffectivenessMetrics,
    exact_coverage_pct: float,
) -> dict:
    """Format a GateEffectivenessMetrics into a summary dict for CEO.

    Requirement 11.2: Include coverage percentage and sample size.
    """
    return {
        "gate_name": metrics.gate_name,
        "profile": metrics.profile,
        "setup_type": metrics.setup_type,
        "policy_version": metrics.policy_version,
        "candidates_evaluated": metrics.candidates_evaluated,
        "exact_count": metrics.exact_count,
        "partial_count": metrics.partial_count,
        "unscorable_count": metrics.unscorable_count,
        "exact_coverage_pct": (
            (metrics.exact_count / metrics.candidates_evaluated * 100)
            if metrics.candidates_evaluated > 0
            else 0.0
        ),
        "overall_exact_coverage_pct": exact_coverage_pct,
        "blocked_winners": metrics.blocked_winners,
        "correctly_blocked_losers": metrics.correctly_blocked_losers,
        "false_allows": metrics.false_allows,
        "estimated_return_impact_pct": metrics.estimated_return_impact_pct,
        "stop_first_rate": metrics.stop_first_rate,
        "target_first_rate": metrics.target_first_rate,
        "near_boundary_count": metrics.near_boundary_count,
        "meets_sample_size": metrics.meets_sample_size,
        "sample_size": metrics.candidates_evaluated,
        "directional_conclusion": metrics.directional_conclusion,
        "era": metrics.era,
    }


def _row_to_reviewer_result(row) -> ReviewerReplayResult:
    """Convert a database row to ReviewerReplayResult."""
    return ReviewerReplayResult(
        replay_id=row[0],
        candidate_id=row[1],
        replay_status=row[2],
        replay_cutoff=_parse_datetime(row[3]),
        policy_version=_safe_json_loads(row[4]) or {},
        gate_trace=_parse_gate_trace(row[5]),
        decision_delta_classification=row[6],
        decision_delta=_safe_json_loads(row[7]),
        counterfactual_outcome=_safe_json_loads(row[8]),
        divergence_cause=row[9],
        divergence_evidence=_safe_json_loads(row[10]),
        era=row[11],
        created_at=_parse_datetime(row[12]),
    )


def _parse_datetime(value) -> datetime:
    """Parse a datetime value that may be a string or datetime."""
    if value is None:
        return datetime.min
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return datetime.min
    return datetime.min


def _safe_json_loads(value: str | None) -> dict | list | None:
    """Safely parse a JSON string, returning None on failure."""
    if value is None:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def _parse_gate_trace(value: str | None) -> list[dict] | None:
    """Parse gate trace JSON into a list of trace entry dicts."""
    parsed = _safe_json_loads(value)
    if parsed is None:
        return None
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and "entries" in parsed:
        return parsed["entries"]
    return [parsed]


def _compute_aggregate_impact(records: list[dict]) -> float | None:
    """Compute aggregate estimated return impact from records."""
    impacts = [
        r["estimated_return_impact_pct"]
        for r in records
        if r.get("estimated_return_impact_pct") is not None
    ]
    if not impacts:
        return None
    return sum(impacts)


def _extract_candidate_metadata(record: dict) -> tuple[str, str, str | None]:
    """Extract symbol, profile, setup_type from a record.

    Attempts to parse from source_candidate_ids_json or delta_json.
    Falls back to empty strings.
    """
    # Try to extract from decision_delta_json
    delta_json = record.get("decision_delta_json")
    if delta_json:
        delta = _safe_json_loads(delta_json) if isinstance(delta_json, str) else delta_json
        if isinstance(delta, dict):
            symbol = delta.get("symbol", "")
            profile = delta.get("profile", "")
            setup_type = delta.get("setup_type")
            if symbol or profile:
                return symbol, profile, setup_type

    # Try source_candidate_ids_json
    source_json = record.get("source_candidate_ids_json")
    if source_json:
        source = _safe_json_loads(source_json) if isinstance(source_json, str) else source_json
        if isinstance(source, list) and source:
            first = source[0] if isinstance(source[0], dict) else {}
            return (
                first.get("symbol", ""),
                first.get("profile", ""),
                first.get("setup_type"),
            )

    return "", "", None


def _build_example_summary(record: dict) -> str:
    """Build a brief human-readable summary for a replay example."""
    delta_cls = record.get("decision_delta_classification", "unknown")
    cause = record.get("divergence_cause", "")
    status = record.get("replay_status", "unknown")

    parts = [f"Delta: {delta_cls}"]
    if cause:
        parts.append(f"cause: {cause}")
    parts.append(f"confidence: {status}")

    return "; ".join(parts)


def _format_example(example: ReplayExample) -> dict:
    """Format a ReplayExample for structured output."""
    return {
        "replay_id": example.replay_id,
        "candidate_id": example.candidate_id,
        "symbol": example.symbol,
        "setup_type": example.setup_type,
        "profile": example.profile,
        "delta_classification": example.delta_classification,
        "divergence_cause": example.divergence_cause,
        "replay_confidence": example.replay_confidence,
        "summary": example.summary,
    }
