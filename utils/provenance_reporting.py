"""Provenance Coverage Metrics and Reporting Helpers.

Computes lineage coverage, generates daily/weekly reports broken down by
profile, setup type, symbol class, stage, and reason code. Provides CEO-level
summary with attribution evidence.

Key rules:
- Label categories with <20 occurrences as `exploratory`
- Order CEO summary: repeated upstream defects (>=3) before threshold-tuning
- Do NOT recommend loosening a gate that blocked winners with malformed upstream
- Reconstruction analysis is report-only (no automated policy changes)
- Separate counts from economic outcomes (dollar risk, potential reward)
- Include 1-5 representative examples per finding category with lineage IDs

Requirements: 1.6, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 11.6, 11.7
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import text

logger = logging.getLogger(__name__)


# Minimum occurrences for a finding category to be non-exploratory (Req 13.5)
EXPLORATORY_THRESHOLD = 20

# Minimum occurrences for a defect to be "repeated" in CEO summary (Req 13.6)
REPEATED_DEFECT_THRESHOLD = 3

# Maximum representative examples per category (Req 13.3)
MAX_EXAMPLES_PER_CATEGORY = 5

# Expected stages by PM mode (mirrors ProvenanceChain.expected_stages)
_EXPECTED_STAGES_CANDIDATE_ID = [
    "trusted_input", "raw_pm_output", "parsed_pm_decision",
    "candidate_resolution", "behavioral_adjustment", "pre_gate_snapshot",
]
_EXPECTED_STAGES_LEGACY = [
    "trusted_input", "raw_pm_output", "parsed_pm_decision",
    "price_repair", "behavioral_adjustment", "pre_gate_snapshot",
]


@dataclass
class CoverageMetrics:
    """Lineage coverage metrics (Requirement 1.6).

    Coverage = candidates with complete stage linkage / total initiated.
    """

    total_initiated: int = 0
    complete_provenance: int = 0
    incomplete_provenance: int = 0
    coverage_pct: float = 0.0
    by_profile: dict[str, dict] = field(default_factory=dict)
    by_stage: dict[str, int] = field(default_factory=dict)


@dataclass
class EconomicOutcomes:
    """Economic outcomes separated from counts (Requirement 13.4)."""

    total_dollar_risk: Decimal = Decimal("0")
    total_potential_reward: Decimal = Decimal("0")
    count: int = 0


@dataclass
class FindingCategory:
    """A single finding category with count, examples, and exploratory label."""

    category: str
    count: int = 0
    is_exploratory: bool = False
    representative_examples: list[str] = field(default_factory=list)
    economic_outcomes: EconomicOutcomes = field(default_factory=EconomicOutcomes)


@dataclass
class ProvenanceReport:
    """Structured provenance report for a time period (Requirements 13.1-13.7)."""

    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_type: str = "daily"  # "daily" or "weekly"
    coverage: CoverageMetrics = field(default_factory=CoverageMetrics)
    total_candidates: int = 0
    malformed_at_pm_stage: int = 0
    defects_by_attribution: dict[str, FindingCategory] = field(default_factory=dict)
    policy_rejections: int = 0
    integrity_rejections: int = 0
    by_profile: dict[str, dict] = field(default_factory=dict)
    by_setup_type: dict[str, dict] = field(default_factory=dict)
    by_symbol_class: dict[str, dict] = field(default_factory=dict)
    by_stage: dict[str, dict] = field(default_factory=dict)
    by_reason_code: dict[str, int] = field(default_factory=dict)
    reconstruction_outcomes: dict[str, int] = field(default_factory=dict)
    exploratory_categories: list[str] = field(default_factory=list)



def compute_lineage_coverage(
    engine,
    *,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    profile: str | None = None,
) -> CoverageMetrics:
    """Compute lineage coverage as percentage with stage and profile breakdown.

    Coverage = candidates with complete stage linkage / total initiated.
    A lineage is "complete" if it has all expected stages recorded (based on
    the PM mode detected from its stages), OR if it terminated early via a
    terminal event (is_terminal=True) — terminal lineages are considered
    complete because no further stages are expected.

    Broken down by:
    - Stage: count of lineages missing each specific stage
    - PM profile: coverage percentage per profile

    Args:
        engine: SQLAlchemy engine for the provenance database.
        start_time: Optional start of time window (inclusive).
        end_time: Optional end of time window (exclusive).
        profile: Optional PM profile filter.

    Returns:
        CoverageMetrics with totals and breakdowns.
    """
    metrics = CoverageMetrics()

    # Build time filter clause
    time_filter = ""
    params: dict = {}
    if start_time:
        time_filter += " AND pe.timestamp >= :start_time"
        params["start_time"] = start_time.isoformat()
    if end_time:
        time_filter += " AND pe.timestamp < :end_time"
        params["end_time"] = end_time.isoformat()

    # Get all distinct lineage IDs with their stages, terminal status, and profile
    query = text(f"""
        SELECT 
            pe.lineage_id,
            pe.stage_name,
            pe.is_terminal,
            pe.input_contract_json
        FROM provenance_events pe
        WHERE 1=1 {time_filter}
        ORDER BY pe.lineage_id, pe.sequence_number
    """)

    try:
        with engine.connect() as conn:
            rows = conn.execute(query, params).fetchall()
    except Exception as exc:
        logger.error("Failed to query provenance events for coverage: %s", exc)
        return metrics

    if not rows:
        return metrics

    # Group by lineage_id
    lineages: dict[str, dict] = {}
    for row in rows:
        lineage_id = row[0]
        stage_name = row[1]
        is_terminal = bool(row[2])
        input_contract_json = row[3]

        if lineage_id not in lineages:
            lineages[lineage_id] = {
                "stages": set(),
                "is_terminal": False,
                "profile": None,
                "input_contract_json": input_contract_json,
            }

        lineages[lineage_id]["stages"].add(stage_name)
        if is_terminal:
            lineages[lineage_id]["is_terminal"] = True

        # Extract profile from first event's input_contract_json if available
        if lineages[lineage_id]["profile"] is None and input_contract_json:
            try:
                contract = json.loads(input_contract_json)
                if "profile" in contract:
                    lineages[lineage_id]["profile"] = contract["profile"]
            except (json.JSONDecodeError, TypeError):
                pass

    # Filter by profile if requested
    if profile:
        lineages = {
            lid: data for lid, data in lineages.items()
            if data["profile"] == profile
        }

    metrics.total_initiated = len(lineages)
    if metrics.total_initiated == 0:
        return metrics

    # Determine completeness for each lineage
    stage_missing_counts: dict[str, int] = {}
    profile_stats: dict[str, dict] = {}

    for lineage_id, data in lineages.items():
        stages = data["stages"]
        is_terminal = data["is_terminal"]
        lineage_profile = data["profile"] or "unknown"

        # Determine mode from stages present
        if "candidate_resolution" in stages:
            expected = set(_EXPECTED_STAGES_CANDIDATE_ID)
        elif "price_repair" in stages:
            expected = set(_EXPECTED_STAGES_LEGACY)
        else:
            # Default to candidate-ID mode expected stages
            expected = set(_EXPECTED_STAGES_CANDIDATE_ID)

        # Terminal lineages are considered complete
        if is_terminal:
            is_complete = True
        else:
            missing = expected - stages
            is_complete = len(missing) == 0

            # Track missing stage counts
            for stage in missing:
                stage_missing_counts[stage] = stage_missing_counts.get(stage, 0) + 1

        # Update totals
        if is_complete:
            metrics.complete_provenance += 1
        else:
            metrics.incomplete_provenance += 1

        # Track per-profile
        if lineage_profile not in profile_stats:
            profile_stats[lineage_profile] = {"total": 0, "complete": 0}
        profile_stats[lineage_profile]["total"] += 1
        if is_complete:
            profile_stats[lineage_profile]["complete"] += 1

    # Compute percentages
    metrics.coverage_pct = round(
        (metrics.complete_provenance / metrics.total_initiated) * 100, 2
    )
    metrics.by_stage = stage_missing_counts

    # Build per-profile breakdowns
    for prof, stats in profile_stats.items():
        pct = round((stats["complete"] / stats["total"]) * 100, 2) if stats["total"] > 0 else 0.0
        metrics.by_profile[prof] = {
            "total": stats["total"],
            "complete": stats["complete"],
            "incomplete": stats["total"] - stats["complete"],
            "coverage_pct": pct,
        }

    return metrics


def _query_report_data(
    engine,
    start_time: datetime,
    end_time: datetime,
) -> tuple[list, list]:
    """Query provenance events and findings for a reporting period.

    Returns (events_rows, findings_rows).
    """
    events_query = text("""
        SELECT 
            pe.lineage_id,
            pe.stage_name,
            pe.mutation_reason_code,
            pe.validation_before,
            pe.validation_after,
            pe.is_terminal,
            pe.input_contract_json,
            pe.output_contract_json,
            pe.geometry_before_json,
            pe.geometry_after_json
        FROM provenance_events pe
        WHERE pe.timestamp >= :start_time 
          AND pe.timestamp < :end_time
        ORDER BY pe.lineage_id, pe.sequence_number
    """)

    findings_query = text("""
        SELECT 
            pf.lineage_id,
            pf.finding_type,
            pf.stage_name,
            pf.severity,
            pf.details_json
        FROM provenance_findings pf
        WHERE pf.created_at >= :start_time 
          AND pf.created_at < :end_time
    """)

    params = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
    }

    events_rows = []
    findings_rows = []

    try:
        with engine.connect() as conn:
            events_rows = conn.execute(events_query, params).fetchall()
            findings_rows = conn.execute(findings_query, params).fetchall()
    except Exception as exc:
        logger.error("Failed to query report data: %s", exc)

    return events_rows, findings_rows


def _extract_contract_field(input_contract_json: str | None, field_name: str) -> str | None:
    """Safely extract a field from input_contract_json."""
    if not input_contract_json:
        return None
    try:
        contract = json.loads(input_contract_json)
        return contract.get(field_name)
    except (json.JSONDecodeError, TypeError):
        return None


def _extract_economic_data(geometry_json: str | None) -> tuple[Decimal, Decimal]:
    """Extract dollar_risk and potential reward from geometry JSON.

    Returns (total_dollar_risk, potential_reward).
    potential_reward = reward_distance * quantity.
    """
    if not geometry_json:
        return Decimal("0"), Decimal("0")
    try:
        geo = json.loads(geometry_json)
        dollar_risk = Decimal(str(geo.get("total_dollar_risk", "0") or "0"))
        reward_dist = Decimal(str(geo.get("reward_distance", "0") or "0"))
        quantity = Decimal(str(geo.get("quantity", "0") or "0"))
        potential_reward = reward_dist * quantity
        return dollar_risk, potential_reward
    except (json.JSONDecodeError, TypeError, Exception):
        return Decimal("0"), Decimal("0")


def _classify_symbol_class(symbol: str | None) -> str:
    """Classify symbol into asset class for reporting breakdown."""
    if not symbol:
        return "unknown"
    try:
        from utils.symbol_class import classify_symbol
        result = classify_symbol(symbol)
        # For reporting purposes, "unknown" from classify_symbol means single stock
        return result if result != "unknown" else "single_stock"
    except ImportError:
        return "unknown"


def _build_report_from_data(
    events_rows: list,
    findings_rows: list,
    start_time: datetime,
    end_time: datetime,
    period_type: str,
    engine,
) -> ProvenanceReport:
    """Build a ProvenanceReport from queried event and finding data.

    Aggregates by attribution category, profile, setup_type, symbol_class,
    stage, and reason_code. Includes representative examples and economic
    outcomes.
    """
    report = ProvenanceReport(
        period_start=start_time,
        period_end=end_time,
        period_type=period_type,
    )

    # Group events by lineage
    lineage_data: dict[str, dict] = {}
    for row in events_rows:
        lineage_id = row[0]
        stage_name = row[1]
        reason_code = row[2]
        validation_before = row[3]
        validation_after = row[4]
        is_terminal = bool(row[5])
        input_contract_json = row[6]
        output_contract_json = row[7]
        geometry_before_json = row[8]
        geometry_after_json = row[9]

        if lineage_id not in lineage_data:
            lineage_data[lineage_id] = {
                "stages": [],
                "profile": None,
                "symbol": None,
                "setup_type": None,
                "is_terminal": False,
                "has_invalid": False,
                "first_invalid_stage": None,
                "reason_codes": [],
                "geometry_after_json": geometry_after_json,
            }

        ld = lineage_data[lineage_id]
        ld["stages"].append(stage_name)
        ld["reason_codes"].append(reason_code)

        if is_terminal:
            ld["is_terminal"] = True

        if validation_after == "invalid" and ld["first_invalid_stage"] is None:
            ld["first_invalid_stage"] = stage_name

        if validation_after == "invalid":
            ld["has_invalid"] = True

        # Extract metadata from input contract
        if ld["profile"] is None:
            ld["profile"] = _extract_contract_field(input_contract_json, "profile")
        if ld["symbol"] is None:
            ld["symbol"] = _extract_contract_field(input_contract_json, "symbol")
        if ld["setup_type"] is None:
            ld["setup_type"] = _extract_contract_field(input_contract_json, "setup_type")

        # Keep last geometry for economic outcomes
        if geometry_after_json:
            ld["geometry_after_json"] = geometry_after_json

    report.total_candidates = len(lineage_data)

    # Compute coverage for this period
    report.coverage = compute_lineage_coverage(
        engine, start_time=start_time, end_time=end_time
    )

    # Attribution mapping (stage_name -> attribution category)
    from utils.first_invalid_stage import ATTRIBUTION_CATEGORIES
    from utils.provenance_capture import STAGE_TO_ATTRIBUTION

    # Track breakdowns
    profile_counts: dict[str, dict] = {}
    setup_type_counts: dict[str, dict] = {}
    symbol_class_counts: dict[str, dict] = {}
    stage_counts: dict[str, dict] = {}
    reason_code_counts: dict[str, int] = {}
    attribution_counts: dict[str, FindingCategory] = {}
    reconstruction_outcomes: dict[str, int] = {}

    # Examples tracker: category -> list of lineage_ids
    category_examples: dict[str, list[str]] = {}

    for lineage_id, ld in lineage_data.items():
        lineage_profile = ld["profile"] or "unknown"
        lineage_symbol = ld["symbol"]
        lineage_setup_type = ld["setup_type"] or "unknown"
        symbol_class = _classify_symbol_class(lineage_symbol)

        # Determine attribution category
        first_invalid = ld["first_invalid_stage"]
        if first_invalid:
            attribution = STAGE_TO_ATTRIBUTION.get(first_invalid, "unknown")
        elif ld["has_invalid"]:
            attribution = "unknown"
        else:
            # Check if it was a policy rejection (valid geometry but rejected)
            # Look for pre_gate_contract_invalid or policy rejection reason codes
            if "pre_gate_contract_invalid" in ld["reason_codes"]:
                attribution = "pre_gate_contract_invalid"
            elif any(
                rc in ("policy_rejection", "risk_policy_rejection", "portfolio_concentration")
                for rc in ld["reason_codes"]
            ):
                attribution = "policy_rejection_of_valid_contract"
            else:
                attribution = None  # Valid, not rejected

        # Count malformed at PM stage
        if attribution in ("raw_pm_output_invalid", "parse_or_normalization_invalid"):
            report.malformed_at_pm_stage += 1

        # Count policy vs integrity rejections
        if attribution == "policy_rejection_of_valid_contract":
            report.policy_rejections += 1
        elif attribution and attribution != "policy_rejection_of_valid_contract":
            report.integrity_rejections += 1

        # Track attribution category
        if attribution:
            if attribution not in attribution_counts:
                attribution_counts[attribution] = FindingCategory(category=attribution)
            attribution_counts[attribution].count += 1

            # Add representative example (up to MAX_EXAMPLES_PER_CATEGORY)
            if attribution not in category_examples:
                category_examples[attribution] = []
            if len(category_examples[attribution]) < MAX_EXAMPLES_PER_CATEGORY:
                category_examples[attribution].append(lineage_id)

            # Extract economic outcomes
            dollar_risk, potential_reward = _extract_economic_data(
                ld["geometry_after_json"]
            )
            attribution_counts[attribution].economic_outcomes.total_dollar_risk += dollar_risk
            attribution_counts[attribution].economic_outcomes.total_potential_reward += potential_reward
            attribution_counts[attribution].economic_outcomes.count += 1

        # Profile breakdown
        if lineage_profile not in profile_counts:
            profile_counts[lineage_profile] = {
                "total": 0, "malformed_at_pm": 0,
                "policy_rejections": 0, "integrity_rejections": 0,
            }
        profile_counts[lineage_profile]["total"] += 1
        if attribution in ("raw_pm_output_invalid", "parse_or_normalization_invalid"):
            profile_counts[lineage_profile]["malformed_at_pm"] += 1
        if attribution == "policy_rejection_of_valid_contract":
            profile_counts[lineage_profile]["policy_rejections"] += 1
        elif attribution and attribution != "policy_rejection_of_valid_contract":
            profile_counts[lineage_profile]["integrity_rejections"] += 1

        # Setup type breakdown
        if lineage_setup_type not in setup_type_counts:
            setup_type_counts[lineage_setup_type] = {
                "total": 0, "defects": 0, "policy_rejections": 0,
            }
        setup_type_counts[lineage_setup_type]["total"] += 1
        if attribution and attribution != "policy_rejection_of_valid_contract":
            setup_type_counts[lineage_setup_type]["defects"] += 1
        if attribution == "policy_rejection_of_valid_contract":
            setup_type_counts[lineage_setup_type]["policy_rejections"] += 1

        # Symbol class breakdown
        if symbol_class not in symbol_class_counts:
            symbol_class_counts[symbol_class] = {
                "total": 0, "defects": 0, "policy_rejections": 0,
            }
        symbol_class_counts[symbol_class]["total"] += 1
        if attribution and attribution != "policy_rejection_of_valid_contract":
            symbol_class_counts[symbol_class]["defects"] += 1
        if attribution == "policy_rejection_of_valid_contract":
            symbol_class_counts[symbol_class]["policy_rejections"] += 1

        # Stage breakdown (which stages have issues)
        if first_invalid:
            if first_invalid not in stage_counts:
                stage_counts[first_invalid] = {"defects": 0, "total_seen": 0}
            stage_counts[first_invalid]["defects"] += 1
        for stage in ld["stages"]:
            if stage not in stage_counts:
                stage_counts[stage] = {"defects": 0, "total_seen": 0}
            stage_counts[stage]["total_seen"] += 1

        # Reason code breakdown
        for rc in ld["reason_codes"]:
            if rc and rc != "passthrough":
                reason_code_counts[rc] = reason_code_counts.get(rc, 0) + 1

        # Reconstruction outcomes (from gate_reconstruction stage events)
        if "gate_reconstruction" in ld["stages"]:
            # Classification is stored in the reason code for gate_reconstruction events
            for rc in ld["reason_codes"]:
                if rc in (
                    "valid_geometry_preserved", "valid_geometry_degraded",
                    "invalid_geometry_rejected", "invalid_geometry_repaired",
                    "reconstruction_introduced_defect",
                ):
                    reconstruction_outcomes[rc] = reconstruction_outcomes.get(rc, 0) + 1

    # Label exploratory categories (Req 13.5)
    exploratory_cats: list[str] = []
    for cat, finding in attribution_counts.items():
        if finding.count < EXPLORATORY_THRESHOLD:
            finding.is_exploratory = True
            exploratory_cats.append(cat)
        # Attach representative examples
        finding.representative_examples = category_examples.get(cat, [])

    # Also process findings table data for additional context
    for row in findings_rows:
        finding_lineage_id = row[0]
        finding_type = row[1]
        finding_stage = row[2]
        _severity = row[3]
        _details_json = row[4]

        # Track finding types that aren't already captured by attribution
        key = f"finding:{finding_type}"
        if key not in attribution_counts:
            attribution_counts[key] = FindingCategory(category=key)
        attribution_counts[key].count += 1
        if key not in category_examples:
            category_examples[key] = []
        if len(category_examples[key]) < MAX_EXAMPLES_PER_CATEGORY:
            category_examples[key].append(finding_lineage_id)
        attribution_counts[key].representative_examples = category_examples.get(key, [])
        if attribution_counts[key].count < EXPLORATORY_THRESHOLD:
            attribution_counts[key].is_exploratory = True
            if key not in exploratory_cats:
                exploratory_cats.append(key)

    report.defects_by_attribution = attribution_counts
    report.by_profile = profile_counts
    report.by_setup_type = setup_type_counts
    report.by_symbol_class = symbol_class_counts
    report.by_stage = stage_counts
    report.by_reason_code = reason_code_counts
    report.reconstruction_outcomes = reconstruction_outcomes
    report.exploratory_categories = exploratory_cats

    return report


def generate_daily_report(
    engine,
    *,
    report_date: datetime | None = None,
) -> ProvenanceReport:
    """Generate a daily provenance report for the prior calendar day.

    If report_date is None, uses the current UTC date and reports on yesterday.
    If report_date is provided, reports on that calendar day (00:00 to 23:59:59 UTC).

    Returns a ProvenanceReport with all breakdowns and representative examples.
    """
    if report_date is None:
        today = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        start_time = today - timedelta(days=1)
    else:
        # Normalize to start of that day
        start_time = report_date.replace(
            hour=0, minute=0, second=0, microsecond=0,
            tzinfo=report_date.tzinfo or timezone.utc,
        )

    end_time = start_time + timedelta(days=1)

    events_rows, findings_rows = _query_report_data(engine, start_time, end_time)

    return _build_report_from_data(
        events_rows, findings_rows, start_time, end_time, "daily", engine
    )


def generate_weekly_report(
    engine,
    *,
    end_date: datetime | None = None,
) -> ProvenanceReport:
    """Generate a weekly provenance report for the prior 7 calendar days.

    If end_date is None, uses the current UTC date. Reports on the 7 days
    ending at the start of end_date (i.e., the prior 7 full days).

    Returns a ProvenanceReport with all breakdowns and representative examples.
    """
    if end_date is None:
        end_time = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    else:
        end_time = end_date.replace(
            hour=0, minute=0, second=0, microsecond=0,
            tzinfo=end_date.tzinfo or timezone.utc,
        )

    start_time = end_time - timedelta(days=7)

    events_rows, findings_rows = _query_report_data(engine, start_time, end_time)

    return _build_report_from_data(
        events_rows, findings_rows, start_time, end_time, "weekly", engine
    )


def build_ceo_summary(report: ProvenanceReport) -> dict:
    """Build CEO-level summary from a provenance report.

    Ordering (Requirement 13.6):
    1. Repeated upstream defects (>=3 occurrences in period)
    2. Threshold-tuning recommendations

    Rules:
    - Do NOT recommend loosening a gate that blocked winners with malformed
      upstream contracts (Requirement 13.7)
    - Reconstruction analysis is report-only — no automated policy changes
      until operator authorizes (Requirement 11.7)
    - Label categories with <20 occurrences as 'exploratory' (Requirement 13.5)
    - Separate counts from economic outcomes (Requirement 13.4)

    Returns a structured dict suitable for rendering in CEO reports.
    """
    summary: dict = {
        "period": {
            "start": report.period_start.isoformat(),
            "end": report.period_end.isoformat(),
            "type": report.period_type,
        },
        "headline_metrics": {
            "total_candidates": report.total_candidates,
            "coverage_pct": report.coverage.coverage_pct,
            "complete_provenance": report.coverage.complete_provenance,
            "malformed_at_pm_stage": report.malformed_at_pm_stage,
            "policy_rejections": report.policy_rejections,
            "integrity_rejections": report.integrity_rejections,
        },
        # Section 1: Repeated upstream defects (ordered first per Req 13.6)
        "repeated_upstream_defects": [],
        # Section 2: Threshold-tuning recommendations
        "threshold_tuning_recommendations": [],
        # Section 3: All defect categories with counts and examples
        "defect_categories": [],
        # Section 4: Reconstruction analysis (report-only, Req 11.7)
        "reconstruction_analysis": {
            "note": "Report-only. No automated policy changes until operator authorizes.",
            "outcomes": {},
        },
        # Section 5: Breakdowns
        "breakdowns": {
            "by_profile": report.by_profile,
            "by_setup_type": report.by_setup_type,
            "by_symbol_class": report.by_symbol_class,
        },
        "exploratory_categories": report.exploratory_categories,
    }

    # ── Section 1: Repeated upstream defects (>=3 occurrences) ──
    # These appear BEFORE threshold-tuning recommendations (Req 13.6)
    upstream_defect_categories = [
        "trusted_input_invalid",
        "raw_pm_output_invalid",
        "parse_or_normalization_invalid",
        "candidate_resolution_invalid",
        "price_repair_invalid",
        "behavioral_adjustment_invalid",
    ]

    repeated_defects = []
    for cat in upstream_defect_categories:
        if cat in report.defects_by_attribution:
            finding = report.defects_by_attribution[cat]
            if finding.count >= REPEATED_DEFECT_THRESHOLD:
                repeated_defects.append({
                    "category": cat,
                    "count": finding.count,
                    "is_exploratory": finding.is_exploratory,
                    "representative_examples": finding.representative_examples,
                    "economic_impact": {
                        "total_dollar_risk": str(finding.economic_outcomes.total_dollar_risk),
                        "total_potential_reward": str(
                            finding.economic_outcomes.total_potential_reward
                        ),
                        "affected_candidates": finding.economic_outcomes.count,
                    },
                })

    # Sort by count descending (most frequent first)
    repeated_defects.sort(key=lambda d: d["count"], reverse=True)
    summary["repeated_upstream_defects"] = repeated_defects

    # ── Section 2: Threshold-tuning recommendations ──
    # Only recommend tightening or adjusting thresholds, never loosening a gate
    # that blocked winners with malformed upstream contracts (Req 13.7).
    recommendations = []

    # Look for high-volume policy rejections of valid contracts
    policy_cat = report.defects_by_attribution.get("policy_rejection_of_valid_contract")
    if policy_cat and policy_cat.count >= REPEATED_DEFECT_THRESHOLD:
        recommendations.append({
            "type": "review_policy_thresholds",
            "reason": (
                f"{policy_cat.count} candidates with valid geometry "
                f"rejected by risk policy in this period."
            ),
            "count": policy_cat.count,
            "is_exploratory": policy_cat.is_exploratory,
            "representative_examples": policy_cat.representative_examples,
            "note": (
                "Only candidates with fully valid upstream contracts are included. "
                "Do NOT loosen gates that blocked winners with malformed upstream contracts."
            ),
        })

    # Check for reconstruction-degraded geometry that might suggest threshold tuning
    degraded_count = report.reconstruction_outcomes.get("valid_geometry_degraded", 0)
    if degraded_count >= REPEATED_DEFECT_THRESHOLD:
        recommendations.append({
            "type": "review_reconstruction_policy",
            "reason": (
                f"{degraded_count} candidates had valid geometry degraded by "
                f"gate reconstruction."
            ),
            "count": degraded_count,
            "is_exploratory": degraded_count < EXPLORATORY_THRESHOLD,
            "note": (
                "Reconstruction analysis is report-only. "
                "No automated policy changes until operator authorizes."
            ),
        })

    summary["threshold_tuning_recommendations"] = recommendations

    # ── Section 3: All defect categories ──
    all_categories = []
    for cat, finding in report.defects_by_attribution.items():
        all_categories.append({
            "category": cat,
            "count": finding.count,
            "is_exploratory": finding.is_exploratory,
            "representative_examples": finding.representative_examples,
            "economic_impact": {
                "total_dollar_risk": str(finding.economic_outcomes.total_dollar_risk),
                "total_potential_reward": str(
                    finding.economic_outcomes.total_potential_reward
                ),
                "affected_candidates": finding.economic_outcomes.count,
            },
        })

    # Sort: non-exploratory first, then by count descending
    all_categories.sort(key=lambda d: (d["is_exploratory"], -d["count"]))
    summary["defect_categories"] = all_categories

    # ── Section 4: Reconstruction analysis (report-only) ──
    summary["reconstruction_analysis"]["outcomes"] = report.reconstruction_outcomes

    return summary
