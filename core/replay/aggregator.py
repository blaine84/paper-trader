"""Gate Effectiveness Metrics and Reporting — Aggregation layer.

Aggregates replay audit results by gate, profile, setup_type, symbol_class,
and policy_version. Computes gate effectiveness metrics including blocked
winners, correctly blocked losers, false allows, estimated return impact,
MFE/MAE averages, stop/target-first rates, near-boundary analysis,
and repeated-omission detection for engineering defect surfacing.

Key behaviors:
- Groups with < min_sample_size candidates are labeled `exploratory`
  and receive no better/worse conclusion.
- Near-boundary: candidates whose gate input value falls within
  ±proximity_band (default 10%) of the threshold boundary.
- Repeated omissions: ≥threshold occurrences of the same metadata
  omission in a rolling window_days-day window → engineering defect.
- Directional conclusions require BOTH payoff ratio AND per-profile
  breakdown to agree (Requirement 8.6).

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
See: design.md §core/replay/aggregator.py
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class GateEffectivenessMetrics:
    """Gate effectiveness metrics for a single aggregation group.

    Fields cover all aspects of gate quality assessment:
    - Coverage breakdown (exact/partial/unscorable)
    - Decision distribution (original allows/rejects)
    - Decision delta counts
    - Outcome quality (blocked_winners, correctly_blocked_losers, false_allows)
    - Economic impact (estimated_return_impact_pct, avg_mfe_pct, avg_mae_pct)
    - Rate metrics (stop_first_rate, target_first_rate)
    - Near-boundary analysis
    - Allowed-to-rejected deltas
    - Sample size enforcement
    """

    # Group identity
    gate_name: str
    profile: str
    setup_type: str | None
    symbol_class: str | None
    policy_version: str
    era: str  # "historical" | "post-snapshot"

    # Coverage
    candidates_evaluated: int = 0
    exact_count: int = 0
    partial_count: int = 0
    unscorable_count: int = 0

    # Decision distribution
    original_allows: int = 0
    original_rejects: int = 0

    # Delta classification counts
    delta_counts: dict[str, int] = field(default_factory=dict)

    # Outcome quality
    blocked_winners: int = 0
    correctly_blocked_losers: int = 0
    false_allows: int = 0

    # Economic impact
    estimated_return_impact_pct: float = 0.0
    avg_mfe_pct: float = 0.0
    avg_mae_pct: float = 0.0

    # Rate metrics
    stop_first_rate: float = 0.0
    target_first_rate: float = 0.0

    # Near-boundary analysis
    near_boundary_count: int = 0
    near_boundary_target_rate: float = 0.0
    near_boundary_stop_rate: float = 0.0

    # Allowed-to-rejected deltas
    allowed_to_rejected_avoided_loss_pct: float = 0.0
    allowed_to_rejected_forgone_gain_pct: float = 0.0
    allowed_to_rejected_count: int = 0

    # Sample size enforcement
    meets_sample_size: bool = False

    # Directional conclusion (only populated when meets_sample_size AND
    # payoff ratio + per-profile breakdown agree)
    directional_conclusion: str | None = None  # "better" | "worse" | None


@dataclass
class DirectionalEvidence:
    """Evidence required for a directional conclusion (Requirement 8.6).

    Both payoff_ratio_supports AND per_profile_supports must agree
    before a better/worse label is assigned.
    """

    payoff_ratio: float | None = None  # avg_win_size / avg_loss_size
    payoff_ratio_supports: str | None = None  # "better" | "worse" | None
    per_profile_breakdown: dict[str, str] = field(default_factory=dict)
    per_profile_supports: str | None = None  # "better" | "worse" | None


@dataclass
class RepeatedOmission:
    """An engineering defect surfaced from repeated metadata omissions.

    Triggered when the same omission appears ≥threshold times in a
    rolling window_days-day window.
    """

    field_name: str
    gate_name: str
    occurrence_count: int
    first_occurrence: datetime
    last_occurrence: datetime
    affected_candidate_ids: list[str]
    defect_type: str = "metadata_wiring_defect"


# ---------------------------------------------------------------------------
# Supported group-by dimensions
# ---------------------------------------------------------------------------

VALID_GROUP_BY_DIMENSIONS: list[str] = [
    "gate",
    "profile",
    "setup_type",
    "symbol_class",
    "policy_version",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def aggregate_by_group(
    results: list[dict],
    group_by: list[str],
    *,
    min_sample_size: int = 30,
    near_boundary_pct: float = 0.10,
) -> list[GateEffectivenessMetrics]:
    """Aggregate replay audit results into gate effectiveness metrics.

    Args:
        results: List of replay audit record dicts. Each dict should contain:
            - replay_id, candidate_id
            - gate_name: the gate being evaluated
            - profile, setup_type, symbol_class, policy_version, era
            - replay_status: "exact" | "partial" | "unscorable"
            - original_decision: "allow" | "reject"
            - decision_delta_classification: one of DELTA_CATEGORIES
            - counterfactual_outcome: dict with first_hit, mfe, mae, return fields
              or None
            - gate_input_value: numeric value that the gate compared to threshold
            - gate_threshold_value: numeric threshold the gate used
            - missing_fields: list of field names missing at evaluation time
            - entry_timestamp: datetime of the candidate
            - allowed_to_rejected_return_pct: float or None (for allow→reject deltas)
        group_by: Dimensions to group by. Valid values:
            "gate", "profile", "setup_type", "symbol_class", "policy_version"
        min_sample_size: Minimum candidates before a group can produce
            a better/worse conclusion. Default 30.
        near_boundary_pct: Proximity band as fraction of threshold.
            Default 0.10 (10%).

    Returns:
        List of GateEffectivenessMetrics, one per unique group combination.
    """
    # Validate group_by dimensions
    for dim in group_by:
        if dim not in VALID_GROUP_BY_DIMENSIONS:
            raise ValueError(
                f"Invalid group_by dimension: {dim!r}. "
                f"Must be one of {VALID_GROUP_BY_DIMENSIONS}"
            )

    # Build groups
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in results:
        key = _build_group_key(record, group_by)
        groups[key].append(record)

    # Compute metrics for each group
    metrics_list: list[GateEffectivenessMetrics] = []
    for group_key, group_records in groups.items():
        metrics = _compute_group_metrics(
            group_key=group_key,
            group_by=group_by,
            records=group_records,
            min_sample_size=min_sample_size,
            near_boundary_pct=near_boundary_pct,
        )
        metrics_list.append(metrics)

    return metrics_list


def detect_repeated_omissions(
    results: list[dict],
    *,
    window_days: int = 7,
    threshold: int = 3,
) -> list[RepeatedOmission]:
    """Surface engineering defects from repeated metadata omissions.

    Scans replay audit records for patterns where the same field is
    missing (metadata omission) at the same gate ≥threshold times
    within a rolling window_days-day window.

    Args:
        results: List of replay audit record dicts containing:
            - candidate_id
            - gate_name
            - missing_fields: list of field names missing at evaluation
            - entry_timestamp: datetime of the candidate
        window_days: Rolling window size in days. Default 7.
        threshold: Minimum occurrences to surface as defect. Default 3.

    Returns:
        List of RepeatedOmission instances representing detected defects.
    """
    # Index omissions by (gate_name, field_name)
    omission_index: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for record in results:
        gate_name = record.get("gate_name", "")
        missing_fields = record.get("missing_fields") or []
        entry_timestamp = record.get("entry_timestamp")

        if not entry_timestamp or not missing_fields:
            continue

        for field_name in missing_fields:
            omission_index[(gate_name, field_name)].append(record)

    # Detect patterns within rolling window
    defects: list[RepeatedOmission] = []

    for (gate_name, field_name), records in omission_index.items():
        # Sort by timestamp
        sorted_records = sorted(
            records,
            key=lambda r: r.get("entry_timestamp", datetime.min),
        )

        # Sliding window: check if any window of window_days contains ≥threshold
        window = timedelta(days=window_days)
        i = 0
        detected_ranges: list[tuple[int, int]] = []

        while i < len(sorted_records):
            window_start = sorted_records[i].get("entry_timestamp")
            if window_start is None:
                i += 1
                continue

            window_end = window_start + window
            # Find all records within this window
            j = i
            while (
                j < len(sorted_records)
                and sorted_records[j].get("entry_timestamp") is not None
                and sorted_records[j]["entry_timestamp"] <= window_end
            ):
                j += 1

            count_in_window = j - i
            if count_in_window >= threshold:
                # Check we haven't already reported an overlapping range
                if not _overlaps_detected(detected_ranges, i, j):
                    detected_ranges.append((i, j))
                    window_records = sorted_records[i:j]
                    timestamps = [
                        r["entry_timestamp"]
                        for r in window_records
                        if r.get("entry_timestamp") is not None
                    ]
                    candidate_ids = [
                        r.get("candidate_id", "unknown")
                        for r in window_records
                    ]

                    defects.append(
                        RepeatedOmission(
                            field_name=field_name,
                            gate_name=gate_name,
                            occurrence_count=count_in_window,
                            first_occurrence=min(timestamps),
                            last_occurrence=max(timestamps),
                            affected_candidate_ids=candidate_ids,
                        )
                    )
            i += 1

    return defects


def compute_directional_evidence(
    group_metrics: GateEffectivenessMetrics,
    per_profile_metrics: list[GateEffectivenessMetrics],
) -> DirectionalEvidence:
    """Compute directional evidence for a group's conclusion.

    Directional conclusions require BOTH:
    1. Payoff ratio (avg win size / avg loss size) supports the direction
    2. Per-profile breakdown supports the same direction

    Args:
        group_metrics: The aggregate metrics for the group.
        per_profile_metrics: Metrics broken down by profile within the group.

    Returns:
        DirectionalEvidence with conclusion populated only when both
        payoff ratio and per-profile breakdown agree.
    """
    evidence = DirectionalEvidence()

    # Compute payoff ratio
    if group_metrics.blocked_winners > 0 and group_metrics.correctly_blocked_losers > 0:
        avg_win = (
            group_metrics.avg_mfe_pct
            if group_metrics.blocked_winners > 0
            else 0.0
        )
        avg_loss = abs(group_metrics.avg_mae_pct) if group_metrics.avg_mae_pct != 0.0 else 1.0
        if avg_loss > 0:
            evidence.payoff_ratio = avg_win / avg_loss
        else:
            evidence.payoff_ratio = None

        # A payoff ratio > 1.0 suggests the gate is blocking more losers
        # than winners in dollar terms → "better"
        if evidence.payoff_ratio is not None:
            if evidence.payoff_ratio > 1.0:
                evidence.payoff_ratio_supports = "better"
            elif evidence.payoff_ratio < 1.0:
                evidence.payoff_ratio_supports = "worse"
            else:
                evidence.payoff_ratio_supports = None

    # Per-profile breakdown
    profile_directions: list[str] = []
    for pm in per_profile_metrics:
        if not pm.meets_sample_size:
            continue
        # Profile supports "better" if correctly_blocked_losers > blocked_winners
        if pm.correctly_blocked_losers > pm.blocked_winners:
            profile_directions.append("better")
            evidence.per_profile_breakdown[pm.profile] = "better"
        elif pm.blocked_winners > pm.correctly_blocked_losers:
            profile_directions.append("worse")
            evidence.per_profile_breakdown[pm.profile] = "worse"
        else:
            evidence.per_profile_breakdown[pm.profile] = "neutral"

    # Per-profile supports a direction only if ALL qualifying profiles agree
    if profile_directions:
        if all(d == "better" for d in profile_directions):
            evidence.per_profile_supports = "better"
        elif all(d == "worse" for d in profile_directions):
            evidence.per_profile_supports = "worse"
        else:
            evidence.per_profile_supports = None

    return evidence


def apply_directional_conclusion(
    metrics: GateEffectivenessMetrics,
    evidence: DirectionalEvidence,
) -> None:
    """Apply directional conclusion to metrics if evidence supports it.

    Only assigns a conclusion when:
    1. The group meets minimum sample size
    2. Payoff ratio supports a direction
    3. Per-profile breakdown supports the SAME direction
    """
    if not metrics.meets_sample_size:
        metrics.directional_conclusion = None
        return

    if (
        evidence.payoff_ratio_supports is not None
        and evidence.per_profile_supports is not None
        and evidence.payoff_ratio_supports == evidence.per_profile_supports
    ):
        metrics.directional_conclusion = evidence.payoff_ratio_supports
    else:
        metrics.directional_conclusion = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_group_key(record: dict, group_by: list[str]) -> tuple:
    """Extract group key from a record based on the group_by dimensions."""
    key_parts: list[Any] = []
    for dim in group_by:
        if dim == "gate":
            key_parts.append(record.get("gate_name", "unknown"))
        elif dim == "profile":
            key_parts.append(record.get("profile", "unknown"))
        elif dim == "setup_type":
            key_parts.append(record.get("setup_type"))
        elif dim == "symbol_class":
            key_parts.append(record.get("symbol_class"))
        elif dim == "policy_version":
            key_parts.append(record.get("policy_version", "unknown"))
    return tuple(key_parts)


def _compute_group_metrics(
    group_key: tuple,
    group_by: list[str],
    records: list[dict],
    min_sample_size: int,
    near_boundary_pct: float,
) -> GateEffectivenessMetrics:
    """Compute all metrics for a single aggregation group."""
    # Extract group identity from key
    identity = _extract_group_identity(group_key, group_by)

    metrics = GateEffectivenessMetrics(
        gate_name=identity.get("gate", "all"),
        profile=identity.get("profile", "all"),
        setup_type=identity.get("setup_type"),
        symbol_class=identity.get("symbol_class"),
        policy_version=identity.get("policy_version", "unknown"),
        era=_determine_era(records),
    )

    metrics.candidates_evaluated = len(records)

    # Coverage breakdown
    for r in records:
        status = r.get("replay_status", "")
        if status == "exact":
            metrics.exact_count += 1
        elif status == "partial":
            metrics.partial_count += 1
        elif status == "unscorable":
            metrics.unscorable_count += 1

    # Decision distribution
    for r in records:
        original = _normalize_decision(r.get("original_decision", ""))
        if original == "allow":
            metrics.original_allows += 1
        elif original == "reject":
            metrics.original_rejects += 1

    # Delta classification counts
    delta_counts: dict[str, int] = defaultdict(int)
    for r in records:
        delta_class = r.get("decision_delta_classification", "")
        if delta_class:
            delta_counts[delta_class] += 1
    metrics.delta_counts = dict(delta_counts)

    # Outcome quality metrics
    _compute_outcome_metrics(metrics, records)

    # Rate metrics
    _compute_rate_metrics(metrics, records)

    # Near-boundary analysis
    _compute_near_boundary(metrics, records, near_boundary_pct)

    # Allowed-to-rejected deltas
    _compute_allowed_to_rejected(metrics, records)

    # Sample size enforcement
    metrics.meets_sample_size = metrics.candidates_evaluated >= min_sample_size

    return metrics


def _extract_group_identity(group_key: tuple, group_by: list[str]) -> dict[str, Any]:
    """Map group key tuple back to named dimensions."""
    identity: dict[str, Any] = {}
    for i, dim in enumerate(group_by):
        if i < len(group_key):
            identity[dim] = group_key[i]
    return identity


def _determine_era(records: list[dict]) -> str:
    """Determine era for the group. Uses majority era if mixed."""
    era_counts: dict[str, int] = defaultdict(int)
    for r in records:
        era = r.get("era", "historical")
        era_counts[era] += 1

    if not era_counts:
        return "historical"

    return max(era_counts, key=era_counts.get)  # type: ignore[arg-type]


def _normalize_decision(decision: str) -> str:
    """Normalize decision to canonical form."""
    mapping = {
        "allow": "allow",
        "allowed": "allow",
        "adjusted_allowed": "allow",
        "reject": "reject",
        "rejected": "reject",
        "block": "reject",
        "blocked": "reject",
    }
    return mapping.get(decision.lower(), decision.lower()) if decision else ""


def _compute_outcome_metrics(metrics: GateEffectivenessMetrics, records: list[dict]) -> None:
    """Compute blocked_winners, correctly_blocked_losers, false_allows, and impact metrics."""
    mfe_values: list[float] = []
    mae_values: list[float] = []
    return_impacts: list[float] = []

    for r in records:
        original = _normalize_decision(r.get("original_decision", ""))
        outcome = r.get("counterfactual_outcome") or {}
        first_hit = outcome.get("first_hit", "")

        # blocked_winners: blocked candidates whose shadow outcome hit target first
        if original == "reject" and first_hit == "target":
            metrics.blocked_winners += 1
            # Add to return impact (positive — money left on table)
            mfe = outcome.get("mfe")
            if mfe is not None:
                return_impacts.append(float(mfe))

        # correctly_blocked_losers: blocked candidates whose shadow outcome hit stop first
        elif original == "reject" and first_hit == "stop":
            metrics.correctly_blocked_losers += 1
            # Add to return impact (negative — loss avoided)
            mae = outcome.get("mae")
            if mae is not None:
                return_impacts.append(float(mae))

        # false_allows: allowed candidates that hit stop first
        if original == "allow" and first_hit == "stop":
            metrics.false_allows += 1

        # Collect MFE/MAE for averages
        mfe_val = outcome.get("mfe")
        mae_val = outcome.get("mae")
        if mfe_val is not None:
            mfe_values.append(float(mfe_val))
        if mae_val is not None:
            mae_values.append(float(mae_val))

    # Compute averages
    if mfe_values:
        metrics.avg_mfe_pct = sum(mfe_values) / len(mfe_values)
    if mae_values:
        metrics.avg_mae_pct = sum(mae_values) / len(mae_values)

    # Estimated return impact: sum of individual counterfactual returns
    metrics.estimated_return_impact_pct = sum(return_impacts)


def _compute_rate_metrics(metrics: GateEffectivenessMetrics, records: list[dict]) -> None:
    """Compute stop_first_rate and target_first_rate across all scored outcomes."""
    stop_first_count = 0
    target_first_count = 0
    scored_count = 0

    for r in records:
        outcome = r.get("counterfactual_outcome") or {}
        first_hit = outcome.get("first_hit", "")
        if first_hit in ("stop", "target", "neither"):
            scored_count += 1
            if first_hit == "stop":
                stop_first_count += 1
            elif first_hit == "target":
                target_first_count += 1

    if scored_count > 0:
        metrics.stop_first_rate = stop_first_count / scored_count
        metrics.target_first_rate = target_first_count / scored_count


def _compute_near_boundary(
    metrics: GateEffectivenessMetrics,
    records: list[dict],
    near_boundary_pct: float,
) -> None:
    """Identify candidates within proximity band of the threshold boundary.

    A candidate is near-boundary if its gate_input_value falls within
    ±(near_boundary_pct * threshold_value) of the threshold.
    """
    near_boundary_records: list[dict] = []

    for r in records:
        gate_input_value = r.get("gate_input_value")
        gate_threshold_value = r.get("gate_threshold_value")

        if gate_input_value is None or gate_threshold_value is None:
            continue

        try:
            input_val = float(gate_input_value)
            threshold_val = float(gate_threshold_value)
        except (ValueError, TypeError):
            continue

        if threshold_val == 0:
            continue

        proximity_band = abs(threshold_val * near_boundary_pct)
        lower_bound = threshold_val - proximity_band
        upper_bound = threshold_val + proximity_band

        if lower_bound <= input_val <= upper_bound:
            near_boundary_records.append(r)

    metrics.near_boundary_count = len(near_boundary_records)

    # Compute target-first and stop-first rates for near-boundary subset
    if near_boundary_records:
        nb_stop = 0
        nb_target = 0
        nb_scored = 0

        for r in near_boundary_records:
            outcome = r.get("counterfactual_outcome") or {}
            first_hit = outcome.get("first_hit", "")
            if first_hit in ("stop", "target", "neither"):
                nb_scored += 1
                if first_hit == "stop":
                    nb_stop += 1
                elif first_hit == "target":
                    nb_target += 1

        if nb_scored > 0:
            metrics.near_boundary_target_rate = nb_target / nb_scored
            metrics.near_boundary_stop_rate = nb_stop / nb_scored


def _compute_allowed_to_rejected(
    metrics: GateEffectivenessMetrics, records: list[dict]
) -> None:
    """Compute allowed-to-rejected delta metrics.

    These represent the estimated return that the proposed policy would
    have avoided or missed by rejecting originally-allowed candidates.
    """
    avoided_losses: list[float] = []
    forgone_gains: list[float] = []

    for r in records:
        delta_class = r.get("decision_delta_classification", "")
        if delta_class != "replay_rejects_original_allow":
            continue

        metrics.allowed_to_rejected_count += 1

        return_pct = r.get("allowed_to_rejected_return_pct")
        if return_pct is None:
            continue

        return_val = float(return_pct)
        if return_val < 0:
            # Negative return = loss that would have been avoided
            avoided_losses.append(abs(return_val))
        else:
            # Positive return = gain that would have been missed
            forgone_gains.append(return_val)

    if avoided_losses:
        metrics.allowed_to_rejected_avoided_loss_pct = sum(avoided_losses) / len(avoided_losses)
    if forgone_gains:
        metrics.allowed_to_rejected_forgone_gain_pct = sum(forgone_gains) / len(forgone_gains)


def _overlaps_detected(
    detected_ranges: list[tuple[int, int]], start: int, end: int
) -> bool:
    """Check if a proposed range overlaps with any already-detected range."""
    for d_start, d_end in detected_ranges:
        if start < d_end and end > d_start:
            return True
    return False
