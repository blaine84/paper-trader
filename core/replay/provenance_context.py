"""Provenance Context for Decision Replay.

Provides provenance-enriched context for replay operations, linking replay
records to original provenance chains and surfacing First_Invalid_Stage
attribution, upstream defect flags, and counterfactual labels.

This module is an optional enrichment layer — the existing gate_replayer
continues to function identically without it. All provenance queries are
guarded by PM_PROVENANCE_MODE check and fail-open to "historical_partial"
when provenance data is unavailable.

Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from typing import Any

from utils.gate_config import PM_PROVENANCE_MODE
from utils.provenance_capture import STAGE_TO_ATTRIBUTION

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ReplayProvenanceContext:
    """Provenance context attached to a replay operation.

    Provides enriched provenance data for replay reports per Requirements 12.1-12.6.

    Fields:
        lineage_id: The original candidate lineage identifier (Req 12.1)
        has_provenance: Whether provenance records exist for this candidate
        label: "full_provenance" or "historical_partial" (Req 12.2)
        first_invalid_stage: Earliest stage with invalid geometry, or "not_available" (Req 12.3)
        attribution_category: Attribution category from STAGE_TO_ATTRIBUTION mapping
        final_pre_gate_geometry: Geometry dict from pre_gate_snapshot event (Req 12.3)
        original_gate_decision: Original gate decision from production run (Req 12.3)
        original_gate_reason: Original gate reason code from production run (Req 12.3)
        upstream_valid_contract: True if all stages before gate were valid (Req 12.3)
        upstream_defect_present: True if geometry invalid at any stage before reconstruction (Req 12.4)
        is_counterfactual: True when corrected geometry differs from original (Req 12.6)
        counterfactual_corrected_fields: Fields that were corrected in counterfactual (Req 12.6)
        provenance_unavailable_reason: Why provenance is unavailable (for diagnostics)
    """

    lineage_id: str | None = None
    has_provenance: bool = False
    label: str = "historical_partial"  # "full_provenance" | "historical_partial"
    first_invalid_stage: str = "not_available"  # stage name or "not_available"
    attribution_category: str | None = None
    final_pre_gate_geometry: dict | None = None
    original_gate_decision: str | None = None
    original_gate_reason: str | None = None
    upstream_valid_contract: bool = True  # True if all stages before gate were valid
    upstream_defect_present: bool = False
    is_counterfactual: bool = False
    counterfactual_corrected_fields: list[str] = field(default_factory=list)
    provenance_unavailable_reason: str | None = None


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------


def build_replay_provenance_context(
    engine,
    lineage_id: str | None,
) -> ReplayProvenanceContext:
    """Build provenance context for a replay operation.

    If provenance mode is disabled or lineage has no provenance records,
    returns a "historical_partial" context (Requirement 12.2).

    Fail-open: any query failure results in "historical_partial" label
    with diagnostic reason recorded.

    Args:
        engine: SQLAlchemy engine for provenance database queries
        lineage_id: The candidate's Lineage_Identifier (Req 12.1)

    Returns:
        ReplayProvenanceContext with provenance data or historical_partial fallback
    """
    ctx = ReplayProvenanceContext(lineage_id=lineage_id)

    # Guard: provenance mode must be active
    if PM_PROVENANCE_MODE == "disabled" or lineage_id is None:
        ctx.label = "historical_partial"
        ctx.first_invalid_stage = "not_available"
        ctx.provenance_unavailable_reason = (
            "provenance_disabled" if PM_PROVENANCE_MODE == "disabled" else "no_lineage_id"
        )
        return ctx

    try:
        from sqlalchemy import text

        with engine.connect() as conn:
            events = conn.execute(
                text(
                    "SELECT stage_name, validation_after, geometry_after_json, "
                    "sequence_number FROM provenance_events "
                    "WHERE lineage_id = :lid ORDER BY sequence_number"
                ),
                {"lid": lineage_id},
            ).fetchall()

        if not events:
            ctx.label = "historical_partial"
            ctx.first_invalid_stage = "not_available"
            ctx.provenance_unavailable_reason = "no_provenance_records"
            return ctx

        # Has provenance — build full context (Req 12.1)
        ctx.has_provenance = True
        ctx.label = "full_provenance"

        # Find first invalid stage (Req 12.3)
        for event in events:
            row = event._mapping
            if row["validation_after"] == "invalid":
                ctx.first_invalid_stage = row["stage_name"]
                break

        if ctx.first_invalid_stage == "not_available":
            # No invalid stage found — geometry was valid throughout
            pass

        # Check if any stage before gate_reconstruction had invalid geometry (Req 12.4)
        for event in events:
            row = event._mapping
            stage = row["stage_name"]
            if stage == "gate_reconstruction":
                break
            if row["validation_after"] == "invalid":
                ctx.upstream_defect_present = True
                ctx.upstream_valid_contract = False
                break

        # Get attribution category from STAGE_TO_ATTRIBUTION mapping
        if ctx.first_invalid_stage and ctx.first_invalid_stage != "not_available":
            ctx.attribution_category = STAGE_TO_ATTRIBUTION.get(
                ctx.first_invalid_stage, "unknown"
            )

        # Get final pre-gate geometry from pre_gate_snapshot event (Req 12.3)
        for event in reversed(events):
            row = event._mapping
            if row["stage_name"] == "pre_gate_snapshot":
                geometry_json = row["geometry_after_json"]
                if geometry_json:
                    ctx.final_pre_gate_geometry = json.loads(geometry_json)
                break

        return ctx

    except Exception:
        # Fail-open: log error, return historical_partial (Req 12.2)
        logger.error(
            "Failed to build provenance context for lineage %s",
            lineage_id,
            exc_info=True,
        )
        ctx.label = "historical_partial"
        ctx.first_invalid_stage = "not_available"
        ctx.provenance_unavailable_reason = "query_failed"
        return ctx


# ---------------------------------------------------------------------------
# Replay result enrichment
# ---------------------------------------------------------------------------


def enrich_replay_result(
    replay_result: dict,
    provenance_ctx: ReplayProvenanceContext,
    replay_decision: str,
    replay_reason: str,
) -> dict:
    """Enrich a replay result dict with provenance context fields.

    Adds provenance fields to the replay result for reporting as specified
    in Requirement 12.3: original First_Invalid_Stage, final pre-gate geometry,
    original gate decision/reason, replay decision/reason, upstream-valid flag.

    Args:
        replay_result: The existing replay result dict to enrich
        provenance_ctx: Built provenance context for this candidate
        replay_decision: The replay's gate decision
        replay_reason: The replay's gate reason code

    Returns:
        The enriched replay_result dict (mutated in place and returned)
    """
    replay_result["provenance_label"] = provenance_ctx.label
    replay_result["first_invalid_stage"] = provenance_ctx.first_invalid_stage
    replay_result["attribution_category"] = provenance_ctx.attribution_category
    replay_result["upstream_defect_present"] = provenance_ctx.upstream_defect_present
    replay_result["upstream_valid_contract"] = provenance_ctx.upstream_valid_contract
    replay_result["original_gate_decision"] = provenance_ctx.original_gate_decision
    replay_result["original_gate_reason"] = provenance_ctx.original_gate_reason
    replay_result["replay_decision"] = replay_decision
    replay_result["replay_reason"] = replay_reason
    replay_result["final_pre_gate_geometry"] = provenance_ctx.final_pre_gate_geometry

    if provenance_ctx.is_counterfactual:
        replay_result["counterfactual_contract"] = True
        replay_result["counterfactual_corrected_fields"] = (
            provenance_ctx.counterfactual_corrected_fields
        )

    return replay_result


# ---------------------------------------------------------------------------
# Counterfactual labeling (Req 12.6)
# ---------------------------------------------------------------------------


def label_counterfactual(
    provenance_ctx: ReplayProvenanceContext,
    corrected_fields: list[str],
) -> ReplayProvenanceContext:
    """Label a replay as counterfactual when corrected geometry differs from original.

    Per Requirement 12.6: when analysis evaluates a corrected or repaired geometry
    that differs from the original pre-gate contract, label the result as
    'counterfactual_contract' and record which fields were corrected.

    Args:
        provenance_ctx: The current provenance context
        corrected_fields: List of field names that were corrected

    Returns:
        New ReplayProvenanceContext with counterfactual labeling applied
    """
    return replace(
        provenance_ctx,
        is_counterfactual=True,
        counterfactual_corrected_fields=corrected_fields,
    )


# ---------------------------------------------------------------------------
# Candidate-policy comparison mode (Req 12.5)
# ---------------------------------------------------------------------------


def build_policy_comparison_context(
    engine,
    lineage_id: str | None,
    original_gate_decision: str | None = None,
    original_gate_reason: str | None = None,
) -> ReplayProvenanceContext:
    """Build provenance context for candidate-policy comparison mode.

    Holds upstream geometry constant (same entry, stop, target, quantity) while
    the caller applies an alternate gate policy configuration (Requirement 12.5).

    This builds the provenance context and stamps the original gate decision
    and reason so the replay report can compare original vs alternate policy.

    Args:
        engine: SQLAlchemy engine for provenance database queries
        lineage_id: The candidate's Lineage_Identifier
        original_gate_decision: The original production gate decision
        original_gate_reason: The original production gate reason code

    Returns:
        ReplayProvenanceContext configured for policy comparison
    """
    ctx = build_replay_provenance_context(engine, lineage_id)
    ctx.original_gate_decision = original_gate_decision
    ctx.original_gate_reason = original_gate_reason
    return ctx
