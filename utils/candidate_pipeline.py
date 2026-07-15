"""Candidate Pipeline — authoritative single-pass execution pipeline.

Performs: resolve → reserve (CAS) → size → gates → execute → terminal state.
One function, exactly once, no double gate evaluation.

See: design.md §utils/candidate_pipeline.py
Requirements: 2.1, 2.2, 2.3, 2.4, 4.1, 6.1–6.10, 11.1, 11.2, 11.3, 11.4, 11.5
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from utils.candidate_registry import CandidateRecord, CandidateRegistry
from utils.decision_contract import CandidateDecision
from utils.gate_config import PM_PROVENANCE_MODE
from utils.position_sizer import SizingResult, calculate_position_size

if TYPE_CHECKING:
    from typing import Any
    from utils.checkpoint_logger import CheckpointLogger
    from utils.provenance_capture import ProvenanceChain

logger = logging.getLogger(__name__)


@dataclass
class ResolvedOrder:
    """Fully resolved order ready for sizing and gate evaluation."""

    candidate_id: str
    execution_key: str
    symbol: str
    action: str  # "BUY" or "SHORT"
    entry_price: float
    stop_price: float
    target_price: float
    setup_type: str
    risk_reward: float
    source_signal: dict
    profile_id: str
    geometry_name: str
    risk_multiplier: float  # 1.0 default, or PM-requested downward
    pm_rationale: str


@dataclass
class PipelineResult:
    """Result of the candidate execution pipeline."""

    candidate_id: str
    outcome: str  # "executed" | "reservation_failed" | "sizing_rejected" | "gate_rejected" | "execution_failed"
    resolved_order: ResolvedOrder | None = None
    sizing_result: SizingResult | None = None
    gate_notes: str | None = None
    error: str | None = None


def _generate_execution_key(candidate_id: str, cycle_id: str, profile_id: str) -> str:
    """Generate a deterministic execution key for crash recovery deduplication.

    The key is a 32-char hex prefix of SHA-256(candidate_id:cycle_id:profile_id).
    """
    raw = f"{candidate_id}:{cycle_id}:{profile_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def record_price_repair_provenance(
    chain: ProvenanceChain,
    *,
    direction: str,
    original_entry: float,
    replacement_entry: float,
    stop_price: float,
    target_price: float,
    quantity: int | float,
    repair_reason_code: str,
    original_source_timestamp: str | None = None,
    replacement_source_timestamp: str | None = None,
) -> None:
    """Record provenance event for live-price repair (legacy mode).

    Captures original price, replacement price, repair reason code, source
    timestamps, and geometry before/after. Recomputes geometry after price
    repair via compute_geometry(). Invalidates any previously recorded
    claimed_reward_to_risk after price change.

    Guarded by PM_PROVENANCE_MODE check at call site.
    Fail-open: caller wraps in try/except.

    Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7
    """
    from utils.geometry_calculator import compute_geometry

    # Geometry before repair: computed from original entry price
    geometry_before = compute_geometry(
        direction, original_entry, stop_price, target_price, quantity,
    )

    # Geometry after repair: recomputed from replacement entry price
    geometry_after = compute_geometry(
        direction, replacement_entry, stop_price, target_price, quantity,
    )

    # Input contract: original trade fields
    input_contract = {
        "entry_price": original_entry,
        "stop_price": stop_price,
        "target_price": target_price,
        "direction": direction,
        "quantity": quantity,
        "source_timestamp": original_source_timestamp,
    }

    # Output contract: repaired trade fields
    output_contract = {
        "entry_price": replacement_entry,
        "stop_price": stop_price,
        "target_price": target_price,
        "direction": direction,
        "quantity": quantity,
        "source_timestamp": replacement_source_timestamp,
        "claimed_reward_to_risk_invalidated": True,
    }

    # Determine fields changed
    fields_changed = ["entry_price"]

    chain.record_event(
        stage_name="price_repair",
        stage_version="1.0",
        input_contract=input_contract,
        output_contract=output_contract,
        fields_changed=fields_changed,
        mutation_reason_code=repair_reason_code,
        rule_id=None,
        geometry_before=geometry_before,
        geometry_after=geometry_after,
    )


def _resolve_candidate(
    registry: CandidateRegistry,
    decision: CandidateDecision,
    profile_id: str,
) -> tuple[CandidateRecord | None, str | None]:
    """Look up and validate candidate from registry.

    Returns (candidate, None) on success, or (None, error_reason) on failure.
    Validates: existence, profile match, cycle match.
    """
    candidate = registry.get(decision.candidate_id)

    if candidate is None:
        return None, "candidate not found"

    # Validate profile match (Requirement 6.2)
    if candidate.profile_id != profile_id:
        return None, f"profile mismatch: candidate belongs to {candidate.profile_id}, not {profile_id}"

    # Validate cycle match (Requirement 6.3)
    if candidate.cycle_id != registry.cycle_id:
        return None, f"cycle mismatch: candidate belongs to {candidate.cycle_id}, not {registry.cycle_id}"

    return candidate, None


def _build_resolved_order(
    candidate: CandidateRecord,
    decision: CandidateDecision,
    execution_key: str,
    profile_id: str,
) -> ResolvedOrder:
    """Build a ResolvedOrder from the candidate record and PM decision.

    All trade fields are derived exclusively from the registry (Req 2.1, 2.2).
    """
    # Parse signal snapshot from canonical JSON
    source_signal = json.loads(candidate.signal_snapshot_json)

    # Determine risk_multiplier: use PM-requested if provided, else 1.0
    risk_multiplier = decision.risk_multiplier if decision.risk_multiplier is not None else 1.0

    return ResolvedOrder(
        candidate_id=candidate.candidate_id,
        execution_key=execution_key,
        symbol=candidate.symbol,
        action=candidate.direction,  # "BUY" or "SHORT"
        entry_price=candidate.entry_price,
        stop_price=candidate.stop_price,
        target_price=candidate.target_price,
        setup_type=candidate.setup_type,
        risk_reward=candidate.risk_reward,
        source_signal=source_signal,
        profile_id=profile_id,
        geometry_name=candidate.geometry_name,
        risk_multiplier=risk_multiplier,
        pm_rationale=decision.rationale,
    )


def _build_gate_decision(resolved_order: ResolvedOrder, quantity: int) -> dict:
    """Build a decision dict compatible with _run_gate_pipeline and execute_trade.

    Provides all keys the gate pipeline and execution path expect.
    """
    return {
        "symbol": resolved_order.symbol,
        "action": resolved_order.action,
        "price": resolved_order.entry_price,
        "entry_price": resolved_order.entry_price,
        "stop": resolved_order.stop_price,
        "stop_price": resolved_order.stop_price,
        "stop_loss": resolved_order.stop_price,
        "target": resolved_order.target_price,
        "target_price": resolved_order.target_price,
        "profit_target": resolved_order.target_price,
        "setup_type": resolved_order.setup_type,
        "quantity": quantity,
        "rationale": resolved_order.pm_rationale,
        "geometry_name": resolved_order.geometry_name,
        "execution_key": resolved_order.execution_key,
    }


def _extract_rejecting_gate(gate_notes: list[dict] | None) -> tuple[str, str | None]:
    """Return the first hard-blocking gate name and reason code from gate notes."""
    if not gate_notes:
        return "gate_pipeline", None

    for note in gate_notes:
        if not isinstance(note, dict):
            continue
        if note.get("decision") in ("reject", "rejected", "override_required", "block"):
            return (
                note.get("gate") or note.get("gate_name") or "gate_pipeline",
                note.get("reason_type") or note.get("reason_code"),
            )

    return "gate_pipeline", None


def _commit_candidate_pipeline_session(db, candidate_id: str, context: str) -> None:
    """Release the ORM session before registry writes on a separate connection."""
    try:
        db.commit()
    except Exception:
        logger.error(
            "Candidate pipeline session commit failed for %s during %s",
            candidate_id,
            context,
            exc_info=True,
        )
        try:
            db.rollback()
        except Exception:
            logger.error(
                "Candidate pipeline session rollback failed for %s during %s",
                candidate_id,
                context,
                exc_info=True,
            )


def _record_pipeline_shadow_block(
    db,
    resolved_order: ResolvedOrder,
    sizing_result: SizingResult | None,
    *,
    outcome: str,
    block_reason: str,
    blocked_by: str,
    reason_code: str | None = None,
    gate_notes: list[dict] | str | None = None,
) -> None:
    """Mirror candidate-ID terminal rejects into the shadow ledger."""
    try:
        from utils.shadow_ledger import record_blocked_candidate

        decision_snapshot = {
            "candidate_id": resolved_order.candidate_id,
            "execution_key": resolved_order.execution_key,
            "pipeline_outcome": outcome,
            "geometry_name": resolved_order.geometry_name,
            "risk_reward": resolved_order.risk_reward,
            "risk_multiplier": resolved_order.risk_multiplier,
            "pm_rationale": resolved_order.pm_rationale,
            "rejection_reason": block_reason,
            "rejected_by": blocked_by,
        }
        if sizing_result is not None:
            decision_snapshot["sizing"] = {
                "quantity": sizing_result.quantity,
                "dollar_risk": sizing_result.dollar_risk,
                "position_value": sizing_result.position_value,
                "sizing_method": sizing_result.sizing_method,
                "applied_multiplier": sizing_result.applied_multiplier,
                "rejection_reason": sizing_result.rejection_reason,
            }

        record_blocked_candidate(
            db,
            resolved_order.symbol,
            resolved_order.action,
            blocked_by,
            block_reason,
            profile=resolved_order.profile_id,
            setup_type=resolved_order.setup_type,
            entry_price=resolved_order.entry_price,
            stop_price=resolved_order.stop_price,
            target_price=resolved_order.target_price,
            quantity=(sizing_result.quantity if sizing_result is not None else None),
            reason_code=reason_code or outcome,
            decision_snapshot=decision_snapshot,
            signal_snapshot=resolved_order.source_signal,
            gate_notes=gate_notes,
            source="candidate_id_pipeline",
            agent="portfolio_manager",
            geometry_candidate_id=resolved_order.candidate_id,
            geometry_candidate_name=resolved_order.geometry_name,
        )
    except Exception:
        logger.error(
            "Failed to record candidate pipeline shadow block for %s",
            resolved_order.candidate_id,
            exc_info=True,
        )


def _write_candidate_event(
    engine,
    candidate_id: str,
    cycle_id: str,
    profile_id: str,
    event_type: str,
    event_data: dict,
    candidate_type: str = "intraday",
) -> None:
    """Generic helper to INSERT a row into pm_candidate_events.

    This is the low-level writer used by all preflight event recording
    functions. It does NOT wrap in try/except — callers are responsible
    for fail-open guards.

    Args:
        engine: SQLAlchemy engine with access to pm_candidate_events table.
        candidate_id: UUID of the candidate.
        cycle_id: Current cycle identifier.
        profile_id: Active profile identifier.
        event_type: Event classification string.
        event_data: Dict payload serialized to JSON.
        candidate_type: "intraday" or "swing".
    """
    from datetime import datetime, timezone
    from sqlalchemy import text as sql_text

    now = datetime.now(timezone.utc).isoformat()
    with engine.connect() as conn:
        conn.execute(
            sql_text("""
                INSERT INTO pm_candidate_events
                (candidate_id, cycle_id, profile_id, event_type, event_data, created_at, candidate_type)
                VALUES (:candidate_id, :cycle_id, :profile_id, :event_type, :event_data, :created_at, :candidate_type)
            """),
            {
                "candidate_id": candidate_id,
                "cycle_id": cycle_id,
                "profile_id": profile_id,
                "event_type": event_type,
                "event_data": json.dumps(event_data),
                "created_at": now,
                "candidate_type": candidate_type,
            },
        )
        conn.commit()


def _write_preflight_passed_event(
    engine,
    candidate: CandidateRecord,
    cycle_id: str,
    profile_id: str,
    preflight_summary,
) -> None:
    """Write preflight_passed event with full summary dict. Fail-open.

    Records all boolean check results and an empty blocking_reason_codes list
    when a candidate passes all preflight checks.

    Requirements: 2.2, 2.3, 9.4
    """
    try:
        event_data = {
            "has_entry_stop_target": preflight_summary.has_entry_stop_target,
            "min_risk_reward_met": preflight_summary.min_risk_reward_met,
            "direction_valid": preflight_summary.direction_valid,
            "profile_allowed": preflight_summary.profile_allowed,
            "candidate_not_expired": preflight_summary.candidate_not_expired,
            "cash_available": preflight_summary.cash_available,
            "sizing_possible": preflight_summary.sizing_possible,
            "max_positions_available": preflight_summary.max_positions_available,
            "same_symbol_allowed": preflight_summary.same_symbol_allowed,
            "blocking_reason_codes": [],
        }
        _write_candidate_event(
            engine, candidate.candidate_id, cycle_id, profile_id,
            "preflight_passed", event_data, candidate.candidate_type,
        )
    except Exception:
        logger.error(
            "Failed to write preflight_passed event for %s",
            candidate.candidate_id,
            exc_info=True,
        )


def _write_preflight_failed_event(
    engine,
    candidate: CandidateRecord,
    cycle_id: str,
    profile_id: str,
    preflight_summary,
) -> None:
    """Write preflight_failed event with blocking_reason_codes. Fail-open.

    Records all boolean check results and the list of blocking reason codes
    when a candidate fails one or more preflight checks.

    Requirements: 2.2, 9.4
    """
    try:
        event_data = {
            "has_entry_stop_target": preflight_summary.has_entry_stop_target,
            "min_risk_reward_met": preflight_summary.min_risk_reward_met,
            "direction_valid": preflight_summary.direction_valid,
            "profile_allowed": preflight_summary.profile_allowed,
            "candidate_not_expired": preflight_summary.candidate_not_expired,
            "cash_available": preflight_summary.cash_available,
            "sizing_possible": preflight_summary.sizing_possible,
            "max_positions_available": preflight_summary.max_positions_available,
            "same_symbol_allowed": preflight_summary.same_symbol_allowed,
            "blocking_reason_codes": list(preflight_summary.blocking_reason_codes),
        }
        _write_candidate_event(
            engine, candidate.candidate_id, cycle_id, profile_id,
            "preflight_failed", event_data, candidate.candidate_type,
        )
    except Exception:
        logger.error(
            "Failed to write preflight_failed event for %s",
            candidate.candidate_id,
            exc_info=True,
        )


def _write_preflight_excluded_event(
    engine,
    candidate: CandidateRecord,
    cycle_id: str,
    profile_id: str,
    preflight_summary,
) -> None:
    """Write preflight_excluded event with geometry and shadow_eligible. Fail-open.

    Records blocking reason codes, the candidate's signal snapshot, geometry
    fields, and whether the candidate is eligible for shadow analysis.

    shadow_eligible is true if all of entry_price, stop_price, target_price,
    and risk_reward are present and non-zero.

    Requirements: 3.1, 3.3, 3.4, 9.4
    """
    try:
        # shadow_eligible = all geometry fields present and non-zero
        shadow_eligible = (
            candidate.entry_price is not None and candidate.entry_price != 0
            and candidate.stop_price is not None and candidate.stop_price != 0
            and candidate.target_price is not None and candidate.target_price != 0
            and candidate.risk_reward is not None and candidate.risk_reward != 0
        )
        event_data = {
            "blocking_reason_codes": list(preflight_summary.blocking_reason_codes),
            "signal_snapshot_json": candidate.signal_snapshot_json,
            "entry_price": candidate.entry_price,
            "stop_price": candidate.stop_price,
            "target_price": candidate.target_price,
            "risk_reward": candidate.risk_reward,
            "geometry_name": candidate.geometry_name,
            "shadow_eligible": shadow_eligible,
        }
        _write_candidate_event(
            engine, candidate.candidate_id, cycle_id, profile_id,
            "preflight_excluded", event_data, candidate.candidate_type,
        )
    except Exception:
        logger.error(
            "Failed to write preflight_excluded event for %s",
            candidate.candidate_id,
            exc_info=True,
        )


def _write_execution_failed_event(
    engine,
    candidate: CandidateRecord,
    cycle_id: str,
    profile_id: str,
    intended_action: str,
    attempted_quantity: int,
    failure_reason: str,
) -> None:
    """Write execution_failed event with truncated failure_reason. Fail-open.

    This is DISTINCT from pm_reject and gate_fail events. It indicates that
    a candidate passed PM selection and all gates but no trade row was created.

    The failure_reason is truncated to a maximum of 1024 characters.

    Requirements: 6.1, 6.2
    """
    try:
        event_data = {
            "candidate_id": candidate.candidate_id,
            "profile": profile_id,
            "symbol": candidate.symbol,
            "intended_action": intended_action,
            "attempted_quantity": attempted_quantity,
            "failure_reason": str(failure_reason)[:1024],
        }
        _write_candidate_event(
            engine, candidate.candidate_id, cycle_id, profile_id,
            "execution_failed", event_data, candidate.candidate_type,
        )
    except Exception:
        logger.error(
            "Failed to write execution_failed event for %s",
            candidate.candidate_id,
            exc_info=True,
        )


def _write_execution_fallback_blocked_event(
    engine,
    candidate: CandidateRecord,
    cycle_id: str,
    profile_id: str,
    blocked_path: str,
) -> None:
    """Write execution_fallback_blocked event when legacy path is blocked. Fail-open.

    Emitted when the execution path targets the legacy free-form order
    construction path while the pipeline is operating in candidate-ID mode.

    Requirements: 6.3
    """
    try:
        event_data = {
            "candidate_id": candidate.candidate_id,
            "blocked_path": blocked_path,
        }
        _write_candidate_event(
            engine, candidate.candidate_id, cycle_id, profile_id,
            "execution_fallback_blocked", event_data, candidate.candidate_type,
        )
    except Exception:
        logger.error(
            "Failed to write execution_fallback_blocked event for %s",
            candidate.candidate_id,
            exc_info=True,
        )


def _attempt_execution_failed_transition(engine, candidate_id: str) -> None:
    """Attempt CAS transition to EXECUTION_FAILED state. Fail-open.

    Attempts to transition the candidate from RESERVED to EXECUTION_FAILED
    using a Compare-And-Swap update. If CAS fails (rowcount == 0), this means
    another thread/process already moved the state — log and continue, relying
    on finalize_cycle() to enforce the terminal state guarantee.

    Requirements: 6.4, 6.5
    """
    try:
        from sqlalchemy import text as sql_text

        with engine.connect() as conn:
            result = conn.execute(
                sql_text(
                    "UPDATE pm_candidates SET state = 'execution_failed' "
                    "WHERE candidate_id = :candidate_id AND state = 'reserved'"
                ),
                {"candidate_id": candidate_id},
            )
            conn.commit()
            if result.rowcount == 0:
                logger.error(
                    "CAS transition to EXECUTION_FAILED failed for %s (rowcount=0), "
                    "relying on finalize_cycle()",
                    candidate_id,
                )
    except Exception:
        logger.error(
            "Exception during EXECUTION_FAILED CAS transition for %s, "
            "relying on finalize_cycle()",
            candidate_id,
            exc_info=True,
        )


def _write_pm_accept_event(
    engine,
    candidate: CandidateRecord,
    cycle_id: str,
    profile_id: str,
    risk_multiplier: float,
) -> None:
    """Write pm_accept event with candidate_id, profile, risk_multiplier. Fail-open.

    Requirements: 5.1, 9.4
    """
    try:
        event_data = {
            "candidate_id": candidate.candidate_id,
            "profile": profile_id,
            "risk_multiplier": risk_multiplier,
        }
        _write_candidate_event(
            engine, candidate.candidate_id, cycle_id, profile_id,
            "pm_accept", event_data, candidate.candidate_type,
        )
    except Exception:
        logger.error(
            "Failed to write pm_accept event for %s",
            candidate.candidate_id,
            exc_info=True,
        )


def _write_sizing_event(
    engine,
    candidate: CandidateRecord,
    cycle_id: str,
    profile_id: str,
    passed: bool,
    quantity: int,
    dollar_risk: float,
    risk_percent: float,
    reason_code: str | None = None,
) -> None:
    """Write sizing_pass or sizing_fail event. Fail-open.

    Args:
        engine: SQLAlchemy engine.
        candidate: CandidateRecord being evaluated.
        cycle_id: Current cycle identifier.
        profile_id: Active profile identifier.
        passed: True if sizing succeeded, False if rejected.
        quantity: Computed quantity (0 if rejected).
        dollar_risk: Dollar amount at risk.
        risk_percent: Percentage of portfolio equity risked.
        reason_code: Rejection rule identifier (only when passed=False).

    Requirements: 5.2, 9.4
    """
    try:
        event_type = "sizing_pass" if passed else "sizing_fail"
        event_data = {
            "quantity": quantity,
            "dollar_risk": str(dollar_risk),
            "risk_percent": str(risk_percent),
        }
        if not passed and reason_code:
            event_data["reason_code"] = reason_code
        _write_candidate_event(
            engine, candidate.candidate_id, cycle_id, profile_id,
            event_type, event_data, candidate.candidate_type,
        )
    except Exception:
        logger.error(
            "Failed to write sizing event for %s",
            candidate.candidate_id,
            exc_info=True,
        )


# Fixed gate pipeline order for telemetry events.
# Maps canonical gate names to the gate key used in _run_gate_pipeline notes.
GATE_PIPELINE_ORDER = [
    "setup_quality",
    "pre_trade_quality",
    "catalyst_specificity",
    "risk_geometry",
    "concentration",
]

# Map from _run_gate_pipeline note "gate" field to canonical gate name
_GATE_NOTE_TO_CANONICAL = {
    "setup_quality_gate": "setup_quality",
    "pre_trade_quality_gate": "pre_trade_quality",
    "catalyst_specificity_gate": "catalyst_specificity",
    "risk_geometry_gate": "risk_geometry",
    "concentration_gate": "concentration",
}


def _write_gate_events(
    engine,
    candidate: CandidateRecord,
    cycle_id: str,
    profile_id: str,
    gate_notes: list[dict],
) -> None:
    """Write gate_pass/gate_fail events in pipeline order, stopping after first failure. Fail-open.

    Iterates through the gate notes in pipeline order. For each gate that
    has a result in gate_notes, writes a gate_pass or gate_fail event.
    Stops after the first gate_fail event.

    A gate is considered failed if its decision is "reject", "rejected",
    "override_required", or "block".

    Requirements: 5.3, 9.4
    """
    try:
        if not gate_notes:
            return

        # Index gate notes by canonical name for ordered iteration
        notes_by_canonical = {}
        for note in gate_notes:
            if not isinstance(note, dict):
                continue
            gate_key = note.get("gate") or note.get("gate_name")
            if gate_key:
                canonical = _GATE_NOTE_TO_CANONICAL.get(gate_key, gate_key)
                notes_by_canonical[canonical] = note

        # Emit events in fixed pipeline order, stopping after first failure
        for gate_name in GATE_PIPELINE_ORDER:
            note = notes_by_canonical.get(gate_name)
            if note is None:
                continue

            decision = note.get("decision", "")
            gate_failed = decision in ("reject", "rejected", "override_required", "block")
            event_type = "gate_fail" if gate_failed else "gate_pass"

            event_data = {"gate_name": gate_name}
            if gate_failed:
                event_data["reason_code"] = (
                    note.get("reason_type")
                    or note.get("reason_code")
                    or note.get("reason")
                    or gate_name
                )

            _write_candidate_event(
                engine, candidate.candidate_id, cycle_id, profile_id,
                event_type, event_data, candidate.candidate_type,
            )

            if gate_failed:
                break  # Stop after first failure

    except Exception:
        logger.error(
            "Failed to write gate events for %s",
            candidate.candidate_id,
            exc_info=True,
        )


def _determine_first_blocking_stage(
    sizing_failed: bool,
    failing_gate_name: str | None,
) -> str | None:
    """Determine first_blocking_stage from fixed pipeline ordering.

    Uses the fixed ordering: position_sizer → setup_quality → pre_trade_quality
    → catalyst_specificity → risk_geometry → concentration.

    Args:
        sizing_failed: True if the position sizer rejected the candidate.
        failing_gate_name: Canonical name of the first failing gate (from gate notes),
            or None if no gate failed.

    Returns:
        The first blocking stage name, or None if neither sizing nor gates failed.

    Requirements: 5.4
    """
    if sizing_failed:
        return "position_sizer"
    if failing_gate_name:
        return failing_gate_name
    return None


def record_behavioral_adjustment_provenance(
    chain: 'ProvenanceChain | None',
    resolved_order: ResolvedOrder,
    sizing_result: SizingResult,
    risk_multiplier: float,
    profile_id: str,
) -> bool:
    """Record provenance for behavioral/profile adjustments.

    Called after position sizing is complete.
    Guarded by PM_PROVENANCE_MODE check at call site.

    For candidate-ID mode, the behavioral adjustment is the risk_multiplier
    application — a quantity-only change where per-unit geometry is preserved.

    Returns True if geometry is valid after adjustment, False if invalid
    (caller should NOT advance the contract).

    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8
    """
    from utils.geometry_calculator import compute_geometry

    if chain is None:
        return True

    try:
        direction = resolved_order.action
        entry = resolved_order.entry_price
        stop = resolved_order.stop_price
        target = resolved_order.target_price
        final_quantity = sizing_result.quantity

        # Geometry before: per-unit (quantity=1) — represents the candidate's
        # intrinsic geometry before risk_multiplier scaled the quantity
        geometry_before = compute_geometry(direction, entry, stop, target, 1)

        # Geometry after: with the final sized quantity from position_sizer
        geometry_after = compute_geometry(direction, entry, stop, target, final_quantity)

        # If risk_multiplier == 1.0, no material behavioral adjustment occurred
        if risk_multiplier == 1.0:
            chain.record_passthrough(
                stage_name="behavioral_adjustment",
                stage_version="1.0",
                validation_status=geometry_after.validation_status.value,
            )
            return geometry_after.is_valid

        # risk_multiplier != 1.0 — material quantity-only adjustment (Req 8.4)
        # Per-unit geometry preserved: entry, stop, target unchanged
        # Only quantity and total_dollar_risk differ

        # Input contract: state before behavioral adjustment (per-unit baseline)
        input_contract = {
            "entry_price": entry,
            "stop_price": stop,
            "target_price": target,
            "direction": direction,
            "risk_multiplier": risk_multiplier,
            "profile_id": profile_id,
            "quantity_before": 1,
            "total_dollar_risk_before": str(geometry_before.total_dollar_risk),
            "per_unit_risk": str(geometry_before.per_unit_risk),
            "reward_to_risk": str(geometry_before.reward_to_risk),
        }

        # Output contract: state after behavioral adjustment
        output_contract = {
            "entry_price": entry,
            "stop_price": stop,
            "target_price": target,
            "direction": direction,
            "quantity_after": final_quantity,
            "total_dollar_risk_after": str(geometry_after.total_dollar_risk),
            "per_unit_risk": str(geometry_after.per_unit_risk),
            "reward_to_risk": str(geometry_after.reward_to_risk),
            "risk_multiplier_applied": risk_multiplier,
        }

        # Fields changed: quantity and total_dollar_risk (per-unit geometry preserved)
        fields_changed = ["quantity", "total_dollar_risk"]

        chain.record_event(
            stage_name="behavioral_adjustment",
            stage_version="1.0",
            input_contract=input_contract,
            output_contract=output_contract,
            fields_changed=fields_changed,
            mutation_reason_code="risk_multiplier_application",
            rule_id=f"risk_multiplier:{risk_multiplier}",
            geometry_before=geometry_before,
            geometry_after=geometry_after,
        )

        # Check if adjustment produced invalid geometry (Req 8.6)
        if not geometry_after.is_valid:
            logger.warning(
                "Behavioral adjustment created invalid geometry for %s: "
                "risk_multiplier=%s, quantity=%d",
                resolved_order.candidate_id,
                risk_multiplier,
                final_quantity,
            )
            return False

        # Check if reward-to-risk degraded (Req 8.8) — informational logging
        # For quantity-only changes, R:R should be identical; log if different
        if (geometry_before.reward_to_risk > 0
                and geometry_after.reward_to_risk < geometry_before.reward_to_risk):
            logger.debug(
                "Behavioral adjustment degraded R:R for %s: %s -> %s",
                resolved_order.candidate_id,
                geometry_before.reward_to_risk,
                geometry_after.reward_to_risk,
            )

        return True

    except Exception:
        # Fail-open: provenance must never block the pipeline
        logger.error(
            "Failed to record behavioral adjustment provenance for %s",
            resolved_order.candidate_id,
            exc_info=True,
        )
        return True


def record_pre_gate_snapshot_provenance(
    chain: 'ProvenanceChain | None',
    resolved_order: ResolvedOrder,
    quantity: int,
    engine=None,
) -> bool:
    """Record pre-gate snapshot provenance event and validate structural geometry.

    Returns True if the contract is structurally valid (can proceed to gates).
    Returns False if the contract has missing/non-finite/zero/directionally-invalid
    fields — these MUST be rejected with reason_code pre_gate_contract_invalid
    regardless of provenance mode.

    This function:
    1. Computes authoritative geometry via compute_geometry()
    2. Validates structural geometry (always fail-closed)
    3. Records pre_gate_snapshot provenance event in the chain
    4. Checks claimed-vs-computed mismatch (record finding only)
    5. Batch-persists provenance chain if engine provided

    Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7
    """
    from utils.geometry_calculator import compute_geometry
    from utils.provenance_capture import persist_provenance_chain
    from utils.claimed_vs_computed import (
        extract_claimed_reward_risk,
        compare_claimed_vs_computed,
    )

    direction = resolved_order.action
    entry = resolved_order.entry_price
    stop = resolved_order.stop_price
    target = resolved_order.target_price

    # 1. Compute authoritative geometry
    geometry = compute_geometry(direction, entry, stop, target, quantity)

    # 2. Structural geometry validation (always fail-closed — Req 9.4)
    if not geometry.is_valid:
        # Record the pre_gate_snapshot event with invalid geometry before rejecting
        if chain is not None:
            # Geometry before: same as after (no mutation at this stage)
            geometry_before = geometry

            # Collect accumulated mutation reason codes from prior chain events
            accumulated_reason_codes = [
                e.mutation_reason_code for e in chain.events
                if e.mutation_reason_code and e.mutation_reason_code != "passthrough"
            ]

            input_contract = {
                "lineage_id": chain.lineage_id,
                "candidate_id": resolved_order.candidate_id,
                "symbol": resolved_order.symbol,
                "profile": resolved_order.profile_id,
                "direction": direction,
                "setup_type": resolved_order.setup_type,
                "entry_price": entry,
                "stop_price": stop,
                "target_price": target,
                "quantity": quantity,
                "computed_reward_to_risk": str(geometry.reward_to_risk),
                "total_dollar_risk": str(geometry.total_dollar_risk),
                "validation_status": geometry.validation_status.value,
                "accumulated_mutation_reason_codes": accumulated_reason_codes,
            }

            output_contract = {
                "rejected": True,
                "reason_code": "pre_gate_contract_invalid",
                "validation_errors": [
                    {"field": e.field_name, "reason": e.reason, "value": e.value}
                    for e in geometry.validation_errors
                ],
            }

            chain.record_event(
                stage_name="pre_gate_snapshot",
                stage_version="1.0",
                input_contract=input_contract,
                output_contract=output_contract,
                fields_changed=[],
                mutation_reason_code="pre_gate_contract_invalid",
                rule_id=None,
                geometry_before=geometry_before,
                geometry_after=geometry,
            )

            # Batch-persist even on rejection (Req 9.1 — persist before gate)
            if engine is not None and chain.events:
                persist_provenance_chain(engine, chain)

        return False

    # 3. Record pre_gate_snapshot provenance event (valid contract)
    if chain is not None:
        # Geometry before: same as after (no mutation at this stage — snapshot only)
        geometry_before = geometry

        # Collect accumulated mutation reason codes
        accumulated_reason_codes = [
            e.mutation_reason_code for e in chain.events
            if e.mutation_reason_code and e.mutation_reason_code != "passthrough"
        ]

        # Market-data source timestamps from source_signal if available
        source_signal = resolved_order.source_signal
        market_data_timestamps = {
            "entry_source_timestamp": source_signal.get("entry_source_timestamp")
            or source_signal.get("source_timestamp"),
            "stop_source_timestamp": source_signal.get("stop_source_timestamp"),
            "target_source_timestamp": source_signal.get("target_source_timestamp"),
            "signal_timestamp": source_signal.get("timestamp")
            or source_signal.get("signal_timestamp"),
        }

        input_contract = {
            "lineage_id": chain.lineage_id,
            "candidate_id": resolved_order.candidate_id,
            "symbol": resolved_order.symbol,
            "profile": resolved_order.profile_id,
            "direction": direction,
            "setup_type": resolved_order.setup_type,
            "entry_price": entry,
            "stop_price": stop,
            "target_price": target,
            "quantity": quantity,
            "computed_reward_to_risk": str(geometry.reward_to_risk),
            "total_dollar_risk": str(geometry.total_dollar_risk),
            "market_data_timestamps": market_data_timestamps,
            "validation_status": geometry.validation_status.value,
            "accumulated_mutation_reason_codes": accumulated_reason_codes,
        }

        # 4. Claimed-vs-computed mismatch check (Req 9.5)
        claimed_numeric, claimed_phrase = extract_claimed_reward_risk(
            resolved_order.source_signal,
            resolved_order.pm_rationale,
        )
        comparison = compare_claimed_vs_computed(
            claimed_numeric, claimed_phrase, geometry.reward_to_risk,
        )

        # Apply Req 9.5 thresholds: 20% relative OR 0.25 absolute (whichever smaller)
        claimed_computed_finding = None
        if comparison.claimed_value is not None and comparison.absolute_difference is not None:
            from decimal import Decimal
            abs_threshold = Decimal("0.25")
            # 20% relative of the computed value
            relative_threshold = abs(geometry.reward_to_risk) * Decimal("0.20")
            # Use whichever threshold is smaller
            effective_threshold = min(abs_threshold, relative_threshold)
            if comparison.absolute_difference > effective_threshold:
                claimed_computed_finding = {
                    "classification": "claimed_computed_mismatch",
                    "claimed_value": str(comparison.claimed_value),
                    "computed_value": str(comparison.computed_value),
                    "absolute_difference": str(comparison.absolute_difference),
                    "threshold_applied": str(effective_threshold),
                }

        output_contract = {
            "contract_valid": True,
            "reason_code": "pre_gate_contract_valid",
            "claimed_vs_computed": {
                "claimed_value": str(comparison.claimed_value) if comparison.claimed_value else None,
                "claimed_phrase": comparison.claimed_phrase,
                "computed_reward_to_risk": str(comparison.computed_value),
                "is_numeric_mismatch": comparison.is_numeric_mismatch,
                "is_categorical": comparison.is_categorical,
                "claim_absent": comparison.claim_absent,
            },
        }
        if claimed_computed_finding:
            output_contract["claimed_computed_mismatch"] = claimed_computed_finding

        chain.record_event(
            stage_name="pre_gate_snapshot",
            stage_version="1.0",
            input_contract=input_contract,
            output_contract=output_contract,
            fields_changed=[],
            mutation_reason_code="pre_gate_snapshot_recorded",
            rule_id=None,
            geometry_before=geometry_before,
            geometry_after=geometry,
        )

        # 5. Batch-persist all pre-gate provenance events (Req 9.1)
        if engine is not None and chain.events:
            persist_provenance_chain(engine, chain)

    return True


def record_gate_reconstruction_provenance(
    chain: 'ProvenanceChain | None',
    resolved_order: ResolvedOrder,
    gate_result: dict,
    original_quantity: int,
    engine=None,
) -> None:
    """Record post-gate provenance event for reconstruction (append-only).

    Called AFTER gate pipeline completes. Records:
    - Original and reconstructed values (entry, stop, target, qty, R:R, dollar risk)
    - Whether original geometry was valid before reconstruction
    - Reconstruction outcome classification
    - Defect info if reconstruction introduced a new defect

    This is a SEPARATE INSERT after gate pipeline (Req 11.1).
    Fail-open: all operations in try/except.

    Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
    """
    from utils.geometry_calculator import compute_geometry
    from utils.first_invalid_stage import classify_reconstruction_outcome
    from utils.provenance_capture import persist_provenance_chain

    if chain is None:
        return

    try:
        direction = resolved_order.action
        entry = resolved_order.entry_price
        stop = resolved_order.stop_price
        target = resolved_order.target_price

        # Original geometry (before reconstruction)
        geometry_before = compute_geometry(direction, entry, stop, target, original_quantity)

        # Post-reconstruction values from gate result
        adjusted_stop = gate_result.get("adjusted_stop_price") or stop
        adjusted_quantity = gate_result.get("adjusted_quantity") or original_quantity

        # Geometry after reconstruction
        geometry_after = compute_geometry(direction, entry, adjusted_stop, target, adjusted_quantity)

        # Determine if reconstruction occurred
        decision_str = gate_result.get("decision", "")
        if decision_str in ("passed_unchanged", "allow") and adjusted_stop == stop and adjusted_quantity == original_quantity:
            # No reconstruction — record passthrough
            chain.record_passthrough(
                stage_name="gate_reconstruction",
                stage_version="1.0",
                validation_status=geometry_after.validation_status.value,
            )
        else:
            # Reconstruction occurred
            pre_valid = geometry_before.is_valid
            post_valid = geometry_after.is_valid
            rr_before = geometry_before.reward_to_risk
            rr_after = geometry_after.reward_to_risk

            outcome = classify_reconstruction_outcome(pre_valid, post_valid, rr_before, rr_after)

            # Input contract: original values (Req 11.1 — paired before fields)
            input_contract = {
                "entry_price": entry,
                "stop_price": stop,
                "target_price": target,
                "quantity": original_quantity,
                "reward_to_risk": str(geometry_before.reward_to_risk),
                "dollar_risk": str(geometry_before.total_dollar_risk),
                "pre_reconstruction_valid": pre_valid,
            }

            # Output contract: reconstructed values (Req 11.1 — paired after fields)
            output_contract = {
                "entry_price": entry,  # entry typically unchanged
                "stop_price": adjusted_stop,
                "target_price": target,  # target typically unchanged
                "quantity": adjusted_quantity,
                "reward_to_risk": str(geometry_after.reward_to_risk),
                "dollar_risk": str(geometry_after.total_dollar_risk),
                "reconstruction_outcome": outcome,
                "gate_decision": decision_str,
                "gate_reason_code": gate_result.get("reason_code", ""),
            }

            # If reconstruction introduced a new defect (Req 11.4)
            if outcome == "reconstruction_introduced_defect":
                output_contract["defect_info"] = {
                    "defect_type": "geometry_invalidated",
                    "fields_affected": [e.field_name for e in geometry_after.validation_errors],
                    "pre_reconstruction_validity": pre_valid,
                }

            fields_changed = []
            if adjusted_stop != stop:
                fields_changed.append("stop_price")
            if adjusted_quantity != original_quantity:
                fields_changed.append("quantity")

            chain.record_event(
                stage_name="gate_reconstruction",
                stage_version="1.0",
                input_contract=input_contract,
                output_contract=output_contract,
                fields_changed=fields_changed,
                mutation_reason_code=f"gate_reconstruction:{outcome}",
                rule_id=gate_result.get("rule_name"),
                geometry_before=geometry_before,
                geometry_after=geometry_after,
            )

        # Persist the additional gate reconstruction event (separate INSERT — Req 11.1)
        if engine is not None:
            persist_provenance_chain(engine, chain)

    except Exception:
        logger.error(
            "Gate reconstruction provenance failed for %s",
            resolved_order.candidate_id,
            exc_info=True,
        )


def execute_candidate_pipeline(
    db,
    engine,
    registry: CandidateRegistry,
    decision: CandidateDecision,
    portfolio: dict,
    profile: dict,
    profile_id: str,
    *,
    recovery_multiplier: float = 1.0,
    provenance_chain: ProvenanceChain | None = None,
    parse_violations: list[dict] | None = None,
    checkpoint_logger: 'CheckpointLogger | None' = None,
) -> PipelineResult:
    """Authoritative single-pass execution pipeline.

    Steps (exactly once, no double gate evaluation):
    1. RESOLVE: Look up candidate in registry, validate state/profile/cycle
    2. RESERVE: Generate execution_key, CAS transition registered → reserved
    3. SIZE: Calculate deterministic position size
    4. GATES: Run full gate pipeline
    5. EXECUTE: Create trade record with execution_key

    Returns PipelineResult with outcome and all intermediate data.
    """
    candidate_id = decision.candidate_id

    # ── Step 0: ADJUSTMENT DEFERRAL ──
    # Valid "adjust" decisions are logged and rejected — adjustment execution
    # is deferred to a follow-up phase (Requirements 5.1, 5.3).
    if decision.decision == "adjust":
        adj_type = None
        if isinstance(decision.adjustment_request, dict):
            adj_type = decision.adjustment_request.get("type")
        logger.info(
            "Adjustment request deferred for candidate %s: type=%s, "
            "reason=adjustment_not_yet_enabled",
            candidate_id,
            adj_type,
        )
        if checkpoint_logger is not None:
            try:
                from utils.checkpoint_logger import CheckpointEvent
                checkpoint_logger.emit_outcome("pm_accepted_and_gate_rejected", CheckpointEvent(
                    stage="pm_candidate_rejected",
                    cycle_id=registry.cycle_id,
                    profile=profile_id,
                    candidate_id=candidate_id,
                    decision="rejected",
                    reason_code="adjustment_deferred",
                    metadata={"adjustment_type": adj_type},
                ))
            except Exception:
                logger.error(
                    "Checkpoint emission failed at adjustment_deferred for %s",
                    candidate_id,
                    exc_info=True,
                )
        return PipelineResult(
            candidate_id=candidate_id,
            outcome="adjustment_deferred",
            error="adjustment_not_yet_enabled",
        )

    # ── Step 0b: CONTRACT VIOLATION CHECK ──
    # If upstream parsing detected prohibited fields for this candidate,
    # reject immediately (Requirements 1.3, 3.3).
    if parse_violations:
        matching = [
            v for v in parse_violations
            if v.get("type") == "PROHIBITED_FIELD"
            and v.get("candidate_id") == candidate_id
        ]
        if matching:
            fields = [v.get("field") for v in matching]
            error_msg = f"Contract violation: prohibited fields {fields}"
            logger.warning(
                "Pipeline rejected %s: contract violation, prohibited fields=%s",
                candidate_id,
                fields,
            )
            if checkpoint_logger is not None:
                try:
                    from utils.checkpoint_logger import CheckpointEvent
                    checkpoint_logger.emit_outcome("pm_contract_violation", CheckpointEvent(
                        stage="order_rejected",
                        cycle_id=registry.cycle_id,
                        profile=profile_id,
                        candidate_id=candidate_id,
                        decision="reject",
                        reason_code=f"prohibited_fields:{fields}",
                    ))
                except Exception:
                    logger.error("Checkpoint emission failed at contract_violation for %s", candidate_id, exc_info=True)
            return PipelineResult(
                candidate_id=candidate_id,
                outcome="contract_violation",
                error=error_msg,
            )

    # ── Step 1: RESOLVE ──
    candidate, error = _resolve_candidate(registry, decision, profile_id)
    if candidate is None:
        logger.warning(
            "Pipeline resolve failed for %s: %s", candidate_id, error
        )

        # ── Provenance: record terminal event for candidate_mismatch ──
        if PM_PROVENANCE_MODE != "disabled" and provenance_chain is not None:
            try:
                provenance_chain.record_terminal(
                    stage_name="candidate_mismatch",
                    stage_version="1.0",
                    reason=error or "candidate_not_found",
                )
            except Exception:
                logger.error(
                    "Provenance capture failed at terminal for %s",
                    candidate_id,
                    exc_info=True,
                )

        if checkpoint_logger is not None:
            try:
                from utils.checkpoint_logger import CheckpointEvent
                checkpoint_logger.emit_outcome("invalid_stale_candidate_id", CheckpointEvent(
                    stage="order_rejected",
                    cycle_id=registry.cycle_id,
                    profile=profile_id,
                    candidate_id=candidate_id,
                    decision="reject",
                    reason_code=error or "candidate_not_found",
                ))
            except Exception:
                logger.error("Checkpoint emission failed at reservation_failed for %s", candidate_id, exc_info=True)

        return PipelineResult(
            candidate_id=candidate_id,
            outcome="reservation_failed",
            error=error,
        )

    # ── Provenance: candidate resolution ──
    if PM_PROVENANCE_MODE != "disabled" and provenance_chain is not None:
        try:
            from utils.geometry_calculator import compute_geometry
            from utils.candidate_fidelity import check_candidate_fidelity

            # Geometry before resolution: incomplete (no geometry known yet)
            geometry_before = compute_geometry(None, None, None, None, None)

            # Geometry after resolution: computed from resolved candidate fields
            geometry_after = compute_geometry(
                candidate.direction,
                candidate.entry_price,
                candidate.stop_price,
                candidate.target_price,
                1,  # quantity not yet determined at resolution stage
            )

            # Input contract: what the PM decision referenced
            input_contract = {
                "candidate_id": decision.candidate_id,
                "decision": decision.decision,
                "risk_multiplier": decision.risk_multiplier,
                "rationale": decision.rationale,
            }

            # Output contract: resolved candidate fields from registry
            output_contract = {
                "candidate_id": candidate.candidate_id,
                "symbol": candidate.symbol,
                "direction": candidate.direction,
                "entry_price": candidate.entry_price,
                "stop_price": candidate.stop_price,
                "target_price": candidate.target_price,
                "setup_type": candidate.setup_type,
                "risk_reward": candidate.risk_reward,
                "profile_id": candidate.profile_id,
                "cycle_id": candidate.cycle_id,
            }

            provenance_chain.record_event(
                stage_name="candidate_resolution",
                stage_version="1.0",
                input_contract=input_contract,
                output_contract=output_contract,
                fields_changed=[
                    "symbol", "direction", "entry_price", "stop_price",
                    "target_price", "setup_type",
                ],
                mutation_reason_code="candidate_resolved",
                rule_id=None,
                geometry_before=geometry_before,
                geometry_after=geometry_after,
            )

            # Check candidate fidelity (Req 6.1–6.6)
            trusted_candidate = {
                "candidate_id": candidate.candidate_id,
                "symbol": candidate.symbol,
                "direction": candidate.direction,
                "setup_type": candidate.setup_type,
                "entry_price": candidate.entry_price,
                "stop_price": candidate.stop_price,
                "target_price": candidate.target_price,
            }
            pm_decision_dict = {
                "candidate_id": decision.candidate_id,
                "symbol": candidate.symbol,
                "direction": candidate.direction,
                "setup_type": candidate.setup_type,
                "entry_price": candidate.entry_price,
                "stop_price": candidate.stop_price,
                "target_price": candidate.target_price,
            }
            # In candidate-ID mode, PM doesn't supply prices — they come from registry.
            # Fidelity check validates the PM referenced the correct candidate.
            supplied_ids = [candidate.candidate_id]
            fidelity_result = check_candidate_fidelity(
                pm_decision=pm_decision_dict,
                trusted_candidate=trusted_candidate,
                supplied_candidate_ids=supplied_ids,
            )

            if fidelity_result.field_differences:
                logger.info(
                    "Candidate fidelity findings for %s: classification=%s, diffs=%d",
                    candidate_id,
                    fidelity_result.classification,
                    len(fidelity_result.field_differences),
                )

            # Invalidate any previously recorded claimed_reward_to_risk
            # The geometry_after now contains the authoritative computed reward_to_risk
            # from the registry — any prior PM claim is superseded.

        except Exception:
            logger.error(
                "Provenance capture failed at resolution for %s",
                candidate_id,
                exc_info=True,
            )

    # ── Audit: PM geometry discrepancy logging (Requirement 2.3) ──
    if parse_violations:
        pm_geometry_violations = [
            v for v in parse_violations
            if v.get("type") == "PROHIBITED_FIELD"
            and v.get("candidate_id") == candidate_id
            and v.get("pm_value") is not None
        ]
        if pm_geometry_violations:
            # Map prohibited field names to registry field names
            _FIELD_MAP = {
                "entry_price": "entry_price",
                "stop": "stop_price",
                "stop_loss": "stop_price",
                "target": "target_price",
                "target_price": "target_price",
                "symbol": "symbol",
                "quantity": None,  # no direct registry equivalent
            }
            for v in pm_geometry_violations:
                field = v["field"]
                pm_val = v["pm_value"]
                registry_field = _FIELD_MAP.get(field)
                if registry_field:
                    registry_val = getattr(candidate, registry_field, None)
                    if registry_val is not None and str(pm_val) != str(registry_val):
                        logger.info(
                            "PM geometry discrepancy: candidate=%s field=%s pm_value=%s registry_value=%s",
                            candidate_id,
                            field,
                            pm_val,
                            registry_val,
                        )

    # ── Step 2: RESERVE ──
    execution_key = _generate_execution_key(
        candidate_id, registry.cycle_id, profile_id
    )

    success, reason = registry.reserve(candidate_id, execution_key)
    if not success:
        logger.warning(
            "Pipeline reservation failed for %s: %s", candidate_id, reason
        )
        if checkpoint_logger is not None:
            try:
                from utils.checkpoint_logger import CheckpointEvent
                checkpoint_logger.emit_outcome("invalid_stale_candidate_id", CheckpointEvent(
                    stage="order_rejected",
                    cycle_id=registry.cycle_id,
                    profile=profile_id,
                    candidate_id=candidate_id,
                    symbol=candidate.symbol,
                    setup_type=candidate.setup_type,
                    decision="reject",
                    reason_code=reason or "reservation_failed",
                ))
            except Exception:
                logger.error("Checkpoint emission failed at reservation_failed for %s", candidate_id, exc_info=True)
        return PipelineResult(
            candidate_id=candidate_id,
            outcome="reservation_failed",
            error=reason,
        )

    # ── Step 3: SIZE ──
    resolved_order = _build_resolved_order(
        candidate, decision, execution_key, profile_id
    )

    sizing_result = calculate_position_size(
        resolved_order,
        portfolio,
        profile,
        profile_id,
        recovery_multiplier=recovery_multiplier,
    )

    if sizing_result.rejected:
        _record_pipeline_shadow_block(
            db,
            resolved_order,
            sizing_result,
            outcome="sizing_rejected",
            block_reason=sizing_result.rejection_reason or "Position sizing rejected",
            blocked_by="position_sizer",
            reason_code="sizing_rejected",
        )
        _commit_candidate_pipeline_session(db, candidate_id, "sizing_rejected")
        registry.mark_sizing_rejected(candidate_id, sizing_result.rejection_reason)
        logger.info(
            "Pipeline sizing rejected for %s: %s",
            candidate_id,
            sizing_result.rejection_reason,
        )
        if checkpoint_logger is not None:
            try:
                from utils.checkpoint_logger import CheckpointEvent
                checkpoint_logger.emit_outcome("pm_accepted_and_gate_rejected", CheckpointEvent(
                    stage="order_rejected",
                    cycle_id=registry.cycle_id,
                    profile=profile_id,
                    candidate_id=candidate_id,
                    symbol=candidate.symbol,
                    setup_type=candidate.setup_type,
                    decision="reject",
                    reason_code=sizing_result.rejection_reason,
                ))
            except Exception:
                logger.error("Checkpoint emission failed at sizing_rejected for %s", candidate_id, exc_info=True)
        return PipelineResult(
            candidate_id=candidate_id,
            outcome="sizing_rejected",
            resolved_order=resolved_order,
            sizing_result=sizing_result,
            error=sizing_result.rejection_reason,
        )

    # ── Checkpoint: order_materialized — sizing succeeded (Req 11.1, 11.3) ──
    if checkpoint_logger is not None:
        try:
            from utils.checkpoint_logger import CheckpointEvent
            checkpoint_logger.emit(CheckpointEvent(
                stage="order_materialized",
                cycle_id=registry.cycle_id,
                profile=profile_id,
                candidate_id=candidate_id,
                symbol=candidate.symbol,
                setup_type=candidate.setup_type,
                decision="advance",
            ))
        except Exception:
            logger.error("Checkpoint emission failed at order_materialized for %s", candidate_id, exc_info=True)

    # ── Provenance: behavioral adjustment (risk_multiplier application) ──
    if PM_PROVENANCE_MODE != "disabled":
        try:
            geometry_valid = record_behavioral_adjustment_provenance(
                chain=provenance_chain,
                resolved_order=resolved_order,
                sizing_result=sizing_result,
                risk_multiplier=resolved_order.risk_multiplier,
                profile_id=profile_id,
            )
            if not geometry_valid:
                # Behavioral adjustment created invalid geometry — do NOT advance (Req 8.6)
                error_msg = "behavioral_adjustment_invalid"
                _record_pipeline_shadow_block(
                    db,
                    resolved_order,
                    sizing_result,
                    outcome="sizing_rejected",
                    block_reason=error_msg,
                    blocked_by="behavioral_adjustment",
                    reason_code="behavioral_adjustment_invalid",
                )
                _commit_candidate_pipeline_session(db, candidate_id, "behavioral_adjustment_invalid")
                registry.mark_sizing_rejected(candidate_id, error_msg)
                logger.warning(
                    "Pipeline behavioral adjustment invalid for %s: "
                    "risk_multiplier=%s produced invalid geometry",
                    candidate_id,
                    resolved_order.risk_multiplier,
                )
                return PipelineResult(
                    candidate_id=candidate_id,
                    outcome="sizing_rejected",
                    resolved_order=resolved_order,
                    sizing_result=sizing_result,
                    error=error_msg,
                )
        except Exception:
            # Fail-open: provenance must never block the pipeline
            logger.error(
                "Provenance behavioral adjustment capture failed for %s",
                candidate_id,
                exc_info=True,
            )

    # ── Provenance: pre-gate snapshot and structural validation (Req 9.1–9.7) ──
    if PM_PROVENANCE_MODE != "disabled":
        try:
            is_structurally_valid = record_pre_gate_snapshot_provenance(
                chain=provenance_chain,
                resolved_order=resolved_order,
                quantity=sizing_result.quantity,
                engine=engine,
            )
            if not is_structurally_valid:
                error_msg = "pre_gate_contract_invalid"
                _record_pipeline_shadow_block(
                    db,
                    resolved_order,
                    sizing_result,
                    outcome="gate_rejected",
                    block_reason=error_msg,
                    blocked_by="pre_gate_contract",
                    reason_code="pre_gate_contract_invalid",
                )
                _commit_candidate_pipeline_session(db, candidate_id, "pre_gate_contract_invalid")
                registry.mark_gate_rejected(candidate_id, error_msg)
                logger.warning(
                    "Pipeline pre-gate contract invalid for %s: "
                    "structural geometry validation failed",
                    candidate_id,
                )
                return PipelineResult(
                    candidate_id=candidate_id,
                    outcome="gate_rejected",
                    resolved_order=resolved_order,
                    sizing_result=sizing_result,
                    gate_notes=error_msg,
                    error=error_msg,
                )
        except Exception:
            # Fail-open: provenance must never block the pipeline
            logger.error(
                "Pre-gate snapshot provenance failed for %s",
                candidate_id,
                exc_info=True,
            )

    # ── Step 4: GATES ──
    gate_decision = _build_gate_decision(resolved_order, sizing_result.quantity)

    try:
        from agents.portfolio_manager import _run_gate_pipeline

        proceed, gate_notes_list, gate_multiplier, multiplier_breakdown = (
            _run_gate_pipeline(db, engine, gate_decision, resolved_order.source_signal, profile_id)
        )
    except Exception as exc:
        # Gate pipeline failure — fail closed
        error_msg = f"Gate pipeline error: {exc}"
        logger.error("Pipeline gate error for %s: %s", candidate_id, exc)
        _record_pipeline_shadow_block(
            db,
            resolved_order,
            sizing_result,
            outcome="gate_rejected",
            block_reason=error_msg,
            blocked_by="gate_pipeline",
            reason_code="gate_pipeline_error",
            gate_notes=error_msg,
        )
        _commit_candidate_pipeline_session(db, candidate_id, "gate_pipeline_error")
        registry.mark_gate_rejected(candidate_id, error_msg)
        if checkpoint_logger is not None:
            try:
                from utils.checkpoint_logger import CheckpointEvent
                checkpoint_logger.emit_outcome("pm_accepted_and_gate_rejected", CheckpointEvent(
                    stage="gate_evaluated",
                    cycle_id=registry.cycle_id,
                    profile=profile_id,
                    candidate_id=candidate_id,
                    symbol=candidate.symbol,
                    setup_type=candidate.setup_type,
                    decision="reject",
                    reason_code=error_msg,
                ))
            except Exception:
                logger.error("Checkpoint emission failed at gate_evaluated for %s", candidate_id, exc_info=True)
        return PipelineResult(
            candidate_id=candidate_id,
            outcome="gate_rejected",
            resolved_order=resolved_order,
            sizing_result=sizing_result,
            gate_notes=error_msg,
            error=error_msg,
        )

    if not proceed:
        # Extract rejection reasons from gate notes
        rejection_reasons = "; ".join(
            n.get("reason", "")
            for n in gate_notes_list
            if n.get("decision") in ("reject", "rejected", "override_required", "block")
        )
        gate_notes_str = rejection_reasons or "Gate pipeline rejected"
        blocked_by, reason_code = _extract_rejecting_gate(gate_notes_list)
        _record_pipeline_shadow_block(
            db,
            resolved_order,
            sizing_result,
            outcome="gate_rejected",
            block_reason=gate_notes_str,
            blocked_by=blocked_by,
            reason_code=reason_code or "gate_rejected",
            gate_notes=gate_notes_list,
        )
        _commit_candidate_pipeline_session(db, candidate_id, "gate_rejected")
        registry.mark_gate_rejected(candidate_id, gate_notes_str)
        logger.info(
            "Pipeline gate rejected for %s: %s", candidate_id, gate_notes_str
        )
        if checkpoint_logger is not None:
            try:
                from utils.checkpoint_logger import CheckpointEvent
                checkpoint_logger.emit_outcome("pm_accepted_and_gate_rejected", CheckpointEvent(
                    stage="gate_evaluated",
                    cycle_id=registry.cycle_id,
                    profile=profile_id,
                    candidate_id=candidate_id,
                    symbol=candidate.symbol,
                    setup_type=candidate.setup_type,
                    decision="reject",
                    reason_code=gate_notes_str,
                ))
            except Exception:
                logger.error("Checkpoint emission failed at gate_evaluated for %s", candidate_id, exc_info=True)
        return PipelineResult(
            candidate_id=candidate_id,
            outcome="gate_rejected",
            resolved_order=resolved_order,
            sizing_result=sizing_result,
            gate_notes=gate_notes_str,
        )

    # Apply gate multiplier to quantity if gates reduced size
    final_quantity = sizing_result.quantity
    if gate_multiplier < 1.0 and final_quantity > 0:
        final_quantity = max(1, int(final_quantity * gate_multiplier))
        gate_decision["quantity"] = final_quantity

    # ── Checkpoint: gate_evaluated — gates passed (Req 11.1, 11.3) ──
    if checkpoint_logger is not None:
        try:
            from utils.checkpoint_logger import CheckpointEvent
            checkpoint_logger.emit(CheckpointEvent(
                stage="gate_evaluated",
                cycle_id=registry.cycle_id,
                profile=profile_id,
                candidate_id=candidate_id,
                symbol=candidate.symbol,
                setup_type=candidate.setup_type,
                decision="advance",
            ))
        except Exception:
            logger.error("Checkpoint emission failed at gate_evaluated for %s", candidate_id, exc_info=True)

    # ── Provenance: gate reconstruction (append-only, post-gate — Req 11.1) ──
    if PM_PROVENANCE_MODE != "disabled" and provenance_chain is not None:
        try:
            # Extract the risk_geometry_gate result from gate notes
            rg_gate_result = None
            for note in gate_notes_list:
                if note.get("gate") == "risk_geometry_gate":
                    rg_gate_result = note
                    break
            if rg_gate_result is not None:
                record_gate_reconstruction_provenance(
                    chain=provenance_chain,
                    resolved_order=resolved_order,
                    gate_result=rg_gate_result,
                    original_quantity=sizing_result.quantity,
                    engine=engine,
                )
        except Exception:
            logger.error(
                "Gate reconstruction provenance failed for %s",
                candidate_id,
                exc_info=True,
            )

    # ── Step 5: EXECUTE ──
    try:
        from agents.portfolio_manager import execute_trade

        success_exec, exec_msg = execute_trade(
            db, gate_decision, profile_id, normalized=True
        )
    except Exception as exc:
        error_msg = f"Execution error: {exc}"
        logger.error("Pipeline execution error for %s: %s", candidate_id, exc)
        # Best-effort state transition — finalize_cycle() is the hard guarantee
        try:
            registry.mark_execution_failed(candidate_id, error_msg[:500])
        except Exception as mark_exc:
            logger.error(
                "mark_execution_failed raised for %s (finalize_cycle will enforce): %s",
                candidate_id, mark_exc
            )
        if checkpoint_logger is not None:
            try:
                from utils.checkpoint_logger import CheckpointEvent
                checkpoint_logger.emit_outcome("pm_accepted_and_gate_rejected", CheckpointEvent(
                    stage="order_rejected",
                    cycle_id=registry.cycle_id,
                    profile=profile_id,
                    candidate_id=candidate_id,
                    symbol=candidate.symbol,
                    setup_type=candidate.setup_type,
                    decision="reject",
                    reason_code=error_msg,
                ))
            except Exception:
                logger.error("Checkpoint emission failed at execution_failed for %s", candidate_id, exc_info=True)
        return PipelineResult(
            candidate_id=candidate_id,
            outcome="execution_failed",
            resolved_order=resolved_order,
            sizing_result=sizing_result,
            error=error_msg,
        )

    if not success_exec:
        logger.warning(
            "Pipeline execution failed for %s: %s", candidate_id, exec_msg
        )
        # Best-effort state transition — finalize_cycle() is the hard guarantee
        try:
            registry.mark_execution_failed(candidate_id, (exec_msg or "")[:1024])
        except Exception as mark_exc:
            logger.error(
                "mark_execution_failed raised for %s (finalize_cycle will enforce): %s",
                candidate_id, mark_exc
            )
        if checkpoint_logger is not None:
            try:
                from utils.checkpoint_logger import CheckpointEvent
                checkpoint_logger.emit_outcome("pm_accepted_and_gate_rejected", CheckpointEvent(
                    stage="order_rejected",
                    cycle_id=registry.cycle_id,
                    profile=profile_id,
                    candidate_id=candidate_id,
                    symbol=candidate.symbol,
                    setup_type=candidate.setup_type,
                    decision="reject",
                    reason_code=exec_msg,
                ))
            except Exception:
                logger.error("Checkpoint emission failed at execution_failed for %s", candidate_id, exc_info=True)
        return PipelineResult(
            candidate_id=candidate_id,
            outcome="execution_failed",
            resolved_order=resolved_order,
            sizing_result=sizing_result,
            error=exec_msg,
        )

    # Success — mark executed in registry
    registry.mark_executed(candidate_id)
    logger.info("Pipeline executed candidate %s successfully", candidate_id)

    # ── Checkpoint: order_fired — execution succeeded (Req 11.1, 11.3, 9.1) ──
    if checkpoint_logger is not None:
        try:
            from utils.checkpoint_logger import CheckpointEvent
            checkpoint_logger.emit_outcome("pm_accepted_and_executed", CheckpointEvent(
                stage="order_fired",
                cycle_id=registry.cycle_id,
                profile=profile_id,
                candidate_id=candidate_id,
                symbol=candidate.symbol,
                setup_type=candidate.setup_type,
                decision="executed",
            ))
        except Exception:
            logger.error("Checkpoint emission failed at order_fired for %s", candidate_id, exc_info=True)

    return PipelineResult(
        candidate_id=candidate_id,
        outcome="executed",
        resolved_order=resolved_order,
        sizing_result=sizing_result,
    )


def dry_run_candidate_pipeline(
    db,
    engine,
    registry: CandidateRegistry,
    decision: CandidateDecision,
    portfolio_snapshot: dict,
    profile: dict,
    profile_id: str,
    *,
    recovery_multiplier: float = 1.0,
    provenance_chain: ProvenanceChain | None = None,
) -> PipelineResult:
    """Shadow-mode dry-run pipeline.

    Identical logic to execute_candidate_pipeline EXCEPT:
    - Does NOT call reserve() (no state mutation)
    - Does NOT execute trades (no balance/position mutation)
    - Does NOT emit real execution events
    - Records hypothetical sizing and gate results
    - Evaluates against the provided frozen portfolio_snapshot

    Returns a PipelineResult with hypothetical outcome for comparison.
    """
    candidate_id = decision.candidate_id

    # ── Step 1: RESOLVE (same as authoritative) ──
    candidate, error = _resolve_candidate(registry, decision, profile_id)
    if candidate is None:
        return PipelineResult(
            candidate_id=candidate_id,
            outcome="reservation_failed",
            error=error,
        )

    # ── NO RESERVE — dry run does not mutate state ──
    execution_key = _generate_execution_key(
        candidate_id, registry.cycle_id, profile_id
    )

    # ── Step 3: SIZE (against frozen snapshot) ──
    resolved_order = _build_resolved_order(
        candidate, decision, execution_key, profile_id
    )

    sizing_result = calculate_position_size(
        resolved_order,
        portfolio_snapshot,
        profile,
        profile_id,
        recovery_multiplier=recovery_multiplier,
    )

    if sizing_result.rejected:
        return PipelineResult(
            candidate_id=candidate_id,
            outcome="sizing_rejected",
            resolved_order=resolved_order,
            sizing_result=sizing_result,
            error=sizing_result.rejection_reason,
        )

    # ── Step 4: GATES (hypothetical evaluation) ──
    gate_decision = _build_gate_decision(resolved_order, sizing_result.quantity)

    try:
        from agents.portfolio_manager import _run_gate_pipeline

        proceed, gate_notes_list, gate_multiplier, multiplier_breakdown = (
            _run_gate_pipeline(db, engine, gate_decision, resolved_order.source_signal, profile_id)
        )
    except Exception as exc:
        error_msg = f"Gate pipeline error: {exc}"
        return PipelineResult(
            candidate_id=candidate_id,
            outcome="gate_rejected",
            resolved_order=resolved_order,
            sizing_result=sizing_result,
            gate_notes=error_msg,
            error=error_msg,
        )

    if not proceed:
        rejection_reasons = "; ".join(
            n.get("reason", "")
            for n in gate_notes_list
            if n.get("decision") in ("reject", "rejected", "override_required", "block")
        )
        gate_notes_str = rejection_reasons or "Gate pipeline rejected"
        return PipelineResult(
            candidate_id=candidate_id,
            outcome="gate_rejected",
            resolved_order=resolved_order,
            sizing_result=sizing_result,
            gate_notes=gate_notes_str,
        )

    # ── NO EXECUTE — dry run stops here with hypothetical "executed" outcome ──
    return PipelineResult(
        candidate_id=candidate_id,
        outcome="executed",
        resolved_order=resolved_order,
        sizing_result=sizing_result,
    )


# ── Missing geometry claim detection (Requirement 4.2) ─────────────────────

# Phrases that indicate the PM is claiming missing geometry in rejection rationale
_MISSING_GEOMETRY_PHRASES = [
    "missing entry", "no entry", "no entry price",
    "missing stop", "no stop", "no stop price", "without stop",
    "missing target", "no target", "no target price", "without target",
    "missing geometry", "incomplete geometry",
    "no risk reward", "no r:r", "missing r:r",
]


def _detect_missing_geometry_claim(
    engine,
    candidate_id: str,
    cycle_id: str,
    profile_id: str,
    rationale: str | None,
) -> None:
    """Detect and emit event when PM claims missing geometry on preflight-passed candidate.

    When a PM rejects a candidate that passed preflight (meaning geometry IS valid),
    and the rationale contains phrases claiming missing geometry, emit a contract
    violation event.

    Fail-open: logs and continues on error.
    Requirements: 4.2
    """
    try:
        if not rationale:
            return
        rationale_lower = rationale.lower()
        for phrase in _MISSING_GEOMETRY_PHRASES:
            if phrase in rationale_lower:
                event_data = {
                    "candidate_id": candidate_id,
                    "rationale": str(rationale)[:2000],
                    "matched_phrase": phrase,
                }
                _write_candidate_event(
                    engine,
                    candidate_id,
                    cycle_id,
                    profile_id,
                    "contract_violation_missing_geometry_claim",
                    event_data,
                )
                break  # Only emit one event even if multiple phrases match
    except Exception:
        logger.error(
            "Failed to detect missing geometry claim for candidate %s",
            candidate_id,
            exc_info=True,
        )
