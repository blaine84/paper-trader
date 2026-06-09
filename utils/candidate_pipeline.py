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
from utils.position_sizer import SizingResult, calculate_position_size

if TYPE_CHECKING:
    from typing import Any

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

    # ── Step 1: RESOLVE ──
    candidate, error = _resolve_candidate(registry, decision, profile_id)
    if candidate is None:
        logger.warning(
            "Pipeline resolve failed for %s: %s", candidate_id, error
        )
        return PipelineResult(
            candidate_id=candidate_id,
            outcome="reservation_failed",
            error=error,
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
        registry.mark_sizing_rejected(candidate_id, sizing_result.rejection_reason)
        logger.info(
            "Pipeline sizing rejected for %s: %s",
            candidate_id,
            sizing_result.rejection_reason,
        )
        return PipelineResult(
            candidate_id=candidate_id,
            outcome="sizing_rejected",
            resolved_order=resolved_order,
            sizing_result=sizing_result,
            error=sizing_result.rejection_reason,
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
        registry.mark_gate_rejected(candidate_id, error_msg)
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
        registry.mark_gate_rejected(candidate_id, gate_notes_str)
        logger.info(
            "Pipeline gate rejected for %s: %s", candidate_id, gate_notes_str
        )
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

    # ── Step 5: EXECUTE ──
    try:
        from agents.portfolio_manager import execute_trade

        success_exec, exec_msg = execute_trade(
            db, gate_decision, profile_id, normalized=True
        )
    except Exception as exc:
        error_msg = f"Execution error: {exc}"
        logger.error("Pipeline execution error for %s: %s", candidate_id, exc)
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
