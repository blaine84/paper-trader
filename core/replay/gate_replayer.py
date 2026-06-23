"""Deterministic gate evaluation — core gate sequence replay.

Executes the gate pipeline using the ReplayGateAdapter layer. Each gate is
wrapped in an adapter that normalizes its interface. Supports two modes:
- Production mode: stops at first hard rejection (reject/block/override_required)
- Diagnostic mode: evaluates ALL gates, labels downstream as non_executable

No globals are mutated. Safe to run in-process alongside production.

See: design.md §core/replay/gate_replayer.py, requirements §5.1–§5.8
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from core.replay.gate_adapter import (
    GATE_REQUIRED_FIELDS,
    GatePolicyConfig,
    ReplayGateAdapter,
    ReplayGateContext,
    build_deterministic_id_provider,
    build_replay_clock,
)
from core.replay.extended_gate_adapter import (
    EXTENDED_GATE_REQUIRED_FIELDS,
    EXTENDED_GATE_SEQUENCE,
    ExtendedGateAdapter,
)
from core.replay.policy_version import PolicyVersion


# ---------------------------------------------------------------------------
# Data models — frozen trace entries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateTraceEntry:
    """A single gate's evaluation result within the replay trace.

    Fields:
        gate_name: Identifier of the gate evaluated
        decision: Canonical decision (allow|reject|reduce_size|override_required|warn|error)
        reason_code: Gate-specific reason code for the decision
        threshold_applied: Thresholds that were active for this gate evaluation
        input_fields: Material input field names and values consumed by the gate
        adjusted_values: Any adjusted values produced (e.g. widened stop, reduced quantity)
        cumulative_size_multiplier: Cumulative size multiplier after this gate
        is_non_executable: True when run after earlier rejection in diagnostic mode
        missing_fields: Required fields that are null/missing but would alter behavior
    """

    gate_name: str
    decision: str
    reason_code: str
    threshold_applied: dict = field(default_factory=dict)
    input_fields: dict = field(default_factory=dict)
    adjusted_values: dict | None = None
    cumulative_size_multiplier: float = 1.0
    is_non_executable: bool = False
    missing_fields: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GateTrace:
    """Complete ordered record of gate evaluation results for a replay.

    Fields:
        entries: Ordered list of per-gate trace entries
        final_decision: Overall replay decision ("allow" or "reject")
        final_gate: Gate that produced the final decision (None if all gates allow)
        final_reason_code: Reason code from the final decision gate
        diagnostic_mode: Whether this trace was produced in diagnostic mode
        policy_version: The PolicyVersion used for this replay
    """

    entries: list[GateTraceEntry]
    final_decision: str
    final_gate: str | None
    final_reason_code: str | None
    diagnostic_mode: bool
    policy_version: PolicyVersion


# ---------------------------------------------------------------------------
# Gate sequences
# ---------------------------------------------------------------------------

# Phase 1: Core gate sequence (inside _run_gate_pipeline)
CORE_GATE_SEQUENCE: list[str] = [
    "setup_quality_gate",
    "pre_trade_quality_gate",
    "catalyst_specificity_gate",
    "risk_geometry_gate",
]

# Hard rejection decisions — stops pipeline evaluation in production mode
HARD_REJECTION_DECISIONS: frozenset[str] = frozenset({
    "reject",
    "block",
    "override_required",
})


# ---------------------------------------------------------------------------
# Main replay function
# ---------------------------------------------------------------------------


def replay_gates(
    context: ReplayGateContext,
    policy_config: GatePolicyConfig,
    policy_version: PolicyVersion,
    *,
    replay_id: str = "replay",
    candidate_id: str = "candidate",
    cutoff: datetime | None = None,
    diagnostic_mode: bool = False,
    include_extended: bool = False,  # Phase 2: include extended boundary
) -> GateTrace:
    """Evaluate all gates in production order via ReplayGateAdapter.

    In production mode: stops at first hard rejection.
    In diagnostic mode: evaluates all gates, labels downstream as non-executable.

    Each gate is called with explicit dependency injection:
    - policy=GatePolicyConfig (no module-constant reads)
    - event_sink=noop (no log_trade_event calls)
    - clock=replay_clock (deterministic time)
    - id_provider=deterministic_id (reproducible operational IDs)

    No globals are mutated. Safe to run in-process alongside production.

    Args:
        context: ReplayGateContext with all captured historical inputs
        policy_config: Frozen policy configuration for gate thresholds
        policy_version: PolicyVersion identifying the policy bundle
        replay_id: Unique replay execution identifier (for deterministic IDs)
        candidate_id: Candidate identifier (for deterministic IDs)
        cutoff: Replay cutoff timestamp for the frozen clock (defaults to utcnow)
        diagnostic_mode: If True, evaluate ALL gates even after rejection
        include_extended: If True, include extended decision boundary (Phase 2)

    Returns:
        GateTrace with ordered entries, final decision, and metadata
    """
    if cutoff is None:
        cutoff = datetime.utcnow()

    # Build the gate sequence
    gate_sequence = list(CORE_GATE_SEQUENCE)
    if include_extended:
        gate_sequence.extend(EXTENDED_GATE_SEQUENCE)

    # Build frozen clock for deterministic time
    replay_clock = build_replay_clock(cutoff)

    entries: list[GateTraceEntry] = []
    hard_rejection_hit = False
    final_decision = "allow"
    final_gate: str | None = None
    final_reason_code: str | None = None
    cumulative_size_multiplier: float = 1.0

    for gate_name in gate_sequence:
        # Build deterministic ID provider for this gate
        id_provider = build_deterministic_id_provider(replay_id, gate_name, candidate_id)

        if hard_rejection_hit:
            if diagnostic_mode:
                # Diagnostic mode: evaluate anyway, mark as non-executable
                entry = _evaluate_gate_safe(
                    gate_name=gate_name,
                    context=context,
                    policy_config=policy_config,
                    replay_clock=replay_clock,
                    id_provider=id_provider,
                    cumulative_size_multiplier=cumulative_size_multiplier,
                    is_non_executable=True,
                    diagnostic_mode=True,
                )
                entries.append(entry)
            else:
                # Production mode: stop evaluating (already broke out above)
                break
        else:
            # Normal evaluation
            entry = _evaluate_gate_safe(
                gate_name=gate_name,
                context=context,
                policy_config=policy_config,
                replay_clock=replay_clock,
                id_provider=id_provider,
                cumulative_size_multiplier=cumulative_size_multiplier,
                is_non_executable=False,
                diagnostic_mode=diagnostic_mode,
            )
            entries.append(entry)

            # Track cumulative size multiplier adjustments
            cumulative_size_multiplier = entry.cumulative_size_multiplier

            # Check for hard rejection
            if entry.decision in HARD_REJECTION_DECISIONS:
                hard_rejection_hit = True
                final_decision = "reject"
                final_gate = gate_name
                final_reason_code = entry.reason_code

                if not diagnostic_mode:
                    # Production mode: stop at first hard rejection
                    break

            elif entry.decision == "error":
                # Gate exception in production mode → unscorable, stop
                hard_rejection_hit = True
                final_decision = "reject"
                final_gate = gate_name
                final_reason_code = entry.reason_code

                if not diagnostic_mode:
                    break

    return GateTrace(
        entries=entries,
        final_decision=final_decision,
        final_gate=final_gate,
        final_reason_code=final_reason_code,
        diagnostic_mode=diagnostic_mode,
        policy_version=policy_version,
    )


# ---------------------------------------------------------------------------
# Single gate evaluation (public)
# ---------------------------------------------------------------------------


def evaluate_single_gate(
    gate_name: str,
    context: ReplayGateContext,
    policy_config: GatePolicyConfig,
    replay_clock: Callable[[], datetime],
    id_provider: Callable[[], str],
) -> GateTraceEntry:
    """Pure evaluation of a single gate via ReplayGateAdapter.

    No DB session, no logging side effects, no env-var reads, no global mutation.

    Args:
        gate_name: Name of the gate to evaluate
        context: ReplayGateContext with historical inputs
        policy_config: Frozen policy configuration
        replay_clock: Frozen clock callable returning the cutoff timestamp
        id_provider: Deterministic ID provider callable

    Returns:
        GateTraceEntry with the evaluation result
    """
    adapter = ReplayGateAdapter(
        gate_name=gate_name,
        policy_config=policy_config,
        context=context,
        replay_clock=replay_clock,
        id_provider=id_provider,
    )

    try:
        result = adapter.evaluate()
    except Exception as exc:
        # Record error decision with exception details
        missing_fields = _detect_missing_fields_for_gate(gate_name, context)
        return GateTraceEntry(
            gate_name=gate_name,
            decision="error",
            reason_code=f"{type(exc).__name__}: {exc}",
            threshold_applied={},
            input_fields={},
            adjusted_values=None,
            cumulative_size_multiplier=1.0,
            is_non_executable=False,
            missing_fields=missing_fields,
        )

    return _build_trace_entry(
        gate_name=gate_name,
        result=result,
        cumulative_size_multiplier=1.0,
        is_non_executable=False,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _evaluate_gate_safe(
    *,
    gate_name: str,
    context: ReplayGateContext,
    policy_config: GatePolicyConfig,
    replay_clock: Callable[[], datetime],
    id_provider: Callable[[], str],
    cumulative_size_multiplier: float,
    is_non_executable: bool,
    diagnostic_mode: bool,
) -> GateTraceEntry:
    """Evaluate a gate with full exception handling.

    If the gate raises an exception:
    - Record 'error' decision with exception type and message
    - In production mode: the caller will mark replay unscorable and stop
    - In diagnostic mode: record error, label downstream as non-executable, continue

    Uses ReplayGateAdapter for core gates and ExtendedGateAdapter for
    extended decision boundary gates (Phase 2).

    Args:
        gate_name: Name of the gate to evaluate
        context: ReplayGateContext with historical inputs
        policy_config: Frozen policy configuration
        replay_clock: Frozen clock callable
        id_provider: Deterministic ID provider callable
        cumulative_size_multiplier: Current cumulative multiplier from prior gates
        is_non_executable: Whether this gate is downstream of a prior rejection
        diagnostic_mode: Whether operating in diagnostic mode

    Returns:
        GateTraceEntry with evaluation result or error record
    """
    # Select adapter based on gate type: core vs extended
    is_extended = gate_name in EXTENDED_GATE_SEQUENCE
    if is_extended:
        adapter = ExtendedGateAdapter(
            gate_name=gate_name,
            policy_config=policy_config,
            context=context,
            replay_clock=replay_clock,
            id_provider=id_provider,
        )
    else:
        adapter = ReplayGateAdapter(
            gate_name=gate_name,
            policy_config=policy_config,
            context=context,
            replay_clock=replay_clock,
            id_provider=id_provider,
        )

    try:
        result = adapter.evaluate()
    except Exception as exc:
        # Gate raised an unhandled exception (beyond what adapter already catches)
        missing_fields = _detect_missing_fields_for_gate(gate_name, context)
        return GateTraceEntry(
            gate_name=gate_name,
            decision="error",
            reason_code=f"{type(exc).__name__}: {exc}",
            threshold_applied={},
            input_fields={},
            adjusted_values=None,
            cumulative_size_multiplier=cumulative_size_multiplier,
            is_non_executable=is_non_executable,
            missing_fields=missing_fields,
        )

    # Build trace entry from adapter result
    entry = _build_trace_entry(
        gate_name=gate_name,
        result=result,
        cumulative_size_multiplier=cumulative_size_multiplier,
        is_non_executable=is_non_executable,
    )

    return entry


def _build_trace_entry(
    *,
    gate_name: str,
    result: dict,
    cumulative_size_multiplier: float,
    is_non_executable: bool,
) -> GateTraceEntry:
    """Build a GateTraceEntry from a ReplayGateAdapter result dict.

    The adapter returns a dict with keys:
    - decision: canonical decision string
    - reason_type: gate-specific reason code
    - reason: human-readable reason
    - raw_result: the unmodified dict returned by the gate
    - missing_fields: list of required fields that are None/missing
    """
    decision = result.get("decision", "error")
    reason_code = result.get("reason_type", "") or result.get("reason", "")
    raw_result = result.get("raw_result", {})
    missing_fields = result.get("missing_fields", [])

    # Extract threshold_applied from the raw gate result if available
    threshold_applied = _extract_thresholds(raw_result)

    # Extract input_fields from the raw gate result
    input_fields = _extract_input_fields(raw_result)

    # Extract adjusted values (e.g. widened stop, reduced quantity)
    adjusted_values = _extract_adjusted_values(raw_result)

    # Update cumulative size multiplier if gate adjusts size
    new_multiplier = cumulative_size_multiplier
    if decision == "reduce_size":
        size_adjustment = raw_result.get("size_multiplier")
        if size_adjustment is not None:
            try:
                new_multiplier = cumulative_size_multiplier * float(size_adjustment)
            except (TypeError, ValueError):
                pass

    return GateTraceEntry(
        gate_name=gate_name,
        decision=decision,
        reason_code=reason_code,
        threshold_applied=threshold_applied,
        input_fields=input_fields,
        adjusted_values=adjusted_values,
        cumulative_size_multiplier=new_multiplier,
        is_non_executable=is_non_executable,
        missing_fields=missing_fields,
    )


def _extract_thresholds(raw_result: dict) -> dict:
    """Extract threshold information from a gate's raw result dict."""
    thresholds: dict = {}

    # Common threshold keys produced by gates
    threshold_keys = [
        "threshold_applied",
        "threshold",
        "min_rr",
        "min_win_rate",
        "max_stop_distance",
        "min_confidence",
        "rr_threshold",
        "signal_strength_threshold",
    ]

    for key in threshold_keys:
        if key in raw_result and raw_result[key] is not None:
            thresholds[key] = raw_result[key]

    # Also check nested thresholds dict
    if "thresholds" in raw_result and isinstance(raw_result["thresholds"], dict):
        thresholds.update(raw_result["thresholds"])

    return thresholds


def _extract_input_fields(raw_result: dict) -> dict:
    """Extract material input fields consumed by the gate from its result."""
    input_fields: dict = {}

    # Common input field keys that gates report in their results
    input_keys = [
        "symbol",
        "profile",
        "setup_type",
        "direction",
        "entry_price",
        "stop_price",
        "target_price",
        "quantity",
        "signal_strength",
        "confidence_level",
        "confidence_score",
        "atr_value",
        "selection_score",
        "execution_score",
        "catalyst_type",
        "rr_ratio",
        "stop_distance_pct",
    ]

    for key in input_keys:
        if key in raw_result and raw_result[key] is not None:
            input_fields[key] = raw_result[key]

    # Also check for an 'inputs' dict within the raw result
    if "inputs" in raw_result and isinstance(raw_result["inputs"], dict):
        input_fields.update(raw_result["inputs"])

    return input_fields


def _extract_adjusted_values(raw_result: dict) -> dict | None:
    """Extract any adjusted values produced by the gate (e.g. widened stop)."""
    adjusted: dict = {}

    adjusted_keys = [
        "adjusted_stop",
        "adjusted_quantity",
        "adjusted_target",
        "new_stop_price",
        "new_quantity",
        "size_multiplier",
        "adjusted_size",
    ]

    for key in adjusted_keys:
        if key in raw_result and raw_result[key] is not None:
            adjusted[key] = raw_result[key]

    # Also check for a nested 'adjustments' dict
    if "adjustments" in raw_result and isinstance(raw_result["adjustments"], dict):
        adjusted.update(raw_result["adjustments"])

    return adjusted if adjusted else None


def _detect_missing_fields_for_gate(gate_name: str, context: ReplayGateContext) -> list[str]:
    """Detect required fields that are null/missing in the context for a given gate.

    This mirrors the adapter's _detect_missing_fields but can be called independently
    when the adapter itself fails before getting to its own detection.
    Handles both core gates and extended gates (Phase 2).
    """
    # Check core gates first, then extended gates
    required = GATE_REQUIRED_FIELDS.get(gate_name, [])
    if not required:
        required = EXTENDED_GATE_REQUIRED_FIELDS.get(gate_name, [])

    missing: list[str] = []

    for field_name in required:
        value = getattr(context, field_name, None)
        if value is None:
            missing.append(field_name)
        elif isinstance(value, str) and not value.strip():
            missing.append(field_name)

    return missing
