"""Decision Delta Classifier — compares original and replay decisions.

Pure functions that classify the relationship between original and replay decisions,
identify the first diverging gate, and determine root-cause divergence categories.

All geometry comparisons use Geometry_Hash (tick-normalized Decimal + SHA-256)
rather than floating-point equality.

See: design.md §core/replay/delta_classifier.py
Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, TYPE_CHECKING

from utils.decision_snapshot import compute_geometry_hash

if TYPE_CHECKING:
    from core.replay.gate_replayer import GateTrace


# ---------------------------------------------------------------------------
# Core data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionDelta:
    """Classified difference between original and replay decisions.

    Exactly one of the 11 defined categories is assigned per comparison.
    """

    classification: str  # one of DELTA_CATEGORIES
    first_diverging_gate: str | None
    divergence_cause: str | None  # one of DIVERGENCE_CAUSES or None
    divergence_evidence: dict | None  # specific field/threshold/value
    original_decision: str
    replay_decision: str
    geometry_differs: bool
    size_differs: bool


# ---------------------------------------------------------------------------
# Category and cause enumerations
# ---------------------------------------------------------------------------

DELTA_CATEGORIES: list[str] = [
    "same_allow",
    "same_reject",
    "same_final_reject_different_trace",
    "same_final_allow_different_trace",
    "replay_allows_original_reject",
    "replay_rejects_original_allow",
    "same_direction_different_size",
    "same_direction_different_geometry",
    "same_direction_different_size_and_geometry",
    "partial_comparison",
    "unscorable",
]

DIVERGENCE_CAUSES: list[str] = [
    "metadata_wiring_defect",
    "data_quality_failure",
    "configuration_change",
    "code_defect",
    "policy_threshold_change",
]

# Canonical gate sequence for ordering divergence identification
CORE_GATE_SEQUENCE: list[str] = [
    "setup_quality_gate",
    "pre_trade_quality_gate",
    "catalyst_specificity_gate",
    "risk_geometry_gate",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_delta(
    original_decision: str,
    original_gate: str | None,
    original_reason_code: str | None,
    original_geometry: dict,
    original_size: Decimal,
    replay_trace: "GateTrace",
    replay_classification: str,
) -> DecisionDelta:
    """Classify the difference between original and replayed decisions.

    Follows strict precedence:
      unscorable > partial_comparison > same_direction_different_size_and_geometry > others

    Args:
        original_decision: Normalized original decision ("allow" or "reject")
        original_gate: Gate that produced the original decision (or None if allow)
        original_reason_code: Reason code from the original rejecting gate
        original_geometry: Dict with entry_price, stop_price, target_price as Decimal
        original_size: Original position size as Decimal
        replay_trace: The GateTrace from replaying the gate sequence
        replay_classification: Input classification ("exact", "partial", "unscorable")

    Returns:
        DecisionDelta with exactly one classification from DELTA_CATEGORIES
    """
    # Normalize decisions to canonical form
    orig_decision = _normalize_decision(original_decision)
    replay_decision = replay_trace.final_decision

    # --- Precedence 1: unscorable ---
    if replay_classification == "unscorable":
        return DecisionDelta(
            classification="unscorable",
            first_diverging_gate=None,
            divergence_cause=None,
            divergence_evidence=None,
            original_decision=orig_decision,
            replay_decision=replay_decision,
            geometry_differs=False,
            size_differs=False,
        )

    # --- Precedence 2: partial_comparison ---
    if replay_classification == "partial":
        return DecisionDelta(
            classification="partial_comparison",
            first_diverging_gate=None,
            divergence_cause=None,
            divergence_evidence=None,
            original_decision=orig_decision,
            replay_decision=replay_decision,
            geometry_differs=False,
            size_differs=False,
        )

    # Compute geometry and size differences using Geometry_Hash
    geometry_differs = _geometry_differs(original_geometry, replay_trace)
    size_differs = _size_differs(original_size, replay_trace)

    # --- Precedence 3: same_direction_different_size_and_geometry ---
    # When both size and geometry differ, the combined category takes precedence
    if orig_decision == "allow" and replay_decision == "allow":
        if geometry_differs and size_differs:
            first_gate, evidence = identify_first_diverging_gate(
                None, replay_trace
            )
            cause = None
            if first_gate and evidence:
                cause, cause_evidence = classify_divergence_cause(
                    first_gate,
                    evidence.get("original_inputs", {}),
                    evidence.get("replay_inputs", {}),
                    evidence.get("original_policy"),
                    evidence.get("replay_policy", {}),
                )
                evidence["divergence_cause"] = cause
            return DecisionDelta(
                classification="same_direction_different_size_and_geometry",
                first_diverging_gate=first_gate,
                divergence_cause=cause,
                divergence_evidence=evidence,
                original_decision=orig_decision,
                replay_decision=replay_decision,
                geometry_differs=True,
                size_differs=True,
            )

    # --- Other categories ---

    # Direction change: replay allows but original rejected
    if orig_decision == "reject" and replay_decision == "allow":
        first_gate, evidence = identify_first_diverging_gate(
            _build_original_trace_from_gate(original_gate, original_reason_code),
            replay_trace,
        )
        cause = None
        if first_gate and evidence:
            cause, _ = classify_divergence_cause(
                first_gate,
                evidence.get("original_inputs", {}),
                evidence.get("replay_inputs", {}),
                evidence.get("original_policy"),
                evidence.get("replay_policy", {}),
            )
        return DecisionDelta(
            classification="replay_allows_original_reject",
            first_diverging_gate=first_gate,
            divergence_cause=cause,
            divergence_evidence=evidence,
            original_decision=orig_decision,
            replay_decision=replay_decision,
            geometry_differs=geometry_differs,
            size_differs=size_differs,
        )

    # Direction change: replay rejects but original allowed
    if orig_decision == "allow" and replay_decision == "reject":
        first_gate, evidence = identify_first_diverging_gate(
            None, replay_trace
        )
        cause = None
        if first_gate and evidence:
            cause, _ = classify_divergence_cause(
                first_gate,
                evidence.get("original_inputs", {}),
                evidence.get("replay_inputs", {}),
                evidence.get("original_policy"),
                evidence.get("replay_policy", {}),
            )
        return DecisionDelta(
            classification="replay_rejects_original_allow",
            first_diverging_gate=first_gate,
            divergence_cause=cause,
            divergence_evidence=evidence,
            original_decision=orig_decision,
            replay_decision=replay_decision,
            geometry_differs=geometry_differs,
            size_differs=size_differs,
        )

    # Both allow — check geometry/size individually
    if orig_decision == "allow" and replay_decision == "allow":
        if geometry_differs and not size_differs:
            first_gate, evidence = identify_first_diverging_gate(None, replay_trace)
            return DecisionDelta(
                classification="same_direction_different_geometry",
                first_diverging_gate=first_gate,
                divergence_cause=None,
                divergence_evidence=evidence,
                original_decision=orig_decision,
                replay_decision=replay_decision,
                geometry_differs=True,
                size_differs=False,
            )
        if size_differs and not geometry_differs:
            first_gate, evidence = identify_first_diverging_gate(None, replay_trace)
            return DecisionDelta(
                classification="same_direction_different_size",
                first_diverging_gate=first_gate,
                divergence_cause=None,
                divergence_evidence=evidence,
                original_decision=orig_decision,
                replay_decision=replay_decision,
                geometry_differs=False,
                size_differs=True,
            )
        # Both allow, same geometry and size — check trace differences
        if _traces_differ(original_gate, original_reason_code, replay_trace):
            return DecisionDelta(
                classification="same_final_allow_different_trace",
                first_diverging_gate=None,
                divergence_cause=None,
                divergence_evidence=None,
                original_decision=orig_decision,
                replay_decision=replay_decision,
                geometry_differs=False,
                size_differs=False,
            )
        return DecisionDelta(
            classification="same_allow",
            first_diverging_gate=None,
            divergence_cause=None,
            divergence_evidence=None,
            original_decision=orig_decision,
            replay_decision=replay_decision,
            geometry_differs=False,
            size_differs=False,
        )

    # Both reject
    if orig_decision == "reject" and replay_decision == "reject":
        replay_gate = replay_trace.final_gate
        replay_reason = replay_trace.final_reason_code

        if original_gate == replay_gate and original_reason_code == replay_reason:
            return DecisionDelta(
                classification="same_reject",
                first_diverging_gate=None,
                divergence_cause=None,
                divergence_evidence=None,
                original_decision=orig_decision,
                replay_decision=replay_decision,
                geometry_differs=geometry_differs,
                size_differs=size_differs,
            )
        # Both reject but at different gates or for different reasons
        return DecisionDelta(
            classification="same_final_reject_different_trace",
            first_diverging_gate=replay_gate if replay_gate != original_gate else original_gate,
            divergence_cause=None,
            divergence_evidence={
                "original_gate": original_gate,
                "original_reason": original_reason_code,
                "replay_gate": replay_gate,
                "replay_reason": replay_reason,
            },
            original_decision=orig_decision,
            replay_decision=replay_decision,
            geometry_differs=geometry_differs,
            size_differs=size_differs,
        )

    # Fallback (should not be reached with proper normalization)
    return DecisionDelta(
        classification="unscorable",
        first_diverging_gate=None,
        divergence_cause=None,
        divergence_evidence={"reason": "unhandled_decision_combination"},
        original_decision=orig_decision,
        replay_decision=replay_decision,
        geometry_differs=geometry_differs,
        size_differs=size_differs,
    )


def identify_first_diverging_gate(
    original_trace: list[dict] | None,
    replay_trace: "GateTrace",
) -> tuple[str | None, dict | None]:
    """Find the first gate in pipeline order where decisions or inputs differ.

    Walks the gate sequence in production order. For each gate, compares
    the original trace entry (if available) with the replay trace entry.
    Returns the gate name and evidence dict containing the differing fields.

    Args:
        original_trace: List of dicts representing original gate trace entries,
                        or None if original trace is unavailable.
                        Each dict should have: gate_name, decision, input_fields
        replay_trace: The GateTrace from replay execution.

    Returns:
        Tuple of (gate_name, evidence_dict) or (None, None) if no divergence found.
    """
    if not replay_trace or not replay_trace.entries:
        return None, None

    # Build lookup from replay trace entries
    replay_entries_by_gate: dict[str, Any] = {}
    for entry in replay_trace.entries:
        replay_entries_by_gate[entry.gate_name] = entry

    # Build lookup from original trace if available
    original_entries_by_gate: dict[str, dict] = {}
    if original_trace:
        for entry in original_trace:
            gate_name = entry.get("gate_name", "")
            if gate_name:
                original_entries_by_gate[gate_name] = entry

    # Walk in pipeline order
    for gate_name in CORE_GATE_SEQUENCE:
        replay_entry = replay_entries_by_gate.get(gate_name)
        if replay_entry is None:
            continue

        original_entry = original_entries_by_gate.get(gate_name)

        # If no original trace entry exists for this gate, we can't compare inputs
        if original_entry is None:
            # If the replay gate produced a rejection, this is the divergence point
            if replay_entry.decision in ("reject", "error"):
                return gate_name, {
                    "replay_decision": replay_entry.decision,
                    "replay_reason": replay_entry.reason_code,
                    "replay_inputs": replay_entry.input_fields,
                    "original_inputs": {},
                    "original_policy": None,
                    "replay_policy": replay_entry.threshold_applied,
                    "reason": "no_original_trace_for_gate",
                }
            continue

        # Compare decisions
        original_decision = original_entry.get("decision", "")
        replay_decision_str = replay_entry.decision

        if original_decision != replay_decision_str:
            return gate_name, {
                "original_decision": original_decision,
                "replay_decision": replay_decision_str,
                "original_inputs": original_entry.get("input_fields", {}),
                "replay_inputs": replay_entry.input_fields,
                "original_policy": original_entry.get("threshold_applied"),
                "replay_policy": replay_entry.threshold_applied,
                "reason": "decision_differs",
            }

        # Compare input fields
        original_inputs = original_entry.get("input_fields", {})
        replay_inputs = replay_entry.input_fields or {}

        differing_fields = _find_differing_fields(original_inputs, replay_inputs)
        if differing_fields:
            return gate_name, {
                "original_decision": original_decision,
                "replay_decision": replay_decision_str,
                "original_inputs": original_inputs,
                "replay_inputs": replay_inputs,
                "original_policy": original_entry.get("threshold_applied"),
                "replay_policy": replay_entry.threshold_applied,
                "differing_fields": differing_fields,
                "reason": "inputs_differ",
            }

    return None, None


def classify_divergence_cause(
    gate_name: str,
    original_inputs: dict,
    replay_inputs: dict,
    original_policy: dict | None,
    replay_policy: dict,
) -> tuple[str, dict]:
    """Classify the root cause of divergence at a specific gate.

    Precedence (closest to root cause wins):
      metadata_wiring_defect / code_defect > configuration_change > policy_threshold_change

    Args:
        gate_name: The gate where divergence was detected
        original_inputs: Input field values from the original trace
        replay_inputs: Input field values from the replay trace
        original_policy: Thresholds/config from original policy (may be None)
        replay_policy: Thresholds/config from replay policy

    Returns:
        Tuple of (cause_category, evidence_dict)
    """
    evidence: dict[str, Any] = {"gate": gate_name}

    # --- Check 1: metadata_wiring_defect ---
    # A required field was null/dropped/incorrectly typed in original but present in replay
    null_or_dropped = _detect_metadata_wiring_defect(
        gate_name, original_inputs, replay_inputs
    )
    if null_or_dropped:
        evidence["defective_fields"] = null_or_dropped
        evidence["description"] = (
            "Required field was null, dropped, or incorrectly typed in original, "
            "causing gate to use fallback/default value"
        )
        return "metadata_wiring_defect", evidence

    # --- Check 2: code_defect ---
    # Gate logic or computation behavior differs between revisions
    # Detected when inputs are identical but decisions differ (same thresholds)
    code_defect = _detect_code_defect(
        original_inputs, replay_inputs, original_policy, replay_policy
    )
    if code_defect:
        evidence["code_defect_indicators"] = code_defect
        evidence["description"] = (
            "Gate logic, field routing, or computation behavior differs "
            "between original and replay code revisions"
        )
        return "code_defect", evidence

    # --- Check 3: configuration_change ---
    # A threshold, feature flag, or mapping differs between policy versions
    # with no code change
    config_change = _detect_configuration_change(original_policy, replay_policy)
    if config_change:
        evidence["changed_config"] = config_change
        evidence["description"] = (
            "Gate threshold, feature flag, or mapping differs between "
            "original and replay policy versions"
        )
        return "configuration_change", evidence

    # --- Check 4: data_quality_failure ---
    # A required market-data or signal input was missing, stale, or malformed
    data_quality = _detect_data_quality_failure(gate_name, original_inputs, replay_inputs)
    if data_quality:
        evidence["data_quality_issues"] = data_quality
        evidence["description"] = (
            "Required market-data or signal input was missing, stale, "
            "or malformed at decision time"
        )
        return "data_quality_failure", evidence

    # --- Check 5: policy_threshold_change (lowest precedence) ---
    # The declared policy version intentionally defines different thresholds
    if original_policy and replay_policy:
        threshold_diffs = _find_threshold_differences(original_policy, replay_policy)
        if threshold_diffs:
            evidence["threshold_differences"] = threshold_diffs
            evidence["description"] = (
                "Declared policy version intentionally defines different "
                "threshold values from original policy"
            )
            return "policy_threshold_change", evidence

    # Fallback: if no specific cause can be determined
    evidence["description"] = "Divergence cause could not be specifically determined"
    return "data_quality_failure", evidence


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_decision(decision: str) -> str:
    """Normalize decision string to canonical 'allow' or 'reject'."""
    normalized = decision.lower().strip()
    if normalized in ("allow", "allowed", "adjusted_allowed"):
        return "allow"
    if normalized in ("reject", "rejected", "block", "blocked"):
        return "reject"
    return normalized


def _geometry_differs(
    original_geometry: dict, replay_trace: "GateTrace"
) -> bool:
    """Compare geometries using Geometry_Hash (not floating-point equality).

    Extracts replay geometry from the trace's final adjusted values or
    from the risk_geometry_gate entry's input fields.
    """
    orig_entry = original_geometry.get("entry_price")
    orig_stop = original_geometry.get("stop_price")
    orig_target = original_geometry.get("target_price")

    # Get replay geometry from trace
    replay_geometry = _extract_replay_geometry(replay_trace)
    replay_entry = replay_geometry.get("entry_price")
    replay_stop = replay_geometry.get("stop_price")
    replay_target = replay_geometry.get("target_price")

    # Use Geometry_Hash for comparison
    orig_hash = compute_geometry_hash(
        Decimal(str(orig_entry)) if orig_entry is not None else None,
        Decimal(str(orig_stop)) if orig_stop is not None else None,
        Decimal(str(orig_target)) if orig_target is not None else None,
    )
    replay_hash = compute_geometry_hash(
        Decimal(str(replay_entry)) if replay_entry is not None else None,
        Decimal(str(replay_stop)) if replay_stop is not None else None,
        Decimal(str(replay_target)) if replay_target is not None else None,
    )

    # Empty hashes (incomplete geometry) are treated as non-comparable
    if not orig_hash or not replay_hash:
        return False

    return orig_hash != replay_hash


def _size_differs(original_size: Decimal, replay_trace: "GateTrace") -> bool:
    """Check if position size differs between original and replay."""
    replay_size = _extract_replay_size(replay_trace)
    if replay_size is None:
        return False
    return Decimal(str(original_size)) != Decimal(str(replay_size))


def _extract_replay_geometry(replay_trace: "GateTrace") -> dict:
    """Extract final geometry from replay trace.

    Looks for adjusted values from the risk_geometry_gate first,
    then falls back to input fields.
    """
    if not replay_trace or not replay_trace.entries:
        return {}

    # Check entries in reverse order for adjusted values
    for entry in reversed(replay_trace.entries):
        if entry.adjusted_values and "stop_price" in entry.adjusted_values:
            return {
                "entry_price": entry.adjusted_values.get(
                    "entry_price",
                    entry.input_fields.get("entry_price") if entry.input_fields else None,
                ),
                "stop_price": entry.adjusted_values.get("stop_price"),
                "target_price": entry.adjusted_values.get(
                    "target_price",
                    entry.input_fields.get("target_price") if entry.input_fields else None,
                ),
            }

    # Fall back to risk_geometry_gate input fields
    for entry in replay_trace.entries:
        if entry.gate_name == "risk_geometry_gate" and entry.input_fields:
            return {
                "entry_price": entry.input_fields.get("entry_price"),
                "stop_price": entry.input_fields.get("stop_price"),
                "target_price": entry.input_fields.get("target_price"),
            }

    return {}


def _extract_replay_size(replay_trace: "GateTrace") -> Decimal | None:
    """Extract final position size from replay trace.

    Looks for adjusted quantity in trace entries, then falls back to input fields.
    """
    if not replay_trace or not replay_trace.entries:
        return None

    # Check for adjusted quantity (size reduction)
    for entry in reversed(replay_trace.entries):
        if entry.adjusted_values and "quantity" in entry.adjusted_values:
            return Decimal(str(entry.adjusted_values["quantity"]))

    # Fall back to input fields from risk_geometry_gate
    for entry in replay_trace.entries:
        if entry.gate_name == "risk_geometry_gate" and entry.input_fields:
            qty = entry.input_fields.get("quantity")
            if qty is not None:
                return Decimal(str(qty))

    return None


def _traces_differ(
    original_gate: str | None,
    original_reason_code: str | None,
    replay_trace: "GateTrace",
) -> bool:
    """Detect whether traces differ when both decisions are 'allow'.

    For allow decisions, the trace differs if any intermediate gate
    produced a different decision, adjustment, or reason code between
    original and replay runs.
    """
    # If original had a gate involved (e.g. adjustment), check if replay shows the same
    if original_gate is not None:
        # Original had some intermediate gate activity — check if replay matches
        for entry in replay_trace.entries:
            if entry.gate_name == original_gate:
                if entry.reason_code != original_reason_code:
                    return True
                break
        else:
            # Gate not found in replay trace — traces differ
            return True

    # Check if replay trace has any intermediate adjustments
    for entry in replay_trace.entries:
        if entry.decision == "reduce_size" or entry.adjusted_values:
            return True

    return False


def _build_original_trace_from_gate(
    gate: str | None, reason_code: str | None
) -> list[dict] | None:
    """Build a minimal original trace from the known rejecting gate."""
    if gate is None:
        return None
    return [
        {
            "gate_name": gate,
            "decision": "reject",
            "reason_code": reason_code,
            "input_fields": {},
            "threshold_applied": {},
        }
    ]


def _find_differing_fields(
    original_inputs: dict, replay_inputs: dict
) -> list[dict]:
    """Find fields that differ between original and replay inputs."""
    diffs: list[dict] = []
    all_keys = set(original_inputs.keys()) | set(replay_inputs.keys())

    for key in sorted(all_keys):
        orig_val = original_inputs.get(key)
        replay_val = replay_inputs.get(key)

        if orig_val != replay_val:
            diffs.append({
                "field": key,
                "original_value": orig_val,
                "replay_value": replay_val,
            })

    return diffs


def _detect_metadata_wiring_defect(
    gate_name: str, original_inputs: dict, replay_inputs: dict
) -> list[dict]:
    """Detect when a required field was null/dropped in original but present in replay.

    This indicates a metadata wiring defect: the field was available but not
    properly routed to the gate.
    """
    from core.replay.gate_adapter import GATE_REQUIRED_FIELDS

    required_fields = GATE_REQUIRED_FIELDS.get(gate_name, [])
    defects: list[dict] = []

    for field_name in required_fields:
        orig_val = original_inputs.get(field_name)
        replay_val = replay_inputs.get(field_name)

        # Field was null/missing in original but present in replay
        if _is_null_or_missing(orig_val) and not _is_null_or_missing(replay_val):
            defects.append({
                "field": field_name,
                "original_value": orig_val,
                "replay_value": replay_val,
                "issue": "null_or_dropped_in_original",
            })

    return defects


def _detect_code_defect(
    original_inputs: dict,
    replay_inputs: dict,
    original_policy: dict | None,
    replay_policy: dict,
) -> list[dict]:
    """Detect code defects: inputs identical and policies identical but decisions differ.

    This suggests the gate logic itself has changed between revisions.
    """
    # If policies differ, it's likely a configuration change, not code defect
    if original_policy and replay_policy and original_policy != replay_policy:
        return []

    # Check if inputs are substantially the same
    if not original_inputs or not replay_inputs:
        return []

    # If inputs are the same but we know decisions differ (we got here from
    # identify_first_diverging_gate), and policies match, it's a code defect
    differing = _find_differing_fields(original_inputs, replay_inputs)

    # No input diffs and no policy diffs → pure code behavior change
    if not differing and (original_policy == replay_policy or original_policy is None):
        return [{"indicator": "same_inputs_same_policy_different_result"}]

    return []


def _detect_configuration_change(
    original_policy: dict | None, replay_policy: dict
) -> list[dict]:
    """Detect configuration changes between policy versions."""
    if original_policy is None:
        return []

    changes: list[dict] = []
    all_keys = set(original_policy.keys()) | set(replay_policy.keys())

    for key in sorted(all_keys):
        orig_val = original_policy.get(key)
        replay_val = replay_policy.get(key)
        if orig_val != replay_val:
            changes.append({
                "config_key": key,
                "original_value": orig_val,
                "replay_value": replay_val,
            })

    return changes


def _detect_data_quality_failure(
    gate_name: str, original_inputs: dict, replay_inputs: dict
) -> list[dict]:
    """Detect data quality issues: market data or signal inputs missing/stale/malformed."""
    from core.replay.gate_adapter import GATE_REQUIRED_FIELDS

    # Data quality fields are market data and signal-related
    data_quality_fields = frozenset({
        "atr_value", "atr_timestamp", "current_price", "signal_strength",
        "confidence_value", "selection_score", "execution_score",
        "analyst_signal_payload",
    })

    required = set(GATE_REQUIRED_FIELDS.get(gate_name, []))
    relevant_fields = required & data_quality_fields

    issues: list[dict] = []
    for field_name in sorted(relevant_fields):
        orig_val = original_inputs.get(field_name)
        # Field was missing/null in original (data quality issue at decision time)
        if _is_null_or_missing(orig_val):
            issues.append({
                "field": field_name,
                "original_value": orig_val,
                "issue": "missing_or_stale_at_decision_time",
            })

    return issues


def _find_threshold_differences(
    original_policy: dict, replay_policy: dict
) -> list[dict]:
    """Find intentional threshold differences between policy versions."""
    diffs: list[dict] = []
    all_keys = set(original_policy.keys()) | set(replay_policy.keys())

    for key in sorted(all_keys):
        orig_val = original_policy.get(key)
        replay_val = replay_policy.get(key)
        if orig_val != replay_val:
            diffs.append({
                "threshold": key,
                "original_value": orig_val,
                "replay_value": replay_val,
            })

    return diffs


def _is_null_or_missing(value: Any) -> bool:
    """Check if a value is effectively null or missing."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False
