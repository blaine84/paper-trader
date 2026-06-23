"""Decision Contract Parser — bounded PM response validation.

Parses and validates the Portfolio Manager's bounded decision response
against the candidate-ID selection contract. Ensures only valid,
contract-compliant decisions reach the execution pipeline.

See: design.md §utils/decision_contract.py
Requirements: 3.1, 3.2, 3.4, 3.5, 3.6, 3.7, 7.6
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.candidate_registry import CandidateRegistry

logger = logging.getLogger(__name__)

# Valid fields in a single decision entry
_VALID_DECISION_FIELDS = {"candidate_id", "decision", "risk_multiplier", "rationale"}

# Valid decision values (P0: accept/reject only, no "adjust")
_VALID_DECISIONS = {"accept", "reject"}


@dataclass
class CandidateDecision:
    """A single parsed PM decision for one candidate."""

    candidate_id: str
    decision: str  # "accept" | "reject"
    risk_multiplier: float | None = None  # 0.0 < x <= 1.0 (downward only)
    rationale: str = ""


@dataclass
class ParseResult:
    """Aggregate parse result for a PM cycle response."""

    accepted: list[CandidateDecision] = field(default_factory=list)
    rejected: list[CandidateDecision] = field(default_factory=list)
    not_selected_ids: set[str] = field(default_factory=set)
    violations: list[dict] = field(default_factory=list)
    duplicate_ids: set[str] = field(default_factory=set)
    raw_response: dict = field(default_factory=dict)


def parse_decision_contract(
    raw_response: dict | list | None,
    valid_candidate_ids: set[str],
    candidate_metadata: dict[str, dict],
) -> ParseResult:
    """Parse and validate PM response against the bounded decision contract.

    Args:
        raw_response: Raw LLM output (JSON-parsed). Expected format:
            {"decisions": [{"candidate_id": "...", "decision": "accept"|"reject",
                           "risk_multiplier": 0.5, "rationale": "..."}]}
        valid_candidate_ids: Set of IDs offered this cycle (REGISTERED state).
        candidate_metadata: Immutable map of {candidate_id: {symbol, source_signal_id, profile_id}}
            used to enforce one-accept-per-(symbol, signal, profile) deduplication.

    Validations:
        1. candidate_id must be in valid_candidate_ids
        2. decision must be "accept" or "reject" (P0: no "adjust")
        3. risk_multiplier must be None or in (0.0, 1.0]
        4. duplicate candidate_ids → reject ALL entries for that ID
        5. extra fields → log violation, strip from execution
        6. at most one accept per (symbol, source_signal_id, profile_id)

    Returns:
        ParseResult with accepted/rejected decisions, violations, and metadata.
    """
    result = ParseResult()

    # Step 1: Handle None or invalid top-level type
    if raw_response is None:
        result.violations.append(
            {"type": "MALFORMED_RESPONSE", "reason": "raw_response is None"}
        )
        result.not_selected_ids = set(valid_candidate_ids)
        return result

    # Step 2: If raw_response is a list, wrap it
    if isinstance(raw_response, list):
        raw_response = {"decisions": raw_response}

    # Step 3: Must be a dict at this point
    if not isinstance(raw_response, dict):
        result.violations.append(
            {
                "type": "MALFORMED_RESPONSE",
                "reason": f"raw_response is not a dict or list, got {type(raw_response).__name__}",
            }
        )
        result.not_selected_ids = set(valid_candidate_ids)
        return result

    result.raw_response = raw_response

    # Step 4: Extract decisions key
    decisions = raw_response.get("decisions")
    if decisions is None:
        result.violations.append(
            {"type": "MALFORMED_RESPONSE", "reason": "missing 'decisions' key"}
        )
        result.not_selected_ids = set(valid_candidate_ids)
        return result

    if not isinstance(decisions, list):
        result.violations.append(
            {
                "type": "MALFORMED_RESPONSE",
                "reason": f"'decisions' is not a list, got {type(decisions).__name__}",
            }
        )
        result.not_selected_ids = set(valid_candidate_ids)
        return result

    # Step 5: First pass — detect duplicate candidate_ids
    id_counts: dict[str, int] = {}
    for entry in decisions:
        if isinstance(entry, dict) and "candidate_id" in entry:
            cid = entry["candidate_id"]
            id_counts[cid] = id_counts.get(cid, 0) + 1

    duplicate_ids: set[str] = {cid for cid, count in id_counts.items() if count > 1}
    result.duplicate_ids = set(duplicate_ids)

    # Record violations for duplicates
    for cid in duplicate_ids:
        result.violations.append({"type": "DUPLICATE_CANDIDATE", "candidate_id": cid})

    # Step 6: Second pass — validate each decision entry
    mentioned_ids: set[str] = set()
    # Track accepted (symbol, source_signal_id, profile_id) tuples for dedup
    accepted_tuples: set[tuple[str, str, str]] = {}  # type: ignore[assignment]
    accepted_tuples = set()

    for entry in decisions:
        if not isinstance(entry, dict):
            result.violations.append(
                {
                    "type": "MALFORMED_RESPONSE",
                    "reason": f"decision entry is not a dict, got {type(entry).__name__}",
                }
            )
            continue

        candidate_id = entry.get("candidate_id")
        if candidate_id is None:
            result.violations.append(
                {
                    "type": "MALFORMED_RESPONSE",
                    "reason": "decision entry missing 'candidate_id'",
                }
            )
            continue

        mentioned_ids.add(candidate_id)

        # 5a: Check for extra fields and log violation (strip them)
        extra_fields = set(entry.keys()) - _VALID_DECISION_FIELDS
        if extra_fields:
            result.violations.append(
                {
                    "type": "EXTRA_FIELDS",
                    "candidate_id": candidate_id,
                    "fields": sorted(extra_fields),
                }
            )
            logger.warning(
                "Contract violation: extra fields %s for candidate %s",
                sorted(extra_fields),
                candidate_id,
            )

        # 5b: If candidate_id is in duplicate_ids, skip (all duplicates rejected)
        if candidate_id in duplicate_ids:
            continue

        # 5c: candidate_id must be in valid_candidate_ids
        if candidate_id not in valid_candidate_ids:
            result.violations.append(
                {"type": "UNKNOWN_CANDIDATE", "candidate_id": candidate_id}
            )
            continue

        # 5d: decision must be "accept" or "reject"
        decision = entry.get("decision")
        if decision not in _VALID_DECISIONS:
            result.violations.append(
                {
                    "type": "INVALID_DECISION",
                    "candidate_id": candidate_id,
                    "decision": decision,
                }
            )
            continue

        # 5e: risk_multiplier validation
        risk_multiplier = entry.get("risk_multiplier")
        if risk_multiplier is not None:
            # Must be numeric and in (0.0, 1.0]
            if not isinstance(risk_multiplier, (int, float)):
                result.violations.append(
                    {
                        "type": "INVALID_RISK_MULTIPLIER",
                        "candidate_id": candidate_id,
                        "value": risk_multiplier,
                    }
                )
                continue
            risk_multiplier = float(risk_multiplier)
            if risk_multiplier <= 0.0 or risk_multiplier > 1.0:
                result.violations.append(
                    {
                        "type": "INVALID_RISK_MULTIPLIER",
                        "candidate_id": candidate_id,
                        "value": risk_multiplier,
                    }
                )
                continue

        # 5f: Create CandidateDecision from valid fields
        rationale = entry.get("rationale", "")
        if not isinstance(rationale, str):
            rationale = str(rationale)

        candidate_decision = CandidateDecision(
            candidate_id=candidate_id,
            decision=decision,
            risk_multiplier=risk_multiplier,
            rationale=rationale,
        )

        # 5g: Route to accepted or rejected list
        if decision == "accept":
            result.accepted.append(candidate_decision)
        else:
            result.rejected.append(candidate_decision)

    # Step 7: One-accept-per-(symbol, source_signal_id, profile_id) enforcement
    final_accepted: list[CandidateDecision] = []
    for candidate_decision in result.accepted:
        cid = candidate_decision.candidate_id
        metadata = candidate_metadata.get(cid)
        if metadata is None:
            # Should not happen if valid_candidate_ids is consistent with metadata
            # but handle gracefully
            final_accepted.append(candidate_decision)
            continue

        key = (
            metadata.get("symbol", ""),
            metadata.get("source_signal_id", ""),
            metadata.get("profile_id", ""),
        )

        if key in accepted_tuples:
            # Duplicate symbol+signal+profile accept — move to violations
            result.violations.append(
                {
                    "type": "DUPLICATE_SYMBOL_SIGNAL_ACCEPT",
                    "candidate_id": cid,
                    "symbol": metadata.get("symbol", ""),
                    "source_signal_id": metadata.get("source_signal_id", ""),
                }
            )
        else:
            accepted_tuples.add(key)
            final_accepted.append(candidate_decision)

    result.accepted = final_accepted

    # Step 8: Compute not_selected_ids
    # All valid IDs that were not mentioned at all (including in duplicates/invalid)
    result.not_selected_ids = valid_candidate_ids - mentioned_ids

    return result


def should_retry_candidate_contract(parse_result: ParseResult) -> bool:
    """Determine if a malformed candidate response warrants one retry.

    Retry conditions (Requirement 8.1):
    - Schema-level failures: response had MALFORMED_RESPONSE violations
      (invalid JSON structure, missing 'decisions' key)
    - All decisions had violations and none were valid

    Do NOT retry (Requirement 8.6):
    - Valid response with zero accepts (normal no-trade outcome)
    - Response with some valid and some invalid decisions (partial success)

    Returns True if retry is warranted, False otherwise.
    """
    # If there are any accepted or rejected decisions, the response had
    # some valid content — no retry needed
    if parse_result.accepted or parse_result.rejected:
        return False

    # Check for malformed response violations (schema-level failures)
    malformed_violations = [
        v for v in parse_result.violations
        if v.get("type") == "MALFORMED_RESPONSE"
    ]
    if malformed_violations:
        return True

    # If all entries had violations and none produced valid decisions
    # (e.g., all IDs unknown, all decisions invalid)
    if parse_result.violations and not parse_result.accepted and not parse_result.rejected:
        return True

    return False


def build_candidate_retry_prompt(
    parse_result: ParseResult,
    registry: 'CandidateRegistry',
) -> str:
    """Build retry prompt with current valid candidate IDs only.

    Rules (Requirements 8.2–8.7):
    - Include only IDs from registry that are still in REGISTERED state
    - Instruct correction of decision shape only
    - Never suggest symbol replacement or ID guessing
    - State that empty accepted set is valid
    - Never ask the model to repair an unknown candidate_id

    Args:
        parse_result: The failed parse result from the first attempt.
        registry: The CandidateRegistry to get current registered IDs.

    Returns:
        A formatted retry prompt string.
    """
    # Get current valid IDs (still REGISTERED)
    valid_ids = registry.get_registered_ids()

    # Build error description
    error_descriptions = []
    for v in parse_result.violations:
        vtype = v.get("type", "unknown")
        if vtype == "MALFORMED_RESPONSE":
            error_descriptions.append(f"Schema error: {v.get('reason', 'unknown')}")
        elif vtype == "UNKNOWN_CANDIDATE":
            error_descriptions.append("Unknown candidate_id referenced")
        elif vtype == "INVALID_DECISION":
            error_descriptions.append("Invalid decision value (must be 'accept' or 'reject')")
        elif vtype == "INVALID_RISK_MULTIPLIER":
            error_descriptions.append("Invalid risk_multiplier (must be > 0.0 and <= 1.0)")
        else:
            error_descriptions.append(f"Contract violation: {vtype}")

    errors_text = "; ".join(error_descriptions[:5])  # Limit to 5 errors

    # Build valid ID list for prompt
    id_list = ", ".join(sorted(valid_ids)[:20])  # Limit display to 20 IDs

    prompt = (
        "Your previous response could not be processed. "
        f"Issues: {errors_text}\n\n"
        "Please provide a corrected response using ONLY these valid candidate IDs:\n"
        f"{id_list}\n\n"
        "Requirements:\n"
        "- Each decision must have: candidate_id, decision ('accept' or 'reject')\n"
        "- Optional: risk_multiplier (number > 0.0 and <= 1.0), rationale (string)\n"
        "- Response format: {\"decisions\": [{\"candidate_id\": \"...\", \"decision\": \"accept\"|\"reject\"}]}\n"
        "- An empty decisions list [] is a valid response (no trades is acceptable)\n"
        "- Do NOT guess or invent candidate IDs — use only the IDs listed above\n"
        "- Do NOT replace symbols or suggest alternative candidates\n"
    )

    return prompt


def record_parse_provenance(
    chain: 'ProvenanceChain',
    parse_result: ParseResult,
    raw_response: dict | None,
    stage_version: str = "1.0",
) -> None:
    """Record provenance events for the parse/normalization stage.

    Called by portfolio_manager AFTER parse_decision_contract() completes.
    Guarded by PM_PROVENANCE_MODE check at call site.

    For accepted decisions: records a provenance event with geometry before/after.
    For unrecoverable parse failures: records a terminal event.

    Fail-open: all provenance operations are wrapped in try/except to never
    block the pipeline.

    Requirements: 3.1, 3.2, 4.5
    """
    from utils.geometry_calculator import compute_geometry
    from utils.provenance_capture import ProvenanceChain  # noqa: F811 — runtime import

    try:
        # If parse failed completely (no accepted, violations present)
        if not parse_result.accepted and parse_result.violations:
            chain.record_terminal(
                stage_name="parsed_pm_decision",
                stage_version=stage_version,
                reason="parse_failure",
            )
            return

        # For each accepted decision, record a provenance event
        for decision in parse_result.accepted:
            # Input contract: truncated raw response for auditability
            input_contract = {
                "raw_response": str(raw_response)[:500] if raw_response else "",
            }

            # Output contract: the parsed decision fields
            output_contract = {
                "candidate_id": decision.candidate_id,
                "decision": decision.decision,
                "risk_multiplier": decision.risk_multiplier,
                "rationale": decision.rationale,
            }

            # Geometry is not fully available at parse stage in candidate-ID mode
            # (prices come from candidate registry, not PM output).
            # Record with incomplete geometry — downstream stages will populate full geometry.
            geometry_before = compute_geometry(None, None, None, None, None)
            geometry_after = compute_geometry(None, None, None, None, None)

            chain.record_event(
                stage_name="parsed_pm_decision",
                stage_version=stage_version,
                input_contract=input_contract,
                output_contract=output_contract,
                fields_changed=["candidate_id", "decision", "risk_multiplier"],
                mutation_reason_code="parse_normalization",
                rule_id=None,
                geometry_before=geometry_before,
                geometry_after=geometry_after,
            )
    except Exception:
        # Fail-open: provenance must never block the pipeline
        logger.error(
            "Failed to record parse provenance: raw_response_type=%s, "
            "accepted=%d, violations=%d",
            type(raw_response).__name__ if raw_response is not None else "None",
            len(parse_result.accepted),
            len(parse_result.violations),
            exc_info=True,
        )
