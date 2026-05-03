"""Validator for LLM diagnostic output against the overlap diagnostic JSON schema."""

import json
from pathlib import Path

import jsonschema


# Resolve schema path relative to this module's location.
# Module is at paper-trader-orchestrator/lib/validator.py
# Schema is at paper-trader-orchestrator/schemas/overlap_diagnostic.schema.json
_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "overlap_diagnostic.schema.json"


def _load_schema() -> dict:
    """Load the JSON schema from disk."""
    with open(_SCHEMA_PATH) as f:
        return json.load(f)


def validate_diagnostic(output: dict) -> list[str]:
    """
    Validate the LLM diagnostic output against the JSON schema and
    perform post-schema semantic validation.

    Returns an empty list if valid, or a list of error message strings
    if invalid.
    """
    errors: list[str] = []

    # --- JSON Schema validation ---
    schema = _load_schema()
    validator = jsonschema.Draft7Validator(schema)
    for error in sorted(validator.iter_errors(output), key=lambda e: list(e.path)):
        errors.append(error.message)

    # If schema validation failed, return early — semantic checks
    # assume the structure is at least well-formed.
    if errors:
        return errors

    # --- Post-schema semantic validation ---
    # Every value in recommended_policy.applies_to_candidate_ids must
    # match a candidate_id in the findings array.
    finding_ids = {f["candidate_id"] for f in output.get("findings", [])}
    policy = output.get("recommended_policy", {})
    for ref_id in policy.get("applies_to_candidate_ids", []):
        if ref_id not in finding_ids:
            errors.append(
                f"recommended_policy.applies_to_candidate_ids references "
                f"'{ref_id}' which does not match any candidate_id in findings"
            )

    return errors
