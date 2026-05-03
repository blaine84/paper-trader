"""Diagnostic runner for same-symbol overlap analysis.

Constructs a prompt from preprocessed overlap candidates, sends it to a
local Ollama LLM, validates the response against the JSON schema, and
enriches findings with deterministic severity data from the preprocessor.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

from lib.validator import validate_diagnostic


logger = logging.getLogger(__name__)

# Resolve project root: this module is at paper-trader-orchestrator/lib/diagnostic_runner.py
# Project root is one level up from lib/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROMPT_TEMPLATE_PATH = _PROJECT_ROOT / "prompts" / "overlap_diagnostic.md"
_DEBUG_DIR = _PROJECT_ROOT / "reports" / "local_orchestrator" / "debug"

# Severity ordering for downgrade detection
_SEVERITY_ORDER = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


class ValidationError(Exception):
    """Raised when LLM output fails schema validation after retry."""
    pass


def _load_prompt_template():
    """Load the overlap diagnostic prompt template from disk."""
    with open(_PROMPT_TEMPLATE_PATH) as f:
        return f.read()


def _build_prompt(template, candidates):
    """Inject candidates JSON into the prompt template."""
    candidates_json = json.dumps(candidates, indent=2)
    return template.replace("{{OVERLAP_CANDIDATES}}", candidates_json)


def _build_repair_prompt(original_prompt, raw_output, errors):
    """Build a repair prompt with validation error messages."""
    error_text = "\n".join(f"- {e}" for e in errors)
    return (
        f"Your previous response failed JSON schema validation. "
        f"Here are the errors:\n\n{error_text}\n\n"
        f"Please fix these errors and return valid JSON only. "
        f"Here is your previous output for reference:\n\n{raw_output}\n\n"
        f"Remember the original instructions:\n\n{original_prompt}"
    )


def _call_ollama(prompt, model, ollama_base_url):
    """Call the Ollama generate API and return the parsed response text.

    Raises:
        ConnectionError: If the Ollama API is unreachable.
        ValueError: If the response cannot be parsed as JSON.
    """
    url = f"{ollama_base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }

    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(
            f"Ollama API unreachable at {ollama_base_url}: {e}"
        ) from e
    except requests.exceptions.RequestException as e:
        raise ConnectionError(
            f"Ollama API request failed: {e}"
        ) from e

    response_data = resp.json()
    raw_text = response_data.get("response", "")

    try:
        return json.loads(raw_text), raw_text
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM response is not valid JSON: {e}\nRaw output: {raw_text[:500]}"
        ) from e


def _save_debug_output(raw_output, errors):
    """Save failed LLM output to the debug directory."""
    os.makedirs(_DEBUG_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    debug_path = _DEBUG_DIR / f"failed_llm_output_{timestamp}.json"

    debug_payload = {
        "timestamp": timestamp,
        "raw_output": raw_output,
        "validation_errors": errors,
    }

    with open(debug_path, "w") as f:
        json.dump(debug_payload, f, indent=2)

    logger.info("Saved failed LLM output to %s", debug_path)
    return debug_path


def _enrich_with_deterministic_severity(diagnostic, candidates):
    """Copy suggested_severity from preprocessor candidates into findings.

    For each finding, look up the matching candidate by candidate_id and
    add the deterministic_suggested_severity field. Also log a warning
    when the LLM's severity is materially lower than the deterministic one.
    """
    # Build a lookup from candidate_id to suggested_severity
    candidate_severity_map = {}
    for candidate in candidates.get("overlap_candidates", []):
        cid = candidate.get("candidate_id")
        if cid:
            candidate_severity_map[cid] = candidate.get("suggested_severity")

    for finding in diagnostic.get("findings", []):
        cid = finding.get("candidate_id")
        deterministic_sev = candidate_severity_map.get(cid)
        if deterministic_sev is not None:
            finding["deterministic_suggested_severity"] = deterministic_sev

            # Check for severity downgrade
            llm_severity = diagnostic.get("severity", "none")
            det_rank = _SEVERITY_ORDER.get(deterministic_sev, 0)
            llm_rank = _SEVERITY_ORDER.get(llm_severity, 0)

            if llm_rank < det_rank:
                logger.warning(
                    "LLM severity downgrade detected for %s: "
                    "deterministic=%s, llm=%s",
                    cid,
                    deterministic_sev,
                    llm_severity,
                )

    return diagnostic


def run_diagnostic(
    candidates,
    model="qwen2.5:14b",
    ollama_base_url="http://localhost:11434",
):
    """Send preprocessed candidates to the local LLM and return validated output.

    If no overlap candidates exist, returns a canned no_overlap_detected
    response without calling the LLM.

    Args:
        candidates: Dict from compute_overlap_candidates() with keys
            'diagnostic', 'window_minutes', 'overlap_candidates'.
        model: Ollama model tag to use.
        ollama_base_url: Base URL for the Ollama API.

    Returns:
        Validated diagnostic output dict.

    Raises:
        ConnectionError: Ollama API unreachable.
        ValidationError: LLM output fails schema after retry.
    """
    # Short-circuit: no overlap candidates → canned response
    if not candidates.get("overlap_candidates"):
        logger.info("No overlap candidates detected — skipping LLM call.")
        return {
            "diagnostic_schema_version": "1.0",
            "run_type": "same_symbol_overlap_diagnostic",
            "verdict": "no_overlap_detected",
            "severity": "none",
            "summary": "No overlap candidates detected in the snapshot.",
            "findings": [],
            "recommended_policy": {
                "policy_type": "no_change",
                "applies_to_candidate_ids": [],
                "title": "No action needed",
                "recommendation": "No same-symbol overlap was detected.",
                "requires_human_approval": True,
            },
        }

    # Load prompt template and inject candidates
    template = _load_prompt_template()
    prompt = _build_prompt(template, candidates)

    logger.info(
        "Calling Ollama API: model=%s, url=%s, candidates=%d",
        model,
        ollama_base_url,
        len(candidates["overlap_candidates"]),
    )

    # First attempt
    diagnostic, raw_text = _call_ollama(prompt, model, ollama_base_url)
    errors = validate_diagnostic(diagnostic)

    if not errors:
        logger.info("LLM output passed validation on first attempt.")
        return _enrich_with_deterministic_severity(diagnostic, candidates)

    # First attempt failed — retry with repair prompt
    logger.warning(
        "LLM output failed validation (%d errors). Retrying with repair prompt.",
        len(errors),
    )
    repair_prompt = _build_repair_prompt(prompt, raw_text, errors)

    try:
        diagnostic_retry, raw_text_retry = _call_ollama(
            repair_prompt, model, ollama_base_url
        )
    except (ValueError, ConnectionError):
        # Retry call itself failed — save original output and raise
        _save_debug_output(raw_text, errors)
        raise ValidationError(
            f"LLM output failed validation and retry call failed. "
            f"Errors: {errors}"
        )

    errors_retry = validate_diagnostic(diagnostic_retry)

    if not errors_retry:
        logger.info("LLM output passed validation on retry.")
        return _enrich_with_deterministic_severity(diagnostic_retry, candidates)

    # Second failure — save debug output and raise
    logger.error(
        "LLM output failed validation after retry (%d errors). "
        "Saving raw output to debug directory.",
        len(errors_retry),
    )
    _save_debug_output(raw_text_retry, errors_retry)
    raise ValidationError(
        f"LLM output failed schema validation after retry. "
        f"Errors: {errors_retry}"
    )
