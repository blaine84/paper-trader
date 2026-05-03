"""CLI entry point for the overlap diagnostic runner.

Orchestrates: scp snapshot pull → validate → preprocess → LLM diagnose →
write reports → optionally write pending tasks.

Usage:
    python run_overlap_diagnostic.py \
        --ssh-host pi \
        --model qwen2.5:14b \
        --ollama-url http://localhost:11434 \
        --window-minutes 120 \
        --trading-timezone America/New_York \
        --write-tasks
"""

import argparse
import json
import logging
import subprocess
import sys
import time

from lib.diagnostic_runner import ValidationError, run_diagnostic
from lib.preprocessor import compute_overlap_candidates
from lib.report_writer import write_json_report, write_markdown_report
from lib.task_writer import write_pending_task


logger = logging.getLogger(__name__)

# Report output paths (relative to project root)
_JSON_REPORT_PATH = "reports/local_orchestrator/latest_overlap_diagnostic.json"
_MD_REPORT_PATH = "reports/local_orchestrator/latest_overlap_diagnostic.md"
_PENDING_TASKS_DIR = "reports/local_orchestrator/pending_tasks"

# Required top-level keys in the snapshot
_REQUIRED_SNAPSHOT_KEYS = {
    "trades",
    "trade_events",
    "dynamic_strategies",
    "scope",
    "snapshot_schema_version",
}


def validate_snapshot(snapshot):
    """Validate the top-level structure of a snapshot dict.

    Checks that all required keys are present and have the correct types.
    Specifically validates that trade_events is an object with
    'included_event_types' (list) and 'rows' (list) keys, not a flat array.

    Args:
        snapshot: The parsed snapshot dict.

    Returns:
        A list of error message strings. Empty if valid.
    """
    errors = []

    if not isinstance(snapshot, dict):
        return ["Snapshot must be a JSON object (dict), got %s" % type(snapshot).__name__]

    # Check required top-level keys
    missing = _REQUIRED_SNAPSHOT_KEYS - set(snapshot.keys())
    if missing:
        errors.append("Missing required top-level keys: %s" % ", ".join(sorted(missing)))

    # Validate 'trades' is a list
    if "trades" in snapshot:
        if not isinstance(snapshot["trades"], list):
            errors.append(
                "'trades' must be a list, got %s" % type(snapshot["trades"]).__name__
            )

    # Validate 'trade_events' is a dict with required sub-keys
    if "trade_events" in snapshot:
        te = snapshot["trade_events"]
        if isinstance(te, list):
            errors.append(
                "'trade_events' must be an object with 'included_event_types' and "
                "'rows' keys, not a flat array"
            )
        elif not isinstance(te, dict):
            errors.append(
                "'trade_events' must be an object, got %s" % type(te).__name__
            )
        else:
            if "included_event_types" not in te:
                errors.append(
                    "'trade_events' is missing required key 'included_event_types'"
                )
            elif not isinstance(te["included_event_types"], list):
                errors.append(
                    "'trade_events.included_event_types' must be a list, got %s"
                    % type(te["included_event_types"]).__name__
                )

            if "rows" not in te:
                errors.append("'trade_events' is missing required key 'rows'")
            elif not isinstance(te["rows"], list):
                errors.append(
                    "'trade_events.rows' must be a list, got %s"
                    % type(te["rows"]).__name__
                )

    # Validate 'dynamic_strategies' is a list
    if "dynamic_strategies" in snapshot:
        if not isinstance(snapshot["dynamic_strategies"], list):
            errors.append(
                "'dynamic_strategies' must be a list, got %s"
                % type(snapshot["dynamic_strategies"]).__name__
            )

    # Validate 'snapshot_schema_version' is a string
    if "snapshot_schema_version" in snapshot:
        if not isinstance(snapshot["snapshot_schema_version"], str):
            errors.append(
                "'snapshot_schema_version' must be a string, got %s"
                % type(snapshot["snapshot_schema_version"]).__name__
            )

    return errors


def _should_write_task(diagnostic):
    """Check whether all gating conditions are met for writing a pending task.

    Conditions:
    - verdict is 'overlap_detected'
    - severity is one of {'medium', 'high', 'critical'}
    - policy_type is one of {'cooldown', 'max_concurrent_profiles', 'throttle_size'}
    - at least one finding has impact 'harmful' or 'unclear'

    Returns:
        True if all conditions are met, False otherwise.
    """
    if diagnostic.get("verdict") != "overlap_detected":
        return False

    if diagnostic.get("severity") not in {"medium", "high", "critical"}:
        return False

    policy = diagnostic.get("recommended_policy", {})
    if policy.get("policy_type") not in {
        "cooldown",
        "max_concurrent_profiles",
        "throttle_size",
    }:
        return False

    findings = diagnostic.get("findings", [])
    has_actionable_impact = any(
        f.get("impact") in {"harmful", "unclear"} for f in findings
    )
    if not has_actionable_impact:
        return False

    return True


def main():
    """Main entry point for the overlap diagnostic CLI."""
    parser = argparse.ArgumentParser(
        description="Run overlap diagnostic on a Pi snapshot using a local LLM."
    )
    parser.add_argument(
        "--ssh-host",
        required=True,
        help="SSH host alias for the Pi",
    )
    parser.add_argument(
        "--remote-snapshot",
        default="/home/blaine/paper-trader/reports/orchestration_snapshots/latest.json",
        help="Remote path to snapshot on Pi (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:14b",
        help="Ollama model tag (default: %(default)s)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=120,
        help="Overlap window in minutes (default: %(default)s)",
    )
    parser.add_argument(
        "--trading-timezone",
        default="America/New_York",
        help="IANA timezone string (default: %(default)s)",
    )
    parser.add_argument(
        "--write-tasks",
        action="store_true",
        default=False,
        help="Enable pending task file generation (default: report-only mode)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    start_time = time.time()

    # Step 1: Log model name and Ollama URL
    logger.info("Model: %s, Ollama URL: %s", args.model, args.ollama_url)

    # Step 2: scp snapshot from Pi
    local_snapshot_path = "snapshots/latest.json"
    scp_source = "%s:%s" % (args.ssh_host, args.remote_snapshot)
    logger.info("Pulling snapshot: scp %s %s", scp_source, local_snapshot_path)

    result = subprocess.run(
        ["scp", scp_source, local_snapshot_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(
            "scp failed with exit code %d: %s",
            result.returncode,
            result.stderr.strip(),
        )
        sys.exit(1)

    logger.info("Snapshot transferred successfully.")

    # Step 3: Load and validate snapshot JSON structure
    try:
        with open(local_snapshot_path) as f:
            snapshot = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load snapshot JSON: %s", e)
        sys.exit(1)

    validation_errors = validate_snapshot(snapshot)
    if validation_errors:
        for err in validation_errors:
            logger.error("Snapshot validation error: %s", err)
        sys.exit(1)

    logger.info("Snapshot loaded and validated.")

    # Step 4: Compute overlap candidates
    candidates = compute_overlap_candidates(
        snapshot,
        window_minutes=args.window_minutes,
        trading_timezone=args.trading_timezone,
    )
    candidate_count = len(candidates.get("overlap_candidates", []))
    logger.info("Preprocessor found %d overlap candidate(s).", candidate_count)

    # Step 5: Run diagnostic (skips LLM if no candidates)
    try:
        diagnostic = run_diagnostic(
            candidates,
            model=args.model,
            ollama_base_url=args.ollama_url,
        )
    except ConnectionError as e:
        logger.error("Ollama connection error: %s", e)
        sys.exit(1)
    except ValidationError as e:
        logger.error("LLM output validation failed: %s", e)
        sys.exit(1)

    # Step 6: Write reports
    write_json_report(diagnostic, _JSON_REPORT_PATH)
    write_markdown_report(diagnostic, _MD_REPORT_PATH)
    logger.info("Reports written to %s and %s", _JSON_REPORT_PATH, _MD_REPORT_PATH)

    # Step 7: Conditionally write pending task
    if args.write_tasks:
        if _should_write_task(diagnostic):
            policy = diagnostic.get("recommended_policy", {})
            task_id = write_pending_task(
                recommendation=policy,
                source_report=_JSON_REPORT_PATH,
                output_dir=_PENDING_TASKS_DIR,
            )
            logger.info("Pending task written: %s", task_id)
        else:
            logger.info(
                "Gating conditions not met for task generation — skipping."
            )
    else:
        logger.info("Report-only mode (--write-tasks not passed) — skipping task generation.")

    # Step 8: Log run outcome
    duration = time.time() - start_time
    logger.info(
        "Run complete: verdict=%s, candidates=%d, duration=%.1fs",
        diagnostic.get("verdict", "unknown"),
        candidate_count,
        duration,
    )


if __name__ == "__main__":
    main()
