"""Task writer for pending policy recommendation tasks.

Writes policy recommendations as individual JSON files under the pending_tasks
directory. Task IDs follow the pattern orch_YYYY_MM_DD_NNN where NNN is a
zero-padded 3-digit sequence number for that date.

Uses atomic write pattern (write to .tmp, then os.replace) to prevent
partial files on crash or timeout.
"""

import glob
import json
import os
import re
from datetime import datetime, timezone


def write_pending_task(
    recommendation: dict,
    source_report: str,
    output_dir: str = "reports/local_orchestrator/pending_tasks",
) -> str:
    """Writes a pending task JSON file and returns the task_id.

    Args:
        recommendation: Dict with policy recommendation fields. Expected keys
            include 'title', 'recommendation', 'policy_type',
            'applies_to_candidate_ids'. Optional keys: 'agent', 'type'.
        source_report: Path to the source diagnostic report file.
        output_dir: Directory where task files are written.

    Returns:
        The generated task_id string (e.g. "orch_2026_05_03_001").
    """
    os.makedirs(output_dir, exist_ok=True)

    task_id = _generate_task_id(output_dir)

    payload = {
        "task_id": task_id,
        "status": "pending_review",
        "agent": recommendation.get("agent", "portfolio_manager"),
        "type": recommendation.get("type", "policy_change"),
        "title": recommendation.get("title", ""),
        "source_report": source_report,
        "requires_human_approval": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    output_path = os.path.join(output_dir, f"{task_id}.json")
    tmp_path = f"{output_path}.tmp"

    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, output_path)

    return task_id


def _generate_task_id(output_dir: str) -> str:
    """Generate the next task_id by scanning existing files for today's date.

    Scans output_dir for files matching orch_{today}_*.json, finds the max
    sequence number, and increments. Zero-pads to 3 digits.

    Returns:
        A task_id string like "orch_2026_05_03_001".
    """
    today = datetime.now(timezone.utc).strftime("%Y_%m_%d")
    pattern = os.path.join(output_dir, f"orch_{today}_*.json")
    existing = glob.glob(pattern)

    max_seq = 0
    seq_re = re.compile(rf"orch_{re.escape(today)}_(\d{{3}})\.json$")

    for filepath in existing:
        basename = os.path.basename(filepath)
        match = seq_re.match(basename)
        if match:
            seq = int(match.group(1))
            if seq > max_seq:
                max_seq = seq

    next_seq = max_seq + 1
    return f"orch_{today}_{next_seq:03d}"
