"""Report writer for overlap diagnostic results.

Renders validated diagnostic output as JSON and human-readable markdown.
Both functions use atomic writes (write to .tmp, then os.replace).
"""

import json
import os


def write_json_report(diagnostic: dict, path: str) -> None:
    """Atomically writes the diagnostic dict as JSON."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(diagnostic, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def write_markdown_report(diagnostic: dict, path: str) -> None:
    """Atomically writes a human-readable markdown report."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    lines = []
    lines.append("# Overlap Diagnostic Report")
    lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")
    lines.append(diagnostic.get("verdict", "unknown"))
    lines.append("")

    # Severity
    lines.append("## Severity")
    lines.append("")
    lines.append(diagnostic.get("severity", "unknown"))
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(diagnostic.get("summary", "No summary provided."))
    lines.append("")

    # Findings
    lines.append("## Findings")
    lines.append("")
    findings = diagnostic.get("findings", [])
    if not findings:
        lines.append("No findings.")
        lines.append("")
    else:
        for finding in findings:
            lines.append(f"### {finding.get('candidate_id', 'unknown')}: {finding.get('symbol', '')} {finding.get('direction', '')}")
            lines.append("")
            lines.append(f"- **Profiles:** {', '.join(finding.get('profiles', []))}")
            lines.append(f"- **Trade IDs:** {', '.join(str(tid) for tid in finding.get('trade_ids', []))}")
            lines.append(f"- **Impact:** {finding.get('impact', 'unknown')}")
            lines.append(f"- **Combined PnL:** {finding.get('combined_pnl', 0)}")
            lines.append(f"- **Confidence:** {finding.get('confidence', 'unknown')}")
            lines.append("")
            evidence = finding.get("evidence", [])
            if evidence:
                lines.append("**Evidence:**")
                lines.append("")
                for item in evidence:
                    lines.append(f"- {item}")
                lines.append("")

    # Recommended Policy
    lines.append("## Recommended Policy")
    lines.append("")
    policy = diagnostic.get("recommended_policy", {})
    if policy:
        lines.append(f"- **Policy Type:** {policy.get('policy_type', 'unknown')}")
        lines.append(f"- **Title:** {policy.get('title', '')}")
        lines.append(f"- **Recommendation:** {policy.get('recommendation', '')}")
        lines.append(f"- **Applies to Candidate IDs:** {', '.join(policy.get('applies_to_candidate_ids', []))}")
    else:
        lines.append("No policy recommendation.")
    lines.append("")

    # Human Review Needed
    lines.append("## Human Review Needed")
    lines.append("")
    lines.append("All policy recommendations require human approval before implementation.")
    lines.append("")

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        f.write("\n".join(lines))
    os.replace(tmp_path, path)
