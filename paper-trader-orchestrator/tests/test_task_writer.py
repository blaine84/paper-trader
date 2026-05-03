"""Tests for the task writer module."""

import json
import os
import re

from lib.task_writer import write_pending_task


class TestWritePendingTask:
    """Unit tests for write_pending_task."""

    def test_writes_task_file_with_required_fields(self, tmp_path):
        """Verify all required payload fields are present."""
        recommendation = {
            "policy_type": "cooldown",
            "title": "Add same-symbol overlap guard",
            "recommendation": "Reject new entries when overlap detected.",
            "applies_to_candidate_ids": ["overlap_001"],
        }

        task_id = write_pending_task(
            recommendation,
            source_report="reports/local_orchestrator/latest_overlap_diagnostic.json",
            output_dir=str(tmp_path),
        )

        task_file = tmp_path / f"{task_id}.json"
        assert task_file.exists()

        payload = json.loads(task_file.read_text())
        assert payload["task_id"] == task_id
        assert payload["status"] == "pending_review"
        assert payload["agent"] == "portfolio_manager"
        assert payload["type"] == "policy_change"
        assert payload["title"] == "Add same-symbol overlap guard"
        assert payload["source_report"] == "reports/local_orchestrator/latest_overlap_diagnostic.json"
        assert payload["requires_human_approval"] is True
        assert "created_at" in payload

    def test_task_id_matches_pattern(self, tmp_path):
        """Verify task_id follows orch_YYYY_MM_DD_NNN pattern."""
        recommendation = {"title": "Test task"}

        task_id = write_pending_task(
            recommendation,
            source_report="report.json",
            output_dir=str(tmp_path),
        )

        assert re.match(r"orch_\d{4}_\d{2}_\d{2}_\d{3}$", task_id)

    def test_sequence_increments(self, tmp_path):
        """Verify sequential calls produce incrementing sequence numbers."""
        recommendation = {"title": "Test task"}

        id1 = write_pending_task(recommendation, "report.json", str(tmp_path))
        id2 = write_pending_task(recommendation, "report.json", str(tmp_path))
        id3 = write_pending_task(recommendation, "report.json", str(tmp_path))

        # Extract sequence numbers
        seq1 = int(id1.split("_")[-1])
        seq2 = int(id2.split("_")[-1])
        seq3 = int(id3.split("_")[-1])

        assert seq1 == 1
        assert seq2 == 2
        assert seq3 == 3

    def test_uses_custom_agent_and_type(self, tmp_path):
        """Verify agent and type are taken from recommendation when provided."""
        recommendation = {
            "title": "Custom task",
            "agent": "risk_manager",
            "type": "risk_adjustment",
        }

        task_id = write_pending_task(recommendation, "report.json", str(tmp_path))
        payload = json.loads((tmp_path / f"{task_id}.json").read_text())

        assert payload["agent"] == "risk_manager"
        assert payload["type"] == "risk_adjustment"

    def test_defaults_agent_and_type(self, tmp_path):
        """Verify defaults when agent/type not in recommendation."""
        recommendation = {"title": "Minimal task"}

        task_id = write_pending_task(recommendation, "report.json", str(tmp_path))
        payload = json.loads((tmp_path / f"{task_id}.json").read_text())

        assert payload["agent"] == "portfolio_manager"
        assert payload["type"] == "policy_change"

    def test_creates_output_dir_if_missing(self, tmp_path):
        """Verify output_dir is created when it doesn't exist."""
        nested_dir = str(tmp_path / "nested" / "tasks")
        recommendation = {"title": "Test task"}

        task_id = write_pending_task(recommendation, "report.json", nested_dir)

        assert os.path.isdir(nested_dir)
        assert os.path.isfile(os.path.join(nested_dir, f"{task_id}.json"))

    def test_no_tmp_file_left_behind(self, tmp_path):
        """Verify atomic write leaves no .tmp file."""
        recommendation = {"title": "Test task"}

        write_pending_task(recommendation, "report.json", str(tmp_path))

        tmp_files = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
        assert tmp_files == []
