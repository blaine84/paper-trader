"""
Tests for gather_git_commits and categorize_commit in agents/daily_review.py.
Covers Requirements 3.1, 3.2, 3.3, 3.4, 3.5.
"""

import subprocess
from unittest.mock import patch, MagicMock

import pytest

from agents.daily_review import (
    categorize_commit,
    gather_git_commits,
    _parse_git_log_output,
)


# ---------------------------------------------------------------------------
# categorize_commit — deterministic categorization
# ---------------------------------------------------------------------------


class TestCategorizeCommit:
    """Requirement 3.3: categorize each commit into one of the valid categories."""

    # -- bugfix (highest priority, any file) --
    def test_bugfix_keyword_fix(self):
        assert categorize_commit("Fix analyst RSI threshold", ["agents/analyst.py"]) == "bugfix"

    def test_bugfix_keyword_bug(self):
        assert categorize_commit("bug in edge score calc", ["core/edge_score.py"]) == "bugfix"

    def test_bugfix_keyword_patch(self):
        assert categorize_commit("patch deployment script", ["deploy/setup_pi.sh"]) == "bugfix"

    def test_bugfix_keyword_hotfix(self):
        assert categorize_commit("hotfix for position timer", ["agents/position_timer.py"]) == "bugfix"

    # -- agent_logic (file path) --
    def test_agent_logic_by_file(self):
        assert categorize_commit("Refactor threshold logic", ["agents/analyst.py"]) == "agent_logic"

    # -- agent_logic (message keyword) --
    def test_agent_logic_by_keyword_agent(self):
        assert categorize_commit("Improve agent logging", ["README.md"]) == "agent_logic"

    def test_agent_logic_by_keyword_signal(self):
        assert categorize_commit("Add new signal type", ["README.md"]) == "agent_logic"

    # -- risk_management (file path) --
    def test_risk_management_by_core_file(self):
        assert categorize_commit("Update scoring", ["core/edge_score.py"]) == "risk_management"

    def test_risk_management_by_validator_file(self):
        assert categorize_commit("Tweak limits", ["utils/trade_validator.py"]) == "risk_management"

    # -- risk_management (message keyword) --
    def test_risk_management_by_keyword_risk(self):
        assert categorize_commit("Adjust risk parameters", ["README.md"]) == "risk_management"

    def test_risk_management_by_keyword_position_size(self):
        assert categorize_commit("Change position size logic", ["README.md"]) == "risk_management"

    # -- infrastructure (file path) --
    def test_infrastructure_by_orchestrator(self):
        assert categorize_commit("Add new job", ["orchestrator.py"]) == "infrastructure"

    def test_infrastructure_by_deploy(self):
        assert categorize_commit("Update service file", ["deploy/paper-trader.service"]) == "infrastructure"

    def test_infrastructure_by_db(self):
        assert categorize_commit("Add column", ["db/schema.py"]) == "infrastructure"

    # -- infrastructure (message keyword) --
    def test_infrastructure_by_keyword_deploy(self):
        assert categorize_commit("deploy new version", ["README.md"]) == "infrastructure"

    def test_infrastructure_by_keyword_migrate(self):
        assert categorize_commit("migrate database tables", ["README.md"]) == "infrastructure"

    # -- strategy (file path) --
    def test_strategy_by_strategies_file(self):
        assert categorize_commit("Add momentum setup", ["models/strategies.py"]) == "strategy"

    def test_strategy_by_strategy_store(self):
        assert categorize_commit("Persist new setup", ["utils/strategy_store.py"]) == "strategy"

    # -- strategy (message keyword) --
    def test_strategy_by_keyword(self):
        assert categorize_commit("New backtest results", ["README.md"]) == "strategy"

    # -- other (fallback) --
    def test_other_fallback(self):
        assert categorize_commit("Update readme", ["README.md"]) == "other"

    def test_other_empty_files(self):
        assert categorize_commit("Update readme", []) == "other"

    # -- bugfix takes priority over file-based match --
    def test_bugfix_priority_over_agent_logic(self):
        assert categorize_commit("Fix agent crash", ["agents/analyst.py"]) == "bugfix"

    # -- file-based match takes priority over keyword-based match --
    def test_file_priority_over_keyword(self):
        # File matches agent_logic, message has "risk" keyword
        assert categorize_commit("Adjust risk in agent", ["agents/analyst.py"]) == "agent_logic"

    # -- valid output set --
    def test_always_returns_valid_category(self):
        valid = {"agent_logic", "risk_management", "infrastructure", "strategy", "bugfix", "other"}
        for msg, files in [
            ("random stuff", []),
            ("fix something", ["x.py"]),
            ("", []),
            ("AGENT uppercase", ["agents/foo.py"]),
        ]:
            assert categorize_commit(msg, files) in valid


# ---------------------------------------------------------------------------
# _parse_git_log_output — parsing raw git log text
# ---------------------------------------------------------------------------


class TestParseGitLogOutput:
    """Requirement 3.2: parse each commit to extract message, author, timestamp, files."""

    def test_single_commit(self):
        raw = (
            "abc123|John Doe|2026-03-22T14:30:00-04:00|Fix analyst RSI threshold\n"
            "agents/analyst.py\n"
            "core/edge_score.py\n"
        )
        commits = _parse_git_log_output(raw)
        assert len(commits) == 1
        c = commits[0]
        assert c["hash"] == "abc123"
        assert c["author"] == "John Doe"
        assert c["timestamp"] == "2026-03-22T14:30:00-04:00"
        assert c["message"] == "Fix analyst RSI threshold"
        assert c["files"] == ["agents/analyst.py", "core/edge_score.py"]
        assert c["category"] == "bugfix"  # "Fix" keyword

    def test_multiple_commits(self):
        raw = (
            "abc123|John|2026-03-22T14:30:00-04:00|Fix analyst RSI\n"
            "agents/analyst.py\n"
            "\n"
            "def456|Jane|2026-03-22T10:15:00-04:00|Add new strategy\n"
            "models/strategies.py\n"
        )
        commits = _parse_git_log_output(raw)
        assert len(commits) == 2
        assert commits[0]["hash"] == "abc123"
        assert commits[1]["hash"] == "def456"
        assert commits[1]["category"] == "strategy"

    def test_empty_output(self):
        assert _parse_git_log_output("") == []
        assert _parse_git_log_output("\n") == []

    def test_commit_with_no_files(self):
        raw = "abc123|John|2026-03-22T14:30:00-04:00|Update readme\n\n"
        commits = _parse_git_log_output(raw)
        assert len(commits) == 1
        assert commits[0]["files"] == []

    def test_pipe_in_message(self):
        """Message may contain pipes — split("|", 3) handles this."""
        raw = "abc123|John|2026-03-22T14:30:00-04:00|Fix edge|case handling\nREADME.md\n"
        commits = _parse_git_log_output(raw)
        assert len(commits) == 1
        assert commits[0]["message"] == "Fix edge|case handling"


# ---------------------------------------------------------------------------
# gather_git_commits — subprocess integration
# ---------------------------------------------------------------------------


class TestGatherGitCommits:
    """Requirements 3.1, 3.4, 3.5: retrieve commits, handle no-commits and failures."""

    @patch("agents.daily_review.subprocess.run")
    def test_successful_parse(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="abc123|John|2026-03-22T14:30:00-04:00|Add agent logging\nagents/analyst.py\n",
            stderr="",
        )
        commits = gather_git_commits("2026-03-21")
        assert len(commits) == 1
        assert commits[0]["hash"] == "abc123"
        mock_run.assert_called_once()

    @patch("agents.daily_review.subprocess.run")
    def test_no_commits(self, mock_run):
        """Req 3.4: no commits returns empty list."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        commits = gather_git_commits("2026-03-21")
        assert commits == []

    @patch("agents.daily_review.subprocess.run", side_effect=FileNotFoundError)
    def test_git_not_found(self, mock_run):
        """Req 3.5: git unavailable returns empty list."""
        commits = gather_git_commits("2026-03-21")
        assert commits == []

    @patch("agents.daily_review.subprocess.run")
    def test_non_zero_exit(self, mock_run):
        """Req 3.5: git error returns empty list."""
        mock_run.return_value = MagicMock(returncode=128, stdout="", stderr="fatal: not a git repository")
        commits = gather_git_commits("2026-03-21")
        assert commits == []

    @patch("agents.daily_review.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="git", timeout=30))
    def test_timeout(self, mock_run):
        """Req 3.5: timeout returns empty list."""
        commits = gather_git_commits("2026-03-21")
        assert commits == []
