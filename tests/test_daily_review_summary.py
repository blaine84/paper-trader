"""Tests for build_deterministic_summary in agents/daily_review.py."""

from datetime import date
from agents.daily_review import build_deterministic_summary


class TestBuildDeterministicSummary:
    """Unit tests for the deterministic summary builder."""

    def test_all_sources_present_high_confidence(self):
        """When all sources are present, confidence should be 'high'."""
        trade_perf = {
            "total_trades": 3, "wins": 2, "losses": 1,
            "total_pnl": 100.0, "no_trades": False,
        }
        git_commits = [
            {"hash": "abc", "author": "dev", "timestamp": "2025-01-01T10:00:00",
             "message": "Fix bug", "files": ["agents/analyst.py"], "category": "bugfix"},
        ]
        agent_context = {
            "market_context": "Risk-on day",
            "selection_feedback": "Good picks",
            "execution_feedback": {"score": 8},
            "analyst_signals": {"AAPL": {"rsi": 55}},
            "missing_sources": [],
        }
        cases = [{"symbol": "AAPL", "outcome": "win"}]
        previous_review = {"market_summary": "Yesterday was calm"}

        result = build_deterministic_summary(
            trade_perf, git_commits, agent_context, cases, previous_review
        )

        assert result["date"] == date.today().isoformat()
        assert result["trade_performance"] == trade_perf
        assert result["git_changes"]["total_commits"] == 1
        assert result["git_changes"]["categories"] == {"bugfix": 1}
        assert result["git_changes"]["no_commits"] is False
        assert result["agent_context"]["market_context"] == "Risk-on day"
        assert result["cases_today"] == cases
        assert result["previous_review_summary"] == "Yesterday was calm"
        assert result["completeness"]["confidence"] == "high"
        assert all(
            result["completeness"][k] is True
            for k in ["trade_data", "git_data", "researcher_context",
                       "reviewer_feedback", "analyst_signals", "previous_review"]
        )

    def test_no_sources_low_confidence(self):
        """When no sources are present, confidence should be 'low'."""
        result = build_deterministic_summary(None, None, None, None, None)

        assert result["trade_performance"] == {}
        assert result["git_changes"]["total_commits"] == 0
        assert result["git_changes"]["no_commits"] is True
        assert result["cases_today"] == []
        assert result["previous_review_summary"] is None
        assert result["completeness"]["confidence"] == "low"
        assert all(
            result["completeness"][k] is False
            for k in ["trade_data", "git_data", "researcher_context",
                       "reviewer_feedback", "analyst_signals", "previous_review"]
        )

    def test_no_trades_flag_means_no_trade_data(self):
        """trade_data completeness should be False when no_trades is True."""
        trade_perf = {"total_trades": 0, "no_trades": True}
        result = build_deterministic_summary(trade_perf, [], {}, [], None)
        assert result["completeness"]["trade_data"] is False

    def test_medium_confidence_some_missing(self):
        """When 4 of 6 sources present, confidence should be 'medium'."""
        trade_perf = {"total_trades": 2, "no_trades": False}
        git_commits = [{"hash": "a", "author": "x", "timestamp": "t",
                        "message": "m", "files": [], "category": "other"}]
        agent_context = {
            "market_context": "context",
            "selection_feedback": "feedback",
            "execution_feedback": None,
            "analyst_signals": None,
            "missing_sources": ["execution_feedback", "analyst_signals"],
        }
        result = build_deterministic_summary(
            trade_perf, git_commits, agent_context, [], None
        )
        # trade_data=True, git_data=True, researcher=True, reviewer=True (selection_feedback present),
        # analyst=False, previous=False => 4 of 6 => medium
        assert result["completeness"]["confidence"] == "medium"

    def test_git_categories_counted(self):
        """Git commit categories should be counted correctly."""
        commits = [
            {"hash": "1", "author": "a", "timestamp": "t", "message": "m",
             "files": [], "category": "bugfix"},
            {"hash": "2", "author": "a", "timestamp": "t", "message": "m",
             "files": [], "category": "bugfix"},
            {"hash": "3", "author": "a", "timestamp": "t", "message": "m",
             "files": [], "category": "agent_logic"},
        ]
        result = build_deterministic_summary(None, commits, None, None, None)
        assert result["git_changes"]["categories"] == {"bugfix": 2, "agent_logic": 1}
        assert result["git_changes"]["total_commits"] == 3

    def test_previous_review_summary_extracts_market_summary(self):
        """previous_review_summary should come from the market_summary field."""
        prev = {"market_summary": "Bull run", "other_field": "ignored"}
        result = build_deterministic_summary(None, [], {}, [], prev)
        assert result["previous_review_summary"] == "Bull run"
        assert result["completeness"]["previous_review"] is True

    def test_previous_review_without_market_summary(self):
        """If previous_review exists but has no market_summary, summary is None."""
        prev = {"outlook": "something"}
        result = build_deterministic_summary(None, [], {}, [], prev)
        assert result["previous_review_summary"] is None
        assert result["completeness"]["previous_review"] is True

    def test_reviewer_feedback_true_with_only_execution_feedback(self):
        """reviewer_feedback should be True if only execution_feedback is present."""
        ctx = {
            "market_context": None,
            "selection_feedback": None,
            "execution_feedback": {"score": 7},
            "analyst_signals": None,
            "missing_sources": [],
        }
        result = build_deterministic_summary(None, [], ctx, [], None)
        assert result["completeness"]["reviewer_feedback"] is True

    def test_boundary_three_sources_is_low(self):
        """Exactly 3 of 6 sources present should be 'low' confidence."""
        trade_perf = {"total_trades": 1, "no_trades": False}
        commits = [{"hash": "a", "author": "x", "timestamp": "t",
                     "message": "m", "files": [], "category": "other"}]
        ctx = {"market_context": "ctx", "selection_feedback": None,
               "execution_feedback": None, "analyst_signals": None,
               "missing_sources": []}
        # trade_data=True, git_data=True, researcher=True, reviewer=False,
        # analyst=False, previous=False => 3 of 6 => low
        result = build_deterministic_summary(trade_perf, commits, ctx, [], None)
        assert result["completeness"]["confidence"] == "low"

    def test_boundary_four_sources_is_medium(self):
        """Exactly 4 of 6 sources present should be 'medium' confidence."""
        trade_perf = {"total_trades": 1, "no_trades": False}
        commits = [{"hash": "a", "author": "x", "timestamp": "t",
                     "message": "m", "files": [], "category": "other"}]
        ctx = {"market_context": "ctx", "selection_feedback": "fb",
               "execution_feedback": None, "analyst_signals": None,
               "missing_sources": []}
        prev = {"market_summary": "prev"}
        # trade_data=True, git_data=True, researcher=True, reviewer=True,
        # analyst=False, previous=True => 5 of 6 => medium
        result = build_deterministic_summary(trade_perf, commits, ctx, [], prev)
        assert result["completeness"]["confidence"] == "medium"
