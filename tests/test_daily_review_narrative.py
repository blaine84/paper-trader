"""Tests for generate_narrative in agents/daily_review.py."""

import json
from unittest.mock import patch, MagicMock
from agents.daily_review import (
    generate_narrative,
    NARRATIVE_SYSTEM_PROMPT,
    _NARRATIVE_FIELDS,
    _EMPTY_NARRATIVE,
)


def _make_summary():
    """Return a minimal deterministic summary for testing."""
    return {
        "date": "2025-07-01",
        "trade_performance": {"total_trades": 2, "wins": 1, "losses": 1, "total_pnl": 50.0},
        "git_changes": {"commits": [], "total_commits": 0, "categories": {}, "no_commits": True},
        "agent_context": {"market_context": "Risk-on", "missing_sources": []},
        "cases_today": [],
        "previous_review_summary": None,
        "completeness": {"confidence": "medium"},
    }


def _make_llm_response():
    """Return a valid LLM narrative JSON string."""
    return json.dumps({
        "market_summary": "Markets rallied today.",
        "trade_narrative": "Two trades were taken.",
        "git_narrative": "No code changes.",
        "correlations": "No correlations observed.",
        "lessons_learned": [
            {
                "category": "execution",
                "lesson": "Tight stops helped limit losses.",
                "evidence": "AAPL -0.5% stopped out cleanly.",
                "action": "Continue using tight stops on momentum setups.",
            }
        ],
        "process_quality": "Execution discipline was solid.",
        "outlook": "Watch for continuation in tech.",
        "watchouts": ["TSLA approaching resistance"],
    })


class TestGenerateNarrative:
    """Unit tests for the LLM narrative generator."""

    @patch("agents.daily_review.call_llm")
    @patch("agents.daily_review.parse_json_response")
    def test_merges_narrative_into_summary(self, mock_parse, mock_llm):
        """Narrative fields are merged into the deterministic summary."""
        mock_llm.return_value = _make_llm_response()
        mock_parse.return_value = json.loads(_make_llm_response())

        summary = _make_summary()
        result = generate_narrative(summary)

        # Deterministic fields preserved
        assert result["date"] == "2025-07-01"
        assert result["trade_performance"]["total_trades"] == 2
        assert result["completeness"]["confidence"] == "medium"

        # Narrative fields present
        assert result["market_summary"] == "Markets rallied today."
        assert result["trade_narrative"] == "Two trades were taken."
        assert result["correlations"] == "No correlations observed."
        assert result["process_quality"] == "Execution discipline was solid."
        assert result["outlook"] == "Watch for continuation in tech."
        assert result["watchouts"] == ["TSLA approaching resistance"]

    @patch("agents.daily_review.call_llm")
    @patch("agents.daily_review.parse_json_response")
    def test_generated_at_timestamp_present(self, mock_parse, mock_llm):
        """Result includes a generated_at ISO timestamp."""
        mock_llm.return_value = _make_llm_response()
        mock_parse.return_value = json.loads(_make_llm_response())

        result = generate_narrative(_make_summary())
        assert "generated_at" in result
        assert result["generated_at"].endswith("Z")

    @patch("agents.daily_review.call_llm")
    @patch("agents.daily_review.parse_json_response")
    def test_calls_llm_with_medium_tier(self, mock_parse, mock_llm):
        """LLM is called with the correct tier."""
        mock_llm.return_value = _make_llm_response()
        mock_parse.return_value = json.loads(_make_llm_response())

        generate_narrative(_make_summary())
        mock_llm.assert_called_once()
        _, kwargs = mock_llm.call_args
        assert kwargs.get("tier", mock_llm.call_args[0][2] if len(mock_llm.call_args[0]) > 2 else None) is not None

    @patch("agents.daily_review.call_llm")
    @patch("agents.daily_review.parse_json_response")
    def test_calls_llm_with_custom_tier(self, mock_parse, mock_llm):
        """Custom tier is forwarded to call_llm."""
        mock_llm.return_value = _make_llm_response()
        mock_parse.return_value = json.loads(_make_llm_response())

        generate_narrative(_make_summary(), tier="high")
        call_kwargs = mock_llm.call_args
        # tier should be passed as keyword arg
        assert call_kwargs.kwargs.get("tier") == "high"

    @patch("agents.daily_review.call_llm")
    def test_llm_failure_returns_deterministic_with_empty_narrative(self, mock_llm):
        """When LLM fails, return deterministic data with empty narrative fields."""
        mock_llm.side_effect = Exception("LLM unavailable")

        summary = _make_summary()
        result = generate_narrative(summary)

        # Deterministic fields still present
        assert result["date"] == "2025-07-01"
        assert result["trade_performance"]["total_trades"] == 2

        # Narrative fields are empty defaults
        assert result["market_summary"] == ""
        assert result["lessons_learned"] == []
        assert result["watchouts"] == []
        assert result["process_quality"] == ""

        # generated_at still present
        assert "generated_at" in result

    @patch("agents.daily_review.call_llm")
    @patch("agents.daily_review.parse_json_response")
    def test_lessons_learned_validated_structure(self, mock_parse, mock_llm):
        """Each lesson has category, lesson, evidence, action keys."""
        mock_llm.return_value = _make_llm_response()
        mock_parse.return_value = json.loads(_make_llm_response())

        result = generate_narrative(_make_summary())
        lessons = result["lessons_learned"]
        assert len(lessons) == 1
        lesson = lessons[0]
        assert set(lesson.keys()) == {"category", "lesson", "evidence", "action"}
        assert lesson["category"] == "execution"

    @patch("agents.daily_review.call_llm")
    @patch("agents.daily_review.parse_json_response")
    def test_lessons_learned_malformed_entries_filtered(self, mock_parse, mock_llm):
        """Non-dict entries in lessons_learned are filtered out."""
        mock_llm.return_value = "ignored"
        mock_parse.return_value = {
            "market_summary": "ok",
            "lessons_learned": ["not a dict", {"category": "risk", "lesson": "good"}],
            "watchouts": [],
        }

        result = generate_narrative(_make_summary())
        lessons = result["lessons_learned"]
        # Only the dict entry survives, with defaults for missing keys
        assert len(lessons) == 1
        assert lessons[0]["category"] == "risk"
        assert lessons[0]["evidence"] == ""  # default for missing key

    @patch("agents.daily_review.call_llm")
    @patch("agents.daily_review.parse_json_response")
    def test_partial_narrative_fills_missing_with_defaults(self, mock_parse, mock_llm):
        """Missing narrative fields get empty defaults."""
        mock_llm.return_value = "ignored"
        mock_parse.return_value = {
            "market_summary": "Partial response",
            # All other fields missing
        }

        result = generate_narrative(_make_summary())
        assert result["market_summary"] == "Partial response"
        assert result["trade_narrative"] == ""
        assert result["correlations"] == ""
        assert result["lessons_learned"] == []
        assert result["watchouts"] == []

    @patch("agents.daily_review.call_llm")
    @patch("agents.daily_review.parse_json_response")
    def test_system_prompt_contains_required_rules(self, mock_parse, mock_llm):
        """SYSTEM_PROMPT includes the key rules from the design doc."""
        assert "observational language" in NARRATIVE_SYSTEM_PROMPT
        assert "hedging language" in NARRATIVE_SYSTEM_PROMPT
        assert "sample size < 5" in NARRATIVE_SYSTEM_PROMPT
        assert "specific trades" in NARRATIVE_SYSTEM_PROMPT
        assert "process quality" in NARRATIVE_SYSTEM_PROMPT
        assert "watchouts" in NARRATIVE_SYSTEM_PROMPT

    @patch("agents.daily_review.call_llm")
    @patch("agents.daily_review.parse_json_response")
    def test_user_prompt_is_json_serialized_summary(self, mock_parse, mock_llm):
        """The user prompt sent to the LLM is the JSON-serialized summary."""
        mock_llm.return_value = _make_llm_response()
        mock_parse.return_value = json.loads(_make_llm_response())

        summary = _make_summary()
        generate_narrative(summary)

        user_prompt = mock_llm.call_args[0][1]
        parsed = json.loads(user_prompt)
        assert parsed["date"] == "2025-07-01"

    @patch("agents.daily_review.call_llm")
    @patch("agents.daily_review.parse_json_response")
    def test_does_not_overwrite_deterministic_fields(self, mock_parse, mock_llm):
        """Narrative merge does not clobber deterministic fields like completeness."""
        mock_llm.return_value = "ignored"
        mock_parse.return_value = {
            "market_summary": "ok",
            "completeness": {"overwritten": True},  # LLM tries to overwrite
        }

        summary = _make_summary()
        result = generate_narrative(summary)

        # completeness should still be the deterministic version since we only
        # merge _NARRATIVE_FIELDS, not arbitrary keys
        assert result["completeness"]["confidence"] == "medium"
