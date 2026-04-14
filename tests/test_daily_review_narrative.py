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
        "executive_summary": "Net +$50 on 2 trades. Tight stops saved the day.",
        "day_classification": "modest_win",
        "primary_driver": "Disciplined stop management on AAPL limited downside.",
        "driver_ranking": [
            {"driver": "Stop discipline on AAPL", "impact": "+$50", "controllable": True}
        ],
        "performance_story": "Two trades were taken. One won, one lost. Net positive.",
        "system_observations": "Agents coordinated well on entry signals.",
        "what_worked": ["Tight stops on momentum setups"],
        "what_failed": ["Late entry on second trade"],
        "highest_leverage_fix": "Improve entry timing on momentum setups.",
        "lessons_learned": [
            {
                "category": "execution",
                "lesson": "Tight stops helped limit losses.",
                "evidence": "AAPL -0.5% stopped out cleanly.",
                "action": "Continue using tight stops on momentum setups.",
            }
        ],
        "process_quality": "Execution discipline was solid.",
        "correlations": "No correlations observed.",
        "git_narrative": "No code changes.",
        "tomorrows_focus": ["Monitor AAPL continuation"],
        "watchouts": ["TSLA approaching resistance"],
        "email_subject": "Modest win: +$50 on 2 trades",
        "email_preview": "Tight stops saved the day with a net +$50 across 2 trades.",
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
        assert result["executive_summary"] == "Net +$50 on 2 trades. Tight stops saved the day."
        assert result["day_classification"] == "modest_win"
        assert result["primary_driver"] == "Disciplined stop management on AAPL limited downside."
        assert result["correlations"] == "No correlations observed."
        assert result["process_quality"] == "Execution discipline was solid."
        assert result["watchouts"] == ["TSLA approaching resistance"]
        assert result["what_worked"] == ["Tight stops on momentum setups"]
        assert result["what_failed"] == ["Late entry on second trade"]
        assert result["highest_leverage_fix"] == "Improve entry timing on momentum setups."
        assert result["tomorrows_focus"] == ["Monitor AAPL continuation"]

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
        """LLM is called twice (diagnostic + analysis) with the correct tier."""
        mock_llm.return_value = _make_llm_response()
        mock_parse.return_value = json.loads(_make_llm_response())

        generate_narrative(_make_summary())
        assert mock_llm.call_count == 2
        for call_args in mock_llm.call_args_list:
            assert call_args.kwargs.get("tier") == "medium"

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
        assert result["executive_summary"] == ""
        assert result["day_classification"] == "breakeven"
        assert result["primary_driver"] == ""
        assert result["lessons_learned"] == []
        assert result["watchouts"] == []
        assert result["process_quality"] == ""
        assert result["what_worked"] == []
        assert result["what_failed"] == []
        assert result["tomorrows_focus"] == []

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
            "executive_summary": "ok",
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
            "executive_summary": "Partial response",
            # All other fields missing
        }

        result = generate_narrative(_make_summary())
        assert result["executive_summary"] == "Partial response"
        assert result["performance_story"] == ""
        assert result["correlations"] == ""
        assert result["lessons_learned"] == []
        assert result["watchouts"] == []
        assert result["day_classification"] == "breakeven"  # default

    @patch("agents.daily_review.call_llm")
    @patch("agents.daily_review.parse_json_response")
    def test_system_prompt_contains_required_rules(self, mock_parse, mock_llm):
        """System prompts include the key rules from the design doc."""
        from agents.daily_review import _DIAGNOSTIC_PROMPT, _ANALYSIS_PROMPT
        for prompt in [_DIAGNOSTIC_PROMPT, _ANALYSIS_PROMPT]:
            assert "observational" in prompt
            assert "sample size < 5" in prompt
            assert "specific trades" in prompt
            assert "process quality" in prompt

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
            "executive_summary": "ok",
            "completeness": {"overwritten": True},  # LLM tries to overwrite
        }

        summary = _make_summary()
        result = generate_narrative(summary)

        # completeness should still be the deterministic version since we only
        # merge _NARRATIVE_FIELDS, not arbitrary keys
        assert result["completeness"]["confidence"] == "medium"

    @patch("agents.daily_review.call_llm")
    @patch("agents.daily_review.parse_json_response")
    def test_invalid_day_classification_defaults_to_breakeven(self, mock_parse, mock_llm):
        """Invalid day_classification values are corrected to 'breakeven'."""
        mock_llm.return_value = "ignored"
        mock_parse.return_value = {
            "executive_summary": "ok",
            "day_classification": "invalid_value",
        }

        result = generate_narrative(_make_summary())
        assert result["day_classification"] == "breakeven"

    @patch("agents.daily_review.call_llm")
    @patch("agents.daily_review.parse_json_response")
    def test_valid_day_classifications_accepted(self, mock_parse, mock_llm):
        """All valid day_classification values are preserved."""
        for classification in ["strong_win", "modest_win", "breakeven", "modest_loss", "bad_day", "system_failure"]:
            mock_llm.return_value = "ignored"
            mock_parse.return_value = {
                "executive_summary": "ok",
                "day_classification": classification,
            }
            result = generate_narrative(_make_summary())
            assert result["day_classification"] == classification

    @patch("agents.daily_review.call_llm")
    @patch("agents.daily_review.parse_json_response")
    def test_list_fields_validated(self, mock_parse, mock_llm):
        """Non-list values for list fields are replaced with defaults."""
        mock_llm.return_value = "ignored"
        mock_parse.return_value = {
            "executive_summary": "ok",
            "what_worked": "not a list",
            "what_failed": 42,
            "tomorrows_focus": None,
            "driver_ranking": "bad",
        }

        result = generate_narrative(_make_summary())
        assert result["what_worked"] == []
        assert result["what_failed"] == []
        assert result["tomorrows_focus"] == []
        assert result["driver_ranking"] == []

    @patch("agents.daily_review.call_llm")
    @patch("agents.daily_review.parse_json_response")
    def test_string_fields_validated(self, mock_parse, mock_llm):
        """Non-string values for string fields are coerced to strings."""
        mock_llm.return_value = "ignored"
        mock_parse.return_value = {
            "executive_summary": 42,
            "primary_driver": ["not", "a", "string"],
        }

        result = generate_narrative(_make_summary())
        assert isinstance(result["executive_summary"], str)
        assert isinstance(result["primary_driver"], str)
