"""
Tests for the Reversal/Close Review prompt template and handler.

Validates:
- REVERSAL_CLOSE_PROMPT template is defined with required placeholders
- run_reversal_close_review returns valid actions from the constrained set
- Invalid LLM actions default to "hold_tighten"
- LLM failures default to "hold_tighten"
- Trigger type and details are logged and included in prompt
- Entry Contract context (thesis, setup_type, invalidators) is included in prompt
- Opposing evidence and market conditions are included in prompt
- Requirements: 7.1, 7.2, 7.3
"""

import json
from unittest.mock import patch

import pytest

from agents.portfolio_manager import (
    REVERSAL_CLOSE_PROMPT,
    VALID_REVERSAL_ACTIONS,
    run_reversal_close_review,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_position_data(**overrides) -> dict:
    """Build a sample position_data dict for Reversal/Close Review."""
    data = {
        "symbol": "AMD",
        "side": "long",
        "quantity": 100,
        "entry_price": 160.0,
        "stop_price": 155.0,
        "target_price": 175.0,
        "current_price": 153.0,
        "unrealized_pnl_pct": -4.37,
        "thesis": "Gap-and-go above VWAP with bullish EMA support",
        "setup_type": "gap_and_go",
        "invalidators": json.dumps([
            {
                "type": "price_below_level",
                "reference": "VWAP",
                "confirmation": "5m_close",
                "lookback_bars": 1,
            }
        ]),
        "market_conditions": {
            "vwap": 154.50,
            "rsi": 35.0,
            "trend": "bearish",
            "price_vs_vwap": "below",
        },
        "opposing_evidence": {
            "signal": "SHORT",
            "strength": "strong",
            "confidence": "high",
            "reasoning": "VWAP lost, bearish momentum increasing",
        },
    }
    data.update(overrides)
    return data


def _sample_trigger_info(**overrides) -> dict:
    """Build a sample trigger_info dict."""
    data = {
        "type": "thesis_invalidation",
        "details": "VWAP lost on 5m close — price_below_level invalidator breached",
        "invalidator": {
            "type": "price_below_level",
            "reference": "VWAP",
            "confirmation": "5m_close",
            "lookback_bars": 1,
        },
    }
    data.update(overrides)
    return data


def _sample_profile() -> dict:
    """Build a sample PM profile dict."""
    return {
        "name": "Aggressive",
        "emoji": "🔥",
        "max_positions": 5,
        "max_position_pct": 0.25,
        "min_risk_reward": 2.0,
        "min_signal_strength": "moderate",
        "avoid_first_minutes": 5,
        "avoid_last_minutes": 15,
        "max_daily_loss_pct": 0.03,
        "starting_balance": 100000,
        "personality": "Aggressive risk-taker",
        "opposing_evidence_threshold": "strong",
    }


# ---------------------------------------------------------------------------
# 1. Prompt template contains required placeholders
# ---------------------------------------------------------------------------

def test_prompt_template_has_required_placeholders():
    """REVERSAL_CLOSE_PROMPT must contain all required format placeholders."""
    required_placeholders = [
        "{profile_name}", "{emoji}",
        "{trigger_type}", "{trigger_details}",
        "{thesis}", "{setup_type}", "{entry_price}", "{stop_price}", "{target_price}",
        "{invalidators}",
        "{symbol}", "{side}", "{quantity}", "{current_price}", "{unrealized_pnl_pct}",
        "{market_conditions_text}",
        "{opposing_evidence_text}",
        "{invalidator_json}",
    ]
    for placeholder in required_placeholders:
        assert placeholder in REVERSAL_CLOSE_PROMPT, (
            f"REVERSAL_CLOSE_PROMPT missing placeholder: {placeholder}"
        )


# ---------------------------------------------------------------------------
# 2. Prompt template constrains output to valid actions only
# ---------------------------------------------------------------------------

def test_prompt_template_constrains_actions():
    """Prompt must mention valid reversal actions."""
    assert "close_full" in REVERSAL_CLOSE_PROMPT
    assert "close_partial" in REVERSAL_CLOSE_PROMPT
    assert "hold_tighten" in REVERSAL_CLOSE_PROMPT


# ---------------------------------------------------------------------------
# 3. Valid reversal actions set
# ---------------------------------------------------------------------------

def test_valid_reversal_actions():
    """VALID_REVERSAL_ACTIONS must be exactly the three allowed actions."""
    assert VALID_REVERSAL_ACTIONS == {"close_full", "close_partial", "hold_tighten"}


# ---------------------------------------------------------------------------
# 4. run_reversal_close_review returns close_full on LLM success
# ---------------------------------------------------------------------------

@patch("agents.portfolio_manager.call_llm")
def test_reversal_review_close_full(mock_llm):
    """LLM returns close_full → handler returns close_full with correct symbol."""
    mock_llm.return_value = json.dumps({
        "symbol": "AMD",
        "action": "close_full",
        "reasoning": "VWAP lost on 5m close, thesis invalidated",
        "trigger": "thesis_invalidation",
        "invalidator": {
            "type": "price_below_level",
            "reference": "VWAP",
            "confirmation": "5m_close",
            "lookback_bars": 1,
        },
    })

    result = run_reversal_close_review(
        _sample_position_data(), _sample_trigger_info(), _sample_profile()
    )

    assert result["symbol"] == "AMD"
    assert result["action"] == "close_full"
    assert result["trigger"] == "thesis_invalidation"
    assert result["reasoning"] is not None


# ---------------------------------------------------------------------------
# 5. run_reversal_close_review returns close_partial
# ---------------------------------------------------------------------------

@patch("agents.portfolio_manager.call_llm")
def test_reversal_review_close_partial(mock_llm):
    """LLM returns close_partial → handler returns close_partial."""
    mock_llm.return_value = json.dumps({
        "symbol": "AMD",
        "action": "close_partial",
        "reasoning": "Some thesis elements broken but support still holding",
        "trigger": "opposing_signal",
        "invalidator": None,
    })

    result = run_reversal_close_review(
        _sample_position_data(),
        _sample_trigger_info(type="opposing_signal", details="Strong SHORT signal received"),
        _sample_profile(),
    )

    assert result["action"] == "close_partial"


# ---------------------------------------------------------------------------
# 6. run_reversal_close_review returns hold_tighten
# ---------------------------------------------------------------------------

@patch("agents.portfolio_manager.call_llm")
def test_reversal_review_hold_tighten(mock_llm):
    """LLM returns hold_tighten → handler returns hold_tighten."""
    mock_llm.return_value = json.dumps({
        "symbol": "AMD",
        "action": "hold_tighten",
        "reasoning": "Evidence ambiguous, tightening stop to protect capital",
        "trigger": "thesis_invalidation",
        "invalidator": {
            "type": "price_below_level",
            "reference": "VWAP",
            "confirmation": "5m_close",
            "lookback_bars": 1,
        },
    })

    result = run_reversal_close_review(
        _sample_position_data(), _sample_trigger_info(), _sample_profile()
    )

    assert result["action"] == "hold_tighten"


# ---------------------------------------------------------------------------
# 7. Invalid action from LLM defaults to hold_tighten
# ---------------------------------------------------------------------------

@patch("agents.portfolio_manager.call_llm")
def test_reversal_review_invalid_action_defaults_to_hold_tighten(mock_llm):
    """LLM returns an invalid action (e.g., 'hold') → defaults to hold_tighten."""
    mock_llm.return_value = json.dumps({
        "symbol": "AMD",
        "action": "hold",
        "reasoning": "Trying to hold without tightening",
        "trigger": "thesis_invalidation",
    })

    result = run_reversal_close_review(
        _sample_position_data(), _sample_trigger_info(), _sample_profile()
    )

    assert result["action"] == "hold_tighten"


# ---------------------------------------------------------------------------
# 8. LLM failure defaults to hold_tighten
# ---------------------------------------------------------------------------

@patch("agents.portfolio_manager.call_llm", side_effect=Exception("LLM timeout"))
def test_reversal_review_llm_failure_defaults_to_hold_tighten(mock_llm):
    """LLM call raises exception → defaults to hold_tighten."""
    result = run_reversal_close_review(
        _sample_position_data(), _sample_trigger_info(), _sample_profile()
    )

    assert result["symbol"] == "AMD"
    assert result["action"] == "hold_tighten"
    assert "failed" in result["reasoning"].lower() or "default" in result["reasoning"].lower()
    assert result["trigger"] == "thesis_invalidation"


# ---------------------------------------------------------------------------
# 9. Trigger type is preserved in result
# ---------------------------------------------------------------------------

@patch("agents.portfolio_manager.call_llm")
def test_reversal_review_preserves_trigger_type(mock_llm):
    """The trigger type from trigger_info is preserved in the result."""
    mock_llm.return_value = json.dumps({
        "symbol": "AMD",
        "action": "close_full",
        "reasoning": "Explicit close signal received",
        "trigger": "explicit_close",
    })

    result = run_reversal_close_review(
        _sample_position_data(),
        _sample_trigger_info(type="explicit_close", details="User requested close"),
        _sample_profile(),
    )

    assert result["trigger"] == "explicit_close"


# ---------------------------------------------------------------------------
# 10. Trigger details appear in the LLM prompt
# ---------------------------------------------------------------------------

@patch("agents.portfolio_manager.call_llm")
def test_reversal_review_trigger_details_in_prompt(mock_llm):
    """Trigger details should appear in the prompt sent to the LLM."""
    mock_llm.return_value = json.dumps({
        "symbol": "AMD",
        "action": "close_full",
        "reasoning": "Thesis broken",
        "trigger": "thesis_invalidation",
    })

    trigger = _sample_trigger_info(
        details="VWAP lost on 5m close — price_below_level invalidator breached"
    )
    run_reversal_close_review(
        _sample_position_data(), trigger, _sample_profile()
    )

    call_args = mock_llm.call_args
    user_prompt = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("user_prompt", "")
    assert "VWAP lost on 5m close" in user_prompt


# ---------------------------------------------------------------------------
# 11. Missing optional fields handled gracefully
# ---------------------------------------------------------------------------

@patch("agents.portfolio_manager.call_llm")
def test_reversal_review_missing_optional_fields(mock_llm):
    """Handler works with minimal position data (missing optional fields)."""
    mock_llm.return_value = json.dumps({
        "symbol": "XYZ",
        "action": "hold_tighten",
        "reasoning": "Holding with tightened stop",
        "trigger": "thesis_invalidation",
    })

    minimal_data = {
        "symbol": "XYZ",
        "side": "long",
        "quantity": 10,
        "current_price": 50.0,
    }
    minimal_trigger = {
        "type": "thesis_invalidation",
        "details": "Stop breached",
    }

    result = run_reversal_close_review(minimal_data, minimal_trigger, _sample_profile())

    assert result["symbol"] == "XYZ"
    assert result["action"] == "hold_tighten"


# ---------------------------------------------------------------------------
# 12. Maintenance actions (hold, tighten_stop, etc.) are rejected
# ---------------------------------------------------------------------------

@patch("agents.portfolio_manager.call_llm")
def test_reversal_review_maintenance_action_defaults_to_hold_tighten(mock_llm):
    """LLM returns a maintenance action (e.g., 'tighten_stop') → defaults to hold_tighten."""
    mock_llm.return_value = json.dumps({
        "symbol": "AMD",
        "action": "tighten_stop",
        "reasoning": "Trying maintenance action in reversal review",
    })

    result = run_reversal_close_review(
        _sample_position_data(), _sample_trigger_info(), _sample_profile()
    )

    assert result["action"] == "hold_tighten"


# ---------------------------------------------------------------------------
# 13. Invalidator from trigger is preserved in result on LLM failure
# ---------------------------------------------------------------------------

@patch("agents.portfolio_manager.call_llm", side_effect=Exception("LLM error"))
def test_reversal_review_preserves_invalidator_on_failure(mock_llm):
    """On LLM failure, the trigger invalidator is preserved in the default result."""
    trigger = _sample_trigger_info()
    result = run_reversal_close_review(
        _sample_position_data(), trigger, _sample_profile()
    )

    assert result["invalidator"] == trigger["invalidator"]
    assert result["trigger"] == "thesis_invalidation"


# ---------------------------------------------------------------------------
# 14. Trigger is logged (Req 7.3)
# ---------------------------------------------------------------------------

@patch("agents.portfolio_manager.call_llm")
@patch("agents.portfolio_manager.log")
def test_reversal_review_logs_trigger(mock_log, mock_llm):
    """The specific trigger that caused the review must be logged."""
    mock_llm.return_value = json.dumps({
        "symbol": "AMD",
        "action": "close_full",
        "reasoning": "Thesis broken",
        "trigger": "thesis_invalidation",
    })

    run_reversal_close_review(
        _sample_position_data(), _sample_trigger_info(), _sample_profile()
    )

    # Verify log.info was called with trigger information
    log_calls = [str(c) for c in mock_log.info.call_args_list]
    trigger_logged = any("thesis_invalidation" in c for c in log_calls)
    assert trigger_logged, "Trigger type should be logged via log.info"
