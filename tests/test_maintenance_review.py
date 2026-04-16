"""
Tests for the Maintenance Review prompt template and handler.

Validates:
- MAINTENANCE_REVIEW_PROMPT template is defined with required placeholders
- run_maintenance_review returns valid actions from the constrained set
- Invalid LLM actions default to "hold"
- LLM failures default to "hold"
- Entry Contract context (thesis, setup_type, invalidators) is included in prompt
- DRIFTING state label is included in prompt
- Position health assessments are included in prompt
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from agents.portfolio_manager import (
    MAINTENANCE_REVIEW_PROMPT,
    VALID_MAINTENANCE_ACTIONS,
    run_maintenance_review,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_position_data(**overrides) -> dict:
    """Build a sample position_data dict for Maintenance Review."""
    data = {
        "symbol": "AMD",
        "side": "long",
        "quantity": 100,
        "entry_price": 160.0,
        "stop_price": 155.0,
        "target_price": 175.0,
        "current_price": 165.0,
        "unrealized_pnl_pct": 3.12,
        "drifting": False,
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
        "indicators": {
            "rsi": 58.0,
            "trend": "bullish",
            "macd_cross": "bullish",
            "price_vs_vwap": "above",
        },
        "advisory_signals": {
            "signal": "LONG",
            "strength": "strong",
            "confidence": "high",
        },
        "health_text": "AMD (aggressive): healthy — thesis intact, price above VWAP",
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
    """MAINTENANCE_REVIEW_PROMPT must contain all required format placeholders."""
    required_placeholders = [
        "{profile_name}", "{emoji}",
        "{thesis}", "{setup_type}", "{entry_price}", "{stop_price}", "{target_price}",
        "{invalidators}",
        "{symbol}", "{side}", "{quantity}", "{current_price}", "{unrealized_pnl_pct}",
        "{drifting}",
        "{indicators_text}",
        "{advisory_signals_text}",
        "{health_text}",
    ]
    for placeholder in required_placeholders:
        assert placeholder in MAINTENANCE_REVIEW_PROMPT, (
            f"MAINTENANCE_REVIEW_PROMPT missing placeholder: {placeholder}"
        )


# ---------------------------------------------------------------------------
# 2. Prompt template constrains output to valid actions only
# ---------------------------------------------------------------------------

def test_prompt_template_constrains_actions():
    """Prompt must mention valid actions and explicitly forbid CLOSE."""
    assert "hold" in MAINTENANCE_REVIEW_PROMPT
    assert "tighten_stop" in MAINTENANCE_REVIEW_PROMPT
    assert "raise_target" in MAINTENANCE_REVIEW_PROMPT
    assert "trim_partial" in MAINTENANCE_REVIEW_PROMPT
    # Must explicitly forbid close
    assert "CLOSE" in MAINTENANCE_REVIEW_PROMPT or "close" in MAINTENANCE_REVIEW_PROMPT


# ---------------------------------------------------------------------------
# 3. Valid maintenance actions set
# ---------------------------------------------------------------------------

def test_valid_maintenance_actions():
    """VALID_MAINTENANCE_ACTIONS must be exactly the four allowed actions."""
    assert VALID_MAINTENANCE_ACTIONS == {"hold", "tighten_stop", "raise_target", "trim_partial"}


# ---------------------------------------------------------------------------
# 4. run_maintenance_review returns hold on LLM success
# ---------------------------------------------------------------------------

@patch("agents.portfolio_manager.call_llm")
def test_maintenance_review_hold(mock_llm):
    """LLM returns hold → handler returns hold with correct symbol."""
    mock_llm.return_value = json.dumps({
        "reviews": [{
            "symbol": "AMD",
            "action": "hold",
            "new_stop": None,
            "new_target": None,
            "trim_pct": None,
            "reasoning": "Thesis intact, price above VWAP, holding to target",
        }],
        "notes": "All positions healthy",
    })

    result = run_maintenance_review(_sample_position_data(), _sample_profile())

    assert result["symbol"] == "AMD"
    assert result["action"] == "hold"
    assert result["reasoning"] is not None


# ---------------------------------------------------------------------------
# 5. run_maintenance_review returns tighten_stop
# ---------------------------------------------------------------------------

@patch("agents.portfolio_manager.call_llm")
def test_maintenance_review_tighten_stop(mock_llm):
    """LLM returns tighten_stop → handler returns tighten_stop with new_stop."""
    mock_llm.return_value = json.dumps({
        "reviews": [{
            "symbol": "AMD",
            "action": "tighten_stop",
            "new_stop": 162.0,
            "new_target": None,
            "trim_pct": None,
            "reasoning": "Price moved favorably, locking in gains",
        }],
        "notes": "Tightening stop on AMD",
    })

    result = run_maintenance_review(_sample_position_data(), _sample_profile())

    assert result["action"] == "tighten_stop"
    assert result["new_stop"] == 162.0


# ---------------------------------------------------------------------------
# 6. run_maintenance_review returns raise_target
# ---------------------------------------------------------------------------

@patch("agents.portfolio_manager.call_llm")
def test_maintenance_review_raise_target(mock_llm):
    """LLM returns raise_target → handler returns raise_target with new_target."""
    mock_llm.return_value = json.dumps({
        "reviews": [{
            "symbol": "AMD",
            "action": "raise_target",
            "new_stop": None,
            "new_target": 180.0,
            "trim_pct": None,
            "reasoning": "Strong momentum supports higher target",
        }],
        "notes": "Raising target on AMD",
    })

    result = run_maintenance_review(_sample_position_data(), _sample_profile())

    assert result["action"] == "raise_target"
    assert result["new_target"] == 180.0


# ---------------------------------------------------------------------------
# 7. run_maintenance_review returns trim_partial
# ---------------------------------------------------------------------------

@patch("agents.portfolio_manager.call_llm")
def test_maintenance_review_trim_partial(mock_llm):
    """LLM returns trim_partial → handler returns trim_partial with trim_pct."""
    mock_llm.return_value = json.dumps({
        "reviews": [{
            "symbol": "AMD",
            "action": "trim_partial",
            "new_stop": None,
            "new_target": None,
            "trim_pct": 25,
            "reasoning": "Significantly profitable, reducing risk",
        }],
        "notes": "Trimming AMD position",
    })

    result = run_maintenance_review(_sample_position_data(), _sample_profile())

    assert result["action"] == "trim_partial"
    assert result["trim_pct"] == 25


# ---------------------------------------------------------------------------
# 8. Invalid action from LLM defaults to hold
# ---------------------------------------------------------------------------

@patch("agents.portfolio_manager.call_llm")
def test_maintenance_review_invalid_action_defaults_to_hold(mock_llm):
    """LLM returns an invalid action (e.g., 'close') → defaults to hold."""
    mock_llm.return_value = json.dumps({
        "reviews": [{
            "symbol": "AMD",
            "action": "close",
            "new_stop": None,
            "new_target": None,
            "trim_pct": None,
            "reasoning": "Trying to close",
        }],
        "notes": "Attempted close",
    })

    result = run_maintenance_review(_sample_position_data(), _sample_profile())

    assert result["action"] == "hold"


# ---------------------------------------------------------------------------
# 9. LLM failure defaults to hold
# ---------------------------------------------------------------------------

@patch("agents.portfolio_manager.call_llm", side_effect=Exception("LLM timeout"))
def test_maintenance_review_llm_failure_defaults_to_hold(mock_llm):
    """LLM call raises exception → defaults to hold."""
    result = run_maintenance_review(_sample_position_data(), _sample_profile())

    assert result["symbol"] == "AMD"
    assert result["action"] == "hold"
    assert "failed" in result["reasoning"].lower() or "default" in result["reasoning"].lower()


# ---------------------------------------------------------------------------
# 10. Drifting position data is formatted correctly
# ---------------------------------------------------------------------------

@patch("agents.portfolio_manager.call_llm")
def test_maintenance_review_drifting_position(mock_llm):
    """Drifting position should have DRIFTING=YES in the prompt."""
    mock_llm.return_value = json.dumps({
        "reviews": [{
            "symbol": "AMD",
            "action": "hold",
            "new_stop": None,
            "new_target": None,
            "trim_pct": None,
            "reasoning": "Drifting but thesis intact",
        }],
        "notes": "Drifting position held",
    })

    result = run_maintenance_review(
        _sample_position_data(drifting=True),
        _sample_profile(),
    )

    assert result["action"] == "hold"
    # Verify the LLM was called with a prompt containing "YES" for drifting
    call_args = mock_llm.call_args
    user_prompt = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("user_prompt", "")
    assert "YES" in user_prompt


# ---------------------------------------------------------------------------
# 11. Missing optional fields handled gracefully
# ---------------------------------------------------------------------------

@patch("agents.portfolio_manager.call_llm")
def test_maintenance_review_missing_optional_fields(mock_llm):
    """Handler works with minimal position data (missing optional fields)."""
    mock_llm.return_value = json.dumps({
        "reviews": [{
            "symbol": "XYZ",
            "action": "hold",
            "new_stop": None,
            "new_target": None,
            "trim_pct": None,
            "reasoning": "Holding",
        }],
        "notes": "ok",
    })

    minimal_data = {
        "symbol": "XYZ",
        "side": "long",
        "quantity": 10,
        "current_price": 50.0,
    }

    result = run_maintenance_review(minimal_data, _sample_profile())

    assert result["symbol"] == "XYZ"
    assert result["action"] == "hold"


# ---------------------------------------------------------------------------
# 12. close_full and close_partial are also rejected
# ---------------------------------------------------------------------------

@patch("agents.portfolio_manager.call_llm")
def test_maintenance_review_close_full_defaults_to_hold(mock_llm):
    """LLM returns close_full → defaults to hold."""
    mock_llm.return_value = json.dumps({
        "reviews": [{
            "symbol": "AMD",
            "action": "close_full",
            "reasoning": "Trying to close full",
        }],
        "notes": "",
    })

    result = run_maintenance_review(_sample_position_data(), _sample_profile())
    assert result["action"] == "hold"


@patch("agents.portfolio_manager.call_llm")
def test_maintenance_review_close_partial_defaults_to_hold(mock_llm):
    """LLM returns close_partial → defaults to hold."""
    mock_llm.return_value = json.dumps({
        "reviews": [{
            "symbol": "AMD",
            "action": "close_partial",
            "reasoning": "Trying to close partial",
        }],
        "notes": "",
    })

    result = run_maintenance_review(_sample_position_data(), _sample_profile())
    assert result["action"] == "hold"
