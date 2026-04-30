"""
Tests for narrator system prompts and user prompt builders (Task 2.2).
Validates: Requirements 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9
"""

import json
from datetime import date
from unittest.mock import patch

from agents.narrator import (
    NARRATOR_SYSTEM_PROMPT,
    FLASH_SYSTEM_PROMPT,
    build_system_prompt,
    build_user_prompt,
    _build_morning_prompt,
    _build_hourly_prompt,
    _build_afternoon_prompt,
    _build_daily_wrap_prompt,
    _build_weekly_wrap_prompt,
    _build_sunday_prep_prompt,
    _build_flash_prompt,
    UPDATE_TYPES,
)


# --- System prompt constants ---

def test_narrator_system_prompt_is_nonempty():
    assert len(NARRATOR_SYSTEM_PROMPT.strip()) > 100


def test_flash_system_prompt_is_nonempty():
    assert len(FLASH_SYSTEM_PROMPT.strip()) > 50


def test_narrator_system_prompt_enforces_prose_rules():
    """Req 1.3, 1.4, 1.5, 1.8, 1.9: system prompt mentions key rules."""
    prompt = NARRATOR_SYSTEM_PROMPT.lower()
    assert "3 to 8 sentences" in prompt or "3-8 sentences" in prompt
    assert "no headers" in prompt
    assert "no bullet points" in prompt
    assert "no markdown" in prompt
    assert "forward-looking" in prompt
    assert "p&l" in prompt or "pnl" in prompt


def test_flash_system_prompt_enforces_brevity():
    """Req 1.8: flash updates are 2-4 sentences."""
    prompt = FLASH_SYSTEM_PROMPT.lower()
    assert "2 to 4 sentences" in prompt or "2-4 sentences" in prompt


def test_narrator_system_prompt_mentions_divergence():
    """Req 1.6: system prompt instructs to note PM divergences."""
    assert "diverge" in NARRATOR_SYSTEM_PROMPT.lower()


def test_narrator_system_prompt_mentions_unusual_events():
    """Req 1.7: system prompt instructs to flag unusual events."""
    prompt = NARRATOR_SYSTEM_PROMPT.lower()
    assert "drawdown" in prompt or "unusual" in prompt
    assert "2%" in NARRATOR_SYSTEM_PROMPT


# --- build_system_prompt ---

def test_build_system_prompt_returns_narrator_for_scheduled():
    for ut in UPDATE_TYPES:
        if ut != "flash_update":
            assert build_system_prompt(ut) == NARRATOR_SYSTEM_PROMPT


def test_build_system_prompt_returns_flash_for_flash():
    assert build_system_prompt("flash_update") == FLASH_SYSTEM_PROMPT


# --- build_user_prompt ---

def test_build_user_prompt_data_gap():
    ctx = {"data_gap": True, "error": "DB timeout"}
    prompt = build_user_prompt("morning_briefing", ctx)
    assert "DB timeout" in prompt
    assert "limited data" in prompt


def test_build_user_prompt_dispatches_to_all_types():
    """Every update type should produce a non-empty prompt."""
    ctx = {"regime": "risk_on", "trigger": "atr_spike", "symbol": "TSLA"}
    for ut in UPDATE_TYPES:
        prompt = build_user_prompt(ut, ctx)
        assert isinstance(prompt, str)
        assert len(prompt) > 20


# --- Individual prompt builders ---

def test_morning_prompt_includes_key_fields():
    ctx = {
        "regime": "risk_on",
        "primary_strategy": "gap_and_go",
        "strategies_to_avoid": ["momentum_fade"],
        "analyst_signals": {"TSLA": {"signal": "LONG"}},
        "positions": [{"symbol": "TSLA", "side": "long"}],
        "portfolio_summary": {"moderate": {"equity": 100000}},
        "story_arc": {},
        "confidence_regime": {"overall": "high conviction"},
    }
    prompt = _build_morning_prompt(ctx, "2026-04-24")
    assert "2026-04-24" in prompt
    assert "risk_on" in prompt
    assert "gap_and_go" in prompt
    assert "TSLA" in prompt
    assert "LONG" in prompt
    assert "morning briefing" in prompt.lower()


def test_hourly_prompt_includes_key_fields():
    ctx = {
        "hour_label": "10:00 AM",
        "recent_trades": [{"symbol": "AMD", "action": "BUY"}],
        "position_pnl_changes": [],
        "signal_changes": {},
        "quiet_period": False,
        "story_arc": {},
        "confidence_regime": {},
    }
    prompt = _build_hourly_prompt(ctx, "2026-04-24")
    assert "10:00 AM" in prompt
    assert "AMD" in prompt
    assert "hourly recap" in prompt.lower()


def test_afternoon_prompt_includes_aggregate_pnl():
    ctx = {
        "recent_trades": [],
        "aggregate_pnl": {"moderate": 250.0, "aggressive": -100.0},
        "win_loss": {"moderate": {"wins": 2, "losses": 1}},
        "equity_change": {"moderate": 250.0},
        "quiet_period": True,
        "story_arc": {},
        "confidence_regime": {},
    }
    prompt = _build_afternoon_prompt(ctx, "2026-04-24")
    assert "afternoon recap" in prompt.lower()
    assert "2:00 PM" in prompt
    assert "250.0" in prompt
    assert "MIDDAY AGGREGATE" in prompt


def test_daily_wrap_prompt_includes_key_fields():
    ctx = {
        "closed_trades": [{"symbol": "TSLA", "pnl": 500}],
        "open_positions": [],
        "profile_summary": {"moderate": {"equity": 100500, "daily_pnl": 500}},
        "win_loss": {"moderate": {"wins": 3, "losses": 1}},
        "reviewer_scores": {"TSLA": 8.5},
        "lessons": ["Cut losers faster"],
        "story_arc": {},
        "confidence_regime": {},
    }
    prompt = _build_daily_wrap_prompt(ctx, "2026-04-24")
    assert "daily wrap" in prompt.lower()
    assert "TSLA" in prompt
    assert "500" in prompt
    assert "REVIEWER SCORES" in prompt
    assert "LESSONS" in prompt


def test_weekly_wrap_prompt_includes_key_fields():
    ctx = {
        "week_pnl": {"moderate": 1200},
        "best_trades": [{"symbol": "NVDA", "pnl": 800}],
        "worst_trades": [{"symbol": "AMD", "pnl": -300}],
        "strategy_performance": {"gap_and_go": {"win_rate": 0.65}},
        "agent_grades": {"analyst": "A", "researcher": "B+"},
        "story_arc": {},
        "confidence_regime": {},
    }
    prompt = _build_weekly_wrap_prompt(ctx, "2026-04-25")
    assert "weekly wrap" in prompt.lower()
    assert "NVDA" in prompt
    assert "AMD" in prompt
    assert "gap_and_go" in prompt
    assert "AGENT GRADES" in prompt


def test_sunday_prep_prompt_includes_key_fields():
    ctx = {
        "weekly_briefing": {"summary": "Bullish bias for tech"},
        "watchlist": ["TSLA", "NVDA"],
        "strategy_recommendation": {"primary": "momentum"},
        "pm_stances": {"conservative": "defensive", "aggressive": "risk-on"},
        "story_arc": {},
        "confidence_regime": {},
    }
    prompt = _build_sunday_prep_prompt(ctx, "2026-04-27")
    assert "sunday prep" in prompt.lower()
    assert "TSLA" in prompt
    assert "momentum" in prompt
    assert "PM STANCES" in prompt


def test_flash_prompt_includes_event_fields():
    ctx = {
        "trigger": "atr_spike",
        "symbol": "AMD",
        "details": "AMD moved 2.3x ATR in 4 minutes",
        "price": 165.40,
        "profile": "aggressive",
        "pnl_impact": -450.0,
        "position_data": {"symbol": "AMD", "side": "long", "quantity": 100},
        "analyst_signal": {"signal": "LONG"},
        "catalyst_freshness": {"freshness_state": "fresh"},
        "story_arc": {},
    }
    prompt = _build_flash_prompt(ctx, "2026-04-24")
    assert "Flash update" in prompt
    assert "atr_spike" in prompt
    assert "AMD" in prompt
    assert "165.4" in prompt
    assert "aggressive" in prompt
    assert "-450.0" in prompt


def test_build_user_prompt_unknown_type_returns_fallback():
    """Unknown update types should get a generic fallback prompt."""
    ctx = {}
    prompt = build_user_prompt("unknown_type", ctx)
    assert "unknown_type" in prompt
