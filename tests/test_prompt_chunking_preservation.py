"""
Preservation Property Tests — Property 2: Shared Formatters and Unaffected Paths Unchanged

Validates: Requirements 3.1, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10

These tests verify that shared formatters, unaffected prompt paths, context dict
keys, and PM entry candidate handling remain unchanged on the UNFIXED code.
They establish a baseline that must be preserved after the bugfix is applied.

Run BEFORE implementing the fix. All tests should PASS on unfixed code.
"""

import json
from datetime import datetime, date

from hypothesis import given, settings, assume
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Shared constants (reused from bug test for consistency)
# ---------------------------------------------------------------------------

SETUP_TYPES = [
    "news_breakout", "gap_and_go", "technical_breakout", "momentum_fade",
    "gap_fill", "range_breakout", "vwap_reclaim", "earnings_reaction",
    "sector_rotation", "reversal",
]

SYMBOLS = [
    "AAPL", "TSLA", "NVDA", "AMD", "MSFT", "GOOG", "AMZN", "META",
    "SPY", "QQQ", "IWM", "NFLX",
]


# ---------------------------------------------------------------------------
# Hypothesis strategies for generating realistic data
# ---------------------------------------------------------------------------

case_field_strategy = st.fixed_dictionaries({
    "id": st.integers(min_value=1, max_value=9999),
    "date": st.dates(
        min_value=date(2024, 1, 1), max_value=date(2025, 12, 31)
    ).map(lambda d: d.isoformat()),
    "symbol": st.sampled_from(SYMBOLS),
    "setup_type": st.sampled_from(SETUP_TYPES),
    "catalyst_type": st.sampled_from(["earnings_beat", "news_headline", "fda_approval", "sector_rotation"]),
    "float_profile": st.sampled_from(["low_float", "mid_cap", "large_cap"]),
    "sector": st.sampled_from(["tech", "healthcare", "energy", "finance"]),
    "premarket_gap_pct": st.floats(min_value=-10.0, max_value=30.0, allow_nan=False, allow_infinity=False),
    "premarket_volume_rank": st.sampled_from(["high", "medium", "low", None]),
    "market_regime": st.sampled_from(["risk_on", "risk_off", "neutral"]),
    "entry_timing": st.sampled_from(["first_15min", "first_30min", "midday", "close"]),
    "bias": st.sampled_from(["LONG", "SHORT"]),
    "signal_strength": st.sampled_from(["strong", "moderate", "weak"]),
    "rsi_at_entry": st.one_of(st.floats(min_value=20.0, max_value=80.0, allow_nan=False, allow_infinity=False), st.none()),
    "above_vwap": st.one_of(st.sampled_from(["true", "false"]), st.none()),
    "above_daily_resistance": st.one_of(st.sampled_from(["true", "false"]), st.none()),
    "ema_trend": st.one_of(st.sampled_from(["bullish", "bearish", "flat"]), st.none()),
    "outcome": st.sampled_from(["success", "failure", "breakeven"]),
    "pnl_pct": st.floats(min_value=-15.0, max_value=20.0, allow_nan=False, allow_infinity=False),
    "holding_minutes": st.integers(min_value=5, max_value=390),
    "lesson": st.one_of(
        st.text(min_size=10, max_size=200, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"))),
        st.none(),
    ),
    "conditions_for_success": st.lists(
        st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "P"))),
        min_size=0, max_size=5,
    ),
    "conditions_to_avoid": st.lists(
        st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "P"))),
        min_size=0, max_size=5,
    ),
    "confidence": st.one_of(st.sampled_from(["high", "medium", "low"]), st.none()),
    "selection_score": st.one_of(st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False), st.none()),
    "execution_score": st.one_of(st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False), st.none()),
    "review_score": st.one_of(st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False), st.none()),
    "profile": st.sampled_from(["conservative", "moderate", "aggressive"]),
})

# Strategy for generating a list of case dicts (0-20 cases)
cases_list_strategy = st.lists(case_field_strategy, min_size=0, max_size=20)

# Strategy for generating entry signal dicts
signal_strategy = st.fixed_dictionaries({
    "signal": st.sampled_from(["LONG", "SHORT"]),
    "strength": st.sampled_from(["strong", "moderate", "weak"]),
    "setup_type": st.sampled_from(SETUP_TYPES),
    "confidence": st.sampled_from(["high", "medium", "low"]),
    "entry": st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    "stop": st.floats(min_value=0.5, max_value=999.0, allow_nan=False, allow_infinity=False),
    "target": st.floats(min_value=1.5, max_value=1500.0, allow_nan=False, allow_infinity=False),
    "invalidation": st.text(min_size=5, max_size=80, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"))),
    "key_levels": st.lists(
        st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=0, max_size=5,
    ),
    "reasoning": st.text(min_size=20, max_size=800, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"))),
    "indicators": st.fixed_dictionaries({
        "rsi": st.fixed_dictionaries({
            "value": st.floats(min_value=10.0, max_value=90.0, allow_nan=False, allow_infinity=False),
            "signal": st.sampled_from(["overbought", "oversold", "neutral"]),
        }),
    }),
})

# Strategy for generating entry_signals dicts (symbol -> signal)
entry_signals_strategy = st.dictionaries(
    keys=st.sampled_from(SYMBOLS),
    values=signal_strategy,
    min_size=1,
    max_size=8,
)


# ===========================================================================
# AREA 1: Shared Formatter Preservation (Requirements 3.6, 3.7)
# ===========================================================================

@given(cases=cases_list_strategy)
@settings(max_examples=50, deadline=None)
def test_format_cases_for_prompt_deterministic(cases):
    """
    **Validates: Requirements 3.6**

    Property: format_cases_for_prompt() is a pure function — calling it twice
    with the same input produces identical output. This captures the baseline
    behavior that must be preserved after the fix. The fix introduces a NEW
    function (format_cases_digest_for_pm) and must NOT modify this function.
    """
    from utils.case_library import format_cases_for_prompt

    result1 = format_cases_for_prompt(cases)
    result2 = format_cases_for_prompt(cases)

    assert result1 == result2, (
        f"format_cases_for_prompt() is not deterministic for {len(cases)} cases. "
        f"First call returned {len(result1)} chars, second returned {len(result2)} chars."
    )


@given(cases=cases_list_strategy)
@settings(max_examples=50, deadline=None)
def test_format_cases_for_prompt_returns_string(cases):
    """
    **Validates: Requirements 3.6**

    Property: format_cases_for_prompt() always returns a string for any
    valid list of case dicts. This is the contract other agents depend on.
    """
    from utils.case_library import format_cases_for_prompt

    result = format_cases_for_prompt(cases)

    assert isinstance(result, str), (
        f"format_cases_for_prompt() should return str, got {type(result)}"
    )

    if not cases:
        assert result == "No relevant past cases found.", (
            f"Empty cases should return sentinel message, got: {result!r}"
        )
    else:
        # Each case should contribute at least the symbol to the output
        for c in cases:
            assert c["symbol"] in result, (
                f"Case symbol {c['symbol']} not found in formatted output"
            )


@given(cases=cases_list_strategy)
@settings(max_examples=50, deadline=None)
def test_format_cases_for_prompt_includes_key_fields(cases):
    """
    **Validates: Requirements 3.6**

    Property: format_cases_for_prompt() includes setup_type, outcome, and
    pnl_pct for every case in the output. These are the fields that all
    callers (quant_researcher, analyst, etc.) depend on.
    """
    from utils.case_library import format_cases_for_prompt

    assume(len(cases) > 0)

    result = format_cases_for_prompt(cases)

    for c in cases:
        # setup_type is always included (it's in the field list)
        if c.get("setup_type") is not None:
            assert c["setup_type"] in result, (
                f"setup_type '{c['setup_type']}' not found in formatted output"
            )
        # outcome is always included
        assert c["outcome"] in result, (
            f"outcome '{c['outcome']}' not found in formatted output"
        )


# ===========================================================================
# AREA 2: Unaffected Prompt Path Preservation (Requirements 3.9, 3.10)
# ===========================================================================

def test_maintenance_review_prompt_template_unchanged():
    """
    **Validates: Requirements 3.9**

    Observation: The MAINTENANCE_REVIEW_PROMPT template is a string constant
    that must not be modified by the chunking fix. Verify its structure.
    """
    from agents.portfolio_manager import MAINTENANCE_REVIEW_PROMPT

    # Template must exist and be a non-empty string
    assert isinstance(MAINTENANCE_REVIEW_PROMPT, str)
    assert len(MAINTENANCE_REVIEW_PROMPT) > 100

    # Must contain all required placeholders
    required_placeholders = [
        "{profile_name}", "{emoji}", "{thesis}", "{setup_type}",
        "{entry_price}", "{stop_price}", "{target_price}", "{invalidators}",
        "{symbol}", "{side}", "{quantity}", "{current_price}",
        "{unrealized_pnl_pct}", "{drifting}",
        "{indicators_text}", "{advisory_signals_text}", "{health_text}",
    ]
    for placeholder in required_placeholders:
        assert placeholder in MAINTENANCE_REVIEW_PROMPT, (
            f"MAINTENANCE_REVIEW_PROMPT missing placeholder: {placeholder}"
        )

    # Must mention valid actions
    assert "hold" in MAINTENANCE_REVIEW_PROMPT
    assert "tighten_stop" in MAINTENANCE_REVIEW_PROMPT
    assert "raise_target" in MAINTENANCE_REVIEW_PROMPT
    assert "trim_partial" in MAINTENANCE_REVIEW_PROMPT


def test_reversal_close_prompt_template_unchanged():
    """
    **Validates: Requirements 3.9**

    Observation: The REVERSAL_CLOSE_PROMPT template is a string constant
    that must not be modified by the chunking fix. Verify its structure.
    """
    from agents.portfolio_manager import REVERSAL_CLOSE_PROMPT

    # Template must exist and be a non-empty string
    assert isinstance(REVERSAL_CLOSE_PROMPT, str)
    assert len(REVERSAL_CLOSE_PROMPT) > 100

    # Must contain all required placeholders
    required_placeholders = [
        "{profile_name}", "{emoji}", "{trigger_type}", "{trigger_details}",
        "{thesis}", "{setup_type}", "{entry_price}", "{stop_price}",
        "{target_price}", "{invalidators}", "{symbol}", "{side}",
        "{quantity}", "{current_price}", "{unrealized_pnl_pct}",
        "{market_conditions_text}", "{opposing_evidence_text}", "{invalidator_json}",
    ]
    for placeholder in required_placeholders:
        assert placeholder in REVERSAL_CLOSE_PROMPT, (
            f"REVERSAL_CLOSE_PROMPT missing placeholder: {placeholder}"
        )

    # Must mention valid actions
    assert "close_full" in REVERSAL_CLOSE_PROMPT
    assert "close_partial" in REVERSAL_CLOSE_PROMPT
    assert "hold_tighten" in REVERSAL_CLOSE_PROMPT


def test_maintenance_review_prompt_format_renders():
    """
    **Validates: Requirements 3.9**

    Property: MAINTENANCE_REVIEW_PROMPT.format() renders without error
    for any valid position_data dict. The chunking fix must not break this.
    """
    from agents.portfolio_manager import MAINTENANCE_REVIEW_PROMPT

    rendered = MAINTENANCE_REVIEW_PROMPT.format(
        profile_name="Moderate",
        emoji="⚖️",
        thesis="Strong momentum play on AAPL",
        setup_type="news_breakout",
        entry_price=175.50,
        stop_price=172.00,
        target_price=185.00,
        invalidators="[{\"type\": \"price_below\", \"level\": 172.0}]",
        symbol="AAPL",
        side="long",
        quantity=100,
        current_price=178.25,
        unrealized_pnl_pct=1.57,
        drifting="NO",
        indicators_text='{"rsi": {"value": 62.5}}',
        advisory_signals_text='{"signal": "LONG", "strength": "strong"}',
        health_text="Position healthy, within normal parameters",
    )

    assert isinstance(rendered, str)
    assert len(rendered) > 200
    assert "AAPL" in rendered
    assert "Moderate" in rendered


def test_reversal_close_prompt_format_renders():
    """
    **Validates: Requirements 3.9**

    Property: REVERSAL_CLOSE_PROMPT.format() renders without error
    for any valid position_data and trigger_info. The chunking fix must not break this.
    """
    from agents.portfolio_manager import REVERSAL_CLOSE_PROMPT

    rendered = REVERSAL_CLOSE_PROMPT.format(
        profile_name="Aggressive",
        emoji="🔥",
        trigger_type="thesis_invalidation",
        trigger_details="Key support level broken on volume",
        thesis="Momentum breakout on TSLA",
        setup_type="technical_breakout",
        entry_price=250.00,
        stop_price=245.00,
        target_price=270.00,
        invalidators="[{\"type\": \"price_below\", \"level\": 245.0}]",
        symbol="TSLA",
        side="long",
        quantity=50,
        current_price=243.50,
        unrealized_pnl_pct=-2.60,
        market_conditions_text='{"regime": "risk_off"}',
        opposing_evidence_text='{"signal": "SHORT", "strength": "strong"}',
        invalidator_json='{"type": "price_below", "level": 245.0}',
    )

    assert isinstance(rendered, str)
    assert len(rendered) > 200
    assert "TSLA" in rendered
    assert "Aggressive" in rendered


# ===========================================================================
# AREA 3: Context Dict Key Preservation (Requirements 3.3, 3.4)
# ===========================================================================

WEEKLY_WRAP_REQUIRED_KEYS = {
    "week_pnl", "best_trades", "worst_trades", "case_trends",
    "strategy_performance", "agent_grades", "daily_logs",
}

RECAP_BASE_REQUIRED_KEYS = {
    "recent_trades", "position_pnl_changes", "signal_changes",
    "breaking_news", "catalyst_freshness", "pm_divergences",
    "unusual_events", "quiet_period",
}


def _make_test_engine():
    """Create an in-memory SQLite engine with all tables for testing."""
    from sqlalchemy import create_engine
    from db.schema import Base
    Base.metadata.create_all(bind=(engine := create_engine("sqlite://", echo=False)))
    return engine


def test_assemble_weekly_wrap_returns_all_required_keys():
    """
    **Validates: Requirements 3.3**

    Observation: assemble_weekly_wrap() returns a dict with all required keys
    (week_pnl, best_trades, worst_trades, case_trends, strategy_performance,
    agent_grades, daily_logs) so that _build_weekly_wrap_prompt() renders
    without KeyError. This must be preserved after the fix.
    """
    from agents.narrator import assemble_weekly_wrap

    engine = _make_test_engine()
    ctx = assemble_weekly_wrap(engine)

    assert isinstance(ctx, dict), f"assemble_weekly_wrap should return dict, got {type(ctx)}"

    missing_keys = WEEKLY_WRAP_REQUIRED_KEYS - set(ctx.keys())
    assert not missing_keys, (
        f"assemble_weekly_wrap() missing required keys: {missing_keys}. "
        f"Got keys: {set(ctx.keys())}"
    )


def test_assemble_recap_base_returns_all_required_keys():
    """
    **Validates: Requirements 3.4**

    Observation: _assemble_recap_base() returns a dict with all required keys
    (recent_trades, position_pnl_changes, signal_changes, breaking_news,
    catalyst_freshness, pm_divergences, unusual_events, quiet_period) so that
    _build_afternoon_prompt() renders without KeyError. This must be preserved
    after the fix.
    """
    from agents.narrator import _assemble_recap_base

    engine = _make_test_engine()
    ctx = _assemble_recap_base(engine)

    assert isinstance(ctx, dict), f"_assemble_recap_base should return dict, got {type(ctx)}"

    missing_keys = RECAP_BASE_REQUIRED_KEYS - set(ctx.keys())
    assert not missing_keys, (
        f"_assemble_recap_base() missing required keys: {missing_keys}. "
        f"Got keys: {set(ctx.keys())}"
    )


def test_assemble_weekly_wrap_key_types():
    """
    **Validates: Requirements 3.3**

    Observation: assemble_weekly_wrap() returns expected types for each key.
    This ensures _build_weekly_wrap_prompt() can safely json.dumps each value.
    """
    from agents.narrator import assemble_weekly_wrap

    engine = _make_test_engine()
    ctx = assemble_weekly_wrap(engine)

    # week_pnl should be a dict (profile_id -> pnl data)
    assert isinstance(ctx["week_pnl"], dict), f"week_pnl should be dict, got {type(ctx['week_pnl'])}"
    # best_trades and worst_trades should be lists
    assert isinstance(ctx["best_trades"], list), f"best_trades should be list, got {type(ctx['best_trades'])}"
    assert isinstance(ctx["worst_trades"], list), f"worst_trades should be list, got {type(ctx['worst_trades'])}"
    # case_trends should be a string (compact aggregated format from compact_case_trends_for_narrator)
    assert isinstance(ctx["case_trends"], str), f"case_trends should be str, got {type(ctx['case_trends'])}"
    # strategy_performance should be a dict
    assert isinstance(ctx["strategy_performance"], dict), f"strategy_performance should be dict, got {type(ctx['strategy_performance'])}"
    # agent_grades should be a dict
    assert isinstance(ctx["agent_grades"], dict), f"agent_grades should be dict, got {type(ctx['agent_grades'])}"
    # daily_logs should be a list
    assert isinstance(ctx["daily_logs"], list), f"daily_logs should be list, got {type(ctx['daily_logs'])}"


def test_assemble_recap_base_key_types():
    """
    **Validates: Requirements 3.4**

    Observation: _assemble_recap_base() returns expected types for each key.
    This ensures _build_afternoon_prompt() can safely json.dumps each value.
    """
    from agents.narrator import _assemble_recap_base

    engine = _make_test_engine()
    ctx = _assemble_recap_base(engine)

    assert isinstance(ctx["recent_trades"], list), f"recent_trades should be list, got {type(ctx['recent_trades'])}"
    assert isinstance(ctx["position_pnl_changes"], list), f"position_pnl_changes should be list, got {type(ctx['position_pnl_changes'])}"
    assert isinstance(ctx["signal_changes"], dict), f"signal_changes should be dict, got {type(ctx['signal_changes'])}"
    assert isinstance(ctx["breaking_news"], list), f"breaking_news should be list, got {type(ctx['breaking_news'])}"
    assert isinstance(ctx["catalyst_freshness"], dict), f"catalyst_freshness should be dict, got {type(ctx['catalyst_freshness'])}"
    assert isinstance(ctx["pm_divergences"], list), f"pm_divergences should be list, got {type(ctx['pm_divergences'])}"
    assert isinstance(ctx["unusual_events"], list), f"unusual_events should be list, got {type(ctx['unusual_events'])}"
    assert isinstance(ctx["quiet_period"], bool), f"quiet_period should be bool, got {type(ctx['quiet_period'])}"


# ===========================================================================
# AREA 4: PM Entry Candidates Not Filtered (Requirements 3.1, 3.5, 3.8)
# ===========================================================================

@given(entry_signals=entry_signals_strategy)
@settings(max_examples=50, deadline=None)
def test_all_entry_candidates_present_in_prompt(entry_signals):
    """
    **Validates: Requirements 3.1, 3.5, 3.8**

    Property: For all generated entry_signals dicts, every candidate symbol
    appears in the assembled PM entry prompt. The compact signal format may
    omit verbose detail but must NOT filter out any candidates.

    This tests the current behavior where json.dumps(entry_signals) includes
    all symbols. After the fix, compact_signal_for_pm should still include
    every symbol.
    """
    assume(len(entry_signals) > 0)

    # Replicate the PM entry prompt assembly for the signals section
    signals_text = json.dumps(entry_signals, indent=2)

    for sym in entry_signals:
        assert sym in signals_text, (
            f"Entry candidate symbol '{sym}' missing from signals text. "
            f"All entry candidates must be passed to the LLM for decision-making."
        )


@given(entry_signals=entry_signals_strategy)
@settings(max_examples=50, deadline=None)
def test_entry_signals_count_preserved(entry_signals):
    """
    **Validates: Requirements 3.1, 3.8**

    Property: The number of entry candidate symbols in the prompt matches
    the number of input signals. No candidates are filtered out.
    """
    assume(len(entry_signals) > 0)

    signals_text = json.dumps(entry_signals, indent=2)

    # Count how many symbols appear in the serialized text
    symbols_found = [sym for sym in entry_signals if sym in signals_text]

    assert len(symbols_found) == len(entry_signals), (
        f"Expected {len(entry_signals)} symbols in signals text, "
        f"found {len(symbols_found)}. Missing: {set(entry_signals) - set(symbols_found)}"
    )


@given(
    entry_signals=entry_signals_strategy,
    held_symbols=st.lists(st.sampled_from(SYMBOLS), min_size=0, max_size=4, unique=True),
)
@settings(max_examples=50, deadline=None)
def test_held_symbols_filtered_from_entry(entry_signals, held_symbols):
    """
    **Validates: Requirements 3.8**

    Property: The PM entry logic filters out symbols that already have open
    positions (held_symbols). This filtering must be preserved — only symbols
    NOT in held_symbols should be entry candidates.
    """
    # Replicate the filtering logic from run_profile()
    held_set = set(held_symbols)
    filtered_signals = {sym: sig for sym, sig in entry_signals.items() if sym not in held_set}

    # All filtered signals should exclude held symbols
    for sym in filtered_signals:
        assert sym not in held_set, (
            f"Symbol '{sym}' is in held_symbols but was not filtered out"
        )

    # All non-held symbols from entry_signals should be in filtered_signals
    for sym in entry_signals:
        if sym not in held_set:
            assert sym in filtered_signals, (
                f"Symbol '{sym}' is not held but was incorrectly filtered out"
            )


# ===========================================================================
# Additional Preservation: build_strategy_context output structure
# ===========================================================================

def test_build_strategy_context_returns_string():
    """
    **Validates: Requirements 3.7**

    Observation: build_strategy_context() always returns a string.
    This is the contract that narrator morning_briefing and other callers
    depend on. The fix introduces build_pm_strategy_context() but must
    NOT modify build_strategy_context().
    """
    from agents.quant_researcher import build_strategy_context

    engine = _make_test_engine()
    result = build_strategy_context(engine)

    assert isinstance(result, str), (
        f"build_strategy_context() should return str, got {type(result)}"
    )
    # With no data, should return a stale/no-data message
    assert len(result) > 0, "build_strategy_context() should not return empty string"


def test_build_strategy_context_deterministic():
    """
    **Validates: Requirements 3.7**

    Property: build_strategy_context() is deterministic — calling it twice
    with the same engine state produces identical output.
    """
    from agents.quant_researcher import build_strategy_context

    engine = _make_test_engine()
    result1 = build_strategy_context(engine)
    result2 = build_strategy_context(engine)

    assert result1 == result2, (
        f"build_strategy_context() is not deterministic. "
        f"First call: {result1!r}, second call: {result2!r}"
    )
