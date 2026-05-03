"""
Bug Condition Exploration Test — Property 1: Prompt Assembly Exceeds Character Budgets

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10

This test encodes the EXPECTED (correct) behavior: prompt assembly functions
should produce prompts within character budgets and use compact representations.

On UNFIXED code this test is EXPECTED TO FAIL — failure confirms the bug exists:
- PM entry prompts exceed 18k chars and contain low-signal fields / full indicator objects
- Weekly wrap prompts exceed 25k chars with individual case objects and verbose logs
- Afternoon recap prompts exceed 20k chars with full signal objects and redundant P&L

DO NOT attempt to fix the test or the code when it fails.
"""

import json
from datetime import datetime, date

from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Realistic data fixture builders (deterministic, fast)
# ---------------------------------------------------------------------------

SETUP_TYPES = [
    "news_breakout", "gap_and_go", "technical_breakout", "momentum_fade",
    "gap_fill", "range_breakout", "vwap_reclaim", "earnings_reaction",
    "sector_rotation", "reversal",
]

SYMBOLS = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "GOOG", "AMZN", "META",
           "SPY", "QQQ", "IWM", "NFLX"]


def make_case(i: int) -> dict:
    """Build a realistic case dict with 13+ fields."""
    sym = SYMBOLS[i % len(SYMBOLS)]
    setup = SETUP_TYPES[i % len(SETUP_TYPES)]
    return {
        "id": 100 + i,
        "date": f"2025-01-{10 + i:02d}",
        "symbol": sym,
        "setup_type": setup,
        "catalyst_type": "earnings_beat",
        "float_profile": "mid_cap",
        "sector": "tech",
        "premarket_gap_pct": 5.2 + i * 0.3,
        "premarket_volume_rank": "high",
        "market_regime": "risk_on",
        "entry_timing": "first_15min",
        "bias": "LONG",
        "signal_strength": "strong",
        "rsi_at_entry": 55.0 + i,
        "above_vwap": "true",
        "above_daily_resistance": "true",
        "ema_trend": "bullish",
        "outcome": "success" if i % 3 != 2 else "failure",
        "pnl_pct": 3.5 + i * 0.5 if i % 3 != 2 else -(2.0 + i * 0.3),
        "holding_minutes": 45 + i * 10,
        "lesson": f"Strong momentum play works best when market regime is risk_on and stock is above daily resistance with high premarket volume rank and clear catalyst {i}",
        "conditions_for_success": [
            "market_regime=risk_on", "above_daily_resistance=true",
            f"premarket_gap_pct>{3 + i}", "signal_strength=strong",
        ],
        "conditions_to_avoid": [
            "entry_timing=open", f"volume_rank<{3 + i}",
            "market_regime=risk_off",
        ],
        "confidence": "high",
        "selection_score": 7.5 + i * 0.2,
        "execution_score": 6.8 + i * 0.3,
        "review_score": 7.0 + i * 0.1,
        "profile": "moderate",
    }


def make_full_signal(sym: str, i: int) -> dict:
    """Build a full analyst signal dict with indicator objects and verbose reasoning (~700 chars)."""
    reasoning = (
        f"The {sym} setup shows strong momentum with bullish technical indicators. "
        f"RSI is trending upward from oversold territory, MACD histogram is expanding, "
        f"and price has reclaimed VWAP with increasing volume. The catalyst is fresh "
        f"and the sector is showing relative strength. Key support at the 20 EMA "
        f"provides a natural stop level. The risk/reward is favorable with a clear "
        f"invalidation level. Multiple timeframes are aligned bullish. The stock has "
        f"been consolidating near resistance and a breakout above the key level would "
        f"trigger significant buying pressure from short covering and momentum traders. "
        f"Volume profile suggests strong institutional interest at current levels. "
        f"The options flow is also bullish with heavy call buying at the next strike. "
        f"Overall this is a high-conviction setup with multiple confirming factors."
    )
    return {
        "signal": "LONG",
        "strength": "strong",
        "setup_type": SETUP_TYPES[i % len(SETUP_TYPES)],
        "confidence": "high",
        "entry": 150.0 + i * 10,
        "stop": 145.0 + i * 10,
        "target": 165.0 + i * 10,
        "invalidation": f"Below {143.0 + i * 10} on volume",
        "key_levels": [148.0 + i * 10, 152.0 + i * 10, 160.0 + i * 10],
        "symbol_class": "momentum",
        "catalyst_warning": "stale catalyst" if i == 2 else None,
        "reasoning": reasoning,
        "indicators": {
            "rsi": {
                "value": 58.5 + i,
                "signal": "neutral",
                "period": 14,
                "history": [52.0, 54.5, 56.0, 57.8, 58.5 + i],
            },
            "macd": {
                "value": 1.25 + i * 0.1,
                "signal_line": 0.95 + i * 0.1,
                "histogram": 0.30,
                "crossover": "bullish",
            },
            "bollinger": {
                "upper": 165.0 + i * 10,
                "middle": 155.0 + i * 10,
                "lower": 145.0 + i * 10,
                "position": "upper",
            },
            "vwap": {
                "value": 152.0 + i * 10,
                "above": True,
                "distance_pct": 1.3,
            },
            "ema": {
                "ema_9": 153.0 + i * 10,
                "ema_20": 150.0 + i * 10,
                "ema_50": 145.0 + i * 10,
                "trend": "bullish",
            },
        },
    }


def make_win_rate(setup_type: str, total: int, wins: int) -> dict:
    return {
        "setup_type": setup_type,
        "total": total,
        "wins": wins,
        "win_rate": round(wins / total * 100, 1) if total > 0 else 0.0,
        "avg_pnl_pct": round((wins - (total - wins)) * 0.5, 2),
        "avg_score": 7.0,
    }


def make_strategy_context() -> str:
    """Build a strategy context string that excludes pipeline-stage strategies (fixed behavior)."""
    lines = [
        "Market conditions: Bullish momentum with sector rotation into tech and growth names",
        "Primary strategy today: news_breakout",
        "Regime note: Risk-on environment with low VIX and strong breadth",
        "",
        "✅ News Breakout (fit: 8/10, internal: 72% over 25 cases)",
        "   → Analyst: Focus on high-gap stocks with fresh catalysts",
        "   → PM: Size up on strong signals, use tight stops",
        "✅ Gap and Go (fit: 7/10, internal: 65% over 18 cases)",
        "   → Analyst: Best in first 30 minutes with volume confirmation",
        "   → PM: Quick entries, trail stops aggressively",
        "",
        "Agent-proposed strategies (live):",
        "  📌 VWAP Reclaim EOD (vwap_reclaim_eod) — stage: live [68%, 12 trades]",
    ]
    return "\n".join(lines)


def make_daily_log(i: int) -> dict:
    """Build a DailyLog dict with verbose notes."""
    verbose_notes = (
        f"Day {i+1} was a mixed session with early strength fading into the close. "
        f"The conservative profile avoided most of the volatility by sitting out the "
        f"first 30 minutes. The moderate profile took two trades, winning on NVDA but "
        f"losing on AMD due to a late-day reversal. The aggressive profile was the most "
        f"active with four trades, capturing a nice move in TSLA but giving back gains "
        f"on a failed gap-and-go in GOOG. Overall market breadth was positive but "
        f"narrowing, suggesting caution for tomorrow. Key lessons: respect the entry "
        f"timing windows and don't chase extended moves in the afternoon session."
    )
    return {
        "date": f"2025-01-{13 + i:02d}",
        "starting_equity": 100000.0 + i * 50,
        "ending_equity": 100000.0 + i * 50 + (-50 + i * 100),
        "trades_taken": 3 + i,
        "winning_trades": 1 + i,
        "losing_trades": 2,
        "daily_pnl": -50.0 + i * 100,
        "daily_pnl_pct": round((-50.0 + i * 100) / 100000 * 100, 2),
        "notes": verbose_notes,
    }


def make_case_trend(i: int) -> dict:
    """Build a case trend dict with 10+ fields."""
    sym = SYMBOLS[i % len(SYMBOLS)]
    setup = SETUP_TYPES[i % len(SETUP_TYPES)]
    outcome = "success" if i % 3 != 2 else "failure"
    return {
        "symbol": sym,
        "date": f"2025-01-{13 + (i % 5):02d}",
        "setup_type": setup,
        "outcome": outcome,
        "pnl_pct": 3.0 + i * 0.2 if outcome == "success" else -(2.0 + i * 0.1),
        "lesson": f"Setup {setup} works best with strong volume and clear catalyst in risk-on regime for {sym}",
        "profile": ["conservative", "moderate", "aggressive"][i % 3],
        "catalyst_type": "earnings_beat" if i % 2 == 0 else "news_headline",
        "selection_score": 7.0 + (i % 4) * 0.5,
        "execution_score": 6.5 + (i % 4) * 0.3,
    }


def make_dynamic_strategy(i: int, has_trades: bool) -> tuple:
    """Build a DynamicStrategy performance dict."""
    key = f"strategy_{i}"
    total = (5 + i * 2) if has_trades else 0
    wins = (3 + i) if has_trades else 0
    return key, {
        "name": f"Dynamic Strategy {i}",
        "status": "active" if has_trades else "retired",
        "total_trades": total,
        "wins": wins,
        "win_rate": round(wins / total * 100, 1) if total > 0 else None,
        "avg_pnl_pct": round(1.5 + i * 0.3, 2) if has_trades else None,
        "retired_at": None,
        "retire_reason": None,
    }


# ---------------------------------------------------------------------------
# Helper: replicate PM entry prompt assembly (mirrors run_profile logic)
# ---------------------------------------------------------------------------

def assemble_pm_entry_prompt(cases, entry_signals, win_rates, strategy_context):
    """
    Replicate the PM entry user_prompt assembly from run_profile() — FIXED version.
    Uses compact helpers: format_cases_digest_for_pm, compact_signal_for_pm,
    filtered win rates, and pipeline-filtered strategy context.
    """
    from utils.prompt_compaction import format_cases_digest_for_pm, compact_signal_for_pm

    cases_text = format_cases_digest_for_pm(cases)

    # Filter win rates to only include setup types matching entry candidates' signals
    entry_setup_types = {sig.get("setup_type") for sig in entry_signals.values() if sig.get("setup_type")}
    if entry_setup_types:
        win_rates = [wr for wr in win_rates if wr.get("setup_type") in entry_setup_types]

    # Win rate formatting (mirrors run_profile)
    if win_rates:
        win_rate_lines = ["Setup type win rates from case library:"]
        for r in sorted(win_rates, key=lambda x: x["win_rate"], reverse=True):
            flag = " ⚠️ avoid or reduce size" if r["win_rate"] < 40 and r["total"] >= 5 else ""
            win_rate_lines.append(
                f"  {r['setup_type']}: {r['win_rate']}% ({r['wins']}/{r['total']}) "
                f"avg pnl {r['avg_pnl_pct'] or 0:+.1f}%{flag}"
            )
        win_rate_text = "\n".join(win_rate_lines)
    else:
        win_rate_text = "No setup win rate data yet."

    portfolio = {"cash": 95000, "positions": [], "daily_pnl": 0}
    feedback_text = "No execution feedback yet."
    meta_text = "None yet"
    news_text = "No breaking news"
    health_text = "No health data"

    # Build compact signal text for entry candidates
    compact_signals_text = "\n".join(compact_signal_for_pm(sym, sig) for sym, sig in entry_signals.items())

    user_prompt = f"""
Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
Profile: Moderate ⚖️

CURRENT PORTFOLIO:
{json.dumps(portfolio, indent=2)}

ANALYST SIGNALS:
{compact_signals_text if compact_signals_text else "No entry signals"}

EXECUTION FEEDBACK (your profile only):
{feedback_text}

META-REVIEWER RECOMMENDATIONS (system-level feedback for your profile):
{meta_text}

SETUP WIN RATES (from case library — use to adjust position sizing):
{win_rate_text}

STRATEGY RECOMMENDATIONS (from Quant Researcher):
{strategy_context}

RELEVANT PAST CASES:
{cases_text}

BREAKING NEWS (from news monitor):
{news_text}

POSITION HEALTH (from health monitor):
{health_text}

BEHAVIORAL ADJUSTMENTS (auto-extracted from feedback — applied to your decisions):
No adjustments active

Make your trading decisions for this cycle.
NOTE: Open positions are managed by the two-tier review system. Only consider NEW entries here.
"""
    return user_prompt


# ---------------------------------------------------------------------------
# Helper: replicate weekly wrap prompt assembly
# ---------------------------------------------------------------------------

def assemble_weekly_wrap_prompt(case_trends, daily_logs, strategy_performance):
    """Replicate _build_weekly_wrap_prompt() — FIXED version.
    Uses compact_case_trends_for_narrator for case_trends,
    compact_daily_log_for_narrator for daily_logs,
    and filters strategy_performance to only traded strategies.
    """
    from utils.prompt_compaction import compact_case_trends_for_narrator, compact_daily_log_for_narrator

    today = date.today().isoformat()

    # Apply compaction: aggregate case trends
    compact_trends = compact_case_trends_for_narrator(case_trends)

    # Apply compaction: compact daily logs
    compact_logs = [compact_daily_log_for_narrator(log) for log in daily_logs]

    # Filter strategy_performance to only traded strategies
    traded_strategies = {
        k: v for k, v in strategy_performance.items()
        if (v.get("total_trades") or 0) > 0
    }

    ctx = {
        "week_pnl": {
            "conservative": {"total_pnl": 150.0, "wins": 2, "losses": 1, "total_trades": 3},
            "moderate": {"total_pnl": -50.0, "wins": 1, "losses": 2, "total_trades": 3},
            "aggressive": {"total_pnl": 300.0, "wins": 3, "losses": 1, "total_trades": 4},
        },
        "best_trades": [{"symbol": "NVDA", "pnl": 200.0, "pnl_pct": 3.5, "setup_type": "news_breakout"}],
        "worst_trades": [{"symbol": "AMD", "pnl": -150.0, "pnl_pct": -2.1, "setup_type": "gap_and_go"}],
        "case_trends": compact_trends,
        "strategy_performance": traded_strategies,
        "agent_grades": {"analyst": 7.5, "pm_moderate": 6.8},
        "daily_logs": compact_logs,
        "story_arc": {"current_arc": "recovery", "momentum": "building"},
        "confidence_regime": {"regime": "moderate", "score": 0.65},
    }

    prompt = f"""Write the weekly wrap for the week ending {today}.

WEEK P&L PER PROFILE:
{json.dumps(ctx.get('week_pnl', {}), indent=2)}

BEST TRADES OF THE WEEK:
{json.dumps(ctx.get('best_trades', []), indent=2)}

WORST TRADES OF THE WEEK:
{json.dumps(ctx.get('worst_trades', []), indent=2)}

CASE LIBRARY TRENDS:
{ctx.get('case_trends', 'No case trends.')}

STRATEGY PERFORMANCE:
{json.dumps(ctx.get('strategy_performance', {}), indent=2)}

AGENT GRADES (META REVIEWER):
{json.dumps(ctx.get('agent_grades', {}), indent=2)}

DAILY LOGS FOR THE WEEK:
{chr(10).join(ctx.get('daily_logs', [])) if ctx.get('daily_logs') else 'No daily logs.'}

STORY ARC:
{json.dumps(ctx.get('story_arc', {}), indent=2)}

CONFIDENCE REGIME:
{json.dumps(ctx.get('confidence_regime', {}), indent=2)}

Write the weekly wrap. Cover the week's P&L, highlight the best and worst trades, note strategy performance and any retirements, include agent grades if available, and end with what to watch next week."""

    return prompt


# ---------------------------------------------------------------------------
# Helper: replicate afternoon recap prompt assembly
# ---------------------------------------------------------------------------

def assemble_afternoon_recap_prompt(signal_changes, catalyst_freshness, recent_trades, aggregate_pnl):
    """Replicate _build_afternoon_prompt() — FIXED version.
    Uses compact signals (direction/strength/key_level only),
    filters catalyst_freshness to stale/degrading only,
    and strips individual pnl/pnl_pct when aggregate is present.
    """
    today = date.today().isoformat()

    # Compact signals: keep only direction, strength, key_level
    compact_signals = {}
    for sym, sig in signal_changes.items():
        compact_signals[sym] = {
            "signal": sig.get("signal"),
            "strength": sig.get("strength"),
            "key_level": sig.get("key_level"),
        }

    # Filter catalyst_freshness to only stale/degrading
    filtered_freshness = {
        sym: data for sym, data in catalyst_freshness.items()
        if data.get("freshness_state") in ("stale", "degrading")
    }

    # Strip individual pnl/pnl_pct when aggregate is present
    if recent_trades and aggregate_pnl:
        recent_trades = [
            {k: v for k, v in trade.items() if k not in ("pnl", "pnl_pct")}
            for trade in recent_trades
        ]

    ctx = {
        "recent_trades": recent_trades,
        "position_pnl_changes": [
            {"symbol": "AAPL", "profile": "moderate", "side": "long", "quantity": 10, "avg_cost": 175.0},
        ],
        "signal_changes": compact_signals,
        "breaking_news": [],
        "catalyst_freshness": filtered_freshness,
        "pm_divergences": [],
        "unusual_events": [],
        "aggregate_pnl": aggregate_pnl,
        "win_loss": {
            "conservative": {"wins": 1, "losses": 0, "total": 1},
            "moderate": {"wins": 1, "losses": 1, "total": 2},
            "aggressive": {"wins": 2, "losses": 1, "total": 3},
        },
        "equity_change": {
            "conservative": {"opening_equity": 100000, "current_equity": 100150, "change": 150, "change_pct": 0.15},
            "moderate": {"opening_equity": 100000, "current_equity": 99950, "change": -50, "change_pct": -0.05},
            "aggressive": {"opening_equity": 100000, "current_equity": 100300, "change": 300, "change_pct": 0.30},
        },
        "quiet_period": False,
        "story_arc": {"current_arc": "recovery", "momentum": "building"},
        "confidence_regime": {"regime": "moderate", "score": 0.65},
    }

    prompt = f"""Write the afternoon recap for {today} at 2:00 PM ET.

TRADES SINCE LAST UPDATE:
{json.dumps(ctx.get('recent_trades', []), indent=2)}

POSITION P&L CHANGES:
{json.dumps(ctx.get('position_pnl_changes', []), indent=2)}

SIGNAL CHANGES:
{json.dumps(ctx.get('signal_changes', {}), indent=2)}

BREAKING NEWS:
{json.dumps(ctx.get('breaking_news', []), indent=2)}

CATALYST FRESHNESS:
{json.dumps(ctx.get('catalyst_freshness', {}), indent=2)}

PM DIVERGENCES:
{json.dumps(ctx.get('pm_divergences', []), indent=2)}

UNUSUAL EVENTS:
{json.dumps(ctx.get('unusual_events', []), indent=2)}

MIDDAY AGGREGATE P&L PER PROFILE:
{json.dumps(ctx.get('aggregate_pnl', {}), indent=2)}

WIN/LOSS COUNT PER PROFILE:
{json.dumps(ctx.get('win_loss', {}), indent=2)}

TOTAL EQUITY CHANGE PER PROFILE:
{json.dumps(ctx.get('equity_change', {}), indent=2)}

QUIET PERIOD: {ctx.get('quiet_period', False)}

STORY ARC:
{json.dumps(ctx.get('story_arc', {}), indent=2)}

CONFIDENCE REGIME:
{json.dumps(ctx.get('confidence_regime', {}), indent=2)}

Write the afternoon recap. Include the midday performance summary across all profiles alongside recent activity."""

    return prompt


# ---------------------------------------------------------------------------
# LOW-SIGNAL FIELDS that should NOT appear in PM entry case digest
# ---------------------------------------------------------------------------

PM_ENTRY_LOW_SIGNAL_FIELDS = [
    "premarket_volume_rank",
    "rsi_at_entry",
    "ema_trend",
    "above_daily_resistance",
]


# ===========================================================================
# TEST 1: PM Entry Bug Condition
# ===========================================================================

@given(
    # Use Hypothesis to vary the number of signals (3-5) and which symbols
    num_signals=st.integers(min_value=3, max_value=5),
    num_extra_win_rates=st.integers(min_value=3, max_value=7),
)
@settings(max_examples=10, deadline=None)
def test_pm_entry_bug_condition(num_signals, num_extra_win_rates):
    """
    **Validates: Requirements 1.1, 1.2, 1.3, 1.4**

    Property: For PM entry prompts assembled with 5 cases (13+ fields each),
    3+ full analyst signals, all setup win rates, and full strategy context
    including pipeline-stage strategies:

    - Total assembled user_prompt chars <= 18,000
    - Cases section does NOT contain low-signal fields
    - Signals section does NOT contain full indicator detail objects
    - Win rates only include setup types matching entry candidates' signal setup_types
    - Strategy context does NOT include strategies with pipeline_stage in (backtest, paper_trade)

    On UNFIXED code this FAILS because:
    - format_cases_for_prompt includes all 13+ fields (~6k chars for 5 cases)
    - json.dumps(entry_signals) includes full indicator objects (~500-1000 chars each)
    - All win rates are included regardless of relevance
    - build_strategy_context includes pipeline-stage strategies
    """
    # Build 5 cases
    cases = [make_case(i) for i in range(5)]

    # Build entry signals
    signal_symbols = SYMBOLS[:num_signals]
    entry_signals = {sym: make_full_signal(sym, i) for i, sym in enumerate(signal_symbols)}

    # Build win rates — include signal setup types plus extras (irrelevant ones)
    entry_setup_types = {sig["setup_type"] for sig in entry_signals.values()}
    win_rates = [make_win_rate(st, 20, 12) for st in entry_setup_types]
    # Add irrelevant setup types
    extra_setups = [s for s in SETUP_TYPES if s not in entry_setup_types][:num_extra_win_rates]
    win_rates.extend([make_win_rate(st, 15, 6) for st in extra_setups])

    strategy_ctx = make_strategy_context()

    # Assemble the prompt using the same logic as run_profile()
    user_prompt = assemble_pm_entry_prompt(cases, entry_signals, win_rates, strategy_ctx)

    # --- Assert 1: Total prompt chars <= 18,000 ---
    assert len(user_prompt) <= 18_000, (
        f"Bug confirmed: PM entry prompt is {len(user_prompt)} chars, exceeds 18,000 budget. "
        f"Cases contribute ~{len(format_cases_text(cases))} chars of unfiltered data. "
        f"Signals contribute ~{len(json.dumps(entry_signals, indent=2))} chars with full indicator objects."
    )

    # --- Assert 2: Cases section does NOT contain low-signal fields ---
    cases_start = user_prompt.find("RELEVANT PAST CASES:")
    cases_end = user_prompt.find("BREAKING NEWS")
    if cases_start != -1 and cases_end != -1:
        cases_section = user_prompt[cases_start:cases_end]
        for field in PM_ENTRY_LOW_SIGNAL_FIELDS:
            assert field not in cases_section, (
                f"Bug confirmed: PM entry cases section contains low-signal field '{field}'. "
                f"format_cases_for_prompt() includes all fields without filtering for PM entry context."
            )

    # --- Assert 3: Signals section does NOT contain full indicator detail objects ---
    signals_start = user_prompt.find("ANALYST SIGNALS:")
    signals_end = user_prompt.find("EXECUTION FEEDBACK")
    if signals_start != -1 and signals_end != -1:
        signals_section = user_prompt[signals_start:signals_end]
        assert '"indicators"' not in signals_section, (
            f"Bug confirmed: PM entry signals section contains full 'indicators' objects. "
            f"json.dumps(entry_signals) includes verbose indicator detail that PM doesn't need."
        )

    # --- Assert 4: Win rates only include setup types matching entry candidates ---
    win_rate_start = user_prompt.find("SETUP WIN RATES")
    win_rate_end = user_prompt.find("STRATEGY RECOMMENDATIONS")
    if win_rate_start != -1 and win_rate_end != -1 and entry_setup_types:
        win_rate_section = user_prompt[win_rate_start:win_rate_end]
        for wr in win_rates:
            st_type = wr["setup_type"]
            if st_type not in entry_setup_types:
                assert st_type not in win_rate_section, (
                    f"Bug confirmed: PM entry win rates include irrelevant setup type '{st_type}' "
                    f"not matching any entry candidate signal setup_types {entry_setup_types}."
                )

    # --- Assert 5: Strategy context does NOT include pipeline-stage strategies ---
    strategy_start = user_prompt.find("STRATEGY RECOMMENDATIONS")
    strategy_end = user_prompt.find("RELEVANT PAST CASES")
    if strategy_start != -1 and strategy_end != -1:
        strategy_section = user_prompt[strategy_start:strategy_end]
        assert "backtest" not in strategy_section.lower() or "in pipeline" not in strategy_section.lower(), (
            f"Bug confirmed: PM entry strategy context includes pipeline-stage strategies "
            f"(backtest/paper_trade) that are not actionable for entry decisions."
        )


def format_cases_text(cases):
    """Helper to measure cases text size."""
    from utils.prompt_compaction import format_cases_digest_for_pm
    return format_cases_digest_for_pm(cases)


# ===========================================================================
# TEST 2: Weekly Wrap Bug Condition
# ===========================================================================

@given(
    num_case_trends=st.integers(min_value=20, max_value=30),
    num_untraded=st.integers(min_value=3, max_value=6),
)
@settings(max_examples=10, deadline=None)
def test_weekly_wrap_bug_condition(num_case_trends, num_untraded):
    """
    **Validates: Requirements 1.5, 1.6, 1.7**

    Property: For weekly_wrap prompts assembled with 20-30 case trend objects
    (10+ fields each), 5 full DailyLog entries with verbose notes, and
    10 DynamicStrategy records (including untraded ones):

    - Total assembled prompt chars <= 25,000
    - case_trends are aggregated (not individual objects)
    - daily_logs use compact format (not verbose notes)
    - strategy_performance only includes strategies with trades

    On UNFIXED code this FAILS because:
    - assemble_weekly_wrap() dumps all individual case objects via json.dumps
    - DailyLog entries include full verbose notes fields
    - All DynamicStrategy records included regardless of trade count
    """
    case_trends = [make_case_trend(i) for i in range(num_case_trends)]
    daily_logs = [make_daily_log(i) for i in range(5)]

    # Build strategy_performance: some with trades, some without
    strategy_performance = {}
    for i in range(10):
        has_trades = i >= num_untraded  # first num_untraded have 0 trades
        key, data = make_dynamic_strategy(i, has_trades)
        strategy_performance[key] = data

    prompt = assemble_weekly_wrap_prompt(case_trends, daily_logs, strategy_performance)

    # --- Assert 1: Total prompt chars <= 25,000 ---
    assert len(prompt) <= 25_000, (
        f"Bug confirmed: Weekly wrap prompt is {len(prompt)} chars, exceeds 25,000 budget. "
        f"Case trends contribute ~{len(json.dumps(case_trends, indent=2))} chars as individual objects. "
        f"Daily logs contribute ~{len(json.dumps(daily_logs, indent=2))} chars with verbose notes."
    )

    # --- Assert 2: case_trends are aggregated (not individual objects) ---
    trends_start = prompt.find("CASE LIBRARY TRENDS:")
    trends_end = prompt.find("STRATEGY PERFORMANCE:")
    if trends_start != -1 and trends_end != -1:
        trends_section = prompt[trends_start:trends_end]
        individual_count = trends_section.count('"symbol"')
        assert individual_count <= 5, (
            f"Bug confirmed: Weekly wrap case_trends contains {individual_count} individual case objects "
            f"instead of aggregated statistics. Expected aggregated win rate by setup_type, top lessons, "
            f"notable failures — not {num_case_trends} individual case dicts."
        )

    # --- Assert 3: daily_logs use compact format (not verbose notes) ---
    logs_start = prompt.find("DAILY LOGS FOR THE WEEK:")
    logs_end = prompt.find("STORY ARC:")
    if logs_start != -1 and logs_end != -1:
        logs_section = prompt[logs_start:logs_end]
        assert len(logs_section) <= 2000, (
            f"Bug confirmed: Weekly wrap daily_logs section is {len(logs_section)} chars. "
            f"Full verbose notes are included instead of compact format "
            f"(date, P&L, trade count, 1-line summary). Expected < 2000 chars for 5 logs."
        )

    # --- Assert 4: strategy_performance only includes strategies with trades ---
    strat_start = prompt.find("STRATEGY PERFORMANCE:")
    strat_end = prompt.find("AGENT GRADES")
    if strat_start != -1 and strat_end != -1:
        strat_section = prompt[strat_start:strat_end]
        untraded = [k for k, v in strategy_performance.items() if v.get("total_trades", 0) == 0]
        for key in untraded:
            assert f'"{key}"' not in strat_section, (
                f"Bug confirmed: Weekly wrap strategy_performance includes untraded strategy '{key}' "
                f"(total_trades=0). Only strategies with trades should be included."
            )


# ===========================================================================
# TEST 3: Afternoon Recap Bug Condition
# ===========================================================================

@given(
    num_signals=st.integers(min_value=10, max_value=12),
)
@settings(max_examples=10, deadline=None)
def test_afternoon_recap_bug_condition(num_signals):
    """
    **Validates: Requirements 1.8, 1.9, 1.10**

    Property: For afternoon_recap prompts assembled with 10-12 full analyst signal
    objects, catalyst freshness for 8 positions (5 "fresh"), and redundant
    individual + aggregate P&L:

    - Total assembled prompt chars <= 20,000
    - Signals use compact format (not full objects)
    - catalyst_freshness only includes "stale" or "degrading" positions
    - P&L data does not contain redundant individual trade P&L when aggregate is present

    On UNFIXED code this FAILS because:
    - _assemble_recap_base() stores full analyst signal objects
    - catalyst_freshness includes all positions regardless of freshness_state
    - assemble_afternoon_recap() includes both individual and aggregate P&L
    """
    # Build signal_changes dict
    signal_symbols = SYMBOLS[:num_signals]
    signal_changes = {sym: make_full_signal(sym, i) for i, sym in enumerate(signal_symbols)}

    # Build catalyst_freshness: 5 "fresh", 2 "stale", 1 "degrading"
    freshness_symbols = SYMBOLS[:8]
    catalyst_freshness = {}
    for i, sym in enumerate(freshness_symbols):
        if i < 5:
            state = "fresh"
        elif i < 7:
            state = "stale"
        else:
            state = "degrading"
        catalyst_freshness[sym] = {
            "last_researcher_update": datetime.utcnow().isoformat(),
            "last_breaking_news_update": None,
            "freshness_state": state,
            "source_type": "premarket_synthesis",
            "confidence": 0.7,
        }

    # Build recent_trades with individual P&L (redundant with aggregate)
    recent_trades = [
        {
            "symbol": sym,
            "direction": "LONG",
            "quantity": 10,
            "entry_price": 150.0 + i * 5,
            "exit_price": 155.0 + i * 5,
            "profile": "moderate",
            "status": "closed",
            "pnl": 50.0,
            "pnl_pct": 3.33,
            "entry_time": datetime.utcnow().isoformat(),
            "exit_time": datetime.utcnow().isoformat(),
            "reason_entry": "Strong signal",
            "reason_exit": "Target hit",
        }
        for i, sym in enumerate(freshness_symbols[:4])
    ]

    aggregate_pnl = {
        "conservative": 150.0,
        "moderate": -50.0,
        "aggressive": 300.0,
    }

    prompt = assemble_afternoon_recap_prompt(signal_changes, catalyst_freshness, recent_trades, aggregate_pnl)

    # --- Assert 1: Total prompt chars <= 20,000 ---
    assert len(prompt) <= 20_000, (
        f"Bug confirmed: Afternoon recap prompt is {len(prompt)} chars, exceeds 20,000 budget. "
        f"Signal changes contribute ~{len(json.dumps(signal_changes, indent=2))} chars with full objects. "
        f"Catalyst freshness includes {sum(1 for v in catalyst_freshness.values() if v['freshness_state'] == 'fresh')} "
        f"'fresh' positions that need no attention."
    )

    # --- Assert 2: Signals use compact format (not full objects) ---
    signals_start = prompt.find("SIGNAL CHANGES:")
    signals_end = prompt.find("BREAKING NEWS:")
    if signals_start != -1 and signals_end != -1:
        signals_section = prompt[signals_start:signals_end]
        assert '"indicators"' not in signals_section, (
            f"Bug confirmed: Afternoon recap signals section contains full 'indicators' objects. "
            f"Signals should use compact format (direction, strength, key level only)."
        )

    # --- Assert 3: catalyst_freshness only includes "stale" or "degrading" positions ---
    freshness_start = prompt.find("CATALYST FRESHNESS:")
    freshness_end = prompt.find("PM DIVERGENCES:")
    if freshness_start != -1 and freshness_end != -1:
        freshness_section = prompt[freshness_start:freshness_end]
        fresh_positions = [sym for sym, data in catalyst_freshness.items()
                          if data["freshness_state"] == "fresh"]
        for sym in fresh_positions:
            assert f'"{sym}"' not in freshness_section, (
                f"Bug confirmed: Afternoon recap catalyst_freshness includes 'fresh' position '{sym}'. "
                f"Only 'stale' or 'degrading' positions should be included."
            )

    # --- Assert 4: P&L data does not contain redundant individual trade P&L ---
    trades_start = prompt.find("TRADES SINCE LAST UPDATE:")
    trades_end = prompt.find("POSITION P&L CHANGES:")
    agg_start = prompt.find("MIDDAY AGGREGATE P&L PER PROFILE:")
    if trades_start != -1 and trades_end != -1 and agg_start != -1:
        trades_section = prompt[trades_start:trades_end]
        has_aggregate = '"conservative"' in prompt[agg_start:] or '"moderate"' in prompt[agg_start:]
        if has_aggregate:
            assert '"pnl":' not in trades_section or '"pnl_pct":' not in trades_section, (
                f"Bug confirmed: Afternoon recap contains both individual trade P&L "
                f"(in recent_trades) and aggregate P&L per profile. "
                f"Individual trade pnl/pnl_pct should be stripped when aggregate is present."
            )
