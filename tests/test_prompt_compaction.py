"""
Unit tests for prompt compaction helper functions.

Validates: Requirements 2.1, 2.4, 2.5, 2.7

Tests each helper function in utils/prompt_compaction.py:
- format_cases_digest_for_pm() — compact case digest for PM entry
- compact_signal_for_pm() — compact analyst signal for PM entry
- compact_daily_log_for_narrator() — compact daily log for weekly wrap
- compact_case_trends_for_narrator() — aggregated case trends for weekly wrap
"""

from utils.prompt_compaction import (
    format_cases_digest_for_pm,
    compact_signal_for_pm,
    compact_daily_log_for_narrator,
    compact_case_trends_for_narrator,
)


# ---------------------------------------------------------------------------
# Fixture builders (reuse patterns from test_prompt_chunking_bug.py)
# ---------------------------------------------------------------------------

SETUP_TYPES = [
    "news_breakout", "gap_and_go", "technical_breakout", "momentum_fade",
    "gap_fill", "range_breakout", "vwap_reclaim", "earnings_reaction",
]

SYMBOLS = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "GOOG", "AMZN", "META"]


def _make_case(i: int) -> dict:
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
        "lesson": (
            f"Strong momentum play works best when market regime is risk_on "
            f"and stock is above daily resistance with high premarket volume rank "
            f"and clear catalyst {i}"
        ),
        "conditions_for_success": [
            "market_regime=risk_on", "above_daily_resistance=true",
        ],
        "conditions_to_avoid": [
            "entry_timing=open", f"rsi_at_entry>{70 + i}",
            "market_regime=risk_off",
        ],
        "confidence": "high",
        "selection_score": 7.5,
        "execution_score": 6.8,
        "review_score": 7.0,
        "profile": "moderate",
    }


def _make_full_signal(sym: str, i: int, **overrides) -> dict:
    """Build a full analyst signal dict."""
    reasoning = (
        f"The {sym} setup shows strong momentum with bullish technical indicators. "
        f"RSI is trending upward from oversold territory, MACD histogram is expanding, "
        f"and price has reclaimed VWAP with increasing volume. The catalyst is fresh "
        f"and the sector is showing relative strength. Key support at the 20 EMA "
        f"provides a natural stop level. The risk/reward is favorable with a clear "
        f"invalidation level. Multiple timeframes are aligned bullish."
    )
    sig = {
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
        "catalyst_warning": None,
        "freshness_warning": None,
        "reasoning": reasoning,
        "indicators": {
            "rsi": {"value": 58.5, "signal": "neutral", "period": 14},
            "macd": {"value": 1.25, "signal_line": 0.95, "histogram": 0.30},
        },
    }
    sig.update(overrides)
    return sig


def _make_daily_log(i: int) -> dict:
    """Build a DailyLog dict with verbose notes."""
    verbose_notes = (
        f"Day {i+1} was a mixed session with early strength fading into the close. "
        f"The conservative profile avoided most of the volatility by sitting out the "
        f"first 30 minutes. The moderate profile took two trades, winning on NVDA but "
        f"losing on AMD due to a late-day reversal."
    )
    return {
        "date": f"2025-01-{13 + i:02d}",
        "starting_equity": 100000.0,
        "ending_equity": 100050.0 + i * 100,
        "trades_taken": 3 + i,
        "winning_trades": 1 + i,
        "losing_trades": 2,
        "daily_pnl": -50.0 + i * 100,
        "daily_pnl_pct": round((-50.0 + i * 100) / 100000 * 100, 2),
        "notes": verbose_notes,
    }


def _make_case_trend(i: int) -> dict:
    """Build a case trend dict."""
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
        "selection_score": 7.0,
        "execution_score": 6.5,
    }


# ===========================================================================
# Tests for format_cases_digest_for_pm
# ===========================================================================

class TestFormatCasesDigestForPm:
    """Validates: Requirement 2.1"""

    def test_zero_cases(self):
        """Empty case list returns fallback message."""
        result = format_cases_digest_for_pm([])
        assert result == "No relevant past cases found."

    def test_one_case_contains_required_fields(self):
        """Single case digest contains setup_type, outcome, pnl_pct, lesson."""
        case = _make_case(0)
        result = format_cases_digest_for_pm([case])

        assert case["setup_type"] in result
        assert case["outcome"] in result
        assert str(case["pnl_pct"]) in result
        # Lesson should be present (possibly truncated)
        assert "lesson:" in result
        # Avoid conditions should be present
        assert "avoid:" in result

    def test_five_cases_total_chars_under_2k(self):
        """5 cases produce output under 2k chars total."""
        cases = [_make_case(i) for i in range(5)]
        result = format_cases_digest_for_pm(cases)

        assert len(result) < 2000, (
            f"Digest for 5 cases is {len(result)} chars, expected < 2000"
        )

    def test_five_cases_all_have_required_fields(self):
        """Each of 5 cases has setup_type, outcome, pnl_pct, lesson in output."""
        cases = [_make_case(i) for i in range(5)]
        result = format_cases_digest_for_pm(cases)

        for c in cases:
            assert c["setup_type"] in result
            assert c["outcome"] in result
            assert str(c["pnl_pct"]) in result

    def test_does_not_contain_low_signal_fields_as_keys(self):
        """Digest omits low-signal fields as labeled keys (e.g. 'ema_trend: bullish')."""
        cases = [_make_case(i) for i in range(5)]
        result = format_cases_digest_for_pm(cases)

        # These fields should NOT appear as "field_name: value" entries in the digest.
        # They may appear inside avoid conditions (e.g. "rsi_at_entry>70") which is fine.
        low_signal_field_patterns = [
            "premarket_volume_rank: ",
            "rsi_at_entry: ",
            "ema_trend: ",
            "above_daily_resistance: ",
        ]
        for pattern in low_signal_field_patterns:
            assert pattern not in result, (
                f"Digest should not contain low-signal field key '{pattern.strip()}'"
            )

    def test_lesson_truncated_to_80_chars(self):
        """Long lessons are truncated to ~80 chars."""
        case = _make_case(0)
        # Ensure the lesson is longer than 80 chars
        case["lesson"] = "A" * 200
        result = format_cases_digest_for_pm([case])

        # The lesson portion should end with "..."
        assert "..." in result

    def test_conditions_to_avoid_as_string(self):
        """Handles conditions_to_avoid stored as JSON string."""
        case = _make_case(0)
        import json
        case["conditions_to_avoid"] = json.dumps(["entry_timing=open", "rsi>70"])
        result = format_cases_digest_for_pm([case])
        assert "avoid:" in result
        assert "entry_timing=open" in result

    def test_missing_optional_fields(self):
        """Handles cases with missing optional fields gracefully."""
        case = {
            "date": "2025-01-10",
            "symbol": "AAPL",
            "setup_type": "gap_and_go",
            "outcome": "success",
            "pnl_pct": 2.5,
        }
        result = format_cases_digest_for_pm([case])
        assert "gap_and_go" in result
        assert "success" in result
        assert "2.5" in result


# ===========================================================================
# Tests for compact_signal_for_pm
# ===========================================================================

class TestCompactSignalForPm:
    """Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5"""

    def test_analyst_owned_fields_preserved(self):
        """Signal with all fields preserves direction, strength, setup, confidence."""
        sig = _make_full_signal("AAPL", 0)
        result = compact_signal_for_pm("AAPL", sig)

        assert "AAPL" in result
        assert "LONG" in result
        assert "strong" in result
        assert "news_breakout" in result
        assert "high" in result   # confidence

    def test_no_entry_stop_target_displayed(self):
        """Entry, stop, and target fields are NOT displayed (Requirement 2.1)."""
        sig = _make_full_signal("AAPL", 0)
        result = compact_signal_for_pm("AAPL", sig)

        # Must NOT contain entry/stop/target lines or placeholders
        assert "entry:" not in result.lower()
        assert "stop:" not in result.lower()
        # "target:" should not appear as a field display
        # (it may appear in scaffold table header, but not as signal field)
        assert "entry: 150.0" not in result
        assert "stop: 145.0" not in result
        assert "target: 165.0" not in result
        assert "entry: ?" not in result
        assert "stop: ?" not in result
        assert "target: ?" not in result

    def test_invalidation_preserved(self):
        """Invalidation level is included when present."""
        sig = _make_full_signal("TSLA", 1)
        result = compact_signal_for_pm("TSLA", sig)
        assert "invalidation:" in result
        assert "Below" in result

    def test_key_levels_preserved(self):
        """Key levels are included when present."""
        sig = _make_full_signal("NVDA", 0)
        result = compact_signal_for_pm("NVDA", sig)
        assert "key_levels:" in result

    def test_key_levels_dict_format(self):
        """Key levels as dict shows support, resistance, VWAP."""
        sig = {
            "signal": "LONG",
            "strength": "strong",
            "setup_type": "breakout",
            "confidence": "high",
            "current_price": 150.0,
            "key_levels": {"support": 145.0, "resistance": 155.0, "vwap": 149.5},
        }
        result = compact_signal_for_pm("AAPL", sig)
        assert "support: 145.0" in result
        assert "resistance: 155.0" in result
        assert "VWAP: 149.5" in result

    def test_missing_null_fields_omitted(self):
        """Missing or null Analyst-owned fields are omitted, not shown as placeholders (Requirement 2.3)."""
        sig = {
            "signal": "LONG",
            # strength is missing
            # setup_type is missing
            # confidence is missing
            "current_price": 100.0,
        }
        result = compact_signal_for_pm("SPY", sig)
        assert "SPY" in result
        assert "LONG" in result
        assert "current_price: 100.0" in result
        # Missing fields should NOT appear with "?" placeholders
        assert "?" not in result
        assert "None" not in result
        assert "setup:" not in result
        assert "confidence:" not in result

    def test_indicators_omitted(self):
        """Full indicator objects are NOT included in compact output."""
        sig = _make_full_signal("AAPL", 0)
        result = compact_signal_for_pm("AAPL", sig)
        # The actual indicator object keys/structures should not appear
        assert '"rsi"' not in result
        assert '"macd"' not in result
        assert "signal_line" not in result
        assert '"period"' not in result
        assert '"histogram"' not in result
        # The indicators dict itself should not be serialized
        assert "58.5" not in result  # rsi value from indicators
        assert "0.95" not in result  # macd signal_line value

    def test_missing_optional_fields(self):
        """Signal with missing optional fields still works."""
        sig = {
            "signal": "SHORT",
            "strength": "moderate",
            "setup_type": "momentum_fade",
            "confidence": "medium",
        }
        result = compact_signal_for_pm("TSLA", sig)
        assert "TSLA" in result
        assert "SHORT" in result
        assert "moderate" in result
        # No invalidation, key_levels
        assert "invalidation:" not in result
        assert "key_levels:" not in result

    def test_current_price_included(self):
        """Current price from quote context is included when present."""
        sig = _make_full_signal("AAPL", 0, current_price=152.35)
        result = compact_signal_for_pm("AAPL", sig)
        assert "current_price: 152.35" in result

    def test_no_current_price_no_extra_output(self):
        """Signal without current_price does not include current_price line."""
        sig = _make_full_signal("TSLA", 0)
        result = compact_signal_for_pm("TSLA", sig)
        assert "current_price:" not in result

    def test_scaffold_none_shows_unavailable_message(self):
        """When scaffold_result is None, shows 'No geometry scaffold available' message (Requirement 2.4)."""
        sig = {"signal": "LONG", "strength": "strong", "setup_type": "breakout", "confidence": "high"}
        result = compact_signal_for_pm("AAPL", sig, scaffold_result=None)
        assert "No geometry scaffold available" in result
        assert "PM must not trade this signal" in result

    def test_scaffold_insufficient_data_shows_message(self):
        """When scaffold status is 'insufficient_data', shows unavailable message with reason (Requirement 2.4)."""
        sig = {"signal": "LONG", "strength": "strong", "setup_type": "breakout", "confidence": "high"}
        scaffold = {"status": "insufficient_data", "reason": "Missing current_price", "candidates": []}
        result = compact_signal_for_pm("AAPL", sig, scaffold_result=scaffold)
        assert "No geometry scaffold available" in result
        assert "PM must not trade this signal" in result
        assert "Missing current_price" in result

    def test_scaffold_not_tradeable_shows_message(self):
        """When scaffold status is 'not_tradeable_signal', shows no executable candidates message (Requirement 2.5)."""
        sig = {"signal": "HOLD", "strength": "weak"}
        scaffold = {"status": "not_tradeable_signal", "reason": "Signal direction is HOLD", "candidates": []}
        result = compact_signal_for_pm("AAPL", sig, scaffold_result=scaffold)
        assert "No executable geometry scaffold candidates" in result
        assert "PM must not trade this signal" in result
        assert "Signal direction is HOLD" in result

    def test_scaffold_ok_shows_candidates_table(self):
        """When scaffold status is 'ok', renders candidates table (Requirement 9.1)."""
        sig = {"signal": "LONG", "strength": "strong", "setup_type": "breakout", "confidence": "high"}
        scaffold = {
            "status": "ok",
            "reason": "",
            "candidates": [
                {
                    "candidate_id": "aapl_long_pullback_to_vwap_1",
                    "name": "pullback_to_vwap",
                    "entry_price": 149.50,
                    "stop_loss": 149.20,
                    "target": 150.10,
                    "risk_reward": 2.0,
                    "trigger": "Price touches VWAP",
                },
            ],
        }
        result = compact_signal_for_pm("AAPL", sig, scaffold_result=scaffold)
        assert "Geometry Scaffold Candidates" in result
        assert "aapl_long_pullback_to_vwap_1" in result
        assert "pullback_to_vwap" in result
        assert "149.5" in result
        assert "2.0" in result

    def test_backward_compatible_without_scaffold(self):
        """Calling without scaffold_result (old call pattern) still works."""
        sig = _make_full_signal("AAPL", 0)
        # Old callers pass (symbol, signal) — scaffold_result defaults to None
        result = compact_signal_for_pm("AAPL", sig)
        assert "AAPL" in result
        assert "LONG" in result
        # Should show scaffold unavailable message
        assert "No geometry scaffold available" in result


# ===========================================================================
# Tests for compact_daily_log_for_narrator
# ===========================================================================

class TestCompactDailyLogForNarrator:
    """Validates: Requirement 2.7"""

    def test_contains_date_pnl_trades_notes(self):
        """Output contains date, P&L, trade count, and truncated notes."""
        log = _make_daily_log(0)
        result = compact_daily_log_for_narrator(log)

        assert log["date"] in result
        assert "P&L:" in result
        assert "trades:" in result
        assert "note:" in result

    def test_output_under_200_chars(self):
        """Output is under 200 chars per log entry."""
        log = _make_daily_log(0)
        result = compact_daily_log_for_narrator(log)
        assert len(result) < 200, (
            f"Compact daily log is {len(result)} chars, expected < 200"
        )

    def test_notes_truncated(self):
        """Verbose notes are truncated to ~100 chars."""
        log = _make_daily_log(0)
        # The fixture has notes > 100 chars
        assert len(log["notes"]) > 100
        result = compact_daily_log_for_narrator(log)
        # The note portion should be truncated
        assert "..." in result

    def test_win_loss_counts(self):
        """Output includes winning and losing trade counts."""
        log = _make_daily_log(2)
        result = compact_daily_log_for_narrator(log)
        assert f"W{log['winning_trades']}" in result
        assert f"L{log['losing_trades']}" in result

    def test_missing_notes(self):
        """Handles log with no notes gracefully."""
        log = _make_daily_log(0)
        log["notes"] = ""
        result = compact_daily_log_for_narrator(log)
        assert log["date"] in result
        assert "P&L:" in result
        # No note section when notes are empty
        assert "note:" not in result


# ===========================================================================
# Tests for compact_case_trends_for_narrator
# ===========================================================================

class TestCompactCaseTrendsForNarrator:
    """Validates: Requirement 2.5"""

    def test_empty_trends(self):
        """Empty list returns fallback message."""
        result = compact_case_trends_for_narrator([])
        assert result == "No case trends this week."

    def test_win_rate_by_setup_type(self):
        """Output contains win rate aggregated by setup_type."""
        trends = [_make_case_trend(i) for i in range(10)]
        result = compact_case_trends_for_narrator(trends)

        assert "Win rate by setup:" in result
        # Should have setup types from the trends
        setup_types_in_trends = set(t["setup_type"] for t in trends)
        for st in setup_types_in_trends:
            assert st in result

    def test_top_lessons(self):
        """Output contains top lessons section."""
        trends = [_make_case_trend(i) for i in range(10)]
        result = compact_case_trends_for_narrator(trends)
        assert "Top lessons:" in result

    def test_notable_failures(self):
        """Output contains notable failures (outcome=failure, pnl_pct < -2%)."""
        trends = [_make_case_trend(i) for i in range(10)]
        result = compact_case_trends_for_narrator(trends)

        assert "Notable failures:" in result
        # Some trends have outcome=failure and pnl_pct < -2%
        failures = [
            t for t in trends
            if t["outcome"] == "failure" and t["pnl_pct"] < -2.0
        ]
        for f in failures:
            assert f["symbol"] in result

    def test_no_failures_shows_none(self):
        """When no notable failures exist, shows 'None'."""
        # All successes
        trends = []
        for i in range(5):
            t = _make_case_trend(i)
            t["outcome"] = "success"
            t["pnl_pct"] = 3.0
            trends.append(t)
        result = compact_case_trends_for_narrator(trends)
        assert "Notable failures:" in result
        assert "None" in result

    def test_win_rate_calculation_correct(self):
        """Win rates are calculated correctly per setup_type."""
        # Create 4 trends: 2 news_breakout (1 win, 1 loss), 2 gap_and_go (2 wins)
        trends = [
            {"setup_type": "news_breakout", "outcome": "success", "pnl_pct": 3.0,
             "lesson": "Good entry", "symbol": "AAPL"},
            {"setup_type": "news_breakout", "outcome": "failure", "pnl_pct": -1.0,
             "lesson": "Bad timing", "symbol": "TSLA"},
            {"setup_type": "gap_and_go", "outcome": "success", "pnl_pct": 2.5,
             "lesson": "Volume confirmed", "symbol": "NVDA"},
            {"setup_type": "gap_and_go", "outcome": "success", "pnl_pct": 1.8,
             "lesson": "Clean breakout", "symbol": "AMD"},
        ]
        result = compact_case_trends_for_narrator(trends)

        # news_breakout: 50% (1/2)
        assert "news_breakout: 50.0% (1/2)" in result
        # gap_and_go: 100% (2/2)
        assert "gap_and_go: 100.0% (2/2)" in result

    def test_output_is_aggregated_not_individual(self):
        """Output is aggregated stats, not individual case objects."""
        trends = [_make_case_trend(i) for i in range(20)]
        result = compact_case_trends_for_narrator(trends)

        # Should NOT contain individual case fields like "selection_score", "execution_score"
        assert "selection_score" not in result
        assert "execution_score" not in result
        # Should NOT be a JSON dump of individual objects
        assert '"symbol"' not in result


# ===========================================================================
# Tests for build_pm_strategy_context
# ===========================================================================

import json
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from db.schema import Base, AgentMemory, DynamicStrategy, get_session
from agents.quant_researcher import build_pm_strategy_context, build_strategy_context


def _create_test_engine():
    """Create an in-memory SQLite engine with all tables."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def _seed_strategy_recommendations(engine, strategies=None, timestamp=None):
    """Seed AgentMemory with quant_researcher strategy_recommendations."""
    if strategies is None:
        strategies = [
            {
                "strategy_key": "gap_and_go",
                "strategy_name": "Gap and Go",
                "fit_score": 8.5,
                "recommendation": "lean_into",
                "internal_win_rate": 0.65,
                "internal_cases": 20,
                "analyst_guidance": "Focus on high gap stocks",
                "pm_guidance": "Use tight stops",
            },
            {
                "strategy_key": "momentum_fade",
                "strategy_name": "Momentum Fade",
                "fit_score": 6.0,
                "recommendation": "use_with_caution",
                "internal_win_rate": 0.50,
                "internal_cases": 10,
                "analyst_guidance": "Watch for exhaustion signals",
                "pm_guidance": "Smaller position size",
            },
            {
                "strategy_key": "range_breakout",
                "strategy_name": "Range Breakout",
                "fit_score": 3.0,
                "recommendation": "avoid",
                "internal_win_rate": 0.30,
                "internal_cases": 5,
                "analyst_guidance": None,
                "pm_guidance": None,
            },
        ]
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat()

    data = {
        "timestamp": timestamp,
        "market_conditions_summary": "Risk-on regime with strong momentum",
        "primary_strategy": "gap_and_go",
        "regime_note": "Bullish bias across sectors",
        "strategies": strategies,
        "strategies_to_avoid": ["range_breakout"],
    }
    db = get_session(engine)
    db.add(AgentMemory(
        agent="quant_researcher",
        symbol=None,
        key="strategy_recommendations",
        value=json.dumps(data),
    ))
    db.commit()
    db.close()


def _seed_dynamic_strategies(engine, strategies_data=None):
    """Seed DynamicStrategy records with various pipeline stages."""
    if strategies_data is None:
        strategies_data = [
            # Live strategy — should appear in PM context
            {
                "key": "vwap_reclaim_eod",
                "name": "VWAP Reclaim EOD",
                "description": "Reclaim VWAP in last hour",
                "status": "live_100",
                "pipeline_stage": "live_100",
                "win_rate": 62.0,
                "total_trades": 15,
            },
            # Backtest strategy — should NOT appear in PM context
            {
                "key": "opening_range_fade",
                "name": "Opening Range Fade",
                "description": "Fade the opening range breakout",
                "status": "backtest",
                "pipeline_stage": "backtest",
                "win_rate": None,
                "total_trades": 0,
            },
            # Paper trade strategy — should NOT appear in PM context
            {
                "key": "gap_fill_reversal",
                "name": "Gap Fill Reversal",
                "description": "Trade gap fill reversals",
                "status": "paper_trade",
                "pipeline_stage": "paper_trade",
                "win_rate": 55.0,
                "total_trades": 5,
            },
            # Another live strategy
            {
                "key": "momentum_scalp",
                "name": "Momentum Scalp",
                "description": "Quick scalps on momentum",
                "status": "live_50",
                "pipeline_stage": "live_50",
                "win_rate": 58.0,
                "total_trades": 8,
            },
        ]
    db = get_session(engine)
    for sd in strategies_data:
        db.add(DynamicStrategy(
            key=sd["key"],
            name=sd["name"],
            description=sd["description"],
            status=sd["status"],
            pipeline_stage=sd["pipeline_stage"],
            win_rate=sd.get("win_rate"),
            total_trades=sd.get("total_trades", 0),
        ))
    db.commit()
    db.close()


class TestBuildPmStrategyContext:
    """
    Validates: Requirements 2.2, 3.7

    Tests that build_pm_strategy_context() excludes backtest/paper_trade
    pipeline strategies while build_strategy_context() remains unchanged.
    """

    def test_no_backtest_strategies_in_output(self):
        """PM strategy context must NOT contain backtest-stage strategies."""
        engine = _create_test_engine()
        _seed_strategy_recommendations(engine)
        _seed_dynamic_strategies(engine)

        result = build_pm_strategy_context(engine)

        assert "backtest" not in result.lower() or "backtest" not in result.split("pipeline")[0] if "pipeline" in result else "backtest" not in result
        # More precise: the backtest strategy name/key should not appear
        assert "opening_range_fade" not in result
        assert "Opening Range Fade" not in result

    def test_no_paper_trade_strategies_in_output(self):
        """PM strategy context must NOT contain paper_trade-stage strategies."""
        engine = _create_test_engine()
        _seed_strategy_recommendations(engine)
        _seed_dynamic_strategies(engine)

        result = build_pm_strategy_context(engine)

        assert "gap_fill_reversal" not in result
        assert "Gap Fill Reversal" not in result

    def test_no_pipeline_section_in_output(self):
        """PM strategy context must NOT contain the 'in pipeline' section."""
        engine = _create_test_engine()
        _seed_strategy_recommendations(engine)
        _seed_dynamic_strategies(engine)

        result = build_pm_strategy_context(engine)

        assert "in pipeline" not in result.lower()
        assert "backtesting" not in result
        assert "paper trading" not in result

    def test_live_dynamic_strategies_included(self):
        """Live dynamic strategies (live_50, live_100) should appear in PM context."""
        engine = _create_test_engine()
        _seed_strategy_recommendations(engine)
        _seed_dynamic_strategies(engine)

        result = build_pm_strategy_context(engine)

        assert "VWAP Reclaim EOD" in result
        assert "Momentum Scalp" in result

    def test_lean_into_strategies_included(self):
        """Strategies with recommendation 'lean_into' should appear."""
        engine = _create_test_engine()
        _seed_strategy_recommendations(engine)

        result = build_pm_strategy_context(engine)

        assert "Gap and Go" in result

    def test_use_with_caution_strategies_included(self):
        """Strategies with recommendation 'use_with_caution' should appear."""
        engine = _create_test_engine()
        _seed_strategy_recommendations(engine)

        result = build_pm_strategy_context(engine)

        assert "Momentum Fade" in result

    def test_avoid_strategies_excluded_from_main_list(self):
        """Strategies with recommendation 'avoid' should not appear in main list."""
        engine = _create_test_engine()
        _seed_strategy_recommendations(engine)

        result = build_pm_strategy_context(engine)

        # "Range Breakout" has recommendation "avoid" — should not appear as a
        # recommended strategy line (but may appear in "Avoid today" section)
        lines = result.split("\n")
        strategy_lines = [l for l in lines if l.startswith("✅") or l.startswith("⚠️")]
        for line in strategy_lines:
            assert "Range Breakout" not in line

    def test_no_recommendations_returns_fallback(self):
        """When no strategy recommendations exist, returns fallback message."""
        engine = _create_test_engine()

        result = build_pm_strategy_context(engine)

        assert result == "No strategy recommendations available yet."

    def test_stale_recommendations_returns_stale_message(self):
        """When recommendations are stale (>1 day old), returns stale message."""
        engine = _create_test_engine()
        stale_ts = (datetime.utcnow() - timedelta(days=2)).isoformat()
        _seed_strategy_recommendations(engine, timestamp=stale_ts)

        result = build_pm_strategy_context(engine)

        assert "stale" in result.lower()

    def test_build_strategy_context_unchanged(self):
        """
        Validates: Requirement 3.7

        build_strategy_context() must still include pipeline strategies
        (backtest/paper_trade) — it is NOT modified by this fix.
        """
        engine = _create_test_engine()
        _seed_strategy_recommendations(engine)
        _seed_dynamic_strategies(engine)

        result = build_strategy_context(engine)

        # The original function SHOULD include the pipeline section
        assert "in pipeline" in result.lower() or "Opening Range Fade" in result or "Gap Fill Reversal" in result

    def test_market_conditions_preserved(self):
        """PM strategy context includes market conditions summary."""
        engine = _create_test_engine()
        _seed_strategy_recommendations(engine)

        result = build_pm_strategy_context(engine)

        assert "Market conditions:" in result
        assert "Risk-on regime" in result

    def test_primary_strategy_preserved(self):
        """PM strategy context includes primary strategy."""
        engine = _create_test_engine()
        _seed_strategy_recommendations(engine)

        result = build_pm_strategy_context(engine)

        assert "Primary strategy today:" in result
        assert "gap_and_go" in result

    def test_regime_note_preserved(self):
        """PM strategy context includes regime note."""
        engine = _create_test_engine()
        _seed_strategy_recommendations(engine)

        result = build_pm_strategy_context(engine)

        assert "Regime note:" in result
        assert "Bullish bias" in result
