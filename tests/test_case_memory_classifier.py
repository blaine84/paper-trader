"""
Tests for case-memory exit classification.

Validates that closed trades are classified into exactly one of:
- bad_entry
- valid_entry_bad_exit_policy
- valid_exit_thesis_invalidated
- forced_exit_missing_metadata

Requirements: 6.6
"""

import pytest
from utils.case_memory_classifier import (
    CASE_MEMORY_EXIT_CATEGORIES,
    classify_trade_exit,
)


class TestThesisInvalidation:
    """Trade with thesis invalidation event → valid_exit_thesis_invalidated."""

    def test_thesis_invalidation_long(self):
        trade = {
            "id": 1,
            "symbol": "AMD",
            "setup_type": "news_breakout",
            "direction": "LONG",
            "entry_price": 150.0,
            "exit_price": 145.0,
            "pnl": -50.0,
            "status": "closed",
        }
        events = [
            {"event_type": "entry_executed"},
            {"event_type": "setup_exit_thesis_invalidated"},
        ]
        assert classify_trade_exit(trade, events) == "valid_exit_thesis_invalidated"

    def test_thesis_invalidation_short(self):
        trade = {
            "id": 2,
            "symbol": "TSLA",
            "setup_type": "news_catalyst",
            "direction": "SHORT",
            "entry_price": 200.0,
            "exit_price": 210.0,
            "pnl": -100.0,
            "status": "closed",
        }
        events = [
            {"event_type": "setup_exit_thesis_invalidated"},
        ]
        assert classify_trade_exit(trade, events) == "valid_exit_thesis_invalidated"


class TestForcedExitMissingMetadata:
    """Trade with missing criteria revalidation failure → forced_exit_missing_metadata."""

    def test_missing_criteria_in_reason(self):
        trade = {
            "id": 3,
            "symbol": "NVDA",
            "setup_type": "news_breakout",
            "direction": "LONG",
            "entry_price": 100.0,
            "exit_price": 102.0,
            "pnl": 20.0,
            "status": "closed",
        }
        events = [
            {
                "event_type": "setup_exit_revalidation_failed",
                "reason": "Extension denied: missing criteria for revalidation",
            },
        ]
        assert classify_trade_exit(trade, events) == "forced_exit_missing_metadata"

    def test_stale_data_in_message(self):
        trade = {
            "id": 4,
            "symbol": "AAPL",
            "setup_type": "trend_pullback",
            "direction": "LONG",
            "entry_price": 180.0,
            "exit_price": 181.0,
            "pnl": 10.0,
            "status": "closed",
        }
        events = [
            {
                "event_type": "setup_exit_revalidation_failed",
                "message": "Revalidation failed due to stale data (>30s)",
            },
        ]
        assert classify_trade_exit(trade, events) == "forced_exit_missing_metadata"

    def test_stale_data_in_payload(self):
        trade = {
            "id": 5,
            "symbol": "META",
            "setup_type": "news_catalyst",
            "direction": "LONG",
            "entry_price": 300.0,
            "exit_price": 305.0,
            "pnl": 50.0,
            "status": "closed",
        }
        events = [
            {
                "event_type": "setup_exit_revalidation_failed",
                "payload": {"reason": "Market data stale data unavailable"},
            },
        ]
        assert classify_trade_exit(trade, events) == "forced_exit_missing_metadata"


class TestValidEntryBadExitPolicy:
    """Profitable thesis-development trade force-closed by timer → valid_entry_bad_exit_policy."""

    def test_profitable_news_breakout_force_closed(self):
        """Profitable news_breakout force-closed by timer."""
        trade = {
            "id": 6,
            "symbol": "AMD",
            "setup_type": "news_breakout",
            "direction": "LONG",
            "entry_price": 150.0,
            "exit_price": 155.0,
            "pnl": 50.0,
            "status": "closed",
        }
        events = [
            {"event_type": "entry_executed"},
            {"event_type": "setup_exit_force_close"},
        ]
        assert classify_trade_exit(trade, events) == "valid_entry_bad_exit_policy"

    def test_profitable_short_news_catalyst_force_closed(self):
        """Profitable SHORT news_catalyst force-closed by timer."""
        trade = {
            "id": 7,
            "symbol": "TSLA",
            "setup_type": "news_catalyst",
            "direction": "SHORT",
            "entry_price": 200.0,
            "exit_price": 195.0,
            "pnl": 50.0,
            "status": "closed",
        }
        events = [
            {"event_type": "setup_exit_force_close"},
        ]
        assert classify_trade_exit(trade, events) == "valid_entry_bad_exit_policy"

    def test_profitable_trend_pullback_force_closed(self):
        """Profitable trend_pullback force-closed by timer."""
        trade = {
            "id": 8,
            "symbol": "MSFT",
            "setup_type": "trend_pullback",
            "direction": "LONG",
            "entry_price": 400.0,
            "exit_price": 405.0,
            "pnl": 50.0,
            "status": "closed",
        }
        events = [
            {"event_type": "setup_exit_force_close"},
        ]
        assert classify_trade_exit(trade, events) == "valid_entry_bad_exit_policy"

    def test_losing_news_breakout_force_closed_is_bad_entry(self):
        """Losing news_breakout force-closed → bad_entry (not profitable)."""
        trade = {
            "id": 9,
            "symbol": "AMD",
            "setup_type": "news_breakout",
            "direction": "LONG",
            "entry_price": 150.0,
            "exit_price": 148.0,
            "pnl": -20.0,
            "status": "closed",
        }
        events = [
            {"event_type": "setup_exit_force_close"},
        ]
        assert classify_trade_exit(trade, events) == "bad_entry"

    def test_profitable_momentum_fade_force_closed_is_bad_entry(self):
        """Profitable momentum_fade (fast tactical) force-closed → bad_entry.
        Only thesis-development setups qualify for valid_entry_bad_exit_policy."""
        trade = {
            "id": 10,
            "symbol": "GME",
            "setup_type": "momentum_fade",
            "direction": "SHORT",
            "entry_price": 50.0,
            "exit_price": 48.0,
            "pnl": 20.0,
            "status": "closed",
        }
        events = [
            {"event_type": "setup_exit_force_close"},
        ]
        assert classify_trade_exit(trade, events) == "bad_entry"


class TestBadEntry:
    """Losing trade with no special events → bad_entry."""

    def test_losing_trade_no_events(self):
        trade = {
            "id": 11,
            "symbol": "PLTR",
            "setup_type": "gap_and_go",
            "direction": "LONG",
            "entry_price": 25.0,
            "exit_price": 23.0,
            "pnl": -20.0,
            "status": "closed",
        }
        events = [
            {"event_type": "entry_executed"},
            {"event_type": "stop_triggered"},
        ]
        assert classify_trade_exit(trade, events) == "bad_entry"

    def test_empty_events(self):
        trade = {
            "id": 12,
            "symbol": "SPY",
            "setup_type": "orb",
            "direction": "LONG",
            "entry_price": 450.0,
            "exit_price": 449.0,
            "pnl": -10.0,
            "status": "closed",
        }
        events = []
        assert classify_trade_exit(trade, events) == "bad_entry"

    def test_unknown_setup_type(self):
        trade = {
            "id": 13,
            "symbol": "XYZ",
            "setup_type": "unknown_setup",
            "direction": "LONG",
            "entry_price": 100.0,
            "exit_price": 95.0,
            "pnl": -50.0,
            "status": "closed",
        }
        events = [
            {"event_type": "setup_exit_force_close"},
        ]
        # Unknown setup is not thesis-development, so even with force_close → bad_entry
        assert classify_trade_exit(trade, events) == "bad_entry"


class TestPriorityOrder:
    """Thesis invalidation takes precedence over other categories."""

    def test_thesis_invalidation_over_missing_metadata(self):
        """If both thesis invalidation and missing metadata events exist,
        thesis invalidation wins (priority 1 > priority 2)."""
        trade = {
            "id": 14,
            "symbol": "AMD",
            "setup_type": "news_breakout",
            "direction": "LONG",
            "entry_price": 150.0,
            "exit_price": 145.0,
            "pnl": -50.0,
            "status": "closed",
        }
        events = [
            {
                "event_type": "setup_exit_revalidation_failed",
                "reason": "missing criteria",
            },
            {"event_type": "setup_exit_thesis_invalidated"},
        ]
        assert classify_trade_exit(trade, events) == "valid_exit_thesis_invalidated"

    def test_thesis_invalidation_over_force_close(self):
        """Thesis invalidation takes precedence over force-close classification."""
        trade = {
            "id": 15,
            "symbol": "NVDA",
            "setup_type": "news_breakout",
            "direction": "LONG",
            "entry_price": 100.0,
            "exit_price": 105.0,
            "pnl": 50.0,
            "status": "closed",
        }
        events = [
            {"event_type": "setup_exit_force_close"},
            {"event_type": "setup_exit_thesis_invalidated"},
        ]
        assert classify_trade_exit(trade, events) == "valid_exit_thesis_invalidated"

    def test_missing_metadata_over_bad_exit_policy(self):
        """Missing metadata takes precedence over bad exit policy (priority 2 > 3)."""
        trade = {
            "id": 16,
            "symbol": "AMD",
            "setup_type": "news_breakout",
            "direction": "LONG",
            "entry_price": 150.0,
            "exit_price": 155.0,
            "pnl": 50.0,
            "status": "closed",
        }
        events = [
            {
                "event_type": "setup_exit_revalidation_failed",
                "reason": "Extension denied: missing criteria",
            },
            {"event_type": "setup_exit_force_close"},
        ]
        assert classify_trade_exit(trade, events) == "forced_exit_missing_metadata"


class TestCategoryCompleteness:
    """Verify all categories are valid and classification always returns one."""

    def test_all_categories_are_strings(self):
        for cat in CASE_MEMORY_EXIT_CATEGORIES:
            assert isinstance(cat, str)

    def test_exactly_four_categories(self):
        assert len(CASE_MEMORY_EXIT_CATEGORIES) == 4

    def test_result_always_in_categories(self):
        """Any classification result must be in CASE_MEMORY_EXIT_CATEGORIES."""
        trades_and_events = [
            # Thesis invalidation
            (
                {"id": 1, "setup_type": "news_breakout", "direction": "LONG",
                 "entry_price": 100.0, "exit_price": 95.0},
                [{"event_type": "setup_exit_thesis_invalidated"}],
            ),
            # Missing metadata
            (
                {"id": 2, "setup_type": "news_breakout", "direction": "LONG",
                 "entry_price": 100.0, "exit_price": 102.0},
                [{"event_type": "setup_exit_revalidation_failed",
                  "reason": "missing criteria"}],
            ),
            # Bad exit policy
            (
                {"id": 3, "setup_type": "news_breakout", "direction": "LONG",
                 "entry_price": 100.0, "exit_price": 105.0},
                [{"event_type": "setup_exit_force_close"}],
            ),
            # Bad entry
            (
                {"id": 4, "setup_type": "orb", "direction": "LONG",
                 "entry_price": 100.0, "exit_price": 95.0},
                [],
            ),
        ]
        for trade, events in trades_and_events:
            result = classify_trade_exit(trade, events)
            assert result in CASE_MEMORY_EXIT_CATEGORIES
