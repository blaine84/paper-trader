"""Unit tests for TelemetryCollector."""
from __future__ import annotations

import logging
import threading

import pytest

from utils.market_data_reliability.telemetry import TelemetryCollector


class TestTelemetryCollectorBasic:
    """Tests for basic TelemetryCollector functionality."""

    def test_initial_state_all_zeros(self):
        """Fresh collector should have all-zero summary."""
        tc = TelemetryCollector()
        summary = tc.get_cycle_summary()
        assert summary["provider_calls_success"] == 0
        assert summary["provider_calls_failure"] == 0
        assert summary["cache_hits"] == 0
        assert summary["fallback_usage"] == 0
        assert summary["stale_snapshots"] == 0
        assert summary["unavailable_snapshots"] == 0
        assert summary["fail_closed_decisions"] == 0
        assert summary["fail_closed_details"] == []
        assert summary["fallback_details"] == []

    def test_record_provider_call_success(self):
        """Successful provider calls increment success counter."""
        tc = TelemetryCollector()
        tc.record_provider_call("finnhub", "quote", True)
        tc.record_provider_call("yfinance", "candle", True)
        summary = tc.get_cycle_summary()
        assert summary["provider_calls_success"] == 2
        assert summary["provider_calls_failure"] == 0

    def test_record_provider_call_failure(self):
        """Failed provider calls increment failure counter."""
        tc = TelemetryCollector()
        tc.record_provider_call("finnhub", "quote", False)
        summary = tc.get_cycle_summary()
        assert summary["provider_calls_success"] == 0
        assert summary["provider_calls_failure"] == 1

    def test_record_cache_hit(self):
        """Cache hits are counted."""
        tc = TelemetryCollector()
        tc.record_cache_hit("quote")
        tc.record_cache_hit("atr")
        tc.record_cache_hit("quote")
        summary = tc.get_cycle_summary()
        assert summary["cache_hits"] == 3

    def test_record_fallback_usage(self):
        """Fallback usage is counted with details."""
        tc = TelemetryCollector()
        tc.record_fallback_usage("finnhub", "yfinance", "quote")
        summary = tc.get_cycle_summary()
        assert summary["fallback_usage"] == 1
        assert summary["fallback_details"] == [
            {"primary": "finnhub", "fallback": "yfinance", "data_type": "quote"}
        ]

    def test_record_fail_closed(self):
        """Fail-closed decisions are counted with details."""
        tc = TelemetryCollector()
        tc.record_fail_closed("cand-123", "market_data_unavailable")
        tc.record_fail_closed("cand-456", "quote_stale")
        summary = tc.get_cycle_summary()
        assert summary["fail_closed_decisions"] == 2
        assert len(summary["fail_closed_details"]) == 2
        assert summary["fail_closed_details"][0] == {
            "candidate_id": "cand-123",
            "reason": "market_data_unavailable",
        }
        assert summary["fail_closed_details"][1] == {
            "candidate_id": "cand-456",
            "reason": "quote_stale",
        }

    def test_record_stale_snapshot(self):
        """Stale snapshots are counted."""
        tc = TelemetryCollector()
        tc.record_stale_snapshot("AAPL", "quote")
        summary = tc.get_cycle_summary()
        assert summary["stale_snapshots"] == 1

    def test_record_unavailable_snapshot(self):
        """Unavailable snapshots are counted."""
        tc = TelemetryCollector()
        tc.record_unavailable_snapshot("TSLA", "candle")
        summary = tc.get_cycle_summary()
        assert summary["unavailable_snapshots"] == 1


class TestTelemetryCollectorReset:
    """Tests for reset behavior."""

    def test_reset_zeros_all_counters(self):
        """Reset clears all counters and lists."""
        tc = TelemetryCollector()
        tc.record_provider_call("finnhub", "quote", True)
        tc.record_provider_call("finnhub", "quote", False)
        tc.record_cache_hit("quote")
        tc.record_fallback_usage("finnhub", "yfinance", "quote")
        tc.record_fail_closed("cand-1", "reason")
        tc.record_stale_snapshot("AAPL", "quote")
        tc.record_unavailable_snapshot("TSLA", "candle")

        tc.reset()
        summary = tc.get_cycle_summary()

        assert summary["provider_calls_success"] == 0
        assert summary["provider_calls_failure"] == 0
        assert summary["cache_hits"] == 0
        assert summary["fallback_usage"] == 0
        assert summary["stale_snapshots"] == 0
        assert summary["unavailable_snapshots"] == 0
        assert summary["fail_closed_decisions"] == 0
        assert summary["fail_closed_details"] == []
        assert summary["fallback_details"] == []

    def test_reset_allows_new_accumulation(self):
        """After reset, new calls accumulate from zero."""
        tc = TelemetryCollector()
        tc.record_provider_call("finnhub", "quote", True)
        tc.reset()
        tc.record_provider_call("finnhub", "quote", True)
        summary = tc.get_cycle_summary()
        assert summary["provider_calls_success"] == 1


class TestTelemetryCollectorFailOpen:
    """Tests that telemetry never crashes — fail-open behavior."""

    def _make_broken_collector(self):
        """Create a collector whose internal counter raises on access."""
        tc = TelemetryCollector()
        # Replace internal counter with a property that raises
        # We use a simpler approach: replace _lock with a broken mock
        from unittest.mock import MagicMock
        broken_lock = MagicMock()
        broken_lock.__enter__ = MagicMock(side_effect=RuntimeError("lock broken"))
        broken_lock.__exit__ = MagicMock(return_value=False)
        tc._lock = broken_lock
        return tc

    def test_get_cycle_summary_returns_dict_on_internal_error(self, caplog):
        """get_cycle_summary returns safe default dict if internals break."""
        tc = self._make_broken_collector()

        with caplog.at_level(logging.ERROR):
            summary = tc.get_cycle_summary()

        # Should return a safe empty dict, not raise
        assert summary["provider_calls_success"] == 0
        assert summary["fail_closed_details"] == []
        assert "Telemetry error" in caplog.text

    def test_record_provider_call_does_not_raise_on_error(self, caplog):
        """record_provider_call logs but does not raise on internal error."""
        tc = self._make_broken_collector()

        with caplog.at_level(logging.ERROR):
            # Should not raise
            tc.record_provider_call("finnhub", "quote", True)

        assert "Telemetry error" in caplog.text

    def test_record_cache_hit_does_not_raise_on_error(self, caplog):
        """record_cache_hit logs but does not raise on internal error."""
        tc = self._make_broken_collector()

        with caplog.at_level(logging.ERROR):
            tc.record_cache_hit("quote")

        assert "Telemetry error" in caplog.text

    def test_record_fallback_usage_does_not_raise_on_error(self, caplog):
        """record_fallback_usage logs but does not raise on internal error."""
        tc = self._make_broken_collector()

        with caplog.at_level(logging.ERROR):
            tc.record_fallback_usage("finnhub", "yfinance", "quote")

        assert "Telemetry error" in caplog.text

    def test_record_fail_closed_does_not_raise_on_error(self, caplog):
        """record_fail_closed logs but does not raise on internal error."""
        tc = self._make_broken_collector()

        with caplog.at_level(logging.ERROR):
            tc.record_fail_closed("cand-1", "reason")

        assert "Telemetry error" in caplog.text

    def test_reset_does_not_raise_on_error(self, caplog):
        """reset logs but does not raise on internal error."""
        tc = self._make_broken_collector()

        with caplog.at_level(logging.ERROR):
            tc.reset()

        assert "Telemetry error" in caplog.text


class TestTelemetryCollectorThreadSafety:
    """Tests for thread-safe concurrent access."""

    def test_concurrent_record_calls(self):
        """Multiple threads recording concurrently produces consistent results."""
        tc = TelemetryCollector()
        num_threads = 10
        calls_per_thread = 100
        barrier = threading.Barrier(num_threads)

        def worker():
            barrier.wait()
            for _ in range(calls_per_thread):
                tc.record_provider_call("finnhub", "quote", True)
                tc.record_cache_hit("quote")

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        summary = tc.get_cycle_summary()
        expected = num_threads * calls_per_thread
        assert summary["provider_calls_success"] == expected
        assert summary["cache_hits"] == expected


class TestTelemetryCollectorSummaryStructure:
    """Tests for the structure of get_cycle_summary return value."""

    def test_summary_has_all_required_keys(self):
        """Summary dict contains all documented keys."""
        tc = TelemetryCollector()
        summary = tc.get_cycle_summary()
        required_keys = {
            "provider_calls_success",
            "provider_calls_failure",
            "cache_hits",
            "fallback_usage",
            "stale_snapshots",
            "unavailable_snapshots",
            "fail_closed_decisions",
            "fail_closed_details",
            "fallback_details",
        }
        assert set(summary.keys()) == required_keys

    def test_detail_lists_are_copies(self):
        """Detail lists in summary should be copies, not references to internals."""
        tc = TelemetryCollector()
        tc.record_fail_closed("cand-1", "reason")
        summary = tc.get_cycle_summary()

        # Mutating the returned list should not affect internal state
        summary["fail_closed_details"].append({"candidate_id": "x", "reason": "y"})
        new_summary = tc.get_cycle_summary()
        assert len(new_summary["fail_closed_details"]) == 1
