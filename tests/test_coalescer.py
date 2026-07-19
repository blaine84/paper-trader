"""Unit tests for RequestCoalescer.

Tests the request coalescing behavior including:
- Basic fetch (first caller executes fetch_fn)
- Coalescing (second caller waits on first)
- Timeout behavior (waiter gets timeout Snapshot)
- Exception cleanup (entry removed, allows retry)
- Cancel (unblocks waiters with timeout Snapshot)
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from utils.market_data_reliability.coalescer import RequestCoalescer
from utils.market_data_reliability.snapshot import CoalesceKey, Snapshot


def _make_snapshot(symbol: str = "AAPL") -> Snapshot:
    """Helper to create a valid Snapshot for testing."""
    now = datetime.now(timezone.utc)
    return Snapshot(
        symbol=symbol,
        data_type="quote",
        requested_at=now,
        provider="finnhub",
        provider_status="success",
        market_session="open",
        last_price=Decimal("150.00"),
        bid=Decimal("149.99"),
        ask=Decimal("150.01"),
        previous_close=Decimal("149.50"),
        open=Decimal("149.75"),
        high=Decimal("150.50"),
        low=Decimal("149.00"),
        volume=1000000,
        fetched_at=now,
        source_timestamp=now,
        age_seconds=0.5,
        freshness_state="fresh",
        trust_state="trusted",
        degradation_reasons=(),
        raw_provider_latency_ms=45.0,
        fallback_primary_provider=None,
    )


class TestRequestCoalescerBasicFetch:
    """Test basic fetch behavior when no in-flight request exists."""

    def test_first_caller_executes_fetch_fn(self):
        coalescer = RequestCoalescer(timeout_seconds=5.0)
        key = CoalesceKey(symbol="AAPL", data_type="quote", provider_policy="primary")

        called = []

        def fetch_fn():
            called.append(True)
            return _make_snapshot()

        result = coalescer.get_or_start(key, fetch_fn)
        assert result.symbol == "AAPL"
        assert result.provider_status == "success"
        assert len(called) == 1

    def test_second_call_after_first_completes_fetches_again(self):
        """After first request completes, a new request starts a fresh fetch."""
        coalescer = RequestCoalescer(timeout_seconds=5.0)
        key = CoalesceKey(symbol="AAPL", data_type="quote", provider_policy="primary")

        call_count = []

        def fetch_fn():
            call_count.append(True)
            return _make_snapshot()

        coalescer.get_or_start(key, fetch_fn)
        coalescer.get_or_start(key, fetch_fn)
        assert len(call_count) == 2


class TestRequestCoalescerCoalescing:
    """Test that concurrent requests for the same key are coalesced."""

    def test_concurrent_requests_coalesce(self):
        coalescer = RequestCoalescer(timeout_seconds=5.0)
        key = CoalesceKey(symbol="AAPL", data_type="quote", provider_policy="primary")

        fetch_count = []
        results = []
        fetcher_started = threading.Event()

        def slow_fetch():
            fetch_count.append(True)
            fetcher_started.set()
            time.sleep(0.3)
            return _make_snapshot()

        def fetcher():
            r = coalescer.get_or_start(key, slow_fetch)
            results.append(r)

        def waiter():
            # This fetch_fn should never be called because request is coalesced
            r = coalescer.get_or_start(key, lambda: _make_snapshot("WRONG"))
            results.append(r)

        t1 = threading.Thread(target=fetcher)
        t1.start()
        fetcher_started.wait(timeout=2.0)  # Wait until fetch has started

        t2 = threading.Thread(target=waiter)
        t2.start()

        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert len(fetch_count) == 1, "Only one fetch should have been made"
        assert len(results) == 2
        assert all(r.symbol == "AAPL" for r in results)
        assert all(r.provider_status == "success" for r in results)


class TestRequestCoalescerTimeout:
    """Test timeout behavior when in-flight request takes too long."""

    def test_waiter_receives_timeout_snapshot(self):
        coalescer = RequestCoalescer(timeout_seconds=0.2)
        key = CoalesceKey(symbol="AAPL", data_type="quote", provider_policy="primary")

        waiter_result = []
        fetcher_started = threading.Event()
        fetcher_can_finish = threading.Event()

        def controlled_fetch():
            fetcher_started.set()
            fetcher_can_finish.wait(timeout=5.0)
            return _make_snapshot()

        def fetcher():
            coalescer.get_or_start(key, controlled_fetch)

        def waiter():
            r = coalescer.get_or_start(key, lambda: None)
            waiter_result.append(r)

        t1 = threading.Thread(target=fetcher, daemon=True)
        t1.start()
        fetcher_started.wait(timeout=2.0)

        t2 = threading.Thread(target=waiter)
        t2.start()
        t2.join(timeout=3.0)

        # Let the fetcher finish to avoid dangling threads
        fetcher_can_finish.set()
        t1.join(timeout=2.0)

        # The waiter should have received a timeout snapshot
        assert len(waiter_result) == 1
        timeout_snap = waiter_result[0]
        assert timeout_snap.provider_status == "timeout"
        assert timeout_snap.trust_state == "untrusted"
        assert timeout_snap.freshness_state == "unavailable"
        assert "network_timeout" in timeout_snap.degradation_reasons


class TestRequestCoalescerExceptionCleanup:
    """Test that exceptions in fetch_fn clean up the in-flight entry."""

    def test_exception_removes_entry_and_allows_retry(self):
        coalescer = RequestCoalescer(timeout_seconds=5.0)
        key = CoalesceKey(symbol="AAPL", data_type="quote", provider_policy="primary")

        # First call raises
        def failing_fetch():
            raise RuntimeError("Provider down")

        with pytest.raises(RuntimeError, match="Provider down"):
            coalescer.get_or_start(key, failing_fetch)

        # Second call should be able to fetch (entry was cleaned up)
        result = coalescer.get_or_start(key, _make_snapshot)
        assert result.symbol == "AAPL"
        assert result.provider_status == "success"

    def test_waiter_gets_timeout_snapshot_on_fetch_error(self):
        coalescer = RequestCoalescer(timeout_seconds=5.0)
        key = CoalesceKey(symbol="AAPL", data_type="quote", provider_policy="primary")

        results = []
        fetcher_started = threading.Event()

        def slow_failing_fetch():
            fetcher_started.set()
            time.sleep(0.2)
            raise RuntimeError("Network error")

        def fetcher():
            try:
                coalescer.get_or_start(key, slow_failing_fetch)
            except RuntimeError:
                results.append(("fetcher", "raised"))

        def waiter():
            r = coalescer.get_or_start(key, lambda: None)
            results.append(("waiter", r))

        t1 = threading.Thread(target=fetcher)
        t1.start()
        fetcher_started.wait(timeout=2.0)

        t2 = threading.Thread(target=waiter)
        t2.start()

        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        # Fetcher raised, waiter gets timeout snapshot (error path)
        waiter_results = [(label, r) for label, r in results if label == "waiter"]
        assert len(waiter_results) == 1
        snap = waiter_results[0][1]
        assert snap.provider_status == "timeout"
        assert snap.trust_state == "untrusted"


class TestRequestCoalescerCancel:
    """Test cancel behavior."""

    def test_cancel_unblocks_waiters(self):
        coalescer = RequestCoalescer(timeout_seconds=10.0)
        key = CoalesceKey(symbol="TSLA", data_type="quote", provider_policy="primary")

        waiter_result = []
        fetcher_started = threading.Event()
        fetcher_can_finish = threading.Event()

        def controlled_fetch():
            fetcher_started.set()
            fetcher_can_finish.wait(timeout=10.0)
            return _make_snapshot("TSLA")

        def fetcher():
            try:
                coalescer.get_or_start(key, controlled_fetch)
            except Exception:
                pass

        def waiter():
            r = coalescer.get_or_start(key, lambda: None)
            waiter_result.append(r)

        t1 = threading.Thread(target=fetcher, daemon=True)
        t1.start()
        fetcher_started.wait(timeout=2.0)

        t2 = threading.Thread(target=waiter)
        t2.start()
        time.sleep(0.05)  # Let waiter start waiting

        # Cancel the in-flight request
        coalescer.cancel(key)

        t2.join(timeout=3.0)

        # Let the fetcher finish cleanly
        fetcher_can_finish.set()
        t1.join(timeout=2.0)

        # Waiter should have gotten a timeout snapshot
        assert len(waiter_result) == 1
        snap = waiter_result[0]
        assert snap.provider_status == "timeout"
        assert snap.trust_state == "untrusted"
        assert "network_timeout" in snap.degradation_reasons

    def test_cancel_nonexistent_key_is_noop(self):
        coalescer = RequestCoalescer(timeout_seconds=5.0)
        key = CoalesceKey(symbol="NOPE", data_type="quote", provider_policy="primary")

        # Should not raise
        coalescer.cancel(key)


class TestRequestCoalescerTimeoutSnapshot:
    """Test the timeout snapshot structure."""

    def test_timeout_snapshot_has_correct_fields(self):
        coalescer = RequestCoalescer(timeout_seconds=0.05)
        key = CoalesceKey(symbol="GOOG", data_type="candle", provider_policy="fallback")

        fetcher_started = threading.Event()
        fetcher_can_finish = threading.Event()

        def controlled_fetch():
            fetcher_started.set()
            fetcher_can_finish.wait(timeout=5.0)
            return _make_snapshot("GOOG")

        def fetcher():
            coalescer.get_or_start(key, controlled_fetch)

        t1 = threading.Thread(target=fetcher, daemon=True)
        t1.start()
        fetcher_started.wait(timeout=2.0)

        # Second caller times out
        result = coalescer.get_or_start(key, lambda: None)

        # Clean up
        fetcher_can_finish.set()
        t1.join(timeout=2.0)

        assert result.symbol == "GOOG"
        assert result.data_type == "candle"
        assert result.provider_status == "timeout"
        assert result.trust_state == "untrusted"
        assert result.freshness_state == "unavailable"
        assert result.degradation_reasons == ("network_timeout",)
        assert result.last_price is None
        assert result.bid is None
        assert result.ask is None
        assert result.previous_close is None
        assert result.open is None
        assert result.high is None
        assert result.low is None
        assert result.volume is None
