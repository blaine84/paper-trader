"""Request coalescing for market data provider calls.

Prevents duplicate in-flight provider calls for the same symbol and data type.
When multiple consumers request the same key concurrently, the coalescer ensures
only one network call is made; all waiters receive the same result.

Thread safety: Internal state is protected by a threading.Lock held only for
O(1) dictionary bookkeeping, never during the provider call itself.

Requirements: 5.2
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from utils.market_data_reliability.snapshot import CoalesceKey, Snapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal in-flight entry
# ---------------------------------------------------------------------------


@dataclass
class _InFlightEntry:
    """Tracks a single in-flight provider call.

    The event is set when the fetch completes (success or failure).
    Waiters block on event.wait() with a timeout.
    """

    event: threading.Event = field(default_factory=threading.Event)
    result: Optional[Snapshot] = None
    error: Optional[Exception] = None


# ---------------------------------------------------------------------------
# RequestCoalescer
# ---------------------------------------------------------------------------


class RequestCoalescer:
    """Coalesces concurrent provider requests for the same key.

    Only one network call is made per key at a time. Subsequent callers
    wait on a threading.Event until the first caller completes. On timeout,
    waiters receive a Snapshot with provider_status="timeout".

    Args:
        timeout_seconds: Maximum time (seconds) to wait for an in-flight
            request to complete. Default is provider_timeout + 2s = 12.0s.
    """

    def __init__(self, timeout_seconds: float = 12.0) -> None:
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._in_flight: dict[CoalesceKey, _InFlightEntry] = {}

    def get_or_start(
        self, key: CoalesceKey, fetch_fn: Callable[[], Snapshot]
    ) -> Snapshot:
        """Get a snapshot for key, coalescing with any in-flight request.

        If another thread is already fetching for this key, this call waits
        on the in-flight event (up to timeout_seconds). Otherwise, this call
        becomes the fetcher.

        Args:
            key: The coalesce key (symbol, data_type, provider_policy).
            fetch_fn: A callable that performs the provider call and returns
                a Snapshot. Called outside the lock.

        Returns:
            The fetched Snapshot, or a timeout Snapshot if the wait expires.
        """
        is_fetcher = False
        with self._lock:
            entry = self._in_flight.get(key)
            if entry is not None:
                # Another thread is already fetching — we become a waiter
                logger.debug(
                    "Coalescing request for %s/%s/%s — waiting on in-flight.",
                    key.symbol, key.data_type, key.provider_policy,
                )
            else:
                # We are the first — create entry and become the fetcher
                entry = _InFlightEntry()
                self._in_flight[key] = entry
                is_fetcher = True

        # Lock is released here — perform fetch or wait outside the lock
        if is_fetcher:
            return self._do_fetch(key, entry, fetch_fn)
        else:
            return self._wait_for_result(key, entry)

    def cancel(self, key: CoalesceKey) -> None:
        """Cancel an in-flight request, unblocking any waiters.

        Waiters will receive None from the entry (treated as if an error
        occurred), allowing them to retry on a subsequent call.

        Args:
            key: The coalesce key to cancel.
        """
        with self._lock:
            entry = self._in_flight.pop(key, None)

        if entry is not None:
            # Set event so waiters unblock
            entry.error = _CancelledError(f"Request cancelled for {key}")
            entry.event.set()
            logger.info(
                "Cancelled in-flight request for %s/%s/%s.",
                key.symbol, key.data_type, key.provider_policy,
            )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _do_fetch(
        self,
        key: CoalesceKey,
        entry: _InFlightEntry,
        fetch_fn: Callable[[], Snapshot],
    ) -> Snapshot:
        """Execute the fetch function and populate the entry result.

        Called by the first thread to request a given key. The lock is NOT
        held during the fetch_fn call.
        """
        try:
            result = fetch_fn()
            entry.result = result
            entry.event.set()
            return result
        except Exception as exc:
            logger.warning(
                "Fetch failed for %s/%s/%s: %s",
                key.symbol, key.data_type, key.provider_policy, exc,
            )
            entry.error = exc
            entry.event.set()
            # Remove the entry so subsequent callers can retry
            with self._lock:
                self._in_flight.pop(key, None)
            raise
        finally:
            # Clean up entry on success (waiters already have their result)
            if entry.result is not None:
                with self._lock:
                    # Only remove if it's still our entry (not replaced)
                    if self._in_flight.get(key) is entry:
                        del self._in_flight[key]

    def _wait_for_result(
        self, key: CoalesceKey, entry: _InFlightEntry
    ) -> Snapshot:
        """Wait on an in-flight entry's event with timeout.

        Returns the fetched Snapshot if available, or a timeout Snapshot
        if the wait expires.
        """
        completed = entry.event.wait(timeout=self._timeout_seconds)

        if not completed:
            # Timeout — produce a timeout Snapshot
            logger.warning(
                "Coalesced wait timed out after %.1fs for %s/%s/%s.",
                self._timeout_seconds, key.symbol, key.data_type, key.provider_policy,
            )
            return self._make_timeout_snapshot(key)

        # Event was set — check for error or result
        if entry.error is not None:
            logger.warning(
                "Coalesced request for %s/%s/%s completed with error: %s",
                key.symbol, key.data_type, key.provider_policy, entry.error,
            )
            return self._make_timeout_snapshot(key)

        if entry.result is not None:
            return entry.result

        # Shouldn't reach here, but defensive
        logger.error(
            "Coalesced entry for %s/%s/%s has no result and no error.",
            key.symbol, key.data_type, key.provider_policy,
        )
        return self._make_timeout_snapshot(key)

    def _make_timeout_snapshot(self, key: CoalesceKey) -> Snapshot:
        """Create a Snapshot representing a timeout/failed coalesced request.

        The snapshot has:
            provider_status = "timeout"
            trust_state = "untrusted"
            freshness_state = "unavailable"
            degradation_reasons = ("network_timeout",)
            All price fields = None
        """
        now = datetime.now(timezone.utc)
        return Snapshot(
            symbol=key.symbol,
            data_type=key.data_type,
            requested_at=now,
            provider="unknown",
            provider_status="timeout",
            market_session="open",
            last_price=None,
            bid=None,
            ask=None,
            previous_close=None,
            open=None,
            high=None,
            low=None,
            volume=None,
            fetched_at=now,
            source_timestamp=None,
            age_seconds=0.0,
            freshness_state="unavailable",
            trust_state="untrusted",
            degradation_reasons=("network_timeout",),
            raw_provider_latency_ms=None,
            fallback_primary_provider=None,
        )


class _CancelledError(Exception):
    """Internal exception for cancelled coalesced requests."""

    pass
