"""BackoffTracker for the Market Data Reliability Layer.

Tracks per-(provider, data_type) backoff state so that a failure for one
data type does not block requests for unrelated data types from the same
provider.

Thread-safe: all dict access is protected by a threading.Lock.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class BackoffTracker:
    """Tracks provider backoff state scoped by (provider, data_type).

    Backoff durations are configurable via the backoff_durations dict
    (loaded from ReliabilityConfig). Known failure types:
        - "rate_limit": 60s default
        - "network_error": 30s default
        - "empty_response": 15s default

    Unknown failure types use the maximum configured duration as a safe
    default to avoid hammering a provider with an unrecognized error.
    """

    def __init__(self, backoff_durations: dict[str, float]) -> None:
        self._durations = dict(backoff_durations)
        self._max_duration = max(self._durations.values()) if self._durations else 60.0
        # Internal state: (provider, data_type) -> backoff_until monotonic time
        self._backoff_until: dict[tuple[str, str], float] = {}
        # Track last failure type per (provider, data_type) for diagnostics
        self._failure_types: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()

    def record_failure(self, provider: str, data_type: str, failure_type: str) -> None:
        """Record a provider failure and enter backoff for (provider, data_type).

        The backoff duration is determined by failure_type. Unknown failure
        types use the maximum configured duration as a safe default.
        """
        duration = self._durations.get(failure_type, self._max_duration)
        backoff_until = time.monotonic() + duration

        with self._lock:
            self._backoff_until[(provider, data_type)] = backoff_until
            self._failure_types[(provider, data_type)] = failure_type

        logger.warning(
            "Backoff recorded: provider=%s data_type=%s failure_type=%s duration=%.1fs",
            provider, data_type, failure_type, duration,
        )

    def is_in_backoff(self, provider: str, data_type: str) -> bool:
        """Check if (provider, data_type) is currently in backoff.

        Returns False if no backoff has been recorded or if the backoff
        period has expired.
        """
        with self._lock:
            backoff_until = self._backoff_until.get((provider, data_type))

        if backoff_until is None:
            return False

        return time.monotonic() < backoff_until

    def record_success(self, provider: str, data_type: str) -> None:
        """Clear any backoff for (provider, data_type) after a successful call."""
        with self._lock:
            removed = self._backoff_until.pop((provider, data_type), None)
            self._failure_types.pop((provider, data_type), None)

        if removed is not None:
            logger.info(
                "Backoff cleared: provider=%s data_type=%s",
                provider, data_type,
            )

    def get_failure_type(self, provider: str, data_type: str) -> str | None:
        """Get the last recorded failure type for (provider, data_type).

        Returns None if no failure recorded or backoff has expired.
        """
        if not self.is_in_backoff(provider, data_type):
            return None
        with self._lock:
            return self._failure_types.get((provider, data_type))
