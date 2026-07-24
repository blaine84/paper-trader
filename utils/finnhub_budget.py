"""Cycle-scoped Finnhub API budget counter.

Tracks API calls per cycle to prevent exceeding the configured budget.
Thread-safe via threading.Lock for concurrent analyst symbol processing.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger("cycle_coordinator")


class CycleFinnhubBudget:
    """Tracks Finnhub API calls within a single market cycle.

    Args:
        budget: Maximum allowed calls this cycle (from CYCLE_FINNHUB_BUDGET).
    """

    def __init__(self, budget: int) -> None:
        self._budget = budget
        self._used = 0
        self._lock = threading.Lock()

    def increment(self, count: int = 1) -> bool:
        """Record API call(s). Returns True if within budget, False if budget exhausted.

        Args:
            count: Number of API calls to record (default 1).

        Returns:
            True if the calls were recorded successfully (within budget).
            False if the budget would be exceeded (calls NOT recorded).
        """
        with self._lock:
            if self._used + count > self._budget:
                logger.warning(
                    "Finnhub budget exhausted: used=%d, budget=%d, requested=%d",
                    self._used,
                    self._budget,
                    count,
                )
                return False
            self._used += count
            return True

    def remaining(self) -> int:
        """Return the number of API calls remaining in the budget."""
        with self._lock:
            return max(0, self._budget - self._used)

    def is_exhausted(self) -> bool:
        """Return True if the budget has been fully consumed."""
        with self._lock:
            return self._used >= self._budget

    @property
    def used(self) -> int:
        """Number of API calls consumed so far."""
        with self._lock:
            return self._used

    @property
    def budget(self) -> int:
        """Total budget for this cycle."""
        return self._budget

    def reset(self) -> None:
        """Reset the counter for a new cycle."""
        with self._lock:
            self._used = 0
