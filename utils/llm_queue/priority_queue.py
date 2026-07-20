"""Thread-safe priority queue for pending LLM requests.

Uses heapq with (priority, created_at, record) tuples for ordering.
Lower priority value = higher priority. FIFO within same priority
via created_at tiebreaker.

Protected by threading.Lock wrapping all mutations and threading.Condition
for wait/notify on get() blocking.
"""

from __future__ import annotations

import heapq
import logging
import threading
from datetime import datetime
from typing import Optional

from utils.llm_queue.config import QueueConfig
from utils.llm_queue.models import RequestRecord

logger = logging.getLogger(__name__)


class PriorityQueue:
    """Thread-safe priority queue for pending RequestRecords.

    Ordering: (priority, created_at, record) — lower priority value dispatched
    first. Within same priority, earlier created_at dispatched first (FIFO).
    """

    def __init__(self, config: QueueConfig) -> None:
        self._config = config
        self._heap: list[tuple[int, datetime, RequestRecord]] = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def put(self, record: RequestRecord) -> None:
        """Add a request to the queue and notify waiting consumers."""
        with self._condition:
            heapq.heappush(self._heap, (record.priority, record.created_at, record))
            logger.debug(
                "Queue put: request_id=%s class=%s priority=%d depth=%d",
                record.request_id, record.request_class, record.priority, len(self._heap),
            )
            self._condition.notify()

    def get(self, timeout: Optional[float] = None) -> Optional[RequestRecord]:
        """Pop the highest-priority request, blocking until available or timeout.

        Returns None if timeout expires without an item becoming available.
        """
        with self._condition:
            while not self._heap:
                if not self._condition.wait(timeout=timeout):
                    # Timed out waiting for an item
                    return None
                # If timeout was specified and we were woken, check if there's
                # actually an item (spurious wakeup protection)
                if not self._heap and timeout is not None:
                    return None
            _, _, record = heapq.heappop(self._heap)
            logger.debug(
                "Queue get: request_id=%s class=%s priority=%d depth=%d",
                record.request_id, record.request_class, record.priority, len(self._heap),
            )
            return record

    def peek_lowest_priority(self) -> Optional[RequestRecord]:
        """Return the record with the HIGHEST priority value (lowest urgency).

        This is the eviction candidate — the least urgent item in the queue.
        Does not remove the item. Returns None if queue is empty.

        Since heapq is a min-heap by priority, the max priority value is
        somewhere in the heap (not necessarily at the end). We iterate to find it.
        """
        with self._lock:
            if not self._heap:
                return None
            # Find the entry with the maximum priority value (lowest urgency)
            worst = self._heap[0]
            for entry in self._heap:
                if entry[0] > worst[0]:
                    worst = entry
                elif entry[0] == worst[0] and entry[1] > worst[1]:
                    # Same priority, later created_at = less urgent (added later)
                    worst = entry
            return worst[2]

    def evict_lowest(self) -> Optional[RequestRecord]:
        """Remove and return the lowest-priority (highest priority value) record.

        Used by admission controller to make room for execution_critical requests.
        Returns None if queue is empty.

        Finds the max priority value entry, removes it from the heap, and
        re-heapifies.
        """
        with self._lock:
            if not self._heap:
                return None
            # Find the index of the entry with the maximum priority value
            worst_idx = 0
            for i, entry in enumerate(self._heap):
                if entry[0] > self._heap[worst_idx][0]:
                    worst_idx = i
                elif entry[0] == self._heap[worst_idx][0] and entry[1] > self._heap[worst_idx][1]:
                    # Same priority, later created_at = less urgent
                    worst_idx = i

            evicted_entry = self._heap[worst_idx]
            # Remove by swapping with last element and re-heapifying
            self._heap[worst_idx] = self._heap[-1]
            self._heap.pop()
            if self._heap and worst_idx < len(self._heap):
                heapq.heapify(self._heap)

            record = evicted_entry[2]
            logger.debug(
                "Queue evict: request_id=%s class=%s priority=%d depth=%d",
                record.request_id, record.request_class, record.priority, len(self._heap),
            )
            return record

    def depth(self) -> int:
        """Return current queue size."""
        with self._lock:
            return len(self._heap)

    def depth_by_class(self) -> dict[str, int]:
        """Return dict mapping request_class to count of queued requests."""
        with self._lock:
            counts: dict[str, int] = {}
            for _, _, record in self._heap:
                counts[record.request_class] = counts.get(record.request_class, 0) + 1
            return counts
