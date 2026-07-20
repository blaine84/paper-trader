"""Admission controller for the LLM request queue.

Decides whether a new request may enter the priority queue based on:
- Prompt size guardrails
- Global queue depth limits
- Per-class queue limits
- Market-hour pressure policy (advisory/review rejected earlier)
- Execution-critical reserved admission (eviction of lowest-priority)

Admission rules are evaluated in order; first rejection reason wins.
"""

from __future__ import annotations

import logging
from typing import Optional

from utils.llm_queue.config import QueueConfig
from utils.llm_queue.market_session import MarketSessionHelper
from utils.llm_queue.models import RequestRecord
from utils.llm_queue.priority_queue import PriorityQueue

logger = logging.getLogger(__name__)

# Request classes subject to market-hour pressure (rejected earlier under load)
_MARKET_HOUR_PRESSURE_CLASSES = frozenset({"advisory", "review"})


class AdmissionController:
    """Decides whether a new request may enter the queue.

    Returns (admitted, reason_code) where reason_code is set on rejection.
    execution_critical requests are never rejected outright — they may evict
    the lowest-priority queued request to make room.
    """

    def __init__(
        self,
        config: QueueConfig,
        queue: PriorityQueue,
        market_session_helper: Optional[MarketSessionHelper] = None,
    ) -> None:
        self._config = config
        self._queue = queue
        self._market_session_helper = market_session_helper or MarketSessionHelper()

    def admit(self, record: RequestRecord) -> tuple[bool, Optional[str]]:
        """Evaluate admission rules for the given request record.

        Returns:
            (True, None) if the request is admitted to the queue.
            (False, reason_code) if the request is rejected.

        Admission rules (evaluated in order):
        1. Prompt size exceeds reject threshold → reject with prompt_too_large
        2. Queue at max_queue_size and request not execution_critical
           → reject lower-priority with queue_full
        3. Per-class queue limit exceeded → reject with class_queue_full
        4. During market hours, advisory/review at 50% capacity → market_hour_pressure
        5. execution_critical: evict lowest-priority if queue full;
           if no safe path (all execution_critical) → no_admission_path
        """
        # Rule 1: Prompt size guardrail
        if record.approx_prompt_tokens >= self._config.prompt_token_reject_threshold:
            logger.warning(
                "Admission rejected: prompt_too_large request_id=%s "
                "tokens=%d threshold=%d class=%s",
                record.request_id,
                record.approx_prompt_tokens,
                self._config.prompt_token_reject_threshold,
                record.request_class,
            )
            return False, "prompt_too_large"

        # execution_critical has reserved admission — skip normal rejection rules
        if record.request_class == "execution_critical":
            return self._admit_execution_critical(record)

        # Rule 4: Market-hour pressure (checked before general queue-full since
        # the threshold is lower — 50% of max_queue_size)
        if record.request_class in _MARKET_HOUR_PRESSURE_CLASSES:
            if self._is_market_hours(record):
                pressure_threshold = self._config.max_queue_size // 2
                if self._queue.depth() >= pressure_threshold:
                    logger.info(
                        "Admission rejected: market_hour_pressure request_id=%s "
                        "class=%s depth=%d threshold=%d",
                        record.request_id,
                        record.request_class,
                        self._queue.depth(),
                        pressure_threshold,
                    )
                    return False, "market_hour_pressure"

        # Rule 2: Global queue depth
        if self._queue.depth() >= self._config.max_queue_size:
            logger.warning(
                "Admission rejected: queue_full request_id=%s class=%s "
                "priority=%d depth=%d max=%d",
                record.request_id,
                record.request_class,
                record.priority,
                self._queue.depth(),
                self._config.max_queue_size,
            )
            return False, "queue_full"

        # Rule 3: Per-class queue limit
        class_limit = self._config.max_queue_size_by_class.get(record.request_class)
        if class_limit is not None:
            class_depths = self._queue.depth_by_class()
            current_class_depth = class_depths.get(record.request_class, 0)
            if current_class_depth >= class_limit:
                logger.info(
                    "Admission rejected: class_queue_full request_id=%s "
                    "class=%s class_depth=%d class_limit=%d",
                    record.request_id,
                    record.request_class,
                    current_class_depth,
                    class_limit,
                )
                return False, "class_queue_full"

        # Admitted
        logger.debug(
            "Admission granted: request_id=%s class=%s priority=%d depth=%d",
            record.request_id,
            record.request_class,
            record.priority,
            self._queue.depth(),
        )
        return True, None

    def admit_virtual(self, record: RequestRecord) -> tuple[bool, Optional[str]]:
        """Evaluate admission rules WITHOUT side effects (no eviction).

        Used by observe mode to compute what the queue decision WOULD have been
        without actually modifying queue state.

        Returns:
            (True, None) if the request would be admitted.
            (False, reason_code) if the request would be rejected.
        """
        # Rule 1: Prompt size guardrail
        if record.approx_prompt_tokens >= self._config.prompt_token_reject_threshold:
            return False, "prompt_too_large"

        # execution_critical virtual admission (no eviction)
        if record.request_class == "execution_critical":
            return self._admit_execution_critical_virtual(record)

        # Rule 4: Market-hour pressure
        if record.request_class in _MARKET_HOUR_PRESSURE_CLASSES:
            if self._is_market_hours(record):
                pressure_threshold = self._config.max_queue_size // 2
                if self._queue.depth() >= pressure_threshold:
                    return False, "market_hour_pressure"

        # Rule 2: Global queue depth
        if self._queue.depth() >= self._config.max_queue_size:
            return False, "queue_full"

        # Rule 3: Per-class queue limit
        class_limit = self._config.max_queue_size_by_class.get(record.request_class)
        if class_limit is not None:
            class_depths = self._queue.depth_by_class()
            current_class_depth = class_depths.get(record.request_class, 0)
            if current_class_depth >= class_limit:
                return False, "class_queue_full"

        return True, None

    def _admit_execution_critical_virtual(
        self, record: RequestRecord
    ) -> tuple[bool, Optional[str]]:
        """Virtual admission check for execution_critical (no eviction).

        Computes whether admission would succeed but does NOT evict.
        """
        # Check per-class limit for execution_critical
        class_limit = self._config.max_queue_size_by_class.get("execution_critical")
        if class_limit is not None:
            class_depths = self._queue.depth_by_class()
            current_class_depth = class_depths.get("execution_critical", 0)
            if current_class_depth >= class_limit:
                return False, "class_queue_full"

        # If queue is not full, would admit directly
        if self._queue.depth() < self._config.max_queue_size:
            return True, None

        # Queue is full — check if eviction WOULD be possible
        lowest = self._queue.peek_lowest_priority()
        if lowest is None:
            return False, "no_admission_path"

        if lowest.request_class == "execution_critical":
            return False, "no_admission_path"

        # Would evict lowest-priority to make room
        return True, None

    def _admit_execution_critical(self, record: RequestRecord) -> tuple[bool, Optional[str]]:
        """Handle admission for execution_critical requests.

        execution_critical requests are never rejected for queue_full or
        market_hour_pressure. If the queue is full, they evict the lowest-priority
        queued request to make room.

        If the queue is full of only execution_critical requests (nothing to evict
        safely), admission fails closed with no_admission_path.
        """
        # Check per-class limit for execution_critical (unlikely but configurable)
        class_limit = self._config.max_queue_size_by_class.get("execution_critical")
        if class_limit is not None:
            class_depths = self._queue.depth_by_class()
            current_class_depth = class_depths.get("execution_critical", 0)
            if current_class_depth >= class_limit:
                logger.warning(
                    "Admission rejected: class_queue_full for execution_critical "
                    "request_id=%s class_depth=%d class_limit=%d",
                    record.request_id,
                    current_class_depth,
                    class_limit,
                )
                return False, "class_queue_full"

        # If queue is not full, admit directly
        if self._queue.depth() < self._config.max_queue_size:
            logger.debug(
                "Admission granted (execution_critical): request_id=%s depth=%d",
                record.request_id,
                self._queue.depth(),
            )
            return True, None

        # Queue is full — attempt eviction of lowest-priority request
        lowest = self._queue.peek_lowest_priority()
        if lowest is None:
            # Queue reports full but has no items — should not happen, fail closed
            logger.error(
                "Admission failed: no_admission_path request_id=%s "
                "queue reports full but peek_lowest_priority returned None",
                record.request_id,
            )
            return False, "no_admission_path"

        # Only evict if the lowest-priority request is lower priority (higher value)
        # than execution_critical (priority 0). If the queue is all execution_critical,
        # we cannot safely evict.
        if lowest.request_class == "execution_critical":
            logger.warning(
                "Admission failed: no_admission_path request_id=%s "
                "queue full of execution_critical requests, cannot evict",
                record.request_id,
            )
            return False, "no_admission_path"

        # Evict the lowest-priority request to make room
        evicted = self._queue.evict_lowest()
        if evicted is None:
            # Race condition: another thread evicted first — fail closed
            logger.error(
                "Admission failed: no_admission_path request_id=%s "
                "eviction returned None (race condition)",
                record.request_id,
            )
            return False, "no_admission_path"

        logger.info(
            "Admission granted (execution_critical via eviction): "
            "request_id=%s evicted_request_id=%s evicted_class=%s "
            "evicted_priority=%d",
            record.request_id,
            evicted.request_id,
            evicted.request_class,
            evicted.priority,
        )
        return True, None

    def _is_market_hours(self, record: RequestRecord) -> bool:
        """Determine if market hours apply for admission pressure.

        Uses the record's market_session field to determine if we are
        in regular market hours.
        """
        return record.market_session == "regular"
