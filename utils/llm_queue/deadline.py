"""Deadline enforcement for queued LLM requests.

Checks timing constraints (queue wait, overall deadline, fallback viability)
against request records to determine whether requests have expired or whether
fallback can still meet the deadline.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from utils.llm_queue.config import QueueConfig
from utils.llm_queue.models import RequestRecord

logger = logging.getLogger(__name__)


class DeadlineEnforcer:
    """Checks timing constraints before and after execution.

    All time comparisons use UTC wall-clock time against record.created_at.
    """

    def __init__(self, config: QueueConfig) -> None:
        self._config = config

    def check_queue_wait(self, record: RequestRecord) -> tuple[bool, float]:
        """Check whether a request has exceeded its max queue wait time.

        Returns (expired, wait_seconds).
        expired=True means max_queue_wait_seconds has been exceeded.
        """
        now = datetime.now(timezone.utc)
        elapsed = (now - record.created_at).total_seconds()
        expired = elapsed > record.max_queue_wait_seconds
        if expired:
            logger.warning(
                "Request %s exceeded max_queue_wait: %.1fs > %.1fs "
                "(class=%s, purpose=%s)",
                record.request_id,
                elapsed,
                record.max_queue_wait_seconds,
                record.request_class,
                record.purpose,
            )
        return (expired, elapsed)

    def check_deadline(self, record: RequestRecord) -> tuple[bool, float]:
        """Check whether a request has exceeded its overall deadline.

        Returns (expired, elapsed_seconds).
        expired=True means deadline_seconds has been exceeded.
        """
        now = datetime.now(timezone.utc)
        elapsed = (now - record.created_at).total_seconds()
        expired = elapsed > record.deadline_seconds
        if expired:
            logger.warning(
                "Request %s exceeded deadline: %.1fs > %.1fs "
                "(class=%s, purpose=%s)",
                record.request_id,
                elapsed,
                record.deadline_seconds,
                record.request_class,
                record.purpose,
            )
        return (expired, elapsed)

    def can_fallback_meet_deadline(
        self, record: RequestRecord, fallback_budget_seconds: float
    ) -> bool:
        """Check whether enough time remains for a fallback call.

        Returns True if the remaining time before the overall deadline is
        at least fallback_budget_seconds.
        """
        now = datetime.now(timezone.utc)
        elapsed = (now - record.created_at).total_seconds()
        remaining = record.deadline_seconds - elapsed
        can_meet = remaining >= fallback_budget_seconds
        if not can_meet:
            logger.debug(
                "Request %s cannot meet fallback budget: %.1fs remaining < %.1fs needed "
                "(class=%s, purpose=%s)",
                record.request_id,
                remaining,
                fallback_budget_seconds,
                record.request_class,
                record.purpose,
            )
        return can_meet
