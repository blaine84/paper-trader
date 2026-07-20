"""Staleness checking for queued LLM requests.

Determines whether a request or its result has exceeded the useful decision
window (stale_after_seconds). A stale request is one whose output is too old
to be actionable for its original purpose.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from utils.llm_queue.config import QueueConfig
from utils.llm_queue.models import RequestRecord

logger = logging.getLogger(__name__)


class StalenessChecker:
    """Checks whether a request has exceeded its staleness window.

    The config is accepted for interface consistency with other dispatcher
    components, but staleness thresholds are carried on each RequestRecord
    (assigned at classification time).
    """

    def __init__(self, config: QueueConfig) -> None:
        self._config = config

    def is_stale_before_execution(self, record: RequestRecord) -> bool:
        """True if request has waited past stale_after_seconds before starting.

        Compares elapsed time from record.created_at to now against the
        record's stale_after_seconds threshold.
        """
        now = datetime.now(timezone.utc)
        elapsed = (now - record.created_at).total_seconds()
        return elapsed > record.stale_after_seconds

    def is_stale_after_execution(self, record: RequestRecord, completed_at: datetime) -> bool:
        """True if result arrived after stale_after_seconds from creation.

        Compares elapsed time from record.created_at to the provided
        completed_at timestamp against the record's stale_after_seconds
        threshold.
        """
        elapsed = (completed_at - record.created_at).total_seconds()
        return elapsed > record.stale_after_seconds
