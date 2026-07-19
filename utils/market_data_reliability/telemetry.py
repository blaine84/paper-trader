"""TelemetryCollector for the Market Data Reliability Layer.

Accumulates per-cycle metrics for observability. All operations are
fail-open: exceptions are logged at ERROR level and never propagate.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class TelemetryCollector:
    """Accumulates per-cycle market-data reliability metrics.

    Thread-safe via threading.Lock. All public methods are wrapped in
    try/except so that telemetry failures never crash the trading loop.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._provider_calls_success: int = 0
        self._provider_calls_failure: int = 0
        self._cache_hits: int = 0
        self._fallback_usage: int = 0
        self._stale_snapshots: int = 0
        self._unavailable_snapshots: int = 0
        self._fail_closed_decisions: int = 0
        self._fail_closed_details: list[dict[str, str]] = []
        self._fallback_details: list[dict[str, str]] = []

    def record_provider_call(self, provider: str, data_type: str, success: bool) -> None:
        """Record a provider call outcome (success or failure)."""
        try:
            with self._lock:
                if success:
                    self._provider_calls_success += 1
                else:
                    self._provider_calls_failure += 1
        except Exception:
            logger.error(
                "Telemetry error in record_provider_call: provider=%s, data_type=%s",
                provider,
                data_type,
                exc_info=True,
            )

    def record_cache_hit(self, data_type: str) -> None:
        """Record a cache hit for a data type."""
        try:
            with self._lock:
                self._cache_hits += 1
        except Exception:
            logger.error(
                "Telemetry error in record_cache_hit: data_type=%s",
                data_type,
                exc_info=True,
            )

    def record_fallback_usage(self, primary: str, fallback: str, data_type: str) -> None:
        """Record that a fallback provider was used."""
        try:
            with self._lock:
                self._fallback_usage += 1
                self._fallback_details.append({
                    "primary": primary,
                    "fallback": fallback,
                    "data_type": data_type,
                })
        except Exception:
            logger.error(
                "Telemetry error in record_fallback_usage: primary=%s, fallback=%s, data_type=%s",
                primary,
                fallback,
                data_type,
                exc_info=True,
            )

    def record_fail_closed(self, candidate_id: str, reason: str) -> None:
        """Record a fail-closed decision blocking a candidate."""
        try:
            with self._lock:
                self._fail_closed_decisions += 1
                self._fail_closed_details.append({
                    "candidate_id": candidate_id,
                    "reason": reason,
                })
        except Exception:
            logger.error(
                "Telemetry error in record_fail_closed: candidate_id=%s, reason=%s",
                candidate_id,
                reason,
                exc_info=True,
            )

    def record_stale_snapshot(self, symbol: str, data_type: str) -> None:
        """Record that a stale snapshot was served."""
        try:
            with self._lock:
                self._stale_snapshots += 1
        except Exception:
            logger.error(
                "Telemetry error in record_stale_snapshot: symbol=%s, data_type=%s",
                symbol,
                data_type,
                exc_info=True,
            )

    def record_unavailable_snapshot(self, symbol: str, data_type: str) -> None:
        """Record that an unavailable snapshot was produced."""
        try:
            with self._lock:
                self._unavailable_snapshots += 1
        except Exception:
            logger.error(
                "Telemetry error in record_unavailable_snapshot: symbol=%s, data_type=%s",
                symbol,
                data_type,
                exc_info=True,
            )

    def get_cycle_summary(self) -> dict:
        """Return a summary of all metrics accumulated this cycle.

        Returns a dict with success/failure/cache_hit/fallback/stale/
        unavailable/fail_closed counts and detail lists.
        """
        try:
            with self._lock:
                return {
                    "provider_calls_success": self._provider_calls_success,
                    "provider_calls_failure": self._provider_calls_failure,
                    "cache_hits": self._cache_hits,
                    "fallback_usage": self._fallback_usage,
                    "stale_snapshots": self._stale_snapshots,
                    "unavailable_snapshots": self._unavailable_snapshots,
                    "fail_closed_decisions": self._fail_closed_decisions,
                    "fail_closed_details": list(self._fail_closed_details),
                    "fallback_details": list(self._fallback_details),
                }
        except Exception:
            logger.error(
                "Telemetry error in get_cycle_summary",
                exc_info=True,
            )
            return {
                "provider_calls_success": 0,
                "provider_calls_failure": 0,
                "cache_hits": 0,
                "fallback_usage": 0,
                "stale_snapshots": 0,
                "unavailable_snapshots": 0,
                "fail_closed_decisions": 0,
                "fail_closed_details": [],
                "fallback_details": [],
            }

    def reset(self) -> None:
        """Reset all counters and detail lists for a new cycle."""
        try:
            with self._lock:
                self._provider_calls_success = 0
                self._provider_calls_failure = 0
                self._cache_hits = 0
                self._fallback_usage = 0
                self._stale_snapshots = 0
                self._unavailable_snapshots = 0
                self._fail_closed_decisions = 0
                self._fail_closed_details = []
                self._fallback_details = []
        except Exception:
            logger.error(
                "Telemetry error in reset",
                exc_info=True,
            )
