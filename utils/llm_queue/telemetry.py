"""Structured telemetry for the LLM queue and backpressure layer.

Records terminal lifecycle events for every queued request and provides
per-cycle aggregate summaries for review surfaces (daily review, CEO output).

All telemetry logic is fail-open: emit() wraps all work in try/except and
never raises. Thread-safe via threading.Lock on the cycle counters.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from utils.llm_queue.models import DispatchResult, RequestRecord

logger = logging.getLogger(__name__)


class TelemetryEmitter:
    """Emits structured telemetry events and maintains per-cycle aggregates.

    Every request that reaches a terminal state (completed, timed_out,
    rejected, deferred, fallback, stale, cancelled, prompt_too_large) produces
    exactly one INFO-level structured log entry. Cycle summaries accumulate
    counts until reset_cycle() is called.

    Fail-open: all emit() logic is wrapped in try/except. If any error occurs,
    a WARNING is logged and execution continues. Never raises.

    Never persists full prompts or secrets.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset_counters()

    def _reset_counters(self) -> None:
        """Initialize or reset all per-cycle counters. Caller must hold lock."""
        self._total_requests: int = 0
        self._completed: int = 0
        self._timed_out: int = 0
        self._rejected: int = 0
        self._deferred: int = 0
        self._fallback_used: int = 0
        self._stale_results: int = 0

        # Wait time tracking
        self._queue_wait_seconds_sum: float = 0.0
        self._queue_wait_seconds_max: float = 0.0

        # Per-class breakdown: class_name -> {"count": int, "wait_sum": float}
        self._by_class: dict[str, dict[str, float]] = {}

        # Per-model breakdown: model_name -> {"count": int, "elapsed_sum": float}
        self._by_model: dict[str, dict[str, float]] = {}

    def emit(self, record: RequestRecord, result: DispatchResult, queue_depth: int) -> None:
        """Log a terminal telemetry event. Fail-open: never raises.

        Args:
            record: The classified request metadata.
            result: The dispatch result with status and timing.
            queue_depth: Current queue depth at emission time.
        """
        try:
            self._log_event(record, result, queue_depth)
            self._update_counters(record, result)
        except Exception:
            try:
                logger.warning(
                    "telemetry emit failed for request_id=%s",
                    getattr(record, "request_id", "unknown"),
                    exc_info=True,
                )
            except Exception:
                pass

    def get_cycle_summary(self) -> dict:
        """Returns per-cycle aggregate counts for review surfaces.

        Thread-safe snapshot of accumulated counters since last reset.
        """
        with self._lock:
            total = self._total_requests
            avg_wait = (
                self._queue_wait_seconds_sum / total if total > 0 else 0.0
            )

            by_class = {}
            for cls_name, data in self._by_class.items():
                count = int(data["count"])
                avg_cls_wait = data["wait_sum"] / count if count > 0 else 0.0
                by_class[cls_name] = {"count": count, "avg_wait_seconds": avg_cls_wait}

            by_model = {}
            for model_name, data in self._by_model.items():
                count = int(data["count"])
                avg_elapsed = data["elapsed_sum"] / count if count > 0 else 0.0
                by_model[model_name] = {"count": count, "avg_elapsed_seconds": avg_elapsed}

            return {
                "total_requests": total,
                "completed": self._completed,
                "timed_out": self._timed_out,
                "rejected": self._rejected,
                "deferred": self._deferred,
                "fallback_used": self._fallback_used,
                "stale_results": self._stale_results,
                "avg_queue_wait_seconds": avg_wait,
                "max_queue_wait_seconds": self._queue_wait_seconds_max,
                "by_class": by_class,
                "by_model": by_model,
            }

    def reset_cycle(self) -> None:
        """Clear per-cycle counters (called at cycle boundary)."""
        with self._lock:
            self._reset_counters()

    def _log_event(self, record: RequestRecord, result: DispatchResult, queue_depth: int) -> None:
        """Emit structured INFO log with all required telemetry fields.

        Never includes full prompts or secrets.
        """
        logger.info(
            "llm_queue_event",
            extra={
                "request_id": record.request_id,
                "purpose": record.purpose,
                "agent": record.agent,
                "request_class": record.request_class,
                "priority": record.priority,
                "tier": record.tier,
                "provider": result.provider,
                "model": result.model,
                "prompt_chars": record.prompt_chars,
                "approx_prompt_tokens": record.approx_prompt_tokens,
                "queue_depth": queue_depth,
                "queue_wait_seconds": result.queue_wait_seconds,
                "elapsed_seconds": result.elapsed_seconds,
                "status": result.status,
                "reason_code": result.reason_code,
                "fallback_used": result.fallback_used,
                "stale": result.stale,
                "market_session": record.market_session,
            },
        )

    def _update_counters(self, record: RequestRecord, result: DispatchResult) -> None:
        """Update per-cycle aggregate counters. Thread-safe."""
        with self._lock:
            self._total_requests += 1

            # Status counters
            status = result.status
            if status == "completed":
                self._completed += 1
            elif status == "timed_out":
                self._timed_out += 1
            elif status in ("rejected", "prompt_too_large"):
                self._rejected += 1
            elif status == "deferred":
                self._deferred += 1

            if result.fallback_used:
                self._fallback_used += 1

            if result.stale:
                self._stale_results += 1

            # Wait time tracking
            wait = result.queue_wait_seconds
            self._queue_wait_seconds_sum += wait
            if wait > self._queue_wait_seconds_max:
                self._queue_wait_seconds_max = wait

            # Per-class breakdown
            cls_name = record.request_class
            if cls_name not in self._by_class:
                self._by_class[cls_name] = {"count": 0.0, "wait_sum": 0.0}
            self._by_class[cls_name]["count"] += 1
            self._by_class[cls_name]["wait_sum"] += wait

            # Per-model breakdown
            model_name = result.model
            if model_name not in self._by_model:
                self._by_model[model_name] = {"count": 0.0, "elapsed_sum": 0.0}
            self._by_model[model_name]["count"] += 1
            self._by_model[model_name]["elapsed_sum"] += result.elapsed_seconds
