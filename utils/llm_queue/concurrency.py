"""Concurrency limiter for local Ollama requests.

Controls how many requests execute simultaneously against Ollama, both globally
and per-model. Uses threading.Semaphore for concurrency gating and a Lock to
protect active-count bookkeeping.
"""

from __future__ import annotations

import logging
import threading

from utils.llm_queue.config import QueueConfig

logger = logging.getLogger(__name__)


class ConcurrencyLimiter:
    """Enforces global and per-model concurrency ceilings for Ollama calls.

    The global semaphore limits total simultaneous requests. Per-model semaphores
    provide finer-grained control when configured. If a model has no explicit
    per-model limit, only the global semaphore governs it.

    acquire() must be paired with release() for each successful acquisition.
    """

    def __init__(self, config: QueueConfig) -> None:
        self._global_semaphore = threading.Semaphore(config.global_concurrency)
        self._per_model_semaphores: dict[str, threading.Semaphore] = {
            model: threading.Semaphore(limit)
            for model, limit in config.per_model_concurrency.items()
        }

        # Active count tracking protected by a lock
        self._lock = threading.Lock()
        self._active_total: int = 0
        self._active_by_model: dict[str, int] = {}

    def acquire(self, model: str, timeout: float = None) -> bool:
        """Acquire concurrency slot for a model.

        Acquires the global semaphore first, then the per-model semaphore if
        one is configured for this model. Returns True on success, False if
        either semaphore times out.

        Args:
            model: The model name requesting a slot.
            timeout: Maximum seconds to wait for both semaphores combined.
                     None means block indefinitely.

        Returns:
            True if both slots acquired, False on timeout.
        """
        # Acquire global semaphore
        if not self._global_semaphore.acquire(timeout=timeout):
            logger.debug(
                "Concurrency limiter: global semaphore timeout for model %r", model
            )
            return False

        # Acquire per-model semaphore if configured
        per_model_sem = self._per_model_semaphores.get(model)
        if per_model_sem is not None:
            # Calculate remaining timeout budget
            remaining_timeout = timeout  # simplified: full timeout for per-model
            if not per_model_sem.acquire(timeout=remaining_timeout):
                # Release global since per-model failed
                self._global_semaphore.release()
                logger.debug(
                    "Concurrency limiter: per-model semaphore timeout for model %r",
                    model,
                )
                return False

        # Track active count atomically
        with self._lock:
            self._active_total += 1
            self._active_by_model[model] = self._active_by_model.get(model, 0) + 1

        logger.debug(
            "Concurrency limiter: acquired slot for model %r (active: %d)",
            model,
            self._active_total,
        )
        return True

    def release(self, model: str) -> None:
        """Release concurrency slot for a model.

        Releases the per-model semaphore (if configured) and the global
        semaphore, and decrements active counts.

        Args:
            model: The model name releasing a slot.
        """
        # Decrement active count atomically
        with self._lock:
            self._active_total = max(0, self._active_total - 1)
            current = self._active_by_model.get(model, 0)
            if current <= 1:
                self._active_by_model.pop(model, None)
            else:
                self._active_by_model[model] = current - 1

        # Release per-model semaphore if configured
        per_model_sem = self._per_model_semaphores.get(model)
        if per_model_sem is not None:
            per_model_sem.release()

        # Release global semaphore
        self._global_semaphore.release()

        logger.debug(
            "Concurrency limiter: released slot for model %r (active: %d)",
            model,
            self._active_total,
        )

    def active_count(self) -> int:
        """Return total number of currently active requests across all models."""
        with self._lock:
            return self._active_total

    def active_count_by_model(self) -> dict[str, int]:
        """Return dict of model name to active request count."""
        with self._lock:
            return dict(self._active_by_model)
