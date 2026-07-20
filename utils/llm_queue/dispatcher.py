"""OllamaDispatcher facade — top-level orchestrator for the LLM queue layer.

Wires together all sub-components (classifier, admission, priority queue,
concurrency limiter, deadline enforcer, staleness checker, fallback router,
telemetry emitter) and exposes a single dispatch() entry point.

Worker-thread pool drains the priority queue:
- Pool size = global_concurrency (default 1)
- Each worker loops: pop from queue → check deadline/staleness → acquire
  per-model concurrency slot → execute → signal caller → emit telemetry
- Workers are daemon threads started on first dispatcher use
- Callers enqueue and block on a per-request threading.Event

The dispatcher NEVER raises exceptions — all paths return DispatchResult.
"""

from __future__ import annotations

import atexit
import logging
import threading
from datetime import datetime, timezone
from typing import Callable, Optional

from utils.llm_queue.admission import AdmissionController
from utils.llm_queue.classifier import RequestClassifier
from utils.llm_queue.concurrency import ConcurrencyLimiter
from utils.llm_queue.config import QueueConfig
from utils.llm_queue.deadline import DeadlineEnforcer
from utils.llm_queue.fallback import FallbackRouter
from utils.llm_queue.market_session import MarketSessionHelper
from utils.llm_queue.models import DispatchResult, RequestRecord
from utils.llm_queue.priority_queue import PriorityQueue
from utils.llm_queue.staleness import StalenessChecker
from utils.llm_queue.telemetry import TelemetryEmitter

logger = logging.getLogger(__name__)

# Type alias for the execute callback.
# Signature: (system_prompt, user_prompt, model, json_mode, timeout, num_ctx) -> str
# Returns the model response text. Raises on failure.
ExecuteFn = Callable[[str, str, str, bool, int, int], str]


def _default_execute_fn(
    system_prompt: str,
    user_prompt: str,
    model: str,
    json_mode: bool,
    timeout: int,
    num_ctx: int,
) -> str:
    """Placeholder execute function — replaced by integration layer (task 11.2).

    Raises NotImplementedError so tests can inject their own callable.
    """
    raise NotImplementedError(
        "execute_fn not wired — use OllamaDispatcher(config, execute_fn=...) "
        "or wait for integration task 11.2"
    )


class _PendingRequest:
    """Internal bookkeeping for a request waiting in the queue.

    Bundles the request record with its prompts, options, and a threading.Event
    that the worker signals when the result is ready.
    """

    __slots__ = (
        "record",
        "system_prompt",
        "user_prompt",
        "json_mode",
        "timeout",
        "num_ctx",
        "event",
        "result",
    )

    def __init__(
        self,
        record: RequestRecord,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool,
        timeout: int,
        num_ctx: int,
    ) -> None:
        self.record = record
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.json_mode = json_mode
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.event = threading.Event()
        self.result: Optional[DispatchResult] = None


class OllamaDispatcher:
    """Top-level facade orchestrating all queue/backpressure sub-components.

    Usage:
        config = QueueConfig.from_environment()
        dispatcher = OllamaDispatcher(config, execute_fn=my_ollama_call)
        result = dispatcher.dispatch(system_prompt, user_prompt, ...)
    """

    def __init__(
        self,
        config: QueueConfig,
        execute_fn: Optional[ExecuteFn] = None,
    ) -> None:
        self._config = config
        self._execute_fn: ExecuteFn = execute_fn or _default_execute_fn

        # Sub-components
        self._classifier = RequestClassifier(config)
        self._queue = PriorityQueue(config)
        self._market_session_helper = MarketSessionHelper()
        self._admission = AdmissionController(
            config, self._queue, self._market_session_helper
        )
        self._concurrency = ConcurrencyLimiter(config)
        self._deadline = DeadlineEnforcer(config)
        self._staleness = StalenessChecker(config)
        self._fallback = FallbackRouter(config)
        self._telemetry = TelemetryEmitter()

        # Pending requests registry: request_id → _PendingRequest
        # Workers look up the pending request after popping from the priority queue.
        self._pending: dict[str, _PendingRequest] = {}
        self._pending_lock = threading.Lock()

        # Worker pool state
        self._workers_started = False
        self._shutdown = False
        self._workers: list[threading.Thread] = []
        self._start_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_enabled(self) -> bool:
        """Return True if the dispatcher is in enforcing mode.

        When False, callers should use the existing lock-based path.
        """
        return self._config.mode == "enforcing"

    def disabled_classify(
        self,
        purpose: str,
        tier: str,
        model: str,
        json_mode: bool,
        prompt_chars: int,
    ) -> Optional[RequestRecord]:
        """Disabled mode: classify at DEBUG level only.

        Returns the RequestRecord (for potential future diagnostics) but the caller
        should proceed via the existing lock-based path. No queue/admission/telemetry.
        """
        record = self._classifier.classify(
            purpose=purpose,
            tier=tier,
            provider="ollama",
            model=model,
            json_mode=json_mode,
            prompt_chars=prompt_chars,
        )
        logger.debug(
            "Disabled mode classify: request_id=%s purpose=%s class=%s priority=%d model=%s",
            record.request_id,
            record.purpose,
            record.request_class,
            record.priority,
            record.model,
        )
        return record

    def dispatch(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        purpose: str,
        tier: str,
        json_mode: bool,
        timeout: int,
        num_ctx: int,
    ) -> DispatchResult:
        """Classify, admit, queue, execute, and return structured result.

        Never raises exceptions — always returns DispatchResult.
        """
        try:
            return self._dispatch_inner(
                system_prompt, user_prompt, model, purpose, tier,
                json_mode, timeout, num_ctx,
            )
        except Exception as exc:
            # Catastrophic fallback — should never reach here, but we guarantee
            # no exceptions escape.
            logger.error(
                "Dispatcher internal error (unhandled): %s", exc, exc_info=True
            )
            return DispatchResult(
                ok=False,
                text="",
                request_id="unknown",
                status="rejected",
                reason_code="dispatcher_internal_error",
                provider="ollama",
                model=model,
                fallback_used=False,
                queue_wait_seconds=0.0,
                elapsed_seconds=0.0,
                stale=False,
            )

    def get_cycle_summary(self) -> dict:
        """Delegate to TelemetryEmitter.get_cycle_summary()."""
        return self._telemetry.get_cycle_summary()

    def reset_cycle(self) -> None:
        """Delegate to TelemetryEmitter.reset_cycle()."""
        self._telemetry.reset_cycle()

    def observe_dispatch(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        purpose: str,
        tier: str,
        json_mode: bool,
        timeout: int,
        num_ctx: int,
    ) -> RequestRecord:
        """Observe mode: classify, log virtual decisions, emit telemetry.

        Performs full classification and admission check without enforcing.
        Logs what the queue decision WOULD have been (rejection, timeout, etc.)
        but does NOT enqueue or execute — the caller proceeds via the existing
        lock-based path.

        Returns the RequestRecord for the caller to use in telemetry. The caller
        is responsible for execution via the existing lock path.
        """
        prompt_chars = len(system_prompt) + len(user_prompt)

        # Step 1: Full classification at INFO level
        record = self._classifier.classify(
            purpose=purpose,
            tier=tier,
            provider="ollama",
            model=model,
            json_mode=json_mode,
            prompt_chars=prompt_chars,
        )

        logger.info(
            "observe: classified request_id=%s purpose=%s class=%s "
            "priority=%d model=%s market_session=%s",
            record.request_id,
            record.purpose,
            record.request_class,
            record.priority,
            record.model,
            record.market_session,
        )

        # Step 2: Virtual admission check (compute but don't enforce)
        admitted, reason_code = self._admission.admit_virtual(record)

        # Step 3: Log what queue decision WOULD have been
        if not admitted:
            logger.info(
                "observe: would have rejected request_id=%s class=%s "
                "reason=%s",
                record.request_id,
                record.request_class,
                reason_code,
            )
        else:
            # Compute virtual queue wait scenario
            current_depth = self._queue.depth()
            if current_depth >= self._config.max_queue_size:
                logger.info(
                    "observe: would have queued at capacity request_id=%s "
                    "class=%s depth=%d max=%d",
                    record.request_id,
                    record.request_class,
                    current_depth,
                    self._config.max_queue_size,
                )
            else:
                logger.info(
                    "observe: would have admitted request_id=%s class=%s "
                    "depth=%d",
                    record.request_id,
                    record.request_class,
                    current_depth,
                )

        # Step 4: Emit telemetry with classification + virtual decision
        # Build a synthetic DispatchResult representing the observe outcome
        observe_status = "completed" if admitted else self._observe_status_from_reason(reason_code)
        observe_result = DispatchResult(
            ok=admitted,
            text="",
            request_id=record.request_id,
            status=observe_status,
            reason_code=reason_code,
            provider="ollama",
            model=record.model,
            fallback_used=False,
            queue_wait_seconds=0.0,
            elapsed_seconds=0.0,
            stale=False,
        )

        try:
            self._telemetry.emit(record, observe_result, self._queue.depth())
        except Exception:
            # Telemetry is fail-open
            pass

        return record

    @staticmethod
    def _observe_status_from_reason(reason_code: Optional[str]) -> str:
        """Map a rejection reason_code to a DispatchResult status for observe telemetry."""
        if reason_code is None:
            return "completed"
        if reason_code == "prompt_too_large":
            return "prompt_too_large"
        if reason_code == "market_hour_pressure":
            return "deferred"
        # queue_full, class_queue_full, no_admission_path → rejected
        return "rejected"

    # ------------------------------------------------------------------
    # Internal dispatch orchestration
    # ------------------------------------------------------------------

    def _dispatch_inner(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        purpose: str,
        tier: str,
        json_mode: bool,
        timeout: int,
        num_ctx: int,
    ) -> DispatchResult:
        """Core dispatch flow — may raise; outer dispatch() catches everything."""
        prompt_chars = len(system_prompt) + len(user_prompt)

        # Step 1: Classify
        record = self._classifier.classify(
            purpose=purpose,
            tier=tier,
            provider="ollama",
            model=model,
            json_mode=json_mode,
            prompt_chars=prompt_chars,
        )

        logger.info(
            "Dispatch: classified request_id=%s purpose=%s class=%s priority=%d model=%s",
            record.request_id,
            record.purpose,
            record.request_class,
            record.priority,
            record.model,
        )

        # Step 2: Admission check
        admitted, reason_code = self._admission.admit(record)
        if not admitted:
            result = self._make_rejection_result(record, reason_code)
            self._telemetry.emit(record, result, self._queue.depth())
            return result

        # Step 3: Enqueue and wait
        # Ensure workers are running
        self._ensure_workers_started()

        # Create pending request
        pending = _PendingRequest(
            record=record,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=json_mode,
            timeout=timeout,
            num_ctx=num_ctx,
        )

        # Register in pending map so worker can find it
        with self._pending_lock:
            self._pending[record.request_id] = pending

        # Put record into priority queue (worker will pop it)
        self._queue.put(record)

        # Block caller on event with deadline timeout
        wait_timeout = record.deadline_seconds
        signalled = pending.event.wait(timeout=wait_timeout)

        # Remove from pending registry
        with self._pending_lock:
            self._pending.pop(record.request_id, None)

        if signalled and pending.result is not None:
            return pending.result

        # Caller timed out waiting for worker — deadline exceeded
        now = datetime.now(timezone.utc)
        elapsed = (now - record.created_at).total_seconds()

        # Attempt fallback on timeout
        fallback_result = self._try_fallback(
            record, "deadline_exceeded", system_prompt, user_prompt,
            json_mode, timeout, num_ctx, elapsed,
        )
        if fallback_result is not None:
            self._telemetry.emit(record, fallback_result, self._queue.depth())
            return fallback_result

        result = DispatchResult(
            ok=False,
            text="",
            request_id=record.request_id,
            status="timed_out",
            reason_code="deadline_exceeded",
            provider="ollama",
            model=record.model,
            fallback_used=False,
            queue_wait_seconds=elapsed,
            elapsed_seconds=elapsed,
            stale=False,
        )
        self._telemetry.emit(record, result, self._queue.depth())
        return result

    # ------------------------------------------------------------------
    # Worker pool
    # ------------------------------------------------------------------

    def _ensure_workers_started(self) -> None:
        """Start daemon worker threads on first use."""
        if self._workers_started:
            return
        with self._start_lock:
            if self._workers_started:
                return
            pool_size = self._config.global_concurrency
            for i in range(pool_size):
                t = threading.Thread(
                    target=self._worker_loop,
                    name=f"llm-queue-worker-{i}",
                    daemon=True,
                )
                t.start()
                self._workers.append(t)
            self._workers_started = True
            atexit.register(self._shutdown_workers)
            logger.info(
                "Dispatcher: started %d worker thread(s)", pool_size
            )

    def _shutdown_workers(self) -> None:
        """Signal workers to exit (called at interpreter shutdown)."""
        self._shutdown = True
        # Wake up workers blocked on queue.get()
        # Put None sentinel records — workers check for shutdown
        for _ in self._workers:
            try:
                # Notify the queue's condition to wake blocked workers
                with self._queue._condition:
                    self._queue._condition.notify_all()
            except Exception:
                pass

    def _worker_loop(self) -> None:
        """Worker thread main loop: pop → process → signal caller."""
        while not self._shutdown:
            try:
                # Block waiting for next request (with short timeout to check shutdown)
                record = self._queue.get(timeout=1.0)
                if record is None:
                    # Timeout or spurious wakeup — loop back to check shutdown
                    continue

                self._process_request(record)

            except Exception as exc:
                logger.error(
                    "Worker unhandled exception: %s", exc, exc_info=True
                )
                # Worker continues — no death spiral

    def _process_request(self, record: RequestRecord) -> None:
        """Process a single request popped from the queue.

        Steps: check deadline → check staleness → acquire concurrency →
        execute → release → check result staleness → signal caller.
        """
        # Look up the pending request
        with self._pending_lock:
            pending = self._pending.get(record.request_id)

        if pending is None:
            # Caller already timed out and cleaned up — request is orphaned
            logger.debug(
                "Worker: orphaned request_id=%s (caller timed out), skipping",
                record.request_id,
            )
            return

        now = datetime.now(timezone.utc)
        queue_wait = (now - record.created_at).total_seconds()

        # Step 5: Check queue wait deadline
        expired, _ = self._deadline.check_queue_wait(record)
        if expired:
            # Try fallback before giving up
            fallback_result = self._try_fallback(
                record, "queue_wait_exceeded",
                pending.system_prompt, pending.user_prompt,
                pending.json_mode, pending.timeout, pending.num_ctx,
                queue_wait,
            )
            if fallback_result is not None:
                self._signal_caller(pending, fallback_result)
                return

            result = DispatchResult(
                ok=False,
                text="",
                request_id=record.request_id,
                status="timed_out",
                reason_code="queue_wait_exceeded",
                provider="ollama",
                model=record.model,
                fallback_used=False,
                queue_wait_seconds=queue_wait,
                elapsed_seconds=queue_wait,
                stale=False,
            )
            self._signal_caller(pending, result)
            return

        # Step 6: Check staleness before execution
        if self._staleness.is_stale_before_execution(record):
            # Try fallback before giving up
            fallback_result = self._try_fallback(
                record, "stale_before_execution",
                pending.system_prompt, pending.user_prompt,
                pending.json_mode, pending.timeout, pending.num_ctx,
                queue_wait,
            )
            if fallback_result is not None:
                self._signal_caller(pending, fallback_result)
                return

            result = DispatchResult(
                ok=False,
                text="",
                request_id=record.request_id,
                status="stale",
                reason_code="stale_before_execution",
                provider="ollama",
                model=record.model,
                fallback_used=False,
                queue_wait_seconds=queue_wait,
                elapsed_seconds=queue_wait,
                stale=True,
            )
            self._signal_caller(pending, result)
            return

        # Step 7: Acquire concurrency slot
        # Calculate remaining time budget for concurrency acquisition
        deadline_remaining = record.deadline_seconds - queue_wait
        concurrency_timeout = max(0.1, deadline_remaining)

        acquired = self._concurrency.acquire(record.model, timeout=concurrency_timeout)
        if not acquired:
            elapsed = (datetime.now(timezone.utc) - record.created_at).total_seconds()

            # Try fallback
            fallback_result = self._try_fallback(
                record, "concurrency_timeout",
                pending.system_prompt, pending.user_prompt,
                pending.json_mode, pending.timeout, pending.num_ctx,
                queue_wait,
            )
            if fallback_result is not None:
                self._signal_caller(pending, fallback_result)
                return

            result = DispatchResult(
                ok=False,
                text="",
                request_id=record.request_id,
                status="timed_out",
                reason_code="concurrency_timeout",
                provider="ollama",
                model=record.model,
                fallback_used=False,
                queue_wait_seconds=queue_wait,
                elapsed_seconds=elapsed,
                stale=False,
            )
            self._signal_caller(pending, result)
            return

        # Step 8: Execute against Ollama
        try:
            text = self._execute_fn(
                pending.system_prompt,
                pending.user_prompt,
                record.model,
                pending.json_mode,
                pending.timeout,
                pending.num_ctx,
            )
            execution_success = True
        except Exception as exc:
            logger.warning(
                "Ollama execution failed for request_id=%s: %s",
                record.request_id,
                exc,
            )
            text = ""
            execution_success = False
        finally:
            # Step 9: Release concurrency slot
            self._concurrency.release(record.model)

        completed_at = datetime.now(timezone.utc)
        elapsed = (completed_at - record.created_at).total_seconds()

        if not execution_success:
            # Try fallback on execution failure
            fallback_result = self._try_fallback(
                record, "local_execution_failed",
                pending.system_prompt, pending.user_prompt,
                pending.json_mode, pending.timeout, pending.num_ctx,
                queue_wait,
            )
            if fallback_result is not None:
                self._signal_caller(pending, fallback_result)
                return

            result = DispatchResult(
                ok=False,
                text="",
                request_id=record.request_id,
                status="timed_out",
                reason_code="local_execution_failed",
                provider="ollama",
                model=record.model,
                fallback_used=False,
                queue_wait_seconds=queue_wait,
                elapsed_seconds=elapsed,
                stale=False,
            )
            self._signal_caller(pending, result)
            return

        # Step 10: Check result staleness after execution
        stale_after = self._staleness.is_stale_after_execution(record, completed_at)

        # For execution_critical, stale results are failures
        ok = True
        status = "completed"
        if stale_after and record.request_class == "execution_critical":
            ok = False
            status = "stale"

        result = DispatchResult(
            ok=ok,
            text=text,
            request_id=record.request_id,
            status=status,
            reason_code="stale_after_execution" if stale_after else None,
            provider="ollama",
            model=record.model,
            fallback_used=False,
            queue_wait_seconds=queue_wait,
            elapsed_seconds=elapsed,
            stale=stale_after,
        )
        self._signal_caller(pending, result)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _signal_caller(self, pending: _PendingRequest, result: DispatchResult) -> None:
        """Store result and signal the caller's event. Emit telemetry."""
        pending.result = result
        pending.event.set()
        # Step 11: Emit telemetry
        try:
            self._telemetry.emit(pending.record, result, self._queue.depth())
        except Exception:
            # Telemetry is fail-open
            pass

    def _make_rejection_result(
        self, record: RequestRecord, reason_code: Optional[str]
    ) -> DispatchResult:
        """Build a DispatchResult for a rejected request."""
        now = datetime.now(timezone.utc)
        elapsed = (now - record.created_at).total_seconds()

        # Determine status from reason_code
        if reason_code == "prompt_too_large":
            status = "prompt_too_large"
        elif reason_code == "market_hour_pressure":
            status = "deferred"
        else:
            status = "rejected"

        return DispatchResult(
            ok=False,
            text="",
            request_id=record.request_id,
            status=status,
            reason_code=reason_code,
            provider="ollama",
            model=record.model,
            fallback_used=False,
            queue_wait_seconds=0.0,
            elapsed_seconds=elapsed,
            stale=False,
        )

    def _try_fallback(
        self,
        record: RequestRecord,
        reason: str,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool,
        timeout: int,
        num_ctx: int,
        queue_wait: float,
    ) -> Optional[DispatchResult]:
        """Attempt fallback routing when local execution fails.

        Returns a DispatchResult if fallback succeeded, None otherwise.
        """
        try:
            decision = self._fallback.resolve_fallback(record, reason)
            if decision is None:
                return None

            # Execute via fallback — for now this is a placeholder that the
            # integration layer (task 11.2) will wire up. The fallback execute_fn
            # would call the remote provider. Since we don't have that wired yet,
            # we log and return a structured fallback result indicating the
            # decision was made but execution is not yet implemented.
            logger.info(
                "Fallback resolved: request_id=%s provider=%s model=%s reason=%s",
                record.request_id,
                decision.provider,
                decision.model,
                reason,
            )

            # Attempt fallback execution via the same execute_fn pattern.
            # In production, the integration layer will provide a fallback-aware
            # execute_fn or the dispatcher will have a separate fallback_execute_fn.
            # For now, we report the fallback was resolved but cannot execute it.
            now = datetime.now(timezone.utc)
            elapsed = (now - record.created_at).total_seconds()

            return DispatchResult(
                ok=False,
                text="",
                request_id=record.request_id,
                status="fallback",
                reason_code=reason,
                provider=decision.provider,
                model=decision.model,
                fallback_used=True,
                queue_wait_seconds=queue_wait,
                elapsed_seconds=elapsed,
                stale=False,
            )

        except Exception as exc:
            logger.warning(
                "Fallback resolution failed for request_id=%s: %s",
                record.request_id,
                exc,
            )
            return None
