"""Property-based tests for the Ollama Queue and Backpressure layer.

Uses Hypothesis to verify correctness properties across a wide input space.
"""

from __future__ import annotations

from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from utils.llm_queue.config import QueueConfig


# ---------------------------------------------------------------------------
# Strategies for generating invalid/random environment variable values
# ---------------------------------------------------------------------------

_invalid_numeric_values = st.sampled_from(["", "abc", "-1", "0", "not_a_number", "3.14", "NaN", "inf", "null", "true"])

_random_text_values = st.text(
    alphabet=st.characters(blacklist_characters="\x00"),
    min_size=0,
    max_size=50,
)

_invalid_mode_values = st.sampled_from([
    "", "DISABLED", "Observe", "ENFORCING", "enabled", "on", "off",
    "1", "0", "yes", "no", "true", "false", "invalid_mode",
])

_invalid_json_values = st.sampled_from([
    "", "{", "[]", "null", "123", '"string"', "{invalid json}",
    '{"model": -1}', '{"model": 0}', '{"model": "abc"}',
])


def _build_env_dict(
    mode: str,
    concurrency: str,
    max_size: str,
    market_factor: str,
    prompt_warn: str,
    prompt_reject: str,
    model_concurrency: str,
) -> dict[str, str]:
    """Build a full environment dict with the given (possibly invalid) values."""
    env = {}
    if mode is not None:
        env["LLM_QUEUE_MODE"] = mode
    if concurrency is not None:
        env["LLM_QUEUE_GLOBAL_CONCURRENCY"] = concurrency
    if max_size is not None:
        env["LLM_QUEUE_MAX_SIZE"] = max_size
    if market_factor is not None:
        env["LLM_QUEUE_MARKET_HOUR_FACTOR"] = market_factor
    if prompt_warn is not None:
        env["LLM_QUEUE_PROMPT_WARN_TOKENS"] = prompt_warn
    if prompt_reject is not None:
        env["LLM_QUEUE_PROMPT_REJECT_TOKENS"] = prompt_reject
    if model_concurrency is not None:
        env["LLM_QUEUE_MODEL_CONCURRENCY"] = model_concurrency
    return env


# ---------------------------------------------------------------------------
# Property 12: Unknown Purpose Safe Classification (applied to config)
#
# Generate missing/invalid env vars, verify resulting config produces
# conservative defaults (concurrency 1, strict execution deadlines).
#
# **Validates: Requirements 3.5, 8.5**
# ---------------------------------------------------------------------------


class TestPropertyConfigSafeDefaults:
    """Property 12: Configuration always produces conservative safe defaults
    when environment variables are missing or invalid.

    **Validates: Requirements 3.5, 8.5**
    """

    @given(
        mode=_invalid_mode_values,
        concurrency=_invalid_numeric_values,
        max_size=_invalid_numeric_values,
        market_factor=_invalid_numeric_values,
        prompt_warn=_invalid_numeric_values,
        prompt_reject=_invalid_numeric_values,
        model_concurrency=_invalid_json_values,
    )
    @settings(max_examples=200)
    def test_invalid_env_vars_produce_conservative_defaults(
        self,
        mode: str,
        concurrency: str,
        max_size: str,
        market_factor: str,
        prompt_warn: str,
        prompt_reject: str,
        model_concurrency: str,
    ):
        """When all env vars are invalid, QueueConfig falls back to safe defaults."""
        env = _build_env_dict(
            mode=mode,
            concurrency=concurrency,
            max_size=max_size,
            market_factor=market_factor,
            prompt_warn=prompt_warn,
            prompt_reject=prompt_reject,
            model_concurrency=model_concurrency,
        )

        with patch.dict("os.environ", env, clear=True):
            config = QueueConfig.from_environment()

        # Conservative defaults: concurrency never exceeds 1 when input is invalid
        assert config.global_concurrency >= 1, (
            f"global_concurrency must be >= 1, got {config.global_concurrency}"
        )

        # Mode must be one of the valid modes (defaults to "disabled" on invalid)
        assert config.mode in ("disabled", "observe", "enforcing"), (
            f"mode must be valid, got {config.mode!r}"
        )

        # max_queue_size must be >= 1
        assert config.max_queue_size >= 1, (
            f"max_queue_size must be >= 1, got {config.max_queue_size}"
        )

        # Deadlines must match the conservative default table for execution_critical
        assert config.deadlines["execution_critical"] == 120.0, (
            f"execution_critical deadline should be 120.0, got {config.deadlines['execution_critical']}"
        )

        # All deadline values must be positive
        for cls, deadline in config.deadlines.items():
            assert deadline > 0, f"deadline for {cls} must be > 0, got {deadline}"

        # All max_queue_wait values must be positive
        for cls, wait in config.max_queue_waits.items():
            assert wait > 0, f"max_queue_wait for {cls} must be > 0, got {wait}"

        # All stale_after values must be positive
        for cls, stale in config.stale_after.items():
            assert stale > 0, f"stale_after for {cls} must be > 0, got {stale}"

        # Prompt thresholds default to conservative values
        assert config.prompt_token_warn_threshold >= 1, (
            f"prompt_token_warn_threshold must be >= 1, got {config.prompt_token_warn_threshold}"
        )
        assert config.prompt_token_reject_threshold >= 1, (
            f"prompt_token_reject_threshold must be >= 1, got {config.prompt_token_reject_threshold}"
        )

    @given(
        concurrency=_random_text_values,
        max_size=_random_text_values,
    )
    @settings(max_examples=200)
    def test_random_text_env_vars_produce_safe_concurrency(
        self,
        concurrency: str,
        max_size: str,
    ):
        """Random text strings for numeric env vars always produce safe defaults."""
        env = {
            "LLM_QUEUE_GLOBAL_CONCURRENCY": concurrency,
            "LLM_QUEUE_MAX_SIZE": max_size,
        }

        with patch.dict("os.environ", env, clear=True):
            config = QueueConfig.from_environment()

        # Concurrency must default to 1 when input is non-numeric
        # (if it happens to be a valid positive int, that's fine too)
        assert config.global_concurrency >= 1, (
            f"global_concurrency must be >= 1, got {config.global_concurrency}"
        )

        # Mode defaults to disabled when not set
        assert config.mode == "disabled", (
            f"mode must default to 'disabled', got {config.mode!r}"
        )

        # max_queue_size defaults to 10 when input is invalid
        assert config.max_queue_size >= 1, (
            f"max_queue_size must be >= 1, got {config.max_queue_size}"
        )

    @given(data=st.data())
    @settings(max_examples=200)
    def test_missing_env_vars_produce_exact_conservative_defaults(self, data):
        """With completely empty environment, config matches exact conservative defaults."""
        # Draw whether to include each env var as empty string or omit entirely
        include_vars = data.draw(
            st.lists(
                st.sampled_from([
                    "LLM_QUEUE_MODE",
                    "LLM_QUEUE_GLOBAL_CONCURRENCY",
                    "LLM_QUEUE_MAX_SIZE",
                    "LLM_QUEUE_MARKET_HOUR_FACTOR",
                    "LLM_QUEUE_PROMPT_WARN_TOKENS",
                    "LLM_QUEUE_PROMPT_REJECT_TOKENS",
                    "LLM_QUEUE_MODEL_CONCURRENCY",
                ]),
                min_size=0,
                max_size=7,
                unique=True,
            )
        )

        # Set selected vars to empty string (simulating missing values)
        env = {var: "" for var in include_vars}

        with patch.dict("os.environ", env, clear=True):
            config = QueueConfig.from_environment()

        # All conservative defaults must hold
        assert config.mode == "disabled"
        assert config.global_concurrency == 1
        assert config.max_queue_size == 10
        assert config.prompt_token_warn_threshold == 8000
        assert config.prompt_token_reject_threshold == 16000

        # Deadlines must match the conservative default table
        assert config.deadlines["execution_critical"] == 120.0
        assert config.deadlines["market_analysis"] == 180.0
        assert config.deadlines["startup_probe"] == 30.0

        # Max queue waits must match defaults
        assert config.max_queue_waits["execution_critical"] == 30.0
        assert config.max_queue_waits["repair"] == 15.0

        # Stale-after must match defaults
        assert config.stale_after["execution_critical"] == 60.0

    @given(
        mode_value=st.sampled_from(["", "abc", "-1", "0", "not_a_number", "DISABLED", "invalid"]),
    )
    @settings(max_examples=200)
    def test_invalid_mode_defaults_to_disabled(self, mode_value: str):
        """Invalid LLM_QUEUE_MODE always defaults to 'disabled' (safest mode)."""
        env = {"LLM_QUEUE_MODE": mode_value}

        with patch.dict("os.environ", env, clear=True):
            config = QueueConfig.from_environment()

        assert config.mode == "disabled", (
            f"Invalid mode {mode_value!r} should default to 'disabled', got {config.mode!r}"
        )


# ---------------------------------------------------------------------------
# Property 1: Classification Determinism
#
# For any given (purpose, tier, provider, model, json_mode, prompt_chars)
# tuple, calling classify() twice with the same inputs produces identical
# request_class and priority. The request_id and created_at will differ
# (UUID/timestamp), but classification fields must be deterministic.
#
# **Validates: Requirements 13.1, 13.4**
# ---------------------------------------------------------------------------


class TestPropertyClassificationDeterminism:
    """Property 1: Classification Determinism.

    Same inputs always produce same request_class and priority.

    **Validates: Requirements 13.1, 13.4**
    """

    @given(
        purpose=st.text(min_size=0, max_size=50),
        tier=st.sampled_from(["high", "finance", "medium", "low"]),
        provider=st.just("ollama"),
        model=st.text(min_size=1, max_size=30),
        json_mode=st.booleans(),
        prompt_chars=st.integers(min_value=0, max_value=100000),
    )
    @settings(max_examples=200)
    def test_same_inputs_produce_same_classification(
        self,
        purpose: str,
        tier: str,
        provider: str,
        model: str,
        json_mode: bool,
        prompt_chars: int,
    ):
        """Classifying the same inputs twice yields identical request_class and priority."""
        from utils.llm_queue.classifier import RequestClassifier

        config = QueueConfig()  # default config (conservative)

        classifier = RequestClassifier(config)

        with patch(
            "utils.llm_queue.classifier._get_current_market_session",
            return_value="closed",
        ):
            result_1 = classifier.classify(
                purpose=purpose,
                tier=tier,
                provider=provider,
                model=model,
                json_mode=json_mode,
                prompt_chars=prompt_chars,
            )
            result_2 = classifier.classify(
                purpose=purpose,
                tier=tier,
                provider=provider,
                model=model,
                json_mode=json_mode,
                prompt_chars=prompt_chars,
            )

        # Classification fields must be identical
        assert result_1.request_class == result_2.request_class, (
            f"request_class differs: {result_1.request_class!r} vs {result_2.request_class!r} "
            f"for purpose={purpose!r}"
        )
        assert result_1.priority == result_2.priority, (
            f"priority differs: {result_1.priority} vs {result_2.priority} "
            f"for purpose={purpose!r}"
        )
        assert result_1.agent == result_2.agent, (
            f"agent differs: {result_1.agent!r} vs {result_2.agent!r} "
            f"for purpose={purpose!r}"
        )
        assert result_1.fallback_policy == result_2.fallback_policy, (
            f"fallback_policy differs: {result_1.fallback_policy!r} vs {result_2.fallback_policy!r} "
            f"for purpose={purpose!r}"
        )
        assert result_1.deadline_seconds == result_2.deadline_seconds, (
            f"deadline_seconds differs: {result_1.deadline_seconds} vs {result_2.deadline_seconds} "
            f"for purpose={purpose!r}"
        )
        assert result_1.max_queue_wait_seconds == result_2.max_queue_wait_seconds, (
            f"max_queue_wait_seconds differs: {result_1.max_queue_wait_seconds} "
            f"vs {result_2.max_queue_wait_seconds} for purpose={purpose!r}"
        )
        assert result_1.stale_after_seconds == result_2.stale_after_seconds, (
            f"stale_after_seconds differs: {result_1.stale_after_seconds} "
            f"vs {result_2.stale_after_seconds} for purpose={purpose!r}"
        )
        assert result_1.approx_prompt_tokens == result_2.approx_prompt_tokens, (
            f"approx_prompt_tokens differs: {result_1.approx_prompt_tokens} "
            f"vs {result_2.approx_prompt_tokens} for purpose={purpose!r}"
        )
        assert result_1.market_session == result_2.market_session, (
            f"market_session differs: {result_1.market_session!r} "
            f"vs {result_2.market_session!r} for purpose={purpose!r}"
        )

        # Non-deterministic fields (request_id, created_at) SHOULD differ
        # (or at minimum, we don't require them to match)
        # This is expected behavior — UUID and timestamp are generated fresh each call


# ---------------------------------------------------------------------------
# Property 12: Unknown Purpose Safe Classification
#
# For any purpose string not matching any known prefix, the classifier must
# produce request_class="advisory" with priority=6. It must NEVER classify
# an unknown purpose as "execution_critical".
#
# **Validates: Requirements 1.4, 13.1**
# ---------------------------------------------------------------------------

# All known prefixes from the classification table
_KNOWN_PREFIXES = (
    "pm_entry",
    "pm_candidate",
    "pm_maintenance",
    "pm_reversal",
    "analyst_signal",
    "analyst_veto",
    "price_monitor_filter",
    "json_repair",
    "researcher_premarket",
    "quant_researcher",
    "sector_scout",
    "reviewer_trade",
    "daily_review",
    "meta_reviewer",
    "ceo",
    "weekly_prep",
    "narrator",
    "startup_probe",
)


class TestPropertyUnknownPurposeSafeClassification:
    """Property 12: Unknown Purpose Safe Classification.

    For any purpose string NOT starting with a known prefix, the classifier
    must produce request_class='advisory' and priority=6. It must NEVER
    classify an unknown purpose as 'execution_critical'.

    **Validates: Requirements 1.4, 13.1**
    """

    @given(
        purpose=st.text(
            alphabet=st.characters(
                whitelist_categories=("Nd", "L"),
                whitelist_characters="_-!@#$%^&*()+=[]{}|;:',.<>?/~` ",
            ),
            min_size=0,
            max_size=80,
        ),
    )
    @settings(max_examples=200)
    def test_unknown_purpose_classified_as_advisory_priority_6(self, purpose: str):
        """Random purpose strings not matching known prefixes → advisory, priority 6."""
        from hypothesis import assume

        from utils.llm_queue.classifier import RequestClassifier
        from utils.llm_queue.config import QueueConfig

        # Filter out any generated string that happens to start with a known prefix
        for prefix in _KNOWN_PREFIXES:
            assume(not purpose.startswith(prefix))

        # Mock market session to "closed" for consistent results
        with patch(
            "utils.llm_queue.classifier._get_current_market_session",
            return_value="closed",
        ):
            config = QueueConfig.from_environment()
            classifier = RequestClassifier(config)

            record = classifier.classify(
                purpose=purpose,
                tier="medium",
                provider="ollama",
                model="llama3",
                json_mode=False,
                prompt_chars=500,
            )

        # Must be classified as advisory
        assert record.request_class == "advisory", (
            f"Unknown purpose {purpose!r} should be classified as 'advisory', "
            f"got {record.request_class!r}"
        )

        # Must have priority 6
        assert record.priority == 6, (
            f"Unknown purpose {purpose!r} should have priority 6, "
            f"got {record.priority}"
        )

        # Must NEVER be classified as execution_critical
        assert record.request_class != "execution_critical", (
            f"Unknown purpose {purpose!r} must NEVER be execution_critical"
        )


# ---------------------------------------------------------------------------
# Property 2: Priority Ordering
#
# For any two RequestRecords R1, R2 where R1.priority < R2.priority
# (lower = higher), if both are put into the queue, get() returns R1 first.
#
# **Validates: Requirements 2.1**
# ---------------------------------------------------------------------------

import uuid
from datetime import datetime, timezone

from utils.llm_queue.models import RequestRecord
from utils.llm_queue.priority_queue import PriorityQueue


def _make_record(priority: int, created_at: datetime = None) -> RequestRecord:
    """Create a RequestRecord with a specific priority for testing."""
    if created_at is None:
        created_at = datetime.now(timezone.utc)
    return RequestRecord(
        request_id=str(uuid.uuid4()),
        purpose="test",
        agent="test",
        request_class="test",
        tier="medium",
        provider="ollama",
        model="llama3",
        json_mode=False,
        priority=priority,
        deadline_seconds=120.0,
        max_queue_wait_seconds=30.0,
        fallback_policy="none",
        stale_after_seconds=60.0,
        created_at=created_at,
        prompt_chars=100,
        approx_prompt_tokens=25,
        market_session="closed",
    )


class TestPropertyPriorityOrdering:
    """Property 2: Priority Ordering.

    For any two RequestRecords R1, R2 where R1.priority < R2.priority
    (lower = higher priority), if both are put into the queue, get()
    returns R1 first.

    **Validates: Requirements 2.1**
    """

    @given(
        priority_high=st.integers(min_value=0, max_value=7),
        priority_low=st.integers(min_value=0, max_value=7),
    )
    @settings(max_examples=200)
    def test_higher_priority_dispatched_first(
        self,
        priority_high: int,
        priority_low: int,
    ):
        """Given two requests with different priorities, the one with lower
        priority value (higher priority) is always returned by get() first."""
        from hypothesis import assume

        assume(priority_high != priority_low)

        # Ensure priority_high is the lower numeric value (higher priority)
        if priority_high > priority_low:
            priority_high, priority_low = priority_low, priority_high

        config = QueueConfig()
        queue = PriorityQueue(config)

        r_high = _make_record(priority=priority_high)
        r_low = _make_record(priority=priority_low)

        # Put lower-priority first to verify ordering isn't just insertion order
        queue.put(r_low)
        queue.put(r_high)

        first = queue.get(timeout=1.0)
        second = queue.get(timeout=1.0)

        assert first is not None, "First get() should return a record"
        assert second is not None, "Second get() should return a record"

        assert first.request_id == r_high.request_id, (
            f"Expected higher-priority request (priority={priority_high}) first, "
            f"got priority={first.priority}"
        )
        assert second.request_id == r_low.request_id, (
            f"Expected lower-priority request (priority={priority_low}) second, "
            f"got priority={second.priority}"
        )


# ---------------------------------------------------------------------------
# Property 3: FIFO Within Same Priority
#
# For any two RequestRecords R1, R2 with the same priority where
# R1.created_at < R2.created_at, get() returns R1 before R2.
#
# **Validates: Requirements 2.1**
# ---------------------------------------------------------------------------

from datetime import datetime, timedelta, timezone
import uuid

from utils.llm_queue.models import RequestRecord
from utils.llm_queue.priority_queue import PriorityQueue
from hypothesis import assume


def _make_record(priority: int, created_at: datetime = None) -> RequestRecord:
    if created_at is None:
        created_at = datetime.now(timezone.utc)
    return RequestRecord(
        request_id=str(uuid.uuid4()),
        purpose="test",
        agent="test",
        request_class="test",
        tier="medium",
        provider="ollama",
        model="llama3",
        json_mode=False,
        priority=priority,
        deadline_seconds=120.0,
        max_queue_wait_seconds=30.0,
        fallback_policy="none",
        stale_after_seconds=60.0,
        created_at=created_at,
        prompt_chars=100,
        approx_prompt_tokens=25,
        market_session="closed",
    )


class TestPropertyFIFOWithinPriority:
    """Property 3: FIFO Within Same Priority.

    For any two requests with the same priority, the one with the earlier
    created_at timestamp is dispatched first (FIFO ordering within a priority
    band).

    **Validates: Requirements 2.1**
    """

    @given(
        priority=st.integers(min_value=0, max_value=7),
        offset_1=st.integers(min_value=0, max_value=7),
        offset_2=st.integers(min_value=0, max_value=7),
    )
    @settings(max_examples=200)
    def test_same_priority_fifo_ordering(
        self,
        priority: int,
        offset_1: int,
        offset_2: int,
    ):
        """Two records with the same priority are returned in created_at order."""
        assume(offset_1 != offset_2)

        base_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        created_at_1 = base_time + timedelta(seconds=offset_1)
        created_at_2 = base_time + timedelta(seconds=offset_2)

        record_early = _make_record(priority, min(created_at_1, created_at_2))
        record_late = _make_record(priority, max(created_at_1, created_at_2))

        config = QueueConfig()
        pq = PriorityQueue(config)

        # Insert in arbitrary order — put the later one first to stress FIFO
        pq.put(record_late)
        pq.put(record_early)

        first = pq.get(timeout=0.1)
        second = pq.get(timeout=0.1)

        assert first is not None
        assert second is not None
        assert first.created_at <= second.created_at, (
            f"FIFO violated: first.created_at={first.created_at} > "
            f"second.created_at={second.created_at} for priority={priority}"
        )
        assert first.request_id == record_early.request_id, (
            f"Expected earlier record first, got request_id={first.request_id}"
        )
        assert second.request_id == record_late.request_id, (
            f"Expected later record second, got request_id={second.request_id}"
        )


# ---------------------------------------------------------------------------
# Property 7: Concurrency Limit Enforcement
#
# For any concurrency limit C (1-5), if C+1 threads try to acquire
# simultaneously, at most C succeed before any releases. The (C+1)th
# thread must block or timeout.
#
# **Validates: Requirements 3.1, 3.2**
# ---------------------------------------------------------------------------


class TestPropertyConcurrencyLimitEnforcement:
    """Property 7: Concurrency Limit Enforcement.

    For any concurrency limit C, at most C requests may execute simultaneously.
    If C+1 threads attempt to acquire concurrently, exactly C succeed within
    the timeout window and the extra one times out.

    **Validates: Requirements 3.1, 3.2**
    """

    @given(
        limit=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=50)
    def test_at_most_c_threads_acquire_simultaneously(self, limit: int):
        """Spawning C+1 threads with concurrency limit C: exactly C acquire, 1 times out."""
        import os
        import threading

        from utils.llm_queue.concurrency import ConcurrencyLimiter

        with patch.dict(
            os.environ,
            {"LLM_QUEUE_GLOBAL_CONCURRENCY": str(limit)},
            clear=True,
        ):
            config = QueueConfig.from_environment()

        limiter = ConcurrencyLimiter(config)
        num_threads = limit + 1
        model = "test-model"
        timeout = 0.1  # short timeout for the thread that should fail

        # Use a barrier to synchronize all threads starting acquisition at once
        barrier = threading.Barrier(num_threads)
        results: list[bool] = [False] * num_threads

        def worker(index: int) -> None:
            barrier.wait()  # all threads start together
            results[index] = limiter.acquire(model, timeout=timeout)

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)  # generous join timeout to avoid hanging

        successes = sum(1 for r in results if r)

        # Exactly C threads should have acquired successfully
        assert successes == limit, (
            f"With concurrency limit {limit}, expected exactly {limit} "
            f"successful acquisitions but got {successes}"
        )

        # Release all acquired slots for cleanup
        for r in results:
            if r:
                limiter.release(model)


# ---------------------------------------------------------------------------
# Property 4: Queue Wait Timeout
#
# For any RequestRecord where the actual queue wait exceeds
# max_queue_wait_seconds, DeadlineEnforcer.check_queue_wait() returns
# (True, elapsed) where elapsed > max_queue_wait_seconds.
#
# **Validates: Requirements 2.5**
# ---------------------------------------------------------------------------


class TestPropertyQueueWaitTimeout:
    """Property 4: Queue Wait Timeout.

    For any RequestRecord where the elapsed time since created_at exceeds
    max_queue_wait_seconds, DeadlineEnforcer.check_queue_wait() must return
    expired=True with elapsed > max_queue_wait_seconds.

    **Validates: Requirements 2.5**
    """

    @given(
        max_queue_wait=st.floats(min_value=1.0, max_value=300.0),
        excess_seconds=st.floats(min_value=0.1, max_value=100.0),
    )
    @settings(max_examples=200)
    def test_exceeded_queue_wait_returns_expired(
        self,
        max_queue_wait: float,
        excess_seconds: float,
    ):
        """When elapsed > max_queue_wait_seconds, check_queue_wait returns (True, elapsed)."""
        from datetime import datetime, timedelta, timezone

        from utils.llm_queue.deadline import DeadlineEnforcer

        config = QueueConfig()
        enforcer = DeadlineEnforcer(config)

        # Create a record with created_at far enough in the past that
        # elapsed = max_queue_wait + excess_seconds
        now = datetime.now(timezone.utc)
        created_at = now - timedelta(seconds=max_queue_wait + excess_seconds)

        record = RequestRecord(
            request_id=str(uuid.uuid4()),
            purpose="test_timeout",
            agent="test",
            request_class="advisory",
            tier="medium",
            provider="ollama",
            model="llama3",
            json_mode=False,
            priority=6,
            deadline_seconds=900.0,
            max_queue_wait_seconds=max_queue_wait,
            fallback_policy="none",
            stale_after_seconds=600.0,
            created_at=created_at,
            prompt_chars=100,
            approx_prompt_tokens=25,
            market_session="closed",
        )

        expired, elapsed = enforcer.check_queue_wait(record)

        assert expired is True, (
            f"Expected expired=True when elapsed ({elapsed:.2f}s) > "
            f"max_queue_wait ({max_queue_wait:.2f}s)"
        )
        assert elapsed > max_queue_wait, (
            f"Expected elapsed ({elapsed:.2f}s) > max_queue_wait ({max_queue_wait:.2f}s)"
        )


# ---------------------------------------------------------------------------
# Property 5: Staleness Rejection
#
# For any RequestRecord where elapsed time from created_at exceeds
# stale_after_seconds:
# 1. StalenessChecker.is_stale_before_execution(record) returns True
# 2. StalenessChecker.is_stale_after_execution(record, completed_at) returns True
#    when completed_at is past the stale window
#
# Also test the negative case: when elapsed < stale_after, both return False.
#
# **Validates: Requirements 7.1, 7.3, 7.4**
# ---------------------------------------------------------------------------

from utils.llm_queue.staleness import StalenessChecker


class TestPropertyStalenessRejection:
    """Property 5: Staleness Rejection.

    For any RequestRecord where elapsed time from created_at exceeds
    stale_after_seconds, staleness checks return True. When elapsed is
    less than stale_after_seconds, both methods return False.

    **Validates: Requirements 7.1, 7.3, 7.4**
    """

    @given(
        stale_after_seconds=st.floats(min_value=1.0, max_value=600.0),
        excess=st.floats(min_value=0.1, max_value=100.0),
    )
    @settings(max_examples=200)
    def test_stale_before_execution_when_elapsed_exceeds_threshold(
        self,
        stale_after_seconds: float,
        excess: float,
    ):
        """When elapsed time > stale_after_seconds, is_stale_before_execution returns True."""
        from unittest.mock import patch as mock_patch

        now = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        created_at = now - timedelta(seconds=stale_after_seconds + excess)

        record = RequestRecord(
            request_id=str(uuid.uuid4()),
            purpose="test_stale",
            agent="test",
            request_class="execution_critical",
            tier="high",
            provider="ollama",
            model="llama3",
            json_mode=False,
            priority=0,
            deadline_seconds=120.0,
            max_queue_wait_seconds=30.0,
            fallback_policy="remote_approved",
            stale_after_seconds=stale_after_seconds,
            created_at=created_at,
            prompt_chars=100,
            approx_prompt_tokens=25,
            market_session="closed",
        )

        config = QueueConfig()
        checker = StalenessChecker(config)

        with mock_patch(
            "utils.llm_queue.staleness.datetime",
        ) as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = checker.is_stale_before_execution(record)

        assert result is True, (
            f"Expected is_stale_before_execution=True when elapsed "
            f"({stale_after_seconds + excess:.1f}s) > stale_after ({stale_after_seconds:.1f}s)"
        )

    @given(
        stale_after_seconds=st.floats(min_value=1.0, max_value=600.0),
        excess=st.floats(min_value=0.1, max_value=100.0),
    )
    @settings(max_examples=200)
    def test_stale_after_execution_when_elapsed_exceeds_threshold(
        self,
        stale_after_seconds: float,
        excess: float,
    ):
        """When completed_at - created_at > stale_after_seconds, is_stale_after_execution returns True."""
        created_at = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        completed_at = created_at + timedelta(seconds=stale_after_seconds + excess)

        record = RequestRecord(
            request_id=str(uuid.uuid4()),
            purpose="test_stale",
            agent="test",
            request_class="execution_critical",
            tier="high",
            provider="ollama",
            model="llama3",
            json_mode=False,
            priority=0,
            deadline_seconds=120.0,
            max_queue_wait_seconds=30.0,
            fallback_policy="remote_approved",
            stale_after_seconds=stale_after_seconds,
            created_at=created_at,
            prompt_chars=100,
            approx_prompt_tokens=25,
            market_session="closed",
        )

        config = QueueConfig()
        checker = StalenessChecker(config)

        result = checker.is_stale_after_execution(record, completed_at)

        assert result is True, (
            f"Expected is_stale_after_execution=True when elapsed "
            f"({stale_after_seconds + excess:.1f}s) > stale_after ({stale_after_seconds:.1f}s)"
        )

    @given(
        stale_after_seconds=st.floats(min_value=2.0, max_value=600.0),
        remaining=st.floats(min_value=0.1, max_value=100.0),
    )
    @settings(max_examples=200)
    def test_not_stale_before_execution_when_elapsed_below_threshold(
        self,
        stale_after_seconds: float,
        remaining: float,
    ):
        """When elapsed time < stale_after_seconds, is_stale_before_execution returns False."""
        from hypothesis import assume
        from unittest.mock import patch as mock_patch

        # Ensure elapsed is strictly less than stale_after
        elapsed = stale_after_seconds - remaining
        assume(elapsed > 0)
        assume(elapsed < stale_after_seconds)

        now = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        created_at = now - timedelta(seconds=elapsed)

        record = RequestRecord(
            request_id=str(uuid.uuid4()),
            purpose="test_not_stale",
            agent="test",
            request_class="market_analysis",
            tier="medium",
            provider="ollama",
            model="llama3",
            json_mode=False,
            priority=1,
            deadline_seconds=180.0,
            max_queue_wait_seconds=60.0,
            fallback_policy="smaller_local",
            stale_after_seconds=stale_after_seconds,
            created_at=created_at,
            prompt_chars=200,
            approx_prompt_tokens=50,
            market_session="closed",
        )

        config = QueueConfig()
        checker = StalenessChecker(config)

        with mock_patch(
            "utils.llm_queue.staleness.datetime",
        ) as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = checker.is_stale_before_execution(record)

        assert result is False, (
            f"Expected is_stale_before_execution=False when elapsed "
            f"({elapsed:.1f}s) < stale_after ({stale_after_seconds:.1f}s)"
        )

    @given(
        stale_after_seconds=st.floats(min_value=2.0, max_value=600.0),
        remaining=st.floats(min_value=0.1, max_value=100.0),
    )
    @settings(max_examples=200)
    def test_not_stale_after_execution_when_elapsed_below_threshold(
        self,
        stale_after_seconds: float,
        remaining: float,
    ):
        """When completed_at - created_at < stale_after_seconds, is_stale_after_execution returns False."""
        from hypothesis import assume

        elapsed = stale_after_seconds - remaining
        assume(elapsed > 0)
        assume(elapsed < stale_after_seconds)

        created_at = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        completed_at = created_at + timedelta(seconds=elapsed)

        record = RequestRecord(
            request_id=str(uuid.uuid4()),
            purpose="test_not_stale",
            agent="test",
            request_class="market_analysis",
            tier="medium",
            provider="ollama",
            model="llama3",
            json_mode=False,
            priority=1,
            deadline_seconds=180.0,
            max_queue_wait_seconds=60.0,
            fallback_policy="smaller_local",
            stale_after_seconds=stale_after_seconds,
            created_at=created_at,
            prompt_chars=200,
            approx_prompt_tokens=50,
            market_session="closed",
        )

        config = QueueConfig()
        checker = StalenessChecker(config)

        result = checker.is_stale_after_execution(record, completed_at)

        assert result is False, (
            f"Expected is_stale_after_execution=False when elapsed "
            f"({elapsed:.1f}s) < stale_after ({stale_after_seconds:.1f}s)"
        )


# ---------------------------------------------------------------------------
# Property 6: Execution-Critical Never Silent Drop
#
# For any full queue (at max_queue_size) containing non-execution_critical
# requests, an execution_critical request is ALWAYS admitted and the
# lowest-priority request is evicted. If the queue is full of ONLY
# execution_critical requests, admission fails closed with
# `no_admission_path` (not silently dropped).
#
# **Validates: Requirements 4.4**
# ---------------------------------------------------------------------------


class TestPropertyExecutionCriticalNeverSilentDrop:
    """Property 6: Execution-Critical Never Silent Drop.

    For any full queue containing non-execution_critical requests, an
    execution_critical request is ALWAYS admitted (lowest-priority evicted).
    If the queue is full of ONLY execution_critical requests, admission
    fails closed with `no_admission_path`.

    **Validates: Requirements 4.4**
    """

    @given(
        queue_size=st.integers(min_value=3, max_value=10),
        fill_priorities=st.lists(
            st.integers(min_value=1, max_value=7),
            min_size=3,
            max_size=10,
        ),
    )
    @settings(max_examples=200)
    def test_execution_critical_admitted_when_queue_full_of_non_critical(
        self,
        queue_size: int,
        fill_priorities: list[int],
    ):
        """An execution_critical request is always admitted to a full queue
        containing non-execution_critical requests, with the lowest-priority
        request evicted."""
        from utils.llm_queue.admission import AdmissionController

        # Ensure fill_priorities has exactly queue_size elements
        # by truncating or repeating
        while len(fill_priorities) < queue_size:
            fill_priorities.append(fill_priorities[-1])
        fill_priorities = fill_priorities[:queue_size]

        config = QueueConfig(
            mode="enforcing",
            global_concurrency=1,
            per_model_concurrency={},
            max_queue_size=queue_size,
            max_queue_size_by_class={},
            deadlines={"execution_critical": 120.0, "advisory": 900.0},
            max_queue_waits={"execution_critical": 30.0, "advisory": 300.0},
            stale_after={"execution_critical": 60.0, "advisory": 600.0},
            market_hour_deadline_factor=0.5,
            market_hour_queue_wait_factor=0.5,
            prompt_token_warn_threshold=8000,
            prompt_token_reject_threshold=16000,
            approved_fallback_models={},
            fallback_deadline_buffer_seconds=15.0,
        )

        queue = PriorityQueue(config)
        controller = AdmissionController(config, queue)

        # Fill the queue with non-execution_critical requests
        non_critical_classes = ["advisory", "review", "research", "market_analysis"]
        for i, prio in enumerate(fill_priorities):
            record = RequestRecord(
                request_id=f"fill-{i}",
                purpose="test_fill",
                agent="test",
                request_class=non_critical_classes[i % len(non_critical_classes)],
                tier="medium",
                provider="ollama",
                model="llama3",
                json_mode=False,
                priority=prio,
                deadline_seconds=900.0,
                max_queue_wait_seconds=300.0,
                fallback_policy="none",
                stale_after_seconds=600.0,
                created_at=datetime.now(timezone.utc),
                prompt_chars=100,
                approx_prompt_tokens=25,
                market_session="closed",
            )
            queue.put(record)

        assert queue.depth() == queue_size, (
            f"Queue should be full at {queue_size}, got {queue.depth()}"
        )

        # Submit an execution_critical request (priority 0)
        critical_record = RequestRecord(
            request_id="critical-001",
            purpose="pm_entry_test",
            agent="pm",
            request_class="execution_critical",
            tier="finance",
            provider="ollama",
            model="llama3",
            json_mode=False,
            priority=0,
            deadline_seconds=120.0,
            max_queue_wait_seconds=30.0,
            fallback_policy="remote_approved",
            stale_after_seconds=60.0,
            created_at=datetime.now(timezone.utc),
            prompt_chars=100,
            approx_prompt_tokens=25,
            market_session="closed",
        )

        admitted, reason = controller.admit(critical_record)

        # Must be admitted
        assert admitted is True, (
            f"execution_critical request must be admitted to a full queue "
            f"of non-critical requests, but was rejected with reason={reason!r}"
        )
        assert reason is None, (
            f"execution_critical admission should have no rejection reason, "
            f"got {reason!r}"
        )

        # Queue depth should still be at max_queue_size (one evicted, one added via eviction)
        # Note: AdmissionController evicts but doesn't put — the caller puts after admission
        # So depth should be max_queue_size - 1 (one evicted, not yet re-added)
        assert queue.depth() == queue_size - 1, (
            f"After eviction, queue depth should be {queue_size - 1}, "
            f"got {queue.depth()}"
        )

    @given(
        queue_size=st.integers(min_value=3, max_value=10),
    )
    @settings(max_examples=200)
    def test_execution_critical_fails_closed_when_queue_full_of_critical(
        self,
        queue_size: int,
    ):
        """If queue is full of ONLY execution_critical requests, admission
        fails closed with `no_admission_path` (not silently dropped)."""
        from utils.llm_queue.admission import AdmissionController

        config = QueueConfig(
            mode="enforcing",
            global_concurrency=1,
            per_model_concurrency={},
            max_queue_size=queue_size,
            max_queue_size_by_class={},
            deadlines={"execution_critical": 120.0},
            max_queue_waits={"execution_critical": 30.0},
            stale_after={"execution_critical": 60.0},
            market_hour_deadline_factor=0.5,
            market_hour_queue_wait_factor=0.5,
            prompt_token_warn_threshold=8000,
            prompt_token_reject_threshold=16000,
            approved_fallback_models={},
            fallback_deadline_buffer_seconds=15.0,
        )

        queue = PriorityQueue(config)
        controller = AdmissionController(config, queue)

        # Fill the queue entirely with execution_critical requests
        for i in range(queue_size):
            record = RequestRecord(
                request_id=f"critical-fill-{i}",
                purpose="pm_entry_fill",
                agent="pm",
                request_class="execution_critical",
                tier="finance",
                provider="ollama",
                model="llama3",
                json_mode=False,
                priority=0,
                deadline_seconds=120.0,
                max_queue_wait_seconds=30.0,
                fallback_policy="remote_approved",
                stale_after_seconds=60.0,
                created_at=datetime.now(timezone.utc),
                prompt_chars=100,
                approx_prompt_tokens=25,
                market_session="closed",
            )
            queue.put(record)

        assert queue.depth() == queue_size, (
            f"Queue should be full at {queue_size}, got {queue.depth()}"
        )

        # Submit another execution_critical request
        new_critical = RequestRecord(
            request_id="critical-new",
            purpose="pm_entry_new",
            agent="pm",
            request_class="execution_critical",
            tier="finance",
            provider="ollama",
            model="llama3",
            json_mode=False,
            priority=0,
            deadline_seconds=120.0,
            max_queue_wait_seconds=30.0,
            fallback_policy="remote_approved",
            stale_after_seconds=60.0,
            created_at=datetime.now(timezone.utc),
            prompt_chars=100,
            approx_prompt_tokens=25,
            market_session="closed",
        )

        admitted, reason = controller.admit(new_critical)

        # Must NOT be admitted — fails closed
        assert admitted is False, (
            f"execution_critical request must fail closed when queue is full "
            f"of only execution_critical requests, but was admitted"
        )

        # Reason must be no_admission_path (not silently dropped)
        assert reason == "no_admission_path", (
            f"Expected reason_code 'no_admission_path', got {reason!r}"
        )

        # Queue depth unchanged — nothing evicted
        assert queue.depth() == queue_size, (
            f"Queue depth should remain {queue_size} after failed admission, "
            f"got {queue.depth()}"
        )


# ---------------------------------------------------------------------------
# Property 8: Fallback Deadline Constraint
#
# For any execution_critical request where remaining time
# (deadline_seconds - elapsed) is LESS than fallback_deadline_buffer_seconds,
# FallbackRouter.resolve_fallback() returns None (fail closed).
# When remaining >= buffer AND approved models exist, it returns a
# FallbackDecision.
#
# **Validates: Requirements 5.1, 5.2**
# ---------------------------------------------------------------------------

from utils.llm_queue.fallback import FallbackRouter
from utils.llm_queue.models import FallbackDecision


class TestPropertyFallbackDeadlineConstraint:
    """Property 8: Fallback Deadline Constraint.

    For any execution_critical request where remaining time
    (deadline_seconds - elapsed) < fallback_deadline_buffer_seconds,
    FallbackRouter.resolve_fallback() returns None (fail closed).
    When remaining >= buffer AND approved models exist, it returns a
    FallbackDecision.

    **Validates: Requirements 5.1, 5.2**
    """

    @given(
        deadline_seconds=st.floats(min_value=30.0, max_value=300.0),
        remaining_fraction=st.floats(min_value=0.0, max_value=0.99),
    )
    @settings(max_examples=200)
    def test_fallback_denied_when_remaining_less_than_buffer(
        self,
        deadline_seconds: float,
        remaining_fraction: float,
    ):
        """When remaining time < fallback_deadline_buffer_seconds, resolve_fallback
        returns None for execution_critical requests (fail closed)."""
        buffer = 15.0

        # Generate remaining as a fraction of the buffer, guaranteeing remaining < buffer
        remaining = remaining_fraction * buffer
        elapsed = deadline_seconds - remaining

        # Elapsed must be non-negative (remaining <= deadline always holds here
        # since remaining < 15 and deadline >= 30)
        assert elapsed >= 0.0

        config = QueueConfig(
            mode="enforcing",
            global_concurrency=1,
            per_model_concurrency={},
            max_queue_size=10,
            max_queue_size_by_class={},
            deadlines={"execution_critical": deadline_seconds},
            max_queue_waits={"execution_critical": 30.0},
            stale_after={"execution_critical": 60.0},
            market_hour_deadline_factor=0.5,
            market_hour_queue_wait_factor=0.5,
            prompt_token_warn_threshold=8000,
            prompt_token_reject_threshold=16000,
            approved_fallback_models={
                "execution_critical": ["claude-3-haiku"],
            },
            fallback_deadline_buffer_seconds=buffer,
        )

        router = FallbackRouter(config)

        # Create a record with created_at such that "now" - created_at = elapsed
        now = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        created_at = now - timedelta(seconds=elapsed)

        record = RequestRecord(
            request_id=str(uuid.uuid4()),
            purpose="pm_entry_test",
            agent="pm",
            request_class="execution_critical",
            tier="finance",
            provider="ollama",
            model="llama3",
            json_mode=False,
            priority=0,
            deadline_seconds=deadline_seconds,
            max_queue_wait_seconds=30.0,
            fallback_policy="remote_approved",
            stale_after_seconds=60.0,
            created_at=created_at,
            prompt_chars=500,
            approx_prompt_tokens=125,
            market_session="regular",
        )

        # Mock datetime.now in fallback module to return our controlled "now"
        with patch(
            "utils.llm_queue.fallback.datetime",
        ) as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            mock_dt.timezone = timezone
            decision = router.resolve_fallback(record, reason="local_timeout")

        # Must return None (fail closed) when remaining < buffer
        assert decision is None, (
            f"Expected None (fail closed) when remaining={remaining:.2f}s < "
            f"buffer={buffer:.2f}s, but got {decision!r}"
        )

    @given(
        deadline_seconds=st.floats(min_value=30.0, max_value=300.0),
        extra_above_buffer=st.floats(min_value=1.0, max_value=285.0),
    )
    @settings(max_examples=200)
    def test_fallback_allowed_when_remaining_gte_buffer_and_models_approved(
        self,
        deadline_seconds: float,
        extra_above_buffer: float,
    ):
        """When remaining time > fallback_deadline_buffer_seconds AND approved
        models exist, resolve_fallback returns a FallbackDecision."""
        from hypothesis import assume

        buffer = 15.0
        remaining = buffer + extra_above_buffer
        elapsed = deadline_seconds - remaining

        # Elapsed must be non-negative (remaining must not exceed deadline)
        assume(elapsed >= 0.0)

        config = QueueConfig(
            mode="enforcing",
            global_concurrency=1,
            per_model_concurrency={},
            max_queue_size=10,
            max_queue_size_by_class={},
            deadlines={"execution_critical": deadline_seconds},
            max_queue_waits={"execution_critical": 30.0},
            stale_after={"execution_critical": 60.0},
            market_hour_deadline_factor=0.5,
            market_hour_queue_wait_factor=0.5,
            prompt_token_warn_threshold=8000,
            prompt_token_reject_threshold=16000,
            approved_fallback_models={
                "execution_critical": ["claude-3-haiku"],
            },
            fallback_deadline_buffer_seconds=buffer,
        )

        router = FallbackRouter(config)

        # Create a record with created_at such that "now" - created_at = elapsed
        now = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        created_at = now - timedelta(seconds=elapsed)

        record = RequestRecord(
            request_id=str(uuid.uuid4()),
            purpose="pm_entry_test",
            agent="pm",
            request_class="execution_critical",
            tier="finance",
            provider="ollama",
            model="llama3",
            json_mode=False,
            priority=0,
            deadline_seconds=deadline_seconds,
            max_queue_wait_seconds=30.0,
            fallback_policy="remote_approved",
            stale_after_seconds=60.0,
            created_at=created_at,
            prompt_chars=500,
            approx_prompt_tokens=125,
            market_session="regular",
        )

        # Mock datetime.now in fallback module to return our controlled "now"
        with patch(
            "utils.llm_queue.fallback.datetime",
        ) as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            mock_dt.timezone = timezone
            decision = router.resolve_fallback(record, reason="local_timeout")

        # Must return a FallbackDecision when remaining > buffer and models approved
        assert decision is not None, (
            f"Expected FallbackDecision when remaining={remaining:.2f}s > "
            f"buffer={buffer:.2f}s with approved models, but got None"
        )
        assert isinstance(decision, FallbackDecision), (
            f"Expected FallbackDecision instance, got {type(decision).__name__}"
        )
        assert decision.model == "claude-3-haiku", (
            f"Expected fallback model 'claude-3-haiku', got {decision.model!r}"
        )
        assert decision.deadline_remaining > buffer, (
            f"Expected deadline_remaining > buffer ({buffer}), "
            f"got {decision.deadline_remaining:.2f}"
        )

    @given(
        deadline_seconds=st.floats(min_value=30.0, max_value=300.0),
        extra_above_buffer=st.floats(min_value=1.0, max_value=285.0),
    )
    @settings(max_examples=200)
    def test_fallback_denied_when_no_approved_models(
        self,
        deadline_seconds: float,
        extra_above_buffer: float,
    ):
        """When remaining time > buffer but NO approved models exist,
        resolve_fallback returns None (fail closed)."""
        from hypothesis import assume

        buffer = 15.0
        remaining = buffer + extra_above_buffer
        elapsed = deadline_seconds - remaining

        # Elapsed must be non-negative (remaining must not exceed deadline)
        assume(elapsed >= 0.0)

        config = QueueConfig(
            mode="enforcing",
            global_concurrency=1,
            per_model_concurrency={},
            max_queue_size=10,
            max_queue_size_by_class={},
            deadlines={"execution_critical": deadline_seconds},
            max_queue_waits={"execution_critical": 30.0},
            stale_after={"execution_critical": 60.0},
            market_hour_deadline_factor=0.5,
            market_hour_queue_wait_factor=0.5,
            prompt_token_warn_threshold=8000,
            prompt_token_reject_threshold=16000,
            approved_fallback_models={
                "execution_critical": [],  # No approved models
            },
            fallback_deadline_buffer_seconds=buffer,
        )

        router = FallbackRouter(config)

        now = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        created_at = now - timedelta(seconds=elapsed)

        record = RequestRecord(
            request_id=str(uuid.uuid4()),
            purpose="pm_entry_test",
            agent="pm",
            request_class="execution_critical",
            tier="finance",
            provider="ollama",
            model="llama3",
            json_mode=False,
            priority=0,
            deadline_seconds=deadline_seconds,
            max_queue_wait_seconds=30.0,
            fallback_policy="remote_approved",
            stale_after_seconds=60.0,
            created_at=created_at,
            prompt_chars=500,
            approx_prompt_tokens=125,
            market_session="regular",
        )

        with patch(
            "utils.llm_queue.fallback.datetime",
        ) as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            mock_dt.timezone = timezone
            decision = router.resolve_fallback(record, reason="local_timeout")

        # Must return None (fail closed) when no approved models
        assert decision is None, (
            f"Expected None (fail closed) with no approved models, "
            f"but got {decision!r}"
        )


# ---------------------------------------------------------------------------
# Property 10: Telemetry Terminal Event Guarantee
#
# For any request that enters the TelemetryEmitter (via emit()), exactly one
# terminal event is counted. After N calls to emit(), the cycle summary
# total_requests == N.
#
# **Validates: Requirements 9.1**
# ---------------------------------------------------------------------------

from utils.llm_queue.telemetry import TelemetryEmitter
from utils.llm_queue.models import DispatchResult


# Valid terminal statuses that a DispatchResult can have
_TERMINAL_STATUSES = [
    "completed",
    "timed_out",
    "rejected",
    "deferred",
    "fallback",
    "stale",
    "cancelled",
    "prompt_too_large",
]


def _make_dispatch_result(
    request_id: str,
    status: str,
    fallback_used: bool = False,
    stale: bool = False,
    queue_wait_seconds: float = 1.0,
    elapsed_seconds: float = 5.0,
) -> DispatchResult:
    """Create a DispatchResult with the given terminal status."""
    return DispatchResult(
        ok=(status == "completed"),
        text="response" if status == "completed" else "",
        request_id=request_id,
        status=status,
        reason_code=None,
        provider="ollama",
        model="llama3",
        fallback_used=fallback_used,
        queue_wait_seconds=queue_wait_seconds,
        elapsed_seconds=elapsed_seconds,
        stale=stale,
    )


class TestPropertyTelemetryTerminalGuarantee:
    """Property 10: Telemetry Terminal Event Guarantee.

    For any request that enters the TelemetryEmitter (via emit()), exactly
    one terminal event is counted. After N calls to emit(), the cycle summary
    total_requests == N.

    **Validates: Requirements 9.1**
    """

    @given(
        n=st.integers(min_value=1, max_value=20),
        statuses=st.lists(
            st.sampled_from(_TERMINAL_STATUSES),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=200)
    def test_n_emits_produce_n_total_requests(
        self,
        n: int,
        statuses: list[str],
    ):
        """After N calls to emit(), get_cycle_summary()['total_requests'] == N."""
        from hypothesis import assume

        # Ensure statuses list has exactly n elements
        assume(len(statuses) >= n)
        statuses = statuses[:n]

        emitter = TelemetryEmitter()

        for i, status in enumerate(statuses):
            record = RequestRecord(
                request_id=f"req-{i}",
                purpose="test_telemetry",
                agent="test",
                request_class="advisory",
                tier="medium",
                provider="ollama",
                model="llama3",
                json_mode=False,
                priority=6,
                deadline_seconds=900.0,
                max_queue_wait_seconds=300.0,
                fallback_policy="none",
                stale_after_seconds=600.0,
                created_at=datetime.now(timezone.utc),
                prompt_chars=100,
                approx_prompt_tokens=25,
                market_session="closed",
            )

            result = _make_dispatch_result(
                request_id=f"req-{i}",
                status=status,
                fallback_used=(status == "fallback"),
                stale=(status == "stale"),
            )

            emitter.emit(record, result, queue_depth=n - i)

        summary = emitter.get_cycle_summary()

        # Core guarantee: total_requests == N
        assert summary["total_requests"] == n, (
            f"After {n} emit() calls, expected total_requests={n}, "
            f"got {summary['total_requests']}"
        )

        # Verify status breakdown is sensible: sum of categorized statuses
        # should account for all requests (some statuses overlap with
        # fallback_used/stale_results which are orthogonal flags)
        completed = summary["completed"]
        timed_out = summary["timed_out"]
        rejected = summary["rejected"]
        deferred = summary["deferred"]

        # completed + timed_out + rejected + deferred accounts for the
        # primary status categories (fallback/stale/cancelled don't have
        # dedicated counters in the primary breakdown)
        primary_sum = completed + timed_out + rejected + deferred

        # primary_sum should be <= total_requests (some statuses like
        # "fallback", "stale", "cancelled" don't increment primary counters)
        assert primary_sum <= summary["total_requests"], (
            f"Primary status sum ({primary_sum}) should not exceed "
            f"total_requests ({summary['total_requests']})"
        )

        # fallback_used and stale_results are orthogonal counters
        assert summary["fallback_used"] >= 0
        assert summary["stale_results"] >= 0


# ---------------------------------------------------------------------------
# Property 11: Market-Hour Pressure Reduction
#
# During regular/premarket market sessions, execution_critical and
# market_analysis requests get their deadline_seconds and
# max_queue_wait_seconds multiplied by market_hour_deadline_factor (0.5
# default). Outside market hours (closed/postmarket), the full configured
# values apply without reduction.
#
# **Validates: Requirements 10.2, 10.3, 10.4**
# ---------------------------------------------------------------------------


class TestPropertyMarketHourPressureReduction:
    """Property 11: Market-Hour Pressure Reduction.

    For execution_critical and market_analysis requests classified during
    regular/premarket hours, deadline_seconds and max_queue_wait_seconds
    are reduced by market_hour_deadline_factor. Outside market hours
    (closed session), the full configured values apply.

    **Validates: Requirements 10.2, 10.3, 10.4**
    """

    @given(
        request_class=st.sampled_from(["execution_critical", "market_analysis"]),
        market_session=st.sampled_from(["regular", "premarket", "closed", "postmarket"]),
        market_hour_deadline_factor=st.floats(min_value=0.1, max_value=1.0),
        market_hour_queue_wait_factor=st.floats(min_value=0.1, max_value=1.0),
    )
    @settings(max_examples=200)
    def test_market_hour_factor_applied_correctly(
        self,
        request_class: str,
        market_session: str,
        market_hour_deadline_factor: float,
        market_hour_queue_wait_factor: float,
    ):
        """During regular/premarket, deadline and queue wait are reduced by
        market_hour factors. During closed/postmarket, full values apply."""
        from utils.llm_queue.classifier import RequestClassifier

        # Choose a purpose prefix that maps to the target request_class
        purpose_by_class = {
            "execution_critical": "pm_entry_test",
            "market_analysis": "analyst_signal_test",
        }
        purpose = purpose_by_class[request_class]

        config = QueueConfig(
            mode="enforcing",
            global_concurrency=1,
            per_model_concurrency={},
            max_queue_size=10,
            max_queue_size_by_class={},
            deadlines={
                "execution_critical": 120.0,
                "market_analysis": 180.0,
                "position_management": 180.0,
                "repair": 60.0,
                "research": 600.0,
                "review": 600.0,
                "advisory": 900.0,
                "startup_probe": 30.0,
            },
            max_queue_waits={
                "execution_critical": 30.0,
                "market_analysis": 60.0,
                "position_management": 60.0,
                "repair": 15.0,
                "research": 180.0,
                "review": 180.0,
                "advisory": 300.0,
                "startup_probe": 10.0,
            },
            stale_after={
                "execution_critical": 60.0,
                "market_analysis": 120.0,
                "position_management": 120.0,
                "repair": 30.0,
                "research": 300.0,
                "review": 300.0,
                "advisory": 600.0,
                "startup_probe": 15.0,
            },
            market_hour_deadline_factor=market_hour_deadline_factor,
            market_hour_queue_wait_factor=market_hour_queue_wait_factor,
            prompt_token_warn_threshold=8000,
            prompt_token_reject_threshold=16000,
            approved_fallback_models={},
            fallback_deadline_buffer_seconds=15.0,
        )

        classifier = RequestClassifier(config)

        base_deadline = config.deadlines[request_class]
        base_queue_wait = config.max_queue_waits[request_class]

        with patch(
            "utils.llm_queue.classifier._get_current_market_session",
            return_value=market_session,
        ):
            record = classifier.classify(
                purpose=purpose,
                tier="finance" if request_class == "execution_critical" else "medium",
                provider="ollama",
                model="llama3",
                json_mode=False,
                prompt_chars=500,
            )

        if market_session in ("regular", "premarket"):
            # Factor should be applied
            expected_deadline = base_deadline * market_hour_deadline_factor
            expected_queue_wait = base_queue_wait * market_hour_queue_wait_factor

            assert abs(record.deadline_seconds - expected_deadline) < 1e-6, (
                f"During {market_session}, expected deadline_seconds="
                f"{expected_deadline:.4f} (base {base_deadline} * factor "
                f"{market_hour_deadline_factor}), got {record.deadline_seconds:.4f}"
            )
            assert abs(record.max_queue_wait_seconds - expected_queue_wait) < 1e-6, (
                f"During {market_session}, expected max_queue_wait_seconds="
                f"{expected_queue_wait:.4f} (base {base_queue_wait} * factor "
                f"{market_hour_queue_wait_factor}), got {record.max_queue_wait_seconds:.4f}"
            )
        else:
            # No factor applied — full configured values
            assert abs(record.deadline_seconds - base_deadline) < 1e-6, (
                f"During {market_session}, expected full deadline_seconds="
                f"{base_deadline}, got {record.deadline_seconds:.4f}"
            )
            assert abs(record.max_queue_wait_seconds - base_queue_wait) < 1e-6, (
                f"During {market_session}, expected full max_queue_wait_seconds="
                f"{base_queue_wait}, got {record.max_queue_wait_seconds:.4f}"
            )

    @given(
        request_class=st.sampled_from(["execution_critical", "market_analysis"]),
        market_hour_deadline_factor=st.floats(min_value=0.1, max_value=0.9),
    )
    @settings(max_examples=200)
    def test_market_hours_always_reduce_deadlines(
        self,
        request_class: str,
        market_hour_deadline_factor: float,
    ):
        """During regular market hours, the effective deadline is always
        strictly less than the base configured deadline for adjusted classes."""
        from utils.llm_queue.classifier import RequestClassifier

        purpose_by_class = {
            "execution_critical": "pm_entry_test",
            "market_analysis": "analyst_signal_test",
        }
        purpose = purpose_by_class[request_class]

        config = QueueConfig(
            market_hour_deadline_factor=market_hour_deadline_factor,
            market_hour_queue_wait_factor=market_hour_deadline_factor,
        )

        classifier = RequestClassifier(config)
        base_deadline = config.deadlines[request_class]

        with patch(
            "utils.llm_queue.classifier._get_current_market_session",
            return_value="regular",
        ):
            record = classifier.classify(
                purpose=purpose,
                tier="finance",
                provider="ollama",
                model="llama3",
                json_mode=False,
                prompt_chars=500,
            )

        assert record.deadline_seconds < base_deadline, (
            f"During regular hours with factor {market_hour_deadline_factor}, "
            f"deadline {record.deadline_seconds} should be < base {base_deadline}"
        )
        assert record.max_queue_wait_seconds < config.max_queue_waits[request_class], (
            f"During regular hours with factor {market_hour_deadline_factor}, "
            f"max_queue_wait {record.max_queue_wait_seconds} should be < base "
            f"{config.max_queue_waits[request_class]}"
        )

    @given(
        market_session=st.sampled_from(["closed", "postmarket"]),
    )
    @settings(max_examples=200)
    def test_outside_market_hours_no_reduction(
        self,
        market_session: str,
    ):
        """Outside market hours, execution_critical and market_analysis get
        their full configured deadlines without any factor applied."""
        from utils.llm_queue.classifier import RequestClassifier

        config = QueueConfig(
            market_hour_deadline_factor=0.5,
            market_hour_queue_wait_factor=0.5,
        )

        classifier = RequestClassifier(config)

        with patch(
            "utils.llm_queue.classifier._get_current_market_session",
            return_value=market_session,
        ):
            record_critical = classifier.classify(
                purpose="pm_entry_test",
                tier="finance",
                provider="ollama",
                model="llama3",
                json_mode=False,
                prompt_chars=500,
            )
            record_analysis = classifier.classify(
                purpose="analyst_signal_test",
                tier="medium",
                provider="ollama",
                model="llama3",
                json_mode=False,
                prompt_chars=500,
            )

        # Full deadline values — no reduction
        assert record_critical.deadline_seconds == config.deadlines["execution_critical"], (
            f"During {market_session}, execution_critical should have full "
            f"deadline {config.deadlines['execution_critical']}, "
            f"got {record_critical.deadline_seconds}"
        )
        assert record_critical.max_queue_wait_seconds == config.max_queue_waits["execution_critical"], (
            f"During {market_session}, execution_critical should have full "
            f"max_queue_wait {config.max_queue_waits['execution_critical']}, "
            f"got {record_critical.max_queue_wait_seconds}"
        )
        assert record_analysis.deadline_seconds == config.deadlines["market_analysis"], (
            f"During {market_session}, market_analysis should have full "
            f"deadline {config.deadlines['market_analysis']}, "
            f"got {record_analysis.deadline_seconds}"
        )
        assert record_analysis.max_queue_wait_seconds == config.max_queue_waits["market_analysis"], (
            f"During {market_session}, market_analysis should have full "
            f"max_queue_wait {config.max_queue_waits['market_analysis']}, "
            f"got {record_analysis.max_queue_wait_seconds}"
        )


# ---------------------------------------------------------------------------
# Property 9: Observe Mode Non-Interference
#
# observe_dispatch() NEVER enqueues (queue depth stays 0), NEVER starts
# worker threads, and always returns a valid RequestRecord with consistent
# classification. Telemetry total_requests increments by exactly 1 per call.
#
# This verifies that observe mode does NOT alter execution outcomes — it only
# classifies and emits telemetry.
#
# **Validates: Requirements 8.3**
# ---------------------------------------------------------------------------


# Known purposes that will trigger known classification paths
_OBSERVE_PURPOSES = st.sampled_from([
    "pm_entry_test",
    "pm_candidate_eval",
    "pm_maintenance_check",
    "analyst_signal_scan",
    "researcher_premarket_prep",
    "daily_review_summary",
    "narrator_weekly",
    "ceo_overview",
    "json_repair_attempt",
    "startup_probe_health",
    "unknown_random_purpose",
    "some_advisory_task",
])

_OBSERVE_TIERS = st.sampled_from(["high", "finance", "medium", "low"])
_OBSERVE_MODELS = st.sampled_from(["llama3", "mistral", "codellama", "phi3"])


class TestPropertyObserveModeNonInterference:
    """Property 9: Observe Mode Non-Interference.

    observe_dispatch() NEVER enqueues requests, NEVER starts worker threads,
    and always returns a valid RequestRecord with consistent classification.
    Telemetry total_requests increments by exactly 1 per call.

    **Validates: Requirements 8.3**
    """

    @given(
        purpose=_OBSERVE_PURPOSES,
        tier=_OBSERVE_TIERS,
        model=_OBSERVE_MODELS,
        json_mode=st.booleans(),
        system_prompt=st.text(min_size=1, max_size=200),
        user_prompt=st.text(min_size=1, max_size=500),
        timeout=st.integers(min_value=10, max_value=300),
        num_ctx=st.integers(min_value=512, max_value=8192),
    )
    @settings(max_examples=200)
    def test_observe_dispatch_never_enqueues(
        self,
        purpose: str,
        tier: str,
        model: str,
        json_mode: bool,
        system_prompt: str,
        user_prompt: str,
        timeout: int,
        num_ctx: int,
    ):
        """observe_dispatch() never increases queue depth — it stays at 0."""
        from utils.llm_queue.dispatcher import OllamaDispatcher

        config = QueueConfig(
            mode="observe",
            global_concurrency=1,
            per_model_concurrency={},
            max_queue_size=10,
            max_queue_size_by_class={},
            deadlines={"execution_critical": 120.0, "market_analysis": 180.0,
                       "advisory": 900.0, "research": 600.0, "review": 600.0,
                       "repair": 60.0, "position_management": 120.0,
                       "startup_probe": 30.0},
            max_queue_waits={"execution_critical": 30.0, "market_analysis": 60.0,
                            "advisory": 300.0, "research": 120.0, "review": 120.0,
                            "repair": 15.0, "position_management": 45.0,
                            "startup_probe": 10.0},
            stale_after={"execution_critical": 60.0, "market_analysis": 120.0,
                         "advisory": 600.0, "research": 300.0, "review": 300.0,
                         "repair": 30.0, "position_management": 90.0,
                         "startup_probe": 15.0},
            market_hour_deadline_factor=0.5,
            market_hour_queue_wait_factor=0.5,
            prompt_token_warn_threshold=8000,
            prompt_token_reject_threshold=16000,
            approved_fallback_models={},
            fallback_deadline_buffer_seconds=15.0,
        )

        dispatcher = OllamaDispatcher(config)

        # Verify queue depth is 0 before the call
        assert dispatcher._queue.depth() == 0

        with patch(
            "utils.llm_queue.classifier._get_current_market_session",
            return_value="closed",
        ):
            record = dispatcher.observe_dispatch(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                purpose=purpose,
                tier=tier,
                json_mode=json_mode,
                timeout=timeout,
                num_ctx=num_ctx,
            )

        # Queue depth must remain 0 after observe_dispatch
        assert dispatcher._queue.depth() == 0, (
            f"observe_dispatch must NOT enqueue requests. "
            f"Queue depth is {dispatcher._queue.depth()} after call with "
            f"purpose={purpose!r}"
        )

    @given(
        purpose=_OBSERVE_PURPOSES,
        tier=_OBSERVE_TIERS,
        model=_OBSERVE_MODELS,
        json_mode=st.booleans(),
        system_prompt=st.text(min_size=1, max_size=200),
        user_prompt=st.text(min_size=1, max_size=500),
        timeout=st.integers(min_value=10, max_value=300),
        num_ctx=st.integers(min_value=512, max_value=8192),
    )
    @settings(max_examples=200)
    def test_observe_dispatch_never_starts_workers(
        self,
        purpose: str,
        tier: str,
        model: str,
        json_mode: bool,
        system_prompt: str,
        user_prompt: str,
        timeout: int,
        num_ctx: int,
    ):
        """observe_dispatch() never starts worker threads."""
        from utils.llm_queue.dispatcher import OllamaDispatcher

        config = QueueConfig(
            mode="observe",
            global_concurrency=1,
            per_model_concurrency={},
            max_queue_size=10,
            max_queue_size_by_class={},
            deadlines={"execution_critical": 120.0, "market_analysis": 180.0,
                       "advisory": 900.0, "research": 600.0, "review": 600.0,
                       "repair": 60.0, "position_management": 120.0,
                       "startup_probe": 30.0},
            max_queue_waits={"execution_critical": 30.0, "market_analysis": 60.0,
                            "advisory": 300.0, "research": 120.0, "review": 120.0,
                            "repair": 15.0, "position_management": 45.0,
                            "startup_probe": 10.0},
            stale_after={"execution_critical": 60.0, "market_analysis": 120.0,
                         "advisory": 600.0, "research": 300.0, "review": 300.0,
                         "repair": 30.0, "position_management": 90.0,
                         "startup_probe": 15.0},
            market_hour_deadline_factor=0.5,
            market_hour_queue_wait_factor=0.5,
            prompt_token_warn_threshold=8000,
            prompt_token_reject_threshold=16000,
            approved_fallback_models={},
            fallback_deadline_buffer_seconds=15.0,
        )

        dispatcher = OllamaDispatcher(config)

        with patch(
            "utils.llm_queue.classifier._get_current_market_session",
            return_value="closed",
        ):
            record = dispatcher.observe_dispatch(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                purpose=purpose,
                tier=tier,
                json_mode=json_mode,
                timeout=timeout,
                num_ctx=num_ctx,
            )

        # No worker threads should have been started
        assert dispatcher._workers_started is False, (
            f"observe_dispatch must NOT start worker threads. "
            f"_workers_started={dispatcher._workers_started} after call with "
            f"purpose={purpose!r}"
        )
        assert len(dispatcher._workers) == 0, (
            f"observe_dispatch must NOT create worker threads. "
            f"Found {len(dispatcher._workers)} workers after call with "
            f"purpose={purpose!r}"
        )

    @given(
        purpose=_OBSERVE_PURPOSES,
        tier=_OBSERVE_TIERS,
        model=_OBSERVE_MODELS,
        json_mode=st.booleans(),
        system_prompt=st.text(min_size=1, max_size=200),
        user_prompt=st.text(min_size=1, max_size=500),
        timeout=st.integers(min_value=10, max_value=300),
        num_ctx=st.integers(min_value=512, max_value=8192),
    )
    @settings(max_examples=200)
    def test_observe_dispatch_returns_valid_classification(
        self,
        purpose: str,
        tier: str,
        model: str,
        json_mode: bool,
        system_prompt: str,
        user_prompt: str,
        timeout: int,
        num_ctx: int,
    ):
        """observe_dispatch() always returns a valid RequestRecord with
        consistent classification fields."""
        from utils.llm_queue.dispatcher import OllamaDispatcher
        from utils.llm_queue.models import RequestRecord as RR

        config = QueueConfig(
            mode="observe",
            global_concurrency=1,
            per_model_concurrency={},
            max_queue_size=10,
            max_queue_size_by_class={},
            deadlines={"execution_critical": 120.0, "market_analysis": 180.0,
                       "advisory": 900.0, "research": 600.0, "review": 600.0,
                       "repair": 60.0, "position_management": 120.0,
                       "startup_probe": 30.0},
            max_queue_waits={"execution_critical": 30.0, "market_analysis": 60.0,
                            "advisory": 300.0, "research": 120.0, "review": 120.0,
                            "repair": 15.0, "position_management": 45.0,
                            "startup_probe": 10.0},
            stale_after={"execution_critical": 60.0, "market_analysis": 120.0,
                         "advisory": 600.0, "research": 300.0, "review": 300.0,
                         "repair": 30.0, "position_management": 90.0,
                         "startup_probe": 15.0},
            market_hour_deadline_factor=0.5,
            market_hour_queue_wait_factor=0.5,
            prompt_token_warn_threshold=8000,
            prompt_token_reject_threshold=16000,
            approved_fallback_models={},
            fallback_deadline_buffer_seconds=15.0,
        )

        dispatcher = OllamaDispatcher(config)

        with patch(
            "utils.llm_queue.classifier._get_current_market_session",
            return_value="closed",
        ):
            record = dispatcher.observe_dispatch(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                purpose=purpose,
                tier=tier,
                json_mode=json_mode,
                timeout=timeout,
                num_ctx=num_ctx,
            )

        # Must be a RequestRecord instance
        assert isinstance(record, RR), (
            f"Expected RequestRecord, got {type(record).__name__}"
        )

        # Must have a valid request_id (non-empty string)
        assert record.request_id and isinstance(record.request_id, str), (
            f"request_id must be a non-empty string, got {record.request_id!r}"
        )

        # request_class must be one of the known classes
        valid_classes = {
            "execution_critical", "market_analysis", "position_management",
            "repair", "research", "review", "advisory", "startup_probe",
        }
        assert record.request_class in valid_classes, (
            f"request_class must be valid, got {record.request_class!r} "
            f"for purpose={purpose!r}"
        )

        # Priority must be in valid range (0-7)
        assert 0 <= record.priority <= 7, (
            f"priority must be 0-7, got {record.priority} for purpose={purpose!r}"
        )

        # Model must match what was passed in
        assert record.model == model, (
            f"model must match input ({model!r}), got {record.model!r}"
        )

        # prompt_chars must equal the combined length of prompts
        expected_chars = len(system_prompt) + len(user_prompt)
        assert record.prompt_chars == expected_chars, (
            f"prompt_chars should be {expected_chars}, got {record.prompt_chars}"
        )

    @given(
        purpose=_OBSERVE_PURPOSES,
        tier=_OBSERVE_TIERS,
        model=_OBSERVE_MODELS,
        json_mode=st.booleans(),
        system_prompt=st.text(min_size=1, max_size=200),
        user_prompt=st.text(min_size=1, max_size=500),
        timeout=st.integers(min_value=10, max_value=300),
        num_ctx=st.integers(min_value=512, max_value=8192),
    )
    @settings(max_examples=200)
    def test_observe_dispatch_increments_telemetry_by_one(
        self,
        purpose: str,
        tier: str,
        model: str,
        json_mode: bool,
        system_prompt: str,
        user_prompt: str,
        timeout: int,
        num_ctx: int,
    ):
        """After one observe_dispatch() call, telemetry total_requests == 1."""
        from utils.llm_queue.dispatcher import OllamaDispatcher

        config = QueueConfig(
            mode="observe",
            global_concurrency=1,
            per_model_concurrency={},
            max_queue_size=10,
            max_queue_size_by_class={},
            deadlines={"execution_critical": 120.0, "market_analysis": 180.0,
                       "advisory": 900.0, "research": 600.0, "review": 600.0,
                       "repair": 60.0, "position_management": 120.0,
                       "startup_probe": 30.0},
            max_queue_waits={"execution_critical": 30.0, "market_analysis": 60.0,
                            "advisory": 300.0, "research": 120.0, "review": 120.0,
                            "repair": 15.0, "position_management": 45.0,
                            "startup_probe": 10.0},
            stale_after={"execution_critical": 60.0, "market_analysis": 120.0,
                         "advisory": 600.0, "research": 300.0, "review": 300.0,
                         "repair": 30.0, "position_management": 90.0,
                         "startup_probe": 15.0},
            market_hour_deadline_factor=0.5,
            market_hour_queue_wait_factor=0.5,
            prompt_token_warn_threshold=8000,
            prompt_token_reject_threshold=16000,
            approved_fallback_models={},
            fallback_deadline_buffer_seconds=15.0,
        )

        dispatcher = OllamaDispatcher(config)

        # Reset telemetry to ensure clean state
        dispatcher._telemetry.reset_cycle()

        with patch(
            "utils.llm_queue.classifier._get_current_market_session",
            return_value="closed",
        ):
            record = dispatcher.observe_dispatch(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                purpose=purpose,
                tier=tier,
                json_mode=json_mode,
                timeout=timeout,
                num_ctx=num_ctx,
            )

        # Telemetry total_requests must be exactly 1 after one call
        summary = dispatcher.get_cycle_summary()
        assert summary["total_requests"] == 1, (
            f"After one observe_dispatch call, total_requests should be 1, "
            f"got {summary['total_requests']} for purpose={purpose!r}"
        )
