"""Tests for CycleFinnhubBudget — cycle-scoped Finnhub API budget counter."""

from __future__ import annotations

import threading

from utils.finnhub_budget import CycleFinnhubBudget


class TestIncrementWithinBudget:
    """Test: increment returns True within budget."""

    def test_single_increment_within_budget(self):
        budget = CycleFinnhubBudget(budget=10)
        assert budget.increment() is True
        assert budget.used == 1

    def test_multiple_increments_within_budget(self):
        budget = CycleFinnhubBudget(budget=10)
        for _ in range(10):
            assert budget.increment() is True
        assert budget.used == 10

    def test_increment_with_count_within_budget(self):
        budget = CycleFinnhubBudget(budget=10)
        assert budget.increment(count=5) is True
        assert budget.used == 5


class TestIncrementExhausted:
    """Test: increment returns False when budget exhausted."""

    def test_increment_at_exact_limit_rejected(self):
        budget = CycleFinnhubBudget(budget=5)
        for _ in range(5):
            budget.increment()
        assert budget.increment() is False
        # Used count should NOT increase when rejected
        assert budget.used == 5

    def test_increment_count_exceeding_remaining(self):
        budget = CycleFinnhubBudget(budget=5)
        budget.increment(count=3)
        # Requesting 3 more when only 2 remain
        assert budget.increment(count=3) is False
        assert budget.used == 3

    def test_zero_budget_always_rejects(self):
        budget = CycleFinnhubBudget(budget=0)
        assert budget.increment() is False
        assert budget.used == 0


class TestRemaining:
    """Test: remaining decreases correctly."""

    def test_remaining_starts_at_budget(self):
        budget = CycleFinnhubBudget(budget=40)
        assert budget.remaining() == 40

    def test_remaining_decreases_after_increment(self):
        budget = CycleFinnhubBudget(budget=40)
        budget.increment(count=15)
        assert budget.remaining() == 25

    def test_remaining_is_zero_when_exhausted(self):
        budget = CycleFinnhubBudget(budget=5)
        budget.increment(count=5)
        assert budget.remaining() == 0

    def test_remaining_never_negative(self):
        budget = CycleFinnhubBudget(budget=0)
        assert budget.remaining() == 0


class TestIsExhausted:
    """Test: is_exhausted returns True at limit."""

    def test_not_exhausted_initially(self):
        budget = CycleFinnhubBudget(budget=10)
        assert budget.is_exhausted() is False

    def test_exhausted_at_exact_limit(self):
        budget = CycleFinnhubBudget(budget=5)
        budget.increment(count=5)
        assert budget.is_exhausted() is True

    def test_not_exhausted_when_partially_used(self):
        budget = CycleFinnhubBudget(budget=10)
        budget.increment(count=9)
        assert budget.is_exhausted() is False

    def test_zero_budget_is_immediately_exhausted(self):
        budget = CycleFinnhubBudget(budget=0)
        assert budget.is_exhausted() is True


class TestReset:
    """Test: reset clears the counter."""

    def test_reset_clears_used_count(self):
        budget = CycleFinnhubBudget(budget=10)
        budget.increment(count=7)
        assert budget.used == 7
        budget.reset()
        assert budget.used == 0

    def test_reset_restores_remaining(self):
        budget = CycleFinnhubBudget(budget=10)
        budget.increment(count=10)
        assert budget.remaining() == 0
        budget.reset()
        assert budget.remaining() == 10

    def test_reset_allows_new_increments(self):
        budget = CycleFinnhubBudget(budget=5)
        budget.increment(count=5)
        assert budget.increment() is False
        budget.reset()
        assert budget.increment() is True

    def test_reset_clears_exhausted_state(self):
        budget = CycleFinnhubBudget(budget=5)
        budget.increment(count=5)
        assert budget.is_exhausted() is True
        budget.reset()
        assert budget.is_exhausted() is False


class TestThreadSafety:
    """Test: thread safety (concurrent increments don't exceed budget)."""

    def test_concurrent_increments_respect_budget(self):
        budget_limit = 100
        budget = CycleFinnhubBudget(budget=budget_limit)
        num_threads = 20
        increments_per_thread = 10
        # Total attempts = 200, but budget is only 100

        barrier = threading.Barrier(num_threads)
        results: list[bool] = []
        results_lock = threading.Lock()

        def worker():
            barrier.wait()
            local_results = []
            for _ in range(increments_per_thread):
                local_results.append(budget.increment())
            with results_lock:
                results.extend(local_results)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Budget should never be exceeded
        assert budget.used == budget_limit
        assert budget.used <= budget_limit
        # Exactly budget_limit successes
        assert sum(1 for r in results if r is True) == budget_limit
        # Remaining should be False results
        assert sum(1 for r in results if r is False) == (num_threads * increments_per_thread) - budget_limit

    def test_concurrent_increments_with_varying_counts(self):
        budget = CycleFinnhubBudget(budget=50)
        num_threads = 10
        barrier = threading.Barrier(num_threads)

        def worker():
            barrier.wait()
            for _ in range(10):
                budget.increment(count=1)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Total attempts = 100, budget = 50
        assert budget.used == 50
        assert budget.remaining() == 0
        assert budget.is_exhausted() is True
