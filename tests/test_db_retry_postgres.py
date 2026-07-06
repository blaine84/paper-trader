"""Unit tests for db_retry Postgres error handling.

Tests that `is_retryable_error()` correctly identifies Postgres transient errors
(serialization_failure, deadlock_detected) as retryable, rejects non-transient
Postgres errors, and that the `@with_lock_retry` decorator behaves correctly
for both cases.

Validates: Requirements 2.1, 2.2
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError as SAOperationalError

from utils.db_retry import is_retryable_error, with_lock_retry


def _make_pg_error(pgcode: str) -> SAOperationalError:
    """Create a mock SQLAlchemy OperationalError with a Postgres pgcode."""
    orig = MagicMock()
    orig.pgcode = pgcode
    exc = SAOperationalError("test", {}, orig)
    return exc


def _make_sqlite_lock_error() -> SAOperationalError:
    """Create a mock SQLAlchemy OperationalError for SQLite lock contention."""
    orig = Exception("database is locked")
    exc = SAOperationalError("database is locked", {}, orig)
    return exc


# ── is_retryable_error() — Postgres retryable codes ─────────────────────────


class TestIsRetryableErrorPostgres:
    """Tests for is_retryable_error() with Postgres error codes."""

    def test_serialization_failure_is_retryable(self):
        """pgcode 40001 (serialization_failure) should be retryable."""
        exc = _make_pg_error("40001")
        assert is_retryable_error(exc) is True

    def test_deadlock_detected_is_retryable(self):
        """pgcode 40P01 (deadlock_detected) should be retryable."""
        exc = _make_pg_error("40P01")
        assert is_retryable_error(exc) is True

    def test_unique_violation_not_retryable(self):
        """pgcode 23505 (unique_violation) should NOT be retryable."""
        exc = _make_pg_error("23505")
        assert is_retryable_error(exc) is False

    def test_undefined_table_not_retryable(self):
        """pgcode 42P01 (undefined_table) should NOT be retryable."""
        exc = _make_pg_error("42P01")
        assert is_retryable_error(exc) is False

    def test_sqlite_lock_contention_still_retryable(self):
        """SQLite lock contention remains retryable (backward compat)."""
        exc = _make_sqlite_lock_error()
        assert is_retryable_error(exc) is True


# ── @with_lock_retry — Postgres retry behavior ──────────────────────────────


class TestWithLockRetryPostgres:
    """Tests for @with_lock_retry decorator with Postgres errors."""

    @patch("utils.db_retry.time.sleep", return_value=None)
    def test_retries_on_serialization_failure_then_succeeds(self, mock_sleep):
        """Decorator retries on pgcode 40001 and succeeds on third attempt."""
        call_count = 0

        @with_lock_retry
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise _make_pg_error("40001")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 3
        # Two retries means two sleep calls
        assert mock_sleep.call_count == 2

    @patch("utils.db_retry.time.sleep", return_value=None)
    def test_does_not_retry_on_non_retryable_postgres_error(self, mock_sleep):
        """Decorator raises immediately on non-retryable pgcode (23505)."""
        call_count = 0

        @with_lock_retry
        def failing_func():
            nonlocal call_count
            call_count += 1
            raise _make_pg_error("23505")

        with pytest.raises(SAOperationalError):
            failing_func()

        # Should have been called exactly once — no retry
        assert call_count == 1
        assert mock_sleep.call_count == 0
