"""SQLite Lock Retry — bounded retry for transient lock contention.

Provides a decorator that retries database operations on SQLite lock
contention errors. Only retries on lock-related OperationalErrors;
all other errors (constraint violations, syntax errors, corruption)
raise immediately without retry.

Requirements: 12.3, 12.4, 12.5, 12.6
"""

from __future__ import annotations

import functools
import logging
import time
from sqlite3 import OperationalError as Sqlite3OperationalError

from sqlalchemy.exc import OperationalError as SAOperationalError

logger = logging.getLogger(__name__)

# Messages indicating lock contention (SQLite)
_LOCK_CONTENTION_MESSAGES = (
    "database is locked",
    "database table is locked",
)

# Backoff schedule: 50ms, 100ms, 200ms
_BACKOFF_MS = (50, 100, 200)


def is_lock_contention(exc: Exception) -> bool:
    """Check if an exception represents SQLite lock contention.

    Returns True for OperationalError (sqlite3 or SQLAlchemy) with
    lock-related messages. Returns False for all other errors
    (constraint violations, syntax errors, corruption).
    """
    if isinstance(exc, SAOperationalError):
        # SQLAlchemy wraps the original — check both the wrapper and orig
        msg = str(exc).lower()
        if any(pattern in msg for pattern in _LOCK_CONTENTION_MESSAGES):
            return True
        # Also check the wrapped original exception
        if exc.orig is not None:
            orig_msg = str(exc.orig).lower()
            return any(pattern in orig_msg for pattern in _LOCK_CONTENTION_MESSAGES)
        return False
    if isinstance(exc, Sqlite3OperationalError):
        msg = str(exc).lower()
        return any(pattern in msg for pattern in _LOCK_CONTENTION_MESSAGES)
    return False


def with_lock_retry(func=None, *, max_retries: int = 3):
    """Decorator: retry on SQLite lock contention with bounded backoff.

    Retries the decorated function up to max_retries times when it raises
    an OperationalError containing a lock contention message.

    Non-lock errors (IntegrityError, ProgrammingError, etc.) raise immediately.

    Backoff schedule: 50ms, 100ms, 200ms (capped at max_retries).

    Usage:
        @with_lock_retry
        def my_db_write():
            ...

        @with_lock_retry(max_retries=5)
        def my_db_write():
            ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):  # 0 = first try, 1..N = retries
                try:
                    return fn(*args, **kwargs)
                except (SAOperationalError, Sqlite3OperationalError) as exc:
                    if not is_lock_contention(exc):
                        # Not lock contention — raise immediately, no retry
                        raise
                    last_exc = exc
                    if attempt < max_retries:
                        wait_ms = _BACKOFF_MS[min(attempt, len(_BACKOFF_MS) - 1)]
                        logger.warning(
                            "Lock contention retry: func=%s attempt=%d/%d wait_ms=%d error=%s",
                            fn.__qualname__,
                            attempt + 1,
                            max_retries,
                            wait_ms,
                            str(exc),
                        )
                        time.sleep(wait_ms / 1000.0)
                    else:
                        # All retries exhausted
                        logger.error(
                            "Lock contention retries exhausted: func=%s attempts=%d error=%s",
                            fn.__qualname__,
                            max_retries,
                            str(exc),
                        )
                        raise
            # Should not reach here, but safety net
            if last_exc:
                raise last_exc  # pragma: no cover
        return wrapper

    if func is not None:
        # Called as @with_lock_retry without parentheses
        return decorator(func)
    return decorator
