"""Runtime resource telemetry helpers.

The paper-trader scheduler can run several market jobs at the same timestamp.
These helpers are intentionally fail-open so observability never breaks trading.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

# `resource` is POSIX-only. It is present on the macOS and Linux hosts this runs
# on, so nothing below changes there. The guard exists so that importing this
# module — and therefore orchestrator.py, which imports it at module scope — does
# not hard-fail on Windows during development. Every other function here already
# degrades on platforms without the relevant facility (_fd_dir() returns None
# when neither /dev/fd nor /proc/self/fd exists); this keeps the import itself
# consistent with that.
try:
    import resource
except ImportError:  # pragma: no cover - only taken on non-POSIX platforms
    resource = None  # type: ignore[assignment]


def _fd_dir() -> str | None:
    for path in ("/dev/fd", "/proc/self/fd"):
        if os.path.isdir(path):
            return path
    return None


def get_open_fd_count() -> int | None:
    """Return this process's numeric open file descriptor count, if available."""
    path = _fd_dir()
    if not path:
        return None

    try:
        count = 0
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.name.isdigit():
                    count += 1
        return count
    except OSError:
        return None


def get_fd_limit() -> tuple[int | None, int | None]:
    """Return the soft/hard RLIMIT_NOFILE values for this process.

    Returns ``(None, None)`` where RLIMIT_NOFILE is unavailable, which callers
    already handle: ``log_fd_snapshot()`` omits None values from its output and
    skips the usage-percentage calculation.
    """
    if resource is None:
        return None, None

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError, AttributeError):
        return None, None

    def normalize(value: int) -> int | None:
        return None if value == resource.RLIM_INFINITY else int(value)

    return normalize(soft), normalize(hard)


def _fmt_kv(items: dict[str, Any]) -> str:
    parts = []
    for key, value in items.items():
        if value is None:
            continue
        text = str(value).replace("\n", " ")
        parts.append(f"{key}={text}")
    return " ".join(parts)


def log_fd_snapshot(
    logger: logging.Logger,
    *,
    event: str,
    label: str,
    job_id: str | None = None,
    cycle_id: str | None = None,
    phase: str | None = None,
    status: str | None = None,
    duration_ms: float | None = None,
    fd_start: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit a single structured FD telemetry log line.

    The line is key=value text instead of JSON to match existing logs and make
    `rg FD_TELEMETRY` useful during incidents.
    """
    try:
        fd_count = get_open_fd_count()
        soft, hard = get_fd_limit()
        usage_pct = None
        if fd_count is not None and soft and soft > 0:
            usage_pct = round((fd_count / soft) * 100, 1)

        payload: dict[str, Any] = {
            "event": event,
            "label": label,
            "job_id": job_id,
            "cycle_id": cycle_id,
            "phase": phase,
            "status": status,
            "fd_count": fd_count,
            "fd_soft_limit": soft,
            "fd_hard_limit": hard,
            "fd_usage_pct": usage_pct,
            "fd_delta": (
                fd_count - fd_start
                if fd_count is not None and fd_start is not None
                else None
            ),
            "duration_ms": round(duration_ms, 1) if duration_ms is not None else None,
        }
        if extra:
            payload.update(extra)

        level = logging.WARNING if usage_pct is not None and usage_pct >= 80 else logging.INFO
        logger.log(level, "FD_TELEMETRY %s", _fmt_kv(payload))
    except Exception:
        logger.debug("FD telemetry snapshot failed", exc_info=True)


@contextmanager
def fd_trace(
    logger: logging.Logger,
    *,
    label: str,
    job_id: str | None = None,
    cycle_id: str | None = None,
    phase: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Log start/end/error FD snapshots for a scoped operation."""
    start_fd = get_open_fd_count()
    started = time.monotonic()
    log_fd_snapshot(
        logger,
        event="start",
        label=label,
        job_id=job_id,
        cycle_id=cycle_id,
        phase=phase,
        fd_start=start_fd,
        extra=extra,
    )
    try:
        yield
    except Exception:
        log_fd_snapshot(
            logger,
            event="end",
            label=label,
            job_id=job_id,
            cycle_id=cycle_id,
            phase=phase,
            status="error",
            duration_ms=(time.monotonic() - started) * 1000,
            fd_start=start_fd,
            extra=extra,
        )
        raise
    else:
        log_fd_snapshot(
            logger,
            event="end",
            label=label,
            job_id=job_id,
            cycle_id=cycle_id,
            phase=phase,
            status="completed",
            duration_ms=(time.monotonic() - started) * 1000,
            fd_start=start_fd,
            extra=extra,
        )
