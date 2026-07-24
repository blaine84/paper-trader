"""Cycle Logger — structured observability for coordinated market cycles.

Provides PhaseLog, CycleSummary dataclasses and a CycleLogger class that
tracks phase timing, freshness decisions, provider usage, and skip reasons.
All log output via logging.getLogger("cycle_coordinator") with structured fields.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("cycle_coordinator")


@dataclass(frozen=True)
class PhaseLog:
    """Immutable record of a single phase execution."""

    phase_name: str
    started_at: datetime
    ended_at: datetime
    duration_seconds: float
    symbols_processed: int
    status: str  # "completed" | "timeout" | "error"
    details: dict[str, Any]


@dataclass(frozen=True)
class CycleSummary:
    """Immutable summary of a complete market cycle."""

    cycle_id: str
    trigger_source: str
    total_duration_seconds: float
    phases: tuple[PhaseLog, ...]
    symbols_fresh: int
    symbols_stale_skipped: int
    symbols_data_unavailable: int
    provider_calls_total: int
    provider_calls_by_phase: dict[str, int]
    rate_limit_events: int


class CycleLogger:
    """Structured logging for market cycle phases and summary.

    Tracks internal state (current phase start time, accumulated phases,
    freshness counters) to build the CycleSummary at cycle end.
    """

    def __init__(self, cycle_id: str) -> None:
        self._cycle_id = cycle_id
        self._phases: list[PhaseLog] = []
        self._cycle_started_at: datetime | None = None
        self._trigger_source: str = ""
        self._current_phase_name: str | None = None
        self._current_phase_started_at: datetime | None = None
        self._symbols_fresh: int = 0
        self._symbols_stale_skipped: int = 0
        self._symbols_data_unavailable: int = 0
        self._provider_calls_total: int = 0
        self._provider_calls_by_phase: dict[str, int] = {}
        self._rate_limit_events: int = 0

    def log_cycle_start(self, ctx: Any) -> None:
        """Log cycle start with context details.

        Args:
            ctx: A CycleContext-like object with attributes:
                 cycle_id, trigger_source, focused_symbols, phases (optional).
        """
        self._cycle_started_at = datetime.now(timezone.utc)
        self._trigger_source = getattr(ctx, "trigger_source", "unknown")

        focused_symbols = getattr(ctx, "focused_symbols", ())
        logger.info(
            "Cycle started",
            extra={
                "cycle_id": self._cycle_id,
                "trigger_source": self._trigger_source,
                "focused_symbols": list(focused_symbols),
                "symbol_count": len(focused_symbols),
            },
        )

    def log_phase_start(self, phase_name: str) -> None:
        """Record the start of a new phase.

        Args:
            phase_name: Name of the phase starting (e.g., "analyst_refresh").
        """
        self._current_phase_name = phase_name
        self._current_phase_started_at = datetime.now(timezone.utc)

        logger.info(
            "Phase started: %s",
            phase_name,
            extra={
                "cycle_id": self._cycle_id,
                "phase_name": phase_name,
                "phase_start_time": self._current_phase_started_at.isoformat(),
            },
        )

    def log_phase_complete(
        self,
        phase_name: str,
        *,
        symbols_processed: int = 0,
        status: str = "completed",
        provider_calls: int = 0,
        **details: Any,
    ) -> None:
        """Record phase completion and build a PhaseLog entry.

        Args:
            phase_name: Name of the completed phase.
            symbols_processed: Number of symbols processed in this phase.
            status: Phase outcome — "completed", "timeout", or "error".
            provider_calls: Number of provider API calls made during this phase.
            **details: Additional phase-specific details to store.
        """
        ended_at = datetime.now(timezone.utc)

        # Use tracked start time if available, otherwise use ended_at as fallback
        if self._current_phase_name == phase_name and self._current_phase_started_at is not None:
            started_at = self._current_phase_started_at
        else:
            started_at = ended_at

        duration_seconds = (ended_at - started_at).total_seconds()

        phase_log = PhaseLog(
            phase_name=phase_name,
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=duration_seconds,
            symbols_processed=symbols_processed,
            status=status,
            details=details,
        )
        self._phases.append(phase_log)

        # Track provider calls
        if provider_calls > 0:
            self._provider_calls_total += provider_calls
            self._provider_calls_by_phase[phase_name] = (
                self._provider_calls_by_phase.get(phase_name, 0) + provider_calls
            )

        logger.info(
            "Phase completed: %s (%s) — %.2fs, %d symbols",
            phase_name,
            status,
            duration_seconds,
            symbols_processed,
            extra={
                "cycle_id": self._cycle_id,
                "phase_name": phase_name,
                "phase_start_time": started_at.isoformat(),
                "phase_end_time": ended_at.isoformat(),
                "duration_seconds": duration_seconds,
                "symbols_processed": symbols_processed,
                "status": status,
                "provider_calls": provider_calls,
                "details": details,
            },
        )

        # Reset current phase tracking
        self._current_phase_name = None
        self._current_phase_started_at = None

    def log_freshness_skip(self, skip: Any) -> None:
        """Log a symbol skipped due to stale or missing signal.

        Args:
            skip: A StaleSignalSkip-like object with attributes:
                  cycle_id, symbol, signal_age_seconds,
                  freshness_threshold_seconds, signal_timestamp, reason.
        """
        reason = getattr(skip, "reason", "unknown")

        if reason in ("stale_signal", "previous_cycle"):
            self._symbols_stale_skipped += 1
        elif reason == "missing_signal":
            self._symbols_data_unavailable += 1
        elif reason == "freshness_gate_error":
            self._symbols_data_unavailable += 1

        logger.info(
            "Freshness skip: %s — %s (age=%.1fs, threshold=%ds)",
            getattr(skip, "symbol", "unknown"),
            reason,
            getattr(skip, "signal_age_seconds", 0.0),
            getattr(skip, "freshness_threshold_seconds", 0),
            extra={
                "cycle_id": self._cycle_id,
                "symbol": getattr(skip, "symbol", "unknown"),
                "reason": reason,
                "signal_age_seconds": getattr(skip, "signal_age_seconds", 0.0),
                "freshness_threshold_seconds": getattr(skip, "freshness_threshold_seconds", 0),
                "signal_timestamp": str(getattr(skip, "signal_timestamp", None)),
            },
        )

    def log_rate_limit_event(self) -> None:
        """Record a provider rate-limit (429) event during the cycle."""
        self._rate_limit_events += 1

    def set_symbols_fresh(self, count: int) -> None:
        """Set the count of symbols that passed the freshness gate."""
        self._symbols_fresh = count

    def log_cycle_end(self) -> CycleSummary:
        """Finalize the cycle and return a complete CycleSummary.

        Returns:
            CycleSummary with all accumulated phase logs and counters.
        """
        ended_at = datetime.now(timezone.utc)

        if self._cycle_started_at is not None:
            total_duration = (ended_at - self._cycle_started_at).total_seconds()
        else:
            total_duration = 0.0

        summary = CycleSummary(
            cycle_id=self._cycle_id,
            trigger_source=self._trigger_source,
            total_duration_seconds=total_duration,
            phases=tuple(self._phases),
            symbols_fresh=self._symbols_fresh,
            symbols_stale_skipped=self._symbols_stale_skipped,
            symbols_data_unavailable=self._symbols_data_unavailable,
            provider_calls_total=self._provider_calls_total,
            provider_calls_by_phase=dict(self._provider_calls_by_phase),
            rate_limit_events=self._rate_limit_events,
        )

        logger.info(
            "Cycle completed: %s — %.2fs total, %d phases, "
            "%d fresh, %d stale skipped, %d unavailable, "
            "%d provider calls, %d rate-limit events",
            self._cycle_id,
            total_duration,
            len(self._phases),
            self._symbols_fresh,
            self._symbols_stale_skipped,
            self._symbols_data_unavailable,
            self._provider_calls_total,
            self._rate_limit_events,
            extra={
                "cycle_id": self._cycle_id,
                "trigger_source": self._trigger_source,
                "total_duration_seconds": total_duration,
                "phases_completed": len(self._phases),
                "symbols_fresh": self._symbols_fresh,
                "symbols_stale_skipped": self._symbols_stale_skipped,
                "symbols_data_unavailable": self._symbols_data_unavailable,
                "provider_calls_total": self._provider_calls_total,
                "provider_calls_by_phase": self._provider_calls_by_phase,
                "rate_limit_events": self._rate_limit_events,
            },
        )

        return summary
