"""Cycle Coordinator — orchestrates coordinated market cycles.

Provides CycleContext (immutable context for a single cycle), CycleCoordinator
(phase-based orchestrator), and cycle_id generation.

The coordinator owns phase ordering, timeout advancement, freshness gating,
and late-write protection. It does NOT own trade logic, signal generation,
or position sizing.

Requirements: 1.1, 1.2, 1.3, 1.6, 2.1, 2.2, 2.3, 3.1, 6.1, 6.2, 10.1–10.8
"""

from __future__ import annotations

import logging
import secrets
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger("cycle_coordinator")

# ---------------------------------------------------------------------------
# Module-level state for price monitor coordination (P1 scope)
# ---------------------------------------------------------------------------

_cycle_decision_window_end: datetime | None = None


def get_decision_window_end() -> datetime | None:
    """Return the end of the current decision window, or None if no cycle active."""
    return _cycle_decision_window_end


# ---------------------------------------------------------------------------
# Cycle ID generation
# ---------------------------------------------------------------------------


def generate_cycle_id(trigger_source: str) -> str:
    """Generate a unique cycle identifier.

    Format: {YYYYMMDD}_{HHMM}_{source}_{4-char-hex}
    Example: 20260723_1030_scheduled_a3f2

    The 4-char hex suffix (from secrets.token_hex(2)) guarantees uniqueness
    across rapid manual triggers within the same minute.

    Args:
        trigger_source: Origin of the cycle trigger (e.g., "scheduled", "manual",
                        "alert_dispatch").

    Returns:
        A human-readable cycle identifier string.
    """
    now = datetime.now(timezone.utc)
    date_part = now.strftime("%Y%m%d")
    time_part = now.strftime("%H%M")
    suffix = secrets.token_hex(2)  # 4 hex chars
    return f"{date_part}_{time_part}_{trigger_source}_{suffix}"


# ---------------------------------------------------------------------------
# CycleContext — immutable context for a single market cycle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CycleContext:
    """Immutable context for a single market cycle.

    Attributes:
        cycle_id: Unique identifier for this cycle.
        trigger_source: Origin of the cycle ("scheduled", "manual", "alert_dispatch").
        started_at: UTC timestamp when the cycle started.
        focused_symbols: Symbols selected for this cycle.
        decision_window_end: When the decision window expires (started_at + window seconds).
        analyst_timeout_seconds: Maximum wait for analyst phase.
        pm_timeout_seconds: Maximum wait for PM phase.
        freshness_window_seconds: Signal age threshold for freshness gate.
        finnhub_budget: Maximum Finnhub API calls allowed during this cycle.
    """

    cycle_id: str
    trigger_source: str
    started_at: datetime
    focused_symbols: tuple[str, ...]
    decision_window_end: datetime
    analyst_timeout_seconds: int
    pm_timeout_seconds: int
    freshness_window_seconds: int
    finnhub_budget: int


# ---------------------------------------------------------------------------
# CycleCoordinator — phase-based orchestrator
# ---------------------------------------------------------------------------


class CycleCoordinator:
    """Orchestrates a single market cycle through sequential phases.

    The coordinator owns:
    - Phase ordering (analyst completes before PM starts)
    - Freshness gating (PM only sees fresh symbols)
    - Timeout advancement (cycle never blocks indefinitely)
    - Late-write protection (timed-out phases cannot revive a completed PM cycle)

    The coordinator does NOT own:
    - Trade logic (PM decides)
    - Signal generation (analyst decides)
    - Position sizing, gate evaluation, execution
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._pm_phase_completed: bool = False

    def run_market_cycle(
        self,
        *,
        trigger_source: str = "scheduled",
        override_symbols: list[str] | None = None,
    ) -> "CycleSummary":
        """Execute a full coordinated market cycle.

        Phases execute sequentially. Each phase is wrapped in try/except
        so a failure in one phase does not prevent subsequent phases (fail-open).

        Args:
            trigger_source: Origin of the trigger ("scheduled", "manual", "alert_dispatch").
            override_symbols: If provided, use these symbols instead of building a watchlist.

        Returns:
            CycleSummary with phase logs and outcome counts.
        """
        global _cycle_decision_window_end

        # Lazy imports to avoid circular dependencies
        from utils.cycle_logger import CycleLogger
        from utils.gate_config import (
            CYCLE_ANALYST_TIMEOUT_SECONDS,
            CYCLE_DECISION_WINDOW_SECONDS,
            CYCLE_FINNHUB_BUDGET,
            CYCLE_PM_TIMEOUT_SECONDS,
            PM_SIGNAL_FRESHNESS_WINDOW_SECONDS,
        )

        # Generate cycle identity
        cycle_id = generate_cycle_id(trigger_source)
        started_at = datetime.now(timezone.utc)
        decision_window_end = started_at + timedelta(seconds=CYCLE_DECISION_WINDOW_SECONDS)

        # Set module-level decision window for price monitor coordination
        _cycle_decision_window_end = decision_window_end

        # Reset PM phase flag for this cycle
        self._pm_phase_completed = False

        cycle_logger = CycleLogger(cycle_id)

        # Phase 1: Focus selection
        symbols: list[str] = []
        try:
            symbols = self._phase_focus_selection(override_symbols)
        except Exception as exc:
            logger.error(
                "Phase focus_selection failed for cycle_id=%s: %s",
                cycle_id,
                exc,
                exc_info=True,
            )
            cycle_logger.log_phase_complete("focus_selection", status="error")

        # Build context now that we have symbols
        ctx = CycleContext(
            cycle_id=cycle_id,
            trigger_source=trigger_source,
            started_at=started_at,
            focused_symbols=tuple(symbols),
            decision_window_end=decision_window_end,
            analyst_timeout_seconds=CYCLE_ANALYST_TIMEOUT_SECONDS,
            pm_timeout_seconds=CYCLE_PM_TIMEOUT_SECONDS,
            freshness_window_seconds=PM_SIGNAL_FRESHNESS_WINDOW_SECONDS,
            finnhub_budget=CYCLE_FINNHUB_BUDGET,
        )

        cycle_logger.log_cycle_start(ctx)

        if symbols:
            cycle_logger.log_phase_complete(
                "focus_selection",
                symbols_processed=len(symbols),
                status="completed",
            )

        # Phase 2: Analyst refresh
        try:
            cycle_logger.log_phase_start("analyst_refresh")
            self._phase_analyst_refresh(ctx, symbols)
            cycle_logger.log_phase_complete(
                "analyst_refresh",
                symbols_processed=len(symbols),
                status="completed",
            )
        except _AnalystTimeoutError:
            logger.warning(
                "Analyst phase timed out for cycle_id=%s after %ds. Advancing to freshness gate.",
                cycle_id,
                ctx.analyst_timeout_seconds,
            )
            cycle_logger.log_phase_complete(
                "analyst_refresh",
                symbols_processed=len(symbols),
                status="timeout",
            )
        except Exception as exc:
            logger.error(
                "Phase analyst_refresh failed for cycle_id=%s: %s",
                cycle_id,
                exc,
                exc_info=True,
            )
            cycle_logger.log_phase_complete(
                "analyst_refresh",
                symbols_processed=len(symbols),
                status="error",
            )

        # Phase 3: Freshness gate
        fresh_symbols: list[str] = []
        try:
            cycle_logger.log_phase_start("freshness_gate")
            freshness_result = self._phase_freshness_gate(ctx, symbols)
            fresh_symbols = list(freshness_result.fresh_symbols)
            cycle_logger.set_symbols_fresh(len(fresh_symbols))

            # Log skip events
            for skip in freshness_result.skip_events:
                cycle_logger.log_freshness_skip(skip)

            cycle_logger.log_phase_complete(
                "freshness_gate",
                symbols_processed=len(symbols),
                status="completed",
                fresh=len(fresh_symbols),
                stale=len(freshness_result.stale_symbols),
                missing=len(freshness_result.missing_symbols),
                error=len(freshness_result.error_symbols),
            )
        except Exception as exc:
            logger.error(
                "Phase freshness_gate failed for cycle_id=%s: %s",
                cycle_id,
                exc,
                exc_info=True,
            )
            cycle_logger.log_phase_complete(
                "freshness_gate",
                symbols_processed=len(symbols),
                status="error",
            )

        # Phase 4: PM decisioning
        pm_decisions: list[dict] = []
        try:
            cycle_logger.log_phase_start("pm_decisioning")
            pm_decisions = self._phase_pm_decisioning(ctx, fresh_symbols)
            self._pm_phase_completed = True
            cycle_logger.log_phase_complete(
                "pm_decisioning",
                symbols_processed=len(fresh_symbols),
                status="completed",
                decisions=len(pm_decisions),
            )
        except _PmTimeoutError:
            self._pm_phase_completed = True
            logger.warning(
                "PM phase timed out for cycle_id=%s after %ds. Advancing to safety checks.",
                cycle_id,
                ctx.pm_timeout_seconds,
            )
            cycle_logger.log_phase_complete(
                "pm_decisioning",
                symbols_processed=len(fresh_symbols),
                status="timeout",
            )
        except Exception as exc:
            self._pm_phase_completed = True
            logger.error(
                "Phase pm_decisioning failed for cycle_id=%s: %s",
                cycle_id,
                exc,
                exc_info=True,
            )
            cycle_logger.log_phase_complete(
                "pm_decisioning",
                symbols_processed=len(fresh_symbols),
                status="error",
            )

        # Phase 5: Safety checks
        try:
            cycle_logger.log_phase_start("safety_checks")
            self._phase_safety_checks(ctx)
            cycle_logger.log_phase_complete(
                "safety_checks",
                status="completed",
            )
        except Exception as exc:
            logger.error(
                "Phase safety_checks failed for cycle_id=%s: %s",
                cycle_id,
                exc,
                exc_info=True,
            )
            cycle_logger.log_phase_complete(
                "safety_checks",
                status="error",
            )

        # Phase 6: Bookkeeping (P&L snapshot)
        try:
            cycle_logger.log_phase_start("bookkeeping")
            self._phase_bookkeeping(ctx)
            cycle_logger.log_phase_complete(
                "bookkeeping",
                status="completed",
            )
        except Exception as exc:
            logger.error(
                "Phase bookkeeping failed for cycle_id=%s: %s",
                cycle_id,
                exc,
                exc_info=True,
            )
            cycle_logger.log_phase_complete(
                "bookkeeping",
                status="error",
            )

        # Clear decision window at cycle end
        _cycle_decision_window_end = None

        summary = cycle_logger.log_cycle_end()
        return summary

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _phase_focus_selection(
        self,
        override_symbols: list[str] | None,
    ) -> list[str]:
        """Phase 1: Build focus list.

        If override_symbols are provided, use those directly.
        Otherwise, build a basic watchlist from the WATCHLIST env var.

        The full _apply_focus_list() logic in orchestrator is complex (caching,
        source bonuses, scout picks). For the coordinator, we simplify: use
        override_symbols if provided, else fall back to the base watchlist.
        """
        if override_symbols:
            logger.info(
                "Focus selection: using %d override symbols: %s",
                len(override_symbols),
                override_symbols,
            )
            return list(override_symbols)

        # Build watchlist from environment (same source as orchestrator)
        import os

        watchlist_raw = os.getenv(
            "WATCHLIST", "SPY,QQQ,IWM,DIA,TLT,GLD,XLK,XLF,XLE,TSLA,NVDA,AMD"
        )
        pm_tradable_raw = os.getenv("PM_TRADABLE_SYMBOLS", "META,MU")

        symbols = [
            s.strip().upper()
            for s in (watchlist_raw + "," + pm_tradable_raw).split(",")
            if s.strip()
        ]
        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for s in symbols:
            if s not in seen:
                seen.add(s)
                deduped.append(s)

        focus_mode = os.getenv("ANALYST_FOCUS_MODE", "disabled").strip().lower()
        if focus_mode == "enabled":
            from utils.focus_list import select_focus_symbols

            try:
                max_symbols = int(os.getenv("ANALYST_FOCUS_MAX_SYMBOLS", "3"))
            except ValueError:
                max_symbols = 3

            focused = select_focus_symbols(
                self._engine,
                deduped,
                max_symbols=max(1, max_symbols),
                context="coordinated_cycle",
            )
            logger.info(
                "Focus selection: narrowed %d candidates to %d symbols: %s",
                len(deduped),
                len(focused),
                focused,
            )
            return focused

        logger.info(
            "Focus selection: built watchlist with %d symbols: %s",
            len(deduped),
            deduped,
        )
        return deduped

    def _phase_analyst_refresh(self, ctx: CycleContext, symbols: list[str]) -> dict:
        """Phase 2: Run analyst refresh with timeout.

        On timeout: cycle advances. Late-arriving analyst writes may persist
        to AgentMemory, but they carry the current cycle_id. The freshness
        gate (Phase 3) will still classify them correctly if they land before
        the gate runs.

        Raises:
            _AnalystTimeoutError: If analyst does not complete within timeout.
        """
        if not symbols:
            logger.info("Analyst phase skipped: no symbols to process.")
            return {}

        def _run_analyst() -> dict:
            # Lazy import to avoid circular dependencies
            import agents.analyst as analyst

            # Try passing cycle_id; fall back gracefully if param not added yet
            try:
                return analyst.run(self._engine, symbols, cycle_id=ctx.cycle_id)
            except TypeError:
                logger.warning(
                    "analyst.run() does not accept cycle_id yet. Calling without it."
                )
                return analyst.run(self._engine, symbols)

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analyst")
        future = executor.submit(_run_analyst)
        done, _ = wait([future], timeout=ctx.analyst_timeout_seconds)

        if not done:
            executor.shutdown(wait=False, cancel_futures=True)
            raise _AnalystTimeoutError(
                f"Analyst did not complete within {ctx.analyst_timeout_seconds}s"
            )

        try:
            # Re-raise any exception from the analyst
            return future.result()
        finally:
            executor.shutdown(wait=True)

    def _phase_freshness_gate(self, ctx: CycleContext, symbols: list[str]) -> "FreshnessResult":
        """Phase 3: Validate signal freshness. Runs BEFORE PM.

        Returns only fresh symbols. PM receives the fresh list — it never
        sees stale symbols and does not need freshness awareness.

        The freshness gate itself is fail-closed (handled in signal_freshness.py).
        The phase wrapper is fail-open (if the whole phase crashes, the caller
        catches it and advances).
        """
        from utils.signal_freshness import FreshnessResult, check_signal_freshness

        if not symbols:
            return FreshnessResult(
                fresh_symbols=(),
                stale_symbols=(),
                missing_symbols=(),
                error_symbols=(),
                skip_events=(),
            )

        return check_signal_freshness(
            self._engine,
            symbols,
            ctx.cycle_id,
            freshness_window_seconds=ctx.freshness_window_seconds,
        )

    def _phase_pm_decisioning(self, ctx: CycleContext, fresh_symbols: list[str]) -> list[dict]:
        """Phase 4: Run PM profiles against fresh symbols only.

        Passes cycle_id to run_profile() so candidate registry uses the
        coordinator's cycle_id (matching analyst signals).

        Respects _try_begin_pm_cycle() lock from orchestrator.

        Raises:
            _PmTimeoutError: If PM does not complete within timeout.
        """
        if not fresh_symbols:
            logger.info(
                "PM phase skipped for cycle_id=%s: no fresh symbols after freshness gate.",
                ctx.cycle_id,
            )
            return []

        # Lazy imports to avoid circular deps
        from orchestrator import _end_pm_cycle, _try_begin_pm_cycle

        owner = f"coordinator_{ctx.cycle_id}"
        if not _try_begin_pm_cycle(owner):
            logger.warning(
                "PM phase skipped for cycle_id=%s: another PM cycle is active.",
                ctx.cycle_id,
            )
            return []

        try:
            decisions = self._run_pm_with_timeout(ctx, fresh_symbols)
            return decisions
        finally:
            _end_pm_cycle(owner)

    def _run_pm_with_timeout(self, ctx: CycleContext, fresh_symbols: list[str]) -> list[dict]:
        """Execute PM run_profile with ThreadPoolExecutor timeout.

        Raises:
            _PmTimeoutError: If PM does not complete within timeout.
        """
        import agents.portfolio_manager as pm
        from models.pm_profiles import ACTIVE_PROFILES

        def _run_pm() -> list[dict]:
            decisions: list[dict] = []
            for profile_id in ACTIVE_PROFILES:
                # Try passing cycle_id; fall back gracefully if param not added yet
                try:
                    result = pm.run_profile(
                        self._engine,
                        fresh_symbols,
                        profile_id,
                        cycle_id=ctx.cycle_id,
                    )
                except TypeError:
                    logger.warning(
                        "pm.run_profile() does not accept cycle_id yet. Calling without it."
                    )
                    result = pm.run_profile(self._engine, fresh_symbols, profile_id)

                if isinstance(result, dict):
                    decisions.extend(result.get("decisions", []))

            return decisions

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pm")
        future = executor.submit(_run_pm)
        done, _ = wait([future], timeout=ctx.pm_timeout_seconds)

        if not done:
            executor.shutdown(wait=False, cancel_futures=True)
            raise _PmTimeoutError(
                f"PM did not complete within {ctx.pm_timeout_seconds}s"
            )

        try:
            return future.result()
        finally:
            executor.shutdown(wait=True)

    def _phase_safety_checks(self, ctx: CycleContext) -> None:
        """Phase 5: Stop losses and position health.

        Calls bookkeeper.check_stop_losses(engine) to evaluate open positions.
        """
        import agents.bookkeeper as bookkeeper

        logger.info("Running safety checks for cycle_id=%s", ctx.cycle_id)
        stops_triggered = bookkeeper.check_stop_losses(self._engine)

        if stops_triggered:
            logger.info(
                "Safety checks: %d stop losses triggered for cycle_id=%s",
                len(stops_triggered),
                ctx.cycle_id,
            )
        else:
            logger.info(
                "Safety checks: no stop losses triggered for cycle_id=%s",
                ctx.cycle_id,
            )

    def _phase_bookkeeping(self, ctx: CycleContext) -> None:
        """Phase 6: Record daily P&L snapshot and portfolio summary.

        Calls bookkeeper.record_daily_snapshot() if available.
        Fail-open: errors are logged but do not block cycle completion.
        """
        import agents.bookkeeper as bookkeeper

        logger.info("Running bookkeeping for cycle_id=%s", ctx.cycle_id)

        if hasattr(bookkeeper, "record_daily_snapshot"):
            bookkeeper.record_daily_snapshot(self._engine)
            logger.info("Daily snapshot recorded for cycle_id=%s", ctx.cycle_id)
        else:
            logger.debug(
                "bookkeeper.record_daily_snapshot() not available; skipping snapshot for cycle_id=%s",
                ctx.cycle_id,
            )

    @property
    def pm_phase_completed(self) -> bool:
        """Whether the PM phase has completed for the current cycle.

        Once set, late analyst writes cannot re-trigger PM for this cycle.
        """
        return self._pm_phase_completed


# ---------------------------------------------------------------------------
# Internal exceptions for phase control flow
# ---------------------------------------------------------------------------


class _AnalystTimeoutError(Exception):
    """Raised when analyst phase exceeds its timeout."""


class _PmTimeoutError(Exception):
    """Raised when PM phase exceeds its timeout."""
