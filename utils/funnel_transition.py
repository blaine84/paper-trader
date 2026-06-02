"""Funnel transition logic — migrates from expanded_watchlist to FunnelCandidate.

Implements Phase A (parallel) of the expanded_watchlist → FunnelCandidate migration:
- Uses get_pm_eligible_candidates() as primary source
- Falls back to get_expanded_watchlist() ONLY if no FunnelRunLog exists for today
  with stage="discovery" and result_status in ("completed", "degraded")
- If FunnelRunLog exists with completed/degraded and funnel returns empty,
  does NOT fall back (valid empty funnel result)
- Deduplicates symbols when both sources active during transition

See: design.md §Migration from Expanded Watchlist, requirements.md §13
"""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from db.schema import FunnelRunLog, get_session
from utils.funnel_pool import get_pm_eligible_candidates
from utils.expanded_watchlist import get_expanded_watchlist

logger = logging.getLogger(__name__)

_NY_TZ = ZoneInfo("America/New_York")


def _has_valid_discovery_run_today(engine) -> bool:
    """Check if a FunnelRunLog exists for today's discovery with completed/degraded status.

    Returns True if the funnel discovery ran successfully today (even if it
    found zero candidates). This distinguishes "funnel ran and found nothing"
    from "funnel did not run".

    A FunnelRunLog with result_status="error" (e.g., missed job) does NOT
    count as valid — this intentionally triggers the legacy fallback during
    the transition period.
    """
    today_ny = datetime.now(_NY_TZ).date()

    session = get_session(engine)
    try:
        exists = (
            session.query(FunnelRunLog)
            .filter(
                FunnelRunLog.date == today_ny,
                FunnelRunLog.stage == "discovery",
                FunnelRunLog.result_status.in_(["completed", "degraded"]),
            )
            .first()
        ) is not None
        return exists
    finally:
        session.close()


def get_funnel_or_fallback_candidates(
    engine,
    max_pm_handoff: int = 3,
) -> tuple[list[str], str]:
    """Return funnel-sourced PM candidates with legacy fallback during transition.

    Implements the Phase A transition logic (Req 13.1–13.4):
    1. Try get_pm_eligible_candidates() as primary source
    2. If no valid discovery FunnelRunLog exists for today → fall back to
       get_expanded_watchlist() with max_pm_handoff ceiling
    3. If valid FunnelRunLog exists and funnel returns empty → return empty
       (do NOT fall back)
    4. Deduplicate symbols when both sources contribute

    Args:
        engine: SQLAlchemy engine.
        max_pm_handoff: Maximum funnel/fallback candidates to return (default 3).

    Returns:
        Tuple of (symbols, source) where source is one of:
        - "funnel" — symbols came from get_pm_eligible_candidates()
        - "legacy_fallback" — symbols came from get_expanded_watchlist()
        - "funnel_empty" — funnel ran validly but produced no candidates
    """
    # Step 1: Check if funnel discovery ran successfully today
    funnel_ran_today = _has_valid_discovery_run_today(engine)

    if funnel_ran_today:
        # Funnel discovery ran — use funnel as authoritative source
        funnel_symbols = get_pm_eligible_candidates(engine, max_handoff=max_pm_handoff)

        if funnel_symbols:
            logger.info(
                "Funnel transition: using %d pm_eligible candidates from funnel: %s",
                len(funnel_symbols),
                ", ".join(funnel_symbols),
            )
            return funnel_symbols, "funnel"
        else:
            # Valid empty funnel result — do NOT fall back (Req 13.3)
            logger.info(
                "Funnel transition: discovery ran (completed/degraded) but zero "
                "pm_eligible candidates — returning empty (no legacy fallback)."
            )
            return [], "funnel_empty"

    else:
        # No valid discovery run today — fall back to legacy expanded_watchlist
        # This covers: funnel not yet running, missed jobs (result_status="error"),
        # or days before funnel is deployed.
        logger.info(
            "Funnel transition: no valid discovery FunnelRunLog for today — "
            "falling back to get_expanded_watchlist()."
        )
        try:
            legacy_symbols = get_expanded_watchlist(engine)
            # Apply max_pm_handoff ceiling to legacy results (Req 13.2)
            capped = legacy_symbols[:max_pm_handoff]
            if capped:
                logger.info(
                    "Funnel transition: legacy fallback returning %d symbols "
                    "(capped from %d): %s",
                    len(capped),
                    len(legacy_symbols),
                    ", ".join(capped),
                )
            return capped, "legacy_fallback"
        except Exception as e:
            logger.error(f"Funnel transition: legacy fallback error: {e}", exc_info=True)
            return [], "legacy_fallback"


def build_deduplicated_watchlist(
    core_watchlist: list[str],
    scout_picks: list[str],
    expanded_symbols: list[str],
) -> list[str]:
    """Construct a deduplicated watchlist from all sources.

    Deduplication priority: core_watchlist > scout_picks > expanded_symbols.
    A symbol appearing in multiple sources is included only once (Req 13.4).

    Args:
        core_watchlist: Core watchlist symbols (WATCHLIST env var).
        scout_picks: Today's Scout picks.
        expanded_symbols: Funnel or legacy expanded candidates.

    Returns:
        Deduplicated list preserving priority order.
    """
    seen: set[str] = set()
    result: list[str] = []

    for sym in core_watchlist:
        if sym not in seen:
            seen.add(sym)
            result.append(sym)

    for sym in scout_picks:
        if sym not in seen:
            seen.add(sym)
            result.append(sym)

    for sym in expanded_symbols:
        if sym not in seen:
            seen.add(sym)
            result.append(sym)

    return result
