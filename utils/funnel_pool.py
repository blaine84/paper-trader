"""Funnel Candidate Pool — PM-facing interface for eligible funnel candidates.

Provides functions to retrieve today's pm_eligible candidates, build compact
analysis context for PM input, and expire yesterday's candidates at EOD.

Replaces get_expanded_watchlist() for funnel-sourced candidates.

See: design.md §Component 6, requirements.md §9, §3.9
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from db.schema import FunnelCandidate, get_session

logger = logging.getLogger(__name__)

_NY_TZ = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_pm_eligible_candidates(engine, max_handoff: int = 3) -> list[str]:
    """Return today's pm_eligible funnel candidate symbols, capped by max_pm_handoff.

    Replaces get_expanded_watchlist() for funnel-sourced candidates.
    Returns symbols with stage_status='pm_eligible' for today's date,
    ordered by confirmation decision timestamp (earliest first), then
    scout_rank (lowest first), then scout_score (highest first).
    Enforces max_pm_handoff ceiling — returns at most max_handoff symbols.

    Args:
        engine: SQLAlchemy engine.
        max_handoff: Maximum number of symbols to return (default 3).

    Returns:
        List of symbol strings, ordered deterministically. Empty list if
        no pm_eligible candidates exist for today.
    """
    today_ny = datetime.now(_NY_TZ).date()

    session = get_session(engine)
    try:
        candidates = (
            session.query(FunnelCandidate)
            .filter(
                FunnelCandidate.date == today_ny,
                FunnelCandidate.stage_status == "pm_eligible",
                FunnelCandidate.expired == False,  # noqa: E712
            )
            .all()
        )

        if not candidates:
            return []

        # Apply deterministic ordering:
        # 1. Confirmation stage-decision timestamp ASC (earliest promoted first)
        # 2. scout_rank ASC (lower rank first)
        # 3. scout_score DESC (higher score first)
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (
                _get_confirmation_timestamp(c),
                c.scout_rank,
                -c.scout_score,
            ),
        )

        # Enforce max_pm_handoff ceiling
        return [c.symbol for c in sorted_candidates[:max_handoff]]

    finally:
        session.close()


def get_candidate_context(engine, symbol: str) -> dict | None:
    """Return linked Scout/Researcher/Analyst/Confirmation decisions for a symbol.

    Used by PM to receive compact analysis context for promoted candidates.
    Returns today's FunnelCandidate with its full stage_decisions parsed
    and organized by agent.

    Args:
        engine: SQLAlchemy engine.
        symbol: The ticker symbol to look up.

    Returns:
        Dictionary with candidate metadata and decisions organized by agent,
        or None if no candidate exists for this symbol today.
    """
    today_ny = datetime.now(_NY_TZ).date()

    session = get_session(engine)
    try:
        candidate = (
            session.query(FunnelCandidate)
            .filter(
                FunnelCandidate.date == today_ny,
                FunnelCandidate.symbol == symbol,
                FunnelCandidate.expired == False,  # noqa: E712
            )
            .first()
        )

        if candidate is None:
            return None

        # Parse stage_decisions
        try:
            stage_decisions = json.loads(candidate.stage_decisions or "[]")
        except (json.JSONDecodeError, TypeError):
            stage_decisions = []

        # Organize decisions by agent
        decisions_by_agent: dict[str, list[dict]] = {}
        for sd in stage_decisions:
            agent = sd.get("agent", "unknown")
            if agent not in decisions_by_agent:
                decisions_by_agent[agent] = []
            decisions_by_agent[agent].append(sd)

        # Build context payload matching PM input requirements (Req 9.3):
        # - Scout selection_reason and scout_score
        # - Researcher catalyst_validation decision and reasoning
        # - Analyst authoritative_setup_type with key_levels and invalidation
        # - Confirmation decision with reasoning
        context = {
            "symbol": candidate.symbol,
            "candidate_id": candidate.candidate_id,
            "date": str(candidate.date),
            "source_run": candidate.source_run,
            "selection_mode": candidate.selection_mode,
            "stage_status": candidate.stage_status,
            # Scout context
            "scout": {
                "scout_rank": candidate.scout_rank,
                "scout_score": candidate.scout_score,
                "selection_reason": candidate.selection_reason,
                "direction_bias": candidate.direction_bias,
                "primary_risk": candidate.primary_risk,
                "catalyst_evidence": _safe_json_parse(candidate.catalyst_evidence),
                "sector_context": _safe_json_parse(candidate.sector_context),
                "preliminary_setup_type": candidate.preliminary_setup_type,
            },
            # Researcher context (most recent promoted/needs_confirmation decision)
            "researcher": _extract_agent_context(decisions_by_agent.get("researcher", [])),
            # Analyst context
            "analyst": _extract_analyst_context(
                decisions_by_agent.get("analyst", []),
                candidate.authoritative_setup_type,
            ),
            # Confirmation context
            "confirmation": _extract_agent_context(decisions_by_agent.get("confirmation", [])),
            # Full stage history for audit
            "stage_decisions": stage_decisions,
        }

        return context

    finally:
        session.close()


def build_full_watchlist_with_funnel(
    engine,
    core_watchlist: list[str],
    scout_picks: list[str],
    max_pm_handoff: int = 3,
) -> dict:
    """Construct deduplicated watchlist integrating funnel candidates for PM.

    Combines core watchlist symbols, scout picks, and pm_eligible funnel
    candidates into a single deduplicated list. Enforces max_pm_handoff
    ceiling on funnel additions. Provides candidate context for PM input.

    Deduplication priority: core watchlist first, then scout_picks, then
    funnel candidates. If a symbol appears in multiple sources, it appears
    only once (in its highest-priority source position).

    All existing execution/risk gates (setup-quality, stop authority,
    catalyst specificity, entry geometry, sizing, concentration) apply
    unchanged to funnel candidates — they receive no special treatment.

    Args:
        engine: SQLAlchemy engine.
        core_watchlist: Core watchlist symbols (e.g., WATCHLIST env var).
        scout_picks: Today's Scout picks (daily LLM picks + expanded).
        max_pm_handoff: Maximum funnel candidates to include (default 3).

    Returns:
        Dictionary with:
        - full_watchlist: list[str] — deduplicated combined watchlist
        - funnel_symbols: list[str] — funnel candidates included (subset
          of full_watchlist)
        - funnel_context: dict[str, dict] — per-symbol context from
          get_candidate_context() for each funnel candidate. Contains
          Scout selection_reason/score, Researcher catalyst_validation,
          Analyst setup_type/levels/invalidation, Confirmation decision.

    See: requirements.md §9.2, §9.3, §9.4, §9.5, §9.6, §9.8
    """
    # Get pm_eligible funnel candidates (already ordered deterministically
    # and capped by max_pm_handoff)
    pm_eligible_symbols = get_pm_eligible_candidates(engine, max_handoff=max_pm_handoff)

    # Build deduplicated watchlist: core → scout_picks → funnel candidates
    seen: set[str] = set()
    full_watchlist: list[str] = []

    for sym in core_watchlist:
        if sym not in seen:
            seen.add(sym)
            full_watchlist.append(sym)

    for sym in scout_picks:
        if sym not in seen:
            seen.add(sym)
            full_watchlist.append(sym)

    # Add funnel candidates not already present in core or scout_picks
    funnel_added: list[str] = []
    for sym in pm_eligible_symbols:
        if sym not in seen:
            seen.add(sym)
            full_watchlist.append(sym)
            funnel_added.append(sym)
        else:
            # Symbol already in watchlist from core or scout — still
            # track it as funnel-sourced for context (PM may benefit
            # from funnel analysis even for symbols discovered elsewhere)
            funnel_added.append(sym)

    # Build context for all funnel candidates (including those already
    # in core/scout_picks — PM benefits from the full funnel analysis
    # chain regardless of how the symbol entered the watchlist)
    funnel_context: dict[str, dict] = {}
    for sym in pm_eligible_symbols:
        ctx = get_candidate_context(engine, sym)
        if ctx is not None:
            funnel_context[sym] = ctx

    logger.info(
        "Built full watchlist: %d core + %d scout + %d funnel = %d total "
        "(funnel symbols: %s)",
        len(core_watchlist),
        len(scout_picks),
        len(funnel_added),
        len(full_watchlist),
        ", ".join(pm_eligible_symbols) if pm_eligible_symbols else "none",
    )

    return {
        "full_watchlist": full_watchlist,
        "funnel_symbols": pm_eligible_symbols,
        "funnel_context": funnel_context,
    }


def expire_daily_candidates(engine) -> int:
    """Mark yesterday's active candidates as expired. Called at EOD.

    Sets expired=True for all FunnelCandidate records from yesterday's
    New York trading date that are not already expired. Does not delete
    records — they remain queryable for review and learning.

    Args:
        engine: SQLAlchemy engine.

    Returns:
        Count of expired candidates.
    """
    yesterday_ny = (datetime.now(_NY_TZ) - timedelta(days=1)).date()

    session = get_session(engine)
    try:
        candidates = (
            session.query(FunnelCandidate)
            .filter(
                FunnelCandidate.date == yesterday_ny,
                FunnelCandidate.expired == False,  # noqa: E712
            )
            .all()
        )

        count = len(candidates)
        if count > 0:
            for candidate in candidates:
                candidate.expired = True
                candidate.updated_at = datetime.now(timezone.utc)
            session.commit()
            logger.info(
                "Expired %d funnel candidates from %s", count, yesterday_ny
            )
        else:
            logger.debug(
                "No active funnel candidates to expire for %s", yesterday_ny
            )

        return count

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _get_confirmation_timestamp(candidate: FunnelCandidate) -> str:
    """Extract the confirmation stage-decision timestamp for ordering.

    Finds the most recent confirmation decision with decision='promoted'
    and returns its timestamp. Falls back to a high sentinel value if
    no confirmation timestamp is found (shouldn't happen for pm_eligible).

    Returns:
        ISO 8601 timestamp string for sorting (earliest first).
    """
    try:
        stage_decisions = json.loads(candidate.stage_decisions or "[]")
    except (json.JSONDecodeError, TypeError):
        return "9999-12-31T23:59:59Z"

    for sd in reversed(stage_decisions):
        if sd.get("agent") == "confirmation" and sd.get("decision") == "promoted":
            return sd.get("timestamp", "9999-12-31T23:59:59Z")

    return "9999-12-31T23:59:59Z"


def _extract_agent_context(decisions: list[dict]) -> dict | None:
    """Extract the most recent actionable decision from an agent's decisions.

    Looks for the most recent promoted or needs_confirmation decision.
    Falls back to the most recent decision of any type.

    Returns:
        Dictionary with decision, reasoning, and evidence, or None.
    """
    if not decisions:
        return None

    # Prefer the most recent promoted/needs_confirmation decision
    for sd in reversed(decisions):
        if sd.get("decision") in ("promoted", "needs_confirmation"):
            return {
                "decision": sd.get("decision"),
                "reasoning": sd.get("reasoning"),
                "evidence": sd.get("evidence", {}),
                "timestamp": sd.get("timestamp"),
            }

    # Fall back to most recent decision
    last = decisions[-1]
    return {
        "decision": last.get("decision"),
        "reasoning": last.get("reasoning"),
        "evidence": last.get("evidence", {}),
        "timestamp": last.get("timestamp"),
    }


def _extract_analyst_context(
    decisions: list[dict], authoritative_setup_type: str | None
) -> dict | None:
    """Extract Analyst context including authoritative setup type.

    Returns:
        Dictionary with decision, setup_type, key_levels, invalidation, etc.
    """
    base = _extract_agent_context(decisions)
    if base is None:
        return None

    # Enrich with authoritative_setup_type from the FunnelCandidate field
    base["authoritative_setup_type"] = authoritative_setup_type

    # Extract key_levels and invalidation from evidence if present
    evidence = base.get("evidence", {})
    if evidence:
        base["key_levels"] = evidence.get("key_levels")
        base["invalidation"] = evidence.get("invalidation")
        base["signal_direction"] = evidence.get("signal_direction")
        base["signal_strength"] = evidence.get("signal_strength")
        base["volume_requirements"] = evidence.get("volume_requirements")

    return base


def _safe_json_parse(text: str | None) -> dict | list | str | None:
    """Safely parse a JSON text field. Returns parsed object or raw text."""
    if text is None:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
