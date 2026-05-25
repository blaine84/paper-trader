"""Expanded Watchlist management for the Sector Scout pipeline.

Provides functions to update, retrieve, and expire the daily Expanded Watchlist
stored in AgentMemory. The Expanded Watchlist is informational only — it carries
no trade pressure and never permanently mutates the Core_Watchlist.

Key pattern: agent="sector_scout", key="expanded_watchlist:{YYYY-MM-DD}"

See: design.md §4 (Expanded Watchlist Management), requirements.md §7, §8.5, §8.7, §9.4

No-Trade Pressure Verification (Requirements 7.1, 7.2, 7.3, 7.4, 12.1, 12.2):
- This module contains NO urgency language, trade recommendations, or position sizing.
- Expanded candidates are NOT counted toward "trades attempted" or "opportunities missed" metrics.
- The orchestrator passes expanded symbols through the identical Analyst → PM gate pipeline
  (catalyst specificity, risk geometry, stop authority) as Core_Watchlist symbols.
- No special "fast track" or "skip gates" logic exists for expanded candidates.
- Scout output flows: Chief Scout → Expanded Watchlist → Analyst → PM (same gates).
- No direct trade execution occurs from scout output.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from db.schema import AgentMemory, get_session
from utils.scout_logging import emit_scout_event

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def update_expanded_watchlist(
    engine,
    picks: list[dict],
    run_type: str,
    config: dict,
) -> list[str]:
    """Write picks to AgentMemory as today's Expanded Watchlist.

    Merges with existing picks from earlier runs today.
    Enforces max_expanded_watchlist ceiling.
    Deduplicates symbols across runs (keeps highest scout_score version).
    Returns final list of expanded symbols.

    Args:
        engine: SQLAlchemy engine.
        picks: List of ChiefScoutPick dicts (must have 'symbol' and
               'source_candidate_score' fields).
        run_type: "premarket" | "confirmation" | "midday"
        config: Parsed sector_scout_config.yaml dict.

    Returns:
        Final list of symbol strings in today's Expanded Watchlist.
    """
    budget_ceilings = config.get("budget_ceilings", {})
    max_expanded_watchlist: int = int(
        budget_ceilings.get("max_expanded_watchlist", 12)
    )

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    memory_key = f"expanded_watchlist:{today}"

    db = get_session(engine)
    try:
        # 1. Read existing watchlist for today
        existing_record = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout", key=memory_key)
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )

        existing_picks: list[dict] = []
        if existing_record:
            try:
                data = json.loads(existing_record.value)
                existing_picks = data.get("picks", [])
            except (json.JSONDecodeError, TypeError):
                existing_picks = []

        # 2. Merge new picks with existing — deduplicate by symbol,
        #    keeping the version with the highest scout_score
        merged: dict[str, dict] = {}

        for pick in existing_picks:
            symbol = pick.get("symbol", "")
            if symbol:
                merged[symbol] = pick

        for pick in picks:
            symbol = pick.get("symbol", "")
            if not symbol:
                continue
            existing = merged.get(symbol)
            if existing is None:
                merged[symbol] = pick
            else:
                # Keep the version with the higher score
                new_score = _get_score(pick)
                old_score = _get_score(existing)
                if new_score > old_score:
                    merged[symbol] = pick

        # 3. Enforce max_expanded_watchlist ceiling
        #    Sort by score descending to keep the best picks when truncating
        all_picks = list(merged.values())
        all_picks.sort(key=lambda p: _get_score(p), reverse=True)

        if len(all_picks) > max_expanded_watchlist:
            emit_scout_event("BUDGET_CEILING_HIT", {
                "ceiling_type": "max_expanded_watchlist",
                "limit_value": max_expanded_watchlist,
                "context": f"Watchlist truncated from {len(all_picks)} to {max_expanded_watchlist} symbols",
            })
            all_picks = all_picks[:max_expanded_watchlist]

        # 4. Write merged result back to AgentMemory
        watchlist_value = json.dumps({
            "date": today,
            "run_type": run_type,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "picks": all_picks,
            "symbols": [p.get("symbol", "") for p in all_picks],
            "size": len(all_picks),
        })

        if existing_record:
            # Update in place
            existing_record.value = watchlist_value
            existing_record.timestamp = datetime.utcnow()
        else:
            # Create new record
            db.add(AgentMemory(
                agent="sector_scout",
                symbol=None,
                key=memory_key,
                value=watchlist_value,
            ))

        db.commit()

        final_symbols = [p.get("symbol", "") for p in all_picks]

        # Emit structured EXPANDED_WATCHLIST event
        emit_scout_event("EXPANDED_WATCHLIST", {
            "symbols": final_symbols,
            "total_size": len(final_symbols),
            "run_type": run_type,
        })

        return final_symbols

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_expanded_watchlist(engine) -> list[str]:
    """Return today's Expanded Watchlist symbols from AgentMemory.

    Returns:
        List of symbol strings. Empty list if no watchlist exists for today.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    memory_key = f"expanded_watchlist:{today}"

    db = get_session(engine)
    try:
        record = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout", key=memory_key)
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )

        if not record:
            return []

        try:
            data = json.loads(record.value)
        except (json.JSONDecodeError, TypeError):
            return []

        # Verify the date matches today (defensive check)
        if data.get("date") != today:
            return []

        return data.get("symbols", [])

    finally:
        db.close()


def expire_expanded_watchlist(engine) -> None:
    """Mark yesterday's Expanded Watchlist as expired. Called at EOD.

    Deletes yesterday's watchlist record from AgentMemory.
    Never permanently mutates Core_Watchlist.
    """
    yesterday = (
        datetime.now(timezone.utc) - timedelta(days=1)
    ).strftime("%Y-%m-%d")
    memory_key = f"expanded_watchlist:{yesterday}"

    db = get_session(engine)
    try:
        records = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout", key=memory_key)
            .all()
        )

        if records:
            for record in records:
                # Mark as expired by updating the value
                try:
                    data = json.loads(record.value)
                    data["expired"] = True
                    data["expired_at"] = datetime.now(timezone.utc).isoformat()
                    record.value = json.dumps(data)
                except (json.JSONDecodeError, TypeError):
                    # If we can't parse it, just delete it
                    db.delete(record)

            db.commit()
            logger.info(
                "Expired Expanded Watchlist for %s (%d records)",
                yesterday,
                len(records),
            )
        else:
            logger.debug(
                "No Expanded Watchlist to expire for %s", yesterday
            )

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _get_score(pick: dict) -> float:
    """Extract the scout score from a pick dict.

    Checks 'source_candidate_score' first (ChiefScoutPick format),
    then falls back to 'scout_score'.
    """
    score = pick.get("source_candidate_score")
    if score is not None:
        try:
            return float(score)
        except (ValueError, TypeError):
            pass

    score = pick.get("scout_score")
    if score is not None:
        try:
            return float(score)
        except (ValueError, TypeError):
            pass

    return 0.0
