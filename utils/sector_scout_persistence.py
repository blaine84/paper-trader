"""AgentMemory persistence for Sector Scout pipeline results.

Provides functions to persist run summaries and per-symbol CandidateRows
to AgentMemory for auditability and observability.

Key patterns:
- run_summary:{YYYY-MM-DD}:{run_type}   → Full RunSummary JSON
- candidate_row:{YYYY-MM-DD}:{run_type}:{symbol} → Per-symbol CandidateRow JSON

Uses the same AgentMemory pattern as expanded_watchlist.py:
  agent="sector_scout", key=pattern

See: design.md §7 (Observability), requirements.md §10.1, §10.2, §10.3
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone

from db.schema import AgentMemory, get_session
from utils.sector_scout_models import CandidateRow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def persist_run_summary(
    engine,
    screener_result: dict,
    chief_result: dict,
    run_type: str,
    duration: float,
) -> None:
    """Persist a RunSummary to AgentMemory.

    Key pattern: run_summary:{YYYY-MM-DD}:{run_type}

    Builds a RunSummary dict from the screener and chief scout results,
    serializes to JSON, and stores in AgentMemory.

    Args:
        engine: SQLAlchemy engine.
        screener_result: Dict returned by run_sector_screeners().
        chief_result: Dict returned by run_chief_scout().
        run_type: "premarket" | "confirmation" | "midday"
        duration: Pipeline execution duration in seconds.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    memory_key = f"run_summary:{today}:{run_type}"
    now_iso = datetime.now(timezone.utc).isoformat()

    # Build RunSummary from pipeline results
    finalists_by_sector = screener_result.get("finalists_by_sector", {})
    finalists_count = sum(
        len(candidates) for candidates in finalists_by_sector.values()
    )

    # Count total candidates evaluated (all candidates before hard gates)
    candidates_by_sector = screener_result.get("candidates_by_sector", {})
    total_candidates_evaluated = sum(
        len(candidates) for candidates in candidates_by_sector.values()
    )

    # Count hard gate rejections
    rejections = screener_result.get("rejections", [])
    hard_gate_rejections = len(rejections)

    # Extract chief scout picks
    picks = chief_result.get("picks", [])
    fallback_used = chief_result.get("fallback_used", False)

    # Get expanded watchlist symbols from picks
    expanded_watchlist_symbols = [
        p.get("symbol", "") for p in picks if p.get("symbol")
    ]

    run_summary = {
        "run_type": run_type,
        "timestamp": now_iso,
        "sectors_scanned": screener_result.get("sectors_scanned", 0),
        "total_candidates_evaluated": total_candidates_evaluated,
        "hard_gate_rejections": hard_gate_rejections,
        "finalists_count": finalists_count,
        "chief_scout_picks": picks,
        "fallback_used": fallback_used,
        "expanded_watchlist_symbols": expanded_watchlist_symbols,
        "expanded_watchlist_size": len(expanded_watchlist_symbols),
        "reason_counts": screener_result.get("reason_counts", {}),
        "budget_hits": screener_result.get("budget_hits", []),
        "duration_seconds": round(duration, 2),
    }

    db = get_session(engine)
    try:
        # Check for existing record (upsert pattern)
        existing = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout", key=memory_key)
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )

        value = json.dumps(run_summary)

        if existing:
            existing.value = value
            existing.timestamp = datetime.utcnow()
        else:
            db.add(AgentMemory(
                agent="sector_scout",
                symbol=None,
                key=memory_key,
                value=value,
            ))

        db.commit()

        logger.info(
            "Persisted run_summary: key=%s, sectors=%d, finalists=%d, picks=%d",
            memory_key,
            run_summary["sectors_scanned"],
            finalists_count,
            len(picks),
        )

    except Exception:
        db.rollback()
        logger.error("Failed to persist run_summary: key=%s", memory_key, exc_info=True)
        raise
    finally:
        db.close()


def persist_candidate_rows(
    engine,
    finalists_by_sector: dict,
    run_type: str,
) -> None:
    """Persist per-symbol CandidateRows for finalists to AgentMemory.

    Key pattern: candidate_row:{YYYY-MM-DD}:{run_type}:{symbol}

    For each finalist candidate, serializes the CandidateRow to JSON and
    stores it with a symbol-specific key for auditability.

    Args:
        engine: SQLAlchemy engine.
        finalists_by_sector: Dict mapping sector_key -> list of CandidateRow
                             objects or dicts.
        run_type: "premarket" | "confirmation" | "midday"
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    db = get_session(engine)
    try:
        persisted_count = 0

        for sector_key, finalists in finalists_by_sector.items():
            for candidate in finalists:
                # Extract symbol and serialize candidate
                if isinstance(candidate, CandidateRow):
                    symbol = candidate.symbol
                    candidate_dict = asdict(candidate)
                else:
                    symbol = candidate.get("symbol", "")
                    candidate_dict = dict(candidate)

                if not symbol:
                    continue

                memory_key = f"candidate_row:{today}:{run_type}:{symbol}"

                # Check for existing record (upsert pattern)
                existing = (
                    db.query(AgentMemory)
                    .filter_by(agent="sector_scout", key=memory_key)
                    .order_by(AgentMemory.timestamp.desc())
                    .first()
                )

                value = json.dumps(candidate_dict, default=str)

                if existing:
                    existing.value = value
                    existing.timestamp = datetime.utcnow()
                else:
                    db.add(AgentMemory(
                        agent="sector_scout",
                        symbol=symbol,
                        key=memory_key,
                        value=value,
                    ))

                persisted_count += 1

        db.commit()

        logger.info(
            "Persisted %d candidate_rows for run_type=%s",
            persisted_count,
            run_type,
        )

    except Exception:
        db.rollback()
        logger.error(
            "Failed to persist candidate_rows for run_type=%s",
            run_type,
            exc_info=True,
        )
        raise
    finally:
        db.close()
