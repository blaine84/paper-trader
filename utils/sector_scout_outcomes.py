"""Outcome tracking for expanded watchlist candidates.

Tracks the full lifecycle of each expanded candidate through the pipeline:
  1. Analyst signal (LONG/SHORT/HOLD/NO_SIGNAL)
  2. PM status (eligible/rejected/executed/no_entry)
  3. Trade outcome (pnl_pct, direction, entry_price, exit_price, etc.)

Key pattern: scout_outcome:{YYYY-MM-DD}:{symbol}

Each outcome record is stored in AgentMemory with agent="sector_scout" and
is updated incrementally as the candidate progresses through the pipeline.
Only symbols that appear in the expanded watchlist for the given date are tracked.

See: design.md §7 (Observability), requirements.md §10.5
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from db.schema import AgentMemory, get_session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def record_analyst_outcome(engine, symbol: str, date: str, signal: str) -> None:
    """Record that an expanded candidate reached Analyst LONG/SHORT/HOLD status.

    Only records if the symbol is in the expanded watchlist for the given date.

    Args:
        engine: SQLAlchemy engine.
        symbol: The symbol that was analyzed.
        date: Date string (YYYY-MM-DD).
        signal: Analyst signal ("LONG", "SHORT", "HOLD", "NO_SIGNAL").
    """
    if not _is_expanded_candidate(engine, symbol, date):
        return

    memory_key = f"scout_outcome:{date}:{symbol}"
    record = _get_or_create_outcome(engine, symbol, date)
    record["analyst_signal"] = signal
    record["analyst_recorded_at"] = datetime.now(timezone.utc).isoformat()
    _save_outcome(engine, memory_key, symbol, record)

    logger.debug(
        "Recorded analyst outcome: symbol=%s, date=%s, signal=%s",
        symbol, date, signal,
    )


def record_pm_outcome(engine, symbol: str, date: str, status: str) -> None:
    """Record PM eligible/rejected/executed status for an expanded candidate.

    Only records if the symbol is in the expanded watchlist for the given date.

    Args:
        engine: SQLAlchemy engine.
        symbol: The symbol.
        date: Date string (YYYY-MM-DD).
        status: PM status ("eligible", "rejected", "executed", "no_entry").
    """
    if not _is_expanded_candidate(engine, symbol, date):
        return

    memory_key = f"scout_outcome:{date}:{symbol}"
    record = _get_or_create_outcome(engine, symbol, date)
    record["pm_status"] = status
    record["pm_recorded_at"] = datetime.now(timezone.utc).isoformat()
    _save_outcome(engine, memory_key, symbol, record)

    logger.debug(
        "Recorded PM outcome: symbol=%s, date=%s, status=%s",
        symbol, date, status,
    )


def record_trade_outcome(engine, symbol: str, date: str, outcome: dict) -> None:
    """Record eventual trade outcome for an expanded candidate.

    Only records if the symbol is in the expanded watchlist for the given date.

    Args:
        engine: SQLAlchemy engine.
        symbol: The symbol.
        date: Date string (YYYY-MM-DD).
        outcome: Dict with trade result (pnl_pct, direction, entry_price,
                 exit_price, etc.)
    """
    if not _is_expanded_candidate(engine, symbol, date):
        return

    memory_key = f"scout_outcome:{date}:{symbol}"
    record = _get_or_create_outcome(engine, symbol, date)
    record["trade_outcome"] = outcome
    record["trade_recorded_at"] = datetime.now(timezone.utc).isoformat()
    _save_outcome(engine, memory_key, symbol, record)

    logger.debug(
        "Recorded trade outcome: symbol=%s, date=%s, pnl_pct=%s",
        symbol, date, outcome.get("pnl_pct"),
    )


def get_candidate_outcomes(engine, date: str) -> list[dict]:
    """Get all outcome records for expanded candidates on a given date.

    Args:
        engine: SQLAlchemy engine.
        date: Date string (YYYY-MM-DD).

    Returns:
        List of outcome dicts, each containing symbol, date, analyst_signal,
        pm_status, and trade_outcome fields (where recorded).
    """
    key_prefix = f"scout_outcome:{date}:"

    db = get_session(engine)
    try:
        records = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout")
            .filter(AgentMemory.key.like(f"{key_prefix}%"))
            .all()
        )

        outcomes = []
        for record in records:
            try:
                data = json.loads(record.value)
                outcomes.append(data)
            except (json.JSONDecodeError, TypeError):
                continue

        return outcomes

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _is_expanded_candidate(engine, symbol: str, date: str) -> bool:
    """Check if a symbol is in the expanded watchlist for the given date."""
    watchlist_key = f"expanded_watchlist:{date}"

    db = get_session(engine)
    try:
        record = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout", key=watchlist_key)
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )

        if not record:
            return False

        try:
            data = json.loads(record.value)
        except (json.JSONDecodeError, TypeError):
            return False

        symbols = data.get("symbols", [])
        return symbol in symbols

    finally:
        db.close()


def _get_or_create_outcome(engine, symbol: str, date: str) -> dict:
    """Get existing outcome record or create a new empty one."""
    memory_key = f"scout_outcome:{date}:{symbol}"

    db = get_session(engine)
    try:
        existing = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout", key=memory_key)
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )

        if existing:
            try:
                return json.loads(existing.value)
            except (json.JSONDecodeError, TypeError):
                pass

        # Return a fresh outcome record
        return {
            "symbol": symbol,
            "date": date,
            "analyst_signal": None,
            "analyst_recorded_at": None,
            "pm_status": None,
            "pm_recorded_at": None,
            "trade_outcome": None,
            "trade_recorded_at": None,
        }

    finally:
        db.close()


def _save_outcome(engine, memory_key: str, symbol: str, record: dict) -> None:
    """Save an outcome record to AgentMemory (upsert pattern)."""
    db = get_session(engine)
    try:
        existing = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout", key=memory_key)
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )

        value = json.dumps(record, default=str)

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

        db.commit()

    except Exception:
        db.rollback()
        logger.error(
            "Failed to save outcome: key=%s", memory_key, exc_info=True
        )
        raise
    finally:
        db.close()
