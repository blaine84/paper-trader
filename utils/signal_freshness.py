"""Signal Freshness Gate — validates analyst signals before PM evaluation.

This module implements the freshness gate that runs as a coordinator phase
BEFORE PM decisioning. The coordinator passes only fresh symbols into PM;
PM itself has no freshness awareness.

The freshness gate is **fail-closed**: if we cannot verify freshness for a
symbol (DB error, parse error), that symbol is EXCLUDED from PM evaluation.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StaleSignalSkip:
    """Record of a symbol skipped due to stale or missing signal.

    Attributes:
        cycle_id: The current market cycle identifier.
        symbol: The symbol that was skipped.
        signal_age_seconds: Age of the signal in seconds (0.0 for missing).
        freshness_threshold_seconds: The configured freshness window.
        signal_timestamp: Timestamp of the signal (epoch for missing).
        reason: Classification of why the symbol was skipped.
    """

    cycle_id: str
    symbol: str
    signal_age_seconds: float
    freshness_threshold_seconds: int
    signal_timestamp: datetime
    reason: str  # "stale_signal" | "missing_signal" | "previous_cycle" | "freshness_gate_error"


@dataclass(frozen=True)
class FreshnessResult:
    """Result of freshness gate evaluation.

    Attributes:
        fresh_symbols: Symbols with fresh analyst signals (pass to PM).
        stale_symbols: Symbols with signals that are too old.
        missing_symbols: Symbols with no analyst signal at all.
        error_symbols: Symbols where DB error prevented evaluation (excluded from PM).
        skip_events: Detailed skip records for observability.
    """

    fresh_symbols: tuple[str, ...]
    stale_symbols: tuple[str, ...]
    missing_symbols: tuple[str, ...]
    error_symbols: tuple[str, ...]
    skip_events: tuple[StaleSignalSkip, ...]


def check_signal_freshness(
    engine,
    symbols: list[str],
    cycle_id: str,
    freshness_window_seconds: int = 120,
) -> FreshnessResult:
    """Check whether analyst signals are fresh enough for PM evaluation.

    A signal is considered fresh if:
    - It was produced during the current cycle (cycle_id matches), OR
    - Its timestamp is within freshness_window_seconds of now.

    Failure behavior (fail-closed):
    - On DB error, ALL symbols in the query are classified as
      `freshness_gate_error` and EXCLUDED from PM evaluation.
    - On JSON parse error for a single symbol, only THAT symbol is
      classified as `freshness_gate_error`.

    Args:
        engine: SQLAlchemy engine for database access.
        symbols: List of symbols to evaluate.
        cycle_id: The current market cycle identifier.
        freshness_window_seconds: Maximum signal age (seconds) to accept.

    Returns:
        FreshnessResult with categorized symbols and skip events.
    """
    if not symbols:
        return FreshnessResult(
            fresh_symbols=(),
            stale_symbols=(),
            missing_symbols=(),
            error_symbols=(),
            skip_events=(),
        )

    now = datetime.now(timezone.utc)
    fresh: list[str] = []
    stale: list[str] = []
    missing: list[str] = []
    error: list[str] = []
    skip_events: list[StaleSignalSkip] = []

    # Query latest analyst signal per symbol.
    # SQLAlchemy text() doesn't support IN with list params directly,
    # so we build placeholders dynamically.
    try:
        signal_map = _fetch_latest_signals(engine, symbols)
    except Exception as exc:
        # DB error — fail-closed: classify ALL symbols as error
        logger.error(
            "Freshness gate DB error for cycle_id=%s: %s. "
            "All %d symbols classified as freshness_gate_error (fail-closed).",
            cycle_id,
            exc,
            len(symbols),
        )
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        for sym in symbols:
            error.append(sym)
            skip_events.append(StaleSignalSkip(
                cycle_id=cycle_id,
                symbol=sym,
                signal_age_seconds=0.0,
                freshness_threshold_seconds=freshness_window_seconds,
                signal_timestamp=epoch,
                reason="freshness_gate_error",
            ))
        return FreshnessResult(
            fresh_symbols=(),
            stale_symbols=(),
            missing_symbols=(),
            error_symbols=tuple(error),
            skip_events=tuple(skip_events),
        )

    # Classify each symbol
    for sym in symbols:
        row = signal_map.get(sym)

        if row is None:
            # No signal found for this symbol
            missing.append(sym)
            skip_events.append(StaleSignalSkip(
                cycle_id=cycle_id,
                symbol=sym,
                signal_age_seconds=0.0,
                freshness_threshold_seconds=freshness_window_seconds,
                signal_timestamp=datetime(1970, 1, 1, tzinfo=timezone.utc),
                reason="missing_signal",
            ))
            continue

        signal_timestamp, signal_value = row

        # Parse JSON to extract _cycle_id
        try:
            signal_data = json.loads(signal_value)
        except (json.JSONDecodeError, TypeError) as exc:
            # JSON parse error — fail-closed for this symbol only
            logger.warning(
                "Freshness gate JSON parse error for symbol=%s cycle_id=%s: %s. "
                "Classifying as freshness_gate_error (fail-closed).",
                sym,
                cycle_id,
                exc,
            )
            error.append(sym)
            skip_events.append(StaleSignalSkip(
                cycle_id=cycle_id,
                symbol=sym,
                signal_age_seconds=0.0,
                freshness_threshold_seconds=freshness_window_seconds,
                signal_timestamp=signal_timestamp,
                reason="freshness_gate_error",
            ))
            continue

        signal_cycle_id = signal_data.get("_cycle_id")

        # Check 1: cycle_id match → FRESH (produced this cycle)
        if signal_cycle_id == cycle_id:
            fresh.append(sym)
            continue

        # Check 2: within freshness window → FRESH (recent enough)
        # signal_timestamp is already timezone-aware from _parse_timestamp
        signal_age = (now - signal_timestamp).total_seconds()

        if signal_age < freshness_window_seconds:
            fresh.append(sym)
            continue

        # Signal is stale — determine reason
        reason = "stale_signal"
        if signal_cycle_id is not None and signal_cycle_id != cycle_id:
            reason = "previous_cycle"

        stale.append(sym)
        skip_events.append(StaleSignalSkip(
            cycle_id=cycle_id,
            symbol=sym,
            signal_age_seconds=signal_age,
            freshness_threshold_seconds=freshness_window_seconds,
            signal_timestamp=signal_timestamp,
            reason=reason,
        ))

    logger.info(
        "Freshness gate result for cycle_id=%s: fresh=%d stale=%d missing=%d error=%d",
        cycle_id,
        len(fresh),
        len(stale),
        len(missing),
        len(error),
    )

    return FreshnessResult(
        fresh_symbols=tuple(fresh),
        stale_symbols=tuple(stale),
        missing_symbols=tuple(missing),
        error_symbols=tuple(error),
        skip_events=tuple(skip_events),
    )


def _parse_timestamp(raw_ts) -> datetime:
    """Parse a timestamp value from the database.

    SQLite with raw text() queries may return timestamps as strings
    rather than datetime objects. This handles both cases.

    Args:
        raw_ts: Either a datetime object or an ISO-format string.

    Returns:
        A timezone-aware datetime (UTC assumed if naive).
    """
    if isinstance(raw_ts, datetime):
        if raw_ts.tzinfo is None:
            return raw_ts.replace(tzinfo=timezone.utc)
        return raw_ts

    # Parse ISO-format string from SQLite
    ts_str = str(raw_ts)
    # Try common SQLite datetime formats
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(ts_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    # Last resort: fromisoformat (Python 3.11+)
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _fetch_latest_signals(
    engine,
    symbols: list[str],
) -> dict[str, tuple[datetime, str]]:
    """Fetch the latest analyst signal per symbol from AgentMemory.

    Uses a single query with dynamic placeholders to fetch all symbols
    at once, then groups by symbol taking the latest by timestamp.

    Args:
        engine: SQLAlchemy engine.
        symbols: List of symbols to query.

    Returns:
        Dict mapping symbol → (timestamp, value_json) for the latest signal.

    Raises:
        Any database exception (caller handles fail-closed).
    """
    # Build dynamic placeholder list for IN clause
    placeholders = ", ".join(f":sym_{i}" for i in range(len(symbols)))
    params: dict[str, str] = {f"sym_{i}": sym for i, sym in enumerate(symbols)}
    params["agent"] = "analyst"
    params["key"] = "signal"

    query = text(f"""
        SELECT symbol, timestamp, value
        FROM agent_memory
        WHERE agent = :agent
          AND key = :key
          AND symbol IN ({placeholders})
        ORDER BY timestamp DESC
    """)

    signal_map: dict[str, tuple[datetime, str]] = {}

    with engine.connect() as conn:
        result = conn.execute(query, params)
        for row in result:
            sym = row[0]
            # Only keep the first (latest) row per symbol due to ORDER BY DESC
            if sym not in signal_map:
                ts = _parse_timestamp(row[1])
                signal_map[sym] = (ts, row[2])

    return signal_map
