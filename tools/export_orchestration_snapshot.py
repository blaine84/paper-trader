#!/usr/bin/env python3
"""
Snapshot Exporter — exports a minimal JSON snapshot of recent trading activity
from the Pi database for the Mac mini diagnostic loop.

Usage:
    python tools/export_orchestration_snapshot.py \
        --days 5 \
        --trading-timezone America/New_York \
        --out reports/orchestration_snapshots/latest.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import select

# Ensure project root is on sys.path so `db.schema` resolves when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from db.schema import init_db, get_session, Trade, TradeEvent, DynamicStrategy

log = logging.getLogger("snapshot_exporter")

# ---------------------------------------------------------------------------
# Configurable constants
# ---------------------------------------------------------------------------

# Corrected event types based on live database audit (Task 1):
#   stop_triggered  (NOT stop_loss)
#   target_triggered (NOT target_hit)
#   exit_filled     (replaces non-existent overlap_rejected)
ALLOWED_EVENT_TYPES: list[str] = [
    "entry_requested",
    "entry_filled",
    "stop_triggered",
    "target_triggered",
    "exit_filled",
]

TRADE_COLUMNS: list[str] = [
    "id", "symbol", "profile", "direction", "setup_type",
    "entry_time", "exit_time", "status", "entry_price", "exit_price",
    "pnl", "pnl_pct", "reason_exit",
]

DYNAMIC_STRATEGY_COLUMNS: list[str] = [
    "id", "name", "status", "pipeline_stage",
]

SNAPSHOT_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Core export logic (importable for testing)
# ---------------------------------------------------------------------------

def export_snapshot(
    engine,
    days: int = 5,
    trading_timezone: str = "America/New_York",
    allowed_event_types: list[str] | None = None,
) -> dict:
    """
    Query the database and return a snapshot dict.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        Database engine to query.
    days : int
        Number of calendar days to look back (in the trading timezone).
    trading_timezone : str
        IANA timezone string for interpreting the day boundary.
    allowed_event_types : list[str] | None
        Event types to include. Defaults to ``ALLOWED_EVENT_TYPES``.

    Returns
    -------
    dict
        The snapshot payload ready to be serialised as JSON.
    """
    if allowed_event_types is None:
        allowed_event_types = list(ALLOWED_EVENT_TYPES)

    tz = ZoneInfo(trading_timezone)

    # Compute the cutoff: midnight N days ago in the trading timezone,
    # then convert to a naive UTC datetime for comparison with DB values.
    now_tz = datetime.now(tz)
    cutoff_local = now_tz.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days - 1)
    cutoff_utc_naive = cutoff_local.astimezone(timezone.utc).replace(tzinfo=None)

    session = get_session(engine)
    try:
        # --- Trades -----------------------------------------------------------
        trades_query = (
            select(Trade)
            .where(Trade.entry_time >= cutoff_utc_naive)
        )
        trade_rows = session.execute(trades_query).scalars().all()

        trades_out = []
        for t in trade_rows:
            trades_out.append(_project_trade(t))

        if not trades_out:
            log.warning("Zero trades found in the %d-day window — snapshot will have empty arrays.", days)

        # --- Trade Events -----------------------------------------------------
        event_query = (
            select(TradeEvent)
            .where(TradeEvent.event_type.in_(allowed_event_types))
        )
        event_rows = session.execute(event_query).scalars().all()
        events_out = [_project_event(e) for e in event_rows]

        # --- Dynamic Strategies -----------------------------------------------
        strat_query = select(DynamicStrategy)
        strat_rows = session.execute(strat_query).scalars().all()
        strats_out = [_project_strategy(s) for s in strat_rows]

    finally:
        session.close()

    snapshot = {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_time": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "days": days,
            "tables": ["trades", "trade_events", "dynamic_strategies"],
            "diagnostic": "same_symbol_overlap",
        },
        "trades": trades_out,
        "trade_events": {
            "included_event_types": allowed_event_types,
            "rows": events_out,
        },
        "dynamic_strategies": strats_out,
    }
    return snapshot


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def _format_datetime(dt) -> str | None:
    """Convert a datetime to an ISO-friendly string, or None."""
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return str(dt)


def _project_trade(t: Trade) -> dict:
    return {
        "id": t.id,
        "symbol": t.symbol,
        "profile": t.profile,
        "direction": t.direction,
        "setup_type": t.setup_type,
        "entry_time": _format_datetime(t.entry_time),
        "exit_time": _format_datetime(t.exit_time),
        "status": t.status,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "pnl": t.pnl,
        "pnl_pct": t.pnl_pct,
        "reason_exit": t.reason_exit,
    }


def _project_event(e: TradeEvent) -> dict:
    return {
        "id": e.id,
        "trade_id": e.trade_id,
        "timestamp": _format_datetime(e.timestamp),
        "event_type": e.event_type,
        "agent": e.agent,
        "symbol": e.symbol,
        "profile": e.profile,
        "message": e.message,
    }


def _project_strategy(s: DynamicStrategy) -> dict:
    return {
        "id": s.id,
        "name": s.name,
        "status": s.status,
        "pipeline_stage": s.pipeline_stage,
    }


# ---------------------------------------------------------------------------
# File writing (atomic)
# ---------------------------------------------------------------------------

def write_snapshot(snapshot: dict, out_path: str) -> None:
    """Write snapshot JSON atomically (write to .tmp, then os.replace)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(out) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=str)
        f.write("\n")
    os.replace(tmp_path, str(out))
    log.info("Snapshot written to %s", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a minimal JSON snapshot of recent trading activity.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=5,
        help="Number of calendar days to include (default: 5).",
    )
    parser.add_argument(
        "--trading-timezone",
        type=str,
        default="America/New_York",
        help="IANA timezone for day-boundary calculation (default: America/New_York).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="reports/orchestration_snapshots/latest.json",
        help="Output file path (default: reports/orchestration_snapshots/latest.json).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    log.info(
        "Exporting snapshot: days=%d, timezone=%s, out=%s",
        args.days, args.trading_timezone, args.out,
    )

    engine = init_db()
    snapshot = export_snapshot(
        engine,
        days=args.days,
        trading_timezone=args.trading_timezone,
    )
    write_snapshot(snapshot, args.out)

    trade_count = len(snapshot["trades"])
    event_count = len(snapshot["trade_events"]["rows"])
    strat_count = len(snapshot["dynamic_strategies"])
    log.info(
        "Snapshot complete: %d trades, %d events, %d strategies.",
        trade_count, event_count, strat_count,
    )


if __name__ == "__main__":
    main()
