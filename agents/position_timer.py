"""
Position Timer Monitor
Enforces time-based exits on intraday setups.

Rules:
  - Alert PM after 60 min (configurable per setup type)
  - Force close after 90 min
  - Hard wall: close all intraday positions at 3:45 PM ET regardless
"""

import json
import logging
from datetime import datetime
from pytz import timezone

from db.schema import Trade, Position, AgentMemory, get_session

log = logging.getLogger(__name__)

# Max hold times per setup type (minutes)
SETUP_TIME_LIMITS = {
    "momentum_fade": {"alert": 45, "force_close": 75},
    "gap_and_go": {"alert": 60, "force_close": 90},
    "vwap_reclaim": {"alert": 60, "force_close": 90},
    "orb": {"alert": 45, "force_close": 75},
    "trend_pullback": {"alert": 90, "force_close": 120},
    "news_catalyst": {"alert": 60, "force_close": 90},
    "short_squeeze": {"alert": 30, "force_close": 60},
}

# Default for setup types not listed above
DEFAULT_LIMITS = {"alert": 60, "force_close": 90}

# All intraday setups — positions with these types get the 3:45 PM hard wall
INTRADAY_SETUPS = set(SETUP_TIME_LIMITS.keys())

# Hard wall time (ET)
HARD_WALL_HOUR = 15
HARD_WALL_MINUTE = 45


def _get_setup_type_for_trade(db, trade) -> str:
    """Look up the analyst's setup_type for this trade's symbol at entry time."""
    mem = (
        db.query(AgentMemory)
        .filter_by(agent="analyst", symbol=trade.symbol, key="signal")
        .filter(AgentMemory.timestamp <= trade.entry_time)
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    if mem:
        try:
            return json.loads(mem.value).get("setup_type", "")
        except Exception:
            pass
    # Fallback: check the most recent signal
    mem = (
        db.query(AgentMemory)
        .filter_by(agent="analyst", symbol=trade.symbol, key="signal")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    if mem:
        try:
            return json.loads(mem.value).get("setup_type", "")
        except Exception:
            pass
    return ""


def _close_position(engine, trade, position, price, reason):
    """Force close a position."""
    from agents.portfolio_manager import execute_trade
    db = get_session(engine)
    try:
        execute_trade(db, {
            "symbol": trade.symbol,
            "action": "CLOSE",
            "quantity": 0,
            "price": price,
            "rationale": reason,
        }, trade.profile)
        log.warning(f"⏰ FORCE CLOSED: {trade.symbol} ({trade.profile}) — {reason}")
    except Exception as e:
        log.error(f"Force close failed for {trade.symbol}: {e}")
    finally:
        db.close()


def run(engine) -> dict:
    """Check all open positions for time-based exit conditions."""
    db = get_session(engine)
    et_tz = timezone("America/New_York")
    now_et = datetime.now(et_tz)
    now_utc = datetime.utcnow()

    open_trades = db.query(Trade).filter_by(status="open").all()
    if not open_trades:
        db.close()
        return {"alerts": [], "force_closes": [], "hard_wall_closes": []}

    alerts = []
    force_closes = []
    hard_wall_closes = []

    # Check hard wall first
    past_hard_wall = (now_et.hour > HARD_WALL_HOUR or
                      (now_et.hour == HARD_WALL_HOUR and now_et.minute >= HARD_WALL_MINUTE))

    for trade in open_trades:
        if not trade.entry_time:
            continue

        setup_type = _get_setup_type_for_trade(db, trade)
        is_intraday = setup_type in INTRADAY_SETUPS or setup_type == ""  # default to intraday

        # Calculate hold time
        minutes_held = (now_utc - trade.entry_time).total_seconds() / 60

        # Get current price
        try:
            import yfinance as yf
            price = float(yf.Ticker(trade.symbol).fast_info.get("lastPrice", trade.entry_price))
        except Exception:
            price = trade.entry_price

        # Hard wall: 3:45 PM ET — close all intraday positions
        if past_hard_wall and is_intraday:
            hard_wall_closes.append({
                "symbol": trade.symbol,
                "profile": trade.profile,
                "minutes_held": round(minutes_held),
                "setup_type": setup_type,
            })
            _close_position(engine, trade, None, price,
                            f"Hard wall 3:45 PM ET: intraday {setup_type} held {round(minutes_held)} min")
            continue

        # Time limits per setup type
        limits = SETUP_TIME_LIMITS.get(setup_type, DEFAULT_LIMITS)

        # Force close
        if minutes_held > limits["force_close"]:
            force_closes.append({
                "symbol": trade.symbol,
                "profile": trade.profile,
                "minutes_held": round(minutes_held),
                "limit": limits["force_close"],
                "setup_type": setup_type,
            })
            _close_position(engine, trade, None, price,
                            f"Time-based forced exit: held {round(minutes_held)} min on {setup_type} setup (limit: {limits['force_close']} min)")
            continue

        # Alert (store in agent memory for PM to read)
        if minutes_held > limits["alert"]:
            alerts.append({
                "symbol": trade.symbol,
                "profile": trade.profile,
                "minutes_held": round(minutes_held),
                "alert_limit": limits["alert"],
                "force_limit": limits["force_close"],
                "setup_type": setup_type,
            })

    # Store alerts for PM to read
    if alerts:
        db2 = get_session(engine)
        db2.add(AgentMemory(
            agent="position_timer",
            symbol=None,
            key="time_alerts",
            value=json.dumps(alerts),
        ))
        db2.commit()
        db2.close()
        for a in alerts:
            log.warning(f"⏰ TIME ALERT: {a['symbol']} ({a['profile']}) held {a['minutes_held']} min "
                        f"on {a['setup_type']} (alert: {a['alert_limit']}, force: {a['force_limit']})")

    db.close()

    return {
        "alerts": alerts,
        "force_closes": force_closes,
        "hard_wall_closes": hard_wall_closes,
    }
