"""
Position Timer Monitor
Enforces time-based exits on intraday setups.

General rules:
  - Alert PM after setup-specific alert time
  - Force close after setup-specific force time
  - Hard wall: close all intraday positions at 3:45 PM ET

Momentum Fade specific:
  - 30-40 min: mark as "stale" if <0.5R achieved
  - 45 min: alert if stale
  - 60 min: thesis revalidation via LLM (below VWAP? volume fading? lower highs?)
  - If revalidation fails: exit immediately
  - 75 min: force exit regardless
  - Suppress duplicate warnings unless status escalates
"""

import json
import logging
from datetime import datetime
from pytz import timezone

from db.schema import Trade, Position, AgentMemory, get_session
from utils.finnhub_client import FinnhubClient

log = logging.getLogger(__name__)

SETUP_TIME_LIMITS = {
    "momentum_fade": {"stale": 35, "alert": 45, "revalidate": 60, "force_close": 75},
    "gap_and_go":    {"alert": 60, "force_close": 90},
    "vwap_reclaim":  {"alert": 60, "force_close": 90},
    "orb":           {"alert": 45, "force_close": 75},
    "trend_pullback": {"alert": 90, "force_close": 120},
    "news_catalyst": {"alert": 60, "force_close": 90},
    "short_squeeze": {"alert": 30, "force_close": 60},
}

DEFAULT_LIMITS = {"alert": 60, "force_close": 90}
INTRADAY_SETUPS = set(SETUP_TIME_LIMITS.keys())
HARD_WALL_HOUR = 15
HARD_WALL_MINUTE = 45

# Track alert status per trade to suppress duplicates
# {trade_id: "stale" | "alert" | "revalidating" | "force_close"}
_alert_status = {}


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


def _get_current_price(symbol, fallback):
    try:
        import yfinance as yf
        return float(yf.Ticker(symbol).fast_info.get("lastPrice", fallback))
    except Exception:
        return fallback


def _close_position(engine, trade, price, reason):
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
    # Clear alert status
    _alert_status.pop(trade.id, None)


def _calculate_r_achieved(trade, current_price):
    """Calculate how much R (risk units) the trade has achieved."""
    if not trade.stop_price or not trade.entry_price:
        return None
    risk = abs(trade.entry_price - trade.stop_price)
    if risk == 0:
        return None
    if trade.direction == "LONG":
        move = current_price - trade.entry_price
    else:
        move = trade.entry_price - current_price
    return round(move / risk, 2)


def _revalidate_momentum_fade(engine, trade, price) -> bool:
    """
    LLM thesis revalidation for momentum_fade at 60 min.
    Returns True if thesis still valid, False if should exit.
    Checks: below VWAP? volume fading? lower highs/lower lows intact?
    """
    from utils.llm import call_llm, parse_json_response
    from utils.technicals import compute_indicators

    fh = FinnhubClient()
    candles = fh.get_candles(trade.symbol, resolution="5", days=1)
    indicators = compute_indicators(candles)

    if not indicators or "error" in indicators:
        log.warning(f"Revalidation: no indicator data for {trade.symbol}, failing safe → exit")
        return False

    prompt = f"""You are validating whether a momentum_fade SHORT trade thesis is still intact.

Trade: {trade.direction} {trade.symbol} entered at ${trade.entry_price:.2f}, now ${price:.2f}
Held for 60 minutes. Stop: ${trade.stop_price or 'none'}, Target: ${trade.target_price or 'none'}

Current indicators:
  RSI: {indicators.get('rsi')}
  VWAP: {indicators.get('vwap')} (price {'below' if price < indicators.get('vwap', 0) else 'above'} VWAP)
  EMA trend: {indicators.get('trend')}
  MACD: {indicators.get('macd_cross')}
  BB position: price at {indicators.get('bb_lower', 0):.2f} - {indicators.get('bb_upper', 0):.2f}

Is the momentum fade thesis still valid? Check:
1. Is price still below VWAP? (required for short thesis)
2. Is selling pressure intact? (bearish MACD, RSI not recovering above 50)
3. Are lower highs / lower lows intact on the 5-min chart?

Respond in JSON:
{{"valid": true or false, "reasoning": "one sentence", "confidence": "high|medium|low"}}
"""

    try:
        raw = call_llm(
            "You are a trade thesis validator. Be strict — if in doubt, say invalid.",
            prompt, json_mode=True, tier="medium"
        )
        result = parse_json_response(raw)
        valid = result.get("valid", False)
        log.info(f"Revalidation {trade.symbol}: {'VALID' if valid else 'INVALID'} — {result.get('reasoning', '')}")
        return valid
    except Exception as e:
        log.warning(f"Revalidation LLM failed for {trade.symbol}: {e} — failing safe → exit")
        return False


def _escalate(trade_id, new_status) -> bool:
    """Returns True if this is a new escalation (status changed). Suppresses duplicates."""
    old = _alert_status.get(trade_id)
    levels = {"stale": 0, "alert": 1, "revalidating": 2, "force_close": 3}
    if old and levels.get(old, -1) >= levels.get(new_status, -1):
        return False  # already at this level or higher
    _alert_status[trade_id] = new_status
    return True


def run(engine) -> dict:
    """Check all open positions for time-based exit conditions."""
    db = get_session(engine)
    et_tz = timezone("America/New_York")
    now_et = datetime.now(et_tz)
    now_utc = datetime.utcnow()

    open_trades = db.query(Trade).filter_by(status="open").all()
    if not open_trades:
        db.close()
        return {"alerts": [], "force_closes": [], "hard_wall_closes": [], "stale": [], "revalidations": []}

    # Snapshot trade data before closing session
    trade_data = []
    for t in open_trades:
        if not t.entry_time:
            continue
        trade_data.append({
            "id": t.id, "symbol": t.symbol, "profile": t.profile,
            "direction": t.direction, "entry_price": t.entry_price,
            "entry_time": t.entry_time, "stop_price": t.stop_price,
            "target_price": t.target_price,
            "setup_type": _get_setup_type_for_trade(db, t),
        })
    db.close()

    alerts = []
    force_closes = []
    hard_wall_closes = []
    stale_trades = []
    revalidations = []

    past_hard_wall = (now_et.hour > HARD_WALL_HOUR or
                      (now_et.hour == HARD_WALL_HOUR and now_et.minute >= HARD_WALL_MINUTE))

    for td in trade_data:
        minutes_held = (now_utc - td["entry_time"]).total_seconds() / 60
        setup_type = td["setup_type"]
        is_intraday = setup_type in INTRADAY_SETUPS or setup_type == ""
        price = _get_current_price(td["symbol"], td["entry_price"])
        limits = SETUP_TIME_LIMITS.get(setup_type, DEFAULT_LIMITS)

        # Reconstruct a minimal trade object for _close_position
        class _T:
            pass
        trade = _T()
        trade.id = td["id"]; trade.symbol = td["symbol"]; trade.profile = td["profile"]
        trade.direction = td["direction"]; trade.entry_price = td["entry_price"]
        trade.stop_price = td["stop_price"]; trade.target_price = td["target_price"]

        # Hard wall: 3:45 PM ET
        if past_hard_wall and is_intraday:
            hard_wall_closes.append({"symbol": td["symbol"], "profile": td["profile"],
                                     "minutes_held": round(minutes_held), "setup_type": setup_type})
            _close_position(engine, trade, price,
                            f"Hard wall 3:45 PM ET: {setup_type} held {round(minutes_held)} min")
            continue

        # ── MOMENTUM FADE SPECIFIC LOGIC ──
        if setup_type == "momentum_fade":
            r_achieved = _calculate_r_achieved(trade, price)

            # 30-40 min: stale check
            if minutes_held >= limits.get("stale", 35) and (r_achieved is None or r_achieved < 0.5):
                if _escalate(td["id"], "stale"):
                    stale_trades.append({"symbol": td["symbol"], "profile": td["profile"],
                                         "minutes_held": round(minutes_held), "r_achieved": r_achieved})
                    log.warning(f"⏰ STALE: {td['symbol']} ({td['profile']}) {round(minutes_held)} min, "
                                f"R achieved: {r_achieved} (<0.5R)")

            # 45 min: alert if stale
            if minutes_held >= limits["alert"] and _alert_status.get(td["id"]) == "stale":
                if _escalate(td["id"], "alert"):
                    alerts.append({"symbol": td["symbol"], "profile": td["profile"],
                                   "minutes_held": round(minutes_held), "setup_type": setup_type,
                                   "r_achieved": r_achieved, "status": "stale_alert"})
                    log.warning(f"⏰ STALE ALERT: {td['symbol']} ({td['profile']}) stale for {round(minutes_held)} min")

            # 60 min: thesis revalidation
            if minutes_held >= limits.get("revalidate", 60) and _alert_status.get(td["id"]) != "revalidating":
                if _escalate(td["id"], "revalidating"):
                    log.info(f"⏰ REVALIDATING: {td['symbol']} ({td['profile']}) at {round(minutes_held)} min")
                    valid = _revalidate_momentum_fade(engine, trade, price)
                    revalidations.append({"symbol": td["symbol"], "profile": td["profile"],
                                          "minutes_held": round(minutes_held), "valid": valid})
                    if not valid:
                        _close_position(engine, trade, price,
                                        f"Thesis revalidation FAILED at {round(minutes_held)} min: momentum_fade no longer valid")
                        continue

            # 75 min: force close regardless
            if minutes_held >= limits["force_close"]:
                if _escalate(td["id"], "force_close"):
                    force_closes.append({"symbol": td["symbol"], "profile": td["profile"],
                                         "minutes_held": round(minutes_held), "setup_type": setup_type})
                    _close_position(engine, trade, price,
                                    f"Time-based forced exit: momentum_fade held {round(minutes_held)} min (max: {limits['force_close']})")
            continue

        # ── GENERIC SETUP LOGIC ──
        if minutes_held > limits["force_close"]:
            force_closes.append({"symbol": td["symbol"], "profile": td["profile"],
                                 "minutes_held": round(minutes_held), "setup_type": setup_type})
            _close_position(engine, trade, price,
                            f"Time-based forced exit: {setup_type} held {round(minutes_held)} min (limit: {limits['force_close']})")
            continue

        if minutes_held > limits["alert"]:
            if _escalate(td["id"], "alert"):
                alerts.append({"symbol": td["symbol"], "profile": td["profile"],
                               "minutes_held": round(minutes_held), "setup_type": setup_type,
                               "alert_limit": limits["alert"], "force_limit": limits["force_close"]})
                log.warning(f"⏰ TIME ALERT: {td['symbol']} ({td['profile']}) held {round(minutes_held)} min "
                            f"on {setup_type} (alert: {limits['alert']}, force: {limits['force_close']})")

    # Store alerts for PM
    if alerts or stale_trades:
        db2 = get_session(engine)
        db2.add(AgentMemory(
            agent="position_timer",
            symbol=None,
            key="time_alerts",
            value=json.dumps({"alerts": alerts, "stale": stale_trades}),
        ))
        db2.commit()
        db2.close()

    # Clean up status for trades that are no longer open
    open_ids = {td["id"] for td in trade_data}
    for tid in list(_alert_status.keys()):
        if tid not in open_ids:
            del _alert_status[tid]

    return {
        "alerts": alerts,
        "force_closes": force_closes,
        "hard_wall_closes": hard_wall_closes,
        "stale": stale_trades,
        "revalidations": revalidations,
    }
