"""
Price Monitor
Watches prices continuously and triggers actions when conditions are met.
Uses yfinance for quotes (no rate limit) to avoid burning Finnhub quota.

Checks:
  1. Stop losses on open positions
  2. Target prices on open positions
  3. Key level breaches from analyst signals (entry triggers)
"""

import json
import logging
import yfinance as yf
from datetime import datetime
from db.schema import Trade, Position, AgentMemory, get_session

log = logging.getLogger(__name__)


def get_batch_quotes(symbols: list[str]) -> dict:
    """Get current prices for multiple symbols via yfinance."""
    quotes = {}
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            price = t.fast_info.get("lastPrice")
            if price:
                quotes[sym] = round(float(price), 2)
        except Exception:
            pass
    return quotes


def check_stops_and_targets(engine) -> list[dict]:
    """Check open trades against their stop/target levels."""
    db = get_session(engine)
    open_trades = db.query(Trade).filter_by(status="open").all()
    if not open_trades:
        db.close()
        return []

    symbols = list(set(t.symbol for t in open_trades))
    quotes = get_batch_quotes(symbols)
    triggers = []

    for trade in open_trades:
        price = quotes.get(trade.symbol)
        if not price:
            continue

        pos = db.query(Position).filter_by(
            symbol=trade.symbol, profile=trade.profile
        ).first()
        side = pos.side if pos else "long"

        # Stop loss check
        if trade.stop_price:
            hit = (side == "long" and price <= trade.stop_price) or \
                  (side == "short" and price >= trade.stop_price)
            if hit:
                triggers.append({
                    "type": "stop_loss",
                    "symbol": trade.symbol,
                    "profile": trade.profile,
                    "price": price,
                    "level": trade.stop_price,
                    "side": side,
                    "trade_id": trade.id,
                })

        # Target check
        if trade.target_price:
            hit = (side == "long" and price >= trade.target_price) or \
                  (side == "short" and price <= trade.target_price)
            if hit:
                triggers.append({
                    "type": "target_hit",
                    "symbol": trade.symbol,
                    "profile": trade.profile,
                    "price": price,
                    "level": trade.target_price,
                    "side": side,
                    "trade_id": trade.id,
                })

    db.close()
    return triggers
def check_entry_triggers(engine) -> list[dict]:
    """Check analyst signals for key level breaches that could trigger entries."""
    db = get_session(engine)
    triggers = []

    # Get latest analyst signals
    seen = set()
    signals = (
        db.query(AgentMemory)
        .filter_by(agent="analyst", key="signal")
        .order_by(AgentMemory.timestamp.desc())
        .all()
    )

    signal_data = {}
    for s in signals:
        if s.symbol not in seen:
            try:
                signal_data[s.symbol] = json.loads(s.value)
            except Exception:
                pass
            seen.add(s.symbol)

    if not signal_data:
        db.close()
        return triggers

    symbols = list(signal_data.keys())
    quotes = get_batch_quotes(symbols)

    for sym, sig in signal_data.items():
        price = quotes.get(sym)
        if not price or sig.get("signal") == "HOLD":
            continue

        levels = sig.get("key_levels", {})
        strength = sig.get("strength", "weak")
        if strength == "weak":
            continue

        # Check for breakout above resistance
        resistance = levels.get("resistance")
        if sig.get("signal") == "LONG" and resistance:
            try:
                r = float(resistance)
                if price > r:
                    triggers.append({
                        "type": "breakout",
                        "symbol": sym,
                        "signal": "LONG",
                        "price": price,
                        "level": r,
                        "level_name": "resistance",
                        "strength": strength,
                        "setup_type": sig.get("setup_type"),
                    })
            except (ValueError, TypeError):
                pass

        # Check for breakdown below support
        support = levels.get("support")
        if sig.get("signal") == "SHORT" and support:
            try:
                s = float(support)
                if price < s:
                    triggers.append({
                        "type": "breakdown",
                        "symbol": sym,
                        "signal": "SHORT",
                        "price": price,
                        "level": s,
                        "level_name": "support",
                        "strength": strength,
                        "setup_type": sig.get("setup_type"),
                    })
            except (ValueError, TypeError):
                pass

    db.close()
    return triggers


# ─── MOMENTUM / CHANGE DETECTION ─────────────────────────────────────────────

# In-memory price history for change detection
_price_history = {}  # {symbol: [(timestamp, price), ...]}
ALERT_THRESHOLDS = {
    "rapid_move_pct": 1.5,       # alert if price moves >1.5% in 5 min
    "approach_level_pct": 0.3,   # alert if price is within 0.3% of a key level
    "history_window": 5,         # minutes of history to keep
}


def check_momentum(engine) -> list[dict]:
    """Detect rapid price changes and level approaches."""
    db = get_session(engine)
    alerts = []
    now = datetime.utcnow()

    # Get all symbols we care about (positions + watchlist signals)
    import os
    watchlist = [s.strip() for s in os.getenv("WATCHLIST", "SPY,QQQ,IWM,TSLA,NVDA,AMD").split(",")]

    # Add symbols from open positions
    positions = db.query(Position).all()
    pos_symbols = [p.symbol for p in positions]
    all_symbols = list(set(watchlist + pos_symbols))

    quotes = get_batch_quotes(all_symbols)

    for sym, price in quotes.items():
        if not price:
            continue

        # Update history
        if sym not in _price_history:
            _price_history[sym] = []
        _price_history[sym].append((now, price))

        # Trim to window
        cutoff = now - __import__('datetime').timedelta(minutes=ALERT_THRESHOLDS["history_window"])
        _price_history[sym] = [(t, p) for t, p in _price_history[sym] if t >= cutoff]

        # Check rapid move
        if len(_price_history[sym]) >= 2:
            oldest_price = _price_history[sym][0][1]
            change_pct = abs((price - oldest_price) / oldest_price) * 100
            if change_pct >= ALERT_THRESHOLDS["rapid_move_pct"]:
                direction = "up" if price > oldest_price else "down"
                alerts.append({
                    "type": "rapid_move",
                    "symbol": sym,
                    "price": price,
                    "change_pct": round(change_pct, 2),
                    "direction": direction,
                    "window_minutes": ALERT_THRESHOLDS["history_window"],
                })

        # Check approaching key levels from analyst signals
        sig_mem = (
            db.query(AgentMemory)
            .filter_by(agent="analyst", symbol=sym, key="signal")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if sig_mem:
            try:
                sig = json.loads(sig_mem.value)
                levels = sig.get("key_levels", {})
                for level_name, level_val in levels.items():
                    try:
                        lv = float(level_val)
                        dist_pct = abs((price - lv) / lv) * 100
                        if dist_pct <= ALERT_THRESHOLDS["approach_level_pct"]:
                            alerts.append({
                                "type": "approaching_level",
                                "symbol": sym,
                                "price": price,
                                "level_name": level_name,
                                "level_value": lv,
                                "distance_pct": round(dist_pct, 2),
                            })
                    except (ValueError, TypeError):
                        pass
            except Exception:
                pass

    db.close()
    return alerts


def run(engine) -> dict:
    """
    Full price monitor check. Returns all triggers found.
    Called every 60 seconds by the orchestrator.
    """
    stop_triggers = check_stops_and_targets(engine)
    entry_triggers = check_entry_triggers(engine)
    momentum_alerts = check_momentum(engine)

    for t in stop_triggers:
        log.warning(f"⚡ {t['type'].upper()}: {t['symbol']} ({t['profile']}) "
                     f"price={t['price']} level={t['level']}")

    for t in entry_triggers:
        log.info(f"⚡ {t['type'].upper()}: {t['symbol']} {t['signal']} "
                  f"price={t['price']} broke {t['level_name']}={t['level']}")

    for a in momentum_alerts:
        if a["type"] == "rapid_move":
            log.warning(f"📊 RAPID MOVE: {a['symbol']} {a['direction']} {a['change_pct']}% in {a['window_minutes']}min (${a['price']})")
        elif a["type"] == "approaching_level":
            log.info(f"📊 APPROACHING: {a['symbol']} ${a['price']} within {a['distance_pct']}% of {a['level_name']}={a['level_value']}")

    return {
        "stop_triggers": stop_triggers,
        "entry_triggers": entry_triggers,
        "momentum_alerts": momentum_alerts,
        "checked_at": datetime.utcnow().isoformat(),
    }


# ─── LLM-ASSISTED ALERT FILTERING ────────────────────────────────────────────

from utils.llm import call_llm, parse_json_response

FILTER_PROMPT = """You are a quick-reaction trading filter. You receive a price alert and current market context.
Decide if this alert warrants immediate PM attention or is just noise.

Respond in JSON:
{
  "actionable": true or false,
  "urgency": "high|medium|low",
  "reasoning": "one sentence why",
  "suggested_action": "BUY|SHORT|CLOSE|HOLD|none"
}

Be conservative — only flag as actionable if the move is clearly tradeable, not just volatility noise.
"""

_alert_cooldowns = {}  # {symbol: last_alert_time}
COOLDOWN_MINUTES = 15


def filter_alert_with_llm(alert: dict, engine) -> dict:
    """Use local LLM to assess whether a price alert is actionable."""
    sym = alert.get("symbol", "")

    # Cooldown check
    now = datetime.utcnow()
    last = _alert_cooldowns.get(sym)
    if last and (now - last).total_seconds() < COOLDOWN_MINUTES * 60:
        return {"actionable": False, "reasoning": "cooldown active"}

    # Get current analyst signal for context
    db = get_session(engine)
    sig_mem = (
        db.query(AgentMemory)
        .filter_by(agent="analyst", symbol=sym, key="signal")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    signal_ctx = sig_mem.value if sig_mem else "{}"
    db.close()

    user_prompt = f"""
Alert: {json.dumps(alert)}
Current analyst signal: {signal_ctx}
Time: {now.strftime('%Y-%m-%d %H:%M UTC')}

Is this alert actionable? Should the PM be notified?
"""

    try:
        raw = call_llm(FILTER_PROMPT, user_prompt, json_mode=True, tier="low")
        result = parse_json_response(raw)
        if result.get("actionable"):
            _alert_cooldowns[sym] = now
        return result
    except Exception as e:
        log.warning(f"Alert filter LLM failed: {e}")
        return {"actionable": True, "reasoning": "filter failed, passing through"}
