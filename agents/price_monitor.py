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


def run(engine) -> dict:
    """
    Full price monitor check. Returns all triggers found.
    Called every 60 seconds by the orchestrator.
    """
    stop_triggers = check_stops_and_targets(engine)
    entry_triggers = check_entry_triggers(engine)

    for t in stop_triggers:
        log.warning(f"⚡ {t['type'].upper()}: {t['symbol']} ({t['profile']}) "
                     f"price={t['price']} level={t['level']}")

    for t in entry_triggers:
        log.info(f"⚡ {t['type'].upper()}: {t['symbol']} {t['signal']} "
                  f"price={t['price']} broke {t['level_name']}={t['level']}")

    return {
        "stop_triggers": stop_triggers,
        "entry_triggers": entry_triggers,
        "checked_at": datetime.utcnow().isoformat(),
    }
