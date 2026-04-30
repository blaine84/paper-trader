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
from utils.trade_events import log_trade_event

log = logging.getLogger(__name__)


def get_batch_quotes(symbols: list[str]) -> dict:
    """Get current prices for multiple symbols via yfinance — batch fetch."""
    quotes = {}
    try:
        import yfinance as yf
        tickers = yf.Tickers(" ".join(symbols))
        for sym in symbols:
            try:
                price = tickers.tickers[sym].fast_info.get("lastPrice")
                if price:
                    quotes[sym] = round(float(price), 2)
            except Exception:
                pass
    except Exception:
        # Fallback to individual fetches
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

        # Determine side from the trade's direction field
        side = "short" if trade.direction == "SHORT" else "long"

        # Stop loss check — require price to exceed stop by a small buffer (0.1%)
        # to avoid false triggers from bid-ask spread noise
        if trade.stop_price:
            # Sanity: for long, stop should be below entry; for short, above
            stop_valid = True
            if side == "long" and trade.stop_price >= trade.entry_price:
                stop_valid = False
                log.warning(f"⚠️ {trade.symbol} ({trade.profile}): LONG stop {trade.stop_price} is above entry {trade.entry_price} — skipping stop check")
            elif side == "short" and trade.stop_price <= trade.entry_price:
                stop_valid = False
                log.warning(f"⚠️ {trade.symbol} ({trade.profile}): SHORT stop {trade.stop_price} is below entry {trade.entry_price} — skipping stop check")

            if stop_valid:
                buffer = trade.stop_price * 0.001
                if side == "long":
                    hit = price <= (trade.stop_price - buffer)
                else:
                    hit = price >= (trade.stop_price + buffer)
                if hit:
                    trigger = {
                        "type": "stop_loss",
                        "symbol": trade.symbol,
                        "profile": trade.profile,
                        "price": price,
                        "level": trade.stop_price,
                        "side": side,
                        "trade_id": trade.id,
                    }
                    triggers.append(trigger)
                    log_trade_event(
                        db, "stop_triggered", trade_id=trade.id, agent="price_monitor",
                        symbol=trade.symbol, profile=trade.profile, price=price,
                        message=f"Stop triggered at {price} vs level {trade.stop_price}",
                        payload=trigger,
                    )

        # Target check — verify target is on the correct side before triggering
        if trade.target_price:
            # Sanity: for long, target should be above entry; for short, below
            target_valid = True
            if side == "long" and trade.target_price <= trade.entry_price:
                target_valid = False
                log.warning(f"⚠️ {trade.symbol} ({trade.profile}): LONG target {trade.target_price} is below entry {trade.entry_price} — skipping target check")
            elif side == "short" and trade.target_price >= trade.entry_price:
                target_valid = False
                log.warning(f"⚠️ {trade.symbol} ({trade.profile}): SHORT target {trade.target_price} is above entry {trade.entry_price} — skipping target check")

            if target_valid:
                hit = (side == "long" and price >= trade.target_price) or \
                      (side == "short" and price <= trade.target_price)
                if hit:
                    trigger = {
                        "type": "target_hit",
                        "symbol": trade.symbol,
                        "profile": trade.profile,
                        "price": price,
                        "level": trade.target_price,
                        "side": side,
                        "trade_id": trade.id,
                    }
                    triggers.append(trigger)
                    log_trade_event(
                        db, "target_triggered", trade_id=trade.id, agent="price_monitor",
                        symbol=trade.symbol, profile=trade.profile, price=price,
                        message=f"Target triggered at {price} vs level {trade.target_price}",
                        payload=trigger,
                    )

        # Thesis invalidator evaluation — check structured invalidation conditions
        if getattr(trade, "invalidators", None):
            breached = evaluate_invalidators(trade, price)
            for inv in breached:
                now = datetime.utcnow()
                trigger = {
                    "type": "thesis_invalidation",
                    "symbol": trade.symbol,
                    "profile": trade.profile,
                    "trade_id": trade.id,
                    "price": price,
                    "invalidator": inv,
                    "timestamp": now.isoformat() + "Z",
                }
                triggers.append(trigger)
                log_trade_event(
                    db, "thesis_invalidated", trade_id=trade.id, agent="price_monitor",
                    symbol=trade.symbol, profile=trade.profile, price=price,
                    message=f"Thesis invalidator breached: {inv.get('type')}@{inv.get('reference')}",
                    payload=trigger,
                    timestamp=now,
                )
                # Store in AgentMemory for PM to read during Reversal/Close Review
                db.add(AgentMemory(
                    agent="price_monitor",
                    symbol=trade.symbol,
                    key="thesis_invalidation",
                    value=json.dumps(trigger),
                    timestamp=now,
                ))
                log.warning(
                    f"⚡ THESIS INVALIDATION: {trade.symbol} ({trade.profile}) "
                    f"trade_id={trade.id} price={price} "
                    f"invalidator={inv.get('type')}@{inv.get('reference')}"
                )
            if breached:
                db.commit()

    if triggers:
        db.commit()

    # Run profit management on all open trades
    from agents.profit_manager import run as run_profit
    trades_and_prices = []
    for trade in open_trades:
        price = quotes.get(trade.symbol)
        if price:
            trades_and_prices.append((trade, price))
    if trades_and_prices:
        run_profit(engine, trades_and_prices)

    db.close()
    return triggers


def _resolve_reference(reference: str, current_price: float, candle_data: dict | None = None) -> float | None:
    """
    Resolve an invalidator reference to a numeric price value.

    Handles:
    - Literal numeric strings (e.g., "162.50") → float
    - Named indicators (e.g., "VWAP") → looked up from candle_data indicators
    Returns None if the reference cannot be resolved.
    """
    # Try literal numeric first
    try:
        return float(reference)
    except (ValueError, TypeError):
        pass

    # Try named indicator from candle_data
    if candle_data and isinstance(candle_data, dict):
        # Normalize lookup: try lowercase key match
        ref_lower = reference.lower()
        for key, val in candle_data.items():
            if key.lower() == ref_lower:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass

    return None


def _check_candle_confirmation(invalidator: dict, reference_value: float, candle_data: dict | None) -> bool:
    """
    Check if a 5m_close confirmation is met using candle close data.

    For `price_below_level`: candle close must be below reference_value
    for `lookback_bars` consecutive bars.
    For `price_above_level`: candle close must be above reference_value
    for `lookback_bars` consecutive bars.

    candle_data should contain a "closes" key with a list of recent close prices
    (most recent last).
    """
    if not candle_data or not isinstance(candle_data, dict):
        return False

    closes = candle_data.get("closes")
    if not closes or not isinstance(closes, list):
        return False

    lookback = invalidator.get("lookback_bars", 1)
    if lookback < 1:
        lookback = 1

    inv_type = invalidator.get("type", "")

    # Need at least lookback_bars worth of close data
    if len(closes) < lookback:
        return False

    # Check the most recent `lookback` candle closes
    recent_closes = closes[-lookback:]

    if inv_type == "price_below_level":
        return all(c < reference_value for c in recent_closes)
    elif inv_type == "price_above_level":
        return all(c > reference_value for c in recent_closes)

    return False


def evaluate_invalidators(trade, current_price: float, candle_data: dict | None = None) -> list[dict]:
    """
    Evaluate structured invalidator conditions against current market data.

    Args:
        trade: Trade object with an `invalidators` JSON text column.
        current_price: The latest price for the trade's symbol.
        candle_data: Optional dict with indicator values (e.g., {"vwap": 163.20})
                     and/or candle closes (e.g., {"closes": [162.1, 161.8]})
                     for 5m_close confirmation.

    Returns:
        List of breached invalidator dicts. Empty list if none breached or on error.
    """
    # Parse invalidators JSON
    raw = getattr(trade, "invalidators", None)
    if not raw or (isinstance(raw, str) and not raw.strip()):
        return []

    try:
        invalidators = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        log.error(f"Malformed invalidators JSON for trade {getattr(trade, 'id', '?')}: {e}")
        return []

    if not isinstance(invalidators, list):
        log.error(f"Invalidators is not a list for trade {getattr(trade, 'id', '?')}")
        return []

    breached = []

    for inv in invalidators:
        if not isinstance(inv, dict):
            log.warning(f"Skipping non-dict invalidator for trade {getattr(trade, 'id', '?')}: {inv}")
            continue

        inv_type = inv.get("type", "")

        # Skip structure_break — handled by LLM in Reversal Review
        if inv_type == "structure_break":
            continue

        if inv_type not in ("price_below_level", "price_above_level"):
            log.warning(f"Unknown invalidator type '{inv_type}' for trade {getattr(trade, 'id', '?')}, skipping")
            continue

        reference = inv.get("reference")
        if reference is None:
            log.warning(f"Invalidator missing reference for trade {getattr(trade, 'id', '?')}, skipping")
            continue

        # Resolve reference to a numeric value
        ref_value = _resolve_reference(reference, current_price, candle_data)
        if ref_value is None:
            log.warning(
                f"Cannot resolve reference '{reference}' for trade {getattr(trade, 'id', '?')}, skipping"
            )
            continue

        confirmation = inv.get("confirmation", "tick")

        if confirmation == "tick":
            # Immediate tick-level breach check
            if inv_type == "price_below_level" and current_price < ref_value:
                breached.append(inv)
            elif inv_type == "price_above_level" and current_price > ref_value:
                breached.append(inv)

        elif confirmation == "5m_close":
            # Candle close confirmation required
            if candle_data is None:
                # No candle data available this cycle — skip, don't treat as breached
                continue
            if _check_candle_confirmation(inv, ref_value, candle_data):
                breached.append(inv)

        else:
            log.warning(
                f"Unknown confirmation type '{confirmation}' for trade {getattr(trade, 'id', '?')}, skipping"
            )

    return breached


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


def get_price_history(symbol: str) -> list[tuple]:
    """Return the price history for a symbol as a list of (timestamp, price) tuples.

    Returns an empty list if no history exists for the symbol.
    This is the public API — callers should use this instead of accessing
    _price_history directly.
    """
    return list(_price_history.get(symbol, []))


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
        if t["type"] == "thesis_invalidation":
            # Already logged inside check_stops_and_targets
            continue
        log.warning(f"⚡ {t['type'].upper()}: {t['symbol']} ({t['profile']}) "
                     f"price={t['price']} level={t['level']}")

    for t in entry_triggers:
        log.info(f"⚡ {t['type'].upper()}: {t['symbol']} {t['signal']} "
                  f"price={t['price']} broke {t['level_name']}={t['level']}")

    for a in momentum_alerts:
        if a["type"] == "rapid_move":
            log.warning(f"📊 RAPID MOVE: {a['symbol']} {a['direction']} {a['change_pct']}% in {a['window_minutes']}min (${a['price']})")
            # Trigger narrator flash update for significant rapid moves
            try:
                import agents.narrator as narrator
                narrator.run(engine, "flash_update", event_context={
                    "trigger": "atr_spike",
                    "symbol": a["symbol"],
                    "details": f"{a['symbol']} moved {a['change_pct']}% {a['direction']} in {a['window_minutes']}min",
                    "price": a["price"],
                })
            except Exception:
                pass  # never block price monitoring
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
