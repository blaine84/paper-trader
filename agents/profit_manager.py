"""
Profit Manager
Handles partial profit taking and trailing stops.

Rules:
  At +1R: take 25-50% off, move stop to breakeven
  At +2R: trail stop to +1R or EMA/VWAP
  At +3R: trail stop to +2R, tighten further

Runs as part of the price monitor every 60 seconds.
"""

import json
import logging
from datetime import datetime
from db.schema import Trade, Position, Balance, AgentMemory, get_session
from utils.trade_events import log_trade_event

log = logging.getLogger(__name__)

# Profit taking rules per profile
PROFIT_RULES = {
    "conservative": {
        "partial_at_1r": 0.50,   # take 50% off at +1R
        "partial_at_2r": 0.25,   # take another 25% at +2R (of remaining)
        "trail_at_2r": True,
    },
    "moderate": {
        "partial_at_1r": 0.33,   # take 33% off at +1R
        "partial_at_2r": 0.25,
        "trail_at_2r": True,
    },
    "aggressive": {
        "partial_at_1r": 0.25,   # take 25% off at +1R
        "partial_at_2r": 0.0,    # let it ride
        "trail_at_2r": True,
    },
}

# Track what we've already done per trade to avoid repeating
# {trade_id: {"partial_1r": True, "breakeven": True, "partial_2r": True, "trailing": True}}
_profit_actions = {}


def _calc_r(trade, current_price) -> float:
    """Calculate current R multiple achieved."""
    if not trade.stop_price or not trade.entry_price:
        return 0.0
    risk = abs(trade.entry_price - trade.stop_price)
    if risk == 0:
        return 0.0
    if trade.direction == "LONG":
        move = current_price - trade.entry_price
    else:
        move = trade.entry_price - current_price
    return round(move / risk, 2)


def _partial_close(engine, trade, quantity_pct, price, reason):
    """Close a percentage of the position."""
    db = get_session(engine)
    pos = db.query(Position).filter_by(
        symbol=trade.symbol, profile=trade.profile
    ).first()
    if not pos or pos.quantity <= 0:
        db.close()
        return

    close_qty = max(1, int(pos.quantity * quantity_pct))
    if close_qty >= pos.quantity:
        close_qty = pos.quantity

    # Calculate P&L for the partial
    if trade.direction == "LONG":
        pnl = (price - trade.entry_price) * close_qty
    else:
        pnl = (trade.entry_price - price) * close_qty

    # Reduce position
    pos.quantity -= close_qty
    if pos.quantity <= 0:
        db.delete(pos)

    # Return cash
    bal = db.query(Balance).filter_by(profile=trade.profile).order_by(Balance.timestamp.desc()).first()
    if bal:
        if trade.direction == "LONG":
            cash_back = close_qty * price
        else:
            cash_back = close_qty * trade.entry_price + pnl
        db.add(Balance(cash=bal.cash + cash_back, profile=trade.profile))

    log_trade_event(
        db, "partial_profit", trade_id=trade.id, agent=f"profit_manager",
        symbol=trade.symbol, profile=trade.profile, price=price,
        message=reason,
        payload={
            "closed_qty": close_qty,
            "remaining_qty": max(0, pos.quantity) if pos else 0,
            "pnl": round(pnl, 2),
            "quantity_pct": quantity_pct,
        },
    )

    # Log the partial as an agent memory note
    db.add(AgentMemory(
        agent=f"pm_{trade.profile}",
        symbol=trade.symbol,
        key="partial_profit",
        value=json.dumps({
            "trade_id": trade.id,
            "closed_qty": close_qty,
            "remaining_qty": max(0, pos.quantity) if pos else 0,
            "price": price,
            "pnl": round(pnl, 2),
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        }),
    ))

    db.commit()
    db.close()
    log.info(f"💰 PARTIAL PROFIT: {trade.symbol} ({trade.profile}) closed {close_qty} @ ${price:.2f} "
             f"(P&L: ${pnl:+.2f}) — {reason}")


def _move_stop(engine, trade_id, new_stop, reason):
    """Update the stop price on a trade."""
    db = get_session(engine)
    trade = db.query(Trade).filter_by(id=trade_id).first()
    if trade:
        old_stop = trade.stop_price
        trade.stop_price = new_stop
        log_trade_event(
            db, "stop_set", trade_id=trade.id, agent="profit_manager",
            symbol=trade.symbol, profile=trade.profile, price=new_stop,
            message=reason,
            payload={"old_stop": old_stop, "new_stop": new_stop},
        )
        db.commit()
        log.info(f"🔒 STOP MOVED: {trade.symbol} ({trade.profile}) "
                 f"${old_stop} → ${new_stop:.2f} — {reason}")
    db.close()


def check_profit_management(engine, trade, current_price) -> list[dict]:
    """
    Check if a trade qualifies for partial profit taking or stop trailing.
    Returns list of actions taken.
    """
    if not trade.stop_price or not trade.entry_price:
        return []

    r_achieved = _calc_r(trade, current_price)
    if r_achieved <= 0:
        return []

    actions_taken = []
    trade_actions = _profit_actions.setdefault(trade.id, {})
    rules = PROFIT_RULES.get(trade.profile, PROFIT_RULES["moderate"])
    risk = abs(trade.entry_price - trade.stop_price)

    # ── +1R: Partial profit + move stop to breakeven ──
    if r_achieved >= 1.0 and not trade_actions.get("partial_1r"):
        pct = rules["partial_at_1r"]
        if pct > 0:
            _partial_close(engine, trade, pct, current_price,
                           f"+{r_achieved:.1f}R — taking {pct:.0%} off")
            trade_actions["partial_1r"] = True
            actions_taken.append({"action": "partial_1r", "r": r_achieved, "pct": pct})

        # Move stop to breakeven
        if not trade_actions.get("breakeven"):
            _move_stop(engine, trade.id, trade.entry_price, f"+{r_achieved:.1f}R — stop to breakeven")
            trade_actions["breakeven"] = True
            actions_taken.append({"action": "breakeven", "r": r_achieved})

    # ── +2R: Second partial + trail stop to +1R ──
    if r_achieved >= 2.0 and not trade_actions.get("partial_2r"):
        pct = rules["partial_at_2r"]
        if pct > 0:
            _partial_close(engine, trade, pct, current_price,
                           f"+{r_achieved:.1f}R — taking {pct:.0%} of remaining")
            trade_actions["partial_2r"] = True
            actions_taken.append({"action": "partial_2r", "r": r_achieved, "pct": pct})

        # Trail stop to +1R level
        if rules["trail_at_2r"] and not trade_actions.get("trail_2r"):
            if trade.direction == "LONG":
                new_stop = trade.entry_price + risk  # +1R above entry
            else:
                new_stop = trade.entry_price - risk  # +1R below entry (for short, stop moves down)
            _move_stop(engine, trade.id, round(new_stop, 2), f"+{r_achieved:.1f}R — trailing stop to +1R")
            trade_actions["trail_2r"] = True
            actions_taken.append({"action": "trail_to_1r", "r": r_achieved})

    # ── +3R: Tighten trail to +2R ──
    if r_achieved >= 3.0 and not trade_actions.get("trail_3r"):
        if trade.direction == "LONG":
            new_stop = trade.entry_price + (risk * 2)
        else:
            new_stop = trade.entry_price - (risk * 2)
        _move_stop(engine, trade.id, round(new_stop, 2), f"+{r_achieved:.1f}R — trailing stop to +2R")
        trade_actions["trail_3r"] = True
        actions_taken.append({"action": "trail_to_2r", "r": r_achieved})

    return actions_taken


def run(engine, trades_and_prices: list[tuple]) -> dict:
    """
    Check all open trades for profit management.
    trades_and_prices: list of (Trade, current_price) tuples.
    """
    all_actions = []
    for trade, price in trades_and_prices:
        actions = check_profit_management(engine, trade, price)
        if actions:
            all_actions.extend(actions)

    # Clean up tracking for closed trades
    from db.schema import get_session
    db = get_session(engine)
    open_ids = {t.id for t, _ in trades_and_prices}
    for tid in list(_profit_actions.keys()):
        if tid not in open_ids:
            del _profit_actions[tid]
    db.close()

    return {"actions": all_actions}
