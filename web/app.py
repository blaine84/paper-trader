"""
Paper Trader — Web Dashboard
Run with: python web/app.py
Access at: http://localhost:5000
"""

import json
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, render_template
from db.schema import init_db, get_session, Position, Balance, Trade, AgentMemory, DailyLog
from models.pm_profiles import PM_PROFILES, ACTIVE_PROFILES
from utils.finnhub_client import FinnhubClient

app = Flask(__name__)
engine = init_db("db/paper_trader.db")


def get_quotes(symbols: list[str]) -> dict:
    fh = FinnhubClient()
    quotes = {}
    for sym in symbols:
        try:
            quotes[sym] = fh.get_quote(sym)
        except Exception:
            quotes[sym] = {"price": 0, "change_pct": 0}
    return quotes


def get_market_open() -> bool:
    try:
        return FinnhubClient().is_market_open()
    except Exception:
        return False


def get_analyst_signals(db, symbols: list[str]) -> dict:
    signals = {}
    for sym in symbols:
        mem = (
            db.query(AgentMemory)
            .filter_by(agent="analyst", symbol=sym, key="signal")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if mem:
            signals[sym] = json.loads(mem.value)
    return signals


def get_researcher_sentiment(db, symbols: list[str]) -> dict:
    sentiment = {}
    for sym in symbols:
        mem = (
            db.query(AgentMemory)
            .filter_by(agent="researcher", symbol=sym, key="sentiment")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if mem:
            sentiment[sym] = json.loads(mem.value)
    return sentiment


def get_scout_picks(db) -> list:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    mem = (
        db.query(AgentMemory)
        .filter_by(agent="scout", key="daily_picks")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    if not mem:
        return []
    data = json.loads(mem.value)
    if data.get("date") != today:
        return []
    return data.get("picks", [])


def get_portfolio_summary(db) -> dict:
    summaries = {}
    for profile_id in ACTIVE_PROFILES:
        positions = db.query(Position).filter_by(profile=profile_id).all()
        bal = (
            db.query(Balance)
            .filter_by(profile=profile_id)
            .order_by(Balance.timestamp.desc())
            .first()
        )
        starting = float(PM_PROFILES[profile_id]["starting_balance"])
        cash = bal.cash if bal else starting
        pos_value = sum(p.quantity * p.avg_cost for p in positions)
        equity = cash + pos_value
        pnl = equity - starting

        summaries[profile_id] = {
            "name": PM_PROFILES[profile_id]["name"],
            "emoji": PM_PROFILES[profile_id]["emoji"],
            "equity": round(equity, 2),
            "cash": round(cash, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / starting * 100, 2),
            "position_count": len(positions),
        }
    return summaries


# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    db = get_session(engine)
    core = [s.strip() for s in os.getenv("WATCHLIST", "SPY,QQQ,IWM,TSLA,NVDA,AMD").split(",")]
    scout_picks = get_scout_picks(db)
    scout_symbols = [p["symbol"] for p in scout_picks]
    all_symbols = core + [s for s in scout_symbols if s not in core]

    signals = get_analyst_signals(db, all_symbols)
    sentiment = get_researcher_sentiment(db, all_symbols)
    portfolio = get_portfolio_summary(db)
    quotes = get_quotes(all_symbols)
    market_open = get_market_open()

    watchlist = []
    for sym in all_symbols:
        q = quotes.get(sym, {})
        sig = signals.get(sym, {})
        sent = sentiment.get(sym, {})
        watchlist.append({
            "symbol": sym,
            "is_scout": sym in scout_symbols,
            "price": q.get("price", 0),
            "change_pct": q.get("change_pct", 0),
            "signal": sig.get("signal", "—"),
            "strength": sig.get("strength", "—"),
            "confidence": sig.get("confidence", "—"),
            "setup_type": sig.get("setup_type", "—"),
            "reasoning": sig.get("reasoning", ""),
            "invalidation": sig.get("invalidation", ""),
            "key_levels": sig.get("key_levels", {}),
            "sentiment": sent.get("sentiment", "—"),
            "catalysts": sent.get("catalysts", []),
            "risks": sent.get("risks", []),
        })

    db.close()
    return jsonify({
        "timestamp": datetime.utcnow().isoformat(),
        "market_open": market_open,
        "watchlist": watchlist,
        "scout_picks": scout_picks,
        "portfolio": portfolio,
    })


@app.route("/api/positions")
def api_positions():
    db = get_session(engine)
    result = []
    for profile_id in ACTIVE_PROFILES:
        positions = db.query(Position).filter_by(profile=profile_id).all()
        for p in positions:
            result.append({
                "profile": profile_id,
                "symbol": p.symbol,
                "side": p.side,
                "quantity": p.quantity,
                "avg_cost": p.avg_cost,
                "opened_at": p.opened_at.isoformat() if p.opened_at else None,
            })
    db.close()
    return jsonify(result)


@app.route("/api/trades")
def api_trades():
    db = get_session(engine)
    cutoff = datetime.utcnow() - timedelta(days=30)
    trades = (
        db.query(Trade)
        .filter(Trade.entry_time >= cutoff)
        .order_by(Trade.entry_time.desc())
        .limit(100)
        .all()
    )
    result = [
        {
            "id": t.id,
            "profile": t.profile,
            "symbol": t.symbol,
            "direction": t.direction,
            "quantity": t.quantity,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "entry_time": t.entry_time.isoformat() if t.entry_time else None,
            "exit_time": t.exit_time.isoformat() if t.exit_time else None,
            "status": t.status,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "review_score": t.review_score,
        }
        for t in trades
    ]
    db.close()
    return jsonify(result)


@app.route("/api/daily")
def api_daily():
    db = get_session(engine)
    logs = db.query(DailyLog).order_by(DailyLog.date.desc()).limit(30).all()
    result = [
        {
            "date": l.date,
            "daily_pnl": l.daily_pnl,
            "daily_pnl_pct": l.daily_pnl_pct,
            "trades": l.trades_taken,
            "wins": l.winning_trades,
            "losses": l.losing_trades,
        }
        for l in logs
    ]
    db.close()
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.getenv("WEB_PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
