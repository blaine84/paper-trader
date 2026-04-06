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
            "setup_reasoning": sig.get("setup_reasoning", ""),
            "reasoning": sig.get("reasoning", ""),
            "invalidation": sig.get("invalidation", ""),
            "key_levels": sig.get("key_levels", {}),
            "sentiment": sent.get("sentiment", "—"),
            "catalysts": sent.get("catalysts", []),
            "risks": sent.get("risks", []),
        })

    # Market regime from quant researcher
    regime = None
    mem = (
        db.query(AgentMemory)
        .filter_by(agent="quant_researcher", key="regime")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    if mem:
        try:
            regime_data = json.loads(mem.value)
            regime = regime_data.get("regime") or regime_data.get("market_regime")
        except Exception:
            regime = mem.value

    db.close()
    return jsonify({
        "timestamp": datetime.utcnow().isoformat(),
        "market_open": market_open,
        "watchlist": watchlist,
        "scout_picks": scout_picks,
        "portfolio": portfolio,
        "regime": regime,
    })


@app.route("/api/positions")
def api_positions():
    db = get_session(engine)
    fh = FinnhubClient()
    result = []
    for profile_id in ACTIVE_PROFILES:
        positions = db.query(Position).filter_by(profile=profile_id).all()
        for p in positions:
            try:
                q = fh.get_quote(p.symbol)
                current_price = q.get("price", p.avg_cost)
            except Exception:
                current_price = p.avg_cost
            if p.side == "long":
                unrealized_pnl = (current_price - p.avg_cost) * p.quantity
            else:
                unrealized_pnl = (p.avg_cost - current_price) * p.quantity
            unrealized_pct = (unrealized_pnl / (p.avg_cost * p.quantity)) * 100 if (p.avg_cost and p.quantity) else 0

            # pull stop/target from the trade record
            stop = target = None
            open_trade = (
                db.query(Trade)
                .filter_by(symbol=p.symbol, profile=profile_id, status="open")
                .order_by(Trade.entry_time.desc())
                .first()
            )
            if open_trade:
                stop = open_trade.stop_price
                target = open_trade.target_price

            result.append({
                "profile": profile_id,
                "symbol": p.symbol,
                "side": p.side,
                "quantity": p.quantity,
                "avg_cost": p.avg_cost,
                "current_price": round(current_price, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "unrealized_pct": round(unrealized_pct, 2),
                "stop": stop,
                "target": target,
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


@app.route("/api/feedback")
def api_feedback():
    db = get_session(engine)
    result = {"selection": None, "execution": {}, "regime": None, "quant": None}

    # Selection feedback (for Scout + Analyst)
    mem = (
        db.query(AgentMemory)
        .filter_by(agent="reviewer", key="selection_feedback")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    if mem:
        try:
            result["selection"] = json.loads(mem.value)
        except Exception:
            result["selection"] = {"feedback": mem.value}

    # Execution feedback per profile
    for profile_id in ACTIVE_PROFILES:
        mem = (
            db.query(AgentMemory)
            .filter_by(agent="reviewer", key="execution_feedback")
            .filter(AgentMemory.symbol == profile_id)
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if mem:
            try:
                result["execution"][profile_id] = json.loads(mem.value)
            except Exception:
                result["execution"][profile_id] = {"feedback": mem.value}

    # Market regime from quant researcher
    mem = (
        db.query(AgentMemory)
        .filter_by(agent="quant_researcher", key="regime")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    if mem:
        try:
            result["regime"] = json.loads(mem.value)
        except Exception:
            result["regime"] = {"regime": mem.value}

    # Weekly stance per profile
    stances = {}
    for profile_id in ACTIVE_PROFILES:
        mem = (
            db.query(AgentMemory)
            .filter_by(agent="weekly_prep", symbol=profile_id, key="stance")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if mem:
            try:
                stances[profile_id] = json.loads(mem.value)
            except Exception:
                pass
    result["stances"] = stances

    db.close()
    return jsonify(result)


@app.route("/api/performance")
def api_performance():
    from models.case import Case
    from sqlalchemy import func
    from sqlalchemy.types import Integer as SAInteger
    db = get_session(engine)

    # Win rates by setup type
    setup_rows = (
        db.query(
            Case.setup_type,
            func.count(Case.id).label("total"),
            func.sum((Case.outcome == "success").cast(SAInteger)).label("wins"),
            func.avg(Case.pnl_pct).label("avg_pnl"),
            func.avg(Case.review_score).label("avg_score"),
        )
        .filter(Case.setup_type != None)
        .group_by(Case.setup_type)
        .all()
    )
    setup_stats = [
        {
            "setup_type": r.setup_type,
            "total": r.total,
            "wins": r.wins or 0,
            "win_rate": round((r.wins or 0) / r.total * 100, 1),
            "avg_pnl": round(r.avg_pnl or 0, 2),
            "avg_score": round(r.avg_score or 0, 1),
        }
        for r in setup_rows
    ]

    # Per-profile stats
    profile_stats = {}
    for profile_id in ACTIVE_PROFILES:
        trades = db.query(Trade).filter_by(profile=profile_id, status="closed").all()
        if not trades:
            profile_stats[profile_id] = {"total": 0, "wins": 0, "win_rate": 0, "avg_pnl": 0, "total_pnl": 0}
            continue
        wins = sum(1 for t in trades if (t.pnl or 0) > 0)
        total_pnl = sum(t.pnl or 0 for t in trades)
        avg_pnl = total_pnl / len(trades)
        profile_stats[profile_id] = {
            "total": len(trades),
            "wins": wins,
            "win_rate": round(wins / len(trades) * 100, 1),
            "avg_pnl": round(avg_pnl, 2),
            "total_pnl": round(total_pnl, 2),
        }

    # Recent cases for case library preview
    cases = (
        db.query(Case)
        .order_by(Case.created_at.desc())
        .limit(20)
        .all()
    )
    case_list = [
        {
            "date": c.date,
            "symbol": c.symbol,
            "setup_type": c.setup_type,
            "outcome": c.outcome,
            "pnl_pct": c.pnl_pct,
            "lesson": c.lesson,
            "selection_score": c.selection_score,
            "execution_score": c.execution_score,
            "profile": c.profile,
        }
        for c in cases
    ]

    db.close()
    return jsonify({
        "setup_stats": setup_stats,
        "profile_stats": profile_stats,
        "cases": case_list,
    })


@app.route("/api/decisions")
def api_decisions():
    db = get_session(engine)
    cutoff = datetime.utcnow() - timedelta(days=7)
    result = []

    for profile_id in ACTIVE_PROFILES:
        notes = (
            db.query(AgentMemory)
            .filter_by(agent=f"pm_{profile_id}", key="notes")
            .filter(AgentMemory.timestamp >= cutoff)
            .order_by(AgentMemory.timestamp.desc())
            .all()
        )
        for n in notes:
            result.append({
                "profile": profile_id,
                "type": "notes",
                "timestamp": n.timestamp.isoformat() if n.timestamp else None,
                "content": n.value,
            })

    trades = (
        db.query(Trade)
        .filter(Trade.entry_time >= cutoff)
        .order_by(Trade.entry_time.desc())
        .all()
    )
    for t in trades:
        result.append({
            "profile": t.profile,
            "type": "trade",
            "timestamp": t.entry_time.isoformat() if t.entry_time else None,
            "content": f"{t.direction} {t.quantity} {t.symbol} @ ${t.entry_price:.2f}"
                       + (f" | stop: ${t.stop_price:.2f}" if t.stop_price else "")
                       + (f" | target: ${t.target_price:.2f}" if t.target_price else "")
                       + (f" | {t.reason_entry}" if t.reason_entry else ""),
            "status": t.status,
            "pnl": t.pnl,
        })

    # Sort all by timestamp descending
    result.sort(key=lambda x: x.get("timestamp") or "", reverse=True)

    db.close()
    return jsonify(result)


@app.route("/api/strategies")
def api_strategies():
    from models.strategies import STRATEGIES, SETUP_TYPE_MAP
    from utils.strategy_store import get_all_strategies
    result = {}
    # Map setup_type names to strategy descriptions
    for setup_type, strategy_key in SETUP_TYPE_MAP.items():
        strat = STRATEGIES.get(strategy_key, {})
        result[setup_type] = {
            "name": strat.get("name", setup_type),
            "description": strat.get("description", ""),
            "timeframe": strat.get("timeframe", ""),
            "bias": strat.get("bias", ""),
            "win_rate": strat.get("win_rate_documented"),
            "source": "hardcoded",
        }
    # Add direct strategy keys
    for key, strat in STRATEGIES.items():
        if key not in result:
            result[key] = {
                "name": strat.get("name", key),
                "description": strat.get("description", ""),
                "timeframe": strat.get("timeframe", ""),
                "bias": strat.get("bias", ""),
                "win_rate": strat.get("win_rate_documented"),
                "source": "hardcoded",
            }
    # Add dynamic strategies
    all_strats = get_all_strategies(engine)
    for key, strat in all_strats.items():
        if strat.get("source") == "dynamic":
            result[key] = {
                "name": strat.get("name", key),
                "description": strat.get("description", ""),
                "timeframe": strat.get("timeframe", ""),
                "bias": strat.get("bias", ""),
                "win_rate": strat.get("win_rate_documented"),
                "source": "dynamic",
                "total_trades": strat.get("total_trades", 0),
                "status": strat.get("status"),
            }
    return jsonify(result)


@app.route("/api/company/<symbol>")
def api_company(symbol):

    try:
        fh = FinnhubClient()
        fh._rate_limit()
        profile = fh.client.company_profile2(symbol=symbol)
        return jsonify({
            "name": profile.get("name"),
            "industry": profile.get("finnhubIndustry"),
            "market_cap": profile.get("marketCapitalization"),
            "employees": profile.get("employeeTotal"),
            "country": profile.get("country"),
            "exchange": profile.get("exchange"),
            "ipo": profile.get("ipo"),
            "logo": profile.get("logo"),
            "weburl": profile.get("weburl"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    port = int(os.getenv("WEB_PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
