"""
Bookkeeper Agent
Tracks P&L, generates daily summaries, monitors stop losses.
"""

import os
import json
from datetime import datetime, date
from utils.finnhub_client import FinnhubClient
from db.schema import Position, Balance, Trade, DailyLog, AgentMemory, get_session
from models.pm_profiles import PM_PROFILES, ACTIVE_PROFILES
from rich.console import Console
from rich.table import Table

console = Console()


def check_stop_losses(engine, profile_id: str = None) -> list:
    """
    Check if any open positions have hit their stop loss.
    Returns list of symbols that need to be closed.
    """
    fh = FinnhubClient()
    db = get_session(engine)
    to_close = []

    query = db.query(Trade).filter_by(status="open")
    if profile_id:
        query = query.filter_by(profile=profile_id)
    open_trades = query.all()
    for trade in open_trades:
        quote = fh.get_quote(trade.symbol)
        price = quote["price"]
        # Stop loss is stored in the analyst signal memory
        sig = (
            db.query(AgentMemory)
            .filter_by(agent="analyst", symbol=trade.symbol, key="signal")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if sig:
            signal = json.loads(sig.value)
            stop = signal.get("stop_loss")
            if stop:
                # Long: stop triggers when price falls below stop
                # Short: stop triggers when price rises above stop
                pos = db.query(Position).filter_by(
                    symbol=trade.symbol, profile=trade.profile
                ).first()
                side = pos.side if pos else "long"
                triggered = (side == "long" and price <= stop) or \
                            (side == "short" and price >= stop)
                if triggered:
                    to_close.append({
                        "symbol": trade.symbol,
                        "price": price,
                        "stop_loss": stop,
                        "trade_id": trade.id,
                        "profile": trade.profile,
                    })

    db.close()
    return to_close


def get_portfolio_summary(engine) -> dict:
    """Current portfolio snapshot with P&L."""
    fh = FinnhubClient()
    db = get_session(engine)

    positions = db.query(Position).all()
    pos_rows = []
    total_value = 0

    for p in positions:
        quote = fh.get_quote(p.symbol)
        price = quote["price"]
        value = p.quantity * price
        pnl = (price - p.avg_cost) * p.quantity
        pnl_pct = (price - p.avg_cost) / p.avg_cost * 100
        total_value += value
        pos_rows.append({
            "symbol": p.symbol,
            "qty": p.quantity,
            "avg_cost": p.avg_cost,
            "price": price,
            "value": round(value, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
        })

    bal = db.query(Balance).order_by(Balance.timestamp.desc()).first()
    cash = bal.cash if bal else float(os.getenv("STARTING_BALANCE", 100000))
    total_equity = cash + total_value

    # Closed trades stats
    closed = db.query(Trade).filter_by(status="closed").all()
    wins = [t for t in closed if (t.pnl or 0) > 0]
    losses = [t for t in closed if (t.pnl or 0) <= 0]
    total_pnl = sum(t.pnl or 0 for t in closed)

    db.close()

    return {
        "cash": round(cash, 2),
        "positions": pos_rows,
        "total_equity": round(total_equity, 2),
        "total_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "total_realized_pnl": round(total_pnl, 2),
    }


def end_of_day(engine) -> dict:
    """Save daily summary log."""
    db = get_session(engine)
    today = date.today().isoformat()

    summary = get_portfolio_summary(engine)
    starting = float(os.getenv("STARTING_BALANCE", 100000))

    # Count today's trades
    today_trades = (
        db.query(Trade)
        .filter(Trade.status == "closed")
        .filter(Trade.exit_time >= datetime.utcnow().replace(hour=0, minute=0))
        .all()
    )
    today_pnl = sum(t.pnl or 0 for t in today_trades)
    today_pnl_pct = today_pnl / starting * 100 if starting else 0
    wins = len([t for t in today_trades if (t.pnl or 0) > 0])
    losses = len([t for t in today_trades if (t.pnl or 0) <= 0])

    log = DailyLog(
        date=today,
        starting_equity=starting,
        ending_equity=summary["total_equity"],
        trades_taken=len(today_trades),
        winning_trades=wins,
        losing_trades=losses,
        daily_pnl=round(today_pnl, 2),
        daily_pnl_pct=round(today_pnl_pct, 2),
    )
    db.add(log)
    db.commit()
    db.close()

    return {
        "date": today,
        "equity": summary["total_equity"],
        "daily_pnl": round(today_pnl, 2),
        "trades": len(today_trades),
        "wins": wins,
        "losses": losses,
    }


def print_dashboard(engine):
    """Print a rich terminal dashboard showing all PM profiles side by side."""
    from agents.portfolio_manager import get_portfolio_for_profile
    fh = FinnhubClient()
    db = get_session(engine)

    console.print("\n[bold cyan]📊 Paper Trader Dashboard[/bold cyan]")
    console.print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n")

    # Profile summary table
    summary_table = Table(title="Portfolio Comparison")
    summary_table.add_column("Profile")
    summary_table.add_column("Cash")
    summary_table.add_column("Equity")
    summary_table.add_column("Daily P&L")
    summary_table.add_column("Positions")

    for profile_id in ACTIVE_PROFILES:
        profile = PM_PROFILES[profile_id]
        port = get_portfolio_for_profile(db, fh, profile_id)
        pnl_color = "green" if port["daily_pnl"] >= 0 else "red"
        summary_table.add_row(
            f"{profile['emoji']} {profile['name']}",
            f"${port['cash']:,.2f}",
            f"${port['total_equity']:,.2f}",
            f"[{pnl_color}]${port['daily_pnl']:+,.2f} ({port['daily_pnl_pct']:+.2f}%)[/{pnl_color}]",
            str(port["position_count"]),
        )
    console.print(summary_table)

    # Per-profile open positions
    for profile_id in ACTIVE_PROFILES:
        profile = PM_PROFILES[profile_id]
        positions = db.query(Position).filter_by(profile=profile_id).all()
        if not positions:
            continue

        pos_table = Table(title=f"{profile['emoji']} {profile['name']} — Open Positions")
        pos_table.add_column("Symbol")
        pos_table.add_column("Qty")
        pos_table.add_column("Avg Cost")
        pos_table.add_column("Price")
        pos_table.add_column("Value")
        pos_table.add_column("P&L")

        for p in positions:
            try:
                price = fh.get_quote(p.symbol)["price"]
            except Exception:
                price = p.avg_cost
            value = p.quantity * price
            if p.side == "short":
                pnl = (p.avg_cost - price) * p.quantity
            else:
                pnl = (price - p.avg_cost) * p.quantity
            pnl_pct = pnl / (p.avg_cost * p.quantity) * 100
            color = "green" if pnl >= 0 else "red"
            side_label = f"[red]SHORT[/red]" if p.side == "short" else "LONG"
            pos_table.add_row(
                p.symbol,
                f"{side_label} {p.quantity}",
                f"${p.avg_cost:.2f}",
                f"${price:.2f}",
                f"${value:,.2f}",
                f"[{color}]${pnl:+,.2f} ({pnl_pct:+.1f}%)[/{color}]",
            )
        console.print(pos_table)

    db.close()
