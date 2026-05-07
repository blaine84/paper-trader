"""
Paper Trader — Live Display
Real-time terminal display of prices, signals, and analysis summaries.

Usage:
  python display.py                  # refresh every 30s, uses .env watchlist
  python display.py --interval 60    # refresh every 60s
  python display.py --once           # print once and exit
  python display.py --symbols SPY,TSLA,NVDA  # override watchlist
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.columns import Columns

from db.schema import init_db, get_session, Position, Balance, Trade
from utils.finnhub_client import FinnhubClient
from utils.catalyst_freshness import compute_catalyst_freshness, get_breaking_news_for_symbols, get_market_day_start, ET
from utils.news_trade_governance import (
    NewsGovernanceClassifier, NewsGovernancePolicy, NEWS_GOVERNANCE,
)
from agents.quant_researcher import build_strategy_context
from models.pm_profiles import PM_PROFILES, ACTIVE_PROFILES

console = Console()
engine = init_db("db/paper_trader.db")


def get_analyst_signals(db, symbols: list[str]) -> dict:
    from db.schema import AgentMemory
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
    from db.schema import AgentMemory
    cutoff = datetime.utcnow() - timedelta(hours=36)
    sentiment = {}
    for sym in symbols:
        mem = (
            db.query(AgentMemory)
            .filter_by(agent="researcher", symbol=sym, key="sentiment")
            .filter(AgentMemory.timestamp >= cutoff)
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if mem:
            sentiment[sym] = json.loads(mem.value)
    return sentiment


def get_scout_picks(db) -> list[str]:
    from db.schema import AgentMemory
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
    return [p["symbol"] for p in data.get("picks", [])]


def get_portfolio_summary(db, fh: FinnhubClient) -> dict:
    """Quick equity snapshot across all profiles."""
    summaries = {}
    for profile_id in ACTIVE_PROFILES:
        positions = db.query(Position).filter_by(profile=profile_id).all()
        total_pos_value = 0.0
        for p in positions:
            try:
                price = fh.get_quote(p.symbol)["price"]
            except Exception:
                price = p.avg_cost
            total_pos_value += p.quantity * price

        bal = (
            db.query(Balance)
            .filter_by(profile=profile_id)
            .order_by(Balance.timestamp.desc())
            .first()
        )
        starting = PM_PROFILES[profile_id]["starting_balance"]
        cash = bal.cash if bal else float(starting)
        equity = cash + total_pos_value
        pnl = equity - starting
        pnl_pct = pnl / starting * 100

        summaries[profile_id] = {
            "equity": equity,
            "cash": cash,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "positions": len(positions),
        }
    return summaries


# ─── PANEL BUILDERS ───────────────────────────────────────────────────────────

def build_header(market_open: bool) -> Panel:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "[bold green]● MARKET OPEN[/bold green]" if market_open else "[bold red]● MARKET CLOSED[/bold red]"
    return Panel(
        f"[bold white]📈 Paper Trader[/bold white]   {status}   [dim]{now}[/dim]",
        box=box.HORIZONTALS,
        style="bold blue",
    )


def build_prices_table(symbols: list[str], fh: FinnhubClient,
                        signals: dict, scout_picks: list[str]) -> Panel:
    table = Table(
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        show_edge=False,
        expand=True,
    )
    table.add_column("Symbol", style="bold", width=8)
    table.add_column("Price", justify="right", width=10)
    table.add_column("Change", justify="right", width=10)
    table.add_column("Signal", justify="center", width=8)
    table.add_column("Strength", justify="center", width=10)
    table.add_column("Setup", width=20)
    table.add_column("Confidence", justify="center", width=10)
    table.add_column("Source", justify="center", width=8)

    for sym in symbols:
        try:
            q = fh.get_quote(sym)
            price = q.get("price", 0)
            chg_pct = q.get("change_pct", 0)
        except Exception:
            price, chg_pct = 0.0, 0.0

        price_str = f"${price:,.2f}"
        chg_str = f"{chg_pct:+.2f}%"
        chg_color = "green" if chg_pct >= 0 else "red"

        sig = signals.get(sym, {})
        signal = sig.get("signal", "—")
        strength = sig.get("strength", "—")
        setup = sig.get("setup_type", "—") or "—"
        confidence = sig.get("confidence", "—") or "—"

        signal_color = {"LONG": "green", "SHORT": "red", "HOLD": "yellow"}.get(signal, "dim")
        strength_color = {"strong": "green", "moderate": "yellow", "weak": "dim"}.get(strength, "dim")
        conf_color = {"high": "green", "medium": "yellow", "low": "dim"}.get(confidence, "dim")

        source = "[cyan]SCOUT[/cyan]" if sym in scout_picks else "[dim]CORE[/dim]"

        table.add_row(
            sym,
            price_str,
            f"[{chg_color}]{chg_str}[/{chg_color}]",
            f"[{signal_color}]{signal}[/{signal_color}]",
            f"[{strength_color}]{strength}[/{strength_color}]",
            setup[:20],
            f"[{conf_color}]{confidence}[/{conf_color}]",
            source,
        )

    return Panel(table, title="[bold cyan]📊 Watchlist[/bold cyan]", box=box.ROUNDED)


def build_analysis_table(symbols: list[str], signals: dict, sentiment: dict,
                         freshness_data: dict = None, breaking_news_data: dict = None) -> Panel:
    if freshness_data is None:
        freshness_data = {}
    if breaking_news_data is None:
        breaking_news_data = {}

    table = Table(
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        show_edge=False,
        expand=True,
    )
    table.add_column("Symbol", style="bold", width=8)
    table.add_column("Fresh", justify="center", width=7)
    table.add_column("Sentiment", justify="center", width=10)
    table.add_column("Catalysts", width=30)
    table.add_column("Invalidation", width=35)
    table.add_column("Key Levels", width=30)

    for sym in symbols:
        sig = signals.get(sym, {})
        sent = sentiment.get(sym, {})

        # Freshness indicator
        freshness = freshness_data.get(sym, {})
        state = freshness.get("freshness_state", "stale")
        state_color = {"fresh": "green", "aging": "yellow", "stale": "red"}.get(state, "red")
        fresh_str = f"[{state_color}]{state.upper()}[/{state_color}]"

        # Sentiment
        s = sent.get("sentiment", "—")
        s_color = {"bullish": "green", "bearish": "red", "neutral": "yellow"}.get(s, "dim")

        # Catalysts (truncated)
        catalysts = sent.get("catalysts", [])
        cat_str = (" · ".join(catalysts))[:30] if catalysts else "—"

        # Append breaking news headline if available
        alerts = breaking_news_data.get(sym, [])
        if alerts:
            headline = alerts[0].get("headline", "")[:40]
            if headline:
                cat_str += f" | 📰 {headline}"

        # Invalidation
        invalidation = (sig.get("invalidation") or "—")[:35]

        # Key levels
        levels = sig.get("key_levels", {})
        level_parts = []
        if levels.get("support"):
            level_parts.append(f"S:{levels['support']:.2f}")
        if levels.get("resistance"):
            level_parts.append(f"R:{levels['resistance']:.2f}")
        if levels.get("vwap"):
            level_parts.append(f"VWAP:{levels['vwap']:.2f}")
        level_str = "  ".join(level_parts) if level_parts else "—"

        table.add_row(
            sym,
            fresh_str,
            f"[{s_color}]{s}[/{s_color}]",
            cat_str,
            invalidation,
            level_str,
        )

    return Panel(table, title="[bold yellow]📰 Analysis Summaries[/bold yellow]", box=box.ROUNDED)


def build_reasoning_panels(symbols: list[str], signals: dict) -> Panel:
    """Per-symbol reasoning from Analyst."""
    lines = []
    for sym in symbols:
        sig = signals.get(sym, {})
        reasoning = sig.get("reasoning", "")
        if not reasoning:
            continue
        signal = sig.get("signal", "?")
        sc = {"LONG": "green", "SHORT": "red", "HOLD": "yellow"}.get(signal, "dim")
        lines.append(
            f"[bold]{sym}[/bold] [{sc}]{signal}[/{sc}]  "
            f"[dim]{reasoning[:100]}[/dim]"
        )

    content = "\n".join(lines) if lines else "[dim]No analyst reasoning available yet.[/dim]"
    return Panel(content, title="[bold blue]🧠 Analyst Reasoning[/bold blue]", box=box.ROUNDED)


def build_portfolio_panel(portfolio: dict) -> Panel:
    table = Table(box=box.SIMPLE, header_style="bold", show_edge=False, expand=True)
    table.add_column("Profile")
    table.add_column("Equity", justify="right")
    table.add_column("Cash", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("Pos", justify="right")

    for profile_id in ACTIVE_PROFILES:
        profile = PM_PROFILES[profile_id]
        p = portfolio.get(profile_id, {})
        pnl = p.get("pnl", 0)
        pnl_pct = p.get("pnl_pct", 0)
        pnl_color = "green" if pnl >= 0 else "red"
        table.add_row(
            f"{profile['emoji']} {profile['name']}",
            f"${p.get('equity', 0):,.2f}",
            f"${p.get('cash', 0):,.2f}",
            f"[{pnl_color}]${pnl:+,.2f} ({pnl_pct:+.2f}%)[/{pnl_color}]",
            str(p.get("positions", 0)),
        )

    return Panel(table, title="[bold green]💼 Portfolios[/bold green]", box=box.ROUNDED)


def build_open_positions_panel(db) -> Panel:
    rows = []
    # Pre-compute news governance status for open trades
    news_status_map = {}  # symbol+profile -> display_status
    if NEWS_GOVERNANCE.get("enabled", False):
        try:
            classifier = NewsGovernanceClassifier()
            policy = NewsGovernancePolicy()
            now_utc = datetime.utcnow()

            open_trades = db.query(Trade).filter_by(status="open").all()
            for trade in open_trades:
                # Check persisted classification first
                persisted = classifier.get_persisted_classification(db, trade.id)
                if persisted:
                    is_governed = True
                else:
                    trade_dict = {
                        "setup_type": trade.setup_type or "",
                        "reason_entry": trade.reason_entry or "",
                        "thesis": trade.thesis or "",
                        "invalidators": trade.invalidators or "",
                    }
                    is_governed, _evidence = classifier.classify(trade_dict)

                if is_governed and trade.entry_time:
                    status_info = policy.evaluate(db, trade.id, trade.entry_time, now_utc)
                    # Derive display status
                    if status_info.get("hold_authorized"):
                        display_status = "authorized_hold"
                    else:
                        display_status = status_info["status"]
                    key = f"{trade.symbol}:{trade.profile}"
                    news_status_map[key] = display_status
        except Exception:
            pass  # Graceful degradation — don't break display if governance fails

    # Status color mapping
    status_colors = {
        "ok": "green",
        "warning": "yellow",
        "grace": "dark_orange",
        "expired": "red",
        "authorized_hold": "cyan",
    }
    status_icons = {
        "ok": "✓",
        "warning": "⚠",
        "grace": "⏳",
        "expired": "✗",
        "authorized_hold": "🔒",
    }

    for profile_id in ACTIVE_PROFILES:
        profile = PM_PROFILES[profile_id]
        positions = db.query(Position).filter_by(profile=profile_id).all()
        for p in positions:
            row = (f"{profile['emoji']} [bold]{p.symbol}[/bold] "
                   f"[dim]{p.side.upper()} {p.quantity} @ ${p.avg_cost:.2f}[/dim]")

            # Append news governance status badge if governed
            key = f"{p.symbol}:{profile_id}"
            if key in news_status_map:
                status = news_status_map[key]
                color = status_colors.get(status, "dim")
                icon = status_icons.get(status, "?")
                row += f" [{color}]{icon} NEWS:{status}[/{color}]"

            rows.append(row)

    content = "\n".join(rows) if rows else "[dim]No open positions[/dim]"
    return Panel(content, title="[bold green]📋 Open Positions[/bold green]", box=box.ROUNDED)


def build_strategy_panel() -> Panel:
    ctx = build_strategy_context(engine)
    return Panel(ctx, title="[bold blue]📐 Strategy Today[/bold blue]", box=box.ROUNDED)


# ─── RENDER ───────────────────────────────────────────────────────────────────

def render(symbols: list[str], scout_picks: list[str]) -> str:
    """Build the full display. Returns a renderable."""
    fh = FinnhubClient()
    db = get_session(engine)

    market_open = False
    try:
        market_open = fh.is_market_open()
    except Exception:
        pass

    signals = get_analyst_signals(db, symbols)
    sentiment = get_researcher_sentiment(db, symbols)
    portfolio = get_portfolio_summary(db, fh)

    # Compute freshness data and breaking news for the analysis table
    now_et = datetime.now(ET)
    market_day_start = get_market_day_start(now_et)

    breaking_news_by_symbol = {}
    try:
        breaking_news_by_symbol = get_breaking_news_for_symbols(
            db, symbols, market_day_start
        )
    except Exception:
        breaking_news_by_symbol = {sym: [] for sym in symbols}

    freshness_by_symbol = {}
    try:
        freshness_by_symbol = compute_catalyst_freshness(
            db, symbols, now=now_et,
            breaking_news_by_symbol=breaking_news_by_symbol,
        )
    except Exception:
        pass

    from rich.console import Group

    return Group(
        build_header(market_open),
        build_prices_table(symbols, fh, signals, scout_picks),
        build_analysis_table(symbols, signals, sentiment,
                             freshness_data=freshness_by_symbol,
                             breaking_news_data=breaking_news_by_symbol),
        build_reasoning_panels(symbols, signals),
        Columns([
            build_portfolio_panel(portfolio),
            build_open_positions_panel(db),
        ]),
        build_strategy_panel(),
        Panel(
            f"[dim]Last refresh: {datetime.now().strftime('%H:%M:%S')}  "
            f"Press Ctrl+C to exit[/dim]",
            box=box.HORIZONTALS, style="dim",
        ),
    )


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Paper Trader — Live Display")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    parser.add_argument("--symbols", type=str, help="Override watchlist (comma-separated)")
    args = parser.parse_args()

    core = [s.strip() for s in os.getenv("WATCHLIST", "SPY,QQQ,IWM,DIA,TLT,GLD,XLK,XLF,XLE,TSLA,NVDA,AMD").split(",")]
    if args.symbols:
        core = [s.strip() for s in args.symbols.split(",")]

    db = get_session(engine)
    scout_picks = get_scout_picks(db)
    db.close()

    all_symbols = core + [s for s in scout_picks if s not in core]

    if args.once:
        console.print(render(all_symbols, scout_picks))
        return

    console.print(f"[dim]Watching: {', '.join(all_symbols)} | Refresh: {args.interval}s | Ctrl+C to exit[/dim]")

    with Live(render(all_symbols, scout_picks), refresh_per_second=0.5, screen=True) as live:
        try:
            while True:
                time.sleep(args.interval)
                # Refresh scout picks each cycle
                db = get_session(engine)
                scout_picks = get_scout_picks(db)
                db.close()
                all_symbols = core + [s for s in scout_picks if s not in core]
                live.update(render(all_symbols, scout_picks))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
