"""
Paper Trader — Inspect CLI
Query the case library, scores, feedback, and trade history.

Usage:
  python inspect.py cases                   # recent cases
  python inspect.py cases --setup gap_and_go
  python inspect.py cases --outcome failure
  python inspect.py cases --symbol TSLA
  python inspect.py cases --regime risk_off
  python inspect.py scores                  # score trends over time
  python inspect.py feedback                # latest agent feedback
  python inspect.py feedback --profile aggressive
  python inspect.py winrates                # win rates by setup type
  python inspect.py trades                  # recent closed trades
  python inspect.py trades --profile conservative
  python inspect.py positions               # current open positions (all profiles)
  python inspect.py summary                 # daily P&L summary log
  python inspect.py weekly                  # last weekly prep report
  python inspect.py quant                   # latest strategy recommendations
"""

import argparse
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from db.schema import init_db, get_session, Trade, DailyLog
from models.pm_profiles import PM_PROFILES, ACTIVE_PROFILES
from utils.case_library import (
    query_cases, get_win_rate_by_setup,
    get_selection_feedback, get_execution_feedback,
)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()
engine = init_db("db/paper_trader.db")


# ─── CASES ────────────────────────────────────────────────────────────────────

def cmd_cases(args):
    cases = query_cases(
        engine,
        setup_type=args.setup,
        catalyst_type=args.catalyst,
        symbol=args.symbol,
        market_regime=args.regime,
        outcome=args.outcome,
        bias=args.bias,
        limit=args.limit,
    )

    if not cases:
        console.print("[dim]No cases found matching filters.[/dim]")
        return

    table = Table(
        title=f"Case Library ({len(cases)} results)",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Date", style="dim")
    table.add_column("Symbol", style="bold")
    table.add_column("Profile", style="dim")
    table.add_column("Setup")
    table.add_column("Catalyst")
    table.add_column("Regime")
    table.add_column("Bias")
    table.add_column("Outcome")
    table.add_column("P&L%")
    table.add_column("Sel", justify="right")
    table.add_column("Exe", justify="right")

    for c in cases:
        outcome = c.get("outcome", "?")
        oc = {"success": "green", "failure": "red", "partial": "yellow"}.get(outcome, "white")
        pnl = c.get("pnl_pct")
        pnl_str = f"{pnl:+.2f}%" if pnl is not None else "?"
        pc = "green" if (pnl or 0) >= 0 else "red"

        def sc(s):
            if s is None: return "[dim]—[/dim]"
            c_ = "green" if s >= 7 else "yellow" if s >= 5 else "red"
            return f"[{c_}]{s:.1f}[/{c_}]"

        table.add_row(
            c.get("date", "?"),
            c.get("symbol", "?"),
            c.get("profile", "?") or "?",
            c.get("setup_type", "?") or "?",
            c.get("catalyst_type", "?") or "?",
            c.get("market_regime", "?") or "?",
            c.get("bias", "?") or "?",
            f"[{oc}]{outcome}[/{oc}]",
            f"[{pc}]{pnl_str}[/{pc}]",
            sc(c.get("selection_score")),
            sc(c.get("execution_score")),
        )

    console.print(table)

    # Expand lessons if verbose
    if args.verbose:
        for c in cases:
            lines = []
            if c.get("lesson"):
                lines.append(f"[bold]Lesson:[/bold] {c['lesson']}")
            if c.get("invalidation"):
                lines.append(f"[bold]Invalidation:[/bold] {c['invalidation']}")
            if c.get("conditions_for_success"):
                lines.append(f"[bold]Works when:[/bold] {', '.join(c['conditions_for_success'])}")
            if c.get("conditions_to_avoid"):
                lines.append(f"[bold]Avoid when:[/bold] {', '.join(c['conditions_to_avoid'])}")
            if lines:
                console.print(Panel(
                    "\n".join(lines),
                    title=f"[bold]{c['date']} {c['symbol']}[/bold] — {c.get('setup_type', '?')}",
                    box=box.ROUNDED, style="dim",
                ))


# ─── SCORES ───────────────────────────────────────────────────────────────────

def cmd_scores(args):
    cases = query_cases(engine, limit=args.limit)
    if not cases:
        console.print("[dim]No scored cases yet.[/dim]")
        return

    table = Table(
        title="Score Trends (most recent first)",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
    )
    table.add_column("Date")
    table.add_column("Symbol", style="bold")
    table.add_column("Setup")
    table.add_column("Outcome")
    table.add_column("Selection", justify="right")
    table.add_column("Execution", justify="right")
    table.add_column("Overall", justify="right")
    table.add_column("Lesson")

    for c in cases:
        outcome = c.get("outcome", "?")
        oc = {"success": "green", "failure": "red", "partial": "yellow"}.get(outcome, "white")

        def sc(s):
            if s is None: return "[dim]—[/dim]"
            c_ = "green" if s >= 7 else "yellow" if s >= 5 else "red"
            return f"[{c_}]{s:.1f}[/{c_}]"

        table.add_row(
            c.get("date", "?"),
            c.get("symbol", "?"),
            c.get("setup_type", "?") or "?",
            f"[{oc}]{outcome}[/{oc}]",
            sc(c.get("selection_score")),
            sc(c.get("execution_score")),
            sc(c.get("review_score")),
            (c.get("lesson") or "")[:60],
        )

    console.print(table)

    # Averages
    sel_scores = [c["selection_score"] for c in cases if c.get("selection_score")]
    exe_scores = [c["execution_score"] for c in cases if c.get("execution_score")]
    if sel_scores:
        console.print(f"\n  Avg selection score: [cyan]{sum(sel_scores)/len(sel_scores):.1f}[/cyan]")
    if exe_scores:
        console.print(f"  Avg execution score: [cyan]{sum(exe_scores)/len(exe_scores):.1f}[/cyan]")


# ─── FEEDBACK ─────────────────────────────────────────────────────────────────

def cmd_feedback(args):
    db = get_session(engine)

    # Selection feedback (Scout + Analyst)
    from db.schema import AgentMemory
    sel_fb = (
        db.query(AgentMemory)
        .filter_by(agent="reviewer", key="selection_feedback")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    if sel_fb and (not args.profile):
        console.print(Panel(
            sel_fb.value,
            title=f"[bold yellow]📊 Selection Feedback → Scout + Analyst[/bold yellow]\n[dim]{sel_fb.timestamp}[/dim]",
            box=box.ROUNDED,
        ))

    # Execution feedback per profile
    profiles = [args.profile] if args.profile else ACTIVE_PROFILES
    for profile_id in profiles:
        exe_fb = (
            db.query(AgentMemory)
            .filter_by(agent="reviewer", key=f"execution_feedback_{profile_id}")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if exe_fb:
            emoji = {"conservative": "🛡️", "moderate": "⚖️", "aggressive": "🔥"}.get(profile_id, "")
            console.print(Panel(
                exe_fb.value,
                title=f"[bold green]⚙️ Execution Feedback → {emoji} {profile_id.capitalize()} PM[/bold green]\n[dim]{exe_fb.timestamp}[/dim]",
                box=box.ROUNDED,
            ))
        else:
            console.print(f"[dim]No execution feedback yet for {profile_id}.[/dim]")

    db.close()


# ─── WIN RATES ────────────────────────────────────────────────────────────────

def cmd_winrates(args):
    rates = get_win_rate_by_setup(engine)
    if not rates:
        console.print("[dim]No cases in library yet.[/dim]")
        return

    table = Table(
        title="Win Rates by Setup Type",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
    )
    table.add_column("Setup Type", style="bold")
    table.add_column("Total", justify="right")
    table.add_column("Wins", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Avg P&L %", justify="right")
    table.add_column("Avg Score", justify="right")

    rates.sort(key=lambda x: x["win_rate"], reverse=True)
    for r in rates:
        wr = r["win_rate"]
        wr_color = "green" if wr >= 60 else "yellow" if wr >= 40 else "red"
        pnl = r["avg_pnl_pct"]
        pc = "green" if pnl >= 0 else "red"
        table.add_row(
            r["setup_type"],
            str(r["total"]),
            str(r["wins"]),
            f"[{wr_color}]{wr}%[/{wr_color}]",
            f"[{pc}]{pnl:+.2f}%[/{pc}]",
            f"{r['avg_score']:.1f}",
        )

    console.print(table)


# ─── TRADES ───────────────────────────────────────────────────────────────────

def cmd_trades(args):
    db = get_session(engine)
    q = db.query(Trade).filter_by(status="closed")
    if args.profile:
        q = q.filter_by(profile=args.profile)
    trades = q.order_by(Trade.exit_time.desc()).limit(args.limit).all()
    db.close()

    if not trades:
        console.print("[dim]No closed trades found.[/dim]")
        return

    table = Table(
        title=f"Closed Trades ({len(trades)})",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
    )
    table.add_column("Exit Time", style="dim")
    table.add_column("Symbol", style="bold")
    table.add_column("Profile")
    table.add_column("Direction")
    table.add_column("Qty", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Exit", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("Score", justify="right")

    for t in trades:
        pnl_color = "green" if (t.pnl or 0) >= 0 else "red"
        sc = f"[cyan]{t.review_score:.1f}[/cyan]" if t.review_score else "[dim]—[/dim]"
        table.add_row(
            t.exit_time.strftime("%m-%d %H:%M") if t.exit_time else "?",
            t.symbol,
            t.profile or "?",
            t.direction,
            str(t.quantity),
            f"${t.entry_price:.2f}",
            f"${t.exit_price:.2f}" if t.exit_price else "?",
            f"[{pnl_color}]${t.pnl:+,.2f} ({t.pnl_pct:+.1f}%)[/{pnl_color}]" if t.pnl else "?",
            sc,
        )

    console.print(table)


# ─── POSITIONS ────────────────────────────────────────────────────────────────

def cmd_positions(args):
    from agents.bookkeeper import print_dashboard
    print_dashboard(engine)


# ─── SUMMARY ──────────────────────────────────────────────────────────────────

def cmd_quant(args):
    """Show the latest strategy recommendations from the Quant Researcher."""
    from db.schema import AgentMemory
    from agents.quant_researcher import print_report
    db = get_session(engine)
    mem = (
        db.query(AgentMemory)
        .filter_by(agent="quant_researcher", key="strategy_recommendations")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    db.close()
    if not mem:
        console.print("[dim]No quant researcher data yet. Run: python orchestrator.py once[/dim]")
        return
    data = json.loads(mem.value)
    console.print(f"[dim]Generated: {data.get('timestamp', '?')}[/dim]")
    print_report(data)


def cmd_weekly(args):
    """Show the last weekly prep report."""
    from db.schema import AgentMemory
    db = get_session(engine)

    keys = ["weekly_briefing", "weekly_watchlist", "weekly_rollup",
            "weekly_stance_conservative", "weekly_stance_moderate", "weekly_stance_aggressive"]

    found_any = False
    for key in keys:
        mem = (
            db.query(AgentMemory)
            .filter_by(agent="weekly_prep", key=key)
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if not mem:
            continue
        found_any = True
        data = json.loads(mem.value)
        label = key.replace("_", " ").title()

        if key == "weekly_briefing":
            regime = data.get("market_regime", "?")
            rc = {"risk_on": "green", "risk_off": "red", "mixed": "yellow"}.get(regime, "white")
            console.print(Panel(
                f"[bold]Week:[/bold] {data.get('week')}\n"
                f"[bold]Last week:[/bold] {data.get('last_week_summary', '')}\n"
                f"[bold]Regime:[/bold] [{rc}]{regime}[/{rc}] ({data.get('regime_confidence', '?')} confidence)\n"
                f"[bold]Key risk:[/bold] {data.get('key_risk', '')}",
                title="[bold cyan]📰 Weekly Briefing[/bold cyan]", box=box.ROUNDED,
            ))

        elif key == "weekly_watchlist":
            wl = data.get("watchlist", [])
            if wl:
                t = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan", title="🔭 Weekly Watchlist")
                t.add_column("Symbol", style="bold")
                t.add_column("Bias")
                t.add_column("Conviction")
                t.add_column("Setup")
                t.add_column("Thesis")
                for w in wl:
                    bias = w.get("weekly_bias", "?")
                    bc = {"LONG": "green", "SHORT": "red", "NEUTRAL": "yellow"}.get(bias, "white")
                    t.add_row(
                        w.get("symbol", "?"),
                        f"[{bc}]{bias}[/{bc}]",
                        w.get("conviction", "?"),
                        w.get("setup_type", "?") or "?",
                        (w.get("thesis", "") or "")[:70],
                    )
                console.print(t)

        elif key == "weekly_rollup":
            lessons = data.get("top_lessons", [])
            patterns = data.get("patterns_identified", [])
            lines = []
            if lessons:
                lines.append("[bold]Top Lessons:[/bold]")
                for l in lessons:
                    lines.append(f"  • [{l.get('applies_to', 'all')}] {l.get('lesson', '')}")
            if patterns:
                lines.append("\n[bold]Patterns:[/bold]")
                for p in patterns:
                    lines.append(f"  • {p.get('pattern', '')} → {p.get('action', '')}")
            if data.get("next_week_focus"):
                lines.append("\n[bold]Next Week Focus:[/bold]")
                for f in data["next_week_focus"]:
                    lines.append(f"  • {f}")
            if lines:
                console.print(Panel("\n".join(lines), title="[bold magenta]🔍 Weekly Rollup[/bold magenta]", box=box.ROUNDED))

        elif key.startswith("weekly_stance_"):
            profile_id = key.replace("weekly_stance_", "")
            emoji = {"conservative": "🛡️", "moderate": "⚖️", "aggressive": "🔥"}.get(profile_id, "")
            s = data.get("weekly_stance", "neutral")
            sc = {"defensive": "red", "neutral": "yellow", "aggressive": "green"}.get(s, "white")
            adj = data.get("size_adjustment", 0)
            lines = [
                f"Stance: [{sc}]{s}[/{sc}]",
                f"Reason: {data.get('stance_reason', '')}",
                f"Size adjustment: {adj:+.0%}" if adj else "Size: normal",
                f"Signal threshold: {data.get('signal_threshold_adjustment', 'normal')}",
            ]
            if data.get("symbols_avoid"):
                lines.append(f"Avoid: {', '.join(data['symbols_avoid'])}")
            if data.get("symbols_favor"):
                lines.append(f"Favor: {', '.join(data['symbols_favor'])}")
            if data.get("symbols_short_bias"):
                lines.append(f"Short bias: {', '.join(data['symbols_short_bias'])}")
            console.print(Panel(
                "\n".join(lines),
                title=f"[bold green]{emoji} {profile_id.capitalize()} PM Stance[/bold green]",
                box=box.ROUNDED,
            ))

    if not found_any:
        console.print("[dim]No weekly prep data found. Run: python orchestrator.py weekly[/dim]")

    db.close()


def cmd_summary(args):
    db = get_session(engine)
    logs = db.query(DailyLog).order_by(DailyLog.date.desc()).limit(args.limit).all()
    db.close()

    if not logs:
        console.print("[dim]No daily logs yet.[/dim]")
        return

    table = Table(
        title="Daily P&L Summary",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
    )
    table.add_column("Date", style="bold")
    table.add_column("Equity", justify="right")
    table.add_column("Daily P&L", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("W/L", justify="right")
    table.add_column("Win Rate", justify="right")

    for l in logs:
        pnl_color = "green" if (l.daily_pnl or 0) >= 0 else "red"
        wr = round(l.winning_trades / l.trades_taken * 100, 1) if l.trades_taken else 0
        wr_color = "green" if wr >= 60 else "yellow" if wr >= 40 else "red"
        table.add_row(
            l.date,
            f"${l.ending_equity:,.2f}" if l.ending_equity else "?",
            f"[{pnl_color}]${l.daily_pnl:+,.2f} ({l.daily_pnl_pct:+.2f}%)[/{pnl_color}]",
            str(l.trades_taken),
            f"{l.winning_trades}W / {l.losing_trades}L",
            f"[{wr_color}]{wr}%[/{wr_color}]",
        )

    console.print(table)


# ─── CLI ENTRY ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Paper Trader — Inspect CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # cases
    p_cases = sub.add_parser("cases", help="Browse the case library")
    p_cases.add_argument("--setup", help="Filter by setup_type")
    p_cases.add_argument("--catalyst", help="Filter by catalyst_type")
    p_cases.add_argument("--symbol", help="Filter by symbol")
    p_cases.add_argument("--regime", help="Filter by market_regime")
    p_cases.add_argument("--outcome", help="Filter by outcome (success/failure/partial)")
    p_cases.add_argument("--bias", help="Filter by bias (LONG/SHORT)")
    p_cases.add_argument("--limit", type=int, default=20)
    p_cases.add_argument("-v", "--verbose", action="store_true", help="Show lessons + conditions")

    # scores
    p_scores = sub.add_parser("scores", help="Score trends over time")
    p_scores.add_argument("--limit", type=int, default=20)

    # feedback
    p_fb = sub.add_parser("feedback", help="Latest agent feedback")
    p_fb.add_argument("--profile", help="PM profile (conservative/moderate/aggressive)")

    # winrates
    sub.add_parser("winrates", help="Win rates by setup type")

    # trades
    p_trades = sub.add_parser("trades", help="Recent closed trades")
    p_trades.add_argument("--profile", help="Filter by PM profile")
    p_trades.add_argument("--limit", type=int, default=20)

    # positions
    sub.add_parser("positions", help="Current open positions (all profiles)")

    # summary
    p_summary = sub.add_parser("summary", help="Daily P&L summary log")
    p_summary.add_argument("--limit", type=int, default=30)

    args = parser.parse_args()

    # weekly
    sub.add_parser("weekly", help="Last weekly prep report")

    # quant
    sub.add_parser("quant", help="Latest strategy recommendations")

    dispatch = {
        "cases":     cmd_cases,
        "scores":    cmd_scores,
        "feedback":  cmd_feedback,
        "winrates":  cmd_winrates,
        "trades":    cmd_trades,
        "positions": cmd_positions,
        "summary":   cmd_summary,
        "weekly":    cmd_weekly,
        "quant":     cmd_quant,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
