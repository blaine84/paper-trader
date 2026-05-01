"""
Weekly Prep Agent
Runs Sunday afternoon/evening. Sets the tone for the upcoming week.

Produces:
  - Preliminary watchlist with weekly thesis per symbol
  - Weekend news + macro event calendar summary
  - Last week's lesson rollup (from case library)
  - PM stance adjustments (more defensive / more aggressive / neutral)
  - Monday morning context injected into agent memory

Monday's 8:30 AM run reads this context and hits the ground running.
"""

import json
import os
from datetime import datetime, timedelta
from utils.finnhub_client import FinnhubClient
from utils.llm import call_llm, parse_json_response
from utils.case_library import (
    query_cases, get_win_rate_by_setup,
    get_selection_feedback, get_execution_feedback,
    format_cases_for_prompt,
)
from db.schema import AgentMemory, DailyLog, Trade, get_session
from models.pm_profiles import PM_PROFILES, ACTIVE_PROFILES
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


# ─── SCOUT: Weekly Watchlist ──────────────────────────────────────────────────

SCOUT_WEEKLY_PROMPT = """You are building a preliminary watchlist for the coming trading week.

Your job: identify which stocks from the candidate list are most worth watching
next week and why. For each pick, write a weekly thesis — not a trade, just a read
on what to watch for.

Consider:
- Technical setup heading into the week (where is price relative to key levels?)
- Any catalysts on the calendar (earnings, macro events, Fed, options expiry)
- Sector / macro tailwinds or headwinds
- How this name has behaved recently (from case history)
- Whether the setup favors long, short, or is ambiguous

Respond in JSON:
{
  "watchlist": [
    {
      "symbol": "NVDA",
      "weekly_bias": "LONG|SHORT|NEUTRAL",
      "conviction": "low|medium|high",
      "thesis": "what to watch for this week",
      "key_levels": { "support": 0.0, "resistance": 0.0 },
      "catalysts_this_week": ["earnings Thursday", "CPI Tuesday"],
      "setup_type": "consolidation|breakout_watch|reversal_watch|trend_continuation|avoid",
      "risk": "main risk to the thesis"
    }
  ],
  "symbols_to_avoid": ["TICKER"],
  "scout_notes": "overall market structure heading into the week"
}

Return all core watchlist symbols plus any additional names worth adding.
Flag any symbols to avoid this week (e.g. earnings landmines, overextended).
"""


def run_scout_weekly(engine, core_watchlist: list[str], fh: FinnhubClient) -> dict:
    db = get_session(engine)

    # Weekend news
    market_news = fh.get_market_news("general")
    forex_news = fh.get_market_news("forex")

    # EOW quotes for all core symbols
    quotes = {}
    for sym in core_watchlist:
        try:
            quotes[sym] = fh.get_quote(sym)
        except Exception:
            pass

    # Case win rates — what has been working?
    win_rates = get_win_rate_by_setup(engine)
    win_rate_text = "\n".join(
        f"  {r['setup_type']}: {r['win_rate']}% ({r['total']} cases, avg pnl {r['avg_pnl_pct']:+.2f}%)"
        for r in win_rates
    ) if win_rates else "No case history yet."

    # Recent selection feedback
    sel_fb = get_selection_feedback(engine, limit=5)

    db.close()

    user_prompt = f"""
Week starting: {(datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%d')}
Core watchlist: {', '.join(core_watchlist)}

WEEKEND MARKET NEWS:
{json.dumps(market_news[:10], indent=2)}

MACRO / FOREX:
{json.dumps(forex_news[:5], indent=2)}

END-OF-WEEK QUOTES:
{json.dumps(quotes, indent=2)}

SETUP WIN RATES (historical):
{win_rate_text}

RECENT SELECTION FEEDBACK:
{sel_fb}

Build the weekly watchlist and thesis for each symbol.
"""

    raw = call_llm(SCOUT_WEEKLY_PROMPT, user_prompt, json_mode=True, tier="low", purpose="weekly_prep:scout")
    return parse_json_response(raw)


# ─── RESEARCHER: Weekend News Summary ─────────────────────────────────────────

RESEARCHER_WEEKLY_PROMPT = """You are writing a weekly market briefing for day traders.

Summarize:
1. What happened last week (market tone, key moves, sector rotation)
2. Major events on the calendar this week (Fed, CPI, PPI, earnings, options expiry)
3. Macro headwinds and tailwinds
4. Sector themes to watch

Respond in JSON:
{
  "last_week_summary": "2-3 sentences on last week",
  "market_regime": "risk_on|risk_off|mixed",
  "regime_confidence": "low|medium|high",
  "weekly_events": [
    { "date": "Mon", "event": "...", "impact": "low|medium|high" }
  ],
  "sector_themes": [
    { "sector": "tech", "bias": "bullish|bearish|neutral", "reason": "..." }
  ],
  "macro_headwinds": ["..."],
  "macro_tailwinds": ["..."],
  "key_risk": "the single biggest risk to watch this week"
}
"""


def run_researcher_weekly(fh: FinnhubClient) -> dict:
    market_news = fh.get_market_news("general")
    merger_news = fh.get_market_news("merger")

    user_prompt = f"""
Date: {datetime.utcnow().strftime('%Y-%m-%d')} (Sunday)

WEEKEND NEWS:
{json.dumps(market_news[:12], indent=2)}

M&A / CORPORATE:
{json.dumps(merger_news[:5], indent=2)}

Write the weekly briefing.
"""
    raw = call_llm(RESEARCHER_WEEKLY_PROMPT, user_prompt, json_mode=True, tier="low", purpose="weekly_prep:researcher")
    return parse_json_response(raw)


# ─── REVIEWER: Weekly Lesson Rollup ───────────────────────────────────────────

REVIEWER_WEEKLY_PROMPT = """You are rolling up last week's trading lessons into a
concise, structured weekly retrospective.

Do NOT write prose. Produce typed, actionable outputs.

Respond in JSON:
{
  "week": "YYYY-MM-DD",
  "total_trades": 0,
  "winning_trades": 0,
  "losing_trades": 0,
  "avg_selection_score": 0.0,
  "avg_execution_score": 0.0,
  "best_setup_type": "...",
  "worst_setup_type": "...",
  "top_lessons": [
    {
      "lesson": "one actionable sentence",
      "setup_type": "...",
      "applies_to": "scout|analyst|pm_conservative|pm_moderate|pm_aggressive|all"
    }
  ],
  "patterns_identified": [
    {
      "pattern": "description of recurring pattern",
      "frequency": "occasional|frequent|consistent",
      "action": "what to do about it next week"
    }
  ],
  "profile_retrospective": {
    "conservative": "how did this profile perform and why",
    "moderate": "...",
    "aggressive": "..."
  },
  "next_week_focus": [
    "one specific thing each agent should focus on improving"
  ]
}
"""


def run_reviewer_weekly(engine) -> dict:
    db = get_session(engine)

    # Last week's trades
    week_ago = datetime.utcnow() - timedelta(days=7)
    last_week_trades = (
        db.query(Trade)
        .filter(Trade.status == "closed")
        .filter(Trade.exit_time >= week_ago)
        .all()
    )

    # Last week's daily logs
    last_week_logs = (
        db.query(DailyLog)
        .order_by(DailyLog.date.desc())
        .limit(5)
        .all()
    )

    # Last week's cases
    last_week_cases = query_cases(engine, limit=20)

    db.close()

    trade_data = [
        {
            "symbol": t.symbol,
            "profile": t.profile,
            "direction": t.direction,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "review_score": t.review_score,
            "review_notes": t.review_notes,
        }
        for t in last_week_trades
    ]

    log_data = [
        {
            "date": l.date,
            "daily_pnl": l.daily_pnl,
            "daily_pnl_pct": l.daily_pnl_pct,
            "trades": l.trades_taken,
            "wins": l.winning_trades,
            "losses": l.losing_trades,
        }
        for l in last_week_logs
    ]

    user_prompt = f"""
Week ending: {datetime.utcnow().strftime('%Y-%m-%d')}

LAST WEEK'S TRADES:
{json.dumps(trade_data, indent=2)}

DAILY LOGS:
{json.dumps(log_data, indent=2)}

CASE LIBRARY SAMPLE (last 20):
{format_cases_for_prompt(last_week_cases)}

Produce the weekly retrospective.
"""
    raw = call_llm(REVIEWER_WEEKLY_PROMPT, user_prompt, json_mode=True, tier="low", purpose="weekly_prep:reviewer")
    return parse_json_response(raw)


# ─── PM: Weekly Stance ────────────────────────────────────────────────────────

PM_STANCE_PROMPT = """You are a portfolio manager setting your stance for the coming week.

Based on the weekly briefing, watchlist thesis, and last week's performance,
decide your posture for Monday open.

Stance options:
  - defensive: tighten rules, require stronger signals, reduce max position size
  - neutral:   follow normal profile rules
  - aggressive: slightly looser rules, willing to act on moderate signals earlier

Also set any symbol-specific overrides for the week.

Respond in JSON:
{
  "profile": "conservative|moderate|aggressive",
  "weekly_stance": "defensive|neutral|aggressive",
  "stance_reason": "one sentence explaining why",
  "size_adjustment": -0.25,  // multiplier on normal max position size (-0.5 = half size, 0 = normal, +0.25 = 25% larger)
  "signal_threshold_adjustment": "tighter|normal|looser",
  "symbols_avoid": ["TICKER"],       // skip these entirely this week
  "symbols_favor": ["TICKER"],       // be more willing to trade these
  "symbols_short_bias": ["TICKER"],  // lean short on these specifically
  "notes": "any other weekly context for this profile"
}
"""


def run_pm_stances(engine, briefing: dict, watchlist_result: dict) -> dict:
    stances = {}

    for profile_id in ACTIVE_PROFILES:
        profile = PM_PROFILES[profile_id]

        # Profile-specific execution feedback
        exe_fb = get_execution_feedback(engine, profile_id=profile_id, limit=5)

        user_prompt = f"""
Profile: {profile['name']} {profile['emoji']}
Profile personality: {profile['personality'].strip()}

WEEKLY BRIEFING:
{json.dumps(briefing, indent=2)}

WEEKLY WATCHLIST THESIS:
{json.dumps(watchlist_result.get('watchlist', []), indent=2)}

YOUR RECENT EXECUTION FEEDBACK:
{exe_fb}

Set your stance for the coming week.
"""
        raw = call_llm(PM_STANCE_PROMPT, user_prompt, json_mode=True, purpose=f"weekly_prep:pm_stance:{profile_id}")
        stance = parse_json_response(raw)
        stance["profile"] = profile_id
        stances[profile_id] = stance

    return stances


# ─── ORCHESTRATOR: Write Monday Context ───────────────────────────────────────

def write_monday_context(engine, briefing: dict, watchlist: dict,
                          rollup: dict, stances: dict):
    """Persist weekly prep results to agent memory for Monday morning."""
    db = get_session(engine)
    week_str = datetime.utcnow().strftime("%Y-%m-%d")

    # Weekly briefing → Researcher context
    db.add(AgentMemory(
        agent="weekly_prep",
        symbol=None,
        key="weekly_briefing",
        value=json.dumps({**briefing, "week": week_str}),
    ))

    # Weekly watchlist → Scout + Analyst context
    db.add(AgentMemory(
        agent="weekly_prep",
        symbol=None,
        key="weekly_watchlist",
        value=json.dumps({**watchlist, "week": week_str}),
    ))

    # Lesson rollup → all agents
    db.add(AgentMemory(
        agent="weekly_prep",
        symbol=None,
        key="weekly_rollup",
        value=json.dumps({**rollup, "week": week_str}),
    ))

    # Per-profile stances → each PM
    for profile_id, stance in stances.items():
        db.add(AgentMemory(
            agent="weekly_prep",
            symbol=None,
            key=f"weekly_stance_{profile_id}",
            value=json.dumps(stance),
        ))

    db.commit()
    db.close()


# ─── PRINT REPORT ─────────────────────────────────────────────────────────────

def print_weekly_report(briefing: dict, watchlist: dict,
                         rollup: dict, stances: dict):
    console.print()
    console.print(Panel(
        f"[bold white]Weekly Prep — Week of {(datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%d')}[/bold white]",
        style="bold blue",
        box=box.DOUBLE,
    ))

    # Market briefing
    regime = briefing.get("market_regime", "?")
    regime_color = {"risk_on": "green", "risk_off": "red", "mixed": "yellow"}.get(regime, "white")
    console.print(Panel(
        f"[bold]Last Week:[/bold] {briefing.get('last_week_summary', '')}\n"
        f"[bold]Regime:[/bold] [{regime_color}]{regime}[/{regime_color}] "
        f"({briefing.get('regime_confidence', '?')} confidence)\n"
        f"[bold]Key Risk:[/bold] {briefing.get('key_risk', '')}",
        title="[bold cyan]📰 Weekly Briefing[/bold cyan]",
        box=box.ROUNDED,
    ))

    # Calendar events
    events = briefing.get("weekly_events", [])
    if events:
        cal = Table(box=box.SIMPLE, header_style="bold")
        cal.add_column("Day")
        cal.add_column("Event")
        cal.add_column("Impact")
        for e in events:
            impact = e.get("impact", "?")
            ic = {"high": "red", "medium": "yellow", "low": "dim"}.get(impact, "white")
            cal.add_row(e.get("date", "?"), e.get("event", "?"), f"[{ic}]{impact}[/{ic}]")
        console.print(Panel(cal, title="[bold cyan]📅 Calendar This Week[/bold cyan]", box=box.ROUNDED))

    # Weekly watchlist
    wl = watchlist.get("watchlist", [])
    if wl:
        wt = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan", title="🔭 Weekly Watchlist")
        wt.add_column("Symbol", style="bold")
        wt.add_column("Bias")
        wt.add_column("Conviction")
        wt.add_column("Setup")
        wt.add_column("Thesis")
        for w in wl:
            bias = w.get("weekly_bias", "?")
            bc = {"LONG": "green", "SHORT": "red", "NEUTRAL": "yellow"}.get(bias, "white")
            conv = w.get("conviction", "?")
            cc = {"high": "green", "medium": "yellow", "low": "dim"}.get(conv, "white")
            wt.add_row(
                w.get("symbol", "?"),
                f"[{bc}]{bias}[/{bc}]",
                f"[{cc}]{conv}[/{cc}]",
                w.get("setup_type", "?") or "?",
                (w.get("thesis", "") or "")[:60],
            )
        console.print(wt)

    avoid = watchlist.get("symbols_to_avoid", [])
    if avoid:
        console.print(f"  [red]Avoid this week:[/red] {', '.join(avoid)}")

    # Weekly rollup
    top_lessons = rollup.get("top_lessons", [])
    patterns = rollup.get("patterns_identified", [])
    if top_lessons or patterns:
        lines = []
        if top_lessons:
            lines.append("[bold]Top Lessons:[/bold]")
            for l in top_lessons[:5]:
                lines.append(f"  • [{l.get('applies_to', 'all')}] {l.get('lesson', '')}")
        if patterns:
            lines.append("\n[bold]Patterns:[/bold]")
            for p in patterns[:3]:
                lines.append(f"  • {p.get('pattern', '')} → {p.get('action', '')}")
        console.print(Panel(
            "\n".join(lines),
            title="[bold magenta]🔍 Last Week's Rollup[/bold magenta]",
            box=box.ROUNDED,
        ))

    # PM stances
    stance_lines = []
    for profile_id, stance in stances.items():
        profile = PM_PROFILES[profile_id]
        s = stance.get("weekly_stance", "neutral")
        sc = {"defensive": "red", "neutral": "yellow", "aggressive": "green"}.get(s, "white")
        adj = stance.get("size_adjustment", 0)
        adj_str = f"{adj:+.0%}" if adj else "normal"
        stance_lines.append(
            f"{profile['emoji']} [bold]{profile['name']}[/bold]: "
            f"[{sc}]{s}[/{sc}] | size {adj_str} | {stance.get('stance_reason', '')}"
        )
        if stance.get("symbols_avoid"):
            stance_lines.append(f"   avoid: {', '.join(stance['symbols_avoid'])}")
        if stance.get("symbols_favor"):
            stance_lines.append(f"   favor: {', '.join(stance['symbols_favor'])}")

    if stance_lines:
        console.print(Panel(
            "\n".join(stance_lines),
            title="[bold green]🧠 PM Stances for Monday[/bold green]",
            box=box.ROUNDED,
        ))

    console.print()
    console.print("[bold green]✅ Monday context written. Agents are ready.[/bold green]\n")


# ─── MAIN ENTRY ───────────────────────────────────────────────────────────────

def run(engine, core_watchlist: list[str]) -> dict:
    fh = FinnhubClient()
    console.print(Panel(
        "[bold]Sunday Weekly Prep running...[/bold]",
        style="blue", box=box.ROUNDED,
    ))

    console.print("[bold cyan]🔭 Scout: building weekly watchlist...[/bold cyan]")
    watchlist = run_scout_weekly(engine, core_watchlist, fh)

    console.print("[bold yellow]📰 Researcher: summarizing weekend news...[/bold yellow]")
    briefing = run_researcher_weekly(fh)

    console.print("[bold magenta]🔍 Reviewer: rolling up last week...[/bold magenta]")
    rollup = run_reviewer_weekly(engine)

    console.print("[bold blue]📐 Quant Researcher: weekly strategy fit...[/bold blue]")
    from agents.quant_researcher import run as qr_run, print_report as qr_print
    qr_result = qr_run(engine, market_regime=briefing.get("market_regime"))
    qr_print(qr_result)

    console.print("[bold green]🧠 PMs: setting weekly stances...[/bold green]")
    stances = run_pm_stances(engine, briefing, watchlist)

    console.print("[bold]📝 Writing Monday context...[/bold]")
    write_monday_context(engine, briefing, watchlist, rollup, stances)

    print_weekly_report(briefing, watchlist, rollup, stances)

    return {
        "watchlist": watchlist,
        "briefing": briefing,
        "rollup": rollup,
        "stances": stances,
    }
