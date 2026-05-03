"""
Quant Researcher Agent
Matches current market conditions to historically effective strategies.
Cross-references the strategy library against the internal case library
to surface agreements and disagreements.
Runs Sunday (weekly prep) and pre-market Monday.
Outputs strategy recommendations that feed Analyst and PM.
"""

import json
from datetime import datetime, timedelta, date
from utils.llm import call_llm, parse_json_response
from utils.case_library import query_cases, get_win_rate_by_setup, format_cases_for_prompt
from db.schema import AgentMemory, get_session
from models.strategies import STRATEGIES, SETUP_TYPE_MAP
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

import logging
log = logging.getLogger(__name__)

console = Console()


SYSTEM_PROMPT = """You are a quantitative researcher for day trading.
You have access to a strategy library (documented historical edges) and an
internal case library (actual trade results from this system).

Your job: given the current market conditions, determine which strategies are
most likely to have edge TODAY, and flag any divergence between what the
textbooks say and what the internal case data shows.

You are NOT making trade recommendations. You are answering:
  "Which strategy playbook should the Analyst and PM be running today?"

For each strategy you evaluate, assess:
  - Fit: do current conditions match the strategy's ideal conditions?
  - Textbook edge: what does documented history say?
  - Internal confirmation: does our case library agree or disagree?
  - Divergence: if our cases contradict the textbook, why might that be?
  - Recommendation: lean into | use with caution | avoid today

Respond in JSON:
{
  "market_conditions_summary": "brief read on current conditions",
  "strategies": [
    {
      "strategy_key": "gap_and_go",
      "strategy_name": "Gap and Go",
      "fit_score": 8.5,
      "recommendation": "lean_into|use_with_caution|avoid",
      "textbook_win_rate": 0.58,
      "internal_win_rate": 0.71,
      "internal_cases": 14,
      "agreement": "confirmed|diverges|insufficient_data",
      "divergence_note": "if agreement=diverges, explain why",
      "conditions_met": ["risk_on regime", "high premarket volume"],
      "conditions_missing": ["catalyst required"],
      "analyst_guidance": "what the Analyst should weight in signals today",
      "pm_guidance": "what PM should watch for in execution today"
    }
  ],
  "primary_strategy": "the single best strategy for today's conditions",
  "strategies_to_avoid": ["strategy_key"],
  "regime_note": "one line on how regime affects all strategies today"
}

Only include strategies with fit_score >= 5. Rank by fit_score descending.
"""


def get_internal_stats(engine) -> dict:
    """
    Pull internal win rates per setup_type from the case library,
    mapped back to strategy keys.
    """
    win_rates = get_win_rate_by_setup(engine)
    stats = {}
    for r in win_rates:
        setup_type = r["setup_type"]
        strategy_key = SETUP_TYPE_MAP.get(setup_type)
        if strategy_key:
            if strategy_key not in stats:
                stats[strategy_key] = {
                    "total": 0, "wins": 0,
                    "avg_pnl_pct": 0.0, "cases": []
                }
            stats[strategy_key]["total"] += r["total"]
            stats[strategy_key]["wins"] += r["wins"]
            stats[strategy_key]["avg_pnl_pct"] = round(
                (stats[strategy_key]["avg_pnl_pct"] + r["avg_pnl_pct"]) / 2, 2
            )

    # Add win rate
    for key in stats:
        t = stats[key]["total"]
        w = stats[key]["wins"]
        stats[key]["win_rate"] = round(w / t, 3) if t > 0 else None

    return stats


def build_strategy_context(engine, market_regime: str = None) -> str:
    """
    Build a compact strategy context block for injection into agent prompts.
    Shows recommended strategies and their guidance.
    """
    db = get_session(engine)

    # Get latest quant researcher output
    mem = (
        db.query(AgentMemory)
        .filter_by(agent="quant_researcher", key="strategy_recommendations")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    db.close()

    if not mem:
        return "No strategy recommendations available yet."

    data = json.loads(mem.value)

    # Only use if from today or this week
    ts = data.get("timestamp", "")
    if ts < (datetime.utcnow() - timedelta(days=1)).isoformat():
        return "Strategy recommendations are stale — run quant_researcher again."

    lines = [f"Market conditions: {data.get('market_conditions_summary', '')}"]
    lines.append(f"Primary strategy today: {data.get('primary_strategy', '?')}")
    lines.append(f"Regime note: {data.get('regime_note', '')}")
    lines.append("")

    for s in data.get("strategies", []):
        if s.get("recommendation") == "avoid":
            continue
        rec = s.get("recommendation", "?")
        rec_color_char = "✅" if rec == "lean_into" else "⚠️"
        lines.append(
            f"{rec_color_char} {s.get('strategy_name') or s.get('name', '?')} "
            f"(fit: {s.get('fit_score', '?')}/10, "
            f"internal: {int((s.get('internal_win_rate') or 0)*100)}% "
            f"over {s.get('internal_cases', 0)} cases)"
        )
        if s.get("analyst_guidance"):
            lines.append(f"   → Analyst: {s['analyst_guidance']}")
        if s.get("pm_guidance"):
            lines.append(f"   → PM: {s['pm_guidance']}")

    avoid = data.get("strategies_to_avoid", [])
    if avoid:
        lines.append(f"\nAvoid today: {', '.join(avoid)}")

    # Include dynamic strategies across all pipeline stages
    from utils.strategy_store import get_all_strategies, get_pipeline_strategies
    all_strats = get_all_strategies(engine)
    live_dynamic = [k for k, v in all_strats.items() if v.get("source") == "dynamic"]
    if live_dynamic:
        lines.append("\nAgent-proposed strategies (live):")
        for key in live_dynamic:
            s = all_strats[key]
            stage = s.get("pipeline_stage", s.get("status", "?"))
            wr = f"{s['win_rate_documented']:.0f}%" if s.get('win_rate_documented') else "new"
            lines.append(f"  📌 {s['name']} ({key}) — stage: {stage} [{wr}, {s.get('total_trades', 0)} trades]")

    # Show strategies still in pipeline (backtest, paper_trade) for full visibility
    pipeline_strats = get_pipeline_strategies(engine)
    in_pipeline = [s for s in pipeline_strats if s.status in ("backtest", "paper_trade")]
    if in_pipeline:
        stage_labels = {
            "backtest": "🔬 backtesting",
            "paper_trade": "📝 paper trading",
        }
        lines.append("\nAgent-proposed strategies (in pipeline):")
        for s in in_pipeline:
            label = stage_labels.get(s.status, s.status)
            wr = f"{s.win_rate:.0f}%" if s.win_rate else "pending"
            lines.append(f"  {label} {s.name} ({s.key}) — {(s.description or '')[:80]} [{wr}, {s.total_trades or 0} trades]")

    return "\n".join(lines)


def build_pm_strategy_context(engine, market_regime: str = None) -> str:
    """
    Build a filtered strategy context block for PM entry prompts.

    Similar to build_strategy_context() but excludes backtest/paper_trade
    pipeline-stage strategies that are not actionable for PM entry decisions.
    Only includes:
    - Strategies with recommendation "lean_into" or "use_with_caution"
    - Dynamic strategies that are live (source="dynamic", pipeline_stage in live stages)

    Does NOT modify build_strategy_context() — that function remains unchanged
    for all other callers (narrator morning_briefing, quant_researcher, etc.).
    """
    db = get_session(engine)

    # Get latest quant researcher output
    mem = (
        db.query(AgentMemory)
        .filter_by(agent="quant_researcher", key="strategy_recommendations")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    db.close()

    if not mem:
        return "No strategy recommendations available yet."

    data = json.loads(mem.value)

    # Only use if from today or this week
    ts = data.get("timestamp", "")
    if ts < (datetime.utcnow() - timedelta(days=1)).isoformat():
        return "Strategy recommendations are stale — run quant_researcher again."

    lines = [f"Market conditions: {data.get('market_conditions_summary', '')}"]
    lines.append(f"Primary strategy today: {data.get('primary_strategy', '?')}")
    lines.append(f"Regime note: {data.get('regime_note', '')}")
    lines.append("")

    for s in data.get("strategies", []):
        if s.get("recommendation") == "avoid":
            continue
        rec = s.get("recommendation", "?")
        rec_color_char = "✅" if rec == "lean_into" else "⚠️"
        lines.append(
            f"{rec_color_char} {s.get('strategy_name') or s.get('name', '?')} "
            f"(fit: {s.get('fit_score', '?')}/10, "
            f"internal: {int((s.get('internal_win_rate') or 0)*100)}% "
            f"over {s.get('internal_cases', 0)} cases)"
        )
        if s.get("analyst_guidance"):
            lines.append(f"   → Analyst: {s['analyst_guidance']}")
        if s.get("pm_guidance"):
            lines.append(f"   → PM: {s['pm_guidance']}")

    avoid = data.get("strategies_to_avoid", [])
    if avoid:
        lines.append(f"\nAvoid today: {', '.join(avoid)}")

    # Include only live dynamic strategies — exclude backtest/paper_trade pipeline stages
    from utils.strategy_store import get_all_strategies
    all_strats = get_all_strategies(engine)
    live_dynamic = [
        k for k, v in all_strats.items()
        if v.get("source") == "dynamic"
        and v.get("pipeline_stage") not in ("backtest", "paper_trade")
    ]
    if live_dynamic:
        lines.append("\nAgent-proposed strategies (live):")
        for key in live_dynamic:
            s = all_strats[key]
            stage = s.get("pipeline_stage", s.get("status", "?"))
            wr = f"{s['win_rate_documented']:.0f}%" if s.get('win_rate_documented') else "new"
            lines.append(f"  📌 {s['name']} ({key}) — stage: {stage} [{wr}, {s.get('total_trades', 0)} trades]")

    # NOTE: Deliberately omits the "Agent-proposed strategies (in pipeline)" section
    # that shows backtest/paper_trade stage strategies. Those are not actionable
    # for PM entry decisions.

    return "\n".join(lines)


def run(engine, market_regime: str = None, context: dict = None) -> dict:
    """
    Run the Quant Researcher.
    market_regime: risk_on | risk_off | mixed (from Researcher)
    context: any additional market context dict
    """
    db = get_session(engine)

    # Pull latest market context from Researcher
    regime_mem = (
        db.query(AgentMemory)
        .filter_by(agent="researcher", key="market_context")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    market_context_text = regime_mem.value if regime_mem else "No market context available."

    # Pull weekly briefing if available
    weekly_mem = (
        db.query(AgentMemory)
        .filter_by(agent="weekly_prep", key="weekly_briefing")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    weekly_briefing = {}
    if weekly_mem:
        data = json.loads(weekly_mem.value)
        if data.get("week", "") >= (date.today() - timedelta(days=6)).isoformat():
            weekly_briefing = data
            if not market_regime:
                market_regime = data.get("market_regime")

    db.close()

    # Build internal stats from case library
    internal_stats = get_internal_stats(engine)

    # Run backtest for recent edge data (weekly prep only — skip if already fresh)
    backtest_results = {}
    watchlist = [s.strip() for s in __import__('os').getenv("WATCHLIST", "SPY,QQQ,IWM,TSLA,NVDA,AMD").split(",")]
    existing_bt = (
        db.query(AgentMemory)
        .filter_by(agent="quant_researcher", key="backtest_results")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    bt_stale = True
    if existing_bt:
        try:
            bt_data = json.loads(existing_bt.value)
            generated = datetime.fromisoformat(bt_data.get("generated_at", "2000-01-01"))
            bt_stale = (datetime.utcnow() - generated).days >= 3
        except Exception:
            pass

    if bt_stale:
        try:
            log.info("Quant Researcher: running backtest...")
            from backtest import run_backtest
            backtest_results = run_backtest(symbols=watchlist, days=90)
            db2 = get_session(engine)
            db2.add(AgentMemory(
                agent="quant_researcher",
                symbol=None,
                key="backtest_results",
                value=json.dumps(backtest_results),
            ))
            db2.commit()
            db2.close()
            log.info(f"Backtest complete: {backtest_results.get('total_trades', 0)} trades analyzed")
        except Exception as e:
            log.warning(f"Backtest failed (non-fatal): {e}")
    else:
        try:
            backtest_results = json.loads(existing_bt.value)
        except Exception:
            pass

    # Format strategy library for the prompt — compact version to reduce token usage
    strategy_lib = {}
    for key, strat in STRATEGIES.items():
        internal = internal_stats.get(key, {})
        strategy_lib[key] = {
            "name": strat["name"],
            "bias": strat["bias"],
            "textbook_wr": strat["win_rate_documented"],
            "internal_wr": internal.get("win_rate"),
            "internal_cases": internal.get("total", 0),
            "internal_pnl": internal.get("avg_pnl_pct"),
        }

    # Recent successful cases for additional context
    recent_success = query_cases(engine, outcome="success", limit=3)
    recent_fail = query_cases(engine, outcome="failure", limit=3)

    user_prompt = f"""
Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
Current market regime: {market_regime or 'unknown'}

MARKET CONTEXT (from Researcher):
{market_context_text}

WEEKLY BRIEFING:
{json.dumps(weekly_briefing, indent=2) if weekly_briefing else 'Not available'}

STRATEGY LIBRARY (with internal case data):
{json.dumps(strategy_lib, indent=2)}

BACKTEST RESULTS (last 90 days, rule-based simulation):
{json.dumps(backtest_results.get('strategies', {}), indent=2) if backtest_results else 'Not available'}

RECENT SUCCESSFUL CASES:
{format_cases_for_prompt(recent_success)}

RECENT FAILED CASES:
{format_cases_for_prompt(recent_fail)}

Evaluate which strategies have edge in today's conditions.
Cross-reference textbook expectations against our internal case results and backtest data.
Prefer strategies where backtest win_rate >= 50% AND has_edge=true in current regime.
Flag strategies where backtest shows no edge even if textbook says otherwise.

If you identify a recurring pattern in the case data that doesn't fit any existing strategy,
propose it as a new strategy. Include in your response:
"proposed_strategies": [
  {{
    "key": "snake_case_name",
    "name": "Human Readable Name",
    "description": "What this strategy does and when it works",
    "timeframe": "5-15 min",
    "bias": "LONG|SHORT|either",
    "ideal_conditions": {{"market_regime": ["risk_on"]}},
    "failure_conditions": ["condition1", "condition2"],
    "execution_notes": ["note1", "note2"]
  }}
]
Only propose strategies backed by at least 3 similar cases with >50% win rate.
If no new strategies are warranted, return "proposed_strategies": [].
"""

    raw = call_llm(SYSTEM_PROMPT, user_prompt, json_mode=True, tier="medium", purpose="quant_researcher")
    result = parse_json_response(raw)
    result["timestamp"] = datetime.utcnow().isoformat()

    inferred_regime = (
        market_regime
        or result.get("market_regime")
        or result.get("regime")
        or "unknown"
    )
    result.setdefault("market_regime", inferred_regime)

    # Some models return a looser schema (for example top_strategies). Normalize
    # it so downstream reports and agents do not silently show unavailable data.
    if "strategies" not in result and isinstance(result.get("top_strategies"), list):
        result["strategies"] = [
            {
                "strategy_key": s.get("strategy_key") or s.get("name", "").lower().replace(" ", "_"),
                "strategy_name": s.get("strategy_name") or s.get("name"),
                "fit_score": s.get("fit_score"),
                "recommendation": s.get("recommendation") or "use_with_caution",
                "analyst_guidance": s.get("description"),
                "pm_guidance": s.get("required_confirmation"),
            }
            for s in result["top_strategies"]
            if isinstance(s, dict)
        ]

    # Persist to agent memory
    db = get_session(engine)
    db.add(AgentMemory(
        agent="quant_researcher",
        symbol=None,
        key="strategy_recommendations",
        value=json.dumps(result),
    ))
    db.add(AgentMemory(
        agent="quant_researcher",
        symbol=None,
        key="regime",
        value=json.dumps({"regime": inferred_regime, "source": "quant_researcher"}),
    ))
    db.commit()
    db.close()

    # Update dynamic strategy stats and retire underperformers
    from utils.strategy_store import update_strategy_stats, propose_strategy
    try:
        update_strategy_stats(engine)
    except Exception as e:
        log.warning(f"Strategy stats update failed: {e}")

    # Propose new strategies if the LLM suggested any
    for proposed in result.get("proposed_strategies", []):
        if proposed.get("key") and proposed.get("name") and proposed.get("description"):
            try:
                propose_strategy(engine, proposed)
                log.info(f"New strategy proposed: {proposed['key']} — {proposed['name']}")
            except Exception as e:
                log.warning(f"Failed to propose strategy {proposed.get('key')}: {e}")

    return result


def print_report(result: dict):
    """Print a rich strategy report to terminal."""
    console.print()
    console.print(Panel(
        f"[bold white]Quant Researcher — {datetime.utcnow().strftime('%Y-%m-%d')}[/bold white]\n"
        f"[dim]{result.get('market_conditions_summary', '')}[/dim]",
        style="bold blue", box=box.DOUBLE,
    ))

    strategies = result.get("strategies", [])
    if strategies:
        table = Table(
            title="Strategy Fit Today",
            box=box.SIMPLE_HEAVY,
            header_style="bold cyan",
        )
        table.add_column("Strategy", style="bold")
        table.add_column("Fit", justify="right")
        table.add_column("Rec")
        table.add_column("Textbook WR", justify="right")
        table.add_column("Internal WR", justify="right")
        table.add_column("Cases", justify="right")
        table.add_column("Agreement")

        for s in sorted(strategies, key=lambda x: x.get("fit_score", 0), reverse=True):
            rec = s.get("recommendation", "?")
            rec_color = {"lean_into": "green", "use_with_caution": "yellow", "avoid": "red"}.get(rec, "white")
            agree = s.get("agreement", "?")
            agree_color = {"confirmed": "green", "diverges": "red", "insufficient_data": "dim"}.get(agree, "white")
            fit = s.get("fit_score", 0)
            fit_color = "green" if fit >= 7 else "yellow" if fit >= 5 else "red"

            iwr = s.get("internal_win_rate")
            iwr_str = f"{int(iwr*100)}%" if iwr else "[dim]—[/dim]"

            table.add_row(
                s.get("strategy_name", "?"),
                f"[{fit_color}]{fit}[/{fit_color}]",
                f"[{rec_color}]{rec}[/{rec_color}]",
                f"{int(s.get('textbook_win_rate', 0)*100)}%",
                iwr_str,
                str(s.get("internal_cases", 0)),
                f"[{agree_color}]{agree}[/{agree_color}]",
            )

        console.print(table)

    # Primary strategy
    primary = result.get("primary_strategy")
    if primary:
        console.print(f"\n  [bold green]Primary strategy today:[/bold green] {primary}")

    avoid = result.get("strategies_to_avoid", [])
    if avoid:
        console.print(f"  [bold red]Avoid today:[/bold red] {', '.join(avoid)}")

    regime_note = result.get("regime_note", "")
    if regime_note:
        console.print(f"  [dim]{regime_note}[/dim]")

    # Divergence alerts
    for s in strategies:
        if s.get("agreement") == "diverges" and s.get("divergence_note"):
            console.print(Panel(
                f"[bold]Strategy:[/bold] {s['strategy_name']}\n"
                f"[bold]Textbook:[/bold] {int(s.get('textbook_win_rate', 0)*100)}% WR  "
                f"[bold]Internal:[/bold] {int((s.get('internal_win_rate') or 0)*100)}% WR "
                f"over {s.get('internal_cases', 0)} cases\n"
                f"[bold]Why it diverges:[/bold] {s['divergence_note']}",
                title="[bold red]⚠️ Strategy Divergence[/bold red]",
                box=box.ROUNDED,
            ))

    console.print()
