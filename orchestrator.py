"""
Orchestrator
Runs the full agent pipeline on a schedule aligned with market hours.
Uses APScheduler for intraday loops.
"""

import os
import json
import signal
import logging
import logging.handlers
from datetime import datetime, timedelta
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from rich.console import Console

load_dotenv()

from db.schema import init_db, Balance
from sqlalchemy.orm import sessionmaker
import agents.researcher as researcher
import agents.analyst as analyst
import agents.portfolio_manager as pm
import agents.bookkeeper as bookkeeper
import agents.reviewer as reviewer
import agents.scout as scout
import agents.weekly_prep as weekly_prep
import agents.quant_researcher as quant_researcher
import agents.daily_review as daily_review
import agents.ceo as ceo

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler(
            "logs/orchestrator.log", maxBytes=10_000_000, backupCount=5
        ),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

WATCHLIST = [s.strip() for s in os.getenv("WATCHLIST", "SPY,QQQ,IWM,TSLA,NVDA,AMD").split(",")]
LOOP_INTERVAL = int(os.getenv("LOOP_INTERVAL_MINUTES", 15))


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = init_db("db/paper_trader.db")
    return _engine


def ensure_initial_balance(engine):
    """Seed starting cash balance for all profiles if DB is fresh."""
    from models.pm_profiles import PM_PROFILES, ACTIVE_PROFILES
    Session = sessionmaker(bind=engine)
    db = Session()
    for profile_id in ACTIVE_PROFILES:
        count = db.query(Balance).filter_by(profile=profile_id).count()
        if count == 0:
            starting = float(PM_PROFILES[profile_id]["starting_balance"])
            db.add(Balance(cash=starting, profile=profile_id))
            log.info(f"Seeded {profile_id} balance: ${starting:,.2f}")
    db.commit()
    db.close()


def check_schema(engine):
    """Verify the DB schema has all expected columns. Fail fast if not."""
    import sqlite3
    from sqlalchemy import inspect as sa_inspect

    inspector = sa_inspect(engine)

    # Expected columns per table that have been added over time.
    # If a column is missing, the system will crash on first query anyway —
    # better to catch it here with a clear message and auto-fix.
    expected = {
        "trades": ["thesis", "setup_type", "invalidators"],
    }

    missing = {}
    for table, columns in expected.items():
        if not inspector.has_table(table):
            continue
        existing = {col["name"] for col in inspector.get_columns(table)}
        table_missing = [c for c in columns if c not in existing]
        if table_missing:
            missing[table] = table_missing

    if not missing:
        return

    # Auto-fix: add missing columns
    col_types = {
        "thesis": "TEXT",
        "setup_type": "VARCHAR(64)",
        "invalidators": "TEXT",
    }

    raw_conn = engine.raw_connection()
    cursor = raw_conn.cursor()
    for table, cols in missing.items():
        for col in cols:
            col_type = col_types.get(col, "TEXT")
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
            log.warning(f"Schema migration: added {table}.{col} ({col_type})")
    raw_conn.commit()
    raw_conn.close()

    console.print(f"   [yellow]⚠ Auto-migrated missing columns: {missing}[/yellow]")


def run_pre_market():
    """8:30 AM ET — Research + Analysis prep before open."""
    log.info("=== PRE-MARKET RUN ===")
    engine = get_engine()

    # Scout — find additional symbols
    scout_symbols = []
    try:
        console.print("[bold cyan]🔭 Scout scanning for movers...[/bold cyan]")
        scout_result = scout.run(engine, WATCHLIST)
        scout_symbols = scout_result.get("symbols", [])
        if scout_symbols:
            log.info(f"Scout picks: {', '.join(scout_symbols)} (tone: {scout_result.get('market_tone')})")
        else:
            log.info("Scout: no picks today")
    except Exception as e:
        log.error(f"Scout error: {e}", exc_info=True)

    full_watchlist = WATCHLIST + scout_symbols

    # Researcher
    market_regime = None
    try:
        console.print("[bold yellow]📰 Researcher running...[/bold yellow]")
        res = researcher.run(engine, full_watchlist)
        market_regime = res.get("market_regime")
        log.info(f"Researcher: {res.get('market_context', '')[:100]}")
    except Exception as e:
        log.error(f"Researcher error: {e}", exc_info=True)

    # Quant Researcher
    try:
        console.print("[bold blue]📐 Quant Researcher: matching strategies to conditions...[/bold blue]")
        qr_result = quant_researcher.run(engine, market_regime=market_regime)
        quant_researcher.print_report(qr_result)
        log.info(f"Quant Researcher: primary={qr_result.get('primary_strategy')} avoid={qr_result.get('strategies_to_avoid')}")
    except Exception as e:
        log.error(f"Quant Researcher error: {e}", exc_info=True)

    # Pipeline evaluation — evaluate all dynamic strategies in pipeline stages
    try:
        console.print("[bold magenta]🔄 Pipeline: evaluating strategy stages...[/bold magenta]")
        pipeline_results = run_pipeline_evaluation()
        for r in pipeline_results:
            log.info(f"Pipeline: {r['strategy_key']} @ {r['stage']} → {r['decision']} ({r['reason']})")
        if not pipeline_results:
            log.info("Pipeline: no strategies in pipeline stages")
    except Exception as e:
        log.error(f"Pipeline evaluation error: {e}", exc_info=True)

    # Analyst
    try:
        console.print("[bold blue]📊 Analyst running...[/bold blue]")
        sigs = analyst.run(engine, full_watchlist)
        for sym, sig in sigs.items():
            log.info(f"  {sym}: {sig.get('signal')} ({sig.get('strength')})")
    except Exception as e:
        log.error(f"Analyst error: {e}", exc_info=True)

    # Slack morning report — non-blocking, failures never affect trading
    try:
        from utils.slack_notifier import SlackNotifier
        notifier = SlackNotifier()
        if notifier.is_enabled():
            notifier.send_morning_report(engine)
    except Exception as e:
        log.error(f"Slack morning report error: {e}", exc_info=True)

    # Narrator morning briefing - non-blocking, failures never affect trading
    try:
        import agents.narrator as narrator
        console.print("[bold cyan]📝 Narrator: morning briefing...[/bold cyan]")
        narrator.run(engine, "morning_briefing")
    except Exception as e:
        log.error(f"Narrator morning briefing error: {e}", exc_info=True)


def run_pipeline_evaluation():
    """Pre-market: evaluate all strategies in pipeline stages.

    1. Trigger backtests for new proposals (status=backtest, no report yet).
    2. Run the full pipeline evaluation for all strategies in pipeline stages.
    """
    from deployment_pipeline import (
        run_pipeline_evaluation as evaluate,
        evaluate_backtest_gate,
        apply_gate_result,
    )
    from strategy_backtester import StrategyBacktester
    from db.schema import get_session, DynamicStrategy

    engine = get_engine()

    # --- Phase 1: trigger backtests for pending proposals ---
    db = get_session(engine)
    try:
        pending = (
            db.query(DynamicStrategy)
            .filter(
                DynamicStrategy.status == "backtest",
                (DynamicStrategy.backtest_report_id == None) | (DynamicStrategy.backtest_report_id == ""),  # noqa: E711
            )
            .all()
        )
        # Detach so we can close the session before long-running backtests
        for s in pending:
            db.expunge(s)
    finally:
        db.close()

    for strategy in pending:
        try:
            log.info(f"Pipeline: triggering backtest for '{strategy.key}'")
            backtester = StrategyBacktester(engine)
            report = backtester.run(strategy)

            # Update the strategy's backtest_report_id
            memory_key = f"backtest_report_{strategy.key}"
            db = get_session(engine)
            try:
                strat = db.query(DynamicStrategy).filter_by(id=strategy.id).first()
                if strat:
                    strat.backtest_report_id = memory_key
                    db.commit()
            finally:
                db.close()

            # Evaluate the backtest gate and apply the result
            gate_result = evaluate_backtest_gate(report)
            apply_gate_result(engine, strategy, gate_result)

            log.info(
                f"Pipeline: backtest for '{strategy.key}' → {gate_result.decision} ({gate_result.reason})"
            )
        except Exception as e:
            log.error(f"Pipeline: backtest failed for '{strategy.key}': {e}", exc_info=True)

    # --- Phase 2: evaluate all strategies in pipeline stages ---
    results = evaluate(engine)
    for r in results:
        log.info(f"Pipeline: {r['strategy_key']} @ {r['stage']} → {r['decision']} ({r['reason']})")
    return results


def run_analyst_refresh():
    """Analyst-only refresh — runs every 15 min, free via local LLM."""
    log.info("=== ANALYST REFRESH ===")
    engine = get_engine()

    scout_picks = []
    try:
        scout_picks = scout.get_todays_picks(engine)
    except Exception as e:
        log.error(f"Scout picks error: {e}", exc_info=True)
    full_watchlist = WATCHLIST + scout_picks

    try:
        console.print("[bold blue]📊 Analyst refresh...[/bold blue]")
        analyst.run(engine, full_watchlist)
    except Exception as e:
        log.error(f"Analyst error: {e}", exc_info=True)


def run_intraday():
    """PM decisions + stop checks — runs on the split schedule."""
    log.info("=== INTRADAY CYCLE ===")
    engine = get_engine()

    # Check stop losses
    try:
        stops = bookkeeper.check_stop_losses(engine)
        if stops:
            log.warning(f"Stop losses triggered: {[(s['symbol'], s['profile']) for s in stops]}")
            from db.schema import get_session
            from agents.portfolio_manager import execute_trade
            for stop in stops:
                db = get_session(engine)
                execute_trade(db, {
                    "symbol": stop["symbol"],
                    "action": "CLOSE",
                    "quantity": 0,
                    "price": stop["price"],
                    "rationale": f"Stop loss hit at {stop['stop_loss']}",
                }, stop["profile"])
                db.close()
    except Exception as e:
        log.error(f"Stop loss check error: {e}", exc_info=True)

    # Full watchlist
    scout_picks = []
    try:
        scout_picks = scout.get_todays_picks(engine)
    except Exception as e:
        log.error(f"Scout picks error: {e}", exc_info=True)
    full_watchlist = WATCHLIST + scout_picks

    # Analyst refresh (also runs here so PMs have fresh signals)
    try:
        console.print("[bold blue]📊 Analyst refresh...[/bold blue]")
        analyst.run(engine, full_watchlist)
    except Exception as e:
        log.error(f"Analyst error: {e}", exc_info=True)

    # PM profiles decide — each independently, in parallel
    console.print("[bold green]🧠 Portfolio Managers deciding...[/bold green]")
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _run_pm(profile_id):
        try:
            eng = get_engine()
            result = pm.run_profile(eng, full_watchlist, profile_id)
            for d in result.get("decisions", []):
                status = "✅" if d.get("executed") else "❌"
                log.info(f"  [{profile_id}] {status} {d['action']} {d.get('quantity', '')} {d['symbol']} @ ${d.get('price', 0):.2f}")
        except Exception as e:
            log.error(f"PM {profile_id} error: {e}", exc_info=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_run_pm, pid) for pid in pm.ACTIVE_PROFILES]
        for f in as_completed(futures):
            pass

    # Dashboard
    try:
        bookkeeper.print_dashboard(engine)
    except Exception as e:
        log.error(f"Dashboard error: {e}", exc_info=True)


def run_weekly_prep():
    """Sunday 5:00 PM ET — Weekly prep, sets Monday context."""
    log.info("=== SUNDAY WEEKLY PREP ===")
    engine = get_engine()
    try:
        result = weekly_prep.run(engine, WATCHLIST)
        log.info(f"Weekly prep complete. Regime: {result['briefing'].get('market_regime')}")
        for profile_id, stance in result["stances"].items():
            log.info(f"  {profile_id}: {stance.get('weekly_stance')} | {stance.get('stance_reason', '')}")
    except Exception as e:
        log.error(f"Weekly prep error: {e}", exc_info=True)

    # Meta Reviewer — runs after weekly prep
    try:
        import agents.meta_reviewer as meta_reviewer
        console.print("[bold magenta]🔬 Meta Reviewer analyzing system performance...[/bold magenta]")
        meta_result = meta_reviewer.run(engine)
        log.info(f"Meta Reviewer: {meta_result.get('overall_assessment', '')[:200]}")
    except Exception as e:
        log.error(f"Meta Reviewer error: {e}", exc_info=True)


def run_post_market():
    """4:15 PM ET — End of day wrap-up."""
    log.info("=== POST-MARKET / END OF DAY ===")
    engine = get_engine()

    # Reviewer runs independently — failure here doesn't block EOD
    try:
        console.print("[bold magenta]🔍 Reviewer scoring trades...[/bold magenta]")
        review = reviewer.run(engine, min_unreviewed=1)
        log.info(f"Reviewer: {review.get('batch_feedback', review.get('message', ''))[:200]}")
    except Exception as e:
        log.error(f"Reviewer error: {e}", exc_info=True)

    # EOD bookkeeping always runs
    try:
        summary = bookkeeper.end_of_day(engine)
        log.info(f"Day summary: P&L ${summary['daily_pnl']:+,.2f} | {summary['trades']} trades | {summary['wins']}W {summary['losses']}L")
        bookkeeper.print_dashboard(engine)
    except Exception as e:
        log.error(f"EOD bookkeeper error: {e}", exc_info=True)


def run_daily_review():
    """4:30 PM ET — Daily review journal generation."""
    log.info("=== DAILY REVIEW ===")
    engine = get_engine()
    try:
        console.print("[bold cyan]📓 Daily Review generating journal...[/bold cyan]")
        result = daily_review.run(engine)
        log.info(f"Daily Review: {result.get('date', 'unknown')} — confidence: {result.get('completeness', {}).get('confidence', 'unknown')}")
    except Exception as e:
        log.error(f"Daily Review error: {e}", exc_info=True)

    # Slack afternoon report — non-blocking, failures never affect trading
    try:
        from utils.slack_notifier import SlackNotifier
        notifier = SlackNotifier()
        if notifier.is_enabled():
            notifier.send_afternoon_report(engine)
    except Exception as e:
        log.error(f"Slack afternoon report error: {e}", exc_info=True)


def run_ceo_daily():
    """4:45 PM ET — CEO daily operating memo."""
    log.info("=== CEO DAILY MEMO ===")
    engine = get_engine()
    try:
        console.print("[bold magenta]🧭 CEO: daily operating memo...[/bold magenta]")
        memo = ceo.run(engine, period="daily")
        log.info(f"CEO daily: constraint={memo.get('biggest_constraint', '')[:200]}")
    except Exception as e:
        log.error(f"CEO daily memo error: {e}", exc_info=True)


def run_ceo_weekly():
    """Friday 4:50 PM ET — CEO weekly strategy memo."""
    log.info("=== CEO WEEKLY STRATEGY MEMO ===")
    engine = get_engine()
    try:
        console.print("[bold magenta]🧭 CEO: weekly strategy memo...[/bold magenta]")
        memo = ceo.run(engine, period="weekly")
        log.info(f"CEO weekly: constraint={memo.get('biggest_constraint', '')[:200]}")
    except Exception as e:
        log.error(f"CEO weekly memo error: {e}", exc_info=True)


def run_once():
    """Run a single full cycle manually (for testing)."""
    engine = get_engine()
    ensure_initial_balance(engine)
    check_schema(engine)
    console.print("[bold]Running single cycle...[/bold]")
    run_pre_market()
    run_intraday()


def check_llm_connectivity():
    """Ping LLM providers at startup to catch config issues early."""
    from utils.llm import call_llm
    probe = "Reply with one word: ok"

    # High tier
    try:
        result = call_llm("You are a test.", probe, tier="high", purpose="startup_probe:high")
        console.print(f"   [green]✓ LLM high tier OK[/green] ({os.getenv('LLM_PROVIDER')} / {os.getenv('LLM_MODEL')})")
    except Exception as e:
        console.print(f"   [red]✗ LLM high tier FAILED: {e}[/red]")
        log.error(f"LLM high tier check failed: {e}")

    # Low tier (only if configured separately)
    if os.getenv("LLM_LOW_PROVIDER"):
        try:
            result = call_llm("You are a test.", probe, tier="low", purpose="startup_probe:low")
            console.print(f"   [green]✓ LLM low tier OK[/green] ({os.getenv('LLM_LOW_PROVIDER')} / {os.getenv('LLM_LOW_MODEL')})")
        except Exception as e:
            console.print(f"   [yellow]⚠ LLM low tier FAILED: {e} (will use fallback)[/yellow]")
            log.warning(f"LLM low tier check failed: {e}")


def run_price_monitor():
    """Every 60 seconds — check prices against stops/targets/key levels."""
    # Only run during market hours (9:30 AM - 4:00 PM ET)
    from pytz import timezone
    et = datetime.now(timezone("America/New_York"))
    if et.weekday() >= 5:  # weekend
        return
    market_open = et.replace(hour=9, minute=30, second=0)
    market_close = et.replace(hour=16, minute=0, second=0)
    if not (market_open <= et <= market_close):
        return

    engine = get_engine()
    try:
        import agents.price_monitor as price_monitor
        result = price_monitor.run(engine)

        # Persist live alerts for the web UI
        all_live_alerts = []
        for t in result.get("entry_triggers", []):
            all_live_alerts.append({
                "type": t.get("type", "entry"),
                "symbol": t.get("symbol"),
                "price": t.get("price"),
                "detail": f"{t.get('signal', '')} broke {t.get('level_name', '')}={t.get('level', '')}",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })
        for a in result.get("momentum_alerts", []):
            if a["type"] == "rapid_move":
                all_live_alerts.append({
                    "type": "rapid_move",
                    "symbol": a["symbol"],
                    "price": a["price"],
                    "detail": f"{a['direction']} {a['change_pct']}% in {a['window_minutes']}min",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                })
            elif a["type"] == "approaching_level":
                all_live_alerts.append({
                    "type": "approaching",
                    "symbol": a["symbol"],
                    "price": a["price"],
                    "detail": f"within {a['distance_pct']}% of {a['level_name']}={a['level_value']}",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                })
        for t in result.get("stop_triggers", []):
            all_live_alerts.append({
                "type": t.get("type", "stop"),
                "symbol": t.get("symbol"),
                "price": t.get("price"),
                "detail": f"{t['type']} at {t.get('level', '')} ({t.get('profile', '')})",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })
        if all_live_alerts:
            from db.schema import get_session as _gs, AgentMemory
            _db = _gs(engine)
            _db.add(AgentMemory(
                agent="price_monitor",
                symbol=None,
                key="live_alerts",
                value=json.dumps(all_live_alerts),
            ))
            _db.commit()
            _db.close()

        # Execute stop losses immediately
        for trigger in result.get("stop_triggers", []):
            try:
                from db.schema import get_session
                from agents.portfolio_manager import execute_trade
                db = get_session(engine)
                execute_trade(db, {
                    "symbol": trigger["symbol"],
                    "action": "CLOSE",
                    "quantity": 0,
                    "price": trigger["price"],
                    "rationale": f"Price monitor: {trigger['type']} at {trigger['price']} (level: {trigger['level']})",
                }, trigger["profile"])
                db.close()
                log.warning(f"Price monitor closed {trigger['symbol']} ({trigger['profile']}): {trigger['type']}")
            except Exception as e:
                log.error(f"Price monitor close error: {e}")

        # For entry triggers or rapid moves, filter with local LLM then queue PM
        entry_triggers = result.get("entry_triggers", [])
        momentum_alerts = [a for a in result.get("momentum_alerts", []) if a["type"] == "rapid_move"]
        all_alerts = entry_triggers + momentum_alerts

        if all_alerts:
            from agents.price_monitor import filter_alert_with_llm
            actionable_symbols = []
            for alert in all_alerts:
                try:
                    assessment = filter_alert_with_llm(alert, engine)
                    if assessment.get("actionable"):
                        actionable_symbols.append(alert["symbol"])
                        log.info(f"  Alert ACTIONABLE: {alert['symbol']} — {assessment.get('reasoning', '')}")
                    else:
                        log.info(f"  Alert filtered out: {alert['symbol']} — {assessment.get('reasoning', '')}")
                except Exception as e:
                    actionable_symbols.append(alert["symbol"])  # pass through on error

            action_symbols = list(set(actionable_symbols))
            if action_symbols:
                log.info(f"Price monitor: triggering PM (local LLM) for {action_symbols}")
                import threading
                def _run_pm_async(syms):
                    eng = get_engine()
                    for pid in pm.ACTIVE_PROFILES:
                        try:
                            pm_result = pm.run_profile(eng, WATCHLIST + syms, pid, tier="medium")
                            for d in pm_result.get("decisions", []):
                                if d.get("executed"):
                                    log.info(f"  ⚡ [{pid}] {d['action']} {d.get('quantity','')} {d['symbol']} @ ${d.get('price',0):.2f}")
                        except Exception as e:
                            log.error(f"Price monitor PM {pid} error: {e}")
                threading.Thread(target=_run_pm_async, args=(action_symbols,), daemon=True).start()

    except Exception as e:
        log.error(f"Price monitor error: {e}", exc_info=True)


def run_news_monitor():
    """Every 2 hours — check for breaking news catalysts."""
    log.info("=== NEWS MONITOR ===")
    engine = get_engine()
    try:
        import agents.news_monitor as news_monitor
        result = news_monitor.run(engine)
        if result.get("alerts"):
            log.info(f"News monitor: {len(result['alerts'])} alerts found")
        else:
            log.info(f"News monitor: {result.get('market_update', 'no updates')}")
    except Exception as e:
        log.error(f"News monitor error: {e}", exc_info=True)


def run_position_health():
    """Every hour — review open position health."""
    log.info("=== POSITION HEALTH CHECK ===")
    engine = get_engine()
    try:
        import agents.position_health as position_health
        result = position_health.run(engine)
        log.info(f"Position health: {result.get('summary', '')}")
    except Exception as e:
        log.error(f"Position health error: {e}", exc_info=True)


def run_price_spike_news_check():
    """Every ~15 min — check for price spikes and fetch news for spiking symbols."""
    from pytz import timezone
    et = datetime.now(timezone("America/New_York"))
    if et.weekday() >= 5:  # weekend
        return
    market_open = et.replace(hour=9, minute=30, second=0)
    market_close = et.replace(hour=16, minute=0, second=0)
    if not (market_open <= et <= market_close):
        return

    engine = get_engine()
    from agents.price_monitor import get_batch_quotes, get_price_history
    from utils.catalyst_freshness import PRICE_SPIKE_THRESHOLD_PCT, PRICE_SPIKE_WINDOW_MINUTES

    quotes = get_batch_quotes(WATCHLIST)

    spiking = []
    for sym, price in quotes.items():
        history = get_price_history(sym)
        if len(history) < 2:
            continue
        # Find price from ~PRICE_SPIKE_WINDOW_MINUTES ago
        cutoff = datetime.utcnow() - timedelta(minutes=PRICE_SPIKE_WINDOW_MINUTES)
        old_prices = [(t, p) for t, p in history if t <= cutoff]
        if not old_prices:
            continue
        old_price = old_prices[-1][1]
        change_pct = abs((price - old_price) / old_price) * 100
        if change_pct >= PRICE_SPIKE_THRESHOLD_PCT:
            spiking.append(sym)
            log.info(f"📰 Price spike detected: {sym} moved {change_pct:.1f}% in {PRICE_SPIKE_WINDOW_MINUTES}min — fetching news")

    if spiking:
        from agents.news_monitor import fetch_and_store_news
        fetch_and_store_news(engine, spiking, source_tag="price_spike")


def run_position_news_poll():
    """Every ~30 min — fetch news for symbols with open positions."""
    from pytz import timezone
    et = datetime.now(timezone("America/New_York"))
    if et.weekday() >= 5:  # weekend
        return
    market_open = et.replace(hour=9, minute=30, second=0)
    market_close = et.replace(hour=16, minute=0, second=0)
    if not (market_open <= et <= market_close):
        return

    engine = get_engine()
    from db.schema import get_session, Position
    db = get_session(engine)
    positions = db.query(Position).all()
    db.close()

    held_symbols = list(set(p.symbol for p in positions))
    if not held_symbols:
        return

    log.info(f"📰 Position news poll for: {', '.join(held_symbols)}")
    from agents.news_monitor import fetch_and_store_news
    fetch_and_store_news(engine, held_symbols, source_tag="position_poll")


def run_position_timer():
    """Every 5 minutes — check position hold times and enforce exits."""
    engine = get_engine()
    try:
        import agents.position_timer as position_timer
        result = position_timer.run(engine)
        if result.get("force_closes"):
            log.warning(f"Position timer: {len(result['force_closes'])} force closes")
        if result.get("hard_wall_closes"):
            log.warning(f"Position timer: {len(result['hard_wall_closes'])} hard wall closes")
    except Exception as e:
        log.error(f"Position timer error: {e}", exc_info=True)


def run_narrator(update_type: str):
    """Generic narrator runner for cron-triggered update types."""
    log.info(f"=== NARRATOR: {update_type} ===")
    engine = get_engine()
    try:
        import agents.narrator as narrator
        console.print(f"[bold cyan]📝 Narrator: {update_type}...[/bold cyan]")
        result = narrator.run(engine, update_type)
        if result.get("skipped"):
            log.info(f"Narrator {update_type}: skipped (already exists)")
        else:
            log.info(f"Narrator {update_type}: generated")
    except Exception as e:
        log.error(f"Narrator {update_type} error: {e}", exc_info=True)


def main():
    engine = get_engine()
    ensure_initial_balance(engine)
    check_schema(engine)
    check_llm_connectivity()

    scheduler = BlockingScheduler(timezone="America/New_York")

    # Pre-market: 8:30 AM ET, Mon-Fri
    scheduler.add_job(
        run_pre_market,
        CronTrigger(day_of_week="mon-fri", hour=8, minute=30, timezone="America/New_York"),
        id="pre_market",
    )

    # Analyst refresh: every 15 min all day (free via local LLM)
    scheduler.add_job(
        run_analyst_refresh,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-15",
            minute=f"*/{LOOP_INTERVAL}",
            timezone="America/New_York",
        ),
        id="analyst_refresh",
    )

    # Intraday morning (PM decisions): every 15 min, 9:30 AM – 12:00 PM ET
    scheduler.add_job(
        run_intraday,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-11",
            minute=f"*/{LOOP_INTERVAL}",
            timezone="America/New_York",
        ),
        id="intraday_morning",
    )

    # Intraday afternoon: every 30 min, 12:00 PM – 4:00 PM ET
    scheduler.add_job(
        run_intraday,
        CronTrigger(
            day_of_week="mon-fri",
            hour="12-15",
            minute="0,30",
            timezone="America/New_York",
        ),
        id="intraday_afternoon",
    )

    # Price monitor: every 60 seconds during market hours only (uses yfinance, free)
    from apscheduler.triggers.cron import CronTrigger as CT
    scheduler.add_job(
        run_price_monitor,
        CT(day_of_week="mon-fri", hour="9-15", second="0", timezone="America/New_York"),
        id="price_monitor",
        max_instances=1,
        coalesce=True,
    )

    # News monitor: every 2 hours during market hours (local LLM, free)
    scheduler.add_job(
        run_news_monitor,
        CronTrigger(day_of_week="mon-fri", hour="10,12,14", minute=0, timezone="America/New_York"),
        id="news_monitor",
    )

    # Position health check: every hour during market hours (local LLM, free)
    scheduler.add_job(
        run_position_health,
        CronTrigger(day_of_week="mon-fri", hour="10-15", minute=30, timezone="America/New_York"),
        id="position_health",
    )

    # Position timer: every 5 minutes during market hours (no LLM, pure math)
    scheduler.add_job(
        run_position_timer,
        CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/5", timezone="America/New_York"),
        id="position_timer",
        max_instances=1,
        coalesce=True,
    )

    # Price-spike news check: every 15 min during market hours
    scheduler.add_job(
        run_price_spike_news_check,
        CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/15", timezone="America/New_York"),
        id="price_spike_news",
        max_instances=1,
        coalesce=True,
    )

    # Position-based news poll: every 30 min during market hours
    scheduler.add_job(
        run_position_news_poll,
        CronTrigger(day_of_week="mon-fri", hour="9-15", minute="0,30", timezone="America/New_York"),
        id="position_news_poll",
        max_instances=1,
        coalesce=True,
    )

    # Reviewer queue: every 15 min during market hours, process pending reviews
    def run_reviewer_queue():
        engine = get_engine()
        try:
            review = reviewer.run(engine, min_unreviewed=1)
            msg = review.get("batch_feedback", review.get("message", ""))
            if msg:
                log.info(f"Reviewer queue: {str(msg)[:100]}")

            # Check for stale pending reviews (>24 hours)
            from db.schema import get_session, ReviewQueue
            from datetime import timedelta as td24
            db = get_session(engine)
            stale_cutoff = datetime.utcnow() - td24(hours=24)
            stale = db.query(ReviewQueue).filter_by(status="pending").filter(
                ReviewQueue.queued_at < stale_cutoff
            ).count()
            if stale > 0:
                log.critical(f"🚨 STALE REVIEWS: {stale} trades pending review for >24 hours!")
            db.close()
        except Exception as e:
            log.error(f"Reviewer queue error: {e}", exc_info=True)

    scheduler.add_job(
        run_reviewer_queue,
        CronTrigger(day_of_week="mon-fri", hour="10-16", minute="*/15", timezone="America/New_York"),
        id="reviewer_queue",
        max_instances=1,
        coalesce=True,
    )

    # Post-market: 4:15 PM ET
    scheduler.add_job(
        run_post_market,
        CronTrigger(day_of_week="mon-fri", hour=16, minute=15, timezone="America/New_York"),
        id="post_market",
    )

    # Daily Review: 4:30 PM ET (after Reviewer + Bookkeeper at 4:15)
    scheduler.add_job(
        run_daily_review,
        CronTrigger(day_of_week="mon-fri", hour=16, minute=30, timezone="America/New_York"),
        id="daily_review",
    )

    # CEO daily memo: 4:45 PM ET, Mon-Thu. Friday gets the deeper weekly memo.
    scheduler.add_job(
        run_ceo_daily,
        CronTrigger(day_of_week="mon-thu", hour=16, minute=45, timezone="America/New_York"),
        id="ceo_daily",
        max_instances=1,
        coalesce=True,
    )

    # CEO weekly strategy memo: Friday after close and daily review.
    scheduler.add_job(
        run_ceo_weekly,
        CronTrigger(day_of_week="fri", hour=16, minute=50, timezone="America/New_York"),
        id="ceo_weekly",
        max_instances=1,
        coalesce=True,
    )

    # Sunday weekly prep: 5:00 PM ET
    scheduler.add_job(
        run_weekly_prep,
        CronTrigger(day_of_week="sun", hour=17, minute=0, timezone="America/New_York"),
        id="weekly_prep",
    )

    # --- Narrator cron jobs ---

    # Hourly recaps: 10 AM, 11 AM, 12 PM ET, Mon-Fri
    scheduler.add_job(
        lambda: run_narrator("hourly_recap"),
        CronTrigger(day_of_week="mon-fri", hour="10,11,12", minute=0,
                    timezone="America/New_York"),
        id="narrator_hourly",
    )

    # Afternoon recap: 2 PM ET, Mon-Fri
    scheduler.add_job(
        lambda: run_narrator("afternoon_recap"),
        CronTrigger(day_of_week="mon-fri", hour=14, minute=0,
                    timezone="America/New_York"),
        id="narrator_afternoon",
    )

    # Daily wrap: 4:15 PM ET, Mon-Thu
    scheduler.add_job(
        lambda: run_narrator("daily_wrap"),
        CronTrigger(day_of_week="mon-thu", hour=16, minute=15,
                    timezone="America/New_York"),
        id="narrator_daily_wrap",
    )

    # Weekly wrap: 4:15 PM ET, Friday
    scheduler.add_job(
        lambda: run_narrator("weekly_wrap"),
        CronTrigger(day_of_week="fri", hour=16, minute=15,
                    timezone="America/New_York"),
        id="narrator_weekly_wrap",
    )

    # Sunday prep: 5:15 PM ET, Sunday
    scheduler.add_job(
        lambda: run_narrator("sunday_prep"),
        CronTrigger(day_of_week="sun", hour=17, minute=15,
                    timezone="America/New_York"),
        id="narrator_sunday_prep",
    )

    console.print(f"[bold green]🚀 Paper Trader started[/bold green]")
    console.print(f"   Watchlist: {', '.join(WATCHLIST)}")
    console.print(f"   Loop interval: {LOOP_INTERVAL} min")
    console.print(f"   Schedule: Sun 5PM weekly prep | 8:30 pre-market | 9:30-12 every {LOOP_INTERVAL}min | 12-4 every 30min | 4:15 EOD | 4:30 daily review")

    def _shutdown(signum, frame):
        console.print("\n[yellow]Shutting down...[/yellow]")
        scheduler.shutdown(wait=False)

    def _refresh(signum, frame):
        log.info("SIGUSR1 received — triggering manual refresh")
        console.print("[bold cyan]⚡ Manual refresh triggered...[/bold cyan]")
        scheduler.add_job(run_pre_market, id="manual_refresh", replace_existing=True)

    def _intraday_now(signum, frame):
        log.info("SIGUSR2 received — triggering manual intraday cycle")
        console.print("[bold cyan]⚡ Manual intraday cycle triggered...[/bold cyan]")
        scheduler.add_job(run_intraday, id="manual_intraday", replace_existing=True)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _refresh)
        signal.signal(signal.SIGUSR2, _intraday_now)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        console.print("\n[yellow]Shutting down...[/yellow]")
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "once":
        run_once()
    elif len(sys.argv) > 1 and sys.argv[1] == "weekly":
        engine = get_engine()
        ensure_initial_balance(engine)
        run_weekly_prep()
    elif len(sys.argv) > 1 and sys.argv[1] == "ceo-daily":
        engine = get_engine()
        ensure_initial_balance(engine)
        check_schema(engine)
        run_ceo_daily()
    elif len(sys.argv) > 1 and sys.argv[1] == "ceo-weekly":
        engine = get_engine()
        ensure_initial_balance(engine)
        check_schema(engine)
        run_ceo_weekly()
    else:
        main()
