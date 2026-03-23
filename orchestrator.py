"""
Orchestrator
Runs the full agent pipeline on a schedule aligned with market hours.
Uses APScheduler for intraday loops.
"""

import os
import signal
import logging
from datetime import datetime
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

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/orchestrator.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

WATCHLIST = [s.strip() for s in os.getenv("WATCHLIST", "SPY,QQQ,IWM,TSLA,NVDA,AMD").split(",")]
LOOP_INTERVAL = int(os.getenv("LOOP_INTERVAL_MINUTES", 15))


def get_engine():
    return init_db("db/paper_trader.db")


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


def run_pre_market():
    """8:30 AM ET — Research + Analysis prep before open."""
    log.info("=== PRE-MARKET RUN ===")
    engine = get_engine()
    try:
        # Scout runs first to find additional symbols
        console.print("[bold cyan]🔭 Scout scanning for movers...[/bold cyan]")
        scout_result = scout.run(engine, WATCHLIST)
        scout_symbols = scout_result.get("symbols", [])
        if scout_symbols:
            log.info(f"Scout picks: {', '.join(scout_symbols)} (tone: {scout_result.get('market_tone')})")
            for pick in scout_result.get("picks", []):
                log.info(f"  {pick['symbol']}: {pick['catalyst']} [{pick['conviction']} conviction]")
        else:
            log.info("Scout: no picks today")

        # Full watchlist = core + scout picks
        full_watchlist = WATCHLIST + scout_symbols

        console.print("[bold yellow]📰 Researcher running...[/bold yellow]")
        res = researcher.run(engine, full_watchlist)
        log.info(f"Researcher: {res.get('market_context', '')[:100]}")

        console.print("[bold blue]📐 Quant Researcher: matching strategies to conditions...[/bold blue]")
        qr_result = quant_researcher.run(engine, market_regime=res.get("market_regime"))
        quant_researcher.print_report(qr_result)
        log.info(f"Quant Researcher: primary={qr_result.get('primary_strategy')} avoid={qr_result.get('strategies_to_avoid')}")

        console.print("[bold blue]📊 Analyst running...[/bold blue]")
        sigs = analyst.run(engine, full_watchlist)
        for sym, sig in sigs.items():
            log.info(f"  {sym}: {sig.get('signal')} ({sig.get('strength')})")
    except Exception as e:
        log.error(f"Pre-market error: {e}", exc_info=True)


def run_intraday():
    """Every N minutes during market hours."""
    log.info("=== INTRADAY CYCLE ===")
    engine = get_engine()
    try:
        # Check stop losses for all profiles
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

        # Full watchlist = core + today's scout picks
        scout_picks = scout.get_todays_picks(engine)
        full_watchlist = WATCHLIST + scout_picks

        # Refresh analysis
        console.print("[bold blue]📊 Analyst refresh...[/bold blue]")
        analyst.run(engine, full_watchlist)

        # All PM profiles decide
        console.print("[bold green]🧠 Portfolio Managers deciding...[/bold green]")
        all_decisions = pm.run(engine, full_watchlist)
        for profile_id, result in all_decisions.items():
            profile_name = profile_id.capitalize()
            for d in result.get("decisions", []):
                status = "✅" if d.get("executed") else "❌"
                log.info(f"  [{profile_name}] {status} {d['action']} {d.get('quantity', '')} {d['symbol']} @ ${d.get('price', 0):.2f}")

        # Print dashboard
        bookkeeper.print_dashboard(engine)

    except Exception as e:
        log.error(f"Intraday error: {e}", exc_info=True)


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


def run_post_market():
    """4:15 PM ET — End of day wrap-up."""
    log.info("=== POST-MARKET / END OF DAY ===")
    engine = get_engine()
    try:
        # Run reviewer on today's closed trades
        console.print("[bold magenta]🔍 Reviewer scoring trades...[/bold magenta]")
        review = reviewer.run(engine, min_unreviewed=1)
        log.info(f"Reviewer: {review.get('batch_feedback', '')[:200]}")

        # Save daily log
        summary = bookkeeper.end_of_day(engine)
        log.info(f"Day summary: P&L ${summary['daily_pnl']:+,.2f} | {summary['trades']} trades | {summary['wins']}W {summary['losses']}L")

        # Final dashboard
        bookkeeper.print_dashboard(engine)
    except Exception as e:
        log.error(f"Post-market error: {e}", exc_info=True)


def run_once():
    """Run a single full cycle manually (for testing)."""
    engine = get_engine()
    ensure_initial_balance(engine)
    console.print("[bold]Running single cycle...[/bold]")
    run_pre_market()
    run_intraday()


def check_llm_connectivity():
    """Ping LLM providers at startup to catch config issues early."""
    from utils.llm import call_llm
    probe = "Reply with one word: ok"

    # High tier
    try:
        result = call_llm("You are a test.", probe, tier="high")
        console.print(f"   [green]✓ LLM high tier OK[/green] ({os.getenv('LLM_PROVIDER')} / {os.getenv('LLM_MODEL')})")
    except Exception as e:
        console.print(f"   [red]✗ LLM high tier FAILED: {e}[/red]")
        log.error(f"LLM high tier check failed: {e}")

    # Low tier (only if configured separately)
    if os.getenv("LLM_LOW_PROVIDER"):
        try:
            result = call_llm("You are a test.", probe, tier="low")
            console.print(f"   [green]✓ LLM low tier OK[/green] ({os.getenv('LLM_LOW_PROVIDER')} / {os.getenv('LLM_LOW_MODEL')})")
        except Exception as e:
            console.print(f"   [yellow]⚠ LLM low tier FAILED: {e} (will use fallback)[/yellow]")
            log.warning(f"LLM low tier check failed: {e}")


def main():
    engine = get_engine()
    ensure_initial_balance(engine)
    check_llm_connectivity()

    scheduler = BlockingScheduler(timezone="America/New_York")

    # Pre-market: 8:30 AM ET, Mon-Fri
    scheduler.add_job(
        run_pre_market,
        CronTrigger(day_of_week="mon-fri", hour=8, minute=30, timezone="America/New_York"),
        id="pre_market",
    )

    # Intraday loop: every N minutes, 9:30 AM – 4:00 PM ET
    scheduler.add_job(
        run_intraday,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-15",
            minute=f"*/{LOOP_INTERVAL}",
            timezone="America/New_York",
        ),
        id="intraday",
    )

    # Post-market: 4:15 PM ET
    scheduler.add_job(
        run_post_market,
        CronTrigger(day_of_week="mon-fri", hour=16, minute=15, timezone="America/New_York"),
        id="post_market",
    )

    # Sunday weekly prep: 5:00 PM ET
    scheduler.add_job(
        run_weekly_prep,
        CronTrigger(day_of_week="sun", hour=17, minute=0, timezone="America/New_York"),
        id="weekly_prep",
    )

    console.print(f"[bold green]🚀 Paper Trader started[/bold green]")
    console.print(f"   Watchlist: {', '.join(WATCHLIST)}")
    console.print(f"   Loop interval: {LOOP_INTERVAL} min")
    console.print(f"   Schedule: Sun 5PM weekly prep | 8:30 pre-market | 9:30-4:00 intraday | 4:15 EOD")

    def _shutdown(signum, frame):
        console.print("\n[yellow]Shutting down...[/yellow]")
        scheduler.shutdown(wait=False)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

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
    else:
        main()
