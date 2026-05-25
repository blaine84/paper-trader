"""
Scout Agent
Scans for high-potential stocks to add to the daily watchlist.
Runs pre-market, surfaces movers, news catalysts, and unusual activity.
Writes daily watchlist additions to AgentMemory.
Reviewer scores Scout picks at EOD to improve future scanning.

Extended with Sector Scout pipeline: multi-sector deterministic screening
followed by Chief Scout LLM curation for Expanded Watchlist discovery.
"""

import os
import json
import logging
import signal
import time
from datetime import datetime, timedelta
from functools import partial

from utils.finnhub_client import FinnhubClient
from utils.llm import call_llm, parse_json_response
from utils.symbol_class import classify_symbol
from db.schema import AgentMemory, get_session
from utils.case_library import (
    query_cases,
    get_win_rate_by_setup,
    format_cases_for_prompt,
    get_selection_feedback,
)

# Sector Scout pipeline imports
from utils.sector_scout import load_sector_scout_config, run_sector_screeners
from utils.sector_scout_chief import run_chief_scout, chief_scout_fallback
from utils.expanded_watchlist import update_expanded_watchlist, get_expanded_watchlist
from utils.sector_scout_persistence import persist_run_summary, persist_candidate_rows
from utils.scout_logging import emit_scout_event
from utils.sector_scout_models import CooldownState

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a stock scout for day trading. Your job is to find the best
trading opportunities for today beyond the core watchlist.

You receive:
- Market movers (top gainers/losers by % change)
- Recent high-volume stocks
- Breaking market news
- Your past performance scoring Scout picks

Criteria for a good day-trade candidate:
- High relative volume (unusual activity)
- Clear catalyst (news, earnings, sector move)
- Enough liquidity (not a micro-cap penny stock)
- Price between $5 and $1000
- Volatile enough to be worth trading

Respond in JSON:
{
  "picks": [
    {
      "symbol": "TSLA",
      "price": 250.00,
      "catalyst": "what's driving it today",
      "direction_bias": "bullish|bearish|neutral",
      "conviction": "low|medium|high",
      "reason": "why this is worth watching today",
      "risk": "main risk to the thesis"
    }
  ],
  "max_picks": 3,
  "market_tone": "risk-on|risk-off|mixed",
  "scout_notes": "overall observations about today's tape"
}

Return 1-3 picks max. Quality over quantity. If nothing stands out, return empty picks.
Do NOT duplicate symbols already in the core watchlist.
"""


# ---------------------------------------------------------------------------
# Pipeline Timeout Support
# ---------------------------------------------------------------------------


class PipelineTimeoutError(Exception):
    """Raised when the sector scout pipeline exceeds its timeout."""
    pass


def _timeout_handler(signum, frame):
    """Signal handler for pipeline timeout (Unix only)."""
    raise PipelineTimeoutError("Sector scout pipeline timeout reached")


# ---------------------------------------------------------------------------
# Market Movers (existing logic)
# ---------------------------------------------------------------------------


def get_market_movers(fh: FinnhubClient, scout_candidates: list[str] | None = None) -> dict:
    """
    Finnhub doesn't have a direct movers endpoint on free tier,
    so we approximate using stock symbols from major indices
    and check their quotes for big movers.
    """
    # Use provided candidate pool or fall back to default
    candidates = scout_candidates or [
        s.strip() for s in os.getenv(
            "SCOUT_CANDIDATES",
            "AAPL,MSFT,META,AMZN,GOOGL,AVGO,SMCI,PLTR,COIN,MSTR,ARM,MU,INTC,NFLX"
        ).split(",")
    ]

    movers = []
    for sym in candidates:
        try:
            q = fh.get_quote(sym)
            if abs(q.get("change_pct", 0)) >= 3.0:  # 3%+ move
                movers.append(q)
        except Exception:
            continue

    # Sort by absolute % change
    movers.sort(key=lambda x: abs(x.get("change_pct", 0)), reverse=True)
    return movers[:15]  # top 15 movers


# ---------------------------------------------------------------------------
# Main Scout Run (enhanced with Sector Scout pipeline)
# ---------------------------------------------------------------------------


def run(engine, core_watchlist: list[str], scout_candidates: list[str] | None = None) -> dict:
    fh = FinnhubClient()
    db = get_session(engine)

    # Get market movers
    movers = get_market_movers(fh, scout_candidates=scout_candidates)

    # Get market news for context
    market_news = fh.get_market_news("general")

    # Pull selection feedback from Reviewer (Scout + Analyst channel)
    selection_fb = get_selection_feedback(engine, limit=10)

    # Pull structured case win rates — what setup types have worked?
    win_rates = get_win_rate_by_setup(engine)
    win_rate_text = "\n".join(
        f"  {r['setup_type']}: {r['win_rate']}% win rate over {r['total']} cases (avg pnl: {r['avg_pnl_pct']}%)"
        for r in win_rates
    ) if win_rates else "No case history yet."

    # Pull successful cases for top setup types to show what works
    successful_cases = query_cases(engine, outcome="success", limit=5)
    successful_text = format_cases_for_prompt(successful_cases)

    # Filter out core watchlist symbols from movers
    movers_filtered = [m for m in movers if m["symbol"] not in core_watchlist]

    # Pull weekly prep context if available (written Sunday)
    weekly_ctx = (
        db.query(AgentMemory)
        .filter_by(agent="weekly_prep", key="weekly_watchlist")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    weekly_watchlist_text = ""
    if weekly_ctx:
        data = json.loads(weekly_ctx.value)
        # Only use if from this week
        from datetime import date
        if data.get("week") >= (date.today() - timedelta(days=2)).isoformat():
            weekly_watchlist_text = f"\nWEEKLY WATCHLIST THESIS (from Sunday prep):\n{json.dumps(data.get('watchlist', []), indent=2)}"

    user_prompt = f"""
Today: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
Core watchlist (already covered): {', '.join(core_watchlist)}{weekly_watchlist_text}

TOP MOVERS (>3% change, excluding core watchlist):
{json.dumps(movers_filtered, indent=2)}

MARKET NEWS:
{json.dumps(market_news[:8], indent=2)}

SELECTION FEEDBACK FROM REVIEWER:
{selection_fb}

SETUP WIN RATES FROM CASE LIBRARY:
{win_rate_text}

RECENT SUCCESSFUL CASES (what's worked before):
{successful_text}

Use the case library to bias toward setup types with proven win rates.
Find the best 1-3 additional stocks worth day trading today.
"""

    raw = call_llm(SYSTEM_PROMPT, user_prompt, json_mode=True, tier="low", purpose="scout_daily")
    result = parse_json_response(raw)

    picks = result.get("picks", [])

    # Attach deterministic symbol_class to each pick (Req 2.1, 3.3)
    for pick in picks:
        pick["symbol_class"] = classify_symbol(pick["symbol"])

    symbols = [p["symbol"] for p in picks]

    # Save picks to agent memory
    if picks:
        mem = AgentMemory(
            agent="scout",
            symbol=None,
            key="daily_picks",
            value=json.dumps({
                "date": datetime.utcnow().strftime("%Y-%m-%d"),
                "picks": picks,
                "market_tone": result.get("market_tone"),
                "notes": result.get("scout_notes"),
            }),
        )
        db.add(mem)

    # Save each pick individually (for per-symbol tracking by Reviewer)
    for pick in picks:
        mem = AgentMemory(
            agent="scout",
            symbol=pick["symbol"],
            key="pick",
            value=json.dumps(pick),
        )
        db.add(mem)

    db.commit()
    db.close()

    # --- Sector Scout Pipeline Integration ---
    sector_scout_result = _run_sector_scout_pipeline(
        engine, core_watchlist, fh, run_type="premarket"
    )

    return {
        "symbols": symbols,
        "picks": picks,
        "market_tone": result.get("market_tone"),
        "scout_notes": result.get("scout_notes"),
        "sector_scout": sector_scout_result,
    }


# ---------------------------------------------------------------------------
# Sector Scout Pipeline (internal)
# ---------------------------------------------------------------------------


def _run_sector_scout_pipeline(
    engine,
    core_watchlist: list[str],
    fh: FinnhubClient,
    run_type: str = "premarket",
) -> dict:
    """Execute the sector scout pipeline with timeout enforcement.

    Loads config, runs sector screeners, calls Chief Scout LLM (with
    deterministic fallback on failure), and updates the Expanded Watchlist.

    Enforces pipeline_timeout_seconds (default 90s). Returns partial results
    if timeout is reached.

    Args:
        engine: SQLAlchemy engine.
        core_watchlist: Current Core_Watchlist symbols.
        fh: FinnhubClient instance.
        run_type: "premarket" | "confirmation" | "midday"

    Returns:
        Dict with pipeline results including picks and expanded watchlist.
    """
    start_time = time.time()
    partial_result = {
        "run_type": run_type,
        "picks": [],
        "expanded_watchlist": [],
        "fallback_used": False,
        "timed_out": False,
        "error": None,
    }

    try:
        # 1. Load sector scout config
        config = load_sector_scout_config()

        if not config.get("enabled", True):
            logger.info("Sector Scout pipeline disabled via config")
            return partial_result

        # Get pipeline timeout
        budget_ceilings = config.get("budget_ceilings", {})
        pipeline_timeout = int(budget_ceilings.get("pipeline_timeout_seconds", 90))

        # 2. Run sector screeners (deterministic)
        screener_result = _run_with_timeout(
            lambda: run_sector_screeners(config, core_watchlist, fh),
            timeout_seconds=pipeline_timeout,
            start_time=start_time,
        )

        if screener_result is None:
            # Timeout reached during screening
            partial_result["timed_out"] = True
            logger.warning(
                "Sector Scout pipeline timed out during screening (timeout=%ds)",
                pipeline_timeout,
            )
            return partial_result

        finalists_by_sector = screener_result.get("finalists_by_sector", {})

        # Check remaining time before Chief Scout
        elapsed = time.time() - start_time
        if elapsed >= pipeline_timeout:
            partial_result["timed_out"] = True
            logger.warning(
                "Sector Scout pipeline timed out before Chief Scout (elapsed=%.1fs, timeout=%ds)",
                elapsed,
                pipeline_timeout,
            )
            return partial_result

        # 3. Call Chief Scout LLM for curation
        chief_result = run_chief_scout(
            finalists_by_sector, core_watchlist, config, engine
        )

        picks = chief_result.get("picks", [])
        llm_error = chief_result.get("llm_error")

        # 4. If LLM failed, use deterministic fallback
        if llm_error:
            logger.warning("Chief Scout LLM error: %s — using fallback", llm_error)
            current_watchlist = get_expanded_watchlist(engine)
            fallback_result = chief_scout_fallback(
                finalists_by_sector, config, len(current_watchlist)
            )
            picks = fallback_result.get("picks", [])
            partial_result["fallback_used"] = True

        # 5. Update Expanded Watchlist
        expanded_symbols = update_expanded_watchlist(engine, picks, run_type, config)

        partial_result["picks"] = picks
        partial_result["expanded_watchlist"] = expanded_symbols

        # 6. Persist run summary and candidate rows to AgentMemory
        duration = time.time() - start_time
        try:
            persist_run_summary(engine, screener_result, chief_result, run_type, duration)
            persist_candidate_rows(engine, finalists_by_sector, run_type)
        except Exception as persist_exc:
            # Persistence failure should not break the pipeline
            logger.warning(
                "Failed to persist scout run data: %s", persist_exc
            )

        # Log scout run event
        emit_scout_event("SCOUT_RUN", {
            "run_type": run_type,
            "sectors_scanned": screener_result.get("sectors_scanned", 0),
            "total_candidates": sum(
                len(c) for c in screener_result.get("candidates_by_sector", {}).values()
            ) + len(screener_result.get("rejections", [])),
            "finalists_count": len(screener_result.get("global_finalists", [])),
            "picks_count": len(picks),
        })

        return partial_result

    except PipelineTimeoutError:
        partial_result["timed_out"] = True
        logger.warning("Sector Scout pipeline timed out (run_type=%s)", run_type)
        return partial_result
    except FileNotFoundError as exc:
        partial_result["error"] = str(exc)
        logger.error("Sector Scout config error: %s", exc)
        return partial_result
    except ValueError as exc:
        partial_result["error"] = str(exc)
        logger.error("Sector Scout config validation error: %s", exc)
        return partial_result
    except Exception as exc:
        partial_result["error"] = f"{type(exc).__name__}: {exc}"
        logger.error("Sector Scout pipeline error: %s", exc, exc_info=True)
        return partial_result


def _run_with_timeout(func, timeout_seconds: int, start_time: float):
    """Run a function with remaining timeout enforcement.

    Uses wall-clock time checking rather than signals for cross-platform
    compatibility (Windows does not support SIGALRM).

    Returns None if timeout is exceeded before the function completes.
    Otherwise returns the function's result.
    """
    elapsed = time.time() - start_time
    remaining = timeout_seconds - elapsed

    if remaining <= 0:
        return None

    # Execute the function — we rely on the pipeline's internal iteration
    # to check elapsed time. For truly blocking calls, we accept that the
    # timeout is best-effort (checked between major pipeline stages).
    result = func()

    # Check if we exceeded timeout after execution
    if time.time() - start_time > timeout_seconds:
        # Still return the result since we have it — partial is better than nothing
        logger.warning(
            "Pipeline stage completed but exceeded timeout (elapsed=%.1fs, limit=%ds)",
            time.time() - start_time,
            timeout_seconds,
        )

    return result


# ---------------------------------------------------------------------------
# get_todays_picks (enhanced with Expanded Watchlist)
# ---------------------------------------------------------------------------


def get_todays_picks(engine) -> list[str]:
    """Return today's Scout pick symbols from memory, including Expanded Watchlist.

    Combines the original daily LLM picks with Expanded Watchlist symbols
    from the sector scout pipeline. Deduplicates the result.
    """
    db = get_session(engine)
    today = datetime.utcnow().strftime("%Y-%m-%d")

    picks_mem = (
        db.query(AgentMemory)
        .filter_by(agent="scout", key="daily_picks")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    db.close()

    core_picks: list[str] = []
    if picks_mem:
        data = json.loads(picks_mem.value)
        if data.get("date") == today:
            core_picks = [p["symbol"] for p in data.get("picks", [])]

    # Include Expanded Watchlist symbols from sector scout pipeline
    expanded_symbols = get_expanded_watchlist(engine)

    # Combine and deduplicate while preserving order
    seen: set[str] = set()
    combined: list[str] = []
    for sym in core_picks + expanded_symbols:
        if sym not in seen:
            seen.add(sym)
            combined.append(sym)

    return combined


# ---------------------------------------------------------------------------
# Intraday Scan with Reanalysis Cooldown
# ---------------------------------------------------------------------------


def run_intraday_scan(
    engine,
    core_watchlist: list[str],
    run_type: str = "confirmation",
) -> dict:
    """Execute an intraday sector scout scan with reanalysis cooldown logic.

    Runs the sector scout pipeline for intraday windows (10:00 ET confirmation,
    12:30 ET midday). Applies cooldown logic to skip re-analysis of symbols
    already evaluated today unless significant changes occurred.

    Cooldown skip conditions (ANY triggers re-analysis):
    - scout_score changed by >= score_change_threshold (default 15 pts)
    - A new catalyst headline appeared that wasn't in prior analysis
    - move_pct changed by >= move_pct_change_threshold (default 2 ppts)

    Args:
        engine: SQLAlchemy engine.
        core_watchlist: Current Core_Watchlist symbols.
        run_type: "confirmation" or "midday"

    Returns:
        Dict with pipeline results.
    """
    fh = FinnhubClient()
    start_time = time.time()

    try:
        config = load_sector_scout_config()

        if not config.get("enabled", True):
            logger.info("Sector Scout pipeline disabled via config")
            return {
                "run_type": run_type,
                "picks": [],
                "expanded_watchlist": [],
                "skipped_symbols": [],
                "reanalyzed_symbols": [],
                "fallback_used": False,
                "timed_out": False,
                "error": None,
            }

        budget_ceilings = config.get("budget_ceilings", {})
        pipeline_timeout = int(budget_ceilings.get("pipeline_timeout_seconds", 90))

        # Load cooldown config
        cooldown_cfg = config.get("reanalysis_cooldown", {})
        score_change_threshold = float(cooldown_cfg.get("score_change_threshold", 15.0))
        move_pct_change_threshold = float(cooldown_cfg.get("move_pct_change_threshold", 2.0))

        # Load prior cooldown states from today's earlier runs
        prior_states = _load_cooldown_states(engine)

        # Run sector screeners
        screener_result = _run_with_timeout(
            lambda: run_sector_screeners(config, core_watchlist, fh),
            timeout_seconds=pipeline_timeout,
            start_time=start_time,
        )

        if screener_result is None:
            logger.warning("Intraday scan timed out during screening")
            return {
                "run_type": run_type,
                "picks": [],
                "expanded_watchlist": get_expanded_watchlist(engine),
                "skipped_symbols": [],
                "reanalyzed_symbols": [],
                "fallback_used": False,
                "timed_out": True,
                "error": None,
            }

        # Apply cooldown logic to finalists
        finalists_by_sector = screener_result.get("finalists_by_sector", {})
        filtered_finalists: dict[str, list] = {}
        skipped_symbols: list[str] = []
        reanalyzed_symbols: list[str] = []

        for sector_key, finalists in finalists_by_sector.items():
            sector_filtered = []
            for candidate in finalists:
                symbol = candidate.symbol if hasattr(candidate, "symbol") else candidate.get("symbol", "")
                prior = prior_states.get(symbol)

                if prior is None:
                    # Never analyzed today — include
                    sector_filtered.append(candidate)
                    reanalyzed_symbols.append(symbol)
                else:
                    # Check cooldown conditions
                    should_reanalyze = _should_reanalyze(
                        candidate, prior,
                        score_change_threshold,
                        move_pct_change_threshold,
                    )
                    if should_reanalyze:
                        sector_filtered.append(candidate)
                        reanalyzed_symbols.append(symbol)
                    else:
                        skipped_symbols.append(symbol)

            filtered_finalists[sector_key] = sector_filtered

        # Check remaining time
        elapsed = time.time() - start_time
        if elapsed >= pipeline_timeout:
            logger.warning("Intraday scan timed out before Chief Scout")
            return {
                "run_type": run_type,
                "picks": [],
                "expanded_watchlist": get_expanded_watchlist(engine),
                "skipped_symbols": skipped_symbols,
                "reanalyzed_symbols": reanalyzed_symbols,
                "fallback_used": False,
                "timed_out": True,
                "error": None,
            }

        # Call Chief Scout on filtered finalists
        chief_result = run_chief_scout(
            filtered_finalists, core_watchlist, config, engine
        )

        picks = chief_result.get("picks", [])
        llm_error = chief_result.get("llm_error")
        fallback_used = False

        if llm_error:
            logger.warning("Chief Scout LLM error in intraday scan: %s", llm_error)
            current_watchlist = get_expanded_watchlist(engine)
            fallback_result = chief_scout_fallback(
                filtered_finalists, config, len(current_watchlist)
            )
            picks = fallback_result.get("picks", [])
            fallback_used = True

        # Update Expanded Watchlist
        expanded_symbols = update_expanded_watchlist(engine, picks, run_type, config)

        # Save cooldown states for symbols analyzed this run
        _save_cooldown_states(engine, screener_result, run_type)

        emit_scout_event("SCOUT_RUN", {
            "run_type": run_type,
            "sectors_scanned": screener_result.get("sectors_scanned", 0),
            "total_candidates": sum(
                len(c) for c in screener_result.get("candidates_by_sector", {}).values()
            ) + len(screener_result.get("rejections", [])),
            "finalists_count": len(reanalyzed_symbols),
            "picks_count": len(picks),
        })

        return {
            "run_type": run_type,
            "picks": picks,
            "expanded_watchlist": expanded_symbols,
            "skipped_symbols": skipped_symbols,
            "reanalyzed_symbols": reanalyzed_symbols,
            "fallback_used": fallback_used,
            "timed_out": False,
            "error": None,
        }

    except PipelineTimeoutError:
        return {
            "run_type": run_type,
            "picks": [],
            "expanded_watchlist": get_expanded_watchlist(engine),
            "skipped_symbols": [],
            "reanalyzed_symbols": [],
            "fallback_used": False,
            "timed_out": True,
            "error": None,
        }
    except Exception as exc:
        logger.error("Intraday scan error: %s", exc, exc_info=True)
        return {
            "run_type": run_type,
            "picks": [],
            "expanded_watchlist": get_expanded_watchlist(engine),
            "skipped_symbols": [],
            "reanalyzed_symbols": [],
            "fallback_used": False,
            "timed_out": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# Cooldown State Helpers
# ---------------------------------------------------------------------------


def _should_reanalyze(
    candidate,
    prior: CooldownState,
    score_change_threshold: float,
    move_pct_change_threshold: float,
) -> bool:
    """Determine if a previously-analyzed symbol should be re-analyzed.

    Returns True if ANY of the following conditions are met:
    1. scout_score changed by >= score_change_threshold points
    2. A new catalyst headline appeared (not in prior analysis)
    3. move_pct changed by >= move_pct_change_threshold percentage points
    """
    # Get current values from candidate
    if hasattr(candidate, "scout_score"):
        current_score = candidate.scout_score
        current_move_pct = candidate.move_pct
        current_headlines = candidate.news_headlines
    else:
        current_score = candidate.get("scout_score", 0.0)
        current_move_pct = candidate.get("move_pct")
        current_headlines = candidate.get("news_headlines")

    # Condition 1: Score changed significantly
    last_score = prior.get("last_scout_score", 0.0)
    if abs(current_score - last_score) >= score_change_threshold:
        return True

    # Condition 2: New catalyst headline appeared
    last_headline = prior.get("last_news_headline")
    if current_headlines and isinstance(current_headlines, list) and len(current_headlines) > 0:
        newest_headline = current_headlines[0].get("headline", "") if isinstance(current_headlines[0], dict) else str(current_headlines[0])
        if newest_headline and newest_headline != last_headline:
            return True

    # Condition 3: move_pct changed significantly
    if current_move_pct is not None:
        last_move_pct = prior.get("last_move_pct", 0.0)
        if abs(current_move_pct - last_move_pct) >= move_pct_change_threshold:
            return True

    return False


def _load_cooldown_states(engine) -> dict[str, CooldownState]:
    """Load today's cooldown states from AgentMemory.

    Returns a dict mapping symbol -> CooldownState for all symbols
    analyzed in earlier runs today.
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")
    memory_key = f"cooldown_states:{today}"

    db = get_session(engine)
    try:
        record = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout", key=memory_key)
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )

        if not record:
            return {}

        try:
            data = json.loads(record.value)
            return data.get("states", {})
        except (json.JSONDecodeError, TypeError):
            return {}
    finally:
        db.close()


def _save_cooldown_states(engine, screener_result: dict, run_type: str) -> None:
    """Save cooldown states for symbols analyzed in this run.

    Merges with existing states from earlier runs today.
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")
    memory_key = f"cooldown_states:{today}"
    now_iso = datetime.utcnow().isoformat()

    # Load existing states
    existing_states = _load_cooldown_states(engine)

    # Build new states from this run's finalists
    finalists_by_sector = screener_result.get("finalists_by_sector", {})
    for sector_key, finalists in finalists_by_sector.items():
        for candidate in finalists:
            if hasattr(candidate, "symbol"):
                symbol = candidate.symbol
                score = candidate.scout_score
                move_pct = candidate.move_pct or 0.0
                headlines = candidate.news_headlines
            else:
                symbol = candidate.get("symbol", "")
                score = candidate.get("scout_score", 0.0)
                move_pct = candidate.get("move_pct") or 0.0
                headlines = candidate.get("news_headlines")

            # Get newest headline
            newest_headline = None
            if headlines and isinstance(headlines, list) and len(headlines) > 0:
                first = headlines[0]
                if isinstance(first, dict):
                    newest_headline = first.get("headline")
                else:
                    newest_headline = str(first)

            existing_states[symbol] = {
                "symbol": symbol,
                "last_scout_score": score,
                "last_move_pct": move_pct,
                "last_news_headline": newest_headline,
                "last_analyzed_at": now_iso,
            }

    # Persist merged states
    db = get_session(engine)
    try:
        record = (
            db.query(AgentMemory)
            .filter_by(agent="sector_scout", key=memory_key)
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )

        value = json.dumps({
            "date": today,
            "last_run_type": run_type,
            "last_updated": now_iso,
            "states": existing_states,
        })

        if record:
            record.value = value
            record.timestamp = datetime.utcnow()
        else:
            db.add(AgentMemory(
                agent="sector_scout",
                symbol=None,
                key=memory_key,
                value=value,
            ))

        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
