"""
Scout Agent
Scans for high-potential stocks to add to the daily watchlist.
Runs pre-market, surfaces movers, news catalysts, and unusual activity.
Writes daily watchlist additions to AgentMemory.
Reviewer scores Scout picks at EOD to improve future scanning.
"""

import os
import json
from datetime import datetime, timedelta
from utils.finnhub_client import FinnhubClient
from utils.llm import call_llm, parse_json_response
from db.schema import AgentMemory, get_session
from utils.case_library import query_cases, get_win_rate_by_setup, format_cases_for_prompt, get_selection_feedback


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


def get_market_movers(fh: FinnhubClient) -> dict:
    """
    Finnhub doesn't have a direct movers endpoint on free tier,
    so we approximate using stock symbols from major indices
    and check their quotes for big movers.
    """
    # Sample a broad set of liquid names to screen
    candidates = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NFLX", "BABA",
        "BA", "DIS", "JPM", "GS", "MS", "WFC", "BAC",
        "XOM", "CVX", "OXY", "SLB",
        "UBER", "LYFT", "SNAP", "PINS", "RBLX",
        "PLTR", "COIN", "HOOD", "SOFI",
        "MRNA", "PFE", "JNJ", "ABBV",
        "F", "GM", "RIVN", "LCID",
        "X", "CLF", "FCX",
        "SQ", "PYPL", "SHOP",
        "CRWD", "PANW", "ZS", "OKTA",
        "SNOW", "DDOG", "MDB", "NET",
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


def run(engine, core_watchlist: list[str]) -> dict:
    fh = FinnhubClient()
    db = get_session(engine)

    # Get market movers
    movers = get_market_movers(fh)

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

    raw = call_llm(SYSTEM_PROMPT, user_prompt, json_mode=True)
    result = parse_json_response(raw)

    picks = result.get("picks", [])
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

    return {
        "symbols": symbols,
        "picks": picks,
        "market_tone": result.get("market_tone"),
        "scout_notes": result.get("scout_notes"),
    }


def get_todays_picks(engine) -> list[str]:
    """Return today's Scout pick symbols from memory."""
    db = get_session(engine)
    today = datetime.utcnow().strftime("%Y-%m-%d")

    picks_mem = (
        db.query(AgentMemory)
        .filter_by(agent="scout", key="daily_picks")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    db.close()

    if not picks_mem:
        return []

    data = json.loads(picks_mem.value)
    if data.get("date") != today:
        return []

    return [p["symbol"] for p in data.get("picks", [])]
