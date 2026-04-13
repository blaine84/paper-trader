"""
News Monitor
Mid-day lightweight news check using local LLM.
Catches breaking catalysts that the morning researcher missed.
Runs every 2 hours during market hours.
"""

import json
import logging
import os
from datetime import datetime
from utils.finnhub_client import FinnhubClient
from utils.llm import call_llm, parse_json_response
from db.schema import AgentMemory, get_session

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a breaking news monitor for day trading.
You receive recent headlines for watched symbols.
Identify any NEW material catalysts that could move prices significantly.
Ignore routine news, minor analyst notes, or already-priced-in events.

Respond in JSON:
{
  "alerts": [
    {
      "symbol": "TSLA",
      "headline": "key headline",
      "impact": "bullish|bearish|neutral",
      "urgency": "high|medium|low",
      "summary": "one sentence on why this matters for today's trading"
    }
  ],
  "market_update": "one sentence on any broad market shifts since morning"
}

If nothing material, return {"alerts": [], "market_update": "no significant changes"}.
"""


def run(engine) -> dict:
    """Check for breaking news on watchlist symbols."""
    fh = FinnhubClient()
    watchlist = [s.strip() for s in os.getenv("WATCHLIST", "SPY,QQQ,IWM,TSLA,NVDA,AMD").split(",")]

    # Get today's scout picks too
    db = get_session(engine)
    scout_mem = (
        db.query(AgentMemory)
        .filter_by(agent="scout", key="daily_picks")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    scout_symbols = []
    if scout_mem:
        try:
            data = json.loads(scout_mem.value)
            scout_symbols = [p["symbol"] for p in data.get("picks", [])]
        except Exception:
            pass
    db.close()

    all_symbols = list(set(watchlist + scout_symbols))

    # Fetch recent news
    all_news = {}
    for sym in all_symbols:
        try:
            news = fh.get_news(sym, days=1)
            if news:
                all_news[sym] = [{"headline": n["headline"], "source": n["source"]} for n in news[:3]]
        except Exception:
            pass

    # Also get market-wide news
    try:
        market_news = fh.get_market_news()
        all_news["MARKET"] = [{"headline": n["headline"], "source": n["source"]} for n in market_news[:5]]
    except Exception:
        pass

    if not all_news:
        return {"alerts": [], "market_update": "no news available"}

    user_prompt = f"""
Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

RECENT HEADLINES:
{json.dumps(all_news, indent=2)}

Flag any breaking catalysts that could affect today's trading.
"""

    try:
        raw = call_llm(SYSTEM_PROMPT, user_prompt, json_mode=True, tier="low")
        result = parse_json_response(raw)
    except Exception as e:
        log.error(f"News monitor LLM error: {e}")
        return {"alerts": [], "market_update": "error"}

    # Store alerts in agent memory for PM to read
    if result.get("alerts"):
        db = get_session(engine)
        db.add(AgentMemory(
            agent="news_monitor",
            symbol=None,
            key="breaking_news",
            value=json.dumps(result),
        ))
        db.commit()
        db.close()
        for alert in result["alerts"]:
            log.info(f"📰 BREAKING: {alert['symbol']} [{alert['impact']}] {alert['headline']}")

    return result
