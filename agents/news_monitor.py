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


def fetch_and_store_news(engine, symbols: list, source_tag: str) -> dict:
    """
    Lightweight news fetch for a list of symbols.
    Fetches raw news via Finnhub, stores in AgentMemory as breaking_news.
    Does NOT run LLM analysis — the analyst interprets on its next run.

    News is APPENDED to the current market day's alerts by querying existing
    records and merging, rather than writing a new row that displaces prior data.

    source_tag is required (no default) to force callers to be explicit about
    the origin: "price_spike", "position_poll", or "scheduled".

    Returns {symbol: [news_items]}.
    """
    from utils.catalyst_freshness import get_market_day_start, ET, MAX_ALERTS_PER_SYMBOL

    fh = FinnhubClient()
    db = get_session(engine)
    results = {}
    for sym in symbols:
        try:
            news = fh.get_news(sym, days=1)
            results[sym] = news or []
        except Exception as e:
            log.error(f"News fetch failed for {sym} ({source_tag}): {e}")
            results[sym] = []

    # Build alerts in the same format as the existing news monitor
    # MAX_ALERTS_PER_SYMBOL is configurable in catalyst_freshness.py
    new_alerts = []
    for sym, news_items in results.items():
        for item in news_items[:MAX_ALERTS_PER_SYMBOL]:
            new_alerts.append({
                "symbol": sym,
                "headline": item.get("headline", ""),
                "impact": "unknown",  # no LLM classification
                "urgency": "medium",
                "summary": (item.get("summary", "") or "")[:200],
                "source_tag": source_tag,
            })

    if new_alerts:
        # Merge with existing alerts from today rather than clobbering.
        # Query the most recent N records for the current market day and
        # union their alerts with the new ones, deduplicating by headline.
        now_et = datetime.now(ET)
        market_day_start = get_market_day_start(now_et)
        existing_alerts = []
        existing_rows = (
            db.query(AgentMemory)
            .filter_by(agent="news_monitor", key="breaking_news")
            .filter(AgentMemory.timestamp >= market_day_start)
            .order_by(AgentMemory.timestamp.desc())
            .limit(10)
            .all()
        )
        seen_headlines = set()
        for row in existing_rows:
            try:
                data = json.loads(row.value)
                for alert in data.get("alerts", []):
                    hl = alert.get("headline", "")
                    if hl and hl not in seen_headlines:
                        existing_alerts.append(alert)
                        seen_headlines.add(hl)
            except Exception:
                pass

        # Add new alerts, skipping duplicates
        for alert in new_alerts:
            hl = alert.get("headline", "")
            if hl and hl not in seen_headlines:
                existing_alerts.append(alert)
                seen_headlines.add(hl)

        db.add(AgentMemory(
            agent="news_monitor",
            symbol=None,
            key="breaking_news",
            value=json.dumps({"alerts": existing_alerts, "market_update": f"{source_tag} check"}),
        ))
        db.commit()

    db.close()
    return results


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

        # Query held positions to detect catalyst shocks for narrator flash updates
        from db.schema import Position
        held_symbols = set()
        try:
            held_symbols = {p.symbol for p in db.query(Position).all()}
        except Exception:
            pass

        db.close()

        for alert in result["alerts"]:
            log.info(f"📰 BREAKING: {alert['symbol']} [{alert['impact']}] {alert['headline']}")

            # Trigger narrator flash update for high-urgency breaking news on held positions
            sym = alert.get("symbol", "")
            headline = alert.get("headline", "")
            if alert.get("urgency") == "high" and sym in held_symbols:
                try:
                    import agents.narrator as narrator
                    narrator.run(engine, "flash_update", event_context={
                        "trigger": "catalyst_shock",
                        "symbol": sym,
                        "details": f"Breaking: {headline}",
                    })
                except Exception:
                    pass  # never block news monitoring

    return result
