"""
Researcher Agent
Gathers news, sentiment, and market context for the watchlist.
Writes findings to AgentMemory for other agents to read.
"""

import os
import json
from datetime import datetime
from utils.finnhub_client import FinnhubClient
from utils.llm import call_llm, parse_json_response
from db.schema import AgentMemory, get_session


SYSTEM_PROMPT = """You are a financial research agent for day trading.
Your job is to analyze news and market context for a list of stocks and ETFs.

For each symbol, assess:
- Sentiment: bullish, bearish, or neutral
- Key catalysts or risks today
- Any major news that could move price
- Confidence: low, medium, high

Respond in JSON format:
{
  "market_context": "brief overall market narrative",
  "market_regime": "risk_on|risk_off|mixed|unknown",
  "symbols": {
    "SPY": {
      "sentiment": "bullish|bearish|neutral",
      "confidence": "low|medium|high",
      "catalysts": ["..."],
      "risks": ["..."],
      "summary": "1-2 sentence summary"
    }
  }
}
"""


def run(engine, symbols: list[str]) -> dict:
    fh = FinnhubClient()
    db = get_session(engine)

    # Gather market news + per-symbol news
    market_news = fh.get_market_news("general")
    symbol_news = {}
    quotes = {}

    for sym in symbols:
        symbol_news[sym] = fh.get_news(sym, days=1)
        quotes[sym] = fh.get_quote(sym)

    # Build prompt
    user_prompt = f"""
Today is {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}.
Watchlist: {', '.join(symbols)}

MARKET NEWS:
{json.dumps(market_news, indent=2)}

PER-SYMBOL NEWS:
{json.dumps(symbol_news, indent=2)}

CURRENT QUOTES:
{json.dumps(quotes, indent=2)}

Analyze the above and return your research JSON.
"""

    raw = call_llm(SYSTEM_PROMPT, user_prompt, json_mode=True, tier="low")
    result = parse_json_response(raw)
    result["market_context"] = result.get("market_context", "")
    result["market_regime"] = result.get("market_regime", "unknown")
    for sym, data in result.get("symbols", {}).items():
        mem = AgentMemory(
            agent="researcher",
            symbol=sym,
            key="sentiment",
            value=json.dumps(data),
        )
        db.add(mem)

    market_mem = AgentMemory(
        agent="researcher",
        symbol=None,
        key="market_context",
        value=result.get("market_context", ""),
    )
    db.add(market_mem)
    db.commit()
    db.close()

    return result
