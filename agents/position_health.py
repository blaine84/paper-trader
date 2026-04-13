"""
Position Health Check
Hourly review of all open positions using local LLM.
Flags positions that are deteriorating even if stops haven't hit.
"""

import json
import logging
import os
from datetime import datetime
from utils.llm import call_llm, parse_json_response
from utils.technicals import compute_indicators
from utils.finnhub_client import FinnhubClient
from db.schema import Trade, Position, AgentMemory, get_session
from models.pm_profiles import ACTIVE_PROFILES

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a position health monitor for a paper trading system.
You review open positions against current price action and indicators.
Flag any positions that are deteriorating or at risk, even if stops haven't triggered.

Respond in JSON:
{
  "assessments": [
    {
      "symbol": "SPY",
      "profile": "moderate",
      "health": "healthy|warning|critical",
      "reasoning": "why this position is or isn't at risk",
      "recommendation": "hold|tighten_stop|close_partial|close_full|none"
    }
  ],
  "summary": "one sentence overall portfolio health"
}

Be specific. Reference actual numbers. "Critical" means the position should probably be closed.
"""


def run(engine) -> dict:
    """Review all open positions for health status."""
    fh = FinnhubClient()
    db = get_session(engine)

    positions = db.query(Position).all()
    if not positions:
        db.close()
        return {"assessments": [], "summary": "no open positions"}

    pos_data = []
    for p in positions:
        # Get current price
        try:
            import yfinance as yf
            t = yf.Ticker(p.symbol)
            price = float(t.fast_info.get("lastPrice", p.avg_cost))
        except Exception:
            price = p.avg_cost

        # Get indicators
        candles = fh.get_candles(p.symbol, resolution="5", days=2)
        indicators = compute_indicators(candles)

        # Get the open trade for stop/target
        trade = (
            db.query(Trade)
            .filter_by(symbol=p.symbol, profile=p.profile, status="open")
            .order_by(Trade.entry_time.desc())
            .first()
        )

        if p.side == "short":
            unrealized_pnl_pct = (p.avg_cost - price) / p.avg_cost * 100
        else:
            unrealized_pnl_pct = (price - p.avg_cost) / p.avg_cost * 100

        pos_data.append({
            "symbol": p.symbol,
            "profile": p.profile,
            "side": p.side,
            "avg_cost": p.avg_cost,
            "current_price": round(price, 2),
            "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
            "stop_price": trade.stop_price if trade else None,
            "target_price": trade.target_price if trade else None,
            "indicators": {
                "rsi": indicators.get("rsi"),
                "trend": indicators.get("trend"),
                "macd_cross": indicators.get("macd_cross"),
                "price_vs_vwap": "above" if price > indicators.get("vwap", 0) else "below",
            } if indicators and "error" not in indicators else {},
        })

    db.close()

    if not pos_data:
        return {"assessments": [], "summary": "no position data"}

    user_prompt = f"""
Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

OPEN POSITIONS:
{json.dumps(pos_data, indent=2)}

Assess the health of each position. Flag any that are deteriorating.
"""

    try:
        raw = call_llm(SYSTEM_PROMPT, user_prompt, json_mode=True, tier="medium")
        result = parse_json_response(raw)
    except Exception as e:
        log.error(f"Position health LLM error: {e}")
        return {"assessments": [], "summary": "error"}

    # Store in agent memory for PM to read
    db = get_session(engine)
    db.add(AgentMemory(
        agent="position_health",
        symbol=None,
        key="health_check",
        value=json.dumps(result),
    ))
    db.commit()
    db.close()

    # Log warnings
    for a in result.get("assessments", []):
        if a.get("health") in ("warning", "critical"):
            emoji = "⚠️" if a["health"] == "warning" else "🚨"
            log.warning(f"{emoji} {a['symbol']} ({a['profile']}): {a['health']} — {a.get('reasoning', '')}")

    log.info(f"Position health: {result.get('summary', '')}")
    return result
