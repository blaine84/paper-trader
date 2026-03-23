"""
Analyst Agent
Runs technical analysis on each symbol.
Reads researcher sentiment, combines with technicals.
Writes signal recommendations to AgentMemory.
"""

import json
from datetime import datetime
from utils.finnhub_client import FinnhubClient
from utils.technicals import compute_indicators
from utils.llm import call_llm, parse_json_response
from db.schema import AgentMemory, get_session
from utils.case_library import get_relevant_cases, format_cases_for_prompt
from agents.quant_researcher import build_strategy_context


SYSTEM_PROMPT = """You are a technical analyst for day trading.
You receive price data, technical indicators, and research sentiment for a stock.

Your job: characterize the current setup. You are a signal engine, not a trader.

You do NOT decide:
  - Whether to trade
  - Entry price or timing
  - Stop placement
  - Target or position size
  Those are the Portfolio Manager's decisions.

You DO decide:
  - What direction the setup favors (LONG / SHORT / HOLD)
  - How strong and clean the setup is
  - What the key price levels are (support, resistance, VWAP, prior highs/lows)
  - What would invalidate the setup (the line in the sand)
  - How confident you are in the read

Respond in JSON:
{
  "symbol": "...",
  "signal": "LONG|SHORT|HOLD",
  "strength": "weak|moderate|strong",
  "confidence": "low|medium|high",
  "setup_type": "gap_and_go|vwap_reclaim|technical_breakout|momentum_fade|reversal|range|etc",
  "reasoning": "what the indicators and tape are saying — be specific",
  "key_levels": {
    "support": 122.00,
    "resistance": 126.00,
    "vwap": 123.50,
    "prior_high": 127.00,
    "prior_low": 121.00
  },
  "invalidation": "the condition that would make this setup wrong (e.g. price closes below VWAP, loses 122 support)",
  "indicators": {
    "rsi": 58.3,
    "macd_bias": "bullish|bearish|neutral",
    "ema_trend": "bullish|bearish|neutral",
    "above_vwap": true,
    "bb_position": "upper|middle|lower|outside_upper|outside_lower"
  }
}

HOLD is a valid and useful signal. Output it whenever the setup is ambiguous or low quality.
"""


def run(engine, symbols: list[str]) -> dict:
    fh = FinnhubClient()
    db = get_session(engine)

    # Pull latest researcher sentiment from memory
    recent_sentiment = {}
    sentiments = (
        db.query(AgentMemory)
        .filter_by(agent="researcher", key="sentiment")
        .order_by(AgentMemory.timestamp.desc())
        .all()
    )
    seen = set()
    for s in sentiments:
        if s.symbol not in seen:
            recent_sentiment[s.symbol] = json.loads(s.value)
            seen.add(s.symbol)

    # Pull selection feedback (from Reviewer → for Analyst)
    sel_fb = (
        db.query(AgentMemory)
        .filter_by(agent="reviewer", key="selection_feedback")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    lesson_text = sel_fb.value if sel_fb else "No selection feedback yet."

    # Strategy context from Quant Researcher
    strategy_context = build_strategy_context(engine)

    signals = {}

    for sym in symbols:
        candles = fh.get_candles(sym, resolution="5", days=2)
        indicators = compute_indicators(candles)
        quote = fh.get_quote(sym)
        sentiment = recent_sentiment.get(sym, {})

        # Query case library for relevant precedents
        case_context = {
            "market_regime": "risk_on" if sentiment.get("sentiment") == "bullish" else
                             "risk_off" if sentiment.get("sentiment") == "bearish" else "mixed",
            "bias": "long" if indicators.get("trend") == "bullish" else "short",
        }
        relevant_cases = get_relevant_cases(engine, case_context, limit=5)
        cases_text = format_cases_for_prompt(relevant_cases)

        user_prompt = f"""
Symbol: {sym}
Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

CURRENT QUOTE:
{json.dumps(quote, indent=2)}

TECHNICAL INDICATORS:
{json.dumps(indicators, indent=2)}

RESEARCH SENTIMENT:
{json.dumps(sentiment, indent=2)}

STRATEGY RECOMMENDATIONS (from Quant Researcher):
{strategy_context}

RELEVANT PAST CASES:
{cases_text}

Produce your trading signal JSON for {sym}.
"""
        raw = call_llm(SYSTEM_PROMPT, user_prompt, json_mode=True)
        signal = parse_json_response(raw)
        signals[sym] = signal

        # Save to memory
        mem = AgentMemory(
            agent="analyst",
            symbol=sym,
            key="signal",
            value=json.dumps(signal),
        )
        db.add(mem)

    db.commit()
    db.close()
    return signals
