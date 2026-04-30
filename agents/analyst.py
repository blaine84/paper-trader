"""
Analyst Agent
Runs technical analysis on each symbol.
Reads researcher sentiment, combines with technicals.
Writes signal recommendations to AgentMemory.
"""

import json
import logging
from datetime import datetime, timezone as dt_tz
from utils.finnhub_client import FinnhubClient
from utils.technicals import compute_indicators
from utils.llm import call_llm, parse_json_response
from db.schema import AgentMemory, get_session
from utils.case_library import get_relevant_cases, format_cases_for_prompt
from agents.quant_researcher import build_strategy_context
from utils.catalyst_freshness import (
    compute_freshness_state,
    get_breaking_news_for_symbols,
    get_market_day_start,
    ET,
)
from feedback_loop.analyst_feedback import (
    apply_signal_mitigation,
    build_feedback_prompt_context,
    get_active_mitigations,
    process_pending_feedback,
)

log = logging.getLogger(__name__)


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
  "setup_type": "gap_and_go|vwap_reclaim|orb|momentum_fade|trend_pullback|news_catalyst|sector_rotation|short_squeeze",
  "setup_reasoning": "why this setup type was chosen — what specific price action, indicators, or conditions match this setup",
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

    try:
        process_pending_feedback(engine)
    except Exception as exc:
        log.warning("Analyst feedback processing failed: %s", exc)

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

    # Meta-reviewer recommendations for analyst
    meta_rec = (
        db.query(AgentMemory)
        .filter_by(agent="meta_reviewer", symbol="analyst", key="agent_recommendation")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    meta_text = meta_rec.value if meta_rec else ""

    # Strategy context from Quant Researcher
    strategy_context = build_strategy_context(engine)
    feedback_context = build_feedback_prompt_context(engine)
    active_mitigations = get_active_mitigations(engine)

    # Get all valid setup types (hardcoded + dynamic)
    from utils.strategy_store import get_all_setup_types
    valid_setups = get_all_setup_types(engine)

    signals = {}

    def _analyze_symbol(sym):
        """Analyze a single symbol — runs in its own thread."""
        try:
            candles = fh.get_candles(sym, resolution="5", days=2)
            indicators = compute_indicators(candles)
            quote = fh.get_quote(sym)
            sentiment = recent_sentiment.get(sym, {})

            case_context = {
                "market_regime": "risk_on" if sentiment.get("sentiment") == "bullish" else
                                 "risk_off" if sentiment.get("sentiment") == "bearish" else "mixed",
                "bias": "long" if indicators.get("trend") == "bullish" else "short",
            }
            relevant_cases = get_relevant_cases(engine, case_context, limit=3)
            cases_text = format_cases_for_prompt(relevant_cases)

            # --- Catalyst freshness: query breaking news (isolated) ---
            breaking_alerts = []
            breaking_news_ts = None
            try:
                thread_db = get_session(engine)
                now_et = datetime.now(ET)
                mds = get_market_day_start(now_et)
                bn = get_breaking_news_for_symbols(thread_db, [sym], mds)
                breaking_alerts = bn.get(sym, [])
                if breaking_alerts:
                    breaking_news_ts = now_et
                thread_db.close()
            except Exception as e:
                log.error(f"Analyst breaking news query failed for {sym}: {e}")

            # Compute freshness state — use timezone-aware now (not datetime.utcnow())
            researcher_ts = None
            sent_rows = (
                db.query(AgentMemory)
                .filter_by(agent="researcher", key="sentiment", symbol=sym)
                .order_by(AgentMemory.timestamp.desc())
                .first()
            )
            if sent_rows and sent_rows.timestamp:
                researcher_ts = sent_rows.timestamp
                # Ensure timezone-aware
                if researcher_ts.tzinfo is None:
                    from pytz import utc as UTC
                    researcher_ts = UTC.localize(researcher_ts)

            now_aware = datetime.now(dt_tz.utc)
            # Normalize breaking_news_ts to UTC for comparison
            bn_ts_utc = None
            if breaking_news_ts is not None:
                bn_ts_utc = breaking_news_ts.astimezone(dt_tz.utc) if breaking_news_ts.tzinfo else None

            freshness_state = compute_freshness_state(researcher_ts, bn_ts_utc, now_aware)

            # Build freshness context block for the prompt
            freshness_context = f"""
CATALYST FRESHNESS:
  Freshness state: {freshness_state}
  Last researcher update: {researcher_ts or 'unknown'}
  Last breaking news: {bn_ts_utc or 'none'}
"""
            if freshness_state == "stale":
                freshness_context += "  ⚠️ WARNING: Catalyst data is STALE (>3 hours old). Reduce signal confidence accordingly.\n"
            elif freshness_state == "aging":
                freshness_context += "  ℹ️ NOTE: Catalyst data is AGING (1-3 hours old). May not reflect current conditions.\n"

            if breaking_alerts:
                freshness_context += f"\nBREAKING NEWS ALERTS:\n{json.dumps(breaking_alerts, indent=2)}\n"

            user_prompt = f"""
Symbol: {sym}
Time: {datetime.now(dt_tz.utc).strftime('%Y-%m-%d %H:%M UTC')}

VALID SETUP TYPES (use one of these):
{', '.join(valid_setups)}

SELECTION FEEDBACK (from Reviewer — your past signal quality):
{lesson_text}

META-REVIEWER RECOMMENDATIONS (system-level feedback for you):
{meta_text if meta_text else 'None yet'}

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

ANALYST FEEDBACK LOOP:
{feedback_context}
{freshness_context}
Produce your trading signal JSON for {sym}.
"""
            raw = call_llm(SYSTEM_PROMPT, user_prompt, json_mode=True, tier="medium")
            signal = parse_json_response(raw)
            signal = apply_signal_mitigation(signal, active_mitigations)
            signals[sym] = signal
        except Exception as e:
            log.error(f"Analyst error for {sym}: {e}")
            signals[sym] = {"signal": "HOLD", "strength": "weak", "confidence": "low", "setup_type": "error", "reasoning": str(e)}

    # Run all symbols in parallel
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_analyze_symbol, sym): sym for sym in symbols}
        for f in as_completed(futures):
            pass  # results stored in signals dict

    # Save all signals to memory
    db2 = get_session(engine)
    for sym, signal in signals.items():
        db2.add(AgentMemory(
            agent="analyst",
            symbol=sym,
            key="signal",
            value=json.dumps(signal),
        ))
    db2.commit()
    db2.close()
    return signals
