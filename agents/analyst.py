"""
Analyst Agent
Runs technical analysis on each symbol.
Reads researcher sentiment, combines with technicals.
Writes signal recommendations to AgentMemory.
"""

import json
import logging
from datetime import datetime, timedelta, timezone as dt_tz
from utils.finnhub_client import FinnhubClient
from utils.technicals import compute_indicators
from utils.llm import call_llm, parse_json_response
from db.schema import AgentMemory, get_session
from utils.trade_events import log_trade_event
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
    write_feedback_health_status,
)
from utils.symbol_class import classify_symbol, validate_setup_for_symbol

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

gap_and_go is ONLY valid for individual stocks. Do NOT assign gap_and_go to ETFs (SPY, QQQ, IWM, XLK, etc.), indices (VIX), or other non-stock instruments. Use technical_breakout or orb instead.

HOLD is a valid and useful signal. Output it whenever the setup is ambiguous or low quality.

STRICT OUTPUT CONTRACT:
- Return exactly the analyst signal object above.
- Do NOT return a `decisions` array.
- Do NOT output BUY, SELL, SHORT, entry_price, stop_loss, target, quantity, or portfolio_notes.
- You are not the Portfolio Manager. If tempted to propose a trade, express only LONG/SHORT/HOLD signal quality and key levels.
"""


def normalize_analyst_signal_shape(signal: dict, symbol: str) -> dict:
    """Coerce malformed analyst LLM output back into the analyst signal schema.

    Local models occasionally drift into Portfolio Manager format, e.g.
    {"decisions": [{"action": "BUY", ...}], "portfolio_notes": ...}.  That
    must never be persisted as an analyst signal because PM treats analyst
    memory as authoritative signal context.  When schema drift is detected,
    quarantine it as a weak HOLD rather than letting contaminated decisions
    flow downstream.
    """
    if not isinstance(signal, dict):
        return {
            "symbol": symbol,
            "signal": "HOLD",
            "strength": "weak",
            "confidence": "low",
            "setup_type": "malformed_analyst_output",
            "setup_reasoning": "Analyst LLM returned a non-object response; quarantined.",
            "reasoning": "Malformed analyst output was downgraded to HOLD before persistence.",
            "key_levels": {},
            "invalidation": "N/A — malformed analyst output",
            "indicators": {},
            "malformed_output_quarantined": True,
            "raw_malformed_output": str(signal)[:1000],
        }

    # Hard quarantine PM-shaped responses.  Do not salvage BUY/SHORT into a
    # tradeable signal; the model crossed agent boundaries, so confidence is low.
    if "decisions" in signal or "portfolio_notes" in signal:
        return {
            "symbol": symbol,
            "signal": "HOLD",
            "strength": "weak",
            "confidence": "low",
            "setup_type": "malformed_analyst_output",
            "setup_reasoning": "Analyst LLM returned Portfolio Manager decision schema; quarantined.",
            "reasoning": "Malformed analyst output was downgraded to HOLD before persistence to avoid contaminating PM entry decisions.",
            "key_levels": {},
            "invalidation": "N/A — malformed analyst output",
            "indicators": {},
            "malformed_output_quarantined": True,
            "raw_malformed_output": json.dumps(signal, default=str)[:2000],
        }

    normalized = dict(signal)
    normalized["symbol"] = symbol

    valid_signals = {"LONG", "SHORT", "HOLD"}
    sig = str(normalized.get("signal", "HOLD")).upper().strip()
    normalized["signal"] = sig if sig in valid_signals else "HOLD"

    valid_strengths = {"weak", "moderate", "strong"}
    strength = str(normalized.get("strength", "weak")).lower().strip()
    normalized["strength"] = strength if strength in valid_strengths else "weak"

    valid_conf = {"low", "medium", "high"}
    confidence = str(normalized.get("confidence", "low")).lower().strip()
    normalized["confidence"] = confidence if confidence in valid_conf else "low"

    normalized.setdefault("setup_type", "unknown")
    normalized.setdefault("setup_reasoning", "")
    normalized.setdefault("reasoning", "")
    if not isinstance(normalized.get("key_levels"), dict):
        normalized["key_levels"] = {}
    normalized.setdefault("invalidation", "")
    if not isinstance(normalized.get("indicators"), dict):
        normalized["indicators"] = {}

    return normalized


def sanitize_analyst_key_levels(signal: dict, quote: dict, indicators: dict) -> dict:
    """Replace hallucinated/cross-symbol key levels with market-data levels.

    The LLM is allowed to interpret structure, but it must not invent price
    levels.  We keep numeric LLM levels only when they are plausibly near the
    current quote; otherwise we fall back to deterministic quote/indicator data.
    """
    current = quote.get("price") if isinstance(quote, dict) else None
    try:
        current = float(current)
    except (TypeError, ValueError):
        current = None

    if not current or current <= 0:
        signal["key_levels"] = {}
        signal["key_levels_sanitized"] = True
        signal["key_levels_sanitize_reason"] = "missing_current_price"
        return signal

    raw_levels = signal.get("key_levels") if isinstance(signal.get("key_levels"), dict) else {}
    sanitized = {}
    removed = {}

    # Anything farther than 20% from current price is almost certainly copied
    # from another ticker/timeframe for this intraday system.
    max_distance_pct = 20.0

    for name, value in raw_levels.items():
        try:
            val = float(value)
        except (TypeError, ValueError):
            removed[name] = value
            continue
        if val <= 0:
            removed[name] = value
            continue
        dist_pct = abs((current - val) / current) * 100
        if dist_pct <= max_distance_pct:
            sanitized[name] = round(val, 4)
        else:
            removed[name] = value

    def _num(source: dict, key: str):
        try:
            val = source.get(key)
            return round(float(val), 4) if val is not None and float(val) > 0 else None
        except (TypeError, ValueError):
            return None

    fallback = {
        "support": _num(quote, "low"),
        "resistance": _num(quote, "high"),
        "vwap": _num(indicators, "vwap"),
        "prior_high": _num(quote, "high"),
        "prior_low": _num(quote, "low"),
    }

    for key, val in fallback.items():
        if key not in sanitized and val is not None:
            sanitized[key] = val

    if removed:
        signal["key_levels_sanitized"] = True
        signal["removed_key_levels"] = removed
        signal["key_levels_sanitize_reason"] = "level_too_far_from_current_price_or_non_numeric"

    signal["key_levels"] = sanitized
    return signal


def enrich_signal_with_quote_context(signal: dict, quote: dict, candles: dict) -> dict:
    """
    Inject structured quote fields into the analyst signal for downstream gates.

    Persists: current_price, quote_timestamp, day_open, day_high, day_low,
    prev_close, change_pct, and relative_volume (when computable).

    These fields are required by the Catalyst Specificity Gate's confirmation
    scoring to have discrimination power.

    Args:
        signal: The LLM-generated analyst signal dict (modified in place).
        quote: The quote dict from FinnhubClient.get_quote().
        candles: The candle data dict from FinnhubClient.get_candles().

    Returns:
        The enriched signal dict (same reference as input).
    """
    # Persist structured quote fields
    if quote:
        signal["current_price"] = quote.get("price")
        signal["quote_timestamp"] = quote.get("timestamp")
        signal["day_open"] = quote.get("open")
        signal["day_high"] = quote.get("high")
        signal["day_low"] = quote.get("low")
        signal["prev_close"] = quote.get("prev_close")
        signal["change_pct"] = quote.get("change_pct")

    # Compute relative_volume from candle data if available.
    # relative_volume = current period volume / average volume over lookback.
    if candles and candles.get("volume"):
        volumes = candles["volume"]
        if len(volumes) >= 2:
            current_vol = volumes[-1]
            avg_vol = sum(volumes[:-1]) / len(volumes[:-1])
            if avg_vol > 0:
                signal["relative_volume"] = round(current_vol / avg_vol, 2)

    return signal


def run(engine, symbols: list[str]) -> dict:
    fh = FinnhubClient()
    db = get_session(engine)

    try:
        process_pending_feedback(engine)
    except Exception as exc:
        log.exception("Analyst feedback processing failed")
        write_feedback_health_status(engine, status="failed", errors=[str(exc)])

    # Pull latest researcher sentiment from memory (ignore entries older than 36h)
    recent_sentiment = {}
    sentiment_cutoff = datetime.now(dt_tz.utc) - timedelta(hours=36)
    sentiments = (
        db.query(AgentMemory)
        .filter_by(agent="researcher", key="sentiment")
        .filter(AgentMemory.timestamp >= sentiment_cutoff)
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
            raw = call_llm(SYSTEM_PROMPT, user_prompt, json_mode=True, tier="finance", purpose=f"analyst_signal:{sym}")
            signal = parse_json_response(raw)
            signal = normalize_analyst_signal_shape(signal, sym)
            signal = sanitize_analyst_key_levels(signal, quote, indicators)
            signal = apply_signal_mitigation(signal, active_mitigations)
            validation_result = validate_setup_for_symbol(sym, signal.get("setup_type"))
            signal.update(validation_result)

            # Enrich signal with structured quote context for downstream gates
            enrich_signal_with_quote_context(signal, quote, candles)

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

    # Drift warning: detect if LLM is over-assigning gap_and_go to non-stock symbols
    raw_gap_and_go = 0
    reclassified_gap_and_go = 0
    for sig in signals.values():
        # A signal counts as "raw gap_and_go" if its current setup_type is gap_and_go
        # OR if it was reclassified FROM gap_and_go (original_setup_type == "gap_and_go")
        is_current_gap = sig.get("setup_type") == "gap_and_go"
        is_reclassified_from_gap = (
            sig.get("setup_reclassified") is True
            and sig.get("original_setup_type") == "gap_and_go"
        )
        if is_current_gap or is_reclassified_from_gap:
            raw_gap_and_go += 1
        if is_reclassified_from_gap:
            reclassified_gap_and_go += 1

    if raw_gap_and_go >= 10 and (reclassified_gap_and_go / raw_gap_and_go) > 0.10:
        log.warning(
            "Signal drift detected: %d/%d raw gap_and_go signals were reclassified (%.1f%%). "
            "LLM may be over-assigning gap_and_go to non-stock symbols.",
            reclassified_gap_and_go,
            raw_gap_and_go,
            (reclassified_gap_and_go / raw_gap_and_go) * 100,
        )

    # Save all signals to memory
    db2 = get_session(engine)
    for sym, signal in signals.items():
        db2.add(AgentMemory(
            agent="analyst",
            symbol=sym,
            key="signal",
            value=json.dumps(signal),
        ))
        log_trade_event(
            db2, "signal_seen", agent="analyst", symbol=sym,
            price=signal.get("entry") or signal.get("entry_price"),
            message=signal.get("reasoning"),
            payload=signal,
        )
    db2.commit()
    db2.close()
    return signals
