"""
Analyst Agent
Runs technical analysis on each symbol.
Reads researcher sentiment, combines with technicals.
Writes signal recommendations to AgentMemory.
"""

import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone as dt_tz
from utils.finnhub_client import FinnhubClient
from utils.technicals import compute_indicators
from utils.llm import call_llm, parse_json_response
from utils.multitimeframe_context import (
    build_multitimeframe_context,
    format_multitimeframe_context_for_prompt,
)
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
from utils.trigger_status import compute_trigger_status
from utils.symbol_class import classify_symbol, validate_setup_for_symbol

log = logging.getLogger(__name__)

ANALYST_CONTEXT_BOUNDARY = """CONTEXT BOUNDARY:
- The Time line above is the current analysis time.
- CURRENT QUOTE, TECHNICAL INDICATORS, MULTI-TIMEFRAME CONTEXT, RESEARCH SENTIMENT, and CATALYST FRESHNESS are the current evidence for this symbol.
- SELECTION FEEDBACK, META-REVIEWER RECOMMENDATIONS, RELEVANT PAST CASES, and ANALYST FEEDBACK LOOP are historical lessons only.
- Do not cite dated catalysts, old macro events, old earnings dates, or prior trade-review facts from historical lesson sections as current setup blockers unless they also appear in the current evidence sections for this symbol.
- If a historical lesson references a calendar date before the Time line, treat it as a past example, not today's catalyst.
"""

_STALE_HISTORICAL_FEEDBACK_RE = re.compile(
    r"\b("
    r"CPI|inflation print|July\s*15|Jul\s*15|"
    r"scheduled\s+(?:major\s+)?macro catalyst|"
    r"economic calendar intersection|intraday rotation setup mandate"
    r")\b",
    re.IGNORECASE,
)


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
  - How confident you are that the observed price action matches an executable setup

Respond in JSON:
{
  "symbol": "...",
  "signal": "LONG|SHORT|HOLD",
  "strength": "weak|moderate|strong",
  "confidence": "low|medium|high",
  "setup_type": "one of the VALID SETUP TYPES from the user prompt",
  "normalized_setup_suggestion": "null OR one of: sector_rotation_swing|risk_off_macro_short|breakout_retest|pullback_continuation|relative_strength_swing|support_bounce_swing|failed_breakdown_reclaim — populate only when your setup_type is NOT already in the executable intraday or swing set (e.g. 'sector_rotation' → suggest 'sector_rotation_swing'). Leave null when setup_type is already executable.",
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
  },
  "llm_veto_reason": "required when you output HOLD while deterministic sanity favors LONG/SHORT; otherwise null",
  "veto_evidence": ["specific evidence that overrules the deterministic directional read"]
}

Prefer the exact VALID SETUP TYPES listed in the user prompt. If the setup genuinely fits a label outside that list, preserve the best label and explain it in setup_reasoning.

gap_and_go is ONLY valid for individual stocks. Do NOT assign gap_and_go to ETFs (SPY, QQQ, IWM, XLK, etc.), indices (VIX), or other non-stock instruments. Use technical_breakout or orb instead.

Do NOT use directional_confusion_breakout, technical_confusion_breakout, or
any *_confusion_* label as a setup_type. If the executable setup is confused,
output HOLD with setup_type="unclear_direction", strength="weak",
confidence="low", and explain the conflict in setup_reasoning.

SWING SETUP LABELS:
When your confidence is at least medium, strength is at least moderate, and you output a directional signal (LONG or SHORT), prefer these canonical swing setup types:
  - sector_rotation_swing: stock rotating into a strong sector with multi-day follow-through potential
  - risk_off_macro_short: broad risk-off environment favoring macro short on a weak name
  - breakout_retest: price broke out above resistance and is retesting the breakout level as support
  - pullback_continuation: stock in an established trend pulling back to a key level for continuation
  - relative_strength_swing: stock showing persistent relative strength vs. sector/market over multiple days
  - support_bounce_swing: price testing a well-defined support level with signs of a multi-day reversal
  - failed_breakdown_reclaim: price broke below support but quickly reclaimed, trapping shorts

Use diagnostic labels (labels NOT in the canonical sets above) only when:
  - Your signal is HOLD
  - Your confidence is low
  - Indicators conflict on direction and you cannot resolve a clean setup
  - No canonical setup type matches the observed price action

Confidence is setup confidence, not just directional confidence. If price action
looks bullish or bearish but does not match a clean executable setup, output
HOLD with low confidence. HOLD is a valid and useful signal. Output it whenever
the setup is ambiguous or low quality.

VETO ACCOUNTABILITY:
- If the DETERMINISTIC TECHNICAL SANITY CHECK below says bias=LONG or bias=SHORT and you still output HOLD, you MUST fill llm_veto_reason with the specific disqualifying evidence.
- Acceptable veto reasons include: stale/missing catalyst for a catalyst-dependent setup, conflicting key levels, thin/invalid volume, overextension/chop, invalid symbol/setup mapping, or explicit reviewer mitigation.
- Do not use vague vetoes like "risk-off", "uncertain market", or "mixed indicators" unless you name the exact conflicting indicators/levels.
- If no concrete veto exists, output the deterministic direction as the signal with appropriate weak/moderate strength; PM decides whether to trade.

STRICT OUTPUT CONTRACT:
- Return exactly the analyst signal object above.
- Do NOT return a `decisions` array.
- Do NOT output BUY, SELL, SHORT, entry_price, stop_loss, target, quantity, or portfolio_notes.
- You are not the Portfolio Manager. If tempted to propose a trade, express only LONG/SHORT/HOLD signal quality and key levels.
"""

VETO_REPAIR_SYSTEM_PROMPT = """You repair one Analyst signal contract violation.

The original Analyst returned HOLD while deterministic technical sanity strongly
favored LONG or SHORT, but did not provide a valid veto.

Return exactly this JSON patch:
{
  "signal": "LONG|SHORT|HOLD",
  "strength": "weak|moderate|strong",
  "confidence": "low|medium|high",
  "llm_veto_reason": "specific reason or null",
  "veto_evidence": ["specific measurable evidence"]
}

Rules:
- If concrete evidence invalidates the deterministic direction, keep HOLD and
  provide both a specific veto reason and at least one evidence item.
- Otherwise use the deterministic LONG or SHORT direction.
- Never return the opposite directional signal.
- Do not provide entry, stop, target, quantity, or portfolio decisions.
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

    # Validate normalized_setup_suggestion before deciding whether a diagnostic
    # setup label can be safely mapped into an executable swing setup.
    _suggestion = normalized.get("normalized_setup_suggestion")
    if _suggestion is not None:
        from utils.gate_config import SWING_EXECUTABLE_SETUP_TYPES
        if _suggestion not in SWING_EXECUTABLE_SETUP_TYPES:
            normalized["normalized_setup_suggestion"] = None
    else:
        normalized.setdefault("normalized_setup_suggestion", None)

    normalized.setdefault("setup_type", "unknown")
    setup_type = str(normalized.get("setup_type", "unknown")).lower().strip()
    if _is_diagnostic_confusion_setup(setup_type) and setup_type != "unclear_direction":
        normalized["setup_type"] = "unclear_direction"
        normalized["signal"] = "HOLD"
        normalized["strength"] = "weak"
        normalized["confidence"] = "low"
        normalized["normalized_setup_suggestion"] = None
        normalized["setup_validation_warning"] = (
            f"{setup_type} is not a valid setup label; "
            "rewritten to unclear_direction/HOLD"
        )
        normalized["needs_setup_type_review"] = True
    elif setup_type == "unclear_direction":
        actionable_direction = normalized["signal"] in {"LONG", "SHORT"}
        actionable_strength = normalized["strength"] in {"moderate", "strong"}
        actionable_confidence = normalized["confidence"] in {"medium", "high"}
        suggestion = normalized.get("normalized_setup_suggestion")
        if suggestion is None:
            suggestion = _infer_unclear_direction_swing_setup(normalized)
            normalized["normalized_setup_suggestion"] = suggestion
        if (
            actionable_direction
            and actionable_strength
            and actionable_confidence
            and suggestion is not None
        ):
            normalized["original_setup_type"] = "unclear_direction"
            normalized["setup_type"] = suggestion
            normalized["setup_validation_warning"] = (
                "unclear_direction carried a directional signal and valid "
                "normalized_setup_suggestion; promoted to canonical swing setup"
            )
            normalized["needs_setup_type_review"] = True
        else:
            normalized["setup_type"] = "unclear_direction"
            normalized["signal"] = "HOLD"
            normalized["strength"] = "weak"
            normalized["confidence"] = "low"
            normalized["normalized_setup_suggestion"] = None
            normalized["setup_validation_warning"] = (
                "unclear_direction is diagnostic-only; forced to HOLD"
            )
            normalized["needs_setup_type_review"] = True
    else:
        normalized["setup_type"] = setup_type
    normalized.setdefault("setup_reasoning", "")
    normalized.setdefault("reasoning", "")
    if not isinstance(normalized.get("key_levels"), dict):
        normalized["key_levels"] = {}
    normalized.setdefault("invalidation", "")
    if not isinstance(normalized.get("indicators"), dict):
        normalized["indicators"] = {}

    _enforce_directional_setup_contract(normalized)
    sanitize_historical_feedback_bleed(normalized)

    return normalized


def _remove_historical_feedback_sentences(text: str) -> tuple[str, bool]:
    """Remove stale reviewer/calendar sentences copied into current signal text."""
    if not isinstance(text, str) or not text.strip():
        return text, False
    if not _STALE_HISTORICAL_FEEDBACK_RE.search(text):
        return text, False

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip() and not _STALE_HISTORICAL_FEEDBACK_RE.search(sentence)
    ]
    return " ".join(kept), True


def sanitize_historical_feedback_bleed(signal: dict) -> dict:
    """Prevent historical reviewer feedback from becoming persisted current evidence."""
    if not isinstance(signal, dict):
        return signal

    redacted_fields = []
    for field in ("setup_reasoning", "reasoning", "invalidation", "llm_veto_reason"):
        value = signal.get(field)
        cleaned, changed = _remove_historical_feedback_sentences(value)
        if not changed:
            continue
        redacted_fields.append(field)
        if cleaned:
            signal[field] = cleaned
        elif field == "invalidation":
            signal[field] = "Use current technical invalidation levels; stale historical catalyst reference removed."
        else:
            signal[field] = (
                "Stale historical reviewer catalyst reference removed; reassess using "
                "current quote, technical, research, and catalyst freshness evidence."
            )

    if redacted_fields:
        signal["historical_feedback_redacted"] = True
        signal["historical_feedback_redacted_fields"] = redacted_fields
    return signal


def _is_diagnostic_confusion_setup(setup_type: str) -> bool:
    """Return True when a setup label is diagnostic-only confusion drift."""
    return (
        setup_type in {
            "unclear_direction",
            "directional_confusion_breakout",
            "technical_confusion_breakout",
        }
        or "_confusion_" in setup_type
    )


def _enforce_directional_setup_contract(signal: dict) -> None:
    """Prevent directional analyst signals from carrying non-executable labels."""
    if signal.get("signal") not in {"LONG", "SHORT"}:
        return

    from utils.gate_config import (
        CANDIDATE_EXECUTABLE_SETUP_TYPES,
        SWING_EXECUTABLE_SETUP_TYPES,
    )

    setup_type = signal.get("setup_type")
    executable_or_mappable = (
        set(CANDIDATE_EXECUTABLE_SETUP_TYPES)
        | set(SWING_EXECUTABLE_SETUP_TYPES)
        | {"sector_rotation"}
    )
    if setup_type in executable_or_mappable:
        return

    signal["original_signal"] = signal.get("signal")
    signal["original_strength"] = signal.get("strength")
    signal["original_confidence"] = signal.get("confidence")
    signal["original_setup_type"] = setup_type
    signal["signal"] = "HOLD"
    signal["strength"] = "weak"
    signal["confidence"] = "low"
    signal["setup_type"] = "unclear_direction"
    signal["normalized_setup_suggestion"] = None
    signal["setup_validation_warning"] = (
        f"Directional signal carried non-executable setup_type '{setup_type}'; "
        "forced to HOLD"
    )
    signal["needs_setup_type_review"] = True


def _infer_unclear_direction_swing_setup(signal: dict) -> str | None:
    """Infer a conservative swing setup for directional unclear_direction rows."""
    direction = signal.get("signal")
    strength = signal.get("strength")
    confidence = signal.get("confidence")
    if direction not in {"LONG", "SHORT"}:
        return None
    if strength not in {"moderate", "strong"} or confidence not in {"medium", "high"}:
        return None

    indicators = signal.get("indicators") if isinstance(signal.get("indicators"), dict) else {}
    key_levels = signal.get("key_levels") if isinstance(signal.get("key_levels"), dict) else {}
    has_levels = key_levels.get("support") is not None and key_levels.get("resistance") is not None
    if not has_levels:
        return None

    ema_trend = str(signal.get("ema_trend") or indicators.get("ema_trend") or "").lower()
    macd_bias = str(indicators.get("macd_bias") or "").lower()
    above_vwap = indicators.get("above_vwap")
    text = " ".join(
        str(signal.get(field) or "")
        for field in ("setup_reasoning", "reasoning", "invalidation")
    ).lower()

    if direction == "LONG":
        bullish_context = (
            ema_trend == "bullish"
            and macd_bias == "bullish"
            and above_vwap is True
        )
        if not bullish_context:
            return None
        if "breakout" in text or "resistance" in text:
            return "breakout_retest"
        if "support" in text or "vwap" in text:
            return "support_bounce_swing"
        return "pullback_continuation"

    try:
        rsi = float(indicators.get("rsi"))
    except (TypeError, ValueError):
        rsi = None
    bearish_context = (
        ema_trend == "bearish"
        and macd_bias == "bearish"
        and above_vwap is False
        and (rsi is None or rsi >= 40)
    )
    risk_off_context = (
        signal.get("market_regime") == "risk_off"
        or "risk-off" in text
        or "risk off" in text
    )
    if bearish_context and risk_off_context:
        return "risk_off_macro_short"
    return None


def annotate_unregistered_setup(signal: dict, valid_setups: list[str]) -> dict:
    """Flag setup labels outside the registry without rewriting them."""
    setup_type = signal.get("setup_type")
    if not setup_type or setup_type in set(valid_setups):
        return signal

    signal.setdefault(
        "setup_validation_warning",
        (
            f"setup_type '{setup_type}' is not in the current setup registry; "
            "preserved for review"
        ),
    )
    signal["needs_setup_type_review"] = True
    log.warning(
        "Analyst emitted unregistered setup_type=%s for %s; preserving with warning",
        setup_type,
        signal.get("symbol", "unknown"),
    )
    return signal


def _current_price_from_quote(quote: dict) -> float | None:
    current = quote.get("price") if isinstance(quote, dict) else None
    try:
        current = float(current)
    except (TypeError, ValueError):
        return None
    return current if current > 0 else None


def _price_level_is_plausible(value, current: float, max_distance_pct: float = 20.0) -> bool:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return False
    if val <= 0 or current <= 0:
        return False
    dist_pct = abs((current - val) / current) * 100
    return dist_pct <= max_distance_pct


def sanitize_analyst_session_levels(signal: dict, quote: dict, max_distance_pct: float = 20.0) -> dict:
    """Remove candle-derived session levels that are implausible vs current price.

    yfinance/Finnhub occasionally return cross-symbol or stale candle levels.
    This protects both top-level structured fields and mirrored key_levels after
    quote/candle enrichment has overwritten the LLM output.
    """
    current = _current_price_from_quote(quote)
    if not current:
        return signal

    removed = {}
    for field in ("day_open", "day_high", "day_low", "prior_day_high", "prior_day_low", "prior_day_close"):
        if field in signal and signal.get(field) is not None:
            if not _price_level_is_plausible(signal.get(field), current, max_distance_pct):
                removed[field] = signal.pop(field)

    key_levels = signal.get("key_levels")
    if isinstance(key_levels, dict):
        for key in ("day_high", "day_low", "prior_high", "prior_low"):
            if key in key_levels and key_levels.get(key) is not None:
                if not _price_level_is_plausible(key_levels.get(key), current, max_distance_pct):
                    removed[f"key_levels.{key}"] = key_levels.pop(key)

    if removed:
        signal["session_levels_sanitized"] = True
        signal["removed_session_levels"] = removed
        signal["session_levels_sanitize_reason"] = "level_too_far_from_current_price_or_non_numeric"

    return signal


def validate_candle_indicator_alignment(
    symbol: str,
    quote: dict,
    candles: dict,
    indicators: dict,
    max_close_distance_pct: float = 5.0,
    max_vwap_distance_pct: float = 20.0,
) -> None:
    """Reject cross-symbol/stale candle contamination before prompting Analyst."""
    current = _current_price_from_quote(quote)
    closes = candles.get("close") if isinstance(candles, dict) else None
    if current and closes:
        last_close = _safe_float(closes[-1])
        if last_close is not None and last_close > 0:
            dist_pct = abs((current - last_close) / current) * 100
            if dist_pct > max_close_distance_pct:
                raise ValueError(
                    f"candle_quote_mismatch for {symbol}: quote={current} "
                    f"last_close={last_close} dist_pct={dist_pct:.2f}"
                )

    vwap = _safe_float(indicators.get("vwap") if isinstance(indicators, dict) else None)
    if current and vwap and vwap > 0:
        dist_pct = abs((current - vwap) / current) * 100
        if dist_pct > max_vwap_distance_pct:
            raise ValueError(
                f"indicator_quote_mismatch for {symbol}: quote={current} "
                f"vwap={vwap} dist_pct={dist_pct:.2f}"
            )


def sanitize_analyst_key_levels(signal: dict, quote: dict, indicators: dict) -> dict:
    """Replace hallucinated/cross-symbol key levels with market-data levels.

    The LLM is allowed to interpret structure, but it must not invent price
    levels.  We keep numeric LLM levels only when they are plausibly near the
    current quote; otherwise we fall back to deterministic quote/indicator data.
    """
    current = _current_price_from_quote(quote)

    if not current:
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
            if _price_level_is_plausible(val, current, max_distance_pct):
                sanitized[key] = val
            else:
                removed[f"fallback.{key}"] = val

    if removed:
        signal["key_levels_sanitized"] = True
        signal["removed_key_levels"] = removed
        signal["key_levels_sanitize_reason"] = "level_too_far_from_current_price_or_non_numeric"

    signal["key_levels"] = sanitized
    return signal


def _session_levels_from_candles(candles: dict) -> dict:
    """Derive current/prior session levels from candle timestamps.

    get_candles(days=2) returns yesterday + today for indicator warmup. The
    analyst/PM need explicit session boundaries so today's day_high/day_low are
    not confused with prior session levels.
    """
    if not candles or not candles.get("close"):
        return {}

    timestamps = candles.get("timestamps") or candles.get("timestamp") or candles.get("t") or []
    if not timestamps or len(timestamps) != len(candles.get("close", [])):
        return {}

    from datetime import datetime, timezone as _timezone

    dates = []
    for ts in timestamps:
        try:
            if isinstance(ts, (int, float)):
                dates.append(datetime.fromtimestamp(ts, _timezone.utc).date().isoformat())
            else:
                dates.append(str(ts)[:10])
        except Exception:
            dates.append(None)

    valid_dates = [d for d in dates if d]
    if not valid_dates:
        return {}

    current_session = valid_dates[-1]
    current_idx = [i for i, d in enumerate(dates) if d == current_session]
    prior_idx = [i for i, d in enumerate(dates) if d and d != current_session]
    levels = {"session_date": current_session}

    if current_idx:
        levels.update({
            "day_open": candles["open"][current_idx[0]],
            "day_high": max(candles["high"][i] for i in current_idx),
            "day_low": min(candles["low"][i] for i in current_idx),
        })
    if prior_idx:
        levels.update({
            "prior_day_high": max(candles["high"][i] for i in prior_idx),
            "prior_day_low": min(candles["low"][i] for i in prior_idx),
            "prior_day_close": candles["close"][prior_idx[-1]],
        })

    return levels


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_deterministic_sanity_prompt_context(precheck: dict) -> str:
    """Format deterministic sanity for the Analyst prompt."""
    if not isinstance(precheck, dict):
        return "No deterministic sanity check available."
    bias = precheck.get("bias", "HOLD")
    score = precheck.get("score", 0)
    reasons = precheck.get("reasons", [])
    if bias in {"LONG", "SHORT"}:
        instruction = (
            f"Deterministic sanity favors {bias} (score={score}). "
            "If you output HOLD, populate llm_veto_reason with concrete disqualifying evidence."
        )
    else:
        instruction = "Deterministic sanity is neutral; HOLD is acceptable when the setup is ambiguous."
    return json.dumps({
        "bias": bias,
        "score": score,
        "reasons": reasons,
        "instruction": instruction,
    }, indent=2)


def enforce_veto_accountability(signal: dict) -> dict:
    """Require a structured veto when LLM HOLD conflicts with deterministic sanity."""
    sanity = signal.get("deterministic_sanity")
    if not isinstance(sanity, dict) or not sanity.get("conflict"):
        signal.setdefault("llm_veto_required", False)
        return signal

    llm_signal = str(sanity.get("llm_signal") or signal.get("signal") or "").upper()
    deterministic_bias = sanity.get("bias")
    veto_text = str(signal.get("llm_veto_reason") or "").strip()
    veto_evidence = signal.get("veto_evidence")

    if llm_signal == "HOLD" and deterministic_bias in {"LONG", "SHORT"}:
        if not isinstance(veto_evidence, list):
            veto_evidence = [] if veto_evidence in (None, "") else [str(veto_evidence)]
        veto_evidence = [
            str(item).strip()
            for item in veto_evidence
            if str(item).strip()
        ]
        signal["veto_evidence"] = veto_evidence

        has_reason = bool(veto_text) and not veto_text.startswith("MISSING_LLM_VETO_REASON")
        has_evidence = bool(veto_evidence)
        signal["llm_veto_required"] = True
        signal["llm_veto_present"] = has_reason and has_evidence
        if not has_reason:
            signal["llm_veto_reason"] = (
                "MISSING_LLM_VETO_REASON: Analyst output HOLD despite deterministic "
                f"{deterministic_bias} sanity score={sanity.get('score')} without concrete veto evidence."
            )
            signal["llm_veto_contract_error"] = "missing_veto_reason"
        elif not has_evidence:
            signal["llm_veto_contract_error"] = "missing_veto_evidence"
        else:
            signal.pop("llm_veto_contract_error", None)
        signal["llm_veto_missing"] = not signal["llm_veto_present"]
    else:
        signal.setdefault("llm_veto_required", False)

    return signal


def repair_missing_veto_contract(signal: dict, symbol: str) -> dict:
    """Repair a conflicted HOLD once using the primary model.

    This does not auto-promote deterministic bias into a signal. The primary
    model must either accept that direction or return a concrete HOLD veto with
    evidence. Invalid retries remain HOLD and are explicitly quarantined.
    """
    if not signal.get("llm_veto_missing"):
        return signal

    if signal.get("mitigation"):
        mitigation = signal.get("mitigation") or {}
        signal["llm_veto_reason"] = (
            "ACTIVE_REVIEWER_MITIGATION: deterministic feedback throttle converted "
            f"{signal.get('original_signal', 'directional signal')} to HOLD."
        )
        signal["veto_evidence"] = [
            f"mitigation_level={mitigation.get('level', 'unknown')}",
            f"setup_type={mitigation.get('setup_type', signal.get('setup_type', 'unknown'))}",
        ]
        signal["veto_contract_repair_skipped"] = "active_reviewer_mitigation"
        return enforce_veto_accountability(signal)

    enabled = os.getenv("ANALYST_VETO_REPAIR_ENABLED", "true").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        signal["veto_contract_repair_skipped"] = "feature_disabled"
        return signal

    sanity = signal.get("deterministic_sanity")
    if not isinstance(sanity, dict) or sanity.get("bias") not in {"LONG", "SHORT"}:
        signal["veto_contract_repair_skipped"] = "missing_directional_sanity"
        return signal

    repair_context = {
        "symbol": symbol,
        "deterministic_sanity": sanity,
        "original_analyst_output": {
            "signal": signal.get("signal"),
            "strength": signal.get("strength"),
            "confidence": signal.get("confidence"),
            "setup_type": signal.get("setup_type"),
            "reasoning": signal.get("reasoning"),
            "llm_veto_reason": signal.get("llm_veto_reason"),
            "veto_evidence": signal.get("veto_evidence"),
            "current_price": signal.get("current_price"),
            "relative_volume": signal.get("relative_volume"),
            "key_levels": signal.get("key_levels"),
            "indicators": signal.get("indicators"),
        },
    }

    signal["veto_contract_repair_attempted"] = True
    try:
        raw = call_llm(
            VETO_REPAIR_SYSTEM_PROMPT,
            json.dumps(repair_context, indent=2, default=str),
            json_mode=True,
            tier=os.getenv("ANALYST_VETO_REPAIR_TIER", "high"),
            purpose=f"analyst_veto_repair:{symbol}",
        )
        repair = parse_json_response(raw)
        if not isinstance(repair, dict):
            raise ValueError("repair response was not an object")

        repaired_direction = str(repair.get("signal") or "").upper().strip()
        deterministic_bias = sanity["bias"]
        if repaired_direction not in {"HOLD", deterministic_bias}:
            raise ValueError(
                f"repair direction {repaired_direction!r} must be HOLD or {deterministic_bias}"
            )

        strength = str(repair.get("strength") or signal.get("strength") or "weak").lower()
        confidence = str(repair.get("confidence") or signal.get("confidence") or "low").lower()
        if strength not in {"weak", "moderate", "strong"}:
            strength = "weak"
        if confidence not in {"low", "medium", "high"}:
            confidence = "low"

        evidence = repair.get("veto_evidence")
        if not isinstance(evidence, list):
            evidence = [] if evidence in (None, "") else [str(evidence)]
        evidence = [str(item).strip() for item in evidence if str(item).strip()]
        veto_reason = str(repair.get("llm_veto_reason") or "").strip()

        if repaired_direction == "HOLD":
            if len(veto_reason) < 12 or not evidence:
                raise ValueError("HOLD repair still lacks a concrete reason and evidence")
            signal["llm_veto_reason"] = veto_reason
            signal["veto_evidence"] = evidence
        else:
            signal["llm_veto_reason"] = None
            signal["veto_evidence"] = []

        signal["signal"] = repaired_direction
        signal["strength"] = strength
        signal["confidence"] = confidence
        signal["veto_contract_repaired"] = True
        signal["veto_repair_method"] = "primary_llm"
        signal.pop("veto_contract_repair_failed", None)
        signal.pop("analyst_contract_failure", None)
        signal.pop("veto_repair_error", None)

        updated_sanity = dict(sanity)
        updated_sanity["llm_signal"] = repaired_direction
        updated_sanity["conflict"] = repaired_direction == "HOLD"
        signal["deterministic_sanity"] = updated_sanity

        if repaired_direction == deterministic_bias:
            signal["llm_veto_required"] = False
            signal["llm_veto_present"] = False
            signal["llm_veto_missing"] = False
            signal.pop("llm_veto_contract_error", None)
            return signal
        return enforce_veto_accountability(signal)
    except Exception as exc:
        log.warning("Analyst veto contract repair failed for %s: %s", symbol, exc)
        signal["veto_contract_repair_failed"] = True
        signal["analyst_contract_failure"] = "missing_veto_after_primary_retry"
        signal["veto_repair_error"] = str(exc)[:300]
        return signal


def compute_deterministic_signal_sanity(signal: dict, quote: dict, indicators: dict) -> dict:
    """Simple non-LLM directional sanity check for Analyst output.

    This is deliberately *not* a trade decision and does not override the LLM.
    It gives us a stable instrument panel when the Analyst goes globally timid:
    if the model says HOLD while deterministic trend/VWAP/momentum inputs lean
    clearly LONG or SHORT, PM skips become diagnosable instead of mysterious.
    """
    signal_direction = str(signal.get("signal", "HOLD") or "HOLD").upper()
    score = 0
    reasons = []

    current = _safe_float(quote.get("price") if isinstance(quote, dict) else None)
    change_pct = _safe_float(quote.get("change_pct") if isinstance(quote, dict) else None)
    vwap = _safe_float(indicators.get("vwap") if isinstance(indicators, dict) else None)
    rsi = _safe_float(indicators.get("rsi") if isinstance(indicators, dict) else None)
    relative_volume = _safe_float(signal.get("relative_volume"))

    trend = str(indicators.get("trend", "") if isinstance(indicators, dict) else "").lower()
    ema_trend = str(indicators.get("ema_trend", "") if isinstance(indicators, dict) else "").lower()
    macd_bias = str(
        indicators.get("macd_bias", indicators.get("macd", ""))
        if isinstance(indicators, dict) else ""
    ).lower()

    if current is not None and vwap is not None and vwap > 0:
        dist_pct = ((current - vwap) / vwap) * 100
        if dist_pct >= 0.15:
            score += 2
            reasons.append(f"price_above_vwap_{dist_pct:.2f}%")
        elif dist_pct <= -0.15:
            score -= 2
            reasons.append(f"price_below_vwap_{dist_pct:.2f}%")
        else:
            reasons.append("price_near_vwap")

    if trend == "bullish" or ema_trend == "bullish":
        score += 1
        reasons.append("bullish_trend")
    elif trend == "bearish" or ema_trend == "bearish":
        score -= 1
        reasons.append("bearish_trend")

    if "bull" in macd_bias:
        score += 1
        reasons.append("bullish_macd")
    elif "bear" in macd_bias:
        score -= 1
        reasons.append("bearish_macd")

    if rsi is not None:
        if 50 <= rsi <= 70:
            score += 1
            reasons.append(f"constructive_rsi_{rsi:.1f}")
        elif 30 <= rsi <= 50:
            score -= 1
            reasons.append(f"soft_rsi_{rsi:.1f}")
        elif rsi > 78:
            score -= 1
            reasons.append(f"overextended_rsi_{rsi:.1f}")
        elif rsi < 22:
            score += 1
            reasons.append(f"capitulation_rsi_{rsi:.1f}")

    if change_pct is not None:
        if change_pct >= 0.35:
            score += 1
            reasons.append(f"positive_change_{change_pct:.2f}%")
        elif change_pct <= -0.35:
            score -= 1
            reasons.append(f"negative_change_{change_pct:.2f}%")

    if relative_volume is not None:
        if relative_volume >= 1.5:
            reasons.append(f"relative_volume_confirming_{relative_volume:.2f}x")
        elif relative_volume < 0.7:
            reasons.append(f"thin_relative_volume_{relative_volume:.2f}x")

    if score >= 3:
        deterministic_bias = "LONG"
    elif score <= -3:
        deterministic_bias = "SHORT"
    else:
        deterministic_bias = "HOLD"

    conflict = (
        signal_direction == "HOLD"
        and deterministic_bias in {"LONG", "SHORT"}
    ) or (
        signal_direction in {"LONG", "SHORT"}
        and deterministic_bias in {"LONG", "SHORT"}
        and signal_direction != deterministic_bias
    )

    return {
        "bias": deterministic_bias,
        "score": score,
        "reasons": reasons,
        "llm_signal": signal_direction,
        "conflict": conflict,
    }


def enrich_signal_with_quote_context(signal: dict, quote: dict, candles: dict) -> dict:
    """
    Inject structured quote fields into the analyst signal for downstream gates.

    Persists: current_price, quote_timestamp, session_date, day_open, day_high,
    day_low, prior_day_high, prior_day_low, prev_close, change_pct, and
    relative_volume (when computable).

    These fields are required by the Catalyst Specificity Gate's confirmation
    scoring to have discrimination power.

    Args:
        signal: The LLM-generated analyst signal dict (modified in place).
        quote: The quote dict from FinnhubClient.get_quote().
        candles: The candle data dict from FinnhubClient.get_candles().

    Returns:
        The enriched signal dict (same reference as input).
    """
    session_levels = _session_levels_from_candles(candles)

    # Persist structured quote fields. Prefer candle-derived session levels for
    # day_open/day_high/day_low because quote APIs can be ambiguous around
    # premarket/regular-session boundaries.
    if quote:
        signal["current_price"] = quote.get("price")
        signal["quote_timestamp"] = quote.get("timestamp")
        signal["day_open"] = session_levels.get("day_open", quote.get("open"))
        signal["day_high"] = session_levels.get("day_high", quote.get("high"))
        signal["day_low"] = session_levels.get("day_low", quote.get("low"))
        signal["prev_close"] = quote.get("prev_close")
        signal["change_pct"] = quote.get("change_pct")

    for key in ("session_date", "prior_day_high", "prior_day_low", "prior_day_close"):
        if key in session_levels:
            signal[key] = session_levels[key]

    # Keep key_levels honest too; the LLM often maps quote day high/low into
    # prior_high/prior_low unless we overwrite with deterministic candle levels.
    key_levels = signal.get("key_levels")
    if isinstance(key_levels, dict):
        if "prior_day_high" in session_levels:
            key_levels["prior_high"] = session_levels["prior_day_high"]
        if "prior_day_low" in session_levels:
            key_levels["prior_low"] = session_levels["prior_day_low"]
        if "day_high" in session_levels:
            key_levels["day_high"] = session_levels["day_high"]
        if "day_low" in session_levels:
            key_levels["day_low"] = session_levels["day_low"]

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
            quote = fh.get_quote(sym)
            candles = fh.get_candles(sym, resolution="5", days=2)
            indicators = compute_indicators(candles)
            validate_candle_indicator_alignment(sym, quote, candles, indicators)
            try:
                multitimeframe_context = build_multitimeframe_context(
                    sym,
                    fh,
                    candles_5m=candles,
                    indicators_5m=indicators,
                    breadth_symbols=symbols,
                )
            except Exception as e:
                log.warning("Multi-timeframe context failed for %s: %s", sym, e)
                multitimeframe_context = {
                    "symbol": sym,
                    "errors": [f"context_build_failed:{e}"],
                }
            multitimeframe_prompt_context = format_multitimeframe_context_for_prompt(
                multitimeframe_context
            )
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

            precheck_signal = {"signal": "HOLD"}
            enrich_signal_with_quote_context(precheck_signal, quote, candles)
            deterministic_precheck = compute_deterministic_signal_sanity(
                precheck_signal, quote, indicators
            )
            deterministic_precheck_context = build_deterministic_sanity_prompt_context(
                deterministic_precheck
            )

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

{multitimeframe_prompt_context}

DETERMINISTIC TECHNICAL SANITY CHECK:
{deterministic_precheck_context}

{ANALYST_CONTEXT_BOUNDARY}

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
            signal["multitimeframe_context"] = multitimeframe_context
            signal = sanitize_analyst_key_levels(signal, quote, indicators)
            signal = apply_signal_mitigation(signal, active_mitigations)
            validation_result = validate_setup_for_symbol(sym, signal.get("setup_type"))
            signal.update(validation_result)
            signal = annotate_unregistered_setup(signal, valid_setups)

            # Enrich signal with structured quote context for downstream gates,
            # then sanitize again because enrichment can overwrite key/session
            # levels with provider-derived values.
            enrich_signal_with_quote_context(signal, quote, candles)
            signal = sanitize_analyst_session_levels(signal, quote)
            signal = sanitize_analyst_key_levels(signal, quote, indicators)
            signal["deterministic_sanity"] = compute_deterministic_signal_sanity(
                signal, quote, indicators
            )
            signal["trigger_status"] = compute_trigger_status(signal, quote, indicators)
            signal = enforce_veto_accountability(signal)
            signal = repair_missing_veto_contract(signal, sym)
            signal = sanitize_historical_feedback_bleed(signal)
            if signal["deterministic_sanity"].get("conflict"):
                log.warning(
                    "Analyst sanity conflict for %s: llm=%s deterministic=%s score=%s veto_required=%s veto_present=%s reasons=%s",
                    sym,
                    signal["deterministic_sanity"].get("llm_signal"),
                    signal["deterministic_sanity"].get("bias"),
                    signal["deterministic_sanity"].get("score"),
                    signal.get("llm_veto_required"),
                    signal.get("llm_veto_present"),
                    signal["deterministic_sanity"].get("reasons"),
                )

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
