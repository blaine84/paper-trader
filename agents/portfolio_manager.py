"""
Portfolio Manager Agent
Each profile (conservative/moderate/aggressive) runs with its own isolated portfolio.
Scout provides symbols only — all entry/exit decisions are made here.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from utils.finnhub_client import FinnhubClient
from utils.llm import call_llm, parse_json_response
from db.schema import AgentMemory, Position, Balance, Trade, get_session
from utils.trade_events import log_trade_event
from models.pm_profiles import PM_PROFILES, ACTIVE_PROFILES
from utils.case_library import get_relevant_cases, format_cases_for_prompt, get_win_rate_by_setup
from agents.quant_researcher import build_strategy_context
from agents.lesson_registry import check_track_record
from core.similarity import find_similar_cases, compute_similarity_stats
from core.edge_score import (
    compute_edge_score, check_hard_rejection, cap_position_size,
    confluence_score, similarity_quality,
)
from core.portfolio_risk import (
    validate_portfolio_risk, compute_portfolio_risk, adaptive_risk_throttle,
)

log = logging.getLogger(__name__)


def _coerce_price(value, field_name: str, symbol: str) -> float | None:
    """Normalize LLM/database price values before persisting or %.2f logging."""
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.strip().replace("$", "").replace(",", "")
        return float(value)
    except (TypeError, ValueError):
        log.warning(
            "Ignoring invalid %s for %s from maintenance review: %r",
            field_name, symbol, value,
        )
        return None

# Max minutes after market open (9:30 AM ET) that each setup type may be entered.
# Setup types NOT listed here have no entry-window restriction.
ENTRY_WINDOW_LIMITS = {
    "gap_and_go": 60,
    "orb": 60,
    "momentum_fade": 60,
    "short_squeeze": 60,
}

# Hard guardrails from 2026-04-30 AMD review. High-WR, fast intraday
# setups need enough breathing room; otherwise execution noise negates the edge.
HIGH_WR_STOP_BUFFER_THRESHOLD = 0.60
MIN_HIGH_WR_INTRADAY_STOP_BUFFER_PCT = 0.015
HIGH_MOMENTUM_ASSETS = {"AMD", "NVDA", "TSLA"}
HIGH_MOMENTUM_COOLDOWN_MINUTES = 30
_FAST_INTRADAY_SETUPS = {
    "gap_and_go",
    "vwap_reclaim",
    "orb",
    "momentum_fade",
    "trend_pullback",
    "news_catalyst",
    "short_squeeze",
}


def _extract_minutes(value) -> int | None:
    """Best-effort conversion of timeframe/horizon fields into minutes."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)

    text = str(value).lower()
    if any(token in text for token in ("swing", "overnight", "multi-day", "multiday")):
        return None

    import re
    minute_values = [int(n) for n in re.findall(r"(\d+)\s*(?:m|min|mins|minute|minutes)\b", text)]
    if minute_values:
        return max(minute_values)

    hour_values = [int(n) * 60 for n in re.findall(r"(\d+)\s*(?:h|hr|hrs|hour|hours)\b", text)]
    if hour_values:
        return max(hour_values)

    return None


def _is_fast_intraday_trade(decision: dict, signal: dict) -> bool:
    """Return True when the trade is clearly intraday and intended under ~60 minutes."""
    setup_type = (
        decision.get("setup_type") or decision.get("setup")
        or signal.get("setup_type") or signal.get("setup") or ""
    )
    setup_type = str(setup_type).lower()

    for key in (
        "holding_minutes", "expected_holding_minutes", "max_holding_minutes",
        "time_horizon_minutes", "duration_minutes",
    ):
        minutes = _extract_minutes(decision.get(key) or signal.get(key))
        if minutes is not None:
            return minutes <= 60

    timeframe = (
        decision.get("timeframe") or decision.get("time_horizon") or decision.get("duration")
        or signal.get("timeframe") or signal.get("time_horizon") or signal.get("duration")
    )
    minutes = _extract_minutes(timeframe)
    if minutes is not None:
        return minutes <= 60

    text = str(timeframe or "").lower()
    if "intraday" in text and not any(token in text for token in ("multi-day", "multiday", "swing")):
        return True

    return setup_type in _FAST_INTRADAY_SETUPS


def _apply_high_wr_stop_buffer(
    action: str,
    price: float,
    stop: float | None,
    decision: dict,
    signal: dict,
    case_stats: dict,
) -> float | None:
    """Widen too-tight stops for high-win-rate, fast intraday setups."""
    if action not in ("BUY", "SHORT") or not price or not stop:
        return stop

    win_rate = float(case_stats.get("win_rate") or 0.0)
    if win_rate <= HIGH_WR_STOP_BUFFER_THRESHOLD:
        return stop
    if not _is_fast_intraday_trade(decision, signal):
        return stop

    min_distance = price * MIN_HIGH_WR_INTRADAY_STOP_BUFFER_PCT
    if action == "BUY":
        min_stop = round(price - min_distance, 2)
        if stop > min_stop:
            log.warning(
                "Stop buffer enforced for %s: high-WR setup %.0f%% requires >=%.1f%% breathing room; %.2f → %.2f",
                decision.get("symbol", "?"), win_rate * 100,
                MIN_HIGH_WR_INTRADAY_STOP_BUFFER_PCT * 100, stop, min_stop,
            )
            return min_stop
    else:
        min_stop = round(price + min_distance, 2)
        if stop < min_stop:
            log.warning(
                "Stop buffer enforced for %s: high-WR setup %.0f%% requires >=%.1f%% breathing room; %.2f → %.2f",
                decision.get("symbol", "?"), win_rate * 100,
                MIN_HIGH_WR_INTRADAY_STOP_BUFFER_PCT * 100, stop, min_stop,
            )
            return min_stop

    return stop


def _high_momentum_cooldown_message(db, symbol: str) -> str | None:
    """Block fresh entries shortly after a stop/loss on volatile names across profiles."""
    symbol = (symbol or "").upper()
    if symbol not in HIGH_MOMENTUM_ASSETS:
        return None

    cutoff = datetime.utcnow() - timedelta(minutes=HIGH_MOMENTUM_COOLDOWN_MINUTES)
    recent = (
        db.query(Trade)
        .filter_by(symbol=symbol, status="closed")
        .filter(Trade.exit_time >= cutoff)
        .order_by(Trade.exit_time.desc())
        .first()
    )
    if not recent:
        return None

    reason = (recent.reason_exit or "").lower()
    stopped_or_loss = "stop" in reason or (recent.pnl is not None and recent.pnl < 0)
    if not stopped_or_loss:
        return None

    return (
        f"Cooldown active for {symbol}: recent stop/loss in {recent.profile} profile "
        f"within {HIGH_MOMENTUM_COOLDOWN_MINUTES} minutes — blocking cascading re-entry"
    )


SYSTEM_PROMPT_TEMPLATE = """You are a portfolio manager for paper day trading.
Profile: {profile_name} {emoji}
{personality}

You receive analyst signals (LONG/SHORT/HOLD with entry, stop, target),
your current positions, cash balance, and reviewer feedback.

The Analyst tells you: direction, setup quality, key levels, invalidation, confidence.
The Analyst does NOT tell you how to trade. That is entirely your job.

Given the Analyst's read, you decide:
  - Whether to act at all (maybe the setup is right but timing is wrong for your profile)
  - Action: BUY / SHORT / CLOSE / pass
  - Entry price (based on key levels — don't just use current price blindly)
  - Stop placement (use invalidation level + your profile's risk tolerance)
  - Target (based on key levels and your profile's R:R requirements)
  - Position size (based on your profile's max position % and stop distance)
  - Scale in / scale out if appropriate for your profile

Your constraints:
- Max positions: {max_positions}
- Max position size: {max_position_pct}% of portfolio
- Minimum R:R ratio: {min_risk_reward}:1
- Min signal strength to act: {min_signal_strength}
- Daily loss limit: {max_daily_loss_pct}%
- Avoid first {avoid_first_minutes} min and last {avoid_last_minutes} min of session

Decide which trades to make. For each:
{{
  "decisions": [
    {{
      "symbol": "SPY",
      "action": "BUY|SHORT|CLOSE|HOLD",
      "quantity": 10,
      "price": 450.00,
      "stop_loss": 447.00,
      "target": 455.00,
      "rationale": "why you're doing this given your risk profile"
    }}
  ],
  "portfolio_notes": "your overall thinking this cycle"
}}

action=BUY   — enter or add to a long position
action=SHORT — enter or add to a short position (paper margin reserved)
action=CLOSE — exit an existing long or short position
HOLD decisions don't need to be listed — only include actionable trades.
If no trades make sense for your profile, return empty decisions array.
"""


MAINTENANCE_REVIEW_PROMPT = """You are a portfolio manager performing a routine maintenance review of open positions.
Your job is to decide whether to HOLD, TIGHTEN STOP, RAISE TARGET, or TRIM a partial position.
You CANNOT close a position in this review. Closing requires a separate Reversal/Close Review triggered by thesis invalidation.

Profile: {profile_name} {emoji}

ENTRY CONTRACT (the original trade thesis — your anchor):
  Thesis: {thesis}
  Setup Type: {setup_type}
  Entry Price: ${entry_price}
  Stop Price: ${stop_price}
  Target Price: ${target_price}
  Invalidators: {invalidators}

CURRENT POSITION:
  Symbol: {symbol}
  Side: {side}
  Quantity: {quantity}
  Current Price: ${current_price}
  Unrealized P&L: {unrealized_pnl_pct}%
  DRIFTING: {drifting} (no recent analyst signal since entry)

CURRENT INDICATORS:
{indicators_text}

ADVISORY ANALYST SIGNALS (informational only — do NOT override the Entry Contract):
{advisory_signals_text}

POSITION HEALTH ASSESSMENT:
{health_text}

RULES:
1. The Entry Contract thesis is your anchor. Hold unless there is a clear reason to adjust.
2. You CANNOT produce a close action. Only Reversal/Close Review can close positions.
3. If the position is DRIFTING, that alone is NOT a reason to act. Evaluate against the Entry Contract.
4. Tighten stop only if the position has moved favorably and you want to lock in gains.
5. Raise target only if new evidence supports a higher target while the thesis remains intact.
6. Trim partial only if the position is significantly profitable and you want to reduce risk.

Respond with JSON only:
{{
  "reviews": [
    {{
      "symbol": "{symbol}",
      "action": "hold|tighten_stop|raise_target|trim_partial",
      "new_stop": null,
      "new_target": null,
      "trim_pct": null,
      "reasoning": "why you chose this action, referencing the Entry Contract thesis"
    }}
  ],
  "notes": "overall maintenance review summary"
}}

VALID ACTIONS: hold, tighten_stop, raise_target, trim_partial
DO NOT use close, close_full, close_partial, or CLOSE.
"""

VALID_MAINTENANCE_ACTIONS = {"hold", "tighten_stop", "raise_target", "trim_partial"}


REVERSAL_CLOSE_PROMPT = """You are a portfolio manager performing a Reversal/Close Review.
A specific trigger has fired that may warrant closing this position. Your job is to evaluate
whether the original trade thesis is truly broken and decide whether to CLOSE FULL, CLOSE PARTIAL,
or HOLD with a tightened stop.

Profile: {profile_name} {emoji}

TRIGGER THAT CAUSED THIS REVIEW:
  Trigger Type: {trigger_type}
  Trigger Details: {trigger_details}

ENTRY CONTRACT (the original trade thesis — your anchor):
  Thesis: {thesis}
  Setup Type: {setup_type}
  Entry Price: ${entry_price}
  Stop Price: ${stop_price}
  Target Price: ${target_price}
  Invalidators: {invalidators}

CURRENT POSITION:
  Symbol: {symbol}
  Side: {side}
  Quantity: {quantity}
  Current Price: ${current_price}
  Unrealized P&L: {unrealized_pnl_pct}%

CURRENT MARKET CONDITIONS:
{market_conditions_text}

OPPOSING EVIDENCE:
{opposing_evidence_text}

RULES:
1. This review was triggered by: {trigger_type}. Evaluate whether the thesis is truly broken.
2. If the thesis is clearly invalidated (e.g., key level lost on confirmed close), close the position.
3. If the evidence is ambiguous, hold with a tightened stop to protect capital while giving the trade room.
4. close_partial is appropriate when some thesis elements are broken but others remain intact.
5. hold_tighten means: tighten stop to breakeven if profitable, otherwise hold current stop.

Respond with JSON only:
{{
  "symbol": "{symbol}",
  "action": "close_full|close_partial|hold_tighten",
  "reasoning": "why you chose this action, referencing the Entry Contract thesis and the trigger",
  "trigger": "{trigger_type}",
  "invalidator": {invalidator_json}
}}

VALID ACTIONS: close_full, close_partial, hold_tighten
"""

VALID_REVERSAL_ACTIONS = {"close_full", "close_partial", "hold_tighten"}


def run_maintenance_review(position_data: dict, profile: dict, tier: str = "high") -> dict:
    """
    Run a Maintenance Review for a single open position.

    Accepts position data (with Entry Contract, current price, indicators,
    advisory signals, health data, drifting state), formats the prompt,
    calls the LLM, validates the action, and returns the parsed review result.

    On LLM failure, defaults to "hold" (no action taken).

    Args:
        position_data: dict with keys:
            symbol, side, quantity, entry_price, stop_price, target_price,
            current_price, unrealized_pnl_pct, drifting,
            thesis, setup_type, invalidators,
            indicators, advisory_signals, health_text
        profile: PM profile dict from PM_PROFILES
        tier: LLM tier to use (default "high")

    Returns:
        dict with keys: symbol, action, new_stop, new_target, trim_pct, reasoning
    """
    symbol = position_data.get("symbol", "UNKNOWN")

    # Format invalidators for display
    invalidators_raw = position_data.get("invalidators")
    if isinstance(invalidators_raw, str):
        try:
            invalidators_list = json.loads(invalidators_raw)
        except (json.JSONDecodeError, TypeError):
            invalidators_list = []
    elif isinstance(invalidators_raw, list):
        invalidators_list = invalidators_raw
    else:
        invalidators_list = []
    invalidators_text = json.dumps(invalidators_list, indent=2) if invalidators_list else "None"

    # Format indicators
    indicators = position_data.get("indicators")
    if isinstance(indicators, dict):
        indicators_text = json.dumps(indicators, indent=2)
    elif isinstance(indicators, str):
        indicators_text = indicators
    else:
        indicators_text = "No indicator data available"

    # Format advisory signals
    advisory_signals = position_data.get("advisory_signals")
    if isinstance(advisory_signals, dict):
        advisory_signals_text = json.dumps(advisory_signals, indent=2)
    elif isinstance(advisory_signals, str):
        advisory_signals_text = advisory_signals
    else:
        advisory_signals_text = "No advisory signals available"

    # Format health text
    health_text = position_data.get("health_text")
    if not health_text:
        health_text = "No health assessment available"

    prompt = MAINTENANCE_REVIEW_PROMPT.format(
        profile_name=profile.get("name", "Unknown"),
        emoji=profile.get("emoji", ""),
        thesis=position_data.get("thesis") or "No thesis recorded",
        setup_type=position_data.get("setup_type") or "unknown",
        entry_price=position_data.get("entry_price") or "N/A",
        stop_price=position_data.get("stop_price") or "N/A",
        target_price=position_data.get("target_price") or "N/A",
        invalidators=invalidators_text,
        symbol=symbol,
        side=position_data.get("side") or "unknown",
        quantity=position_data.get("quantity") or 0,
        current_price=position_data.get("current_price") or "N/A",
        unrealized_pnl_pct=position_data.get("unrealized_pnl_pct") or 0,
        drifting="YES" if position_data.get("drifting") else "NO",
        indicators_text=indicators_text,
        advisory_signals_text=advisory_signals_text,
        health_text=health_text,
    )

    system_prompt = (
        "You are a portfolio manager performing routine maintenance reviews. "
        "Respond with valid JSON only. No markdown, no explanation outside the JSON."
    )

    # Default result on failure
    default_result = {
        "symbol": symbol,
        "action": "hold",
        "new_stop": None,
        "new_target": None,
        "trim_pct": None,
        "reasoning": "LLM review failed — defaulting to hold (no action taken)",
    }

    try:
        raw = call_llm(system_prompt, prompt, json_mode=True, tier=tier)
        result = parse_json_response(raw)
    except Exception as exc:
        log.error("Maintenance Review LLM call failed for %s: %s", symbol, exc)
        return default_result

    # Extract the review for this symbol from the reviews array
    reviews = result.get("reviews", [])
    review = None
    for r in reviews:
        if r.get("symbol") == symbol:
            review = r
            break
    # If no matching symbol found, use the first review or default
    if review is None:
        review = reviews[0] if reviews else {}

    # Validate the action
    action = review.get("action", "hold")
    if action not in VALID_MAINTENANCE_ACTIONS:
        log.warning(
            "Maintenance Review returned invalid action '%s' for %s. Defaulting to hold.",
            action, symbol,
        )
        action = "hold"

    return {
        "symbol": symbol,
        "action": action,
        "new_stop": review.get("new_stop"),
        "new_target": review.get("new_target"),
        "trim_pct": review.get("trim_pct"),
        "reasoning": review.get("reasoning") or "No reasoning provided",
    }


def run_reversal_close_review(position_data: dict, trigger_info: dict, profile: dict, tier: str = "high") -> dict:
    """
    Run a Reversal/Close Review for a single open position.

    Only invoked when a trigger fires (thesis_invalidation, opposing signal,
    explicit CLOSE). Evaluates the Entry Contract thesis against current
    market conditions and produces one of: close_full, close_partial,
    or hold_tighten.

    On LLM failure, defaults to "hold_tighten" (tighten stop to breakeven
    if profitable, otherwise hold).

    Args:
        position_data: dict with keys:
            symbol, side, quantity, entry_price, stop_price, target_price,
            current_price, unrealized_pnl_pct,
            thesis, setup_type, invalidators,
            market_conditions, opposing_evidence
        trigger_info: dict with keys:
            type: str — one of "thesis_invalidation", "opposing_signal", "explicit_close"
            details: str — human-readable description of the trigger
            invalidator: dict | None — the specific invalidator that was breached (if applicable)
        profile: PM profile dict from PM_PROFILES
        tier: LLM tier to use (default "high")

    Returns:
        dict with keys: symbol, action, reasoning, trigger, invalidator
    """
    symbol = position_data.get("symbol", "UNKNOWN")
    trigger_type = trigger_info.get("type", "unknown")
    trigger_details = trigger_info.get("details", "No details available")
    trigger_invalidator = trigger_info.get("invalidator")

    # Log the specific trigger that caused this review
    log.info(
        "Reversal/Close Review triggered for %s: trigger=%s, details=%s",
        symbol, trigger_type, trigger_details,
    )

    # Format invalidators for display
    invalidators_raw = position_data.get("invalidators")
    if isinstance(invalidators_raw, str):
        try:
            invalidators_list = json.loads(invalidators_raw)
        except (json.JSONDecodeError, TypeError):
            invalidators_list = []
    elif isinstance(invalidators_raw, list):
        invalidators_list = invalidators_raw
    else:
        invalidators_list = []
    invalidators_text = json.dumps(invalidators_list, indent=2) if invalidators_list else "None"

    # Format market conditions
    market_conditions = position_data.get("market_conditions")
    if isinstance(market_conditions, dict):
        market_conditions_text = json.dumps(market_conditions, indent=2)
    elif isinstance(market_conditions, str):
        market_conditions_text = market_conditions
    else:
        market_conditions_text = "No market condition data available"

    # Format opposing evidence
    opposing_evidence = position_data.get("opposing_evidence")
    if isinstance(opposing_evidence, dict):
        opposing_evidence_text = json.dumps(opposing_evidence, indent=2)
    elif isinstance(opposing_evidence, str):
        opposing_evidence_text = opposing_evidence
    else:
        opposing_evidence_text = "No opposing evidence available"

    # Format the trigger invalidator for the prompt
    if trigger_invalidator and isinstance(trigger_invalidator, dict):
        invalidator_json = json.dumps(trigger_invalidator)
    else:
        invalidator_json = "null"

    prompt = REVERSAL_CLOSE_PROMPT.format(
        profile_name=profile.get("name", "Unknown"),
        emoji=profile.get("emoji", ""),
        trigger_type=trigger_type,
        trigger_details=trigger_details,
        thesis=position_data.get("thesis") or "No thesis recorded",
        setup_type=position_data.get("setup_type") or "unknown",
        entry_price=position_data.get("entry_price") or "N/A",
        stop_price=position_data.get("stop_price") or "N/A",
        target_price=position_data.get("target_price") or "N/A",
        invalidators=invalidators_text,
        symbol=symbol,
        side=position_data.get("side") or "unknown",
        quantity=position_data.get("quantity") or 0,
        current_price=position_data.get("current_price") or "N/A",
        unrealized_pnl_pct=position_data.get("unrealized_pnl_pct") or 0,
        market_conditions_text=market_conditions_text,
        opposing_evidence_text=opposing_evidence_text,
        invalidator_json=invalidator_json,
    )

    system_prompt = (
        "You are a portfolio manager performing a Reversal/Close Review. "
        "Respond with valid JSON only. No markdown, no explanation outside the JSON."
    )

    # Default result on failure — hold_tighten is conservative
    default_result = {
        "symbol": symbol,
        "action": "hold_tighten",
        "reasoning": "LLM review failed — defaulting to hold_tighten (tighten stop to breakeven if profitable)",
        "trigger": trigger_type,
        "invalidator": trigger_invalidator,
    }

    try:
        raw = call_llm(system_prompt, prompt, json_mode=True, tier=tier)
        result = parse_json_response(raw)
    except Exception as exc:
        log.error("Reversal/Close Review LLM call failed for %s: %s", symbol, exc)
        return default_result

    # Validate the action
    action = result.get("action", "hold_tighten")
    if action not in VALID_REVERSAL_ACTIONS:
        log.warning(
            "Reversal/Close Review returned invalid action '%s' for %s. Defaulting to hold_tighten.",
            action, symbol,
        )
        action = "hold_tighten"

    return {
        "symbol": symbol,
        "action": action,
        "reasoning": result.get("reasoning") or "No reasoning provided",
        "trigger": result.get("trigger") or trigger_type,
        "invalidator": result.get("invalidator") or trigger_invalidator,
    }


def get_portfolio_for_profile(db, fh, profile_id: str) -> dict:
    """Build portfolio snapshot for a specific profile."""
    positions = db.query(Position).filter_by(profile=profile_id).all()
    pos_data = []
    total_pos_value = 0.0

    for p in positions:
        try:
            quote = fh.get_quote(p.symbol)
            price = quote["price"]
        except Exception:
            price = p.avg_cost
        market_value = p.quantity * price
        if p.side == "short":
            unrealized_pnl = (p.avg_cost - price) * p.quantity
        else:
            unrealized_pnl = (price - p.avg_cost) * p.quantity
        total_pos_value += market_value

        # Get stop/target from the open trade
        open_trade = (
            db.query(Trade)
            .filter_by(symbol=p.symbol, profile=profile_id, status="open")
            .order_by(Trade.entry_time.desc())
            .first()
        )

        # Detect DRIFTING state for this position
        drifting = detect_drifting(db, open_trade) if open_trade else True

        pos_data.append({
            "symbol": p.symbol,
            "side": p.side,
            "quantity": p.quantity,
            "avg_cost": p.avg_cost,
            "current_price": price,
            "market_value": round(market_value, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "unrealized_pnl_pct": round(unrealized_pnl / (p.avg_cost * p.quantity) * 100, 2) if p.avg_cost and p.quantity else 0,
            "stop_price": open_trade.stop_price if open_trade else None,
            "target_price": open_trade.target_price if open_trade else None,
            "entry_time": open_trade.entry_time.isoformat() if open_trade and open_trade.entry_time else None,
            "drifting": drifting,
        })

    bal = (
        db.query(Balance)
        .filter_by(profile=profile_id)
        .order_by(Balance.timestamp.desc())
        .first()
    )
    starting = PM_PROFILES[profile_id]["starting_balance"]
    cash = bal.cash if bal else float(starting)
    total_equity = cash + total_pos_value

    # Today's realized P&L
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0)
    today_trades = (
        db.query(Trade)
        .filter_by(profile=profile_id, status="closed")
        .filter(Trade.exit_time >= today_start)
        .all()
    )
    daily_pnl = sum(t.pnl or 0 for t in today_trades)

    return {
        "profile": profile_id,
        "cash": round(cash, 2),
        "positions": pos_data,
        "total_equity": round(total_equity, 2),
        "position_count": len(pos_data),
        "daily_pnl": round(daily_pnl, 2),
        "daily_pnl_pct": round(daily_pnl / starting * 100, 2),
        "starting_balance": starting,
    }


def _count_recent_consecutive_losses(db, profile_id: str) -> int:
    """Count consecutive recent losing trades for a profile (most recent first)."""
    recent_trades = (
        db.query(Trade)
        .filter_by(profile=profile_id, status="closed")
        .order_by(Trade.exit_time.desc())
        .limit(20)
        .all()
    )
    count = 0
    for t in recent_trades:
        if t.pnl is not None and t.pnl < 0:
            count += 1
        else:
            break
    return count


def _build_signal_for_symbol(db, symbol: str, decision: dict) -> dict:
    """Build a signal dict for the similarity/edge score engines from analyst memory."""
    sig_mem = (
        db.query(AgentMemory)
        .filter_by(agent="analyst", symbol=symbol, key="signal")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    if sig_mem:
        try:
            return json.loads(sig_mem.value)
        except Exception:
            pass
    # Fallback: build minimal signal from decision fields
    return {
        "setup_type": decision.get("setup_type") or decision.get("setup") or "",
        "market_regime": decision.get("market_regime") or decision.get("regime") or "",
        "strength": decision.get("strength") or "moderate",
        "confidence": decision.get("confidence") or "medium",
        "bias": "LONG" if decision.get("action") == "BUY" else "SHORT",
        "indicators": {},
    }


def _build_case_stats(db, setup_type: str, market_regime: str = None) -> dict:
    """Build case_stats dict from the case library for edge score computation."""
    from utils.trade_validator import adjust_confidence
    conf_result = adjust_confidence(db.bind, setup_type, market_regime)
    return {
        "win_rate": conf_result.get("win_rate") or 0.0,
        "sample_size": conf_result.get("total_cases", 0),
    }


# Strength ordering for opposing evidence threshold comparison (Req 7.5)
# Higher value = stronger signal. A signal "meets" a threshold when its
# numeric strength >= the threshold's numeric strength.
STRENGTH_ORDER = {"weak": 1, "moderate": 2, "strong": 3}


def _meets_threshold(signal_strength: str, threshold: str) -> bool:
    """Return True if signal_strength meets or exceeds the opposing_evidence_threshold."""
    sig_val = STRENGTH_ORDER.get(str(signal_strength).lower(), 0)
    thr_val = STRENGTH_ORDER.get(str(threshold).lower(), 0)
    return sig_val >= thr_val


def _check_reversal_triggers(
    db, trade, position_data: dict, signal: dict | None, profile: dict
) -> dict | None:
    """
    Check whether a position has any Reversal/Close Review triggers.

    Returns a trigger_info dict if a trigger is found, or None if the
    position should go to Maintenance Review.

    Trigger types:
      - thesis_invalidation: Price Monitor detected an invalidator breach
      - opposing_signal: Analyst signal contradicts Entry Contract direction
        and meets the profile's opposing_evidence_threshold
      - explicit_close: Analyst signal contains an explicit CLOSE action
    """
    symbol = trade.symbol

    # 1. Check AgentMemory for thesis_invalidation triggers from Price Monitor
    invalidation_mem = (
        db.query(AgentMemory)
        .filter_by(agent="price_monitor", symbol=symbol, key="thesis_invalidation")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    if invalidation_mem:
        try:
            inv_data = json.loads(invalidation_mem.value)
        except (json.JSONDecodeError, TypeError):
            inv_data = {}
        return {
            "type": "thesis_invalidation",
            "details": (
                f"Price Monitor detected thesis invalidation for {symbol}: "
                f"{inv_data.get('invalidator', 'unknown invalidator')}"
            ),
            "invalidator": inv_data.get("invalidator"),
        }

    if not signal:
        return None

    # 2. Check for explicit CLOSE signal
    signal_action = str(signal.get("signal", "") or signal.get("action", "")).upper()
    if signal_action == "CLOSE":
        return {
            "type": "explicit_close",
            "details": f"Analyst issued explicit CLOSE signal for {symbol}",
            "invalidator": None,
        }

    # 3. Check for opposing signal that meets the profile threshold
    signal_bias = str(signal.get("bias", "")).upper()
    trade_direction = str(trade.direction).upper()  # LONG or SHORT

    # Determine if the signal opposes the position direction
    is_opposing = (
        (trade_direction == "LONG" and signal_bias == "SHORT")
        or (trade_direction == "SHORT" and signal_bias == "LONG")
    )

    if is_opposing:
        signal_strength = str(signal.get("strength", "")).lower()
        threshold = profile.get("opposing_evidence_threshold", "strong")
        if _meets_threshold(signal_strength, threshold):
            return {
                "type": "opposing_signal",
                "details": (
                    f"Opposing {signal_bias} signal (strength={signal_strength}) "
                    f"for {symbol} meets {threshold} threshold"
                ),
                "invalidator": None,
            }

    return None


VALID_INVALIDATOR_TYPES = {"price_below_level", "price_above_level", "structure_break"}
VALID_CONFIRMATION_METHODS = {"tick", "5m_close"}


def _parse_invalidator(raw: dict) -> dict | None:
    """Parse and validate a single invalidator dict. Returns None if invalid."""
    if not isinstance(raw, dict):
        return None
    inv_type = raw.get("type", "")
    reference = raw.get("reference", "")
    confirmation = raw.get("confirmation", "5m_close")
    lookback_bars = raw.get("lookback_bars", 1)

    # Validate type
    if inv_type not in VALID_INVALIDATOR_TYPES:
        return None
    # Reference must be a non-empty string
    if not reference:
        return None
    reference = str(reference)
    # Validate confirmation
    if confirmation not in VALID_CONFIRMATION_METHODS:
        confirmation = "5m_close"
    # Validate lookback_bars
    try:
        lookback_bars = int(lookback_bars)
        if lookback_bars < 0:
            lookback_bars = 0
    except (TypeError, ValueError):
        lookback_bars = 1

    return {
        "type": inv_type,
        "reference": reference,
        "confirmation": confirmation,
        "lookback_bars": lookback_bars,
    }


def _default_invalidator(stop: float) -> dict:
    """Build a default stop-price-based invalidator."""
    return {
        "type": "price_below_level",
        "reference": str(stop),
        "confirmation": "5m_close",
        "lookback_bars": 1,
    }


def build_entry_contract(decision: dict, signal: dict, stop: float, target: float) -> dict:
    """
    Build an Entry Contract from a trade decision and analyst signal.

    Extracts thesis, setup_type, and structured invalidators.
    Falls back to a stop-price-based default invalidator when the signal
    lacks an invalidation field.

    Returns:
        {"thesis": str, "setup_type": str, "invalidators": list[dict]}
    """
    # --- Thesis ---
    rationale = decision.get("rationale") or ""
    signal_context_parts = []
    if signal.get("bias"):
        signal_context_parts.append(f"Bias: {signal['bias']}")
    if signal.get("confidence"):
        signal_context_parts.append(f"Confidence: {signal['confidence']}")
    if signal.get("setup_type") or signal.get("setup"):
        signal_context_parts.append(
            f"Setup: {signal.get('setup_type') or signal.get('setup')}"
        )
    if signal.get("key_levels"):
        signal_context_parts.append(f"Key levels: {signal['key_levels']}")

    signal_context = "; ".join(signal_context_parts)
    if rationale and signal_context:
        thesis = f"{rationale} [Signal context: {signal_context}]"
    elif rationale:
        thesis = rationale
    elif signal_context:
        thesis = f"[Signal context: {signal_context}]"
    else:
        thesis = "No thesis recorded"

    # --- Setup type ---
    setup_type = (
        signal.get("setup_type")
        or signal.get("setup")
        or decision.get("setup_type")
        or decision.get("setup")
        or "unknown"
    )

    # --- Invalidators ---
    invalidators = []
    invalidation_raw = signal.get("invalidation")

    if invalidation_raw:
        # Parse invalidation field — could be a list of dicts, a single dict,
        # or a string description
        parsed_any = False
        if isinstance(invalidation_raw, list):
            for item in invalidation_raw:
                inv = _parse_invalidator(item)
                if inv:
                    invalidators.append(inv)
                    parsed_any = True
        elif isinstance(invalidation_raw, dict):
            inv = _parse_invalidator(invalidation_raw)
            if inv:
                invalidators.append(inv)
                parsed_any = True
        # If invalidation was present but we couldn't parse any structured
        # invalidators from it, fall back to default
        if not parsed_any:
            log.warning(
                "Could not parse structured invalidators from signal invalidation "
                "field (value: %s). Falling back to stop-price default.",
                invalidation_raw,
            )
            invalidators.append(_default_invalidator(stop))
    else:
        # No invalidation field at all — use stop-price default
        log.warning(
            "Signal lacks invalidation field. Using stop-price default "
            "invalidator (stop=%.2f).",
            stop,
        )
        invalidators.append(_default_invalidator(stop))

    return {
        "thesis": thesis,
        "setup_type": setup_type,
        "invalidators": invalidators,
    }


def build_legacy_entry_contract(trade) -> dict | None:
    """
    Build a best-effort Entry Contract for a legacy trade that was opened
    before the thesis-anchored exits feature.

    Migration rules:
      - If trade.thesis is already populated → return None (no migration needed)
      - If trade.stop_price and trade.target_price exist → full Entry Contract
      - If only trade.stop_price exists → partial contract with stop-based invalidator
      - If neither exists → return None (fall back to signal-based evaluation)

    Logs a warning for each legacy trade migrated, identifying the trade
    and which fields were missing or inferred.

    Returns:
        dict with keys {"thesis", "setup_type", "invalidators"} or None
    """
    # Already has a thesis — no migration needed
    if trade.thesis:
        return None

    has_stop = trade.stop_price is not None
    has_target = trade.target_price is not None

    # Neither stop nor target — cannot construct a meaningful contract
    if not has_stop and not has_target:
        return None

    # Build thesis from reason_entry or use a default
    thesis = trade.reason_entry or "Legacy trade — no thesis recorded"
    setup_type = "unknown"

    # Build invalidator from stop price
    invalidators = []
    if has_stop:
        invalidators.append(_default_invalidator(trade.stop_price))

    # Determine what was missing/inferred for the log message
    missing_parts = []
    if not trade.reason_entry:
        missing_parts.append("reason_entry (used default thesis)")
    if not has_target:
        missing_parts.append("target_price")
    missing_parts.append("setup_type (inferred as 'unknown')")

    trade_id = getattr(trade, "id", "?")
    symbol = getattr(trade, "symbol", "?")
    log.warning(
        "Legacy trade migration: trade_id=%s symbol=%s — "
        "constructed %s Entry Contract. Missing/inferred: %s",
        trade_id,
        symbol,
        "full" if (has_stop and has_target) else "partial",
        ", ".join(missing_parts),
    )

    return {
        "thesis": thesis,
        "setup_type": setup_type,
        "invalidators": invalidators,
    }


def detect_drifting(db, trade) -> bool:
    """
    Detect whether a position is in DRIFTING state.

    A position is DRIFTING when no analyst signal for the trade's symbol
    has been recorded after the trade's entry_time. This is a computed
    state — not stored in the DB — to avoid stale state if signals arrive
    between cycles.

    Args:
        db: SQLAlchemy session
        trade: Trade record with .symbol and .entry_time

    Returns:
        True if no analyst signal exists after entry_time (drifting),
        False if a signal exists after entry_time (not drifting).
    """
    if not trade.entry_time:
        # No entry time recorded — treat as drifting (conservative)
        return True

    latest_signal = (
        db.query(AgentMemory)
        .filter_by(agent="analyst", symbol=trade.symbol, key="signal")
        .filter(AgentMemory.timestamp > trade.entry_time)
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )

    return latest_signal is None


def execute_trade(db, decision: dict, profile_id: str):
    """
    Apply a trade decision to the paper portfolio.

    Supported actions:
      BUY   — open or add to a long position
      SHORT — open or add to a short position
      CLOSE — close an existing long or short position
    """
    action = decision["action"]
    symbol = decision["symbol"]
    quantity = decision.get("quantity", 0)
    price = decision.get("price") or decision.get("entry_price") or 0
    if not price:
        return False, "No price in decision"

    # Sanity-check the LLM's price against a live quote.
    # Reject if the decision price deviates more than 5% from current market.
    try:
        fh = FinnhubClient()
        live_quote = fh.get_quote(symbol)
        live_price = live_quote.get("price", 0)
        if live_price and live_price > 0:
            deviation = abs(price - live_price) / live_price
            if deviation > 0.05:
                log.warning(
                    "Price sanity check failed for %s: LLM price=%.2f, "
                    "live price=%.2f (%.1f%% deviation). Using live price.",
                    symbol, price, live_price, deviation * 100,
                )
                price = live_price
    except Exception as exc:
        log.warning("Could not verify price for %s: %s", symbol, exc)

    # Extract stop/target from multiple possible keys the LLM might use
    stop = decision.get("stop") or decision.get("stop_price") or decision.get("stop_loss")
    target = decision.get("target") or decision.get("target_price") or decision.get("profit_target")

    # If stop/target are in rationale text but not fields, try to parse them out
    if not stop or not target:
        rationale = decision.get("rationale", "")
        import re
        if not stop:
            m = re.search(r'stop[:\s]*\$?([\d.]+)', rationale, re.IGNORECASE)
            if m:
                try: stop = float(m.group(1))
                except: pass
        if not target:
            m = re.search(r'target[:\s]*\$?([\d.]+)', rationale, re.IGNORECASE)
            if m:
                try: target = float(m.group(1))
                except: pass

    # If still no stop, derive from ATR or analyst key levels — never use flat %
    if action in ("BUY", "SHORT") and not stop and price:
        import logging
        _log = logging.getLogger(__name__)

        # Try 1: ATR-based stop (1.5x ATR from entry)
        try:
            from utils.technicals import compute_indicators
            fh = FinnhubClient()
            candles = fh.get_candles(symbol, resolution="5", days=2)
            indicators = compute_indicators(candles)
            atr = indicators.get("atr")
            if atr and atr > 0:
                if action == "BUY":
                    stop = round(price - (atr * 1.5), 2)
                else:
                    stop = round(price + (atr * 1.5), 2)
                _log.info(f"Stop derived from ATR ({atr:.2f} × 1.5) for {symbol}: {stop}")
        except Exception:
            pass

        # Try 2: Key level from analyst signal
        if not stop:
            try:
                sig_mem = (
                    db.query(AgentMemory)
                    .filter_by(agent="analyst", symbol=symbol, key="signal")
                    .order_by(AgentMemory.timestamp.desc())
                    .first()
                )
                if sig_mem:
                    import json as _json
                    sig = _json.loads(sig_mem.value)
                    levels = sig.get("key_levels", {})
                    if action == "BUY" and levels.get("support"):
                        stop = round(float(levels["support"]) * 0.995, 2)  # just below support
                        _log.info(f"Stop derived from support level for {symbol}: {stop}")
                    elif action == "SHORT" and levels.get("resistance"):
                        stop = round(float(levels["resistance"]) * 1.005, 2)  # just above resistance
                        _log.info(f"Stop derived from resistance level for {symbol}: {stop}")
            except Exception:
                pass

        # Try 3: Last resort — 2x ATR or 1.5% (whichever is available)
        if not stop:
            if action == "BUY":
                stop = round(price * 0.985, 2)
            else:
                stop = round(price * 1.015, 2)
            _log.warning(f"No ATR or key level for {symbol}, using 1.5% fallback: {stop}")

    if action in ("BUY", "SHORT", "CLOSE"):
        log_trade_event(
            db,
            "entry_requested" if action in ("BUY", "SHORT") else "exit_requested",
            agent=f"pm_{profile_id}",
            symbol=symbol,
            profile=profile_id,
            price=price,
            message=decision.get("rationale"),
            payload={
                "action": action,
                "quantity": quantity,
                "stop": stop,
                "target": target,
                "decision": decision,
            },
        )

    starting = PM_PROFILES[profile_id]["starting_balance"]
    bal = (
        db.query(Balance)
        .filter_by(profile=profile_id)
        .order_by(Balance.timestamp.desc())
        .first()
    )
    cash = bal.cash if bal else float(starting)

    # ── Tier-1 pre-validation: similarity → edge score → portfolio risk ──
    # Tracks edge data to store on the Trade record later
    _edge_data = {}

    if action in ("BUY", "SHORT"):
        base_quantity = quantity  # preserve original for cap calculation

        # --- Build signal context for this symbol ---
        signal_for_symbol = _build_signal_for_symbol(db, symbol, decision)

        # --- 1. Similarity engine (fail-open: proceed with zero stats on error) ---
        sim_stats = {
            "similarity_winrate": 0.0, "similarity_avg_r": 0.0,
            "sample_size": 0, "similarity_confidence": 0.0, "skip_similarity": True,
        }
        try:
            similar_cases = find_similar_cases(signal_for_symbol, db.bind)
            sim_stats = compute_similarity_stats(similar_cases)
        except Exception as exc:
            log.warning("Similarity engine error (proceeding with zero stats): %s", exc)

        # --- 2. Case stats from existing win rate data ---
        setup_type = decision.get("setup_type") or decision.get("setup") or signal_for_symbol.get("setup_type") or ""
        regime = decision.get("market_regime") or decision.get("regime") or signal_for_symbol.get("market_regime")
        case_stats = _build_case_stats(db, setup_type, regime)

        # High-WR fast intraday setups need mandatory breathing room. Enforce
        # before edge/validation so the persisted stop and Entry Contract agree.
        adjusted_stop = _apply_high_wr_stop_buffer(
            action, price, stop, decision, signal_for_symbol, case_stats
        )
        if adjusted_stop != stop:
            stop = adjusted_stop
            decision["stop"] = stop
            decision["stop_loss"] = stop

        cooldown_msg = _high_momentum_cooldown_message(db, symbol)
        if cooldown_msg:
            log.warning(cooldown_msg)
            return False, cooldown_msg

        # --- 3. Hard rejection check (fail-closed) ---
        try:
            if check_hard_rejection(case_stats):
                log.warning(
                    "DECISION: status=REJECTED reason=hard_rejection "
                    "setup_winrate=%.2f sample_size=%d",
                    case_stats["win_rate"], case_stats["sample_size"],
                )
                return False, (
                    f"Hard reject: setup winrate too low "
                    f"({case_stats['win_rate']:.2f} over {case_stats['sample_size']} cases)"
                )
        except Exception as exc:
            log.error("Hard rejection check failed (rejecting trade): %s", exc)
            return False, f"Edge score pre-check error: {exc}"

        # --- 4. Compute edge score (fail-closed) ---
        try:
            edge = compute_edge_score(signal_for_symbol, case_stats, sim_stats)
        except Exception as exc:
            log.error("Edge score computation failed (rejecting trade): %s", exc)
            return False, f"Edge score computation error: {exc}"

        # Compute sub-components for logging
        _confluence = confluence_score(
            signal_for_symbol.get("indicators", {}),
            signal_for_symbol.get("bias", ""),
        )
        _sim_qual = similarity_quality(sim_stats.get("sample_size", 0))

        # --- EDGE SCORE structured log ---
        log.info(
            "EDGE SCORE: %.3f | setup_winrate=%.2f (n=%d) | "
            "similarity_winrate=%.2f (n=%d) | similarity_confidence=%.2f | "
            "confluence=%.2f | similarity_quality=%.2f",
            edge,
            case_stats.get("win_rate", 0), case_stats.get("sample_size", 0),
            sim_stats.get("similarity_winrate", 0), sim_stats.get("sample_size", 0),
            sim_stats.get("similarity_confidence", 0),
            _confluence, _sim_qual,
        )

        if edge < 0.4:
            log.info(
                "DECISION: status=REJECTED reason=edge_score_too_low (%.3f < 0.4)", edge
            )
            return False, f"Edge score too low ({edge:.3f})"

        # --- 5. Scale position size by edge score, cap at 1.2× base ---
        scaled_size = max(1, int(quantity * edge))
        quantity = int(cap_position_size(scaled_size, base_quantity))
        decision["quantity"] = quantity

        # --- 6. Adaptive risk throttling (fail-open) ---
        try:
            recent_losses = _count_recent_consecutive_losses(db, profile_id)
            if recent_losses >= 3:
                throttled = adaptive_risk_throttle(quantity, recent_losses)
                quantity = max(1, int(throttled))
                decision["quantity"] = quantity
                log.info(
                    "Adaptive risk throttle: recent_losses=%d, size %d → %d",
                    recent_losses, scaled_size, quantity,
                )
        except Exception as exc:
            log.warning("Adaptive risk throttle error (proceeding): %s", exc)
            recent_losses = 0

        # --- 7. Portfolio risk validation (fail-open) ---
        try:
            positions = db.query(Position).filter_by(profile=profile_id).all()
            pos_list = [
                {"symbol": p.symbol, "quantity": p.quantity, "avg_cost": p.avg_cost, "side": p.side}
                for p in positions
            ]
            pos_value = sum(p.quantity * p.avg_cost for p in positions)
            total_equity = cash + pos_value

            risk_result = compute_portfolio_risk(pos_list, total_equity)

            # --- PORTFOLIO RISK structured log ---
            bucket_str = ", ".join(
                f"{k}={v:.2f}" for k, v in risk_result.get("bucket_exposure", {}).items()
            )
            log.info(
                "PORTFOLIO RISK: total_exposure=%.2f | %s",
                risk_result.get("total_exposure", 0), bucket_str,
            )

            risk_ok, risk_msg = validate_portfolio_risk(
                {"symbol": symbol, "quantity": quantity, "price": price},
                pos_list, total_equity,
            )
            if not risk_ok:
                log.info("DECISION: status=REJECTED reason=%s", risk_msg)
                return False, risk_msg
        except Exception as exc:
            log.warning("Portfolio risk check error (proceeding with existing validation): %s", exc)

        # --- Store edge data for Trade record (Task 5.4) ---
        _edge_data = {
            "edge_score": round(edge, 4),
            "similarity_winrate": round(sim_stats.get("similarity_winrate", 0), 4),
            "similarity_sample_size": sim_stats.get("sample_size", 0),
            "similarity_confidence": round(sim_stats.get("similarity_confidence", 0), 4),
        }

        # --- DECISION structured log (executed) ---
        log.info(
            "DECISION: size_scaled=%d status=EXECUTED edge=%.3f",
            quantity, edge,
        )

    # Validate trade before execution (existing validation)
    if action in ("BUY", "SHORT"):
        from utils.trade_validator import validate_trade, TradeValidationError
        direction = "LONG" if action == "BUY" else "SHORT"
        # Build a normalized decision for validation
        validated = {**decision, "price": price, "stop": stop, "target": target, "quantity": quantity}
        positions = db.query(Position).filter_by(profile=profile_id).all()
        pos_value = sum(p.quantity * p.avg_cost for p in positions)
        total_equity = cash + pos_value
        try:
            validate_trade(validated, profile_id, cash, total_equity, direction)
        except TradeValidationError as e:
            import logging
            logging.getLogger(__name__).warning(f"Trade rejected: {e}")
            return False, str(e)

        # Check correlated exposure
        from utils.trade_validator import check_correlation
        corr_warning = check_correlation(symbol, direction, profile_id, db)
        if corr_warning:
            import logging
            logging.getLogger(__name__).warning(f"Trade rejected: {corr_warning}")
            return False, corr_warning

        # Confidence adjustment based on case library win rates
        from utils.trade_validator import adjust_confidence
        setup_type = decision.get("setup_type") or decision.get("setup") or ""
        regime = decision.get("market_regime") or decision.get("regime")
        conf_adj = adjust_confidence(db.bind, setup_type, regime)
        if conf_adj["block"]:
            import logging
            logging.getLogger(__name__).warning(f"Trade BLOCKED: {conf_adj['reason']}")
            return False, conf_adj["reason"]
        if conf_adj["modifier"] < 1.0:
            import logging
            logging.getLogger(__name__).info(f"Confidence adjusted: {conf_adj['reason']}")

    # ── Build Entry Contract for BUY/SHORT actions ──
    _entry_contract = {}
    if action in ("BUY", "SHORT"):
        try:
            signal_for_contract = _build_signal_for_symbol(db, symbol, decision)
            _entry_contract = build_entry_contract(
                decision, signal_for_contract,
                stop or 0.0, target or 0.0,
            )
            log.info(
                "Entry contract built for %s: setup_type=%s, invalidators=%d",
                symbol,
                _entry_contract.get("setup_type", "unknown"),
                len(_entry_contract.get("invalidators", [])),
            )
        except Exception as exc:
            log.warning("Failed to build entry contract for %s: %s", symbol, exc)

    if action == "BUY":
        cost = quantity * price
        if cost > cash:
            return False, "Insufficient cash"

        pos = db.query(Position).filter_by(
            symbol=symbol, profile=profile_id, side="long"
        ).first()
        if pos:
            total_qty = pos.quantity + quantity
            pos.avg_cost = (pos.avg_cost * pos.quantity + price * quantity) / total_qty
            pos.quantity = total_qty
        else:
            pos = Position(
                symbol=symbol, quantity=quantity,
                avg_cost=price, profile=profile_id, side="long"
            )
            db.add(pos)

        trade = Trade(
            symbol=symbol, direction="LONG", quantity=quantity,
            entry_price=price, reason_entry=decision.get("rationale"),
            stop_price=stop,
            target_price=target,
            profile=profile_id,
            edge_score=_edge_data.get("edge_score"),
            similarity_winrate=_edge_data.get("similarity_winrate"),
            similarity_sample_size=_edge_data.get("similarity_sample_size"),
            similarity_confidence=_edge_data.get("similarity_confidence"),
            thesis=_entry_contract.get("thesis"),
            setup_type=_entry_contract.get("setup_type"),
            invalidators=json.dumps(_entry_contract["invalidators"]) if _entry_contract.get("invalidators") else None,
        )
        db.add(trade)
        db.flush()
        log_trade_event(db, "entry_filled", trade_id=trade.id, agent=f"pm_{profile_id}", symbol=symbol, profile=profile_id, price=price, message=decision.get("rationale"), payload={"action": action, "quantity": quantity, "side": "long", "edge": _edge_data})
        if stop:
            log_trade_event(db, "stop_set", trade_id=trade.id, agent=f"pm_{profile_id}", symbol=symbol, profile=profile_id, price=float(stop), message="Initial stop set", payload={"stop_price": stop})
        if target:
            log_trade_event(db, "target_set", trade_id=trade.id, agent=f"pm_{profile_id}", symbol=symbol, profile=profile_id, price=float(target), message="Initial target set", payload={"target_price": target})
        db.add(Balance(cash=cash - cost, profile=profile_id))

    elif action == "SHORT":
        # Paper short: reserve margin equal to position value
        margin_required = quantity * price
        if margin_required > cash:
            return False, "Insufficient margin for short"

        pos = db.query(Position).filter_by(
            symbol=symbol, profile=profile_id, side="short"
        ).first()
        if pos:
            total_qty = pos.quantity + quantity
            pos.avg_cost = (pos.avg_cost * pos.quantity + price * quantity) / total_qty
            pos.quantity = total_qty
        else:
            pos = Position(
                symbol=symbol, quantity=quantity,
                avg_cost=price, profile=profile_id, side="short"
            )
            db.add(pos)

        trade = Trade(
            symbol=symbol, direction="SHORT", quantity=quantity,
            entry_price=price, reason_entry=decision.get("rationale"),
            stop_price=stop,
            target_price=target,
            profile=profile_id,
            edge_score=_edge_data.get("edge_score"),
            similarity_winrate=_edge_data.get("similarity_winrate"),
            similarity_sample_size=_edge_data.get("similarity_sample_size"),
            similarity_confidence=_edge_data.get("similarity_confidence"),
            thesis=_entry_contract.get("thesis"),
            setup_type=_entry_contract.get("setup_type"),
            invalidators=json.dumps(_entry_contract["invalidators"]) if _entry_contract.get("invalidators") else None,
        )
        db.add(trade)
        db.flush()
        log_trade_event(db, "entry_filled", trade_id=trade.id, agent=f"pm_{profile_id}", symbol=symbol, profile=profile_id, price=price, message=decision.get("rationale"), payload={"action": action, "quantity": quantity, "side": "short", "edge": _edge_data})
        if stop:
            log_trade_event(db, "stop_set", trade_id=trade.id, agent=f"pm_{profile_id}", symbol=symbol, profile=profile_id, price=float(stop), message="Initial stop set", payload={"stop_price": stop})
        if target:
            log_trade_event(db, "target_set", trade_id=trade.id, agent=f"pm_{profile_id}", symbol=symbol, profile=profile_id, price=float(target), message="Initial target set", payload={"target_price": target})
        # Deduct margin from cash (returned + P&L on close)
        db.add(Balance(cash=cash - margin_required, profile=profile_id))

    elif action == "CLOSE":
        # Find the open position (long or short)
        pos = db.query(Position).filter_by(
            symbol=symbol, profile=profile_id
        ).first()
        if not pos:
            return False, "No position to close"

        close_qty = quantity if quantity and quantity < pos.quantity else pos.quantity
        close_qty = abs(close_qty)  # Guard against negative qty from LLM decisions
        side = pos.side

        # Find ALL open trades for this symbol/profile (handles averaged-in positions)
        open_trades = (
            db.query(Trade)
            .filter_by(symbol=symbol, status="open", profile=profile_id)
            .order_by(Trade.entry_time)
            .all()
        )

        pnl_total = 0.0
        remaining_to_close = close_qty
        first_trade = open_trades[0] if open_trades else None

        for open_trade in open_trades:
            if remaining_to_close <= 0:
                break
            trade_close_qty = min(open_trade.quantity, remaining_to_close)
            remaining_to_close -= trade_close_qty

            if side == "long":
                pnl = (price - open_trade.entry_price) * trade_close_qty
            else:  # short: profit when price falls
                pnl = (open_trade.entry_price - price) * trade_close_qty
            pnl_pct = pnl / (open_trade.entry_price * trade_close_qty) * 100

            open_trade.exit_price = price
            open_trade.exit_time = datetime.utcnow()
            open_trade.status = "closed"
            open_trade.pnl = round(pnl, 2)
            open_trade.pnl_pct = round(pnl_pct, 2)
            open_trade.reason_exit = decision.get("rationale")

            # Post-trade PnL sign consistency check
            if pnl != 0 and ((pnl > 0) != (pnl_pct > 0)):
                log.warning(
                    "PnL sign mismatch for %s: pnl=%.2f, pnl_pct=%.2f — signs should be consistent",
                    symbol, pnl, pnl_pct,
                )

            log_trade_event(
                db, "exit_filled", trade_id=open_trade.id, agent=f"pm_{profile_id}",
                symbol=symbol, profile=profile_id, price=price, message=decision.get("rationale"),
                payload={"quantity": trade_close_qty, "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2), "side": side},
            )

            # Queue for review
            from db.schema import ReviewQueue
            db.add(ReviewQueue(trade_id=open_trade.id))

            pnl_total += pnl

        if close_qty >= pos.quantity:
            db.delete(pos)
        else:
            pos.quantity -= close_qty

        # Return margin + P&L to cash
        if side == "long":
            cash_delta = close_qty * price
        else:
            # Return original margin + profit (or minus loss)
            margin_back = close_qty * (first_trade.entry_price if first_trade else price)
            profit = (first_trade.entry_price - price) * close_qty if first_trade else 0
            cash_delta = margin_back + profit

        db.add(Balance(cash=cash + cash_delta, profile=profile_id))

    db.commit()
    return True, "OK"


def run_profile(engine, symbols: list[str], profile_id: str, tier: str = "high") -> dict:
    """
    Run a single PM profile for one cycle with two-tier review routing.

    The decision loop:
    1. Load all open positions with their Entry Contracts
    2. Check for pending Reversal triggers (thesis_invalidation from AgentMemory,
       opposing signals, explicit CLOSE)
    3. For positions WITH a Reversal trigger → call Reversal/Close Review
    4. For positions WITHOUT a Reversal trigger → call Maintenance Review
    5. For NEW entries (no existing position) → use existing entry logic unchanged
    6. Execute resulting decisions via execute_trade()
    7. Log each cycle whether signals were used in advisory or authoritative capacity

    tier controls which LLM is used.
    """
    profile = PM_PROFILES[profile_id]
    fh = FinnhubClient()
    db = get_session(engine)

    # Get analyst signals
    signals = {}
    for sym in symbols:
        sig = (
            db.query(AgentMemory)
            .filter_by(agent="analyst", symbol=sym, key="signal")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if sig:
            signals[sym] = json.loads(sig.value)

    # Get profile-specific execution feedback from Reviewer
    exec_fb = (
        db.query(AgentMemory)
        .filter_by(agent="reviewer", key=f"execution_feedback_{profile_id}")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    # Fall back to general execution feedback
    if not exec_fb:
        exec_fb = (
            db.query(AgentMemory)
            .filter_by(agent="reviewer", key="execution_feedback")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
    feedback_text = exec_fb.value if exec_fb else "No execution feedback yet."

    # Meta-reviewer recommendations for this PM profile
    meta_rec = (
        db.query(AgentMemory)
        .filter_by(agent="meta_reviewer", symbol=f"pm_{profile_id}", key="agent_recommendation")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    meta_text = meta_rec.value if meta_rec else ""

    portfolio = get_portfolio_for_profile(db, fh, profile_id)

    # Check daily loss limit before doing any work
    max_loss = portfolio["starting_balance"] * profile["max_daily_loss_pct"]
    if abs(portfolio["daily_pnl"]) >= max_loss and portfolio["daily_pnl"] < 0:
        notes = f"Daily loss limit hit (${portfolio['daily_pnl']:,.2f}). No more trades today."
        db.close()
        return {"decisions": [], "portfolio_notes": notes, "profile": profile_id}

    # Position health from health monitor
    health_mem = (
        db.query(AgentMemory)
        .filter_by(agent="position_health", key="health_check")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    health_text = health_mem.value if health_mem else "No health data"

    # ── PHASE 1: Two-tier review for existing open positions ──
    # Track signal usage for audit logging (Req 4.4)
    signal_usage_log = []  # list of {"symbol", "usage": "advisory"|"authoritative"}
    review_decisions = []

    open_positions = db.query(Position).filter_by(profile=profile_id).all()

    for pos in open_positions:
        symbol = pos.symbol

        # Load the open trade with Entry Contract
        open_trade = (
            db.query(Trade)
            .filter_by(symbol=symbol, profile=profile_id, status="open")
            .order_by(Trade.entry_time.desc())
            .first()
        )
        if not open_trade:
            continue

        # Check if this position has an Entry Contract
        has_entry_contract = bool(open_trade.thesis)

        if not has_entry_contract:
            # Attempt legacy migration before skipping (Req 8.2, 8.3)
            contract = build_legacy_entry_contract(open_trade)
            if contract:
                open_trade.thesis = contract["thesis"]
                open_trade.setup_type = contract["setup_type"]
                open_trade.invalidators = json.dumps(contract["invalidators"])
                db.commit()
                has_entry_contract = True
            else:
                # No Entry Contract and migration not possible — skip two-tier review.
                log.info(
                    "Position %s has no Entry Contract — skipping two-tier review "
                    "(will be handled by legacy migration or existing logic).",
                    symbol,
                )
                continue

        # Get current price
        try:
            quote = fh.get_quote(symbol)
            current_price = quote["price"]
        except Exception:
            current_price = pos.avg_cost

        # Compute unrealized P&L
        if pos.side == "short":
            unrealized_pnl = (pos.avg_cost - current_price) * pos.quantity
        else:
            unrealized_pnl = (current_price - pos.avg_cost) * pos.quantity
        unrealized_pnl_pct = round(
            unrealized_pnl / (pos.avg_cost * pos.quantity) * 100, 2
        ) if pos.avg_cost and pos.quantity else 0

        # Detect DRIFTING state
        drifting = detect_drifting(db, open_trade)

        # Get analyst signal for this symbol (if any)
        signal_for_symbol = signals.get(symbol)

        # Check for Reversal triggers
        trigger_info = _check_reversal_triggers(
            db, open_trade, {}, signal_for_symbol, profile
        )

        # Build position data dict for review handlers
        position_data = {
            "symbol": symbol,
            "side": pos.side,
            "quantity": pos.quantity,
            "entry_price": open_trade.entry_price,
            "stop_price": open_trade.stop_price,
            "target_price": open_trade.target_price,
            "current_price": current_price,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "drifting": drifting,
            "thesis": open_trade.thesis,
            "setup_type": open_trade.setup_type,
            "invalidators": open_trade.invalidators,
            "indicators": signal_for_symbol.get("indicators") if signal_for_symbol else None,
            "advisory_signals": signal_for_symbol if signal_for_symbol else None,
            "health_text": health_text,
            "market_conditions": signal_for_symbol if signal_for_symbol else None,
            "opposing_evidence": signal_for_symbol if (
                trigger_info and trigger_info.get("type") == "opposing_signal"
            ) else None,
        }

        if trigger_info:
            # ── Reversal/Close Review (authoritative signal usage) ──
            log.info(
                "Routing %s to Reversal/Close Review: trigger=%s",
                symbol, trigger_info["type"],
            )
            review_result = run_reversal_close_review(
                position_data, trigger_info, profile, tier=tier
            )

            # Log authoritative signal usage (Req 4.4)
            if signal_for_symbol:
                signal_usage_log.append({
                    "symbol": symbol,
                    "usage": "authoritative",
                    "trigger": trigger_info["type"],
                })

            # Convert review result to an executable decision
            action = review_result.get("action", "hold_tighten")
            if action in ("close_full", "close_partial"):
                close_decision = {
                    "symbol": symbol,
                    "action": "CLOSE",
                    "quantity": pos.quantity if action == "close_full" else max(1, int(pos.quantity * 0.5)),
                    "price": current_price,
                    "rationale": (
                        f"Reversal/Close Review ({trigger_info['type']}): "
                        f"{review_result.get('reasoning', 'No reasoning')}"
                    ),
                }
                review_decisions.append(close_decision)
            elif action == "hold_tighten":
                # Tighten stop to breakeven if profitable
                if unrealized_pnl > 0 and open_trade.stop_price:
                    new_stop = open_trade.entry_price  # breakeven
                    open_trade.stop_price = new_stop
                    db.commit()
                    log.info(
                        "Reversal/Close Review hold_tighten for %s: "
                        "tightened stop to breakeven (%.2f)",
                        symbol, new_stop,
                    )
        else:
            # ── Maintenance Review (advisory signal usage) ──
            log.info("Routing %s to Maintenance Review", symbol)
            review_result = run_maintenance_review(
                position_data, profile, tier=tier
            )

            # Log advisory signal usage (Req 4.4)
            if signal_for_symbol:
                signal_usage_log.append({
                    "symbol": symbol,
                    "usage": "advisory",
                })

            # Apply maintenance actions
            action = review_result.get("action", "hold")
            if action == "tighten_stop" and review_result.get("new_stop"):
                new_stop = _coerce_price(review_result.get("new_stop"), "new_stop", symbol)
                if new_stop is None:
                    continue
                open_trade.stop_price = new_stop
                db.commit()
                log.info(
                    "Maintenance Review tighten_stop for %s: new stop=%.2f",
                    symbol, new_stop,
                )
            elif action == "raise_target" and review_result.get("new_target"):
                new_target = _coerce_price(review_result.get("new_target"), "new_target", symbol)
                if new_target is None:
                    continue
                open_trade.target_price = new_target
                db.commit()
                log.info(
                    "Maintenance Review raise_target for %s: new target=%.2f",
                    symbol, new_target,
                )
            elif action == "trim_partial" and review_result.get("trim_pct"):
                trim_qty = max(1, int(pos.quantity * review_result["trim_pct"] / 100))
                trim_decision = {
                    "symbol": symbol,
                    "action": "CLOSE",
                    "quantity": trim_qty,
                    "price": current_price,
                    "rationale": (
                        f"Maintenance Review trim_partial ({review_result['trim_pct']}%): "
                        f"{review_result.get('reasoning', 'No reasoning')}"
                    ),
                }
                review_decisions.append(trim_decision)

    # Log signal usage summary for this cycle (Req 4.4)
    advisory_count = sum(1 for s in signal_usage_log if s["usage"] == "advisory")
    authoritative_count = sum(1 for s in signal_usage_log if s["usage"] == "authoritative")
    log.info(
        "SIGNAL USAGE [%s]: advisory=%d, authoritative=%d, details=%s",
        profile_id, advisory_count, authoritative_count,
        json.dumps(signal_usage_log) if signal_usage_log else "no signals used",
    )

    # Execute review-generated decisions (close/trim from two-tier review)
    executed = []
    for decision in review_decisions:
        ok, msg = execute_trade(db, decision, profile_id)
        executed.append({
            **decision, "executed": ok, "message": msg,
            "profile": profile_id, "source": "two_tier_review",
        })

    # ── PHASE 2: Existing entry logic for NEW positions (unchanged) ──
    # Symbols that already have open positions are excluded from new entry consideration.
    held_symbols = {p.symbol for p in db.query(Position).filter_by(profile=profile_id).all()}

    # Build profile-specific system prompt for entry decisions
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        profile_name=profile["name"],
        emoji=profile["emoji"],
        personality=profile["personality"],
        max_positions=profile["max_positions"],
        max_position_pct=int(profile["max_position_pct"] * 100),
        min_risk_reward=profile["min_risk_reward"],
        min_signal_strength=profile["min_signal_strength"],
        avoid_first_minutes=profile["avoid_first_minutes"],
        avoid_last_minutes=profile["avoid_last_minutes"],
        max_daily_loss_pct=int(profile["max_daily_loss_pct"] * 100),
    )

    # Pull weekly stance if available (written Sunday, applies Mon–Fri)
    from datetime import date, timedelta as td
    weekly_stance_mem = (
        db.query(AgentMemory)
        .filter_by(agent="weekly_prep", key=f"weekly_stance_{profile_id}")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    weekly_stance_text = ""
    if weekly_stance_mem:
        data = json.loads(weekly_stance_mem.value)
        week_str = data.get("week", "")
        if week_str >= (date.today() - td(days=6)).isoformat():
            weekly_stance_text = (
                f"\nWEEKLY STANCE (set Sunday):\n"
                f"  stance: {data.get('weekly_stance')}\n"
                f"  reason: {data.get('stance_reason')}\n"
                f"  size_adjustment: {data.get('size_adjustment') or 0:+.0%}\n"
                f"  signal_threshold: {data.get('signal_threshold_adjustment', 'normal')}\n"
                f"  avoid: {data.get('symbols_avoid', [])}\n"
                f"  favor: {data.get('symbols_favor', [])}\n"
                f"  short_bias: {data.get('symbols_short_bias', [])}\n"
                f"  notes: {data.get('notes', '')}"
            )

    # Query case library — find cases relevant to this profile's style
    case_context = {
        "market_regime": None,  # will match broadly
        "bias": "long",
    }
    relevant_cases = get_relevant_cases(engine, case_context, limit=5)
    cases_text = format_cases_for_prompt(relevant_cases)
    strategy_context = build_strategy_context(engine)

    # Win rates by setup type — PM uses this to adjust sizing
    win_rates = get_win_rate_by_setup(engine)
    if win_rates:
        win_rate_lines = ["Setup type win rates from case library:"]
        for r in sorted(win_rates, key=lambda x: x["win_rate"], reverse=True):
            flag = " ⚠️ avoid or reduce size" if r["win_rate"] < 40 and r["total"] >= 5 else ""
            win_rate_lines.append(
                f"  {r['setup_type']}: {r['win_rate']}% ({r['wins']}/{r['total']}) "
                f"avg pnl {r['avg_pnl_pct'] or 0:+.1f}%{flag}"
            )
        win_rate_text = "\n".join(win_rate_lines)
    else:
        win_rate_text = "No setup win rate data yet."

    # Breaking news from news monitor
    news_mem = (
        db.query(AgentMemory)
        .filter_by(agent="news_monitor", key="breaking_news")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    news_text = news_mem.value if news_mem else "No breaking news"

    # Get behavioral parameters (auto-extracted from reviewer feedback)
    from utils.behavioral_params import get_behavioral_params
    behav_params = get_behavioral_params(engine, profile_id)

    # Filter signals to only symbols without open positions (entry candidates)
    entry_signals = {sym: sig for sym, sig in signals.items() if sym not in held_symbols}

    # Refresh portfolio snapshot (may have changed from review-phase closes)
    portfolio = get_portfolio_for_profile(db, fh, profile_id)

    user_prompt = f"""
Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
Profile: {profile['name']} {profile['emoji']}

CURRENT PORTFOLIO:
{json.dumps(portfolio, indent=2)}

ANALYST SIGNALS:
{json.dumps(entry_signals, indent=2)}

EXECUTION FEEDBACK (your profile only):
{feedback_text}{weekly_stance_text}

META-REVIEWER RECOMMENDATIONS (system-level feedback for your profile):
{meta_text if meta_text else 'None yet'}

SETUP WIN RATES (from case library — use to adjust position sizing):
{win_rate_text}

STRATEGY RECOMMENDATIONS (from Quant Researcher):
{strategy_context}

RELEVANT PAST CASES:
{cases_text}

BREAKING NEWS (from news monitor):
{news_text}

POSITION HEALTH (from health monitor):
{health_text}

BEHAVIORAL ADJUSTMENTS (auto-extracted from feedback — applied to your decisions):
{json.dumps(behav_params, indent=2) if behav_params.get('notes') else 'No adjustments active'}

Make your trading decisions for this cycle.
NOTE: Open positions are managed by the two-tier review system. Only consider NEW entries here.
"""

    raw = call_llm(system_prompt, user_prompt, json_mode=True, tier=tier)
    result = parse_json_response(raw)

    # Apply behavioral parameters to decisions
    from utils.behavioral_params import apply_params_to_decision

    # Execute entry decisions (only BUY/SHORT for new positions)
    for decision in result.get("decisions", []):
        decision = apply_params_to_decision(decision, behav_params, profile)
        if decision.get("action") == "PASS":
            executed.append({
                **decision, "executed": False,
                "message": "Blocked by behavioral params",
                "profile": profile_id, "source": "entry_logic",
            })
            continue

        # Filter out CLOSE/HOLD actions for held symbols — those are handled
        # by the two-tier review system above
        action = decision.get("action", "").upper()
        sym = decision.get("symbol", "")
        if action == "CLOSE" and sym in held_symbols:
            log.info(
                "Ignoring LLM CLOSE for %s — close decisions are handled by "
                "two-tier review system only.",
                sym,
            )
            continue
        if action == "HOLD":
            continue

        # ── Entry timing gate ──
        setup = decision.get("setup_type") or ""
        if not setup:
            signal_for_timing = _build_signal_for_symbol(db, sym, decision)
            setup = signal_for_timing.get("setup_type", "")
        max_minutes = ENTRY_WINDOW_LIMITS.get(setup)
        if max_minutes is not None:
            from pytz import timezone as _tz
            et_tz = _tz("America/New_York")
            now_et = datetime.now(et_tz)
            market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            minutes_since_open = (now_et - market_open).total_seconds() / 60
            if minutes_since_open > max_minutes:
                log.info(
                    "ENTRY TIMING GATE: %s %s blocked — %s entry window closed "
                    "(%d min since open, max %d)",
                    sym, setup, setup, int(minutes_since_open), max_minutes,
                )
                decision["action"] = "PASS"
                decision["rationale"] = (
                    f"Entry timing gate: {setup} window closed "
                    f"({int(minutes_since_open)} min since open, max {max_minutes})"
                )
                executed.append({
                    **decision, "executed": False,
                    "message": f"Blocked by entry timing gate",
                    "profile": profile_id, "source": "entry_logic",
                })
                continue

        # ── LessonRegistry pre-trade gate ──
        signal = _build_signal_for_symbol(db, sym, decision)
        verdict = check_track_record(
            engine,
            symbol=sym,
            bias=signal.get("bias", ""),
            setup_type=signal.get("setup_type", ""),
        )

        if verdict["verdict"] == "BLOCK":
            log.warning(
                "LESSON BLOCK: %s %s %s — avg_score=%.2f (n=%d). Trade blocked.",
                sym, signal.get("bias"), signal.get("setup_type"),
                verdict.get("avg_score_5") or verdict.get("avg_score_3") or 0,
                verdict["sample_size"],
            )
            executed.append({
                **decision, "executed": False,
                "message": f"Blocked by LessonRegistry: {verdict['verdict']}",
                "profile": profile_id, "source": "entry_logic",
            })
            continue

        if verdict["verdict"] == "POOR_TRACK_RECORD":
            original_qty = decision.get("quantity", 0)
            decision["quantity"] = max(1, int(original_qty * verdict["size_multiplier"]))
            log.info(
                "LESSON WARNING: %s %s %s — avg_score=%.2f (n=%d). "
                "Reducing qty %d → %d.",
                sym, signal.get("bias"), signal.get("setup_type"),
                verdict.get("avg_score_3") or 0, verdict["sample_size"],
                original_qty, decision["quantity"],
            )

        ok, msg = execute_trade(db, decision, profile_id)
        executed.append({
            **decision, "executed": ok, "message": msg,
            "profile": profile_id, "source": "entry_logic",
        })

    # Save PM notes
    notes = result.get("portfolio_notes", "")
    if notes:
        mem = AgentMemory(
            agent=f"pm_{profile_id}",
            symbol=None,
            key="notes",
            value=notes,
        )
        db.add(mem)
        db.commit()

    db.close()
    return {"decisions": executed, "portfolio_notes": notes, "profile": profile_id}


def run(engine, symbols: list[str]) -> dict:
    """Run all active PM profiles in sequence."""
    all_results = {}
    for profile_id in ACTIVE_PROFILES:
        all_results[profile_id] = run_profile(engine, symbols, profile_id)
    return all_results
