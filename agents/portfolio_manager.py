"""
Portfolio Manager Agent
Each profile (conservative/moderate/aggressive) runs with its own isolated portfolio.
Scout provides symbols only — all entry/exit decisions are made here.
"""

import json
import os
from datetime import datetime
from utils.finnhub_client import FinnhubClient
from utils.llm import call_llm, parse_json_response
from db.schema import AgentMemory, Position, Balance, Trade, get_session
from models.pm_profiles import PM_PROFILES, ACTIVE_PROFILES
from utils.case_library import get_relevant_cases, format_cases_for_prompt, get_win_rate_by_setup
from agents.quant_researcher import build_strategy_context


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

    # If still no stop, calculate a default (2% from entry) — only for new positions
    if action in ("BUY", "SHORT") and not stop and price:
        if action == "BUY":
            stop = round(price * 0.98, 2)
        elif action == "SHORT":
            stop = round(price * 1.02, 2)
        import logging
        logging.getLogger(__name__).warning(f"No stop provided for {symbol}, using default 2%: {stop}")

    starting = PM_PROFILES[profile_id]["starting_balance"]
    bal = (
        db.query(Balance)
        .filter_by(profile=profile_id)
        .order_by(Balance.timestamp.desc())
        .first()
    )
    cash = bal.cash if bal else float(starting)

    # Validate trade before execution
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

        db.add(Trade(
            symbol=symbol, direction="LONG", quantity=quantity,
            entry_price=price, reason_entry=decision.get("rationale"),
            stop_price=stop,
            target_price=target,
            profile=profile_id,
        ))
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

        db.add(Trade(
            symbol=symbol, direction="SHORT", quantity=quantity,
            entry_price=price, reason_entry=decision.get("rationale"),
            stop_price=stop,
            target_price=target,
            profile=profile_id,
        ))
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
        side = pos.side

        open_trade = (
            db.query(Trade)
            .filter_by(symbol=symbol, status="open", profile=profile_id)
            .order_by(Trade.entry_time)
            .first()
        )
        if open_trade:
            if side == "long":
                pnl = (price - open_trade.entry_price) * close_qty
            else:  # short: profit when price falls
                pnl = (open_trade.entry_price - price) * close_qty
            pnl_pct = pnl / (open_trade.entry_price * close_qty) * 100
            open_trade.exit_price = price
            open_trade.exit_time = datetime.utcnow()
            open_trade.status = "closed"
            open_trade.pnl = round(pnl, 2)
            open_trade.pnl_pct = round(pnl_pct, 2)
            open_trade.reason_exit = decision.get("rationale")

            # Queue for review
            from db.schema import ReviewQueue
            db.add(ReviewQueue(trade_id=open_trade.id))

        if close_qty >= pos.quantity:
            db.delete(pos)
        else:
            pos.quantity -= close_qty

        # Return margin + P&L to cash
        if side == "long":
            cash_delta = close_qty * price
        else:
            # Return original margin + profit (or minus loss)
            margin_back = close_qty * open_trade.entry_price if open_trade else close_qty * price
            profit = (open_trade.entry_price - price) * close_qty if open_trade else 0
            cash_delta = margin_back + profit

        db.add(Balance(cash=cash + cash_delta, profile=profile_id))

    db.commit()
    return True, "OK"


def run_profile(engine, symbols: list[str], profile_id: str, tier: str = "high") -> dict:
    """Run a single PM profile for one cycle. tier controls which LLM is used."""
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

    # Build profile-specific system prompt
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

    # Position health from health monitor
    health_mem = (
        db.query(AgentMemory)
        .filter_by(agent="position_health", key="health_check")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    health_text = health_mem.value if health_mem else "No health data"

    # Get behavioral parameters (auto-extracted from reviewer feedback)
    from utils.behavioral_params import get_behavioral_params
    behav_params = get_behavioral_params(engine, profile_id)

    user_prompt = f"""
Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
Profile: {profile['name']} {profile['emoji']}

CURRENT PORTFOLIO:
{json.dumps(portfolio, indent=2)}

ANALYST SIGNALS:
{json.dumps(signals, indent=2)}

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
"""

    raw = call_llm(system_prompt, user_prompt, json_mode=True, tier=tier)
    result = parse_json_response(raw)

    # Check daily loss limit before executing
    max_loss = portfolio["starting_balance"] * profile["max_daily_loss_pct"]
    if abs(portfolio["daily_pnl"]) >= max_loss and portfolio["daily_pnl"] < 0:
        notes = f"Daily loss limit hit (${portfolio['daily_pnl']:,.2f}). No more trades today."
        db.close()
        return {"decisions": [], "portfolio_notes": notes, "profile": profile_id}

    # Apply behavioral parameters to decisions
    from utils.behavioral_params import apply_params_to_decision

    # Execute decisions
    executed = []
    for decision in result.get("decisions", []):
        decision = apply_params_to_decision(decision, behav_params, profile)
        if decision.get("action") == "PASS":
            executed.append({**decision, "executed": False, "message": "Blocked by behavioral params", "profile": profile_id})
            continue
        ok, msg = execute_trade(db, decision, profile_id)
        executed.append({**decision, "executed": ok, "message": msg, "profile": profile_id})

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
