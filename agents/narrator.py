"""
Desk Narrator Agent
Generates short narrative updates from trading data throughout the day.
Read-only: only SELECTs from trading tables, writes to AgentMemory + Blogger.
"""

import json
import logging
from datetime import date, datetime, timedelta
from db.schema import (
    Trade, Position, Balance, DailyLog, AgentMemory,
    get_session, DynamicStrategy,
)
from models.case import Case
from utils.llm import call_llm
from models.pm_profiles import PM_PROFILES, ACTIVE_PROFILES

logger = logging.getLogger("narrator")

UPDATE_TYPES = [
    "morning_briefing", "hourly_recap", "afternoon_recap",
    "daily_wrap", "weekly_wrap", "sunday_prep", "flash_update",
]

UPDATE_TYPE_DISPLAY = {
    "morning_briefing": "Morning Briefing",
    "hourly_recap": "Hourly Recap",
    "afternoon_recap": "Afternoon Recap",
    "daily_wrap": "Daily Wrap",
    "weekly_wrap": "Weekly Wrap",
    "sunday_prep": "Sunday Prep",
    "flash_update": "🚨 Desk Flash",
}

# Maps update_type -> assembly function name
ASSEMBLY_FUNCTIONS = {
    "morning_briefing": "assemble_morning_briefing",
    "hourly_recap": "assemble_hourly_recap",
    "afternoon_recap": "assemble_afternoon_recap",
    "daily_wrap": "assemble_daily_wrap",
    "weekly_wrap": "assemble_weekly_wrap",
    "sunday_prep": "assemble_sunday_prep",
    "flash_update": "assemble_flash_update",
}

# Rate limit: max 1 flash update per symbol per 15 min
FLASH_COOLDOWN_MINUTES = 15
_flash_cooldowns = {}  # {symbol: last_flash_datetime}


def run(engine, update_type: str, event_context: dict = None) -> dict:
    """
    Main entry point. Called by orchestrator for each narrative run.
    event_context is optional — only used for flash_update triggers.
    Returns {"narrative": str, "update_type": str, "date": str, "skipped": bool}
    """
    if update_type not in UPDATE_TYPES:
        logger.error(f"Unknown update_type: {update_type}")
        return {"narrative": "", "update_type": update_type, "skipped": True}

    today = date.today().isoformat()

    # --- Build dedup key ---
    dedup_key = _build_dedup_key(update_type, today, event_context)

    # --- Flash update rate limiting ---
    if update_type == "flash_update" and event_context:
        symbol = event_context.get("symbol", "")
        if symbol in _flash_cooldowns:
            elapsed = (datetime.utcnow() - _flash_cooldowns[symbol]).total_seconds() / 60
            if elapsed < FLASH_COOLDOWN_MINUTES:
                logger.info(
                    f"Flash update for {symbol} rate-limited "
                    f"({elapsed:.0f}min < {FLASH_COOLDOWN_MINUTES}min)"
                )
                return {
                    "narrative": "", "update_type": update_type,
                    "date": today, "skipped": True,
                }

    # --- Deduplication check ---
    db = get_session(engine)
    try:
        existing = (
            db.query(AgentMemory)
            .filter_by(agent="narrator", key=update_type, symbol=dedup_key)
            .first()
        )
        if existing:
            logger.info(f"Narrative already exists for {update_type} [{dedup_key}], skipping")
            try:
                return json.loads(existing.value)
            except (json.JSONDecodeError, TypeError):
                return {
                    "narrative": "", "update_type": update_type,
                    "date": today, "skipped": True,
                }
    finally:
        db.close()

    # --- Read story arc ---
    story_arc = _read_story_arc(engine, today)

    # --- Data assembly ---
    try:
        assembly_fn = globals()[ASSEMBLY_FUNCTIONS[update_type]]
        if update_type == "flash_update":
            context = assembly_fn(engine, event_context)
        else:
            context = assembly_fn(engine)
    except Exception as e:
        logger.error(f"Data assembly failed for {update_type}: {e}", exc_info=True)
        context = {"error": str(e), "data_gap": True}

    # --- Compute confidence regime ---
    try:
        context["confidence_regime"] = compute_confidence_regime(engine)
    except Exception as e:
        logger.error(f"Confidence regime computation failed: {e}")
        context["confidence_regime"] = {}

    # --- Inject story arc into context ---
    context["story_arc"] = story_arc

    # --- LLM generation ---
    narrative = ""
    try:
        system_prompt = build_system_prompt(update_type)
        user_prompt = build_user_prompt(update_type, context)
        narrative = call_llm(system_prompt, user_prompt, tier="medium")
    except Exception as e:
        logger.error(f"LLM generation failed for {update_type}: {e}", exc_info=True)
        narrative = ""

    result = {
        "narrative": narrative,
        "update_type": update_type,
        "date": today,
        "dedup_key": dedup_key,
        "generated_at": datetime.utcnow().isoformat(),
        "skipped": False,
    }

    # --- Store to AgentMemory ---
    try:
        db = get_session(engine)
        db.add(AgentMemory(
            agent="narrator",
            symbol=dedup_key,
            key=update_type,
            value=json.dumps(result, default=str),
        ))
        db.commit()
        logger.info(f"Narrative stored: {update_type} [{dedup_key}]")
    except Exception as e:
        logger.error(f"AgentMemory write failed for {update_type}: {e}")
    finally:
        db.close()

    # --- Update story arc ---
    try:
        _update_story_arc(engine, today, update_type, narrative, context)
    except Exception as e:
        logger.error(f"Story arc update failed: {e}")

    # --- Update flash cooldown ---
    if update_type == "flash_update" and event_context:
        symbol = event_context.get("symbol", "")
        if symbol:
            _flash_cooldowns[symbol] = datetime.utcnow()

    # --- Publish to Blogger ---
    try:
        from utils.blogger_publisher import BloggerPublisher
        publisher = BloggerPublisher()
        if publisher.is_enabled() and narrative:
            title = format_blog_title(update_type, today)
            publisher.publish(title=title, content=narrative)
    except Exception as e:
        logger.error(f"Blogger publish failed for {update_type}: {e}")

    return result


def _build_dedup_key(update_type: str, today: str, event_context: dict = None) -> str:
    """Build the dedup key for AgentMemory symbol field."""
    if update_type == "hourly_recap":
        from pytz import timezone
        et = datetime.now(timezone("America/New_York"))
        hour = et.hour % 12 or 12  # 0→12, 1-12 as-is
        ampm = "AM" if et.hour < 12 else "PM"
        hour_label = f"{hour}{ampm}"  # e.g., "10AM", "11AM"
        return f"{today}:{hour_label}"
    elif update_type == "afternoon_recap":
        return f"{today}:2PM"
    elif update_type == "flash_update":
        ts = datetime.utcnow().strftime("%H%M%S")
        return f"{today}:flash:{ts}"
    else:
        return today


# ---------------------------------------------------------------------------
# Stubs for functions implemented in later tasks (2.2–2.5)
# ---------------------------------------------------------------------------

# --- Task 2.2: System prompts and user prompt builders ---

NARRATOR_SYSTEM_PROMPT = """You are the desk narrator for a multi-agent paper trading system.
You write like a senior trader briefing the desk: confident, direct, slightly informal.
You have seen it all and you are not afraid to question logic.

Rules:
- Lead with P&L, positions, and trades. The money line comes first.
- Use specific prices, levels, and percentages. Never say "moved higher" when you can say "rallied 2.3% to $187.40".
- 3 to 8 sentences. Clean prose paragraphs. No headers, no bullet points, no markdown.
- End with one forward-looking sentence about what to watch next.
- When PMs diverge on the same signal, note it and explain why.
- Flag unusual events: missed signals, stale data, drawdowns over 2%.
- Do not hedge or equivocate. If the desk lost money, say so and say why.

Story continuity:
- You are given the day's STORY ARC — the running thesis and unresolved themes.
- Reference earlier updates when relevant: "This extends the morning risk-on thesis" or "Desk now unwinding the opening conviction."
- If today's action confirms or refutes the thesis, say so explicitly.
- If a new theme emerges, name it.

Confidence regime (market weather):
- You are given a CONFIDENCE REGIME assessment: edge quality, tape noise, signal disagreement, catalyst freshness.
- Narrate it naturally — don't list the metrics, weave them into the prose.
- If P&L diverges from signal quality, call it out: "Desk up 1.3%, but signal quality deteriorating; conviction lower than P&L suggests."
"""

FLASH_SYSTEM_PROMPT = """You are the desk narrator issuing an urgent flash update.
Something significant just happened. Be immediate, specific, and brief — 2 to 4 sentences max.
Lead with what happened, then the impact, then what to watch.
No preamble. No hedging. This is a desk alert, not a recap.
"""


def build_system_prompt(update_type: str) -> str:
    """Return the appropriate system prompt for the given update type."""
    if update_type == "flash_update":
        return FLASH_SYSTEM_PROMPT
    return NARRATOR_SYSTEM_PROMPT


def build_user_prompt(update_type: str, context: dict) -> str:
    """Build the user prompt from assembled context data."""
    if context.get("data_gap"):
        return (
            f"Data assembly had errors. Note the gap: {context.get('error', 'unknown')}. "
            "Write a brief narrative acknowledging limited data."
        )

    today = date.today().isoformat()
    prompt_builders = {
        "morning_briefing": _build_morning_prompt,
        "hourly_recap": _build_hourly_prompt,
        "afternoon_recap": _build_afternoon_prompt,
        "daily_wrap": _build_daily_wrap_prompt,
        "weekly_wrap": _build_weekly_wrap_prompt,
        "sunday_prep": _build_sunday_prep_prompt,
        "flash_update": _build_flash_prompt,
    }
    builder = prompt_builders.get(update_type)
    if builder is None:
        return f"Write a brief narrative for update type '{update_type}' on {today}."
    return builder(context, today)


def _build_morning_prompt(ctx: dict, today: str) -> str:
    return f"""Write the morning briefing for {today}.

MARKET REGIME: {ctx.get('regime', 'unknown')}
QUANT STRATEGY: Primary: {ctx.get('primary_strategy', 'none')} | Avoid: {ctx.get('strategies_to_avoid', 'none')}

ANALYST SIGNALS:
{json.dumps(ctx.get('analyst_signals', {}), indent=2)}

RESEARCHER SENTIMENT:
{json.dumps(ctx.get('researcher_sentiment', {}), indent=2)}

PM WEEKLY STANCES:
{json.dumps(ctx.get('pm_stances', {}), indent=2)}

OPEN POSITIONS:
{json.dumps(ctx.get('positions', []), indent=2)}

PORTFOLIO SUMMARY:
{json.dumps(ctx.get('portfolio_summary', {}), indent=2)}

SCOUT PICKS: {json.dumps(ctx.get('scout_picks', []))}
BREAKING NEWS: {json.dumps(ctx.get('breaking_news', []))}

STORY ARC:
{json.dumps(ctx.get('story_arc', {}), indent=2)}

CONFIDENCE REGIME:
{json.dumps(ctx.get('confidence_regime', {}), indent=2)}

Write the morning briefing. Lead with the desk's starting position and key signals."""


def _build_hourly_prompt(ctx: dict, today: str) -> str:
    return f"""Write the hourly recap for {today} at {ctx.get('hour_label', 'unknown')}.

TRADES SINCE LAST UPDATE:
{json.dumps(ctx.get('recent_trades', []), indent=2)}

POSITION P&L CHANGES:
{json.dumps(ctx.get('position_pnl_changes', []), indent=2)}

SIGNAL CHANGES:
{json.dumps(ctx.get('signal_changes', {}), indent=2)}

BREAKING NEWS:
{json.dumps(ctx.get('breaking_news', []), indent=2)}

CATALYST FRESHNESS:
{json.dumps(ctx.get('catalyst_freshness', {}), indent=2)}

PM DIVERGENCES:
{json.dumps(ctx.get('pm_divergences', []), indent=2)}

UNUSUAL EVENTS:
{json.dumps(ctx.get('unusual_events', []), indent=2)}

QUIET PERIOD: {ctx.get('quiet_period', False)}

STORY ARC:
{json.dumps(ctx.get('story_arc', {}), indent=2)}

CONFIDENCE REGIME:
{json.dumps(ctx.get('confidence_regime', {}), indent=2)}

Write the hourly recap. If no trades occurred, focus on position movement and signal changes."""


def _build_afternoon_prompt(ctx: dict, today: str) -> str:
    return f"""Write the afternoon recap for {today} at 2:00 PM ET.

TRADES SINCE LAST UPDATE:
{json.dumps(ctx.get('recent_trades', []), indent=2)}

POSITION P&L CHANGES:
{json.dumps(ctx.get('position_pnl_changes', []), indent=2)}

SIGNAL CHANGES:
{json.dumps(ctx.get('signal_changes', {}), indent=2)}

BREAKING NEWS:
{json.dumps(ctx.get('breaking_news', []), indent=2)}

CATALYST FRESHNESS:
{json.dumps(ctx.get('catalyst_freshness', {}), indent=2)}

PM DIVERGENCES:
{json.dumps(ctx.get('pm_divergences', []), indent=2)}

UNUSUAL EVENTS:
{json.dumps(ctx.get('unusual_events', []), indent=2)}

MIDDAY AGGREGATE P&L PER PROFILE:
{json.dumps(ctx.get('aggregate_pnl', {}), indent=2)}

WIN/LOSS COUNT PER PROFILE:
{json.dumps(ctx.get('win_loss', {}), indent=2)}

TOTAL EQUITY CHANGE PER PROFILE:
{json.dumps(ctx.get('equity_change', {}), indent=2)}

QUIET PERIOD: {ctx.get('quiet_period', False)}

STORY ARC:
{json.dumps(ctx.get('story_arc', {}), indent=2)}

CONFIDENCE REGIME:
{json.dumps(ctx.get('confidence_regime', {}), indent=2)}

Write the afternoon recap. Include the midday performance summary across all profiles alongside recent activity."""


def _build_daily_wrap_prompt(ctx: dict, today: str) -> str:
    return f"""Write the daily wrap for {today}.

CLOSED TRADES TODAY:
{json.dumps(ctx.get('closed_trades', []), indent=2)}

OPEN POSITIONS (carried overnight):
{json.dumps(ctx.get('open_positions', []), indent=2)}

PER-PROFILE P&L AND EQUITY:
{json.dumps(ctx.get('profile_summary', {}), indent=2)}

WIN/LOSS RECORD:
{json.dumps(ctx.get('win_loss', {}), indent=2)}

DAILY LOG:
{json.dumps(ctx.get('daily_log', {}), indent=2)}

REVIEWER SCORES:
{json.dumps(ctx.get('reviewer_scores', {}), indent=2)}

DAILY REVIEW:
{json.dumps(ctx.get('daily_review', {}), indent=2)}

LESSONS LEARNED:
{json.dumps(ctx.get('lessons', []), indent=2)}

STORY ARC:
{json.dumps(ctx.get('story_arc', {}), indent=2)}

CONFIDENCE REGIME:
{json.dumps(ctx.get('confidence_regime', {}), indent=2)}

Write the daily wrap. Lead with the day's P&L across all profiles, cover all trades with outcomes, reference reviewer feedback, and end with what worked versus what failed."""


def _build_weekly_wrap_prompt(ctx: dict, today: str) -> str:
    return f"""Write the weekly wrap for the week ending {today}.

WEEK P&L PER PROFILE:
{json.dumps(ctx.get('week_pnl', {}), indent=2)}

BEST TRADES OF THE WEEK:
{json.dumps(ctx.get('best_trades', []), indent=2)}

WORST TRADES OF THE WEEK:
{json.dumps(ctx.get('worst_trades', []), indent=2)}

CASE LIBRARY TRENDS:
{json.dumps(ctx.get('case_trends', []), indent=2)}

STRATEGY PERFORMANCE:
{json.dumps(ctx.get('strategy_performance', {}), indent=2)}

AGENT GRADES (META REVIEWER):
{json.dumps(ctx.get('agent_grades', {}), indent=2)}

DAILY LOGS FOR THE WEEK:
{json.dumps(ctx.get('daily_logs', []), indent=2)}

STORY ARC:
{json.dumps(ctx.get('story_arc', {}), indent=2)}

CONFIDENCE REGIME:
{json.dumps(ctx.get('confidence_regime', {}), indent=2)}

Write the weekly wrap. Cover the week's P&L, highlight the best and worst trades, note strategy performance and any retirements, include agent grades if available, and end with what to watch next week."""


def _build_sunday_prep_prompt(ctx: dict, today: str) -> str:
    return f"""Write the Sunday prep narrative for the week starting tomorrow.

WEEKLY PREP BRIEFING:
{json.dumps(ctx.get('weekly_briefing', {}), indent=2)}

WATCHLIST:
{json.dumps(ctx.get('watchlist', []), indent=2)}

QUANT STRATEGY RECOMMENDATION:
{json.dumps(ctx.get('strategy_recommendation', {}), indent=2)}

META REVIEWER AGENT GRADES:
{json.dumps(ctx.get('agent_grades', {}), indent=2)}

PM STANCES PER PROFILE:
{json.dumps(ctx.get('pm_stances', {}), indent=2)}

STORY ARC:
{json.dumps(ctx.get('story_arc', {}), indent=2)}

CONFIDENCE REGIME:
{json.dumps(ctx.get('confidence_regime', {}), indent=2)}

Write the Sunday prep narrative. Cover the desk's stance and strategy going into Monday, note each PM's posture, and end with what to watch in the first session."""


def _build_flash_prompt(ctx: dict, today: str) -> str:
    return f"""🚨 Flash update for {today}.

TRIGGER: {ctx.get('trigger', 'unknown')}
SYMBOL: {ctx.get('symbol', 'unknown')}
DETAILS: {ctx.get('details', '')}
PRICE: {ctx.get('price', 'N/A')}
PROFILE: {ctx.get('profile', 'N/A')}
P&L IMPACT: {ctx.get('pnl_impact', 'N/A')}

CURRENT POSITION:
{json.dumps(ctx.get('position_data', {}), indent=2)}

ANALYST SIGNAL:
{json.dumps(ctx.get('analyst_signal', {}), indent=2)}

CATALYST FRESHNESS:
{json.dumps(ctx.get('catalyst_freshness', {}), indent=2)}

STORY ARC:
{json.dumps(ctx.get('story_arc', {}), indent=2)}

Write the flash update. Lead with what happened, then the impact, then what to watch."""


# --- Task 2.3: Story arc functions ---

def _read_story_arc(engine, today: str) -> dict:
    """Read the current story arc for today. Falls back to yesterday's arc for morning briefing."""
    db = get_session(engine)
    try:
        arc = (
            db.query(AgentMemory)
            .filter_by(agent="narrator_meta", key="current_story_arc", symbol=today)
            .first()
        )
        if arc:
            return json.loads(arc.value)

        # Morning briefing: try yesterday's arc for continuity
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        arc = (
            db.query(AgentMemory)
            .filter_by(agent="narrator_meta", key="current_story_arc", symbol=yesterday)
            .first()
        )
        if arc:
            data = json.loads(arc.value)
            data["is_yesterday"] = True
            return data

        return {}
    except Exception as e:
        logger.error(f"Story arc read failed: {e}")
        return {}
    finally:
        db.close()


def _update_story_arc(engine, today: str, update_type: str, narrative: str, context: dict):
    """Update the story arc after generating a narrative."""
    db = get_session(engine)
    try:
        # Read existing arc
        existing = (
            db.query(AgentMemory)
            .filter_by(agent="narrator_meta", key="current_story_arc", symbol=today)
            .first()
        )
        current_arc = json.loads(existing.value) if existing else {}

        # Initialize on morning briefing
        if update_type == "morning_briefing":
            current_arc = {
                "opening_thesis": context.get("regime", "unknown") + " regime",
                "key_convictions": [],
                "unresolved_themes": [],
                "conviction_level": context.get("confidence_regime", {}).get("overall", "moderate"),
                "thesis_status": "active",
                "updates": [],
            }
            # Extract convictions from analyst signals
            for sym, sig in context.get("analyst_signals", {}).items():
                if sig.get("strength") in ("strong", "moderate"):
                    current_arc["key_convictions"].append(
                        f"{sig.get('signal', '?')} {sym} ({sig.get('setup_type', '?')})"
                    )

        # Append update summary
        current_arc.setdefault("updates", []).append({
            "type": update_type,
            "time": datetime.utcnow().isoformat(),
            "summary": narrative[:200] if narrative else "",
        })

        # Update conviction level from confidence regime
        cr = context.get("confidence_regime", {})
        if cr.get("overall"):
            current_arc["conviction_level"] = cr["overall"]

        # Resolve arc on daily wrap
        if update_type == "daily_wrap":
            current_arc["thesis_status"] = "resolved"

        # Write back
        if existing:
            existing.value = json.dumps(current_arc, default=str)
        else:
            db.add(AgentMemory(
                agent="narrator_meta",
                symbol=today,
                key="current_story_arc",
                value=json.dumps(current_arc, default=str),
            ))
        db.commit()
    except Exception as e:
        logger.error(f"Story arc update failed: {e}")
    finally:
        db.close()


# --- Task 2.4: Confidence regime ---

def compute_confidence_regime(engine) -> dict:
    """
    Compute the desk's confidence regime across four dimensions.
    Returns qualitative assessment for narrative injection.
    """
    db = get_session(engine)
    try:
        today = date.today().isoformat()

        # 1. Edge quality: average edge score of today's trades
        trades_today = db.query(Trade).filter(
            Trade.entry_time >= today,
            Trade.edge_score.isnot(None),
        ).all()
        avg_edge = (
            sum(t.edge_score for t in trades_today) / len(trades_today)
            if trades_today
            else None
        )

        # 2. Signal disagreement: HOLD ratio from analyst signals
        analyst_signals = {}
        for mem in db.query(AgentMemory).filter_by(agent="analyst", key="signal").all():
            try:
                data = json.loads(mem.value)
                analyst_signals[mem.symbol] = data.get("signal", "HOLD")
            except (json.JSONDecodeError, TypeError):
                pass
        hold_ratio = (
            sum(1 for s in analyst_signals.values() if s == "HOLD")
            / max(len(analyst_signals), 1)
        )

        # 3. Catalyst freshness: aggregate freshness across held positions
        positions = db.query(Position).all()
        held_symbols = [p.symbol for p in positions]
        stale_count = 0
        if held_symbols:
            from utils.catalyst_freshness import compute_catalyst_freshness
            try:
                freshness = compute_catalyst_freshness(db, held_symbols)
                stale_count = sum(
                    1 for f in freshness.values()
                    if f.get("freshness_state") == "stale"
                )
            except Exception:
                pass

        # 4. Tape noise: high HOLD ratio = noisy tape
        tape_noisy = hold_ratio > 0.5

        # Compute overall label
        if avg_edge and avg_edge >= 0.6 and not tape_noisy and stale_count == 0:
            overall = "high conviction"
        elif avg_edge and avg_edge < 0.4 or stale_count > len(held_symbols) / 2:
            overall = "low conviction"
        elif tape_noisy:
            overall = "deteriorating"
        else:
            overall = "moderate conviction"

        return {
            "overall": overall,
            "avg_edge_score": round(avg_edge, 2) if avg_edge else None,
            "hold_ratio": round(hold_ratio, 2),
            "stale_positions": stale_count,
            "total_positions": len(held_symbols),
            "tape_noisy": tape_noisy,
        }
    except Exception as e:
        logger.error(f"Confidence regime computation failed: {e}")
        return {"overall": "unknown"}
    finally:
        db.close()


# --- Task 2.5: Helper functions ---

def format_blog_title(update_type: str, date_str: str) -> str:
    """Generate blog post title like 'Morning Briefing - Apr 24, 2026'."""
    display = UPDATE_TYPE_DISPLAY.get(update_type, update_type.replace("_", " ").title())
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        formatted = dt.strftime("%b %d, %Y")
    except (ValueError, TypeError):
        formatted = date_str
    return f"{display} - {formatted}"


def get_last_narrative_timestamp(engine, update_types=None) -> datetime | None:
    """
    Get the timestamp of the most recent narrative of any type (or specific types).
    Used by hourly/afternoon recaps to determine the 'since' window.
    """
    db = get_session(engine)
    try:
        query = db.query(AgentMemory).filter_by(agent="narrator")
        if update_types:
            query = query.filter(AgentMemory.key.in_(update_types))
        latest = query.order_by(AgentMemory.timestamp.desc()).first()
        return latest.timestamp if latest else None
    finally:
        db.close()


def detect_pm_divergences(engine, since: datetime) -> list[dict]:
    """
    Detect cases where PMs made different decisions on the same analyst signal.
    Returns list of divergence dicts.
    """
    db = get_session(engine)
    try:
        # Get PM notes since the timestamp
        pm_decisions = {}
        for profile_id in ACTIVE_PROFILES:
            notes = (
                db.query(AgentMemory)
                .filter_by(agent=f"pm_{profile_id}", key="notes")
                .filter(AgentMemory.timestamp >= since)
                .all()
            )
            for note in notes:
                try:
                    data = json.loads(note.value)
                    symbol = data.get("symbol") or note.symbol
                    if symbol:
                        pm_decisions.setdefault(symbol, {})[profile_id] = data.get("action", "PASS")
                except (json.JSONDecodeError, TypeError):
                    pass

        # Find symbols where PMs diverged
        divergences = []
        for symbol, decisions in pm_decisions.items():
            actions = set(decisions.values())
            if len(actions) > 1:
                divergences.append({"symbol": symbol, **decisions})
        return divergences
    finally:
        db.close()


def detect_unusual_events(engine, today: str) -> list[dict]:
    """
    Detect unusual events: large drawdowns (>= 2%), stale catalyst data.
    Returns list of event dicts.
    """
    events = []
    db = get_session(engine)
    try:
        # Check daily P&L per profile for drawdowns
        for profile_id in ACTIVE_PROFILES:
            bal = (
                db.query(Balance)
                .filter_by(profile=profile_id)
                .order_by(Balance.timestamp.desc())
                .first()
            )
            if bal and bal.total_equity and bal.cash:
                # The actual daily P&L comes from DailyLog
                log_entry = db.query(DailyLog).filter_by(date=today).first()
                if log_entry and log_entry.daily_pnl_pct is not None:
                    if log_entry.daily_pnl_pct <= -2.0:
                        events.append({
                            "type": "large_drawdown",
                            "profile": profile_id,
                            "daily_pnl_pct": log_entry.daily_pnl_pct,
                        })
    finally:
        db.close()
    return events


# --- Task 4.1–4.4: Data assembly functions ---

def assemble_morning_briefing(engine) -> dict:
    """Assemble context data for morning briefing narrative.

    Queries AgentMemory for analyst signals, researcher sentiment, quant regime,
    strategy recommendation, weekly stances, scout picks, and breaking news.
    Also queries Position and Balance tables for open positions and portfolio summary.

    Each DB query is wrapped in its own try/except with fallback to empty data
    so that a single query failure doesn't break the entire assembly.
    """
    today = date.today().isoformat()
    ctx: dict = {}

    # --- 1. Analyst signals (latest per symbol) ---
    try:
        db = get_session(engine)
        try:
            rows = (
                db.query(AgentMemory)
                .filter_by(agent="analyst", key="signal")
                .order_by(AgentMemory.timestamp.desc())
                .all()
            )
            signals = {}
            for row in rows:
                sym = row.symbol
                if sym and sym not in signals:
                    try:
                        signals[sym] = json.loads(row.value)
                    except (json.JSONDecodeError, TypeError):
                        signals[sym] = {"raw": row.value}
            ctx["analyst_signals"] = signals
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Morning briefing: analyst signals query failed: {e}")
        ctx["analyst_signals"] = {}

    # --- 2. Researcher sentiment (latest per symbol) ---
    try:
        db = get_session(engine)
        try:
            rows = (
                db.query(AgentMemory)
                .filter_by(agent="researcher", key="sentiment")
                .order_by(AgentMemory.timestamp.desc())
                .all()
            )
            sentiment = {}
            for row in rows:
                sym = row.symbol
                if sym and sym not in sentiment:
                    try:
                        sentiment[sym] = json.loads(row.value)
                    except (json.JSONDecodeError, TypeError):
                        sentiment[sym] = {"raw": row.value}
            ctx["researcher_sentiment"] = sentiment
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Morning briefing: researcher sentiment query failed: {e}")
        ctx["researcher_sentiment"] = {}

    # --- 3. Quant regime (latest) ---
    try:
        db = get_session(engine)
        try:
            row = (
                db.query(AgentMemory)
                .filter_by(agent="quant_researcher", key="regime")
                .order_by(AgentMemory.timestamp.desc())
                .first()
            )
            if row:
                try:
                    val = json.loads(row.value)
                    # Value may be a plain string or a dict with a "regime" key
                    if isinstance(val, str):
                        ctx["regime"] = val
                    elif isinstance(val, dict):
                        ctx["regime"] = val.get("regime", str(val))
                    else:
                        ctx["regime"] = str(val)
                except (json.JSONDecodeError, TypeError):
                    ctx["regime"] = row.value
            else:
                ctx["regime"] = "unknown"
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Morning briefing: quant regime query failed: {e}")
        ctx["regime"] = "unknown"

    # --- 4. Strategy recommendation (latest) ---
    try:
        db = get_session(engine)
        try:
            row = (
                db.query(AgentMemory)
                .filter_by(agent="quant_researcher", key="strategy_recommendation")
                .order_by(AgentMemory.timestamp.desc())
                .first()
            )
            if row:
                try:
                    rec = json.loads(row.value)
                    ctx["primary_strategy"] = rec.get("recommended_strategies", [])
                    ctx["strategies_to_avoid"] = rec.get("strategies_to_avoid", [])
                except (json.JSONDecodeError, TypeError):
                    ctx["primary_strategy"] = []
                    ctx["strategies_to_avoid"] = []
            else:
                ctx["primary_strategy"] = []
                ctx["strategies_to_avoid"] = []
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Morning briefing: strategy recommendation query failed: {e}")
        ctx["primary_strategy"] = []
        ctx["strategies_to_avoid"] = []

    # --- 5. Weekly stances (latest per profile) ---
    try:
        db = get_session(engine)
        try:
            stances = {}
            for profile_id in ACTIVE_PROFILES:
                stance_key = f"weekly_stance_{profile_id}"
                row = (
                    db.query(AgentMemory)
                    .filter_by(agent="weekly_prep", key=stance_key)
                    .order_by(AgentMemory.timestamp.desc())
                    .first()
                )
                if row:
                    try:
                        stances[profile_id] = json.loads(row.value)
                    except (json.JSONDecodeError, TypeError):
                        stances[profile_id] = {"raw": row.value}
            ctx["pm_stances"] = stances
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Morning briefing: weekly stances query failed: {e}")
        ctx["pm_stances"] = {}

    # --- 6. Scout picks (latest) ---
    try:
        db = get_session(engine)
        try:
            row = (
                db.query(AgentMemory)
                .filter_by(agent="scout", key="daily_picks")
                .order_by(AgentMemory.timestamp.desc())
                .first()
            )
            if row:
                try:
                    val = json.loads(row.value)
                    # Value may be a dict with a "picks" key or a list directly
                    if isinstance(val, dict):
                        ctx["scout_picks"] = val.get("picks", [val])
                    elif isinstance(val, list):
                        ctx["scout_picks"] = val
                    else:
                        ctx["scout_picks"] = [val]
                except (json.JSONDecodeError, TypeError):
                    ctx["scout_picks"] = []
            else:
                ctx["scout_picks"] = []
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Morning briefing: scout picks query failed: {e}")
        ctx["scout_picks"] = []

    # --- 7. Breaking news (today) ---
    try:
        db = get_session(engine)
        try:
            rows = (
                db.query(AgentMemory)
                .filter_by(agent="news_monitor", key="breaking_news")
                .filter(AgentMemory.symbol == today)
                .order_by(AgentMemory.timestamp.desc())
                .all()
            )
            news = []
            for row in rows:
                try:
                    news.append(json.loads(row.value))
                except (json.JSONDecodeError, TypeError):
                    news.append({"raw": row.value})
            ctx["breaking_news"] = news
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Morning briefing: breaking news query failed: {e}")
        ctx["breaking_news"] = []

    # --- 8. Open positions ---
    try:
        db = get_session(engine)
        try:
            positions = db.query(Position).all()
            ctx["positions"] = [
                {
                    "symbol": p.symbol,
                    "profile": p.profile,
                    "side": p.side,
                    "quantity": p.quantity,
                    "avg_cost": p.avg_cost,
                    "opened_at": p.opened_at.isoformat() if p.opened_at else None,
                }
                for p in positions
            ]
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Morning briefing: positions query failed: {e}")
        ctx["positions"] = []

    # --- 9. Portfolio summary (latest balance per profile) ---
    try:
        db = get_session(engine)
        try:
            summary = {}
            for profile_id in ACTIVE_PROFILES:
                bal = (
                    db.query(Balance)
                    .filter_by(profile=profile_id)
                    .order_by(Balance.timestamp.desc())
                    .first()
                )
                if bal:
                    summary[profile_id] = {
                        "cash": bal.cash,
                        "portfolio_value": bal.portfolio_value,
                        "total_equity": bal.total_equity,
                    }
            ctx["portfolio_summary"] = summary
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Morning briefing: portfolio summary query failed: {e}")
        ctx["portfolio_summary"] = {}

    return ctx


def _assemble_recap_base(engine) -> dict:
    """Shared data assembly for hourly and afternoon recaps.

    Queries trades since last narrative, current positions, analyst signals,
    breaking news, catalyst freshness, PM divergences, and unusual events.
    Each DB query is wrapped in its own try/except with fallback to empty data.
    """
    today = date.today().isoformat()
    ctx: dict = {}

    # --- Determine "since" window ---
    since = None
    try:
        since = get_last_narrative_timestamp(engine)
    except Exception as e:
        logger.error(f"Recap: get_last_narrative_timestamp failed: {e}")
    if since is None:
        # Fallback: start of today (midnight UTC)
        since = datetime.strptime(today, "%Y-%m-%d")

    # --- Hour label (current ET time) ---
    try:
        from pytz import timezone as pytz_tz
        et_now = datetime.now(pytz_tz("America/New_York"))
        hour = et_now.hour % 12 or 12
        ampm = "AM" if et_now.hour < 12 else "PM"
        ctx["hour_label"] = f"{hour}:00 {ampm}"
    except Exception as e:
        logger.error(f"Recap: hour label computation failed: {e}")
        ctx["hour_label"] = "unknown"

    # --- 1. Recent trades (entry_time or exit_time since last narrative) ---
    try:
        db = get_session(engine)
        try:
            from sqlalchemy import or_
            trades = (
                db.query(Trade)
                .filter(
                    or_(
                        Trade.entry_time >= since,
                        Trade.exit_time >= since,
                    )
                )
                .order_by(Trade.entry_time.desc())
                .all()
            )
            ctx["recent_trades"] = [
                {
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "quantity": t.quantity,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "profile": t.profile,
                    "status": t.status,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                    "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                    "reason_entry": t.reason_entry,
                    "reason_exit": t.reason_exit,
                }
                for t in trades
            ]
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Recap: recent trades query failed: {e}")
        ctx["recent_trades"] = []

    # --- 2. Current positions for P&L changes ---
    try:
        db = get_session(engine)
        try:
            positions = db.query(Position).all()
            ctx["position_pnl_changes"] = [
                {
                    "symbol": p.symbol,
                    "profile": p.profile,
                    "side": p.side,
                    "quantity": p.quantity,
                    "avg_cost": p.avg_cost,
                    "opened_at": p.opened_at.isoformat() if p.opened_at else None,
                }
                for p in positions
            ]
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Recap: positions query failed: {e}")
        ctx["position_pnl_changes"] = []

    # --- 3. Analyst signals for signal changes ---
    try:
        db = get_session(engine)
        try:
            rows = (
                db.query(AgentMemory)
                .filter_by(agent="analyst", key="signal")
                .order_by(AgentMemory.timestamp.desc())
                .all()
            )
            signals = {}
            for row in rows:
                sym = row.symbol
                if sym and sym not in signals:
                    try:
                        signals[sym] = json.loads(row.value)
                    except (json.JSONDecodeError, TypeError):
                        signals[sym] = {"raw": row.value}
            ctx["signal_changes"] = signals
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Recap: analyst signals query failed: {e}")
        ctx["signal_changes"] = {}

    # --- 4. Breaking news (today) ---
    try:
        db = get_session(engine)
        try:
            rows = (
                db.query(AgentMemory)
                .filter_by(agent="news_monitor", key="breaking_news")
                .filter(AgentMemory.symbol == today)
                .order_by(AgentMemory.timestamp.desc())
                .all()
            )
            news = []
            for row in rows:
                try:
                    news.append(json.loads(row.value))
                except (json.JSONDecodeError, TypeError):
                    news.append({"raw": row.value})
            ctx["breaking_news"] = news
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Recap: breaking news query failed: {e}")
        ctx["breaking_news"] = []

    # --- 5. Catalyst freshness for held positions ---
    try:
        db = get_session(engine)
        try:
            positions = db.query(Position).all()
            held_symbols = [p.symbol for p in positions]
            if held_symbols:
                try:
                    from utils.catalyst_freshness import compute_catalyst_freshness
                    freshness = compute_catalyst_freshness(db, held_symbols)
                    ctx["catalyst_freshness"] = freshness
                except Exception as e:
                    logger.error(f"Recap: catalyst freshness computation failed: {e}")
                    ctx["catalyst_freshness"] = {}
            else:
                ctx["catalyst_freshness"] = {}
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Recap: catalyst freshness query failed: {e}")
        ctx["catalyst_freshness"] = {}

    # --- 6. PM divergences ---
    try:
        ctx["pm_divergences"] = detect_pm_divergences(engine, since)
    except Exception as e:
        logger.error(f"Recap: PM divergence detection failed: {e}")
        ctx["pm_divergences"] = []

    # --- 7. Unusual events (drawdowns, stale data) ---
    try:
        ctx["unusual_events"] = detect_unusual_events(engine, today)
    except Exception as e:
        logger.error(f"Recap: unusual event detection failed: {e}")
        ctx["unusual_events"] = []

    # --- 8. Quiet period flag ---
    ctx["quiet_period"] = len(ctx.get("recent_trades", [])) == 0

    return ctx


def assemble_hourly_recap(engine) -> dict:
    """Assemble context data for hourly recap narrative.

    Queries trades since last narrative update, current positions, analyst signals,
    breaking news, catalyst freshness, PM divergences, and unusual events.
    Sets quiet_period=True when no trades occurred since last update.
    Each DB query is wrapped in its own try/except with fallback to empty data.
    """
    return _assemble_recap_base(engine)


def assemble_afternoon_recap(engine) -> dict:
    """Assemble context data for afternoon recap narrative.

    Same scope as hourly recap, plus aggregate P&L per profile, win/loss count,
    and total equity change from market open.
    Each DB query is wrapped in its own try/except with fallback to empty data.
    """
    ctx = _assemble_recap_base(engine)
    today = date.today().isoformat()

    # --- Aggregate P&L per profile (sum of realized trade P&L today) ---
    try:
        db = get_session(engine)
        try:
            closed_today = (
                db.query(Trade)
                .filter(
                    Trade.exit_time >= today,
                    Trade.status == "closed",
                )
                .all()
            )
            aggregate_pnl = {}
            win_loss = {}
            for profile_id in ACTIVE_PROFILES:
                profile_trades = [t for t in closed_today if t.profile == profile_id]
                total_pnl = sum(t.pnl or 0.0 for t in profile_trades)
                wins = sum(1 for t in profile_trades if (t.pnl or 0) > 0)
                losses = sum(1 for t in profile_trades if (t.pnl or 0) <= 0 and t.pnl is not None)
                aggregate_pnl[profile_id] = round(total_pnl, 2)
                win_loss[profile_id] = {
                    "wins": wins,
                    "losses": losses,
                    "total": len(profile_trades),
                }
            ctx["aggregate_pnl"] = aggregate_pnl
            ctx["win_loss"] = win_loss
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Afternoon recap: aggregate P&L query failed: {e}")
        ctx["aggregate_pnl"] = {}
        ctx["win_loss"] = {}

    # --- Total equity change per profile from market open ---
    try:
        db = get_session(engine)
        try:
            equity_change = {}
            for profile_id in ACTIVE_PROFILES:
                # Get earliest balance record for today (market open snapshot)
                opening_bal = (
                    db.query(Balance)
                    .filter(
                        Balance.profile == profile_id,
                        Balance.timestamp >= today,
                    )
                    .order_by(Balance.timestamp.asc())
                    .first()
                )
                # Get latest balance record (current)
                current_bal = (
                    db.query(Balance)
                    .filter_by(profile=profile_id)
                    .order_by(Balance.timestamp.desc())
                    .first()
                )
                if opening_bal and current_bal:
                    opening_equity = opening_bal.total_equity or opening_bal.cash or 0
                    current_equity = current_bal.total_equity or current_bal.cash or 0
                    change = current_equity - opening_equity
                    change_pct = (
                        (change / opening_equity * 100) if opening_equity else 0
                    )
                    equity_change[profile_id] = {
                        "opening_equity": round(opening_equity, 2),
                        "current_equity": round(current_equity, 2),
                        "change": round(change, 2),
                        "change_pct": round(change_pct, 2),
                    }
                else:
                    equity_change[profile_id] = {
                        "opening_equity": None,
                        "current_equity": None,
                        "change": None,
                        "change_pct": None,
                    }
            ctx["equity_change"] = equity_change
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Afternoon recap: equity change query failed: {e}")
        ctx["equity_change"] = {}

    return ctx


def assemble_daily_wrap(engine) -> dict:
    """Assemble context data for daily wrap narrative.

    Queries all closed trades for today, open positions, balance per profile,
    DailyLog, reviewer scores, daily_review, and Case lessons.
    Returns per-profile P&L, win/loss, ending equity, reviewer scores, and lessons.
    Each DB query is wrapped in its own try/except with fallback to empty data.
    """
    today = date.today().isoformat()
    ctx: dict = {}

    # --- 1. Closed trades today (exit_time is today, status="closed") ---
    try:
        db = get_session(engine)
        try:
            closed = (
                db.query(Trade)
                .filter(
                    Trade.exit_time >= today,
                    Trade.status == "closed",
                )
                .order_by(Trade.exit_time.desc())
                .all()
            )
            ctx["closed_trades"] = [
                {
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "quantity": t.quantity,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "profile": t.profile,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                    "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                    "reason_entry": t.reason_entry,
                    "reason_exit": t.reason_exit,
                    "review_score": t.review_score,
                    "review_notes": t.review_notes,
                    "edge_score": t.edge_score,
                    "setup_type": t.setup_type,
                }
                for t in closed
            ]
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Daily wrap: closed trades query failed: {e}")
        ctx["closed_trades"] = []

    # --- 2. Open positions (carried overnight) ---
    try:
        db = get_session(engine)
        try:
            open_pos = (
                db.query(Trade)
                .filter(Trade.status == "open")
                .all()
            )
            ctx["open_positions"] = [
                {
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "quantity": t.quantity,
                    "entry_price": t.entry_price,
                    "profile": t.profile,
                    "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                    "stop_price": t.stop_price,
                    "target_price": t.target_price,
                }
                for t in open_pos
            ]
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Daily wrap: open positions query failed: {e}")
        ctx["open_positions"] = []

    # --- 3. Per-profile P&L, win/loss, ending equity ---
    try:
        db = get_session(engine)
        try:
            closed_today = (
                db.query(Trade)
                .filter(
                    Trade.exit_time >= today,
                    Trade.status == "closed",
                )
                .all()
            )
            profile_summary = {}
            for profile_id in ACTIVE_PROFILES:
                profile_trades = [t for t in closed_today if t.profile == profile_id]
                total_pnl = sum(t.pnl or 0.0 for t in profile_trades)
                wins = sum(1 for t in profile_trades if (t.pnl or 0) > 0)
                losses = sum(1 for t in profile_trades if (t.pnl or 0) <= 0 and t.pnl is not None)

                # Get latest balance for ending equity
                bal = (
                    db.query(Balance)
                    .filter_by(profile=profile_id)
                    .order_by(Balance.timestamp.desc())
                    .first()
                )
                ending_equity = bal.total_equity if bal and bal.total_equity else (bal.cash if bal else None)

                profile_summary[profile_id] = {
                    "total_pnl": round(total_pnl, 2),
                    "wins": wins,
                    "losses": losses,
                    "total_trades": len(profile_trades),
                    "ending_equity": round(ending_equity, 2) if ending_equity else None,
                }
            ctx["profile_summary"] = profile_summary
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Daily wrap: profile summary query failed: {e}")
        ctx["profile_summary"] = {}

    # --- 4. Win/loss record (aggregate across all profiles) ---
    try:
        closed_trades = ctx.get("closed_trades", [])
        total_wins = sum(1 for t in closed_trades if (t.get("pnl") or 0) > 0)
        total_losses = sum(1 for t in closed_trades if (t.get("pnl") or 0) <= 0 and t.get("pnl") is not None)
        ctx["win_loss"] = {
            "wins": total_wins,
            "losses": total_losses,
            "total": len(closed_trades),
        }
    except Exception as e:
        logger.error(f"Daily wrap: win/loss computation failed: {e}")
        ctx["win_loss"] = {}

    # --- 5. DailyLog for today ---
    try:
        db = get_session(engine)
        try:
            log_entry = db.query(DailyLog).filter_by(date=today).first()
            if log_entry:
                ctx["daily_log"] = {
                    "date": log_entry.date,
                    "starting_equity": log_entry.starting_equity,
                    "ending_equity": log_entry.ending_equity,
                    "trades_taken": log_entry.trades_taken,
                    "winning_trades": log_entry.winning_trades,
                    "losing_trades": log_entry.losing_trades,
                    "daily_pnl": log_entry.daily_pnl,
                    "daily_pnl_pct": log_entry.daily_pnl_pct,
                    "notes": log_entry.notes,
                }
            else:
                ctx["daily_log"] = {}
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Daily wrap: DailyLog query failed: {e}")
        ctx["daily_log"] = {}

    # --- 6. Reviewer scores (today's feedback) ---
    try:
        db = get_session(engine)
        try:
            reviewer_rows = (
                db.query(AgentMemory)
                .filter_by(agent="reviewer")
                .filter(AgentMemory.timestamp >= today)
                .order_by(AgentMemory.timestamp.desc())
                .all()
            )
            scores = {}
            for row in reviewer_rows:
                sym = row.symbol
                if sym and sym not in scores:
                    try:
                        scores[sym] = json.loads(row.value)
                    except (json.JSONDecodeError, TypeError):
                        scores[sym] = {"raw": row.value}
            ctx["reviewer_scores"] = scores
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Daily wrap: reviewer scores query failed: {e}")
        ctx["reviewer_scores"] = {}

    # --- 7. Daily review (today's review if available) ---
    try:
        db = get_session(engine)
        try:
            review_row = (
                db.query(AgentMemory)
                .filter_by(agent="daily_review", key="daily_review", symbol=today)
                .order_by(AgentMemory.timestamp.desc())
                .first()
            )
            if review_row:
                try:
                    ctx["daily_review"] = json.loads(review_row.value)
                except (json.JSONDecodeError, TypeError):
                    ctx["daily_review"] = {"raw": review_row.value}
            else:
                ctx["daily_review"] = {}
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Daily wrap: daily review query failed: {e}")
        ctx["daily_review"] = {}

    # --- 8. Case lessons for today ---
    try:
        db = get_session(engine)
        try:
            cases = (
                db.query(Case)
                .filter_by(date=today)
                .all()
            )
            ctx["lessons"] = [
                {
                    "symbol": c.symbol,
                    "setup_type": c.setup_type,
                    "outcome": c.outcome,
                    "pnl_pct": c.pnl_pct,
                    "lesson": c.lesson,
                    "profile": c.profile,
                    "selection_score": c.selection_score,
                    "execution_score": c.execution_score,
                }
                for c in cases
            ]
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Daily wrap: Case lessons query failed: {e}")
        ctx["lessons"] = []

    return ctx


def assemble_weekly_wrap(engine) -> dict:
    """Assemble context data for weekly wrap narrative.

    Queries trades for the current Mon-Fri week, DailyLog for the week,
    Case trends, DynamicStrategy performance, and meta_reviewer agent grades.
    Returns week P&L, best/worst trades, strategy performance, agent grades.
    Each DB query is wrapped in its own try/except with fallback to empty data.
    """
    today = date.today()
    # Compute Monday of the current week
    monday = today - timedelta(days=today.weekday())
    friday = monday + timedelta(days=4)
    monday_str = monday.isoformat()
    friday_str = friday.isoformat()
    # Use friday end-of-day as upper bound (inclusive)
    saturday_str = (friday + timedelta(days=1)).isoformat()
    ctx: dict = {}

    # --- 1. All closed trades for the week ---
    try:
        db = get_session(engine)
        try:
            week_trades = (
                db.query(Trade)
                .filter(
                    Trade.exit_time >= monday_str,
                    Trade.exit_time < saturday_str,
                    Trade.status == "closed",
                )
                .order_by(Trade.exit_time.desc())
                .all()
            )
            trades_list = [
                {
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "quantity": t.quantity,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "profile": t.profile,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                    "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                    "reason_entry": t.reason_entry,
                    "reason_exit": t.reason_exit,
                    "review_score": t.review_score,
                    "setup_type": t.setup_type,
                    "edge_score": t.edge_score,
                }
                for t in week_trades
            ]

            # Per-profile P&L for the week
            week_pnl = {}
            for profile_id in ACTIVE_PROFILES:
                profile_trades = [t for t in trades_list if t["profile"] == profile_id]
                total_pnl = sum(t["pnl"] or 0.0 for t in profile_trades)
                wins = sum(1 for t in profile_trades if (t["pnl"] or 0) > 0)
                losses = sum(1 for t in profile_trades if (t["pnl"] or 0) <= 0 and t["pnl"] is not None)
                week_pnl[profile_id] = {
                    "total_pnl": round(total_pnl, 2),
                    "wins": wins,
                    "losses": losses,
                    "total_trades": len(profile_trades),
                }
            ctx["week_pnl"] = week_pnl

            # Best and worst trades (by pnl)
            trades_with_pnl = [t for t in trades_list if t["pnl"] is not None]
            sorted_by_pnl = sorted(trades_with_pnl, key=lambda t: t["pnl"] or 0, reverse=True)
            ctx["best_trades"] = sorted_by_pnl[:3] if sorted_by_pnl else []
            ctx["worst_trades"] = sorted_by_pnl[-3:][::-1] if sorted_by_pnl else []
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Weekly wrap: trades query failed: {e}")
        ctx["week_pnl"] = {}
        ctx["best_trades"] = []
        ctx["worst_trades"] = []

    # --- 2. DailyLog for the week ---
    try:
        db = get_session(engine)
        try:
            logs = (
                db.query(DailyLog)
                .filter(
                    DailyLog.date >= monday_str,
                    DailyLog.date <= friday_str,
                )
                .order_by(DailyLog.date.asc())
                .all()
            )
            ctx["daily_logs"] = [
                {
                    "date": log_entry.date,
                    "starting_equity": log_entry.starting_equity,
                    "ending_equity": log_entry.ending_equity,
                    "trades_taken": log_entry.trades_taken,
                    "winning_trades": log_entry.winning_trades,
                    "losing_trades": log_entry.losing_trades,
                    "daily_pnl": log_entry.daily_pnl,
                    "daily_pnl_pct": log_entry.daily_pnl_pct,
                    "notes": log_entry.notes,
                }
                for log_entry in logs
            ]
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Weekly wrap: DailyLog query failed: {e}")
        ctx["daily_logs"] = []

    # --- 3. Case trends for the week ---
    try:
        db = get_session(engine)
        try:
            cases = (
                db.query(Case)
                .filter(
                    Case.date >= monday_str,
                    Case.date <= friday_str,
                )
                .all()
            )
            ctx["case_trends"] = [
                {
                    "symbol": c.symbol,
                    "date": c.date,
                    "setup_type": c.setup_type,
                    "outcome": c.outcome,
                    "pnl_pct": c.pnl_pct,
                    "lesson": c.lesson,
                    "profile": c.profile,
                    "catalyst_type": c.catalyst_type,
                    "selection_score": c.selection_score,
                    "execution_score": c.execution_score,
                }
                for c in cases
            ]
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Weekly wrap: Case trends query failed: {e}")
        ctx["case_trends"] = []

    # --- 4. DynamicStrategy performance (win rates, retirements this week) ---
    try:
        db = get_session(engine)
        try:
            strategies = db.query(DynamicStrategy).all()
            ctx["strategy_performance"] = {
                s.key: {
                    "name": s.name,
                    "status": s.status,
                    "total_trades": s.total_trades,
                    "wins": s.wins,
                    "win_rate": s.win_rate,
                    "avg_pnl_pct": s.avg_pnl_pct,
                    "retired_at": s.retired_at.isoformat() if s.retired_at else None,
                    "retire_reason": s.retire_reason,
                }
                for s in strategies
            }
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Weekly wrap: DynamicStrategy query failed: {e}")
        ctx["strategy_performance"] = {}

    # --- 5. Meta reviewer agent grades ---
    try:
        db = get_session(engine)
        try:
            grade_row = (
                db.query(AgentMemory)
                .filter_by(agent="meta_reviewer", key="weekly_review")
                .order_by(AgentMemory.timestamp.desc())
                .first()
            )
            if grade_row:
                try:
                    ctx["agent_grades"] = json.loads(grade_row.value)
                except (json.JSONDecodeError, TypeError):
                    ctx["agent_grades"] = {"raw": grade_row.value}
            else:
                ctx["agent_grades"] = {}
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Weekly wrap: meta reviewer grades query failed: {e}")
        ctx["agent_grades"] = {}

    return ctx


def assemble_sunday_prep(engine) -> dict:
    """Assemble context data for Sunday prep narrative.

    Queries weekly_prep briefing, watchlist, quant strategy recommendation,
    meta_reviewer agent grades, and PM stances per profile.
    Returns {"weekly_prep_missing": True} if no weekly_briefing record
    exists with today's date.
    Each DB query is wrapped in its own try/except with fallback to empty data.
    """
    today = date.today().isoformat()
    ctx: dict = {}

    # --- 1. Weekly prep briefing (latest) ---
    weekly_briefing_found = False
    try:
        db = get_session(engine)
        try:
            row = (
                db.query(AgentMemory)
                .filter_by(agent="weekly_prep", key="weekly_briefing")
                .order_by(AgentMemory.timestamp.desc())
                .first()
            )
            if row:
                try:
                    val = json.loads(row.value)
                    ctx["weekly_briefing"] = val
                except (json.JSONDecodeError, TypeError):
                    ctx["weekly_briefing"] = {"raw": row.value}
                # Check if the record is from today
                if row.timestamp and row.timestamp.strftime("%Y-%m-%d") == today:
                    weekly_briefing_found = True
                elif row.symbol == today:
                    weekly_briefing_found = True
                else:
                    # Accept the latest record even if not from today,
                    # but only mark as "found" if it's from today
                    weekly_briefing_found = False
            else:
                ctx["weekly_briefing"] = {}
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Sunday prep: weekly briefing query failed: {e}")
        ctx["weekly_briefing"] = {}

    # If no weekly prep data for today, return early with skip flag
    if not weekly_briefing_found:
        logger.warning(
            f"Sunday prep: no weekly_briefing record for today ({today}), skipping"
        )
        return {"weekly_prep_missing": True}

    # --- 2. Weekly watchlist (latest) ---
    try:
        db = get_session(engine)
        try:
            row = (
                db.query(AgentMemory)
                .filter_by(agent="weekly_prep", key="weekly_watchlist")
                .order_by(AgentMemory.timestamp.desc())
                .first()
            )
            if row:
                try:
                    val = json.loads(row.value)
                    if isinstance(val, dict):
                        ctx["watchlist"] = val.get("watchlist", val.get("symbols", [val]))
                    elif isinstance(val, list):
                        ctx["watchlist"] = val
                    else:
                        ctx["watchlist"] = [val]
                except (json.JSONDecodeError, TypeError):
                    ctx["watchlist"] = []
            else:
                ctx["watchlist"] = []
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Sunday prep: weekly watchlist query failed: {e}")
        ctx["watchlist"] = []

    # --- 3. Quant strategy recommendation (latest) ---
    try:
        db = get_session(engine)
        try:
            row = (
                db.query(AgentMemory)
                .filter_by(agent="quant_researcher", key="strategy_recommendation")
                .order_by(AgentMemory.timestamp.desc())
                .first()
            )
            if row:
                try:
                    ctx["strategy_recommendation"] = json.loads(row.value)
                except (json.JSONDecodeError, TypeError):
                    ctx["strategy_recommendation"] = {"raw": row.value}
            else:
                ctx["strategy_recommendation"] = {}
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Sunday prep: strategy recommendation query failed: {e}")
        ctx["strategy_recommendation"] = {}

    # --- 4. Meta reviewer agent grades (latest) ---
    try:
        db = get_session(engine)
        try:
            grade_row = (
                db.query(AgentMemory)
                .filter_by(agent="meta_reviewer", key="weekly_review")
                .order_by(AgentMemory.timestamp.desc())
                .first()
            )
            if grade_row:
                try:
                    ctx["agent_grades"] = json.loads(grade_row.value)
                except (json.JSONDecodeError, TypeError):
                    ctx["agent_grades"] = {"raw": grade_row.value}
            else:
                ctx["agent_grades"] = {}
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Sunday prep: meta reviewer grades query failed: {e}")
        ctx["agent_grades"] = {}

    # --- 5. PM stances per profile (latest per profile) ---
    try:
        db = get_session(engine)
        try:
            stances = {}
            for profile_id in ACTIVE_PROFILES:
                stance_key = f"weekly_stance_{profile_id}"
                row = (
                    db.query(AgentMemory)
                    .filter_by(agent="weekly_prep", key=stance_key)
                    .order_by(AgentMemory.timestamp.desc())
                    .first()
                )
                if row:
                    try:
                        stances[profile_id] = json.loads(row.value)
                    except (json.JSONDecodeError, TypeError):
                        stances[profile_id] = {"raw": row.value}
            ctx["pm_stances"] = stances
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Sunday prep: PM stances query failed: {e}")
        ctx["pm_stances"] = {}

    return ctx


def assemble_flash_update(engine, event_context: dict) -> dict:
    """Assemble context data for flash update narrative.

    Starts with the event_context dict passed by the triggering agent
    (Price Monitor, Position Timer, or News Monitor) and enriches it with:
    - Current position data for the symbol (from Position table)
    - Recent analyst signal for the symbol (from AgentMemory)
    - Catalyst freshness state (from utils/catalyst_freshness)

    Each enrichment query is wrapped in its own try/except so that
    a single query failure doesn't break the entire assembly.
    """
    # Start with a copy of event_context to avoid mutating the caller's dict
    ctx = dict(event_context) if event_context else {}
    symbol = ctx.get("symbol", "")

    # --- 1. Current position data for the symbol ---
    try:
        db = get_session(engine)
        try:
            positions = (
                db.query(Position)
                .filter_by(symbol=symbol)
                .all()
            )
            if positions:
                ctx["position_data"] = [
                    {
                        "symbol": p.symbol,
                        "profile": p.profile,
                        "side": p.side,
                        "quantity": p.quantity,
                        "avg_cost": p.avg_cost,
                        "opened_at": p.opened_at.isoformat() if p.opened_at else None,
                    }
                    for p in positions
                ]
            else:
                ctx["position_data"] = {}
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Flash update: position data query failed for {symbol}: {e}")
        ctx["position_data"] = {}

    # --- 2. Recent analyst signal for the symbol ---
    try:
        db = get_session(engine)
        try:
            row = (
                db.query(AgentMemory)
                .filter_by(agent="analyst", key="signal", symbol=symbol)
                .order_by(AgentMemory.timestamp.desc())
                .first()
            )
            if row:
                try:
                    ctx["analyst_signal"] = json.loads(row.value)
                except (json.JSONDecodeError, TypeError):
                    ctx["analyst_signal"] = {"raw": row.value}
            else:
                ctx["analyst_signal"] = {}
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Flash update: analyst signal query failed for {symbol}: {e}")
        ctx["analyst_signal"] = {}

    # --- 3. Catalyst freshness state ---
    try:
        if symbol:
            db = get_session(engine)
            try:
                from utils.catalyst_freshness import compute_catalyst_freshness
                freshness = compute_catalyst_freshness(db, [symbol])
                ctx["catalyst_freshness"] = freshness.get(symbol, {})
            finally:
                db.close()
        else:
            ctx["catalyst_freshness"] = {}
    except Exception as e:
        logger.error(f"Flash update: catalyst freshness query failed for {symbol}: {e}")
        ctx["catalyst_freshness"] = {}

    return ctx
