"""
Daily Review Agent
Synthesizes the full trading day into a cohesive narrative journal entry.
Runs after Reviewer and Bookkeeper complete at 4:15 PM ET.
"""

import json
import logging
import subprocess
from datetime import date, datetime, timedelta, timezone
from db.schema import Trade, Position, DailyLog, AgentMemory, get_session
from models.case import Case
from utils.llm import call_llm, parse_json_response

logger = logging.getLogger("daily_review")


def run(engine) -> dict:
    """
    Main entry point. Called by orchestrator after post-market.
    Returns the complete daily review dict.
    """
    try:
        today = date.today().isoformat()
        yesterday = (date.today() - timedelta(days=1)).isoformat()

        # Check for existing review on same date — no-overwrite
        db = get_session(engine)
        try:
            existing = (
                db.query(AgentMemory)
                .filter_by(agent="daily_review", symbol=today, key="daily_review")
                .first()
            )
            if existing:
                logger.info(f"Daily review already exists for {today}, skipping generation")
                try:
                    return json.loads(existing.value)
                except (json.JSONDecodeError, TypeError):
                    return {}
        finally:
            db.close()

        # Gather all inputs
        trade_perf = gather_trade_performance(engine, today)
        git_commits = gather_git_commits(since_date=yesterday)
        agent_context = gather_agent_context(engine)
        previous_review = load_previous_review(engine, today)
        cases = gather_cases_today(engine, today)

        # Build deterministic summary, then generate narrative
        summary = build_deterministic_summary(
            trade_perf, git_commits, agent_context, cases, previous_review
        )
        review = generate_narrative(summary)

        # Store in agent_memory
        db = get_session(engine)
        try:
            db.add(AgentMemory(
                agent="daily_review",
                symbol=today,
                key="daily_review",
                value=json.dumps(review, default=str),
            ))
            db.commit()
            logger.info(f"Daily review stored for {today}")
        except Exception as e:
            logger.error(f"Failed to store daily review: {e}")
        finally:
            db.close()

        return review

    except Exception as e:
        logger.error(f"Daily review failed: {e}")
        return {}


def gather_trade_performance(engine, today: str) -> dict:
    """
    Query trades, daily_log, and positions for today's performance.
    Returns Trade_Performance_Summary dict.

    Args:
        engine: SQLAlchemy engine
        today: date string in YYYY-MM-DD format
    """
    db = get_session(engine)
    try:
        return _build_trade_performance(db, today)
    except Exception as e:
        logger.error(f"Error gathering trade performance: {e}")
        return _empty_performance()
    finally:
        db.close()


def _build_trade_performance(db, today: str) -> dict:
    """Core logic for building trade performance summary from DB queries."""
    # Query trades closed today
    today_start = datetime.strptime(today, "%Y-%m-%d").replace(hour=0, minute=0, second=0)
    today_end = datetime.strptime(today, "%Y-%m-%d").replace(hour=23, minute=59, second=59)

    closed_trades = (
        db.query(Trade)
        .filter(Trade.status == "closed")
        .filter(Trade.exit_time >= today_start)
        .filter(Trade.exit_time <= today_end)
        .all()
    )

    if not closed_trades:
        return _no_trades_performance(db)

    # Core aggregates
    total_trades = len(closed_trades)
    wins = sum(1 for t in closed_trades if (t.pnl or 0) > 0)
    losses = total_trades - wins
    total_pnl = sum(t.pnl or 0 for t in closed_trades)

    # Total P&L percentage from daily_log if available, else compute from trades
    daily_log = db.query(DailyLog).filter(DailyLog.date == today).first()
    total_pnl_pct = daily_log.daily_pnl_pct if daily_log and daily_log.daily_pnl_pct is not None else _compute_pnl_pct(closed_trades)

    # Realized P&L = sum of closed trade P&L
    realized_pnl = total_pnl

    # Unrealized P&L from open positions
    unrealized_pnl = _compute_unrealized_pnl(db)

    # Net daily change
    net_daily_change = realized_pnl + unrealized_pnl

    # Per-profile breakdowns
    per_profile = _compute_per_profile(closed_trades)

    # Best and worst trades by pnl_pct
    best_trade = _identify_best_trade(closed_trades)
    worst_trade = _identify_worst_trade(closed_trades)

    # Setup breakdown
    setup_breakdown = _compute_setup_breakdown(closed_trades)

    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 4) if total_pnl_pct is not None else 0,
        "realized_pnl": round(realized_pnl, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "net_daily_change": round(realized_pnl + unrealized_pnl, 2),
        "per_profile": per_profile,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "setup_breakdown": setup_breakdown,
        "no_trades": False,
    }


def _no_trades_performance(db) -> dict:
    """Build performance summary when no trades were closed today."""
    unrealized_pnl = _compute_unrealized_pnl(db)
    return {
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "total_pnl": 0,
        "total_pnl_pct": 0,
        "realized_pnl": 0,
        "unrealized_pnl": round(unrealized_pnl, 2),
        "net_daily_change": round(unrealized_pnl, 2),
        "per_profile": {
            "conservative": {"trades": 0, "wins": 0, "pnl": 0, "pnl_pct": 0},
            "moderate": {"trades": 0, "wins": 0, "pnl": 0, "pnl_pct": 0},
            "aggressive": {"trades": 0, "wins": 0, "pnl": 0, "pnl_pct": 0},
        },
        "best_trade": None,
        "worst_trade": None,
        "setup_breakdown": {},
        "no_trades": True,
    }


def _compute_pnl_pct(trades) -> float:
    """Compute aggregate P&L percentage from individual trade pnl_pct values."""
    pcts = [t.pnl_pct for t in trades if t.pnl_pct is not None]
    return sum(pcts) if pcts else 0


def _compute_unrealized_pnl(db) -> float:
    """Compute unrealized P&L from open positions using avg_cost."""
    positions = db.query(Position).all()
    unrealized = 0.0
    for pos in positions:
        # Without live prices, unrealized P&L is 0 relative to avg_cost
        # The position value at avg_cost is the baseline
        # This will be 0 unless we have a way to get current price
        # For now, report 0 unrealized since we don't call external APIs
        unrealized += 0.0
    return unrealized


def _compute_per_profile(trades) -> dict:
    """Compute per-profile breakdowns for conservative, moderate, aggressive."""
    profiles = {"conservative", "moderate", "aggressive"}
    result = {}
    for profile in profiles:
        profile_trades = [t for t in trades if t.profile == profile]
        profile_wins = sum(1 for t in profile_trades if (t.pnl or 0) > 0)
        profile_pnl = sum(t.pnl or 0 for t in profile_trades)
        profile_pnl_pct = sum(t.pnl_pct or 0 for t in profile_trades)
        result[profile] = {
            "trades": len(profile_trades),
            "wins": profile_wins,
            "pnl": round(profile_pnl, 2),
            "pnl_pct": round(profile_pnl_pct, 4),
        }
    return result


def _identify_best_trade(trades) -> dict | None:
    """Identify the best trade by pnl_pct."""
    trades_with_pct = [t for t in trades if t.pnl_pct is not None]
    if not trades_with_pct:
        return None
    best = max(trades_with_pct, key=lambda t: t.pnl_pct)
    return {
        "symbol": best.symbol,
        "pnl_pct": best.pnl_pct,
        "setup_type": _extract_setup_type(best.reason_entry),
    }


def _identify_worst_trade(trades) -> dict | None:
    """Identify the worst trade by pnl_pct."""
    trades_with_pct = [t for t in trades if t.pnl_pct is not None]
    if not trades_with_pct:
        return None
    worst = min(trades_with_pct, key=lambda t: t.pnl_pct)
    return {
        "symbol": worst.symbol,
        "pnl_pct": worst.pnl_pct,
        "setup_type": _extract_setup_type(worst.reason_entry),
    }


def _extract_setup_type(reason_entry: str | None) -> str:
    """Extract setup type from reason_entry field. Falls back to 'unknown'."""
    if not reason_entry:
        return "unknown"
    # reason_entry often contains setup type info
    # Try to extract a recognizable setup keyword
    text = reason_entry.lower()
    setup_keywords = [
        "gap_and_go", "momentum_fade", "breakout", "reversal",
        "vwap_bounce", "mean_reversion", "trend_follow", "pullback",
        "earnings_play", "sector_rotation",
    ]
    for kw in setup_keywords:
        if kw in text:
            return kw
    # If no keyword match, use the first few words as a label
    words = reason_entry.strip().split()
    if words:
        return "_".join(words[:3]).lower()[:32]
    return "unknown"


def _compute_setup_breakdown(trades) -> dict:
    """Compute setup type breakdown: counts and win/loss outcomes."""
    breakdown = {}
    for t in trades:
        setup = _extract_setup_type(t.reason_entry)
        if setup not in breakdown:
            breakdown[setup] = {"count": 0, "wins": 0}
        breakdown[setup]["count"] += 1
        if (t.pnl or 0) > 0:
            breakdown[setup]["wins"] += 1
    return breakdown


def _empty_performance() -> dict:
    """Return an empty performance summary for error cases."""
    return {
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "total_pnl": 0,
        "total_pnl_pct": 0,
        "realized_pnl": 0,
        "unrealized_pnl": 0,
        "net_daily_change": 0,
        "per_profile": {
            "conservative": {"trades": 0, "wins": 0, "pnl": 0, "pnl_pct": 0},
            "moderate": {"trades": 0, "wins": 0, "pnl": 0, "pnl_pct": 0},
            "aggressive": {"trades": 0, "wins": 0, "pnl": 0, "pnl_pct": 0},
        },
        "best_trade": None,
        "worst_trade": None,
        "setup_breakdown": {},
        "no_trades": True,
    }


# ---------------------------------------------------------------------------
# Git commit parsing and categorization
# ---------------------------------------------------------------------------

GIT_LOG_FORMAT = "--format=%H|%an|%aI|%s"

# Categorization rules: ordered list of (category, file_patterns, message_keywords)
# File patterns use fnmatch-style matching.
_CATEGORY_RULES = [
    # bugfix is checked first — keywords can appear with any file
    (
        "bugfix",
        None,  # any file
        ["fix", "bug", "patch", "hotfix"],
    ),
    (
        "agent_logic",
        ["agents/*.py"],
        ["agent", "signal", "decision"],
    ),
    (
        "risk_management",
        ["core/*.py", "utils/trade_validator.py"],
        ["risk", "stop", "edge", "position size"],
    ),
    (
        "infrastructure",
        ["orchestrator.py", "deploy/*", "db/*"],
        ["deploy", "schedule", "migrate", "schema"],
    ),
    (
        "strategy",
        ["models/strategies.py", "utils/strategy_store.py"],
        ["strategy", "setup", "backtest"],
    ),
]


def categorize_commit(message: str, files: list[str]) -> str:
    """
    Deterministic categorization of a commit based on message and changed files.

    Priority order:
      1. bugfix keywords (checked first, any file)
      2. File-path-based categories
      3. Message-keyword-based categories
      4. Fallback to "other"

    Returns one of: agent_logic, risk_management, infrastructure, strategy, bugfix, other.
    """
    from fnmatch import fnmatch

    msg_lower = message.lower()

    # --- 1. Check bugfix keywords first (file-independent) ---
    for kw in _CATEGORY_RULES[0][2]:  # bugfix keywords
        if kw in msg_lower:
            return "bugfix"

    # --- 2. File-path-based matching (skip bugfix entry) ---
    for category, file_patterns, _keywords in _CATEGORY_RULES[1:]:
        if file_patterns is None:
            continue
        for f in files:
            for pattern in file_patterns:
                if fnmatch(f, pattern):
                    return category

    # --- 3. Message-keyword-based matching (skip bugfix entry) ---
    for category, _file_patterns, keywords in _CATEGORY_RULES[1:]:
        for kw in keywords:
            if kw in msg_lower:
                return category

    return "other"


def gather_git_commits(since_date: str) -> list[dict]:
    """
    Parse git log since the given date.

    Runs:
        git log --since="{since_date}" --format="%H|%an|%aI|%s" --name-only

    Returns list of commit dicts:
        [{"hash", "author", "timestamp", "message", "files", "category"}, ...]

    On failure (git unavailable, not a repo, etc.) logs a warning and returns [].
    """
    try:
        result = subprocess.run(
            [
                "git", "log",
                f"--since={since_date}",
                GIT_LOG_FORMAT,
                "--name-only",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        logger.warning("git command not found — skipping git commit gathering")
        return []
    except subprocess.TimeoutExpired:
        logger.warning("git log timed out — skipping git commit gathering")
        return []
    except Exception as e:
        logger.warning(f"git log failed: {e} — skipping git commit gathering")
        return []

    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else ""
        logger.warning(f"git log returned non-zero exit code ({result.returncode}): {stderr}")
        return []

    return _parse_git_log_output(result.stdout)


def _parse_git_log_output(raw: str) -> list[dict]:
    """
    Parse the raw stdout of ``git log --format=%H|%an|%aI|%s --name-only``.

    Format:
        <hash>|<author>|<iso-timestamp>|<subject>
        file1
        file2
        <blank line>
        <hash>|<author>|<iso-timestamp>|<subject>
        ...

    Returns a list of commit dicts with keys:
        hash, author, timestamp, message, files, category
    """
    commits: list[dict] = []
    lines = raw.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Skip blank lines between commits
        if not line:
            i += 1
            continue

        # Expect a commit header line with pipe separators
        parts = line.split("|", 3)
        if len(parts) < 4:
            # Not a valid header line — skip
            i += 1
            continue

        commit_hash, author, timestamp, message = parts

        # Collect file names until blank line or end
        files: list[str] = []
        i += 1
        while i < len(lines) and lines[i].strip():
            files.append(lines[i].strip())
            i += 1

        category = categorize_commit(message, files)

        commits.append({
            "hash": commit_hash,
            "author": author,
            "timestamp": timestamp,
            "message": message,
            "files": files,
            "category": category,
        })

    return commits


# ---------------------------------------------------------------------------
# Agent context gathering and previous review loading
# ---------------------------------------------------------------------------


def gather_agent_context(engine) -> dict:
    """
    Read latest context from agent_memory for researcher, reviewer, analyst.

    Returns dict with:
        market_context: str or None
        selection_feedback: str or None
        execution_feedback: dict or None
        analyst_signals: dict or None
        missing_sources: list of missing source names
    """
    db = get_session(engine)
    missing_sources = []

    try:
        # --- Researcher: market_context ---
        market_ctx_mem = (
            db.query(AgentMemory)
            .filter_by(agent="researcher", key="market_context")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if market_ctx_mem:
            market_context = market_ctx_mem.value
        else:
            market_context = None
            missing_sources.append("researcher_context")

        # --- Reviewer: selection_feedback ---
        sel_fb_mem = (
            db.query(AgentMemory)
            .filter_by(agent="reviewer", key="selection_feedback")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if sel_fb_mem:
            selection_feedback = sel_fb_mem.value
        else:
            selection_feedback = None
            missing_sources.append("selection_feedback")

        # --- Reviewer: execution_feedback ---
        exec_fb_mem = (
            db.query(AgentMemory)
            .filter_by(agent="reviewer", key="execution_feedback")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if exec_fb_mem:
            try:
                execution_feedback = json.loads(exec_fb_mem.value)
            except (json.JSONDecodeError, TypeError):
                execution_feedback = exec_fb_mem.value
        else:
            execution_feedback = None
            missing_sources.append("execution_feedback")

        # --- Analyst: signals (all recent, keyed by symbol) ---
        analyst_mems = (
            db.query(AgentMemory)
            .filter_by(agent="analyst", key="signal")
            .order_by(AgentMemory.timestamp.desc())
            .all()
        )
        if analyst_mems:
            analyst_signals = {}
            for mem in analyst_mems:
                symbol = mem.symbol
                if symbol and symbol not in analyst_signals:
                    try:
                        analyst_signals[symbol] = json.loads(mem.value)
                    except (json.JSONDecodeError, TypeError):
                        analyst_signals[symbol] = mem.value
        else:
            analyst_signals = None
            missing_sources.append("analyst_signals")

        return {
            "market_context": market_context,
            "selection_feedback": selection_feedback,
            "execution_feedback": execution_feedback,
            "analyst_signals": analyst_signals,
            "missing_sources": missing_sources,
        }

    except Exception as e:
        logger.error(f"Error gathering agent context: {e}")
        return {
            "market_context": None,
            "selection_feedback": None,
            "execution_feedback": None,
            "analyst_signals": None,
            "missing_sources": [
                "researcher_context",
                "selection_feedback",
                "execution_feedback",
                "analyst_signals",
            ],
        }
    finally:
        db.close()


def load_previous_review(engine, today: str) -> dict | None:
    """
    Load the previous trading day's review from agent_memory.

    Looks for the most recent daily_review entry with a date before today.

    Args:
        engine: SQLAlchemy engine
        today: date string in YYYY-MM-DD format

    Returns:
        The previous review dict, or None if not found.
    """
    db = get_session(engine)
    try:
        prev_mem = (
            db.query(AgentMemory)
            .filter_by(agent="daily_review", key="daily_review")
            .filter(AgentMemory.symbol < today)
            .order_by(AgentMemory.symbol.desc())
            .first()
        )
        if prev_mem:
            try:
                return json.loads(prev_mem.value)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Failed to parse previous review JSON")
                return None
        return None
    except Exception as e:
        logger.error(f"Error loading previous review: {e}")
        return None
    finally:
        db.close()


def gather_cases_today(engine, today: str) -> list[dict]:
    """
    Query the Case table for entries matching today's date.

    Args:
        engine: SQLAlchemy engine
        today: date string in YYYY-MM-DD format

    Returns:
        List of case dicts, or empty list if none found or on error.
    """
    db = get_session(engine)
    try:
        cases = db.query(Case).filter(Case.date == today).all()
        result = []
        for c in cases:
            result.append({
                "symbol": c.symbol,
                "setup_type": c.setup_type,
                "catalyst_type": c.catalyst_type,
                "outcome": c.outcome,
                "pnl_pct": c.pnl_pct,
                "lesson": c.lesson,
                "conditions_for_success": c.conditions_for_success,
                "conditions_to_avoid": c.conditions_to_avoid,
                "selection_score": c.selection_score,
                "execution_score": c.execution_score,
                "confidence": c.confidence,
            })
        return result
    except Exception as e:
        logger.error(f"Error gathering cases for today: {e}")
        return []
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Deterministic summary builder
# ---------------------------------------------------------------------------


def build_deterministic_summary(
    trade_perf: dict | None,
    git_commits: list[dict] | None,
    agent_context: dict | None,
    cases: list[dict] | None,
    previous_review: dict | None,
) -> dict:
    """
    Assemble all gathered data into a structured summary dict.
    No LLM calls. Pure data transformation.

    Uses empty defaults for any None component data and flags
    missing sources in the completeness section.

    Args:
        trade_perf: Trade performance dict from gather_trade_performance, or None
        git_commits: List of commit dicts from gather_git_commits, or None
        agent_context: Dict from gather_agent_context, or None
        cases: List of case dicts from gather_cases_today, or None
        previous_review: Previous review dict from load_previous_review, or None

    Returns:
        Deterministic summary dict per the design schema.
    """
    # Normalise None inputs to safe defaults
    if trade_perf is None:
        trade_perf = {}
    if git_commits is None:
        git_commits = []
    if agent_context is None:
        agent_context = {}
    if cases is None:
        cases = []

    # --- git_changes section ---
    categories: dict[str, int] = {}
    for commit in git_commits:
        cat = commit.get("category", "other")
        categories[cat] = categories.get(cat, 0) + 1

    git_changes = {
        "commits": git_commits,
        "total_commits": len(git_commits),
        "categories": categories,
        "no_commits": len(git_commits) == 0,
    }

    # --- previous_review_summary ---
    previous_review_summary = None
    if previous_review is not None:
        previous_review_summary = previous_review.get("market_summary")

    # --- completeness section ---
    has_trade_data = trade_perf.get("no_trades") is not True and bool(trade_perf)
    has_git_data = len(git_commits) > 0
    has_researcher_context = agent_context.get("market_context") is not None
    has_reviewer_feedback = (
        agent_context.get("selection_feedback") is not None
        or agent_context.get("execution_feedback") is not None
    )
    has_analyst_signals = agent_context.get("analyst_signals") is not None
    has_previous_review = previous_review is not None

    flags = [
        has_trade_data,
        has_git_data,
        has_researcher_context,
        has_reviewer_feedback,
        has_analyst_signals,
        has_previous_review,
    ]
    present_count = sum(flags)

    if present_count == 6:
        confidence = "high"
    elif present_count <= 3:
        confidence = "low"
    else:
        confidence = "medium"

    completeness = {
        "trade_data": has_trade_data,
        "git_data": has_git_data,
        "researcher_context": has_researcher_context,
        "reviewer_feedback": has_reviewer_feedback,
        "analyst_signals": has_analyst_signals,
        "previous_review": has_previous_review,
        "confidence": confidence,
    }

    # --- process_metrics (deterministic) ---
    tp = trade_perf or {}
    total = tp.get("total_trades", 0)
    wins = tp.get("wins", 0)
    process_metrics = {
        "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
        "avg_pnl_per_trade": round(tp.get("total_pnl", 0) / total, 2) if total > 0 else 0,
        "largest_win": tp.get("best_trade", {}).get("pnl_pct") if tp.get("best_trade") else None,
        "largest_loss": tp.get("worst_trade", {}).get("pnl_pct") if tp.get("worst_trade") else None,
        "profiles_active": [p for p, v in (tp.get("per_profile") or {}).items() if v.get("trades", 0) > 0],
    }

    # --- activity_flags ---
    activity_flags = {
        "had_trades": not tp.get("no_trades", True),
        "had_git_commits": len(git_commits) > 0,
        "had_cases": len(cases) > 0,
        "had_previous_review": previous_review is not None,
        "high_volume_day": total >= 10,
        "loss_day": tp.get("total_pnl", 0) < 0,
    }

    # --- day_context ---
    day_context = {
        "day_of_week": date.today().strftime("%A"),
        "is_monday": date.today().weekday() == 0,
        "is_friday": date.today().weekday() == 4,
    }

    return {
        "date": date.today().isoformat(),
        "trade_performance": trade_perf,
        "git_changes": git_changes,
        "agent_context": {
            "market_context": agent_context.get("market_context"),
            "selection_feedback": agent_context.get("selection_feedback"),
            "execution_feedback": agent_context.get("execution_feedback"),
            "analyst_signals": agent_context.get("analyst_signals"),
            "missing_sources": agent_context.get("missing_sources", []),
        },
        "cases_today": cases,
        "previous_review_summary": previous_review_summary,
        "completeness": completeness,
        "process_metrics": process_metrics,
        "activity_flags": activity_flags,
        "day_context": day_context,
    }


# ---------------------------------------------------------------------------
# LLM narrative generator
# ---------------------------------------------------------------------------

NARRATIVE_SYSTEM_PROMPT = """You are the chief diagnostician for a multi-agent paper trading system.
You receive a structured data summary and produce an opinionated, diagnostic daily review.
Your audience is the system's builder — they want to know what broke, what worked, and what to fix first.

Voice rules:
- Be direct and opinionated. "The system lost money because X" not "The system experienced losses."
- Lead with the single most important thing that happened today.
- Rank drivers by impact — don't give equal weight to everything.
- When sample size < 5, say so explicitly but still give your best read.
- Reference specific trades by symbol, setup type, and P&L.
- Correlations are observational, never causal. Say "coincided with" not "caused by."
- Separate process quality from outcomes — a good process can lose money.
- Be concise. Every sentence should earn its place.
"""

# --- Two-phase prompts to keep output complexity manageable for local models ---

_DIAGNOSTIC_PROMPT = NARRATIVE_SYSTEM_PROMPT + """
Return JSON with ONLY these fields:
{
    "executive_summary": "2-3 sentence TL;DR. Lead with the headline number and the why.",
    "day_classification": "one of: strong_win | modest_win | breakeven | modest_loss | bad_day | system_failure",
    "primary_driver": "Single sentence: the #1 factor that determined today's outcome.",
    "performance_story": "3-5 sentence narrative: setup, execution, outcome.",
    "what_worked": ["specific thing 1", "specific thing 2"],
    "what_failed": ["specific thing 1", "specific thing 2"],
    "highest_leverage_fix": "The single change with the biggest positive impact if done tomorrow.",
    "email_subject": "Short subject line for email digest (under 60 chars).",
    "email_preview": "One sentence preview (under 120 chars)."
}
"""

_ANALYSIS_PROMPT = NARRATIVE_SYSTEM_PROMPT + """
Return JSON with ONLY these fields:
{
    "driver_ranking": [{"driver": "description", "impact": "$ or % impact", "controllable": true/false}],
    "system_observations": "What the agents did well or poorly as a system.",
    "lessons_learned": [{"category": "...", "lesson": "...", "evidence": "...", "action": "..."}],
    "process_quality": "Evaluate execution discipline and risk management independent of P&L.",
    "correlations": "Observational connections between code changes and trading outcomes.",
    "git_narrative": "Brief summary of code changes and their relevance.",
    "tomorrows_focus": ["priority 1", "priority 2", "priority 3"],
    "watchouts": ["specific risk or flag for tomorrow"]
}
"""

# Phase 1 fields (diagnostic)
_DIAGNOSTIC_FIELDS = [
    "executive_summary", "day_classification", "primary_driver",
    "performance_story", "what_worked", "what_failed",
    "highest_leverage_fix", "email_subject", "email_preview",
]

# Phase 2 fields (analysis)
_ANALYSIS_FIELDS = [
    "driver_ranking", "system_observations", "lessons_learned",
    "process_quality", "correlations", "git_narrative",
    "tomorrows_focus", "watchouts",
]

# Fields the LLM is expected to produce
_NARRATIVE_FIELDS = [
    "executive_summary",
    "day_classification",
    "primary_driver",
    "driver_ranking",
    "performance_story",
    "system_observations",
    "what_worked",
    "what_failed",
    "highest_leverage_fix",
    "lessons_learned",
    "process_quality",
    "correlations",
    "git_narrative",
    "tomorrows_focus",
    "watchouts",
    "email_subject",
    "email_preview",
]

# Empty defaults used when the LLM fails or returns incomplete data
_EMPTY_NARRATIVE = {
    "executive_summary": "",
    "day_classification": "breakeven",
    "primary_driver": "",
    "driver_ranking": [],
    "performance_story": "",
    "system_observations": "",
    "what_worked": [],
    "what_failed": [],
    "highest_leverage_fix": "",
    "lessons_learned": [],
    "process_quality": "",
    "correlations": "",
    "git_narrative": "",
    "tomorrows_focus": [],
    "watchouts": [],
    "email_subject": "",
    "email_preview": "",
}


def _build_llm_prompt(summary: dict) -> str:
    """
    Build a condensed prompt from the deterministic summary.

    Strips verbose data (full commit file lists, raw analyst signal dicts)
    and keeps only what the LLM needs to write the narrative. This reduces
    token count significantly for local models.
    """
    prompt = {"date": summary.get("date")}

    # Trade performance — pass through as-is, it's already compact
    tp = summary.get("trade_performance", {})
    prompt["trade_performance"] = {
        k: tp[k] for k in [
            "total_trades", "wins", "losses", "total_pnl", "total_pnl_pct",
            "realized_pnl", "unrealized_pnl", "net_daily_change",
            "per_profile", "best_trade", "worst_trade", "setup_breakdown",
            "no_trades",
        ] if k in tp
    }

    # Git changes — summarize commits to one line each, drop file lists
    gc = summary.get("git_changes", {})
    condensed_commits = []
    for c in (gc.get("commits") or [])[:15]:  # cap at 15 most recent
        condensed_commits.append(
            f"{c.get('category', 'other')}: {c.get('message', '')} ({', '.join((c.get('files') or [])[:3])})"
        )
    prompt["git_changes"] = {
        "total_commits": gc.get("total_commits", 0),
        "categories": gc.get("categories", {}),
        "no_commits": gc.get("no_commits", True),
        "commit_summaries": condensed_commits,
    }

    # Agent context — keep market context and feedback as strings, summarize signals
    ctx = summary.get("agent_context", {})
    prompt["agent_context"] = {
        "market_context": ctx.get("market_context"),
        "selection_feedback": ctx.get("selection_feedback"),
        "execution_feedback": ctx.get("execution_feedback"),
        "missing_sources": ctx.get("missing_sources", []),
    }
    # Analyst signals: just symbol + bias + strength, not full dicts
    signals = ctx.get("analyst_signals")
    if signals and isinstance(signals, dict):
        prompt["agent_context"]["analyst_signals_summary"] = {
            sym: f"{s.get('signal', '?')} ({s.get('strength', '?')})"
            if isinstance(s, dict) else str(s)
            for sym, s in list(signals.items())[:10]  # cap at 10 symbols
        }

    # Cases — keep only key fields
    cases = summary.get("cases_today") or []
    prompt["cases_today"] = [
        {k: c.get(k) for k in ["symbol", "setup_type", "outcome", "pnl_pct", "lesson"]}
        for c in cases[:10]
    ]

    prompt["previous_review_summary"] = summary.get("previous_review_summary")
    prompt["completeness"] = summary.get("completeness", {})

    # Include new deterministic fields for the LLM
    prompt["process_metrics"] = summary.get("process_metrics", {})
    prompt["activity_flags"] = summary.get("activity_flags", {})
    prompt["day_context"] = summary.get("day_context", {})

    return json.dumps(prompt, default=str)


def generate_narrative(deterministic_summary: dict, tier: str = "medium") -> dict:
    """
    Generate the narrative review via two focused LLM calls.

    Phase 1 (diagnostic): executive summary, classification, what worked/failed, fix
    Phase 2 (analysis): driver ranking, lessons, correlations, process quality, watchouts

    Each call gets the same condensed input but a simpler output schema,
    keeping output complexity manageable for local models.

    On failure of either call, the other's fields still populate.
    """
    review = dict(deterministic_summary)
    review["generated_at"] = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
    user_prompt = _build_llm_prompt(deterministic_summary)

    # --- Phase 1: Diagnostic ---
    diagnostic = {}
    try:
        raw = call_llm(_DIAGNOSTIC_PROMPT, user_prompt, tier=tier)
        diagnostic = parse_json_response(raw)
        logger.info("Daily review phase 1 (diagnostic) complete")
    except Exception as e:
        logger.error(f"LLM diagnostic phase failed: {e}")

    # --- Phase 2: Analysis ---
    analysis = {}
    try:
        raw = call_llm(_ANALYSIS_PROMPT, user_prompt, tier=tier)
        analysis = parse_json_response(raw)
        logger.info("Daily review phase 2 (analysis) complete")
    except Exception as e:
        logger.error(f"LLM analysis phase failed: {e}")

    # Merge both phases into the review
    narrative = {**diagnostic, **analysis}
    for field in _NARRATIVE_FIELDS:
        review[field] = narrative.get(field, _EMPTY_NARRATIVE[field])

    # Validate lessons_learned structure — ensure each entry has required keys
    validated_lessons = []
    raw_lessons = review.get("lessons_learned") or []
    if isinstance(raw_lessons, list):
        for item in raw_lessons:
            if isinstance(item, dict):
                validated_lessons.append({
                    "category": str(item.get("category", "")),
                    "lesson": str(item.get("lesson", "")),
                    "evidence": str(item.get("evidence", "")),
                    "action": str(item.get("action", "")),
                })
    review["lessons_learned"] = validated_lessons

    # Validate day_classification
    valid_classifications = {"strong_win", "modest_win", "breakeven", "modest_loss", "bad_day", "system_failure"}
    if review.get("day_classification") not in valid_classifications:
        review["day_classification"] = "breakeven"

    # Validate list fields
    for list_field in ["driver_ranking", "what_worked", "what_failed", "tomorrows_focus", "watchouts"]:
        if not isinstance(review.get(list_field), list):
            review[list_field] = _EMPTY_NARRATIVE[list_field]

    # Validate string fields
    for str_field in ["executive_summary", "primary_driver", "performance_story",
                      "system_observations", "highest_leverage_fix", "process_quality",
                      "correlations", "git_narrative", "email_subject", "email_preview"]:
        if not isinstance(review.get(str_field), str):
            review[str_field] = str(review.get(str_field, "")) if review.get(str_field) else ""

    return review
