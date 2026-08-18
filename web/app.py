"""
Paper Trader — Web Dashboard
Run with: python web/app.py
Access at: http://localhost:5000
"""

import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger(__name__)

from flask import Flask, jsonify, render_template, request
from sqlalchemy import text
from db.schema import init_db, get_session, Position, Balance, Trade, TradeEvent, AgentMemory, DailyLog
from models.pm_profiles import PM_PROFILES, ACTIVE_PROFILES
from utils.finnhub_client import FinnhubClient
from utils.catalyst_freshness import (
    compute_catalyst_freshness,
    get_breaking_news_for_symbols,
    get_market_day_start,
    build_freshness_label,
    ET,
)
from feedback_loop.analyst_feedback import get_quality_metrics
from agents.daily_review import normalize_daily_review_exit_policy
from utils.shadow_ledger import ensure_shadow_ledger_schema
from utils.dialect_sql import _date_cutoff_filter
from utils.market_data_reliability.config import ReliabilityConfig
from utils.market_data_reliability.dashboard_integration import get_freshness_label
from utils.market_data_reliability.freshness import FreshnessClassifier
from utils.market_data_reliability.snapshot import Snapshot
from utils.market_data_reliability.trust import TrustClassifier
from utils.market_data_reliability.validator import ResponseValidator
from utils.trigger_status import compute_trigger_status
from utils.error_sanitizer import sanitize_error_text

app = Flask(__name__)
engine = init_db("db/paper_trader.db")
ensure_shadow_ledger_schema(engine)

_QUOTE_CACHE_SECONDS = int(os.getenv("DASHBOARD_QUOTE_CACHE_SECONDS", "25"))
_YFINANCE_CIRCUIT_BREAK_SECONDS = int(
    os.getenv("DASHBOARD_YFINANCE_CIRCUIT_BREAK_SECONDS", "600")
)
_quote_cache: dict[str, tuple[float, dict]] = {}
_yfinance_disabled_until = 0.0

_SHADOW_EVENT_TYPES = (
    "pm_reject",
    "pm_not_selected",
    "preflight_excluded",
    "pipeline_executed",
    "pipeline_gate_rejected",
    "pipeline_execution_failed",
    "execution_failed",
    "plan_rejected_at_creation",
    "swing_candidate_rejected",
    "contract_violation_missing_geometry_claim",
)
_SHADOW_EVENT_SQL_LIST = ", ".join(f"'{event_type}'" for event_type in _SHADOW_EVENT_TYPES)
_SHADOW_EVENT_LABELS = {
    "pm_reject": "PM Reject",
    "pm_not_selected": "PM Not Selected",
    "preflight_excluded": "Preflight Excluded",
    "pipeline_executed": "Executed",
    "pipeline_gate_rejected": "Pipeline Gate",
    "pipeline_execution_failed": "Execution Failed",
    "execution_failed": "Execution Failed",
    "plan_rejected_at_creation": "Plan Creation",
    "swing_candidate_rejected": "Swing Candidate",
    "contract_violation_missing_geometry_claim": "Contract Violation",
}


def _parse_shadow_event_data(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _first_present(*values: object) -> object:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _first_text(*values: object) -> str | None:
    value = _first_present(*values)
    if value is None:
        return None
    return str(value)


def _format_shadow_blocking_reasons(data: dict, row: dict) -> str | None:
    codes = data.get("blocking_reason_codes")
    if not isinstance(codes, list) or not codes:
        return None

    reason = ", ".join(str(code) for code in codes if code)
    risk_reward = _first_present(data.get("risk_reward"), row.get("risk_reward"))
    if risk_reward is not None and "min_risk_reward_not_met" in reason:
        reason = f"{reason} (R:R {risk_reward})"
    return reason or None


def _normalize_shadow_event(row: dict) -> dict:
    data = _parse_shadow_event_data(row.get("event_data"))
    event_type = row.get("event_type")
    gate_verdict = "executed" if event_type == "pipeline_executed" else "pending"
    raw_label = data.get("raw_label")
    reason_code = data.get("reason_code") or data.get("rejection_reason_code")
    reason = _first_text(
        data.get("rationale"),
        data.get("reason"),
        data.get("block_reason"),
        _format_shadow_blocking_reasons(data, row),
        row.get("rejection_reason"),
        reason_code,
        data.get("error"),
        raw_label,
    )
    if raw_label and reason_code and reason == str(reason_code):
        reason = f"{reason_code}: {raw_label}"

    return {
        "id": f"event-{row.get('event_id')}",
        "source": "pm_candidate_events",
        "created_at": row.get("created_at"),
        "symbol": _first_text(data.get("symbol"), row.get("symbol"), data.get("signal_id")),
        "action": None,
        "direction": _first_text(data.get("direction"), row.get("direction")),
        "profile": _first_text(row.get("profile_id"), data.get("profile"), row.get("candidate_profile")),
        "setup_type": _first_text(data.get("setup_type"), row.get("setup_type"), row.get("geometry_name"), raw_label),
        "entry_price": _first_present(
            data.get("entry_price"),
            row.get("entry_price"),
            data.get("observed_price"),
            data.get("entry_zone_upper"),
        ),
        "stop_price": _first_present(data.get("stop_price"), row.get("stop_price")),
        "target_price": _first_present(data.get("target_price"), row.get("target_price")),
        "blocked_by": _SHADOW_EVENT_LABELS.get(str(event_type), str(event_type or "PM Telemetry")),
        "block_reason": reason,
        "row_type": "event",
        "display_window": "event",
        "eval_window": None,
        "evaluated_at": None,
        "eval_price": None,
        "pnl_pct": row.get("trade_pnl_pct") if event_type == "pipeline_executed" else None,
        "mfe_pct": None,
        "mae_pct": None,
        "stop_hit": None,
        "target_hit": None,
        "first_hit": None,
        "outcome_label": event_type,
        "gate_verdict": gate_verdict,
        "candidate_id": row.get("candidate_id"),
        "trade_id": row.get("trade_id"),
        "trade_status": row.get("trade_status"),
        "trade_entry_time": row.get("trade_entry_time"),
        "trade_exit_time": row.get("trade_exit_time"),
        "trade_pnl": row.get("trade_pnl"),
        "trade_pnl_pct": row.get("trade_pnl_pct"),
        "trade_exit_reason": row.get("trade_exit_reason"),
    }


def _shadow_sort_key(row: dict) -> str:
    value = row.get("created_at")
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value or "")


def _parse_quote_timestamp(value: object, fallback: datetime) -> datetime:
    """Parse dashboard quote timestamp values into timezone-aware UTC datetimes."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        try:
            parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            parsed = fallback
    elif isinstance(value, str) and value.strip():
        raw = value.strip()
        try:
            parsed = datetime.fromisoformat(raw)
        except (ValueError, TypeError):
            try:
                parsed = datetime.fromtimestamp(float(raw), tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                parsed = fallback
    else:
        parsed = fallback

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _to_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _build_dashboard_market_data_snapshot(
    symbol: str,
    quote: dict,
    market_open: bool,
    now: datetime | None = None,
) -> Snapshot | None:
    """Build a dashboard reliability snapshot from already-fetched quote data.

    This intentionally avoids an extra provider call on each dashboard refresh.
    """
    mode = os.getenv("MARKET_DATA_RELIABILITY_MODE", "disabled")
    if mode == "disabled":
        return None

    fetched_at = now or datetime.now(timezone.utc)
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)

    price = quote.get("price")
    raw = {
        "s": symbol,
        "c": price,
        "o": quote.get("_open"),
        "h": quote.get("_high"),
        "l": quote.get("_low"),
        "pc": quote.get("_prev_close"),
        "timestamp": quote.get("_timestamp"),
    }

    config = ReliabilityConfig.from_environment()
    market_session = "open" if market_open else "closed"
    validator = ResponseValidator(
        staleness_threshold_seconds=(
            config.freshness_thresholds[("quote", "display")].aging_threshold
        )
    )
    validation_result = validator.validate(raw, symbol, "quote")

    source_timestamp = _parse_quote_timestamp(quote.get("_timestamp"), fetched_at)
    age_seconds = max(0.0, (fetched_at - source_timestamp).total_seconds())
    freshness_state = FreshnessClassifier(config.freshness_thresholds).classify(
        age_seconds, "quote", "Dashboard_API", market_session
    )
    trust_state, degradation_reasons = TrustClassifier().classify(
        validation_result, freshness_state, "Dashboard_API", market_session
    )

    provider_status = "success"
    if "rate_limited" in validation_result.degradation_reasons:
        provider_status = "rate_limited"
    elif "provider_error" in validation_result.degradation_reasons:
        provider_status = "error"
    elif "empty_response" in validation_result.degradation_reasons:
        provider_status = "empty"
    elif not validation_result.is_valid:
        provider_status = "error"

    return Snapshot(
        symbol=symbol,
        data_type="quote",
        requested_at=fetched_at,
        provider=quote.get("_provider", "unknown"),
        provider_status=provider_status,
        market_session=market_session,
        last_price=_to_decimal(price),
        bid=None,
        ask=None,
        previous_close=_to_decimal(quote.get("_prev_close")),
        open=_to_decimal(quote.get("_open")),
        high=_to_decimal(quote.get("_high")),
        low=_to_decimal(quote.get("_low")),
        volume=None,
        fetched_at=fetched_at,
        source_timestamp=source_timestamp,
        age_seconds=age_seconds,
        freshness_state=freshness_state,
        trust_state=trust_state,
        degradation_reasons=tuple(
            dict.fromkeys((*validation_result.degradation_reasons, *degradation_reasons))
        ),
        raw_provider_latency_ms=None,
        fallback_primary_provider=None,
    )


def _add_market_data_reliability_fields(
    row: dict,
    snapshot: Snapshot | None,
) -> dict:
    """Attach market-data reliability fields without touching catalyst labels."""
    if snapshot is None:
        return row

    try:
        row["freshness_state"] = snapshot.freshness_state
        row["trust_state"] = snapshot.trust_state
        row["market_data_freshness_label"] = get_freshness_label(
            snapshot.freshness_state, snapshot.trust_state
        )
        row["is_actionable"] = (
            snapshot.trust_state == "trusted"
            and snapshot.freshness_state in ("fresh", "aging")
        )
    except Exception:
        log.error(
            "Dashboard market-data reliability enrichment failed for %s",
            getattr(snapshot, "symbol", "unknown"),
            exc_info=True,
        )

    return row


def get_quotes(symbols: list[str]) -> dict:
    """Get current prices via Finnhub, falling back to yfinance if Finnhub fails."""
    global _yfinance_disabled_until
    quotes = {}
    finnhub = None
    now = time.time()
    yfinance_available = now >= _yfinance_disabled_until
    yf = None
    for sym in symbols:
        cached = _quote_cache.get(sym)
        if cached and now - cached[0] < _QUOTE_CACHE_SECONDS:
            quotes[sym] = cached[1]
            continue

        try:
            if finnhub is None:
                finnhub = FinnhubClient()
            q = finnhub.get_quote(sym, retries=0)
            price = round(float(q.get("price", 0)), 2)
            if price <= 0:
                raise ValueError(f"Finnhub returned non-positive price for {sym}: {price}")
            quote = {
                "price": price,
                "change_pct": float(q.get("change_pct", 0)),
                "_provider": "finnhub",
                "_timestamp": q.get("timestamp"),
                "_open": q.get("open"),
                "_high": q.get("high"),
                "_low": q.get("low"),
                "_prev_close": q.get("prev_close"),
            }
            quotes[sym] = quote
            _quote_cache[sym] = (now, quote)
            continue
        except Exception as e:
            safe_error = sanitize_error_text(e)
            if yfinance_available:
                log.warning("Finnhub quote failed for %s; trying yfinance fallback: %s", sym, safe_error)
            else:
                log.warning("Finnhub quote failed for %s; yfinance fallback circuit-open: %s", sym, safe_error)

        if yfinance_available:
            try:
                if yf is None:
                    import yfinance as yf
                t = yf.Ticker(sym)
                info = t.fast_info
                price = float(info.get("lastPrice", 0))
                prev = float(info.get("previousClose", price))
                if price <= 0:
                    raise ValueError(f"yfinance returned non-positive price for {sym}: {price}")
                change_pct = round((price - prev) / prev * 100, 2) if prev else 0
                quote = {
                    "price": round(price, 2),
                    "change_pct": change_pct,
                    "_provider": "yfinance",
                    "_timestamp": datetime.now(timezone.utc).isoformat(),
                    "_prev_close": prev,
                }
                quotes[sym] = quote
                _quote_cache[sym] = (now, quote)
                continue
            except Exception as e:
                yfinance_available = False
                _yfinance_disabled_until = now + _YFINANCE_CIRCUIT_BREAK_SECONDS
                log.warning(
                    "yfinance quote fallback failed for %s; disabled for %ss: %s",
                    sym,
                    _YFINANCE_CIRCUIT_BREAK_SECONDS,
                    e,
                )

        quotes[sym] = {
            "price": 0,
            "change_pct": 0,
            "_provider": "none",
            "_timestamp": datetime.now(timezone.utc).isoformat(),
        }
    return quotes


def get_market_open() -> bool:
    try:
        return FinnhubClient().is_market_open(retries=0)
    except Exception:
        return False


def get_analyst_signals(db, symbols: list[str]) -> dict:
    """Return latest analyst signal per symbol, ignoring entries older than 36 hours."""
    cutoff = datetime.utcnow() - timedelta(hours=36)
    signals = {}
    for sym in symbols:
        mem = (
            db.query(AgentMemory)
            .filter_by(agent="analyst", symbol=sym, key="signal")
            .filter(AgentMemory.timestamp >= cutoff)
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if mem:
            signals[sym] = json.loads(mem.value)
    return signals


def get_researcher_sentiment(db, symbols: list[str]) -> dict:
    """Return latest researcher sentiment per symbol, ignoring entries older than 36 hours."""
    cutoff = datetime.utcnow() - timedelta(hours=36)
    sentiment = {}
    for sym in symbols:
        mem = (
            db.query(AgentMemory)
            .filter_by(agent="researcher", symbol=sym, key="sentiment")
            .filter(AgentMemory.timestamp >= cutoff)
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if mem:
            sentiment[sym] = json.loads(mem.value)
    return sentiment


def get_scout_picks(db) -> list:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    mem = (
        db.query(AgentMemory)
        .filter_by(agent="scout", key="daily_picks")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    if not mem:
        return []
    data = json.loads(mem.value)
    if data.get("date") != today:
        return []
    return data.get("picks", [])


def get_active_focus(db) -> dict:
    today = datetime.utcnow().date().isoformat()
    mem = (
        db.query(AgentMemory)
        .filter(AgentMemory.agent == "scout")
        .filter(AgentMemory.key.like(f"focus_list:{today}:%"))
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    if not mem:
        return {"symbols": [], "context": None, "generated_at": None, "max_symbols": None}
    try:
        data = json.loads(mem.value)
    except Exception:
        log.warning("Active focus memory row is not valid JSON: %s", mem.key)
        return {"symbols": [], "context": None, "generated_at": None, "max_symbols": None}
    if data.get("date") != today:
        return {"symbols": [], "context": None, "generated_at": None, "max_symbols": None}

    symbols = []
    for symbol in data.get("selected", []):
        sym = str(symbol or "").strip().upper()
        if sym and sym not in symbols:
            symbols.append(sym)
    return {
        "symbols": symbols,
        "context": data.get("context"),
        "generated_at": data.get("generated_at"),
        "max_symbols": data.get("max_symbols"),
    }


def get_portfolio_summary(db) -> dict:
    summaries = {}
    for profile_id in ACTIVE_PROFILES:
        positions = db.query(Position).filter_by(profile=profile_id).all()
        bal = (
            db.query(Balance)
            .filter_by(profile=profile_id)
            .order_by(Balance.timestamp.desc())
            .first()
        )
        starting = float(PM_PROFILES[profile_id]["starting_balance"])
        cash = bal.cash if bal else starting
        pos_value = sum(p.quantity * p.avg_cost for p in positions)
        equity = cash + pos_value
        pnl = equity - starting

        summaries[profile_id] = {
            "name": PM_PROFILES[profile_id]["name"],
            "emoji": PM_PROFILES[profile_id]["emoji"],
            "equity": round(equity, 2),
            "cash": round(cash, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / starting * 100, 2),
            "position_count": len(positions),
        }
    return summaries


# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    db = get_session(engine)
    core = [s.strip() for s in os.getenv("WATCHLIST", "SPY,QQQ,IWM,DIA,TLT,GLD,XLK,XLF,XLE,TSLA,NVDA,AMD").split(",")]
    scout_picks = get_scout_picks(db)
    scout_symbols = [p["symbol"] for p in scout_picks]
    active_focus = get_active_focus(db)
    focus_symbol_list = active_focus.get("symbols", [])
    focus_symbols = set(focus_symbol_list)
    all_symbols = (
        core
        + [s for s in scout_symbols if s not in core]
        + [s for s in focus_symbol_list if s not in core and s not in scout_symbols]
    )

    signals = get_analyst_signals(db, all_symbols)
    sentiment = get_researcher_sentiment(db, all_symbols)
    portfolio = get_portfolio_summary(db)
    quotes = get_quotes(all_symbols)
    market_open = get_market_open()

    # Compute now_et ONCE and thread it through all freshness calls
    now_et = datetime.now(ET)
    market_day_start = get_market_day_start(now_et)

    # Breaking news — isolated from sentiment query
    breaking_news_by_symbol = {}
    try:
        breaking_news_by_symbol = get_breaking_news_for_symbols(
            db, all_symbols, market_day_start
        )
    except Exception as e:
        log.error(f"Breaking news query failed: {e}")
        breaking_news_by_symbol = {sym: [] for sym in all_symbols}

    # Catalyst freshness — isolated
    # Pass pre-fetched breaking_news to avoid redundant DB query
    freshness_by_symbol = {}
    try:
        freshness_by_symbol = compute_catalyst_freshness(
            db, all_symbols, now=now_et,
            breaking_news_by_symbol=breaking_news_by_symbol,
        )
    except Exception as e:
        log.error(f"Catalyst freshness computation failed: {e}")

    watchlist = []
    for sym in all_symbols:
        q = quotes.get(sym, {})
        sig = signals.get(sym, {})
        sent = sentiment.get(sym, {})
        trigger_status = sig.get("trigger_status", {})
        if not isinstance(trigger_status, dict) or not trigger_status:
            try:
                trigger_status = compute_trigger_status(sig, q, {})
            except Exception as exc:
                log.warning("Dashboard trigger_status derivation failed for %s: %s", sym, exc)
                trigger_status = {}
        row = {
            "symbol": sym,
            "is_scout": sym in scout_symbols,
            "is_focus": sym in focus_symbols,
            "price": q.get("price", 0),
            "change_pct": q.get("change_pct", 0),
            "signal": sig.get("signal", "—"),
            "strength": sig.get("strength", "—"),
            "confidence": sig.get("confidence", "—"),
            "setup_type": sig.get("setup_type", "—"),
            "trigger_status": trigger_status,
            "setup_reasoning": sig.get("setup_reasoning", ""),
            "reasoning": sig.get("reasoning", ""),
            "invalidation": sig.get("invalidation", ""),
            "key_levels": sig.get("key_levels", {}),
            "sentiment": sent.get("sentiment", "—"),
            "catalysts": sent.get("catalysts", []),
            "risks": sent.get("risks", []),
            "breaking_news": breaking_news_by_symbol.get(sym, []),
            "catalyst_freshness": freshness_by_symbol.get(sym, {}),
            "freshness_label": build_freshness_label(sym, freshness_by_symbol.get(sym, {})),
            "market_state": sig.get("market_state", "confounded"),
            "timeframe_authority": sig.get("timeframe_authority", {}),
            "setup_lifecycle_state": sig.get("setup_lifecycle_state", "no_setup"),
            "if_then_triggers": sig.get("if_then_triggers", [])[:4],
            "setup_reclassification": sig.get("setup_reclassification"),
        }
        snapshot = _build_dashboard_market_data_snapshot(sym, q, market_open)
        watchlist.append(_add_market_data_reliability_fields(row, snapshot))

    # Market regime from quant researcher
    regime = None
    mem = (
        db.query(AgentMemory)
        .filter_by(agent="quant_researcher", key="regime")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    if mem:
        try:
            regime_data = json.loads(mem.value)
            regime = regime_data.get("regime") or regime_data.get("market_regime")
        except Exception:
            regime = mem.value

    db.close()
    return jsonify({
        "timestamp": datetime.utcnow().isoformat(),
        "market_open": market_open,
        "watchlist": watchlist,
        "analysis": watchlist,
        "scout_picks": scout_picks,
        "active_focus": active_focus,
        "portfolio": portfolio,
        "regime": regime,
    })


@app.route("/api/watch-candidates")
def api_watch_candidates():
    """Return active watch candidates, optionally filtered by profile_id."""
    from utils.gate_config import MARKET_STATE_MODE
    if MARKET_STATE_MODE == "disabled":
        return jsonify([])

    profile_id = request.args.get("profile_id")
    try:
        from sqlalchemy import text as _text
        with engine.connect() as conn:
            if profile_id:
                rows = conn.execute(
                    _text(
                        "SELECT watch_id, symbol, direction_watch, trade_posture, "
                        "       activation_conditions_json, invalidation_conditions_json, "
                        "       state, created_at, expires_at, market_state, setup_lifecycle_state "
                        "FROM watch_candidates "
                        "WHERE state = 'active' AND profile_id = :pid"
                    ),
                    {"pid": profile_id},
                ).fetchall()
            else:
                rows = conn.execute(
                    _text(
                        "SELECT watch_id, symbol, direction_watch, trade_posture, "
                        "       activation_conditions_json, invalidation_conditions_json, "
                        "       state, created_at, expires_at, market_state, setup_lifecycle_state "
                        "FROM watch_candidates "
                        "WHERE state = 'active'"
                    )
                ).fetchall()

        result = []
        for row in rows:
            import json as _json
            result.append({
                "watch_id": row[0],
                "symbol": row[1],
                "direction": row[2],
                "posture": row[3],
                "activation_levels": _json.loads(row[4] or "[]"),
                "invalidation_levels": _json.loads(row[5] or "[]"),
                "state": row[6],
                "created_at": row[7],
                "expires_at": row[8],
                "market_state": row[9],
                "lifecycle_state": row[10],
            })
        return jsonify(result)
    except Exception as exc:
        log.warning("Watch candidates API failed: %s", exc)
        return jsonify([])


@app.route("/api/positions")
def api_positions():
    db = get_session(engine)
    fh = FinnhubClient()

    # Reconcile orphaned positions: remove positions with no matching open trade
    all_positions = db.query(Position).all()
    for p in all_positions:
        has_open_trade = (
            db.query(Trade)
            .filter_by(symbol=p.symbol, profile=p.profile, status="open")
            .first()
        )
        if not has_open_trade:
            log.warning("Reconciled orphan position: %s %s (%s)", p.symbol, p.side, p.profile)
            db.delete(p)
    db.commit()

    result = []
    for profile_id in ACTIVE_PROFILES:
        positions = db.query(Position).filter_by(profile=profile_id).all()
        for p in positions:
            try:
                q = fh.get_quote(p.symbol)
                current_price = q.get("price", p.avg_cost)
            except Exception:
                current_price = p.avg_cost
            if p.side == "long":
                unrealized_pnl = (current_price - p.avg_cost) * p.quantity
            else:
                unrealized_pnl = (p.avg_cost - current_price) * p.quantity
            unrealized_pct = (unrealized_pnl / (p.avg_cost * p.quantity)) * 100 if (p.avg_cost and p.quantity) else 0

            # pull stop/target from the trade record
            stop = target = None
            open_trade = (
                db.query(Trade)
                .filter_by(symbol=p.symbol, profile=profile_id, status="open")
                .order_by(Trade.entry_time.desc())
                .first()
            )
            if open_trade:
                stop = open_trade.stop_price
                target = open_trade.target_price

            result.append({
                "profile": profile_id,
                "symbol": p.symbol,
                "side": p.side,
                "quantity": p.quantity,
                "avg_cost": p.avg_cost,
                "current_price": round(current_price, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "unrealized_pct": round(unrealized_pct, 2),
                "stop": stop,
                "target": target,
                "opened_at": p.opened_at.isoformat() if p.opened_at else None,
            })
    db.close()
    return jsonify(result)


@app.route("/api/trades")
def api_trades():
    db = get_session(engine)
    import yfinance as yf

    # Surface orphaned trades without mutating them. Silent reconciliation hides
    # accounting bugs by creating closed trades with no exit price.
    open_trades_all = db.query(Trade).filter_by(status="open").all()
    orphan_trade_ids = set()
    for ot in open_trades_all:
        side = "long" if ot.direction == "LONG" else "short"
        matching_pos = db.query(Position).filter_by(
            symbol=ot.symbol, profile=ot.profile, side=side
        ).first()
        if not matching_pos:
            orphan_trade_ids.add(ot.id)
            log.warning("Open orphan trade #%d (%s %s) has no matching position", ot.id, ot.symbol, ot.profile)

    cutoff = datetime.utcnow() - timedelta(days=30)
    trades = (
        db.query(Trade)
        .filter(Trade.entry_time >= cutoff)
        .order_by(Trade.entry_time.desc())
        .limit(100)
        .all()
    )

    # Batch fetch current prices for open trades
    open_symbols = list(set(t.symbol for t in trades if t.status == "open"))
    current_prices = {}
    for sym in open_symbols:
        try:
            current_prices[sym] = float(yf.Ticker(sym).fast_info.get("lastPrice", 0))
        except Exception:
            pass

    result = []
    for t in trades:
        unrealized_pnl = None
        unrealized_pct = None
        if t.status == "open" and t.symbol in current_prices:
            price = current_prices[t.symbol]
            if t.direction == "LONG":
                unrealized_pnl = round((price - t.entry_price) * t.quantity, 2)
            else:
                unrealized_pnl = round((t.entry_price - price) * t.quantity, 2)
            unrealized_pct = round(unrealized_pnl / (t.entry_price * t.quantity) * 100, 2) if t.entry_price and t.quantity else 0

        result.append({
            "id": t.id,
            "profile": t.profile,
            "symbol": t.symbol,
            "direction": t.direction,
            "quantity": t.quantity,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "entry_time": t.entry_time.isoformat() if t.entry_time else None,
            "exit_time": t.exit_time.isoformat() if t.exit_time else None,
            "status": t.status,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "pm_candidate_id": getattr(t, "pm_candidate_id", None),
            "candidate_lineage_id": getattr(t, "candidate_lineage_id", None),
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pct": unrealized_pct,
            "review_score": t.review_score,
            "orphaned": t.id in orphan_trade_ids,
        })
    db.close()
    return jsonify(result)


@app.route("/api/daily")
def api_daily():
    db = get_session(engine)
    logs = db.query(DailyLog).order_by(DailyLog.date.desc()).limit(30).all()
    result = [
        {
            "date": l.date,
            "daily_pnl": l.daily_pnl,
            "daily_pnl_pct": l.daily_pnl_pct,
            "trades": l.trades_taken,
            "wins": l.winning_trades,
            "losses": l.losing_trades,
        }
        for l in logs
    ]
    db.close()
    return jsonify(result)


@app.route("/api/feedback")
def api_feedback():
    db = get_session(engine)
    result = {"selection": None, "execution": {}, "regime": None, "quant": None}

    # Selection feedback (for Scout + Analyst)
    mem = (
        db.query(AgentMemory)
        .filter_by(agent="reviewer", key="selection_feedback")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    if mem:
        try:
            result["selection"] = json.loads(mem.value)
        except Exception:
            result["selection"] = {"feedback": mem.value}

    # Execution feedback per profile
    for profile_id in ACTIVE_PROFILES:
        mem = (
            db.query(AgentMemory)
            .filter_by(agent="reviewer", key="execution_feedback")
            .filter(AgentMemory.symbol == profile_id)
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if mem:
            try:
                result["execution"][profile_id] = json.loads(mem.value)
            except Exception:
                result["execution"][profile_id] = {"feedback": mem.value}

    # Market regime from quant researcher
    mem = (
        db.query(AgentMemory)
        .filter_by(agent="quant_researcher", key="regime")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    if mem:
        try:
            result["regime"] = json.loads(mem.value)
        except Exception:
            result["regime"] = {"regime": mem.value}

    # Weekly stance per profile
    stances = {}
    for profile_id in ACTIVE_PROFILES:
        mem = (
            db.query(AgentMemory)
            .filter_by(agent="weekly_prep", symbol=profile_id, key="stance")
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if mem:
            try:
                stances[profile_id] = json.loads(mem.value)
            except Exception:
                pass
    result["stances"] = stances
    result["analyst_quality"] = get_quality_metrics(engine)

    db.close()
    return jsonify(result)


@app.route("/api/performance")
def api_performance():
    from models.case import Case
    from sqlalchemy import func
    from sqlalchemy.types import Integer as SAInteger
    from db.schema import DynamicStrategy
    db = get_session(engine)

    # Win rates by setup type
    setup_rows = (
        db.query(
            Case.setup_type,
            func.count(Case.id).label("total"),
            func.sum((Case.outcome == "success").cast(SAInteger)).label("wins"),
            func.avg(Case.pnl_pct).label("avg_pnl"),
            func.avg(Case.review_score).label("avg_score"),
        )
        .filter(Case.setup_type != None)
        .group_by(Case.setup_type)
        .all()
    )
    setup_stats = [
        {
            "setup_type": r.setup_type,
            "total": r.total,
            "wins": r.wins or 0,
            "win_rate": round((r.wins or 0) / r.total * 100, 1),
            "avg_pnl": round(r.avg_pnl or 0, 2),
            "avg_score": round(r.avg_score or 0, 1),
        }
        for r in setup_rows
    ]

    # Per-profile stats
    profile_stats = {}
    for profile_id in ACTIVE_PROFILES:
        trades = db.query(Trade).filter_by(profile=profile_id, status="closed").all()
        if not trades:
            profile_stats[profile_id] = {"total": 0, "wins": 0, "win_rate": 0, "avg_pnl": 0, "total_pnl": 0}
            continue
        wins = sum(1 for t in trades if (t.pnl or 0) > 0)
        total_pnl = sum(t.pnl or 0 for t in trades)
        avg_pnl = total_pnl / len(trades)
        profile_stats[profile_id] = {
            "total": len(trades),
            "wins": wins,
            "win_rate": round(wins / len(trades) * 100, 1),
            "avg_pnl": round(avg_pnl, 2),
            "total_pnl": round(total_pnl, 2),
        }

    # Recent cases for case library preview
    cases = (
        db.query(Case)
        .order_by(Case.created_at.desc())
        .limit(20)
        .all()
    )
    case_list = [
        {
            "date": c.date,
            "symbol": c.symbol,
            "setup_type": c.setup_type,
            "outcome": c.outcome,
            "pnl_pct": c.pnl_pct,
            "lesson": c.lesson,
            "selection_score": c.selection_score,
            "execution_score": c.execution_score,
            "profile": c.profile,
        }
        for c in cases
    ]

    # Dynamic strategies
    dynamic_strats = db.query(DynamicStrategy).order_by(DynamicStrategy.created_at.desc()).all()
    dynamic_list = [
        {
            "key": d.key,
            "name": d.name,
            "description": d.description,
            "status": d.status,
            "total_trades": d.total_trades,
            "wins": d.wins,
            "win_rate": d.win_rate,
            "avg_pnl_pct": d.avg_pnl_pct,
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "retired_at": d.retired_at.isoformat() if d.retired_at else None,
            "retire_reason": d.retire_reason,
        }
        for d in dynamic_strats
    ]

    db.close()
    return jsonify({
        "setup_stats": setup_stats,
        "profile_stats": profile_stats,
        "cases": case_list,
        "dynamic_strategies": dynamic_list,
    })


@app.route("/api/alerts")
def api_alerts():
    """Return the latest live alerts from the price monitor, deduplicated per symbol."""
    db = get_session(engine)
    cutoff = datetime.utcnow() - timedelta(minutes=15)
    rows = (
        db.query(AgentMemory)
        .filter_by(agent="price_monitor", key="live_alerts")
        .filter(AgentMemory.timestamp >= cutoff)
        .order_by(AgentMemory.timestamp.desc())
        .limit(5)
        .all()
    )

    # Collect all alerts, keeping only the most recent per symbol+type
    by_sym_type = {}  # (symbol, type) -> alert
    for row in rows:
        try:
            batch = json.loads(row.value)
            for a in batch:
                key = (a.get("symbol"), a.get("type"))
                if key not in by_sym_type:
                    by_sym_type[key] = a
        except Exception:
            pass

    # For approaching alerts, consolidate all levels into one alert per symbol
    consolidated = []
    approaching_by_sym = {}  # symbol -> list of level details
    for (sym, atype), a in by_sym_type.items():
        if atype == "approaching":
            approaching_by_sym.setdefault(sym, []).append(a)
        else:
            consolidated.append(a)

    for sym, alerts in approaching_by_sym.items():
        # Pick the closest level as the primary, list others as context
        alerts.sort(key=lambda x: x.get("detail", ""))
        levels = []
        for a in alerts:
            detail = a.get("detail", "")
            # Extract level name from "within X% of name=value"
            m = re.search(r"of (\w+)=", detail)
            if m:
                levels.append(m.group(1))
        primary = alerts[0]
        level_summary = ", ".join(levels[:4])
        if len(levels) > 4:
            level_summary += f" +{len(levels)-4}"
        consolidated.append({
            "type": "approaching",
            "symbol": sym,
            "price": primary["price"],
            "detail": f"near {level_summary}",
            "timestamp": primary.get("timestamp"),
        })

    # Sort: rapid_move and stop first, then entry, then approaching
    type_order = {"market_data_outage": 0, "rapid_move": 0, "stop": 0, "thesis_invalidation": 0,
                  "entry": 1, "breakdown": 1, "breakout": 1, "approaching": 2}
    consolidated.sort(key=lambda a: type_order.get(a.get("type"), 3))

    db.close()
    return jsonify(consolidated)


@app.route("/api/decisions")
def api_decisions():
    db = get_session(engine)
    cutoff = datetime.utcnow() - timedelta(days=7)
    result = []

    for profile_id in ACTIVE_PROFILES:
        notes = (
            db.query(AgentMemory)
            .filter_by(agent=f"pm_{profile_id}", key="notes")
            .filter(AgentMemory.timestamp >= cutoff)
            .order_by(AgentMemory.timestamp.desc())
            .all()
        )
        for n in notes:
            result.append({
                "profile": profile_id,
                "type": "notes",
                "timestamp": n.timestamp.isoformat() if n.timestamp else None,
                "content": n.value,
            })

    trades = (
        db.query(Trade)
        .filter(Trade.entry_time >= cutoff)
        .order_by(Trade.entry_time.desc())
        .all()
    )
    for t in trades:
        result.append({
            "profile": t.profile,
            "type": "trade",
            "timestamp": t.entry_time.isoformat() if t.entry_time else None,
            "content": f"{t.direction} {t.quantity} {t.symbol} @ ${t.entry_price:.2f}"
                       + (f" | stop: ${t.stop_price:.2f}" if t.stop_price else "")
                       + (f" | target: ${t.target_price:.2f}" if t.target_price else "")
                       + (f" | {t.reason_entry}" if t.reason_entry else ""),
            "status": t.status,
            "pnl": t.pnl,
        })

    # Sort all by timestamp descending
    result.sort(key=lambda x: x.get("timestamp") or "", reverse=True)

    db.close()
    return jsonify(result)


@app.route("/api/meta_review")
def api_meta_review():
    db = get_session(engine)
    mem = (
        db.query(AgentMemory)
        .filter_by(agent="meta_reviewer", key="weekly_review")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    result = {}
    if mem:
        try:
            result = json.loads(mem.value)
        except Exception:
            result = {"raw": mem.value}

    # Code suggestions
    code_mem = (
        db.query(AgentMemory)
        .filter_by(agent="meta_reviewer", key="code_suggestions")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    if code_mem:
        try:
            result["code_suggestions"] = json.loads(code_mem.value)
        except Exception:
            pass

    db.close()
    return jsonify(result)


@app.route("/api/strategies")
def api_strategies():
    from models.strategies import STRATEGIES, SETUP_TYPE_MAP
    from utils.strategy_store import get_all_strategies
    result = {}
    # Map setup_type names to strategy descriptions
    for setup_type, strategy_key in SETUP_TYPE_MAP.items():
        strat = STRATEGIES.get(strategy_key, {})
        result[setup_type] = {
            "name": strat.get("name", setup_type),
            "description": strat.get("description", ""),
            "timeframe": strat.get("timeframe", ""),
            "bias": strat.get("bias", ""),
            "win_rate": strat.get("win_rate_documented"),
            "source": "hardcoded",
        }
    # Add direct strategy keys
    for key, strat in STRATEGIES.items():
        if key not in result:
            result[key] = {
                "name": strat.get("name", key),
                "description": strat.get("description", ""),
                "timeframe": strat.get("timeframe", ""),
                "bias": strat.get("bias", ""),
                "win_rate": strat.get("win_rate_documented"),
                "source": "hardcoded",
            }
    # Add dynamic strategies
    all_strats = get_all_strategies(engine)
    for key, strat in all_strats.items():
        if strat.get("source") == "dynamic":
            result[key] = {
                "name": strat.get("name", key),
                "description": strat.get("description", ""),
                "timeframe": strat.get("timeframe", ""),
                "bias": strat.get("bias", ""),
                "win_rate": strat.get("win_rate_documented"),
                "source": "dynamic",
                "total_trades": strat.get("total_trades", 0),
                "status": strat.get("status"),
            }
    return jsonify(result)


@app.route("/api/company/<symbol>")
def api_company(symbol):

    try:
        fh = FinnhubClient()
        fh._rate_limit()
        profile = fh.client.company_profile2(symbol=symbol)
        return jsonify({
            "name": profile.get("name"),
            "industry": profile.get("finnhubIndustry"),
            "market_cap": profile.get("marketCapitalization"),
            "employees": profile.get("employeeTotal"),
            "country": profile.get("country"),
            "exchange": profile.get("exchange"),
            "ipo": profile.get("ipo"),
            "logo": profile.get("logo"),
            "weburl": profile.get("weburl"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/journal")
def api_journal():
    """Returns Daily_Review entries from agent_memory, paginated.
    Only returns the fields needed by the Journal tab UI to keep responses small."""
    try:
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 10))
    except (ValueError, TypeError):
        page = 1
        per_page = 10

    if page < 1:
        page = 1
    if per_page < 1:
        per_page = 10

    offset = (page - 1) * per_page

    # Fields the Journal tab UI actually renders
    _UI_FIELDS = [
        "date", "generated_at",
        "executive_summary", "day_classification", "primary_driver",
        "driver_ranking", "performance_story", "system_observations",
        "what_worked", "what_failed", "highest_leverage_fix",
        "lessons_learned", "process_quality", "correlations",
        "git_narrative", "tomorrows_focus", "watchouts",
        "email_subject", "email_preview", "completeness",
        # Legacy fields for backward compat with old entries
        "market_summary", "trade_narrative", "outlook",
    ]

    db = get_session(engine)
    try:
        entries = (
            db.query(AgentMemory)
            .filter_by(agent="daily_review", key="daily_review")
            .order_by(AgentMemory.symbol.desc())
            .offset(offset)
            .limit(per_page)
            .all()
        )

        result = []
        for entry in entries:
            try:
                review = normalize_daily_review_exit_policy(json.loads(entry.value))
                # Build a slim response with only UI-relevant fields
                slim = {k: review[k] for k in _UI_FIELDS if k in review}
                slim["date"] = entry.symbol

                # Include trade performance summary stats (not the raw object)
                tp = review.get("trade_performance", {})
                if tp and isinstance(tp, dict):
                    slim["trade_stats"] = {
                        "total_trades": tp.get("total_trades", 0),
                        "wins": tp.get("wins", 0),
                        "losses": tp.get("losses", 0),
                        "total_pnl": tp.get("total_pnl", 0),
                        "no_trades": tp.get("no_trades", False),
                    }

                result.append(slim)
            except (json.JSONDecodeError, TypeError):
                continue

        return jsonify(result)
    except Exception:
        return jsonify([])
    finally:
        db.close()


@app.route("/narratives")
def narratives_page():
    return render_template("narratives.html")


@app.route("/api/narratives")
def api_narratives():
    """Returns narrator entries from AgentMemory, paginated, reverse chronological."""
    try:
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 20))
    except (ValueError, TypeError):
        page, per_page = 1, 20

    if page < 1:
        page = 1
    if per_page < 1 or per_page > 100:
        per_page = 20

    offset = (page - 1) * per_page

    db = get_session(engine)
    try:
        entries = (
            db.query(AgentMemory)
            .filter_by(agent="narrator")
            .order_by(AgentMemory.timestamp.desc())
            .offset(offset)
            .limit(per_page)
            .all()
        )

        total = db.query(AgentMemory).filter_by(agent="narrator").count()

        result = []
        for entry in entries:
            try:
                data = json.loads(entry.value)
                result.append({
                    "update_type": entry.key,
                    "date": entry.symbol,
                    "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
                    "narrative": data.get("narrative", ""),
                    "generated_at": data.get("generated_at"),
                })
            except (json.JSONDecodeError, TypeError):
                continue

        return jsonify({
            "narratives": result,
            "page": page,
            "per_page": per_page,
            "total": total,
        })
    except Exception:
        return jsonify({"narratives": [], "page": page, "per_page": per_page, "total": 0})
    finally:
        db.close()


@app.route("/api/shadow-outcomes")
def api_shadow_outcomes():
    """Return historical shadow outcomes plus current PM rejection telemetry."""
    days = request.args.get("days", default=7, type=int)
    days = max(1, min(days or 7, 30))
    limit = request.args.get("limit", default=100, type=int)
    limit = max(1, min(limit or 100, 500))

    blocked_date_filter = _date_cutoff_filter(engine, "b.created_at")
    modern_date_filter = _date_cutoff_filter(engine, "m.candidate_created_at")
    event_date_filter = _date_cutoff_filter(engine, "e.created_at")
    with engine.connect() as conn:
        summary_rows = conn.execute(
            text(
                f"""
                SELECT
                  COALESCE(o.gate_verdict, 'pending') AS gate_verdict,
                  COUNT(*) AS count
                FROM blocked_trade_candidates b
                LEFT JOIN blocked_trade_candidate_outcomes o
                  ON o.blocked_candidate_id = b.id AND o.eval_window = '60m'
                WHERE {blocked_date_filter}
                GROUP BY COALESCE(o.gate_verdict, 'pending')
                """
            ),
            {"cutoff": f"-{days} days"},
        ).mappings().all()

        modern_summary_rows = conn.execute(
            text(
                f"""
                SELECT
                  COALESCE(m.gate_verdict, 'pending') AS gate_verdict,
                  COUNT(*) AS count
                FROM modern_shadow_candidate_outcomes m
                WHERE m.eval_window = '60m'
                  AND {modern_date_filter}
                GROUP BY COALESCE(m.gate_verdict, 'pending')
                """
            ),
            {"cutoff": f"-{days} days"},
        ).mappings().all()

        rows = conn.execute(
            text(
                f"""
                SELECT
                  b.id, b.created_at, b.symbol, b.action, b.direction, b.profile,
                  b.setup_type, b.entry_price, b.stop_price, b.target_price,
                  b.blocked_by, b.block_reason,
                  o.eval_window, o.evaluated_at, o.eval_price, o.pnl_pct,
                  o.mfe_pct, o.mae_pct, o.stop_hit, o.target_hit, o.first_hit,
                  o.outcome_label, o.gate_verdict,
                  'blocked_trade_candidates' AS source,
                  'outcome' AS row_type,
                  COALESCE(o.eval_window, 'pending') AS display_window
                FROM blocked_trade_candidates b
                LEFT JOIN blocked_trade_candidate_outcomes o
                  ON o.blocked_candidate_id = b.id
                WHERE {blocked_date_filter}
                ORDER BY b.created_at DESC, o.eval_window ASC
                LIMIT :limit
                """
            ),
            {"cutoff": f"-{days} days", "limit": limit},
        ).mappings().all()

        modern_rows = conn.execute(
            text(
                f"""
                SELECT
                  m.id,
                  m.candidate_created_at AS created_at,
                  m.symbol,
                  m.action,
                  m.direction,
                  m.profile,
                  m.setup_type,
                  m.entry_price,
                  m.stop_price,
                  m.target_price,
                  m.blocked_by,
                  m.block_reason,
                  m.eval_window,
                  m.evaluated_at,
                  m.eval_price,
                  m.pnl_pct,
                  m.mfe_pct,
                  m.mae_pct,
                  m.stop_hit,
                  m.target_hit,
                  m.first_hit,
                  m.outcome_label,
                  m.gate_verdict,
                  'modern_shadow_candidate_outcomes' AS source,
                  'outcome' AS row_type,
                  m.eval_window AS display_window,
                  m.candidate_id,
                  m.event_type
                FROM modern_shadow_candidate_outcomes m
                WHERE {modern_date_filter}
                ORDER BY m.candidate_created_at DESC, m.eval_window ASC
                LIMIT :limit
                """
            ),
            {"cutoff": f"-{days} days", "limit": limit},
        ).mappings().all()

        event_count = conn.execute(
            text(
                f"""
                SELECT COUNT(*) AS count
                FROM pm_candidate_events e
                WHERE {event_date_filter}
                  AND e.event_type IN ({_SHADOW_EVENT_SQL_LIST})
                """
            ),
            {"cutoff": f"-{days} days"},
        ).scalar() or 0

        event_summary_rows = conn.execute(
            text(
                f"""
                SELECT e.event_type, COUNT(*) AS count
                FROM pm_candidate_events e
                WHERE {event_date_filter}
                  AND e.event_type IN ({_SHADOW_EVENT_SQL_LIST})
                GROUP BY e.event_type
                """
            ),
            {"cutoff": f"-{days} days"},
        ).mappings().all()

        event_rows = conn.execute(
            text(
                f"""
                SELECT
                  e.id AS event_id,
                  e.created_at,
                  e.candidate_id,
                  e.cycle_id,
                  e.profile_id,
                  e.event_type,
                  e.event_data,
                  e.candidate_type,
                  c.profile_id AS candidate_profile,
                  c.symbol,
                  c.direction,
                  c.setup_type,
                  c.geometry_name,
                  c.entry_price,
                  c.stop_price,
                  c.target_price,
                  c.risk_reward,
                  c.rejection_reason,
                  t.id AS trade_id,
                  t.status AS trade_status,
                  t.entry_time AS trade_entry_time,
                  t.exit_time AS trade_exit_time,
                  t.pnl AS trade_pnl,
                  t.pnl_pct AS trade_pnl_pct,
                  t.reason_exit AS trade_exit_reason
                FROM pm_candidate_events e
                LEFT JOIN pm_candidates c
                  ON c.candidate_id = e.candidate_id
                LEFT JOIN trades t
                  ON t.pm_candidate_id = e.candidate_id
                WHERE {event_date_filter}
                  AND e.event_type IN ({_SHADOW_EVENT_SQL_LIST})
                ORDER BY e.created_at DESC
                LIMIT :limit
                """
            ),
            {"cutoff": f"-{days} days", "limit": limit},
        ).mappings().all()

    summary = {r["gate_verdict"]: r["count"] for r in summary_rows}
    for row in modern_summary_rows:
        verdict = row["gate_verdict"]
        summary[verdict] = summary.get(verdict, 0) + row["count"]
    event_summary = {r["event_type"]: r["count"] for r in event_summary_rows}
    event_summary["total"] = event_count

    combined_rows = [dict(r) for r in rows]
    combined_rows.extend(dict(r) for r in modern_rows)
    combined_rows.extend(_normalize_shadow_event(dict(r)) for r in event_rows)
    combined_rows.sort(key=_shadow_sort_key, reverse=True)
    return jsonify({
        "summary": summary,
        "event_summary": event_summary,
        "rows": combined_rows[:limit],
    })


@app.route("/api/trade-events")
def api_trade_events():
    """Return stop and pending-order lifecycle events for a specific trade."""
    trade_id = request.args.get("trade_id", type=int)
    if not trade_id:
        return jsonify({"error": "trade_id parameter required"}), 400

    db = get_session(engine)

    STOP_EVENT_TYPES = [
        "stop_update_requested",
        "stop_update_accepted",
        "stop_update_rejected",
        "stop_geometry_invalid",
        "stop_triggered",
        "stop_repaired",
        "stop_review_required",
    ]

    # Pending-order events are unioned in from the shared constant rather than a
    # second hardcoded list. Note this only surfaces POST-fill events: this
    # endpoint requires a trade_id, and pending_order_created / _expired /
    # _canceled / _declined are all pre-trade with trade_id = None. Those are
    # served order-scoped by /api/pending_orders and
    # /api/pending_orders/<order_id>/events instead.
    try:
        from utils.gate_config import PENDING_ORDER_EVENT_TYPES

        event_types = STOP_EVENT_TYPES + sorted(PENDING_ORDER_EVENT_TYPES)
    except Exception:
        event_types = STOP_EVENT_TYPES

    events = (
        db.query(TradeEvent)
        .filter(
            TradeEvent.trade_id == trade_id,
            TradeEvent.event_type.in_(event_types),
        )
        .order_by(TradeEvent.timestamp.asc())
        .all()
    )

    result = []
    for e in events:
        payload = None
        if e.payload_json:
            try:
                payload = json.loads(e.payload_json)
            except (json.JSONDecodeError, TypeError):
                payload = e.payload_json

        result.append({
            "id": e.id,
            "trade_id": e.trade_id,
            "event_type": e.event_type,
            "timestamp": e.timestamp.isoformat() if e.timestamp else None,
            "agent": e.agent,
            "symbol": e.symbol,
            "profile": e.profile,
            "price": e.price,
            "message": e.message,
            "payload": payload,
            "pm_candidate_id": getattr(e, "pm_candidate_id", None),
            "candidate_lineage_id": getattr(e, "candidate_lineage_id", None),
        })

    db.close()
    return jsonify(result)


# ---------------------------------------------------------------------------
# Pending Limit Orders
#
# Deliberately separate from /api/decisions. A resting order is NOT a rejected
# decision, and folding it into the notes-and-trades stream is exactly how the
# two got conflated in the first place.
#
# Requirements: 8.8, 10.5, 11.1-11.10, 12.1-12.6
# ---------------------------------------------------------------------------

PENDING_ORDER_ACTIVE_STATES = ("pending", "filling")
PENDING_ORDER_TERMINAL_STATES = ("filled", "expired", "canceled", "rejected")


def _pending_orders_available() -> bool:
    """Whether the pending_orders table exists.

    Lets the dashboard degrade to an empty view instead of erroring when the
    feature has never been initialized (Requirement 11.10).
    """
    try:
        from sqlalchemy import inspect as sa_inspect

        return sa_inspect(engine).has_table("pending_orders")
    except Exception:
        return False


def _pending_order_current_prices(symbols: set[str]) -> dict[str, float]:
    """Latest price per unique symbol, fail-open to an empty mapping.

    Fetched once per symbol, matching how api_positions works.
    """
    prices: dict[str, float] = {}
    if not symbols:
        return prices
    try:
        fh = FinnhubClient()
    except Exception:
        return prices

    for symbol in sorted(symbols):
        try:
            quote = fh.get_quote(symbol, retries=0)
            price = float(quote.get("price") or 0)
            if price > 0:
                prices[symbol] = round(price, 2)
        except Exception:
            continue
    return prices


def _serialize_pending_order(order, current_price, events=None) -> dict:
    """Shape one order for the dashboard.

    `distance_to_limit` is signed from the perspective of "how far the market
    still has to travel": positive means the limit has not been reached yet.
    """
    now = datetime.now(timezone.utc)
    seconds_remaining = None
    if order.expires_at is not None:
        seconds_remaining = max(
            0, int((order.expires_at - now).total_seconds())
        )

    distance = None
    distance_pct = None
    if current_price is not None and order.limit_price:
        if order.side == "BUY":
            distance = round(current_price - order.limit_price, 4)
        else:
            distance = round(order.limit_price - current_price, 4)
        distance_pct = round(distance / order.limit_price * 100, 4)

    payload = {
        "order_id": order.order_id,
        "profile": order.profile_id,
        "symbol": order.symbol,
        "side": order.side,
        "setup_type": order.setup_type,
        "state": order.state.value,
        "is_active": order.state.value in PENDING_ORDER_ACTIVE_STATES,
        "limit_price": order.limit_price,
        "stop_price": order.stop_price,
        "target_price": order.target_price,
        "risk_reward": order.risk_reward,
        "intended_quantity": order.intended_quantity,
        "current_price": current_price,
        "distance_to_limit": distance,
        "distance_to_limit_pct": distance_pct,
        "fresh_price_at_creation": order.fresh_price_at_creation,
        "runaway_pct_at_creation": order.runaway_pct_at_creation,
        "created_at": order.created_at.isoformat() if order.created_at else None,
        "expires_at": order.expires_at.isoformat() if order.expires_at else None,
        "seconds_remaining": seconds_remaining,
        "reason": order.terminal_reason,
        "rationale": order.pm_rationale,
        "fill_price": order.fill_price,
        "fill_policy": order.fill_policy,
        "fill_bar_ts": (
            order.fill_bar_ts.isoformat() if order.fill_bar_ts else None
        ),
        "filled_at": order.filled_at.isoformat() if order.filled_at else None,
        "terminal_at": (
            order.terminal_at.isoformat() if order.terminal_at else None
        ),
        "trade_id": order.trade_id,
        "candidate_id": order.candidate_id,
        "cycle_id": order.cycle_id,
    }
    if events is not None:
        payload["recent_events"] = events
    return payload


def _serialize_pending_order_event(event: dict) -> dict:
    return {
        "id": event.get("id"),
        "order_id": event.get("order_id"),
        "event_type": event.get("event_type"),
        "from_state": event.get("from_state"),
        "to_state": event.get("to_state"),
        "reference_price": event.get("reference_price"),
        "created_at": event.get("created_at"),
        "event_data": event.get("event_data"),
    }


@app.route("/api/pending_orders")
def api_pending_orders():
    """Resting limit orders plus recent terminal outcomes.

    Query params:
        profile        restrict to one profile (default: all ACTIVE_PROFILES)
        include_terminal  "false" to return only active orders
        limit          max orders per profile (default 100)

    Returns an empty payload when the feature has never been initialized, so the
    front end needs no conditional.

    Pending orders are NOT included in position, exposure, or equity figures
    anywhere — they are reported here as committed intent only (Requirement 8.8).
    """
    if not _pending_orders_available():
        return jsonify({
            "available": False,
            "mode": os.getenv("PENDING_ORDER_MODE", "disabled"),
            "orders": [],
            "counts": {},
        })

    from utils.pending_order_registry import PendingOrderRegistry

    requested_profile = request.args.get("profile")
    include_terminal = request.args.get("include_terminal", "true") != "false"
    limit = request.args.get("limit", default=100, type=int)
    profiles = [requested_profile] if requested_profile else list(ACTIVE_PROFILES)

    registry = PendingOrderRegistry(engine)

    orders = []
    try:
        for profile_id in profiles:
            orders.extend(
                registry.get_orders_for_profile(
                    profile_id, include_terminal=include_terminal, limit=limit
                )
            )
    except Exception as exc:
        log.warning("Could not load pending orders: %s", exc)
        return jsonify({
            "available": False,
            "error": sanitize_error_text(str(exc)),
            "orders": [],
            "counts": {},
        })

    # Current prices only matter for orders still resting.
    active_symbols = {
        o.symbol for o in orders
        if o.state.value in PENDING_ORDER_ACTIVE_STATES
    }
    prices = _pending_order_current_prices(active_symbols)

    serialized = []
    for order in orders:
        events = []
        try:
            events = [
                _serialize_pending_order_event(e)
                for e in registry.get_events(order.order_id)[-10:]
            ]
        except Exception:
            events = []
        serialized.append(
            _serialize_pending_order(
                order, prices.get(order.symbol), events=events
            )
        )

    serialized.sort(key=lambda o: o.get("created_at") or "", reverse=True)

    counts: dict[str, int] = {}
    for order in serialized:
        counts[order["state"]] = counts.get(order["state"], 0) + 1

    return jsonify({
        "available": True,
        "mode": os.getenv("PENDING_ORDER_MODE", "disabled"),
        "orders": serialized,
        "counts": counts,
        "active_count": sum(
            counts.get(state, 0) for state in PENDING_ORDER_ACTIVE_STATES
        ),
    })


@app.route("/api/pending_orders/<order_id>/events")
def api_pending_order_events(order_id: str):
    """Full lifecycle history for one order.

    This is the mechanism satisfying Requirement 11.5. /api/trade-events cannot
    serve it: that endpoint requires a trade_id and returns HTTP 400 without one,
    while pending_order_created / _expired / _canceled / _declined all carry
    trade_id = None because log_trade_event() leaves it nullable precisely so
    pre-fill events can be recorded.
    """
    if not _pending_orders_available():
        return jsonify({"error": "pending orders are not initialized"}), 404

    from utils.pending_order_registry import PendingOrderRegistry

    registry = PendingOrderRegistry(engine)
    try:
        order = registry.get_order(order_id)
        if order is None:
            return jsonify({"error": f"unknown order_id {order_id}"}), 404
        events = [
            _serialize_pending_order_event(e)
            for e in registry.get_events(order_id)
        ]
    except Exception as exc:
        log.warning("Could not load events for order %s: %s", order_id, exc)
        return jsonify({"error": sanitize_error_text(str(exc))}), 500

    return jsonify({
        "order_id": order_id,
        "symbol": order.symbol,
        "side": order.side,
        "state": order.state.value,
        "events": events,
    })


@app.route("/api/pending_orders/summary")
def api_pending_orders_summary():
    """Structured outcome rollup for reviewer and CEO inputs.

    Distinguishes the four terminal outcomes plus declines, so case learning can
    separate "bad idea" from "good idea, order did not fill" and "good idea,
    filled later" (Requirements 12.1, 12.3, 12.4, 12.5).

    Also surfaces the two decline rates that gate design decisions:
      - target_already_exceeded  -> whether that branch should rest orders
      - repaired_before_check    -> how much missed-entry volume the Tier 2
                                    repair band absorbs, which is the evidence
                                    for PENDING_ORDER_DIVERT_REPAIR_BAND
    """
    days = request.args.get("days", default=7, type=int)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    result = {
        "available": _pending_orders_available(),
        "window_days": days,
        "mode": os.getenv("PENDING_ORDER_MODE", "disabled"),
        "outcomes": {},
        "declines": {},
        "fill_rate_by_setup": {},
        "fill_rate_by_profile": {},
        "near_misses": [],
        "fill_bar_age_seconds": {},
    }

    if not result["available"]:
        return jsonify(result)

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT state, setup_type, profile_id, COUNT(*) AS n
                    FROM pending_orders
                    WHERE created_at >= :cutoff
                    GROUP BY state, setup_type, profile_id
                    """
                ),
                {"cutoff": cutoff.isoformat()},
            ).mappings().all()
    except Exception as exc:
        log.warning("Could not summarize pending orders: %s", exc)
        result["available"] = False
        result["error"] = sanitize_error_text(str(exc))
        return jsonify(result)

    by_setup: dict[str, dict[str, int]] = {}
    by_profile: dict[str, dict[str, int]] = {}
    for row in rows:
        state = row["state"]
        result["outcomes"][state] = result["outcomes"].get(state, 0) + row["n"]
        by_setup.setdefault(row["setup_type"], {})[state] = row["n"]
        by_profile.setdefault(row["profile_id"], {})[state] = row["n"]

    def _fill_rate(buckets: dict[str, int]) -> dict:
        resolved = sum(
            buckets.get(state, 0) for state in PENDING_ORDER_TERMINAL_STATES
        )
        filled = buckets.get("filled", 0)
        return {
            "filled": filled,
            "resolved": resolved,
            "fill_rate": round(filled / resolved, 4) if resolved else None,
            "by_state": buckets,
        }

    result["fill_rate_by_setup"] = {k: _fill_rate(v) for k, v in by_setup.items()}
    result["fill_rate_by_profile"] = {
        k: _fill_rate(v) for k, v in by_profile.items()
    }

    # Decline reasons and near-miss telemetry come from the event stream, which
    # is the only place declines are recorded (no order row is ever created).
    try:
        db = get_session(engine)
        try:
            events = (
                db.query(TradeEvent)
                .filter(
                    TradeEvent.event_type.in_([
                        "pending_order_declined", "pending_order_expired"
                    ])
                )
                .filter(TradeEvent.timestamp >= cutoff.replace(tzinfo=None))
                .all()
            )
        finally:
            db.close()

        near_misses = []
        for event in events:
            try:
                payload = json.loads(event.payload_json or "{}")
            except (ValueError, TypeError):
                continue

            if event.event_type == "pending_order_declined":
                reason = payload.get("reason") or "unknown"
                result["declines"][reason] = result["declines"].get(reason, 0) + 1
                continue

            distance = payload.get("closest_approach_distance")
            if distance is not None:
                near_misses.append({
                    "order_id": payload.get("order_id"),
                    "symbol": event.symbol,
                    "side": payload.get("side"),
                    "setup_type": payload.get("setup_type"),
                    "limit_price": payload.get("limit_price"),
                    "closest_approach_price": payload.get(
                        "closest_approach_price"
                    ),
                    "closest_approach_distance": distance,
                    "expired_at": payload.get("expires_at"),
                })

        near_misses.sort(key=lambda n: abs(n["closest_approach_distance"]))
        result["near_misses"] = near_misses[:25]
    except Exception as exc:
        log.warning("Could not summarize pending order events: %s", exc)

    # Opt-in, because it costs one provider call per expired symbol. Off by
    # default so the common dashboard poll stays cheap.
    if request.args.get("check_post_expiry") == "true":
        result["post_expiry"] = _pending_order_post_expiry_report(cutoff)

    return jsonify(result)


def _pending_order_post_expiry_report(cutoff, grace_minutes: int = 30) -> dict:
    """Did expired orders' limits get hit shortly after they expired?

    This is the signal that says whether the active windows are too short
    (Requirement 12.2). It distinguishes "good idea, order did not fill" from
    "good idea, filled later" — which is the distinction case learning needs, and
    the one that can't be inferred from the order rows alone.

    Recomputed from bars rather than tracked, so it needs no extra column and no
    background job.
    """
    report = {
        "grace_minutes": grace_minutes,
        "checked": 0,
        "would_have_filled": 0,
        "orders": [],
    }

    try:
        from decimal import Decimal as _Decimal

        from utils.finnhub_client import FinnhubClient
        from utils.gate_config import (
            PENDING_ORDER_BAR_RESOLUTION,
            PENDING_ORDER_MAX_GAP_THROUGH_PCT,
        )
        from utils.pending_order_fill import (
            bars_from_candles,
            detect_crossing,
            eligible_bars,
        )
        from utils.pending_order_registry import PendingOrderRegistry

        registry = PendingOrderRegistry(engine)
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT order_id FROM pending_orders
                    WHERE state = 'expired' AND created_at >= :cutoff
                    ORDER BY terminal_at DESC LIMIT 50
                    """
                ),
                {"cutoff": cutoff.isoformat()},
            ).fetchall()

        if not rows:
            return report

        orders = [registry.get_order(r[0]) for r in rows]
        orders = [o for o in orders if o is not None]

        bars_by_symbol: dict[str, list] = {}
        client = FinnhubClient()
        for symbol in {o.symbol for o in orders}:
            try:
                bars_by_symbol[symbol] = bars_from_candles(
                    client.get_candles(
                        symbol, resolution=PENDING_ORDER_BAR_RESOLUTION, days=1
                    )
                )
            except Exception:
                continue

        for order in orders:
            bars = bars_by_symbol.get(order.symbol)
            if not bars or order.expires_at is None:
                continue

            report["checked"] += 1
            grace_end = order.expires_at + timedelta(minutes=grace_minutes)
            try:
                windowed = eligible_bars(
                    bars,
                    created_at=order.expires_at,   # strictly AFTER expiry
                    expires_at=grace_end,
                    watermark=None,
                )
                if not windowed:
                    continue
                crossing = detect_crossing(
                    windowed,
                    side=order.side,
                    limit_price=_Decimal(str(order.limit_price)),
                    gap_through_pct=_Decimal(
                        str(PENDING_ORDER_MAX_GAP_THROUGH_PCT)
                    ),
                )
            except Exception:
                continue

            if crossing.crossed:
                report["would_have_filled"] += 1
                report["orders"].append({
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "setup_type": order.setup_type,
                    "limit_price": order.limit_price,
                    "expired_at": order.expires_at.isoformat(),
                    "crossed_at": (
                        crossing.bar.ts.isoformat() if crossing.bar else None
                    ),
                    "minutes_after_expiry": (
                        round(
                            (crossing.bar.ts - order.expires_at).total_seconds()
                            / 60.0,
                            1,
                        )
                        if crossing.bar else None
                    ),
                })

        if report["checked"]:
            report["would_have_filled_rate"] = round(
                report["would_have_filled"] / report["checked"], 4
            )
    except Exception as exc:
        log.warning("Could not build the post-expiry report: %s", exc)
        report["error"] = sanitize_error_text(str(exc))

    return report


if __name__ == "__main__":
    port = int(os.getenv("WEB_PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
