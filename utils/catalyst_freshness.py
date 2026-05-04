"""Catalyst freshness computation — shared by web API, analyst, and display."""

from datetime import datetime, timedelta, timezone as dt_tz
from typing import Optional
import json
import logging
from pytz import timezone, utc as UTC

log = logging.getLogger(__name__)

ET = timezone("America/New_York")

# ─── Configurable thresholds ─────────────────────────────────────────────────
FRESHNESS_THRESHOLDS = {
    "fresh_minutes": 60,
    "aging_minutes": 180,
}

# Confidence mapping: (researcher_confidence_level, freshness_state) → score
# Configurable — change this dict to adjust confidence decay.
CONFIDENCE_MAP = {
    ("high", "fresh"): 0.9,
    ("high", "aging"): 0.6,
    ("high", "stale"): 0.3,
    ("medium", "fresh"): 0.7,
    ("medium", "aging"): 0.4,
    ("medium", "stale"): 0.2,
    ("low", "fresh"): 0.4,
    ("low", "aging"): 0.2,
    ("low", "stale"): 0.0,
}

PRICE_SPIKE_THRESHOLD_PCT = 2.0
PRICE_SPIKE_WINDOW_MINUTES = 15
POSITION_POLL_INTERVAL_MINUTES = 30
MAX_ALERTS_PER_SYMBOL = 3  # cap per fetch; configurable


def get_market_day_start(now_et: datetime) -> datetime:
    """
    Return the start of the current market day in ET.
    Market day boundary is 4:00 AM ET. Activity between midnight and 4 AM
    belongs to the previous calendar day's market day.
    """
    cutoff = now_et.replace(hour=4, minute=0, second=0, microsecond=0)
    if now_et < cutoff:
        # Before 4 AM ET — roll back to previous calendar day's 4 AM
        cutoff -= timedelta(days=1)
    return cutoff


def compute_freshness_state(
    researcher_ts: Optional[datetime],
    breaking_news_ts: Optional[datetime],
    now: datetime,
) -> str:
    """
    Return 'fresh', 'aging', or 'stale' based on the most recent timestamp.

    Uses the later of researcher_ts and breaking_news_ts. Thresholds:
      - < 60 minutes  → "fresh"
      - >= 60 and < 180 minutes → "aging"
      - >= 180 minutes → "stale"

    If both timestamps are None, returns "stale".
    """
    # Pick the most recent non-None timestamp
    candidates = [ts for ts in (researcher_ts, breaking_news_ts) if ts is not None]
    if not candidates:
        return "stale"

    latest = max(candidates)
    age = now - latest
    age_minutes = age.total_seconds() / 60.0

    fresh_limit = FRESHNESS_THRESHOLDS["fresh_minutes"]   # 60
    aging_limit = FRESHNESS_THRESHOLDS["aging_minutes"]    # 180

    if age_minutes < fresh_limit:
        return "fresh"
    elif age_minutes < aging_limit:
        return "aging"
    else:
        return "stale"


def get_breaking_news_for_symbols(
    db_session,
    symbols: list,
    market_day_start: datetime,
) -> dict:
    """
    Query AgentMemory for the most recent breaking_news record from news_monitor
    that falls on or after market_day_start. Parse the alerts and bucket them
    by symbol. Returns {symbol: [alert_dict, ...]} for symbols in the list.
    Symbols with no alerts get an empty list.
    """
    from db.schema import AgentMemory

    result = {sym: [] for sym in symbols}
    symbols_set = set(symbols)

    # Query all breaking_news rows on or after market_day_start, newest first
    rows = (
        db_session.query(AgentMemory)
        .filter_by(agent="news_monitor", key="breaking_news")
        .filter(AgentMemory.timestamp >= market_day_start)
        .order_by(AgentMemory.timestamp.desc())
        .all()
    )

    # Collect alerts from all rows, deduplicating by headline per symbol
    seen = {sym: set() for sym in symbols}
    for row in rows:
        try:
            data = json.loads(row.value)
        except (json.JSONDecodeError, TypeError):
            continue
        for alert in data.get("alerts", []):
            sym = alert.get("symbol")
            if sym not in symbols_set:
                continue
            headline = alert.get("headline", "")
            if headline in seen[sym]:
                continue
            seen[sym].add(headline)
            result[sym].append(alert)

    return result


def get_researcher_timestamps(
    db_session,
    symbols: list,
) -> dict:
    """
    Return {symbol: (timestamp, confidence_level)} for the most recent
    researcher sentiment per symbol.  Ignores entries older than 36 hours
    so that truly stale data is treated as absent rather than misleadingly
    displayed.
    """
    from db.schema import AgentMemory

    cutoff = datetime.now(dt_tz.utc) - timedelta(hours=36)
    result = {}
    for sym in symbols:
        row = (
            db_session.query(AgentMemory)
            .filter_by(agent="researcher", key="sentiment", symbol=sym)
            .filter(AgentMemory.timestamp >= cutoff)
            .order_by(AgentMemory.timestamp.desc())
            .first()
        )
        if row is None:
            continue
        try:
            data = json.loads(row.value)
        except (json.JSONDecodeError, TypeError):
            continue
        confidence_level = data.get("confidence", "low")
        result[sym] = (row.timestamp, confidence_level)

    return result


def compute_confidence(
    researcher_confidence_level: str,
    freshness_state: str,
) -> float:
    """
    Map (researcher_confidence_level, freshness_state) → numeric score.

    Uses CONFIDENCE_MAP; returns 0.0 for unknown combinations (e.g., when
    the researcher never ran for a symbol). Logs a warning on fallback.
    """
    key = (researcher_confidence_level, freshness_state)
    score = CONFIDENCE_MAP.get(key)
    if score is not None:
        return score
    log.warning(
        "Unknown confidence combination (%s, %s) — defaulting to 0.0",
        researcher_confidence_level,
        freshness_state,
    )
    return 0.0


def compute_catalyst_freshness(
    db_session,
    symbols: list,
    now: Optional[datetime] = None,
    breaking_news_by_symbol: Optional[dict] = None,
) -> dict:
    """
    Main entry point. Returns {symbol: {
        "last_researcher_update": ISO str or None,
        "last_breaking_news_update": ISO str or None,
        "freshness_state": "fresh"|"aging"|"stale",
        "source_type": "premarket_synthesis"|"intraday_alert",
        "confidence": float 0.0–1.0,
    }}

    If breaking_news_by_symbol is provided, uses it instead of querying
    the DB again (avoids redundant queries when the caller already has
    the data).
    """
    if now is None:
        now = datetime.now(ET)

    # Ensure now_et is in ET for market day boundary
    now_et = now.astimezone(ET) if now.tzinfo else ET.localize(now)
    market_day_start = get_market_day_start(now_et)

    # Fetch breaking news if not pre-fetched
    if breaking_news_by_symbol is None:
        breaking_news_by_symbol = get_breaking_news_for_symbols(
            db_session, symbols, market_day_start
        )

    # Fetch researcher timestamps
    researcher_data = get_researcher_timestamps(db_session, symbols)

    result = {}
    for sym in symbols:
        # Researcher timestamp and confidence level
        researcher_ts = None
        confidence_level = "low"
        if sym in researcher_data:
            researcher_ts, confidence_level = researcher_data[sym]
            # Ensure timezone-aware
            if researcher_ts is not None and researcher_ts.tzinfo is None:
                researcher_ts = UTC.localize(researcher_ts)

        # Breaking news: find the most recent alert timestamp
        alerts = breaking_news_by_symbol.get(sym, [])
        breaking_news_ts = None
        # Alerts don't carry their own timestamp — the presence of alerts
        # on the current market day means news exists for today.
        # Use market_day_start as a proxy if alerts exist but have no timestamp field.
        # The actual "last breaking news update" is the timestamp of the AgentMemory row,
        # but since we bucket by symbol from potentially multiple rows, we use now_et
        # as the breaking news timestamp when alerts exist (they are from the current market day).
        if alerts:
            # Breaking news exists for the current market day
            breaking_news_ts = now_et

        # Compute freshness state using UTC-normalized timestamps
        now_utc = now.astimezone(dt_tz.utc) if now.tzinfo else datetime.now(dt_tz.utc)
        researcher_ts_for_state = researcher_ts
        breaking_ts_for_state = breaking_news_ts
        if breaking_ts_for_state is not None:
            breaking_ts_for_state = breaking_ts_for_state.astimezone(dt_tz.utc) if breaking_ts_for_state.tzinfo else breaking_ts_for_state

        freshness_state = compute_freshness_state(
            researcher_ts_for_state, breaking_ts_for_state, now_utc
        )

        # Source type: intraday_alert if breaking news exists for current market day
        source_type = "intraday_alert" if alerts else "premarket_synthesis"

        # Confidence
        confidence = compute_confidence(confidence_level, freshness_state)

        # Format timestamps as ISO strings
        last_researcher_update = None
        if researcher_ts is not None:
            last_researcher_update = researcher_ts.isoformat()

        last_breaking_news_update = None
        if breaking_news_ts is not None:
            last_breaking_news_update = breaking_news_ts.isoformat()

        result[sym] = {
            "last_researcher_update": last_researcher_update,
            "last_breaking_news_update": last_breaking_news_update,
            "freshness_state": freshness_state,
            "source_type": source_type,
            "confidence": confidence,
        }

    return result


def _format_time_et(dt_obj: datetime) -> str:
    """Format a datetime as HH:MM AM/PM in ET, without leading zero on the hour.

    Includes the date (e.g. "Apr 7") when the timestamp is NOT from today,
    so stale data is immediately obvious.
    """
    et_dt = dt_obj.astimezone(ET)
    now_et = datetime.now(ET)
    hour = int(et_dt.strftime("%I"))  # %I gives 01-12; int() strips leading zero
    minute = et_dt.strftime("%M")
    ampm = et_dt.strftime("%p")
    time_str = f"{hour}:{minute} {ampm}"
    if et_dt.date() != now_et.date():
        date_str = f"{et_dt.strftime('%b')} {et_dt.day}"
        return f"{date_str} {time_str}"
    return time_str


def build_freshness_label(
    symbol: str,
    freshness: dict,
) -> str:
    """
    Build a human-readable label like:
    "TSLA catalyst view is based on 8:30 AM premarket synthesis;
     last intraday news update was 12:05 PM; catalyst freshness is fresh."

    Times are in ET, HH:MM AM/PM format.
    When no breaking news exists, uses "no intraday news updates".
    """
    if not freshness:
        return f"{symbol} catalyst view is unavailable."

    # Format researcher timestamp in ET
    researcher_iso = freshness.get("last_researcher_update")
    source_type = freshness.get("source_type", "premarket_synthesis")
    freshness_state = freshness.get("freshness_state", "stale")

    if researcher_iso:
        researcher_dt = datetime.fromisoformat(researcher_iso)
        researcher_time_str = _format_time_et(researcher_dt)
    else:
        researcher_time_str = "unknown time"

    # Source description
    source_description = (
        "premarket synthesis" if source_type == "premarket_synthesis"
        else "intraday alert"
    )

    # Breaking news timestamp
    breaking_iso = freshness.get("last_breaking_news_update")
    if breaking_iso:
        breaking_dt = datetime.fromisoformat(breaking_iso)
        news_time_str = _format_time_et(breaking_dt)
    else:
        news_time_str = "no intraday news updates"

    return (
        f"{symbol} catalyst view is based on {researcher_time_str} {source_description}; "
        f"last intraday news update was {news_time_str}; "
        f"catalyst freshness is {freshness_state}."
    )
