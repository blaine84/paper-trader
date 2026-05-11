"""
EOD Open Exposure Audit

Generates end-of-day open exposure reports that enumerate all remaining
open positions with authorization status, stop status, and required actions.
"""

import json
import logging
from datetime import datetime, timezone

from db.schema import Trade, TradeEvent, Balance, AgentMemory, get_session
from utils.position_lifecycle_governance import (
    evaluate_position_lifecycle,
    latest_valid_overnight_authorization,
    validate_stop_geometry,
    _parse_dt,
)
from utils.trade_events import log_trade_event

log = logging.getLogger(__name__)


def _get_current_price(symbol: str, fallback: float | None) -> float | None:
    """Get current market price for a symbol with yfinance, falling back to provided value."""
    try:
        import yfinance as yf
        return float(yf.Ticker(symbol).fast_info.get("lastPrice", fallback))
    except Exception:
        return fallback


def _get_setup_type_for_trade(db, trade) -> str:
    """Look up the analyst's setup_type for this trade's symbol at entry time."""
    mem = (
        db.query(AgentMemory)
        .filter_by(agent="analyst", symbol=trade.symbol, key="signal")
        .filter(AgentMemory.timestamp <= trade.entry_time)
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    if mem:
        try:
            return json.loads(mem.value).get("setup_type", "")
        except Exception:
            pass
    mem = (
        db.query(AgentMemory)
        .filter_by(agent="analyst", symbol=trade.symbol, key="signal")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    if mem:
        try:
            return json.loads(mem.value).get("setup_type", "")
        except Exception:
            pass
    return ""


def _fetch_open_trades_with_events(db) -> list[tuple[dict, list[dict]]]:
    """Query open trades and their trade_events for EOD audit.

    Returns list of (trade_dict, events_list) tuples.
    """
    open_trades = (
        db.query(Trade)
        .filter_by(status="open")
        .filter(Trade.entry_time.isnot(None))
        .all()
    )

    results = []
    for trade in open_trades:
        trade_dict = {
            "id": trade.id,
            "symbol": trade.symbol,
            "profile": trade.profile,
            "direction": trade.direction,
            "entry_price": trade.entry_price,
            "entry_time": trade.entry_time,
            "stop_price": trade.stop_price,
            "target_price": trade.target_price,
            "setup_type": _get_setup_type_for_trade(db, trade),
            "status": trade.status,
            "quantity": trade.quantity,
        }

        trade_events = (
            db.query(TradeEvent)
            .filter_by(trade_id=trade.id)
            .order_by(TradeEvent.timestamp.asc())
            .all()
        )

        events_list = []
        for event in trade_events:
            payload = {}
            if event.payload_json:
                try:
                    payload = json.loads(event.payload_json)
                except (json.JSONDecodeError, TypeError):
                    payload = {}
            events_list.append({
                "event_type": event.event_type,
                "timestamp": event.timestamp,
                "payload": payload,
            })

        results.append((trade_dict, events_list))

    return results


def _get_current_equity(db) -> float | None:
    """Get current portfolio equity from latest balance row."""
    import os

    latest_balance = (
        db.query(Balance)
        .order_by(Balance.timestamp.desc())
        .first()
    )

    if latest_balance:
        if latest_balance.total_equity is not None:
            return latest_balance.total_equity
        if latest_balance.portfolio_value is not None:
            return latest_balance.portfolio_value
        return latest_balance.cash

    try:
        starting_equity = os.environ.get("STARTING_EQUITY", "")
        if starting_equity:
            return float(starting_equity)
    except (ValueError, TypeError):
        pass

    return None


def _classify_authorization_status(
    events: list[dict],
    trade: dict,
    now_utc: datetime,
) -> str:
    """Classify a position's authorization_status as 'valid', 'expired', or 'missing'.

    Logic:
    - If a valid overnight auth exists with expires_at > now_utc → "valid"
    - If any overnight auth exists but all have expires_at <= now_utc → "expired"
    - If no overnight auth events exist for this trade → "missing"
    """
    # Check for valid (non-expired) auth first
    valid_auth = latest_valid_overnight_authorization(events, trade, now_utc)
    if valid_auth is not None:
        return "valid"

    # Check if any overnight_authorized events exist at all (even expired ones)
    has_any_auth = False
    for event in events:
        if event.get("event_type") == "overnight_authorized":
            payload = event.get("payload")
            if isinstance(payload, dict):
                # Verify it matches this trade
                if (payload.get("trade_id") == trade.get("id") and
                        payload.get("symbol") == trade.get("symbol") and
                        payload.get("profile") == trade.get("profile")):
                    has_any_auth = True
                    break

    if has_any_auth:
        return "expired"

    return "missing"


def _classify_stop_status(
    trade: dict,
    auth: dict | None,
    current_price: float | None,
) -> str:
    """Classify a position's stop_status as 'valid', 'invalid', 'missing', or 'unknown_price'.

    Logic:
    - If current_price is None → "unknown_price"
    - If trade has no stop_price → "missing"
    - Call validate_stop_geometry and return "valid" or "invalid"
    """
    if current_price is None:
        return "unknown_price"

    stop_price = trade.get("stop_price")
    if stop_price is None or stop_price == 0:
        return "missing"

    is_valid, _reason = validate_stop_geometry(trade, auth, current_price)
    if is_valid:
        return "valid"
    return "invalid"


def _count_force_closed_today(db, now_utc: datetime) -> int:
    """Count positions force-closed by governance during today's session.

    Looks for force-close event types in trade_events from today.
    """
    today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

    force_close_event_types = [
        "news_expiry_force_close",
        "overnight_unauthorized_force_close",
        "overnight_invalid_stop_force_close",
        "intraday_force_close",
        "lifecycle_force_close",
        "overnight_unauthorized_close",
    ]

    count = (
        db.query(TradeEvent)
        .filter(TradeEvent.event_type.in_(force_close_event_types))
        .filter(TradeEvent.timestamp >= today_start)
        .count()
    )

    return count


def handle_eod_exceptions(db, audit_result: dict, now_utc: datetime) -> None:
    """Handle EOD audit exceptions: log errors, write AgentMemory, write audit event.

    - If unauthorized positions remain: log ERROR + write AgentMemory eod_exposure_exception
    - Write eod_exposure_audit trade event with daily deduplication
    """
    # 1. If unauthorized positions remain, log ERROR and write AgentMemory
    unauthorized_count = audit_result.get("unauthorized_positions_remaining", 0)
    if unauthorized_count > 0:
        # Gather details of unauthorized positions
        unauthorized_details = [
            p for p in audit_result.get("positions", [])
            if p.get("authorization_status") != "valid"
        ]
        detail_summary = ", ".join(
            f"{p['symbol']}(trade_id={p['trade_id']})" for p in unauthorized_details
        )

        log.error(
            "EOD exposure exception: %d unauthorized position(s) remaining: %s",
            unauthorized_count,
            detail_summary,
        )

        # Write AgentMemory record
        memory_value = json.dumps({
            "unauthorized_count": unauthorized_count,
            "positions": unauthorized_details,
            "force_closed_by_governance": audit_result.get("force_closed_by_governance", 0),
            "timestamp": now_utc.isoformat(),
        })
        memory = AgentMemory(
            agent="position_timer",
            key="eod_exposure_exception",
            timestamp=now_utc,
            value=memory_value,
        )
        db.add(memory)

    # 2. Write eod_exposure_audit trade event with daily deduplication
    dedupe_key = f"eod_exposure_audit:{now_utc.strftime('%Y-%m-%d')}"

    # Check if event with this dedupe_key already exists
    existing = (
        db.query(TradeEvent)
        .filter_by(event_type="eod_exposure_audit", dedupe_key=dedupe_key)
        .first()
    )

    if existing is None:
        event = log_trade_event(
            db,
            "eod_exposure_audit",
            agent="position_timer",
            message=f"EOD audit: {audit_result.get('open_positions_total', 0)} open, "
                    f"{unauthorized_count} unauthorized, "
                    f"{audit_result.get('force_closed_by_governance', 0)} force-closed",
            payload=audit_result,
            timestamp=now_utc,
        )
        event.dedupe_key = dedupe_key

    db.commit()


def eod_open_exposure_audit(
    db,
    engine,
    *,
    now_utc: datetime,
    now_et: datetime,
) -> dict:
    """
    Generate end-of-day open exposure report.

    Queries all open positions, evaluates each through the lifecycle governance
    evaluator, and produces a structured audit report with per-position detail
    including authorization and stop status classifications.

    Args:
        db: Active database session.
        engine: SQLAlchemy engine (for sub-operations if needed).
        now_utc: Current time in UTC.
        now_et: Current time in US/Eastern.

    Returns:
        {
            "open_positions_total": int,
            "by_profile": {profile: {"count": int, "gross_exposure_pct": float}},
            "positions": [per-position detail dicts],
            "unauthorized_positions_remaining": int,
            "force_closed_by_governance": int,
        }
    """
    # Fetch all open trades with their events
    open_trades = _fetch_open_trades_with_events(db)

    # Get current equity for exposure calculations
    current_equity = _get_current_equity(db)

    # Build per-position details
    positions = []
    profile_data: dict[str, dict] = {}  # {profile: {"count": 0, "gross_exposure": 0.0}}
    unauthorized_count = 0

    for trade_dict, events in open_trades:
        symbol = trade_dict["symbol"]
        profile = trade_dict.get("profile", "unknown")

        # Get current price
        current_price = _get_current_price(symbol, trade_dict.get("entry_price"))

        # Call the lifecycle evaluator
        decision = evaluate_position_lifecycle(
            trade_dict,
            events,
            now_utc=now_utc,
            now_et=now_et,
            current_price=current_price,
            current_equity=current_equity,
            overnight_position_counts=None,
            overnight_exposure_pcts=None,
        )

        # Classify authorization_status
        authorization_status = _classify_authorization_status(events, trade_dict, now_utc)

        # Get the valid auth for stop classification
        valid_auth = latest_valid_overnight_authorization(events, trade_dict, now_utc)

        # Classify stop_status
        stop_status = _classify_stop_status(trade_dict, valid_auth, current_price)

        # Compute hours_held
        entry_time = _parse_dt(trade_dict.get("entry_time"))
        if entry_time is not None:
            hours_held = (now_utc - entry_time).total_seconds() / 3600
        else:
            hours_held = 0.0

        # Build per-position detail dict
        position_detail = {
            "trade_id": trade_dict["id"],
            "symbol": symbol,
            "profile": profile,
            "setup_type": trade_dict.get("setup_type", ""),
            "hours_held": round(hours_held, 2),
            "authorization_status": authorization_status,
            "stop_status": stop_status,
            "required_action": decision.get("decision", "allow"),
            "lifecycle_state": decision.get("state", "intraday_ok"),
        }
        positions.append(position_detail)

        # Track unauthorized positions
        if authorization_status != "valid":
            unauthorized_count += 1

        # Accumulate per-profile data
        if profile not in profile_data:
            profile_data[profile] = {"count": 0, "gross_exposure": 0.0}
        profile_data[profile]["count"] += 1

        # Calculate gross exposure for this position
        price_for_exposure = current_price if current_price is not None else trade_dict.get("entry_price")
        quantity = trade_dict.get("quantity", 0) or 0
        if price_for_exposure is not None and quantity:
            profile_data[profile]["gross_exposure"] += abs(quantity * price_for_exposure)

    # Compute per-profile breakdown with exposure percentages
    by_profile = {}
    for profile, data in profile_data.items():
        gross_exposure_pct = 0.0
        if current_equity and current_equity > 0:
            gross_exposure_pct = data["gross_exposure"] / current_equity
        by_profile[profile] = {
            "count": data["count"],
            "gross_exposure_pct": round(gross_exposure_pct, 6),
        }

    # Count force-closed positions today
    force_closed_count = _count_force_closed_today(db, now_utc)

    result = {
        "open_positions_total": len(open_trades),
        "by_profile": by_profile,
        "positions": positions,
        "unauthorized_positions_remaining": unauthorized_count,
        "force_closed_by_governance": force_closed_count,
    }

    # Handle exceptions and write audit event
    handle_eod_exceptions(db, result, now_utc)

    return result
