"""Helpers for recording normalized trade lifecycle events."""

import json
import logging
from datetime import datetime
from typing import Any

from db.schema import TradeEvent

log = logging.getLogger(__name__)


def _json_default(value: Any):
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _coerce_optional_float(value: Any, field_name: str, event_type: str, symbol: str | None):
    """Return a float for valid numeric values, otherwise None.

    LLM repair can occasionally leave placeholders like ``N.NN`` or strings
    with currency formatting. TradeEvent.price is nullable, so dropping a bad
    event price is safer than letting autoflush abort the trading cycle.
    """
    if value is None:
        return None
    try:
        if isinstance(value, str):
            cleaned = value.strip().replace("$", "").replace(",", "")
            if not cleaned or cleaned.upper() in {"N/A", "NA", "NONE", "NULL", "N.NN"}:
                raise ValueError("placeholder/non-numeric price")
            value = cleaned
        return float(value)
    except (TypeError, ValueError):
        log.warning(
            "Dropping invalid %s for trade event %s%s: %r",
            field_name,
            event_type,
            f" ({symbol})" if symbol else "",
            value,
        )
        return None


def log_trade_event(
    db,
    event_type: str,
    *,
    trade_id: int | None = None,
    agent: str | None = None,
    symbol: str | None = None,
    profile: str | None = None,
    price: float | None = None,
    message: str | None = None,
    payload: dict | list | str | None = None,
    timestamp: datetime | None = None,
    candidate_lineage_id: str | None = None,
):
    """
    Add a normalized trade event to the current SQLAlchemy session.

    The caller owns commit/rollback. `trade_id` is nullable so pre-fill events
    like signal_seen/entry_requested can be recorded before a Trade row exists.
    """
    if payload is None:
        payload_json = None
    elif isinstance(payload, str):
        payload_json = payload
    else:
        payload_json = json.dumps(payload, default=_json_default)

    safe_price = _coerce_optional_float(price, "price", event_type, symbol)

    event = TradeEvent(
        trade_id=trade_id,
        timestamp=timestamp or datetime.utcnow(),
        event_type=event_type,
        agent=agent,
        symbol=symbol,
        profile=profile,
        price=safe_price,
        message=message,
        payload_json=payload_json,
        candidate_lineage_id=candidate_lineage_id,
    )
    db.add(event)
    return event
