"""Helpers for recording normalized trade lifecycle events."""

import json
from datetime import datetime
from typing import Any

from db.schema import TradeEvent


def _json_default(value: Any):
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


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

    event = TradeEvent(
        trade_id=trade_id,
        timestamp=timestamp or datetime.utcnow(),
        event_type=event_type,
        agent=agent,
        symbol=symbol,
        profile=profile,
        price=price,
        message=message,
        payload_json=payload_json,
    )
    db.add(event)
    return event
