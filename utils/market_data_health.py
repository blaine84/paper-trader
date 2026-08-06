"""Market-data outage health signaling."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any

from db.schema import AgentMemory, get_session
from utils.market_data_remediation import maybe_restart_orchestrator_after_market_data_outage
from utils.trade_events import log_trade_event

log = logging.getLogger(__name__)

_last_alert_times: dict[str, float] = {}


def reset_market_data_health_throttle() -> None:
    """Clear in-process alert throttles for focused tests."""
    _last_alert_times.clear()


def record_market_data_health_alert(
    engine: Any,
    *,
    source: str,
    consumer: str,
    symbols: list[str],
    reason: str,
    message: str | None = None,
    details: dict | None = None,
    throttle_seconds: int = 300,
) -> dict | None:
    """Persist a throttled health alert for all-symbol market-data outages.

    Returns the payload when a new alert is persisted, or None when throttled.
    """
    unique_symbols = list(dict.fromkeys(sym for sym in symbols if sym))
    if not unique_symbols:
        return None

    throttle_key = f"{source}:{consumer}:{reason}"
    now = time.time()
    last = _last_alert_times.get(throttle_key, 0.0)
    if now - last < throttle_seconds:
        return None
    _last_alert_times[throttle_key] = now

    timestamp = datetime.utcnow().isoformat() + "Z"
    payload = {
        "source": source,
        "consumer": consumer,
        "symbols": unique_symbols,
        "symbol_count": len(unique_symbols),
        "reason": reason,
        "severity": "critical",
        "timestamp": timestamp,
    }
    if details:
        payload["details"] = details

    alert_message = (
        message
        or f"{consumer}: zero quotes returned for {len(unique_symbols)} symbols"
    )
    live_alert = {
        "type": "market_data_outage",
        "symbol": "SYSTEM",
        "price": None,
        "detail": alert_message,
        "severity": "critical",
        "timestamp": timestamp,
    }

    log.error(
        "MARKET_DATA_HEALTH_ALERT source=%s consumer=%s reason=%s symbols=%d",
        source,
        consumer,
        reason,
        len(unique_symbols),
    )

    db = get_session(engine)
    try:
        db.add(
            AgentMemory(
                agent="market_data",
                symbol=None,
                key="quote_outage",
                value=json.dumps(payload),
            )
        )
        db.add(
            AgentMemory(
                agent="market_data",
                symbol=None,
                key="health_alert",
                value=json.dumps(payload),
            )
        )
        db.add(
            AgentMemory(
                agent="price_monitor",
                symbol=None,
                key="live_alerts",
                value=json.dumps([live_alert]),
            )
        )
        log_trade_event(
            db,
            "market_data_unavailable",
            agent=source,
            message=alert_message,
            payload=payload,
        )
        db.commit()
    finally:
        db.close()

    maybe_restart_orchestrator_after_market_data_outage(payload)
    return payload
