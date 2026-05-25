"""Structured logging helper for the Sector Scout pipeline.

Provides a standardized function for emitting structured log events with
JSON payloads. All scout pipeline events (SCOUT_RUN, SECTOR_SCREEN,
CHIEF_SCOUT, EXPANDED_WATCHLIST, BUDGET_CEILING_HIT) use this helper
to ensure consistent format.

Event format:
    {event_type}: {json_payload}

See: design.md §7 (Observability), requirements.md §10.4
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def emit_scout_event(event_type: str, payload: dict[str, Any]) -> None:
    """Emit a structured scout pipeline log event.

    Logs at INFO level with the format:
        {event_type}: {json_payload}

    The payload is serialized as compact JSON for machine parseability
    while remaining human-readable in log output.

    Args:
        event_type: One of SCOUT_RUN, SECTOR_SCREEN, CHIEF_SCOUT,
                    EXPANDED_WATCHLIST, BUDGET_CEILING_HIT.
        payload: Dict of event-specific fields to include in the log.
    """
    # Add timestamp to all events
    enriched_payload = {
        "event": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }

    logger.info(
        "%s: %s",
        event_type,
        json.dumps(enriched_payload, default=str),
    )
