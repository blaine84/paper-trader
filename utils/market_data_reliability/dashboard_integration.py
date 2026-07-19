"""Dashboard API integration for the Market Data Reliability Layer.

Enriches API response dicts with freshness/trust metadata so the dashboard
can display explicit degradation labels and machine-readable reliability state.

Production safety:
    - Guarded by MARKET_DATA_RELIABILITY_MODE feature flag
    - Fail-open: if enrichment fails, return original data without modification
    - Try/except around all enrichment logic
    - Minimal, additive changes only (never removes existing fields)

Requirements: 9.1, 9.2, 9.3, 9.4
"""
from __future__ import annotations

import logging
from typing import Optional

from utils.gate_config import MARKET_DATA_RELIABILITY_MODE
from utils.market_data_reliability.snapshot import Snapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Freshness label mapping
# ---------------------------------------------------------------------------

_FRESHNESS_LABEL_MAP: dict[tuple[str, str], Optional[str]] = {
    ("fresh", "trusted"): None,
    ("aging", "trusted"): "Data is slightly delayed",
    ("aging", "degraded"): "Data is slightly delayed",
    ("stale", "degraded"): "Data may be stale - do not use for trading decisions",
    ("stale", "untrusted"): "Data is unreliable - do not use for trading decisions",
    ("unavailable", "untrusted"): "Data unavailable from all providers",
    ("unavailable", "degraded"): "Data unavailable from all providers",
    ("market_closed", "trusted"): "Market closed - showing last session data",
    ("market_closed", "degraded"): "Market closed - showing last session data",
    ("market_closed", "untrusted"): "Market closed - showing last session data",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_freshness_label(freshness_state: str, trust_state: str) -> Optional[str]:
    """Return a human-readable label for the given freshness/trust state combination.

    Returns None when the data is fresh and trusted (no label needed).
    Returns a descriptive string for degraded/stale/unavailable states.

    Args:
        freshness_state: One of "fresh", "aging", "stale", "unavailable", "market_closed".
        trust_state: One of "trusted", "degraded", "untrusted".

    Returns:
        Human-readable label string, or None if no label is needed.
    """
    return _FRESHNESS_LABEL_MAP.get((freshness_state, trust_state))


def enrich_market_data_response(
    response_data: dict,
    snapshot: Optional[Snapshot],
) -> dict:
    """Enrich an API response dict with reliability metadata from a Snapshot.

    Adds freshness_state, trust_state, freshness_label, and is_actionable fields
    to the response dict. The enrichment is purely additive — existing fields are
    never removed or modified.

    Production safety:
        - If MARKET_DATA_RELIABILITY_MODE is "disabled", returns data unchanged.
        - If snapshot is None or enrichment raises, returns data unchanged (fail-open).
        - All enrichment logic is wrapped in try/except.

    Args:
        response_data: The API response dict to enrich (a single row/item).
        snapshot: The Snapshot instance for this data row, or None.

    Returns:
        The enriched response dict (same reference, mutated in place) with
        reliability metadata added. On failure, returns the original dict unchanged.
    """
    # Guard: feature flag must be active
    if MARKET_DATA_RELIABILITY_MODE == "disabled":
        return response_data

    # Guard: snapshot must be provided
    if snapshot is None:
        return response_data

    try:
        freshness_state = snapshot.freshness_state
        trust_state = snapshot.trust_state
        freshness_label = get_freshness_label(freshness_state, trust_state)

        # Determine actionability: fresh+trusted or aging+trusted data is actionable
        # (aging data is still trusted, just slightly delayed)
        is_actionable = (
            trust_state == "trusted"
            and freshness_state in ("fresh", "aging")
        )

        # Enrich the response — additive only
        response_data["freshness_state"] = freshness_state
        response_data["trust_state"] = trust_state
        response_data["freshness_label"] = freshness_label
        response_data["is_actionable"] = is_actionable

    except Exception:
        # Fail-open: log error and return original data unchanged
        logger.error(
            "Dashboard enrichment failed for symbol=%s; returning unenriched data",
            getattr(snapshot, "symbol", "unknown"),
            exc_info=True,
        )

    return response_data
