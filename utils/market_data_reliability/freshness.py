"""Freshness classification for market data snapshots.

Classifies data freshness based on age_seconds against per-(data_type, consumer)
thresholds loaded from ReliabilityConfig. Returns exactly one state from the
valid enum: fresh, aging, stale, unavailable, market_closed.

Requirements: 2.1, 2.2, 2.3, 2.4
"""
from __future__ import annotations

import logging
from typing import Optional

from utils.market_data_reliability.snapshot import FreshnessThreshold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Consumer → category mapping
# ---------------------------------------------------------------------------

_CONSUMER_CATEGORY_MAP: dict[str, str] = {
    # Execution consumers (fail-closed, strict thresholds)
    "PM": "execution",
    "Risk_Geometry_Gate": "execution",
    # Display consumers (lenient thresholds, serve degraded with label)
    "Dashboard_API": "display",
    "Analyst": "display",
    "Reviewer": "display",
    "CEO_Output": "display",
    # Monitoring consumers (use execution thresholds — strict)
    "Price_Monitor": "execution",
    "Alert_Dispatcher": "execution",
}


def resolve_consumer_category(consumer: str) -> str:
    """Resolve a consumer name to its threshold category.

    Monitoring consumers (Price_Monitor, Alert_Dispatcher) map to "execution"
    thresholds since they need strict freshness for trigger evaluation.

    Unknown consumers default to "execution" (fail-closed safe default).
    """
    category = _CONSUMER_CATEGORY_MAP.get(consumer)
    if category is None:
        logger.warning(
            "Unknown consumer '%s' in freshness classification; "
            "defaulting to 'execution' category (fail-closed).",
            consumer,
        )
        return "execution"
    return category


class FreshnessClassifier:
    """Classifies data freshness from age_seconds and per-(data_type, consumer) thresholds.

    Classification logic:
        1. age_seconds < 0 → "unavailable" (defensive, shouldn't happen)
        2. market_session == "closed" → "market_closed"
        3. Look up threshold for (data_type, consumer_category)
        4. age_seconds < fresh_threshold → "fresh"
        5. age_seconds < aging_threshold → "aging"
        6. age_seconds >= aging_threshold → "stale"

    If no threshold is found for the (data_type, consumer_category) pair,
    returns "stale" (fail-closed safe default).
    """

    def __init__(self, freshness_thresholds: dict[tuple[str, str], FreshnessThreshold]) -> None:
        """Initialize with freshness thresholds from ReliabilityConfig.

        Args:
            freshness_thresholds: Mapping of (data_type, consumer_category) to
                FreshnessThreshold instances defining fresh and aging boundaries.
        """
        self._thresholds = freshness_thresholds

    def classify(
        self,
        age_seconds: float,
        data_type: str,
        consumer: str,
        market_session: str,
    ) -> str:
        """Classify freshness state for given parameters.

        Args:
            age_seconds: Age of the data in seconds since source_timestamp.
            data_type: Type of market data ("quote", "candle", "atr", "volume", "previous_close").
            consumer: Name of the requesting consumer (e.g., "PM", "Dashboard_API").
            market_session: Current market session state ("open", "pre_market", "after_hours", "closed").

        Returns:
            Exactly one of: "fresh", "aging", "stale", "unavailable", "market_closed".
        """
        # 1. Defensive: negative age indicates data is unavailable
        if age_seconds < 0:
            logger.debug(
                "Negative age_seconds (%.2f) for %s/%s; classifying as unavailable.",
                age_seconds, data_type, consumer,
            )
            return "unavailable"

        # 2. Market closed: data belongs to most recent completed session
        if market_session == "closed":
            return "market_closed"

        # 3. Resolve threshold for (data_type, consumer_category)
        threshold = self._resolve_threshold(data_type, consumer)
        if threshold is None:
            logger.warning(
                "No freshness threshold found for (%s, %s); "
                "classifying as stale (fail-closed).",
                data_type, consumer,
            )
            return "stale"

        # 4-6. Apply threshold boundaries
        if age_seconds < threshold.fresh_threshold:
            return "fresh"
        if age_seconds < threshold.aging_threshold:
            return "aging"
        return "stale"

    def _resolve_threshold(
        self, data_type: str, consumer: str
    ) -> Optional[FreshnessThreshold]:
        """Look up the FreshnessThreshold for a (data_type, consumer) pair.

        Resolution order:
            1. (data_type, consumer_category) — specific match
            2. (data_type, "all") — wildcard consumer (e.g., previous_close)

        Returns None if no threshold is found (caller should fail-closed).
        """
        consumer_category = resolve_consumer_category(consumer)

        # Try specific (data_type, category) first
        threshold = self._thresholds.get((data_type, consumer_category))
        if threshold is not None:
            return threshold

        # Try wildcard "all" category (used by previous_close)
        threshold = self._thresholds.get((data_type, "all"))
        if threshold is not None:
            return threshold

        return None
