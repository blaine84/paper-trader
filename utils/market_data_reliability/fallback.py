"""FallbackRouter for the Market Data Reliability Layer.

Determines provider ordering and fallback eligibility per (data_type, consumer).
Uses the fallback matrix from ReliabilityConfig and filters out providers that
are currently in backoff.

Consumer category resolution follows the same mapping as trust.py:
    - PM, Risk_Geometry_Gate → "execution"
    - Dashboard_API, Analyst, Reviewer, CEO_Output → "display"
    - Price_Monitor, Alert_Dispatcher → "monitoring"
    - Unknown → "execution" (fail-closed)

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
"""

from __future__ import annotations

import logging

from utils.market_data_reliability.backoff import BackoffTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Consumer category mappings (same as trust.py)
# ---------------------------------------------------------------------------

EXECUTION_CONSUMERS: frozenset[str] = frozenset({
    "PM",
    "Risk_Geometry_Gate",
})

DISPLAY_CONSUMERS: frozenset[str] = frozenset({
    "Dashboard_API",
    "Analyst",
    "Reviewer",
    "CEO_Output",
})

MONITORING_CONSUMERS: frozenset[str] = frozenset({
    "Price_Monitor",
    "Alert_Dispatcher",
})


def _consumer_category(consumer: str) -> str:
    """Resolve consumer name to its category.

    Returns one of: "execution", "display", "monitoring".
    Unknown consumers default to "execution" (fail-closed).
    """
    if consumer in EXECUTION_CONSUMERS:
        return "execution"
    if consumer in DISPLAY_CONSUMERS:
        return "display"
    if consumer in MONITORING_CONSUMERS:
        return "monitoring"
    # Unknown consumers default to execution for safety (fail-closed)
    logger.warning(
        "Unknown consumer '%s' in fallback routing; defaulting to execution category.",
        consumer,
    )
    return "execution"


# ---------------------------------------------------------------------------
# FallbackRouter
# ---------------------------------------------------------------------------


class FallbackRouter:
    """Determines provider ordering and fallback eligibility per (data_type, consumer).

    Constructor accepts:
        fallback_matrix: dict mapping (data_type, consumer_category) to ordered
            provider lists. Loaded from ReliabilityConfig.
        backoff_tracker: BackoffTracker instance used to filter out providers
            that are currently in backoff for a given data_type.

    Fallback eligibility rules:
        - PM/Risk (execution): quote/candle fallback allowed; must still pass
          trust checks. If all providers fail → fail closed.
        - Dashboard (display): all fallbacks allowed; degraded data served
          with explicit labeling.
        - Price Monitor (monitoring): quote fallback allowed; stale fallback
          suppresses triggers rather than firing them.
    """

    def __init__(
        self,
        fallback_matrix: dict[tuple[str, str], list[str]],
        backoff_tracker: BackoffTracker,
    ) -> None:
        self._fallback_matrix = dict(fallback_matrix)
        self._backoff_tracker = backoff_tracker

    def resolve_providers(self, data_type: str, consumer: str) -> list[str]:
        """Resolve the ordered provider list for a (data_type, consumer) pair.

        Steps:
            1. Resolve consumer to consumer_category (execution, display, monitoring)
            2. Look up (data_type, consumer_category) in fallback_matrix
            3. Filter out providers that are currently in backoff for this data_type
            4. Return the ordered provider list (primary first, then fallbacks)

        If no entry exists in the matrix for the resolved key, falls back to
        checking (data_type, "all") as a catch-all. Returns an empty list if
        no configuration is found at all (fail-closed — no providers available).

        Args:
            data_type: The category of market data ("quote", "candle", "atr",
                "volume", "previous_close").
            consumer: Consumer name (e.g., "PM", "Dashboard_API").

        Returns:
            Ordered list of provider names with backoff-affected providers removed.
            May be empty if all providers are in backoff or no configuration exists.
        """
        category = _consumer_category(consumer)

        # Look up in matrix, try specific category first, then "all" catch-all
        providers = self._fallback_matrix.get((data_type, category))
        if providers is None:
            providers = self._fallback_matrix.get((data_type, "all"))

        if providers is None:
            logger.debug(
                "No fallback matrix entry for data_type=%s consumer_category=%s; "
                "returning empty provider list.",
                data_type, category,
            )
            return []

        # Filter out providers currently in backoff for this data_type
        available = [
            p for p in providers
            if not self._backoff_tracker.is_in_backoff(p, data_type)
        ]

        if len(available) < len(providers):
            skipped = [p for p in providers if p not in available]
            logger.info(
                "Fallback routing: skipped providers in backoff %s for "
                "data_type=%s consumer=%s",
                skipped, data_type, consumer,
            )

        return available

    def is_fallback_eligible(self, data_type: str, consumer: str) -> bool:
        """Check if fallback is available for a (data_type, consumer) pair.

        Fallback is eligible when the fallback matrix contains more than one
        provider for the resolved (data_type, consumer_category) key.

        This does NOT account for backoff state — it reflects static
        configuration eligibility. Use resolve_providers() to get the actual
        available provider list after backoff filtering.

        Args:
            data_type: The category of market data.
            consumer: Consumer name.

        Returns:
            True if the matrix entry has > 1 provider (fallback possible).
            False if only 1 or 0 providers configured.
        """
        category = _consumer_category(consumer)

        # Look up in matrix, try specific category first, then "all" catch-all
        providers = self._fallback_matrix.get((data_type, category))
        if providers is None:
            providers = self._fallback_matrix.get((data_type, "all"))

        if providers is None:
            return False

        return len(providers) > 1
