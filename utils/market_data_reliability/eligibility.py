"""Eligibility resolution for the Market Data Reliability Layer.

Determines whether a snapshot is eligible for a specific consumer given
that consumer's freshness and trust requirements. Execution consumers
(PM, Risk_Geometry_Gate) require trusted data; display consumers
(Dashboard_API) may accept degraded data with explicit labeling;
monitoring consumers (Price_Monitor) require trusted data to suppress
false triggers.

Requirements: 3.5, 3.6, 6.3, 6.4, 7.1
"""
from __future__ import annotations

import logging

from utils.market_data_reliability.snapshot import EligibilityResult, Snapshot
from utils.market_data_reliability.trust import (
    DISPLAY_CONSUMERS,
    EXECUTION_CONSUMERS,
    MONITORING_CONSUMERS,
)

logger = logging.getLogger(__name__)


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
    logger.warning(
        "Unknown consumer '%s' in eligibility check; defaulting to execution (fail-closed).",
        consumer,
    )
    return "execution"


class EligibilityResolver:
    """Determines whether a snapshot is eligible for a specific consumer.

    Consumer categories and their eligibility rules:
      - Execution (PM, Risk_Geometry_Gate): eligible ONLY if trust_state == "trusted".
      - Display (Dashboard_API, Analyst, Reviewer, CEO_Output): eligible if trusted,
        OR if degraded AND allow_stale_for_display=True.
      - Monitoring (Price_Monitor, Alert_Dispatcher): eligible ONLY if trusted.
        Suppress triggers on degraded/untrusted data.
      - Unknown consumers: treated as execution (fail-closed).
    """

    def is_eligible(
        self,
        snapshot: Snapshot,
        consumer: str,
        data_type: str,
        allow_stale_for_display: bool = False,
    ) -> EligibilityResult:
        """Check eligibility of a snapshot for a given consumer.

        Args:
            snapshot: The market data snapshot to evaluate.
            consumer: Consumer name (e.g., "PM", "Dashboard_API").
            data_type: The data type requested (e.g., "quote", "candle").
            allow_stale_for_display: When True, display consumers may receive
                degraded data with explicit labeling.

        Returns:
            EligibilityResult with eligible bool, reason_code, and snapshot.
        """
        trust_state = snapshot.trust_state
        category = _consumer_category(consumer)

        # Untrusted data is never eligible for any consumer
        if trust_state == "untrusted":
            logger.debug(
                "Eligibility: ineligible for consumer=%s (category=%s), "
                "trust_state=untrusted, symbol=%s, data_type=%s",
                consumer, category, snapshot.symbol, data_type,
            )
            return EligibilityResult(
                eligible=False,
                reason_code="data_untrusted",
                snapshot=snapshot,
            )

        # Trusted data is always eligible for all consumers
        if trust_state == "trusted":
            return EligibilityResult(
                eligible=True,
                reason_code=None,
                snapshot=snapshot,
            )

        # From here, trust_state == "degraded"
        if category == "execution":
            logger.debug(
                "Eligibility: ineligible for execution consumer=%s, "
                "trust_state=degraded, symbol=%s, data_type=%s",
                consumer, snapshot.symbol, data_type,
            )
            return EligibilityResult(
                eligible=False,
                reason_code="data_degraded_execution",
                snapshot=snapshot,
            )

        if category == "monitoring":
            logger.debug(
                "Eligibility: ineligible for monitoring consumer=%s, "
                "trust_state=degraded, symbol=%s, data_type=%s",
                consumer, snapshot.symbol, data_type,
            )
            return EligibilityResult(
                eligible=False,
                reason_code="data_degraded_monitoring",
                snapshot=snapshot,
            )

        # Display consumers
        if allow_stale_for_display:
            logger.debug(
                "Eligibility: eligible (degraded) for display consumer=%s "
                "with allow_stale_for_display=True, symbol=%s, data_type=%s",
                consumer, snapshot.symbol, data_type,
            )
            return EligibilityResult(
                eligible=True,
                reason_code=None,
                snapshot=snapshot,
            )

        # Display consumer without allow_stale_for_display
        logger.debug(
            "Eligibility: ineligible for display consumer=%s, "
            "trust_state=degraded, allow_stale_for_display=False, "
            "symbol=%s, data_type=%s",
            consumer, snapshot.symbol, data_type,
        )
        return EligibilityResult(
            eligible=False,
            reason_code="data_stale_display",
            snapshot=snapshot,
        )
