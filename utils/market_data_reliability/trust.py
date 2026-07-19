"""Trust state classification for the Market Data Reliability Layer.

Computes trust_state from validation results, freshness state, consumer
category, and market session. Safety-critical failures always produce
untrusted; display-only failures produce degraded; otherwise trusted.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
"""
from __future__ import annotations

import logging

from utils.market_data_reliability.snapshot import ValidationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants: reason codes by severity
# ---------------------------------------------------------------------------

# Safety-critical reason codes that always produce untrusted state.
# These represent data integrity failures where the data cannot be relied upon.
SAFETY_CRITICAL_REASONS: frozenset[str] = frozenset({
    "cross_symbol_response",
    "invalid_price",
    "all_providers_failed",
})

# Non-critical reason codes that produce degraded state.
# Data is incomplete or suboptimal but may be renderable for display consumers.
NON_CRITICAL_REASONS: frozenset[str] = frozenset({
    "missing_source_timestamp",
    "stale_source_timestamp",
    "provider_error",
    "rate_limited",
    "empty_response",
})

# ---------------------------------------------------------------------------
# Consumer category mappings
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
        "Unknown consumer '%s' in trust classification; defaulting to execution category.",
        consumer,
    )
    return "execution"


# ---------------------------------------------------------------------------
# TrustClassifier
# ---------------------------------------------------------------------------


class TrustClassifier:
    """Classifies trust state based on validation, freshness, consumer, and market session.

    Returns a tuple of (trust_state, degradation_reasons) where:
      - trust_state is one of: "trusted", "degraded", "untrusted"
      - degradation_reasons is a tuple of stable reason code strings
        (empty for trusted, >= 1 entry for degraded/untrusted)
    """

    def classify(
        self,
        validation_result: ValidationResult,
        freshness_state: str,
        consumer: str,
        market_session: str,
    ) -> tuple[str, tuple[str, ...]]:
        """Classify trust state for a snapshot.

        Args:
            validation_result: Result from ResponseValidator containing
                is_valid and degradation_reasons.
            freshness_state: One of "fresh", "aging", "stale",
                "unavailable", "market_closed".
            consumer: Consumer name (e.g., "PM", "Dashboard_API").
            market_session: One of "open", "pre_market", "after_hours", "closed".

        Returns:
            Tuple of (trust_state, degradation_reasons).
            trust_state: "trusted", "degraded", or "untrusted".
            degradation_reasons: tuple of reason code strings.
        """
        category = _consumer_category(consumer)
        reasons: list[str] = []

        # ---------------------------------------------------------------
        # Step 1: Check for safety-critical validation failures → untrusted
        # ---------------------------------------------------------------
        safety_critical_found = [
            r for r in validation_result.degradation_reasons
            if r in SAFETY_CRITICAL_REASONS
        ]
        if safety_critical_found:
            reasons.extend(safety_critical_found)
            logger.debug(
                "Trust classification: untrusted due to safety-critical reasons %s "
                "for consumer=%s",
                safety_critical_found, consumer,
            )
            return ("untrusted", tuple(reasons))

        # ---------------------------------------------------------------
        # Step 2: Check freshness_state == "unavailable" → untrusted
        # ---------------------------------------------------------------
        if freshness_state == "unavailable":
            reasons.append("data_unavailable")
            logger.debug(
                "Trust classification: untrusted due to unavailable freshness "
                "for consumer=%s",
                consumer,
            )
            return ("untrusted", tuple(reasons))

        # ---------------------------------------------------------------
        # Step 3: market_closed + execution consumer → untrusted
        # (no explicit after-hours flag support in this context)
        # ---------------------------------------------------------------
        if freshness_state == "market_closed" and category == "execution":
            reasons.append("market_closed_execution")
            logger.debug(
                "Trust classification: untrusted due to market_closed for "
                "execution consumer=%s",
                consumer,
            )
            return ("untrusted", tuple(reasons))

        # ---------------------------------------------------------------
        # Step 4: Check freshness_state == "stale" for execution → untrusted
        # ---------------------------------------------------------------
        if freshness_state == "stale" and category == "execution":
            reasons.append("stale_data_execution")
            logger.debug(
                "Trust classification: untrusted due to stale data for "
                "execution consumer=%s",
                consumer,
            )
            return ("untrusted", tuple(reasons))

        # ---------------------------------------------------------------
        # Step 5: Check for non-critical validation failures → degraded
        # ---------------------------------------------------------------
        non_critical_found = [
            r for r in validation_result.degradation_reasons
            if r in NON_CRITICAL_REASONS
        ]
        if non_critical_found:
            reasons.extend(non_critical_found)
            logger.debug(
                "Trust classification: degraded due to non-critical reasons %s "
                "for consumer=%s",
                non_critical_found, consumer,
            )
            return ("degraded", tuple(reasons))

        # ---------------------------------------------------------------
        # Step 6: Stale freshness for non-execution consumers → degraded
        # ---------------------------------------------------------------
        if freshness_state == "stale":
            reasons.append("stale_data")
            logger.debug(
                "Trust classification: degraded due to stale data for "
                "consumer=%s (non-execution)",
                consumer,
            )
            return ("degraded", tuple(reasons))

        # ---------------------------------------------------------------
        # Step 7: Otherwise → trusted
        # ---------------------------------------------------------------
        return ("trusted", ())
