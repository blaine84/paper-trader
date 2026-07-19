"""Price Monitor integration for the Market Data Reliability Layer.

Provides trigger eligibility evaluation for the Price Monitor component.
Determines whether stop/target/alert thresholds should be evaluated or
suppressed based on snapshot trust state. Rate-limits degradation log
entries, emits data-degradation safety alerts, and writes structured
telemetry on suppressed triggers.

Key behaviors:
- mode == "disabled": always proceed (do not interfere with existing behavior)
- trust_state == "trusted": proceed with normal trigger evaluation
- trust_state in ("degraded", "untrusted"): suppress trigger, log warning
- Rate-limit log entries: at most once per 60s per symbol
- Emit "data_degradation_safety_alert" telemetry on degradation
- Fail-open: if reliability check fails, proceed with trigger evaluation

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from utils.market_data_reliability.layer import ReliabilityLayer
from utils.market_data_reliability.snapshot import Snapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate-limit interval (seconds) for degraded-check log entries per symbol
# ---------------------------------------------------------------------------

_LOG_RATE_LIMIT_SECONDS: float = 60.0


@dataclass(frozen=True)
class TriggerEligibility:
    """Result of trigger eligibility evaluation for Price Monitor.

    Attributes:
        proceed: True = evaluate triggers normally; False = suppress trigger.
        reason: Human-readable reason why suppressed (None if proceeding).
        snapshot: The Snapshot used for evaluation (None if mode is disabled
            or an error occurred and we are failing open).
    """

    proceed: bool
    reason: Optional[str]
    snapshot: Optional[Snapshot]


class PriceMonitorIntegration:
    """Integration between the Price Monitor and the Market Data Reliability Layer.

    Provides trigger eligibility checks and manages rate-limited logging,
    degradation safety alerts, and suppressed-trigger telemetry.

    Thread-safe: internal rate-limit state uses simple dict access (single-threaded
    scheduler context assumed, matching the Paper Trader APScheduler model).
    """

    def __init__(self) -> None:
        # Rate-limiting: symbol -> last log timestamp (monotonic)
        self._last_log_time: dict[str, float] = {}

    def evaluate_trigger_eligibility(
        self,
        symbol: str,
        reliability_layer: ReliabilityLayer,
        mode: str,
    ) -> TriggerEligibility:
        """Evaluate whether Price Monitor should fire triggers for a symbol.

        This is the primary integration point. Wraps the entire check in
        try/except to ensure fail-open behavior: if the reliability check
        fails for any reason, we proceed with trigger evaluation rather
        than suppressing on error.

        Args:
            symbol: Ticker symbol to evaluate (e.g., "AAPL").
            reliability_layer: The initialized ReliabilityLayer facade.
            mode: Current MARKET_DATA_RELIABILITY_MODE value
                ("disabled", "observe", "enforcing").

        Returns:
            TriggerEligibility indicating whether to proceed or suppress.
        """
        # Guard with feature flag: disabled means do not interfere
        if mode == "disabled":
            return TriggerEligibility(proceed=True, reason=None, snapshot=None)

        try:
            return self._evaluate_inner(symbol, reliability_layer, mode)
        except Exception:
            # Fail-open: reliability check failure should never suppress triggers
            logger.error(
                "PriceMonitorIntegration: error evaluating trigger eligibility "
                "for symbol=%s; failing open (proceeding with evaluation).",
                symbol,
                exc_info=True,
            )
            return TriggerEligibility(proceed=True, reason=None, snapshot=None)

    def _evaluate_inner(
        self,
        symbol: str,
        reliability_layer: ReliabilityLayer,
        mode: str,
    ) -> TriggerEligibility:
        """Inner evaluation logic (may raise, wrapped by evaluate_trigger_eligibility)."""
        # Fetch snapshot via reliability layer for Price_Monitor consumer
        result = reliability_layer.get_snapshot(
            symbol=symbol,
            data_type="quote",
            consumer="Price_Monitor",
            allow_stale_for_display=False,
        )

        snapshot = result.snapshot
        eligible = result.eligibility.eligible

        if eligible:
            # Trusted data: proceed with normal trigger evaluation
            return TriggerEligibility(
                proceed=True,
                reason=None,
                snapshot=snapshot,
            )

        # Data is degraded or untrusted — suppress triggers
        reason = (
            f"trust_state={snapshot.trust_state}, "
            f"freshness_state={snapshot.freshness_state}, "
            f"degradation_reasons={snapshot.degradation_reasons}"
        )

        # Rate-limited logging for degraded-check entries (Requirement 8.4)
        self._log_suppression_rate_limited(symbol, snapshot, reason)

        # Emit data-degradation safety alert telemetry (Requirement 8.2)
        self._emit_degradation_alert(symbol, snapshot)

        # Write structured telemetry on suppressed trigger (Requirement 8.3)
        self._record_suppressed_trigger_telemetry(symbol, snapshot)

        # In observe mode: log but still proceed (don't actually suppress)
        if mode == "observe":
            logger.info(
                "PriceMonitorIntegration OBSERVE: trigger would be suppressed "
                "for symbol=%s but observe mode allows proceed. reason=%s",
                symbol,
                reason,
            )
            return TriggerEligibility(
                proceed=True,
                reason=None,
                snapshot=snapshot,
            )

        # Enforcing mode: suppress the trigger
        return TriggerEligibility(
            proceed=False,
            reason=reason,
            snapshot=snapshot,
        )

    def _log_suppression_rate_limited(
        self, symbol: str, snapshot: Snapshot, reason: str
    ) -> None:
        """Log trigger suppression with rate-limiting per symbol.

        Logs at most once per _LOG_RATE_LIMIT_SECONDS per symbol to
        avoid flooding logs when data remains degraded.
        """
        now = time.monotonic()
        last_time = self._last_log_time.get(symbol)

        if last_time is not None and (now - last_time) < _LOG_RATE_LIMIT_SECONDS:
            # Rate-limited: skip logging
            return

        self._last_log_time[symbol] = now
        logger.warning(
            "PriceMonitorIntegration: trigger suppressed for symbol=%s — "
            "data not trusted. trust_state=%s freshness_state=%s "
            "degradation_reasons=%s",
            symbol,
            snapshot.trust_state,
            snapshot.freshness_state,
            snapshot.degradation_reasons,
        )

    def _emit_degradation_alert(self, symbol: str, snapshot: Snapshot) -> None:
        """Emit a data-degradation safety alert (Requirement 8.2).

        Uses structured logging as telemetry channel. This is a separate
        alert type from trigger suppression — it indicates the data
        degradation condition itself, not just a suppressed trigger.
        """
        try:
            logger.warning(
                "data_degradation_safety_alert: symbol=%s provider=%s "
                "trust_state=%s freshness_state=%s degradation_reasons=%s "
                "provider_status=%s age_seconds=%.1f",
                symbol,
                snapshot.provider,
                snapshot.trust_state,
                snapshot.freshness_state,
                snapshot.degradation_reasons,
                snapshot.provider_status,
                snapshot.age_seconds,
            )
        except Exception:
            logger.error(
                "PriceMonitorIntegration: failed to emit degradation alert "
                "for symbol=%s",
                symbol,
                exc_info=True,
            )

    def _record_suppressed_trigger_telemetry(
        self, symbol: str, snapshot: Snapshot
    ) -> None:
        """Write structured telemetry for a suppressed trigger (Requirement 8.3).

        Includes reason code and latest snapshot state for diagnostics.
        """
        try:
            logger.info(
                "suppressed_trigger_telemetry: symbol=%s trust_state=%s "
                "freshness_state=%s degradation_reasons=%s "
                "provider=%s provider_status=%s age_seconds=%.1f "
                "data_type=quote consumer=Price_Monitor",
                symbol,
                snapshot.trust_state,
                snapshot.freshness_state,
                snapshot.degradation_reasons,
                snapshot.provider,
                snapshot.provider_status,
                snapshot.age_seconds,
            )
        except Exception:
            logger.error(
                "PriceMonitorIntegration: failed to record suppressed trigger "
                "telemetry for symbol=%s",
                symbol,
                exc_info=True,
            )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

# Singleton instance for use by the Price Monitor
_integration = PriceMonitorIntegration()


def evaluate_trigger_eligibility(
    symbol: str,
    reliability_layer: ReliabilityLayer,
    mode: str,
) -> TriggerEligibility:
    """Evaluate whether triggers should fire for a symbol.

    Convenience function using module-level singleton. This is the primary
    entry point for the Price Monitor to call.

    Args:
        symbol: Ticker symbol.
        reliability_layer: The initialized ReliabilityLayer.
        mode: MARKET_DATA_RELIABILITY_MODE value.

    Returns:
        TriggerEligibility with proceed/suppress decision.
    """
    return _integration.evaluate_trigger_eligibility(symbol, reliability_layer, mode)
