"""Pipeline integration for Market Data Reliability Layer.

Provides a guarded integration point for the candidate pipeline to check
market-data readiness before gate execution. Fail-open: if the reliability
layer itself errors, log and continue (never block on infrastructure failures).

Feature-flag guarded by MARKET_DATA_RELIABILITY_MODE from gate_config.

The integration function returns a simple MarketDataReadinessResult rather
than exposing the full CandidateReadiness / Snapshot internals to the pipeline.
This keeps the pipeline contract minimal and decoupled from layer internals.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Default minimum data types required for candidate execution.
_DEFAULT_REQUIRED_DATA_TYPES: list[str] = ["quote"]

# Module-level singleton to avoid re-creating the layer per candidate.
_reliability_layer_instance: Optional[object] = None


@dataclass(frozen=True)
class MarketDataReadinessResult:
    """Simplified pipeline-facing result of a market-data readiness check.

    Decouples the candidate pipeline from Snapshot/CandidateReadiness internals.

    Attributes:
        proceed: True if the candidate can proceed; False if blocked.
        reason_codes: Structured reason codes explaining why blocked (empty if proceed=True).
        missing_data_types: Which required data types are missing/untrusted (empty if proceed=True).
    """

    proceed: bool
    reason_codes: tuple[str, ...]
    missing_data_types: tuple[str, ...]


def _get_reliability_layer():
    """Lazy-initialize and cache a ReliabilityLayer singleton.

    Returns None if construction fails (fail-open).
    """
    global _reliability_layer_instance
    if _reliability_layer_instance is not None:
        return _reliability_layer_instance

    try:
        from utils.market_data_reliability.config import ReliabilityConfig
        from utils.market_data_reliability.layer import ReliabilityLayer

        config = ReliabilityConfig.from_environment()
        _reliability_layer_instance = ReliabilityLayer(config)
        return _reliability_layer_instance
    except Exception:
        logger.error(
            "Failed to initialize ReliabilityLayer singleton; "
            "market data readiness checks will be skipped (fail-open)",
            exc_info=True,
        )
        return None


def check_market_data_readiness(
    symbol: str,
    required_data_types: list[str] | None = None,
    consumer: str = "PM",
) -> MarketDataReadinessResult:
    """Check market-data readiness for a candidate before gate execution.

    This is the single integration point called by the candidate pipeline.
    It is guarded by the MARKET_DATA_RELIABILITY_MODE feature flag:
        - disabled: always returns proceed=True (no interference)
        - observe: runs checks, logs warnings for blocked candidates,
          but returns proceed=True (never blocks)
        - enforcing: runs checks, blocks candidates when data is
          unavailable or untrusted

    The entire function is wrapped in try/except (fail-open). If any
    internal error occurs, it logs and returns proceed=True (pipeline continues).

    Args:
        symbol: Ticker symbol for the candidate.
        required_data_types: Data types needed for execution.
            Defaults to ["quote"] if None (minimum for execution).
        consumer: Consumer name for eligibility checks (default: "PM").

    Returns:
        MarketDataReadinessResult with proceed status, reason codes, and
        missing data types.
    """
    from utils.gate_config import MARKET_DATA_RELIABILITY_MODE

    # --- disabled mode: no interference, always proceed ---
    if MARKET_DATA_RELIABILITY_MODE == "disabled":
        return MarketDataReadinessResult(
            proceed=True,
            reason_codes=(),
            missing_data_types=(),
        )

    # Apply default required data types
    if required_data_types is None:
        required_data_types = list(_DEFAULT_REQUIRED_DATA_TYPES)

    try:
        layer = _get_reliability_layer()
        if layer is None:
            # Layer failed to initialize — fail-open, proceed
            return MarketDataReadinessResult(
                proceed=True,
                reason_codes=(),
                missing_data_types=(),
            )

        readiness = layer.check_candidate_readiness(
            symbol=symbol,
            required_data_types=required_data_types,
            consumer=consumer,
        )

        # Map CandidateReadiness to the pipeline-facing result
        if readiness.ready:
            return MarketDataReadinessResult(
                proceed=True,
                reason_codes=(),
                missing_data_types=(),
            )

        # Candidate is not ready — data unavailable or untrusted
        reason_codes = readiness.reason_codes
        missing_data_types = readiness.missing_data_types

        # Determine proceed based on mode
        if MARKET_DATA_RELIABILITY_MODE == "observe":
            # Observe mode: log warning but allow proceed
            logger.warning(
                "market_data_readiness_check OBSERVE: candidate would be blocked "
                "symbol=%s consumer=%s missing_data_types=%s reason_codes=%s",
                symbol,
                consumer,
                missing_data_types,
                reason_codes,
            )
            return MarketDataReadinessResult(
                proceed=True,
                reason_codes=reason_codes,
                missing_data_types=missing_data_types,
            )

        # Enforcing mode: block candidate, record telemetry
        logger.info(
            "market_data_readiness_check ENFORCING: blocking candidate "
            "symbol=%s consumer=%s missing_data_types=%s reason_codes=%s",
            symbol,
            consumer,
            missing_data_types,
            reason_codes,
        )

        # Write structured reason to candidate blocker telemetry (fail-open)
        _record_blocker_telemetry(symbol, reason_codes, consumer)

        return MarketDataReadinessResult(
            proceed=False,
            reason_codes=reason_codes,
            missing_data_types=missing_data_types,
        )

    except Exception:
        logger.error(
            "Market data readiness check failed for symbol=%s consumer=%s; "
            "continuing (fail-open)",
            symbol,
            consumer,
            exc_info=True,
        )
        return MarketDataReadinessResult(
            proceed=True,
            reason_codes=(),
            missing_data_types=(),
        )


def _record_blocker_telemetry(
    symbol: str,
    reason_codes: tuple[str, ...],
    consumer: str,
) -> None:
    """Record structured telemetry for a blocked candidate (fail-open).

    Writes to the ReliabilityLayer's TelemetryCollector so that blocked
    candidates are attributed to market-data readiness in cycle summaries.
    """
    try:
        layer = _get_reliability_layer()
        if layer is None:
            return

        # Use the layer's telemetry collector to record the fail-closed decision
        reason_str = ",".join(reason_codes) if reason_codes else "unknown"
        layer._telemetry.record_fail_closed(
            candidate_id=f"{symbol}:{consumer}",
            reason=reason_str,
        )
    except Exception:
        logger.error(
            "Failed to record blocker telemetry for symbol=%s; continuing",
            symbol,
            exc_info=True,
        )


def reset_reliability_layer_singleton() -> None:
    """Reset the cached ReliabilityLayer singleton (for testing).

    Allows tests to force re-initialization with different config/env vars.
    """
    global _reliability_layer_instance
    _reliability_layer_instance = None
