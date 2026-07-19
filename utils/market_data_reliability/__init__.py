"""Market Data Reliability Layer.

Shared module that normalizes provider responses into trusted snapshots
with freshness/trust classification for all quote-sensitive consumers.
"""
from __future__ import annotations

from utils.market_data_reliability.backoff import BackoffTracker
from utils.market_data_reliability.cache import SnapshotCache
from utils.market_data_reliability.config import ReliabilityConfig
from utils.market_data_reliability.dashboard_integration import (
    enrich_market_data_response,
    get_freshness_label,
)
from utils.market_data_reliability.pipeline_integration import (
    MarketDataReadinessResult,
    check_market_data_readiness,
)
from utils.market_data_reliability.layer import ReliabilityLayer
from utils.market_data_reliability.serialization import deserialize, serialize
from utils.market_data_reliability.telemetry import TelemetryCollector
from utils.market_data_reliability.snapshot import (
    CacheKey,
    CandidateReadiness,
    CoalesceKey,
    EligibilityResult,
    FreshnessThreshold,
    Snapshot,
    SnapshotResult,
    ValidationResult,
)

__all__ = [
    "BackoffTracker",
    "CacheKey",
    "CandidateReadiness",
    "check_market_data_readiness",
    "CoalesceKey",
    "deserialize",
    "EligibilityResult",
    "enrich_market_data_response",
    "FreshnessThreshold",
    "get_freshness_label",
    "MarketDataReadinessResult",
    "ReliabilityConfig",
    "ReliabilityLayer",
    "serialize",
    "Snapshot",
    "SnapshotCache",
    "SnapshotResult",
    "TelemetryCollector",
    "ValidationResult",
]
