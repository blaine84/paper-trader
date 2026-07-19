"""Snapshot frozen dataclass and supporting data models.

Core data contract for the Market Data Reliability Layer. All quote-sensitive
consumers receive normalized Snapshot instances rather than raw provider dicts.

Requirements: 1.1, 1.2, 1.3, 7.1
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional


@dataclass(frozen=True)
class Snapshot:
    """Frozen, normalized representation of market data for one symbol at one point in time.

    All price fields are Optional[Decimal] — never float.
    Fields not available from the provider are set to None (never omitted).
    degradation_reasons is an immutable tuple of stable reason codes.
    """

    # Identity
    symbol: str
    data_type: str  # "quote", "candle", "atr", "volume", "previous_close"
    requested_at: datetime

    # Provider metadata
    provider: str  # "finnhub", "yfinance", "alpaca"
    provider_status: str  # "success", "error", "rate_limited", "timeout", "empty"
    market_session: str  # "open", "pre_market", "after_hours", "closed"

    # Price fields — all Decimal or None
    last_price: Optional[Decimal]
    bid: Optional[Decimal]
    ask: Optional[Decimal]
    previous_close: Optional[Decimal]
    open: Optional[Decimal]
    high: Optional[Decimal]
    low: Optional[Decimal]
    volume: Optional[int]

    # Temporal metadata
    fetched_at: datetime
    source_timestamp: Optional[datetime]
    age_seconds: float

    # Classification
    freshness_state: str  # "fresh", "aging", "stale", "unavailable", "market_closed"
    trust_state: str  # "trusted", "degraded", "untrusted"
    degradation_reasons: tuple[str, ...]  # stable reason codes, immutable

    # Diagnostics
    raw_provider_latency_ms: Optional[float]
    fallback_primary_provider: Optional[str]  # set when fallback was used


@dataclass(frozen=True)
class EligibilityResult:
    """Result of eligibility check for a snapshot against a specific consumer.

    Couples the eligibility verdict with a reason code and the evaluated snapshot.
    """

    eligible: bool
    reason_code: Optional[str]
    snapshot: Snapshot


@dataclass(frozen=True)
class SnapshotResult:
    """Couples a Snapshot with its eligibility verdict so consumers cannot
    forget to check eligibility before acting on the data.
    """

    snapshot: Snapshot
    eligibility: EligibilityResult


@dataclass(frozen=True)
class CandidateReadiness:
    """Result of market-data readiness check for a candidate.

    Aggregates snapshot checks across all required data types for a symbol.
    """

    ready: bool
    missing_data_types: tuple[str, ...]
    reason_codes: tuple[str, ...]  # "market_data_unavailable", "quote_stale", etc.
    snapshots: dict[str, Snapshot]  # data_type -> snapshot for checked types


@dataclass(frozen=True)
class ValidationResult:
    """Result of response validation before normalization."""

    is_valid: bool
    degradation_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CacheKey:
    """Cache lookup key for snapshot storage.

    Scoped by symbol, data_type, provider_policy, and market_session.
    """

    symbol: str
    data_type: str
    provider_policy: str  # "primary", "fallback", or specific provider name
    market_session: str


@dataclass(frozen=True)
class CoalesceKey:
    """Key for request coalescing to prevent duplicate in-flight provider calls.

    Mirrors CacheKey scope minus market_session for coalescing purposes.
    """

    symbol: str
    data_type: str
    provider_policy: str


@dataclass(frozen=True)
class FreshnessThreshold:
    """Configurable freshness thresholds for a (data_type, consumer) pair.

    age_seconds < fresh_threshold → fresh
    age_seconds < aging_threshold → aging
    age_seconds >= aging_threshold → stale
    """

    fresh_threshold: float
    aging_threshold: float
