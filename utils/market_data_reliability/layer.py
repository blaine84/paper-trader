"""ReliabilityLayer facade for the Market Data Reliability Layer.

Orchestrates the full snapshot retrieval pipeline: cache check → coalescer →
provider call → validate → normalize → cache store → eligibility → return.

Always returns SnapshotResult (never raises). On failure, produces untrusted/
unavailable snapshots with appropriate degradation_reasons.

Requirements: 1.5, 3.6, 5.4, 7.6
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, Optional

from utils.market_data_reliability.backoff import BackoffTracker
from utils.market_data_reliability.cache import SnapshotCache
from utils.market_data_reliability.coalescer import RequestCoalescer
from utils.market_data_reliability.config import ReliabilityConfig
from utils.market_data_reliability.eligibility import EligibilityResolver
from utils.market_data_reliability.fallback import FallbackRouter
from utils.market_data_reliability.freshness import FreshnessClassifier
from utils.market_data_reliability.normalizer import SnapshotNormalizer
from utils.market_data_reliability.snapshot import (
    CacheKey,
    CandidateReadiness,
    CoalesceKey,
    EligibilityResult,
    Snapshot,
    SnapshotResult,
)
from utils.market_data_reliability.telemetry import TelemetryCollector
from utils.market_data_reliability.trust import TrustClassifier
from utils.market_data_reliability.validator import ResponseValidator

logger = logging.getLogger(__name__)


class ReliabilityLayer:
    """Facade orchestrating the full market data reliability pipeline.

    Wires all sub-components from a ReliabilityConfig and exposes a simple
    get_snapshot() method that always returns a SnapshotResult — never raises.

    The provider integration point is a callable injected at construction:
        fetch_from_provider(provider: str, symbol: str, data_type: str) -> dict

    This callable should return a raw provider response dict. Raise on failure.
    If no callable is provided, a default that raises NotImplementedError is used.
    """

    def __init__(
        self,
        config: ReliabilityConfig,
        fetch_from_provider: Optional[Callable[[str, str, str], dict]] = None,
    ) -> None:
        self._config = config

        # Provider fetch callable (injectable for testing)
        self._fetch_from_provider: Callable[[str, str, str], dict] = (
            fetch_from_provider if fetch_from_provider is not None
            else self._default_fetch_from_provider
        )

        # Wire sub-components from config
        self._freshness_classifier = FreshnessClassifier(config.freshness_thresholds)
        self._trust_classifier = TrustClassifier()
        self._validator = ResponseValidator(staleness_threshold_seconds=300.0)
        self._normalizer = SnapshotNormalizer(
            validator=self._validator,
            freshness_classifier=self._freshness_classifier,
            trust_classifier=self._trust_classifier,
            default_market_session="open",
        )
        self._cache = SnapshotCache(
            cache_ttls=config.cache_ttls,
            freshness_classifier=self._freshness_classifier,
        )
        self._coalescer = RequestCoalescer(timeout_seconds=12.0)
        self._backoff_tracker = BackoffTracker(config.backoff_durations)
        self._fallback_router = FallbackRouter(
            fallback_matrix=config.fallback_matrix,
            backoff_tracker=self._backoff_tracker,
        )
        self._eligibility_resolver = EligibilityResolver()
        self._telemetry = TelemetryCollector()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def get_snapshot(
        self,
        symbol: str,
        data_type: str,
        consumer: str,
        allow_stale_for_display: bool = False,
    ) -> SnapshotResult:
        """Retrieve a market data snapshot through the full reliability pipeline.

        Orchestration flow:
            1. Check cache (key = CacheKey(symbol, data_type, "primary", market_session))
            2. If cache hit: run eligibility, record telemetry, return
            3. If cache miss: use coalescer to deduplicate
            4. Inside coalescer fetch_fn: resolve providers via FallbackRouter,
               try each in order (call provider → validate → normalize → cache)
            5. If all providers fail: produce untrusted/unavailable snapshot
            6. Run EligibilityResolver on the snapshot
            7. Return SnapshotResult(snapshot, eligibility)

        Always returns a SnapshotResult — never raises.

        Mode behavior (MARKET_DATA_RELIABILITY_MODE):
            disabled: Return a passthrough result (always eligible) without
                making provider calls or running the pipeline.
            observe: Run the full pipeline and produce snapshots/telemetry,
                but override eligibility to always eligible. Log a warning
                when a snapshot would have been blocked in enforcing mode.
            enforcing: Full fail-closed behavior (current pipeline logic).

        Args:
            symbol: Ticker symbol (e.g., "AAPL").
            data_type: Data type ("quote", "candle", "atr", "volume", "previous_close").
            consumer: Consumer name (e.g., "PM", "Dashboard_API").
            allow_stale_for_display: When True, display consumers may receive
                degraded data with explicit labeling.

        Returns:
            SnapshotResult coupling the snapshot with its eligibility verdict.
        """
        # --- disabled mode: passthrough, no provider calls ---
        if self._config.mode == "disabled":
            return self._make_passthrough_result(symbol, data_type)

        try:
            result = self._get_snapshot_inner(
                symbol, data_type, consumer, allow_stale_for_display
            )
        except Exception:
            # Internal error — log and return unavailable snapshot
            logger.error(
                "ReliabilityLayer.get_snapshot internal error for "
                "symbol=%s data_type=%s consumer=%s",
                symbol, data_type, consumer,
                exc_info=True,
            )
            snapshot = self._make_unavailable_snapshot(symbol, data_type)
            self._telemetry.record_unavailable_snapshot(symbol, data_type)
            eligibility = self._eligibility_resolver.is_eligible(
                snapshot, consumer, data_type, allow_stale_for_display
            )
            result = SnapshotResult(snapshot=snapshot, eligibility=eligibility)

        # --- observe mode: log but do not block ---
        if self._config.mode == "observe":
            if not result.eligibility.eligible:
                logger.warning(
                    "ReliabilityLayer OBSERVE mode: snapshot would be blocked "
                    "in enforcing mode — symbol=%s data_type=%s consumer=%s "
                    "reason=%s trust_state=%s freshness_state=%s",
                    symbol, data_type, consumer,
                    result.eligibility.reason_code,
                    result.snapshot.trust_state,
                    result.snapshot.freshness_state,
                )
                # Override eligibility to allow (observe does not block)
                observe_eligibility = EligibilityResult(
                    eligible=True,
                    reason_code=None,
                    snapshot=result.snapshot,
                )
                return SnapshotResult(
                    snapshot=result.snapshot, eligibility=observe_eligibility
                )

        # --- enforcing mode: return as-is (full fail-closed) ---
        return result

    def check_candidate_readiness(
        self,
        symbol: str,
        required_data_types: list[str],
        consumer: str,
    ) -> CandidateReadiness:
        """Check market-data readiness for a candidate across required data types.

        Fetches snapshots for each required data type and evaluates eligibility.
        A candidate is ready only if all required data types have eligible snapshots.

        Mode behavior (MARKET_DATA_RELIABILITY_MODE):
            disabled: Return immediately with ready=True, no provider calls.
            observe: Run the full pipeline, log warnings for blocked candidates,
                but always return ready=True.
            enforcing: Full fail-closed behavior — candidate blocked when data
                is unavailable or untrusted.

        Produces structured reason codes per Requirements 7.1, 7.2, 7.3:
            - market_data_unavailable: data type has no provider or all failed
            - quote_stale: quote data is stale or untrusted
            - atr_stale: ATR data is stale or untrusted
            - volume_unavailable: volume data is unavailable
            - provider_rate_limited: provider is rate-limited for the data type

        Args:
            symbol: Ticker symbol.
            required_data_types: List of data types to check (e.g., ["quote", "atr"]).
            consumer: Consumer name for eligibility evaluation.

        Returns:
            CandidateReadiness with ready status, missing types, and reason codes.
        """
        # --- disabled mode: passthrough, always ready ---
        if self._config.mode == "disabled":
            return CandidateReadiness(
                ready=True,
                missing_data_types=(),
                reason_codes=(),
                snapshots={},
            )

        missing_data_types: list[str] = []
        reason_codes: list[str] = []
        snapshots: dict[str, Snapshot] = {}

        for dt in required_data_types:
            result = self.get_snapshot(symbol, dt, consumer)
            snapshots[dt] = result.snapshot

            if not result.eligibility.eligible:
                missing_data_types.append(dt)
                reason_code = self._map_reason_code(dt, result.snapshot)
                if reason_code and reason_code not in reason_codes:
                    reason_codes.append(reason_code)

        ready = len(missing_data_types) == 0

        # --- observe mode: log warnings but always return ready ---
        if self._config.mode == "observe" and not ready:
            logger.warning(
                "ReliabilityLayer OBSERVE mode: candidate would be blocked "
                "in enforcing mode — symbol=%s consumer=%s "
                "missing_data_types=%s reason_codes=%s",
                symbol, consumer,
                missing_data_types, reason_codes,
            )
            return CandidateReadiness(
                ready=True,
                missing_data_types=(),
                reason_codes=(),
                snapshots=snapshots,
            )

        # --- enforcing mode: full fail-closed ---
        return CandidateReadiness(
            ready=ready,
            missing_data_types=tuple(missing_data_types),
            reason_codes=tuple(reason_codes),
            snapshots=snapshots,
        )

    def _map_reason_code(self, data_type: str, snapshot: Snapshot) -> str:
        """Map a data_type and snapshot state to a structured reason code.

        Produces one of the canonical reason codes defined in Requirement 7.3:
            - provider_rate_limited: any degradation reason contains "rate_limited"
              or all providers are in rate-limit backoff for this data type
            - quote_stale: quote data is stale/unavailable/untrusted
            - atr_stale: ATR data is stale/unavailable/untrusted
            - volume_unavailable: volume data is unavailable/untrusted
            - market_data_unavailable: fallback for unrecognized data types
        """
        # Check for rate-limiting first — applies to any data type
        if any("rate_limited" in r for r in snapshot.degradation_reasons):
            return "provider_rate_limited"

        # Check backoff tracker: if all known providers for this data type are
        # in rate-limit backoff, treat as provider_rate_limited
        if self._is_all_providers_rate_limited(data_type):
            return "provider_rate_limited"

        # Map by data_type — specific codes take priority over generic
        if data_type == "quote":
            return "quote_stale"
        if data_type == "atr":
            return "atr_stale"
        if data_type == "volume":
            return "volume_unavailable"

        # Fallback for other data types (candle, previous_close, etc.)
        return "market_data_unavailable"

    def _is_all_providers_rate_limited(self, data_type: str) -> bool:
        """Check if all configured providers are in rate-limit backoff for a data type.

        Returns True only if every provider for this data_type is in backoff
        AND the failure type for at least one is "rate_limit".
        """
        # Get the providers from the fallback matrix for execution consumers
        providers: list[str] = []
        for (dt, _category), provider_list in self._config.fallback_matrix.items():
            if dt == data_type:
                for p in provider_list:
                    if p not in providers:
                        providers.append(p)

        if not providers:
            return False

        # All providers must be in backoff
        all_in_backoff = all(
            self._backoff_tracker.is_in_backoff(p, data_type) for p in providers
        )
        if not all_in_backoff:
            return False

        # At least one must be specifically rate-limited (not just network error)
        any_rate_limited = any(
            self._backoff_tracker.get_failure_type(p, data_type) == "rate_limit"
            for p in providers
        )
        return any_rate_limited

    def get_cycle_telemetry(self) -> dict:
        """Return accumulated telemetry metrics for the current cycle.

        Delegates to TelemetryCollector.get_cycle_summary().
        """
        return self._telemetry.get_cycle_summary()

    def reset_cycle(self) -> None:
        """Reset state for a new cycle.

        Clears the cache and resets telemetry counters.
        """
        self._cache.clear()
        self._telemetry.reset()

    # -----------------------------------------------------------------------
    # Internal orchestration
    # -----------------------------------------------------------------------

    def _get_snapshot_inner(
        self,
        symbol: str,
        data_type: str,
        consumer: str,
        allow_stale_for_display: bool,
    ) -> SnapshotResult:
        """Inner orchestration logic (may raise — wrapped by get_snapshot)."""
        market_session = "open"  # Default; future: detect real market hours

        # Step 1: Check cache
        cache_key = CacheKey(
            symbol=symbol,
            data_type=data_type,
            provider_policy="primary",
            market_session=market_session,
        )
        cached_snapshot = self._cache.get(cache_key)

        if cached_snapshot is not None:
            # Cache hit — record telemetry and run eligibility
            self._telemetry.record_cache_hit(data_type)
            eligibility = self._eligibility_resolver.is_eligible(
                cached_snapshot, consumer, data_type, allow_stale_for_display
            )
            return SnapshotResult(snapshot=cached_snapshot, eligibility=eligibility)

        # Step 2: Cache miss — use coalescer to fetch
        coalesce_key = CoalesceKey(
            symbol=symbol,
            data_type=data_type,
            provider_policy="primary",
        )

        def fetch_fn() -> Snapshot:
            return self._fetch_with_fallback(
                symbol, data_type, consumer, market_session, cache_key
            )

        snapshot = self._coalescer.get_or_start(coalesce_key, fetch_fn)

        # Step 3: Run eligibility on the fetched snapshot
        eligibility = self._eligibility_resolver.is_eligible(
            snapshot, consumer, data_type, allow_stale_for_display
        )

        # Step 4: Record stale/unavailable telemetry
        if snapshot.freshness_state == "stale":
            self._telemetry.record_stale_snapshot(symbol, data_type)
        elif snapshot.freshness_state == "unavailable":
            self._telemetry.record_unavailable_snapshot(symbol, data_type)

        return SnapshotResult(snapshot=snapshot, eligibility=eligibility)

    def _fetch_with_fallback(
        self,
        symbol: str,
        data_type: str,
        consumer: str,
        market_session: str,
        cache_key: CacheKey,
    ) -> Snapshot:
        """Resolve providers and try each in order until one succeeds.

        On success: validate → normalize → cache → return.
        On failure: record in backoff tracker, try next provider.
        If all fail: produce untrusted/unavailable snapshot.
        """
        providers = self._fallback_router.resolve_providers(data_type, consumer)

        if not providers:
            logger.warning(
                "No providers available for symbol=%s data_type=%s consumer=%s",
                symbol, data_type, consumer,
            )
            self._telemetry.record_unavailable_snapshot(symbol, data_type)
            return self._make_unavailable_snapshot(symbol, data_type)

        primary_provider = providers[0]
        last_exception: Optional[Exception] = None

        for idx, provider in enumerate(providers):
            requested_at = datetime.now(timezone.utc)
            try:
                raw = self._fetch_from_provider(provider, symbol, data_type)
                fetched_at = datetime.now(timezone.utc)

                # Normalize the raw response into a Snapshot
                if data_type == "candle":
                    snapshot = self._normalizer.normalize_candles(
                        raw, symbol, provider, requested_at, fetched_at
                    )
                else:
                    snapshot = self._normalizer.normalize_quote(
                        raw, symbol, provider, requested_at, fetched_at
                    )

                # If this was a fallback provider, annotate it
                if idx > 0:
                    from dataclasses import replace
                    snapshot = replace(
                        snapshot, fallback_primary_provider=primary_provider
                    )
                    self._telemetry.record_fallback_usage(
                        primary_provider, provider, data_type
                    )

                # Record success
                self._telemetry.record_provider_call(provider, data_type, success=True)
                self._backoff_tracker.record_success(provider, data_type)

                # Store in cache
                self._cache.put(cache_key, snapshot)

                return snapshot

            except Exception as exc:
                last_exception = exc
                fetched_at = datetime.now(timezone.utc)
                logger.warning(
                    "Provider %s failed for symbol=%s data_type=%s: %s",
                    provider, symbol, data_type, exc,
                )
                self._telemetry.record_provider_call(provider, data_type, success=False)
                self._backoff_tracker.record_failure(
                    provider, data_type, self._classify_failure(exc)
                )
                continue

        # All providers failed
        logger.error(
            "All providers failed for symbol=%s data_type=%s consumer=%s. "
            "Last error: %s",
            symbol, data_type, consumer, last_exception,
        )
        self._telemetry.record_unavailable_snapshot(symbol, data_type)
        return self._make_all_providers_failed_snapshot(symbol, data_type)

    # -----------------------------------------------------------------------
    # Snapshot factories for failure cases
    # -----------------------------------------------------------------------

    def _make_passthrough_result(
        self, symbol: str, data_type: str
    ) -> SnapshotResult:
        """Create a passthrough SnapshotResult for disabled mode.

        Returns a minimal trusted snapshot that is always eligible, without
        making any provider calls or touching the cache. Used when the
        reliability layer is disabled and should not intercept data paths.
        """
        now = datetime.now(timezone.utc)
        snapshot = Snapshot(
            symbol=symbol,
            data_type=data_type,
            requested_at=now,
            provider="passthrough",
            provider_status="success",
            market_session="open",
            last_price=None,
            bid=None,
            ask=None,
            previous_close=None,
            open=None,
            high=None,
            low=None,
            volume=None,
            fetched_at=now,
            source_timestamp=None,
            age_seconds=0.0,
            freshness_state="fresh",
            trust_state="trusted",
            degradation_reasons=(),
            raw_provider_latency_ms=None,
            fallback_primary_provider=None,
        )
        eligibility = EligibilityResult(
            eligible=True,
            reason_code=None,
            snapshot=snapshot,
        )
        return SnapshotResult(snapshot=snapshot, eligibility=eligibility)

    def _make_unavailable_snapshot(self, symbol: str, data_type: str) -> Snapshot:
        """Create a snapshot representing unavailable data (no providers configured)."""
        now = datetime.now(timezone.utc)
        return Snapshot(
            symbol=symbol,
            data_type=data_type,
            requested_at=now,
            provider="none",
            provider_status="error",
            market_session="open",
            last_price=None,
            bid=None,
            ask=None,
            previous_close=None,
            open=None,
            high=None,
            low=None,
            volume=None,
            fetched_at=now,
            source_timestamp=None,
            age_seconds=0.0,
            freshness_state="unavailable",
            trust_state="untrusted",
            degradation_reasons=("no_providers_available",),
            raw_provider_latency_ms=None,
            fallback_primary_provider=None,
        )

    def _make_all_providers_failed_snapshot(
        self, symbol: str, data_type: str
    ) -> Snapshot:
        """Create a snapshot representing all providers failed."""
        now = datetime.now(timezone.utc)
        return Snapshot(
            symbol=symbol,
            data_type=data_type,
            requested_at=now,
            provider="none",
            provider_status="error",
            market_session="open",
            last_price=None,
            bid=None,
            ask=None,
            previous_close=None,
            open=None,
            high=None,
            low=None,
            volume=None,
            fetched_at=now,
            source_timestamp=None,
            age_seconds=0.0,
            freshness_state="unavailable",
            trust_state="untrusted",
            degradation_reasons=("all_providers_failed",),
            raw_provider_latency_ms=None,
            fallback_primary_provider=None,
        )

    # -----------------------------------------------------------------------
    # Mode helpers
    # -----------------------------------------------------------------------

    def _make_passthrough_result(self, symbol: str, data_type: str) -> SnapshotResult:
        """Create a passthrough result for disabled mode.

        Returns a minimal trusted/eligible SnapshotResult without making
        any provider calls. This ensures the layer is transparent when disabled.
        """
        now = datetime.now(timezone.utc)
        snapshot = Snapshot(
            symbol=symbol,
            data_type=data_type,
            requested_at=now,
            provider="passthrough",
            provider_status="success",
            market_session="open",
            last_price=None,
            bid=None,
            ask=None,
            previous_close=None,
            open=None,
            high=None,
            low=None,
            volume=None,
            fetched_at=now,
            source_timestamp=None,
            age_seconds=0.0,
            freshness_state="fresh",
            trust_state="trusted",
            degradation_reasons=(),
            raw_provider_latency_ms=None,
            fallback_primary_provider=None,
        )
        eligibility = EligibilityResult(
            eligible=True,
            reason_code=None,
            snapshot=snapshot,
        )
        return SnapshotResult(snapshot=snapshot, eligibility=eligibility)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _default_fetch_from_provider(
        provider: str, symbol: str, data_type: str
    ) -> dict:
        """Default provider fetch — raises NotImplementedError.

        Real provider adapters should be injected via the constructor.
        """
        raise NotImplementedError(
            f"No provider adapter configured for {provider} "
            f"(symbol={symbol}, data_type={data_type}). "
            f"Inject a fetch_from_provider callable at construction."
        )

    @staticmethod
    def _classify_failure(exc: Exception) -> str:
        """Classify an exception into a backoff failure_type.

        Returns one of: "rate_limit", "network_error", "empty_response".
        """
        exc_str = str(exc).lower()
        if "rate" in exc_str or "429" in exc_str or "limit" in exc_str:
            return "rate_limit"
        if "empty" in exc_str or "no data" in exc_str:
            return "empty_response"
        return "network_error"
