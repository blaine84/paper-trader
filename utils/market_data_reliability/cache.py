"""In-process snapshot cache with TTL-based expiration and freshness recomputation.

Dict-based cache keyed by CacheKey (symbol, data_type, provider_policy, market_session).
On cache hit, age_seconds and freshness_state are recomputed from elapsed time since
the snapshot was stored. Expired entries are marked stale rather than served as fresh.

Requirements: 5.1, 5.3, 5.5
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace
from typing import Optional

from utils.market_data_reliability.freshness import FreshnessClassifier
from utils.market_data_reliability.snapshot import CacheKey, Snapshot

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    """Internal mutable entry wrapping a cached snapshot with store metadata."""

    snapshot: Snapshot
    stored_at: float  # monotonic time when put() was called
    ttl: float  # TTL in seconds for this entry's data_type


class SnapshotCache:
    """In-process dict-based cache for Snapshot instances.

    Thread-safe via threading.Lock (brief holds only for dict operations).
    TTLs are configurable per data_type via the cache_ttls dict from ReliabilityConfig.

    On cache hit:
        - Preserves original fetched_at and source_timestamp
        - Recomputes age_seconds = original_age + elapsed_since_store
        - Recomputes freshness_state from the new age_seconds
        - Expired entries (past TTL) are returned with freshness_state reflecting
          their true age (stale) rather than being served as fresh
    """

    def __init__(
        self,
        cache_ttls: dict[str, float],
        freshness_classifier: FreshnessClassifier,
    ) -> None:
        """Initialize the cache.

        Args:
            cache_ttls: Mapping of data_type -> TTL in seconds.
                Default TTLs: quote=15, candle=60, atr=120, volume=30, previous_close=3600.
            freshness_classifier: Used to recompute freshness_state on cache hits.
        """
        self._cache_ttls = cache_ttls
        self._freshness_classifier = freshness_classifier
        self._store: dict[CacheKey, _CacheEntry] = {}
        self._lock = threading.Lock()

    def get(self, key: CacheKey) -> Optional[Snapshot]:
        """Retrieve a cached snapshot, recomputing age and freshness.

        If the key exists in cache:
            1. Compute elapsed time since store
            2. Check if entry has exceeded its TTL
            3. Recompute age_seconds = original_age + elapsed_time
            4. Recompute freshness_state from new age_seconds
            5. Return updated snapshot (preserving original fetched_at, source_timestamp)

        Expired entries are still returned but with updated freshness_state
        reflecting their true staleness. The caller can inspect freshness_state
        to decide whether to re-fetch.

        Args:
            key: CacheKey(symbol, data_type, provider_policy, market_session).

        Returns:
            Updated Snapshot with recomputed age/freshness, or None if not cached.
        """
        now = time.monotonic()

        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            # Copy values under lock to avoid races
            snapshot = entry.snapshot
            stored_at = entry.stored_at

        # Recompute outside lock (classification is pure computation)
        elapsed = now - stored_at
        new_age_seconds = snapshot.age_seconds + elapsed

        new_freshness_state = self._freshness_classifier.classify(
            age_seconds=new_age_seconds,
            data_type=key.data_type,
            consumer="execution",  # Use strict thresholds for cache classification
            market_session=key.market_session,
        )

        # Snapshot is frozen — create a new instance with updated fields
        updated_snapshot = replace(
            snapshot,
            age_seconds=new_age_seconds,
            freshness_state=new_freshness_state,
        )

        logger.debug(
            "Cache hit for %s/%s: age_seconds=%.1f -> %.1f, freshness=%s -> %s",
            key.symbol, key.data_type,
            snapshot.age_seconds, new_age_seconds,
            snapshot.freshness_state, new_freshness_state,
        )

        return updated_snapshot

    def put(self, key: CacheKey, snapshot: Snapshot) -> None:
        """Store a snapshot in the cache.

        Records the current monotonic time as the store time for TTL
        and age recomputation purposes.

        Args:
            key: CacheKey(symbol, data_type, provider_policy, market_session).
            snapshot: The Snapshot to cache.
        """
        ttl = self._cache_ttls.get(key.data_type, 15.0)  # Default 15s if unknown
        entry = _CacheEntry(
            snapshot=snapshot,
            stored_at=time.monotonic(),
            ttl=ttl,
        )

        with self._lock:
            self._store[key] = entry

        logger.debug(
            "Cache put for %s/%s (ttl=%.1fs)",
            key.symbol, key.data_type, ttl,
        )

    def invalidate_expired(self) -> int:
        """Remove all entries that have exceeded their TTL.

        Returns:
            Count of entries removed.
        """
        now = time.monotonic()
        expired_keys: list[CacheKey] = []

        with self._lock:
            for key, entry in self._store.items():
                elapsed = now - entry.stored_at
                if elapsed >= entry.ttl:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._store[key]

        if expired_keys:
            logger.debug(
                "Cache invalidated %d expired entries.", len(expired_keys),
            )

        return len(expired_keys)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            count = len(self._store)
            self._store.clear()

        logger.debug("Cache cleared (%d entries removed).", count)
