"""Configuration for the Market Data Reliability Layer.

Reads all thresholds from environment variables with safe defaults that
preserve fail-closed behavior for PM and execution consumers. Missing or
invalid environment variables always fall back to strict defaults.

Environment variable naming convention:
    MDR_<SECTION>_<KEY> for reliability-specific settings
    MARKET_DATA_RELIABILITY_MODE for the feature flag
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from utils.market_data_reliability.snapshot import FreshnessThreshold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default freshness thresholds (fail-closed for execution consumers)
# ---------------------------------------------------------------------------

_DEFAULT_FRESHNESS_THRESHOLDS: dict[tuple[str, str], FreshnessThreshold] = {
    ("quote", "execution"): FreshnessThreshold(fresh_threshold=30.0, aging_threshold=120.0),
    ("quote", "display"): FreshnessThreshold(fresh_threshold=60.0, aging_threshold=300.0),
    ("candle", "execution"): FreshnessThreshold(fresh_threshold=120.0, aging_threshold=600.0),
    ("candle", "display"): FreshnessThreshold(fresh_threshold=300.0, aging_threshold=900.0),
    ("atr", "execution"): FreshnessThreshold(fresh_threshold=300.0, aging_threshold=900.0),
    ("atr", "display"): FreshnessThreshold(fresh_threshold=600.0, aging_threshold=1800.0),
    ("volume", "execution"): FreshnessThreshold(fresh_threshold=60.0, aging_threshold=300.0),
    ("previous_close", "all"): FreshnessThreshold(fresh_threshold=3600.0, aging_threshold=7200.0),
}

# ---------------------------------------------------------------------------
# Default cache TTLs (seconds)
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_TTLS: dict[str, float] = {
    "quote": 15.0,
    "candle": 60.0,
    "atr": 120.0,
    "volume": 30.0,
    "previous_close": 3600.0,
}

# ---------------------------------------------------------------------------
# Default provider timeouts (seconds)
# ---------------------------------------------------------------------------

_DEFAULT_PROVIDER_TIMEOUTS: dict[str, float] = {
    "finnhub": 10.0,
    "yfinance": 15.0,
    "alpaca": 10.0,
}

# ---------------------------------------------------------------------------
# Default provider retry limits
# ---------------------------------------------------------------------------

_DEFAULT_PROVIDER_RETRY_LIMITS: dict[str, int] = {
    "finnhub": 2,
    "yfinance": 2,
    "alpaca": 2,
}

# ---------------------------------------------------------------------------
# Default backoff durations (seconds)
# ---------------------------------------------------------------------------

_DEFAULT_BACKOFF_DURATIONS: dict[str, float] = {
    "rate_limit": 60.0,
    "network_error": 30.0,
    "empty_response": 15.0,
}

# ---------------------------------------------------------------------------
# Default fallback matrix: (data_type, consumer_category) -> ordered providers
# ---------------------------------------------------------------------------

_DEFAULT_FALLBACK_MATRIX: dict[tuple[str, str], list[str]] = {
    ("quote", "execution"): ["finnhub", "yfinance", "alpaca"],
    ("quote", "display"): ["finnhub", "yfinance", "alpaca"],
    ("quote", "monitoring"): ["finnhub", "yfinance", "alpaca"],
    ("candle", "execution"): ["finnhub", "yfinance"],
    ("candle", "display"): ["finnhub", "yfinance"],
    ("atr", "execution"): ["finnhub", "yfinance"],
    ("atr", "display"): ["finnhub", "yfinance"],
    ("volume", "execution"): ["finnhub", "yfinance"],
    ("volume", "display"): ["finnhub", "yfinance"],
    ("previous_close", "all"): ["finnhub", "yfinance", "alpaca"],
}


# ---------------------------------------------------------------------------
# Helper: safe float/int parsing with default
# ---------------------------------------------------------------------------


def _safe_float(value: str | None, default: float, name: str) -> float:
    """Parse a string as float, returning default on failure.

    Rejects non-positive values, NaN, and infinity to preserve safe defaults.
    """
    import math

    if value is None:
        return default
    try:
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            logger.warning(
                "MDR config %s has non-finite value '%s'; using default %.1f",
                name, value, default,
            )
            return default
        if parsed <= 0:
            logger.warning(
                "MDR config %s has non-positive value '%s'; using default %.1f",
                name, value, default,
            )
            return default
        return parsed
    except (ValueError, TypeError):
        logger.warning(
            "MDR config %s has invalid value '%s'; using default %.1f",
            name, value, default,
        )
        return default


def _safe_int(value: str | None, default: int, name: str) -> int:
    """Parse a string as int, returning default on failure."""
    if value is None:
        return default
    try:
        parsed = int(value)
        if parsed < 0:
            logger.warning(
                "MDR config %s has negative value '%s'; using default %d",
                name, value, default,
            )
            return default
        return parsed
    except (ValueError, TypeError):
        logger.warning(
            "MDR config %s has invalid value '%s'; using default %d",
            name, value, default,
        )
        return default


# ---------------------------------------------------------------------------
# ReliabilityConfig frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReliabilityConfig:
    """Immutable configuration for the Market Data Reliability Layer.

    All values are loaded from environment variables via from_environment().
    Missing or invalid values produce safe defaults that preserve fail-closed
    behavior for execution consumers (PM, Risk Geometry Gate).
    """

    mode: str  # "disabled", "observe", "enforcing"
    freshness_thresholds: dict[tuple[str, str], FreshnessThreshold]
    cache_ttls: dict[str, float]
    provider_timeouts: dict[str, float]
    provider_retry_limits: dict[str, int]
    backoff_durations: dict[str, float]
    fallback_matrix: dict[tuple[str, str], list[str]]

    @classmethod
    def from_environment(cls) -> ReliabilityConfig:
        """Load configuration from environment variables with safe defaults.

        Environment variables read:
            MARKET_DATA_RELIABILITY_MODE - Feature flag (disabled/observe/enforcing)
            MDR_FRESHNESS_<DATA_TYPE>_<CONSUMER>_FRESH - Fresh threshold seconds
            MDR_FRESHNESS_<DATA_TYPE>_<CONSUMER>_AGING - Aging threshold seconds
            MDR_CACHE_TTL_<DATA_TYPE> - Cache TTL seconds
            MDR_PROVIDER_TIMEOUT_<PROVIDER> - Provider timeout seconds
            MDR_PROVIDER_RETRIES_<PROVIDER> - Provider retry limit
            MDR_BACKOFF_<FAILURE_TYPE> - Backoff duration seconds
            MDR_FALLBACK_<DATA_TYPE>_<CONSUMER> - Comma-separated provider list

        Missing or invalid values always produce safe strict defaults.
        """
        mode = cls._load_mode()
        freshness_thresholds = cls._load_freshness_thresholds()
        cache_ttls = cls._load_cache_ttls()
        provider_timeouts = cls._load_provider_timeouts()
        provider_retry_limits = cls._load_provider_retry_limits()
        backoff_durations = cls._load_backoff_durations()
        fallback_matrix = cls._load_fallback_matrix()

        return cls(
            mode=mode,
            freshness_thresholds=freshness_thresholds,
            cache_ttls=cache_ttls,
            provider_timeouts=provider_timeouts,
            provider_retry_limits=provider_retry_limits,
            backoff_durations=backoff_durations,
            fallback_matrix=fallback_matrix,
        )

    @classmethod
    def _load_mode(cls) -> str:
        """Load MARKET_DATA_RELIABILITY_MODE, defaulting to 'disabled'."""
        raw = os.environ.get("MARKET_DATA_RELIABILITY_MODE", "disabled")
        valid_modes = ("disabled", "observe", "enforcing")
        if raw not in valid_modes:
            logger.warning(
                "MARKET_DATA_RELIABILITY_MODE has unrecognized value '%s'; "
                "defaulting to 'disabled'.",
                raw,
            )
            return "disabled"
        return raw

    @classmethod
    def _load_freshness_thresholds(cls) -> dict[tuple[str, str], FreshnessThreshold]:
        """Load freshness thresholds from MDR_FRESHNESS_* env vars.

        Env var pattern:
            MDR_FRESHNESS_QUOTE_EXECUTION_FRESH=30
            MDR_FRESHNESS_QUOTE_EXECUTION_AGING=120
        """
        thresholds: dict[tuple[str, str], FreshnessThreshold] = {}

        for (data_type, consumer), default_threshold in _DEFAULT_FRESHNESS_THRESHOLDS.items():
            prefix = f"MDR_FRESHNESS_{data_type.upper()}_{consumer.upper()}"

            fresh_val = _safe_float(
                os.environ.get(f"{prefix}_FRESH"),
                default_threshold.fresh_threshold,
                f"{prefix}_FRESH",
            )
            aging_val = _safe_float(
                os.environ.get(f"{prefix}_AGING"),
                default_threshold.aging_threshold,
                f"{prefix}_AGING",
            )

            # Ensure aging >= fresh (invariant: fresh < aging boundary)
            if aging_val < fresh_val:
                logger.warning(
                    "MDR config %s_AGING (%.1f) < %s_FRESH (%.1f); "
                    "using defaults to preserve fail-closed behavior.",
                    prefix, aging_val, prefix, fresh_val,
                )
                fresh_val = default_threshold.fresh_threshold
                aging_val = default_threshold.aging_threshold

            thresholds[(data_type, consumer)] = FreshnessThreshold(
                fresh_threshold=fresh_val,
                aging_threshold=aging_val,
            )

        return thresholds

    @classmethod
    def _load_cache_ttls(cls) -> dict[str, float]:
        """Load cache TTLs from MDR_CACHE_TTL_* env vars.

        Env var pattern: MDR_CACHE_TTL_QUOTE=15
        """
        ttls: dict[str, float] = {}
        for data_type, default_ttl in _DEFAULT_CACHE_TTLS.items():
            env_name = f"MDR_CACHE_TTL_{data_type.upper()}"
            ttls[data_type] = _safe_float(
                os.environ.get(env_name),
                default_ttl,
                env_name,
            )
        return ttls

    @classmethod
    def _load_provider_timeouts(cls) -> dict[str, float]:
        """Load provider timeouts from MDR_PROVIDER_TIMEOUT_* env vars.

        Env var pattern: MDR_PROVIDER_TIMEOUT_FINNHUB=10
        """
        timeouts: dict[str, float] = {}
        for provider, default_timeout in _DEFAULT_PROVIDER_TIMEOUTS.items():
            env_name = f"MDR_PROVIDER_TIMEOUT_{provider.upper()}"
            timeouts[provider] = _safe_float(
                os.environ.get(env_name),
                default_timeout,
                env_name,
            )
        return timeouts

    @classmethod
    def _load_provider_retry_limits(cls) -> dict[str, int]:
        """Load provider retry limits from MDR_PROVIDER_RETRIES_* env vars.

        Env var pattern: MDR_PROVIDER_RETRIES_FINNHUB=2
        """
        retries: dict[str, int] = {}
        for provider, default_retries in _DEFAULT_PROVIDER_RETRY_LIMITS.items():
            env_name = f"MDR_PROVIDER_RETRIES_{provider.upper()}"
            retries[provider] = _safe_int(
                os.environ.get(env_name),
                default_retries,
                env_name,
            )
        return retries

    @classmethod
    def _load_backoff_durations(cls) -> dict[str, float]:
        """Load backoff durations from MDR_BACKOFF_* env vars.

        Env var pattern: MDR_BACKOFF_RATE_LIMIT=60
        """
        durations: dict[str, float] = {}
        for failure_type, default_duration in _DEFAULT_BACKOFF_DURATIONS.items():
            env_name = f"MDR_BACKOFF_{failure_type.upper()}"
            durations[failure_type] = _safe_float(
                os.environ.get(env_name),
                default_duration,
                env_name,
            )
        return durations

    @classmethod
    def _load_fallback_matrix(cls) -> dict[tuple[str, str], list[str]]:
        """Load fallback matrix from MDR_FALLBACK_* env vars.

        Env var pattern: MDR_FALLBACK_QUOTE_EXECUTION=finnhub,yfinance,alpaca

        Empty or whitespace-only values are treated as missing (use default).
        """
        matrix: dict[tuple[str, str], list[str]] = {}

        for (data_type, consumer), default_providers in _DEFAULT_FALLBACK_MATRIX.items():
            env_name = f"MDR_FALLBACK_{data_type.upper()}_{consumer.upper()}"
            raw = os.environ.get(env_name)

            if raw is None or raw.strip() == "":
                matrix[(data_type, consumer)] = list(default_providers)
                continue

            providers = [p.strip() for p in raw.split(",") if p.strip()]
            if not providers:
                logger.warning(
                    "MDR config %s has no valid providers after parsing '%s'; "
                    "using default fallback list.",
                    env_name, raw,
                )
                matrix[(data_type, consumer)] = list(default_providers)
            else:
                matrix[(data_type, consumer)] = providers

        return matrix
