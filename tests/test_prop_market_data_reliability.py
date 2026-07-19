"""
Property-based tests for Market Data Reliability Layer.

Tests correctness properties from the design document using Hypothesis.
Feature: market-data-reliability-layer
"""

from __future__ import annotations

import os
from unittest.mock import patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from utils.market_data_reliability.config import ReliabilityConfig
from utils.market_data_reliability.snapshot import FreshnessThreshold


# ---------------------------------------------------------------------------
# Hypothesis Strategies for Property 12
# ---------------------------------------------------------------------------

# Strategy for invalid env var values: empty strings, non-numeric, negative numbers,
# whitespace-only, special characters
st_invalid_value = st.one_of(
    st.just(""),
    st.just("   "),
    st.just("abc"),
    st.just("not_a_number"),
    st.just("-1"),
    st.just("-999.5"),
    st.just("0"),
    st.just("0.0"),
    st.just("NaN"),
    st.just("inf"),
    st.just("true"),
    st.just("null"),
    st.just("!@#$"),
)

# Strategy for env var key presence: either a key is missing (None) or has an invalid value
st_env_entry = st.one_of(
    st.none(),  # Key not present in env
    st_invalid_value,  # Key present with invalid value
)

# All MDR env var keys that affect config
MDR_FRESHNESS_KEYS = [
    "MDR_FRESHNESS_QUOTE_EXECUTION_FRESH",
    "MDR_FRESHNESS_QUOTE_EXECUTION_AGING",
    "MDR_FRESHNESS_QUOTE_DISPLAY_FRESH",
    "MDR_FRESHNESS_QUOTE_DISPLAY_AGING",
    "MDR_FRESHNESS_CANDLE_EXECUTION_FRESH",
    "MDR_FRESHNESS_CANDLE_EXECUTION_AGING",
    "MDR_FRESHNESS_CANDLE_DISPLAY_FRESH",
    "MDR_FRESHNESS_CANDLE_DISPLAY_AGING",
    "MDR_FRESHNESS_ATR_EXECUTION_FRESH",
    "MDR_FRESHNESS_ATR_EXECUTION_AGING",
    "MDR_FRESHNESS_ATR_DISPLAY_FRESH",
    "MDR_FRESHNESS_ATR_DISPLAY_AGING",
    "MDR_FRESHNESS_VOLUME_EXECUTION_FRESH",
    "MDR_FRESHNESS_VOLUME_EXECUTION_AGING",
    "MDR_FRESHNESS_PREVIOUS_CLOSE_ALL_FRESH",
    "MDR_FRESHNESS_PREVIOUS_CLOSE_ALL_AGING",
]

MDR_CACHE_KEYS = [
    "MDR_CACHE_TTL_QUOTE",
    "MDR_CACHE_TTL_CANDLE",
    "MDR_CACHE_TTL_ATR",
    "MDR_CACHE_TTL_VOLUME",
    "MDR_CACHE_TTL_PREVIOUS_CLOSE",
]

MDR_PROVIDER_TIMEOUT_KEYS = [
    "MDR_PROVIDER_TIMEOUT_FINNHUB",
    "MDR_PROVIDER_TIMEOUT_YFINANCE",
    "MDR_PROVIDER_TIMEOUT_ALPACA",
]

MDR_PROVIDER_RETRY_KEYS = [
    "MDR_PROVIDER_RETRIES_FINNHUB",
    "MDR_PROVIDER_RETRIES_YFINANCE",
    "MDR_PROVIDER_RETRIES_ALPACA",
]

MDR_BACKOFF_KEYS = [
    "MDR_BACKOFF_RATE_LIMIT",
    "MDR_BACKOFF_NETWORK_ERROR",
    "MDR_BACKOFF_EMPTY_RESPONSE",
]

MDR_FALLBACK_KEYS = [
    "MDR_FALLBACK_QUOTE_EXECUTION",
    "MDR_FALLBACK_QUOTE_DISPLAY",
    "MDR_FALLBACK_QUOTE_MONITORING",
    "MDR_FALLBACK_CANDLE_EXECUTION",
    "MDR_FALLBACK_CANDLE_DISPLAY",
    "MDR_FALLBACK_ATR_EXECUTION",
    "MDR_FALLBACK_ATR_DISPLAY",
    "MDR_FALLBACK_VOLUME_EXECUTION",
    "MDR_FALLBACK_VOLUME_DISPLAY",
    "MDR_FALLBACK_PREVIOUS_CLOSE_ALL",
]

MDR_MODE_KEY = "MARKET_DATA_RELIABILITY_MODE"

ALL_MDR_KEYS = (
    [MDR_MODE_KEY]
    + MDR_FRESHNESS_KEYS
    + MDR_CACHE_KEYS
    + MDR_PROVIDER_TIMEOUT_KEYS
    + MDR_PROVIDER_RETRY_KEYS
    + MDR_BACKOFF_KEYS
    + MDR_FALLBACK_KEYS
)

# Strategy: generate a dict of env vars where each key is either missing or has invalid value
st_invalid_env = st.fixed_dictionaries(
    {key: st_env_entry for key in ALL_MDR_KEYS}
)


# ---------------------------------------------------------------------------
# Property 12: Configuration Safe Defaults Preserve Fail-Closed
# Feature: market-data-reliability-layer, Property 12: Configuration Safe Defaults Preserve Fail-Closed
# ---------------------------------------------------------------------------


class TestProperty12ConfigSafeDefaultsPreserveFailClosed:
    """
    For any missing or invalid environment variable for freshness thresholds,
    cache TTLs, or provider settings, the resulting ReliabilityConfig SHALL
    produce defaults that cause execution consumers to fail-closed (i.e.,
    defaults are strict, not permissive).

    **Validates: Requirements 12.4, 12.5**
    """

    @given(env_overrides=st_invalid_env)
    @settings(max_examples=200)
    def test_config_defaults_to_disabled_mode_on_invalid_input(self, env_overrides):
        """Mode defaults to 'disabled' when env var is missing or invalid."""
        # Build env dict: only include keys that have non-None values
        env = {k: v for k, v in env_overrides.items() if v is not None}

        # Clear all MDR keys and set only the generated ones
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("MDR_") and k != MDR_MODE_KEY}
        clean_env.update(env)

        with patch.dict(os.environ, clean_env, clear=True):
            config = ReliabilityConfig.from_environment()

        # Mode must be one of the valid values
        assert config.mode in ("disabled", "observe", "enforcing")
        # If mode env var was invalid (not one of the valid modes), defaults to disabled
        mode_value = env_overrides.get(MDR_MODE_KEY)
        if mode_value is None or mode_value not in ("disabled", "observe", "enforcing"):
            assert config.mode == "disabled"

    @given(env_overrides=st_invalid_env)
    @settings(max_examples=200)
    def test_execution_freshness_thresholds_are_strict(self, env_overrides):
        """Execution freshness thresholds remain strict (quote execution: fresh<=30s, aging<=120s)."""
        env = {k: v for k, v in env_overrides.items() if v is not None}

        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("MDR_") and k != MDR_MODE_KEY}
        clean_env.update(env)

        with patch.dict(os.environ, clean_env, clear=True):
            config = ReliabilityConfig.from_environment()

        # Quote execution: fresh <= 30s, aging <= 120s (strict defaults)
        quote_exec = config.freshness_thresholds[("quote", "execution")]
        assert quote_exec.fresh_threshold <= 30.0
        assert quote_exec.aging_threshold <= 120.0

        # Candle execution: fresh <= 120s, aging <= 600s
        candle_exec = config.freshness_thresholds[("candle", "execution")]
        assert candle_exec.fresh_threshold <= 120.0
        assert candle_exec.aging_threshold <= 600.0

    @given(env_overrides=st_invalid_env)
    @settings(max_examples=200)
    def test_cache_ttls_are_positive(self, env_overrides):
        """All cache TTLs must be positive values."""
        env = {k: v for k, v in env_overrides.items() if v is not None}

        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("MDR_") and k != MDR_MODE_KEY}
        clean_env.update(env)

        with patch.dict(os.environ, clean_env, clear=True):
            config = ReliabilityConfig.from_environment()

        for data_type, ttl in config.cache_ttls.items():
            assert ttl > 0, f"Cache TTL for {data_type} must be positive, got {ttl}"

    @given(env_overrides=st_invalid_env)
    @settings(max_examples=200)
    def test_provider_timeouts_are_positive(self, env_overrides):
        """All provider timeouts must be positive values."""
        env = {k: v for k, v in env_overrides.items() if v is not None}

        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("MDR_") and k != MDR_MODE_KEY}
        clean_env.update(env)

        with patch.dict(os.environ, clean_env, clear=True):
            config = ReliabilityConfig.from_environment()

        for provider, timeout in config.provider_timeouts.items():
            assert timeout > 0, f"Timeout for {provider} must be positive, got {timeout}"

    @given(env_overrides=st_invalid_env)
    @settings(max_examples=200)
    def test_backoff_durations_are_positive(self, env_overrides):
        """All backoff durations must be positive values."""
        env = {k: v for k, v in env_overrides.items() if v is not None}

        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("MDR_") and k != MDR_MODE_KEY}
        clean_env.update(env)

        with patch.dict(os.environ, clean_env, clear=True):
            config = ReliabilityConfig.from_environment()

        for failure_type, duration in config.backoff_durations.items():
            assert duration > 0, f"Backoff duration for {failure_type} must be positive, got {duration}"

    @given(env_overrides=st_invalid_env)
    @settings(max_examples=200)
    def test_fallback_matrix_has_execution_entries(self, env_overrides):
        """Fallback matrix has entries for execution consumers (fail-closed requires known fallback chain)."""
        env = {k: v for k, v in env_overrides.items() if v is not None}

        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("MDR_") and k != MDR_MODE_KEY}
        clean_env.update(env)

        with patch.dict(os.environ, clean_env, clear=True):
            config = ReliabilityConfig.from_environment()

        # Execution consumers must have fallback entries for key data types
        assert ("quote", "execution") in config.fallback_matrix
        assert len(config.fallback_matrix[("quote", "execution")]) > 0

        assert ("candle", "execution") in config.fallback_matrix
        assert len(config.fallback_matrix[("candle", "execution")]) > 0

        assert ("atr", "execution") in config.fallback_matrix
        assert len(config.fallback_matrix[("atr", "execution")]) > 0

        assert ("volume", "execution") in config.fallback_matrix
        assert len(config.fallback_matrix[("volume", "execution")]) > 0


# ---------------------------------------------------------------------------
# Strategies for Property 6
# ---------------------------------------------------------------------------

import time

from utils.market_data_reliability.validator import ResponseValidator

# Symbol keys recognized by the validator
_PROP6_SYMBOL_KEYS = ("s", "symbol", "ticker")

# Price keys recognized by the validator
_PROP6_PRICE_KEYS = ("c", "current_price", "last_price", "price", "close", "l", "o", "h", "pc")

# Data types where price validation applies
_PROP6_PRICE_DATA_TYPES = ("quote", "candle", "previous_close")

# Strategy: generate valid ticker symbols (uppercase, 1-5 chars)
st_symbol = st.text(
    alphabet=st.characters(whitelist_categories=("Lu",), whitelist_characters=""),
    min_size=1,
    max_size=5,
)

# Strategy: generate non-positive prices (zero and negative floats)
st_non_positive_price = st.one_of(
    st.just(0.0),
    st.just(0),
    st.floats(max_value=-0.01, min_value=-10000.0, allow_nan=False, allow_infinity=False),
)

# Strategy: symbol key to use in response dict
st_symbol_key = st.sampled_from(list(_PROP6_SYMBOL_KEYS))

# Strategy: price key to use in response dict
st_price_key = st.sampled_from(list(_PROP6_PRICE_KEYS))

# Strategy: data type for price checks
st_price_data_type = st.sampled_from(list(_PROP6_PRICE_DATA_TYPES))


# ---------------------------------------------------------------------------
# Property 6: Validation Rejection Produces Correct Reason Code
# Feature: market-data-reliability-layer, Property 6: Validation Rejection Produces Correct Reason Code
# ---------------------------------------------------------------------------


class TestProperty6ValidationRejectionProducesCorrectReasonCode:
    """
    For any provider response where the returned symbol differs from the
    requested symbol, validation SHALL produce a degradation_reasons containing
    "cross_symbol_response". For any response with a non-positive price for an
    equity/ETF, validation SHALL produce "invalid_price".

    **Validates: Requirements 4.1, 4.2**
    """

    @given(
        requested_symbol=st_symbol,
        response_symbol=st_symbol,
        symbol_key=st_symbol_key,
    )
    @settings(max_examples=200)
    def test_cross_symbol_response_produces_correct_reason(
        self, requested_symbol, response_symbol, symbol_key
    ):
        """Cross-symbol mismatch produces 'cross_symbol_response' degradation reason."""
        # Ensure the two symbols are actually different (case-insensitive)
        assume(requested_symbol.strip().upper() != response_symbol.strip().upper())

        # Build response with a valid timestamp to avoid missing_source_timestamp noise
        raw = {
            symbol_key: response_symbol,
            "c": 150.0,  # Valid price to isolate cross-symbol check
            "t": time.time(),
        }

        validator = ResponseValidator(staleness_threshold_seconds=300.0)
        result = validator.validate(raw, symbol=requested_symbol, data_type="quote")

        assert not result.is_valid, (
            f"Expected is_valid=False for cross-symbol mismatch: "
            f"requested={requested_symbol!r}, got={response_symbol!r}"
        )
        assert "cross_symbol_response" in result.degradation_reasons, (
            f"Expected 'cross_symbol_response' in degradation_reasons, "
            f"got {result.degradation_reasons}"
        )

    @given(
        symbol=st_symbol,
        price=st_non_positive_price,
        price_key=st_price_key,
        data_type=st_price_data_type,
    )
    @settings(max_examples=200)
    def test_invalid_price_produces_correct_reason(
        self, symbol, price, price_key, data_type
    ):
        """Non-positive price for equity/ETF produces 'invalid_price' degradation reason."""
        # Build response with matching symbol and valid timestamp
        raw = {
            "s": symbol,
            price_key: price,
            "t": time.time(),
        }

        validator = ResponseValidator(staleness_threshold_seconds=300.0)
        result = validator.validate(raw, symbol=symbol, data_type=data_type)

        assert not result.is_valid, (
            f"Expected is_valid=False for non-positive price: "
            f"price_key={price_key!r}, price={price!r}, data_type={data_type!r}"
        )
        assert "invalid_price" in result.degradation_reasons, (
            f"Expected 'invalid_price' in degradation_reasons, "
            f"got {result.degradation_reasons}"
        )


# ---------------------------------------------------------------------------
# Strategies for Property 4
# ---------------------------------------------------------------------------

_VALID_DATA_TYPES = ["quote", "candle", "atr", "volume", "previous_close"]
_VALID_CONSUMERS = [
    "PM", "Risk_Geometry_Gate", "Dashboard_API", "Analyst",
    "Reviewer", "CEO_Output", "Price_Monitor", "Alert_Dispatcher",
]
_VALID_MARKET_SESSIONS = ["open", "pre_market", "after_hours", "closed"]

_VALID_FRESHNESS_STATES = {"fresh", "aging", "stale", "unavailable", "market_closed"}

# Include unknown data types and consumers to test fail-closed defaults
st_data_type = st.one_of(
    st.sampled_from(_VALID_DATA_TYPES),
    st.just("unknown_type"),
    st.just("option_chain"),
)

st_consumer = st.one_of(
    st.sampled_from(_VALID_CONSUMERS),
    st.just("unknown_consumer"),
    st.just("SomeNewService"),
)

st_market_session = st.sampled_from(_VALID_MARKET_SESSIONS)

# age_seconds: include negative, zero, small, large, and edge values
st_age_seconds = st.one_of(
    st.floats(min_value=-1000.0, max_value=-0.01),  # negative
    st.just(0.0),  # zero
    st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False),  # positive range
    st.just(float("inf")),  # infinity (very stale)
)


# ---------------------------------------------------------------------------
# Property 4: Freshness State Mutual Exclusivity
# Feature: market-data-reliability-layer, Property 4: Freshness State Mutual Exclusivity
# ---------------------------------------------------------------------------


class TestProperty4FreshnessStateMutualExclusivity:
    """
    For any age_seconds value, data_type, consumer, and market_session combination,
    the FreshnessClassifier SHALL assign exactly one freshness_state from the valid
    enum set {fresh, aging, stale, unavailable, market_closed}.

    **Validates: Requirements 2.1**
    """

    @given(
        age_seconds=st_age_seconds,
        data_type=st_data_type,
        consumer=st_consumer,
        market_session=st_market_session,
    )
    @settings(max_examples=200)
    def test_freshness_classifier_returns_exactly_one_valid_state(
        self, age_seconds, data_type, consumer, market_session
    ):
        """FreshnessClassifier always returns exactly one state from the valid enum."""
        from utils.market_data_reliability.freshness import FreshnessClassifier
        from utils.market_data_reliability.config import _DEFAULT_FRESHNESS_THRESHOLDS

        classifier = FreshnessClassifier(_DEFAULT_FRESHNESS_THRESHOLDS)

        result = classifier.classify(
            age_seconds=age_seconds,
            data_type=data_type,
            consumer=consumer,
            market_session=market_session,
        )

        # Result must be exactly one of the valid freshness states
        assert result in _VALID_FRESHNESS_STATES, (
            f"FreshnessClassifier returned '{result}' which is not in "
            f"valid states {_VALID_FRESHNESS_STATES} for inputs: "
            f"age_seconds={age_seconds}, data_type={data_type}, "
            f"consumer={consumer}, market_session={market_session}"
        )


# ---------------------------------------------------------------------------
# Hypothesis Strategies for Property 5
# ---------------------------------------------------------------------------

# All known degradation reason codes from the domain
ALL_DEGRADATION_REASONS = [
    "cross_symbol_response",
    "invalid_price",
    "all_providers_failed",
    "missing_source_timestamp",
    "stale_source_timestamp",
    "provider_error",
    "rate_limited",
    "empty_response",
    "malformed_json",
    "network_timeout",
]

# Strategy for generating a ValidationResult with various combinations
st_degradation_reasons = st.lists(
    st.sampled_from(ALL_DEGRADATION_REASONS),
    min_size=0,
    max_size=5,
    unique=True,
).map(tuple)

st_validation_result = st.builds(
    lambda reasons: __import__(
        "utils.market_data_reliability.snapshot", fromlist=["ValidationResult"]
    ).ValidationResult(
        is_valid=len(reasons) == 0,
        degradation_reasons=reasons,
    ),
    reasons=st_degradation_reasons,
)

# Strategy for freshness_state values
st_freshness_state = st.sampled_from([
    "fresh", "aging", "stale", "unavailable", "market_closed",
])

# Strategy for consumer names (all categories)
st_consumer = st.sampled_from([
    "PM", "Risk_Geometry_Gate",  # execution
    "Dashboard_API", "Analyst", "Reviewer", "CEO_Output",  # display
    "Price_Monitor", "Alert_Dispatcher",  # monitoring
])

# Strategy for market session values
st_market_session = st.sampled_from([
    "open", "pre_market", "after_hours", "closed",
])


# ---------------------------------------------------------------------------
# Property 5: Trust State Implies Degradation Reasons
# Feature: market-data-reliability-layer, Property 5: Trust State Implies Degradation Reasons
# ---------------------------------------------------------------------------


class TestProperty5TrustStateImpliesDegradationReasons:
    """
    For any Snapshot with trust_state of "untrusted", the degradation_reasons
    tuple SHALL contain at least one reason code. Conversely, for any Snapshot
    with trust_state of "trusted", degradation_reasons SHALL be empty.

    **Validates: Requirements 3.1, 3.3**
    """

    @given(
        validation_result=st_validation_result,
        freshness_state=st_freshness_state,
        consumer=st_consumer,
        market_session=st_market_session,
    )
    @settings(max_examples=200)
    def test_untrusted_has_at_least_one_degradation_reason(
        self, validation_result, freshness_state, consumer, market_session
    ):
        """If trust_state is untrusted, degradation_reasons must have >= 1 entry."""
        from utils.market_data_reliability.trust import TrustClassifier

        classifier = TrustClassifier()
        trust_state, degradation_reasons = classifier.classify(
            validation_result=validation_result,
            freshness_state=freshness_state,
            consumer=consumer,
            market_session=market_session,
        )

        if trust_state == "untrusted":
            assert len(degradation_reasons) >= 1, (
                f"untrusted state must have at least one degradation reason, "
                f"got empty reasons with validation_result={validation_result}, "
                f"freshness_state={freshness_state}, consumer={consumer}, "
                f"market_session={market_session}"
            )

    @given(
        validation_result=st_validation_result,
        freshness_state=st_freshness_state,
        consumer=st_consumer,
        market_session=st_market_session,
    )
    @settings(max_examples=200)
    def test_trusted_has_empty_degradation_reasons(
        self, validation_result, freshness_state, consumer, market_session
    ):
        """If trust_state is trusted, degradation_reasons must be empty."""
        from utils.market_data_reliability.trust import TrustClassifier

        classifier = TrustClassifier()
        trust_state, degradation_reasons = classifier.classify(
            validation_result=validation_result,
            freshness_state=freshness_state,
            consumer=consumer,
            market_session=market_session,
        )

        if trust_state == "trusted":
            assert degradation_reasons == (), (
                f"trusted state must have empty degradation_reasons, "
                f"got {degradation_reasons} with validation_result={validation_result}, "
                f"freshness_state={freshness_state}, consumer={consumer}, "
                f"market_session={market_session}"
            )

    @given(
        validation_result=st_validation_result,
        freshness_state=st_freshness_state,
        consumer=st_consumer,
        market_session=st_market_session,
    )
    @settings(max_examples=200)
    def test_degraded_has_at_least_one_degradation_reason(
        self, validation_result, freshness_state, consumer, market_session
    ):
        """If trust_state is degraded, degradation_reasons must have >= 1 entry."""
        from utils.market_data_reliability.trust import TrustClassifier

        classifier = TrustClassifier()
        trust_state, degradation_reasons = classifier.classify(
            validation_result=validation_result,
            freshness_state=freshness_state,
            consumer=consumer,
            market_session=market_session,
        )

        if trust_state == "degraded":
            assert len(degradation_reasons) >= 1, (
                f"degraded state must have at least one degradation reason, "
                f"got empty reasons with validation_result={validation_result}, "
                f"freshness_state={freshness_state}, consumer={consumer}, "
                f"market_session={market_session}"
            )


# ---------------------------------------------------------------------------
# Strategies for Property 8
# ---------------------------------------------------------------------------

from datetime import datetime, timezone
from decimal import Decimal

from utils.market_data_reliability.eligibility import EligibilityResolver
from utils.market_data_reliability.snapshot import Snapshot

# Execution consumers that must always be blocked on untrusted/degraded data
_EXECUTION_CONSUMERS = ["PM", "Risk_Geometry_Gate"]

# Trust states that should block execution consumers
_UNTRUSTED_TRUST_STATES = ["untrusted", "degraded"]

# Strategy: snapshot with untrusted or degraded trust_state
st_untrusted_snapshot = st.builds(
    Snapshot,
    symbol=st.text(
        alphabet=st.characters(whitelist_categories=("Lu",)),
        min_size=1,
        max_size=5,
    ),
    data_type=st.sampled_from(["quote", "candle", "atr", "volume", "previous_close"]),
    requested_at=st.just(datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)),
    provider=st.sampled_from(["finnhub", "yfinance", "alpaca"]),
    provider_status=st.sampled_from(["success", "error", "rate_limited", "timeout", "empty"]),
    market_session=st.sampled_from(["open", "pre_market", "after_hours", "closed"]),
    last_price=st.one_of(st.none(), st.decimals(min_value="0.01", max_value="5000", places=2)),
    bid=st.one_of(st.none(), st.decimals(min_value="0.01", max_value="5000", places=2)),
    ask=st.one_of(st.none(), st.decimals(min_value="0.01", max_value="5000", places=2)),
    previous_close=st.one_of(st.none(), st.decimals(min_value="0.01", max_value="5000", places=2)),
    open=st.one_of(st.none(), st.decimals(min_value="0.01", max_value="5000", places=2)),
    high=st.one_of(st.none(), st.decimals(min_value="0.01", max_value="5000", places=2)),
    low=st.one_of(st.none(), st.decimals(min_value="0.01", max_value="5000", places=2)),
    volume=st.one_of(st.none(), st.integers(min_value=0, max_value=1000000)),
    fetched_at=st.just(datetime(2025, 1, 15, 10, 0, 1, tzinfo=timezone.utc)),
    source_timestamp=st.one_of(
        st.none(),
        st.just(datetime(2025, 1, 15, 9, 59, 50, tzinfo=timezone.utc)),
    ),
    age_seconds=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    freshness_state=st.sampled_from(["fresh", "aging", "stale", "unavailable", "market_closed"]),
    trust_state=st.sampled_from(_UNTRUSTED_TRUST_STATES),
    degradation_reasons=st.lists(
        st.sampled_from([
            "cross_symbol_response", "invalid_price", "all_providers_failed",
            "missing_source_timestamp", "stale_source_timestamp", "provider_error",
            "rate_limited", "empty_response",
        ]),
        min_size=1,
        max_size=3,
        unique=True,
    ).map(tuple),
    raw_provider_latency_ms=st.one_of(st.none(), st.floats(min_value=0.0, max_value=5000.0)),
    fallback_primary_provider=st.one_of(st.none(), st.sampled_from(["finnhub", "yfinance", "alpaca"])),
)

st_execution_consumer = st.sampled_from(_EXECUTION_CONSUMERS)

st_data_type_prop8 = st.sampled_from(["quote", "candle", "atr", "volume", "previous_close"])


# ---------------------------------------------------------------------------
# Property 8: Execution Consumer Fail-Closed on Untrusted Data
# Feature: market-data-reliability-layer, Property 8: Execution Consumer Fail-Closed on Untrusted Data
# ---------------------------------------------------------------------------


class TestProperty8ExecutionConsumerFailClosedOnUntrustedData:
    """
    For any Snapshot with trust_state of "untrusted" or "degraded" AND consumer
    in the execution category (PM, Risk_Geometry_Gate), the EligibilityResolver
    SHALL return eligible=False with a non-empty reason_code.

    **Validates: Requirements 6.3, 7.1, 7.6**
    """

    @given(
        snapshot=st_untrusted_snapshot,
        consumer=st_execution_consumer,
        data_type=st_data_type_prop8,
    )
    @settings(max_examples=200)
    def test_execution_consumer_blocked_on_untrusted_data(
        self, snapshot, consumer, data_type
    ):
        """Execution consumers are always blocked when trust_state is untrusted or degraded."""
        resolver = EligibilityResolver()
        result = resolver.is_eligible(
            snapshot=snapshot,
            consumer=consumer,
            data_type=data_type,
            allow_stale_for_display=False,
        )

        assert result.eligible is False, (
            f"Execution consumer '{consumer}' must be blocked on "
            f"trust_state='{snapshot.trust_state}', but got eligible=True"
        )
        assert result.reason_code is not None and len(result.reason_code) > 0, (
            f"Execution consumer '{consumer}' blocked on trust_state='{snapshot.trust_state}' "
            f"must have a non-empty reason_code, got: {result.reason_code!r}"
        )

    @given(
        snapshot=st_untrusted_snapshot,
        consumer=st_execution_consumer,
        data_type=st_data_type_prop8,
    )
    @settings(max_examples=200)
    def test_execution_consumer_blocked_even_with_allow_stale_for_display(
        self, snapshot, consumer, data_type
    ):
        """allow_stale_for_display=True does not bypass fail-closed for execution consumers."""
        resolver = EligibilityResolver()
        result = resolver.is_eligible(
            snapshot=snapshot,
            consumer=consumer,
            data_type=data_type,
            allow_stale_for_display=True,
        )

        assert result.eligible is False, (
            f"Execution consumer '{consumer}' must be blocked on "
            f"trust_state='{snapshot.trust_state}' even with allow_stale_for_display=True, "
            f"but got eligible=True"
        )
        assert result.reason_code is not None and len(result.reason_code) > 0, (
            f"Execution consumer '{consumer}' blocked on trust_state='{snapshot.trust_state}' "
            f"with allow_stale_for_display=True must have a non-empty reason_code, "
            f"got: {result.reason_code!r}"
        )


# ---------------------------------------------------------------------------
# Strategies for Property 9
# ---------------------------------------------------------------------------

from datetime import datetime, timezone
from decimal import Decimal

from utils.market_data_reliability.eligibility import EligibilityResolver
from utils.market_data_reliability.snapshot import Snapshot

# Display consumers that may receive degraded data with labeling
_DISPLAY_CONSUMERS_P9 = ["Dashboard_API", "Analyst", "Reviewer", "CEO_Output"]

# Freshness states that pair with degraded trust (non-critical staleness)
_DEGRADED_FRESHNESS_STATES = ["stale", "aging"]

# Data types relevant to display consumers
_DATA_TYPES_P9 = ["quote", "candle", "atr", "volume", "previous_close"]

st_display_consumer = st.sampled_from(_DISPLAY_CONSUMERS_P9)
st_degraded_freshness = st.sampled_from(_DEGRADED_FRESHNESS_STATES)
st_data_type_p9 = st.sampled_from(_DATA_TYPES_P9)
st_symbol_p9 = st.text(
    alphabet=st.characters(whitelist_categories=("Lu",)),
    min_size=1,
    max_size=5,
)
st_positive_decimal = st.decimals(min_value="0.01", max_value="9999.99", places=2)
st_age_seconds_p9 = st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False)

# Strategy: build a degraded Snapshot
_NOW = datetime(2025, 7, 15, 14, 30, 0, tzinfo=timezone.utc)

st_degraded_snapshot = st.builds(
    Snapshot,
    symbol=st_symbol_p9,
    data_type=st_data_type_p9,
    requested_at=st.just(_NOW),
    provider=st.sampled_from(["finnhub", "yfinance", "alpaca"]),
    provider_status=st.just("success"),
    market_session=st.sampled_from(["open", "pre_market", "after_hours"]),
    last_price=st_positive_decimal,
    bid=st.just(None),
    ask=st.just(None),
    previous_close=st_positive_decimal,
    open=st.just(None),
    high=st.just(None),
    low=st.just(None),
    volume=st.just(None),
    fetched_at=st.just(_NOW),
    source_timestamp=st.just(None),
    age_seconds=st_age_seconds_p9,
    freshness_state=st_degraded_freshness,
    trust_state=st.just("degraded"),
    degradation_reasons=st.sampled_from([
        ("stale_data",),
        ("missing_source_timestamp",),
        ("stale_source_timestamp",),
        ("stale_data", "missing_source_timestamp"),
    ]),
    raw_provider_latency_ms=st.just(None),
    fallback_primary_provider=st.just(None),
)


# ---------------------------------------------------------------------------
# Property 9: Display Consumer Receives Degraded Data with Label
# Feature: market-data-reliability-layer, Property 9: Display Consumer Receives Degraded Data with Label
# ---------------------------------------------------------------------------


class TestProperty9DisplayConsumerReceivesDegradedDataWithLabel:
    """
    For any Snapshot with trust_state of "degraded" AND consumer in the display
    category (Dashboard_API, Analyst, Reviewer, CEO_Output) AND
    allow_stale_for_display=True, the EligibilityResolver SHALL return
    eligible=True and the snapshot SHALL retain its non-trusted freshness_state
    and trust_state (labels are preserved, not stripped).

    **Validates: Requirements 5.4, 6.4, 9.1, 9.4**
    """

    @given(
        snapshot=st_degraded_snapshot,
        consumer=st_display_consumer,
    )
    @settings(max_examples=200)
    def test_display_consumer_eligible_for_degraded_data_with_allow_stale(
        self, snapshot, consumer
    ):
        """Display consumer with allow_stale_for_display=True receives degraded data as eligible."""
        resolver = EligibilityResolver()

        result = resolver.is_eligible(
            snapshot=snapshot,
            consumer=consumer,
            data_type=snapshot.data_type,
            allow_stale_for_display=True,
        )

        # Must be eligible
        assert result.eligible is True, (
            f"Expected eligible=True for display consumer={consumer!r} with "
            f"allow_stale_for_display=True, trust_state='degraded', "
            f"freshness_state={snapshot.freshness_state!r}, "
            f"got eligible={result.eligible}, reason_code={result.reason_code!r}"
        )

        # trust_state must be preserved (degraded label not stripped to trusted)
        assert result.snapshot.trust_state == "degraded", (
            f"Expected trust_state='degraded' preserved on returned snapshot, "
            f"got trust_state={result.snapshot.trust_state!r} for consumer={consumer!r}"
        )

        # freshness_state must be preserved (not modified to 'fresh')
        assert result.snapshot.freshness_state == snapshot.freshness_state, (
            f"Expected freshness_state={snapshot.freshness_state!r} preserved, "
            f"got freshness_state={result.snapshot.freshness_state!r} for consumer={consumer!r}"
        )


# ---------------------------------------------------------------------------
# Strategies for Property 7
# ---------------------------------------------------------------------------

from unittest.mock import patch as mock_patch

from utils.market_data_reliability.cache import SnapshotCache
from utils.market_data_reliability.freshness import FreshnessClassifier
from utils.market_data_reliability.config import _DEFAULT_FRESHNESS_THRESHOLDS, _DEFAULT_CACHE_TTLS
from utils.market_data_reliability.snapshot import CacheKey, Snapshot

# Data types for cache freshness testing (non-closed sessions to get fresh/aging/stale)
_CACHE_DATA_TYPES = ["quote", "candle"]
_CACHE_MARKET_SESSIONS = ["open", "pre_market", "after_hours"]

# Strategy: original age_seconds (0-1000s)
st_original_age = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)

# Strategy: elapsed time since cache store (0-1000s)
st_elapsed_seconds = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)

# Strategy: data type for cache test
st_cache_data_type = st.sampled_from(_CACHE_DATA_TYPES)

# Strategy: market session (exclude closed to get threshold-based classification)
st_cache_market_session = st.sampled_from(_CACHE_MARKET_SESSIONS)

# Fixed timestamps for snapshot construction
_CACHE_FETCHED_AT = datetime(2025, 7, 15, 14, 30, 0, tzinfo=timezone.utc)
_CACHE_SOURCE_TS = datetime(2025, 7, 15, 14, 29, 55, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Property 7: Cache Freshness Recomputation
# Feature: market-data-reliability-layer, Property 7: Cache Freshness Recomputation
# ---------------------------------------------------------------------------


class TestProperty7CacheFreshnessRecomputation:
    """
    For any cached Snapshot served after elapsed time, the returned Snapshot's
    age_seconds SHALL equal the original age_seconds plus the elapsed time since
    caching, and freshness_state SHALL be recomputed from the new age_seconds
    (potentially transitioning from fresh -> aging -> stale).

    **Validates: Requirements 5.3, 5.5**
    """

    @given(
        original_age=st_original_age,
        elapsed_seconds=st_elapsed_seconds,
        data_type=st_cache_data_type,
        market_session=st_cache_market_session,
    )
    @settings(max_examples=200)
    def test_cache_recomputes_age_and_freshness_on_get(
        self, original_age, elapsed_seconds, data_type, market_session
    ):
        """Cache get() recomputes age_seconds = original + elapsed and reclassifies freshness."""
        classifier = FreshnessClassifier(_DEFAULT_FRESHNESS_THRESHOLDS)
        cache = SnapshotCache(
            cache_ttls=_DEFAULT_CACHE_TTLS,
            freshness_classifier=classifier,
        )

        # Build a snapshot with known age_seconds
        snapshot = Snapshot(
            symbol="AAPL",
            data_type=data_type,
            requested_at=_CACHE_FETCHED_AT,
            provider="finnhub",
            provider_status="success",
            market_session=market_session,
            last_price=Decimal("150.00"),
            bid=None,
            ask=None,
            previous_close=Decimal("149.00"),
            open=None,
            high=None,
            low=None,
            volume=None,
            fetched_at=_CACHE_FETCHED_AT,
            source_timestamp=_CACHE_SOURCE_TS,
            age_seconds=original_age,
            freshness_state="fresh",  # Initial state doesn't matter; will be recomputed
            trust_state="trusted",
            degradation_reasons=(),
            raw_provider_latency_ms=None,
            fallback_primary_provider=None,
        )

        key = CacheKey(
            symbol="AAPL",
            data_type=data_type,
            provider_policy="primary",
            market_session=market_session,
        )

        # Mock time.monotonic(): first call for put(), second call for get()
        store_time = 1000.0
        get_time = store_time + elapsed_seconds

        with mock_patch("utils.market_data_reliability.cache.time.monotonic") as mock_mono:
            # put() reads monotonic once for stored_at
            mock_mono.return_value = store_time
            cache.put(key, snapshot)

            # get() reads monotonic to compute elapsed
            mock_mono.return_value = get_time
            result = cache.get(key)

        assert result is not None, "Cache get() should return a snapshot after put()"

        # 1. age_seconds == original + elapsed (float comparison with tolerance)
        expected_age = original_age + elapsed_seconds
        assert abs(result.age_seconds - expected_age) < 1e-6, (
            f"Expected age_seconds ~= {expected_age}, got {result.age_seconds} "
            f"(original={original_age}, elapsed={elapsed_seconds})"
        )

        # 2. freshness_state matches what FreshnessClassifier produces for new age
        expected_freshness = classifier.classify(
            age_seconds=expected_age,
            data_type=data_type,
            consumer="execution",
            market_session=market_session,
        )
        assert result.freshness_state == expected_freshness, (
            f"Expected freshness_state={expected_freshness!r} for age={expected_age:.2f}s, "
            f"data_type={data_type!r}, market_session={market_session!r}, "
            f"got {result.freshness_state!r}"
        )

        # 3. fetched_at preserved from original
        assert result.fetched_at == snapshot.fetched_at, (
            f"Expected fetched_at preserved: {snapshot.fetched_at}, got {result.fetched_at}"
        )

        # 4. source_timestamp preserved from original
        assert result.source_timestamp == snapshot.source_timestamp, (
            f"Expected source_timestamp preserved: {snapshot.source_timestamp}, "
            f"got {result.source_timestamp}"
        )


# ---------------------------------------------------------------------------
# Strategies for Property 1
# ---------------------------------------------------------------------------

from utils.market_data_reliability.serialization import serialize, deserialize

# Strategy: generate arbitrary Snapshot instances for round-trip testing
st_optional_decimal = st.one_of(
    st.none(),
    st.decimals(min_value="0.01", max_value="99999.99", places=2, allow_nan=False, allow_infinity=False),
)

st_optional_datetime = st.one_of(
    st.none(),
    st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 12, 31),
        timezones=st.just(timezone.utc),
    ),
)

st_degradation_reasons_p1 = st.lists(
    st.sampled_from([
        "cross_symbol_response",
        "invalid_price",
        "all_providers_failed",
        "missing_source_timestamp",
        "stale_source_timestamp",
        "provider_error",
        "rate_limited",
        "empty_response",
        "malformed_json",
        "network_timeout",
        "stale_data",
    ]),
    min_size=0,
    max_size=4,
    unique=True,
).map(tuple)

st_snapshot = st.builds(
    Snapshot,
    symbol=st.text(
        alphabet=st.characters(whitelist_categories=("Lu",)),
        min_size=1,
        max_size=5,
    ),
    data_type=st.sampled_from(["quote", "candle", "atr", "volume", "previous_close"]),
    requested_at=st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 12, 31),
        timezones=st.just(timezone.utc),
    ),
    provider=st.sampled_from(["finnhub", "yfinance", "alpaca"]),
    provider_status=st.sampled_from(["success", "error", "rate_limited", "timeout", "empty"]),
    market_session=st.sampled_from(["open", "pre_market", "after_hours", "closed"]),
    last_price=st_optional_decimal,
    bid=st_optional_decimal,
    ask=st_optional_decimal,
    previous_close=st_optional_decimal,
    open=st_optional_decimal,
    high=st_optional_decimal,
    low=st_optional_decimal,
    volume=st.one_of(st.none(), st.integers(min_value=0, max_value=10000000)),
    fetched_at=st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 12, 31),
        timezones=st.just(timezone.utc),
    ),
    source_timestamp=st_optional_datetime,
    age_seconds=st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
    freshness_state=st.sampled_from(["fresh", "aging", "stale", "unavailable", "market_closed"]),
    trust_state=st.sampled_from(["trusted", "degraded", "untrusted"]),
    degradation_reasons=st_degradation_reasons_p1,
    raw_provider_latency_ms=st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    ),
    fallback_primary_provider=st.one_of(st.none(), st.sampled_from(["finnhub", "yfinance", "alpaca"])),
)


# ---------------------------------------------------------------------------
# Property 1: Snapshot Serialization Round-Trip
# Feature: market-data-reliability-layer, Property 1: Snapshot Serialization Round-Trip
# ---------------------------------------------------------------------------


class TestProperty1SnapshotSerializationRoundTrip:
    """
    For any valid Snapshot instance with arbitrary Decimal prices, timestamps,
    and degradation reasons, serializing to dict and then deserializing back
    SHALL produce a Snapshot equal to the original.

    **Validates: Requirements 13.3, 13.4, 13.5**
    """

    @given(snapshot=st_snapshot)
    @settings(max_examples=200)
    def test_serialize_then_deserialize_equals_original(self, snapshot):
        """Serializing a Snapshot and deserializing the result produces the original."""
        assert deserialize(serialize(snapshot)) == snapshot


# ---------------------------------------------------------------------------
# Strategies for Property 10
# ---------------------------------------------------------------------------

from utils.market_data_reliability.backoff import BackoffTracker

_PROVIDERS_P10 = ["finnhub", "yfinance", "alpaca"]
_DATA_TYPES_P10 = ["quote", "candle", "atr", "volume", "previous_close"]
_FAILURE_TYPES_P10 = ["rate_limit", "network_error", "empty_response"]

st_provider_p10 = st.sampled_from(_PROVIDERS_P10)
st_data_type_p10 = st.sampled_from(_DATA_TYPES_P10)
st_failure_type_p10 = st.sampled_from(_FAILURE_TYPES_P10)


# ---------------------------------------------------------------------------
# Property 10: Backoff Scoping Isolation
# Feature: market-data-reliability-layer, Property 10: Backoff Scoping Isolation
# ---------------------------------------------------------------------------


class TestProperty10BackoffScopingIsolation:
    """
    For any provider P and data_types D1 != D2, recording a failure for (P, D1)
    and entering backoff SHALL NOT cause is_in_backoff(P, D2) to return True.

    **Validates: Requirements 10.3**
    """

    @given(
        provider=st_provider_p10,
        data_type_1=st_data_type_p10,
        data_type_2=st_data_type_p10,
        failure_type=st_failure_type_p10,
    )
    @settings(max_examples=200)
    def test_backoff_scoped_to_provider_data_type_pair(
        self, provider, data_type_1, data_type_2, failure_type
    ):
        """Backoff for (provider, data_type_1) does not affect (provider, data_type_2)."""
        assume(data_type_1 != data_type_2)

        tracker = BackoffTracker(
            backoff_durations={
                "rate_limit": 60,
                "network_error": 30,
                "empty_response": 15,
            }
        )

        # Record failure for (provider, data_type_1)
        tracker.record_failure(provider, data_type_1, failure_type)

        # Sanity check: the failed pair IS in backoff
        assert tracker.is_in_backoff(provider, data_type_1) is True, (
            f"Expected is_in_backoff({provider!r}, {data_type_1!r}) == True after "
            f"record_failure with failure_type={failure_type!r}"
        )

        # Isolation property: the OTHER data type is NOT in backoff
        assert tracker.is_in_backoff(provider, data_type_2) is False, (
            f"Expected is_in_backoff({provider!r}, {data_type_2!r}) == False, "
            f"but backoff leaked from ({provider!r}, {data_type_1!r})"
        )


# ---------------------------------------------------------------------------
# Strategies for Property 11
# ---------------------------------------------------------------------------

import time
from unittest.mock import patch as mock_patch_p11

from utils.market_data_reliability.layer import ReliabilityLayer
from utils.market_data_reliability.config import ReliabilityConfig

# Execution consumers for feature flag mode tests
_EXECUTION_CONSUMERS_P11 = ["PM", "Risk_Geometry_Gate"]

# Valid data types and symbol strategies
_DATA_TYPES_P11 = ["quote", "candle", "atr", "volume", "previous_close"]

st_symbol_p11 = st.text(
    alphabet=st.characters(whitelist_categories=("Lu",)),
    min_size=1,
    max_size=5,
)
st_data_type_p11 = st.sampled_from(_DATA_TYPES_P11)
st_execution_consumer_p11 = st.sampled_from(_EXECUTION_CONSUMERS_P11)


def _bad_provider(provider: str, symbol: str, data_type: str) -> dict:
    """Mock provider that always returns data for a WRONG symbol → untrusted."""
    return {
        "s": "WRONG_SYMBOL",
        "c": 150.0,
        "h": 152.0,
        "l": 149.0,
        "o": 150.5,
        "pc": 149.0,
        "t": int(time.time()),
        "v": 1000000,
    }


def _make_config_with_mode(mode: str) -> ReliabilityConfig:
    """Create a ReliabilityConfig by patching the env var for mode."""
    env_overrides = {"MARKET_DATA_RELIABILITY_MODE": mode}
    # Clear all MDR keys to avoid interference from environment
    clean_env = {
        k: v for k, v in os.environ.items()
        if not k.startswith("MDR_") and k != "MARKET_DATA_RELIABILITY_MODE"
    }
    clean_env.update(env_overrides)
    with patch.dict(os.environ, clean_env, clear=True):
        return ReliabilityConfig.from_environment()


# ---------------------------------------------------------------------------
# Property 11: Feature Flag Mode Behavior
# Feature: market-data-reliability-layer, Property 11: Feature Flag Mode Behavior
# ---------------------------------------------------------------------------


class TestProperty11FeatureFlagModeBehavior:
    """
    For any market data request, WHEN MARKET_DATA_RELIABILITY_MODE is "disabled",
    the layer SHALL not modify the data path (passthrough, always eligible).
    WHEN mode is "observe", fail-closed checks SHALL log but not block.
    WHEN mode is "enforcing", fail-closed checks SHALL block execution consumers.

    **Validates: Requirements 14.2, 14.3, 14.4**
    """

    @given(
        symbol=st_symbol_p11,
        data_type=st_data_type_p11,
        consumer=st_execution_consumer_p11,
    )
    @settings(max_examples=200)
    def test_disabled_mode_passthrough_always_eligible(
        self, symbol, data_type, consumer
    ):
        """Disabled mode: get_snapshot returns eligible=True (passthrough) for any request."""
        config = _make_config_with_mode("disabled")
        layer = ReliabilityLayer(config=config, fetch_from_provider=_bad_provider)

        result = layer.get_snapshot(symbol, data_type, consumer)

        assert result.eligibility.eligible is True, (
            f"Disabled mode must always return eligible=True (passthrough), "
            f"but got eligible={result.eligibility.eligible} for "
            f"symbol={symbol!r}, data_type={data_type!r}, consumer={consumer!r}"
        )

    @given(
        symbol=st_symbol_p11,
        data_type=st_data_type_p11,
        consumer=st_execution_consumer_p11,
    )
    @settings(max_examples=200)
    def test_observe_mode_does_not_block_execution_consumer(
        self, symbol, data_type, consumer
    ):
        """Observe mode: untrusted data still returns eligible=True (observe doesn't block)."""
        config = _make_config_with_mode("observe")
        layer = ReliabilityLayer(config=config, fetch_from_provider=_bad_provider)

        result = layer.get_snapshot(symbol, data_type, consumer)

        assert result.eligibility.eligible is True, (
            f"Observe mode must not block execution consumers, "
            f"but got eligible={result.eligibility.eligible} for "
            f"symbol={symbol!r}, data_type={data_type!r}, consumer={consumer!r}"
        )

    @given(
        symbol=st_symbol_p11,
        data_type=st_data_type_p11,
        consumer=st_execution_consumer_p11,
    )
    @settings(max_examples=200)
    def test_enforcing_mode_blocks_execution_consumer_on_untrusted(
        self, symbol, data_type, consumer
    ):
        """Enforcing mode: untrusted data blocks execution consumers (eligible=False)."""
        config = _make_config_with_mode("enforcing")
        layer = ReliabilityLayer(config=config, fetch_from_provider=_bad_provider)

        result = layer.get_snapshot(symbol, data_type, consumer)

        assert result.eligibility.eligible is False, (
            f"Enforcing mode must block execution consumers on untrusted data, "
            f"but got eligible={result.eligibility.eligible} for "
            f"symbol={symbol!r}, data_type={data_type!r}, consumer={consumer!r}"
        )
