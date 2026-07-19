"""Snapshot normalization for the Market Data Reliability Layer.

Converts raw provider response dicts into frozen Snapshot instances with
Decimal prices, computed age_seconds, and integrated validation/freshness/trust
classification.

Requirements: 1.1, 1.2, 1.3, 1.4, 4.6
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Optional

from utils.market_data_reliability.freshness import FreshnessClassifier
from utils.market_data_reliability.snapshot import Snapshot, ValidationResult
from utils.market_data_reliability.trust import TrustClassifier
from utils.market_data_reliability.validator import ResponseValidator

logger = logging.getLogger(__name__)


def _to_decimal(value: object) -> Optional[Decimal]:
    """Safely convert a value to Decimal.

    Uses Decimal(str(value)) pattern for safe float-to-Decimal conversion.
    Returns None if value is None or cannot be converted.
    """
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _to_int(value: object) -> Optional[int]:
    """Safely convert a value to int. Returns None if invalid."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _parse_unix_timestamp(value: object) -> Optional[datetime]:
    """Parse a Unix epoch timestamp (int or float) into a UTC datetime.

    Returns None if value is None, non-numeric, or out of range.
    """
    if value is None:
        return None
    try:
        epoch = float(value)
        if epoch <= 0:
            return None
        return datetime.fromtimestamp(epoch, tz=timezone.utc)
    except (ValueError, TypeError, OSError, OverflowError):
        return None


def _determine_provider_status(validation_result: ValidationResult) -> str:
    """Determine provider_status from validation degradation reasons.

    Maps validation failure codes to provider_status values.
    """
    reasons = validation_result.degradation_reasons

    if "rate_limited" in reasons:
        return "rate_limited"
    if "provider_error" in reasons:
        return "error"
    if "empty_response" in reasons or "malformed_json" in reasons:
        return "empty"
    if not validation_result.is_valid:
        return "error"
    return "success"


class SnapshotNormalizer:
    """Converts raw provider response dicts into normalized Snapshot instances.

    Integrates ResponseValidator, FreshnessClassifier, and TrustClassifier
    to produce fully classified snapshots with Decimal prices and computed
    temporal metadata.

    Parameters
    ----------
    validator : ResponseValidator
        Validates raw responses before normalization.
    freshness_classifier : FreshnessClassifier
        Classifies freshness state from age_seconds.
    trust_classifier : TrustClassifier
        Classifies trust state from validation and freshness results.
    default_market_session : str
        Market session to use when caller does not specify one.
        Defaults to "open" for conservative classification.
    """

    def __init__(
        self,
        validator: ResponseValidator,
        freshness_classifier: FreshnessClassifier,
        trust_classifier: TrustClassifier,
        default_market_session: str = "open",
    ) -> None:
        self._validator = validator
        self._freshness_classifier = freshness_classifier
        self._trust_classifier = trust_classifier
        self._default_market_session = default_market_session

    def normalize_quote(
        self,
        raw: dict,
        symbol: str,
        provider: str,
        requested_at: datetime,
        fetched_at: datetime,
    ) -> Snapshot:
        """Normalize a raw quote response into a Snapshot.

        Handles Finnhub quote format:
            {"s": "AAPL", "c": 187.43, "h": 188.50, "l": 186.20,
             "o": 187.00, "pc": 186.90, "t": 1689424200, "v": 55000000}

        Parameters
        ----------
        raw : dict
            Raw provider response dictionary.
        symbol : str
            Requested ticker symbol.
        provider : str
            Provider name ("finnhub", "yfinance", "alpaca").
        requested_at : datetime
            When the consumer made the request.
        fetched_at : datetime
            When the provider response was received.

        Returns
        -------
        Snapshot
            Fully classified frozen snapshot with Decimal prices.
        """
        data_type = "quote"

        # Step 1: Validate the raw response
        validation_result = self._validator.validate(raw, symbol, data_type)

        # Step 2: Parse source_timestamp from raw response
        source_timestamp = _parse_unix_timestamp(raw.get("t"))

        # Step 3: Compute age_seconds
        age_seconds = self._compute_age_seconds(source_timestamp, fetched_at)

        # Step 4: Classify freshness (use "PM" consumer for conservative defaults)
        freshness_state = self._freshness_classifier.classify(
            age_seconds, data_type, "PM", self._default_market_session
        )

        # Step 5: Classify trust
        trust_state, degradation_reasons = self._trust_classifier.classify(
            validation_result, freshness_state, "PM", self._default_market_session
        )

        # Step 6: Determine provider_status
        provider_status = _determine_provider_status(validation_result)

        # Step 7: Extract and convert price fields
        # If provider returned error/rate_limited/empty, price fields are None
        if provider_status in ("error", "rate_limited", "empty"):
            last_price = None
            bid = None
            ask = None
            previous_close = None
            open_price = None
            high = None
            low = None
            volume = None
        else:
            last_price = _to_decimal(raw.get("c"))
            bid = None  # Finnhub quote does not provide bid
            ask = None  # Finnhub quote does not provide ask
            previous_close = _to_decimal(raw.get("pc"))
            open_price = _to_decimal(raw.get("o"))
            high = _to_decimal(raw.get("h"))
            low = _to_decimal(raw.get("l"))
            volume = _to_int(raw.get("v"))

        # Step 8: Compute raw provider latency
        raw_provider_latency_ms = self._compute_latency_ms(requested_at, fetched_at)

        # Step 9: Build and return the Snapshot
        return Snapshot(
            symbol=symbol,
            data_type=data_type,
            requested_at=requested_at,
            provider=provider,
            provider_status=provider_status,
            market_session=self._default_market_session,
            last_price=last_price,
            bid=bid,
            ask=ask,
            previous_close=previous_close,
            open=open_price,
            high=high,
            low=low,
            volume=volume,
            fetched_at=fetched_at,
            source_timestamp=source_timestamp,
            age_seconds=age_seconds,
            freshness_state=freshness_state,
            trust_state=trust_state,
            degradation_reasons=degradation_reasons,
            raw_provider_latency_ms=raw_provider_latency_ms,
            fallback_primary_provider=None,
        )

    def normalize_candles(
        self,
        raw: dict,
        symbol: str,
        provider: str,
        requested_at: datetime,
        fetched_at: datetime,
    ) -> Snapshot:
        """Normalize a raw candle response into a Snapshot.

        Uses the latest candle (last element of arrays) for the snapshot.

        Handles Finnhub candle format:
            {"s": "AAPL", "c": [187.43, 188.10], "h": [188.50, 189.00],
             "l": [186.20, 187.00], "o": [187.00, 187.50],
             "t": [1689424200, 1689427800], "v": [5000, 6000]}

        Parameters
        ----------
        raw : dict
            Raw provider response dictionary.
        symbol : str
            Requested ticker symbol.
        provider : str
            Provider name ("finnhub", "yfinance", "alpaca").
        requested_at : datetime
            When the consumer made the request.
        fetched_at : datetime
            When the provider response was received.

        Returns
        -------
        Snapshot
            Fully classified frozen snapshot with Decimal prices from
            the most recent candle.
        """
        data_type = "candle"

        # Step 1: Validate the raw response
        validation_result = self._validator.validate(raw, symbol, data_type)

        # Step 2: Parse source_timestamp from the last candle timestamp
        source_timestamp = self._parse_candle_timestamp(raw)

        # Step 3: Compute age_seconds
        age_seconds = self._compute_age_seconds(source_timestamp, fetched_at)

        # Step 4: Classify freshness (use "PM" consumer for conservative defaults)
        freshness_state = self._freshness_classifier.classify(
            age_seconds, data_type, "PM", self._default_market_session
        )

        # Step 5: Classify trust
        trust_state, degradation_reasons = self._trust_classifier.classify(
            validation_result, freshness_state, "PM", self._default_market_session
        )

        # Step 6: Determine provider_status
        provider_status = _determine_provider_status(validation_result)

        # Step 7: Extract and convert price fields from last candle
        if provider_status in ("error", "rate_limited", "empty"):
            last_price = None
            bid = None
            ask = None
            previous_close = None
            open_price = None
            high = None
            low = None
            volume = None
        else:
            last_price = self._get_last_element_decimal(raw.get("c"))
            bid = None  # Candles do not provide bid
            ask = None  # Candles do not provide ask
            previous_close = None  # Candles do not provide previous_close
            open_price = self._get_last_element_decimal(raw.get("o"))
            high = self._get_last_element_decimal(raw.get("h"))
            low = self._get_last_element_decimal(raw.get("l"))
            volume = self._get_last_element_int(raw.get("v"))

        # Step 8: Compute raw provider latency
        raw_provider_latency_ms = self._compute_latency_ms(requested_at, fetched_at)

        # Step 9: Build and return the Snapshot
        return Snapshot(
            symbol=symbol,
            data_type=data_type,
            requested_at=requested_at,
            provider=provider,
            provider_status=provider_status,
            market_session=self._default_market_session,
            last_price=last_price,
            bid=bid,
            ask=ask,
            previous_close=previous_close,
            open=open_price,
            high=high,
            low=low,
            volume=volume,
            fetched_at=fetched_at,
            source_timestamp=source_timestamp,
            age_seconds=age_seconds,
            freshness_state=freshness_state,
            trust_state=trust_state,
            degradation_reasons=degradation_reasons,
            raw_provider_latency_ms=raw_provider_latency_ms,
            fallback_primary_provider=None,
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _compute_age_seconds(
        self, source_timestamp: Optional[datetime], fetched_at: datetime
    ) -> float:
        """Compute age_seconds from source_timestamp vs fetched_at.

        If source_timestamp is None, uses fetched_at as the reference point
        (age = 0.0), indicating we cannot determine true age.
        """
        if source_timestamp is None:
            return 0.0

        # Ensure both are timezone-aware for comparison
        src_ts = source_timestamp
        ref = fetched_at
        if src_ts.tzinfo is None:
            src_ts = src_ts.replace(tzinfo=timezone.utc)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)

        age = (ref - src_ts).total_seconds()
        # Clamp to non-negative — negative age means source is in the future
        return max(0.0, age)

    def _compute_latency_ms(
        self, requested_at: datetime, fetched_at: datetime
    ) -> Optional[float]:
        """Compute raw provider latency in milliseconds.

        Returns (fetched_at - requested_at).total_seconds() * 1000.
        """
        req = requested_at
        fetch = fetched_at
        if req.tzinfo is None:
            req = req.replace(tzinfo=timezone.utc)
        if fetch.tzinfo is None:
            fetch = fetch.replace(tzinfo=timezone.utc)

        latency_seconds = (fetch - req).total_seconds()
        return latency_seconds * 1000.0

    def _parse_candle_timestamp(self, raw: dict) -> Optional[datetime]:
        """Parse the most recent candle timestamp from the raw response.

        Expects raw["t"] to be a list of Unix epoch timestamps.
        Returns the last element parsed as a datetime, or None.
        """
        timestamps = raw.get("t")
        if not timestamps:
            return None

        if isinstance(timestamps, list) and len(timestamps) > 0:
            return _parse_unix_timestamp(timestamps[-1])

        # If "t" is a single value (non-list), try to parse it directly
        return _parse_unix_timestamp(timestamps)

    @staticmethod
    def _get_last_element_decimal(values: object) -> Optional[Decimal]:
        """Get the last element from a list and convert to Decimal.

        Returns None if values is None, empty, or not a list.
        """
        if values is None:
            return None
        if isinstance(values, list) and len(values) > 0:
            return _to_decimal(values[-1])
        # Single value (not a list) — convert directly
        if not isinstance(values, list):
            return _to_decimal(values)
        return None

    @staticmethod
    def _get_last_element_int(values: object) -> Optional[int]:
        """Get the last element from a list and convert to int.

        Returns None if values is None, empty, or not a list.
        """
        if values is None:
            return None
        if isinstance(values, list) and len(values) > 0:
            return _to_int(values[-1])
        # Single value (not a list) — convert directly
        if not isinstance(values, list):
            return _to_int(values)
        return None
