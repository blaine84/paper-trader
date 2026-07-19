"""Response validation for the Market Data Reliability Layer.

Validates raw provider responses before normalization. Detects cross-symbol
responses, invalid prices, missing/stale timestamps, provider errors, rate
limiting, empty responses, and malformed JSON.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from utils.market_data_reliability.snapshot import ValidationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known provider response key mappings
# ---------------------------------------------------------------------------

# Keys where a provider may include the symbol in the response
_SYMBOL_KEYS = ("s", "symbol", "ticker", "Symbol", "T")

# Keys where a provider may include price fields (Finnhub quote format)
_PRICE_KEYS = ("c", "current_price", "last_price", "price", "close", "l", "o", "h", "pc")

# Keys that indicate a provider error
_ERROR_KEYS = ("error", "Error", "err", "message")

# Keys that may hold a source timestamp
_TIMESTAMP_KEYS = ("t", "timestamp", "time", "updated", "latestTime", "regularMarketTime")

# Rate-limit indicators in response values
_RATE_LIMIT_MARKERS = (
    "rate limit",
    "too many requests",
    "429",
    "limit exceeded",
    "API limit",
)


class ResponseValidator:
    """Validates raw provider responses before normalization.

    Checks for safety-critical and non-critical issues in raw provider dicts.
    Safety-critical failures (cross_symbol_response, invalid_price) cause
    is_valid=False. Non-critical issues (missing_source_timestamp,
    stale_source_timestamp) produce degradation_reasons but is_valid=True.

    Parameters
    ----------
    staleness_threshold_seconds : float
        Maximum acceptable age (in seconds) for a source timestamp before
        it is considered stale. Defaults to 300 seconds (5 minutes).
    """

    def __init__(self, staleness_threshold_seconds: float = 300.0) -> None:
        self._staleness_threshold = staleness_threshold_seconds

    def validate(self, raw: dict, symbol: str, data_type: str) -> ValidationResult:
        """Validate a raw provider response.

        Parameters
        ----------
        raw : dict
            The raw response dictionary from the provider.
        symbol : str
            The requested ticker symbol (what we asked for).
        data_type : str
            The type of data requested (e.g., "quote", "candle").

        Returns
        -------
        ValidationResult
            is_valid=False for safety-critical failures (cross_symbol_response,
            invalid_price). is_valid=True with degradation_reasons for
            non-critical issues. is_valid=True with empty degradation_reasons
            when no issues detected.
        """
        degradation_reasons: list[str] = []
        is_valid = True

        # Check for malformed/empty response first
        if self._check_malformed_json(raw):
            logger.warning(
                "Validation failure: malformed_json | symbol=%s data_type=%s",
                symbol, data_type,
            )
            return ValidationResult(is_valid=False, degradation_reasons=("malformed_json",))

        if self._check_empty_response(raw):
            logger.warning(
                "Validation failure: empty_response | symbol=%s data_type=%s",
                symbol, data_type,
            )
            return ValidationResult(is_valid=False, degradation_reasons=("empty_response",))

        # Check for provider error
        if self._check_provider_error(raw):
            degradation_reasons.append("provider_error")
            is_valid = False

        # Check for rate limiting
        if self._check_rate_limited(raw):
            degradation_reasons.append("rate_limited")
            is_valid = False

        # Check cross-symbol response (safety-critical)
        if self._check_cross_symbol(raw, symbol):
            degradation_reasons.append("cross_symbol_response")
            is_valid = False

        # Check invalid price (safety-critical for equity/ETF)
        if self._check_invalid_price(raw, data_type):
            degradation_reasons.append("invalid_price")
            is_valid = False

        # Check missing source timestamp (non-critical)
        source_ts = self._parse_source_timestamp(raw)
        if source_ts is None:
            degradation_reasons.append("missing_source_timestamp")
            logger.warning(
                "Validation failure: missing_source_timestamp | "
                "symbol=%s data_type=%s",
                symbol, data_type,
            )
        else:
            # Check stale source timestamp (non-critical)
            if self._check_stale_timestamp(source_ts):
                degradation_reasons.append("stale_source_timestamp")
                logger.warning(
                    "Validation failure: stale_source_timestamp | "
                    "symbol=%s data_type=%s threshold=%.1fs",
                    symbol, data_type, self._staleness_threshold,
                )

        # Log safety-critical failures
        if not is_valid:
            reasons_str = ", ".join(
                r for r in degradation_reasons
                if r not in ("missing_source_timestamp", "stale_source_timestamp")
            )
            if reasons_str:
                logger.warning(
                    "Validation failure (safety-critical): %s | "
                    "symbol=%s data_type=%s",
                    reasons_str, symbol, data_type,
                )

        return ValidationResult(
            is_valid=is_valid,
            degradation_reasons=tuple(degradation_reasons),
        )

    # -----------------------------------------------------------------------
    # Individual check methods
    # -----------------------------------------------------------------------

    def _check_malformed_json(self, raw: dict) -> bool:
        """Check if the response is not a valid dict (malformed parse).

        This handles the case where upstream code caught a JSON parse error
        and passed a marker dict, or a non-dict slipped through.
        """
        if not isinstance(raw, dict):
            return True
        # Convention: upstream may pass {"_parse_error": True} on JSON failure
        if raw.get("_parse_error"):
            return True
        return False

    def _check_empty_response(self, raw: dict) -> bool:
        """Check if the response is effectively empty.

        An empty dict, a dict with only None values, or an explicit empty
        marker indicates the provider returned no useful data.
        """
        if not raw:
            return True
        # Check for explicit empty markers
        if raw.get("_empty") or raw.get("empty"):
            return True
        # A dict where all values are None is effectively empty
        if all(v is None for v in raw.values()):
            return True
        return False

    def _check_provider_error(self, raw: dict) -> bool:
        """Check if the response contains provider error indicators."""
        for key in _ERROR_KEYS:
            value = raw.get(key)
            if value is not None and value != "" and value is not False:
                return True
        return False

    def _check_rate_limited(self, raw: dict) -> bool:
        """Check if the response indicates rate limiting."""
        # Check for explicit rate-limit status codes or markers
        status = raw.get("status") or raw.get("status_code")
        if status is not None:
            if str(status) == "429":
                return True

        # Check error messages for rate-limit language
        for key in _ERROR_KEYS:
            value = raw.get(key)
            if isinstance(value, str):
                value_lower = value.lower()
                for marker in _RATE_LIMIT_MARKERS:
                    if marker.lower() in value_lower:
                        return True

        return False

    def _check_cross_symbol(self, raw: dict, requested_symbol: str) -> bool:
        """Check if the response symbol differs from the requested symbol.

        Returns True if a symbol IS present in the response AND it does not
        match the requested symbol (case-insensitive).
        """
        for key in _SYMBOL_KEYS:
            response_symbol = raw.get(key)
            if response_symbol is not None and isinstance(response_symbol, str):
                if response_symbol.strip().upper() != requested_symbol.strip().upper():
                    return True
                # Symbol found and matches — no cross-symbol issue
                return False
        # No symbol key found in response — can't determine cross-symbol
        return False

    def _check_invalid_price(self, raw: dict, data_type: str) -> bool:
        """Check if price fields are non-positive for equity/ETF data.

        Only applies to data types where a positive price is expected
        (quote, candle, previous_close). ATR and volume have different
        semantics.
        """
        price_relevant_types = ("quote", "candle", "previous_close")
        if data_type not in price_relevant_types:
            return False

        for key in _PRICE_KEYS:
            value = raw.get(key)
            if value is None:
                continue
            try:
                price = float(value)
                if price <= 0:
                    return True
            except (ValueError, TypeError):
                # Non-numeric price is handled separately (could be malformed)
                continue

        return False

    def _parse_source_timestamp(self, raw: dict) -> Optional[datetime]:
        """Attempt to parse a source timestamp from the response.

        Tries multiple known keys and format conventions:
        - Unix epoch (int or float)
        - ISO 8601 string
        """
        for key in _TIMESTAMP_KEYS:
            value = raw.get(key)
            if value is None:
                continue

            parsed = self._try_parse_timestamp(value)
            if parsed is not None:
                return parsed

        return None

    def _try_parse_timestamp(self, value: object) -> Optional[datetime]:
        """Try to parse a single timestamp value into a datetime."""
        # Integer/float → Unix epoch
        if isinstance(value, (int, float)):
            if value <= 0:
                return None
            try:
                return datetime.fromtimestamp(value, tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                return None

        # String → ISO 8601 or common formats
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            # Try ISO 8601
            try:
                return datetime.fromisoformat(value)
            except (ValueError, TypeError):
                pass
            # Try as numeric string (epoch seconds)
            try:
                epoch = float(value)
                if epoch > 0:
                    return datetime.fromtimestamp(epoch, tz=timezone.utc)
            except (ValueError, TypeError):
                pass

        # datetime already
        if isinstance(value, datetime):
            return value

        return None

    def _check_stale_timestamp(self, source_ts: datetime) -> bool:
        """Check if the source timestamp is older than the staleness threshold.

        Compares the source timestamp against the current time (UTC).
        """
        now = datetime.now(tz=timezone.utc)
        # Ensure source_ts is timezone-aware for comparison
        if source_ts.tzinfo is None:
            source_ts = source_ts.replace(tzinfo=timezone.utc)
        age_seconds = (now - source_ts).total_seconds()
        return age_seconds > self._staleness_threshold
