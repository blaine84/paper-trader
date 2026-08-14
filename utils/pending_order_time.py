"""Time normalization for the pending limit order subsystem.

Every datetime that crosses a pending-order boundary — order fields, bar
timestamps, window bounds, watermarks — must be timezone-aware UTC before any
comparison or storage. This module is the single place that rule is enforced.

Why this exists as its own module: the surrounding codebase genuinely mixes
representations, and Python raises ``TypeError`` when a naive datetime is
compared to an aware one rather than degrading. A single missed coercion is
therefore a hard failure in the fill path, not a subtle drift.

The representations in play:

- ``utils.trade_events.log_trade_event()`` defaults to naive ``utcnow()``
- ``FinnhubClient.get_quote()`` returns ``utcnow().isoformat()`` (naive string)
- ``FinnhubClient.get_candles()`` returns epoch **seconds** as ints
- the yfinance candle fallback yields pandas/numpy values whose timezone
  awareness varies by interval
- ``trade_plans`` / ``pm_candidates`` store ``.isoformat()`` of aware values
- ``_get_recent_closed_trades_for_preflight()`` strips ``tzinfo`` to match
  naive SQLite columns

Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.8, 15.9
"""
from __future__ import annotations

import numbers
from datetime import datetime, timezone

__all__ = ["to_utc", "to_iso", "now_utc", "TimeNormalizationError"]


# Epoch values beyond this are almost certainly milliseconds that a provider
# handed us as if they were seconds. Interpreting them as seconds would land in
# the year 50000+, which silently makes every bar fail the `ts <= expires_at`
# window test instead of failing loudly. Reject them instead.
_MAX_PLAUSIBLE_EPOCH_SECONDS = 4_102_444_800  # 2100-01-01T00:00:00Z

# Fallback formats for strings that datetime.fromisoformat() cannot parse.
_FALLBACK_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d",
)


class TimeNormalizationError(ValueError):
    """Raised when a value cannot be normalized to an aware UTC datetime.

    Subclasses ``ValueError`` so existing ``except (TypeError, ValueError)``
    handlers in the codebase keep working unchanged.
    """


def now_utc() -> datetime:
    """Current time as an aware UTC datetime.

    Exists as a single seam so tests can patch one symbol instead of chasing
    ``datetime.now`` across the monitor, the filler, and the expiry sweep.
    """
    return datetime.now(timezone.utc)


def to_utc(value: datetime | str | int | float | None) -> datetime | None:
    """Normalize any supported representation to an aware UTC datetime.

    Conversion rules:

    - ``None``            -> ``None``
    - aware ``datetime``  -> converted to UTC
    - naive ``datetime``  -> assumed UTC, ``tzinfo`` attached
    - ``str``             -> parsed as ISO-8601 (``Z`` suffix accepted), then
                             the two datetime rules above
    - real number         -> epoch **seconds**, via
                             ``datetime.fromtimestamp(v, tz=timezone.utc)``

    ``datetime.utcfromtimestamp()`` is deliberately not used anywhere here: it
    returns a naive value, which is the exact bug this module exists to prevent.

    ``bool`` is rejected even though it is an ``int`` subclass, because
    ``to_utc(True)`` silently meaning 1970-01-01T00:00:01Z is never intended.

    Args:
        value: The value to normalize.

    Returns:
        An aware UTC datetime, or ``None`` when ``value`` is ``None``.

    Raises:
        TimeNormalizationError: If the value cannot be interpreted. Callers on
            fail-open paths (for example ``bars_from_candles()``, which drops
            ragged provider rows) are expected to catch this.
    """
    if value is None:
        return None

    # Guard before the numeric branch: bool is a subclass of int.
    if isinstance(value, bool):
        raise TimeNormalizationError(
            f"Refusing to interpret bool {value!r} as a timestamp"
        )

    # datetime first — this also covers pandas.Timestamp, which subclasses it.
    if isinstance(value, datetime):
        return _attach_or_convert(value)

    if isinstance(value, str):
        return _attach_or_convert(_parse_string(value))

    # numbers.Real covers Python int/float and numpy int64/float64, which are
    # registered with the numbers ABCs but are NOT int/float subclasses. The
    # yfinance candle fallback is the path that produces numpy scalars.
    if isinstance(value, numbers.Real):
        return _from_epoch_seconds(value)

    # Last resort: anything float()-able (e.g. Decimal, or an exotic scalar).
    try:
        return _from_epoch_seconds(float(value))
    except (TypeError, ValueError) as exc:
        raise TimeNormalizationError(
            f"Cannot normalize {type(value).__name__} {value!r} to a UTC datetime"
        ) from exc


def to_iso(value: datetime) -> str:
    """Serialize a datetime as ISO-8601 with an explicit UTC offset.

    Normalizes through :func:`to_utc` first, so a naive value passed here is
    stored with an offset rather than silently persisting as ambiguous.

    Args:
        value: The datetime to serialize. Must not be ``None``.

    Returns:
        An ISO-8601 string carrying ``+00:00``.

    Raises:
        TimeNormalizationError: If ``value`` is ``None`` or not normalizable.
    """
    normalized = to_utc(value)
    if normalized is None:
        raise TimeNormalizationError("Cannot serialize None as an ISO timestamp")
    return normalized.isoformat()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _attach_or_convert(value: datetime) -> datetime:
    """Attach UTC to a naive datetime, or convert an aware one to UTC."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_string(value: str) -> datetime:
    """Parse an ISO-8601-ish string into a datetime (tz handled by caller)."""
    text = value.strip()
    if not text:
        raise TimeNormalizationError("Cannot normalize an empty string")

    # fromisoformat() accepts "Z" only from Python 3.11. Normalize it anyway so
    # behavior does not depend on the interpreter minor version.
    candidate = text
    if candidate.endswith(("Z", "z")):
        candidate = candidate[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        pass

    for fmt in _FALLBACK_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    raise TimeNormalizationError(f"Cannot parse {value!r} as an ISO-8601 timestamp")


def _from_epoch_seconds(value: numbers.Real | float) -> datetime:
    """Convert epoch seconds to an aware UTC datetime, rejecting nonsense."""
    seconds = float(value)

    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        raise TimeNormalizationError(
            f"Cannot normalize non-finite epoch value {value!r}"
        )

    if seconds < 0:
        raise TimeNormalizationError(
            f"Refusing negative epoch value {value!r} (pre-1970 is never valid here)"
        )

    if seconds > _MAX_PLAUSIBLE_EPOCH_SECONDS:
        raise TimeNormalizationError(
            f"Epoch value {value!r} exceeds year 2100 as seconds — the provider "
            f"likely supplied milliseconds. Rejecting rather than silently "
            f"producing a timestamp that fails every window comparison."
        )

    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise TimeNormalizationError(
            f"Cannot convert epoch value {value!r} to a UTC datetime"
        ) from exc
