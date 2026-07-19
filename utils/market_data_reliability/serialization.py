"""Snapshot serialization and deserialization.

Converts Snapshot instances to/from JSON-compatible dicts with stable keys.
Decimal fields are encoded as strings to preserve full precision.
Datetime fields are encoded as ISO 8601 strings.
Tuple fields are encoded as lists.

Requirements: 13.1, 13.2, 13.3, 13.4, 13.5
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from utils.market_data_reliability.snapshot import Snapshot


# Fields that hold Decimal values (Optional[Decimal])
_DECIMAL_FIELDS = frozenset({
    "last_price",
    "bid",
    "ask",
    "previous_close",
    "open",
    "high",
    "low",
})

# Fields that hold datetime values
_DATETIME_FIELDS = frozenset({
    "requested_at",
    "fetched_at",
    "source_timestamp",
})

# Fields that hold tuple values
_TUPLE_FIELDS = frozenset({
    "degradation_reasons",
})


def serialize(snapshot: Snapshot) -> dict:
    """Convert a Snapshot to a JSON-compatible dictionary.

    - Decimal values are encoded as strings to preserve full precision.
    - datetime values are encoded as ISO 8601 strings.
    - tuple values are encoded as lists.
    - None values are preserved as None.
    - int, float, str values pass through as-is.

    Args:
        snapshot: A frozen Snapshot dataclass instance.

    Returns:
        A dictionary with stable string keys and JSON-compatible values.
    """
    result: dict = {}

    for field_name in snapshot.__dataclass_fields__:
        value = getattr(snapshot, field_name)

        if field_name in _DECIMAL_FIELDS:
            result[field_name] = str(value) if value is not None else None
        elif field_name in _DATETIME_FIELDS:
            result[field_name] = value.isoformat() if value is not None else None
        elif field_name in _TUPLE_FIELDS:
            result[field_name] = list(value)
        else:
            # str, int, float, Optional[str], Optional[int], Optional[float] pass through
            result[field_name] = value

    return result


def deserialize(data: dict) -> Snapshot:
    """Reconstruct a Snapshot from its serialized dictionary form.

    - String values for Decimal fields are converted back to Decimal.
    - ISO 8601 strings for datetime fields are converted back to datetime.
    - List values for tuple fields are converted back to tuples.
    - None values are preserved as None.
    - int, float, str values pass through as-is.

    Args:
        data: A dictionary produced by serialize().

    Returns:
        A frozen Snapshot dataclass instance equal to the original.
    """
    kwargs: dict = {}

    for field_name in Snapshot.__dataclass_fields__:
        value = data[field_name]

        if field_name in _DECIMAL_FIELDS:
            kwargs[field_name] = Decimal(value) if value is not None else None
        elif field_name in _DATETIME_FIELDS:
            kwargs[field_name] = datetime.fromisoformat(value) if value is not None else None
        elif field_name in _TUPLE_FIELDS:
            kwargs[field_name] = tuple(value)
        else:
            kwargs[field_name] = value

    return Snapshot(**kwargs)
