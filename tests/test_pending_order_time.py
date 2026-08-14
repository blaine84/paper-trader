"""Tests for utils/pending_order_time.py.

The invariant under test: every supported representation of a single instant
normalizes to the *same* aware UTC datetime, and nothing naive ever escapes.

Requirements: 15.1, 15.2, 15.3, 15.5
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from utils.pending_order_time import (
    TimeNormalizationError,
    now_utc,
    to_iso,
    to_utc,
)

# One fixed instant, expressed every way the codebase might hand it over.
# 2026-08-14T14:30:00Z == 10:30 ET (EDT, UTC-4)
INSTANT = datetime(2026, 8, 14, 14, 30, 0, tzinfo=timezone.utc)
EPOCH_SECONDS = INSTANT.timestamp()


# ---------------------------------------------------------------------------
# Cross-representation equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "representation",
    [
        pytest.param(INSTANT, id="aware_utc_datetime"),
        pytest.param(datetime(2026, 8, 14, 14, 30, 0), id="naive_datetime"),
        pytest.param("2026-08-14T14:30:00+00:00", id="iso_with_offset"),
        pytest.param("2026-08-14T14:30:00Z", id="iso_with_Z"),
        pytest.param("2026-08-14T14:30:00z", id="iso_with_lowercase_z"),
        pytest.param("2026-08-14T14:30:00", id="iso_naive"),
        pytest.param("2026-08-14 14:30:00", id="space_separated"),
        pytest.param(int(EPOCH_SECONDS), id="epoch_int"),
        pytest.param(float(EPOCH_SECONDS), id="epoch_float"),
    ],
)
def test_all_representations_normalize_to_the_same_instant(representation):
    """Every accepted input type round-trips to the same aware UTC instant."""
    result = to_utc(representation)

    assert result == INSTANT
    assert result.tzinfo is not None
    assert result.utcoffset() == timedelta(0)


def test_naive_and_aware_equivalents_produce_equal_results():
    """A naive value is assumed UTC, so it must equal its aware counterpart."""
    naive = datetime(2026, 8, 14, 14, 30, 0)
    aware = datetime(2026, 8, 14, 14, 30, 0, tzinfo=timezone.utc)

    assert to_utc(naive) == to_utc(aware)


def test_non_utc_aware_input_is_converted_not_relabeled():
    """An aware non-UTC value must be converted, preserving the instant."""
    # 10:30 ET on 2026-08-14 is EDT (UTC-4), i.e. 14:30 UTC.
    eastern = timezone(timedelta(hours=-4))
    result = to_utc(datetime(2026, 8, 14, 10, 30, 0, tzinfo=eastern))

    assert result == INSTANT
    assert result.hour == 14, "wall-clock hour must shift, not be relabeled"


# ---------------------------------------------------------------------------
# Epoch handling — the utcfromtimestamp() trap
# ---------------------------------------------------------------------------


def test_epoch_input_yields_aware_datetime():
    """Epoch conversion must not use utcfromtimestamp(), which returns naive.

    Asserted behaviorally: the result carries tzinfo. This is the single most
    important property in the module, because a naive result here would raise
    TypeError on the first window comparison in the fill path.
    """
    result = to_utc(EPOCH_SECONDS)

    assert result.tzinfo is not None
    assert result.utcoffset() == timedelta(0)


def test_epoch_zero_is_the_unix_epoch_in_utc():
    assert to_utc(0) == datetime(1970, 1, 1, tzinfo=timezone.utc)


def test_millisecond_epoch_is_rejected_loudly():
    """A provider handing over milliseconds must fail, not silently misjudge.

    Interpreted as seconds, a 2026 millisecond timestamp lands past year 50000,
    which would make every bar fail `ts <= expires_at` and silently produce
    zero fills forever.
    """
    milliseconds = int(EPOCH_SECONDS * 1000)

    with pytest.raises(TimeNormalizationError, match="milliseconds"):
        to_utc(milliseconds)


def test_negative_epoch_is_rejected():
    with pytest.raises(TimeNormalizationError, match="negative"):
        to_utc(-1)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_epoch_is_rejected(bad):
    with pytest.raises(TimeNormalizationError):
        to_utc(bad)


def test_numpy_style_scalars_are_accepted():
    """numpy int64/float64 are numbers.Real but NOT int/float subclasses.

    The yfinance candle fallback is the path that produces them, so an
    isinstance(value, int) check would silently reject real provider data.
    """
    numpy = pytest.importorskip("numpy")

    assert to_utc(numpy.int64(int(EPOCH_SECONDS))) == INSTANT
    assert to_utc(numpy.float64(EPOCH_SECONDS)) == INSTANT


def test_pandas_timestamp_is_accepted():
    """pandas.Timestamp subclasses datetime, so the datetime branch covers it."""
    pandas = pytest.importorskip("pandas")

    naive = pandas.Timestamp("2026-08-14 14:30:00")
    aware = pandas.Timestamp("2026-08-14 14:30:00", tz="UTC")

    assert to_utc(naive) == INSTANT
    assert to_utc(aware) == INSTANT


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


def test_none_passes_through_as_none():
    assert to_utc(None) is None


@pytest.mark.parametrize("bad", [True, False])
def test_bool_is_rejected(bad):
    """bool is an int subclass; to_utc(True) meaning 1970-01-01T00:00:01Z is
    never intended and would be a silent wrong answer."""
    with pytest.raises(TimeNormalizationError, match="bool"):
        to_utc(bad)


@pytest.mark.parametrize("bad", ["", "   ", "not a date", "2026-13-45"])
def test_unparseable_strings_are_rejected(bad):
    with pytest.raises(TimeNormalizationError):
        to_utc(bad)


def test_error_is_a_valueerror_subclass():
    """Existing `except (TypeError, ValueError)` handlers must keep working."""
    assert issubclass(TimeNormalizationError, ValueError)


# ---------------------------------------------------------------------------
# to_iso
# ---------------------------------------------------------------------------


def test_to_iso_round_trips_through_to_utc():
    serialized = to_iso(INSTANT)
    assert to_utc(serialized) == INSTANT


def test_to_iso_always_carries_an_explicit_offset():
    assert to_iso(INSTANT).endswith("+00:00")
    # A naive input must also come back with an offset attached.
    assert to_iso(datetime(2026, 8, 14, 14, 30, 0)).endswith("+00:00")


def test_to_iso_rejects_none():
    with pytest.raises(TimeNormalizationError):
        to_iso(None)


# ---------------------------------------------------------------------------
# DST boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "iso_utc,expected_et_hour,label",
    [
        # US DST 2026: starts Mar 8, ends Nov 1.
        ("2026-03-08T06:59:00Z", 1, "just_before_spring_forward"),
        ("2026-03-08T07:00:00Z", 3, "at_spring_forward"),
        ("2026-11-01T05:59:00Z", 1, "just_before_fall_back_EDT"),
        ("2026-11-01T06:00:00Z", 1, "at_fall_back_EST"),
    ],
)
def test_dst_boundary_epochs_normalize_correctly(iso_utc, expected_et_hour, label):
    """UTC normalization must be DST-agnostic, and converting to ET afterwards
    must land on the expected wall clock.

    Expiry resolution is the only place that reasons in Eastern time, so this
    pins down that the UTC values it receives are unambiguous across the two
    2026 transitions.
    """
    pytz = pytest.importorskip("pytz")

    instant = to_utc(iso_utc)
    assert instant.tzinfo is not None

    # Round-trip through epoch seconds — the representation get_candles() uses.
    assert to_utc(instant.timestamp()) == instant

    eastern = instant.astimezone(pytz.timezone("America/New_York"))
    assert eastern.hour == expected_et_hour, label


# ---------------------------------------------------------------------------
# now_utc
# ---------------------------------------------------------------------------


def test_now_utc_is_aware_and_current():
    before = datetime.now(timezone.utc)
    result = now_utc()
    after = datetime.now(timezone.utc)

    assert result.tzinfo is not None
    assert before <= result <= after


def test_now_utc_is_comparable_to_normalized_values():
    """The seam's whole purpose: no TypeError against normalized values."""
    assert now_utc() > to_utc("2020-01-01T00:00:00Z")
