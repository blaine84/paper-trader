"""Tests for lineage deduplication logic.

Validates Requirements: 1.5
"""
from datetime import datetime, timedelta, timezone

import pytest

from utils.lineage_deduplication import (
    LineageRecord,
    TIMESTAMP_TOLERANCE,
    detect_duplicate_lineages,
    deduplicate_execution_intents,
    is_duplicate_lineage,
)


def _make_record(
    lineage_id: str = "lin-001",
    symbol: str = "AAPL",
    direction: str = "BUY",
    timestamp: datetime | None = None,
    pm_cycle_id: str = "cycle-1",
) -> LineageRecord:
    """Helper to create a LineageRecord with defaults."""
    if timestamp is None:
        timestamp = datetime(2025, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
    return LineageRecord(
        lineage_id=lineage_id,
        symbol=symbol,
        direction=direction,
        timestamp=timestamp,
        pm_cycle_id=pm_cycle_id,
    )


class TestDetectDuplicateLineages:
    """Tests for detect_duplicate_lineages."""

    def test_empty_input(self):
        assert detect_duplicate_lineages([]) == []

    def test_single_record_no_duplicates(self):
        records = [_make_record()]
        assert detect_duplicate_lineages(records) == []

    def test_two_identical_records_detected(self):
        ts = datetime(2025, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        r1 = _make_record(lineage_id="lin-001", timestamp=ts)
        r2 = _make_record(lineage_id="lin-002", timestamp=ts)
        groups = detect_duplicate_lineages([r1, r2])
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_symbol_case_insensitive(self):
        ts = datetime(2025, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        r1 = _make_record(lineage_id="lin-001", symbol="AAPL", timestamp=ts)
        r2 = _make_record(lineage_id="lin-002", symbol="aapl", timestamp=ts)
        groups = detect_duplicate_lineages([r1, r2])
        assert len(groups) == 1

    def test_direction_case_insensitive(self):
        ts = datetime(2025, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        r1 = _make_record(lineage_id="lin-001", direction="BUY", timestamp=ts)
        r2 = _make_record(lineage_id="lin-002", direction="buy", timestamp=ts)
        groups = detect_duplicate_lineages([r1, r2])
        assert len(groups) == 1

    def test_different_symbols_not_duplicates(self):
        ts = datetime(2025, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        r1 = _make_record(lineage_id="lin-001", symbol="AAPL", timestamp=ts)
        r2 = _make_record(lineage_id="lin-002", symbol="MSFT", timestamp=ts)
        groups = detect_duplicate_lineages([r1, r2])
        assert groups == []

    def test_different_directions_not_duplicates(self):
        ts = datetime(2025, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        r1 = _make_record(lineage_id="lin-001", direction="BUY", timestamp=ts)
        r2 = _make_record(lineage_id="lin-002", direction="SHORT", timestamp=ts)
        groups = detect_duplicate_lineages([r1, r2])
        assert groups == []

    def test_different_pm_cycles_not_duplicates(self):
        ts = datetime(2025, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        r1 = _make_record(lineage_id="lin-001", pm_cycle_id="cycle-1", timestamp=ts)
        r2 = _make_record(lineage_id="lin-002", pm_cycle_id="cycle-2", timestamp=ts)
        groups = detect_duplicate_lineages([r1, r2])
        assert groups == []

    def test_timestamps_within_tolerance_are_duplicates(self):
        ts1 = datetime(2025, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        ts2 = ts1 + timedelta(seconds=4)  # within 5s tolerance
        r1 = _make_record(lineage_id="lin-001", timestamp=ts1)
        r2 = _make_record(lineage_id="lin-002", timestamp=ts2)
        groups = detect_duplicate_lineages([r1, r2])
        assert len(groups) == 1

    def test_timestamps_at_boundary_are_duplicates(self):
        ts1 = datetime(2025, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        ts2 = ts1 + TIMESTAMP_TOLERANCE  # exactly at boundary
        r1 = _make_record(lineage_id="lin-001", timestamp=ts1)
        r2 = _make_record(lineage_id="lin-002", timestamp=ts2)
        groups = detect_duplicate_lineages([r1, r2])
        assert len(groups) == 1

    def test_timestamps_beyond_tolerance_not_duplicates(self):
        ts1 = datetime(2025, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        ts2 = ts1 + timedelta(seconds=6)  # beyond 5s tolerance
        r1 = _make_record(lineage_id="lin-001", timestamp=ts1)
        r2 = _make_record(lineage_id="lin-002", timestamp=ts2)
        groups = detect_duplicate_lineages([r1, r2])
        assert groups == []

    def test_three_duplicates_in_one_group(self):
        ts = datetime(2025, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        r1 = _make_record(lineage_id="lin-001", timestamp=ts)
        r2 = _make_record(lineage_id="lin-002", timestamp=ts + timedelta(seconds=1))
        r3 = _make_record(lineage_id="lin-003", timestamp=ts + timedelta(seconds=2))
        groups = detect_duplicate_lineages([r1, r2, r3])
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_multiple_independent_duplicate_groups(self):
        ts = datetime(2025, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        # Group 1: AAPL BUY
        r1 = _make_record(lineage_id="lin-001", symbol="AAPL", direction="BUY", timestamp=ts)
        r2 = _make_record(lineage_id="lin-002", symbol="AAPL", direction="BUY", timestamp=ts)
        # Group 2: MSFT SHORT
        r3 = _make_record(lineage_id="lin-003", symbol="MSFT", direction="SHORT", timestamp=ts)
        r4 = _make_record(lineage_id="lin-004", symbol="MSFT", direction="SHORT", timestamp=ts)
        groups = detect_duplicate_lineages([r1, r2, r3, r4])
        assert len(groups) == 2


class TestDeduplicateExecutionIntents:
    """Tests for deduplicate_execution_intents."""

    def test_empty_input(self):
        canonical, duplicates = deduplicate_execution_intents([])
        assert canonical == []
        assert duplicates == []

    def test_no_duplicates_all_canonical(self):
        ts = datetime(2025, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        r1 = _make_record(lineage_id="lin-001", symbol="AAPL", timestamp=ts)
        r2 = _make_record(lineage_id="lin-002", symbol="MSFT", timestamp=ts)
        canonical, duplicates = deduplicate_execution_intents([r1, r2])
        assert len(canonical) == 2
        assert duplicates == []

    def test_duplicates_keep_lowest_lineage_id(self):
        ts = datetime(2025, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        r1 = _make_record(lineage_id="lin-002", timestamp=ts)
        r2 = _make_record(lineage_id="lin-001", timestamp=ts)
        canonical, duplicates = deduplicate_execution_intents([r1, r2])
        assert len(canonical) == 1
        assert canonical[0].lineage_id == "lin-001"
        assert len(duplicates) == 1
        assert duplicates[0].lineage_id == "lin-002"

    def test_three_duplicates_keep_one_canonical(self):
        ts = datetime(2025, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        r1 = _make_record(lineage_id="lin-003", timestamp=ts)
        r2 = _make_record(lineage_id="lin-001", timestamp=ts)
        r3 = _make_record(lineage_id="lin-002", timestamp=ts)
        canonical, duplicates = deduplicate_execution_intents([r1, r2, r3])
        assert len(canonical) == 1
        assert canonical[0].lineage_id == "lin-001"
        assert len(duplicates) == 2
        dup_ids = {d.lineage_id for d in duplicates}
        assert dup_ids == {"lin-002", "lin-003"}

    def test_mixed_duplicates_and_unique(self):
        ts = datetime(2025, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        # Duplicate group
        r1 = _make_record(lineage_id="lin-001", symbol="AAPL", timestamp=ts)
        r2 = _make_record(lineage_id="lin-002", symbol="AAPL", timestamp=ts)
        # Unique record
        r3 = _make_record(lineage_id="lin-003", symbol="MSFT", timestamp=ts)
        canonical, duplicates = deduplicate_execution_intents([r1, r2, r3])
        assert len(canonical) == 2
        assert len(duplicates) == 1
        canonical_ids = {c.lineage_id for c in canonical}
        assert "lin-001" in canonical_ids  # canonical from dup group
        assert "lin-003" in canonical_ids  # unique record
        assert duplicates[0].lineage_id == "lin-002"

    def test_canonical_plus_duplicates_equals_total(self):
        """Total of canonical + duplicates should equal input count."""
        ts = datetime(2025, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        records = [
            _make_record(lineage_id=f"lin-{i:03d}", timestamp=ts)
            for i in range(5)
        ]
        canonical, duplicates = deduplicate_execution_intents(records)
        assert len(canonical) + len(duplicates) == len(records)


class TestIsDuplicateLineage:
    """Tests for is_duplicate_lineage."""

    def test_no_existing_records(self):
        candidate = _make_record()
        assert is_duplicate_lineage(candidate, []) is False

    def test_matches_existing_record(self):
        ts = datetime(2025, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        candidate = _make_record(lineage_id="lin-002", timestamp=ts)
        existing = [_make_record(lineage_id="lin-001", timestamp=ts)]
        assert is_duplicate_lineage(candidate, existing) is True

    def test_no_match_different_symbol(self):
        ts = datetime(2025, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        candidate = _make_record(lineage_id="lin-002", symbol="MSFT", timestamp=ts)
        existing = [_make_record(lineage_id="lin-001", symbol="AAPL", timestamp=ts)]
        assert is_duplicate_lineage(candidate, existing) is False

    def test_no_match_different_direction(self):
        ts = datetime(2025, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        candidate = _make_record(lineage_id="lin-002", direction="SHORT", timestamp=ts)
        existing = [_make_record(lineage_id="lin-001", direction="BUY", timestamp=ts)]
        assert is_duplicate_lineage(candidate, existing) is False

    def test_no_match_different_cycle(self):
        ts = datetime(2025, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        candidate = _make_record(lineage_id="lin-002", pm_cycle_id="cycle-2", timestamp=ts)
        existing = [_make_record(lineage_id="lin-001", pm_cycle_id="cycle-1", timestamp=ts)]
        assert is_duplicate_lineage(candidate, existing) is False

    def test_no_match_timestamp_too_far(self):
        ts1 = datetime(2025, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        ts2 = ts1 + timedelta(seconds=10)
        candidate = _make_record(lineage_id="lin-002", timestamp=ts2)
        existing = [_make_record(lineage_id="lin-001", timestamp=ts1)]
        assert is_duplicate_lineage(candidate, existing) is False

    def test_matches_one_of_multiple_existing(self):
        ts = datetime(2025, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        candidate = _make_record(
            lineage_id="lin-003", symbol="MSFT", direction="SHORT", timestamp=ts
        )
        existing = [
            _make_record(lineage_id="lin-001", symbol="AAPL", direction="BUY", timestamp=ts),
            _make_record(lineage_id="lin-002", symbol="MSFT", direction="SHORT", timestamp=ts),
        ]
        assert is_duplicate_lineage(candidate, existing) is True
