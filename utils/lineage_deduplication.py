"""Lineage Deduplication Logic.

Detects and handles duplicate audit records that share the same lineage_id,
symbol, direction, and originating timestamp within the same PM cycle.
These are treated as representations of one trade opportunity, not separate
execution intents.

Requirements: 1.5
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LineageRecord:
    """A record representing a candidate lineage for dedup comparison."""

    lineage_id: str
    symbol: str
    direction: str
    timestamp: datetime
    pm_cycle_id: str


# Timestamp tolerance for "same originating timestamp" (within same PM cycle).
# Records within this window are considered the same trade opportunity.
TIMESTAMP_TOLERANCE = timedelta(seconds=5)


def _matches_dedup_key(a: LineageRecord, b: LineageRecord) -> bool:
    """Check if two records match on all dedup criteria.

    Two records are duplicates if they share:
    - Same pm_cycle_id (case-sensitive, these are UUIDs)
    - Same symbol (case-insensitive)
    - Same direction (case-insensitive)
    - Originating timestamps within TIMESTAMP_TOLERANCE of each other
    """
    if a.pm_cycle_id != b.pm_cycle_id:
        return False
    if a.symbol.upper() != b.symbol.upper():
        return False
    if a.direction.upper() != b.direction.upper():
        return False
    time_diff = abs((a.timestamp - b.timestamp).total_seconds())
    if time_diff > TIMESTAMP_TOLERANCE.total_seconds():
        return False
    return True


def detect_duplicate_lineages(records: list[LineageRecord]) -> list[list[LineageRecord]]:
    """Detect groups of duplicate lineage records within the same PM cycle.

    Two or more records are considered duplicates if they share:
    - Same pm_cycle_id
    - Same symbol (case-insensitive)
    - Same direction (case-insensitive)
    - Originating timestamps within TIMESTAMP_TOLERANCE of each other

    Returns a list of duplicate groups (each group has 2+ records).
    Records that are not duplicates are excluded from the result.
    """
    if not records:
        return []

    # Build groups using union-find style grouping.
    # For each record, check if it belongs to an existing group.
    groups: list[list[LineageRecord]] = []

    for record in records:
        merged = False
        for group in groups:
            # A record belongs to a group if it matches ANY member of that group.
            # Since all members of a group already match each other transitively
            # via the pm_cycle_id + symbol + direction exact keys, we only need
            # to check timestamp proximity against any member.
            if _matches_dedup_key(record, group[0]):
                group.append(record)
                merged = True
                break
        if not merged:
            groups.append([record])

    # Return only groups with 2+ records (actual duplicates)
    return [group for group in groups if len(group) >= 2]


def deduplicate_execution_intents(
    records: list[LineageRecord],
) -> tuple[list[LineageRecord], list[LineageRecord]]:
    """Separate records into unique execution intents and duplicates.

    For each duplicate group, keeps the FIRST record (lowest lineage_id
    lexicographically as a tiebreaker) as the canonical representative.

    Returns (canonical_records, duplicate_records).
    Duplicate records should NOT generate separate execution intents.
    """
    if not records:
        return [], []

    duplicate_groups = detect_duplicate_lineages(records)

    # Collect all records that are part of any duplicate group
    duplicate_record_ids: set[tuple[str, str]] = set()  # (lineage_id, pm_cycle_id)
    canonical_from_groups: list[LineageRecord] = []
    duplicates: list[LineageRecord] = []

    for group in duplicate_groups:
        # Sort group by lineage_id lexicographically to pick canonical
        sorted_group = sorted(group, key=lambda r: r.lineage_id)
        canonical = sorted_group[0]
        canonical_from_groups.append(canonical)

        for record in sorted_group[1:]:
            duplicates.append(record)
            duplicate_record_ids.add((record.lineage_id, record.pm_cycle_id))

        # Mark all group members so we know which are accounted for
        for record in sorted_group:
            duplicate_record_ids.add((record.lineage_id, record.pm_cycle_id))

    # Build canonical list: records NOT in any duplicate group + canonical from each group
    canonical_records: list[LineageRecord] = []
    for record in records:
        if (record.lineage_id, record.pm_cycle_id) not in duplicate_record_ids:
            canonical_records.append(record)

    canonical_records.extend(canonical_from_groups)

    # Remove canonical entries from duplicates set (they were added to both)
    # Re-build duplicates: only non-canonical members of duplicate groups
    duplicates = []
    for group in duplicate_groups:
        sorted_group = sorted(group, key=lambda r: r.lineage_id)
        for record in sorted_group[1:]:
            duplicates.append(record)

    logger.debug(
        "Lineage deduplication: %d records -> %d canonical, %d duplicates",
        len(records),
        len(canonical_records),
        len(duplicates),
    )

    return canonical_records, duplicates


def is_duplicate_lineage(
    candidate: LineageRecord,
    existing_records: list[LineageRecord],
) -> bool:
    """Check if a candidate record is a duplicate of any existing record.

    Used at insertion time to prevent duplicate execution intents.
    Returns True if the candidate matches any existing record on all
    dedup criteria (pm_cycle_id, symbol, direction, timestamp within tolerance).
    """
    for existing in existing_records:
        if _matches_dedup_key(candidate, existing):
            return True
    return False
