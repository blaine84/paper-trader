"""Candidate Loading & Correlation for the Decision Replay Agent.

Pure functions for loading replay candidates from source tables, correlating
records across tables, deduplicating, and filtering.

Source tables:
- blocked_trade_candidates (gate-rejected)
- trade_events (PM-rejected decisions, entry events)
- trades (executed entry trades)
- pm_candidates (PM decisions that were adjusted and allowed)

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
See: design.md §core/replay/candidate_sourcer.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from sqlalchemy import text

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifecycle event types excluded from replay (Requirement 1.6)
# ---------------------------------------------------------------------------

LIFECYCLE_EVENT_TYPES: frozenset[str] = frozenset({
    "maintenance",
    "trim",
    "trim_partial",
    "exit",
    "exit_requested",
    "stop_loss_adjustment",
    "tighten_stop",
    "raise_target",
    "setup_exit_force_close",
    "setup_exit_thesis_invalidated",
    "setup_exit_revalidation_failed",
    "setup_exit_revalidated_hold",
    "setup_exit_alert",
    "stop_updated",
    "close",
    "close_full",
    "close_partial",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceReference:
    """A single source record contributing to a replay candidate.

    Attributes:
        source_table: One of "blocked_trade_candidates", "trade_events",
                      "trades", "pm_candidates"
        source_id: The primary key in the source table
        field_contributions: Which candidate fields this source provided
    """
    source_table: str
    source_id: int
    field_contributions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplayCandidate:
    """A single replay candidate representing one historical entry decision.

    Attributes:
        candidate_id: Unique identifier for this candidate (UUID)
        lineage_id: Explicit Candidate_Lineage_ID linking across tables (nullable)
        symbol: Trading symbol
        profile: Portfolio profile
        direction: "LONG" or "SHORT"
        setup_type: Analyst setup classification (nullable)
        entry_price: Entry price as Decimal (nullable if geometry-incomplete)
        stop_price: Stop price as Decimal (nullable if geometry-incomplete)
        target_price: Target price as Decimal (nullable if geometry-incomplete)
        quantity: Position size as Decimal (nullable)
        entry_timestamp: When the entry decision was made
        original_decision: "allow" | "reject" | "adjusted"
        original_gate: Gate that produced the decision (nullable)
        original_reason_code: Reason code from the gate (nullable)
        source_records: All contributing source records
        geometry_complete: True when entry/stop/target are all present
        correlation_warnings: Warnings from ambiguous correlation
    """
    candidate_id: str
    lineage_id: str | None
    symbol: str
    profile: str
    direction: str
    setup_type: str | None
    entry_price: Decimal | None
    stop_price: Decimal | None
    target_price: Decimal | None
    quantity: Decimal | None
    entry_timestamp: datetime
    original_decision: str
    original_gate: str | None
    original_reason_code: str | None
    source_records: tuple[SourceReference, ...]
    geometry_complete: bool
    correlation_warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Geometry hash computation
# ---------------------------------------------------------------------------

def compute_geometry_hash(
    entry_price: Decimal | None,
    stop_price: Decimal | None,
    target_price: Decimal | None,
    tick_size: Decimal = Decimal("0.01"),
) -> str:
    """Compute canonical geometry hash using tick-normalized Decimal with SHA-256.

    Normalizes each price to the specified tick size using ROUND_HALF_UP,
    then produces a stable SHA-256 hash for comparison. This avoids
    floating-point comparison issues.

    Returns empty string if any price is None (geometry incomplete).

    This matches the implementation in utils/decision_snapshot.py.
    """
    if entry_price is None or stop_price is None or target_price is None:
        return ""

    entry_norm = _tick_normalize(entry_price, tick_size)
    stop_norm = _tick_normalize(stop_price, tick_size)
    target_norm = _tick_normalize(target_price, tick_size)

    canonical = json.dumps(
        {
            "entry": str(entry_norm),
            "stop": str(stop_norm),
            "target": str(target_norm),
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _tick_normalize(price: Decimal, tick_size: Decimal) -> Decimal:
    """Normalize a price to the nearest tick using ROUND_HALF_UP."""
    if tick_size <= 0:
        return price
    return price.quantize(tick_size, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Candidate loading
# ---------------------------------------------------------------------------

def load_candidates(
    session,
    *,
    date_range: tuple[datetime, datetime] | None = None,
    filters: dict | None = None,
) -> list[ReplayCandidate]:
    """Load replay candidates from all source tables.

    Queries blocked_trade_candidates, trade_events, trades, and pm_candidates.
    Applies date_range and filter criteria. Excludes lifecycle events
    (maintenance, trim, exit, stop-loss adjustment) per Requirement 1.6.

    Args:
        session: SQLAlchemy session or connection
        date_range: Optional (start, end) datetime tuple for filtering
        filters: Optional dict with keys: profile, symbol, setup_type, gate,
                 original_decision, outcome

    Returns:
        List of ReplayCandidate instances (not yet deduplicated)
    """
    candidates: list[ReplayCandidate] = []

    candidates.extend(_load_from_blocked_trade_candidates(session, date_range, filters))
    candidates.extend(_load_from_trade_events(session, date_range, filters))
    candidates.extend(_load_from_trades(session, date_range, filters))
    candidates.extend(_load_from_pm_candidates(session, date_range, filters))

    # Apply remaining filters that are cross-table
    candidates = _apply_filters(candidates, filters)

    return candidates


def _load_from_blocked_trade_candidates(
    session,
    date_range: tuple[datetime, datetime] | None,
    filters: dict | None,
) -> list[ReplayCandidate]:
    """Load candidates from blocked_trade_candidates table (gate-rejected)."""
    query = """
        SELECT id, symbol, profile, direction, setup_type,
               entry_price, stop_price, target_price, quantity,
               created_at, blocked_by, reason_code, action,
               candidate_lineage_id
        FROM blocked_trade_candidates
        WHERE 1=1
    """
    params: dict[str, Any] = {}

    if date_range:
        query += " AND created_at >= :start_date AND created_at <= :end_date"
        params["start_date"] = date_range[0]
        params["end_date"] = date_range[1]

    if filters:
        if filters.get("symbol"):
            query += " AND symbol = :symbol"
            params["symbol"] = filters["symbol"]
        if filters.get("profile"):
            query += " AND profile = :profile"
            params["profile"] = filters["profile"]
        if filters.get("setup_type"):
            query += " AND setup_type = :setup_type"
            params["setup_type"] = filters["setup_type"]

    result = session.execute(text(query), params)
    rows = result.fetchall()

    candidates = []
    for row in rows:
        row_dict = row._mapping

        # Exclude lifecycle events (Requirement 1.6)
        action = row_dict.get("action", "") or ""
        if action.lower() in LIFECYCLE_EVENT_TYPES:
            continue

        entry_price = _to_decimal(row_dict.get("entry_price"))
        stop_price = _to_decimal(row_dict.get("stop_price"))
        target_price = _to_decimal(row_dict.get("target_price"))
        geometry_complete = all(
            p is not None for p in (entry_price, stop_price, target_price)
        )

        direction = row_dict.get("direction") or "LONG"
        profile = row_dict.get("profile") or "moderate"

        candidate = ReplayCandidate(
            candidate_id=str(uuid.uuid4()),
            lineage_id=row_dict.get("candidate_lineage_id"),
            symbol=row_dict["symbol"],
            profile=profile,
            direction=direction.upper(),
            setup_type=row_dict.get("setup_type"),
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=_to_decimal(row_dict.get("quantity")),
            entry_timestamp=_parse_timestamp(row_dict["created_at"]),
            original_decision="reject",
            original_gate=row_dict.get("blocked_by"),
            original_reason_code=row_dict.get("reason_code"),
            source_records=(
                SourceReference(
                    source_table="blocked_trade_candidates",
                    source_id=row_dict["id"],
                    field_contributions=(
                        "symbol", "profile", "direction", "setup_type",
                        "entry_price", "stop_price", "target_price",
                        "quantity", "entry_timestamp", "original_decision",
                        "original_gate", "original_reason_code",
                    ),
                ),
            ),
            geometry_complete=geometry_complete,
        )
        candidates.append(candidate)

    return candidates


def _load_from_trade_events(
    session,
    date_range: tuple[datetime, datetime] | None,
    filters: dict | None,
) -> list[ReplayCandidate]:
    """Load candidates from trade_events table (PM-rejected decisions).

    Focuses on entry-related events: pm_reject, pipeline_gate_rejected,
    offered events with rejection decisions.
    """
    # Only load entry-related rejection events (not lifecycle)
    query = """
        SELECT id, symbol, profile, timestamp, event_type, agent,
               payload_json, candidate_lineage_id
        FROM trade_events
        WHERE event_type IN (
            'pm_reject', 'pm_not_selected',
            'pipeline_gate_rejected', 'pipeline_sizing_rejected'
        )
    """
    params: dict[str, Any] = {}

    if date_range:
        query += " AND timestamp >= :start_date AND timestamp <= :end_date"
        params["start_date"] = date_range[0]
        params["end_date"] = date_range[1]

    if filters:
        if filters.get("symbol"):
            query += " AND symbol = :symbol"
            params["symbol"] = filters["symbol"]
        if filters.get("profile"):
            query += " AND profile = :profile"
            params["profile"] = filters["profile"]

    result = session.execute(text(query), params)
    rows = result.fetchall()

    candidates = []
    for row in rows:
        row_dict = row._mapping

        # Skip lifecycle events (Requirement 1.6)
        event_type = row_dict.get("event_type", "")
        if event_type in LIFECYCLE_EVENT_TYPES:
            continue

        payload = _parse_json(row_dict.get("payload_json"))

        # Extract geometry from payload if available
        entry_price = _to_decimal(payload.get("entry_price"))
        stop_price = _to_decimal(payload.get("stop_price"))
        target_price = _to_decimal(payload.get("target_price"))
        geometry_complete = all(
            p is not None for p in (entry_price, stop_price, target_price)
        )

        direction = payload.get("direction", "LONG")
        setup_type = payload.get("setup_type")
        profile = row_dict.get("profile") or payload.get("profile", "moderate")
        symbol = row_dict.get("symbol") or payload.get("symbol", "")

        if not symbol:
            continue  # Cannot create candidate without symbol

        # Determine original decision based on event type
        original_decision = "reject"
        original_gate = payload.get("gate_name") or payload.get("blocked_by")
        reason_code = payload.get("reason_code") or payload.get("reason")

        candidate = ReplayCandidate(
            candidate_id=str(uuid.uuid4()),
            lineage_id=row_dict.get("candidate_lineage_id"),
            symbol=symbol,
            profile=profile,
            direction=direction.upper() if direction else "LONG",
            setup_type=setup_type,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=_to_decimal(payload.get("quantity")),
            entry_timestamp=_parse_timestamp(row_dict["timestamp"]),
            original_decision=original_decision,
            original_gate=original_gate,
            original_reason_code=reason_code,
            source_records=(
                SourceReference(
                    source_table="trade_events",
                    source_id=row_dict["id"],
                    field_contributions=(
                        "symbol", "profile", "direction", "setup_type",
                        "entry_price", "stop_price", "target_price",
                        "quantity", "entry_timestamp", "original_decision",
                        "original_gate", "original_reason_code",
                    ),
                ),
            ),
            geometry_complete=geometry_complete,
        )
        candidates.append(candidate)

    return candidates


def _load_from_trades(
    session,
    date_range: tuple[datetime, datetime] | None,
    filters: dict | None,
) -> list[ReplayCandidate]:
    """Load candidates from trades table (executed entry trades)."""
    query = """
        SELECT id, symbol, profile, direction, setup_type,
               entry_price, stop_price, target_price, quantity,
               entry_time, candidate_lineage_id
        FROM trades
        WHERE status IN ('open', 'closed')
    """
    params: dict[str, Any] = {}

    if date_range:
        query += " AND entry_time >= :start_date AND entry_time <= :end_date"
        params["start_date"] = date_range[0]
        params["end_date"] = date_range[1]

    if filters:
        if filters.get("symbol"):
            query += " AND symbol = :symbol"
            params["symbol"] = filters["symbol"]
        if filters.get("profile"):
            query += " AND profile = :profile"
            params["profile"] = filters["profile"]
        if filters.get("setup_type"):
            query += " AND setup_type = :setup_type"
            params["setup_type"] = filters["setup_type"]

    result = session.execute(text(query), params)
    rows = result.fetchall()

    candidates = []
    for row in rows:
        row_dict = row._mapping

        entry_price = _to_decimal(row_dict.get("entry_price"))
        stop_price = _to_decimal(row_dict.get("stop_price"))
        target_price = _to_decimal(row_dict.get("target_price"))
        geometry_complete = all(
            p is not None for p in (entry_price, stop_price, target_price)
        )

        candidate = ReplayCandidate(
            candidate_id=str(uuid.uuid4()),
            lineage_id=row_dict.get("candidate_lineage_id"),
            symbol=row_dict["symbol"],
            profile=row_dict.get("profile") or "moderate",
            direction=(row_dict.get("direction") or "LONG").upper(),
            setup_type=row_dict.get("setup_type"),
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=_to_decimal(row_dict.get("quantity")),
            entry_timestamp=_parse_timestamp(row_dict["entry_time"]),
            original_decision="allow",
            original_gate=None,
            original_reason_code=None,
            source_records=(
                SourceReference(
                    source_table="trades",
                    source_id=row_dict["id"],
                    field_contributions=(
                        "symbol", "profile", "direction", "setup_type",
                        "entry_price", "stop_price", "target_price",
                        "quantity", "entry_timestamp", "original_decision",
                    ),
                ),
            ),
            geometry_complete=geometry_complete,
        )
        candidates.append(candidate)

    return candidates


def _load_from_pm_candidates(
    session,
    date_range: tuple[datetime, datetime] | None,
    filters: dict | None,
) -> list[ReplayCandidate]:
    """Load candidates from pm_candidates table (adjusted and allowed)."""
    query = """
        SELECT id, candidate_id, symbol, profile_id, direction, setup_type,
               entry_price, stop_price, target_price, state,
               created_at, rejection_reason, candidate_lineage_id
        FROM pm_candidates
        WHERE state NOT IN ('expired')
    """
    params: dict[str, Any] = {}

    if date_range:
        query += " AND created_at >= :start_date AND created_at <= :end_date"
        params["start_date"] = date_range[0]
        params["end_date"] = date_range[1]

    if filters:
        if filters.get("symbol"):
            query += " AND symbol = :symbol"
            params["symbol"] = filters["symbol"]
        if filters.get("profile"):
            query += " AND profile_id = :profile"
            params["profile"] = filters["profile"]
        if filters.get("setup_type"):
            query += " AND setup_type = :setup_type"
            params["setup_type"] = filters["setup_type"]

    result = session.execute(text(query), params)
    rows = result.fetchall()

    candidates = []
    for row in rows:
        row_dict = row._mapping

        state = row_dict.get("state", "")

        # Determine original decision from state
        if state in ("consumed", "reserved"):
            original_decision = "allow"
        elif state in ("rejected", "not_selected"):
            original_decision = "reject"
        else:
            original_decision = "adjusted"

        entry_price = _to_decimal(row_dict.get("entry_price"))
        stop_price = _to_decimal(row_dict.get("stop_price"))
        target_price = _to_decimal(row_dict.get("target_price"))
        geometry_complete = all(
            p is not None for p in (entry_price, stop_price, target_price)
        )

        candidate = ReplayCandidate(
            candidate_id=str(uuid.uuid4()),
            lineage_id=row_dict.get("candidate_lineage_id"),
            symbol=row_dict["symbol"],
            profile=row_dict.get("profile_id") or "moderate",
            direction=(row_dict.get("direction") or "LONG").upper(),
            setup_type=row_dict.get("setup_type"),
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity=None,  # pm_candidates doesn't store quantity
            entry_timestamp=_parse_timestamp(row_dict["created_at"]),
            original_decision=original_decision,
            original_gate=None,
            original_reason_code=row_dict.get("rejection_reason"),
            source_records=(
                SourceReference(
                    source_table="pm_candidates",
                    source_id=row_dict["id"],
                    field_contributions=(
                        "symbol", "profile", "direction", "setup_type",
                        "entry_price", "stop_price", "target_price",
                        "entry_timestamp", "original_decision",
                    ),
                ),
            ),
            geometry_complete=geometry_complete,
        )
        candidates.append(candidate)

    return candidates


# ---------------------------------------------------------------------------
# Correlation and deduplication
# ---------------------------------------------------------------------------

def correlate_and_deduplicate(
    candidates: list[ReplayCandidate],
) -> list[ReplayCandidate]:
    """Merge candidates representing the same decision point.

    Precedence (Requirement 1.3):
        (a) Match on explicit Candidate_Lineage_ID when present
        (b) Fallback: match on (symbol, profile, direction, setup_type,
            timestamp ±60s, geometry_hash)
        (c) Ambiguous fallback: keep as separate candidates with correlation warning

    Requirement 1.4: Unambiguous matches merge into one replay candidate
    and are counted only once in aggregate reports.
    """
    if not candidates:
        return []

    # Phase 1: Group by lineage_id (highest precedence)
    lineage_groups: dict[str, list[ReplayCandidate]] = {}
    no_lineage: list[ReplayCandidate] = []

    for c in candidates:
        if c.lineage_id:
            lineage_groups.setdefault(c.lineage_id, []).append(c)
        else:
            no_lineage.append(c)

    merged: list[ReplayCandidate] = []

    # Merge lineage-matched groups
    for lineage_id, group in lineage_groups.items():
        if len(group) == 1:
            merged.append(group[0])
        else:
            merged.append(_merge_candidates(group))

    # Phase 2: Fallback matching for candidates without lineage_id
    # Build fallback key: (symbol, profile, direction, setup_type, geometry_hash)
    # with timestamp ±60s tolerance
    fallback_groups: list[list[ReplayCandidate]] = []
    assigned: set[int] = set()  # track indices already grouped

    for i, ci in enumerate(no_lineage):
        if i in assigned:
            continue

        group = [ci]
        assigned.add(i)

        ci_geom_hash = compute_geometry_hash(
            ci.entry_price, ci.stop_price, ci.target_price
        )

        for j, cj in enumerate(no_lineage):
            if j in assigned or j <= i:
                continue

            if not _fallback_keys_match(ci, cj, ci_geom_hash):
                continue

            # Check timestamp within ±60s
            time_diff = abs(
                (ci.entry_timestamp - cj.entry_timestamp).total_seconds()
            )
            if time_diff <= 60:
                group.append(cj)
                assigned.add(j)

        fallback_groups.append(group)

    # Phase 3: Process fallback groups
    for group in fallback_groups:
        if len(group) == 1:
            merged.append(group[0])
        else:
            # Check for ambiguity: if records differ in fields beyond the
            # deduplication key, attach a correlation warning
            if _is_ambiguous_match(group):
                # Keep as separate candidates with warnings
                for c in group:
                    warned = ReplayCandidate(
                        candidate_id=c.candidate_id,
                        lineage_id=c.lineage_id,
                        symbol=c.symbol,
                        profile=c.profile,
                        direction=c.direction,
                        setup_type=c.setup_type,
                        entry_price=c.entry_price,
                        stop_price=c.stop_price,
                        target_price=c.target_price,
                        quantity=c.quantity,
                        entry_timestamp=c.entry_timestamp,
                        original_decision=c.original_decision,
                        original_gate=c.original_gate,
                        original_reason_code=c.original_reason_code,
                        source_records=c.source_records,
                        geometry_complete=c.geometry_complete,
                        correlation_warnings=(
                            *c.correlation_warnings,
                            f"Ambiguous fallback match: {len(group)} records share "
                            f"(symbol={c.symbol}, profile={c.profile}, "
                            f"direction={c.direction}, setup_type={c.setup_type}) "
                            f"within 60s but differ in other fields",
                        ),
                    )
                    merged.append(warned)
            else:
                # Unambiguous match — merge
                merged.append(_merge_candidates(group))

    # Also check for lineage candidates that match no-lineage candidates
    # (a lineage-matched candidate supersedes any fallback match)
    # Already handled: lineage groups were processed first

    return merged


def _fallback_keys_match(
    a: ReplayCandidate,
    b: ReplayCandidate,
    a_geom_hash: str | None = None,
) -> bool:
    """Check if two candidates match on fallback deduplication key.

    Fallback key: (symbol, profile, direction, setup_type, geometry_hash)
    """
    if a.symbol != b.symbol:
        return False
    if a.profile != b.profile:
        return False
    if a.direction != b.direction:
        return False
    if a.setup_type != b.setup_type:
        return False

    # Compare geometry hashes
    a_hash = a_geom_hash or compute_geometry_hash(
        a.entry_price, a.stop_price, a.target_price
    )
    b_hash = compute_geometry_hash(
        b.entry_price, b.stop_price, b.target_price
    )

    # Both empty = both geometry-incomplete, treat as potentially matching
    # One empty and one not = not a match
    if a_hash and b_hash:
        return a_hash == b_hash
    elif not a_hash and not b_hash:
        # Both incomplete — match on other key fields (already checked above)
        return True
    else:
        return False


def _is_ambiguous_match(group: list[ReplayCandidate]) -> bool:
    """Determine if a fallback-matched group is ambiguous.

    Ambiguous: records share the deduplication key but differ in fields
    that suggest they might be different decisions (e.g., different
    original_decision, different gates, significantly different quantities).
    """
    if len(group) <= 1:
        return False

    # If original decisions differ AND both have the same source table,
    # it's likely ambiguous
    decisions = {c.original_decision for c in group}
    source_tables = set()
    for c in group:
        for sr in c.source_records:
            source_tables.add(sr.source_table)

    # If they come from the same source table with different decisions, ambiguous
    if len(decisions) > 1 and len(source_tables) == 1:
        return True

    # If they come from different source tables but have contradicting info
    # (e.g., different reason codes that can't be reconciled), that's expected
    # and should be merged (one is the rejection, the other might be related).
    # Only flag if truly conflicting.

    # Different quantities (more than 20% divergence) suggests different decisions
    quantities = [c.quantity for c in group if c.quantity is not None]
    if len(quantities) >= 2:
        min_q = min(quantities)
        max_q = max(quantities)
        if min_q > 0 and max_q / min_q > Decimal("1.2"):
            return True

    return False


def _merge_candidates(group: list[ReplayCandidate]) -> ReplayCandidate:
    """Merge multiple candidates into one.

    Picks the best available data from the group:
    - Prefers geometry-complete records
    - Combines source_records from all contributors
    - Uses the earliest timestamp
    """
    if len(group) == 1:
        return group[0]

    # Sort: prefer geometry-complete, then by timestamp (earliest)
    sorted_group = sorted(
        group,
        key=lambda c: (not c.geometry_complete, c.entry_timestamp),
    )
    primary = sorted_group[0]

    # Collect all source records
    all_sources: list[SourceReference] = []
    for c in group:
        all_sources.extend(c.source_records)

    # Use earliest timestamp
    earliest_ts = min(c.entry_timestamp for c in group)

    # Prefer non-None values from the primary, falling back to others
    entry_price = primary.entry_price
    stop_price = primary.stop_price
    target_price = primary.target_price
    quantity = primary.quantity

    for c in sorted_group[1:]:
        if entry_price is None and c.entry_price is not None:
            entry_price = c.entry_price
        if stop_price is None and c.stop_price is not None:
            stop_price = c.stop_price
        if target_price is None and c.target_price is not None:
            target_price = c.target_price
        if quantity is None and c.quantity is not None:
            quantity = c.quantity

    geometry_complete = all(
        p is not None for p in (entry_price, stop_price, target_price)
    )

    # Prefer lineage_id from any record that has it
    lineage_id = primary.lineage_id
    for c in group:
        if c.lineage_id:
            lineage_id = c.lineage_id
            break

    return ReplayCandidate(
        candidate_id=primary.candidate_id,
        lineage_id=lineage_id,
        symbol=primary.symbol,
        profile=primary.profile,
        direction=primary.direction,
        setup_type=primary.setup_type or next(
            (c.setup_type for c in group if c.setup_type), None
        ),
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        quantity=quantity,
        entry_timestamp=earliest_ts,
        original_decision=primary.original_decision,
        original_gate=primary.original_gate or next(
            (c.original_gate for c in group if c.original_gate), None
        ),
        original_reason_code=primary.original_reason_code or next(
            (c.original_reason_code for c in group if c.original_reason_code), None
        ),
        source_records=tuple(all_sources),
        geometry_complete=geometry_complete,
    )


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def _apply_filters(
    candidates: list[ReplayCandidate],
    filters: dict | None,
) -> list[ReplayCandidate]:
    """Apply cross-table filters that couldn't be done at the SQL level.

    Supported filters: date_range (already applied), profile, symbol,
    setup_type, gate, original_decision, outcome.
    """
    if not filters:
        return candidates

    result = candidates

    # gate filter
    gate_filter = filters.get("gate")
    if gate_filter:
        result = [c for c in result if c.original_gate == gate_filter]

    # original_decision filter
    decision_filter = filters.get("original_decision")
    if decision_filter:
        result = [c for c in result if c.original_decision == decision_filter]

    # setup_type filter (for records that didn't support SQL-level filtering)
    setup_filter = filters.get("setup_type")
    if setup_filter:
        result = [c for c in result if c.setup_type == setup_filter]

    # symbol filter (catch-all for sources without SQL-level support)
    symbol_filter = filters.get("symbol")
    if symbol_filter:
        result = [c for c in result if c.symbol == symbol_filter]

    # profile filter (catch-all)
    profile_filter = filters.get("profile")
    if profile_filter:
        result = [c for c in result if c.profile == profile_filter]

    return result


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _to_decimal(value: Any) -> Decimal | None:
    """Convert a numeric value to Decimal, returning None if not convertible."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _parse_timestamp(value: Any) -> datetime:
    """Parse a timestamp value into a datetime object."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        # Handle ISO-8601 format and common SQLite formats
        for fmt in (
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    # Fallback: return current time (should not happen in practice)
    log.warning("Could not parse timestamp: %r, using current time", value)
    return datetime.utcnow()


def _parse_json(value: Any) -> dict:
    """Parse a JSON string into a dict, returning empty dict on failure."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}
