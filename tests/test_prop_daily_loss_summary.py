"""
Property-based tests for daily loss summary (Properties 14, 15, 16).

Feature: candidate-blocker-mitigation

Tests that:
- Property 14: For any set of candidate events for a trading day (across all cycles),
  the DailyLossSummary counts SHALL equal the number of events of each corresponding
  type in pm_candidate_events for that trade_date. The pm_rejected_by_reason breakdown
  SHALL sum to pm_rejected.

- Property 15: For any pm_rejected_by_reason distribution, top_blocking_reasons SHALL
  contain at most 3 reason codes sorted by descending count, with alphabetical ordering
  to break ties. If fewer than 3 distinct codes exist, only available codes SHALL be reported.

- Property 16: For any daily summary where executed == 0, the dominant_blocker_stage
  field SHALL identify the pipeline stage that rejected the most candidates, selected
  from the fixed set: no_signals, pm_rejection, gate_rejection, sizing_rejection,
  execution_failure.

**Validates: Requirements 8.1, 8.2, 8.3, 5.6**
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from hypothesis import given, settings, strategies as st, assume
from sqlalchemy import create_engine, text

from utils.daily_loss_summary import (
    DailyLossSummary,
    compute_daily_loss_summary,
    _compute_top_blocking_reasons,
    _compute_dominant_blocker_stage,
)


# ---------------------------------------------------------------------------
# Database setup helpers
# ---------------------------------------------------------------------------


def _create_test_engine():
    """Create in-memory SQLite with pm_candidate_events table for testing."""
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        conn.execute(text("""
            CREATE TABLE pm_candidate_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                cycle_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_data TEXT,
                created_at TEXT NOT NULL,
                candidate_type TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE daily_loss_summaries (
                id INTEGER PRIMARY KEY,
                trade_date VARCHAR(10) NOT NULL,
                profile_id VARCHAR(64) NOT NULL,
                signals_seen INTEGER NOT NULL DEFAULT 0,
                candidates_built INTEGER NOT NULL DEFAULT 0,
                preflight_failed INTEGER NOT NULL DEFAULT 0,
                offered_to_pm INTEGER NOT NULL DEFAULT 0,
                pm_rejected INTEGER NOT NULL DEFAULT 0,
                pm_rejected_by_reason_json TEXT,
                pm_accepted INTEGER NOT NULL DEFAULT 0,
                gate_sizing_rejected INTEGER NOT NULL DEFAULT 0,
                execution_failed INTEGER NOT NULL DEFAULT 0,
                executed INTEGER NOT NULL DEFAULT 0,
                lifecycle_incomplete INTEGER NOT NULL DEFAULT 0,
                top_blocking_reasons_json TEXT,
                dominant_blocker_stage VARCHAR(64),
                error_indication TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(trade_date, profile_id)
            )
        """))
    return eng


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Event types that map to specific summary count fields
_EVENT_TYPES_FOR_COUNTS = [
    "preflight_failed",
    "preflight_passed",
    "pm_reject",
    "pm_accept",
    "gate_fail",
    "sizing_fail",
    "execution_failed",
    "lifecycle_incomplete",
]

# Reason codes for pm_reject events
_REASON_CODES = [
    "low_confidence",
    "mixed_timeframes",
    "late_day",
    "hostile_breadth",
    "thin_volume",
    "risk_off_conflict",
    "profile_rule_failed",
    "exposure_conflict",
    "liquidity_or_spread",
    "event_risk",
    "other",
]

# Strategy for generating a list of events to insert
event_list_st = st.lists(
    st.tuples(
        st.sampled_from(_EVENT_TYPES_FOR_COUNTS),
        st.sampled_from(_REASON_CODES),  # reason_code (used only for pm_reject)
    ),
    min_size=0,
    max_size=50,
)

# Strategy for pm_rejected_by_reason dicts (Property 15)
reason_dict_st = st.dictionaries(
    keys=st.sampled_from(_REASON_CODES),
    values=st.integers(min_value=1, max_value=100),
    min_size=0,
    max_size=11,
)

# Strategy for counts used in dominant_blocker_stage (Property 16)
count_st = st.integers(min_value=0, max_value=200)


# ---------------------------------------------------------------------------
# Property 14: Daily loss summary count accuracy
# Validates: Requirements 8.1, 5.6
#
# For any set of candidate events for a trading day (across all cycles), the
# DailyLossSummary counts SHALL equal the number of events of each corresponding
# type in pm_candidate_events for that trade_date. The pm_rejected_by_reason
# breakdown SHALL sum to pm_rejected.
# ---------------------------------------------------------------------------


@given(events=event_list_st)
@settings(max_examples=200)
def test_daily_loss_summary_count_accuracy(events):
    """Property 14: Daily loss summary count accuracy.

    For any set of candidate events inserted into pm_candidate_events for a
    trading day, the computed DailyLossSummary counts match the actual event
    counts, and pm_rejected_by_reason sums to pm_rejected.

    Strategy: Generate random events (with various event_types), insert into
    in-memory DB, call compute_daily_loss_summary(), verify counts match.

    **Validates: Requirements 8.1, 8.2**
    """
    engine = _create_test_engine()
    trade_date = "2024-06-15"
    profile_id = "moderate"
    created_at = f"{trade_date}T10:00:00+00:00"

    # Insert events into the database
    with engine.begin() as conn:
        for i, (event_type, reason_code) in enumerate(events):
            candidate_id = str(uuid.uuid4())
            cycle_id = f"cycle-{i % 3}"  # Spread across multiple cycles

            # Build event_data (only pm_reject needs reason_code)
            if event_type == "pm_reject":
                event_data = json.dumps({"rejection_reason_code": reason_code})
            else:
                event_data = json.dumps({})

            conn.execute(text("""
                INSERT INTO pm_candidate_events
                (candidate_id, cycle_id, profile_id, event_type, event_data, created_at)
                VALUES (:cid, :cycle_id, :profile_id, :event_type, :event_data, :created_at)
            """), {
                "cid": candidate_id,
                "cycle_id": cycle_id,
                "profile_id": profile_id,
                "event_type": event_type,
                "event_data": event_data,
                "created_at": created_at,
            })

    # Compute the summary
    summary = compute_daily_loss_summary(engine, trade_date, profile_id)

    # Count expected values from the event list
    expected_preflight_failed = sum(1 for et, _ in events if et == "preflight_failed")
    expected_offered_to_pm = sum(1 for et, _ in events if et == "preflight_passed")
    expected_pm_rejected = sum(1 for et, _ in events if et == "pm_reject")
    expected_pm_accepted = sum(1 for et, _ in events if et == "pm_accept")
    expected_gate_sizing_rejected = sum(
        1 for et, _ in events if et in ("gate_fail", "sizing_fail")
    )
    expected_execution_failed = sum(1 for et, _ in events if et == "execution_failed")
    expected_lifecycle_incomplete = sum(
        1 for et, _ in events if et == "lifecycle_incomplete"
    )

    # INVARIANT: counts match event type occurrences
    assert summary.preflight_failed == expected_preflight_failed, (
        f"preflight_failed: expected {expected_preflight_failed}, got {summary.preflight_failed}"
    )
    assert summary.offered_to_pm == expected_offered_to_pm, (
        f"offered_to_pm: expected {expected_offered_to_pm}, got {summary.offered_to_pm}"
    )
    assert summary.pm_rejected == expected_pm_rejected, (
        f"pm_rejected: expected {expected_pm_rejected}, got {summary.pm_rejected}"
    )
    assert summary.pm_accepted == expected_pm_accepted, (
        f"pm_accepted: expected {expected_pm_accepted}, got {summary.pm_accepted}"
    )
    assert summary.gate_sizing_rejected == expected_gate_sizing_rejected, (
        f"gate_sizing_rejected: expected {expected_gate_sizing_rejected}, "
        f"got {summary.gate_sizing_rejected}"
    )
    assert summary.execution_failed == expected_execution_failed, (
        f"execution_failed: expected {expected_execution_failed}, "
        f"got {summary.execution_failed}"
    )
    assert summary.lifecycle_incomplete == expected_lifecycle_incomplete, (
        f"lifecycle_incomplete: expected {expected_lifecycle_incomplete}, "
        f"got {summary.lifecycle_incomplete}"
    )

    # INVARIANT: pm_rejected_by_reason values sum to pm_rejected
    reason_sum = sum(summary.pm_rejected_by_reason.values())
    assert reason_sum == summary.pm_rejected, (
        f"pm_rejected_by_reason sum ({reason_sum}) != pm_rejected ({summary.pm_rejected}). "
        f"Breakdown: {summary.pm_rejected_by_reason}"
    )

    # INVARIANT: no error indication (clean query)
    assert summary.error_indication is None, (
        f"Unexpected error_indication: {summary.error_indication}"
    )


# ---------------------------------------------------------------------------
# Property 15: Top 3 blocking reasons with alphabetical tie-break
# Validates: Requirements 8.2
#
# For any pm_rejected_by_reason distribution, top_blocking_reasons SHALL contain
# at most 3 reason codes sorted by descending count, with alphabetical ordering
# to break ties. If fewer than 3 distinct codes exist, only available codes
# SHALL be reported.
# ---------------------------------------------------------------------------


@given(reason_counts=reason_dict_st)
@settings(max_examples=200)
def test_top_3_blocking_reasons_with_alphabetical_tie_break(reason_counts):
    """Property 15: Top 3 blocking reasons with alphabetical tie-break.

    For any pm_rejected_by_reason distribution, _compute_top_blocking_reasons()
    returns at most 3 reason codes sorted by descending count with alphabetical
    tie-break. If fewer than 3 distinct codes exist, only those are reported.

    Strategy: Generate random dicts of {reason_code: count}, call
    _compute_top_blocking_reasons() directly, verify ordering and length.

    **Validates: Requirements 8.2**
    """
    result = _compute_top_blocking_reasons(reason_counts)

    # INVARIANT: at most 3 reason codes returned
    assert len(result) <= 3, (
        f"Expected at most 3 reasons, got {len(result)}: {result}"
    )

    # INVARIANT: no more reasons than distinct codes available
    assert len(result) <= len(reason_counts), (
        f"Result has {len(result)} items but only {len(reason_counts)} distinct codes exist"
    )

    # INVARIANT: if there are fewer than 3 distinct codes, report only what's available
    if len(reason_counts) <= 3:
        assert len(result) == len(reason_counts), (
            f"With {len(reason_counts)} distinct codes, expected {len(reason_counts)} "
            f"results, got {len(result)}"
        )

    # INVARIANT: all returned codes are from the input
    for code in result:
        assert code in reason_counts, (
            f"Result code {code!r} not in input reason_counts"
        )

    # INVARIANT: sorted by descending count, then alphabetical for ties
    for i in range(len(result) - 1):
        count_i = reason_counts[result[i]]
        count_next = reason_counts[result[i + 1]]
        assert count_i >= count_next, (
            f"Results not sorted by descending count: "
            f"{result[i]}={count_i} before {result[i+1]}={count_next}"
        )
        # If same count, alphabetical ordering
        if count_i == count_next:
            assert result[i] < result[i + 1], (
                f"Tie-break not alphabetical: {result[i]!r} should come before "
                f"{result[i+1]!r} (both count={count_i})"
            )

    # INVARIANT: the returned codes have counts >= all non-returned codes
    if result and reason_counts:
        min_returned_count = min(reason_counts[code] for code in result)
        for code, count in reason_counts.items():
            if code not in result:
                assert count <= min_returned_count, (
                    f"Non-returned code {code!r} (count={count}) has higher count "
                    f"than returned code with count {min_returned_count}"
                )
                # If equal to min_returned_count, alphabetical must put it after
                if count == min_returned_count:
                    # The non-returned code must be alphabetically after the last
                    # returned code with the same count
                    same_count_returned = [
                        r for r in result if reason_counts[r] == count
                    ]
                    if same_count_returned:
                        last_same = same_count_returned[-1]
                        assert code > last_same, (
                            f"Non-returned code {code!r} should be after {last_same!r} "
                            f"alphabetically (both count={count})"
                        )


# ---------------------------------------------------------------------------
# Property 16: Zero-trade dominant blocker stage
# Validates: Requirements 8.3
#
# For any daily summary where executed == 0, the dominant_blocker_stage field
# SHALL identify the pipeline stage that rejected the most candidates, selected
# from the fixed set: no_signals, pm_rejection, gate_rejection, sizing_rejection,
# execution_failure.
# ---------------------------------------------------------------------------

_VALID_BLOCKER_STAGES = frozenset({
    "no_signals",
    "pm_rejection",
    "gate_rejection",
    "sizing_rejection",
    "execution_failure",
})


@given(
    candidates_built=count_st,
    pm_rejected=count_st,
    gate_sizing_rejected=count_st,
    execution_failed=count_st,
)
@settings(max_examples=200)
def test_zero_trade_dominant_blocker_stage(
    candidates_built, pm_rejected, gate_sizing_rejected, execution_failed
):
    """Property 16: Zero-trade dominant blocker stage.

    For any daily summary where executed == 0, _compute_dominant_blocker_stage()
    returns a stage from the fixed set that corresponds to the highest count,
    with alphabetical tie-break.

    Strategy: Generate random count distributions, call
    _compute_dominant_blocker_stage() directly, verify result is in fixed set
    and corresponds to highest count (with alphabetical tie-break).

    **Validates: Requirements 8.3**
    """
    result = _compute_dominant_blocker_stage(
        candidates_built=candidates_built,
        pm_rejected=pm_rejected,
        gate_sizing_rejected=gate_sizing_rejected,
        execution_failed=execution_failed,
    )

    # INVARIANT: result is always from the fixed set
    assert result in _VALID_BLOCKER_STAGES, (
        f"dominant_blocker_stage {result!r} not in valid set {_VALID_BLOCKER_STAGES}"
    )

    # INVARIANT: when candidates_built == 0, result is "no_signals"
    if candidates_built == 0:
        assert result == "no_signals", (
            f"When candidates_built == 0, expected 'no_signals', got {result!r}"
        )
    else:
        # INVARIANT: result is NOT "no_signals" when candidates_built > 0
        assert result != "no_signals", (
            f"When candidates_built > 0, should not be 'no_signals'"
        )

        # The stage counts used by the function
        stage_counts = {
            "pm_rejection": pm_rejected,
            "gate_rejection": gate_sizing_rejected,
            "sizing_rejection": 0,  # sizing is combined into gate_sizing_rejected
            "execution_failure": execution_failed,
        }

        max_count = max(stage_counts.values())

        # INVARIANT: result corresponds to a stage with the maximum count
        assert stage_counts.get(result, 0) == max_count or max_count == 0, (
            f"Result {result!r} has count {stage_counts.get(result, 0)} "
            f"but max count is {max_count}. Stage counts: {stage_counts}"
        )

        # INVARIANT: if multiple stages tie, alphabetical ordering breaks the tie
        if max_count > 0:
            tied_stages = sorted(
                [s for s, c in stage_counts.items() if c == max_count]
            )
            assert result == tied_stages[0], (
                f"Tie-break: expected {tied_stages[0]!r} (first alphabetically "
                f"among {tied_stages}), got {result!r}"
            )
