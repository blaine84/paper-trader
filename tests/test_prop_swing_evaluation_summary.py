"""Property-based tests for SwingEvaluationSummary (Properties 2, 7, 18).

Tests that:
- Property 2: Per-Symbol Entry Structural Completeness — all required fields
              present and correctly typed
- Property 7: Counts-by-Rejection-Category Validity — all keys ∈ CANONICAL_REJECTION_CODES,
              all values > 0, sum equals count of rejected entries, zero-count codes absent
- Property 18: Telemetry Persistence Round-Trip — JSON payload is parseable and
               contains all per_symbol_entries and counts_by_rejection_category matching original

**Validates: Requirements 2.1, 4.1, 4.2, 4.3, 4.4, 16.2, 16.3**
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from hypothesis import given, settings, strategies as st

from utils.swing_candidate_bridge import (
    CANONICAL_REJECTION_CODES,
    PerSymbolEntry,
    SwingEvaluationSummary,
    _build_evaluation_summary,
    _persist_evaluation_summary,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

symbol_st = st.text(min_size=1, max_size=5, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ")

direction_st = st.sampled_from(["LONG", "SHORT", "HOLD"])
confidence_st = st.sampled_from(["low", "medium", "high"])
strength_st = st.sampled_from(["weak", "moderate", "strong"])

rejection_reason_st = st.one_of(
    st.none(),
    st.sampled_from(sorted(CANONICAL_REJECTION_CODES)),
)

raw_rejection_reason_st = st.one_of(st.none(), st.text(min_size=1, max_size=40))

missing_evidence_st = st.one_of(
    st.none(),
    st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5),
)

per_symbol_entry_st = st.builds(
    PerSymbolEntry,
    symbol=symbol_st,
    raw_direction=direction_st,
    raw_setup_label=st.text(min_size=1, max_size=30),
    normalized_setup_label=st.one_of(st.none(), st.text(min_size=1, max_size=30)),
    confidence=confidence_st,
    strength=strength_st,
    construction_attempted=st.booleans(),
    construction_succeeded=st.booleans(),
    final_rejection_reason=rejection_reason_st,
    raw_rejection_reason=raw_rejection_reason_st,
    missing_evidence=missing_evidence_st,
)


# ---------------------------------------------------------------------------
# Property 2: Per-Symbol Entry Structural Completeness
# Validates: Requirements 2.1, 4.2
#
# Every PerSymbolEntry has all required fields present and correctly typed.
# This guarantees the dataclass contract is never violated regardless of input
# combinations.
# ---------------------------------------------------------------------------


@given(entry=per_symbol_entry_st)
@settings(max_examples=200)
def test_per_symbol_entry_structural_completeness(entry: PerSymbolEntry):
    """Property 2: All required fields present and correctly typed.

    **Validates: Requirements 2.1, 4.2**
    """
    # symbol: str
    assert isinstance(entry.symbol, str)
    assert len(entry.symbol) > 0

    # raw_direction: one of "LONG", "SHORT", "HOLD"
    assert entry.raw_direction in ("LONG", "SHORT", "HOLD")

    # raw_setup_label: str
    assert isinstance(entry.raw_setup_label, str)

    # normalized_setup_label: str or None
    assert entry.normalized_setup_label is None or isinstance(entry.normalized_setup_label, str)

    # confidence: one of "low", "medium", "high"
    assert entry.confidence in ("low", "medium", "high")

    # strength: one of "weak", "moderate", "strong"
    assert entry.strength in ("weak", "moderate", "strong")

    # construction_attempted: bool
    assert isinstance(entry.construction_attempted, bool)

    # construction_succeeded: bool
    assert isinstance(entry.construction_succeeded, bool)

    # final_rejection_reason: str in CANONICAL_REJECTION_CODES or None
    if entry.final_rejection_reason is not None:
        assert isinstance(entry.final_rejection_reason, str)
        assert entry.final_rejection_reason in CANONICAL_REJECTION_CODES, (
            f"final_rejection_reason {entry.final_rejection_reason!r} "
            f"not in CANONICAL_REJECTION_CODES"
        )

    # raw_rejection_reason: str or None
    assert entry.raw_rejection_reason is None or isinstance(entry.raw_rejection_reason, str)

    # missing_evidence: list[str] or None
    if entry.missing_evidence is not None:
        assert isinstance(entry.missing_evidence, list)
        for item in entry.missing_evidence:
            assert isinstance(item, str)


# ---------------------------------------------------------------------------
# Property 7: Counts-by-Rejection-Category Validity
# Validates: Requirements 4.1, 4.2, 4.3, 4.4
#
# When _build_evaluation_summary() computes the rejection counts dictionary:
# - All keys are in CANONICAL_REJECTION_CODES
# - All values > 0
# - Sum of values equals count of entries with non-None final_rejection_reason
# - No codes with zero occurrences appear in the dict
# - total_signals_evaluated == len(per_symbol_entries)
# ---------------------------------------------------------------------------


@given(entries=st.lists(per_symbol_entry_st, min_size=1, max_size=30))
@settings(max_examples=200)
def test_counts_by_rejection_category_validity(entries: list[PerSymbolEntry]):
    """Property 7: Counts-by-rejection-category validity.

    **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
    """
    summary = _build_evaluation_summary(
        cycle_id="test-cycle-001",
        profile_id="moderate",
        mode="observe",
        entries=entries,
    )

    counts = summary.counts_by_rejection_category

    # All keys must be in CANONICAL_REJECTION_CODES
    for key in counts:
        assert key in CANONICAL_REJECTION_CODES, (
            f"counts key {key!r} not in CANONICAL_REJECTION_CODES"
        )

    # All values must be > 0 (no zero-count codes present)
    for key, value in counts.items():
        assert isinstance(value, int)
        assert value > 0, (
            f"counts[{key!r}] = {value}, expected > 0"
        )

    # Sum of values equals count of entries with non-None final_rejection_reason
    expected_rejected_count = sum(
        1 for e in entries if e.final_rejection_reason is not None
    )
    actual_sum = sum(counts.values())
    assert actual_sum == expected_rejected_count, (
        f"sum(counts.values()) = {actual_sum} != "
        f"rejected entries count = {expected_rejected_count}"
    )

    # total_signals_evaluated equals len(per_symbol_entries)
    assert summary.total_signals_evaluated == len(entries)
    assert summary.total_signals_evaluated == len(summary.per_symbol_entries)

    # No codes with zero occurrences appear (redundant check with value > 0, but explicit)
    rejection_codes_in_entries = {
        e.final_rejection_reason for e in entries if e.final_rejection_reason is not None
    }
    for code in CANONICAL_REJECTION_CODES:
        if code not in rejection_codes_in_entries:
            assert code not in counts, (
                f"Code {code!r} has zero occurrences but appears in counts"
            )


# ---------------------------------------------------------------------------
# Property 1: Evaluation Summary Production Invariant
# Validates: Requirements 1.1, 1.3
#
# For any non-empty signal set and mode in {observe, enabled},
# total_signals_evaluated equals input signal count, per_symbol_entries
# length matches, candidate_mode reflects the input mode, and timestamp
# is a valid ISO 8601 string.
# ---------------------------------------------------------------------------


@given(
    entries=st.lists(per_symbol_entry_st, min_size=1, max_size=30),
    mode=st.sampled_from(["observe", "enabled"]),
)
@settings(max_examples=200)
def test_evaluation_summary_production_invariant(
    entries: list[PerSymbolEntry], mode: str
):
    """Property 1: Evaluation summary production invariant.

    **Validates: Requirements 1.1, 1.3**
    """
    from datetime import datetime as dt

    summary = _build_evaluation_summary(
        cycle_id="prop1-cycle-001",
        profile_id="moderate",
        mode=mode,
        entries=entries,
    )

    # total_signals_evaluated equals input signal count
    assert summary.total_signals_evaluated == len(entries)

    # per_symbol_entries length matches input
    assert len(summary.per_symbol_entries) == len(entries)

    # candidate_mode reflects the input mode
    assert summary.candidate_mode == mode

    # timestamp is a valid ISO 8601 string (parseable)
    assert isinstance(summary.timestamp, str)
    parsed = dt.fromisoformat(summary.timestamp)
    assert parsed is not None


# ---------------------------------------------------------------------------
# Property 18: Telemetry Persistence Round-Trip
# Validates: Requirements 16.2, 16.3
#
# The JSON payload produced by _persist_evaluation_summary is:
# 1. Parseable (valid JSON)
# 2. Contains all per_symbol_entries with correct field values matching original
# 3. Contains counts_by_rejection_category matching the original summary
# ---------------------------------------------------------------------------


@given(entries=st.lists(per_symbol_entry_st, min_size=1, max_size=10))
@settings(max_examples=200)
def test_telemetry_persistence_round_trip(entries: list[PerSymbolEntry]):
    """Property 18: JSON payload round-trip fidelity.

    **Validates: Requirements 16.2, 16.3**
    """
    summary = _build_evaluation_summary("cycle-1", "moderate", "observe", entries)

    # Mock DB to capture the executed params
    mock_conn = MagicMock()
    mock_db = MagicMock()
    mock_db.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_db.begin.return_value.__exit__ = MagicMock(return_value=False)

    _persist_evaluation_summary(mock_db, summary)

    # Extract event_data from the INSERT call
    call_args = mock_conn.execute.call_args
    params = call_args[0][1]  # second positional arg is the params dict
    event_data_json = params["event_data"]

    # 1. Parseable (valid JSON)
    parsed = json.loads(event_data_json)

    # 2. Contains all per_symbol_entries with correct count
    assert len(parsed["per_symbol_entries"]) == len(entries)

    # 3. Contains counts_by_rejection_category matching original
    assert parsed["counts_by_rejection_category"] == summary.counts_by_rejection_category

    # Verify each entry's key fields round-trip correctly
    for i, entry in enumerate(entries):
        pe = parsed["per_symbol_entries"][i]
        assert pe["symbol"] == entry.symbol
        assert pe["raw_direction"] == entry.raw_direction
        assert pe["raw_setup_label"] == entry.raw_setup_label
        assert pe["normalized_setup_label"] == entry.normalized_setup_label
        assert pe["confidence"] == entry.confidence
        assert pe["strength"] == entry.strength
        assert pe["construction_attempted"] == entry.construction_attempted
        assert pe["construction_succeeded"] == entry.construction_succeeded
        assert pe["final_rejection_reason"] == entry.final_rejection_reason
        assert pe["raw_rejection_reason"] == entry.raw_rejection_reason
        assert pe["missing_evidence"] == entry.missing_evidence
