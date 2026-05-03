"""Smoke tests for the preprocessor module.

Verifies basic correctness against the AMD overlap and no-overlap fixtures.
"""

import json
from pathlib import Path

from lib.preprocessor import compute_overlap_candidates

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_amd_overlap_produces_one_candidate():
    """The AMD overlap fixture should produce exactly 1 candidate with trade_ids [259, 260]."""
    with open(FIXTURES_DIR / "amd_overlap_snapshot.json") as f:
        snapshot = json.load(f)

    result = compute_overlap_candidates(snapshot)

    assert result["diagnostic"] == "same_symbol_overlap"
    assert result["window_minutes"] == 120
    assert len(result["overlap_candidates"]) == 1

    candidate = result["overlap_candidates"][0]
    assert candidate["candidate_id"] == "overlap_001"
    assert candidate["symbol"] == "AMD"
    assert candidate["direction"] == "LONG"
    assert candidate["trade_ids"] == [259, 260]
    assert set(candidate["profiles"]) == {"moderate", "aggressive"}
    assert candidate["setup_type"] == "technical_breakout"

    # Outcome checks
    assert candidate["outcome"]["combined_pnl"] == round(-91.76 + -309.96, 2)
    assert candidate["outcome"]["all_stopped_out"] is True

    # Evidence should have 2 entries
    assert len(candidate["evidence"]) == 2

    # Severity: 2 profiles, combined PnL < 0, all stopped out → high
    assert candidate["suggested_severity"] == "high"

    # Trade 260 entered while trade 259 was still open
    # (259 exit: 14:30:13, 260 entry: 14:20:10 → 260 entered before 259 exited)
    assert candidate["opened_while_existing_position_open"] is True

    # minutes_between_entries should be ~59.3 minutes
    assert candidate["minutes_between_entries"] is not None
    assert 59.0 <= candidate["minutes_between_entries"] <= 60.0

    # Since 260 opened while 259 was still open, minutes_since_prior_exit should be None
    assert candidate["minutes_since_prior_exit"] is None


def test_no_overlap_produces_empty_candidates():
    """The no-overlap fixture should produce 0 candidates."""
    with open(FIXTURES_DIR / "no_overlap_snapshot.json") as f:
        snapshot = json.load(f)

    result = compute_overlap_candidates(snapshot)

    assert result["diagnostic"] == "same_symbol_overlap"
    assert result["window_minutes"] == 120
    assert result["overlap_candidates"] == []


def test_empty_trades_produces_empty_candidates():
    """A snapshot with no trades should produce 0 candidates."""
    snapshot = {
        "snapshot_schema_version": "1.0",
        "snapshot_time": "2026-05-03T17:00:00-04:00",
        "scope": {"days": 5, "tables": ["trades"], "diagnostic": "same_symbol_overlap"},
        "trades": [],
        "trade_events": {"included_event_types": [], "rows": []},
        "dynamic_strategies": [],
    }

    result = compute_overlap_candidates(snapshot)

    assert result["overlap_candidates"] == []


def test_determinism():
    """Same input should always produce the same output."""
    with open(FIXTURES_DIR / "amd_overlap_snapshot.json") as f:
        snapshot = json.load(f)

    result1 = compute_overlap_candidates(snapshot)
    result2 = compute_overlap_candidates(snapshot)

    assert result1 == result2
