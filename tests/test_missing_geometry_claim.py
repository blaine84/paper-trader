"""Unit tests for missing geometry claim detection (Requirement 4.2).

Validates that _detect_missing_geometry_claim correctly identifies
PM rationale text claiming missing geometry and emits the appropriate
contract_violation_missing_geometry_claim event.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from utils.candidate_pipeline import (
    _detect_missing_geometry_claim,
    _MISSING_GEOMETRY_PHRASES,
)


@pytest.fixture
def mock_engine():
    """Create a mock SQLAlchemy engine that captures executed SQL."""
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    return engine, conn


class TestMissingGeometryClaimDetection:
    """Tests for _detect_missing_geometry_claim function."""

    def test_emits_event_for_missing_entry_phrase(self, mock_engine):
        """Rationale containing 'missing entry' triggers event."""
        engine, conn = mock_engine
        _detect_missing_geometry_claim(
            engine, "cand-123", "cycle-1", "profile-a",
            "Rejected because missing entry price for this symbol",
        )
        conn.execute.assert_called_once()
        call_args = conn.execute.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert params["event_type"] == "contract_violation_missing_geometry_claim"
        event_data = json.loads(params["event_data"])
        assert event_data["matched_phrase"] == "missing entry"
        assert event_data["candidate_id"] == "cand-123"

    def test_emits_event_for_no_stop_price(self, mock_engine):
        """Rationale containing 'no stop price' triggers event."""
        engine, conn = mock_engine
        _detect_missing_geometry_claim(
            engine, "cand-456", "cycle-2", "profile-b",
            "This candidate has no stop price defined",
        )
        conn.execute.assert_called_once()
        params = conn.execute.call_args[0][1] if len(conn.execute.call_args[0]) > 1 else conn.execute.call_args[1]
        event_data = json.loads(params["event_data"])
        # "no stop" matches first since it appears before "no stop price" in phrase list
        assert event_data["matched_phrase"] == "no stop"

    def test_emits_event_for_no_target(self, mock_engine):
        """Rationale containing 'no target' triggers event."""
        engine, conn = mock_engine
        _detect_missing_geometry_claim(
            engine, "cand-789", "cycle-3", "profile-c",
            "Rejected: no target was specified",
        )
        conn.execute.assert_called_once()
        params = conn.execute.call_args[0][1] if len(conn.execute.call_args[0]) > 1 else conn.execute.call_args[1]
        event_data = json.loads(params["event_data"])
        assert event_data["matched_phrase"] == "no target"

    def test_no_event_for_unrelated_rationale(self, mock_engine):
        """Rationale without geometry phrases does not trigger event."""
        engine, conn = mock_engine
        _detect_missing_geometry_claim(
            engine, "cand-111", "cycle-4", "profile-d",
            "Low confidence in current market conditions",
        )
        conn.execute.assert_not_called()

    def test_no_event_for_none_rationale(self, mock_engine):
        """None rationale does not trigger event."""
        engine, conn = mock_engine
        _detect_missing_geometry_claim(
            engine, "cand-222", "cycle-5", "profile-e",
            None,
        )
        conn.execute.assert_not_called()

    def test_no_event_for_empty_rationale(self, mock_engine):
        """Empty string rationale does not trigger event."""
        engine, conn = mock_engine
        _detect_missing_geometry_claim(
            engine, "cand-333", "cycle-6", "profile-f",
            "",
        )
        conn.execute.assert_not_called()

    def test_case_insensitive_matching(self, mock_engine):
        """Phrase matching is case-insensitive."""
        engine, conn = mock_engine
        _detect_missing_geometry_claim(
            engine, "cand-444", "cycle-7", "profile-g",
            "MISSING ENTRY price in the signal",
        )
        conn.execute.assert_called_once()

    def test_only_first_phrase_match_emits_event(self, mock_engine):
        """Only one event emitted even if multiple phrases match."""
        engine, conn = mock_engine
        _detect_missing_geometry_claim(
            engine, "cand-555", "cycle-8", "profile-h",
            "Missing entry and no stop price and no target price",
        )
        # Should only emit ONE event (first matching phrase)
        assert conn.execute.call_count == 1

    def test_rationale_truncated_to_2000_chars(self, mock_engine):
        """Rationale in event_data is truncated to 2000 characters."""
        engine, conn = mock_engine
        long_rationale = "missing entry " + "x" * 3000
        _detect_missing_geometry_claim(
            engine, "cand-666", "cycle-9", "profile-i",
            long_rationale,
        )
        conn.execute.assert_called_once()
        params = conn.execute.call_args[0][1] if len(conn.execute.call_args[0]) > 1 else conn.execute.call_args[1]
        event_data = json.loads(params["event_data"])
        assert len(event_data["rationale"]) <= 2000

    def test_fail_open_on_db_error(self, mock_engine):
        """Database errors are caught and logged, not raised."""
        engine, conn = mock_engine
        conn.execute.side_effect = RuntimeError("DB connection lost")
        # Should NOT raise
        _detect_missing_geometry_claim(
            engine, "cand-777", "cycle-10", "profile-j",
            "Rejected: missing entry price",
        )
        # Function completed without raising

    def test_all_phrases_are_detectable(self, mock_engine):
        """All defined phrases are properly detected."""
        engine, conn = mock_engine
        for phrase in _MISSING_GEOMETRY_PHRASES:
            conn.reset_mock()
            _detect_missing_geometry_claim(
                engine, "cand-x", "cycle-x", "profile-x",
                f"Rejected because {phrase} in the data",
            )
            assert conn.execute.called, f"Phrase '{phrase}' was not detected"
