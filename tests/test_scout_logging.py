"""Tests for the structured scout logging helper.

Validates that emit_scout_event produces correctly formatted log output
with all required fields for each event type.

Validates: Requirements 10.4
"""

import json
import logging

from utils.scout_logging import emit_scout_event


class TestEmitScoutEvent:
    """Tests for emit_scout_event structured logging."""

    def test_scout_run_event(self, caplog):
        """SCOUT_RUN event emits with required payload fields."""
        with caplog.at_level(logging.INFO, logger="utils.scout_logging"):
            emit_scout_event("SCOUT_RUN", {
                "run_type": "premarket",
                "sectors_scanned": 6,
                "total_candidates": 42,
                "finalists_count": 15,
                "picks_count": 3,
            })

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert "SCOUT_RUN" in record.message

        # Parse the JSON payload from the log message
        json_str = record.message.split("SCOUT_RUN: ", 1)[1]
        payload = json.loads(json_str)

        assert payload["event"] == "SCOUT_RUN"
        assert payload["run_type"] == "premarket"
        assert payload["sectors_scanned"] == 6
        assert payload["total_candidates"] == 42
        assert payload["finalists_count"] == 15
        assert payload["picks_count"] == 3
        assert "timestamp" in payload

    def test_sector_screen_event(self, caplog):
        """SECTOR_SCREEN event emits with required payload fields."""
        with caplog.at_level(logging.INFO, logger="utils.scout_logging"):
            emit_scout_event("SECTOR_SCREEN", {
                "sector_key": "ai_semi",
                "candidates_found": 10,
                "passed_gates": 7,
                "top_score": 82.5,
            })

        assert len(caplog.records) == 1
        record = caplog.records[0]
        json_str = record.message.split("SECTOR_SCREEN: ", 1)[1]
        payload = json.loads(json_str)

        assert payload["event"] == "SECTOR_SCREEN"
        assert payload["sector_key"] == "ai_semi"
        assert payload["candidates_found"] == 10
        assert payload["passed_gates"] == 7
        assert payload["top_score"] == 82.5
        assert "timestamp" in payload

    def test_chief_scout_event(self, caplog):
        """CHIEF_SCOUT event emits with required payload fields."""
        with caplog.at_level(logging.INFO, logger="utils.scout_logging"):
            emit_scout_event("CHIEF_SCOUT", {
                "picks_count": 4,
                "fallback_used": False,
                "symbols_selected": ["AVGO", "SMCI", "CRM", "PANW"],
            })

        assert len(caplog.records) == 1
        record = caplog.records[0]
        json_str = record.message.split("CHIEF_SCOUT: ", 1)[1]
        payload = json.loads(json_str)

        assert payload["event"] == "CHIEF_SCOUT"
        assert payload["picks_count"] == 4
        assert payload["fallback_used"] is False
        assert payload["symbols_selected"] == ["AVGO", "SMCI", "CRM", "PANW"]
        assert "timestamp" in payload

    def test_expanded_watchlist_event(self, caplog):
        """EXPANDED_WATCHLIST event emits with required payload fields."""
        with caplog.at_level(logging.INFO, logger="utils.scout_logging"):
            emit_scout_event("EXPANDED_WATCHLIST", {
                "symbols": ["AVGO", "SMCI"],
                "total_size": 2,
                "run_type": "premarket",
            })

        assert len(caplog.records) == 1
        record = caplog.records[0]
        json_str = record.message.split("EXPANDED_WATCHLIST: ", 1)[1]
        payload = json.loads(json_str)

        assert payload["event"] == "EXPANDED_WATCHLIST"
        assert payload["symbols"] == ["AVGO", "SMCI"]
        assert payload["total_size"] == 2
        assert payload["run_type"] == "premarket"
        assert "timestamp" in payload

    def test_budget_ceiling_hit_event(self, caplog):
        """BUDGET_CEILING_HIT event emits with required payload fields."""
        with caplog.at_level(logging.INFO, logger="utils.scout_logging"):
            emit_scout_event("BUDGET_CEILING_HIT", {
                "ceiling_type": "max_sectors_per_run",
                "limit_value": 7,
                "context": "Reached 7 sectors, stopping iteration",
            })

        assert len(caplog.records) == 1
        record = caplog.records[0]
        json_str = record.message.split("BUDGET_CEILING_HIT: ", 1)[1]
        payload = json.loads(json_str)

        assert payload["event"] == "BUDGET_CEILING_HIT"
        assert payload["ceiling_type"] == "max_sectors_per_run"
        assert payload["limit_value"] == 7
        assert payload["context"] == "Reached 7 sectors, stopping iteration"
        assert "timestamp" in payload

    def test_event_payload_is_valid_json(self, caplog):
        """All events produce valid JSON in the log message."""
        with caplog.at_level(logging.INFO, logger="utils.scout_logging"):
            emit_scout_event("SCOUT_RUN", {
                "run_type": "midday",
                "sectors_scanned": 3,
                "total_candidates": 20,
                "finalists_count": 8,
                "picks_count": 2,
            })

        record = caplog.records[0]
        # The format is "EVENT_TYPE: {json}"
        parts = record.message.split(": ", 1)
        assert len(parts) == 2
        # Should not raise
        parsed = json.loads(parts[1])
        assert isinstance(parsed, dict)

    def test_timestamp_is_iso_format(self, caplog):
        """Timestamp in payload is ISO 8601 format."""
        from datetime import datetime

        with caplog.at_level(logging.INFO, logger="utils.scout_logging"):
            emit_scout_event("SCOUT_RUN", {
                "run_type": "confirmation",
                "sectors_scanned": 1,
                "total_candidates": 5,
                "finalists_count": 2,
                "picks_count": 1,
            })

        record = caplog.records[0]
        json_str = record.message.split("SCOUT_RUN: ", 1)[1]
        payload = json.loads(json_str)

        # Should parse as ISO datetime without error
        ts = datetime.fromisoformat(payload["timestamp"])
        assert ts is not None
