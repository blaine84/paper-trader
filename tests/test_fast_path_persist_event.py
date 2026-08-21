"""Tests for _persist_event and _generate_simple_narration in fast_path_monitor.

Validates:
- UUID4 event_id generation
- All FastPathOutcome fields persisted correctly
- Template narration generated for each outcome type
- annotation_status = 'annotation_pending' on all new events
- narration_source = 'template' on all new events
- metadata serialized to JSON
- market_data_age_ms and evaluation_duration_ms from metadata
- source_signal_id extracted from trigger
- Fail-open: returns event_id even on DB error

Requirements: 9.1-9.5
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from db.schema import init_fast_path_events_schema, init_fast_path_triggers_schema
from utils.fast_path_evaluator import FastPathOutcome
from utils.fast_path_monitor import _generate_simple_narration, _persist_event


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    init_fast_path_triggers_schema(eng)
    init_fast_path_events_schema(eng)
    return eng


@dataclass
class FakeTrigger:
    """Minimal trigger stand-in for testing _persist_event."""

    trigger_id: str = "trig-001"
    symbol: str = "TSLA"
    profile_id: str = "moderate"
    direction: str = "SHORT"
    setup_type: str = "momentum_fade"
    source_signal_id: str | None = "signal-abc"
    source_watch_id: str | None = None


def _make_outcome(**overrides) -> FastPathOutcome:
    """Build a default FastPathOutcome for testing."""
    defaults = {
        "outcome_type": "missed_move",
        "outcome_reason_code": "target_already_crossed",
        "trigger_id": "trig-001",
        "symbol": "TSLA",
        "profile_id": "moderate",
        "direction": "SHORT",
        "setup_type": "momentum_fade",
        "current_price": 342.08,
        "entry_price": 351.61,
        "stop_price": 355.00,
        "target_price": 348.97,
        "reward_to_risk": 0.78,
        "blocking_rule_name": None,
        "blocking_rule_threshold": None,
        "metadata": None,
    }
    defaults.update(overrides)
    return FastPathOutcome(**defaults)


class TestGenerateSimpleNarration:
    """Tests for the _generate_simple_narration helper."""

    def test_missed_move_narration(self):
        outcome = _make_outcome(outcome_type="missed_move")
        narration = _generate_simple_narration(outcome)
        assert narration == "TSLA target already crossed; no order created."

    def test_stand_down_narration(self):
        outcome = _make_outcome(
            outcome_type="stand_down",
            outcome_reason_code="invalid_geometry",
        )
        narration = _generate_simple_narration(outcome)
        assert narration == "TSLA setup blocked: invalid_geometry."

    def test_trade_executed_narration(self):
        outcome = _make_outcome(
            outcome_type="trade_executed",
            outcome_reason_code="gates_passed",
            current_price=350.50,
        )
        narration = _generate_simple_narration(outcome)
        assert narration == "TSLA trade executed at 350.5."

    def test_pending_order_created_narration(self):
        outcome = _make_outcome(
            outcome_type="pending_order_created",
            outcome_reason_code="price_away_limit_valid",
        )
        narration = _generate_simple_narration(outcome)
        assert narration == "TSLA pending limit order created."

    def test_watch_created_narration(self):
        outcome = _make_outcome(
            outcome_type="watch_created",
            outcome_reason_code="awaiting_confirmation",
        )
        narration = _generate_simple_narration(outcome)
        assert narration == "TSLA watch created: awaiting_confirmation."

    def test_watch_promoted_narration(self):
        outcome = _make_outcome(
            outcome_type="watch_promoted",
            outcome_reason_code="level_confirmed",
        )
        narration = _generate_simple_narration(outcome)
        assert narration == "TSLA watch promoted to actionable."

    def test_unknown_outcome_type_fallback(self):
        outcome = _make_outcome(
            outcome_type="unknown_type",
            outcome_reason_code="some_reason",
        )
        narration = _generate_simple_narration(outcome)
        assert narration == "TSLA: some_reason"


class TestPersistEvent:
    """Tests for the _persist_event function."""

    def test_returns_uuid4_event_id(self, engine):
        outcome = _make_outcome()
        trigger = FakeTrigger()
        event_id = _persist_event(outcome, trigger, engine)
        # Validate UUID4 format
        parsed = uuid.UUID(event_id, version=4)
        assert str(parsed) == event_id

    def test_inserts_row_with_all_outcome_fields(self, engine):
        outcome = _make_outcome(
            reward_to_risk=1.25,
            blocking_rule_name="concentration",
            blocking_rule_threshold="0.15",
        )
        trigger = FakeTrigger(source_signal_id="sig-123")
        event_id = _persist_event(outcome, trigger, engine)

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM fast_path_events WHERE event_id = :eid"),
                {"eid": event_id},
            ).mappings().first()

        assert row is not None
        assert row["event_id"] == event_id
        assert row["trigger_id"] == "trig-001"
        assert row["symbol"] == "TSLA"
        assert row["profile_id"] == "moderate"
        assert row["setup_type"] == "momentum_fade"
        assert row["direction"] == "SHORT"
        assert row["entry_price"] == 351.61
        assert row["stop_price"] == 355.00
        assert row["target_price"] == 348.97
        assert row["current_price"] == 342.08
        assert row["reward_to_risk"] == 1.25
        assert row["outcome_type"] == "missed_move"
        assert row["outcome_reason_code"] == "target_already_crossed"
        assert row["blocking_rule_name"] == "concentration"
        assert row["blocking_rule_threshold"] == "0.15"
        assert row["source_signal_id"] == "sig-123"

    def test_annotation_status_is_annotation_pending(self, engine):
        outcome = _make_outcome()
        trigger = FakeTrigger()
        event_id = _persist_event(outcome, trigger, engine)

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT annotation_status FROM fast_path_events WHERE event_id = :eid"),
                {"eid": event_id},
            ).mappings().first()

        assert row["annotation_status"] == "annotation_pending"

    def test_narration_source_is_template(self, engine):
        outcome = _make_outcome()
        trigger = FakeTrigger()
        event_id = _persist_event(outcome, trigger, engine)

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT narration_source FROM fast_path_events WHERE event_id = :eid"),
                {"eid": event_id},
            ).mappings().first()

        assert row["narration_source"] == "template"

    def test_narration_populated_from_template(self, engine):
        outcome = _make_outcome(outcome_type="missed_move")
        trigger = FakeTrigger()
        event_id = _persist_event(outcome, trigger, engine)

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT narration FROM fast_path_events WHERE event_id = :eid"),
                {"eid": event_id},
            ).mappings().first()

        assert row["narration"] == "TSLA target already crossed; no order created."

    def test_metadata_serialized_to_json(self, engine):
        metadata = {"market_data_age_ms": 1500, "extra_field": "test_value"}
        outcome = _make_outcome(metadata=metadata)
        trigger = FakeTrigger()
        event_id = _persist_event(outcome, trigger, engine)

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT outcome_metadata_json FROM fast_path_events WHERE event_id = :eid"),
                {"eid": event_id},
            ).mappings().first()

        parsed = json.loads(row["outcome_metadata_json"])
        assert parsed["market_data_age_ms"] == 1500
        assert parsed["extra_field"] == "test_value"

    def test_market_data_age_ms_from_metadata(self, engine):
        outcome = _make_outcome(metadata={"market_data_age_ms": 2300})
        trigger = FakeTrigger()
        event_id = _persist_event(outcome, trigger, engine)

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT market_data_age_ms FROM fast_path_events WHERE event_id = :eid"),
                {"eid": event_id},
            ).mappings().first()

        assert row["market_data_age_ms"] == 2300

    def test_evaluation_duration_ms_from_metadata(self, engine):
        outcome = _make_outcome(metadata={"evaluation_duration_ms": 45})
        trigger = FakeTrigger()
        event_id = _persist_event(outcome, trigger, engine)

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT evaluation_duration_ms FROM fast_path_events WHERE event_id = :eid"),
                {"eid": event_id},
            ).mappings().first()

        assert row["evaluation_duration_ms"] == 45

    def test_null_metadata_produces_null_json_and_null_timing(self, engine):
        outcome = _make_outcome(metadata=None)
        trigger = FakeTrigger()
        event_id = _persist_event(outcome, trigger, engine)

        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT outcome_metadata_json, market_data_age_ms, "
                    "evaluation_duration_ms FROM fast_path_events WHERE event_id = :eid"
                ),
                {"eid": event_id},
            ).mappings().first()

        assert row["outcome_metadata_json"] is None
        assert row["market_data_age_ms"] is None
        assert row["evaluation_duration_ms"] is None

    def test_source_signal_id_from_trigger(self, engine):
        outcome = _make_outcome()
        trigger = FakeTrigger(source_signal_id="sig-xyz-999")
        event_id = _persist_event(outcome, trigger, engine)

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT source_signal_id FROM fast_path_events WHERE event_id = :eid"),
                {"eid": event_id},
            ).mappings().first()

        assert row["source_signal_id"] == "sig-xyz-999"

    def test_source_signal_id_none_when_trigger_lacks_field(self, engine):
        outcome = _make_outcome()
        trigger = MagicMock()
        del trigger.source_signal_id  # getattr will fall through to default None
        event_id = _persist_event(outcome, trigger, engine)

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT source_signal_id FROM fast_path_events WHERE event_id = :eid"),
                {"eid": event_id},
            ).mappings().first()

        assert row["source_signal_id"] is None

    def test_fail_open_returns_event_id_on_db_error(self):
        """Persistence failure should log and return event_id, not raise."""
        outcome = _make_outcome()
        trigger = FakeTrigger()

        # Use a broken engine that will fail on connect
        broken_engine = MagicMock()
        broken_engine.connect.side_effect = RuntimeError("DB unavailable")

        event_id = _persist_event(outcome, trigger, broken_engine)

        # Still returns a valid UUID
        parsed = uuid.UUID(event_id, version=4)
        assert str(parsed) == event_id

    def test_evaluated_at_and_created_at_populated(self, engine):
        outcome = _make_outcome()
        trigger = FakeTrigger()
        event_id = _persist_event(outcome, trigger, engine)

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT evaluated_at, created_at FROM fast_path_events WHERE event_id = :eid"),
                {"eid": event_id},
            ).mappings().first()

        assert row["evaluated_at"] is not None
        assert row["created_at"] is not None
        # Both should be ISO format timestamps
        assert "T" in row["evaluated_at"]
        assert "T" in row["created_at"]
