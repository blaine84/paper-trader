"""Tests for the fast-path LLM annotation system.

Validates:
- Unannotated events are returned in evaluated_at ASC order
- Annotation updates status and payload
- Veto on pending_order_created cancels the order
- Veto on trade_executed is rejected
- Annotation failure does not modify event outcome

Requirements: 7.1-7.7, cross-cutting acceptance test 4
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from db.schema import init_fast_path_events_schema, init_fast_path_triggers_schema
from utils.fast_path_annotation import (
    annotate_event,
    get_unannotated_events,
    process_pm_veto,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2026, 8, 20, 14, 30, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _insert_event(conn, *, event_id=None, profile_id="moderate", symbol="TSLA",
                  outcome_type="missed_move", outcome_reason_code="target_already_crossed",
                  annotation_status="annotation_pending", evaluated_at=None,
                  trigger_id=None):
    """Insert a minimal fast_path_events row for testing."""
    if event_id is None:
        event_id = str(uuid.uuid4())
    if evaluated_at is None:
        evaluated_at = _iso(NOW)
    if trigger_id is None:
        trigger_id = str(uuid.uuid4())
    conn.execute(
        text(
            "INSERT INTO fast_path_events "
            "(event_id, trigger_id, symbol, profile_id, setup_type, direction, "
            "outcome_type, outcome_reason_code, current_price, evaluated_at, "
            "annotation_status, entry_price, stop_price, target_price) "
            "VALUES (:event_id, :trigger_id, :symbol, :profile_id, 'momentum_fade', "
            "'SHORT', :outcome_type, :outcome_reason_code, 342.08, :evaluated_at, "
            ":annotation_status, 351.61, 355.00, 348.97)"
        ),
        {
            "event_id": event_id,
            "trigger_id": trigger_id,
            "symbol": symbol,
            "profile_id": profile_id,
            "outcome_type": outcome_type,
            "outcome_reason_code": outcome_reason_code,
            "evaluated_at": evaluated_at,
            "annotation_status": annotation_status,
        },
    )
    conn.commit()
    return event_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    init_fast_path_triggers_schema(eng)
    init_fast_path_events_schema(eng)
    return eng


# ---------------------------------------------------------------------------
# get_unannotated_events — Requirement 7.1, 7.3
# ---------------------------------------------------------------------------


class TestGetUnannotatedEvents:
    """Unannotated events are returned in evaluated_at ASC order."""

    def test_returns_events_ordered_by_evaluated_at(self, engine):
        """Events are returned oldest first."""
        with engine.connect() as conn:
            e1 = _insert_event(conn, evaluated_at=_iso(NOW - timedelta(minutes=5)))
            e2 = _insert_event(conn, evaluated_at=_iso(NOW - timedelta(minutes=10)))
            e3 = _insert_event(conn, evaluated_at=_iso(NOW))

        result = get_unannotated_events(engine, "moderate")

        assert len(result) == 3
        # Oldest first
        assert result[0]["event_id"] == e2
        assert result[1]["event_id"] == e1
        assert result[2]["event_id"] == e3

    def test_filters_by_profile_id(self, engine):
        """Only returns events for the requested profile."""
        with engine.connect() as conn:
            _insert_event(conn, profile_id="moderate")
            _insert_event(conn, profile_id="aggressive")

        result = get_unannotated_events(engine, "moderate")

        assert len(result) == 1
        assert result[0]["profile_id"] == "moderate"

    def test_excludes_already_annotated_events(self, engine):
        """Events with annotation_status != 'annotation_pending' are excluded."""
        with engine.connect() as conn:
            pending_id = _insert_event(conn, annotation_status="annotation_pending")
            _insert_event(conn, annotation_status="annotated")
            _insert_event(conn, annotation_status="annotation_failed")

        result = get_unannotated_events(engine, "moderate")

        assert len(result) == 1
        assert result[0]["event_id"] == pending_id

    def test_respects_limit(self, engine):
        """Respects the limit parameter."""
        with engine.connect() as conn:
            for i in range(5):
                _insert_event(
                    conn,
                    evaluated_at=_iso(NOW + timedelta(minutes=i)),
                )

        result = get_unannotated_events(engine, "moderate", limit=3)

        assert len(result) == 3

    def test_returns_empty_list_when_no_events(self, engine):
        """Returns empty list when no events match."""
        result = get_unannotated_events(engine, "moderate")
        assert result == []

    def test_returns_empty_list_on_error(self):
        """Fail-open: returns empty list on database error."""
        broken_engine = MagicMock()
        broken_engine.connect.side_effect = RuntimeError("db unavailable")

        result = get_unannotated_events(broken_engine, "moderate")

        assert result == []


# ---------------------------------------------------------------------------
# annotate_event — Requirement 7.1, 7.2, 7.4
# ---------------------------------------------------------------------------


class TestAnnotateEvent:
    """Annotation updates status and payload."""

    def test_updates_annotation_fields(self, engine):
        """Sets annotation_status, annotation_json, and annotation_timestamp."""
        with engine.connect() as conn:
            event_id = _insert_event(conn)

        payload = {"thesis": "momentum exhaustion near resistance", "confidence": "high"}
        annotate_event(engine, event_id, payload)

        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT annotation_status, annotation_json, annotation_timestamp "
                    "FROM fast_path_events WHERE event_id = :eid"
                ),
                {"eid": event_id},
            ).mappings().one()

        assert row["annotation_status"] == "annotated"
        assert json.loads(row["annotation_json"]) == payload
        assert row["annotation_timestamp"] is not None

    def test_accepts_pre_serialized_json_string(self, engine):
        """Accepts a pre-serialized JSON string as annotation payload."""
        with engine.connect() as conn:
            event_id = _insert_event(conn)

        payload_str = '{"thesis":"pre-serialized"}'
        annotate_event(engine, event_id, payload_str)

        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT annotation_json FROM fast_path_events WHERE event_id = :eid"
                ),
                {"eid": event_id},
            ).mappings().one()

        assert row["annotation_json"] == payload_str

    def test_does_not_modify_outcome_fields(self, engine):
        """Annotation never changes outcome_type, current_price, or other core fields."""
        with engine.connect() as conn:
            event_id = _insert_event(
                conn,
                outcome_type="stand_down",
                outcome_reason_code="stale_market_data",
            )

        annotate_event(engine, event_id, {"note": "annotated"})

        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT outcome_type, outcome_reason_code, current_price "
                    "FROM fast_path_events WHERE event_id = :eid"
                ),
                {"eid": event_id},
            ).mappings().one()

        assert row["outcome_type"] == "stand_down"
        assert row["outcome_reason_code"] == "stale_market_data"
        assert row["current_price"] == 342.08

    def test_annotation_failure_does_not_raise(self):
        """Fail-open: annotation failure is swallowed, not raised."""
        broken_engine = MagicMock()
        broken_engine.connect.side_effect = RuntimeError("connection lost")

        # Should not raise
        annotate_event(broken_engine, "nonexistent-event-id", {"note": "test"})


# ---------------------------------------------------------------------------
# process_pm_veto — Requirement 7.6, 7.7
# ---------------------------------------------------------------------------


class TestProcessPmVeto:
    """Veto processing for vetoable and non-vetoable outcomes."""

    def test_veto_on_pending_order_cancels_order(self, engine):
        """Veto on pending_order_created calls the order cancellation logic."""
        with engine.connect() as conn:
            event_id = _insert_event(
                conn,
                outcome_type="pending_order_created",
                outcome_reason_code="price_away_valid_limit",
            )

        with patch(
            "utils.fast_path_annotation._cancel_pending_order_for_event"
        ) as mock_cancel:
            result = process_pm_veto(engine, event_id, "Thesis invalidated by news")

        assert result is True
        mock_cancel.assert_called_once()
        # Verify it was called with the right event_id and symbol
        call_args = mock_cancel.call_args
        assert call_args[0][1] == event_id  # event_id
        assert call_args[0][3] == "TSLA"    # symbol

        # Verify annotation was recorded as a veto
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT annotation_status, annotation_json "
                    "FROM fast_path_events WHERE event_id = :eid"
                ),
                {"eid": event_id},
            ).mappings().one()

        assert row["annotation_status"] == "annotated"
        annotation = json.loads(row["annotation_json"])
        assert annotation["veto"] is True
        assert annotation["veto_rationale"] == "Thesis invalidated by news"
        assert annotation["vetoed_outcome_type"] == "pending_order_created"

    def test_veto_on_watch_created_invalidates_watch(self, engine):
        """Veto on watch_created calls watch invalidation logic."""
        with engine.connect() as conn:
            event_id = _insert_event(
                conn,
                outcome_type="watch_created",
                outcome_reason_code="awaiting_confirmation",
            )

        with patch(
            "utils.fast_path_annotation._invalidate_watch_for_event"
        ) as mock_invalidate:
            result = process_pm_veto(engine, event_id, "Watch no longer valid")

        assert result is True
        mock_invalidate.assert_called_once()
        call_args = mock_invalidate.call_args
        assert call_args[0][1] == event_id  # event_id
        assert call_args[0][3] == "TSLA"    # symbol

        # Verify veto annotation was recorded
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT annotation_status, annotation_json "
                    "FROM fast_path_events WHERE event_id = :eid"
                ),
                {"eid": event_id},
            ).mappings().one()

        assert row["annotation_status"] == "annotated"
        annotation = json.loads(row["annotation_json"])
        assert annotation["veto"] is True
        assert annotation["vetoed_outcome_type"] == "watch_created"

    def test_veto_on_trade_executed_is_rejected(self, engine):
        """Cannot veto a trade that has already been executed."""
        with engine.connect() as conn:
            event_id = _insert_event(
                conn,
                outcome_type="trade_executed",
                outcome_reason_code="gates_passed",
            )

        result = process_pm_veto(engine, event_id, "Would like to undo trade")

        assert result is False

        # Verify event was NOT annotated
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT annotation_status FROM fast_path_events WHERE event_id = :eid"
                ),
                {"eid": event_id},
            ).mappings().one()

        assert row["annotation_status"] == "annotation_pending"

    def test_veto_on_missed_move_is_rejected(self, engine):
        """Cannot veto a missed_move — it is informational, no action to reverse."""
        with engine.connect() as conn:
            event_id = _insert_event(
                conn,
                outcome_type="missed_move",
                outcome_reason_code="target_already_crossed",
            )

        result = process_pm_veto(engine, event_id, "Disagree with assessment")

        assert result is False

    def test_veto_on_stand_down_is_rejected(self, engine):
        """Cannot veto a stand_down — no action was taken."""
        with engine.connect() as conn:
            event_id = _insert_event(
                conn,
                outcome_type="stand_down",
                outcome_reason_code="stale_market_data",
            )

        result = process_pm_veto(engine, event_id, "Data was fine")

        assert result is False

    def test_veto_on_nonexistent_event_returns_false(self, engine):
        """Veto on an event that does not exist returns False."""
        result = process_pm_veto(engine, "nonexistent-event-id", "Reason")
        assert result is False


# ---------------------------------------------------------------------------
# Annotation failure isolation — cross-cutting acceptance test 4
# ---------------------------------------------------------------------------


class TestAnnotationFailureIsolation:
    """Annotation failure does not modify event outcome."""

    def test_failed_annotation_preserves_event_outcome(self, engine):
        """If annotate_event fails internally, the event's outcome fields are unchanged."""
        with engine.connect() as conn:
            event_id = _insert_event(
                conn,
                outcome_type="pending_order_created",
                outcome_reason_code="price_away_valid_limit",
            )

        # Force annotate_event to fail by making the engine's connect raise
        # after the first successful insertion
        failing_engine = MagicMock()
        failing_engine.connect.side_effect = RuntimeError("disk full")

        annotate_event(failing_engine, event_id, {"note": "should not persist"})

        # Verify outcome is intact on the real engine
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT outcome_type, outcome_reason_code, annotation_status, "
                    "current_price, entry_price, stop_price, target_price "
                    "FROM fast_path_events WHERE event_id = :eid"
                ),
                {"eid": event_id},
            ).mappings().one()

        assert row["outcome_type"] == "pending_order_created"
        assert row["outcome_reason_code"] == "price_away_valid_limit"
        assert row["annotation_status"] == "annotation_pending"
        assert row["current_price"] == 342.08
        assert row["entry_price"] == 351.61
        assert row["stop_price"] == 355.00
        assert row["target_price"] == 348.97

    def test_veto_failure_preserves_event_outcome(self, engine):
        """If process_pm_veto fails internally, the event remains unchanged."""
        with engine.connect() as conn:
            event_id = _insert_event(
                conn,
                outcome_type="pending_order_created",
                outcome_reason_code="price_away_valid_limit",
            )

        # Patch the internal cancel helper to raise an exception
        with patch(
            "utils.fast_path_annotation._cancel_pending_order_for_event",
            side_effect=RuntimeError("registry unavailable"),
        ):
            result = process_pm_veto(engine, event_id, "Veto attempt")

        # Fail-open: returns False, event untouched
        assert result is False

        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT outcome_type, annotation_status "
                    "FROM fast_path_events WHERE event_id = :eid"
                ),
                {"eid": event_id},
            ).mappings().one()

        assert row["outcome_type"] == "pending_order_created"
        assert row["annotation_status"] == "annotation_pending"
