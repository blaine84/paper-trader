"""Tests for the fast-path public stream event generation and narration.

Validates:
- Each outcome type produces valid narration containing the symbol
- Stream event contains all required fields per Requirement 8.1
- Cycle summary counts are accurate against database state

Requirements: 8.1-8.5, 9.6
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text

from db.schema import init_fast_path_events_schema, init_fast_path_triggers_schema
from utils.fast_path_evaluator import FastPathOutcome
from utils.fast_path_stream import (
    build_stream_event,
    generate_cycle_summary,
    generate_template_narration,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2026, 8, 20, 14, 30, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _make_outcome_dict(outcome_type: str, **overrides) -> dict:
    """Build a minimal outcome dict suitable for generate_template_narration."""
    base = {
        "outcome_type": outcome_type,
        "symbol": "TSLA",
        "entry_price": 351.61,
        "stop_price": 355.00,
        "target_price": 348.97,
        "current_price": 342.08,
        "reward_to_risk": 2.5,
        "outcome_reason_code": "target_already_crossed",
    }
    base.update(overrides)
    return base


def _make_outcome_dataclass(outcome_type: str, **overrides) -> FastPathOutcome:
    """Build a FastPathOutcome dataclass for narration testing."""
    kwargs = {
        "outcome_type": outcome_type,
        "outcome_reason_code": overrides.pop("outcome_reason_code", "target_already_crossed"),
        "trigger_id": overrides.pop("trigger_id", str(uuid.uuid4())),
        "symbol": overrides.pop("symbol", "TSLA"),
        "profile_id": overrides.pop("profile_id", "moderate"),
        "direction": overrides.pop("direction", "SHORT"),
        "setup_type": overrides.pop("setup_type", "momentum_fade"),
        "current_price": overrides.pop("current_price", 342.08),
        "entry_price": overrides.pop("entry_price", 351.61),
        "stop_price": overrides.pop("stop_price", 355.00),
        "target_price": overrides.pop("target_price", 348.97),
        "reward_to_risk": overrides.pop("reward_to_risk", 2.5),
    }
    kwargs.update(overrides)
    return FastPathOutcome(**kwargs)


def _insert_event(conn, *, event_id=None, symbol="TSLA", outcome_type="missed_move",
                  outcome_reason_code="target_already_crossed",
                  evaluated_at=None, trigger_id=None, profile_id="moderate"):
    """Insert a minimal fast_path_events row for testing."""
    if event_id is None:
        event_id = str(uuid.uuid4())
    if trigger_id is None:
        trigger_id = str(uuid.uuid4())
    if evaluated_at is None:
        evaluated_at = _iso(NOW)
    conn.execute(
        text(
            "INSERT INTO fast_path_events "
            "(event_id, trigger_id, symbol, profile_id, setup_type, direction, "
            "outcome_type, outcome_reason_code, current_price, evaluated_at, "
            "annotation_status, entry_price, stop_price, target_price) "
            "VALUES (:event_id, :trigger_id, :symbol, :profile_id, 'momentum_fade', "
            "'SHORT', :outcome_type, :outcome_reason_code, 342.08, :evaluated_at, "
            "'annotation_pending', 351.61, 355.00, 348.97)"
        ),
        {
            "event_id": event_id,
            "trigger_id": trigger_id,
            "symbol": symbol,
            "profile_id": profile_id,
            "outcome_type": outcome_type,
            "outcome_reason_code": outcome_reason_code,
            "evaluated_at": evaluated_at,
        },
    )
    conn.commit()
    return event_id


def _insert_trigger(conn, *, trigger_id=None, symbol="TSLA", state="fired",
                    fired_at=None, expires_at=None):
    """Insert a minimal fast_path_triggers row for testing."""
    if trigger_id is None:
        trigger_id = str(uuid.uuid4())
    if expires_at is None:
        expires_at = _iso(NOW + timedelta(minutes=5))
    conn.execute(
        text(
            "INSERT INTO fast_path_triggers "
            "(trigger_id, symbol, profile_id, direction, setup_type, "
            "trigger_type, trigger_level, entry_price, stop_price, target_price, "
            "state, registered_at, expires_at, fired_at) "
            "VALUES (:trigger_id, :symbol, 'moderate', 'SHORT', 'momentum_fade', "
            "'entry_zone', 351.61, 351.61, 355.00, 348.97, "
            ":state, :registered_at, :expires_at, :fired_at)"
        ),
        {
            "trigger_id": trigger_id,
            "symbol": symbol,
            "state": state,
            "registered_at": _iso(NOW - timedelta(minutes=2)),
            "expires_at": expires_at,
            "fired_at": fired_at,
        },
    )
    conn.commit()
    return trigger_id


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
# Tests: generate_template_narration — each outcome type produces valid narration
# ---------------------------------------------------------------------------


class TestTemplateNarration:
    """Each outcome type produces a non-empty narration containing the symbol."""

    def test_missed_move_dict(self):
        outcome = _make_outcome_dict("missed_move")
        narration = generate_template_narration(outcome)
        assert narration
        assert "TSLA" in narration
        assert "target" in narration.lower()

    def test_missed_move_dataclass(self):
        outcome = _make_outcome_dataclass("missed_move")
        narration = generate_template_narration(outcome)
        assert narration
        assert "TSLA" in narration
        assert "target" in narration.lower()

    def test_watch_created_dict(self):
        outcome = _make_outcome_dict("watch_created", outcome_reason_code="awaiting_confirmation")
        narration = generate_template_narration(outcome)
        assert narration
        assert "TSLA" in narration
        assert "watch" in narration.lower()

    def test_watch_created_dataclass(self):
        outcome = _make_outcome_dataclass("watch_created", outcome_reason_code="awaiting_confirmation")
        narration = generate_template_narration(outcome)
        assert narration
        assert "TSLA" in narration
        assert "watch" in narration.lower()

    def test_pending_order_created_dict(self):
        outcome = _make_outcome_dict("pending_order_created", outcome_reason_code="price_away_limit_valid")
        narration = generate_template_narration(outcome)
        assert narration
        assert "TSLA" in narration
        assert "pending" in narration.lower() or "limit" in narration.lower()

    def test_pending_order_created_dataclass(self):
        outcome = _make_outcome_dataclass("pending_order_created", outcome_reason_code="price_away_limit_valid")
        narration = generate_template_narration(outcome)
        assert narration
        assert "TSLA" in narration
        assert "pending" in narration.lower() or "limit" in narration.lower()

    def test_trade_executed_dict(self):
        outcome = _make_outcome_dict("trade_executed", outcome_reason_code="all_gates_passed")
        narration = generate_template_narration(outcome)
        assert narration
        assert "TSLA" in narration
        assert "trade" in narration.lower() or "executed" in narration.lower()

    def test_trade_executed_dataclass(self):
        outcome = _make_outcome_dataclass("trade_executed", outcome_reason_code="all_gates_passed")
        narration = generate_template_narration(outcome)
        assert narration
        assert "TSLA" in narration
        assert "trade" in narration.lower() or "executed" in narration.lower()

    def test_stand_down_dict(self):
        outcome = _make_outcome_dict("stand_down", outcome_reason_code="cooldown:standard_cooldown")
        narration = generate_template_narration(outcome)
        assert narration
        assert "TSLA" in narration
        assert "blocked" in narration.lower()

    def test_stand_down_dataclass(self):
        outcome = _make_outcome_dataclass("stand_down", outcome_reason_code="invalid_geometry")
        narration = generate_template_narration(outcome)
        assert narration
        assert "TSLA" in narration
        assert "blocked" in narration.lower()

    def test_watch_promoted_dict(self):
        outcome = _make_outcome_dict("watch_promoted", outcome_reason_code="matured")
        narration = generate_template_narration(outcome)
        assert narration
        assert "TSLA" in narration
        assert "matured" in narration.lower() or "watch" in narration.lower()

    def test_watch_promoted_dataclass(self):
        outcome = _make_outcome_dataclass("watch_promoted", outcome_reason_code="matured")
        narration = generate_template_narration(outcome)
        assert narration
        assert "TSLA" in narration
        assert "matured" in narration.lower() or "watch" in narration.lower()

    def test_unrecognized_outcome_type_returns_fallback(self):
        outcome = _make_outcome_dict("unknown_type", outcome_reason_code="something")
        narration = generate_template_narration(outcome)
        assert narration
        assert "TSLA" in narration

    def test_different_symbol(self):
        outcome = _make_outcome_dict("trade_executed", symbol="AAPL")
        narration = generate_template_narration(outcome)
        assert "AAPL" in narration


# ---------------------------------------------------------------------------
# Tests: build_stream_event — returned dict has all required fields
# ---------------------------------------------------------------------------


REQUIRED_STREAM_FIELDS = {
    "symbol",
    "outcome_type",
    "outcome_reason_code",
    "entry_price",
    "stop_price",
    "target_price",
    "current_price",
    "timestamp",
    "profile_id",
    "setup_type",
    "narration",
}


class TestBuildStreamEvent:
    """Stream event contains all required fields per Requirement 8.1."""

    def test_all_required_keys_present(self):
        event_row = {
            "symbol": "TSLA",
            "outcome_type": "trade_executed",
            "outcome_reason_code": "all_gates_passed",
            "entry_price": 351.61,
            "stop_price": 355.00,
            "target_price": 348.97,
            "current_price": 342.08,
            "evaluated_at": _iso(NOW),
            "profile_id": "moderate",
            "setup_type": "momentum_fade",
            "narration": "TSLA setup triggered at 342.08; trade executed with 2.5:1 reward/risk.",
            "narration_source": "template",
            "annotation_status": "annotation_pending",
        }
        result = build_stream_event(event_row)
        assert REQUIRED_STREAM_FIELDS.issubset(result.keys())

    def test_field_values_match_input(self):
        event_row = {
            "symbol": "AAPL",
            "outcome_type": "missed_move",
            "outcome_reason_code": "target_already_crossed",
            "entry_price": 180.00,
            "stop_price": 183.00,
            "target_price": 175.00,
            "current_price": 174.50,
            "evaluated_at": _iso(NOW),
            "profile_id": "aggressive",
            "setup_type": "gap_and_go",
            "narration": "AAPL moved past target.",
            "narration_source": "template",
            "annotation_status": "annotation_pending",
        }
        result = build_stream_event(event_row)
        assert result["symbol"] == "AAPL"
        assert result["outcome_type"] == "missed_move"
        assert result["outcome_reason_code"] == "target_already_crossed"
        assert result["entry_price"] == 180.00
        assert result["stop_price"] == 183.00
        assert result["target_price"] == 175.00
        assert result["current_price"] == 174.50
        assert result["timestamp"] == _iso(NOW)
        assert result["profile_id"] == "aggressive"
        assert result["setup_type"] == "gap_and_go"

    def test_narration_uses_llm_enriched_when_annotated(self):
        event_row = {
            "symbol": "TSLA",
            "outcome_type": "trade_executed",
            "outcome_reason_code": "all_gates_passed",
            "entry_price": 351.61,
            "stop_price": 355.00,
            "target_price": 348.97,
            "current_price": 342.08,
            "evaluated_at": _iso(NOW),
            "profile_id": "moderate",
            "setup_type": "momentum_fade",
            "narration": "LLM-enriched narration with thesis context.",
            "narration_source": "llm_enriched",
            "annotation_status": "annotated",
        }
        result = build_stream_event(event_row)
        assert result["narration"] == "LLM-enriched narration with thesis context."

    def test_narration_falls_back_to_template_when_no_annotation(self):
        event_row = {
            "symbol": "NVDA",
            "outcome_type": "stand_down",
            "outcome_reason_code": "invalid_geometry",
            "entry_price": 500.00,
            "stop_price": 510.00,
            "target_price": 480.00,
            "current_price": 505.00,
            "evaluated_at": _iso(NOW),
            "profile_id": "moderate",
            "setup_type": "technical_breakout",
            "narration": None,
            "narration_source": "template",
            "annotation_status": "annotation_pending",
        }
        result = build_stream_event(event_row)
        # Should regenerate template narration from fields
        assert result["narration"]
        assert "NVDA" in result["narration"]

    def test_minimal_event_row_still_produces_all_keys(self):
        """Even a sparse row produces all required keys (with defaults)."""
        event_row = {
            "symbol": "AMD",
            "outcome_type": "stand_down",
            "outcome_reason_code": "stale_market_data",
            "evaluated_at": _iso(NOW),
        }
        result = build_stream_event(event_row)
        assert REQUIRED_STREAM_FIELDS.issubset(result.keys())
        assert result["symbol"] == "AMD"


# ---------------------------------------------------------------------------
# Tests: generate_cycle_summary — counts are accurate
# ---------------------------------------------------------------------------


class TestCycleSummary:
    """Cycle summary counts match data in the database."""

    def test_empty_window_returns_zeros(self, engine):
        cycle_start = _iso(NOW - timedelta(minutes=30))
        cycle_end = _iso(NOW + timedelta(minutes=30))
        summary = generate_cycle_summary(engine, cycle_start, cycle_end)
        assert summary["total_events"] == 0
        assert summary["outcome_counts"] == {}
        assert summary["total_triggers_fired"] == 0
        assert summary["total_triggers_expired"] == 0
        assert summary["total_triggers_evaluated"] == 0
        assert summary["cycle_start"] == cycle_start
        assert summary["cycle_end"] == cycle_end

    def test_counts_events_by_outcome_type(self, engine):
        cycle_start = _iso(NOW - timedelta(minutes=5))
        cycle_end = _iso(NOW + timedelta(minutes=5))

        with engine.connect() as conn:
            _insert_event(conn, outcome_type="missed_move",
                          outcome_reason_code="target_already_crossed",
                          evaluated_at=_iso(NOW))
            _insert_event(conn, outcome_type="missed_move",
                          outcome_reason_code="target_already_crossed",
                          evaluated_at=_iso(NOW + timedelta(seconds=10)))
            _insert_event(conn, outcome_type="stand_down",
                          outcome_reason_code="invalid_geometry",
                          evaluated_at=_iso(NOW + timedelta(seconds=20)))
            _insert_event(conn, outcome_type="trade_executed",
                          outcome_reason_code="all_gates_passed",
                          evaluated_at=_iso(NOW + timedelta(seconds=30)))

        summary = generate_cycle_summary(engine, cycle_start, cycle_end)
        assert summary["total_events"] == 4
        assert summary["outcome_counts"]["missed_move"] == 2
        assert summary["outcome_counts"]["stand_down"] == 1
        assert summary["outcome_counts"]["trade_executed"] == 1

    def test_excludes_events_outside_window(self, engine):
        cycle_start = _iso(NOW)
        cycle_end = _iso(NOW + timedelta(minutes=5))

        with engine.connect() as conn:
            # Inside window
            _insert_event(conn, outcome_type="trade_executed",
                          outcome_reason_code="all_gates_passed",
                          evaluated_at=_iso(NOW + timedelta(seconds=30)))
            # Outside window (before)
            _insert_event(conn, outcome_type="missed_move",
                          outcome_reason_code="target_already_crossed",
                          evaluated_at=_iso(NOW - timedelta(minutes=10)))
            # Outside window (after)
            _insert_event(conn, outcome_type="stand_down",
                          outcome_reason_code="stale_data",
                          evaluated_at=_iso(NOW + timedelta(minutes=10)))

        summary = generate_cycle_summary(engine, cycle_start, cycle_end)
        assert summary["total_events"] == 1
        assert summary["outcome_counts"].get("trade_executed") == 1
        assert "missed_move" not in summary["outcome_counts"]
        assert "stand_down" not in summary["outcome_counts"]

    def test_counts_fired_triggers(self, engine):
        cycle_start = _iso(NOW - timedelta(minutes=5))
        cycle_end = _iso(NOW + timedelta(minutes=5))

        with engine.connect() as conn:
            _insert_trigger(conn, state="fired", fired_at=_iso(NOW))
            _insert_trigger(conn, state="fired", fired_at=_iso(NOW + timedelta(seconds=15)))
            # Fired outside window
            _insert_trigger(conn, state="fired", fired_at=_iso(NOW - timedelta(minutes=10)))

        summary = generate_cycle_summary(engine, cycle_start, cycle_end)
        assert summary["total_triggers_fired"] == 2

    def test_counts_expired_triggers(self, engine):
        cycle_start = _iso(NOW - timedelta(minutes=5))
        cycle_end = _iso(NOW + timedelta(minutes=5))

        with engine.connect() as conn:
            _insert_trigger(conn, state="expired",
                            expires_at=_iso(NOW + timedelta(minutes=1)))
            _insert_trigger(conn, state="expired",
                            expires_at=_iso(NOW + timedelta(minutes=2)))
            # Expired outside window
            _insert_trigger(conn, state="expired",
                            expires_at=_iso(NOW - timedelta(minutes=10)))

        summary = generate_cycle_summary(engine, cycle_start, cycle_end)
        assert summary["total_triggers_expired"] == 2

    def test_total_triggers_evaluated_is_fired_plus_expired(self, engine):
        cycle_start = _iso(NOW - timedelta(minutes=5))
        cycle_end = _iso(NOW + timedelta(minutes=5))

        with engine.connect() as conn:
            _insert_trigger(conn, state="fired", fired_at=_iso(NOW))
            _insert_trigger(conn, state="expired",
                            expires_at=_iso(NOW + timedelta(minutes=1)))

        summary = generate_cycle_summary(engine, cycle_start, cycle_end)
        assert summary["total_triggers_evaluated"] == 2
        assert summary["total_triggers_evaluated"] == (
            summary["total_triggers_fired"] + summary["total_triggers_expired"]
        )

    def test_summary_on_db_error_returns_zeros(self):
        """Fail-open: DB errors return zeroed summary, no exception."""
        from unittest.mock import MagicMock

        bad_engine = MagicMock()
        bad_engine.connect.side_effect = RuntimeError("db unavailable")

        cycle_start = _iso(NOW)
        cycle_end = _iso(NOW + timedelta(minutes=5))
        summary = generate_cycle_summary(bad_engine, cycle_start, cycle_end)
        assert summary["total_events"] == 0
        assert summary["outcome_counts"] == {}
        assert summary["cycle_start"] == cycle_start
        assert summary["cycle_end"] == cycle_end
