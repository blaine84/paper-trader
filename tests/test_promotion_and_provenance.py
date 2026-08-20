"""Tests for promotion path and provenance — setup_watch_manager.

Requirements: 5.1-5.8, 6.1-6.7, 11.1-11.8, 12.1-12.6
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from utils.setup_watch_manager import (
    _promote_ready_watch,
    _validate_source_provenance,
    create_setup_watch,
)
from utils.setup_watch_registry import (
    SetupWatch,
    SetupWatchRegistry,
    SetupWatchRegistryError,
    WatchState,
)

NOW = datetime(2026, 8, 17, 14, 30, 0, tzinfo=timezone.utc)
FUTURE = NOW + timedelta(hours=24)
PROFILE = "moderate"
CYCLE = "cycle_001"


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _make_watch(
    *,
    watch_id: str = "watch-001",
    profile_id: str = PROFILE,
    symbol: str = "AAPL",
    side: str = "BUY",
    setup_type: str = "support_bounce_swing",
    state: WatchState = WatchState.READY,
    maturity_score: float = 0.85,
    observed_cycles: int = 5,
    promoted_cycle_id: str | None = None,
    draft_geometry_json: str | None = None,
    entry_zone_json: str | None = None,
    last_evaluation_json: str | None = None,
    ready_at: datetime | None = None,
    ready_reference_price: float | None = None,
    source_type: str = "analyst",
    source_id: str | None = "sig-001",
    source_cycle_id: str = CYCLE,
    thesis: str = "A solid support bounce thesis for testing",
) -> SetupWatch:
    """Build a SetupWatch for test use."""
    return SetupWatch(
        watch_id=watch_id,
        profile_id=profile_id,
        symbol=symbol,
        side=side,
        setup_type=setup_type,
        state=state,
        thesis=thesis,
        source_type=source_type,
        source_id=source_id,
        source_cycle_id=source_cycle_id,
        maturation_conditions_json=json.dumps([
            {"type": "price_zone", "params": {"low": 99.0, "high": 101.0}, "weight": 1.0},
            {"type": "level_reclaim", "params": {"level": 100.0, "side": "BUY"}, "weight": 1.0},
            {"type": "support_hold", "params": {"level": 98.0, "tolerance_pct": 0.5}, "weight": 1.0},
        ]),
        invalidation_conditions_json=json.dumps([
            {"type": "price_breach", "params": {"level": 95.0, "direction": "below"}},
        ]),
        last_evaluation_json=last_evaluation_json,
        entry_zone_json=entry_zone_json,
        draft_geometry_json=draft_geometry_json,
        maturity_score=maturity_score,
        created_at=NOW,
        updated_at=NOW,
        expires_at=FUTURE,
        state_changed_at=NOW,
        observed_cycles=observed_cycles,
        ready_at=ready_at or NOW,
        ready_reference_price=ready_reference_price or 150.0,
        terminal_reason=None,
        promoted_cycle_id=promoted_cycle_id,
        execution_ref_type=None,
        execution_ref_id=None,
        integrity_hash="test_hash_001",
    )


@dataclass(frozen=True)
class FakeReadinessResult:
    """Mimics MarketDataReadinessResult for test mocks."""
    proceed: bool
    reason_codes: tuple[str, ...]
    missing_data_types: tuple[str, ...]


# ────────────────────────────────────────────────────────────────────────────
# 16.2: _promote_ready_watch succeeds when all criteria met
# ────────────────────────────────────────────────────────────────────────────


@patch("utils.setup_watch_manager.MARKET_DATA_RELIABILITY_MODE", "enabled")
@patch("utils.setup_watch_manager.SETUP_WATCH_MISSED_MOVE_ENABLED", True)
@patch("utils.setup_watch_manager.SETUP_WATCH_REALTIME_MODE", "enabled")
def test_promote_ready_watch_succeeds_when_all_criteria_met():
    """16.2: Promotion succeeds when idempotency, missed-move, reliability, CAS all pass."""
    from utils.missed_move_detector import MissedMoveResult

    watch = _make_watch(
        draft_geometry_json=json.dumps({"entry": "100", "stop": "95", "target": "110", "risk_reward": "2.0"}),
    )
    registry = MagicMock(spec=SetupWatchRegistry)
    registry.transition_state = MagicMock(return_value=None)
    registry._emit_event = MagicMock(return_value=None)

    signals = {"AAPL": {"current_price": 101.0, "key_levels": {"support": 98.0}}}

    with patch(
        "utils.missed_move_detector.check_missed_move"
    ) as mock_missed, patch(
        "utils.market_data_reliability.pipeline_integration.check_market_data_readiness"
    ) as mock_readiness:
        mock_missed.return_value = MissedMoveResult(
            watch_id="watch-001", missed=False,
            target_price=Decimal("110"), current_price=Decimal("101"),
            side="BUY", reason=None,
        )
        mock_readiness.return_value = FakeReadinessResult(
            proceed=True, reason_codes=(), missing_data_types=(),
        )

        result = _promote_ready_watch(
            registry, watch, cycle_id="cycle_002", signals=signals,
        )

    assert result is True
    registry.transition_state.assert_called_once_with(
        "watch-001", WatchState.READY, WatchState.PROMOTED,
        promoted_cycle_id="cycle_002",
    )


# ────────────────────────────────────────────────────────────────────────────
# 16.3: Promotion blocked by idempotency (promoted_cycle_id == current)
# ────────────────────────────────────────────────────────────────────────────


def test_promote_blocked_by_idempotency(caplog):
    """16.3: Skip promotion if watch already promoted this cycle."""
    watch = _make_watch(promoted_cycle_id="cycle_002")
    registry = MagicMock(spec=SetupWatchRegistry)

    with caplog.at_level(logging.WARNING):
        result = _promote_ready_watch(
            registry, watch, cycle_id="cycle_002", signals={},
        )

    assert result is False
    registry.transition_state.assert_not_called()
    assert "idempotency" in caplog.text.lower()


# ────────────────────────────────────────────────────────────────────────────
# 16.4: Promotion blocked by missed-move (target crossed)
# ────────────────────────────────────────────────────────────────────────────


@patch("utils.setup_watch_manager.MARKET_DATA_RELIABILITY_MODE", "enabled")
@patch("utils.setup_watch_manager.SETUP_WATCH_MISSED_MOVE_ENABLED", True)
def test_promote_blocked_by_missed_move(caplog):
    """16.4: If target crossed, watch transitions to MISSED, promotion returns False."""
    from utils.missed_move_detector import MissedMoveResult

    watch = _make_watch(
        draft_geometry_json=json.dumps({"entry": "100", "stop": "95", "target": "110", "risk_reward": "2.0"}),
    )
    registry = MagicMock(spec=SetupWatchRegistry)

    signals = {"AAPL": {"current_price": 115.0, "key_levels": {}}}

    with patch(
        "utils.missed_move_detector.check_missed_move"
    ) as mock_missed, patch(
        "utils.missed_move_detector.apply_missed_move_transition"
    ) as mock_apply:
        mock_missed.return_value = MissedMoveResult(
            watch_id="watch-001", missed=True,
            target_price=Decimal("110"), current_price=Decimal("115"),
            side="BUY", reason="target_already_crossed",
        )

        with caplog.at_level(logging.WARNING):
            result = _promote_ready_watch(
                registry, watch, cycle_id="cycle_002", signals=signals,
            )

    assert result is False
    mock_apply.assert_called_once()
    assert "missed move" in caplog.text.lower() or "missed" in caplog.text.lower()


# ────────────────────────────────────────────────────────────────────────────
# 16.5: Promotion deferred by reliability (proceed=False)
# ────────────────────────────────────────────────────────────────────────────


@patch("utils.setup_watch_manager.MARKET_DATA_RELIABILITY_MODE", "enabled")
@patch("utils.setup_watch_manager.SETUP_WATCH_MISSED_MOVE_ENABLED", False)
@patch("utils.setup_watch_manager.SETUP_WATCH_REALTIME_MODE", "enabled")
def test_promote_deferred_by_reliability():
    """16.5: Market-data reliability returns proceed=False → emits deferral event."""
    watch = _make_watch()
    registry = MagicMock(spec=SetupWatchRegistry)
    registry._emit_event = MagicMock(return_value=None)

    signals = {"AAPL": {"current_price": 101.0}}

    with patch(
        "utils.market_data_reliability.pipeline_integration.check_market_data_readiness"
    ) as mock_readiness:
        mock_readiness.return_value = FakeReadinessResult(
            proceed=False,
            reason_codes=("stale_quote", "low_confidence"),
            missing_data_types=("realtime_quote",),
        )

        result = _promote_ready_watch(
            registry, watch, cycle_id="cycle_002", signals=signals,
        )

    assert result is False
    # Should emit promotion_deferred_market_data event
    registry._emit_event.assert_called()
    call_kwargs = registry._emit_event.call_args[1]
    assert call_kwargs["event_type"] == "promotion_deferred_market_data"
    event_data = json.loads(call_kwargs["event_data"])
    assert "stale_quote" in event_data["reason_codes"]
    registry.transition_state.assert_not_called()


# ────────────────────────────────────────────────────────────────────────────
# 16.6: Reliability exception → defer with reason_code="reliability_check_error"
# ────────────────────────────────────────────────────────────────────────────


@patch("utils.setup_watch_manager.MARKET_DATA_RELIABILITY_MODE", "enabled")
@patch("utils.setup_watch_manager.SETUP_WATCH_MISSED_MOVE_ENABLED", False)
@patch("utils.setup_watch_manager.SETUP_WATCH_REALTIME_MODE", "enabled")
def test_promote_reliability_exception_defers(caplog):
    """16.6: Exception from reliability check → treat as unreliable, defer."""
    watch = _make_watch()
    registry = MagicMock(spec=SetupWatchRegistry)
    registry._emit_event = MagicMock(return_value=None)

    signals = {"AAPL": {"current_price": 101.0}}

    with patch(
        "utils.market_data_reliability.pipeline_integration.check_market_data_readiness"
    ) as mock_readiness:
        mock_readiness.side_effect = RuntimeError("Connection timeout")

        with caplog.at_level(logging.WARNING):
            result = _promote_ready_watch(
                registry, watch, cycle_id="cycle_002", signals=signals,
            )

    assert result is False
    registry._emit_event.assert_called()
    call_kwargs = registry._emit_event.call_args[1]
    event_data = json.loads(call_kwargs["event_data"])
    assert "reliability_check_error" in event_data["reason_codes"]


# ────────────────────────────────────────────────────────────────────────────
# 16.7: Reliability check skipped when MARKET_DATA_RELIABILITY_MODE="disabled"
# ────────────────────────────────────────────────────────────────────────────


@patch("utils.setup_watch_manager.MARKET_DATA_RELIABILITY_MODE", "disabled")
@patch("utils.setup_watch_manager.SETUP_WATCH_MISSED_MOVE_ENABLED", False)
def test_promote_reliability_skipped_when_disabled():
    """16.7: When MARKET_DATA_RELIABILITY_MODE='disabled', reliability check skipped."""
    watch = _make_watch()
    registry = MagicMock(spec=SetupWatchRegistry)
    registry.transition_state = MagicMock(return_value=None)
    registry._emit_event = MagicMock(return_value=None)

    signals = {"AAPL": {"current_price": 101.0, "key_levels": {"support": 98.0}}}

    with patch(
        "utils.market_data_reliability.pipeline_integration.check_market_data_readiness"
    ) as mock_readiness:
        result = _promote_ready_watch(
            registry, watch, cycle_id="cycle_002", signals=signals,
        )

    # Reliability check should NOT have been called
    mock_readiness.assert_not_called()
    assert result is True
    registry.transition_state.assert_called_once()


# ────────────────────────────────────────────────────────────────────────────
# 16.8: Realtime "observe" mode doesn't block promotion from scheduled path
# ────────────────────────────────────────────────────────────────────────────


@patch("utils.setup_watch_manager.MARKET_DATA_RELIABILITY_MODE", "enabled")
@patch("utils.setup_watch_manager.SETUP_WATCH_MISSED_MOVE_ENABLED", False)
@patch("utils.setup_watch_manager.SETUP_WATCH_REALTIME_MODE", "observe")
def test_promote_observe_mode_does_not_block():
    """16.8: In observe mode, reliability is evaluated but does NOT block promotion."""
    watch = _make_watch()
    registry = MagicMock(spec=SetupWatchRegistry)
    registry.transition_state = MagicMock(return_value=None)
    registry._emit_event = MagicMock(return_value=None)

    signals = {"AAPL": {"current_price": 101.0, "key_levels": {"support": 98.0}}}

    with patch(
        "utils.market_data_reliability.pipeline_integration.check_market_data_readiness"
    ) as mock_readiness:
        mock_readiness.return_value = FakeReadinessResult(
            proceed=False,
            reason_codes=("stale_quote",),
            missing_data_types=("realtime_quote",),
        )

        result = _promote_ready_watch(
            registry, watch, cycle_id="cycle_002", signals=signals,
        )

    # Should succeed despite reliability saying proceed=False
    assert result is True
    mock_readiness.assert_called_once()
    registry.transition_state.assert_called_once()


# ────────────────────────────────────────────────────────────────────────────
# 16.9: CAS race (rowcount == 0) logs WARNING, doesn't raise
# ────────────────────────────────────────────────────────────────────────────


@patch("utils.setup_watch_manager.MARKET_DATA_RELIABILITY_MODE", "disabled")
@patch("utils.setup_watch_manager.SETUP_WATCH_MISSED_MOVE_ENABLED", False)
def test_promote_cas_race_logs_warning(caplog):
    """16.9: CAS race (transition raises RegistryError) logs WARNING, returns False."""
    watch = _make_watch()
    registry = MagicMock(spec=SetupWatchRegistry)
    registry.transition_state.side_effect = SetupWatchRegistryError(
        "CAS failed: rowcount == 0"
    )

    with caplog.at_level(logging.WARNING):
        result = _promote_ready_watch(
            registry, watch, cycle_id="cycle_002", signals={},
        )

    assert result is False
    assert "race" in caplog.text.lower() or "cas" in caplog.text.lower()


# ────────────────────────────────────────────────────────────────────────────
# 16.10: Evidence package has all required keys, NULLs as JSON null
# ────────────────────────────────────────────────────────────────────────────


@patch("utils.setup_watch_manager.MARKET_DATA_RELIABILITY_MODE", "disabled")
@patch("utils.setup_watch_manager.SETUP_WATCH_MISSED_MOVE_ENABLED", False)
def test_promote_evidence_package_all_keys():
    """16.10: Evidence package includes all required keys with NULL fields as JSON null."""
    watch = _make_watch(
        last_evaluation_json=None,
        entry_zone_json=None,
        draft_geometry_json=None,
        ready_at=NOW,
        ready_reference_price=150.0,
        source_type="analyst",
        source_id="sig-001",
        source_cycle_id=CYCLE,
    )
    registry = MagicMock(spec=SetupWatchRegistry)
    registry.transition_state = MagicMock(return_value=None)
    registry._emit_event = MagicMock(return_value=None)

    signals = {"AAPL": {"current_price": 101.0, "key_levels": {"support": 98.0}}}

    result = _promote_ready_watch(
        registry, watch, cycle_id="cycle_002", signals=signals,
    )

    assert result is True
    # Verify evidence package structure from the emitted event
    registry._emit_event.assert_called()
    call_kwargs = registry._emit_event.call_args[1]
    event_data = json.loads(call_kwargs["event_data"])
    evidence = event_data["evidence_package"]

    # All required keys must be present
    required_keys = [
        "watch_id", "setup_type", "thesis", "maturity_score",
        "observed_cycles", "last_evaluation_json", "ready_reference_price",
        "ready_at", "source_trigger", "entry_zone", "draft_geometry",
        "current_price", "key_levels", "source_provenance",
    ]
    for key in required_keys:
        assert key in evidence, f"Missing key: {key}"

    # NULL fields present as JSON null (not omitted)
    assert evidence["last_evaluation_json"] is None
    assert evidence["entry_zone"] is None
    assert evidence["draft_geometry"] is None

    # Provenance fields
    assert evidence["source_provenance"]["source_type"] == "analyst"
    assert evidence["source_provenance"]["source_id"] == "sig-001"
    assert evidence["source_provenance"]["source_cycle_id"] == CYCLE


# ────────────────────────────────────────────────────────────────────────────
# 16.11: _validate_source_provenance rejects unknown source_type
# ────────────────────────────────────────────────────────────────────────────


def test_validate_provenance_rejects_unknown_type(caplog):
    """16.11: Unknown source_type → return False and log warning."""
    with caplog.at_level(logging.WARNING):
        result = _validate_source_provenance("unknown_source", "some-id")

    assert result is False
    assert "unknown" in caplog.text.lower()


# ────────────────────────────────────────────────────────────────────────────
# 16.12: _validate_source_provenance rejects null source_id for new types
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("source_type", ["price_monitor", "scout"])
def test_validate_provenance_rejects_null_id_for_new_types(source_type, caplog):
    """16.12: New types (price_monitor, scout) require source_id."""
    with caplog.at_level(logging.WARNING):
        result = _validate_source_provenance(source_type, None)

    assert result is False
    assert "source_id" in caplog.text.lower() or "requires" in caplog.text.lower()


# ────────────────────────────────────────────────────────────────────────────
# 16.13: _validate_source_provenance tolerates null source_id for legacy types
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("source_type", [
    "analyst", "candidate_reject", "market_state", "pm_defer",
])
def test_validate_provenance_tolerates_null_id_for_legacy(source_type, caplog):
    """16.13: Legacy types tolerate null source_id with DEBUG log."""
    with caplog.at_level(logging.DEBUG):
        result = _validate_source_provenance(source_type, None)

    assert result is True
    assert "tolerated" in caplog.text.lower() or "null" in caplog.text.lower()


# ────────────────────────────────────────────────────────────────────────────
# 16.14: source_cycle_id stored at creation time
# ────────────────────────────────────────────────────────────────────────────


@patch("utils.setup_watch_manager.SETUP_WATCH_MODE", "enabled")
def test_source_cycle_id_stored_at_creation(tmp_path):
    """16.14: source_cycle_id is stored when a watch is created."""
    from sqlalchemy import create_engine
    from db.schema import init_setup_watch_schema

    engine = create_engine("sqlite:///:memory:")
    init_setup_watch_schema(engine)

    cycle_id = "cycle_creation_test"
    watch_id = create_setup_watch(
        engine,
        symbol="TSLA",
        profile_id=PROFILE,
        side="BUY",
        setup_type="support_bounce_swing",
        thesis="Testing source cycle_id storage",
        source_type="analyst",
        source_id="sig-test",
        source_cycle_id=cycle_id,
        maturation_conditions=[
            {"type": "price_zone", "params": {"low": 99.0, "high": 101.0}, "weight": 1.0},
            {"type": "level_reclaim", "params": {"level": 100.0, "side": "BUY"}, "weight": 1.0},
            {"type": "support_hold", "params": {"level": 98.0, "tolerance_pct": 0.5}, "weight": 1.0},
        ],
        invalidation_conditions=[
            {"type": "price_breach", "params": {"level": 95.0, "direction": "below"}},
        ],
    )

    assert watch_id is not None

    # Verify it's stored in the database
    from sqlalchemy import text
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT source_cycle_id FROM setup_watches WHERE watch_id = :wid"),
            {"wid": watch_id},
        ).fetchone()

    assert row is not None
    assert row[0] == cycle_id


# ────────────────────────────────────────────────────────────────────────────
# 16.15: Draft geometry at creation (present with levels)
# ────────────────────────────────────────────────────────────────────────────


@patch("utils.setup_watch_manager.SETUP_WATCH_MODE", "enabled")
def test_draft_geometry_at_creation_with_levels():
    """16.15: Draft geometry computed when signal has sufficient key levels."""
    from sqlalchemy import create_engine
    from db.schema import init_setup_watch_schema

    engine = create_engine("sqlite:///:memory:")
    init_setup_watch_schema(engine)

    key_levels = {
        "support": 95.0,
        "resistance": 110.0,
    }

    watch_id = create_setup_watch(
        engine,
        symbol="AMD",
        profile_id=PROFILE,
        side="BUY",
        setup_type="support_bounce_swing",
        thesis="AMD support bounce with good levels",
        source_type="analyst",
        source_id="sig-002",
        source_cycle_id=CYCLE,
        maturation_conditions=[
            {"type": "price_zone", "params": {"low": 94.0, "high": 96.0}, "weight": 1.0},
            {"type": "level_reclaim", "params": {"level": 95.0, "side": "BUY"}, "weight": 1.0},
            {"type": "support_hold", "params": {"level": 94.0, "tolerance_pct": 0.5}, "weight": 1.0},
        ],
        invalidation_conditions=[
            {"type": "price_breach", "params": {"level": 90.0, "direction": "below"}},
        ],
        key_levels=key_levels,
    )

    assert watch_id is not None

    from sqlalchemy import text
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT entry_zone_json, draft_geometry_json "
                "FROM setup_watches WHERE watch_id = :wid"
            ),
            {"wid": watch_id},
        ).fetchone()

    # Both should be populated
    assert row[0] is not None  # entry_zone_json
    assert row[1] is not None  # draft_geometry_json

    # Verify geometry structure
    geom = json.loads(row[1])
    assert "entry" in geom
    assert "stop" in geom
    assert "target" in geom
    assert "risk_reward" in geom


# ────────────────────────────────────────────────────────────────────────────
# 16.16: Draft geometry NULL when signal lacks required levels
# ────────────────────────────────────────────────────────────────────────────


@patch("utils.setup_watch_manager.SETUP_WATCH_MODE", "enabled")
def test_draft_geometry_null_without_levels():
    """16.16: Draft geometry is NULL when signal lacks required levels."""
    from sqlalchemy import create_engine
    from db.schema import init_setup_watch_schema

    engine = create_engine("sqlite:///:memory:")
    init_setup_watch_schema(engine)

    # No key_levels provided → no geometry
    watch_id = create_setup_watch(
        engine,
        symbol="GOOG",
        profile_id=PROFILE,
        side="BUY",
        setup_type="support_bounce_swing",
        thesis="Google watch without key levels data",
        source_type="analyst",
        source_id="sig-003",
        source_cycle_id=CYCLE,
        maturation_conditions=[
            {"type": "price_zone", "params": {"low": 130.0, "high": 135.0}, "weight": 1.0},
            {"type": "level_reclaim", "params": {"level": 132.0, "side": "BUY"}, "weight": 1.0},
            {"type": "support_hold", "params": {"level": 130.0, "tolerance_pct": 0.5}, "weight": 1.0},
        ],
        invalidation_conditions=[
            {"type": "price_breach", "params": {"level": 125.0, "direction": "below"}},
        ],
        key_levels=None,
    )

    assert watch_id is not None

    from sqlalchemy import text
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT entry_zone_json, draft_geometry_json "
                "FROM setup_watches WHERE watch_id = :wid"
            ),
            {"wid": watch_id},
        ).fetchone()

    # Both should be NULL
    assert row[0] is None  # entry_zone_json
    assert row[1] is None  # draft_geometry_json


# ────────────────────────────────────────────────────────────────────────────
# 16.17: Entry zone tightening (narrower replaces, equal doesn't)
# ────────────────────────────────────────────────────────────────────────────


def test_entry_zone_tightening():
    """16.17: Narrower zone replaces stored zone, equal-width does not."""
    from utils.draft_geometry import EntryZone, should_replace_entry_zone, entry_zone_to_json

    existing_json = json.dumps({"low": "95.00", "high": "105.00"})  # width = 10

    # Tighter zone should replace
    tighter = EntryZone(low=Decimal("97.00"), high=Decimal("103.00"))  # width = 6
    assert should_replace_entry_zone(existing_json, tighter) is True

    # Equal-width zone should NOT replace
    equal = EntryZone(low=Decimal("96.00"), high=Decimal("106.00"))  # width = 10
    assert should_replace_entry_zone(existing_json, equal) is False

    # Wider zone should NOT replace
    wider = EntryZone(low=Decimal("90.00"), high=Decimal("110.00"))  # width = 20
    assert should_replace_entry_zone(existing_json, wider) is False

    # None existing → should replace (any zone is better than none)
    assert should_replace_entry_zone(None, tighter) is True
