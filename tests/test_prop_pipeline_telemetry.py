"""
Property-based tests for candidate pipeline telemetry.

Feature: candidate-blocker-mitigation
Properties: 10, 11, 12, 13

This file contains:
- Property 10: Gate events stop after first failure
- Property 11: First blocking stage uses fixed ordering
- Property 12: Execution failure event structure with truncation
- Property 13: Lifecycle checklist correctness and incomplete event
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from hypothesis import given, strategies as st, settings, assume
from sqlalchemy import create_engine, text

from utils.candidate_pipeline import (
    _write_gate_events,
    _determine_first_blocking_stage,
    _write_execution_failed_event,
    _write_candidate_event,
    GATE_PIPELINE_ORDER,
)
from utils.candidate_registry import CandidateRecord
from utils.lifecycle_checklist import LifecycleChecklist, write_lifecycle_checklist


# ---------------------------------------------------------------------------
# Helpers for Properties 10, 11, 12
# ---------------------------------------------------------------------------


def _make_candidate_record(
    candidate_id: str | None = None,
    symbol: str = "AAPL",
    direction: str = "BUY",
    candidate_type: str = "intraday",
) -> CandidateRecord:
    """Create a minimal CandidateRecord for testing event writers."""
    cid = candidate_id or str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    return CandidateRecord(
        candidate_id=cid,
        cycle_id="cycle-test-1",
        profile_id="moderate",
        symbol=symbol,
        direction=direction,
        setup_type="breakout",
        geometry_name="analyst_geometry",
        entry_price=150.0,
        stop_price=145.0,
        target_price=160.0,
        risk_reward=2.0,
        trigger="price_level",
        invalidation_basis="below_stop",
        target_basis="measured_move",
        source_signal_id="signal-1",
        signal_snapshot_json='{"source": "test"}',
        created_at=now,
        expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
        integrity_hash="hash123",
        candidate_type=candidate_type,
    )


def _create_events_engine():
    """Create an in-memory SQLite with just the pm_candidate_events table."""
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
    return eng


def _get_events(engine, candidate_id: str) -> list[dict]:
    """Retrieve all events for a candidate, ordered by id."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT event_type, event_data FROM pm_candidate_events WHERE candidate_id = :cid ORDER BY id"),
            {"cid": candidate_id},
        ).fetchall()
    return [{"event_type": r[0], "event_data": json.loads(r[1]) if r[1] else {}} for r in rows]


# ---------------------------------------------------------------------------
# Property 10: Gate events stop after first failure
# ---------------------------------------------------------------------------

# Strategy: generate a list of gate results in pipeline order
# Each gate is (gate_name, proceed: bool)
_gate_result_strategy = st.lists(
    st.tuples(
        st.sampled_from(GATE_PIPELINE_ORDER),
        st.booleans(),
    ),
    min_size=1,
    max_size=5,
    unique_by=lambda x: x[0],  # unique gate names
)


@given(gate_results=_gate_result_strategy)
@settings(max_examples=200)
def test_gate_events_stop_after_first_failure(gate_results):
    """Property 10: Gate events stop after first failure.

    For any accepted candidate entering the gate pipeline, gate events
    SHALL be written in fixed pipeline order (setup_quality, pre_trade_quality,
    catalyst_specificity, risk_geometry, concentration) and SHALL stop after
    the first gate that returns proceed=false. No gate events SHALL be emitted
    for gates after the first failure.

    Strategy: Generate gate results (list of (gate_name, proceed: bool) tuples
    in pipeline order). Simulate the event writing logic. Verify that no events
    after the first failure are written.

    **Validates: Requirements 5.3**
    """
    # Build gate_notes in the format _write_gate_events expects
    gate_notes = []
    for gate_name, proceed in gate_results:
        # The function maps canonical names or suffixed gate keys
        note = {
            "gate": gate_name,
            "decision": "pass" if proceed else "reject",
        }
        if not proceed:
            note["reason_code"] = f"{gate_name}_threshold"
        gate_notes.append(note)

    # Set up DB and candidate
    engine = _create_events_engine()
    candidate = _make_candidate_record()

    # Write gate events
    _write_gate_events(engine, candidate, "cycle-test-1", "moderate", gate_notes)

    # Retrieve persisted events
    events = _get_events(engine, candidate.candidate_id)

    # Determine expected events: iterate GATE_PIPELINE_ORDER, emit for each
    # gate that has a result, stop after first failure
    results_by_gate = {name: proceed for name, proceed in gate_results}
    expected_events = []
    for gate_name in GATE_PIPELINE_ORDER:
        if gate_name not in results_by_gate:
            continue
        proceed = results_by_gate[gate_name]
        expected_events.append({
            "event_type": "gate_pass" if proceed else "gate_fail",
            "gate_name": gate_name,
        })
        if not proceed:
            break  # Stop after first failure

    # Verify count matches
    assert len(events) == len(expected_events), (
        f"Expected {len(expected_events)} events, got {len(events)}. "
        f"Events: {events}"
    )

    # Verify each event matches expected order and type
    for i, (actual, expected) in enumerate(zip(events, expected_events)):
        assert actual["event_type"] == expected["event_type"], (
            f"Event {i}: expected type {expected['event_type']}, got {actual['event_type']}"
        )
        assert actual["event_data"]["gate_name"] == expected["gate_name"], (
            f"Event {i}: expected gate {expected['gate_name']}, "
            f"got {actual['event_data']['gate_name']}"
        )

    # Verify no events exist after first failure in pipeline order
    if events and events[-1]["event_type"] == "gate_fail":
        failing_gate = events[-1]["event_data"]["gate_name"]
        failing_idx = GATE_PIPELINE_ORDER.index(failing_gate)
        # No subsequent gates should have events
        later_gates = set(GATE_PIPELINE_ORDER[failing_idx + 1:])
        for event in events:
            assert event["event_data"]["gate_name"] not in later_gates, (
                f"Found event for gate {event['event_data']['gate_name']} "
                f"which comes after the failing gate {failing_gate}"
            )


# ---------------------------------------------------------------------------
# Property 11: First blocking stage uses fixed ordering
# ---------------------------------------------------------------------------

# The fixed ordering for first_blocking_stage
_BLOCKING_STAGE_ORDER = [
    "position_sizer",
    "setup_quality",
    "pre_trade_quality",
    "catalyst_specificity",
    "risk_geometry",
    "concentration",
]


@given(
    sizing_failed=st.booleans(),
    failing_gate_name=st.one_of(
        st.none(),
        st.sampled_from(GATE_PIPELINE_ORDER),
    ),
)
@settings(max_examples=200)
def test_first_blocking_stage_uses_fixed_ordering(sizing_failed, failing_gate_name):
    """Property 11: First blocking stage uses fixed ordering.

    For any accepted candidate that reaches a terminal state other than EXECUTED,
    _determine_first_blocking_stage() SHALL return the earliest rejecting stage:
    position_sizer → setup_quality → pre_trade_quality → catalyst_specificity →
    risk_geometry → concentration.

    Strategy: Generate combinations of (sizing_failed: bool, failing_gate_name:
    str|None). Verify the result follows the fixed ordering.

    **Validates: Requirements 5.4**
    """
    result = _determine_first_blocking_stage(sizing_failed, failing_gate_name)

    if not sizing_failed and failing_gate_name is None:
        # Neither failed — no blocking stage
        assert result is None, (
            f"Expected None when nothing failed, got {result}"
        )
    elif sizing_failed:
        # Position sizer is always first in the ordering
        assert result == "position_sizer", (
            f"Expected 'position_sizer' when sizing failed, got {result}"
        )
        # Verify position_sizer comes before any gate in the ordering
        if failing_gate_name is not None:
            sizer_idx = _BLOCKING_STAGE_ORDER.index("position_sizer")
            gate_idx = _BLOCKING_STAGE_ORDER.index(failing_gate_name)
            assert sizer_idx < gate_idx, (
                f"position_sizer (idx={sizer_idx}) should come before "
                f"{failing_gate_name} (idx={gate_idx}) in fixed ordering"
            )
    else:
        # Only gate failed — should return the failing gate name
        assert result == failing_gate_name, (
            f"Expected '{failing_gate_name}' when only that gate failed, got {result}"
        )
        # Verify the result is a valid stage in the ordering
        assert result in _BLOCKING_STAGE_ORDER, (
            f"Result '{result}' is not in the fixed ordering"
        )


# ---------------------------------------------------------------------------
# Property 12: Execution failure event structure with truncation
# ---------------------------------------------------------------------------

@given(
    failure_reason=st.text(min_size=0, max_size=3000),
    symbol=st.text(
        alphabet=st.characters(whitelist_categories=("Lu",), whitelist_characters=""),
        min_size=1,
        max_size=5,
    ),
    intended_action=st.sampled_from(["BUY", "SHORT"]),
    attempted_quantity=st.integers(min_value=1, max_value=10000),
)
@settings(max_examples=200)
def test_execution_failure_event_structure_with_truncation(
    failure_reason, symbol, intended_action, attempted_quantity
):
    """Property 12: Execution failure event structure with truncation.

    For any execution failure, the event SHALL contain candidate_id, profile,
    symbol, intended_action, attempted_quantity, and failure_reason truncated
    to at most 1024 characters.

    Strategy: Generate arbitrary failure_reason strings (including > 1024 chars).
    Call _write_execution_failed_event() against an in-memory DB. Verify the
    persisted event_data has all required fields and failure_reason <= 1024 chars.

    **Validates: Requirements 6.1, 6.2**
    """
    engine = _create_events_engine()
    candidate = _make_candidate_record(symbol=symbol, direction=intended_action)

    # Write the execution failed event
    _write_execution_failed_event(
        engine,
        candidate,
        "cycle-test-1",
        "moderate",
        intended_action,
        attempted_quantity,
        failure_reason,
    )

    # Retrieve the persisted event
    events = _get_events(engine, candidate.candidate_id)

    # Exactly one event should be written
    assert len(events) == 1, f"Expected 1 event, got {len(events)}"

    event = events[0]

    # Event type must be 'execution_failed' (distinct from pm_reject and gate_fail)
    assert event["event_type"] == "execution_failed", (
        f"Expected event_type 'execution_failed', got '{event['event_type']}'"
    )

    data = event["event_data"]

    # All required fields must be present
    assert "candidate_id" in data, "Missing 'candidate_id' in event_data"
    assert "profile" in data, "Missing 'profile' in event_data"
    assert "symbol" in data, "Missing 'symbol' in event_data"
    assert "intended_action" in data, "Missing 'intended_action' in event_data"
    assert "attempted_quantity" in data, "Missing 'attempted_quantity' in event_data"
    assert "failure_reason" in data, "Missing 'failure_reason' in event_data"

    # Field values must match inputs
    assert data["candidate_id"] == candidate.candidate_id
    assert data["profile"] == "moderate"
    assert data["symbol"] == symbol
    assert data["intended_action"] == intended_action
    assert data["attempted_quantity"] == attempted_quantity

    # failure_reason MUST be truncated to at most 1024 characters
    assert len(data["failure_reason"]) <= 1024, (
        f"failure_reason length {len(data['failure_reason'])} exceeds 1024"
    )

    # If input was <= 1024, it should be preserved exactly
    if len(failure_reason) <= 1024:
        assert data["failure_reason"] == failure_reason, (
            "failure_reason should be preserved when <= 1024 chars"
        )
    else:
        # If input was > 1024, it should be the first 1024 chars
        assert data["failure_reason"] == failure_reason[:1024], (
            "failure_reason should be truncated to first 1024 chars"
        )


# ---------------------------------------------------------------------------
# Database setup helpers for Property 13
# ---------------------------------------------------------------------------


def _create_test_engine():
    """Create an in-memory SQLite database with all tables needed for lifecycle checklist."""
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        conn.execute(text("""
            CREATE TABLE pm_candidates (
                id INTEGER PRIMARY KEY,
                candidate_id VARCHAR(36) NOT NULL UNIQUE,
                cycle_id VARCHAR(64) NOT NULL,
                profile_id VARCHAR(64) NOT NULL,
                symbol VARCHAR(10) NOT NULL,
                direction VARCHAR(10) NOT NULL,
                setup_type VARCHAR(64) NOT NULL,
                geometry_name VARCHAR(64) NOT NULL,
                entry_price REAL NOT NULL,
                stop_price REAL NOT NULL,
                target_price REAL NOT NULL,
                risk_reward REAL NOT NULL,
                source_signal_id VARCHAR(64) NOT NULL,
                signal_snapshot_json TEXT NOT NULL,
                state VARCHAR(32) DEFAULT 'registered',
                integrity_hash VARCHAR(64) NOT NULL,
                expires_at DATETIME NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY,
                symbol VARCHAR(10) NOT NULL,
                direction VARCHAR(5) NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                stop_price REAL,
                target_price REAL,
                status VARCHAR(8) DEFAULT 'open',
                profile VARCHAR(16) DEFAULT 'moderate',
                invalidators TEXT,
                candidate_lineage_id VARCHAR(36)
            )
        """))
        conn.execute(text("""
            CREATE TABLE positions (
                id INTEGER PRIMARY KEY,
                profile VARCHAR(16) DEFAULT 'moderate',
                symbol VARCHAR(10) NOT NULL,
                side VARCHAR(5) DEFAULT 'long',
                quantity REAL NOT NULL,
                avg_cost REAL NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE response_lineage_links (
                id INTEGER PRIMARY KEY,
                response_id VARCHAR(36) NOT NULL,
                lineage_id VARCHAR(36) NOT NULL,
                candidate_id VARCHAR(36),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE candidate_lifecycle_checklists (
                id INTEGER PRIMARY KEY,
                candidate_id VARCHAR(36) NOT NULL,
                trade_id VARCHAR(64) NOT NULL,
                cycle_id VARCHAR(64) NOT NULL,
                profile_id VARCHAR(64) NOT NULL,
                trade_row_created BOOLEAN NOT NULL DEFAULT 0,
                position_row_created_or_updated BOOLEAN NOT NULL DEFAULT 0,
                stop_registered BOOLEAN NOT NULL DEFAULT 0,
                target_registered BOOLEAN NOT NULL DEFAULT 0,
                thesis_invalidation_recorded BOOLEAN NOT NULL DEFAULT 0,
                position_monitor_armed BOOLEAN NOT NULL DEFAULT 0,
                review_lineage_linked BOOLEAN NOT NULL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))
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
    return eng


def _setup_lifecycle_state(
    engine,
    candidate_id: str,
    profile_id: str,
    symbol: str,
    trade_exists: bool,
    position_exists: bool,
    has_stop: bool,
    has_target: bool,
    has_invalidators: bool,
    trade_is_open: bool,
    has_lineage: bool,
):
    """Set up database rows matching the given boolean flags.

    - trade_exists: insert a trade row linked to candidate_id
    - position_exists: insert a position row for symbol/profile
    - has_stop: set stop_price on the trade (requires trade_exists)
    - has_target: set target_price on the trade (requires trade_exists)
    - has_invalidators: set invalidators on the trade (requires trade_exists)
    - trade_is_open: set trade status to 'open' (requires trade_exists)
    - has_lineage: insert a response_lineage_links row for candidate_id
    """
    with engine.begin() as conn:
        # Always insert a candidate record so symbol lookup works
        conn.execute(text("""
            INSERT INTO pm_candidates (
                candidate_id, cycle_id, profile_id, symbol, direction,
                setup_type, geometry_name, entry_price, stop_price,
                target_price, risk_reward, source_signal_id,
                signal_snapshot_json, integrity_hash, expires_at
            ) VALUES (
                :cid, 'cycle-1', :pid, :symbol, 'BUY',
                'breakout', 'analyst_geometry', 150.0, 145.0,
                160.0, 2.0, 'signal-1',
                '{}', 'hash123', '2099-01-01T00:00:00'
            )
        """), {"cid": candidate_id, "pid": profile_id, "symbol": symbol})

        # Insert trade row if requested
        if trade_exists:
            stop_price = 145.0 if has_stop else None
            target_price = 160.0 if has_target else None
            invalidators = '[{"type":"price_level"}]' if has_invalidators else None
            status = "open" if trade_is_open else "closed"

            conn.execute(text("""
                INSERT INTO trades (
                    symbol, direction, quantity, entry_price, stop_price,
                    target_price, status, profile, invalidators,
                    candidate_lineage_id
                ) VALUES (
                    :symbol, 'LONG', 100, 150.0, :stop_price,
                    :target_price, :status, :profile, :invalidators,
                    :candidate_id
                )
            """), {
                "symbol": symbol,
                "stop_price": stop_price,
                "target_price": target_price,
                "status": status,
                "profile": profile_id,
                "invalidators": invalidators,
                "candidate_id": candidate_id,
            })

        # Insert position row if requested
        if position_exists:
            conn.execute(text("""
                INSERT INTO positions (profile, symbol, side, quantity, avg_cost)
                VALUES (:profile, :symbol, 'long', 100, 150.0)
            """), {"profile": profile_id, "symbol": symbol})

        # Insert lineage link if requested
        if has_lineage:
            conn.execute(text("""
                INSERT INTO response_lineage_links (response_id, lineage_id, candidate_id)
                VALUES (:rid, :lid, :cid)
            """), {
                "rid": str(uuid.uuid4()),
                "lid": str(uuid.uuid4()),
                "cid": candidate_id,
            })


# ---------------------------------------------------------------------------
# Property 13: Lifecycle checklist correctness and incomplete event
# ---------------------------------------------------------------------------


@given(
    trade_exists=st.booleans(),
    position_exists=st.booleans(),
    has_stop=st.booleans(),
    has_target=st.booleans(),
    has_invalidators=st.booleans(),
    trade_is_open=st.booleans(),
    has_lineage=st.booleans(),
)
@settings(max_examples=200)
def test_lifecycle_checklist_correctness(
    trade_exists, position_exists, has_stop, has_target,
    has_invalidators, trade_is_open, has_lineage
):
    """Property 13: Lifecycle checklist correctness and incomplete event.

    For any candidate reaching the EXECUTED state, the lifecycle checklist
    SHALL set each boolean field to true if the corresponding row/registration
    exists and false otherwise. If any field is false, a lifecycle_incomplete
    event SHALL be written listing exactly the field names that are false.

    Strategy: Generate random combinations of which lifecycle components exist
    (7 booleans). Set up an in-memory DB with the corresponding rows present
    or absent. Call write_lifecycle_checklist() and verify each boolean matches
    expectation. Also verify: checklist.missing_components == list of field
    names where the bool is False. And checklist.complete == (all bools are True).

    **Validates: Requirements 7.1, 7.2**
    """
    # -- Setup --
    engine = _create_test_engine()
    candidate_id = str(uuid.uuid4())
    trade_id = "trade-1"
    cycle_id = "cycle-1"
    profile_id = "moderate"
    symbol = "AAPL"

    _setup_lifecycle_state(
        engine=engine,
        candidate_id=candidate_id,
        profile_id=profile_id,
        symbol=symbol,
        trade_exists=trade_exists,
        position_exists=position_exists,
        has_stop=has_stop,
        has_target=has_target,
        has_invalidators=has_invalidators,
        trade_is_open=trade_is_open,
        has_lineage=has_lineage,
    )

    # -- Act --
    checklist = write_lifecycle_checklist(
        engine, candidate_id, trade_id, cycle_id, profile_id
    )

    # -- Assert: checklist was successfully created --
    assert checklist is not None, "write_lifecycle_checklist should not return None"

    # -- Assert: each boolean field reflects database state --
    # trade_row_created: True iff a trade with this candidate_lineage_id exists
    assert checklist.trade_row_created == trade_exists, (
        f"trade_row_created should be {trade_exists}, got {checklist.trade_row_created}"
    )

    # position_row_created_or_updated: True iff a position for symbol/profile exists
    assert checklist.position_row_created_or_updated == position_exists, (
        f"position_row_created_or_updated should be {position_exists}, "
        f"got {checklist.position_row_created_or_updated}"
    )

    # stop_registered: True iff trade exists AND has non-null non-zero stop_price
    expected_stop = trade_exists and has_stop
    assert checklist.stop_registered == expected_stop, (
        f"stop_registered should be {expected_stop}, got {checklist.stop_registered}"
    )

    # target_registered: True iff trade exists AND has non-null non-zero target_price
    expected_target = trade_exists and has_target
    assert checklist.target_registered == expected_target, (
        f"target_registered should be {expected_target}, got {checklist.target_registered}"
    )

    # thesis_invalidation_recorded: True iff trade exists AND has non-empty invalidators
    expected_invalidation = trade_exists and has_invalidators
    assert checklist.thesis_invalidation_recorded == expected_invalidation, (
        f"thesis_invalidation_recorded should be {expected_invalidation}, "
        f"got {checklist.thesis_invalidation_recorded}"
    )

    # position_monitor_armed: True iff trade exists AND status == 'open'
    expected_monitor = trade_exists and trade_is_open
    assert checklist.position_monitor_armed == expected_monitor, (
        f"position_monitor_armed should be {expected_monitor}, "
        f"got {checklist.position_monitor_armed}"
    )

    # review_lineage_linked: True iff response_lineage_links has a row
    assert checklist.review_lineage_linked == has_lineage, (
        f"review_lineage_linked should be {has_lineage}, got {checklist.review_lineage_linked}"
    )

    # -- Assert: missing_components lists exactly the false field names --
    expected_missing = []
    if not trade_exists:
        expected_missing.append("trade_row_created")
    if not position_exists:
        expected_missing.append("position_row_created_or_updated")
    if not expected_stop:
        expected_missing.append("stop_registered")
    if not expected_target:
        expected_missing.append("target_registered")
    if not expected_invalidation:
        expected_missing.append("thesis_invalidation_recorded")
    if not expected_monitor:
        expected_missing.append("position_monitor_armed")
    if not has_lineage:
        expected_missing.append("review_lineage_linked")

    assert checklist.missing_components == expected_missing, (
        f"missing_components should be {expected_missing}, "
        f"got {checklist.missing_components}"
    )

    # -- Assert: complete is True iff all fields are True --
    all_present = (
        trade_exists and position_exists and has_stop and has_target
        and has_invalidators and trade_is_open and has_lineage
    )
    assert checklist.complete == all_present, (
        f"complete should be {all_present}, got {checklist.complete}"
    )

    # -- Assert: checklist was persisted to database --
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT * FROM candidate_lifecycle_checklists WHERE candidate_id = :cid"
        ), {"cid": candidate_id}).fetchone()
    assert row is not None, "Checklist should be persisted to database"
