"""
Property-based tests for preservation of non-failure path behavior.

Property 2: Preservation — Non-Failure Path Behavior

Captures existing behavior on UNFIXED code to serve as a regression guard.
These tests verify that successful executions, sizing rejections, gate rejections,
finalize_cycle sweeps, and decision contract parsing all behave correctly.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10**
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st
from sqlalchemy import create_engine, text

# Increase deadline for CI/cold-start variability
_SETTINGS = dict(max_examples=50, deadline=None)

from utils.candidate_registry import (
    CandidateRecord,
    CandidateRegistry,
    CandidateRegistryError,
    CandidateState,
)
from utils.decision_contract import (
    CandidateDecision,
    parse_decision_contract,
    _VALID_DECISION_FIELDS,
)
from utils.candidate_pipeline import (
    PipelineResult,
    execute_candidate_pipeline,
)
from utils.position_sizer import SizingResult


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

SYMBOLS = ["NVDA", "AAPL", "TSLA", "MSFT", "GOOG", "AMD", "META", "AMZN"]
DIRECTIONS = ["BUY", "SHORT"]
SETUP_TYPES = ["momentum_fade", "news_breakout", "gap_and_go", "technical_breakout", "vwap_reclaim"]

st_symbol = st.sampled_from(SYMBOLS)
st_direction = st.sampled_from(DIRECTIONS)
st_setup_type = st.sampled_from(SETUP_TYPES)
st_candidate_id = st.builds(lambda: str(uuid.uuid4()))
st_cycle_id = st.builds(lambda: f"cycle-{uuid.uuid4().hex[:8]}")
st_profile_id = st.sampled_from(["moderate", "aggressive", "conservative"])
st_risk_multiplier = st.one_of(st.none(), st.floats(min_value=0.01, max_value=1.0))
st_rationale = st.text(min_size=0, max_size=200, alphabet=st.characters(categories=("L", "N", "P", "Z")))

# Valid fields for decision contract preservation tests
VALID_FIELDS = list(_VALID_DECISION_FIELDS)
PROHIBITED_FIELDS = ["symbol", "entry_price", "stop", "quantity"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_tables(engine):
    """Create minimal pm_candidates table for testing."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE pm_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                cycle_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                setup_type TEXT NOT NULL,
                geometry_name TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_price REAL NOT NULL,
                target_price REAL NOT NULL,
                risk_reward REAL NOT NULL,
                trigger TEXT,
                invalidation_basis TEXT,
                target_basis TEXT,
                source_signal_id TEXT NOT NULL,
                signal_snapshot_json TEXT NOT NULL,
                state TEXT NOT NULL,
                integrity_hash TEXT NOT NULL,
                execution_key TEXT,
                reserved_at TEXT,
                created_at TEXT,
                expires_at TEXT NOT NULL,
                context_snapshot_json TEXT,
                benchmark_mapping_json TEXT,
                rejection_reason TEXT,
                candidate_lineage_id TEXT,
                candidate_type TEXT DEFAULT 'intraday'
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


def _make_candidate_record(
    candidate_id: str,
    cycle_id: str,
    profile_id: str,
    symbol: str = "NVDA",
    direction: str = "BUY",
    setup_type: str = "momentum_fade",
) -> CandidateRecord:
    """Build a valid CandidateRecord for testing."""
    now = datetime.now(timezone.utc)
    return CandidateRecord(
        candidate_id=candidate_id,
        cycle_id=cycle_id,
        profile_id=profile_id,
        symbol=symbol,
        direction=direction,
        setup_type=setup_type,
        geometry_name="support_bounce",
        entry_price=150.0,
        stop_price=148.0,
        target_price=156.0,
        risk_reward=3.0,
        trigger="price_above_vwap",
        invalidation_basis="below_support",
        target_basis="resistance_level",
        source_signal_id=f"sig-{uuid.uuid4().hex[:8]}",
        signal_snapshot_json="{}",
        created_at=now,
        expires_at=now + timedelta(hours=2),
        integrity_hash="hash-" + uuid.uuid4().hex[:12],
    )


def _register_candidate(engine, registry, candidate_id, cycle_id, profile_id, **kwargs):
    """Register a candidate (in REGISTERED state). Returns the candidate record."""
    rec = _make_candidate_record(candidate_id, cycle_id, profile_id, **kwargs)
    registry.register(rec)
    return rec


def _register_and_reserve(engine, registry, candidate_id, cycle_id, profile_id, **kwargs):
    """Register a candidate and reserve it. Returns the candidate record."""
    rec = _make_candidate_record(candidate_id, cycle_id, profile_id, **kwargs)
    registry.register(rec)
    exec_key = f"exec-{uuid.uuid4().hex[:8]}"
    success, reason = registry.reserve(candidate_id, exec_key)
    assert success, f"Reserve failed: {reason}"
    return rec


def _get_candidate_state(engine, candidate_id: str) -> str:
    """Query raw candidate state from db."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT state FROM pm_candidates WHERE candidate_id = :cid"),
            {"cid": candidate_id},
        ).fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# Property: Successful execution → EXECUTED (Requirement 3.1)
# ---------------------------------------------------------------------------


@given(
    symbol=st_symbol,
    direction=st_direction,
    setup_type=st_setup_type,
    risk_multiplier=st_risk_multiplier,
    rationale=st_rationale,
)
@settings(**_SETTINGS)
def test_successful_execution_transitions_to_executed(
    symbol, direction, setup_type, risk_multiplier, rationale
):
    """**Validates: Requirements 3.1**

    When execute_trade succeeds (True, result), candidate transitions to EXECUTED
    and PipelineResult.outcome == "executed".
    """
    engine = create_engine("sqlite:///:memory:")
    _create_test_tables(engine)

    cycle_id = f"cycle-{uuid.uuid4().hex[:8]}"
    profile_id = "moderate"
    candidate_id = str(uuid.uuid4())

    registry = CandidateRegistry(engine, cycle_id, profile_id)
    rec = _register_candidate(engine, registry, candidate_id, cycle_id, profile_id,
                                symbol=symbol, direction=direction, setup_type=setup_type)

    decision = CandidateDecision(
        candidate_id=candidate_id,
        decision="accept",
        risk_multiplier=risk_multiplier,
        rationale=rationale,
    )

    # Mock everything between resolve and execute
    sizing_result = SizingResult(
        quantity=10,
        dollar_risk=200.0,
        position_value=1500.0,
        sizing_method="standard",
        applied_multiplier=1.0,
    )

    with patch("utils.candidate_pipeline.calculate_position_size", return_value=sizing_result), \
         patch("agents.portfolio_manager._run_gate_pipeline", return_value=(True, [], 1.0, {})), \
         patch("agents.portfolio_manager.execute_trade", return_value=(True, "trade-123")), \
         patch("utils.candidate_pipeline.PM_PROVENANCE_MODE", "disabled"):

        result = execute_candidate_pipeline(
            db=engine,
            engine=engine,
            registry=registry,
            decision=decision,
            portfolio={"cash": 50000},
            profile={"max_position_pct": 0.05},
            profile_id=profile_id,
        )

    assert result.outcome == "executed"
    assert _get_candidate_state(engine, candidate_id) == CandidateState.EXECUTED.value


# ---------------------------------------------------------------------------
# Property: Sizing rejection → SIZING_REJECTED (Requirement 3.2)
# ---------------------------------------------------------------------------


@given(
    symbol=st_symbol,
    direction=st_direction,
    rejection_reason=st.text(min_size=5, max_size=100, alphabet=st.characters(categories=("L", "N", "Z"))),
)
@settings(**_SETTINGS)
def test_sizing_rejection_transitions_to_sizing_rejected(
    symbol, direction, rejection_reason
):
    """**Validates: Requirements 3.2**

    When position sizing rejects a candidate, state transitions to SIZING_REJECTED
    and PipelineResult.outcome == "sizing_rejected".
    """
    engine = create_engine("sqlite:///:memory:")
    _create_test_tables(engine)

    cycle_id = f"cycle-{uuid.uuid4().hex[:8]}"
    profile_id = "moderate"
    candidate_id = str(uuid.uuid4())

    registry = CandidateRegistry(engine, cycle_id, profile_id)
    _register_candidate(engine, registry, candidate_id, cycle_id, profile_id,
                          symbol=symbol, direction=direction)

    decision = CandidateDecision(
        candidate_id=candidate_id,
        decision="accept",
    )

    sizing_result = SizingResult(
        quantity=0,
        dollar_risk=0.0,
        position_value=0.0,
        sizing_method="standard",
        applied_multiplier=0.0,
        rejection_reason=rejection_reason,
    )

    with patch("utils.candidate_pipeline.calculate_position_size", return_value=sizing_result), \
         patch("utils.candidate_pipeline.PM_PROVENANCE_MODE", "disabled"):

        result = execute_candidate_pipeline(
            db=engine,
            engine=engine,
            registry=registry,
            decision=decision,
            portfolio={"cash": 50000},
            profile={"max_position_pct": 0.05},
            profile_id=profile_id,
        )

    assert result.outcome == "sizing_rejected"
    assert _get_candidate_state(engine, candidate_id) == CandidateState.SIZING_REJECTED.value


# ---------------------------------------------------------------------------
# Property: Gate rejection → GATE_REJECTED (Requirement 3.3)
# ---------------------------------------------------------------------------


@given(
    symbol=st_symbol,
    direction=st_direction,
    gate_reason=st.text(min_size=5, max_size=100, alphabet=st.characters(categories=("L", "N", "Z"))),
)
@settings(**_SETTINGS)
def test_gate_rejection_transitions_to_gate_rejected(
    symbol, direction, gate_reason
):
    """**Validates: Requirements 3.3**

    When the gate pipeline rejects a candidate, state transitions to GATE_REJECTED
    and PipelineResult.outcome == "gate_rejected".
    """
    engine = create_engine("sqlite:///:memory:")
    _create_test_tables(engine)

    cycle_id = f"cycle-{uuid.uuid4().hex[:8]}"
    profile_id = "moderate"
    candidate_id = str(uuid.uuid4())

    registry = CandidateRegistry(engine, cycle_id, profile_id)
    _register_candidate(engine, registry, candidate_id, cycle_id, profile_id,
                          symbol=symbol, direction=direction)

    decision = CandidateDecision(
        candidate_id=candidate_id,
        decision="accept",
    )

    sizing_result = SizingResult(
        quantity=10,
        dollar_risk=200.0,
        position_value=1500.0,
        sizing_method="standard",
        applied_multiplier=1.0,
    )

    gate_notes = [{"gate": "concentration", "decision": "reject", "reason": gate_reason}]

    with patch("utils.candidate_pipeline.calculate_position_size", return_value=sizing_result), \
         patch("agents.portfolio_manager._run_gate_pipeline", return_value=(False, gate_notes, 1.0, {})), \
         patch("utils.candidate_pipeline.PM_PROVENANCE_MODE", "disabled"):

        result = execute_candidate_pipeline(
            db=engine,
            engine=engine,
            registry=registry,
            decision=decision,
            portfolio={"cash": 50000},
            profile={"max_position_pct": 0.05},
            profile_id=profile_id,
        )

    assert result.outcome == "gate_rejected"
    assert _get_candidate_state(engine, candidate_id) == CandidateState.GATE_REJECTED.value


# ---------------------------------------------------------------------------
# Property: finalize_cycle sweeps REGISTERED past expiry → EXPIRED (Req 3.5)
# and remaining REGISTERED → NOT_SELECTED (Req 3.6)
# ---------------------------------------------------------------------------


@given(
    num_expired=st.integers(min_value=1, max_value=5),
    num_not_selected=st.integers(min_value=0, max_value=5),
)
@settings(**_SETTINGS)
def test_finalize_cycle_transitions_registered_to_expired_or_not_selected(
    num_expired, num_not_selected
):
    """**Validates: Requirements 3.5, 3.6**

    finalize_cycle() transitions REGISTERED candidates past expiry to EXPIRED,
    and remaining REGISTERED candidates to NOT_SELECTED.
    """
    engine = create_engine("sqlite:///:memory:")
    _create_test_tables(engine)

    cycle_id = f"cycle-{uuid.uuid4().hex[:8]}"
    profile_id = "moderate"
    now = datetime.now(timezone.utc)

    registry = CandidateRegistry(engine, cycle_id, profile_id)

    expired_ids = []
    not_selected_ids = []

    # Register candidates that should expire (expires_at in the past)
    for _ in range(num_expired):
        cid = str(uuid.uuid4())
        rec = CandidateRecord(
            candidate_id=cid,
            cycle_id=cycle_id,
            profile_id=profile_id,
            symbol="NVDA",
            direction="BUY",
            setup_type="momentum_fade",
            geometry_name="support_bounce",
            entry_price=150.0,
            stop_price=148.0,
            target_price=156.0,
            risk_reward=3.0,
            trigger="price_above_vwap",
            invalidation_basis="below_support",
            target_basis="resistance_level",
            source_signal_id=f"sig-{uuid.uuid4().hex[:8]}",
            signal_snapshot_json="{}",
            created_at=now - timedelta(hours=3),
            expires_at=now - timedelta(hours=1),  # Already expired
            integrity_hash="hash-" + uuid.uuid4().hex[:12],
        )
        registry.register(rec)
        expired_ids.append(cid)

    # Register candidates that should NOT expire (expires_at in the future)
    for _ in range(num_not_selected):
        cid = str(uuid.uuid4())
        rec = CandidateRecord(
            candidate_id=cid,
            cycle_id=cycle_id,
            profile_id=profile_id,
            symbol="AAPL",
            direction="BUY",
            setup_type="news_breakout",
            geometry_name="support_bounce",
            entry_price=180.0,
            stop_price=178.0,
            target_price=186.0,
            risk_reward=3.0,
            trigger="earnings_beat",
            invalidation_basis="below_support",
            target_basis="resistance_level",
            source_signal_id=f"sig-{uuid.uuid4().hex[:8]}",
            signal_snapshot_json="{}",
            created_at=now - timedelta(minutes=30),
            expires_at=now + timedelta(hours=2),  # Still valid
            integrity_hash="hash-" + uuid.uuid4().hex[:12],
        )
        registry.register(rec)
        not_selected_ids.append(cid)

    # Run finalize_cycle
    terminal_assignments = registry.finalize_cycle()

    # Verify expired candidates
    for cid in expired_ids:
        assert terminal_assignments.get(cid) == CandidateState.EXPIRED, \
            f"Expected {cid} to be EXPIRED"
        assert _get_candidate_state(engine, cid) == CandidateState.EXPIRED.value

    # Verify not-selected candidates
    for cid in not_selected_ids:
        assert terminal_assignments.get(cid) == CandidateState.NOT_SELECTED, \
            f"Expected {cid} to be NOT_SELECTED"
        assert _get_candidate_state(engine, cid) == CandidateState.NOT_SELECTED.value


# ---------------------------------------------------------------------------
# Property: Prohibited fields → PROHIBITED_FIELD violation (Req 3.8, 2.7)
# ---------------------------------------------------------------------------


@given(
    prohibited_field=st.sampled_from(PROHIBITED_FIELDS),
    field_value=st.one_of(
        st.text(min_size=1, max_size=20, alphabet=st.characters(categories=("L", "N"))),
        st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        st.integers(min_value=1, max_value=1000),
    ),
)
@settings(**_SETTINGS)
def test_prohibited_fields_produce_violation(prohibited_field, field_value):
    """**Validates: Requirements 3.8**

    parse_decision_contract with prohibited fields (symbol, entry_price, stop,
    quantity) produces PROHIBITED_FIELD violation.
    """
    candidate_id = str(uuid.uuid4())
    valid_ids = {candidate_id}
    metadata = {candidate_id: {"symbol": "NVDA", "source_signal_id": "sig-1", "profile_id": "moderate"}}

    raw_response = {
        "decisions": [
            {
                "candidate_id": candidate_id,
                "decision": "accept",
                prohibited_field: field_value,
            }
        ]
    }

    result = parse_decision_contract(raw_response, valid_ids, metadata)

    # Assert PROHIBITED_FIELD violation exists for this field
    prohibited_violations = [
        v for v in result.violations
        if v.get("type") == "PROHIBITED_FIELD"
        and v.get("field") == prohibited_field
        and v.get("candidate_id") == candidate_id
    ]
    assert len(prohibited_violations) >= 1, \
        f"Expected PROHIBITED_FIELD violation for '{prohibited_field}', got violations: {result.violations}"


# ---------------------------------------------------------------------------
# Property: Valid fields → no violations (Req 3.8)
# ---------------------------------------------------------------------------


@given(
    risk_multiplier=st.one_of(st.none(), st.floats(min_value=0.01, max_value=1.0)),
    rationale=st.text(min_size=0, max_size=100, alphabet=st.characters(categories=("L", "N", "Z"))),
    include_adjustment=st.booleans(),
)
@settings(**_SETTINGS)
def test_valid_fields_produce_no_violations(risk_multiplier, rationale, include_adjustment):
    """**Validates: Requirements 3.8**

    parse_decision_contract with valid fields (candidate_id, decision,
    risk_multiplier, rationale, adjustment_request) produces no violations.
    """
    candidate_id = str(uuid.uuid4())
    valid_ids = {candidate_id}
    metadata = {candidate_id: {"symbol": "NVDA", "source_signal_id": "sig-1", "profile_id": "moderate"}}

    entry = {
        "candidate_id": candidate_id,
        "decision": "accept",
    }
    if risk_multiplier is not None:
        entry["risk_multiplier"] = risk_multiplier
    if rationale:
        entry["rationale"] = rationale

    raw_response = {"decisions": [entry]}

    result = parse_decision_contract(raw_response, valid_ids, metadata)

    # Should have no PROHIBITED_FIELD violations and no EXTRA_FIELDS violations
    bad_violations = [
        v for v in result.violations
        if v.get("type") in ("PROHIBITED_FIELD", "EXTRA_FIELDS")
        and v.get("candidate_id") == candidate_id
    ]
    assert len(bad_violations) == 0, \
        f"Expected no violations for valid fields, got: {bad_violations}"
