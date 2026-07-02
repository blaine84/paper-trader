"""
Property-based bug condition exploration tests for execution-first bugfix.

These tests encode the EXPECTED behavior after the fix is applied. They are
written against the UNFIXED code and are EXPECTED TO FAIL — failure confirms
the bug exists (candidates strand in RESERVED state after execution failure).

Bug Conditions Explored:
1. execute_candidate_pipeline() does not transition RESERVED → EXECUTION_FAILED
   when execute_trade() raises an exception
2. execute_candidate_pipeline() does not transition RESERVED → EXECUTION_FAILED
   when execute_trade() returns (False, error_message)
3. finalize_cycle() does not sweep RESERVED candidates to EXECUTION_FAILED
4. build_candidate_set() does not exclude non-executable setup types
5. parse_decision_contract() produces EXTRA_FIELDS violation for PM "reason" field

**Validates: Requirements 1.1, 1.2, 1.3, 1.5, 1.6, 2.1, 2.2, 2.3, 2.5, 2.6**
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from sqlalchemy import create_engine, text

from utils.candidate_pipeline import (
    PipelineResult,
    execute_candidate_pipeline,
)
from utils.candidate_registry import (
    CandidateRecord,
    CandidateRegistry,
    CandidateState,
    _compute_integrity_hash,
)
from utils.decision_contract import (
    CandidateDecision,
    parse_decision_contract,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid candidate IDs (UUID4 format)
candidate_id_strategy = st.uuids().map(str)

# Error messages (text up to 500 chars)
error_message_strategy = st.text(
    min_size=1, max_size=500, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"))
)

# Setup types NOT in the executable set
NON_EXECUTABLE_SETUP_TYPES = [
    "sector_rotation",
    "risk_off_macro_short_fade",
    "unknown",
    "pairs_trade",
    "mean_reversion",
    "earnings_play",
]
non_executable_setup_type_strategy = st.sampled_from(NON_EXECUTABLE_SETUP_TYPES)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _create_pm_candidates_table(engine):
    """Create the pm_candidates table schema for testing."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS pm_candidates (
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
                    candidate_lineage_id TEXT
                )
                """
            )
        )


def _create_pm_candidate_events_table(engine):
    """Create the pm_candidate_events table for testing."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS pm_candidate_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT NOT NULL,
                    cycle_id TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_data TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
        )


def _setup_test_db():
    """Create an in-memory SQLite engine with all required tables."""
    engine = create_engine("sqlite:///:memory:")
    _create_pm_candidates_table(engine)
    _create_pm_candidate_events_table(engine)
    return engine


def _register_candidate(engine, candidate_id, cycle_id, profile_id, setup_type="momentum_fade"):
    """Register a candidate directly into the DB in REGISTERED state."""
    now = datetime.now(timezone.utc)
    record_dict = {
        "candidate_id": candidate_id,
        "symbol": "NVDA",
        "direction": "BUY",
        "entry_price": 100.0,
        "stop_price": 98.0,
        "target_price": 104.0,
        "setup_type": setup_type,
        "profile_id": profile_id,
        "cycle_id": cycle_id,
    }
    integrity_hash = _compute_integrity_hash(record_dict)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO pm_candidates (
                    candidate_id, cycle_id, profile_id, symbol, direction,
                    setup_type, geometry_name, entry_price, stop_price,
                    target_price, risk_reward, trigger, invalidation_basis,
                    target_basis, source_signal_id, signal_snapshot_json,
                    state, integrity_hash, created_at, expires_at
                ) VALUES (
                    :candidate_id, :cycle_id, :profile_id, :symbol, :direction,
                    :setup_type, :geometry_name, :entry_price, :stop_price,
                    :target_price, :risk_reward, :trigger, :invalidation_basis,
                    :target_basis, :source_signal_id, :signal_snapshot_json,
                    :state, :integrity_hash, :created_at, :expires_at
                )
                """
            ),
            {
                "candidate_id": candidate_id,
                "cycle_id": cycle_id,
                "profile_id": profile_id,
                "symbol": "NVDA",
                "direction": "BUY",
                "setup_type": setup_type,
                "geometry_name": "base_breakout",
                "entry_price": 100.0,
                "stop_price": 98.0,
                "target_price": 104.0,
                "risk_reward": 2.0,
                "trigger": "Price breaks above resistance",
                "invalidation_basis": "Price falls below stop",
                "target_basis": "Entry + risk x RR",
                "source_signal_id": f"sig_{candidate_id[:8]}",
                "signal_snapshot_json": json.dumps({"symbol": "NVDA", "signal": "BUY", "strength": "strong"}),
                "state": CandidateState.REGISTERED.value,
                "integrity_hash": integrity_hash,
                "created_at": now.isoformat(),
                "expires_at": (now + timedelta(hours=1)).isoformat(),
            },
        )


def _reserve_candidate(engine, candidate_id):
    """Transition a candidate to RESERVED state (simulating successful reserve step)."""
    now = datetime.now(timezone.utc).isoformat()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE pm_candidates
                SET state = 'reserved', reserved_at = :now, execution_key = :key
                WHERE candidate_id = :cid AND state = 'registered'
                """
            ),
            {"now": now, "key": f"exec_{candidate_id[:16]}", "cid": candidate_id},
        )


def _get_candidate_state(engine, candidate_id) -> str | None:
    """Query the current state of a candidate from the DB."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT state FROM pm_candidates WHERE candidate_id = :cid"),
            {"cid": candidate_id},
        ).fetchone()
        return row[0] if row else None


def _get_candidate_rejection_reason(engine, candidate_id) -> str | None:
    """Query the rejection_reason of a candidate from the DB."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT rejection_reason FROM pm_candidates WHERE candidate_id = :cid"),
            {"cid": candidate_id},
        ).fetchone()
        return row[0] if row else None


# ---------------------------------------------------------------------------
# Property 1: Bug Condition — Execution Failure Terminal Transition
# **Validates: Requirements 1.1, 1.2, 2.1, 2.2**
# ---------------------------------------------------------------------------


class TestBugConditionExceptionPath:
    """
    execute_candidate_pipeline() SHALL transition candidate from RESERVED →
    EXECUTION_FAILED when execute_trade() raises an exception.

    BUG: On unfixed code, the candidate remains in RESERVED state because
    no registry.mark_*() method is called on the exception path.

    **Validates: Requirements 1.1, 2.1**
    """

    @given(error_msg=error_message_strategy)
    @settings(max_examples=50, deadline=None)
    def test_exception_path_transitions_to_execution_failed(self, error_msg):
        """After execute_trade raises, candidate state must be execution_failed."""
        engine = _setup_test_db()
        candidate_id = str(uuid.uuid4())
        cycle_id = "cycle_test"
        profile_id = "moderate"

        _register_candidate(engine, candidate_id, cycle_id, profile_id)
        _reserve_candidate(engine, candidate_id)

        # Confirm candidate is in RESERVED state
        assert _get_candidate_state(engine, candidate_id) == "reserved"

        registry = CandidateRegistry(engine, cycle_id, profile_id)
        decision = CandidateDecision(
            candidate_id=candidate_id,
            decision="accept",
            risk_multiplier=1.0,
            rationale="test",
        )

        candidate_record = CandidateRecord(
            candidate_id=candidate_id,
            cycle_id=cycle_id,
            profile_id=profile_id,
            symbol="NVDA",
            direction="BUY",
            setup_type="momentum_fade",
            geometry_name="base_breakout",
            entry_price=100.0,
            stop_price=98.0,
            target_price=104.0,
            risk_reward=2.0,
            trigger="Price breaks above resistance",
            invalidation_basis="Price falls below stop",
            target_basis="Entry + risk x RR",
            source_signal_id=f"sig_{candidate_id[:8]}",
            signal_snapshot_json="{}",
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            integrity_hash="abc123",
        )

        sizing_mock = MagicMock(
            quantity=10, dollar_risk=20.0, position_value=1000.0,
            sizing_method="standard", applied_multiplier=1.0, rejected=False,
            rejection_reason=None,
        )

        # Patch lazy imports and internal functions
        with patch("utils.candidate_pipeline._resolve_candidate", return_value=(candidate_record, None)):
            with patch.object(registry, "reserve", return_value=(True, None)):
                with patch("utils.candidate_pipeline.calculate_position_size", return_value=sizing_mock):
                    with patch("agents.portfolio_manager._run_gate_pipeline", return_value=(True, [], 1.0, {})):
                        with patch("agents.portfolio_manager.execute_trade", side_effect=RuntimeError(error_msg)):
                            result = execute_candidate_pipeline(
                                db=engine,
                                engine=engine,
                                registry=registry,
                                decision=decision,
                                portfolio={"positions": {}, "cash": 100000},
                                profile={"name": "moderate"},
                                profile_id=profile_id,
                            )

        # Assert: The pipeline returned execution_failed outcome
        assert result.outcome == "execution_failed"

        # Assert: The candidate MUST be in execution_failed state (BUG: it's still "reserved")
        state = _get_candidate_state(engine, candidate_id)
        assert state == "execution_failed", (
            f"Expected state='execution_failed' but got state='{state}'. "
            f"Bug confirmed: candidate strands in RESERVED after exception."
        )

        # Assert: rejection_reason MUST be set
        reason = _get_candidate_rejection_reason(engine, candidate_id)
        assert reason is not None, (
            "Expected rejection_reason to be set after execution failure. "
            "Bug confirmed: no rejection reason recorded."
        )


class TestBugConditionSuccessFalsePath:
    """
    execute_candidate_pipeline() SHALL transition candidate from RESERVED →
    EXECUTION_FAILED when execute_trade() returns (False, error_message).

    BUG: On unfixed code, the candidate remains in RESERVED state because
    no registry.mark_*() method is called on the success=False path.

    **Validates: Requirements 1.2, 2.2**
    """

    @given(error_msg=error_message_strategy)
    @settings(max_examples=50, deadline=None)
    def test_success_false_path_transitions_to_execution_failed(self, error_msg):
        """After execute_trade returns (False, msg), candidate state must be execution_failed."""
        engine = _setup_test_db()
        candidate_id = str(uuid.uuid4())
        cycle_id = "cycle_test"
        profile_id = "moderate"

        _register_candidate(engine, candidate_id, cycle_id, profile_id)
        _reserve_candidate(engine, candidate_id)

        assert _get_candidate_state(engine, candidate_id) == "reserved"

        registry = CandidateRegistry(engine, cycle_id, profile_id)
        decision = CandidateDecision(
            candidate_id=candidate_id,
            decision="accept",
            risk_multiplier=1.0,
            rationale="test",
        )

        candidate_record = CandidateRecord(
            candidate_id=candidate_id,
            cycle_id=cycle_id,
            profile_id=profile_id,
            symbol="NVDA",
            direction="BUY",
            setup_type="momentum_fade",
            geometry_name="base_breakout",
            entry_price=100.0,
            stop_price=98.0,
            target_price=104.0,
            risk_reward=2.0,
            trigger="Price breaks above resistance",
            invalidation_basis="Price falls below stop",
            target_basis="Entry + risk x RR",
            source_signal_id=f"sig_{candidate_id[:8]}",
            signal_snapshot_json="{}",
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            integrity_hash="abc123",
        )

        sizing_mock = MagicMock(
            quantity=10, dollar_risk=20.0, position_value=1000.0,
            sizing_method="standard", applied_multiplier=1.0, rejected=False,
            rejection_reason=None,
        )

        # Mock execute_trade to return (False, error_message)
        with patch("utils.candidate_pipeline._resolve_candidate", return_value=(candidate_record, None)):
            with patch.object(registry, "reserve", return_value=(True, None)):
                with patch("utils.candidate_pipeline.calculate_position_size", return_value=sizing_mock):
                    with patch("agents.portfolio_manager._run_gate_pipeline", return_value=(True, [], 1.0, {})):
                        with patch("agents.portfolio_manager.execute_trade", return_value=(False, error_msg)):
                            result = execute_candidate_pipeline(
                                db=engine,
                                engine=engine,
                                registry=registry,
                                decision=decision,
                                portfolio={"positions": {}, "cash": 100000},
                                profile={"name": "moderate"},
                                profile_id=profile_id,
                            )

        # Assert: outcome is execution_failed
        assert result.outcome == "execution_failed"

        # Assert: candidate MUST be in execution_failed state (BUG: it's still "reserved")
        state = _get_candidate_state(engine, candidate_id)
        assert state == "execution_failed", (
            f"Expected state='execution_failed' but got state='{state}'. "
            f"Bug confirmed: candidate strands in RESERVED after success=False."
        )

        # Assert: rejection_reason MUST be set
        reason = _get_candidate_rejection_reason(engine, candidate_id)
        assert reason is not None, (
            "Expected rejection_reason to be set after execution failure. "
            "Bug confirmed: no rejection reason recorded."
        )


# ---------------------------------------------------------------------------
# Property 3: Bug Condition — finalize_cycle RESERVED Sweep
# **Validates: Requirements 1.5, 2.5**
# ---------------------------------------------------------------------------


class TestBugConditionFinalizeCycleSweep:
    """
    finalize_cycle() SHALL sweep any RESERVED candidates to EXECUTION_FAILED
    with reason "cycle_finalized_while_reserved".

    BUG: On unfixed code, finalize_cycle() only sweeps REGISTERED candidates.
    RESERVED candidates are invisible to end-of-cycle cleanup.

    **Validates: Requirements 1.5, 2.5**
    """

    @given(candidate_id=candidate_id_strategy)
    @settings(max_examples=50, deadline=None)
    def test_finalize_cycle_sweeps_reserved_to_execution_failed(self, candidate_id):
        """finalize_cycle() must transition RESERVED → EXECUTION_FAILED."""
        engine = _setup_test_db()
        cycle_id = "cycle_finalize_test"
        profile_id = "moderate"

        _register_candidate(engine, candidate_id, cycle_id, profile_id)
        _reserve_candidate(engine, candidate_id)

        # Confirm candidate is RESERVED
        assert _get_candidate_state(engine, candidate_id) == "reserved"

        registry = CandidateRegistry(engine, cycle_id, profile_id)
        terminal_assignments = registry.finalize_cycle()

        # Assert: RESERVED candidate must appear in terminal_assignments
        assert candidate_id in terminal_assignments, (
            f"Expected candidate {candidate_id} in terminal_assignments dict. "
            f"Bug confirmed: finalize_cycle() does not sweep RESERVED candidates."
        )

        # Assert: terminal state must be EXECUTION_FAILED
        # Note: On unfixed code, CandidateState.EXECUTION_FAILED doesn't exist,
        # so this assertion may error differently
        state = _get_candidate_state(engine, candidate_id)
        assert state == "execution_failed", (
            f"Expected state='execution_failed' but got state='{state}'. "
            f"Bug confirmed: RESERVED candidate not swept by finalize_cycle()."
        )

        # Assert: rejection_reason must be set
        reason = _get_candidate_rejection_reason(engine, candidate_id)
        assert reason == "cycle_finalized_while_reserved", (
            f"Expected rejection_reason='cycle_finalized_while_reserved' but got '{reason}'."
        )


# ---------------------------------------------------------------------------
# Property 4: Bug Condition — Setup Type Allowlist Filtering
# **Validates: Requirements 1.3, 2.3**
# ---------------------------------------------------------------------------


class TestBugConditionSetupTypeFiltering:
    """
    build_candidate_set() SHALL exclude signals with setup_type not in
    CANDIDATE_EXECUTABLE_SETUP_TYPES.

    BUG: On unfixed code, no setup type allowlist exists. All setup types
    (including sector_rotation, risk_off_macro_short_fade) are registered
    as candidates.

    **Validates: Requirements 1.3, 2.3**
    """

    @given(setup_type=non_executable_setup_type_strategy)
    @settings(max_examples=30, deadline=None)
    def test_non_executable_setup_types_excluded(self, setup_type):
        """Signals with non-executable setup types must not be registered."""
        engine = _setup_test_db()
        cycle_id = "cycle_filter_test"
        profile_id = "moderate"

        # Create a signal with the non-executable setup type
        signals = {
            "AAPL": {
                "symbol": "AAPL",
                "signal": "BUY",
                "strength": "strong",
                "setup_type": setup_type,
                "current_price": 150.0,
            }
        }

        profile = {"min_signal_strength": "moderate"}
        portfolio = {"positions": {}}

        # Mock the geometry scaffold to return a valid candidate
        def fake_scaffold(signal, profile_id=None, profile_context=None):
            return {
                "symbol": signal["symbol"],
                "direction": "LONG",
                "status": "ok",
                "candidates": [
                    {
                        "name": "base_breakout",
                        "entry_price": 150.0,
                        "stop_loss": 148.0,
                        "target": 154.0,
                        "risk_reward": 2.0,
                        "trigger": "Price breaks above",
                        "invalidation_basis": "Falls below stop",
                        "target_basis": "Entry + RR * risk",
                    }
                ],
            }

        with patch("utils.candidate_builder.build_entry_geometry_scaffold", fake_scaffold):
            from utils.candidate_builder import build_candidate_set
            registry = build_candidate_set(
                engine,
                signals,
                profile_id,
                profile,
                portfolio,
                cycle_id,
            )

        # Assert: registry must be empty (non-executable type excluded)
        # BUG: On unfixed code, the candidate IS registered (no filter exists)
        assert registry.is_empty, (
            f"Expected registry to be empty for setup_type='{setup_type}' "
            f"(not in CANDIDATE_EXECUTABLE_SETUP_TYPES). "
            f"Bug confirmed: non-executable setup types enter PM selection."
        )


# ---------------------------------------------------------------------------
# Property 5: Bug Condition — Reason Field Normalization
# **Validates: Requirements 1.6, 2.6**
# ---------------------------------------------------------------------------


class TestBugConditionReasonFieldNormalization:
    """
    parse_decision_contract() SHALL NOT produce an EXTRA_FIELDS violation when
    PM includes a "reason" field. Instead, "reason" should be normalized to
    "rationale".

    BUG: On unfixed code, "reason" is not in _VALID_DECISION_FIELDS, so it
    triggers an EXTRA_FIELDS violation and the PM's reasoning is discarded.

    **Validates: Requirements 1.6, 2.6**
    """

    @given(
        reason_text=st.text(min_size=1, max_size=200, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"))),
    )
    @settings(max_examples=50, deadline=None)
    def test_reason_field_does_not_trigger_extra_fields_violation(self, reason_text):
        """PM 'reason' field must not produce EXTRA_FIELDS violation."""
        candidate_id = str(uuid.uuid4())
        valid_ids = {candidate_id}
        metadata = {
            candidate_id: {
                "symbol": "NVDA",
                "source_signal_id": "sig_test",
                "profile_id": "moderate",
            }
        }

        raw_response = {
            "decisions": [
                {
                    "candidate_id": candidate_id,
                    "decision": "accept",
                    "reason": reason_text,
                }
            ]
        }

        result = parse_decision_contract(raw_response, valid_ids, metadata)

        # Check for EXTRA_FIELDS violations mentioning "reason"
        extra_fields_violations = [
            v for v in result.violations
            if v.get("type") == "EXTRA_FIELDS"
            and "reason" in v.get("fields", [])
        ]

        # Assert: no EXTRA_FIELDS violation for "reason"
        # BUG: On unfixed code, "reason" triggers EXTRA_FIELDS violation
        assert len(extra_fields_violations) == 0, (
            f"Expected no EXTRA_FIELDS violation for 'reason' field, but got: "
            f"{extra_fields_violations}. "
            f"Bug confirmed: PM 'reason' field triggers spurious EXTRA_FIELDS violation."
        )

    @given(
        reason_text=st.text(min_size=1, max_size=200, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"))),
    )
    @settings(max_examples=50, deadline=None)
    def test_reason_field_normalized_to_rationale(self, reason_text):
        """PM 'reason' field must be copied to 'rationale' when rationale is absent."""
        candidate_id = str(uuid.uuid4())
        valid_ids = {candidate_id}
        metadata = {
            candidate_id: {
                "symbol": "NVDA",
                "source_signal_id": "sig_test",
                "profile_id": "moderate",
            }
        }

        raw_response = {
            "decisions": [
                {
                    "candidate_id": candidate_id,
                    "decision": "accept",
                    "reason": reason_text,
                }
            ]
        }

        result = parse_decision_contract(raw_response, valid_ids, metadata)

        # Assert: the accepted decision should have the reason as rationale
        # BUG: On unfixed code, the candidate may still be accepted but
        # rationale remains empty (reason is stripped as extra field)
        assert len(result.accepted) == 1, (
            f"Expected 1 accepted decision but got {len(result.accepted)}"
        )
        assert result.accepted[0].rationale == reason_text, (
            f"Expected rationale='{reason_text}' but got '{result.accepted[0].rationale}'. "
            f"Bug confirmed: PM 'reason' field is not normalized to 'rationale'."
        )
