"""Property-based tests for rejection reason code normalization (Properties 1, 2, 3).

Tests that:
- Property 1: For any PM rejection response containing a rejection_reason_code value,
  the parsed CandidateDecision SHALL have a rejection_reason_code that is a member of
  VALID_REJECTION_REASON_CODES. If the raw code is not in the set or is an empty string,
  it SHALL be normalized to "other" and an INVALID_REJECTION_CODE violation SHALL be appended.

- Property 2: For any PM rejection response where rejection_reason_code is missing or null,
  and for any value of PM_REJECTION_CODE_MODE: if mode is "enforcing", a MISSING_REJECTION_CODE
  violation SHALL be appended; if mode is "warn" or any unrecognized value, no violation SHALL
  be appended. In all cases, the code SHALL be normalized to "other".

- Property 3: For any PM rejection with a rationale string of arbitrary length, the persisted
  event_data JSON in pm_candidate_events SHALL contain the rationale truncated to at most
  2000 characters, and the rejection_reason_code column in pm_candidates SHALL contain the
  bounded reason code.

Feature: candidate-blocker-mitigation
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from hypothesis import given, settings, strategies as st, assume
from sqlalchemy import create_engine, text

from utils.decision_contract import (
    VALID_REJECTION_REASON_CODES,
    parse_decision_contract,
)
from utils.candidate_registry import CandidateState, _compute_integrity_hash


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Generate a valid candidate_id (fixed-format UUID-like string)
candidate_id_st = st.uuids().map(str)

# Strategy for arbitrary rejection_reason_code values (including valid, empty, random)
rejection_code_st = st.one_of(
    st.sampled_from(sorted(VALID_REJECTION_REASON_CODES)),  # valid codes
    st.just(""),  # empty string
    st.text(min_size=1, max_size=50),  # arbitrary non-empty strings
)

# Strategy for feature flag mode values (including valid and arbitrary)
mode_st = st.one_of(
    st.just("enforcing"),
    st.just("warn"),
    st.text(min_size=0, max_size=30),  # arbitrary values including empty string
)


# ---------------------------------------------------------------------------
# Property 1: Rejection reason code normalization
# Validates: Requirements 1.1, 1.2
#
# For any PM rejection response containing a rejection_reason_code value,
# the parsed CandidateDecision SHALL have a rejection_reason_code that is a
# member of VALID_REJECTION_REASON_CODES. If the raw code is not in the set
# or is an empty string, it SHALL be normalized to "other" and an
# INVALID_REJECTION_CODE violation SHALL be appended.
# ---------------------------------------------------------------------------


@given(
    candidate_id=candidate_id_st,
    raw_code=rejection_code_st,
)
@settings(max_examples=200)
def test_rejection_reason_code_normalization(candidate_id: str, raw_code: str):
    """Property 1: Rejection reason code normalization.

    For any PM rejection with a rejection_reason_code present, the parsed
    CandidateDecision always has a code in VALID_REJECTION_REASON_CODES.
    Invalid/empty codes are normalized to "other" with an INVALID_REJECTION_CODE violation.

    **Validates: Requirements 1.1, 1.2**
    """
    raw_response = {
        "decisions": [
            {
                "candidate_id": candidate_id,
                "decision": "reject",
                "rationale": "test rationale",
                "rejection_reason_code": raw_code,
            }
        ]
    }

    valid_ids = {candidate_id}
    metadata = {candidate_id: {"symbol": "TEST", "source_signal_id": "sig1", "profile_id": "prof1"}}

    result = parse_decision_contract(raw_response, valid_ids, metadata)

    # The candidate should be in the rejected list (not dropped by other validation)
    assert len(result.rejected) == 1, (
        f"Expected 1 rejected decision, got {len(result.rejected)}. Violations: {result.violations}"
    )

    decision = result.rejected[0]

    # INVARIANT: The rejection_reason_code is always a member of VALID_REJECTION_REASON_CODES
    assert decision.rejection_reason_code in VALID_REJECTION_REASON_CODES, (
        f"rejection_reason_code {decision.rejection_reason_code!r} "
        f"not in VALID_REJECTION_REASON_CODES"
    )

    # If the raw code was valid (in the set and non-empty), it should be preserved as-is
    if raw_code in VALID_REJECTION_REASON_CODES and raw_code != "":
        assert decision.rejection_reason_code == raw_code
        # No INVALID_REJECTION_CODE violation should be present
        invalid_violations = [
            v for v in result.violations if v["type"] == "INVALID_REJECTION_CODE"
        ]
        assert len(invalid_violations) == 0, (
            f"Unexpected INVALID_REJECTION_CODE violation for valid code {raw_code!r}"
        )
    else:
        # Invalid or empty code: normalized to "other"
        assert decision.rejection_reason_code == "other"
        # An INVALID_REJECTION_CODE violation should be appended
        invalid_violations = [
            v for v in result.violations if v["type"] == "INVALID_REJECTION_CODE"
        ]
        assert len(invalid_violations) == 1, (
            f"Expected 1 INVALID_REJECTION_CODE violation for invalid code {raw_code!r}, "
            f"got {len(invalid_violations)}"
        )
        assert invalid_violations[0]["candidate_id"] == candidate_id
        assert invalid_violations[0]["raw_code"] == raw_code


# ---------------------------------------------------------------------------
# Property 2: Feature flag mode routing for missing rejection codes
# Validates: Requirements 1.4, 1.5, 1.6
#
# For any PM rejection response where rejection_reason_code is missing or null,
# and for any value of PM_REJECTION_CODE_MODE: if mode is "enforcing", a
# MISSING_REJECTION_CODE violation SHALL be appended; if mode is "warn" or any
# unrecognized value, no violation SHALL be appended. In all cases, the code
# SHALL be normalized to "other".
# ---------------------------------------------------------------------------


@given(
    candidate_id=candidate_id_st,
    mode_value=mode_st,
    use_null=st.booleans(),  # True = explicit None in payload, False = key absent
)
@settings(max_examples=200)
def test_feature_flag_mode_routing_for_missing_rejection_codes(
    candidate_id: str, mode_value: str, use_null: bool
):
    """Property 2: Feature flag mode routing for missing rejection codes.

    When rejection_reason_code is missing or null:
    - If PM_REJECTION_CODE_MODE == "enforcing": MISSING_REJECTION_CODE violation appended
    - If PM_REJECTION_CODE_MODE == "warn" or any other value: no violation appended
    - In all cases: code normalized to "other"

    **Validates: Requirements 1.4, 1.5, 1.6**
    """
    # Build the decision entry — either with null code or without the key
    decision_entry = {
        "candidate_id": candidate_id,
        "decision": "reject",
        "rationale": "test rationale",
    }
    if use_null:
        decision_entry["rejection_reason_code"] = None

    raw_response = {"decisions": [decision_entry]}
    valid_ids = {candidate_id}
    metadata = {candidate_id: {"symbol": "TEST", "source_signal_id": "sig1", "profile_id": "prof1"}}

    # Patch the feature flag at gate_config where it's lazily imported inside the function
    with patch("utils.gate_config.PM_REJECTION_CODE_MODE", mode_value):
        result = parse_decision_contract(raw_response, valid_ids, metadata)

    # The candidate should be in the rejected list
    assert len(result.rejected) == 1, (
        f"Expected 1 rejected decision, got {len(result.rejected)}. Violations: {result.violations}"
    )

    decision = result.rejected[0]

    # INVARIANT: rejection_reason_code always normalized to "other" when missing
    assert decision.rejection_reason_code == "other", (
        f"Expected 'other', got {decision.rejection_reason_code!r}"
    )

    # Check violation based on mode
    missing_violations = [
        v for v in result.violations if v["type"] == "MISSING_REJECTION_CODE"
    ]

    if mode_value == "enforcing":
        # Enforcing mode: MISSING_REJECTION_CODE violation SHALL be appended
        assert len(missing_violations) == 1, (
            f"Expected 1 MISSING_REJECTION_CODE violation in enforcing mode, "
            f"got {len(missing_violations)}. All violations: {result.violations}"
        )
        assert missing_violations[0]["candidate_id"] == candidate_id
    else:
        # Warn mode or any unrecognized value: no MISSING_REJECTION_CODE violation
        assert len(missing_violations) == 0, (
            f"Expected no MISSING_REJECTION_CODE violation in mode {mode_value!r}, "
            f"got {len(missing_violations)}. All violations: {result.violations}"
        )



# ---------------------------------------------------------------------------
# Property 3: Rejection rationale persistence with truncation
# Validates: Requirements 1.3
#
# For any PM rejection with a rationale string of arbitrary length, the
# persisted event_data JSON in pm_candidate_events SHALL contain the rationale
# truncated to at most 2000 characters, and the rejection_reason_code column
# in pm_candidates SHALL contain the bounded reason code.
# ---------------------------------------------------------------------------


def _create_persistence_tables(engine):
    """Create pm_candidates and pm_candidate_events tables for testing."""
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
                    candidate_lineage_id TEXT,
                    candidate_type TEXT DEFAULT 'intraday',
                    rejection_reason_code VARCHAR(64)
                )
                """
            )
        )
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
                    created_at TEXT NOT NULL,
                    candidate_type TEXT
                )
                """
            )
        )


def _register_candidate_for_persistence(engine, candidate_id, cycle_id, profile_id):
    """Insert a candidate in REGISTERED state for persistence testing."""
    now = datetime.now(timezone.utc)
    record_dict = {
        "candidate_id": candidate_id,
        "symbol": "TEST",
        "direction": "BUY",
        "entry_price": 100.0,
        "stop_price": 98.0,
        "target_price": 104.0,
        "setup_type": "momentum_fade",
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
                    target_price, risk_reward, source_signal_id,
                    signal_snapshot_json, state, integrity_hash,
                    created_at, expires_at
                ) VALUES (
                    :candidate_id, :cycle_id, :profile_id, :symbol, :direction,
                    :setup_type, :geometry_name, :entry_price, :stop_price,
                    :target_price, :risk_reward, :source_signal_id,
                    :signal_snapshot_json, :state, :integrity_hash,
                    :created_at, :expires_at
                )
                """
            ),
            {
                "candidate_id": candidate_id,
                "cycle_id": cycle_id,
                "profile_id": profile_id,
                "symbol": "TEST",
                "direction": "BUY",
                "setup_type": "momentum_fade",
                "geometry_name": "base_breakout",
                "entry_price": 100.0,
                "stop_price": 98.0,
                "target_price": 104.0,
                "risk_reward": 2.0,
                "source_signal_id": f"sig_{candidate_id[:8]}",
                "signal_snapshot_json": json.dumps({"symbol": "TEST"}),
                "state": CandidateState.REGISTERED.value,
                "integrity_hash": integrity_hash,
                "created_at": now.isoformat(),
                "expires_at": (now + timedelta(hours=1)).isoformat(),
            },
        )


def persist_rejection(
    engine,
    candidate_id: str,
    cycle_id: str,
    profile_id: str,
    rejection_reason_code: str,
    rationale: str,
) -> None:
    """Persist PM rejection: write rejection_reason_code to pm_candidates and
    truncated rationale to pm_candidate_events event_data JSON.

    This implements the persistence behavior specified in Requirement 1.3:
    - rejection_reason_code → pm_candidates.rejection_reason_code column
    - rationale (truncated to 2000 chars) → event_data JSON in pm_candidate_events
    - Also transitions candidate to REJECTED state with full rationale in rejection_reason
    """
    truncated_rationale = rationale[:2000]

    with engine.begin() as conn:
        # Persist rejection_reason_code and transition state
        conn.execute(
            text(
                """
                UPDATE pm_candidates
                SET state = :new_state,
                    rejection_reason = :rejection_reason,
                    rejection_reason_code = :rejection_reason_code
                WHERE candidate_id = :candidate_id
                  AND state = :expected_state
                """
            ),
            {
                "new_state": CandidateState.REJECTED.value,
                "rejection_reason": rationale,
                "rejection_reason_code": rejection_reason_code,
                "candidate_id": candidate_id,
                "expected_state": CandidateState.REGISTERED.value,
            },
        )

        # Write rejection event with truncated rationale in event_data
        event_data = json.dumps({
            "rationale": truncated_rationale,
            "rejection_reason_code": rejection_reason_code,
        })
        conn.execute(
            text(
                """
                INSERT INTO pm_candidate_events
                (candidate_id, cycle_id, profile_id, event_type, event_data, created_at)
                VALUES (:cid, :cycle_id, :profile_id, :event_type, :event_data, :created_at)
                """
            ),
            {
                "cid": candidate_id,
                "cycle_id": cycle_id,
                "profile_id": profile_id,
                "event_type": "pm_reject",
                "event_data": event_data,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )


# Strategy for arbitrary-length rationale strings (including very long ones > 2000 chars)
rationale_st = st.one_of(
    st.text(min_size=0, max_size=100),  # short strings
    st.text(min_size=1900, max_size=2100),  # around the boundary
    st.text(min_size=2001, max_size=5000),  # clearly over limit
)


@given(
    candidate_id=candidate_id_st,
    reason_code=st.sampled_from(sorted(VALID_REJECTION_REASON_CODES)),
    rationale=rationale_st,
)
@settings(max_examples=200)
def test_rejection_rationale_persistence_with_truncation(
    candidate_id: str, reason_code: str, rationale: str
):
    """Property 3: Rejection rationale persistence with truncation.

    For any PM rejection with a rationale string of arbitrary length:
    - The persisted event_data JSON in pm_candidate_events SHALL contain the
      rationale truncated to at most 2000 characters.
    - The rejection_reason_code column in pm_candidates SHALL contain the
      bounded reason code.

    **Validates: Requirements 1.3**
    """
    # Set up fresh in-memory database for each test case
    engine = create_engine("sqlite:///:memory:")
    _create_persistence_tables(engine)

    cycle_id = "cycle_test"
    profile_id = "moderate"

    # Register candidate
    _register_candidate_for_persistence(engine, candidate_id, cycle_id, profile_id)

    # Persist the rejection
    persist_rejection(engine, candidate_id, cycle_id, profile_id, reason_code, rationale)

    # Verify pm_candidates.rejection_reason_code contains the bounded code
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT state, rejection_reason_code, rejection_reason "
                "FROM pm_candidates WHERE candidate_id = :cid"
            ),
            {"cid": candidate_id},
        ).fetchone()

    assert row is not None, f"Candidate {candidate_id} not found in pm_candidates"
    assert row[0] == CandidateState.REJECTED.value, (
        f"Expected state REJECTED, got {row[0]}"
    )
    assert row[1] == reason_code, (
        f"Expected rejection_reason_code={reason_code!r}, got {row[1]!r}"
    )
    # INVARIANT: rejection_reason_code is always in VALID_REJECTION_REASON_CODES
    assert row[1] in VALID_REJECTION_REASON_CODES, (
        f"Persisted rejection_reason_code {row[1]!r} not in VALID_REJECTION_REASON_CODES"
    )

    # Verify pm_candidate_events has the rejection event with truncated rationale
    with engine.connect() as conn:
        event_row = conn.execute(
            text(
                "SELECT event_type, event_data "
                "FROM pm_candidate_events WHERE candidate_id = :cid AND event_type = 'pm_reject'"
            ),
            {"cid": candidate_id},
        ).fetchone()

    assert event_row is not None, (
        f"No pm_reject event found for candidate {candidate_id}"
    )
    assert event_row[0] == "pm_reject"

    event_data = json.loads(event_row[1])

    # INVARIANT: rationale in event_data is truncated to at most 2000 characters
    persisted_rationale = event_data["rationale"]
    assert len(persisted_rationale) <= 2000, (
        f"Persisted rationale length {len(persisted_rationale)} exceeds 2000 char limit"
    )

    # If original rationale was <= 2000, it should be preserved exactly
    if len(rationale) <= 2000:
        assert persisted_rationale == rationale, (
            f"Short rationale was not preserved exactly. "
            f"Expected length {len(rationale)}, got {len(persisted_rationale)}"
        )

    # If original rationale was > 2000, it should be truncated to exactly 2000
    if len(rationale) > 2000:
        assert len(persisted_rationale) == 2000, (
            f"Long rationale was not truncated to exactly 2000. "
            f"Got length {len(persisted_rationale)}"
        )
        # The truncated content should be the first 2000 chars of the original
        assert persisted_rationale == rationale[:2000], (
            "Truncated rationale does not match first 2000 chars of original"
        )

    # INVARIANT: event_data also contains the rejection_reason_code
    assert event_data["rejection_reason_code"] == reason_code, (
        f"Event data rejection_reason_code mismatch: "
        f"expected {reason_code!r}, got {event_data['rejection_reason_code']!r}"
    )
