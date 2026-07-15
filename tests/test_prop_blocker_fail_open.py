"""
Property-based tests for fail-open observability guards.

Feature: candidate-blocker-mitigation
Property: 17

Property 17: Fail-open guards on observability emissions

For any observability emission point (preflight events, lifecycle checklist,
loss summary, gate/sizing telemetry), if an exception occurs during emission,
the pipeline SHALL continue execution without blocking candidate state
transitions or trade execution.

**Validates: Requirements 9.4, 2.6, 5.5, 7.3**
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, PropertyMock

from hypothesis import given, settings, strategies as st

from utils.candidate_pipeline import (
    _write_preflight_passed_event,
    _write_preflight_failed_event,
    _write_preflight_excluded_event,
    _write_pm_accept_event,
    _write_sizing_event,
    _write_gate_events,
    _write_execution_failed_event,
)
from utils.candidate_registry import CandidateRecord
from utils.daily_loss_summary import (
    DailyLossSummary,
    compute_daily_loss_summary,
    persist_daily_loss_summary,
)
from utils.lifecycle_checklist import write_lifecycle_checklist
from utils.preflight_validator import (
    PreflightSummary,
    compute_preflight_safe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate_record(
    candidate_id: str | None = None,
    symbol: str = "AAPL",
    direction: str = "BUY",
    entry_price: float = 150.0,
    stop_price: float = 145.0,
    target_price: float = 160.0,
    risk_reward: float = 2.0,
    setup_type: str = "breakout",
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
        setup_type=setup_type,
        geometry_name="analyst_geometry",
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        risk_reward=risk_reward,
        trigger="price_level",
        invalidation_basis="below_stop",
        target_basis="measured_move",
        source_signal_id="signal-1",
        signal_snapshot_json='{"source": "test"}',
        created_at=now,
        expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
        integrity_hash="hash123",
        candidate_type="intraday",
    )


def _make_preflight_summary(
    candidate_id: str,
    passed: bool = True,
) -> PreflightSummary:
    """Create a PreflightSummary for testing."""
    if passed:
        return PreflightSummary(
            candidate_id=candidate_id,
            has_entry_stop_target=True,
            min_risk_reward_met=True,
            direction_valid=True,
            profile_allowed=True,
            candidate_not_expired=True,
            cash_available=True,
            sizing_possible=True,
            max_positions_available=True,
            same_symbol_allowed=True,
            blocking_reason_codes=[],
        )
    return PreflightSummary(
        candidate_id=candidate_id,
        has_entry_stop_target=True,
        min_risk_reward_met=False,
        direction_valid=True,
        profile_allowed=False,
        candidate_not_expired=True,
        cash_available=True,
        sizing_possible=True,
        max_positions_available=True,
        same_symbol_allowed=True,
        blocking_reason_codes=["min_risk_reward_not_met", "profile_not_allowed"],
    )


def _broken_engine():
    """Create a MagicMock engine whose connect() raises RuntimeError."""
    engine = MagicMock()
    engine.connect.side_effect = RuntimeError("DB down")
    return engine


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_candidate_id_st = st.uuids().map(str)
_cycle_id_st = st.text(min_size=1, max_size=32, alphabet=st.characters(whitelist_categories=("L", "N", "Pd")))
_profile_id_st = st.sampled_from(["moderate", "aggressive", "conservative", "scalp"])
_symbol_st = st.sampled_from(["AAPL", "TSLA", "MSFT", "XLE", "SPY", "QQQ", "NVDA"])
_direction_st = st.sampled_from(["BUY", "SHORT"])
_price_st = st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False)
_risk_reward_st = st.floats(min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False)
_quantity_st = st.integers(min_value=0, max_value=10000)
_risk_multiplier_st = st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)
_reason_st = st.text(min_size=0, max_size=200, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")))
_trade_date_st = st.dates().map(lambda d: d.isoformat())


# ---------------------------------------------------------------------------
# Property 17: Fail-open guards on observability emissions
# ---------------------------------------------------------------------------


@given(candidate_id=_candidate_id_st, cycle_id=_cycle_id_st, profile_id=_profile_id_st)
@settings(max_examples=200)
def test_write_preflight_passed_event_fail_open(candidate_id, cycle_id, profile_id):
    """_write_preflight_passed_event never raises on engine failure.

    **Validates: Requirements 9.4, 2.6, 5.5, 7.3**
    """
    broken_engine = _broken_engine()
    candidate = _make_candidate_record(candidate_id=candidate_id)
    summary = _make_preflight_summary(candidate_id, passed=True)

    # Must not raise
    _write_preflight_passed_event(broken_engine, candidate, cycle_id, profile_id, summary)


@given(candidate_id=_candidate_id_st, cycle_id=_cycle_id_st, profile_id=_profile_id_st)
@settings(max_examples=200)
def test_write_preflight_failed_event_fail_open(candidate_id, cycle_id, profile_id):
    """_write_preflight_failed_event never raises on engine failure.

    **Validates: Requirements 9.4, 2.6, 5.5, 7.3**
    """
    broken_engine = _broken_engine()
    candidate = _make_candidate_record(candidate_id=candidate_id)
    summary = _make_preflight_summary(candidate_id, passed=False)

    # Must not raise
    _write_preflight_failed_event(broken_engine, candidate, cycle_id, profile_id, summary)


@given(candidate_id=_candidate_id_st, cycle_id=_cycle_id_st, profile_id=_profile_id_st)
@settings(max_examples=200)
def test_write_preflight_excluded_event_fail_open(candidate_id, cycle_id, profile_id):
    """_write_preflight_excluded_event never raises on engine failure.

    **Validates: Requirements 9.4, 2.6, 5.5, 7.3**
    """
    broken_engine = _broken_engine()
    candidate = _make_candidate_record(candidate_id=candidate_id)
    summary = _make_preflight_summary(candidate_id, passed=False)

    # Must not raise
    _write_preflight_excluded_event(broken_engine, candidate, cycle_id, profile_id, summary)


@given(candidate_id=_candidate_id_st, cycle_id=_cycle_id_st, profile_id=_profile_id_st, risk_multiplier=_risk_multiplier_st)
@settings(max_examples=200)
def test_write_pm_accept_event_fail_open(candidate_id, cycle_id, profile_id, risk_multiplier):
    """_write_pm_accept_event never raises on engine failure.

    **Validates: Requirements 9.4, 2.6, 5.5, 7.3**
    """
    broken_engine = _broken_engine()
    candidate = _make_candidate_record(candidate_id=candidate_id)

    # Must not raise
    _write_pm_accept_event(broken_engine, candidate, cycle_id, profile_id, risk_multiplier)


@given(
    candidate_id=_candidate_id_st,
    cycle_id=_cycle_id_st,
    profile_id=_profile_id_st,
    passed=st.booleans(),
    quantity=_quantity_st,
    dollar_risk=_price_st,
    risk_percent=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    reason_code=st.one_of(st.none(), _reason_st),
)
@settings(max_examples=200)
def test_write_sizing_event_fail_open(candidate_id, cycle_id, profile_id, passed, quantity, dollar_risk, risk_percent, reason_code):
    """_write_sizing_event never raises on engine failure.

    **Validates: Requirements 9.4, 2.6, 5.5, 7.3**
    """
    broken_engine = _broken_engine()
    candidate = _make_candidate_record(candidate_id=candidate_id)

    # Must not raise
    _write_sizing_event(
        broken_engine, candidate, cycle_id, profile_id,
        passed=passed, quantity=quantity, dollar_risk=dollar_risk,
        risk_percent=risk_percent, reason_code=reason_code,
    )


@given(
    candidate_id=_candidate_id_st,
    cycle_id=_cycle_id_st,
    profile_id=_profile_id_st,
    gate_notes=st.lists(
        st.fixed_dictionaries({
            "gate": st.sampled_from(["setup_quality", "pre_trade_quality", "catalyst_specificity", "risk_geometry", "concentration"]),
            "decision": st.sampled_from(["pass", "reject", "approved", "block"]),
        }),
        min_size=0,
        max_size=5,
    ),
)
@settings(max_examples=200)
def test_write_gate_events_fail_open(candidate_id, cycle_id, profile_id, gate_notes):
    """_write_gate_events never raises on engine failure.

    **Validates: Requirements 9.4, 2.6, 5.5, 7.3**
    """
    broken_engine = _broken_engine()
    candidate = _make_candidate_record(candidate_id=candidate_id)

    # Must not raise
    _write_gate_events(broken_engine, candidate, cycle_id, profile_id, gate_notes)


@given(
    candidate_id=_candidate_id_st,
    cycle_id=_cycle_id_st,
    profile_id=_profile_id_st,
    intended_action=_direction_st,
    attempted_quantity=_quantity_st,
    failure_reason=st.text(min_size=0, max_size=2048, alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"))),
)
@settings(max_examples=200)
def test_write_execution_failed_event_fail_open(candidate_id, cycle_id, profile_id, intended_action, attempted_quantity, failure_reason):
    """_write_execution_failed_event never raises on engine failure.

    **Validates: Requirements 9.4, 2.6, 5.5, 7.3**
    """
    broken_engine = _broken_engine()
    candidate = _make_candidate_record(candidate_id=candidate_id)

    # Must not raise
    _write_execution_failed_event(
        broken_engine, candidate, cycle_id, profile_id,
        intended_action, attempted_quantity, failure_reason,
    )


@given(
    candidate_id=_candidate_id_st,
    trade_id=st.text(min_size=1, max_size=64, alphabet=st.characters(whitelist_categories=("L", "N", "Pd"))),
    cycle_id=_cycle_id_st,
    profile_id=_profile_id_st,
)
@settings(max_examples=200)
def test_write_lifecycle_checklist_fail_open(candidate_id, trade_id, cycle_id, profile_id):
    """write_lifecycle_checklist returns None (not raises) on engine failure.

    **Validates: Requirements 9.4, 2.6, 5.5, 7.3**
    """
    broken_engine = _broken_engine()

    # Must not raise; must return None
    result = write_lifecycle_checklist(broken_engine, candidate_id, trade_id, cycle_id, profile_id)
    assert result is None


@given(trade_date=_trade_date_st, profile_id=_profile_id_st)
@settings(max_examples=200)
def test_compute_daily_loss_summary_fail_open(trade_date, profile_id):
    """compute_daily_loss_summary returns summary with error_indication on engine failure.

    **Validates: Requirements 9.4, 2.6, 5.5, 7.3**
    """
    broken_engine = _broken_engine()

    # Must not raise; must return a DailyLossSummary with error_indication set
    result = compute_daily_loss_summary(broken_engine, trade_date, profile_id)
    assert result is not None
    assert result.error_indication is not None
    assert "DB down" in result.error_indication or "Query failure" in result.error_indication


@given(trade_date=_trade_date_st, profile_id=_profile_id_st)
@settings(max_examples=200)
def test_persist_daily_loss_summary_fail_open(trade_date, profile_id):
    """persist_daily_loss_summary returns False (not raises) on engine failure.

    **Validates: Requirements 9.4, 2.6, 5.5, 7.3**
    """
    broken_engine = _broken_engine()
    summary = DailyLossSummary(
        trade_date=trade_date,
        profile_id=profile_id,
        signals_seen=5,
        candidates_built=4,
        preflight_failed=1,
        offered_to_pm=3,
        pm_rejected=1,
        pm_rejected_by_reason={"low_confidence": 1},
        pm_accepted=2,
        gate_sizing_rejected=0,
        execution_failed=0,
        executed=2,
        lifecycle_incomplete=0,
        top_blocking_reasons=["low_confidence"],
        dominant_blocker_stage=None,
        error_indication=None,
    )

    # Must not raise; must return False
    result = persist_daily_loss_summary(broken_engine, summary)
    assert result is False


@given(
    candidate_id=_candidate_id_st,
    entry_price=_price_st,
    stop_price=_price_st,
    target_price=_price_st,
    risk_reward=_risk_reward_st,
    direction=_direction_st,
)
@settings(max_examples=200)
def test_compute_preflight_safe_fail_open(candidate_id, entry_price, stop_price, target_price, risk_reward, direction):
    """compute_preflight_safe returns a passing summary on broken inputs/exceptions.

    When the underlying compute_preflight raises (e.g., due to broken inputs
    that cause unexpected attribute errors), compute_preflight_safe catches
    the exception and returns a passing PreflightSummary.

    **Validates: Requirements 9.4, 2.6, 5.5, 7.3**
    """
    # Create a candidate with potentially problematic values
    candidate = _make_candidate_record(
        candidate_id=candidate_id,
        direction=direction,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        risk_reward=risk_reward,
    )

    # Use a broken profile dict that will cause exceptions
    # (missing "min_risk_reward" key won't raise, but a non-dict will)
    broken_profile = MagicMock()
    broken_profile.get.side_effect = TypeError("broken profile")

    broken_portfolio = MagicMock()
    broken_portfolio.get.side_effect = TypeError("broken portfolio")

    now = datetime.now(timezone.utc)

    # Must not raise; should return a passing summary on error
    result = compute_preflight_safe(
        candidate, broken_profile, broken_portfolio, [], now,
    )
    assert result is not None
    assert result.passed is True
    assert result.blocking_reason_codes == []
    assert result.candidate_id == candidate_id
