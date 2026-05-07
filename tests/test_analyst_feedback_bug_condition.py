"""
Bug Condition Exploration Test — NameError and DetachedInstanceError in Feedback Pipeline.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6**

This test encodes the EXPECTED (correct) behavior. It is designed to FAIL on
unfixed code, confirming the bugs exist. After the fix is applied, these tests
should PASS, confirming the bugs are resolved.

Bug conditions tested:
- NameError: AgentMemory not imported in feedback_loop/analyst_feedback.py
- DetachedInstanceError: ORM rows accessed after session close in
  build_feedback_prompt_context, get_active_mitigations, get_quality_metrics
- Silent failure: agents/analyst.py swallows exceptions with no health record
"""

import json
import logging
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from sqlalchemy import create_engine

from db.schema import (
    Base,
    AgentMemory,
    AnalystFeedbackQueue,
    AnalystMitigation,
    get_session,
)
from models.case import Case  # noqa: F401 — registers with Base


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_engine():
    """Create a fresh in-memory SQLite engine with all tables."""
    eng = create_engine("sqlite://")
    Base.metadata.create_all(eng)
    return eng


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

def pending_row_count():
    """Generate a random number of pending rows (1-10)."""
    return st.integers(min_value=1, max_value=10)


def mitigation_level():
    """Generate a random mitigation level."""
    return st.integers(min_value=1, max_value=4)


def deployment_multiplier():
    """Generate a random deployment multiplier."""
    return st.floats(min_value=0.25, max_value=1.0, allow_nan=False, allow_infinity=False)


def signal_threshold_bump():
    """Generate a random signal threshold bump."""
    return st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Test 1: process_pending_feedback — NameError on AgentMemory
# ---------------------------------------------------------------------------

@settings(
    max_examples=5,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
@given(num_rows=pending_row_count())
def test_process_pending_feedback_no_name_error(num_rows):
    """
    **Validates: Requirements 1.1**

    Call process_pending_feedback(engine) with pending AnalystFeedbackQueue rows.
    Assert no NameError is raised.

    On UNFIXED code, this will fail with:
      NameError: name 'AgentMemory' is not defined
    """
    from feedback_loop.analyst_feedback import process_pending_feedback

    engine = make_engine()

    # Seed pending rows
    db = get_session(engine)
    now = datetime.utcnow()
    for i in range(num_rows):
        db.add(AnalystFeedbackQueue(
            symbol=f"SYM{i}",
            setup_type="gap_and_go",
            date="2026-05-01",
            flag_type="selection_score_below_threshold",
            severity="high",
            recommendation="Tighten classification",
            reviewer_context=json.dumps({"test": True}),
            due_at=now + timedelta(hours=24),
            status="pending",
        ))
    db.commit()
    db.close()

    # Mock call_llm to avoid real LLM calls
    mock_llm_response = json.dumps({
        "responses": [
            {"id": i + 1, "action": "accept", "note": "ok", "supporting_data": [], "mitigation_plan": ""}
            for i in range(num_rows)
        ]
    })

    with patch("feedback_loop.analyst_feedback.call_llm", return_value=mock_llm_response):
        with patch("feedback_loop.analyst_feedback.parse_json_response", return_value={"responses": [
            {"id": i + 1, "action": "accept", "note": "ok", "supporting_data": [], "mitigation_plan": ""}
            for i in range(num_rows)
        ]}):
            # This should NOT raise NameError
            result = process_pending_feedback(engine)

    assert isinstance(result, dict)
    assert "responses" in result


# ---------------------------------------------------------------------------
# Test 2: build_feedback_prompt_context — DetachedInstanceError
# ---------------------------------------------------------------------------

@settings(
    max_examples=5,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
@given(
    num_rows=pending_row_count(),
    level=mitigation_level(),
    mult=deployment_multiplier(),
    bump=signal_threshold_bump(),
)
def test_build_feedback_prompt_context_no_detached_error(num_rows, level, mult, bump):
    """
    **Validates: Requirements 1.2**

    Call build_feedback_prompt_context(engine) with pending rows and active mitigations.
    Assert it returns a str without DetachedInstanceError.

    On UNFIXED code, this will fail with:
      sqlalchemy.orm.exc.DetachedInstanceError
    """
    from feedback_loop.analyst_feedback import build_feedback_prompt_context

    engine = make_engine()

    # Seed pending rows and active mitigations
    db = get_session(engine)
    now = datetime.utcnow()
    for i in range(num_rows):
        db.add(AnalystFeedbackQueue(
            symbol=f"TEST{i}",
            setup_type="gap_and_go",
            date="2026-05-01",
            flag_type="selection_score_below_threshold",
            severity="high",
            recommendation="Tighten classification",
            reviewer_context=json.dumps({}),
            due_at=now + timedelta(hours=24),
            status="pending",
        ))

    # Add active mitigation
    db.add(AnalystMitigation(
        setup_type="gap_and_go",
        level=level,
        deployment_multiplier=mult,
        signal_threshold_bump=bump,
        active=True,
        reason="Test mitigation",
    ))
    db.commit()
    db.close()

    # This should NOT raise DetachedInstanceError
    result = build_feedback_prompt_context(engine)

    assert isinstance(result, str)
    # Should contain content from the pending rows and mitigations
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Test 3: get_active_mitigations — DetachedInstanceError
# ---------------------------------------------------------------------------

@settings(
    max_examples=5,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
@given(
    level=mitigation_level(),
    mult=deployment_multiplier(),
    bump=signal_threshold_bump(),
)
def test_get_active_mitigations_no_detached_error(level, mult, bump):
    """
    **Validates: Requirements 1.3**

    Call get_active_mitigations(engine) with active AnalystMitigation rows.
    Assert it returns a dict without DetachedInstanceError.

    On UNFIXED code, this will fail with:
      sqlalchemy.orm.exc.DetachedInstanceError
    """
    from feedback_loop.analyst_feedback import get_active_mitigations

    engine = make_engine()

    # Seed active mitigations
    db = get_session(engine)
    db.add(AnalystMitigation(
        setup_type="orb",
        level=level,
        deployment_multiplier=mult,
        signal_threshold_bump=bump,
        active=True,
        reason="Test mitigation",
    ))
    db.commit()
    db.close()

    # This should NOT raise DetachedInstanceError
    result = get_active_mitigations(engine)

    assert isinstance(result, dict)
    assert "orb" in result
    entry = result["orb"]
    assert "level" in entry
    assert "deployment_multiplier" in entry
    assert "signal_threshold_bump" in entry
    assert "reason" in entry


# ---------------------------------------------------------------------------
# Test 4: get_quality_metrics — DetachedInstanceError
# ---------------------------------------------------------------------------

@settings(
    max_examples=5,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
@given(num_rows=pending_row_count())
def test_get_quality_metrics_no_detached_error(num_rows):
    """
    **Validates: Requirements 1.4**

    Call get_quality_metrics(engine) with today's AnalystFeedbackQueue rows.
    Assert it returns a dict without DetachedInstanceError.

    On UNFIXED code, this will fail with:
      sqlalchemy.orm.exc.DetachedInstanceError
    """
    from feedback_loop.analyst_feedback import get_quality_metrics

    engine = make_engine()

    # Seed today's rows (some responded, some pending)
    db = get_session(engine)
    now = datetime.utcnow()
    for i in range(num_rows):
        responded_at = now - timedelta(hours=1) if i % 2 == 0 else None
        db.add(AnalystFeedbackQueue(
            symbol=f"QM{i}",
            setup_type="orb",
            date="2026-05-01",
            flag_type="signal_strength_below_threshold",
            severity="medium",
            recommendation="Require stronger confirmation",
            reviewer_context=json.dumps({}),
            due_at=now - timedelta(hours=1),  # already due
            status="responded" if responded_at else "pending",
            created_at=now - timedelta(hours=2),
            responded_at=responded_at,
            analyst_response="accept" if responded_at else None,
            analyst_supporting_data=json.dumps([]) if responded_at else None,
            no_data_reject=False,
        ))
    # Add an active mitigation to exercise that code path
    db.add(AnalystMitigation(
        setup_type="orb",
        level=1,
        deployment_multiplier=0.75,
        signal_threshold_bump=0.5,
        active=True,
        reason="Test",
    ))
    db.commit()
    db.close()

    # This should NOT raise DetachedInstanceError
    result = get_quality_metrics(engine)

    assert isinstance(result, dict)
    assert "flags_received" in result
    assert "flags_accepted" in result
    assert "response_rate" in result
    assert "acceptance_rate" in result
    assert "current_mitigation_level" in result
    assert "active_mitigations" in result


# ---------------------------------------------------------------------------
# Test 5: agents/analyst.py run() — silent failure, no health record
# ---------------------------------------------------------------------------

@settings(
    max_examples=3,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
@given(num_symbols=st.integers(min_value=1, max_value=3))
def test_run_exception_handler_writes_health_record(num_symbols):
    """
    **Validates: Requirements 1.5, 1.6**

    Call agents/analyst.py run() with process_pending_feedback mocked to raise.
    Assert an AgentMemory health record with key="feedback_processing_status" exists.

    On UNFIXED code, this will fail because:
      - The exception handler only does log.warning, no health record is written
      - No AgentMemory row with key="feedback_processing_status" will exist
    """
    engine = make_engine()
    symbols = [f"TST{i}" for i in range(num_symbols)]

    # We need to mock many things to isolate the exception handler path
    with patch("agents.analyst.FinnhubClient") as mock_fh_cls, \
         patch("agents.analyst.process_pending_feedback", side_effect=RuntimeError("Test failure")), \
         patch("agents.analyst.build_feedback_prompt_context", return_value=""), \
         patch("agents.analyst.get_active_mitigations", return_value={}), \
         patch("agents.analyst.build_strategy_context", return_value=""), \
         patch("agents.analyst.call_llm", return_value='{"signal":"HOLD","strength":"weak","confidence":"low","setup_type":"error","reasoning":"test"}'), \
         patch("agents.analyst.parse_json_response", return_value={"signal": "HOLD", "strength": "weak", "confidence": "low", "setup_type": "error", "reasoning": "test"}), \
         patch("agents.analyst.compute_indicators", return_value={}), \
         patch("agents.analyst.get_relevant_cases", return_value=[]), \
         patch("agents.analyst.format_cases_for_prompt", return_value=""), \
         patch("agents.analyst.validate_setup_for_symbol", return_value={}), \
         patch("agents.analyst.get_breaking_news_for_symbols", return_value={}), \
         patch("utils.strategy_store.get_all_setup_types", return_value=["gap_and_go", "orb"]):

        mock_fh = MagicMock()
        mock_fh.get_candles.return_value = []
        mock_fh.get_quote.return_value = {}
        mock_fh_cls.return_value = mock_fh

        from agents.analyst import run
        run(engine, symbols)

    # Check that a health record was written
    db = get_session(engine)
    health_record = (
        db.query(AgentMemory)
        .filter_by(agent="analyst_feedback", key="feedback_processing_status")
        .order_by(AgentMemory.timestamp.desc())
        .first()
    )
    db.close()

    # On unfixed code, this assertion will FAIL because no health record is written
    assert health_record is not None, (
        "Expected AgentMemory health record with key='feedback_processing_status' "
        "but none was found. The exception handler in agents/analyst.py only logs "
        "a warning without writing a structured health record."
    )

    # Verify the health record has the expected structure
    parsed = json.loads(health_record.value)
    assert parsed["status"] == "failed"
    assert "errors" in parsed
    assert len(parsed["errors"]) > 0
