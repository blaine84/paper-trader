"""
Example-based unit tests for News Trade Governance.
Tests specific scenarios from the requirements document.
"""

import json
import logging
import pytest
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, Trade, TradeEvent
from utils.news_trade_governance import (
    NEWS_GOVERNANCE,
    NewsGovernanceClassifier,
    NewsGovernancePolicy,
    ReconfirmationValidator,
    validate_news_governance_config,
    submit_news_reconfirmation,
    log_trade_event_once,
)


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


# ─── Test 1: XLE trade with "Strait of Hormuz" classified as governed ─────────

def test_xle_strait_of_hormuz_classified_as_governed():
    """XLE trade with 'Strait of Hormuz' in reason_entry, setup_type='swing' → governed."""
    classifier = NewsGovernanceClassifier()
    trade = {
        "setup_type": "swing",
        "reason_entry": "Oil spike due to Strait of Hormuz tensions",
        "thesis": "Geopolitical risk premium in energy sector",
        "invalidators": "",
    }
    is_governed, evidence = classifier.classify(trade)
    assert is_governed is True
    assert evidence["triggered_by"] == "entry_text_terms"
    assert evidence["matched_field"] == "reason_entry"
    assert "strait of hormuz" in evidence["matched_value"].lower()


# ─── Test 2: Plain technical breakout not governed ────────────────────────────

def test_plain_technical_breakout_not_governed():
    """Plain technical breakout with no news terms → not governed."""
    classifier = NewsGovernanceClassifier()
    trade = {
        "setup_type": "swing",
        "reason_entry": "Breakout above 200-day moving average with volume confirmation",
        "thesis": "Technical momentum continuation pattern",
        "invalidators": "Close below breakout level",
    }
    is_governed, evidence = classifier.classify(trade)
    assert is_governed is False
    assert evidence == {}


# ─── Test 3: Reconfirmation with thesis_status='deteriorating' rejected ───────

def test_reconfirmation_deteriorating_thesis_rejected():
    """Reconfirmation with thesis_status='deteriorating' → rejected."""
    validator = ReconfirmationValidator()
    entry_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
    payload = {
        "trade_id": 1,
        "symbol": "XLE",
        "profile": "aggressive",
        "decision": "RECONFIRM_AND_HOLD",
        "decided_by": "pm_aggressive",
        "original_catalyst": "Oil spike from Hormuz tensions",
        "fresh_catalyst_evidence": "Updated satellite imagery shows continued vessel buildup",
        "fresh_catalyst_timestamp": datetime(2024, 6, 2, 8, 0, tzinfo=timezone.utc),
        "thesis_status": "deteriorating",
        "new_expiry_time": datetime(2024, 6, 2, 10, 0, tzinfo=timezone.utc),
        "risk_plan": "Tighten stop to breakeven",
    }
    is_valid, errors = validator.validate(payload, entry_time=entry_time)
    assert is_valid is False
    assert any("deteriorating" in e for e in errors)


# ─── Test 4: EXIT_NOW with only exit_reason accepted ─────────────────────────

def test_exit_now_with_exit_reason_accepted():
    """EXIT_NOW with only exit_reason → accepted."""
    validator = ReconfirmationValidator()
    entry_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
    payload = {
        "trade_id": 1,
        "symbol": "XLE",
        "profile": "aggressive",
        "decision": "EXIT_NOW",
        "decided_by": "pm_aggressive",
        "exit_reason": "Catalyst exhausted, no further upside expected",
    }
    is_valid, errors = validator.validate(payload, entry_time=entry_time)
    assert is_valid is True
    assert errors == []


# ─── Test 5: LET_EXPIRE with only decline_reason accepted ────────────────────

def test_let_expire_with_decline_reason_accepted():
    """LET_EXPIRE with only decline_reason → accepted."""
    validator = ReconfirmationValidator()
    entry_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
    payload = {
        "trade_id": 1,
        "symbol": "XLE",
        "profile": "aggressive",
        "decision": "LET_EXPIRE",
        "decided_by": "pm_aggressive",
        "decline_reason": "Position is near breakeven, let governance window expire naturally",
    }
    is_valid, errors = validator.validate(payload, entry_time=entry_time)
    assert is_valid is True
    assert errors == []


# ─── Test 6: allow_swing_reclassify with EXIT_NOW rejected ───────────────────

def test_allow_swing_reclassify_with_exit_now_rejected():
    """allow_swing_reclassify=true with decision='EXIT_NOW' → rejected."""
    validator = ReconfirmationValidator()
    entry_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
    payload = {
        "trade_id": 1,
        "symbol": "XLE",
        "profile": "aggressive",
        "decision": "EXIT_NOW",
        "decided_by": "pm_aggressive",
        "exit_reason": "Catalyst exhausted",
        "allow_swing_reclassify": True,
    }
    is_valid, errors = validator.validate(payload, entry_time=entry_time)
    assert is_valid is False
    assert any("allow_swing_reclassify" in e for e in errors)


# ─── Test 7: Naive datetime in new_expiry_time rejected ───────────────────────

def test_naive_datetime_in_new_expiry_time_rejected():
    """Naive datetime in new_expiry_time → rejected."""
    validator = ReconfirmationValidator()
    entry_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
    payload = {
        "trade_id": 1,
        "symbol": "XLE",
        "profile": "aggressive",
        "decision": "RECONFIRM_AND_HOLD",
        "decided_by": "pm_aggressive",
        "original_catalyst": "Oil spike from Hormuz tensions",
        "fresh_catalyst_evidence": "Updated satellite imagery shows continued vessel buildup",
        "fresh_catalyst_timestamp": datetime(2024, 6, 2, 8, 0, tzinfo=timezone.utc),
        "thesis_status": "strengthened",
        "new_expiry_time": datetime(2024, 6, 2, 10, 0),  # naive - no tzinfo!
        "risk_plan": "Tighten stop to breakeven",
    }
    is_valid, errors = validator.validate(payload, entry_time=entry_time)
    assert is_valid is False
    assert any("naive" in e.lower() or "timezone" in e.lower() for e in errors)


# ─── Test 8: validate_news_governance_config logs WARNING when disabled ───────

def test_validate_config_logs_warning_when_disabled(caplog):
    """validate_news_governance_config() logs WARNING when disabled."""
    config = {**NEWS_GOVERNANCE, "enabled": False}
    with caplog.at_level(logging.WARNING):
        result = validate_news_governance_config(config)
    assert "DISABLED" in caplog.text
    assert result == config


# ─── Test 9: validate_news_governance_config raises ValueError on malformed ──

def test_validate_config_raises_on_malformed():
    """validate_news_governance_config() raises ValueError on malformed config."""
    config = {**NEWS_GOVERNANCE, "max_hold_hours": 0}
    with pytest.raises(ValueError, match="max_hold_hours"):
        validate_news_governance_config(config)


# ─── Test 10: validate_news_governance_config returns config on valid ─────────

def test_validate_config_returns_config_on_valid():
    """validate_news_governance_config() returns config dict on valid config."""
    result = validate_news_governance_config(NEWS_GOVERNANCE)
    assert result == NEWS_GOVERNANCE
    assert result["enabled"] is True
    assert result["max_hold_hours"] == 24


# ─── Test 11: news_exit_requested payload includes close_confirmed ────────────

def test_exit_now_includes_close_confirmed(db_session):
    """news_exit_requested payload includes close_confirmed: true after successful close."""
    # Create a trade
    trade = Trade(
        symbol="XLE", direction="LONG", quantity=100,
        entry_price=85.0, status="open"
    )
    db_session.add(trade)
    db_session.flush()

    entry_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
    payload = {
        "trade_id": trade.id,
        "symbol": "XLE",
        "profile": "aggressive",
        "decision": "EXIT_NOW",
        "decided_by": "pm_aggressive",
        "exit_reason": "Catalyst exhausted",
    }

    success, result = submit_news_reconfirmation(db_session, payload, entry_time=entry_time)
    assert success is True

    # Check the news_exit_requested event has close_confirmed: true
    event = db_session.query(TradeEvent).filter_by(
        event_type="news_exit_requested", trade_id=trade.id
    ).first()
    assert event is not None
    event_payload = json.loads(event.payload_json)
    assert event_payload["close_confirmed"] is True
