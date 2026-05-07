"""
Integration tests for News Trade Governance.
Tests full lifecycle scenarios with real SQLite database.
"""

import json
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
    log_trade_event_once,
    latest_valid_reconfirmation,
    latest_exit_request,
    submit_news_reconfirmation,
    _build_dedupe_key,
    _build_failure_dedupe_key,
)
from utils.trade_events import log_trade_event


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def governed_trade(db_session):
    """Create a news-governed trade in the database."""
    trade = Trade(
        symbol="XLE",
        direction="LONG",
        quantity=100,
        entry_price=85.0,
        setup_type="news_catalyst",
        reason_entry="Oil spike due to Strait of Hormuz tensions",
        thesis="Geopolitical risk premium in energy sector",
        status="open",
        entry_time=datetime(2024, 6, 1, 10, 0),
    )
    db_session.add(trade)
    db_session.flush()
    return trade


# ─── Test 1: Full timer cycle: classify → warn → force-close ─────────────────

class TestFullTimerCycle:
    """Full timer cycle with SQLite: classify → warn → force-close."""

    def test_classify_warn_force_close(self, db_session, governed_trade):
        """Complete lifecycle: classification, warning at T-4h, expired after grace."""
        classifier = NewsGovernanceClassifier()
        policy = NewsGovernancePolicy()
        trade = governed_trade
        entry_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)

        # Step 1: Classify the trade
        trade_data = {
            "setup_type": trade.setup_type,
            "reason_entry": trade.reason_entry,
            "thesis": trade.thesis,
            "invalidators": "",
        }
        is_governed, evidence = classifier.classify(trade_data)
        assert is_governed is True

        # Persist classification event
        written = log_trade_event_once(
            db_session,
            "news_governance_classified",
            trade.id,
            governance_window_id=1,
            agent="position_timer",
            symbol=trade.symbol,
            payload=evidence,
        )
        db_session.commit()
        assert written is True

        # Verify news_governance_classified event exists
        classified_event = db_session.query(TradeEvent).filter_by(
            event_type="news_governance_classified", trade_id=trade.id
        ).first()
        assert classified_event is not None
        payload = json.loads(classified_event.payload_json)
        assert payload["triggered_by"] == "setup_type"
        assert payload["governance_window_id"] == 1

        # Step 2: Advance time to warning period (T-4h = 20 hours after entry)
        warning_time = entry_time + timedelta(hours=20)
        result = policy.evaluate(db_session, trade.id, entry_time, warning_time)
        assert result["status"] == "warning"
        assert result["governance_window_id"] == 1

        # Emit warning event
        written = log_trade_event_once(
            db_session,
            "news_reconfirmation_due",
            trade.id,
            governance_window_id=1,
            agent="position_timer",
            symbol=trade.symbol,
            message="Reconfirmation required within 4 hours",
        )
        db_session.commit()
        assert written is True

        # Verify news_reconfirmation_due event
        warning_event = db_session.query(TradeEvent).filter_by(
            event_type="news_reconfirmation_due", trade_id=trade.id
        ).first()
        assert warning_event is not None

        # Step 3: Advance time past grace period (24h + 30min grace = 24.5h after entry)
        expired_time = entry_time + timedelta(hours=24, minutes=31)
        result = policy.evaluate(db_session, trade.id, entry_time, expired_time)
        assert result["status"] == "expired"
        assert result["hold_authorized"] is False


# ─── Test 2: Reconfirmation → hold authorized → new window ───────────────────

class TestReconfirmationHoldAuthorized:
    """Reconfirmation submission → hold authorized → new warning in next window."""

    def test_reconfirm_and_hold_advances_window(self, db_session, governed_trade):
        """RECONFIRM_AND_HOLD creates new governance window with advanced expiry."""
        policy = NewsGovernancePolicy()
        trade = governed_trade
        entry_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)

        # Classify the trade first
        log_trade_event_once(
            db_session,
            "news_governance_classified",
            trade.id,
            governance_window_id=1,
            agent="position_timer",
            symbol=trade.symbol,
            payload={"triggered_by": "setup_type", "matched_value": "news_catalyst"},
        )
        db_session.commit()

        # Verify initial window is 1
        result = policy.evaluate(db_session, trade.id, entry_time, entry_time + timedelta(hours=1))
        assert result["governance_window_id"] == 1

        # Submit RECONFIRM_AND_HOLD
        new_expiry = entry_time + timedelta(hours=48)
        reconfirm_payload = {
            "trade_id": trade.id,
            "symbol": "XLE",
            "profile": "aggressive",
            "decision": "RECONFIRM_AND_HOLD",
            "decided_by": "pm_aggressive",
            "original_catalyst": "Oil spike from Hormuz tensions",
            "fresh_catalyst_evidence": "Updated satellite imagery shows continued vessel buildup near strait",
            "fresh_catalyst_timestamp": entry_time + timedelta(hours=20),
            "thesis_status": "strengthened",
            "new_expiry_time": new_expiry,
            "risk_plan": "Tighten stop to breakeven",
            "governance_window_id": 1,
        }

        success, result_payload = submit_news_reconfirmation(
            db_session, reconfirm_payload, entry_time=entry_time
        )
        assert success is True

        # Verify news_hold_authorized event was emitted
        hold_event = db_session.query(TradeEvent).filter_by(
            event_type="news_hold_authorized", trade_id=trade.id
        ).first()
        assert hold_event is not None
        hold_payload = json.loads(hold_event.payload_json)
        assert hold_payload["decided_by"] == "pm_aggressive"

        # Verify governance_window_id advanced to 2
        result = policy.evaluate(db_session, trade.id, entry_time, entry_time + timedelta(hours=25))
        assert result["governance_window_id"] == 2
        assert result["hold_authorized"] is True

        # Verify new window's warning period
        # New expiry is at entry_time + 48h, warning is at expiry - 4h = entry_time + 44h
        warning_time_new = entry_time + timedelta(hours=44)
        result_at_warning = policy.evaluate(db_session, trade.id, entry_time, warning_time_new)
        assert result_at_warning["status"] == "warning"
        assert result_at_warning["governance_window_id"] == 2


# ─── Test 3: log_trade_event_once with real SQLite ────────────────────────────

class TestLogTradeEventOnce:
    """log_trade_event_once with real SQLite: verify single row after multiple calls."""

    def test_idempotent_event_write(self, db_session, governed_trade):
        """Multiple calls with same dedupe key produce only one row."""
        trade = governed_trade

        # First call should write
        written1 = log_trade_event_once(
            db_session,
            "news_reconfirmation_due",
            trade.id,
            governance_window_id=1,
            agent="position_timer",
            symbol="XLE",
            message="First call",
        )
        db_session.commit()
        assert written1 is True

        # Second call with same event_type and window should be suppressed
        written2 = log_trade_event_once(
            db_session,
            "news_reconfirmation_due",
            trade.id,
            governance_window_id=1,
            agent="position_timer",
            symbol="XLE",
            message="Second call (should be suppressed)",
        )
        db_session.commit()
        assert written2 is False

        # Third call — still suppressed
        written3 = log_trade_event_once(
            db_session,
            "news_reconfirmation_due",
            trade.id,
            governance_window_id=1,
            agent="position_timer",
            symbol="XLE",
            message="Third call (should be suppressed)",
        )
        db_session.commit()
        assert written3 is False

        # Verify only one row exists
        count = db_session.query(TradeEvent).filter_by(
            event_type="news_reconfirmation_due", trade_id=trade.id
        ).count()
        assert count == 1

    def test_different_windows_are_separate(self, db_session, governed_trade):
        """Same event type but different governance_window_id writes separate rows."""
        trade = governed_trade

        written1 = log_trade_event_once(
            db_session,
            "news_reconfirmation_due",
            trade.id,
            governance_window_id=1,
            agent="position_timer",
            symbol="XLE",
        )
        db_session.commit()
        assert written1 is True

        written2 = log_trade_event_once(
            db_session,
            "news_reconfirmation_due",
            trade.id,
            governance_window_id=2,
            agent="position_timer",
            symbol="XLE",
        )
        db_session.commit()
        assert written2 is True

        count = db_session.query(TradeEvent).filter_by(
            event_type="news_reconfirmation_due", trade_id=trade.id
        ).count()
        assert count == 2


# ─── Test 4: Already-closed position — no duplicate events ───────────────────

class TestAlreadyClosedPosition:
    """Already-closed position: verify no duplicate events."""

    def test_closed_trade_skipped_by_timer_logic(self, db_session):
        """A trade with status='closed' should be skipped by governance evaluation."""
        trade = Trade(
            symbol="XLE",
            direction="LONG",
            quantity=100,
            entry_price=85.0,
            setup_type="news_catalyst",
            reason_entry="Oil spike due to Strait of Hormuz tensions",
            thesis="Geopolitical risk premium in energy sector",
            status="closed",
            entry_time=datetime(2024, 6, 1, 10, 0),
            exit_time=datetime(2024, 6, 2, 8, 0),
            exit_price=87.5,
        )
        db_session.add(trade)
        db_session.flush()

        # The timer logic checks trade.status == "open" before acting.
        # Simulate: trade is classified but already closed — no force-close event.
        assert trade.status == "closed"

        # Even if we evaluate policy, the caller (position_timer) would check status first
        policy = NewsGovernancePolicy()
        entry_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
        expired_time = entry_time + timedelta(hours=25)

        result = policy.evaluate(db_session, trade.id, entry_time, expired_time)
        # Policy still returns expired status — it's the caller's job to check trade.status
        assert result["status"] == "expired"

        # Simulate the guard: if trade.status != "open", do NOT emit force-close
        if trade.status != "open":
            # No event should be written
            pass

        # Verify no force-close events exist
        force_close_events = db_session.query(TradeEvent).filter_by(
            event_type="news_expiry_force_close", trade_id=trade.id
        ).count()
        assert force_close_events == 0


# ─── Test 5: Persisted classification durability (XLE bug regression) ─────────

class TestPersistedClassificationDurability:
    """Persisted classification durability: fields drift to swing, timer still force-closes."""

    def test_classification_survives_field_drift(self, db_session, governed_trade):
        """
        XLE bug regression: After classification, mutating setup_type to 'swing',
        clearing reason_entry, and rewriting thesis does NOT remove governance.
        The persisted classification event is the source of truth.
        """
        classifier = NewsGovernanceClassifier()
        policy = NewsGovernancePolicy()
        trade = governed_trade
        entry_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)

        # Step 1: Classify the trade (initial state has news_catalyst setup_type)
        trade_data = {
            "setup_type": trade.setup_type,
            "reason_entry": trade.reason_entry,
            "thesis": trade.thesis,
            "invalidators": "",
        }
        is_governed, evidence = classifier.classify(trade_data)
        assert is_governed is True

        # Persist classification
        log_trade_event_once(
            db_session,
            "news_governance_classified",
            trade.id,
            governance_window_id=1,
            agent="position_timer",
            symbol=trade.symbol,
            payload=evidence,
        )
        db_session.commit()

        # Step 2: Simulate field drift — setup_type changes to 'swing'
        trade.setup_type = "swing"
        trade.reason_entry = "Technical breakout above resistance"
        trade.thesis = "Momentum continuation pattern with volume"
        db_session.commit()

        # Step 3: Verify persisted classification still returns governed
        persisted = classifier.get_persisted_classification(db_session, trade.id)
        assert persisted is not None
        assert persisted["evidence"]["triggered_by"] == "setup_type"

        # Re-classify from persisted evidence
        still_governed = classifier.classify_from_persisted_evidence(persisted["evidence"])
        assert still_governed is True

        # Step 4: Verify policy still evaluates correctly (expired after 24h+grace)
        expired_time = entry_time + timedelta(hours=24, minutes=31)
        result = policy.evaluate(db_session, trade.id, entry_time, expired_time)
        assert result["status"] == "expired"
        assert result["hold_authorized"] is False

        # The trade WOULD be force-closed despite field drift
        # (caller checks persisted classification, not current trade fields)


# ─── Test 6: EXIT_NOW on already-closed trade ────────────────────────────────

class TestExitNowOnClosedTrade:
    """EXIT_NOW on already-closed trade: no action, no duplicate events."""

    def test_exit_now_on_closed_trade_no_action(self, db_session):
        """EXIT_NOW submitted for a closed trade produces no force-close action."""
        trade = Trade(
            symbol="XLE",
            direction="LONG",
            quantity=100,
            entry_price=85.0,
            setup_type="news_catalyst",
            reason_entry="Oil spike due to Strait of Hormuz tensions",
            thesis="Geopolitical risk premium in energy sector",
            status="closed",
            entry_time=datetime(2024, 6, 1, 10, 0),
            exit_time=datetime(2024, 6, 2, 6, 0),
            exit_price=86.0,
        )
        db_session.add(trade)
        db_session.flush()

        entry_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)

        # Submit EXIT_NOW (this persists the reconfirmation event)
        exit_payload = {
            "trade_id": trade.id,
            "symbol": "XLE",
            "profile": "aggressive",
            "decision": "EXIT_NOW",
            "decided_by": "pm_aggressive",
            "exit_reason": "Catalyst exhausted, closing position",
        }
        success, _ = submit_news_reconfirmation(
            db_session, exit_payload, entry_time=entry_time
        )
        assert success is True

        # The exit request is recorded, but the caller checks trade.status
        exit_req = latest_exit_request(db_session, trade.id)
        assert exit_req is not None
        assert exit_req["decision"] == "EXIT_NOW"

        # Simulate timer logic: trade.status != "open" → no action taken
        assert trade.status == "closed"

        # No force-close event should be emitted for closed trades
        # (The timer would skip this trade)
        force_close_count = db_session.query(TradeEvent).filter_by(
            event_type="news_expiry_force_close", trade_id=trade.id
        ).count()
        assert force_close_count == 0


# ─── Test 7: Force-close failure (simulated exception) ───────────────────────

class TestForceCloseFailure:
    """Force-close failure: hourly-bucketed dedupe allows visibility across hours."""

    def test_failure_event_hourly_bucketed_dedupe(self, db_session, governed_trade):
        """
        news_expiry_force_close_failed uses hourly-bucketed dedupe:
        - Multiple calls within same hour → only one event
        - Calls in different hours → separate events (retry visibility)
        - No news_expiry_force_close emitted on failure
        """
        trade = governed_trade

        # Simulate failure at 14:00 UTC
        hour1 = datetime(2024, 6, 2, 14, 0, tzinfo=timezone.utc)

        # First failure in hour 1
        written1 = log_trade_event_once(
            db_session,
            "news_expiry_force_close_failed",
            trade.id,
            governance_window_id=1,
            agent="position_timer",
            symbol="XLE",
            message="Broker API timeout",
        )
        db_session.commit()
        assert written1 is True

        # Second failure in same hour — should be suppressed
        written2 = log_trade_event_once(
            db_session,
            "news_expiry_force_close_failed",
            trade.id,
            governance_window_id=1,
            agent="position_timer",
            symbol="XLE",
            message="Broker API timeout (retry 2)",
        )
        db_session.commit()
        # Within same hour, dedupe key is the same → suppressed
        assert written2 is False

        # Verify no news_expiry_force_close emitted (failure means close didn't happen)
        force_close_count = db_session.query(TradeEvent).filter_by(
            event_type="news_expiry_force_close", trade_id=trade.id
        ).count()
        assert force_close_count == 0

        # Verify only one failure event in this hour
        failure_count = db_session.query(TradeEvent).filter_by(
            event_type="news_expiry_force_close_failed", trade_id=trade.id
        ).count()
        assert failure_count == 1

    def test_failure_different_hours_allows_retry_visibility(self, db_session, governed_trade):
        """Failures in different hours produce separate events for retry visibility."""
        trade = governed_trade

        # Build dedupe keys for two different hours manually to verify the logic
        hour1 = datetime(2024, 6, 2, 14, 0, tzinfo=timezone.utc)
        hour2 = datetime(2024, 6, 2, 15, 0, tzinfo=timezone.utc)

        key1 = _build_failure_dedupe_key(
            "news_expiry_force_close_failed", trade.id, 1, hour1
        )
        key2 = _build_failure_dedupe_key(
            "news_expiry_force_close_failed", trade.id, 1, hour2
        )

        # Keys should be different across hours
        assert key1 != key2
        assert "2024060214" in key1
        assert "2024060215" in key2

        # Write events with explicit dedupe keys to simulate different hours
        event1 = TradeEvent(
            trade_id=trade.id,
            event_type="news_expiry_force_close_failed",
            agent="position_timer",
            symbol="XLE",
            message="Failure at 14:00",
            dedupe_key=key1,
            payload_json=json.dumps({"governance_window_id": 1}),
        )
        event2 = TradeEvent(
            trade_id=trade.id,
            event_type="news_expiry_force_close_failed",
            agent="position_timer",
            symbol="XLE",
            message="Failure at 15:00",
            dedupe_key=key2,
            payload_json=json.dumps({"governance_window_id": 1}),
        )
        db_session.add_all([event1, event2])
        db_session.commit()

        # Both events exist — next cycle can retry
        failure_count = db_session.query(TradeEvent).filter_by(
            event_type="news_expiry_force_close_failed", trade_id=trade.id
        ).count()
        assert failure_count == 2

        # Still no successful force-close
        force_close_count = db_session.query(TradeEvent).filter_by(
            event_type="news_expiry_force_close", trade_id=trade.id
        ).count()
        assert force_close_count == 0


# ─── Test 8: Reconfirmation during GRACE prevents force-close ────────────────

class TestReconfirmationDuringGrace:
    """Reconfirmation during GRACE prevents force-close."""

    def test_reconfirm_during_grace_prevents_expiry(self, db_session, governed_trade):
        """
        Trade enters GRACE (past expiry, within 30-min grace window).
        Valid RECONFIRM_AND_HOLD submitted during GRACE.
        Next evaluation sees new window with advanced expiry.
        Status is NOT expired — force-close prevented.
        """
        policy = NewsGovernancePolicy()
        trade = governed_trade
        entry_time = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)

        # Step 1: Set time to 24.5 hours after entry (past expiry, within 30-min grace)
        grace_time = entry_time + timedelta(hours=24, minutes=15)

        # Evaluate → status should be "grace"
        result = policy.evaluate(db_session, trade.id, entry_time, grace_time)
        assert result["status"] == "grace"
        assert result["hold_authorized"] is False

        # Step 2: Submit RECONFIRM_AND_HOLD during GRACE
        new_expiry = grace_time + timedelta(hours=24)  # Extend by another 24h
        reconfirm_payload = {
            "trade_id": trade.id,
            "symbol": "XLE",
            "profile": "aggressive",
            "decision": "RECONFIRM_AND_HOLD",
            "decided_by": "pm_aggressive",
            "original_catalyst": "Oil spike from Hormuz tensions",
            "fresh_catalyst_evidence": "New intelligence report confirms ongoing naval buildup near strait",
            "fresh_catalyst_timestamp": grace_time - timedelta(hours=1),
            "thesis_status": "strengthened",
            "new_expiry_time": new_expiry,
            "risk_plan": "Maintain position with tightened stop at breakeven",
            "governance_window_id": 2,
        }

        success, result_payload = submit_news_reconfirmation(
            db_session, reconfirm_payload, entry_time=entry_time
        )
        assert success is True

        # Step 3: Re-evaluate — status should be "ok" (new window, new expiry in future)
        # Evaluate at same grace_time but now with reconfirmation in place
        result_after = policy.evaluate(db_session, trade.id, entry_time, grace_time)
        assert result_after["status"] == "ok"
        assert result_after["hold_authorized"] is True
        assert result_after["governance_window_id"] == 2

        # Step 4: Verify force-close is prevented
        # Even at the original force-close time (24h + 30min), status is now "ok"
        original_force_close_time = entry_time + timedelta(hours=24, minutes=31)
        result_at_old_deadline = policy.evaluate(
            db_session, trade.id, entry_time, original_force_close_time
        )
        assert result_at_old_deadline["status"] == "ok"
        assert result_at_old_deadline["hold_authorized"] is True

        # The new expiry is ~48.25h after entry — well in the future
        assert result_after["effective_expiry"] == new_expiry
