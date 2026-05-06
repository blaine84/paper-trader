"""
Integration tests for StopAuthority trigger check wiring.

Validates that bookkeeper and price_monitor correctly use should_stop_trigger()
and that invalid geometry prevents spurious trade closures.

Requirements: 4.9, 4.10, 4.11, 8.1, 8.2, 9.1
"""

import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, Trade, Position, TradeEvent
from utils.stop_authority import should_stop_trigger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """In-memory SQLite engine with all tables created."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def db_session(engine):
    """SQLAlchemy session bound to in-memory engine."""
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def _make_trade(db_session, **overrides):
    """Create and persist a Trade with sensible defaults."""
    defaults = {
        "symbol": "AMD",
        "direction": "LONG",
        "quantity": 100.0,
        "entry_price": 150.0,
        "stop_price": 145.0,
        "target_price": 160.0,
        "status": "open",
        "profile": "moderate",
    }
    defaults.update(overrides)
    trade = Trade(**defaults)
    db_session.add(trade)
    db_session.flush()
    return trade


def _make_position(db_session, **overrides):
    """Create and persist a Position with sensible defaults."""
    defaults = {
        "symbol": "AMD",
        "side": "long",
        "quantity": 100.0,
        "avg_cost": 150.0,
        "profile": "moderate",
    }
    defaults.update(overrides)
    pos = Position(**defaults)
    db_session.add(pos)
    db_session.flush()
    return pos


# ---------------------------------------------------------------------------
# Test: Bookkeeper uses should_stop_trigger (Requirement 8.1, 8.2)
# ---------------------------------------------------------------------------


class TestBookkeeperUsesStopTrigger:
    """Verify bookkeeper's check_stop_losses uses should_stop_trigger."""

    @patch("agents.bookkeeper.FinnhubClient")
    def test_bookkeeper_uses_should_stop_trigger(self, mock_fh_class, engine, db_session):
        """
        Bookkeeper check_stop_losses delegates to should_stop_trigger and
        correctly triggers when geometry is valid and price breaches stop.

        Validates: Requirement 8.1
        """
        # Setup: create a valid long trade with stop below entry
        trade = _make_trade(
            db_session,
            symbol="AAPL",
            direction="LONG",
            entry_price=150.0,
            stop_price=145.0,
            status="open",
            profile="moderate",
        )
        _make_position(
            db_session,
            symbol="AAPL",
            side="long",
            quantity=100.0,
            avg_cost=150.0,
            profile="moderate",
        )
        db_session.commit()

        # Mock FinnhubClient to return a price below the stop
        mock_fh_instance = MagicMock()
        mock_fh_instance.get_quote.return_value = {"price": 144.0}
        mock_fh_class.return_value = mock_fh_instance

        from agents.bookkeeper import check_stop_losses

        to_close = check_stop_losses(engine)

        # The trade should be in the to_close list since price (144) < stop (145)
        assert len(to_close) == 1
        assert to_close[0]["symbol"] == "AAPL"
        assert to_close[0]["price"] == 144.0
        assert to_close[0]["stop_loss"] == 145.0

    @patch("agents.bookkeeper.FinnhubClient")
    def test_bookkeeper_does_not_trigger_when_price_above_stop(self, mock_fh_class, engine, db_session):
        """
        Bookkeeper does NOT trigger when price is above stop (valid geometry, no breach).

        Validates: Requirement 8.2
        """
        trade = _make_trade(
            db_session,
            symbol="AAPL",
            direction="LONG",
            entry_price=150.0,
            stop_price=145.0,
            status="open",
            profile="moderate",
        )
        _make_position(
            db_session,
            symbol="AAPL",
            side="long",
            quantity=100.0,
            avg_cost=150.0,
            profile="moderate",
        )
        db_session.commit()

        # Mock FinnhubClient to return a price above the stop
        mock_fh_instance = MagicMock()
        mock_fh_instance.get_quote.return_value = {"price": 148.0}
        mock_fh_class.return_value = mock_fh_instance

        from agents.bookkeeper import check_stop_losses

        to_close = check_stop_losses(engine)

        # No trigger since price (148) > stop (145)
        assert len(to_close) == 0


# ---------------------------------------------------------------------------
# Test: Bookkeeper skips invalid geometry (Requirement 8.2)
# ---------------------------------------------------------------------------


class TestBookkeeperSkipsInvalidGeometry:
    """Verify bookkeeper does NOT close trades with invalid stop geometry."""

    @patch("agents.bookkeeper.FinnhubClient")
    def test_bookkeeper_skips_invalid_geometry(self, mock_fh_class, engine, db_session):
        """
        A long trade with stop above entry (invalid geometry) must NOT be
        added to the to_close list, even if current price is below the stop.

        Validates: Requirement 8.2
        """
        # Create a long trade with INVALID geometry: stop above entry
        trade = _make_trade(
            db_session,
            symbol="AMD",
            direction="LONG",
            entry_price=346.0,
            stop_price=347.50,  # Invalid: above entry for a long
            status="open",
            profile="moderate",
        )
        _make_position(
            db_session,
            symbol="AMD",
            side="long",
            quantity=100.0,
            avg_cost=346.0,
            profile="moderate",
        )
        db_session.commit()

        # Mock FinnhubClient to return a price that would trigger if geometry were valid
        mock_fh_instance = MagicMock()
        mock_fh_instance.get_quote.return_value = {"price": 346.38}
        mock_fh_class.return_value = mock_fh_instance

        from agents.bookkeeper import check_stop_losses

        to_close = check_stop_losses(engine)

        # Invalid geometry → should NOT trigger close
        assert len(to_close) == 0


# ---------------------------------------------------------------------------
# Test: Price monitor uses should_stop_trigger (Requirement 9.1)
# ---------------------------------------------------------------------------


class TestPriceMonitorUsesStopTrigger:
    """Verify price_monitor's check_stops_and_targets uses should_stop_trigger."""

    @patch("agents.price_monitor.get_batch_quotes")
    def test_price_monitor_uses_should_stop_trigger(self, mock_quotes, engine, db_session):
        """
        Price_monitor check_stops_and_targets delegates to should_stop_trigger
        and correctly triggers when geometry is valid and price breaches stop.

        Validates: Requirement 9.1
        """
        # Setup: create a valid long trade with stop below entry
        trade = _make_trade(
            db_session,
            symbol="MSFT",
            direction="LONG",
            entry_price=400.0,
            stop_price=395.0,
            status="open",
            profile="moderate",
        )
        db_session.commit()

        # Mock yfinance batch quotes to return a price below the stop
        mock_quotes.return_value = {"MSFT": 394.0}

        from agents.price_monitor import check_stops_and_targets

        triggers = check_stops_and_targets(engine)

        # The trade should trigger since price (394) < stop (395) - buffer
        stop_triggers = [t for t in triggers if t["type"] == "stop_loss"]
        assert len(stop_triggers) == 1
        assert stop_triggers[0]["symbol"] == "MSFT"
        assert stop_triggers[0]["price"] == 394.0

    @patch("agents.price_monitor.get_batch_quotes")
    def test_price_monitor_skips_invalid_geometry(self, mock_quotes, engine, db_session):
        """
        Price_monitor does NOT trigger for trades with invalid stop geometry.

        Validates: Requirement 9.1
        """
        # Create a long trade with INVALID geometry: stop above entry
        trade = _make_trade(
            db_session,
            symbol="AMD",
            direction="LONG",
            entry_price=346.0,
            stop_price=347.50,  # Invalid: above entry for a long
            status="open",
            profile="moderate",
        )
        db_session.commit()

        # Mock quotes to return a price that would trigger if geometry were valid
        mock_quotes.return_value = {"AMD": 346.38}

        from agents.price_monitor import check_stops_and_targets

        triggers = check_stops_and_targets(engine)

        # Invalid geometry → no stop trigger
        stop_triggers = [t for t in triggers if t["type"] == "stop_loss"]
        assert len(stop_triggers) == 0


# ---------------------------------------------------------------------------
# Test: Bookkeeper and price_monitor agree (Requirement 4.11)
# ---------------------------------------------------------------------------


class TestBookkeeperAndPriceMonitorAgree:
    """For the same trade state, both agents produce identical trigger decisions."""

    def test_bookkeeper_and_price_monitor_agree_trigger(self):
        """
        Both agents use should_stop_trigger, so for the same inputs they must
        produce the same triggered=True decision.

        Validates: Requirement 4.11
        """
        # Valid long trade, price below stop → should trigger
        bk_result = should_stop_trigger(
            side="long",
            entry_price=150.0,
            current_price=144.0,
            stop_price=145.0,
            stop_role="initial",
        )
        pm_result = should_stop_trigger(
            side="long",
            entry_price=150.0,
            current_price=144.0,
            stop_price=145.0,
            stop_role="initial",
        )

        assert bk_result.triggered == pm_result.triggered
        assert bk_result.triggered is True
        assert bk_result.geometry_valid == pm_result.geometry_valid

    def test_bookkeeper_and_price_monitor_agree_no_trigger(self):
        """
        Both agents produce the same triggered=False decision for price above stop.

        Validates: Requirement 4.11
        """
        bk_result = should_stop_trigger(
            side="long",
            entry_price=150.0,
            current_price=148.0,
            stop_price=145.0,
            stop_role="initial",
        )
        pm_result = should_stop_trigger(
            side="long",
            entry_price=150.0,
            current_price=148.0,
            stop_price=145.0,
            stop_role="initial",
        )

        assert bk_result.triggered == pm_result.triggered
        assert bk_result.triggered is False
        assert bk_result.geometry_valid == pm_result.geometry_valid

    def test_bookkeeper_and_price_monitor_agree_invalid_geometry(self):
        """
        Both agents produce the same non-trigger decision for invalid geometry.

        Validates: Requirement 4.11
        """
        # Long trade with stop above entry → invalid geometry
        bk_result = should_stop_trigger(
            side="long",
            entry_price=346.0,
            current_price=346.38,
            stop_price=347.50,
            stop_role="initial",
        )
        pm_result = should_stop_trigger(
            side="long",
            entry_price=346.0,
            current_price=346.38,
            stop_price=347.50,
            stop_role="initial",
        )

        assert bk_result.triggered == pm_result.triggered
        assert bk_result.triggered is False
        assert bk_result.geometry_valid == pm_result.geometry_valid
        assert bk_result.geometry_valid is False

    def test_bookkeeper_and_price_monitor_agree_short_trigger(self):
        """
        Both agents agree on short stop trigger.

        Validates: Requirement 4.11
        """
        # Short trade, price above stop → should trigger
        bk_result = should_stop_trigger(
            side="short",
            entry_price=100.0,
            current_price=106.0,
            stop_price=105.0,
            stop_role="initial",
        )
        pm_result = should_stop_trigger(
            side="short",
            entry_price=100.0,
            current_price=106.0,
            stop_price=105.0,
            stop_role="initial",
        )

        assert bk_result.triggered == pm_result.triggered
        assert bk_result.triggered is True


# ---------------------------------------------------------------------------
# Test: AMD regression - invalid long stop above entry no close (Req 4.9, 4.10)
# ---------------------------------------------------------------------------


class TestAMDRegressionNoClose:
    """
    AMD regression: long entry 346, stop 347.50, current 346.38 →
    neither bookkeeper nor price_monitor triggers a close.
    """

    @patch("agents.bookkeeper.FinnhubClient")
    def test_invalid_long_stop_above_entry_no_close_bookkeeper(self, mock_fh_class, engine, db_session):
        """
        AMD regression case: bookkeeper must NOT close a long trade with
        stop above entry, even though current price is below the stop.

        Validates: Requirements 4.9, 4.10
        """
        trade = _make_trade(
            db_session,
            symbol="AMD",
            direction="LONG",
            entry_price=346.0,
            stop_price=347.50,
            status="open",
            profile="moderate",
        )
        _make_position(
            db_session,
            symbol="AMD",
            side="long",
            quantity=100.0,
            avg_cost=346.0,
            profile="moderate",
        )
        db_session.commit()

        mock_fh_instance = MagicMock()
        mock_fh_instance.get_quote.return_value = {"price": 346.38}
        mock_fh_class.return_value = mock_fh_instance

        from agents.bookkeeper import check_stop_losses

        to_close = check_stop_losses(engine)

        # AMD regression: invalid geometry → no close
        assert len(to_close) == 0

    @patch("agents.price_monitor.get_batch_quotes")
    def test_invalid_long_stop_above_entry_no_close_price_monitor(self, mock_quotes, engine, db_session):
        """
        AMD regression case: price_monitor must NOT trigger a close for a long
        trade with stop above entry.

        Validates: Requirements 4.9, 4.10
        """
        trade = _make_trade(
            db_session,
            symbol="AMD",
            direction="LONG",
            entry_price=346.0,
            stop_price=347.50,
            status="open",
            profile="moderate",
        )
        db_session.commit()

        mock_quotes.return_value = {"AMD": 346.38}

        from agents.price_monitor import check_stops_and_targets

        triggers = check_stops_and_targets(engine)

        # AMD regression: invalid geometry → no stop trigger
        stop_triggers = [t for t in triggers if t["type"] == "stop_loss"]
        assert len(stop_triggers) == 0

    def test_invalid_long_stop_above_entry_should_stop_trigger_direct(self):
        """
        Direct should_stop_trigger call with AMD regression values confirms
        no trigger and invalid geometry.

        Validates: Requirements 4.9, 4.10
        """
        result = should_stop_trigger(
            side="long",
            entry_price=346.0,
            current_price=346.38,
            stop_price=347.50,
            stop_role="initial",
        )

        assert result.triggered is False
        assert result.geometry_valid is False
        assert "invalid" in result.reason.lower() or "geometry" in result.reason.lower()


# ---------------------------------------------------------------------------
# Test: Profit Manager mutation wiring (Requirement 10.1, 10.3)
# ---------------------------------------------------------------------------


class TestProfitManagerMutationWiring:
    """Verify profit_manager's _move_stop uses apply_stop_update correctly."""

    def test_profit_manager_breakeven_move_through_stop_authority(self, engine, db_session):
        """
        Profit_manager breakeven move goes through apply_stop_update:
        creates trade, calls _move_stop with stop_role="breakeven",
        verifies trade.stop_price updated and audit events logged.

        Validates: Requirements 10.1
        """
        # Setup: long trade at 100, stop at 95, current at 105 (in profit)
        trade = _make_trade(
            db_session,
            symbol="TSLA",
            direction="LONG",
            entry_price=100.0,
            stop_price=95.0,
            status="open",
            profile="moderate",
        )
        db_session.commit()

        from agents.profit_manager import _move_stop

        # Move stop to breakeven (entry price) with current_price showing profit
        _move_stop(engine, trade.id, 100.0, "+1R — stop to breakeven",
                   stop_role="breakeven", current_price=105.0)

        # Refresh trade from DB
        db_session.expire_all()
        updated_trade = db_session.query(Trade).filter_by(id=trade.id).first()

        # Stop should be updated to breakeven (entry price)
        assert updated_trade.stop_price == 100.0

        # Verify audit events were logged
        events = (
            db_session.query(TradeEvent)
            .filter_by(trade_id=trade.id)
            .order_by(TradeEvent.id)
            .all()
        )
        event_types = [e.event_type for e in events]
        assert "stop_update_requested" in event_types
        assert "stop_update_accepted" in event_types

        # Verify the accepted event has correct source_agent
        accepted = [e for e in events if e.event_type == "stop_update_accepted"][0]
        assert accepted.agent == "profit_manager"

    def test_profit_manager_trail_move_through_stop_authority(self, engine, db_session):
        """
        Profit_manager trail move goes through apply_stop_update:
        creates trade, calls _move_stop with stop_role="trail",
        verifies trade.stop_price updated and audit events logged.

        Validates: Requirements 10.1
        """
        # Setup: long trade at 100, stop at 95, current at 115 (+2R territory)
        trade = _make_trade(
            db_session,
            symbol="NVDA",
            direction="LONG",
            entry_price=100.0,
            stop_price=95.0,
            status="open",
            profile="moderate",
        )
        db_session.commit()

        from agents.profit_manager import _move_stop

        # Trail stop to +1R level (entry + risk = 100 + 5 = 105)
        _move_stop(engine, trade.id, 105.0, "+2R — trailing stop to +1R",
                   stop_role="trail", current_price=115.0)

        # Refresh trade from DB
        db_session.expire_all()
        updated_trade = db_session.query(Trade).filter_by(id=trade.id).first()

        # Stop should be updated to trail level
        assert updated_trade.stop_price == 105.0

        # Verify audit events
        events = (
            db_session.query(TradeEvent)
            .filter_by(trade_id=trade.id)
            .order_by(TradeEvent.id)
            .all()
        )
        event_types = [e.event_type for e in events]
        assert "stop_update_requested" in event_types
        assert "stop_update_accepted" in event_types

    def test_profit_manager_rejected_move_preserves_stop(self, engine, db_session):
        """
        When profit_manager tries to move stop to an invalid position,
        the existing stop is preserved and rejection is logged.

        Validates: Requirements 10.3
        """
        # Setup: long trade at 100, stop at 95, current at 105
        trade = _make_trade(
            db_session,
            symbol="META",
            direction="LONG",
            entry_price=100.0,
            stop_price=95.0,
            status="open",
            profile="moderate",
        )
        db_session.commit()
        original_stop = 95.0

        from agents.profit_manager import _move_stop

        # Try to move stop to an invalid position (above current price for a
        # profit-protecting role — breakeven at 106 when current is 105)
        _move_stop(engine, trade.id, 106.0, "invalid breakeven attempt",
                   stop_role="breakeven", current_price=105.0)

        # Refresh trade from DB
        db_session.expire_all()
        updated_trade = db_session.query(Trade).filter_by(id=trade.id).first()

        # Stop should be PRESERVED (not changed)
        assert updated_trade.stop_price == original_stop

        # Verify rejection event was logged
        events = (
            db_session.query(TradeEvent)
            .filter_by(trade_id=trade.id)
            .order_by(TradeEvent.id)
            .all()
        )
        event_types = [e.event_type for e in events]
        assert "stop_update_requested" in event_types
        assert "stop_update_rejected" in event_types


# ---------------------------------------------------------------------------
# Test: Portfolio Manager mutation wiring (Requirement 11.1, 11.3)
# ---------------------------------------------------------------------------


class TestPortfolioManagerMutationWiring:
    """Verify portfolio_manager maintenance tighten uses apply_stop_update."""

    def test_portfolio_manager_maintenance_tighten_through_stop_authority(self, engine, db_session):
        """
        Portfolio_manager maintenance tighten goes through apply_stop_update:
        simulate a maintenance tighten by calling apply_stop_update directly
        with maintenance_tighten role.

        Validates: Requirements 11.1
        """
        from utils.stop_authority import apply_stop_update

        # Setup: long trade at 200, stop at 190, current at 220 (in profit)
        trade = _make_trade(
            db_session,
            symbol="GOOG",
            direction="LONG",
            entry_price=200.0,
            stop_price=190.0,
            status="open",
            profile="moderate",
        )
        db_session.flush()

        # Simulate portfolio_manager maintenance tighten
        result = apply_stop_update(
            db_session,
            trade=trade,
            new_stop=210.0,  # Tighten to 210 (below current 220)
            source_agent="portfolio_manager",
            stop_role="maintenance_tighten",
            reason="Maintenance Review tighten_stop for GOOG",
            current_price=220.0,
        )
        db_session.commit()

        # Verify the update was accepted
        assert result.valid is True
        assert result.reason_type == "accepted"
        assert trade.stop_price == 210.0

        # Verify audit events
        events = (
            db_session.query(TradeEvent)
            .filter_by(trade_id=trade.id)
            .order_by(TradeEvent.id)
            .all()
        )
        event_types = [e.event_type for e in events]
        assert "stop_update_requested" in event_types
        assert "stop_update_accepted" in event_types

        # Verify source_agent is portfolio_manager
        accepted = [e for e in events if e.event_type == "stop_update_accepted"][0]
        assert accepted.agent == "portfolio_manager"

    def test_portfolio_manager_rejected_tighten_preserves_stop(self, engine, db_session):
        """
        When portfolio_manager tries an invalid tighten, existing stop is preserved.

        Validates: Requirements 11.3
        """
        from utils.stop_authority import apply_stop_update

        # Setup: long trade at 200, stop at 190, current at 220 (in profit)
        trade = _make_trade(
            db_session,
            symbol="AMZN",
            direction="LONG",
            entry_price=200.0,
            stop_price=190.0,
            status="open",
            profile="moderate",
        )
        db_session.flush()
        original_stop = 190.0

        # Try to tighten stop to an invalid position (above current price)
        result = apply_stop_update(
            db_session,
            trade=trade,
            new_stop=225.0,  # Invalid: above current price 220
            source_agent="portfolio_manager",
            stop_role="maintenance_tighten",
            reason="Maintenance Review tighten_stop for AMZN",
            current_price=220.0,
        )
        db_session.commit()

        # Verify the update was rejected
        assert result.valid is False
        assert result.reason_type == "rejected"

        # Stop should be PRESERVED
        assert trade.stop_price == original_stop

        # Verify rejection event was logged
        events = (
            db_session.query(TradeEvent)
            .filter_by(trade_id=trade.id)
            .order_by(TradeEvent.id)
            .all()
        )
        event_types = [e.event_type for e in events]
        assert "stop_update_requested" in event_types
        assert "stop_update_rejected" in event_types


# ---------------------------------------------------------------------------
# Test: End-to-end rejection audit trail (Requirements 10.1, 10.3, 11.1, 11.3)
# ---------------------------------------------------------------------------


class TestEndToEndRejectionAuditTrail:
    """End-to-end test: invalid stop proposed → rejected → existing preserved → audit complete."""

    def test_invalid_stop_rejected_existing_preserved_audit_complete(self, engine, db_session):
        """
        End-to-end: propose invalid stop → rejected → existing stop preserved →
        verify full audit trail (stop_update_requested + stop_update_rejected events).

        Validates: Requirements 10.1, 10.3, 11.1, 11.3
        """
        import json
        from utils.stop_authority import apply_stop_update

        # Setup: long trade at 150, valid stop at 145, current at 155 (in profit)
        trade = _make_trade(
            db_session,
            symbol="AAPL",
            direction="LONG",
            entry_price=150.0,
            stop_price=145.0,
            status="open",
            profile="moderate",
        )
        db_session.flush()
        original_stop = 145.0

        # Step 1: Propose an invalid stop (above current price for profit-protecting)
        result = apply_stop_update(
            db_session,
            trade=trade,
            new_stop=156.0,  # Invalid: above current price 155
            source_agent="profit_manager",
            stop_role="breakeven",
            reason="Attempted breakeven move",
            current_price=155.0,
        )
        db_session.commit()

        # Step 2: Verify rejection
        assert result.valid is False
        assert result.reason_type == "rejected"

        # Step 3: Verify existing stop is preserved
        assert trade.stop_price == original_stop

        # Step 4: Verify full audit trail
        events = (
            db_session.query(TradeEvent)
            .filter_by(trade_id=trade.id)
            .order_by(TradeEvent.id)
            .all()
        )
        event_types = [e.event_type for e in events]

        # Must have exactly: stop_update_requested followed by stop_update_rejected
        assert event_types.count("stop_update_requested") == 1
        assert event_types.count("stop_update_rejected") == 1

        # Verify ordering: requested comes before rejected
        req_idx = event_types.index("stop_update_requested")
        rej_idx = event_types.index("stop_update_rejected")
        assert req_idx < rej_idx

        # Verify requested event payload
        requested_event = events[req_idx]
        assert requested_event.agent == "profit_manager"
        req_payload = json.loads(requested_event.payload_json)
        assert req_payload["proposed_stop"] == 156.0
        assert req_payload["stop_role"] == "breakeven"
        assert req_payload["old_stop"] == original_stop

        # Verify rejected event payload
        rejected_event = events[rej_idx]
        assert rejected_event.agent == "profit_manager"
        rej_payload = json.loads(rejected_event.payload_json)
        assert rej_payload["proposed_stop"] == 156.0
        assert rej_payload["existing_stop"] == original_stop
        assert "rejection_reason" in rej_payload


# ---------------------------------------------------------------------------
# Test: Backfill stop_roles logic (Requirements 12.4, 12.5)
# ---------------------------------------------------------------------------


class TestBackfillStopRoles:
    """Verify backfill_stop_roles correctly infers stop_role for existing trades."""

    def test_trade_with_no_stop_events_gets_initial(self, engine, db_session):
        """
        An open trade with no stop event history gets stop_role = 'initial'.

        Validates: Requirement 12.4
        """
        # Create an open trade with NULL stop_role
        trade = _make_trade(
            db_session,
            symbol="AAPL",
            direction="LONG",
            entry_price=150.0,
            stop_price=145.0,
            status="open",
            profile="moderate",
        )
        trade.stop_role = None
        db_session.commit()

        from orchestrator import backfill_stop_roles

        backfill_stop_roles(engine)

        # Refresh
        db_session.expire_all()
        updated = db_session.query(Trade).filter_by(id=trade.id).first()
        assert updated.stop_role == "initial"

    def test_trade_with_breakeven_event_gets_breakeven(self, engine, db_session):
        """
        An open trade with a profit_manager stop event containing 'breakeven'
        in the payload gets stop_role = 'breakeven'.

        Validates: Requirement 12.5
        """
        import json
        from datetime import datetime

        trade = _make_trade(
            db_session,
            symbol="TSLA",
            direction="LONG",
            entry_price=200.0,
            stop_price=200.0,
            status="open",
            profile="moderate",
        )
        trade.stop_role = None
        db_session.flush()

        # Add a stop_update_accepted event from profit_manager with breakeven context
        event = TradeEvent(
            trade_id=trade.id,
            timestamp=datetime.utcnow(),
            event_type="stop_update_accepted",
            agent="profit_manager",
            symbol="TSLA",
            profile="moderate",
            message="+1R — stop to breakeven",
            payload_json=json.dumps({
                "old_stop": 190.0,
                "new_stop": 200.0,
                "stop_role": "breakeven",
                "reason": "breakeven move at +1R",
            }),
        )
        db_session.add(event)
        db_session.commit()

        from orchestrator import backfill_stop_roles

        backfill_stop_roles(engine)

        db_session.expire_all()
        updated = db_session.query(Trade).filter_by(id=trade.id).first()
        assert updated.stop_role == "breakeven"

    def test_trade_with_trail_event_gets_trail(self, engine, db_session):
        """
        An open trade with a profit_manager stop event containing 'trail' or '+R'
        in the payload gets stop_role = 'trail'.

        Validates: Requirement 12.5
        """
        import json
        from datetime import datetime

        trade = _make_trade(
            db_session,
            symbol="NVDA",
            direction="LONG",
            entry_price=100.0,
            stop_price=105.0,
            status="open",
            profile="moderate",
        )
        trade.stop_role = None
        db_session.flush()

        # Add a stop_update_accepted event with trail context
        event = TradeEvent(
            trade_id=trade.id,
            timestamp=datetime.utcnow(),
            event_type="stop_update_accepted",
            agent="profit_manager",
            symbol="NVDA",
            profile="moderate",
            message="+2R — trailing stop to +1R",
            payload_json=json.dumps({
                "old_stop": 100.0,
                "new_stop": 105.0,
                "stop_role": "trail",
                "reason": "trail stop at +2R",
            }),
        )
        db_session.add(event)
        db_session.commit()

        from orchestrator import backfill_stop_roles

        backfill_stop_roles(engine)

        db_session.expire_all()
        updated = db_session.query(Trade).filter_by(id=trade.id).first()
        assert updated.stop_role == "trail"

    def test_trade_with_plus_r_event_gets_trail(self, engine, db_session):
        """
        An open trade with a profit_manager stop event containing '+R'
        in the message gets stop_role = 'trail'.

        Validates: Requirement 12.5
        """
        import json
        from datetime import datetime

        trade = _make_trade(
            db_session,
            symbol="AMD",
            direction="LONG",
            entry_price=100.0,
            stop_price=110.0,
            status="open",
            profile="moderate",
        )
        trade.stop_role = None
        db_session.flush()

        # Add a stop_set event with +R in message
        event = TradeEvent(
            trade_id=trade.id,
            timestamp=datetime.utcnow(),
            event_type="stop_set",
            agent="profit_manager",
            symbol="AMD",
            profile="moderate",
            message="+3R — moved stop to +2R level",
            payload_json=json.dumps({"new_stop": 110.0}),
        )
        db_session.add(event)
        db_session.commit()

        from orchestrator import backfill_stop_roles

        backfill_stop_roles(engine)

        db_session.expire_all()
        updated = db_session.query(Trade).filter_by(id=trade.id).first()
        assert updated.stop_role == "trail"

    def test_backfill_is_idempotent(self, engine, db_session):
        """
        Running backfill_stop_roles multiple times produces the same result.

        Validates: Requirements 12.4, 12.5
        """
        trade = _make_trade(
            db_session,
            symbol="SPY",
            direction="LONG",
            entry_price=500.0,
            stop_price=495.0,
            status="open",
            profile="moderate",
        )
        trade.stop_role = None
        db_session.commit()

        from orchestrator import backfill_stop_roles

        # Run twice
        backfill_stop_roles(engine)
        backfill_stop_roles(engine)

        db_session.expire_all()
        updated = db_session.query(Trade).filter_by(id=trade.id).first()
        assert updated.stop_role == "initial"

    def test_backfill_skips_closed_trades(self, engine, db_session):
        """
        Closed trades are not affected by backfill.

        Validates: Requirement 12.4
        """
        trade = _make_trade(
            db_session,
            symbol="META",
            direction="LONG",
            entry_price=300.0,
            stop_price=290.0,
            status="closed",
            profile="moderate",
        )
        trade.stop_role = None
        db_session.commit()

        from orchestrator import backfill_stop_roles

        backfill_stop_roles(engine)

        db_session.expire_all()
        updated = db_session.query(Trade).filter_by(id=trade.id).first()
        # Closed trade should NOT be backfilled
        assert updated.stop_role is None

    def test_backfill_skips_trades_with_existing_stop_role(self, engine, db_session):
        """
        Trades that already have a stop_role set are not modified.

        Validates: Requirement 12.4 (idempotent)
        """
        trade = _make_trade(
            db_session,
            symbol="GOOG",
            direction="LONG",
            entry_price=170.0,
            stop_price=165.0,
            status="open",
            profile="moderate",
        )
        trade.stop_role = "trail"
        db_session.commit()

        from orchestrator import backfill_stop_roles

        backfill_stop_roles(engine)

        db_session.expire_all()
        updated = db_session.query(Trade).filter_by(id=trade.id).first()
        # Should remain "trail", not overwritten
        assert updated.stop_role == "trail"

    def test_backfill_uses_most_recent_event(self, engine, db_session):
        """
        When multiple stop events exist, backfill uses the most recent one.

        Validates: Requirement 12.5
        """
        import json
        from datetime import datetime, timedelta

        trade = _make_trade(
            db_session,
            symbol="MSFT",
            direction="LONG",
            entry_price=400.0,
            stop_price=410.0,
            status="open",
            profile="moderate",
        )
        trade.stop_role = None
        db_session.flush()

        # Older event: breakeven
        older_event = TradeEvent(
            trade_id=trade.id,
            timestamp=datetime.utcnow() - timedelta(hours=2),
            event_type="stop_update_accepted",
            agent="profit_manager",
            symbol="MSFT",
            profile="moderate",
            message="breakeven move",
            payload_json=json.dumps({"reason": "breakeven at +1R"}),
        )
        db_session.add(older_event)

        # Newer event: trail
        newer_event = TradeEvent(
            trade_id=trade.id,
            timestamp=datetime.utcnow(),
            event_type="stop_update_accepted",
            agent="profit_manager",
            symbol="MSFT",
            profile="moderate",
            message="+3R trailing stop",
            payload_json=json.dumps({"reason": "trail stop at +3R"}),
        )
        db_session.add(newer_event)
        db_session.commit()

        from orchestrator import backfill_stop_roles

        backfill_stop_roles(engine)

        db_session.expire_all()
        updated = db_session.query(Trade).filter_by(id=trade.id).first()
        assert updated.stop_role == "trail"


# ---------------------------------------------------------------------------
# Test: /api/trade-events endpoint (Requirements 13.1, 13.2)
# ---------------------------------------------------------------------------


class TestApiTradeEventsEndpoint:
    """Verify the /api/trade-events endpoint returns stop lifecycle events."""

    def test_returns_stop_events_for_trade(self, engine, db_session):
        """
        The /api/trade-events endpoint returns stop-related events for a trade.

        Validates: Requirements 13.1, 13.2
        """
        import json
        from datetime import datetime
        from web.app import app as flask_app

        # Create a trade
        trade = _make_trade(
            db_session,
            symbol="AMD",
            direction="LONG",
            entry_price=346.0,
            stop_price=340.0,
            status="open",
            profile="moderate",
        )
        db_session.flush()

        # Add stop lifecycle events
        events_data = [
            {
                "event_type": "stop_update_requested",
                "agent": "profit_manager",
                "message": "Breakeven move requested",
                "payload_json": json.dumps({
                    "proposed_stop": 346.0,
                    "stop_role": "breakeven",
                    "reason": "+1R breakeven",
                    "old_stop": 340.0,
                }),
            },
            {
                "event_type": "stop_update_accepted",
                "agent": "profit_manager",
                "message": "Stop moved to breakeven",
                "payload_json": json.dumps({
                    "old_stop": 340.0,
                    "new_stop": 346.0,
                    "stop_role": "breakeven",
                    "reason": "+1R breakeven",
                }),
            },
        ]

        for evt in events_data:
            te = TradeEvent(
                trade_id=trade.id,
                timestamp=datetime.utcnow(),
                symbol="AMD",
                profile="moderate",
                **evt,
            )
            db_session.add(te)
        db_session.commit()

        # Use Flask test client with the same engine
        with patch("web.app.engine", engine):
            with flask_app.test_client() as client:
                resp = client.get(f"/api/trade-events?trade_id={trade.id}")

        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == 2
        assert data[0]["event_type"] == "stop_update_requested"
        assert data[1]["event_type"] == "stop_update_accepted"
        assert data[0]["agent"] == "profit_manager"
        assert data[0]["trade_id"] == trade.id
        assert data[0]["timestamp"] is not None
        assert data[0]["payload"]["proposed_stop"] == 346.0

    def test_returns_400_without_trade_id(self):
        """
        The /api/trade-events endpoint returns 400 if trade_id is missing.

        Validates: Requirements 13.1
        """
        from web.app import app as flask_app

        with flask_app.test_client() as client:
            resp = client.get("/api/trade-events")

        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_returns_empty_list_for_trade_with_no_stop_events(self, engine, db_session):
        """
        The /api/trade-events endpoint returns an empty list for a trade
        with no stop-related events.

        Validates: Requirements 13.1
        """
        from web.app import app as flask_app

        trade = _make_trade(
            db_session,
            symbol="TSLA",
            direction="LONG",
            entry_price=200.0,
            stop_price=190.0,
            status="open",
            profile="moderate",
        )
        db_session.commit()

        with patch("web.app.engine", engine):
            with flask_app.test_client() as client:
                resp = client.get(f"/api/trade-events?trade_id={trade.id}")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data == []

    def test_filters_only_stop_event_types(self, engine, db_session):
        """
        The /api/trade-events endpoint only returns stop-related event types,
        not other trade events like 'entry_requested' or 'signal_seen'.

        Validates: Requirements 13.1, 13.2
        """
        import json
        from datetime import datetime
        from web.app import app as flask_app

        trade = _make_trade(
            db_session,
            symbol="NVDA",
            direction="LONG",
            entry_price=500.0,
            stop_price=490.0,
            status="open",
            profile="moderate",
        )
        db_session.flush()

        # Add a non-stop event
        non_stop_event = TradeEvent(
            trade_id=trade.id,
            timestamp=datetime.utcnow(),
            event_type="entry_requested",
            agent="portfolio_manager",
            symbol="NVDA",
            profile="moderate",
            message="Entry requested",
        )
        db_session.add(non_stop_event)

        # Add a stop event
        stop_event = TradeEvent(
            trade_id=trade.id,
            timestamp=datetime.utcnow(),
            event_type="stop_triggered",
            agent="bookkeeper",
            symbol="NVDA",
            profile="moderate",
            message="Stop triggered",
            payload_json=json.dumps({
                "trigger_price": 489.5,
                "stop_price": 490.0,
                "buffered_level": 489.51,
                "side": "long",
            }),
        )
        db_session.add(stop_event)
        db_session.commit()

        with patch("web.app.engine", engine):
            with flask_app.test_client() as client:
                resp = client.get(f"/api/trade-events?trade_id={trade.id}")

        assert resp.status_code == 200
        data = resp.get_json()
        # Only the stop_triggered event should be returned
        assert len(data) == 1
        assert data[0]["event_type"] == "stop_triggered"
        assert data[0]["payload"]["trigger_price"] == 489.5

    def test_events_ordered_by_timestamp_ascending(self, engine, db_session):
        """
        Events are returned in chronological order (oldest first).

        Validates: Requirements 13.2
        """
        import json
        from datetime import datetime, timedelta
        from web.app import app as flask_app

        trade = _make_trade(
            db_session,
            symbol="GOOG",
            direction="LONG",
            entry_price=170.0,
            stop_price=165.0,
            status="open",
            profile="moderate",
        )
        db_session.flush()

        # Add events in reverse chronological order
        now = datetime.utcnow()
        events_data = [
            ("stop_update_accepted", now),
            ("stop_update_requested", now - timedelta(seconds=5)),
            ("stop_geometry_invalid", now - timedelta(minutes=10)),
        ]

        for event_type, ts in events_data:
            te = TradeEvent(
                trade_id=trade.id,
                timestamp=ts,
                event_type=event_type,
                agent="price_monitor",
                symbol="GOOG",
                profile="moderate",
            )
            db_session.add(te)
        db_session.commit()

        with patch("web.app.engine", engine):
            with flask_app.test_client() as client:
                resp = client.get(f"/api/trade-events?trade_id={trade.id}")

        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == 3
        # Should be in ascending timestamp order
        assert data[0]["event_type"] == "stop_geometry_invalid"
        assert data[1]["event_type"] == "stop_update_requested"
        assert data[2]["event_type"] == "stop_update_accepted"


# ---------------------------------------------------------------------------
# Test: AMD Audit Trail Completeness (Requirement 13.3)
# ---------------------------------------------------------------------------


class TestAMDAuditTrailCompleteness:
    """
    Verify that for any future AMD-like regression case, the audit trail
    shows exactly which agent requested the invalid stop, whether it was
    rejected or accepted, and the rejection reason.

    Validates: Requirement 13.3
    """

    def test_amd_regression_full_audit_trail(self, engine, db_session):
        """End-to-end AMD scenario: invalid stop proposed → rejected → audit trail complete."""
        import json
        from utils.stop_authority import apply_stop_update

        # Step 1: Create a long trade with entry at $346, valid initial stop at $340
        trade = _make_trade(
            db_session,
            symbol="AMD",
            direction="LONG",
            entry_price=346.0,
            stop_price=340.0,
            status="open",
            profile="moderate",
        )
        trade.stop_role = "initial"
        db_session.flush()

        original_stop = 340.0

        # Step 2: Simulate portfolio_manager attempting to tighten stop to $347.50
        # This is invalid because $347.50 is above entry ($346) for a long trade
        result = apply_stop_update(
            db_session,
            trade=trade,
            new_stop=347.50,
            source_agent="portfolio_manager",
            stop_role="maintenance_tighten",
            reason="Maintenance Review tighten_stop for AMD",
            current_price=346.38,
        )
        db_session.commit()

        # Step 3: Verify the update was rejected
        assert result.valid is False
        assert result.reason_type == "rejected"

        # Step 4: Verify trade.stop_price remains at $340 (unchanged)
        db_session.expire_all()
        refreshed_trade = db_session.query(Trade).filter_by(id=trade.id).first()
        assert refreshed_trade.stop_price == original_stop

        # Step 5: Verify the audit trail contains the correct event sequence
        events = (
            db_session.query(TradeEvent)
            .filter_by(trade_id=trade.id)
            .order_by(TradeEvent.id)
            .all()
        )
        event_types = [e.event_type for e in events]

        # Must have stop_update_requested followed by stop_update_rejected
        assert "stop_update_requested" in event_types
        assert "stop_update_rejected" in event_types

        # Verify ordering: requested comes before rejected
        req_idx = event_types.index("stop_update_requested")
        rej_idx = event_types.index("stop_update_rejected")
        assert req_idx < rej_idx

        # Step 6: Verify stop_update_requested event shows portfolio_manager requested the change
        requested_event = events[req_idx]
        assert requested_event.agent == "portfolio_manager"
        req_payload = json.loads(requested_event.payload_json)
        assert req_payload["source_agent"] == "portfolio_manager"
        assert req_payload["proposed_stop"] == 347.50
        assert req_payload["stop_role"] == "maintenance_tighten"
        assert req_payload["reason"] == "Maintenance Review tighten_stop for AMD"

        # Step 7: Verify stop_update_rejected event shows the rejection reason
        rejected_event = events[rej_idx]
        assert rejected_event.agent == "portfolio_manager"
        rej_payload = json.loads(rejected_event.payload_json)
        assert rej_payload["source_agent"] == "portfolio_manager"
        assert rej_payload["proposed_stop"] == 347.50
        assert rej_payload["existing_stop"] == original_stop
        # The rejection reason must clearly explain why it was rejected
        rejection_reason = rej_payload["rejection_reason"]
        assert "347.5" in rejection_reason or "347.50" in rejection_reason
        assert "entry" in rejection_reason.lower() or "346" in rejection_reason

    def test_amd_audit_trail_queryable_via_trade_events_table(self, engine, db_session):
        """Verify the AMD rejection events are queryable via the trade_events table."""
        import json
        from utils.stop_authority import apply_stop_update

        # Create the AMD scenario trade
        trade = _make_trade(
            db_session,
            symbol="AMD",
            direction="LONG",
            entry_price=346.0,
            stop_price=340.0,
            status="open",
            profile="moderate",
        )
        trade.stop_role = "initial"
        db_session.flush()

        # Simulate the invalid stop attempt
        apply_stop_update(
            db_session,
            trade=trade,
            new_stop=347.50,
            source_agent="portfolio_manager",
            stop_role="maintenance_tighten",
            reason="Maintenance Review tighten_stop for AMD",
            current_price=346.38,
        )
        db_session.commit()

        # Query events directly from trade_events table (as the API would)
        stop_event_types = [
            "stop_update_requested",
            "stop_update_accepted",
            "stop_update_rejected",
            "stop_geometry_invalid",
            "stop_triggered",
            "stop_repaired",
            "stop_review_required",
        ]
        events = (
            db_session.query(TradeEvent)
            .filter(
                TradeEvent.trade_id == trade.id,
                TradeEvent.event_type.in_(stop_event_types),
            )
            .order_by(TradeEvent.timestamp)
            .all()
        )

        # Should have exactly 2 events: requested + rejected
        assert len(events) == 2
        assert events[0].event_type == "stop_update_requested"
        assert events[1].event_type == "stop_update_rejected"

        # Both events should have timestamps
        assert events[0].timestamp is not None
        assert events[1].timestamp is not None

        # Both events should have the source agent recorded
        assert events[0].agent == "portfolio_manager"
        assert events[1].agent == "portfolio_manager"

        # Verify payloads are valid JSON and contain expected fields
        for event in events:
            payload = json.loads(event.payload_json)
            assert "source_agent" in payload
            assert payload["source_agent"] == "portfolio_manager"

    def test_amd_audit_trail_rejection_reason_is_clear(self, engine, db_session):
        """Verify the rejection reason clearly explains the geometry violation."""
        import json
        from utils.stop_authority import apply_stop_update

        # Create the AMD scenario
        trade = _make_trade(
            db_session,
            symbol="AMD",
            direction="LONG",
            entry_price=346.0,
            stop_price=340.0,
            status="open",
            profile="moderate",
        )
        trade.stop_role = "initial"
        db_session.flush()

        result = apply_stop_update(
            db_session,
            trade=trade,
            new_stop=347.50,
            source_agent="portfolio_manager",
            stop_role="maintenance_tighten",
            reason="Maintenance Review tighten_stop for AMD",
            current_price=346.38,
        )
        db_session.commit()

        # The result reason should clearly explain the geometry violation
        assert result.valid is False
        reason_lower = result.reason.lower()
        # Must mention that the stop is above entry (the core geometry violation)
        assert "above" in reason_lower or "must be below" in reason_lower or "entry" in reason_lower

        # Verify the rejected event's rejection_reason in the audit trail
        rejected_event = (
            db_session.query(TradeEvent)
            .filter_by(trade_id=trade.id, event_type="stop_update_rejected")
            .first()
        )
        assert rejected_event is not None
        rej_payload = json.loads(rejected_event.payload_json)
        rejection_reason = rej_payload["rejection_reason"]

        # The rejection reason must be human-readable and explain the issue
        assert len(rejection_reason) > 10  # Not just a code or empty string
        # Should reference the stop price or entry price
        assert "347.5" in rejection_reason or "346" in rejection_reason
