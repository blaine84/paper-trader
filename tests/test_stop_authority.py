"""
Unit tests for StopAuthority core logic.

Validates:
- AMD regression: long entry 346, stop 347.50, current 346.38 → rejected (Req 2.5)
- Valid trailing: long entry 346, current 350, stop 347.50 → accepted (Req 3.6)
- Short breakeven: entry 492.80, current 491.98, stop 492.80 → accepted (Req 3.7)
- Transaction semantics: apply_stop_update flushes but does not commit (Req 1.3)
- Buffer computation: 0.10% of various prices (Req 3.5)
- Role metadata persistence: stop_role/stop_updated_by/stop_updated_at set (Req 1.4)
- Input validation: ValueError for invalid side, non-positive prices, invalid role
- maintenance_tighten conditional rules
"""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, Trade, TradeEvent
from utils.stop_authority import (
    validate_stop_geometry,
    apply_stop_update,
    should_stop_trigger,
    StopValidationResult,
    StopTriggerResult,
    _compute_buffer,
)


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
        "entry_price": 346.00,
        "stop_price": 340.00,
        "target_price": 360.00,
        "status": "open",
        "profile": "moderate",
    }
    defaults.update(overrides)
    trade = Trade(**defaults)
    db_session.add(trade)
    db_session.flush()
    return trade


# ---------------------------------------------------------------------------
# Test: AMD Regression (Requirement 2.5)
# ---------------------------------------------------------------------------


class TestAMDRegression:
    """AMD regression: long entry 346, stop 347.50, current 346.38 → rejected."""

    def test_amd_regression_long_stop_above_entry_rejected(self):
        """A long trade with stop above entry must be rejected as geometrically invalid."""
        result = validate_stop_geometry(
            side="long",
            entry_price=346.00,
            current_price=346.38,
            stop_price=347.50,
            stop_role="initial",
        )
        assert result.valid is False
        assert result.reason_type in ("rejected", "repair")
        assert "346" in result.reason or "347.5" in result.reason

    def test_amd_regression_trigger_does_not_fire(self):
        """should_stop_trigger must NOT trigger on a geometrically invalid stop."""
        result = should_stop_trigger(
            side="long",
            entry_price=346.00,
            current_price=346.38,
            stop_price=347.50,
            stop_role="initial",
        )
        assert result.triggered is False
        assert result.geometry_valid is False


# ---------------------------------------------------------------------------
# Test: Valid Trailing Stop (Requirement 3.6)
# ---------------------------------------------------------------------------


class TestValidTrailingStop:
    """Valid trailing: long entry 346, current 350, stop 347.50 → accepted."""

    def test_valid_trailing_stop_long(self):
        """A trailing stop below current (in profit) should be accepted."""
        result = validate_stop_geometry(
            side="long",
            entry_price=346.00,
            current_price=350.00,
            stop_price=347.50,
            stop_role="trail",
        )
        assert result.valid is True
        assert result.reason_type == "accepted"

    def test_valid_breakeven_stop_long(self):
        """A breakeven stop at entry level (in profit) should be accepted."""
        # stop at 346 with current at 350 → buffer = 0.001 * 350 = 0.35
        # stop must be < 350 - 0.35 = 349.65 → 346 < 349.65 ✓
        result = validate_stop_geometry(
            side="long",
            entry_price=346.00,
            current_price=350.00,
            stop_price=346.00,
            stop_role="breakeven",
        )
        assert result.valid is True
        assert result.reason_type == "accepted"


# ---------------------------------------------------------------------------
# Test: Short Breakeven (Requirement 3.7)
# ---------------------------------------------------------------------------


class TestShortBreakeven:
    """Short breakeven: entry 492.80, current 491.98, stop 492.80 → accepted if buffer ok."""

    def test_short_breakeven_accepted(self):
        """Short breakeven stop at entry should be accepted when in profit with buffer."""
        # Short: in profit when current < entry → 491.98 < 492.80 ✓
        # Buffer = 0.001 * 491.98 = 0.49198
        # Stop must be > current + buffer = 491.98 + 0.49198 = 492.47198
        # Stop = 492.80 > 492.47198 ✓
        result = validate_stop_geometry(
            side="short",
            entry_price=492.80,
            current_price=491.98,
            stop_price=492.80,
            stop_role="breakeven",
        )
        assert result.valid is True
        assert result.reason_type == "accepted"

    def test_short_breakeven_rejected_when_not_in_profit(self):
        """Short breakeven stop rejected when trade is not in profit."""
        # Short: NOT in profit when current > entry → 493.50 > 492.80
        result = validate_stop_geometry(
            side="short",
            entry_price=492.80,
            current_price=493.50,
            stop_price=492.80,
            stop_role="breakeven",
        )
        assert result.valid is False


# ---------------------------------------------------------------------------
# Test: Transaction Semantics (Requirement 1.3)
# ---------------------------------------------------------------------------


class TestTransactionSemantics:
    """apply_stop_update flushes but does not commit."""

    def test_apply_stop_update_flush_not_commit(self, db_session):
        """apply_stop_update should flush (data visible in session) but not commit."""
        trade = _make_trade(db_session, entry_price=100.0, stop_price=95.0, direction="LONG")
        db_session.commit()  # commit the initial trade

        # Now apply a valid stop update
        result = apply_stop_update(
            db_session,
            trade=trade,
            new_stop=96.0,
            source_agent="profit_manager",
            stop_role="trail",
            reason="trailing stop move",
            current_price=110.0,
        )

        assert result.valid is True
        # The trade's stop_price should be updated in the session (flushed)
        assert trade.stop_price == 96.0

        # But if we rollback, the change should be lost (proving it wasn't committed)
        db_session.rollback()
        db_session.refresh(trade)
        # After rollback, stop_price should revert to original
        assert trade.stop_price == 95.0

    def test_apply_stop_update_events_flushed(self, db_session):
        """Events should be visible in session after flush (before commit)."""
        trade = _make_trade(db_session, entry_price=100.0, stop_price=95.0, direction="LONG")
        db_session.commit()

        apply_stop_update(
            db_session,
            trade=trade,
            new_stop=96.0,
            source_agent="profit_manager",
            stop_role="trail",
            reason="trailing stop move",
            current_price=110.0,
        )

        # Events should be queryable in the session (flushed)
        events = db_session.query(TradeEvent).filter_by(trade_id=trade.id).all()
        assert len(events) >= 1  # At least stop_update_requested


# ---------------------------------------------------------------------------
# Test: Buffer Computation (Requirement 3.5)
# ---------------------------------------------------------------------------


class TestBufferComputation:
    """Buffer computation: 0.10% of various prices."""

    def test_buffer_100(self):
        """0.10% of 100 = 0.10."""
        assert _compute_buffer(100.0, 0.001) == pytest.approx(0.10)

    def test_buffer_346(self):
        """0.10% of 346 = 0.346."""
        assert _compute_buffer(346.0, 0.001) == pytest.approx(0.346)

    def test_buffer_492_80(self):
        """0.10% of 492.80 = 0.4928."""
        assert _compute_buffer(492.80, 0.001) == pytest.approx(0.4928)

    def test_buffer_1000(self):
        """0.10% of 1000 = 1.0."""
        assert _compute_buffer(1000.0, 0.001) == pytest.approx(1.0)

    def test_buffer_50(self):
        """0.10% of 50 = 0.05."""
        assert _compute_buffer(50.0, 0.001) == pytest.approx(0.05)

    def test_buffer_custom_pct(self):
        """Custom buffer_pct of 0.5% of 200 = 1.0."""
        assert _compute_buffer(200.0, 0.005) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test: Role Metadata Persistence (Requirement 1.4, 5.1-5.7)
# ---------------------------------------------------------------------------


class TestRoleMetadataPersistence:
    """Verify stop_role/stop_updated_by/stop_updated_at are set after apply_stop_update."""

    def test_role_metadata_persistence(self, db_session):
        """apply_stop_update should set stop_role, stop_updated_by, stop_updated_at."""
        trade = _make_trade(db_session, entry_price=100.0, stop_price=95.0, direction="LONG")
        # Ensure metadata attributes exist on the trade object for testing
        # (apply_stop_update uses hasattr guards)
        trade.stop_role = None
        trade.stop_updated_by = None
        trade.stop_updated_at = None
        db_session.commit()

        before_time = datetime.utcnow()

        result = apply_stop_update(
            db_session,
            trade=trade,
            new_stop=96.0,
            source_agent="profit_manager",
            stop_role="trail",
            reason="trailing stop move",
            current_price=110.0,
        )

        assert result.valid is True
        # Verify metadata was set
        assert trade.stop_role == "trail"
        assert trade.stop_updated_by == "profit_manager"
        assert trade.stop_updated_at is not None
        assert trade.stop_updated_at >= before_time

    def test_breakeven_role_set(self, db_session):
        """Breakeven stop move sets stop_role to 'breakeven'."""
        trade = _make_trade(db_session, entry_price=100.0, stop_price=95.0, direction="LONG")
        trade.stop_role = None
        trade.stop_updated_by = None
        trade.stop_updated_at = None
        db_session.commit()

        result = apply_stop_update(
            db_session,
            trade=trade,
            new_stop=100.0,
            source_agent="profit_manager",
            stop_role="breakeven",
            reason="move to breakeven",
            current_price=110.0,
        )

        assert result.valid is True
        assert trade.stop_role == "breakeven"

    def test_maintenance_tighten_role_set(self, db_session):
        """Maintenance tighten sets stop_role to 'maintenance_tighten'."""
        trade = _make_trade(db_session, entry_price=100.0, stop_price=95.0, direction="LONG")
        trade.stop_role = None
        trade.stop_updated_by = None
        trade.stop_updated_at = None
        db_session.commit()

        result = apply_stop_update(
            db_session,
            trade=trade,
            new_stop=97.0,
            source_agent="portfolio_manager",
            stop_role="maintenance_tighten",
            reason="maintenance review tighten",
            current_price=110.0,
        )

        assert result.valid is True
        assert trade.stop_role == "maintenance_tighten"


# ---------------------------------------------------------------------------
# Test: Input Validation (ValueError cases)
# ---------------------------------------------------------------------------


class TestInputValidation:
    """ValueError for invalid side, non-positive prices, invalid role."""

    def test_invalid_side_raises_valueerror(self):
        """Invalid side value should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid side"):
            validate_stop_geometry(
                side="up",
                entry_price=100.0,
                current_price=105.0,
                stop_price=95.0,
                stop_role="initial",
            )

    def test_empty_side_raises_valueerror(self):
        """Empty side string should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid side"):
            validate_stop_geometry(
                side="",
                entry_price=100.0,
                current_price=105.0,
                stop_price=95.0,
                stop_role="initial",
            )

    def test_non_positive_entry_price_raises_valueerror(self):
        """Zero or negative entry_price should raise ValueError."""
        with pytest.raises(ValueError, match="entry_price must be positive"):
            validate_stop_geometry(
                side="long",
                entry_price=0.0,
                current_price=105.0,
                stop_price=95.0,
                stop_role="initial",
            )

    def test_negative_entry_price_raises_valueerror(self):
        """Negative entry_price should raise ValueError."""
        with pytest.raises(ValueError, match="entry_price must be positive"):
            validate_stop_geometry(
                side="long",
                entry_price=-10.0,
                current_price=105.0,
                stop_price=95.0,
                stop_role="initial",
            )

    def test_non_positive_stop_price_raises_valueerror(self):
        """Zero or negative stop_price should raise ValueError."""
        with pytest.raises(ValueError, match="stop_price must be positive"):
            validate_stop_geometry(
                side="long",
                entry_price=100.0,
                current_price=105.0,
                stop_price=0.0,
                stop_role="initial",
            )

    def test_negative_stop_price_raises_valueerror(self):
        """Negative stop_price should raise ValueError."""
        with pytest.raises(ValueError, match="stop_price must be positive"):
            validate_stop_geometry(
                side="long",
                entry_price=100.0,
                current_price=105.0,
                stop_price=-5.0,
                stop_role="initial",
            )

    def test_invalid_role_raises_valueerror(self):
        """Invalid stop_role should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid stop_role"):
            validate_stop_geometry(
                side="long",
                entry_price=100.0,
                current_price=105.0,
                stop_price=95.0,
                stop_role="invalid_role",
            )

    def test_unknown_role_raises_valueerror(self):
        """Unknown stop_role string should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid stop_role"):
            validate_stop_geometry(
                side="short",
                entry_price=100.0,
                current_price=95.0,
                stop_price=105.0,
                stop_role="aggressive",
            )


# ---------------------------------------------------------------------------
# Test: maintenance_tighten conditional rules
# ---------------------------------------------------------------------------


class TestMaintenanceTighten:
    """maintenance_tighten uses profit-protecting rules when in profit, protective otherwise."""

    def test_maintenance_tighten_in_profit_uses_profit_protecting_rules(self):
        """When in profit, maintenance_tighten uses profit-protecting rules."""
        # Long: entry 100, current 110 (in profit)
        # Stop at 105 → must be < current - buffer = 110 - 0.11 = 109.89
        # 105 < 109.89 ✓ → accepted
        result = validate_stop_geometry(
            side="long",
            entry_price=100.0,
            current_price=110.0,
            stop_price=105.0,
            stop_role="maintenance_tighten",
        )
        assert result.valid is True
        assert result.reason_type == "accepted"

    def test_maintenance_tighten_not_in_profit_uses_protective_rules(self):
        """When not in profit, maintenance_tighten uses protective rules."""
        # Long: entry 100, current 98 (not in profit)
        # Protective rules: stop must be < entry AND < current
        # Stop at 97 → 97 < 100 ✓ AND 97 < 98 ✓ → accepted
        result = validate_stop_geometry(
            side="long",
            entry_price=100.0,
            current_price=98.0,
            stop_price=97.0,
            stop_role="maintenance_tighten",
        )
        assert result.valid is True
        assert result.reason_type == "accepted"

    def test_maintenance_tighten_not_in_profit_rejects_stop_above_entry(self):
        """When not in profit, maintenance_tighten rejects stop above entry (protective rules)."""
        # Long: entry 100, current 98 (not in profit)
        # Stop at 101 → 101 >= 100 → rejected
        result = validate_stop_geometry(
            side="long",
            entry_price=100.0,
            current_price=98.0,
            stop_price=101.0,
            stop_role="maintenance_tighten",
        )
        assert result.valid is False

    def test_maintenance_tighten_short_in_profit(self):
        """Short maintenance_tighten in profit uses profit-protecting rules."""
        # Short: entry 100, current 95 (in profit)
        # Buffer = 0.001 * 95 = 0.095
        # Stop must be > current + buffer = 95 + 0.095 = 95.095
        # Stop at 96 → 96 > 95.095 ✓ → accepted
        result = validate_stop_geometry(
            side="short",
            entry_price=100.0,
            current_price=95.0,
            stop_price=96.0,
            stop_role="maintenance_tighten",
        )
        assert result.valid is True


# ---------------------------------------------------------------------------
# Test: apply_stop_update rejection preserves existing stop
# ---------------------------------------------------------------------------


class TestApplyStopUpdateRejection:
    """Invalid proposed stop does not overwrite valid existing stop."""

    def test_invalid_stop_preserves_existing(self, db_session):
        """An invalid proposed stop should not change trade.stop_price."""
        trade = _make_trade(
            db_session,
            entry_price=346.00,
            stop_price=340.00,
            direction="LONG",
        )
        db_session.commit()

        result = apply_stop_update(
            db_session,
            trade=trade,
            new_stop=347.50,  # Above entry → invalid for initial
            source_agent="portfolio_manager",
            stop_role="initial",
            reason="attempted invalid stop",
            current_price=346.38,
        )

        assert result.valid is False
        assert result.reason_type == "rejected"
        # Existing stop preserved
        assert trade.stop_price == 340.00

    def test_rejection_logs_events(self, db_session):
        """Rejected stop update should still log audit events."""
        trade = _make_trade(
            db_session,
            entry_price=346.00,
            stop_price=340.00,
            direction="LONG",
        )
        db_session.commit()

        apply_stop_update(
            db_session,
            trade=trade,
            new_stop=347.50,
            source_agent="portfolio_manager",
            stop_role="initial",
            reason="attempted invalid stop",
            current_price=346.38,
        )

        events = db_session.query(TradeEvent).filter_by(trade_id=trade.id).all()
        event_types = [e.event_type for e in events]
        assert "stop_update_requested" in event_types
        assert "stop_update_rejected" in event_types


# ---------------------------------------------------------------------------
# Test: StopValidationResult completeness (Requirement 1.4)
# ---------------------------------------------------------------------------


class TestValidationResultCompleteness:
    """Returned StopValidationResult always has required fields populated."""

    def test_accepted_result_fields(self):
        """Accepted result has all required fields."""
        result = validate_stop_geometry(
            side="long",
            entry_price=100.0,
            current_price=105.0,
            stop_price=95.0,
            stop_role="initial",
        )
        assert result.valid is not None
        assert result.reason_type is not None
        assert result.reason is not None
        assert result.side == "long"
        assert result.stop_role == "initial"

    def test_rejected_result_fields(self):
        """Rejected result has all required fields."""
        result = validate_stop_geometry(
            side="long",
            entry_price=100.0,
            current_price=105.0,
            stop_price=110.0,
            stop_role="initial",
        )
        assert result.valid is False
        assert result.reason_type in ("rejected", "repair")
        assert result.reason != ""
        assert result.side == "long"
        assert result.stop_role == "initial"

    def test_repair_result_has_repair_price(self):
        """When reason_type is 'repair', repair_price should be set."""
        result = validate_stop_geometry(
            side="long",
            entry_price=100.0,
            current_price=105.0,
            stop_price=110.0,
            stop_role="initial",
        )
        if result.reason_type == "repair":
            assert result.repair_price is not None
            assert result.repair_price > 0
