"""
Tests for pilot override trade event tagging (Task 9.2).

Verifies that when a pilot override trade executes at reduced size,
the entry_filled trade event is tagged with `pilot_override: true` and
`pilot_size_multiplier` in its payload.
"""

import json
from unittest.mock import patch, MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, Balance, TradeEvent
from agents.portfolio_manager import execute_trade


def _make_engine():
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    return engine


def _make_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()


def _seed_balance(db, profile_id: str, cash: float = 100_000.0):
    db.add(Balance(profile=profile_id, cash=cash))
    db.commit()


def _base_decision(action="BUY", symbol="AAPL", price=150.0,
                   stop=145.0, target=160.0, quantity=50):
    return {
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "price": price,
        "stop_loss": stop,
        "target": target,
        "rationale": "pilot override test",
        "setup_type": "gap_and_go",
        "market_regime": "risk_on",
    }


def _mock_finnhub():
    mock_fh = MagicMock()
    mock_fh.get_quote.return_value = {"price": 150.0}
    return mock_fh


def _gate_pipeline_with_pilot_override(reason_type="near_miss_pilot_override", size_multiplier=0.25):
    """Simulate gate pipeline returning notes indicating a pilot override."""
    notes = [
        {
            "gate": "setup_quality_gate",
            "decision": "reduce_size",
            "reason_type": reason_type,
            "reason": "Near-miss pilot override",
            "size_multiplier": size_multiplier,
            "confirming_signals": ["confidence_score >= 7.0"],
            "near_miss_margin": 0.05,
        }
    ]
    return True, notes, size_multiplier, [{"gate": "setup_quality_gate", "multiplier": size_multiplier}]


def _gate_pipeline_with_rr_pilot_override(size_multiplier=0.25):
    """Simulate gate pipeline returning notes indicating a risk geometry pilot override."""
    notes = [
        {
            "gate": "setup_quality_gate",
            "decision": "allow",
            "reason_type": "sufficient_data",
            "reason": "Allowed",
        },
        {
            "gate": "risk_geometry_gate",
            "decision": "adjusted_allowed",
            "reason_type": "pilot_rr_override",
            "reason": "Pilot R:R override",
            "pilot_override": True,
            "size_multiplier": size_multiplier,
            "entry_price": 150.0,
            "stop_price": 144.0,
            "target_price": 160.0,
            "quantity": 12,
        },
    ]
    return True, notes, size_multiplier, [{"gate": "risk_geometry_gate", "multiplier": size_multiplier}]


def _gate_pipeline_no_pilot():
    """Simulate gate pipeline passing with no pilot override."""
    notes = [
        {
            "gate": "setup_quality_gate",
            "decision": "allow",
            "reason_type": "sufficient_data",
            "reason": "Allowed",
        }
    ]
    return True, notes, 1.0, []


# Common mock targets
_GATE_PIPELINE = "agents.portfolio_manager._run_gate_pipeline"
_FIND_SIMILAR = "agents.portfolio_manager.find_similar_cases"
_COMPUTE_SIM_STATS = "agents.portfolio_manager.compute_similarity_stats"
_ADJUST_CONFIDENCE = "utils.trade_validator.adjust_confidence"
_VALIDATE_TRADE = "utils.trade_validator.validate_trade"
_CHECK_CORRELATION = "utils.trade_validator.check_correlation"

_CONF_OK = {"modifier": 1.0, "block": False, "reason": "ok", "win_rate": 0.60, "total_cases": 10}
_GOOD_SIM = {
    "similarity_winrate": 0.80, "similarity_avg_r": 2.5,
    "sample_size": 15, "similarity_confidence": 1.0,
}


def _get_entry_filled_events(db):
    """Return all entry_filled trade events from the database."""
    events = db.query(TradeEvent).filter_by(event_type="entry_filled").all()
    return [
        (e, json.loads(e.payload_json) if e.payload_json else {})
        for e in events
    ]


class TestPilotOverrideTradeTagging:
    """Task 9.2: Pilot override trade events tagged with pilot metadata."""

    def test_near_miss_pilot_override_tags_entry_filled_event(self):
        """When setup_quality_gate applies near-miss pilot override,
        the entry_filled event payload includes pilot_override and pilot_size_multiplier."""
        engine = _make_engine()
        db = _make_session(engine)
        _seed_balance(db, "moderate")

        decision = _base_decision()

        with (
            patch(_GATE_PIPELINE, return_value=_gate_pipeline_with_pilot_override()),
            patch(_FIND_SIMILAR, return_value=[]),
            patch(_COMPUTE_SIM_STATS, return_value=_GOOD_SIM),
            patch(_ADJUST_CONFIDENCE, return_value=_CONF_OK),
            patch(_VALIDATE_TRADE),
            patch(_CHECK_CORRELATION, return_value=""),
            patch("agents.portfolio_manager.FinnhubClient") as mock_fh_cls,
        ):
            mock_fh_cls.return_value = _mock_finnhub()
            ok, msg = execute_trade(db, decision, "moderate")

        assert ok is True, f"Trade should have succeeded: {msg}"

        events = _get_entry_filled_events(db)
        assert len(events) == 1
        _, payload = events[0]

        assert payload["pilot_override"] is True
        assert payload["pilot_size_multiplier"] == 0.25
        db.close()

    def test_rr_pilot_override_tags_entry_filled_event(self):
        """When risk_geometry_gate applies pilot R:R override,
        the entry_filled event payload includes pilot_override and pilot_size_multiplier."""
        engine = _make_engine()
        db = _make_session(engine)
        _seed_balance(db, "moderate")

        decision = _base_decision()

        with (
            patch(_GATE_PIPELINE, return_value=_gate_pipeline_with_rr_pilot_override()),
            patch(_FIND_SIMILAR, return_value=[]),
            patch(_COMPUTE_SIM_STATS, return_value=_GOOD_SIM),
            patch(_ADJUST_CONFIDENCE, return_value=_CONF_OK),
            patch(_VALIDATE_TRADE),
            patch(_CHECK_CORRELATION, return_value=""),
            patch("agents.portfolio_manager.FinnhubClient") as mock_fh_cls,
        ):
            mock_fh_cls.return_value = _mock_finnhub()
            ok, msg = execute_trade(db, decision, "moderate")

        assert ok is True, f"Trade should have succeeded: {msg}"

        events = _get_entry_filled_events(db)
        assert len(events) == 1
        _, payload = events[0]

        assert payload["pilot_override"] is True
        assert payload["pilot_size_multiplier"] == 0.25
        db.close()

    def test_no_pilot_override_does_not_tag_event(self):
        """When no pilot override occurs, entry_filled event has no pilot metadata."""
        engine = _make_engine()
        db = _make_session(engine)
        _seed_balance(db, "aggressive")

        decision = _base_decision()

        with (
            patch(_GATE_PIPELINE, return_value=_gate_pipeline_no_pilot()),
            patch(_FIND_SIMILAR, return_value=[]),
            patch(_COMPUTE_SIM_STATS, return_value=_GOOD_SIM),
            patch(_ADJUST_CONFIDENCE, return_value=_CONF_OK),
            patch(_VALIDATE_TRADE),
            patch(_CHECK_CORRELATION, return_value=""),
            patch("agents.portfolio_manager.FinnhubClient") as mock_fh_cls,
        ):
            mock_fh_cls.return_value = _mock_finnhub()
            ok, msg = execute_trade(db, decision, "aggressive")

        assert ok is True, f"Trade should have succeeded: {msg}"

        events = _get_entry_filled_events(db)
        assert len(events) == 1
        _, payload = events[0]

        assert "pilot_override" not in payload
        assert "pilot_size_multiplier" not in payload
        db.close()

    def test_short_pilot_override_tags_entry_filled_event(self):
        """Pilot override also works for SHORT trades."""
        engine = _make_engine()
        db = _make_session(engine)
        _seed_balance(db, "moderate")

        decision = _base_decision(action="SHORT", stop=155.0, target=140.0)

        with (
            patch(_GATE_PIPELINE, return_value=_gate_pipeline_with_pilot_override()),
            patch(_FIND_SIMILAR, return_value=[]),
            patch(_COMPUTE_SIM_STATS, return_value=_GOOD_SIM),
            patch(_ADJUST_CONFIDENCE, return_value=_CONF_OK),
            patch(_VALIDATE_TRADE),
            patch(_CHECK_CORRELATION, return_value=""),
            patch("agents.portfolio_manager.FinnhubClient") as mock_fh_cls,
        ):
            mock_fh_cls.return_value = _mock_finnhub()
            ok, msg = execute_trade(db, decision, "moderate")

        assert ok is True, f"Trade should have succeeded: {msg}"

        events = _get_entry_filled_events(db)
        assert len(events) == 1
        _, payload = events[0]

        assert payload["pilot_override"] is True
        assert payload["pilot_size_multiplier"] == 0.25
        assert payload["side"] == "short"
        db.close()

    def test_pilot_override_queryable_by_filter(self):
        """Requirement 9.3: pilot override trades queryable by filtering on pilot_override."""
        engine = _make_engine()
        db = _make_session(engine)
        _seed_balance(db, "moderate", cash=200_000.0)

        # Execute a pilot override trade
        decision1 = _base_decision(symbol="AAPL")
        # Execute a non-pilot trade
        decision2 = _base_decision(symbol="MSFT")

        with (
            patch(_GATE_PIPELINE, return_value=_gate_pipeline_with_pilot_override()),
            patch(_FIND_SIMILAR, return_value=[]),
            patch(_COMPUTE_SIM_STATS, return_value=_GOOD_SIM),
            patch(_ADJUST_CONFIDENCE, return_value=_CONF_OK),
            patch(_VALIDATE_TRADE),
            patch(_CHECK_CORRELATION, return_value=""),
            patch("agents.portfolio_manager.FinnhubClient") as mock_fh_cls,
        ):
            mock_fh_cls.return_value = _mock_finnhub()
            ok1, _ = execute_trade(db, decision1, "moderate")

        with (
            patch(_GATE_PIPELINE, return_value=_gate_pipeline_no_pilot()),
            patch(_FIND_SIMILAR, return_value=[]),
            patch(_COMPUTE_SIM_STATS, return_value=_GOOD_SIM),
            patch(_ADJUST_CONFIDENCE, return_value=_CONF_OK),
            patch(_VALIDATE_TRADE),
            patch(_CHECK_CORRELATION, return_value=""),
            patch("agents.portfolio_manager.FinnhubClient") as mock_fh_cls,
        ):
            mock_fh_cls.return_value = _mock_finnhub()
            ok2, _ = execute_trade(db, decision2, "moderate")

        assert ok1 is True
        assert ok2 is True

        # Query all entry_filled events and filter by pilot_override
        all_events = db.query(TradeEvent).filter_by(event_type="entry_filled").all()
        pilot_events = [
            e for e in all_events
            if e.payload_json and json.loads(e.payload_json).get("pilot_override") is True
        ]
        non_pilot_events = [
            e for e in all_events
            if not e.payload_json or json.loads(e.payload_json).get("pilot_override") is not True
        ]

        assert len(pilot_events) == 1
        assert pilot_events[0].symbol == "AAPL"
        assert len(non_pilot_events) == 1
        assert non_pilot_events[0].symbol == "MSFT"
        db.close()
