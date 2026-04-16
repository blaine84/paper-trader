"""
Tests for the two-tier review routing in run_profile().

Covers:
- _meets_threshold: opposing evidence threshold comparison
- _check_reversal_triggers: trigger detection (thesis_invalidation, opposing signal, explicit CLOSE)
- run_profile routing: positions routed to correct review handler
- Signal usage logging (advisory vs authoritative)
- Entry logic preserved for new positions
"""

import json
import logging
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, Balance, Trade, Position, AgentMemory, get_session
from models.case import Case  # noqa: F401 — registers with Base
from agents.portfolio_manager import (
    _meets_threshold,
    _check_reversal_triggers,
    STRENGTH_ORDER,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _make_trade(db, symbol="AMD", profile="moderate", direction="LONG",
                entry_price=150.0, stop_price=145.0, target_price=160.0,
                thesis="Test thesis", setup_type="gap_and_go",
                invalidators=None, entry_time=None):
    """Create and persist an open Trade with Entry Contract fields."""
    if invalidators is None:
        invalidators = json.dumps([{
            "type": "price_below_level",
            "reference": str(stop_price),
            "confirmation": "5m_close",
            "lookback_bars": 1,
        }])
    elif isinstance(invalidators, list):
        invalidators = json.dumps(invalidators)

    trade = Trade(
        symbol=symbol,
        profile=profile,
        direction=direction,
        quantity=50,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        status="open",
        thesis=thesis,
        setup_type=setup_type,
        invalidators=invalidators,
        entry_time=entry_time or datetime.utcnow() - timedelta(hours=2),
    )
    db.add(trade)
    db.commit()
    return trade


def _make_position(db, symbol="AMD", profile="moderate", side="long",
                   quantity=50, avg_cost=150.0):
    """Create and persist a Position."""
    pos = Position(
        symbol=symbol,
        profile=profile,
        side=side,
        quantity=quantity,
        avg_cost=avg_cost,
    )
    db.add(pos)
    db.commit()
    return pos


# ---------------------------------------------------------------------------
# Tests for _meets_threshold
# ---------------------------------------------------------------------------

class TestMeetsThreshold:
    def test_strong_meets_moderate(self):
        assert _meets_threshold("strong", "moderate") is True

    def test_strong_meets_strong(self):
        assert _meets_threshold("strong", "strong") is True

    def test_moderate_meets_moderate(self):
        assert _meets_threshold("moderate", "moderate") is True

    def test_moderate_does_not_meet_strong(self):
        assert _meets_threshold("moderate", "strong") is False

    def test_weak_does_not_meet_moderate(self):
        assert _meets_threshold("weak", "moderate") is False

    def test_weak_does_not_meet_strong(self):
        assert _meets_threshold("weak", "strong") is False

    def test_weak_meets_weak(self):
        assert _meets_threshold("weak", "weak") is True

    def test_unknown_strength_does_not_meet_any(self):
        assert _meets_threshold("unknown", "weak") is False

    def test_case_insensitive(self):
        assert _meets_threshold("Strong", "moderate") is True
        assert _meets_threshold("MODERATE", "moderate") is True


# ---------------------------------------------------------------------------
# Tests for _check_reversal_triggers
# ---------------------------------------------------------------------------

class TestCheckReversalTriggers:
    """Test the trigger detection logic for routing to Reversal/Close Review."""

    def test_thesis_invalidation_trigger(self):
        """Thesis invalidation from Price Monitor triggers Reversal Review."""
        engine = _make_engine()
        db = _make_session(engine)
        trade = _make_trade(db, symbol="AMD", direction="LONG")

        # Add thesis_invalidation trigger in AgentMemory
        inv_data = {
            "type": "thesis_invalidation",
            "symbol": "AMD",
            "invalidator": {"type": "price_below_level", "reference": "145.0"},
        }
        db.add(AgentMemory(
            agent="price_monitor",
            symbol="AMD",
            key="thesis_invalidation",
            value=json.dumps(inv_data),
        ))
        db.commit()

        profile = {"opposing_evidence_threshold": "strong"}
        trigger = _check_reversal_triggers(db, trade, {}, None, profile)

        assert trigger is not None
        assert trigger["type"] == "thesis_invalidation"
        assert "AMD" in trigger["details"]
        db.close()

    def test_no_trigger_returns_none(self):
        """No triggers → returns None (Maintenance Review)."""
        engine = _make_engine()
        db = _make_session(engine)
        trade = _make_trade(db, symbol="AMD", direction="LONG")

        # Confirming signal (same direction as position)
        signal = {"bias": "LONG", "strength": "strong", "signal": "LONG"}
        profile = {"opposing_evidence_threshold": "strong"}

        trigger = _check_reversal_triggers(db, trade, {}, signal, profile)
        assert trigger is None
        db.close()

    def test_explicit_close_signal_triggers_reversal(self):
        """Explicit CLOSE signal triggers Reversal Review."""
        engine = _make_engine()
        db = _make_session(engine)
        trade = _make_trade(db, symbol="AMD", direction="LONG")

        signal = {"signal": "CLOSE", "bias": "LONG", "strength": "strong"}
        profile = {"opposing_evidence_threshold": "strong"}

        trigger = _check_reversal_triggers(db, trade, {}, signal, profile)
        assert trigger is not None
        assert trigger["type"] == "explicit_close"
        db.close()

    def test_opposing_signal_meets_threshold(self):
        """Opposing signal meeting threshold triggers Reversal Review."""
        engine = _make_engine()
        db = _make_session(engine)
        trade = _make_trade(db, symbol="AMD", direction="LONG")

        # SHORT signal with strong strength against LONG position
        signal = {"bias": "SHORT", "strength": "strong", "signal": "SHORT"}
        profile = {"opposing_evidence_threshold": "strong"}

        trigger = _check_reversal_triggers(db, trade, {}, signal, profile)
        assert trigger is not None
        assert trigger["type"] == "opposing_signal"
        assert "strong" in trigger["details"]
        db.close()

    def test_opposing_signal_below_threshold(self):
        """Opposing signal below threshold does NOT trigger Reversal Review."""
        engine = _make_engine()
        db = _make_session(engine)
        trade = _make_trade(db, symbol="AMD", direction="LONG")

        # SHORT signal with moderate strength, but threshold is strong
        signal = {"bias": "SHORT", "strength": "moderate", "signal": "SHORT"}
        profile = {"opposing_evidence_threshold": "strong"}

        trigger = _check_reversal_triggers(db, trade, {}, signal, profile)
        assert trigger is None
        db.close()

    def test_opposing_signal_moderate_threshold_conservative(self):
        """Conservative profile with moderate threshold triggers on moderate opposing signal."""
        engine = _make_engine()
        db = _make_session(engine)
        trade = _make_trade(db, symbol="AMD", direction="LONG")

        signal = {"bias": "SHORT", "strength": "moderate", "signal": "SHORT"}
        profile = {"opposing_evidence_threshold": "moderate"}

        trigger = _check_reversal_triggers(db, trade, {}, signal, profile)
        assert trigger is not None
        assert trigger["type"] == "opposing_signal"
        db.close()

    def test_short_position_long_signal_is_opposing(self):
        """SHORT position receiving LONG signal is opposing."""
        engine = _make_engine()
        db = _make_session(engine)
        trade = _make_trade(db, symbol="AMD", direction="SHORT")

        signal = {"bias": "LONG", "strength": "strong", "signal": "LONG"}
        profile = {"opposing_evidence_threshold": "strong"}

        trigger = _check_reversal_triggers(db, trade, {}, signal, profile)
        assert trigger is not None
        assert trigger["type"] == "opposing_signal"
        db.close()

    def test_no_signal_no_trigger(self):
        """No signal at all → no trigger (Maintenance Review)."""
        engine = _make_engine()
        db = _make_session(engine)
        trade = _make_trade(db, symbol="AMD", direction="LONG")

        profile = {"opposing_evidence_threshold": "strong"}
        trigger = _check_reversal_triggers(db, trade, {}, None, profile)
        assert trigger is None
        db.close()

    def test_thesis_invalidation_takes_priority_over_signal(self):
        """Thesis invalidation trigger takes priority even if signal is confirming."""
        engine = _make_engine()
        db = _make_session(engine)
        trade = _make_trade(db, symbol="AMD", direction="LONG")

        # Add thesis_invalidation trigger
        db.add(AgentMemory(
            agent="price_monitor",
            symbol="AMD",
            key="thesis_invalidation",
            value=json.dumps({"invalidator": {"type": "price_below_level"}}),
        ))
        db.commit()

        # Confirming signal (same direction)
        signal = {"bias": "LONG", "strength": "strong", "signal": "LONG"}
        profile = {"opposing_evidence_threshold": "strong"}

        trigger = _check_reversal_triggers(db, trade, {}, signal, profile)
        assert trigger is not None
        assert trigger["type"] == "thesis_invalidation"
        db.close()


# ---------------------------------------------------------------------------
# Tests for signal usage logging in run_profile
# ---------------------------------------------------------------------------

class TestSignalUsageLogging:
    """Verify that run_profile logs advisory vs authoritative signal usage (Req 4.4)."""

    def test_advisory_signal_logged(self, caplog):
        """Confirming signal for open position is logged as advisory."""
        engine = _make_engine()
        db = _make_session(engine)
        profile_id = "moderate"
        _seed_balance(db, profile_id)

        # Create position with Entry Contract
        _make_position(db, symbol="AMD", profile=profile_id)
        _make_trade(db, symbol="AMD", profile=profile_id, direction="LONG",
                    thesis="Test thesis")

        # Add confirming analyst signal
        db.add(AgentMemory(
            agent="analyst", symbol="AMD", key="signal",
            value=json.dumps({
                "bias": "LONG", "strength": "strong", "signal": "LONG",
                "confidence": "high", "indicators": {},
            }),
        ))
        db.commit()
        db.close()

        from agents.portfolio_manager import run_profile

        with caplog.at_level(logging.INFO, logger="agents.portfolio_manager"):
            with (
                patch("agents.portfolio_manager.FinnhubClient") as mock_fh_cls,
                patch("agents.portfolio_manager.call_llm") as mock_llm,
                patch("agents.portfolio_manager.parse_json_response") as mock_parse,
                patch("agents.portfolio_manager.run_maintenance_review") as mock_maint,
                patch("agents.portfolio_manager.get_relevant_cases", return_value=[]),
                patch("agents.portfolio_manager.format_cases_for_prompt", return_value=""),
                patch("agents.portfolio_manager.build_strategy_context", return_value=""),
                patch("agents.portfolio_manager.get_win_rate_by_setup", return_value=[]),
                patch("utils.behavioral_params.get_behavioral_params", return_value={}),
            ):
                mock_fh = MagicMock()
                mock_fh.get_quote.return_value = {"price": 155.0}
                mock_fh_cls.return_value = mock_fh

                mock_maint.return_value = {
                    "symbol": "AMD", "action": "hold",
                    "reasoning": "Thesis intact", "new_stop": None,
                    "new_target": None, "trim_pct": None,
                }
                mock_llm.return_value = '{"decisions": [], "portfolio_notes": ""}'
                mock_parse.return_value = {"decisions": [], "portfolio_notes": ""}

                run_profile(engine, ["AMD"], profile_id)

        log_text = caplog.text
        assert "SIGNAL USAGE" in log_text
        assert "advisory" in log_text.lower()

    def test_authoritative_signal_logged(self, caplog):
        """Opposing signal triggering Reversal Review is logged as authoritative."""
        engine = _make_engine()
        db = _make_session(engine)
        profile_id = "moderate"
        _seed_balance(db, profile_id)

        _make_position(db, symbol="AMD", profile=profile_id)
        _make_trade(db, symbol="AMD", profile=profile_id, direction="LONG",
                    thesis="Test thesis")

        # Add opposing analyst signal (SHORT against LONG position, strong)
        db.add(AgentMemory(
            agent="analyst", symbol="AMD", key="signal",
            value=json.dumps({
                "bias": "SHORT", "strength": "strong", "signal": "SHORT",
                "confidence": "high", "indicators": {},
            }),
        ))
        db.commit()
        db.close()

        from agents.portfolio_manager import run_profile

        with caplog.at_level(logging.INFO, logger="agents.portfolio_manager"):
            with (
                patch("agents.portfolio_manager.FinnhubClient") as mock_fh_cls,
                patch("agents.portfolio_manager.call_llm") as mock_llm,
                patch("agents.portfolio_manager.parse_json_response") as mock_parse,
                patch("agents.portfolio_manager.run_reversal_close_review") as mock_rev,
                patch("agents.portfolio_manager.get_relevant_cases", return_value=[]),
                patch("agents.portfolio_manager.format_cases_for_prompt", return_value=""),
                patch("agents.portfolio_manager.build_strategy_context", return_value=""),
                patch("agents.portfolio_manager.get_win_rate_by_setup", return_value=[]),
                patch("utils.behavioral_params.get_behavioral_params", return_value={}),
            ):
                mock_fh = MagicMock()
                mock_fh.get_quote.return_value = {"price": 155.0}
                mock_fh_cls.return_value = mock_fh

                mock_rev.return_value = {
                    "symbol": "AMD", "action": "hold_tighten",
                    "reasoning": "Ambiguous", "trigger": "opposing_signal",
                    "invalidator": None,
                }
                mock_llm.return_value = '{"decisions": [], "portfolio_notes": ""}'
                mock_parse.return_value = {"decisions": [], "portfolio_notes": ""}

                run_profile(engine, ["AMD"], profile_id)

        log_text = caplog.text
        assert "SIGNAL USAGE" in log_text
        assert "authoritative" in log_text.lower()


# ---------------------------------------------------------------------------
# Tests for routing logic in run_profile
# ---------------------------------------------------------------------------

class TestRunProfileRouting:
    """Verify positions are routed to the correct review handler."""

    def test_position_without_entry_contract_skips_review(self, caplog):
        """Position without Entry Contract but no stop/target is skipped."""
        engine = _make_engine()
        db = _make_session(engine)
        profile_id = "moderate"
        _seed_balance(db, profile_id)

        _make_position(db, symbol="AMD", profile=profile_id)
        # Trade WITHOUT thesis AND without stop/target — migration cannot help
        _make_trade(db, symbol="AMD", profile=profile_id, thesis=None,
                    stop_price=None, target_price=None, invalidators="[]")
        db.close()

        from agents.portfolio_manager import run_profile

        with caplog.at_level(logging.INFO, logger="agents.portfolio_manager"):
            with (
                patch("agents.portfolio_manager.FinnhubClient") as mock_fh_cls,
                patch("agents.portfolio_manager.call_llm") as mock_llm,
                patch("agents.portfolio_manager.parse_json_response") as mock_parse,
                patch("agents.portfolio_manager.run_maintenance_review") as mock_maint,
                patch("agents.portfolio_manager.run_reversal_close_review") as mock_rev,
                patch("agents.portfolio_manager.get_relevant_cases", return_value=[]),
                patch("agents.portfolio_manager.format_cases_for_prompt", return_value=""),
                patch("agents.portfolio_manager.build_strategy_context", return_value=""),
                patch("agents.portfolio_manager.get_win_rate_by_setup", return_value=[]),
                patch("utils.behavioral_params.get_behavioral_params", return_value={}),
            ):
                mock_fh = MagicMock()
                mock_fh.get_quote.return_value = {"price": 155.0}
                mock_fh_cls.return_value = mock_fh
                mock_llm.return_value = '{"decisions": [], "portfolio_notes": ""}'
                mock_parse.return_value = {"decisions": [], "portfolio_notes": ""}

                run_profile(engine, ["AMD"], profile_id)

        # Neither review handler should have been called
        mock_maint.assert_not_called()
        mock_rev.assert_not_called()
        assert "no Entry Contract" in caplog.text

    def test_legacy_migration_proceeds_to_review(self, caplog):
        """Position without Entry Contract but with stop/target gets migrated and reviewed."""
        engine = _make_engine()
        db = _make_session(engine)
        profile_id = "moderate"
        _seed_balance(db, profile_id)

        _make_position(db, symbol="AMD", profile=profile_id)
        # Trade WITHOUT thesis but WITH stop/target — legacy migration should succeed
        _make_trade(db, symbol="AMD", profile=profile_id, thesis=None)
        db.close()

        from agents.portfolio_manager import run_profile

        with caplog.at_level(logging.WARNING, logger="agents.portfolio_manager"):
            with (
                patch("agents.portfolio_manager.FinnhubClient") as mock_fh_cls,
                patch("agents.portfolio_manager.call_llm") as mock_llm,
                patch("agents.portfolio_manager.parse_json_response") as mock_parse,
                patch("agents.portfolio_manager.run_maintenance_review") as mock_maint,
                patch("agents.portfolio_manager.run_reversal_close_review") as mock_rev,
                patch("agents.portfolio_manager.get_relevant_cases", return_value=[]),
                patch("agents.portfolio_manager.format_cases_for_prompt", return_value=""),
                patch("agents.portfolio_manager.build_strategy_context", return_value=""),
                patch("agents.portfolio_manager.get_win_rate_by_setup", return_value=[]),
                patch("utils.behavioral_params.get_behavioral_params", return_value={}),
            ):
                mock_fh = MagicMock()
                mock_fh.get_quote.return_value = {"price": 155.0}
                mock_fh_cls.return_value = mock_fh
                mock_llm.return_value = '{"decisions": [], "portfolio_notes": ""}'
                mock_parse.return_value = {"decisions": [], "portfolio_notes": ""}
                mock_maint.return_value = {"symbol": "AMD", "action": "hold",
                                           "new_stop": None, "new_target": None,
                                           "trim_pct": None, "reasoning": "test"}

                run_profile(engine, ["AMD"], profile_id)

        # Legacy migration should have triggered Maintenance Review
        mock_maint.assert_called_once()
        mock_rev.assert_not_called()
        assert "Legacy trade migration" in caplog.text

    def test_position_with_trigger_routes_to_reversal(self):
        """Position with thesis_invalidation trigger routes to Reversal/Close Review."""
        engine = _make_engine()
        db = _make_session(engine)
        profile_id = "moderate"
        _seed_balance(db, profile_id)

        _make_position(db, symbol="AMD", profile=profile_id)
        _make_trade(db, symbol="AMD", profile=profile_id, direction="LONG",
                    thesis="Test thesis")

        # Add thesis_invalidation trigger
        db.add(AgentMemory(
            agent="price_monitor", symbol="AMD", key="thesis_invalidation",
            value=json.dumps({"invalidator": {"type": "price_below_level"}}),
        ))
        db.commit()
        db.close()

        from agents.portfolio_manager import run_profile

        with (
            patch("agents.portfolio_manager.FinnhubClient") as mock_fh_cls,
            patch("agents.portfolio_manager.call_llm") as mock_llm,
            patch("agents.portfolio_manager.parse_json_response") as mock_parse,
            patch("agents.portfolio_manager.run_maintenance_review") as mock_maint,
            patch("agents.portfolio_manager.run_reversal_close_review") as mock_rev,
            patch("agents.portfolio_manager.get_relevant_cases", return_value=[]),
            patch("agents.portfolio_manager.format_cases_for_prompt", return_value=""),
            patch("agents.portfolio_manager.build_strategy_context", return_value=""),
            patch("agents.portfolio_manager.get_win_rate_by_setup", return_value=[]),
            patch("utils.behavioral_params.get_behavioral_params", return_value={}),
        ):
            mock_fh = MagicMock()
            mock_fh.get_quote.return_value = {"price": 155.0}
            mock_fh_cls.return_value = mock_fh

            mock_rev.return_value = {
                "symbol": "AMD", "action": "hold_tighten",
                "reasoning": "Ambiguous", "trigger": "thesis_invalidation",
                "invalidator": None,
            }
            mock_llm.return_value = '{"decisions": [], "portfolio_notes": ""}'
            mock_parse.return_value = {"decisions": [], "portfolio_notes": ""}

            run_profile(engine, ["AMD"], profile_id)

        mock_rev.assert_called_once()
        mock_maint.assert_not_called()

    def test_position_without_trigger_routes_to_maintenance(self):
        """Position without any trigger routes to Maintenance Review."""
        engine = _make_engine()
        db = _make_session(engine)
        profile_id = "moderate"
        _seed_balance(db, profile_id)

        _make_position(db, symbol="AMD", profile=profile_id)
        _make_trade(db, symbol="AMD", profile=profile_id, direction="LONG",
                    thesis="Test thesis")
        db.close()

        from agents.portfolio_manager import run_profile

        with (
            patch("agents.portfolio_manager.FinnhubClient") as mock_fh_cls,
            patch("agents.portfolio_manager.call_llm") as mock_llm,
            patch("agents.portfolio_manager.parse_json_response") as mock_parse,
            patch("agents.portfolio_manager.run_maintenance_review") as mock_maint,
            patch("agents.portfolio_manager.run_reversal_close_review") as mock_rev,
            patch("agents.portfolio_manager.get_relevant_cases", return_value=[]),
            patch("agents.portfolio_manager.format_cases_for_prompt", return_value=""),
            patch("agents.portfolio_manager.build_strategy_context", return_value=""),
            patch("agents.portfolio_manager.get_win_rate_by_setup", return_value=[]),
            patch("utils.behavioral_params.get_behavioral_params", return_value={}),
        ):
            mock_fh = MagicMock()
            mock_fh.get_quote.return_value = {"price": 155.0}
            mock_fh_cls.return_value = mock_fh

            mock_maint.return_value = {
                "symbol": "AMD", "action": "hold",
                "reasoning": "Thesis intact", "new_stop": None,
                "new_target": None, "trim_pct": None,
            }
            mock_llm.return_value = '{"decisions": [], "portfolio_notes": ""}'
            mock_parse.return_value = {"decisions": [], "portfolio_notes": ""}

            run_profile(engine, ["AMD"], profile_id)

        mock_maint.assert_called_once()
        mock_rev.assert_not_called()

    def test_entry_logic_preserved_for_new_symbols(self):
        """New symbols (no existing position) go through existing entry logic."""
        engine = _make_engine()
        db = _make_session(engine)
        profile_id = "moderate"
        _seed_balance(db, profile_id)

        # Add analyst signal for a symbol with no position
        db.add(AgentMemory(
            agent="analyst", symbol="NVDA", key="signal",
            value=json.dumps({
                "bias": "LONG", "strength": "strong", "signal": "LONG",
                "confidence": "high", "indicators": {},
            }),
        ))
        db.commit()
        db.close()

        from agents.portfolio_manager import run_profile

        with (
            patch("agents.portfolio_manager.FinnhubClient") as mock_fh_cls,
            patch("agents.portfolio_manager.call_llm") as mock_llm,
            patch("agents.portfolio_manager.parse_json_response") as mock_parse,
            patch("agents.portfolio_manager.run_maintenance_review") as mock_maint,
            patch("agents.portfolio_manager.run_reversal_close_review") as mock_rev,
            patch("agents.portfolio_manager.get_relevant_cases", return_value=[]),
            patch("agents.portfolio_manager.format_cases_for_prompt", return_value=""),
            patch("agents.portfolio_manager.build_strategy_context", return_value=""),
            patch("agents.portfolio_manager.get_win_rate_by_setup", return_value=[]),
            patch("utils.behavioral_params.get_behavioral_params", return_value={}),
        ):
            mock_fh = MagicMock()
            mock_fh.get_quote.return_value = {"price": 500.0}
            mock_fh_cls.return_value = mock_fh

            # LLM returns a BUY decision for NVDA
            mock_llm.return_value = '{"decisions": [{"symbol": "NVDA", "action": "BUY", "quantity": 10, "price": 500.0, "stop_loss": 490.0, "target": 520.0, "rationale": "test"}], "portfolio_notes": ""}'
            mock_parse.return_value = {
                "decisions": [{
                    "symbol": "NVDA", "action": "BUY", "quantity": 10,
                    "price": 500.0, "stop_loss": 490.0, "target": 520.0,
                    "rationale": "test",
                }],
                "portfolio_notes": "",
            }

            result = run_profile(engine, ["NVDA"], profile_id)

        # LLM was called for entry decisions
        mock_llm.assert_called_once()
        # No review handlers called (no open positions)
        mock_maint.assert_not_called()
        mock_rev.assert_not_called()

    def test_close_from_llm_for_held_symbol_is_ignored(self, caplog):
        """LLM CLOSE decision for a held symbol is ignored (two-tier review handles closes)."""
        engine = _make_engine()
        db = _make_session(engine)
        profile_id = "moderate"
        _seed_balance(db, profile_id)

        _make_position(db, symbol="AMD", profile=profile_id)
        _make_trade(db, symbol="AMD", profile=profile_id, direction="LONG",
                    thesis="Test thesis")
        db.close()

        from agents.portfolio_manager import run_profile

        with caplog.at_level(logging.INFO, logger="agents.portfolio_manager"):
            with (
                patch("agents.portfolio_manager.FinnhubClient") as mock_fh_cls,
                patch("agents.portfolio_manager.call_llm") as mock_llm,
                patch("agents.portfolio_manager.parse_json_response") as mock_parse,
                patch("agents.portfolio_manager.run_maintenance_review") as mock_maint,
                patch("agents.portfolio_manager.execute_trade") as mock_exec,
                patch("agents.portfolio_manager.get_relevant_cases", return_value=[]),
                patch("agents.portfolio_manager.format_cases_for_prompt", return_value=""),
                patch("agents.portfolio_manager.build_strategy_context", return_value=""),
                patch("agents.portfolio_manager.get_win_rate_by_setup", return_value=[]),
                patch("utils.behavioral_params.get_behavioral_params", return_value={}),
            ):
                mock_fh = MagicMock()
                mock_fh.get_quote.return_value = {"price": 155.0}
                mock_fh_cls.return_value = mock_fh

                mock_maint.return_value = {
                    "symbol": "AMD", "action": "hold",
                    "reasoning": "Thesis intact", "new_stop": None,
                    "new_target": None, "trim_pct": None,
                }
                # LLM tries to CLOSE AMD
                mock_parse.return_value = {
                    "decisions": [{
                        "symbol": "AMD", "action": "CLOSE",
                        "quantity": 50, "price": 155.0,
                        "rationale": "taking profits",
                    }],
                    "portfolio_notes": "",
                }
                mock_llm.return_value = "{}"

                result = run_profile(engine, ["AMD"], profile_id)

        # execute_trade should NOT have been called for the CLOSE
        # (only the maintenance review hold was processed, which doesn't call execute_trade)
        close_calls = [
            c for c in mock_exec.call_args_list
            if c[0][1].get("action") == "CLOSE" and c[0][1].get("symbol") == "AMD"
        ]
        assert len(close_calls) == 0
        assert "Ignoring LLM CLOSE" in caplog.text
