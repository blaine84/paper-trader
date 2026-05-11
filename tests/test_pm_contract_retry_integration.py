"""
Integration tests for PM Contract Validation Retry.

Tests the full run_profile() flow with mocked LLM calls to verify
retry behavior end-to-end: malformed decisions trigger retry, telemetry
events are emitted, and execution proceeds correctly.
"""

import json
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.schema import Base, Balance, AgentMemory, TradeEvent, get_session
from agents.portfolio_manager import run_profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine():
    """Create an in-memory SQLite engine with all tables."""
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    return engine


def _seed_balance(engine, profile_id: str, cash: float = 100_000.0):
    """Insert a Balance record so run_profile can find cash."""
    db = get_session(engine)
    db.add(Balance(profile=profile_id, cash=cash))
    db.commit()
    db.close()


def _seed_analyst_signal(engine, symbol: str, signal: dict):
    """Insert an analyst signal into AgentMemory."""
    db = get_session(engine)
    db.add(AgentMemory(
        agent="analyst",
        symbol=symbol,
        key="signal",
        value=json.dumps(signal),
    ))
    db.commit()
    db.close()


def _amd_strong_signal() -> dict:
    """Return a strong analyst signal for AMD that passes threshold filters."""
    return {
        "signal": "LONG",
        "strength": "strong",
        "confidence": "high",
        "setup_type": "news_catalyst_breakout",
        "market_regime": "risk_on",
        "bias": "LONG",
        "direction": "LONG",
        "entry_price": 161.0,
        "stop_loss": 152.0,
        "target": 179.0,
        "indicators": {
            "above_vwap": True,
            "ema_trend": "bullish",
            "rsi": 55.0,
            "macd_bias": "bullish",
            "bb_position": "upper",
        },
    }


# ---------------------------------------------------------------------------
# Test 12.1: malformed BUY → retry returns empty decisions with notes → zero executions
# ---------------------------------------------------------------------------

class TestMalformedBuyRetryEmptyDecisions:
    """
    Full run_profile() flow:
    - First LLM call returns malformed BUY AMD (null quantity, missing stop_loss)
    - Retry LLM call returns {"decisions": [], "portfolio_notes": "AMD not executable"}
    - Verify: zero execute_trade() calls, retry_triggered event, retry_failed event, notes merged
    """

    def test_malformed_buy_retry_returns_empty_decisions_zero_executions(self, monkeypatch):
        """
        Validates: Requirements 10.7

        End-to-end: malformed BUY with null fields triggers retry,
        retry returns empty decisions with notes, no trades execute,
        correct telemetry events are emitted, and notes are merged.
        """
        # Enable retry via env var
        monkeypatch.setenv("PM_CONTRACT_RETRY_ENABLED", "true")

        engine = _make_engine()
        profile_id = "aggressive"
        _seed_balance(engine, profile_id, cash=100_000.0)
        _seed_analyst_signal(engine, "AMD", _amd_strong_signal())

        # First LLM call: PM entry decision with malformed BUY (null quantity, missing stop_loss)
        first_llm_response = json.dumps({
            "decisions": [
                {
                    "symbol": "AMD",
                    "action": "BUY",
                    "quantity": None,
                    "entry_price": 161.0,
                    "target": 179.0,
                    "rationale": "AMD looks attractive with strong momentum",
                    "setup_type": "news_catalyst_breakout",
                }
            ],
            "portfolio_notes": "Watching AMD for breakout continuation",
        })

        # Retry LLM call: returns empty decisions with explanatory notes
        retry_llm_response = json.dumps({
            "decisions": [],
            "portfolio_notes": "AMD not executable",
        })

        # Track call count to return different responses
        call_count = {"n": 0}

        def mock_call_llm(system_prompt, user_prompt, json_mode=False, tier="high", purpose=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call: PM entry decision
                return first_llm_response
            else:
                # Second call: retry
                return retry_llm_response

        # Mock FinnhubClient
        mock_fh = MagicMock()
        mock_fh.get_quote.return_value = {"price": 161.0}

        with (
            patch("agents.portfolio_manager.call_llm", side_effect=mock_call_llm),
            patch("agents.portfolio_manager.FinnhubClient", return_value=mock_fh),
            patch("agents.portfolio_manager.get_relevant_cases", return_value=[]),
            patch("agents.portfolio_manager.format_cases_digest_for_pm", return_value=""),
            patch("agents.portfolio_manager.build_pm_strategy_context", return_value=""),
            patch("agents.portfolio_manager.get_win_rate_by_setup", return_value=[]),
            patch("agents.portfolio_manager.check_track_record", return_value={
                "verdict": "OK", "size_multiplier": 1.0, "sample_size": 0,
            }),
        ):
            result = run_profile(engine, ["AMD"], profile_id)

        # ── Assertions ──

        # 1. Zero executions: no decisions should have executed=True
        decisions = result.get("decisions", [])
        executed_trades = [d for d in decisions if d.get("executed")]
        assert len(executed_trades) == 0, (
            f"Expected zero executed trades, got {len(executed_trades)}: {executed_trades}"
        )

        # 2. Verify telemetry events in the DB
        db = get_session(engine)

        # pm_contract_retry_triggered event should exist
        triggered_events = (
            db.query(TradeEvent)
            .filter_by(event_type="pm_contract_retry_triggered", profile=profile_id)
            .all()
        )
        assert len(triggered_events) == 1, (
            f"Expected 1 pm_contract_retry_triggered event, got {len(triggered_events)}"
        )
        triggered_payload = json.loads(triggered_events[0].payload_json)
        assert triggered_payload["profile"] == profile_id
        assert triggered_payload["initial_rejection_count"] >= 1

        # pm_contract_retry_failed event should exist (retry returned empty decisions)
        failed_events = (
            db.query(TradeEvent)
            .filter_by(event_type="pm_contract_retry_failed", profile=profile_id)
            .all()
        )
        assert len(failed_events) == 1, (
            f"Expected 1 pm_contract_retry_failed event, got {len(failed_events)}"
        )
        failed_payload = json.loads(failed_events[0].payload_json)
        assert failed_payload["profile"] == profile_id
        assert failed_payload["initial_rejection_count"] >= 1

        # No pm_contract_retry_succeeded event
        succeeded_events = (
            db.query(TradeEvent)
            .filter_by(event_type="pm_contract_retry_succeeded", profile=profile_id)
            .all()
        )
        assert len(succeeded_events) == 0, (
            f"Expected 0 pm_contract_retry_succeeded events, got {len(succeeded_events)}"
        )

        # 3. pm_decision_rejected event should exist for the unresolved rejection
        rejected_events = (
            db.query(TradeEvent)
            .filter_by(event_type="pm_decision_rejected", profile=profile_id)
            .all()
        )
        assert len(rejected_events) >= 1, (
            f"Expected at least 1 pm_decision_rejected event, got {len(rejected_events)}"
        )

        # 4. Notes should be merged: original notes + retry notes
        notes = result.get("portfolio_notes", "")
        assert "Watching AMD for breakout continuation" in notes, (
            f"Original notes not preserved in: {notes}"
        )
        assert "AMD not executable" in notes, (
            f"Retry notes not merged in: {notes}"
        )
        assert "Contract retry note:" in notes, (
            f"Expected 'Contract retry note:' separator in: {notes}"
        )

        # 5. Verify the LLM was called exactly twice (entry + retry)
        assert call_count["n"] == 2, (
            f"Expected 2 LLM calls (entry + retry), got {call_count['n']}"
        )

        db.close()


# ---------------------------------------------------------------------------
# Test 12.2: mixed output → retry corrects malformed → both orders proceed to gates
# ---------------------------------------------------------------------------

def _nvda_strong_signal() -> dict:
    """Return a strong analyst signal for NVDA that passes threshold filters."""
    return {
        "signal": "LONG",
        "strength": "strong",
        "confidence": "high",
        "setup_type": "momentum_breakout",
        "market_regime": "risk_on",
        "bias": "LONG",
        "direction": "LONG",
        "entry_price": 950.0,
        "stop_loss": 920.0,
        "target": 1000.0,
        "indicators": {
            "above_vwap": True,
            "ema_trend": "bullish",
            "rsi": 60.0,
            "macd_bias": "bullish",
            "bb_position": "upper",
        },
    }


class TestMixedOutputRetryCorrectionBothProceed:
    """
    Full run_profile() flow:
    - First LLM call returns one valid BUY NVDA + one malformed BUY AMD (null quantity)
    - Retry LLM call returns valid BUY AMD with complete fields
    - Verify: both orders reach gate pipeline, pm_contract_retry_succeeded event emitted,
      corrected rejection gets pm_contract_retry_corrected event (not pm_decision_rejected)
    """

    def test_mixed_output_retry_corrects_malformed_both_proceed(self, monkeypatch):
        """
        Validates: Requirements 10.8

        End-to-end: first LLM call returns one valid NVDA order and one malformed
        AMD order (null quantity). Retry corrects AMD. Both orders reach the gate
        pipeline. Correct telemetry events are emitted.

        Note: execute_trade is mocked to avoid in-memory SQLite session interference
        from the gate pipeline's internal session management. The test verifies that
        both orders are passed to execute_trade (i.e., they reach the gate pipeline).
        """
        # Enable retry via env var
        monkeypatch.setenv("PM_CONTRACT_RETRY_ENABLED", "true")

        engine = _make_engine()
        profile_id = "aggressive"
        _seed_balance(engine, profile_id, cash=100_000.0)
        _seed_analyst_signal(engine, "AMD", _amd_strong_signal())
        _seed_analyst_signal(engine, "NVDA", _nvda_strong_signal())

        # First LLM call: one valid BUY NVDA + one malformed BUY AMD (null quantity)
        first_llm_response = json.dumps({
            "decisions": [
                {
                    "symbol": "NVDA",
                    "action": "BUY",
                    "quantity": 5,
                    "entry_price": 950.0,
                    "stop_loss": 920.0,
                    "target": 1000.0,
                    "setup_type": "momentum_breakout",
                    "rationale": "NVDA strong momentum breakout setup",
                },
                {
                    "symbol": "AMD",
                    "action": "BUY",
                    "quantity": None,
                    "entry_price": 161.0,
                    "target": 179.0,
                    "rationale": "AMD looks attractive with strong momentum",
                    "setup_type": "news_catalyst_breakout",
                },
            ],
            "portfolio_notes": "Both NVDA and AMD showing strength",
        })

        # Retry LLM call: returns corrected BUY AMD with all required fields
        retry_llm_response = json.dumps({
            "decisions": [
                {
                    "symbol": "AMD",
                    "action": "BUY",
                    "quantity": 10,
                    "entry_price": 161.0,
                    "stop_loss": 152.0,
                    "target": 179.0,
                    "setup_type": "news_catalyst_breakout",
                    "rationale": "AMD corrected with proper stop and quantity",
                },
            ],
            "portfolio_notes": "",
        })

        # Track call count to return different responses
        call_count = {"n": 0}

        def mock_call_llm(system_prompt, user_prompt, json_mode=False, tier="high", purpose=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return first_llm_response
            else:
                return retry_llm_response

        # Mock FinnhubClient — return appropriate prices for each symbol
        mock_fh = MagicMock()

        def mock_get_quote(symbol):
            prices = {"NVDA": 950.0, "AMD": 161.0}
            return {"price": prices.get(symbol, 100.0)}

        mock_fh.get_quote.side_effect = mock_get_quote

        # Track which symbols reach execute_trade (gate pipeline entry point)
        executed_symbols = []

        def mock_execute_trade(db, decision, profile_id, **kwargs):
            executed_symbols.append(decision.get("symbol"))
            return True, "OK"

        with (
            patch("agents.portfolio_manager.call_llm", side_effect=mock_call_llm),
            patch("agents.portfolio_manager.FinnhubClient", return_value=mock_fh),
            patch("agents.portfolio_manager.get_relevant_cases", return_value=[]),
            patch("agents.portfolio_manager.format_cases_digest_for_pm", return_value=""),
            patch("agents.portfolio_manager.build_pm_strategy_context", return_value=""),
            patch("agents.portfolio_manager.get_win_rate_by_setup", return_value=[]),
            patch("agents.portfolio_manager.check_track_record", return_value={
                "verdict": "OK", "size_multiplier": 1.0, "sample_size": 0,
            }),
            patch("agents.portfolio_manager.execute_trade", side_effect=mock_execute_trade),
        ):
            result = run_profile(engine, ["AMD", "NVDA"], profile_id)

        # ── Assertions ──

        # 1. Both orders should reach the gate pipeline (execute_trade called for both)
        assert "NVDA" in executed_symbols, (
            f"Expected NVDA to reach gate pipeline (execute_trade), "
            f"but only got: {executed_symbols}"
        )
        assert "AMD" in executed_symbols, (
            f"Expected AMD to reach gate pipeline (execute_trade), "
            f"but only got: {executed_symbols}"
        )

        # Also verify both appear in the result decisions
        decisions = result.get("decisions", [])
        nvda_decisions = [d for d in decisions if d.get("symbol") == "NVDA"]
        amd_decisions = [d for d in decisions if d.get("symbol") == "AMD"]
        assert len(nvda_decisions) >= 1, (
            f"Expected NVDA in result decisions, got: {decisions}"
        )
        assert len(amd_decisions) >= 1, (
            f"Expected AMD in result decisions, got: {decisions}"
        )

        # 2. Verify telemetry events in the DB
        db = get_session(engine)

        # pm_contract_retry_triggered event should exist
        triggered_events = (
            db.query(TradeEvent)
            .filter_by(event_type="pm_contract_retry_triggered", profile=profile_id)
            .all()
        )
        assert len(triggered_events) == 1, (
            f"Expected 1 pm_contract_retry_triggered event, got {len(triggered_events)}"
        )
        triggered_payload = json.loads(triggered_events[0].payload_json)
        assert triggered_payload["profile"] == profile_id
        assert triggered_payload["initial_rejection_count"] >= 1

        # pm_contract_retry_succeeded event should exist (retry produced valid AMD order)
        succeeded_events = (
            db.query(TradeEvent)
            .filter_by(event_type="pm_contract_retry_succeeded", profile=profile_id)
            .all()
        )
        assert len(succeeded_events) == 1, (
            f"Expected 1 pm_contract_retry_succeeded event, got {len(succeeded_events)}"
        )
        succeeded_payload = json.loads(succeeded_events[0].payload_json)
        assert succeeded_payload["profile"] == profile_id
        assert succeeded_payload["initial_rejection_count"] >= 1
        assert succeeded_payload["retry_order_count"] >= 1

        # No pm_contract_retry_failed event
        failed_events = (
            db.query(TradeEvent)
            .filter_by(event_type="pm_contract_retry_failed", profile=profile_id)
            .all()
        )
        assert len(failed_events) == 0, (
            f"Expected 0 pm_contract_retry_failed events, got {len(failed_events)}"
        )

        # 3. Corrected rejection gets pm_contract_retry_corrected event (NOT pm_decision_rejected)
        corrected_events = (
            db.query(TradeEvent)
            .filter_by(event_type="pm_contract_retry_corrected", profile=profile_id)
            .all()
        )
        assert len(corrected_events) >= 1, (
            f"Expected at least 1 pm_contract_retry_corrected event, got {len(corrected_events)}"
        )
        # Verify the corrected event is for AMD
        corrected_payload = json.loads(corrected_events[0].payload_json)
        assert corrected_payload["symbol"] == "AMD", (
            f"Expected corrected event for AMD, got: {corrected_payload}"
        )

        # 4. AMD should NOT have a pm_decision_rejected event (it was corrected, not rejected)
        rejected_events = (
            db.query(TradeEvent)
            .filter_by(event_type="pm_decision_rejected", profile=profile_id)
            .all()
        )
        # Check that none of the rejected events are for AMD
        amd_rejected = [
            e for e in rejected_events
            if "AMD" in (json.loads(e.payload_json).get("symbol", "") or "")
        ]
        assert len(amd_rejected) == 0, (
            f"AMD should NOT have pm_decision_rejected event (was corrected by retry), "
            f"but found {len(amd_rejected)} rejection events for AMD"
        )

        # 5. Verify the LLM was called exactly twice (entry + retry)
        assert call_count["n"] == 2, (
            f"Expected 2 LLM calls (entry + retry), got {call_count['n']}"
        )

        db.close()


# ---------------------------------------------------------------------------
# Test 12.3: disabled flag → no retry call made
# ---------------------------------------------------------------------------

class TestDisabledFlagNoRetryCall:
    """
    Full run_profile() flow with PM_CONTRACT_RETRY_ENABLED not set or "false":
    - Same malformed input as test 12.1 (malformed BUY AMD with null quantity)
    - Verify: no retry LLM call made (only 1 LLM call total), only original
      valid orders proceed, rejections logged as pm_decision_rejected
    - No retry telemetry events emitted
    """

    def test_retry_disabled_env_not_set_no_retry_call(self, monkeypatch):
        """
        Validates: Requirements 7.1, 7.3, 10.9

        When PM_CONTRACT_RETRY_ENABLED is not set, the retry path is skipped
        entirely. Only one LLM call is made (the initial entry call), no retry
        telemetry events are emitted, and the malformed rejection is logged
        as pm_decision_rejected.
        """
        # Ensure PM_CONTRACT_RETRY_ENABLED is NOT set
        monkeypatch.delenv("PM_CONTRACT_RETRY_ENABLED", raising=False)

        engine = _make_engine()
        profile_id = "aggressive"
        _seed_balance(engine, profile_id, cash=100_000.0)
        _seed_analyst_signal(engine, "AMD", _amd_strong_signal())

        # First (and only) LLM call: malformed BUY AMD with null quantity
        first_llm_response = json.dumps({
            "decisions": [
                {
                    "symbol": "AMD",
                    "action": "BUY",
                    "quantity": None,
                    "entry_price": 161.0,
                    "target": 179.0,
                    "rationale": "AMD looks attractive with strong momentum",
                    "setup_type": "news_catalyst_breakout",
                }
            ],
            "portfolio_notes": "Watching AMD for breakout continuation",
        })

        call_count = {"n": 0}

        def mock_call_llm(system_prompt, user_prompt, json_mode=False, tier="high", purpose=None):
            call_count["n"] += 1
            return first_llm_response

        mock_fh = MagicMock()
        mock_fh.get_quote.return_value = {"price": 161.0}

        with (
            patch("agents.portfolio_manager.call_llm", side_effect=mock_call_llm),
            patch("agents.portfolio_manager.FinnhubClient", return_value=mock_fh),
            patch("agents.portfolio_manager.get_relevant_cases", return_value=[]),
            patch("agents.portfolio_manager.format_cases_digest_for_pm", return_value=""),
            patch("agents.portfolio_manager.build_pm_strategy_context", return_value=""),
            patch("agents.portfolio_manager.get_win_rate_by_setup", return_value=[]),
            patch("agents.portfolio_manager.check_track_record", return_value={
                "verdict": "OK", "size_multiplier": 1.0, "sample_size": 0,
            }),
        ):
            result = run_profile(engine, ["AMD"], profile_id)

        # ── Assertions ──

        # 1. Only ONE LLM call made (no retry)
        assert call_count["n"] == 1, (
            f"Expected exactly 1 LLM call (no retry), got {call_count['n']}"
        )

        # 2. Zero executions: malformed decision should not execute
        decisions = result.get("decisions", [])
        executed_trades = [d for d in decisions if d.get("executed")]
        assert len(executed_trades) == 0, (
            f"Expected zero executed trades, got {len(executed_trades)}: {executed_trades}"
        )

        # 3. Verify NO retry telemetry events in the DB
        db = get_session(engine)

        # No pm_contract_retry_triggered event
        triggered_events = (
            db.query(TradeEvent)
            .filter_by(event_type="pm_contract_retry_triggered", profile=profile_id)
            .all()
        )
        assert len(triggered_events) == 0, (
            f"Expected 0 pm_contract_retry_triggered events (retry disabled), "
            f"got {len(triggered_events)}"
        )

        # No pm_contract_retry_succeeded event
        succeeded_events = (
            db.query(TradeEvent)
            .filter_by(event_type="pm_contract_retry_succeeded", profile=profile_id)
            .all()
        )
        assert len(succeeded_events) == 0, (
            f"Expected 0 pm_contract_retry_succeeded events (retry disabled), "
            f"got {len(succeeded_events)}"
        )

        # No pm_contract_retry_failed event
        failed_events = (
            db.query(TradeEvent)
            .filter_by(event_type="pm_contract_retry_failed", profile=profile_id)
            .all()
        )
        assert len(failed_events) == 0, (
            f"Expected 0 pm_contract_retry_failed events (retry disabled), "
            f"got {len(failed_events)}"
        )

        # No pm_contract_retry_corrected event
        corrected_events = (
            db.query(TradeEvent)
            .filter_by(event_type="pm_contract_retry_corrected", profile=profile_id)
            .all()
        )
        assert len(corrected_events) == 0, (
            f"Expected 0 pm_contract_retry_corrected events (retry disabled), "
            f"got {len(corrected_events)}"
        )

        # 4. Rejection logged as pm_decision_rejected (normal rejection path)
        rejected_events = (
            db.query(TradeEvent)
            .filter_by(event_type="pm_decision_rejected", profile=profile_id)
            .all()
        )
        assert len(rejected_events) >= 1, (
            f"Expected at least 1 pm_decision_rejected event, got {len(rejected_events)}"
        )

        db.close()

    def test_retry_disabled_env_set_false_no_retry_call(self, monkeypatch):
        """
        Validates: Requirements 7.1, 7.3, 10.9

        When PM_CONTRACT_RETRY_ENABLED is explicitly set to "false", the retry
        path is skipped entirely. Identical behavior to env var not being set.
        """
        # Explicitly set PM_CONTRACT_RETRY_ENABLED to "false"
        monkeypatch.setenv("PM_CONTRACT_RETRY_ENABLED", "false")

        engine = _make_engine()
        profile_id = "aggressive"
        _seed_balance(engine, profile_id, cash=100_000.0)
        _seed_analyst_signal(engine, "AMD", _amd_strong_signal())

        # First (and only) LLM call: malformed BUY AMD with null quantity
        first_llm_response = json.dumps({
            "decisions": [
                {
                    "symbol": "AMD",
                    "action": "BUY",
                    "quantity": None,
                    "entry_price": 161.0,
                    "target": 179.0,
                    "rationale": "AMD looks attractive with strong momentum",
                    "setup_type": "news_catalyst_breakout",
                }
            ],
            "portfolio_notes": "Watching AMD for breakout continuation",
        })

        call_count = {"n": 0}

        def mock_call_llm(system_prompt, user_prompt, json_mode=False, tier="high", purpose=None):
            call_count["n"] += 1
            return first_llm_response

        mock_fh = MagicMock()
        mock_fh.get_quote.return_value = {"price": 161.0}

        with (
            patch("agents.portfolio_manager.call_llm", side_effect=mock_call_llm),
            patch("agents.portfolio_manager.FinnhubClient", return_value=mock_fh),
            patch("agents.portfolio_manager.get_relevant_cases", return_value=[]),
            patch("agents.portfolio_manager.format_cases_digest_for_pm", return_value=""),
            patch("agents.portfolio_manager.build_pm_strategy_context", return_value=""),
            patch("agents.portfolio_manager.get_win_rate_by_setup", return_value=[]),
            patch("agents.portfolio_manager.check_track_record", return_value={
                "verdict": "OK", "size_multiplier": 1.0, "sample_size": 0,
            }),
        ):
            result = run_profile(engine, ["AMD"], profile_id)

        # ── Assertions ──

        # 1. Only ONE LLM call made (no retry)
        assert call_count["n"] == 1, (
            f"Expected exactly 1 LLM call (no retry), got {call_count['n']}"
        )

        # 2. Zero executions
        decisions = result.get("decisions", [])
        executed_trades = [d for d in decisions if d.get("executed")]
        assert len(executed_trades) == 0, (
            f"Expected zero executed trades, got {len(executed_trades)}: {executed_trades}"
        )

        # 3. No retry telemetry events
        db = get_session(engine)

        triggered_events = (
            db.query(TradeEvent)
            .filter_by(event_type="pm_contract_retry_triggered", profile=profile_id)
            .all()
        )
        assert len(triggered_events) == 0, (
            f"Expected 0 pm_contract_retry_triggered events, got {len(triggered_events)}"
        )

        succeeded_events = (
            db.query(TradeEvent)
            .filter_by(event_type="pm_contract_retry_succeeded", profile=profile_id)
            .all()
        )
        assert len(succeeded_events) == 0, (
            f"Expected 0 pm_contract_retry_succeeded events, got {len(succeeded_events)}"
        )

        failed_events = (
            db.query(TradeEvent)
            .filter_by(event_type="pm_contract_retry_failed", profile=profile_id)
            .all()
        )
        assert len(failed_events) == 0, (
            f"Expected 0 pm_contract_retry_failed events, got {len(failed_events)}"
        )

        # 4. Rejection logged as pm_decision_rejected
        rejected_events = (
            db.query(TradeEvent)
            .filter_by(event_type="pm_decision_rejected", profile=profile_id)
            .all()
        )
        assert len(rejected_events) >= 1, (
            f"Expected at least 1 pm_decision_rejected event, got {len(rejected_events)}"
        )

        db.close()
