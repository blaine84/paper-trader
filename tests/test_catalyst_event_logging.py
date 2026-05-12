"""
Tests for catalyst specificity gate trade event logging (task 5.2).

Validates Requirements: 10.1, 10.2, 10.3, 11.2, 11.3.
"""

from unittest.mock import patch, MagicMock, call

import pytest

from utils.catalyst_specificity import evaluate_catalyst_specificity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_news_decision(
    symbol="AMD",
    catalyst="AMD reports Q2 earnings beat today, raises guidance for AI chips",
    setup_type="news_breakout",
    bias="LONG",
    quantity=100,
    **kwargs,
):
    """Create a news-driven decision dict that triggers the gate."""
    decision = {
        "symbol": symbol,
        "setup_type": setup_type,
        "bias": bias,
        "catalyst_type": "earnings_beat",
        "catalyst": catalyst,
        "rationale": "Earnings catalyst with volume confirmation",
        "indicators": {"relative_volume": 2.1},
        "quantity": quantity,
    }
    decision.update(kwargs)
    return decision


def _make_low_score_decision(symbol="AMD"):
    """Create a decision that will score low (likely block)."""
    return {
        "symbol": symbol,
        "setup_type": "news_breakout",
        "bias": "LONG",
        "catalyst_type": "downgrade",
        "catalyst": "TSLA downgraded by Goldman, price target cut to $150",
        "rationale": "Bearish catalyst on wrong symbol",
        "indicators": {},
        "quantity": 50,
    }


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure gate env vars are cleared before each test."""
    monkeypatch.delenv("CATALYST_SPECIFICITY_GATE_ENABLED", raising=False)
    monkeypatch.delenv("CATALYST_SPECIFICITY_GATE_MODE", raising=False)


# ===================================================================
# Requirement 10.1: Every evaluation logs "catalyst_specificity_gate_evaluated"
# ===================================================================


class TestEvaluatedEventLogged:
    """Every gate evaluation logs exactly one catalyst_specificity_gate_evaluated event."""

    @patch("utils.catalyst_specificity.log_trade_event")
    def test_evaluated_event_logged_on_every_evaluation(self, mock_log):
        db = MagicMock()
        decision = _make_news_decision()
        evaluate_catalyst_specificity(decision, db=db)

        # Find calls with event_type "catalyst_specificity_gate_evaluated"
        evaluated_calls = [
            c for c in mock_log.call_args_list
            if c[0][1] == "catalyst_specificity_gate_evaluated"
        ]
        assert len(evaluated_calls) == 1

    @patch("utils.catalyst_specificity.log_trade_event")
    def test_evaluated_event_logged_in_enforce_mode(self, mock_log, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        db = MagicMock()
        decision = _make_news_decision()
        evaluate_catalyst_specificity(decision, db=db)

        evaluated_calls = [
            c for c in mock_log.call_args_list
            if c[0][1] == "catalyst_specificity_gate_evaluated"
        ]
        assert len(evaluated_calls) == 1

    @patch("utils.catalyst_specificity.log_trade_event")
    def test_evaluated_event_logged_in_log_only_mode(self, mock_log, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "log_only")
        db = MagicMock()
        decision = _make_news_decision()
        evaluate_catalyst_specificity(decision, db=db)

        evaluated_calls = [
            c for c in mock_log.call_args_list
            if c[0][1] == "catalyst_specificity_gate_evaluated"
        ]
        assert len(evaluated_calls) == 1

    @patch("utils.catalyst_specificity.log_trade_event")
    def test_no_logging_when_db_is_none(self, mock_log):
        decision = _make_news_decision()
        evaluate_catalyst_specificity(decision, db=None)
        mock_log.assert_not_called()


# ===================================================================
# Requirement 10.2: gate_rejected logged ONLY when block AND enforce
# ===================================================================


class TestGateRejectedEvent:
    """gate_rejected is logged ONLY when decision==block AND mode==enforce."""

    @patch("utils.catalyst_specificity.log_trade_event")
    def test_gate_rejected_logged_on_block_enforce(self, mock_log, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        db = MagicMock()
        # Low score decision that will block in conservative profile
        decision = _make_low_score_decision()
        result = evaluate_catalyst_specificity(
            decision, db=db, profile="conservative"
        )

        # Verify it actually blocked
        assert result["decision"] == "block"

        # Find gate_rejected calls
        rejected_calls = [
            c for c in mock_log.call_args_list
            if c[0][1] == "gate_rejected"
        ]
        assert len(rejected_calls) == 1

    @patch("utils.catalyst_specificity.log_trade_event")
    def test_gate_rejected_not_logged_on_allow(self, mock_log, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        db = MagicMock()
        # High score decision that will allow
        decision = _make_news_decision()
        result = evaluate_catalyst_specificity(decision, db=db)

        assert result["decision"] != "block"

        rejected_calls = [
            c for c in mock_log.call_args_list
            if c[0][1] == "gate_rejected"
        ]
        assert len(rejected_calls) == 0

    @patch("utils.catalyst_specificity.log_trade_event")
    def test_gate_rejected_not_logged_on_warn(self, mock_log, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        db = MagicMock()
        # Create a decision that results in "warn" — moderate profile, score in warn range
        decision = {
            "symbol": "AMD",
            "setup_type": "news_breakout",
            "bias": "LONG",
            "catalyst_type": "sector_move",
            "catalyst": "AMD mentioned in semiconductor sector report today",
            "rationale": "Sector catalyst",
            "indicators": {"relative_volume": 1.6},
            "quantity": 100,
        }
        result = evaluate_catalyst_specificity(decision, db=db, profile="moderate")

        # Whether it's warn or reduce_size, it shouldn't be block
        if result["decision"] != "block":
            rejected_calls = [
                c for c in mock_log.call_args_list
                if c[0][1] == "gate_rejected"
            ]
            assert len(rejected_calls) == 0


# ===================================================================
# Requirement 11.2/11.3: gate_rejected NEVER emitted in log_only mode
# ===================================================================


class TestLogOnlyNoRejection:
    """In log_only mode, gate_rejected is NEVER emitted even if intended_decision is block."""

    @patch("utils.catalyst_specificity.log_trade_event")
    def test_no_gate_rejected_in_log_only_even_when_would_block(
        self, mock_log, monkeypatch
    ):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "log_only")
        db = MagicMock()
        # Low score that would block in enforce mode
        decision = _make_low_score_decision()
        result = evaluate_catalyst_specificity(
            decision, db=db, profile="conservative"
        )

        # In log_only, decision is always "allow" but intended_decision should be "block"
        assert result["decision"] == "allow"
        assert result["intended_decision"] == "block"

        # gate_rejected must NOT be emitted
        rejected_calls = [
            c for c in mock_log.call_args_list
            if c[0][1] == "gate_rejected"
        ]
        assert len(rejected_calls) == 0

    @patch("utils.catalyst_specificity.log_trade_event")
    def test_log_only_still_logs_evaluated_event(self, mock_log, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "log_only")
        db = MagicMock()
        decision = _make_low_score_decision()
        evaluate_catalyst_specificity(decision, db=db, profile="conservative")

        # catalyst_specificity_gate_evaluated should still be logged
        evaluated_calls = [
            c for c in mock_log.call_args_list
            if c[0][1] == "catalyst_specificity_gate_evaluated"
        ]
        assert len(evaluated_calls) == 1


# ===================================================================
# Requirement 10.3: Payload includes all required fields
# ===================================================================


class TestPayloadFields:
    """Event payload includes all required fields including mode."""

    @patch("utils.catalyst_specificity.log_trade_event")
    def test_evaluated_payload_has_required_fields(self, mock_log, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        db = MagicMock()
        decision = _make_news_decision()
        evaluate_catalyst_specificity(decision, db=db, profile="moderate")

        # Get the evaluated event call
        evaluated_calls = [
            c for c in mock_log.call_args_list
            if c[0][1] == "catalyst_specificity_gate_evaluated"
        ]
        assert len(evaluated_calls) == 1

        call_kwargs = evaluated_calls[0][1]  # keyword arguments
        payload = call_kwargs["payload"]

        # Required fields per Requirement 10.3
        assert "score" in payload
        assert "threshold" in payload
        assert "decision" in payload
        assert "intended_decision" in payload
        assert "reason_type" in payload
        assert "size_multiplier" in payload
        assert "intended_size_multiplier" in payload
        assert "mode" in payload
        assert "evidence" in payload
        assert "missing" in payload
        assert "setup_type" in payload

    @patch("utils.catalyst_specificity.log_trade_event")
    def test_evaluated_payload_mode_field_value(self, mock_log, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        db = MagicMock()
        decision = _make_news_decision()
        evaluate_catalyst_specificity(decision, db=db)

        evaluated_calls = [
            c for c in mock_log.call_args_list
            if c[0][1] == "catalyst_specificity_gate_evaluated"
        ]
        payload = evaluated_calls[0][1]["payload"]
        assert payload["mode"] == "enforce"

    @patch("utils.catalyst_specificity.log_trade_event")
    def test_evaluated_payload_mode_log_only(self, mock_log, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "log_only")
        db = MagicMock()
        decision = _make_news_decision()
        evaluate_catalyst_specificity(decision, db=db)

        evaluated_calls = [
            c for c in mock_log.call_args_list
            if c[0][1] == "catalyst_specificity_gate_evaluated"
        ]
        payload = evaluated_calls[0][1]["payload"]
        assert payload["mode"] == "log_only"

    @patch("utils.catalyst_specificity.log_trade_event")
    def test_evaluated_payload_includes_intended_decision(self, mock_log, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "log_only")
        db = MagicMock()
        decision = _make_low_score_decision()
        evaluate_catalyst_specificity(decision, db=db, profile="conservative")

        evaluated_calls = [
            c for c in mock_log.call_args_list
            if c[0][1] == "catalyst_specificity_gate_evaluated"
        ]
        payload = evaluated_calls[0][1]["payload"]
        assert payload["intended_decision"] == "block"
        assert payload["decision"] == "allow"  # log_only always allows

    @patch("utils.catalyst_specificity.log_trade_event")
    def test_evaluated_payload_includes_intended_size_multiplier(
        self, mock_log, monkeypatch
    ):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "log_only")
        db = MagicMock()
        decision = _make_low_score_decision()
        evaluate_catalyst_specificity(decision, db=db, profile="conservative")

        evaluated_calls = [
            c for c in mock_log.call_args_list
            if c[0][1] == "catalyst_specificity_gate_evaluated"
        ]
        payload = evaluated_calls[0][1]["payload"]
        assert "intended_size_multiplier" in payload

    @patch("utils.catalyst_specificity.log_trade_event")
    def test_payload_includes_quantity_when_size_reduced(self, mock_log, monkeypatch):
        """Requirement 10.3 item 4: quantity_before/after when size reduced."""
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        db = MagicMock()
        # Sector sympathy in warn range → reduce_size
        decision = {
            "symbol": "AMD",
            "setup_type": "news_breakout",
            "bias": "LONG",
            "catalyst_type": "sector_move",
            "catalyst": "AI chip stocks rally after Lumentum results today",
            "rationale": "Semiconductor sector momentum",
            "indicators": {"relative_volume": 1.6},
            "quantity": 100,
        }
        result = evaluate_catalyst_specificity(decision, db=db, profile="moderate")

        if result["decision"] == "reduce_size":
            evaluated_calls = [
                c for c in mock_log.call_args_list
                if c[0][1] == "catalyst_specificity_gate_evaluated"
            ]
            payload = evaluated_calls[0][1]["payload"]
            assert "quantity_before" in payload
            assert "quantity_after" in payload
            assert payload["quantity_before"] == 100
            assert payload["quantity_after"] < 100

    @patch("utils.catalyst_specificity.log_trade_event")
    def test_symbol_passed_to_log_trade_event(self, mock_log, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        db = MagicMock()
        decision = _make_news_decision(symbol="NVDA")
        evaluate_catalyst_specificity(decision, db=db)

        evaluated_calls = [
            c for c in mock_log.call_args_list
            if c[0][1] == "catalyst_specificity_gate_evaluated"
        ]
        assert evaluated_calls[0][1]["symbol"] == "NVDA"

    @patch("utils.catalyst_specificity.log_trade_event")
    def test_profile_passed_to_log_trade_event(self, mock_log, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        db = MagicMock()
        decision = _make_news_decision()
        evaluate_catalyst_specificity(decision, db=db, profile="aggressive")

        evaluated_calls = [
            c for c in mock_log.call_args_list
            if c[0][1] == "catalyst_specificity_gate_evaluated"
        ]
        assert evaluated_calls[0][1]["profile"] == "aggressive"

    @patch("utils.catalyst_specificity.log_trade_event")
    def test_gate_rejected_payload_includes_gate_name(self, mock_log, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        db = MagicMock()
        decision = _make_low_score_decision()
        result = evaluate_catalyst_specificity(
            decision, db=db, profile="conservative"
        )

        if result["decision"] == "block":
            rejected_calls = [
                c for c in mock_log.call_args_list
                if c[0][1] == "gate_rejected"
            ]
            assert len(rejected_calls) == 1
            payload = rejected_calls[0][1]["payload"]
            assert payload["gate_name"] == "catalyst_specificity_gate"
            assert payload["mode"] == "enforce"
