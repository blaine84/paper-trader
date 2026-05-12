"""
Unit tests for evaluate_catalyst_specificity() public orchestrator (task 5.1).

Tests the full orchestration flow: enabled flag, applicability check,
scoring, decision engine, result schema, and reason string generation.

Requirements: 10.1, 10.2, 10.3, 11.1, 11.2, 11.3, 11.4, 11.5, 12.1, 12.2, 12.3
"""

import os

import pytest

from utils.catalyst_specificity import evaluate_catalyst_specificity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure gate env vars are cleared before each test."""
    monkeypatch.delenv("CATALYST_SPECIFICITY_GATE_ENABLED", raising=False)
    monkeypatch.delenv("CATALYST_SPECIFICITY_GATE_MODE", raising=False)


def _make_news_decision(
    symbol="AMD",
    setup_type="news_breakout",
    bias="LONG",
    catalyst_type="earnings_beat",
    catalyst="AMD reports Q2 earnings beat, raises guidance for AI chips",
    quantity=100,
    **kwargs,
):
    """Helper to build a standard news-driven decision dict."""
    d = {
        "symbol": symbol,
        "setup_type": setup_type,
        "bias": bias,
        "catalyst_type": catalyst_type,
        "catalyst": catalyst,
        "rationale": kwargs.get("rationale", "AMD earnings catalyst with strong volume"),
        "indicators": kwargs.get("indicators", {"relative_volume": 2.1}),
        "quantity": quantity,
    }
    d.update(kwargs)
    return d


# ---------------------------------------------------------------------------
# Requirement 11.5: Gate disabled via environment
# ---------------------------------------------------------------------------


class TestGateDisabled:
    """When CATALYST_SPECIFICITY_GATE_ENABLED=false, gate returns immediate allow."""

    def test_disabled_returns_allow(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_ENABLED", "false")
        decision = _make_news_decision()
        result = evaluate_catalyst_specificity(decision)

        assert result["decision"] == "allow"
        assert result["intended_decision"] == "allow"
        assert result["reason_type"] == "gate_disabled"
        assert result["score"] == 10
        assert result["size_multiplier"] == 1.0
        assert result["intended_size_multiplier"] == 1.0

    def test_disabled_returns_correct_gate_name(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_ENABLED", "false")
        result = evaluate_catalyst_specificity(_make_news_decision())
        assert result["gate"] == "catalyst_specificity_gate"

    def test_disabled_includes_symbol(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_ENABLED", "false")
        result = evaluate_catalyst_specificity(_make_news_decision(symbol="TSLA"))
        assert result["symbol"] == "TSLA"

    def test_disabled_reason_string(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_ENABLED", "false")
        result = evaluate_catalyst_specificity(_make_news_decision())
        assert "disabled" in result["reason"].lower()

    def test_disabled_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_ENABLED", "False")
        # "False" lowered == "false"
        result = evaluate_catalyst_specificity(_make_news_decision())
        assert result["reason_type"] == "gate_disabled"


# ---------------------------------------------------------------------------
# Gate not applicable (technical setup)
# ---------------------------------------------------------------------------


class TestGateNotApplicable:
    """When gate doesn't apply, return allow with reason_type=not_applicable."""

    def test_technical_setup_returns_not_applicable(self):
        decision = {
            "symbol": "AMD",
            "setup_type": "technical_breakout",
            "bias": "LONG",
            "rationale": "Breaking above resistance with momentum",
            "indicators": {"relative_volume": 1.8},
            "quantity": 50,
        }
        result = evaluate_catalyst_specificity(decision)

        assert result["decision"] == "allow"
        assert result["intended_decision"] == "allow"
        assert result["reason_type"] == "not_applicable"
        assert result["score"] == 10
        assert result["size_multiplier"] == 1.0

    def test_not_applicable_includes_gate_name(self):
        decision = {
            "symbol": "NVDA",
            "setup_type": "momentum_fade",
            "bias": "SHORT",
            "rationale": "Overextended move fading",
            "quantity": 25,
        }
        result = evaluate_catalyst_specificity(decision)
        assert result["gate"] == "catalyst_specificity_gate"
        assert result["reason_type"] == "not_applicable"

    def test_gap_and_go_without_catalyst_not_applicable(self):
        """gap_and_go without explicit catalyst fields should not trigger gate."""
        decision = {
            "symbol": "AMD",
            "setup_type": "gap_and_go",
            "bias": "LONG",
            "rationale": "Strong opening momentum move",
            "quantity": 100,
        }
        result = evaluate_catalyst_specificity(decision)
        assert result["reason_type"] == "not_applicable"


# ---------------------------------------------------------------------------
# Result schema completeness
# ---------------------------------------------------------------------------


class TestResultSchema:
    """Verify result dict matches the Gate Result Schema."""

    REQUIRED_KEYS = {
        "gate", "mode", "decision", "intended_decision", "reason_type",
        "score", "threshold", "symbol", "setup_type", "evidence",
        "missing", "size_multiplier", "intended_size_multiplier",
        "reason", "quantity_before", "quantity_after",
    }

    def test_all_keys_present_for_evaluated_candidate(self):
        decision = _make_news_decision()
        result = evaluate_catalyst_specificity(decision)
        assert set(result.keys()) == self.REQUIRED_KEYS

    def test_all_keys_present_when_disabled(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_ENABLED", "false")
        result = evaluate_catalyst_specificity(_make_news_decision())
        assert set(result.keys()) == self.REQUIRED_KEYS

    def test_all_keys_present_when_not_applicable(self):
        decision = {
            "symbol": "AMD",
            "setup_type": "technical_breakout",
            "bias": "LONG",
            "rationale": "Pure technical setup",
            "quantity": 50,
        }
        result = evaluate_catalyst_specificity(decision)
        assert set(result.keys()) == self.REQUIRED_KEYS

    def test_gate_field_is_constant(self):
        result = evaluate_catalyst_specificity(_make_news_decision())
        assert result["gate"] == "catalyst_specificity_gate"

    def test_score_is_integer(self):
        result = evaluate_catalyst_specificity(_make_news_decision())
        assert isinstance(result["score"], int)

    def test_score_bounded_0_10(self):
        result = evaluate_catalyst_specificity(_make_news_decision())
        assert 0 <= result["score"] <= 10

    def test_evidence_is_list(self):
        result = evaluate_catalyst_specificity(_make_news_decision())
        assert isinstance(result["evidence"], list)

    def test_missing_is_list(self):
        result = evaluate_catalyst_specificity(_make_news_decision())
        assert isinstance(result["missing"], list)

    def test_reason_is_string(self):
        result = evaluate_catalyst_specificity(_make_news_decision())
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0


# ---------------------------------------------------------------------------
# Requirement 11.1, 11.2, 11.3: Log-only mode behavior
# ---------------------------------------------------------------------------


class TestLogOnlyMode:
    """Log-only mode returns allow but preserves intended decision."""

    def test_log_only_always_allows(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "log_only")
        # Low-score decision that would normally block
        decision = _make_news_decision(
            catalyst="Semiconductor sector rally",
            catalyst_type="sector_move",
            rationale="Sector momentum",
            indicators={"relative_volume": 0.8},
        )
        result = evaluate_catalyst_specificity(decision, profile="conservative")

        assert result["decision"] == "allow"
        assert result["size_multiplier"] == 1.0

    def test_log_only_preserves_intended_block(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "log_only")
        # Direction mismatch — would block in enforce
        decision = _make_news_decision(
            symbol="TSLA",
            bias="LONG",
            catalyst_type="downgrade",
            catalyst="TSLA downgraded by Goldman, price target cut to $150",
            indicators={"relative_volume": 1.8},
        )
        result = evaluate_catalyst_specificity(decision, profile="conservative")

        assert result["decision"] == "allow"
        assert result["mode"] == "log_only"
        # intended_decision should reflect what enforcement would have done
        assert result["intended_decision"] in ("block", "reduce_size", "warn")

    def test_log_only_preserves_intended_size_multiplier(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "log_only")
        decision = _make_news_decision(
            catalyst="AI chip stocks rally after Lumentum results",
            catalyst_type="sector_move",
            rationale="Semiconductor sector momentum",
            indicators={"relative_volume": 1.2},
        )
        result = evaluate_catalyst_specificity(decision, profile="moderate")

        assert result["size_multiplier"] == 1.0
        # intended_size_multiplier may be < 1.0
        assert isinstance(result["intended_size_multiplier"], float)

    def test_log_only_mode_field_in_result(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "log_only")
        result = evaluate_catalyst_specificity(_make_news_decision())
        assert result["mode"] == "log_only"

    def test_log_only_reason_indicates_would_action(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "log_only")
        # Low score that would block
        decision = _make_news_decision(
            catalyst="Generic market news",
            catalyst_type="sector_move",
            rationale="Broad sector move",
            indicators={},
        )
        result = evaluate_catalyst_specificity(decision, profile="conservative")

        if result["intended_decision"] != "allow":
            assert "log_only" in result["reason"].lower() or "would" in result["reason"].lower()


# ---------------------------------------------------------------------------
# Requirement 11.4: Enforce mode
# ---------------------------------------------------------------------------


class TestEnforceMode:
    """Enforce mode applies the computed decision."""

    def test_enforce_blocks_low_score(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        # Direction mismatch with conservative profile
        decision = _make_news_decision(
            symbol="TSLA",
            bias="LONG",
            catalyst_type="downgrade",
            catalyst="TSLA downgraded by Goldman, price target cut to $150",
            indicators={"relative_volume": 1.8},
        )
        result = evaluate_catalyst_specificity(decision, profile="conservative")

        assert result["mode"] == "enforce"
        # With direction conflict (-3) this should score low
        assert result["score"] <= 5
        assert result["decision"] in ("block", "reduce_size")

    def test_enforce_allows_high_score(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        decision = _make_news_decision(
            catalyst="AMD reports Q2 earnings beat today, raises guidance for AI chips",
            indicators={"relative_volume": 2.5},
        )
        result = evaluate_catalyst_specificity(decision, profile="moderate")

        assert result["mode"] == "enforce"
        assert result["decision"] == "allow"
        assert result["score"] >= 7

    def test_enforce_intended_matches_decision(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        result = evaluate_catalyst_specificity(_make_news_decision())

        assert result["decision"] == result["intended_decision"]
        assert result["size_multiplier"] == result["intended_size_multiplier"]


# ---------------------------------------------------------------------------
# Requirement 12.1, 12.2, 12.3: No LLM calls, deterministic
# ---------------------------------------------------------------------------


class TestDeterministic:
    """Gate is purely deterministic — same inputs produce same outputs."""

    def test_same_input_same_output(self):
        decision = _make_news_decision()
        result1 = evaluate_catalyst_specificity(decision, profile="moderate")
        result2 = evaluate_catalyst_specificity(decision, profile="moderate")

        assert result1 == result2

    def test_no_external_calls(self):
        """Gate uses only data from decision/signal — no network calls."""
        decision = _make_news_decision()
        # If this completes without timeout/error, no external calls were made
        result = evaluate_catalyst_specificity(decision)
        assert result["gate"] == "catalyst_specificity_gate"


# ---------------------------------------------------------------------------
# Quantity before/after (size reduction)
# ---------------------------------------------------------------------------


class TestQuantityTracking:
    """Verify quantity_before and quantity_after are set correctly."""

    def test_quantity_set_on_reduce_size(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        # Sector sympathy in warn range → reduce_size with 0.5 multiplier
        decision = _make_news_decision(
            catalyst="AI chip stocks rally after Lumentum results",
            catalyst_type="sector_move",
            rationale="Semiconductor sector momentum",
            indicators={"relative_volume": 1.6},
            quantity=100,
        )
        result = evaluate_catalyst_specificity(decision, profile="moderate")

        if result["decision"] == "reduce_size":
            assert result["quantity_before"] == 100
            assert result["quantity_after"] is not None
            assert result["quantity_after"] <= 100
            assert result["quantity_after"] >= 1

    def test_quantity_none_when_allow(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        decision = _make_news_decision(
            catalyst="AMD reports Q2 earnings beat today, raises guidance",
            indicators={"relative_volume": 2.5},
        )
        result = evaluate_catalyst_specificity(decision, profile="moderate")

        if result["decision"] == "allow":
            assert result["quantity_before"] is None
            assert result["quantity_after"] is None

    def test_quantity_none_when_no_quantity_in_decision(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        decision = _make_news_decision()
        del decision["quantity"]
        result = evaluate_catalyst_specificity(decision, profile="moderate")

        assert result["quantity_before"] is None
        assert result["quantity_after"] is None

    def test_quantity_none_in_log_only_mode(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "log_only")
        decision = _make_news_decision(quantity=100)
        result = evaluate_catalyst_specificity(decision, profile="moderate")

        # In log_only, decision is always "allow" with multiplier 1.0
        # so quantity_before/after should be None
        assert result["quantity_before"] is None
        assert result["quantity_after"] is None


# ---------------------------------------------------------------------------
# Mode environment variable handling
# ---------------------------------------------------------------------------


class TestModeEnvironment:
    """Test CATALYST_SPECIFICITY_GATE_MODE environment variable handling."""

    def test_default_mode_is_log_only(self):
        """When env var not set, default to log_only."""
        result = evaluate_catalyst_specificity(_make_news_decision())
        assert result["mode"] == "log_only"

    def test_enforce_mode_from_env(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        result = evaluate_catalyst_specificity(_make_news_decision())
        assert result["mode"] == "enforce"

    def test_invalid_mode_defaults_to_log_only(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "invalid_mode")
        result = evaluate_catalyst_specificity(_make_news_decision())
        assert result["mode"] == "log_only"


# ---------------------------------------------------------------------------
# Profile handling
# ---------------------------------------------------------------------------


class TestProfileHandling:
    """Test profile parameter affects thresholds."""

    def test_conservative_profile_higher_threshold(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        decision = _make_news_decision()
        result = evaluate_catalyst_specificity(decision, profile="conservative")
        assert result["threshold"] == 8

    def test_moderate_profile_threshold(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        decision = _make_news_decision()
        result = evaluate_catalyst_specificity(decision, profile="moderate")
        assert result["threshold"] == 7

    def test_aggressive_profile_lower_threshold(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        decision = _make_news_decision()
        result = evaluate_catalyst_specificity(decision, profile="aggressive")
        assert result["threshold"] == 6

    def test_unknown_profile_defaults_to_moderate(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        decision = _make_news_decision()
        result = evaluate_catalyst_specificity(decision, profile="unknown_profile")
        assert result["threshold"] == 7


# ---------------------------------------------------------------------------
# Signal merging
# ---------------------------------------------------------------------------


class TestSignalMerging:
    """Test that signal dict provides fallback context."""

    def test_signal_provides_setup_type(self):
        """Signal setup_type used when decision doesn't have one."""
        decision = {
            "symbol": "AMD",
            "bias": "LONG",
            "catalyst_type": "earnings_beat",
            "catalyst": "AMD earnings beat today",
            "rationale": "Strong earnings",
            "quantity": 50,
        }
        signal = {"setup_type": "news_breakout"}
        result = evaluate_catalyst_specificity(decision, signal=signal)

        # Gate should apply since signal has news_breakout setup_type
        assert result["reason_type"] != "not_applicable"
        assert result["setup_type"] == "news_breakout"

    def test_signal_provides_indicators(self, monkeypatch):
        """Signal indicators used for confirmation scoring."""
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        decision = _make_news_decision(indicators={})
        signal = {"relative_volume": 2.5, "setup_type": "news_breakout"}
        result = evaluate_catalyst_specificity(decision, signal=signal)

        # With high volume from signal, should get confirmation bonus
        assert any("volume" in e.lower() for e in result["evidence"])

    def test_none_signal_handled_gracefully(self):
        """None signal should not cause errors."""
        decision = _make_news_decision()
        result = evaluate_catalyst_specificity(decision, signal=None)
        assert result["gate"] == "catalyst_specificity_gate"


# ---------------------------------------------------------------------------
# Reason string generation
# ---------------------------------------------------------------------------


class TestReasonString:
    """Test human-readable reason string generation."""

    def test_reason_contains_symbol(self):
        result = evaluate_catalyst_specificity(_make_news_decision(symbol="AMD"))
        assert "AMD" in result["reason"]

    def test_reason_contains_score(self):
        result = evaluate_catalyst_specificity(_make_news_decision())
        # Score should appear in reason as "score X/Y"
        assert str(result["score"]) in result["reason"]

    def test_reason_not_empty(self):
        result = evaluate_catalyst_specificity(_make_news_decision())
        assert len(result["reason"]) > 0

    def test_disabled_reason_mentions_disabled(self, monkeypatch):
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_ENABLED", "false")
        result = evaluate_catalyst_specificity(_make_news_decision())
        assert "disabled" in result["reason"].lower()

    def test_not_applicable_reason(self):
        decision = {
            "symbol": "AMD",
            "setup_type": "technical_breakout",
            "bias": "LONG",
            "rationale": "Pure technical",
            "quantity": 50,
        }
        result = evaluate_catalyst_specificity(decision)
        assert "not apply" in result["reason"].lower() or "not_applicable" in result["reason_type"]


# ---------------------------------------------------------------------------
# End-to-end scoring scenarios
# ---------------------------------------------------------------------------


class TestEndToEndScenarios:
    """Integration-style tests verifying full orchestration."""

    def test_direct_symbol_high_score(self, monkeypatch):
        """Direct symbol catalyst with fresh news and volume → high score, allow."""
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        decision = _make_news_decision(
            symbol="AMD",
            catalyst="AMD reports Q2 earnings beat today, raises guidance for AI chips",
            catalyst_type="earnings_beat",
            bias="LONG",
            indicators={"relative_volume": 2.1},
        )
        result = evaluate_catalyst_specificity(decision, profile="moderate")

        assert result["reason_type"] == "direct_symbol"
        assert result["score"] >= 7
        assert result["decision"] == "allow"

    def test_sector_sympathy_lower_score(self, monkeypatch):
        """Sector sympathy catalyst → lower score, may reduce."""
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        decision = _make_news_decision(
            symbol="AMD",
            catalyst="AI chip stocks rally after Lumentum results",
            catalyst_type="sector_move",
            rationale="Semiconductor sector momentum",
            indicators={"relative_volume": 1.2},
        )
        result = evaluate_catalyst_specificity(decision, profile="moderate")

        # Lumentum is in AMD's readthrough relationships
        assert result["score"] < 10
        assert result["reason_type"] in ("named_readthrough", "sector_sympathy")

    def test_direction_mismatch_penalized(self, monkeypatch):
        """Direction conflict → low score, mismatch classification."""
        monkeypatch.setenv("CATALYST_SPECIFICITY_GATE_MODE", "enforce")
        decision = _make_news_decision(
            symbol="TSLA",
            bias="LONG",
            catalyst_type="downgrade",
            catalyst="TSLA downgraded by Goldman, price target cut to $150",
            indicators={"relative_volume": 1.8},
        )
        result = evaluate_catalyst_specificity(decision, profile="moderate")

        assert result["reason_type"] == "mismatch"
        assert result["score"] <= 5

    def test_no_db_does_not_error(self):
        """Passing db=None should not cause errors."""
        result = evaluate_catalyst_specificity(_make_news_decision(), db=None)
        assert result["gate"] == "catalyst_specificity_gate"
