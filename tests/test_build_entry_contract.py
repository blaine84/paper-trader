"""
Unit tests for build_entry_contract in agents/portfolio_manager.py.
Validates Requirements 1.1, 1.3, 1.4.
"""

import logging
from agents.portfolio_manager import build_entry_contract


class TestBuildEntryContract:
    """Tests for the build_entry_contract function."""

    def test_basic_with_structured_invalidation(self):
        """Signal with structured invalidation produces correct contract."""
        decision = {
            "action": "BUY",
            "symbol": "AMD",
            "rationale": "Gap-and-go with VWAP support",
        }
        signal = {
            "bias": "LONG",
            "confidence": "high",
            "setup_type": "gap_and_go",
            "key_levels": {"support": 160.0, "resistance": 170.0},
            "invalidation": [
                {
                    "type": "price_below_level",
                    "reference": "VWAP",
                    "confirmation": "5m_close",
                    "lookback_bars": 1,
                },
                {
                    "type": "price_below_level",
                    "reference": "162.50",
                    "confirmation": "tick",
                    "lookback_bars": 0,
                },
            ],
        }
        result = build_entry_contract(decision, signal, stop=160.0, target=170.0)

        assert "Gap-and-go with VWAP support" in result["thesis"]
        assert "Signal context:" in result["thesis"]
        assert result["setup_type"] == "gap_and_go"
        assert len(result["invalidators"]) == 2
        assert result["invalidators"][0]["reference"] == "VWAP"
        assert result["invalidators"][1]["reference"] == "162.50"

    def test_single_dict_invalidation(self):
        """Signal with a single invalidation dict (not a list) is handled."""
        decision = {"rationale": "VWAP reclaim play"}
        signal = {
            "setup_type": "vwap_reclaim",
            "invalidation": {
                "type": "price_below_level",
                "reference": "161.00",
                "confirmation": "5m_close",
                "lookback_bars": 1,
            },
        }
        result = build_entry_contract(decision, signal, stop=160.0, target=170.0)

        assert result["setup_type"] == "vwap_reclaim"
        assert len(result["invalidators"]) == 1
        assert result["invalidators"][0]["reference"] == "161.00"

    def test_fallback_when_no_invalidation_field(self, caplog):
        """Missing invalidation field falls back to stop-price default and logs warning."""
        decision = {"rationale": "Momentum breakout"}
        signal = {"setup_type": "breakout", "bias": "LONG"}

        with caplog.at_level(logging.WARNING):
            result = build_entry_contract(decision, signal, stop=155.50, target=170.0)

        assert len(result["invalidators"]) == 1
        inv = result["invalidators"][0]
        assert inv["type"] == "price_below_level"
        assert inv["reference"] == "155.5"
        assert inv["confirmation"] == "5m_close"
        assert inv["lookback_bars"] == 1
        assert "stop-price default" in caplog.text.lower() or "lacks invalidation" in caplog.text.lower()

    def test_fallback_when_invalidation_is_unparseable_string(self, caplog):
        """String invalidation field that can't be parsed falls back to default."""
        decision = {"rationale": "Some trade"}
        signal = {
            "setup_type": "ema_bounce",
            "invalidation": "Below VWAP on 5m close",
        }

        with caplog.at_level(logging.WARNING):
            result = build_entry_contract(decision, signal, stop=100.0, target=110.0)

        assert len(result["invalidators"]) == 1
        assert result["invalidators"][0]["reference"] == "100.0"

    def test_thesis_combines_rationale_and_signal_context(self):
        """Thesis includes both rationale and signal context."""
        decision = {"rationale": "Strong momentum play"}
        signal = {"bias": "LONG", "confidence": "high", "setup_type": "momentum"}

        result = build_entry_contract(decision, signal, stop=50.0, target=60.0)

        assert "Strong momentum play" in result["thesis"]
        assert "Bias: LONG" in result["thesis"]
        assert "Confidence: high" in result["thesis"]

    def test_thesis_with_no_rationale(self):
        """Thesis falls back to signal context when rationale is missing."""
        decision = {}
        signal = {"bias": "SHORT", "confidence": "medium"}

        result = build_entry_contract(decision, signal, stop=50.0, target=40.0)

        assert "Bias: SHORT" in result["thesis"]
        assert result["thesis"] != ""

    def test_thesis_with_no_signal_context(self):
        """Thesis uses rationale alone when signal has no context fields."""
        decision = {"rationale": "Pure price action"}
        signal = {}

        result = build_entry_contract(decision, signal, stop=50.0, target=60.0)

        assert result["thesis"] == "Pure price action"

    def test_thesis_with_nothing(self):
        """Thesis has a fallback when both rationale and signal are empty."""
        decision = {}
        signal = {}

        result = build_entry_contract(decision, signal, stop=50.0, target=60.0)

        assert result["thesis"] == "No thesis recorded"

    def test_setup_type_from_signal(self):
        """setup_type is extracted from signal first."""
        decision = {"setup_type": "from_decision"}
        signal = {"setup_type": "from_signal"}

        result = build_entry_contract(decision, signal, stop=50.0, target=60.0)
        assert result["setup_type"] == "from_signal"

    def test_setup_type_fallback_to_decision(self):
        """setup_type falls back to decision when signal lacks it."""
        decision = {"setup_type": "from_decision"}
        signal = {}

        result = build_entry_contract(decision, signal, stop=50.0, target=60.0)
        assert result["setup_type"] == "from_decision"

    def test_setup_type_fallback_to_unknown(self):
        """setup_type defaults to 'unknown' when neither has it."""
        decision = {}
        signal = {}

        result = build_entry_contract(decision, signal, stop=50.0, target=60.0)
        assert result["setup_type"] == "unknown"

    def test_invalidator_with_invalid_type_is_skipped(self, caplog):
        """Invalidators with invalid type are skipped, falls back to default."""
        decision = {"rationale": "test"}
        signal = {
            "invalidation": [
                {"type": "invalid_type", "reference": "100", "confirmation": "tick", "lookback_bars": 0},
            ],
        }

        with caplog.at_level(logging.WARNING):
            result = build_entry_contract(decision, signal, stop=95.0, target=110.0)

        # Invalid type skipped, falls back to stop-price default
        assert len(result["invalidators"]) == 1
        assert result["invalidators"][0]["reference"] == "95.0"

    def test_invalidator_confirmation_defaults_to_5m_close(self):
        """Invalid confirmation value defaults to 5m_close."""
        decision = {"rationale": "test"}
        signal = {
            "invalidation": {
                "type": "price_below_level",
                "reference": "100",
                "confirmation": "invalid_conf",
                "lookback_bars": 0,
            },
        }

        result = build_entry_contract(decision, signal, stop=95.0, target=110.0)

        assert result["invalidators"][0]["confirmation"] == "5m_close"

    def test_invalidator_negative_lookback_clamped_to_zero(self):
        """Negative lookback_bars is clamped to 0."""
        decision = {"rationale": "test"}
        signal = {
            "invalidation": {
                "type": "price_above_level",
                "reference": "200",
                "confirmation": "tick",
                "lookback_bars": -5,
            },
        }

        result = build_entry_contract(decision, signal, stop=210.0, target=190.0)

        assert result["invalidators"][0]["lookback_bars"] == 0

    def test_return_structure(self):
        """Return value has exactly the expected keys."""
        result = build_entry_contract(
            {"rationale": "test"}, {"setup_type": "test"}, stop=100.0, target=110.0
        )

        assert set(result.keys()) == {"thesis", "setup_type", "invalidators"}
        assert isinstance(result["thesis"], str)
        assert isinstance(result["setup_type"], str)
        assert isinstance(result["invalidators"], list)
        assert len(result["invalidators"]) > 0
