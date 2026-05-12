"""
Tests for structured quote context persistence in analyst signals.

Validates that the analyst agent reliably stores:
- current_price, quote_timestamp, day_open, day_high, day_low, prev_close, change_pct
- relative_volume when computable from candle data

Requirements: 4.1, 4.2, 4.4, 14.6
"""

import json
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy import create_engine

from agents.analyst import enrich_signal_with_quote_context
from db.schema import Base, AgentMemory, get_session


@pytest.fixture
def engine():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    return eng


def _mock_quote():
    """Standard mock quote from FinnhubClient.get_quote()."""
    return {
        "symbol": "AMD",
        "price": 155.42,
        "open": 152.10,
        "high": 156.80,
        "low": 151.50,
        "prev_close": 151.00,
        "change_pct": 2.93,
        "timestamp": "2025-01-15T14:30:00",
    }


def _mock_candles(volume_pattern="normal"):
    """Mock candle data with configurable volume patterns."""
    base_volumes = [100000, 120000, 110000, 105000, 115000,
                    108000, 112000, 109000, 111000, 107000,
                    113000, 106000, 114000, 110000, 108000,
                    112000, 109000, 111000, 107000, 113000]
    if volume_pattern == "high":
        # Last candle has 2x average volume
        avg = sum(base_volumes) / len(base_volumes)
        current_vol = int(avg * 2)
        base_volumes.append(current_vol)
    elif volume_pattern == "low":
        # Last candle has 0.5x average volume
        avg = sum(base_volumes) / len(base_volumes)
        current_vol = int(avg * 0.5)
        base_volumes.append(current_vol)
    else:
        # Normal: last candle is roughly average
        base_volumes.append(110000)

    return {
        "symbol": "AMD",
        "resolution": "5",
        "timestamps": list(range(len(base_volumes))),
        "open": [150.0] * len(base_volumes),
        "high": [155.0] * len(base_volumes),
        "low": [149.0] * len(base_volumes),
        "close": [153.0] * len(base_volumes),
        "volume": base_volumes,
    }


# --- Unit tests for enrich_signal_with_quote_context ---


class TestEnrichSignalWithQuoteContext:
    """Unit tests for the enrich_signal_with_quote_context helper."""

    def test_persists_all_required_quote_fields(self):
        """Signal must persist current_price, quote_timestamp, day_open,
        day_high, day_low, prev_close, change_pct."""
        signal = {"symbol": "AMD", "signal": "LONG"}
        quote = _mock_quote()
        candles = _mock_candles()

        result = enrich_signal_with_quote_context(signal, quote, candles)

        assert result["current_price"] == 155.42
        assert result["quote_timestamp"] == "2025-01-15T14:30:00"
        assert result["day_open"] == 152.10
        assert result["day_high"] == 156.80
        assert result["day_low"] == 151.50
        assert result["prev_close"] == 151.00
        assert result["change_pct"] == 2.93

    def test_computes_relative_volume_from_candles(self):
        """Signal must persist relative_volume computed from candle data."""
        signal = {"symbol": "AMD", "signal": "LONG"}
        quote = _mock_quote()
        candles = _mock_candles(volume_pattern="high")

        result = enrich_signal_with_quote_context(signal, quote, candles)

        assert "relative_volume" in result
        # High volume pattern = 2x average
        assert result["relative_volume"] >= 1.9
        assert result["relative_volume"] <= 2.1

    def test_low_relative_volume(self):
        """Low volume should produce relative_volume < 1.0."""
        signal = {"symbol": "AMD", "signal": "LONG"}
        quote = _mock_quote()
        candles = _mock_candles(volume_pattern="low")

        result = enrich_signal_with_quote_context(signal, quote, candles)

        assert "relative_volume" in result
        assert result["relative_volume"] < 1.0

    def test_handles_empty_quote_gracefully(self):
        """When quote is empty dict, quote fields should be None."""
        signal = {"symbol": "AMD", "signal": "LONG"}
        quote = {}
        candles = _mock_candles()

        result = enrich_signal_with_quote_context(signal, quote, candles)

        assert result.get("current_price") is None
        assert result.get("quote_timestamp") is None
        assert result.get("day_open") is None
        assert result.get("day_high") is None
        assert result.get("day_low") is None
        assert result.get("prev_close") is None
        assert result.get("change_pct") is None

    def test_handles_none_quote_gracefully(self):
        """When quote is None, no quote fields should be added."""
        signal = {"symbol": "AMD", "signal": "LONG"}
        candles = _mock_candles()

        result = enrich_signal_with_quote_context(signal, None, candles)

        assert "current_price" not in result
        assert "quote_timestamp" not in result

    def test_handles_empty_candles_gracefully(self):
        """When candles is empty, relative_volume should not be set."""
        signal = {"symbol": "AMD", "signal": "LONG"}
        quote = _mock_quote()

        result = enrich_signal_with_quote_context(signal, quote, {})

        # Quote fields should still be present
        assert result["current_price"] == 155.42
        # But relative_volume should not be set
        assert "relative_volume" not in result

    def test_handles_none_candles_gracefully(self):
        """When candles is None, relative_volume should not be set."""
        signal = {"symbol": "AMD", "signal": "LONG"}
        quote = _mock_quote()

        result = enrich_signal_with_quote_context(signal, quote, None)

        assert result["current_price"] == 155.42
        assert "relative_volume" not in result

    def test_handles_single_candle_no_relative_volume(self):
        """With only one candle, can't compute relative volume."""
        signal = {"symbol": "AMD", "signal": "LONG"}
        quote = _mock_quote()
        candles = {"volume": [100000]}  # Only one candle

        result = enrich_signal_with_quote_context(signal, quote, candles)

        assert "relative_volume" not in result

    def test_handles_zero_average_volume(self):
        """If all prior volumes are 0, relative_volume should not be set."""
        signal = {"symbol": "AMD", "signal": "LONG"}
        quote = _mock_quote()
        candles = {"volume": [0, 0, 0, 100000]}

        result = enrich_signal_with_quote_context(signal, quote, candles)

        # avg_vol of prior candles is 0, so we skip
        assert "relative_volume" not in result

    def test_modifies_signal_in_place(self):
        """The function should modify the signal dict in place and return it."""
        signal = {"symbol": "AMD", "signal": "LONG"}
        quote = _mock_quote()
        candles = _mock_candles()

        result = enrich_signal_with_quote_context(signal, quote, candles)

        assert result is signal  # Same reference

    def test_does_not_overwrite_existing_signal_fields(self):
        """Quote enrichment should not overwrite existing LLM signal fields."""
        signal = {
            "symbol": "AMD",
            "signal": "LONG",
            "strength": "strong",
            "confidence": "high",
            "setup_type": "news_catalyst",
        }
        quote = _mock_quote()
        candles = _mock_candles()

        result = enrich_signal_with_quote_context(signal, quote, candles)

        # Original fields preserved
        assert result["signal"] == "LONG"
        assert result["strength"] == "strong"
        assert result["confidence"] == "high"
        assert result["setup_type"] == "news_catalyst"
        # New fields added
        assert result["current_price"] == 155.42


# --- Integration test: verify signal is persisted to AgentMemory with quote fields ---


@patch("agents.analyst.FinnhubClient")
@patch("agents.analyst.call_llm")
@patch("agents.analyst.compute_indicators")
@patch("agents.analyst.get_relevant_cases")
@patch("agents.analyst.format_cases_for_prompt")
@patch("agents.analyst.build_strategy_context")
@patch("agents.analyst.build_feedback_prompt_context")
@patch("agents.analyst.get_active_mitigations")
@patch("agents.analyst.process_pending_feedback")
@patch("agents.analyst.write_feedback_health_status")
@patch("agents.analyst.get_breaking_news_for_symbols")
@patch("agents.analyst.validate_setup_for_symbol")
@patch("agents.analyst.compute_freshness_state")
@patch("utils.strategy_store.get_all_setup_types")
def test_integration_signal_persisted_with_quote_fields(
    mock_setup_types,
    mock_freshness,
    mock_validate_setup,
    mock_breaking_news,
    mock_write_health,
    mock_process_feedback,
    mock_get_mitigations,
    mock_feedback_ctx,
    mock_strategy_ctx,
    mock_format_cases,
    mock_cases,
    mock_indicators,
    mock_llm,
    mock_fh_class,
    engine,
):
    """Integration: verify the full run() persists quote fields to AgentMemory."""
    mock_fh = MagicMock()
    mock_fh_class.return_value = mock_fh
    mock_fh.get_quote.return_value = _mock_quote()
    mock_fh.get_candles.return_value = _mock_candles(volume_pattern="high")
    mock_indicators.return_value = {"price": 155.42, "trend": "bullish"}
    mock_llm.return_value = json.dumps({
        "symbol": "AMD",
        "signal": "LONG",
        "strength": "strong",
        "confidence": "high",
        "setup_type": "news_catalyst",
        "setup_reasoning": "Strong earnings beat",
        "reasoning": "AMD bullish momentum",
        "key_levels": {"support": 150.0, "resistance": 160.0, "vwap": 153.5},
        "invalidation": "price closes below 150",
        "indicators": {"rsi": 62.5, "macd_bias": "bullish"},
    })
    mock_cases.return_value = []
    mock_format_cases.return_value = ""
    mock_strategy_ctx.return_value = ""
    mock_feedback_ctx.return_value = ""
    mock_get_mitigations.return_value = {}
    mock_breaking_news.return_value = {}
    mock_validate_setup.return_value = {}
    mock_freshness.return_value = "fresh"
    mock_setup_types.return_value = ["news_catalyst", "gap_and_go", "news_breakout"]

    from agents.analyst import run
    result = run(engine, ["AMD"])

    # Verify the returned signal has structured quote fields
    signal = result["AMD"]
    assert signal["current_price"] == 155.42
    assert signal["quote_timestamp"] == "2025-01-15T14:30:00"
    assert signal["day_open"] == 152.10
    assert signal["day_high"] == 156.80
    assert signal["day_low"] == 151.50
    assert signal["prev_close"] == 151.00
    assert signal["change_pct"] == 2.93
    assert signal["relative_volume"] >= 1.9

    # Verify persisted to AgentMemory
    db = get_session(engine)
    mem = (
        db.query(AgentMemory)
        .filter_by(agent="analyst", symbol="AMD", key="signal")
        .first()
    )
    assert mem is not None
    stored = json.loads(mem.value)
    assert stored["current_price"] == 155.42
    assert stored["quote_timestamp"] == "2025-01-15T14:30:00"
    assert stored["day_open"] == 152.10
    assert stored["day_high"] == 156.80
    assert stored["day_low"] == 151.50
    assert stored["prev_close"] == 151.00
    assert stored["change_pct"] == 2.93
    assert stored["relative_volume"] >= 1.9
    db.close()
