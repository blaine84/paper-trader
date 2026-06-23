"""
Tests for the finance tier in utils/llm.py.

Covers:
- Task 2.1: Finance tier routing in call_llm()
- Task 2.2: _call_ollama_finance() wrapper with fallback
- Task 2.3: Tier-level logging for all tiers
"""

import logging
import os
from unittest.mock import patch, MagicMock

import pytest

from utils.llm import call_llm, _call_ollama_finance


# ---------------------------------------------------------------------------
# Task 2.1: Finance tier routing
# ---------------------------------------------------------------------------

class TestFinanceTierRouting:
    """Verify that tier='finance' routes correctly based on env vars."""

    @patch("utils.llm._call_ollama")
    def test_finance_tier_routes_to_ollama(self, mock_ollama, monkeypatch):
        """When LLM_FINANCE_PROVIDER=ollama, finance tier calls _call_ollama via the wrapper."""
        monkeypatch.setenv("LLM_FINANCE_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_FINANCE_MODEL", "fin-llama3.1:8b-finance")

        mock_ollama.return_value = '{"decisions": []}'

        result = call_llm("system", "user", tier="finance", purpose="test")

        assert result == '{"decisions": []}'
        mock_ollama.assert_called_once()
        # Verify the model passed is the finance model
        call_args = mock_ollama.call_args
        assert call_args[0][2] == "fin-llama3.1:8b-finance"  # model positional arg
        assert call_args[1]["purpose"] == "test"

    @patch("utils.llm._call_ollama")
    def test_finance_tier_fallback_when_provider_not_set(self, mock_ollama, monkeypatch, caplog):
        """When LLM_FINANCE_PROVIDER is not set, falls back to medium tier with WARNING."""
        monkeypatch.delenv("LLM_FINANCE_PROVIDER", raising=False)
        monkeypatch.setenv("LLM_MED_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_MED_MODEL", "llama3.1:8b")

        mock_ollama.return_value = '{"result": "medium"}'

        with caplog.at_level(logging.WARNING, logger="utils.llm"):
            result = call_llm("system", "user", tier="finance", purpose="test")

        assert result == '{"result": "medium"}'
        assert "Finance tier not configured" in caplog.text
        assert "falling back to medium" in caplog.text

    @patch("utils.llm._call_ollama")
    def test_finance_tier_fallback_when_provider_empty(self, mock_ollama, monkeypatch, caplog):
        """When LLM_FINANCE_PROVIDER is empty string, falls back to medium tier."""
        monkeypatch.setenv("LLM_FINANCE_PROVIDER", "")
        monkeypatch.setenv("LLM_MED_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_MED_MODEL", "llama3.1:8b")

        mock_ollama.return_value = '{"result": "medium"}'

        with caplog.at_level(logging.WARNING, logger="utils.llm"):
            result = call_llm("system", "user", tier="finance")

        assert result == '{"result": "medium"}'
        assert "Finance tier not configured" in caplog.text

    @patch("utils.llm._call_ollama")
    def test_finance_tier_reads_env_vars(self, mock_ollama, monkeypatch):
        """Finance tier reads LLM_FINANCE_PROVIDER and LLM_FINANCE_MODEL from env."""
        monkeypatch.setenv("LLM_FINANCE_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_FINANCE_MODEL", "my-custom-finance-model")

        mock_ollama.return_value = '{"ok": true}'

        call_llm("sys", "usr", tier="finance")

        # Model should be the custom one from env
        call_args = mock_ollama.call_args
        assert call_args[0][2] == "my-custom-finance-model"

    @patch("utils.llm._call_ollama")
    def test_finance_tier_passes_purpose(self, mock_ollama, monkeypatch):
        """Finance tier passes purpose parameter through to _call_ollama."""
        monkeypatch.setenv("LLM_FINANCE_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_FINANCE_MODEL", "fin-llama3.1:8b-finance")

        mock_ollama.return_value = '{"ok": true}'

        call_llm("sys", "usr", tier="finance", purpose="quant_research")

        call_args = mock_ollama.call_args
        assert call_args[1]["purpose"] == "quant_research"


# ---------------------------------------------------------------------------
# Task 2.2: _call_ollama_finance() wrapper with fallback
# ---------------------------------------------------------------------------

class TestCallOllamaFinance:
    """Verify _call_ollama_finance handles success and fallback correctly."""

    @patch("utils.llm._call_ollama")
    def test_success_returns_response_and_model(self, mock_ollama, monkeypatch):
        """On success, returns (response, 'ollama', requested_model)."""
        monkeypatch.setenv("OLLAMA_FINANCE_TIMEOUT", "120")
        monkeypatch.setenv("OLLAMA_FINANCE_NUM_CTX", "8192")

        mock_ollama.return_value = '{"analysis": "bullish"}'

        result, provider, model = _call_ollama_finance(
            "system", "user", model="fin-llama3.1:8b-finance", purpose="test"
        )

        assert result == '{"analysis": "bullish"}'
        assert provider == "ollama"
        assert model == "fin-llama3.1:8b-finance"
        # Verify timeout was passed
        assert mock_ollama.call_args[1]["timeout"] == 120
        assert mock_ollama.call_args[1]["num_ctx"] == 8192

    @patch("utils.llm._call_ollama")
    def test_num_ctx_defaults_to_ollama_num_ctx(self, mock_ollama, monkeypatch):
        """Finance tier uses OLLAMA_NUM_CTX when finance-specific context is not set."""
        monkeypatch.delenv("OLLAMA_FINANCE_NUM_CTX", raising=False)
        monkeypatch.setenv("OLLAMA_NUM_CTX", "6144")

        mock_ollama.return_value = '{"ok": true}'

        _call_ollama_finance("system", "user", model="fin-llama", purpose="test")

        assert mock_ollama.call_args[1]["num_ctx"] == 6144

    @patch("utils.llm._call_ollama")
    def test_num_ctx_defaults_to_8192_when_unset(self, mock_ollama, monkeypatch):
        """Finance tier explicitly requests enough context for analyst prompts by default."""
        monkeypatch.delenv("OLLAMA_FINANCE_NUM_CTX", raising=False)
        monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)

        mock_ollama.return_value = '{"ok": true}'

        _call_ollama_finance("system", "user", model="fin-llama", purpose="test")

        assert mock_ollama.call_args[1]["num_ctx"] == 8192

    @patch("utils.llm._call_ollama")
    def test_fallback_on_exception(self, mock_ollama, monkeypatch, caplog):
        """On exception, falls back to medium tier model and logs WARNING."""
        monkeypatch.setenv("LLM_MED_MODEL", "llama3.1:8b")

        # First call (finance model) raises, second call (medium model) succeeds
        mock_ollama.side_effect = [
            ConnectionError("Connection refused"),
            '{"fallback": "response"}',
        ]

        with caplog.at_level(logging.WARNING, logger="utils.llm"):
            result, provider, model = _call_ollama_finance(
                "system", "user", model="fin-llama3.1:8b-finance", purpose="test"
            )

        assert result == '{"fallback": "response"}'
        assert provider == "ollama"
        assert model == "llama3.1:8b"
        assert "Finance tier failed" in caplog.text
        assert "falling back to medium tier" in caplog.text

    @patch("utils.llm._call_ollama")
    def test_timeout_defaults_to_ollama_timeout(self, mock_ollama, monkeypatch):
        """When OLLAMA_FINANCE_TIMEOUT not set, falls back to OLLAMA_TIMEOUT."""
        monkeypatch.delenv("OLLAMA_FINANCE_TIMEOUT", raising=False)
        monkeypatch.setenv("OLLAMA_TIMEOUT", "450")

        mock_ollama.return_value = '{"ok": true}'

        _call_ollama_finance("system", "user", model="fin-llama", purpose="test")

        assert mock_ollama.call_args[1]["timeout"] == 450

    @patch("utils.llm._call_ollama")
    def test_timeout_defaults_to_600_when_nothing_set(self, mock_ollama, monkeypatch):
        """When neither OLLAMA_FINANCE_TIMEOUT nor OLLAMA_TIMEOUT set, defaults to 600."""
        monkeypatch.delenv("OLLAMA_FINANCE_TIMEOUT", raising=False)
        monkeypatch.delenv("OLLAMA_TIMEOUT", raising=False)

        mock_ollama.return_value = '{"ok": true}'

        _call_ollama_finance("system", "user", model="fin-llama", purpose="test")

        assert mock_ollama.call_args[1]["timeout"] == 600

    @patch("utils.llm._call_ollama")
    def test_fallback_uses_llm_low_model_when_med_not_set(self, mock_ollama, monkeypatch, caplog):
        """Fallback uses LLM_LOW_MODEL when LLM_MED_MODEL is not set."""
        monkeypatch.delenv("LLM_MED_MODEL", raising=False)
        monkeypatch.setenv("LLM_LOW_MODEL", "mistral:latest")

        mock_ollama.side_effect = [
            RuntimeError("model not found"),
            '{"fallback": true}',
        ]

        with caplog.at_level(logging.WARNING, logger="utils.llm"):
            result, provider, model = _call_ollama_finance(
                "system", "user", model="fin-llama", purpose="test"
            )

        assert result == '{"fallback": true}'
        assert model == "mistral:latest"


# ---------------------------------------------------------------------------
# Task 2.3: Tier-level logging for all tiers
# ---------------------------------------------------------------------------

class TestTierLevelLogging:
    """Verify that call_llm logs tier, provider, model, and elapsed time for all tiers."""

    @patch("utils.llm._call_ollama")
    def test_finance_tier_logs_info(self, mock_ollama, monkeypatch, caplog):
        """Finance tier logs INFO with tier, provider, model, elapsed."""
        monkeypatch.setenv("LLM_FINANCE_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_FINANCE_MODEL", "fin-llama3.1:8b-finance")

        mock_ollama.return_value = '{"ok": true}'

        with caplog.at_level(logging.INFO, logger="utils.llm"):
            call_llm("sys", "usr", tier="finance")

        assert "LLM call:" in caplog.text
        assert "tier=finance" in caplog.text
        assert "provider=ollama" in caplog.text
        assert "model=fin-llama3.1:8b-finance" in caplog.text
        assert "elapsed=" in caplog.text

    @patch("utils.llm._call_ollama")
    def test_medium_tier_logs_info(self, mock_ollama, monkeypatch, caplog):
        """Medium tier logs INFO with tier, provider, model, elapsed."""
        monkeypatch.setenv("LLM_MED_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_MED_MODEL", "llama3.1:8b")

        mock_ollama.return_value = '{"ok": true}'

        with caplog.at_level(logging.INFO, logger="utils.llm"):
            call_llm("sys", "usr", tier="medium")

        assert "tier=medium" in caplog.text
        assert "provider=ollama" in caplog.text
        assert "model=llama3.1:8b" in caplog.text
        assert "elapsed=" in caplog.text

    @patch("utils.llm._call_ollama")
    def test_low_tier_logs_info(self, mock_ollama, monkeypatch, caplog):
        """Low tier logs INFO with tier, provider, model, elapsed."""
        monkeypatch.setenv("LLM_LOW_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_LOW_MODEL", "mistral:latest")

        mock_ollama.return_value = '{"ok": true}'

        with caplog.at_level(logging.INFO, logger="utils.llm"):
            call_llm("sys", "usr", tier="low")

        assert "tier=low" in caplog.text
        assert "provider=ollama" in caplog.text
        assert "model=mistral:latest" in caplog.text
        assert "elapsed=" in caplog.text

    @patch("utils.llm._call_anthropic")
    def test_high_tier_logs_info(self, mock_anthropic, monkeypatch, caplog):
        """High tier logs INFO with tier, provider, model, elapsed."""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-6")

        mock_anthropic.return_value = '{"ok": true}'

        with caplog.at_level(logging.INFO, logger="utils.llm"):
            call_llm("sys", "usr", tier="high")

        assert "tier=high" in caplog.text
        assert "provider=anthropic" in caplog.text
        assert "elapsed=" in caplog.text

    @patch("utils.llm._call_ollama")
    def test_finance_fallback_logs_actual_model(self, mock_ollama, monkeypatch, caplog):
        """When finance tier falls back, the INFO log reports the ACTUAL model that served."""
        monkeypatch.setenv("LLM_FINANCE_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_FINANCE_MODEL", "fin-llama3.1:8b-finance")
        monkeypatch.setenv("LLM_MED_MODEL", "llama3.1:8b")

        # First call fails (finance model), second succeeds (medium model)
        mock_ollama.side_effect = [
            ConnectionError("Connection refused"),
            '{"fallback": true}',
        ]

        with caplog.at_level(logging.INFO, logger="utils.llm"):
            result = call_llm("sys", "usr", tier="finance")

        assert result == '{"fallback": true}'
        # The INFO log should report the actual model (llama3.1:8b), not the requested one
        info_logs = [r for r in caplog.records if r.levelno == logging.INFO and "LLM call:" in r.message]
        assert len(info_logs) >= 1
        final_log = info_logs[-1].message
        assert "model=llama3.1:8b" in final_log
        assert "tier=finance" in final_log

    @patch("utils.llm._call_ollama")
    def test_elapsed_time_is_positive(self, mock_ollama, monkeypatch, caplog):
        """Elapsed time in the log is a positive number."""
        monkeypatch.setenv("LLM_MED_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_MED_MODEL", "llama3.1:8b")

        mock_ollama.return_value = '{"ok": true}'

        with caplog.at_level(logging.INFO, logger="utils.llm"):
            call_llm("sys", "usr", tier="medium")

        info_logs = [r for r in caplog.records if r.levelno == logging.INFO and "LLM call:" in r.message]
        assert len(info_logs) == 1
        # Extract elapsed value
        msg = info_logs[0].message
        assert "elapsed=" in msg
        elapsed_str = msg.split("elapsed=")[1].split("s")[0]
        elapsed = float(elapsed_str)
        assert elapsed >= 0.0
