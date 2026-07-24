"""Tests for FinnhubClient cycle budget integration.

Validates:
- Requirements 10.3: Cycle-scoped Finnhub budget enforcement
- Requirements 10.4: Budget exhaustion prevents API calls
"""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest

from utils.finnhub_client import FinnhubClient, FinnhubBudgetExhaustedError
from utils.finnhub_budget import CycleFinnhubBudget


@pytest.fixture(autouse=True)
def clear_shared_rate_limit():
    """Clear shared rate-limit state between tests to avoid interference."""
    FinnhubClient._shared_call_times = []
    yield
    FinnhubClient._shared_call_times = []


@pytest.fixture
def mock_finnhub_api():
    """Patch finnhub.Client so no real API calls are made."""
    with patch("utils.finnhub_client.finnhub.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client.quote.return_value = {
            "c": 250.0,
            "o": 245.0,
            "h": 252.0,
            "l": 244.0,
            "pc": 248.0,
        }
        mock_client_class.return_value = mock_client
        yield mock_client


class TestFinnhubBudgetIntegration:
    @patch.dict(os.environ, {"FINNHUB_API_KEY": "test_key"})
    def test_no_budget_operates_normally(self, mock_finnhub_api):
        """Without cycle_budget, FinnhubClient operates as before (backward compat)."""
        client = FinnhubClient()
        result = client.get_quote("TSLA")
        assert result["price"] == 250.0
        mock_finnhub_api.quote.assert_called_once_with("TSLA")

    @patch.dict(os.environ, {"FINNHUB_API_KEY": "test_key"})
    def test_budget_incremented_on_api_call(self, mock_finnhub_api):
        """cycle_budget.increment() is called for each API call."""
        budget = CycleFinnhubBudget(budget=10)
        client = FinnhubClient(cycle_budget=budget)
        client.get_quote("TSLA")
        assert budget.used == 1
        client.get_quote("AAPL")
        assert budget.used == 2

    @patch.dict(os.environ, {"FINNHUB_API_KEY": "test_key"})
    def test_raises_exhausted_error_when_budget_empty(self, mock_finnhub_api):
        """FinnhubBudgetExhaustedError raised when budget is exhausted."""
        budget = CycleFinnhubBudget(budget=2)
        client = FinnhubClient(cycle_budget=budget)
        client.get_quote("TSLA")  # uses 1
        client.get_quote("AAPL")  # uses 2
        with pytest.raises(FinnhubBudgetExhaustedError):
            client.get_quote("MSFT")  # budget exhausted

    @patch.dict(os.environ, {"FINNHUB_API_KEY": "test_key"})
    def test_no_api_call_after_budget_exhaustion(self, mock_finnhub_api):
        """After budget exhaustion, no API calls are made."""
        budget = CycleFinnhubBudget(budget=1)
        client = FinnhubClient(cycle_budget=budget)
        client.get_quote("TSLA")  # uses 1

        with pytest.raises(FinnhubBudgetExhaustedError):
            client.get_quote("AAPL")

        # Only 1 API call should have been made (the first successful one)
        assert mock_finnhub_api.quote.call_count == 1
