"""Tests for price monitor decision window coordination.

Validates that during the PM decision window:
- Monitoring path (prefer_finnhub=False) suppresses Finnhub fallback
- Stop-loss path (prefer_finnhub=True) still calls Finnhub
- Outside the window (None or expired), normal Finnhub fallback applies

**Validates: Requirements 7.1, 7.2, 7.3**
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from agents.price_monitor import get_batch_quotes, _quote_cache, _is_within_decision_window


@pytest.fixture(autouse=True)
def clear_quote_cache():
    """Ensure each test starts with a clean cache."""
    _quote_cache.clear()
    yield
    _quote_cache.clear()


class TestDecisionWindowSuppression:
    """Test Finnhub suppression behavior during the decision window."""

    @patch("agents.price_monitor._get_finnhub_quotes", return_value={})
    @patch("agents.price_monitor._get_yfinance_quotes", return_value={})
    @patch("agents.price_monitor._is_within_decision_window", return_value=True)
    def test_monitoring_path_suppresses_finnhub_during_window(
        self, mock_window, mock_yf, mock_fh
    ):
        """During decision window, monitoring path does NOT call Finnhub.

        **Validates: Requirements 7.2, 7.3**
        """
        get_batch_quotes(["TSLA", "AAPL"], prefer_finnhub=False)
        mock_fh.assert_not_called()

    @patch("agents.price_monitor._get_finnhub_quotes", return_value={"TSLA": 250.0})
    @patch("agents.price_monitor._get_yfinance_quotes", return_value={})
    @patch("agents.price_monitor._is_within_decision_window", return_value=True)
    def test_stop_loss_path_calls_finnhub_during_window(
        self, mock_window, mock_yf, mock_fh
    ):
        """During decision window, stop-loss path (prefer_finnhub=True) STILL calls Finnhub.

        **Validates: Requirements 7.3**
        """
        result = get_batch_quotes(["TSLA"], prefer_finnhub=True)
        mock_fh.assert_called_once()
        assert result.get("TSLA") == 250.0

    @patch("agents.price_monitor._get_finnhub_quotes", return_value={"TSLA": 250.0})
    @patch("agents.price_monitor._get_yfinance_quotes", return_value={})
    @patch("agents.price_monitor._is_within_decision_window", return_value=False)
    def test_monitoring_path_calls_finnhub_outside_window(
        self, mock_window, mock_yf, mock_fh
    ):
        """Outside decision window, monitoring Finnhub fallback works normally.

        **Validates: Requirements 7.6**
        """
        result = get_batch_quotes(["TSLA"], prefer_finnhub=False)
        mock_fh.assert_called_once()
        assert result.get("TSLA") == 250.0

    @patch("agents.price_monitor._get_finnhub_quotes", return_value={"AAPL": 180.0})
    @patch("agents.price_monitor._get_yfinance_quotes", return_value={})
    @patch("agents.price_monitor._is_within_decision_window", return_value=True)
    def test_cache_used_for_watchlist_when_yfinance_empty_during_window(
        self, mock_window, mock_yf, mock_fh
    ):
        """During decision window with yfinance returning nothing, watchlist uses cache only.

        When yfinance is circuit-broken and Finnhub is suppressed for watchlist,
        only cached quotes are available (no blind spot for safety symbols).

        **Validates: Requirements 7.4**
        """
        # No cache entries, yfinance returns nothing, Finnhub suppressed
        result = get_batch_quotes(["AAPL"], prefer_finnhub=False)
        # Finnhub should NOT be called for watchlist during window
        mock_fh.assert_not_called()
        # No price available (cache miss + yfinance empty + Finnhub suppressed)
        assert result.get("AAPL") is None


class TestIsWithinDecisionWindow:
    """Test the _is_within_decision_window() helper itself."""

    @patch("utils.cycle_coordinator.get_decision_window_end", return_value=None)
    def test_returns_false_when_no_window(self, mock_get):
        """When coordinator is disabled (window_end is None), no suppression.

        **Validates: Requirements 7.6**
        """
        # Re-import to use patched module
        from agents.price_monitor import _is_within_decision_window

        assert _is_within_decision_window() is False

    @patch("utils.cycle_coordinator.get_decision_window_end")
    def test_returns_true_when_within_window(self, mock_get):
        """When current time is before window_end, we are inside the window.

        **Validates: Requirements 7.2**
        """
        mock_get.return_value = datetime.now(timezone.utc) + timedelta(seconds=60)
        from agents.price_monitor import _is_within_decision_window

        assert _is_within_decision_window() is True

    @patch("utils.cycle_coordinator.get_decision_window_end")
    def test_returns_false_when_window_expired(self, mock_get):
        """When window_end is in the past, we are outside the window.

        **Validates: Requirements 7.6**
        """
        mock_get.return_value = datetime.now(timezone.utc) - timedelta(seconds=10)
        from agents.price_monitor import _is_within_decision_window

        assert _is_within_decision_window() is False
