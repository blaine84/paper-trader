"""Tests for ad-hoc replay operator controls (Task 16.2).

Tests cover:
- Ad-hoc replay for specific candidate, date range, symbol, setup_type, profile, or gate
- Operator override flag required for market-hours ad-hoc replay
- Configurable resource limit during market hours (default: max 10 candidates)
- Structured failure reporting with reason codes

Requirements: 10.4, 10.7, 13.5
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import patch, MagicMock

import pytest

from agents.decision_replay import (
    run,
    _is_market_hours,
    _classify_failure,
    DEFAULT_MARKET_HOURS_MAX_CANDIDATES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_engine():
    """Mock SQLAlchemy engine."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Tests: _is_market_hours()
# ---------------------------------------------------------------------------


class TestIsMarketHours:
    """Tests for market-hours detection logic."""

    def test_during_market_hours_trading_day(self):
        """10:00 AM ET on a Monday should be market hours."""
        # Monday 2024-01-08 10:00 AM ET
        now_et = datetime(2024, 1, 8, 10, 0, 0)
        with patch("agents.decision_replay.is_trading_day", return_value=True):
            assert _is_market_hours(now_et) is True

    def test_before_market_open(self):
        """9:00 AM ET (before 9:30) should not be market hours."""
        now_et = datetime(2024, 1, 8, 9, 0, 0)
        with patch("agents.decision_replay.is_trading_day", return_value=True):
            assert _is_market_hours(now_et) is False

    def test_at_market_open(self):
        """Exactly 9:30 AM ET should be market hours."""
        now_et = datetime(2024, 1, 8, 9, 30, 0)
        with patch("agents.decision_replay.is_trading_day", return_value=True):
            assert _is_market_hours(now_et) is True

    def test_at_market_close(self):
        """Exactly 4:00 PM ET should NOT be market hours (exclusive)."""
        now_et = datetime(2024, 1, 8, 16, 0, 0)
        with patch("agents.decision_replay.is_trading_day", return_value=True):
            assert _is_market_hours(now_et) is False

    def test_after_market_close(self):
        """5:00 PM ET should not be market hours."""
        now_et = datetime(2024, 1, 8, 17, 0, 0)
        with patch("agents.decision_replay.is_trading_day", return_value=True):
            assert _is_market_hours(now_et) is False

    def test_weekend_midday(self):
        """Saturday at noon should not be market hours."""
        now_et = datetime(2024, 1, 6, 12, 0, 0)  # Saturday
        with patch("agents.decision_replay.is_trading_day", return_value=False):
            assert _is_market_hours(now_et) is False

    def test_holiday(self):
        """Market holiday during normal hours should not be market hours."""
        now_et = datetime(2024, 1, 15, 10, 0, 0)  # MLK Day
        with patch("agents.decision_replay.is_trading_day", return_value=False):
            assert _is_market_hours(now_et) is False


# ---------------------------------------------------------------------------
# Tests: Ad-hoc replay market-hours enforcement
# ---------------------------------------------------------------------------


class TestAdHocMarketHoursEnforcement:
    """Tests for Requirement 10.4: operator override and resource limit enforcement."""

    def test_adhoc_during_market_hours_without_override_raises(self, mock_engine):
        """Ad-hoc replay during market hours without operator_override raises ValueError."""
        with patch("agents.decision_replay._is_market_hours", return_value=True):
            with pytest.raises(ValueError, match="operator_override=True"):
                run(
                    mock_engine,
                    mode="adhoc",
                    operator_override=False,
                )

    def test_adhoc_outside_market_hours_no_override_needed(self, mock_engine):
        """Ad-hoc replay outside market hours does not require operator_override."""
        with (
            patch("agents.decision_replay._is_market_hours", return_value=False),
            patch("agents.decision_replay.init_replay_db"),
            patch("agents.decision_replay._build_policy") as mock_policy,
            patch("agents.decision_replay.get_session") as mock_session,
            patch("agents.decision_replay.load_candidates", return_value=[]),
            patch("agents.decision_replay.correlate_and_deduplicate", return_value=[]),
            patch("agents.decision_replay._persist_batch_run"),
        ):
            mock_pv = MagicMock()
            mock_policy.return_value = (mock_pv, MagicMock())
            mock_session.return_value = MagicMock()

            # Should not raise
            summary = run(mock_engine, mode="adhoc", operator_override=False)
            assert summary.mode == "adhoc"

    def test_adhoc_during_market_hours_with_override_proceeds(self, mock_engine):
        """Ad-hoc replay during market hours with operator_override=True proceeds."""
        with (
            patch("agents.decision_replay._is_market_hours", return_value=True),
            patch("agents.decision_replay.init_replay_db"),
            patch("agents.decision_replay._build_policy") as mock_policy,
            patch("agents.decision_replay.get_session") as mock_session,
            patch("agents.decision_replay.load_candidates", return_value=[]),
            patch("agents.decision_replay.correlate_and_deduplicate", return_value=[]),
            patch("agents.decision_replay._persist_batch_run"),
        ):
            mock_pv = MagicMock()
            mock_policy.return_value = (mock_pv, MagicMock())
            mock_session.return_value = MagicMock()

            summary = run(mock_engine, mode="adhoc", operator_override=True)
            assert summary.mode == "adhoc"

    def test_adhoc_market_hours_enforces_default_max_candidates(self, mock_engine):
        """During market hours with override, max_candidates defaults to 10."""
        mock_candidates = [MagicMock() for _ in range(20)]
        for i, c in enumerate(mock_candidates):
            c.candidate_id = f"cand_{i}"

        with (
            patch("agents.decision_replay._is_market_hours", return_value=True),
            patch("agents.decision_replay.init_replay_db"),
            patch("agents.decision_replay._build_policy") as mock_policy,
            patch("agents.decision_replay.get_session") as mock_session,
            patch("agents.decision_replay.load_candidates", return_value=mock_candidates),
            patch("agents.decision_replay.correlate_and_deduplicate", return_value=mock_candidates),
            patch("agents.decision_replay._process_candidate") as mock_process,
            patch("agents.decision_replay._persist_batch_run"),
        ):
            mock_pv = MagicMock()
            mock_policy.return_value = (mock_pv, MagicMock())
            mock_session.return_value = MagicMock()

            summary = run(mock_engine, mode="adhoc", operator_override=True)
            # Should only process 10 candidates (default limit)
            assert summary.candidates_total == DEFAULT_MARKET_HOURS_MAX_CANDIDATES
            assert mock_process.call_count == DEFAULT_MARKET_HOURS_MAX_CANDIDATES

    def test_adhoc_market_hours_respects_lower_user_max(self, mock_engine):
        """If user provides max_candidates < market_hours limit, use user's value."""
        mock_candidates = [MagicMock() for _ in range(20)]
        for i, c in enumerate(mock_candidates):
            c.candidate_id = f"cand_{i}"

        with (
            patch("agents.decision_replay._is_market_hours", return_value=True),
            patch("agents.decision_replay.init_replay_db"),
            patch("agents.decision_replay._build_policy") as mock_policy,
            patch("agents.decision_replay.get_session") as mock_session,
            patch("agents.decision_replay.load_candidates", return_value=mock_candidates),
            patch("agents.decision_replay.correlate_and_deduplicate", return_value=mock_candidates),
            patch("agents.decision_replay._process_candidate") as mock_process,
            patch("agents.decision_replay._persist_batch_run"),
        ):
            mock_pv = MagicMock()
            mock_policy.return_value = (mock_pv, MagicMock())
            mock_session.return_value = MagicMock()

            summary = run(
                mock_engine,
                mode="adhoc",
                operator_override=True,
                max_candidates=5,
            )
            # Should use min(5, 10) = 5
            assert summary.candidates_total == 5
            assert mock_process.call_count == 5

    def test_adhoc_market_hours_caps_at_market_limit(self, mock_engine):
        """If user provides max_candidates > market_hours limit, cap at market limit."""
        mock_candidates = [MagicMock() for _ in range(20)]
        for i, c in enumerate(mock_candidates):
            c.candidate_id = f"cand_{i}"

        with (
            patch("agents.decision_replay._is_market_hours", return_value=True),
            patch("agents.decision_replay.init_replay_db"),
            patch("agents.decision_replay._build_policy") as mock_policy,
            patch("agents.decision_replay.get_session") as mock_session,
            patch("agents.decision_replay.load_candidates", return_value=mock_candidates),
            patch("agents.decision_replay.correlate_and_deduplicate", return_value=mock_candidates),
            patch("agents.decision_replay._process_candidate") as mock_process,
            patch("agents.decision_replay._persist_batch_run"),
        ):
            mock_pv = MagicMock()
            mock_policy.return_value = (mock_pv, MagicMock())
            mock_session.return_value = MagicMock()

            summary = run(
                mock_engine,
                mode="adhoc",
                operator_override=True,
                max_candidates=50,  # Exceeds market-hours limit
            )
            # Should be capped at 10 (default market-hours limit)
            assert summary.candidates_total == DEFAULT_MARKET_HOURS_MAX_CANDIDATES

    def test_adhoc_market_hours_custom_limit(self, mock_engine):
        """Custom market_hours_max_candidates is respected."""
        mock_candidates = [MagicMock() for _ in range(20)]
        for i, c in enumerate(mock_candidates):
            c.candidate_id = f"cand_{i}"

        with (
            patch("agents.decision_replay._is_market_hours", return_value=True),
            patch("agents.decision_replay.init_replay_db"),
            patch("agents.decision_replay._build_policy") as mock_policy,
            patch("agents.decision_replay.get_session") as mock_session,
            patch("agents.decision_replay.load_candidates", return_value=mock_candidates),
            patch("agents.decision_replay.correlate_and_deduplicate", return_value=mock_candidates),
            patch("agents.decision_replay._process_candidate") as mock_process,
            patch("agents.decision_replay._persist_batch_run"),
        ):
            mock_pv = MagicMock()
            mock_policy.return_value = (mock_pv, MagicMock())
            mock_session.return_value = MagicMock()

            summary = run(
                mock_engine,
                mode="adhoc",
                operator_override=True,
                market_hours_max_candidates=3,
            )
            assert summary.candidates_total == 3

    def test_batch_mode_not_affected_by_market_hours(self, mock_engine):
        """Batch mode does not require operator_override even during market hours."""
        with (
            patch("agents.decision_replay._is_market_hours", return_value=True),
            patch("agents.decision_replay.init_replay_db"),
            patch("agents.decision_replay._build_policy") as mock_policy,
            patch("agents.decision_replay.get_session") as mock_session,
            patch("agents.decision_replay.load_candidates", return_value=[]),
            patch("agents.decision_replay.correlate_and_deduplicate", return_value=[]),
            patch("agents.decision_replay._persist_batch_run"),
        ):
            mock_pv = MagicMock()
            mock_policy.return_value = (mock_pv, MagicMock())
            mock_session.return_value = MagicMock()

            # Batch mode should not raise even during market hours
            summary = run(mock_engine, mode="batch")
            assert summary.mode == "batch"


# ---------------------------------------------------------------------------
# Tests: Ad-hoc filtering support
# ---------------------------------------------------------------------------


class TestAdHocFiltering:
    """Tests for ad-hoc replay supporting specific filters."""

    def test_adhoc_filters_by_candidate_ids(self, mock_engine):
        """Ad-hoc replay filters by specific candidate IDs."""
        cand1 = MagicMock()
        cand1.candidate_id = "cand_1"
        cand2 = MagicMock()
        cand2.candidate_id = "cand_2"
        cand3 = MagicMock()
        cand3.candidate_id = "cand_3"

        with (
            patch("agents.decision_replay._is_market_hours", return_value=False),
            patch("agents.decision_replay.init_replay_db"),
            patch("agents.decision_replay._build_policy") as mock_policy,
            patch("agents.decision_replay.get_session") as mock_session,
            patch("agents.decision_replay.load_candidates", return_value=[cand1, cand2, cand3]),
            patch("agents.decision_replay.correlate_and_deduplicate", return_value=[cand1, cand2, cand3]),
            patch("agents.decision_replay._process_candidate") as mock_process,
            patch("agents.decision_replay._persist_batch_run"),
        ):
            mock_pv = MagicMock()
            mock_policy.return_value = (mock_pv, MagicMock())
            mock_session.return_value = MagicMock()

            summary = run(
                mock_engine,
                mode="adhoc",
                candidate_ids=["cand_1", "cand_3"],
            )
            # Should only process cand_1 and cand_3
            assert summary.candidates_total == 2

    def test_adhoc_passes_filters_to_load_candidates(self, mock_engine):
        """Ad-hoc replay passes filter dict (symbol, setup_type, etc.) to load_candidates."""
        filters = {"symbol": "TSLA", "setup_type": "news_breakout", "profile": "aggressive"}

        with (
            patch("agents.decision_replay._is_market_hours", return_value=False),
            patch("agents.decision_replay.init_replay_db"),
            patch("agents.decision_replay._build_policy") as mock_policy,
            patch("agents.decision_replay.get_session") as mock_session,
            patch("agents.decision_replay.load_candidates", return_value=[]) as mock_load,
            patch("agents.decision_replay.correlate_and_deduplicate", return_value=[]),
            patch("agents.decision_replay._persist_batch_run"),
        ):
            mock_pv = MagicMock()
            mock_policy.return_value = (mock_pv, MagicMock())
            mock_session.return_value = MagicMock()

            run(mock_engine, mode="adhoc", filters=filters)

            # Verify filters were passed to load_candidates
            mock_load.assert_called_once()
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs["filters"] == filters

    def test_adhoc_passes_date_range(self, mock_engine):
        """Ad-hoc replay passes date_range to load_candidates."""
        dr = (datetime(2024, 1, 1), datetime(2024, 1, 31))

        with (
            patch("agents.decision_replay._is_market_hours", return_value=False),
            patch("agents.decision_replay.init_replay_db"),
            patch("agents.decision_replay._build_policy") as mock_policy,
            patch("agents.decision_replay.get_session") as mock_session,
            patch("agents.decision_replay.load_candidates", return_value=[]) as mock_load,
            patch("agents.decision_replay.correlate_and_deduplicate", return_value=[]),
            patch("agents.decision_replay._persist_batch_run"),
        ):
            mock_pv = MagicMock()
            mock_policy.return_value = (mock_pv, MagicMock())
            mock_session.return_value = MagicMock()

            run(mock_engine, mode="adhoc", date_range=dr)

            mock_load.assert_called_once()
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs["date_range"] == dr


# ---------------------------------------------------------------------------
# Tests: Structured failure reporting
# ---------------------------------------------------------------------------


class TestClassifyFailure:
    """Tests for _classify_failure() structured reason codes (Requirement 13.5)."""

    def test_missing_input_from_message(self):
        """Exception with 'missing' in message → missing_input."""
        exc = RuntimeError("Signal strength is missing for candidate")
        assert _classify_failure(exc) == "missing_input"

    def test_missing_input_from_unavailable(self):
        """Exception with 'unavailable' in message → missing_input."""
        exc = RuntimeError("ATR data unavailable at cutoff time")
        assert _classify_failure(exc) == "missing_input"

    def test_policy_not_found(self):
        """Exception with 'policy' in message → policy_not_found."""
        exc = ValueError("Policy version v2.1 cannot be reconstructed")
        assert _classify_failure(exc) == "policy_not_found"

    def test_policy_not_found_from_not_found(self):
        """Exception mentioning 'policy' and 'not found' → policy_not_found."""
        exc = ValueError("Policy configuration not found in snapshot")
        assert _classify_failure(exc) == "policy_not_found"

    def test_snapshot_corrupt(self):
        """Exception with 'snapshot' and 'corrupt' → snapshot_corrupt."""
        exc = RuntimeError("Snapshot data is corrupt: missing required fields")
        assert _classify_failure(exc) == "snapshot_corrupt"

    def test_snapshot_integrity_error(self):
        """Exception with 'integrity' → snapshot_corrupt."""
        exc = RuntimeError("Integrity check failed for replay data")
        assert _classify_failure(exc) == "snapshot_corrupt"

    def test_snapshot_malformed(self):
        """Exception with 'snapshot' and 'malformed' → snapshot_corrupt."""
        exc = RuntimeError("Snapshot JSON is malformed")
        assert _classify_failure(exc) == "snapshot_corrupt"

    def test_timeout_from_message(self):
        """Exception with 'timeout' in message → timeout."""
        exc = RuntimeError("Operation timeout exceeded")
        assert _classify_failure(exc) == "timeout"

    def test_timeout_from_timed_out(self):
        """Exception with 'timed out' in message → timeout."""
        exc = RuntimeError("Gate evaluation timed out after 30s")
        assert _classify_failure(exc) == "timeout"

    def test_timeout_from_exception_type(self):
        """TimeoutError exception → timeout."""
        exc = TimeoutError("Connection timed out")
        assert _classify_failure(exc) == "timeout"

    def test_gate_execution_error_default(self):
        """Unknown exceptions default to gate_execution_error."""
        exc = RuntimeError("Unexpected NoneType in gate evaluation")
        assert _classify_failure(exc) == "gate_execution_error"

    def test_gate_execution_error_for_type_error(self):
        """TypeError during gate evaluation → gate_execution_error."""
        exc = TypeError("unsupported operand type(s) for +: 'NoneType' and 'int'")
        assert _classify_failure(exc) == "gate_execution_error"

    def test_all_reason_codes_covered(self):
        """All required reason codes from Requirement 13.5 are reachable."""
        required_codes = {
            "missing_input",
            "policy_not_found",
            "snapshot_corrupt",
            "gate_execution_error",
            "timeout",
        }
        # Map exceptions to expected codes
        test_cases = [
            (RuntimeError("Input is missing"), "missing_input"),
            (ValueError("Policy version unknown"), "policy_not_found"),
            (RuntimeError("Snapshot corrupt"), "snapshot_corrupt"),
            (TimeoutError("Timed out"), "timeout"),
            (RuntimeError("Division by zero in gate"), "gate_execution_error"),
        ]
        actual_codes = set()
        for exc, expected in test_cases:
            code = _classify_failure(exc)
            assert code == expected
            actual_codes.add(code)

        assert actual_codes == required_codes
