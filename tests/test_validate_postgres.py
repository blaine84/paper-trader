"""Unit tests for the post-migration validation script.

Tests report generation with known pass/fail scenarios and HTTP health
check timeout handling.

Validates: Requirements 5.8
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from scripts.validate_postgres import (
    CheckResult,
    ValidationReport,
    run_validation,
    validate_candidate_uniqueness,
    validate_fk_integrity,
    validate_proxy_health,
    validate_row_counts,
    validate_stale_reservations,
    validate_web_health,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_engine_with_scalar_sequence(values: list):
    """Create a mock engine that returns scalar values in sequence."""
    engine = MagicMock()
    conn = MagicMock()
    idx = {"i": 0}

    def execute_side_effect(query):
        result = MagicMock()
        result.scalar.return_value = values[idx["i"]]
        idx["i"] += 1
        return result

    conn.execute = execute_side_effect
    conn.__enter__ = lambda s: conn
    conn.__exit__ = MagicMock(return_value=False)
    engine.connect.return_value = conn
    return engine


def _make_engine_with_single_scalar(value):
    """Create a mock engine that always returns a single scalar value."""
    engine = MagicMock()
    conn = MagicMock()
    result = MagicMock()
    result.scalar.return_value = value
    conn.execute = MagicMock(return_value=result)
    conn.__enter__ = lambda s: conn
    conn.__exit__ = MagicMock(return_value=False)
    engine.connect.return_value = conn
    return engine


# ── validate_row_counts ──────────────────────────────────────────────────────


class TestValidateRowCounts:
    """Tests for validate_row_counts()."""

    @patch("scripts.validate_postgres.TABLE_ORDER", ["trades", "positions", "balance"])
    def test_pass_when_all_tables_have_rows(self):
        """Returns PASS when all tables have > 0 rows."""
        engine = _make_engine_with_scalar_sequence([10, 5, 3])

        check = validate_row_counts(engine)
        assert check.status == "PASS"
        assert check.name == "row_counts"

    @patch("scripts.validate_postgres.TABLE_ORDER", ["trades", "positions", "balance"])
    def test_fail_when_some_tables_empty(self):
        """Returns FAIL when some tables have 0 rows."""
        engine = _make_engine_with_scalar_sequence([10, 0, 3])

        check = validate_row_counts(engine)
        assert check.status == "FAIL"
        assert "positions" in check.detail


# ── validate_stale_reservations ──────────────────────────────────────────────


class TestValidateStaleReservations:
    """Tests for validate_stale_reservations()."""

    def test_pass_when_count_is_zero(self):
        """Returns PASS when no candidates are in RESERVED state."""
        engine = _make_engine_with_single_scalar(0)

        check = validate_stale_reservations(engine)
        assert check.status == "PASS"
        assert check.name == "stale_reservations"

    def test_warning_when_count_greater_than_zero(self):
        """Returns WARNING when candidates are in RESERVED state."""
        engine = _make_engine_with_single_scalar(3)

        check = validate_stale_reservations(engine)
        assert check.status == "WARNING"
        assert "3" in check.detail


# ── validate_candidate_uniqueness ────────────────────────────────────────────


class TestValidateCandidateUniqueness:
    """Tests for validate_candidate_uniqueness()."""

    def test_pass_with_no_duplicates(self):
        """Returns PASS when no duplicate candidate_ids exist."""
        engine = _make_engine_with_single_scalar(0)

        check = validate_candidate_uniqueness(engine)
        assert check.status == "PASS"
        assert check.name == "candidate_id_uniqueness"

    def test_fail_with_duplicates(self):
        """Returns FAIL when duplicate candidate_ids exist."""
        engine = _make_engine_with_single_scalar(2)

        check = validate_candidate_uniqueness(engine)
        assert check.status == "FAIL"
        assert "2" in check.detail


# ── validate_fk_integrity ────────────────────────────────────────────────────


class TestValidateFkIntegrity:
    """Tests for validate_fk_integrity()."""

    def test_pass_with_no_orphans(self):
        """Returns PASS when all FK references are valid."""
        # Both queries (orphan_trades, orphan_candidates) return 0
        engine = _make_engine_with_scalar_sequence([0, 0])

        check = validate_fk_integrity(engine)
        assert check.status == "PASS"
        assert check.name == "fk_integrity"


# ── validate_web_health ──────────────────────────────────────────────────────


class TestValidateWebHealth:
    """Tests for validate_web_health()."""

    @patch("requests.get")
    def test_pass_on_http_200(self, mock_get):
        """Returns PASS when the endpoint returns HTTP 200."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        check = validate_web_health("http://127.0.0.1:5000")
        assert check.status == "PASS"
        assert check.name == "web_api_health"

    @patch("requests.get")
    def test_fail_on_timeout(self, mock_get):
        """Returns FAIL when the request times out."""
        mock_get.side_effect = requests.exceptions.Timeout("timed out")

        check = validate_web_health("http://127.0.0.1:5000", timeout=5)
        assert check.status == "FAIL"
        assert "timed out" in check.detail.lower() or "Timeout" in check.actual

    @patch("requests.get")
    def test_fail_on_connection_error(self, mock_get):
        """Returns FAIL on connection error."""
        mock_get.side_effect = requests.exceptions.ConnectionError("refused")

        check = validate_web_health("http://127.0.0.1:5000")
        assert check.status == "FAIL"
        assert "connection" in check.detail.lower() or "Connection error" in check.actual


# ── validate_proxy_health ────────────────────────────────────────────────────


class TestValidateProxyHealth:
    """Tests for validate_proxy_health()."""

    @patch("requests.get")
    def test_fail_on_empty_body(self, mock_get):
        """Returns FAIL when response body is empty."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ""
        mock_get.return_value = mock_resp

        check = validate_proxy_health("https://proxy.ts.net")
        assert check.status == "FAIL"
        assert "empty" in check.detail.lower()


# ── run_validation (report aggregation) ──────────────────────────────────────


class TestRunValidation:
    """Tests for run_validation() report aggregation."""

    @patch("scripts.validate_postgres.create_engine")
    @patch("scripts.validate_postgres.validate_row_counts")
    @patch("scripts.validate_postgres.validate_stale_reservations")
    @patch("scripts.validate_postgres.validate_candidate_uniqueness")
    @patch("scripts.validate_postgres.validate_fk_integrity")
    def test_overall_pass_when_all_checks_pass(
        self,
        mock_fk,
        mock_unique,
        mock_stale,
        mock_rows,
        mock_create_engine,
    ):
        """Report status is PASS when all checks pass."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        mock_rows.return_value = CheckResult(name="row_counts", status="PASS", detail="ok")
        mock_stale.return_value = CheckResult(name="stale_reservations", status="PASS", detail="ok")
        mock_unique.return_value = CheckResult(name="candidate_id_uniqueness", status="PASS", detail="ok")
        mock_fk.return_value = CheckResult(name="fk_integrity", status="PASS", detail="ok")

        report = run_validation(
            database_url="postgresql+psycopg://user:pass@localhost/test",
            skip_http=True,
        )

        assert report.status == "PASS"
        # 4 DB checks + 2 SKIP HTTP checks
        assert len(report.checks) == 6

    @patch("scripts.validate_postgres.create_engine")
    @patch("scripts.validate_postgres.validate_row_counts")
    @patch("scripts.validate_postgres.validate_stale_reservations")
    @patch("scripts.validate_postgres.validate_candidate_uniqueness")
    @patch("scripts.validate_postgres.validate_fk_integrity")
    def test_overall_fail_when_any_check_fails(
        self,
        mock_fk,
        mock_unique,
        mock_stale,
        mock_rows,
        mock_create_engine,
    ):
        """Report status is FAIL when at least one check fails."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        mock_rows.return_value = CheckResult(name="row_counts", status="FAIL", detail="empty tables")
        mock_stale.return_value = CheckResult(name="stale_reservations", status="PASS", detail="ok")
        mock_unique.return_value = CheckResult(name="candidate_id_uniqueness", status="PASS", detail="ok")
        mock_fk.return_value = CheckResult(name="fk_integrity", status="PASS", detail="ok")

        report = run_validation(
            database_url="postgresql+psycopg://user:pass@localhost/test",
            skip_http=True,
        )

        assert report.status == "FAIL"
        failing_checks = [c for c in report.checks if c.status == "FAIL"]
        assert len(failing_checks) == 1
        assert failing_checks[0].name == "row_counts"
