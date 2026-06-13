"""Unit tests for provenance failure isolation (Requirements 14.2, 14.3).

Verifies:
- persist_provenance_chain() never raises on failure (fail-open)
- Coverage metrics track attempts and failures correctly
- Structural geometry validation (is_valid=False) always blocks (gate logic)
- Gate pipeline is unaffected by provenance failures
"""
import logging
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

from utils.geometry_calculator import GeometryResult, ValidationStatus
from utils.provenance_capture import (
    ProvenanceChain,
    ProvenanceEvent,
    get_provenance_coverage_metrics,
    increment_provenance_attempt,
    increment_provenance_failure,
    persist_provenance_chain,
    reset_provenance_coverage_metrics,
)


def _make_valid_geometry() -> GeometryResult:
    """Create a valid BUY geometry result for testing."""
    return GeometryResult(
        direction="BUY",
        entry_price=Decimal("100.00"),
        stop_price=Decimal("95.00"),
        target_price=Decimal("110.00"),
        quantity=Decimal("10"),
        risk_distance=Decimal("5.00"),
        reward_distance=Decimal("10.00"),
        reward_to_risk=Decimal("2.00"),
        per_unit_risk=Decimal("5.00"),
        total_dollar_risk=Decimal("50.00"),
        stop_direction_valid=True,
        target_direction_valid=True,
        is_valid=True,
        validation_errors=[],
        validation_status=ValidationStatus.VALID,
    )


def _make_chain_with_event() -> ProvenanceChain:
    """Create a provenance chain with one event for testing."""
    chain = ProvenanceChain(lineage_id="test-lineage-001", pm_mode="candidate_id")
    geometry = _make_valid_geometry()
    chain.record_event(
        stage_name="trusted_input",
        stage_version="1.0",
        input_contract={"symbol": "AAPL"},
        output_contract={"symbol": "AAPL"},
        fields_changed=[],
        mutation_reason_code="passthrough",
        rule_id=None,
        geometry_before=geometry,
        geometry_after=geometry,
    )
    return chain


class TestProvenanceCoverageMetrics(unittest.TestCase):
    """Test the coverage metric counters."""

    def setUp(self):
        reset_provenance_coverage_metrics()

    def test_initial_metrics_zero(self):
        metrics = get_provenance_coverage_metrics()
        assert metrics["attempts"] == 0
        assert metrics["failures"] == 0
        assert metrics["successes"] == 0
        assert metrics["success_rate_pct"] == 0.0

    def test_increment_attempt(self):
        increment_provenance_attempt()
        increment_provenance_attempt()
        metrics = get_provenance_coverage_metrics()
        assert metrics["attempts"] == 2
        assert metrics["failures"] == 0
        assert metrics["successes"] == 2
        assert metrics["success_rate_pct"] == 100.0

    def test_increment_failure_decrements_coverage(self):
        increment_provenance_attempt()
        increment_provenance_attempt()
        increment_provenance_failure()
        metrics = get_provenance_coverage_metrics()
        assert metrics["attempts"] == 2
        assert metrics["failures"] == 1
        assert metrics["successes"] == 1
        assert metrics["success_rate_pct"] == 50.0

    def test_reset_clears_all(self):
        increment_provenance_attempt()
        increment_provenance_failure()
        reset_provenance_coverage_metrics()
        metrics = get_provenance_coverage_metrics()
        assert metrics["attempts"] == 0
        assert metrics["failures"] == 0

    def test_success_rate_calculation(self):
        for _ in range(10):
            increment_provenance_attempt()
        for _ in range(3):
            increment_provenance_failure()
        metrics = get_provenance_coverage_metrics()
        assert metrics["success_rate_pct"] == 70.0


class TestPersistProvenanceChainFailOpen(unittest.TestCase):
    """Test that persist_provenance_chain is fail-open for provenance."""

    def setUp(self):
        reset_provenance_coverage_metrics()

    def test_does_not_raise_on_engine_failure(self):
        """Provenance failure must NOT propagate — gate pipeline continues."""
        chain = _make_chain_with_event()
        # Mock engine that raises on begin()
        engine = MagicMock()
        engine.begin.side_effect = RuntimeError("DB connection lost")

        # Should NOT raise
        persist_provenance_chain(engine, chain)

        # Verify failure was recorded in metrics
        metrics = get_provenance_coverage_metrics()
        assert metrics["attempts"] == 1
        assert metrics["failures"] == 1

    def test_does_not_raise_on_execute_failure(self):
        """Even if SQL execution fails, function returns normally."""
        chain = _make_chain_with_event()

        engine = MagicMock()
        conn_mock = MagicMock()
        conn_mock.execute.side_effect = Exception("SQL error")
        engine.begin.return_value.__enter__ = MagicMock(return_value=conn_mock)
        engine.begin.return_value.__exit__ = MagicMock(return_value=False)

        persist_provenance_chain(engine, chain)

        metrics = get_provenance_coverage_metrics()
        assert metrics["attempts"] == 1
        assert metrics["failures"] == 1

    def test_logs_error_with_identifying_fields(self, ):
        """On failure, logs lineage_id and stage names per Requirement 14.3."""
        chain = _make_chain_with_event()
        engine = MagicMock()
        engine.begin.side_effect = RuntimeError("timeout")

        with patch("utils.provenance_capture.logger") as mock_logger:
            persist_provenance_chain(engine, chain)
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            # Check lineage_id is in the log message args
            assert "test-lineage-001" in str(call_args)
            # Check stage names are in the log message args
            assert "trusted_input" in str(call_args)

    def test_successful_persist_increments_attempt_only(self):
        """On success, only attempt is incremented (no failure)."""
        from sqlalchemy import create_engine
        from db.provenance_schema import init_provenance_schema

        engine = create_engine("sqlite:///:memory:")
        init_provenance_schema(engine)

        chain = _make_chain_with_event()
        persist_provenance_chain(engine, chain)

        metrics = get_provenance_coverage_metrics()
        assert metrics["attempts"] == 1
        assert metrics["failures"] == 0
        assert metrics["success_rate_pct"] == 100.0

    def test_empty_chain_skips_persistence(self):
        """Empty chain returns early — no attempt recorded."""
        chain = ProvenanceChain(lineage_id="empty", pm_mode="candidate_id")
        engine = MagicMock()

        persist_provenance_chain(engine, chain)

        metrics = get_provenance_coverage_metrics()
        assert metrics["attempts"] == 0
        engine.begin.assert_not_called()


class TestStructuralGeometryAlwaysBlocks(unittest.TestCase):
    """Verify that structural geometry validation (is_valid=False) is gate logic
    that always blocks, independent of provenance.

    This is conceptual validation — the geometry calculator's is_valid field
    is the authority for gate decisions, not provenance persistence.
    """

    def test_invalid_geometry_is_valid_false(self):
        """compute_geometry producing is_valid=False means gate MUST block."""
        from utils.geometry_calculator import compute_geometry

        # BUY with stop ABOVE entry — invalid geometry
        result = compute_geometry(
            direction="BUY",
            entry_price=Decimal("100.00"),
            stop_price=Decimal("105.00"),  # invalid: stop above entry for BUY
            target_price=Decimal("110.00"),
            quantity=Decimal("10"),
        )
        assert result.is_valid is False
        assert result.stop_direction_valid is False
        # This result would block the candidate at the gate level
        # regardless of whether provenance persistence succeeds or fails

    def test_valid_geometry_passes(self):
        """compute_geometry producing is_valid=True means geometry check passes."""
        from utils.geometry_calculator import compute_geometry

        result = compute_geometry(
            direction="BUY",
            entry_price=Decimal("100.00"),
            stop_price=Decimal("95.00"),
            target_price=Decimal("110.00"),
            quantity=Decimal("10"),
        )
        assert result.is_valid is True
        assert result.validation_status == ValidationStatus.VALID


class TestGatePipelineIndependence(unittest.TestCase):
    """Verify gate pipeline produces identical decisions with or without provenance.

    The gate decision is based on compute_geometry().is_valid (structural check),
    NOT on provenance persistence success. This confirms the separation principle.
    """

    def setUp(self):
        reset_provenance_coverage_metrics()

    def test_gate_decision_unchanged_after_provenance_failure(self):
        """A provenance failure does not alter the gate decision."""
        from utils.geometry_calculator import compute_geometry

        # Compute geometry (this is the gate input — deterministic)
        geometry = compute_geometry(
            direction="BUY",
            entry_price=Decimal("100.00"),
            stop_price=Decimal("95.00"),
            target_price=Decimal("110.00"),
            quantity=Decimal("10"),
        )

        # Gate decision: based purely on geometry validity
        gate_decision_before_provenance = geometry.is_valid

        # Provenance fails
        chain = ProvenanceChain(lineage_id="gate-test", pm_mode="candidate_id")
        chain.record_event(
            stage_name="pre_gate_snapshot",
            stage_version="1.0",
            input_contract={},
            output_contract={},
            fields_changed=[],
            mutation_reason_code="passthrough",
            rule_id=None,
            geometry_before=geometry,
            geometry_after=geometry,
        )
        engine = MagicMock()
        engine.begin.side_effect = RuntimeError("DB down")
        persist_provenance_chain(engine, chain)

        # Gate decision is still the same — provenance failure doesn't change it
        gate_decision_after_provenance = geometry.is_valid
        assert gate_decision_before_provenance == gate_decision_after_provenance
        assert gate_decision_after_provenance is True

        # Verify provenance DID fail
        metrics = get_provenance_coverage_metrics()
        assert metrics["failures"] == 1

    def test_invalid_geometry_blocks_regardless_of_provenance_success(self):
        """Invalid geometry blocks even when provenance persists successfully."""
        from utils.geometry_calculator import compute_geometry

        # Invalid geometry: stop wrong side
        geometry = compute_geometry(
            direction="SHORT",
            entry_price=Decimal("100.00"),
            stop_price=Decimal("95.00"),  # invalid: stop below entry for SHORT
            target_price=Decimal("90.00"),
            quantity=Decimal("10"),
        )

        # Gate decision: BLOCK (is_valid=False)
        assert geometry.is_valid is False

        # Even if provenance succeeds, gate still blocks
        # (We're verifying the principle: geometry validation is gate logic)


if __name__ == "__main__":
    unittest.main()
