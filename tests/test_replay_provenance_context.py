"""Tests for core/replay/provenance_context.py — Replay provenance enrichment.

Tests cover:
- Building provenance context when provenance mode is disabled
- Building provenance context when lineage_id is None
- Building provenance context with no provenance records (historical_partial)
- Building provenance context with full provenance records
- First-invalid-stage detection from provenance events
- Upstream defect detection (invalid before gate_reconstruction)
- Final pre-gate geometry extraction
- Enriching replay results with provenance fields
- Counterfactual labeling
- Policy comparison mode
- Fail-open on database errors

Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6
"""

import json
import os
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text

from core.replay.provenance_context import (
    ReplayProvenanceContext,
    build_replay_provenance_context,
    build_policy_comparison_context,
    enrich_replay_result,
    label_counterfactual,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """In-memory SQLite engine with provenance_events table."""
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        conn.execute(text("""
            CREATE TABLE provenance_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lineage_id TEXT NOT NULL,
                stage_name TEXT NOT NULL,
                stage_version TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                mutation_ordinal INTEGER NOT NULL DEFAULT 1,
                timestamp TEXT NOT NULL,
                input_contract_json TEXT,
                output_contract_json TEXT,
                fields_changed_json TEXT,
                mutation_reason_code TEXT,
                rule_id TEXT,
                geometry_before_json TEXT,
                geometry_after_json TEXT,
                validation_before TEXT NOT NULL,
                validation_after TEXT NOT NULL,
                attempt_ordinal INTEGER NOT NULL DEFAULT 1,
                is_terminal INTEGER NOT NULL DEFAULT 0,
                payload_truncated INTEGER NOT NULL DEFAULT 0,
                UNIQUE(lineage_id, sequence_number)
            )
        """))
    return eng


def _insert_event(engine, lineage_id, stage_name, seq, validation_after, geometry_after=None):
    """Helper to insert a provenance event row."""
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO provenance_events (
                    lineage_id, stage_name, stage_version, sequence_number,
                    mutation_ordinal, timestamp, input_contract_json,
                    output_contract_json, fields_changed_json,
                    mutation_reason_code, geometry_before_json,
                    geometry_after_json, validation_before, validation_after
                ) VALUES (
                    :lid, :stage, 'v1', :seq, 1, '2024-01-01T00:00:00',
                    '{}', '{}', '[]', 'test', '{}', :geom_after, 'valid', :val_after
                )
            """),
            {
                "lid": lineage_id,
                "stage": stage_name,
                "seq": seq,
                "val_after": validation_after,
                "geom_after": geometry_after or "{}",
            },
        )


# ---------------------------------------------------------------------------
# Tests: Provenance mode disabled / no lineage
# ---------------------------------------------------------------------------


class TestBuildContextDisabled:
    """Tests for build_replay_provenance_context when provenance is disabled."""

    @patch("core.replay.provenance_context.PM_PROVENANCE_MODE", "disabled")
    def test_returns_historical_partial_when_disabled(self, engine):
        """Req 12.2: When provenance mode is disabled, label as historical_partial."""
        ctx = build_replay_provenance_context(engine, "some-lineage-id")

        assert ctx.label == "historical_partial"
        assert ctx.first_invalid_stage == "not_available"
        assert ctx.has_provenance is False
        assert ctx.provenance_unavailable_reason == "provenance_disabled"

    @patch("core.replay.provenance_context.PM_PROVENANCE_MODE", "observe")
    def test_returns_historical_partial_when_no_lineage_id(self, engine):
        """Req 12.2: When lineage_id is None, label as historical_partial."""
        ctx = build_replay_provenance_context(engine, None)

        assert ctx.label == "historical_partial"
        assert ctx.first_invalid_stage == "not_available"
        assert ctx.has_provenance is False
        assert ctx.provenance_unavailable_reason == "no_lineage_id"


# ---------------------------------------------------------------------------
# Tests: No provenance records
# ---------------------------------------------------------------------------


class TestBuildContextNoRecords:
    """Tests for candidates lacking provenance records."""

    @patch("core.replay.provenance_context.PM_PROVENANCE_MODE", "observe")
    def test_returns_historical_partial_when_no_records(self, engine):
        """Req 12.2: Candidates with no provenance records get historical_partial."""
        ctx = build_replay_provenance_context(engine, "no-records-lineage")

        assert ctx.label == "historical_partial"
        assert ctx.first_invalid_stage == "not_available"
        assert ctx.has_provenance is False
        assert ctx.provenance_unavailable_reason == "no_provenance_records"


# ---------------------------------------------------------------------------
# Tests: Full provenance context
# ---------------------------------------------------------------------------


class TestBuildContextFullProvenance:
    """Tests for candidates with complete provenance records."""

    @patch("core.replay.provenance_context.PM_PROVENANCE_MODE", "observe")
    def test_builds_full_context_all_valid(self, engine):
        """Req 12.1: Link replay to provenance lineage with full context."""
        lid = "full-valid-lineage"
        _insert_event(engine, lid, "parsed_pm_decision", 1, "valid")
        _insert_event(engine, lid, "behavioral_adjustment", 2, "valid")
        _insert_event(engine, lid, "pre_gate_snapshot", 3, "valid",
                      geometry_after=json.dumps({"entry_price": "100.0", "stop_price": "95.0"}))

        ctx = build_replay_provenance_context(engine, lid)

        assert ctx.label == "full_provenance"
        assert ctx.has_provenance is True
        assert ctx.lineage_id == lid
        assert ctx.first_invalid_stage == "not_available"
        assert ctx.upstream_defect_present is False
        assert ctx.upstream_valid_contract is True
        assert ctx.final_pre_gate_geometry == {"entry_price": "100.0", "stop_price": "95.0"}

    @patch("core.replay.provenance_context.PM_PROVENANCE_MODE", "observe")
    def test_detects_first_invalid_stage(self, engine):
        """Req 12.3: Exposes original First_Invalid_Stage in replay context."""
        lid = "has-invalid-lineage"
        _insert_event(engine, lid, "parsed_pm_decision", 1, "valid")
        _insert_event(engine, lid, "price_repair", 2, "invalid")
        _insert_event(engine, lid, "pre_gate_snapshot", 3, "invalid")

        ctx = build_replay_provenance_context(engine, lid)

        assert ctx.first_invalid_stage == "price_repair"
        assert ctx.attribution_category == "price_repair_invalid"

    @patch("core.replay.provenance_context.PM_PROVENANCE_MODE", "observe")
    def test_flags_upstream_defect_present(self, engine):
        """Req 12.4: Flags upstream_defect_present when invalid before reconstruction."""
        lid = "upstream-defect-lineage"
        _insert_event(engine, lid, "parsed_pm_decision", 1, "invalid")
        _insert_event(engine, lid, "behavioral_adjustment", 2, "invalid")
        _insert_event(engine, lid, "pre_gate_snapshot", 3, "invalid")
        _insert_event(engine, lid, "gate_reconstruction", 4, "valid")

        ctx = build_replay_provenance_context(engine, lid)

        assert ctx.upstream_defect_present is True
        assert ctx.upstream_valid_contract is False

    @patch("core.replay.provenance_context.PM_PROVENANCE_MODE", "observe")
    def test_no_upstream_defect_when_only_gate_invalid(self, engine):
        """Req 12.4: No upstream_defect when only gate_reconstruction introduces invalidity."""
        lid = "gate-only-defect"
        _insert_event(engine, lid, "parsed_pm_decision", 1, "valid")
        _insert_event(engine, lid, "pre_gate_snapshot", 2, "valid")
        _insert_event(engine, lid, "gate_reconstruction", 3, "invalid")

        ctx = build_replay_provenance_context(engine, lid)

        assert ctx.upstream_defect_present is False
        assert ctx.upstream_valid_contract is True

    @patch("core.replay.provenance_context.PM_PROVENANCE_MODE", "observe")
    def test_extracts_pre_gate_geometry(self, engine):
        """Req 12.3: Extracts final pre-gate geometry from pre_gate_snapshot."""
        lid = "geometry-lineage"
        geom = {"entry_price": "150.25", "stop_price": "145.00",
                "target_price": "160.50", "reward_to_risk": "2.05"}
        _insert_event(engine, lid, "parsed_pm_decision", 1, "valid")
        _insert_event(engine, lid, "pre_gate_snapshot", 2, "valid",
                      geometry_after=json.dumps(geom))

        ctx = build_replay_provenance_context(engine, lid)

        assert ctx.final_pre_gate_geometry == geom


# ---------------------------------------------------------------------------
# Tests: Fail-open behavior
# ---------------------------------------------------------------------------


class TestBuildContextFailOpen:
    """Tests for fail-open behavior on database errors."""

    @patch("core.replay.provenance_context.PM_PROVENANCE_MODE", "observe")
    def test_returns_historical_partial_on_query_failure(self):
        """Fail-open: DB error results in historical_partial label."""
        # Use an engine that will fail (table doesn't exist)
        bad_engine = create_engine("sqlite:///:memory:")

        ctx = build_replay_provenance_context(bad_engine, "some-lineage")

        assert ctx.label == "historical_partial"
        assert ctx.first_invalid_stage == "not_available"
        assert ctx.provenance_unavailable_reason == "query_failed"


# ---------------------------------------------------------------------------
# Tests: Replay result enrichment
# ---------------------------------------------------------------------------


class TestEnrichReplayResult:
    """Tests for enrich_replay_result function."""

    def test_enriches_result_with_provenance_fields(self):
        """Req 12.3: Adds provenance fields to replay result."""
        ctx = ReplayProvenanceContext(
            lineage_id="test-lineage",
            has_provenance=True,
            label="full_provenance",
            first_invalid_stage="price_repair",
            attribution_category="price_repair_invalid",
            final_pre_gate_geometry={"entry_price": "100.0"},
            original_gate_decision="reject",
            original_gate_reason="rr_below_minimum",
            upstream_valid_contract=False,
            upstream_defect_present=True,
        )

        result = {}
        enriched = enrich_replay_result(result, ctx, "allow", "meets_threshold")

        assert enriched["provenance_label"] == "full_provenance"
        assert enriched["first_invalid_stage"] == "price_repair"
        assert enriched["attribution_category"] == "price_repair_invalid"
        assert enriched["upstream_defect_present"] is True
        assert enriched["upstream_valid_contract"] is False
        assert enriched["original_gate_decision"] == "reject"
        assert enriched["original_gate_reason"] == "rr_below_minimum"
        assert enriched["replay_decision"] == "allow"
        assert enriched["replay_reason"] == "meets_threshold"
        assert enriched["final_pre_gate_geometry"] == {"entry_price": "100.0"}

    def test_includes_counterfactual_fields_when_labeled(self):
        """Req 12.6: Counterfactual labeling in enriched results."""
        ctx = ReplayProvenanceContext(
            lineage_id="cf-lineage",
            is_counterfactual=True,
            counterfactual_corrected_fields=["stop_price", "reward_to_risk"],
        )

        result = {}
        enriched = enrich_replay_result(result, ctx, "allow", "corrected_geometry")

        assert enriched["counterfactual_contract"] is True
        assert enriched["counterfactual_corrected_fields"] == ["stop_price", "reward_to_risk"]

    def test_no_counterfactual_fields_when_not_labeled(self):
        """No counterfactual keys when is_counterfactual is False."""
        ctx = ReplayProvenanceContext(lineage_id="normal-lineage")

        result = {}
        enriched = enrich_replay_result(result, ctx, "reject", "rr_too_low")

        assert "counterfactual_contract" not in enriched
        assert "counterfactual_corrected_fields" not in enriched


# ---------------------------------------------------------------------------
# Tests: Counterfactual labeling
# ---------------------------------------------------------------------------


class TestLabelCounterfactual:
    """Tests for label_counterfactual function."""

    def test_labels_context_as_counterfactual(self):
        """Req 12.6: Labels replay as counterfactual with corrected fields."""
        ctx = ReplayProvenanceContext(
            lineage_id="cf-test",
            has_provenance=True,
            label="full_provenance",
        )

        labeled = label_counterfactual(ctx, ["stop_price", "target_price"])

        assert labeled.is_counterfactual is True
        assert labeled.counterfactual_corrected_fields == ["stop_price", "target_price"]
        # Original context is unchanged (immutable via replace)
        assert ctx.is_counterfactual is False

    def test_preserves_existing_context_fields(self):
        """Counterfactual labeling preserves all other context fields."""
        ctx = ReplayProvenanceContext(
            lineage_id="preserve-test",
            has_provenance=True,
            label="full_provenance",
            first_invalid_stage="parsed_pm_decision",
            upstream_defect_present=True,
        )

        labeled = label_counterfactual(ctx, ["entry_price"])

        assert labeled.lineage_id == "preserve-test"
        assert labeled.label == "full_provenance"
        assert labeled.first_invalid_stage == "parsed_pm_decision"
        assert labeled.upstream_defect_present is True


# ---------------------------------------------------------------------------
# Tests: Policy comparison mode
# ---------------------------------------------------------------------------


class TestPolicyComparisonContext:
    """Tests for build_policy_comparison_context."""

    @patch("core.replay.provenance_context.PM_PROVENANCE_MODE", "observe")
    def test_builds_context_with_original_decision(self, engine):
        """Req 12.5: Builds context for candidate-policy comparison mode."""
        lid = "policy-compare-lineage"
        _insert_event(engine, lid, "parsed_pm_decision", 1, "valid")
        _insert_event(engine, lid, "pre_gate_snapshot", 2, "valid",
                      geometry_after=json.dumps({"entry_price": "100.0"}))

        ctx = build_policy_comparison_context(
            engine,
            lid,
            original_gate_decision="reject",
            original_gate_reason="position_too_large",
        )

        assert ctx.label == "full_provenance"
        assert ctx.original_gate_decision == "reject"
        assert ctx.original_gate_reason == "position_too_large"
        assert ctx.final_pre_gate_geometry == {"entry_price": "100.0"}

    @patch("core.replay.provenance_context.PM_PROVENANCE_MODE", "disabled")
    def test_policy_comparison_fails_gracefully_when_disabled(self, engine):
        """Policy comparison gracefully degrades when provenance is disabled."""
        ctx = build_policy_comparison_context(
            engine,
            "some-lineage",
            original_gate_decision="allow",
            original_gate_reason="meets_all_criteria",
        )

        assert ctx.label == "historical_partial"
        assert ctx.original_gate_decision == "allow"
        assert ctx.original_gate_reason == "meets_all_criteria"
