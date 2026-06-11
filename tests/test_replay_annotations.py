"""Tests for core/replay/annotations.py — append-only annotations on replay_audit_records.

Validates:
- Annotations can be appended to any existing replay record
- Multiple annotations can be added to the same replay
- Annotations do not alter computed fields (no UPDATE to replay_audit_records triggered)
- Author and timestamp are preserved correctly
- Append-only contract: no UPDATE/DELETE of existing annotations

Requirements: 13.4
"""

import pytest
from datetime import datetime
from sqlalchemy import create_engine, text

from db.replay_schema import init_replay_db
from core.replay.annotations import add_annotation, get_annotations, ReplayAnnotation


@pytest.fixture
def engine():
    """In-memory SQLite engine with replay schema initialized."""
    eng = create_engine("sqlite:///:memory:")
    # Create minimal production tables required for lineage migrations
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE blocked_trade_candidates (id INTEGER PRIMARY KEY)"))
        conn.execute(text("CREATE TABLE funnel_candidates (id INTEGER PRIMARY KEY)"))
        conn.execute(text("CREATE TABLE trade_events (id INTEGER PRIMARY KEY)"))
        conn.execute(text("CREATE TABLE trades (id INTEGER PRIMARY KEY)"))
        conn.execute(text("CREATE TABLE pm_candidates (id INTEGER PRIMARY KEY)"))
    init_replay_db(eng)
    return eng


@pytest.fixture
def session(engine):
    """A connection that acts as the session for annotation operations."""
    conn = engine.connect()
    yield conn
    conn.close()


@pytest.fixture
def replay_record(engine):
    """Insert a replay_audit_record and return its replay_id."""
    replay_id = "replay-ann-001"
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO replay_audit_records (
                replay_id, candidate_id, source_candidate_ids_json,
                replay_cutoff, input_sources_json, policy_version_json,
                replay_status, gate_trace_json, decision_delta_classification,
                counterfactual_outcome_json, era
            ) VALUES (
                :replay_id, 'cand-001', '["src-1"]',
                '2024-01-15 10:00:00', '{}', '{"name": "current"}',
                'exact', '[{"gate": "risk_geometry_gate", "decision": "allow"}]',
                'same_allow', '{"mfe": 0.05}', 'post-snapshot'
            )
        """), {"replay_id": replay_id})
    return replay_id


@pytest.fixture
def second_replay_record(engine):
    """Insert a second replay_audit_record."""
    replay_id = "replay-ann-002"
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO replay_audit_records (
                replay_id, candidate_id, source_candidate_ids_json,
                replay_cutoff, input_sources_json, policy_version_json,
                replay_status, era
            ) VALUES (
                :replay_id, 'cand-002', '["src-2"]',
                '2024-01-16 10:00:00', '{}', '{"name": "current"}',
                'partial', 'post-snapshot'
            )
        """), {"replay_id": replay_id})
    return replay_id


class TestAddAnnotation:
    """Test the add_annotation function."""

    def test_annotation_can_be_appended_to_existing_replay(self, session, replay_record):
        annotation = add_annotation(session, replay_record, "operator-1", "Reviewed: looks correct")

        assert isinstance(annotation, ReplayAnnotation)
        assert annotation.replay_id == replay_record
        assert annotation.author == "operator-1"
        assert annotation.content == "Reviewed: looks correct"
        assert annotation.id is not None
        assert annotation.annotation_timestamp is not None
        assert annotation.created_at is not None

    def test_multiple_annotations_on_same_replay(self, session, replay_record):
        ann1 = add_annotation(session, replay_record, "operator-1", "First note")
        ann2 = add_annotation(session, replay_record, "ceo-agent", "Second note")
        ann3 = add_annotation(session, replay_record, "reviewer", "Third note")

        annotations = get_annotations(session, replay_record)
        assert len(annotations) == 3
        assert annotations[0].content == "First note"
        assert annotations[1].content == "Second note"
        assert annotations[2].content == "Third note"

    def test_author_preserved(self, session, replay_record):
        annotation = add_annotation(session, replay_record, "ceo-agent", "Important finding")
        assert annotation.author == "ceo-agent"

    def test_timestamp_is_populated(self, session, replay_record):
        annotation = add_annotation(session, replay_record, "operator", "Note")
        assert isinstance(annotation.annotation_timestamp, datetime)

    def test_annotation_on_different_replay_records(self, session, replay_record, second_replay_record):
        add_annotation(session, replay_record, "op", "On first replay")
        add_annotation(session, second_replay_record, "op", "On second replay")

        first_anns = get_annotations(session, replay_record)
        second_anns = get_annotations(session, second_replay_record)
        assert len(first_anns) == 1
        assert len(second_anns) == 1
        assert first_anns[0].content == "On first replay"
        assert second_anns[0].content == "On second replay"

    def test_fk_constraint_rejects_invalid_replay_id(self, engine):
        """Annotations cannot reference a non-existent replay_audit_record.

        Note: We explicitly enable foreign_keys on this connection because
        in-memory SQLite with connection pooling may reuse a connection that
        was created before the pragma listener was registered.
        """
        with engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys=ON"))
            with pytest.raises(Exception):
                conn.execute(text("""
                    INSERT INTO replay_annotations (replay_id, author, annotation_timestamp, content, created_at)
                    VALUES ('nonexistent-replay-999', 'op', '2024-01-15 10:00:00', 'Should fail', '2024-01-15 10:00:00')
                """))
                conn.commit()

    def test_empty_replay_id_raises(self, session):
        with pytest.raises(ValueError, match="replay_id must not be empty"):
            add_annotation(session, "", "op", "content")

    def test_empty_author_raises(self, session):
        with pytest.raises(ValueError, match="author must not be empty"):
            add_annotation(session, "replay-001", "", "content")

    def test_empty_content_raises(self, session):
        with pytest.raises(ValueError, match="content must not be empty"):
            add_annotation(session, "replay-001", "op", "")


class TestGetAnnotations:
    """Test the get_annotations function."""

    def test_returns_empty_list_for_no_annotations(self, session, replay_record):
        annotations = get_annotations(session, replay_record)
        assert annotations == []

    def test_returns_all_annotations_ordered_by_timestamp(self, session, replay_record):
        add_annotation(session, replay_record, "a", "First")
        add_annotation(session, replay_record, "b", "Second")
        add_annotation(session, replay_record, "c", "Third")

        annotations = get_annotations(session, replay_record)
        assert len(annotations) == 3
        # Ordered by timestamp ascending
        assert annotations[0].author == "a"
        assert annotations[1].author == "b"
        assert annotations[2].author == "c"

    def test_empty_replay_id_raises(self, session):
        with pytest.raises(ValueError, match="replay_id must not be empty"):
            get_annotations(session, "")


class TestAnnotationsDoNotAlterComputedFields:
    """Verify that adding annotations does NOT modify computed fields on replay_audit_records.

    The immutability triggers on replay_audit_records prevent any UPDATE.
    Annotations are stored in a separate table (replay_annotations) linked via FK.
    This test verifies that the computed fields remain unchanged after annotations are added.
    """

    def test_computed_fields_unchanged_after_annotation(self, session, engine, replay_record):
        """Adding annotations leaves Gate_Trace, Decision_Delta, outcome, and status intact."""
        # Read computed fields before annotation
        before = session.execute(text("""
            SELECT replay_status, gate_trace_json, decision_delta_classification,
                   counterfactual_outcome_json
            FROM replay_audit_records WHERE replay_id = :rid
        """), {"rid": replay_record}).fetchone()

        # Add multiple annotations
        add_annotation(session, replay_record, "operator", "Annotation 1")
        add_annotation(session, replay_record, "reviewer", "Annotation 2")
        add_annotation(session, replay_record, "ceo-agent", "Annotation 3")

        # Read computed fields after annotations
        after = session.execute(text("""
            SELECT replay_status, gate_trace_json, decision_delta_classification,
                   counterfactual_outcome_json
            FROM replay_audit_records WHERE replay_id = :rid
        """), {"rid": replay_record}).fetchone()

        assert before[0] == after[0], "replay_status must not change"
        assert before[1] == after[1], "gate_trace_json must not change"
        assert before[2] == after[2], "decision_delta_classification must not change"
        assert before[3] == after[3], "counterfactual_outcome_json must not change"

    def test_audit_record_immutability_trigger_still_active(self, session, engine, replay_record):
        """The immutability trigger on replay_audit_records still fires (UPDATE blocked)."""
        with pytest.raises(Exception, match="immutable.*UPDATE prohibited"):
            session.execute(text("""
                UPDATE replay_audit_records SET replay_status = 'failed'
                WHERE replay_id = :rid
            """), {"rid": replay_record})
            session.commit()


class TestAnnotationsAppendOnly:
    """Verify annotations follow append-only semantics."""

    def test_annotations_have_no_update_mechanism(self, session, replay_record):
        """
        The annotations module only provides add_annotation and get_annotations.
        There is no update or delete function by design.
        Directly attempting SQL UPDATE/DELETE on replay_annotations is allowed
        by SQLite (no trigger prevents it), but the module API enforces append-only
        by not exposing mutation operations.
        """
        ann = add_annotation(session, replay_record, "op", "Original content")

        # The module provides no way to update or delete — this is the append-only guarantee.
        # Verify the annotation persists unchanged.
        annotations = get_annotations(session, replay_record)
        assert len(annotations) == 1
        assert annotations[0].content == "Original content"
        assert annotations[0].id == ann.id
