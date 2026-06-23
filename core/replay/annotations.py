"""
Replay annotations — append-only annotations on replay_audit_records.

Annotations allow operators, reviewers, and the CEO agent to attach commentary
to individual replay audit records without altering any computed fields
(Gate_Trace, Decision_Delta, Counterfactual_Outcome, replay_status).

Requirements: 13.4
"""

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import text


@dataclass(frozen=True)
class ReplayAnnotation:
    """A single annotation attached to a replay_audit_record."""
    id: int
    replay_id: str
    author: str
    annotation_timestamp: datetime
    content: str
    created_at: datetime


def add_annotation(session, replay_id: str, author: str, content: str) -> ReplayAnnotation:
    """
    Append a new annotation to a replay_audit_record.

    This is strictly append-only — no UPDATE or DELETE of existing annotations.
    Annotations never alter computed fields on the parent audit record.

    Args:
        session: SQLAlchemy connection or session (supports execute+commit).
        replay_id: The replay_id of the parent replay_audit_record.
        author: Author identifier (e.g. operator name, agent name).
        content: Free-text annotation content.

    Returns:
        The persisted ReplayAnnotation with its generated ID and timestamps.

    Raises:
        ValueError: If replay_id, author, or content is empty/None.
        sqlalchemy.exc.IntegrityError: If replay_id does not reference an existing
            replay_audit_record (FK constraint).
    """
    if not replay_id or not replay_id.strip():
        raise ValueError("replay_id must not be empty")
    if not author or not author.strip():
        raise ValueError("author must not be empty")
    if not content or not content.strip():
        raise ValueError("content must not be empty")

    now = datetime.now(timezone.utc).replace(tzinfo=None)  # UTC-naive per project convention

    result = session.execute(
        text("""
            INSERT INTO replay_annotations (replay_id, author, annotation_timestamp, content, created_at)
            VALUES (:replay_id, :author, :annotation_timestamp, :content, :created_at)
        """),
        {
            "replay_id": replay_id.strip(),
            "author": author.strip(),
            "annotation_timestamp": now,
            "content": content.strip(),
            "created_at": now,
        },
    )
    session.commit()

    # Retrieve the inserted row
    row = session.execute(
        text("""
            SELECT id, replay_id, author, annotation_timestamp, content, created_at
            FROM replay_annotations
            WHERE id = :row_id
        """),
        {"row_id": result.lastrowid},
    ).fetchone()

    return ReplayAnnotation(
        id=row[0],
        replay_id=row[1],
        author=row[2],
        annotation_timestamp=row[3] if isinstance(row[3], datetime) else datetime.fromisoformat(row[3]),
        content=row[4],
        created_at=row[5] if isinstance(row[5], datetime) else datetime.fromisoformat(row[5]),
    )


def get_annotations(session, replay_id: str) -> list[ReplayAnnotation]:
    """
    Return all annotations for a given replay_audit_record, ordered by timestamp.

    Args:
        session: SQLAlchemy connection or session.
        replay_id: The replay_id to fetch annotations for.

    Returns:
        List of ReplayAnnotation ordered by annotation_timestamp ascending.
    """
    if not replay_id or not replay_id.strip():
        raise ValueError("replay_id must not be empty")

    rows = session.execute(
        text("""
            SELECT id, replay_id, author, annotation_timestamp, content, created_at
            FROM replay_annotations
            WHERE replay_id = :replay_id
            ORDER BY annotation_timestamp ASC, id ASC
        """),
        {"replay_id": replay_id.strip()},
    ).fetchall()

    return [
        ReplayAnnotation(
            id=row[0],
            replay_id=row[1],
            author=row[2],
            annotation_timestamp=row[3] if isinstance(row[3], datetime) else datetime.fromisoformat(row[3]),
            content=row[4],
            created_at=row[5] if isinstance(row[5], datetime) else datetime.fromisoformat(row[5]),
        )
        for row in rows
    ]
