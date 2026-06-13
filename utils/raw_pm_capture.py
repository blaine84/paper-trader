"""Raw PM Response Capture.

Captures and persists raw PM model outputs before any parsing, JSON repair,
normalization, or field mutation occurs. One PM response can contain decisions
for multiple candidates; raw responses are keyed by pm_cycle_id + attempt_ordinal,
not by candidate lineage.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from utils.gate_config import PM_PROVENANCE_DETAIL

log = logging.getLogger(__name__)


MAX_RAW_PAYLOAD_BYTES = 256 * 1024  # 256 KB

# Credential keys to redact from stored payloads.
# Values matching these JSON keys are replaced with "[REDACTED]".
_CREDENTIAL_KEYS = frozenset({
    "api_key",
    "bearer_token",
    "authorization",
    "session_cookie",
    "access_token",
    "refresh_token",
    "private_key",
    "secret_key",
})

# Fields that must NEVER be redacted even if they look credential-like.
_PRESERVED_FIELDS = frozenset({
    "rationale",
    "reasoning",
    "setup_reasoning",
})

# Regex pattern matching credential keys in JSON text (case-insensitive).
# Captures: "key_name" : "value" (with optional whitespace).
_CREDENTIAL_PATTERN = re.compile(
    r'"(' + "|".join(re.escape(k) for k in _CREDENTIAL_KEYS) + r')"\s*:\s*"([^"]*)"',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawPMResponse:
    """Immutable record of raw PM model output (one per response, not per candidate)."""

    response_id: str  # UUID4, unique per stored response
    pm_cycle_id: str  # links all responses in a cycle
    profile: str
    model_id: str  # model/provider identifier
    timestamp: datetime  # ISO 8601 UTC, millisecond precision
    prompt_version_id: str
    candidate_ids_supplied: list[str]  # ALL candidate IDs offered to the model
    raw_payload: str | None  # stored payload (None in minimal mode, truncated if > 256KB)
    original_payload_hash: str  # SHA-256 of ORIGINAL full payload (always computed)
    stored_payload_hash: str | None  # SHA-256 of STORED payload (None when raw_payload is None)
    parse_status: str  # parse_success | parse_failed_json_repaired | parse_failed_unrecoverable | parse_not_attempted
    attempt_ordinal: int  # 1 for first, 2 for retry, etc.
    payload_size_bytes: int  # original UTF-8 byte length (always recorded)
    payload_truncated: bool


@dataclass(frozen=True)
class ResponseLineageLink:
    """Join record linking one raw response to one candidate lineage."""

    response_id: str
    lineage_id: str
    candidate_id: str | None  # pm_candidates.candidate_id if candidate-ID mode


# ---------------------------------------------------------------------------
# Credential Redaction
# ---------------------------------------------------------------------------


def _strip_credentials(payload: str) -> str:
    """Strip credential values from JSON payload while preserving rationale fields.

    Replaces values for known credential keys with "[REDACTED]".
    Preserves rationale, reasoning, and setup_reasoning fields untouched.
    """
    def _replacer(match: re.Match) -> str:
        key = match.group(1).lower()
        if key in _PRESERVED_FIELDS:
            return match.group(0)
        return f'"{match.group(1)}": "[REDACTED]"'

    return _CREDENTIAL_PATTERN.sub(_replacer, payload)


# ---------------------------------------------------------------------------
# UTF-8 Safe Truncation
# ---------------------------------------------------------------------------


def _truncate_utf8(payload: str, max_bytes: int) -> tuple[str, bool]:
    """Truncate a string at a UTF-8 byte boundary without splitting multibyte characters.

    Returns (truncated_string, was_truncated).
    If the payload is within max_bytes, returns the original string unchanged.
    """
    encoded = payload.encode("utf-8")
    if len(encoded) <= max_bytes:
        return payload, False

    # Cut at max_bytes then decode, ignoring incomplete trailing bytes.
    # This safely handles multibyte characters by discarding any incomplete
    # sequence at the cut boundary.
    truncated_bytes = encoded[:max_bytes]
    # Walk back from the cut to find valid UTF-8 boundary
    truncated_str = truncated_bytes.decode("utf-8", errors="ignore")
    return truncated_str, True


# ---------------------------------------------------------------------------
# Core Capture Function
# ---------------------------------------------------------------------------


def capture_raw_pm_response(
    pm_cycle_id: str,
    profile: str,
    model_id: str,
    prompt_version_id: str,
    candidate_ids_supplied: list[str],
    raw_payload: str,
    parse_status: str,
    attempt_ordinal: int = 1,
) -> RawPMResponse:
    """Create immutable raw response record with dual integrity hashes.

    Stores:
    - original_payload_hash: SHA-256 of the raw payload BEFORE any processing (always)
    - stored_payload_hash: SHA-256 of what is actually stored (None in minimal mode)

    Truncation:
    - UTF-8 byte truncation at 256 KB boundary, never splitting multibyte characters
    - Records original byte length in payload_size_bytes
    - Sets payload_truncated=True when truncated

    In minimal mode (PM_PROVENANCE_DETAIL="minimal"):
    - raw_payload is None (not stored)
    - stored_payload_hash is None
    - original_payload_hash, metadata, and parse_status still recorded

    Strips: API keys, bearer tokens, auth headers, provider credentials.
    PRESERVES: rationale, reasoning, setup_reasoning, and other user-visible fields.
    """
    response_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc)

    # Always compute original payload metrics before any processing
    original_encoded = raw_payload.encode("utf-8")
    payload_size_bytes = len(original_encoded)
    original_payload_hash = hashlib.sha256(original_encoded).hexdigest()

    # Determine storage mode
    is_minimal = PM_PROVENANCE_DETAIL == "minimal"

    if is_minimal:
        stored_payload: str | None = None
        stored_payload_hash: str | None = None
        payload_truncated = payload_size_bytes > MAX_RAW_PAYLOAD_BYTES
    else:
        # Strip credentials before storing
        redacted_payload = _strip_credentials(raw_payload)

        # Truncate at UTF-8 boundary
        stored_payload, payload_truncated = _truncate_utf8(
            redacted_payload, MAX_RAW_PAYLOAD_BYTES
        )

        # Hash the stored (redacted + possibly truncated) payload
        stored_payload_hash = hashlib.sha256(
            stored_payload.encode("utf-8")
        ).hexdigest()

    return RawPMResponse(
        response_id=response_id,
        pm_cycle_id=pm_cycle_id,
        profile=profile,
        model_id=model_id,
        timestamp=timestamp,
        prompt_version_id=prompt_version_id,
        candidate_ids_supplied=candidate_ids_supplied,
        raw_payload=stored_payload,
        original_payload_hash=original_payload_hash,
        stored_payload_hash=stored_payload_hash,
        parse_status=parse_status,
        attempt_ordinal=attempt_ordinal,
        payload_size_bytes=payload_size_bytes,
        payload_truncated=payload_truncated,
    )


# ---------------------------------------------------------------------------
# Lineage Linking
# ---------------------------------------------------------------------------


def link_response_to_lineages(
    response_id: str,
    lineage_ids: list[str],
    candidate_ids: list[str | None],
) -> list[ResponseLineageLink]:
    """Create join records linking a response to multiple candidate lineages.

    lineage_ids and candidate_ids must be the same length; use None for
    candidate_id when operating in legacy free-form mode.
    """
    if len(lineage_ids) != len(candidate_ids):
        raise ValueError(
            f"lineage_ids ({len(lineage_ids)}) and candidate_ids "
            f"({len(candidate_ids)}) must have the same length"
        )

    return [
        ResponseLineageLink(
            response_id=response_id,
            lineage_id=lineage_id,
            candidate_id=candidate_id,
        )
        for lineage_id, candidate_id in zip(lineage_ids, candidate_ids)
    ]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


_INSERT_RAW_RESPONSE_SQL = text("""
    INSERT INTO pm_raw_responses (
        response_id, pm_cycle_id, profile, model_id, timestamp,
        prompt_version_id, candidate_ids_supplied_json, raw_payload,
        original_payload_hash, stored_payload_hash, parse_status,
        attempt_ordinal, payload_size_bytes, payload_truncated
    ) VALUES (
        :response_id, :pm_cycle_id, :profile, :model_id, :timestamp,
        :prompt_version_id, :candidate_ids_supplied_json, :raw_payload,
        :original_payload_hash, :stored_payload_hash, :parse_status,
        :attempt_ordinal, :payload_size_bytes, :payload_truncated
    )
""")

_INSERT_LINEAGE_LINK_SQL = text("""
    INSERT INTO response_lineage_links (response_id, lineage_id, candidate_id)
    VALUES (:response_id, :lineage_id, :candidate_id)
""")


def persist_raw_response(engine, response: RawPMResponse) -> None:
    """Persist a raw PM response record with immutable INSERT and UNIQUE enforcement.

    Handles IntegrityError (duplicate response_id or cycle+attempt) gracefully.
    Fail-open: catches all exceptions, logs error, does not block pipeline.
    """
    try:
        with engine.connect() as conn:
            conn.execute(
                _INSERT_RAW_RESPONSE_SQL,
                {
                    "response_id": response.response_id,
                    "pm_cycle_id": response.pm_cycle_id,
                    "profile": response.profile,
                    "model_id": response.model_id,
                    "timestamp": response.timestamp.isoformat(),
                    "prompt_version_id": response.prompt_version_id,
                    "candidate_ids_supplied_json": json.dumps(
                        response.candidate_ids_supplied
                    ),
                    "raw_payload": response.raw_payload,
                    "original_payload_hash": response.original_payload_hash,
                    "stored_payload_hash": response.stored_payload_hash,
                    "parse_status": response.parse_status,
                    "attempt_ordinal": response.attempt_ordinal,
                    "payload_size_bytes": response.payload_size_bytes,
                    "payload_truncated": response.payload_truncated,
                },
            )
            conn.commit()
    except IntegrityError:
        log.warning(
            "Duplicate raw response record skipped: response_id=%s, "
            "pm_cycle_id=%s, attempt=%d",
            response.response_id,
            response.pm_cycle_id,
            response.attempt_ordinal,
        )
    except Exception:
        log.error(
            "Failed to persist raw PM response: response_id=%s, "
            "pm_cycle_id=%s, attempt=%d",
            response.response_id,
            response.pm_cycle_id,
            response.attempt_ordinal,
            exc_info=True,
        )


def persist_lineage_links(engine, links: list[ResponseLineageLink]) -> None:
    """Batch INSERT response-to-lineage link records.

    Handles IntegrityError for duplicates gracefully.
    Fail-open: catches all exceptions, logs error, does not block pipeline.
    """
    if not links:
        return

    try:
        with engine.connect() as conn:
            for link in links:
                try:
                    conn.execute(
                        _INSERT_LINEAGE_LINK_SQL,
                        {
                            "response_id": link.response_id,
                            "lineage_id": link.lineage_id,
                            "candidate_id": link.candidate_id,
                        },
                    )
                except IntegrityError:
                    log.warning(
                        "Duplicate lineage link skipped: response_id=%s, lineage_id=%s",
                        link.response_id,
                        link.lineage_id,
                    )
            conn.commit()
    except Exception:
        log.error(
            "Failed to persist lineage links for response_id=%s",
            links[0].response_id if links else "unknown",
            exc_info=True,
        )
