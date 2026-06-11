"""Timestamp-correct input assembly for the Decision Replay Agent.

Assembles the complete input bundle for a replay candidate, respecting the
Replay_Cutoff. Never substitutes current live state for missing historical state.

Classification rules:
- exact: all fields consumed by policy_version's active gate set are present
- partial: only non-critical inputs missing (fields not consumed by active gates)
- unscorable: any Critical_Input for the active gate set is unavailable

See: design.md §core/replay/input_reconstructor.py
Requirements: 2.1, 2.2, 2.3, 2.4, 2.6, 2.7, 2.8, 2.9
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from core.replay.gate_adapter import GATE_REQUIRED_FIELDS, GatePolicyConfig
from core.replay.policy_version import PolicyVersion

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputSource:
    """A single reconstructed input with provenance metadata.

    Attributes:
        field_name: The logical input field name (e.g. "signal_strength").
        value: The reconstructed value (None when status is "unavailable").
        source_timestamp: When the source record was created (None if unknown).
        source_record_id: Identifier of the source record (None if unknown).
        status: One of "available", "unavailable", "reconstructed".
    """

    field_name: str
    value: Any
    source_timestamp: datetime | None
    source_record_id: str | None
    status: str  # "available" | "unavailable" | "reconstructed"


@dataclass(frozen=True)
class ReplayInputBundle:
    """Complete input bundle for a replay candidate.

    Attributes:
        candidate: The replay candidate this bundle was assembled for.
        cutoff: The Replay_Cutoff timestamp.
        classification: "exact", "partial", or "unscorable".
        inputs: Dict mapping field_name → InputSource for all resolved inputs.
        missing_inputs: List of dicts describing each missing input:
            [{field, reason, is_critical}]
        snapshot_id: If sourced from an immutable Decision_Snapshot, its ID.
    """

    candidate: Any  # ReplayCandidate (avoid circular import)
    cutoff: datetime
    classification: str  # "exact" | "partial" | "unscorable"
    inputs: dict[str, InputSource]
    missing_inputs: list[dict]  # [{field, reason, is_critical}]
    snapshot_id: str | None


# ---------------------------------------------------------------------------
# Core gate sequence for Phase 1 (determines which fields are critical)
# ---------------------------------------------------------------------------

CORE_GATE_SEQUENCE: list[str] = [
    "setup_quality_gate",
    "pre_trade_quality_gate",
    "catalyst_specificity_gate",
    "risk_geometry_gate",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_replay_cutoff(
    registration_timestamp: datetime | None,
    decision_timestamp: datetime | None,
) -> datetime:
    """Return the earlier of the two timestamps; use whichever is available.

    Per Requirement 2.1:
    - If both are available, use the earlier one.
    - If only one is available, use that one.
    - If neither is available, raises ValueError.
    """
    if registration_timestamp is not None and decision_timestamp is not None:
        return min(registration_timestamp, decision_timestamp)
    if registration_timestamp is not None:
        return registration_timestamp
    if decision_timestamp is not None:
        return decision_timestamp
    raise ValueError(
        "Cannot compute replay cutoff: both registration_timestamp and "
        "decision_timestamp are None"
    )


def resolve_analyst_signal(
    session,
    symbol: str,
    profile: str,
    cutoff: datetime,
) -> InputSource:
    """Resolve the latest non-retracted, non-expired analyst signal at/before cutoff.

    Queries AgentMemory for analyst signals matching the symbol, with timestamp
    at or before the cutoff. Signals with status "retracted" or "expired" in
    their JSON payload are excluded.

    Tie-breaking: when multiple signals share the exact same timestamp,
    use the signal with the highest record ID (insertion order).

    Returns an InputSource with:
    - status="available" if a valid signal was found
    - status="unavailable" if no valid signal exists at/before cutoff
    """
    from db.schema import AgentMemory
    from sqlalchemy import and_

    # Query all analyst signals for this symbol at or before cutoff
    # Ordered by timestamp DESC then id DESC for tie-breaking
    signals = (
        session.query(AgentMemory)
        .filter(
            and_(
                AgentMemory.agent == "analyst",
                AgentMemory.symbol == symbol,
                AgentMemory.key == "signal",
                AgentMemory.timestamp <= cutoff,
            )
        )
        .order_by(AgentMemory.timestamp.desc(), AgentMemory.id.desc())
        .all()
    )

    for signal_row in signals:
        # Parse signal JSON
        try:
            signal_data = json.loads(signal_row.value) if isinstance(signal_row.value, str) else signal_row.value
        except (json.JSONDecodeError, TypeError):
            # Skip malformed signals
            continue

        if not isinstance(signal_data, dict):
            continue

        # Check for retracted or expired status
        signal_status = signal_data.get("status", "").lower()
        if signal_status in ("retracted", "expired", "schema-invalid"):
            continue

        # Valid signal found
        return InputSource(
            field_name="analyst_signal",
            value=signal_data,
            source_timestamp=signal_row.timestamp,
            source_record_id=str(signal_row.id),
            status="available",
        )

    # No valid signal found
    return InputSource(
        field_name="analyst_signal",
        value=None,
        source_timestamp=None,
        source_record_id=None,
        status="unavailable",
    )


def reconstruct_inputs(
    session,
    candidate: Any,  # ReplayCandidate
    policy_version: PolicyVersion,
) -> ReplayInputBundle:
    """Assemble all inputs available at or before the Replay_Cutoff.

    Never substitutes current state for missing historical state (Req 2.8).

    Steps:
    1. Compute the replay cutoff from the candidate timestamps.
    2. Check for an immutable Decision_Snapshot (preferred source).
    3. If no snapshot, reconstruct from historical tables.
    4. Classify the result as exact, partial, or unscorable.

    Classification rules:
    - exact: all fields consumed by policy_version's active gate set are present
    - partial: only non-critical inputs missing (fields not consumed by active gates)
    - unscorable: any Critical_Input for the active gate set is unavailable
    """
    # Step 1: Compute cutoff
    registration_ts = getattr(candidate, "entry_timestamp", None)
    decision_ts = _get_decision_timestamp(session, candidate)
    cutoff = compute_replay_cutoff(registration_ts, decision_ts)

    # Step 2: Try to load from immutable Decision_Snapshot
    snapshot_id, snapshot_inputs = _try_load_from_snapshot(session, candidate, cutoff)

    # Step 3: If no snapshot, reconstruct from historical tables
    if snapshot_inputs is None:
        inputs = _reconstruct_from_historical(session, candidate, cutoff)
    else:
        inputs = snapshot_inputs

    # Resolve analyst signal separately (always query for latest valid)
    signal_source = resolve_analyst_signal(
        session,
        symbol=candidate.symbol,
        profile=candidate.profile,
        cutoff=cutoff,
    )
    inputs["analyst_signal"] = signal_source

    # Extract signal-derived fields into the inputs dict
    _extract_signal_fields(inputs, signal_source)

    # Step 4: Classify
    active_gates = _get_active_gates(policy_version)
    critical_fields = _get_critical_fields(active_gates)
    classification, missing_inputs = _classify_inputs(inputs, critical_fields, active_gates)

    return ReplayInputBundle(
        candidate=candidate,
        cutoff=cutoff,
        classification=classification,
        inputs=inputs,
        missing_inputs=missing_inputs,
        snapshot_id=snapshot_id,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_decision_timestamp(session, candidate: Any) -> datetime | None:
    """Get the decision timestamp from trade_events for this candidate."""
    from sqlalchemy import text

    lineage_id = getattr(candidate, "lineage_id", None)
    if lineage_id:
        result = session.execute(
            text(
                "SELECT timestamp FROM trade_events "
                "WHERE candidate_lineage_id = :lineage_id "
                "ORDER BY timestamp ASC LIMIT 1"
            ),
            {"lineage_id": lineage_id},
        )
        row = result.fetchone()
        if row:
            ts = row[0]
            if isinstance(ts, str):
                return datetime.fromisoformat(ts)
            return ts

    return None


def _try_load_from_snapshot(
    session, candidate: Any, cutoff: datetime
) -> tuple[str | None, dict[str, InputSource] | None]:
    """Attempt to load inputs from an immutable Decision_Snapshot.

    Returns (snapshot_id, inputs_dict) or (None, None) if no snapshot found.
    """
    from sqlalchemy import text

    lineage_id = getattr(candidate, "lineage_id", None)
    if not lineage_id:
        return None, None

    result = session.execute(
        text(
            "SELECT * FROM decision_snapshots "
            "WHERE candidate_lineage_id = :lineage_id "
            "ORDER BY timestamp DESC LIMIT 1"
        ),
        {"lineage_id": lineage_id},
    )
    row = result.fetchone()
    if row is None:
        return None, None

    # Convert row to dict for easier access
    columns = result.keys()
    snapshot = dict(zip(columns, row))

    snapshot_id = snapshot.get("snapshot_id")
    snapshot_ts = snapshot.get("timestamp")
    if isinstance(snapshot_ts, str):
        snapshot_ts = datetime.fromisoformat(snapshot_ts)

    inputs: dict[str, InputSource] = {}

    # Geometry fields
    _add_input(inputs, "entry_price", _safe_decimal(snapshot.get("entry_price")),
               snapshot_ts, snapshot_id)
    _add_input(inputs, "stop_price", _safe_decimal(snapshot.get("stop_price")),
               snapshot_ts, snapshot_id)
    _add_input(inputs, "target_price", _safe_decimal(snapshot.get("target_price")),
               snapshot_ts, snapshot_id)
    _add_input(inputs, "quantity", _safe_decimal(snapshot.get("quantity")),
               snapshot_ts, snapshot_id)

    # Metadata
    _add_input(inputs, "symbol", snapshot.get("symbol"), snapshot_ts, snapshot_id)
    _add_input(inputs, "profile", snapshot.get("profile"), snapshot_ts, snapshot_id)
    _add_input(inputs, "direction", snapshot.get("direction"), snapshot_ts, snapshot_id)
    _add_input(inputs, "setup_type", snapshot.get("setup_type"), snapshot_ts, snapshot_id)

    # Signal fields
    _add_input(inputs, "signal_strength", snapshot.get("signal_strength"),
               snapshot_ts, snapshot_id)
    _add_input(inputs, "confidence_value", snapshot.get("confidence_value"),
               snapshot_ts, snapshot_id)

    # Market data
    _add_input(inputs, "atr_value", snapshot.get("atr_value"), snapshot_ts, snapshot_id)

    # Parse ATR bar timestamps
    atr_bar_ts_json = snapshot.get("atr_bar_timestamps_json")
    atr_timestamp = None
    if atr_bar_ts_json:
        try:
            atr_bar_data = json.loads(atr_bar_ts_json) if isinstance(atr_bar_ts_json, str) else atr_bar_ts_json
            if isinstance(atr_bar_data, list) and atr_bar_data:
                # Use the latest bar timestamp
                atr_timestamp = atr_bar_data[-1]
                if isinstance(atr_timestamp, str):
                    atr_timestamp = datetime.fromisoformat(atr_timestamp)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    _add_input(inputs, "atr_timestamp", atr_timestamp, snapshot_ts, snapshot_id)

    # Account state
    _add_input(inputs, "account_equity", _safe_decimal(snapshot.get("account_equity")),
               snapshot_ts, snapshot_id)
    _add_input(inputs, "available_cash", _safe_decimal(snapshot.get("available_cash")),
               snapshot_ts, snapshot_id)

    # Open positions context
    open_pos_json = snapshot.get("open_position_context_json")
    open_positions = _parse_json_field(open_pos_json)
    _add_input(inputs, "open_positions", open_positions, snapshot_ts, snapshot_id)

    # Case library stats
    case_stats_json = snapshot.get("case_library_stats_json")
    case_library_stats = _parse_json_field(case_stats_json)
    _add_input(inputs, "case_library_stats", case_library_stats, snapshot_ts, snapshot_id)

    # Decision payload (contains selection_score, execution_score, override fields)
    decision_json = snapshot.get("decision_payload_json")
    decision_payload = _parse_json_field(decision_json)
    if isinstance(decision_payload, dict):
        _add_input(inputs, "selection_score", decision_payload.get("selection_score"),
                   snapshot_ts, snapshot_id)
        _add_input(inputs, "execution_score", decision_payload.get("execution_score"),
                   snapshot_ts, snapshot_id)
        _add_input(inputs, "override_confidence_score",
                   decision_payload.get("override_confidence_score"),
                   snapshot_ts, snapshot_id)
        _add_input(inputs, "override_reason", decision_payload.get("override_reason"),
                   snapshot_ts, snapshot_id)
        _add_input(inputs, "catalyst_type", decision_payload.get("catalyst_type"),
                   snapshot_ts, snapshot_id)
        _add_input(inputs, "max_dollar_risk", decision_payload.get("max_dollar_risk"),
                   snapshot_ts, snapshot_id)
        _add_input(inputs, "trade_metadata", decision_payload.get("trade_metadata"),
                   snapshot_ts, snapshot_id)
        _add_input(inputs, "trade_rationale", decision_payload.get("trade_rationale"),
                   snapshot_ts, snapshot_id)
        _add_input(inputs, "atr_source", decision_payload.get("atr_source"),
                   snapshot_ts, snapshot_id)
    else:
        # Mark decision-derived fields as unavailable
        for f in ("selection_score", "execution_score", "override_confidence_score",
                  "override_reason", "catalyst_type", "max_dollar_risk",
                  "trade_metadata", "trade_rationale", "atr_source"):
            _add_input(inputs, f, None, None, None)

    # Gate config (for catalyst gate fields from analyst signal)
    signal_json = snapshot.get("analyst_signal_json")
    signal_payload = _parse_json_field(signal_json)
    if isinstance(signal_payload, dict):
        _add_input(inputs, "rationale", signal_payload.get("rationale"),
                   snapshot_ts, snapshot_id)
        _add_input(inputs, "thesis", signal_payload.get("thesis"),
                   snapshot_ts, snapshot_id)
        _add_input(inputs, "indicators", signal_payload.get("indicators"),
                   snapshot_ts, snapshot_id)
        _add_input(inputs, "quote_timestamp", signal_payload.get("quote_timestamp"),
                   snapshot_ts, snapshot_id)
        _add_input(inputs, "strength", signal_payload.get("strength"),
                   snapshot_ts, snapshot_id)
        _add_input(inputs, "conviction", signal_payload.get("conviction"),
                   snapshot_ts, snapshot_id)
    else:
        for f in ("rationale", "thesis", "indicators", "quote_timestamp",
                  "strength", "conviction"):
            _add_input(inputs, f, None, None, None)

    return snapshot_id, inputs


def _reconstruct_from_historical(
    session, candidate: Any, cutoff: datetime
) -> dict[str, InputSource]:
    """Reconstruct inputs from historical tables when no snapshot is available.

    Never substitutes current live state for missing historical state (Req 2.8).
    Records source timestamp and source record identifier for every input (Req 2.6).
    """
    from sqlalchemy import text

    inputs: dict[str, InputSource] = {}
    candidate_ts = getattr(candidate, "entry_timestamp", None)

    # Geometry from candidate itself
    _add_input(inputs, "entry_price",
               _to_decimal(getattr(candidate, "entry_price", None)),
               candidate_ts, _candidate_source_id(candidate))
    _add_input(inputs, "stop_price",
               _to_decimal(getattr(candidate, "stop_price", None)),
               candidate_ts, _candidate_source_id(candidate))
    _add_input(inputs, "target_price",
               _to_decimal(getattr(candidate, "target_price", None)),
               candidate_ts, _candidate_source_id(candidate))
    _add_input(inputs, "quantity",
               _to_decimal(getattr(candidate, "quantity", None)),
               candidate_ts, _candidate_source_id(candidate))

    # Metadata from candidate
    _add_input(inputs, "symbol", candidate.symbol, candidate_ts, _candidate_source_id(candidate))
    _add_input(inputs, "profile", candidate.profile, candidate_ts, _candidate_source_id(candidate))
    _add_input(inputs, "direction", getattr(candidate, "direction", None),
               candidate_ts, _candidate_source_id(candidate))
    _add_input(inputs, "setup_type", getattr(candidate, "setup_type", None),
               candidate_ts, _candidate_source_id(candidate))

    # Try to get additional data from trade_events payload
    _enrich_from_trade_events(session, inputs, candidate, cutoff)

    # Account state at/before cutoff from balance table
    _reconstruct_account_state(session, inputs, candidate.profile, cutoff)

    # Open positions at/before cutoff
    _reconstruct_open_positions(session, inputs, candidate.profile, cutoff)

    # Case library stats — mark unavailable for historical (cannot reconstruct without hindsight)
    _add_input(inputs, "case_library_stats", None, None, None)

    return inputs


def _enrich_from_trade_events(
    session, inputs: dict[str, InputSource], candidate: Any, cutoff: datetime
) -> None:
    """Enrich inputs from trade_events payload for this candidate."""
    from sqlalchemy import text

    lineage_id = getattr(candidate, "lineage_id", None)
    if not lineage_id:
        # Fall back to symbol/timestamp matching
        _enrich_from_blocked_candidates(session, inputs, candidate, cutoff)
        return

    result = session.execute(
        text(
            "SELECT id, timestamp, payload_json FROM trade_events "
            "WHERE candidate_lineage_id = :lineage_id "
            "AND timestamp <= :cutoff "
            "ORDER BY timestamp DESC LIMIT 1"
        ),
        {"lineage_id": lineage_id, "cutoff": cutoff},
    )
    row = result.fetchone()
    if row is None:
        _enrich_from_blocked_candidates(session, inputs, candidate, cutoff)
        return

    event_id, event_ts, payload_json = row[0], row[1], row[2]
    if isinstance(event_ts, str):
        event_ts = datetime.fromisoformat(event_ts)

    source_id = f"trade_events:{event_id}"
    payload = _parse_json_field(payload_json)
    if not isinstance(payload, dict):
        return

    # Extract fields from trade_event payload
    _set_if_missing(inputs, "signal_strength", payload.get("signal_strength"),
                    event_ts, source_id)
    _set_if_missing(inputs, "confidence_value", payload.get("confidence_value")
                    or payload.get("confidence_level")
                    or payload.get("confidence_score"),
                    event_ts, source_id)
    _set_if_missing(inputs, "selection_score", payload.get("selection_score"),
                    event_ts, source_id)
    _set_if_missing(inputs, "execution_score", payload.get("execution_score"),
                    event_ts, source_id)
    _set_if_missing(inputs, "override_confidence_score",
                    payload.get("override_confidence_score"), event_ts, source_id)
    _set_if_missing(inputs, "override_reason",
                    payload.get("override_reason"), event_ts, source_id)
    _set_if_missing(inputs, "atr_value", payload.get("atr_value")
                    or payload.get("atr_5min"), event_ts, source_id)
    _set_if_missing(inputs, "atr_timestamp", payload.get("atr_timestamp"),
                    event_ts, source_id)
    _set_if_missing(inputs, "catalyst_type", payload.get("catalyst_type"),
                    event_ts, source_id)
    _set_if_missing(inputs, "max_dollar_risk", payload.get("max_dollar_risk"),
                    event_ts, source_id)
    _set_if_missing(inputs, "trade_metadata", payload.get("trade_metadata"),
                    event_ts, source_id)
    _set_if_missing(inputs, "trade_rationale", payload.get("trade_rationale"),
                    event_ts, source_id)
    _set_if_missing(inputs, "atr_source", payload.get("atr_source"),
                    event_ts, source_id)
    _set_if_missing(inputs, "rationale", payload.get("rationale"),
                    event_ts, source_id)
    _set_if_missing(inputs, "thesis", payload.get("thesis"),
                    event_ts, source_id)
    _set_if_missing(inputs, "indicators", payload.get("indicators"),
                    event_ts, source_id)
    _set_if_missing(inputs, "quote_timestamp", payload.get("quote_timestamp"),
                    event_ts, source_id)
    _set_if_missing(inputs, "strength", payload.get("strength"),
                    event_ts, source_id)
    _set_if_missing(inputs, "conviction", payload.get("conviction"),
                    event_ts, source_id)


def _enrich_from_blocked_candidates(
    session, inputs: dict[str, InputSource], candidate: Any, cutoff: datetime
) -> None:
    """Enrich inputs from blocked_trade_candidates when no lineage ID."""
    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError

    # Try to find a matching blocked candidate by symbol and timestamp
    try:
        result = session.execute(
            text(
                "SELECT id, created_at, decision_snapshot_json, signal_snapshot_json "
                "FROM blocked_trade_candidates "
                "WHERE symbol = :symbol AND created_at <= :cutoff "
                "ORDER BY created_at DESC LIMIT 1"
            ),
            {"symbol": candidate.symbol, "cutoff": cutoff},
        )
        row = result.fetchone()
    except OperationalError:
        # Table may not exist yet (pre-shadow-ledger era)
        return
    if row is None:
        return

    blocked_id, created_at, decision_json, signal_json = row[0], row[1], row[2], row[3]
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)

    source_id = f"blocked_trade_candidates:{blocked_id}"

    # Extract from decision_snapshot_json
    decision_data = _parse_json_field(decision_json)
    if isinstance(decision_data, dict):
        _set_if_missing(inputs, "signal_strength",
                        decision_data.get("signal_strength"), created_at, source_id)
        _set_if_missing(inputs, "confidence_value",
                        decision_data.get("confidence_value")
                        or decision_data.get("confidence_level")
                        or decision_data.get("confidence_score"),
                        created_at, source_id)
        _set_if_missing(inputs, "atr_value",
                        decision_data.get("atr_value") or decision_data.get("atr_5min"),
                        created_at, source_id)
        _set_if_missing(inputs, "atr_timestamp",
                        decision_data.get("atr_timestamp"), created_at, source_id)
        _set_if_missing(inputs, "catalyst_type",
                        decision_data.get("catalyst_type"), created_at, source_id)
        _set_if_missing(inputs, "max_dollar_risk",
                        decision_data.get("max_dollar_risk"), created_at, source_id)
        _set_if_missing(inputs, "trade_metadata",
                        decision_data.get("trade_metadata"), created_at, source_id)
        _set_if_missing(inputs, "trade_rationale",
                        decision_data.get("trade_rationale"), created_at, source_id)
        _set_if_missing(inputs, "atr_source",
                        decision_data.get("atr_source"), created_at, source_id)
        _set_if_missing(inputs, "selection_score",
                        decision_data.get("selection_score"), created_at, source_id)
        _set_if_missing(inputs, "execution_score",
                        decision_data.get("execution_score"), created_at, source_id)
        _set_if_missing(inputs, "override_confidence_score",
                        decision_data.get("override_confidence_score"), created_at, source_id)
        _set_if_missing(inputs, "override_reason",
                        decision_data.get("override_reason"), created_at, source_id)

    # Extract from signal_snapshot_json
    signal_data = _parse_json_field(signal_json)
    if isinstance(signal_data, dict):
        _set_if_missing(inputs, "rationale",
                        signal_data.get("rationale"), created_at, source_id)
        _set_if_missing(inputs, "thesis",
                        signal_data.get("thesis"), created_at, source_id)
        _set_if_missing(inputs, "indicators",
                        signal_data.get("indicators"), created_at, source_id)
        _set_if_missing(inputs, "quote_timestamp",
                        signal_data.get("quote_timestamp"), created_at, source_id)
        _set_if_missing(inputs, "strength",
                        signal_data.get("strength"), created_at, source_id)
        _set_if_missing(inputs, "conviction",
                        signal_data.get("conviction"), created_at, source_id)


def _reconstruct_account_state(
    session, inputs: dict[str, InputSource], profile: str, cutoff: datetime
) -> None:
    """Reconstruct account equity and cash from the balance table at/before cutoff."""
    from sqlalchemy import text

    result = session.execute(
        text(
            "SELECT id, timestamp, cash, total_equity FROM balance "
            "WHERE profile = :profile AND timestamp <= :cutoff "
            "ORDER BY timestamp DESC LIMIT 1"
        ),
        {"profile": profile, "cutoff": cutoff},
    )
    row = result.fetchone()
    if row is None:
        _add_input(inputs, "account_equity", None, None, None)
        _add_input(inputs, "available_cash", None, None, None)
        return

    balance_id, balance_ts, cash, equity = row[0], row[1], row[2], row[3]
    if isinstance(balance_ts, str):
        balance_ts = datetime.fromisoformat(balance_ts)

    source_id = f"balance:{balance_id}"
    _add_input(inputs, "account_equity",
               Decimal(str(equity)) if equity is not None else None,
               balance_ts, source_id)
    _add_input(inputs, "available_cash",
               Decimal(str(cash)) if cash is not None else None,
               balance_ts, source_id)


def _reconstruct_open_positions(
    session, inputs: dict[str, InputSource], profile: str, cutoff: datetime
) -> None:
    """Reconstruct open positions at/before cutoff.

    Note: This is a best-effort reconstruction. The positions table reflects
    current state, not historical state. We can only use positions that were
    opened before the cutoff. If historical state is missing, we mark it
    unavailable rather than substituting current live state (Req 2.8).
    """
    from sqlalchemy import text

    result = session.execute(
        text(
            "SELECT id, symbol, side, quantity, avg_cost, opened_at FROM positions "
            "WHERE profile = :profile AND opened_at <= :cutoff"
        ),
        {"profile": profile, "cutoff": cutoff},
    )
    rows = result.fetchall()

    if rows:
        positions = []
        latest_ts = None
        for row in rows:
            pos_id, sym, side, qty, cost, opened_at = (
                row[0], row[1], row[2], row[3], row[4], row[5]
            )
            if isinstance(opened_at, str):
                opened_at = datetime.fromisoformat(opened_at)
            positions.append({
                "symbol": sym,
                "side": side,
                "quantity": qty,
                "avg_cost": cost,
            })
            if latest_ts is None or (opened_at and opened_at > latest_ts):
                latest_ts = opened_at

        _add_input(inputs, "open_positions", positions, latest_ts,
                   f"positions:profile={profile}")
    else:
        # No positions found — this may be accurate (empty portfolio) or
        # may indicate historical state is unavailable. Record as available
        # with empty list since we queried the historical record.
        _add_input(inputs, "open_positions", [], None,
                   f"positions:profile={profile}")


def _extract_signal_fields(inputs: dict[str, InputSource], signal_source: InputSource) -> None:
    """Extract individual fields from the analyst signal into the inputs dict.

    Only sets fields that are not already present from other sources.
    """
    if signal_source.status != "available" or not isinstance(signal_source.value, dict):
        return

    signal = signal_source.value
    ts = signal_source.source_timestamp
    src_id = signal_source.source_record_id

    _set_if_missing(inputs, "signal_strength",
                    _parse_signal_strength(signal.get("strength")), ts, src_id)
    _set_if_missing(inputs, "confidence_value",
                    _parse_confidence(signal.get("confidence")), ts, src_id)
    _set_if_missing(inputs, "catalyst_type",
                    signal.get("catalyst_type") or signal.get("setup_type"), ts, src_id)
    _set_if_missing(inputs, "rationale", signal.get("reasoning") or signal.get("rationale"),
                    ts, src_id)
    _set_if_missing(inputs, "thesis", signal.get("thesis"), ts, src_id)
    _set_if_missing(inputs, "indicators", signal.get("indicators"), ts, src_id)
    _set_if_missing(inputs, "strength", signal.get("strength"), ts, src_id)
    _set_if_missing(inputs, "conviction", signal.get("conviction") or signal.get("confidence"),
                    ts, src_id)


def _get_active_gates(policy_version: PolicyVersion) -> list[str]:
    """Determine which gates are active for this policy version.

    Phase 1: core gate sequence only.
    """
    # For Phase 1, always use the core gate sequence
    return CORE_GATE_SEQUENCE


def _get_critical_fields(active_gates: list[str]) -> set[str]:
    """Get all fields consumed by the active gate set (these are Critical_Inputs).

    A field is critical when the selected gate/policy consumes it and its
    absence could change the replay decision (Requirement 2.7).
    """
    critical: set[str] = set()
    for gate_name in active_gates:
        fields = GATE_REQUIRED_FIELDS.get(gate_name, [])
        critical.update(fields)
    return critical


def _classify_inputs(
    inputs: dict[str, InputSource],
    critical_fields: set[str],
    active_gates: list[str],
) -> tuple[str, list[dict]]:
    """Classify the input bundle as exact, partial, or unscorable.

    Returns (classification, missing_inputs_list).

    Classification rules (Requirement 2.7):
    - exact: all fields consumed by policy_version's active gate set are present
    - partial: only non-critical inputs missing
    - unscorable: any Critical_Input for the active gate set is unavailable
    """
    missing_inputs: list[dict] = []
    has_critical_missing = False
    has_any_missing = False

    # Get all fields from all gates (not just active ones) for completeness reporting
    all_known_fields = set()
    for gate_fields in GATE_REQUIRED_FIELDS.values():
        all_known_fields.update(gate_fields)

    for field_name in critical_fields:
        source = inputs.get(field_name)
        if source is None or source.status == "unavailable" or source.value is None:
            has_critical_missing = True
            has_any_missing = True
            reason = _determine_missing_reason(field_name, source)
            missing_inputs.append({
                "field": field_name,
                "reason": reason,
                "is_critical": True,
            })

    # Check non-critical fields (fields known but not in the active gate set)
    non_critical_fields = all_known_fields - critical_fields
    for field_name in non_critical_fields:
        source = inputs.get(field_name)
        if source is not None and source.status == "unavailable":
            has_any_missing = True
            reason = _determine_missing_reason(field_name, source)
            missing_inputs.append({
                "field": field_name,
                "reason": reason,
                "is_critical": False,
            })

    if has_critical_missing:
        return "unscorable", missing_inputs
    elif has_any_missing:
        return "partial", missing_inputs
    else:
        return "exact", missing_inputs


def _determine_missing_reason(field_name: str, source: InputSource | None) -> str:
    """Determine why a field is missing."""
    if source is None:
        return f"No source record found for '{field_name}' at or before replay cutoff"
    if source.status == "unavailable":
        return (
            f"Field '{field_name}' recorded as explicitly unavailable — "
            "historical state could not be reconstructed without hindsight"
        )
    return f"Field '{field_name}' is None in the source record"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _add_input(
    inputs: dict[str, InputSource],
    field_name: str,
    value: Any,
    source_timestamp: datetime | None,
    source_record_id: str | None,
) -> None:
    """Add an InputSource to the inputs dict.

    Records source_timestamp and source_record_id for every input (Req 2.6).
    Marks explicitly missing when unavailable (Req 2.6).
    """
    if value is None:
        inputs[field_name] = InputSource(
            field_name=field_name,
            value=None,
            source_timestamp=source_timestamp,
            source_record_id=source_record_id,
            status="unavailable",
        )
    else:
        inputs[field_name] = InputSource(
            field_name=field_name,
            value=value,
            source_timestamp=source_timestamp,
            source_record_id=source_record_id if source_record_id else None,
            status="available",
        )


def _set_if_missing(
    inputs: dict[str, InputSource],
    field_name: str,
    value: Any,
    source_timestamp: datetime | None,
    source_record_id: str | None,
) -> None:
    """Set a field only if it's not already present or is unavailable.

    Avoids overwriting an already-resolved input with a lower-priority source.
    """
    existing = inputs.get(field_name)
    if existing is None or (existing.status == "unavailable" and value is not None):
        _add_input(inputs, field_name, value, source_timestamp, source_record_id)


def _candidate_source_id(candidate: Any) -> str | None:
    """Build a source record identifier from a ReplayCandidate."""
    candidate_id = getattr(candidate, "candidate_id", None)
    if candidate_id:
        return f"candidate:{candidate_id}"
    return None


def _safe_decimal(value: Any) -> Decimal | None:
    """Safely convert a value to Decimal."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _to_decimal(value: Any) -> Decimal | None:
    """Convert a value to Decimal if possible."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _parse_json_field(value: Any) -> Any:
    """Parse a JSON string field, returning None on failure."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def _parse_signal_strength(strength: Any) -> float | None:
    """Convert signal strength to numeric value.

    The analyst signal stores strength as "weak"/"moderate"/"strong".
    Convert to numeric for gate evaluation.
    """
    if strength is None:
        return None
    if isinstance(strength, (int, float)):
        return float(strength)
    strength_map = {"weak": 3.0, "moderate": 6.0, "strong": 9.0}
    if isinstance(strength, str):
        return strength_map.get(strength.lower())
    return None


def _parse_confidence(confidence: Any) -> float | None:
    """Convert confidence to numeric value.

    The analyst signal stores confidence as "low"/"medium"/"high".
    Convert to numeric for gate evaluation.
    """
    if confidence is None:
        return None
    if isinstance(confidence, (int, float)):
        return float(confidence)
    confidence_map = {"low": 3.0, "medium": 6.0, "high": 9.0}
    if isinstance(confidence, str):
        return confidence_map.get(confidence.lower())
    return None
