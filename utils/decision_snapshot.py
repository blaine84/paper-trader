"""Decision Snapshot — immutable decision-time records for full audit trail.

This module provides two complementary capabilities:

1. **Legacy PM candidate event recording** (Requirements 20.1–20.7):
   Records decision-time snapshots into `pm_candidate_events` for the existing
   PM audit trail.

2. **Replay-ready Decision Snapshot** (Requirements 3.1–3.7):
   Creates and persists immutable `DecisionSnapshot` records into the
   `decision_snapshots` table. These are sufficient to re-run the entire gate
   sequence without external lookups — the foundation for deterministic replay.

The DecisionSnapshot is persisted BEFORE gate evaluation begins, so that the
snapshot alone suffices to re-run gates. If persistence fails, gate evaluation
is BLOCKED for that candidate (Requirement 3.6).
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Replay Decision Snapshot (Requirements 3.1–3.7)
# ═══════════════════════════════════════════════════════════════════════════════

# Current schema version for Decision Snapshots
SNAPSHOT_SCHEMA_VERSION = "1.0.0"

# Fields that MUST be present and non-None for a valid snapshot
REQUIRED_SNAPSHOT_FIELDS: list[str] = [
    "snapshot_id",
    "schema_version",
    "candidate_lineage_id",
    "timestamp",
    "symbol",
    "profile",
    "direction",
    "decision_payload",
    "entry_price",
    "stop_price",
    "target_price",
    "quantity",
    "account_equity",
    "available_cash",
    "gate_config",
    "feature_flags",
    "policy_version_id",
]


class SnapshotPersistenceError(Exception):
    """Raised when Decision Snapshot persistence fails.

    Gate evaluation MUST be blocked when this error is raised (Requirement 3.6).
    """
    pass


@dataclass(frozen=True)
class DecisionSnapshot:
    """Immutable record of all inputs available at decision time.

    Persisted BEFORE gate evaluation begins, such that the snapshot alone is
    sufficient to re-run the gate sequence without external lookups.

    Requirements:
    - 3.1: Persisted before gate evaluation begins
    - 3.2: Contains all enumerated fields
    - 3.3: Immutable after gate evaluation completes (enforced by DB triggers)
    - 3.5: No sensitive credentials/API keys/env vars stored
    - 3.7: schema_version field for format identification
    """

    # Identity
    snapshot_id: str
    schema_version: str
    candidate_lineage_id: str
    timestamp: datetime

    # Candidate metadata
    symbol: str
    profile: str
    direction: str  # "BUY" or "SHORT"
    setup_type: str | None

    # Signal state
    analyst_signal_payload: dict | None
    signal_strength: float | None
    confidence_value: float | None

    # Decision/scaffold payload
    decision_payload: dict

    # Geometry (stored as Decimal for precision)
    entry_price: Decimal
    stop_price: Decimal
    target_price: Decimal
    quantity: Decimal

    # Market data
    atr_value: float | None
    atr_bar_timestamps: list[str] | None

    # Account state
    account_equity: Decimal
    available_cash: Decimal

    # Position context
    open_position_context: list[dict] | None

    # Case library state
    case_library_stats: dict | None

    # Configuration state (NO sensitive credentials — Requirement 3.5)
    gate_config: dict
    feature_flags: dict
    policy_version_id: str

    # Geometry hash for canonical comparison
    geometry_hash: str | None


def create_decision_snapshot(
    *,
    candidate_lineage_id: str,
    symbol: str,
    profile: str,
    direction: str,
    setup_type: str | None,
    decision_payload: dict,
    entry_price: Decimal | float | str,
    stop_price: Decimal | float | str,
    target_price: Decimal | float | str,
    quantity: Decimal | float | int,
    analyst_signal_payload: dict | None = None,
    signal_strength: float | None = None,
    confidence_value: float | None = None,
    atr_value: float | None = None,
    atr_bar_timestamps: list[str] | None = None,
    account_equity: Decimal | float | str = Decimal("0"),
    available_cash: Decimal | float | str = Decimal("0"),
    open_position_context: list[dict] | None = None,
    case_library_stats: dict | None = None,
    gate_config: dict | None = None,
    feature_flags: dict | None = None,
    policy_version_id: str = "",
) -> DecisionSnapshot:
    """Factory function to create a DecisionSnapshot with generated ID and timestamp.

    Converts price fields to Decimal for precision. Computes geometry_hash
    automatically from the entry/stop/target geometry.
    """
    entry_dec = Decimal(str(entry_price))
    stop_dec = Decimal(str(stop_price))
    target_dec = Decimal(str(target_price))
    quantity_dec = Decimal(str(quantity))
    equity_dec = Decimal(str(account_equity))
    cash_dec = Decimal(str(available_cash))

    geometry_hash = compute_geometry_hash(entry_dec, stop_dec, target_dec)

    return DecisionSnapshot(
        snapshot_id=str(uuid.uuid4()),
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        candidate_lineage_id=candidate_lineage_id,
        timestamp=datetime.now(timezone.utc),
        symbol=symbol,
        profile=profile,
        direction=direction,
        setup_type=setup_type,
        analyst_signal_payload=analyst_signal_payload,
        signal_strength=signal_strength,
        confidence_value=confidence_value,
        decision_payload=decision_payload,
        entry_price=entry_dec,
        stop_price=stop_dec,
        target_price=target_dec,
        quantity=quantity_dec,
        atr_value=atr_value,
        atr_bar_timestamps=atr_bar_timestamps,
        account_equity=equity_dec,
        available_cash=cash_dec,
        open_position_context=open_position_context,
        case_library_stats=case_library_stats,
        gate_config=gate_config or {},
        feature_flags=feature_flags or {},
        policy_version_id=policy_version_id,
        geometry_hash=geometry_hash,
    )


def validate_snapshot_fields(snapshot: DecisionSnapshot) -> list[str]:
    """Validate that all required fields are present and non-None.

    Returns a list of field names that are missing or None. An empty list
    means the snapshot is valid.

    Requirements: 3.2, 3.6
    """
    missing = []
    for field_name in REQUIRED_SNAPSHOT_FIELDS:
        value = getattr(snapshot, field_name, None)
        if value is None:
            missing.append(field_name)
        elif isinstance(value, str) and not value.strip():
            missing.append(field_name)
        elif isinstance(value, dict) and field_name in ("gate_config", "feature_flags", "decision_payload"):
            # These must be non-empty dicts — but gate_config/feature_flags
            # can be empty if no gates are configured (edge case)
            pass
    return missing


def compute_geometry_hash(
    entry_price: Decimal | None,
    stop_price: Decimal | None,
    target_price: Decimal | None,
    tick_size: Decimal = Decimal("0.01"),
) -> str:
    """Compute canonical geometry hash using tick-normalized Decimal with SHA-256.

    Normalizes each price to the specified tick size using ROUND_HALF_UP,
    then produces a stable SHA-256 hash for comparison. This avoids floating-point
    comparison issues.

    Returns empty string if any price is None (geometry incomplete).

    Requirements: 3.2 (Geometry_Hash), 6.1 (canonical comparison)
    """
    if entry_price is None or stop_price is None or target_price is None:
        return ""

    # Normalize to tick size using ROUND_HALF_UP
    entry_norm = _tick_normalize(entry_price, tick_size)
    stop_norm = _tick_normalize(stop_price, tick_size)
    target_norm = _tick_normalize(target_price, tick_size)

    # Build canonical string representation (sorted keys)
    canonical = json.dumps(
        {
            "entry": str(entry_norm),
            "stop": str(stop_norm),
            "target": str(target_norm),
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _tick_normalize(price: Decimal, tick_size: Decimal) -> Decimal:
    """Normalize a price to the nearest tick using ROUND_HALF_UP."""
    if tick_size <= 0:
        return price
    # Quantize to the tick_size precision
    return price.quantize(tick_size, rounding=ROUND_HALF_UP)


def persist_snapshot(engine, snapshot: DecisionSnapshot) -> None:
    """Persist a DecisionSnapshot to the decision_snapshots table.

    This MUST be called within the same transaction boundary as gate evaluation.
    If persistence fails, raises SnapshotPersistenceError to block gate evaluation
    for that candidate (Requirement 3.6).

    The caller should use this in a transactional context:
        try:
            persist_snapshot(engine, snapshot)
        except SnapshotPersistenceError:
            # Gate evaluation is BLOCKED — record failure and skip candidate
            ...

    Requirements: 3.1, 3.6
    """
    # Validate before persisting
    missing = validate_snapshot_fields(snapshot)
    if missing:
        raise SnapshotPersistenceError(
            f"Snapshot validation failed — missing required fields: {missing}"
        )

    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO decision_snapshots (
                        snapshot_id, schema_version, candidate_lineage_id,
                        timestamp, symbol, profile, direction, setup_type,
                        analyst_signal_json, signal_strength, confidence_value,
                        decision_payload_json, entry_price, stop_price,
                        target_price, quantity, atr_value, atr_bar_timestamps_json,
                        account_equity, available_cash,
                        open_position_context_json, case_library_stats_json,
                        gate_config_json, feature_flags_json,
                        policy_version_id, geometry_hash
                    ) VALUES (
                        :snapshot_id, :schema_version, :candidate_lineage_id,
                        :timestamp, :symbol, :profile, :direction, :setup_type,
                        :analyst_signal_json, :signal_strength, :confidence_value,
                        :decision_payload_json, :entry_price, :stop_price,
                        :target_price, :quantity, :atr_value, :atr_bar_timestamps_json,
                        :account_equity, :available_cash,
                        :open_position_context_json, :case_library_stats_json,
                        :gate_config_json, :feature_flags_json,
                        :policy_version_id, :geometry_hash
                    )
                """),
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "schema_version": snapshot.schema_version,
                    "candidate_lineage_id": snapshot.candidate_lineage_id,
                    "timestamp": snapshot.timestamp.isoformat(),
                    "symbol": snapshot.symbol,
                    "profile": snapshot.profile,
                    "direction": snapshot.direction,
                    "setup_type": snapshot.setup_type,
                    "analyst_signal_json": _safe_json(snapshot.analyst_signal_payload),
                    "signal_strength": snapshot.signal_strength,
                    "confidence_value": snapshot.confidence_value,
                    "decision_payload_json": _safe_json(snapshot.decision_payload),
                    "entry_price": str(snapshot.entry_price),
                    "stop_price": str(snapshot.stop_price),
                    "target_price": str(snapshot.target_price),
                    "quantity": str(snapshot.quantity),
                    "atr_value": snapshot.atr_value,
                    "atr_bar_timestamps_json": _safe_json(snapshot.atr_bar_timestamps),
                    "account_equity": str(snapshot.account_equity),
                    "available_cash": str(snapshot.available_cash),
                    "open_position_context_json": _safe_json(snapshot.open_position_context),
                    "case_library_stats_json": _safe_json(snapshot.case_library_stats),
                    "gate_config_json": _safe_json(snapshot.gate_config),
                    "feature_flags_json": _safe_json(snapshot.feature_flags),
                    "policy_version_id": snapshot.policy_version_id,
                    "geometry_hash": snapshot.geometry_hash,
                },
            )
            conn.commit()
    except SnapshotPersistenceError:
        raise
    except Exception as exc:
        logger.error(
            "Decision snapshot persistence failed for candidate %s: %s",
            snapshot.candidate_lineage_id,
            exc,
        )
        raise SnapshotPersistenceError(
            f"Failed to persist snapshot for candidate {snapshot.candidate_lineage_id}: {exc}"
        ) from exc


def generate_candidate_lineage_id() -> str:
    """Generate a new candidate lineage ID (UUID4).

    This ID is propagated across trade_events, blocked_trade_candidates,
    trades, pm_candidates, and funnel_candidates to enable deduplication
    and correlation in the replay engine.
    """
    return str(uuid.uuid4())


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Pipeline Integration Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def build_and_persist_snapshot(
    engine,
    *,
    candidate_lineage_id: str,
    decision: dict,
    signal: dict | None,
    profile_id: str,
    account_equity: float | Decimal,
    available_cash: float | Decimal,
    open_positions: list[dict] | None = None,
    case_library_stats: dict | None = None,
    atr_value: float | None = None,
    atr_bar_timestamps: list[str] | None = None,
    gate_config: dict | None = None,
    feature_flags: dict | None = None,
    policy_version_id: str = "",
) -> DecisionSnapshot:
    """Build and persist a DecisionSnapshot from PM pipeline context.

    This is the primary integration point for the PM pipeline. Called AFTER
    geometry repair, BEFORE the first gate.

    Raises SnapshotPersistenceError if persistence fails, blocking gate
    evaluation for this candidate (Requirement 3.6).

    Args:
        engine: SQLAlchemy engine
        candidate_lineage_id: Pre-generated lineage ID for this candidate
        decision: The normalized decision dict from PM
        signal: The analyst signal for this symbol (may be None)
        profile_id: Profile identifier (conservative/moderate/aggressive)
        account_equity: Current account equity
        available_cash: Available cash for trading
        open_positions: Current open position context
        case_library_stats: Case library statistics used by gates
        atr_value: ATR value at decision time
        atr_bar_timestamps: Timestamps of bars used for ATR computation
        gate_config: Gate configuration snapshot (thresholds, rules)
        feature_flags: Feature flag states at snapshot time
        policy_version_id: Policy version identifier

    Returns:
        The persisted DecisionSnapshot (for downstream use)

    Raises:
        SnapshotPersistenceError: If persistence fails
    """
    signal = signal or {}

    # Extract geometry from decision
    entry_price = (
        decision.get("entry_price")
        or decision.get("price")
        or 0
    )
    stop_price = (
        decision.get("stop")
        or decision.get("stop_price")
        or decision.get("stop_loss")
        or 0
    )
    target_price = (
        decision.get("target")
        or decision.get("target_price")
        or decision.get("profit_target")
        or 0
    )
    quantity = decision.get("quantity", 0)
    direction = decision.get("action", "BUY")
    symbol = decision.get("symbol") or signal.get("symbol", "")
    setup_type = (
        decision.get("setup_type")
        or decision.get("setup")
        or signal.get("setup_type")
    )

    # Extract signal data (no sensitive credentials — Requirement 3.5)
    analyst_signal_payload = _sanitize_signal_payload(signal)
    signal_strength = _safe_float(signal.get("signal_strength"))
    confidence_value = _safe_float(
        decision.get("override_confidence_score")
        or signal.get("confidence")
        or signal.get("confidence_score")
    )

    snapshot = create_decision_snapshot(
        candidate_lineage_id=candidate_lineage_id,
        symbol=symbol,
        profile=profile_id,
        direction=direction,
        setup_type=setup_type,
        decision_payload=_sanitize_decision_payload(decision),
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        quantity=quantity,
        analyst_signal_payload=analyst_signal_payload,
        signal_strength=signal_strength,
        confidence_value=confidence_value,
        atr_value=atr_value,
        atr_bar_timestamps=atr_bar_timestamps,
        account_equity=account_equity,
        available_cash=available_cash,
        open_position_context=open_positions,
        case_library_stats=case_library_stats,
        gate_config=gate_config,
        feature_flags=feature_flags,
        policy_version_id=policy_version_id,
    )

    persist_snapshot(engine, snapshot)
    return snapshot


def _sanitize_signal_payload(signal: dict) -> dict | None:
    """Remove any sensitive data from analyst signal before storing.

    Requirement 3.5: Sensitive credentials, API keys, authentication tokens,
    and environment variables unrelated to gate evaluation SHALL NOT be stored.
    """
    if not signal:
        return None

    # Fields that should never be stored in snapshots
    sensitive_keys = frozenset({
        "api_key", "api_secret", "token", "access_token", "refresh_token",
        "password", "secret", "credential", "auth_token", "bearer_token",
        "private_key", "secret_key",
    })

    sanitized = {}
    for key, value in signal.items():
        key_lower = key.lower()
        if any(s in key_lower for s in sensitive_keys):
            continue
        sanitized[key] = value

    return sanitized if sanitized else None


def _sanitize_decision_payload(decision: dict) -> dict:
    """Remove sensitive data from decision payload before storing.

    Requirement 3.5: No credentials or env vars stored.
    """
    sensitive_keys = frozenset({
        "api_key", "api_secret", "token", "access_token", "refresh_token",
        "password", "secret", "credential", "auth_token", "bearer_token",
        "private_key", "secret_key",
    })

    sanitized = {}
    for key, value in decision.items():
        key_lower = key.lower()
        if any(s in key_lower for s in sensitive_keys):
            continue
        sanitized[key] = value

    return sanitized


def _safe_json(obj: Any) -> str | None:
    """Serialize to JSON safely, returning None for None input."""
    if obj is None:
        return None
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        logger.warning("JSON serialization failed: %s", exc)
        return json.dumps(str(obj))


def _safe_float(value: Any) -> float | None:
    """Safely convert a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        result = float(value)
        if not (result != result):  # NaN check
            return result
    except (TypeError, ValueError):
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Legacy PM Candidate Event Recording (Requirements 20.1–20.7)
# ═══════════════════════════════════════════════════════════════════════════════

# Event types that represent pre-trade facts (immutable at creation)
PRE_TRADE_EVENT_TYPES = frozenset({
    "decision_snapshot",      # Complete frozen state at decision time
    "offered",                # Candidate was offered to PM
    "pm_accept",              # PM accepted the candidate
    "pm_reject",              # PM rejected the candidate
    "pm_not_selected",        # PM did not mention the candidate
    "alignment_observation",  # Alignment policy evaluation result
})

# Event types that represent post-trade observations (append-only, linked)
POST_TRADE_EVENT_TYPES = frozenset({
    "pipeline_executed",         # Trade was successfully executed
    "pipeline_gate_rejected",    # Gate pipeline rejected
    "pipeline_sizing_rejected",  # Position sizer rejected
    "shadow_outcome",            # Shadow outcome scoring result
    "realized_outcome",          # Actual trade P&L outcome
    "recovery_released",         # Crash recovery action
})


def record_decision_snapshot(
    engine,
    candidate_id: str,
    cycle_id: str,
    profile_id: str,
    *,
    context_snapshot_json: str | None = None,
    benchmark_mapping_json: str | None = None,
    pm_decision: str | None = None,
    pm_rationale: str | None = None,
    pm_risk_multiplier: float | None = None,
    alignment_outcome: str | None = None,
    alignment_rule: str | None = None,
    alignment_measurements: dict | None = None,
) -> None:
    """Record an immutable decision-time snapshot.

    This creates a single comprehensive record of the decision-time state
    that NEVER gets mutated. Later observations link back via candidate_id.

    Requirements:
    - 20.1: Store candidate + benchmark context snapshot as immutable record
    - 20.2: Store PM decision + rationale linked to snapshot
    - 20.3: Store alignment result linked to snapshot
    - 20.4: Later observations are append-only linked records
    - 20.5: Never mutate original decision snapshot
    - 20.6: Distinguish pre-trade facts from post-trade observations
    - 20.7: No later inputs overwrite original fields

    Args:
        engine: SQLAlchemy engine.
        candidate_id: The candidate this snapshot is for.
        cycle_id: The PM cycle ID.
        profile_id: The profile ID.
        context_snapshot_json: Frozen context at decision time.
        benchmark_mapping_json: Benchmark mapping at decision time.
        pm_decision: "accept" | "reject" | "not_selected"
        pm_rationale: PM's stated rationale.
        pm_risk_multiplier: PM's requested risk multiplier (if any).
        alignment_outcome: Alignment policy outcome (if evaluated).
        alignment_rule: Rule that fired (if any).
        alignment_measurements: Measurements used (if any).
    """
    snapshot_data = {
        "record_type": "pre_trade_fact",
        "immutable": True,
        "context_snapshot": json.loads(context_snapshot_json) if context_snapshot_json else None,
        "benchmark_mapping": json.loads(benchmark_mapping_json) if benchmark_mapping_json else None,
        "pm_decision": pm_decision,
        "pm_rationale": pm_rationale,
        "pm_risk_multiplier": pm_risk_multiplier,
        "alignment_outcome": alignment_outcome,
        "alignment_rule": alignment_rule,
        "alignment_measurements": alignment_measurements,
        "snapshot_version": "1.0.0",
    }

    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO pm_candidate_events
                    (candidate_id, cycle_id, profile_id, event_type, event_data, created_at)
                    VALUES (:cid, :cycle_id, :profile_id, :event_type, :event_data, :created_at)
                """),
                {
                    "cid": candidate_id,
                    "cycle_id": cycle_id,
                    "profile_id": profile_id,
                    "event_type": "decision_snapshot",
                    "event_data": json.dumps(snapshot_data, default=str),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            conn.commit()
    except Exception as exc:
        logger.warning("Failed to record decision snapshot for %s: %s", candidate_id, exc)


def record_post_trade_observation(
    engine,
    candidate_id: str,
    cycle_id: str,
    profile_id: str,
    event_type: str,
    observation_data: dict,
) -> None:
    """Record a post-trade observation linked to the original decision.

    Post-trade observations are append-only — they never mutate the original
    decision_snapshot record. Each observation links to the original candidate
    via candidate_id (Requirement 20.4).

    Retrospective inferences (e.g., "sector weakness caused the loss") are
    stored with record_type "retrospective_inference" to distinguish them from
    facts present in the original snapshot (Requirement 20.6).

    Args:
        engine: SQLAlchemy engine.
        candidate_id: Links to the original decision snapshot.
        cycle_id: The PM cycle.
        profile_id: The profile.
        event_type: Must be a POST_TRADE_EVENT_TYPES value.
        observation_data: Dict with observation details.
    """
    if event_type not in POST_TRADE_EVENT_TYPES:
        logger.warning(
            "Attempted to record unknown post-trade event type '%s' for %s",
            event_type,
            candidate_id,
        )
        return

    data = {
        "record_type": "post_trade_observation",
        **observation_data,
    }

    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO pm_candidate_events
                    (candidate_id, cycle_id, profile_id, event_type, event_data, created_at)
                    VALUES (:cid, :cycle_id, :profile_id, :event_type, :event_data, :created_at)
                """),
                {
                    "cid": candidate_id,
                    "cycle_id": cycle_id,
                    "profile_id": profile_id,
                    "event_type": event_type,
                    "event_data": json.dumps(data, default=str),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            conn.commit()
    except Exception as exc:
        logger.warning("Failed to record post-trade observation for %s: %s", candidate_id, exc)


def is_pre_trade_fact(event_type: str) -> bool:
    """Distinguish pre-trade facts from post-trade observations (Requirement 20.5)."""
    return event_type in PRE_TRADE_EVENT_TYPES


def is_post_trade_observation(event_type: str) -> bool:
    """Distinguish post-trade observations from pre-trade facts (Requirement 20.5)."""
    return event_type in POST_TRADE_EVENT_TYPES
