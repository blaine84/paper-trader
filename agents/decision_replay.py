"""
Decision Replay Agent
Reconstructs historical entry decisions using timestamp-correct inputs, replays them
through a declared policy version via dependency injection, and compares the replay
result with the original decision. Provides evidence-based gate-quality assessment
without mutating production behavior.

Operates in report-only mode at all times (Requirement 9.1). Writes only to the
replay namespace (replay_audit_records, replay_batch_runs, replay_batch_items).

Requirements: 10.3, 10.4, 10.5
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from pytz import timezone as pytz_timezone

from db.schema import get_session
from db.replay_schema import init_replay_db
from utils.position_lifecycle_governance import is_trading_day
from core.replay.policy_version import (
    PolicyVersion,
    build_current_policy_version,
    validate_candidate_policy,
)
from core.replay.gate_adapter import (
    GatePolicyConfig,
    ReplayGateContext,
    build_current_gate_policy_config,
    build_gate_policy_config_from_snapshot,
)
from core.replay.candidate_sourcer import (
    ReplayCandidate,
    load_candidates,
    correlate_and_deduplicate,
)
from core.replay.input_reconstructor import (
    ReplayInputBundle,
    reconstruct_inputs,
)
from core.replay.gate_replayer import replay_gates, GateTrace
from core.replay.delta_classifier import classify_delta, DecisionDelta
from core.replay.scheduler import (
    is_market_hours,
    should_suspend,
    get_default_batch_window,
    should_checkpoint_batch,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Market-hours detection and resource limits (Requirement 10.4)
# ---------------------------------------------------------------------------

# Default maximum candidates per ad-hoc run during market hours
DEFAULT_MARKET_HOURS_MAX_CANDIDATES = 10

# Market hours: 9:30 AM - 4:00 PM ET on trading days
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0


def _is_market_hours(now_et: datetime | None = None) -> bool:
    """Determine if current time is within regular market hours (9:30-16:00 ET on trading days).

    Requirement 10.1: Scheduled replay runs only outside regular market hours.
    Requirement 10.4: Ad-hoc replay during market hours requires operator override.

    Args:
        now_et: Optional datetime in ET timezone for testing. If None, uses current time.

    Returns:
        True if within market hours on a trading day.
    """
    if now_et is None:
        et_tz = pytz_timezone("America/New_York")
        now_et = datetime.now(et_tz)

    if not is_trading_day(now_et):
        return False

    market_open = now_et.replace(
        hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0
    )
    market_close = now_et.replace(
        hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0
    )

    return market_open <= now_et < market_close


# ---------------------------------------------------------------------------
# BatchRunSummary — returned by the agent run()
# ---------------------------------------------------------------------------


@dataclass
class BatchRunSummary:
    """Summary statistics for a completed replay batch run.

    Requirement 10.8: total duration, candidates processed, failed with reason codes,
    coverage breakdown (exact/partial/unscorable), delta classification counts.
    """

    batch_run_id: str
    mode: str
    started_at: datetime
    ended_at: datetime | None = None
    duration_seconds: float = 0.0
    candidates_total: int = 0
    candidates_processed: int = 0
    candidates_failed: int = 0
    exact_count: int = 0
    partial_count: int = 0
    unscorable_count: int = 0
    delta_counts: dict[str, int] = field(default_factory=dict)
    failure_reasons: dict[str, int] = field(default_factory=dict)
    policy_version: PolicyVersion | None = None
    watermark_start: datetime | None = None
    watermark_end: datetime | None = None
    status: str = "completed"


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------


def run(
    engine,
    *,
    mode: str = "batch",
    date_range: tuple[datetime, datetime] | None = None,
    candidate_ids: list[str] | None = None,
    policy_version: str = "current",
    candidate_policy: dict | None = None,
    diagnostic_mode: bool = False,
    filters: dict | None = None,
    max_candidates: int | None = None,
    operator_override: bool = False,
    market_hours_max_candidates: int = DEFAULT_MARKET_HOURS_MAX_CANDIDATES,
) -> BatchRunSummary:
    """Execute a Decision Replay batch or ad-hoc run.

    Orchestrates the full pipeline:
      1. Validate inputs (candidate_policy if provided)
      2. Enforce market-hours constraints for ad-hoc mode
      3. Build PolicyVersion and GatePolicyConfig
      4. Source candidates (CandidateSourcer)
      5. For each candidate:
         a. Reconstruct inputs (InputReconstructor)
         b. Replay gates (GateReplayer)
         c. Classify delta (DeltaClassifier)
         d. Score outcome (Phase 2 — stub)
         e. Persist audit record
      6. Return BatchRunSummary

    Args:
        engine: SQLAlchemy engine for database access.
        mode: "batch" for scheduled runs, "adhoc" for operator-triggered replays.
        date_range: Optional (start, end) datetime tuple for candidate filtering.
        candidate_ids: Optional list of specific candidate IDs to replay.
        policy_version: Policy selection — "current" uses deployed config.
        candidate_policy: Optional dict specifying a report-only policy variant.
        diagnostic_mode: If True, evaluate all gates even after rejection.
        filters: Optional dict of additional filters (profile, symbol, setup_type, etc.).
        max_candidates: Optional cap on number of candidates to process.
        operator_override: If True, allows ad-hoc replay during market hours.
            Required for market-hours ad-hoc replay (Requirement 10.4).
        market_hours_max_candidates: Maximum candidates per ad-hoc run during market
            hours. Default: 10. Only applies when operator_override=True during
            market hours (Requirement 10.4).

    Returns:
        BatchRunSummary with run statistics.

    Raises:
        ValueError: If candidate_policy is provided but invalid, or if ad-hoc
            replay is attempted during market hours without operator_override.
    """
    started_at = datetime.utcnow()
    batch_run_id = str(uuid.uuid4())

    # --- Step 1: Validate inputs ---
    if mode not in ("batch", "adhoc"):
        raise ValueError(f"Unsupported mode: {mode!r}. Must be 'batch' or 'adhoc'.")

    # --- Step 1b: Market-hours enforcement for ad-hoc mode (Requirement 10.4) ---
    if mode == "adhoc" and _is_market_hours():
        if not operator_override:
            raise ValueError(
                "Ad-hoc replay during market hours requires operator_override=True. "
                "Market hours are 9:30 AM - 4:00 PM ET on trading days. "
                "Set operator_override=True to proceed with resource-limited replay."
            )
        # Enforce configurable resource limit during market hours
        # Default: max 10 candidates per ad-hoc run (Requirement 10.4)
        effective_max = market_hours_max_candidates
        if max_candidates is not None:
            effective_max = min(max_candidates, market_hours_max_candidates)
        max_candidates = effective_max
        logger.info(
            "Ad-hoc replay during market hours with operator override. "
            "Resource limit enforced: max %d candidates.",
            max_candidates,
        )

    # --- Step 1c: Market-hours enforcement for batch mode (Requirement 10.1) ---
    if mode == "batch" and is_market_hours():
        logger.warning(
            "Scheduled batch replay blocked: market hours are active "
            "(9:30 AM - 4:00 PM ET). Replay runs only outside market hours."
        )
        ended_at = datetime.utcnow()
        return BatchRunSummary(
            batch_run_id=batch_run_id,
            mode=mode,
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=(ended_at - started_at).total_seconds(),
            status="blocked_market_hours",
        )

    if candidate_policy is not None:
        is_valid, missing_fields = validate_candidate_policy(candidate_policy)
        if not is_valid:
            raise ValueError(
                f"Candidate policy is incomplete. Missing or ambiguous fields: "
                f"{', '.join(missing_fields)}"
            )

    # Ensure replay schema is initialized
    init_replay_db(engine)

    # --- Step 2: Build PolicyVersion and GatePolicyConfig ---
    pv, gate_policy_config = _build_policy(policy_version, candidate_policy)

    # --- Step 2b: Compute default batch window for batch mode (Requirement 10.3) ---
    if mode == "batch" and date_range is None:
        session = get_session(engine)
        try:
            date_range = get_default_batch_window(session, now_utc=started_at)
            logger.info(
                "Default batch window computed: %s to %s",
                date_range[0].isoformat(),
                date_range[1].isoformat(),
            )
        finally:
            session.close()

    # --- Step 3: Source candidates ---
    session = get_session(engine)
    try:
        candidates = load_candidates(
            session,
            date_range=date_range,
            filters=filters,
        )
    finally:
        session.close()

    # Correlate and deduplicate
    candidates = correlate_and_deduplicate(candidates)

    # Filter to specific candidate IDs if provided (adhoc mode)
    if candidate_ids is not None:
        candidates = [
            c for c in candidates
            if c.candidate_id in candidate_ids
        ]

    # Apply max_candidates cap
    if max_candidates is not None and len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]

    # --- Step 4: Process each candidate ---
    summary = BatchRunSummary(
        batch_run_id=batch_run_id,
        mode=mode,
        started_at=started_at,
        candidates_total=len(candidates),
        policy_version=pv,
        watermark_start=date_range[0] if date_range else None,
        watermark_end=date_range[1] if date_range else None,
    )

    for candidate in candidates:
        # --- Requirement 10.2: Check if market hours have started ---
        # If replay is running when market hours begin, checkpoint and suspend.
        if mode == "batch" and should_suspend():
            logger.warning(
                "Market hours started while batch running. "
                "Checkpointing progress and suspending. "
                "Processed %d/%d candidates so far.",
                summary.candidates_processed,
                summary.candidates_total,
            )
            ended_at = datetime.utcnow()
            summary.ended_at = ended_at
            summary.duration_seconds = (ended_at - started_at).total_seconds()
            summary.status = "suspended"
            _persist_batch_run(engine, summary)
            return summary

        try:
            _process_candidate(
                engine=engine,
                candidate=candidate,
                policy_version=pv,
                gate_policy_config=gate_policy_config,
                batch_run_id=batch_run_id,
                diagnostic_mode=diagnostic_mode,
                summary=summary,
            )
        except Exception as exc:
            # Requirement 10.7: record failure, continue to next candidate
            reason_code = _classify_failure(exc)
            summary.candidates_failed += 1
            summary.failure_reasons[reason_code] = (
                summary.failure_reasons.get(reason_code, 0) + 1
            )
            logger.error(
                "Replay failed for candidate %s: [%s] %s",
                candidate.candidate_id,
                reason_code,
                exc,
                exc_info=True,
            )
            # Persist failure audit record
            _persist_failure_record(
                engine=engine,
                candidate=candidate,
                batch_run_id=batch_run_id,
                policy_version=pv,
                reason_code=reason_code,
                error=exc,
            )

    # --- Step 5: Finalize and return summary ---
    ended_at = datetime.utcnow()
    summary.ended_at = ended_at
    summary.duration_seconds = (ended_at - started_at).total_seconds()

    # Persist batch run record
    _persist_batch_run(engine, summary)

    return summary


# ---------------------------------------------------------------------------
# Internal pipeline helpers
# ---------------------------------------------------------------------------


def _build_policy(
    policy_version: str,
    candidate_policy: dict | None,
) -> tuple[PolicyVersion, GatePolicyConfig]:
    """Build PolicyVersion and GatePolicyConfig from the specified policy selection.

    Args:
        policy_version: "current" or a named policy version.
        candidate_policy: Optional candidate policy dict (report-only variant).

    Returns:
        Tuple of (PolicyVersion, GatePolicyConfig).
    """
    if candidate_policy is not None:
        # Build from candidate policy specification
        pv = PolicyVersion(
            name=candidate_policy.get("name", "candidate_policy"),
            gate_revision=candidate_policy.get("gate_revision", "unknown"),
            config_digest="",  # Will be computed from the config
            feature_flags=candidate_policy.get("feature_flags", {}),
            benchmark_version=candidate_policy.get("benchmark_version"),
            config_source_timestamp=datetime.utcnow(),
            gate_ordering_version=candidate_policy.get("gate_ordering_version", "v1.0"),
            adapter_version=candidate_policy.get("adapter_version", "1.0.0"),
        )
        # Build GatePolicyConfig from candidate policy thresholds
        gate_config = build_gate_policy_config_from_snapshot({
            "gate_config": candidate_policy.get("thresholds", candidate_policy),
            "feature_flags": candidate_policy.get("feature_flags", {}),
        })
        # Update config_digest from built config
        pv = PolicyVersion(
            name=pv.name,
            gate_revision=pv.gate_revision,
            config_digest=gate_config.config_digest(),
            feature_flags=pv.feature_flags,
            benchmark_version=pv.benchmark_version,
            config_source_timestamp=pv.config_source_timestamp,
            gate_ordering_version=pv.gate_ordering_version,
            adapter_version=pv.adapter_version,
        )
        return pv, gate_config
    else:
        # Use current deployed policy
        pv = build_current_policy_version()
        gate_config = build_current_gate_policy_config()
        return pv, gate_config


def _process_candidate(
    engine,
    candidate: ReplayCandidate,
    policy_version: PolicyVersion,
    gate_policy_config: GatePolicyConfig,
    batch_run_id: str,
    diagnostic_mode: bool,
    summary: BatchRunSummary,
) -> None:
    """Process a single replay candidate through the full pipeline.

    Steps: reconstruct → replay → classify → (score outcome — Phase 2) → persist.
    Updates the summary counters in-place.
    """
    replay_id = str(uuid.uuid4())

    # Step 4a: Reconstruct inputs
    session = get_session(engine)
    try:
        input_bundle = reconstruct_inputs(session, candidate, policy_version)
    finally:
        session.close()

    # Track coverage classification
    classification = input_bundle.classification
    if classification == "exact":
        summary.exact_count += 1
    elif classification == "partial":
        summary.partial_count += 1
    else:
        summary.unscorable_count += 1

    # Step 4b: Replay gates (only for exact or partial)
    gate_trace: GateTrace | None = None
    if classification in ("exact", "partial"):
        context = _build_gate_context(input_bundle, candidate)
        gate_trace = replay_gates(
            context=context,
            policy_config=gate_policy_config,
            policy_version=policy_version,
            replay_id=replay_id,
            candidate_id=candidate.candidate_id,
            cutoff=input_bundle.cutoff,
            diagnostic_mode=diagnostic_mode,
        )

    # Step 4c: Classify delta
    delta: DecisionDelta | None = None
    if gate_trace is not None:
        original_geometry = {
            "entry_price": candidate.entry_price or Decimal("0"),
            "stop_price": candidate.stop_price or Decimal("0"),
            "target_price": candidate.target_price or Decimal("0"),
        }
        original_size = candidate.quantity or Decimal("0")

        delta = classify_delta(
            original_decision=candidate.original_decision,
            original_gate=candidate.original_gate,
            original_reason_code=candidate.original_reason_code,
            original_geometry=original_geometry,
            original_size=original_size,
            replay_trace=gate_trace,
            replay_classification=classification,
        )

        # Track delta classification counts
        summary.delta_counts[delta.classification] = (
            summary.delta_counts.get(delta.classification, 0) + 1
        )

    # Step 4d: Outcome scoring — Phase 2 stub
    # Counterfactual outcome scoring is deferred to Phase 2.
    # When implemented, this will call score_counterfactual() or
    # score_allowed_to_rejected() depending on the delta classification.

    # Step 4e: Persist audit record
    _persist_audit_record(
        engine=engine,
        replay_id=replay_id,
        batch_run_id=batch_run_id,
        candidate=candidate,
        input_bundle=input_bundle,
        policy_version=policy_version,
        gate_trace=gate_trace,
        delta=delta,
        diagnostic_mode=diagnostic_mode,
    )

    summary.candidates_processed += 1


def _build_gate_context(
    input_bundle: ReplayInputBundle,
    candidate: ReplayCandidate,
) -> ReplayGateContext:
    """Build a ReplayGateContext from the reconstructed input bundle.

    Maps InputSource values from the input bundle to the frozen ReplayGateContext
    fields expected by the gate replayer.
    """
    inputs = input_bundle.inputs

    def _get_value(field_name: str, default=None):
        """Extract value from an InputSource in the bundle."""
        source = inputs.get(field_name)
        if source is not None and source.value is not None:
            return source.value
        return default

    def _get_decimal(field_name: str, default: Decimal = Decimal("0")) -> Decimal:
        """Extract a Decimal value from an InputSource."""
        val = _get_value(field_name)
        if val is None:
            return default
        if isinstance(val, Decimal):
            return val
        try:
            return Decimal(str(val))
        except Exception:
            return default

    def _get_float(field_name: str, default: float | None = None) -> float | None:
        """Extract a float value from an InputSource."""
        val = _get_value(field_name)
        if val is None:
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    # Build open_positions as a tuple of dicts
    open_positions_raw = _get_value("open_positions", [])
    if isinstance(open_positions_raw, (list, tuple)):
        open_positions = tuple(
            p if isinstance(p, dict) else {} for p in open_positions_raw
        )
    else:
        open_positions = ()

    # Build indicators as a tuple if present
    indicators_raw = _get_value("indicators")
    indicators = tuple(indicators_raw) if isinstance(indicators_raw, (list, tuple)) else None

    return ReplayGateContext(
        # Account state
        account_equity=_get_decimal("account_equity"),
        available_cash=_get_decimal("available_cash"),
        open_positions=open_positions,
        # Case library
        case_library_stats=_get_value("case_library_stats", {}),
        similarity_stats=_get_value("similarity_stats"),
        # Signal state
        analyst_signal_payload=_get_value("analyst_signal_payload"),
        signal_strength=_get_float("signal_strength"),
        confidence_value=_get_float("confidence_value"),
        selection_score=_get_float("selection_score"),
        execution_score=_get_float("execution_score"),
        override_confidence_score=_get_float("override_confidence_score"),
        override_reason=_get_value("override_reason"),
        # Market data
        atr_value=_get_float("atr_value"),
        atr_timestamp=_get_value("atr_timestamp"),
        current_price=_get_float("current_price"),
        # Geometry
        entry_price=_get_decimal("entry_price", candidate.entry_price or Decimal("0")),
        stop_price=_get_decimal("stop_price", candidate.stop_price or Decimal("0")),
        target_price=_get_decimal("target_price", candidate.target_price or Decimal("0")),
        quantity=_get_decimal("quantity", candidate.quantity or Decimal("0")),
        max_dollar_risk=_get_float("max_dollar_risk"),
        # Metadata
        symbol=candidate.symbol,
        profile=candidate.profile,
        direction=candidate.direction,
        setup_type=candidate.setup_type,
        catalyst_type=_get_value("catalyst_type"),
        trade_metadata=_get_value("trade_metadata"),
        trade_rationale=_get_value("trade_rationale"),
        atr_source=_get_value("atr_source"),
        # Catalyst gate fields
        rationale=_get_value("rationale"),
        thesis=_get_value("thesis"),
        indicators=indicators,
        quote_timestamp=_get_value("quote_timestamp"),
        strength=_get_value("strength"),
        conviction=_get_value("conviction"),
    )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _persist_audit_record(
    engine,
    replay_id: str,
    batch_run_id: str,
    candidate: ReplayCandidate,
    input_bundle: ReplayInputBundle,
    policy_version: PolicyVersion,
    gate_trace: GateTrace | None,
    delta: DecisionDelta | None,
    diagnostic_mode: bool,
) -> None:
    """Persist a replay audit record to the replay namespace.

    Atomic write to replay_audit_records table. Append-only (Requirement 13.3).
    """
    from sqlalchemy import text

    # Determine era based on whether candidate has a snapshot
    era = "post-snapshot" if input_bundle.snapshot_id else "historical"

    # Serialize components
    source_ids = json.dumps([
        {"source_table": sr.source_table, "source_id": sr.source_id}
        for sr in candidate.source_records
    ])
    input_sources = json.dumps(
        _serialize_input_sources(input_bundle.inputs),
        default=_json_default,
    )
    policy_json = json.dumps(_serialize_policy_version(policy_version), default=_json_default)

    gate_trace_json = None
    if gate_trace is not None:
        gate_trace_json = json.dumps(_serialize_gate_trace(gate_trace), default=_json_default)

    delta_classification = delta.classification if delta else None
    delta_json = None
    if delta is not None:
        delta_json = json.dumps(_serialize_delta(delta), default=_json_default)

    # Get code revision from policy version
    code_revision = policy_version.gate_revision if policy_version else None

    session = get_session(engine)
    try:
        session.execute(
            text("""
                INSERT INTO replay_audit_records (
                    replay_id, batch_run_id, candidate_id,
                    source_candidate_ids_json, snapshot_id, replay_cutoff,
                    input_sources_json, policy_version_json, replay_status,
                    gate_trace_json, decision_delta_classification,
                    decision_delta_json, divergence_cause, divergence_evidence_json,
                    code_revision, era, diagnostic_mode
                ) VALUES (
                    :replay_id, :batch_run_id, :candidate_id,
                    :source_ids, :snapshot_id, :replay_cutoff,
                    :input_sources, :policy_json, :replay_status,
                    :gate_trace_json, :delta_classification,
                    :delta_json, :divergence_cause, :divergence_evidence,
                    :code_revision, :era, :diagnostic_mode
                )
            """),
            {
                "replay_id": replay_id,
                "batch_run_id": batch_run_id,
                "candidate_id": candidate.candidate_id,
                "source_ids": source_ids,
                "snapshot_id": input_bundle.snapshot_id,
                "replay_cutoff": input_bundle.cutoff.isoformat(),
                "input_sources": input_sources,
                "policy_json": policy_json,
                "replay_status": input_bundle.classification,
                "gate_trace_json": gate_trace_json,
                "delta_classification": delta_classification,
                "delta_json": delta_json,
                "divergence_cause": delta.divergence_cause if delta else None,
                "divergence_evidence": (
                    json.dumps(delta.divergence_evidence, default=_json_default)
                    if delta and delta.divergence_evidence
                    else None
                ),
                "code_revision": code_revision,
                "era": era,
                "diagnostic_mode": 1 if diagnostic_mode else 0,
            },
        )
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _persist_failure_record(
    engine,
    candidate: ReplayCandidate,
    batch_run_id: str,
    policy_version: PolicyVersion,
    reason_code: str,
    error: Exception,
) -> None:
    """Persist a failure audit record when candidate processing fails.

    Requirement 10.7: record failure with structured reason code.
    """
    from sqlalchemy import text

    replay_id = str(uuid.uuid4())
    source_ids = json.dumps([
        {"source_table": sr.source_table, "source_id": sr.source_id}
        for sr in candidate.source_records
    ])
    policy_json = json.dumps(_serialize_policy_version(policy_version), default=_json_default)

    session = get_session(engine)
    try:
        session.execute(
            text("""
                INSERT INTO replay_audit_records (
                    replay_id, batch_run_id, candidate_id,
                    source_candidate_ids_json, snapshot_id, replay_cutoff,
                    input_sources_json, policy_version_json, replay_status,
                    code_revision, era, diagnostic_mode,
                    failure_reason_code, failure_details
                ) VALUES (
                    :replay_id, :batch_run_id, :candidate_id,
                    :source_ids, NULL, :replay_cutoff,
                    '{}', :policy_json, 'failed',
                    :code_revision, 'unknown', 0,
                    :reason_code, :failure_details
                )
            """),
            {
                "replay_id": replay_id,
                "batch_run_id": batch_run_id,
                "candidate_id": candidate.candidate_id,
                "source_ids": source_ids,
                "replay_cutoff": (
                    candidate.entry_timestamp.isoformat()
                    if candidate.entry_timestamp
                    else datetime.utcnow().isoformat()
                ),
                "policy_json": policy_json,
                "code_revision": policy_version.gate_revision if policy_version else None,
                "reason_code": reason_code,
                "failure_details": f"{type(error).__name__}: {error}",
            },
        )
        session.commit()
    except Exception:
        session.rollback()
        logger.warning(
            "Failed to persist failure record for candidate %s: %s",
            candidate.candidate_id,
            error,
        )
    finally:
        session.close()


def _persist_batch_run(engine, summary: BatchRunSummary) -> None:
    """Persist the batch run summary to replay_batch_runs.

    Requirement 10.8: record completion statistics.
    """
    from sqlalchemy import text

    policy_json = json.dumps(
        _serialize_policy_version(summary.policy_version),
        default=_json_default,
    ) if summary.policy_version else "{}"

    delta_counts_json = json.dumps(summary.delta_counts)

    session = get_session(engine)
    try:
        session.execute(
            text("""
                INSERT INTO replay_batch_runs (
                    batch_run_id, started_at, ended_at, mode,
                    policy_version_json, candidates_total,
                    candidates_processed, candidates_failed,
                    exact_count, partial_count, unscorable_count,
                    delta_counts_json, duration_seconds, status,
                    watermark_start, watermark_end
                ) VALUES (
                    :batch_run_id, :started_at, :ended_at, :mode,
                    :policy_json, :candidates_total,
                    :candidates_processed, :candidates_failed,
                    :exact_count, :partial_count, :unscorable_count,
                    :delta_counts_json, :duration_seconds, :status,
                    :watermark_start, :watermark_end
                )
            """),
            {
                "batch_run_id": summary.batch_run_id,
                "started_at": summary.started_at.isoformat(),
                "ended_at": summary.ended_at.isoformat() if summary.ended_at else None,
                "mode": summary.mode,
                "policy_json": policy_json,
                "candidates_total": summary.candidates_total,
                "candidates_processed": summary.candidates_processed,
                "candidates_failed": summary.candidates_failed,
                "exact_count": summary.exact_count,
                "partial_count": summary.partial_count,
                "unscorable_count": summary.unscorable_count,
                "delta_counts_json": delta_counts_json,
                "duration_seconds": summary.duration_seconds,
                "status": summary.status,
                "watermark_start": (
                    summary.watermark_start.isoformat() if summary.watermark_start else None
                ),
                "watermark_end": (
                    summary.watermark_end.isoformat() if summary.watermark_end else None
                ),
            },
        )
        session.commit()
    except Exception:
        session.rollback()
        logger.warning("Failed to persist batch run summary: %s", summary.batch_run_id)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_input_sources(inputs: dict) -> dict:
    """Serialize InputSource dict for JSON persistence."""
    result = {}
    for field_name, source in inputs.items():
        result[field_name] = {
            "field_name": source.field_name,
            "value": _safe_serialize(source.value),
            "source_timestamp": (
                source.source_timestamp.isoformat()
                if source.source_timestamp
                else None
            ),
            "source_record_id": source.source_record_id,
            "status": source.status,
        }
    return result


def _serialize_policy_version(pv: PolicyVersion) -> dict:
    """Serialize PolicyVersion to a dict for JSON persistence."""
    return {
        "name": pv.name,
        "gate_revision": pv.gate_revision,
        "config_digest": pv.config_digest,
        "feature_flags": pv.feature_flags,
        "benchmark_version": pv.benchmark_version,
        "config_source_timestamp": (
            pv.config_source_timestamp.isoformat()
            if pv.config_source_timestamp
            else None
        ),
        "gate_ordering_version": pv.gate_ordering_version,
        "adapter_version": pv.adapter_version,
    }


def _serialize_gate_trace(trace: GateTrace) -> dict:
    """Serialize GateTrace for JSON persistence."""
    return {
        "entries": [
            {
                "gate_name": e.gate_name,
                "decision": e.decision,
                "reason_code": e.reason_code,
                "threshold_applied": e.threshold_applied,
                "input_fields": _safe_serialize(e.input_fields),
                "adjusted_values": _safe_serialize(e.adjusted_values),
                "cumulative_size_multiplier": e.cumulative_size_multiplier,
                "is_non_executable": e.is_non_executable,
                "missing_fields": e.missing_fields,
            }
            for e in trace.entries
        ],
        "final_decision": trace.final_decision,
        "final_gate": trace.final_gate,
        "final_reason_code": trace.final_reason_code,
        "diagnostic_mode": trace.diagnostic_mode,
    }


def _serialize_delta(delta: DecisionDelta) -> dict:
    """Serialize DecisionDelta for JSON persistence."""
    return {
        "classification": delta.classification,
        "first_diverging_gate": delta.first_diverging_gate,
        "divergence_cause": delta.divergence_cause,
        "divergence_evidence": _safe_serialize(delta.divergence_evidence),
        "original_decision": delta.original_decision,
        "replay_decision": delta.replay_decision,
        "geometry_differs": delta.geometry_differs,
        "size_differs": delta.size_differs,
    }


def _safe_serialize(obj: Any) -> Any:
    """Convert non-JSON-serializable types for safe serialization."""
    if obj is None:
        return None
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(item) for item in obj]
    if isinstance(obj, (frozenset, set)):
        return sorted(_safe_serialize(item) for item in obj)
    return obj


def _json_default(obj: Any) -> Any:
    """JSON serializer for types not natively supported."""
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (frozenset, set)):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


def _classify_failure(exc: Exception) -> str:
    """Classify an exception into a structured reason code.

    Requirement 13.5: reason codes include missing_input, policy_not_found,
    snapshot_corrupt, gate_execution_error, timeout.

    Reason code categories:
        - missing_input: Required input data is unavailable or cannot be reconstructed
        - policy_not_found: The requested policy version cannot be located or reconstructed
        - snapshot_corrupt: The decision snapshot is malformed, incomplete, or fails integrity checks
        - gate_execution_error: A gate raised an unhandled exception during evaluation
        - timeout: The operation exceeded its time budget
    """
    exc_type = type(exc).__name__
    exc_msg = str(exc).lower()

    # Timeout detection (both exception type and message-based)
    if exc_type in ("TimeoutError", "asyncio.TimeoutError"):
        return "timeout"
    if "timeout" in exc_msg or "timed out" in exc_msg or "time limit" in exc_msg:
        return "timeout"

    # Snapshot corruption
    if "snapshot" in exc_msg and ("corrupt" in exc_msg or "invalid" in exc_msg or "malformed" in exc_msg):
        return "snapshot_corrupt"
    if "corrupt" in exc_msg or "integrity" in exc_msg:
        return "snapshot_corrupt"

    # Missing input
    if "missing" in exc_msg or "unavailable" in exc_msg or "not found" in exc_msg:
        # Distinguish between missing input vs policy not found
        if "policy" in exc_msg:
            return "policy_not_found"
        return "missing_input"

    # Policy not found
    if "policy" in exc_msg:
        return "policy_not_found"

    # Default: gate execution error
    return "gate_execution_error"
