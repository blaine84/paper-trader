"""Candidate Builder — constructs the closed candidate set for a PM cycle.

Filters eligible Analyst signals by profile constraints, generates geometry
scaffolds, and registers CandidateRecords in the pm_candidates table. Returns
a CandidateRegistry instance bound to the cycle (may be empty).

See: design.md §utils/candidate_builder.py
Requirements: 1.1, 1.2, 1.5, 2.1
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from utils.candidate_registry import (
    CandidateRecord,
    CandidateRegistry,
    _compute_integrity_hash,
)
from utils.entry_geometry import build_entry_geometry_scaffold
from utils.gate_config import (
    PM_BENCHMARK_CONTEXT_ENABLED,
    CANDIDATE_EXECUTABLE_SETUP_TYPES,
    SWING_EXECUTABLE_SETUP_TYPES,
    SWING_MAX_CANDIDATE_AGE_HOURS,
)

logger = logging.getLogger(__name__)

# Reason codes for JSON explanation when no swing candidates are built
SWING_NO_CANDIDATES_REASONS = frozenset({
    "no_fresh_signals",
    "no_executable_mapping",
    "missing_geometry",
    "failed_risk_gates",
    "stale_data",
    "same_symbol_exposure",
    "profile_policy",
})

# Signal strength ordering — mirrors portfolio_manager.STRENGTH_ORDER
STRENGTH_ORDER: dict[str, int] = {"weak": 1, "moderate": 2, "strong": 3}


def _meets_threshold(signal_strength: str, threshold: str) -> bool:
    """Return True if signal_strength meets or exceeds the threshold.

    Replicates the logic from agents.portfolio_manager._meets_threshold().
    """
    sig_val = STRENGTH_ORDER.get(str(signal_strength).lower(), 0)
    thr_val = STRENGTH_ORDER.get(str(threshold).lower(), 0)
    return sig_val >= thr_val


def build_candidate_set(
    db: Any,
    signals: dict[str, dict],
    profile_id: str,
    profile: dict,
    portfolio: dict,
    cycle_id: str,
    *,
    cycle_expires_at: datetime | None = None,
) -> CandidateRegistry:
    """Build the closed candidate set for a PM cycle.

    Steps:
    1. Filter signals by profile eligibility (strength threshold, direction,
       held symbols).
    2. For each eligible signal, call build_entry_geometry_scaffold().
    3. For each scaffold candidate, create a CandidateRecord with full UUID4,
       deep-copied signal snapshot, and integrity hash.
    4. INSERT all candidates into pm_candidates (fails closed on DB error).
    5. Return a CandidateRegistry instance bound to this cycle.

    Returns registry (may be empty — that's valid per Requirement 1.5).
    """
    registry = CandidateRegistry(db, cycle_id, profile_id)

    # P1: Context snapshot builder (when benchmark context flag enabled)
    context_builder = None
    if PM_BENCHMARK_CONTEXT_ENABLED:
        from utils.benchmark_mapping import get_benchmark_mapping, DEFAULT_FRESHNESS_CONFIG
        from utils.context_snapshot import build_context_snapshot
        try:
            from utils.finnhub_client import FinnhubClient
            context_builder = FinnhubClient()
        except Exception as exc:
            logger.warning("Failed to create market data provider for context snapshots: %s", exc)

    # Derive held symbols from portfolio positions
    held_symbols = _get_held_symbols(portfolio)

    # Profile minimum signal strength
    min_signal_strength = profile.get("min_signal_strength", "moderate")

    # ---------------------------------------------------------------------------
    # Watch Candidate Management (guarded by MARKET_STATE_MODE feature flag)
    # ---------------------------------------------------------------------------
    from utils.gate_config import MARKET_STATE_MODE
    if MARKET_STATE_MODE != "disabled":
        try:
            from utils.watch_candidates import (
                evaluate_and_create_watch_candidates,
                evaluate_active_watch_candidates,
                get_promotable_candidates,
            )
            # 13.1: Create watch candidates for eligible symbols
            evaluate_and_create_watch_candidates(
                engine=db,
                signals=signals,
                cycle_id=cycle_id,
                profile_id=profile_id,
            )
            # 13.2: Evaluate active watch candidates
            evaluate_active_watch_candidates(
                engine=db,
                signals=signals,
                profile_id=profile_id,
            )
            # 13.3: Promotion (enforcing mode only)
            if MARKET_STATE_MODE == "enforcing":
                promotable = get_promotable_candidates(
                    engine=db,
                    signals=signals,
                    profile_id=profile_id,
                )
                for promo in promotable:
                    try:
                        promo_signal = promo["signal"]
                        promo_scaffold = build_entry_geometry_scaffold(promo_signal, profile_id=profile_id)
                        if promo_scaffold.get("status") != "ok":
                            continue
                        promo_candidates = promo_scaffold.get("candidates", [])
                        if not promo_candidates:
                            continue
                        # Use first scaffold candidate for promoted watch
                        pc = promo_candidates[0]
                        promo_candidate_id = str(uuid.uuid4())
                        promo_created_at = datetime.now(timezone.utc)
                        promo_expires_at = cycle_expires_at or (promo_created_at + timedelta(hours=1))
                        promo_direction = "BUY" if promo_scaffold["direction"] == "LONG" else "SHORT"
                        promo_signal_json = json.dumps(promo_signal, default=str, sort_keys=True)
                        promo_record_dict = {
                            "candidate_id": promo_candidate_id,
                            "symbol": promo["symbol"],
                            "direction": promo_direction,
                            "entry_price": pc["entry_price"],
                            "stop_price": pc["stop_loss"],
                            "target_price": pc["target"],
                            "setup_type": promo_signal.get("setup_type", "unknown"),
                            "profile_id": profile_id,
                            "cycle_id": cycle_id,
                        }
                        promo_hash = _compute_integrity_hash(promo_record_dict)
                        promo_record = CandidateRecord(
                            candidate_id=promo_candidate_id,
                            cycle_id=cycle_id,
                            profile_id=profile_id,
                            symbol=promo["symbol"],
                            direction=promo_direction,
                            setup_type=promo_signal.get("setup_type", "unknown"),
                            geometry_name=pc["name"],
                            entry_price=pc["entry_price"],
                            stop_price=pc["stop_loss"],
                            target_price=pc["target"],
                            risk_reward=pc["risk_reward"],
                            trigger=pc["trigger"],
                            invalidation_basis=pc["invalidation_basis"],
                            target_basis=pc["target_basis"],
                            source_signal_id=promo["watch_id"],  # traceability
                            signal_snapshot_json=promo_signal_json,
                            created_at=promo_created_at,
                            expires_at=promo_expires_at,
                            integrity_hash=promo_hash,
                            candidate_type="intraday",
                        )
                        registry.register(promo_record)
                        logger.info(
                            "Promoted watch candidate %s for %s → PM candidate %s",
                            promo["watch_id"], promo["symbol"], promo_candidate_id,
                        )
                    except Exception as promo_exc:
                        logger.warning(
                            "Watch candidate promotion failed for %s: %s",
                            promo.get("symbol"), promo_exc,
                        )
        except Exception as wc_exc:
            logger.warning("Watch candidate management failed: %s", wc_exc)

    # Filter eligible signals
    eligible_signals = _filter_eligible_signals(
        signals, held_symbols, min_signal_strength
    )

    if not eligible_signals:
        logger.info(
            "No eligible signals for profile=%s cycle=%s (total=%d, held=%d)",
            profile_id,
            cycle_id,
            len(signals),
            len(held_symbols),
        )
        return registry

    # Process each eligible signal through geometry scaffold
    now = datetime.now(timezone.utc)
    default_expires_at = cycle_expires_at or (now + timedelta(hours=1))

    for symbol, signal in eligible_signals.items():
        scaffold = build_entry_geometry_scaffold(signal, profile_id=profile_id)

        # Only process scaffolds with status == "ok" and non-empty candidates
        if scaffold.get("status") != "ok":
            logger.debug(
                "Scaffold status=%s for symbol=%s reason=%s",
                scaffold.get("status"),
                symbol,
                scaffold.get("reason", ""),
            )
            continue

        candidates = scaffold.get("candidates", [])
        if not candidates:
            logger.debug(
                "Scaffold ok but no candidates for symbol=%s", symbol
            )
            continue

        # Filter by executable setup type (only types in the closed set are eligible)
        setup_type = signal.get("setup_type", "unknown")
        if setup_type not in CANDIDATE_EXECUTABLE_SETUP_TYPES:
            if setup_type not in SWING_EXECUTABLE_SETUP_TYPES:
                logger.debug(
                    "Non-executable setup type: symbol=%s raw_label=%s reason=non_executable_type",
                    symbol, setup_type,
                )
            else:
                logger.debug(
                    "Excluding intraday candidate %s: setup_type '%s' is swing-only",
                    symbol, setup_type,
                )
            continue

        # Deep-copy signal to canonical JSON string (once per signal)
        signal_snapshot_json = json.dumps(signal, default=str, sort_keys=True)

        # Create a CandidateRecord for each scaffold candidate
        for candidate in candidates:
            candidate_id = str(uuid.uuid4())
            created_at = datetime.now(timezone.utc)
            expires_at = cycle_expires_at or (created_at + timedelta(hours=1))

            # Map direction: scaffold uses LONG/SHORT, registry uses BUY/SHORT
            direction = (
                "BUY" if scaffold["direction"] == "LONG" else "SHORT"
            )

            # Derive source signal ID
            source_signal_id = (
                signal.get("signal_id")
                or signal.get("id")
                or f"{symbol}_{cycle_id}"
            )

            # P1: Attach context snapshot if enabled
            context_snapshot_json = None
            benchmark_mapping_json = None
            if PM_BENCHMARK_CONTEXT_ENABLED and context_builder:
                mapping = get_benchmark_mapping(scaffold["symbol"])
                if mapping:
                    snapshot = build_context_snapshot(
                        scaffold["symbol"], mapping, context_builder, DEFAULT_FRESHNESS_CONFIG
                    )
                    if snapshot:
                        context_snapshot_json = snapshot.to_json()
                        benchmark_mapping_json = json.dumps(mapping, sort_keys=True)

            # Build record dict for integrity hash computation
            record_dict = {
                "candidate_id": candidate_id,
                "symbol": scaffold["symbol"],
                "direction": direction,
                "entry_price": candidate["entry_price"],
                "stop_price": candidate["stop_loss"],
                "target_price": candidate["target"],
                "setup_type": signal.get("setup_type", "unknown"),
                "profile_id": profile_id,
                "cycle_id": cycle_id,
            }

            integrity_hash = _compute_integrity_hash(record_dict)

            record = CandidateRecord(
                candidate_id=candidate_id,
                cycle_id=cycle_id,
                profile_id=profile_id,
                symbol=scaffold["symbol"],
                direction=direction,
                setup_type=signal.get("setup_type", "unknown"),
                geometry_name=candidate["name"],
                entry_price=candidate["entry_price"],
                stop_price=candidate["stop_loss"],
                target_price=candidate["target"],
                risk_reward=candidate["risk_reward"],
                trigger=candidate["trigger"],
                invalidation_basis=candidate["invalidation_basis"],
                target_basis=candidate["target_basis"],
                source_signal_id=source_signal_id,
                signal_snapshot_json=signal_snapshot_json,
                created_at=created_at,
                expires_at=expires_at,
                integrity_hash=integrity_hash,
                context_snapshot_json=context_snapshot_json,
                benchmark_mapping_json=benchmark_mapping_json,
                candidate_type="intraday",
            )

            # INSERT into registry (fails closed on DB error)
            registry.register(record)

    logger.info(
        "Built candidate set for profile=%s cycle=%s: is_empty=%s",
        profile_id,
        cycle_id,
        registry.is_empty,
    )

    # ---------------------------------------------------------------------------
    # Swing candidate integration (guarded by SWING_CANDIDATE_MODE feature flag)
    # ---------------------------------------------------------------------------
    _build_swing_candidates(
        db=db,
        signals=signals,
        profile_id=profile_id,
        profile=profile,
        portfolio=portfolio,
        cycle_id=cycle_id,
        registry=registry,
    )

    return registry


def _build_swing_candidates(
    db: Any,
    signals: dict[str, dict],
    profile_id: str,
    profile: dict,
    portfolio: dict,
    cycle_id: str,
    registry: CandidateRegistry,
) -> None:
    """Process swing signals and register swing candidates when mode != 'disabled'.

    Lazy-imports process_swing_signals to avoid circular imports.
    Fail-open: exceptions are caught and logged, never block the pipeline.

    When no swing candidates are built, records a JSON explanation in PM notes.
    """
    from utils.gate_config import get_swing_candidate_mode

    mode = get_swing_candidate_mode()
    if mode == "disabled":
        return

    try:
        from utils.swing_candidate_bridge import process_swing_signals

        swing_candidates = process_swing_signals(
            signals=signals,
            profile_id=profile_id,
            profile=profile,
            portfolio=portfolio,
            cycle_id=cycle_id,
            db=db,
            engine=db,
        )

        if not swing_candidates:
            # swing_evaluation_summary (persisted inside process_swing_signals)
            # supersedes the old swing_no_candidates event — no duplicate recording.
            return

        # Register each swing candidate returned by the bridge
        now = datetime.now(timezone.utc)
        for sc in swing_candidates:
            candidate_id = str(uuid.uuid4())
            created_at = datetime.now(timezone.utc)
            expires_at = created_at + timedelta(hours=SWING_MAX_CANDIDATE_AGE_HOURS)

            # Build signal snapshot
            symbol = sc["symbol"]
            original_signal = signals.get(symbol, {})
            signal_snapshot_json = json.dumps(original_signal, default=str, sort_keys=True)

            # Map direction
            direction = "BUY" if sc["direction"] == "LONG" else "SHORT"

            # Source signal ID
            source_signal_id = sc.get("signal_id") or f"{symbol}_{cycle_id}"

            # Geometry from bridge result
            geometry = sc["geometry"]

            # Build record dict for integrity hash
            record_dict = {
                "candidate_id": candidate_id,
                "symbol": symbol,
                "direction": direction,
                "entry_price": float(geometry.entry_price),
                "stop_price": float(geometry.stop_price),
                "target_price": float(geometry.target_price),
                "setup_type": sc["normalized_setup_type"],
                "profile_id": profile_id,
                "cycle_id": cycle_id,
            }

            integrity_hash = _compute_integrity_hash(record_dict)

            record = CandidateRecord(
                candidate_id=candidate_id,
                cycle_id=cycle_id,
                profile_id=profile_id,
                symbol=symbol,
                direction=direction,
                setup_type=sc["normalized_setup_type"],
                geometry_name=f"swing_{sc['normalized_setup_type']}",
                entry_price=float(geometry.entry_price),
                stop_price=float(geometry.stop_price),
                target_price=float(geometry.target_price),
                risk_reward=float(geometry.risk_reward),
                trigger=f"Swing entry: {sc['normalized_setup_type']}",
                invalidation_basis=geometry.invalidation_basis,
                target_basis=f"Swing target for {sc['normalized_setup_type']}",
                source_signal_id=source_signal_id,
                signal_snapshot_json=signal_snapshot_json,
                created_at=created_at,
                expires_at=expires_at,
                integrity_hash=integrity_hash,
                candidate_type="swing",
            )

            # INSERT into registry (fails closed on DB error)
            registry.register(record)

        logger.info(
            "Swing candidates registered: profile=%s cycle=%s count=%d",
            profile_id, cycle_id, len(swing_candidates),
        )

    except Exception as exc:
        # Fail-open: swing candidate processing errors never block intraday pipeline
        logger.warning(
            "Swing candidate processing failed (fail-open): profile=%s cycle=%s error=%s",
            profile_id, cycle_id, exc,
        )


def _record_no_swing_explanation(
    db: Any,
    cycle_id: str,
    profile_id: str,
    signals: dict[str, dict],
) -> None:
    """Record JSON explanation when no swing candidates are built for a cycle.

    Determines the most relevant reason from the available signal data and
    persists it as a pm_candidate_events row with event_type 'swing_no_candidates'.
    Fail-open: exceptions are caught and logged.
    """
    # Determine reason based on available signals
    if not signals:
        reason = "no_fresh_signals"
    else:
        # Check if any signals have swing-eligible setup types
        has_swing_eligible = any(
            sig.get("setup_type", "") in SWING_EXECUTABLE_SETUP_TYPES
            or sig.get("setup_type", "") in (
                "sector_rotation", "risk_off_macro_short",
                "directional_confusion_breakout",
            )
            for sig in signals.values()
        )
        if not has_swing_eligible:
            reason = "no_executable_mapping"
        else:
            reason = "failed_risk_gates"

    explanation = json.dumps({"reason": reason}, sort_keys=True)

    try:
        from sqlalchemy import text as sql_text
        now = datetime.now(timezone.utc).isoformat()
        with db.connect() as conn:
            conn.execute(
                sql_text("""
                    INSERT INTO pm_candidate_events
                    (candidate_id, cycle_id, profile_id, event_type, event_data, created_at, candidate_type)
                    VALUES (:cid, :cycle_id, :profile_id, :event_type, :event_data, :created_at, :candidate_type)
                """),
                {
                    "cid": "",
                    "cycle_id": cycle_id,
                    "profile_id": profile_id,
                    "event_type": "swing_no_candidates",
                    "event_data": explanation,
                    "created_at": now,
                    "candidate_type": "swing",
                },
            )
            conn.commit()
    except Exception as exc:
        logger.warning(
            "Failed to record swing no-candidates explanation (fail-open): %s", exc
        )

    logger.debug(
        "No swing candidates built: profile=%s cycle=%s reason=%s",
        profile_id, cycle_id, reason,
    )


def _get_held_symbols(portfolio: dict) -> set[str]:
    """Derive set of symbols with active positions from portfolio dict."""
    positions = portfolio.get("positions", {})
    # Positions may be a dict keyed by symbol, or a list of position dicts
    if isinstance(positions, dict):
        return set(positions.keys())
    if isinstance(positions, list):
        return {p.get("symbol", "") for p in positions if p.get("symbol")}
    return set()


def _filter_eligible_signals(
    signals: dict[str, dict],
    held_symbols: set[str],
    min_signal_strength: str,
) -> dict[str, dict]:
    """Filter signals to only eligible entry candidates.

    Excludes:
    - Symbols with active positions (held_symbols)
    - Signals with direction == "HOLD"
    - Signals below profile's min_signal_strength threshold
    """
    eligible = {}
    for sym, sig in signals.items():
        # Skip symbols with active positions
        if sym in held_symbols:
            continue

        # Skip HOLD signals
        direction = sig.get("signal", "").upper()
        if direction == "HOLD":
            continue

        # Skip signals below strength threshold
        strength = sig.get("strength", "weak")
        if not _meets_threshold(strength, min_signal_strength):
            continue

        eligible[sym] = sig

    return eligible
