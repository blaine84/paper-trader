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
from utils.gate_config import PM_BENCHMARK_CONTEXT_ENABLED, CANDIDATE_EXECUTABLE_SETUP_TYPES

logger = logging.getLogger(__name__)

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
            logger.debug(
                "Excluding candidate %s: setup_type '%s' not in executable set",
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
            )

            # INSERT into registry (fails closed on DB error)
            registry.register(record)

    logger.info(
        "Built candidate set for profile=%s cycle=%s: is_empty=%s",
        profile_id,
        cycle_id,
        registry.is_empty,
    )

    return registry


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
