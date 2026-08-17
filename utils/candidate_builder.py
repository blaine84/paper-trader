"""Candidate Builder — constructs the closed candidate set for a PM cycle.

Filters eligible Analyst signals by profile constraints, generates geometry
scaffolds, and registers CandidateRecords in the pm_candidates table. Returns
a CandidateRegistry instance bound to the cycle (may be empty).

See: design.md §utils/candidate_builder.py
Requirements: 1.1, 1.2, 1.5, 2.1
"""

from __future__ import annotations

import copy
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
    SETUP_WATCH_MODE,
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
SWING_MAPPABLE_SETUP_TYPES = frozenset({
    "sector_rotation",
    "risk_off_macro_short",
    "directional_confusion_breakout",
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


def _process_promoted_watch(
    engine: Any,
    promo: dict,
    registry: CandidateRegistry,
    held_symbols: set[str],
    min_signal_strength: str,
    profile_id: str,
    cycle_id: str,
    cycle_expires_at: datetime | None,
) -> None:
    """Process a single promoted watch through eligibility -> geometry -> register.

    Consumes one promoted watch candidate (from ``get_promotable_candidates()``)
    and either creates a PM candidate (transitioning the watch to ``registered``)
    or expires the watch with a terminal_reason describing why promotion was
    blocked.

    Eligibility checks (short-circuit on first failure, fail-closed):
    1. held_symbols exclusion -> ``promotion_blocked_held_symbol``
    2. min_signal_strength threshold -> ``promotion_blocked_weak_signal``
    3. exception during checks -> ``promotion_blocked_eligibility_error``

    Idempotent consumption:
    - Before ``registry.register()``, check whether a PM candidate already exists
      for ``(source_signal_id=watch_id, profile_id, cycle_id)``. If found, skip
      registration, transition the watch to ``registered``, and log at DEBUG.

    Terminal failure states in the promotion loop:
    - ``promotion_blocked_held_symbol``
    - ``promotion_blocked_weak_signal``
    - ``promotion_blocked_eligibility_error`` (exception during eligibility)
    - ``promotion_blocked_geometry_failed`` (scaffold status != 'ok' or exception)
    - ``promotion_blocked_no_geometry_candidates`` (scaffold ok, empty candidates)
    - ``promotion_blocked_registry_error`` (registry.register() raises)

    All watch transitions here use ``_transition_watch_state()`` with
    ``expected_state='promoted'``. A CAS failure (rowcount == 0) is logged at
    WARNING and processing continues (fail-open) — the watch was already
    consumed/transitioned concurrently or is swept by TTL/stale cleanup.

    See design.md §9 (Promotion Loop with Idempotent Consumption).
    Requirements: 1.2, 1.3, 1.7, 2.1-2.10, 5.5.
    """
    # Local (lazy) import mirrors the fail-open import pattern used by the caller
    # and avoids a hard module-level coupling to watch_candidates.
    from utils.watch_candidates import _transition_watch_state

    watch_id = promo["watch_id"]
    symbol = promo["symbol"]
    signal = promo.get("signal", {}) or {}

    def _expire(reason: str) -> None:
        """Transition promoted -> expired with the given terminal_reason (fail-open CAS)."""
        outcome_json = json.dumps(
            {"terminal_state": "expired", "terminal_reason": reason}
        )
        ok = _transition_watch_state(
            engine,
            watch_id,
            "expired",
            outcome_json,
            expected_state="promoted",
        )
        if not ok:
            logger.warning(
                "CAS promoted->expired (%s) failed for watch %s (already transitioned)",
                reason,
                watch_id,
            )

    # -------------------------------------------------------------------------
    # Eligibility checks (fail-closed): held_symbols before strength, short-circuit
    # -------------------------------------------------------------------------
    try:
        # Check 1 (short-circuit): held-symbol exclusion
        if symbol in held_symbols:
            logger.info(
                "Promotion blocked for watch %s (%s): symbol held",
                watch_id,
                symbol,
            )
            _expire("promotion_blocked_held_symbol")
            return

        # Check 2: signal strength threshold
        signal_strength = signal.get("strength")
        if not _meets_threshold(signal_strength, min_signal_strength):
            logger.info(
                "Promotion blocked for watch %s (%s): weak signal (strength=%s < %s)",
                watch_id,
                symbol,
                signal_strength,
                min_signal_strength,
            )
            _expire("promotion_blocked_weak_signal")
            return
    except Exception as elig_exc:
        # Fail-closed: an eligibility check error must block promotion.
        logger.warning(
            "Eligibility check raised for watch %s (%s): %s — blocking (fail-closed)",
            watch_id,
            symbol,
            elig_exc,
        )
        _expire("promotion_blocked_eligibility_error")
        return

    # -------------------------------------------------------------------------
    # Idempotent dedup: skip register if a PM candidate already exists for
    # (source_signal_id=watch_id, profile_id, cycle_id).
    # -------------------------------------------------------------------------
    try:
        from sqlalchemy import text as sql_text

        with engine.connect() as conn:
            existing = conn.execute(
                sql_text(
                    "SELECT candidate_id FROM pm_candidates "
                    "WHERE source_signal_id = :watch_id "
                    "  AND profile_id = :profile_id "
                    "  AND cycle_id = :cycle_id "
                    "LIMIT 1"
                ),
                {
                    "watch_id": watch_id,
                    "profile_id": profile_id,
                    "cycle_id": cycle_id,
                },
            ).fetchone()
        if existing is not None:
            logger.debug(
                "Idempotent promotion: PM candidate already exists for watch %s "
                "(%s) in cycle %s — skipping register, transitioning to registered",
                watch_id,
                symbol,
                cycle_id,
            )
            ok = _transition_watch_state(
                engine,
                watch_id,
                "registered",
                expected_state="promoted",
            )
            if not ok:
                logger.warning(
                    "CAS promoted->registered failed for watch %s (dedup path)",
                    watch_id,
                )
            return
    except Exception as dedup_exc:
        # Dedup is a safety net; if the lookup itself fails, fall through to the
        # normal register path (registry.register / integrity is authoritative).
        logger.warning(
            "Idempotent dedup query failed for watch %s (%s): %s — proceeding to register",
            watch_id,
            symbol,
            dedup_exc,
        )

    # -------------------------------------------------------------------------
    # Geometry scaffold (fail-closed on failure / empty result)
    # -------------------------------------------------------------------------
    try:
        promo_scaffold = build_entry_geometry_scaffold(signal, profile_id=profile_id)
    except Exception as geo_exc:
        logger.warning(
            "Geometry scaffold raised for watch %s (%s): %s",
            watch_id,
            symbol,
            geo_exc,
        )
        _expire("promotion_blocked_geometry_failed")
        return

    if promo_scaffold.get("status") != "ok":
        logger.warning(
            "Geometry scaffold not ok for watch %s (%s): status=%s reason=%s",
            watch_id,
            symbol,
            promo_scaffold.get("status"),
            promo_scaffold.get("reason", ""),
        )
        _expire("promotion_blocked_geometry_failed")
        return

    promo_candidates = promo_scaffold.get("candidates", [])
    if not promo_candidates:
        logger.warning(
            "Geometry scaffold ok but zero candidates for watch %s (%s)",
            watch_id,
            symbol,
        )
        _expire("promotion_blocked_no_geometry_candidates")
        return

    # -------------------------------------------------------------------------
    # Build the PM candidate record from the first scaffold candidate
    # -------------------------------------------------------------------------
    pc = promo_candidates[0]
    promo_candidate_id = str(uuid.uuid4())
    promo_created_at = datetime.now(timezone.utc)
    promo_expires_at = cycle_expires_at or (promo_created_at + timedelta(hours=1))
    promo_direction = "BUY" if promo_scaffold["direction"] == "LONG" else "SHORT"
    promo_signal_json = json.dumps(signal, default=str, sort_keys=True)
    promo_record_dict = {
        "candidate_id": promo_candidate_id,
        "symbol": symbol,
        "direction": promo_direction,
        "entry_price": pc["entry_price"],
        "stop_price": pc["stop_loss"],
        "target_price": pc["target"],
        "setup_type": signal.get("setup_type", "unknown"),
        "profile_id": profile_id,
        "cycle_id": cycle_id,
    }
    promo_hash = _compute_integrity_hash(promo_record_dict)
    promo_record = CandidateRecord(
        candidate_id=promo_candidate_id,
        cycle_id=cycle_id,
        profile_id=profile_id,
        symbol=symbol,
        direction=promo_direction,
        setup_type=signal.get("setup_type", "unknown"),
        geometry_name=pc["name"],
        entry_price=pc["entry_price"],
        stop_price=pc["stop_loss"],
        target_price=pc["target"],
        risk_reward=pc["risk_reward"],
        trigger=pc["trigger"],
        invalidation_basis=pc["invalidation_basis"],
        target_basis=pc["target_basis"],
        source_signal_id=watch_id,  # traceability
        signal_snapshot_json=promo_signal_json,
        created_at=promo_created_at,
        expires_at=promo_expires_at,
        integrity_hash=promo_hash,
        candidate_type="intraday",
    )

    # -------------------------------------------------------------------------
    # Register (fail-closed on registry error)
    # -------------------------------------------------------------------------
    try:
        registry.register(promo_record)
    except Exception as reg_exc:
        logger.warning(
            "Registry.register failed for promoted watch %s (%s): %s",
            watch_id,
            symbol,
            reg_exc,
        )
        _expire("promotion_blocked_registry_error")
        return

    # -------------------------------------------------------------------------
    # Success: transition promoted -> registered (fail-open on CAS failure).
    # PM candidate is already persisted; a CAS failure only means the watch row
    # was already consumed/transitioned — the idempotent dedup / stale cleanup
    # covers correctness in that case.
    # -------------------------------------------------------------------------
    ok = _transition_watch_state(
        engine,
        watch_id,
        "registered",
        expected_state="promoted",
    )
    if not ok:
        logger.warning(
            "CAS promoted->registered failed for watch %s (PM candidate %s already created)",
            watch_id,
            promo_candidate_id,
        )
    else:
        logger.info(
            "Promoted watch candidate %s for %s -> PM candidate %s (registered)",
            watch_id,
            symbol,
            promo_candidate_id,
        )


def _process_promoted_setup_watch(
    *,
    engine: Any,
    watch: Any,
    registry: CandidateRegistry,
    sw_registry: Any,
    signals: dict[str, dict],
    held_symbols: set[str],
    min_signal_strength: str,
    profile_id: str,
    cycle_id: str,
    cycle_expires_at: datetime | None,
) -> None:
    """Create a PM candidate from a promoted setup watch.

    Geometry is rebuilt from the CURRENT cycle signal — the watch's
    draft_geometry is never used for execution, only carried as prompt context.

    Eligibility checks (fail-closed, short-circuit):
    1. Current-cycle signal required — expire if absent
    2. Held-symbol exclusion — reject
    3. Signal strength threshold (categorical) — reject
    4. CANDIDATE_EXECUTABLE_SETUP_TYPES allowlist — reject (Req 10.6)
    5. Geometry scaffold from current signal — reject if empty/failed
    6. Build signal snapshot with source_type="setup_watch" encoding
    7. Register PM candidate — reject on registry error

    Requirements: 6.3-6.6, 6.10, 7.1-7.10, 10.6, 12.2, 12.3
    """
    from utils.setup_watch_registry import WatchState as _SWState

    watch_id = watch.watch_id
    symbol = watch.symbol

    def _expire_sw(reason: str) -> None:
        """Transition promoted -> expired (fail-open CAS)."""
        try:
            sw_registry.transition_state(
                watch_id, _SWState.PROMOTED, _SWState.EXPIRED,
                terminal_reason=reason,
            )
        except Exception as exc:
            logger.warning(
                "CAS promoted->expired (%s) failed for setup watch %s: %s",
                reason, watch_id, exc,
            )

    def _reject_sw(reason: str) -> None:
        """Transition promoted -> rejected (fail-open CAS)."""
        try:
            sw_registry.transition_state(
                watch_id, _SWState.PROMOTED, _SWState.REJECTED,
                terminal_reason=reason,
            )
        except Exception as exc:
            logger.warning(
                "CAS promoted->rejected (%s) failed for setup watch %s: %s",
                reason, watch_id, exc,
            )

    # -------------------------------------------------------------------------
    # Check 1: Current-cycle signal required (Req 6.4.1)
    # -------------------------------------------------------------------------
    signal = signals.get(symbol)
    if signal is None:
        logger.info(
            "Setup watch promotion blocked for %s (%s): no current signal",
            watch_id, symbol,
        )
        _expire_sw("no_current_signal")
        return

    # -------------------------------------------------------------------------
    # Check 2: Held-symbol exclusion
    # -------------------------------------------------------------------------
    if symbol in held_symbols:
        logger.info(
            "Setup watch promotion blocked for %s (%s): symbol held",
            watch_id, symbol,
        )
        _reject_sw("promotion_blocked_held_symbol")
        return

    # -------------------------------------------------------------------------
    # Check 3: Signal strength threshold (categorical via STRENGTH_ORDER)
    # -------------------------------------------------------------------------
    signal_strength = signal.get("strength", signal.get("signal_strength", ""))
    if not _meets_threshold(signal_strength, min_signal_strength):
        logger.info(
            "Setup watch promotion blocked for %s (%s): weak signal (strength=%s < %s)",
            watch_id, symbol, signal_strength, min_signal_strength,
        )
        _reject_sw("promotion_blocked_weak_signal")
        return

    # -------------------------------------------------------------------------
    # Check 4: CANDIDATE_EXECUTABLE_SETUP_TYPES allowlist (Req 10.6)
    # -------------------------------------------------------------------------
    if watch.setup_type not in CANDIDATE_EXECUTABLE_SETUP_TYPES:
        logger.info(
            "Setup watch promotion blocked for %s (%s): non-executable setup_type=%s",
            watch_id, symbol, watch.setup_type,
        )
        _reject_sw("non_executable_setup_type")
        return

    # -------------------------------------------------------------------------
    # Check 5: Geometry scaffold from CURRENT signal (Req 6.4)
    # -------------------------------------------------------------------------
    try:
        scaffold = build_entry_geometry_scaffold(signal, profile_id=profile_id)
    except Exception as geo_exc:
        logger.warning(
            "Geometry scaffold raised for setup watch %s (%s): %s",
            watch_id, symbol, geo_exc,
        )
        _reject_sw("promotion_blocked_no_geometry")
        return

    if scaffold.get("status") != "ok":
        logger.info(
            "Geometry scaffold not ok for setup watch %s (%s): status=%s",
            watch_id, symbol, scaffold.get("status"),
        )
        _reject_sw("promotion_blocked_no_geometry")
        return

    candidates = scaffold.get("candidates", [])
    if not candidates:
        logger.info(
            "Geometry scaffold ok but zero candidates for setup watch %s (%s)",
            watch_id, symbol,
        )
        _reject_sw("promotion_blocked_no_geometry")
        return

    # -------------------------------------------------------------------------
    # Build signal snapshot with explicit source encoding (Req 6.6)
    # -------------------------------------------------------------------------
    def _maybe_json(raw: str | None) -> Any:
        """Parse JSON string if non-None, else None."""
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    snapshot = {
        "source_type": "setup_watch",
        "watch_id": watch_id,
        "thesis": watch.thesis,
        "maturity_score": watch.maturity_score,
        "observed_cycles": watch.observed_cycles,
        "last_evaluation": _maybe_json(watch.last_evaluation_json),
        "entry_zone": _maybe_json(watch.entry_zone_json),
        "draft_geometry": _maybe_json(watch.draft_geometry_json),  # context only
        "signal": copy.deepcopy(signal),
    }
    signal_snapshot_json = json.dumps(snapshot, default=str, sort_keys=True)

    # -------------------------------------------------------------------------
    # Build the PM candidate record from the first scaffold candidate
    # -------------------------------------------------------------------------
    pc = candidates[0]
    candidate_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)
    expires_at = cycle_expires_at or (created_at + timedelta(hours=1))
    direction = "BUY" if scaffold["direction"] == "LONG" else "SHORT"

    record_dict = {
        "candidate_id": candidate_id,
        "symbol": symbol,
        "direction": direction,
        "entry_price": pc["entry_price"],
        "stop_price": pc["stop_loss"],
        "target_price": pc["target"],
        "setup_type": watch.setup_type,
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
        setup_type=watch.setup_type,
        geometry_name=pc["name"],
        entry_price=pc["entry_price"],
        stop_price=pc["stop_loss"],
        target_price=pc["target"],
        risk_reward=pc["risk_reward"],
        trigger=pc["trigger"],
        invalidation_basis=pc["invalidation_basis"],
        target_basis=pc["target_basis"],
        source_signal_id=watch_id,  # traceability (Req 6.6)
        signal_snapshot_json=signal_snapshot_json,
        created_at=created_at,
        expires_at=expires_at,
        integrity_hash=integrity_hash,
        candidate_type="intraday",
    )

    # -------------------------------------------------------------------------
    # Register (fail-closed on registry error -> reject watch)
    # -------------------------------------------------------------------------
    try:
        registry.register(record)
    except Exception as reg_exc:
        logger.warning(
            "Registry.register failed for setup watch %s (%s): %s",
            watch_id, symbol, reg_exc,
        )
        _expire_sw("promotion_blocked_registry_error")
        return

    logger.info(
        "Promoted setup watch %s for %s -> PM candidate %s (registered)",
        watch_id, symbol, candidate_id,
    )


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
    # Watch Candidate Management (EVALUATION ORDER IS CRITICAL)
    # Guarded by MARKET_STATE_MODE feature flag.
    # Mandated order: expire stale promoted -> evaluate -> consume -> create
    # (see design.md §8). New watches created in Step 3 are NOT evaluated this
    # pass — there is no same-cycle fast-path; they are first evaluated next cycle.
    # ---------------------------------------------------------------------------
    from utils.gate_config import MARKET_STATE_MODE
    if MARKET_STATE_MODE != "disabled":
        try:
            from utils.watch_candidates import (
                evaluate_and_create_watch_candidates,
                evaluate_active_watch_candidates,
                expire_stale_promoted_watches,
                get_promotable_candidates,
                _transition_watch_state,
            )

            # Step 0: Expire stale promoted watches from prior cycles FIRST.
            # Decisively retires promoted rows left over from a crashed prior
            # cycle before any active watch is evaluated. Only runs when a
            # cycle_id is available (the stale check is cycle-scoped).
            if cycle_id is not None:
                expire_stale_promoted_watches(
                    engine=db,
                    profile_id=profile_id,
                    cycle_id=cycle_id,
                )

            # Step 1: Evaluate existing active watches
            evaluate_active_watch_candidates(
                engine=db,
                signals=signals,
                profile_id=profile_id,
                cycle_id=cycle_id,
            )

            # Step 2: Consume promoted watches (enforcing mode only)
            if MARKET_STATE_MODE == "enforcing":
                promotable = get_promotable_candidates(
                    engine=db,
                    signals=signals,
                    profile_id=profile_id,
                    cycle_id=cycle_id,
                )
                for promo in promotable:
                    _process_promoted_watch(
                        db,
                        promo,
                        registry,
                        held_symbols,
                        min_signal_strength,
                        profile_id,
                        cycle_id,
                        cycle_expires_at,
                    )

            # Step 3: Create new watches LAST (NOT evaluated this pass)
            evaluate_and_create_watch_candidates(
                engine=db,
                signals=signals,
                cycle_id=cycle_id,
                profile_id=profile_id,
            )
        except Exception as wc_exc:
            logger.warning("Watch candidate management failed: %s", wc_exc)

    # ---------------------------------------------------------------------------
    # Setup Watch Layer Evaluation (INDEPENDENT from watch_candidates above)
    # Guarded by SETUP_WATCH_MODE feature flag. Uses lazy imports so that when
    # mode is "disabled", setup watch modules never load.
    # Mandated order: expire → invalidate → evaluate → promote → create
    # (see design.md §Setup Watch Layer)
    # ---------------------------------------------------------------------------
    if SETUP_WATCH_MODE != "disabled":
        try:
            from utils.setup_watch_manager import (
                evaluate_cycle as sw_evaluate_cycle,
                get_promotable_watches as sw_get_promotable_watches,
            )
            from utils.setup_watch_registry import (
                SetupWatchRegistry as _SWRegistry,
                WatchState as _SWState,
            )

            # Run full evaluation cycle (expire, invalidate, mature, promote, create)
            sw_eval_result = sw_evaluate_cycle(
                engine=db,
                profile_id=profile_id,
                cycle_id=cycle_id,
                signals=signals,
                portfolio=portfolio,
            )
            logger.info(
                "Setup watch eval: expired=%d invalidated=%d matured=%d "
                "regressed=%d promoted=%d active=%d created=%d",
                sw_eval_result.expired_ttl + sw_eval_result.expired_stale_promoted,
                sw_eval_result.invalidated,
                sw_eval_result.matured,
                sw_eval_result.regressed,
                sw_eval_result.promoted,
                sw_eval_result.still_active,
                sw_eval_result.created,
            )

            # Only enabled mode consumes promoted watches into PM candidates
            if SETUP_WATCH_MODE == "enabled":
                sw_promoted = sw_get_promotable_watches(db, profile_id, cycle_id)

                # Cross-system dedupe (Req 7.9): build set of (symbol, direction)
                # already registered this cycle (from market-state watch promotion
                # or any other source that ran before us).
                claimed_keys: set[tuple[str, str]] = set()
                try:
                    existing_candidates = registry.get_offered_summary()
                    for c in existing_candidates:
                        claimed_keys.add((c["symbol"], c["direction"]))
                except Exception:
                    # If registry lookup fails, proceed without dedup — fail-open
                    pass

                sw_registry = _SWRegistry(db)
                for sw_watch in sw_promoted:
                    try:
                        # Dedupe: skip watches that collide with already-claimed keys
                        if (sw_watch.symbol, sw_watch.side) in claimed_keys:
                            sw_registry.transition_state(
                                sw_watch.watch_id,
                                _SWState.PROMOTED,
                                _SWState.EXPIRED,
                                terminal_reason="superseded_by_market_state_watch",
                            )
                            continue

                        _process_promoted_setup_watch(
                            engine=db,
                            watch=sw_watch,
                            registry=registry,
                            sw_registry=sw_registry,
                            signals=signals,
                            held_symbols=held_symbols,
                            min_signal_strength=min_signal_strength,
                            profile_id=profile_id,
                            cycle_id=cycle_id,
                            cycle_expires_at=cycle_expires_at,
                        )
                        # Add to claimed keys after successful processing
                        claimed_keys.add((sw_watch.symbol, sw_watch.side))
                    except Exception as sw_promo_exc:
                        logger.warning(
                            "Setup watch promotion failed for watch %s (%s): %s",
                            sw_watch.watch_id, sw_watch.symbol, sw_promo_exc,
                        )

        except Exception as sw_exc:
            logger.warning(
                "Setup watch evaluation failed (fail-open): %s", sw_exc
            )

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

        swing_signals = {
            signal_id: signal
            for signal_id, signal in signals.items()
            if signal.get("setup_type", "") in SWING_EXECUTABLE_SETUP_TYPES
            or signal.get("setup_type", "") in SWING_MAPPABLE_SETUP_TYPES
        }
        if not swing_signals:
            logger.debug(
                "No swing-routable signals: profile=%s cycle=%s total_signals=%d",
                profile_id, cycle_id, len(signals),
            )
            return

        swing_candidates = process_swing_signals(
            signals=swing_signals,
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
            original_signal = swing_signals.get(symbol, {})
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
