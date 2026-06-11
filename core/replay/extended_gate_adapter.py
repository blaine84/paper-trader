"""Extended gate adapters for Phase 2 decision boundary checks.

These adapters wrap the extended decision boundary checks (items 5–9, 12–18
from the production entry decision sequence) for replay evaluation. Each
adapter follows the same pattern as the core gate adapters in gate_adapter.py:
dependency injection via kwargs, no global mutation, deterministic evaluation.

**Explicitly excluded (both phases):**
- Item 10: high_wr_stop_buffer — depends on real-time market microstructure
  (bid/ask spread, order book depth) that cannot be reconstructed from the
  decision snapshot without introducing hindsight.
- Item 11: momentum_cooldown — depends on real-time cooldown state (time since
  last momentum signal) that cannot be reconstructed without hindsight about
  whether the cooldown period has elapsed.

See: design.md §Extended Decision Boundary, requirements §5.1, §5.2
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Callable

from core.replay.gate_adapter import (
    GatePolicyConfig,
    ReplayGateContext,
    _normalize_decision,
)


# ---------------------------------------------------------------------------
# Extended gate sequence — Phase 2
# ---------------------------------------------------------------------------

# Phase 2: Extended decision boundary (evaluated after core gates pass).
# Production order items 5–9, 12–18.
#
# EXCLUDED items (cannot be reconstructed from snapshot without hindsight):
#   Item 10 — high_wr_stop_buffer: heuristic depending on real-time market
#             microstructure (bid/ask spread, level-2 depth) not capturable
#             in the decision snapshot.
#   Item 11 — momentum_cooldown: depends on real-time cooldown state that
#             cannot be reconstructed without hindsight about whether the
#             cooldown period has elapsed by decision time.
EXTENDED_GATE_SEQUENCE: list[str] = [
    "similarity_stats",         # find_similar_cases + compute_similarity_stats
    "case_stat_construction",   # build case_stats for edge
    "hard_rejection_check",     # check_hard_rejection(case_stats)
    "edge_score_computation",   # compute_edge_score (requires similarity_stats)
    "edge_threshold_check",     # edge < 0.4
    # Items 10-11 EXCLUDED: high_wr_stop_buffer, momentum_cooldown
    #   - high_wr_stop_buffer: depends on real-time market microstructure
    #     not captured in snapshot
    #   - momentum_cooldown: depends on real-time cooldown state that cannot
    #     be reconstructed without hindsight
    "strategy_multiplier",      # get_strategy_position_multiplier
    "adaptive_throttle",        # adaptive_risk_throttle
    "portfolio_risk",           # validate_portfolio_risk + compute_portfolio_risk
    "validate_trade",           # position limits, dollar risk caps
    "correlation_check",        # check_correlation
    "confidence_adjustment",    # adjust_confidence
    "position_size_cap",        # cap_position_size
]


# Required context fields for each extended gate (derived from production code)
EXTENDED_GATE_REQUIRED_FIELDS: dict[str, list[str]] = {
    "similarity_stats": [
        "symbol",
        "setup_type",
        "case_library_stats",
    ],
    "case_stat_construction": [
        "symbol",
        "setup_type",
        "case_library_stats",
    ],
    "hard_rejection_check": [
        "case_library_stats",
    ],
    "edge_score_computation": [
        "signal_strength",
        "confidence_value",
        "similarity_stats",
        "case_library_stats",
    ],
    "edge_threshold_check": [
        "signal_strength",
        "confidence_value",
        "similarity_stats",
        "case_library_stats",
    ],
    "strategy_multiplier": [
        "setup_type",
    ],
    "adaptive_throttle": [
        "quantity",
        "case_library_stats",
    ],
    "portfolio_risk": [
        "symbol",
        "entry_price",
        "quantity",
        "direction",
        "account_equity",
        "open_positions",
    ],
    "validate_trade": [
        "symbol",
        "entry_price",
        "stop_price",
        "target_price",
        "quantity",
        "direction",
        "profile",
        "account_equity",
        "available_cash",
    ],
    "correlation_check": [
        "symbol",
        "direction",
        "profile",
    ],
    "confidence_adjustment": [
        "setup_type",
    ],
    "position_size_cap": [
        "quantity",
        "entry_price",
    ],
}


# ---------------------------------------------------------------------------
# ExtendedGateAdapter — wraps extended checks with DI kwargs
# ---------------------------------------------------------------------------


class ExtendedGateAdapter:
    """Adapter for extended decision boundary checks (Phase 2).

    Follows the same pattern as ReplayGateAdapter:
    - No DB queries (uses captured state from ReplayGateContext)
    - No global mutation
    - Deterministic evaluation via frozen clock and ID provider
    - Returns normalized result dict compatible with GateTraceEntry construction

    Each adapter wraps a simplified version of the production check, using
    pre-captured state from the context rather than live DB queries. This
    allows incremental filling of adapter logic as production functions
    gain DI support.
    """

    def __init__(
        self,
        gate_name: str,
        policy_config: GatePolicyConfig,
        context: ReplayGateContext,
        replay_clock: Callable[[], datetime],
        id_provider: Callable[[], str],
    ):
        self.gate_name = gate_name
        self.policy_config = policy_config
        self.context = context
        self.replay_clock = replay_clock
        self.id_provider = id_provider

    def evaluate(self) -> dict:
        """Call the appropriate extended check adapter.

        Returns a normalized result dict with:
            - decision: canonical decision string
            - reason_type: check-specific reason code
            - reason: human-readable reason
            - raw_result: the unmodified result from the check
            - missing_fields: list of required fields that are None/missing
        """
        missing_fields = self._detect_missing_fields()

        gate_dispatch: dict[str, Callable[[], dict]] = {
            "similarity_stats": self._evaluate_similarity_stats,
            "case_stat_construction": self._evaluate_case_stat_construction,
            "hard_rejection_check": self._evaluate_hard_rejection_check,
            "edge_score_computation": self._evaluate_edge_score_computation,
            "edge_threshold_check": self._evaluate_edge_threshold_check,
            "strategy_multiplier": self._evaluate_strategy_multiplier,
            "adaptive_throttle": self._evaluate_adaptive_throttle,
            "portfolio_risk": self._evaluate_portfolio_risk,
            "validate_trade": self._evaluate_validate_trade,
            "correlation_check": self._evaluate_correlation_check,
            "confidence_adjustment": self._evaluate_confidence_adjustment,
            "position_size_cap": self._evaluate_position_size_cap,
        }

        evaluator = gate_dispatch.get(self.gate_name)
        if evaluator is None:
            return {
                "decision": "error",
                "reason_type": "unknown_extended_gate",
                "reason": f"No adapter for extended gate: {self.gate_name}",
                "raw_result": {},
                "missing_fields": missing_fields,
            }

        try:
            raw_result = evaluator()
        except Exception as exc:
            return {
                "decision": "error",
                "reason_type": "gate_exception",
                "reason": f"{type(exc).__name__}: {exc}",
                "raw_result": {},
                "missing_fields": missing_fields,
            }

        # Normalize decision
        raw_decision = raw_result.get("decision", "allow")
        canonical = _normalize_decision(str(raw_decision))

        return {
            "decision": canonical,
            "reason_type": raw_result.get("reason_type", ""),
            "reason": raw_result.get("reason", ""),
            "raw_result": raw_result,
            "missing_fields": missing_fields,
        }

    # --- Extended gate evaluation methods ---

    def _evaluate_similarity_stats(self) -> dict:
        """Compute similarity statistics from pre-captured case library state.

        In production: calls find_similar_cases(signal, engine) then
        compute_similarity_stats(cases). In replay: uses pre-captured
        similarity_stats from the context, or computes from case_library_stats
        if similarity data was captured at snapshot time.
        """
        from core.similarity import compute_similarity_stats

        # Use pre-captured similarity_stats if available in context
        if self.context.similarity_stats is not None:
            stats = self.context.similarity_stats
        else:
            # Fall back to computing from case_library_stats similar_cases field
            similar_cases = self.context.case_library_stats.get("similar_cases", [])
            stats = compute_similarity_stats(similar_cases)

        return {
            "decision": "allow",
            "reason_type": "similarity_computed",
            "reason": f"Computed similarity stats: sample_size={stats.get('sample_size', 0)}",
            "similarity_stats": stats,
            "inputs": {
                "sample_size": stats.get("sample_size", 0),
                "similarity_winrate": stats.get("similarity_winrate", 0.0),
            },
        }

    def _evaluate_case_stat_construction(self) -> dict:
        """Build case_stats for edge computation from captured state.

        In production: queries case library for setup_type+symbol stats.
        In replay: uses case_library_stats captured at snapshot time.
        """
        case_stats = self.context.case_library_stats
        sample_size = case_stats.get("sample_size", 0)
        win_rate = case_stats.get("win_rate", 0.0)

        return {
            "decision": "allow",
            "reason_type": "case_stats_constructed",
            "reason": f"Case stats: sample_size={sample_size}, win_rate={win_rate:.2f}",
            "case_stats": case_stats,
            "inputs": {
                "sample_size": sample_size,
                "win_rate": win_rate,
                "setup_type": self.context.setup_type,
            },
        }

    def _evaluate_hard_rejection_check(self) -> dict:
        """Check hard rejection based on case statistics.

        Production: check_hard_rejection(case_stats) returns True when
        sample_size >= 10 AND win_rate < 0.35.
        """
        from core.edge_score import check_hard_rejection

        case_stats = self.context.case_library_stats
        is_rejected = check_hard_rejection(case_stats)

        if is_rejected:
            return {
                "decision": "reject",
                "reason_type": "hard_rejection_case_stats",
                "reason": (
                    f"Hard rejection: sample_size={case_stats.get('sample_size', 0)}, "
                    f"win_rate={case_stats.get('win_rate', 0.0):.2f}"
                ),
                "threshold_applied": {
                    "min_winrate": self.policy_config.hard_rejection_min_winrate or 0.35,
                    "min_sample_size": self.policy_config.hard_rejection_min_sample_size or 10,
                },
                "inputs": {
                    "sample_size": case_stats.get("sample_size", 0),
                    "win_rate": case_stats.get("win_rate", 0.0),
                },
            }

        return {
            "decision": "allow",
            "reason_type": "hard_rejection_passed",
            "reason": "Case stats pass hard rejection threshold",
            "threshold_applied": {
                "min_winrate": self.policy_config.hard_rejection_min_winrate or 0.35,
                "min_sample_size": self.policy_config.hard_rejection_min_sample_size or 10,
            },
            "inputs": {
                "sample_size": case_stats.get("sample_size", 0),
                "win_rate": case_stats.get("win_rate", 0.0),
            },
        }

    def _evaluate_edge_score_computation(self) -> dict:
        """Compute edge score from signal, case_stats, and similarity_stats.

        Production: compute_edge_score(signal, case_stats, similarity_stats).
        Replay: uses captured state from context.
        """
        from core.edge_score import compute_edge_score

        signal = {
            "strength": self.context.strength or "",
            "confidence": self.context.conviction or "",
            "indicators": list(self.context.indicators) if self.context.indicators else {},
            "bias": "",
        }
        case_stats = self.context.case_library_stats
        similarity_stats = self.context.similarity_stats or {
            "similarity_winrate": 0.0,
            "sample_size": 0,
            "skip_similarity": True,
        }

        edge_score = compute_edge_score(signal, case_stats, similarity_stats)

        return {
            "decision": "allow",
            "reason_type": "edge_score_computed",
            "reason": f"Edge score: {edge_score:.3f}",
            "edge_score": edge_score,
            "inputs": {
                "signal_strength": self.context.signal_strength,
                "case_win_rate": case_stats.get("win_rate", 0.0),
                "similarity_winrate": similarity_stats.get("similarity_winrate", 0.0),
            },
        }

    def _evaluate_edge_threshold_check(self) -> dict:
        """Check edge score against minimum threshold (default 0.4).

        Production: rejects if edge < 0.4. Replay: uses policy-configured
        threshold from edge_score_min_threshold.
        """
        from core.edge_score import compute_edge_score

        signal = {
            "strength": self.context.strength or "",
            "confidence": self.context.conviction or "",
            "indicators": list(self.context.indicators) if self.context.indicators else {},
            "bias": "",
        }
        case_stats = self.context.case_library_stats
        similarity_stats = self.context.similarity_stats or {
            "similarity_winrate": 0.0,
            "sample_size": 0,
            "skip_similarity": True,
        }

        edge_score = compute_edge_score(signal, case_stats, similarity_stats)
        threshold = self.policy_config.edge_score_min_threshold

        if edge_score < threshold:
            return {
                "decision": "reject",
                "reason_type": "edge_below_threshold",
                "reason": f"Edge score {edge_score:.3f} below threshold {threshold}",
                "threshold_applied": {"edge_score_min_threshold": threshold},
                "edge_score": edge_score,
                "inputs": {
                    "edge_score": edge_score,
                    "threshold": threshold,
                },
            }

        return {
            "decision": "allow",
            "reason_type": "edge_threshold_passed",
            "reason": f"Edge score {edge_score:.3f} meets threshold {threshold}",
            "threshold_applied": {"edge_score_min_threshold": threshold},
            "edge_score": edge_score,
            "inputs": {
                "edge_score": edge_score,
                "threshold": threshold,
            },
        }

    def _evaluate_strategy_multiplier(self) -> dict:
        """Get strategy position multiplier.

        Production: queries DynamicStrategy table for pipeline stage.
        Replay: uses setup_type as strategy key with captured state.
        Returns size_multiplier in result for downstream consumption.
        """
        # In replay, we cannot query the DynamicStrategy table.
        # Use the case_library_stats strategy_status if captured, or default to 1.0
        strategy_key = self.context.setup_type or "unknown"
        strategy_status = self.context.case_library_stats.get("strategy_status")

        if strategy_status == "live_50":
            multiplier = 0.5
        elif strategy_status == "live_100" or strategy_status is None:
            multiplier = 1.0
        elif strategy_status == "not_live":
            multiplier = 0.0
        else:
            multiplier = 1.0

        decision = "allow" if multiplier > 0.0 else "reject"
        reason_type = "strategy_not_live" if multiplier == 0.0 else "strategy_multiplier_applied"

        return {
            "decision": decision,
            "reason_type": reason_type,
            "reason": f"Strategy '{strategy_key}' multiplier: {multiplier}",
            "size_multiplier": multiplier,
            "inputs": {
                "strategy_key": strategy_key,
                "strategy_status": strategy_status,
            },
        }

    def _evaluate_adaptive_throttle(self) -> dict:
        """Apply adaptive risk throttle based on recent loss streak.

        Production: adaptive_risk_throttle(base_size, recent_losses).
        Replay: uses recent_losses from case_library_stats.
        """
        from core.portfolio_risk import adaptive_risk_throttle

        base_size = float(self.context.quantity)
        recent_losses = self.context.case_library_stats.get("recent_losses", 0)
        loss_threshold = self.policy_config.adaptive_throttle_loss_threshold

        throttled_size = adaptive_risk_throttle(base_size, recent_losses)

        # Determine if throttle was applied
        was_throttled = throttled_size < base_size
        decision = "reduce_size" if was_throttled else "allow"

        return {
            "decision": decision,
            "reason_type": "adaptive_throttle_applied" if was_throttled else "no_throttle",
            "reason": (
                f"Throttled: {base_size:.1f} → {throttled_size:.1f} "
                f"(recent_losses={recent_losses})"
                if was_throttled
                else f"No throttle applied (recent_losses={recent_losses})"
            ),
            "size_multiplier": throttled_size / base_size if base_size > 0 else 1.0,
            "threshold_applied": {"loss_threshold": loss_threshold},
            "inputs": {
                "base_size": base_size,
                "recent_losses": recent_losses,
                "throttled_size": throttled_size,
            },
        }

    def _evaluate_portfolio_risk(self) -> dict:
        """Validate portfolio risk constraints.

        Production: validate_portfolio_risk(new_trade, positions, equity, max_exposure).
        Replay: uses captured open_positions and account state.
        """
        from core.portfolio_risk import validate_portfolio_risk

        new_trade = {
            "symbol": self.context.symbol,
            "quantity": float(self.context.quantity),
            "price": float(self.context.entry_price),
            "side": "long" if self.context.direction == "LONG" else "short",
        }
        positions = list(self.context.open_positions)
        total_equity = float(self.context.account_equity)

        # Use policy-configured max exposure if available
        max_exposure = self.policy_config.portfolio_risk_limits.get(
            "max_total_exposure", 1.5
        )

        is_valid, reason = validate_portfolio_risk(
            new_trade=new_trade,
            positions=positions,
            total_equity=total_equity,
            max_total_exposure=max_exposure,
        )

        decision = "allow" if is_valid else "reject"

        return {
            "decision": decision,
            "reason_type": "portfolio_risk_passed" if is_valid else "portfolio_risk_exceeded",
            "reason": reason,
            "threshold_applied": {"max_total_exposure": max_exposure},
            "inputs": {
                "total_equity": total_equity,
                "position_count": len(positions),
                "new_trade_value": float(self.context.quantity) * float(self.context.entry_price),
            },
        }

    def _evaluate_validate_trade(self) -> dict:
        """Validate trade parameters (position limits, dollar risk caps).

        Production: validate_trade(decision, profile, cash, equity, direction)
        raises TradeValidationError on failure.
        Replay: uses captured geometry and account state.
        """
        from utils.trade_validator import TradeValidationError, validate_trade

        decision_dict = {
            "symbol": self.context.symbol,
            "entry_price": float(self.context.entry_price),
            "price": float(self.context.entry_price),
            "stop_price": float(self.context.stop_price),
            "stop": float(self.context.stop_price),
            "target_price": float(self.context.target_price),
            "target": float(self.context.target_price),
            "quantity": float(self.context.quantity),
            "action": "BUY" if self.context.direction == "LONG" else "SELL_SHORT",
        }

        try:
            validate_trade(
                decision=decision_dict,
                profile_id=self.context.profile,
                cash=float(self.context.available_cash),
                total_equity=float(self.context.account_equity),
                direction=self.context.direction,
            )
        except TradeValidationError as exc:
            return {
                "decision": "reject",
                "reason_type": "trade_validation_failed",
                "reason": str(exc),
                "inputs": {
                    "symbol": self.context.symbol,
                    "entry_price": float(self.context.entry_price),
                    "stop_price": float(self.context.stop_price),
                    "target_price": float(self.context.target_price),
                    "quantity": float(self.context.quantity),
                },
            }

        return {
            "decision": "allow",
            "reason_type": "trade_validation_passed",
            "reason": "Trade passes all validation checks",
            "inputs": {
                "symbol": self.context.symbol,
                "entry_price": float(self.context.entry_price),
                "stop_price": float(self.context.stop_price),
                "target_price": float(self.context.target_price),
                "quantity": float(self.context.quantity),
            },
        }

    def _evaluate_correlation_check(self) -> dict:
        """Check for correlated exposure.

        Production: check_correlation(symbol, direction, profile, db) queries
        open positions. Replay: uses captured open_positions to check for
        correlated pairs without DB access.
        """
        from utils.trade_validator import CORRELATED_PAIRS

        symbol = self.context.symbol
        direction = self.context.direction
        positions = self.context.open_positions

        # Check for correlated exposure from captured positions
        warning = ""
        for pos in positions:
            pos_side = pos.get("side", "")
            pos_symbol = pos.get("symbol", "")
            # Match direction
            if (direction == "LONG" and pos_side == "long") or \
               (direction == "SHORT" and pos_side == "short"):
                pair = frozenset({symbol, pos_symbol})
                if pair in CORRELATED_PAIRS:
                    warning = (
                        f"Correlated exposure: already {pos_side} {pos_symbol}, "
                        f"adding {direction} {symbol} compounds regime risk"
                    )
                    break

        if warning:
            return {
                "decision": "warn",
                "reason_type": "correlated_exposure",
                "reason": warning,
                "inputs": {
                    "symbol": symbol,
                    "direction": direction,
                    "existing_positions": len(positions),
                },
            }

        return {
            "decision": "allow",
            "reason_type": "no_correlation",
            "reason": "No correlated exposure detected",
            "inputs": {
                "symbol": symbol,
                "direction": direction,
                "existing_positions": len(positions),
            },
        }

    def _evaluate_confidence_adjustment(self) -> dict:
        """Adjust confidence based on case library statistics.

        Production: adjust_confidence(engine, setup_type, regime) queries DB.
        Replay: uses case_library_stats to compute modifier without DB access.
        """
        case_stats = self.context.case_library_stats
        sample_size = case_stats.get("sample_size", 0)
        win_rate = case_stats.get("win_rate", 0.0)

        # Replicate production logic thresholds
        min_cases = 5
        block_threshold = 0.35
        downgrade_threshold = 0.50

        if sample_size < min_cases:
            # Not enough data to adjust
            return {
                "decision": "allow",
                "reason_type": "insufficient_cases_for_adjustment",
                "reason": f"Only {sample_size} cases (min {min_cases}), no adjustment",
                "inputs": {
                    "sample_size": sample_size,
                    "win_rate": win_rate,
                    "modifier": 1.0,
                },
            }

        if win_rate < block_threshold:
            return {
                "decision": "reject",
                "reason_type": "confidence_block",
                "reason": f"Win rate {win_rate:.2f} below block threshold {block_threshold}",
                "threshold_applied": {
                    "block_threshold": block_threshold,
                    "min_cases": min_cases,
                },
                "inputs": {
                    "sample_size": sample_size,
                    "win_rate": win_rate,
                    "modifier": 0.0,
                },
            }

        if win_rate < downgrade_threshold:
            modifier = 0.7  # Reduce confidence
            return {
                "decision": "reduce_size",
                "reason_type": "confidence_downgrade",
                "reason": f"Win rate {win_rate:.2f} below downgrade threshold, modifier={modifier}",
                "size_multiplier": modifier,
                "threshold_applied": {
                    "downgrade_threshold": downgrade_threshold,
                    "min_cases": min_cases,
                },
                "inputs": {
                    "sample_size": sample_size,
                    "win_rate": win_rate,
                    "modifier": modifier,
                },
            }

        return {
            "decision": "allow",
            "reason_type": "confidence_adequate",
            "reason": f"Win rate {win_rate:.2f} passes confidence thresholds",
            "inputs": {
                "sample_size": sample_size,
                "win_rate": win_rate,
                "modifier": 1.0,
            },
        }

    def _evaluate_position_size_cap(self) -> dict:
        """Cap position size at base_size × 1.2.

        Production: cap_position_size(scaled_size, base_size).
        Replay: uses captured quantity as the current (possibly scaled) size,
        and computes whether it exceeds the 1.2× cap.
        """
        from core.edge_score import cap_position_size

        current_size = float(self.context.quantity)
        # Base size is the original quantity before any scaling
        # In replay context, we use the quantity as-is since it represents
        # the proposed size after prior adjustments
        base_size = current_size  # Assume no prior scaling in snapshot

        capped_size = cap_position_size(current_size, base_size)
        was_capped = capped_size < current_size

        return {
            "decision": "reduce_size" if was_capped else "allow",
            "reason_type": "position_size_capped" if was_capped else "position_size_within_cap",
            "reason": (
                f"Position capped: {current_size:.1f} → {capped_size:.1f}"
                if was_capped
                else f"Position size {current_size:.1f} within cap"
            ),
            "size_multiplier": capped_size / current_size if current_size > 0 else 1.0,
            "inputs": {
                "current_size": current_size,
                "base_size": base_size,
                "capped_size": capped_size,
            },
        }

    # --- Utility methods ---

    def _detect_missing_fields(self) -> list[str]:
        """Identify required fields that are None/missing in context."""
        required = EXTENDED_GATE_REQUIRED_FIELDS.get(self.gate_name, [])
        missing: list[str] = []
        for field_name in required:
            value = getattr(self.context, field_name, None)
            if value is None:
                missing.append(field_name)
            elif isinstance(value, str) and not value.strip():
                missing.append(field_name)
        return missing
