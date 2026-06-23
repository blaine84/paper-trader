"""ReplayGateAdapter layer — mediates between replay engine and production gates.

Production gate functions accept optional dependency kwargs that default to current
behavior. Replay passes explicit overrides: GatePolicyConfig for thresholds, noop
event sink, frozen clock, and deterministic ID provider.

No global mutation, no patched os.environ, no module-constant overrides.
If replay shares the process with production, live behavior is never altered.

See: design.md §core/replay/gate_adapter.py, requirements §5.1, §5.2, §5.3, §13.2
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable


# ---------------------------------------------------------------------------
# GatePolicyConfig — frozen policy passed via `policy` kwarg at replay time
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GatePolicyConfig:
    """Frozen configuration passed to gate functions at replay time via the `policy` kwarg.

    Replaces module-level constant reads and os.environ lookups during replay.
    NOT hashable via Python __hash__ (process-dependent). Use config_digest() for
    stable identity suitable for persistence and comparison.
    """

    # --- Setup quality gate thresholds ---
    min_win_rate_by_setup: dict[str, float]
    default_min_win_rate: float
    min_win_rate_by_setup_profile: dict[str, dict[str, float]]
    default_min_win_rate_by_profile: dict[str, float]
    rolling_window: int
    min_cases_for_block: int
    min_rolling_cases: int
    consecutive_loss_pause_threshold: int
    recovery_min_rolling_cases: int
    recovery_win_rate_margin: float
    require_positive_rolling_avg_pnl_for_recovery: bool
    rolling_recovery_probe_size_multiplier: float
    near_miss_margin_pct: float

    # --- Pre-trade quality gate thresholds ---
    override_min_confidence_score: float

    # --- Risk geometry gate thresholds ---
    stop_distance_rules: dict[str, dict]
    default_stop_distance_rule: dict
    reduced_rr_thresholds_by_profile: dict[str, float]
    high_beta_cluster: frozenset[str]
    qualifying_min_signal_strength: float
    qualifying_setup_types: frozenset[str]

    # --- Feature flags (replaces os.environ reads) ---
    feature_flags: dict[str, bool]
    # Specifically: SETUP_SPECIFIC_RR_THRESHOLDS, MODERATE_NEAR_MISS_PILOT

    # --- Extended decision boundary (Phase 2) ---
    edge_score_min_threshold: float
    hard_rejection_min_winrate: float
    hard_rejection_min_sample_size: int
    adaptive_throttle_loss_threshold: int
    portfolio_risk_limits: dict[str, float]
    correlation_limits: dict[str, Any]

    # --- Catalyst specificity gate thresholds ---
    catalyst_specificity_profile_thresholds: dict[str, dict[str, int]]
    catalyst_specificity_sector_sympathy_size_multiplier: dict[str, float]

    # --- Policy identity (for versioning, NOT Python __hash__) ---
    gate_ordering_version: str
    adapter_version: str

    def config_digest(self) -> str:
        """Stable SHA-256 hex digest for policy identity.

        NOT process-dependent like __hash__. Safe for persistence and comparison.
        Uses canonical JSON serialization: sort keys, Decimal→str, set→sorted list,
        datetime→ISO-8601, float→normalized string.
        """
        return hashlib.sha256(self._canonical_json().encode("utf-8")).hexdigest()

    def _canonical_json(self) -> str:
        """Canonicalize all fields for stable hashing."""
        return json.dumps(
            _normalize_for_hash(self._to_dict()),
            sort_keys=True,
            separators=(",", ":"),
            default=_json_serializer,
        )

    def _to_dict(self) -> dict:
        """Convert to a plain dict, handling frozensets → sorted lists."""
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, frozenset):
                result[k] = sorted(v)
            else:
                result[k] = v
        return result


# ---------------------------------------------------------------------------
# ReplayGateContext — all captured historical inputs for gate evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayGateContext:
    """Injectable context supplying captured historical inputs to gates
    instead of DB queries. Passed to gate functions alongside GatePolicyConfig.
    """

    # Account state at cutoff
    account_equity: Decimal
    available_cash: Decimal
    open_positions: tuple[dict, ...] = field(default_factory=tuple)

    # Case library state at cutoff
    case_library_stats: dict = field(default_factory=dict)
    similarity_stats: dict | None = None  # Phase 2: for edge computation

    # Signal state at cutoff
    analyst_signal_payload: dict | None = None
    signal_strength: float | None = None
    confidence_value: float | None = None
    selection_score: float | None = None  # for pre_trade_quality_gate
    execution_score: float | None = None  # for pre_trade_quality_gate
    override_confidence_score: float | None = None
    override_reason: str | None = None

    # Market data at cutoff
    atr_value: float | None = None
    atr_timestamp: datetime | None = None
    current_price: float | None = None

    # Geometry (Decimal for precision)
    entry_price: Decimal = Decimal("0")
    stop_price: Decimal = Decimal("0")
    target_price: Decimal = Decimal("0")
    quantity: Decimal = Decimal("0")
    max_dollar_risk: float | None = None

    # Metadata
    symbol: str = ""
    profile: str = ""
    direction: str = ""
    setup_type: str | None = None
    catalyst_type: str | None = None
    trade_metadata: str | None = None
    trade_rationale: str | None = None
    atr_source: str | None = None

    # Catalyst gate fields
    rationale: str | None = None
    thesis: str | None = None
    indicators: tuple | None = None
    quote_timestamp: datetime | None = None
    strength: str | None = None
    conviction: str | None = None


# ---------------------------------------------------------------------------
# Canonical decision normalization mapping
# ---------------------------------------------------------------------------

DECISION_NORMALIZATION: dict[str, str] = {
    "allow": "allow",
    "allowed": "allow",
    "adjusted_allowed": "allow",
    "reject": "reject",
    "rejected": "reject",
    "block": "reject",
    "blocked": "reject",
    "reduce_size": "reduce_size",
    "downgrade": "reduce_size",
    "override_required": "override_required",
    "warn": "warn",
    "error": "error",
}


def _normalize_decision(raw_decision: str) -> str:
    """Map heterogeneous gate output to canonical decision string."""
    return DECISION_NORMALIZATION.get(raw_decision.lower().strip(), "error")


# ---------------------------------------------------------------------------
# Gate required fields — derived from actual gate function contracts
# ---------------------------------------------------------------------------

GATE_REQUIRED_FIELDS: dict[str, list[str]] = {
    "setup_quality_gate": [
        "setup_type",
        "symbol",
        "profile",
        "case_library_stats",
        "confidence_value",
        "catalyst_type",
    ],
    "pre_trade_quality_gate": [
        "selection_score",
        "execution_score",
        "override_confidence_score",
        "override_reason",
    ],
    "catalyst_specificity_gate": [
        "catalyst_type",
        "setup_type",
        "rationale",
        "thesis",
        "indicators",
        "quote_timestamp",
        "strength",
        "conviction",
        "quantity",
        "direction",
        "profile",
        "symbol",
    ],
    "risk_geometry_gate": [
        "entry_price",
        "stop_price",
        "target_price",
        "quantity",
        "direction",
        "atr_value",
        "atr_timestamp",
        "signal_strength",
        "confidence_value",
        "setup_type",
        "profile",
        "symbol",
        "max_dollar_risk",
        "trade_metadata",
        "trade_rationale",
        "atr_source",
    ],
}


# ---------------------------------------------------------------------------
# ReplayGateAdapter — wraps production gate calls with DI kwargs
# ---------------------------------------------------------------------------


class ReplayGateAdapter:
    """Mediates between replay engine and production gate functions.

    Does NOT monkeypatch globals. Instead calls gate functions with explicit
    dependency injection kwargs:
    - policy=GatePolicyConfig (overrides module constant reads)
    - event_sink=noop (suppresses log_trade_event calls)
    - clock=frozen_clock (deterministic time)
    - id_provider=deterministic_id (reproducible UUIDs)
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
        """Call the production gate function with injected dependencies.

        Returns a normalized result dict with at minimum:
            - decision: canonical decision string
            - reason_type: gate-specific reason code
            - raw_result: the unmodified dict returned by the gate
            - missing_fields: list of required fields that are None/missing
        """
        from utils.catalyst_specificity import evaluate_catalyst_specificity
        from utils.pre_trade_quality_gate import evaluate_pre_trade_quality
        from utils.risk_geometry_gate import evaluate_risk_geometry
        from utils.setup_quality_gate import evaluate_setup_quality

        missing_fields = self._detect_missing_fields()

        gate_dispatch = {
            "setup_quality_gate": self._evaluate_setup_quality,
            "pre_trade_quality_gate": self._evaluate_pre_trade_quality,
            "catalyst_specificity_gate": self._evaluate_catalyst_specificity,
            "risk_geometry_gate": self._evaluate_risk_geometry,
        }

        evaluator = gate_dispatch.get(self.gate_name)
        if evaluator is None:
            return {
                "decision": "error",
                "reason_type": "unknown_gate",
                "reason": f"No adapter for gate: {self.gate_name}",
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
        raw_decision = raw_result.get("decision", "error")
        canonical = _normalize_decision(str(raw_decision))

        return {
            "decision": canonical,
            "reason_type": raw_result.get("reason_type", ""),
            "reason": raw_result.get("reason", ""),
            "raw_result": raw_result,
            "missing_fields": missing_fields,
        }

    # --- Gate-specific evaluation methods ---

    def _evaluate_setup_quality(self) -> dict:
        """Call setup_quality_gate with DI kwargs."""
        from utils.setup_quality_gate import evaluate_setup_quality

        return evaluate_setup_quality(
            engine=None,  # Replay does not perform DB queries
            db=None,
            setup_type=self.context.setup_type or "",
            market_regime=None,
            symbol=self.context.symbol,
            profile=self.context.profile,
            agent="replay",
            confidence_score=self.context.confidence_value,
            catalyst_type=self.context.catalyst_type,
            policy=self.policy_config,
            event_sink=self._noop_sink,
            clock=self.replay_clock,
            id_provider=self.id_provider,
        )

    def _evaluate_pre_trade_quality(self) -> dict:
        """Call pre_trade_quality_gate with DI kwargs."""
        from utils.pre_trade_quality_gate import evaluate_pre_trade_quality

        decision_dict = {
            "selection_score": self.context.selection_score,
            "execution_score": self.context.execution_score,
            "override_confidence_score": self.context.override_confidence_score,
            "override_reason": self.context.override_reason,
        }
        signal_dict = self.context.analyst_signal_payload

        return evaluate_pre_trade_quality(
            db=None,
            decision=decision_dict,
            signal=signal_dict,
            symbol=self.context.symbol,
            profile=self.context.profile,
            agent="replay",
            policy=self.policy_config,
            event_sink=self._noop_sink,
            clock=self.replay_clock,
            id_provider=self.id_provider,
        )

    def _evaluate_catalyst_specificity(self) -> dict:
        """Call catalyst_specificity_gate with DI kwargs."""
        from utils.catalyst_specificity import evaluate_catalyst_specificity

        decision_dict = {
            "symbol": self.context.symbol,
            "setup_type": self.context.setup_type,
            "direction": self.context.direction,
            "quantity": float(self.context.quantity),
            "rationale": self.context.rationale,
            "thesis": self.context.thesis,
            "indicators": list(self.context.indicators) if self.context.indicators else [],
            "quote_timestamp": self.context.quote_timestamp,
            "strength": self.context.strength,
            "conviction": self.context.conviction,
            "catalyst_type": self.context.catalyst_type,
            "trade_metadata": self.context.trade_metadata,
        }
        signal_dict = self.context.analyst_signal_payload

        return evaluate_catalyst_specificity(
            decision=decision_dict,
            signal=signal_dict,
            profile=self.context.profile or "moderate",
            db=None,
            policy=self.policy_config,
            event_sink=self._noop_sink,
            clock=self.replay_clock,
            id_provider=self.id_provider,
        )

    def _evaluate_risk_geometry(self) -> dict:
        """Call risk_geometry_gate with DI kwargs."""
        from utils.risk_geometry_gate import evaluate_risk_geometry

        return evaluate_risk_geometry(
            entry_price=float(self.context.entry_price),
            stop_price=float(self.context.stop_price),
            target_price=float(self.context.target_price),
            quantity=float(self.context.quantity),
            direction=self.context.direction,
            symbol=self.context.symbol,
            setup_type=self.context.setup_type,
            atr_5min=self.context.atr_value,
            atr_timestamp=self.context.atr_timestamp,
            atr_source=self.context.atr_source,
            trade_timestamp=self.replay_clock(),
            max_dollar_risk=self.context.max_dollar_risk or 0.0,
            db=None,
            profile=self.context.profile,
            agent="replay",
            trade_metadata=self.context.trade_metadata,
            trade_rationale=self.context.trade_rationale,
            signal_strength=self.context.signal_strength,
            confidence_level=self.context.confidence_value,
            policy=self.policy_config,
            event_sink=self._noop_sink,
            clock=self.replay_clock,
            id_provider=self.id_provider,
        )

    # --- Utility methods ---

    def _noop_sink(self, *args: Any, **kwargs: Any) -> None:
        """No-op event sink replacing log_trade_event during replay."""
        pass

    def _detect_missing_fields(self) -> list[str]:
        """Identify required fields that are None/missing in context."""
        required = GATE_REQUIRED_FIELDS.get(self.gate_name, [])
        missing: list[str] = []
        for field_name in required:
            value = getattr(self.context, field_name, None)
            if value is None:
                missing.append(field_name)
            elif isinstance(value, str) and not value.strip():
                missing.append(field_name)
        return missing


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------


def build_replay_clock(cutoff: datetime) -> Callable[[], datetime]:
    """Return a frozen clock that always returns the replay cutoff.

    During replay, all time-dependent logic sees the same timestamp
    ensuring deterministic results.
    """

    def _clock() -> datetime:
        return cutoff

    return _clock


def build_deterministic_id_provider(
    replay_id: str, gate_name: str, candidate_id: str
) -> Callable[[], str]:
    """Return a deterministic ID provider seeded from replay context.

    Uses UUID5 with a fixed namespace and a seed derived from the replay_id,
    gate_name, and candidate_id. Produces reproducible IDs for the same
    replay execution context.
    """
    namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # fixed namespace
    seed = f"{replay_id}:{gate_name}:{candidate_id}"

    def _provider() -> str:
        return str(uuid.uuid5(namespace, seed))

    return _provider


def build_gate_policy_config_from_snapshot(snapshot: dict) -> GatePolicyConfig:
    """Reconstruct GatePolicyConfig from a stored Decision_Snapshot's gate_config field.

    The snapshot's gate_config is expected to contain all threshold values
    that were active at decision time. Falls back to sensible defaults where
    a field is absent (historical snapshots may predate some fields).
    """
    gate_config = snapshot.get("gate_config", {})
    if isinstance(gate_config, str):
        try:
            gate_config = json.loads(gate_config)
        except (json.JSONDecodeError, TypeError):
            gate_config = {}

    feature_flags = snapshot.get("feature_flags", {})
    if isinstance(feature_flags, str):
        try:
            feature_flags = json.loads(feature_flags)
        except (json.JSONDecodeError, TypeError):
            feature_flags = {}
    feature_flags = {k: bool(v) for k, v in feature_flags.items()}

    return GatePolicyConfig(
        # Setup quality gate
        min_win_rate_by_setup=gate_config.get("min_win_rate_by_setup", {}),
        default_min_win_rate=gate_config.get("default_min_win_rate", 0.40),
        min_win_rate_by_setup_profile=gate_config.get("min_win_rate_by_setup_profile", {}),
        default_min_win_rate_by_profile=gate_config.get("default_min_win_rate_by_profile", {}),
        rolling_window=gate_config.get("rolling_window", 5),
        min_cases_for_block=gate_config.get("min_cases_for_block", 5),
        min_rolling_cases=gate_config.get("min_rolling_cases", 3),
        consecutive_loss_pause_threshold=gate_config.get(
            "consecutive_loss_pause_threshold", 3
        ),
        recovery_min_rolling_cases=gate_config.get("recovery_min_rolling_cases", 5),
        recovery_win_rate_margin=gate_config.get("recovery_win_rate_margin", 0.15),
        require_positive_rolling_avg_pnl_for_recovery=gate_config.get(
            "require_positive_rolling_avg_pnl_for_recovery", True
        ),
        rolling_recovery_probe_size_multiplier=gate_config.get(
            "rolling_recovery_probe_size_multiplier", 0.25
        ),
        near_miss_margin_pct=gate_config.get("near_miss_margin_pct", 0.05),
        # Pre-trade quality gate
        override_min_confidence_score=gate_config.get(
            "override_min_confidence_score", 8.0
        ),
        # Risk geometry gate
        stop_distance_rules=gate_config.get("stop_distance_rules", {}),
        default_stop_distance_rule=gate_config.get("default_stop_distance_rule", {}),
        reduced_rr_thresholds_by_profile=gate_config.get(
            "reduced_rr_thresholds_by_profile", {}
        ),
        high_beta_cluster=frozenset(gate_config.get("high_beta_cluster", [])),
        qualifying_min_signal_strength=gate_config.get(
            "qualifying_min_signal_strength", 7.5
        ),
        qualifying_setup_types=frozenset(
            gate_config.get("qualifying_setup_types", [])
        ),
        # Feature flags
        feature_flags=feature_flags,
        # Extended decision boundary (Phase 2)
        edge_score_min_threshold=gate_config.get("edge_score_min_threshold", 0.4),
        hard_rejection_min_winrate=gate_config.get("hard_rejection_min_winrate", 0.0),
        hard_rejection_min_sample_size=gate_config.get(
            "hard_rejection_min_sample_size", 0
        ),
        adaptive_throttle_loss_threshold=gate_config.get(
            "adaptive_throttle_loss_threshold", 3
        ),
        portfolio_risk_limits=gate_config.get("portfolio_risk_limits", {}),
        correlation_limits=gate_config.get("correlation_limits", {}),
        # Catalyst specificity gate
        catalyst_specificity_profile_thresholds=gate_config.get(
            "catalyst_specificity_profile_thresholds", {}
        ),
        catalyst_specificity_sector_sympathy_size_multiplier=gate_config.get(
            "catalyst_specificity_sector_sympathy_size_multiplier", {}
        ),
        # Policy identity
        gate_ordering_version=gate_config.get("gate_ordering_version", "v1.0"),
        adapter_version=gate_config.get("adapter_version", "1.0.0"),
    )


def build_current_gate_policy_config() -> GatePolicyConfig:
    """Build GatePolicyConfig from currently deployed gate_config.py values and env vars.

    Reads all threshold values from the gate_config module and captures current
    feature flag state from environment variables.
    """
    from utils import gate_config as gc

    feature_flags: dict[str, bool] = {
        "SETUP_SPECIFIC_RR_THRESHOLDS": (
            os.environ.get("SETUP_SPECIFIC_RR_THRESHOLDS", "").strip().lower() == "true"
        ),
        "MODERATE_NEAR_MISS_PILOT": (
            os.environ.get("MODERATE_NEAR_MISS_PILOT", "").strip().lower() == "true"
        ),
        "PM_CANDIDATE_MODE": os.environ.get("PM_CANDIDATE_MODE", "disabled") != "disabled",
        "PM_BENCHMARK_CONTEXT_ENABLED": (
            os.environ.get("PM_BENCHMARK_CONTEXT_ENABLED", "false").strip().lower()
            == "true"
        ),
        "CATALYST_SPECIFICITY_GATE_ENABLED": (
            os.environ.get("CATALYST_SPECIFICITY_GATE_ENABLED", "true").strip().lower()
            != "false"
        ),
    }

    return GatePolicyConfig(
        # Setup quality gate
        min_win_rate_by_setup=gc.MIN_WIN_RATE_BY_SETUP,
        default_min_win_rate=gc.DEFAULT_MIN_WIN_RATE,
        min_win_rate_by_setup_profile=gc.MIN_WIN_RATE_BY_SETUP_PROFILE,
        default_min_win_rate_by_profile=gc.DEFAULT_MIN_WIN_RATE_BY_PROFILE,
        rolling_window=gc.ROLLING_WINDOW,
        min_cases_for_block=gc.MIN_CASES_FOR_BLOCK,
        min_rolling_cases=gc.MIN_ROLLING_CASES,
        consecutive_loss_pause_threshold=gc.CONSECUTIVE_LOSS_PAUSE_THRESHOLD,
        recovery_min_rolling_cases=gc.RECOVERY_MIN_ROLLING_CASES,
        recovery_win_rate_margin=gc.RECOVERY_WIN_RATE_MARGIN,
        require_positive_rolling_avg_pnl_for_recovery=gc.REQUIRE_POSITIVE_ROLLING_AVG_PNL_FOR_RECOVERY,
        rolling_recovery_probe_size_multiplier=gc.ROLLING_RECOVERY_PROBE_SIZE_MULTIPLIER,
        near_miss_margin_pct=gc.NEAR_MISS_MARGIN_PCT,
        # Pre-trade quality gate
        override_min_confidence_score=gc.OVERRIDE_MIN_CONFIDENCE_SCORE,
        # Risk geometry gate
        stop_distance_rules=gc.STOP_DISTANCE_RULES,
        default_stop_distance_rule=gc.DEFAULT_STOP_DISTANCE_RULE,
        reduced_rr_thresholds_by_profile=gc.REDUCED_RR_THRESHOLDS_BY_PROFILE,
        high_beta_cluster=frozenset(gc.HIGH_BETA_CLUSTER),
        qualifying_min_signal_strength=gc.QUALIFYING_MIN_SIGNAL_STRENGTH,
        qualifying_setup_types=frozenset(gc.QUALIFYING_SETUP_TYPES),
        # Feature flags
        feature_flags=feature_flags,
        # Extended decision boundary (Phase 2)
        edge_score_min_threshold=0.4,
        hard_rejection_min_winrate=0.0,
        hard_rejection_min_sample_size=0,
        adaptive_throttle_loss_threshold=3,
        portfolio_risk_limits={},
        correlation_limits={},
        # Catalyst specificity gate
        catalyst_specificity_profile_thresholds=gc.CATALYST_SPECIFICITY_PROFILE_THRESHOLDS,
        catalyst_specificity_sector_sympathy_size_multiplier=gc.CATALYST_SPECIFICITY_SECTOR_SYMPATHY_SIZE_MULTIPLIER,
        # Policy identity
        gate_ordering_version="v1.0",
        adapter_version="1.0.0",
    )


# ---------------------------------------------------------------------------
# Canonical JSON serialization helpers (shared with policy_version.py)
# ---------------------------------------------------------------------------


def _normalize_for_hash(obj: Any) -> Any:
    """Recursively normalize complex types for canonical serialization.

    - dict → sorted by key
    - set/frozenset → sorted list
    - list/tuple → preserved order, elements normalized
    - Decimal → string (normalized, no trailing zeros)
    - datetime → ISO-8601 string
    - float → normalized string (no trailing zeros)
    - bool/int/str/None → pass through
    """
    if isinstance(obj, dict):
        return {str(k): _normalize_for_hash(v) for k, v in sorted(obj.items(), key=lambda x: str(x[0]))}
    if isinstance(obj, (set, frozenset)):
        return sorted(_normalize_for_hash(x) for x in obj)
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_hash(x) for x in obj]
    if isinstance(obj, Decimal):
        return str(obj.normalize())
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, float):
        return f"{obj:.10g}"
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, str):
        return obj
    if obj is None:
        return None
    return str(obj)


def _json_serializer(obj: Any) -> Any:
    """JSON serializer for types not natively handled by json.dumps."""
    if isinstance(obj, (set, frozenset)):
        return sorted(str(x) for x in obj)
    if isinstance(obj, Decimal):
        return str(obj.normalize())
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Not serializable: {type(obj)}")
