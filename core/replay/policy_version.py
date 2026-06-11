"""Policy Version data model and construction utilities.

A PolicyVersion is an immutable identifier for the code/configuration bundle
used by a replay. It comprises: gate implementation revision, configuration
hash, benchmark/mapping version, feature-flag state, gate ordering version,
and adapter version.

See: design.md §core/replay/policy_version.py, requirements §4.1, §4.2, §4.4
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class PolicyVersion:
    """Immutable identifier for the code/configuration bundle used by a replay.

    Fields:
        name: Human-readable policy label (e.g. "current_v1.0", "candidate_aggressive_rr")
        gate_revision: Commit hash or build tag identifying the gate implementation code
        config_digest: Stable SHA-256 hex digest of canonicalized threshold/config values
        feature_flags: Feature-flag states active for this policy
        benchmark_version: Benchmark/mapping version identifier (nullable for historical)
        config_source_timestamp: When the configuration was captured/deployed (nullable)
        gate_ordering_version: Identifies the gate sequence version (e.g. "v1.0")
        adapter_version: Identifies the adapter logic version (e.g. "1.0.0")
    """

    name: str
    gate_revision: str
    config_digest: str
    feature_flags: dict[str, bool]
    benchmark_version: str | None
    config_source_timestamp: datetime | None
    gate_ordering_version: str
    adapter_version: str


# ---------------------------------------------------------------------------
# Required fields for candidate policy validation
# ---------------------------------------------------------------------------

_REQUIRED_POLICY_FIELDS: list[str] = [
    "name",
    "gate_revision",
    "feature_flags",
    "gate_ordering_version",
    "adapter_version",
]

# Gate thresholds that must be present for deterministic replay
_REQUIRED_THRESHOLD_FIELDS: list[str] = [
    "min_win_rate_by_setup",
    "default_min_win_rate",
    "stop_distance_rules",
    "default_stop_distance_rule",
    "override_min_confidence_score",
    "reduced_rr_thresholds_by_profile",
    "qualifying_min_signal_strength",
    "qualifying_setup_types",
]


# ---------------------------------------------------------------------------
# Construction utilities
# ---------------------------------------------------------------------------


def build_current_policy_version() -> PolicyVersion:
    """Construct PolicyVersion from currently deployed gate_config.py values and env vars.

    Reads all threshold values from the gate_config module, captures current
    feature flag state from environment variables, and attempts to resolve the
    current git revision for gate_revision.
    """
    from utils import gate_config

    config_values = _collect_current_config(gate_config)
    config_digest = _compute_config_digest(config_values)
    gate_revision = _resolve_git_revision()
    feature_flags = _collect_feature_flags()

    return PolicyVersion(
        name="current",
        gate_revision=gate_revision,
        config_digest=config_digest,
        feature_flags=feature_flags,
        benchmark_version=None,
        config_source_timestamp=datetime.utcnow(),
        gate_ordering_version="v1.0",
        adapter_version="1.0.0",
    )


def build_policy_from_snapshot(snapshot: dict) -> PolicyVersion | None:
    """Reconstruct PolicyVersion from a stored Decision_Snapshot.

    Args:
        snapshot: Dictionary containing snapshot fields. Expected keys:
            - policy_version_id (str): Previously recorded policy version identifier
            - gate_config (dict or str): Gate configuration at snapshot time
            - feature_flags (dict or str): Feature flags at snapshot time

    Returns:
        PolicyVersion if sufficient information is present, None otherwise.
    """
    # Extract policy version fields from the snapshot
    policy_version_id = snapshot.get("policy_version_id")
    if not policy_version_id:
        return None

    # Parse gate_config — may be stored as JSON string or dict
    gate_config_data = snapshot.get("gate_config")
    if isinstance(gate_config_data, str):
        try:
            gate_config_data = json.loads(gate_config_data)
        except (json.JSONDecodeError, TypeError):
            gate_config_data = {}
    elif gate_config_data is None:
        gate_config_data = {}

    # Parse feature_flags — may be stored as JSON string or dict
    feature_flags = snapshot.get("feature_flags")
    if isinstance(feature_flags, str):
        try:
            feature_flags = json.loads(feature_flags)
        except (json.JSONDecodeError, TypeError):
            feature_flags = {}
    elif feature_flags is None:
        feature_flags = {}

    # Ensure feature_flags values are booleans
    feature_flags = {k: bool(v) for k, v in feature_flags.items()}

    # Compute config digest from stored gate config
    config_digest = _compute_config_digest(gate_config_data)

    # Extract gate_revision and other metadata from the snapshot
    gate_revision = gate_config_data.get("gate_revision", "unknown")
    gate_ordering_version = gate_config_data.get("gate_ordering_version", "v1.0")
    adapter_version = gate_config_data.get("adapter_version", "1.0.0")
    benchmark_version = gate_config_data.get("benchmark_version")

    # Parse config_source_timestamp if present
    config_source_timestamp = None
    ts_raw = gate_config_data.get("config_source_timestamp")
    if ts_raw:
        try:
            config_source_timestamp = datetime.fromisoformat(ts_raw)
        except (ValueError, TypeError):
            config_source_timestamp = None

    return PolicyVersion(
        name=gate_config_data.get("name", policy_version_id),
        gate_revision=gate_revision,
        config_digest=config_digest,
        feature_flags=feature_flags,
        benchmark_version=benchmark_version,
        config_source_timestamp=config_source_timestamp,
        gate_ordering_version=gate_ordering_version,
        adapter_version=adapter_version,
    )


def validate_candidate_policy(policy_spec: dict) -> tuple[bool, list[str]]:
    """Validate a candidate policy has all required fields for deterministic replay.

    A Candidate_Policy must include a complete, deterministic set of gate
    thresholds, feature-flag states, and gate implementation revision sufficient
    to reproduce its gate decisions.

    Args:
        policy_spec: Dictionary specifying the candidate policy configuration.

    Returns:
        Tuple of (is_valid, list_of_missing_or_ambiguous_fields).
        is_valid is True only when missing_fields is empty.
    """
    missing_fields: list[str] = []

    # Check top-level required fields
    for field in _REQUIRED_POLICY_FIELDS:
        value = policy_spec.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing_fields.append(field)

    # Check that feature_flags is a dict with boolean values
    feature_flags = policy_spec.get("feature_flags")
    if isinstance(feature_flags, dict):
        for key, val in feature_flags.items():
            if not isinstance(val, bool):
                missing_fields.append(f"feature_flags.{key} (must be boolean)")
    elif feature_flags is not None:
        missing_fields.append("feature_flags (must be a dict)")

    # Check required gate threshold fields
    thresholds = policy_spec.get("thresholds", policy_spec)
    for field in _REQUIRED_THRESHOLD_FIELDS:
        value = thresholds.get(field)
        if value is None:
            missing_fields.append(f"thresholds.{field}")

    # Validate name uniqueness requirement (must have a name)
    name = policy_spec.get("name")
    if name is not None and isinstance(name, str) and not name.strip():
        if "name" not in missing_fields:
            missing_fields.append("name (must be non-empty)")

    is_valid = len(missing_fields) == 0
    return (is_valid, missing_fields)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_current_config(gate_config_module: Any) -> dict:
    """Collect all threshold/configuration values from the deployed gate_config module."""
    return {
        # Setup quality gate
        "min_win_rate_by_setup": gate_config_module.MIN_WIN_RATE_BY_SETUP,
        "default_min_win_rate": gate_config_module.DEFAULT_MIN_WIN_RATE,
        "min_win_rate_by_setup_profile": gate_config_module.MIN_WIN_RATE_BY_SETUP_PROFILE,
        "default_min_win_rate_by_profile": gate_config_module.DEFAULT_MIN_WIN_RATE_BY_PROFILE,
        "rolling_window": gate_config_module.ROLLING_WINDOW,
        "min_cases_for_block": gate_config_module.MIN_CASES_FOR_BLOCK,
        "min_rolling_cases": gate_config_module.MIN_ROLLING_CASES,
        "consecutive_loss_pause_threshold": gate_config_module.CONSECUTIVE_LOSS_PAUSE_THRESHOLD,
        "recovery_min_rolling_cases": gate_config_module.RECOVERY_MIN_ROLLING_CASES,
        "recovery_win_rate_margin": gate_config_module.RECOVERY_WIN_RATE_MARGIN,
        "require_positive_rolling_avg_pnl_for_recovery": gate_config_module.REQUIRE_POSITIVE_ROLLING_AVG_PNL_FOR_RECOVERY,
        "rolling_recovery_probe_size_multiplier": gate_config_module.ROLLING_RECOVERY_PROBE_SIZE_MULTIPLIER,
        "near_miss_margin_pct": gate_config_module.NEAR_MISS_MARGIN_PCT,
        # Pre-trade quality gate
        "override_min_confidence_score": gate_config_module.OVERRIDE_MIN_CONFIDENCE_SCORE,
        # Risk geometry gate
        "stop_distance_rules": gate_config_module.STOP_DISTANCE_RULES,
        "default_stop_distance_rule": gate_config_module.DEFAULT_STOP_DISTANCE_RULE,
        "reduced_rr_thresholds_by_profile": gate_config_module.REDUCED_RR_THRESHOLDS_BY_PROFILE,
        "high_beta_cluster": gate_config_module.HIGH_BETA_CLUSTER,
        "qualifying_min_signal_strength": gate_config_module.QUALIFYING_MIN_SIGNAL_STRENGTH,
        "qualifying_setup_types": gate_config_module.QUALIFYING_SETUP_TYPES,
        # Catalyst specificity gate
        "catalyst_specificity_profile_thresholds": gate_config_module.CATALYST_SPECIFICITY_PROFILE_THRESHOLDS,
        "catalyst_specificity_sector_sympathy_size_multiplier": gate_config_module.CATALYST_SPECIFICITY_SECTOR_SYMPATHY_SIZE_MULTIPLIER,
    }


def _collect_feature_flags() -> dict[str, bool]:
    """Collect current feature flag state from environment variables."""
    return {
        "SETUP_SPECIFIC_RR_THRESHOLDS": (
            os.environ.get("SETUP_SPECIFIC_RR_THRESHOLDS", "").strip().lower() == "true"
        ),
        "MODERATE_NEAR_MISS_PILOT": (
            os.environ.get("MODERATE_NEAR_MISS_PILOT", "").strip().lower() == "true"
        ),
        "PM_CANDIDATE_MODE": os.environ.get("PM_CANDIDATE_MODE", "disabled") != "disabled",
        "PM_BENCHMARK_CONTEXT_ENABLED": (
            os.environ.get("PM_BENCHMARK_CONTEXT_ENABLED", "false").strip().lower() == "true"
        ),
    }


def _resolve_git_revision() -> str:
    """Attempt to resolve the current git HEAD commit hash.

    Returns 'unknown' if git is not available or the project is not a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return "unknown"


def _compute_config_digest(config_values: dict) -> str:
    """Compute a stable SHA-256 hex digest of canonicalized configuration values.

    Canonical form:
    - Sort keys recursively
    - Decimal → string representation (normalized)
    - set/frozenset → sorted list
    - datetime → ISO-8601 string
    - float → normalized string (no trailing zeros)
    - Remove whitespace (compact separators)
    """
    canonical = json.dumps(
        _normalize_for_hash(config_values),
        sort_keys=True,
        separators=(",", ":"),
        default=_json_serializer,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_for_hash(obj: Any) -> Any:
    """Recursively normalize complex types for canonical serialization."""
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
    # Fallback: convert to string
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
