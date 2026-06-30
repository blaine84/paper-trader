"""Dispatch mode resolution — pure functions for per-alert-type mode control.

Resolves effective dispatch mode from global and per-alert environment
configuration, applying restrictiveness precedence.

Requirements: 1.1, 1.5, 1.6, 1.7, 10.1, 10.5, 10.6
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Mode restrictiveness: disabled (most) > observe > dispatch (least)
_MODE_PRECEDENCE = {"disabled": 2, "observe": 1, "dispatch": 0}
_VALID_MODES = frozenset(_MODE_PRECEDENCE.keys())
# "enabled" is a legacy alias for "dispatch" (existing PM_ALERT_DISPATCH_MODE uses it)
_MODE_ALIASES = {"enabled": "dispatch"}
_ALERT_TYPES = frozenset({"entry_alert", "breakout", "rapid_move", "target_hit"})


def _normalize_mode(raw: str) -> Optional[str]:
    """Normalize a raw mode string. Returns canonical mode or None if invalid.

    Handles: whitespace trimming, lowercasing, 'enabled' → 'dispatch' alias.
    Returns None for empty strings OR whitespace-only strings (indicating "unset/inherit").
    Returns "INVALID" sentinel for non-empty strings that don't match valid modes.
    """
    stripped = raw.strip().lower() if raw else ""
    if not stripped:
        return None  # unset or whitespace-only — caller decides inheritance
    # Apply alias
    canonical = _MODE_ALIASES.get(stripped, stripped)
    if canonical in _VALID_MODES:
        return canonical
    return "INVALID"  # sentinel — caller handles


@dataclass(frozen=True)
class DispatchModeConfig:
    """Immutable resolved dispatch mode configuration."""

    global_mode: str  # always one of: dispatch, observe, disabled
    per_alert_modes: dict[str, str]  # alert_type → effective mode

    def effective_mode(self, alert_type: str) -> str:
        """Return the effective mode for an alert type."""
        return self.per_alert_modes.get(alert_type, self.global_mode)


def resolve_per_alert_mode(
    raw_value: str,
    global_mode: str,
    alert_type: str,
) -> str:
    """Resolve a single alert type's effective mode.

    Rules:
    1. If raw_value is empty/unset/whitespace-only → INHERIT global_mode
    2. If raw_value is invalid (not dispatch/enabled/observe/disabled) → treat as disabled, log warning
    3. Effective = more restrictive of (resolved per-alert, global_mode)

    IMPORTANT: Empty/unset/whitespace-only is NOT the same as invalid.
    - Empty or whitespace → inherit global (could be dispatch, observe, or disabled)
    - Invalid (non-empty with content) → disabled (fail-closed for unknown values)
    """
    normalized = _normalize_mode(raw_value)

    if normalized is None:
        # Unset/empty: inherit from global
        per_alert = global_mode
    elif normalized == "INVALID":
        # Invalid value: log warning, treat as disabled
        logger.warning(
            "PM_ALERT_MODE_%s: invalid value '%s', treating as disabled",
            alert_type.upper(),
            raw_value,
        )
        per_alert = "disabled"
    else:
        per_alert = normalized

    # Apply restrictiveness precedence: max precedence wins
    global_prec = _MODE_PRECEDENCE.get(global_mode, 0)
    per_alert_prec = _MODE_PRECEDENCE.get(per_alert, 0)
    if global_prec > per_alert_prec:
        return global_mode
    return per_alert


def normalize_global_mode(raw: str) -> str:
    """Normalize the global PM_ALERT_DISPATCH_MODE value.

    Handles the legacy 'enabled' → 'dispatch' mapping.
    Invalid/empty defaults to 'disabled' (fail-closed).
    """
    normalized = _normalize_mode(raw)
    if normalized is None or normalized == "INVALID":
        return "disabled"
    return normalized


def build_dispatch_mode_config(
    global_mode_raw: str,
    env_values: dict[str, str],
) -> DispatchModeConfig:
    """Build complete mode config from global mode and env var dict.

    global_mode_raw: raw PM_ALERT_DISPATCH_MODE value (may be "enabled")
    env_values keys: alert_type names (entry_alert, breakout, etc.)
    env_values values: raw env var strings (may be empty/invalid)
    """
    global_mode = normalize_global_mode(global_mode_raw)
    per_alert_modes = {}
    for alert_type in _ALERT_TYPES:
        raw = env_values.get(alert_type, "")
        per_alert_modes[alert_type] = resolve_per_alert_mode(raw, global_mode, alert_type)
    return DispatchModeConfig(global_mode=global_mode, per_alert_modes=per_alert_modes)


def serialize_mode_config(config: DispatchModeConfig) -> dict[str, str]:
    """Serialize config back to env-var-format dict (round-trip property).

    Returns dict: {"entry_alert": "dispatch", "breakout": "observe", ...}
    Always uses canonical form (never "enabled").
    """
    return dict(config.per_alert_modes)
