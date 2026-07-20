"""Queue configuration loaded from environment variables with safe defaults.

All dispatcher behavior is configured through QueueConfig, a frozen dataclass
constructed via the from_environment() classmethod. Missing or invalid env vars
always produce conservative defaults (concurrency 1, strict deadlines).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid modes for the dispatcher
# ---------------------------------------------------------------------------
_VALID_MODES = frozenset({"disabled", "observe", "enforcing"})

# ---------------------------------------------------------------------------
# Request classes recognized by the system
# ---------------------------------------------------------------------------
REQUEST_CLASSES = (
    "execution_critical",
    "market_analysis",
    "position_management",
    "repair",
    "research",
    "review",
    "advisory",
    "startup_probe",
)

# ---------------------------------------------------------------------------
# Default per-class timing values (seconds)
# ---------------------------------------------------------------------------
_DEFAULT_DEADLINES: dict[str, float] = {
    "execution_critical": 120.0,
    "market_analysis": 180.0,
    "position_management": 180.0,
    "repair": 60.0,
    "research": 600.0,
    "review": 600.0,
    "advisory": 900.0,
    "startup_probe": 30.0,
}

_DEFAULT_MAX_QUEUE_WAITS: dict[str, float] = {
    "execution_critical": 30.0,
    "market_analysis": 60.0,
    "position_management": 60.0,
    "repair": 15.0,
    "research": 180.0,
    "review": 180.0,
    "advisory": 300.0,
    "startup_probe": 10.0,
}

_DEFAULT_STALE_AFTER: dict[str, float] = {
    "execution_critical": 60.0,
    "market_analysis": 120.0,
    "position_management": 120.0,
    "repair": 30.0,
    "research": 300.0,
    "review": 300.0,
    "advisory": 600.0,
    "startup_probe": 15.0,
}


def _read_int(env_var: str, default: int) -> int:
    """Read an integer from an env var, returning default on missing/invalid."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        if value < 1:
            logger.warning(
                "Environment variable %s has non-positive value %r, using default %d",
                env_var, raw, default,
            )
            return default
        return value
    except ValueError:
        logger.warning(
            "Environment variable %s has invalid integer value %r, using default %d",
            env_var, raw, default,
        )
        return default


def _read_float(env_var: str, default: float) -> float:
    """Read a float from an env var, returning default on missing/invalid."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
        if value <= 0:
            logger.warning(
                "Environment variable %s has non-positive value %r, using default %s",
                env_var, raw, default,
            )
            return default
        return value
    except ValueError:
        logger.warning(
            "Environment variable %s has invalid float value %r, using default %s",
            env_var, raw, default,
        )
        return default


def _read_json_dict(env_var: str, default: dict) -> dict:
    """Read a JSON object from an env var, returning default on missing/invalid."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return dict(default)
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            logger.warning(
                "Environment variable %s is not a JSON object, using default",
                env_var,
            )
            return dict(default)
        return parsed
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "Environment variable %s has invalid JSON %r, using default",
            env_var, raw,
        )
        return dict(default)


def _read_per_class_float(env_prefix: str, defaults: dict[str, float]) -> dict[str, float]:
    """Read per-class float overrides from env vars like LLM_QUEUE_DEADLINE_EXECUTION_CRITICAL.

    Falls back to the provided defaults for any class without a valid override.
    """
    result: dict[str, float] = dict(defaults)
    for request_class in REQUEST_CLASSES:
        env_var = f"{env_prefix}{request_class.upper()}"
        raw = os.environ.get(env_var, "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
            if value <= 0:
                logger.warning(
                    "Environment variable %s has non-positive value %r, "
                    "using default %s for class %s",
                    env_var, raw, defaults.get(request_class), request_class,
                )
                continue
            result[request_class] = value
        except ValueError:
            logger.warning(
                "Environment variable %s has invalid float value %r, "
                "using default %s for class %s",
                env_var, raw, defaults.get(request_class), request_class,
            )
    return result


def _read_approved_fallback_models() -> dict[str, list[str]]:
    """Read approved fallback models from per-class env vars.

    Env var pattern: LLM_QUEUE_FALLBACK_MODELS_{CLASS} with comma-separated
    model names, e.g. LLM_QUEUE_FALLBACK_MODELS_EXECUTION_CRITICAL=claude-3-haiku,gpt-4o-mini
    """
    result: dict[str, list[str]] = {}
    for request_class in REQUEST_CLASSES:
        env_var = f"LLM_QUEUE_FALLBACK_MODELS_{request_class.upper()}"
        raw = os.environ.get(env_var, "").strip()
        if not raw:
            result[request_class] = []
            continue
        models = [m.strip() for m in raw.split(",") if m.strip()]
        result[request_class] = models
    return result


def _read_max_queue_size_by_class() -> dict[str, int]:
    """Read per-class max queue size from env vars.

    Env var pattern: LLM_QUEUE_MAX_SIZE_{CLASS}, e.g. LLM_QUEUE_MAX_SIZE_ADVISORY=3
    """
    result: dict[str, int] = {}
    for request_class in REQUEST_CLASSES:
        env_var = f"LLM_QUEUE_MAX_SIZE_{request_class.upper()}"
        raw = os.environ.get(env_var, "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
            if value < 1:
                logger.warning(
                    "Environment variable %s has non-positive value %r, skipping",
                    env_var, raw,
                )
                continue
            result[request_class] = value
        except ValueError:
            logger.warning(
                "Environment variable %s has invalid integer value %r, skipping",
                env_var, raw,
            )
    return result


def _validate_model_concurrency(raw_dict: dict) -> dict[str, int]:
    """Validate and coerce a parsed JSON dict into model_name → int mapping."""
    result: dict[str, int] = {}
    for key, value in raw_dict.items():
        if not isinstance(key, str):
            continue
        try:
            int_value = int(value)
            if int_value < 1:
                logger.warning(
                    "Per-model concurrency for %r has non-positive value %r, skipping",
                    key, value,
                )
                continue
            result[key] = int_value
        except (ValueError, TypeError):
            logger.warning(
                "Per-model concurrency for %r has invalid value %r, skipping",
                key, value,
            )
    return result


@dataclass(frozen=True)
class QueueConfig:
    """All dispatcher configuration, read from environment with safe defaults.

    Constructed via from_environment() classmethod. All fields are immutable.
    Missing or invalid environment variables always produce conservative defaults.
    """

    mode: str = "disabled"
    global_concurrency: int = 1
    per_model_concurrency: dict[str, int] = field(default_factory=dict)
    max_queue_size: int = 10
    max_queue_size_by_class: dict[str, int] = field(default_factory=dict)

    # Per-class timing defaults (seconds)
    deadlines: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_DEADLINES))
    max_queue_waits: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_MAX_QUEUE_WAITS))
    stale_after: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_STALE_AFTER))

    # Market-hour adjustments
    market_hour_deadline_factor: float = 0.5
    market_hour_queue_wait_factor: float = 0.5

    # Prompt size thresholds
    prompt_token_warn_threshold: int = 8000
    prompt_token_reject_threshold: int = 16000

    # Fallback configuration
    approved_fallback_models: dict[str, list[str]] = field(default_factory=dict)
    fallback_deadline_buffer_seconds: float = 15.0

    @classmethod
    def from_environment(cls) -> QueueConfig:
        """Construct QueueConfig from environment variables.

        Missing or invalid values always produce conservative defaults:
        - concurrency 1
        - strict (short) deadlines
        - mode "disabled"
        """
        # Mode validation
        raw_mode = os.environ.get("LLM_QUEUE_MODE", "disabled").strip().lower()
        if raw_mode not in _VALID_MODES:
            logger.warning(
                "LLM_QUEUE_MODE has invalid value %r, defaulting to 'disabled'",
                raw_mode,
            )
            raw_mode = "disabled"

        # Global scalars
        global_concurrency = _read_int("LLM_QUEUE_GLOBAL_CONCURRENCY", 1)
        max_queue_size = _read_int("LLM_QUEUE_MAX_SIZE", 10)

        # Market-hour factors
        market_hour_deadline_factor = _read_float(
            "LLM_QUEUE_MARKET_HOUR_FACTOR", 0.5
        )
        # Use same env var for both deadline and queue wait factors by default;
        # separate var available for override
        market_hour_queue_wait_factor = _read_float(
            "LLM_QUEUE_MARKET_HOUR_WAIT_FACTOR",
            market_hour_deadline_factor,
        )

        # Prompt size thresholds
        prompt_token_warn_threshold = _read_int("LLM_QUEUE_PROMPT_WARN_TOKENS", 8000)
        prompt_token_reject_threshold = _read_int("LLM_QUEUE_PROMPT_REJECT_TOKENS", 16000)

        # Fallback buffer
        fallback_deadline_buffer_seconds = _read_float(
            "LLM_QUEUE_FALLBACK_BUFFER_S", 15.0
        )

        # Per-model concurrency (JSON object)
        raw_model_concurrency = _read_json_dict("LLM_QUEUE_MODEL_CONCURRENCY", {})
        per_model_concurrency = _validate_model_concurrency(raw_model_concurrency)

        # Per-class timing overrides
        deadlines = _read_per_class_float("LLM_QUEUE_DEADLINE_", _DEFAULT_DEADLINES)
        max_queue_waits = _read_per_class_float("LLM_QUEUE_MAX_WAIT_", _DEFAULT_MAX_QUEUE_WAITS)
        stale_after = _read_per_class_float("LLM_QUEUE_STALE_", _DEFAULT_STALE_AFTER)

        # Per-class queue size limits
        max_queue_size_by_class = _read_max_queue_size_by_class()

        # Approved fallback models
        approved_fallback_models = _read_approved_fallback_models()

        return cls(
            mode=raw_mode,
            global_concurrency=global_concurrency,
            per_model_concurrency=per_model_concurrency,
            max_queue_size=max_queue_size,
            max_queue_size_by_class=max_queue_size_by_class,
            deadlines=deadlines,
            max_queue_waits=max_queue_waits,
            stale_after=stale_after,
            market_hour_deadline_factor=market_hour_deadline_factor,
            market_hour_queue_wait_factor=market_hour_queue_wait_factor,
            prompt_token_warn_threshold=prompt_token_warn_threshold,
            prompt_token_reject_threshold=prompt_token_reject_threshold,
            approved_fallback_models=approved_fallback_models,
            fallback_deadline_buffer_seconds=fallback_deadline_buffer_seconds,
        )
