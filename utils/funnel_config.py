"""Funnel configuration loader.

Loads the premarket candidate funnel configuration from YAML and provides
sensible defaults when the file is missing or fields are absent.

See: design.md §Component 7, requirements.md §12.6, §7.2, §1.2, §1.3, §6.5
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Default config path relative to project root
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "funnel_config.yaml"

# ---------------------------------------------------------------------------
# Default values — used when YAML is missing or fields are absent
# ---------------------------------------------------------------------------

_DEFAULTS: dict = {
    "funnel": {
        "enabled": True,
        "schedule": {
            "discovery_time": "06:00",
            "research_time": "06:30",
            "analysis_time": "07:15",
            "confirmation_time": "09:35",
            "confirmation_end": "09:45",
        },
        "ceilings": {
            "max_discovery_shortlist": 5,
            "max_researcher_promoted": 3,
            "max_pm_handoff": 3,
        },
        "budgets": {
            "per_sector_seconds": 15,
            "total_pipeline_seconds": 90,
            "confirmation_budget_seconds": 45,
            "market_hours_confirmation_budget_seconds": 60,
        },
        "midday_scan": {
            "enabled": False,
            "revalidation_only": True,
        },
    }
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict.

    Values in *override* take precedence. Missing keys in *override*
    retain their *base* defaults.
    """
    merged = base.copy()
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_funnel_config(config_path: Path | None = None) -> dict:
    """Load the funnel configuration from YAML, falling back to defaults.

    Args:
        config_path: Optional override path to the config file.
            Defaults to ``config/funnel_config.yaml`` relative to the
            project root.

    Returns:
        Parsed configuration dictionary. The returned dict always contains
        the full ``funnel`` structure with all required sub-keys populated
        (either from the YAML file or from built-in defaults).

    Notes:
        Unlike :func:`load_sector_scout_config`, this function never raises
        on a missing or malformed file — it logs a warning and returns
        defaults. This ensures the funnel pipeline can always start with
        safe values even if the config file is absent or partially defined.
    """
    path = config_path or _CONFIG_PATH

    if not path.exists():
        logger.warning(
            "Funnel config file not found at %s — using built-in defaults.", path
        )
        return _DEFAULTS.copy()

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "Unable to read funnel config file at %s: %s — using defaults.", path, exc
        )
        return _DEFAULTS.copy()

    try:
        config = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        logger.warning(
            "Funnel config file is not valid YAML (%s) — using defaults.", exc
        )
        return _DEFAULTS.copy()

    if not isinstance(config, dict):
        logger.warning(
            "Funnel config must be a YAML mapping, got %s — using defaults.",
            type(config).__name__,
        )
        return _DEFAULTS.copy()

    # Deep-merge file values over defaults so missing fields get safe values
    merged = _deep_merge(_DEFAULTS, config)

    logger.info("Funnel config loaded from %s", path)
    return merged
