"""Unit tests for triggered trade plan feature flags in utils/gate_config.py.

Validates flag names, defaults, env-var parsing, and invalid-input handling.

Requirements: 0.1-0.12
"""

from __future__ import annotations

import importlib

import pytest
from unittest.mock import patch

import utils.gate_config as gc


# Env vars owned by this feature — cleared so code defaults are observable.
_FEATURE_VARS = [
    "TRIGGERED_PLAN_MODE",
    "PLAN_MONITOR_INTERVAL_SECONDS",
    "PLAN_DEFAULT_EXPIRATION_MINUTES",
    "PLAN_ENTRY_ZONE_TOLERANCE_PCT",
    "PLAN_TRIGGER_CONFIRMATION_TICKS",
    "PLAN_EXECUTION_MAX_QUOTE_AGE_SECONDS",
    "PLAN_TRIGGER_QUOTE_MAX_AGE_SECONDS",
    "QUOTE_PROVIDER_MIN_SECONDS_PER_SYMBOL",
    "QUOTE_PROVIDER_MAX_CALLS_PER_MINUTE",
]


@pytest.fixture
def reloaded():
    """Reload gate_config with feature env vars removed, restore afterwards."""
    import os

    saved = {k: os.environ[k] for k in _FEATURE_VARS if k in os.environ}
    for key in _FEATURE_VARS:
        os.environ.pop(key, None)
    module = importlib.reload(gc)
    try:
        yield module
    finally:
        os.environ.update(saved)
        importlib.reload(gc)


class TestDefaults:
    """Code defaults must match the spec when no env vars are set."""

    def test_mode_defaults_to_disabled(self, reloaded):
        assert reloaded.TRIGGERED_PLAN_MODE == "disabled"

    def test_monitor_interval_default(self, reloaded):
        assert reloaded.PLAN_MONITOR_INTERVAL_SECONDS == 30

    def test_expiration_default(self, reloaded):
        assert reloaded.PLAN_DEFAULT_EXPIRATION_MINUTES == 60

    def test_zone_tolerance_default(self, reloaded):
        assert reloaded.PLAN_ENTRY_ZONE_TOLERANCE_PCT == pytest.approx(0.005)

    def test_confirmation_ticks_default(self, reloaded):
        assert reloaded.PLAN_TRIGGER_CONFIRMATION_TICKS == 2

    def test_execution_max_quote_age_default(self, reloaded):
        assert reloaded.PLAN_EXECUTION_MAX_QUOTE_AGE_SECONDS == 5

    def test_trigger_quote_max_age_default(self, reloaded):
        assert reloaded.PLAN_TRIGGER_QUOTE_MAX_AGE_SECONDS == 30

    def test_min_seconds_per_symbol_default(self, reloaded):
        assert reloaded.QUOTE_PROVIDER_MIN_SECONDS_PER_SYMBOL == 30

    def test_max_calls_per_minute_default(self, reloaded):
        assert reloaded.QUOTE_PROVIDER_MAX_CALLS_PER_MINUTE == 40

    def test_types(self, reloaded):
        assert isinstance(reloaded.TRIGGERED_PLAN_MODE, str)
        assert isinstance(reloaded.PLAN_MONITOR_INTERVAL_SECONDS, int)
        assert isinstance(reloaded.PLAN_DEFAULT_EXPIRATION_MINUTES, int)
        assert isinstance(reloaded.PLAN_ENTRY_ZONE_TOLERANCE_PCT, float)
        assert isinstance(reloaded.PLAN_TRIGGER_CONFIRMATION_TICKS, int)
        assert isinstance(reloaded.PLAN_EXECUTION_MAX_QUOTE_AGE_SECONDS, int)
        assert isinstance(reloaded.PLAN_TRIGGER_QUOTE_MAX_AGE_SECONDS, int)
        assert isinstance(reloaded.QUOTE_PROVIDER_MIN_SECONDS_PER_SYMBOL, int)
        assert isinstance(reloaded.QUOTE_PROVIDER_MAX_CALLS_PER_MINUTE, int)


class TestEnvOverrides:
    """Values come from os.environ at import time."""

    @pytest.mark.parametrize("mode", ["disabled", "observe", "enabled"])
    def test_valid_modes_accepted(self, mode):
        with patch.dict("os.environ", {"TRIGGERED_PLAN_MODE": mode}):
            module = importlib.reload(gc)
            assert module.TRIGGERED_PLAN_MODE == mode
        importlib.reload(gc)

    def test_unrecognized_mode_falls_back_to_disabled(self, caplog):
        with patch.dict("os.environ", {"TRIGGERED_PLAN_MODE": "banana"}):
            with caplog.at_level("WARNING"):
                module = importlib.reload(gc)
            assert module.TRIGGERED_PLAN_MODE == "disabled"
            assert "TRIGGERED_PLAN_MODE" in caplog.text
        importlib.reload(gc)

    def test_numeric_overrides_applied(self):
        env = {
            "PLAN_MONITOR_INTERVAL_SECONDS": "15",
            "PLAN_DEFAULT_EXPIRATION_MINUTES": "120",
            "PLAN_ENTRY_ZONE_TOLERANCE_PCT": "0.01",
            "PLAN_TRIGGER_CONFIRMATION_TICKS": "3",
            "PLAN_EXECUTION_MAX_QUOTE_AGE_SECONDS": "10",
            "PLAN_TRIGGER_QUOTE_MAX_AGE_SECONDS": "45",
            "QUOTE_PROVIDER_MIN_SECONDS_PER_SYMBOL": "60",
            "QUOTE_PROVIDER_MAX_CALLS_PER_MINUTE": "20",
        }
        with patch.dict("os.environ", env):
            module = importlib.reload(gc)
            assert module.PLAN_MONITOR_INTERVAL_SECONDS == 15
            assert module.PLAN_DEFAULT_EXPIRATION_MINUTES == 120
            assert module.PLAN_ENTRY_ZONE_TOLERANCE_PCT == pytest.approx(0.01)
            assert module.PLAN_TRIGGER_CONFIRMATION_TICKS == 3
            assert module.PLAN_EXECUTION_MAX_QUOTE_AGE_SECONDS == 10
            assert module.PLAN_TRIGGER_QUOTE_MAX_AGE_SECONDS == 45
            assert module.QUOTE_PROVIDER_MIN_SECONDS_PER_SYMBOL == 60
            assert module.QUOTE_PROVIDER_MAX_CALLS_PER_MINUTE == 20
        importlib.reload(gc)

    def test_non_numeric_values_fall_back_to_defaults(self, caplog):
        env = {
            "PLAN_MONITOR_INTERVAL_SECONDS": "abc",
            "PLAN_ENTRY_ZONE_TOLERANCE_PCT": "not-a-float",
        }
        with patch.dict("os.environ", env):
            with caplog.at_level("WARNING"):
                module = importlib.reload(gc)
            assert module.PLAN_MONITOR_INTERVAL_SECONDS == 30
            assert module.PLAN_ENTRY_ZONE_TOLERANCE_PCT == pytest.approx(0.005)
        importlib.reload(gc)

    def test_below_minimum_values_are_clamped(self):
        env = {
            "PLAN_MONITOR_INTERVAL_SECONDS": "0",
            "PLAN_ENTRY_ZONE_TOLERANCE_PCT": "-0.01",
            "PLAN_TRIGGER_CONFIRMATION_TICKS": "0",
        }
        with patch.dict("os.environ", env):
            module = importlib.reload(gc)
            assert module.PLAN_MONITOR_INTERVAL_SECONDS == 1
            assert module.PLAN_ENTRY_ZONE_TOLERANCE_PCT == pytest.approx(0.0)
            assert module.PLAN_TRIGGER_CONFIRMATION_TICKS == 1
        importlib.reload(gc)
