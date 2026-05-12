"""
Tests for utils.catalyst_specificity — load_catalyst_config().

Covers task 1.2 requirements: 13.1, 13.2, 13.3, 13.4, 13.5.
"""

import json
import logging
import os
import tempfile

import pytest

from utils.catalyst_specificity import (
    _HARDCODED_DEFAULTS,
    load_catalyst_config,
)
import utils.catalyst_specificity as catalyst_mod


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset the module-level config cache before each test."""
    catalyst_mod._CONFIG_CACHE = None
    yield
    catalyst_mod._CONFIG_CACHE = None


@pytest.fixture
def valid_config_file(tmp_path):
    """Create a valid config JSON file in a temp directory."""
    config = {
        "symbol_aliases": {"AAPL": ["AAPL", "Apple"]},
        "readthrough_relationships": {"AAPL": ["Foxconn", "TSMC"]},
    }
    path = tmp_path / "catalyst_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return str(path)


@pytest.fixture
def malformed_config_file(tmp_path):
    """Create a malformed JSON file."""
    path = tmp_path / "catalyst_config.json"
    path.write_text("{ not valid json !!!", encoding="utf-8")
    return str(path)


@pytest.fixture
def invalid_structure_config_file(tmp_path):
    """Create a JSON file with wrong structure (missing expected keys)."""
    config = {"foo": "bar"}
    path = tmp_path / "catalyst_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return str(path)


# ===================================================================
# Tests: Loading from disk (Requirement 13.1)
# ===================================================================


class TestLoadFromDisk:
    """Verify config loads from JSON file on first call."""

    def test_loads_valid_config(self, valid_config_file):
        result = load_catalyst_config(config_path=valid_config_file)
        assert result["aliases"] == {"AAPL": ["AAPL", "Apple"]}
        assert result["relationships"] == {"AAPL": ["Foxconn", "TSMC"]}

    def test_loads_default_path_when_none(self):
        """When config_path is None, loads from the default co-located file."""
        result = load_catalyst_config()
        # Should load the actual catalyst_config.json
        assert "AMD" in result["aliases"]
        assert "NVDA" in result["relationships"]


# ===================================================================
# Tests: Caching behavior (Requirement 13.4)
# ===================================================================


class TestCaching:
    """Verify lazy loading and caching."""

    def test_subsequent_calls_return_cached(self, valid_config_file):
        first = load_catalyst_config(config_path=valid_config_file)
        # Mutate the returned dict to prove same object is returned
        first["_marker"] = True
        second = load_catalyst_config(config_path=valid_config_file)
        assert second.get("_marker") is True
        assert first is second

    def test_cache_avoids_disk_io(self, valid_config_file, tmp_path):
        """After first load, even if file is deleted, cache still works."""
        load_catalyst_config(config_path=valid_config_file)
        os.remove(valid_config_file)
        # Should still return cached result without error
        result = load_catalyst_config(config_path=valid_config_file)
        assert result["aliases"] == {"AAPL": ["AAPL", "Apple"]}


# ===================================================================
# Tests: force_reload (Requirement 13.5)
# ===================================================================


class TestForceReload:
    """Verify force_reload re-reads from disk."""

    def test_force_reload_updates_cache(self, tmp_path):
        # Initial config
        config_v1 = {
            "symbol_aliases": {"V1": ["Version1"]},
            "readthrough_relationships": {"V1": ["Related1"]},
        }
        path = tmp_path / "catalyst_config.json"
        path.write_text(json.dumps(config_v1), encoding="utf-8")

        result1 = load_catalyst_config(config_path=str(path))
        assert result1["aliases"] == {"V1": ["Version1"]}

        # Update config on disk
        config_v2 = {
            "symbol_aliases": {"V2": ["Version2"]},
            "readthrough_relationships": {"V2": ["Related2"]},
        }
        path.write_text(json.dumps(config_v2), encoding="utf-8")

        # Without force_reload, still returns v1
        result_cached = load_catalyst_config(config_path=str(path))
        assert result_cached["aliases"] == {"V1": ["Version1"]}

        # With force_reload, gets v2
        result2 = load_catalyst_config(config_path=str(path), force_reload=True)
        assert result2["aliases"] == {"V2": ["Version2"]}
        assert result2["relationships"] == {"V2": ["Related2"]}


# ===================================================================
# Tests: Fallback to defaults (Requirements 13.2, 13.3)
# ===================================================================


class TestFallbackDefaults:
    """Verify fallback to hardcoded defaults on error."""

    def test_missing_file_falls_back(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = load_catalyst_config(config_path="/nonexistent/path.json")
        assert result["aliases"] == _HARDCODED_DEFAULTS["symbol_aliases"]
        assert result["relationships"] == _HARDCODED_DEFAULTS["readthrough_relationships"]
        assert "Catalyst config load failed" in caplog.text

    def test_malformed_json_falls_back(self, malformed_config_file, caplog):
        with caplog.at_level(logging.WARNING):
            result = load_catalyst_config(config_path=malformed_config_file)
        assert result["aliases"] == _HARDCODED_DEFAULTS["symbol_aliases"]
        assert result["relationships"] == _HARDCODED_DEFAULTS["readthrough_relationships"]
        assert "Catalyst config load failed" in caplog.text

    def test_invalid_structure_falls_back(self, invalid_structure_config_file, caplog):
        with caplog.at_level(logging.WARNING):
            result = load_catalyst_config(config_path=invalid_structure_config_file)
        assert result["aliases"] == _HARDCODED_DEFAULTS["symbol_aliases"]
        assert result["relationships"] == _HARDCODED_DEFAULTS["readthrough_relationships"]
        assert "Catalyst config load failed" in caplog.text

    def test_fallback_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            load_catalyst_config(config_path="/does/not/exist.json")
        assert any("Catalyst config load failed" in r.message for r in caplog.records)
        assert any(r.levelno == logging.WARNING for r in caplog.records)
