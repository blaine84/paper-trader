"""Unit tests for core/replay/policy_version.py.

Validates: Requirements 4.1, 4.2, 4.4
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from unittest.mock import patch

import pytest

from core.replay.policy_version import (
    PolicyVersion,
    build_current_policy_version,
    build_policy_from_snapshot,
    validate_candidate_policy,
    _collect_feature_flags,
    _compute_config_digest,
    _normalize_for_hash,
)


class TestPolicyVersionDataclass:
    """PolicyVersion frozen dataclass behavior."""

    def test_frozen_immutability(self):
        """PolicyVersion fields cannot be reassigned after creation."""
        pv = PolicyVersion(
            name="test",
            gate_revision="abc123",
            config_digest="sha256digest",
            feature_flags={"FLAG_A": True},
            benchmark_version="2024-01-01",
            config_source_timestamp=datetime(2024, 1, 15, 8, 0, 0),
            gate_ordering_version="v1.0",
            adapter_version="1.0.0",
        )
        with pytest.raises(AttributeError):
            pv.name = "modified"

    def test_all_fields_accessible(self):
        """All declared fields are present and accessible."""
        ts = datetime(2024, 3, 10, 12, 0, 0)
        pv = PolicyVersion(
            name="current_v1.2",
            gate_revision="def456",
            config_digest="abcdef123456",
            feature_flags={"SETUP_SPECIFIC_RR_THRESHOLDS": True, "MODERATE_NEAR_MISS_PILOT": False},
            benchmark_version="2024-01-15",
            config_source_timestamp=ts,
            gate_ordering_version="v1.0",
            adapter_version="1.0.0",
        )
        assert pv.name == "current_v1.2"
        assert pv.gate_revision == "def456"
        assert pv.config_digest == "abcdef123456"
        assert pv.feature_flags == {"SETUP_SPECIFIC_RR_THRESHOLDS": True, "MODERATE_NEAR_MISS_PILOT": False}
        assert pv.benchmark_version == "2024-01-15"
        assert pv.config_source_timestamp == ts
        assert pv.gate_ordering_version == "v1.0"
        assert pv.adapter_version == "1.0.0"

    def test_nullable_fields(self):
        """benchmark_version and config_source_timestamp can be None."""
        pv = PolicyVersion(
            name="historical",
            gate_revision="unknown",
            config_digest="digest",
            feature_flags={},
            benchmark_version=None,
            config_source_timestamp=None,
            gate_ordering_version="v1.0",
            adapter_version="1.0.0",
        )
        assert pv.benchmark_version is None
        assert pv.config_source_timestamp is None

    def test_equality(self):
        """Two PolicyVersions with same fields are equal."""
        kwargs = dict(
            name="test",
            gate_revision="abc",
            config_digest="digest",
            feature_flags={"A": True},
            benchmark_version=None,
            config_source_timestamp=None,
            gate_ordering_version="v1.0",
            adapter_version="1.0.0",
        )
        assert PolicyVersion(**kwargs) == PolicyVersion(**kwargs)


class TestBuildCurrentPolicyVersion:
    """build_current_policy_version() reads from deployed gate_config.py."""

    @patch("core.replay.policy_version._resolve_git_revision", return_value="abc123def")
    def test_returns_policy_version(self, _mock_git, monkeypatch):
        """Returns a valid PolicyVersion with expected structure."""
        monkeypatch.setenv("SETUP_SPECIFIC_RR_THRESHOLDS", "true")
        monkeypatch.setenv("MODERATE_NEAR_MISS_PILOT", "false")

        pv = build_current_policy_version()

        assert isinstance(pv, PolicyVersion)
        assert pv.name == "current"
        assert pv.gate_revision == "abc123def"
        assert pv.gate_ordering_version == "v1.0"
        assert pv.adapter_version == "1.0.0"
        assert isinstance(pv.config_digest, str)
        assert len(pv.config_digest) == 64  # SHA-256 hex

    @patch("core.replay.policy_version._resolve_git_revision", return_value="abc123def")
    def test_feature_flags_captured(self, _mock_git, monkeypatch):
        """Feature flags are captured from environment."""
        monkeypatch.setenv("SETUP_SPECIFIC_RR_THRESHOLDS", "true")
        monkeypatch.setenv("MODERATE_NEAR_MISS_PILOT", "true")
        monkeypatch.setenv("PM_CANDIDATE_MODE", "enabled")
        monkeypatch.setenv("PM_BENCHMARK_CONTEXT_ENABLED", "true")

        pv = build_current_policy_version()

        assert pv.feature_flags["SETUP_SPECIFIC_RR_THRESHOLDS"] is True
        assert pv.feature_flags["MODERATE_NEAR_MISS_PILOT"] is True
        assert pv.feature_flags["PM_CANDIDATE_MODE"] is True
        assert pv.feature_flags["PM_BENCHMARK_CONTEXT_ENABLED"] is True

    @patch("core.replay.policy_version._resolve_git_revision", return_value="abc123def")
    def test_config_digest_is_stable(self, _mock_git, monkeypatch):
        """Same configuration produces same digest."""
        monkeypatch.setenv("SETUP_SPECIFIC_RR_THRESHOLDS", "false")
        monkeypatch.setenv("MODERATE_NEAR_MISS_PILOT", "false")
        monkeypatch.setenv("PM_CANDIDATE_MODE", "disabled")
        monkeypatch.setenv("PM_BENCHMARK_CONTEXT_ENABLED", "false")

        pv1 = build_current_policy_version()
        pv2 = build_current_policy_version()

        assert pv1.config_digest == pv2.config_digest

    @patch("core.replay.policy_version._resolve_git_revision", return_value="unknown")
    def test_unknown_git_revision(self, _mock_git, monkeypatch):
        """Returns 'unknown' when git is not available."""
        monkeypatch.setenv("SETUP_SPECIFIC_RR_THRESHOLDS", "false")
        monkeypatch.setenv("MODERATE_NEAR_MISS_PILOT", "false")

        pv = build_current_policy_version()
        assert pv.gate_revision == "unknown"


class TestBuildPolicyFromSnapshot:
    """build_policy_from_snapshot() reconstructs from stored Decision_Snapshot."""

    def test_reconstructs_from_valid_snapshot(self):
        """Reconstructs PolicyVersion from a complete snapshot dict."""
        snapshot = {
            "policy_version_id": "current_v1.0",
            "gate_config": json.dumps({
                "gate_revision": "abc123",
                "gate_ordering_version": "v1.0",
                "adapter_version": "1.0.0",
                "benchmark_version": "2024-01-15",
                "config_source_timestamp": "2024-01-15T08:00:00",
                "name": "snapshot_policy",
                "min_win_rate_by_setup": {"news_breakout": 0.40},
            }),
            "feature_flags": json.dumps({
                "SETUP_SPECIFIC_RR_THRESHOLDS": True,
                "MODERATE_NEAR_MISS_PILOT": False,
            }),
        }

        pv = build_policy_from_snapshot(snapshot)

        assert pv is not None
        assert pv.name == "snapshot_policy"
        assert pv.gate_revision == "abc123"
        assert pv.gate_ordering_version == "v1.0"
        assert pv.adapter_version == "1.0.0"
        assert pv.benchmark_version == "2024-01-15"
        assert pv.config_source_timestamp == datetime(2024, 1, 15, 8, 0, 0)
        assert pv.feature_flags == {
            "SETUP_SPECIFIC_RR_THRESHOLDS": True,
            "MODERATE_NEAR_MISS_PILOT": False,
        }

    def test_returns_none_when_no_policy_version_id(self):
        """Returns None when policy_version_id is missing."""
        snapshot = {"gate_config": "{}", "feature_flags": "{}"}
        assert build_policy_from_snapshot(snapshot) is None

    def test_handles_gate_config_as_dict(self):
        """gate_config can be a dict (not just JSON string)."""
        snapshot = {
            "policy_version_id": "v1",
            "gate_config": {
                "gate_revision": "def456",
                "name": "dict_policy",
            },
            "feature_flags": {"FLAG_A": True},
        }

        pv = build_policy_from_snapshot(snapshot)
        assert pv is not None
        assert pv.gate_revision == "def456"
        assert pv.name == "dict_policy"

    def test_handles_missing_gate_config(self):
        """Returns PolicyVersion with defaults when gate_config is None."""
        snapshot = {
            "policy_version_id": "v1",
            "gate_config": None,
            "feature_flags": None,
        }

        pv = build_policy_from_snapshot(snapshot)
        assert pv is not None
        assert pv.gate_revision == "unknown"
        assert pv.feature_flags == {}

    def test_handles_malformed_json(self):
        """Handles malformed JSON in gate_config gracefully."""
        snapshot = {
            "policy_version_id": "v1",
            "gate_config": "not valid json {{{",
            "feature_flags": "also broken",
        }

        pv = build_policy_from_snapshot(snapshot)
        assert pv is not None
        assert pv.gate_revision == "unknown"
        assert pv.feature_flags == {}

    def test_feature_flags_coerced_to_bool(self):
        """Feature flag values are coerced to boolean."""
        snapshot = {
            "policy_version_id": "v1",
            "gate_config": "{}",
            "feature_flags": json.dumps({"A": 1, "B": 0, "C": "true"}),
        }

        pv = build_policy_from_snapshot(snapshot)
        assert pv.feature_flags == {"A": True, "B": False, "C": True}


class TestValidateCandidatePolicy:
    """validate_candidate_policy() checks completeness for deterministic replay."""

    def _valid_policy(self) -> dict:
        """Construct a minimally valid candidate policy spec."""
        return {
            "name": "candidate_test",
            "gate_revision": "abc123",
            "feature_flags": {
                "SETUP_SPECIFIC_RR_THRESHOLDS": True,
                "MODERATE_NEAR_MISS_PILOT": False,
            },
            "gate_ordering_version": "v1.0",
            "adapter_version": "1.0.0",
            "thresholds": {
                "min_win_rate_by_setup": {"news_breakout": 0.40},
                "default_min_win_rate": 0.40,
                "stop_distance_rules": {"default": {}},
                "default_stop_distance_rule": {"min_pct": 0.012},
                "override_min_confidence_score": 8.0,
                "reduced_rr_thresholds_by_profile": {"aggressive": 0.5},
                "qualifying_min_signal_strength": 7.5,
                "qualifying_setup_types": ["news_breakout"],
            },
        }

    def test_valid_policy_passes(self):
        """Complete policy with all fields passes validation."""
        is_valid, missing = validate_candidate_policy(self._valid_policy())
        assert is_valid is True
        assert missing == []

    def test_missing_name_fails(self):
        """Policy without name is rejected."""
        policy = self._valid_policy()
        del policy["name"]
        is_valid, missing = validate_candidate_policy(policy)
        assert is_valid is False
        assert "name" in missing

    def test_empty_name_fails(self):
        """Policy with empty string name is rejected."""
        policy = self._valid_policy()
        policy["name"] = "   "
        is_valid, missing = validate_candidate_policy(policy)
        assert is_valid is False

    def test_missing_gate_revision_fails(self):
        """Policy without gate_revision is rejected."""
        policy = self._valid_policy()
        del policy["gate_revision"]
        is_valid, missing = validate_candidate_policy(policy)
        assert is_valid is False
        assert "gate_revision" in missing

    def test_missing_feature_flags_fails(self):
        """Policy without feature_flags is rejected."""
        policy = self._valid_policy()
        del policy["feature_flags"]
        is_valid, missing = validate_candidate_policy(policy)
        assert is_valid is False
        assert "feature_flags" in missing

    def test_non_boolean_feature_flag_fails(self):
        """Feature flags with non-boolean values are flagged."""
        policy = self._valid_policy()
        policy["feature_flags"]["BAD_FLAG"] = "yes"
        is_valid, missing = validate_candidate_policy(policy)
        assert is_valid is False
        assert any("BAD_FLAG" in f for f in missing)

    def test_missing_threshold_fields_fails(self):
        """Policy without required threshold fields is rejected."""
        policy = self._valid_policy()
        del policy["thresholds"]["min_win_rate_by_setup"]
        del policy["thresholds"]["stop_distance_rules"]
        is_valid, missing = validate_candidate_policy(policy)
        assert is_valid is False
        assert any("min_win_rate_by_setup" in f for f in missing)
        assert any("stop_distance_rules" in f for f in missing)

    def test_missing_gate_ordering_version_fails(self):
        """Policy without gate_ordering_version is rejected."""
        policy = self._valid_policy()
        del policy["gate_ordering_version"]
        is_valid, missing = validate_candidate_policy(policy)
        assert is_valid is False
        assert "gate_ordering_version" in missing

    def test_missing_adapter_version_fails(self):
        """Policy without adapter_version is rejected."""
        policy = self._valid_policy()
        del policy["adapter_version"]
        is_valid, missing = validate_candidate_policy(policy)
        assert is_valid is False
        assert "adapter_version" in missing

    def test_thresholds_at_top_level_also_works(self):
        """Threshold fields can be at top level (not nested in 'thresholds' key)."""
        policy = {
            "name": "flat_policy",
            "gate_revision": "abc123",
            "feature_flags": {"A": True},
            "gate_ordering_version": "v1.0",
            "adapter_version": "1.0.0",
            "min_win_rate_by_setup": {"news_breakout": 0.40},
            "default_min_win_rate": 0.40,
            "stop_distance_rules": {"default": {}},
            "default_stop_distance_rule": {"min_pct": 0.012},
            "override_min_confidence_score": 8.0,
            "reduced_rr_thresholds_by_profile": {"aggressive": 0.5},
            "qualifying_min_signal_strength": 7.5,
            "qualifying_setup_types": ["news_breakout"],
        }
        is_valid, missing = validate_candidate_policy(policy)
        assert is_valid is True
        assert missing == []


class TestConfigDigest:
    """Config digest computation is stable and deterministic."""

    def test_same_input_same_digest(self):
        """Identical config produces identical digest."""
        config = {"a": 1, "b": [2, 3], "c": {"d": 4.0}}
        assert _compute_config_digest(config) == _compute_config_digest(config)

    def test_key_order_irrelevant(self):
        """Key ordering does not affect digest."""
        config_a = {"z": 1, "a": 2}
        config_b = {"a": 2, "z": 1}
        assert _compute_config_digest(config_a) == _compute_config_digest(config_b)

    def test_different_values_different_digest(self):
        """Different values produce different digest."""
        config_a = {"threshold": 0.5}
        config_b = {"threshold": 0.6}
        assert _compute_config_digest(config_a) != _compute_config_digest(config_b)

    def test_handles_sets(self):
        """Sets are normalized to sorted lists for hashing."""
        config = {"symbols": {"TSLA", "AMD", "NVDA"}}
        digest = _compute_config_digest(config)
        assert isinstance(digest, str)
        assert len(digest) == 64

    def test_handles_nested_dicts(self):
        """Deeply nested structures are handled."""
        config = {"level1": {"level2": {"level3": [1, 2, 3]}}}
        digest = _compute_config_digest(config)
        assert len(digest) == 64


class TestNormalizeForHash:
    """_normalize_for_hash correctly canonicalizes complex types."""

    def test_dict_sorted(self):
        """Dicts are sorted by key."""
        result = _normalize_for_hash({"b": 1, "a": 2})
        assert list(result.keys()) == ["a", "b"]

    def test_set_to_sorted_list(self):
        """Sets become sorted lists."""
        result = _normalize_for_hash({"s": {"c", "a", "b"}})
        assert result["s"] == ["a", "b", "c"]

    def test_datetime_to_iso(self):
        """Datetimes become ISO-8601 strings."""
        dt = datetime(2024, 1, 15, 8, 0, 0)
        result = _normalize_for_hash({"ts": dt})
        assert result["ts"] == "2024-01-15T08:00:00"

    def test_float_normalized(self):
        """Floats are normalized to 10-significant-figure strings."""
        result = _normalize_for_hash({"f": 0.1})
        assert result["f"] == "0.1"

    def test_none_preserved(self):
        """None values are preserved."""
        result = _normalize_for_hash({"x": None})
        assert result["x"] is None


class TestCollectFeatureFlags:
    """_collect_feature_flags() reads from environment."""

    def test_all_disabled(self, monkeypatch):
        """All flags disabled when env vars are unset."""
        monkeypatch.delenv("SETUP_SPECIFIC_RR_THRESHOLDS", raising=False)
        monkeypatch.delenv("MODERATE_NEAR_MISS_PILOT", raising=False)
        monkeypatch.setenv("PM_CANDIDATE_MODE", "disabled")
        monkeypatch.setenv("PM_BENCHMARK_CONTEXT_ENABLED", "false")

        flags = _collect_feature_flags()

        assert flags["SETUP_SPECIFIC_RR_THRESHOLDS"] is False
        assert flags["MODERATE_NEAR_MISS_PILOT"] is False
        assert flags["PM_CANDIDATE_MODE"] is False
        assert flags["PM_BENCHMARK_CONTEXT_ENABLED"] is False

    def test_all_enabled(self, monkeypatch):
        """All flags enabled when env vars are set."""
        monkeypatch.setenv("SETUP_SPECIFIC_RR_THRESHOLDS", "true")
        monkeypatch.setenv("MODERATE_NEAR_MISS_PILOT", "TRUE")
        monkeypatch.setenv("PM_CANDIDATE_MODE", "enabled")
        monkeypatch.setenv("PM_BENCHMARK_CONTEXT_ENABLED", "True")

        flags = _collect_feature_flags()

        assert flags["SETUP_SPECIFIC_RR_THRESHOLDS"] is True
        assert flags["MODERATE_NEAR_MISS_PILOT"] is True
        assert flags["PM_CANDIDATE_MODE"] is True
        assert flags["PM_BENCHMARK_CONTEXT_ENABLED"] is True
