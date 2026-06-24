from pathlib import Path


def test_empty_shadow_registry_falls_through_to_legacy_path():
    source = (
        Path(__file__).parents[1] / "agents" / "portfolio_manager.py"
    ).read_text(encoding="utf-8")

    assert 'registry.is_empty and PM_CANDIDATE_MODE == "enabled"' in source
    assert "if not registry.is_empty:" in source
    assert (
        "Candidate-ID shadow mode produced no eligible candidates"
        in source
    )


def test_shadow_mode_skips_legacy_entry_by_default():
    root = Path(__file__).parents[1]
    pm_source = (root / "agents" / "portfolio_manager.py").read_text(encoding="utf-8")
    gate_source = (root / "utils" / "gate_config.py").read_text(encoding="utf-8")

    assert "PM_SHADOW_RUN_LEGACY_ENTRY" in gate_source
    assert '"PM_SHADOW_RUN_LEGACY_ENTRY", "false"' in gate_source
    assert 'PM_CANDIDATE_MODE == "shadow" and not PM_SHADOW_RUN_LEGACY_ENTRY' in pm_source
    assert "legacy_skipped" in pm_source
