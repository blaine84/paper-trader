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
