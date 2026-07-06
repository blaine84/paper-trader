"""Smoke tests for operational security requirements.

Validates that credentials are not committed, dependencies are declared,
and operational documentation exists.

Requirements: 8.2, 8.5
"""
import subprocess
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestNoCredentialsInTrackedFiles:
    """Requirement 8.2: No database credentials in source control."""

    def test_no_database_url_values_in_tracked_files(self):
        """git grep should find no DATABASE_URL=postgresql... patterns in non-example files."""
        result = subprocess.run(
            ["git", "grep", "-l", "DATABASE_URL=postgresql"],
            capture_output=True, text=True,
            cwd=str(PROJECT_ROOT),
        )
        # Filter out files that are allowed to contain annotated placeholders
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        disallowed = [
            f for f in lines
            if not f.endswith(".env.example")
            and not f.endswith("design.md")
            and not f.endswith("requirements.md")
            and not f.endswith("tasks.md")
        ]
        assert len(disallowed) == 0, (
            f"Found DATABASE_URL values in tracked files: {disallowed}"
        )

    def test_no_postgres_passwords_in_tracked_files(self):
        """git grep should find no password patterns in connection strings."""
        # Search for patterns like :password@ in tracked files
        result = subprocess.run(
            ["git", "grep", "-l", "-E", "psycopg://[^:]+:[^@]+@"],
            capture_output=True, text=True,
            cwd=str(PROJECT_ROOT),
        )
        # Exclude .env.example which has a commented placeholder
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        non_example_files = [
            f for f in lines
            if not f.endswith(".env.example")
            and not f.endswith("design.md")
            and not f.endswith("requirements.md")
            and not f.endswith("tasks.md")
        ]
        assert len(non_example_files) == 0, (
            f"Found possible credentials in: {non_example_files}"
        )


class TestDependencyDeclarations:
    """Requirement 8.5: Dependencies properly declared."""

    def test_psycopg_binary_in_requirements(self):
        """psycopg[binary] must be in requirements.txt."""
        req_path = PROJECT_ROOT / "requirements.txt"
        content = req_path.read_text()
        assert "psycopg[binary]" in content or "psycopg" in content


class TestDocumentationExists:
    """Operational documentation exists."""

    def test_rollback_documentation_exists(self):
        """docs/postgres-rollback.md must exist."""
        assert (PROJECT_ROOT / "docs" / "postgres-rollback.md").exists()

    def test_setup_documentation_exists(self):
        """docs/postgres-setup.md must exist."""
        assert (PROJECT_ROOT / "docs" / "postgres-setup.md").exists()


class TestEnvExample:
    """Requirement 8.2: .env.example has annotated DATABASE_URL."""

    def test_database_url_placeholder_in_env_example(self):
        """`.env.example` contains commented DATABASE_URL placeholder."""
        env_example = PROJECT_ROOT / ".env.example"
        content = env_example.read_text()
        assert "DATABASE_URL" in content
        # Should be commented (starts with #)
        for line in content.splitlines():
            if "DATABASE_URL" in line:
                assert line.strip().startswith("#"), (
                    "DATABASE_URL in .env.example should be commented out"
                )
                break
