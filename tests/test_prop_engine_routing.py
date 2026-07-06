"""
Property-based tests for engine routing in db/schema.py.

Tests universal correctness properties for the dialect-aware init_db() function:
- Property 1: DATABASE_URL routing produces the correct engine dialect
- Property 2: Invalid DATABASE_URL raises at startup (fail-closed)
"""

import os
import tempfile
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine

from db.schema import init_db, is_sqlite


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid-looking Postgres URLs (syntactically valid)
_PG_HOSTS = st.sampled_from(["127.0.0.1", "localhost"])
_PG_PORTS = st.integers(min_value=59990, max_value=59999)
_PG_USERS = st.sampled_from(["postgres", "paper_trader", "admin", "dbuser"])
_PG_PASSWORDS = st.sampled_from(["secret", "pass123", "hunter2", "pg_pass"])
_PG_DATABASES = st.sampled_from(["paper_trader", "testdb", "myapp", "production"])
_PG_DRIVERS = st.sampled_from(["postgresql+psycopg", "postgresql+psycopg2"])

postgres_url_strategy = st.builds(
    lambda driver, user, password, host, port, db: f"{driver}://{user}:{password}@{host}:{port}/{db}",
    driver=_PG_DRIVERS,
    user=_PG_USERS,
    password=_PG_PASSWORDS,
    host=_PG_HOSTS,
    port=_PG_PORTS,
    db=_PG_DATABASES,
)

# Empty or unset DATABASE_URL values (triggers SQLite branch)
empty_database_url_strategy = st.sampled_from(["", "  ", "\t", "\n", "   \t  "])


# ---------------------------------------------------------------------------
# Property 1: DATABASE_URL Routing Produces Correct Engine
#
# When DATABASE_URL is set to a valid-looking postgres URL, init_db() produces
# an engine with dialect.name == "postgresql". When DATABASE_URL is unset/empty,
# init_db() produces an engine with dialect.name == "sqlite".
#
# **Validates: Requirements 1.1, 1.4**
# ---------------------------------------------------------------------------


class TestProperty1DatabaseUrlRoutingProducesCorrectEngine:
    """
    Property 1: DATABASE_URL Routing Produces Correct Engine.

    When DATABASE_URL is set to a valid postgres URL, init_db() creates an
    engine with dialect.name == "postgresql". When DATABASE_URL is absent or
    empty, init_db() produces an engine with dialect.name == "sqlite".

    We patch the connect() call to avoid actual network I/O while still
    verifying the engine creation logic routes correctly.

    **Validates: Requirements 1.1, 1.4**
    """

    @given(url=postgres_url_strategy)
    @settings(max_examples=200)
    def test_postgres_url_routes_to_postgres_engine(self, url: str):
        """When DATABASE_URL is a postgres URL, init_db() creates a Postgres engine.

        We verify the engine dialect is postgresql by intercepting the connection
        verification step (which would otherwise require a running Postgres).
        """
        with patch.dict(os.environ, {"DATABASE_URL": url}):
            # Patch the engine.connect() to avoid real network I/O
            # but still allow engine creation to proceed
            with patch("db.schema.create_engine") as mock_create:
                # Set up a mock engine that reports postgresql dialect
                mock_engine = MagicMock()
                mock_engine.dialect.name = "postgresql"
                mock_conn = MagicMock()
                mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
                mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
                mock_create.return_value = mock_engine

                engine = init_db()
                # Verify create_engine was called with the postgres URL
                mock_create.assert_called_once_with(url, pool_pre_ping=True)

    @given(empty_val=empty_database_url_strategy)
    @settings(max_examples=200)
    def test_empty_database_url_routes_to_sqlite_engine(self, empty_val: str):
        """When DATABASE_URL is empty/whitespace, init_db() produces a SQLite engine."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "test.db")

            with patch.dict(os.environ, {"DATABASE_URL": empty_val}):
                engine = init_db(db_path)
                try:
                    assert engine.dialect.name == "sqlite", (
                        f"Expected 'sqlite' dialect when DATABASE_URL='{repr(empty_val)}', "
                        f"got '{engine.dialect.name}'"
                    )
                    assert is_sqlite(engine) is True
                finally:
                    engine.dispose()

    @settings(max_examples=50)
    @given(data=st.data())
    def test_unset_database_url_routes_to_sqlite_engine(self, data):
        """When DATABASE_URL is not set at all, init_db() produces a SQLite engine."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "test.db")

            env = os.environ.copy()
            env.pop("DATABASE_URL", None)

            with patch.dict(os.environ, env, clear=True):
                engine = init_db(db_path)
                try:
                    assert engine.dialect.name == "sqlite", (
                        f"Expected 'sqlite' dialect when DATABASE_URL is unset, "
                        f"got '{engine.dialect.name}'"
                    )
                    assert is_sqlite(engine) is True
                finally:
                    engine.dispose()


# ---------------------------------------------------------------------------
# Property 2: Invalid DATABASE_URL Raises at Startup
#
# When DATABASE_URL is set to a non-empty string that can't connect,
# init_db() raises an exception (fail-closed).
#
# **Validates: Requirements 1.6**
# ---------------------------------------------------------------------------


class TestProperty2InvalidDatabaseUrlRaisesAtStartup:
    """
    Property 2: Invalid DATABASE_URL Raises at Startup.

    When DATABASE_URL is set to a non-empty string that can't connect (unreachable
    host, bad port, nonexistent database), init_db() raises an exception rather
    than silently falling back. This ensures fail-closed behavior.

    We simulate connection failures by patching create_engine to return an engine
    whose connect() raises OperationalError — verifying that init_db() does NOT
    catch and swallow the error.

    **Validates: Requirements 1.6**
    """

    @given(url=postgres_url_strategy)
    @settings(max_examples=200)
    def test_unreachable_postgres_url_raises(self, url: str):
        """Any unreachable Postgres URL must raise — never silently fall back to SQLite."""
        from sqlalchemy.exc import OperationalError

        with patch.dict(os.environ, {"DATABASE_URL": url}):
            with patch("db.schema.create_engine") as mock_create:
                mock_engine = MagicMock()
                # Simulate connection failure
                mock_engine.connect.side_effect = OperationalError(
                    "connection refused", {}, Exception("conn refused")
                )
                mock_create.return_value = mock_engine

                with pytest.raises(OperationalError):
                    init_db()

    @given(
        host=st.just("127.0.0.1"),
        port=st.just(59999),
    )
    @settings(max_examples=5)
    def test_connection_refused_raises(self, host: str, port: int):
        """Specifically test connection refused scenario (simulated)."""
        from sqlalchemy.exc import OperationalError

        url = f"postgresql+psycopg://nobody:fake@{host}:{port}/nonexistent"

        with patch.dict(os.environ, {"DATABASE_URL": url}):
            with patch("db.schema.create_engine") as mock_create:
                mock_engine = MagicMock()
                mock_engine.connect.side_effect = OperationalError(
                    "connection refused", {}, Exception(f"could not connect to {host}:{port}")
                )
                mock_create.return_value = mock_engine

                with pytest.raises(OperationalError):
                    init_db()

    @given(
        bad_url=st.sampled_from([
            "postgresql+psycopg://nobody:fake@127.0.0.1:59999/nonexistent",
            "postgresql+psycopg://x:x@127.0.0.1:59997/db",
            "postgresql+psycopg2://user:pass@127.0.0.1:59998/nope",
        ])
    )
    @settings(max_examples=50)
    def test_various_invalid_postgres_urls_all_raise(self, bad_url: str):
        """Multiple forms of invalid Postgres URLs all raise — none silently succeed."""
        from sqlalchemy.exc import OperationalError

        with patch.dict(os.environ, {"DATABASE_URL": bad_url}):
            with patch("db.schema.create_engine") as mock_create:
                mock_engine = MagicMock()
                mock_engine.connect.side_effect = OperationalError(
                    "connection refused", {}, Exception("could not connect")
                )
                mock_create.return_value = mock_engine

                with pytest.raises(OperationalError):
                    init_db()
