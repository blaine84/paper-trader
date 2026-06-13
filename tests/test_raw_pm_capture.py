"""Unit tests for utils/raw_pm_capture.py."""
import hashlib
import json
import os
import unittest
from unittest.mock import patch

from sqlalchemy import text

from utils.raw_pm_capture import (
    MAX_RAW_PAYLOAD_BYTES,
    RawPMResponse,
    ResponseLineageLink,
    _strip_credentials,
    _truncate_utf8,
    capture_raw_pm_response,
    link_response_to_lineages,
    persist_lineage_links,
    persist_raw_response,
)


class TestStripCredentials(unittest.TestCase):
    """Test credential stripping from payloads."""

    def test_strips_api_key(self):
        payload = '{"api_key": "sk-secret123", "data": "keep"}'
        result = _strip_credentials(payload)
        assert "sk-secret123" not in result
        assert "[REDACTED]" in result
        assert '"data": "keep"' in result

    def test_strips_bearer_token(self):
        payload = '{"bearer_token": "tok_abc", "reasoning": "valid logic"}'
        result = _strip_credentials(payload)
        assert "tok_abc" not in result
        assert "valid logic" in result

    def test_strips_all_credential_keys(self):
        payload = json.dumps({
            "api_key": "secret1",
            "bearer_token": "secret2",
            "authorization": "secret3",
            "session_cookie": "secret4",
            "access_token": "secret5",
            "refresh_token": "secret6",
            "private_key": "secret7",
            "secret_key": "secret8",
        })
        result = _strip_credentials(payload)
        for i in range(1, 9):
            assert f"secret{i}" not in result

    def test_preserves_rationale(self):
        payload = '{"rationale": "good trade setup", "api_key": "x"}'
        result = _strip_credentials(payload)
        assert "good trade setup" in result
        assert '"api_key": "[REDACTED]"' in result

    def test_preserves_reasoning(self):
        payload = '{"reasoning": "based on momentum", "secret_key": "y"}'
        result = _strip_credentials(payload)
        assert "based on momentum" in result
        assert "y" not in result or '"secret_key": "[REDACTED]"' in result

    def test_preserves_setup_reasoning(self):
        payload = '{"setup_reasoning": "breakout pattern", "access_token": "z"}'
        result = _strip_credentials(payload)
        assert "breakout pattern" in result

    def test_no_credentials_unchanged(self):
        payload = '{"symbol": "AAPL", "direction": "BUY"}'
        result = _strip_credentials(payload)
        assert result == payload


class TestTruncateUtf8(unittest.TestCase):
    """Test UTF-8 safe truncation."""

    def test_short_string_no_truncation(self):
        result, truncated = _truncate_utf8("hello", 100)
        assert result == "hello"
        assert truncated is False

    def test_exact_boundary_no_truncation(self):
        s = "a" * 10
        result, truncated = _truncate_utf8(s, 10)
        assert result == s
        assert truncated is False

    def test_truncation_at_ascii_boundary(self):
        s = "a" * 20
        result, truncated = _truncate_utf8(s, 10)
        assert len(result) == 10
        assert truncated is True

    def test_truncation_preserves_multibyte_chars(self):
        # Chinese chars are 3 bytes each in UTF-8
        s = "\u4e2d" * 100  # 300 bytes
        result, truncated = _truncate_utf8(s, 10)
        assert truncated is True
        encoded = result.encode("utf-8")
        assert len(encoded) <= 10
        # Should be 3 complete Chinese chars = 9 bytes
        assert len(encoded) == 9
        assert result == "\u4e2d" * 3

    def test_truncation_does_not_split_2byte_chars(self):
        # é is 2 bytes in UTF-8
        s = "\u00e9" * 100  # 200 bytes
        result, truncated = _truncate_utf8(s, 5)
        assert truncated is True
        encoded = result.encode("utf-8")
        assert len(encoded) <= 5
        # 2 complete chars = 4 bytes (fits in 5), 3 chars = 6 bytes (doesn't fit)
        assert len(encoded) == 4

    def test_truncation_does_not_split_4byte_chars(self):
        # Emoji 🎉 is 4 bytes in UTF-8
        s = "\U0001f389" * 100  # 400 bytes
        result, truncated = _truncate_utf8(s, 7)
        assert truncated is True
        encoded = result.encode("utf-8")
        assert len(encoded) <= 7
        # 1 complete emoji = 4 bytes
        assert len(encoded) == 4

    def test_empty_string(self):
        result, truncated = _truncate_utf8("", 100)
        assert result == ""
        assert truncated is False


class TestCaptureRawPMResponse(unittest.TestCase):
    """Test the main capture function."""

    @patch("utils.raw_pm_capture.PM_PROVENANCE_DETAIL", "full")
    def test_full_mode_basic(self):
        payload = '{"action": "BUY", "symbol": "AAPL"}'
        resp = capture_raw_pm_response(
            pm_cycle_id="cycle-001",
            profile="aggressive",
            model_id="gpt-4",
            prompt_version_id="v2.1",
            candidate_ids_supplied=["cand-1", "cand-2"],
            raw_payload=payload,
            parse_status="parse_success",
            attempt_ordinal=1,
        )
        assert isinstance(resp, RawPMResponse)
        assert resp.pm_cycle_id == "cycle-001"
        assert resp.profile == "aggressive"
        assert resp.model_id == "gpt-4"
        assert resp.prompt_version_id == "v2.1"
        assert resp.candidate_ids_supplied == ["cand-1", "cand-2"]
        assert resp.parse_status == "parse_success"
        assert resp.attempt_ordinal == 1
        assert resp.raw_payload == payload  # no creds to strip
        assert resp.payload_truncated is False

        # Verify hashes
        expected_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        assert resp.original_payload_hash == expected_hash
        assert resp.stored_payload_hash == expected_hash  # no changes
        assert resp.payload_size_bytes == len(payload.encode("utf-8"))

    @patch("utils.raw_pm_capture.PM_PROVENANCE_DETAIL", "minimal")
    def test_minimal_mode(self):
        payload = '{"action": "BUY"}'
        resp = capture_raw_pm_response(
            pm_cycle_id="cycle-002",
            profile="moderate",
            model_id="claude-3",
            prompt_version_id="v1.0",
            candidate_ids_supplied=[],
            raw_payload=payload,
            parse_status="parse_success",
        )
        assert resp.raw_payload is None
        assert resp.stored_payload_hash is None
        assert resp.original_payload_hash is not None
        assert resp.payload_size_bytes == len(payload.encode("utf-8"))

    @patch("utils.raw_pm_capture.PM_PROVENANCE_DETAIL", "full")
    def test_credential_stripping_in_capture(self):
        payload = '{"api_key": "sk-secret", "reasoning": "good", "action": "BUY"}'
        resp = capture_raw_pm_response(
            pm_cycle_id="cycle-003",
            profile="aggressive",
            model_id="gpt-4",
            prompt_version_id="v1",
            candidate_ids_supplied=[],
            raw_payload=payload,
            parse_status="parse_success",
        )
        assert "sk-secret" not in resp.raw_payload
        assert "good" in resp.raw_payload
        assert "[REDACTED]" in resp.raw_payload
        # Original hash is from the UNMODIFIED payload
        assert resp.original_payload_hash == hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()
        # Stored hash is from the REDACTED payload
        assert resp.stored_payload_hash != resp.original_payload_hash

    @patch("utils.raw_pm_capture.PM_PROVENANCE_DETAIL", "full")
    def test_large_payload_truncation(self):
        # Create a payload larger than 256KB
        large_payload = '{"data": "' + "x" * (MAX_RAW_PAYLOAD_BYTES + 1000) + '"}'
        resp = capture_raw_pm_response(
            pm_cycle_id="cycle-004",
            profile="moderate",
            model_id="gpt-4",
            prompt_version_id="v1",
            candidate_ids_supplied=[],
            raw_payload=large_payload,
            parse_status="parse_success",
        )
        assert resp.payload_truncated is True
        assert len(resp.raw_payload.encode("utf-8")) <= MAX_RAW_PAYLOAD_BYTES
        assert resp.payload_size_bytes == len(large_payload.encode("utf-8"))
        # Original hash is of the full (untruncated) payload
        assert resp.original_payload_hash == hashlib.sha256(
            large_payload.encode("utf-8")
        ).hexdigest()

    @patch("utils.raw_pm_capture.PM_PROVENANCE_DETAIL", "minimal")
    def test_minimal_mode_large_payload_sets_truncated_flag(self):
        large_payload = "x" * (MAX_RAW_PAYLOAD_BYTES + 100)
        resp = capture_raw_pm_response(
            pm_cycle_id="cycle-005",
            profile="moderate",
            model_id="gpt-4",
            prompt_version_id="v1",
            candidate_ids_supplied=[],
            raw_payload=large_payload,
            parse_status="parse_success",
        )
        assert resp.raw_payload is None
        assert resp.payload_truncated is True
        assert resp.payload_size_bytes > MAX_RAW_PAYLOAD_BYTES

    def test_uuid_generation(self):
        """Each capture produces a unique response_id."""
        with patch("utils.raw_pm_capture.PM_PROVENANCE_DETAIL", "full"):
            resp1 = capture_raw_pm_response(
                pm_cycle_id="c1", profile="p", model_id="m",
                prompt_version_id="v", candidate_ids_supplied=[],
                raw_payload="x", parse_status="parse_success",
            )
            resp2 = capture_raw_pm_response(
                pm_cycle_id="c1", profile="p", model_id="m",
                prompt_version_id="v", candidate_ids_supplied=[],
                raw_payload="x", parse_status="parse_success",
            )
        assert resp1.response_id != resp2.response_id

    def test_timestamp_is_utc(self):
        with patch("utils.raw_pm_capture.PM_PROVENANCE_DETAIL", "full"):
            resp = capture_raw_pm_response(
                pm_cycle_id="c1", profile="p", model_id="m",
                prompt_version_id="v", candidate_ids_supplied=[],
                raw_payload="x", parse_status="parse_success",
            )
        import datetime as dt
        assert resp.timestamp.tzinfo == dt.timezone.utc


class TestLinkResponseToLineages(unittest.TestCase):
    """Test lineage linking."""

    def test_creates_correct_links(self):
        links = link_response_to_lineages(
            response_id="resp-001",
            lineage_ids=["lin-1", "lin-2", "lin-3"],
            candidate_ids=["cand-1", "cand-2", None],
        )
        assert len(links) == 3
        assert all(isinstance(link, ResponseLineageLink) for link in links)
        assert links[0].response_id == "resp-001"
        assert links[0].lineage_id == "lin-1"
        assert links[0].candidate_id == "cand-1"
        assert links[2].candidate_id is None

    def test_empty_lists(self):
        links = link_response_to_lineages("resp-001", [], [])
        assert links == []

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            link_response_to_lineages("resp-001", ["a"], ["b", "c"])


class TestPersistRawResponse(unittest.TestCase):
    """Test persistence with a real SQLite database."""

    def setUp(self):
        from sqlalchemy import create_engine
        from db.provenance_schema import init_provenance_schema

        self.engine = create_engine("sqlite:///:memory:")
        init_provenance_schema(self.engine)

    @patch("utils.raw_pm_capture.PM_PROVENANCE_DETAIL", "full")
    def test_persist_and_read_back(self):
        resp = capture_raw_pm_response(
            pm_cycle_id="cycle-010",
            profile="aggressive",
            model_id="gpt-4",
            prompt_version_id="v1",
            candidate_ids_supplied=["c1"],
            raw_payload='{"symbol": "AAPL"}',
            parse_status="parse_success",
            attempt_ordinal=1,
        )
        persist_raw_response(self.engine, resp)

        # Read back
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM pm_raw_responses WHERE response_id = :rid"),
                {"rid": resp.response_id},
            ).fetchone()
        assert row is not None
        assert row._mapping["pm_cycle_id"] == "cycle-010"
        assert row._mapping["profile"] == "aggressive"
        assert row._mapping["original_payload_hash"] == resp.original_payload_hash

    @patch("utils.raw_pm_capture.PM_PROVENANCE_DETAIL", "full")
    def test_duplicate_response_id_handled(self):
        resp = capture_raw_pm_response(
            pm_cycle_id="cycle-011",
            profile="moderate",
            model_id="gpt-4",
            prompt_version_id="v1",
            candidate_ids_supplied=[],
            raw_payload='{"x": 1}',
            parse_status="parse_success",
        )
        persist_raw_response(self.engine, resp)
        # Second insert should not raise
        persist_raw_response(self.engine, resp)

        with self.engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM pm_raw_responses WHERE response_id = :rid"),
                {"rid": resp.response_id},
            ).scalar()
        assert count == 1

    @patch("utils.raw_pm_capture.PM_PROVENANCE_DETAIL", "full")
    def test_persist_lineage_links(self):
        resp = capture_raw_pm_response(
            pm_cycle_id="cycle-012",
            profile="moderate",
            model_id="gpt-4",
            prompt_version_id="v1",
            candidate_ids_supplied=["c1", "c2"],
            raw_payload='{"x": 1}',
            parse_status="parse_success",
        )
        persist_raw_response(self.engine, resp)

        links = link_response_to_lineages(
            resp.response_id, ["lin-1", "lin-2"], ["c1", "c2"]
        )
        persist_lineage_links(self.engine, links)

        with self.engine.connect() as conn:
            rows = conn.execute(
                text("SELECT * FROM response_lineage_links WHERE response_id = :rid"),
                {"rid": resp.response_id},
            ).fetchall()
        assert len(rows) == 2

    @patch("utils.raw_pm_capture.PM_PROVENANCE_DETAIL", "full")
    def test_duplicate_lineage_link_handled(self):
        resp = capture_raw_pm_response(
            pm_cycle_id="cycle-013",
            profile="moderate",
            model_id="gpt-4",
            prompt_version_id="v1",
            candidate_ids_supplied=["c1"],
            raw_payload='{"x": 1}',
            parse_status="parse_success",
        )
        persist_raw_response(self.engine, resp)

        links = link_response_to_lineages(resp.response_id, ["lin-1"], ["c1"])
        persist_lineage_links(self.engine, links)
        # Second insert should not raise
        persist_lineage_links(self.engine, links)

        with self.engine.connect() as conn:
            count = conn.execute(
                text(
                    "SELECT COUNT(*) FROM response_lineage_links "
                    "WHERE response_id = :rid AND lineage_id = :lid"
                ),
                {"rid": resp.response_id, "lid": "lin-1"},
            ).scalar()
        assert count == 1


if __name__ == "__main__":
    unittest.main()
