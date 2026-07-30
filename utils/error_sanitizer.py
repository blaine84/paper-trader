"""Helpers for safely rendering exception text in logs and persisted records."""

from __future__ import annotations

import re
from typing import Any


_URL_SECRET_PARAM_RE = re.compile(
    r"(?i)([?&](?:token|api_key|apikey|key|access_token|refresh_token|secret)=[^&\s)\"']+)"
)
_URL_SECRET_PARAM_VALUE_RE = re.compile(
    r"(?i)([?&](?:token|api_key|apikey|key|access_token|refresh_token|secret)=)([^&\s)\"']+)"
)
_HEADER_SECRET_RE = re.compile(
    r"(?i)\b(authorization|x-api-key|api-key)\s*[:=]\s*([^\s,;}]+)"
)


def sanitize_error_text(error: Any) -> str:
    """Return exception text with common credential-bearing fields redacted."""
    text = str(error)
    text = _URL_SECRET_PARAM_VALUE_RE.sub(r"\1[REDACTED]", text)
    text = _HEADER_SECRET_RE.sub(r"\1=[REDACTED]", text)
    return text

