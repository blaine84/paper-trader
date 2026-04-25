"""
Blogger API v3 publisher.
Posts narrative updates to a Google Blogger blog.
Credentials from env vars. Graceful degradation when unconfigured.
"""

import json
import logging
import os
import time
from datetime import datetime

import requests

logger = logging.getLogger("blogger_publisher")

REQUIRED_ENV_VARS = [
    "BLOGGER_BLOG_ID",
    "GOOGLE_CLIENT_ID",
    "GOOGLE_CLIENT_SECRET",
    "GOOGLE_REFRESH_TOKEN",
]

TOKEN_URL = "https://oauth2.googleapis.com/token"
BLOGGER_API_BASE = "https://www.googleapis.com/blogger/v3/blogs"

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds


class BloggerPublisher:
    """Blogger post delivery. Reads credentials from env vars at init."""

    def __init__(self):
        self._blog_id = (os.getenv("BLOGGER_BLOG_ID") or "").strip()
        self._client_id = (os.getenv("GOOGLE_CLIENT_ID") or "").strip()
        self._client_secret = (os.getenv("GOOGLE_CLIENT_SECRET") or "").strip()
        self._refresh_token = (os.getenv("GOOGLE_REFRESH_TOKEN") or "").strip()
        self._access_token = None
        self._token_expiry = 0

        missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v, "").strip()]
        if missing:
            logger.warning(f"Blogger publishing disabled: missing env vars {missing}")

    def is_enabled(self) -> bool:
        """True only when all four required env vars are present and non-empty."""
        return bool(self._blog_id and self._client_id
                     and self._client_secret and self._refresh_token)

    def _refresh_access_token(self) -> str:
        """Refresh the OAuth2 access token using the refresh token."""
        resp = requests.post(TOKEN_URL, data={
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "refresh_token": self._refresh_token,
            "grant_type": "refresh_token",
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        self._access_token = data["access_token"]
        self._token_expiry = time.time() + data.get("expires_in", 3600) - 60
        return self._access_token

    def _get_token(self) -> str:
        """Get a valid access token, refreshing if expired."""
        if self._access_token and time.time() < self._token_expiry:
            return self._access_token
        return self._refresh_access_token()

    def publish(self, title: str, content: str) -> dict:
        """Post a narrative to Blogger.

        Returns:
            {"ok": bool, "post_id": str | None, "url": str | None, "error": str | None}
        """
        if not self.is_enabled():
            return {"ok": False, "error": "disabled", "post_id": None, "url": None}

        url = f"{BLOGGER_API_BASE}/{self._blog_id}/posts/"
        # Wrap narrative in HTML paragraph tags
        html_content = f"<p>{content}</p>"
        payload = {"kind": "blogger#post", "title": title, "content": html_content}

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                token = self._get_token()
                resp = requests.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=30,
                )
                if resp.status_code in (200, 201):
                    data = resp.json()
                    logger.info(f"Blogger post published: {data.get('url', '')}")
                    return {"ok": True, "post_id": data.get("id"),
                            "url": data.get("url"), "error": None}

                logger.error(f"Blogger API error (attempt {attempt}/{MAX_RETRIES}): "
                             f"{resp.status_code} {resp.text[:200]}")

                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)))

            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError) as exc:
                logger.warning(f"Blogger request failed (attempt {attempt}/{MAX_RETRIES}): {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)))

        logger.error(f"Blogger publish failed after {MAX_RETRIES} attempts")
        return {"ok": False, "error": "max_retries_exceeded",
                "post_id": None, "url": None}


def format_blog_title(update_type: str, date_str: str) -> str:
    """Generate a descriptive blog post title.

    E.g., "Morning Briefing - Apr 24, 2026"
    """
    display_name = {
        "morning_briefing": "Morning Briefing",
        "hourly_recap": "Hourly Recap",
        "afternoon_recap": "Afternoon Recap",
        "daily_wrap": "Daily Wrap",
        "weekly_wrap": "Weekly Wrap",
        "sunday_prep": "Sunday Prep",
        "flash_update": "🚨 Desk Flash",
    }.get(update_type, update_type.replace("_", " ").title())

    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        formatted_date = dt.strftime("%b %d, %Y")
    except (ValueError, TypeError):
        formatted_date = date_str

    return f"{display_name} - {formatted_date}"
