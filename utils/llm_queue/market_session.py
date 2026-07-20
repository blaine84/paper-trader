"""Market session helper for queue policy adjustments.

Determines the current US market session based on Eastern time.
Used by the dispatcher to apply market-hour pressure policies
(stricter deadlines and admission for non-critical requests during
regular trading hours).

Sessions:
- premarket: weekday 04:00-09:30 ET
- regular: weekday 09:30-16:00 ET
- postmarket: weekday 16:00-20:00 ET
- closed: weekends and weekday 20:00-04:00 ET

No holiday calendar in v1 — weekday hours are treated as market hours
conservatively.
"""

from __future__ import annotations

import logging
from datetime import datetime, time
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# US Eastern timezone
_ET = ZoneInfo("America/New_York")

# Session boundary times (Eastern)
_PREMARKET_OPEN = time(4, 0)
_REGULAR_OPEN = time(9, 30)
_REGULAR_CLOSE = time(16, 0)
_POSTMARKET_CLOSE = time(20, 0)


class MarketSessionHelper:
    """Determines current US market session for queue policy adjustments.

    Simple time-of-day + day-of-week check against US Eastern time.
    Does not use a holiday calendar for v1 (conservative: treats all
    weekday hours as potential market hours).
    """

    def current_session(self) -> str:
        """Returns 'regular', 'premarket', 'postmarket', or 'closed'.

        Uses the current wall-clock time in US Eastern timezone.
        Weekends always return 'closed'.
        """
        try:
            now_et = datetime.now(tz=_ET)
            return self._session_for_datetime(now_et)
        except Exception:
            # Fail-open: assume regular (conservative) if timezone lookup fails
            logger.warning(
                "Failed to determine market session, assuming 'regular' (conservative)"
            )
            return "regular"

    def is_market_hours(self) -> bool:
        """True during regular market hours (9:30-16:00 ET weekdays).

        Returns False on weekends and outside regular session.
        """
        try:
            now_et = datetime.now(tz=_ET)
            return self._is_regular_session(now_et)
        except Exception:
            # Fail-open: assume regular (conservative) if timezone lookup fails
            logger.warning(
                "Failed to determine market hours, assuming True (conservative)"
            )
            return True

    def _session_for_datetime(self, dt: datetime) -> str:
        """Determine session for a specific datetime (must be ET-aware)."""
        # Saturday = 5, Sunday = 6
        if dt.weekday() >= 5:
            return "closed"

        current_time = dt.time()

        if _PREMARKET_OPEN <= current_time < _REGULAR_OPEN:
            return "premarket"
        elif _REGULAR_OPEN <= current_time < _REGULAR_CLOSE:
            return "regular"
        elif _REGULAR_CLOSE <= current_time < _POSTMARKET_CLOSE:
            return "postmarket"
        else:
            return "closed"

    def _is_regular_session(self, dt: datetime) -> bool:
        """Check if datetime falls within regular market hours."""
        if dt.weekday() >= 5:
            return False

        current_time = dt.time()
        return _REGULAR_OPEN <= current_time < _REGULAR_CLOSE
