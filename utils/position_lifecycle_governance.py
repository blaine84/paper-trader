"""
Position Lifecycle Governance - Core Data Models and Constants

Deterministic lifecycle governance evaluator that resolves every open trading
position to exactly one lifecycle state and decision. This module defines the
data models, enums, and constants used throughout the governance system.

The evaluator is a pure function — it reads trade data and event history,
computes a governance decision, and returns a structured result without
executing trades or mutating state.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone


# ---------------------------------------------------------------------------
# Lifecycle States
# ---------------------------------------------------------------------------

LIFECYCLE_STATES = {
    "skipped",                  # Position already closed/cancelled
    "intraday_ok",              # Within time limits, no governance issues
    "intraday_warning",         # Approaching force-close time limit
    "intraday_expired",         # Past force-close time limit
    "news_reconfirmation_due",  # 4h before news expiry, no reconfirmation
    "news_grace",               # Between expiry and grace deadline
    "news_expired",             # Past grace deadline, no valid reconfirmation
    "overnight_authorized",     # Valid overnight auth + valid stop geometry
    "overnight_unauthorized",   # Past Hard_Wall, no valid overnight auth
    "invalid_stop_geometry",    # Overnight auth exists but stop is invalid
    "close_required",           # Generic close (EXIT_NOW, profile cap, etc.)
}

# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

DECISIONS = {"skip", "allow", "warn", "close", "repair_stop", "authorize_required"}

# ---------------------------------------------------------------------------
# LifecycleDecision Dataclass
# ---------------------------------------------------------------------------


@dataclass
class LifecycleDecision:
    """Structured return type from evaluate_position_lifecycle.

    Every open position resolves to exactly one LifecycleDecision containing
    the governance state, decision, and all context needed by the executor
    and audit systems.
    """

    # Required fields
    decision: str          # One of DECISIONS
    state: str             # One of LIFECYCLE_STATES
    reason_type: str       # Machine-readable reason (e.g., "news_reconfirmation_missing")
    trade_id: int
    symbol: str
    profile: str
    setup_type: str
    entry_time: datetime
    hours_held: float
    requires_event: bool   # Whether executor should log this decision

    # Conditional fields
    close_reason: str | None = None  # Human-readable, required when decision=="close"

    # Context metadata
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Overnight Authorization Schema
# ---------------------------------------------------------------------------

OVERNIGHT_AUTH_REQUIRED_FIELDS = {
    "trade_id",          # int — must match trade
    "symbol",            # str — must match trade
    "profile",           # str — must match trade
    "authorized_by",     # str — PM/CEO/operator identity
    "authorized_at",     # ISO-8601 timestamp
    "expires_at",        # ISO-8601 timestamp (must be > now_utc)
    "overnight_thesis",  # str — non-empty justification
    "risk_plan",         # str — non-empty risk management plan
    "stop_price",        # float — required protective stop
}

OVERNIGHT_AUTH_OPTIONAL_FIELDS = {
    "target_price",               # float
    "max_overnight_exposure_pct",  # float
}

# ---------------------------------------------------------------------------
# Profile Overnight Caps
# ---------------------------------------------------------------------------

PROFILE_OVERNIGHT_CAPS = {
    "conservative": {"max_positions": 0, "max_gross_exposure_pct": 0.0},
    "moderate": {"max_positions": 1, "max_gross_exposure_pct": 0.10},
    "aggressive": {"max_positions": 2, "max_gross_exposure_pct": 0.25},
}

# ---------------------------------------------------------------------------
# Trading Calendar: Hard_Wall Constants and Holiday Sets
# ---------------------------------------------------------------------------

HARD_WALL_HOUR = 15
HARD_WALL_MINUTE = 45

US_MARKET_HOLIDAYS_2025 = {
    date(2025, 1, 1),   # New Year's Day
    date(2025, 1, 20),  # MLK Day
    date(2025, 2, 17),  # Presidents' Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),   # Independence Day
    date(2025, 9, 1),   # Labor Day
    date(2025, 11, 27), # Thanksgiving
    date(2025, 12, 25), # Christmas
}

US_MARKET_HOLIDAYS_2026 = {
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 19),  # MLK Day
    date(2026, 2, 16),  # Presidents' Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 6, 19),  # Juneteenth
    date(2026, 7, 3),   # Independence Day (observed)
    date(2026, 9, 7),   # Labor Day
    date(2026, 11, 26), # Thanksgiving
    date(2026, 12, 25), # Christmas
}

ALL_MARKET_HOLIDAYS = US_MARKET_HOLIDAYS_2025 | US_MARKET_HOLIDAYS_2026


def is_trading_day(now_et: datetime) -> bool:
    """Determine if the given ET datetime falls on a trading day.

    A trading day is a weekday (Mon-Fri) that is not a US market holiday.
    Hard_Wall enforcement only applies on trading days.

    Args:
        now_et: A datetime (or date) in US/Eastern timezone.

    Returns:
        True if the date is a trading day, False otherwise.
    """
    d = now_et.date() if isinstance(now_et, datetime) else now_et
    # Weekday check: Mon=0, Fri=4
    if d.weekday() >= 5:  # Saturday or Sunday
        return False
    if d in ALL_MARKET_HOLIDAYS:
        return False
    return True


def most_recent_completed_hard_wall(now_et: datetime) -> datetime | None:
    """Return the most recent completed Hard_Wall datetime (3:45 PM ET on the
    most recent trading day at or before now_et).

    Used for "missed prior Hard_Wall" detection: if a position's entry_time
    is before this timestamp and no valid overnight auth exists, the position
    is overnight_unauthorized.

    Logic:
    1. If today is a trading day AND now_et >= 15:45 ET, then the most recent
       completed hard wall is today at 15:45 ET.
    2. Otherwise, go back day by day (up to 10 days) until we find a trading
       day — that day's 15:45 ET is the answer.

    The returned datetime is timezone-aware in the same timezone as now_et.

    Args:
        now_et: Current time in US/Eastern (timezone-aware).

    Returns:
        Timezone-aware datetime at 15:45 ET on the most recent completed
        trading day, or None if no prior trading day can be determined.
    """
    tz = now_et.tzinfo
    today = now_et.date()
    hard_wall_time_today = datetime(
        today.year, today.month, today.day,
        HARD_WALL_HOUR, HARD_WALL_MINUTE, 0,
        tzinfo=tz,
    )

    # Check if today is a trading day and we're past the hard wall
    if is_trading_day(now_et) and now_et >= hard_wall_time_today:
        return hard_wall_time_today

    # Go back day by day to find the most recent trading day
    candidate = today - timedelta(days=1)
    for _ in range(10):
        if is_trading_day(candidate):
            return datetime(
                candidate.year, candidate.month, candidate.day,
                HARD_WALL_HOUR, HARD_WALL_MINUTE, 0,
                tzinfo=tz,
            )
        candidate -= timedelta(days=1)

    return None


# ---------------------------------------------------------------------------
# Intraday Setup Time Limits (minutes)
# ---------------------------------------------------------------------------

SETUP_TIME_LIMITS = {
    "momentum_fade": {"stale": 35, "alert": 45, "revalidate": 60, "force_close": 75},
    "gap_and_go":    {"alert": 60, "force_close": 90},
    "vwap_reclaim":  {"alert": 60, "force_close": 90},
    "orb":           {"alert": 45, "force_close": 75},
    "trend_pullback": {"alert": 90, "force_close": 120},
    "news_catalyst": {"alert": 60, "force_close": 90},
    "short_squeeze": {"alert": 30, "force_close": 60},
}

DEFAULT_LIMITS = {"alert": 60, "force_close": 90}
INTRADAY_SETUPS = set(SETUP_TIME_LIMITS.keys())

# ---------------------------------------------------------------------------
# News Governance Classification Constants (local to avoid coupling to DB module)
# ---------------------------------------------------------------------------

NEWS_GOVERNANCE_SETUP_TYPES = {
    "news_catalyst",
    "news_breakout",
    "news_headline",
    "confirmed_strait_of",
    "sector_rotation",
}

NEWS_GOVERNANCE_ENTRY_TERMS = {
    "catalyst", "headline", "news", "geopolitical", "strait of hormuz",
    "earnings", "guidance", "tariff", "fed", "cpi", "jobs report",
    "oil spike", "vessel", "incident", "escalation",
}

NEWS_GOVERNANCE_CATALYST_TYPES = {"news", "geopolitical", "earnings", "macro", "headline"}

# ---------------------------------------------------------------------------
# Reconfirmation Validity Constants
# ---------------------------------------------------------------------------

HOLD_BLOCKING_THESIS_STATUSES = {"deteriorating", "invalidated"}


# ---------------------------------------------------------------------------
# Helper: Build Decision Dict
# ---------------------------------------------------------------------------


def _build_decision(
    trade: dict,
    now_utc: datetime,
    *,
    decision: str,
    state: str,
    reason_type: str,
    requires_event: bool,
    close_reason: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Construct a full LifecycleDecision result dict with computed hours_held.

    This helper ensures every return from the evaluator has a consistent
    schema matching Requirement 11.1.
    """
    entry_time = _parse_dt(trade.get("entry_time"))
    if entry_time is not None:
        # Ensure now_utc is also timezone-aware for subtraction
        now_aware = now_utc if now_utc.tzinfo is not None else now_utc.replace(tzinfo=timezone.utc)
        hours_held = (now_aware - entry_time).total_seconds() / 3600
    else:
        hours_held = 0.0

    result = {
        "decision": decision,
        "state": state,
        "reason_type": reason_type,
        "trade_id": trade.get("id"),
        "symbol": trade.get("symbol", ""),
        "profile": trade.get("profile", ""),
        "setup_type": trade.get("setup_type", ""),
        "entry_time": entry_time,
        "hours_held": hours_held,
        "requires_event": requires_event,
        "close_reason": close_reason,
        "metadata": metadata or {},
    }
    return result


# ---------------------------------------------------------------------------
# Helper: News Governance Classification (Pure Function)
# ---------------------------------------------------------------------------


def classify_news_governed(trade: dict, events: list[dict]) -> bool:
    """
    Pure classification function. Determines if a trade is news-governed
    based on setup_type, entry text terms, catalyst_type, or persisted
    classification event.

    No DB access — operates purely on pre-fetched trade dict and events list.

    Classification checks in order:
    1. Persisted event: if a news_governance_classified event exists in events
    2. setup_type match: if trade's setup_type is in the governed set
    3. Entry text term match: if any term from NEWS_GOVERNANCE_ENTRY_TERMS
       appears (case-insensitive) in reason_entry, thesis, or invalidators
    4. catalyst_type match: if trade's catalyst_type is in NEWS_GOVERNANCE_CATALYST_TYPES

    Args:
        trade: Dict with keys: setup_type, reason_entry, thesis, invalidators,
               catalyst_type (all optional, missing/None treated as empty).
        events: List of trade_events dicts for this trade (pre-fetched by caller).
               Each dict has: event_type, timestamp, payload (parsed dict).

    Returns:
        True if the trade is news-governed, False otherwise.
    """
    # 1. Check for persisted news_governance_classified event
    for event in events:
        if event.get("event_type") == "news_governance_classified":
            return True

    # 2. setup_type match
    setup_type = trade.get("setup_type", "") or ""
    if setup_type in NEWS_GOVERNANCE_SETUP_TYPES:
        return True

    # 3. Entry text term match (case-insensitive)
    text_fields = [
        str(trade.get("reason_entry", "") or ""),
        str(trade.get("thesis", "") or ""),
        str(trade.get("invalidators", "") or ""),
    ]
    combined_text = " ".join(text_fields).lower()
    for term in NEWS_GOVERNANCE_ENTRY_TERMS:
        if term.lower() in combined_text:
            return True

    # 4. catalyst_type match
    catalyst_type = trade.get("catalyst_type", "") or ""
    if catalyst_type in NEWS_GOVERNANCE_CATALYST_TYPES:
        return True

    return False


# ---------------------------------------------------------------------------
# Internal Check: Skip (closed/cancelled positions)
# ---------------------------------------------------------------------------


def _check_skip(trade: dict, now_utc: datetime) -> dict | None:
    """Return a skipped decision for closed/cancelled positions.

    Priority 1: If the position is already closed or cancelled, skip it
    without further evaluation.
    """
    status = trade.get("status", "").lower()
    if status in ("closed", "cancelled"):
        return _build_decision(
            trade,
            now_utc,
            decision="skip",
            state="skipped",
            reason_type="position_already_closed",
            requires_event=False,
        )
    return None


# ---------------------------------------------------------------------------
# Helper: Parse datetime values (ISO-8601 strings or datetime objects)
# ---------------------------------------------------------------------------


def _parse_dt(value) -> datetime | None:
    """Parse a datetime value, returning a timezone-aware UTC datetime or None.

    Handles:
    - datetime objects: returned as-is if tz-aware, or with UTC attached if naive
    - ISO-8601 strings: parsed via fromisoformat, made UTC-aware if naive
    - None or invalid values: returns None
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except (ValueError, TypeError):
            return None
    return None


# ---------------------------------------------------------------------------
# Internal Check: EXIT_NOW governance event
# ---------------------------------------------------------------------------


def _check_exit_now(events: list[dict], trade: dict, now_utc: datetime) -> dict | None:
    """Priority 2: Check for the latest valid EXIT_NOW governance event.

    An EXIT_NOW event is found in events with:
    - event_type == "news_reconfirmation_submitted"
    - payload.decision == "EXIT_NOW"

    A valid EXIT_NOW requires:
    - decided_at: parseable datetime (canonical timestamp for ordering)
    - decided_by: non-empty string (authorizer identity)
    - decision: "EXIT_NOW"

    Only the LATEST valid EXIT_NOW event (by decided_at) is considered.
    Returns close_required state with reason_type="pm_operator_exit_requested".
    """
    latest_exit: dict | None = None
    latest_decided_at: datetime | None = None

    for event in events:
        if event.get("event_type") != "news_reconfirmation_submitted":
            continue

        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue

        if payload.get("decision") != "EXIT_NOW":
            continue

        # Validate required fields
        decided_at = _parse_dt(payload.get("decided_at"))
        if decided_at is None:
            continue

        decided_by = payload.get("decided_by")
        if not decided_by or not isinstance(decided_by, str) or not decided_by.strip():
            continue

        # This is a valid EXIT_NOW event — track the latest by decided_at
        if latest_decided_at is None or decided_at > latest_decided_at:
            latest_decided_at = decided_at
            latest_exit = payload

    if latest_exit is None:
        return None

    return _build_decision(
        trade,
        now_utc,
        decision="close",
        state="close_required",
        reason_type="pm_operator_exit_requested",
        requires_event=True,
        close_reason="PM/operator requested exit (EXIT_NOW)",
        metadata={
            "decided_by": latest_exit.get("decided_by"),
            "decided_at": str(latest_decided_at),
        },
    )


# ---------------------------------------------------------------------------
# Helper: Reconfirmation Validity Checking
# ---------------------------------------------------------------------------

_RECONFIRM_REQUIRED_FIELDS = (
    "decided_at",
    "decided_by",
    "new_expiry_time",
    "risk_plan",
    "original_catalyst",
    "fresh_catalyst_evidence",
    "fresh_catalyst_timestamp",
)


def _is_valid_reconfirmation(
    payload: dict,
    entry_time: datetime,
    prior_reconfirmation: dict | None = None,
) -> bool:
    """Check if a RECONFIRM_AND_HOLD payload meets all validity requirements.

    Validation rules (ALL must pass):
    1. Required fields present and non-empty: decided_at, decided_by,
       new_expiry_time, risk_plan, original_catalyst, fresh_catalyst_evidence,
       fresh_catalyst_timestamp
    2. fresh_catalyst_evidence length >= 20 characters
    3. fresh_catalyst_timestamp > entry_time
    4. If prior_reconfirmation exists: fresh_catalyst_timestamp > prior's
       fresh_catalyst_timestamp
    5. thesis_status not in {"deteriorating", "invalidated"}
    6. new_expiry_time <= decided_at + 24 hours

    Args:
        payload: The RECONFIRM_AND_HOLD event payload dict.
        entry_time: The trade's entry time (timezone-aware UTC datetime).
        prior_reconfirmation: The prior valid reconfirmation payload, or None.

    Returns:
        True if the reconfirmation is valid, False otherwise.
    """
    # Rule 1: All required fields present and non-empty
    for field_name in _RECONFIRM_REQUIRED_FIELDS:
        value = payload.get(field_name)
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False

    # Rule 2: fresh_catalyst_evidence >= 20 characters
    fresh_evidence = payload.get("fresh_catalyst_evidence", "")
    if not isinstance(fresh_evidence, str) or len(fresh_evidence) < 20:
        return False

    # Parse timestamps needed for remaining checks
    fresh_catalyst_ts = _parse_dt(payload.get("fresh_catalyst_timestamp"))
    if fresh_catalyst_ts is None:
        return False

    decided_at = _parse_dt(payload.get("decided_at"))
    if decided_at is None:
        return False

    new_expiry_time = _parse_dt(payload.get("new_expiry_time"))
    if new_expiry_time is None:
        return False

    # Rule 3: fresh_catalyst_timestamp > entry_time
    if fresh_catalyst_ts <= entry_time:
        return False

    # Rule 4: If prior reconfirmation exists, fresh_catalyst_timestamp > prior's
    if prior_reconfirmation is not None:
        prior_ts = _parse_dt(prior_reconfirmation.get("fresh_catalyst_timestamp"))
        if prior_ts is not None and fresh_catalyst_ts <= prior_ts:
            return False

    # Rule 5: thesis_status not in blocking set
    thesis_status = payload.get("thesis_status", "")
    if isinstance(thesis_status, str) and thesis_status in HOLD_BLOCKING_THESIS_STATUSES:
        return False

    # Rule 6: new_expiry_time <= decided_at + 24 hours
    max_expiry = decided_at + timedelta(hours=24)
    if new_expiry_time > max_expiry:
        return False

    return True


# ---------------------------------------------------------------------------
# Helper: Full Overnight Authorization Validation
# ---------------------------------------------------------------------------


def latest_valid_overnight_authorization(
    events: list[dict],
    trade: dict,
    now_utc: datetime,
) -> dict | None:
    """
    Find the latest valid overnight_authorized event from pre-fetched events.

    Validation rules:
    - event_type == "overnight_authorized"
    - payload matches trade's trade_id, symbol, profile
    - Non-empty: overnight_thesis, risk_plan, stop_price
    - expires_at > now_utc

    Returns parsed payload dict or None.
    """
    valid_auths: list[tuple[datetime, dict]] = []

    for event in events:
        if event.get("event_type") != "overnight_authorized":
            continue

        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue

        # Validate all required fields are present and non-empty
        # Required: trade_id, symbol, profile, authorized_by, authorized_at,
        #           expires_at, overnight_thesis, risk_plan, stop_price
        trade_id = payload.get("trade_id")
        symbol = payload.get("symbol")
        profile = payload.get("profile")
        authorized_by = payload.get("authorized_by")
        authorized_at_raw = payload.get("authorized_at")
        expires_at_raw = payload.get("expires_at")
        overnight_thesis = payload.get("overnight_thesis")
        risk_plan = payload.get("risk_plan")
        stop_price = payload.get("stop_price")

        # Check required fields are present
        if trade_id is None or symbol is None or profile is None:
            continue
        if authorized_by is None or authorized_at_raw is None or expires_at_raw is None:
            continue
        if overnight_thesis is None or risk_plan is None or stop_price is None:
            continue

        # Validate authorized_by is a non-empty string
        if not isinstance(authorized_by, str) or not authorized_by.strip():
            continue

        # Validate overnight_thesis is a non-empty string
        if not isinstance(overnight_thesis, str) or not overnight_thesis.strip():
            continue

        # Validate risk_plan is a non-empty string
        if not isinstance(risk_plan, str) or not risk_plan.strip():
            continue

        # Validate stop_price is a valid number (not None, not 0)
        try:
            stop_price_num = float(stop_price)
        except (TypeError, ValueError):
            continue
        if stop_price_num == 0:
            continue

        # Field matching: trade_id, symbol, profile must match trade
        if trade_id != trade.get("id"):
            continue
        if symbol != trade.get("symbol"):
            continue
        if profile != trade.get("profile"):
            continue

        # Parse and validate expires_at > now_utc
        expires_at = _parse_dt(expires_at_raw)
        if expires_at is None:
            continue
        if expires_at <= now_utc:
            continue

        # Parse authorized_at for ordering (latest wins)
        authorized_at = _parse_dt(authorized_at_raw)
        if authorized_at is None:
            # Fallback to event timestamp if authorized_at is unparseable
            event_ts = _parse_dt(event.get("timestamp"))
            if event_ts is None:
                # Use epoch as last resort so we can still include this valid auth
                authorized_at = datetime(1970, 1, 1, tzinfo=timezone.utc)
            else:
                authorized_at = event_ts

        valid_auths.append((authorized_at, payload))

    if not valid_auths:
        return None

    # Return the one with the latest authorized_at
    valid_auths.sort(key=lambda x: x[0])
    return valid_auths[-1][1]


# ---------------------------------------------------------------------------
# Helper: Profile Exposure Cap Enforcement
# ---------------------------------------------------------------------------


def check_profile_exposure_cap(
    profile: str,
    trade: dict,
    current_price: float | None,
    current_equity: float | None,
    overnight_position_counts: dict[str, int] | None,
    overnight_exposure_pcts: dict[str, float] | None,
) -> tuple[bool, str]:
    """
    Check whether adding this trade to overnight carry would exceed profile caps.

    IMPORTANT: overnight_position_counts and overnight_exposure_pcts MUST
    exclude the current trade being evaluated. This function adds the
    candidate's exposure to determine if the cap would be exceeded.

    Exposure formula: abs(quantity * price) / equity
    - price: current_price if available, else entry_price
    - equity: current_equity (latest balance row / portfolio equity;
              fallback to starting_equity from env if unavailable)

    Returns:
        (allowed: bool, reason: str)
        - allowed=True, reason="" if within caps
        - allowed=False, reason="profile_cap_exceeded: ..." if cap exceeded
    """
    # Step 1: Look up caps for this profile. Unknown profile → conservative (most restrictive).
    caps = PROFILE_OVERNIGHT_CAPS.get(profile, PROFILE_OVERNIGHT_CAPS["conservative"])
    max_positions = caps["max_positions"]
    max_gross_exposure_pct = caps["max_gross_exposure_pct"]

    # Step 2: Get existing count for this profile (default 0 if None/missing)
    counts = overnight_position_counts if overnight_position_counts is not None else {}
    existing_count = counts.get(profile, 0)

    # Step 3: Check position count cap
    if existing_count >= max_positions:
        return (
            False,
            f"profile_cap_exceeded: {profile} already has {existing_count} overnight "
            f"position(s), max allowed is {max_positions}",
        )

    # Step 4: Calculate candidate exposure
    price = current_price if current_price is not None else trade.get("entry_price")
    quantity = trade.get("quantity", 0) or 0

    # If current_equity is None or <= 0, cannot calculate exposure.
    # Fail open for exposure calc only (position count still enforced above).
    if current_equity is None or current_equity <= 0:
        return (True, "")

    # price could still be None if both current_price and entry_price are missing
    if price is None:
        price = 0

    candidate_exposure_pct = abs(quantity * price) / current_equity

    # Step 5: Get existing exposure for this profile (default 0.0 if None/missing)
    exposures = overnight_exposure_pcts if overnight_exposure_pcts is not None else {}
    existing_exposure = exposures.get(profile, 0.0)

    # Step 6: Check total exposure cap
    total_exposure = existing_exposure + candidate_exposure_pct
    if total_exposure > max_gross_exposure_pct:
        return (
            False,
            f"profile_cap_exceeded: {profile} total overnight exposure "
            f"{total_exposure:.4f} ({existing_exposure:.4f} existing + "
            f"{candidate_exposure_pct:.4f} candidate) exceeds max "
            f"{max_gross_exposure_pct:.2f}",
        )

    # Step 7: Within all caps — allowed
    return (True, "")


# ---------------------------------------------------------------------------
# Helper: Stop Geometry Validation for Overnight Carry
# ---------------------------------------------------------------------------


def validate_stop_geometry(
    trade: dict,
    overnight_auth: dict | None,
    current_price: float | None,
) -> tuple[bool, str]:
    """
    Validate stop geometry for overnight carry.

    Checks are evaluated in order — the first failing check determines the result.

    Check order:
    1. current_price is None → fail safe to invalid
    2. trade has no stop_price (None or 0) → missing stop
    3. trade stop != auth stop → authorization mismatch
    4. Direction-based inversion (LONG stop >= price, SHORT stop <= price)

    Args:
        trade: Dict with keys: stop_price, direction (may be any case).
        overnight_auth: The overnight authorization payload dict, or None.
        current_price: Latest market price, or None if unavailable.

    Returns:
        (is_valid: bool, reason: str)
        - (True, "") if stop geometry is valid
        - (False, reason) if invalid

    Reasons for invalidity:
    - "missing_price_fail_safe": current_price unavailable
    - "missing_stop": no stop_price on trade
    - "authorization_stop_mismatch": trade stop != auth stop
    - "inverted_long": stop >= current_price for LONG
    - "inverted_short": stop <= current_price for SHORT
    """
    # Check 1: current_price unavailable → fail safe
    if current_price is None:
        return (False, "missing_price_fail_safe")

    # Check 2: Missing stop_price on trade (None or 0)
    stop_price = trade.get("stop_price")
    if stop_price is None or stop_price == 0:
        return (False, "missing_stop")

    stop_price = float(stop_price)

    # Check 3: Authorization stop mismatch
    if overnight_auth is not None and "stop_price" in overnight_auth:
        auth_stop = overnight_auth["stop_price"]
        if auth_stop is not None and float(auth_stop) != stop_price:
            return (False, "authorization_stop_mismatch")

    # Check 4: Direction-based inversion (normalize direction to upper case)
    direction = (trade.get("direction") or "").upper()

    if direction == "LONG" and stop_price >= current_price:
        return (False, "inverted_long")

    if direction == "SHORT" and stop_price <= current_price:
        return (False, "inverted_short")

    # All checks pass
    return (True, "")


def compute_news_governance_state(
    trade: dict,
    events: list[dict],
    now_utc: datetime,
) -> dict | None:
    """Evaluate news governance state using pure classification and temporal logic.

    Determines the current governance state for a news-governed trade by:
    1. Checking EXIT_NOW vs RECONFIRM_AND_HOLD precedence (latest decided_at wins)
    2. Computing effective_expiry from entry_time + 24h or latest valid reconfirmation
    3. Applying temporal logic: warning (4h before), grace (expiry to +30m), expired (past grace)

    Returns a LifecycleDecision dict if news governance is decisive,
    or None to continue evaluation (position is within safe window).
    """
    # -----------------------------------------------------------------------
    # Step 1: Gather all news_reconfirmation_submitted events and determine
    # the latest governance decision by decided_at timestamp.
    # -----------------------------------------------------------------------
    latest_exit_now: dict | None = None
    latest_exit_decided_at: datetime | None = None

    # Collect all RECONFIRM_AND_HOLD events for sequential validation
    reconfirm_candidates: list[tuple[datetime, dict]] = []

    for event in events:
        if event.get("event_type") != "news_reconfirmation_submitted":
            continue

        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue

        decision_value = payload.get("decision")
        decided_at = _parse_dt(payload.get("decided_at"))
        if decided_at is None:
            continue

        if decision_value == "EXIT_NOW":
            # Validate EXIT_NOW required fields
            decided_by = payload.get("decided_by")
            if not decided_by or not isinstance(decided_by, str) or not decided_by.strip():
                continue
            if latest_exit_decided_at is None or decided_at > latest_exit_decided_at:
                latest_exit_decided_at = decided_at
                latest_exit_now = payload

        elif decision_value == "RECONFIRM_AND_HOLD":
            # Collect for sequential validation below
            reconfirm_candidates.append((decided_at, payload))

    # -----------------------------------------------------------------------
    # Step 1b: Validate RECONFIRM_AND_HOLD events sequentially (earliest first)
    # Each reconfirmation is validated against the prior valid one.
    # -----------------------------------------------------------------------
    entry_time = _parse_dt(trade.get("entry_time"))
    if entry_time is None:
        # Cannot evaluate without entry_time — skip news governance
        return None

    # Sort by decided_at ascending (earliest first)
    reconfirm_candidates.sort(key=lambda x: x[0])

    latest_reconfirm: dict | None = None
    latest_reconfirm_decided_at: datetime | None = None
    prior_valid_reconfirmation: dict | None = None

    for decided_at, payload in reconfirm_candidates:
        if _is_valid_reconfirmation(payload, entry_time, prior_valid_reconfirmation):
            prior_valid_reconfirmation = payload
            latest_reconfirm = payload
            latest_reconfirm_decided_at = decided_at

    # -----------------------------------------------------------------------
    # Step 2: Determine precedence — the event with the LATEST decided_at wins.
    # A stale EXIT_NOW SHALL NOT override a later valid RECONFIRM_AND_HOLD.
    # -----------------------------------------------------------------------
    if latest_exit_now is not None and latest_reconfirm is not None:
        # Both exist — latest decided_at determines current state
        if latest_exit_decided_at >= latest_reconfirm_decided_at:
            # EXIT_NOW is latest → close
            return _build_decision(
                trade,
                now_utc,
                decision="close",
                state="close_required",
                reason_type="pm_operator_exit_requested",
                requires_event=True,
                close_reason="PM/operator requested exit (EXIT_NOW)",
                metadata={
                    "decided_by": latest_exit_now.get("decided_by"),
                    "decided_at": str(latest_exit_decided_at),
                },
            )
        # else: RECONFIRM_AND_HOLD is latest — fall through to temporal logic
    elif latest_exit_now is not None:
        # Only EXIT_NOW exists (no reconfirmation) → close
        return _build_decision(
            trade,
            now_utc,
            decision="close",
            state="close_required",
            reason_type="pm_operator_exit_requested",
            requires_event=True,
            close_reason="PM/operator requested exit (EXIT_NOW)",
            metadata={
                "decided_by": latest_exit_now.get("decided_by"),
                "decided_at": str(latest_exit_decided_at),
            },
        )

    # -----------------------------------------------------------------------
    # Step 3: Compute effective_expiry
    # If valid reconfirmation exists (and is the latest decision): use new_expiry_time
    # Otherwise: entry_time + 24 hours
    # -----------------------------------------------------------------------
    if latest_reconfirm is not None:
        effective_expiry = _parse_dt(latest_reconfirm.get("new_expiry_time"))
        if effective_expiry is None:
            # Fallback to entry_time + 24h if new_expiry_time is unparseable
            effective_expiry = entry_time + timedelta(hours=24)
    else:
        effective_expiry = entry_time + timedelta(hours=24)

    # -----------------------------------------------------------------------
    # Step 4: Temporal state computation
    # -----------------------------------------------------------------------
    warning_time = effective_expiry - timedelta(hours=4)
    grace_deadline = effective_expiry + timedelta(minutes=30)
    hours_held = (now_utc - entry_time).total_seconds() / 3600

    if now_utc < warning_time:
        # No governance issue — continue evaluation
        return None

    if now_utc < effective_expiry:
        # Warning window: 4h before expiry
        return _build_decision(
            trade,
            now_utc,
            decision="warn",
            state="news_reconfirmation_due",
            reason_type="news_reconfirmation_missing",
            requires_event=True,
            metadata={
                "effective_expiry": str(effective_expiry),
                "warning_time": str(warning_time),
                "hours_held": hours_held,
            },
        )

    if now_utc < grace_deadline:
        # Grace window: between expiry and expiry + 30min
        return _build_decision(
            trade,
            now_utc,
            decision="warn",
            state="news_grace",
            reason_type="news_grace_period",
            requires_event=True,
            metadata={
                "effective_expiry": str(effective_expiry),
                "grace_deadline": str(grace_deadline),
            },
        )

    # Past grace deadline → expired → close
    return _build_decision(
        trade,
        now_utc,
        decision="close",
        state="news_expired",
        reason_type="news_expiry_force_close",
        requires_event=True,
        close_reason="News-governed position expired without valid reconfirmation",
        metadata={
            "effective_expiry": str(effective_expiry),
            "grace_deadline": str(grace_deadline),
            "close_reason": "News-governed position past grace deadline without valid reconfirmation",
        },
    )


def _check_news_governance(
    trade: dict, events: list[dict], now_utc: datetime
) -> dict | None:
    """Priority 3: Check news governance expiry/grace/warning.

    First checks if the trade is news-governed via classify_news_governed.
    If not news-governed, returns None to continue evaluation.
    If news-governed, delegates to compute_news_governance_state for
    temporal logic and EXIT_NOW/RECONFIRM_AND_HOLD precedence.
    """
    if not classify_news_governed(trade, events):
        return None
    return compute_news_governance_state(trade, events, now_utc)


def _check_hard_wall(
    trade: dict,
    events: list[dict],
    now_et: datetime,
    now_utc: datetime,
    current_price: float | None,
    current_equity: float | None,
    overnight_position_counts: dict[str, int] | None,
    overnight_exposure_pcts: dict[str, float] | None,
) -> dict | None:
    """Priority 4: Check Hard_Wall + overnight authorization.

    Two Hard_Wall checks:
    1. Current-session: trading day AND now_et >= 15:45 ET
    2. Missed prior-session: entry_time < most_recent_completed_hard_wall

    If either triggers:
    - Check for valid overnight auth → if valid, enforce profile caps
    - If caps OK → return None (continue to stop geometry in priority 5)
    - If caps exceeded → return close_required
    - If no valid auth → return overnight_unauthorized with close

    If neither triggers → return None (continue evaluation).
    """
    hard_wall_triggered = False

    # --- Check 1: Current-session Hard_Wall ---
    if is_trading_day(now_et):
        if (now_et.hour > HARD_WALL_HOUR or
                (now_et.hour == HARD_WALL_HOUR and now_et.minute >= HARD_WALL_MINUTE)):
            hard_wall_triggered = True

    # --- Check 2: Missed prior-session Hard_Wall ---
    if not hard_wall_triggered:
        recent_hw = most_recent_completed_hard_wall(now_et)
        if recent_hw is not None:
            entry_time = _parse_dt(trade.get("entry_time"))
            if entry_time is not None:
                # Convert hard_wall (ET with tzinfo) to UTC for comparison
                # with entry_time (which is UTC-aware via _parse_dt)
                recent_hw_utc = recent_hw.astimezone(timezone.utc)
                entry_time_utc = entry_time.astimezone(timezone.utc)
                if entry_time_utc < recent_hw_utc:
                    hard_wall_triggered = True

    if not hard_wall_triggered:
        return None

    # --- Hard_Wall triggered: check for valid overnight authorization ---
    auth = latest_valid_overnight_authorization(events, trade, now_utc)

    if auth is not None:
        # Valid auth exists — enforce profile exposure caps
        profile = trade.get("profile", "")
        allowed, reason = check_profile_exposure_cap(
            profile,
            trade,
            current_price,
            current_equity,
            overnight_position_counts,
            overnight_exposure_pcts,
        )
        if not allowed:
            return _build_decision(
                trade,
                now_utc,
                decision="close",
                state="close_required",
                reason_type="profile_cap_exceeded",
                requires_event=True,
                close_reason=reason,
                metadata={
                    "authorization_status": "valid",
                    "cap_reason": reason,
                },
            )
        # Caps OK — return None to continue to stop geometry validation (priority 5)
        return None

    # No valid overnight auth → overnight_unauthorized → close
    return _build_decision(
        trade,
        now_utc,
        decision="close",
        state="overnight_unauthorized",
        reason_type="overnight_authorization_missing",
        requires_event=True,
        close_reason="Position past Hard_Wall without valid overnight authorization",
        metadata={
            "authorization_status": "missing",
            "hard_wall_time": f"{HARD_WALL_HOUR}:{HARD_WALL_MINUTE:02d} ET",
        },
    )


def _check_overnight_stop_geometry(
    trade: dict,
    events: list[dict],
    now_utc: datetime,
    current_price: float | None,
) -> dict | None:
    """Priority 5: Validate stop geometry for overnight carry.

    This check is only reached when a valid overnight authorization exists
    (i.e., _check_hard_wall returned None because auth was found and caps passed).

    If stop geometry is invalid, returns invalid_stop_geometry with repair_stop decision.
    If valid, returns None to continue evaluation.
    """
    # Find the valid overnight auth (we know one exists since _check_hard_wall passed)
    auth = latest_valid_overnight_authorization(events, trade, now_utc)

    if auth is None:
        # Defensive: shouldn't happen since _check_hard_wall passed, but safe fallback
        return None

    # Validate stop geometry against the authorization and current price
    is_valid, reason = validate_stop_geometry(trade, auth, current_price)

    if not is_valid:
        return _build_decision(
            trade,
            now_utc,
            decision="repair_stop",
            state="invalid_stop_geometry",
            reason_type=reason,
            requires_event=True,
            metadata={
                "stop_price": trade.get("stop_price"),
                "current_price": current_price,
                "direction": trade.get("direction"),
                "auth_stop_price": auth.get("stop_price"),
                "reason": reason,
            },
        )

    # Stop geometry is valid — continue to next priority check
    return None


def _check_intraday_force_close(
    trade: dict, events: list[dict], now_utc: datetime
) -> dict | None:
    """Priority 6: Check intraday setup time limits.

    If the position has been held past its force-close limit, return
    intraday_expired with decision close. Does NOT reclassify to swing.
    """
    entry_time = _parse_dt(trade.get("entry_time"))
    if entry_time is None:
        return None

    setup_type = trade.get("setup_type", "") or ""
    limits = SETUP_TIME_LIMITS.get(setup_type, DEFAULT_LIMITS)
    force_close_minutes = limits.get("force_close", DEFAULT_LIMITS["force_close"])

    minutes_held = (now_utc - entry_time).total_seconds() / 60

    if minutes_held > force_close_minutes:
        return _build_decision(
            trade,
            now_utc,
            decision="close",
            state="intraday_expired",
            reason_type="intraday_time_limit_exceeded",
            requires_event=True,
            close_reason=f"Time-based forced exit: {setup_type or 'unknown'} held {round(minutes_held)} min (limit: {force_close_minutes})",
            metadata={
                "minutes_held": round(minutes_held),
                "force_close_limit": force_close_minutes,
                "setup_type": setup_type,
            },
        )

    return None


def _check_intraday_warning(trade: dict, now_utc: datetime) -> dict | None:
    """Priority 7: Check intraday warning windows.

    Returns intraday_warning with decision warn when the position is past
    the alert threshold but before the force-close limit.
    """
    entry_time = _parse_dt(trade.get("entry_time"))
    if entry_time is None:
        return None

    setup_type = trade.get("setup_type", "") or ""
    limits = SETUP_TIME_LIMITS.get(setup_type, DEFAULT_LIMITS)
    alert_minutes = limits.get("alert", DEFAULT_LIMITS["alert"])
    force_close_minutes = limits.get("force_close", DEFAULT_LIMITS["force_close"])

    minutes_held = (now_utc - entry_time).total_seconds() / 60

    # Only warn if past alert threshold but before force-close
    # (force-close is handled by priority 6, so this should only fire
    # when minutes_held > alert but <= force_close)
    if minutes_held > alert_minutes:
        return _build_decision(
            trade,
            now_utc,
            decision="warn",
            state="intraday_warning",
            reason_type="intraday_time_warning",
            requires_event=True,
            metadata={
                "minutes_held": round(minutes_held),
                "alert_limit": alert_minutes,
                "force_limit": force_close_minutes,
                "setup_type": setup_type,
            },
        )

    return None


# ---------------------------------------------------------------------------
# Helper: Check for valid overnight authorization in events
# ---------------------------------------------------------------------------


def _has_valid_overnight_auth(events: list[dict], now_utc: datetime) -> bool:
    """Check if any overnight_authorized event exists with valid expires_at.

    Used by the default case to determine whether to return
    overnight_authorized or intraday_ok state.
    """
    for event in events:
        if event.get("event_type") == "overnight_authorized":
            payload = event.get("payload", {})
            if isinstance(payload, dict):
                expires_at = payload.get("expires_at")
                if expires_at is not None:
                    # Handle both datetime objects and ISO string timestamps
                    if isinstance(expires_at, datetime):
                        if expires_at > now_utc:
                            return True
                    elif isinstance(expires_at, str):
                        try:
                            parsed = datetime.fromisoformat(expires_at)
                            # If naive, assume UTC
                            if parsed.tzinfo is None:
                                parsed = parsed.replace(tzinfo=timezone.utc)
                            if parsed > now_utc:
                                return True
                        except (ValueError, TypeError):
                            continue
    return False


# ---------------------------------------------------------------------------
# Main Evaluator Entry Point
# ---------------------------------------------------------------------------


def evaluate_position_lifecycle(
    trade: dict,
    events: list[dict],
    *,
    now_utc: datetime,
    now_et: datetime,
    current_price: float | None = None,
    current_equity: float | None = None,
    overnight_position_counts: dict[str, int] | None = None,
    overnight_exposure_pcts: dict[str, float] | None = None,
) -> dict:
    """Deterministic lifecycle evaluator. Pure function — no DB writes, no side effects.

    Evaluates a single open position against all governance conditions in
    strict priority order and returns exactly one lifecycle decision.

    Args:
        trade: Dict with keys: id, symbol, profile, direction, entry_price,
               entry_time, stop_price, target_price, setup_type, status, quantity.
        events: List of trade_events dicts for this trade (pre-fetched by caller).
               Each dict has: event_type, timestamp, payload (parsed dict).
        now_utc: Current time in UTC.
        now_et: Current time in US/Eastern (for Hard_Wall evaluation).
        current_price: Latest market price (None if unavailable).
        current_equity: Portfolio equity for exposure calculation.
        overnight_position_counts: {profile: count} of existing overnight positions
            (excluding the current trade).
        overnight_exposure_pcts: {profile: pct} of existing overnight exposure
            (excluding the current trade).

    Returns:
        Dict with all LifecycleDecision fields (decision, state, reason_type,
        trade_id, symbol, profile, setup_type, entry_time, hours_held,
        requires_event, close_reason, metadata).
    """
    # Priority 1: Skip closed/cancelled positions
    result = _check_skip(trade, now_utc)
    if result is not None:
        return result

    # Priority 2: EXIT_NOW governance event
    result = _check_exit_now(events, trade, now_utc)
    if result is not None:
        return result

    # Priority 3: News governance expiry/grace/warning
    result = _check_news_governance(trade, events, now_utc)
    if result is not None:
        return result

    # Priority 4: Hard_Wall + overnight authorization check
    result = _check_hard_wall(
        trade,
        events,
        now_et,
        now_utc,
        current_price,
        current_equity,
        overnight_position_counts,
        overnight_exposure_pcts,
    )
    if result is not None:
        return result

    # Priority 5: Overnight stop geometry validation
    result = _check_overnight_stop_geometry(trade, events, now_utc, current_price)
    if result is not None:
        return result

    # Priority 6: Intraday force-close time limits
    result = _check_intraday_force_close(trade, events, now_utc)
    if result is not None:
        return result

    # Priority 7: Intraday warning windows
    result = _check_intraday_warning(trade, now_utc)
    if result is not None:
        return result

    # Default: No decisive governance condition fired.
    # Determine whether position has valid overnight authorization.
    if _has_valid_overnight_auth(events, now_utc):
        return _build_decision(
            trade,
            now_utc,
            decision="allow",
            state="overnight_authorized",
            reason_type="overnight_authorization_valid",
            requires_event=False,
            metadata={"authorization_status": "valid"},
        )

    return _build_decision(
        trade,
        now_utc,
        decision="allow",
        state="intraday_ok",
        reason_type="within_time_limits",
        requires_event=False,
        metadata={},
    )
