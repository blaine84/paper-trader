"""
News Trade Governance Module.

Provides deterministic classification, policy evaluation, reconfirmation validation,
and idempotent event helpers for news-governed trades with a 24-hour maximum hold
duration and structured reconfirmation support.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from db.schema import TradeEvent
from utils.trade_events import log_trade_event

log = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────

NEWS_GOVERNANCE: dict[str, Any] = {
    "enabled": True,
    "max_hold_hours": 24,
    "warning_lead_hours": 4,
    "reconfirm_grace_minutes": 30,
    "min_evidence_length": 20,
    "allow_implicit_swing_reclassify": False,
    "setup_types": {
        "news_catalyst",
        "news_breakout",
        "news_headline",
        "confirmed_strait_of",
        "sector_rotation",
    },
    "entry_text_terms": {
        "catalyst", "headline", "news", "geopolitical", "strait of hormuz",
        "earnings", "guidance", "tariff", "fed", "cpi", "jobs report",
        "oil spike", "vessel", "incident", "escalation",
    },
}

VALID_CATALYST_TYPES = {"news", "geopolitical", "earnings", "macro", "headline"}
VALID_DECISIONS = {"RECONFIRM_AND_HOLD", "EXIT_NOW", "LET_EXPIRE"}
HOLD_BLOCKING_THESIS_STATUSES = {"deteriorating", "invalidated"}
VALID_THESIS_STATUSES = {"strengthened", "unchanged", "deteriorating", "invalidated"}


# ─── Task 2.2: Config Validation ─────────────────────────────────────────────

def validate_news_governance_config(config: dict = NEWS_GOVERNANCE) -> dict:
    """
    Validate the NEWS_GOVERNANCE config dict at startup.

    - Checks config["enabled"] and logs WARNING if disabled.
    - Validates critical fields (max_hold_hours > 0, warning_lead_hours > 0, etc.).
    - Raises ValueError if critically malformed (non-positive durations, missing keys).
    - Returns the config dict on success.

    Called explicitly by position_timer.py at startup (not at import time),
    making it easily testable in isolation.
    """
    required_keys = [
        "enabled", "max_hold_hours", "warning_lead_hours",
        "reconfirm_grace_minutes", "min_evidence_length",
        "allow_implicit_swing_reclassify", "setup_types", "entry_text_terms",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(
                f"NEWS_GOVERNANCE config is missing required key: '{key}'"
            )

    if not config["enabled"]:
        log.warning(
            "NEWS_GOVERNANCE is DISABLED — news-catalyst 24h governance checks "
            "will be skipped. Set enabled=True to enforce hold duration limits."
        )

    if not isinstance(config["max_hold_hours"], (int, float)) or config["max_hold_hours"] <= 0:
        raise ValueError(
            f"NEWS_GOVERNANCE.max_hold_hours must be a positive number, "
            f"got: {config['max_hold_hours']!r}"
        )

    if not isinstance(config["warning_lead_hours"], (int, float)) or config["warning_lead_hours"] <= 0:
        raise ValueError(
            f"NEWS_GOVERNANCE.warning_lead_hours must be a positive number, "
            f"got: {config['warning_lead_hours']!r}"
        )

    if not isinstance(config["reconfirm_grace_minutes"], (int, float)) or config["reconfirm_grace_minutes"] < 0:
        raise ValueError(
            f"NEWS_GOVERNANCE.reconfirm_grace_minutes must be >= 0, "
            f"got: {config['reconfirm_grace_minutes']!r}"
        )

    return config


# ─── Task 2.3: NewsGovernanceClassifier ───────────────────────────────────────

class NewsGovernanceClassifier:
    """Deterministic classifier for news-governed trades."""

    def __init__(self, config: dict = NEWS_GOVERNANCE):
        self.config = config

    def classify(
        self,
        trade: dict,
        *,
        entry_signal: dict | None = None,
    ) -> tuple[bool, dict]:
        """
        Classify whether a trade is news-governed.

        Args:
            trade: Dict with keys: setup_type, reason_entry, thesis,
                   invalidators, catalyst_type (optional).
            entry_signal: Optional dict with entry-time setup_type and metadata.

        Returns:
            (is_governed: bool, evidence: dict)
            evidence contains: triggered_by, matched_field, matched_value, details
        """
        governed_setup_types = self.config.get("setup_types", set())
        entry_text_terms = self.config.get("entry_text_terms", set())

        # 1. setup_type match
        trade_setup_type = trade.get("setup_type", "")
        if trade_setup_type and trade_setup_type in governed_setup_types:
            return True, {
                "triggered_by": "setup_type",
                "matched_field": "setup_type",
                "matched_value": trade_setup_type,
                "details": f"Trade setup_type '{trade_setup_type}' is in governed set",
            }

        # 2. entry signal setup_type match
        if entry_signal and isinstance(entry_signal, dict):
            signal_setup_type = entry_signal.get("setup_type", "")
            if signal_setup_type and signal_setup_type in governed_setup_types:
                return True, {
                    "triggered_by": "entry_signal_setup_type",
                    "matched_field": "entry_signal.setup_type",
                    "matched_value": signal_setup_type,
                    "details": f"Entry signal setup_type '{signal_setup_type}' is in governed set",
                }

        # 3. entry text term match (case-insensitive)
        text_fields = {
            "reason_entry": str(trade.get("reason_entry", "") or ""),
            "thesis": str(trade.get("thesis", "") or ""),
            "invalidators": str(trade.get("invalidators", "") or ""),
        }
        for field_name, field_value in text_fields.items():
            field_lower = field_value.lower()
            for term in entry_text_terms:
                if term.lower() in field_lower:
                    return True, {
                        "triggered_by": "entry_text_terms",
                        "matched_field": field_name,
                        "matched_value": term,
                        "details": f"Term '{term}' found in {field_name} field",
                    }

        # 4. catalyst_type match
        catalyst_type = trade.get("catalyst_type", "")
        if catalyst_type and catalyst_type in VALID_CATALYST_TYPES:
            return True, {
                "triggered_by": "catalyst_type",
                "matched_field": "catalyst_type",
                "matched_value": catalyst_type,
                "details": f"catalyst_type '{catalyst_type}' is in valid catalyst types",
            }

        return False, {}

    def classify_from_persisted_evidence(self, evidence: dict) -> bool:
        """Re-evaluate governance from previously persisted evidence dict."""
        if evidence and isinstance(evidence, dict) and "triggered_by" in evidence:
            return True
        return False

    def get_persisted_classification(self, db, trade_id: int) -> dict | None:
        """
        Check for an existing persisted news_governance_classified event.

        Returns:
            dict with {"evidence": {...}, "classified_at": datetime} if persisted,
            None if no prior classification exists.
        """
        event = (
            db.query(TradeEvent)
            .filter_by(event_type="news_governance_classified", trade_id=trade_id)
            .order_by(TradeEvent.timestamp.desc())
            .first()
        )
        if event is None:
            return None

        try:
            payload = json.loads(event.payload_json) if event.payload_json else {}
        except (json.JSONDecodeError, TypeError):
            payload = {}

        return {
            "evidence": payload,
            "classified_at": event.timestamp,
        }


# ─── Task 2.6: Dedupe Key Helpers ────────────────────────────────────────────

def _build_dedupe_key(
    event_type: str, trade_id: int, governance_window_id: int | None
) -> str:
    """
    Build a deterministic deduplication key.
    Format: "{event_type}:{trade_id}:window:{governance_window_id}"
    If governance_window_id is None, format: "{event_type}:{trade_id}"
    """
    if governance_window_id is not None:
        return f"{event_type}:{trade_id}:window:{governance_window_id}"
    return f"{event_type}:{trade_id}"


def _build_failure_dedupe_key(
    event_type: str,
    trade_id: int,
    governance_window_id: int | None,
    now_utc: datetime,
) -> str:
    """
    Build an hourly-bucketed dedupe key for failure events.
    Format: "{event_type}:{trade_id}:window:{gw_id}:{YYYYMMDDHH}"
    Records at most one failure event per hour per (trade_id, window).
    """
    hour_bucket = now_utc.strftime("%Y%m%d%H")
    if governance_window_id is not None:
        return f"{event_type}:{trade_id}:window:{governance_window_id}:{hour_bucket}"
    return f"{event_type}:{trade_id}:{hour_bucket}"


# ─── Task 2.7: log_trade_event_once ──────────────────────────────────────────

def log_trade_event_once(
    db,
    event_type: str,
    trade_id: int,
    *,
    governance_window_id: int | None = None,
    agent: str | None = None,
    symbol: str | None = None,
    profile: str | None = None,
    price: float | None = None,
    message: str | None = None,
    payload: dict | None = None,
) -> bool:
    """
    Write a trade event only if no event with the same dedupe_key exists.

    For `news_expiry_force_close_failed` events, uses hourly-bucketed dedupe
    (at most one per hour per trade/window) so persistent failures stay visible.

    Returns True if event was written, False if suppressed as duplicate.
    """
    # Build the appropriate dedupe key
    if event_type == "news_expiry_force_close_failed":
        now_utc = datetime.now(timezone.utc)
        dedupe_key = _build_failure_dedupe_key(
            event_type, trade_id, governance_window_id, now_utc
        )
    else:
        dedupe_key = _build_dedupe_key(event_type, trade_id, governance_window_id)

    # Check for existing event with same dedupe_key
    existing = (
        db.query(TradeEvent)
        .filter_by(event_type=event_type, trade_id=trade_id, dedupe_key=dedupe_key)
        .first()
    )
    if existing:
        return False

    # Inject governance_window_id into payload for audit visibility
    if payload is None:
        payload = {}
    if governance_window_id is not None:
        payload["governance_window_id"] = governance_window_id

    # Create the event using log_trade_event and set dedupe_key on the returned object
    event = log_trade_event(
        db,
        event_type,
        trade_id=trade_id,
        agent=agent,
        symbol=symbol,
        profile=profile,
        price=price,
        message=message,
        payload=payload,
    )
    event.dedupe_key = dedupe_key
    return True


# ─── Task 2.8: Reconfirmation Query Helpers ──────────────────────────────────

def latest_valid_reconfirmation(db, trade_id: int) -> dict | None:
    """
    Read the latest valid hold-authorizing reconfirmation from trade_events.

    Queries event_type='news_reconfirmation_submitted' ordered by timestamp DESC.
    Returns the first event with decision='RECONFIRM_AND_HOLD'.
    LET_EXPIRE and EXIT_NOW are NOT returned as hold authorizations.

    Returns parsed payload dict or None.
    """
    events = (
        db.query(TradeEvent)
        .filter_by(event_type="news_reconfirmation_submitted", trade_id=trade_id)
        .order_by(TradeEvent.timestamp.desc())
        .all()
    )
    for event in events:
        try:
            payload = json.loads(event.payload_json) if event.payload_json else {}
        except (json.JSONDecodeError, TypeError):
            continue
        if payload.get("decision") == "RECONFIRM_AND_HOLD":
            return payload
    return None


def latest_exit_request(db, trade_id: int) -> dict | None:
    """
    Read the latest EXIT_NOW reconfirmation from trade_events.
    Returns parsed payload dict or None.

    IMPORTANT: Callers MUST check trade.status == "open" before acting
    on the returned EXIT_NOW event.
    """
    events = (
        db.query(TradeEvent)
        .filter_by(event_type="news_reconfirmation_submitted", trade_id=trade_id)
        .order_by(TradeEvent.timestamp.desc())
        .all()
    )
    for event in events:
        try:
            payload = json.loads(event.payload_json) if event.payload_json else {}
        except (json.JSONDecodeError, TypeError):
            continue
        if payload.get("decision") == "EXIT_NOW":
            return payload
    return None


# ─── Helper: Datetime Parsing ─────────────────────────────────────────────────

def _ensure_aware(dt: datetime | str | None) -> datetime | None:
    """
    Ensure a datetime is timezone-aware (UTC). Handles:
    - datetime objects (naive → UTC, aware → as-is)
    - ISO format strings → parsed to UTC-aware datetime
    - None → None
    """
    if dt is None:
        return None
    if isinstance(dt, str):
        try:
            parsed = datetime.fromisoformat(dt)
        except (ValueError, TypeError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    return None


# ─── Task 2.4: NewsGovernancePolicy ──────────────────────────────────────────

class NewsGovernancePolicy:
    """Computes governance window state for a news-governed trade."""

    def __init__(self, config: dict = NEWS_GOVERNANCE):
        self.config = config

    def get_original_expiry(self, entry_time: datetime) -> datetime:
        """Compute entry_time + max_hold_hours (UTC-aware)."""
        entry = _ensure_aware(entry_time) or entry_time
        max_hours = self.config.get("max_hold_hours", 24)
        return entry + timedelta(hours=max_hours)

    def get_effective_expiry(
        self,
        entry_time: datetime,
        latest_reconfirmation: dict | None,
    ) -> datetime:
        """
        Return new_expiry_time from latest valid reconfirmation,
        or original expiry if none exists.
        """
        if latest_reconfirmation and "new_expiry_time" in latest_reconfirmation:
            new_expiry = latest_reconfirmation["new_expiry_time"]
            parsed = _ensure_aware(new_expiry)
            if parsed is not None:
                return parsed
        return self.get_original_expiry(entry_time)

    def get_governance_window_id(self, db, trade_id: int) -> int:
        """
        Return current governance window ID.
        1 = original window, increments with each valid RECONFIRM_AND_HOLD.
        """
        events = (
            db.query(TradeEvent)
            .filter_by(event_type="news_reconfirmation_submitted", trade_id=trade_id)
            .all()
        )
        count = 0
        for event in events:
            try:
                payload = json.loads(event.payload_json) if event.payload_json else {}
            except (json.JSONDecodeError, TypeError):
                continue
            if payload.get("decision") == "RECONFIRM_AND_HOLD":
                count += 1
        return count + 1

    def evaluate(
        self,
        db,
        trade_id: int,
        entry_time: datetime,
        now_utc: datetime,
    ) -> dict:
        """
        Evaluate the current governance status for a trade.

        Returns dict with:
            status: "ok" | "warning" | "grace" | "expired" | "exit_requested"
            hold_authorized: bool
            effective_expiry: datetime
            governance_window_id: int
            warning_time: datetime
            force_close_time: datetime
            latest_reconfirmation: dict | None
        """
        entry = _ensure_aware(entry_time) or entry_time
        now = _ensure_aware(now_utc) or now_utc

        reconfirmation = latest_valid_reconfirmation(db, trade_id)
        exit_req = latest_exit_request(db, trade_id)

        effective_expiry = self.get_effective_expiry(entry, reconfirmation)
        window_id = self.get_governance_window_id(db, trade_id)

        warning_lead_hours = self.config.get("warning_lead_hours", 4)
        grace_minutes = self.config.get("reconfirm_grace_minutes", 30)

        warning_time = effective_expiry - timedelta(hours=warning_lead_hours)
        force_close_time = effective_expiry + timedelta(minutes=grace_minutes)

        has_valid_hold = reconfirmation is not None
        has_exit_request = exit_req is not None

        status_result = self.compute_status(
            now_utc=now,
            effective_expiry=effective_expiry,
            grace_minutes=grace_minutes,
            warning_lead_hours=warning_lead_hours,
            has_valid_hold=has_valid_hold,
            has_exit_request=has_exit_request,
        )

        return {
            "status": status_result["status"],
            "hold_authorized": status_result["hold_authorized"],
            "effective_expiry": effective_expiry,
            "governance_window_id": window_id,
            "warning_time": warning_time,
            "force_close_time": force_close_time,
            "latest_reconfirmation": reconfirmation,
        }

    def compute_status(
        self,
        now_utc: datetime,
        effective_expiry: datetime,
        grace_minutes: int,
        warning_lead_hours: int,
        has_valid_hold: bool,
        has_exit_request: bool,
    ) -> dict:
        """
        Pure function: compute status dict from timestamps and flags.
        Separated for unit testing without DB.

        Returns dict with:
            status: "ok" | "warning" | "grace" | "expired" | "exit_requested"
            hold_authorized: bool
        """
        now = _ensure_aware(now_utc) or now_utc
        expiry = _ensure_aware(effective_expiry) or effective_expiry

        warning_threshold = expiry - timedelta(hours=warning_lead_hours)
        grace_deadline = expiry + timedelta(minutes=grace_minutes)

        # Exit request takes priority
        if has_exit_request:
            return {"status": "exit_requested", "hold_authorized": has_valid_hold}

        # Time-based status
        if now < warning_threshold:
            status = "ok"
        elif now < expiry:
            status = "warning"
        elif now < grace_deadline:
            status = "grace"
        else:
            status = "expired"

        return {"status": status, "hold_authorized": has_valid_hold}


# ─── Task 2.5: ReconfirmationValidator ───────────────────────────────────────

class ReconfirmationValidator:
    """Validates reconfirmation payloads for schema, evidence quality, and temporal constraints."""

    def __init__(self, config: dict = NEWS_GOVERNANCE):
        self.config = config

    def validate(
        self,
        payload: dict,
        *,
        entry_time: datetime,
        prior_reconfirmation: dict | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Validate a reconfirmation payload.

        Args:
            payload: The reconfirmation decision payload.
            entry_time: Trade entry time (UTC-aware).
            prior_reconfirmation: Previous reconfirmation if any (for temporal checks).

        Returns:
            (is_valid: bool, errors: list[str])
        """
        errors: list[str] = []

        # Common required fields
        common_required = ["trade_id", "symbol", "profile", "decision", "decided_by"]
        for field in common_required:
            if not payload.get(field):
                errors.append(f"Missing required field: '{field}'")

        decision = payload.get("decision", "")

        # Validate decision enum
        if decision and decision not in VALID_DECISIONS:
            errors.append(
                f"Invalid decision '{decision}'. "
                f"Must be one of: {', '.join(sorted(VALID_DECISIONS))}"
            )
            # Can't validate decision-specific fields with invalid decision
            return False, errors

        # Decision-specific required fields
        if decision == "RECONFIRM_AND_HOLD":
            hold_required = [
                "original_catalyst", "fresh_catalyst_evidence",
                "fresh_catalyst_timestamp", "thesis_status",
                "new_expiry_time", "risk_plan",
            ]
            for field in hold_required:
                if not payload.get(field):
                    errors.append(
                        f"Missing required field for RECONFIRM_AND_HOLD: '{field}'"
                    )

        elif decision == "EXIT_NOW":
            if not payload.get("exit_reason"):
                errors.append("Missing required field for EXIT_NOW: 'exit_reason'")

        elif decision == "LET_EXPIRE":
            if not payload.get("decline_reason"):
                errors.append("Missing required field for LET_EXPIRE: 'decline_reason'")

        # If there are already schema errors, return early
        if errors:
            return False, errors

        # Business rules for RECONFIRM_AND_HOLD
        if decision == "RECONFIRM_AND_HOLD":
            errors.extend(self._validate_hold_business_rules(
                payload, entry_time=entry_time, prior_reconfirmation=prior_reconfirmation
            ))

        # Swing authorization validation
        if payload.get("allow_swing_reclassify"):
            if decision != "RECONFIRM_AND_HOLD":
                errors.append(
                    "allow_swing_reclassify=True is only valid with "
                    "decision='RECONFIRM_AND_HOLD'"
                )
            else:
                if not payload.get("authorized_setup_type"):
                    errors.append(
                        "allow_swing_reclassify=True requires non-empty "
                        "'authorized_setup_type'"
                    )
                if not payload.get("reclassification_reason"):
                    errors.append(
                        "allow_swing_reclassify=True requires non-empty "
                        "'reclassification_reason'"
                    )

        return (len(errors) == 0), errors

    def _validate_hold_business_rules(
        self,
        payload: dict,
        *,
        entry_time: datetime,
        prior_reconfirmation: dict | None = None,
    ) -> list[str]:
        """Validate business rules specific to RECONFIRM_AND_HOLD."""
        errors: list[str] = []
        max_hold_hours = self.config.get("max_hold_hours", 24)
        min_evidence_length = self.config.get("min_evidence_length", 20)

        # thesis_status must not be in blocking set
        thesis_status = payload.get("thesis_status", "")
        if thesis_status in HOLD_BLOCKING_THESIS_STATUSES:
            errors.append(
                f"Cannot RECONFIRM_AND_HOLD with thesis_status='{thesis_status}'. "
                f"Blocking statuses: {sorted(HOLD_BLOCKING_THESIS_STATUSES)}"
            )

        # new_expiry_time must be timezone-aware
        new_expiry_raw = payload.get("new_expiry_time")
        new_expiry = _ensure_aware(new_expiry_raw)
        if new_expiry_raw is not None and new_expiry is not None:
            # Check if the original value was a naive datetime object
            if isinstance(new_expiry_raw, datetime) and new_expiry_raw.tzinfo is None:
                errors.append(
                    "new_expiry_time must be timezone-aware (not naive datetime)"
                )
        elif new_expiry_raw is not None and new_expiry is None:
            errors.append("new_expiry_time could not be parsed as a valid datetime")

        # fresh_catalyst_timestamp must be timezone-aware
        fresh_ts_raw = payload.get("fresh_catalyst_timestamp")
        fresh_ts = _ensure_aware(fresh_ts_raw)
        if fresh_ts_raw is not None and fresh_ts is not None:
            if isinstance(fresh_ts_raw, datetime) and fresh_ts_raw.tzinfo is None:
                errors.append(
                    "fresh_catalyst_timestamp must be timezone-aware (not naive datetime)"
                )
        elif fresh_ts_raw is not None and fresh_ts is None:
            errors.append(
                "fresh_catalyst_timestamp could not be parsed as a valid datetime"
            )

        # new_expiry_time <= decided_at + max_hold_hours
        if new_expiry is not None:
            decided_at_raw = payload.get("decided_at")
            decided_at = _ensure_aware(decided_at_raw)
            if decided_at is None:
                decided_at = datetime.now(timezone.utc)
            max_allowed_expiry = decided_at + timedelta(hours=max_hold_hours)
            if new_expiry > max_allowed_expiry:
                errors.append(
                    f"new_expiry_time exceeds maximum allowed "
                    f"(decided_at + {max_hold_hours}h). "
                    f"Max allowed: {max_allowed_expiry.isoformat()}, "
                    f"got: {new_expiry.isoformat()}"
                )

        # fresh_catalyst_evidence length >= min_evidence_length
        evidence_text = payload.get("fresh_catalyst_evidence", "")
        if isinstance(evidence_text, str) and len(evidence_text) < min_evidence_length:
            errors.append(
                f"fresh_catalyst_evidence must be at least {min_evidence_length} "
                f"characters, got {len(evidence_text)}"
            )

        # fresh_catalyst_timestamp must be after entry_time
        entry = _ensure_aware(entry_time)
        if fresh_ts is not None and entry is not None:
            if fresh_ts <= entry:
                errors.append(
                    "fresh_catalyst_timestamp must be after trade entry_time"
                )

        # If prior_reconfirmation exists, fresh_catalyst_timestamp must be after prior's
        if prior_reconfirmation and fresh_ts is not None:
            prior_ts_raw = prior_reconfirmation.get("fresh_catalyst_timestamp")
            prior_ts = _ensure_aware(prior_ts_raw)
            if prior_ts is not None and fresh_ts <= prior_ts:
                errors.append(
                    "fresh_catalyst_timestamp must be after prior reconfirmation's "
                    "fresh_catalyst_timestamp"
                )

        return errors


# ─── Task 2.9: submit_news_reconfirmation ────────────────────────────────────

def submit_news_reconfirmation(
    db,
    payload: dict,
    *,
    entry_time: datetime,
    prior_reconfirmation: dict | None = None,
) -> tuple[bool, dict | list[str]]:
    """
    Validate and persist a reconfirmation decision.

    On success: writes news_reconfirmation_submitted + decision-specific event.
    On failure: returns validation errors without persisting.

    Returns:
        (success: bool, result: dict on success | list[str] errors on failure)
    """
    # Set decided_at to current UTC timestamp
    payload["decided_at"] = datetime.now(timezone.utc)

    # Validate via ReconfirmationValidator
    validator = ReconfirmationValidator()
    is_valid, errors = validator.validate(
        payload, entry_time=entry_time, prior_reconfirmation=prior_reconfirmation
    )

    if not is_valid:
        return False, errors

    # Write the main reconfirmation event
    log_trade_event(
        db,
        "news_reconfirmation_submitted",
        trade_id=payload["trade_id"],
        agent=payload.get("decided_by"),
        symbol=payload.get("symbol"),
        profile=payload.get("profile"),
        payload=payload,
    )

    decision = payload["decision"]

    # Write decision-specific events
    if decision == "RECONFIRM_AND_HOLD":
        log_trade_event(
            db,
            "news_hold_authorized",
            trade_id=payload["trade_id"],
            agent=payload.get("decided_by"),
            symbol=payload.get("symbol"),
            profile=payload.get("profile"),
            payload={
                "trade_id": payload["trade_id"],
                "symbol": payload.get("symbol"),
                "profile": payload.get("profile"),
                "new_expiry_time": payload.get("new_expiry_time"),
                "decided_by": payload.get("decided_by"),
                "fresh_catalyst_evidence": payload.get("fresh_catalyst_evidence"),
                "fresh_catalyst_timestamp": payload.get("fresh_catalyst_timestamp"),
                "thesis_status": payload.get("thesis_status"),
                "governance_window_id": payload.get("governance_window_id"),
            },
        )

    elif decision == "EXIT_NOW":
        log_trade_event(
            db,
            "news_exit_requested",
            trade_id=payload["trade_id"],
            agent=payload.get("decided_by"),
            symbol=payload.get("symbol"),
            profile=payload.get("profile"),
            payload={
                "trade_id": payload["trade_id"],
                "symbol": payload.get("symbol"),
                "profile": payload.get("profile"),
                "exit_reason": payload.get("exit_reason"),
                "decided_by": payload.get("decided_by"),
                "decided_at": payload.get("decided_at"),
                "governance_window_id": payload.get("governance_window_id"),
                "close_confirmed": True,
            },
        )

    elif decision == "LET_EXPIRE":
        log_trade_event(
            db,
            "news_let_expire_acknowledged",
            trade_id=payload["trade_id"],
            agent=payload.get("decided_by"),
            symbol=payload.get("symbol"),
            profile=payload.get("profile"),
            payload={
                "trade_id": payload["trade_id"],
                "symbol": payload.get("symbol"),
                "profile": payload.get("profile"),
                "decline_reason": payload.get("decline_reason"),
                "decided_by": payload.get("decided_by"),
                "decided_at": payload.get("decided_at"),
                "governance_window_id": payload.get("governance_window_id"),
            },
        )

    # Handle swing reclassification authorization
    if payload.get("allow_swing_reclassify"):
        log_trade_event(
            db,
            "news_swing_reclassification_authorized",
            trade_id=payload["trade_id"],
            agent=payload.get("decided_by"),
            symbol=payload.get("symbol"),
            profile=payload.get("profile"),
            payload={
                "trade_id": payload["trade_id"],
                "symbol": payload.get("symbol"),
                "profile": payload.get("profile"),
                "authorized_setup_type": payload.get("authorized_setup_type"),
                "reclassification_reason": payload.get("reclassification_reason"),
                "decided_by": payload.get("decided_by"),
                "governance_window_id": payload.get("governance_window_id"),
            },
        )

    db.commit()
    return True, payload
