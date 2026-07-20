"""Request classifier for the LLM queue and backpressure layer.

Maps (purpose, tier, provider, model) into a fully populated RequestRecord
with deterministic classification based on purpose-prefix matching.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from utils.llm_queue.config import QueueConfig, REQUEST_CLASSES
from utils.llm_queue.models import RequestRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Purpose prefix → (agent, request_class) mapping
# Sorted longest prefix first for deterministic matching.
# ---------------------------------------------------------------------------
_PREFIX_TABLE: list[tuple[str, str, str]] = sorted(
    [
        ("pm_entry", "pm", "execution_critical"),
        ("pm_candidate", "pm", "execution_critical"),
        ("pm_maintenance", "pm", "position_management"),
        ("pm_reversal", "pm", "position_management"),
        ("analyst_signal", "analyst", "market_analysis"),
        ("analyst_veto", "analyst", "market_analysis"),
        ("price_monitor_filter", "price_monitor", "market_analysis"),
        ("json_repair", "system", "repair"),
        ("researcher_premarket", "researcher", "research"),
        ("quant_researcher", "quant", "research"),
        ("sector_scout", "scout", "research"),
        ("reviewer_trade", "reviewer", "review"),
        ("daily_review", "reviewer", "review"),
        ("meta_reviewer", "reviewer", "review"),
        ("ceo", "ceo", "advisory"),
        ("weekly_prep", "ceo", "advisory"),
        ("narrator", "narrator", "advisory"),
        ("startup_probe", "system", "startup_probe"),
    ],
    key=lambda entry: len(entry[0]),
    reverse=True,
)

# ---------------------------------------------------------------------------
# Priority by request class (lower = higher priority)
# ---------------------------------------------------------------------------
_PRIORITY_BY_CLASS: dict[str, int] = {
    "execution_critical": 0,
    "market_analysis": 1,
    "position_management": 2,
    "repair": 3,
    "research": 4,
    "review": 5,
    "advisory": 6,
    "startup_probe": 7,
}

# ---------------------------------------------------------------------------
# Fallback policy by request class
# ---------------------------------------------------------------------------
_FALLBACK_POLICY_BY_CLASS: dict[str, str] = {
    "execution_critical": "remote_approved",
    "market_analysis": "smaller_local",
    "position_management": "remote_approved",
    "repair": "remote_approved",
    "research": "defer",
    "review": "defer",
    "advisory": "defer",
    "startup_probe": "none",
}

# ---------------------------------------------------------------------------
# Classes that receive market-hour deadline adjustments
# ---------------------------------------------------------------------------
_MARKET_HOUR_ADJUSTED_CLASSES: frozenset[str] = frozenset({
    "execution_critical",
    "market_analysis",
})


def _get_current_market_session() -> str:
    """Query MarketSessionHelper for the current session.

    Falls back to "closed" if the helper is unavailable (parallel task).
    """
    try:
        from utils.llm_queue.market_session import MarketSessionHelper

        helper = MarketSessionHelper()
        return helper.current_session()
    except (ImportError, Exception) as exc:
        logger.debug(
            "MarketSessionHelper unavailable, defaulting to 'closed': %s", exc
        )
        return "closed"


class RequestClassifier:
    """Maps purpose + tier + provider into a fully populated RequestRecord."""

    def __init__(self, config: QueueConfig) -> None:
        self._config = config

    def classify(
        self,
        purpose: str,
        tier: str,
        provider: str,
        model: str,
        json_mode: bool,
        prompt_chars: int,
        request_class_override: Optional[str] = None,
    ) -> RequestRecord:
        """Classify a request and produce a RequestRecord.

        Parameters
        ----------
        purpose : str
            Caller-supplied purpose label (e.g., "pm_entry_AAPL").
        tier : str
            Original tier from call_llm (e.g., "finance", "medium").
        provider : str
            Resolved provider (e.g., "ollama").
        model : str
            Resolved model name.
        json_mode : bool
            Whether JSON format is requested.
        prompt_chars : int
            Total prompt character count (system + user).
        request_class_override : Optional[str]
            Caller-supplied override for request_class (experimental callers).

        Returns
        -------
        RequestRecord
            Fully populated frozen dataclass with classification metadata.
        """
        # Determine agent and request_class from purpose prefix
        agent, request_class = self._match_purpose(purpose)

        # Apply caller-supplied override if valid
        if request_class_override and request_class_override in REQUEST_CLASSES:
            request_class = request_class_override
            logger.debug(
                "Request class overridden to %r for purpose %r",
                request_class_override,
                purpose,
            )

        # Derive priority from request class
        priority = _PRIORITY_BY_CLASS.get(request_class, 6)

        # Derive fallback policy from request class
        fallback_policy = _FALLBACK_POLICY_BY_CLASS.get(request_class, "defer")

        # Get timing values from config
        deadline_seconds = self._config.deadlines.get(request_class, 900.0)
        max_queue_wait_seconds = self._config.max_queue_waits.get(request_class, 300.0)
        stale_after_seconds = self._config.stale_after.get(request_class, 600.0)

        # Query market session
        market_session = _get_current_market_session()

        # Apply market-hour factors to critical classes during market hours
        if (
            market_session in ("regular", "premarket")
            and request_class in _MARKET_HOUR_ADJUSTED_CLASSES
        ):
            deadline_seconds = deadline_seconds * self._config.market_hour_deadline_factor
            max_queue_wait_seconds = (
                max_queue_wait_seconds * self._config.market_hour_queue_wait_factor
            )

        # Compute approximate prompt tokens
        approx_prompt_tokens = prompt_chars // 4

        # Generate request ID
        request_id = str(uuid.uuid4())

        now = datetime.now(timezone.utc)

        return RequestRecord(
            request_id=request_id,
            purpose=purpose,
            agent=agent,
            request_class=request_class,
            tier=tier,
            provider=provider,
            model=model,
            json_mode=json_mode,
            priority=priority,
            deadline_seconds=deadline_seconds,
            max_queue_wait_seconds=max_queue_wait_seconds,
            fallback_policy=fallback_policy,
            stale_after_seconds=stale_after_seconds,
            created_at=now,
            prompt_chars=prompt_chars,
            approx_prompt_tokens=approx_prompt_tokens,
            market_session=market_session,
        )

    def _match_purpose(self, purpose: str) -> tuple[str, str]:
        """Match purpose against the prefix table (longest prefix first).

        Returns
        -------
        tuple[str, str]
            (agent, request_class). Defaults to ("unknown", "advisory") for
            unrecognized purposes.
        """
        for prefix, agent, request_class in _PREFIX_TABLE:
            if purpose.startswith(prefix):
                return agent, request_class

        # Unknown purpose → advisory class (safe default)
        logger.debug("Unknown purpose %r, defaulting to advisory class", purpose)
        return "unknown", "advisory"
