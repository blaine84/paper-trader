"""Fallback routing for queued LLM requests.

Applies per-class fallback policy when local Ollama execution fails or is
unavailable. Determines whether a request may route to an alternate provider
and selects the appropriate fallback model from the approved list.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from utils.llm_queue.config import QueueConfig
from utils.llm_queue.models import FallbackDecision, RequestRecord

logger = logging.getLogger(__name__)


def _infer_provider(model_name: str) -> str:
    """Infer the provider from a fallback model name.

    Heuristic:
    - Model name contains "claude" → "anthropic"
    - Model name contains "gpt" → "openai"
    - Otherwise → "anthropic" (default)
    """
    lower = model_name.lower()
    if "claude" in lower:
        return "anthropic"
    if "gpt" in lower:
        return "openai"
    return "anthropic"


class FallbackRouter:
    """Applies per-class fallback policy when local execution fails.

    Uses config.approved_fallback_models to determine which models are
    allowed for each request class, and config.fallback_deadline_buffer_seconds
    to verify enough time remains for a fallback call to complete.
    """

    def __init__(self, config: QueueConfig) -> None:
        self._config = config

    def _compute_remaining(self, record: RequestRecord) -> float:
        """Compute seconds remaining before the request's overall deadline."""
        now = datetime.now(timezone.utc)
        elapsed = (now - record.created_at).total_seconds()
        return record.deadline_seconds - elapsed

    def _get_approved_models(self, request_class: str) -> list[str]:
        """Return the list of approved fallback models for a request class."""
        return self._config.approved_fallback_models.get(request_class, [])

    def _first_approved_decision(
        self,
        models: list[str],
        reason: str,
        remaining: float,
    ) -> Optional[FallbackDecision]:
        """Return a FallbackDecision for the first model in the approved list."""
        if not models:
            return None
        model = models[0]
        provider = _infer_provider(model)
        return FallbackDecision(
            provider=provider,
            model=model,
            reason=reason,
            deadline_remaining=remaining,
        )

    def resolve_fallback(
        self, record: RequestRecord, reason: str
    ) -> Optional[FallbackDecision]:
        """Determine fallback routing for a failed or unavailable local request.

        Returns a FallbackDecision if fallback is allowed and viable, or None
        if no fallback is permitted (fail closed / defer / skip).

        Policy by request_class:
        - execution_critical: approved models only, must meet deadline buffer
        - market_analysis: approved models, must still be fresh (deadline check)
        - position_management: approved remote models, standard deadline check
        - repair: any approved model, standard deadline check
        - research/review/advisory: defer preferred; fallback only if deadline imminent
        - startup_probe: never fallback (tests local connectivity)
        """
        request_class = record.request_class
        remaining = self._compute_remaining(record)
        buffer = self._config.fallback_deadline_buffer_seconds

        # startup_probe: no fallback ever
        if request_class == "startup_probe":
            logger.debug(
                "Request %s: no fallback for startup_probe (purpose=%s)",
                record.request_id,
                record.purpose,
            )
            return None

        # execution_critical: approved models only, must meet deadline, else fail closed
        if request_class == "execution_critical":
            approved = self._get_approved_models(request_class)
            if not approved:
                logger.warning(
                    "Request %s: no approved fallback models for execution_critical, "
                    "failing closed (purpose=%s)",
                    record.request_id,
                    record.purpose,
                )
                return None
            if remaining < buffer:
                logger.warning(
                    "Request %s: insufficient time for fallback: %.1fs remaining < %.1fs buffer "
                    "(class=execution_critical, purpose=%s)",
                    record.request_id,
                    remaining,
                    buffer,
                    record.purpose,
                )
                return None
            decision = self._first_approved_decision(approved, reason, remaining)
            logger.info(
                "Request %s: fallback to %s/%s (class=execution_critical, "
                "remaining=%.1fs, reason=%s)",
                record.request_id,
                decision.provider if decision else "none",
                decision.model if decision else "none",
                remaining,
                reason,
            )
            return decision

        # market_analysis: smaller local or approved remote, output must be fresh
        if request_class == "market_analysis":
            approved = self._get_approved_models(request_class)
            if not approved:
                logger.debug(
                    "Request %s: no approved fallback models for market_analysis",
                    record.request_id,
                )
                return None
            if remaining < buffer:
                logger.warning(
                    "Request %s: insufficient time for fallback: %.1fs remaining < %.1fs buffer "
                    "(class=market_analysis, purpose=%s)",
                    record.request_id,
                    remaining,
                    buffer,
                    record.purpose,
                )
                return None
            # Also check freshness: remaining must exceed stale_after threshold
            # to ensure the fallback result will still be usable
            stale_remaining = record.stale_after_seconds - (
                record.deadline_seconds - remaining
            )
            if stale_remaining <= 0:
                logger.warning(
                    "Request %s: fallback result would be stale on arrival "
                    "(class=market_analysis, purpose=%s)",
                    record.request_id,
                    record.purpose,
                )
                return None
            decision = self._first_approved_decision(approved, reason, remaining)
            logger.info(
                "Request %s: fallback to %s/%s (class=market_analysis, "
                "remaining=%.1fs, reason=%s)",
                record.request_id,
                decision.provider if decision else "none",
                decision.model if decision else "none",
                remaining,
                reason,
            )
            return decision

        # position_management: approved remote, standard deadline check
        if request_class == "position_management":
            approved = self._get_approved_models(request_class)
            if not approved:
                logger.debug(
                    "Request %s: no approved fallback models for position_management",
                    record.request_id,
                )
                return None
            if remaining < buffer:
                logger.warning(
                    "Request %s: insufficient time for fallback: %.1fs remaining < %.1fs buffer "
                    "(class=position_management, purpose=%s)",
                    record.request_id,
                    remaining,
                    buffer,
                    record.purpose,
                )
                return None
            decision = self._first_approved_decision(approved, reason, remaining)
            logger.info(
                "Request %s: fallback to %s/%s (class=position_management, "
                "remaining=%.1fs, reason=%s)",
                record.request_id,
                decision.provider if decision else "none",
                decision.model if decision else "none",
                remaining,
                reason,
            )
            return decision

        # repair: any approved model, standard deadline check
        if request_class == "repair":
            approved = self._get_approved_models(request_class)
            if not approved:
                logger.debug(
                    "Request %s: no approved fallback models for repair",
                    record.request_id,
                )
                return None
            if remaining < buffer:
                logger.warning(
                    "Request %s: insufficient time for fallback: %.1fs remaining < %.1fs buffer "
                    "(class=repair, purpose=%s)",
                    record.request_id,
                    remaining,
                    buffer,
                    record.purpose,
                )
                return None
            decision = self._first_approved_decision(approved, reason, remaining)
            logger.info(
                "Request %s: fallback to %s/%s (class=repair, "
                "remaining=%.1fs, reason=%s)",
                record.request_id,
                decision.provider if decision else "none",
                decision.model if decision else "none",
                remaining,
                reason,
            )
            return decision

        # research, review, advisory: defer preferred, fallback only if deadline imminent
        if request_class in ("research", "review", "advisory"):
            # "Imminent" means remaining time is less than max_queue_wait_seconds
            # (i.e., there's not enough time to defer and retry later)
            max_wait = record.max_queue_wait_seconds
            if remaining >= max_wait:
                # Plenty of time — prefer deferral over fallback
                logger.debug(
                    "Request %s: deferring rather than falling back "
                    "(class=%s, remaining=%.1fs >= max_wait=%.1fs)",
                    record.request_id,
                    request_class,
                    remaining,
                    max_wait,
                )
                return None

            # Deadline is imminent — try approved fallback if available
            approved = self._get_approved_models(request_class)
            if not approved:
                logger.debug(
                    "Request %s: no approved fallback models for %s, deferring",
                    record.request_id,
                    request_class,
                )
                return None
            if remaining < buffer:
                logger.warning(
                    "Request %s: insufficient time for fallback: %.1fs remaining < %.1fs buffer "
                    "(class=%s, purpose=%s)",
                    record.request_id,
                    remaining,
                    buffer,
                    request_class,
                    record.purpose,
                )
                return None
            decision = self._first_approved_decision(approved, reason, remaining)
            logger.info(
                "Request %s: fallback to %s/%s (class=%s, deadline imminent, "
                "remaining=%.1fs, reason=%s)",
                record.request_id,
                decision.provider if decision else "none",
                decision.model if decision else "none",
                request_class,
                remaining,
                reason,
            )
            return decision

        # Unrecognized request class — fail closed (no fallback)
        logger.warning(
            "Request %s: unrecognized request_class %r, no fallback",
            record.request_id,
            request_class,
        )
        return None
