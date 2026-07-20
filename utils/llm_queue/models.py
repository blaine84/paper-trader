"""Immutable value objects for the LLM queue and backpressure layer.

Defines the core frozen dataclasses used throughout the dispatcher:
- RequestRecord: structured metadata for every local Ollama request
- DispatchResult: structured result returned by the dispatcher
- FallbackDecision: routing decision when local execution fails
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class RequestRecord:
    """Structured metadata created for every local Ollama request.

    All fields are immutable once classified. The request_id and created_at
    are assigned at classification time; remaining fields derive from the
    purpose/tier/config mapping.
    """

    request_id: str                    # UUID4 string
    purpose: str                       # caller-supplied purpose label
    agent: str                         # inferred agent name (e.g., "pm", "analyst")
    request_class: str                 # classification (e.g., "execution_critical")
    tier: str                          # original tier from call_llm
    provider: str                      # resolved provider ("ollama")
    model: str                         # resolved model name
    json_mode: bool                    # whether JSON format requested
    priority: int                      # numeric priority (lower = higher priority)
    deadline_seconds: float            # max total time from creation to completion
    max_queue_wait_seconds: float      # max time waiting in queue
    fallback_policy: str               # "remote_approved", "smaller_local", "defer", "none"
    stale_after_seconds: float         # result unusable after this window
    created_at: datetime               # when the request was classified
    prompt_chars: int                  # total prompt character count
    approx_prompt_tokens: int          # estimated token count (~chars/4)
    market_session: str                # "regular", "premarket", "postmarket", "closed"


@dataclass(frozen=True)
class DispatchResult:
    """Structured result returned by the dispatcher internally.

    Every request that enters the dispatcher (admitted or rejected) produces
    exactly one DispatchResult. The ok field indicates whether the text is
    usable; status and reason_code provide structured failure information.
    """

    ok: bool                           # True if model returned usable text
    text: str                          # model response text (empty on failure)
    request_id: str                    # correlates to RequestRecord
    status: str                        # "completed", "timed_out", "rejected", "deferred",
                                       # "fallback", "stale", "cancelled", "prompt_too_large"
    reason_code: Optional[str]         # detailed reason (e.g., "queue_full", "deadline_exceeded")
    provider: str                      # provider that actually served the response
    model: str                         # model that actually served the response
    fallback_used: bool                # True if served by fallback provider
    queue_wait_seconds: float          # time spent waiting in queue
    elapsed_seconds: float             # total time from creation to result
    stale: bool                        # True if result completed after stale window


@dataclass(frozen=True)
class FallbackDecision:
    """Routing decision when local execution fails or is unavailable.

    Produced by the FallbackRouter when a request's fallback_policy permits
    routing to an alternate provider/model.
    """

    provider: str                      # fallback provider ("anthropic", "openai")
    model: str                         # fallback model name
    reason: str                        # why fallback triggered
    deadline_remaining: float          # seconds remaining before overall deadline
