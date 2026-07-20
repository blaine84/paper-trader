"""LLM Queue and Backpressure Layer.

Priority-aware request dispatcher for local Ollama calls.
"""

from __future__ import annotations

from utils.llm_queue.config import QueueConfig
from utils.llm_queue.fallback import FallbackRouter
from utils.llm_queue.models import DispatchResult, FallbackDecision, RequestRecord

from utils.llm_queue.dispatcher import OllamaDispatcher

__all__ = [
    "QueueConfig",
    "RequestRecord",
    "DispatchResult",
    "FallbackDecision",
    "FallbackRouter",
    "OllamaDispatcher",
]
