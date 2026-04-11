"""
LLM abstraction layer.
Supports OpenAI, Anthropic, Mistral, and Ollama.

Tiers:
  "high" (default) — primary provider set by LLM_PROVIDER / LLM_MODEL
  "low"            — cheap/local provider set by LLM_LOW_PROVIDER / LLM_LOW_MODEL

Set in .env:
  LLM_LOW_PROVIDER=mistral   # or: ollama
  LLM_LOW_MODEL=mistral-small-latest
  MISTRAL_API_KEY=...
  OLLAMA_BASE_URL=http://localhost:11434  # default
  OLLAMA_MODEL=llama3                     # fallback if LLM_LOW_MODEL not set
"""

import os
import json
import time
import logging

log = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_BASE_DELAY = 10  # seconds


def _with_retry(fn):
    """Retry on rate limit / overload errors with exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = "rate limit" in err or "429" in err or "overloaded" in err
            if is_rate_limit and attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                log.warning(f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(delay)
            else:
                raise


def call_llm(system_prompt: str, user_prompt: str, json_mode: bool = False, tier: str = "high") -> str:
    """
    Call the LLM.
    tier="high"   — LLM_PROVIDER/LLM_MODEL (Sonnet for critical decisions)
    tier="medium" — LLM_MED_PROVIDER/LLM_MED_MODEL (local 8b for heavy local work)
    tier="low"    — LLM_LOW_PROVIDER/LLM_LOW_MODEL (fast local for simple tasks)
    """
    if tier == "medium":
        provider = os.getenv("LLM_MED_PROVIDER", os.getenv("LLM_LOW_PROVIDER", os.getenv("LLM_PROVIDER", "openai"))).lower()
        model = os.getenv("LLM_MED_MODEL", os.getenv("LLM_LOW_MODEL", None))
    elif tier == "low":
        provider = os.getenv("LLM_LOW_PROVIDER", os.getenv("LLM_PROVIDER", "openai")).lower()
        model = os.getenv("LLM_LOW_MODEL", None)
    else:
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        model = None

    if provider == "openai":
        return _call_openai(system_prompt, user_prompt, json_mode, model)
    elif provider == "anthropic":
        return _call_anthropic(system_prompt, user_prompt, model)
    elif provider == "mistral":
        return _call_mistral(system_prompt, user_prompt, json_mode, model)
    elif provider == "ollama":
        return _call_ollama(system_prompt, user_prompt, model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def _call_openai(system_prompt: str, user_prompt: str, json_mode: bool, model: str = None) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    return _with_retry(lambda: client.chat.completions.create(**kwargs).choices[0].message.content)


def _call_anthropic(system_prompt: str, user_prompt: str, model: str = None) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    model = model or os.getenv("LLM_MODEL", "claude-3-5-haiku-latest")

    for max_tokens in [4096, 8192]:
        response = _with_retry(lambda: client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ))
        if response.stop_reason == "max_tokens":
            log.warning(f"Anthropic response truncated at {max_tokens} tokens, retrying with {max_tokens * 2}")
            continue
        return response.content[0].text

    log.error("Anthropic response truncated even at max tokens")
    return response.content[0].text


def _call_mistral(system_prompt: str, user_prompt: str, json_mode: bool, model: str = None) -> str:
    from mistralai import Mistral
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    model = model or os.getenv("LLM_MODEL", "mistral-small-latest")

    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    return _with_retry(lambda: client.chat.complete(**kwargs).choices[0].message.content)


def _call_ollama(system_prompt: str, user_prompt: str, model: str = None) -> str:
    import requests
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = model or os.getenv("OLLAMA_MODEL", "llama3")
    timeout = int(os.getenv("OLLAMA_TIMEOUT", 60))

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }

    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except Exception as e:
        fallback_model = os.getenv("OLLAMA_FALLBACK_MODEL", "claude-haiku-4-5")
        fallback_provider = os.getenv("OLLAMA_FALLBACK_PROVIDER", "anthropic")
        log.warning(f"Ollama failed ({e}), falling back to {fallback_provider}/{fallback_model}")
        if fallback_provider == "anthropic":
            return _call_anthropic(system_prompt, user_prompt, fallback_model)
        elif fallback_provider == "openai":
            return _call_openai(system_prompt, user_prompt, False, fallback_model)
        else:
            raise


def parse_json_response(text: str) -> dict:
    """Safely parse JSON from LLM output, even with markdown fences."""
    if not text or not text.strip():
        raise ValueError("LLM returned empty response")
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    # Try to extract JSON from mixed text
    if not text.startswith("{") and "{" in text:
        start = text.index("{")
        end = text.rindex("}") + 1
        text = text[start:end]
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM JSON response: {e}\nRaw: {text[:500]}")
