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
import threading
import requests

log = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_BASE_DELAY = 10  # seconds
_OLLAMA_REQUEST_LOCK = threading.Lock()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _acquire_ollama_slot(purpose: str, model: str) -> tuple[bool, float]:
    """Serialize local Ollama calls so scheduler jobs do not stampede one model server."""
    if not _env_bool("OLLAMA_SERIALIZE_REQUESTS", True):
        return False, 0.0

    queue_timeout = float(os.getenv("OLLAMA_QUEUE_TIMEOUT", os.getenv("OLLAMA_TIMEOUT", "300")))
    wait_started = time.monotonic()
    acquired = _OLLAMA_REQUEST_LOCK.acquire(timeout=max(queue_timeout, 0.0))
    waited = time.monotonic() - wait_started
    if not acquired:
        raise TimeoutError(
            f"Ollama queue timeout after {waited:.1f}s "
            f"(purpose={purpose}, model={model}, queue_timeout={queue_timeout})"
        )
    if waited >= 1.0:
        log.info(
            "Ollama queue acquired after %.1fs: purpose=%s model=%s",
            waited,
            purpose,
            model,
        )
    return True, waited


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


def call_llm(system_prompt: str, user_prompt: str, json_mode: bool = False, tier: str = "high", purpose: str = None) -> str:
    """
    Call the LLM.
    tier="high"    — LLM_PROVIDER/LLM_MODEL (Sonnet for critical decisions)
    tier="finance" — LLM_FINANCE_PROVIDER/LLM_FINANCE_MODEL (fin-llama3.1-8b for financial reasoning)
    tier="medium"  — LLM_MED_PROVIDER/LLM_MED_MODEL (local 8b for heavy local work)
    tier="low"     — LLM_LOW_PROVIDER/LLM_LOW_MODEL (fast local for simple tasks)
    """
    start = time.time()

    if tier == "finance":
        provider = os.getenv("LLM_FINANCE_PROVIDER", "").lower()
        model = os.getenv("LLM_FINANCE_MODEL", None)
        if not provider:
            log.warning("Finance tier not configured (LLM_FINANCE_PROVIDER not set), falling back to medium")
            return call_llm(system_prompt, user_prompt, json_mode, tier="medium", purpose=purpose)
    elif tier == "medium":
        provider = os.getenv("LLM_MED_PROVIDER", os.getenv("LLM_LOW_PROVIDER", os.getenv("LLM_PROVIDER", "openai"))).lower()
        model = os.getenv("LLM_MED_MODEL", os.getenv("LLM_LOW_MODEL", None))
    elif tier == "low":
        provider = os.getenv("LLM_LOW_PROVIDER", os.getenv("LLM_PROVIDER", "openai")).lower()
        model = os.getenv("LLM_LOW_MODEL", None)
    else:
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        model = None

    # Track actual provider/model for logging (may change on fallback)
    actual_provider = provider
    actual_model = model

    if tier == "finance" and provider == "ollama":
        # Use finance-specific wrapper with fallback to medium tier
        result, actual_provider, actual_model = _call_ollama_finance(
            system_prompt, user_prompt, model, purpose=purpose or "unlabeled"
        )
    else:
        dispatch = {
            "openai": lambda: _call_openai(system_prompt, user_prompt, json_mode, model),
            "anthropic": lambda: _call_anthropic(system_prompt, user_prompt, model),
            "mistral": lambda: _call_mistral(system_prompt, user_prompt, json_mode, model),
            "ollama": lambda: _call_ollama(system_prompt, user_prompt, model, purpose=purpose or "unlabeled"),
        }
        if provider not in dispatch:
            raise ValueError(f"Unknown LLM provider: {provider}")

        fn = dispatch[provider]

        # Retry once on empty response (local models sometimes return nothing on overload)
        result = None
        for attempt in range(2):
            result = fn()
            if result and result.strip():
                break
            if attempt == 0:
                log.warning("LLM returned empty response (provider=%s, tier=%s), retrying once", provider, tier)
                time.sleep(2)

        result = result or ""

    elapsed = time.time() - start
    log.info("LLM call: tier=%s provider=%s model=%s elapsed=%.1fs", tier, actual_provider, actual_model, elapsed)

    return result


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

    for max_tokens in [4096, 8192, 16384]:
        response = _with_retry(lambda: client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ))
        if response.stop_reason == "max_tokens" and max_tokens < 16384:
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


def _call_ollama_finance(system_prompt: str, user_prompt: str, model: str = None, purpose: str = "unlabeled") -> tuple:
    """Call Ollama for finance tier with fallback to medium tier on failure.

    Returns a tuple of (response_text, actual_provider, actual_model) so the
    caller can log the real model that served the response.
    """
    timeout = int(os.getenv("OLLAMA_FINANCE_TIMEOUT", os.getenv("OLLAMA_TIMEOUT", 600)))
    num_ctx = int(os.getenv("OLLAMA_FINANCE_NUM_CTX", os.getenv("OLLAMA_NUM_CTX", 8192)))
    try:
        result = _call_ollama(
            system_prompt,
            user_prompt,
            model,
            purpose=purpose,
            timeout=timeout,
            num_ctx=num_ctx,
        )
        return (result, "ollama", model)
    except Exception as e:
        med_model = os.getenv("LLM_MED_MODEL", os.getenv("LLM_LOW_MODEL", None))
        log.warning(
            "Finance tier failed (%s), falling back to medium tier (model=%s)",
            e, med_model,
        )
        result = _call_ollama(system_prompt, user_prompt, med_model, purpose=purpose)
        return (result, "ollama", med_model)


def _call_ollama(
    system_prompt: str,
    user_prompt: str,
    model: str = None,
    purpose: str = "unlabeled",
    timeout: int = None,
    num_ctx: int = None,
) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = model or os.getenv("OLLAMA_MODEL", "llama3")
    timeout = timeout or int(os.getenv("OLLAMA_TIMEOUT", 600))
    num_ctx = num_ctx or int(os.getenv("OLLAMA_NUM_CTX", 8192))

    json_instruction = "\n\nYou MUST respond with valid JSON only. No markdown, no explanation, no preamble."
    system_content = system_prompt + json_instruction

    # Prompt-size telemetry: useful for diagnosing local model timeouts/context bloat.
    # Approx token estimate is intentionally rough and cheap: ~4 chars/token.
    system_chars = len(system_content)
    user_chars = len(user_prompt)
    total_chars = system_chars + user_chars
    approx_tokens = max(1, total_chars // 4)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "num_ctx": num_ctx,
            "temperature": 0.2,
        },
    }

    log.info(
        "Ollama request starting: purpose=%s model=%s base_url=%s timeout=%ss num_ctx=%s prompt_chars=%s "
        "system_chars=%s user_chars=%s approx_tokens=%s",
        purpose,
        model,
        base_url,
        timeout,
        num_ctx,
        total_chars,
        system_chars,
        user_chars,
        approx_tokens,
    )
    chunk_threshold = int(os.getenv("OLLAMA_CHUNK_WARN_TOKENS", "8000"))
    if approx_tokens > chunk_threshold:
        log.warning(
            "Ollama oversized prompt: purpose=%s model=%s approx_tokens=%s prompt_chars=%s "
            "threshold=%s — chunking candidate",
            purpose,
            model,
            approx_tokens,
            total_chars,
            chunk_threshold,
        )
    started = time.monotonic()
    lock_acquired = False

    try:
        lock_acquired, queue_wait_seconds = _acquire_ollama_slot(purpose, model)
        if queue_wait_seconds:
            log.info(
                "Ollama request dequeued: purpose=%s model=%s queue_wait=%.1fs",
                purpose,
                model,
                queue_wait_seconds,
            )
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
        elapsed = time.monotonic() - started
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        log.info(
            "Ollama request completed: purpose=%s model=%s elapsed=%.1fs response_chars=%s",
            purpose,
            model,
            elapsed,
            len(content or ""),
        )
        return content
    except Exception as e:
        elapsed = time.monotonic() - started
        fallback_model = os.getenv("OLLAMA_FALLBACK_MODEL", "claude-haiku-4-5")
        fallback_provider = os.getenv("OLLAMA_FALLBACK_PROVIDER", "anthropic")
        response_text = getattr(getattr(e, "response", None), "text", "")
        log.warning(
            "Ollama failed after %.1fs: purpose=%s model=%s num_ctx=%s prompt_chars=%s approx_tokens=%s "
            "error=%s response=%s; "
            "falling back to %s/%s",
            elapsed,
            purpose,
            model,
            num_ctx,
            total_chars,
            approx_tokens,
            e,
            response_text[:500],
            fallback_provider,
            fallback_model,
        )
        if fallback_provider == "anthropic":
            return _call_anthropic(system_prompt, user_prompt, fallback_model)
        elif fallback_provider == "openai":
            return _call_openai(system_prompt, user_prompt, False, fallback_model)
        else:
            raise
    finally:
        if lock_acquired:
            _OLLAMA_REQUEST_LOCK.release()


_JSON_REPAIR_PROMPT = """You are a JSON extraction assistant. The following text is an LLM response that should have been JSON but came back as prose. Extract the trading decisions from it and return ONLY valid JSON.

If the text recommends one or more trades, return:
{"decisions": [{"symbol": "...", "action": "BUY|SHORT|CLOSE", "quantity": N, "price": N.NN, "stop_loss": N.NN, "target": N.NN, "rationale": "..."}], "portfolio_notes": "..."}

If the text recommends no trades / holding / waiting, return:
{"decisions": [], "portfolio_notes": "summary of why no trades"}

Respond with ONLY the JSON object. No markdown, no explanation."""


def parse_json_response(text: str) -> dict:
    """Safely parse JSON from LLM output, even with markdown fences or trailing text."""
    if not text or not text.strip():
        raise ValueError("LLM returned empty response")
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]).strip()
    # Extract the first complete JSON object by matching braces
    if "{" in text:
        start = text.index("{")
        depth = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        candidate = text[start:end]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass  # fall through to repair below

    # LLM returned prose instead of JSON — try a repair call
    log.warning("LLM returned prose instead of JSON, attempting repair extraction")
    try:
        repair_raw = call_llm(
            _JSON_REPAIR_PROMPT,
            f"Extract JSON from this response:\n\n{text[:2000]}",
            json_mode=True,
            tier="low",
            purpose="json_repair",
        )
        repair_text = repair_raw.strip()
        if repair_text.startswith("```"):
            repair_lines = repair_text.split("\n")
            repair_text = "\n".join(repair_lines[1:-1]).strip()
        if "{" in repair_text:
            rs = repair_text.index("{")
            rd = 0
            re_ = rs
            for i in range(rs, len(repair_text)):
                if repair_text[i] == "{":
                    rd += 1
                elif repair_text[i] == "}":
                    rd -= 1
                    if rd == 0:
                        re_ = i + 1
                        break
            result = json.loads(repair_text[rs:re_])
            log.info("JSON repair succeeded")
            return result
    except Exception as repair_err:
        log.warning("JSON repair call failed: %s", repair_err)

    # Last resort: detect "no action" intent from the prose
    lower = text[:500].lower()
    no_action_phrases = [
        "no trades", "no trade", "no action", "recommend holding",
        "pass on", "stand aside", "stay flat", "no new", "wait for",
        "not recommend", "do not recommend", "skip", "no opportunities",
    ]
    if any(phrase in lower for phrase in no_action_phrases):
        log.warning("Detected no-action intent in prose, returning empty decisions")
        return {"decisions": [], "portfolio_notes": text[:300]}

    raise ValueError(f"Failed to parse LLM JSON response: {text[:500]}")
