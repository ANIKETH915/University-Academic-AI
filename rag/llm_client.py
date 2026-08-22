"""
Unified, multi-provider LLM client.

The LLM is an *assistant* in this system: it may help reconstruct layouts,
judge boundaries, normalise topics and classify recurrence. It is never the
source of truth, and every provider path here is allowed to fail — callers
must degrade to deterministic extraction rather than invent content.

Providers: openai | gemini | groq | openrouter

Configuration (all optional; absence simply disables LLM assistance):

  LLM_PROVIDER            primary provider (default: openai)
  LLM_FALLBACK_PROVIDERS  comma-separated order, e.g. "gemini,groq,openrouter"
  LLM_TIMEOUT_SECONDS     per-request timeout (default 60)
  LLM_MAX_RETRIES         retries per provider for transient errors (default 2)

  OPENAI_API_KEY     / OPENAI_MODEL
  GEMINI_API_KEY     / GEMINI_MODEL
  GROQ_API_KEY       / GROQ_MODEL
  OPENROUTER_API_KEY / OPENROUTER_MODEL

Legacy compatibility: LLM_API_KEY / LLM_MODEL still work and apply to the
resolved provider.

Secrets are read from the environment only. They are never logged, returned,
persisted to the vector store, or sent to the frontend.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

SUPPORTED_PROVIDERS = ("openai", "gemini", "groq", "openrouter")

_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
    "groq": "llama-3.1-8b-instant",
    "openrouter": "openai/gpt-4o-mini",
}

_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

_MODEL_ENV = {
    "openai": "OPENAI_MODEL",
    "gemini": "GEMINI_MODEL",
    "groq": "GROQ_MODEL",
    "openrouter": "OPENROUTER_MODEL",
}

_DEFAULT_FALLBACK_ORDER = ("openai", "gemini", "groq", "openrouter")

_TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "connection",
    "temporarily",
    "rate limit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "overloaded",
    "unavailable",
)


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _timeout_seconds() -> float:
    try:
        return max(1.0, float(_env("LLM_TIMEOUT_SECONDS") or 60))
    except ValueError:
        return 60.0


def _max_retries() -> int:
    try:
        return max(0, min(5, int(_env("LLM_MAX_RETRIES") or 2)))
    except ValueError:
        return 2


def provider_key(provider: str) -> str:
    """API key for a provider, preferring provider-specific env var."""
    key = _env(_KEY_ENV.get(provider, ""))
    if key:
        return key
    # Legacy single-key configuration applies only to the configured provider
    if _env("LLM_PROVIDER").lower() == provider:
        return _env("LLM_API_KEY")
    return ""


def provider_model(provider: str) -> str:
    explicit = _env(_MODEL_ENV.get(provider, ""))
    if explicit:
        return explicit
    if _env("LLM_PROVIDER").lower() == provider and _env("LLM_MODEL"):
        return _env("LLM_MODEL")
    return _DEFAULT_MODELS.get(provider, "")


def provider_chain() -> List[str]:
    """
    Ordered list of providers to attempt: primary first, then configured
    fallbacks, then the remaining supported providers. Only providers with a
    key present are returned.
    """
    primary = _env("LLM_PROVIDER").lower()
    if primary in ("none", "off", "disabled"):
        return []

    order: List[str] = []
    if primary in SUPPORTED_PROVIDERS:
        order.append(primary)

    configured = _env("LLM_FALLBACK_PROVIDERS")
    if configured:
        for part in configured.split(","):
            name = part.strip().lower()
            if name in SUPPORTED_PROVIDERS and name not in order:
                order.append(name)
    else:
        for name in _DEFAULT_FALLBACK_ORDER:
            if name not in order:
                order.append(name)

    return [p for p in order if provider_key(p)]


def llm_configured() -> bool:
    """True when at least one provider has credentials available."""
    return bool(provider_chain())


def llm_status() -> Dict[str, Any]:
    """Non-secret diagnostic snapshot; never includes keys."""
    chain = provider_chain()
    return {
        "configured": bool(chain),
        "provider_chain": chain,
        "primary": chain[0] if chain else None,
        "models": {p: provider_model(p) for p in chain},
        "timeout_seconds": _timeout_seconds(),
        "max_retries": _max_retries(),
        "providers_available": [p for p in SUPPORTED_PROVIDERS if provider_key(p)],
    }


def _is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_MARKERS)


def _call_openai_compatible(
    *,
    base_url: Optional[str],
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    from openai import OpenAI  # type: ignore

    kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": _timeout_seconds()}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def _call_gemini(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=api_key)
    gm = genai.GenerativeModel(model)
    response = gm.generate_content(
        f"{system_prompt}\n\n{user_prompt}",
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        },
        request_options={"timeout": _timeout_seconds()},
    )
    return (getattr(response, "text", None) or "").strip()


def _invoke(
    provider: str,
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    key = provider_key(provider)
    model = provider_model(provider)
    if provider == "openai":
        return _call_openai_compatible(
            base_url=None, api_key=key, model=model,
            system_prompt=system_prompt, user_prompt=user_prompt,
            max_tokens=max_tokens, temperature=temperature,
        )
    if provider == "groq":
        return _call_openai_compatible(
            base_url="https://api.groq.com/openai/v1", api_key=key, model=model,
            system_prompt=system_prompt, user_prompt=user_prompt,
            max_tokens=max_tokens, temperature=temperature,
        )
    if provider == "openrouter":
        return _call_openai_compatible(
            base_url="https://openrouter.ai/api/v1", api_key=key, model=model,
            system_prompt=system_prompt, user_prompt=user_prompt,
            max_tokens=max_tokens, temperature=temperature,
        )
    if provider == "gemini":
        return _call_gemini(
            api_key=key, model=model,
            system_prompt=system_prompt, user_prompt=user_prompt,
            max_tokens=max_tokens, temperature=temperature,
        )
    raise ValueError(f"unsupported provider: {provider}")


def call_llm_with_trace(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Try each configured provider in order with bounded retries.

    Returns (text_or_None, attempts) where attempts records provider, outcome
    and error class for auditing. Never raises, never logs credentials.
    """
    attempts: List[Dict[str, Any]] = []
    chain = provider_chain()
    if not chain:
        return None, attempts

    retries = _max_retries()
    for provider in chain:
        for attempt in range(retries + 1):
            started = time.time()
            try:
                text = _invoke(
                    provider,
                    system_prompt,
                    user_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                elapsed = round(time.time() - started, 2)
                if not text:
                    attempts.append(
                        {
                            "provider": provider,
                            "attempt": attempt + 1,
                            "outcome": "empty_response",
                            "seconds": elapsed,
                        }
                    )
                    break
                attempts.append(
                    {
                        "provider": provider,
                        "attempt": attempt + 1,
                        "outcome": "ok",
                        "seconds": elapsed,
                    }
                )
                return text, attempts
            except Exception as exc:
                elapsed = round(time.time() - started, 2)
                transient = _is_transient(exc)
                attempts.append(
                    {
                        "provider": provider,
                        "attempt": attempt + 1,
                        "outcome": "transient_error" if transient else "error",
                        "error_type": type(exc).__name__,
                        "seconds": elapsed,
                    }
                )
                print(
                    f"[LLM_ATTEMPT] provider={provider} attempt={attempt + 1} "
                    f"outcome={'transient' if transient else 'error'} "
                    f"type={type(exc).__name__}"
                )
                if not transient or attempt >= retries:
                    break
                time.sleep(min(4.0, 0.5 * (2**attempt)))
    return None, attempts


def call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> Optional[str]:
    """Raw LLM text, or None when every configured provider failed."""
    text, _attempts = call_llm_with_trace(
        system_prompt, user_prompt, max_tokens=max_tokens, temperature=temperature
    )
    return text


def parse_json_from_llm(text: str) -> Optional[Any]:
    """Extract JSON object/array from LLM output (tolerates markdown fences)."""
    if not text:
        return None
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, re.I)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = cleaned.find(open_c)
        end = cleaned.rfind(close_c)
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> Optional[Dict[str, Any]]:
    """
    JSON-returning call. Malformed JSON is treated as a provider failure and
    the next provider in the chain is tried.
    """
    attempts: List[Dict[str, Any]] = []
    chain = provider_chain()
    if not chain:
        return None

    for provider in chain:
        raw = None
        for attempt in range(_max_retries() + 1):
            try:
                raw = _invoke(
                    provider,
                    system_prompt,
                    user_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                break
            except Exception as exc:
                transient = _is_transient(exc)
                attempts.append(
                    {"provider": provider, "outcome": "error", "error_type": type(exc).__name__}
                )
                print(
                    f"[LLM_JSON_ATTEMPT] provider={provider} attempt={attempt + 1} "
                    f"type={type(exc).__name__}"
                )
                if not transient or attempt >= _max_retries():
                    raw = None
                    break
                time.sleep(min(4.0, 0.5 * (2**attempt)))
        if not raw:
            continue
        parsed = parse_json_from_llm(raw)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"questions": parsed}
        print(f"[LLM_JSON_INVALID] provider={provider} returned non-JSON; trying next provider")
    return None
