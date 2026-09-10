#!/usr/bin/env python3
"""Central LLM provider configuration for the supervisor.

All supervisor-side LLM calls go through the OpenAI-compatible chat completions
API. This module decides which backend to talk to and how, so individual
components don't need to hard-code provider-specific logic.

Supported providers (selected via ``LLM_PROVIDER`` or auto-detected from keys):

- ``openrouter``: ``OPENROUTER_API_KEY`` -> https://openrouter.ai/api/v1
- ``gemini``:     ``GEMINI_API_KEY``     -> Google's OpenAI-compatible endpoint
- ``openai``:     ``OPENAI_API_KEY``     -> https://api.openai.com/v1

``LLM_BASE_URL`` overrides the base URL for any provider, which also lets you
point at any other OpenAI-compatible service.
"""

import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI


GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openrouter": {
        "label": "OpenRouter",
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "defaults": {
            "supervisor": "openai/o4-mini",
            "summarization": "openai/o4-mini",
            "router": "openai/o4-mini",
            "todo": "anthropic/claude-opus-4.1",
            "prompt_generator": "anthropic/claude-opus-4.1",
        },
        "available_models": "anthropic/claude-sonnet-4,openai/o3,anthropic/claude-opus-4,google/gemini-2.5-pro,openai/o3-pro",
        "available_models_env": "OPENROUTER_AVAILABLE_MODELS",
        "todo_model_env": "TODO_GENERATOR_OPENROUTER_MODEL",
        # OpenRouter accepts the classic chat-completions parameters
        "max_tokens_param": "max_tokens",
        "supports_temperature": True,
    },
    "gemini": {
        "label": "Google Gemini",
        "env_key": "GEMINI_API_KEY",
        "base_url": GEMINI_OPENAI_BASE_URL,
        "defaults": {
            "supervisor": "gemini-2.5-pro",
            "summarization": "gemini-2.5-flash",
            "router": "gemini-2.5-flash",
            "todo": "gemini-2.5-pro",
            "prompt_generator": "gemini-2.5-pro",
        },
        "available_models": "gemini-2.5-pro,gemini-2.5-flash",
        "available_models_env": "GEMINI_AVAILABLE_MODELS",
        "todo_model_env": "TODO_GENERATOR_GEMINI_MODEL",
        "max_tokens_param": "max_tokens",
        "supports_temperature": True,
    },
    "openai": {
        "label": "OpenAI",
        "env_key": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "defaults": {
            "supervisor": "o4-mini",
            "summarization": "o4-mini",
            "router": "o4-mini",
            "todo": "gpt-5",
            "prompt_generator": "gpt-5",
        },
        "available_models": "o3,gpt-5",
        "available_models_env": "OPENAI_AVAILABLE_MODELS",
        "todo_model_env": "TODO_GENERATOR_OPENAI_MODEL",
        # Newer OpenAI reasoning models reject `max_tokens` and custom temperature
        "max_tokens_param": "max_completion_tokens",
        "supports_temperature": False,
    },
}

# Auto-detection order when LLM_PROVIDER is not set
_DETECTION_ORDER = ["openrouter", "gemini", "openai"]


def get_provider_name() -> Optional[str]:
    """Return the active provider name, or None if no key is configured."""
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit:
        if explicit not in PROVIDERS:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{explicit}'. Supported: {', '.join(PROVIDERS)}"
            )
        return explicit
    for name in _DETECTION_ORDER:
        if os.getenv(PROVIDERS[name]["env_key"]):
            return name
    return None


def _provider() -> Dict[str, Any]:
    name = get_provider_name()
    if name is None:
        raise RuntimeError(
            "No LLM API key configured. Set one of: "
            + ", ".join(p["env_key"] for p in PROVIDERS.values())
        )
    return PROVIDERS[name]


def get_provider_label() -> str:
    return _provider()["label"]


def get_api_key() -> Optional[str]:
    name = get_provider_name()
    if name is None:
        return None
    return os.getenv(PROVIDERS[name]["env_key"])


def get_base_url() -> str:
    return os.getenv("LLM_BASE_URL") or _provider()["base_url"]


def create_client(api_key: Optional[str] = None) -> AsyncOpenAI:
    """Create an AsyncOpenAI client pointed at the active provider."""
    return AsyncOpenAI(api_key=api_key or get_api_key(), base_url=get_base_url())


def get_default_model(role: str) -> str:
    """Default model for a role: supervisor, summarization, router, todo, prompt_generator."""
    return _provider()["defaults"][role]


def normalize_model_name(model: str) -> str:
    """Strip an ``openai/`` prefix when talking to OpenAI directly.

    OpenRouter-style names such as ``openai/o4-mini`` are only valid on
    OpenRouter; other providers expect the bare model id.
    """
    if get_provider_name() != "openrouter" and model.startswith("openai/"):
        return model[len("openai/"):]
    return model


def get_todo_model() -> str:
    provider = _provider()
    return os.getenv(provider["todo_model_env"]) or provider["defaults"]["todo"]


def get_available_models() -> List[str]:
    """Models the supervisor may randomly switch between."""
    provider = _provider()
    raw = os.getenv(provider["available_models_env"], provider["available_models"])
    return [m.strip() for m in raw.split(",") if m.strip()]


def completion_params(
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: Optional[float] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build chat.completions.create kwargs using the provider's parameter names."""
    provider = _provider()
    params: Dict[str, Any] = {"model": model, "messages": messages}
    params[provider["max_tokens_param"]] = max_tokens
    if temperature is not None and provider["supports_temperature"]:
        params["temperature"] = temperature
    params.update(extra)
    return params
