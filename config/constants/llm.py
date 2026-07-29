"""LLM provider connection env-var names.

Kept in this leaf module (not ``config.config``) because modules that
``config.config`` imports — e.g. ``config.llm_auth.provider_catalog`` — also
need them, so co-locating with ``config.config`` would be a cyclic import.
"""

from __future__ import annotations

from typing import Final

AZURE_OPENAI_BASE_URL_ENV: Final[str] = "AZURE_OPENAI_BASE_URL"
AZURE_OPENAI_API_VERSION_ENV: Final[str] = "AZURE_OPENAI_API_VERSION"
AZURE_OPENAI_API_KEY_ENV: Final[str] = "AZURE_OPENAI_API_KEY"

# Custom OpenAI-/Anthropic-compatible gateways (LiteLLM proxy, vLLM, LocalAI,
# internal model gateways). The base URL is user-supplied, not hard-coded.
CUSTOM_OPENAI_BASE_URL_ENV: Final[str] = "CUSTOM_OPENAI_BASE_URL"
CUSTOM_OPENAI_API_KEY_ENV: Final[str] = "CUSTOM_OPENAI_API_KEY"
CUSTOM_ANTHROPIC_BASE_URL_ENV: Final[str] = "CUSTOM_ANTHROPIC_BASE_URL"
CUSTOM_ANTHROPIC_API_KEY_ENV: Final[str] = "CUSTOM_ANTHROPIC_API_KEY"
