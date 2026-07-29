"""Custom OpenAI-/Anthropic-compatible gateway providers.

These providers let OpenSRE point at an arbitrary base URL — a LiteLLM proxy,
vLLM, LocalAI, or an internal model gateway — with the user's own API key and
model name. They exist for self-hosted, proxied, and on-prem deployments where
direct calls to the public OpenAI/Anthropic APIs are not allowed.

``custom-openai`` reuses the OpenAI-compatible client boundary (same path as
openrouter/deepseek). ``custom-anthropic`` uses the Anthropic SDK with a
base-URL override, threaded through both the sync LLM client and the agent
loop. Neither infers behavior from the URL: the OpenAI-vs-Anthropic API surface
is fixed by the provider slug, not by the endpoint.
"""

from __future__ import annotations

from typing import Any

from core.llm.types import ModelType

CUSTOM_OPENAI_PROVIDER = "custom-openai"
CUSTOM_ANTHROPIC_PROVIDER = "custom-anthropic"

# Settings-attribute prefixes: the hyphenated slugs are not valid Python
# attribute names, so per-tier model lookups use these underscore forms.
CUSTOM_OPENAI_SETTINGS_PREFIX = "custom_openai"
CUSTOM_ANTHROPIC_SETTINGS_PREFIX = "custom_anthropic"

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1"})


def is_custom_openai_provider(provider: str) -> bool:
    """Return whether *provider* is the custom OpenAI-compatible slug."""
    return provider.strip().lower() == CUSTOM_OPENAI_PROVIDER


def is_custom_anthropic_provider(provider: str) -> bool:
    """Return whether *provider* is the custom Anthropic-compatible slug."""
    return provider.strip().lower() == CUSTOM_ANTHROPIC_PROVIDER


def is_custom_provider(provider: str) -> bool:
    """Return whether *provider* is either custom gateway slug."""
    return is_custom_openai_provider(provider) or is_custom_anthropic_provider(provider)


def normalize_custom_base_url(value: str) -> str:
    """Normalize a user-supplied gateway base URL — exactly once.

    Strips surrounding whitespace and a single trailing slash so downstream
    clients that append operation paths (e.g. ``/chat/completions``) never
    produce a doubled ``/v1/v1``. A missing scheme defaults to ``http://`` for
    loopback hosts (local LiteLLM/vLLM) and ``https://`` otherwise. Any path the
    user supplies (e.g. ``/v1``) is preserved verbatim and never appended to,
    matching how the OpenAI-compatible and Anthropic clients join routes.
    """
    base = (value or "").strip()
    if not base:
        return ""
    if not base.startswith(("http://", "https://")):
        host = base.split("/", 1)[0].split(":", 1)[0].lower()
        scheme = "http" if host in _LOOPBACK_HOSTS else "https"
        base = f"{scheme}://{base}"
    return base.rstrip("/")


def custom_settings_prefix(provider: str) -> str:
    """Return the settings-attribute prefix for a custom provider slug."""
    if is_custom_anthropic_provider(provider):
        return CUSTOM_ANTHROPIC_SETTINGS_PREFIX
    return CUSTOM_OPENAI_SETTINGS_PREFIX


def select_custom_model(settings: Any, provider: str, model_type: ModelType) -> str:
    """Return the configured per-tier model for a custom provider."""
    prefix = custom_settings_prefix(provider)
    return str(getattr(settings, f"{prefix}_{model_type}_model"))


def custom_base_url(settings: Any, provider: str) -> str:
    """Return the configured base URL for a custom provider."""
    prefix = custom_settings_prefix(provider)
    return str(getattr(settings, f"{prefix}_base_url"))
