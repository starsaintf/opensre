"""Provider-agnostic onboarding endpoint (base-URL) collection.

Some providers need a user-supplied endpoint before the validation probe can
run: Azure OpenAI (a resource URL) and the custom OpenAI-/Anthropic-compatible
gateways (an arbitrary base URL). The onboarding flow calls
:func:`ensure_endpoint_settings` for every provider right before validating the
credential; providers that need no endpoint return an empty dict.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from surfaces.cli.wizard.config import ProviderOption


def ensure_endpoint_settings(provider: ProviderOption) -> dict[str, str] | None:
    """Return endpoint env vars for *provider*, prompting when missing.

    Returns an empty dict when the provider needs no endpoint, the endpoint env
    mapping when configured/collected, or ``None`` when the user backs out.
    """
    from core.llm.providers.azure_openai import is_azure_openai_provider
    from core.llm.providers.custom_endpoints import is_custom_provider

    if is_azure_openai_provider(provider.value):
        from surfaces.cli.wizard.azure_openai import ensure_endpoint_settings as _azure

        return _azure(provider)
    if is_custom_provider(provider.value):
        return _ensure_custom_endpoint(provider)
    return {}


def _ensure_custom_endpoint(provider: ProviderOption) -> dict[str, str] | None:
    """Return the custom gateway base URL, prompting when it isn't set yet."""
    from core.llm.providers.custom_endpoints import normalize_custom_base_url

    if not provider.endpoint_env:
        return {}
    configured = normalize_custom_base_url(os.getenv(provider.endpoint_env, ""))
    if configured:
        return {provider.endpoint_env: configured}
    return _prompt_custom_endpoint(provider)


def _prompt_custom_endpoint(provider: ProviderOption) -> dict[str, str] | None:
    from core.llm.providers.custom_endpoints import normalize_custom_base_url
    from platform.terminal.theme import ERROR
    from surfaces.cli.wizard._ui import WizardBack, _console, _prompt_value, _step

    _step("Endpoint")
    try:
        raw = _prompt_value(
            f"Base URL ({provider.endpoint_env})",
            default=os.getenv(provider.endpoint_env, provider.credential_default),
            secret=False,
            back_on_cancel=True,
        )
    except WizardBack:
        return None
    normalized = normalize_custom_base_url(raw)
    if not normalized:
        _console.print(f"[{ERROR}]A base URL is required for this provider.[/]")
        return None
    return {provider.endpoint_env: normalized}
