from __future__ import annotations

from collections.abc import Callable

from agentmesh.providers.base import (
    Completion,
    Provider,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimited,
    ProviderRefusal,
    ProviderUnavailable,
    Usage,
)
from agentmesh.providers.anthropic_provider import AnthropicProvider
from agentmesh.providers.mock import MockProvider
from agentmesh.providers.openai_provider import OpenAIProvider

__all__ = [
    "Completion",
    "Provider",
    "ProviderAuthError",
    "ProviderError",
    "ProviderRateLimited",
    "ProviderRefusal",
    "ProviderUnavailable",
    "Usage",
    "AnthropicProvider",
    "MockProvider",
    "OpenAIProvider",
    "get_provider",
    "list_providers",
    "register_provider",
]

_PROVIDERS: dict[str, Callable[[dict[str, str]], Provider]] = {
    "mock": MockProvider,
    "local": MockProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
}


def register_provider(name: str, factory: Callable[[dict[str, str]], Provider]) -> None:
    _PROVIDERS[name] = factory


def list_providers() -> list[str]:
    return sorted(_PROVIDERS)


def get_provider(name: str, credentials: dict[str, str] | None = None) -> Provider:
    try:
        factory = _PROVIDERS[name]
    except KeyError as exc:
        raise ProviderUnavailable(
            f"Unknown provider '{name}'. Known providers: {', '.join(list_providers())}"
        ) from exc
    return factory(credentials or {})
