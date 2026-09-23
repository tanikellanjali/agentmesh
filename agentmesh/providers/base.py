from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class ProviderError(RuntimeError):
    """Base class for provider failures surfaced to the runtime."""


class ProviderUnavailable(ProviderError):
    """The provider SDK is not installed, or no credentials were supplied."""


class ProviderAuthError(ProviderError):
    """Credentials were rejected by the provider."""


class ProviderTransientError(ProviderError):
    """A transport or server-side failure that is worth retrying."""


class ProviderRateLimited(ProviderTransientError):
    """The provider rate limited the request."""


class ProviderRefusal(ProviderError):
    """The model declined to answer."""


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True)
class Completion:
    text: str
    provider: str
    model: str
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None


@runtime_checkable
class Provider(Protocol):
    """Minimal contract every model backend implements.

    Text in, text out. Structure is the runtime's concern, not the provider's,
    which keeps adapters thin and comparable across vendors.
    """

    name: str

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        model: str,
        max_tokens: int = 4096,
        effort: str | None = None,
    ) -> Completion: ...
