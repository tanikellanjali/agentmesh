from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agentmesh.tools.base import ToolCall, ToolSpec


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

    def __add__(self, other: Usage) -> Usage:
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


@dataclass(frozen=True)
class Message:
    """A provider-neutral conversation turn.

    role is "user", "assistant" or "tool". Adapters translate this shape into
    each vendor's native format, so the agent loop never learns vendor details.
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass(frozen=True)
class Turn:
    """One assistant response, which may ask for tools instead of answering."""

    text: str
    provider: str
    model: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    def as_message(self) -> Message:
        return Message(role="assistant", content=self.text, tool_calls=list(self.tool_calls))


@runtime_checkable
class Provider(Protocol):
    """Contract every model backend implements.

    `converse` is the real surface: a message list plus the tools the model may
    call. `complete` is the single-shot convenience wrapper over it.
    """

    name: str

    def converse(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        model: str,
        max_tokens: int = 4096,
        effort: str | None = None,
        timeout: float | None = None,
    ) -> Turn: ...

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        model: str,
        max_tokens: int = 4096,
        effort: str | None = None,
        timeout: float | None = None,
    ) -> Completion: ...


class SingleShotMixin:
    """Implements `complete` for any provider that implements `converse`."""

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        model: str,
        max_tokens: int = 4096,
        effort: str | None = None,
        timeout: float | None = None,
    ) -> Completion:
        turn = self.converse(  # type: ignore[attr-defined]
            system=system,
            messages=[Message(role="user", content=prompt)],
            tools=None,
            model=model,
            max_tokens=max_tokens,
            effort=effort,
            timeout=timeout,
        )
        return Completion(
            text=turn.text,
            provider=turn.provider,
            model=turn.model,
            usage=turn.usage,
            stop_reason=turn.stop_reason,
        )
