from __future__ import annotations

import json
from typing import Any

from agentmesh.providers.base import (
    Message,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimited,
    ProviderTransientError,
    ProviderUnavailable,
    SingleShotMixin,
    Turn,
    Usage,
)
from agentmesh.tools.base import ToolCall, ToolSpec

DEFAULT_MODEL = "gpt-4.1-mini"


class OpenAIProvider(SingleShotMixin):
    """Adapter for the OpenAI Chat Completions API via the official `openai` SDK."""

    name = "openai"

    def __init__(self, credentials: dict[str, str] | None = None) -> None:
        credentials = credentials or {}
        api_key = credentials.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderUnavailable(
                "OPENAI_API_KEY is not set. Run `agentmesh configure-models`."
            )
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ProviderUnavailable(
                "The openai SDK is not installed. Install with: pip install 'agentmesh[openai]'"
            ) from exc

        self._sdk = openai
        self._client = openai.OpenAI(api_key=api_key)

    @staticmethod
    def _to_native(system: str, messages: list[Message]) -> list[dict[str, Any]]:
        native: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for msg in messages:
            if msg.role == "tool":
                native.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
            elif msg.role == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": msg.content or None}
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in msg.tool_calls
                    ]
                native.append(entry)
            else:
                native.append({"role": "user", "content": msg.content})
        return native

    def converse(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
        effort: str | None = None,
    ) -> Turn:
        openai = self._sdk

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._to_native(system, messages),
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]

        try:
            response = self._call(kwargs, max_tokens)
        except openai.AuthenticationError as exc:
            raise ProviderAuthError(f"OpenAI rejected the API key: {exc}") from exc
        except openai.PermissionDeniedError as exc:
            raise ProviderAuthError(f"OpenAI key lacks permission: {exc}") from exc
        except openai.RateLimitError as exc:
            raise ProviderRateLimited(f"OpenAI rate limited the request: {exc}") from exc
        except openai.APIStatusError as exc:
            message = f"OpenAI API error ({exc.status_code}): {exc}"
            if exc.status_code >= 500:
                raise ProviderTransientError(message) from exc
            raise ProviderError(message) from exc
        except openai.APIConnectionError as exc:
            raise ProviderTransientError(f"Could not reach the OpenAI API: {exc}") from exc

        choice = response.choices[0]
        usage = response.usage
        calls: list[ToolCall] = []
        for raw in getattr(choice.message, "tool_calls", None) or []:
            try:
                arguments = json.loads(raw.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append(ToolCall(id=raw.id, name=raw.function.name, arguments=arguments))

        return Turn(
            text=choice.message.content or "",
            provider=self.name,
            model=response.model,
            tool_calls=calls,
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            stop_reason=choice.finish_reason,
        )

    def _call(self, kwargs: dict[str, Any], max_tokens: int):
        """Newer models require `max_completion_tokens`; older ones only accept
        `max_tokens`. Try the current name first and fall back once."""
        try:
            return self._client.chat.completions.create(
                **kwargs, max_completion_tokens=max_tokens
            )
        except self._sdk.BadRequestError:
            return self._client.chat.completions.create(**kwargs, max_tokens=max_tokens)
