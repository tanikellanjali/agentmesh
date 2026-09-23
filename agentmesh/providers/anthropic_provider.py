from __future__ import annotations

from typing import Any

from agentmesh.providers.base import (
    Message,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimited,
    ProviderRefusal,
    ProviderTransientError,
    ProviderUnavailable,
    SingleShotMixin,
    Turn,
    Usage,
)
from agentmesh.tools.base import ToolCall, ToolSpec

DEFAULT_MODEL = "claude-opus-5"


class AnthropicProvider(SingleShotMixin):
    """Adapter for the Claude Messages API via the official `anthropic` SDK."""

    name = "anthropic"

    def __init__(self, credentials: dict[str, str] | None = None) -> None:
        credentials = credentials or {}
        api_key = credentials.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderUnavailable(
                "ANTHROPIC_API_KEY is not set. Run `agentmesh configure-models`."
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ProviderUnavailable(
                "The anthropic SDK is not installed. Install with: pip install 'agentmesh[anthropic]'"
            ) from exc

        self._sdk = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)

    # -- shape conversion -------------------------------------------------
    @staticmethod
    def _to_native(messages: list[Message]) -> list[dict[str, Any]]:
        """Neutral messages -> Anthropic content blocks.

        Tool results for one assistant turn must arrive in a single user
        message, so consecutive tool messages are grouped.
        """
        native: list[dict[str, Any]] = []
        pending_results: list[dict[str, Any]] = []

        def flush() -> None:
            if pending_results:
                native.append({"role": "user", "content": list(pending_results)})
                pending_results.clear()

        for msg in messages:
            if msg.role == "tool":
                pending_results.append({
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content,
                })
                continue

            flush()
            if msg.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                for call in msg.tool_calls:
                    blocks.append({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    })
                native.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
            else:
                native.append({"role": "user", "content": msg.content})

        flush()
        return native

    # -- api --------------------------------------------------------------
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
        anthropic = self._sdk

        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": self._to_native(messages),
        }
        if tools:
            request["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]
        if effort:
            request["output_config"] = {"effort": effort}

        try:
            response = self._client.messages.create(**request)
        except anthropic.AuthenticationError as exc:
            raise ProviderAuthError(f"Anthropic rejected the API key: {exc}") from exc
        except anthropic.PermissionDeniedError as exc:
            raise ProviderAuthError(f"Anthropic key lacks permission: {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise ProviderRateLimited(f"Anthropic rate limited the request: {exc}") from exc
        except anthropic.APIStatusError as exc:
            message = f"Anthropic API error ({exc.status_code}): {exc}"
            if exc.status_code >= 500:
                raise ProviderTransientError(message) from exc
            raise ProviderError(message) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderTransientError(f"Could not reach the Anthropic API: {exc}") from exc

        # A refusal is a 200 with no usable content, so check before reading it.
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            raise ProviderRefusal(f"Claude declined the request (category: {category}).")

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        calls = [
            ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {}))
            for block in response.content
            if getattr(block, "type", None) == "tool_use"
        ]

        return Turn(
            text=text,
            provider=self.name,
            model=response.model,
            tool_calls=calls,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            stop_reason=response.stop_reason,
        )
