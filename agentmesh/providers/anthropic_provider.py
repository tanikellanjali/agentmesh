from __future__ import annotations

from agentmesh.providers.base import (
    Completion,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimited,
    ProviderRefusal,
    ProviderTransientError,
    ProviderUnavailable,
    Usage,
)

DEFAULT_MODEL = "claude-opus-5"


class AnthropicProvider:
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

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
        effort: str | None = None,
    ) -> Completion:
        anthropic = self._sdk

        request: dict[str, object] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
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
        return Completion(
            text=text,
            provider=self.name,
            model=response.model,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            stop_reason=response.stop_reason,
        )
