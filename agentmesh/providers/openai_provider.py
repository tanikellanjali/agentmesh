from __future__ import annotations

from agentmesh.providers.base import (
    Completion,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimited,
    ProviderTransientError,
    ProviderUnavailable,
    Usage,
)

DEFAULT_MODEL = "gpt-4.1-mini"


class OpenAIProvider:
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

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
        effort: str | None = None,
    ) -> Completion:
        openai = self._sdk
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._call(model, messages, max_tokens)
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
        return Completion(
            text=choice.message.content or "",
            provider=self.name,
            model=response.model,
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            stop_reason=choice.finish_reason,
        )

    def _call(self, model: str, messages: list[dict[str, str]], max_tokens: int):
        """Newer models require `max_completion_tokens`; older ones only accept
        `max_tokens`. Try the current name first and fall back once."""
        try:
            return self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
            )
        except self._sdk.BadRequestError:
            return self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
