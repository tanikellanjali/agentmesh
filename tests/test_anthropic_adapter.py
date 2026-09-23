"""Validates the Anthropic adapter against real SDK response objects.

Skipped when the optional `anthropic` extra is not installed. No network calls:
`messages.create` is replaced with a stub that returns genuine SDK types, so the
adapter's parsing is checked against the real response shape.
"""

import pytest

anthropic = pytest.importorskip("anthropic")

from anthropic.types import Message, TextBlock
from anthropic.types import Usage as SdkUsage

from agentmesh.providers.anthropic_provider import AnthropicProvider
from agentmesh.providers.base import (
    ProviderAuthError,
    ProviderRateLimited,
    ProviderRefusal,
)


def make_message(text="{}", stop_reason="end_turn", stop_details=None):
    return Message(
        id="msg_1",
        type="message",
        role="assistant",
        model="claude-opus-5",
        content=[TextBlock(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=stop_details,
        stop_sequence=None,
        usage=SdkUsage(input_tokens=120, output_tokens=45),
    )


@pytest.fixture()
def provider(monkeypatch):
    return AnthropicProvider({"ANTHROPIC_API_KEY": "sk-ant-test"})


def test_request_is_shaped_for_the_messages_api(provider, monkeypatch) -> None:
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return make_message('{"summary": "ok"}')

    monkeypatch.setattr(provider._client.messages, "create", fake_create)
    provider.complete(system="sys", prompt="hello", model="claude-opus-5", max_tokens=1024)

    assert captured["model"] == "claude-opus-5"
    assert captured["max_tokens"] == 1024
    assert captured["system"] == "sys"
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    # effort is opt-in; omitting it leaves the API default in place
    assert "output_config" not in captured


def test_effort_is_passed_through_output_config(provider, monkeypatch) -> None:
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return make_message()

    monkeypatch.setattr(provider._client.messages, "create", fake_create)
    provider.complete(system="s", prompt="p", model="claude-opus-5", effort="low")

    assert captured["output_config"] == {"effort": "low"}


def test_text_blocks_and_usage_are_extracted(provider, monkeypatch) -> None:
    monkeypatch.setattr(
        provider._client.messages, "create", lambda **_: make_message('{"summary": "done"}')
    )

    completion = provider.complete(system="s", prompt="p", model="claude-opus-5")

    assert completion.text == '{"summary": "done"}'
    assert completion.model == "claude-opus-5"
    assert completion.usage.input_tokens == 120
    assert completion.usage.output_tokens == 45


def test_refusals_raise_instead_of_returning_empty_text(provider, monkeypatch) -> None:
    monkeypatch.setattr(
        provider._client.messages,
        "create",
        lambda **_: make_message(text="", stop_reason="refusal"),
    )

    with pytest.raises(ProviderRefusal):
        provider.complete(system="s", prompt="p", model="claude-opus-5")


@pytest.mark.parametrize(
    ("sdk_error", "expected"),
    [
        (anthropic.AuthenticationError, ProviderAuthError),
        (anthropic.RateLimitError, ProviderRateLimited),
    ],
)
def test_sdk_errors_map_to_provider_errors(provider, monkeypatch, sdk_error, expected) -> None:
    import httpx2

    def fake_create(**_):
        raise sdk_error(
            "nope",
            response=httpx2.Response(
                401, request=httpx2.Request("POST", "https://api.anthropic.com")
            ),
            body=None,
        )

    monkeypatch.setattr(provider._client.messages, "create", fake_create)

    with pytest.raises(expected):
        provider.complete(system="s", prompt="p", model="claude-opus-5")


def test_missing_credentials_is_reported_before_any_call() -> None:
    from agentmesh.providers.base import ProviderUnavailable

    with pytest.raises(ProviderUnavailable, match="ANTHROPIC_API_KEY"):
        AnthropicProvider({})
