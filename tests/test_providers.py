import json

import pytest

from agentmesh.agents.factory import build_agent
from agentmesh.core.agent_synthesizer import synthesize_agent
from agentmesh.core.model_router import ModelBroker, _resolve_key
from agentmesh.providers import get_provider, list_providers, register_provider
from agentmesh.providers.base import (
    Completion,
    ProviderError,
    ProviderRateLimited,
    ProviderUnavailable,
    Usage,
)

ROUTING = {
    "anthropic/claude-opus-5": {
        "provider": "anthropic",
        "model": "claude-opus-5",
        "cost_per_1k_input_tokens": 0.005,
        "cost_per_1k_output_tokens": 0.025,
    },
    "openai/gpt-4.1-mini": {
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "cost_per_1k_input_tokens": 0.0004,
        "cost_per_1k_output_tokens": 0.0016,
    },
}


class RecordingProvider:
    name = "recording"

    def __init__(self, credentials=None, payload=None, error=None):
        self.credentials = credentials or {}
        self.payload = payload
        self.error = error
        self.calls = []

    def complete(self, *, system, prompt, model, max_tokens=4096, effort=None):
        self.calls.append({"system": system, "prompt": prompt, "model": model})
        if self.error:
            raise self.error
        text = self.payload if self.payload is not None else json.dumps(
            {"summary": "ok", "findings": ["f"], "confidence": 0.9}
        )
        return Completion(
            text=text,
            provider=self.name,
            model=model,
            usage=Usage(input_tokens=1000, output_tokens=200),
        )


def broker_with(provider, models=None):
    broker = ModelBroker(models=models or ROUTING, provider_override=provider.name)
    broker._clients[provider.name] = provider
    return broker


def test_builtin_providers_are_registered() -> None:
    for name in ("mock", "anthropic", "openai"):
        assert name in list_providers()


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ProviderUnavailable):
        get_provider("does-not-exist")


def test_providers_without_credentials_are_unavailable() -> None:
    with pytest.raises(ProviderUnavailable):
        get_provider("anthropic", {})
    with pytest.raises(ProviderUnavailable):
        get_provider("openai", {})


def test_custom_providers_can_be_registered() -> None:
    register_provider("recording", RecordingProvider)
    assert isinstance(get_provider("recording", {}), RecordingProvider)


def test_routing_key_resolves_to_provider_model_and_cost() -> None:
    resolved = _resolve_key("anthropic/claude-opus-5", ROUTING)

    assert resolved.provider == "anthropic"
    assert resolved.model == "claude-opus-5"
    assert resolved.cost(Usage(input_tokens=1000, output_tokens=1000)) == pytest.approx(0.03)


def test_unlisted_routing_key_falls_back_to_its_prefix() -> None:
    resolved = _resolve_key("openai/gpt-4.1", {})

    assert resolved.provider == "openai"
    assert resolved.model == "gpt-4.1"


def test_broker_skips_unavailable_providers_and_uses_the_fallback() -> None:
    spec = synthesize_agent("sql_review").model_copy(
        update={
            "models": {
                "default": "anthropic/claude-opus-5",
                "fallback": ["openai/gpt-4.1-mini"],
            }
        }
    )
    broker = ModelBroker(models=ROUTING, credentials={"OPENAI_API_KEY": "sk-test"})
    broker._clients["openai"] = RecordingProvider()

    call = broker.resolve(spec)

    assert call.model.key == "openai/gpt-4.1-mini"


def test_broker_reports_every_attempt_when_nothing_is_usable() -> None:
    spec = synthesize_agent("sql_review")
    broker = ModelBroker(models=ROUTING, credentials={})

    with pytest.raises(ProviderUnavailable) as exc:
        broker.resolve(spec)

    assert "anthropic" in str(exc.value)
    assert "--provider mock" in str(exc.value)


def test_agent_prompt_carries_spec_identity_and_contract() -> None:
    provider = RecordingProvider()
    spec = synthesize_agent("pii_detection")

    build_agent(spec, broker_with(provider)).run("scan this table", {})

    system = provider.calls[0]["system"]
    assert "Pii Detection Agent" in system
    assert "pii_detection" in system
    assert "summary, findings, confidence" in system
    assert "scan this table" in provider.calls[0]["prompt"]


def test_agent_prompt_includes_upstream_agent_output() -> None:
    provider = RecordingProvider()
    spec = synthesize_agent("report_writing")

    context = {"sql_analyzer_agent": {"summary": "index missing on orders.id"}}
    build_agent(spec, broker_with(provider)).run("write it up", context)

    assert "index missing on orders.id" in provider.calls[0]["prompt"]


def test_usage_and_cost_are_recorded_per_agent() -> None:
    provider = RecordingProvider()
    spec = synthesize_agent("sql_review").model_copy(
        update={"models": {"default": "anthropic/claude-opus-5", "fallback": []}}
    )
    broker = broker_with(provider)

    build_agent(spec, broker).run("check this", {})

    recorder = broker.recorder
    assert recorder.usage_for("sql_review_agent") == Usage(input_tokens=1000, output_tokens=200)
    # 1000/1000 * 0.005 + 200/1000 * 0.025
    assert recorder.cost_for("sql_review_agent") == pytest.approx(0.01)
    assert len(recorder.for_agent("sql_review_agent")) == 1


def test_contract_violations_are_reported_not_hidden() -> None:
    provider = RecordingProvider(payload=json.dumps({"summary": "only a summary"}))
    spec = synthesize_agent("sql_review")

    output = build_agent(spec, broker_with(provider)).run("check this", {})

    assert output["contract_violations"] == ["findings", "confidence"]


def test_fenced_json_is_parsed() -> None:
    payload = '```json\n{"summary": "s", "findings": [], "confidence": 0.5}\n```'
    provider = RecordingProvider(payload=payload)
    spec = synthesize_agent("sql_review")

    output = build_agent(spec, broker_with(provider)).run("check this", {})

    assert output["summary"] == "s"
    assert "contract_violations" not in output


def test_unparsable_response_is_flagged_rather_than_crashing() -> None:
    provider = RecordingProvider(payload="I cannot produce JSON right now.")
    spec = synthesize_agent("sql_review")

    output = build_agent(spec, broker_with(provider)).run("check this", {})

    assert output["confidence"] == 0.0
    assert output["contract_violations"]
    assert output["raw_response"] == "I cannot produce JSON right now."


def test_provider_errors_degrade_to_an_error_output() -> None:
    provider = RecordingProvider(error=ProviderRateLimited("slow down"))
    spec = synthesize_agent("sql_review")

    output = build_agent(spec, broker_with(provider)).run("check this", {})

    assert output["confidence"] == 0.0
    assert "slow down" in output["error"]


def test_a_failing_agent_does_not_stop_the_mesh() -> None:
    from agentmesh.core.orchestrator import run_agents

    failing = synthesize_agent("sql_review")
    healthy = synthesize_agent("report_generation")
    provider = RecordingProvider(error=ProviderError("boom"))

    result = run_agents("check this", [failing, healthy], broker_with(provider))

    assert [execution.agent_id for execution in result.executions] == [
        "report_generation_agent",
        "sql_review_agent",
    ]
    assert all("error" in execution.output for execution in result.executions)
