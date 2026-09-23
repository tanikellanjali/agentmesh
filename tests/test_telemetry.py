import json
import logging

import pytest

from agentmesh.agents.factory import build_agent
from agentmesh.core.agent_synthesizer import synthesize_agent
from agentmesh.core.model_router import ModelBroker
from agentmesh.core.orchestrator import run_agents
from agentmesh.core.telemetry import (
    BudgetExceeded,
    RunRecorder,
    configure_logging,
)
from agentmesh.providers.base import SingleShotMixin, Turn, Usage
from agentmesh.providers.retry import NO_RETRY

ROUTING = {
    "anthropic/claude-opus-5": {
        "provider": "anthropic",
        "model": "claude-opus-5",
        "cost_per_1k_input_tokens": 0.005,
        "cost_per_1k_output_tokens": 0.025,
    }
}


class CountingProvider(SingleShotMixin):
    name = "counting"

    def __init__(self, credentials=None):
        self.calls = 0

    def converse(self, *, system, messages, tools=None, model="m", max_tokens=4096, effort=None, timeout=None):
        self.calls += 1
        return Turn(
            text='{"summary": "s", "findings": [], "confidence": 0.9}',
            provider=self.name,
            model=model,
            usage=Usage(input_tokens=1000, output_tokens=1000),
        )


def broker(recorder, provider=None):
    provider = provider or CountingProvider()
    b = ModelBroker(
        models=ROUTING,
        provider_override=provider.name,
        recorder=recorder,
        retry_policy=NO_RETRY,
    )
    b._clients[provider.name] = provider
    return b


def spec_for(capability):
    return synthesize_agent(capability).model_copy(
        update={"models": {"default": "anthropic/claude-opus-5", "fallback": []}}
    )


# -- per-call records -----------------------------------------------------


def test_every_call_is_recorded_with_full_detail() -> None:
    recorder = RunRecorder()
    build_agent(spec_for("sql_review"), broker(recorder)).run("go", {})

    call = recorder.calls[0]
    assert call.agent_id == "sql_review_agent"
    assert call.provider == "anthropic"
    assert call.model_key == "anthropic/claude-opus-5"
    assert call.model == "claude-opus-5"
    assert call.outcome == "ok"
    assert call.input_tokens == 1000
    assert call.output_tokens == 1000
    assert call.cost == pytest.approx(0.030)
    assert call.duration_ms >= 0
    assert call.call_id


def test_costs_aggregate_across_agents() -> None:
    recorder = RunRecorder()
    agents = [spec_for("sql_review"), spec_for("pii_detection")]

    result = run_agents("go", agents, broker(recorder), max_workers=1)

    assert result.llm_calls == 2
    assert result.cost == pytest.approx(0.060)
    assert result.usage == Usage(input_tokens=2000, output_tokens=2000)
    assert recorder.cost_for("sql_review_agent") == pytest.approx(0.030)


def test_run_has_a_stable_id_and_serializes() -> None:
    recorder = RunRecorder()
    build_agent(spec_for("sql_review"), broker(recorder)).run("go", {})

    payload = recorder.as_dict()
    assert payload["run_id"] == recorder.run_id
    assert payload["llm_calls"] == 1
    assert payload["retries"] == 0
    assert payload["estimated_cost_usd"] == pytest.approx(0.030)
    assert len(payload["calls"]) == 1


# -- logging --------------------------------------------------------------


def test_each_call_emits_a_log_record(caplog) -> None:
    recorder = RunRecorder()
    with caplog.at_level(logging.INFO, logger="agentmesh.llm"):
        build_agent(spec_for("sql_review"), broker(recorder)).run("go", {})

    record = next(r for r in caplog.records if r.name == "agentmesh.llm")
    assert record.agentmesh["agent_id"] == "sql_review_agent"
    assert record.agentmesh["run_id"] == recorder.run_id
    assert record.agentmesh["attempts"] == 1


def test_json_logging_emits_parsable_lines(capsys) -> None:
    configure_logging(level="INFO", json_logs=True)
    recorder = RunRecorder()
    try:
        build_agent(spec_for("sql_review"), broker(recorder)).run("go", {})
        line = capsys.readouterr().err.strip().splitlines()[-1]
    finally:
        logging.getLogger("agentmesh").handlers.clear()

    payload = json.loads(line)
    assert payload["agent_id"] == "sql_review_agent"
    assert payload["cost"] == pytest.approx(0.030)
    assert payload["model_key"] == "anthropic/claude-opus-5"


# -- budget is opt-in -----------------------------------------------------


def test_no_budget_means_no_ceiling() -> None:
    recorder = RunRecorder()
    agents = [spec_for(f"cap_{n}") for n in range(5)]

    result = run_agents("go", agents, broker(recorder), max_workers=1)

    assert result.llm_calls == 5
    assert recorder.max_cost is None


def test_a_caller_set_budget_stops_the_run() -> None:
    recorder = RunRecorder(max_cost=0.05)
    provider = CountingProvider()
    agents = [spec_for(f"cap_{n}") for n in range(5)]

    with pytest.raises(BudgetExceeded, match=r"0\.0500"):
        run_agents("go", agents, broker(recorder, provider), max_workers=1)

    # Two calls at $0.03 cross the ceiling; the rest never run.
    assert provider.calls == 2


def test_budget_error_names_the_agent_that_was_blocked() -> None:
    recorder = RunRecorder(max_cost=0.001)
    recorder.check_budget("first_agent")  # nothing spent yet, so this passes

    build_agent(spec_for("sql_review"), broker(recorder)).run("go", {})

    with pytest.raises(BudgetExceeded, match="second_agent"):
        recorder.check_budget("second_agent")
