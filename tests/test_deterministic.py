import pytest

from agentmesh.agents.deterministic import resolve_argument
from agentmesh.agents.factory import build_agent
from agentmesh.core.model_router import ModelBroker
from agentmesh.core.run_context import RunContext
from agentmesh.core.telemetry import RunRecorder
from agentmesh.schemas.agent_spec import AgentSpec


def spec(**overrides):
    base = {
        "id": "profiler_agent", "name": "Profiler", "category": "data",
        "description": "Profiles a dataset.", "capabilities": ["data_profiling"],
        "input_contract": {"required": []},
        "output_contract": {"format": "json", "required_fields": ["summary", "results"]},
        "models": {}, "executor": "deterministic", "tools": ["profile_csv"],
        "deterministic": {"arguments": {"profile_csv": {"csv_text": "$message"}}},
    }
    base.update(overrides)
    return AgentSpec(**base)


@pytest.fixture()
def wiring():
    recorder = RunRecorder()
    return ModelBroker(recorder=recorder), RunContext(recorder=recorder), recorder


CSV = "id,score\n1,10\n2,\n3,30\n"


# -- argument templating --------------------------------------------------


def test_message_token_is_substituted() -> None:
    assert resolve_argument("$message", "hello", {}) == "hello"


def test_context_paths_are_resolved() -> None:
    context = {"upstream_agent": {"results": {"total": 42}}}

    assert resolve_argument("$context.upstream_agent.results.total", "m", context) == 42


def test_a_missing_context_path_is_none_not_an_error() -> None:
    assert resolve_argument("$context.nobody.value", "m", {}) is None


def test_literals_and_nested_structures_pass_through() -> None:
    out = resolve_argument({"a": 1, "b": ["$message", "x"]}, "hi", {})

    assert out == {"a": 1, "b": ["hi", "x"]}


# -- execution ------------------------------------------------------------


def test_a_deterministic_agent_makes_no_model_call(wiring) -> None:
    broker, context, recorder = wiring

    output = build_agent(spec(), broker, context).run(CSV, {})

    assert recorder.calls == []          # the whole point
    assert recorder.cost == 0.0
    assert output["model"] is None
    assert output["executor"] == "deterministic"


def test_tool_results_are_returned(wiring) -> None:
    broker, context, _ = wiring

    output = build_agent(spec(), broker, context).run(CSV, {})

    profile = output["results"]["profile_csv"]
    assert profile["row_count"] == 3
    assert output["confidence"] == 1.0


def test_tool_findings_are_surfaced_for_downstream_agents(wiring) -> None:
    broker, context, _ = wiring

    output = build_agent(spec(), broker, context).run(CSV, {})

    assert any("missing value" in finding for finding in output["findings"])


def test_tool_runs_are_recorded(wiring) -> None:
    broker, context, recorder = wiring

    build_agent(spec(), broker, context).run(CSV, {})

    assert [(t.tool, t.ok) for t in recorder.tool_runs] == [("profile_csv", True)]


def test_a_failing_tool_lowers_confidence_rather_than_raising(wiring) -> None:
    broker, context, _ = wiring

    output = build_agent(spec(), broker, context).run("   ", {})

    assert output["confidence"] == 0.0
    assert output["tools_failed"] == ["profile_csv"]


def test_unknown_tools_are_reported(wiring) -> None:
    broker, context, _ = wiring

    output = build_agent(spec(tools=["profile_csv", "ghost_tool"]), broker, context).run(CSV, {})

    assert output["tools_unavailable"] == ["ghost_tool"]


def test_contract_violations_are_still_enforced(wiring) -> None:
    broker, context, _ = wiring
    strict = spec(output_contract={"format": "json", "required_fields": ["summary", "risk_level"]})

    output = build_agent(strict, broker, context).run(CSV, {})

    assert output["contract_violations"] == ["risk_level"]


def test_a_custom_summary_is_used(wiring) -> None:
    broker, context, _ = wiring
    described = spec(deterministic={"arguments": {"profile_csv": {"csv_text": "$message"}},
                                    "summary": "Dataset profiled."})

    assert build_agent(described, broker, context).run(CSV, {})["summary"] == "Dataset profiled."


def test_upstream_output_can_feed_a_deterministic_agent(wiring) -> None:
    broker, context, _ = wiring
    chained = spec(deterministic={"arguments": {"profile_csv": {"csv_text": "$context.loader.csv"}}})

    output = build_agent(chained, broker, context).run("ignored", {"loader": {"csv": CSV}})

    assert output["results"]["profile_csv"]["row_count"] == 3
