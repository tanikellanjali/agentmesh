import pytest

from agentmesh.core.agent_synthesizer import synthesize_agent
from agentmesh.core.graph import GraphError, build_levels, find_composer
from agentmesh.core.model_router import ModelBroker
from agentmesh.core.orchestrator import run_agents
from agentmesh.core.telemetry import RunRecorder
from agentmesh.providers.base import Completion, Usage
from agentmesh.providers.retry import NO_RETRY


class SlowProvider:
    name = "slow"

    def __init__(self, credentials=None):
        self.order: list[str] = []

    def complete(self, *, system, prompt, model, max_tokens=4096, effort=None):
        return Completion(
            text='{"summary": "s", "findings": [], "confidence": 0.9, '
            '"recommendations": [], "risks": []}',
            provider=self.name,
            model=model,
            usage=Usage(input_tokens=10, output_tokens=5),
        )


def agent(capability, depends_on=None):
    spec = synthesize_agent(capability)
    if depends_on:
        spec = spec.model_copy(update={"depends_on": depends_on})
    return spec


def broker(recorder=None):
    provider = SlowProvider()
    b = ModelBroker(
        provider_override=provider.name, retry_policy=NO_RETRY, recorder=recorder
    )
    b._clients[provider.name] = provider
    return b


# -- graph shape ----------------------------------------------------------


def test_independent_agents_share_one_level() -> None:
    levels = build_levels([agent("sql_analysis"), agent("pii_detection")])

    assert [[a.id for a in level] for level in levels] == [
        ["pii_detection_agent", "sql_analysis_agent"]
    ]


def test_explicit_dependencies_create_levels() -> None:
    first = agent("data_ingestion")
    second = agent("data_quality_check", depends_on=["data_ingestion_agent"])

    levels = build_levels([first, second])

    assert [[a.id for a in level] for level in levels] == [
        ["data_ingestion_agent"],
        ["data_quality_check_agent"],
    ]


def test_a_composer_implicitly_waits_for_every_other_agent() -> None:
    composer = agent("response_composition")
    levels = build_levels([agent("sql_analysis"), composer, agent("pii_detection")])

    assert [[a.id for a in level] for level in levels] == [
        ["pii_detection_agent", "sql_analysis_agent"],
        ["response_composition_agent"],
    ]
    assert find_composer([composer]) is composer


def test_explicit_dependencies_override_the_composer_default() -> None:
    composer = agent("response_composition", depends_on=["sql_analysis_agent"])

    levels = build_levels([agent("sql_analysis"), composer, agent("pii_detection")])

    assert [[a.id for a in level] for level in levels] == [
        ["pii_detection_agent", "sql_analysis_agent"],
        ["response_composition_agent"],
    ]


def test_unknown_dependency_is_rejected() -> None:
    with pytest.raises(GraphError, match="unknown agent"):
        build_levels([agent("sql_analysis", depends_on=["ghost_agent"])])


def test_self_dependency_is_rejected() -> None:
    spec = synthesize_agent("sql_analysis").model_copy(
        update={"depends_on": ["sql_analysis_agent"]}
    )

    with pytest.raises(GraphError, match="depends on itself"):
        build_levels([spec])


def test_cycles_are_rejected() -> None:
    a = agent("sql_analysis", depends_on=["pii_detection_agent"])
    b = agent("pii_detection", depends_on=["sql_analysis_agent"])

    with pytest.raises(GraphError, match="Circular dependency"):
        build_levels([a, b])


def test_levels_are_deterministic() -> None:
    agents = [agent("z_thing"), agent("a_thing"), agent("m_thing")]

    assert [a.id for a in build_levels(agents)[0]] == [
        "a_thing_agent",
        "m_thing_agent",
        "z_thing_agent",
    ]


# -- execution ------------------------------------------------------------


def test_downstream_agents_see_upstream_output() -> None:
    upstream = agent("sql_analysis")
    downstream = agent("report_generation", depends_on=["sql_analysis_agent"])

    result = run_agents("go", [upstream, downstream], broker())

    assert result.levels == [["sql_analysis_agent"], ["report_generation_agent"]]
    assert result.executions[1].level == 1


def test_final_response_comes_from_the_composer_not_the_last_agent() -> None:
    composer = agent("response_composition")
    late = agent("zz_troubleshooting", depends_on=["response_composition_agent"])

    result = run_agents("go", [composer, late], broker())

    assert result.selected_agents[-1] == "zz_troubleshooting_agent"
    assert result.final_agent_id == "response_composition_agent"
    assert result.final_response["agent_id"] == "response_composition_agent"


def test_parallel_execution_produces_the_same_result_as_serial() -> None:
    agents = [agent("sql_analysis"), agent("pii_detection"), agent("data_quality_check")]

    serial = run_agents("go", agents, broker(), max_workers=1)
    parallel = run_agents("go", agents, broker(), max_workers=4)

    assert serial.selected_agents == parallel.selected_agents
    assert serial.levels == parallel.levels


def test_run_totals_aggregate_every_agent() -> None:
    recorder = RunRecorder()
    agents = [agent("sql_analysis"), agent("pii_detection")]

    result = run_agents("go", agents, broker(recorder), max_workers=2)

    assert result.llm_calls == 2
    assert result.usage.input_tokens == 20
    assert result.retries == 0
