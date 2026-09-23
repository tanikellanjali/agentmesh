"""The LangGraph engine must be interchangeable with the native one.

Skipped when the langgraph extra is not installed.
"""

import pytest

pytest.importorskip("langgraph")

from agentmesh.core.agent_synthesizer import synthesize_agent  # noqa: E402
from agentmesh.core.model_router import ModelBroker  # noqa: E402
from agentmesh.core.orchestrator import run_agents  # noqa: E402
from agentmesh.core.telemetry import RunRecorder  # noqa: E402
from agentmesh.execution.langgraph_engine import LangGraphEngine  # noqa: E402
from agentmesh.execution.native import NativeEngine  # noqa: E402
from agentmesh.providers.mock import MockProvider  # noqa: E402
from agentmesh.providers.retry import NO_RETRY  # noqa: E402


def agent(capability, depends_on=None):
    spec = synthesize_agent(capability)
    return spec.model_copy(update={"depends_on": depends_on}) if depends_on else spec


def broker():
    b = ModelBroker(provider_override="mock", retry_policy=NO_RETRY, recorder=RunRecorder())
    b._clients["mock"] = MockProvider()
    return b


@pytest.fixture()
def mesh():
    return [
        agent("sql_analysis"),
        agent("pii_detection"),
        agent("report_generation", ["sql_analysis_agent", "pii_detection_agent"]),
    ]


def test_both_engines_produce_the_same_result(mesh) -> None:
    native = run_agents("go", mesh, broker(), engine=NativeEngine())
    graph = run_agents("go", mesh, broker(), engine=LangGraphEngine())

    assert native.selected_agents == graph.selected_agents
    assert native.levels == graph.levels
    assert native.final_agent_id == graph.final_agent_id
    assert native.llm_calls == graph.llm_calls


def test_downstream_agents_receive_upstream_output(mesh) -> None:
    result = run_agents("go", mesh, broker(), engine=LangGraphEngine())

    report = next(e for e in result.executions if e.agent_id == "report_generation_agent")
    assert report.level == 1
    assert report.output["agent_id"] == "report_generation_agent"


def test_the_composer_owns_the_final_response() -> None:
    mesh = [agent("sql_analysis"), agent("response_composition")]

    result = run_agents("go", mesh, broker(), engine=LangGraphEngine())

    assert result.final_agent_id == "response_composition_agent"


def test_edges_come_from_declared_dependencies(mesh) -> None:
    compiled = LangGraphEngine().compile(mesh, broker())
    graph = compiled.get_graph()
    edges = {(e.source, e.target) for e in graph.edges}

    assert ("sql_analysis_agent", "report_generation_agent") in edges
    assert ("pii_detection_agent", "report_generation_agent") in edges


def test_a_run_can_be_interrupted_and_resumed() -> None:
    """The capability that justifies the dependency: durable supervision."""
    from langgraph.checkpoint.memory import MemorySaver

    mesh = [agent("spend_analysis"), agent("response_composition", ["spend_analysis_agent"])]
    engine = LangGraphEngine(
        checkpointer=MemorySaver(), interrupt_before=["response_composition_agent"]
    )
    compiled = engine.compile(mesh, broker())
    config = {"configurable": {"thread_id": "t1"}}

    paused = compiled.invoke({"message": "go", "outputs": {}}, config=config)
    assert sorted(paused["outputs"]) == ["spend_analysis_agent"]
    assert compiled.get_state(config).next == ("response_composition_agent",)

    resumed = compiled.invoke(None, config=config)
    assert sorted(resumed["outputs"]) == [
        "response_composition_agent",
        "spend_analysis_agent",
    ]


def test_engine_reports_itself() -> None:
    assert LangGraphEngine().name == "langgraph"
    assert NativeEngine().name == "native"
