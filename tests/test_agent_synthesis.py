import shutil
from pathlib import Path

import pytest

from agentmesh.agents.factory import build_agent
from agentmesh.agents.generic import GenericAgent
from agentmesh.agents.registry import ExecutorNotFound, get_executor, list_executors
from agentmesh.core.agent_synthesizer import synthesize_agent
from agentmesh.core.model_router import ModelBroker
from agentmesh.core.runtime import run_project


@pytest.fixture()
def mock_broker() -> ModelBroker:
    return ModelBroker(provider_override="mock")


@pytest.fixture()
def temp_spec_packs(tmp_path: Path) -> Path:
    spec_packs = tmp_path / "spec_packs"
    shutil.copytree(Path("spec_packs"), spec_packs)
    return spec_packs


def test_builtin_executors_are_registered() -> None:
    assert "generic" in list_executors()
    assert "mock" in list_executors()


def test_unknown_executor_is_rejected() -> None:
    spec = synthesize_agent("anomaly_triage").model_copy(update={"executor": "nope"})

    with pytest.raises(ExecutorNotFound):
        build_agent(spec)


def test_specs_default_to_the_generic_executor() -> None:
    spec = synthesize_agent("anomaly_triage")

    assert spec.executor == "generic"
    assert isinstance(build_agent(spec), GenericAgent)
    assert get_executor(spec.executor) is GenericAgent


def test_generic_agent_output_is_shaped_by_its_output_contract(
    mock_broker: ModelBroker,
) -> None:
    spec = synthesize_agent("anomaly_triage")

    output = build_agent(spec, mock_broker).run("find the anomalies", {})

    for required_field in spec.output_contract["required_fields"]:
        assert required_field in output
    assert "contract_violations" not in output


def test_run_spins_up_an_agent_for_an_uncovered_capability(temp_spec_packs: Path) -> None:
    result = run_project(
        "data_ops",
        "Research the options and summarize them.",
        spec_packs_dir=temp_spec_packs,
        provider="mock",
    )

    assert "domain_research" in result.match.missing_capabilities
    assert "domain_research_agent" in result.run.selected_agents
    assert [agent.id for agent in result.match.synthesized_agents] == [
        "domain_research_agent",
        "option_generation_agent",
    ]


def test_synthesized_agents_are_not_stored_by_default(temp_spec_packs: Path) -> None:
    result = run_project(
        "data_ops",
        "Research the options and summarize them.",
        spec_packs_dir=temp_spec_packs,
        provider="mock",
    )

    assert result.stored_agent_paths == []
    assert not (temp_spec_packs / "data_ops/agents/domain_research_agent.yaml").exists()


def test_synthesized_agents_are_stored_when_requested(temp_spec_packs: Path) -> None:
    result = run_project(
        "data_ops",
        "Research the options and summarize them.",
        spec_packs_dir=temp_spec_packs,
        provider="mock",
        store_agents=True,
    )

    stored = temp_spec_packs / "data_ops/agents/domain_research_agent.yaml"
    assert stored in result.stored_agent_paths
    assert stored.exists()


def test_stored_agents_are_reloaded_and_no_longer_synthesized(temp_spec_packs: Path) -> None:
    message = "Research the options and summarize them."
    run_project("data_ops", message, spec_packs_dir=temp_spec_packs,
        provider="mock", store_agents=True)

    second = run_project(
        "data_ops", message, spec_packs_dir=temp_spec_packs, provider="mock"
    )

    assert second.match.missing_capabilities == []
    assert second.match.synthesized_agents == []
    assert "domain_research_agent" in second.run.selected_agents


def test_storing_an_existing_agent_requires_overwrite(temp_spec_packs: Path) -> None:
    message = "Research the options and summarize them."
    run_project("data_ops", message, spec_packs_dir=temp_spec_packs,
        provider="mock", store_agents=True)

    # The stored agent now covers the capability, so nothing is synthesized and
    # nothing is rewritten.
    result = run_project(
        "data_ops", message, spec_packs_dir=temp_spec_packs,
        provider="mock", store_agents=True
    )

    assert result.stored_agent_paths == []
