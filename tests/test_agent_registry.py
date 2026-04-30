import pytest

from agentmesh.core.agent_registry import AgentRegistry
from agentmesh.core.project_loader import load_project


def test_registry_lists_enabled_agents_sorted() -> None:
    loaded = load_project("data_ops")
    registry = AgentRegistry(loaded.agents)

    agents = registry.list()

    assert [agent.id for agent in agents] == sorted(agent.id for agent in agents)
    assert "sql_analyzer_agent" in [agent.id for agent in agents]


def test_registry_finds_agents_by_capability() -> None:
    loaded = load_project("data_ops")
    registry = AgentRegistry(loaded.agents)

    matches = registry.by_capability("pii_detection")

    assert [agent.id for agent in matches] == ["compliance_agent"]


def test_registry_rejects_duplicate_agent_ids() -> None:
    loaded = load_project("data_ops")
    first_agent = loaded.agents[0]

    with pytest.raises(ValueError, match="Duplicate agent id"):
        AgentRegistry([first_agent, first_agent])


def test_registry_unknown_agent_has_clear_error() -> None:
    registry = AgentRegistry()

    with pytest.raises(KeyError, match="Unknown agent id"):
        registry.get("missing_agent")
