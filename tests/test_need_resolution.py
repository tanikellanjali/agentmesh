from agentmesh.core.agent_registry import AgentRegistry
from agentmesh.core.capability_matcher import match_capabilities
from agentmesh.core.need_resolver import resolve_needs
from agentmesh.core.project_loader import load_project


def test_resolve_needs_detects_sql_quality_and_report_capabilities() -> None:
    resolution = resolve_needs(
        "Review this SQL query for performance, missing values, and summarize the risks."
    )

    assert resolution.required_capabilities == [
        "data_quality_check",
        "sql_analysis",
        "query_review",
        "report_generation",
    ]


def test_match_capabilities_selects_enabled_agents() -> None:
    loaded = load_project("data_ops")
    registry = AgentRegistry(loaded.agents)

    match = match_capabilities(["sql_analysis", "pii_detection"], registry)

    assert [agent.id for agent in match.selected_agents] == [
        "compliance_agent",
        "sql_analyzer_agent",
    ]
    assert match.missing_capabilities == []


def test_match_capabilities_synthesizes_agent_for_missing_capability() -> None:
    loaded = load_project("data_ops")
    registry = AgentRegistry(loaded.agents)

    match = match_capabilities(["unknown_capability"], registry)

    assert [agent.id for agent in match.selected_agents] == ["unknown_capability_agent"]
    assert [agent.id for agent in match.synthesized_agents] == ["unknown_capability_agent"]
    assert match.missing_capabilities == ["unknown_capability"]


def test_match_capabilities_falls_back_to_troubleshooting_without_synthesis() -> None:
    loaded = load_project("data_ops")
    registry = AgentRegistry(loaded.agents)

    match = match_capabilities(["unknown_capability"], registry, synthesize=False)

    assert [agent.id for agent in match.selected_agents] == ["troubleshooting_agent"]
    assert match.synthesized_agents == []
    assert match.missing_capabilities == ["unknown_capability"]
