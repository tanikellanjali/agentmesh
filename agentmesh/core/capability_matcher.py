from __future__ import annotations

from dataclasses import dataclass

from agentmesh.core.agent_registry import AgentRegistry
from agentmesh.schemas.agent_spec import AgentSpec


@dataclass(frozen=True)
class CapabilityMatch:
    selected_agents: list[AgentSpec]
    missing_capabilities: list[str]


def match_capabilities(
    required_capabilities: list[str],
    registry: AgentRegistry,
) -> CapabilityMatch:
    selected: dict[str, AgentSpec] = {}
    missing: list[str] = []

    for capability in required_capabilities:
        matches = registry.by_capability(capability)
        if not matches:
            missing.append(capability)
            continue

        for agent in matches:
            selected[agent.id] = agent
            for fallback_agent_id in agent.fallback_agents:
                try:
                    selected[fallback_agent_id] = registry.get(fallback_agent_id)
                except KeyError:
                    missing.append(f"fallback:{fallback_agent_id}")

    if missing and "troubleshooting_agent" not in selected:
        try:
            selected["troubleshooting_agent"] = registry.get("troubleshooting_agent")
        except KeyError:
            pass

    return CapabilityMatch(
        selected_agents=sorted(selected.values(), key=lambda agent: agent.id),
        missing_capabilities=missing,
    )
