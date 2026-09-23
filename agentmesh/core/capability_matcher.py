from __future__ import annotations

from dataclasses import dataclass, field

from agentmesh.core.agent_registry import AgentRegistry
from agentmesh.core.agent_synthesizer import synthesize_agent
from agentmesh.schemas.agent_spec import AgentSpec


@dataclass(frozen=True)
class CapabilityMatch:
    selected_agents: list[AgentSpec]
    missing_capabilities: list[str]
    synthesized_agents: list[AgentSpec] = field(default_factory=list)


def match_capabilities(
    required_capabilities: list[str],
    registry: AgentRegistry,
    synthesize: bool = True,
) -> CapabilityMatch:
    """Resolve capabilities to agents, spinning up ephemeral agents for gaps.

    ``missing_capabilities`` always reports the gaps. When ``synthesize`` is
    true each gap is filled with an ephemeral agent instead of dead-ending into
    the troubleshooting agent.
    """
    selected: dict[str, AgentSpec] = {}
    missing: list[str] = []
    synthesized: dict[str, AgentSpec] = {}

    for capability in required_capabilities:
        matches = registry.by_capability(capability)
        if not matches:
            missing.append(capability)
            if synthesize:
                spec = synthesize_agent(capability)
                synthesized.setdefault(spec.id, spec)
                selected.setdefault(spec.id, spec)
            continue

        for agent in matches:
            selected[agent.id] = agent
            for fallback_agent_id in agent.fallback_agents:
                try:
                    selected[fallback_agent_id] = registry.get(fallback_agent_id)
                except KeyError:
                    missing.append(f"fallback:{fallback_agent_id}")

    if missing and not synthesize and "troubleshooting_agent" not in selected:
        try:
            selected["troubleshooting_agent"] = registry.get("troubleshooting_agent")
        except KeyError:
            pass

    return CapabilityMatch(
        selected_agents=sorted(selected.values(), key=lambda agent: agent.id),
        missing_capabilities=missing,
        synthesized_agents=sorted(synthesized.values(), key=lambda agent: agent.id),
    )
