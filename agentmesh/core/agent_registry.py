from __future__ import annotations

from agentmesh.schemas.agent_spec import AgentSpec


class AgentRegistry:
    def __init__(self, agents: list[AgentSpec] | None = None) -> None:
        self._agents: dict[str, AgentSpec] = {}
        for agent in agents or []:
            self.register(agent)

    def register(self, agent: AgentSpec) -> None:
        if agent.id in self._agents:
            raise ValueError(f"Duplicate agent id registered: {agent.id}")
        self._agents[agent.id] = agent

    def get(self, agent_id: str) -> AgentSpec:
        try:
            return self._agents[agent_id]
        except KeyError as exc:
            raise KeyError(f"Unknown agent id: {agent_id}") from exc

    def list(self, include_disabled: bool = False) -> list[AgentSpec]:
        agents = self._agents.values()
        if include_disabled:
            return sorted(agents, key=lambda agent: agent.id)
        return sorted((agent for agent in agents if agent.enabled), key=lambda agent: agent.id)

    def by_capability(self, capability: str) -> list[AgentSpec]:
        return [
            agent
            for agent in self.list()
            if capability in agent.capabilities
        ]
