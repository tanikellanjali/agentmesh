from __future__ import annotations

from typing import Any

from agentmesh.agents.base_agent import BaseAgent
from agentmesh.schemas.agent_spec import AgentSpec


class MockAgent(BaseAgent):
    def run(self, message: str, context: dict[str, Any]) -> dict[str, Any]:
        output: dict[str, Any] = {
            "agent_id": self.spec.id,
            "summary": f"Mock output for {self.spec.name}.",
            "confidence": self._confidence(),
        }
        context[self.spec.id] = output
        return output


def build_agent(spec: AgentSpec) -> BaseAgent:
    return MockAgent(spec)
