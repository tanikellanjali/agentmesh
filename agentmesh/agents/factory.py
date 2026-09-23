from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentmesh.agents.base_agent import BaseAgent
from agentmesh.agents.registry import get_executor, register_executor
from agentmesh.schemas.agent_spec import AgentSpec

if TYPE_CHECKING:
    from agentmesh.core.model_router import ModelBroker
    from agentmesh.core.run_context import RunContext


@register_executor("mock")
class MockAgent(BaseAgent):
    """Fixed-output executor that never calls a model. For tests and demos."""

    def run(self, message: str, context: dict[str, Any]) -> dict[str, Any]:
        output: dict[str, Any] = {
            "agent_id": self.spec.id,
            "summary": f"Mock output for {self.spec.name}.",
            "confidence": self._confidence_threshold(),
        }
        context[self.spec.id] = output
        return output


def build_agent(
    spec: AgentSpec,
    broker: ModelBroker | None = None,
    run_context: RunContext | None = None,
) -> BaseAgent:
    return get_executor(spec.executor)(spec, broker, run_context)
